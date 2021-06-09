#  Created by Luis A. Sanchez-Perez (l.alejandro.2011@gmail.com).
#  Copyright © Do not distribute or use without authorization from author

import utils
import argparse
import tensorflow as tf
import pandas as pd

BATCH_SIZE = 512
BUFFER_SIZE = BATCH_SIZE * 32


class HitRateAt(tf.keras.metrics.Metric):
  def __init__(self, k: int, **kwargs):
    super(HitRateAt, self).__init__(**kwargs)
    self.k = k
    self.total = self.add_weight(name='total', initializer='zeros', dtype=tf.int32)
    self.hits = self.add_weight(name='hits', initializer='zeros', dtype=tf.int32)

  def update_state(self, prediction, target, **kwargs):
    # Ranks outputs for each observation
    rank = tf.argsort(prediction, axis=1, direction='DESCENDING')
    # A hit is considered when the target appears in the top k elements
    self.hits.assign_add(tf.reduce_sum(tf.cast(tf.reduce_any(tf.equal(
      rank[:, :self.k],
      tf.cast(tf.expand_dims(target, axis=-1), dtype=rank.dtype)
    ), axis=1), dtype=tf.int32)))
    self.total.assign_add(tf.cast(tf.shape(prediction)[0], dtype=tf.int32))

  def result(self):
    return tf.cast(self.hits, dtype=tf.float32) / tf.cast(self.total, dtype=tf.float32)


class RankingAUC(tf.keras.metrics.Metric):
  def __init__(self, **kwargs):
    super(RankingAUC, self).__init__(**kwargs)
    self.total = self.add_weight(name='total', initializer='zeros', dtype=tf.int32)
    self.bad = self.add_weight(name='bad', initializer='zeros', dtype=tf.float32)

  def update_state(self, prediction, target, **kwargs):
    # Ranks outputs for each observation
    rank = tf.argsort(prediction, axis=1, direction='DESCENDING')
    # Indices corresponding to the target rank
    indices = tf.argmax(tf.equal(
      rank,
      tf.cast(tf.expand_dims(target, axis=-1), dtype=rank.dtype)
    ), axis=1)
    # Ratio of how many other ranked items are above the target item versus the total number of items
    self.bad.assign_add(
      tf.reduce_sum(tf.cast(indices, dtype=tf.float32) / tf.cast(tf.shape(prediction)[1], dtype=tf.float32))
    )
    self.total.assign_add(tf.cast(tf.shape(prediction)[0], tf.int32))

  def result(self):
    return 1. - (self.bad / tf.cast(self.total, dtype=self.bad.dtype))


class HorizontalConvolution(tf.keras.layers.Layer):
  def __init__(self, filters_per_expand: int, **kwargs):
    super(HorizontalConvolution, self).__init__(**kwargs)

    self.conv = []
    self.pooling = []
    self.filter_per_expand = filters_per_expand

  def build(self, input_shape):
    _, sequence_length, latent_dimension = input_shape
    # We will create two filters per value in range(1, sequence_length + 1) plus a max pool per filter
    for expand in range(1, sequence_length + 1):
      self.conv.append(
        tf.keras.layers.Conv1D(self.filter_per_expand, expand,
                               activation='relu', input_shape=(sequence_length, latent_dimension))
      )
      self.pooling.append(
        tf.keras.layers.MaxPool1D(sequence_length - expand + 1)
      )

  def call(self, inputs, *args, **kwargs):
    return tf.keras.layers.Concatenate()(
      [tf.squeeze(pooling(conv(inputs)), axis=1) for conv, pooling, in zip(self.conv, self.pooling)]
    )


class VerticalConvolution(tf.keras.layers.Layer):
  def __init__(self, filters: int, **kwargs):
    super(VerticalConvolution, self).__init__(**kwargs)
    self.filters = filters
    self.conv = None
    self.flatten = None

  def build(self, input_shape):
    _, sequence_length, latent_dimension = input_shape
    self.conv = tf.keras.layers.Conv2D(self.filters, kernel_size=(sequence_length, 1), activation='relu')
    self.flatten = tf.keras.layers.Flatten()

  def call(self, inputs, *args, **kwargs):
    return self.flatten(self.conv(tf.expand_dims(inputs, axis=-1)))


class DenseByItems(tf.keras.layers.Layer):
  def __init__(self, num_items: int, latent_size: int, **kwargs):
    super(DenseByItems, self).__init__(**kwargs)
    self.num_items = num_items
    self.latent_size = latent_size
    self.V = None
    self.B = None

  def build(self, input_shape):
    # Simulated weight matrix and bias for the final dense layer (allows to compute only one during training)
    self.V = tf.keras.layers.Embedding(input_dim=self.num_items, output_dim=self.latent_size * 2)
    self.B = tf.keras.layers.Embedding(input_dim=self.num_items, output_dim=1)

  def call(self, inputs, *args, **kwargs):
    sequence_embedding, items = inputs
    v = self.V(items)
    b = self.B(items)
    return tf.matmul(v, tf.expand_dims(sequence_embedding, axis=-1)) + b


class HingeLossRecommender(tf.keras.losses.Loss):
  def __init__(self, margin=1, **kwargs):
    super(HingeLossRecommender, self).__init__(**kwargs)
    self.margin = margin

  def call(self, positive, negative):
    distance = positive - negative
    loss = tf.maximum(self.margin - distance, 0)
    return loss


class Caser(tf.Module):
  def __init__(self,
               train_df: pd.DataFrame,
               test_df: pd.DataFrame,
               sequence_length: int,
               user_to_index: dict,
               movie_to_index: dict,
               latent_size: int,
               learning_rate=1e-3,
               k=50):
    super(Caser, self).__init__()

    # For later use
    self.sequence_length = sequence_length
    self.latent_size = latent_size
    self.learning_rate = learning_rate

    # Creates tf static hash tables
    self.num_users = len(user_to_index)
    self.user_to_index = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
        keys=list(user_to_index.keys()),
        values=list(user_to_index.values())
      ),
      default_value=-1
    )
    self.num_movies = len(movie_to_index)
    self.movie_to_index = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
        keys=list(movie_to_index.keys()),
        values=list(movie_to_index.values())
      ),
      default_value=-1
    )

    self.generator = tf.random.get_global_generator()
    self.model = self.create_model()
    self.train_ds = self.create_pairwise_training_dataset(train_df)
    self.test_ds = self.create_test_dataset(test_df)
    self.optimizer = tf.optimizers.Adam()
    self.loss_fc = HingeLossRecommender(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    self.hit = HitRateAt(k=k)
    self.auc = RankingAUC()

  def generate_pairwise_training_observation(self, tensor):
    # User
    user = tensor[0]
    # Sequence of items
    sequence = tensor[1:-1]
    # Target item (positive)
    positive = tf.expand_dims(tensor[-1], axis=-1)
    # Samples a negative item (not in sequence or target)
    items = tf.expand_dims(tf.range(self.num_movies, dtype=tf.int64), axis=-1)
    rated = tf.expand_dims(tensor[1:], axis=0)
    mask = tf.reduce_all(tf.not_equal(items, rated), axis=-1)
    candidates = tf.squeeze(tf.boolean_mask(items, mask, axis=0))
    index = tf.cast(
      self.generator.uniform(shape=(), maxval=tf.cast(tf.shape(candidates)[0], dtype=tf.float32)),
      dtype=tf.int64
    )
    negative = tf.expand_dims(tf.gather(candidates, index), axis=-1)
    negative.set_shape(shape=(1,))
    return user, sequence, positive, negative

  def create_pairwise_training_dataset(self, train_df: pd.DataFrame) -> tf.data.Dataset:
    pandas_ds = tf.data.Dataset.from_tensor_slices(train_df)
    pairwise_ds = pandas_ds.map(self.generate_pairwise_training_observation, num_parallel_calls=tf.data.AUTOTUNE)
    pairwise_ds = pairwise_ds.repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return pairwise_ds.prefetch(tf.data.AUTOTUNE)

  def create_test_dataset(self, test_df: pd.DataFrame) -> tf.data.Dataset:
    pandas_ds = tf.data.Dataset.from_tensor_slices(test_df)
    test_ds = pandas_ds.map(lambda tensor: (tensor[0], tensor[1:-1], tensor[-1]),
                            num_parallel_calls=tf.data.AUTOTUNE)
    return test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

  def create_model(self) -> tf.keras.models.Model:
    # Inputs
    input_user = tf.keras.layers.Input(shape=())
    input_sequence = tf.keras.layers.Input(shape=(self.sequence_length,))
    input_items = tf.keras.layers.Input(shape=(None,))

    # Embeddings for movies and to generate the matrix E
    Q = tf.keras.layers.Embedding(input_dim=self.num_movies,
                                  output_dim=self.latent_size,
                                  input_length=self.sequence_length)
    # Embeddings of users
    P = tf.keras.layers.Embedding(input_dim=self.num_users, output_dim=self.latent_size)

    # Generates matrix E (stacking vertically all embeddings)
    E = Q(input_sequence)

    # Defines horizontal and vertical convolutional layers
    horizontal = HorizontalConvolution(filters_per_expand=5)
    vertical = VerticalConvolution(filters=5)

    # Dense layer that allows to query only some outputs by items
    dense = DenseByItems(num_items=self.num_movies, latent_size=self.latent_size)
    dropout = tf.keras.layers.Dropout(rate=0.3)

    # Connects rest of the model
    z = tf.keras.layers.Dense(self.latent_size)(
      dropout(tf.concat([horizontal(E), vertical(E)], axis=-1))
    )
    p = P(input_user)
    outputs = dense([tf.concat([z, p], axis=-1), input_items])

    # Builds and return keras model
    model = tf.keras.models.Model(inputs=[input_sequence, input_user, input_items], outputs=outputs)
    model.summary()
    return model

  @tf.function
  def train_step(self, batch):
    # Unpacking inputs
    user, sequence, positive, negative = batch

    # Recording ops to compute gradient
    with tf.GradientTape() as tape:
      y_positive = tf.squeeze(self.model([sequence, user, positive]))
      y_negative = tf.squeeze(self.model([sequence, user, negative]))
      loss = self.loss_fc(y_positive, y_negative)

    # Updates params
    trainable_variables = self.model.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss

  @tf.function
  def test_step(self, batch):
    # Unpacking inputs
    user, sequence, target = batch

    # Prepares inputs for network
    movies = tf.expand_dims(tf.range(self.num_movies), axis=0)

    # Finds ranking of items
    prediction = tf.squeeze(self.model([sequence, user, movies]))

    # Updates metrics
    self.hit.update_state(prediction, target)
    self.auc.update_state(prediction, target)

  def train(self, epochs, steps_per_epoch, evaluate):
    for epoch in range(1, epochs + 1):
      # Training (evaluating loss!)
      train_loss = tf.constant(0, dtype=tf.float32)
      for step, batch in self.train_ds.take(steps_per_epoch).enumerate(start=1):
        train_loss += self.train_step(batch)
        print(f"Completed {step}/{steps_per_epoch} steps with "
              f"loss {train_loss / tf.cast(step, train_loss.dtype):.2f}", end='\r')
      train_loss /= tf.cast(steps_per_epoch, train_loss.dtype)

      # Evaluation test set performance!
      test_steps = tf.data.experimental.cardinality(self.test_ds)
      if evaluate:
        self.hit.reset_state()
        self.auc.reset_state()
        for step, batch in self.test_ds.enumerate():
          self.test_step(batch)
          print(f"Completed {step}/{test_steps}: "
                f"Hit Rate@{self.hit.k} {self.hit.result():.2f}, "
                f"AUC: {self.auc.result():.2f}", end='\r')

      # Prints epoch progress
      print(f"Epoch {epoch} out of {epochs}: Loss {train_loss:.2f}", end='')
      print(f", Test Hit Rate@{self.hit.k} {self.hit.result():.2f}, "
            f"AUC: {self.auc.result():.2f}\n" if evaluate else '\n', end='')


def main(parser):
  # Parse arguments
  args = parser.parse_args()

  # Verifies if a supported dataset version has been selected
  if args.dataset not in version_to_function:
    parser.error(f'Dataset version not supported! Choose one of: {versions}')

  # Allowing memory growth if using GPU
  devices = tf.config.list_physical_devices('GPU')
  if args.gpu and len(devices) > 0:
    tf.config.experimental.set_memory_growth(devices[0], True)

  # Loads data
  train_df, test_df, user_to_index, movie_to_index, movie_to_title = version_to_function[args.dataset](
    split='sequence'
  )
  # Creates and trains model
  with tf.device('GPU:0' if args.gpu else 'CPU:0'):
    caser = Caser(train_df,
                  test_df,
                  sequence_length=5,
                  user_to_index=user_to_index,
                  movie_to_index=movie_to_index,
                  latent_size=args.latent,
                  k=50)
    caser.train(args.epochs, steps_per_epoch=args.steps, evaluate=args.test)


if __name__ == '__main__':
  version_to_function = {
    'ml-20m': utils.load_ratings_20m,
    'ml-100k': utils.load_ratings_100k
  }
  versions = ', '.join(version_to_function.keys())
  args_parser = argparse.ArgumentParser()
  args_parser.add_argument('--dataset',
                           help=f'Dataset to load: {versions}',
                           required=True,
                           type=str)
  args_parser.add_argument('--gpu',
                           help='Trains in GPU if available',
                           action='store_true')
  args_parser.add_argument('--test',
                           help='Evaluates test performance during training or not',
                           action='store_true')
  args_parser.add_argument('--latent',
                           help='Latent size (Default: 64)',
                           type=int,
                           default=64)
  args_parser.add_argument('--epochs',
                           help='Number of epochs to train (Default: 10)',
                           type=int,
                           default=10)
  args_parser.add_argument('--steps',
                           help='Number of steps per epoch to train (Default: 100)',
                           type=int,
                           default=100)
  args_parser.add_argument('--serving',
                           help='Version number for tf serving (Default: 1)',
                           type=int,
                           default=1)
  args_parser.add_argument('--unsaved',
                           help='Trained mode will not be saved on file',
                           action="store_true")
  args_parser.add_argument('--learning',
                           help='Learning rate (Default: 1e-3)',
                           type=float,
                           default=1e-3)

  main(args_parser)
