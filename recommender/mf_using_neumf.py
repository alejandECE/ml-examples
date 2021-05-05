#  Created by Luis Alejandro (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import argparse
import tensorflow as tf
import pandas as pd
import utils

BATCH_SIZE = 512
BUFFER_SIZE = BATCH_SIZE * 32


class HingeLossRecommender(tf.keras.losses.Loss):
  def __init__(self, margin=1, **kwargs):
    super(HingeLossRecommender, self).__init__(**kwargs)
    self.margin = margin

  def call(self, positive, negative):
    distance = positive - negative
    loss = tf.maximum(self.margin - distance, 0)
    return loss


class NeuralMF(tf.Module):
  """
  Implements Neural Collaborative Filtering
  """

  def __init__(self,
               train_ratings: pd.DataFrame,
               test_ratings: pd.DataFrame,
               user_to_index: dict,
               movie_to_index: dict,
               latent_size=30,
               learning_rate=1e-3,
               hidden_neurons: list = None):
    super(NeuralMF, self).__init__()

    # For later use
    self.latent_size = latent_size
    self.learning_rate = learning_rate
    self.hidden_neurons = [self.latent_size] if not hidden_neurons else hidden_neurons
    self.generator = tf.random.get_global_generator()

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

    self.train_ds = self.create_pairwise_training_dataset(train_ratings)
    self.neuralmf = self.create_model()
    self.optimizer = tf.optimizers.Adam(lr=self.learning_rate)
    self.loss = HingeLossRecommender(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

  @tf.function
  def generate_pairwise_observations(self, user, ratings):
    # Number of positive interactions
    observations = tf.shape(ratings.indices)[0]

    # Gets the same number of negative interactions
    movies = tf.expand_dims(tf.range(self.num_movies, dtype=tf.int64), axis=-1)
    rated = tf.transpose(ratings.indices)
    mask = tf.reduce_all(tf.not_equal(movies, rated), axis=-1)
    candidates = tf.squeeze(tf.boolean_mask(movies, mask, axis=0))
    indices = tf.cast(
      self.generator.uniform(shape=(observations,), maxval=tf.cast(tf.shape(candidates)[0], dtype=tf.float32)),
      dtype=tf.int64
    )
    negatives = tf.gather(candidates, indices)

    # Returns tuples of (user, positive, negative)
    return (
      tf.repeat(user, repeats=observations),
      tf.squeeze(ratings.indices),
      negatives
    )

  def create_pairwise_training_dataset(self, train_ratings: pd.DataFrame) -> tf.data.Dataset:
    # Creates interaction dataset (user mapping to movies rated)
    interaction_ds = tf.data.Dataset.from_tensor_slices(
      tf.sparse.reorder(tf.sparse.SparseTensor(
        indices=train_ratings[['userId', 'movieId']].values,
        values=train_ratings['rating'].values,
        dense_shape=(self.num_users, self.num_movies)
      ))
    )

    # Generates pairwise observations with negative interactions sampled at random
    pairwise_ds = interaction_ds.enumerate().map(
      self.generate_pairwise_observations, num_parallel_calls=tf.data.AUTOTUNE
    ).unbatch().repeat()

    # Batches dataset
    batched_ds = pairwise_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    return batched_ds.prefetch(tf.data.AUTOTUNE)

  def create_model(self) -> tf.keras.models.Model:
    # Inputs
    input_user = tf.keras.layers.Input(shape=())
    input_movie = tf.keras.layers.Input(shape=())

    # Embeddings for the GMF
    P = tf.keras.layers.Embedding(self.num_users, self.latent_size)
    Q = tf.keras.layers.Embedding(self.num_movies, self.latent_size)

    # Embeddings for the MLP
    U = tf.keras.layers.Embedding(self.num_users, self.latent_size)
    V = tf.keras.layers.Embedding(self.num_movies, self.latent_size)

    # MLP
    MLP = tf.keras.models.Sequential()
    for neurons in self.hidden_neurons:
      MLP.add(tf.keras.layers.Dense(neurons, activation='relu'))

    # Last prediction layer!
    prediction = tf.keras.layers.Dense(1, activation='sigmoid', use_bias=False)

    # GMF path
    p = P(input_user)
    q = Q(input_movie)
    gmf = p * q

    # MLP path
    u = U(input_user)
    v = V(input_movie)
    mlp = MLP(tf.concat([u, v], axis=1))

    # Final prediction
    combination = tf.concat([gmf, mlp], axis=1)
    outputs = prediction(combination)

    # Creates model
    model = tf.keras.models.Model([input_user, input_movie], outputs)
    model.summary()

    return model

  @tf.function
  def train_step(self, batch):
    # Unpacking inputs
    user, positive, negative = batch

    # Recording ops to compute gradient
    with tf.GradientTape() as tape:
      y_positive = self.neuralmf([user, positive])
      y_negative = self.neuralmf([user, negative])
      loss = self.loss(y_positive, y_negative)

    # Updates params
    trainable_variables = self.neuralmf.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss

  def train(self, epochs, steps_per_epoch):
    for epoch in range(1, epochs + 1):
      train_loss = tf.constant(0, dtype=tf.float32)
      for step, batch in self.train_ds.take(steps_per_epoch).enumerate():
        train_loss += self.train_step(batch)
        print(f"Completed {step}/{steps_per_epoch} steps with "
              f"loss {train_loss / tf.cast(step, train_loss.dtype):.2f}", end='\r')
      train_loss /= tf.cast(steps_per_epoch, train_loss.dtype)

      # Prints epoch progress
      print(f"Loss after epoch {epoch} out of {epochs}: Training {train_loss:.2f}")


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
  train_ratings, test_ratings, user_to_index, movie_to_index, movie_to_title = version_to_function[args.dataset](
    timestamp=True
  )

  # Creates and trains model
  with tf.device('GPU:0' if args.gpu else 'CPU:0'):
    model = NeuralMF(
      train_ratings,
      test_ratings,
      user_to_index=user_to_index,
      movie_to_index=movie_to_index,
      latent_size=args.latent,
      learning_rate=args.learning
    )
    model.train(args.epochs, steps_per_epoch=args.steps)


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
  args_parser.add_argument('--latent',
                           help='Latent size (Default: 30)',
                           type=int,
                           default=30)
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
