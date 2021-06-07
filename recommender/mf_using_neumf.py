#  Created by Luis A. Sanchez-Perez (l.alejandro.2011@gmail.com).
#  Copyright © Do not distribute or use without authorization from author

import argparse
import datetime
import pathlib
import tensorflow as tf
import pandas as pd
import utils

ROOT = pathlib.Path(pathlib.Path(__file__).parent)
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

    # Creates interaction matrix
    self.interactions = self.create_interaction_matrix(train_ratings)

    # Generates training/testing data pipelines
    self.train_ds = self.create_pairwise_training_dataset()
    self.test_ds = self.create_pairwise_test_dataset(test_ratings)

    # Model, optimizers and loss
    self.neuralmf = self.create_model()
    self.optimizer = tf.optimizers.Adam(lr=self.learning_rate)
    self.loss = HingeLossRecommender(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

  def create_interaction_matrix(self, ratings: pd.DataFrame) -> tf.sparse.SparseTensor:
    # Creates sparse tensor representing interaction matrix
    return tf.sparse.reorder(tf.sparse.SparseTensor(
      indices=ratings[['userId', 'movieId']].values,
      values=ratings['rating'].values,
      dense_shape=(self.num_users, self.num_movies)
    ))

  @tf.function
  def get_rated_movies(self, user):
    return tf.sparse.slice(self.interactions, start=[user, 0], size=[1, self.num_movies]).indices[:, 1]

  @tf.function
  def get_unrated_movies(self, user):
    # Gets rated movies from the user
    rated = tf.expand_dims(self.get_rated_movies(user), axis=0)

    # Gets all movies the user has not rated yet
    movies = tf.expand_dims(tf.range(self.num_movies, dtype=tf.int64), axis=-1)
    mask = tf.reduce_all(tf.not_equal(movies, rated), axis=1)
    candidates = tf.boolean_mask(movies, mask, axis=0)

    return tf.squeeze(candidates)

  @tf.function
  def generate_training_pairwise_observations(self, user):
    # Find rated/unrated movies
    positives = self.get_rated_movies(user)
    num_positives = tf.shape(positives)[0]
    candidates = self.get_unrated_movies(user)
    num_candidates = tf.shape(candidates)[0]

    # Samples from unrated movies
    indices = tf.cast(
      tf.random.stateless_uniform(
        shape=(num_positives,), seed=(2, 3), maxval=tf.cast(num_candidates, dtype=tf.float32)
      ), dtype=tf.int64
    )
    negatives = tf.gather(candidates, indices)

    # Returns tuples of (user, positive, negative)
    return (
      tf.repeat(user, repeats=num_positives),
      positives,
      negatives
    )

  def create_pairwise_training_dataset(self) -> tf.data.Dataset:
    # All users indices
    user_ds = tf.data.Dataset.range(self.num_users)

    # Generates pairwise observations with negative interactions sampled at random
    pairwise_ds = user_ds.map(
      self.generate_training_pairwise_observations, num_parallel_calls=tf.data.AUTOTUNE
    ).unbatch().repeat()

    # Batches dataset
    batched_ds = pairwise_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    return batched_ds.prefetch(tf.data.AUTOTUNE)

  @tf.function
  def generate_test_pairwise_observations(self, user):
    return self.get_unrated_movies(user)

  def create_pairwise_test_dataset(self, test_ratings: pd.DataFrame) -> tf.data.Dataset:
    # All users involve in the test set
    users_ds = tf.data.Dataset.from_tensor_slices(test_ratings['userId'])

    # All unrated movies by each user
    movies_ds = users_ds.map(
      lambda user: self.generate_test_pairwise_observations(user),
      num_parallel_calls=tf.data.AUTOTUNE
    )

    # Target movie to be rated
    target_ds = tf.data.Dataset.from_tensor_slices(test_ratings['movieId'])

    return tf.data.Dataset.zip((users_ds, movies_ds, target_ds)).prefetch(tf.data.AUTOTUNE)

  def create_model(self) -> tf.keras.models.Model:
    # Inputs
    input_user = tf.keras.layers.Input(shape=())
    input_movie = tf.keras.layers.Input(shape=())

    # Embeddings for the GMF
    P = tf.keras.layers.Embedding(self.num_users, self.latent_size, name='P')
    Q = tf.keras.layers.Embedding(self.num_movies, self.latent_size, name='Q')

    # Embeddings for the MLP
    U = tf.keras.layers.Embedding(self.num_users, self.latent_size, name='U')
    V = tf.keras.layers.Embedding(self.num_movies, self.latent_size, name='V')

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

  @tf.function(experimental_relax_shapes=True)
  def test_step(self, user, movies, target, k):
    # Number of movies to be rated!
    num_movies = tf.shape(movies)[0]

    # Prepares inputs for network
    users = tf.repeat(user, repeats=num_movies)

    # Finds ranking of movies
    ratings = self.neuralmf([users, movies])
    rank = tf.squeeze(tf.gather(movies, indices=tf.argsort(ratings, direction='DESCENDING', axis=0)))

    # Computes AUC and hit rate at k
    hit = tf.cast(tf.reduce_any(tf.equal(rank[:k], target)), dtype=tf.float32)
    auc = 1 - tf.cast(tf.squeeze(tf.where(tf.equal(rank, target))), dtype=tf.float32) / tf.cast(num_movies,
                                                                                                dtype=tf.float32)
    return hit, auc

  def train(self, epochs, steps_per_epoch, evaluate, k=20):
    for epoch in range(1, epochs + 1):
      # Training
      train_loss = tf.constant(0, dtype=tf.float32)
      for step, batch in self.train_ds.take(steps_per_epoch).enumerate():
        train_loss += self.train_step(batch)
        print(f"Completed {step}/{steps_per_epoch} steps with "
              f"loss {train_loss / tf.cast(step, train_loss.dtype):.2f}", end='\r')
      train_loss /= tf.cast(steps_per_epoch, dtype=train_loss.dtype)

      # Evaluation test set performance!
      test_steps = tf.data.experimental.cardinality(self.test_ds)
      test_hits = tf.constant(0, dtype=tf.float32)
      test_auc = tf.constant(0, dtype=tf.float32)
      if evaluate:
        for step, (user, movies, target) in self.test_ds.enumerate():
          step_hits, step_auc = self.test_step(user, movies, target, k)
          test_hits += step_hits
          test_auc += step_auc
          print(f"Completed {step}/{test_steps}: "
                f"Hit Rate@{k} {test_hits/tf.cast(step, dtype=test_hits.dtype):.2f}, "
                f"AUC: {test_auc/tf.cast(step, dtype=test_auc.dtype):.2f}", end='\r')
        test_hits /= tf.cast(test_steps, dtype=test_hits.dtype)
        test_auc /= tf.cast(test_steps, dtype=test_auc.dtype)

      # Prints epoch progress
      print(f"Epoch {epoch} out of {epochs}: Training Loss {train_loss:.2f}", end='')
      print(f", Test Hit Rate@{k} {test_hits:.2f}, AUC: {test_auc:.2f}" if evaluate else '\n', end='')

  @tf.function
  def get_movies_ranking_for_user(self, user_id, movies_ids):
    movies = self.movie_to_index.lookup(movies_ids)
    num_movies = tf.shape(movies)[0]
    users = tf.repeat(self.user_to_index.lookup(user_id), repeats=num_movies)
    return {'scores': self.neuralmf([users, movies])}

  @tf.function
  def get_users_embeddings(self, users_ids, kind):
    users = self.user_to_index.lookup(users_ids)
    if tf.equal(kind, 'U'):
      tf.print('U!')
      return self.neuralmf.get_layer('U')(users)
    else:
      tf.print('P!')
      return self.neuralmf.get_layer('P')(users)

  @tf.function
  def get_movies_embeddings(self, movies_indexes, kind):
    pass


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
    split='timestamp'
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
    model.train(args.epochs, steps_per_epoch=args.steps, evaluate=args.test)

  # Stores model (ready for serving!)
  if not args.unsaved:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = ROOT / f'models/{model.__class__.__name__}/' / timestamp / str(args.serving)
    tf.saved_model.save(model, str(path), signatures={
      'serving_movies_scoring': model.get_movies_ranking_for_user.get_concrete_function(
        user_id=tf.TensorSpec(shape=(), dtype=tf.int32),
        movies_ids=tf.TensorSpec(shape=(None,), dtype=tf.int32)
      ),
      'serving_users_embeddings': model.get_users_embeddings.get_concrete_function(
        users_ids=tf.TensorSpec(shape=(None, ), dtype=tf.int32),
        kind=tf.TensorSpec(shape=(), dtype=tf.string)
      )
    })


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
