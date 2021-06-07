#  Created by Luis A. Sanchez-Perez (l.alejandro.2011@gmail.com).
#  Copyright © Do not distribute or use without authorization from author

import argparse
import tensorflow as tf
import pathlib
import pandas as pd
import utils
import datetime
from tensorboard.plugins import projector

TEST_BATCH_SIZE = 128

ROOT = pathlib.Path(pathlib.Path(__file__).parent)


class BasicMF(tf.Module):
  """
  This class implements a simple matrix factorization approach with the following characteristics:
  1. Uses gradient descent.
  2. Does not compute the entire interaction matrix but instead access each rating from a sparse tensor. However,
     it does use a lot of memory doing it so. Repeats each user and movie embedding as many times as ratings there are
     involving those two! Still computable for dataset ml-20m using CPU/RAM but not so with my GPU (6GB).
  3. Includes L2 regularization.
  4. Splits dataset into training/test (to evaluate our model).
  5. Exposes a couple of functions for tfserving (just for fun!)
  """

  def __init__(self,
               train_ratings: pd.DataFrame,
               test_ratings: pd.DataFrame,
               user_to_index: dict,
               movie_to_index: dict,
               embedding_size=30,
               alpha=1e-3,
               learning_rate=1):
    super(BasicMF, self).__init__()

    # For later use
    self.alpha = alpha
    self.learning_rate = learning_rate
    self.embedding_size = embedding_size
    self.optimizer = tf.optimizers.Adam(lr=self.learning_rate)

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
    self.interactions = self._create_interaction_matrix(train_ratings)

    # Matrix where each row is an encoded user
    self.users_embeddings = tf.Variable(
      tf.random.normal(shape=(self.num_users, embedding_size)),
      name='users_embeddings'
    )

    # Matrix where each row is an encoded movie
    self.movies_embeddings = tf.Variable(
      tf.random.normal(shape=(self.num_movies, embedding_size)),
      name='movies_embeddings'
    )

    # Creates dataset for testing
    self.test_ds = self._create_test_dataset(test_ratings)

  def _create_interaction_matrix(self, ratings: pd.DataFrame) -> tf.sparse.SparseTensor:
    # Creates sparse tensor representing interaction matrix
    return tf.sparse.reorder(tf.sparse.SparseTensor(
      indices=ratings[['userId', 'movieId']].values,
      values=ratings['rating'].values,
      dense_shape=(self.num_users, self.num_movies)
    ))

  def _create_test_dataset(self, test_ratings: pd.DataFrame) -> tf.data.Dataset:
    users_ds = tf.data.Dataset.from_tensor_slices(test_ratings['userId'])
    movies_ds = tf.data.Dataset.from_tensor_slices(test_ratings['movieId'])
    ratings_ds = tf.data.Dataset.from_tensor_slices(test_ratings['rating'])
    test_ds = tf.data.Dataset.zip((users_ds, movies_ds, ratings_ds)).batch(TEST_BATCH_SIZE)
    return test_ds.prefetch(tf.data.AUTOTUNE)

  @tf.function
  def update_embeddings(self):
    with tf.GradientTape() as tape:
      # This is repeating user and movie embeddings as many times as ratings there are involving those two!
      # Memory intensive (although is better than computing the entire predicted matrix)
      # However still this takes too much space in my GPU and gives OOM (if using ml-20m)
      # If we use the CPU and/or use a reduced size dataset it runs fine.
      y_pred = tf.reduce_sum(
        tf.gather(self.users_embeddings, indices=self.interactions.indices[:, 0]) *  # element-wise product
        tf.gather(self.movies_embeddings, indices=self.interactions.indices[:, 1]),
        axis=1
      )
      y_true = self.interactions.values

      # Computes prediction loss
      prediction_loss = tf.losses.mean_squared_error(y_true, y_pred)

      # Computes regularization term
      regularization = (tf.reduce_sum(self.movies_embeddings * self.movies_embeddings) +
                        tf.reduce_sum(self.users_embeddings * self.users_embeddings))
      regularization /= tf.cast(tf.shape(self.interactions.values)[0], regularization.dtype)
      loss = prediction_loss + self.alpha * regularization

    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    return prediction_loss

  @tf.function
  def compute_test_loss(self, batch):
    # Unpacks data
    users, movies, y_true = batch

    # Computes the prediction by computing the dot product of users and movies embeddings
    y_pred = tf.reduce_sum(
      tf.gather(self.users_embeddings, indices=users, axis=0) *  # element-wise product
      tf.gather(self.movies_embeddings, indices=movies, axis=0),
      axis=1
    )

    return tf.losses.mean_squared_error(y_true, y_pred)

  def train(self, epochs):
    for epoch in range(1, epochs + 1):
      # Train loss
      train_loss = self.update_embeddings()

      # Computes test loss
      test_loss = tf.constant(0, dtype=tf.float32)
      total_steps = tf.data.experimental.cardinality(self.test_ds)
      for batch in self.test_ds:
        test_loss += self.compute_test_loss(batch)
      test_loss /= tf.cast(total_steps, dtype=test_loss.dtype)

      # Prints epoch progress
      print(f"Training Loss (R-MSE) after epoch {epoch} out of {epochs}: "
            f"({tf.sqrt(train_loss):.2f}) - Test Loss (R-MSE): {tf.sqrt(test_loss):.2f}")

  @tf.function
  def find_ratings_by_user(self, user_id):
    index = self.user_to_index.lookup(user_id)
    return {'scores': tf.matmul(
      self.movies_embeddings, tf.expand_dims(self.users_embeddings[index, :], axis=-1)
    )}

  @tf.function
  def find_ratings_for_movie(self, movie_id):
    index = self.movie_to_index.lookup(movie_id)
    return {'scores': tf.matmul(
      self.users_embeddings, tf.expand_dims(self.movies_embeddings[index, :], axis=-1)
    )}

  def export_movies_embeddings(self, path: pathlib, movie_to_title: dict):
    # Creates folders if don't exist
    if not path.exists():
      path.mkdir(parents=True)
    # Creates metadata for the embedding projector
    with open(path / 'metadata.tsv', "w") as f:
      f.writelines([f'{title}\n' for title in movie_to_title.values()])
    # Creates checkpoint storing variable with embeddings matching the metadata info
    indexes = self.movie_to_index[tf.constant(list(movie_to_title.keys()))]
    embeddings = tf.gather(self.movies_embeddings, indices=indexes, axis=0)
    checkpoint = tf.train.Checkpoint(embeddings=tf.Variable(embeddings))
    checkpoint.save(str(path / "embeddings.ckpt"))
    # Set up projector config
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
    embedding.tensor_name = "embeddings/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(str(path), config)


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
  train_ratings, test_ratings, user_to_index, movie_to_index, movie_to_title = version_to_function[args.dataset]()

  # Creates and trains model
  with tf.device('GPU:0' if args.gpu else 'CPU:0'):
    model = BasicMF(
      train_ratings,
      test_ratings,
      embedding_size=args.embedding,
      learning_rate=args.learning,
      alpha=args.alpha,
      user_to_index=user_to_index,
      movie_to_index=movie_to_index
    )
    model.train(epochs=args.epochs)

  if not args.unsaved:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Stores model (ready for serving!)
    path = ROOT / f'models/{model.__class__.__name__}/' / timestamp / str(args.serving)
    tf.saved_model.save(model, str(path), signatures={
      'serving_movies_scoring': model.find_ratings_by_user.get_concrete_function(
        user_id=tf.TensorSpec(shape=(), dtype=tf.int32)
      ),
      'serving_users_scoring': model.find_ratings_for_movie.get_concrete_function(
        movie_id=tf.TensorSpec(shape=(), dtype=tf.int32)
      )
    })

    # Exports user/movies embeddings to visualize in Tensorboard
    path = ROOT / f'models/{model.__class__.__name__}/' / timestamp / str(args.serving) / 'embeddings'
    model.export_movies_embeddings(path, movie_to_title)


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
  args_parser.add_argument('--embedding',
                           help='Embedding size (Default: 30)',
                           type=int,
                           default=30)
  args_parser.add_argument('--epochs',
                           help='Number of epochs to train (Default: 10)',
                           type=int,
                           default=10)
  args_parser.add_argument('--learning',
                           help='Learning rate (Default: 1e-3)',
                           type=float,
                           default=1e-3)
  args_parser.add_argument('--alpha',
                           help='Regularization amount (Default: 1e-3)',
                           type=float,
                           default=1e-3)
  args_parser.add_argument('--serving',
                           help='Version number for tf serving (Default: 1)',
                           type=int,
                           default=1)
  args_parser.add_argument('--unsaved',
                           help='Trained mode will not be saved on file',
                           action="store_true")

  main(args_parser)
