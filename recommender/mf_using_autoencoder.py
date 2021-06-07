#  Created by Luis A. Sanchez-Perez (l.alejandro.2011@gmail.com).
#  Copyright © Do not distribute or use without authorization from author

import utils
import argparse
import tensorflow as tf
import pandas as pd
import datetime
import pathlib

BUFFER_SIZE = 10000
BATCH_SIZE = 64

ROOT = pathlib.Path(pathlib.Path(__file__).parent)


class UserBasedAutoRec(tf.Module):
  """
    This class implements a simple matrix factorization approach with the following characteristics and/or restrictions:
    1. Uses a user-based autoencoder
    2. Includes dropout regularization.
    3. Splits dataset into training/test (to evaluate our model).
    4. Exposes a function for tfserving (just for fun!)
    """

  def __init__(self,
               train_ratings: pd.DataFrame,
               test_ratings: pd.DataFrame,
               user_to_index: dict,
               movie_to_index: dict,
               latent_size=30,
               learning_rate=1e-3,
               dropout=0.1):

    super(UserBasedAutoRec, self).__init__()

    # For later use
    self.dropout = dropout
    self.learning_rate = learning_rate
    self.latent_size = latent_size
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

    # Creates dataset from dataframe to train and test
    self.train_ds = self._create_train_dataset(train_ratings)
    self.test_ds = self._create_test_dataset(test_ratings)

    # Creates model
    self.autoencoder = self._create_model()

  def _create_model(self) -> tf.keras.models.Model:
    # Layers
    inputs = tf.keras.layers.Input(shape=(self.num_movies,))
    encoder_dense = tf.keras.layers.Dense(self.latent_size, activation='sigmoid')
    decoder_dense = tf.keras.layers.Dense(self.num_movies, activation='relu')
    dropout = tf.keras.layers.Dropout(rate=self.dropout)

    # Connections
    x = encoder_dense(inputs)
    x = dropout(x)
    outputs = decoder_dense(x)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)

  def _create_train_dataset(self, train_ratings: pd.DataFrame) -> tf.data.Dataset:
    # Each entry is the user represented as a sparse tensor with rating entries from each movie rated
    users_ds = tf.data.Dataset.from_tensor_slices(
      tf.sparse.reorder(tf.sparse.SparseTensor(
        indices=train_ratings[['userId', 'movieId']].values,
        values=train_ratings['rating'].values,
        dense_shape=(self.num_users, self.num_movies,)
      ))
    )
    users_ds = users_ds.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE)
    return users_ds.prefetch(tf.data.AUTOTUNE)

  def _create_test_dataset(self, test_ratings: pd.DataFrame) -> tf.data.Dataset:
    # Each entry is the user represented as a sparse tensor with rating entries from each movie rated
    users_ds = tf.data.Dataset.from_tensor_slices(
      tf.sparse.reorder(tf.sparse.SparseTensor(
        indices=test_ratings[['userId', 'movieId']].values,
        values=test_ratings['rating'].values,
        dense_shape=(self.num_users, self.num_movies)
      ))
    )
    users_ds = users_ds.batch(BATCH_SIZE)
    return users_ds.prefetch(tf.data.AUTOTUNE)

  @tf.function
  def train_step(self, batch: tf.Tensor):
    with tf.GradientTape() as tape:
      # Predicts with the autoencoder
      outputs = self.autoencoder(batch)

      # Computes loss masking out those elements without known ratings
      y_true = batch.values
      y_pred = tf.gather_nd(outputs, indices=batch.indices)
      loss = tf.losses.mean_squared_error(y_true, y_pred)

    # Get gradient of loss with respect to users embeddings
    trainable_variables = self.autoencoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss

  @tf.function
  def test_step(self, batch: tf.Tensor):
    # Predicts with the autoencoder
    outputs = self.autoencoder(batch)

    # Computes loss masking out those elements without known ratings
    y_true = batch.values
    y_pred = tf.gather_nd(outputs, indices=batch.indices)
    loss = tf.losses.mean_squared_error(y_true, y_pred)

    return loss

  def train(self, epochs):
    for epoch in range(1, epochs + 1):
      # Training steps
      train_loss = tf.constant(0, dtype=tf.float32)
      total_steps = tf.data.experimental.cardinality(self.train_ds)
      for step, batch in self.train_ds.enumerate(start=1):
        train_loss += self.train_step(batch)
        print(f"Completed {step}/{total_steps} steps with "
              f"loss (RMSE) {tf.sqrt(train_loss / tf.cast(step, train_loss.dtype)):.2f}", end='\r')
      train_loss /= tf.cast(total_steps, train_loss.dtype)

      # Computes test error
      test_loss = tf.constant(0, dtype=tf.float32)
      total_steps = tf.data.experimental.cardinality(self.test_ds)
      for step, batch in self.test_ds.enumerate(start=1):
        test_loss += self.test_step(batch)
        print(f"Test: completed {step}/{total_steps} steps with "
              f"loss (RMSE) {tf.sqrt(test_loss / tf.cast(step, dtype=test_loss.dtype)):.2f}", end='\r')
      test_loss /= tf.cast(total_steps, test_loss.dtype)

      # Prints epoch progress
      print(f"Loss (RMSE) after epoch {epoch} out of {epochs}: "
            f"Training {tf.sqrt(train_loss):.2f}, "
            f"Test Loss {tf.sqrt(test_loss):.2f}")

  @tf.function
  def find_missing_ratings_for_user(self, ratings: tf.Tensor):
    return {'scores': self.autoencoder(ratings)}


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
    model = UserBasedAutoRec(
      train_ratings,
      test_ratings,
      latent_size=args.latent,
      learning_rate=args.learning,
      dropout=args.dropout,
      user_to_index=user_to_index,
      movie_to_index=movie_to_index
    )
    model.train(epochs=args.epochs)

  # Stores model (ready for serving!)
  if not args.unsaved:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = ROOT / f'models/{model.__class__.__name__}/' / timestamp / str(args.serving)
    tf.saved_model.save(model, str(path), signatures={
      'serving_users_scoring': model.find_missing_ratings_for_user.get_concrete_function(
        ratings=tf.TensorSpec(shape=(model.num_movies,), dtype=tf.float32)
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
  args_parser.add_argument('--latent',
                           help='Latent size (autoencoder hidden layer) (Default: 30)',
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
  args_parser.add_argument('--dropout',
                           help='Dropout rate (Default: 0.1)',
                           type=float,
                           default=0.1)
  args_parser.add_argument('--serving',
                           help='Version number for tf serving (Default: 1)',
                           type=int,
                           default=1)
  args_parser.add_argument('--unsaved',
                           help='Trained mode will not be saved on file',
                           action="store_true")

  main(args_parser)
