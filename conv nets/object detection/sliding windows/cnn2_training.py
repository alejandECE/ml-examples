#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import pathlib
from typing import Tuple
import tensorflow_datasets as tfds
import tensorflow as tf
import datetime
import argparse
import utils

# Some constants & setups
PATH_PREFIX = 'cnn2'
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMG_HEIGHT = 64
IMG_WIDTH = 64
TEST_SIZE = 1000
BUFFER_SIZE = 10000
AUGMENT_PADDING = 10


# Preprocess applied when no augmentation is used
@tf.function
def preprocess(img: tf.Tensor) -> Tuple:
  # It only normalizes from [0, MAX) if the input arg is not a tf.float32 already
  img = tf.image.convert_image_dtype(img, dtype=tf.float32)
  # Resizes to the network input specs
  img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], antialias=True)
  return img


# Preprocess applied when augmenting data
@tf.function
def augmented_preprocess(img: tf.Tensor) -> Tuple:
  # It only normalizes from [0, MAX) if the input arg is not a tf.float32 already
  img = tf.image.convert_image_dtype(img, dtype=tf.float32)
  # Increase in size a little to then apply a random crop and retain original size
  img = tf.image.resize(img, [IMG_HEIGHT + AUGMENT_PADDING, IMG_WIDTH + AUGMENT_PADDING], antialias=True)
  return img


# Augments image by applying random transformations
@tf.function
def augment(img: tf.Tensor) -> Tuple:
  img = tf.image.random_crop(img, size=(IMG_HEIGHT, IMG_WIDTH, 3))
  img = tf.image.random_flip_left_right(img)
  img = tf.image.random_brightness(img, max_delta=0.5)
  return img


# Creates dataset without augmentation
def create_dataset() -> Tuple:
  # Downloads and creates td.data.Dataset
  ds = tfds.load('cats_vs_dogs',
                 split='train',
                 as_supervised=True,
                 shuffle_files=False,
                 data_dir=utils.TFDS_DATASETS)
  # Preprocess images
  ds = ds.map(lambda img, label: (preprocess(img), label), num_parallel_calls=AUTOTUNE).cache()
  # Splits into training/test sets
  ds = ds.shuffle(buffer_size=BUFFER_SIZE, reshuffle_each_iteration=False)
  test_ds = ds.take(TEST_SIZE)
  train_ds = ds.skip(TEST_SIZE)
  # Optimizes and batches for training
  train_ds = train_ds.shuffle(buffer_size=BUFFER_SIZE)
  train_ds = train_ds.batch(BATCH_SIZE)
  train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
  # Also batches test set
  test_ds = test_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
  return train_ds, test_ds


# Creates augmented dataset
def create_augmented_dataset() -> Tuple:
  # Downloads and creates td.data.Dataset
  ds = tfds.load('cats_vs_dogs',
                 split='train',
                 as_supervised=True,
                 shuffle_files=False,
                 data_dir=utils.TFDS_DATASETS)
  # Splits into training/test sets
  ds = ds.shuffle(buffer_size=BUFFER_SIZE, reshuffle_each_iteration=False)
  test_ds = ds.take(TEST_SIZE)
  train_ds = ds.skip(TEST_SIZE)
  # Preprocess before augmenting (can be cached)
  train_ds = train_ds.map(lambda img, label: (augmented_preprocess(img), label), num_parallel_calls=AUTOTUNE).cache()
  # Augment training dataset (we don't cache this)
  train_ds = train_ds.map(lambda img, label: (augment(img), label), num_parallel_calls=AUTOTUNE)
  # Optimizes and batches for training
  train_ds = train_ds.shuffle(buffer_size=BUFFER_SIZE)
  train_ds = train_ds.batch(BATCH_SIZE)
  train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
  # Preprocess and batches test set
  test_ds = test_ds.map(lambda img, label: (preprocess(img), label), num_parallel_calls=AUTOTUNE).cache()
  test_ds = test_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
  return train_ds, test_ds


def create_model() -> tf.keras.Model:
  # Builds a really simple CNN to classify cats/dogs
  mdl = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Dense(units=100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
  ])
  mdl.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
  )
  # Workaround to show the summary of the model (because we don't want to specify the input shape)
  mdl(tf.keras.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3]))
  mdl.summary()
  # Returns model
  return mdl


if __name__ == '__main__':
  # Generates for later use
  timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  # Allowing memory growth
  physical_devices = tf.config.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  # Defines arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--augment', help='Uses augmentation', action="store_true")
  parser.add_argument('--epochs', help='Number of epochs to train (Default: 10)', type=int)
  parser.add_argument('--unsaved', help='Trained mode will not be saved on file', action="store_true")
  parser.add_argument('--tensorboard', help='Logs information to be displayed in tensorboard', action="store_true")
  # Parses arguments
  args = parser.parse_args()
  augmentation = True if args.augment else False
  epochs = args.epochs if args.epochs else 10
  save = False if args.unsaved else True
  tensorboard = True if args.tensorboard else False
  # Creates dataset to use augmentation or not
  if augmentation:
    training_ds, testing_ds = create_augmented_dataset()
  else:
    training_ds, testing_ds = create_dataset()
  # Creates model
  model = create_model()
  model_path = utils.OUTPUTS / PATH_PREFIX / timestamp / 'model'
  model_path.mkdir(parents=True)
  # Sets callbacks if needed
  callbacks = []
  # Adds logging to show in tensorboard
  log_path = utils.OUTPUTS / PATH_PREFIX / timestamp / 'logs'
  log_path.mkdir(parents=True)
  # Creates docker runner file
  utils.create_tensorboard_docker_runner(utils.OUTPUTS / PATH_PREFIX)
  utils.create_tensorboard_docker_runner(utils.OUTPUTS / PATH_PREFIX)
  # Logs command line options
  utils.create_commandline_options_log(
    log_path,
    {
      'Augmentation': augmentation,
      'Epochs': epochs,
    }
  )
  if tensorboard:
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(log_path), profile_batch=0)
    callbacks.append(tensorboard_callback)
  # Callback to store model based on performance during training
  if save:
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=str(model_path),
      save_best_only=True,
      save_weights_only=False,
      monitor='val_loss',
      save_freq='epoch'
    )
    callbacks.append(checkpoint_callback)
  # Trains model
  model.fit(training_ds, validation_data=testing_ds, epochs=epochs, callbacks=callbacks, verbose=2)
  # Shows performance to console
  if (model_path / 'saved_model.pb').exists():
    saved_model = tf.keras.models.load_model(str(model_path))
    utils.display_performance(saved_model, training_ds, testing_ds)



