#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import argparse
from typing import Tuple
import tensorflow_datasets as tfds
import tensorflow as tf
import datetime
import utils

# Some constants & setups
PATH_PREFIX = 'cnn3'
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMG_HEIGHT = 32
IMG_WIDTH = 32
BUFFER_SIZE = 22000
AUGMENT_PADDING = 4


# Filters out cats observations (label: 0) from cats/dogs dataset
@tf.function
def filter_cats_out(label: tf.Tensor) -> bool:
  if tf.math.equal(label, tf.constant(0, dtype=tf.int64)):
    return False
  else:
    return True


# Preprocess all images when no augmentation is used
@tf.function
def preprocess(img: tf.Tensor) -> Tuple:
  # It only normalizes from [0, MAX) if the input arg is not a tf.float32 already
  img = tf.image.convert_image_dtype(img, dtype=tf.float32)
  # Resizes to the network input specs
  img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], antialias=True)
  return img


# Preprocess all images when augmentation is used
@tf.function
def augmented_preprocess(img: tf.Tensor, label: tf.Tensor) -> Tuple:
  # It only normalizes from [0, MAX) if the input arg is not a tf.float32 already
  img = tf.image.convert_image_dtype(img, dtype=tf.float32)
  # Increase in size a little to then apply a random crop and retain original size
  if tf.equal(label, tf.constant(0, dtype=tf.int64)):
    img = tf.image.resize_with_crop_or_pad(img, IMG_HEIGHT + AUGMENT_PADDING, IMG_WIDTH + AUGMENT_PADDING)
  else:
    img = tf.image.resize(img, [IMG_HEIGHT + AUGMENT_PADDING, IMG_WIDTH + AUGMENT_PADDING], antialias=True)
  return img, label


# Augments image by applying random transformations
@tf.function
def augment(img: tf.Tensor, label: tf.Tensor) -> Tuple:
  img = tf.image.random_crop(img, size=(IMG_HEIGHT, IMG_WIDTH, 3))
  img = tf.image.random_flip_left_right(img)
  img = tf.image.random_brightness(img, max_delta=0.5)
  return img, label


# Creates dataset without augmentation
def create_dataset(observations: int, test_size: float) -> Tuple:
  # Downloads and creates a td.data.Dataset with only dogs pictures
  dogs_ds = tfds.load('cats_vs_dogs',
                      as_supervised=True,
                      split='train',
                      shuffle_files=False,
                      data_dir=utils.TFDS_DATASETS)
  dogs_ds = dogs_ds.filter(lambda image, label: filter_cats_out(label)).take(observations // 2)
  dogs_ds = dogs_ds.map(lambda image, label: (preprocess(image), label), num_parallel_calls=AUTOTUNE).cache()
  # Downloads and creates a td.data.Dataset with other categories pictures
  others_ds = tfds.load('cifar100',
                        as_supervised=True,
                        split='train',
                        shuffle_files=False,
                        data_dir=utils.TFDS_DATASETS)
  others_ds = others_ds.take(observations // 2)
  # Change label of all observations in cifar-100 to 0 (class others)
  others_ds = others_ds.map(lambda image, label: (image, tf.constant(0, dtype=tf.int64)),
                            num_parallel_calls=AUTOTUNE).cache()
  others_ds = others_ds.map(lambda image, label: (preprocess(image), label), num_parallel_calls=AUTOTUNE).cache()
  # Combines both datasets
  mixed_ds = dogs_ds.concatenate(others_ds)
  # Splits into training/test sets
  mixed_ds = mixed_ds.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
  test_ds = mixed_ds.take(int(observations * test_size))
  train_ds = mixed_ds.skip(int(observations * test_size))
  # Optimizes and batches for training
  train_ds = train_ds.shuffle(buffer_size=BUFFER_SIZE)
  train_ds = train_ds.batch(BATCH_SIZE)
  train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
  # Also batches test set
  test_ds = test_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
  # Returns training and test sets datasets
  return train_ds, test_ds


# Creates augmented dataset
def create_augmented_dataset(observations: int, test_size: float) -> Tuple:
  # Downloads and creates a td.data.Dataset with only dogs pictures
  dogs_ds = tfds.load('cats_vs_dogs',
                      as_supervised=True,
                      split='train',
                      shuffle_files=False,
                      data_dir=utils.TFDS_DATASETS)
  dogs_ds = dogs_ds.filter(lambda image, label: filter_cats_out(label)).take(observations // 2)
  # Downloads and creates a td.data.Dataset with other categories pictures
  others_ds = tfds.load('cifar100',
                        as_supervised=True,
                        split='train',
                        shuffle_files=False,
                        data_dir=utils.TFDS_DATASETS)
  others_ds = others_ds.take(observations // 2)
  # Change label of all observations in cifar-100 to 0 (class others)
  others_ds = others_ds.map(lambda image, label: (image, tf.constant(0, dtype=tf.int64)),
                            num_parallel_calls=AUTOTUNE)
  # Combines both datasets
  mixed_ds = dogs_ds.concatenate(others_ds)
  # Splits into training/test sets
  mixed_ds = mixed_ds.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
  test_ds = mixed_ds.take(int(observations * test_size))
  train_ds = mixed_ds.skip(int(observations * test_size))
  # Applies pre-processing transformations to the training dataset
  augmented_ds = train_ds.map(augmented_preprocess, num_parallel_calls=AUTOTUNE).cache()
  # Augment training dataset (we don't cache this)
  augmented_ds = augmented_ds.map(augment, num_parallel_calls=AUTOTUNE)
  # Combines both datasets
  train_ds = train_ds.map(lambda image, label: (preprocess(image), label), num_parallel_calls=AUTOTUNE).cache()
  train_ds = train_ds.concatenate(augmented_ds)
  # Optimizes and batches for training
  train_ds = train_ds.shuffle(buffer_size=BUFFER_SIZE)
  train_ds = train_ds.batch(BATCH_SIZE)
  train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
  # Preprocess and batches test set
  test_ds = test_ds.map(lambda image, label: (preprocess(image), label), num_parallel_calls=AUTOTUNE).cache()
  test_ds = test_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
  return train_ds, test_ds


# Creates NN architecture
def create_model(dropout_rate: float, l2_rate: float) -> tf.keras.Model:
  # Builds a really simple CNN to classify cats/dogs
  mdl = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',
                           kernel_initializer=tf.keras.initializers.he_uniform()),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',
                           kernel_initializer=tf.keras.initializers.he_uniform()),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu',
                           kernel_initializer=tf.keras.initializers.he_uniform()),
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_rate)),
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
  parser.add_argument('--samples', help='Max # of samples to keep in the dataset (Default: 40k)', type=int)
  parser.add_argument('--test_pct', help='% of samples going to the test set (Default: 0.1)', type=float)
  parser.add_argument('--dropout', help='Dropout rate', type=float)
  parser.add_argument('--l2', help='l2 regularization amount', type=float)
  # Parses arguments
  args = parser.parse_args()
  augmentation = True if args.augment else False
  epochs = args.epochs if args.epochs else 10
  samples = args.samples if args.samples else 40_000
  test_pct = args.test_pct if args.test_pct else 0.1
  save = False if args.unsaved else True
  tensorboard = True if args.tensorboard else False
  dropout = args.dropout if args.dropout and args.dropout < 1. else 0.
  l2 = args.l2 if args.l2 else 0.
  # Creates dataset to use augmentation or not
  if augmentation:
    training_ds, testing_ds = create_augmented_dataset(observations=samples, test_size=test_pct)
  else:
    training_ds, testing_ds = create_dataset(observations=samples, test_size=test_pct)
  # Creates model
  model = create_model(dropout_rate=dropout, l2_rate=l2)
  model_path = utils.OUTPUTS / PATH_PREFIX / timestamp / 'model'
  # Sets callbacks if needed
  callbacks = []
  # Adds logging to show in tensorboard
  log_path = utils.OUTPUTS / PATH_PREFIX / timestamp / 'logs'
  log_path.mkdir(parents=True)
  # Creates docker runner file
  utils.create_tensorboard_docker_runner(utils.OUTPUTS / PATH_PREFIX)
  # Logs command line options
  utils.create_commandline_options_log(
    log_path,
    {
      'Augmentation': augmentation,
      'Epochs': epochs,
      'Samples': samples,
      'Test %': test_pct,
      'Dropout': dropout,
      'L2': l2
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
