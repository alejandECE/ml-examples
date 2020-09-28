#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import argparse
from typing import Tuple
import tensorflow as tf
import os
import pathlib
import datetime
import utils

# Some constants & setups
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64
IMG_HEIGHT = 64
IMG_WIDTH = 64
TEST_SIZE = 1000
BUFFER_SIZE = 10000


# Decodes image to tensor, resizes and scale it from 0 to 1
@tf.function
def preprocess(img: tf.Tensor) -> tf.Tensor:
  # It only normalizes from [0, MAX) if the input arg is not a tf.float32 already
  img = tf.image.convert_image_dtype(img, dtype=tf.float32)
  # Resizes to the network input specs
  img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT], antialias=True)
  return img


# Extracts class from path
@tf.function
def get_class_from_path(file_path: tf.Tensor) -> tf.Tensor:
  parts = tf.strings.split(file_path, os.path.sep)
  return tf.equal(parts[-2], tf.constant('dogs', dtype=tf.string))
  # return parts[-2] == 'dogs'


# Gets data from a path returning the image data (width, high, channels), label (str)
@tf.function
def get_data_from_path(file_path: tf.Tensor) -> Tuple:
  label = get_class_from_path(file_path)
  img = tf.io.read_file(file_path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img, label


# Creates dataset
def create_dataset() -> Tuple:
  # Builds path to the images in disk
  data_dir = pathlib.Path('E:/datasets/classification/cats and dogs/')
  # Builds tf.data.Dataset listing filenames (files always shuffled)
  path_ds = tf.data.Dataset.list_files(str(data_dir / '*/*/*.jpg'), shuffle=False)
  print(path_ds.element_spec)
  print('Number of files to load: {}'.format(path_ds.cardinality()))
  # Builds tf.data.Dataset with the image data and its label
  labeled_ds = path_ds.map(get_data_from_path, num_parallel_calls=AUTOTUNE)
  print(labeled_ds.element_spec)
  labeled_ds = labeled_ds.map(lambda img, label: (preprocess(img), label), num_parallel_calls=AUTOTUNE).cache()
  # Splits into training/test sets
  labeled_ds = labeled_ds.shuffle(buffer_size=BUFFER_SIZE, reshuffle_each_iteration=False)
  test_ds = labeled_ds.take(TEST_SIZE)
  train_ds = labeled_ds.skip(TEST_SIZE)
  # Optimizes dataset for training
  train_ds = train_ds.shuffle(buffer_size=BUFFER_SIZE)
  train_ds = train_ds.batch(BATCH_SIZE)
  train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
  # Also batches test set
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
    tf.keras.layers.Dense(units=100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
  ])
  mdl.compile(
    optimizer=tf.keras.optimizers.Adam(decay=1e-3),
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
  parser.add_argument('--epochs', help='Number of epochs to train (Default: 10)', type=int)
  parser.add_argument('--unsaved', help='Trained mode will not be saved on file', action="store_true")
  parser.add_argument('--tensorboard', help='Logs information to be displayed in tensorboard', action="store_true")
  # Parses arguments
  args = parser.parse_args()
  epochs = args.epochs if args.epochs else 10
  save = False if args.unsaved else True
  tensorboard = True if args.tensorboard else False
  # Creates dataset to use augmentation or not
  training_ds, testing_ds = create_dataset()
  # Creates model
  model = create_model()
  model_path = pathlib.Path('trained_model/cnn1/') / timestamp
  model_path.mkdir(parents=True)
  # Sets callbacks if needed
  callbacks = []
  # Adds logging to show in tensorboard
  log_path = pathlib.Path("./logs") / timestamp
  log_path.mkdir(parents=True)
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
  results = model.fit(training_ds, validation_data=testing_ds, epochs=epochs, callbacks=callbacks, verbose=2)
  # Shows performance to console
  if (model_path / 'saved_model.pb').exists():
    saved_model = tf.keras.models.load_model(str(model_path))
    utils.display_performance(saved_model, training_ds, testing_ds)
