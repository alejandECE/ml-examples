#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

from typing import Tuple
import tensorflow as tf
import os
import pathlib
import datetime

# Some constants & setups
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMG_HEIGHT = 64
IMG_WIDTH = 64
TEST_SIZE = 1000
BUFFER_SIZE = 10000


# Normalizes image between 0 and 1
@tf.function
def normalize(img: tf.Tensor) -> tf.Tensor:
  # It only normalizes from [0, MAX) if the input arg is not a tf.float32 already
  img = tf.image.convert_image_dtype(img, dtype=tf.float32)
  minimum = tf.math.reduce_min(img)
  maximum = tf.math.reduce_max(img)
  img = (img - minimum) / (maximum - minimum)
  return img


# Decodes image to tensor, resizes and scale it from 0 to 1
@tf.function
def preprocess(img: tf.Tensor) -> tf.Tensor:
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT], antialias=True)
  return normalize(img)


# Extracts class from path
@tf.function
def get_class_from_path(file_path: str) -> tf.Tensor:
  parts = tf.strings.split(file_path, os.path.sep)
  return parts[-2] == 'dogs'


# Gets data from a path returning the image data (width, high, channels), label (str)
@tf.function
def get_data_from_path(file_path: str) -> Tuple[tf.Tensor, tf.Tensor]:
  label = get_class_from_path(file_path)
  img = tf.io.read_file(file_path)
  img = preprocess(img)
  return img, label


# Creates dataset
def create_dataset():
  # Builds path to the images in disk
  data_dir = '../../../datasets/classification/cats and dogs/'
  data_dir = pathlib.Path(data_dir)

  # Builds tf.data.Dataset listing filenames (files always shuffled)
  path_ds = tf.data.Dataset.list_files(str(data_dir / '*/*/*.jpg'), shuffle=False).cache()

  # Builds tf.data.Dataset with the image data and its label
  labeled_ds = path_ds.map(get_data_from_path, num_parallel_calls=AUTOTUNE).cache()

  # Splits into training/test sets
  labeled_ds = labeled_ds.shuffle(buffer_size=BUFFER_SIZE, reshuffle_each_iteration=False)
  test_ds = labeled_ds.take(TEST_SIZE)
  train_ds = labeled_ds.skip(TEST_SIZE)

  # Builds an optimized and batched tf.data.Dataset for training
  train_ds = train_ds.shuffle(buffer_size=BUFFER_SIZE)
  train_ds = train_ds.batch(BATCH_SIZE)
  train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

  # Also batches test set
  test_ds = test_ds.batch(BATCH_SIZE)

  return train_ds, test_ds


if __name__ == '__main__':
  # Builds a really simple CNN to classify cats/dogs
  mdl = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002)),
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
  # Adds logging to show in tensorboard
  log_dir = pathlib.Path("./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  # Trains model
  training_ds, testing_ds = create_dataset()
  mdl.fit(training_ds, validation_data=testing_ds, epochs=10, callbacks=[tensorboard])
  # Saves model
  # mdl.save('trained_model/cnn1/')
