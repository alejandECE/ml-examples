#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import pathlib
from typing import Tuple
import tensorflow_datasets as tfds
import tensorflow as tf
import datetime
from tensorflow_core.python.keras.callbacks import History

# Some constants & setups
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMG_HEIGHT = 64
IMG_WIDTH = 64
TEST_SIZE = 1000
BUFFER_SIZE = 10000
USE_AUGMENTATION = True
AUGMENT_PADDING = 10


# Normalizes image between 0 and 1
@tf.function
def normalize(img: tf.Tensor) -> tf.Tensor:
  # It only normalizes from [0, MAX) if the input arg is not a tf.float32 already
  img = tf.image.convert_image_dtype(img, dtype=tf.float32)
  minimum = tf.math.reduce_min(img)
  maximum = tf.math.reduce_max(img)
  img = (img - minimum) / (maximum - minimum)
  return img


# Preprocess applied when no augmentation is used
@tf.function
def preprocess(img: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], antialias=True)
  return normalize(img), label


# Preprocess applied when augmenting data
@tf.function
def augmented_preprocess(img: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  img = tf.image.resize(img, [IMG_HEIGHT + AUGMENT_PADDING, IMG_WIDTH + AUGMENT_PADDING], antialias=True)
  return img, label


# Augments image by applying random transformations
@tf.function
def augment(img: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  img = tf.image.random_crop(img, size=(IMG_HEIGHT, IMG_WIDTH, 3))
  img = tf.image.random_flip_left_right(img)
  img = tf.image.random_brightness(img, max_delta=0.5)
  return normalize(img), label


# Creates dataset without augmentation
def create_dataset() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  # Downloads and creates td.data.Dataset
  ds = tfds.load('cats_vs_dogs', as_supervised=True, split='train', shuffle_files=False).cache()

  # Preprocess images
  ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE).cache()

  # Splits into training/test sets
  ds = ds.shuffle(buffer_size=BUFFER_SIZE, reshuffle_each_iteration=False)
  test_ds = ds.take(TEST_SIZE)
  train_ds = ds.skip(TEST_SIZE)

  # Optimizes and batches for training
  train_ds = train_ds.shuffle(buffer_size=BUFFER_SIZE)
  train_ds = train_ds.batch(BATCH_SIZE)
  train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

  # Also batches test set
  test_ds = test_ds.batch(BATCH_SIZE)

  return train_ds, test_ds


# Creates augmented dataset
def create_augmented_dataset() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  # Downloads and creates td.data.Dataset
  ds = tfds.load('cats_vs_dogs', as_supervised=True, split='train', shuffle_files=False).cache()

  # Splits into training/test sets
  ds = ds.shuffle(buffer_size=BUFFER_SIZE, reshuffle_each_iteration=False)
  test_ds = ds.take(TEST_SIZE)
  train_ds = ds.skip(TEST_SIZE)

  # Preprocess before augmenting (can be cached)
  train_ds = train_ds.map(augmented_preprocess, num_parallel_calls=AUTOTUNE).cache()

  # Augment training dataset (we don't cache this)
  train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)

  # Optimizes and batches for training
  train_ds = train_ds.shuffle(buffer_size=BUFFER_SIZE)
  train_ds = train_ds.batch(BATCH_SIZE)
  train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

  # Preprocess and batches test set
  test_ds = test_ds.map(preprocess, num_parallel_calls=AUTOTUNE).cache()
  test_ds = test_ds.batch(BATCH_SIZE)

  return train_ds, test_ds


# Shows performance after training displaying accuracy and confusion matrix
def show_performance(history: History) -> None:
  # Training
  acc = history.history['binary_accuracy'][-1]
  tp = history.history['true_positives'][-1]
  fp = history.history['false_positives'][-1]
  fn = history.history['false_negatives'][-1]
  tn = history.history['true_negatives'][-1]
  print('------- Training -------')
  print('Accuracy: {:.2f}'.format(acc))
  print('Confusion Matrix:')
  print('{:.0f} {:.0f}'.format(tp, fp))
  print('{:.0f} {:.0f}'.format(fn, tn))
  # Testing
  acc = history.history['val_binary_accuracy'][-1]
  tp = history.history['val_true_positives'][-1]
  fp = history.history['val_false_positives'][-1]
  fn = history.history['val_false_negatives'][-1]
  tn = history.history['val_true_negatives'][-1]
  print('\n------- Test -------')
  print('Accuracy: {:.2f}'.format(acc))
  print('Confusion Matrix:')
  print('{:.0f} {:.0f}'.format(tp, fp))
  print('{:.0f} {:.0f}'.format(fn, tn))


if __name__ == '__main__':
  # Builds a really simple CNN to classify cats/dogs
  mdl = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.000)),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
  ])
  mdl.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='binary_crossentropy',
    metrics=[
      tf.keras.metrics.BinaryAccuracy(),
      tf.keras.metrics.TruePositives(),
      tf.keras.metrics.FalsePositives(),
      tf.keras.metrics.FalseNegatives(),
      tf.keras.metrics.TrueNegatives()
    ]
  )
  # Workaround to show the summary of the model (because we don't want to specify the input shape)
  mdl(tf.keras.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3]))
  mdl.summary()
  # Adds logging to show in tensorboard
  log_dir = pathlib.Path("./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  # Trains model
  if USE_AUGMENTATION:
    training_ds, testing_ds = create_augmented_dataset()
  else:
    training_ds, testing_ds = create_dataset()
  results = mdl.fit(training_ds, validation_data=testing_ds, epochs=10, callbacks=[], verbose=2)
  # Shows performance to console
  show_performance(results)
  # Saves model
  # mdl.save('trained_model/cnn2/')



