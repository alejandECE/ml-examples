#  Created by Luis Alejandro (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import pathlib
from typing import Tuple
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import re


# Constants and utilities
ONLY_NAME_REGEX = re.compile(r'([\w]*)')
BATCH_SIZE = 512
BUFFER_SIZE = BATCH_SIZE * 32


# Creates datasets
def create_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset, dict]:
  # Reads csv using pandas since it fits in memory
  path = pathlib.Path('E:/datasets/regression/bodyfat.csv')
  data = pd.read_csv(path)
  data.columns = [re.match(ONLY_NAME_REGEX, column).group(1) for column in data.columns]
  # Extracts label
  labels = data.pop('Bodyfat')
  # Train/test split
  train_features, test_features, train_labels, test_labels = train_test_split(data, labels)
  # Converts to tf.data.Dataset
  train_ds = tf.data.Dataset.from_tensor_slices((dict(train_features), train_labels))
  train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
  eval_ds = tf.data.Dataset.from_tensor_slices((dict(test_features), test_labels))
  eval_ds = eval_ds.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
  return train_ds, eval_ds, {key: key for key in data.columns}


def sift_feature_from_dataset(name: str, dataset: tf.data.Dataset) -> tf.data.Dataset:
  return dataset.map(lambda features, label: features[name])


def build_normalization_layer(dataset: tf.data.Dataset) -> tf.keras.layers.Layer:
  layer = tf.keras.layers.experimental.preprocessing.Normalization()
  layer.adapt(dataset)
  return layer


# Creates model with including some preprocessing layers
def create_model(columns: dict, dataset: tf.data.Dataset, normalize=False) -> tf.keras.Model:
  # Building model
  inputs = {key: tf.keras.layers.Input(name=key, shape=()) for key in columns.keys()}
  normalized = {}
  for key, feature in inputs.items():
    if key is 'Age':
      normalized[key] = feature
    else:
      normalizer = build_normalization_layer(sift_feature_from_dataset(key, dataset))
      normalized[key] = normalizer(feature)
  if normalize:
    x = tf.keras.layers.DenseFeatures(columns.values())(normalized)
  else:
    x = tf.keras.layers.DenseFeatures(columns.values())(inputs)
  x = tf.keras.layers.Dense(128, activation='relu')(x)
  x = tf.keras.layers.Dense(64, activation='relu')(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  outputs = tf.keras.layers.Dense(1)(x)
  mdl = tf.keras.Model(inputs, outputs)
  # Defining loss, metric and optimizer
  mdl.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.MeanSquaredError()
  )
  # Plots summary and stores in into file
  mdl.summary()
  tf.keras.utils.plot_model(mdl, to_file='model.png', show_shapes=True, rankdir="LR")
  return mdl


def train_and_evaluate():
  # Creates datasets
  train_ds, test_ds, keys = create_datasets()
  # Creates columns
  name = keys.pop('Age')
  age = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(name),
    boundaries=[10, 20, 30, 40, 50, 60, 70]
  )
  # demo_feature_column(age, batch)
  columns = {name: tf.feature_column.numeric_column(name) for name in keys}
  columns['Age'] = age
  # Creates keras layer to handle columns
  model = create_model(columns, train_ds)
  # Creates some callbacks
  callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='saved', monitor='val_loss', save_best_only=True)]
  # Trains model
  model.fit(train_ds, epochs=200, validation_data=test_ds, callbacks=callbacks)
