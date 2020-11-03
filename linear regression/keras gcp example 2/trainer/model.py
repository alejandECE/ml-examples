#  Created by Luis Alejandro (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

from typing import Tuple
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pathlib
import re
import matplotlib.pyplot as plt

SEQUENCE_LENGTH = 14
STEP = 1
DATASETS = pathlib.Path('E:/datasets')
EXTRACT_NUMBER = re.compile(r'([0-9]+)')
BATCH_SIZE = 512
BUFFER_SIZE = BATCH_SIZE * 32


class R2Square(tf.keras.metrics.Metric):
  def __init__(self, name: str = "r2_square", **kwargs):
    super().__init__(name=name, **kwargs)
    self.squared_sum = self.add_weight(name="squared_sum", initializer="zeros", dtype=tf.float32)
    self.sum = self.add_weight(name="sum", initializer="zeros", dtype=tf.float32)
    self.residual = self.add_weight(name="residual", initializer="zeros", dtype=tf.float32)
    self.count = self.add_weight(name="count", initializer="zeros", dtype=tf.int32)

  def update_state(self, y_true, y_pred, sample_weight=None) -> None:
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    self.sum.assign_add(tf.reduce_sum(y_true))
    self.squared_sum.assign_add(tf.reduce_sum(y_true ** 2))
    self.residual.assign_add(tf.reduce_sum((y_true - y_pred) ** 2))
    self.count.assign_add(tf.cast(tf.shape(y_true)[0], dtype=self.count.dtype))

  def result(self) -> tf.Tensor:
    mean = self.sum / tf.cast(self.count, dtype=self.sum.dtype)
    total = self.squared_sum - self.sum * mean
    return 1 - (self.residual / total)

  def reset_states(self) -> None:
    for entry in self.variables:
      entry.assign(tf.zeros_like(entry))


def find_datafile_last_version(state: str) -> int:
  files = pathlib.Path('datasets').glob('{}-*.npz'.format(state))
  versions = sorted(
    [re.search(EXTRACT_NUMBER, str(entry)).group(1) for entry in files]
  )
  return None if len(versions) == 0 else int(versions[-1])


def load_datafile_with_version(state: str, version: int):
  npz = pathlib.Path('datasets/{}-{}.npz'.format(state, str(version).zfill(4)))
  if npz.exists():
    data = np.load(npz, allow_pickle=True)
    train = data['train']
    test = data['test']
    dates = data['dates']
    data.close()
    return train, test, dates
  else:
    print('Could not find {}'.format(str(npz)))
    return None, None, None


def restore_datafiles(state: str, version: int) -> Tuple:
  if version is None:
    version = find_datafile_last_version(state)
  return load_datafile_with_version(state, version)


def create_datafiles_version(state: str) -> Tuple:
  # Loads data from csv (original data)
  df = pd.read_csv(DATASETS / 'covid/summary.csv', index_col='Unnamed: 0')
  # Filters by state
  df = df[df['state'] == state]
  # Groups by date (this will add all confirmed cases from all counties)
  cases = df.groupby(by=['date'])['confirmed_cases'].sum().diff().fillna(0)
  # Generates indexes to create sequences (including expected output)
  base = np.arange(SEQUENCE_LENGTH + 1).reshape(1, -1)
  shifter = np.arange(0, len(cases) - SEQUENCE_LENGTH, STEP).reshape(-1, 1)
  indices = shifter + base
  # Actually creates sequences
  samples = np.take(cases.values, indices)
  dates = np.array(cases.index[(shifter + SEQUENCE_LENGTH).flatten()]).reshape(-1, 1)
  # Creates training and test sets and stores them into npy files
  m = int(samples.shape[0] * 0.6)
  train = samples[:m, :]
  test = samples[m:, :]
  version = find_datafile_last_version(state)
  version = 1 if version is None else version + 1
  path = pathlib.Path()
  np.savez('datasets/{}-{}.npz'.format(state, str(version).zfill(4)), train=train, test=test, dates=dates)
  return train, test, dates


def create_datasets(state: str, version: int = None) -> Tuple:
  # Tries to restore datasets
  train, test, dates = restore_datafiles(state, version)
  # If restore fails lets prepare new datasets
  if train is None:
    train, test, dates = create_datafiles_version(state)
  # Creates a dataset with all data for plotting results only
  dates_ds = tf.data.Dataset.from_tensor_slices(dates)
  data_ds = tf.data.Dataset.from_tensor_slices(np.concatenate(
    (train, test), axis=0
  ))
  data_ds = data_ds.map(
    lambda tensor: (tensor[:-1], tf.expand_dims(tensor[-1], axis=-1)),
    num_parallel_calls=tf.data.experimental.AUTOTUNE
  ).cache()
  all_ds = tf.data.Dataset.zip((dates_ds, data_ds))
  # Actually creates the tf.data.Dataset objects
  train_ds = tf.data.Dataset.from_tensor_slices(train).map(
    lambda tensor: (tensor[:-1], tf.expand_dims(tensor[-1], axis=-1)),
    num_parallel_calls=tf.data.experimental.AUTOTUNE
  ).cache()
  train_ds = train_ds.filter(lambda features, label: tf.reduce_any(tf.not_equal(features, 0)))
  train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
  test_ds = tf.data.Dataset.from_tensor_slices(test).map(
    lambda tensor: (tensor[:-1], tf.expand_dims(tensor[-1], axis=-1)),
    num_parallel_calls=tf.data.experimental.AUTOTUNE
  ).cache()
  test_ds = test_ds.filter(lambda features, label: tf.reduce_any(tf.not_equal(features, 0)))
  test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)
  return train_ds, test_ds, all_ds


def plot_results(model: tf.keras.Model, dataset: tf.data.Dataset):
  plt.figure(figsize=(15, 5))
  expected = [label.numpy()[0] for _, (_, label) in dataset]
  estimation = [tf.squeeze(model(tf.expand_dims(features, axis=0))).numpy() for _, (features, label) in dataset]
  dates = [date.numpy()[0].decode('utf-8') for date, _ in dataset]
  plt.plot(expected, 'b')
  plt.plot(estimation, 'm')
  plt.xticks(range(0, len(dates), 5), dates[::5], rotation=90)
  plt.grid(axis='x')
  plt.show()


def create_linear_model():
  mdl = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=(SEQUENCE_LENGTH,), activation='relu')
  ])
  mdl.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[R2Square()]
  )
  mdl.summary()
  tf.keras.utils.plot_model(mdl, to_file='linear.png', show_shapes=True, rankdir="LR")
  return mdl


def create_dnn_model():
  mdl = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_shape=(SEQUENCE_LENGTH,), activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation='relu')
  ])
  mdl.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[R2Square()]
  )
  mdl.summary()
  tf.keras.utils.plot_model(mdl, to_file='dnn.png', show_shapes=True, rankdir="LR")
  return mdl


def create_rnn_model():
  mdl = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(SEQUENCE_LENGTH,)),
    # tf.keras.layers.experimental.preprocessing.Normalization(),
    tf.keras.layers.Reshape(target_shape=(SEQUENCE_LENGTH, 1)),
    tf.keras.layers.LSTM(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())
  ])
  mdl.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[R2Square()]
  )
  mdl.summary()
  tf.keras.utils.plot_model(mdl, to_file='rnn.png', show_shapes=True, rankdir="LR")
  return mdl


def train_and_evaluate(args):
  # Trains on CPU
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  # Parses arguments
  epochs = args.epochs if args.epochs else 100
  state = args.state if args.state else 'MI'
  version = args.version if args.version else None
  model_type = args.model_type if args.model_type else 'rnn'
  # Creates datasets
  train_ds, test_ds, all_ds = create_datasets(state=state, version=version)
  # Creates model
  if model_type == "rnn":
    model = create_rnn_model()
  elif model_type == "dnn":
    model = create_dnn_model()
  else:
    model = create_linear_model()
  # Trains model
  model.fit(train_ds, epochs=epochs, validation_data=test_ds)
  # Plot results
  plot_results(model, all_ds)
