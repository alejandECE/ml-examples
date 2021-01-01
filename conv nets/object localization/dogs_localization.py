#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import datetime
import pathlib
from typing import Tuple
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import patches
import argparse
import stanford_dogs_localization_tfrecords_generator as stanford
import coco_localization_tfrecords_generator as coco
import utils

# Other constants
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = BATCH_SIZE * 32
IMG_HEIGHT = 224
IMG_WIDTH = 224


# Determines localization loss (binary cross entropy with first element and mse with the rest)
class LocalizationLoss(tf.keras.losses.Loss):
  def __init__(self, similarity='huber', **kwargs):
    super().__init__(**kwargs)
    self.classification_loss = tf.keras.losses.binary_crossentropy
    if similarity == 'l2':
      self.regression_loss = tf.keras.losses.MSE
    elif similarity == 'l1':
      self.regression_loss = tf.keras.losses.MAE
    else:
      self.regression_loss = tf.keras.losses.huber

  def call(self, y_true, y_pred):
    # Classification loss
    loss = self.classification_loss(
      tf.expand_dims(y_true[:, 0], axis=-1),
      tf.expand_dims(y_pred[:, 0], axis=-1)
    )
    # Localization loss (only applicable to localizable observations, i.e., from the positive class)
    mask = tf.cast(tf.equal(y_true[:, 0], 1), dtype=loss.dtype)
    loss += tf.multiply(self.regression_loss(y_true[:, 1:], y_pred[:, 1:]), mask)
    return loss


# Computes IoU
@tf.function
def compute_iou(expected_boxes: tf.Tensor, predicted_boxes: tf.Tensor) -> tf.Tensor:
    # Unwraps components from expected boxes
    expected_x = expected_boxes[:, 0]
    expected_y = expected_boxes[:, 1]
    expected_width = expected_boxes[:, 2]
    expected_height = expected_boxes[:, 3]
    # Unwraps components from predicted boxes
    predicted_x = predicted_boxes[:, 0]
    predicted_y = predicted_boxes[:, 1]
    predicted_width = predicted_boxes[:, 2]
    predicted_height = predicted_boxes[:, 3]
    # Determines intersection area
    dy = tf.minimum(expected_y + expected_height, predicted_y + predicted_height) - tf.maximum(expected_y, predicted_y)
    dy = tf.multiply(dy, tf.cast(tf.greater(dy, 0), dtype=dy.dtype))
    dx = tf.minimum(expected_x + expected_width, predicted_x + predicted_width) - tf.maximum(expected_x, predicted_x)
    dx = tf.multiply(dx, tf.cast(tf.greater(dx, 0), dtype=dx.dtype))
    intersection = tf.multiply(dx, dy)
    # Determines union area
    union = tf.multiply(expected_width, expected_height) + tf.multiply(predicted_width, predicted_height) - intersection
    return tf.divide(intersection, union)


# Average IoU
class AverageIoU(tf.keras.metrics.Metric):
  def __init__(self, epsilon=1e-10, **kwargs):
    super().__init__(name='AverageIoU', **kwargs)
    self.cumulative = self.add_weight('cumulative', initializer='zeros')
    self.count = self.add_weight('count', initializer='zeros')
    self.epsilon = epsilon

  def update_state(self, y_true, y_pred, **kwargs):
    # Only localizable observations (positive class) are taken into account for this metric)
    mask = y_true[:, 0]
    # Updates the element count (only localizable observations)
    self.count.assign_add(tf.reduce_sum(mask))
    # Computes IoU
    mask = tf.cast(mask, dtype=tf.bool)
    iou = compute_iou(
      tf.boolean_mask(y_true[:, 1:], mask),
      tf.boolean_mask(y_pred[:, 1:], mask)
    )
    # Updates intersection over union cumulative
    self.cumulative.assign_add(tf.reduce_sum(iou))

  def reset_states(self):
    self.cumulative.assign(0)
    self.count.assign(self.epsilon)

  def result(self):
    return self.cumulative / self.count


# Accuracy (IoU > threshold)
class AccuracyIoU(tf.keras.metrics.Metric):
  def __init__(self, threshold=0.5, epsilon=1e-10, **kwargs):
    super().__init__(name='AccuracyIoU', **kwargs)
    self.correct = self.add_weight('correct', initializer='zeros')
    self.count = self.add_weight('count', initializer='zeros')
    self.threshold = threshold
    self.epsilon = epsilon

  def update_state(self, y_true, y_pred, **kwargs):
    # Only localizable observations (positive class)
    mask = y_true[:, 0]
    # Updates the element count (only localizable observations)
    self.count.assign_add(tf.reduce_sum(mask))
    # Computes IoU
    mask = tf.cast(mask, dtype=tf.bool)
    iou = compute_iou(
      tf.boolean_mask(y_true[:, 1:], mask),
      tf.boolean_mask(y_pred[:, 1:], mask)
    )
    # Updates correct IoU count
    self.correct.assign_add(tf.reduce_sum(tf.cast(tf.greater_equal(iou, self.threshold), dtype=tf.float32)))

  def reset_states(self):
    self.correct.assign(0)
    self.count.assign(self.epsilon)

  def result(self):
    return self.correct / self.count


# Parses proto example and converts it to (img, label, bbox)
def parse_for_localization(example: tf.Tensor) -> Tuple:
  observation = tf.io.parse_single_example(example, stanford.outputs_description)
  image = tf.io.decode_image(observation['image'], channels=3, expand_animations=False)
  label = observation['label']
  bbox = tf.sparse.to_dense(observation['bbox'])
  return image, label, bbox


# Parses proto example and converts into correct format for training (img, label)
def parse_for_classification(example: tf.Tensor) -> Tuple:
  observation = tf.io.parse_single_example(example, stanford.outputs_description)
  image = tf.io.decode_image(observation['image'], channels=3, expand_animations=False)
  label = observation['label']
  return image, label


# Basic pre-processing
def preprocess(img: tf.Tensor) -> Tuple:
  # It only normalizes from [0, MAX) if the input arg is not a tf.float32 already
  img = tf.image.convert_image_dtype(img, dtype=tf.float32)
  # Resizes to the network input specs
  img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], antialias=True)
  return img


# Pre-processing according to the transferred model
def transferred_preprocess(img: tf.Tensor) -> tf.Tensor:
  # Resizes to the network input specs
  img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], antialias=True)
  # Preprocess data according to transferred model
  img = tf.keras.applications.resnet50.preprocess_input(img)
  return img


# Shows nine images from dataset with their corresponding bounding boxes and label
def display_location_dataset(dataset: tf.data.Dataset):
  # Plots nine images
  fig, axes = plt.subplots(3, 3)
  fig.set_tight_layout(tight=0.1)
  axes = axes.ravel()
  for i, entry in dataset.take(9).enumerate():
    # Reads info
    image, output = entry
    label = output[0]
    bbox = output[1:]
    image_height, image_width, _ = image.shape
    box = coco.BBox(*bbox)
    # Plots corresponding image
    axes[i].imshow(image.numpy())
    patch = patches.Rectangle((box.x * image_width, box.y * image_height), box.width * image_width,
                              box.height * image_height,
                              edgecolor='red',
                              facecolor='none',
                              lw=2)
    axes[i].add_patch(patch)
    axes[i].axis('off')
    axes[i].set_title(label.numpy())
  plt.show()


# Shows nine images from dataset with their corresponding label
def display_classification_dataset(dataset: tf.data.Dataset):
  # Plots nine images
  fig, axes = plt.subplots(3, 3)
  fig.set_tight_layout(tight=0.1)
  axes = axes.ravel()
  for i, entry in dataset.take(9).enumerate():
    # Reads info
    image, label = entry
    image_height, image_width, _ = image.shape
    # Plots corresponding image
    axes[i].imshow(image.numpy())
    axes[i].axis('off')
    axes[i].set_title(label.numpy())
  plt.show()


# Creates training dataset to perform classification only
def create_train_classification_dataset(observations: int, transferred=False, display=False) -> Tuple:
  # Loads dogs examples
  dogs_path = utils.DATASETS / 'stanford_dogs/localization_records/train_list.mat/'
  dogs_ds = tf.data.Dataset.from_tensor_slices([str(file) for file in dogs_path.glob("*.tfrecord*")])
  dogs_ds = dogs_ds.interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTOTUNE, deterministic=True)
  dogs_ds = dogs_ds.map(parse_for_classification, num_parallel_calls=AUTOTUNE)
  # Loads others examples
  others_path = utils.DATASETS / 'coco/localization_records/train2014/'
  others_ds = tf.data.Dataset.from_tensor_slices([str(file) for file in others_path.glob('*.tfrecord*')])
  others_ds = others_ds.interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTOTUNE, deterministic=True)
  others_ds = others_ds.map(parse_for_classification, num_parallel_calls=AUTOTUNE)
  # Extracts dogs from this dataset
  extra_dogs_ds = others_ds.filter(lambda img, label: tf.equal(label, b'dog'))
  # Excludes dogs from the others dataset
  others_ds = others_ds.filter(lambda img, label: tf.not_equal(label, b'dog'))
  # Puts together all dogs
  dogs_ds = extra_dogs_ds.concatenate(dogs_ds)
  # Changes dogs labels for classification and only takes as many observations as specified
  dogs_ds = dogs_ds.take(observations)
  dogs_ds = dogs_ds.map(lambda img, label: (img, 1), num_parallel_calls=AUTOTUNE)
  # Changes others labels for classification and only take as many observations as specified
  others_ds = others_ds.take(observations)
  others_ds = others_ds.map(lambda img, label: (img, 0), num_parallel_calls=AUTOTUNE)
  # Combines both datasets
  choice_ds = tf.data.Dataset.range(2).repeat(observations)
  combined_ds = tf.data.experimental.choose_from_datasets([dogs_ds, others_ds], choice_ds)
  # Applies pre-processing
  if transferred:
    combined_ds = combined_ds.map(lambda inputs, outputs: (transferred_preprocess(inputs), outputs),
                                  num_parallel_calls=AUTOTUNE)
  else:
    combined_ds = combined_ds.map(lambda inputs, outputs: (preprocess(inputs), outputs),
                                  num_parallel_calls=AUTOTUNE)
  # Display some random observations if requested
  unprocessed_ds = tf.data.experimental.sample_from_datasets([
    dogs_ds.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False),
    others_ds.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
  ])
  if display:
    display_classification_dataset(unprocessed_ds)
  # Shuffles prior to batch (to obtain new batches every epoch)
  combined_ds = combined_ds.shuffle(buffer_size=BUFFER_SIZE)
  combined_ds = combined_ds.batch(BATCH_SIZE)
  # Optimizes pre-loading some batches
  processed_ds = combined_ds.prefetch(buffer_size=AUTOTUNE)
  return processed_ds, unprocessed_ds


# Creates testing dataset to perform classification only
def create_test_classification_dataset(observations: int, transferred=False, display=False) -> Tuple:
  # Loads dogs examples
  dogs_path = utils.DATASETS / 'stanford_dogs/localization_records/test_list.mat/'
  dogs_ds = tf.data.Dataset.from_tensor_slices([str(file) for file in dogs_path.glob("*.tfrecord*")])
  dogs_ds = dogs_ds.interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTOTUNE, deterministic=True)
  dogs_ds = dogs_ds.map(parse_for_classification, num_parallel_calls=AUTOTUNE)
  # Loads others examples
  others_path = utils.DATASETS / 'coco/localization_records/val2014/'
  others_ds = tf.data.Dataset.from_tensor_slices([str(file) for file in others_path.glob('*.tfrecord*')])
  others_ds = others_ds.interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTOTUNE, deterministic=True)
  others_ds = others_ds.map(parse_for_classification, num_parallel_calls=AUTOTUNE)
  # Extracts dogs from this dataset
  extra_dogs_ds = others_ds.filter(lambda img, label: tf.equal(label, b'dog'))
  # Excludes dogs from the others dataset
  others_ds = others_ds.filter(lambda img, label: tf.not_equal(label, b'dog'))
  # Puts together all dogs
  dogs_ds = extra_dogs_ds.concatenate(dogs_ds)
  # Changes dogs labels for classification and only takes as many observations as specified
  dogs_ds = dogs_ds.take(observations)
  dogs_ds = dogs_ds.map(lambda img, label: (img, 1), num_parallel_calls=AUTOTUNE)
  # Changes others labels for classification and only take as many observations as specified
  others_ds = others_ds.take(observations)
  others_ds = others_ds.map(lambda img, label: (img, 0), num_parallel_calls=AUTOTUNE)
  # Combines both datasets
  combined_ds = dogs_ds.concatenate(others_ds)
  # Applies pre-processing
  if transferred:
    combined_ds = combined_ds.map(lambda inputs, outputs: (transferred_preprocess(inputs), outputs),
                                  num_parallel_calls=AUTOTUNE)
  else:
    combined_ds = combined_ds.map(lambda inputs, outputs: (preprocess(inputs), outputs),
                                  num_parallel_calls=AUTOTUNE)
  # Display some random observations if requested
  unprocessed_ds = tf.data.experimental.sample_from_datasets([
      dogs_ds.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False),
      others_ds.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
    ])
  if display:
    display_classification_dataset(unprocessed_ds)
  # Batches
  combined_ds = combined_ds.batch(BATCH_SIZE)
  # Optimizes pre-loading some batches
  processed_ds = combined_ds.prefetch(buffer_size=AUTOTUNE)
  return processed_ds, unprocessed_ds


# Creates training dataset to perform classification and localization
def create_train_localization_dataset(observations: int, transferred=False, display=False) -> Tuple:
  # Loads dogs examples
  dogs_path = utils.DATASETS / 'stanford_dogs/localization_records/train_list.mat/'
  dogs_ds = tf.data.Dataset.from_tensor_slices([str(file) for file in dogs_path.glob("*.tfrecord*")])
  dogs_ds = dogs_ds.interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTOTUNE, deterministic=True)
  dogs_ds = dogs_ds.map(parse_for_localization, num_parallel_calls=AUTOTUNE)
  # Loads others examples
  others_path = utils.DATASETS / 'coco/localization_records/train2014/'
  others_ds = tf.data.Dataset.from_tensor_slices([str(file) for file in others_path.glob('*.tfrecord*')])
  others_ds = others_ds.interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTOTUNE, deterministic=True)
  others_ds = others_ds.map(parse_for_localization, num_parallel_calls=AUTOTUNE)
  # Extracts dogs from this dataset
  extra_dogs_ds = others_ds.filter(lambda img, label, bbox: tf.equal(label, 'dog'))
  # Excludes dogs from the others dataset
  others_ds = others_ds.filter(lambda img, label, bbox: tf.not_equal(label, 'dog'))
  # Puts together all dogs
  dogs_ds = extra_dogs_ds.concatenate(dogs_ds)
  # Changes dogs labels for localization and only takes as many observations as specified
  dogs_ds = dogs_ds.take(observations)
  dogs_ds = dogs_ds.map(lambda img, label, bbox:
                        (img, tf.concat([tf.constant(1, shape=(1,), dtype=bbox.dtype), bbox], axis=0)),
                        num_parallel_calls=AUTOTUNE)

  # Changes others labels for localization and only take as many observations as specified
  others_ds = others_ds.take(observations)
  others_ds = others_ds.map(lambda img, label, bbox:
                            (img, tf.concat([tf.constant(0, shape=(1,), dtype=bbox.dtype), bbox], axis=0)),
                            num_parallel_calls=AUTOTUNE)
  # Combines both datasets
  choice_ds = tf.data.Dataset.range(2).repeat(observations)
  combined_ds = tf.data.experimental.choose_from_datasets([dogs_ds, others_ds], choice_ds)
  # Applies pre-processing
  if transferred:
    combined_ds = combined_ds.map(lambda inputs, outputs: (transferred_preprocess(inputs), outputs),
                                  num_parallel_calls=AUTOTUNE)
  else:
    combined_ds = combined_ds.map(lambda inputs, outputs: (preprocess(inputs), outputs),
                                  num_parallel_calls=AUTOTUNE)
  # Display some random observations if requested
  unprocessed_ds = tf.data.experimental.sample_from_datasets([
      dogs_ds.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False),
      others_ds.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
    ])
  if display:
    display_location_dataset(unprocessed_ds)
  # Shuffles prior to batch (to obtain new batches every epoch)
  combined_ds = combined_ds.shuffle(buffer_size=BUFFER_SIZE)
  combined_ds = combined_ds.batch(BATCH_SIZE)
  # Optimizes pre-loading some batches
  processed_ds = combined_ds.prefetch(buffer_size=AUTOTUNE)
  return processed_ds, unprocessed_ds


# Creates testing dataset to perform classification and localization
def create_test_localization_dataset(observations: int, transferred=False, display=False) -> Tuple:
  # Loads dogs examples
  dogs_path = utils.DATASETS / 'stanford_dogs/localization_records/test_list.mat/'
  dogs_ds = tf.data.Dataset.from_tensor_slices([str(file) for file in dogs_path.glob("*.tfrecord*")])
  dogs_ds = dogs_ds.interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTOTUNE, deterministic=True)
  dogs_ds = dogs_ds.map(parse_for_localization, num_parallel_calls=AUTOTUNE)
  # Loads others examples
  others_path = utils.DATASETS / 'coco/localization_records/val2014/'
  others_ds = tf.data.Dataset.from_tensor_slices([str(file) for file in others_path.glob('*.tfrecord*')])
  others_ds = others_ds.interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTOTUNE, deterministic=True)
  others_ds = others_ds.map(parse_for_localization, num_parallel_calls=AUTOTUNE)
  # Extracts dogs from this dataset
  extra_dogs_ds = others_ds.filter(lambda img, label, bbox: tf.equal(label, 'dog'))
  # Excludes dogs from the others dataset
  others_ds = others_ds.filter(lambda img, label, bbox: tf.not_equal(label, 'dog'))
  # Puts together all dogs
  dogs_ds = extra_dogs_ds.concatenate(dogs_ds)
  # Changes dogs labels for localization
  dogs_ds = dogs_ds.take(observations)
  dogs_ds = dogs_ds.map(lambda img, label, bbox:
                        (img, tf.concat([tf.constant(1, shape=(1,), dtype=bbox.dtype), bbox], axis=0)),
                        num_parallel_calls=AUTOTUNE)
  # Changes others labels for localization
  others_ds = others_ds.take(observations)
  others_ds = others_ds.map(lambda img, label, bbox:
                            (img, tf.concat([tf.constant(0, shape=(1,), dtype=bbox.dtype), bbox], axis=0)),
                            num_parallel_calls=AUTOTUNE)
  # Combines both datasets
  combined_ds = dogs_ds.concatenate(others_ds)
  # Applies pre-processing
  if transferred:
    combined_ds = combined_ds.map(lambda inputs, outputs: (transferred_preprocess(inputs), outputs),
                                  num_parallel_calls=AUTOTUNE)
  else:
    combined_ds = combined_ds.map(lambda inputs, outputs: (preprocess(inputs), outputs),
                                  num_parallel_calls=AUTOTUNE)
  # Display some random observations if requested
  unprocessed_ds = tf.data.experimental.sample_from_datasets([
      dogs_ds.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False),
      others_ds.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
    ])
  if display:
    display_location_dataset(unprocessed_ds)
  # Batches
  combined_ds = combined_ds.batch(BATCH_SIZE)
  # Optimizes pre-loading some batches
  processed_ds = combined_ds.prefetch(buffer_size=AUTOTUNE)
  return processed_ds, unprocessed_ds


# Creates basic classification model (not too deep)
def create_classification_model(diagram_path: pathlib.Path = None) -> Tuple:
  mdl = tf.keras.Sequential([
    tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3]),
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'),
    tf.keras.layers.GlobalMaxPool2D(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])
  # Displays summary
  mdl.summary()
  # Stores image
  if diagram_path is not None:
    diagram_path.mkdir(parents=True)
    tf.keras.utils.plot_model(mdl, to_file=str(diagram_path / 'classification_model.png'),
                              expand_nested=True, show_shapes=True, show_layer_names=False)
  # Defines optimizer, loss and metric
  optimizer = tf.keras.optimizers.Adam()
  loss_fn = tf.keras.losses.BinaryCrossentropy()
  metric_fn = tf.keras.metrics.BinaryAccuracy()
  return mdl, optimizer, loss_fn, metric_fn


# Creates basic localization model (not too deep)
def create_localization_model(diagram_path: pathlib.Path = None) -> Tuple:
  # Creating base model
  base_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'),
    tf.keras.layers.GlobalMaxPool2D(),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu')
  ])
  # Defining extra layers
  inputs = tf.keras.layers.Input([IMG_HEIGHT, IMG_WIDTH, 3])
  dense1 = tf.keras.layers.Dense(1, activation='sigmoid')
  dense2 = tf.keras.layers.Dense(4, activation='linear')
  concat = tf.keras.layers.Concatenate()
  # Creating connections for the whole model
  x = base_model(inputs)
  class_output = dense1(x)
  location_output = dense2(x)
  concatenated = concat([class_output, location_output])
  # Creates model
  mdl = tf.keras.Model(inputs, concatenated)
  # Displays summary
  mdl.summary()
  # Stores image
  if diagram_path is not None:
    diagram_path.mkdir(parents=True)
    tf.keras.utils.plot_model(mdl, to_file=str(diagram_path / 'localization_model.png'),
                              expand_nested=True, show_shapes=True, show_layer_names=False)
  # Defines optimizer, loss and metric
  optimizer = tf.keras.optimizers.Adam()
  loss_fn = LocalizationLoss(similarity='huber')
  metric_fn = AverageIoU()
  return mdl, optimizer, loss_fn, metric_fn


# Creates transferred classification model
def create_transferred_classification_model(diagram_path: pathlib.Path = None, unfreeze=0) -> Tuple:
  # Loading pre-trained base model
  base_model = tf.keras.applications.resnet50.ResNet50(weights=str(utils.RESNET50), include_top=False)
  # Avoid changing loaded weights of some (if not all) layers
  layer_count = len(base_model.layers)
  for index in range(layer_count - unfreeze):
    base_model.layers[index].trainable = False
  # Defining extra layers
  inputs = tf.keras.layers.Input([IMG_HEIGHT, IMG_WIDTH, 3])
  avg = tf.keras.layers.GlobalAveragePooling2D()
  dense1 = tf.keras.layers.Dense(128, activation='relu')
  dense2 = tf.keras.layers.Dense(1, activation='sigmoid')
  # Creating connections for the whole model
  x = base_model(inputs)
  x = avg(x)
  x = dense1(x)
  outputs = dense2(x)
  # Creates model
  mdl = tf.keras.Model(inputs, outputs)
  # Displays summary
  mdl.summary()
  # Stores image
  if diagram_path is not None:
    diagram_path.mkdir(parents=True)
    tf.keras.utils.plot_model(mdl, to_file=str(diagram_path / 'transferred_classification_model.png'),
                              expand_nested=True, show_shapes=True, show_layer_names=False)
  # Defines optimizer, loss and metric
  optimizer = tf.keras.optimizers.Adam()
  loss_fn = tf.keras.losses.BinaryCrossentropy()
  metric_fn = tf.keras.metrics.BinaryAccuracy()
  return mdl, optimizer, loss_fn, metric_fn


# Creates transferred localization model
def create_transferred_localization_model(diagram_path: pathlib.Path = None, unfreeze=0) -> Tuple:
  base_model = tf.keras.applications.resnet50.ResNet50(weights=str(utils.RESNET50), include_top=False)
  # Avoid changing loaded weights of some (if not all) layers
  layer_count = len(base_model.layers)
  for index in range(layer_count - unfreeze):
    base_model.layers[index].trainable = False
  # Defining extra layers
  inputs = tf.keras.layers.Input([IMG_HEIGHT, IMG_WIDTH, 3])
  avg = tf.keras.layers.GlobalAveragePooling2D()
  concat = tf.keras.layers.Concatenate()
  # Creating connections for the whole model
  x = base_model(inputs)
  x = avg(x)
  units = [200, 100, 50]
  # # Creating classifier
  # classifier_output = x
  # for entry in units:
  #   classifier_output = tf.keras.layers.Dense(entry, activation='relu')(classifier_output)
  # classifier_output = tf.keras.layers.Dense(1, activation='sigmoid')(classifier_output)
  # # Creating localizer
  # localizer_output = x
  # for entry in units:
  #   localizer_output = tf.keras.layers.Dense(entry, activation='relu')(localizer_output)
  # localizer_output = tf.keras.layers.Dense(4, activation='linear')(localizer_output)
  # # Combines both outputs into one simple output
  for entry in units:
    x = tf.keras.layers.Dense(entry, activation='relu')(x)
  classifier_output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
  localizer_output = tf.keras.layers.Dense(4, activation='linear')(x)
  concatenated = concat([classifier_output, localizer_output])
  # Creates model
  mdl = tf.keras.Model(inputs, concatenated)
  # Displays summary
  mdl.summary()
  # Stores image
  if diagram_path is not None:
    diagram_path.mkdir(parents=True)
    tf.keras.utils.plot_model(mdl, to_file=str(diagram_path / 'transferred_localization_model.png'),
                              expand_nested=True, show_shapes=True, show_layer_names=False)
  # Defines optimizer, loss and metric
  optimizer = tf.keras.optimizers.Adam(lr=1e-2)
  loss_fn = LocalizationLoss(similarity='l2')
  metric_fn = AverageIoU()
  return mdl, optimizer, loss_fn, metric_fn


# Creates a model based on selected options
def create_model(timestamp: str = None, transferred: bool = False,
                 unfreeze: int = 0, localization: bool = True) -> tf.keras.Model:
  diagram_path = utils.OUTPUTS / timestamp / 'diagram' if timestamp else None
  if transferred:
    if localization:
      mdl = create_transferred_localization_model(diagram_path, unfreeze)
    else:
      mdl = create_transferred_classification_model(diagram_path, unfreeze)
  else:
    if localization:
      mdl = create_localization_model(diagram_path)
    else:
      mdl = create_classification_model(diagram_path)
  return mdl


# Simple test step
@tf.function
def test_step(model, batch, loss_fn, metric_fn):
  # Unwrapped inputs and expected output
  inputs, y_true = batch
  # Computes output of the model
  y_pred = model(inputs)
  # Computes loss
  loss = loss_fn(y_true, y_pred)
  # Computes metric
  metric = metric_fn(y_true, y_pred)
  # Returns loss and metric
  return loss, metric


# Simple train step
@tf.function
def train_step(model, batch, optimizer, loss_fn, metric_fn):
  # Unwrapped inputs and expected output
  inputs, y_true = batch
  with tf.GradientTape() as tape:
    # Computes output of the model
    y_pred = model(inputs)
    # Computes loss
    loss = loss_fn(y_true, y_pred)
  # Determines gradients
  variables = model.trainable_variables
  gradients = tape.gradient(loss, variables)
  # Applies optimizer step
  optimizer.apply_gradients(zip(gradients, variables))
  # Computes metric (calls updated_state followed by results)
  metric = metric_fn(y_true, y_pred)
  # Returns loss and metric
  return loss, metric


# Entry point to training
def train_and_checkpoint(parser):
  # Generates for later use
  timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  # Parses arguments
  args = parser.parse_args()
  epochs = args.epochs if args.epochs else 2
  samples = args.samples if args.samples else 1000
  save = False if args.unsaved else True
  display = True if args.display else False
  localization = True if args.localization else False
  transferred = True if args.transferred else False
  unfreeze = args.unfreeze if args.unfreeze else 0
  ckpt_load_path = utils.OUTPUTS / args.checkpoint if args.checkpoint else None
  # Logs command line options
  utils.create_commandline_options_log(
    utils.OUTPUTS / timestamp,
    {
      'Localization': localization,
      'Epochs': epochs,
      'Samples': samples,
      'Transferred': transferred,
      'Unfreeze': unfreeze
    }
  )
  # Creates model to train
  model, optimizer, loss_fn, metric_fn = create_model(
    timestamp if save else None,
    transferred=transferred,
    unfreeze=unfreeze,
    localization=localization
  )
  # Creates dataset
  if localization:
    train_ds, train_unprocessed_ds = create_train_localization_dataset(samples, transferred=transferred, display=display)
    test_ds, test_unprocessed_ds = create_test_localization_dataset(samples, transferred=transferred, display=display)
  else:
    train_ds, train_unprocessed_ds = create_train_classification_dataset(samples, transferred=transferred, display=display)
    test_ds, test_unprocessed_ds = create_test_classification_dataset(samples, transferred=transferred, display=display)
  # Restores model and optimizer states from path specified
  if ckpt_load_path:
    # Checkpoint specifying what info to track
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    # Creates manager to load last checkpoint from folder (otherwise need to determine the last one manually! yikes)
    manager = tf.train.CheckpointManager(ckpt, directory=str(ckpt_load_path), max_to_keep=1)
    # Restores the checkpoint
    ckpt.restore(manager.latest_checkpoint)
    if not manager.latest_checkpoint:
     print("No checkpoint found!")
  # Creates a new checkpoint folder and manager to avoid overwriting old ones
  ckpt_save_path = utils.OUTPUTS / timestamp / 'checkpoints'
  ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
  manager = tf.train.CheckpointManager(ckpt, ckpt_save_path, max_to_keep=2)
  # Training loop
  highest_metric = None
  steps_per_epoch = 0
  for epoch in range(1, epochs + 1):
    found_better = False
    cumulative_loss = 0
    # Resets metric states (we want to computed per epoch metric)
    metric_fn.reset_states()
    # Step reset to 1
    step = 1
    for batch in train_ds:
      # Updates steps per epoch
      if step > steps_per_epoch:
        steps_per_epoch = step
      # Performs training step
      loss, metric = train_step(model, batch, optimizer, loss_fn, metric_fn)
      # Updates epoch training performance
      cumulative_loss += loss.numpy()
      avg_step_loss = cumulative_loss / step
      print(
        f'\rEpoch {epoch}/{epochs}, Step {step}/{steps_per_epoch}:'
        f' Train Loss: {avg_step_loss:.4f}, Train {metric_fn.name}: {metric:.4f}',
        end=''
      )
      step += 1
    # Verifies if a better model has been found
    if highest_metric is None or metric > highest_metric:
      found_better = True
      highest_metric = metric
    # Evaluates test set
    cumulative_loss = 0
    # Resets metric states (we want to computed per epoch metric)
    metric_fn.reset_states()
    # Step reset to 1
    step = 1
    for batch in test_ds:
      # Performs test step
      loss, metric = test_step(model, batch, loss_fn, metric_fn)
      # Updates epoch test performance
      cumulative_loss += loss.numpy()
      step += 1
    avg_step_loss = cumulative_loss / step
    print(f' - Test Loss: {avg_step_loss:.4f}, Test {metric_fn.name}: {metric:.4f}')
    # Saves checkpoint if requirements are met and if requested
    if save and found_better:
      manager.save()
  # Explores performance by visualizing examples (we pass the unprocessed ds to be able to recover the original img)
  ckpt.restore(manager.latest_checkpoint)
  utils.explore_results(model, train_unprocessed_ds, localization, transferred)
  utils.explore_results(model, test_unprocessed_ds, localization, transferred)


if __name__ == '__main__':
  # Allowing memory growth
  physical_devices = tf.config.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  # Defines arguments parser
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', help='Number of epochs to train (Default: 2)', type=int)
  parser.add_argument('--unsaved', help='Trained mode will not be saved on file', action='store_true')
  parser.add_argument('--samples', help='Max # of samples to keep in the dataset (Default: 1k)', type=int)
  parser.add_argument('--display', help='Display some examples from dataset', action='store_true')
  parser.add_argument('--localization', help='Locations instead of just classification', action='store_true')
  parser.add_argument('--transferred', help='Use transferred learning', action='store_true')
  parser.add_argument('--checkpoint', help='Path to model to load relative to the outputs root directory', type=str)
  parser.add_argument('--unfreeze', help='Update parameters from transferred layers', type=int)
  # Resumes or start training according to the options selected
  train_and_checkpoint(parser)

