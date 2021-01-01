#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import patches
import coco_localization_tfrecords_generator as coco
import dogs_localization as dogs
import matplotlib
import pathlib
import os


# Root folder to find datasets
DATASETS = pathlib.Path(os.environ['DATASETS'])
# Root folder to place outputs from this script
OUTPUTS = pathlib.Path(os.environ['OUTPUTS'])
# Path to Resnet50 weights
RESNET50 = pathlib.Path(os.environ['WEIGHTS']) / 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'


# Creates file storing options passed to the command line when running script
def create_commandline_options_log(path: pathlib.Path, arguments: dict) -> None:
  options = path / 'options.txt'
  if not options.parent.exists():
    options.parent.mkdir(parents=True)
  with open(options, 'w+') as file:
      file.write(str(arguments) + '\n')


def explore_results(model: tf.keras.Model, dataset: tf.data.Dataset, localization: bool, transferred: bool):
  # Building axes
  rows = 5
  cols = 6
  fig, axes = plt.subplots(rows, cols)
  fig.set_tight_layout(tight=0.1)
  axes = axes.ravel()
  # Performs pre-processing here to keep the original unprocessed image as well
  preprocess_fnc = dogs.transferred_preprocess if transferred else dogs.preprocess
  # Gets some examples
  for i, (image, expected) in dataset.take(rows * cols).enumerate():
    predicted = tf.squeeze(model(tf.expand_dims(preprocess_fnc(image), axis=0)))
    # Plots corresponding image
    image_height, image_width, _ = image.shape
    axes[i].imshow(image.numpy())
    if localization and tf.equal(expected[0], 1):
      # Plots estimated bounding box
      box = coco.BBox(*predicted[1:])
      patch = patches.Rectangle((box.x * image_width, box.y * image_height), box.width * image_width,
                                box.height * image_height,
                                edgecolor='blue',
                                facecolor='none',
                                lw=2)
      axes[i].add_patch(patch)
      # Plots expected bounding box
      box = coco.BBox(*expected[1:])
      patch = patches.Rectangle((box.x * image_width, box.y * image_height), box.width * image_width,
                                box.height * image_height,
                                edgecolor='red',
                                facecolor='none',
                                lw=2)
      axes[i].add_patch(patch)
    # Some extra configurations
    axes[i].axis('off')
    axes[i].set_title("{:.2f}".format(predicted.numpy()))
  plt.show()


def explore_localization_error(model: tf.keras.Model, dataset: tf.data.Dataset, transferred: bool):
  # Building axes
  rows = 5
  cols = 6
  fig, axes = plt.subplots(rows, cols)
  fig.set_tight_layout(tight=0.1)
  axes = axes.ravel()
  # Performs pre-processing here to keep the original unprocessed image as well
  dataset = dataset.filter(lambda image, expected: tf.equal(expected[0], 1))
  preprocess_fnc = dogs.transferred_preprocess if transferred else dogs.preprocess
  # Find the first n images for which the IoU with the GT box is less than 10%
  found = 0
  iterator = iter(dataset)
  while found < rows * cols:
    image, expected = next(iterator)
    predicted = tf.squeeze(model(tf.expand_dims(preprocess_fnc(image), axis=0)))
    iou = tf.squeeze(dogs.compute_iou(tf.expand_dims(expected[1:], axis=0),
                                      tf.expand_dims(predicted[1:], axis=0)))
    if tf.equal(iou, 0):
      # Plots corresponding image
      image_height, image_width, _ = image.shape
      axes[found].imshow(image.numpy())
      # Plots estimated bounding box
      box = coco.BBox(*predicted[1:])
      patch = patches.Rectangle((box.x * image_width, box.y * image_height), box.width * image_width,
                                box.height * image_height,
                                edgecolor='blue',
                                facecolor='none',
                                lw=2)
      axes[found].add_patch(patch)
      # Plots expected bounding box
      box = coco.BBox(*expected[1:])
      patch = patches.Rectangle((box.x * image_width, box.y * image_height), box.width * image_width,
                                box.height * image_height,
                                edgecolor='red',
                                facecolor='none',
                                lw=2)
      axes[found].add_patch(patch)
      # Some extra configurations
      axes[found].axis('off')
      axes[found].set_title("{:.2f}".format(predicted[0].numpy()))
      # Updates count and exits if all needed images have been found
      found += 1
  plt.show()


if __name__ == '__main__':
  # Allowing memory growth
  physical_devices = tf.config.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  # Defines arguments parser
  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint', help='Path to model to load', type=str)
  parser.add_argument('--localization', help='Locations instead of just classification', action='store_true')
  parser.add_argument('--transferred', help='Use transferred learning', action='store_true')
  parser.add_argument('--error', help='Perform error analysis', action='store_true')
  parser.add_argument('--explore', help='Perform a simple exploration of the model', action='store_true')
  parser.add_argument('--samples', help='Max # of samples to keep in the dataset (Default: 1k)', type=int)
  # Parses arguments
  print(matplotlib.get_backend())
  args = parser.parse_args()
  checkpoint_path = dogs.OUTPUTS / args.checkpoint
  localization = True if args.localization else False
  transferred = True if args.transferred else False
  error = True if args.error else False
  explore = True if args.explore else False
  samples = args.samples if args.samples else 1000
  # Loading model
  model, _, _, _ = dogs.create_model(transferred=transferred, localization=localization)
  # Loads weights from checkpoint
  checkpoint = tf.train.Checkpoint(model=model)
  manager = tf.train.CheckpointManager(checkpoint, directory=str(checkpoint_path), max_to_keep=1)
  checkpoint.restore(manager.latest_checkpoint).expect_partial()
  if not manager.latest_checkpoint:
    print("No checkpoint found. Exiting!")
    exit()
  # Creates dataset
  if localization:
    _, train_unprocessed_ds = dogs.create_train_localization_dataset(samples, transferred=transferred)
    _, test_unprocessed_ds = dogs.create_test_localization_dataset(samples, transferred=transferred)
  else:
    _, train_unprocessed_ds = dogs.create_train_classification_dataset(samples, transferred=transferred)
    _, test_unprocessed_ds = dogs.create_test_classification_dataset(samples, transferred=transferred)
  # Explore performance if requested
  if explore:
    explore_results(model, train_unprocessed_ds, localization=localization, transferred=transferred)
    explore_results(model, test_unprocessed_ds, localization=localization, transferred=transferred)
  # Explores images with very high localization error
  if error and localization:
    explore_localization_error(model, train_unprocessed_ds, transferred=transferred)
    explore_localization_error(model, test_unprocessed_ds, transferred=transferred)

  # dogs = tf.constant(0, dtype=tf.float32)
  # others = tf.constant(0, dtype=tf.float32)
  # for batch in train_ds:
  #   inputs, outputs = batch
  #   dogs += tf.reduce_sum(tf.cast(tf.equal(outputs[:, 0], 1), dtype=tf.float32))
  #   others += tf.reduce_sum(tf.cast(tf.equal(outputs[:, 0], 0), dtype=tf.float32))
  # print("Number of dogs: ", dogs.numpy())
  # print("Number of others: ", others.numpy())
