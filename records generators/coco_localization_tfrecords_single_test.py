#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import argparse
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import patches
import coco_localization_tfrecords_generator as loc

# Constants
BUFFER_SIZE = 1000


# Parses examples and returns image, label and bbox
def parse_localization_example(entry: tf.Tensor):
  example = tf.io.parse_single_example(entry, features=loc.outputs_description)
  image = tf.io.decode_image(example['image'], dtype=tf.uint8, channels=3)
  label = example['label']
  bbox = tf.sparse.to_dense(example['bbox'])
  return image, label, bbox


if __name__ == '__main__':
  # Defines arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('folder', help='Folder with raw tfrecords', type=str)
  # Parses arguments
  args = parser.parse_args()
  folder = pathlib.Path(args.folder)
  # Paths of all tfrecord files
  paths = [str(file) for file in folder.glob("*.tfrecord*")]
  # Creates base dataset from tfrecords
  dataset = tf.data.TFRecordDataset(paths).map(parse_localization_example,
                                               num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dogs_ds = dataset.filter(lambda image, label, bbox: tf.equal(label, 'dog')).shuffle(BUFFER_SIZE)
  other_ds = dataset.filter(lambda image, label, bbox: tf.logical_not(tf.equal(label, 'dog'))).shuffle(BUFFER_SIZE)
  mixed_ds = dogs_ds.take(5).concatenate(other_ds.take(4))
  # Plots nine images
  fig, axes = plt.subplots(3, 3)
  fig.set_tight_layout(tight=0.1)
  axes = axes.ravel()
  # Reads info
  for i, (image, label, bbox) in mixed_ds.enumerate():
    # Plots corresponding image
    image_height, image_width, image_depth = image.shape
    axes[i].imshow(image.numpy())
    # Plots bounding box
    if tf.equal(label, 'dog'):
      box = loc.BBox(*bbox.numpy())
      patch = patches.Rectangle((box.x * image_width, box.y * image_height), box.width * image_width,
                                box.height * image_height,
                                edgecolor='red',
                                facecolor='none',
                                alpha=0.3,
                                lw=2)
      axes[i].add_patch(patch)
    axes[i].axis('off')
    axes[i].set_title(label.numpy())
  plt.show()
