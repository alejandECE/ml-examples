#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import argparse
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import patches
import stanford_dogs_raw_tfrecords_generator as standford

# Constants
BUFFER_SIZE = 1000


# Parses examples and returns image, image_id and bboxes
def parse_raw_example(entry: tf.Tensor):
  example = tf.io.parse_single_example(entry, features=standford.features_description)
  image = tf.io.decode_image(example['image'], dtype=tf.uint8)
  num_objects = example['objects']
  bboxes = tf.reshape(tf.sparse.to_dense(example['bbox']), shape=[num_objects, 4])
  path = example['path']
  return image, path, bboxes


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
  dataset = tf.data.TFRecordDataset(paths).map(parse_raw_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.shuffle(BUFFER_SIZE)
  # Plots nine images
  fig, axes = plt.subplots(3, 3)
  fig.set_tight_layout(tight=0.1)
  axes = axes.ravel()
  # Reads info
  for i, (image, path, bboxes) in dataset.take(9).enumerate():
    # Plots corresponding image
    image_height, image_width, image_depth = image.shape
    axes[i].imshow(image.numpy())
    # Plots bounding boxes
    for row in bboxes:
      box = standford.BBox(*row.numpy())
      patch = patches.Rectangle((box.x * image_width, box.y * image_height), box.width * image_width,
                                box.height * image_height,
                                edgecolor='red',
                                facecolor='none',
                                alpha=0.3,
                                lw=2)
      axes[i].add_patch(patch)
      axes[i].axis('off')
      axes[i].set_title(path.numpy().decode().split('/')[-1][:-4])
  plt.show()
