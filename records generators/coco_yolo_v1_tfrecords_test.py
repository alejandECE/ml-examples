#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import argparse
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import patches
import coco_yolo_v1_tfrecords_generator as yolo
import matplotlib.cm as cm
import matplotlib.lines as lines
import random

# Constants
BUFFER_SIZE = 10000


# Parses examples and returns image, label and bbox
def parse_yolo_example(entry: tf.Tensor):
  example = tf.io.parse_single_example(entry, features=yolo.outputs_description)
  image = tf.io.decode_image(example['image'], dtype=tf.uint8, channels=3)
  indices = tf.io.parse_tensor(example['indices'], out_type=tf.int64)
  values = tf.io.parse_tensor(example['values'], out_type=tf.float32)
  shape = tf.io.parse_tensor(example['shape'], out_type=tf.int64)
  output = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)
  return image, output


# Creates dictionary mapping indices to label
def load_categories(folder: pathlib.Path) -> dict:
  file = folder / 'categories.txt'
  index_to_category = {}
  with open(file, 'r') as f:
    line = f.readline()
    while line:
      result = yolo.EXTRACT_CATEGORY_INFO_REGEX.match(line)
      index = int(result.group(1))
      category = result.group(2)
      index_to_category[index] = category
      line = f.readline()
  return index_to_category


if __name__ == '__main__':
  # Defines arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('folder', help='Folder with localization tfrecords', type=str)
  # Parses arguments
  args = parser.parse_args()
  folder = pathlib.Path(args.folder)
  # Loading categories
  index_to_categories = load_categories(folder)
  # Paths of all tfrecord files
  paths = [str(file) for file in folder.glob("*.tfrecord*")]
  # Creates base dataset from tfrecords
  dataset = tf.data.TFRecordDataset(paths).map(parse_yolo_example,
                                               num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(BUFFER_SIZE)
  # Plots one image
  fig, axes = plt.subplots(1, 2)
  fig.set_tight_layout(tight=0.1)
  cmap = cm.get_cmap('hsv', 80)
  for image, output in dataset.take(1):
    # Parses image dimensions
    image_height, image_width, image_depth = image.shape
    # Parses info from sparse output
    steps = list(range(0, output.values.shape[0], 6))
    objectnesses = [output.values[i].numpy() for i in steps]
    cells = [yolo.GridCell(*output.indices[i][:2].numpy()) for i in steps]
    anchors = [0 if output.indices[i+1][2] == 1 else 1 for i in steps]
    bboxes = [
      yolo.decode_from_yolo_format(
        yolo.BBox(*output.values[i + 1: i + 5].numpy()),
        yolo.GridCell(*output.indices[i][:2].numpy())
      ) for i in steps
    ]
    labels = [output.indices[i + 5][2].numpy() - 10 for i in steps]
    # Plots all bounding boxes
    axes[0].imshow(image.numpy())
    for bbox, label, objectness in zip(bboxes, labels, objectnesses):
      # Adding rectangle
      patch = patches.Rectangle(
        (bbox.x * image_width, bbox.y * image_height),
        bbox.width * image_width,
        bbox.height * image_height,
        facecolor='none',
        edgecolor=cmap(label / 80),
        alpha=0.5,
        lw=2
      )
      axes[0].add_patch(patch)
      # Adding label
      axes[0].text(
        bbox.x * image_width + 3,
        bbox.y * image_height - 5,
        s='{:}({:.2f})'.format(index_to_categories[label], objectness),
        fontsize=12,
        backgroundcolor=cmap(label / 80),
        color='white'
      )
    axes[0].axis('off')
    # Randomly selects one bounding box and plots its relationship with anchor boxes
    selected = random.randint(0, len(objectnesses) - 1)
    axes[1].imshow(image.numpy())
    # Adding corresponding grid cell center
    axes[1].plot(
      (cells[selected].x + 0.5) / 7 * image_width,
      (cells[selected].y + 0.5) / 7 * image_height,
      '.', markersize=10, color='white'
    )
    # Adding anchor boxes at the corresponding grid cell
    for index, anchor in enumerate(yolo.anchors):
      patch = patches.Rectangle(
        (
          ((cells[selected].x + 0.5) / 7 - anchor.width / 2) * image_width,
          ((cells[selected].y + 0.5) / 7 - anchor.height / 2) * image_height
        ),
        anchor.width * image_width,
        anchor.height * image_height,
        alpha=0.5,
        color='white' if index == anchors[selected] else 'grey',
        lw=2
      )
      axes[1].add_patch(patch)
    # Adding bounding box
    patch = patches.Rectangle(
      (bboxes[selected].x * image_width, bboxes[selected].y * image_height),
      bboxes[selected].width * image_width,
      bboxes[selected].height * image_height,
      facecolor='none',
      edgecolor=cmap(labels[selected] / 80),
      alpha=0.6,
      lw=2
    )
    axes[1].add_patch(patch)
    # Adding bounding box center
    axes[1].plot(
      (bboxes[selected].x + bboxes[selected].width / 2) * image_width,
      (bboxes[selected].y + bboxes[selected].height / 2) * image_height,
      '.', markersize=10, color=cmap(labels[selected] / 80)
    )
    axes[1].axis('off')
    # Adding grid
    for i in range(8):
      # Horizontal
      axes[1].add_line(lines.Line2D(
        [0, image_width], [i * image_height / 7, i * image_height / 7],
        alpha=0.4,
        color='white')
      )
      # Vertical
      axes[1].add_line(lines.Line2D(
        [i * image_width / 7, i * image_width / 7], [0, image_height],
        alpha=0.4,
        color='white')
      )
  plt.show()
