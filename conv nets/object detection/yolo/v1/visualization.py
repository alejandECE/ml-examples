#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.cm as cm
import matplotlib.lines as lines
import random
import collections
import utils
import yolo

OBJECTNESS_THRESHOLD = 0.6

# Visual representation of a detected object
Object = collections.namedtuple('Object', ['bbox', 'index', 'objectness'])


# Converts anchor shape into a bbox at some cell
def convert_anchor_to_bbox(anchor: np.ndarray, cell: np.ndarray) -> np.ndarray:
  return np.array([
    (cell[0] + 0.5) / 7 - anchor[0] / 2,
    (cell[1] + 0.5) / 7 - anchor[1] / 2,
    anchor[0],
    anchor[1]
  ])


# Returns a denormalized 4D bbox in the image coordinates system
def denormalize_bbox_to_image_size(bbox: np.ndarray, width: int, height: int) -> np.ndarray:
  return np.array([
    bbox[0] * width,
    bbox[1] * height,
    bbox[2] * width,
    bbox[3] * height
  ])


# Returns a denormalized 2D point in the image coordinates system
def denormalize_cell_to_image_size(cell: np.ndarray, width: int, height: int) -> np.ndarray:
  return np.array([
    (cell[0] + 0.5) / 7 * width,
    (cell[1] + 0.5) / 7 * height
  ])


# Adds some text to axes
def add_text_to_axes(ax, text, location, color):
  # Adding label
  ax.text(
    location[0],
    location[1],
    s=text,
    fontsize=12,
    backgroundcolor=color,
    color='white'
  )


# Adds a bbox to axes with certain color
def add_bbox_to_axes(ax, bbox, color, alpha=0.5) -> None:
  # Adding rectangle
  patch = patches.Rectangle(
    (bbox[0], bbox[1]),
    bbox[2],
    bbox[3],
    facecolor='none',
    edgecolor=color,
    alpha=alpha,
    lw=2
  )
  ax.add_patch(patch)


# Adds anchor box to axes
def add_anchor_to_axes(ax, bbox, color) -> None:
  # Adding rectangle
  patch = patches.Rectangle(
    (bbox[0], bbox[1]),
    bbox[2],
    bbox[3],
    color=color,
    alpha=0.5,
    lw=2
  )
  ax.add_patch(patch)


# Adds point to axes
def add_point_to_axes(ax, location, color) -> None:
  ax.plot(location[0], location[1], '.', markersize=10, color=color)


# Adds object (bbox/description) to axes
def add_object_to_axes(ax,
                       obj: Object,
                       index_to_category: tf.lookup.StaticHashTable,
                       cmap) -> None:
  # Determines objects color
  color = cmap(obj.index / 80)
  # Determines text content/location
  text = '{:}({:.2f})'.format(
    index_to_category.lookup(tf.convert_to_tensor(obj.index)).numpy().decode('utf-8'),
    obj.objectness
  )
  location = (obj.bbox[0] + 3, obj.bbox[1] - 5)
  # Adding bounding box of object
  add_bbox_to_axes(ax, obj.bbox, color)
  # Adding label and objectness of object
  add_text_to_axes(ax, text, location, color)


def add_deleted_object_to_axes(ax, obj: Object) -> None:
  # Adding bounding box of object
  add_bbox_to_axes(ax, obj.bbox, 'white', 0.1)


# Adds object without description (only bbox) to axes
def add_object_without_description_to_axes(ax,
                                           obj: Object,
                                           cmap) -> None:
  # Determines objects color
  color = cmap(obj.index / 80)
  # Adding bounding box of object
  add_bbox_to_axes(ax, obj.bbox, color)


# Adds the object center to axes
def add_object_center_to_axes(ax, obj: Object, cmap) -> None:
  # Determines location/color
  location = (obj.bbox[0] + obj.bbox[2] / 2, obj.bbox[1] + obj.bbox[3] / 2)
  color = cmap(obj.index / 80)
  # Adds point to axes
  add_point_to_axes(ax, location, color)


# Adds a grid cell center point to the axes
def add_cell_center_to_axes(ax, cell) -> None:
  add_point_to_axes(ax, cell, color='white')


# Adds a grid to axes
def add_grid_to_axes(ax, width: int, height: int) -> None:
  # Adding grid
  for i in range(1, 7):
    # Horizontal
    ax.add_line(lines.Line2D(
      [0, width], [i * height / 7, i * height / 7],
      alpha=0.4,
      color='white')
    )
    # Vertical
    ax.add_line(lines.Line2D(
      [i * width / 7, i * width / 7], [0, height],
      alpha=0.4,
      color='white')
    )


# Computes IoU between two bboxes
def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
  # Finds interception area
  dx = min(box1[0] + box1[2], box2[0] + box2[2]) - max(box1[0], box2[0])
  if dx < 0:
    return 0
  dy = min(box1[1] + box1[3], box2[1] + box2[3]) - max(box1[1], box2[1])
  if dy < 0:
    return 0
  intersection = dx * dy
  # Determines union area
  union = box1[2] * box1[3] + box2[2] * box2[3] - intersection
  # Computes IoU and returns it
  return intersection / union


# Filter out all elements from the same class having a high IoU with the top element
def non_max_supression(root: Object, entry: Object):
  # From the same class and having a high IoU
  if root.index == entry.index and compute_iou(root.bbox, entry.bbox) > 0.5:
    return True
  # In any other case it should not be removed
  return False


def visualize_dataset_example(dataset: tf.data.Dataset,
                              index_to_category: tf.lookup.StaticHashTable) -> None:
  """
  Displays one observation from the dataset including bounding boxes and yolo detection output information.

  :param dataset: An unbatched and unprocessed (no resizing/no scaling) tf.data.Dataset
  :param index_to_category: A lookup table that maps indexes to categories
  """
  # Getting anchor shapes for latter use
  anchor_shapes = tf.convert_to_tensor(utils.ANCHORS_SHAPE)
  # Gets one observation from dataset
  iterator = iter(dataset)
  image, path, output = next(iterator)
  # Creates figure/axes
  fig, axes = plt.subplots(1, 2)
  fig.set_tight_layout(tight=0.1)
  fig.suptitle(path.numpy().decode('utf-8'))
  cmap = cm.get_cmap('hsv', 80)
  # Parses image dimensions
  image_height, image_width, image_depth = image.shape
  # Parses info from sparse outputs
  steps = list(range(0, output.values.shape[0], 6))
  bboxes = [
    denormalize_bbox_to_image_size(
      yolo.decode_from_yolo_format(
        output.values[i + 1: i + 5],
        output.indices[i][:2]
      ).numpy(),
      image_width,
      image_height
    ) for i in steps
  ]
  labels = [(tf.cast(output.indices[i + 5][2], dtype=tf.int32) - tf.cast(5 * tf.shape(anchor_shapes)[0],
                                                                         dtype=tf.int32)).numpy() for i in steps]
  objectnesses = [output.values[i].numpy() for i in steps]
  objects = [Object(*entry) for entry in zip(bboxes, labels, objectnesses)]
  # Plots all objects
  axes[0].imshow(image.numpy())
  for obj in objects:
    add_object_to_axes(axes[0], obj, index_to_category, cmap)
  # Randomly selects one bounding box and plots its relationship with anchor boxes
  cells = [output.indices[i][:2].numpy() for i in steps]
  anchors = [tf.math.floordiv(output.indices[i + 1][2], 5).numpy() for i in steps]
  selected = random.randint(0, len(cells) - 1)
  # Plots image on the second axes as well
  axes[1].imshow(image.numpy())
  # Adding corresponding grid cell center
  add_cell_center_to_axes(
    axes[1],
    denormalize_cell_to_image_size(
      cells[selected], image_width, image_height
    )
  )
  # Adding anchor boxes at the corresponding grid cell
  for index, anchor in enumerate(utils.ANCHORS_SHAPE):
    add_anchor_to_axes(
      axes[1],
      denormalize_bbox_to_image_size(
        convert_anchor_to_bbox(anchor, cells[selected]),
        image_width,
        image_height
      ),
      color='white' if index == anchors[selected] else 'grey',
    )
  # Adding bounding box of selected object
  add_object_without_description_to_axes(axes[1], objects[selected], cmap)
  # Adding bounding box center
  add_object_center_to_axes(axes[1], objects[selected], cmap)
  # Adding grid cell to image
  add_grid_to_axes(axes[1], image_width, image_height)
  # Let the magic show!
  axes[0].axis('off')
  axes[1].axis('off')
  axes[1].set_xlim(axes[0].get_xlim())
  axes[1].set_ylim(axes[0].get_ylim())
  plt.show()


def visualize_detection_examples(model: tf.keras.Model, dataset: tf.data.Dataset,
                                 index_to_category: tf.lookup.StaticHashTable,
                                 examples: int = 5) -> None:
  """
  Displays some detection examples (one at a time) from the given dataset using the specified model.

  :param model: Model to perform detection
  :param dataset: An unbatched and unprocessed (no resizing/no scaling) tf.data.Dataset
  :param index_to_category: A lookup table that maps indexes to categories
  :param examples: Number of examples to show
  """
  # Getting anchor shapes for latter use
  anchor_shapes = tf.convert_to_tensor(utils.ANCHORS_SHAPE)
  # Colormap for bounding boxes
  cmap = cm.get_cmap('hsv', 80)
  for image, path, output in dataset.take(examples):
    #  Creates figure/axes
    fig, axes = plt.subplots(1, 2)
    fig.set_tight_layout(tight=0.1)
    fig.suptitle(path.numpy().decode('utf-8'))
    # Parses image dimensions
    image_height, image_width, image_depth = image.shape
    # Parses info from sparse outputs
    steps = range(0, output.values.shape[0], 6)
    bboxes = [
      denormalize_bbox_to_image_size(
        yolo.decode_from_yolo_format(
          output.values[i + 1: i + 5],
          output.indices[i][:2]
        ).numpy(),
        image_width,
        image_height
      ) for i in steps
    ]
    labels = [(tf.cast(output.indices[i + 5][2], dtype=tf.int32) - tf.cast(5 * tf.shape(anchor_shapes)[0],
                                                                           dtype=tf.int32)).numpy() for i in steps]
    objectnesses = [output.values[i].numpy() for i in steps]
    objects = [Object(*entry) for entry in zip(bboxes, labels, objectnesses)]
    # Plots all objects
    axes[0].imshow(image.numpy())
    for obj in objects:
      add_object_to_axes(axes[0], obj, index_to_category, cmap)
    # Plots detection results
    axes[1].imshow(image.numpy())
    # Gets all valid bboxes (one per cell)
    predicted = tf.squeeze(model(tf.expand_dims(yolo.preprocess_image(image), axis=0)))
    indices = tf.range(5 * tf.shape(anchor_shapes)[0], tf.shape(predicted)[2])
    probability = tf.gather(predicted, indices=indices, axis=-1)
    category = tf.cast(tf.argmax(probability, axis=-1), dtype=tf.int32)
    indices = tf.range(0, tf.shape(anchor_shapes)[0]) * 5
    objectness = tf.gather(predicted, indices=indices, axis=-1)
    anchors = tf.argmax(objectness, axis=-1)
    objects = [
      Object(
        bbox=denormalize_bbox_to_image_size(
          yolo.clip_bbox_to_image(yolo.decode_from_yolo_format(
            predicted[i, j, anchors[i, j] * 5 + 1: anchors[i, j] * 5 + 1 + 4],
            tf.convert_to_tensor([i, j])
          )).numpy(),
          image_width,
          image_height
        ),
        index=category[i, j],
        objectness=objectness[i, j, anchors[i, j]] * probability[i, j, category[i, j]]
      ) for i in range(7) for j in range(7)
    ]
    # Only objects with high certainty are considered
    detections = filter(lambda entry: entry.objectness > OBJECTNESS_THRESHOLD, objects)
    # Performs non-max suppression
    sorted_detections = sorted(detections, key=lambda entry: entry.objectness, reverse=True)
    included_detections = []
    excluded_detections = []
    while len(sorted_detections) > 0:
      # Top element is always a detection since is the highest confidence object
      root = sorted_detections[0]
      included_detections.append(root)
      # Filter out all elements from the same class having a high IoU with the top element
      suppression = [non_max_supression(root, entry) for entry in sorted_detections[1:]]
      excluded_detections.extend([entry for entry, suppressed in zip(sorted_detections[1:], suppression) if suppressed])
      sorted_detections = [entry for entry, suppressed in zip(sorted_detections[1:], suppression) if not suppressed]
    # Plots included detections
    for obj in included_detections:
      add_object_to_axes(axes[1], obj, index_to_category, cmap)
    # Plots excluded detections
    for obj in excluded_detections:
      add_deleted_object_to_axes(axes[1], obj)
    # Let the magic show!
    axes[0].axis('off')
    axes[1].axis('off')
    axes[1].set_xlim(axes[0].get_xlim())
    axes[1].set_ylim(axes[0].get_ylim())
    plt.show()
