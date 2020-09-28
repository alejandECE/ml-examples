#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import tensorflow as tf
import datetime
import argparse
import sys
sys.path.append('../../../../records generators')
import coco_raw_tfrecords_generator as coco
from typing import Tuple
import pathlib
import utils
import losses
import visualization as viewer


def find_grid_cell_for_bboxes(bboxes: tf.Tensor) -> tf.Tensor:
  """
  Finds corresponding cell (row, col) for the bbox, where row, col are integers 0-6.

  :param bboxes: A 2D tensor of shape [M, 4] where M is the number of bounding boxes.
  :return: A 2D tensor of shape [M, 2] with the corresponding cell for each bounding box.
  """
  # Finds the bbox center
  cells = bboxes[:, :2] + bboxes[:, 2:] / 2
  # Finds corresponding cell
  return tf.cast(tf.floor(cells * 7), dtype=tf.int64)


def convert_anchors_to_bboxes(anchors: tf.Tensor, cells: tf.Tensor) -> tf.Tensor:
  """
  Converts anchors shapes (width, height) to bboxes in the normalized image coordinate system for a given cell
  using (left_top_x,left_top_y,width,height) format

  :param anchors: A 2D tensor of shape [N, 2], where N is the number of anchors.
  :param cells: A 2D tensor of shape [C, 2], where C is the number of cells.
  :return: A 3D tensor of shape [C, N, 4] tensor representing one bounding box per anchor per cell.
  """
  # Assuming anchor box to be centered at the cell's center point
  expanded_anchors = tf.expand_dims(anchors, axis=0)
  expanded_cells = tf.expand_dims(cells, axis=1)
  shape = tf.broadcast_dynamic_shape(tf.shape(expanded_anchors), tf.shape(expanded_cells))
  dimensions = tf.cast(tf.broadcast_to(expanded_anchors, shape=shape), dtype=tf.float32)
  locations = tf.cast(tf.broadcast_to(expanded_cells, shape=shape), dtype=tf.float32)
  locations = (locations + 0.5) / 7 - dimensions / 2
  return tf.concat((locations, dimensions), axis=2)


def compute_batch_iou(box1: tf.Tensor, box2: tf.Tensor) -> tf.Tensor:
  """
  Computes batch IoU

  :param box1: A 2D tensor of shape [Batch Size, 4]
  :param box2: A 2D tensor of shape [Batch Size, 4]
  :return: The compute IoU as a 1D tensor of shape [Batch Size, ]
  """
  # Finds interception area
  dx = tf.minimum(box1[..., 0] + box1[..., 2], box2[..., 0] + box2[..., 2]) - tf.maximum(box1[..., 0], box2[..., 0])
  dx = tf.multiply(dx, tf.cast(tf.greater(dx, 0), dtype=dx.dtype))
  dy = tf.minimum(box1[..., 1] + box1[..., 3], box2[..., 1] + box2[..., 3]) - tf.maximum(box1[..., 1], box2[..., 1])
  dy = tf.multiply(dy, tf.cast(tf.greater(dy, 0), dtype=dy.dtype))
  intersection = tf.multiply(dx, dy)
  # Determines union area
  union = tf.multiply(box1[..., 2], box1[..., 3]) + tf.multiply(box2[..., 2], box2[..., 3]) - intersection
  # Computes IoU and returns it
  return tf.divide(intersection, union)


def encode_to_yolo_format(bboxes: tf.Tensor, cells: tf.Tensor) -> tf.Tensor:
  """
  Encodes a bbox from the image reference system to the given cell reference system using the yolo format

  :param bboxes: A 2D tensor of shape [M, 4] where M is the number of bounding boxes (objects)
  :param cells: A 2D tensor of shape [M, 2]
  :return: A 2D tensor of shape [M, 4] with the encoded bounding boxes
  """
  fx = 7 * (bboxes[..., 0] + bboxes[..., 2] / 2) - tf.cast(cells[..., 0], dtype=bboxes.dtype)
  fy = 7 * (bboxes[..., 1] + bboxes[..., 3] / 2) - tf.cast(cells[..., 1], dtype=bboxes.dtype)
  return tf.stack([
    fx, fy, tf.sqrt(bboxes[..., 2]), tf.sqrt(bboxes[..., 3])
  ], axis=1)


def decode_from_yolo_format(bbox: tf.Tensor, cell: tf.Tensor) -> tf.Tensor:
  """
  Decodes yolo format bbox

  :param bbox: 1D Tensor of shape [4,] representing the bounding box in yolo format
  :param cell: 1D Tensor of shape [2,] representing the grid cell where the bounding box is at
  :return: Normalized bbox in the original image coordinate system (image format)
  """
  width = bbox[2] * bbox[2]
  height = bbox[3] * bbox[3]
  fx = (tf.cast(cell[0], dtype=bbox.dtype) + bbox[0]) / 7 - width / 2
  fy = (tf.cast(cell[1], dtype=bbox.dtype) + bbox[1]) / 7 - height / 2
  return tf.stack([fx, fy, width, height])


def clip_bbox_to_image(bbox: tf.Tensor):
  width = bbox[2] if bbox[2] < 1. else 1.
  height = bbox[3] if bbox[3] < 1. else 1.
  x = bbox[0] if bbox[0] > 0. else 0.
  y = bbox[1] if bbox[1] > 0. else 0.
  return tf.stack([x, y, width, height])


def find_closest_anchor_to_bboxes(bboxes: tf.Tensor, cells: tf.Tensor) -> Tuple:
  """
  Finds the closest anchor shape to each object in its corresponding cell by computing the IoU between each object's
  bounding box and the corresponding anchor box and returning the anchor index corresponding to the maximum IoU.

  :param bboxes: A 2D tensor of shape [C, 4] where C is the number of bounding boxes (objects)
  :param cells: A 2D tensor of shape [C, 2]
  :return: A tuple of tensors of size C containing (anchor box index, maximum iou (objectness))
  """
  # Converts anchors shapes to bounding boxes (returns 2D tensor of shape [C, N, 4] where N is the number of anchors)
  anchor_shapes = tf.convert_to_tensor(utils.ANCHORS_SHAPE)
  anchor_bboxes = convert_anchors_to_bboxes(anchor_shapes, cells)
  # Repeats the same bounding box for every anchor box available
  expanded_bboxes = tf.broadcast_to(tf.expand_dims(bboxes, axis=1),
                                    shape=(tf.shape(bboxes)[0], tf.shape(anchor_shapes)[0], tf.shape(bboxes)[1]))
  # Computes IoU
  iou = compute_batch_iou(anchor_bboxes, expanded_bboxes)
  # Finds index of maximum and maximum
  return tf.argmax(iou, axis=1), tf.reduce_max(iou, axis=1)


def prepare_detection_output(bboxes: tf.Tensor,
                             labels: tf.Tensor,
                             category_to_index: tf.lookup.StaticHashTable) -> tf.sparse.SparseTensor:
  """
  Prepares detection output according to yolo v1 format. The process is as follows:
  1. Finds the closest cell to each object's bounding box.
  2. Finds the closest anchor box to each object in its corresponding cell by computing the IoU between each object's
     bounding box and the corresponding anchor box and returning the anchor index corresponding to the maximum IoU.
  3. Creates the sparse tensor output as specified below.

  :param bboxes: A 2D tensor of shape [M, 4] where M is the number of bounding boxes (objects)
  :param labels: A tensor with M elements.
  :param category_to_index: Lookup table to map categories to index
  :return: A 3D sparse tensor of shape 7x7x(1 + 4 + 1 + 4 + 80) where the last dimension values are (in order):
  objectness, bbox center x, bbox center y, bbox sqrt(w), bbox sqrt(h) for both anchor shapes and one hot encoded
  category (class) index
  """
  # Gets cells for bboxes
  cells = find_grid_cell_for_bboxes(bboxes)
  # There might be repeated cells due to two bounding boxes falling into the same cell. We can only detect one object
  # per cell in yolo v1 so we need to get rid of duplicates
  subscripts = cells[:, 0] * 7 + cells[:, 1]
  filtered_subscripts, _ = tf.unique(subscripts)
  # Comparing one by one every value between original subscripts and the filtered subscripts
  subscripts = tf.expand_dims(subscripts, axis=-1)
  filtered_subscripts = tf.expand_dims(filtered_subscripts, axis=0)
  shape = tf.broadcast_dynamic_shape(tf.shape(subscripts), tf.shape(filtered_subscripts))
  # Find index of positive comparison (index in the original tensor)
  mask = tf.argmax(tf.equal(tf.broadcast_to(subscripts, shape), tf.broadcast_to(filtered_subscripts, shape)), axis=0)
  # Filters all cells, bboxes and labels
  filtered_bboxes = tf.gather(bboxes, indices=mask, axis=0)
  filtered_cells = tf.gather(cells, indices=mask, axis=0)
  filtered_labels = tf.gather(labels, indices=mask, axis=0)
  # Find closest anchor for each object in cell
  anchors, objectness = find_closest_anchor_to_bboxes(filtered_bboxes, filtered_cells)
  # Converts bboxes to yolo format
  yolo_bboxes = encode_to_yolo_format(filtered_bboxes, filtered_cells)
  # Prepares indices and values to initialize sparse tensor
  anchor_shapes = tf.convert_to_tensor(utils.ANCHORS_SHAPE)
  entry = tf.reshape(
    tf.concat([
      tf.tile(
        tf.expand_dims(tf.range(5, dtype=anchors.dtype), axis=0),
        multiples=[tf.shape(anchors)[0], 1]
      ) + tf.expand_dims(anchors, axis=-1) * 5,
      tf.expand_dims(tf.cast(tf.shape(anchor_shapes)[0] * 5 + category_to_index.lookup(filtered_labels),
                             dtype=anchors.dtype), axis=-1)
    ], axis=-1),
    shape=[-1, 1]
  )
  indices = tf.concat([
    tf.repeat(filtered_cells, repeats=6, axis=0),
    entry
  ], axis=-1)
  values = tf.reshape(tf.concat([
    tf.expand_dims(objectness, axis=-1),  # objectness
    yolo_bboxes,  # bbox
    tf.ones(shape=[tf.shape(yolo_bboxes)[0], 1], dtype=objectness.dtype)  # just a one (one hot encoded category)
  ], axis=-1), shape=[-1])
  # Creates sparse output
  output = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=(7, 7, 5 * tf.shape(anchor_shapes)[0] + 80))
  return tf.sparse.reorder(output)


def preprocess_image(img: tf.Tensor) -> tf.Tensor:
  """
  Applies preprocessing to the images:
    1. Resizes the image to the acceptable input size
    2. Preprocess image according to the transferred model preprocessing function

  :param img: A 4D tensor of shape [Batch Size, Height, Width, Channels]
  :return: A 4D tensor representing the pre-processed image
  """
  # Resizes to the network input specs
  img = tf.image.resize(img, [utils.IMG_HEIGHT, utils.IMG_WIDTH], antialias=True)
  # Preprocess data according to transferred model
  img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
  return img


def parse_raw_example(entry: tf.Tensor) -> Tuple:
  """
  Parses a binary example loaded from tfrecord files and returns image, path, bboxes and labels

  :param entry: A 1D string tensor with the binary data loaded from file
  :return: A tuple of 4 tensors
  """
  example = tf.io.parse_single_example(entry, features=coco.features_description)
  image = tf.io.decode_image(example['image'], dtype=tf.uint8, channels=3, expand_animations=False)
  num_objects = example['objects']
  bboxes = tf.reshape(tf.sparse.to_dense(example['bbox']), shape=[num_objects, 4])
  path = example['path']
  labels = tf.sparse.to_dense(example['label'])
  return image, path, bboxes, labels


# Creates training dataset (this includes preparing yolo expected outputs as sparse tensors)
def create_training_dataset(observations: int,
                            category_to_index: tf.lookup.StaticHashTable,
                            index_to_category: tf.lookup.StaticHashTable,
                            display=False):
  # Loads and parse raw examples
  records_ds = tf.data.Dataset.from_tensor_slices(utils.get_coco_train_records_list())
  records_ds = records_ds.interleave(tf.data.TFRecordDataset, num_parallel_calls=utils.AUTOTUNE, deterministic=True)
  records_ds = records_ds.map(parse_raw_example, num_parallel_calls=utils.AUTOTUNE)
  records_ds = records_ds.take(observations)
  # Prepares output for yolo v1 detector
  records_ds = records_ds.map(lambda image, path, bboxes, labels:
                              (image, path, prepare_detection_output(bboxes, labels, category_to_index)),
                              num_parallel_calls=utils.AUTOTUNE)
  # Display if selected
  unprocessed_ds = records_ds.shuffle(utils.BUFFER_SIZE)
  if display:
    viewer.visualize_dataset_example(unprocessed_ds, index_to_category)
  # Applies pre-processing
  records_ds = records_ds.map(lambda image, path, outputs: (preprocess_image(image), outputs),
                              num_parallel_calls=utils.AUTOTUNE)
  # Shuffles prior to batch (to obtain new batches every epoch)
  records_ds = records_ds.shuffle(buffer_size=utils.BUFFER_SIZE)
  records_ds = records_ds.batch(utils.BATCH_SIZE)
  # Optimizes pre-loading some batches
  records_ds = records_ds.prefetch(buffer_size=utils.AUTOTUNE)
  return records_ds, unprocessed_ds


# Creates test dataset (this includes preparing yolo expected outputs as sparse tensors)
def create_test_dataset(observations: int,
                        category_to_index: tf.lookup.StaticHashTable,
                        index_to_category: tf.lookup.StaticHashTable,
                        display=False):
  # Loads and parse raw examples
  records_ds = tf.data.Dataset.from_tensor_slices(utils.get_coco_test_records_list())
  records_ds = records_ds.interleave(tf.data.TFRecordDataset, num_parallel_calls=utils.AUTOTUNE, deterministic=True)
  records_ds = records_ds.map(parse_raw_example, num_parallel_calls=utils.AUTOTUNE)
  records_ds = records_ds.take(observations)
  # Prepares output for yolo v1 detector
  records_ds = records_ds.map(lambda image, path, bboxes, labels:
                              (image, path, prepare_detection_output(bboxes, labels, category_to_index)),
                              num_parallel_calls=utils.AUTOTUNE)
  # Display if selected
  unprocessed_ds = records_ds.shuffle(utils.BUFFER_SIZE)
  if display:
    viewer.visualize_dataset_example(unprocessed_ds, index_to_category)
  # Applies pre-processing
  records_ds = records_ds.map(lambda image, path, outputs: (preprocess_image(image), outputs),
                              num_parallel_calls=utils.AUTOTUNE)
  # Batches dataset
  records_ds = records_ds.batch(utils.BATCH_SIZE)
  # Optimizes pre-loading some batches
  records_ds = records_ds.prefetch(buffer_size=utils.AUTOTUNE)
  return records_ds, unprocessed_ds


# Creates transferred model
def create_model(timestamp: str = None, unfreeze=0) -> Tuple:
  anchor_shapes = tf.convert_to_tensor(utils.ANCHORS_SHAPE)
  base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False,
                                                              input_shape=[utils.IMG_HEIGHT, utils.IMG_WIDTH, 3])
  # Avoid changing loaded weights of some (if not all) layers
  layer_count = len(base_model.layers)
  for index in range(layer_count - unfreeze):
    base_model.layers[index].trainable = False
  # Defining extra layers
  inputs = tf.keras.layers.Input([utils.IMG_HEIGHT, utils.IMG_WIDTH, 3])
  conv1 = tf.keras.layers.Conv2D(filters=128, kernel_size=1, activation='relu')
  conv2 = tf.keras.layers.Conv2D(filters=(5 * tf.shape(anchor_shapes)[0] + 80), kernel_size=1)
  # Creating connections for the whole model
  x = base_model(inputs)
  x = conv1(x)
  outputs = conv2(x)
  # Creates model
  mdl = tf.keras.Model(inputs, outputs)
  # Displays summary
  mdl.summary()
  # Stores image
  if timestamp is not None:
    diagram_path = pathlib.Path('trainings') / timestamp / 'diagram'
    diagram_path.mkdir(parents=True)
    tf.keras.utils.plot_model(mdl, to_file=str(diagram_path / 'model.png'),
                              expand_nested=False, show_shapes=True, show_layer_names=False)
  # Defines optimizer, loss and metric
  optimizer = tf.keras.optimizers.Adam()
  loss_fn = losses.YoloLoss()
  return mdl, optimizer, loss_fn


# Simple test step
@tf.function
def test_step(model, batch, loss_fn):
  # Unwrapped inputs and expected output
  inputs, y_true = batch
  # Computes output of the model
  y_pred = model(inputs)
  # Computes loss
  loss = loss_fn(y_true, y_pred)
  return loss


# Simple train step
@tf.function
def train_step(model, batch, optimizer, loss_fn):
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
  return loss


# Entry point to training
def train_and_record(parser: argparse.ArgumentParser):
  # Generates timestamp for later use
  timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  # Parses arguments
  args = parser.parse_args()
  epochs = args.epochs if args.epochs else 2
  samples = args.samples if args.samples else 1000
  save = False if args.unsaved else True
  display = True if args.display else False
  unfreeze = args.unfreeze if args.unfreeze else 0
  ckpt_load_path = pathlib.Path(args.checkpoint) if args.checkpoint else None
  # Creates model to train
  model, optimizer, loss_fn = create_model(
    timestamp if save else None,
    unfreeze=unfreeze
  )
  # Creates categories lookup tables
  category_to_index, index_to_category = utils.get_coco_categories_lookup_tables()
  # Creates datasets
  train_ds, train_unprocessed_ds = create_training_dataset(samples, category_to_index, index_to_category, display=display)
  test_ds, test_unprocessed_ds = create_test_dataset(samples, category_to_index, index_to_category, display=display)
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
  ckpt_save_path = pathlib.Path('trainings') / timestamp / 'checkpoints'
  ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
  manager = tf.train.CheckpointManager(ckpt, ckpt_save_path, max_to_keep=2)
  # Training loop
  lowest_loss = None
  steps_per_epoch = 0
  for epoch in range(1, epochs + 1):
    found_better = False
    cumulative_loss = 0
    # Step reset to 1
    step = 1
    for batch in train_ds:
      # Updates steps per epoch
      if step > steps_per_epoch:
        steps_per_epoch = step
      # Performs training step
      loss = train_step(model, batch, optimizer, loss_fn)
      # Updates epoch training performance
      cumulative_loss += loss.numpy()
      avg_step_loss = cumulative_loss / step
      print('\rEpoch {:}/{:}, Step {:}/{:}: Train Loss: {:.4f}'.format(
        epoch, epochs, step, steps_per_epoch, avg_step_loss
      ), end='')
      step += 1
    # Verifies if a better model has been found
    if lowest_loss is None or cumulative_loss < lowest_loss:
      found_better = True
      lowest_loss = cumulative_loss
    # Evaluates test set
    cumulative_loss = 0
    # Step reset to 1
    step = 1
    for batch in test_ds:
      # Performs test step
      loss = test_step(model, batch, loss_fn)
      # Updates epoch test performance
      cumulative_loss += loss.numpy()
      avg_step_loss = cumulative_loss / step
      step += 1
    print(' - Test Loss: {:.4f}'.format(
      avg_step_loss
    ))
    # Saves checkpoint if requirements are met and if requested
    if save and found_better:
      manager.save()
  # Visualize a detection example (we pass the unprocessed ds to be able to recover the original img)
  ckpt.restore(manager.latest_checkpoint)
  viewer.visualize_detection_example(model, train_unprocessed_ds, index_to_category)


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
  parser.add_argument('--checkpoint', help='Path to model to load', type=str)
  parser.add_argument('--unfreeze', help='Update parameters from transferred layers', type=int)
  # Resumes or start training according to the options selected
  train_and_record(parser)
