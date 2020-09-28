#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import pathlib
import tensorflow as tf
import sys
sys.path.append('../../../../records generators')
import coco_raw_tfrecords_generator as coco

# Constants
DATASETS = pathlib.Path('E:/datasets')
AUTOTUNE = tf.data.experimental.AUTOTUNE


# Parses examples and returns bboxes width and height
def parse_raw_example(entry: tf.Tensor):
  example = tf.io.parse_single_example(entry, features=coco.features_description)
  num_objects = example['objects']
  bboxes = tf.reshape(tf.sparse.to_dense(example['bbox']), shape=[num_objects, 4])
  return bboxes[:, 2:]


def create_dataset():
  # Loads others examples
  records_path = DATASETS / 'coco/raw_records/train2014/'
  records_ds = tf.data.Dataset.from_tensor_slices([str(file) for file in records_path.glob('*.tfrecord*')])
  records_ds = records_ds.interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTOTUNE, deterministic=True)
  records_ds = records_ds.map(parse_raw_example, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
  return records_ds


def find_average_bboxes():
  # Creates dataset with width, heigh info for all bounding boxes
  dataset = create_dataset()
  cumulative_hbox = tf.zeros(shape=(2,), dtype=tf.float32)
  count_hbox = tf.constant(0, dtype=tf.float32)
  cumulative_vbox = tf.zeros(shape=(2,), dtype=tf.float32)
  count_vbox = tf.constant(0, dtype=tf.float32)

  for bboxes in dataset:
    width = bboxes[:, 0]
    height = bboxes[:, 1]
    mask = tf.greater(width, height)  # width greater than height (we called this horizontal box)
    hboxes = tf.boolean_mask(bboxes, mask, axis=0)
    vboxes = tf.boolean_mask(bboxes, tf.logical_not(mask), axis=0)
    cumulative_hbox += tf.reduce_sum(hboxes, axis=0)
    count_hbox += width.shape[0]
    cumulative_vbox += tf.reduce_sum(vboxes, axis=0)
    count_vbox += height.shape[0]
  return cumulative_hbox / count_hbox, cumulative_vbox / count_vbox


if __name__ == '__main__':
  average_hbox, average_vbox = find_average_bboxes()
  print('Average HBox:', average_hbox)
  print('Average VBox:', average_vbox)
