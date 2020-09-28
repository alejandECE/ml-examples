#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

from typing import Tuple
import tensorflow as tf
import pathlib
import re

# Constants
DATASETS = pathlib.Path('E:/datasets')
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
BUFFER_SIZE = BATCH_SIZE * 32
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Anchors shapes
ANCHORS_SHAPE = [
  [0.1, 0.3],
  [0.2, 0.1],
  [0.05, 0.05]
]

# Regex to extract category label and id
EXTRACT_CATEGORY_INFO_REGEX = re.compile(r"^\([0-9]+, '([a-z ]*)'\)$")


def get_coco_train_records_list():
  path = DATASETS / 'coco/raw_records/train2014'
  return [str(file) for file in path.glob('*.tfrecord*')]


def get_coco_test_records_list():
  path = DATASETS / 'coco/raw_records/val2014'
  return [str(file) for file in path.glob('*.tfrecord*')]


def get_coco_categories_lookup_tables():
  path = DATASETS / 'coco/raw_records/train2014/categories.txt'
  return create_categories_lookup_tables(path)


def create_categories_lookup_tables(path: pathlib.Path) -> Tuple:
  """
  Reads raw category labels from folder and assigns a new index to them (new indices won't correspond to original ones
  since for some reason they only have 80 categories and indices up to 90).

  :param path: Path to the categories.txt file
  :return: A tuple of tf.lookup.StaticHashTable tables mapping categories to indices and indices to categories
  """
  # Read text from raw categories text file
  with tf.io.gfile.GFile(str(path)) as file:
    text = file.read()
  # Parses text into categories assigning new indices
  keys = []
  values = []
  for line in text.splitlines():
    result = EXTRACT_CATEGORY_INFO_REGEX.match(line)
    category = result.group(1)
    values.append(len(values))
    keys.append(category)
  # Returns lookup tables
  category_to_index = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
      tf.convert_to_tensor(keys),
      tf.convert_to_tensor(values)
    ), -1
  )
  index_to_category = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
      tf.convert_to_tensor(values),
      tf.convert_to_tensor(keys)
    ), ''
  )
  return category_to_index, index_to_category
