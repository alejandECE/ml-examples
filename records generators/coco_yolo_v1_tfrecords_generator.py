#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import math
from typing import Tuple
import apache_beam as beam
import tensorflow as tf
import argparse
from apache_beam.options.pipeline_options import PipelineOptions
import os
import pathlib
import collections
import re

# Regex to extract category label and id
EXTRACT_CATEGORY_INFO_REGEX = re.compile(r"^\(([0-9]+), '([a-z ]*)'\)$")

# GCS credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'E:/secrets/ml-examples-ff91761a6a56.json'

# Path constants
PROJECT_ID = 'ml-examples-284704'
CLOUD_STAGING_PREFIX = 'staging/'
CLOUD_TEMP_PREFIX = 'temp/'
LOCAL_DATASET_BUCKET = 'E:/datasets/coco'
CLOUD_DATASET_BUCKET = 'ml-examples-coco'
CLOUD_YOLO_RECORDS_PREFIX = 'yolo_v1_records/'
LOCAL_YOLO_RECORDS_PREFIX = 'yolo_v1_records/'
CLOUD_RAW_RECORDS_PREFIX = 'raw_records/'
LOCAL_RAW_RECORDS_PREFIX = 'raw_records/'

# Feature description to parse input proto example
inputs_description = {
  'id': tf.io.FixedLenFeature([], tf.int64),
  'path': tf.io.FixedLenFeature([], tf.string),
  'image': tf.io.FixedLenFeature([], tf.string),
  'objects': tf.io.FixedLenFeature([], tf.int64),
  'label': tf.io.VarLenFeature(tf.string),
  'bbox': tf.io.VarLenFeature(tf.float32)
}

# Feature description to parse output proto example
outputs_description = {
  'id': tf.io.FixedLenFeature([], tf.int64),
  'path': tf.io.FixedLenFeature([], tf.string),
  'image': tf.io.FixedLenFeature([], tf.string),
  'indices': tf.io.FixedLenFeature([], tf.string),
  'values': tf.io.FixedLenFeature([], tf.string),
  'shape': tf.io.FixedLenFeature([], tf.string)
}

# Helper namedtuples
BBox = collections.namedtuple('BBox', ['x', 'y', 'width', 'height'])
Anchor = collections.namedtuple('Anchor', ['width', 'height'])
GridCell = collections.namedtuple('GridCell', ['x', 'y'])

anchors = [
  Anchor(0.088, 0.169),  # Average vertical bbox
  Anchor(0.092, 0.058)   # Average horizontal bbox
]


# Computes IoU between two bounding boxes in the (left_top_x,left_top_y,width,height) format
def compute_iou(box1: BBox, box2: BBox) -> float:
  # Determines intersection area
  dy = min(box1.y + box1.height, box2.y + box2.height) - max(box1.y, box2.y)
  if dy < 0:
    dy = 0
  dx = min(box1.x + box1.width, box2.x + box2.width) - max(box1.x, box2.x)
  if dx < 0:
    dx = 0
  intersection = dx * dy
  # Determines union area
  union = box1.width * box1.height + box2.width * box2.height - intersection
  return intersection / union


# Finds corresponding cell (row, col) for the bbox, where row, col are integers 0-6
def find_grid_cell_for_bbox(bbox: BBox) -> GridCell:
  # Finds the bbox center
  cx = bbox.x + bbox.width / 2
  cy = bbox.y + bbox.height / 2
  # Finds corresponding cell
  return GridCell(
    x=math.floor(cx * 7),
    y=math.floor(cy * 7)
  )


# Converts an anchor box (width, height) for a given cell to a bbox in the normalized image coordinate system
# using (left_top_x,left_top_y,width,height) format
def convert_anchor_to_bbox(anchor: Anchor, cell: GridCell) -> BBox:
  # Assuming anchor box to be centered at the cell's center point
  fx = (cell.x + 0.5) / 7 - anchor.width / 2
  fy = (cell.y + 0.5) / 7 - anchor.height / 2
  return BBox(fx, fy, anchor.width, anchor.height)


# Encodes a bbox from the image reference system to the given cell reference system using the yolo format
def encode_to_yolo_format(bbox: BBox, cell: GridCell) -> BBox:
  fx = 7 * (bbox.x + bbox.width / 2) - cell.x
  fy = 7 * (bbox.y + bbox.height / 2) - cell.y
  return BBox(fx, fy, math.sqrt(bbox.width), math.sqrt(bbox.height))


# Decodes from yolo format to (left_top_x,left_top_y,width,height) format
def decode_from_yolo_format(yolo_box: BBox, cell: GridCell) -> BBox:
  width = yolo_box.width * yolo_box.width
  height = yolo_box.height * yolo_box.height
  fx = (cell.x + yolo_box.x) / 7 - width / 2
  fy = (cell.y + yolo_box.y) / 7 - height / 2
  return BBox(fx, fy, width, height)


# Finds the closest (higher IoU) anchor for a box in a given cell.
# Returns index of the anchor and the corresponding IoU
def find_closest_anchor_to_bbox(bbox: BBox, cell: GridCell) -> Tuple[int, float]:
  pairs = [(index, compute_iou(bbox, convert_anchor_to_bbox(anchor, cell))) for index, anchor in enumerate(anchors)]
  return max(pairs, key=lambda entry: entry[1])


# Creates path to store the yolo records
def get_save_records_path(version: str, job: int, dataflow_runner=False) -> str:
  if dataflow_runner:
    return 'gs://' + CLOUD_DATASET_BUCKET + '/' + CLOUD_YOLO_RECORDS_PREFIX + \
           'data-job{:}.tfrecord'.format(job)
  else:
    return LOCAL_DATASET_BUCKET + '/' + LOCAL_YOLO_RECORDS_PREFIX + version + \
           '/data-job{:}.tfrecord'.format(job)


# Path to the raw records folder
def get_load_records_path(version: str, dataflow_runner=False) -> str:
  if dataflow_runner:
    return 'gs://' + CLOUD_DATASET_BUCKET + '/' + CLOUD_RAW_RECORDS_PREFIX
  else:
    return LOCAL_DATASET_BUCKET + '/' + LOCAL_RAW_RECORDS_PREFIX + version + '/'


# Distributes filenames across jobs (returning one list per job)
def create_jobs_records_lists(filenames: list, jobs: int, load: int):
  # Assigns number of records per job
  records_count = len(filenames) if load is None or load > len(filenames) else load
  records_per_job = [records_count // jobs] * jobs
  for i in range(records_count % jobs):
    records_per_job[i] += 1
  # Create lists of records (one list per job)
  records_lists = []
  start = 0
  for count in records_per_job:
    # Creates a sublist of records for the current job
    records_lists.append([record for record in filenames[start: start + count]])
    start += count
  return records_lists


# Reads raw category labels from bucket and assigns a new index to them (new indices won't correspond to original ones
# since for some reason they only have 80 categories and indices up to 90).
def create_categories_from_bucket(version: str) -> dict:
  # Lazy import
  from google.cloud import storage
  # Instantiate a gc storage client
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(CLOUD_DATASET_BUCKET)
  blob = bucket.blob(CLOUD_RAW_RECORDS_PREFIX + version + '/categories.txt')
  text = blob.download_as_string().decode('utf-8')
  # Parses text into categories assigning new indices
  index_to_category = []
  category_to_index = {}
  for line in text.splitlines():
    result = EXTRACT_CATEGORY_INFO_REGEX.match(line)
    category = result.group(2)
    category_to_index[category] = len(index_to_category)
    index_to_category.append(category)
  # Stores new categories to text file
  blob = bucket.blob(CLOUD_YOLO_RECORDS_PREFIX + version + '/categories.txt')
  # Stores the actual text to file in the cloud
  output = ''.join(['{:}\n'.format((value, key)) for key, value in category_to_index.items()])
  blob.upload_from_string(output)
  # Returns dictionary
  return category_to_index


# Reads raw category labels from folder and assigns a new index to them (new indices won't correspond to original ones
# since for some reason they only have 80 categories and indices up to 90).
def create_categories_from_folder(version: str) -> dict:
  # Read text from raw categories text file
  path = LOCAL_DATASET_BUCKET + '/' + LOCAL_RAW_RECORDS_PREFIX + version + '/categories.txt'
  with tf.io.gfile.GFile(path) as file:
    text = file.read()
  # Parses text into categories assigning new indices
  index_to_category = []
  category_to_index = {}
  for line in text.splitlines():
    result = EXTRACT_CATEGORY_INFO_REGEX.match(line)
    category = result.group(2)
    category_to_index[category] = len(index_to_category)
    index_to_category.append(category)
  # Stores new categories to text file
  path = pathlib.Path(LOCAL_DATASET_BUCKET) / LOCAL_YOLO_RECORDS_PREFIX / version / 'categories.txt'
  # If folders don't exist create them
  if not path.parent.exists():
    path.mkdir(parents=True)
  # Stores the actual text
  output = ''.join(['{:}\n'.format((value, key)) for key, value in category_to_index.items()])
  with tf.io.gfile.GFile(str(path), 'w') as file:
    file.write(output)
  # Returns dictionary
  return category_to_index


# Returns all tfrecord filenames found in the bucket
def get_tfrecords_filenames_from_bucket(version: str) -> list:
  # Lazy import
  from google.cloud import storage
  # Instantiate a gc storage client
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(CLOUD_DATASET_BUCKET)
  blobs = bucket.list_blobs(prefix=CLOUD_RAW_RECORDS_PREFIX + version + '/')
  # Filters all blobs corresponding to tfrecord files
  filenames = [blob.name.split('/')[-1] for blob in blobs if 'tfrecord' in blob.name]
  return filenames


# Returns all tfrecord filenames found in the folder
def get_tfrecords_filenames_from_folder(version: str) -> list:
  folder = pathlib.Path(LOCAL_DATASET_BUCKET) / LOCAL_RAW_RECORDS_PREFIX / version
  blobs = folder.glob('*.tfrecord*')
  return [blob.name for blob in blobs]


class ReadParseGenerateFn(beam.DoFn):
  def __init__(self, root: str, category_to_index: dict):
    self.root = root
    self.category_to_index = category_to_index

  def process(self, element, *args, **kwargs):
    # Builds path to tfrecord file
    path = self.root + element
    # Verifies record file exist
    if not tf.io.gfile.exists(path):
      return
    # Creates a tf.data.Dataset to read from the tfrecord file
    records_ds = tf.data.TFRecordDataset(path)
    # Goes through every example
    for entry in records_ds:
      # We will generate a sparse tensor encoding all information of the expected output (7x7x(10+80))
      # We use a dictionary with (cell.x, cell.y) as key and another dictionary as value. The later to encode the third
      # dimension as key and the actual information as value. This ensures only one object per cell is encoded. If there
      # is more than one only the last one will remain.
      info = {}
      # Parses example
      example = tf.io.parse_single_example(entry, features=inputs_description)
      # Parses bboxes
      num_objects = example['objects']
      boxes = tf.reshape(tf.sparse.to_dense(example['bbox']), shape=[num_objects, 4])
      labels = tf.sparse.to_dense(example['label'])
      # We have to encode each object bounding box
      for box, label in zip(boxes, labels):
        bbox = BBox(*box.numpy())
        # Finds corresponding cell for bounding box
        cell = find_grid_cell_for_bbox(bbox)
        # Finds closest anchor box
        anchor, iou = find_closest_anchor_to_bbox(bbox, cell)
        # Encodes bounding box into yolo format
        yolo_bbox = encode_to_yolo_format(bbox, cell)
        # Encodes information for the cell in a sparse format
        info[cell] = {}
        # Adds objectness score to the sparse output
        info[cell][5 * anchor] = iou
        # Adds yolo box to the sparse output by adding indices and values
        for i, value in enumerate(yolo_bbox):
          info[cell][5 * anchor + i + 1] = value
        # Adds label to sparse output
        info[cell][10 + self.category_to_index[label.numpy().decode('utf-8')]] = 1
      # Creates sparse tensor to represent output
      indices = []
      values = []
      for cell in info:
        for index in info[cell]:
          indices.append([cell.x, cell.y, index])
          values.append(info[cell][index])
      output = tf.sparse.SparseTensor(
        indices=tf.convert_to_tensor(indices, dtype=tf.int64),
        values=tf.convert_to_tensor(values, dtype=tf.float32),
        dense_shape=(7, 7, 90)
      )
      output = tf.io.serialize_sparse(tf.sparse.reorder(output))
      # Creates tf.train.Example
      features = tf.train.Features(feature={
        'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[example['id'].numpy()])),
        'path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[example['path'].numpy()])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[example['image'].numpy()])),
        'indices': tf.train.Feature(bytes_list=tf.train.BytesList(value=[output[0].numpy()])),
        'values': tf.train.Feature(bytes_list=tf.train.BytesList(value=[output[1].numpy()])),
        'shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[output[2].numpy()]))
      })
      yield tf.train.Example(features=features).SerializeToString()


# Creates pipeline options according to runner selected
def create_pipeline_options(job, workers, dataflow_runner=False):
  # Allowed dataflow regions
  regions = ['us-west1', 'us-central1', 'us-east4', 'us-east1', 'europe-west2',
             'europe-west1', 'europe-west4', 'northamerica-northeast1']
  if dataflow_runner:
    return PipelineOptions([
      '--runner=DataflowRunner',
      '--project=' + PROJECT_ID,
      '--staging_location=gs://' + CLOUD_DATASET_BUCKET + '/' + CLOUD_STAGING_PREFIX,
      '--temp_location=gs://' + CLOUD_DATASET_BUCKET + '/' + CLOUD_TEMP_PREFIX,
      '--job_name=coco-localization-tfrecords-job{}'.format(job),
      '--region={}'.format(regions[job % len(regions)]),
      '--save_main_session',
      '--num_workers={}'.format(workers)
    ])
  else:
    return PipelineOptions(['--direct_num_workers', '{}'.format(workers)])


if __name__ == '__main__':
  # Defines arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('version', help='Folder where the raw tfrecords are', type=str)
  parser.add_argument('--dataflow', help='Whether to use DataflowRunner or not', action='store_true')
  parser.add_argument('--jobs', help='Jobs to work on dataset', type=int)
  parser.add_argument('--workers', help='Workers per job (VMs)', type=int)
  parser.add_argument('--load', help='Maximum number of records to load', type=int)
  parser.add_argument('--shards', help='Number of shards to generate', type=int)
  # Parses arguments
  args = parser.parse_args()
  version = args.version
  dataflow = args.dataflow if args.dataflow else False
  jobs = args.jobs if args.jobs else 1
  workers = args.workers if args.workers else 3
  load = args.load if args.load else None
  shards = args.shards if args.shards else 10
  # Create lists of records (one list per job)
  filenames = get_tfrecords_filenames_from_bucket(version) if dataflow else get_tfrecords_filenames_from_folder(version)
  filenames_lists = create_jobs_records_lists(filenames, jobs, load)
  # Creates a category to index dictionary and stores it in a text file
  category_to_index = create_categories_from_bucket(version) if dataflow else create_categories_from_folder(version)
  # Start all jobs
  for job, filenames in enumerate(filenames_lists):
    # Creates pipeline
    options = create_pipeline_options(job, workers, False)
    pipeline = beam.Pipeline(options=options)
    # Defines pipeline transformations
    records = (pipeline | beam.Create(filenames)
               | 'Read/Generate Records' >> beam.ParDo(ReadParseGenerateFn(get_load_records_path(version, dataflow),
                                                                           category_to_index))
               | "Save Records" >> beam.io.WriteToTFRecord(get_save_records_path(version, job, dataflow),
                                                           num_shards=shards))
    # Runs pipeline
    pipeline.run()
