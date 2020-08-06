#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import apache_beam as beam
import tensorflow as tf
import argparse
from apache_beam.options.pipeline_options import PipelineOptions
import os
import pathlib
import collections

# GCS credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'E:/secrets/ml-examples-ff91761a6a56.json'

# Path constants
PROJECT_ID = 'ml-examples-284704'
CLOUD_STAGING_PREFIX = 'staging/'
CLOUD_TEMP_PREFIX = 'temp/'
LOCAL_DATASET_BUCKET = 'E:/datasets/stanford_dogs'
CLOUD_DATASET_BUCKET = 'ml-examples-stanford-dogs'
CLOUD_LOCALIZATION_RECORDS_PREFIX = 'localization_records/'
LOCAL_LOCALIZATION_RECORDS_PREFIX = 'localization_records/'
CLOUD_RAW_RECORDS_PREFIX = 'raw_records/'
LOCAL_RAW_RECORDS_PREFIX = 'raw_records/'

# Feature description to parse input proto example
inputs_description = {
  'path': tf.io.FixedLenFeature([], tf.string),
  'image': tf.io.FixedLenFeature([], tf.string),
  'objects': tf.io.FixedLenFeature([], tf.int64),
  'label': tf.io.VarLenFeature(tf.string),
  'bbox': tf.io.VarLenFeature(tf.float32),
  'truncated': tf.io.VarLenFeature(tf.int64),
  'difficulty': tf.io.VarLenFeature(tf.int64)
}

# Feature description to parse output proto example
outputs_description = {
  'path': tf.io.FixedLenFeature([], tf.string),
  'image': tf.io.FixedLenFeature([], tf.string),
  'label': tf.io.FixedLenFeature([], tf.string),
  'bbox': tf.io.VarLenFeature(tf.float32)
}

# Bounding box helper namedtuple
BBox = collections.namedtuple('BBox', ['x', 'y', 'width', 'height'])


# Creates path to store the records
def get_save_records_path(version: str, job: int, dataflow_runner=False):
  if dataflow_runner:
    return 'gs://' + CLOUD_DATASET_BUCKET + '/' + CLOUD_LOCALIZATION_RECORDS_PREFIX + \
           'data-job{:}.tfrecord'.format(job)
  else:
    return LOCAL_DATASET_BUCKET + '/' + LOCAL_LOCALIZATION_RECORDS_PREFIX + version + \
           '/data-job{:}.tfrecord'.format(job)


def get_load_records_path(version: str, dataflow_runner=False):
  if dataflow_runner:
    return 'gs://' + CLOUD_DATASET_BUCKET + '/' + CLOUD_RAW_RECORDS_PREFIX
  else:
    return LOCAL_DATASET_BUCKET + '/' + LOCAL_RAW_RECORDS_PREFIX + version + '/'


# Distributes filenames across jobs (returning one list per job)
def create_records_lists(filenames: list, jobs: int, load: int):
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


# Returns all tfrecord filenames found in the bucket
def get_tfrecords_filenames_from_bucket():
  # Lazy import
  from google.cloud import storage
  # Instantiate a gc storage client
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(CLOUD_DATASET_BUCKET)
  blobs = bucket.list_blobs(prefix=CLOUD_RAW_RECORDS_PREFIX)
  # Filters all blobs corresponding to tfrecord files
  filenames = [blob.name.split('/')[-1] for blob in blobs if 'tfrecord' in blob.name]
  return filenames


# Returns all tfrecord filenames found in the folder
def get_tfrecords_filenames_from_folder(version: str):
  folder = pathlib.Path(LOCAL_DATASET_BUCKET) / LOCAL_RAW_RECORDS_PREFIX / version
  blobs = folder.glob('*.tfrecord*')
  return [blob.name for blob in blobs]


# DoFn class to read/parse raw records and convert to localization records
class ReadParseGenerateFn(beam.DoFn):
  def __init__(self, root: str):
    self.root = root

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
      # Parses example
      example = tf.io.parse_single_example(entry, features=inputs_description)
      # Parses bboxes
      num_objects = example['objects']
      boxes = tf.reshape(tf.sparse.to_dense(example['bbox']), shape=[num_objects, 4])
      labels = tf.sparse.to_dense(example['label'])
      # Only keeps the first dog box found
      features = tf.train.Features(feature={
        'path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[example['path'].numpy()])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[example['image'].numpy()])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels[0].numpy()])),
        'bbox': tf.train.Feature(float_list=tf.train.FloatList(value=boxes[0, :].numpy()))
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
      '--job_name=stanford-localization-tfrecords-job{}'.format(job),
      '--region={}'.format(regions[job % len(regions)]),
      '--save_main_session',
      '--num_workers={}'.format(workers)
    ])
  else:
    return PipelineOptions(['--direct_num_workers', '{}'.format(workers)])


if __name__ == '__main__':
  # Defines arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('version', help='Folder name containing the list of files where the raw records are', type=str)
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
  if dataflow:
    filenames_lists = create_records_lists(get_tfrecords_filenames_from_bucket(), jobs, load)
  else:
    filenames_lists = create_records_lists(get_tfrecords_filenames_from_folder(version), jobs, load)
  # Start all jobs
  for job, filenames in enumerate(filenames_lists):
    # Creates pipeline
    options = create_pipeline_options(job, workers, dataflow)
    pipeline = beam.Pipeline(options=options)
    # Defines pipeline transformations
    records = (pipeline | beam.Create(filenames)
               | 'Read/Generate Records' >> beam.ParDo(ReadParseGenerateFn(get_load_records_path(version, dataflow)))
               | "Save Records" >> beam.io.WriteToTFRecord(get_save_records_path(version, job, dataflow),
                                                           num_shards=shards))
    # Runs pipeline
    pipeline.run()
