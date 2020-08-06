#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import argparse
import json
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import tensorflow as tf
import collections
import os

# GCS credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'E:/secrets/ml-examples-ff91761a6a56.json'

# Path constants
PROJECT_ID = 'ml-examples-284704'
LOCAL_DATASET_BUCKET = 'E:/datasets/coco'
CLOUD_DATASET_BUCKET = 'ml-examples-coco'
ANNOTATION_PREFIX = 'annotations/'
IMAGES_PREFIX = 'images/'
CLOUD_STAGING_PREFIX = 'staging/'
CLOUD_TEMP_PREFIX = 'temp/'
CLOUD_RAW_RECORDS_PREFIX = 'raw_records/'
LOCAL_RAW_RECORDS_PREFIX = 'raw_records/'

# Shards constants
MIN_SHARD_SIZE = 64 << 20  # 64 MiB
MAX_SHARD_SIZE = 1024 << 20  # 2 GiB
TFRECORD_REC_OVERHEAD = 16

# Bounding box helper namedtuple
BBox = collections.namedtuple('BBox', ['x', 'y', 'width', 'height'])

# Feature description to parse proto example
features_description = {
  'id': tf.io.FixedLenFeature([], tf.int64),
  'path': tf.io.FixedLenFeature([], tf.string),
  'image': tf.io.FixedLenFeature([], tf.string),
  'objects': tf.io.FixedLenFeature([], tf.int64),
  'label': tf.io.VarLenFeature(tf.string),
  'bbox': tf.io.VarLenFeature(tf.float32)
}


# Normalizes bbox with values between 0 and 1 (based on image dimensions)
def normalize_bbox(box: BBox, image_width: int, image_height: int):
  return BBox(
    x=float(box.x) / image_width,
    y=float(box.y) / image_height,
    width=float(box.width) / image_width,
    height=float(box.height) / image_height
  )


# Reads image from file as a byte string
class ReadImageFn(beam.DoFn):
  def __init__(self, root: str):
    self.root = root

  def process(self, element, *args, **kwargs):
    image_id, image_data = element
    # Verifies the file exists
    path = self.root + image_data['file_name']
    if not tf.io.gfile.exists(path):
      return
    # Reads data from image file in serialized format
    with tf.io.gfile.GFile(path, 'rb') as file:
      image_bytes = file.read()
    yield image_id, {
      'path': image_data['file_name'],
      'bytes': image_bytes,
      'width': image_data['width'],
      'height': image_data['height']
    }


# Generates a tf.train.Example packing together image and annotation info
class GenerateExampleFn(beam.DoFn):
  def process(self, element, *args, **kwargs):
    # Unpacks element
    image_id, (image_data, image_metadata) = element
    # Verifies both image data and metadata are available for given id
    if len(image_data) == 0 or len(image_metadata) == 0:
      return
    # Counts how many boxes we have
    objects_count = len(image_metadata[0])
    # Create two lists (labels and boxes)
    label_features = []
    bbox_features = []
    for annotation in image_metadata[0]:
      label_features.append(annotation['category'].encode())
      bbox_features.extend(normalize_bbox(annotation['bbox'], image_data[0]['width'], image_data[0]['height']))
    # Creates a features dictionary with their corresponding values
    features = tf.train.Features(feature={
      'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_id])),
      'path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data[0]['path'].encode()])),
      'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data[0]['bytes']])),
      'objects': tf.train.Feature(int64_list=tf.train.Int64List(value=[objects_count])),
      'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=label_features)),
      'bbox': tf.train.Feature(float_list=tf.train.FloatList(value=bbox_features))
    })
    # Create a tf.train.Example
    yield tf.train.Example(features=features).SerializeToString()


# Returns the optimal number of shards given the total size and # of examples
def get_number_shards(total_size: int, num_examples: int) -> int:
  total_size += num_examples * TFRECORD_REC_OVERHEAD
  max_shards_number = total_size // MIN_SHARD_SIZE
  min_shards_number = total_size // MAX_SHARD_SIZE
  if min_shards_number <= 1024 <= max_shards_number and num_examples >= 1024:
    return 1024
  elif min_shards_number > 1024:
    i = 2
    while True:
      n = 1024 * i
      if min_shards_number <= n <= num_examples:
        return n
      i += 1
  else:
    for n in [512, 256, 128, 64, 32, 16, 8, 4, 2]:
      if min_shards_number <= n <= max_shards_number and num_examples >= n:
        return n
  return 1


# Returns metadata loaded from local json file
def load_metadata_from_folder(metadata_file: str):
  with open(LOCAL_DATASET_BUCKET + "/" + ANNOTATION_PREFIX + metadata_file) as file:
    return json.load(file)


# Loads metadata from json file in the cloud (this is not needed but allows you to have everything in the cloud)
def load_metadata_from_bucket(metadata_file: str):
  # Lazy import
  from google.cloud import storage
  # Instantiate a gc storage client and specify required bucket and file
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(CLOUD_DATASET_BUCKET)
  blob = bucket.blob(ANNOTATION_PREFIX + metadata_file)
  # Download the contents of the blob as a string and then parse it using json.loads() method
  return json.loads(blob.download_as_string(client=None))


# Create all lists necessary to initialize pipeline
def create_jobs_data(version: str, jobs: int, dataflow_runner=False, load=None):
  # This is the expected json name based on version
  json_name = 'instances_' + version + '.json'
  # Loads metadata from cloud or locally (json)
  if dataflow_runner:
    metadata = load_metadata_from_bucket(json_name)
  else:
    metadata = load_metadata_from_folder(json_name)
  # Creates list of categories mapping index to label
  index_to_label = {entry['id']: entry['name'] for entry in metadata['categories']}
  categories = list(index_to_label.items())
  # We split images for different jobs!
  # Creates a dictionary per job that maps image_id to all its metadata (without loading image from file yet)
  images_count = len(metadata['images']) if load is None or load > len(metadata['images']) else load
  images_per_job = [images_count // jobs] * jobs
  for i in range(images_count % jobs):
    images_per_job[i] += 1
  # Builds images data per job
  jobs_data = []
  jobs_annotations = []
  start = 0
  for count in images_per_job:
    # Creates dictionary mapping image id to the rest of the data
    # Bounding boxes are not initialized yet
    image_to_data = {
      image['id']: {
        'file_name': image['file_name'],
        'width': image['width'],
        'height': image['height'],
      } for image in metadata['images'][start: start + count]
    }
    # Includes bounding boxes information
    image_to_annotations = collections.defaultdict(list)
    for annotation in metadata['annotations']:
      if annotation['image_id'] in image_to_data:
        image_to_annotations[annotation['image_id']].append({
          'category': index_to_label[annotation['category_id']],
          'bbox': BBox(*annotation['bbox'])
        })
    # Stores data for the current job
    jobs_data.append(image_to_data)
    jobs_annotations.append(image_to_annotations)
    # Updates where next job should start
    start += count
  return jobs_data, jobs_annotations, categories


# Creates pipeline options according to runner selected
def create_pipeline_options(job, workers, dataflow_runner=False):
  # Allowed dataflow regions
  regions = ['us-west1', 'us-central1', 'us-east4', 'northamerica-northeast1', 'us-east1', 'europe-west2',
             'europe-west1', 'europe-west4']
  if dataflow_runner:
    return PipelineOptions([
      '--runner=DataflowRunner',
      '--project=' + PROJECT_ID,
      '--staging_location=gs://' + CLOUD_DATASET_BUCKET + '/' + CLOUD_STAGING_PREFIX,
      '--temp_location=gs://' + CLOUD_DATASET_BUCKET + '/' + CLOUD_TEMP_PREFIX,
      '--job_name=coco-raw-tfrecords-job{}'.format(job),
      '--region={}'.format(regions[job % len(regions)]),
      '--save_main_session',
      '--num_workers={}'.format(workers)
    ])
  else:
    return PipelineOptions(['--direct_num_workers', '{}'.format(workers)])


# Creates path to store the records
def get_records_path(version: str, job: int, dataflow_runner=False):
  if dataflow_runner:
    return 'gs://' + CLOUD_DATASET_BUCKET + '/' + CLOUD_RAW_RECORDS_PREFIX + 'data-job{:}.tfrecord'.format(job)
  else:
    return LOCAL_DATASET_BUCKET + '/' + LOCAL_RAW_RECORDS_PREFIX + version + '/data-job{:}.tfrecord'.format(job)


# Creates path to store the list of categories
def get_categories_path(version: str, dataflow_runner=False):
  if dataflow_runner:
    return 'gs://' + CLOUD_DATASET_BUCKET + '/' + CLOUD_RAW_RECORDS_PREFIX + 'categories.txt'
  else:
    return LOCAL_DATASET_BUCKET + '/' + LOCAL_RAW_RECORDS_PREFIX + version + '/categories.txt'


# Get path to images
def get_images_path(version: str, dataflow_runner=False):
  if dataflow_runner:
    return 'gs://' + CLOUD_DATASET_BUCKET + '/' + IMAGES_PREFIX + version + '/'
  else:
    return LOCAL_DATASET_BUCKET + '/' + IMAGES_PREFIX + version + '/'


if __name__ == '__main__':
  # Defines arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('version', help='Path to the images folder version (e.g. val2014)', type=str)
  parser.add_argument('--dataflow', help='Whether to use DataflowRunner or not', action='store_true')
  parser.add_argument('--jobs', help='Jobs to work on dataset', type=int)
  parser.add_argument('--shards', help='Number of shards to generate', type=int)
  parser.add_argument('--load', help='Maximum number of images to load', type=int)
  parser.add_argument('--workers', help='Workers per job (VMs)', type=int)
  # Parses arguments
  args = parser.parse_args()
  version = args.version
  dataflow = args.dataflow if args.dataflow else False
  jobs = args.jobs if args.jobs else 1
  shards = args.shards if args.shards else 20
  load = args.load if args.load else None
  workers = args.workers if args.workers else 3
  # Loads jobs metadata
  jobs_data, jobs_annotations, categories_list = create_jobs_data(version, jobs, dataflow_runner=dataflow, load=load)
  # Start all jobs
  for job, (data, annotations) in enumerate(zip(jobs_data, jobs_annotations)):
    # Creates pipeline
    options = create_pipeline_options(job, workers, dataflow)
    pipeline = beam.Pipeline(options=options)
    # Creates pcollection with the loaded images data (keyed by image id)
    images = (pipeline
              | "Create Images List" >> beam.Create(data)
              | "Read Image" >> beam.ParDo(ReadImageFn(get_images_path(version, dataflow))))
    # Creates pcollection with annotation info (keyed by path)
    annotations = pipeline | "Create Annotations List" >> beam.Create(annotations)
    # Groups image and annotation by key and generates pcollection of tf.train.Example
    records = ((images, annotations)
               | "Combine Annotations/Image" >> beam.CoGroupByKey()
               | "Generate Example" >> beam.ParDo(GenerateExampleFn()))
    # Only first job saves categories
    if job == 0:
      categories = (pipeline
                    | "Create Categories File" >> beam.Create(categories_list)
                    | "Save Categories" >> beam.io.WriteToText(get_categories_path(version, dataflow),
                                                               num_shards=1,
                                                               shard_name_template=''))
    # Stores examples
    records | "Save Records" >> beam.io.WriteToTFRecord(get_records_path(version, job, dataflow), num_shards=shards)
    # Runs pipeline
    pipeline.run()
