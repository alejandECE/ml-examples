#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import argparse
import collections
from typing import List, Tuple
import scipy.io
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import xml.etree.ElementTree as ET
import re
import tensorflow as tf
import os
import tempfile

# GCS credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'E:/secrets/ml-examples-ff91761a6a56.json'

# Path constants
PROJECT_ID = 'ml-examples-284704'
LOCAL_DATASET_BUCKET = 'E:/datasets/stanford_dogs'
CLOUD_DATASET_BUCKET = 'ml-examples-stanford-dogs'
ANNOTATION_PREFIX = 'annotations/'
IMAGES_PREFIX = 'images/'
CLOUD_STAGING_PREFIX = 'staging/'
CLOUD_TEMP_PREFIX = 'temp/'
CLOUD_RAW_RECORDS_PREFIX = 'raw_records/'
LOCAL_RAW_RECORDS_PREFIX = 'raw_records/'


# Regex to help build annotation path from image path
EXTRACT_ANNOTATION = re.compile(r"([\w-]*[/\\]*[\w]*).jpg")

# Bounding box helper namedtuple
BBox = collections.namedtuple('BBox', ['x', 'y', 'width', 'height'])

# Feature description to parse proto example
features_description = {
  'path': tf.io.FixedLenFeature([], tf.string),
  'image': tf.io.FixedLenFeature([], tf.string),
  'objects': tf.io.FixedLenFeature([], tf.int64),
  'label': tf.io.VarLenFeature(tf.string),
  'bbox': tf.io.VarLenFeature(tf.float32),
  'truncated': tf.io.VarLenFeature(tf.int64),
  'difficulty': tf.io.VarLenFeature(tf.int64)
}


# Gets bounding box from object node
def get_xml_object_bbox(node: ET.Element) -> BBox:
  xmin = int(node.find('bndbox/xmin').text)
  ymin = int(node.find('bndbox/ymin').text)
  xmax = int(node.find('bndbox/xmax').text)
  ymax = int(node.find('bndbox/ymax').text)
  return BBox(
    x=xmin,
    y=ymin,
    width=xmax - xmin,
    height=ymax - ymin
  )


# Normalizes bbox with values between 0 and 1 (based on image dimensions)
def normalize_bbox(box: BBox, image_width: int, image_height: int):
  return BBox(
    x=float(box.x) / image_width,
    y=float(box.y) / image_height,
    width=float(box.width) / image_width,
    height=float(box.height) / image_height
  )


# Returns image size as tuple of integers (width, height)
def get_xml_image_size(node: ET.Element) -> Tuple:
  return int(node.find('size/width').text), int(node.find('size/height').text)


# Returns all 'objects' (bounding boxes) in the xml file
def get_xml_objects_nodes(node: ET.Element) -> List[ET.Element]:
  return node.findall('object')


# Returns the name of an object node as string
def get_xml_object_name(node: ET.Element) -> str:
  return node.find('name').text


# Returns the truncated flag of an object node as boolean
def get_xml_object_truncated_flag(node: ET.Element) -> int:
  return int(node.find('truncated').text)


# Returns the difficult flag of an object node as boolean
def get_xml_object_difficult_flag(node: ET.Element) -> int:
  return int(node.find('difficult').text)


# Generates a tf.train.Example packing together image and annotation info
class GenerateExampleFn(beam.DoFn):
  def __init__(self, images_path: str, annotations_path):
    self.images_path = images_path
    self.annotations_path = annotations_path

  def process(self, element, *args, **kwargs):
    # Verifies the image file exists
    image_filepath = self.images_path + element
    if not tf.io.gfile.exists(image_filepath):
      return
    # Verifies annotation file exists
    # Extracts path of file without the jpg extension
    res = EXTRACT_ANNOTATION.match(element)
    if not res:
      return
    annotations_filepath = self.annotations_path + res.group(1)
    if not tf.io.gfile.exists(annotations_filepath):
      return
    # Reads data from image file serialized
    with tf.io.gfile.GFile(image_filepath, 'rb') as file:
      image_bytes = file.read()
    # Parses xml annotations file
    with tf.io.gfile.GFile(annotations_filepath, "rb") as file:
      root_node = ET.parse(file).getroot()
    # Goes through all objects in image (there might be more than one)
    # We also normalize the bounding boxes dimensions from 0 to 1
    width, height = get_xml_image_size(root_node)
    object_nodes = get_xml_objects_nodes(root_node)
    # Counts how many boxes we have
    objects_count = len(object_nodes)
    # Create lists to hold multiple boxes info
    label_features = []
    bbox_features = []
    truncated_features = []
    difficulty_features = []
    for node in object_nodes:
      label_features.append(get_xml_object_name(node).encode())
      bbox_features.extend(normalize_bbox(get_xml_object_bbox(node), width, height))
      truncated_features.append(get_xml_object_truncated_flag(node))
      difficulty_features.append(get_xml_object_difficult_flag(node))
    # Creates a features dictionary with their corresponding values
    features = tf.train.Features(feature={
      'path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[element.encode()])),
      'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
      'objects': tf.train.Feature(int64_list=tf.train.Int64List(value=[objects_count])),
      'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=label_features)),
      'bbox': tf.train.Feature(float_list=tf.train.FloatList(value=bbox_features)),
      'truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated_features)),
      'difficulty': tf.train.Feature(int64_list=tf.train.Int64List(value=difficulty_features))
    })
    # Create a tf.train.Example
    yield tf.train.Example(features=features).SerializeToString()


# Loads files list from a folder locally
def load_file_list_from_folder(version: str) -> list:
  # Generates path to file
  file = LOCAL_DATASET_BUCKET + '/' + version
  # Gets list of files from .mat available. We could also walk the directory but this way is easier!
  data = scipy.io.loadmat(str(file))
  # Files are listed in the following format: subfolder(class)/filename
  return [entry[0][0] for entry in data['file_list']]


# Loads files list from a cloud bucket
def load_file_list_from_bucket(version: str):
  # Lazy import
  from google.cloud import storage
  # Instantiate a gc storage client and specify required bucket and file
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(CLOUD_DATASET_BUCKET)
  blob = bucket.blob(version)
  # Download the contents of the blob to a temporaty file
  fd, path = tempfile.mkstemp()
  with os.fdopen(fd, 'wb+') as file:
    blob.download_to_file(file)
  # Gets list of files from temporary file
  data = scipy.io.loadmat(path)
  # Files are listed in the following format: subfolder(class)/filename
  return [entry[0][0] for entry in data['file_list']]


# Generates lists of files (one per job)
def create_jobs_data(version, jobs: int, dataflow_runner=False, load=None) -> list:
  # Loads metadata from cloud or locally
  if dataflow_runner:
    files = load_file_list_from_bucket(version)
  else:
    files = load_file_list_from_folder(version)
  # Assigns number of files per job
  files_count = len(files) if load is None or load > len(files) else load
  files_per_job = [files_count // jobs] * jobs
  for i in range(files_count % jobs):
    files_per_job[i] += 1
  # Create lists of records (one list per job)
  files_lists = []
  start = 0
  for count in files_per_job:
    # Creates a sublist of records for the current job
    files_lists.append([record for record in files[start: start + count]])
    start += count
  return files_lists


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
      '--job_name=stanford-raw-tfrecords-job{}'.format(job),
      '--region={}'.format(regions[job % len(regions)]),
      '--save_main_session',
      '--num_workers={}'.format(workers)
    ])
  else:
    return PipelineOptions(['--direct_num_workers', '{}'.format(workers)])


# Get path to images
def get_images_path(dataflow_runner=False):
  if dataflow_runner:
    return 'gs://' + CLOUD_DATASET_BUCKET + '/' + IMAGES_PREFIX
  else:
    return LOCAL_DATASET_BUCKET + '/' + IMAGES_PREFIX


# Get path to images
def get_annotations_path(dataflow_runner=False):
  if dataflow_runner:
    return 'gs://' + CLOUD_DATASET_BUCKET + '/' + ANNOTATION_PREFIX
  else:
    return LOCAL_DATASET_BUCKET + '/' + ANNOTATION_PREFIX


# Creates path to store the records
def get_records_path(version: str, job: int, dataflow_runner=False):
  if dataflow_runner:
    return 'gs://' + CLOUD_DATASET_BUCKET + '/' + CLOUD_RAW_RECORDS_PREFIX + 'data-job{:}.tfrecord'.format(job)
  else:
    return LOCAL_DATASET_BUCKET + '/' + LOCAL_RAW_RECORDS_PREFIX + version + '/data-job{:}.tfrecord'.format(job)


if __name__ == '__main__':
  # Defines arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('version', help='Filename containing the list of files to generate raw records for', type=str)
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
  shards = args.shards if args.shards else 5
  load = args.load if args.load else None
  workers = args.workers if args.workers else 3
  # Generates lists of files (one per job)
  files_lists = create_jobs_data(version, jobs, dataflow, load)
  # Start all jobs
  for job, files in enumerate(files_lists):
    # Creates pipeline
    options = create_pipeline_options(job, workers, dataflow)
    pipeline = beam.Pipeline(options=options)
    # Creates pcollection with loaded images (keyed by path)
    records = (pipeline
              | "Lists Files" >> beam.Create(files)
              | "Generate Examples" >> beam.ParDo(GenerateExampleFn(get_images_path(dataflow),
                                                                    get_annotations_path(dataflow))))
    # Stores examples
    records | "Store Examples" >> beam.io.WriteToTFRecord(get_records_path(version, job, dataflow), num_shards=shards)
    # Runs pipeline
    pipeline.run()
