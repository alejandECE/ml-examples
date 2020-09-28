#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import tensorflow as tf
import argparse
import pathlib
import utils
import visualization as viewer
import yolo

if __name__ == '__main__':
  # Allowing memory growth
  physical_devices = tf.config.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  # Defines arguments parser
  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint', help='Path to model to load', type=str)
  parser.add_argument('--samples', help='Max # of samples to keep in the dataset (Default: 1k)', type=int)
  # Parses arguments
  args = parser.parse_args()
  checkpoint_path = pathlib.Path(args.checkpoint)
  samples = args.samples if args.samples else 1000
  # Loading model
  model, _, _ = yolo.create_model()
  # Loads weights from checkpoint
  checkpoint = tf.train.Checkpoint(model=model)
  manager = tf.train.CheckpointManager(checkpoint, directory=str(checkpoint_path), max_to_keep=1)
  checkpoint.restore(manager.latest_checkpoint).expect_partial()
  if not manager.latest_checkpoint:
    print("No checkpoint found. Exiting!")
    exit()
  # Creates categories lookup tables
  category_to_index, index_to_category = utils.get_coco_categories_lookup_tables()
  # Creates datasets
  _, train_unprocessed_ds = yolo.create_training_dataset(samples, category_to_index, index_to_category)
  _, test_unprocessed_ds = yolo.create_test_dataset(samples, category_to_index, index_to_category)
  # Visualizes a detection example from each of the datasets
  viewer.visualize_detection_example(model, train_unprocessed_ds, index_to_category)
  # viewer.visualize_detection_example(model, test_unprocessed_ds, index_to_category)
