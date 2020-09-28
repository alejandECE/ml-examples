import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import patches
import numpy as np
import sys
sys.path.append('../../records generators')
import coco_localization_tfrecords_generator as coco
import dogs_localization as dogs

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224

if __name__ == '__main__':
  # Allowing memory growth
  physical_devices = tf.config.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  # Loading model
  model, _, _, _ = dogs.create_model(transferred=True, localization=True)
  # Loads weights from checkpoint
  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint_path = pathlib.Path('trainings/20200813-114319/checkpoints')
  manager = tf.train.CheckpointManager(checkpoint, directory=str(checkpoint_path), max_to_keep=1)
  checkpoint.restore(manager.latest_checkpoint).expect_partial()
  if not manager.latest_checkpoint:
    print("No checkpoint found!")
  # Loads image
  image_path = pathlib.Path('../object detection/sliding windows/images/search one dog 1.jpg')
  test_img = Image.open(image_path)
  # Preprocess and runs image through the model
  inputs = dogs.transferred_preprocess(tf.expand_dims(tf.convert_to_tensor(np.array(test_img)), axis=0))
  outputs = tf.squeeze(model(inputs)).numpy()
  # Plotting result
  fig, ax = plt.subplots()
  # ax.imshow(test_img.resize((IMG_WIDTH, IMG_HEIGHT)))
  ax.imshow(np.squeeze(inputs.numpy()))
  box = coco.BBox(*outputs[1:])
  patch = patches.Rectangle((box.x * IMG_WIDTH, box.y * IMG_HEIGHT), box.width * IMG_WIDTH,
                            box.height * IMG_HEIGHT,
                            edgecolor='red',
                            facecolor='none',
                            lw=2)
  ax.add_patch(patch)
  ax.set_title(outputs[0])
  plt.show()
