#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cnn3_training as cnn3
import argparse
import utils


if __name__ == '__main__':
  # Allowing memory growth
  physical_devices = tf.config.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  # Defining arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('path', help='Path to image (relative to $DATASETS)', type=str)
  parser.add_argument('model', help='Path to model (relative the $OUTPUT/PATH_PREFIX)', type=str)
  parser.add_argument('-c', '--crop', nargs='*', help='x, y, width and height of the crop!', type=float)
  # Parses arguments
  args = parser.parse_args()
  # Loads image
  test_img = Image.open(utils.DATASETS / args.path)
  # Crops if needed
  if args.crop:
    x, y, width, height = args.crop
    test_img = test_img.crop((x, y, x + width, y + height))
  # Loads trained model
  model = tf.keras.models.load_model(utils.OUTPUTS / cnn3.PATH_PREFIX / args.model)
  # Test the model with the test image
  plt.imshow(test_img)
  # Prepares for model evaluation
  inputs = tf.expand_dims(cnn3.preprocess(tf.convert_to_tensor(np.array(test_img))), axis=0)
  # Evaluates image
  plt.title('Dog (Probability): {}'.format(
    np.squeeze(model.predict(inputs))
  ))
  plt.show()
