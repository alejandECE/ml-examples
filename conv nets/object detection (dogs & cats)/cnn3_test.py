#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys

# Some constants & setups
INPUT_HEIGHT = 32
INPUT_WIDTH = 32


# Crops the image in a given area and resizes it to the size of the smallest window (INPUT_WIDTH, INPUT_HEIGHT)
def preprocess(img) -> np.ndarray:
  img = np.array(img.resize((INPUT_WIDTH, INPUT_HEIGHT), resample=1)) / 255.0
  img = np.expand_dims(img, axis=0)
  return img


if __name__ == '__main__':
  if len(sys.argv) < 2:
    exit()
  path = sys.argv[1]
  # Loads image
  test_img = Image.open(path)
  # Parse arguments and crops if needed
  if len(sys.argv) > 4:
    x = int(sys.argv[2])
    y = int(sys.argv[3])
    width = int(sys.argv[4])
    height = int(sys.argv[5])
    test_img = test_img.crop((x, y, x + width, y + height))
  # Loads trained model
  model = tf.keras.models.load_model('trained_model/cnn3/')
  # Test the model with the test image
  plt.imshow(test_img)
  plt.title('Dog (Probability): {}'.format(
    np.squeeze(model.predict(preprocess(test_img)))
  ))
  plt.show()
