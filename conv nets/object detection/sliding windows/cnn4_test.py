#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import cnn4_training as cnn4


if __name__ == '__main__':
  # Verifying correct # of parameters
  if len(sys.argv) < 2:
    exit()
  # Allowing memory growth
  physical_devices = tf.config.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  # Loads image
  test_img = Image.open(sys.argv[1])
  # Parse arguments and crops if needed
  if len(sys.argv) > 4:
    x = int(sys.argv[2])
    y = int(sys.argv[3])
    width = int(sys.argv[4])
    height = int(sys.argv[5])
    test_img = test_img.crop((x, y, x + width, y + height))
  # Loads trained model
  model = tf.keras.models.load_model('trained_model/cnn4/20200716-145223')
  # Test the model with the test image
  plt.imshow(test_img)
  # Prepares for model evaluation
  inputs = tf.expand_dims(cnn4.preprocess(tf.convert_to_tensor(np.array(test_img))), axis=0)
  # Evaluates image
  plt.title('Dog (Probability): {}'.format(
    np.squeeze(model.predict(inputs))
  ))
  plt.show()
