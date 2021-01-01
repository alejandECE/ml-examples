#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import argparse
import pathlib
from typing import Tuple
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from bbox import BoundingBox
from matplotlib.colors import ListedColormap
import utils

# Constants
INPUT_HEIGHT = 32
INPUT_WIDTH = 32


# Preprocess image to be fed to the CNN
def preprocess(img: np.ndarray) -> tf.Tensor:
  img = tf.convert_to_tensor(img, dtype=tf.float32)
  img = img / 255.0
  img = tf.expand_dims(img, axis=0)
  return img


class DogsDetector:
  def __init__(self, model_path: pathlib.Path, preprocess_fnc, window_step, window_size=None, threshold=.7):
    """
    Creates a dog detector using convolutional sliding windows. The performance of this model is way superior to the
    regular implementation of sliding windows. However, as you increase the side of the window Steps for bigger windows
    the step is always a factor of the size times the original step size of the CNN. For instance:

    Assume a CNN trained on 32x32 pictures outputting one value (dog probability) with 3 pooling layers of size 2x2.
    On a bigger image this is equivalent to a window step size of 8. Therefore, window size 32 and step size 8 are the
    standard of this network. If we now want a window size of 128 = 4 * 32 then the step size will be 32 = 4 * 8.

    :param model_path: Path to the CNN stored as *.pb model.
    :param preprocess_fnc: Function reference to perform preprocessing
    :param window_step: Corresponding step used by the CNN (multiply all pooling layers sizes).
    :param window_size: List of window sizes (factor of the original image size fed to CNN) to try out.
    :param threshold: Probability threshold to assume a dog has been detected.
    """
    # Image to detect dogs
    self.image = None
    # Windows detected and threshold used
    self.dogs = None
    self.threshold = threshold
    # Loads models and pre-processing function
    self.model = tf.keras.models.load_model(str(model_path))
    self.preprocess_fnc = preprocess_fnc
    # Window size and step (this is the step of the original CNN)
    self.window_step = window_step
    self.window_size = [1, 2, 3] if window_size is None else window_size

  def find_dogs(self, image_path: pathlib.Path):
    """
    Search for dogs in the specified image using a convolutional implementation of sliding windows.

    :param image_path: Path to the image to evaluate.
    """
    # Loads image
    self.image = Image.open(image_path)
    # Calls CNN resizing the image before to produce the effect of a bigger window.
    # Steps for bigger windows are always a factor of the size times the original step size of the CNN.
    outputs = []
    for size in self.window_size:
      resized_image = np.array(self.image.resize(
        (int(self.image.width / size), int(self.image.height / size)),
        resample=1)
      )
      outputs.append(self.model.predict(self.preprocess_fnc(resized_image)))
    # Builds windows from CNN outputs
    windows = self.__build_windows(outputs)
    # Reduces the number of windows detected
    self.dogs = self.__filter_windows(windows)
    # Plots final windows
    self.__plot_windows(self.dogs)

  # Builds bounding boxes in the original image coordinates systems from the CNN outputs
  def __build_windows(self, outputs: np.ndarray):
    windows = []
    for size, values in zip(self.window_size, outputs):
      values = np.squeeze(values)
      rows, cols = values.shape
      for i in range(rows):
        for j in range(cols):
          if values[i, j] > self.threshold:
            windows.append((
              BoundingBox(j * size * self.window_step,
                          i * size * self.window_step,
                          size * INPUT_WIDTH,
                          size * INPUT_HEIGHT),
              size,
              values[i, j]
            ))
    return windows

  def __filter_windows(self, windows: Tuple[BoundingBox, float, float]):
    return windows

  # Plot bounding boxes passed
  def __plot_windows(self, windows: Tuple[BoundingBox, float, float]):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(self.image)
    cmap = ListedColormap(['r', 'b', 'm', 'k', 'w'], N=len(self.window_size))
    colors = {size: cmap(i) for i, size in enumerate(self.window_size)}
    for box, size, _ in windows:
      ax.add_patch(box.to_rectangle(colors[size]))
    plt.show()


if __name__ == '__main__':
  # Allowing memory growth
  physical_devices = tf.config.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  # Defines arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('path', help='Path to image (relative to $DATASETS)', type=str)
  parser.add_argument('model', help='Path to model (relative the $OUTPUT)', type=str)
  # Parses arguments
  args = parser.parse_args()
  # Creates sliding window detector
  detector = DogsDetector(utils.OUTPUTS / args.model, preprocess, window_step=8,
                          window_size=[1, 2, 4, 6, 8], threshold=0.90)
  # Finds dogs in image
  detector.find_dogs(utils.DATASETS / args.path)
