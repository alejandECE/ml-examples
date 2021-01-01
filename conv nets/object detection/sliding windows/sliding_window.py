#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import argparse
import pathlib
from typing import Tuple
from PIL import Image
import matplotlib.pyplot as plt
from bbox import BoundingBox
from matplotlib.colors import ListedColormap
import numpy as np
import tensorflow as tf
import math
import utils

# Constants
INPUT_HEIGHT = 32
INPUT_WIDTH = 32


# Preprocess image to be fed to CNN
def preprocess(img: np.ndarray) -> tf.Tensor:
  img = tf.convert_to_tensor(img, dtype=tf.float32)
  img = img / 255.0
  img = tf.expand_dims(img, axis=0)
  return img


class DogsDetector:
  def __init__(self, model_path: pathlib.Path, preprocess_fnc, window_size=None, window_step=None, threshold=.7):
    """
    Creates a dog detector.

    :param model_path: Path to the CNN stored as *.pb model.
    :param preprocess_fnc: Function reference to perform preprocessing
    :param window_size: List of window sizes to try out given as multiples of INPUT_HEIGHT and INPUT_WIDTH.
    :param window_step: Step size (Sliding step)
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
    # Window size and step
    if window_size is None:
      self.window_size = [1]
    if window_step is None:
      self.window_step = 8

  # Initializes the progress bar
  def __setup_progress(self, total_windows: float) -> None:
    self.__current_window = 0
    self.__total_windows = total_windows
    self.__total_length = 20
    print('[', ' ' * self.__total_length, ']: {:3.0f}%'.format(0), end='', sep='')

  # Updates the progress bar
  def __update_progress(self) -> None:
    self.__current_window += 1
    ratio = min(1., self.__current_window / self.__total_windows)
    current_length = int(ratio * self.__total_length)
    print('\b' * (self.__total_length + 9), end='')
    print('[', '=' * current_length, '>', ' ' * (self.__total_length - current_length),
          ']: {:3.0f}%'.format(ratio * 100), end='', sep='')

  def find_dogs(self, image_path: pathlib.Path) -> None:
    """
    Search for dogs in the specified image using a convolutional implementation of sliding windows.

    :param image_path: Path to the image to evaluate.
    """
    # Loads image
    self.image = Image.open(image_path)
    # Pre-computes approximately the total number of windows
    total_windows = sum([
      math.ceil((self.image.width - int(INPUT_WIDTH * size) + 1) / self.window_step) *
      math.ceil((self.image.height - int(INPUT_HEIGHT * size) + 1) / self.window_step)
      for size in self.window_size
    ])
    # Goes through every window detecting if there is a dog in the window
    self.__setup_progress(total_windows)
    windows = []
    for size in self.window_size:
      resized_image = np.array(self.image.resize(
        (int(self.image.width / size), int(self.image.height / size)),
        resample=1)
      )
      height, width, _ = resized_image.shape
      for i in range(0, height - INPUT_HEIGHT + 1, max(1, int(self.window_step / size))):
        for j in range(0, width - INPUT_WIDTH + 1, max(1, int(self.window_step / size))):
          cropped_image = resized_image[i:i + INPUT_HEIGHT, j:j + INPUT_WIDTH, :]
          probability = self.model.predict(self.preprocess_fnc(cropped_image))[0][0]
          # Window is consider only when the probability is higher than some threshold
          if probability >= self.threshold:
            windows.append((
              BoundingBox(j * size,
                          i * size,
                          size * INPUT_WIDTH,
                          size * INPUT_HEIGHT),
              size,
              probability
            ))
          self.__update_progress()
    # Reduces the number of windows detected
    self.dogs = self.__filter_windows(windows)
    # Plots final windows
    self.__plot_windows(self.dogs)

  def __filter_windows(self, windows: Tuple[BoundingBox, float, float]) -> Tuple:
    return windows

  # Plot bounding boxes passed
  def __plot_windows(self, windows: Tuple[BoundingBox, float, float]) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(self.image)
    cmap = ListedColormap(['r', 'b', 'm'], N=len(self.window_size))
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
  detector = DogsDetector(utils.OUTPUTS / args.model, preprocess, window_size=[1, 2], window_step=8, threshold=0.85)
  # Finds dogs in image
  detector.find_dogs(utils.DATASETS / args.path)
