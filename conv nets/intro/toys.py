#  Created by Luis Alejandro (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import numpy as np


# Toy implementation of a 2D convolution
def conv_2d(image, kernel, stride=1, mode='same'):
  # Image and kernel dimensions
  image_h, image_w = image.shape
  kernel_h, kernel_w = kernel.shape

  # Check valid kernel dimensions are provided
  if image_h < kernel_h or image_w < kernel_w:
    return
  if stride > kernel_w or stride > kernel_h:
    return

  # Padding if 'same' is selected
  if mode == 'same' and stride == 1:
    image = np.pad(image,
                   ((kernel_h//2, kernel_h//2), (kernel_w//2, kernel_w//2)),
                   mode='constant')
    image_h, image_w = image.shape

  # Performing convolution
  cols = np.arange(0, image_w - kernel_w + 1, stride)
  rows = np.arange(0, image_h - kernel_h + 1, stride)
  output = np.zeros((len(rows), len(cols)))
  for i, row in enumerate(rows):
    for j, col in enumerate(cols):
      output[i, j] = (image[row:row + kernel_h, col:col + kernel_w] * kernel).sum()

  return output


# Toy implementation of a 2D max pooling
def max_pooling_2d(image, pool, stride=-1, mode='same'):
  # Image and pooling dimensions
  image_h, image_w = image.shape
  pool_h, pool_w = pool[0], pool[1]

  # Check valid kernel dimensions are provided
  if image_h < pool_h or image_w < pool_w:
    return
  if stride > pool_w or stride > pool_h:
    return

  # Default option
  if stride == -1:
    stride = pool_w

  # Configuration
  if mode == 'same':
    cols = np.arange(0, image_w, stride)
    rows = np.arange(0, image_h, stride)
  elif mode == 'valid':
    cols = np.arange(0, image_w - pool_w + 1, stride)
    rows = np.arange(0, image_h - pool_h + 1, stride)

  # Performing pooling
  output = np.zeros((rows.shape[0], cols.shape[0]))
  for i, row in enumerate(rows):
    for j, col in enumerate(cols):
      output[i, j] = image[row:min(row + pool_h, image_h), col:min(col + pool_w, image_w)].max()

  return output
