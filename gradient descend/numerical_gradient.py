#  Created by Luis Alejandro (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author
import numpy as np


def approximate_gradient(f, spacing):
  """
  Computes the numerical gradient for a 2D function using the central difference for interior data points.
  :param f: 2D Function's values
  :param spacing: Time step
  :return: 2D gradient
  """
  nx = f.shape[0]
  ny = f.shape[1]
  dx = np.zeros((nx, ny))
  dy = np.zeros((nx, ny))

  if nx < 2 or ny < 2:
    return dx, dy

  for i in range(1, nx - 1):
    dy[i, :] = (f[i + 1, :] - f[i - 1, :]) / (2 * spacing)
  dy[0, :] = (f[1, :] - f[0, :]) / (2 * spacing)
  dy[nx - 1, :] = (f[nx - 1, :] - f[nx - 2, :]) / (2 * spacing)

  for j in range(1, ny - 1):
    dx[:, j] = (f[:, j + 1] - f[:, j - 1]) / (2 * spacing)
  dx[:, 0] = (f[:, 1] - f[:, 0]) / (2 * spacing)
  dx[:, ny - 1] = (f[:, ny - 1] - f[:, ny - 2]) / (2 * spacing)

  return dx, dy
