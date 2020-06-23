#  Created by Luis Alejandro (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import numpy as np


class Param:
  """
  Simple class to hold parameters
  """

  def __init__(self, shape):
    super().__init__()
    self.value = np.zeros(shape)


class Optimizer:
  """
  Abstract optimizer implementation
  """

  def update(self, params, gradients, hessian=None):
    raise NotImplementedError("Must override update method")

  def reset(self, params):
    raise NotImplementedError("Must override reset method")


class BatchGradientDescendOptimizer(Optimizer):
  """
  Basic gradient descend optimizer (better known as batch gradient descend).
  """

  def __init__(self, **opts):
    self.alpha = 1e-3 if 'alpha' not in opts else opts['alpha']

  def update(self, params, gradients, hessian=None):
    for param, grad in zip(params, gradients):
      step = self.alpha * grad
      param.value -= step

  def reset(self, params):
    pass


class LineSearchOptimizer(Optimizer):
  """
  Basic Line Search optimizer. It needs the Hessian matrix.
  """

  def update(self, params, gradients, hessians=None):
    for param, grad, hessian in zip(params, gradients, hessians):
      alpha = grad.T.dot(grad) / grad.T.dot(hessian).dot(grad)
      step = alpha * grad
      param.value -= step

  def reset(self, params):
    pass


class NewtonsMethodOptimizer(Optimizer):
  """
  Basic Newton's Method optimizer. It needs the Hessian matrix.
  """

  def update(self, params, gradients, hessians=None):
    for param, grad, hessian in zip(params, gradients, hessians):
      step = np.linalg.solve(hessian, grad)
      param.value -= step

  def reset(self, params):
    pass


class AdagradOptimizer(Optimizer):
  """
  Adagrad optimizer. It uses an adaptative learning rate per component of the gradient. The learning rate is
  adjusted based on the cumulative gradient from previous iterations (each component treated independently).
  """

  def __init__(self, **opts):
    self.alpha = 1 if 'alpha' not in opts else opts['alpha']
    self.epsilon = 1e-10 if 'epsilon' not in opts else opts['epsilon']
    self.cumulative = {}

  def reset(self, params):
    for param in params:
      self.cumulative[param] = np.zeros(param.value.shape)

  def update(self, params, gradients):
    for param, grad in zip(params, gradients):
      self.cumulative[param] += (grad ** 2)
      step = self.alpha / (np.sqrt(self.cumulative[param]) + self.epsilon) * grad
      param.value -= step

