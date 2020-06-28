#  Created by Luis Alejandro (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import numpy as np


class Optimizer:
  """
  Abstract optimizer implementation
  """

  def reset(self, params):
    raise NotImplementedError("Must override reset method")

  def update(self, params, gradients):
    raise NotImplementedError("Must override update method")


class BatchGradientDescendOptimizer(Optimizer):
  """
  Basic gradient descend optimizer (better known as batch gradient descend).
  """

  def __init__(self, **opts):
    self.alpha = 1e-3 if 'alpha' not in opts else opts['alpha']

  def reset(self, params):
    pass

  def update(self, params, gradients):
    for param, grad in zip(params, gradients):
      grad = grad.reshape(param.shape)
      if param.trainable:
        step = self.alpha * grad
        param.value -= step


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
      if param.trainable:
        self.cumulative[param] = np.zeros(param.value.shape)

  def update(self, params, gradients):
    for param, grad in zip(params, gradients):
      grad = grad.reshape(param.shape)
      if param.trainable:
        self.cumulative[param] += (grad ** 2)
        step = self.alpha / (np.sqrt(self.cumulative[param]) + self.epsilon) * grad
        param.value -= step


class MomentumOptimizer(Optimizer):
  """
  Gradient descend with momentum (using an exponential weighted average of past gradients).
  """

  def __init__(self, **opts):
    self.alpha = 1e-3 if 'alpha' not in opts else opts['alpha']
    self.beta = 0.9 if 'beta' not in opts else opts['beta']
    self.bias = False if 'bias' not in opts else opts['bias']
    self.momentum = {}
    self.updates = {}

  def reset(self, params):
    for param in params:
      if param.trainable:
        self.momentum[param] = np.zeros(param.value.shape)
        self.updates[param] = 0

  def update(self, params, gradients):
    for param, grad in zip(params, gradients):
      grad = grad.reshape(param.shape)
      if param.trainable:
        # Updates iterations
        self.updates[param] += 1
        # Determining momentum by computing exponential average of previous gradients
        self.momentum[param] = self.beta * self.momentum[param] + (1. - self.beta) * grad
        # Applying bias correction if selected
        if self.bias:
          momentum = self.momentum[param] / (1 - self.beta ** self.updates[param])
        else:
          momentum = self.momentum[param]
        # Updating parameters
        step = self.alpha * momentum
        param.value -= step


class RMSPropOptimizer(Optimizer):
  """
  RMSProp optimizer taking advantage of the idea in Adagrad but using an exponentially
  weighted average of the squared of past gradient instead.
  """

  def __init__(self, **opts):
    self.alpha = 1e-2 if 'alpha' not in opts else opts['alpha']
    self.beta = 0.9 if 'beta' not in opts else opts['beta']
    self.epsilon = 1e-10 if 'epsilon' not in opts else opts['epsilon']
    self.bias = False if 'bias' not in opts else opts['bias']
    self.decay = 0. if 'decay' not in opts else opts['decay']
    self.cumulative = {}
    self.updates = {}

  def reset(self, params):
    for param in params:
      if param.trainable:
        self.cumulative[param] = np.zeros(param.value.shape)
        self.updates[param] = 0

  def update(self, params, gradients):
    for param, grad in zip(params, gradients):
      grad = grad.reshape(param.shape)
      if param.trainable:
        # Updates iterations
        self.updates[param] += 1
        # Computes exponential moving average of the magnitude squared of the gradient
        self.cumulative[param] = self.beta * self.cumulative[param] + (1. - self.beta) * (grad ** 2)
        # Applying bias correction if selected
        if self.bias:
          cumulative = self.cumulative[param] / (1 - self.beta ** self.updates[param])
        else:
          cumulative = self.cumulative[param]
        # Decaying learning rate
        learning = (1 / (1 + self.decay * self.updates[param])) * self.alpha
        # Updating parameters
        step = learning / (np.sqrt(cumulative) + self.epsilon) * grad
        param.value -= step


class AdamOptimizer(Optimizer):
  """
  Adam optimizer that uses momentum and the idea in Adagrad but using an exponentially
  weighted average of the squared of past gradient instead (as in RMSProp)
  """

  def __init__(self, **opts):
    self.alpha = 1e-2 if 'alpha' not in opts else opts['alpha']
    self.beta1 = 0.9 if 'beta1' not in opts else opts['beta1']
    self.beta2 = 0.999 if 'beta2' not in opts else opts['beta2']
    self.epsilon = 1e-10 if 'epsilon' not in opts else opts['epsilon']
    self.bias = True if 'bias' not in opts else opts['bias']
    self.decay = 0 if 'decay' not in opts else opts['decay']
    self.momentum = {}
    self.cumulative = {}
    self.updates = {}

  def reset(self, params):
    for param in params:
      if param.trainable:
        self.momentum[param] = np.zeros(param.value.shape)
        self.cumulative[param] = np.zeros(param.value.shape)
        self.updates[param] = 0

  def update(self, params, gradients):
    for param, grad in zip(params, gradients):
      grad = grad.reshape(param.shape)
      if param.trainable:
        # Updates iterations
        self.updates[param] += 1
        # Determining momentum by computing exponential average of previous gradients
        self.momentum[param] = self.beta1 * self.momentum[param] + (1. - self.beta1) * grad
        # Applying bias correction if selected
        if self.bias:
          momentum = self.momentum[param] / (1 - self.beta1 ** self.updates[param])
        else:
          momentum = self.momentum[param]
        # Computes exponential moving average of the magnitude squared of the gradient
        self.cumulative[param] = self.beta2 * self.cumulative[param] + (1. - self.beta2) * (grad ** 2)
        # Applying bias correction if selected
        if self.bias:
          cumulative = self.cumulative[param] / (1 - self.beta2 ** self.updates[param])
        else:
          cumulative = self.cumulative[param]
        # Decaying learning rate
        learning = (1 / (1 + self.decay * self.updates[param])) * self.alpha
        # Updating parameters
        step = learning / (np.sqrt(cumulative) + self.epsilon) * momentum
        param.value -= step


class RPropOptimizer(Optimizer):
  """
  Rprop implementation. Uses the idea of increasing or decreasing learning rate
  per dimension if the gradient in that dimension stays in the same direction (sign)
  """

  def __init__(self, **opts):
    # Reads config
    self.min_alpha = 1e-10 if 'min_alpha' not in opts else opts['min_alpha']
    self.max_alpha = 1. if 'max_alpha' not in opts else opts['max_alpha']
    self.up_factor = 1.2 if 'up_factor' not in opts else opts['up_factor']
    self.down_factor = 0.5 if 'down_factor' not in opts else opts['down_factor']
    # Inits auxiliary values
    self.alpha = {}
    self.last_sign = {}

  def reset(self, params):
    for param in params:
      if param.trainable:
        self.alpha[param] = self.min_alpha * np.ones(param.shape)
        self.last_sign[param] = np.ones(param.shape, dtype='bool')

  def update(self, params, gradients):
    for param, grad in zip(params, gradients):
      grad = grad.reshape(param.shape)
      if param.trainable:
        # Signs comparison
        curr_sign = grad >= 0
        increase = curr_sign == self.last_sign[param]
        # Increase alpha for corresponding entries
        temp = self.alpha[param][increase] * self.up_factor
        temp[temp > self.max_alpha] = self.max_alpha
        self.alpha[param][increase] = temp
        # Decrease alpha for corresponding entries
        temp = self.alpha[param][~increase] * self.down_factor
        temp[temp < self.min_alpha] = self.min_alpha
        self.alpha[param][~increase] = temp
        # Stores last sign
        self.last_sign[param] = curr_sign
        # Updates param
        step = self.alpha[param] * grad
        param.value -= step
