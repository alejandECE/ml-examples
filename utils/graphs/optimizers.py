#  Created by Luis Alejandro (alejand@umich.edu)
import numpy as np


class Optimizer():
  '''
  Abstract optimizer implementation
  '''

  def update(self, param):
    raise NotImplementedError("Must override update method")


class BasicOptimizer(Optimizer):
  '''
  Basic gradient descend step depending on manually choosing the learning rate
  '''

  def __init__(self, graph, **opts):
    self.alpha = 1e-5 if 'alpha' not in opts else opts['alpha']

  def update(self, param):
    grad = param.gradient.reshape(param.shape)
    step = self.alpha * grad
    param.value -= step


class MomentumOptimizer(Optimizer):
  '''
  Gradient descend with momentum taking advantage of exponential weighted
  average (efficient way to compute a moving average)
  '''

  def __init__(self, graph, **opts):
    self.alpha = 1e-5 if 'alpha' not in opts else opts['alpha']
    self.beta = 0.9 if 'beta' not in opts else opts['beta']
    self.momentum = {}
    for param in graph.params:
      if param.trainable:
        self.momentum[param] = np.zeros(param.shape)

  def update(self, param):
    grad = param.gradient.reshape(param.shape)
    self.momentum[param] = self.beta * self.momentum[param] + (1. - self.beta) * grad
    step = self.alpha * self.momentum[param]
    param.value -= step


class AdagradOptimizer(Optimizer):
  '''
  Adagrad implementation
  '''

  def __init__(self, graph, **opts):
    # Reads config
    self.alpha = 1 if 'alpha' not in opts else opts['alpha']
    self.epsilon = 1e-10 if 'epsilon' not in opts else opts['epsilon']
    # Inits auxiliary values
    self.cummulative = {}
    for param in graph.params:
      if param.trainable:
        self.cummulative[param] = np.zeros(param.shape)

  def update(self, param):
    # Updates cummulative gradient
    grad = param.gradient.reshape(param.shape)
    self.cummulative[param] += (grad ** 2)
    # Updates param
    step = self.alpha / (np.sqrt(self.cummulative[param]) + self.epsilon) * grad
    param.value -= step


class RPropOptimizer(Optimizer):
  '''
  Rprop implementation. Uses the idea of increasing or decreasing learning rate
  per dimension if the gradient in that dimension stays in the same direction (sign)
  '''

  def __init__(self, graph, **opts):
    # Reads config
    self.min_alpha = 1e-10 if 'min_alpha' not in opts else opts['min_alpha']
    self.max_alpha = 1. if 'max_alpha' not in opts else opts['max_alpha']
    self.up_factor = 1.2 if 'up_factor' not in opts else opts['up_factor']
    self.down_factor = 0.5 if 'down_factor' not in opts else opts['down_factor']
    # Inits auxiliary values
    self.alpha = {}
    self.last_sign = {}
    for param in graph.params:
      if param.trainable:
        self.alpha[param] = self.min_alpha * np.ones(param.shape)
        self.last_sign[param] = np.ones(param.shape, dtype='bool')

  def update(self, param):
    grad = param.gradient.reshape(param.shape)
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


class RMSPropOptimizer(Optimizer):
  '''
  RMSProp optimizer taking advantage of the idea in Adagrad and exponentially
  weighted average
  '''

  def __init__(self, graph, **opts):
    self.alpha = 1e-2 if 'alpha' not in opts else opts['alpha']
    self.beta = 0.9 if 'beta' not in opts else opts['beta']
    self.epsilon = 1e-10 if 'epsilon' not in opts else opts['epsilon']
    self.cummulative = {}
    for param in graph.params:
      if param.trainable:
        self.cummulative[param] = np.zeros(param.shape)

  def update(self, param):
    grad = param.gradient.reshape(param.shape)
    self.cummulative[param] = self.beta * self.cummulative[param] + (1. - self.beta) * (grad ** 2)
    step = self.alpha / (np.sqrt(self.cummulative[param]) + self.epsilon) * grad
    param.value -= step


class AdamOptimizer(Optimizer):
  '''
  Adam optimizer combines gradient descend with momentum and RMSProp
  '''

  def __init__(self, graph, **opts):
    self.alpha = 1e-2 if 'alpha' not in opts else opts['alpha']
    self.beta1 = 0.9 if 'beta1' not in opts else opts['beta1']
    self.beta2 = 0.999 if 'beta2' not in opts else opts['beta2']
    self.epsilon = 1e-10 if 'epsilon' not in opts else opts['epsilon']
    self.momentum = {}
    self.cummulative = {}
    for param in graph.params:
      if param.trainable:
        self.momentum[param] = np.zeros(param.shape)
        self.cummulative[param] = np.zeros(param.shape)

  def update(self, param):
    grad = param.gradient.reshape(param.shape)
    self.momentum[param] = self.beta1 * self.momentum[param] + (1. - self.beta1) * grad
    self.cummulative[param] = self.beta2 * self.cummulative[param] + (1. - self.beta2) * (grad ** 2)
    step = self.alpha / (np.sqrt(self.cummulative[param]) + self.epsilon) * self.momentum[param]
    param.value -= step
