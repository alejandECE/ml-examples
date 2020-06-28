#  Created by Luis Alejandro (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import numpy as np
from graphs.core import Operation


class softmax_mce_node(Operation):
  """
  Computes the multiclass cross entropy for an expected output y of [m x 1]
  containing values between 0 and n - 1 elements after computing the sofmax
  of a set of linear combination of [m x n] elements
  """

  def __init__(self, z, y):
    super().__init__(z, y)
    self.h_node = softmax_node(z)

  def forward(self):
    super().forward()
    self.h_node.forward()
    y = self.inputs[1].value
    i = range(len(y))
    j = y.ravel()
    self.value = -(np.log(self.h_node.value[i, j])).sum()

  def backward(self):
    m, n = self.h_node.value.shape
    y = self.inputs[1].value
    indicator = np.zeros((m, n))
    i = range(len(y))
    j = y.ravel()
    indicator[i, j] = 1
    # Gradient with respect to z
    if self.inputs[0].needs_backprop():
      upstream = (-(indicator - self.h_node.value).reshape(-1, 1)).dot(self.gradient)
      self.inputs[0].upgradient(upstream)


class mce_node(Operation):
  """
  Computes the multiclass cross entropy for an expected output y of [m x 1]
  containing values between 0 and n - 1 elements and a predicted output h of
  [m x n] elements
  """

  def __init__(self, h, y):
    super().__init__(h, y)

  def forward(self):
    super().forward()
    h = self.inputs[0].value
    y = self.inputs[1].value
    i = range(len(y))
    j = y.ravel()
    self.value = -(np.log(h[i, j])).sum()

  def backward(self):
    h = self.inputs[0].value
    y = self.inputs[1].value
    i = range(len(y))
    j = y.ravel()
    # Gradient with respect to h
    if self.inputs[0].needs_backprop():
      temp = np.zeros(h.shape)
      temp[i, j] = -1 / h[i, j]
      upstream = temp.reshape((-1, 1)).dot(self.gradient)
      self.inputs[0].upgradient(upstream)


class softmax_node(Operation):
  """
  Computes the softmax for a matrix z of [m x n] where each row is treated as
  an observation
  """

  def __init__(self, z):
    super().__init__(z)

  def forward(self):
    super().forward()
    z = self.inputs[0].value
    shift = z.max(axis=1).reshape(-1, 1)
    exp = np.exp(z - shift)
    self.value = exp / exp.sum(axis=1).reshape(-1, 1)

  def backward(self):
    m, n = self.value.shape
    upstream = np.zeros((m, n))
    # Gradient with respect to z
    if self.inputs[0].needs_backprop():
      downstream = self.gradient.reshape((m, n))
      for k in range(m):
        temp = self.value[k, :].reshape((1, -1))
        H = temp.T.dot(-temp)
        np.fill_diagonal(H, temp * (1 - temp))
        upstream[k, :] = downstream[k, :].dot(H.T)
      upstream = upstream.reshape((-1, 1))
      self.inputs[0].upgradient(upstream)


class bias_node(Operation):
  """
  Adds a bias term b [1 x n] to an already computed linear
  combination r [m x n]
  """

  def __init__(self, r, b):
    super().__init__(r, b)

  def forward(self):
    super().forward()
    r = self.inputs[0].value
    b = self.inputs[1].value
    self.value = r + b

  def backward(self):
    m, _ = self.value.shape
    # Gradient with respect to r
    if self.inputs[0].needs_backprop():
      upstream = self.gradient
      self.inputs[0].upgradient(upstream.copy())
    # Gradient with respect to b
    if self.inputs[1].needs_backprop():
      upstream = (np.ones((1, m)).dot(upstream.reshape((m, -1)))).T
      self.inputs[1].upgradient(upstream)


class linear_node(Operation):
  """
  Computes the linear combination of all m observations in X [m x d] as
  specified in w [d x n]
  """

  def __init__(self, X, w):
    super().__init__(X, w)

  def forward(self):
    super().forward()
    X = self.inputs[0].value
    w = self.inputs[1].value
    self.value = X.dot(w)

  def backward(self):
    X = self.inputs[0].value
    w = self.inputs[1].value
    m, n = self.value.shape
    downstream = self.gradient.reshape((m, n))
    # Gradient with respect to X
    if self.inputs[0].needs_backprop():
      upstream = downstream.dot(w.T).reshape(-1, 1)
      self.inputs[0].upgradient(upstream)
    # Gradient with respect to w
    if self.inputs[1].needs_backprop():
      upstream = X.T.dot(downstream).reshape(-1, 1)
      self.inputs[1].upgradient(upstream)


class sigmoid_node(Operation):
  """
  Computes the sigmoid function of an input of [m x n] elements
  """

  def __init__(self, z):
    super().__init__(z)

  def forward(self):
    super().forward()
    z = self.inputs[0].value
    self.value = 1 / (1 + np.exp(-z))

  def backward(self):
    # Gradient with respect to z
    if self.inputs[0].needs_backprop():
      upstream = (self.value * (1 - self.value)).reshape(-1, 1) * self.gradient
      self.inputs[0].upgradient(upstream)


class relu_node(Operation):
  """
  Computes the relu function of an input of [m x n] elements
  """

  def __init__(self, z):
    super().__init__(z)

  def forward(self):
    super().forward()
    z = self.inputs[0].value.copy()
    z[z < 0] = 0
    self.value = z

  def backward(self):
    # Gradient with respect to z
    if self.inputs[0].needs_backprop():
      z = self.inputs[0].value.reshape(-1, 1)
      upstream = self.gradient.copy()
      upstream[z <= 0] = 0
      self.inputs[0].upgradient(upstream)


class leaky_relu_node(Operation):
  """
  Computes the leaky relu function of an input of [m x n] elements
  """

  def __init__(self, z, epsilon=0.01):
    super().__init__(z)
    self.epsilon = epsilon

  def forward(self):
    super().forward()
    z = self.inputs[0].value.copy()
    z[z < 0] = self.epsilon * z[z < 0]
    self.value = z

  def backward(self):
    # Gradient with respect to z
    if self.inputs[0].needs_backprop():
      z = self.inputs[0].value.reshape(-1, 1)
      upstream = self.gradient.copy()
      upstream[z <= 0] = self.epsilon * upstream[z <= 0]
      self.inputs[0].upgradient(upstream)


class sse_node(Operation):
  """
  Computes the sum of squared error for an expected output y of [m x 1]
  elements and a predicted output h of [m x 1] elements
  """

  def __init__(self, h, y):
    super().__init__(h, y)

  def forward(self):
    h = self.inputs[0].value
    y = self.inputs[1].value
    diff = h - y
    self.value = diff.T.dot(diff)

  def backward(self):
    h = self.inputs[0].value
    y = self.inputs[1].value
    # Gradient with respect to h
    upstream = (2 * (h - y)).dot(self.gradient)
    self.inputs[0].gradient = upstream
    # Gradient with respect to y
    self.inputs[1].gradient = upstream


class bce_node(Operation):
  """
  Computes the binary cross entropy for an expected output y of [m x 1]
  elements and a predicted output h of [m x 1] elements
  """

  def __init__(self, h, y):
    super().__init__(h, y)

  def forward(self):
    h = self.inputs[0].value
    y = self.inputs[1].value
    self.value = -sum([np.log(prob) if label else np.log(1 - prob) for prob, label in zip(h, y)])

  def backward(self):
    h = self.inputs[0].value
    y = self.inputs[1].value
    # Gradient with respect to h
    upstream = (1 / ((1 - y) - h)).dot(self.gradient)
    self.inputs[0].gradient = upstream
    # Gradient with respect to y
    upstream = np.array([-np.log(prob) if label else -np.log(1 - prob) for prob, label in zip(h, y)]).reshape((-1, 1))


class diff_node(Operation):
  """
  Computes the difference between h [m x 1] and y [m x 1].
  """

  def __init__(self, h, y):
    super().__init__(h, y)

  def forward(self):
    h = self.inputs[0].value
    y = self.inputs[1].value
    self.value = h - y

  def backward(self):
    upstream = self.gradient
    # Gradient with respect to h
    self.inputs[0].gradient = upstream
    # Gradient with respect to b
    self.inputs[1].gradient = -upstream


class l2_node(Operation):
  """
  Computes the L2 norm of the vector d [m x 1]
  """

  def __init__(self, d):
    super().__init__(d)

  def forward(self):
    d = self.inputs[0].value
    self.value = d.T.dot(d)

  def backward(self):
    # Gradient with respect to d
    d = self.inputs[0].value
    upstream = (2 * d).dot(self.gradient)
    self.inputs[0].gradient = upstream
