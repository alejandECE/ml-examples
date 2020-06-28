#  Created by Luis Alejandro (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import numpy as np
from collections import deque
from graphs.minibatch import BatchProvider


class Node:
  """
  Base class to act as node in a graph.
  """
  def __init__(self, *args):
    # Update connections between nodes
    self.inputs = []
    for node in args:
      if isinstance(node, Node):
        self.inputs.append(node)
      else:
        raise Exception('Only valid core Node objects are accepted')
    # Holds value of the node
    self.value = None
    # Holds gradient of the node
    self.gradient = None

  def upgradient(self, upstream: np.ndarray) -> None:
    """
    Updates this node gradient (how much it affects the last node in the graph). Notice that it might affect it through
    different paths, that's why the += operation.

    :param upstream: Derivative of the immediate operation with respect to this node.
    """
    if self.gradient is None:
      self.gradient = upstream
    else:
      self.gradient += upstream

  def needs_backprop(self) -> bool:
    """
    Determines if gradient should be back-propagated to the node or not! to save time!
    """
    if isinstance(self, DataHolder):
      return False
    if isinstance(self, Param) and self.trainable is False:
      return False
    return True


class Operation(Node):
  """
  Represents an operation node in a graph (performs some operation between its inputs).
  Must implement the forward method (computing the node's value, aka the operation result) and
  the backward method which computes the gradient with respect to each of the inputs.
  """
  def __init__(self, *args):
    super().__init__(*args)

  def forward(self) -> None:
    """
    Computing the node's value, a.k.a the operation's result.
    """
    # Resets the gradient of each input
    for input_node in self.inputs:
      input_node.gradient = None

  def backward(self) -> None:
    """
    Computes local gradient of this operation with respect to its input and then it propagates it backward calling
    the input nodes upgradient method.
    """
    pass


class DataHolder(Node):
  """
  Dummy class to at as dataholder. This is a leaf node (with no inputs).
  """
  def __init__(self):
    super().__init__()


class Param(Node):
  """
  Represents a parameter in the graph (it can be initialized). This is a leaf node (with no inputs).
  """
  def __init__(self, shape, initializer='glorot_normal', trainable=True):
    super().__init__()
    self.initializer = initializer
    self.shape = shape
    self.trainable = trainable

  def initialize(self) -> None:
    """
    Initializes the parameter's value using the selected initialization technique.
    """
    if self.initializer == 'uniform':
      self.value = np.random.uniform(-0.1, 0.1, self.shape)
    elif self.initializer == 'glorot_uniform':
      std = np.sqrt(2.0 / sum(self.shape))
      self.value = np.random.uniform(-std, std, self.shape)
    elif self.initializer == 'glorot_normal':
      std = np.sqrt(2.0 / sum(self.shape))
      self.value = np.random.normal(0, std, self.shape)


class Graph:
  """
  Represents a simple computational graph.
  """
  def __init__(self):
    self.post_order = []
    self.breadth_order = []
    self.params = []
    self.holders = []
    self.operation = None

  def build(self, operation):
    """
    Builds the graph storing operations in proper order for later use in the
    forward/backward passes as well as params/dataholders for minimization
    """
    nodes = self.get_nodes_post_order(operation)
    for node in nodes:
      if isinstance(node, Operation):
        self.post_order.append(node)
      elif isinstance(node, DataHolder):
        self.holders.append(node)
      elif isinstance(node, Param):
        self.params.append(node)

    nodes = self.get_nodes_breadth_order(operation)
    for node in nodes:
      if isinstance(node, Operation):
        self.breadth_order.append(node)

    self.operation = operation
    return self

  def feed(self, feeder):
    """
    Link graph to data sources for later computation
    """
    if self.operation is None:
      raise Exception('You must build the graph feed data into it')
    for node in self.holders:
      node.value = feeder[node]
    return self

  def initialize(self):
    """
    Initialize all parameters in the graph
    """
    if self.operation is None:
      raise Exception('You must build the graph before initializing it')
    for node in self.params:
      node.initialize()
    return self

  def forward(self):
    """
    Runs selected operation providing parameters in the feed dictionary
    """
    for node in self.post_order:
      node.forward()
    return self

  def backward(self):
    """
    Performs back-propagation of the gradient through the graph starting from the
    operation selected during built.
    """
    if self.operation.gradient is None:
      self.operation.upgradient(np.ones((np.size(self.operation.value), 1)))

    for node in self.breadth_order:
      node.backward()
    return self

  def get_nodes_post_order(self, operation):
    """
    Returns a list of nodes where operations needed by other operations are
    first in the list
    """
    # list
    output = deque()
    # visiting nodes in postorder
    stack = deque()
    stack.append(operation)
    try:
      while True:
        node = stack.pop()
        output.append(node)
        if isinstance(node, Operation):
          for input_node in node.inputs:
            stack.append(input_node)
    except:
      pass

    output.reverse()
    return list(output)

  def get_nodes_breadth_order(self, operation):
    """
    Returns a list of nodes in a backbard format
    """
    # list
    output = []
    # visiting nodes in breadth first order (using queue)
    queue = deque()
    queue.append(operation)
    try:
      while True:
        node = queue.popleft()
        output.append(node)
        if isinstance(node, Operation):
          for input_node in node.inputs:
            queue.append(input_node)
    except:
      pass

    return output

  def minimize(self, optimizer, batch_size=None, max_epochs=1000,
               min_delta=1e-10, verbose=False):
    """
    Minimizes the output of the graph with respect to its params
    """
    if len(self.params) == 0:
      raise Exception('You must initialize the graph before minimizing it')
    # Builds batch provider if needed
    provider = BatchProvider(self.holders, batch_size)
    # Inits training
    epoch = 1
    current = np.finfo(np.float64).max
    history = []
    optimizer.reset(self.params)
    # Training loop
    while epoch < max_epochs:
      # Batch processing
      for batch in provider:
        # Forward & backward pass
        self.feed(batch)
        self.forward().backward()
        # Check delta
        delta = current - self.operation.value
        if abs(delta) < min_delta:
          break
        current = self.operation.value
        history.append(current)
        # Updates each param
        gradients = [param.gradient for param in self.params]
        optimizer.update(self.params, gradients)
      # Print progress
      if verbose:
        print('After {} epochs cost is: {}'.format(epoch, current))
      # Next epoch
      epoch += 1
    # Resets graph data
    self.feed(provider.data)
    # Returns results
    return epoch, history
