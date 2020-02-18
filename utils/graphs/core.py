#  Created by Luis Alejandro (alejand@umich.edu)
import numpy as np
import numpy.random as rnd
from collections import deque
from utils.graphs.optimizers import BasicOptimizer
from utils.graphs.optimizers import AdagradOptimizer
from utils.graphs.optimizers import RPropOptimizer
from utils.graphs.optimizers import RMSPropOptimizer
from utils.graphs.optimizers import MomentumOptimizer
from utils.graphs.optimizers import AdamOptimizer
from utils.graphs.minibatch import BatchProvider


class Node:
  def __init__(self, *args):
    # Update connections between nodes
    self._inputs = []
    for node in args:
      if isinstance(node, Node):
        self._inputs.append(node)
      else:
        raise Exception('Only valid core.Node objects are accepted')

    self.value = None
    self.gradient = None

  def upgradient(self, upstream):
    if self.gradient is None:
      self.gradient = upstream
    else:
      self.gradient += upstream


class Operation(Node):
  def __init__(self, *args):
    super().__init__(*args)

  # Computes result of the operation
  def forward(self):
    for input_node in self._inputs:
      input_node.gradient = None

  # Computes local gradient of this operation with respect to its input
  def backward(self):
    pass


class DataHolder(Node):
  def __init__(self):
    super().__init__()


class Param(Node):
  def __init__(self, shape, initializer='glorot_normal', trainable=True):
    super().__init__()
    self.initializer = initializer
    self.shape = shape
    self.trainable = True

  def initialize(self):
    if self.initializer == 'uniform':
      self.value = rnd.uniform(-0.1, 0.1, self.shape)
    elif self.initializer == 'glorot_uniform':
      std = np.sqrt(2.0 / sum(self.shape))
      self.value = rnd.uniform(-std, std, self.shape)
    elif self.initializer == 'glorot_normal':
      std = np.sqrt(2.0 / sum(self.shape))
      self.value = rnd.normal(0, std, self.shape)


def needs_backprop(node):
  '''
  Determines if gradient should be backpropagated to the node or not! to save time!
  '''
  if isinstance(node, DataHolder):
    return False
  if isinstance(node, Param) and node.trainable == False:
    return False
  return True


class Graph:
  def __init__(self):
    self.post_order = []
    self.breadth_order = []
    self.params = []
    self.holders = []
    self.operation = None

  def build(self, operation):
    '''
    Builds the graph storing operations in proper order for later use in the
    forward/backward passes as well as params/dataholders for minimization
    '''
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
    '''
    Link graph to data sources for later computation
    '''
    if self.operation is None:
      raise Exception('You must build the graph feed data into it')
    for node in self.holders:
      node.value = feeder[node]
    return self

  def initialize(self):
    '''
    Initialize all parameters in the graph
    '''
    if self.operation is None:
      raise Exception('You must build the graph before initializing it')
    for node in self.params:
      node.initialize()
    return self

  def forward(self):
    '''
    Runs selected operation providing parameters in the feed dictionary
    '''
    for node in self.post_order:
      node.forward()
    return self

  def backward(self):
    '''
    Perfoms backpropagation of the gradient through the graph starting from the
    operation selected
    '''
    if self.operation.gradient is None:
      self.operation.upgradient(np.ones((np.size(self.operation.value), 1)))

    for node in self.breadth_order:
      node.backward()
    return self

  def get_nodes_post_order(self, operation):
    '''
    Returns a list of nodes where operations needed by other operations are
    first in the list
    '''
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
          for input_node in node._inputs:
            stack.append(input_node)
    except:
      pass

    output.reverse()
    return list(output)

  def get_nodes_breadth_order(self, operation):
    '''
    Returns a list of nodes in a backbard format
    '''
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
          for input_node in node._inputs:
            queue.append(input_node)
    except:
      pass

    return output

  def minimize(self, optimizer='basic', batches=None, max_epochs=1000,
               min_delta=1e-10, verbose=False, **opts):
    '''
    Minimizes the output of the graph with respect to its params
    '''
    if len(self.params) == 0:
      raise Exception('You must initialize the graph before minimizing it')
    # Creates optimizer
    if optimizer == 'basic':
      optimizer = BasicOptimizer(self, **opts)
    elif optimizer == 'adagrad':
      optimizer = AdagradOptimizer(self, **opts)
    elif optimizer == 'momentum':
      optimizer = MomentumOptimizer(self, **opts)
    elif optimizer == 'rprop':
      optimizer = RPropOptimizer(self, **opts)
    elif optimizer == 'rmsprop':
      optimizer = RMSPropOptimizer(self, **opts)
    elif optimizer == 'adam':
      optimizer = AdamOptimizer(self, **opts)
    # Builds batch provider if needed
    if isinstance(batches, int) or batches is None:
      provider = BatchProvider(self, batches)
    # Inits training
    epoch = 1
    delta = min_delta
    current = np.finfo(np.float64).max
    history = []
    # Training loop
    while epoch < max_epochs:
      # Batch processing
      for batch in provider:
        # Forward & backward pass
        self.forward().backward()
        # Check delta
        delta = current - self.operation.value
        if abs(delta) < min_delta:
          break
        current = self.operation.value
        history.append(current)
        # Updates each param
        for param in self.params:
          if param.trainable:
            optimizer.update(param)
      # Print progress
      if verbose:
        print('After {} epochs cost is: {}'.format(epoch, current))
      # Next epoch
      epoch += 1

    return epoch, history
