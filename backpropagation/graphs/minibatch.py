#  Created by Luis Alejandro (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import numpy as np


class BatchProvider:
  """
  Creates batches of samples for training using the graph.
  """
  def __init__(self, feeder, batch_size=None):
    # Stores data and check for compatibility
    self.data = {}
    observations = None
    for node in feeder:
      self.data[node] = node.value
      if observations is None:
        observations = len(node.value)
      elif observations != len(node.value):
        raise Exception('Incompatible input data: different number of observations')
    self.observations = observations
    # Avoid batches greater than dataset
    if batch_size is None:
      self.batch_size = observations
    else:
      self.batch_size = batch_size if batch_size < observations else observations
    # Determines where to start each batch
    self.indexes = np.random.permutation(range(0, self.observations, self.batch_size))
    self.current = 0

  def __next__(self):
    feeder = {}
    # No more batches
    if self.current >= len(self.indexes):
      self.current = 0
      raise StopIteration
    # Reads data for current batch and creates a feeder dictionary
    if self.batch_size <= self.observations:
        for node in self.data:
          i = self.indexes[self.current]
          j = min(self.indexes[self.current] + self.batch_size, self.observations)
          feeder[node] = self.data[node][i:j]
    # Move to next batch
    self.current += 1
    # Return current batch
    return feeder

  def __iter__(self):
    return self
