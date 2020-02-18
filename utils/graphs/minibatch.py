#  Created by Luis Alejandro (alejand@umich.edu)
import numpy.random as rnd


class BatchProvider:
  def __init__(self, graph, batch_size=None):
    self.graph = graph
    # Stores data and check for compatibility
    data = {}
    observations = None
    for node in graph.holders:
      data[node] = node.value
      if observations is None:
        observations = len(node.value)
      elif observations != len(node.value):
        raise Exception('Incompatible input data: different number of observations')
    self.observations = observations
    self.data = data
    # Avoid batches greater than dataset
    if batch_size is None:
      self.batch_size = observations
    else:
      self.batch_size = batch_size if batch_size < observations else observations
    # Determines where to start each batch
    self.start = rnd.permutation(range(0, self.observations, self.batch_size))
    self.current = 0

  def __next__(self):
    # No more batches
    if self.current >= len(self.start):
      self.current = 0
      raise StopIteration
    # Reads data for current batch and feeds it into graph
    if self.batch_size < self.observations:
      for node in self.graph.holders:
        i = self.start[self.current]
        j = min(self.start[self.current] + self.batch_size, self.observations)
        node.value = self.data[node][i:j]
    # Move to next batch
    self.current += 1
    # Return current batch number
    return self.current

  def __iter__(self):
    return self
