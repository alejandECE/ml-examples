#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import numpy as np
from scipy.sparse.csc import csc_matrix
import multiprocessing as mp


class MutualInfo:
  """
  Approximates mutual infomation (in bits) of each feature in X and the output y using histograms (counts) to estimate the density functions px, py and joint pxy
  Four different implementations are provided:
    1) Computes MI receiving a full matrix
    2) Computes MI receiving a full matrix in parallel (second fastest method)
    3) Computes MI receiving a sparse matrix (in csc format)
    4) Computes MI receiving a sparse matrix (in csc format) in parallel (fastest method)
  """

  def __init__(self, X, y, n_jobs=None):
    """
    Constructor
    :param X: Predictors in one of the following formats np.array or csc_matrix
    :param y: Responses as an np.array
    :param n_jobs: Number of workers
    """
    self.X = X
    self.y = y
    self.__observations = self.X.shape[0]
    self.__features = self.X.shape[1]
    self.__classes = len(np.unique(y))
    self.info = np.zeros(self.__features)
    self.n_jobs = n_jobs
    if n_jobs is not None:
      if n_jobs == -1:
        self.workers = mp.cpu_count()
      if n_jobs > 0:
        self.workers = mp.cpu_count() if n_jobs > mp.cpu_count() else n_jobs
    self.__py = []
    self.__y_edges = []

  def compute(self):
    """
    Computes the mutual information of each feature in X and the output y specified in the constructor
    """
    # Aprox py using histogram (only needs to be computed once)
    self.__py, self.__y_edges = np.histogram(self.y, self.__classes, density=False);
    self.__py = self.__py / self.__observations

    if type(self.X) == np.ndarray:
      if self.n_jobs is None:
        return self.__basic_mutual()
      else:
        return self.__parallel_mutual(self.basic_worker)
    elif type(self.X) == csc_matrix:
      if self.n_jobs is None:
        return self.__csc_mutual()
      else:
        return self.__parallel_mutual(self.csc_worker)

  def __basic_mutual(self):
    print('Using basic matrix version')
    for i in range(self.__features):
      self.info[i] = self.__process_feature(self.X[:, i])
    return self.info

  def __csc_mutual(self):
    print('Using CSC matrix version')
    for i in range(self.__features):
      column = np.zeros(self.__observations)
      column[self.X.indices[self.X.indptr[i]:self.X.indptr[i + 1]]] = self.X.data[self.X.indptr[i]:self.X.indptr[i + 1]]
      self.info[i] = self.__process_feature(column)
    return self.info

  def __parallel_mutual(self, worker):
    print('Using parallel version')
    # Creates queues
    tasks_queue = mp.JoinableQueue()
    results_queue = mp.Queue()
    # Generates workers
    processes = []
    for i in range(self.workers):
      p = mp.Process(target=worker, args=(tasks_queue, results_queue,))
      processes.append(p)
      p.start()
    # Generate actual tasks
    for i in range(self.__features):
      tasks_queue.put(i)
    # Generate stopping tasks
    for i in range(self.workers):
      tasks_queue.put(None)
    # Start and wait for the processes to finish
    tasks_queue.join()
    # Gathers all results
    while results_queue.empty() is False:
      result = results_queue.get()
      self.info[result[0]] = result[1]
    # Close all processes
    for p in processes:
      p.terminate()

    return self.info

  def basic_worker(self, tasks_queue, results_queue):
    while True:
      i = tasks_queue.get()
      if i is None:
        tasks_queue.task_done()
        break
      result = self.__process_feature(self.X[:, i])
      tasks_queue.task_done()
      results_queue.put((i, result))

  def csc_worker(self, tasks_queue, results_queue):
    while True:
      i = tasks_queue.get()
      if i is None:
        tasks_queue.task_done()
        break
      column = np.zeros(self.__observations)
      column[self.X.indices[self.X.indptr[i]:self.X.indptr[i + 1]]] = self.X.data[self.X.indptr[i]:self.X.indptr[i + 1]]
      result = self.__process_feature(column)
      tasks_queue.task_done()
      results_queue.put((i, result))

  def __process_feature(self, column):
    # Aprox px using histogram
    px, x_edges = np.histogram(column, 'sqrt', density=False);
    px = px / self.__observations
    # Receives py approximated before also using histogram
    py = self.__py
    y_edges = self.__y_edges
    # Aprox joint probability pxy using histogram
    pxy, _, _ = np.histogram2d(column, self.y, [x_edges, y_edges], density=False);
    pxy = pxy / self.__observations
    # Build all possible combinations px*py for mutual info computation
    pxpy = np.matmul(px.reshape((len(px), 1)), py.reshape(1, self.__classes));
    # Find non-zero elements
    j = ((pxpy != 0) & (pxy != 0)).nonzero()

    return np.sum(pxy[j] * np.log2(pxy[j] / pxpy[j]))
