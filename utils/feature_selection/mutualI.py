#  Created by Luis Alejandro (alejand@umich.edu)

import numpy as np
from scipy.sparse.csc import csc_matrix
import multiprocessing as mp


class MutualInfoI:
  """
  Approximates mutual infomation I (in bits) of each feature in X using histograms (counts) to estimate the density functions px, py and joints pxy and pxx
  Four different implementations are provided:
      1) Computes MII receiving a full matrix
      2) Computes MII receiving a full matrix in parallel (second fastest method)
      3) Computes MII receiving a sparse matrix (in csc format)
      4) Computes MII receiving a sparse matrix (in csc format) in parallel (fastest method)
  """

  def __init__(self, X, y, n_jobs=None):
    """
    Main constructor

    Parameters:
        X: Predictors in one of the following formats np.array or csc_matrix
        y: Responses as an np.array
        bins: How many bins to use in the histogram. The more bins the better the approximation but the slower it is
        n_jobs: How many cores to use. A value of -1 indicates to use all available cores. In a quad core processor a value of 4 gives the best performance
        info: The computed MII
    """
    # Inputs
    self.X = X
    self.y = y
    self.n_jobs = n_jobs
    if n_jobs is not None:
      if n_jobs == -1:
        self.workers = mp.cpu_count()
      if n_jobs > 0:
        self.workers = mp.cpu_count() if n_jobs > mp.cpu_count() else n_jobs

    # Auxiliary variables in the computation
    self.__observations = self.X.shape[0]  # rows
    self.__features = self.X.shape[1]  # columns
    self.__classes = len(np.unique(y))  # unique values of y
    self.__px = dict()
    self.__x_edges = dict()
    self.__py = []
    self.__y_edges = []
    self.__mi_xy = np.zeros(self.__features)
    self.__mi_xx = dict()

    # Output
    self.info = np.zeros(self.__features)

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
        return self.__parallel_mutual(self.basic_mi_xy_worker, self.basic_mi_xx_worker)
    elif type(self.X) == csc_matrix:
      if self.n_jobs is None:
        return self.__csc_mutual()
      else:
        return self.__parallel_mutual(self.csc_mi_xy_worker, self.csc_mi_xx_worker)

  def __basic_mutual(self):
    print('Using sequential version will full matrix')
    for i in range(self.__features):
      column = self.X[:, i]
      self.info[i] = self.__compute_mi_xy(column)
    return self.info

  def __csc_mutual(self):
    print('Using sequential version with CSC sparse matrix')
    for i in range(self.__features):
      column = np.zeros(self.__observations)
      column[self.X.indices[self.X.indptr[i]:self.X.indptr[i + 1]]] = self.X.data[self.X.indptr[i]:self.X.indptr[i + 1]]
      self.info[i] = self.__compute_mi_xy(column)
    return self.info

  def __parallel_mutual(self, mi_xy_worker, mi_xx_worker):
    print('Using parallel version')

    # Creates queues
    processes = []
    tasks_queue = mp.JoinableQueue()
    results_queue = mp.Queue()

    # We first compute mi(xj,y)
    # For this we use parallel computing having a process work on a column at a time

    # Generates workers
    for i in range(self.workers):
      p = mp.Process(target=mi_xy_worker, args=(tasks_queue, results_queue,))
      processes.append(p)
      p.start()
    # Generate actual tasks
    for i in range(self.__features):
      tasks_queue.put(i)
    # Generate stopping taks
    for i in range(self.workers):
      tasks_queue.put(None)
    # Start and wait for the processes to finish
    tasks_queue.join()
    # Gathers all results (i,(px[i], x_edges[i], mi_xy[i]))
    while results_queue.empty() is False:
      res = results_queue.get()
      i = res[0]
      px = res[1][0]
      x_edges = res[1][1]
      mi_xy = res[1][2]

      self.__mi_xy[i] = mi_xy  # stores mi of the feature with respect to the target
      self.__x_edges[i] = x_edges  # stores x_edges of the feature
      self.__px[i] = px  # stores the approx of px already computed for later use
    # Close all processes
    for p in processes:
      p.terminate()

    # We now compute mi(xj,xi) for all possible combination of i and j
    # We know though that mi(xj,xi) = mi(xi,xj)
    # For this computation we have a process compute one of these mi at a time

    # Generates workers
    processes.clear()
    for i in range(self.workers):
      p = mp.Process(target=mi_xx_worker, args=(tasks_queue, results_queue,))
      processes.append(p)
      p.start()
      # Generate actual tasks
    for i in range(self.__features):
      for j in range(i + 1, self.__features):
        tasks_queue.put((i, j))
    # Generate stopping taks
    for i in range(self.workers):
      tasks_queue.put(None)
    # Start and wait for the processes to finish
    tasks_queue.join()
    # Gathers all results ((i,j),mi_xx[(i,j))
    while results_queue.empty() is False:
      res = results_queue.get()
      pair = res[0]
      mi_xx = res[1]
      self.__mi_xx[pair] = mi_xx
    # Close all processes
    for p in processes:
      p.terminate()

    # We now compute the average of mi(xj,xi) for all j features

    # Generates workers
    processes.clear()
    for i in range(self.workers):
      p = mp.Process(target=self.final_mii_worker, args=(tasks_queue, results_queue,))
      processes.append(p)
      p.start()
    # Generate actual tasks
    for i in range(self.__features):
      tasks_queue.put(i)
    # Generate stopping taks
    for i in range(self.workers):
      tasks_queue.put(None)
    # Start and wait for the processes to finish
    tasks_queue.join()
    # Gathers all results ((i,j),mi_xx[(i,j))
    while results_queue.empty() is False:
      term = results_queue.get()
      self.info = self.__mi_xy - term
    # Close all processes
    for p in processes:
      p.terminate()

    return self.info

  def basic_mi_xy_worker(self, tasks_queue, results_queue):
    while True:
      i = tasks_queue.get()
      if i is None:
        tasks_queue.task_done()
        break
      column = self.X[:, i]
      result = self.__compute_mi_xy(column)
      tasks_queue.task_done()
      results_queue.put((i, result))

  def basic_mi_xx_worker(self, tasks_queue, results_queue):
    while True:
      pair = tasks_queue.get()
      if pair is None:
        tasks_queue.task_done()
        break
      i = pair[0]
      j = pair[1]
      column_i = self.X[:, i]
      column_j = self.X[:, j]
      result = self.__compute_mi_xx(i, j, column_i, column_j)
      tasks_queue.task_done()
      results_queue.put(((i, j), result))

  def csc_mi_xy_worker(self, tasks_queue, results_queue):
    while True:
      i = tasks_queue.get()
      if i is None:
        tasks_queue.task_done()
        break
      column = np.zeros(self.__observations)
      column[self.X.indices[self.X.indptr[i]:self.X.indptr[i + 1]]] = self.X.data[self.X.indptr[i]:self.X.indptr[i + 1]]
      result = self.__compute_mi_xy(column)
      tasks_queue.task_done()
      results_queue.put((i, result))

  def csc_mi_xx_worker(self, tasks_queue, results_queue):
    while True:
      pair = tasks_queue.get()
      if pair is None:
        tasks_queue.task_done()
        break
      i = pair[0]
      j = pair[1]
      column_i = np.zeros(self.__observations)
      column_i[self.X.indices[self.X.indptr[i]:self.X.indptr[i + 1]]] = self.X.data[
                                                                        self.X.indptr[i]:self.X.indptr[i + 1]]
      column_j = np.zeros(self.__observations)
      column_j[self.X.indices[self.X.indptr[j]:self.X.indptr[j + 1]]] = self.X.data[
                                                                        self.X.indptr[j]:self.X.indptr[j + 1]]

      result = self.__compute_mi_xx(i, j, column_i, column_j)
      tasks_queue.task_done()
      results_queue.put(((i, j), result))

  def final_mii_worker(self, tasks_queue, results_queue):
    while True:
      i = tasks_queue.get()
      if i is None:
        tasks_queue.task_done()
        break
      summ = 0
      for j in range(self.__features):
        if i == j:
          continue
        elif i > j:
          summ += self.__mi_xx[(j, i)]
        else:
          summ += self.__mi_xx[(i, j)]

      tasks_queue.task_done()
      results_queue.put(summ / (self.__features - 1))

  def __compute_mi_xy(self, column):
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
    pxpy = np.matmul(px.reshape((len(px), 1)), py.reshape(1, len(py)));
    # Find non-zero elements
    j = ((pxpy != 0) & (pxy != 0)).nonzero()
    mi_xy = np.sum(pxy[j] * np.log2(pxy[j] / pxpy[j]))

    return px, x_edges, mi_xy

  def __compute_mi_xx(self, i, j, column_i, column_j):
    # Receives pxi approximated before also using histogram
    pxi = self.__px[i]
    xi_edges = self.__x_edges[i]
    # Receives pxj approximated before also using histogram
    pxj = self.__px[j]
    xj_edges = self.__x_edges[j]
    # Aprox joint probability pxy using histogram
    pxixj, _, _ = np.histogram2d(column_i, column_j, [xi_edges, xj_edges], density=False);
    pxixj = pxixj / pxixj.sum()
    # Build all possible combinations px*py for mutual info computation
    pxipxj = np.matmul(pxi.reshape((len(pxi), 1)), pxj.reshape(1, len(pxj)));
    # Find non-zero elements
    j = ((pxipxj != 0) & (pxixj != 0)).nonzero()
    mi_xx = np.sum(pxixj[j] * np.log2(pxixj[j] / pxipxj[j]))

    return mi_xx
