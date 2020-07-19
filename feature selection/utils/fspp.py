#  Created by Luis Alejandro (alejand@umich.edu)
import numpy as np
import numpy.random as rnd


def get_fspp(mdl, X):
  """
  Computes a ranking of features for an MLP neural network using feature sensitivity of posterior probability.

  :param mdl: An already trained MLPClassifier from sklearn.
  :param X: An [m x d] data matrix with m observations and d features.
  :return: Ranking of features as a [d x 1] array. The higher the value the more relevant the feature is.
  """
  m, d = X.shape
  temp = np.zeros(m)
  rank = np.zeros(d)
  actual_posterior = mdl.predict_proba(X)
  for j in range(d):
    i = rnd.permutation(m)
    temp[:] = X[:, j]
    X[:, j] = X[i, j]
    affected_posterior = mdl.predict_proba(X)
    X[:, j] = temp
    rank[j] = np.abs(actual_posterior - affected_posterior).sum() / m
  return rank
