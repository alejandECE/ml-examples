#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import numpy as np


def report_feature_ranking(rank, feature_names, print_count=6):
  """
  Prints out the feature with its corresponding rank value.

  :param rank: A rank of each feature of size [d x 1].
  :param feature_names: A list of all feature names with d names.
  :param print_count: How many features to print. It picks half from the top, half form the bottom.
  """
  indexes = rank.flatten().argsort()
  d = len(indexes)
  if print_count > d:
    print_count = d

  # prints top features
  top = int(np.ceil(print_count / 2))
  for i in range(1, top + 1):
    print('Feature ranked {} is \'{}\' with value {:.2E}'.format(i, feature_names[indexes[-i]], rank[indexes[-i]]))

  # prints the points if needed
  if d > print_count:
    print('.\n.\n.\n')

  # prints bottom features
  bottom = print_count - top
  for i in range(bottom - 1, -1, -1):
    print('Feature ranked {} is \'{}\' with value {:.2E}'.format(d - i, feature_names[indexes[i]], rank[indexes[i]]))
