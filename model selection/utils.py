#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

from typing import Tuple
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix


def custom_binary_split(predictors: np.ndarray, responses: np.ndarray,
                        samples: int = None, shuffle: bool = True) -> Tuple:
  """
  Performs a random split in a binary classification dataset including the same number of samples from both classes in
  included dataset. The rest of the observations are sent to the holdout set.

  :param predictors: Array of observations x features
  :param responses: Array of a binary expected output
  :param samples: Number of samples to include from both classes
  :param shuffle: Whether to shuffle or not before splitting
  """
  positives = predictors[responses == 1]
  negatives = predictors[responses == 0]
  if samples is None:
    samples = int(np.minimum(0.8 * negatives.shape[0], 0.8 * positives.shape[0]))
  elif samples >= negatives.shape[0] or samples >= positives.shape[0]:
    raise Exception('Not enough samples to place into the test set.')
  # Shuffles positives and negative samples
  if shuffle:
    np.random.shuffle(positives)
    np.random.shuffle(negatives)
  included_positives = positives[:samples]
  holdout_positives = positives[samples:]
  included_negatives = negatives[:samples]
  holdout_negatives = negatives[samples:]
  # Building included dataset
  included_data = np.vstack([
    included_positives, included_negatives
  ])
  included_label = np.vstack([
    np.ones((samples, 1)),
    np.zeros((samples, 1))
  ])
  # Shuffles end result to prevent showing the model only samples from one class (only effective if batches are used!)
  indexes = np.random.permutation(included_data.shape[0])
  included_data = included_data[indexes, :]
  included_label = included_label[indexes]
  # Building holdout dataset
  holdout_data = np.vstack([
    holdout_positives, holdout_negatives
  ])
  holdout_label = np.vstack([
    np.ones((holdout_positives.shape[0], 1)),
    np.zeros((holdout_negatives.shape[0], 1))
  ])
  # Shuffles end result (optional!)
  indexes = np.random.permutation(holdout_data.shape[0])
  holdout_data = holdout_data[indexes, :]
  holdout_label = holdout_label[indexes]

  return included_data, holdout_data, included_label, holdout_label


def report_search(results, n_top=3):
  """
  Report the results from a grid search using GridSearchCV implementation

  Parameters:
  :param results: The resulting object from GridSearchCV fit method
  :param n_top: Top ranks to report
  """
  indexes = results['rank_test_score'].argsort()
  for i in range(n_top):
    candidate = indexes[i]
    print("\nModel with rank: {}".format(i + 1))
    print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
      results['mean_test_score'][candidate],
      results['std_test_score'][candidate]))
    print("Parameters: {}".format(results['params'][candidate]))


def report_classification(y_true, y_pred, avg='binary', title='Test'):
  """
  Report the classification results using accuracy, f1 score, recall and precision

  :param y_true: Vector of true outputs
  :param y_pred: Vector of predicted outputs
  :param avg: Indicates what average mode (binary, micro, macro) to use
  :param title: Title shown in the output
  """
  print(title, '(Metrics): ')
  print('')
  print('Accuracy:', '%.2f' % accuracy_score(y_true, y_pred))
  print('F1 Score:', '%.2f' % f1_score(y_true, y_pred, average=avg))
  print('Recall:', '%.2f' % recall_score(y_true, y_pred, average=avg))
  print('Precision:', '%.2f' % precision_score(y_true, y_pred, average=avg))
  print('\nConfusion Matrix:\n', confusion_matrix(y_true, y_pred))
