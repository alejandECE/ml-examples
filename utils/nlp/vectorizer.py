#  Created by Luis Alejandro (alejand@umich.edu)
from scipy.sparse import coo_matrix
import numpy as np


class Vectorizer:
  def __init__(self, corpus, stopwords=None, min_freq=1, max_freq=10000, verbose=False):
    self.corpus = corpus
    self.words = list()
    self.column = dict()
    self.counts = dict()
    self.stopwords = stopwords
    self.min_freq = min_freq
    self.max_freq = max_freq
    self.verbose = verbose
    self.matrix = None

  # Builds words frequency
  def __build_counts(self):
    for i in range(len(self.corpus)):
      for word in self.corpus[i].split():
        if word in self.counts:
          self.counts[word] += 1
        else:
          self.counts[word] = 1

  # Builds words frequency removing stopwords
  def __build_counts_without_stopwords(self):
    # Stopwords
    deletions = set(self.stopwords)
    for i in range(len(self.corpus)):
      for word in self.corpus[i].split():
        if word in deletions:
          continue
        if word in self.counts:
          self.counts[word] += 1
        else:
          self.counts[word] = 1

  # Removes non-frequent or too-frequent words
  def __remove_outliers(self):
    deletions = [word for word in self.counts if
                 self.counts[word] <= self.min_freq or self.counts[word] >= self.max_freq]
    for word in deletions:
      del self.counts[word]

  def __build_columns_indexes(self):
    self.words = list(self.counts.keys())
    for i, element in enumerate(self.words):
      self.column[element] = i

  # Builds the data matrix using a coo sparse matrix format
  def __build_data_matrix(self):
    entries = dict()
    for i, observation in enumerate(self.corpus):
      for word in observation.split():
        if word in self.column:
          j = self.column[word]
          if (i, j) in entries:
            entries[(i, j)] += 1
          else:
            entries[(i, j)] = 1

    rows = np.array([row for row, _ in entries])
    columns = np.array([col for _, col in entries])
    data = np.array([entries[key] for key in entries])

    self.matrix = coo_matrix((data, (rows, columns)))

  # Output some validation data from the resulting matrix
  def output_validation(self):
    words = list(self.counts.keys())
    item = np.random.randint(0, len(words))
    i = (self.matrix.row == item)
    print(item)
    print('item:', self.corpus[item])
    print('column:', self.matrix.col[i])
    print('data:', self.matrix.data[i])
    for key in self.matrix.col[i]:
      print(words[key])

  def fit(self):
    """
        Vectorize the corpus given during initialization
    """
    if self.stopwords is not None:
      self.__build_counts_without_stopwords()
    else:
      self.__build_counts()
    self.__remove_outliers()
    self.__build_columns_indexes()
    self.__build_data_matrix()
    if self.verbose:
      self.output_validation()

    return self.matrix
