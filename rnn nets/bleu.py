from typing import Tuple
import numpy as np
import math


def get_ngrams(sequence: list, max_order: int) -> dict:
  """
  Gets ngrams (up to max_order,i.e., max n) from a sequence

  :param sequence: List of tokens
  :param max_order: Max order to build n-grams (max n)
  :return: A dictionary of all n-grams that occurred in the sequence with the count of
  how many time each one occur.
  """
  ngrams = {}
  for n in range(1, max_order + 1):
    for i in range(0, len(sequence) - n + 1):
      key = tuple(sequence[i:i + n])  # actual ngram
      if key not in ngrams:
        ngrams[key] = 1
      else:
        ngrams[key] += 1
  return ngrams


def get_overlap(dict1: dict, dict2: dict) -> dict:
  """
  Finds the overlapping between two dictionaries (key,value) pairs selecting the minimum between two values with
  the same key.

  :param dict1: First dictionary
  :param dict2: Second dictionary
  :return: A dictionary representing the intersection
  """
  intersection = {}
  for key in dict1:
    if key in dict2:
      intersection[key] = min(dict1[key], dict2[key])
  return intersection


def get_counts(references: np.ndarray,
               candidates: np.ndarray,
               ending_token=None,
               max_order=4) -> Tuple[np.ndarray, np.ndarray, float, float]:
  """
  Computes required indicators in order to compute bleu. This function is the previous step to compute the bleu score.

  :param references: A numpy array where each row represents a reference.
  :param candidates: A numpy array where each row represents a candidate.
  :param ending_token: The ending token to detect where each sequence ends.
  :param max_order: The maximum order use for building n-grams
  :return: A tuple containing the number of n-grams hits across all reference/candidate pairs, the total number of
  possible hits, addition of the length of all candidates and the addition of the length of all references.
  """
  # Of all ngrams of order n occurring in the candidate how many occurred in the reference
  current_matches = np.zeros((max_order,))
  # Total number of possible ngram matches of order n
  possible_matches = np.zeros((max_order,))
  # Elements to compute brevity penalty
  candidates_length = 0
  references_length = 0
  for candidate, reference in zip(candidates, references):
    # Tokens after the ending token are ignored
    for i, token in enumerate(candidate):
      if token == ending_token:
        break
    candidate = list(candidate[:i+1])
    for i, token in enumerate(reference):
      if token == ending_token:
        break
    reference = list(reference[:i+1])
    # Updates lengths
    candidates_length += len(candidate)
    references_length += len(reference)

    # Gets ngrams for candidate and reference
    candidate_ngrams = get_ngrams(candidate, max_order)
    reference_ngrams = get_ngrams(reference, max_order)
    # Determines the maximum number of times each ngram or order n happened (repetitions) on both sequences
    overlap = get_overlap(candidate_ngrams, reference_ngrams)

    # Computes the number of common ngrams repetitions for each value of n
    for ngram in overlap:
      current_matches[len(ngram) - 1] += overlap[ngram]
    # Determines the maximum number of times each anagram of order n can be repeated
    for n in range(0, max_order):
      counts = len(candidate) - n
      if counts <= 0:
        break
      possible_matches[n] += counts

  return current_matches, possible_matches, candidates_length, references_length


def get_bleu(current_matches: np.ndarray,
             possible_matches: np.ndarray,
             candidates_length: int,
             references_length: int,
             weights=None, smoothing='method2'):
  """
  Computes bleu score given the
  :param current_matches:
  :param possible_matches:
  :param candidates_length:
  :param references_length:
  :param weights:
  :param smoothing:
  :return:
  """
  # Checking arguments compatibility
  max_order = len(current_matches)
  if weights is None:
    weights = [1.0 / max_order] * max_order
  elif len(weights) != max_order:
    raise Exception('Incompatible arguments: weights and precision should have the same size')

  # Determines precision using smoothing if selected
  precisions = np.zeros((max_order,))
  for n in range(0, max_order):
    # Can't deal with this
    if possible_matches[n] == 0:
      continue
    # Smoothing (dealing with 0 counts) according to SmoothingFunction.method1
    numerator = current_matches[n]
    denominator = possible_matches[n]
    if smoothing == 'method1':
      if numerator == 0:
        numerator = 1
    # Smoothing according to SmoothingFunction.method2
    elif smoothing == 'method2':
      numerator += 1
      denominator += 1
    precisions[n] = float(numerator) / denominator

  # Determines brevity penalty
  ratio = float(candidates_length) / references_length
  if ratio == 0:
    brevity_penalty = 0
  elif ratio > 1.0:
    brevity_penalty = 1.
  else:
    brevity_penalty = math.exp(1 - 1. / ratio)

  # Computes the geometric mean using the exp (log) trick
  return brevity_penalty * math.exp(sum(weights[i] * math.log(precisions[i]) for i in range(0, max_order)
                                        if precisions[i] > 0) / sum(weights))
