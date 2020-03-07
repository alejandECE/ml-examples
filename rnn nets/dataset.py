# Created by Luis Alejandro (alejand@umich.edu)
import tensorflow as tf
import numpy as np
from typing import Tuple

# Some constants
AUTOTUNE = tf.data.experimental.AUTOTUNE


class Tokenizer:
  """
  Supporting class to tokenize sequences of text. You should build the entire vocabulary
  using the update function before encoding anything. You can build the vocabulary
  incrementally.
  """

  def __init__(self):
    self.word_to_index = {}
    self.index_to_word = []
    self.max_seq = 0

  def update(self, words: list) -> None:
    if self.max_seq < len(words):
      self.max_seq = len(words)
    for word in words:
      if word not in self.word_to_index:
        self.word_to_index[word] = len(self.index_to_word)
        self.index_to_word.append(word)

  def encode(self, text: list) -> list:
    sequence = [self.word_to_index[word] for word in text if word in self.word_to_index]
    return sequence[:self.max_seq]

  def decode(self, text: list) -> list:
    sequence = [self.index_to_word[index] for index in text]
    return sequence


class DatasetBuilder:
  """
  Supporting class to generate sequences of words for both input and outputs from text files
  """

  def __init__(self, files, separator='\t', preprocessors=(None, None), batch_size=64, buffer_size=10000,
               max_obs=None, test_obs=None):
    """
    Creates a seq2seq dataset builder that takes a list of files and returns a dataset of encoded sequences

    :param files: List of files or simple file :param separator: Separator used in the text files lines to separate
    input and output
    :param preprocessors: Preprocess functions (source, target) for every text sequence (input and output). It must
    follow the following recipe func_name(tensor: tf.Tensor) -> str. The input tensor is an Eager tensor so you have
    access to the numpy() method.
    :param batch_size: Batch size used in the dataset
    :param buffer_size: Buffer size used to shuffle the observations in the dataset
    :param max_obs: Max number of observations to consider from files
    :param test_obs: Observations in the test set
    """
    self.files = files
    self.source_tokenizer = None
    self.target_tokenizer = None
    self.separator = separator
    self.preprocessors = preprocessors
    self.batch_size = batch_size
    self.buffer_size = buffer_size
    self.max_obs = max_obs
    self.test_obs = test_obs

  def tf_preprocess_wrapper(self, text: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Wraps the pre-processing function for execution in graph mode

    :param text: A tf.string Tensor containing a line of text from a file.
    :return: A tuple of tensor where the first element is the pre-processed source sequence
    tensor and the second is the pre-processed target sequence tensor.
    """
    text = tf.strings.split(text, sep=self.separator)
    if self.preprocessors[0] is not None:
      source = tf.py_function(self.preprocessors[0], [text[0]], tf.string)
    else:
      source = text[0]
    if self.preprocessors[1] is not None:
      target = tf.py_function(self.preprocessors[1], [text[1]], tf.string)
    else:
      target = text[1]
    source.set_shape([])
    target.set_shape([])
    source = tf.strings.split(source)
    target = tf.strings.split(target)
    return source, target

  def encode_source(self, sentence: tf.Tensor) -> np.ndarray:
    sequence = self.source_tokenizer.encode(sentence.numpy())
    return np.array(sequence)

  def encode_target(self, sentence: tf.Tensor) -> np.ndarray:
    sequence = self.target_tokenizer.encode(sentence.numpy())
    return np.array(sequence)

  def tf_encode_wrapper(self, source: tf.Tensor, target: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    shape = source.shape
    source = tf.py_function(self.encode_source, [source], tf.int32)
    source.set_shape(shape)
    shape = target.shape
    target = tf.py_function(self.encode_target, [target], tf.int32)
    target.set_shape(shape)
    return source, target

  def build(self, logged=False) -> tf.data.Dataset:
    self.source_tokenizer = Tokenizer()
    self.target_tokenizer = Tokenizer()

    # Creates from the files
    dataset = tf.data.TextLineDataset(self.files)
    if self.max_obs is not None:
      dataset = dataset.take(self.max_obs)
    if logged:
      print('After reading files the dataset element spec is: \n', dataset.element_spec)

    # Preprocess each line
    dataset = dataset.map(self.tf_preprocess_wrapper).cache()
    if logged:
      print('\nAfter preprocessing files the dataset element spec is: \n', dataset.element_spec)
      # Print some example tensors
      print('\nSome samples from dataset (at this point):')
      for element in dataset.take(5):
        print(element[0].numpy(), element[1].numpy())

    # Generate tokenizers (traverse entire dataset)
    for source_sentence, target_sentence in dataset:
      self.source_tokenizer.update(source_sentence.numpy())
      self.target_tokenizer.update(target_sentence.numpy())

    # Encode each sequence
    dataset = dataset.map(lambda source, target: self.tf_encode_wrapper(source, target))
    if logged:
      print('\nAfter encoding sequences the dataset element spec is: \n', dataset.element_spec)
      print('\nSome samples from dataset (at this point):')
      for element in dataset.take(5):
        print(element[0].numpy(), element[1].numpy())

    # Split train/test if needed
    if self.test_obs is not None:
      dataset = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=False)
      test = dataset.take(self.test_obs)
      train = dataset.skip(self.test_obs)
    else:
      train = dataset
      test = dataset.skip(-1)

    # Shuffle the dataset and create padded batches of sequences (same length).
    padded_shapes = ([self.source_tokenizer.max_seq, ],
                     [self.target_tokenizer.max_seq, ])
    padded_values = (self.source_tokenizer.word_to_index[b'<end>'],
                     self.target_tokenizer.word_to_index[b'<end>'])
    train = train.shuffle(self.buffer_size).padded_batch(self.batch_size, padded_shapes, padded_values).prefetch(1)
    test = test.padded_batch(self.batch_size, padded_shapes, padded_values).prefetch(1)
    if logged:
      print('\nFinal dataset element spec is: \n', train.element_spec)
      print('\nSome samples from the final dataset (training set):')
      for element in train.take(5):
        print(element[0].numpy()[0], element[1].numpy()[0])

    return train, test
