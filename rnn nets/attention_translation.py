# Created by Luis Alejandro (alejand@umich.edu)
import re
import time
import os
import tensorflow as tf
import numpy as np
from utils import unicode_to_ascii
from dataset import Tokenizer
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from bleu import get_counts, get_bleu


def preprocess(tensor: tf.Tensor) -> str:
  """
  Pre-process sequence of text for a translation task

  :param tensor: Eager tf.string tensor (can use .numpy() method)
  :return: str containing the pre-processed sequence
  """
  sentence = tensor.numpy().decode('UTF-8')
  # Converts to lowercase ascii representation
  sentence = unicode_to_ascii(sentence.lower().strip())

  # Creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)

  # Replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)

  # Removing spaces
  sentence = sentence.rstrip().strip()
  return '<start> ' + sentence + ' <end>'


def plot_attention(attention: np.ndarray, source: list, target: list, source_vocab: list, target_vocab: list) -> None:
  """
  Creates attention plot

  :param attention: Attention weights matrix [input_seq_length, output_seq_length]
  :param source: Input sequence
  :param target: Output sequence
  :param source_vocab: Input vocabulary
  :param target_vocab: Output vocabulary
  """
  fig = plt.figure(figsize=(5, 5))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention, cmap='gray')
  ax.set_xticklabels([''] + [source_vocab[word].decode() for word in source], rotation=90)
  ax.set_yticklabels([''] + [target_vocab[word].decode() for word in target])
  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.grid()
  plt.show()


class Encoder(tf.keras.layers.Layer):
  """
  Encoder layer that contains an Embedding layer followed by a Bidirectional LSTM Layer. The latter returns a sequence
  of all hidden states to be utilize by a Decoder layer using an Attention mechanism.
  """

  def __init__(self, latent_size: int,
               vocab: dict,
               embedding_size=None,
               embedding_matrix=None,
               name='Encoder',
               **kwargs):
    super(Encoder, self).__init__(name=name, **kwargs)

    # Stores for later use
    self.vocab = vocab
    self.latent_size = latent_size

    # Defines embedding layer
    if embedding_size is not None:
      self.embedding = tf.keras.layers.Embedding(len(self.vocab), embedding_size, name='Embedding')
    elif embedding_matrix is not None:
      num_vectors, embedding_size = embedding_matrix.shape
      if num_vectors != len(self.vocab):
        raise Exception('Vocabulary (length {}) and embedding matrix (Vectors {}) are not compatible'.format(
          len(self.vocab), num_vectors))
      self.embedding = tf.keras.layers.Embedding(len(self.vocab), embedding_size,
                                                 weights=[embedding_matrix],
                                                 trainable=False,
                                                 name='Embedding')
    else:
      raise Exception("""Must provide either embedding_dim (embeddings will be learned) or embedding
      matrix (reusing embeddings)""")

    # Defines LSTM layer
    self.lstm = tf.keras.layers.Bidirectional(
      tf.keras.layers.LSTM(self.latent_size,
                           return_sequences=True,
                           return_state=False,
                           recurrent_initializer='glorot_uniform',
                           name='RecurrentUnit')
    )

  def call(self, inputs):
    x = self.embedding(inputs)
    return self.lstm(x)


class Attention(tf.keras.layers.Layer):
  """
  Attention layer. It receives the current position encoding (meaning what point in the output sequence we are
  generating) and all possible options to pay "attention" to. Uses Bahdanau's additive style for the score.
  """

  def __init__(self, units, name='Attention', **kwargs):
    super(Attention, self).__init__(name=name, **kwargs)

    # Weights for position encoding (always the same for all options, i.e., encoder outputs). The first part of the
    # input vector is always the same position encoding from the decoder hidden state
    self.W1 = tf.keras.layers.Dense(units)  # --> shape [decoder_latent, units]

    # Weights for each encoder output
    self.W2 = tf.keras.layers.Dense(units)  # --> shape [encoder_latent, units]

    # Weights to compute linear combination and then apply softmax
    self.V = tf.keras.layers.Dense(1)  # --> shape [units, 1]

  def call(self, position, options):
    # Represents position encoding (expanded a dimension to broadcast addition along the time axis). This position
    # vector repeats for every output of the encoder in the options param.
    position = tf.expand_dims(position, axis=1)  # --> shape [batch, 1, decoder_latent]

    # Performs network forward path combining options [batch, sequence, encoder_latent]
    score = tf.nn.tanh(self.W1(position) + self.W2(options))  # --> shape [batch, sequence, units]
    score = self.V(score)  # --> shape [batch, sequence, 1]

    # Computes attention weights (softmax along the time dimension, i.e., number of options)
    weights = tf.nn.softmax(score, axis=1)  # --> shape [batch, sequence, 1]

    # Multiplies every option by its weight (broadcasts the weight along the encoder_latent dimension,
    # meaning multiply the entire vector by a scalar)
    context = weights * options  # --> shape [batch, sequence, encoder_latent]

    # Computes sum along the sequence dimension (weighted sum of all options)
    context = tf.reduce_sum(context, axis=1)  # --> shape [batch, encoder_latent]

    return context, weights


class Decoder(tf.keras.layers.Layer):
  """
  Decoder layer implementing attention mechanism. It generates a context vector using an Attention layer for every step
  in the output sequence. Uses context vector concatenated with true previous word embedding (teacher forcing) as the
  input of a LSTM layer. The latter has initial state of all zeros.
  """

  def __init__(self,
               latent_size: int,
               attention_size: int,
               vocab: dict,
               embedding_size=None,
               embedding_matrix=None,
               name='Decoder',
               **kwargs):
    super(Decoder, self).__init__(name=name, **kwargs)

    # Stores for later use
    self.vocab = vocab
    self.latent_size = latent_size
    self.attention_size = attention_size

    # Defines embedding layer
    if embedding_size is not None:
      self.embedding = tf.keras.layers.Embedding(len(self.vocab), embedding_size, name='Embedding')
    elif embedding_matrix is not None:
      num_vectors, embedding_size = embedding_matrix.shape
      if num_vectors != len(self.vocab):
        raise Exception('Vocabulary (length {}) and embedding matrix (Vectors {}) are not compatible'.format(
          len(self.vocab), num_vectors))
      self.embedding = tf.keras.layers.Embedding(len(self.vocab), embedding_size,
                                                 weights=[embedding_matrix],
                                                 trainable=False,
                                                 name='Embedding')
    else:
      raise Exception("""Must provide either embedding_dim (embeddings will be learned) or embedding
          matrix (reusing embeddings)""")

    # Defines LSTM layer
    self.lstm = tf.keras.layers.LSTM(self.latent_size, return_sequences=False,
                                     return_state=True,
                                     recurrent_initializer='glorot_uniform',
                                     name='RecurrentUnit')

    # Defines attention layer
    self.attention = Attention(self.attention_size)

    # Defines Dense (to get the prob of each word)
    self.dense = tf.keras.layers.Dense(len(self.vocab), activation='softmax')

  def call(self, targets, options, state, training=None):
    # Goes through every element in the target sequence (one by one)
    output = []
    for step in range(0, targets.shape[1]):
      # Gets current target embedding
      x = targets[:, step]
      x = self.embedding(x)  # --> shape [batch, embedding_size]

      # Expands dimension to make it a sequence for the lstm layer
      x = tf.expand_dims(x, 1)  # --> shape [batch, 1, embedding_size]

      # Gets context (for every ons in batch) performing attention
      context, attention = self.attention(state[0], options)

      # Makes context a sequence of length one
      context = tf.expand_dims(context, axis=1)  # --> shape [batch, 1, encoder_latent]

      # Concatenates context and the expected output (teacher forcing)
      x = tf.concat([context, x], axis=-1)  # --> shape [batch, 1, encoder_latent + embedding_size]

      # Pass input through lstm layer
      x, hidden, cell = self.lstm(x, initial_state=state)

      # Stores state for next output in sequence
      state = [hidden, cell]

      # Stores output in list
      output.append(tf.expand_dims(self.dense(x), axis=1))

    if training:
      return tf.concat(output, axis=1), state
    else:
      return tf.concat(output, axis=1), state, attention


class Translator:
  def __init__(self,
               source_tokenizer: Tokenizer,
               target_tokenizer: Tokenizer,
               source_latent_dim=200,
               target_latent_dim=200,
               source_embedding_size=None,
               target_embedding_size=None,
               source_embedding_matrix=None,
               target_embedding_matrix=None,
               attention_size=100,
               max_output_length=100,
               restore=True,
               masking=False):

    # Stored for later use
    self.source_tokenizer = source_tokenizer
    self.target_tokenizer = target_tokenizer
    self.masking = masking

    # Encoder/Decoder layers
    self.encoder = Encoder(source_latent_dim,
                           self.source_tokenizer.word_to_index,
                           source_embedding_size,
                           source_embedding_matrix)
    self.decoder = Decoder(target_latent_dim,
                           attention_size,
                           self.target_tokenizer.word_to_index,
                           target_embedding_size,
                           target_embedding_matrix)

    # Optimizer and loss function
    self.entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
    self.optimizer = tf.keras.optimizers.Adam()

    # Maximum length of the output sequence (use at inference time)
    self.max_output_length = max_output_length

    # Checkpoint configuration
    checkpoint_dir = './checkpoints'
    self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                          encoder=self.encoder,
                                          decoder=self.decoder)
    if restore:
      self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

  @tf.function
  def compute_batch_loss(self, y_true, y_pred, masking):
    """
    Compute the loss in a batch

    :param y_true: Tensor of size [batch_size, max_seq_length]
    :param y_pred: Tensor of size [batch_size, max_seq_length, vocab_size]
    :param masking: Indicates whether to skip the padded portion of the targets
    :return: Batch loss
    """
    loss = 0
    if masking:
      # We have to go one step at a time since observations might have different sequence length
      for step in range(0, y_true.shape[1]):
        # Create a mask if the current step in the sequence is a padded word
        mask = tf.logical_not(tf.equal(y_true[:, step], self.target_tokenizer.word_to_index[b'<unknown>']))
        entropy = self.entropy(y_true[:, step], y_pred[:, step, :])
        loss += tf.reduce_sum(tf.cast(mask, entropy.dtype) * entropy)
    else:
      loss = tf.reduce_sum(self.entropy(y_true, y_pred))

    return loss

  @tf.function
  def test_step(self, sources, targets):
    # Computes encoder outputs for the given input
    options = self.encoder(sources)

    # Calling decoder passing the target as input removing <end> from it
    zeros = tf.zeros([targets.shape[0], self.decoder.latent_size])
    predictions, _ = self.decoder(targets[:, :-1], options, [zeros, zeros], training=True)

    # Computes loss in batch (the <start> token should not be part of the expected output)
    batch_loss = self.compute_batch_loss(targets[:, 1:], predictions, self.masking)

    return batch_loss, targets[:, 1:], tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

  @tf.function
  def train_step(self, sources, targets):
    """
    Performs a single training step for a batch using teacher forcing.

    :param sources: Input sequences in the batch
    :param targets: Expected sequences in the batch
    """
    with tf.GradientTape() as tape:
      # Computes encoder outputs for the given input
      options = self.encoder(sources)

      # Calling decoder passing the target as input removing <end> from it
      zeros = tf.zeros([targets.shape[0], self.decoder.latent_size])
      predictions, _ = self.decoder(targets[:, :-1], options, [zeros, zeros], training=True)

      # Computes loss in batch (the <start> token should not be part of the expected output)
      batch_loss = self.compute_batch_loss(targets[:, 1:], predictions, self.masking)

    # Determines gradients
    variables = self.encoder.trainable_variables + self.decoder.trainable_variables
    gradients = tape.gradient(batch_loss, variables)

    # Updates params
    self.optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss, targets[:, 1:], tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

  def train(self, epochs: int, train: tf.data.Dataset, test=None) -> None:
    """
    Performs training of the translation model. It shows training/test loss and accuracy after each epoch.

    :param epochs: Number of epochs
    :param train: Training dataset
    :param test: Test dataset
    """
    for epoch in range(epochs):
      # Performing a training epoch
      start = time.perf_counter()
      train_loss = 0
      train_matches = 0
      train_possible = 0
      train_predicted_length = 0
      train_expected_length = 0
      for batch, (sources, targets) in enumerate(train):
        # Calls model
        batch_loss, expected, predicted = self.train_step(sources, targets)
        # Update loss and accuracy data for logging
        train_loss += batch_loss
        # Computes BLEU score necessary data
        matches, possible, predicted_length, expected_length = get_counts(
          expected.numpy(), predicted.numpy(), ending_token=self.decoder.vocab[b'<end>']
        )
        train_matches += matches
        train_possible += possible
        train_predicted_length += predicted_length
        train_expected_length += expected_length
      # Computes BLEU score
      bleu = get_bleu(train_matches, train_possible, train_predicted_length, train_expected_length)

      # Logs training results
      print('\nEpoch {} out of {} complete ({:.2f} secs) -- Train Loss: {:.4f} -- Train Bleu: {:.2f}'.format(
        epoch + 1,
        epochs,
        time.perf_counter() - start,
        train_loss / (batch + 1),
        bleu
      ), end='')

      if test is not None:
        # Evaluates performance on test set after epoch training
        test_loss = 0
        test_matches = 0
        test_possible = 0
        test_predicted_length = 0
        test_expected_length = 0
        for batch, (sources, targets) in enumerate(test):
          # Calls model
          batch_loss, batch_positives, batch_samples = self.test_step(sources, targets)
          # Update loss for logging
          test_loss += batch_loss
          # Computes BLEU score necessary data
          matches, possible, predicted_length, expected_length = get_counts(
            expected.numpy(), predicted.numpy(), ending_token=self.decoder.vocab[b'<end>']
          )
          test_matches += matches
          test_possible += possible
          test_predicted_length += predicted_length
          test_expected_length += expected_length
        # Computes BLEU score
        bleu = get_bleu(test_matches, test_possible, test_predicted_length, test_expected_length)
        # Logs test performance
        if batch >= 0:
          print(' -- Test Loss: {:.4f} -- Test Bleu: {:.2f}'.format(
            test_loss / (batch + 1),
            bleu
          ), end='')

      # Save checkpoint every ten epochs
      if (epoch + 1) % 10 == 0:
        print('\nCreating intermediate checkpoint!')
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    # Save weights after training is done
    print('\nCreating final checkpoint!')
    self.checkpoint.save(file_prefix=self.checkpoint_prefix)

  def translate(self, sources, return_attention=False):
    # Prediction (indexes of words)
    output = []
    options = self.encoder(sources)
    # Creates the <start> token to give the decoder first
    target = tf.expand_dims([self.decoder.vocab[b'<start>']], 0)
    # Generates next word
    zeros = tf.zeros([1, self.decoder.latent_size])
    state = [zeros, zeros]
    attention = []
    for i in range(self.max_output_length):
      prediction, state, weights = self.decoder(target, options, state)
      word = tf.math.argmax(tf.squeeze(prediction)).numpy()
      attention.append(weights)
      output.append(word)
      # If word is <end> token finish
      if self.decoder.vocab[b'<end>'] == word:
        break
      target = tf.expand_dims([word], 0)

    if return_attention:
      return output, tf.squeeze(tf.concat(attention, 0)).numpy()
    else:
      return output

  def evaluate(self, dataset):
    total_matches = 0
    total_possible = 0
    total_predicted_length = 0
    total_expected_length = 0
    for batch in dataset:
      for source, target in zip(batch[0], batch[1]):
        # Prepares input
        source = tf.expand_dims(source, 0)
        # Prints expected translation
        words = []
        for word in target.numpy():
          decoded = self.target_tokenizer.index_to_word[word].decode()
          words.append(decoded)
          if decoded == '<end>':
            break
        print('Expected:', ' '.join(words[1:-1]))
        reference = np.array(words[1:], ndmin=2)
        # Prints actual translation
        words = []
        prediction, attention = self.translate(source, return_attention=True)
        for word in prediction:
          decoded = self.target_tokenizer.index_to_word[word].decode()
          words.append(decoded)
        print('Translation:', ' '.join(words[:-1]), end='\n\n')
        # Updates data for BLEU score computation
        candidate = np.array(words, ndmin=2)
        matches, possible, predicted_length, expected_length = get_counts(
          candidate, reference
        )
        total_matches += matches
        total_possible += possible
        total_predicted_length += predicted_length
        total_expected_length += expected_length
        # Plots attention
        plot_attention(attention,
                       tf.squeeze(source).numpy(),
                       prediction,
                       self.source_tokenizer.index_to_word,
                       self.target_tokenizer.index_to_word)

    # Computes BLEU score
    bleu = get_bleu(total_matches, total_possible, total_predicted_length, total_expected_length)
    print('Bleu:', bleu)
