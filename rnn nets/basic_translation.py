# Created by Luis Alejandro (alejand@umich.edu)
import re
import time
import os
import tensorflow as tf
from utils import unicode_to_ascii
from dataset import Tokenizer


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


class Encoder(tf.keras.layers.Layer):
  """
  Encoder layer that contains an Embedding layer followed by a LSTM Layer. The latter creates a thought vector (hidden
  state) to be used by a Decoder layer.
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
    self.lstm = tf.keras.layers.LSTM(self.latent_size,
                                     return_sequences=False,
                                     return_state=True,
                                     recurrent_initializer='glorot_uniform',
                                     name='RecurrentUnit')

  def call(self, inputs, state=None):
    x = self.embedding(inputs)
    return self.lstm(x)


class Decoder(tf.keras.layers.Layer):
  """
  Decoder layer that contains an Embedding layer and a LSTM layer. We init the LSTM layer with the last hidden state of
  the Decoder layer.
  """
  def __init__(self,
               latent_size: int,
               vocab: dict,
               embedding_size=None,
               embedding_matrix=None,
               name='Decoder',
               **kwargs):
    super(Decoder, self).__init__(name=name, **kwargs)
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
    self.lstm = tf.keras.layers.LSTM(self.latent_size, return_sequences=True,
                                     return_state=True,
                                     recurrent_initializer='glorot_uniform',
                                     name='RecurrentUnit')

    # Defines Dense (to get the prob of each word)
    self.dense = tf.keras.layers.Dense(len(self.vocab), activation='softmax')

  def call(self, inputs, state, training=None):
    x = self.embedding(inputs)
    x, hidden, cell = self.lstm(x, initial_state=state)
    if training is not None:
      return self.dense(x)
    else:
      return self.dense(x), hidden, cell


class Translator:
  """
  Wrapper class to perform training and evaluation of a translation model using Encoder/Decoder layers. Teacher forcing
  is used during training and a translation can be obtained for any new input. Please refer to the notebook to see how
  to use this class.
  """
  def __init__(self,
               source_tokenizer: Tokenizer,
               target_tokenizer: Tokenizer,
               source_latent_dim=256,
               target_latent_dim=256,
               source_embedding_size=None,
               target_embedding_size=None,
               source_embedding_matrix=None,
               target_embedding_matrix=None,
               max_output_length=100,
               restore=True,
               masking=False,
               verbose=True):

    # Stored for later use
    self.source_tokenizer = source_tokenizer
    self.target_tokenizer = target_tokenizer
    self.masking = masking
    self.verbose = verbose

    # Encoder/Decoder layers
    self.encoder = Encoder(source_latent_dim,
                           self.source_tokenizer.word_to_index,
                           source_embedding_size,
                           source_embedding_matrix)
    self.decoder = Decoder(target_latent_dim,
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
    positives = 0
    samples = 0
    if masking:
      # We have to go one step at a time since observations might have different sequence length
      for step in range(0, y_true.shape[1]):
        # Create a mask if the current step in the sequence is a padded word
        mask = tf.logical_not(tf.equal(y_true[:, step], self.target_tokenizer.word_to_index[b'<unknown>']))
        entropy = self.entropy(y_true[:, step], y_pred[:, step, :])
        loss += tf.reduce_sum(tf.cast(mask, entropy.dtype) * entropy)
        hits = tf.equal(y_true[:, step], tf.cast(tf.argmax(y_pred[:, step, :], axis=-1), tf.int32))
        positives += tf.reduce_sum(tf.cast(hits, tf.int32))
        samples += tf.reduce_sum(tf.cast(mask, tf.int32))
    else:
      loss = tf.reduce_sum(self.entropy(y_true, y_pred))
      hits = tf.equal(y_true, tf.cast(tf.argmax(y_pred, axis=-1), tf.int32))
      positives = tf.reduce_sum(tf.cast(hits, tf.int32))
      samples = tf.size(y_true)

    return loss, positives, samples

  @tf.function
  def test_step(self, sources, targets):
    # Computes thought vector using the encoder (represented as the hidden and cell state)
    _, hidden, cell = self.encoder(sources)

    # Passing the target as input removing <end> from it)
    predictions = self.decoder(targets[:, :-1], [hidden, cell], training=True)

    # Computes loss in batch (the <start> token should not be part of the expected output)
    batch_loss, batch_positives, batch_samples = self.compute_batch_loss(targets[:, 1:], predictions, self.masking)

    return batch_loss, batch_positives, batch_samples

  @tf.function
  def train_step(self, sources, targets):
    """
    Performs a single training step for a batch using teacher forcing.

    :param sources: Input sequences in the batch
    :param targets: Expected sequences in the batch
    """
    with tf.GradientTape() as tape:
      # Computes thought vector using the encoder (represented as the hidden and cell state)
      _, hidden, cell = self.encoder(sources)

      # Teacher forcing (by passing the target as input removing <end> from it)
      predictions = self.decoder(targets[:, :-1], [hidden, cell], training=True)

      # Computes loss in batch (the <start> token should not be part of the expected output)
      batch_loss, batch_positives, batch_samples = self.compute_batch_loss(targets[:, 1:], predictions, self.masking)

    # Determines gradients
    variables = self.encoder.trainable_variables + self.decoder.trainable_variables
    gradients = tape.gradient(batch_loss, variables)

    # Updates params
    self.optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss, batch_positives, batch_samples

  def train(self, epochs: int, train: tf.data.Dataset, test: tf.data.Dataset) -> None:
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
      train_positives = 0
      train_samples = 0
      for batch, (sources, targets) in enumerate(train):
        # Calls model
        batch_loss, batch_positives, batch_samples = self.train_step(sources, targets)
        # Update loss and accuracy data for logging
        train_loss += batch_loss
        train_positives += batch_positives
        train_samples += batch_samples

      # Logs training results
      print('Epoch {} out of {} complete ({:.2f} secs) -- Train Loss: {:.4f} -- Train Acc: {:.2f}'.format(
        epoch + 1,
        epochs,
        time.perf_counter() - start,
        train_loss / (batch + 1),
        train_positives / train_samples
      ), end='')

      if self.verbose:
        # Evaluates performance on test set after epoch training
        test_loss = 0
        test_positives = 0
        test_samples = 0
        for batch, (sources, targets) in enumerate(test):
          # Calls model
          batch_loss, batch_positives, batch_samples = self.test_step(sources, targets)
          # Update loss and accuracy data for logging
          test_loss += batch_loss
          test_positives += batch_positives
          test_samples += batch_samples

        # Logs test performance
        if test_samples > 0:
          print(' -- Test Loss: {:.4f} -- Test Acc: {:.2f}'.format(
            test_loss / (batch + 1),
            test_positives / test_samples
          ))

      # Save checkpoint every ten epochs
      if (epoch + 1) % 10 == 0:
        print('Creating intermediate checkpoint!')
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    # Save weights after training is done
    print('Creating final checkpoint!')
    self.checkpoint.save(file_prefix=self.checkpoint_prefix)

  def translate(self, sources):
    # Prediction (indexes of words)
    output = []
    _, hidden, cell = self.encoder(sources)
    # Creates the <start> token to give the decoder first
    target = tf.expand_dims([self.decoder.vocab[b'<start>']], 0)
    # Generates next word
    for i in range(self.max_output_length):
      prediction, hidden, cell = self.decoder(target, [hidden, cell])
      word = tf.math.argmax(tf.squeeze(prediction)).numpy()
      output.append(word)
      # If word is <end> token finish
      if self.decoder.vocab[b'<end>'] == word:
        break
      target = tf.expand_dims([word], 0)
    return output

  def evaluate(self, dataset):
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
        # Prints actual translation
        words = []
        prediction = self.translate(source)
        for word in prediction:
          decoded = self.target_tokenizer.index_to_word[word].decode()
          words.append(decoded)
        print('Translation:', ' '.join(words[:-1]), end='\n\n')
