# Created by Luis Alejandro (alejand@umich.edu)
import re
import time
import os
import tensorflow as tf
from utils import unicode_to_ascii


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
               source_vocab: dict,
               target_vocab: dict,
               source_latent_dim=256,
               target_latent_dim=256,
               source_embedding_size=None,
               target_embedding_size=None,
               source_embedding_matrix=None,
               target_embedding_matrix=None,
               max_output_length=100,
               restore=True):
    # Encoder/Decoder layers
    self.encoder = Encoder(source_latent_dim, source_vocab, source_embedding_size, source_embedding_matrix)
    self.decoder = Decoder(target_latent_dim, target_vocab, target_embedding_size, target_embedding_matrix)

    # Optimizer and loss function
    self.loss_fcn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
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
  def get_sparse_positives(self, y_true, y_pred):
    positives = tf.equal(y_true, tf.cast(tf.math.argmax(y_pred, axis=2), tf.int32))
    return tf.reduce_sum(tf.cast(positives, tf.int32)), y_true.shape[0] * y_true.shape[1]

  @tf.function
  def test_step(self, sources, targets):
    # Computes thought vector using the encoder (represented as the hidden and cell state)
    _, hidden, cell = self.encoder(sources)

    # Passing the target as input removing <end> from it)
    predictions = self.decoder(targets[:, :-1], [hidden, cell], training=True)

    # Computes loss in batch
    batch_loss = self.loss_fcn(targets[:, 1:], predictions)

    # Determines true and false positives
    batch_positives, batch_samples = self.get_sparse_positives(targets[:, 1:], predictions)

    return batch_loss, batch_positives, batch_samples

  @tf.function
  def train_step(self, sources, targets):
    with tf.GradientTape() as tape:
      # Computes thought vector using the encoder (represented as the hidden and cell state)
      _, hidden, cell = self.encoder(sources)

      # Teacher forcing (by passing the target as input removing <end> from it)
      predictions = self.decoder(targets[:, :-1], [hidden, cell], training=True)

      # Computes loss in batch
      batch_loss = self.loss_fcn(targets[:, 1:], predictions)

    # Determines gradients
    variables = self.encoder.trainable_variables + self.decoder.trainable_variables
    gradients = tape.gradient(batch_loss, variables)

    # Updates params
    self.optimizer.apply_gradients(zip(gradients, variables))

    # Determines true and false positives
    batch_positives, batch_samples = self.get_sparse_positives(targets[:, 1:], predictions)

    return batch_loss, batch_positives, batch_samples

  def train(self, epochs: int, train: tf.data.Dataset, test: tf.data.Dataset) -> None:
    with tf.device('gpu:0'):
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
      # If word is <end> token finish
      if self.decoder.vocab[b'<end>'] == word:
        break
      output.append(word)
      target = tf.expand_dims([word], 0)
    return output
