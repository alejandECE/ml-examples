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
  def __init__(self, latent_size: int,
               vocab: dict,
               embedding_dim=None,
               embedding_matrix=None,
               name='Encoder',
               **kwargs):
    super(Encoder, self).__init__(name=name, **kwargs)
    # Stores for later use
    self.vocab = vocab
    vocab_size = len(self.vocab)
    self.latent_size = latent_size
    # Defines embedding layer
    if embedding_dim is not None:
      self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, name='Embedding')
    elif embedding_matrix is not None:
      num_vectors, embedding_dim = embedding_matrix.shape
      if num_vectors != vocab_size:
        raise Exception('Vocabulary (length {}) and embedding matrix (Vectors {}) are not compatible'.format(
          vocab_size, num_vectors))
      self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                 weights=[embedding_matrix],
                                                 trainable=False,
                                                 name='Embedding')
    else:
      raise Exception("""Must provide either embedding_dim (embeddings will be learned) or embedding
      matrix (reusing embeddings)""")
    # Defines LSTM layer
    self.lstm = tf.keras.layers.LSTM(self.latent_size,
                                     return_state=True,
                                     recurrent_initializer='glorot_uniform',
                                     name='RecurrentUnit')

  def call(self, inputs, state=None):
    x = self.embedding(inputs)
    if state is None:
      return self.lstm(x)
    else:
      return self.lstm(x, initial_state=state)


class Decoder(tf.keras.layers.Layer):
  def __init__(self,
               latent_size: int,
               vocab: dict,
               embedding_dim=None,
               embedding_matrix=None,
               name='Decoder',
               **kwargs):
    super(Decoder, self).__init__(name=name, **kwargs)
    # Stores for later use
    self.vocab = vocab
    vocab_size = len(self.vocab)
    self.latent_size = latent_size
    # Defines embedding layer
    if embedding_dim is not None:
      self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, name='Embedding')
    elif embedding_matrix is not None:
      num_vectors, embedding_dim = embedding_matrix.shape
      if num_vectors != vocab_size:
        raise Exception('Vocabulary (length {}) and embedding matrix (Vectors {}) are not compatible'.format(
          vocab_size, num_vectors))
      self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
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
    self.dense = tf.keras.layers.Dense(vocab_size, activation='softmax')

  def call(self, inputs, state, training=None):
    x = self.embedding(inputs)
    x, hidden, cell = self.lstm(x, initial_state=state)
    if training is not None:
      return self.dense(x)
    else:
      return self.dense(x), hidden, cell


class Translator:
  def __init__(self,
               source_vocab: dict,
               target_vocab: dict,
               source_latent_dim=256,
               target_latent_dim=256,
               source_embedding_dim=None,
               target_embedding_dim=None,
               source_embedding_matrix=None,
               target_embedding_matrix=None,
               max_output_length=100,
               restore=True):
    # Encoder/Decoder layers
    self.encoder = Encoder(source_latent_dim, source_vocab, source_embedding_dim, source_embedding_matrix)
    self.decoder = Decoder(target_latent_dim, target_vocab, target_embedding_dim, target_embedding_matrix)
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
  def train_step(self, sources, targets):
    with tf.GradientTape() as tape:
      # Init encoder state
      initial = tf.zeros((sources.shape[0], self.encoder.latent_size))
      # Computes thought vector using the encoder (represented as the hidden and cell state)
      _, hidden, cell = self.encoder(sources, [initial, initial])
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

  def train(self, epochs: int, dataset: tf.data.Dataset) -> None:
    with tf.device('gpu:0'):
      for epoch in range(epochs):
        start = time.perf_counter()
        epoch_loss = 0
        epoch_positives = 0
        epoch_samples = 0
        for batch, (sources, targets) in enumerate(dataset):
          # Calls model
          batch_loss, batch_positives, batch_samples = self.train_step(sources, targets)
          # Update loss and accuracy data for logging
          epoch_loss += batch_loss
          epoch_positives += batch_positives
          epoch_samples += batch_samples

        # Outputs log info
        end = time.perf_counter()
        print('Epoch {} out of {} complete ({:.2f} secs) -- Loss: {:.4f} -- Accuracy: {:.2f}'.format(
          epoch + 1,
          epochs,
          end - start,
          epoch_loss / (batch + 1),
          epoch_positives / epoch_samples
        ))

        # Save checkpoint every two epochs
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
