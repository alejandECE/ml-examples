#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import tensorflow as tf
import numpy as np
import losses

if __name__ == '__main__':
  # Builds expected output
  indices = tf.concat([
    tf.zeros(shape=(6, 1), dtype=tf.int64),
    tf.repeat(tf.convert_to_tensor([[1, 1]], dtype=tf.int64), repeats=6, axis=0),
    tf.reshape(tf.cast(
      tf.convert_to_tensor(list(range(5, 10)) + [10 + 2]),
      dtype=tf.int64), shape=[-1, 1]
    )
  ], axis=-1)
  values = tf.convert_to_tensor([0.2, 0., 0.2, 0.1, 0.1, 1], dtype=tf.float32)
  y_true = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=[1, 7, 7, 90])
  # Builds predicted output
  predicted = np.zeros(shape=[7, 7, 90])
  predicted[2, 2, 5] = 0.3
  predicted[2, 2, 6:10] = [0.2, 0.2, 0.1, 0.1]
  predicted[2, 2, 10 + 2] = 1
  y_pred = tf.expand_dims(tf.convert_to_tensor(predicted, dtype=tf.float32), axis=0)
  loss = losses.YoloLoss()(y_true, y_pred)
  print(loss)
