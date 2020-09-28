#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import tensorflow as tf
import utils


# Yolo loss function not only using the squared differences
class YoloLoss(tf.keras.losses.Loss):
  def __init__(self, mode='custom', **kwargs):
    super().__init__(**kwargs)
    self.mode = mode

  def call(self, y_true, y_pred):
    # Some constants
    lambda_coord = 5
    lambda_noobj = 0.5
    anchor_shapes = tf.convert_to_tensor(utils.ANCHORS_SHAPE)
    # Lets parse y_true since it comes as a 4D sparse tensor of dense shape [Batch Size, 7, 7, 90]
    y_expected = tf.sparse.to_dense(y_true)
    # Indices (of the last dimension) for the corresponding objectness values in a [B, 7, 7, 90] output
    indices = tf.range(0, tf.shape(anchor_shapes)[0]) * 5
    # Objectness per batch per cell per grounding box. Shape [B, 7, 7, A], where B is the batch size and A the number
    # of anchor shapes
    true_objectness = tf.gather(y_expected, indices=indices, axis=-1)
    pred_objectness = tf.gather(y_pred, indices=indices, axis=-1)
    objectness_loss = tf.reduce_sum(tf.square(true_objectness - pred_objectness), axis=-1)
    # Indices (of the last dimension) for the corresponding boxes in a [B, 7, 7, 90] output
    indices = tf.tile(tf.convert_to_tensor([[1, 2, 3, 4]], dtype=tf.int32), multiples=[tf.shape(anchor_shapes)[0], 1])
    indices += tf.expand_dims(tf.range(tf.shape(anchor_shapes)[0], dtype=tf.int32) * 5, axis=-1)
    indices = tf.reshape(indices, shape=(-1,))
    true_bboxes = tf.gather(y_expected, indices=indices, axis=-1)
    pred_bboxes = tf.gather(y_pred, indices=indices, axis=-1)
    # Masking out the bounding boxes not corresponding to the active anchor. Shape [B, 7, 7, A]
    bboxes_mask = tf.cast(tf.not_equal(true_objectness, 0), dtype=tf.int32)
    bboxes_mask = tf.repeat(bboxes_mask, repeats=4, axis=-1)
    # Computes the loss corresponding to the bounding using the square of the differences. Shape [B, 7, 7]
    bboxes_loss = tf.reduce_sum(
      tf.square(true_bboxes - tf.cast(bboxes_mask, dtype=pred_bboxes.dtype) * pred_bboxes),
      axis=-1
    )
    # Retrieves expected and predicted labels
    indices = tf.range(5 * tf.shape(anchor_shapes)[0], tf.shape(y_expected)[3])
    true_class = tf.gather(y_expected, indices=indices, axis=-1)
    pred_class = tf.gather(y_pred, indices=indices, axis=-1)
    if self.mode == 'custom':
      class_loss = tf.nn.softmax_cross_entropy_with_logits(labels=true_class, logits=pred_class)
    else:
      class_loss = tf.reduce_sum(tf.square(true_class - pred_class), axis=-1)
    # Masks that represent whether there is an object or not in the cell. Shape [B, 7, 7]
    obj_mask = tf.reduce_any(tf.not_equal(true_objectness, 0), axis=-1)
    noobj_mask = tf.logical_not(obj_mask)
    # Computes total loss. Shape [B,]
    return tf.reduce_sum(
      lambda_coord * tf.cast(obj_mask, dtype=bboxes_loss.dtype) * bboxes_loss +
      tf.cast(obj_mask, dtype=objectness_loss.dtype) * objectness_loss +
      lambda_noobj * tf.cast(noobj_mask, dtype=objectness_loss.dtype) * objectness_loss +
      tf.cast(obj_mask, dtype=objectness_loss.dtype) * class_loss,
      axis=[1, 2]
    )
