
import tensorflow as tf


def loss(preds, labels):
    loss = (tf.sqrt(tf.reduce_mean((preds - labels)) ** 2) + 5 * 10 ** -4 * tf.nn.l2_loss(preds - labels))
    return tf.reduce_mean(loss)


def accuracy(preds, labels):
  
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)

