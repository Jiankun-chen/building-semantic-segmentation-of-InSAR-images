import tensorflow as tf
import numpy as np


def loss_calc(logits, labels):
    #labels1 = tf.one_hot(labels, 6, axis=-1)
    #labels1 = tf.one_hot(labels, 6)              #对于RGB标签
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss', loss)
    return loss


def evaluation(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 3), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    out_label1 = tf.argmax(logits, 3)
    out_label = tf.transpose(out_label1, perm=[1, 2, 0])
    return accuracy, out_label


def iou(logits, labels):
    stand_0 = tf.zeros([256, 256], tf.float32)
    stand_1 = tf.ones([256, 256], tf.float32)
    stand_2 = tf.multiply(tf.cast(2.0, tf.float32), tf.ones([256, 256], tf.float32))
    predict_0 = tf.cast(tf.equal(tf.cast(tf.argmax(logits, 3), tf.float32), stand_0), tf.float32)
    predict_1 = tf.cast(tf.equal(tf.cast(tf.argmax(logits, 3), tf.float32), stand_1), tf.float32)
    predict_2 = tf.cast(tf.equal(tf.cast(tf.argmax(logits, 3), tf.float32), stand_2), tf.float32)
    truth_0 = tf.cast(tf.equal(tf.cast(labels, tf.float32), stand_0), tf.float32)
    truth_1 = tf.cast(tf.equal(tf.cast(labels, tf.float32), stand_1), tf.float32)
    truth_2 = tf.cast(tf.equal(tf.cast(labels, tf.float32), stand_2), tf.float32)

    intersection_0 = tf.multiply(predict_0, truth_0)
    union_0 = stand_1-tf.cast(tf.equal(predict_0+truth_0, stand_0), tf.float32)
    iou_class0 = tf.reduce_mean(intersection_0) / tf.reduce_mean(union_0)
    intersection_1 = tf.multiply(predict_1, truth_1)
    union_1 = stand_1-tf.cast(tf.equal(predict_1+truth_1, stand_0), tf.float32)
    iou_class1 = tf.reduce_mean(intersection_1)/tf.reduce_mean(union_1)
    intersection_2 = tf.multiply(predict_2, truth_2)
    union_2 = stand_1-tf.cast(tf.equal(predict_2 + truth_2, stand_0), tf.float32)
    iou_class2 = tf.reduce_mean(intersection_2) / tf.reduce_mean(union_2)
    tf.summary.scalar('iou_class0', iou_class0)
    tf.summary.scalar('iou_class1', iou_class1)
    tf.summary.scalar('iou_class2', iou_class2)

    return iou_class0, iou_class1, iou_class2
