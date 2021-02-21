import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import SegNet
import scipy.io as scio
import time
import glob
import numpy as np
from PIL import Image
import math
from xml.etree.ElementTree import Element,ElementTree
import warnings

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

WORKING_DIR = os.getcwd()
TRAINING_DIR = os.path.join(WORKING_DIR, 'Data', 'Training')
TEST_DIR = os.path.join(WORKING_DIR, 'Data', 'Test')
ROOT_LOG_DIR = os.path.join(WORKING_DIR, 'Output')
RUN_NAME = "model"
LOG_DIR = os.path.join(ROOT_LOG_DIR, RUN_NAME)
TRAIN_WRITER_DIR = os.path.join(LOG_DIR, 'Training')
TEST_WRITER_DIR = os.path.join(LOG_DIR, 'Test')
CHECKPOINT_FN = 'model.ckpt'
CHECKPOINT_FL = os.path.join(LOG_DIR, CHECKPOINT_FN)

BATCH_NORM_DECAY = 0.95
MAX_STEPS = 30000
BATCH_SIZE = 4
SAVE_INTERVAL = 200

import logging

logging.basicConfig(format='%(asctime)s  %(message)s', level=logging.INFO)


def count_flops(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    print('FLOPs: {}'.format(flops.total_float_ops))


def count():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    logging.info('--------------------------------------------------')
    logging.info('Model size(# params): ' + str(total_parameters))
    logging.info('--------------------------------------------------')
    return total_parameters

def main():
    training_data = SegNet.GetData(TRAINING_DIR)
    test_data = SegNet.GetData(TEST_DIR)

    g = tf.Graph()

    with g.as_default():

        images, labels, is_training = SegNet.placeholder_inputs(batch_size=BATCH_SIZE)

        arg_scope = SegNet.inference_scope(is_training=True, batch_norm_decay=BATCH_NORM_DECAY)

        with slim.arg_scope(arg_scope):
            logits = SegNet.inference(images, class_inc_bg=3)

        SegNet.add_output_images(images=images, logits=logits, labels=labels)

        loss = SegNet.loss_calc(logits=logits, labels=labels)

        train_op, global_step = SegNet.training(loss=loss, learning_rate=1e-04)

        total_parameters = count()

        flops = count_flops(g)

        accuracy, out_label = SegNet.evaluation(logits=logits, labels=labels)

        iou_class0, iou_class1, iou_class2 = SegNet.iou(logits=logits, labels=labels)

        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver([x for x in tf.global_variables() if 'Adam' not in x.name])

        sm = tf.train.SessionManager()

        with sm.prepare_session("", init_op=init, saver=saver, checkpoint_dir=LOG_DIR) as sess:

            sess.run(tf.variables_initializer([x for x in tf.global_variables() if 'Adam' in x.name]))

            train_writer = tf.summary.FileWriter(TRAIN_WRITER_DIR, sess.graph)

            test_writer = tf.summary.FileWriter(TEST_WRITER_DIR)

            global_step_value, = sess.run([global_step])

            for step in range(global_step_value+1, global_step_value+MAX_STEPS+1):

                print("Iteration: ", step)

                images_batch, labels_batch = training_data.next_batch(BATCH_SIZE)

                train_feed_dict = {images: images_batch,
                                   labels: labels_batch,
                                   is_training: True}

                _, train_loss_value, train_accuracy_value, train_summary_str = sess.run([train_op, loss, accuracy, summary], feed_dict=train_feed_dict)

                if step % SAVE_INTERVAL == 0:

                    print("Train Loss: ", train_loss_value)
                    print("Train accuracy: ", train_accuracy_value)
                    train_writer.add_summary(train_summary_str, step)
                    train_writer.flush()

                    images_batch, labels_batch = test_data.next_batch(BATCH_SIZE)

                    test_feed_dict = {images: images_batch,
                                      labels: labels_batch,
                                      is_training: False}

                    test_loss_value, test_accuracy_value, test_iou_class0, test_iou_class1, test_iou_class2, test_summary_str= sess.run([loss, accuracy, iou_class0, iou_class1, iou_class2, summary], feed_dict=test_feed_dict)

                    test_mean_iou = (test_iou_class0+test_iou_class1+test_iou_class2) / 3
                    print("Test Loss: ", test_loss_value)
                    print("Test accuracy: ", test_accuracy_value)
                    print("Test oiu_class0: ", test_iou_class0)
                    print("Test oiu_class1: ", test_iou_class1)
                    print("Test oiu_class2: ", test_iou_class2)
                    print("Test mean_iou: ", test_mean_iou)

                    test_writer.add_summary(test_summary_str, step)
                    test_writer.flush()

                    saver.save(sess, CHECKPOINT_FL, global_step=step)
                    print("Session Saved")
                    print("================")


if __name__ == '__main__':
    main()
