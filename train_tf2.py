#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys

LOG_LEVEL_INDEX = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = sys.argv[LOG_LEVEL_INDEX] if 0 < LOG_LEVEL_INDEX < len(sys.argv) else '3'

import absl.app
import math
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import tensorflow.compat.v2 as tfv2

from functools import partial
from six.moves import zip, range
from tensorflow.keras import layers
from util.config import Config, initialize_globals
from util.feeding import create_dataset
from util.flags import create_flags, FLAGS
from util.logging import log_info, log_error, log_debug, log_progress, create_progressbar


def create_overlapping_windows(batch_x):
    batch_size = tf.shape(input=batch_x)[0]
    window_width = 2 * Config.n_context + 1
    num_channels = Config.n_input

    # Create a constant convolution filter using an identity matrix, so that the
    # convolution returns patches of the input tensor as is, and we can create
    # overlapping windows over the MFCCs.
    eye_filter = tf.constant(np.eye(window_width * num_channels)
                               .reshape(window_width, num_channels, window_width * num_channels), tf.float32) # pylint: disable=bad-continuation

    # Create overlapping windows
    batch_x = tf.nn.conv1d(input=batch_x, filters=eye_filter, stride=1, padding='SAME')

    # Remove dummy depth dimension and reshape into [batch_size, n_windows, window_width, n_input]
    batch_x = tf.reshape(batch_x, [batch_size, -1, window_width, num_channels])

    return batch_x

class Dense(tf.Module):
    def __init__(self, units, input_shape, dropout_rate=None, relu=True, name=None):
        super(Dense, self).__init__(name=name)

        self.dropout_rate = dropout_rate
        self.relu = relu

        with self.name_scope:
            # Bias initialized to zero
            self.bias = tfv2.Variable(tf.zeros([units]))

            # Weights initialized with Glorot uniform initialization
            limit = math.sqrt(6 / (input_shape[-1] + units))
            init_val = tf.random.uniform([x.shape[-1], units], minval=-limit, maxval=limit)
            self.weights = tfv2.Variable(init_val)


    def __call__(x):
        y = tf.nn.bias_add(tf.matmul(x, self.weights), bias)
        if self.relu:
            y = tf.minimum(tf.nn.relu(y), FLAGS.relu_clip)

        if self.dropout_rate:
            y = tfv2.nn.dropout(y, rate=self.dropout_rate)

        return y


class DeepSpeechModel(tf.Module):
    def __init__(self, input_shape):
        super(DeepSpeechModel, self).__init__()

        self.dense1 = Dense(Config.n_hidden_1, activation=clipped_relu)
        self.drop1 = layers.Dropout(rate=FLAGS.dropout_rate)

        self.dense2 = Dense(Config.n_hidden_2, activation=clipped_relu)
        self.drop2 = layers.Dropout(rate=FLAGS.dropout_rate2)

        self.dense3 = Dense(Config.n_hidden_3, activation=clipped_relu)
        self.drop3 = layers.Dropout(rate=FLAGS.dropout_rate3)

        self.lstm = tfv2.keras.layers.LSTM(Config.n_cell_dim, return_sequences=True)

        self.dense5 = Dense(Config.n_hidden_5, activation=clipped_relu)
        self.drop5 = layers.Dropout(rate=FLAGS.dropout_rate5)

        self.dense6 = Dense(Config.n_hidden_6, activation=tf.nn.softmax)


    def call(self, x, training=False):
        # print('input shape:', x.shape)
        if self.create_windows:
            x = self.create_windows(x)
            # print('created windows:', x.shape)

        x = self.reshape_init(x)
        # print('reshaped:', x.shape)
        # x = self.mask(x)
        # print('masked:', x.shape)

        x = self.dense1(x)
        # print('dense1:', x.shape)
        if training:
            x = self.drop1(x, training=training)
            # print('drop1:', x.shape)

        x = self.dense2(x)
        # print('dense2:', x.shape)
        if training:
            x = self.drop2(x, training=training)
            # print('drop2:', x.shape)

        x = self.dense3(x)
        # print('dense3:', x.shape)
        if training:
            x = self.drop3(x, training=training)
            # print('drop3:', x.shape)

        x = self.lstm(x, training=training)
        # print('lstm:', x.shape)

        x = self.dense5(x)
        # print('dense5:', x.shape)
        if training:
            x = self.drop5(x, training=training)
            # print('drop5:', x.shape)

        x = self.dense6(x)
        # print('output:', x.shape)
        return x


def main(_):
    initialize_globals()

    # model = DeepSpeechModel(create_windows=True)
    model = create_model()
    print(model.summary())
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

    train_set = create_dataset(FLAGS.train_files.split(','),
                               batch_size=FLAGS.train_batch_size,
                               cache_path=FLAGS.feature_cache)

    if FLAGS.dev_files:
        dev_csvs = FLAGS.dev_files.split(',')
        dev_sets = [create_dataset([csv], batch_size=FLAGS.dev_batch_size) for csv in dev_csvs]
        dev_init_ops = [iterator.make_initializer(dev_set) for dev_set in dev_sets]

    for epoch in range(FLAGS.epochs):
        for step, (_, batch_x, batch_x_lens, batch_y, batch_y_lens) in enumerate(train_set):
            with tf.GradientTape() as tape:
                logits = model(batch_x, training=True)

                loss = tfv2.nn.ctc_loss(labels=batch_y,
                                        label_length=batch_y_lens,
                                        logits=logits,
                                        logit_length=batch_x_lens,
                                        blank_index=Config.n_hidden_6 - 1,
                                        logits_time_major=False)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            print('Epoch {:>3} - Step {:>3} - Training loss: {:.3f}'.format(epoch, int(step), float(loss)))


if __name__ == '__main__':
    create_flags()
    absl.app.run(main)
