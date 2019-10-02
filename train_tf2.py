#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys

LOG_LEVEL_INDEX = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = sys.argv[LOG_LEVEL_INDEX] if 0 < LOG_LEVEL_INDEX < len(sys.argv) else '3'

import absl.app
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


class CreateOverlappingWindows(tf.keras.Model):
    def __init__(self):
        super(CreateOverlappingWindows, self).__init__()
        window_width = 2 * Config.n_context + 1
        num_channels = Config.n_input
        identity = (np.eye(window_width * num_channels)
                      .reshape(window_width, num_channels, window_width * num_channels))
        self.identity_filter = tfv2.constant(identity, tf.float32)
        self.reshape = layers.Reshape((-1, window_width * num_channels))

    def call(self, x):
        x = tf.nn.conv1d(input=x, filters=self.identity_filter, stride=1, padding='SAME')
        return self.reshape(x)


def create_model():
    inputs = tf.keras.Input(shape=(None, Config.n_input))

    x = CreateOverlappingWindows()(inputs)
    x = layers.Masking()(x)

    clipped_relu = partial(tf.keras.activations.relu, max_value=FLAGS.relu_clip)

    x = layers.Dense(Config.n_hidden_1, activation=clipped_relu)(x)
    x = layers.Dropout(rate=FLAGS.dropout_rate)(x)
    x = layers.Dense(Config.n_hidden_2, activation=clipped_relu)(x)
    x = layers.Dropout(rate=FLAGS.dropout_rate2)(x)
    x = layers.Dense(Config.n_hidden_3, activation=clipped_relu)(x)
    x = layers.Dropout(rate=FLAGS.dropout_rate3)(x)

    x = tfv2.keras.layers.LSTM(Config.n_cell_dim, return_sequences=True)(x)

    x = layers.Dense(Config.n_hidden_5, activation=clipped_relu)(x)
    x = layers.Dropout(rate=FLAGS.dropout_rate5)(x)
    x = layers.Dense(Config.n_hidden_6)(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name='DeepSpeechModel')


def main(_):
    initialize_globals()
    train_set = create_dataset(FLAGS.train_files.split(','),
                               batch_size=FLAGS.train_batch_size,
                               cache_path=FLAGS.feature_cache)

    strategy = tf.distribute.MirroredStrategy()
    train_set = strategy.experimental_distribute_dataset(train_set)

    with strategy.scope():
        model = create_model()
        model.summary()
        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

        def step_fn(inputs):
            batch_x, batch_x_lens, batch_y, batch_y_lens = inputs

            with tf.GradientTape() as tape:
                logits = model(batch_x, training=True)

                loss = tfv2.nn.ctc_loss(labels=batch_y,
                                        label_length=batch_y_lens,
                                        logits=logits,
                                        logit_length=batch_x_lens,
                                        blank_index=Config.n_hidden_6 - 1,
                                        logits_time_major=False)

                mean_loss = tf.reduce_sum(loss) * (1.0 / (FLAGS.train_batch_size * len(Config.available_devices)))

            grads = tape.gradient(mean_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        @tf.function
        def dist_train_step(dataset_inputs):
            per_replica_losses = strategy.experimental_run_v2(step_fn, args=(dataset_inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        for epoch in range(FLAGS.epochs):
            total_loss = 0.0
            num_steps = 0
            for step, inputs in enumerate(train_set):
                total_loss += dist_train_step(inputs)
                num_steps += 1
                print('Epoch {:>3} - Step {:>3} - Avg. training loss: {:.3f}'.format(epoch, step, float(total_loss)/num_steps))

    # for epoch in range(FLAGS.epochs):
    #     for step, (_, batch_x, batch_x_lens, batch_y, batch_y_lens) in enumerate(train_set):
    #         with tf.GradientTape() as tape:
    #             logits = model(batch_x, training=True)

    #             loss = tfv2.nn.ctc_loss(labels=batch_y,
    #                                     label_length=batch_y_lens,
    #                                     logits=logits,
    #                                     logit_length=batch_x_lens,
    #                                     blank_index=Config.n_hidden_6 - 1,
    #                                     logits_time_major=False)

    #         grads = tape.gradient(loss, model.trainable_variables)
    #         optimizer.apply_gradients(zip(grads, model.trainable_variables))

    #         print('Epoch {:>3} - Step {:>3} - Training loss: {:.3f}'.format(epoch, int(step), float(loss)))


if __name__ == '__main__':
    create_flags()
    absl.app.run(main)
