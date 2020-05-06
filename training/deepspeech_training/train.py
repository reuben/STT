#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

LOG_LEVEL_INDEX = sys.argv.index("--log_level") + 1 if "--log_level" in sys.argv else 0
os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    sys.argv[LOG_LEVEL_INDEX] if 0 < LOG_LEVEL_INDEX < len(sys.argv) else "3"
)

import absl.app
import numpy as np
import tensorflow as tf

from functools import partial
from six.moves import zip, range
from tensorflow.keras import layers
from tensorflow.python.keras.saving import saving_utils
from .util.config import Config, initialize_globals
from .util.feeding import create_dataset
from .util.flags import create_flags, FLAGS
from .util.logging import (
    log_info,
    log_error,
    log_debug,
    log_progress,
    create_progressbar,
)


class CreateOverlappingWindows(tf.keras.Model):
    def __init__(self):
        super(CreateOverlappingWindows, self).__init__()
        window_width = 2 * Config.n_context + 1
        num_channels = Config.n_input
        identity = np.eye(window_width * num_channels).reshape(
            window_width, num_channels, window_width * num_channels
        )
        self.identity_filter = tf.constant(identity, tf.float32)
        self.reshape = layers.Reshape((-1, window_width * num_channels))

    def call(self, x):
        x = tf.nn.conv1d(
            input=x, filters=self.identity_filter, stride=1, padding="SAME"
        )
        return self.reshape(x)


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.create_windows = CreateOverlappingWindows()
        self.mask = layers.Masking()

        def clipped_relu(x):
            return tf.keras.activations.relu(x, max_value=FLAGS.relu_clip)

        self.dense1 = layers.Dense(Config.n_hidden_1, activation=clipped_relu)
        self.dense2 = layers.Dense(Config.n_hidden_2, activation=clipped_relu)
        self.dense3 = layers.Dense(Config.n_hidden_3, activation=clipped_relu)
        self.lstm = layers.LSTM(Config.n_cell_dim, return_sequences=True)
        self.dense5 = layers.Dense(Config.n_hidden_5, activation=clipped_relu)
        self.dense6 = layers.Dense(Config.n_hidden_6)

    def call(self, batch_x, training=True, overlap=True):
        if overlap:
            x = self.create_windows(batch_x)
        else:
            x = self.create_windows.reshape(batch_x)

        x = self.mask(x)

        x = self.dense1(x)
        if training:
            x = tf.nn.dropout(x, FLAGS.dropout_rate)

        x = self.dense2(x)
        if training:
            x = tf.nn.dropout(x, FLAGS.dropout_rate2)

        x = self.dense3(x)
        if training:
            x = tf.nn.dropout(x, FLAGS.dropout_rate3)

        x = self.lstm(x)

        x = self.dense5(x)
        if training:
            x = tf.nn.dropout(x, FLAGS.dropout_rate5)

        x = self.dense6(x)
        return tf.identity(x, name='logits')

    def train_step(self, inputs):
        _, batch_x, batch_x_lens, batch_y, batch_y_lens = inputs

        with tf.GradientTape() as tape:
            logits = self(batch_x, training=True)
            loss = tf.nn.ctc_loss(
                labels=batch_y,
                logits=logits,
                label_length=batch_y_lens,
                logit_length=tf.squeeze(batch_x_lens, axis=-1),
                logits_time_major=False,
                blank_index=Config.n_hidden_6 - 1,
            )

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {"loss": loss}

    def test_step(self, inputs):
        _, batch_x, batch_x_lens, batch_y, batch_y_lens = inputs

        logits = self(batch_x, training=False)
        loss = tf.nn.ctc_loss(
            labels=batch_y,
            logits=logits,
            label_length=batch_y_lens,
            logit_length=batch_x_lens,
            logits_time_major=False,
            blank_index=Config.n_hidden_6 - 1,
        )

        return {"loss": loss}

    def predict_step(self, batch_x):
        return self(batch_x, training=False, overlap=False)



def main(_):
    initialize_globals()
    train_set = create_dataset(
        FLAGS.train_files.split(","),
        batch_size=FLAGS.train_batch_size,
        enable_cache=FLAGS.feature_cache and do_cache_dataset,
        cache_path=FLAGS.feature_cache,
        train_phase=True,
        process_ahead=len(Config.available_devices) * FLAGS.train_batch_size * 2,
        buffering=FLAGS.read_buffer,
    )

    validation_data = None

    if FLAGS.dev_files:
        dev_set = create_dataset(
                FLAGS.dev_files.split(","),
                batch_size=FLAGS.dev_batch_size,
                train_phase=False,
                process_ahead=len(Config.available_devices) * FLAGS.dev_batch_size * 2,
                buffering=FLAGS.read_buffer,
            )

        validation_data = dev_set

    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

    model = Model()
    model.compile(optimizer=optimizer)
    model.fit(
        train_set,
        epochs=FLAGS.epochs,
        batch_size=FLAGS.train_batch_size,
        validation_data=validation_data,
    )

    # Reset metrics before saving so that loaded model has same state,
    # since metric states are not preserved by Model.save_weights
    model.reset_metrics()

    model.save(
        os.path.join(FLAGS.save_checkpoint_dir, "train_weights"), save_format="tf"
    )

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, FLAGS.n_steps, 19, 26], dtype=tf.float32)])
    def inference_graph(input_node):
        return model.predict_step(tf.reshape(input_node, [1, FLAGS.n_steps * 19, 26]))


    sample_mfccs = tf.zeros([FLAGS.export_batch_size, FLAGS.n_steps, 19, Config.n_input])

    concrete_func = inference_graph.get_concrete_function(sample_mfccs)

    # func = saving_utils.trace_model_call(model)
    # concrete_func = func.get_concrete_function((sample_mfccs, sample_lengths))

    # frozen_func, frozen_graph = tf.python.framework.convert_to_constants.convert_variables_to_constants_v2_as_graph(concrete_func)
    # print("\n".join(sorted(set(n.name for n in frozen_graph.node))))

    converter = tf.lite.TFLiteConverter([concrete_func])
    tflite_model = converter.convert()

    output_filename = FLAGS.export_file_name + '.pb'
    output_tflite_path = os.path.join(FLAGS.export_dir, output_filename.replace('.pb', '.tflite'))

    with open(output_tflite_path, 'wb') as fout:
        fout.write(tflite_model)

    log_info('Models exported at %s' % (FLAGS.export_dir))

def run_script():
    create_flags()
    absl.app.run(main)


if __name__ == "__main__":
    run_script()
