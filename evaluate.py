#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import json
import sys

from multiprocessing import cpu_count

import absl.app
import progressbar
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

from ds_ctcdecoder import ctc_beam_search_decoder_batch, Scorer
from six.moves import zip

from util.config import Config, initialize_globals
from util.checkpoints import load_or_init_graph
from util.evaluate_tools import calculate_and_print_report
from util.feeding import create_dataset
from util.flags import create_flags, FLAGS
from util.helpers import check_ctcdecoder_version
from util.logging import create_progressbar, log_error, log_progress

check_ctcdecoder_version()

def sparse_tensor_value_to_texts(value, alphabet):
    r"""
    Given a :class:`tf.SparseTensor` ``value``, return an array of Python strings
    representing its values, converting tokens to strings using ``alphabet``.
    """
    return sparse_tuple_to_texts((value.indices, value.values, value.dense_shape), alphabet)


def sparse_tuple_to_texts(sp_tuple, alphabet):
    indices = sp_tuple[0]
    values = sp_tuple[1]
    results = [[] for _ in range(sp_tuple[2][0])]
    for i, index in enumerate(indices):
        results[index[0]].append(values[i])
    # List of strings
    return [alphabet.decode(res) for res in results]


def evaluate(test_csvs, create_model):
    if FLAGS.scorer_path and not FLAGS.use_greedy_decoder:
        scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta,
                        FLAGS.scorer_path, Config.alphabet)
    else:
        scorer = None

    test_csvs = FLAGS.test_files.split(',')
    test_sets = [create_dataset([csv], batch_size=FLAGS.test_batch_size, train_phase=False)[0] for csv in test_csvs]
    iterator = tfv1.data.Iterator.from_structure(tfv1.data.get_output_types(test_sets[0]),
                                                 tfv1.data.get_output_shapes(test_sets[0]),
                                                 output_classes=tfv1.data.get_output_classes(test_sets[0]))
    test_init_ops = [iterator.make_initializer(test_set) for test_set in test_sets]

    batch_wav_filename, (batch_x, batch_x_len), batch_y = iterator.get_next()

    with tf.variable_scope('model'):
        logits, encoded_lens = create_model(batch_x, batch_x_len, is_training=False)

    loss = tfv1.nn.ctc_loss(labels=batch_y,
                            inputs=logits,
                            sequence_length=encoded_lens,
                            time_major=False)

    if FLAGS.use_greedy_decoder:
        time_major = tf.transpose(logits, [1, 0, 2])
        [decoded_tensor, *_], _ = tf.nn.ctc_greedy_decoder(logits, encoded_lens)
    else:
        # Apply softmax for CTC decoder
        logits = tf.nn.softmax(logits)

    tfv1.train.get_or_create_global_step()

    # Get number of accessible CPU cores for this process
    try:
        num_processes = cpu_count()
    except NotImplementedError:
        num_processes = 1

    with tfv1.Session(config=Config.session_config) as session:
        if FLAGS.load == 'auto':
            method_order = ['best', 'last']
        else:
            method_order = [FLAGS.load]
        load_or_init_graph(session, method_order)

        def run_test(init_op, dataset):
            wav_filenames = []
            losses = []
            predictions = []
            ground_truths = []

            bar = create_progressbar(prefix='Test epoch | ',
                                     widgets=['Steps: ', progressbar.Counter(), ' | ', progressbar.Timer()]).start()
            log_progress('Test epoch...')

            step_count = 0

            # Initialize iterator to the appropriate dataset
            session.run(init_op)

            while True:
                if FLAGS.use_greedy_decoder:
                    try:
                        batch_wav_filenames, batch_decoded, batch_loss, batch_transcripts = \
                            session.run([batch_wav_filenames, decoded_tensor, loss, batch_y])
                    except tf.errors.OutOfRangeError:
                        break

                    decoded = sparse_tensor_value_to_texts(batch_decoded, Config.alphabet)
                else:
                    try:
                        batch_wav_filenames, batch_logits, batch_loss, batch_lengths, batch_transcripts = \
                            session.run([batch_wav_filename, logits, loss, encoded_lens, batch_y])
                    except tf.errors.OutOfRangeError:
                        break

                    decoded = ctc_beam_search_decoder_batch(batch_logits, batch_lengths, Config.alphabet, FLAGS.beam_width,
                                                            num_processes=num_processes, scorer=scorer,
                                                            cutoff_prob=FLAGS.cutoff_prob, cutoff_top_n=FLAGS.cutoff_top_n)
                    decoded = [d[0][1] for d in decoded]

                predictions.extend(decoded)
                ground_truths.extend(sparse_tensor_value_to_texts(batch_transcripts, Config.alphabet))
                wav_filenames.extend(wav_filename.decode('UTF-8') for wav_filename in batch_wav_filenames)
                losses.extend(batch_loss)

                step_count += 1
                bar.update(step_count)

            bar.finish()

            # Print test summary
            test_samples = calculate_and_print_report(wav_filenames, ground_truths, predictions, losses, dataset)
            return test_samples

        samples = []
        for csv, init_op in zip(test_csvs, test_init_ops):
            print('Testing model on {}'.format(csv))
            samples.extend(run_test(init_op, dataset=csv))
        return samples


def main(_):
    initialize_globals()

    if not FLAGS.test_files:
        log_error('You need to specify what files to use for evaluation via '
                  'the --test_files flag.')
        sys.exit(1)

    from DeepSpeech import create_model # pylint: disable=cyclic-import,import-outside-toplevel
    samples = evaluate(FLAGS.test_files.split(','), create_model)

    if FLAGS.test_output_file:
        # Save decoded tuples as JSON, converting NumPy floats to Python floats
        json.dump(samples, open(FLAGS.test_output_file, 'w'), default=float)


if __name__ == '__main__':
    create_flags()
    absl.app.run(main)
