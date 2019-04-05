#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import itertools
import json

from multiprocessing import cpu_count

import numpy as np
import progressbar
import tensorflow as tf

from ds_ctcdecoder import ctc_beam_search_decoder_batch, Scorer
from six.moves import zip, range

from util.config import Config, initialize_globals
from util.evaluate_tools import calculate_report
from util.feeding import create_dataset
from util.flags import create_flags, FLAGS
from util.logging import log_error
from util.text import levenshtein


def sparse_tensor_value_to_texts(value, alphabet):
    r"""
    Given a :class:`tf.SparseTensor` ``value``, return an array of Python strings
    representing its values, converting tokens to strings using ``alphabet``.
    """
    return sparse_tuple_to_texts((value.indices, value.values, value.dense_shape), alphabet)


def sparse_tuple_to_texts(sp_tuple, alphabet):
    indices = sp_tuple[0]
    values = sp_tuple[1]
    results = [''] * sp_tuple[2][0]
    for i, index in enumerate(indices):
        results[index[0]] += alphabet.string_from_label(values[i])
    # List of strings
    return results


def evaluate(test_csvs, create_model):
    scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta,
                    FLAGS.lm_binary_path, FLAGS.lm_trie_path,
                    Config.alphabet)

    test_set, test_batches = create_dataset(test_csvs,
                                            batch_size=FLAGS.test_batch_size,
                                            cache_path=FLAGS.test_cached_features_path)
    it = test_set.make_one_shot_iterator()

    (batch_x, batch_x_len), batch_y = it.get_next()

    # One rate per layer
    no_dropout = [None] * 6
    logits, _ = create_model(batch_x=batch_x,
                             seq_length=batch_x_len,
                             dropout=no_dropout)

    # Transpose to batch major and apply softmax for decoder
    transposed = tf.nn.softmax(tf.transpose(logits, [1, 0, 2]))

    loss = tf.nn.ctc_loss(labels=batch_y,
                          inputs=logits,
                          sequence_length=batch_x_len)

    with tf.Session(config=Config.session_config) as session:
        # Create a saver using variables from the above newly created graph
        saver = tf.train.Saver()

        # Restore variables from training checkpoint
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if not checkpoint:
            log_error('Checkpoint directory ({}) does not contain a valid checkpoint state.'.format(FLAGS.checkpoint_dir))
            exit(1)

        checkpoint_path = checkpoint.model_checkpoint_path
        saver.restore(session, checkpoint_path)

        logitses = []
        losses = []
        seq_lengths = []
        ground_truths = []

        print('Computing acoustic model predictions...')
        bar = progressbar.ProgressBar(max_value=test_batches,
                                      widget=progressbar.AdaptiveETA)

        # First pass, compute losses and transposed logits for decoding
        for _ in bar(range(test_batches)):
            logits, loss_, lengths, transcripts = session.run([transposed, loss, batch_x_len, batch_y])

            logitses.append(logits)
            losses.extend(loss_)
            seq_lengths.append(lengths)
            ground_truths.extend(sparse_tensor_value_to_texts(transcripts, Config.alphabet))

    predictions = []

    # Get number of accessible CPU cores for this process
    try:
        num_processes = cpu_count()
    except NotImplementedError:
        num_processes = 1

    print('Decoding predictions...')
    bar = progressbar.ProgressBar(max_value=test_batches,
                                  widget=progressbar.AdaptiveETA)

    # Second pass, decode logits and compute WER and edit distance metrics
    for logits, seq_length in bar(zip(logitses, seq_lengths)):
        decoded = ctc_beam_search_decoder_batch(logits, seq_length, Config.alphabet, FLAGS.beam_width,
                                                num_processes=num_processes, scorer=scorer)
        predictions.extend(d[0][1] for d in decoded)

    distances = [levenshtein(a, b) for a, b in zip(ground_truths, predictions)]

    wer, cer, samples = calculate_report(ground_truths, predictions, distances, losses)
    mean_loss = np.mean(losses)

    # Take only the first report_count items
    report_samples = itertools.islice(samples, FLAGS.report_count)

    print('Test - WER: %f, CER: %f, loss: %f' %
          (wer, cer, mean_loss))
    print('-' * 80)
    for sample in report_samples:
        print('WER: %f, CER: %f, loss: %f' %
              (sample.wer, sample.distance, sample.loss))
        print(' - src: "%s"' % sample.src)
        print(' - res: "%s"' % sample.res)
        print('-' * 80)

    return samples


def main(_):
    initialize_globals()

    if not FLAGS.test_files:
        log_error('You need to specify what files to use for evaluation via '
                  'the --test_files flag.')
        exit(1)

    from DeepSpeech import create_model # pylint: disable=cyclic-import
    samples = evaluate(FLAGS.test_files.split(','), create_model)

    if FLAGS.test_output_file:
        # Save decoded tuples as JSON, converting NumPy floats to Python floats
        json.dump(samples, open(FLAGS.test_output_file, 'w'), default=float)


if __name__ == '__main__':
    create_flags()
    tf.app.flags.DEFINE_string('hdf5_test_set', '', 'path to hdf5 file to cache test set features')
    tf.app.flags.DEFINE_string('test_output_file', '', 'path to a file to save all src/decoded/distance/loss tuples')
    tf.app.run(main)
