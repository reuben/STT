# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os

from functools import partial

import numpy as np
import pandas
import tensorflow as tf

import datetime

from tensorflow.python.ops import gen_audio_ops as contrib_audio

from util.config import Config
from util.logging import log_error
from util.text import text_to_char_array


def read_csvs(csv_files):
    source_data = None
    for csv in csv_files:
        file = pandas.read_csv(csv, encoding='utf-8', na_filter=False)
        #FIXME: not cross-platform
        csv_dir = os.path.dirname(os.path.abspath(csv))
        file['wav_filename'] = file['wav_filename'].str.replace(r'(^[^/])', lambda m: os.path.join(csv_dir, m.group(1))) # pylint: disable=cell-var-from-loop
        if source_data is None:
            source_data = file
        else:
            source_data = source_data.append(file, ignore_index=True)
    return source_data


def samples_to_mfccs(samples, sample_rate):
    spectrogram = contrib_audio.audio_spectrogram(samples,
                                                  window_size=Config.audio_window_samples,
                                                  stride=Config.audio_step_samples,
                                                  magnitude_squared=True)
    mfccs = contrib_audio.mfcc(spectrogram, sample_rate, dct_coefficient_count=Config.n_input)
    mfccs = tf.reshape(mfccs, [-1, Config.n_input])

    return mfccs, tf.shape(input=mfccs)[0]


def audiofile_to_features(wav_filename):
    samples = tf.io.read_file(wav_filename)
    decoded = contrib_audio.decode_wav(samples, desired_channels=1)
    features, features_len = samples_to_mfccs(decoded.audio, decoded.sample_rate)

    return features, features_len


def entry_to_features(wav_filename, transcript, transcript_len):
    # https://bugs.python.org/issue32117
    features, features_len = audiofile_to_features(wav_filename)
    return features, features_len, transcript, transcript_len


def to_sparse_tuple(sequence):
    r"""Creates a sparse representention of ``sequence``.
        Returns a tuple with (indices, values, shape)
    """
    indices = np.asarray(list(zip([0]*len(sequence), range(len(sequence)))), dtype=np.int64)
    shape = np.asarray([1, len(sequence)], dtype=np.int64)
    return indices, sequence, shape


def create_dataset(csvs, batch_size, cache_path=''):
    df = read_csvs(csvs)
    df.sort_values(by='wav_filesize', inplace=True)

    try:
        # Convert to character index arrays
        df = df.apply(partial(text_to_char_array, alphabet=Config.alphabet), result_type='broadcast', axis=1)
    except ValueError as e:
        error_message, series, *_ = e.args
        log_error('While processing {}:\n  {}'.format(series['wav_filename'], error_message))
        exit(1)

    def generate_values():
        for _, row in df.iterrows():
            yield row.wav_filename, row.transcript, len(row.transcript)

    # Batching a dataset of 2D SparseTensors creates 3D batches, which fail
    # when passed to tf.nn.ctc_loss, so we reshape them to remove the extra
    # dimension here.
    def sparse_reshape(sparse):
        shape = sparse.dense_shape
        return tf.sparse.reshape(sparse, [shape[0], shape[2]])

    def batch_fn(features, features_len, transcripts, transcripts_len):
        features = features.padded_batch(batch_size,
                                         padded_shapes=(None, Config.n_input))
        features_len = features_len.batch(batch_size)
        transcripts = transcripts.padded_batch(batch_size,
                                               padded_shapes=(None,))
        transcripts_len = transcripts_len.batch(batch_size)
        return tf.data.Dataset.zip((features, features_len, transcripts, transcripts_len))

    num_gpus = len(Config.available_devices)

    dataset = (tf.data.Dataset.from_generator(generate_values,
                                              output_types=(tf.string, tf.int64, tf.int64),
                                              output_shapes=([], [None], []))
                              .map(entry_to_features, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                              .cache(cache_path)
                              .window(batch_size).flat_map(batch_fn)
                              .prefetch(num_gpus))

    return dataset

def secs_to_hours(secs):
    hours, remainder = divmod(secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%d:%02d:%02d' % (hours, minutes, seconds)
