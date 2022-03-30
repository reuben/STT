#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import sys
import io
from functools import partial
from multiprocessing import JoinableQueue, Manager, Process, cpu_count

import numpy as np
import onnxruntime
import soundfile as sf
from clearml import Task
from coqui_stt_training.util.evaluate_tools import calculate_and_print_report
from coqui_stt_training.util.multiprocessing import PoolBase
from coqui_stt_ctcdecoder import (
    Alphabet,
    Scorer,
    ctc_beam_search_decoder_for_wav2vec2am,
)
from tqdm import tqdm


class EvaluationPool(PoolBase):
    def init(self, model_file, scorer_path):
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.session = onnxruntime.InferenceSession(model_file, sess_options)

        self.am_alphabet = Alphabet()
        self.am_alphabet.InitFromLabels("ABCD etaonihsrdlumwcfgypbvk'xjqz")

        self.scorer = None
        if scorer_path:
            self.scorer_alphabet = Alphabet()
            self.scorer_alphabet.InitFromLabels(" abcdefghijklmnopqrstuvwxyz'")

            self.scorer = Scorer()
            self.scorer.init_from_filepath(
                scorer_path.encode("utf-8"), self.scorer_alphabet
            )

    def run(self, job):
        wav_filename, ground_truth, beam_width, lm_alpha, lm_beta = job
        prediction = self.transcribe_file(wav_filename, beam_width, lm_alpha, lm_beta)
        return wav_filename, ground_truth, prediction

    def transcribe_file(self, wav_filename, beam_width, lm_alpha, lm_beta):
        if lm_alpha and lm_beta:
            self.scorer.reset_params(lm_alpha, lm_beta)

        speech_array, sr = sf.read(wav_filename)
        max_length = 250000
        speech_array = speech_array.astype(np.float32)
        features = speech_array[:max_length]

        def norm(wav, db_level=-27):
            r = 10 ** (db_level / 20)
            a = np.sqrt((len(wav) * (r**2)) / np.sum(wav**2))
            return wav * a

        features = norm(features)

        onnx_outputs = self.session.run(
            None, {self.session.get_inputs()[0].name: [features]}
        )[0].squeeze()

        decoded = ctc_beam_search_decoder_for_wav2vec2am(
            onnx_outputs,
            self.am_alphabet,
            beam_size=beam_width,
            scorer=self.scorer,
            blank_id=0,
            ignored_symbols=[1, 2, 3],
        )[0].transcript.strip()

        return decoded


def evaluate_wav2vec2am(
    model_file,
    csv_file,
    scorer,
    num_processes,
    dump_to_file,
    beam_width,
    lm_alpha=None,
    lm_beta=None,
    existing_pool=None,
):
    jobs = []
    with open(csv_file, "r") as csvfile:
        csvreader = csv.DictReader(csvfile)
        count = 0
        for row in csvreader:
            count += 1
            # Relative paths are relative to the folder the CSV file is in
            if not os.path.isabs(row["wav_filename"]):
                row["wav_filename"] = os.path.join(
                    os.path.dirname(csv_file), row["wav_filename"]
                )
            jobs.append(
                (row["wav_filename"], row["transcript"], beam_width, lm_alpha, lm_beta)
            )

    pool = existing_pool
    if not pool:
        pool = EvaluationPool.create_impl(
            processes=num_processes, initargs=(model_file, scorer)
        )

    process_iterable = tqdm(
        pool.imap_unordered(jobs),
        desc="Transcribing files",
        total=len(jobs),
    )

    wav_filenames = []
    ground_truths = []
    predictions = []
    losses = []

    for wav_filename, ground_truth, prediction in process_iterable:
        wav_filenames.append(wav_filename)
        ground_truths.append(ground_truth)
        predictions.append(prediction)
        losses.append(0.0)

    # Print test summary
    samples = calculate_and_print_report(
        wav_filenames, ground_truths, predictions, losses, csv_file
    )

    if dump_to_file:
        with open(dump_to_file + ".txt", "w") as ftxt, open(
            dump_to_file + ".out", "w"
        ) as fout:
            for wav, txt, out in zip(wav_filenames, ground_truths, predictions):
                ftxt.write("%s %s\n" % (wav, txt))
                fout.write("%s %s\n" % (wav, out))
            print("Reference texts dumped to %s.txt" % dump_to_file)
            print("Transcription   dumped to %s.out" % dump_to_file)

    if not existing_pool:
        pool.close()

    return samples


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluation report using Wav2vec2 ONNX AM"
    )
    parser.add_argument(
        "--model", required=True, help="Path to the model (ONNX export)"
    )
    parser.add_argument("--csv", required=True, help="Path to the CSV source file")
    parser.add_argument(
        "--scorer",
        required=False,
        default=None,
        help="Path to the external scorer file",
    )
    parser.add_argument(
        "--proc",
        required=False,
        default=cpu_count(),
        type=int,
        help="Number of processes to spawn, defaulting to number of CPUs",
    )
    parser.add_argument(
        "--dump",
        required=False,
        help='Path to dump the results as text file, with one line for each wav: "wav transcription".',
    )
    parser.add_argument(
        "--beam_width",
        required=False,
        default=8,
        type=int,
        help="Beam width to use when decoding.",
    )
    parser.add_argument(
        "--clearml_project",
        required=False,
        default="STT/wav2vec2 decoding",
    )
    parser.add_argument(
        "--clearml_task",
        required=False,
        default="evaluation report",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    try:
        task = Task.init(project_name=args.clearml_project, task_name=args.clearml_task)
    except:
        pass
    evaluate_wav2vec2am(
        args.model, args.csv, args.scorer, args.proc, args.dump, args.beam_width
    )


if __name__ == "__main__":
    main()
