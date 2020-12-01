#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import tensorflow.compat.v1 as tfv1
from tensorflow.core.protobuf import saved_model_pb2


def main():
    with tfv1.gfile.FastGFile(sys.argv[1], "rb") as fin:
        saved_model = saved_model_pb2.SavedModel()
        saved_model.ParseFromString(fin.read())

        graph_def = saved_model.meta_graphs[0].graph_def

        print("\n".join(sorted(set("{} - {}".format(n.name, n.op) for n in graph_def.node))))


if __name__ == "__main__":
    main()
