FROM debian:stretch as base

FROM base as tflite-base-build
RUN apt-get update && apt-get install -y --no-install-recommends make python3 python3-venv python3-dev build-essential git wget curl zip unzip && rm -rf /var/lib/apt/lists/*
RUN curl -L https://github.com/bazelbuild/bazelisk/releases/download/v1.7.5/bazelisk-linux-amd64 > /usr/local/bin/bazel && chmod +x /usr/local/bin/bazel
ENV VIRTUAL_ENV=/tmp/cp36-cp36m-venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY tensorflow /code/tensorflow/
WORKDIR /code/tensorflow/
RUN echo "" | TF_ENABLE_XLA=0 TF_NEED_JEMALLOC=1 TF_NEED_OPENCL_SYCL=0 TF_NEED_MKL=0 TF_NEED_VERBS=0 TF_NEED_MPI=0 TF_NEED_IGNITE=0 TF_NEED_GDR=0 TF_NEED_NGRAPH=0 TF_DOWNLOAD_CLANG=0 TF_SET_ANDROID_WORKSPACE=0 TF_NEED_TENSORRT=0 TF_NEED_ROCM=0 TF_NEED_CUDA=0 ./configure
RUN bazel build --verbose_failures --config=noaws --config=nogcp --config=nohdfs --config=nonccl --config=monolithic -c opt --copt=-O3 --copt="-D_GLIBCXX_USE_CXX11_ABI=0" --copt=-fvisibility=hidden //tensorflow/lite/c:libtensorflowlite_c.so

FROM tflite-base-build as tflite-lib-build
COPY native_client /code/native_client/
COPY .git /code/.git/
COPY training/deepspeech_training/VERSION training/deepspeech_training/GRAPH_VERSION /code/training/deepspeech_training/
WORKDIR /code/tensorflow/
RUN bazel build --verbose_failures --workspace_status_command="bash native_client/bazel_workspace_status_cmd.sh" --config=noaws --config=nogcp --config=nohdfs --config=nonccl --config=monolithic --define=runtime=tflite -c opt --copt=-O3 --copt="-D_GLIBCXX_USE_CXX11_ABI=0" --copt=-fvisibility=hidden //native_client:libdeepspeech.so

FROM quay.io/pypa/manylinux_2_24_x86_64:2021-03-10-aff9552 as pybase
RUN apt-get update && apt-get install -y --no-install-recommends wget && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /code/tensorflow/bazel-bin/native_client
COPY --from=tflite-lib-build /code/tensorflow/bazel-bin/native_client/libdeepspeech.so /code/tensorflow/bazel-bin/native_client/libdeepspeech.so
COPY . /code
WORKDIR /code/native_client/python

FROM pybase as py36
ENV VIRTUAL_ENV=/tmp/cp36-cp36m-venv
RUN /opt/python/cp36-cp36m/bin/python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install -U pip
RUN pip install numpy==1.7.0
ENV NUMPY_DEP_VERSION=">=1.7.0"
RUN make bindings TFDIR=/code/tensorflow

FROM scratch as py36-artifact
COPY --from=py36 /code/native_client/python/dist/deepspeech-*-cp36* /
