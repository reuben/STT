FROM debian:stretch-20210311 as base

FROM base as linux-tflite-base-build
RUN apt-get update && apt-get install -y --no-install-recommends make python3 python3-venv python3-dev build-essential git wget curl zip unzip && rm -rf /var/lib/apt/lists/*
RUN curl -L https://github.com/bazelbuild/bazelisk/releases/download/v1.7.5/bazelisk-linux-amd64 > /usr/local/bin/bazel && chmod +x /usr/local/bin/bazel
ENV VIRTUAL_ENV=/tmp/cp36-cp36m-venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY tensorflow /code/tensorflow/
WORKDIR /code/tensorflow/
RUN echo "" | TF_ENABLE_XLA=0 TF_NEED_JEMALLOC=1 TF_NEED_OPENCL_SYCL=0 TF_NEED_MKL=0 TF_NEED_VERBS=0 TF_NEED_MPI=0 TF_NEED_IGNITE=0 TF_NEED_GDR=0 TF_NEED_NGRAPH=0 TF_DOWNLOAD_CLANG=0 TF_SET_ANDROID_WORKSPACE=0 TF_NEED_TENSORRT=0 TF_NEED_ROCM=0 TF_NEED_CUDA=0 ./configure
RUN bazel build --experimental_convenience_symlinks=clean --verbose_failures --config=noaws --config=nogcp --config=nohdfs --config=nonccl --config=monolithic -c opt --copt=-O3 --copt="-D_GLIBCXX_USE_CXX11_ABI=0" --copt=-fvisibility=hidden //tensorflow/lite/c:libtensorflowlite_c.so

FROM linux-tflite-base-build as linux-tflite-lib-build
COPY native_client /code/native_client/
COPY .git /code/.git/
COPY training/deepspeech_training/VERSION training/deepspeech_training/GRAPH_VERSION /code/training/deepspeech_training/
WORKDIR /code/tensorflow/
RUN bazel build --verbose_failures --workspace_status_command="bash native_client/bazel_workspace_status_cmd.sh" --config=noaws --config=nogcp --config=nohdfs --config=nonccl --config=monolithic --define=runtime=tflite -c opt --copt=-O3 --copt="-D_GLIBCXX_USE_CXX11_ABI=0" --copt=-fvisibility=hidden //native_client:libdeepspeech.so
# Remove convenience symlinks created by bazel so that the cache for tensorflow/ matches clean source dirs
RUN mv /code/tensorflow/bazel-bin/native_client/libdeepspeech.so /code/native_client/libdeepspeech.so
RUN bazel build --experimental_convenience_symlinks=clean

FROM scratch as linux-tflite-lib-build-artifact
COPY --from=linux-tflite-lib-build /code/native_client/libdeepspeech.so /

FROM quay.io/pypa/manylinux_2_24_x86_64:2021-03-10-aff9552 as linux-pybase-tflite
RUN apt-get update && apt-get install -y --no-install-recommends wget && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /code/tensorflow/bazel-bin/native_client
COPY --from=linux-tflite-lib-build-artifact /libdeepspeech.so /code/tensorflow/bazel-bin/native_client/libdeepspeech.so
COPY . /code
WORKDIR /code/native_client/python

FROM linux-pybase-tflite as linux-py36-tflite
ENV VIRTUAL_ENV=/tmp/cp36-cp36m-venv
RUN /opt/python/cp36-cp36m/bin/python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install -U pip
RUN pip install numpy==1.7.0
ENV NUMPY_DEP_VERSION=">=1.7.0"
RUN make bindings TFDIR=/code/tensorflow SETUP_FLAGS="--project_name deepspeech-tflite"

FROM scratch as linux-py36-tflite-artifact
COPY --from=linux-py36-tflite /code/native_client/python/dist/deepspeech-*-cp36* /

FROM linux-pybase-tflite as linux-py37-tflite
ENV VIRTUAL_ENV=/tmp/cp37-cp37m-venv
RUN /opt/python/cp37-cp37m/bin/python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install -U pip
RUN pip install numpy==1.14.5
ENV NUMPY_DEP_VERSION=">=1.14.5"
RUN make bindings TFDIR=/code/tensorflow SETUP_FLAGS="--project_name deepspeech-tflite"

FROM scratch as linux-py37-tflite-artifact
COPY --from=linux-py37-tflite /code/native_client/python/dist/deepspeech-*-cp37* /

FROM linux-pybase-tflite as linux-py38-tflite
ENV VIRTUAL_ENV=/tmp/cp38-cp38-venv
RUN /opt/python/cp38-cp38/bin/python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install -U pip
RUN pip install numpy==1.17.3
ENV NUMPY_DEP_VERSION=">=1.17.3"
RUN make bindings TFDIR=/code/tensorflow SETUP_FLAGS="--project_name deepspeech-tflite"

FROM scratch as linux-py38-tflite-artifact
COPY --from=linux-py38-tflite /code/native_client/python/dist/deepspeech-*-cp38* /

FROM linux-pybase-tflite as linux-py39-tflite
ENV VIRTUAL_ENV=/tmp/cp39-cp39-venv
RUN /opt/python/cp39-cp39/bin/python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install -U pip
RUN pip install numpy==1.19.4
ENV NUMPY_DEP_VERSION=">=1.19.4"
RUN make bindings TFDIR=/code/tensorflow SETUP_FLAGS="--project_name deepspeech-tflite"

FROM scratch as linux-py39-tflite-artifact
COPY --from=linux-py39-tflite /code/native_client/python/dist/deepspeech-*-cp39* /
