FROM debian:stretch-20210311 as base

FROM base as linux-cpu-opt-base-build
RUN apt-get update && apt-get install -y --no-install-recommends make python3 python3-venv python3-dev build-essential git wget curl zip unzip && rm -rf /var/lib/apt/lists/*
RUN curl -L https://github.com/bazelbuild/bazelisk/releases/download/v1.7.5/bazelisk-linux-amd64 > /usr/local/bin/bazel && chmod +x /usr/local/bin/bazel
ENV VIRTUAL_ENV=/tmp/cp36-cp36m-venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY tensorflow /code/tensorflow/
WORKDIR /code/tensorflow/
RUN echo "" | TF_ENABLE_XLA=0 TF_NEED_JEMALLOC=1 TF_NEED_OPENCL_SYCL=0 TF_NEED_MKL=0 TF_NEED_VERBS=0 TF_NEED_MPI=0 TF_NEED_IGNITE=0 TF_NEED_GDR=0 TF_NEED_NGRAPH=0 TF_DOWNLOAD_CLANG=0 TF_SET_ANDROID_WORKSPACE=0 TF_NEED_TENSORRT=0 TF_NEED_ROCM=0 TF_NEED_CUDA=0 ./configure
RUN bazel --host_jvm_args=-Xmx1g build --local_ram_resources=1000 --jobs=1 --experimental_convenience_symlinks=clean --verbose_failures --config=noaws --config=nogcp --config=nohdfs --config=nonccl --config=monolithic -c opt --copt=-O3 --copt="-D_GLIBCXX_USE_CXX11_ABI=0" --copt=-fvisibility=hidden //tensorflow:tensorflow_cc

FROM linux-cpu-opt-base-build as linux-cpu-opt-lib-build
COPY native_client /code/native_client/
COPY .git /code/.git/
COPY training/coqui_stt_training/VERSION training/coqui_stt_training/GRAPH_VERSION /code/training/coqui_stt_training/
WORKDIR /code/tensorflow/
RUN bazel build --verbose_failures --workspace_status_command="bash native_client/bazel_workspace_status_cmd.sh" --config=noaws --config=nogcp --config=nohdfs --config=nonccl --config=monolithic --define=runtime=tflite -c opt --copt=-O3 --copt="-D_GLIBCXX_USE_CXX11_ABI=0" --copt=-fvisibility=hidden //native_client:libstt.so
# Remove convenience symlinks created by bazel so that the cache for tensorflow/ matches clean source dirs
RUN mv /code/tensorflow/bazel-bin/native_client/libstt.so /code/native_client/libstt.so
RUN bazel build --experimental_convenience_symlinks=clean

FROM scratch as linux-cpu-opt-lib-build-artifact
COPY --from=linux-cpu-opt-lib-build /code/native_client/libstt.so /

FROM quay.io/pypa/manylinux_2_24_x86_64:2021-03-10-aff9552 as linux-pybase-cpu-opt
RUN apt-get update && apt-get install -y --no-install-recommends wget && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /code/tensorflow/bazel-bin/native_client
COPY --from=linux-cpu-opt-lib-build-artifact /libstt.so /code/tensorflow/bazel-bin/native_client/libstt.so
COPY . /code
WORKDIR /code/native_client/python

FROM linux-pybase-cpu-opt as linux-py36-cpu-opt
ENV VIRTUAL_ENV=/tmp/cp36-cp36m-venv
RUN /opt/python/cp36-cp36m/bin/python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install -U pip
RUN pip install numpy==1.7.0
ENV NUMPY_DEP_VERSION=">=1.7.0"
RUN make bindings TFDIR=/code/tensorflow

FROM scratch as linux-py36-cpu-opt-artifact
COPY --from=linux-py36-cpu-opt /code/native_client/python/dist/*-cp36*.whl /

FROM linux-pybase-cpu-opt as linux-py37-cpu-opt
ENV VIRTUAL_ENV=/tmp/cp37-cp37m-venv
RUN /opt/python/cp37-cp37m/bin/python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install -U pip
RUN pip install numpy==1.14.5
ENV NUMPY_DEP_VERSION=">=1.14.5"
RUN make bindings TFDIR=/code/tensorflow

FROM scratch as linux-py37-cpu-opt-artifact
COPY --from=linux-py37-cpu-opt /code/native_client/python/dist/*-cp37*.whl /

FROM linux-pybase-cpu-opt as linux-py38-cpu-opt
ENV VIRTUAL_ENV=/tmp/cp38-cp38-venv
RUN /opt/python/cp38-cp38/bin/python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install -U pip
RUN pip install numpy==1.17.3
ENV NUMPY_DEP_VERSION=">=1.17.3"
RUN make bindings TFDIR=/code/tensorflow

FROM scratch as linux-py38-cpu-opt-artifact
COPY --from=linux-py38-cpu-opt /code/native_client/python/dist/*-cp38*.whl /

FROM linux-pybase-cpu-opt as linux-py39-cpu-opt
ENV VIRTUAL_ENV=/tmp/cp39-cp39-venv
RUN /opt/python/cp39-cp39/bin/python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install -U pip
RUN pip install numpy==1.19.4
ENV NUMPY_DEP_VERSION=">=1.19.4"
RUN make bindings TFDIR=/code/tensorflow

FROM scratch as linux-py39-cpu-opt-artifact
COPY --from=linux-py39-cpu-opt /code/native_client/python/dist/*-cp39*.whl /
