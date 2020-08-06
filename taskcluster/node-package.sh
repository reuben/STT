#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true

# NodeJS package
ls -lh ${DS_ROOT_TASK}/DeepSpeech/ds/native_client/javascript/
cp ${DS_ROOT_TASK}/DeepSpeech/ds/native_client/javascript/mozilla-voice-stt*.tgz ${TASKCLUSTER_ARTIFACTS}/
