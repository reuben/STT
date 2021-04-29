#!/bin/bash

set -xe

force_java_apk_x86_64()
{
  cd ${DS_DSDIR}/native_client/java/
  cat <<EOF > libdeepspeech/gradle.properties
ABI_FILTERS = x86_64
EOF
}

do_deepspeech_java_apk_build()
{
  cd ${DS_DSDIR}

  export ANDROID_HOME=${ANDROID_SDK_HOME}

  make GRADLE="./gradlew " -C native_client/java/
  make GRADLE="./gradlew " -C native_client/java/ maven-bundle
}

android_run_tests()
{
  cd ${DS_DSDIR}/native_client/java/

  adb shell service list

  adb shell ls -hal /data/local/tmp/test/

  ./gradlew --console=plain libdeepspeech:connectedAndroidTest
}

android_sdk_accept_licenses()
{
  pushd "${ANDROID_SDK_HOME}"
    yes | ./tools/bin/sdkmanager --licenses
  popd
}

android_setup_emulator()
{
  if [ -z "${ANDROID_SDK_HOME}" ]; then
    echo "No Android SDK home available, aborting."
    exit 1
  fi;

  if [ -z "$1" ]; then
    echo "No ARM flavor, please give one."
    exit 1
  fi;

  local _flavor=$1
  local _api_level=${2:-android-25}
  local _api_kind=${3:-google_apis}

  if [ -z "${_api_kind}" ]; then
    _api_kind="google_apis"
  fi

  export PATH=${ANDROID_SDK_HOME}/tools/bin/:${ANDROID_SDK_HOME}/platform-tools/:$PATH
  export DS_BINARY_PREFIX="adb shell TMP=${ANDROID_TMP_DIR}/ LD_LIBRARY_PATH=${ANDROID_TMP_DIR}/ds/ ${ANDROID_TMP_DIR}/ds/"

  # Pipe yes in case of license being shown
  yes | sdkmanager --update
  yes | sdkmanager --install "emulator"

  android_install_sdk_platform "${_api_level}"

  # Same, yes in case of license
  yes | sdkmanager --install "system-images;${_api_level};${_api_kind};${_flavor}"

  android_sdk_accept_licenses

  avdmanager create avd --name "${_flavor}-ds-pixel-${_api_level}" --device 17 --package "system-images;${_api_level};${_api_kind};${_flavor}"
}

android_start_emulator()
{
  local _flavor=$1
  local _api_level=${2:-android-25}

  export PATH=${ANDROID_SDK_HOME}/tools/bin/:${ANDROID_SDK_HOME}/platform-tools/:$PATH
  export DS_BINARY_PREFIX="adb shell TMP=${ANDROID_TMP_DIR}/ LD_LIBRARY_PATH=${ANDROID_TMP_DIR}/ds/ ${ANDROID_TMP_DIR}/ds/"

  # minutes (2 minutes by default)
  export ADB_INSTALL_TIMEOUT=8

  # Use xvfb because:
  #  > emulator: INFO: QtLogger.cpp:68: Warning: could not connect to display  ((null):0, (null))
  # -accel on is needed otherwise it is too slow, but it will require KVM support exposed
  pushd ${ANDROID_SDK_HOME}
    xvfb-run ./tools/emulator -verbose -avd "${_flavor}-ds-pixel-${_api_level}" -no-skin -no-audio -no-window -no-boot-anim -accel on &
    emulator_rc=$?
    export ANDROID_DEVICE_EMULATOR=$!
  popd

  if [ "${emulator_rc}" -ne 0 ]; then
    echo "Error starting Android emulator, aborting."
    exit 1
  fi;

  adb wait-for-device

  adb shell id
  adb shell cat /proc/cpuinfo

  adb shell service list
}

android_install_sdk_platform()
{
  local _api_level=${1:-android-27}

  if [ -z "${ANDROID_SDK_HOME}" ]; then
    echo "No Android SDK home available, aborting."
    exit 1
  fi;

  export PATH=${ANDROID_SDK_HOME}/tools/bin/:${ANDROID_SDK_HOME}/platform-tools/:$PATH

  # Pipe yes in case of license being shown
  yes | sdkmanager --update
  yes | sdkmanager --install "build-tools;28.0.3"
  yes | sdkmanager --install "cmake;3.6.4111459"
  yes | sdkmanager --install "platform-tools"
  yes | sdkmanager --install "platforms;${_api_level}"

  android_sdk_accept_licenses
}

android_wait_for_emulator()
{
  while [ "${boot_completed}" != "1" ]; do
    sleep 15
    boot_completed=$(adb shell getprop sys.boot_completed | tr -d '\r')
  done
}

android_setup_ndk_data()
{
  adb shell mkdir ${ANDROID_TMP_DIR}/ds/
  adb push ${TASKCLUSTER_TMP_DIR}/ds/* ${ANDROID_TMP_DIR}/ds/

  adb push \
    ${TASKCLUSTER_TMP_DIR}/${model_name} \
    ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} \
    ${ANDROID_TMP_DIR}/ds/

  if [ -f "${TASKCLUSTER_TMP_DIR}/kenlm.scorer" ]; then
    adb push ${TASKCLUSTER_TMP_DIR}/kenlm.scorer ${ANDROID_TMP_DIR}/ds/
  fi
}

android_setup_apk_data()
{
  adb shell mkdir ${ANDROID_TMP_DIR}/test/

  adb push \
    ${TASKCLUSTER_TMP_DIR}/${model_name} \
    ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} \
    ${TASKCLUSTER_TMP_DIR}/kenlm.scorer \
    ${ANDROID_TMP_DIR}/test/
}

android_stop_emulator()
{
  if [ -z "${ANDROID_DEVICE_EMULATOR}" ]; then
    echo "No ANDROID_DEVICE_EMULATOR"
    exit 1
  fi;

  # Gracefully stop
  adb shell reboot -p &

  # Just in case, let it 30 seconds before force-killing
  sleep 30
  kill -9 ${ANDROID_DEVICE_EMULATOR} || true
}
