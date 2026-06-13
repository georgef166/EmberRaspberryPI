#!/usr/bin/env bash
# Build a Coral-compatible TensorFlow Lite C++ shared library.
#
# libedgetpu 16.0 is from the TensorFlow Lite 2.5 era. Newer TFLite shared
# libraries can compile, but the Edge TPU delegate ABI is not compatible and can
# segfault at startup. This script builds the local TFLite dependency used by
# src/cpp/CMakeLists.txt:
#
#   .ember-deps/tensorflow-tflite-coral
#   .ember-deps/tflite-coral-build/libtensorflow-lite.so
#
# The .ember-deps directory is intentionally ignored by git.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DEPS_DIR="$HERE/.ember-deps"
TF_SRC="$DEPS_DIR/tensorflow-tflite-coral"
TFLITE_BUILD="$DEPS_DIR/tflite-coral-build"
TF_VERSION="${TF_VERSION:-v2.5.0}"
JOBS="${JOBS:-3}"

mkdir -p "$DEPS_DIR"

if [ ! -d "$TF_SRC/.git" ]; then
    echo "== Cloning TensorFlow $TF_VERSION for Coral-compatible TFLite"
    git clone --depth 1 --branch "$TF_VERSION" \
        https://github.com/tensorflow/tensorflow.git "$TF_SRC"
else
    echo "== TensorFlow source already present: $TF_SRC"
fi

echo "== Configuring TensorFlow Lite"
cmake -S "$TF_SRC/tensorflow/lite" \
    -B "$TFLITE_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DTFLITE_ENABLE_XNNPACK=OFF

# Older Abseil fetched by TensorFlow 2.5 misses <limits> for GCC 12.
GRAPH_CYCLES="$TFLITE_BUILD/abseil-cpp/absl/synchronization/internal/graphcycles.cc"
if [ -f "$GRAPH_CYCLES" ] && ! grep -q '#include <limits>' "$GRAPH_CYCLES"; then
    echo "== Patching Abseil graphcycles.cc for GCC 12"
    tmp_file="$(mktemp)"
    awk '{ print; if ($0 == "#include <array>") print "#include <limits>" }' \
        "$GRAPH_CYCLES" > "$tmp_file"
    mv "$tmp_file" "$GRAPH_CYCLES"
fi

echo "== Building TensorFlow Lite"
cmake --build "$TFLITE_BUILD" --target tensorflow-lite -j "$JOBS"

echo "== TensorFlow Lite ready:"
echo "   $TFLITE_BUILD/libtensorflow-lite.so"
