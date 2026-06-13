#!/bin/sh
# =============================================================================
# Ember — Google Coral Edge TPU setup
#
# Installs the Edge TPU runtime + C++ dev headers (libedgetpu) and downloads the
# Edge TPU-compiled SSD MobileNet v2 COCO person-detection model.
#
# Run this AFTER Install_dependencies.sh, then rebuild with ./compile.sh.
# The CMake build auto-detects libedgetpu and enables the --edgetpu backend.
#
# Hardware: Coral USB Accelerator or Coral M.2/PCIe on a Raspberry Pi 4B.
# =============================================================================
set -e

MODELS_DIR="$(cd "$(dirname "$0")" && pwd)/models"
mkdir -p "$MODELS_DIR"

echo "== Adding Coral Edge TPU apt repository"
if [ ! -f /etc/apt/sources.list.d/coral-edgetpu.list ]; then
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" |
        sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg |
        sudo gpg --dearmor -o /usr/share/keyrings/coral-edgetpu.gpg
    sudo sed -i 's|^deb |deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] |' \
        /etc/apt/sources.list.d/coral-edgetpu.list
fi

sudo apt-get update

# libedgetpu1-std = standard clock (cooler/quieter). Swap for libedgetpu1-max
# for maximum throughput if you have active cooling on the Coral.
# libedgetpu-dev provides edgetpu.h needed for the C++ build.
echo "== Installing libedgetpu runtime + dev headers"
sudo apt-get install -y libedgetpu1-std libedgetpu-dev

# --- Edge TPU model: SSD MobileNet v2 COCO (300x300, quantized) ---
# This model's labelmap is 90-class 0-indexed (person = 0), so run with
# --person-class 0 (see README).
MODEL_URL="https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
LABELS_URL="https://github.com/google-coral/test_data/raw/master/coco_labels.txt"

echo "== Downloading Edge TPU model"
curl -fSL "$MODEL_URL" -o "$MODELS_DIR/ssd_mobilenet_v2_coco_edgetpu.tflite"
curl -fSL "$LABELS_URL" -o "$MODELS_DIR/coco_labels.txt"

echo "== Building Coral-compatible TensorFlow Lite C++ runtime"
"$(cd "$(dirname "$0")" && pwd)/Build_tflite_coral.sh"

echo ""
echo "== Coral setup complete."
echo "   Rebuild:  ./compile.sh   (look for 'Coral Edge TPU support: ENABLED')"
echo "   Run:      QT_QPA_PLATFORM=xcb ./build/src/cpp/tactical_rescue \\"
echo "               --edgetpu \\"
echo "               --tflite-model $MODELS_DIR/ssd_mobilenet_v2_coco_edgetpu.tflite \\"
echo "               --person-class 0"
