#!/usr/bin/env bash
# =============================================================================
# Ember — one-shot setup / build / run for Raspberry Pi.
#
# First time on a fresh Pi (clone once, then everything goes through this script):
#   cd ~/Desktop
#   git clone https://github.com/georgef166/EmberRaspberryPI.git Ember
#   cd Ember
#   ./run.sh --setup            # install deps + optional Coral/Thermal, then build & run
#
# Everyday use:
#   ./run.sh                    # build if needed, then launch (default CPU detector)
#   ./run.sh --coral            # launch using the Coral Edge TPU + SSD MobileNet v2
#   ./run.sh --thermal          # launch with thermal-primary victim detection
#   ./run.sh --update           # git pull, rebuild, then launch
#   ./run.sh --rebuild          # force a clean rebuild, then launch
#   ./run.sh -- --range 6000    # pass extra flags straight through to tactical_rescue
#
# Flags can combine, e.g.:  ./run.sh --setup --coral
# =============================================================================
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

DO_SETUP=0
DO_UPDATE=0
DO_REBUILD=0
BACKEND="default"   # default | coral | thermal
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --setup)   DO_SETUP=1 ;;
        --update)  DO_UPDATE=1 ;;
        --rebuild) DO_REBUILD=1 ;;
        --coral)   BACKEND="coral" ;;
        --thermal) BACKEND="thermal" ;;
        --)        shift; while [ $# -gt 0 ]; do EXTRA_ARGS+=("$1"); shift; done; break ;;
        -h|--help)
            cat <<'USAGE'
Ember — one-shot setup / build / run for Raspberry Pi.

  ./run.sh                build if needed, then launch (default CPU detector)
  ./run.sh --setup        install deps + optional Coral/Thermal, then build & run
  ./run.sh --coral        launch using the Coral Edge TPU + SSD MobileNet v2
  ./run.sh --thermal      launch with thermal-primary victim detection
  ./run.sh --update       git pull, rebuild, then launch
  ./run.sh --rebuild      force a clean rebuild, then launch
  ./run.sh -- ARGS        pass extra flags straight through to tactical_rescue

Flags combine, e.g.:  ./run.sh --setup --coral
USAGE
            exit 0 ;;
        *) EXTRA_ARGS+=("$1") ;;
    esac
    shift
done

# --- 1. Optional first-time setup (idempotent; the installers re-run safely) ---
if [ "$DO_SETUP" = 1 ]; then
    echo "== [1/3] Installing dependencies (ToF SDK, OpenCV, TensorFlow Lite)"
    ./Install_dependencies.sh || true     # may prompt to reboot; that's fine
    sudo apt-get install -y libtensorflow-lite-dev

    echo "== Installing optional Coral Edge TPU support (skip errors if no Coral)"
    ./Install_coral.sh || echo "   (Coral setup skipped/failed — continuing)"

    echo "== Installing optional MLX90640 thermal support (skip errors if no sensor)"
    ./Install_thermal.sh || echo "   (Thermal setup skipped/failed — continuing)"

    echo ""
    echo ">> If config.txt was modified above (camera overlay / 1MHz I2C), REBOOT now,"
    echo ">> then re-run:  ./run.sh ${BACKEND/default/}"
    echo ""
fi

# --- 2. Update + build ---
if [ "$DO_UPDATE" = 1 ]; then
    echo "== Pulling latest from origin/main"
    git pull --ff-only origin main
    DO_REBUILD=1
fi

if [ "$DO_REBUILD" = 1 ]; then
    rm -rf "$HERE/build"
fi

if [ ! -x "$HERE/build/src/cpp/tactical_rescue" ]; then
    echo "== [2/3] Building tactical_rescue"
    ./compile.sh
else
    echo "== [2/3] Binary present — skipping build (use --rebuild to force)"
fi

if [ ! -x "$HERE/build/src/cpp/tactical_rescue" ]; then
    echo "!! Build did not produce build/src/cpp/tactical_rescue — see errors above." >&2
    exit 1
fi

# --- 3. Launch ---
RUN=("$HERE/build/src/cpp/tactical_rescue")
case "$BACKEND" in
    coral)   RUN+=(--edgetpu --tflite-model "$HERE/models/ssd_mobilenet_v2_coco_edgetpu.tflite" --person-class 0) ;;
    thermal) RUN+=(--detector-source thermal) ;;
esac
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    RUN+=("${EXTRA_ARGS[@]}")
fi

echo "== [3/3] Launching: ${RUN[*]}"
echo "   (press q or Esc in the window to quit)"
exec env QT_QPA_PLATFORM=xcb "${RUN[@]}"
