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
#   ./run.sh                    # build if needed, then launch (auto-uses Coral when available)
#   ./run.sh --coral            # launch using the Coral Edge TPU + SSD MobileNet v2
#   ./run.sh --cpu              # force the CPU/default detector path
#   ./run.sh --thermal          # launch with thermal-primary victim detection
#   ./run.sh --stream           # also serve the HUD at http://<pi-ip>:8080/
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
BACKEND="auto"      # auto | cpu | coral | thermal
ENABLE_STREAM=0
STREAM_ARGS=()
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --setup)   DO_SETUP=1 ;;
        --update)  DO_UPDATE=1 ;;
        --rebuild) DO_REBUILD=1 ;;
        --coral)   BACKEND="coral" ;;
        --cpu)     BACKEND="cpu" ;;
        --thermal) BACKEND="thermal" ;;
        --stream)  ENABLE_STREAM=1 ;;
        --stream-bind|--stream-port|--stream-fps|--stream-quality)
            ENABLE_STREAM=1
            opt_name="$1"
            STREAM_ARGS+=("$opt_name")
            shift
            if [ $# -eq 0 ]; then
                echo "Missing value for $opt_name" >&2
                exit 2
            fi
            STREAM_ARGS+=("$1")
            ;;
        --)        shift; while [ $# -gt 0 ]; do EXTRA_ARGS+=("$1"); shift; done; break ;;
        -h|--help)
            cat <<'USAGE'
Ember — one-shot setup / build / run for Raspberry Pi.

  ./run.sh                build if needed, then launch (auto-uses Coral when available)
  ./run.sh --setup        install deps + optional Coral/Thermal, then build & run
  ./run.sh --coral        launch using the Coral Edge TPU + SSD MobileNet v2
  ./run.sh --cpu          force the CPU/default detector path
  ./run.sh --thermal      launch with thermal-primary victim detection
  ./run.sh --stream       also serve the HUD at http://<pi-ip>:8080/
  ./run.sh --stream-port 9090
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
    echo "== [1/3] Installing dependencies (ToF SDK, OpenCV)"
    ./Install_dependencies.sh || true     # may prompt to reboot; that's fine
    # NOTE: TensorFlow Lite has no apt package on Raspberry Pi OS — it is optional
    # and built from source (see README). The ToF + thermal + CV detectors build
    # without it, so we do not block setup on it here.

    echo "== Installing optional Coral Edge TPU support (skip errors if no Coral)"
    ./Install_coral.sh || echo "   (Coral setup skipped/failed — continuing)"

    echo "== Installing optional MLX90640 thermal support (skip errors if no sensor)"
    ./Install_thermal.sh || echo "   (Thermal setup skipped/failed — continuing)"
    DO_REBUILD=1

    echo ""
    echo ">> If config.txt was modified above (camera overlay / 1MHz I2C), REBOOT now,"
    RERUN_FLAG=""
    case "$BACKEND" in
        coral) RERUN_FLAG=" --coral" ;;
        cpu) RERUN_FLAG=" --cpu" ;;
        thermal) RERUN_FLAG=" --thermal" ;;
    esac
    if [ "$ENABLE_STREAM" = 1 ]; then
        RERUN_FLAG="$RERUN_FLAG --stream"
    fi
    echo ">> then re-run:  ./run.sh$RERUN_FLAG"
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
has_coral_runtime() {
    if ldconfig -p 2>/dev/null | grep -q 'libedgetpu'; then
        return 0
    fi
    [ -e /usr/lib/libedgetpu.so ] && return 0
    [ -e /usr/lib/aarch64-linux-gnu/libedgetpu.so ] && return 0
    [ -e /usr/local/lib/libedgetpu.so ] && return 0
    return 1
}

if [ "$BACKEND" = "auto" ] &&
    [ -f "$HERE/models/ssd_mobilenet_v2_coco_edgetpu.tflite" ] &&
    has_coral_runtime; then
    BACKEND="coral"
fi

case "$BACKEND" in
    coral)   RUN+=(--coral) ;;
    thermal) RUN+=(--detector-source thermal) ;;
esac
if [ "$ENABLE_STREAM" = 1 ]; then
    RUN+=(--stream)
    if [ "${#STREAM_ARGS[@]}" -gt 0 ]; then
        RUN+=("${STREAM_ARGS[@]}")
    fi
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    RUN+=("${EXTRA_ARGS[@]}")
fi

echo "== [3/3] Launching: ${RUN[*]}"
echo "   (press q or Esc in the window to quit)"
exec env QT_QPA_PLATFORM=xcb "${RUN[@]}"
