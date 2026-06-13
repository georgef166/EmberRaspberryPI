# Ember

**Hands-free tactical geometry HUD for structural firefighters.**

Ember is a helmet-mounted heads-up display that uses real-time Time-of-Flight (ToF) depth sensing on a Raspberry Pi 4B to map the physical geometry of a room through dense smoke — exactly the conditions where optical and thermal cameras fail. A **TensorFlow Lite** inference pipeline autonomously detects and outlines victims in real time, fully on-device, with no network dependency.

Built for the **GDG on Campus North America Solution Challenge 2026**.

---

## The Problem

In structural firefighting, smoke routinely drops visibility below 15cm. Teams currently rely on a single shared handheld thermal imaging camera (TIC) that:
- Occupies a firefighter's hand
- Washes out in high-heat environments where surfaces merge into a uniform signature
- Provides no spatial geometry for navigation
- Is shared across the team, leaving most crew members entering the structure blind

Ember replaces this handheld heat-map with an individual, hands-free spatial awareness system for every firefighter.

---

## Google Technology

**TensorFlow Lite** (tensorflow/lite C++ API v2.20) — runs Google's MobileNet SSD COCO quantized person-detection model fully on-device. No network required, which is critical for deployment inside active structure fires where radio and network access are unavailable.

- Model file: `models/detect.tflite` (~4 MB, MobileNet SSD v1)
- Inference cadence: 8 FPS on the Pi's 4 ARM Cortex-A72 cores
- Input: fused ToF view (amplitude texture + colorized depth + confidence mask) resized to the model input
- Output: depth-validated person bounding boxes, rendered into the tactical HUD

> **Optional dependency:** there is **no apt package** for the TensorFlow Lite C++ library on Raspberry Pi OS — it must be [built from source](https://www.tensorflow.org/lite/guide/build_cmake). The build auto-detects it (`EMBER_HAVE_TFLITE`) and, when absent, simply omits the TFLite/Coral detectors — the ToF, **thermal**, and classical CV detectors all build and run without it. The thermal warm-body detector is the recommended victim path anyway and needs no TFLite at all.

**Google Coral Edge TPU** (`libedgetpu`, optional) — when a Coral is attached, the `--coral` / `--edgetpu` backend offloads inference to the Edge TPU, freeing all 4 CPU cores for ToF rendering and running a heavier model (SSD MobileNet **v2** COCO) at ~70+ FPS. Setup via `./Install_coral.sh`; it installs `libedgetpu`, downloads the Edge TPU model, and builds the matching local TFLite C++ runtime under `.ember-deps/`. `./run.sh` auto-selects Coral when the Edge TPU runtime and model are present.

> **Note on detection quality:** the Coral accelerates inference and enables a heavier model, but it does not close the domain gap — the COCO model is trained on visible-light RGB while Ember feeds it a fused ToF composite instead of a normal camera image. The largest accuracy gains will come from fine-tuning a detector on real ToF data, which the Coral can then run at high frame rates. The MLX90640 thermal channel (below) sidesteps this gap entirely.

---

## Thermal Imaging — MLX90640 (optional)

The **MLX90640-D55** is a 32×24 far-infrared thermal array (I²C, ~8 Hz). Unlike the mono-IR ToF amplitude channel, it reports a true **per-pixel temperature**, so a human reads as a warm blob against smoke-cooled surroundings — the most reliable victim cue in heavy smoke — and active fire reads as a saturated hotspot. This directly addresses the visible-light/RGB detection blocker the project hit with COCO models.

When the sensor is present (`./Install_thermal.sh` builds the [Melexis driver](https://github.com/melexis/mlx90640-library) and the build defines `EMBER_HAVE_MLX90640`), Ember adds:

- **Thermal heat-map overlay** — warm regions colorized over the ToF wireframe (cold structure stays as geometry).
- **Fire / hotspot warning** — a HUD banner + max-scene-temperature panel when any pixel exceeds `--fire-temp` (default 60 °C).
- **Warm-body victim detector** (`--detector-source thermal`) — thermal blobs in the configured human temperature band, mapped into ToF image space and depth-validated. Selectable, and the AUTO fallback when no TFLite model is loaded.

The overlay and fire warning are always on when the sensor is present, regardless of which person detector is active, so the Coral/TFLite path and the thermal channel complement each other.

> **Calibration caveat:** the thermal↔ToF mapping currently assumes the two fields of view are roughly co-aligned and centered (simple proportional scaling). For accurate fused boxes in the field, a measured homography between the MLX90640 (55° FOV) and the Arducam ToF is required — this is the next calibration step.

Wiring: `VIN→3V3`, `GND→GND`, `SDA→GPIO2 (pin 3)`, `SCL→GPIO3 (pin 5)`. Default I²C address `0x33`.

---

## Hardware

| Component | Spec |
|---|---|
| Raspberry Pi 4B | 4 GB RAM |
| Arducam ToF Camera | 240×180 native, CSI or USB, up to 4 m range |
| MLX90640-D55 *(optional)* | 32×24 thermal IR array, I²C — heat overlay, fire warning, warm-body victim detection |
| Google Coral *(optional)* | Edge TPU accelerator (USB / M.2) — `--coral` / `--edgetpu` backend |
| AM2302 sensor *(optional)* | Temperature / humidity overlay |
| Display | Any HDMI output or helmet-mounted HMD |

Total bill of materials: **under $150** versus $5,000–$15,000 for a commercial thermal imaging camera.

---

## Quick Start

**One-shot script** — clone once, then everything (setup, build, launch) goes through `run.sh`:

```bash
cd ~/Desktop
git clone https://github.com/georgef166/EmberRaspberryPI.git Ember
cd Ember
./run.sh --setup     # first time: install deps (+ optional Coral/Thermal), build, run
                     # (reboot if prompted, then re-run ./run.sh)

./run.sh             # everyday: build if needed, auto-uses Coral when available
./run.sh --coral     # force the Coral Edge TPU path
./run.sh --cpu       # force the CPU/default detector path
./run.sh --thermal   # launch with thermal-primary victim detection
./run.sh --stream    # also serve the HUD at http://<pi-ip>:8080/
./run.sh --update    # git pull + rebuild + launch
```

<details>
<summary>Or run each step manually</summary>

```bash
./Install_dependencies.sh
./Install_coral.sh          # optional — only if a Google Coral is attached
./Install_thermal.sh        # optional — only if the MLX90640 thermal camera is attached
./compile.sh

# CPU (default) — bundled MobileNet SSD v1 (thermal overlay/fire warning auto-on if present)
QT_QPA_PLATFORM=xcb ./build/src/cpp/tactical_rescue

# Coral Edge TPU — SSD MobileNet v2. --coral selects the Edge TPU model,
# person class, and higher detector cadence automatically.
QT_QPA_PLATFORM=xcb ./build/src/cpp/tactical_rescue --coral

# Thermal-primary victim detection (MLX90640)
QT_QPA_PLATFORM=xcb ./build/src/cpp/tactical_rescue --detector-source thermal

# Remote browser feed on the local network
QT_QPA_PLATFORM=xcb ./build/src/cpp/tactical_rescue --coral --stream
```

</details>

The `QT_QPA_PLATFORM=xcb` flag forces the X11 display backend — required on Raspberry Pi OS with Wayland enabled.

Remote viewing: start with `./run.sh --coral --stream`, then open
`http://<raspberry-pi-ip>:8080/` from another computer on the same network. Use
`hostname -I` on the Pi to find its LAN IP address.

A one-click launcher is provided at `TacticalRescue.desktop`. Double-click and select "Execute".

On launch the app will:
1. Open the ToF camera
2. Load the TFLite model automatically — person detection activates if the model file is present (`./run.sh` selects Coral when available; `--coral` forces it)
3. Open a fullscreen window with the live Tactical Geometry feed and HUD

---

## Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make tactical_rescue
```

The CMake configure step prints `Coral Edge TPU support: ENABLED` when `libedgetpu` is found (after `./Install_coral.sh`), or `DISABLED` for a CPU-only build.

---

## CLI Options

```
--detector-source MODE    auto | amplitude | confidence | pseudo | tflite | thermal
--tflite-input MODE       fused | pseudo | depth | amplitude-debug (default: fused, depth fallback)
--tflite-model PATH        Path to TFLite model (default: models/detect.tflite; Coral default: models/ssd_mobilenet_v2_coco_edgetpu.tflite)
--edgetpu, --coral         Run TFLite inference on the Coral Edge TPU
--person-class NUM         Model output index for 'person' (default 1; Coral default 0)
--no-thermal               Disable the MLX90640 thermal overlay/detector
--thermal-address HEX      MLX90640 I²C address (default 0x33)
--thermal-refresh HZ       MLX90640 refresh rate: 1|2|4|8|16|32|64 (default 8)
--emissivity FLOAT         Thermal emissivity (default 0.95)
--fire-temp C              Hotspot/fire warning threshold in °C (default 60)
--victim-temp-min C        Warm-body band lower bound in °C (default 26)
--victim-temp-max C        Warm-body band upper bound in °C (default 45)
--range MM                 ToF range in mm (default: 4000, explicitly requested at startup)
--min-depth MM             Ignore geometry closer than this (default: 50)
--max-depth MM             Ignore geometry farther than this
--confidence NUM           Depth confidence gate (default: 8)
--person-conf FLOAT        TFLite detection confidence threshold (default: 0.50)
--max-people NUM           Max simultaneous detections rendered (default: 4)
--min-person-pixels NUM    Minimum rendered person box area (default: 900)
--detection-fps NUM        Inference cadence in frames per second (default: 8; Coral default: 30)
--no-am2302                Disable AM2302 ambient sensor overlay
--stream                   Serve the rendered HUD over HTTP MJPEG
--stream-bind ADDR         Stream bind address (default: 0.0.0.0)
--stream-port NUM          Stream HTTP port (default: 8080)
--stream-fps NUM           Stream frame rate cap (default: 10)
--stream-quality NUM       Stream JPEG quality 20-95 (default: 75)
--hud-scale NUM            HUD element scale factor (default: 3)
--no-preview               Run capture and inference headless
--show-detector-input      Show the exact frame passed to the TFLite model
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    tactical_rescue                      │
│                                                         │
│  ┌──────────────┐   ┌──────────────────┐   ┌─────────┐  │
│  │ Capture      │   │ Inference Thread │   │ Render  │  │
│  │ Thread       │──▶│                  │──▶│ Thread  │  │
│  │              │   │ TFLiteDetector   │   │         │  │
│  │ Arducam ToF  │   │ (TensorFlow Lite)│   │ OpenCV  │  │
│  │ 240×180      │   │ CPU or Coral TPU │   │ HUD     │  │
│  │ depth_mm     │   │ SSD person det.  │   │ overlay │  │
│  │ confidence   │   │                  │   │         │  │
│  │ amplitude    │   │ Classical CV     │   │         │  │
│  │              │   │ fallback         │   │         │  │
│  └──────────────┘   └──────────────────┘   └─────────┘  │
│                                                         │
│  ┌──────────────┐   ┌──────────────────┐               │
│  │ AM2302       │   │ MLX90640 Thermal │  Heat overlay, │
│  │ Thread       │   │ Thread (I²C)     │  fire warning, │
│  │ temp/humidity│   │ 32×24 °C grid    │  warm-body det.│
│  └──────────────┘   └──────────────────┘               │
└─────────────────────────────────────────────────────────┘
```

**Key files:**

| File | Role |
|---|---|
| `src/cpp/tactical_rescue.cpp` | Main orchestrator, CLI parsing, thread management |
| `src/cpp/tactical_rescue_capture.cpp` | ToF camera acquisition thread |
| `src/cpp/tactical_rescue_perception.cpp` | Classical CV detector (depth/amplitude/confidence heuristics) |
| `src/cpp/tactical_rescue_tflite.cpp` | **TensorFlow Lite** person detection (CPU + Coral Edge TPU) |
| `src/cpp/tactical_rescue_thermal.cpp` | **MLX90640** thermal capture + warm-body victim detector |
| `src/cpp/tactical_rescue_render.cpp` | Wireframe overlay, thermal overlay, HUD, fire warning |
| `src/cpp/tactical_rescue_stream.cpp` | Optional browser-viewable MJPEG stream of the rendered HUD |
| `models/detect.tflite` | MobileNet SSD v1 COCO quantized model (Google TFLite Model Zoo) |

---

## Detection Pipeline (TFLite)

1. ToF amplitude, depth, and confidence frames are fused into a 240×180 detector view
2. Resized to the model input size and converted to RGB
3. Fed into SSD MobileNet COCO (quantized, 4 output tensors: boxes, classes, scores, count) — on the Pi CPU, or the Coral Edge TPU with `--coral` / `--edgetpu`
4. Person detections (class index `--person-class`) above the candidate threshold are filtered
5. Bounding boxes scaled back to camera resolution
6. Each box is validated against the live `depth_mm` and confidence frames
7. Tiny boxes are rejected, duplicate boxes are suppressed, then detections must persist through the temporal stabilizer before rendering

In `AUTO` mode, TFLite activates automatically when the model file is present; otherwise the **thermal** warm-body detector is used when the MLX90640 is present, falling back to the classical ToF geometry heuristic.

## Detection Pipeline (Thermal)

1. MLX90640 chess-mode subpages → 32×24 per-pixel temperature grid (°C)
2. Threshold the warm-body band (`--victim-temp-min`…`--victim-temp-max`, rejecting fire/hot surfaces)
3. Connected-component blobs → confidence from contrast above the scene mean + size
4. Each blob mapped into ToF image space and depth-validated against the live `depth_mm` frame
5. Boxes passed to the renderer; max scene temperature drives the fire/hotspot warning

---

## Repository Layout

```
EmberRaspberryPI/
├── src/cpp/                       Source + reference demos
│   ├── tactical_rescue.cpp        Main orchestrator, multi-thread pipeline
│   ├── tactical_rescue_tflite.cpp Google TensorFlow Lite / Coral inference
│   ├── tactical_rescue_thermal.cpp   MLX90640 thermal capture + warm-body detector
│   ├── tactical_rescue_capture.cpp   ToF camera acquisition
│   ├── tactical_rescue_perception.cpp Classical CV fallback
│   └── tactical_rescue_render.cpp HUD, wireframe + thermal overlay
├── src/python/
│   └── am2302_stream.py           Temp/humidity sensor helper
├── models/
│   ├── detect.tflite              Google MobileNet SSD v1 COCO (CPU)
│   └── labelmap.txt               COCO class labels
├── CMakeLists.txt                 Top-level build
├── compile.sh                     Build script
├── Install_dependencies.sh        ToF SDK + OpenCV + TFLite setup
├── Install_coral.sh               Coral Edge TPU runtime + v2 model setup
├── Install_thermal.sh             MLX90640 driver build + I²C enablement
├── TacticalRescue.desktop         One-click desktop launcher
└── README.md                      This file
```

---

## SDG Alignment

- **SDG 3 — Good Health and Well-Being**: Directly reduces firefighter fatality and injury risk in zero-visibility structural fires.
- **SDG 9 — Industry, Innovation and Infrastructure**: Demonstrates on-device AI inference on affordable embedded hardware as a viable path for life-safety applications that cannot depend on cloud connectivity.

---

*Built for the GDG on Campus North America Solution Challenge 2026.*
*Google technology: TensorFlow Lite (tensorflow/lite C++ API v2.20, MobileNet SSD COCO) + Google Coral Edge TPU (libedgetpu).*
