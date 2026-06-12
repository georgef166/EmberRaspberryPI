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
- Input: ToF amplitude channel (infrared reflection intensity) normalized to 300×300 uint8 RGB
- Output: depth-validated person bounding boxes, rendered into the tactical HUD

**Google Coral Edge TPU** (`libedgetpu`, optional) — when a Coral is attached, the `--edgetpu` backend offloads inference to the Edge TPU, freeing all 4 CPU cores for ToF rendering and running a heavier model (SSD MobileNet **v2** COCO) at ~70+ FPS. Setup via `./Install_coral.sh`; the build auto-detects the Coral and falls back to CPU-only when absent.

> **Note on detection quality:** the Coral accelerates inference and enables a heavier model, but it does not close the domain gap — the COCO model is trained on visible-light RGB while the input is the ToF amplitude (mono IR) channel. The largest accuracy gains will come from fine-tuning a detector on real ToF data, which the Coral can then run at high frame rates.

---

## Hardware

| Component | Spec |
|---|---|
| Raspberry Pi 4B | 4 GB RAM |
| Arducam ToF Camera | 240×180 native, CSI or USB, up to 4 m range |
| Google Coral *(optional)* | Edge TPU accelerator (USB / M.2) — `--edgetpu` backend |
| AM2302 sensor *(optional)* | Temperature / humidity overlay |
| Display | Any HDMI output or helmet-mounted HMD |

Total bill of materials: **under $150** versus $5,000–$15,000 for a commercial thermal imaging camera.

---

## Quick Start

```bash
./Install_dependencies.sh
sudo apt-get install -y libtensorflow-lite-dev
./Install_coral.sh          # optional — only if a Google Coral is attached
./compile.sh

# CPU (default) — bundled MobileNet SSD v1
QT_QPA_PLATFORM=xcb ./build/src/cpp/tactical_rescue

# Coral Edge TPU — SSD MobileNet v2 (note --person-class 0 for the v2 COCO labelmap)
QT_QPA_PLATFORM=xcb ./build/src/cpp/tactical_rescue \
    --edgetpu --tflite-model models/ssd_mobilenet_v2_coco_edgetpu.tflite --person-class 0
```

The `QT_QPA_PLATFORM=xcb` flag forces the X11 display backend — required on Raspberry Pi OS with Wayland enabled.

A one-click launcher is provided at `TacticalRescue.desktop`. Double-click and select "Execute".

On launch the app will:
1. Open the ToF camera
2. Load the TFLite model automatically — person detection activates if the model file is present (Coral when `--edgetpu` is set, else CPU)
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
--detector-source MODE    auto | amplitude | confidence | pseudo | tflite
--tflite-model PATH        Path to TFLite model (default: models/detect.tflite)
--edgetpu                  Run TFLite inference on the Coral Edge TPU
--person-class NUM         Model output index for 'person' (default 1; use 0 for Coral v2 COCO)
--range MM                 ToF range in mm (default: 4000)
--min-depth MM             Ignore geometry closer than this (default: 200)
--max-depth MM             Ignore geometry farther than this
--person-conf FLOAT        TFLite detection confidence threshold (default: 0.50)
--max-people NUM           Max simultaneous detections rendered (default: 4)
--detection-fps NUM        Inference cadence in frames per second (default: 8)
--no-am2302                Disable AM2302 ambient sensor overlay
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
│  ┌──────────────┐                                       │
│  │ AM2302       │  Temperature/humidity overlay         │
│  │ Thread       │  (optional, via Python subprocess)    │
│  └──────────────┘                                       │
└─────────────────────────────────────────────────────────┘
```

**Key files:**

| File | Role |
|---|---|
| `src/cpp/tactical_rescue.cpp` | Main orchestrator, CLI parsing, thread management |
| `src/cpp/tactical_rescue_capture.cpp` | ToF camera acquisition thread |
| `src/cpp/tactical_rescue_perception.cpp` | Classical CV detector (depth/amplitude/confidence heuristics) |
| `src/cpp/tactical_rescue_tflite.cpp` | **TensorFlow Lite** person detection (CPU + Coral Edge TPU) |
| `src/cpp/tactical_rescue_render.cpp` | Wireframe overlay composition and HUD rendering |
| `models/detect.tflite` | MobileNet SSD v1 COCO quantized model (Google TFLite Model Zoo) |

---

## Detection Pipeline (TFLite)

1. ToF amplitude frame (float32, 240×180) is normalized to uint8
2. Resized to the model input size and expanded to 3-channel RGB
3. Fed into SSD MobileNet COCO (quantized, 4 output tensors: boxes, classes, scores, count) — on the Pi CPU, or the Coral Edge TPU with `--edgetpu`
4. Person detections (class index `--person-class`) above the confidence threshold are filtered
5. Bounding boxes scaled back to camera resolution
6. Each box depth-validated against the live `depth_mm` frame
7. Passed to temporal stabilizer → renderer

In `AUTO` mode, TFLite activates automatically when the model file is present. When absent, the system falls back to the classical ToF geometry heuristic.

---

## Repository Layout

```
EmberRaspberryPI/
├── src/cpp/                       Source + reference demos
│   ├── tactical_rescue.cpp        Main orchestrator, 3-thread pipeline
│   ├── tactical_rescue_tflite.cpp Google TensorFlow Lite / Coral inference
│   ├── tactical_rescue_capture.cpp    ToF camera acquisition
│   ├── tactical_rescue_perception.cpp Classical CV fallback
│   └── tactical_rescue_render.cpp HUD and wireframe rendering
├── src/python/
│   └── am2302_stream.py           Temp/humidity sensor helper
├── models/
│   ├── detect.tflite              Google MobileNet SSD v1 COCO (CPU)
│   └── labelmap.txt               COCO class labels
├── CMakeLists.txt                 Top-level build
├── compile.sh                     Build script
├── Install_dependencies.sh        ToF SDK + OpenCV + TFLite setup
├── Install_coral.sh               Coral Edge TPU runtime + v2 model setup
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
