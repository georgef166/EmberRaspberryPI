# Ember — Tactical Geometry HUD for Structural Firefighting

Ember is a helmet-mounted heads-up display for structural firefighters. It processes real-time Time-of-Flight (ToF) depth data on a Raspberry Pi 4B to produce a high-contrast spatial wireframe overlay — mapping the physical geometry of a room through dense smoke where standard optics and thermal imaging fail. A TensorFlow Lite inference pipeline autonomously detects and outlines victims in real time, hands-free.

---

## The Problem

In structural firefighting, smoke routinely drops visibility below 15cm, rendering human sight and standard cameras useless. Teams rely on a single shared handheld thermal imaging camera (TIC) — one that occupies a hand, washes out in high-heat environments where all surfaces merge into a uniform signature, and provides no spatial geometry for navigation. Ember addresses this by giving every firefighter individual, hands-free spatial awareness and real-time victim detection.

---

## Google Technology

**TensorFlow Lite** (TFLite) — `tensorflow/lite` C++ API v2.20  
- Model: MobileNet SSD v1 COCO quantized (`models/detect.tflite`, 4MB)
- Runs fully on-device — no network required in the field
- Processes the ToF amplitude channel (normalized grayscale → 300×300 RGB) at the configured detection FPS
- Detected person bounding boxes are depth-validated against the live depth frame before rendering
- Falls back to classical ToF heuristics if no model file is present

---

## Hardware Requirements

| Component | Spec |
|---|---|
| Raspberry Pi 4B | 4GB RAM recommended |
| Arducam ToF Camera | 240×180, CSI or USB, up to 4m range |
| AM2302 sensor | Optional — temperature/humidity overlay |
| Display | Any HDMI display or compatible glasses/HMD |

---

## Dependencies

Install all dependencies (run from repo root on Raspberry Pi):

```bash
./Install_dependencies.sh
sudo apt-get install -y libtensorflow-lite-dev
```

---

## Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make tactical_rescue
```

---

## Run

```bash
QT_QPA_PLATFORM=xcb ./build/example/cpp/tactical_rescue
```

The `QT_QPA_PLATFORM=xcb` flag forces the X11 display backend — required on Raspberry Pi OS with Wayland enabled.

On launch the app will:
1. Open the ToF camera
2. Load `models/detect.tflite` automatically — TFLite person detection activates if the model file is present
3. Open a fullscreen window with the live Tactical Geometry feed and HUD

### Desktop Launcher

A one-click launcher is provided at `TacticalRescue.desktop` on the Desktop. Double-click and select "Execute".

---

## CLI Options

```
--detector-source MODE    auto | amplitude | confidence | pseudo | tflite
--tflite-model PATH       Path to TFLite model (default: models/detect.tflite)
--range MM                ToF range in mm (default: 4000)
--min-depth MM            Ignore geometry closer than this (default: 200)
--max-depth MM            Ignore geometry farther than this
--person-conf FLOAT       TFLite detection confidence threshold (default: 0.50)
--max-people NUM          Max simultaneous detections rendered (default: 4)
--detection-fps NUM       Inference cadence in frames per second (default: 8)
--no-am2302               Disable AM2302 ambient sensor overlay
--hud-scale NUM           HUD element scale factor (default: 3)
--no-preview              Run capture and inference headless
--show-detector-input     Show the exact frame passed to the TFLite model
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    tactical_rescue                       │
│                                                         │
│  ┌──────────────┐   ┌──────────────────┐   ┌─────────┐ │
│  │ Capture      │   │ Inference Thread  │   │ Render  │ │
│  │ Thread       │──▶│                  │──▶│ Thread  │ │
│  │              │   │ TFLiteDetector   │   │         │ │
│  │ Arducam ToF  │   │ (TensorFlow Lite)│   │ OpenCV  │ │
│  │ 240×180      │   │ MobileNet SSD v1 │   │ HUD     │ │
│  │ depth_mm     │   │ person detection │   │ overlay │ │
│  │ confidence   │   │                  │   │         │ │
│  │ amplitude    │   │ Classical CV     │   │         │ │
│  │              │   │ fallback         │   │         │ │
│  └──────────────┘   └──────────────────┘   └─────────┘ │
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
| `tactical_rescue.cpp` | Main orchestrator, CLI parsing, thread management |
| `tactical_rescue_capture.cpp` | ToF camera acquisition thread |
| `tactical_rescue_perception.cpp` | Classical CV detector (depth/amplitude/confidence heuristics) |
| `tactical_rescue_tflite.cpp` | **TensorFlow Lite** person detection pipeline |
| `tactical_rescue_render.cpp` | Wireframe overlay composition and HUD rendering |
| `models/detect.tflite` | MobileNet SSD v1 COCO quantized model (Google TFLite Model Zoo) |

---

## Detection Pipeline (TFLite)

1. ToF amplitude frame (float32, 240×180) is normalized to uint8
2. Resized to 300×300 and expanded to 3-channel RGB
3. Fed into MobileNet SSD v1 COCO (quantized, 4 output tensors: boxes, classes, scores, count)
4. Person detections (COCO class 1) above confidence threshold are filtered
5. Bounding boxes scaled back to camera resolution
6. Each box depth-validated against live `depth_mm` frame
7. Passed to temporal stabilizer → renderer

In `AUTO` mode, TFLite activates automatically when `models/detect.tflite` is present. When absent, the system falls back to the classical ToF geometry heuristic.

---

## Project Structure

```
Arducam_tof_camera/
├── example/cpp/
│   ├── tactical_rescue.cpp/hpp       # Main app
│   ├── tactical_rescue_tflite.cpp/hpp # TensorFlow Lite integration
│   ├── tactical_rescue_perception.cpp # Classical CV fallback
│   ├── tactical_rescue_render.cpp    # Display and HUD
│   ├── tactical_rescue_capture.cpp   # Camera acquisition
│   ├── preview_depth.cpp             # Simple depth viewer
│   └── capture_raw.cpp               # Raw frame export
├── models/
│   ├── detect.tflite                 # MobileNet SSD v1 COCO (TFLite)
│   └── labelmap.txt                  # COCO class labels
└── example/python/
    └── am2302_stream.py              # AM2302 sensor helper
```

---

## SDG Alignment

**SDG 3 — Good Health and Well-Being**  
Ember directly reduces firefighter fatality and injury risk in zero-visibility structural fire conditions.

**SDG 9 — Industry, Innovation and Infrastructure**  
Demonstrates on-device AI inference on embedded hardware as a viable path for life-safety applications that cannot depend on cloud connectivity.

---

*Built for the GDG on Campus North America Solution Challenge 2026.*  
*Google technology: TensorFlow Lite (tensorflow/lite C++ API v2.20, MobileNet SSD COCO)*
