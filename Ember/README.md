# Ember — Tactical Geometry HUD for Structural Firefighting

Ember is a helmet-mounted heads-up display for structural firefighters. It processes real-time Time-of-Flight (ToF) depth data on a Raspberry Pi 4B to produce a high-contrast spatial wireframe overlay — mapping the physical geometry of a room through dense smoke where standard optics and thermal imaging fail. A TensorFlow Lite inference pipeline autonomously detects and outlines victims in real time, hands-free.

---

## The Problem

In structural firefighting, smoke routinely drops visibility below 15cm, rendering human sight and standard cameras useless. Teams rely on a single shared handheld thermal imaging camera (TIC) — one that occupies a hand, washes out in high-heat environments where all surfaces merge into a uniform signature, and provides no spatial geometry for navigation. Ember addresses this by giving every firefighter individual, hands-free spatial awareness and real-time victim detection.

---

## Google Technology

**TensorFlow Lite** (TFLite) — `tensorflow/lite` C++ API v2.20  
- Model: MobileNet SSD COCO quantized — v1 on CPU (`models/detect.tflite`, 4MB)
- Runs fully on-device — no network required in the field
- Processes the ToF amplitude channel (normalized grayscale → 300×300 RGB) at the configured detection FPS
- Detected person bounding boxes are depth-validated against the live depth frame before rendering
- Falls back to classical ToF heuristics if no model file is present

**Google Coral Edge TPU** (`libedgetpu`) — optional hardware acceleration  
- With a Coral attached, the `--edgetpu` backend offloads the same SSD inference to the Edge TPU instead of the Pi's ARM cores
- Frees all 4 CPU cores for ToF rendering and pushes inference to ~70+ FPS, leaving headroom for a larger model (SSD MobileNet **v2** COCO, `models/ssd_mobilenet_v2_coco_edgetpu.tflite`)
- Requires an Edge TPU-compiled model; the CMake build auto-detects `libedgetpu` and enables the backend (`EMBER_HAVE_EDGETPU`). Without the Coral, the binary still builds and runs CPU-only
- Setup: `./Install_coral.sh` (installs runtime + dev headers, downloads the v2 model)

> **Note on detection quality:** the Coral accelerates inference and enables a heavier model, but it does not close the domain gap — the COCO model is trained on visible-light RGB while the input is the ToF amplitude (mono IR) channel. The largest accuracy gains will come from fine-tuning a detector on real ToF data, which the Coral can then run at high frame rates.

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

### Optional: Google Coral Edge TPU

If a Coral (USB Accelerator or M.2/PCIe) is attached, install the Edge TPU runtime, C++ headers, and the v2 model:

```bash
./Install_coral.sh
```

The next `cmake` configure will print `Coral Edge TPU support: ENABLED` and build the `--edgetpu` backend.

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
# CPU (default) — bundled MobileNet SSD v1
QT_QPA_PLATFORM=xcb ./build/src/cpp/tactical_rescue

# Coral Edge TPU — SSD MobileNet v2 (note --person-class 0 for the v2 COCO labelmap)
QT_QPA_PLATFORM=xcb ./build/src/cpp/tactical_rescue \
    --edgetpu \
    --tflite-model models/ssd_mobilenet_v2_coco_edgetpu.tflite \
    --person-class 0
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
--edgetpu                 Run TFLite inference on the Coral Edge TPU
--person-class NUM        Model output index for 'person' (default 1; use 0 for Coral v2 COCO)
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
│                    tactical_rescue                      │
│                                                         │
│  ┌──────────────┐   ┌──────────────────┐   ┌─────────┐  │
│  │ Capture      │   │ Inference Thread │   │ Render  │  │
│  │ Thread       │──▶│                  │──▶│ Thread  │  │
│  │              │   │ TFLiteDetector   │   │         │  │
│  │ Arducam ToF  │   │ (TensorFlow Lite)│   │ OpenCV  │  │
│  │ 240×180      │   │ MobileNet SSD v1 │   │ HUD     │  │
│  │ depth_mm     │   │ person detection │   │ overlay │  │
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
Ember/
├── src/cpp/
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
└── src/python/
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
