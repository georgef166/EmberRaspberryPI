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

**TensorFlow Lite** (tensorflow/lite C++ API v2.20) — runs Google's MobileNet SSD v1 COCO quantized person-detection model fully on-device on the Raspberry Pi 4B CPU. No network required, which is critical for deployment inside active structure fires where radio and network access are unavailable.

- Model file: `Ember/models/detect.tflite` (~4 MB)
- Inference cadence: 8 FPS on the Pi's 4 ARM Cortex-A72 cores
- Input: ToF amplitude channel (infrared reflection intensity) normalized to 300×300 uint8 RGB
- Output: depth-validated person bounding boxes, rendered into the tactical HUD

---

## Quick Start

```bash
cd Ember
./Install_dependencies.sh
sudo apt-get install -y libtensorflow-lite-dev
./compile.sh
QT_QPA_PLATFORM=xcb ./build/src/cpp/tactical_rescue
```

A desktop launcher (`TacticalRescue.desktop`) is also provided for one-click operation.

Full build instructions, architecture diagram, CLI options, and SDG alignment: see [`Ember/README.md`](Ember/README.md).

---

## Repository Layout

```
EmberPI/
├── Ember/
│   ├── src/cpp/                    Ember source + reference demos
│   │   ├── tactical_rescue.cpp     Main orchestrator, 3-thread pipeline
│   │   ├── tactical_rescue_tflite.cpp   Google TensorFlow Lite inference
│   │   ├── tactical_rescue_capture.cpp  ToF camera acquisition
│   │   ├── tactical_rescue_perception.cpp  Classical CV fallback
│   │   └── tactical_rescue_render.cpp   HUD and wireframe rendering
│   ├── src/python/
│   │   └── am2302_stream.py        Temp/humidity sensor helper
│   ├── models/
│   │   ├── detect.tflite           Google MobileNet SSD v1 COCO
│   │   └── labelmap.txt            COCO class labels
│   ├── compile.sh                  Build script
│   └── README.md                   Full technical documentation
├── TacticalRescue.desktop          One-click desktop launcher
└── README.md                       This file
```

---

## Hardware

| Component | Spec |
|---|---|
| Raspberry Pi 4B | 4 GB RAM |
| Arducam ToF Camera | 240×180 native, CSI or USB, up to 4 m range |
| AM2302 sensor *(optional)* | Temperature / humidity overlay |
| Display | Any HDMI output or helmet-mounted HMD |

Total bill of materials: **under $150** versus $5,000–$15,000 for a commercial thermal imaging camera.

---

## SDG Alignment

- **SDG 3 — Good Health and Well-Being**: Directly reduces firefighter fatality and injury risk in zero-visibility structural fires.
- **SDG 9 — Industry, Innovation and Infrastructure**: Demonstrates on-device AI inference on affordable embedded hardware as a viable path for life-safety applications that cannot depend on cloud connectivity.
