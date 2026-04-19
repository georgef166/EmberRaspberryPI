# EmberRaspberryPI

## Overview

This repository is the current Raspberry Pi implementation workspace for the Ember / IgnisXR firefighter assist system.

The target product is a mission-critical wearable augmented-reality interface for low-visibility and zero-visibility environments. The current focus is on using Arducam Time-of-Flight sensing on Raspberry Pi hardware to produce a tactical structural overlay that can help a firefighter understand room geometry, obstacles, openings, and likely victim locations in smoke-filled spaces where ordinary visual navigation is unreliable.

This workspace currently contains the active experimentation and integration work needed to move from basic sensor preview code toward a usable "tactical rescue" feed.

## Current Goals

The current technical goals are:

1. Render a structurally meaningful LiDAR / ToF overlay rather than a raw depth preview.
2. Preserve fine structural detail so walls, door frames, desks, openings, and near obstacles can be interpreted quickly.
3. Overlay human detections in a simple operational style using yellow boxes and confidence labels.
4. Keep the output responsive enough for real-time decision support on Raspberry Pi hardware.
5. Build the system in a way that can later support true multi-sensor fusion rather than one-off demos.

## Current State Of The Code

The active implementation work has been happening inside the Arducam example tree, specifically around:

- `Arducam_tof_camera/example/cpp/tactical_rescue.cpp`

That work currently includes:

- A custom tactical overlay renderer for ToF data.
- A 1920x1080 presentation canvas with reduced unnecessary scaling.
- Thin white structural edge rendering on a black background.
- Yellow human box overlays and confidence labels.
- Additional depth-banded shading to make near, mid, and far geometry more understandable.
- TensorFlow COCO SSD model integration through OpenCV DNN.
- Threaded separation between sensor capture, inference, and rendering.

## What Is Working

The following parts are confirmed working:

- The Arducam ToF pipeline starts successfully on this Raspberry Pi setup.
- The tactical rescue binary builds successfully.
- The TensorFlow COCO SSD model files were downloaded and integrated.
- OpenCV can load the SSD model and pbtxt configuration.
- The tactical overlay rendering path is functioning and has been iteratively tuned for sharper, thinner, higher-detail structure.
- The process remains stable enough to launch and run live.

## What Is Not Working Yet

The main unresolved problem is human detection overlay quality and reliability.

We investigated this directly and found that:

- `/dev/video0` is not a normal RGB webcam feed.
- The active CSI camera node is `arducam-pivariety 10-000c`.
- The active camera format is `Y12` 12-bit grayscale at `240x180`.
- This appears to be the ToF / mono sensor path, not a separate visible-light RGB stream.
- `rpicam-still` also fails because the current camera stack cannot register a normal Pi camera path for this sensor and reports that no cameras are available.

Because of that, the current system does **not** have a confirmed RGB feed available for SSD person detection.

This matters because COCO SSD is trained on ordinary visible-light images. It is not the right model for a raw mono / depth-like ToF stream. Even though the SSD code is integrated correctly, it cannot produce reliable person boxes without a real visible-light feed.

## Key Technical Finding

The most important engineering conclusion so far is:

**The current Raspberry Pi hardware / camera configuration exposes a ToF / mono camera path, but not a usable visible-light RGB feed for SSD.**

That means yellow detection boxes will not work reliably until one of the following happens:

1. A true RGB camera feed is added and made available to OpenCV or libcamera.
2. The detection system is replaced with a ToF-native heuristic or a model trained for depth / mono imagery.

## Why This README Exists

This top-level repository is intended to record the current integration status at the Raspberry Pi system level, not just the code-level experiments inside the Arducam example tree.

It should make it immediately clear to anyone opening the repo:

- what the project is trying to achieve,
- what has already been implemented,
- what has been validated on hardware,
- what is blocked right now,
- and what the next engineering decision needs to be.

## Recommended Next Steps

The next development work should go in one of two directions.

### Option A: Add A Real RGB Camera Feed

This is the fastest path if the product requires standard object detection models such as COCO SSD.

Tasks:

- Attach or enable a visible-light camera source.
- Confirm a real RGB feed is available through V4L2, libcamera, or GStreamer.
- Run SSD on that RGB feed.
- Project detections back into LiDAR display coordinates.
- Keep LiDAR for structure, depth validation, and tactical overlay.

This is the most practical path if the goal is to replicate the reference-style victim boxes using mainstream pretrained models.

### Option B: Move To A ToF-Native Detection Approach

This is the correct path if the product must work from the current mono / ToF hardware alone.

Tasks:

- Build a person-candidate detector from depth, amplitude, confidence, and shape constraints.
- Use heuristic scoring for upright human-sized clusters.
- Optionally collect depth / mono training data and fine-tune a detector on the real sensor domain.
- Replace RGB-trained SSD assumptions with a model that actually matches the sensor input.

This is the better long-term approach if the system is intended to rely primarily on LiDAR / ToF rather than visible-light imagery.

## Current Repository Contents

At the moment, the main items in this workspace are:

- `Arducam_tof_camera/`
  The upstream Arducam ToF example project plus active tactical-rescue modifications.
- `example_cpp_extracted/`
  A standalone extracted C++ example used earlier for isolated build verification.
- `README.md`
  This repository-level status and direction document.

## Build Notes

The tactical rescue target currently builds from inside the Arducam tree with:

```bash
cmake -B /home/admin/Desktop/Arducam_tof_camera/build -S /home/admin/Desktop/Arducam_tof_camera
cmake --build /home/admin/Desktop/Arducam_tof_camera/build --target tactical_rescue -j4
```

## Current Run Command

The current live run command is:

```bash
/home/admin/Desktop/Arducam_tof_camera/build/example/cpp/tactical_rescue
```

Note: this command can launch the ToF overlay path successfully, but reliable human detection remains blocked by the lack of a confirmed RGB source.

## Summary

This repository is no longer at the stage of "basic sensor demo only." The overlay and pipeline work have advanced substantially. The system now has:

- a tactical visual style,
- tuned LiDAR structural rendering,
- integrated detector infrastructure,
- and documented hardware findings.

The main unresolved issue is no longer "how do we write the overlay code?" It is now:

**What sensor path will provide the human-detection input for the MVP?**

Until that is resolved, the project remains in an active integration and hardware-validation phase.
