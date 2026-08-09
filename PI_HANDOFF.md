# Ember — AR ground-plane grid: bring-up handoff

You are taking over on the Raspberry Pi 4B. The feature is **written but has never been compiled
or parsed** — it was authored on a Windows box with no compiler, no OpenCV, and no Arducam SDK.
You are the first thing that will ever run it. Expect compile errors and treat them as normal,
not as evidence the design is wrong.

## The project, if you are coming to this cold

**Ember** is a helmet-mounted heads-up display for firefighters, running on a Raspberry Pi 4B.
The problem it solves: in a smoke-filled structure a firefighter is effectively blind, and visible
light is useless. Ember maps the room geometry and finds victims through smoke, and renders that
onto a HUD the operator can read while moving.

**Sensors — and note there is no RGB camera anywhere on this rig:**

| Sensor | Role |
|---|---|
| Arducam ToF, 240x180, ~55 deg HFOV, CSI | The primary sensor. Active IR illumination, so it works in zero visible light. Emits depth + confidence + amplitude planes, all `CV_32F`, depth in **millimetres**. Default range 4000 mm. |
| MLX90640-D55 thermal, 32x24, I2C | Warm-body / fire detection. Optional (`libMLX90640`), degrades gracefully if absent. |
| AM2302 / DHT22 | Ambient temp + humidity, read via a forked Python helper. |

The absence of an RGB camera is a deliberate design fact, not a gap — see "Why no RGB matters"
below. Do not add one, and do not assume a color frame exists anywhere in the pipeline.

**What the code already does today (all of it predates this change):**

- `capture_thread` pulls ToF frames and clones depth/confidence/amplitude into a `SharedFrame`
  with a monotonic `sequence`.
- `build_fused_detector_input` (`tactical_rescue_perception.cpp`) synthesizes the visible scene
  image from depth + amplitude: CLAHE'd amplitude base, `COLORMAP_TURBO` depth, confidence-modulated
  brightness, and Canny edges burned in white. This is the "wireframe" look on the HUD.
- `inference_thread` runs a TFLite MobileNet-SSD person detector, on the **Coral Edge TPU** when
  available (~30 FPS) or 4-core CPU otherwise (~8 FPS), and draws amber `VICTIM [nn%]` corner
  brackets. Detections are depth-validated and temporally stabilised.
- `thermal_thread` overlays an INFERNO heat map and raises a fire/hotspot banner.
- The 240x180 scene is upscaled 6x into a 1920x1080 canvas, HUD chrome is drawn on top, and the
  result is served as MJPEG on port 8080 (`--stream`) and/or shown fullscreen via `imshow`.

**What was missing, and why this feature exists:**

Every one of those stages is **2D image-space**. Before this change there was no camera model in
the repo at all — no intrinsics, no unprojection, no point cloud, no notion of a floor. `depth_mm`
was only ever used as a per-pixel scalar: for masking, for colorization, for a per-box mean depth,
and for a nearest-obstacle readout. Grep the history for `fx`, `intrinsic`, or `RANSAC` before this
branch and you get nothing.

So the operator could see *that* something was 1.4 m away, but the system had no idea where the
**ground** was — which means it could not place anything in the world. This feature is the first 3D
stage and the enabling step for AR navigation:

1. **This milestone (done, needs bring-up):** recover the floor as a plane, draw a
   perspective-correct grid on it. Immediate operator value is a readable sense of scale, slope,
   and distance through smoke — a flat green grid receding to a vanishing point tells you far more
   about a room than a false-color depth blob does.
2. **Next milestone (hooks exported, nothing calls them yet):** AR breadcrumbs. Drop a marker on the
   floor, re-project it every frame so it stays pinned as the operator moves — a trail back out of
   the structure. `ray_plane_intersect()` (pixel -> point on the floor) and `project_point()`
   (3D -> pixel) exist for exactly this.

### Why no RGB matters — and why it is an advantage here

On a normal RGB-D rig you unproject from the depth camera and reproject into the color camera
through the extrinsics, and every bit of calibration error shows up as grid lines sliding off the
floor. Ember has no such second sensor. The HUD image is *synthesized from the depth frame itself*,
so depth pixel `(u,v)` and HUD pixel `(u,v)` are the same pixel from the same sensor at the same
instant. Registration is exact by construction, and the occlusion test is free for the same reason.

The one place it will genuinely cost something is the *next* milestone, not this one: with no RGB
there is no feature-based visual odometry, so keeping a breadcrumb pinned while the operator walks
away from it will need depth-only ICP or an IMU. Not your problem today.

## What was added

First 3D stage in Ember: detect the floor as a plane from ToF depth, then paint a
perspective-correct green wireframe grid on it. Groundwork for AR breadcrumbs.

Five files on branch `ar-navigation`:

| File | What |
|---|---|
| `src/cpp/tactical_rescue.hpp` | +109 lines: intrinsics constants, ~24 `Options` fields, `GroundPlane` / `GroundPlaneState` / `GroundPlaneTracker` / `CanvasTransform`, declarations |
| `src/cpp/tactical_rescue_ground.cpp` | NEW, 433 lines: unprojection, RANSAC, covariance refit, temporal tracker, `ray_plane_intersect` |
| `src/cpp/tactical_rescue_render.cpp` | `compute_canvas_transform` (refactored out of `compose_display_canvas`) + `draw_ground_grid` |
| `src/cpp/tactical_rescue.cpp` | intrinsic rescale, 8 CLI flags, ground thread, render hook, join, temporary debug print |
| `src/cpp/CMakeLists.txt` | +1 source file |

Architecture: a dedicated `ground_thread` mirrors the existing `inference_thread` — waits on
`frame_cv`, rate-limits to `--ground-fps` (default 10), publishes `GroundPlaneState` under
`ground_mutex`. The render loop only reads the latest snapshot and draws, in 1920x1080 canvas
space via `CanvasTransform` (scale 6.0, offset x=240 for a 240x180 sensor).

Conventions that matter: camera frame is **X right, Y DOWN, Z forward**, all lengths in **mm**.
Plane is `n.P + d = 0`, `|n| = 1`, `n` always oriented up (`n[1] < 0`). That makes **`d` the camera
height above the floor**, so `d < 0` is a ceiling and `tilt = acos(-n[1]) ~ 90 deg` is a wall. Both
are rejected by `candidate_is_plausible()` before the expensive inlier scan.

## IMPORTANT: you cannot do most of this alone. Plan for a human-assisted session.

You can build, fix compile errors, launch the binary and read the `[ground]` telemetry on stderr.
You **cannot** aim the camera. Almost every meaningful check requires a human physically holding
the helmet in a specific pose while you read the numbers.

| Step | Who |
|---|---|
| 1. Build / fix compile errors | **You, alone.** Do all of this first. |
| 2. Depth convention (flat wall) | Human holds the camera |
| 3-4. Pitch + telemetry | Human holds the camera |
| 5. Visual checks (pan, occlusion) | Human holds camera AND watches the stream |
| 6. Negative cases | Human aims at wall / ceiling / table |
| 7. Performance | Human holds a floor-visible pose |
| 8. Cleanup | **You, alone.** |

**Do not attempt any physical step until you have finished the build and are ready to run.**
Batch all the human-assisted work into one continuous session so the operator is not called back
repeatedly.

### When you are ready, STOP and call the operator explicitly

Say something like: *"Build is clean and the binary runs. I need you for about 15 minutes of
physical testing. Before we start, please have: (a) a tape measure, (b) a phone or laptop on the
same network to view the stream, (c) a chair or a second person for the occlusion test. Tell me
when you are ready and I will walk you through each pose one at a time."*

Then wait. Do not proceed until they answer.

### How to run the guided session

Start the binary yourself in the background with the stream on, capturing stderr so you can read
the telemetry while they watch the video:

    ./run.sh -- --stream 2>&1 | tee /tmp/ember_ground.log

Tell them the URL: `http://<pi-ip>:8080/`. **You watch the `[ground]` lines; they watch the grid.**
That split is the whole point — the two observations together are what diagnose a problem, and
neither alone is sufficient.

For **each** pose below: state the pose, wait for them to confirm they are holding it, let it
settle ~10 seconds, then read the telemetry, tell them what you saw, ask what THEY saw, and record
both before moving on. One pose at a time. Do not fire off the whole list at once.

| # | Pose to ask for | You expect (telemetry) | They should see |
|---|---|---|---|
| A | Camera square-on to a flat wall, ~1.5 m | n/a — this is the `preview_depth` test | centre vs corner distance readings |
| B | Standing, camera **level**, looking down a room | `NONE` | no grid — this is CORRECT, tell them so in advance or they will think it failed |
| C | Standing, pitched **~20 deg down** | `LOCK`, inliers 15-35% | grid appears on the floor |
| D | Same, held steady — ask them to tape-measure the camera height | `height` within ~50 mm of the tape | steady grid |
| E | Pitched down, **pan slowly left-right** | `LOCK` holds, `height`/`tilt` stable | grid lines stay PINNED to floor marks (a tile edge, a scuff) — this is the single most important observation of the whole session |
| F | Pitched down, **pitch slowly up and down** | `tilt` tracks smoothly, no flapping | grid tracks without jumping |
| G | Chair or person walks into frame | `LOCK` unaffected | grid lines vanish behind the object, return cleanly |
| H | Aim at a bare wall | `NONE` | no grid — must NOT lock onto the wall |
| I | Aim at the ceiling | `NONE` | no grid |
| J | Aim at a tabletop | `NONE` | no grid |
| K | Cover the lens with a hand | `STALE` ~1.5 s, then `NONE` | grid dims, then drops — must not blink on/off |
| L | Floor-visible pose, run with and then without `--no-ground` | — | compare the bottom-left `FPS` readout, expect < ~2 FPS delta |

Pose **E** is the one that matters most. A correct plane fit with wrong intrinsics and a wrong
plane fit both produce a grid that *looks* plausible when stationary. Only motion separates them:
if the grid slides across the floor instead of sticking to it, the problem is the intrinsics or the
depth convention, **not** the RANSAC.

If a pose fails, diagnose it from the telemetry before asking them to change anything, and tell
them what you are changing and why. Their time is the scarce resource here — do not make them hold
poses while you guess.

## Do these in order. Order matters.

### 1. Build

    ./compile.sh

Binary lands at `build/src/cpp/tactical_rescue`. Fix compile errors conservatively — match the
existing declaration in `tactical_rescue.hpp`. Do NOT change the header to match a mistake in a
.cpp. If the header and an implementation disagree, the header is the contract.

Most likely failure points, in rough order of probability:

- `std::uniform_int_distribution<size_t>` in `sample_candidate()` — `size_t` is not formally in
  the standard's allowed IntType list. libstdc++ accepts it; if your toolchain complains, switch
  to `std::uniform_int_distribution<unsigned int>` and cast.
- `cv::eigen(covariance, eigenvalues, eigenvectors)` in `refit_least_squares()` — uses the
  `cv::Mat` overload deliberately, because the `Matx` overload's argument order differs across
  OpenCV versions. Eigenvalues come back **descending**, so row 2 of `eigenvectors` is the
  smallest eigenvalue — that is the normal. Do NOT "fix" this to row 0.
- `cv::Vec3f` scalar arithmetic (`n[2] * n`, `-ground.plane.d * n`) and `forward.cross(n)`.
- Missing includes — `<iomanip>`, `<random>`, `<cmath>` were added; add more if needed.

### 2. Settle the depth convention BEFORE looking at the grid

Everything downstream depends on this, and every symptom points the wrong way if it is wrong.

    ./build/src/cpp/preview_depth

Hold the camera square-on to a flat wall at ~1.5 m. The demo reports distance under the mouse.
Compare **frame centre vs a frame corner**:

- Readings equal within noise -> sensor emits **Z (perpendicular) depth**. Keep defaults.
- Corner reads **~15-20% larger** -> sensor emits **radial slant range**. Run everything below
  with `--depth-radial`, which enables the `z = depth / sqrt(1 + nx^2 + ny^2)` correction in
  `unproject_pixel()`. Without it a flat floor bows upward at the frame edges and the inlier
  count collapses.

### 3. Point the camera DOWN — otherwise it will look broken when it is not

The ToF vertical half-FOV is `atan(cy/fy) = atan(90/230.5) = 21.3 deg` (42.7 total). At a standing
helmet height of 1.5 m with the camera **level**, the floor does not enter the bottom of the frame
until **3.84 m** — and `max_depth_mm` is 4000. That is ~160 mm of visible floor, nowhere near the
220 points `ground_min_inliers` requires. A level bench test **correctly** produces no grid.

| Pitch below horizontal | Nearest visible floor @ 1.5 m |
|---|---|
| 0 deg | 3842 mm |
| 10 deg | 2464 mm |
| 20 deg | 1706 mm |
| 30 deg | 1201 mm |
| 40 deg | 820 mm |

**Test at 20-30 deg down.** This is also why `ground_max_tilt_deg` defaults to 45 rather than
tighter: it must pass a genuinely pitched-down camera while still rejecting 90 deg walls.

### 4. Read the bring-up telemetry

There is a **temporary** 1 Hz `std::cerr` in the ground thread, marked
`--- TEMPORARY bring-up instrumentation ---`. It prints:

    [ground] LOCK    height 1480mm  tilt 2.3deg  inliers 612 (23%)  fit 1.71ms

| Symptom | Meaning | Action |
|---|---|---|
| `height` off by a constant % vs a tape measure | `fx`/`fy` wrong, or wrong depth convention | re-run step 2, then `--intrinsics FX,FY,CX,CY` |
| `height` jitters > ~80 mm | too few / too noisy inliers | lower `--ground-stride` (more points), or raise `ground_min_inliers` |
| Always `NONE` with floor clearly visible | some gate too tight | read `tilt` and `inliers` in the print to find WHICH gate fails — do not blindly loosen all of them |
| `fit` > ~5 ms | cloud too large | raise `--ground-stride` (4 -> 6) |
| Flapping `LOCK`/`STALE` | jump-gate rejecting real motion | the `delta_deg > 25.0f` / `350.0f` height check in `GroundPlaneTracker::update` |

Expect `fit` ~1.6 ms and `inliers` ~15-35% of the cloud with the camera pitched down properly.

### 5. Visual checks

    ./run.sh -- --stream        # then open http://<pi-ip>:8080/

- Grid appears within ~0.5 s (3 confirm frames at 10 Hz), lies flat, converges to a vanishing
  point, fades with distance.
- **The real test:** pan and pitch slowly. Grid lines must stay **pinned to floor features** (a
  tile edge, a scuff), not slide with the camera. Sliding means intrinsics or depth convention
  are wrong, NOT the plane fit.
- **Occlusion:** walk a person or push a chair into frame — lines behind them must vanish and
  return cleanly. Punching through objects -> lower `grid_occlusion_tol_mm`; flickering over good
  floor -> raise it.

### 6. Negative cases (all must hold)

- Bare wall -> **no grid**. Must not lock onto the wall.
- Tabletop -> no grid (height gate).
- Ceiling -> no grid (`d < 0`).
- Cover the lens -> grid dims to `STALE` for ~1.5 s, then drops. Must not blink on/off.

### 7. Performance + regression

- Compare bottom-left `FPS` with and without `--no-ground`. Expect < ~2 FPS delta.
- `DETECT <n>ms` must not move — Coral inference is on its own thread and should be unaffected.
- `compose_display_canvas` was **refactored, not changed**. Person boxes, thermal overlay, HUD
  panels and the MJPEG stream must render exactly as before. Verify this explicitly.

### 8. Clean up

Once the gates hold, **delete the temporary debug block** in the ground thread (bounded by
`--- TEMPORARY bring-up instrumentation ---` and `--- end TEMPORARY ---`) and the then-unused
`last_debug_print` declaration.

## New CLI flags

    --no-ground              Disable ground plane detection and the AR grid
    --ground-fps NUM         Fit cadence (default 10)
    --ground-stride NUM      Depth subsample stride (default 4)
    --ground-max-tilt DEG    Max floor tilt vs camera up (default 45)
    --grid-spacing MM        Grid cell size (default 500)
    --grid-extent MM         Grid half-width (default 3000)
    --intrinsics FX,FY,CX,CY Default 230.5,230.5,120,90
    --depth-radial           Depth frame is radial slant range, not Z depth

## Deliberately NOT in this milestone

HUD GND/TILT readout, GRID LOCK/SEARCH state word, `g`-key layer toggle. `ray_plane_intersect()`
and `project_point()` are exported as breadcrumb hooks but nothing calls them yet — that is
intentional.

## Known pre-existing issues — do not fix as a side quest

- `build_fused_detector_input` uses a file-static `cv::Ptr<cv::CLAHE>`
  (`tactical_rescue_perception.cpp:82`) and is called from both the render and inference threads.
  Real data race, predates this work. The ground thread only calls the stateless
  `build_geometry_mask`, so this change does not widen it.
- The render loop reprocesses the same frame every iteration with no `sequence` check. Gating on
  `frame.sequence` would more than pay back the grid's ~1.2 ms. Worth doing, but separately.

## Report back

Compile errors and exactly how you resolved each; the step-2 result (Z vs radial); a few
representative `[ground]` lines; and whether the grid stays pinned under motion.
