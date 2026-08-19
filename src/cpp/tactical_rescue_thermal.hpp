#pragma once
#include <memory>
#include "tactical_rescue.hpp"

namespace tactical_rescue {

// Thin wrapper around the Melexis MLX90640 driver (MLX90640_API). Owns the
// calibration parameters and a persistent 32x24 temperature buffer that the
// chess-mode subpages accumulate into. Compiled to a no-op stub unless the
// binary is built with EMBER_HAVE_MLX90640 (CMake links libMLX90640).
class Mlx90640Camera {
public:
    Mlx90640Camera();
    ~Mlx90640Camera();

    // Initialise the I2C device, load EEPROM calibration, and set the refresh
    // rate. Returns false if MLX90640 support is not compiled in or no sensor
    // responds on the bus.
    bool open(const Options& opt);
    bool is_open() const;

    // Read one subpage, update the running temperature grid, and fill `out`.
    // Blocks roughly at the configured refresh cadence. Returns false on I2C error.
    bool read(ThermalFrame& out);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Is a sensor ambient (die) temperature physically possible? A false here means
// the frame's calibration inputs are garbage and the grid must be discarded.
bool thermal_ta_plausible(float ta);

// How long to sleep before the next read() at the given subpage refresh rate,
// so the vendor's untimed busy-poll does not burn a core waiting.
int thermal_read_delay_ms(int refresh_hz);

// Asserts the two above. Run: ./tactical_rescue --selfcheck
void thermal_selfcheck();

// Warm-body victim detection from a thermal frame. Thresholds the configured
// human temperature band, extracts connected blobs, maps each into the ToF
// image space, and depth-validates it against the live ToF frame. This is the
// most reliable victim signal in heavy smoke, where the mono-IR ToF/SSD path
// degrades. Returns PersonDetection boxes in ToF pixel coordinates.
DetectionState detect_thermal_people(const ThermalFrame& thermal, const SharedFrame& tof, const Options& opt,
                                     float* best_score_out = nullptr);

} // namespace tactical_rescue
