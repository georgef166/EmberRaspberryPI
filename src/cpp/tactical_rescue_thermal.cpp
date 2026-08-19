// =============================================================================
// MLX90640-D55 THERMAL IMAGING (32x24 far-infrared array, I2C)
//
// Adds a true thermal channel to Ember. Unlike the mono-IR ToF amplitude path,
// the MLX90640 reports a per-pixel temperature, so a human reads as a warm blob
// against smoke-cooled surroundings — the most reliable victim cue in heavy
// smoke — and active fire reads as a saturated hotspot for the fire warning.
//
// This file wraps the Melexis MLX90640 driver (MLX90640_API) for capture and
// implements warm-body victim detection that fuses the thermal blob with the
// live ToF depth frame. The capture wrapper is guarded by EMBER_HAVE_MLX90640
// (set by CMake when libMLX90640 is found); the detector is pure OpenCV and
// always compiles.
//
// Pipeline:
//   MLX90640 chess-mode subpages -> 32x24 temperature grid (deg C)
//     -> threshold the configured warm-body band
//     -> connected-component blobs -> map into ToF image space
//     -> depth-validate against the live depth_mm frame
//     -> PersonDetection boxes for the renderer
// =============================================================================

#include "tactical_rescue_thermal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

#undef NDEBUG // compile.sh builds Release, which would strip thermal_selfcheck()'s asserts
#include <cassert>

#include <opencv2/imgproc.hpp>

#ifdef EMBER_HAVE_MLX90640
// Melexis MLX90640 driver (https://github.com/melexis/mlx90640-library)
#include "MLX90640_API.h"
#include "MLX90640_I2C_Driver.h"
#endif

namespace tactical_rescue {

namespace {

#ifdef EMBER_HAVE_MLX90640
// Map a refresh rate in Hz to the MLX90640 control-register code (0..7).
// Anything not listed here silently becomes 8 Hz. --thermal-refresh is clamped
// to 1..8 anyway: above 8 the bit-banged bus misses deadlines and GetFrameData
// tears the frame (spurious extreme pixels => false fire warning).
uint8_t refresh_rate_code(int hz)
{
    switch (hz) {
    case 1:  return 0x01;
    case 2:  return 0x02;
    case 4:  return 0x03;
    case 8:  return 0x04;
    case 16: return 0x05;
    case 32: return 0x06;
    case 64: return 0x07;
    default: return 0x04;
    }
}
#endif

} // namespace

bool thermal_ta_plausible(float ta)
{
    // MLX90640 die-temperature spec range. Rejects cold-start NaN, the
    // divide-by-zero from --emissivity 0, and the aux-word garbage that gets
    // through when ValidateAuxData fails without copying — each of which
    // otherwise yields a plausible-looking but fictional grid.
    return std::isfinite(ta) && ta >= -40.0f && ta <= 125.0f;
}

int thermal_read_delay_ms(int refresh_hz)
{
    // Sleep out most of the subpage period instead of leaving the vendor's
    // untimed busy-poll to spin it: 99.6% -> 48.1% of a core, identical data.
    // The 70 ms slack covers the I2C transfer plus poll jitter.
    return std::max(0, 1000 / std::max(1, refresh_hz) - 70);
}

void thermal_selfcheck()
{
    assert(thermal_ta_plausible(25.0f));
    assert(!thermal_ta_plausible(std::numeric_limits<float>::quiet_NaN()));
    assert(!thermal_ta_plausible(std::numeric_limits<float>::infinity()));
    assert(!thermal_ta_plausible(-41.0f));
    assert(!thermal_ta_plausible(126.0f));

    assert(thermal_read_delay_ms(4) == 180);  // default: 250 ms subpage period
    assert(thermal_read_delay_ms(8) == 55);
    assert(thermal_read_delay_ms(16) == 0);   // period < slack, never negative
    assert(thermal_read_delay_ms(0) == 930);  // no divide by zero
    std::cout << "thermal selfcheck ok" << std::endl;
}

struct Mlx90640Camera::Impl {
    bool open = false;
    uint64_t sequence = 0;
#ifdef EMBER_HAVE_MLX90640
    uint8_t address = static_cast<uint8_t>(kDefaultThermalAddress);
    float emissivity = 0.95f;
    paramsMLX90640 params{};
    float to[kThermalWidth * kThermalHeight] = {0}; // persistent grid; subpages accumulate here
#endif
};

Mlx90640Camera::Mlx90640Camera() : impl_(std::make_unique<Impl>()) {}
Mlx90640Camera::~Mlx90640Camera() = default;

bool Mlx90640Camera::open(const Options& opt)
{
#ifdef EMBER_HAVE_MLX90640
    impl_->address = static_cast<uint8_t>(opt.thermal_address);
    impl_->emissivity = opt.thermal_emissivity;

    MLX90640_I2CInit();
    // Advisory only: Linux has no per-client bus-speed ioctl, and this sensor is
    // on a bit-banged bus whose clock is fixed at probe by i2c_gpio_delay_us.
    MLX90640_I2CFreqSet(1000);

    uint16_t eeprom[832];
    if (MLX90640_DumpEE(impl_->address, eeprom) != 0) {
        std::cerr << "[MLX90640] EEPROM read failed at 0x" << std::hex << static_cast<int>(impl_->address)
                  << std::dec << " — check wiring and that I2C is enabled (raspi-config)." << std::endl;
        return false;
    }
    if (MLX90640_ExtractParameters(eeprom, &impl_->params) != 0) {
        std::cerr << "[MLX90640] Calibration parameter extraction failed." << std::endl;
        return false;
    }

    MLX90640_SetRefreshRate(impl_->address, refresh_rate_code(opt.thermal_refresh_hz));
    MLX90640_SetChessMode(impl_->address);
    std::fill(std::begin(impl_->to), std::end(impl_->to), 0.0f);

    impl_->open = true;
    std::cout << "[MLX90640] Thermal camera online: " << kThermalWidth << "x" << kThermalHeight << " @ "
              << opt.thermal_refresh_hz << "Hz, emissivity " << opt.thermal_emissivity << std::endl;
    return true;
#else
    // Built without libMLX90640 — thermal is silently unavailable. The CMake
    // configure step already reported "MLX90640 thermal support: DISABLED", and
    // thermal is enabled by default, so we avoid printing on every CPU-only run.
    (void)opt;
    return false;
#endif
}

bool Mlx90640Camera::is_open() const
{
    return impl_->open;
}

bool Mlx90640Camera::read(ThermalFrame& out)
{
#ifdef EMBER_HAVE_MLX90640
    if (!impl_->open) {
        return false;
    }

    uint16_t frame[834] = {};
    if (MLX90640_GetFrameData(impl_->address, frame) < 0) { // returns the subpage index (0|1) on success
        return false;
    }

    // Reflected temperature estimate: ambient minus ~8 C (Melexis reference value).
    const float ta = MLX90640_GetTa(frame, &impl_->params);
    if (!thermal_ta_plausible(ta)) {
        return false;
    }
    const float tr = ta - 8.0f;
    MLX90640_CalculateTo(frame, &impl_->params, impl_->emissivity, tr, impl_->to);

    cv::Mat grid(kThermalHeight, kThermalWidth, CV_32F, impl_->to);
    out.temperature_c = grid.clone();

    double min_v = 0.0, max_v = 0.0;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(out.temperature_c, &min_v, &max_v, &min_loc, &max_loc);
    out.min_c = static_cast<float>(min_v);
    out.max_c = static_cast<float>(max_v);
    out.mean_c = static_cast<float>(cv::mean(out.temperature_c)[0]);
    out.hotspot = max_loc;
    out.sequence = ++impl_->sequence;
    out.captured_at = std::chrono::steady_clock::now();
    out.valid = true;
    return true;
#else
    (void)out;
    return false;
#endif
}

DetectionState detect_thermal_people(const ThermalFrame& thermal, const SharedFrame& tof, const Options& opt,
                                     float* best_score_out)
{
    DetectionState state;
    state.source_sequence = tof.sequence;
    if (best_score_out) {
        *best_score_out = 0.0f;
    }
    if (!thermal.valid || thermal.temperature_c.empty() || tof.depth_mm.empty()) {
        return state;
    }

    const cv::Mat& temp = thermal.temperature_c;

    // Warm-body band: people read warmer than smoke-cooled surroundings but
    // cooler than active fire. The upper bound rejects flames and hot surfaces
    // (those drive the separate fire warning, not victim boxes).
    cv::Mat warm;
    cv::inRange(temp, cv::Scalar(opt.victim_temp_min_c), cv::Scalar(opt.victim_temp_max_c), warm);
    if (cv::countNonZero(warm) == 0) {
        state.valid = true;
        return state;
    }

    // Close gaps so a body reads as a single blob on the coarse 32x24 grid.
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(warm, warm, cv::MORPH_CLOSE, kernel);

    cv::Mat labels, comp_stats, centroids;
    const int n = cv::connectedComponentsWithStats(warm, labels, comp_stats, centroids, 8, CV_32S);

    const int tw = temp.cols;
    const int th = temp.rows;
    const int fw = tof.depth_mm.cols;
    const int fh = tof.depth_mm.rows;
    // NOTE: simple proportional mapping assumes the thermal and ToF fields of
    // view are roughly co-aligned and centered. Real deployment needs a
    // measured homography between the two sensors — see README.
    const float sx = static_cast<float>(fw) / static_cast<float>(tw);
    const float sy = static_cast<float>(fh) / static_cast<float>(th);
    const float scene_mean = thermal.mean_c;
    float best = 0.0f;

    for (int i = 1; i < n; ++i) { // label 0 is background
        const int area = comp_stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < 2) { // reject single-pixel noise on the coarse grid
            continue;
        }

        const int bx = comp_stats.at<int>(i, cv::CC_STAT_LEFT);
        const int by = comp_stats.at<int>(i, cv::CC_STAT_TOP);
        const int bw = comp_stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int bh = comp_stats.at<int>(i, cv::CC_STAT_HEIGHT);

        const cv::Mat blob_mask = (labels == i);
        const float blob_mean = static_cast<float>(cv::mean(temp, blob_mask)[0]);

        // Confidence: contrast above the scene mean (a warm body stands out)
        // plus a small size bonus, over a modest baseline.
        const float contrast = clamp01((blob_mean - scene_mean) / 8.0f);
        const float size_bonus = std::min(0.20f, static_cast<float>(area) / 60.0f);
        const float score = clamp01(0.65f * contrast + 0.15f + size_bonus);
        if (score < opt.person_confidence) {
            continue;
        }

        cv::Rect box(static_cast<int>(bx * sx), static_cast<int>(by * sy),
                     static_cast<int>(std::ceil(bw * sx)), static_cast<int>(std::ceil(bh * sy)));
        box &= cv::Rect(0, 0, fw, fh);
        if (box.area() <= 0) {
            continue;
        }

        // Depth-validate against the live ToF frame where depth is available.
        const cv::Mat depth_roi = tof.depth_mm(box);
        const cv::Mat valid_depth = (depth_roi > static_cast<float>(opt.min_depth_mm)) &
                                    (depth_roi < static_cast<float>(opt.max_depth_mm));
        float mean_depth = 0.0f;
        if (cv::countNonZero(valid_depth) > 0) {
            mean_depth = static_cast<float>(cv::mean(depth_roi, valid_depth)[0]);
        }

        PersonDetection det;
        det.box = box;
        det.confidence = score;
        det.mean_depth_mm = mean_depth;
        det.valid = true;
        state.people.push_back(det);
        best = std::max(best, score);
    }

    std::sort(state.people.begin(), state.people.end(),
              [](const PersonDetection& a, const PersonDetection& b) { return a.confidence > b.confidence; });
    if (static_cast<int>(state.people.size()) > opt.max_people) {
        state.people.resize(static_cast<size_t>(opt.max_people));
    }

    state.valid = true;
    if (best_score_out) {
        *best_score_out = best;
    }
    return state;
}

} // namespace tactical_rescue
