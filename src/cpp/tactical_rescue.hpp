#pragma once

#include <chrono>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

namespace tactical_rescue {

constexpr int kDefaultRangeMm = 4000;
constexpr int kFrameTimeoutMs = 200;
constexpr float kMinDepthMm = 50.0f;
constexpr int kDisplayWidth = 1920;
constexpr int kDisplayHeight = 1080;
constexpr int kMaxRenderedPeople = 6;
constexpr int kDefaultDetectionFps = 8;
constexpr int kDefaultEdgeTpuDetectionFps = 30;
constexpr int kDefaultAm2302Gpio = 4;
constexpr const char* kDefaultAm2302HelperPath =
    "/home/admin/Desktop/Ember/src/python/am2302_stream.py";
constexpr const char* kDefaultTfliteModelPath = "/home/admin/Desktop/Ember/models/detect.tflite";
constexpr const char* kDefaultEdgeTpuModelPath =
    "/home/admin/Desktop/Ember/models/ssd_mobilenet_v2_coco_edgetpu.tflite";
constexpr int kDefaultCpuPersonClassId = 1;
constexpr int kDefaultEdgeTpuPersonClassId = 0;
constexpr int kDefaultStreamPort = 8080;
constexpr int kDefaultStreamJpegQuality = 75;
constexpr int kDefaultStreamFps = 10;
constexpr const char* kDefaultStreamBindAddress = "0.0.0.0";
constexpr const char* kDefaultStreamPassword = "admin";

// --- Arducam ToF pinhole intrinsics (240x180 native, ~55 deg HFOV) ---
// Treated as a reference calibration at the resolution below; scale_intrinsics_to_frame()
// rescales them at startup if the sensor reports something else, so nothing here
// re-introduces the hardcoded frame size the capture path carefully avoids.
constexpr float kIntrinsicsRefWidth = 240.0f;
constexpr float kIntrinsicsRefHeight = 180.0f;
constexpr float kDefaultFx = 230.5f;
constexpr float kDefaultFy = 230.5f;
constexpr float kDefaultCx = 120.0f;
constexpr float kDefaultCy = 90.0f;

constexpr int kDefaultGroundFps = 10;
constexpr int kMaxGridSegments = 1400;  // hard cap on cv::line calls per rendered frame

enum class DetectorSource {
    AUTO,
    AMPLITUDE,
    CONFIDENCE,
    PSEUDO,
    TFLITE,
    THERMAL,
};

enum class TfliteInputMode {
    AMPLITUDE,
    FUSED,
    PSEUDO,
    DEPTH,
};

// MLX90640-D55 far-infrared thermal array (32x24, I2C).
constexpr int kThermalWidth = 32;
constexpr int kThermalHeight = 24;
constexpr int kDefaultThermalAddress = 0x33;
constexpr int kDefaultThermalRefreshHz = 8;

struct Options {
    int device = 0;
    int range_mm = kDefaultRangeMm;
    int confidence_threshold = 8;
    int min_depth_mm = static_cast<int>(kMinDepthMm);
    int max_depth_mm = kDefaultRangeMm;
    int hud_scale = 3;
    int detection_fps = kDefaultDetectionFps;
    int am2302_gpio = kDefaultAm2302Gpio;
    int max_people = 4;
    int min_person_box_pixels = 900;
    float person_confidence = 0.50f;
    bool no_preview = false;
    bool show_detector_input = false;
    bool enable_am2302 = true;
    bool enable_stream = false;
    int stream_port = kDefaultStreamPort;
    int stream_jpeg_quality = kDefaultStreamJpegQuality;
    int stream_fps = kDefaultStreamFps;
    std::string stream_bind_address = kDefaultStreamBindAddress;
    // Commander view auth. The remote console can place markup that is burned
    // into the firefighter's HUD, so it is gated by default.
    bool stream_auth_enabled = true;
    std::string stream_password = kDefaultStreamPassword;
    DetectorSource detector_source = DetectorSource::AUTO;
    TfliteInputMode tflite_input_mode = TfliteInputMode::FUSED;
    bool edgetpu = false; // Run TFLite inference on the Coral Edge TPU instead of the Pi CPU
    // COCO "person" output index. The bundled 91-class detect.tflite uses 1
    // (index 0 = background "???"). The Coral ssd_mobilenet_v2_coco model uses
    // a 90-class 0-indexed labelmap where person = 0 — pass --person-class 0.
    int person_class_id = kDefaultCpuPersonClassId;
    bool detection_fps_explicit = false;
    bool person_class_explicit = false;
    bool tflite_model_explicit = false;
    // --- MLX90640 thermal imaging ---
    bool enable_thermal = true;        // Read the MLX90640 and render the thermal overlay
    int thermal_address = kDefaultThermalAddress;
    int thermal_refresh_hz = kDefaultThermalRefreshHz;
    float thermal_emissivity = 0.95f;  // Skin/fabric ~0.95; tune per surface
    float fire_temp_c = 60.0f;         // Hotspot/fire warning threshold
    float victim_temp_min_c = 26.0f;   // Warm-body band: lower bound (people through smoke)
    float victim_temp_max_c = 45.0f;   // Warm-body band: upper bound (above => fire, not a person)
    float thermal_overlay_alpha = 0.45f;
    // --- Ground plane detection / AR navigation grid ---
    bool enable_ground_plane = true;
    // Arducam DEPTH_FRAME is assumed to be perpendicular (Z) depth. If the unit
    // emits radial slant range instead, pass --depth-radial: at 55 deg HFOV the
    // frame corners run ~20% long, which bows a flat floor and starves the fit.
    bool depth_is_radial = false;
    bool intrinsics_explicit = false;
    float fx = kDefaultFx;
    float fy = kDefaultFy;
    float cx = kDefaultCx;
    float cy = kDefaultCy;
    int ground_fps = kDefaultGroundFps;
    int ground_stride = 4;                 // 240x180 at stride 4 -> ~2700 candidate points
    int ground_iterations = 64;
    int ground_min_inliers = 220;
    float ground_inlier_mm = 30.0f;        // base band; 1% of range is added on top
    float ground_min_inlier_ratio = 0.12f;
    float ground_max_tilt_deg = 45.0f;     // rejects walls (~90 deg), survives a head-down operator
    float ground_min_height_mm = 500.0f;   // crouching
    float ground_max_height_mm = 2300.0f;  // standing, helmet-mounted
    float ground_smoothing = 0.25f;        // EMA weight on each accepted fit
    int ground_confirm_frames = 3;         // fits required before the grid is shown
    int ground_hold_frames = 15;           // ~1.5 s of coasting at 10 Hz before dropping the lock
    float grid_spacing_mm = 500.0f;
    float grid_half_width_mm = 3000.0f;
    float grid_near_mm = 300.0f;
    float grid_far_mm = 4000.0f;
    float grid_segment_mm = 200.0f;        // tessellation step: clip + occlusion granularity
    float grid_occlusion_tol_mm = 150.0f;
    std::string am2302_helper_path = kDefaultAm2302HelperPath;
    std::string tflite_model_path = kDefaultTfliteModelPath;
};

class LineFilter {
public:
    explicit LineFilter(int target_fd);
    ~LineFilter();

    bool start();
    void stop();

private:
    static bool should_suppress(const std::string& line);
    void flush_line(const std::string& line);
    void pump();

    int pipe_fds_[2] = {-1, -1};
    int original_fd_ = -1;
    int target_fd_ = -1;
    bool active_ = false;
    std::thread worker_;
};

struct SharedFrame {
    cv::Mat depth_mm;
    cv::Mat confidence;
    cv::Mat amplitude;
    uint64_t sequence = 0;
    std::chrono::steady_clock::time_point captured_at;
};

// A ground plane in ToF camera coordinates: X right, Y DOWN, Z forward, all in
// millimetres to match depth_mm. Satisfies n.P + d = 0 with |n| = 1 and n oriented
// up (normal[1] < 0). That orientation makes d the camera's height above the floor,
// so a single d > 0 test throws out ceilings and a tilt test throws out walls.
struct GroundPlane {
    cv::Vec3f normal{0.0f, -1.0f, 0.0f};
    float d = 0.0f;
    float height_mm = 0.0f;
    float tilt_deg = 0.0f;
    int inlier_count = 0;
    float inlier_ratio = 0.0f;
    bool valid = false;
};

struct GroundPlaneState {
    GroundPlane plane;
    uint64_t source_sequence = 0;
    bool stale = false;   // coasting on the last good fit; the grid is drawn dimmed
    double fit_ms = 0.0;
};

// Owns the RANSAC scratch buffers, the RNG, and the temporal filter. Exactly one
// instance lives on the ground thread. Deliberately NOT function-local statics —
// the existing detection trackers use those and are only accidentally thread-safe.
class GroundPlaneTracker {
public:
    GroundPlaneState update(const SharedFrame& lidar_frame, const Options& opt);

private:
    std::mt19937 rng_{0xE13E2Du};
    std::vector<cv::Point3f> points_;
    std::vector<int> seeds_;
    std::vector<int> all_indices_;
    std::vector<int> inliers_;
    GroundPlane smoothed_;
    int confirm_hits_ = 0;
    int miss_frames_ = 0;
};

// The letterbox mapping compose_display_canvas() applies when it blits the scene
// image into the 1920x1080 canvas. For 240x180 this is scale 6.0, offset (240, 0).
// Exposed so 3D geometry can be rasterised directly at canvas resolution instead
// of being nearest-neighbour upscaled with the scene.
struct CanvasTransform {
    double scale = 1.0;
    int offset_x = 0;
    int offset_y = 0;

    cv::Point2f map(const cv::Point2f& source_px) const
    {
        return cv::Point2f(static_cast<float>(offset_x + source_px.x * scale),
                           static_cast<float>(offset_y + source_px.y * scale));
    }
};

// One MLX90640 readout: a 32x24 grid of per-pixel temperatures in degrees C,
// plus derived scene statistics used by the detector, overlay, and HUD.
struct ThermalFrame {
    cv::Mat temperature_c;          // kThermalHeight x kThermalWidth, CV_32F (deg C)
    float min_c = 0.0f;
    float max_c = 0.0f;
    float mean_c = 0.0f;
    cv::Point hotspot{-1, -1};      // pixel of peak temperature in 32x24 space
    uint64_t sequence = 0;
    bool valid = false;
    std::chrono::steady_clock::time_point captured_at{};
};

struct PersonDetection {
    cv::Rect box;
    cv::Mat mask;
    float confidence = 0.0f;
    float mean_depth_mm = 0.0f;
    bool from_segmentation = false;
    bool valid = false;
};

struct DetectionState {
    std::vector<PersonDetection> people;
    uint64_t source_sequence = 0;
    bool valid = false;
    bool used_segmentation = false;
};

struct RuntimeStats {
    double capture_fps = 0.0;
    double render_fps = 0.0;
    double inference_ms = 0.0;
    double frame_age_ms = 0.0;
    float nearest_obstacle_mm = 0.0f;
    int detected_people = 0;
    float best_person_score = 0.0f;
    bool detector_uses_segmentation = false;
    bool ambient_valid = false;
    bool ambient_enabled = false;
    float ambient_temperature_c = 0.0f;
    float ambient_humidity_percent = 0.0f;
    double ambient_age_s = 0.0;
    std::chrono::steady_clock::time_point ambient_updated_at{};
    std::string ambient_status = "OFF";
    std::string detector_source_label = "OFF";
    // --- MLX90640 thermal ---
    bool thermal_enabled = false;
    bool thermal_valid = false;
    float thermal_max_c = 0.0f;
    float thermal_min_c = 0.0f;
    float thermal_mean_c = 0.0f;
    bool fire_warning = false;
    double thermal_age_s = 0.0;
    std::chrono::steady_clock::time_point thermal_updated_at{};
    std::string thermal_status = "OFF";
};

class FpsCounter {
public:
    void tick()
    {
        using clock = std::chrono::steady_clock;
        const auto now = clock::now();
        if (!started_) {
            started_ = true;
            last_ = now;
            return;
        }
        const auto dt = std::chrono::duration<double>(now - last_).count();
        last_ = now;
        if (dt > 0.0) {
            const double instant = 1.0 / dt;
            fps_ = fps_ == 0.0 ? instant : (fps_ * 0.9 + instant * 0.1);
        }
    }

    double value() const { return fps_; }

private:
    bool started_ = false;
    double fps_ = 0.0;
    std::chrono::steady_clock::time_point last_{};
};

float clamp01(float value);

bool file_exists(const std::string& path);

const char* detector_source_label(DetectorSource source);
cv::Mat to_gray_preview(const cv::Mat& depth_mm, const Options& opt);
cv::Mat build_geometry_mask(const cv::Mat& depth_mm, const cv::Mat& confidence, const Options& opt);
cv::Mat build_fused_detector_input(const SharedFrame& lidar_frame, const Options& opt);
cv::Mat build_wireframe_overlay(const SharedFrame& lidar_frame, const Options& opt, float& nearest_mm);
cv::Mat build_amplitude_detector_input(const SharedFrame& lidar_frame, const Options& opt);
cv::Mat build_confidence_detector_input(const SharedFrame& lidar_frame, const Options& opt);
cv::Mat build_pseudo_detector_input(const SharedFrame& lidar_frame, const Options& opt);
DetectionState run_tof_person_detector(const SharedFrame& lidar_frame, DetectorSource source, const Options& opt,
                                       float* best_person_score_out = nullptr);

// Maps a source (ToF-resolution) point into the composed 1920x1080 display
// canvas, matching compose_display_canvas()'s letterbox fit. Exposed so overlays
// can be drawn at full display resolution instead of being upscaled with the
// image (which made labels huge and box edges chunky).
void display_canvas_transform(const cv::Size& source, double& scale, cv::Point& offset);

void draw_person_detection(cv::Mat& frame, const PersonDetection& detection, int index, double scale = 1.0,
                           const cv::Point& offset = cv::Point(0, 0));
void draw_thermal_overlay(cv::Mat& frame, const ThermalFrame& thermal, const Options& opt);
void draw_hud(cv::Mat& frame, const RuntimeStats& stats, const DetectionState& detections, bool detector_enabled,
              int scale);
cv::Mat compose_display_canvas(const cv::Mat& source);

void scale_intrinsics_to_frame(Options& opt, int width, int height);
cv::Point3f unproject_pixel(float u, float v, float depth_mm, const Options& opt);
bool project_point(const cv::Point3f& camera_point, const Options& opt, cv::Point2f& out_px);
bool ray_plane_intersect(const GroundPlane& plane, float u, float v, const Options& opt,
                         cv::Point3f& out_point);

CanvasTransform compute_canvas_transform(const cv::Size& source_size);
void draw_ground_grid(cv::Mat& canvas, const GroundPlaneState& ground, const SharedFrame& lidar_frame,
                      const CanvasTransform& transform, const Options& opt);

} // namespace tactical_rescue
