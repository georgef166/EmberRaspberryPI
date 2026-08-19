#pragma once

#include <chrono>
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
// Subpage rate: a full 32x24 grid takes two subpages, so 4 Hz = 2 full frames/s.
// Measured on the bit-banged bus: 29% duty, 0 deadline misses in 120 reads.
constexpr int kDefaultThermalRefreshHz = 4;

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

} // namespace tactical_rescue
