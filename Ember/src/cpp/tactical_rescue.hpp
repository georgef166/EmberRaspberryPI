#pragma once

#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

namespace tactical_rescue {

constexpr int kDefaultRangeMm = 4000;
constexpr int kFrameTimeoutMs = 200;
constexpr float kMinDepthMm = 200.0f;
constexpr int kDisplayWidth = 1920;
constexpr int kDisplayHeight = 1080;
constexpr int kMaxRenderedPeople = 6;
constexpr int kDefaultDetectionFps = 8;
constexpr int kDefaultAm2302Gpio = 4;
constexpr const char* kDefaultAm2302HelperPath =
    "/home/admin/Desktop/Ember/src/python/am2302_stream.py";

enum class DetectorSource {
    AUTO,
    AMPLITUDE,
    CONFIDENCE,
    PSEUDO,
    TFLITE,
};

struct Options {
    int device = 0;
    int range_mm = kDefaultRangeMm;
    int confidence_threshold = 30;
    int min_depth_mm = static_cast<int>(kMinDepthMm);
    int max_depth_mm = kDefaultRangeMm;
    int hud_scale = 3;
    int detection_fps = kDefaultDetectionFps;
    int am2302_gpio = kDefaultAm2302Gpio;
    int max_people = 4;
    float person_confidence = 0.50f;
    bool no_preview = false;
    bool show_detector_input = false;
    bool enable_am2302 = true;
    DetectorSource detector_source = DetectorSource::AUTO;
    bool edgetpu = false; // Run TFLite inference on the Coral Edge TPU instead of the Pi CPU
    // COCO "person" output index. The bundled 91-class detect.tflite uses 1
    // (index 0 = background "???"). The Coral ssd_mobilenet_v2_coco model uses
    // a 90-class 0-indexed labelmap where person = 0 — pass --person-class 0.
    int person_class_id = 1;
    std::string am2302_helper_path = kDefaultAm2302HelperPath;
    std::string tflite_model_path = "/home/admin/Desktop/Ember/models/detect.tflite";
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
cv::Mat build_wireframe_overlay(const cv::Mat& depth_mm, const cv::Mat& confidence, const Options& opt,
                                float& nearest_mm);
cv::Mat build_amplitude_detector_input(const SharedFrame& lidar_frame, const Options& opt);
cv::Mat build_confidence_detector_input(const SharedFrame& lidar_frame, const Options& opt);
cv::Mat build_pseudo_detector_input(const SharedFrame& lidar_frame, const Options& opt);
DetectionState run_tof_person_detector(const SharedFrame& lidar_frame, DetectorSource source, const Options& opt,
                                       float* best_person_score_out = nullptr);

void draw_person_detection(cv::Mat& frame, const PersonDetection& detection, int index);
void draw_hud(cv::Mat& frame, const RuntimeStats& stats, const DetectionState& detections, bool detector_enabled,
              int scale);
cv::Mat compose_display_canvas(const cv::Mat& source);

} // namespace tactical_rescue
