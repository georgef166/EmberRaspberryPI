#pragma once

#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/videoio.hpp>

namespace tactical_rescue {

constexpr int kDefaultRangeMm = 4000;
constexpr int kFrameTimeoutMs = 200;
constexpr float kMinDepthMm = 200.0f;
constexpr int kDisplayWidth = 1920;
constexpr int kDisplayHeight = 1080;
constexpr int kMaxDisplayScale = 4;
constexpr int kDefaultDetectorInput = 300;
constexpr int kMaxRenderedPeople = 6;
constexpr const char* kDefaultSsdModelPath =
    "/home/admin/Desktop/Arducam_tof_camera/models/tensorflow_coco_ssd/frozen_inference_graph.pb";
constexpr const char* kDefaultSsdConfigPath =
    "/home/admin/Desktop/Arducam_tof_camera/models/tensorflow_coco_ssd/graph.pbtxt";
constexpr int kCocoPersonClassId = 1;
constexpr int kDefaultDetectionFps = 3;

enum class DetectorSource {
    AUTO,
    RGB,
    AMPLITUDE,
    CONFIDENCE,
    PSEUDO,
};

struct Options {
    int device = 0;
    int rgb_device = 0;
    int rgb_width = 640;
    int rgb_height = 480;
    int range_mm = kDefaultRangeMm;
    int confidence_threshold = 30;
    int min_depth_mm = static_cast<int>(kMinDepthMm);
    int max_depth_mm = kDefaultRangeMm;
    int hud_scale = 3;
    int detector_input = kDefaultDetectorInput;
    int detection_fps = kDefaultDetectionFps;
    int max_people = 4;
    float person_confidence = 0.50f;
    float nms_threshold = 0.45f;
    bool no_preview = false;
    bool show_detector_input = false;
    bool rgb_libcamera = false;
    DetectorSource detector_source = DetectorSource::AUTO;
    std::string model_path = kDefaultSsdModelPath;
    std::string config_path = kDefaultSsdConfigPath;
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

struct SharedRgbFrame {
    cv::Mat bgr;
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
bool open_rgb_capture(cv::VideoCapture& capture, const Options& opt, std::string& opened_path);

const char* detector_source_label(DetectorSource source);
cv::Mat to_gray_preview(const cv::Mat& depth_mm, const Options& opt);
cv::Mat build_geometry_mask(const cv::Mat& depth_mm, const cv::Mat& confidence, const Options& opt);
cv::Mat build_wireframe_overlay(const cv::Mat& depth_mm, const cv::Mat& confidence, const Options& opt,
                                float& nearest_mm);
cv::Mat build_amplitude_detector_input(const SharedFrame& lidar_frame, const Options& opt);
cv::Mat build_confidence_detector_input(const SharedFrame& lidar_frame, const Options& opt);
cv::Mat build_pseudo_detector_input(const SharedFrame& lidar_frame, const Options& opt);
DetectionState run_person_detector(cv::dnn::Net& net, const SharedFrame& lidar_frame, const cv::Mat& detector_input,
                                   const Options& opt, float* best_person_score_out = nullptr);
DetectionState run_tof_person_detector(const SharedFrame& lidar_frame, const Options& opt,
                                       float* best_person_score_out = nullptr);

void draw_person_detection(cv::Mat& frame, const PersonDetection& detection, int index);
void draw_hud(cv::Mat& frame, const RuntimeStats& stats, const DetectionState& detections, bool detector_enabled,
              int scale);
cv::Mat compose_display_canvas(const cv::Mat& source);

} // namespace tactical_rescue
