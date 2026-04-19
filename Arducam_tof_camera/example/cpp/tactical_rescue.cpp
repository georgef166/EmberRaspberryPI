#include "ArducamTOFCamera.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/ximgproc.hpp>

using namespace Arducam;

namespace {

constexpr int kDefaultRangeMm = 4000;
constexpr int kFrameTimeoutMs = 200;
constexpr float kMinDepthMm = 200.0f;
constexpr int kGlowThickness = 1;
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
    int max_people = 4;
    float person_confidence = 0.50f;
    float nms_threshold = 0.45f;
    bool no_preview = false;
    std::string model_path = kDefaultSsdModelPath;
    std::string config_path = kDefaultSsdConfigPath;
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
    bool detector_uses_segmentation = false;
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

bool parse_args(int argc, char* argv[], Options& opt)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << std::endl;
                std::exit(2);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: tactical_rescue [options]\n"
                << "  --device NUM             ToF camera device index\n"
                << "  --rgb-device NUM         RGB camera V4L2 device index\n"
                << "  --rgb-width NUM          RGB capture width\n"
                << "  --rgb-height NUM         RGB capture height\n"
                << "  --range MM               ToF range in mm\n"
                << "  --min-depth MM           Ignore geometry nearer than this\n"
                << "  --max-depth MM           Ignore geometry farther than this\n"
                << "  --confidence NUM         Depth confidence threshold\n"
                << "  --model PATH             TensorFlow COCO SSD .pb model path\n"
                << "  --config PATH            TensorFlow COCO SSD .pbtxt config path\n"
                << "  --person-conf FLOAT      Person detection confidence threshold (default 0.50)\n"
                << "  --nms FLOAT              NMS threshold\n"
                << "  --max-people NUM         Maximum rendered person detections\n"
                << "  --detector-input NUM     SSD input size\n"
                << "  --hud-scale NUM          HUD scale factor\n"
                << "  --no-preview             Run acquisition/inference without UI\n";
            return false;
        } else if (arg == "--device") {
            opt.device = std::atoi(require_value("--device"));
        } else if (arg == "--rgb-device") {
            opt.rgb_device = std::atoi(require_value("--rgb-device"));
        } else if (arg == "--rgb-width") {
            opt.rgb_width = std::max(160, std::atoi(require_value("--rgb-width")));
        } else if (arg == "--rgb-height") {
            opt.rgb_height = std::max(120, std::atoi(require_value("--rgb-height")));
        } else if (arg == "--range") {
            opt.range_mm = std::atoi(require_value("--range"));
            opt.max_depth_mm = opt.range_mm;
        } else if (arg == "--min-depth") {
            opt.min_depth_mm = std::atoi(require_value("--min-depth"));
        } else if (arg == "--max-depth") {
            opt.max_depth_mm = std::atoi(require_value("--max-depth"));
        } else if (arg == "--confidence") {
            opt.confidence_threshold = std::atoi(require_value("--confidence"));
        } else if (arg == "--model") {
            opt.model_path = require_value("--model");
        } else if (arg == "--config") {
            opt.config_path = require_value("--config");
        } else if (arg == "--person-conf") {
            opt.person_confidence = std::atof(require_value("--person-conf"));
        } else if (arg == "--nms") {
            opt.nms_threshold = std::atof(require_value("--nms"));
        } else if (arg == "--max-people") {
            opt.max_people = std::max(1, std::atoi(require_value("--max-people")));
        } else if (arg == "--detector-input") {
            opt.detector_input = std::max(160, std::atoi(require_value("--detector-input")));
        } else if (arg == "--hud-scale") {
            opt.hud_scale = std::max(1, std::atoi(require_value("--hud-scale")));
        } else if (arg == "--no-preview") {
            opt.no_preview = true;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return false;
        }
    }
    return true;
}

float clamp01(float value)
{
    return std::max(0.0f, std::min(1.0f, value));
}

cv::Mat to_gray_preview(const cv::Mat& depth_mm, const Options& opt)
{
    cv::Mat normalized(depth_mm.size(), CV_32F);
    const float span = std::max(1.0f, static_cast<float>(opt.max_depth_mm - opt.min_depth_mm));
    normalized = (depth_mm - static_cast<float>(opt.min_depth_mm)) / span;
    cv::threshold(normalized, normalized, 1.0, 1.0, cv::THRESH_TRUNC);
    cv::threshold(normalized, normalized, 0.0, 0.0, cv::THRESH_TOZERO);
    normalized = 1.0f - normalized;

    cv::Mat preview_u8;
    normalized.convertTo(preview_u8, CV_8U, 255.0);
    return preview_u8;
}

cv::Mat build_geometry_mask(const cv::Mat& depth_mm, const cv::Mat& confidence, const Options& opt)
{
    cv::Mat valid = (depth_mm > static_cast<float>(opt.min_depth_mm)) &
                    (depth_mm < static_cast<float>(opt.max_depth_mm)) &
                    (confidence >= static_cast<float>(opt.confidence_threshold));
    valid.convertTo(valid, CV_8U, 255.0);
    return valid;
}

cv::Mat build_wireframe_overlay(const cv::Mat& depth_mm, const cv::Mat& confidence, const Options& opt,
                                float& nearest_mm)
{
    const cv::Mat geometry_mask = build_geometry_mask(depth_mm, confidence, opt);

    cv::Mat depth_clean;
    depth_mm.copyTo(depth_clean);
    depth_clean.setTo(0.0f, geometry_mask == 0);

    cv::Mat depth_detail;
    cv::bilateralFilter(depth_clean, depth_detail, 3, 12.0, 12.0);

    cv::Mat grad_x, grad_y;
    cv::Scharr(depth_detail, grad_x, CV_32F, 1, 0);
    cv::Scharr(depth_detail, grad_y, CV_32F, 0, 1);

    cv::Mat magnitude;
    cv::magnitude(grad_x, grad_y, magnitude);
    magnitude.setTo(0.0f, geometry_mask == 0);

    cv::Mat laplacian;
    cv::Laplacian(depth_detail, laplacian, CV_32F, 3);
    laplacian = cv::abs(laplacian);
    laplacian.setTo(0.0f, geometry_mask == 0);

    double mag_max = 0.0, lap_max = 0.0;
    cv::minMaxLoc(magnitude, nullptr, &mag_max);
    cv::minMaxLoc(laplacian, nullptr, &lap_max);
    if (mag_max < 1.0) {
        mag_max = 1.0;
    }
    if (lap_max < 1.0) {
        lap_max = 1.0;
    }

    cv::Mat normalized_edges, normalized_detail;
    magnitude.convertTo(normalized_edges, CV_32F, 1.0 / mag_max);
    laplacian.convertTo(normalized_detail, CV_32F, 1.0 / lap_max);
    normalized_edges = normalized_edges * 0.90f + normalized_detail * 0.70f;
    cv::threshold(normalized_edges, normalized_edges, 1.0, 1.0, cv::THRESH_TRUNC);

    cv::Mat near_weight(depth_mm.size(), CV_32F);
    near_weight =
        1.0f - ((depth_mm - kMinDepthMm) / std::max(1.0f, static_cast<float>(opt.max_depth_mm) - kMinDepthMm));
    cv::threshold(near_weight, near_weight, 1.0, 1.0, cv::THRESH_TRUNC);
    cv::threshold(near_weight, near_weight, 0.0, 0.0, cv::THRESH_TOZERO);
    near_weight.setTo(0.0f, geometry_mask == 0);

    cv::Mat hazard_boost = near_weight.clone();
    hazard_boost.forEach<float>([](float& pixel, const int*) {
        pixel = std::pow(pixel, 1.6f);
    });

    cv::Mat mid_weight(depth_mm.size(), CV_32F, cv::Scalar(0.0f));
    for (int y = 0; y < depth_mm.rows; ++y) {
        const float* depth_row = depth_mm.ptr<float>(y);
        const uint8_t* mask_row = geometry_mask.ptr<uint8_t>(y);
        float* out_row = mid_weight.ptr<float>(y);
        for (int x = 0; x < depth_mm.cols; ++x) {
            if (!mask_row[x]) {
                continue;
            }
            const float norm =
                clamp01((depth_row[x] - static_cast<float>(opt.min_depth_mm)) /
                        std::max(1.0f, static_cast<float>(opt.max_depth_mm - opt.min_depth_mm)));
            out_row[x] = 1.0f - std::abs(norm - 0.52f) / 0.32f;
            out_row[x] = clamp01(out_row[x]);
        }
    }

    cv::Mat far_weight(depth_mm.size(), CV_32F, cv::Scalar(0.0f));
    for (int y = 0; y < depth_mm.rows; ++y) {
        const float* depth_row = depth_mm.ptr<float>(y);
        const uint8_t* mask_row = geometry_mask.ptr<uint8_t>(y);
        float* out_row = far_weight.ptr<float>(y);
        for (int x = 0; x < depth_mm.cols; ++x) {
            if (!mask_row[x]) {
                continue;
            }
            const float norm =
                clamp01((depth_row[x] - static_cast<float>(opt.min_depth_mm)) /
                        std::max(1.0f, static_cast<float>(opt.max_depth_mm - opt.min_depth_mm)));
            out_row[x] = std::pow(norm, 1.8f);
        }
    }

    cv::Mat composite = normalized_edges.mul(0.35f + hazard_boost * 1.45f);
    cv::threshold(composite, composite, 1.0, 1.0, cv::THRESH_TRUNC);

    cv::Mat binary_edges;
    composite.convertTo(binary_edges, CV_8U, 255.0);
    cv::threshold(binary_edges, binary_edges, 34, 255, cv::THRESH_BINARY);
    cv::morphologyEx(binary_edges, binary_edges, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)));
    cv::ximgproc::thinning(binary_edges, binary_edges, cv::ximgproc::THINNING_ZHANGSUEN);

    cv::Mat overlay(depth_mm.size(), CV_8UC3, cv::Scalar(0, 0, 0));

    cv::Mat near_mask_f, mid_mask_f, far_mask_f;
    near_weight.convertTo(near_mask_f, CV_32F, 1.0);
    mid_weight.convertTo(mid_mask_f, CV_32F, 1.0);
    far_weight.convertTo(far_mask_f, CV_32F, 1.0);
    cv::threshold(near_mask_f, near_mask_f, 0.60, 1.0, cv::THRESH_BINARY);
    cv::threshold(mid_mask_f, mid_mask_f, 0.45, 1.0, cv::THRESH_BINARY);
    cv::threshold(far_mask_f, far_mask_f, 0.72, 1.0, cv::THRESH_BINARY);

    cv::Mat near_mask, mid_mask, far_mask;
    near_mask_f.convertTo(near_mask, CV_8U, 255.0);
    mid_mask_f.convertTo(mid_mask, CV_8U, 255.0);
    far_mask_f.convertTo(far_mask, CV_8U, 255.0);

    cv::morphologyEx(near_mask, near_mask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    cv::morphologyEx(mid_mask, mid_mask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
    cv::morphologyEx(far_mask, far_mask, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)));

    overlay.setTo(cv::Scalar(18, 18, 18), far_mask);
    overlay.setTo(cv::Scalar(34, 34, 34), mid_mask);
    overlay.setTo(cv::Scalar(54, 54, 54), near_mask);

    cv::Mat surface_lines;
    cv::adaptiveThreshold(to_gray_preview(depth_mm, opt), surface_lines, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                          cv::THRESH_BINARY_INV, 11, 2);
    cv::bitwise_and(surface_lines, geometry_mask, surface_lines);
    cv::threshold(surface_lines, surface_lines, 0, 255, cv::THRESH_BINARY);
    cv::ximgproc::thinning(surface_lines, surface_lines, cv::ximgproc::THINNING_ZHANGSUEN);

    for (int y = 0; y < binary_edges.rows; ++y) {
        const uint8_t* edge_row = binary_edges.ptr<uint8_t>(y);
        const uint8_t* surface_row = surface_lines.ptr<uint8_t>(y);
        const float* near_row = hazard_boost.ptr<float>(y);
        cv::Vec3b* out_row = overlay.ptr<cv::Vec3b>(y);
        for (int x = 0; x < binary_edges.cols; ++x) {
            if (surface_row[x] && !edge_row[x]) {
                const uint8_t detail_value = static_cast<uint8_t>(95 + 60 * clamp01(mid_weight.at<float>(y, x)));
                out_row[x] = cv::Vec3b(detail_value, detail_value, detail_value);
            }
            if (edge_row[x]) {
                const float w = clamp01(near_row[x]);
                const uint8_t edge_value = static_cast<uint8_t>(215 + 40 * w);
                out_row[x] = cv::Vec3b(edge_value, edge_value, edge_value);
            }
        }
    }

    nearest_mm = 0.0f;
    double min_depth = 0.0;
    cv::minMaxLoc(depth_mm, &min_depth, nullptr, nullptr, nullptr, geometry_mask);
    nearest_mm = static_cast<float>(min_depth);

    return overlay;
}

cv::Mat build_amplitude_gate(const cv::Mat& amplitude_roi, const cv::Mat& valid)
{
    if (amplitude_roi.empty() || cv::countNonZero(valid) == 0) {
        return cv::Mat(valid.size(), CV_8U, cv::Scalar(255));
    }

    cv::Scalar mean_amp, std_amp;
    cv::meanStdDev(amplitude_roi, mean_amp, std_amp, valid);
    const float threshold = std::max(2.0f, static_cast<float>(mean_amp[0] - std_amp[0] * 1.2));

    cv::Mat amp_gate = amplitude_roi >= threshold;
    amp_gate.convertTo(amp_gate, CV_8U, 255.0);
    return amp_gate;
}

cv::Mat pick_best_component(const cv::Mat& binary_mask, const cv::Point2f& center_hint)
{
    if (binary_mask.empty() || cv::countNonZero(binary_mask) == 0) {
        return {};
    }

    cv::Mat labels, stats, centroids;
    const int components = cv::connectedComponentsWithStats(binary_mask, labels, stats, centroids, 8, CV_32S);
    if (components <= 1) {
        return binary_mask.clone();
    }

    int best_label = 1;
    double best_score = -1e18;
    for (int label = 1; label < components; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area <= 0) {
            continue;
        }
        const double cx = centroids.at<double>(label, 0);
        const double cy = centroids.at<double>(label, 1);
        const double dx = cx - center_hint.x;
        const double dy = cy - center_hint.y;
        const double score = area - 0.8 * std::sqrt(dx * dx + dy * dy);
        if (score > best_score) {
            best_score = score;
            best_label = label;
        }
    }

    cv::Mat mask = labels == best_label;
    mask.convertTo(mask, CV_8U, 255.0);
    return mask;
}

cv::Mat refine_mask_with_depth(const cv::Mat& seed_mask, const cv::Mat& depth_roi, const cv::Mat& conf_roi,
                               const cv::Mat& amp_roi, const Options& opt)
{
    if (depth_roi.empty() || conf_roi.empty()) {
        return {};
    }

    cv::Mat valid = (depth_roi > static_cast<float>(opt.min_depth_mm)) &
                    (depth_roi < static_cast<float>(opt.max_depth_mm)) &
                    (conf_roi >= static_cast<float>(opt.confidence_threshold));
    valid.convertTo(valid, CV_8U, 255.0);
    if (cv::countNonZero(valid) == 0) {
        return {};
    }

    cv::Mat working_seed;
    if (seed_mask.empty()) {
        working_seed = valid.clone();
    } else {
        seed_mask.convertTo(working_seed, CV_8U, 255.0);
        cv::bitwise_and(working_seed, valid, working_seed);
        if (cv::countNonZero(working_seed) == 0) {
            working_seed = valid.clone();
        }
    }

    cv::Mat amp_gate = build_amplitude_gate(amp_roi, valid);
    cv::bitwise_and(working_seed, amp_gate, working_seed);
    if (cv::countNonZero(working_seed) == 0) {
        working_seed = valid.clone();
    }

    cv::Scalar mean_depth, std_depth;
    cv::meanStdDev(depth_roi, mean_depth, std_depth, working_seed);
    const float center = static_cast<float>(mean_depth[0]);
    const float spread = std::max(120.0f, static_cast<float>(std_depth[0]) * 1.7f);

    cv::Mat band;
    cv::inRange(depth_roi, center - spread, center + spread, band);
    cv::bitwise_and(band, valid, band);
    cv::bitwise_and(band, amp_gate, band);
    if (cv::countNonZero(band) == 0) {
        band = working_seed;
    }

    cv::morphologyEx(band, band, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    cv::dilate(band, band, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

    const cv::Point2f center_hint(depth_roi.cols * 0.5f, depth_roi.rows * 0.5f);
    return pick_best_component(band, center_hint);
}

cv::Mat detection_mask_from_box(const cv::Mat& depth_mm, const cv::Mat& confidence, const cv::Mat& amplitude,
                                const cv::Rect& box, const Options& opt)
{
    const cv::Rect bounds = box & cv::Rect(0, 0, depth_mm.cols, depth_mm.rows);
    if (bounds.area() <= 0) {
        return {};
    }

    const cv::Mat depth_roi = depth_mm(bounds);
    const cv::Mat conf_roi = confidence(bounds);
    const cv::Mat amp_roi = amplitude.empty() ? cv::Mat() : amplitude(bounds);
    return refine_mask_with_depth({}, depth_roi, conf_roi, amp_roi, opt);
}

bool file_exists(const std::string& path)
{
    std::ifstream file(path);
    return file.good();
}

bool open_rgb_capture(cv::VideoCapture& capture, const Options& opt, std::string& opened_path)
{
    capture.open(opt.rgb_device, cv::CAP_V4L2);
    if (capture.isOpened()) {
        capture.set(cv::CAP_PROP_FRAME_WIDTH, opt.rgb_width);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, opt.rgb_height);
        capture.set(cv::CAP_PROP_BUFFERSIZE, 1);
        opened_path = "/dev/video" + std::to_string(opt.rgb_device);
        return true;
    }

    const std::string pipeline =
        "libcamerasrc ! video/x-raw,width=" + std::to_string(opt.rgb_width) + ",height=" +
        std::to_string(opt.rgb_height) + ",framerate=30/1 ! videoconvert ! appsink drop=1 sync=false";
    capture.open(pipeline, cv::CAP_GSTREAMER);
    if (capture.isOpened()) {
        opened_path = "libcamerasrc";
        return true;
    }

    return false;
}

DetectionState run_person_detector(cv::dnn::Net& net, const SharedFrame& lidar_frame, const SharedRgbFrame& rgb_frame,
                                   const Options& opt)
{
    DetectionState state;
    state.source_sequence = lidar_frame.sequence;
    if (rgb_frame.bgr.empty() || lidar_frame.depth_mm.empty()) {
        return state;
    }

    cv::Mat blob = cv::dnn::blobFromImage(rgb_frame.bgr, 1.0 / 127.5, cv::Size(opt.detector_input, opt.detector_input),
                                          cv::Scalar(127.5, 127.5, 127.5), true, false);
    net.setInput(blob);

    cv::Mat detections = net.forward();
    if (detections.empty()) {
        return state;
    }

    cv::Mat det;
    if (detections.dims == 4 && detections.size[3] == 7) {
        det = detections.reshape(1, detections.size[2]);
    } else if (detections.cols == 7) {
        det = detections;
    }
    if (det.empty()) {
        return state;
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    for (int i = 0; i < det.rows; ++i) {
        const float* row = det.ptr<float>(i);
        if (det.cols < 7) {
            continue;
        }

        const int class_id = static_cast<int>(std::round(row[1]));
        const float score = row[2];
        if (class_id != kCocoPersonClassId || score < opt.person_confidence) {
            continue;
        }

        const int rgb_left = static_cast<int>(std::round(row[3] * rgb_frame.bgr.cols));
        const int rgb_top = static_cast<int>(std::round(row[4] * rgb_frame.bgr.rows));
        const int rgb_right = static_cast<int>(std::round(row[5] * rgb_frame.bgr.cols));
        const int rgb_bottom = static_cast<int>(std::round(row[6] * rgb_frame.bgr.rows));

        const float scale_x = static_cast<float>(lidar_frame.depth_mm.cols) / static_cast<float>(rgb_frame.bgr.cols);
        const float scale_y = static_cast<float>(lidar_frame.depth_mm.rows) / static_cast<float>(rgb_frame.bgr.rows);
        cv::Rect box(static_cast<int>(std::round(rgb_left * scale_x)), static_cast<int>(std::round(rgb_top * scale_y)),
                     static_cast<int>(std::round((rgb_right - rgb_left) * scale_x)),
                     static_cast<int>(std::round((rgb_bottom - rgb_top) * scale_y)));
        box &= cv::Rect(0, 0, lidar_frame.depth_mm.cols, lidar_frame.depth_mm.rows);
        if (box.area() <= 0) {
            continue;
        }

        boxes.push_back(box);
        confidences.push_back(score);
    }

    if (boxes.empty()) {
        return state;
    }

    std::vector<int> kept;
    cv::dnn::NMSBoxes(boxes, confidences, opt.person_confidence, opt.nms_threshold, kept);
    if (kept.empty()) {
        return state;
    }

    std::sort(kept.begin(), kept.end(), [&](int lhs, int rhs) {
        return confidences[lhs] > confidences[rhs];
    });

    for (int idx : kept) {
        if (static_cast<int>(state.people.size()) >= opt.max_people) {
            break;
        }

        PersonDetection detection;
        detection.box = boxes[idx];
        detection.confidence = confidences[idx];
        detection.from_segmentation = false;

        const cv::Mat depth_roi = lidar_frame.depth_mm(detection.box);
        const cv::Mat conf_roi = lidar_frame.confidence(detection.box);
        const cv::Mat amp_roi = lidar_frame.amplitude.empty() ? cv::Mat() : lidar_frame.amplitude(detection.box);
        detection.mask = refine_mask_with_depth({}, depth_roi, conf_roi, amp_roi, opt);
        if (detection.mask.empty()) {
            detection.mask =
                detection_mask_from_box(lidar_frame.depth_mm, lidar_frame.confidence, lidar_frame.amplitude, detection.box, opt);
        }
        if (detection.mask.empty() || cv::countNonZero(detection.mask) == 0) {
            continue;
        }

        const cv::Scalar mean_depth = cv::mean(depth_roi, detection.mask);
        detection.mean_depth_mm = static_cast<float>(mean_depth[0]);
        detection.valid = true;
        state.people.push_back(std::move(detection));
    }

    std::sort(state.people.begin(), state.people.end(), [](const PersonDetection& lhs, const PersonDetection& rhs) {
        if (lhs.mean_depth_mm > 0.0f && rhs.mean_depth_mm > 0.0f && lhs.mean_depth_mm != rhs.mean_depth_mm) {
            return lhs.mean_depth_mm < rhs.mean_depth_mm;
        }
        return lhs.confidence > rhs.confidence;
    });

    state.valid = !state.people.empty();
    return state;
}

cv::Scalar person_color(int index)
{
    (void)index;
    return cv::Scalar(0, 215, 255);
}

void draw_person_detection(cv::Mat& frame, const PersonDetection& detection, int index)
{
    if (!detection.valid || detection.box.area() <= 0) {
        return;
    }

    const cv::Scalar color = person_color(index);
    cv::rectangle(frame, detection.box, color, 2, cv::LINE_AA);

    const std::string label =
        "HUMAN " + std::to_string(static_cast<int>(std::round(detection.confidence * 100))) + "%";
    const int baseline_pad = 6;
    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
    const int text_x = detection.box.x;
    const int text_y = std::max(text_size.height + baseline_pad, detection.box.y - 8);
    cv::rectangle(frame,
                  cv::Rect(text_x - 2, text_y - text_size.height - 4, text_size.width + 6, text_size.height + 8),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::rectangle(frame,
                  cv::Rect(text_x - 2, text_y - text_size.height - 4, text_size.width + 6, text_size.height + 8),
                  color, 1, cv::LINE_AA);
    cv::putText(frame, label, cv::Point(text_x + 1, text_y), cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv::LINE_AA);
}

void draw_hud(cv::Mat& frame, const RuntimeStats& stats, const DetectionState& detections, bool detector_enabled,
              int scale)
{
    const int font = cv::FONT_HERSHEY_SIMPLEX;
    const double font_scale = 0.35 * scale;
    const int thickness = std::max(1, scale - 1);
    const cv::Scalar text_color(220, 255, 220);
    const cv::Scalar accent(0, 220, 255);

    cv::rectangle(frame, cv::Rect(0, 0, frame.cols, 22 * scale), cv::Scalar(10, 20, 10), cv::FILLED);
    cv::putText(frame, "IGNISXR TACTICAL RESCUE", cv::Point(8 * scale, 16 * scale), font, font_scale, accent,
                thickness, cv::LINE_AA);

    std::string det_text = "DETECTOR OFF";
    if (detector_enabled) {
        det_text = detections.valid ? "SCAN LOCK x" + std::to_string(static_cast<int>(detections.people.size()))
                                    : "SCAN ACTIVE";
        det_text += " SSD";
    }

    const std::string fps_text =
        "CAP " + std::to_string(static_cast<int>(std::round(stats.capture_fps))) + " FPS  RENDER " +
        std::to_string(static_cast<int>(std::round(stats.render_fps))) + " FPS";
    const std::string latency_text =
        "AI " + std::to_string(static_cast<int>(std::round(stats.inference_ms))) + " ms  AGE " +
        std::to_string(static_cast<int>(std::round(stats.frame_age_ms))) + " ms";
    const std::string range_text =
        "NEAREST " + std::to_string(static_cast<int>(std::round(stats.nearest_obstacle_mm))) + " mm";
    const std::string people_text = "PEOPLE " + std::to_string(stats.detected_people);
    const std::string vitals_text = "VITALS N/A";

    const int bottom = frame.rows - 10 * scale;
    cv::putText(frame, fps_text, cv::Point(8 * scale, bottom - 28 * scale), font, font_scale, text_color, thickness,
                cv::LINE_AA);
    cv::putText(frame, latency_text, cv::Point(8 * scale, bottom - 16 * scale), font, font_scale, text_color,
                thickness, cv::LINE_AA);
    cv::putText(frame, range_text, cv::Point(8 * scale, bottom - 4 * scale), font, font_scale, text_color, thickness,
                cv::LINE_AA);
    cv::putText(frame, people_text, cv::Point(frame.cols - 70 * scale, bottom - 28 * scale), font, font_scale,
                text_color, thickness, cv::LINE_AA);
    cv::putText(frame, vitals_text, cv::Point(frame.cols - 70 * scale, bottom - 16 * scale), font, font_scale,
                text_color, thickness, cv::LINE_AA);
    cv::putText(frame, det_text, cv::Point(frame.cols - 70 * scale, 16 * scale), font, font_scale, accent, thickness,
                cv::LINE_AA);

    cv::line(frame, cv::Point(frame.cols / 2 - 8 * scale, frame.rows / 2),
             cv::Point(frame.cols / 2 + 8 * scale, frame.rows / 2), accent, 1, cv::LINE_AA);
    cv::line(frame, cv::Point(frame.cols / 2, frame.rows / 2 - 8 * scale),
             cv::Point(frame.cols / 2, frame.rows / 2 + 8 * scale), accent, 1, cv::LINE_AA);
}

cv::Mat compose_display_canvas(const cv::Mat& source)
{
    cv::Mat canvas(kDisplayHeight, kDisplayWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    if (source.empty()) {
        return canvas;
    }

    const int scale_x = kDisplayWidth / source.cols;
    const int scale_y = kDisplayHeight / source.rows;
    const int integer_scale = std::max(1, std::min(std::min(scale_x, scale_y), kMaxDisplayScale));
    const int scaled_width = source.cols * integer_scale;
    const int scaled_height = source.rows * integer_scale;

    cv::Mat resized;
    if (integer_scale == 1) {
        resized = source;
    } else {
        cv::resize(source, resized, cv::Size(scaled_width, scaled_height), 0.0, 0.0, cv::INTER_NEAREST);
    }

    const int offset_x = (kDisplayWidth - scaled_width) / 2;
    const int offset_y = (kDisplayHeight - scaled_height) / 2;
    resized.copyTo(canvas(cv::Rect(offset_x, offset_y, scaled_width, scaled_height)));

    return canvas;
}

} // namespace

int main(int argc, char* argv[])
{
    Options options;
    if (!parse_args(argc, argv, options)) {
        return 0;
    }
    if (!file_exists(options.model_path) || !file_exists(options.config_path)) {
        std::cerr << "TensorFlow COCO SSD files missing.\n"
                  << "Expected model: " << options.model_path << "\n"
                  << "Expected config: " << options.config_path << std::endl;
        return -1;
    }

    ArducamTOFCamera tof;
    if (tof.open(Connection::CSI, options.device)) {
        std::cerr << "Failed to open camera" << std::endl;
        return -1;
    }
    if (tof.start(FrameType::DEPTH_FRAME)) {
        std::cerr << "Failed to start camera" << std::endl;
        return -1;
    }

    tof.setControl(Control::RANGE, options.range_mm);
    int actual_range = options.range_mm;
    tof.getControl(Control::RANGE, &actual_range);
    options.max_depth_mm = std::min(options.max_depth_mm, actual_range);

    auto info = tof.getCameraInfo();
    std::cout << "Tactical rescue feed active at " << info.width << "x" << info.height << " range " << actual_range
              << "mm" << std::endl;

    cv::VideoCapture rgb_capture;
    std::string rgb_source;
    if (!open_rgb_capture(rgb_capture, options, rgb_source)) {
        std::cerr << "Failed to open RGB camera from /dev/video" << options.rgb_device
                  << " or Raspberry Pi libcamera pipeline" << std::endl;
        return -1;
    }
    std::cout << "RGB detector feed using " << rgb_source << std::endl;

    std::mutex frame_mutex;
    std::condition_variable frame_cv;
    SharedFrame latest_frame;
    std::mutex rgb_mutex;
    std::condition_variable rgb_cv;
    SharedRgbFrame latest_rgb_frame;
    DetectionState latest_detection;
    std::mutex detection_mutex;
    RuntimeStats stats;
    std::mutex stats_mutex;
    std::atomic<bool> running{true};
    std::atomic<uint64_t> published_sequence{0};
    std::atomic<uint64_t> published_rgb_sequence{0};
    std::atomic<bool> detector_ready{false};

    cv::dnn::Net detector;
    try {
        detector = cv::dnn::readNetFromTensorflow(options.model_path, options.config_path);
        detector.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        detector.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        detector_ready = true;
    } catch (const cv::Exception& e) {
        std::cerr << "Failed to load TensorFlow COCO SSD: " << e.what() << std::endl;
        return -1;
    }

    FpsCounter capture_fps;
    FpsCounter render_fps;

    std::thread capture_thread([&] {
        uint64_t sequence = 0;
        while (running) {
            ArducamFrameBuffer* frame = tof.requestFrame(kFrameTimeoutMs);
            if (!frame) {
                continue;
            }

            float* depth_ptr = reinterpret_cast<float*>(frame->getData(FrameType::DEPTH_FRAME));
            float* conf_ptr = reinterpret_cast<float*>(frame->getData(FrameType::CONFIDENCE_FRAME));
            float* amp_ptr = reinterpret_cast<float*>(frame->getData(FrameType::AMPLITUDE_FRAME));

            FrameFormat format;
            frame->getFormat(FrameType::DEPTH_FRAME, format);

            if (depth_ptr && conf_ptr) {
                SharedFrame local;
                local.depth_mm = cv::Mat(format.height, format.width, CV_32F, depth_ptr).clone();
                local.confidence = cv::Mat(format.height, format.width, CV_32F, conf_ptr).clone();
                if (amp_ptr) {
                    local.amplitude = cv::Mat(format.height, format.width, CV_32F, amp_ptr).clone();
                }
                local.sequence = ++sequence;
                local.captured_at = std::chrono::steady_clock::now();

                {
                    std::lock_guard<std::mutex> lock(frame_mutex);
                    latest_frame = std::move(local);
                }
                published_sequence = sequence;
                frame_cv.notify_all();

                capture_fps.tick();
                std::lock_guard<std::mutex> lock(stats_mutex);
                stats.capture_fps = capture_fps.value();
            }

            tof.releaseFrame(frame);
        }
    });

    std::thread rgb_capture_thread([&] {
        uint64_t sequence = 0;
        cv::Mat bgr;
        while (running) {
            if (!rgb_capture.read(bgr) || bgr.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            SharedRgbFrame local;
            local.bgr = bgr.clone();
            local.sequence = ++sequence;
            local.captured_at = std::chrono::steady_clock::now();
            {
                std::lock_guard<std::mutex> lock(rgb_mutex);
                latest_rgb_frame = std::move(local);
            }
            published_rgb_sequence = sequence;
            rgb_cv.notify_all();
        }
    });

    std::thread inference_thread([&] {
        if (!detector_ready) {
            return;
        }

        uint64_t consumed_rgb = 0;
        while (running) {
            SharedFrame lidar_input;
            SharedRgbFrame rgb_input;
            {
                std::unique_lock<std::mutex> lock(rgb_mutex);
                rgb_cv.wait(lock, [&] { return !running || published_rgb_sequence.load() != consumed_rgb; });
                if (!running) {
                    break;
                }
                rgb_input = latest_rgb_frame;
            }
            {
                std::lock_guard<std::mutex> lock(frame_mutex);
                lidar_input = latest_frame;
            }

            if (lidar_input.depth_mm.empty() || rgb_input.bgr.empty() || rgb_input.sequence == consumed_rgb) {
                continue;
            }
            consumed_rgb = rgb_input.sequence;

            const auto infer_started = std::chrono::steady_clock::now();
            DetectionState result = run_person_detector(detector, lidar_input, rgb_input, options);

            {
                std::lock_guard<std::mutex> lock(detection_mutex);
                latest_detection = result;
            }

            const auto infer_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - infer_started).count();
            std::lock_guard<std::mutex> lock(stats_mutex);
            stats.inference_ms = infer_ms;
            stats.detected_people = static_cast<int>(result.people.size());
            stats.detector_uses_segmentation = false;
        }
    });

    if (!options.no_preview) {
        cv::namedWindow("tactical_rescue", cv::WINDOW_NORMAL);
        cv::resizeWindow("tactical_rescue", kDisplayWidth, kDisplayHeight);
    }

    while (running) {
        SharedFrame frame;
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            frame = latest_frame;
        }

        if (frame.depth_mm.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        float nearest_mm = 0.0f;
        cv::Mat hud = build_wireframe_overlay(frame.depth_mm, frame.confidence, options, nearest_mm);

        DetectionState detections;
        {
            std::lock_guard<std::mutex> lock(detection_mutex);
            detections = latest_detection;
        }
        if (detections.valid && detections.source_sequence <= frame.sequence) {
            const int render_count = std::min(static_cast<int>(detections.people.size()), kMaxRenderedPeople);
            for (int i = 0; i < render_count; ++i) {
                draw_person_detection(hud, detections.people[i], i);
            }
        }

        render_fps.tick();
        RuntimeStats local_stats;
        {
            std::lock_guard<std::mutex> lock(stats_mutex);
            stats.render_fps = render_fps.value();
            stats.nearest_obstacle_mm = nearest_mm;
            stats.frame_age_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - frame.captured_at).count();
            local_stats = stats;
        }
        cv::Mat display = compose_display_canvas(hud);
        draw_hud(display, local_stats, detections, detector_ready, options.hud_scale);

        if (!options.no_preview) {
            cv::imshow("tactical_rescue", display);
            const int key = cv::waitKey(1);
            if (key == 27 || key == 'q') {
                running = false;
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    running = false;
    frame_cv.notify_all();
    rgb_cv.notify_all();

    if (capture_thread.joinable()) {
        capture_thread.join();
    }
    if (rgb_capture_thread.joinable()) {
        rgb_capture_thread.join();
    }
    if (inference_thread.joinable()) {
        inference_thread.join();
    }

    rgb_capture.release();

    if (tof.stop()) {
        std::cerr << "Failed to stop camera" << std::endl;
    }
    if (tof.close()) {
        std::cerr << "Failed to close camera" << std::endl;
    }

    return 0;
}
