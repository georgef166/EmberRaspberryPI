#include "tactical_rescue.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

namespace tactical_rescue {

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
    cv::Mat overlay(depth_mm.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    if (cv::countNonZero(geometry_mask) < 32) {
        nearest_mm = 0.0f;
        return overlay;
    }

    cv::Mat depth_gray = to_gray_preview(depth_mm, opt);
    depth_gray.setTo(0, geometry_mask == 0);

    cv::Mat depth_smooth;
    cv::GaussianBlur(depth_gray, depth_smooth, cv::Size(5, 5), 0.0);
    depth_smooth.setTo(0, geometry_mask == 0);

    cv::applyColorMap(depth_smooth, overlay, cv::COLORMAP_TURBO);
    overlay.setTo(cv::Scalar(0, 0, 0), geometry_mask == 0);

    cv::Mat confidence_u8;
    confidence.convertTo(confidence_u8, CV_8U, 255.0 / 1024.0);
    confidence_u8.setTo(0, geometry_mask == 0);

    cv::Mat brightness;
    confidence_u8.convertTo(brightness, CV_32F, 0.35 / 255.0, 0.65);
    for (int y = 0; y < overlay.rows; ++y) {
        cv::Vec3b* out_row = overlay.ptr<cv::Vec3b>(y);
        const float* gain_row = brightness.ptr<float>(y);
        for (int x = 0; x < overlay.cols; ++x) {
            const float gain = gain_row[x];
            out_row[x][0] = static_cast<uint8_t>(std::min(255.0f, out_row[x][0] * gain));
            out_row[x][1] = static_cast<uint8_t>(std::min(255.0f, out_row[x][1] * gain));
            out_row[x][2] = static_cast<uint8_t>(std::min(255.0f, out_row[x][2] * gain));
        }
    }

    cv::Mat edges;
    cv::Canny(depth_smooth, edges, 28.0, 84.0, 3, true);
    cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    overlay.setTo(cv::Scalar(255, 255, 255), edges > 0);

    nearest_mm = 0.0f;
    double min_depth = 0.0;
    cv::minMaxLoc(depth_mm, &min_depth, nullptr, nullptr, nullptr, geometry_mask);
    nearest_mm = static_cast<float>(min_depth);

    return overlay;
}

namespace {

std::pair<float, float> amplitude_percentiles(const cv::Mat& amplitude, const cv::Mat& valid_mask)
{
    std::vector<float> values;
    values.reserve(static_cast<size_t>(cv::countNonZero(valid_mask)));

    for (int y = 0; y < amplitude.rows; ++y) {
        const float* amp_row = amplitude.ptr<float>(y);
        const uint8_t* mask_row = valid_mask.ptr<uint8_t>(y);
        for (int x = 0; x < amplitude.cols; ++x) {
            if (!mask_row[x]) {
                continue;
            }
            values.push_back(amp_row[x]);
        }
    }

    if (values.size() < 8) {
        return {0.0f, 1.0f};
    }

    std::sort(values.begin(), values.end());
    const size_t low_index = static_cast<size_t>((values.size() - 1) * 0.05f);
    const size_t high_index = static_cast<size_t>((values.size() - 1) * 0.98f);
    const float low = values[low_index];
    const float high = values[high_index];
    return {low, std::max(high, low + 1.0f)};
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

    cv::morphologyEx(band, band, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
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

float estimate_detection_depth_mm(const cv::Mat& depth_roi, const cv::Mat& conf_roi, const cv::Mat& mask,
                                  const Options& opt)
{
    cv::Mat valid = (depth_roi > static_cast<float>(opt.min_depth_mm)) &
                    (depth_roi < static_cast<float>(opt.max_depth_mm)) &
                    (conf_roi >= static_cast<float>(opt.confidence_threshold));
    valid.convertTo(valid, CV_8U, 255.0);

    cv::Mat measure_mask;
    if (!mask.empty() && cv::countNonZero(mask) > 0) {
        cv::bitwise_and(valid, mask, measure_mask);
    } else {
        measure_mask = valid;
    }

    if (cv::countNonZero(measure_mask) == 0) {
        return 0.0f;
    }

    const cv::Scalar mean_depth = cv::mean(depth_roi, measure_mask);
    return static_cast<float>(mean_depth[0]);
}

} // namespace

const char* detector_source_label(DetectorSource source)
{
    switch (source) {
    case DetectorSource::RGB:
        return "RGB";
    case DetectorSource::AMPLITUDE:
        return "AMP";
    case DetectorSource::CONFIDENCE:
        return "CONF";
    case DetectorSource::PSEUDO:
        return "PSEUDO";
    case DetectorSource::AUTO:
    default:
        return "AUTO";
    }
}

cv::Mat build_amplitude_detector_input(const SharedFrame& lidar_frame, const Options& opt)
{
    if (lidar_frame.amplitude.empty() || lidar_frame.depth_mm.empty() || lidar_frame.confidence.empty()) {
        return {};
    }

    const int detector_confidence = std::max(8, opt.confidence_threshold / 2);
    cv::Mat valid = (lidar_frame.depth_mm > static_cast<float>(opt.min_depth_mm)) &
                    (lidar_frame.depth_mm < static_cast<float>(opt.max_depth_mm)) &
                    (lidar_frame.confidence >= static_cast<float>(detector_confidence)) &
                    (lidar_frame.amplitude > 0.0f);
    valid.convertTo(valid, CV_8U, 255.0);
    if (cv::countNonZero(valid) < 32) {
        return {};
    }

    const auto percentiles = amplitude_percentiles(lidar_frame.amplitude, valid);
    const float low = percentiles.first;
    const float high = percentiles.second;
    const float scale = 255.0f / std::max(1.0f, high - low);

    cv::Mat amplitude_shifted;
    lidar_frame.amplitude.convertTo(amplitude_shifted, CV_32F, 1.0, -low);
    cv::threshold(amplitude_shifted, amplitude_shifted, 0.0, 0.0, cv::THRESH_TOZERO);
    cv::threshold(amplitude_shifted, amplitude_shifted, high - low, high - low, cv::THRESH_TRUNC);

    cv::Mat amplitude_u8;
    amplitude_shifted.convertTo(amplitude_u8, CV_8U, scale);
    amplitude_u8.setTo(0, valid == 0);

    static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat contrast_u8;
    clahe->apply(amplitude_u8, contrast_u8);
    contrast_u8.setTo(0, valid == 0);

    cv::Mat detector_bgr;
    cv::cvtColor(contrast_u8, detector_bgr, cv::COLOR_GRAY2BGR);
    return detector_bgr;
}

cv::Mat build_confidence_detector_input(const SharedFrame& lidar_frame, const Options& opt)
{
    if (lidar_frame.confidence.empty() || lidar_frame.depth_mm.empty()) {
        return {};
    }

    cv::Mat valid = (lidar_frame.depth_mm > static_cast<float>(opt.min_depth_mm)) &
                    (lidar_frame.depth_mm < static_cast<float>(opt.max_depth_mm));
    valid.convertTo(valid, CV_8U, 255.0);
    if (cv::countNonZero(valid) < 32) {
        return {};
    }

    cv::Mat confidence_u8;
    cv::normalize(lidar_frame.confidence, confidence_u8, 0, 255, cv::NORM_MINMAX, CV_8U);
    confidence_u8.setTo(0, valid == 0);

    cv::Mat confidence_blur;
    cv::GaussianBlur(confidence_u8, confidence_blur, cv::Size(3, 3), 0.0);

    cv::Mat detector_bgr;
    cv::applyColorMap(confidence_blur, detector_bgr, cv::COLORMAP_INFERNO);
    detector_bgr.setTo(cv::Scalar(0, 0, 0), valid == 0);
    return detector_bgr;
}

cv::Mat build_pseudo_detector_input(const SharedFrame& lidar_frame, const Options& opt)
{
    if (lidar_frame.amplitude.empty() || lidar_frame.depth_mm.empty() || lidar_frame.confidence.empty()) {
        return {};
    }

    cv::Mat amplitude_bgr = build_amplitude_detector_input(lidar_frame, opt);
    if (amplitude_bgr.empty()) {
        return {};
    }

    cv::Mat valid = build_geometry_mask(lidar_frame.depth_mm, lidar_frame.confidence, opt);
    cv::Mat depth_gray = to_gray_preview(lidar_frame.depth_mm, opt);
    depth_gray.setTo(0, valid == 0);

    cv::Mat confidence_u8;
    lidar_frame.confidence.convertTo(confidence_u8, CV_8U, 1.0);
    confidence_u8.setTo(0, valid == 0);

    std::vector<cv::Mat> channels;
    cv::split(amplitude_bgr, channels);
    if (channels.size() != 3) {
        return amplitude_bgr;
    }

    channels[0] = depth_gray;
    channels[2] = confidence_u8;

    cv::Mat pseudo_bgr;
    cv::merge(channels, pseudo_bgr);
    return pseudo_bgr;
}

DetectionState run_person_detector(cv::dnn::Net& net, const SharedFrame& lidar_frame, const cv::Mat& detector_input,
                                   const Options& opt, float* best_person_score_out)
{
    DetectionState state;
    state.source_sequence = lidar_frame.sequence;
    if (detector_input.empty() || lidar_frame.depth_mm.empty()) {
        return state;
    }

    cv::Mat blob = cv::dnn::blobFromImage(detector_input, 1.0 / 127.5, cv::Size(opt.detector_input, opt.detector_input),
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

    float best_person_score = 0.0f;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    for (int i = 0; i < det.rows; ++i) {
        const float* row = det.ptr<float>(i);
        if (det.cols < 7) {
            continue;
        }

        const int class_id = static_cast<int>(std::round(row[1]));
        const float score = row[2];
        if (class_id == kCocoPersonClassId) {
            best_person_score = std::max(best_person_score, score);
        }
        if (class_id != kCocoPersonClassId || score < opt.person_confidence) {
            continue;
        }

        const int src_left = static_cast<int>(std::round(row[3] * detector_input.cols));
        const int src_top = static_cast<int>(std::round(row[4] * detector_input.rows));
        const int src_right = static_cast<int>(std::round(row[5] * detector_input.cols));
        const int src_bottom = static_cast<int>(std::round(row[6] * detector_input.rows));

        const float scale_x = static_cast<float>(lidar_frame.depth_mm.cols) / static_cast<float>(detector_input.cols);
        const float scale_y = static_cast<float>(lidar_frame.depth_mm.rows) / static_cast<float>(detector_input.rows);
        cv::Rect box(static_cast<int>(std::round(src_left * scale_x)), static_cast<int>(std::round(src_top * scale_y)),
                     static_cast<int>(std::round((src_right - src_left) * scale_x)),
                     static_cast<int>(std::round((src_bottom - src_top) * scale_y)));
        box &= cv::Rect(0, 0, lidar_frame.depth_mm.cols, lidar_frame.depth_mm.rows);
        if (box.area() <= 0) {
            continue;
        }

        boxes.push_back(box);
        confidences.push_back(score);
    }

    if (boxes.empty()) {
        if (best_person_score_out) {
            *best_person_score_out = best_person_score;
        }
        return state;
    }

    std::vector<int> kept;
    cv::dnn::NMSBoxes(boxes, confidences, opt.person_confidence, opt.nms_threshold, kept);
    if (kept.empty()) {
        if (best_person_score_out) {
            *best_person_score_out = best_person_score;
        }
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
        detection.mean_depth_mm = estimate_detection_depth_mm(depth_roi, conf_roi, detection.mask, opt);
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
    if (best_person_score_out) {
        *best_person_score_out = best_person_score;
    }
    return state;
}

DetectionState run_tof_person_detector(const SharedFrame& lidar_frame, const Options& opt, float* best_person_score_out)
{
    DetectionState state;
    state.source_sequence = lidar_frame.sequence;
    if (lidar_frame.depth_mm.empty() || lidar_frame.confidence.empty()) {
        return state;
    }

    cv::Mat valid = build_geometry_mask(lidar_frame.depth_mm, lidar_frame.confidence, opt);
    if (cv::countNonZero(valid) < 64) {
        if (best_person_score_out) {
            *best_person_score_out = 0.0f;
        }
        return state;
    }

    std::vector<float> depths;
    depths.reserve(static_cast<size_t>(cv::countNonZero(valid)));
    for (int y = 0; y < lidar_frame.depth_mm.rows; ++y) {
        const float* depth_row = lidar_frame.depth_mm.ptr<float>(y);
        const uint8_t* valid_row = valid.ptr<uint8_t>(y);
        for (int x = 0; x < lidar_frame.depth_mm.cols; ++x) {
            if (valid_row[x]) {
                depths.push_back(depth_row[x]);
            }
        }
    }
    if (depths.size() < 64) {
        if (best_person_score_out) {
            *best_person_score_out = 0.0f;
        }
        return state;
    }

    const size_t percentile_index = depths.size() / 5;
    std::nth_element(depths.begin(), depths.begin() + static_cast<long>(percentile_index), depths.end());
    const float near_anchor = depths[percentile_index];
    const float cutoff_depth = std::min(opt.max_depth_mm * 1.0f, near_anchor + 700.0f);

    cv::Mat candidate = (lidar_frame.depth_mm > static_cast<float>(opt.min_depth_mm)) &
                        (lidar_frame.depth_mm < cutoff_depth) &
                        (lidar_frame.confidence >= static_cast<float>(std::max(12, opt.confidence_threshold / 2)));
    candidate.convertTo(candidate, CV_8U, 255.0);
    cv::morphologyEx(candidate, candidate, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
    cv::morphologyEx(candidate, candidate, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

    cv::Mat labels, stats, centroids;
    const int components = cv::connectedComponentsWithStats(candidate, labels, stats, centroids, 8, CV_32S);
    float best_score = 0.0f;
    const int frame_area = lidar_frame.depth_mm.rows * lidar_frame.depth_mm.cols;

    for (int label = 1; label < components; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        const int x = stats.at<int>(label, cv::CC_STAT_LEFT);
        const int y = stats.at<int>(label, cv::CC_STAT_TOP);
        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        if (area < 140 || area > frame_area / 3 || width <= 0 || height <= 0) {
            continue;
        }

        const float aspect = static_cast<float>(width) / static_cast<float>(height);
        const float height_ratio = static_cast<float>(height) / static_cast<float>(lidar_frame.depth_mm.rows);
        const float fill_ratio = static_cast<float>(area) / static_cast<float>(width * height);
        if (height_ratio < 0.18f || aspect < 0.20f || aspect > 1.15f || fill_ratio < 0.18f) {
            continue;
        }

        const float center_x = static_cast<float>(centroids.at<double>(label, 0));
        const float center_bias =
            1.0f - std::min(1.0f, std::abs(center_x - lidar_frame.depth_mm.cols * 0.5f) / (lidar_frame.depth_mm.cols * 0.5f));
        const float aspect_score = 1.0f - std::min(1.0f, std::abs(aspect - 0.48f) / 0.50f);
        const float height_score = 1.0f - std::min(1.0f, std::abs(height_ratio - 0.38f) / 0.30f);
        const float fill_score = 1.0f - std::min(1.0f, std::abs(fill_ratio - 0.48f) / 0.40f);

        const cv::Rect box(x, y, width, height);
        const cv::Mat mask = labels == label;
        const cv::Mat mask_u8 = mask.clone();
        const float mean_depth = estimate_detection_depth_mm(lidar_frame.depth_mm(box), lidar_frame.confidence(box),
                                                             mask_u8(box), opt);
        const float depth_score = mean_depth > 0.0f
                                      ? 1.0f - std::min(1.0f, (mean_depth - opt.min_depth_mm) /
                                                                  std::max(1.0f, static_cast<float>(opt.max_depth_mm - opt.min_depth_mm)))
                                      : 0.0f;
        const float score =
            0.34f * aspect_score + 0.24f * height_score + 0.18f * fill_score + 0.14f * center_bias + 0.10f * depth_score;
        best_score = std::max(best_score, score);
        if (score < 0.38f) {
            continue;
        }

        PersonDetection detection;
        detection.box = box & cv::Rect(0, 0, lidar_frame.depth_mm.cols, lidar_frame.depth_mm.rows);
        detection.mask = cv::Mat(mask_u8, detection.box).clone();
        detection.confidence = clamp01(0.45f + score * 0.5f);
        detection.mean_depth_mm = mean_depth;
        detection.valid = true;
        state.people.push_back(std::move(detection));
    }

    std::sort(state.people.begin(), state.people.end(), [](const PersonDetection& lhs, const PersonDetection& rhs) {
        if (lhs.confidence != rhs.confidence) {
            return lhs.confidence > rhs.confidence;
        }
        if (lhs.mean_depth_mm > 0.0f && rhs.mean_depth_mm > 0.0f) {
            return lhs.mean_depth_mm < rhs.mean_depth_mm;
        }
        return lhs.box.area() > rhs.box.area();
    });
    if (static_cast<int>(state.people.size()) > opt.max_people) {
        state.people.resize(static_cast<size_t>(opt.max_people));
    }

    state.valid = !state.people.empty();
    if (best_person_score_out) {
        *best_person_score_out = state.valid ? state.people.front().confidence : best_score;
    }
    return state;
}

} // namespace tactical_rescue
