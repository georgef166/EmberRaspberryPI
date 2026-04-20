#include "tactical_rescue.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

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
    cv::GaussianBlur(depth_gray, depth_smooth, cv::Size(3, 3), 0.0);
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

struct TemporalTrack {
    cv::Rect2f box;
    float confidence = 0.0f;
    float mean_depth_mm = 0.0f;
    int hits = 0;
    int misses = 0;
    uint64_t last_sequence = 0;
};

float rect_iou(const cv::Rect2f& lhs, const cv::Rect& rhs)
{
    const cv::Rect2f rhs_f(static_cast<float>(rhs.x), static_cast<float>(rhs.y), static_cast<float>(rhs.width),
                           static_cast<float>(rhs.height));
    const cv::Rect2f intersection = lhs & rhs_f;
    if (intersection.width <= 0.0f || intersection.height <= 0.0f) {
        return 0.0f;
    }
    const float intersection_area = intersection.width * intersection.height;
    const float union_area = lhs.area() + rhs_f.area() - intersection_area;
    if (union_area <= 0.0f) {
        return 0.0f;
    }
    return intersection_area / union_area;
}

cv::Rect blend_rect(const cv::Rect2f& previous, const cv::Rect& current)
{
    const float mix = 0.72f;
    const float inv_mix = 1.0f - mix;
    return cv::Rect(static_cast<int>(std::round(previous.x * inv_mix + current.x * mix)),
                    static_cast<int>(std::round(previous.y * inv_mix + current.y * mix)),
                    std::max(1, static_cast<int>(std::round(previous.width * inv_mix + current.width * mix))),
                    std::max(1, static_cast<int>(std::round(previous.height * inv_mix + current.height * mix))));
}

float average_band_width(const cv::Mat& mask, float start_ratio, float end_ratio)
{
    if (mask.empty()) {
        return 0.0f;
    }

    const int start_row = std::max(0, static_cast<int>(std::floor(mask.rows * start_ratio)));
    const int end_row = std::min(mask.rows, std::max(start_row + 1, static_cast<int>(std::ceil(mask.rows * end_ratio))));
    float sum = 0.0f;
    int contributing_rows = 0;
    for (int y = start_row; y < end_row; ++y) {
        const int row_fill = cv::countNonZero(mask.row(y));
        if (row_fill <= 0) {
            continue;
        }
        sum += static_cast<float>(row_fill);
        ++contributing_rows;
    }
    return contributing_rows > 0 ? sum / static_cast<float>(contributing_rows) : 0.0f;
}

float vertical_coverage_ratio(const cv::Mat& mask)
{
    if (mask.empty()) {
        return 0.0f;
    }

    int occupied_rows = 0;
    for (int y = 0; y < mask.rows; ++y) {
        if (cv::countNonZero(mask.row(y)) > 0) {
            ++occupied_rows;
        }
    }
    return static_cast<float>(occupied_rows) / static_cast<float>(std::max(1, mask.rows));
}

float normalized_center_distance(const cv::Rect& box, const cv::Size& frame_size)
{
    const float cx = static_cast<float>(box.x + box.width / 2);
    const float cy = static_cast<float>(box.y + box.height / 2);
    const float dx = cx - frame_size.width * 0.5f;
    const float dy = cy - frame_size.height * 0.55f;
    const float radius = std::sqrt(dx * dx + dy * dy);
    const float max_radius = std::sqrt(frame_size.width * frame_size.width + frame_size.height * frame_size.height) * 0.65f;
    return clamp01(1.0f - radius / std::max(1.0f, max_radius));
}

void stabilize_temporal_detections(std::vector<PersonDetection>& detections, uint64_t sequence, const Options& opt,
                                   float& best_confidence)
{
    static std::vector<TemporalTrack> tracks;

    for (auto& track : tracks) {
        ++track.misses;
    }

    std::vector<bool> track_claimed(tracks.size(), false);
    std::vector<PersonDetection> synthetic_detections;
    synthetic_detections.reserve(2);

    for (auto& detection : detections) {
        int best_track = -1;
        float best_match_score = 0.0f;
        for (size_t i = 0; i < tracks.size(); ++i) {
            if (track_claimed[i]) {
                continue;
            }
            const float iou = rect_iou(tracks[i].box, detection.box);
            const float depth_delta = (tracks[i].mean_depth_mm > 0.0f && detection.mean_depth_mm > 0.0f)
                                          ? std::abs(tracks[i].mean_depth_mm - detection.mean_depth_mm)
                                          : 0.0f;
            const float depth_penalty = clamp01(depth_delta / 900.0f) * 0.15f;
            const float match_score = iou - depth_penalty;
            if (match_score > best_match_score) {
                best_match_score = match_score;
                best_track = static_cast<int>(i);
            }
        }

        if (best_track >= 0 && best_match_score > 0.10f) {
            auto& track = tracks[best_track];
            track_claimed[static_cast<size_t>(best_track)] = true;
            track.box = cv::Rect2f(blend_rect(track.box, detection.box));
            track.confidence = clamp01(track.confidence * 0.55f + detection.confidence * 0.45f);
            track.mean_depth_mm = detection.mean_depth_mm > 0.0f
                                      ? (track.mean_depth_mm > 0.0f ? track.mean_depth_mm * 0.45f + detection.mean_depth_mm * 0.55f
                                                                    : detection.mean_depth_mm)
                                      : track.mean_depth_mm;
            track.hits += 1;
            track.misses = 0;
            track.last_sequence = sequence;

            detection.box = cv::Rect(track.box);
            detection.mean_depth_mm = track.mean_depth_mm;
            detection.confidence = clamp01(std::max(detection.confidence, track.confidence) +
                                           std::min(0.16f, 0.04f * static_cast<float>(track.hits - 1)));
        } else {
            TemporalTrack track;
            track.box = cv::Rect2f(detection.box);
            track.confidence = detection.confidence;
            track.mean_depth_mm = detection.mean_depth_mm;
            track.hits = 1;
            track.misses = 0;
            track.last_sequence = sequence;
            tracks.push_back(track);
            track_claimed.push_back(true);
            detection.confidence = clamp01(detection.confidence + 0.03f);
        }

        best_confidence = std::max(best_confidence, detection.confidence);
    }

    for (const auto& track : tracks) {
        if (track.misses == 1 && track.hits >= 3 && track.confidence >= opt.person_confidence + 0.08f) {
            PersonDetection held;
            held.box = cv::Rect(track.box);
            held.confidence = clamp01(track.confidence * 0.92f);
            held.mean_depth_mm = track.mean_depth_mm;
            held.valid = true;
            best_confidence = std::max(best_confidence, held.confidence);
            synthetic_detections.push_back(std::move(held));
        }
    }

    detections.insert(detections.end(), synthetic_detections.begin(), synthetic_detections.end());
    tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                                [](const TemporalTrack& track) { return track.misses > 3 || track.confidence < 0.08f; }),
                 tracks.end());
}

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

DetectionState run_tof_person_detector(const SharedFrame& lidar_frame, DetectorSource source, const Options& opt,
                                       float* best_person_score_out)
{
    DetectionState state;
    state.source_sequence = lidar_frame.sequence;
    float best_confidence = 0.0f;
    if (lidar_frame.depth_mm.empty() || lidar_frame.confidence.empty()) {
        return state;
    }

    const cv::Size reduced_size(std::max(80, lidar_frame.depth_mm.cols / 2), std::max(60, lidar_frame.depth_mm.rows / 2));
    cv::Mat reduced_depth;
    cv::Mat reduced_confidence;
    cv::resize(lidar_frame.depth_mm, reduced_depth, reduced_size, 0.0, 0.0, cv::INTER_AREA);
    cv::resize(lidar_frame.confidence, reduced_confidence, reduced_size, 0.0, 0.0, cv::INTER_AREA);

    cv::Mat valid = build_geometry_mask(reduced_depth, reduced_confidence, opt);
    if (cv::countNonZero(valid) < 64) {
        if (best_person_score_out) {
            *best_person_score_out = 0.0f;
        }
        return state;
    }

    static cv::Mat previous_depth;
    static cv::Mat previous_valid;
    cv::Mat motion_mask;
    if (!previous_depth.empty() && previous_depth.size() == reduced_size && !previous_valid.empty()) {
        cv::Mat depth_delta;
        cv::absdiff(reduced_depth, previous_depth, depth_delta);
        cv::threshold(depth_delta, motion_mask, 70.0, 255.0, cv::THRESH_BINARY);
        motion_mask.convertTo(motion_mask, CV_8U);
        cv::bitwise_and(motion_mask, valid, motion_mask);
        cv::bitwise_and(motion_mask, previous_valid, motion_mask);
    }

    std::vector<float> depths;
    depths.reserve(static_cast<size_t>(cv::countNonZero(valid)));
    for (int y = 0; y < reduced_depth.rows; ++y) {
        const float* depth_row = reduced_depth.ptr<float>(y);
        const uint8_t* valid_row = valid.ptr<uint8_t>(y);
        for (int x = 0; x < reduced_depth.cols; ++x) {
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
    const float cutoff_depth = std::min(static_cast<float>(opt.max_depth_mm), near_anchor + 700.0f);
    const float confidence_floor = static_cast<float>(source == DetectorSource::CONFIDENCE
                                                          ? std::max(18, opt.confidence_threshold)
                                                          : std::max(12, opt.confidence_threshold / 2));

    cv::Mat candidate = (reduced_depth > static_cast<float>(opt.min_depth_mm)) &
                        (reduced_depth < cutoff_depth) &
                        (reduced_confidence >= confidence_floor);
    if (source == DetectorSource::AMPLITUDE && !lidar_frame.amplitude.empty()) {
        cv::Mat reduced_amplitude;
        cv::resize(lidar_frame.amplitude, reduced_amplitude, reduced_size, 0.0, 0.0, cv::INTER_AREA);
        cv::Mat amplitude_gate = reduced_amplitude > 4.0f;
        amplitude_gate.convertTo(amplitude_gate, CV_8U, 255.0);
        candidate.convertTo(candidate, CV_8U, 255.0);
        cv::bitwise_and(candidate, amplitude_gate, candidate);
    } else {
        candidate.convertTo(candidate, CV_8U, 255.0);
    }
    cv::morphologyEx(candidate, candidate, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
    cv::morphologyEx(candidate, candidate, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

    cv::Mat labels, stats, centroids;
    const int components = cv::connectedComponentsWithStats(candidate, labels, stats, centroids, 8, CV_32S);
    const int frame_area = reduced_depth.rows * reduced_depth.cols;

    for (int label = 1; label < components; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        const int x = stats.at<int>(label, cv::CC_STAT_LEFT);
        const int y = stats.at<int>(label, cv::CC_STAT_TOP);
        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        if (area < 32 || area > frame_area / 3 || width <= 0 || height <= 0) {
            continue;
        }

        const float aspect = static_cast<float>(width) / static_cast<float>(height);
        const float height_ratio = static_cast<float>(height) / static_cast<float>(reduced_depth.rows);
        const float fill_ratio = static_cast<float>(area) / static_cast<float>(width * height);
        if (height_ratio < 0.15f || aspect < 0.18f || aspect > 1.20f || fill_ratio < 0.14f) {
            continue;
        }

        const cv::Rect reduced_box(x, y, width, height);
        cv::Mat component_mask = labels(reduced_box) == label;
        component_mask.convertTo(component_mask, CV_8U, 255.0);

        const float center_bias = normalized_center_distance(reduced_box, reduced_depth.size());
        const float aspect_score = clamp01(1.0f - std::abs(aspect - 0.46f) / 0.55f);
        const float height_score = clamp01(1.0f - std::abs(height_ratio - 0.34f) / 0.28f);
        const float fill_score = clamp01(1.0f - std::abs(fill_ratio - 0.42f) / 0.34f);
        const float coverage = vertical_coverage_ratio(component_mask);
        const float continuity_score = clamp01((coverage - 0.42f) / 0.45f);

        const float top_width = average_band_width(component_mask, 0.08f, 0.30f);
        const float mid_width = average_band_width(component_mask, 0.34f, 0.68f);
        const float bottom_width = average_band_width(component_mask, 0.72f, 0.96f);
        const float shoulder_ratio = mid_width / std::max(1.0f, top_width);
        const float shoulder_score = clamp01(1.0f - std::abs(shoulder_ratio - 1.35f) / 1.00f);
        const float support_ratio = bottom_width / std::max(1.0f, mid_width);
        const float support_score = clamp01(1.0f - std::abs(support_ratio - 0.95f) / 0.90f);

        cv::Scalar mean_depth_reduced, std_depth_reduced;
        cv::meanStdDev(reduced_depth(reduced_box), mean_depth_reduced, std_depth_reduced, component_mask);
        const float depth_consistency_score =
            clamp01(1.0f - static_cast<float>(std_depth_reduced[0]) / 420.0f);

        float motion_score = 0.35f;
        if (!motion_mask.empty()) {
            cv::Mat motion_component;
            cv::bitwise_and(motion_mask(reduced_box), component_mask, motion_component);
            const int moving_pixels = cv::countNonZero(motion_component);
            const float motion_ratio = static_cast<float>(moving_pixels) / static_cast<float>(std::max(1, area));
            motion_score = clamp01(0.35f + motion_ratio * 2.3f);
        }

        const bool touches_edge = (x <= 1) || (y <= 1) || (x + width >= reduced_depth.cols - 1) ||
                                  (y + height >= reduced_depth.rows - 1);
        const float edge_score = touches_edge ? 0.45f : 1.0f;

        const float scale_x = static_cast<float>(lidar_frame.depth_mm.cols) / static_cast<float>(reduced_depth.cols);
        const float scale_y = static_cast<float>(lidar_frame.depth_mm.rows) / static_cast<float>(reduced_depth.rows);
        cv::Rect box(static_cast<int>(std::round(x * scale_x)), static_cast<int>(std::round(y * scale_y)),
                     std::max(1, static_cast<int>(std::round(width * scale_x))),
                     std::max(1, static_cast<int>(std::round(height * scale_y))));
        box &= cv::Rect(0, 0, lidar_frame.depth_mm.cols, lidar_frame.depth_mm.rows);
        if (box.area() <= 0) {
            continue;
        }
        const float mean_depth = estimate_detection_depth_mm(lidar_frame.depth_mm(box), lidar_frame.confidence(box), {}, opt);
        const float depth_score = mean_depth > 0.0f
                                      ? 1.0f - std::min(1.0f, (mean_depth - opt.min_depth_mm) /
                                                                  std::max(1.0f, static_cast<float>(opt.max_depth_mm - opt.min_depth_mm)))
                                      : 0.0f;
        const float score = 0.20f * aspect_score + 0.14f * height_score + 0.10f * fill_score +
                            0.14f * continuity_score + 0.12f * shoulder_score + 0.10f * support_score +
                            0.10f * depth_consistency_score + 0.04f * center_bias + 0.03f * depth_score +
                            0.03f * motion_score + 0.00f * edge_score;
        const float confidence = clamp01((score - 0.20f) / 0.45f);
        best_confidence = std::max(best_confidence, confidence);
        if (score < 0.28f || confidence < 0.18f) {
            continue;
        }

        PersonDetection detection;
        detection.box = box;
        detection.confidence = confidence;
        detection.mean_depth_mm = mean_depth;
        detection.valid = true;
        state.people.push_back(std::move(detection));
    }

    previous_depth = reduced_depth.clone();
    previous_valid = valid.clone();

    stabilize_temporal_detections(state.people, lidar_frame.sequence, opt, best_confidence);

    state.people.erase(std::remove_if(state.people.begin(), state.people.end(),
                                      [&](const PersonDetection& detection) {
                                          return !detection.valid || detection.confidence < opt.person_confidence;
                                      }),
                       state.people.end());

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
        *best_person_score_out = state.valid ? state.people.front().confidence : best_confidence;
    }
    return state;
}

} // namespace tactical_rescue
