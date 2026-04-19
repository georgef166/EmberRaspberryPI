#include "tactical_rescue.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace tactical_rescue {

namespace {

cv::Scalar person_color(int index)
{
    (void)index;
    return cv::Scalar(0, 215, 255);
}

void fill_translucent_rect(cv::Mat& frame, const cv::Rect& rect, const cv::Scalar& color, double alpha)
{
    const cv::Rect clipped = rect & cv::Rect(0, 0, frame.cols, frame.rows);
    if (clipped.area() <= 0) {
        return;
    }

    cv::Mat roi = frame(clipped);
    cv::Mat tint(roi.size(), roi.type(), color);
    cv::addWeighted(tint, alpha, roi, 1.0 - alpha, 0.0, roi);
}

void draw_frame_corner(cv::Mat& frame, const cv::Point& corner, int horizontal_dir, int vertical_dir, int length,
                       int thickness, const cv::Scalar& color)
{
    cv::line(frame, corner, cv::Point(corner.x + horizontal_dir * length, corner.y), color, thickness, cv::LINE_AA);
    cv::line(frame, corner, cv::Point(corner.x, corner.y + vertical_dir * length), color, thickness, cv::LINE_AA);
}

void draw_panel_shell(cv::Mat& frame, const cv::Rect& rect, const cv::Scalar& border, double alpha, int corner_len)
{
    const cv::Rect clipped = rect & cv::Rect(0, 0, frame.cols, frame.rows);
    if (clipped.area() <= 0) {
        return;
    }

    fill_translucent_rect(frame, clipped, cv::Scalar(1, 8, 4), alpha);
    cv::rectangle(frame, clipped, border, 1, cv::LINE_AA);

    const cv::Point tl(clipped.x, clipped.y);
    const cv::Point tr(clipped.x + clipped.width - 1, clipped.y);
    const cv::Point br(clipped.x + clipped.width - 1, clipped.y + clipped.height - 1);
    const cv::Point bl(clipped.x, clipped.y + clipped.height - 1);

    draw_frame_corner(frame, tl, 1, 1, corner_len, 2, border);
    draw_frame_corner(frame, tr, -1, 1, corner_len, 2, border);
    draw_frame_corner(frame, br, -1, -1, corner_len, 2, border);
    draw_frame_corner(frame, bl, 1, -1, corner_len, 2, border);
}

void draw_reference_grid(cv::Mat& frame, double ui)
{
    const int major_x = std::max(72, static_cast<int>(std::round(96.0 * ui)));
    const int major_y = std::max(54, static_cast<int>(std::round(72.0 * ui)));
    const cv::Scalar vertical_color(6, 26, 18);
    const cv::Scalar horizontal_color(4, 18, 12);

    for (int x = 0; x < frame.cols; x += major_x) {
        cv::line(frame, cv::Point(x, 0), cv::Point(x, frame.rows), vertical_color, 1, cv::LINE_AA);
    }
    for (int y = 0; y < frame.rows; y += major_y) {
        cv::line(frame, cv::Point(0, y), cv::Point(frame.cols, y), horizontal_color, 1, cv::LINE_AA);
    }
}

void draw_status_box(cv::Mat& frame, const cv::Rect& rect, const std::string& label, const cv::Scalar& accent,
                     double ui)
{
    draw_panel_shell(frame, rect, accent, 0.42, std::max(12, static_cast<int>(std::round(18.0 * ui))));
    cv::putText(frame, label,
                cv::Point(rect.x + static_cast<int>(14 * ui), rect.y + rect.height / 2 + static_cast<int>(8 * ui)),
                cv::FONT_HERSHEY_SIMPLEX, 0.66 * ui, accent, 2, cv::LINE_AA);
}

void draw_mode_button(cv::Mat& frame, const cv::Rect& rect, const std::string& label, bool active,
                      const cv::Scalar& accent, double ui)
{
    const cv::Scalar border = active ? accent : cv::Scalar(42, 70, 52);
    fill_translucent_rect(frame, rect, active ? cv::Scalar(10, 54, 30) : cv::Scalar(2, 8, 5), active ? 0.72 : 0.55);
    cv::rectangle(frame, rect, border, 1, cv::LINE_AA);
    if (active) {
        draw_frame_corner(frame, rect.tl(), 1, 1, std::max(10, static_cast<int>(std::round(12.0 * ui))), 2, accent);
        draw_frame_corner(frame, cv::Point(rect.x + rect.width - 1, rect.y + rect.height - 1), -1, -1,
                          std::max(10, static_cast<int>(std::round(12.0 * ui))), 2, accent);
    }

    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.74 * ui, active ? 2 : 1, &baseline);
    const cv::Point text_origin(rect.x + (rect.width - text_size.width) / 2,
                                rect.y + (rect.height + text_size.height) / 2 - baseline / 2);
    cv::putText(frame, label, text_origin, cv::FONT_HERSHEY_SIMPLEX, 0.74 * ui,
                active ? accent : cv::Scalar(235, 235, 235), active ? 2 : 1, cv::LINE_AA);
}

void draw_target_reticle(cv::Mat& frame, const PersonDetection& detection, const cv::Scalar& accent, double ui)
{
    if (!detection.valid || detection.box.area() <= 0) {
        return;
    }

    const cv::Point center(detection.box.x + detection.box.width / 2, detection.box.y + detection.box.height / 2);
    const int radius = std::max(18, std::min(std::min(detection.box.width, detection.box.height) / 3,
                                             static_cast<int>(std::round(34.0 * ui))));
    const int gap = std::max(5, static_cast<int>(std::round(7.0 * ui)));
    const int arm = radius + std::max(6, static_cast<int>(std::round(10.0 * ui)));

    cv::circle(frame, center, radius, accent, 2, cv::LINE_AA);
    cv::line(frame, cv::Point(center.x - arm, center.y), cv::Point(center.x - gap, center.y), accent, 2, cv::LINE_AA);
    cv::line(frame, cv::Point(center.x + gap, center.y), cv::Point(center.x + arm, center.y), accent, 2, cv::LINE_AA);
    cv::line(frame, cv::Point(center.x, center.y - arm), cv::Point(center.x, center.y - gap), accent, 2, cv::LINE_AA);
    cv::line(frame, cv::Point(center.x, center.y + gap), cv::Point(center.x, center.y + arm), accent, 2, cv::LINE_AA);
    cv::circle(frame, center, std::max(3, static_cast<int>(std::round(4.0 * ui))), accent, cv::FILLED, cv::LINE_AA);
}

void draw_reference_heading(cv::Mat& frame, const cv::Rect& label_rect, const cv::Rect& compass_rect, int heading,
                            const cv::Scalar& accent, double ui)
{
    const int heading_clamped = std::max(75, std::min(150, heading));
    draw_panel_shell(frame, label_rect, accent, 0.46, std::max(12, static_cast<int>(std::round(18.0 * ui))));
    draw_panel_shell(frame, compass_rect, accent, 0.38, std::max(12, static_cast<int>(std::round(18.0 * ui))));

    cv::putText(frame, "TARGET HDR: " + std::to_string(heading_clamped) + "\u00B0",
                cv::Point(label_rect.x + static_cast<int>(16 * ui), label_rect.y + label_rect.height / 2 + static_cast<int>(8 * ui)),
                cv::FONT_HERSHEY_SIMPLEX, 0.72 * ui, accent, 2, cv::LINE_AA);

    const int line_y = compass_rect.y + compass_rect.height - static_cast<int>(20 * ui);
    const int line_left = compass_rect.x + static_cast<int>(16 * ui);
    const int line_right = compass_rect.x + compass_rect.width - static_cast<int>(16 * ui);
    cv::line(frame, cv::Point(line_left, line_y), cv::Point(line_right, line_y), accent, 1, cv::LINE_AA);

    const int tick_count = 16;
    for (int i = 0; i <= tick_count; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(tick_count);
        const int x = line_left + static_cast<int>(std::round((line_right - line_left) * t));
        const int tick_height = (i % 4 == 0) ? static_cast<int>(14 * ui) : static_cast<int>(8 * ui);
        cv::line(frame, cv::Point(x, line_y), cv::Point(x, line_y - tick_height), accent, 1, cv::LINE_AA);
    }

    struct CompassLabel {
        const char* text;
        float position;
    };
    const std::vector<CompassLabel> labels = {
        {"75", 0.00f},
        {"E", 0.18f},
        {"105", 0.36f},
        {"120", 0.52f},
        {"SE", 0.68f},
        {"150", 0.88f},
    };

    for (const auto& item : labels) {
        const int x = line_left + static_cast<int>(std::round((line_right - line_left) * item.position));
        cv::putText(frame, item.text, cv::Point(x - static_cast<int>(10 * ui), line_y - static_cast<int>(18 * ui)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.46 * ui, accent, 1, cv::LINE_AA);
    }

    const float heading_t = static_cast<float>(heading_clamped - 75) / 75.0f;
    const int pointer_x = line_left + static_cast<int>(std::round((line_right - line_left) * heading_t));
    cv::line(frame, cv::Point(pointer_x, line_y - static_cast<int>(26 * ui)), cv::Point(pointer_x, line_y), accent, 2,
             cv::LINE_AA);
    const cv::Point triangle[3] = {
        cv::Point(pointer_x - static_cast<int>(7 * ui), line_y - static_cast<int>(28 * ui)),
        cv::Point(pointer_x + static_cast<int>(7 * ui), line_y - static_cast<int>(28 * ui)),
        cv::Point(pointer_x, line_y - static_cast<int>(14 * ui)),
    };
    cv::fillConvexPoly(frame, triangle, 3, accent, cv::LINE_AA);
}

void draw_reference_vitals(cv::Mat& frame, const cv::Rect& rect, int heading, const cv::Scalar& accent, double ui)
{
    draw_panel_shell(frame, rect, accent, 0.52, std::max(12, static_cast<int>(std::round(18.0 * ui))));
    cv::putText(frame, "EMBER VITALS", cv::Point(rect.x + static_cast<int>(18 * ui), rect.y + static_cast<int>(38 * ui)),
                cv::FONT_HERSHEY_SIMPLEX, 0.78 * ui, accent, 2, cv::LINE_AA);

    struct VitalRow {
        std::string label;
        std::string value;
        std::string suffix;
    };
    const std::vector<VitalRow> rows = {
        {"PULSE (HR)", "--", "BPM"},
        {"OXYGEN (SPO2)", "--", "%"},
        {"SKIN TEMP", "--", "\u00B0C"},
        {"HEADING (AZIMUTH)", std::to_string(std::max(75, std::min(150, heading))), "\u00B0"},
    };

    int y = rect.y + static_cast<int>(84 * ui);
    for (size_t i = 0; i < rows.size(); ++i) {
        cv::putText(frame, rows[i].label, cv::Point(rect.x + static_cast<int>(18 * ui), y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.46 * ui, accent, 1, cv::LINE_AA);
        y += static_cast<int>(30 * ui);

        cv::putText(frame, rows[i].value, cv::Point(rect.x + static_cast<int>(18 * ui), y),
                    cv::FONT_HERSHEY_SIMPLEX, 1.18 * ui, accent, 2, cv::LINE_AA);
        cv::putText(frame, rows[i].suffix, cv::Point(rect.x + static_cast<int>(96 * ui), y - static_cast<int>(2 * ui)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.54 * ui, accent, 1, cv::LINE_AA);

        y += static_cast<int>(24 * ui);
        if (i + 1 != rows.size()) {
            cv::line(frame, cv::Point(rect.x + static_cast<int>(16 * ui), y),
                     cv::Point(rect.x + rect.width - static_cast<int>(16 * ui), y), cv::Scalar(26, 74, 46), 1,
                     cv::LINE_AA);
        }
        y += static_cast<int>(30 * ui);
    }
}

} // namespace

void draw_person_detection(cv::Mat& frame, const PersonDetection& detection, int index)
{
    if (!detection.valid || detection.box.area() <= 0) {
        return;
    }

    const cv::Scalar color = person_color(index);
    const int x = detection.box.x;
    const int y = detection.box.y;
    const int width = detection.box.width;
    const int height = detection.box.height;
    const int corner = std::max(4, std::min(std::min(width, height) / 4, 12));

    cv::line(frame, cv::Point(x, y + corner), cv::Point(x, y), color, 1, cv::LINE_AA);
    cv::line(frame, cv::Point(x, y), cv::Point(x + corner, y), color, 1, cv::LINE_AA);

    cv::line(frame, cv::Point(x + width - corner, y), cv::Point(x + width, y), color, 1, cv::LINE_AA);
    cv::line(frame, cv::Point(x + width, y), cv::Point(x + width, y + corner), color, 1, cv::LINE_AA);

    cv::line(frame, cv::Point(x + width, y + height - corner), cv::Point(x + width, y + height), color, 1,
             cv::LINE_AA);
    cv::line(frame, cv::Point(x + width, y + height), cv::Point(x + width - corner, y + height), color, 1,
             cv::LINE_AA);

    cv::line(frame, cv::Point(x + corner, y + height), cv::Point(x, y + height), color, 1, cv::LINE_AA);
    cv::line(frame, cv::Point(x, y + height), cv::Point(x, y + height - corner), color, 1, cv::LINE_AA);

    cv::rectangle(frame, detection.box, cv::Scalar(0, 90, 110), 1, cv::LINE_AA);

    const std::string label =
        "VICTIM [" + std::to_string(static_cast<int>(std::round(detection.confidence * 100))) + "%]";
    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
    const int label_height = text_size.height + 8;
    const int label_width = text_size.width + 8;
    const int label_x = std::max(0, std::min(x, frame.cols - label_width));
    const int label_y = std::max(0, y - label_height - 2);
    cv::rectangle(frame, cv::Rect(label_x, label_y, label_width, label_height), color, cv::FILLED);
    cv::putText(frame, label, cv::Point(label_x + 4, label_y + text_size.height + 2), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

    const int center_x = x + width / 2;
    const int center_y = y + height / 2;
    cv::line(frame, cv::Point(center_x - 4, center_y), cv::Point(center_x + 4, center_y), color, 1, cv::LINE_AA);
    cv::line(frame, cv::Point(center_x, center_y - 4), cv::Point(center_x, center_y + 4), color, 1, cv::LINE_AA);
}

void draw_hud(cv::Mat& frame, const RuntimeStats& stats, const DetectionState& detections, bool detector_enabled,
              int scale)
{
    const double ui = std::max(0.80, static_cast<double>(scale) / 3.0);
    const cv::Scalar accent(74, 255, 158);
    const cv::Scalar text_color(230, 255, 235);
    const cv::Scalar dim_text(145, 210, 165);

    draw_reference_grid(frame, ui);
    fill_translucent_rect(frame, cv::Rect(0, 0, frame.cols, static_cast<int>(56 * ui)), cv::Scalar(0, 12, 6), 0.50);
    fill_translucent_rect(frame, cv::Rect(0, frame.rows - static_cast<int>(44 * ui), frame.cols, static_cast<int>(44 * ui)),
                          cv::Scalar(0, 10, 5), 0.46);

    const int corner_len = std::max(18, static_cast<int>(std::round(28.0 * ui)));
    draw_frame_corner(frame, cv::Point(16, 16), 1, 1, corner_len, 3, accent);
    draw_frame_corner(frame, cv::Point(frame.cols - 17, 16), -1, 1, corner_len, 3, accent);
    draw_frame_corner(frame, cv::Point(frame.cols - 17, frame.rows - 17), -1, -1, corner_len, 3, accent);
    draw_frame_corner(frame, cv::Point(16, frame.rows - 17), 1, -1, corner_len, 3, accent);

    const PersonDetection* primary_detection = nullptr;
    if (detections.valid && !detections.people.empty() && detections.people.front().valid) {
        primary_detection = &detections.people.front();
        draw_target_reticle(frame, *primary_detection, accent, ui);
    }

    const std::string mode_text = detector_enabled
                                      ? (primary_detection ? "TRACK LOCK" : "SCANNING")
                                      : "DETECTOR OFF";
    const std::string info_text = "IGNISXR NAV  |  " + stats.detector_source_label + "  |  " + mode_text;
    const std::string perf_text = "FPS " + std::to_string(static_cast<int>(std::round(stats.render_fps))) +
                                  "  AI " + std::to_string(static_cast<int>(std::round(stats.inference_ms))) + "ms  RANGE " +
                                  std::to_string(static_cast<int>(std::round(stats.nearest_obstacle_mm))) + "mm";
    const std::string people_text = "HUMANS " + std::to_string(stats.detected_people) + "  BEST " +
                                    std::to_string(static_cast<int>(std::round(stats.best_person_score * 100))) + "%";

    cv::putText(frame, info_text, cv::Point(34, static_cast<int>(34 * ui)), cv::FONT_HERSHEY_SIMPLEX, 0.58 * ui,
                accent, 2, cv::LINE_AA);
    cv::putText(frame, perf_text, cv::Point(34, frame.rows - static_cast<int>(18 * ui)), cv::FONT_HERSHEY_SIMPLEX,
                0.50 * ui, text_color, 1, cv::LINE_AA);
    cv::putText(frame, people_text,
                cv::Point(frame.cols - static_cast<int>(260 * ui), frame.rows - static_cast<int>(18 * ui)),
                cv::FONT_HERSHEY_SIMPLEX, 0.50 * ui, dim_text, 1, cv::LINE_AA);
}

cv::Mat compose_display_canvas(const cv::Mat& source)
{
    cv::Mat canvas(kDisplayHeight, kDisplayWidth, CV_8UC3, cv::Scalar(1, 8, 4));
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
        cv::resize(source, resized, cv::Size(scaled_width, scaled_height), 0.0, 0.0, cv::INTER_LINEAR);
    }

    const int offset_x = (kDisplayWidth - scaled_width) / 2;
    const int offset_y = (kDisplayHeight - scaled_height) / 2;
    resized.copyTo(canvas(cv::Rect(offset_x, offset_y, scaled_width, scaled_height)));

    return canvas;
}

} // namespace tactical_rescue
