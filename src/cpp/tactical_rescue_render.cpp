#include "tactical_rescue.hpp"
#include "tactical_rescue_stream.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace tactical_rescue {

namespace {

// Defined further down; the shared panel component below needs it.
void fill_translucent_rect(cv::Mat& frame, const cv::Rect& rect, const cv::Scalar& color, double alpha);

cv::Scalar person_color(int index)
{
    (void)index;
    return cv::Scalar(0, 215, 255);
}

// --- Unified HUD typography -------------------------------------------------
// Every string on the HUD goes through draw_text() so font, weight, and the
// legibility outline stay identical across panels, labels, and commander markup.
// Thickness used to be hardcoded to 1 or 2 regardless of size, which made small
// text look blobby and inconsistent; it is now derived from the rendered scale.
constexpr int kHudFont = cv::FONT_HERSHEY_DUPLEX;

constexpr double kTextBrand = 0.86; // top-left brand mark
constexpr double kTextValue = 1.02; // large numeric readouts
constexpr double kTextTitle = 0.50; // panel headers
constexpr double kTextBody = 0.50;  // status lines
constexpr double kTextLabel = 0.44;  // small field labels
constexpr double kTextBanner = 0.90; // fire / hotspot warning

int text_thickness(double fs, bool bold)
{
    return bold ? std::max(2, static_cast<int>(std::round(fs * 2.0)))
                : std::max(1, static_cast<int>(std::round(fs * 1.35)));
}

cv::Size measure_text(const std::string& text, double size, double ui, bool bold = false)
{
    const double fs = size * ui;
    int baseline = 0;
    return cv::getTextSize(text, kHudFont, fs, text_thickness(fs, bold), &baseline);
}

// Dark outline behind every glyph so text stays readable over bright thermal
// blobs and dense wireframe geometry.
void draw_text(cv::Mat& frame, const std::string& text, const cv::Point& origin, double size,
               const cv::Scalar& color, double ui, bool bold = false)
{
    const double fs = size * ui;
    const int thickness = text_thickness(fs, bold);
    cv::putText(frame, text, origin, kHudFont, fs, cv::Scalar(0, 0, 0), thickness + 2, cv::LINE_AA);
    cv::putText(frame, text, origin, kHudFont, fs, color, thickness, cv::LINE_AA);
}

// --- Shared HUD panel -------------------------------------------------------
// Every corner block on the HUD is built from this one component, so the four
// corners and the sensor readouts are literally the same thing rather than four
// hand-rolled layouts. Structure follows the sensor panel: accent title, dim
// uppercase labels, large light values, optional dim footer line.
struct PanelField {
    std::string label;
    std::string value;
};

struct PanelLayout {
    cv::Size size;
    std::vector<int> col_x;
    int pad = 0;
    int title_baseline = 0;
    int label_baseline = 0; // 0 when no field carries a label
    int value_baseline = 0;
    int footer_baseline = 0; // 0 when there is no footer
};

PanelLayout layout_panel(const std::string& title, const std::vector<PanelField>& fields, const std::string& footer,
                         double ui)
{
    PanelLayout m;
    m.pad = static_cast<int>(std::round(15 * ui));
    const int col_gap = static_cast<int>(std::round(32 * ui));

    const cv::Size title_size = measure_text(title, kTextTitle, ui, true);
    const cv::Size label_probe = measure_text("HG", kTextLabel, ui);
    const cv::Size value_probe = measure_text("0", kTextValue, ui, true);
    const cv::Size footer_size = footer.empty() ? cv::Size(0, 0) : measure_text(footer, kTextLabel, ui);

    bool any_label = false;
    for (const auto& f : fields) {
        if (!f.label.empty()) {
            any_label = true;
        }
    }

    int x = m.pad;
    int content_w = 0;
    for (size_t i = 0; i < fields.size(); ++i) {
        const int lw = fields[i].label.empty() ? 0 : measure_text(fields[i].label, kTextLabel, ui).width;
        const int vw = measure_text(fields[i].value, kTextValue, ui, true).width;
        const int cw = std::max(lw, vw);
        m.col_x.push_back(x);
        x += cw + col_gap;
        content_w += cw + (i + 1 < fields.size() ? col_gap : 0);
    }

    int y = m.pad + title_size.height;
    m.title_baseline = y;
    if (any_label) {
        y += static_cast<int>(std::round(20 * ui)) + label_probe.height;
        m.label_baseline = y;
    }
    if (!fields.empty()) {
        y += static_cast<int>(std::round(15 * ui)) + value_probe.height;
        m.value_baseline = y;
    }
    if (!footer.empty()) {
        y += static_cast<int>(std::round(18 * ui)) + footer_size.height;
        m.footer_baseline = y;
    }

    const int inner = std::max(std::max(content_w, title_size.width), footer_size.width);
    m.size = cv::Size(inner + m.pad * 2, y + m.pad);
    return m;
}

// Draws the panel with its top-left at `origin`. Callers anchor corners by
// offsetting with the size returned from layout_panel().
void draw_panel(cv::Mat& frame, const cv::Point& origin, const std::string& title,
                const std::vector<PanelField>& fields, const std::string& footer, const cv::Scalar& accent,
                const cv::Scalar& value_color, const cv::Scalar& dim_color, double ui)
{
    const PanelLayout m = layout_panel(title, fields, footer, ui);
    const cv::Rect rect(origin.x, origin.y, m.size.width, m.size.height);
    fill_translucent_rect(frame, rect, cv::Scalar(0, 0, 0), 0.58);
    cv::rectangle(frame, rect, accent, 1, cv::LINE_AA);

    draw_text(frame, title, cv::Point(rect.x + m.pad, rect.y + m.title_baseline), kTextTitle, accent, ui, true);
    for (size_t i = 0; i < fields.size() && i < m.col_x.size(); ++i) {
        const int cx = rect.x + m.col_x[i];
        if (m.label_baseline && !fields[i].label.empty()) {
            draw_text(frame, fields[i].label, cv::Point(cx, rect.y + m.label_baseline), kTextLabel, dim_color, ui);
        }
        draw_text(frame, fields[i].value, cv::Point(cx, rect.y + m.value_baseline), kTextValue, value_color, ui, true);
    }
    if (m.footer_baseline) {
        draw_text(frame, footer, cv::Point(rect.x + m.pad, rect.y + m.footer_baseline), kTextLabel, dim_color, ui);
    }
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

// Thin adapter over compute_canvas_transform(). Both names exist because the
// commander-console work and the AR-grid work arrived at this same letterbox
// math independently. Keeping a single implementation is not tidiness: if the
// two copies ever drift, the victim boxes and the ground grid would disagree
// about where a given floor pixel lands on screen.
void display_canvas_transform(const cv::Size& source, double& scale, cv::Point& offset)
{
    const CanvasTransform transform = compute_canvas_transform(source);
    scale = transform.scale;
    offset = cv::Point(transform.offset_x, transform.offset_y);
}

void draw_person_detection(cv::Mat& frame, const PersonDetection& detection, int index, double scale,
                           const cv::Point& offset)
{
    if (!detection.valid || detection.box.area() <= 0) {
        return;
    }

    const cv::Scalar color = person_color(index);
    // Box arrives in ToF-frame coordinates; map it into whatever frame we are
    // drawing on so the outline and label render at that frame's resolution.
    const int x = static_cast<int>(std::round(detection.box.x * scale)) + offset.x;
    const int y = static_cast<int>(std::round(detection.box.y * scale)) + offset.y;
    const int width = static_cast<int>(std::round(detection.box.width * scale));
    const int height = static_cast<int>(std::round(detection.box.height * scale));
    // Stroke weights scale with the target frame, not the sensor frame.
    const double s = std::max(1.0, frame.cols / 1280.0);
    const int stroke = std::max(1, static_cast<int>(std::round(2.0 * s)));
    const int corner = std::max(static_cast<int>(std::round(14 * s)),
                                std::min(std::min(width, height) / 5, static_cast<int>(std::round(46 * s))));

    // Corner brackets only — a full box hides too much of the geometry.
    cv::line(frame, cv::Point(x, y + corner), cv::Point(x, y), color, stroke, cv::LINE_AA);
    cv::line(frame, cv::Point(x, y), cv::Point(x + corner, y), color, stroke, cv::LINE_AA);

    cv::line(frame, cv::Point(x + width - corner, y), cv::Point(x + width, y), color, stroke, cv::LINE_AA);
    cv::line(frame, cv::Point(x + width, y), cv::Point(x + width, y + corner), color, stroke, cv::LINE_AA);

    cv::line(frame, cv::Point(x + width, y + height - corner), cv::Point(x + width, y + height), color, stroke,
             cv::LINE_AA);
    cv::line(frame, cv::Point(x + width, y + height), cv::Point(x + width - corner, y + height), color, stroke,
             cv::LINE_AA);

    cv::line(frame, cv::Point(x + corner, y + height), cv::Point(x, y + height), color, stroke, cv::LINE_AA);
    cv::line(frame, cv::Point(x, y + height), cv::Point(x, y + height - corner), color, stroke, cv::LINE_AA);

    cv::rectangle(frame, cv::Rect(x, y, width, height), cv::Scalar(30, 120, 145), 1, cv::LINE_AA);

    const std::string label =
        "VICTIM " + std::to_string(static_cast<int>(std::round(detection.confidence * 100))) + "%";
    // Solid chip, so this is the one place that skips the outline — black on
    // amber is already maximum contrast. Font/weight come from the shared
    // typography constants, sized to the frame being drawn on.
    const cv::Size text_size = measure_text(label, kTextLabel, s, true);
    const int pad = static_cast<int>(std::round(6 * s));
    const int label_height = text_size.height + pad * 2;
    const int label_width = text_size.width + pad * 2;
    const int label_x = std::max(0, std::min(x, frame.cols - label_width));
    // Prefer above the box; drop inside the top edge when there is no room.
    const int label_y = (y - label_height - static_cast<int>(std::round(3 * s)) >= 0)
                            ? y - label_height - static_cast<int>(std::round(3 * s))
                            : std::min(y, frame.rows - label_height);
    cv::rectangle(frame, cv::Rect(label_x, std::max(0, label_y), label_width, label_height), color, cv::FILLED);
    cv::putText(frame, label, cv::Point(label_x + pad, std::max(0, label_y) + pad + text_size.height), kHudFont,
                kTextLabel * s, cv::Scalar(12, 12, 12), text_thickness(kTextLabel * s, true), cv::LINE_AA);

    const int center_x = x + width / 2;
    const int center_y = y + height / 2;
    const int tick = static_cast<int>(std::round(7 * s));
    cv::line(frame, cv::Point(center_x - tick, center_y), cv::Point(center_x + tick, center_y), color, stroke,
             cv::LINE_AA);
    cv::line(frame, cv::Point(center_x, center_y - tick), cv::Point(center_x, center_y + tick), color, stroke,
             cv::LINE_AA);
}

void draw_thermal_overlay(cv::Mat& frame, const ThermalFrame& thermal, const Options& opt)
{
    if (!thermal.valid || thermal.temperature_c.empty() || frame.empty()) {
        return;
    }

    // Map temperature to a fixed [warm-body floor, fire threshold] display range
    // so the colours stay stable frame-to-frame. Cold structure (below the floor)
    // is left untinted so the wireframe geometry still reads through.
    const float lo = opt.victim_temp_min_c;
    const float hi = std::max(opt.fire_temp_c, lo + 1.0f);
    const float scale = 1.0f / (hi - lo);

    cv::Mat norm;
    thermal.temperature_c.convertTo(norm, CV_32F, scale, -lo * scale);
    cv::threshold(norm, norm, 1.0, 1.0, cv::THRESH_TRUNC);  // clamp high
    cv::threshold(norm, norm, 0.0, 0.0, cv::THRESH_TOZERO); // clamp low
    cv::Mat norm8;
    norm.convertTo(norm8, CV_8U, 255.0);

    cv::Mat heat_small;
    cv::applyColorMap(norm8, heat_small, cv::COLORMAP_INFERNO);
    const cv::Mat warm_small = norm8 > 0; // only tint above the warm-body floor

    cv::Mat heat, mask;
    cv::resize(heat_small, heat, frame.size(), 0, 0, cv::INTER_LINEAR);
    cv::resize(warm_small, mask, frame.size(), 0, 0, cv::INTER_NEAREST);

    cv::Mat blended;
    cv::addWeighted(heat, opt.thermal_overlay_alpha, frame, 1.0 - opt.thermal_overlay_alpha, 0.0, blended);
    blended.copyTo(frame, mask);
}

void draw_hud(cv::Mat& frame, const RuntimeStats& stats, const DetectionState& detections, bool detector_enabled,
              int scale)
{
    const double ui = std::max(0.80, static_cast<double>(scale) / 3.0);
    const cv::Scalar accent(74, 255, 158);
    const cv::Scalar text_color(230, 255, 235);
    const cv::Scalar dim_text(145, 210, 165);

    const PersonDetection* primary_detection = nullptr;
    if (detections.valid && !detections.people.empty() && detections.people.front().valid) {
        primary_detection = &detections.people.front();
    }

    const std::string mode_text = detector_enabled
                                      ? (primary_detection ? "TRACK LOCK" : "SCANNING")
                                      : "DETECTOR OFF";

    const int margin = static_cast<int>(std::round(22 * ui));
    const int gap = static_cast<int>(std::round(12 * ui));

    // --- Top-left: identity. Same panel grammar as the sensor readouts, with
    // the wordmark occupying the "value" slot.
    {
        const std::vector<PanelField> fields = {{"", "Ember"}};
        draw_panel(frame, cv::Point(margin, margin), "IGNISXR", fields, "", accent, text_color, dim_text, ui);
    }

    // --- Top-right: ambient, then thermal stacked beneath it.
    int right_y = margin;
    if (stats.ambient_enabled) {
        std::vector<PanelField> fields;
        std::string footer = "STATUS " + stats.ambient_status;
        if (stats.ambient_valid) {
            fields.push_back({"TEMP", cv::format("%.1fC", stats.ambient_temperature_c)});
            fields.push_back({"HUMIDITY", cv::format("%.1f%%", stats.ambient_humidity_percent)});
            // ASCII only — the Hershey fonts have no glyph for U+00B7 and render
            // it as "??".
            footer += cv::format("   /   AGE %.1fs", stats.ambient_age_s);
        } else {
            // No reading yet: title + status only, rather than a placeholder value.
            footer = "WAITING FOR SENSOR";
        }
        const PanelLayout m = layout_panel("AM2302", fields, footer, ui);
        draw_panel(frame, cv::Point(frame.cols - m.size.width - margin, right_y), "AM2302", fields, footer, accent,
                   text_color, dim_text, ui);
        right_y += m.size.height + gap;
    }

    if (stats.thermal_enabled) {
        const cv::Scalar fire_accent(40, 120, 255); // BGR orange-red
        const cv::Scalar panel_accent = stats.fire_warning ? fire_accent : accent;
        std::vector<PanelField> fields;
        std::string footer = "STATUS " + stats.thermal_status;
        if (stats.thermal_valid) {
            fields.push_back({"MAX", cv::format("%.1fC", stats.thermal_max_c)});
            fields.push_back({"MEAN", cv::format("%.1fC", stats.thermal_mean_c)});
        } else {
            footer = "WAITING FOR SENSOR";
        }
        const PanelLayout m = layout_panel("MLX90640", fields, footer, ui);
        draw_panel(frame, cv::Point(frame.cols - m.size.width - margin, right_y), "MLX90640", fields, footer,
                   panel_accent, stats.fire_warning ? fire_accent : text_color, dim_text, ui);
    }

    // --- Bottom-left: system telemetry.
    {
        const std::vector<PanelField> fields = {
            {"FPS", std::to_string(static_cast<int>(std::round(stats.render_fps)))},
            {"DETECT", std::to_string(static_cast<int>(std::round(stats.inference_ms))) + "ms"},
            {"RANGE", std::to_string(static_cast<int>(std::round(stats.nearest_obstacle_mm))) + "mm"},
        };
        const std::string footer = stats.detector_source_label + "   /   " + mode_text;
        const PanelLayout m = layout_panel("SYSTEM", fields, footer, ui);
        draw_panel(frame, cv::Point(margin, frame.rows - m.size.height - margin), "SYSTEM", fields, footer, accent,
                   text_color, dim_text, ui);
    }

    // --- Bottom-right: detection summary.
    {
        const std::vector<PanelField> fields = {
            {"HUMANS", std::to_string(stats.detected_people)},
            {"BEST", std::to_string(static_cast<int>(std::round(stats.best_person_score * 100))) + "%"},
        };
        const PanelLayout m = layout_panel("DETECTION", fields, "", ui);
        draw_panel(frame, cv::Point(frame.cols - m.size.width - margin, frame.rows - m.size.height - margin),
                   "DETECTION", fields, "", accent, text_color, dim_text, ui);
    }

    // --- Fire / hotspot warning banner (top-center) ---
    if (stats.fire_warning) {
        const cv::Scalar fire_accent(40, 120, 255);
        const std::string warn = cv::format("FIRE / HOTSPOT  %.0fC", stats.thermal_max_c);
        const cv::Size ts = measure_text(warn, kTextBanner, ui, true);
        const cv::Rect banner((frame.cols - ts.width) / 2 - static_cast<int>(20 * ui), static_cast<int>(62 * ui),
                              ts.width + static_cast<int>(40 * ui), ts.height + static_cast<int>(24 * ui));
        fill_translucent_rect(frame, banner, cv::Scalar(0, 0, 30), 0.62);
        cv::rectangle(frame, banner, fire_accent, 2, cv::LINE_AA);
        draw_text(frame, warn,
                  cv::Point(banner.x + static_cast<int>(20 * ui), banner.y + ts.height + static_cast<int>(10 * ui)),
                  kTextBanner, cv::Scalar(60, 200, 255), ui, true);
    }
}

void draw_stream_annotations(cv::Mat& frame, const std::vector<StreamAnnotation>& annotations)
{
    if (frame.empty() || annotations.empty()) {
        return;
    }

    const int w = frame.cols;
    const int h = frame.rows;
    auto to_px = [&](float nx, float ny) {
        return cv::Point(std::max(0, std::min(w - 1, static_cast<int>(std::round(nx * w)))),
                         std::max(0, std::min(h - 1, static_cast<int>(std::round(ny * h)))));
    };

    // Commander palette — deliberately distinct from the green HUD and amber
    // victim boxes: cyan doors, magenta arrows, gold breadcrumb trail.
    const cv::Scalar door_color(255, 210, 40);   // BGR cyan
    const cv::Scalar arrow_color(216, 60, 255);  // BGR magenta
    const cv::Scalar crumb_color(60, 210, 255);  // BGR gold
    const cv::Scalar shadow(0, 0, 0);

    // Pass 1: draw the connecting trail beneath the breadcrumb markers so the
    // firefighter can follow the dropped path in order.
    cv::Point prev_crumb(-1, -1);
    for (const auto& a : annotations) {
        if (a.type != StreamAnnotation::Type::Breadcrumb) {
            continue;
        }
        const cv::Point c = to_px(a.x0, a.y0);
        if (prev_crumb.x >= 0) {
            cv::line(frame, prev_crumb, c, shadow, 5, cv::LINE_AA);
            cv::line(frame, prev_crumb, c, crumb_color, 2, cv::LINE_AA);
        }
        prev_crumb = c;
    }

    // Pass 2: markers on top.
    int crumb_no = 0;
    for (const auto& a : annotations) {
        switch (a.type) {
        case StreamAnnotation::Type::Door: {
            const cv::Point p0 = to_px(a.x0, a.y0);
            const cv::Point p1 = to_px(a.x1, a.y1);
            const cv::Rect box(cv::Point(std::min(p0.x, p1.x), std::min(p0.y, p1.y)),
                               cv::Point(std::max(p0.x, p1.x) + 1, std::max(p0.y, p1.y) + 1));
            fill_translucent_rect(frame, box, door_color, 0.18);
            cv::rectangle(frame, box, shadow, 4, cv::LINE_AA);
            cv::rectangle(frame, box, door_color, 2, cv::LINE_AA);
            const double s = std::max(1.0, frame.cols / 1280.0);
            const cv::Point label(box.x + 4, std::max(18, box.y - static_cast<int>(std::round(9 * s))));
            draw_text(frame, "DOOR", label, kTextTitle, door_color, s, true);
            break;
        }
        case StreamAnnotation::Type::Arrow: {
            const cv::Point tail = to_px(a.x0, a.y0);
            const cv::Point head = to_px(a.x1, a.y1);
            const double len = cv::norm(head - tail);
            const double tip = len > 1.0 ? std::min(0.4, 26.0 / len) : 0.3;
            cv::arrowedLine(frame, tail, head, shadow, 8, cv::LINE_AA, 0, tip);
            cv::arrowedLine(frame, tail, head, arrow_color, 4, cv::LINE_AA, 0, tip);
            break;
        }
        case StreamAnnotation::Type::Breadcrumb: {
            ++crumb_no;
            const cv::Point c = to_px(a.x0, a.y0);
            // Sized off the frame so the waypoint stays readable at a glance in
            // smoke — a firefighter must spot it without studying the HUD.
            const double s = std::max(1.0, frame.cols / 1280.0);
            const int r = static_cast<int>(std::round(13.0 * s));
            cv::circle(frame, c, r + 4, shadow, 3, cv::LINE_AA);
            cv::circle(frame, c, r, shadow, cv::FILLED, cv::LINE_AA);
            cv::circle(frame, c, r - 2, crumb_color, cv::FILLED, cv::LINE_AA);
            const std::string num = std::to_string(crumb_no);
            // Dark digit on the filled gold marker — contrast is already high,
            // so no outline here either.
            const cv::Size ts = measure_text(num, kTextLabel, s, true);
            cv::putText(frame, num, cv::Point(c.x - ts.width / 2, c.y + ts.height / 2), kHudFont, kTextLabel * s,
                        cv::Scalar(20, 20, 20), text_thickness(kTextLabel * s, true), cv::LINE_AA);
            break;
        }
        }
    }
}

CanvasTransform compute_canvas_transform(const cv::Size& source_size)
{
    CanvasTransform transform;
    if (source_size.width <= 0 || source_size.height <= 0) {
        return transform;
    }
    transform.scale = std::min(static_cast<double>(kDisplayWidth) / static_cast<double>(source_size.width),
                               static_cast<double>(kDisplayHeight) / static_cast<double>(source_size.height));
    const int scaled_width = std::max(1, static_cast<int>(std::round(source_size.width * transform.scale)));
    const int scaled_height = std::max(1, static_cast<int>(std::round(source_size.height * transform.scale)));
    transform.offset_x = (kDisplayWidth - scaled_width) / 2;
    transform.offset_y = (kDisplayHeight - scaled_height) / 2;
    return transform;
}

cv::Mat compose_display_canvas(const cv::Mat& source)
{
    cv::Mat canvas(kDisplayHeight, kDisplayWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    if (source.empty()) {
        return canvas;
    }

    const CanvasTransform transform = compute_canvas_transform(source.size());
    const int scaled_width = std::max(1, static_cast<int>(std::round(source.cols * transform.scale)));
    const int scaled_height = std::max(1, static_cast<int>(std::round(source.rows * transform.scale)));
    cv::Mat resized;
    const int interp = transform.scale >= 1.0 ? cv::INTER_NEAREST : cv::INTER_AREA;
    cv::resize(source, resized, cv::Size(scaled_width, scaled_height), 0.0, 0.0, interp);
    resized.copyTo(canvas(cv::Rect(transform.offset_x, transform.offset_y, scaled_width, scaled_height)));

    return canvas;
}

void draw_ground_grid(cv::Mat& canvas, const GroundPlaneState& ground, const SharedFrame& lidar_frame,
                      const CanvasTransform& transform, const Options& opt)
{
    if (!opt.enable_ground_plane || !ground.plane.valid || canvas.empty() || lidar_frame.depth_mm.empty()) {
        return;
    }

    // Build an orthonormal basis ON the floor: n up, forward, right. The grid is
    // laid out in (lateral, forward) plane coordinates and then pushed through
    // the pinhole model, so perspective falls out for free.
    const cv::Vec3f n = ground.plane.normal;
    cv::Vec3f forward = cv::Vec3f(0.0f, 0.0f, 1.0f) - n[2] * n;   // camera +Z projected onto the plane
    if (cv::norm(forward) < 1.0e-2) {
        // Operator looking almost straight down: +Z is parallel to the normal and
        // its projection vanishes. Use the camera's own down axis instead.
        forward = cv::Vec3f(0.0f, 1.0f, 0.0f) - n[1] * n;
    }
    const double forward_len = cv::norm(forward);
    if (forward_len < 1.0e-6) {
        return;
    }
    forward /= static_cast<float>(forward_len);
    const cv::Vec3f right = forward.cross(n);
    const cv::Vec3f origin = -ground.plane.d * n;   // floor point directly beneath the operator

    const float spacing = std::max(100.0f, opt.grid_spacing_mm);
    const float half_width = std::max(spacing, opt.grid_half_width_mm);
    const float near_mm = std::max(150.0f, opt.grid_near_mm);
    const float far_mm =
        std::max(near_mm + spacing, std::min(opt.grid_far_mm, static_cast<float>(opt.max_depth_mm)));
    const float step = std::max(60.0f, opt.grid_segment_mm);
    const float fade_span = std::max(1.0f, far_mm - near_mm);

    // Same green the rest of the HUD uses (tactical_rescue_render.cpp accent),
    // dimmed while we are coasting on a stale fit so a soft lock reads as soft.
    const cv::Scalar grid_color(74, 255, 158);
    const double state_gain = ground.stale ? 0.45 : 1.0;

    const cv::Mat valid_mask = build_geometry_mask(lidar_frame.depth_mm, lidar_frame.confidence, opt);
    const cv::Rect depth_rect(0, 0, lidar_frame.depth_mm.cols, lidar_frame.depth_mm.rows);
    int segment_budget = kMaxGridSegments;

    auto to_camera = [&](float lateral, float forward_mm) {
        return cv::Point3f(origin[0] + lateral * right[0] + forward_mm * forward[0],
                           origin[1] + lateral * right[1] + forward_mm * forward[1],
                           origin[2] + lateral * right[2] + forward_mm * forward[2]);
    };

    // A grid vertex is hidden when the ToF sees something closer along that ray.
    // Where depth is INVALID we draw it: an unknown pixel (smoke, no return) must
    // not punch holes in the navigation grid — that is exactly where the operator
    // needs it most.
    auto is_occluded = [&](const cv::Point3f& p, const cv::Point2f& px) {
        const int u = cvRound(px.x);
        const int v = cvRound(px.y);
        if (!depth_rect.contains(cv::Point(u, v))) {
            return true;
        }
        if (valid_mask.at<uint8_t>(v, u) == 0) {
            return false;
        }
        const float measured = lidar_frame.depth_mm.at<float>(v, u);
        const float expected =
            opt.depth_is_radial ? static_cast<float>(cv::norm(cv::Vec3f(p.x, p.y, p.z))) : p.z;
        return measured < expected - opt.grid_occlusion_tol_mm;
    };

    // Walk a grid line in 3D and emit only its visible, in-front-of-us pieces.
    // A straight 3D line does project to a straight 2D line under a pinhole, so
    // this tessellation exists for near-plane clipping and occlusion, not curvature.
    auto emit_line = [&](float lat0, float fwd0, float lat1, float fwd1, bool emphasise) {
        const float length = std::hypot(lat1 - lat0, fwd1 - fwd0);
        const int steps = std::max(1, static_cast<int>(std::ceil(length / step)));
        bool have_previous = false;
        cv::Point2f previous_canvas;

        for (int i = 0; i <= steps && segment_budget > 0; ++i) {
            const float t = static_cast<float>(i) / static_cast<float>(steps);
            const cv::Point3f p = to_camera(lat0 + (lat1 - lat0) * t, fwd0 + (fwd1 - fwd0) * t);

            cv::Point2f px;
            if (!project_point(p, opt, px) || is_occluded(p, px)) {
                have_previous = false;
                continue;
            }

            const cv::Point2f canvas_px = transform.map(px);
            if (have_previous) {
                const float depth_t = clamp01((p.z - near_mm) / fade_span);
                const double gain =
                    state_gain * (1.0 - 0.72 * static_cast<double>(depth_t)) * (emphasise ? 1.0 : 0.72);
                // Sub-pixel endpoints (1/8 px): stops the grid crawling as it slides.
                constexpr int kShift = 3;
                constexpr float kSubPixel = static_cast<float>(1 << kShift);
                const cv::Point a(cvRound(previous_canvas.x * kSubPixel), cvRound(previous_canvas.y * kSubPixel));
                const cv::Point b(cvRound(canvas_px.x * kSubPixel), cvRound(canvas_px.y * kSubPixel));
                cv::line(canvas, a, b, grid_color * gain, depth_t < 0.45f ? 2 : 1, cv::LINE_AA, kShift);
                --segment_budget;
            }
            previous_canvas = canvas_px;
            have_previous = true;
        }
    };

    // Lines running away from the operator; the centre line is drawn brighter as
    // a heading reference.
    for (float lateral = -half_width; lateral <= half_width + 1.0f; lateral += spacing) {
        emit_line(lateral, near_mm, lateral, far_mm, std::fabs(lateral) < 1.0f);
    }
    // Cross lines at constant range.
    for (float forward_mm = near_mm; forward_mm <= far_mm + 1.0f; forward_mm += spacing) {
        emit_line(-half_width, forward_mm, half_width, forward_mm, false);
    }
}

} // namespace tactical_rescue
