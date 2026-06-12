// =============================================================================
// GOOGLE TECHNOLOGY: TensorFlow Lite (tensorflow/lite C++ API v2.20)
//
// This file implements real-time on-device person detection using Google's
// TensorFlow Lite inference engine. No network or cloud connection is required —
// inference runs entirely on-device, which is critical for deployment inside
// active structure fires where radio and network access are unavailable.
//
// Backends (selected at load() time):
//   - CPU:       all 4 Raspberry Pi 4B ARM cores via the built-in op resolver.
//   - Coral TPU: Google Edge TPU via libedgetpu (--edgetpu, EMBER_HAVE_EDGETPU).
//                Offloads inference to the attached Coral, freeing the CPU for
//                ToF rendering and enabling a larger/faster model.
//
// Model: SSD MobileNet (COCO, quantized uint8) — v1 on CPU, v2 _edgetpu on Coral
//   - Source: Google TFLite Model Zoo / Coral model garden
//   - Input:  [1, H, W, 3] uint8 RGB (H/W read dynamically from the model)
//   - Output: detection_boxes, detection_classes, detection_scores, num_detections
//
// Pipeline:
//   ToF amplitude frame (float32, 240x180)
//     → normalize to uint8
//     → resize to 300x300
//     → expand to 3-channel RGB
//     → TFLite Invoke()
//     → filter class=1 (person) detections above confidence threshold
//     → depth-validate each box against live depth_mm frame
//     → return PersonDetection structs to renderer
// =============================================================================

#include "tactical_rescue_tflite.hpp"

#include <iostream>

// TensorFlow Lite is optional (no apt package on Raspberry Pi OS — built from
// source). When libtensorflow-lite is absent at build time, EMBER_HAVE_TFLITE is
// undefined and this file compiles to a stub so the ToF/thermal/CV pipeline still
// builds; the TFLite (and Coral) detectors simply report unavailable at runtime.
#ifdef EMBER_HAVE_TFLITE

#include <algorithm>
#include <cmath>
#include <cstring>

#include <opencv2/imgproc.hpp>

// TensorFlow Lite C++ API headers (Google)
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

// =============================================================================
// GOOGLE CORAL EDGE TPU (libedgetpu)
//
// When the binary is built with EMBER_HAVE_EDGETPU (set by CMake when libedgetpu
// is found) and run with --edgetpu, the same SSD person-detection model — compiled
// for the Edge TPU — runs on the attached Coral instead of the Pi's ARM cores.
// This frees all 4 CPU cores for ToF rendering and pushes inference to ~70+ FPS,
// leaving headroom for a larger/more accurate detector (e.g. SSD MobileNet v2).
// =============================================================================
#ifdef EMBER_HAVE_EDGETPU
#include "edgetpu.h"
#endif

namespace tactical_rescue {

static constexpr int kModelInputSize = 300; // MobileNet SSD expects 300x300 input
static constexpr int kMaxDetections = 10;   // SSD model outputs up to 10 detections
// COCO "person" output index is model-dependent (1 for the bundled 91-class
// model, 0 for the Coral v2 COCO model) — supplied at runtime via opt.person_class_id.

namespace {

struct StableTfliteTrack {
    cv::Rect2f box;
    float confidence = 0.0f;
    float mean_depth_mm = 0.0f;
    int hits = 0;
    int misses = 0;
    uint64_t last_sequence = 0;
};

cv::Mat build_depth_detector_input(const SharedFrame& frame, const Options& opt)
{
    if (frame.depth_mm.empty()) {
        return {};
    }

    cv::Mat depth_gray = to_gray_preview(frame.depth_mm, opt);
    if (!frame.confidence.empty()) {
        cv::Mat valid = build_geometry_mask(frame.depth_mm, frame.confidence, opt);
        depth_gray.setTo(0, valid == 0);
    }

    cv::Mat depth_bgr;
    cv::cvtColor(depth_gray, depth_bgr, cv::COLOR_GRAY2BGR);
    return depth_bgr;
}

cv::Mat build_tflite_detector_input(const SharedFrame& frame, const Options& opt)
{
    cv::Mat detector_bgr;
    if (opt.tflite_input_mode == TfliteInputMode::PSEUDO) {
        detector_bgr = build_pseudo_detector_input(frame, opt);
    } else if (opt.tflite_input_mode == TfliteInputMode::DEPTH) {
        detector_bgr = build_depth_detector_input(frame, opt);
    } else {
        detector_bgr = build_amplitude_detector_input(frame, opt);
    }

    if (detector_bgr.empty() && opt.tflite_input_mode != TfliteInputMode::DEPTH) {
        detector_bgr = build_depth_detector_input(frame, opt);
    }
    return detector_bgr;
}

float stable_rect_iou(const cv::Rect2f& lhs, const cv::Rect& rhs)
{
    const cv::Rect2f rhs_f(static_cast<float>(rhs.x), static_cast<float>(rhs.y), static_cast<float>(rhs.width),
                           static_cast<float>(rhs.height));
    const cv::Rect2f intersection = lhs & rhs_f;
    if (intersection.width <= 0.0f || intersection.height <= 0.0f) {
        return 0.0f;
    }
    const float intersection_area = intersection.width * intersection.height;
    const float union_area = lhs.area() + rhs_f.area() - intersection_area;
    return union_area > 0.0f ? intersection_area / union_area : 0.0f;
}

cv::Point2f stable_rect_center(const cv::Rect2f& rect)
{
    return cv::Point2f(rect.x + rect.width * 0.5f, rect.y + rect.height * 0.5f);
}

cv::Point2f stable_rect_center(const cv::Rect& rect)
{
    return cv::Point2f(static_cast<float>(rect.x) + static_cast<float>(rect.width) * 0.5f,
                       static_cast<float>(rect.y) + static_cast<float>(rect.height) * 0.5f);
}

cv::Rect2f blend_track_box(const cv::Rect2f& previous, const cv::Rect& current)
{
    const float current_mix = 0.62f;
    const float previous_mix = 1.0f - current_mix;
    return cv::Rect2f(previous.x * previous_mix + static_cast<float>(current.x) * current_mix,
                      previous.y * previous_mix + static_cast<float>(current.y) * current_mix,
                      previous.width * previous_mix + static_cast<float>(current.width) * current_mix,
                      previous.height * previous_mix + static_cast<float>(current.height) * current_mix);
}

void stabilize_tflite_detections(std::vector<PersonDetection>& detections, uint64_t sequence, const cv::Size& frame_size,
                                 const Options& opt, float& best_confidence)
{
    static std::vector<StableTfliteTrack> tracks;

    for (auto& track : tracks) {
        ++track.misses;
        track.confidence *= 0.91f;
    }

    std::vector<bool> claimed(tracks.size(), false);
    for (const auto& detection : detections) {
        int best_track = -1;
        float best_score = 0.0f;
        for (size_t i = 0; i < tracks.size(); ++i) {
            if (claimed[i]) {
                continue;
            }

            const float iou = stable_rect_iou(tracks[i].box, detection.box);
            const cv::Point2f previous_center = stable_rect_center(tracks[i].box);
            const cv::Point2f current_center = stable_rect_center(detection.box);
            const float dx = previous_center.x - current_center.x;
            const float dy = previous_center.y - current_center.y;
            const float diagonal =
                std::sqrt(tracks[i].box.width * tracks[i].box.width + tracks[i].box.height * tracks[i].box.height);
            const float center_score =
                clamp01(1.0f - std::sqrt(dx * dx + dy * dy) / std::max(32.0f, diagonal * 1.7f));
            const float depth_delta = (tracks[i].mean_depth_mm > 0.0f && detection.mean_depth_mm > 0.0f)
                                          ? std::abs(tracks[i].mean_depth_mm - detection.mean_depth_mm)
                                          : 0.0f;
            const float depth_penalty = clamp01(depth_delta / 1400.0f) * 0.12f;
            const float match_score = std::max(iou, center_score) * 0.72f +
                                      std::min(iou, center_score) * 0.28f - depth_penalty;
            if (match_score > best_score) {
                best_score = match_score;
                best_track = static_cast<int>(i);
            }
        }

        if (best_track >= 0 && best_score > 0.10f) {
            auto& track = tracks[static_cast<size_t>(best_track)];
            claimed[static_cast<size_t>(best_track)] = true;
            track.box = blend_track_box(track.box, detection.box);
            track.confidence = clamp01(track.confidence * 0.55f + detection.confidence * 0.45f + 0.035f);
            track.mean_depth_mm =
                detection.mean_depth_mm > 0.0f
                    ? (track.mean_depth_mm > 0.0f ? track.mean_depth_mm * 0.35f + detection.mean_depth_mm * 0.65f
                                                  : detection.mean_depth_mm)
                    : track.mean_depth_mm;
            track.hits += 1;
            track.misses = 0;
            track.last_sequence = sequence;
        } else {
            StableTfliteTrack track;
            track.box = cv::Rect2f(detection.box);
            track.confidence = clamp01(detection.confidence + 0.02f);
            track.mean_depth_mm = detection.mean_depth_mm;
            track.hits = 1;
            track.misses = 0;
            track.last_sequence = sequence;
            tracks.push_back(track);
            claimed.push_back(true);
        }
    }

    std::vector<PersonDetection> stable;
    const float immediate_floor = std::max(0.30f, opt.person_confidence);
    const float confirmed_floor = std::max(0.22f, opt.person_confidence - 0.16f);
    const cv::Rect bounds(0, 0, frame_size.width, frame_size.height);
    for (const auto& track : tracks) {
        const float held_confidence = clamp01(track.confidence * (1.0f - std::min(0.55f, track.misses * 0.08f)));
        const bool strong_new_hit = track.misses == 0 && held_confidence >= immediate_floor;
        const bool confirmed_track = track.hits >= 2 && track.misses <= 6 && held_confidence >= confirmed_floor;
        if (!strong_new_hit && !confirmed_track) {
            continue;
        }

        cv::Rect box = cv::Rect(track.box) & bounds;
        if (box.area() <= 0) {
            continue;
        }

        PersonDetection detection;
        detection.box = box;
        detection.confidence = held_confidence;
        detection.mean_depth_mm = track.mean_depth_mm;
        detection.valid = true;
        stable.push_back(std::move(detection));
        best_confidence = std::max(best_confidence, held_confidence);
    }

    tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [](const StableTfliteTrack& track) {
                     return track.misses > 9 || track.confidence < 0.10f;
                 }),
                 tracks.end());

    std::sort(stable.begin(), stable.end(), [](const PersonDetection& lhs, const PersonDetection& rhs) {
        if (lhs.confidence != rhs.confidence) {
            return lhs.confidence > rhs.confidence;
        }
        if (lhs.mean_depth_mm > 0.0f && rhs.mean_depth_mm > 0.0f) {
            return lhs.mean_depth_mm < rhs.mean_depth_mm;
        }
        return lhs.box.area() > rhs.box.area();
    });
    detections = std::move(stable);
}

} // namespace

struct TFLitePersonDetector::Impl {
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    bool loaded = false;
    bool on_edgetpu = false;
    int input_w = kModelInputSize;
    int input_h = kModelInputSize;
#ifdef EMBER_HAVE_EDGETPU
    // Keep the Coral device context alive for the lifetime of the interpreter.
    std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context;
#endif
};

TFLitePersonDetector::TFLitePersonDetector() : impl_(std::make_unique<Impl>()) {}
TFLitePersonDetector::~TFLitePersonDetector() = default;

bool TFLitePersonDetector::load(const std::string& model_path, bool use_edgetpu)
{
    // [TFLite] Load flatbuffer model from disk into memory
    impl_->model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!impl_->model) {
        std::cerr << "[TFLite] Failed to load model from: " << model_path << std::endl;
        return false;
    }

    // [TFLite] Register built-in ops (Conv2D, DepthwiseConv, etc.).
    tflite::ops::builtin::BuiltinOpResolver resolver;

#ifdef EMBER_HAVE_EDGETPU
    // [Coral] Open the Edge TPU device and register its custom op so the
    // interpreter can dispatch the edgetpu-custom-op subgraph to the Coral.
    if (use_edgetpu) {
        impl_->edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
        if (!impl_->edgetpu_context) {
            std::cerr << "[Coral] No Edge TPU device found — falling back to CPU inference."
                      << std::endl;
            use_edgetpu = false;
        } else {
            resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
        }
    }
#else
    if (use_edgetpu) {
        std::cerr << "[Coral] Binary built without Edge TPU support (libedgetpu not found at "
                     "build time) — falling back to CPU inference."
                  << std::endl;
        use_edgetpu = false;
    }
#endif

    tflite::InterpreterBuilder builder(*impl_->model, resolver);
    if (builder(&impl_->interpreter) != kTfLiteOk) {
        std::cerr << "[TFLite] Failed to build interpreter" << std::endl;
        return false;
    }

#ifdef EMBER_HAVE_EDGETPU
    if (use_edgetpu) {
        // [Coral] Bind the interpreter to the opened Edge TPU context. With the TPU
        // doing the heavy lifting, a single host thread is optimal (more just adds
        // dispatch contention).
        impl_->interpreter->SetExternalContext(kTfLiteEdgeTpuContext, impl_->edgetpu_context.get());
        impl_->interpreter->SetNumThreads(1);
        impl_->on_edgetpu = true;
    } else
#endif
    {
        impl_->interpreter->SetNumThreads(4); // Use all 4 cores of Raspberry Pi 4B
    }

    // [TFLite] Allocate memory for all input/output tensors
    if (impl_->interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "[TFLite] Failed to allocate tensors" << std::endl;
        return false;
    }

    // Read actual input dimensions from model
    const int input_idx = impl_->interpreter->inputs()[0];
    const TfLiteTensor* input_tensor = impl_->interpreter->tensor(input_idx);
    if (input_tensor->dims->size >= 3) {
        impl_->input_h = input_tensor->dims->data[1];
        impl_->input_w = input_tensor->dims->data[2];
    }

    impl_->loaded = true;
    std::cout << "[TFLite] Model loaded: " << model_path
              << " input=" << impl_->input_w << "x" << impl_->input_h
              << " backend=" << (impl_->on_edgetpu ? "Coral Edge TPU" : "CPU") << std::endl;
    return true;
}

bool TFLitePersonDetector::uses_edgetpu() const
{
    return impl_->on_edgetpu;
}

bool TFLitePersonDetector::is_loaded() const
{
    return impl_->loaded;
}

DetectionState TFLitePersonDetector::detect(const SharedFrame& frame, const Options& opt, float* best_score_out)
{
    DetectionState state;
    state.source_sequence = frame.sequence;

    if (!impl_->loaded || frame.depth_mm.empty()) {
        return state;
    }

    // --- INPUT PREPROCESSING FOR TFLITE ---
    // Use a ToF composite by default: depth silhouette, amplitude texture, and
    // confidence mask. It is more stable than raw amplitude when smoke/noise
    // causes frame-wide haze.
    cv::Mat detector_bgr = build_tflite_detector_input(frame, opt);
    if (detector_bgr.empty()) {
        return state;
    }

    // Resize to the model input size and convert OpenCV BGR ordering to RGB.
    cv::Mat resized;
    cv::resize(detector_bgr, resized, cv::Size(impl_->input_w, impl_->input_h), 0, 0, cv::INTER_LINEAR);
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // [TFLite] Write preprocessed frame into the model's input tensor
    const int input_idx = impl_->interpreter->inputs()[0];
    uint8_t* input_data = impl_->interpreter->typed_tensor<uint8_t>(input_idx);
    if (!input_data) {
        return state;
    }
    std::memcpy(input_data, rgb.data, static_cast<size_t>(impl_->input_w * impl_->input_h * 3));

    // [TFLite] Run inference — this is the core Google ML step
    if (impl_->interpreter->Invoke() != kTfLiteOk) {
        return state;
    }

    // --- PARSE TFLITE SSD OUTPUT TENSORS ---
    // MobileNet SSD v1 COCO outputs 4 tensors:
    //   [0] detection_boxes    float[1, N, 4]  — (ymin, xmin, ymax, xmax) normalized 0-1
    //   [1] detection_classes  float[1, N]     — COCO class index (1 = person)
    //   [2] detection_scores   float[1, N]     — confidence 0-1
    //   [3] num_detections     float[1]        — number of valid detections
    const float* boxes   = impl_->interpreter->typed_output_tensor<float>(0);
    const float* classes = impl_->interpreter->typed_output_tensor<float>(1);
    const float* scores  = impl_->interpreter->typed_output_tensor<float>(2);
    const float* num_det = impl_->interpreter->typed_output_tensor<float>(3);

    if (!boxes || !classes || !scores || !num_det) {
        return state;
    }

    const int n = std::min(static_cast<int>(*num_det), kMaxDetections);
    const int fw = frame.depth_mm.cols;
    const int fh = frame.depth_mm.rows;
    float best_confidence = 0.0f;
    const float candidate_floor = std::max(0.20f, opt.person_confidence - 0.25f);

    const float person_class_id = static_cast<float>(opt.person_class_id);
    for (int i = 0; i < n; ++i) {
        if (std::abs(classes[i] - person_class_id) > 0.5f) {
            continue;
        }
        const float score = scores[i];
        best_confidence = std::max(best_confidence, score);
        if (score < candidate_floor) {
            continue;
        }

        // Boxes are (ymin, xmin, ymax, xmax) normalized 0-1
        const float ymin = std::max(0.0f, boxes[i * 4 + 0]);
        const float xmin = std::max(0.0f, boxes[i * 4 + 1]);
        const float ymax = std::min(1.0f, boxes[i * 4 + 2]);
        const float xmax = std::min(1.0f, boxes[i * 4 + 3]);

        cv::Rect box(static_cast<int>(xmin * fw), static_cast<int>(ymin * fh),
                     static_cast<int>((xmax - xmin) * fw), static_cast<int>((ymax - ymin) * fh));
        box &= cv::Rect(0, 0, fw, fh);
        if (box.area() <= 0) {
            continue;
        }

        cv::Mat valid_depth = (frame.depth_mm(box) > static_cast<float>(opt.min_depth_mm)) &
                              (frame.depth_mm(box) < static_cast<float>(opt.max_depth_mm));
        valid_depth.convertTo(valid_depth, CV_8U, 255.0);
        if (!frame.confidence.empty()) {
            cv::Mat confident = frame.confidence(box) >= static_cast<float>(opt.confidence_threshold);
            confident.convertTo(confident, CV_8U, 255.0);
            cv::bitwise_and(valid_depth, confident, valid_depth);
        }
        if (cv::countNonZero(valid_depth) == 0) {
            continue;
        }

        const float depth_mm = static_cast<float>(cv::mean(frame.depth_mm(box), valid_depth)[0]);
        if (depth_mm < static_cast<float>(opt.min_depth_mm) || depth_mm > static_cast<float>(opt.max_depth_mm)) {
            continue;
        }

        PersonDetection det;
        det.box = box;
        det.confidence = score;
        det.mean_depth_mm = depth_mm;
        det.valid = true;
        state.people.push_back(det);
    }

    stabilize_tflite_detections(state.people, frame.sequence, frame.depth_mm.size(), opt, best_confidence);

    std::sort(state.people.begin(), state.people.end(), [](const PersonDetection& a, const PersonDetection& b) {
        if (a.confidence != b.confidence) {
            return a.confidence > b.confidence;
        }
        if (a.mean_depth_mm > 0.0f && b.mean_depth_mm > 0.0f) {
            return a.mean_depth_mm < b.mean_depth_mm;
        }
        return a.box.area() > b.box.area();
    });
    if (static_cast<int>(state.people.size()) > opt.max_people) {
        state.people.resize(static_cast<size_t>(opt.max_people));
    }

    state.valid = true;
    if (best_score_out) {
        *best_score_out = state.people.empty() ? best_confidence : state.people.front().confidence;
    }
    return state;
}

} // namespace tactical_rescue

#else // EMBER_HAVE_TFLITE not defined — stub implementation

namespace tactical_rescue {

struct TFLitePersonDetector::Impl {};

TFLitePersonDetector::TFLitePersonDetector() : impl_(nullptr) {}
TFLitePersonDetector::~TFLitePersonDetector() = default;

bool TFLitePersonDetector::load(const std::string& model_path, bool use_edgetpu)
{
    (void)model_path;
    (void)use_edgetpu;
    std::cerr << "[TFLite] Not compiled in (libtensorflow-lite not found at build time). "
                 "Use --detector-source thermal or the ToF CV detectors instead." << std::endl;
    return false;
}

bool TFLitePersonDetector::is_loaded() const { return false; }
bool TFLitePersonDetector::uses_edgetpu() const { return false; }

DetectionState TFLitePersonDetector::detect(const SharedFrame& frame, const Options& opt, float* best_score_out)
{
    (void)frame;
    (void)opt;
    if (best_score_out) {
        *best_score_out = 0.0f;
    }
    return {};
}

} // namespace tactical_rescue

#endif // EMBER_HAVE_TFLITE
