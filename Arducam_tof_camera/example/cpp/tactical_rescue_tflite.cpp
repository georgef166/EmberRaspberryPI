// =============================================================================
// GOOGLE TECHNOLOGY: TensorFlow Lite (tensorflow/lite C++ API v2.20)
//
// This file implements real-time on-device person detection using Google's
// TensorFlow Lite inference engine. No network or cloud connection is required —
// inference runs entirely on the Raspberry Pi 4B CPU, which is critical for
// deployment inside active structure fires where radio and network access
// are unavailable.
//
// Model: MobileNet SSD v1 COCO quantized (models/detect.tflite, ~4MB)
//   - Source: Google TFLite Model Zoo
//   - Input:  [1, 300, 300, 3] uint8 RGB
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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

#include <opencv2/imgproc.hpp>

// TensorFlow Lite C++ API headers (Google)
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

namespace tactical_rescue {

static constexpr int kModelInputSize = 300;   // MobileNet SSD expects 300x300 input
static constexpr int kMaxDetections = 10;     // SSD model outputs up to 10 detections
static constexpr float kPersonClassId = 1.0f; // COCO label index: 0=background, 1=person

struct TFLitePersonDetector::Impl {
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    bool loaded = false;
    int input_w = kModelInputSize;
    int input_h = kModelInputSize;
};

TFLitePersonDetector::TFLitePersonDetector() : impl_(std::make_unique<Impl>()) {}
TFLitePersonDetector::~TFLitePersonDetector() = default;

bool TFLitePersonDetector::load(const std::string& model_path)
{
    // [TFLite] Load flatbuffer model from disk into memory
    impl_->model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!impl_->model) {
        std::cerr << "[TFLite] Failed to load model from: " << model_path << std::endl;
        return false;
    }

    // [TFLite] Register built-in ops (Conv2D, DepthwiseConv, etc.) and build interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*impl_->model, resolver);
    if (builder(&impl_->interpreter) != kTfLiteOk) {
        std::cerr << "[TFLite] Failed to build interpreter" << std::endl;
        return false;
    }

    impl_->interpreter->SetNumThreads(4); // Use all 4 cores of Raspberry Pi 4B
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
              << " input=" << impl_->input_w << "x" << impl_->input_h << std::endl;
    return true;
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
    // The ToF amplitude channel (infrared reflection intensity) is used as the
    // model input — it captures scene structure through smoke where RGB cameras
    // produce nothing. Falls back to depth_mm if amplitude is unavailable.
    const cv::Mat& source = frame.amplitude.empty() ? frame.depth_mm : frame.amplitude;

    // Normalize float32 ToF data to uint8 range [0, 255] for TFLite quantized model
    double min_val, max_val;
    cv::minMaxLoc(source, &min_val, &max_val);
    cv::Mat normalized;
    source.convertTo(normalized, CV_8U, 255.0 / std::max(1.0, max_val - min_val),
                     -min_val * 255.0 / std::max(1.0, max_val - min_val));

    // Resize to 300x300 (MobileNet SSD input size) and replicate to 3-channel RGB
    cv::Mat resized;
    cv::resize(normalized, resized, cv::Size(impl_->input_w, impl_->input_h), 0, 0, cv::INTER_LINEAR);
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_GRAY2RGB);

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

    for (int i = 0; i < n; ++i) {
        if (std::abs(classes[i] - kPersonClassId) > 0.5f) {
            continue;
        }
        const float score = scores[i];
        if (score < opt.person_confidence) {
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

        const float depth_mm = static_cast<float>(cv::mean(frame.depth_mm(box))[0]);
        if (depth_mm < static_cast<float>(opt.min_depth_mm) || depth_mm > static_cast<float>(opt.max_depth_mm)) {
            continue;
        }

        PersonDetection det;
        det.box = box;
        det.confidence = score;
        det.mean_depth_mm = depth_mm;
        det.valid = true;
        state.people.push_back(det);
        best_confidence = std::max(best_confidence, score);
    }

    std::sort(state.people.begin(), state.people.end(),
              [](const PersonDetection& a, const PersonDetection& b) { return a.confidence > b.confidence; });
    if (static_cast<int>(state.people.size()) > opt.max_people) {
        state.people.resize(static_cast<size_t>(opt.max_people));
    }

    state.valid = true;
    if (best_score_out) {
        *best_score_out = best_confidence;
    }
    return state;
}

} // namespace tactical_rescue
