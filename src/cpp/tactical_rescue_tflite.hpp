#pragma once
#include <memory>
#include <string>
#include "tactical_rescue.hpp"

namespace tactical_rescue {

class TFLitePersonDetector {
public:
    TFLitePersonDetector();
    ~TFLitePersonDetector();

    // use_edgetpu routes inference through the Coral Edge TPU delegate (libedgetpu).
    // Requires an Edge TPU-compiled model (e.g. ssd_mobilenet_v2_coco_edgetpu.tflite)
    // and the binary built with EMBER_HAVE_EDGETPU. Falls back to CPU otherwise.
    bool load(const std::string& model_path, bool use_edgetpu = false);
    bool is_loaded() const;

    // True when inference is actually running on the Coral Edge TPU.
    bool uses_edgetpu() const;

    DetectionState detect(const SharedFrame& frame, const Options& opt, float* best_score_out = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tactical_rescue
