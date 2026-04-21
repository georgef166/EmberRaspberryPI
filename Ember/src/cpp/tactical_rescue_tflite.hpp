#pragma once
#include <memory>
#include <string>
#include "tactical_rescue.hpp"

namespace tactical_rescue {

class TFLitePersonDetector {
public:
    TFLitePersonDetector();
    ~TFLitePersonDetector();

    bool load(const std::string& model_path);
    bool is_loaded() const;

    DetectionState detect(const SharedFrame& frame, const Options& opt, float* best_score_out = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tactical_rescue
