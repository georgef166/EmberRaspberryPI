#include "ArducamTOFCamera.hpp"
#include "tactical_rescue.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdlib>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace Arducam;

namespace {

using namespace tactical_rescue;

struct HelperProcess {
    pid_t pid = -1;
    int read_fd = -1;
    FILE* stream = nullptr;
};

bool start_am2302_helper(const Options& opt, HelperProcess& process)
{
    int pipe_fds[2] = {-1, -1};
    if (pipe(pipe_fds) != 0) {
        return false;
    }

    const pid_t pid = fork();
    if (pid < 0) {
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        return false;
    }

    if (pid == 0) {
        dup2(pipe_fds[1], STDOUT_FILENO);
        close(pipe_fds[0]);
        close(pipe_fds[1]);

        const std::string gpio = std::to_string(opt.am2302_gpio);
        execlp("python3", "python3", "-u", opt.am2302_helper_path.c_str(), "--gpio", gpio.c_str(), "--interval", "2.0",
               static_cast<char*>(nullptr));
        _exit(127);
    }

    close(pipe_fds[1]);
    process.pid = pid;
    process.read_fd = pipe_fds[0];
    process.stream = fdopen(process.read_fd, "r");
    if (process.stream == nullptr) {
        close(process.read_fd);
        process.read_fd = -1;
        kill(process.pid, SIGTERM);
        waitpid(process.pid, nullptr, 0);
        process.pid = -1;
        return false;
    }
    return true;
}

void stop_helper_process(HelperProcess& process)
{
    if (process.stream != nullptr) {
        fclose(process.stream);
        process.stream = nullptr;
        process.read_fd = -1;
    } else if (process.read_fd >= 0) {
        close(process.read_fd);
        process.read_fd = -1;
    }

    if (process.pid > 0) {
        kill(process.pid, SIGTERM);
        waitpid(process.pid, nullptr, 0);
        process.pid = -1;
    }
}

bool parse_am2302_line(const std::string& line, std::string& status, float& temperature_c, float& humidity_percent)
{
    std::stringstream stream(line);
    std::string temp_token;
    std::string humidity_token;
    if (!std::getline(stream, status, ',')) {
        return false;
    }
    if (!std::getline(stream, temp_token, ',')) {
        return false;
    }
    if (!std::getline(stream, humidity_token, ',')) {
        return false;
    }

    temperature_c = temp_token.empty() ? 0.0f : std::strtof(temp_token.c_str(), nullptr);
    humidity_percent = humidity_token.empty() ? 0.0f : std::strtof(humidity_token.c_str(), nullptr);
    return true;
}

bool parse_args(int argc, char* argv[], Options& opt)
{
    auto parse_detector_source = [](const std::string& value) {
        if (value == "auto") {
            return DetectorSource::AUTO;
        }
        if (value == "amplitude") {
            return DetectorSource::AMPLITUDE;
        }
        if (value == "confidence") {
            return DetectorSource::CONFIDENCE;
        }
        if (value == "pseudo") {
            return DetectorSource::PSEUDO;
        }
        std::cerr << "Unknown detector source: " << value << std::endl;
        std::exit(2);
    };

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
                << "  --range MM               ToF range in mm\n"
                << "  --min-depth MM           Ignore geometry nearer than this\n"
                << "  --max-depth MM           Ignore geometry farther than this\n"
                << "  --confidence NUM         Depth confidence threshold\n"
                << "  --person-conf FLOAT      Human candidate confidence threshold (default 0.50)\n"
                << "  --max-people NUM         Maximum rendered person detections\n"
                << "  --detection-fps NUM      Detector cadence in frames per second\n"
                << "  --detector-source MODE   auto | amplitude | confidence | pseudo\n"
                << "  --am2302-gpio NUM        BCM GPIO used for AM2302 data (default 4)\n"
                << "  --no-am2302              Disable AM2302 ambient overlay\n"
                << "  --hud-scale NUM          HUD scale factor\n"
                << "  --show-detector-input    Show the exact image sent to the ToF detector\n"
                << "  --no-preview             Run acquisition/inference without UI\n";
            return false;
        } else if (arg == "--device") {
            opt.device = std::atoi(require_value("--device"));
        } else if (arg == "--range") {
            opt.range_mm = std::atoi(require_value("--range"));
            opt.max_depth_mm = opt.range_mm;
        } else if (arg == "--min-depth") {
            opt.min_depth_mm = std::atoi(require_value("--min-depth"));
        } else if (arg == "--max-depth") {
            opt.max_depth_mm = std::atoi(require_value("--max-depth"));
        } else if (arg == "--confidence") {
            opt.confidence_threshold = std::atoi(require_value("--confidence"));
        } else if (arg == "--person-conf") {
            opt.person_confidence = std::atof(require_value("--person-conf"));
        } else if (arg == "--max-people") {
            opt.max_people = std::max(1, std::atoi(require_value("--max-people")));
        } else if (arg == "--detection-fps") {
            opt.detection_fps = std::max(1, std::atoi(require_value("--detection-fps")));
        } else if (arg == "--detector-source") {
            opt.detector_source = parse_detector_source(require_value("--detector-source"));
        } else if (arg == "--am2302-gpio") {
            opt.am2302_gpio = std::max(0, std::atoi(require_value("--am2302-gpio")));
        } else if (arg == "--no-am2302") {
            opt.enable_am2302 = false;
        } else if (arg == "--hud-scale") {
            opt.hud_scale = std::max(1, std::atoi(require_value("--hud-scale")));
        } else if (arg == "--show-detector-input") {
            opt.show_detector_input = true;
        } else if (arg == "--no-preview") {
            opt.no_preview = true;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char* argv[])
{
    using namespace tactical_rescue;

    LineFilter stdout_filter(STDOUT_FILENO);
    LineFilter stderr_filter(STDERR_FILENO);
    stdout_filter.start();
    stderr_filter.start();

    Options options;
    if (!parse_args(argc, argv, options)) {
        return 0;
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

    int actual_range = options.range_mm;
    if (options.range_mm != kDefaultRangeMm) {
        if (tof.setControl(Control::RANGE, options.range_mm) == ArducamSuccess) {
            tof.getControl(Control::RANGE, &actual_range);
        }
    } else {
        tof.getControl(Control::RANGE, &actual_range);
    }
    options.max_depth_mm = std::min(options.max_depth_mm, actual_range > 0 ? actual_range : options.range_mm);

    auto info = tof.getCameraInfo();
    std::cout << "Tactical rescue feed active at " << info.width << "x" << info.height << " range " << actual_range
              << "mm" << std::endl;

    DetectorSource active_detector_source = DetectorSource::CONFIDENCE;
    if (options.detector_source == DetectorSource::CONFIDENCE) {
        active_detector_source = DetectorSource::CONFIDENCE;
    } else if (options.detector_source == DetectorSource::PSEUDO) {
        active_detector_source = DetectorSource::PSEUDO;
    } else if (options.detector_source == DetectorSource::AUTO) {
        active_detector_source = DetectorSource::CONFIDENCE;
    } else {
        active_detector_source = DetectorSource::AMPLITUDE;
    }

    if (active_detector_source == DetectorSource::CONFIDENCE) {
        std::cout << "Detector input using ToF confidence heuristic" << std::endl;
    } else if (active_detector_source == DetectorSource::PSEUDO) {
        std::cout << "Detector input using ToF pseudo composite heuristic" << std::endl;
    } else {
        std::cout << "Detector input using ToF amplitude heuristic" << std::endl;
    }

    std::mutex frame_mutex;
    std::condition_variable frame_cv;
    SharedFrame latest_frame;
    DetectionState latest_detection;
    std::mutex detection_mutex;
    cv::Mat latest_detector_input;
    std::mutex detector_input_mutex;
    RuntimeStats stats;
    std::mutex stats_mutex;
    std::atomic<bool> running{true};
    std::atomic<uint64_t> published_sequence{0};

    {
        std::lock_guard<std::mutex> lock(stats_mutex);
        stats.ambient_enabled = options.enable_am2302;
        stats.ambient_status = options.enable_am2302 ? "INIT" : "DISABLED";
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

    std::thread ambient_thread;
    if (options.enable_am2302) {
        ambient_thread = std::thread([&] {
            HelperProcess helper;
            if (!start_am2302_helper(options, helper)) {
                std::lock_guard<std::mutex> lock(stats_mutex);
                stats.ambient_enabled = false;
                stats.ambient_status = "ERROR";
                return;
            }

            char buffer[256];
            while (running && std::fgets(buffer, sizeof(buffer), helper.stream) != nullptr) {
                std::string line(buffer);
                while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
                    line.pop_back();
                }
                if (line.empty()) {
                    continue;
                }

                std::string status;
                float temperature_c = 0.0f;
                float humidity_percent = 0.0f;
                if (!parse_am2302_line(line, status, temperature_c, humidity_percent)) {
                    continue;
                }

                std::lock_guard<std::mutex> lock(stats_mutex);
                stats.ambient_enabled = true;
                stats.ambient_status = status;
                if (status == "GOOD") {
                    stats.ambient_valid = true;
                    stats.ambient_temperature_c = temperature_c;
                    stats.ambient_humidity_percent = humidity_percent;
                    stats.ambient_updated_at = std::chrono::steady_clock::now();
                    stats.ambient_age_s = 0.0;
                }
            }

            stop_helper_process(helper);
        });
    }

    std::thread inference_thread([&] {
        uint64_t consumed_frame = 0;
        auto last_inference_started = std::chrono::steady_clock::time_point::min();
        while (running) {
            SharedFrame lidar_input;
            {
                std::unique_lock<std::mutex> lock(frame_mutex);
                frame_cv.wait(lock, [&] { return !running || published_sequence.load() != consumed_frame; });
                if (!running) {
                    break;
                }
                lidar_input = latest_frame;
            }

            if (lidar_input.depth_mm.empty() || lidar_input.sequence == consumed_frame) {
                continue;
            }
            consumed_frame = lidar_input.sequence;

            const auto now = std::chrono::steady_clock::now();
            if (last_inference_started != std::chrono::steady_clock::time_point::min()) {
                const auto min_period = std::chrono::milliseconds(1000 / std::max(1, options.detection_fps));
                if (now - last_inference_started < min_period) {
                    continue;
                }
            }

            if (options.show_detector_input) {
                cv::Mat detector_input;
                if (active_detector_source == DetectorSource::CONFIDENCE) {
                    detector_input = build_confidence_detector_input(lidar_input, options);
                } else if (active_detector_source == DetectorSource::PSEUDO) {
                    detector_input = build_pseudo_detector_input(lidar_input, options);
                } else {
                    detector_input = build_amplitude_detector_input(lidar_input, options);
                }
                std::lock_guard<std::mutex> lock(detector_input_mutex);
                latest_detector_input = detector_input.clone();
            }

            const auto infer_started = std::chrono::steady_clock::now();
            last_inference_started = infer_started;
            float best_person_score = 0.0f;
            const DetectionState result =
                run_tof_person_detector(lidar_input, active_detector_source, options, &best_person_score);

            {
                std::lock_guard<std::mutex> lock(detection_mutex);
                latest_detection = result;
            }

            const auto infer_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - infer_started).count();
            std::lock_guard<std::mutex> lock(stats_mutex);
            stats.inference_ms = infer_ms;
            stats.detected_people = static_cast<int>(result.people.size());
            stats.best_person_score = best_person_score;
            stats.detector_uses_segmentation = false;
            stats.detector_source_label = std::string(detector_source_label(active_detector_source)) + " TOF";
        }
    });

    if (!options.no_preview) {
        cv::namedWindow("tactical_rescue", cv::WINDOW_NORMAL);
        cv::resizeWindow("tactical_rescue", kDisplayWidth, kDisplayHeight);
        if (options.show_detector_input) {
            cv::namedWindow("detector_input", cv::WINDOW_NORMAL);
            cv::resizeWindow("detector_input", 900, 700);
        }
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
            if (stats.ambient_updated_at != std::chrono::steady_clock::time_point{}) {
                stats.ambient_age_s =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - stats.ambient_updated_at).count();
            }
            local_stats = stats;
        }
        cv::Mat display = compose_display_canvas(hud);
        draw_hud(display, local_stats, detections, true, options.hud_scale);

        if (!options.no_preview) {
            cv::imshow("tactical_rescue", display);
            if (options.show_detector_input) {
                cv::Mat detector_preview;
                {
                    std::lock_guard<std::mutex> lock(detector_input_mutex);
                    detector_preview = latest_detector_input.clone();
                }
                if (!detector_preview.empty()) {
                    cv::Mat detector_canvas;
                    cv::resize(detector_preview, detector_canvas, cv::Size(900, 700), 0.0, 0.0, cv::INTER_NEAREST);
                    cv::putText(detector_canvas, "TOF INPUT " + std::string(detector_source_label(active_detector_source)),
                                cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 215, 255), 2,
                                cv::LINE_AA);
                    cv::putText(detector_canvas,
                                "best person score: " +
                                    std::to_string(static_cast<int>(std::round(local_stats.best_person_score * 100))) +
                                    "%",
                                cv::Point(12, 58), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 2,
                                cv::LINE_AA);
                    cv::imshow("detector_input", detector_canvas);
                }
            }
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

    if (capture_thread.joinable()) {
        capture_thread.join();
    }
    if (ambient_thread.joinable()) {
        ambient_thread.join();
    }
    if (inference_thread.joinable()) {
        inference_thread.join();
    }

    if (tof.stop()) {
        std::cerr << "Failed to stop camera" << std::endl;
    }
    if (tof.close()) {
        std::cerr << "Failed to close camera" << std::endl;
    }

    stdout_filter.stop();
    stderr_filter.stop();
    return 0;
}
