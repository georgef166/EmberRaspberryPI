#include "ArducamTOFCamera.hpp"
#include "tactical_rescue.hpp"
#include "tactical_rescue_stream.hpp"
#include "tactical_rescue_tflite.hpp"
#include "tactical_rescue_thermal.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdlib>
#include <csignal>
#include <cstdio>
#include <iomanip>
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
        if (value == "tflite") {
            return DetectorSource::TFLITE;
        }
        if (value == "thermal") {
            return DetectorSource::THERMAL;
        }
        std::cerr << "Unknown detector source: " << value << std::endl;
        std::exit(2);
    };

    auto parse_tflite_input_mode = [](const std::string& value) {
        if (value == "amplitude" || value == "amplitude-debug") {
            return TfliteInputMode::AMPLITUDE;
        }
        if (value == "fused") {
            return TfliteInputMode::FUSED;
        }
        if (value == "pseudo") {
            return TfliteInputMode::PSEUDO;
        }
        if (value == "depth") {
            return TfliteInputMode::DEPTH;
        }
        std::cerr << "Unknown TFLite input mode: " << value << std::endl;
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
                << "  --min-depth MM           Ignore geometry nearer than this (default 50)\n"
                << "  --max-depth MM           Ignore geometry farther than this\n"
                << "  --confidence NUM         Depth confidence threshold (default 8)\n"
                << "  --person-conf FLOAT      Human candidate confidence threshold (default 0.50)\n"
                << "  --max-people NUM         Maximum rendered person detections\n"
                << "  --min-person-pixels NUM  Minimum rendered person box area (default 900)\n"
                << "  --detection-fps NUM      Detector cadence in frames per second\n"
                << "  --detector-source MODE   auto | amplitude | confidence | pseudo | tflite | thermal\n"
                << "  --tflite-input MODE      fused | pseudo | depth | amplitude-debug (default fused)\n"
                << "  --tflite-model PATH      Path to TFLite SSD person detection model\n"
                << "  --edgetpu, --coral       Run TFLite inference on the Coral Edge TPU\n"
                << "  --person-class NUM       Model output index for 'person' (default 1; Coral default 0)\n"
                << "  --no-thermal             Disable the MLX90640 thermal overlay/detector\n"
                << "  --thermal-address HEX    MLX90640 I2C address (default 0x33)\n"
                << "  --thermal-refresh HZ     MLX90640 subpage rate, 1|2|4|8 (default 4; two subpages per grid)\n"
                << "  --emissivity FLOAT       Thermal emissivity (default 0.95)\n"
                << "  --fire-temp C            Hotspot/fire warning threshold in C (default 60)\n"
                << "  --victim-temp-min C      Warm-body band lower bound (default 26)\n"
                << "  --victim-temp-max C      Warm-body band upper bound (default 45)\n"
                << "  --am2302-gpio NUM        BCM GPIO used for AM2302 data (default 4)\n"
                << "  --no-am2302              Disable AM2302 ambient overlay\n"
                << "  --stream                 Serve the rendered HUD over HTTP MJPEG\n"
                << "  --stream-bind ADDR       Stream bind address (default 0.0.0.0)\n"
                << "  --stream-port NUM        Stream HTTP port (default 8080)\n"
                << "  --stream-fps NUM         Stream frame rate cap (default 10)\n"
                << "  --stream-quality NUM     Stream JPEG quality 20-95 (default 75)\n"
                << "  --stream-password PASS   Commander view password (default 'admin')\n"
                << "  --no-stream-auth         Disable the commander view password gate\n"
                << "  --ground                 Enable ground plane detection and the AR grid\n"
                << "  --no-ground              Disable ground plane detection (the default)\n"
                << "  --ground-fps NUM         Ground plane fit cadence (default 10)\n"
                << "  --ground-stride NUM      Depth subsample stride for the fit (default 4)\n"
                << "  --ground-max-tilt DEG    Max floor tilt vs camera up (default 45)\n"
                << "  --grid-spacing MM        AR grid cell size (default 500)\n"
                << "  --grid-extent MM         AR grid half-width (default 3000)\n"
                << "  --intrinsics FX,FY,CX,CY ToF intrinsics (default 230.5,230.5,120,90)\n"
                << "  --depth-radial           Depth frame is radial slant range, not Z depth\n"
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
        } else if (arg == "--min-person-pixels") {
            opt.min_person_box_pixels = std::max(1, std::atoi(require_value("--min-person-pixels")));
        } else if (arg == "--detection-fps") {
            opt.detection_fps = std::max(1, std::atoi(require_value("--detection-fps")));
            opt.detection_fps_explicit = true;
        } else if (arg == "--detector-source") {
            opt.detector_source = parse_detector_source(require_value("--detector-source"));
        } else if (arg == "--tflite-input") {
            opt.tflite_input_mode = parse_tflite_input_mode(require_value("--tflite-input"));
        } else if (arg == "--tflite-model") {
            opt.tflite_model_path = require_value("--tflite-model");
            opt.tflite_model_explicit = true;
        } else if (arg == "--edgetpu" || arg == "--coral") {
            opt.edgetpu = true;
        } else if (arg == "--person-class") {
            opt.person_class_id = std::max(0, std::atoi(require_value("--person-class")));
            opt.person_class_explicit = true;
        } else if (arg == "--no-thermal") {
            opt.enable_thermal = false;
        } else if (arg == "--thermal-address") {
            opt.thermal_address = static_cast<int>(std::strtol(require_value("--thermal-address"), nullptr, 0));
        } else if (arg == "--thermal-refresh") {
            // Capped at 8: above that the bus misses deadlines and a torn frame
            // reads as spurious extreme pixels, i.e. a false fire warning.
            opt.thermal_refresh_hz = std::max(1, std::min(8, std::atoi(require_value("--thermal-refresh"))));
        } else if (arg == "--emissivity") {
            opt.thermal_emissivity = std::atof(require_value("--emissivity"));
        } else if (arg == "--fire-temp") {
            opt.fire_temp_c = std::atof(require_value("--fire-temp"));
        } else if (arg == "--victim-temp-min") {
            opt.victim_temp_min_c = std::atof(require_value("--victim-temp-min"));
        } else if (arg == "--victim-temp-max") {
            opt.victim_temp_max_c = std::atof(require_value("--victim-temp-max"));
        } else if (arg == "--am2302-gpio") {
            opt.am2302_gpio = std::max(0, std::atoi(require_value("--am2302-gpio")));
        } else if (arg == "--no-am2302") {
            opt.enable_am2302 = false;
        } else if (arg == "--stream") {
            opt.enable_stream = true;
        } else if (arg == "--stream-bind") {
            opt.stream_bind_address = require_value("--stream-bind");
        } else if (arg == "--stream-port") {
            opt.stream_port = std::max(1, std::atoi(require_value("--stream-port")));
        } else if (arg == "--stream-fps") {
            opt.stream_fps = std::max(1, std::atoi(require_value("--stream-fps")));
        } else if (arg == "--stream-quality") {
            opt.stream_jpeg_quality = std::max(20, std::min(95, std::atoi(require_value("--stream-quality"))));
        } else if (arg == "--stream-password") {
            opt.stream_password = require_value("--stream-password");
        } else if (arg == "--no-stream-auth") {
            opt.stream_auth_enabled = false;
        } else if (arg == "--hud-scale") {
            opt.hud_scale = std::max(1, std::atoi(require_value("--hud-scale")));
        } else if (arg == "--ground") {
            opt.enable_ground_plane = true;
        } else if (arg == "--no-ground") {
            opt.enable_ground_plane = false;
        } else if (arg == "--ground-fps") {
            opt.ground_fps = std::max(1, std::atoi(require_value("--ground-fps")));
        } else if (arg == "--ground-stride") {
            opt.ground_stride = std::max(1, std::atoi(require_value("--ground-stride")));
        } else if (arg == "--ground-max-tilt") {
            opt.ground_max_tilt_deg = static_cast<float>(std::atof(require_value("--ground-max-tilt")));
        } else if (arg == "--grid-spacing") {
            opt.grid_spacing_mm = static_cast<float>(std::atof(require_value("--grid-spacing")));
        } else if (arg == "--grid-extent") {
            opt.grid_half_width_mm = static_cast<float>(std::atof(require_value("--grid-extent")));
        } else if (arg == "--depth-radial") {
            opt.depth_is_radial = true;
        } else if (arg == "--intrinsics") {
            const char* value = require_value("--intrinsics");
            if (std::sscanf(value, "%f,%f,%f,%f", &opt.fx, &opt.fy, &opt.cx, &opt.cy) != 4) {
                std::cerr << "Expected --intrinsics FX,FY,CX,CY" << std::endl;
                std::exit(2);
            }
            opt.intrinsics_explicit = true;
        } else if (arg == "--show-detector-input") {
            opt.show_detector_input = true;
        } else if (arg == "--no-preview") {
            opt.no_preview = true;
        } else if (arg == "--selfcheck") {
            // Dev-only, needs no hardware: asserts the thermal pure helpers and quits.
            thermal_selfcheck();
            return false;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return false;
        }
    }
    return true;
}

void apply_backend_defaults(Options& opt)
{
    if (!opt.edgetpu) {
        return;
    }

    if (!opt.tflite_model_explicit) {
        opt.tflite_model_path = kDefaultEdgeTpuModelPath;
    }
    if (!opt.person_class_explicit) {
        opt.person_class_id = kDefaultEdgeTpuPersonClassId;
    }
    if (!opt.detection_fps_explicit) {
        opt.detection_fps = kDefaultEdgeTpuDetectionFps;
    }
}

} // namespace

int main(int argc, char* argv[])
{
    using namespace tactical_rescue;

    Options options;
    if (!parse_args(argc, argv, options)) {
        return 0;
    }
    apply_backend_defaults(options);

    LineFilter stdout_filter(STDOUT_FILENO);
    LineFilter stderr_filter(STDERR_FILENO);
    stdout_filter.start();
    stderr_filter.start();

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
    if (tof.setControl(Control::RANGE, options.range_mm) == ArducamSuccess) {
        tof.getControl(Control::RANGE, &actual_range);
    } else {
        tof.getControl(Control::RANGE, &actual_range);
        std::cerr << "Warning: failed to set ToF range to " << options.range_mm << "mm; camera reports "
                  << actual_range << "mm" << std::endl;
    }
    options.max_depth_mm = std::min(options.max_depth_mm, actual_range > 0 ? actual_range : options.range_mm);

    auto info = tof.getCameraInfo();
    scale_intrinsics_to_frame(options, info.width, info.height);
    std::cout << "Tactical rescue feed active at " << info.width << "x" << info.height << " range " << actual_range
              << "mm" << std::endl;

    // --- GOOGLE TECHNOLOGY: TensorFlow Lite ---
    // Load the TFLite MobileNet SSD person detection model at startup.
    // If models/detect.tflite is present, AUTO mode activates TFLite as the
    // primary detector. Falls back to classical CV heuristics if absent.
    // See tactical_rescue_tflite.cpp for the full inference pipeline.
    TFLitePersonDetector tflite_detector;
    const bool tflite_model_present = file_exists(options.tflite_model_path);
    if (options.edgetpu && !tflite_model_present) {
        std::cerr << "[Coral] Edge TPU requested, but model is missing: " << options.tflite_model_path << std::endl;
        std::cerr << "[Coral] Run ./Install_coral.sh, then ./run.sh --rebuild --coral." << std::endl;
    }
    const bool tflite_available =
        tflite_model_present && tflite_detector.load(options.tflite_model_path, options.edgetpu);
    if (options.edgetpu && !tflite_available) {
        std::cerr << "[Coral] Edge TPU detector unavailable; AUTO mode will fall back to the ToF heuristic."
                  << std::endl;
    }

    // --- MLX90640 THERMAL ---
    // Open the thermal sensor. The thermal overlay and fire/hotspot warning
    // render whenever the sensor is present, independent of which person
    // detector is active. Selectable as a detector via --detector-source thermal.
    Mlx90640Camera thermal_cam;
    const bool thermal_available = options.enable_thermal && thermal_cam.open(options);

    DetectorSource active_detector_source = DetectorSource::CONFIDENCE;
    if (options.detector_source == DetectorSource::TFLITE) {
        active_detector_source = DetectorSource::TFLITE;
    } else if (options.detector_source == DetectorSource::THERMAL) {
        active_detector_source = DetectorSource::THERMAL;
    } else if (options.detector_source == DetectorSource::CONFIDENCE) {
        active_detector_source = DetectorSource::CONFIDENCE;
    } else if (options.detector_source == DetectorSource::PSEUDO) {
        active_detector_source = DetectorSource::PSEUDO;
    } else if (options.detector_source == DetectorSource::AUTO) {
        // AUTO preference: TFLite (Coral/CPU) when a model is loaded, else the CV
        // heuristic. Thermal is deliberately NOT in this chain — it scores blobs
        // relative to the grid mean, so in a hot room (the fire case) it returns
        // zero detections. Still selectable with --detector-source thermal.
        active_detector_source = tflite_available ? DetectorSource::TFLITE : DetectorSource::CONFIDENCE;
    } else {
        active_detector_source = DetectorSource::AMPLITUDE;
    }

    if (active_detector_source == DetectorSource::THERMAL && !thermal_available) {
        std::cerr << "Detector: thermal requested but MLX90640 unavailable — falling back to ToF confidence heuristic"
                  << std::endl;
        active_detector_source = DetectorSource::CONFIDENCE;
    }

    if (active_detector_source == DetectorSource::TFLITE) {
        std::cout << "Detector: TensorFlow Lite SSD person detection on "
                  << (tflite_detector.uses_edgetpu() ? "Coral Edge TPU" : "Raspberry Pi CPU")
                  << std::endl;
    } else if (active_detector_source == DetectorSource::THERMAL) {
        std::cout << "Detector: MLX90640 thermal warm-body detection" << std::endl;
    } else if (active_detector_source == DetectorSource::CONFIDENCE) {
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
    ThermalFrame latest_thermal;
    std::mutex thermal_mutex;
    GroundPlaneState latest_ground;
    std::mutex ground_mutex;
    RuntimeStats stats;
    std::mutex stats_mutex;
    std::atomic<bool> running{true};
    std::atomic<uint64_t> published_sequence{0};

    {
        std::lock_guard<std::mutex> lock(stats_mutex);
        stats.ambient_enabled = options.enable_am2302;
        stats.ambient_status = options.enable_am2302 ? "INIT" : "DISABLED";
        stats.thermal_enabled = thermal_available;
        stats.thermal_status = thermal_available ? "INIT" : (options.enable_thermal ? "OFFLINE" : "DISABLED");
    }

    FpsCounter capture_fps;
    FpsCounter render_fps;
    MjpegStreamServer stream_server;
    if (options.enable_stream &&
        !stream_server.start(options.stream_bind_address, options.stream_port, options.stream_jpeg_quality,
                             options.stream_auth_enabled, options.stream_password)) {
        std::cerr << "[stream] Remote feed disabled." << std::endl;
        options.enable_stream = false;
    }

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

            if (running) {
                std::lock_guard<std::mutex> lock(stats_mutex);
                stats.ambient_status = "ERROR";
                stats.ambient_valid = false;
            }

            stop_helper_process(helper);
        });
    }

    std::thread thermal_thread;
    if (thermal_available) {
        thermal_thread = std::thread([&] {
            const auto read_delay = std::chrono::milliseconds(thermal_read_delay_ms(options.thermal_refresh_hz));
            while (running) {
                // Wait out most of the subpage period here; without this the
                // vendor's untimed busy-poll inside read() eats a whole core.
                std::this_thread::sleep_for(read_delay);

                ThermalFrame local;
                if (!thermal_cam.read(local)) {
                    // Drop the last grid as well, or the overlay and the FIRE
                    // banner keep painting a dead sensor's final frame forever.
                    {
                        std::lock_guard<std::mutex> lock(thermal_mutex);
                        latest_thermal.valid = false;
                    }
                    {
                        std::lock_guard<std::mutex> lock(stats_mutex);
                        stats.thermal_status = "ERROR";
                        stats.thermal_valid = false;
                        stats.fire_warning = false;
                    }
                    // Sleep outside the locks: stats_mutex is contended by the
                    // capture, ambient, inference and render loops.
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    continue;
                }

                {
                    std::lock_guard<std::mutex> lock(thermal_mutex);
                    latest_thermal = local;
                }

                std::lock_guard<std::mutex> lock(stats_mutex);
                stats.thermal_valid = true;
                stats.thermal_status = "GOOD";
                stats.thermal_max_c = local.max_c;
                stats.thermal_min_c = local.min_c;
                stats.thermal_mean_c = local.mean_c;
                stats.fire_warning = local.max_c >= options.fire_temp_c;
                stats.thermal_updated_at = std::chrono::steady_clock::now();
                stats.thermal_age_s = 0.0;
            }
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
                if (active_detector_source == DetectorSource::TFLITE) {
                    if (options.tflite_input_mode == TfliteInputMode::FUSED) {
                        detector_input = build_fused_detector_input(lidar_input, options);
                    } else if (options.tflite_input_mode == TfliteInputMode::PSEUDO) {
                        detector_input = build_pseudo_detector_input(lidar_input, options);
                    } else if (options.tflite_input_mode == TfliteInputMode::DEPTH) {
                        cv::Mat depth_gray = to_gray_preview(lidar_input.depth_mm, options);
                        if (!lidar_input.confidence.empty()) {
                            cv::Mat valid = build_geometry_mask(lidar_input.depth_mm, lidar_input.confidence, options);
                            depth_gray.setTo(0, valid == 0);
                        }
                        cv::cvtColor(depth_gray, detector_input, cv::COLOR_GRAY2BGR);
                    } else {
                        detector_input = build_amplitude_detector_input(lidar_input, options);
                    }
                    if (detector_input.empty() && options.tflite_input_mode != TfliteInputMode::DEPTH) {
                        cv::Mat depth_gray = to_gray_preview(lidar_input.depth_mm, options);
                        if (!lidar_input.confidence.empty()) {
                            cv::Mat valid = build_geometry_mask(lidar_input.depth_mm, lidar_input.confidence, options);
                            depth_gray.setTo(0, valid == 0);
                        }
                        cv::cvtColor(depth_gray, detector_input, cv::COLOR_GRAY2BGR);
                    }
                } else if (active_detector_source == DetectorSource::CONFIDENCE) {
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
            // Route to TFLite (Coral/CPU), MLX90640 thermal warm-body detection,
            // or the classical ToF CV heuristic.
            DetectionState result;
            if (active_detector_source == DetectorSource::TFLITE) {
                result = tflite_detector.detect(lidar_input, options, &best_person_score);
            } else if (active_detector_source == DetectorSource::THERMAL) {
                ThermalFrame thermal_input;
                {
                    std::lock_guard<std::mutex> lock(thermal_mutex);
                    thermal_input = latest_thermal;
                }
                result = detect_thermal_people(thermal_input, lidar_input, options, &best_person_score);
            } else {
                result = run_tof_person_detector(lidar_input, active_detector_source, options, &best_person_score);
            }

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
            if (active_detector_source == DetectorSource::TFLITE) {
                stats.detector_source_label = tflite_detector.uses_edgetpu() ? "CORAL TFLITE" : "CPU TFLITE";
            } else if (active_detector_source == DetectorSource::THERMAL) {
                stats.detector_source_label = "THERMAL";
            } else {
                stats.detector_source_label = std::string(detector_source_label(active_detector_source)) + " TOF";
            }
        }
    });

    std::thread ground_thread;
    if (options.enable_ground_plane) {
        ground_thread = std::thread([&] {
            GroundPlaneTracker tracker;
            uint64_t consumed_frame = 0;
            auto last_fit_started = std::chrono::steady_clock::time_point::min();
            auto last_debug_print = std::chrono::steady_clock::time_point::min();
            std::string last_debug_status;
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
                if (last_fit_started != std::chrono::steady_clock::time_point::min()) {
                    const auto min_period = std::chrono::milliseconds(1000 / std::max(1, options.ground_fps));
                    if (now - last_fit_started < min_period) {
                        continue;
                    }
                }
                last_fit_started = now;

                GroundPlaneState state = tracker.update(lidar_input, options);
                state.source_sequence = lidar_input.sequence;
                state.fit_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - now).count();

                {
                    std::lock_guard<std::mutex> lock(ground_mutex);
                    latest_ground = state;
                }

                // --- TEMPORARY bring-up instrumentation ---
                // One [ground] line on status change, otherwise at most 1 Hz (see
                // PI_HANDOFF.md step 4). Delete this block once the fit is validated.
                const char* status = state.stale ? "STALE" : (state.plane.valid ? "LOCK" : "NONE");
                const auto printed_at = std::chrono::steady_clock::now();
                if (last_debug_status != status || printed_at - last_debug_print >= std::chrono::seconds(1)) {
                    last_debug_status = status;
                    last_debug_print = printed_at;
                    std::ostringstream line;
                    line << std::fixed << "[ground] " << std::left << std::setw(6) << status << std::right
                         << "  height " << std::setprecision(0) << state.plane.height_mm << "mm"
                         << "  tilt " << std::setprecision(1) << state.plane.tilt_deg << "deg"
                         << "  inliers " << state.plane.inlier_count
                         << " (" << std::setprecision(0) << state.plane.inlier_ratio * 100.0f << "%)"
                         << "  fit " << std::setprecision(2) << state.fit_ms << "ms";
                    std::cerr << line.str() << std::endl;
                }
                // --- end TEMPORARY ---
            }
        });
    }

    if (!options.no_preview) {
        cv::namedWindow("tactical_rescue", cv::WINDOW_NORMAL);
        cv::setWindowProperty("tactical_rescue", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
        if (options.show_detector_input) {
            cv::namedWindow("detector_input", cv::WINDOW_NORMAL);
            cv::resizeWindow("detector_input", 900, 700);
        }
    }

    auto last_stream_publish = std::chrono::steady_clock::time_point::min();
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
        cv::Mat hud = build_wireframe_overlay(frame, options, nearest_mm);

        // Thermal heat-map overlay on the wireframe (warm regions only), drawn
        // before person boxes so detections sit on top.
        if (thermal_available) {
            ThermalFrame thermal_snapshot;
            {
                std::lock_guard<std::mutex> lock(thermal_mutex);
                thermal_snapshot = latest_thermal;
            }
            draw_thermal_overlay(hud, thermal_snapshot, options);
        }

        DetectionState detections;
        {
            std::lock_guard<std::mutex> lock(detection_mutex);
            detections = latest_detection;
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
            if (stats.thermal_updated_at != std::chrono::steady_clock::time_point{}) {
                stats.thermal_age_s =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - stats.thermal_updated_at).count();
            }
            local_stats = stats;
        }
        cv::Mat display = compose_display_canvas(hud);
        // Ground plane grid first: it is world geometry and belongs UNDER the
        // victim boxes, which are the alert layer the operator must never lose.
        if (options.enable_ground_plane) {
            GroundPlaneState ground_snapshot;
            {
                std::lock_guard<std::mutex> lock(ground_mutex);
                ground_snapshot = latest_ground;
            }
            draw_ground_grid(display, ground_snapshot, frame, compute_canvas_transform(hud.size()), options);
        }
        // Victim boxes are drawn on the full-resolution canvas, not on the
        // 240x180 ToF frame - drawing them before the upscale magnified the
        // label ~6x and turned the outline into chunky blocks.
        if (detections.valid && detections.source_sequence <= frame.sequence) {
            double det_scale = 1.0;
            cv::Point det_offset(0, 0);
            display_canvas_transform(hud.size(), det_scale, det_offset);
            const int render_count = std::min(static_cast<int>(detections.people.size()), kMaxRenderedPeople);
            for (int i = 0; i < render_count; ++i) {
                draw_person_detection(display, detections.people[i], i, det_scale, det_offset);
            }
        }
        draw_hud(display, local_stats, detections, true, options.hud_scale);
        // Commander markup (doors, breadcrumb trail, arrows) is burned into the
        // shared display so it appears on both the remote console and the
        // firefighter's local HUD.
        if (options.enable_stream) {
            draw_stream_annotations(display, stream_server.annotations());
            const auto now = std::chrono::steady_clock::now();
            const auto stream_period = std::chrono::milliseconds(1000 / std::max(1, options.stream_fps));
            if (last_stream_publish == std::chrono::steady_clock::time_point::min() ||
                now - last_stream_publish >= stream_period) {
                stream_server.publish(display);
                last_stream_publish = now;
            }
        }

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
    stream_server.stop();

    if (capture_thread.joinable()) {
        capture_thread.join();
    }
    if (ambient_thread.joinable()) {
        ambient_thread.join();
    }
    if (thermal_thread.joinable()) {
        thermal_thread.join();
    }
    if (inference_thread.joinable()) {
        inference_thread.join();
    }
    if (ground_thread.joinable()) {
        ground_thread.join();
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
