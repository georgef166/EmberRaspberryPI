#include "tactical_rescue.hpp"

#include <cstdio>
#include <fstream>

#include <opencv2/core/utils/logger.hpp>

#include <unistd.h>

namespace tactical_rescue {

LineFilter::LineFilter(int target_fd) : target_fd_(target_fd) {}

LineFilter::~LineFilter()
{
    stop();
}

bool LineFilter::start()
{
    if (active_) {
        return true;
    }
    if (::pipe(pipe_fds_) != 0) {
        return false;
    }

    original_fd_ = ::dup(target_fd_);
    if (original_fd_ < 0) {
        ::close(pipe_fds_[0]);
        ::close(pipe_fds_[1]);
        pipe_fds_[0] = -1;
        pipe_fds_[1] = -1;
        return false;
    }

    if (::dup2(pipe_fds_[1], target_fd_) < 0) {
        ::close(original_fd_);
        ::close(pipe_fds_[0]);
        ::close(pipe_fds_[1]);
        original_fd_ = -1;
        pipe_fds_[0] = -1;
        pipe_fds_[1] = -1;
        return false;
    }

    ::close(pipe_fds_[1]);
    pipe_fds_[1] = -1;
    active_ = true;
    worker_ = std::thread([this] { pump(); });
    return true;
}

void LineFilter::stop()
{
    if (!active_) {
        return;
    }

    if (target_fd_ == STDERR_FILENO) {
        std::fflush(stderr);
    } else if (target_fd_ == STDOUT_FILENO) {
        std::fflush(stdout);
    }
    if (original_fd_ >= 0) {
        ::dup2(original_fd_, target_fd_);
    }
    if (pipe_fds_[0] >= 0) {
        ::close(pipe_fds_[0]);
        pipe_fds_[0] = -1;
    }
    if (worker_.joinable()) {
        worker_.join();
    }
    if (original_fd_ >= 0) {
        ::close(original_fd_);
        original_fd_ = -1;
    }
    active_ = false;
}

bool LineFilter::should_suppress(const std::string& line)
{
    return line.find("UVC: ioctl (") != std::string::npos ||
           line.find("gst_caps_get_structure: assertion 'GST_IS_CAPS (caps)' failed") != std::string::npos ||
           line.find("gst_structure_get_int: assertion 'structure != NULL' failed") != std::string::npos ||
           line.find("gst_structure_get_fraction: assertion 'structure != NULL' failed") != std::string::npos ||
           line.find("OpenCV | GStreamer warning: cannot query video width/height") != std::string::npos ||
           line.find("OpenCV | GStreamer warning: cannot query video fps") != std::string::npos ||
           line.find("OpenCV | GStreamer warning: Cannot query video position") != std::string::npos ||
           line.find("g_object_unref: assertion 'G_IS_OBJECT (object)' failed") != std::string::npos;
}

void LineFilter::flush_line(const std::string& line)
{
    if (line.empty() || should_suppress(line) || original_fd_ < 0) {
        return;
    }
    const std::string with_newline = line + "\n";
    (void)::write(original_fd_, with_newline.c_str(), with_newline.size());
}

void LineFilter::pump()
{
    std::string pending;
    char buffer[512];
    while (true) {
        const ssize_t bytes_read = ::read(pipe_fds_[0], buffer, sizeof(buffer));
        if (bytes_read <= 0) {
            break;
        }

        pending.append(buffer, static_cast<size_t>(bytes_read));
        size_t newline = pending.find('\n');
        while (newline != std::string::npos) {
            flush_line(pending.substr(0, newline));
            pending.erase(0, newline + 1);
            newline = pending.find('\n');
        }
    }

    if (!pending.empty()) {
        flush_line(pending);
    }
}

bool file_exists(const std::string& path)
{
    std::ifstream file(path);
    return file.good();
}

namespace {

std::string read_text_file(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        return {};
    }
    std::string value;
    std::getline(file, value);
    return value;
}

bool looks_like_rgb_video_node(int device_index)
{
    const std::string name_path = "/sys/class/video4linux/video" + std::to_string(device_index) + "/name";
    const std::string name = read_text_file(name_path);
    if (name.empty()) {
        return true;
    }

    if (name.find("unicam-image") != std::string::npos) {
        return false;
    }
    if (name.find("bcm2835") != std::string::npos) {
        return false;
    }
    if (name.find("codec") != std::string::npos) {
        return false;
    }
    if (name.find("isp") != std::string::npos) {
        return false;
    }
    return true;
}

bool try_open_v4l2_capture(cv::VideoCapture& capture, int device_index, const Options& opt, std::string& opened_path)
{
    capture.open(device_index, cv::CAP_V4L2);
    if (!capture.isOpened()) {
        return false;
    }

    capture.set(cv::CAP_PROP_FRAME_WIDTH, opt.rgb_width);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, opt.rgb_height);
    capture.set(cv::CAP_PROP_BUFFERSIZE, 1);
    opened_path = "/dev/video" + std::to_string(device_index);
    return true;
}

} // namespace

bool open_rgb_capture(cv::VideoCapture& capture, const Options& opt, std::string& opened_path)
{
    if (looks_like_rgb_video_node(opt.rgb_device)) {
        if (try_open_v4l2_capture(capture, opt.rgb_device, opt, opened_path)) {
            return true;
        }
    }

    if (!opt.rgb_libcamera) {
        return false;
    }

    const std::string pipeline =
        "libcamerasrc ! video/x-raw,width=" + std::to_string(opt.rgb_width) + ",height=" +
        std::to_string(opt.rgb_height) + ",framerate=30/1 ! videoconvert ! appsink drop=1 sync=false";

    const auto previous_log_level = cv::utils::logging::getLogLevel();
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    capture.open(pipeline, cv::CAP_GSTREAMER);
    cv::utils::logging::setLogLevel(previous_log_level);
    if (capture.isOpened()) {
        opened_path = "libcamerasrc";
        return true;
    }

    return false;
}

} // namespace tactical_rescue
