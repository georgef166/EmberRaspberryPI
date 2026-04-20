#include "tactical_rescue.hpp"

#include <cstdio>
#include <fstream>

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

} // namespace tactical_rescue
