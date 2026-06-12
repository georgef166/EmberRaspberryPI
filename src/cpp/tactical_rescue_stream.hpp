#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

namespace tactical_rescue {

class MjpegStreamServer {
public:
    MjpegStreamServer();
    ~MjpegStreamServer();

    bool start(const std::string& bind_address, int port, int jpeg_quality);
    void stop();
    void publish(const cv::Mat& frame);

private:
    void accept_loop();
    void client_loop(int client_fd);
    bool write_all(int fd, const void* data, size_t size) const;
    bool write_string(int fd, const std::string& text) const;

    std::atomic<bool> running_{false};
    int server_fd_ = -1;
    std::string bind_address_ = "0.0.0.0";
    int port_ = 8080;
    int jpeg_quality_ = 75;
    std::thread accept_thread_;
    std::mutex client_mutex_;
    std::vector<std::thread> client_threads_;

    std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    std::vector<unsigned char> latest_jpeg_;
    uint64_t sequence_ = 0;
};

} // namespace tactical_rescue
