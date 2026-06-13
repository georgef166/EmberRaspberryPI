#include "tactical_rescue_stream.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <opencv2/imgcodecs.hpp>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

namespace tactical_rescue {

namespace {

constexpr const char* kBoundary = "emberframe";

void close_fd(int& fd)
{
    if (fd >= 0) {
        ::close(fd);
        fd = -1;
    }
}

} // namespace

MjpegStreamServer::MjpegStreamServer() = default;

MjpegStreamServer::~MjpegStreamServer()
{
    stop();
}

bool MjpegStreamServer::start(const std::string& bind_address, int port, int jpeg_quality)
{
    if (running_) {
        return true;
    }

    bind_address_ = bind_address.empty() ? "0.0.0.0" : bind_address;
    port_ = std::max(1, std::min(65535, port));
    jpeg_quality_ = std::max(20, std::min(95, jpeg_quality));

    server_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        std::cerr << "[stream] socket() failed: " << std::strerror(errno) << std::endl;
        return false;
    }

    // Mark the listening socket close-on-exec so child processes we fork+exec
    // later (e.g. the AM2302 python helper) do NOT inherit it. Otherwise a
    // leftover helper keeps port 8080 bound after the app exits, and the next
    // launch's bind() fails with EADDRINUSE — silently disabling the feed.
    const int fd_flags = ::fcntl(server_fd_, F_GETFD);
    if (fd_flags >= 0) {
        (void)::fcntl(server_fd_, F_SETFD, fd_flags | FD_CLOEXEC);
    }

    int reuse = 1;
    (void)::setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(port_));
    if (bind_address_ == "0.0.0.0") {
        address.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (::inet_pton(AF_INET, bind_address_.c_str(), &address.sin_addr) != 1) {
        std::cerr << "[stream] invalid bind address: " << bind_address_ << std::endl;
        close_fd(server_fd_);
        return false;
    }

    if (::bind(server_fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        std::cerr << "[stream] bind(" << bind_address_ << ":" << port_ << ") failed: " << std::strerror(errno)
                  << std::endl;
        close_fd(server_fd_);
        return false;
    }

    if (::listen(server_fd_, 8) != 0) {
        std::cerr << "[stream] listen() failed: " << std::strerror(errno) << std::endl;
        close_fd(server_fd_);
        return false;
    }

    running_ = true;
    accept_thread_ = std::thread([this] { accept_loop(); });
    std::cout << "[stream] Remote HUD feed: http://"
              << (bind_address_ == "0.0.0.0" ? "<pi-ip>" : bind_address_) << ":" << port_ << "/" << std::endl;
    return true;
}

void MjpegStreamServer::stop()
{
    if (!running_ && server_fd_ < 0) {
        return;
    }

    running_ = false;
    frame_cv_.notify_all();
    if (server_fd_ >= 0) {
        (void)::shutdown(server_fd_, SHUT_RDWR);
        close_fd(server_fd_);
    }

    if (accept_thread_.joinable()) {
        accept_thread_.join();
    }

    std::vector<std::thread> clients;
    {
        std::lock_guard<std::mutex> lock(client_mutex_);
        clients.swap(client_threads_);
    }
    for (auto& client : clients) {
        if (client.joinable()) {
            client.join();
        }
    }
}

void MjpegStreamServer::publish(const cv::Mat& frame)
{
    if (!running_ || frame.empty()) {
        return;
    }

    std::vector<unsigned char> encoded;
    const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_};
    if (!cv::imencode(".jpg", frame, encoded, params)) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_jpeg_ = std::move(encoded);
        ++sequence_;
    }
    frame_cv_.notify_all();
}

void MjpegStreamServer::accept_loop()
{
    while (running_) {
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(server_fd_, &read_fds);
        timeval timeout{};
        timeout.tv_sec = 0;
        timeout.tv_usec = 200000;

        const int ready = ::select(server_fd_ + 1, &read_fds, nullptr, nullptr, &timeout);
        if (!running_) {
            break;
        }
        if (ready <= 0 || !FD_ISSET(server_fd_, &read_fds)) {
            continue;
        }

        sockaddr_in client_address{};
        socklen_t client_len = sizeof(client_address);
        const int client_fd = ::accept(server_fd_, reinterpret_cast<sockaddr*>(&client_address), &client_len);
        if (client_fd < 0) {
            if (running_) {
                std::cerr << "[stream] accept() failed: " << std::strerror(errno) << std::endl;
            }
            continue;
        }

        std::lock_guard<std::mutex> lock(client_mutex_);
        client_threads_.emplace_back([this, client_fd] { client_loop(client_fd); });
    }
}

void MjpegStreamServer::client_loop(int client_fd)
{
    timeval timeout{};
    timeout.tv_sec = 2;
    timeout.tv_usec = 0;
    (void)::setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    (void)::setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    std::string request;
    char buffer[1024];
    while (request.find("\r\n\r\n") == std::string::npos && request.size() < 8192) {
        const ssize_t received = ::recv(client_fd, buffer, sizeof(buffer), 0);
        if (received <= 0) {
            ::close(client_fd);
            return;
        }
        request.append(buffer, static_cast<size_t>(received));
    }

    std::istringstream request_stream(request);
    std::string method;
    std::string path;
    request_stream >> method >> path;
    if (path.empty()) {
        path = "/";
    }

    if (path != "/stream.mjpg") {
        const std::string body =
            "<!doctype html><html><head><meta charset=\"utf-8\"><title>Ember HUD</title>"
            "<style>html,body{margin:0;background:#000;height:100%;overflow:hidden;}"
            "img{width:100vw;height:100vh;object-fit:contain;display:block;}</style></head>"
            "<body><img src=\"/stream.mjpg\" alt=\"Ember HUD\"></body></html>";
        std::ostringstream response;
        response << "HTTP/1.1 200 OK\r\n"
                 << "Content-Type: text/html; charset=utf-8\r\n"
                 << "Cache-Control: no-store\r\n"
                 << "Content-Length: " << body.size() << "\r\n\r\n"
                 << body;
        (void)write_string(client_fd, response.str());
        ::close(client_fd);
        return;
    }

    std::ostringstream header;
    header << "HTTP/1.1 200 OK\r\n"
           << "Connection: close\r\n"
           << "Cache-Control: no-cache, no-store, must-revalidate\r\n"
           << "Pragma: no-cache\r\n"
           << "Content-Type: multipart/x-mixed-replace; boundary=" << kBoundary << "\r\n\r\n";
    if (!write_string(client_fd, header.str())) {
        ::close(client_fd);
        return;
    }

    uint64_t last_sequence = 0;
    while (running_) {
        std::vector<unsigned char> jpeg;
        {
            std::unique_lock<std::mutex> lock(frame_mutex_);
            frame_cv_.wait_for(lock, std::chrono::milliseconds(1000),
                               [&] { return !running_ || sequence_ != last_sequence; });
            if (!running_) {
                break;
            }
            if (sequence_ == last_sequence || latest_jpeg_.empty()) {
                continue;
            }
            jpeg = latest_jpeg_;
            last_sequence = sequence_;
        }

        std::ostringstream part;
        part << "--" << kBoundary << "\r\n"
             << "Content-Type: image/jpeg\r\n"
             << "Content-Length: " << jpeg.size() << "\r\n\r\n";
        if (!write_string(client_fd, part.str()) || !write_all(client_fd, jpeg.data(), jpeg.size()) ||
            !write_string(client_fd, "\r\n")) {
            break;
        }
    }

    ::close(client_fd);
}

bool MjpegStreamServer::write_all(int fd, const void* data, size_t size) const
{
    const char* bytes = static_cast<const char*>(data);
    size_t written = 0;
    while (written < size) {
        const ssize_t result = ::send(fd, bytes + written, size - written, MSG_NOSIGNAL);
        if (result > 0) {
            written += static_cast<size_t>(result);
            continue;
        }
        if (result < 0 && errno == EINTR) {
            continue;
        }
        return false;
    }
    return true;
}

bool MjpegStreamServer::write_string(int fd, const std::string& text) const
{
    return write_all(fd, text.data(), text.size());
}

} // namespace tactical_rescue
