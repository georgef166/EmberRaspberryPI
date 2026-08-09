#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

namespace tactical_rescue {

// A single piece of commander markup placed from the browser, in normalized
// [0,1] image coordinates so it maps onto the HUD regardless of resolution.
//   Door      : (x0,y0)-(x1,y1) opposite corners of a highlighted opening.
//   Breadcrumb: (x0,y0) a single dropped waypoint (trail marker).
//   Arrow     : (x0,y0) tail -> (x1,y1) head, a directional cue.
struct StreamAnnotation {
    enum class Type { Door, Breadcrumb, Arrow };
    Type type = Type::Breadcrumb;
    float x0 = 0.0f;
    float y0 = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    uint64_t id = 0;
};

class MjpegStreamServer {
public:
    MjpegStreamServer();
    ~MjpegStreamServer();

    // auth_enabled/password gate the remote commander view. The firefighter's
    // local HUD window is never affected.
    bool start(const std::string& bind_address, int port, int jpeg_quality, bool auth_enabled = false,
               const std::string& password = std::string());
    void stop();
    void publish(const cv::Mat& frame);

    // Snapshot of the current commander markup, safe to call from the render
    // thread each frame.
    std::vector<StreamAnnotation> annotations() const;

private:
    void accept_loop();
    void client_loop(int client_fd);
    void handle_request(int client_fd, const std::string& method, const std::string& path, const std::string& body,
                        bool authorized);
    uint64_t add_annotation(const StreamAnnotation& annotation);
    void undo_annotation();
    void clear_annotations();
    // --- commander view auth ---
    bool password_matches(const std::string& candidate) const;
    bool session_valid(const std::string& cookie_header) const;
    std::string create_session();
    bool write_all(int fd, const void* data, size_t size) const;
    bool write_string(int fd, const std::string& text) const;
    bool send_simple(int fd, const std::string& status, const std::string& content_type, const std::string& body,
                     const std::string& extra_headers = std::string()) const;

    std::atomic<bool> running_{false};
    int server_fd_ = -1;
    std::string bind_address_ = "0.0.0.0";
    int port_ = 8080;
    int jpeg_quality_ = 75;
    std::thread accept_thread_;
    std::mutex client_mutex_;
    // Each connection gets a thread. Short-lived request threads (annotate/undo/
    // clear/page loads) are reaped as new ones arrive so a long incident with
    // hundreds of commander clicks does not accumulate thread objects.
    struct ClientThread {
        std::thread thread;
        std::shared_ptr<std::atomic<bool>> finished;
    };
    std::vector<ClientThread> client_threads_;

    std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    std::vector<unsigned char> latest_jpeg_;
    uint64_t sequence_ = 0;
    // Number of clients currently consuming the MJPEG stream. When zero we skip
    // the (expensive) full-frame JPEG encode entirely on the render thread.
    std::atomic<int> stream_clients_{0};

    mutable std::mutex annotations_mutex_;
    std::vector<StreamAnnotation> annotations_;
    uint64_t next_annotation_id_ = 1;

    // Session tokens live in memory only and die with the process.
    bool auth_enabled_ = false;
    std::string password_;
    mutable std::mutex sessions_mutex_;
    std::set<std::string> sessions_;
};

// Renders commander markup (doors, breadcrumb trail, arrows) onto a HUD frame.
// Defined in tactical_rescue_render.cpp; coordinates are normalized [0,1].
void draw_stream_annotations(cv::Mat& frame, const std::vector<StreamAnnotation>& annotations);

} // namespace tactical_rescue
