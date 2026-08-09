#include "tactical_rescue_stream.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
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

// Parse an application/x-www-form-urlencoded body (key=value&key=value). Values
// here are only numbers and short tokens, so no percent-decoding is needed.
std::map<std::string, std::string> parse_form(const std::string& body)
{
    std::map<std::string, std::string> fields;
    size_t pos = 0;
    while (pos < body.size()) {
        const size_t amp = body.find('&', pos);
        const std::string pair = body.substr(pos, amp == std::string::npos ? std::string::npos : amp - pos);
        const size_t eq = pair.find('=');
        if (eq != std::string::npos) {
            fields[pair.substr(0, eq)] = pair.substr(eq + 1);
        }
        if (amp == std::string::npos) {
            break;
        }
        pos = amp + 1;
    }
    return fields;
}

// Percent-decode a form value. The password field goes through
// encodeURIComponent() in the browser, so it can contain %XX escapes and '+'.
std::string url_decode(const std::string& text)
{
    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '+') {
            out.push_back(' ');
        } else if (text[i] == '%' && i + 2 < text.size() && std::isxdigit(static_cast<unsigned char>(text[i + 1])) &&
                   std::isxdigit(static_cast<unsigned char>(text[i + 2]))) {
            out.push_back(static_cast<char>(std::strtol(text.substr(i + 1, 2).c_str(), nullptr, 16)));
            i += 2;
        } else {
            out.push_back(text[i]);
        }
    }
    return out;
}

float form_float(const std::map<std::string, std::string>& fields, const std::string& key)
{
    const auto it = fields.find(key);
    if (it == fields.end()) {
        return 0.0f;
    }
    const float value = std::strtof(it->second.c_str(), nullptr);
    return std::max(0.0f, std::min(1.0f, value)); // clamp to normalized image space
}

} // namespace

MjpegStreamServer::MjpegStreamServer() = default;

MjpegStreamServer::~MjpegStreamServer()
{
    stop();
}

bool MjpegStreamServer::start(const std::string& bind_address, int port, int jpeg_quality, bool auth_enabled,
                              const std::string& password)
{
    if (running_) {
        return true;
    }

    bind_address_ = bind_address.empty() ? "0.0.0.0" : bind_address;
    port_ = std::max(1, std::min(65535, port));
    jpeg_quality_ = std::max(20, std::min(95, jpeg_quality));
    // An empty password would gate nothing, so treat it as "auth off" rather
    // than silently accepting any login.
    auth_enabled_ = auth_enabled && !password.empty();
    password_ = password;

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
    std::cout << "[stream] Commander view auth: " << (auth_enabled_ ? "ENABLED (password required)" : "DISABLED")
              << std::endl;
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

    std::vector<ClientThread> clients;
    {
        std::lock_guard<std::mutex> lock(client_mutex_);
        clients.swap(client_threads_);
    }
    for (auto& client : clients) {
        if (client.thread.joinable()) {
            client.thread.join();
        }
    }
}

void MjpegStreamServer::publish(const cv::Mat& frame)
{
    // Encoding a 1920x1080 JPEG costs real time on the render thread, so skip it
    // entirely when nobody is watching the stream.
    if (!running_ || frame.empty() || stream_clients_.load() == 0) {
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

        auto finished = std::make_shared<std::atomic<bool>>(false);
        std::thread worker([this, client_fd, finished] {
            client_loop(client_fd);
            finished->store(true);
        });

        std::lock_guard<std::mutex> lock(client_mutex_);
        // Reap connections that already completed (every annotate/undo/clear POST
        // is one) so the vector tracks only live clients.
        for (auto it = client_threads_.begin(); it != client_threads_.end();) {
            if (it->finished->load()) {
                if (it->thread.joinable()) {
                    it->thread.join();
                }
                it = client_threads_.erase(it);
            } else {
                ++it;
            }
        }
        client_threads_.push_back(ClientThread{std::move(worker), std::move(finished)});
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
    size_t header_end = std::string::npos;
    // Bounded so a client that dribbles bytes slower than the recv timeout can
    // neither pin this thread forever nor delay stop().
    const auto read_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while ((header_end = request.find("\r\n\r\n")) == std::string::npos && request.size() < 8192 && running_ &&
           std::chrono::steady_clock::now() < read_deadline) {
        const ssize_t received = ::recv(client_fd, buffer, sizeof(buffer), 0);
        if (received <= 0) {
            ::close(client_fd);
            return;
        }
        request.append(buffer, static_cast<size_t>(received));
    }
    if (header_end == std::string::npos) {
        ::close(client_fd);
        return;
    }

    std::istringstream request_stream(request);
    std::string method;
    std::string path;
    request_stream >> method >> path;
    if (path.empty()) {
        path = "/";
    }

    // Determine the request body length (POSTs from the commander UI carry a
    // small form-encoded body) and read the remainder if we don't have it yet.
    size_t content_length = 0;
    std::string cookie_header;
    {
        const std::string headers = request.substr(0, header_end);
        std::istringstream header_stream(headers);
        std::string line;
        while (std::getline(header_stream, line)) {
            const size_t colon = line.find(':');
            if (colon == std::string::npos) {
                continue;
            }
            std::string name = line.substr(0, colon);
            std::transform(name.begin(), name.end(), name.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (name == "content-length") {
                content_length = static_cast<size_t>(std::strtoul(line.c_str() + colon + 1, nullptr, 10));
            } else if (name == "cookie") {
                cookie_header = line.substr(colon + 1);
                // strip leading space and the trailing '\r' std::getline leaves behind
                while (!cookie_header.empty() && (cookie_header.front() == ' ' || cookie_header.front() == '\t')) {
                    cookie_header.erase(cookie_header.begin());
                }
                while (!cookie_header.empty() && (cookie_header.back() == '\r' || cookie_header.back() == '\n')) {
                    cookie_header.pop_back();
                }
            }
        }
    }

    std::string body = request.substr(header_end + 4);
    while (body.size() < content_length && body.size() < 65536) {
        const ssize_t received = ::recv(client_fd, buffer, sizeof(buffer), 0);
        if (received <= 0) {
            break;
        }
        body.append(buffer, static_cast<size_t>(received));
    }

    // Single auth gate covering BOTH the MJPEG stream and every annotation
    // endpoint. Strip any query string so "/stream.mjpg?t=123" cache-busters
    // (and similar) route and authorize identically.
    const size_t query = path.find('?');
    if (query != std::string::npos) {
        path.erase(query);
    }
    const bool authorized = !auth_enabled_ || session_valid(cookie_header);

    // Everything except the long-lived MJPEG stream is a short request/response.
    if (path != "/stream.mjpg") {
        handle_request(client_fd, method, path, body, authorized);
        ::close(client_fd);
        return;
    }

    if (!authorized) {
        (void)send_simple(client_fd, "401 Unauthorized", "application/json", "{\"error\":\"auth required\"}");
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

    // From here on this connection is consuming frames; publish() only pays for
    // the JPEG encode while at least one viewer is counted.
    ++stream_clients_;
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

    --stream_clients_;
    ::close(client_fd);
}

namespace {

// Interactive commander console: the live HUD with a transparent capture layer.
// The commander picks a tool and clicks/drags; each placement is POSTed back to
// the Pi, which burns it into the HUD so the firefighter (and every other
// viewer) sees the same markup within a frame.
const char* const kConsoleHtml = R"HTML(<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ember Commander</title>
<style>
  /* Matches the HUD panel chrome exactly: translucent black fill, 1px accent
     border, accent micro-labels, light values. Same grammar as the sensor
     readouts burned into the video, so browser chrome and HUD read as one. */
  :root{--panel:rgba(0,0,0,.58);--line:#4aff9e;--line-soft:rgba(74,255,158,.32);
        --fg:#e6ffeb;--muted:#91d6ad;--accent:#4aff9e;
        --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}
  *{box-sizing:border-box;}
  html,body{margin:0;height:100%;background:#000;overflow:hidden;
            font:14px/1.4 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
            -webkit-font-smoothing:antialiased;-webkit-user-select:none;user-select:none;}
  #feed{position:fixed;inset:0;width:100vw;height:100vh;object-fit:contain;display:block;}
  #overlay{position:fixed;inset:0;width:100vw;height:100vh;touch-action:none;cursor:crosshair;}
  #bar{position:fixed;top:22px;left:50%;transform:translateX(-50%);display:flex;gap:4px;align-items:center;
       padding:6px;background:var(--panel);border:1px solid var(--line);border-radius:0;z-index:10;}
  .tool{color:var(--fg);background:transparent;border:1px solid transparent;border-radius:0;padding:7px 13px;
        font-size:12px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;
        cursor:pointer;white-space:nowrap;transition:background .14s ease,color .14s ease;}
  .tool:hover{background:rgba(74,255,158,.12);}
  .tool.on{background:var(--accent);color:#04140b;}
  .tool.on:hover{background:var(--accent);}
  .tool.act{color:var(--muted);}
  .tool.act:hover{color:var(--fg);}
  #sep{width:1px;height:20px;background:var(--line-soft);margin:0 4px;}
  #count{font:11px var(--mono);color:var(--muted);padding:0 9px;min-width:70px;
         text-align:center;letter-spacing:.06em;text-transform:uppercase;}
  /* pointer-events:none so these overlays never create dead zones where a
     click on the video silently does nothing. #bar must keep events (buttons). */
  #hint{position:fixed;top:80px;left:50%;transform:translateX(-50%);color:var(--muted);font-size:12px;
        letter-spacing:.04em;z-index:10;background:var(--panel);border:1px solid var(--line-soft);
        padding:5px 11px;border-radius:0;white-space:nowrap;pointer-events:none;}
  #toast{position:fixed;bottom:28px;left:50%;transform:translateX(-50%);z-index:11;
         background:var(--panel);border:1px solid var(--line-soft);color:var(--fg);border-radius:0;
         padding:8px 15px;font-size:12.5px;font-weight:500;opacity:0;transition:opacity .2s ease;
         pointer-events:none;max-width:80vw;text-align:center;}
  /* All four HUD corners are now occupied by burned-in panels, so this
     commander-only badge tucks in just below the top-left IGNISXR panel. */
  #link{position:fixed;top:92px;left:22px;z-index:10;font:10.5px var(--mono);letter-spacing:.08em;
        text-transform:uppercase;padding:4px 9px;border-radius:0;background:var(--panel);
        border:1px solid var(--line-soft);pointer-events:none;}
</style></head>
<body>
  <img id="feed" src="/stream.mjpg" alt="Ember HUD">
  <canvas id="overlay"></canvas>
  <div id="bar">
    <div class="tool on" data-t="crumb">BREADCRUMB</div>
    <div class="tool" data-t="door">DOOR</div>
    <div class="tool" data-t="arrow">ARROW</div>
    <div id="sep"></div>
    <div class="tool act" id="undo">UNDO</div>
    <div class="tool act" id="clear">CLEAR ALL</div>
    <div id="count">0 MARKS</div>
    <div id="sep"></div>
    <div class="tool act" id="logout">LOCK</div>
  </div>
  <div id="hint"></div>
  <div id="toast"></div>
  <div id="link">LINK ...</div>
<script>
(function(){
  var img=document.getElementById('feed'), cv=document.getElementById('overlay'), ctx=cv.getContext('2d');
  var toastEl=document.getElementById('toast'), hintEl=document.getElementById('hint');
  var countEl=document.getElementById('count'), linkEl=document.getElementById('link');
  var tool='crumb', dragging=false, sx=0, sy=0, cx=0, cy=0, pid=null;
  var flashes=[], animating=false, marks=0, videoOk=true;
  var NAMES={crumb:'Breadcrumb', door:'Door', arrow:'Arrow'};
  var HINTS={crumb:'Click the video to drop a numbered waypoint',
             door:'Drag a box around the opening',
             arrow:'Drag from tail to head to point the way'};
  var COLOR={crumb:'#ffd23c', door:'#28d2ff', arrow:'#ff3cd8'};

  function resize(){cv.width=window.innerWidth;cv.height=window.innerHeight;}
  window.addEventListener('resize',resize); resize();

  function toast(msg,bad){
    toastEl.textContent=msg;
    toastEl.style.borderColor = bad ? 'rgba(255,122,122,.6)' : 'rgba(74,255,158,.32)';
    toastEl.style.color       = bad ? '#ff9d9d' : '#e6ffeb';
    toastEl.style.opacity=1;
    clearTimeout(toastEl._h);
    toastEl._h=setTimeout(function(){toastEl.style.opacity=0;}, bad?4000:1500);
  }
  function setLink(ok,txt){
    linkEl.textContent=txt;
    linkEl.style.color = ok?'#91d6ad':'#ff9d9d';
    linkEl.style.borderColor = ok?'rgba(74,255,158,.32)':'rgba(255,122,122,.55)';
  }

  // Map a viewport point to normalized [0,1] coords inside the letterboxed video.
  function box(){
    var r=img.getBoundingClientRect();
    var iw=img.naturalWidth||1920, ih=img.naturalHeight||1080;
    var s=Math.min(r.width/iw, r.height/ih), w=iw*s, h=ih*s;
    return {x:r.left+(r.width-w)/2, y:r.top+(r.height-h)/2, w:w, h:h};
  }
  // Returns null for points OUTSIDE the letterboxed video. Clamping instead
  // would snap black-bar clicks onto the frame edge, dropping a marker where
  // the commander never clicked (very easy to hit on a phone in portrait).
  function norm(px,py){
    var b=box();
    if(b.w<=0||b.h<=0) return null;
    var u=(px-b.x)/b.w, v=(py-b.y)/b.h;
    if(u<0||u>1||v<0||v>1) return null;
    return {x:u,y:v};
  }
  // Keep the drag preview inside the video so preview and committed geometry agree.
  function clampPt(x,y){
    var b=box();
    return {x:Math.min(b.x+b.w,Math.max(b.x,x)), y:Math.min(b.y+b.h,Math.max(b.y,y))};
  }

  // A 401 means the session expired (or the Pi restarted) — bounce to the login gate.
  function expired(){ setLink(false,'SESSION EXPIRED'); location.replace('/'); }

  function post(path,body){
    return fetch(path,{method:'POST',cache:'no-store',
                       headers:{'Content-Type':'application/x-www-form-urlencoded'},
                       body:body||''})
      .then(function(r){ if(r.status===401){ expired(); throw new Error('session expired'); }
                         if(!r.ok) throw new Error('HTTP '+r.status);
                         return r.json().catch(function(){return {};}); })
      .then(function(d){ if(videoOk) setLink(true,'LINK OK'); sync(); return d; })
      .catch(function(e){ if(String(e.message).indexOf('session')<0){
                            setLink(false,'LINK LOST');
                            toast('NOT SENT — '+e.message+'. Check connection to the Pi.',true); }
                          throw e; });
  }

  function sync(){
    fetch('/annotations',{cache:'no-store'})
      .then(function(r){ if(r.status===401){ expired(); throw new Error('session expired'); } return r.json(); })
      .then(function(a){ marks=a.length; countEl.textContent=marks+(marks===1?' MARK':' MARKS');
                         if(videoOk) setLink(true,'LINK OK'); })
      .catch(function(){ setLink(false,'LINK LOST'); });
  }

  function pick(t){
    tool=t; hintEl.textContent=HINTS[t];
    var els=document.querySelectorAll('.tool[data-t]');
    for(var i=0;i<els.length;i++) els[i].classList.toggle('on', els[i].dataset.t===t);
    toast(NAMES[t]+' selected');
  }

  var tools=document.querySelectorAll('.tool[data-t]');
  for(var i=0;i<tools.length;i++)
    (function(el){ el.addEventListener('click',function(){pick(el.dataset.t);}); })(tools[i]);

  document.getElementById('undo').addEventListener('click',function(){
    if(marks===0){toast('Nothing to undo');return;}
    post('/undo').then(function(){toast('Removed last mark');});
  });
  document.getElementById('clear').addEventListener('click',function(){
    if(marks===0){toast('Nothing to clear');return;}
    post('/clear').then(function(){toast('All marks cleared');});
  });
  document.getElementById('logout').addEventListener('click',function(){
    fetch('/logout',{method:'POST',cache:'no-store'})
      .then(function(){location.replace('/');})
      .catch(function(){location.replace('/');});
  });

  // --- Placement ---------------------------------------------------------
  cv.addEventListener('contextmenu',function(e){e.preventDefault();});
  cv.addEventListener('pointerdown',function(e){
    // Only the primary button places markup — a stray right/middle click must
    // never burn a permanent mark into the firefighter's HUD.
    if(e.pointerType==='mouse'&&e.button!==0) return;
    var n=norm(e.clientX,e.clientY);
    if(!n){toast('Click inside the video area',true);return;}
    if(tool==='crumb'){
      flash(e.clientX,e.clientY,COLOR.crumb);
      post('/annotate','type=crumb&x0='+n.x.toFixed(5)+'&y0='+n.y.toFixed(5))
        .then(function(){toast('Breadcrumb placed');});
      return;
    }
    dragging=true; sx=cx=e.clientX; sy=cy=e.clientY;
    pid=e.pointerId; if(cv.setPointerCapture){try{cv.setPointerCapture(pid);}catch(_){}}
    kick();
  });
  cv.addEventListener('pointermove',function(e){ if(dragging){cx=e.clientX;cy=e.clientY;} });
  function finish(e){
    if(!dragging) return;
    dragging=false;
    if(pid!==null&&cv.releasePointerCapture){try{cv.releasePointerCapture(pid);}catch(_){} pid=null;}
    var far=Math.abs(cx-sx)>4||Math.abs(cy-sy)>4;
    if(!far){ toast(NAMES[tool]+' needs a drag, not a click',true); return; }
    var a=norm(sx,sy), b=norm(cx,cy);
    if(!a||!b){ toast('Drag must start and end inside the video',true); return; }
    flash(cx,cy,COLOR[tool]);
    post('/annotate','type='+tool+'&x0='+a.x.toFixed(5)+'&y0='+a.y.toFixed(5)+
                     '&x1='+b.x.toFixed(5)+'&y1='+b.y.toFixed(5))
      .then(function(){toast(tool==='door'?'Door highlighted':'Arrow placed');});
  }
  cv.addEventListener('pointerup',finish);
  cv.addEventListener('pointercancel',finish);

  // --- Local echo: instant feedback while the burned-in frame catches up ---
  function flash(x,y,c){ flashes.push({x:x,y:y,c:c,t:Date.now()}); kick(); }
  function kick(){ if(!animating){animating=true; requestAnimationFrame(loop);} }
  function loop(){
    var now=Date.now();
    ctx.clearRect(0,0,cv.width,cv.height);
    if(dragging){
      var s=clampPt(sx,sy), c=clampPt(cx,cy);
      ctx.lineWidth=3; ctx.strokeStyle=COLOR[tool]; ctx.setLineDash([7,5]);
      if(tool==='door'){
        ctx.strokeRect(Math.min(s.x,c.x),Math.min(s.y,c.y),Math.abs(c.x-s.x),Math.abs(c.y-s.y));
      } else {
        ctx.beginPath(); ctx.moveTo(s.x,s.y); ctx.lineTo(c.x,c.y); ctx.stroke();
      }
      ctx.setLineDash([]);
    }
    var alive=false;
    for(var i=0;i<flashes.length;i++){
      var f=flashes[i], dt=(now-f.t)/650;
      if(dt>=1) continue;
      alive=true;
      ctx.globalAlpha=1-dt; ctx.strokeStyle=f.c; ctx.lineWidth=3;
      ctx.beginPath(); ctx.arc(f.x,f.y,9+dt*34,0,6.2832); ctx.stroke();
      ctx.globalAlpha=1;
    }
    while(flashes.length && (now-flashes[0].t)>650) flashes.shift();
    if(dragging||alive){ requestAnimationFrame(loop); }
    else { animating=false; ctx.clearRect(0,0,cv.width,cv.height); }
  }

  // The MJPEG connection can end on a network hiccup. Without this the picture
  // silently freezes on its last frame — which looks like a live feed, and made
  // it appear that placing markup "did nothing". Reconnect automatically, and
  // never let sync() paint LINK OK while the video is actually dead.
  function reconnect(){ img.src='/stream.mjpg?r='+Date.now(); }
  img.addEventListener('error',function(){
    videoOk=false; setLink(false,'VIDEO LOST — RECONNECTING'); setTimeout(reconnect,1000);
  });
  img.addEventListener('load',function(){ videoOk=true; });

  pick('crumb'); sync(); setInterval(sync,3000);
})();
</script></body></html>)HTML";

// Branded gate for the commander view, styled to match the HUD rather than
// falling back to the browser's unstyled Basic-auth popup.
const char* const kLoginHtml = R"HTML(<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ember — Commander Console</title>
<style>
  :root{
    --bg:#0b0d0c; --line:rgba(255,255,255,.09); --line2:rgba(255,255,255,.055);
    --fg:#e7ebe8; --muted:#7b887f; --dim:#5b675f; --accent:#7fe3a1; --danger:#ff7a7a;
  }
  *{box-sizing:border-box;}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);
    font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;}
  body{display:flex;align-items:center;justify-content:center;padding:28px;}
  form{width:100%;max-width:348px;}
  .rule{height:1px;background:var(--line2);}
  .eyebrow{font-size:10.5px;letter-spacing:.2em;text-transform:uppercase;color:var(--dim);}
  .word{font-size:33px;font-weight:600;letter-spacing:-.021em;line-height:1.1;margin:3px 0 16px;}
  .lede{font-size:13.5px;color:var(--muted);margin:15px 0 27px;}
  label{display:block;font-size:10.5px;letter-spacing:.17em;text-transform:uppercase;
        color:var(--muted);margin-bottom:7px;}
  input{width:100%;background:transparent;border:0;border-bottom:1px solid var(--line);
        padding:9px 1px;color:var(--fg);font-size:16px;outline:none;
        transition:border-color .16s ease;}
  input::placeholder{color:#4a554e;}
  input:focus{border-bottom-color:var(--accent);}
  button{width:100%;margin-top:24px;padding:11px;border:0;border-radius:3px;
         background:var(--accent);color:#06120b;font-size:12.5px;font-weight:700;
         letter-spacing:.11em;text-transform:uppercase;cursor:pointer;
         transition:opacity .16s ease;}
  button:hover{opacity:.87;}
  button:disabled{opacity:.45;cursor:default;}
  .note{margin-top:13px;font-size:12.5px;min-height:19px;opacity:0;transition:opacity .16s ease;}
  .note.on{opacity:1;}
  .note.bad{color:var(--danger);}
  .note.warn{color:#e3c37f;}
  .meta{margin-top:26px;font:11px/1.85 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
        color:var(--dim);}
  .meta i{font-style:normal;display:inline-block;width:52px;color:#47524b;}
</style></head>
<body>
  <form id="f" autocomplete="off" novalidate>
    <div class="eyebrow">IgnisXR</div>
    <div class="word">Ember</div>
    <div class="rule"></div>
    <p class="lede">Commander console. Sign in to view the live helmet feed and place navigation markup.</p>

    <label for="p">Passkey</label>
    <input id="p" type="password" placeholder="Enter passkey" autocomplete="current-password" autofocus>
    <button id="go" type="submit">Unlock</button>
    <div class="note" id="e"></div>

    <div class="rule" style="margin-top:14px"></div>
    <div class="meta">
      <div><i>NODE</i><span id="host"></span></div>
      <div><i>LINK</i>local network &middot; not encrypted</div>
    </div>
  </form>
<script>
(function(){
  var f=document.getElementById('f'), p=document.getElementById('p'),
      e=document.getElementById('e'), go=document.getElementById('go');
  document.getElementById('host').textContent=location.host;

  function note(msg,kind){
    e.textContent=msg||'';
    e.className='note'+(msg?' on':'')+(kind?' '+kind:'');
  }
  p.addEventListener('input',function(){ if(e.classList.contains('bad')) note(''); });
  // Real-world login detail: silent caps lock is the usual reason a correct
  // passkey gets rejected.
  p.addEventListener('keyup',function(ev){
    if(ev.getModifierState&&ev.getModifierState('CapsLock')){ note('Caps Lock is on','warn'); }
    else if(e.classList.contains('warn')){ note(''); }
  });

  f.addEventListener('submit',function(ev){
    ev.preventDefault();
    if(!p.value){ note('Enter the passkey','bad'); p.focus(); return; }
    note(''); go.disabled=true; go.textContent='Unlocking';
    fetch('/login',{method:'POST',cache:'no-store',
                    headers:{'Content-Type':'application/x-www-form-urlencoded'},
                    body:'password='+encodeURIComponent(p.value)})
      .then(function(r){
        if(r.ok){ location.replace('/'); return; }
        go.disabled=false; go.textContent='Unlock';
        note('That passkey was not accepted','bad'); p.value=''; p.focus();
      })
      .catch(function(){
        go.disabled=false; go.textContent='Unlock';
        note('Cannot reach the node — check the link to the Pi','bad');
      });
  });
})();
</script></body></html>)HTML";

} // namespace

void MjpegStreamServer::handle_request(int client_fd, const std::string& method, const std::string& path,
                                       const std::string& body, bool authorized)
{
    // --- auth endpoints (always reachable) ---
    if (method == "POST" && path == "/login") {
        const auto fields = parse_form(body);
        const auto it = fields.find("password");
        const std::string supplied = it == fields.end() ? std::string() : url_decode(it->second);
        if (!auth_enabled_ || password_matches(supplied)) {
            const std::string token = create_session();
            (void)send_simple(client_fd, "200 OK", "application/json", "{\"ok\":true}",
                              "Set-Cookie: ember_session=" + token + "; Path=/; HttpOnly; SameSite=Strict\r\n");
        } else {
            (void)send_simple(client_fd, "401 Unauthorized", "application/json", "{\"error\":\"bad password\"}");
        }
        return;
    }
    if (method == "POST" && path == "/logout") {
        (void)send_simple(client_fd, "200 OK", "application/json", "{\"ok\":true}",
                          "Set-Cookie: ember_session=; Path=/; Max-Age=0\r\n");
        return;
    }

    // --- everything below requires a session ---
    if (!authorized) {
        if (method == "GET" && (path == "/" || path == "/index.html")) {
            (void)send_simple(client_fd, "200 OK", "text/html; charset=utf-8", kLoginHtml);
        } else {
            (void)send_simple(client_fd, "401 Unauthorized", "application/json", "{\"error\":\"auth required\"}");
        }
        return;
    }

    if (method == "POST" && path == "/annotate") {
        const auto fields = parse_form(body);
        const auto type_it = fields.find("type");
        StreamAnnotation annotation;
        if (type_it != fields.end() && type_it->second == "door") {
            annotation.type = StreamAnnotation::Type::Door;
        } else if (type_it != fields.end() && type_it->second == "arrow") {
            annotation.type = StreamAnnotation::Type::Arrow;
        } else {
            annotation.type = StreamAnnotation::Type::Breadcrumb;
        }
        annotation.x0 = form_float(fields, "x0");
        annotation.y0 = form_float(fields, "y0");
        annotation.x1 = form_float(fields, "x1");
        annotation.y1 = form_float(fields, "y1");
        const uint64_t id = add_annotation(annotation);
        (void)send_simple(client_fd, "200 OK", "application/json", "{\"id\":" + std::to_string(id) + "}");
        return;
    }
    if (method == "POST" && path == "/undo") {
        undo_annotation();
        (void)send_simple(client_fd, "200 OK", "application/json", "{\"ok\":true}");
        return;
    }
    if (method == "POST" && path == "/clear") {
        clear_annotations();
        (void)send_simple(client_fd, "200 OK", "application/json", "{\"ok\":true}");
        return;
    }
    if (path == "/annotations") {
        std::ostringstream json;
        json << '[';
        const auto snapshot = annotations();
        for (size_t i = 0; i < snapshot.size(); ++i) {
            const auto& a = snapshot[i];
            const char* type = a.type == StreamAnnotation::Type::Door       ? "door"
                               : a.type == StreamAnnotation::Type::Arrow     ? "arrow"
                                                                             : "crumb";
            json << (i ? "," : "") << "{\"id\":" << a.id << ",\"type\":\"" << type << "\",\"x0\":" << a.x0
                 << ",\"y0\":" << a.y0 << ",\"x1\":" << a.x1 << ",\"y1\":" << a.y1 << '}';
        }
        json << ']';
        (void)send_simple(client_fd, "200 OK", "application/json", json.str());
        return;
    }

    // Default: the commander console page.
    (void)send_simple(client_fd, "200 OK", "text/html; charset=utf-8", kConsoleHtml);
}

uint64_t MjpegStreamServer::add_annotation(const StreamAnnotation& annotation)
{
    std::lock_guard<std::mutex> lock(annotations_mutex_);
    StreamAnnotation stored = annotation;
    stored.id = next_annotation_id_++;
    annotations_.push_back(stored);
    // Bound memory if a commander goes wild; keep the most recent markup.
    constexpr size_t kMaxAnnotations = 256;
    if (annotations_.size() > kMaxAnnotations) {
        annotations_.erase(annotations_.begin(), annotations_.begin() + (annotations_.size() - kMaxAnnotations));
    }
    return stored.id;
}

void MjpegStreamServer::undo_annotation()
{
    std::lock_guard<std::mutex> lock(annotations_mutex_);
    if (!annotations_.empty()) {
        annotations_.pop_back();
    }
}

void MjpegStreamServer::clear_annotations()
{
    std::lock_guard<std::mutex> lock(annotations_mutex_);
    annotations_.clear();
}

std::vector<StreamAnnotation> MjpegStreamServer::annotations() const
{
    std::lock_guard<std::mutex> lock(annotations_mutex_);
    return annotations_;
}

bool MjpegStreamServer::password_matches(const std::string& candidate) const
{
    // Length-checked, constant-time-ish compare so a wrong guess does not leak
    // the password length through an early exit.
    if (candidate.size() != password_.size()) {
        return false;
    }
    unsigned char diff = 0;
    for (size_t i = 0; i < password_.size(); ++i) {
        diff |= static_cast<unsigned char>(candidate[i] ^ password_[i]);
    }
    return diff == 0;
}

std::string MjpegStreamServer::create_session()
{
    // 128-bit random hex token. Sessions are memory-only and die with the process.
    std::random_device rd;
    std::ostringstream token;
    for (int i = 0; i < 4; ++i) {
        token << std::hex << std::setw(8) << std::setfill('0') << rd();
    }
    const std::string value = token.str();

    std::lock_guard<std::mutex> lock(sessions_mutex_);
    constexpr size_t kMaxSessions = 64;
    if (sessions_.size() >= kMaxSessions) {
        sessions_.clear(); // bound memory; forces a re-login, which is acceptable
    }
    sessions_.insert(value);
    return value;
}

bool MjpegStreamServer::session_valid(const std::string& cookie_header) const
{
    if (cookie_header.empty()) {
        return false;
    }
    // Cookie header is "a=1; ember_session=xyz; b=2"
    static const std::string kName = "ember_session=";
    size_t pos = 0;
    while ((pos = cookie_header.find(kName, pos)) != std::string::npos) {
        // Only match at a cookie boundary, not inside another cookie's value.
        if (pos == 0 || cookie_header[pos - 1] == ' ' || cookie_header[pos - 1] == ';') {
            const size_t start = pos + kName.size();
            const size_t end = cookie_header.find(';', start);
            const std::string token = cookie_header.substr(start, end == std::string::npos ? end : end - start);
            if (!token.empty()) {
                std::lock_guard<std::mutex> lock(sessions_mutex_);
                if (sessions_.count(token) > 0) {
                    return true;
                }
            }
        }
        pos += kName.size();
    }
    return false;
}

bool MjpegStreamServer::send_simple(int fd, const std::string& status, const std::string& content_type,
                                    const std::string& body, const std::string& extra_headers) const
{
    // "Connection: close" is REQUIRED here. We close the socket immediately after
    // responding, but HTTP/1.1 defaults to keep-alive — without this header the
    // browser pools the dead socket and the NEXT annotation POST dies with a
    // broken pipe. POST is not idempotent, so browsers never retry it: the
    // commander's click silently does nothing. Measured: 50% of POSTs lost.
    std::ostringstream response;
    response << "HTTP/1.1 " << status << "\r\n"
             << "Content-Type: " << content_type << "\r\n"
             << "Connection: close\r\n"
             << "Cache-Control: no-store\r\n"
             << extra_headers
             << "Content-Length: " << body.size() << "\r\n\r\n"
             << body;
    return write_string(fd, response.str());
}

bool MjpegStreamServer::write_all(int fd, const void* data, size_t size) const
{
    const char* bytes = static_cast<const char*>(data);
    size_t written = 0;
    // The socket carries a 2s SO_SNDTIMEO. A congested Wi-Fi link stalls that
    // long routinely, and treating the resulting EAGAIN as fatal used to kill
    // the feed outright — the commander's video froze on its last frame while
    // the page still claimed the link was healthy. Ride out transient
    // backpressure up to a real deadline, but keep observing running_ so a
    // degraded client can never block stop() from returning.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (written < size) {
        const ssize_t result = ::send(fd, bytes + written, size - written, MSG_NOSIGNAL);
        if (result > 0) {
            written += static_cast<size_t>(result);
            continue;
        }
        if (result < 0 && errno == EINTR) {
            continue;
        }
        if (result < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            if (!running_ || std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
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
