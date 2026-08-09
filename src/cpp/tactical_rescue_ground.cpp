// -----------------------------------------------------------------------------
// Ember — ground plane detection
//
// First 3D stage in the pipeline: recover the floor as a plane in ToF camera
// coordinates so the HUD can paint a perspective-correct grid on it, and so the
// breadcrumb milestone has a surface to anchor waypoints to.
//
// Per ground-thread tick:
//   1. Unproject a strided subsample of depth_mm through the pinhole model
//      (fx, fy, cx, cy). 240x180 at stride 4 is ~2700 points instead of 43200 —
//      that subsample is what makes this affordable on a Pi 4B.
//   2. RANSAC a plane. Seed triplets come from the lower two-thirds of the frame,
//      where a floor almost always is, but are SCORED against the whole cloud.
//      Candidates are rejected on tilt and camera height BEFORE the O(N) inlier
//      scan; that early-out is most of the speed.
//   3. Least-squares refit on the inliers via the smallest-eigenvector of their
//      covariance. A raw 3-point normal is far too jittery to draw a grid with.
//   4. Temporal EMA plus confirm/hold hysteresis, in the same spirit as
//      stabilize_tflite_detections(), so the grid neither shimmers nor blinks.
//
// Convention: camera frame is X right, Y DOWN, Z forward; lengths in mm; the
// plane is n.P + d = 0 with |n| = 1 and n[1] < 0 (pointing up out of the floor,
// toward the camera), which makes d the camera height and d < 0 a ceiling.
// -----------------------------------------------------------------------------

#include "tactical_rescue.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace tactical_rescue {

namespace {

struct PlaneModel {
    cv::Vec3f n{0.0f, -1.0f, 0.0f};
    float d = 0.0f;
};

// Point the normal up out of the floor, toward the camera. Camera Y is down, so
// "up" is the negative-Y hemisphere.
void orient_up(cv::Vec3f& n)
{
    if (n[1] > 0.0f) {
        n = -n;
    }
}

// Angle between the plane normal and camera "up" = (0, -1, 0). Equals the
// camera's pitch/roll magnitude relative to the floor.
float plane_tilt_deg(const cv::Vec3f& n)
{
    const float cosine = std::min(1.0f, std::max(-1.0f, -n[1]));
    return static_cast<float>(std::acos(cosine) * 180.0 / CV_PI);
}

// ToF range noise grows with distance, so a fixed inlier band either rejects good
// far-field floor or swallows junk up close. 30 mm + 1% is ~40 mm at 1 m, ~70 mm at 4 m.
inline float inlier_band_mm(float z_mm, const Options& opt)
{
    return opt.ground_inlier_mm + 0.010f * z_mm;
}

bool plane_from_three(const cv::Point3f& a, const cv::Point3f& b, const cv::Point3f& c, PlaneModel& out)
{
    const cv::Vec3f ab(b.x - a.x, b.y - a.y, b.z - a.z);
    const cv::Vec3f ac(c.x - a.x, c.y - a.y, c.z - a.z);
    cv::Vec3f n = ab.cross(ac);
    // |ab x ac| is twice the triangle area in mm^2. A sliver here means the three
    // samples are near-collinear and the normal would be pure sensor noise.
    const float area2 = static_cast<float>(cv::norm(n));
    if (area2 < 1.0e3f) {
        return false;
    }
    n /= area2;
    orient_up(n);
    out.n = n;
    out.d = -(n[0] * a.x + n[1] * a.y + n[2] * a.z);
    return true;
}

// Least-squares plane through the inliers: the eigenvector of the covariance
// matrix with the smallest eigenvalue is the direction of least variance, i.e.
// the normal. cv::eigen returns eigenvalues descending, so that is the last row.
bool refit_least_squares(const std::vector<cv::Point3f>& points, const std::vector<int>& inliers, PlaneModel& out)
{
    if (inliers.size() < 3) {
        return false;
    }

    double mx = 0.0, my = 0.0, mz = 0.0;
    for (const int i : inliers) {
        mx += points[i].x;
        my += points[i].y;
        mz += points[i].z;
    }
    const double inv_n = 1.0 / static_cast<double>(inliers.size());
    mx *= inv_n;
    my *= inv_n;
    mz *= inv_n;

    double xx = 0.0, xy = 0.0, xz = 0.0, yy = 0.0, yz = 0.0, zz = 0.0;
    for (const int i : inliers) {
        const double dx = points[i].x - mx;
        const double dy = points[i].y - my;
        const double dz = points[i].z - mz;
        xx += dx * dx;
        xy += dx * dy;
        xz += dx * dz;
        yy += dy * dy;
        yz += dy * dz;
        zz += dz * dz;
    }

    const cv::Mat covariance = (cv::Mat_<double>(3, 3) << xx, xy, xz, xy, yy, yz, xz, yz, zz);
    cv::Mat eigenvalues;
    cv::Mat eigenvectors;
    if (!cv::eigen(covariance, eigenvalues, eigenvectors)) {
        return false;
    }

    cv::Vec3f n(static_cast<float>(eigenvectors.at<double>(2, 0)),
                static_cast<float>(eigenvectors.at<double>(2, 1)),
                static_cast<float>(eigenvectors.at<double>(2, 2)));
    const float len = static_cast<float>(cv::norm(n));
    if (len < 1.0e-6f) {
        return false;
    }
    n /= len;
    orient_up(n);
    out.n = n;
    out.d = -static_cast<float>(n[0] * mx + n[1] * my + n[2] * mz);
    return true;
}

// Unproject a strided subsample of the depth map. Also records which points sit
// below the top third of the frame — those become the RANSAC seed pool. Walls and
// ceilings deliberately stay IN the cloud: they are what makes inlier_ratio a
// meaningful "how much of this scene is floor" number.
void build_point_cloud(const SharedFrame& lidar_frame, const Options& opt, std::vector<cv::Point3f>& points,
                       std::vector<int>& seeds)
{
    points.clear();
    seeds.clear();

    const cv::Mat& depth = lidar_frame.depth_mm;
    if (depth.empty() || depth.type() != CV_32F) {
        return;
    }

    // Reuse the existing depth-range + confidence gate rather than re-deriving one.
    const cv::Mat mask = build_geometry_mask(depth, lidar_frame.confidence, opt);
    if (mask.empty()) {
        return;
    }

    const int stride = std::max(1, opt.ground_stride);
    const int seed_row = depth.rows / 3;
    points.reserve(static_cast<size_t>((depth.rows / stride + 1) * (depth.cols / stride + 1)));

    for (int y = 0; y < depth.rows; y += stride) {
        const float* depth_row = depth.ptr<float>(y);
        const uint8_t* mask_row = mask.ptr<uint8_t>(y);
        const bool seed_row_hit = y >= seed_row;
        for (int x = 0; x < depth.cols; x += stride) {
            if (mask_row[x] == 0) {
                continue;
            }
            const float z = depth_row[x];
            if (!std::isfinite(z)) {
                continue;
            }
            if (seed_row_hit) {
                seeds.push_back(static_cast<int>(points.size()));
            }
            points.push_back(unproject_pixel(static_cast<float>(x), static_cast<float>(y), z, opt));
        }
    }
}

// Draw three well-separated seeds and fit a candidate plane through them.
bool sample_candidate(const std::vector<cv::Point3f>& points, const std::vector<int>& pool, std::mt19937& rng,
                      PlaneModel& model)
{
    if (pool.size() < 3) {
        return false;
    }

    std::uniform_int_distribution<size_t> pick(0, pool.size() - 1);
    const cv::Point3f& a = points[pool[pick(rng)]];
    const cv::Point3f& b = points[pool[pick(rng)]];
    const cv::Point3f& c = points[pool[pick(rng)]];

    // Samples closer together than this produce normals dominated by noise.
    constexpr float kMinSeparationSq = 250.0f * 250.0f;
    auto far_apart = [](const cv::Point3f& p, const cv::Point3f& q) {
        const float dx = p.x - q.x;
        const float dy = p.y - q.y;
        const float dz = p.z - q.z;
        return dx * dx + dy * dy + dz * dz > kMinSeparationSq;
    };
    if (!far_apart(a, b) || !far_apart(a, c) || !far_apart(b, c)) {
        return false;
    }
    return plane_from_three(a, b, c, model);
}

// Kill a candidate before the O(N) inlier scan. d is the camera height, so this
// one test rejects ceilings (d < 0), tabletops and beds (d too small), and the
// tilt test rejects walls. Most random triplets die here for a handful of flops.
bool candidate_is_plausible(const PlaneModel& model, const Options& opt)
{
    if (model.d < opt.ground_min_height_mm || model.d > opt.ground_max_height_mm) {
        return false;
    }
    return plane_tilt_deg(model.n) <= opt.ground_max_tilt_deg;
}

int count_inliers(const std::vector<cv::Point3f>& points, const PlaneModel& model, const Options& opt,
                  std::vector<int>* out_inliers)
{
    if (out_inliers) {
        out_inliers->clear();
    }
    int count = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        const cv::Point3f& p = points[i];
        const float distance = std::fabs(model.n[0] * p.x + model.n[1] * p.y + model.n[2] * p.z + model.d);
        if (distance <= inlier_band_mm(p.z, opt)) {
            ++count;
            if (out_inliers) {
                out_inliers->push_back(static_cast<int>(i));
            }
        }
    }
    return count;
}

} // namespace

void scale_intrinsics_to_frame(Options& opt, int width, int height)
{
    // A user-supplied --intrinsics is assumed to already match the live sensor.
    if (opt.intrinsics_explicit || width <= 0 || height <= 0) {
        return;
    }
    const float sx = static_cast<float>(width) / kIntrinsicsRefWidth;
    const float sy = static_cast<float>(height) / kIntrinsicsRefHeight;
    if (std::fabs(sx - 1.0f) < 1.0e-3f && std::fabs(sy - 1.0f) < 1.0e-3f) {
        return;
    }
    opt.fx *= sx;
    opt.cx *= sx;
    opt.fy *= sy;
    opt.cy *= sy;
}

cv::Point3f unproject_pixel(float u, float v, float depth_mm, const Options& opt)
{
    const float nx = (u - opt.cx) / opt.fx;
    const float ny = (v - opt.cy) / opt.fy;
    float z = depth_mm;
    if (opt.depth_is_radial) {
        // Slant range -> perpendicular depth. Skipping this on a radial sensor
        // bows a flat floor upward by ~20% at the frame corners at 55 deg HFOV.
        z = depth_mm / std::sqrt(1.0f + nx * nx + ny * ny);
    }
    return cv::Point3f(nx * z, ny * z, z);
}

bool project_point(const cv::Point3f& camera_point, const Options& opt, cv::Point2f& out_px)
{
    constexpr float kNearClipMm = 120.0f;
    if (camera_point.z <= kNearClipMm) {
        return false;
    }
    out_px.x = opt.fx * camera_point.x / camera_point.z + opt.cx;
    out_px.y = opt.fy * camera_point.y / camera_point.z + opt.cy;
    return true;
}

// Where does the viewing ray through pixel (u, v) meet the floor? This is the hook
// the breadcrumb milestone needs: aim the reticle, get a stable 3D anchor on the
// ground, remember it, and re-project it with project_point() every frame.
bool ray_plane_intersect(const GroundPlane& plane, float u, float v, const Options& opt, cv::Point3f& out_point)
{
    if (!plane.valid) {
        return false;
    }
    const cv::Vec3f dir((u - opt.cx) / opt.fx, (v - opt.cy) / opt.fy, 1.0f);
    const float denominator = plane.normal.dot(dir);
    if (std::fabs(denominator) < 1.0e-6f) {
        return false;   // ray runs parallel to the floor
    }
    const float t = -plane.d / denominator;
    if (t <= 0.0f) {
        return false;   // floor is behind the camera along this ray (above the horizon)
    }
    out_point = cv::Point3f(dir[0] * t, dir[1] * t, dir[2] * t);
    return true;
}

GroundPlaneState GroundPlaneTracker::update(const SharedFrame& lidar_frame, const Options& opt)
{
    GroundPlaneState state;

    build_point_cloud(lidar_frame, opt, points_, seeds_);

    // Seed from the lower frame when we can. If it is mostly invalid — heavy
    // smoke, an obstruction right at the lens, or the operator looking up — fall
    // back to seeding from anywhere rather than giving up on the frame.
    const std::vector<int>* pool = &seeds_;
    if (seeds_.size() < 64) {
        all_indices_.resize(points_.size());
        for (size_t i = 0; i < all_indices_.size(); ++i) {
            all_indices_[i] = static_cast<int>(i);
        }
        pool = &all_indices_;
    }

    PlaneModel best;
    int best_count = 0;
    const size_t min_cloud = static_cast<size_t>(std::max(16, opt.ground_min_inliers / 4));
    if (points_.size() >= min_cloud) {
        const int iterations = std::max(8, opt.ground_iterations);
        for (int iter = 0; iter < iterations; ++iter) {
            PlaneModel candidate;
            if (!sample_candidate(points_, *pool, rng_, candidate)) {
                continue;
            }
            if (!candidate_is_plausible(candidate, opt)) {
                continue;
            }
            const int count = count_inliers(points_, candidate, opt, nullptr);
            if (count > best_count) {
                best_count = count;
                best = candidate;
                // Adaptive stop: once most of the cloud agrees, further sampling
                // buys nothing and we would rather hand the cycles back.
                const float ratio = static_cast<float>(best_count) / static_cast<float>(points_.size());
                if (ratio > 0.55f && iter >= 12) {
                    break;
                }
            }
        }
    }

    bool fit_ok = false;
    GroundPlane measured;
    if (best_count >= opt.ground_min_inliers &&
        static_cast<float>(best_count) / static_cast<float>(points_.size()) >= opt.ground_min_inlier_ratio) {
        count_inliers(points_, best, opt, &inliers_);

        // Two refit passes: the first pulls the plane off the noisy 3-point model
        // onto the true floor, the second re-collects inliers the model had missed.
        PlaneModel refined = best;
        for (int pass = 0; pass < 2; ++pass) {
            PlaneModel next;
            if (!refit_least_squares(points_, inliers_, next)) {
                break;
            }
            refined = next;
            count_inliers(points_, refined, opt, &inliers_);
        }

        if (candidate_is_plausible(refined, opt) &&
            inliers_.size() >= static_cast<size_t>(opt.ground_min_inliers)) {
            measured.normal = refined.n;
            measured.d = refined.d;
            measured.height_mm = refined.d;
            measured.tilt_deg = plane_tilt_deg(refined.n);
            measured.inlier_count = static_cast<int>(inliers_.size());
            measured.inlier_ratio = static_cast<float>(inliers_.size()) / static_cast<float>(points_.size());
            measured.valid = true;
            fit_ok = true;
        }
    }

    // A fit that disagrees violently with the running estimate is usually a
    // tabletop or a wall momentarily winning the vote. Treat it as a miss. If it
    // is real it wins anyway once the hold window expires and the lock resets.
    if (fit_ok && smoothed_.valid) {
        const float dot = std::min(1.0f, std::max(-1.0f, measured.normal.dot(smoothed_.normal)));
        const float delta_deg = static_cast<float>(std::acos(dot) * 180.0 / CV_PI);
        const bool jumped = delta_deg > 25.0f || std::fabs(measured.height_mm - smoothed_.height_mm) > 350.0f;
        if (jumped && miss_frames_ < opt.ground_hold_frames) {
            fit_ok = false;
        }
    }

    if (fit_ok) {
        miss_frames_ = 0;
        confirm_hits_ = std::min(confirm_hits_ + 1, opt.ground_confirm_frames);
        if (!smoothed_.valid) {
            smoothed_ = measured;
        } else {
            // Both normals are orient_up()'d, so there is no sign ambiguity to
            // resolve before blending.
            const float a = std::min(1.0f, std::max(0.05f, opt.ground_smoothing));
            cv::Vec3f blended = a * measured.normal + (1.0f - a) * smoothed_.normal;
            const float len = static_cast<float>(cv::norm(blended));
            if (len > 1.0e-6f) {
                smoothed_.normal = blended / len;
            }
            smoothed_.d = a * measured.d + (1.0f - a) * smoothed_.d;
            smoothed_.height_mm = smoothed_.d;
            smoothed_.tilt_deg = plane_tilt_deg(smoothed_.normal);
            smoothed_.inlier_count = measured.inlier_count;
            smoothed_.inlier_ratio = measured.inlier_ratio;
            smoothed_.valid = true;
        }
        state.plane = smoothed_;
        state.plane.valid = confirm_hits_ >= opt.ground_confirm_frames;
        state.stale = false;
    } else {
        ++miss_frames_;
        if (smoothed_.valid && miss_frames_ <= opt.ground_hold_frames) {
            state.plane = smoothed_;
            state.plane.valid = confirm_hits_ >= opt.ground_confirm_frames;
            state.stale = true;
        } else {
            smoothed_ = GroundPlane{};
            confirm_hits_ = 0;
            state.plane.valid = false;
        }
    }

    return state;
}

} // namespace tactical_rescue
