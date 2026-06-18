/**
 * NovaPanoramaGPU — 4x ZED 360 cylindrical panorama with depth-corrected near
 * field, computed END-TO-END on the GPU in C++.
 *
 * This is the C++/CUDA port of zed_360_panorama_gpu.py + reproject_gpu.py.
 * Differences (all wins):
 *   - Base remap + feather blend: CUDA kernel (was cv2.remap on the CPU).
 *   - Depth reprojection + z-buffer: single atomicMin over a packed uint64
 *     (depth<<32 | BGRA) — nearest color in one pass (was torch scatter + a
 *     masked color write).
 *   - Pinhole sealing: in-kernel splat radius (was cv2.morphologyEx + dilate).
 *   - Display: CUDA-GL interop PBO, zero CPU copies (was pano.cpu().numpy()).
 *   - One XYZRGBA retrieve gives geometry + color (was XYZ + a separate image).
 *
 * Build (Jetson, ZED SDK + CUDA + GLEW + GLUT):
 *   cd depth_reproject/cpp_panorama && mkdir -p build && cd build
 *   cmake .. && make -j$(nproc)
 *   ./NovaPanoramaGPU            # q / ESC to quit
 */

#include "PanoramaViewer.hpp"
#include "kernels.cuh"

#include <sl/Camera.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <thread>
#include <vector>

using namespace sl;

// ----------------------------------------------------------------------------
// CONFIG — same rig as the Python pipelines
// ----------------------------------------------------------------------------
struct RigCameraConfig {
    const char* name;
    unsigned serial;
    float yaw_deg, pitch_deg, roll_deg;
    float tx, ty, tz;
};

static const RigCameraConfig RIG[] = {
    {"front", 46108623, 0.f, 0.f, 0.f, 0.f, 0.f, 0.711f},
    {"right", 47860268, 90.f, 0.f, 0.f, 0.660f, 0.f, -0.216f},
    {"back", 49004271, 180.f, 0.f, 0.f, 0.f, 0.f, -1.422f},
    {"left", 43765493, -90.f, 0.f, 0.f, -0.660f, 0.f, -0.216f},
};
static constexpr size_t kNumCams = sizeof(RIG) / sizeof(RIG[0]);

static const RESOLUTION kResolution = RESOLUTION::SVGA;
static const int kFps = 15;
static const float kScale = 240.f; // pano pixels/radian
static const float kNearMax = 8.f; // meters; closer gets depth reprojection
static const int kPointStride = 2; // subsample the cloud (every Nth px)
static const int kSplat = 2;       // overlay splat radius (seals pinholes); each
                                   // point paints a (2r+1)^2 block. 2 fills the
                                   // gaps left by kPointStride=2.
static const bool kTiming = true;

// Depth is the heaviest per-frame cost and the base panorama doesn't use it.
// Grab color every frame (smooth base @ kFps) but only run the depth engine
// every Nth grab — the near-field overlay refreshes at kFps/kDepthEveryN while
// display + base stay at full rate. 1 = depth every frame (old behavior).
//   kFps=15, kDepthEveryN=3  ->  20 depth computes/sec instead of 60.
static const int kDepthEveryN = 3;

// ----------------------------------------------------------------------------
// Geometry helpers — cam_to_world = Ry(yaw) * Rx(pitch) * Rz(roll)
// ----------------------------------------------------------------------------
static void mat3mul(const float a[9], const float b[9], float out[9]) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            out[r * 3 + c] = a[r * 3 + 0] * b[0 * 3 + c] +
                             a[r * 3 + 1] * b[1 * 3 + c] +
                             a[r * 3 + 2] * b[2 * 3 + c];
}

static void camToWorld(float yaw_deg, float pitch_deg, float roll_deg, float R[9]) {
    const float y = yaw_deg * (float)M_PI / 180.f;
    const float p = pitch_deg * (float)M_PI / 180.f;
    const float r = roll_deg * (float)M_PI / 180.f;
    const float Ry[9] = {cosf(y), 0, sinf(y), 0, 1, 0, -sinf(y), 0, cosf(y)};
    const float Rx[9] = {1, 0, 0, 0, cosf(p), -sinf(p), 0, sinf(p), cosf(p)};
    const float Rz[9] = {cosf(r), -sinf(r), 0, sinf(r), cosf(r), 0, 0, 0, 1};
    float tmp[9];
    mat3mul(Ry, Rx, tmp);
    mat3mul(tmp, Rz, R);
}

// Approximate (chamfer) L2 distance transform: distance from each valid pixel
// to the nearest invalid pixel. Used as the feather blend weight.
static void chamferDT(const std::vector<unsigned char>& valid, int w, int h,
                      std::vector<float>& dist) {
    const float INF = 1e10f;
    dist.assign((size_t)w * h, 0.f);
    for (int i = 0; i < w * h; ++i)
        dist[i] = valid[i] ? INF : 0.f;

    const float d1 = 1.0f, d2 = 1.41421356f;
    auto at = [&](int x, int y) -> float& { return dist[(size_t)y * w + x]; };

    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float d = at(x, y);
            if (x > 0) d = fminf(d, at(x - 1, y) + d1);
            if (y > 0) d = fminf(d, at(x, y - 1) + d1);
            if (x > 0 && y > 0) d = fminf(d, at(x - 1, y - 1) + d2);
            if (x < w - 1 && y > 0) d = fminf(d, at(x + 1, y - 1) + d2);
            at(x, y) = d;
        }
    for (int y = h - 1; y >= 0; --y)
        for (int x = w - 1; x >= 0; --x) {
            float d = at(x, y);
            if (x < w - 1) d = fminf(d, at(x + 1, y) + d1);
            if (y < h - 1) d = fminf(d, at(x, y + 1) + d1);
            if (x < w - 1 && y < h - 1) d = fminf(d, at(x + 1, y + 1) + d2);
            if (x > 0 && y < h - 1) d = fminf(d, at(x - 1, y + 1) + d2);
            at(x, y) = d;
        }
}

// Build the static inverse-warp table for one camera's rotation-only base.
static void buildCylMaps(float fx, float fy, float cx, float cy, int iw, int ih,
                         const float R_cw[9], float scale, int pw, int ph,
                         std::vector<float>& mapx, std::vector<float>& mapy,
                         std::vector<unsigned char>& valid) {
    mapx.assign((size_t)pw * ph, -1.f);
    mapy.assign((size_t)pw * ph, -1.f);
    valid.assign((size_t)pw * ph, 0);

    // R_wc = R_cw^T
    const float Rt[9] = {R_cw[0], R_cw[3], R_cw[6],
                         R_cw[1], R_cw[4], R_cw[7],
                         R_cw[2], R_cw[5], R_cw[8]};

    for (int gy = 0; gy < ph; ++gy) {
        for (int gx = 0; gx < pw; ++gx) {
            const float phi = (gx - pw / 2.f) / scale;
            const float hh = (gy - ph / 2.f) / scale;
            const float rwx = sinf(phi), rwy = hh, rwz = cosf(phi);
            // ray_cam = R_wc * ray_world
            const float xc = Rt[0] * rwx + Rt[1] * rwy + Rt[2] * rwz;
            const float yc = Rt[3] * rwx + Rt[4] * rwy + Rt[5] * rwz;
            const float zc = Rt[6] * rwx + Rt[7] * rwy + Rt[8] * rwz;
            if (zc <= 1e-6f) continue;
            const float u = fx * xc / zc + cx;
            const float v = fy * yc / zc + cy;
            if (u < 0.f || u > iw - 1 || v < 0.f || v > ih - 1) continue;
            const size_t pid = (size_t)gy * pw + gx;
            mapx[pid] = u;
            mapy[pid] = v;
            valid[pid] = 1;
        }
    }
}

template <typename T>
static T* uploadVec(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

// ----------------------------------------------------------------------------
// Threaded ZED capture — keeps the latest color image + XYZRGBA cloud on GPU
// ----------------------------------------------------------------------------
struct CameraWorker {
    RigCameraConfig config;
    CamExtrinsic ext;
    Camera zed;
    Mat image;       // LEFT, BGRA, GPU
    Mat cloud;       // XYZRGBA, GPU
    CUstream stream = nullptr;
    int img_w = 0, img_h = 0;
    float fx = 0, fy = 0, cx = 0, cy = 0;
    std::atomic<bool> running{false};
    std::atomic<bool> frame_ready{false};
    std::atomic<bool> healthy{true};
    std::atomic<float> zed_fps{0.f};
    std::thread thread;
};

static void acquisitionLoop(CameraWorker* w) {
    RuntimeParameters rt;
    Resolution res((size_t)w->img_w, (size_t)w->img_h);
    int consecutive_fail = 0;
    unsigned long frame_idx = 0;
    while (w->running.load()) {
        if (w->frame_ready.load()) { // wait until the GL thread consumes
            sl::sleep_ms(1);
            continue;
        }
        // Only pay for the depth engine every Nth grab. enable_depth=false makes
        // grab() skip stereo matching entirely -> big GPU/thermal savings while
        // color (the base layer) still arrives at full rate.
        const bool want_depth = (kDepthEveryN <= 1) || (frame_idx % kDepthEveryN == 0);
        rt.enable_depth = want_depth;
        const ERROR_CODE gerr = w->zed.grab(rt);
        if (gerr == ERROR_CODE::SUCCESS) {
            if (consecutive_fail > 0) {
                std::cerr << "[" << w->config.name << "] recovered after "
                          << consecutive_fail << " failed grabs" << std::endl;
                consecutive_fail = 0;
            }
            w->zed.retrieveImage(w->image, VIEW::LEFT, MEM::GPU, res);
            // Skip the cloud retrieve on no-depth frames; the overlay reuses the
            // last cloud (near field just lags slightly, base stays sharp).
            if (want_depth)
                w->zed.retrieveMeasure(w->cloud, MEASURE::XYZRGBA, MEM::GPU, res);
            cudaStreamSynchronize(w->stream); // ensure GPU data is ready for our kernels
            w->zed_fps.store(w->zed.getCurrentFPS());
            w->frame_ready.store(true);
            w->healthy.store(true);
            ++frame_idx;
        } else {
            // Argus/GMSL pipelines on Jetson can drop out after long runs. Don't
            // busy-spin grab() on failure (that hammers the daemon and makes it
            // worse) — back off, log sparingly, and keep trying to recover.
            ++consecutive_fail;
            if (consecutive_fail == 1 ||
                consecutive_fail == 30 ||
                (consecutive_fail % 150) == 0) {
                std::cerr << "[" << w->config.name << "] grab failed ("
                          << toString(gerr) << "), x" << consecutive_fail
                          << " — backing off" << std::endl;
            }
            if (consecutive_fail > 15) w->healthy.store(false);
            sl::sleep_ms(consecutive_fail < 30 ? 5 : 33);
        }
    }
}

int main() {
    std::array<CameraWorker, kNumCams> workers;

    InitParameters init;
    init.camera_resolution = kResolution;
    init.camera_fps = kFps;
    init.depth_mode = DEPTH_MODE::PERFORMANCE;
    init.coordinate_units = UNIT::METER;
    init.coordinate_system = COORDINATE_SYSTEM::IMAGE; // X-right, Y-down, Z-forward
    init.sdk_verbose = 1;

    for (size_t i = 0; i < kNumCams; ++i) {
        CameraWorker& w = workers[i];
        w.config = RIG[i];
        camToWorld(w.config.yaw_deg, w.config.pitch_deg, w.config.roll_deg, w.ext.R);
        w.ext.t[0] = w.config.tx;
        w.ext.t[1] = w.config.ty;
        w.ext.t[2] = w.config.tz;

        init.input.setFromSerialNumber(w.config.serial);
        const ERROR_CODE err = w.zed.open(init);
        if (err != ERROR_CODE::SUCCESS) {
            std::cerr << "Failed to open " << w.config.name << " (SN " << w.config.serial
                      << "): " << toString(err) << std::endl;
            return EXIT_FAILURE;
        }
        w.zed.setCameraSettings(VIDEO_SETTINGS::AEC_AGC, 0); // lock exposure (4x ZED X drops)
        w.zed.setCameraSettings(VIDEO_SETTINGS::EXPOSURE, 50);
        w.stream = w.zed.getCUDAStream();

        const auto info = w.zed.getCameraInformation();
        const auto calib = info.camera_configuration.calibration_parameters.left_cam;
        const auto res = info.camera_configuration.resolution;
        w.fx = calib.fx; w.fy = calib.fy; w.cx = calib.cx; w.cy = calib.cy;
        w.img_w = (int)res.width; w.img_h = (int)res.height;
        std::cout << "Opened " << w.config.name << " " << w.img_w << "x" << w.img_h
                  << " fx=" << w.fx << " fy=" << w.fy << std::endl;
    }

    // --- panorama geometry from the reference camera --------------------------
    const CameraWorker& ref = workers[0];
    const int pano_w = (int)lroundf(2.f * (float)M_PI * kScale);
    const int pano_h = (int)lroundf(kScale * ref.img_h / ref.fy);
    std::cout << "panorama: " << pano_w << " x " << pano_h << std::endl;

    // --- build base maps + normalized feather weights (host, once) -----------
    std::vector<std::vector<float>> mapx(kNumCams), mapy(kNumCams), wgt(kNumCams);
    std::vector<std::vector<float>> dist(kNumCams);
    std::vector<float> total((size_t)pano_w * pano_h, 0.f);
    for (size_t i = 0; i < kNumCams; ++i) {
        std::vector<unsigned char> valid;
        buildCylMaps(workers[i].fx, workers[i].fy, workers[i].cx, workers[i].cy,
                     workers[i].img_w, workers[i].img_h, workers[i].ext.R,
                     kScale, pano_w, pano_h, mapx[i], mapy[i], valid);
        chamferDT(valid, pano_w, pano_h, dist[i]);
        for (size_t p = 0; p < total.size(); ++p)
            total[p] += dist[i][p];
    }
    for (size_t i = 0; i < kNumCams; ++i) {
        wgt[i].assign((size_t)pano_w * pano_h, 0.f);
        for (size_t p = 0; p < total.size(); ++p)
            if (total[p] > 0.f)
                wgt[i][p] = dist[i][p] / total[p];
    }

    // --- upload static tables + allocate GPU scratch -------------------------
    std::vector<float*> d_mapx(kNumCams), d_mapy(kNumCams), d_wgt(kNumCams);
    for (size_t i = 0; i < kNumCams; ++i) {
        d_mapx[i] = uploadVec(mapx[i]);
        d_mapy[i] = uploadVec(mapy[i]);
        d_wgt[i] = uploadVec(wgt[i]);
    }
    ::float4* d_accum = nullptr;
    unsigned long long* d_zbuf = nullptr;
    cudaMalloc(&d_accum, (size_t)pano_w * pano_h * sizeof(::float4));
    cudaMalloc(&d_zbuf, (size_t)pano_w * pano_h * sizeof(unsigned long long));

    // --- viewer (CUDA-GL interop) --------------------------------------------
    PanoramaViewer viewer;
    if (!viewer.init(pano_w, pano_h, pano_w / 2, pano_h / 2, "NovaPanoramaGPU (q to quit)")) {
        std::cerr << "viewer init failed" << std::endl;
        return EXIT_FAILURE;
    }

    // --- start capture threads, wait for first frames ------------------------
    for (auto& w : workers) {
        w.running.store(true);
        w.thread = std::thread(acquisitionLoop, &w);
    }
    std::cout << "waiting for first frames..." << std::endl;
    {
        const auto t0 = std::chrono::steady_clock::now();
        bool all = false;
        while (!all) {
            all = true;
            for (auto& w : workers)
                all &= w.frame_ready.load();
            if (std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count() > 15.f) {
                std::cerr << "timed out waiting for camera frames" << std::endl;
                return EXIT_FAILURE;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    std::cout << "running. q/ESC quit  |  b: toggle base  o: toggle overlay"
              << std::endl;
    auto t_fps = std::chrono::steady_clock::now();
    auto last_render = std::chrono::steady_clock::now();
    int frames = 0;
    bool show_base = true;
    bool show_overlay = true;

    while (viewer.isAvailable()) {
        switch (viewer.consumeKey()) {
            case 'b': case 'B':
                show_base = !show_base;
                std::cout << "\n[base " << (show_base ? "ON" : "OFF") << "]\n";
                break;
            case 'o': case 'O':
                show_overlay = !show_overlay;
                std::cout << "\n[overlay " << (show_overlay ? "ON" : "OFF") << "]\n";
                break;
            default: break;
        }

        // Only render a COMPLETE set of fresh frames. Rendering whatever happens
        // to be ready on every fast loop iteration produces black/partial frames
        // between the 15 fps camera grabs -> flicker. Wait for all cameras, but
        // don't starve the display if one camera lags (>80 ms => render partial).
        int nready = 0;
        bool all = true;
        for (auto& w : workers) {
            const bool r = w.frame_ready.load();
            all &= r;
            nready += r ? 1 : 0;
        }
        const auto t_start = std::chrono::steady_clock::now();
        const float since_ms =
            std::chrono::duration<float, std::milli>(t_start - last_render).count();
        if (!(all || (nready > 0 && since_ms > 80.f))) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        last_render = t_start;

        ::uchar4* pano = viewer.mapBuffer();
        if (!pano) break;

        // Snapshot readiness ONCE so base + overlay use the same set of frames.
        std::array<bool, kNumCams> ready;
        for (size_t i = 0; i < kNumCams; ++i)
            ready[i] = workers[i].frame_ready.load();

        // BASE: rotation-only feather-blended background. When the overlay is on,
        // drop the near field from the base so it isn't doubled under the overlay
        // (the overlay redraws it depth-correct). With the overlay off, keep the
        // full base so there are no black holes.
        const float base_near_drop = show_overlay ? kNearMax : 0.f;
        launchClearAccum(d_accum, pano_w, pano_h, 0);
        if (show_base) {
            for (size_t i = 0; i < kNumCams; ++i) {
                if (!ready[i]) continue;
                CameraWorker& w = workers[i];
                const ::uchar4* img = reinterpret_cast<const ::uchar4*>(w.image.getPtr<sl::uchar4>(MEM::GPU));
                if (!img) continue;
                const int istep = (int)(w.image.getStepBytes(MEM::GPU) / sizeof(sl::uchar4));
                const ::float4* pc = reinterpret_cast<const ::float4*>(w.cloud.getPtr<sl::float4>(MEM::GPU));
                const int cstep = pc ? (int)(w.cloud.getStepBytes(MEM::GPU) / sizeof(sl::float4)) : 0;
                launchAccumBase(img, istep, w.img_w, w.img_h,
                                d_mapx[i], d_mapy[i], d_wgt[i],
                                pc, cstep, base_near_drop,
                                d_accum, pano_w, pano_h, 0);
            }
        }
        launchFinalizeBase(d_accum, pano, pano_w, pano_h, 0);

        // OVERLAY: depth reprojection into one global z-buffer, composite on top.
        launchClearZ(d_zbuf, pano_w, pano_h, 0);
        if (show_overlay) {
            for (size_t i = 0; i < kNumCams; ++i) {
                if (!ready[i]) continue;
                CameraWorker& w = workers[i];
                const ::float4* pc = reinterpret_cast<const ::float4*>(w.cloud.getPtr<sl::float4>(MEM::GPU));
                if (!pc) continue;
                const int pstep = (int)(w.cloud.getStepBytes(MEM::GPU) / sizeof(sl::float4));
                launchScatterOverlay(pc, pstep, w.img_w, w.img_h, w.ext, kScale,
                                     pano_w, pano_h, kNearMax, kPointStride, kSplat, d_zbuf, 0);
            }
        }
        launchComposite(d_zbuf, pano, pano_w, pano_h, 0);

        cudaDeviceSynchronize(); // kernels done; cloud/image can be overwritten now

        float min_cam_fps = 0.f;
        for (size_t i = 0; i < kNumCams; ++i) {
            if (!ready[i]) continue;
            const float f = workers[i].zed_fps.load();
            min_cam_fps = (min_cam_fps <= 0.f) ? f : std::min(min_cam_fps, f);
            workers[i].frame_ready.store(false); // release for next grab
        }

        viewer.unmapAndDraw();

        ++frames;
        const auto now = std::chrono::steady_clock::now();
        const float el = std::chrono::duration<float>(now - t_fps).count();
        if (el >= 1.f) {
            const float loop_fps = frames / el;
            char title[160];
            snprintf(title, sizeof(title), "NovaPanoramaGPU — %.1f fps (cam %.1f)",
                     loop_fps, min_cam_fps);
            viewer.setTitle(title);
            if (kTiming) {
                const float ms = std::chrono::duration<float, std::milli>(now - t_start).count();
                std::cout << "\rloop " << loop_fps << " fps  | frame " << ms
                          << " ms | min cam " << min_cam_fps << " fps        " << std::flush;
            }
            frames = 0;
            t_fps = now;
        }
    }

    for (auto& w : workers) {
        w.running.store(false);
        if (w.thread.joinable()) w.thread.join();
        w.image.free();
        w.cloud.free();
        w.zed.close();
    }
    for (size_t i = 0; i < kNumCams; ++i) {
        cudaFree(d_mapx[i]);
        cudaFree(d_mapy[i]);
        cudaFree(d_wgt[i]);
    }
    cudaFree(d_accum);
    cudaFree(d_zbuf);
    viewer.close();

    std::cout << std::endl;
    return EXIT_SUCCESS;
}
