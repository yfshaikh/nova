/**
 * GPU-resident 4× ZED X fused point cloud viewer (C++).
 *
 * Matches the rig extrinsics from fused_pc_fast.py / zed_360_panorama_gpu.py.
 * Each camera thread grabs + retrieveMeasure(XYZRGBA, MEM::GPU). The Stereolabs
 * depth-sensing GLViewer copies GPU→GL via cudaMemcpyDeviceToDevice (no CPU
 * get_data()). Per-camera cam→rig transforms are applied as model matrices.
 *
 * Build on the Jetson (ZED SDK + CUDA + GLEW + GLUT required):
 *   cd depth_reproject/cpp && mkdir -p build && cd build
 *   cmake .. && make -j$(nproc)
 * Run:
 *   ./NovaFusedCloud
 * Controls: left-drag orbit, right-drag pan, wheel zoom, q/ESC quit
 */

#include "GLViewer.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

using namespace sl;

struct RigCameraConfig {
    const char* name;
    unsigned serial;
    float yaw_deg;
    float pitch_deg;
    float roll_deg;
    float tx, ty, tz;
};

// Same rig as the Python pipelines
static const RigCameraConfig RIG[] = {
    {"front", 46108623, 0.f, 0.f, 0.f, 0.f, 0.f, 0.711f},
    {"right", 47860268, 90.f, 0.f, 0.f, 0.660f, 0.f, -0.216f},
    {"back", 49004271, 180.f, 0.f, 0.f, 0.f, 0.f, -1.422f},
    {"left", 43765493, -90.f, 0.f, 0.f, -0.660f, 0.f, -0.216f},
};

static const RESOLUTION kResolution = RESOLUTION::SVGA;
static const int kFps = 15;

static sl::Matrix3f matMul(sl::Matrix3f a, sl::Matrix3f b) {
    sl::Matrix3f out;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out(r, c) = a(r, 0) * b(0, c) + a(r, 1) * b(1, c) + a(r, 2) * b(2, c);
        }
    }
    return out;
}

static sl::Matrix3f rotY(float a) {
    const float c = std::cos(a), s = std::sin(a);
    sl::Matrix3f m;
    m.setIdentity();
    m(0, 0) = c;
    m(0, 2) = s;
    m(2, 0) = -s;
    m(2, 2) = c;
    return m;
}

static sl::Matrix3f rotX(float a) {
    const float c = std::cos(a), s = std::sin(a);
    sl::Matrix3f m;
    m.setIdentity();
    m(1, 1) = c;
    m(1, 2) = -s;
    m(2, 1) = s;
    m(2, 2) = c;
    return m;
}

static sl::Matrix3f rotZ(float a) {
    const float c = std::cos(a), s = std::sin(a);
    sl::Matrix3f m;
    m.setIdentity();
    m(0, 0) = c;
    m(0, 1) = -s;
    m(1, 0) = s;
    m(1, 1) = c;
    return m;
}

/** cam_to_rig rotation + translation — same as Python cam_to_world + t. */
static Transform makeCamToRig(const RigCameraConfig& cfg) {
    const float y = cfg.yaw_deg * static_cast<float>(M_PI) / 180.f;
    const float p = cfg.pitch_deg * static_cast<float>(M_PI) / 180.f;
    const float r = cfg.roll_deg * static_cast<float>(M_PI) / 180.f;
    const sl::Matrix3f R = matMul(matMul(rotY(y), rotX(p)), rotZ(r));

    Transform T;
    T.setIdentity();
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            T(r, c) = R(r, c);
    T.setTranslation(Translation(cfg.tx, cfg.ty, cfg.tz));
    return T;
}

struct CameraWorker {
    RigCameraConfig config;
    Camera zed;
    Mat point_cloud;
    CUstream cuda_stream = nullptr;
    Resolution pc_res;
    std::atomic<bool> running{false};
    std::atomic<bool> frame_ready{false};
    std::atomic<float> zed_fps{0.f};
    std::thread thread;
};

static void acquisitionLoop(CameraWorker* worker) {
    RuntimeParameters rt;
    while (worker->running.load()) {
        // Hold the GPU mat stable until the GL thread has consumed it.
        if (worker->frame_ready.load()) {
            sl::sleep_ms(1);
            continue;
        }
        if (worker->zed.grab(rt) == ERROR_CODE::SUCCESS) {
            worker->zed.retrieveMeasure(
                worker->point_cloud,
                MEASURE::XYZRGBA,
                MEM::GPU,
                worker->pc_res
            );
            worker->zed_fps.store(worker->zed.getCurrentFPS());
            worker->frame_ready.store(true);
        }
    }
}

int main(int argc, char** argv) {
    constexpr size_t kNumCams = sizeof(RIG) / sizeof(RIG[0]);
    std::vector<CameraWorker> workers;
    workers.reserve(kNumCams);

    InitParameters init;
    init.camera_resolution = kResolution;
    init.camera_fps = kFps;
    init.depth_mode = DEPTH_MODE::PERFORMANCE;
    init.coordinate_units = UNIT::METER;
    init.coordinate_system = COORDINATE_SYSTEM::IMAGE; // X-right, Y-down, Z-forward (Python rig)
    init.sdk_verbose = 1;

    Resolution pc_res;
    bool first_open = true;

    for (size_t i = 0; i < kNumCams; ++i) {
        workers.emplace_back();
        CameraWorker& w = workers.back();
        w.config = RIG[i];
        init.input.setFromSerialNumber(w.config.serial);

        const ERROR_CODE err = w.zed.open(init);
        if (err != ERROR_CODE::SUCCESS) {
            std::cerr << "Failed to open " << w.config.name << " (SN "
                      << w.config.serial << "): " << toString(err) << std::endl;
            return EXIT_FAILURE;
        }

        // Lock exposure — avoids known 4× ZED X frame-drop issue with auto-exposure in motion
        w.zed.setCameraSettings(VIDEO_SETTINGS::AEC_AGC, 0);
        w.zed.setCameraSettings(VIDEO_SETTINGS::EXPOSURE, 50);

        w.cuda_stream = w.zed.getCUDAStream();
        w.pc_res = w.zed.getCameraInformation().camera_configuration.resolution;
        if (first_open) {
            pc_res = w.pc_res;
            first_open = false;
        }

        std::cout << "Opened " << w.config.name << " SN " << w.config.serial
                  << "  pc " << w.pc_res.width << "x" << w.pc_res.height << std::endl;
    }

    GLViewer viewer;
    const GLenum gl_err = viewer.initMulti(argc, argv, pc_res);
    if (gl_err != GLEW_OK) {
        std::cerr << "OpenGL init failed: " << glewGetErrorString(gl_err) << std::endl;
        return EXIT_FAILURE;
    }

    for (auto& w : workers) {
        viewer.registerCamera(w.config.serial, w.cuda_stream, makeCamToRig(w.config));
        w.running.store(true);
        w.thread = std::thread(acquisitionLoop, &w);
    }

    std::cout << "Running. Orbit: left-drag | Pan: right-drag | Zoom: wheel | Quit: q/ESC\n";

    auto t_fps = std::chrono::steady_clock::now();
    int frames = 0;
    float min_cam_fps = 0.f;

    while (viewer.isAvailable()) {
        for (auto& w : workers) {
            if (w.frame_ready.load()) {
                viewer.updatePointCloud(w.config.serial, w.point_cloud);
                w.frame_ready.store(false);
                const float f = w.zed_fps.load();
                min_cam_fps = (min_cam_fps <= 0.f) ? f : std::min(min_cam_fps, f);
            }
        }

        ++frames;
        const auto now = std::chrono::steady_clock::now();
        const float elapsed = std::chrono::duration<float>(now - t_fps).count();
        if (elapsed >= 1.f) {
            const float loop_fps = static_cast<float>(frames) / elapsed;
            viewer.setFpsText(loop_fps);
            std::cout << "\rloop " << loop_fps << " fps  |  min cam " << min_cam_fps
                      << " fps          " << std::flush;
            frames = 0;
            min_cam_fps = 0.f;
            t_fps = now;
        }
    }

    for (auto& w : workers) {
        w.running.store(false);
        if (w.thread.joinable())
            w.thread.join();
        w.point_cloud.free();
        w.zed.close();
    }

    std::cout << std::endl;
    return EXIT_SUCCESS;
}
