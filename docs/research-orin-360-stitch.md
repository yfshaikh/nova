# Real-Time 360° Surround-View Stitching on Jetson Orin with ZED X Cameras
## Research Report — June 2026

---

## Executive Summary

The CPU bottleneck in the existing depth-reprojection approach can be eliminated by porting the per-pixel scatter pass to the GPU using either **custom CUDA kernels (via Numba `@cuda.jit`)** or **PyTorch `scatter_reduce_`** on CUDA tensors — both are confirmed available on JetPack 6 / CUDA 12.6. NVIDIA DriveWorks 360 surround-view is **DRIVE-platform only** and cannot run on any Jetson dev kit; no Isaac ROS equivalent for panoramic stitching exists. The automotive industry standard for parking/close-range 360 is **ground-plane IPM/BEV**, not eye-level cylindrical panoramas, and RidgeRun ships a CUDA-accelerated commercial implementation at 30–60 FPS on Jetson Orin. The most practical path to >15 FPS parallax-tolerant surround view for a small team is a **seam-driven approach with depth-guided seam placement + GPU remap** using OpenCV CUDA (built from source) or VPI, bypassing full forward-scatter entirely. A full forward-scatter GPU pipeline remains viable if parallax accuracy is non-negotiable, but requires a custom CUDA kernel with `atomicMin` z-buffer — the scattered global memory writes are slow, so tile-based strategies are needed.

---

## 1. GPU-Accelerating the Per-Pixel Depth Reprojection + Z-Buffer Scatter

### The Problem Restated

The existing NumPy implementation does per-pixel forward scatter: for each source pixel, compute its 3D world coordinate (via depth + camera intrinsics/extrinsics), project into the target panorama canvas, and write color if the projected depth is less than the current z-buffer value. Pure NumPy on Orin's CPU runs at ~1 FPS because: (a) loop-level iteration is serialized, and (b) Orin's 12-core Cortex-A78AE CPU is outclassed by 2048 Ampere CUDA cores sitting idle.

### Option A — PyTorch `scatter_reduce_` / `index_put_`

**Status: Confirmed available on Jetson Orin with JetPack 6.**

PyTorch 2.5.0a0 (NVIDIA's aarch64 wheel for JetPack 6.1/6.2) ships with full CUDA support and `cuda.is_available() == True` is verified on Jetson Orin Nano/NX/AGX Orin. `torch.Tensor.scatter_reduce_` is documented in PyTorch 2.x stable with `reduce='amin'` (atomic minimum), which maps directly onto the z-buffer min-depth write. `index_put_` with `accumulate=False` also works for scatter writes. Note: the PyTorch docs explicitly warn that `scatter_reduce_` **behaves nondeterministically** on CUDA devices — acceptable for rendering but worth knowing. No specific Jetson Orin scatter benchmark was found; the key risk is that Jetson's iGPU may underperform discrete GPUs on atomically contended global memory writes.

**Effort:** Low — PyTorch is already installed on Orin. Requires replacing NumPy array ops with tensor ops and moving arrays to `.cuda()` device.

**Sources:** [NVIDIA PyTorch Jetson install docs](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) · [ninjalabo.ai JetPack 6 PyTorch guide](https://ninjalabo.ai/blogs/jetson_pytorch.html) · [PyTorch scatter_reduce_ docs](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html)

### Option B — OpenCV CUDA Module (`cv2.cuda.remap`, `warpPerspective`, etc.)

**Status: NOT in the default JetPack OpenCV. Must be compiled from source.**

The JetPack OpenCV package (version 4.5.x in JetPack 6.1) is the standard Ubuntu build **without CUDA support**. Every Jetson-focused guide confirms you must build OpenCV from source with `WITH_CUDA=ON`, `WITH_CUDNN=ON`, `OPENCV_DNN_CUDA=ON`, and the correct `CUDA_ARCH_BIN` for Ampere (8.7 for Orin). Once built, the `cv2.cuda` module provides:

- `cv2.cuda.remap(src, xmap, ymap, interpolation)` — applies a precomputed warp map, the key primitive for panoramic projection.
- `cv2.cuda.warpPerspective` / `warpAffine` — homography-based warp.
- `cv2.cuda.buildWarpPerspectiveMaps` / `buildWarpAffineMaps` — generates float32 xmap/ymap for subsequent `remap`.
- **No `buildWarpCylindricalMaps` or `buildWarpSphericalMaps`** — cylindrical/spherical maps must be computed manually (one-time CPU precomputation, then stored as GPU textures).
- Multiband blending: `opencv/modules/stitching/src/cuda/multiband_blend.cu` exists in the OpenCV source tree, but seam finders and exposure compensators **do not accept GpuMat**, so the full Stitcher API cannot be GPU-resident end-to-end.

Community reports (DEV Community, Q-engineering) indicate the build process takes ~4–6 hours on the Jetson itself; a cross-compile or Docker-based pre-built wheel is preferable.

**Limitation for z-buffer scatter:** `cv2.cuda.remap` is an *inverse* warp (pull), not a forward scatter. It cannot directly implement forward-scatter depth reprojection. It is, however, ideal for the per-camera undistort → cylindrical/spherical warp step, and for the final blending pass.

**Sources:** [DEV Community Jetson Orin NX OpenCV CUDA build guide](https://dev.to/rbelshevitz/accelerating-opencv-with-cuda-on-jetson-orin-nx-a-complete-build-guide-525j) · [OpenCV cuda::remap docs](https://docs.opencv.org/4.x/db/d29/group__cudawarping.html) · [NVIDIA forum on JetPack OpenCV CUDA](https://forums.developer.nvidia.com/t/opencv-library-with-cuda-support-on-jetson-jetpack-integration/331916) · [opencv multiband CUDA source](https://github.com/opencv/opencv/blob/master/modules/stitching/src/cuda/multiband_blend.cu)

### Option C — NVIDIA VPI (Vision Programming Interface)

**Status: Confirmed available, ships in JetPack. Version 3.2 in JetPack 6.1.**

VPI is included in JetPack (separate `libnvvpi` package, not compiled from source). On Jetson AGX Orin, it supports **CPU, CUDA, PVA, VIC, and OFA** backends. The OFA backend is Jetson AGX Orin-exclusive (not on Orin Nano/NX). Confirmed VPI 4.0 algorithm support:

- **Remap**: CPU, CUDA, VIC backends. Supports dense precomputed warp maps and custom warp maps for panoramic projection.
- **Perspective Warp**: CPU and CUDA backends.
- **Stereo Disparity**: CUDA and PVA backends (not relevant to stitching itself, but relevant to using ZED X depth in the pipeline).
- **No stitching-specific algorithm** exists in VPI — no built-in multi-camera surround compositor.
- VIC (Video and Image Compositor) backend for Remap is hardware-accelerated and very low-latency; it handles lens distortion correction natively with polynomial and fisheye distortion models.

**Key advantage over OpenCV CUDA:** VPI ships pre-built, zero compilation needed. VPI Remap on VIC is lower power consumption than CUDA for the warp step.

**Limitation for z-buffer scatter:** Same as OpenCV — VPI Remap is inverse warp only. There is no forward scatter primitive in VPI.

**Sources:** [VPI algorithm list (docs.nvidia.com)](https://docs.nvidia.com/vpi/algorithms.html) · [VPI Remap algo docs](https://docs.nvidia.com/vpi/algo_remap.html) · [NVIDIA VPI developer page](https://developer.nvidia.com/embedded/vpi) · [RidgeRun JetPack 6.1 components](https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Orin/JetPack_6.1/Getting_Started/Components)

### Option D — CuPy

**Status: Available on Jetson Orin, but with a documented performance pitfall.**

CuPy (NumPy-compatible GPU array library) runs on Jetson Orin (`cupy-cuda12x` wheel). It supports `__cuda_array_interface__` for zero-copy interop with PyTorch and PyCUDA. However, **benchmarks on AGX Orin show CuPy can be 2.6x slower than NumPy** for matrix multiply operations, traced to the CUTLASS GEMM kernel chosen for the Ampere iGPU architecture. This is specific to GEMM; elementwise and scatter-type operations (which don't rely on CUTLASS) may be unaffected, but no targeted scatter benchmark for Orin was found.

CuPy `cupyx.scatter_add` and fancy indexing with atomic-compatible ops exist. The API is near-identical to NumPy, so porting is low-friction. However, given the GEMM anomaly, benchmarking before committing is mandatory.

**Verdict:** Viable for prototyping due to minimal code changes, but test on target hardware before committing. PyTorch tensors (Option A) are likely more reliable given NVIDIA's explicit Jetson optimization.

**Sources:** [CuPy 2.6x slower than NumPy on AGX Orin issue](https://github.com/cupy/cupy/issues/8151) · [CuPy v12 release notes](https://medium.com/cupy-team/released-cupy-v12-4497315e811e) · [CuPy/Jetson Orin Nano issue tracker](https://github.com/cupy/cupy/issues/9216)

### Option E — Custom CUDA / Numba

**Status: Numba `@cuda.jit` is confirmed working on Jetson Orin (Compute Capability 8.7 ≥ 5.0 minimum).**

Numba compiles Python functions decorated with `@cuda.jit` to PTX at first call. On Jetson's unified memory architecture, data can be accessed from both CPU and GPU without explicit `cudaMemcpy`, reducing transfer overhead — a significant advantage over discrete GPU systems. JetsonHacks documents Numba CUDA usage on Jetson (2024). However, Numba's Python-level CUDA lacks direct access to NVIDIA's hardware rasterizer, so `atomicMin` for the z-buffer must be implemented manually in PTX-level intrinsics or via `cuda.atomic.min()` in Numba's CUDA dialect.

The NVIDIA developer forum discussion on efficient CUDA z-buffers recommends **tile-based rendering into shared memory** rather than naive global memory scatter, because "scattered reading and writing of global memory is quite slow." Naive `atomicMin` on every pixel into a 4K × 2K canvas causes massive global memory contention. A tiled approach bins pixels spatially and resolves per-tile in shared memory, dramatically reducing contention.

Writing a correct, performant tiled z-buffer scatter kernel is non-trivial (~2–4 developer-weeks for a small team unfamiliar with CUDA). This is the maximum-performance path but also the maximum-effort path.

**Sources:** [JetsonHacks Numba CUDA 2024](https://jetsonhacks.com/2024/01/15/cuda-programming-in-python-with-numba/) · [Numba CUDA docs](https://numba.readthedocs.io/en/stable/cuda-reference/kernel.html) · [NVIDIA CUDA z-buffer forum thread](https://forums.developer.nvidia.com/t/efficient-z-buffer-in-cuda-/416079)

### Practical GPU Acceleration Recommendation

For a **small team** seeking the fastest route to a working GPU pipeline:

1. **First:** Port the existing NumPy forward-scatter to **PyTorch CUDA tensors** using `scatter_reduce_(..., reduce='amin')`. No compilation required, minimal code changes, leverages JetPack-provided PyTorch 2.5. Expected speedup: 10–50x over NumPy CPU (rough estimate; depends on contention on Orin's Ampere iGPU — benchmark mandatory).
2. **Then:** Use **VPI Remap** (pre-built, zero-compile) for the per-camera undistort and cylindrical/spherical warp pass — this is the most expensive step in the non-scatter pipeline and maps perfectly to VPI's primitive.
3. **If scatter throughput is still insufficient:** Move to a custom **Numba CUDA kernel** with a tiled z-buffer strategy. This is the ceiling option.
4. **Build OpenCV CUDA from source** only if you need the multiband blending pipeline (Section 4) rather than the z-buffer approach.

---

## 2. NVIDIA DriveWorks 360° Surround View and Isaac ROS Equivalents

### DriveWorks on Jetson Orin

**Verdict: DriveWorks is strictly DRIVE-platform only. Cannot be installed on Jetson.**

An NVIDIA moderator explicitly confirmed in the developer forums: *"This is not supported as the DriveWorks is only for Drive AGX platform."* The DRIVE AGX and Jetson AGX share the same Orin SoC, but the software platforms have different driver stacks and automotive safety certifications. NVIDIA recommends **DeepStream SDK** for Jetson users needing similar pipelines, though DeepStream does not include a 360° panoramic compositor. The DRIVE AGX Orin Developer Kit (distinct from the Jetson AGX Orin dev kit) is available for purchase and runs DriveWorks, but it is not the same hardware product and carries a significantly higher price.

**Sources:** [NVIDIA forum: DriveWorks on Jetson AGX Orin not supported](https://forums.developer.nvidia.com/t/can-i-install-driveworks-sdk-on-jetson-agx-orin/268594) · [NVIDIA DRIVE AGX developer page](https://developer.nvidia.com/drive/agx) · [DRIVE AGX Orin ecosystem vendors](https://developer.nvidia.com/drive/ecosystem-orin)

### Isaac ROS on Jetson Orin

Isaac ROS (Robot Operating System) packages are CUDA-accelerated and run on Jetson Orin with ROS 2. The official ROS 2 Humble support path is Isaac ROS 3.x with the official dev container workflow. Available perception gems include:

- **Isaac ROS Image Pipeline** — GPU-accelerated `image_proc` equivalents (resize, crop, color convert, rectify) via NITROS. **No panoramic stitching or surround-view compositor.**
- **Isaac ROS DNN Stereo Depth (ESS)** — NVIDIA's Efficient Semi-Supervised stereo model, achieves **108.86 FPS on AGX Orin at 576×960**. ESS produces disparity + confidence maps and works with ZED-format stereo pairs. Note: ESS is tested against ROS 2 **Jazzy** as of recent docs; Humble compatibility should be verified against the specific release tag.
- **cuVSLAM** — visual SLAM/odometry running 4 stereo pairs at >30 FPS on AGX Orin. This is pose estimation, not image stitching, but the multi-camera extrinsics output is directly useful for calibrating the stitching pipeline's camera poses.
- **Isaac ROS Stereo Image Proc** — stereo disparity on GPU, feeds depth to downstream nodes.

**No Isaac ROS gem for 360° panoramic image stitching exists.** This capability gap is confirmed by the Isaac ROS package index.

**Sources:** [Isaac ROS Image Pipeline docs](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_pipeline/index.html) · [Isaac ROS DNN Stereo Depth (ESS)](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_stereo_depth/index.html) · [cuVSLAM multi-camera paper](https://arxiv.org/html/2506.04359v2) · [NVIDIA ROS 2 projects list](https://docs.ros.org/en/humble/Related-Projects/Nvidia-ROS2-Projects.html)

---

## 3. Automotive Standard: Ground-Plane IPM / BEV Surround View

### How Production Surround-View Systems Work

Production automotive 360° systems (parking assist, AVM — Around-View Monitoring) use **Inverse Perspective Mapping (IPM)** rather than eye-level cylindrical/spherical panoramas. The pipeline is:

1. **Fisheye undistortion** — wide-angle cameras (typically 180–190° FOV fisheye) are undistorted.
2. **IPM warp** — each undistorted frame is projected onto a flat ground plane (homography derived from camera intrinsic + extrinsic parameters and assumed camera height). This produces a top-down view per camera.
3. **Mosaic + blending** — the four top-down tiles are placed into a common canvas (a virtual bird's-eye view centered on the vehicle), and overlap regions are blended (typically alpha-fade or exposure-matched feather blend).

The math is a planar homography applied per camera — a single 3×3 matrix multiplication per pixel, extremely GPU-friendly. This is why `cv2.cuda.warpPerspective` or VPI Perspective Warp directly implements the core step.

### Known Limitation: Above-Ground Object Distortion

IPM assumes all pixels lie on a flat ground plane. Any object protruding above the ground (curb, pedestrian leg, vehicle body, bollard) is severely distorted: it "fans out" radially from the camera position, creating the characteristic "stretched shadow" artifact. Tall objects appear smeared. The severity is proportional to object height and camera distance.

**This distortion is generally accepted in automotive AVM** because the primary use case is parking clearance and ground-level obstacle proximity — and drivers are trained to interpret the stretched artifacts. For autonomous navigation or object detection, the distortion requires either depth-based correction or a segmentation-gated approach.

### BEV vs. Eye-Level Cylindrical Panorama — Use-Case Comparison

| Criterion | Ground-Plane BEV/IPM | Eye-Level Cylindrical Pano |
|---|---|---|
| Primary use case | Parking, close-range maneuvering | Navigation, obstacle detection at height |
| Parallax handling | Excellent at ground level; breaks for above-ground objects | Poor at any range without depth correction |
| Close-range (~1m) | Very good — ground-plane objects well-represented | Severe ghosting at seams |
| GPU complexity | Very low (single homography per camera) | Moderate to high (depth-dependent warp or seam strategy) |
| Camera requirement | Wide-angle fisheye preferred | Any camera; narrower FOV cameras like ZED X are suboptimal |
| ZED X suitability | Moderate (ZED X has ~110° horizontal FOV — less than ideal fisheye but workable) | Good for eye-level detail |

**Recommendation for parking / close-range scenarios:** BEV/IPM. For navigation at speed where eye-level 360° perception matters more: cylindrical pano with seam strategy.

### Open-Source Jetson Implementations

- **RidgeRun Bird's Eye View (commercial, $4,999)** — C++ library with CUDA + OpenGL acceleration, 30–60 FPS on Jetson Orin (Orin NX confirmed), 4–6 camera support, JSON calibration, GStreamer plugin. Not open source. [RidgeRun shop page](https://shop.ridgerun.com/products/birds-eye-view)
- **Cam2BEV (open source, Apache 2.0)** — TensorFlow implementation from RWTH Aachen for semantically segmented BEV using multiple vehicle cameras. Learning-based, not purely geometric IPM. Not real-time on embedded without TensorRT export. [GitHub: ika-rwth-aachen/Cam2BEV](https://github.com/ika-rwth-aachen/Cam2BEV)
- **OpenCV `warpPerspective` + custom CUDA** — the ground-plane homography can be computed offline from calibration targets and applied per-frame with `cv2.cuda.warpPerspective` (after building OpenCV with CUDA) or VPI Perspective Warp. No complete open-source ROS2 node for this specific 4-camera ZED X setup was found, but the primitives exist. [OpenCV cuda warping docs](https://docs.opencv.org/4.x/db/d29/group__cudawarping.html)
- **OpenDriveLab BEV Perception (research)** — paper list, not an implementation for real-time Jetson deployment. [GitHub: OpenDriveLab/Birds-eye-view-Perception](https://github.com/OpenDriveLab/Birds-eye-view-Perception/blob/master/docs/paper_list/bev_camera.md)

**Sources:** [RidgeRun BEV overview](https://www.ridgerun.com/post/birds-eye-view-on-nvidia-jetson-boards) · [IPM technique overview](https://www.emergentmind.com/topics/inverse-perspective-mapping-ipm) · [Cam2BEV paper](https://github.com/ika-rwth-aachen/Cam2BEV)

---

## 4. Lighter Real-Time Alternatives to Full Forward-Scatter Parallax Correction

Full per-pixel forward-scatter (the current approach) is the most geometrically correct but computationally heavy. Several lighter alternatives exist:

### A. Seam-Driven Stitching with Depth-Guided Seam Placement

**Concept:** Instead of correcting parallax everywhere, *route the seam through depth discontinuities where parallax is smallest or least visible* — specifically through the sky, ground, or low-texture/far-away regions. Parallax artifacts at close range are avoided not by correcting them but by ensuring the seam never passes through a near object.

**How it works:**
1. Compute a per-pixel "parallax cost" map in the overlap region using ZED X disparity: high parallax = high cost.
2. Run graph-cut (OpenCV `cv::detail::GraphCutSeamFinder` or `DpSeamFinder`) to find the minimum-cost cut through the overlap region, routing around near objects.
3. Apply multiband blending (OpenCV `cv::detail::MultiBandBlender`) across the seam.
4. Seam is recomputed when the scene changes significantly; otherwise reused per-frame.

**Performance:** Graph-cut seam finding is expensive to run per-frame (hundreds of ms). The key insight from multiple papers ([Real-Time Panoramic Surveillance, MDPI 2026](https://www.mdpi.com/1424-8220/26/1/186); [motion-aware graph cut, PMC 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12788332/)) is to run seam finding only on scene change detection or on a slow background thread, and reuse the seam for dozens of frames. This decouples seam quality from per-frame latency. The actual per-frame work is then just: warp images (GPU remap) + apply precomputed seam mask + multiband blend. This can run at 15–30+ FPS on GPU.

**Limitation:** Does NOT correct the parallax distortion around near objects — it hides the seam but the object still appears in both cameras inconsistently if it straddles the nominal blend region. Effective only when seams can be placed cleanly away from near objects. With ~90° camera spacing and ~70 cm baseline, the overlap regions are narrow, which makes seam placement easier.

**OpenCV CUDA support:** Seam finders in `cv::detail` are CPU-only (do not accept GpuMat). The warp and blend steps can be GPU-accelerated. Mixed approach: seam finding on CPU (amortized), warp+blend on GPU.

**Sources:** [OpenCV Stitcher detail API](https://docs.opencv.org/4.x/d8/d19/tutorial_stitcher.html) · [OpenCV multiband CUDA source](https://github.com/opencv/opencv/blob/master/modules/stitching/src/cuda/multiband_blend.cu) · [Depth-seam-guided stitching, MDPI 2022](https://www.mdpi.com/2079-9292/11/12/1876) · [Real-time panoramic surveillance, Sensors 2026](https://www.mdpi.com/1424-8220/26/1/186) · [SEAGULL seam-guided parallax-tolerant stitching](http://publish.illinois.edu/visual-modeling-and-analytics/files/2016/08/Seagull.pdf)

### B. Mesh/Grid Warp Driven by Sparse Depth

**Concept:** Rather than per-pixel reprojection, use ZED X depth to derive a coarse mesh warp (e.g., a 32×32 grid over each camera's overlap region) that deforms the image to reduce parallax. The mesh is updated at a slower rate than the frame rate (e.g., every 10 frames), and the warp is interpolated between updates.

**How it works:** For each grid node in the overlap region, project its world position (camera ray × depth) into both adjacent camera views, and compute the displacement. This gives a sparse set of (u,v) → (u',v') correspondences that drive the mesh warp. Apply with `cv2.cuda.remap` (using a precomputed dense warp map upsampled from the sparse grid via bicubic interpolation).

**Performance:** Mesh update is cheap (N×N depth lookups + solve). Per-frame: one `remap` per camera overlap = very fast on GPU. This approach is used in real-time 360 video systems for drone footage. No specific Jetson open-source implementation was found, but the primitives are all available.

**Limitation:** Only as accurate as the depth, and ZED X depth has holes near object edges and at max range. Mesh resolution must be balanced against update rate.

**Sources:** [Shape-optimizing mesh warp for stereo panorama, ScienceDirect 2019](https://www.sciencedirect.com/science/article/abs/pii/S0020025519309077) · [Depth-Supervised Fusion Network for Image Stitching, arXiv Oct 2025](https://arxiv.org/html/2510.21396v1) · [OpenCV cuda::remap docs](https://docs.opencv.org/4.x/db/d29/group__cudawarping.html)

### C. Per-Pixel Inverse Warp with Depth Proxy (Best of Both Worlds)

**Concept:** Instead of forward-scatter (push) from source to canvas, use inverse warp (pull) from canvas back to source using a precomputed inverse depth map. For each output pixel in the panorama, look up which source camera and pixel it came from (computed once from geometry), then adjust by the depth at that point to shift the sampling location. This is GPU-friendly because inverse warp is embarassingly parallel with no scatter collisions.

**How it works:**
1. Precompute a "base" warp map per camera: output pixel (u,v) → source pixel (x,y). Store as float32 xmap/ymap.
2. Per-frame: acquire ZED X depth map. For each output pixel, look up the corresponding source depth, compute the parallax-corrected offset (Δx, Δy) based on the baseline and depth, and add it to the base warp map entry.
3. Apply the corrected warp map using `cv2.cuda.remap` or VPI Remap.

This is equivalent to the correct reprojection math but implemented as inverse warp, avoiding all scatter issues. The depth-corrected warp map update step can be done in a custom Numba/CUDA kernel or with PyTorch ops on the GPU.

**Performance:** On GPU, the warp map update + remap should run well above 15 FPS. The depth-to-warp computation is ~4 arithmetic ops per pixel — highly parallelizable. This is the approach the team should target if full geometric correctness is required without the scatter bottleneck.

**Note on depth quality:** ZED X NEURAL depth (if optimized with one-time model optimization) is higher quality than the default standard mode, especially at close range. However, NEURAL mode requires TensorRT optimization (done once, persisted). Standard depth mode is real-time out-of-the-box and acceptable for the warp correction.

**Sources:** [VPI Remap algorithm docs](https://docs.nvidia.com/vpi/algo_remap.html) · [OpenCV cuda::remap](https://docs.opencv.org/4.x/db/d29/group__cudawarping.html) · [ZED SDK Jetson install](https://www.stereolabs.com/docs/development/zed-sdk/jetson)

### D. Deep Learning Parallax-Tolerant Stitching

Research papers (ICCV 2023: Parallax-Tolerant Unsupervised Deep Image Stitching; Depth-Supervised Fusion Network arXiv Oct 2025) achieve ~15 FPS on desktop GPU for 512×512 images. These are **not real-time on Jetson Orin at full resolution** without significant TensorRT optimization and resolution reduction. Not recommended for this use case without a dedicated prototyping effort.

**Sources:** [PTUDIE ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/) · [Depth-Supervised Fusion, arXiv 2025](https://arxiv.org/html/2510.21396v1)

---

## 5. ZED X SDK Capabilities and Constraints (Confirmed Facts)

These constraints are confirmed from Stereolabs documentation and community forums and are important for any approach:

- **ZED SDK Fusion API does NOT fuse RGB images or point clouds.** It fuses only: object detection results, body tracking/skeletons, positional tracking with GNSS, and spatial mapping (accumulated single-camera, not multi-camera merge). Point cloud fusion across cameras is *on the roadmap* but not released as of 2024–2026 community reports. [Source: Stereolabs Fusion docs](https://www.stereolabs.com/docs/fusion/overview) · [Forum confirmation: no point cloud fusion](https://community.stereolabs.com/t/multi-camera-point-cloud-fusion-streaming/11296)
- **ZED360 tool outputs a JSON calibration file only.** It does not generate panoramic images. [Source: Stereolabs ZED360 docs](https://www.stereolabs.com/docs/fusion/zed360)
- **ZED X supported resolutions:** HD1080 (1920×1080), HD1200 (1920×1200), SVGA (960×600). **HD720 is NOT supported** on ZED X (it is supported on ZED 2i/2). At SVGA with 4-camera GMSL2 Fakra, max frame rate is 60 FPS; at HD1080, max is 30 FPS per camera when 4 cameras share a connector. [Source: Stereolabs forum on resolutions](https://community.stereolabs.com/t/svga-resolution-not-working-on-zed-x-with-zed-depth-viewer-or-with-python-samples/6035)
- **ZED NEURAL depth** requires a one-time model optimization step with `ZEDOptimizationTool` (TensorRT engine generation). Once done, it is persistent. It provides higher quality depth especially at close range. [Source: ZED SDK docs]
- **Depth output is available per-camera** in real-time from the ZED SDK; there is no built-in merge of depth maps from multiple cameras.

---

## Final Ranked Recommendation Table

| Option | Core Approach | Effort (1=low, 5=high) | Payoff (15fps+ parallax-tolerant?) | Risk | Verdict |
|---|---|---|---|---|---|
| **1. GPU Inverse Warp via PyTorch + VPI Remap** | Depth-corrected per-pixel inverse warp (Option 4C); warp map update in PyTorch CUDA; apply with VPI Remap or cv2.cuda.remap | 2 | High — geometrically correct, no scatter contention, GPU-native | Medium (depends on depth quality; holes in depth → artifacts) | **START HERE** |
| **2. Forward Scatter in PyTorch CUDA (scatter_reduce_)** | Direct port of current NumPy approach to PyTorch CUDA tensors | 1 | Medium-High — same correctness as current, 10–50x faster | Low code risk; medium perf risk (iGPU scatter contention) | Good first step to unblock, then evolve to Option 1 |
| **3. Seam-Driven + Depth-Guided Seam + GPU Blend** | Graph-cut seam placement using depth cost; GPU multiband blend; seam reused across frames | 3 | Medium — hides parallax rather than correcting it; breaks if near objects at seam | High if near objects frequent at seam location | Good fallback for far-range accuracy; combine with Option 1 |
| **4. BEV / IPM Ground-Plane Stitching** | Per-camera homography warp onto ground plane; top-down composite | 2 | Medium — excellent for parking/close ground, poor for tall objects | Low (well-understood, fast) | Best fit if use case is parking/AVM; add as parallel output |
| **5. Custom CUDA z-buffer Kernel (Numba/tiled)** | Tiled forward scatter with atomicMin in shared memory | 4–5 | Highest geometric correctness; max performance ceiling | High (CUDA expertise needed; 2–4 weeks) | Reserve for later optimization if Options 1/2 plateau |
| **6. RidgeRun BEV Commercial Library** | Commercial BEV library; CUDA+OpenGL; 30–60 FPS on Jetson Orin | 1 (integration) | Medium — BEV/IPM only, ground-plane limitation applies | Low (commercial support); $4,999 cost | Fastest path to production BEV; not a cylindrical pano |
| **7. DriveWorks / Isaac ROS Stitcher** | Does not exist for Jetson | N/A | N/A | N/A | Not available |

### Single Top Recommendation

**Start with Option 2 (PyTorch CUDA scatter_reduce_) to immediately unblock the 1 FPS wall, then transition to Option 1 (depth-corrected GPU inverse warp) within the same sprint.**

The reasoning: PyTorch is already installed on the Jetson Orin under JetPack 6, requires minimal code change (`.cuda()` + `scatter_reduce_`), and will immediately validate that the GPU path is viable. While the forward scatter runs, the depth-corrected inverse warp (Option 1) should be prototyped in parallel — it avoids the atomicMin contention that is the fundamental weakness of forward scatter, trades scatter collision risk for depth-hole risk, and uses `cv2.cuda.remap`/VPI which are optimized pixel-fetch kernels. The combined effort is roughly one sprint for a single developer, using only tools available out-of-the-box in JetPack 6 + ZED SDK 4.

If the use case is parking/AVM rather than eye-level perception, additionally build the BEV/IPM path (Option 4) as it is simpler, faster, and matches the automotive industry standard — the cylindrical panorama is unnecessary for that use case.

---

## Source Index

| Topic | URL |
|---|---|
| DriveWorks restricted to DRIVE platform | https://forums.developer.nvidia.com/t/can-i-install-driveworks-sdk-on-jetson-agx-orin/268594 |
| ZED SDK Fusion API (no RGB/point cloud fusion) | https://www.stereolabs.com/docs/fusion/overview |
| ZED360 tool (calibration file only) | https://www.stereolabs.com/docs/fusion/zed360 |
| ZED X supported resolutions (no HD720) | https://community.stereolabs.com/t/svga-resolution-not-working-on-zed-x-with-zed-depth-viewer-or-with-python-samples/6035 |
| Multi-camera point cloud fusion (not supported) | https://community.stereolabs.com/t/multi-camera-point-cloud-fusion-streaming/11296 |
| PyTorch 2.5 CUDA on JetPack 6 | https://ninjalabo.ai/blogs/jetson_pytorch.html |
| NVIDIA PyTorch Jetson install docs | https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html |
| PyTorch scatter_reduce_ docs | https://docs.pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html |
| JetPack OpenCV lacks CUDA (must build from source) | https://dev.to/rbelshevitz/accelerating-opencv-with-cuda-on-jetson-orin-nx-a-complete-build-guide-525j |
| OpenCV cuda warping module (remap, warpPerspective) | https://docs.opencv.org/4.x/db/d29/group__cudawarping.html |
| OpenCV multiband blend CUDA source | https://github.com/opencv/opencv/blob/master/modules/stitching/src/cuda/multiband_blend.cu |
| VPI algorithm list | https://docs.nvidia.com/vpi/algorithms.html |
| VPI Remap algorithm | https://docs.nvidia.com/vpi/algo_remap.html |
| NVIDIA VPI developer page | https://developer.nvidia.com/embedded/vpi |
| VPI 3.2 in JetPack 6.1 | https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Orin/JetPack_6.1/Getting_Started/Components |
| CuPy 2.6x slower than NumPy on AGX Orin | https://github.com/cupy/cupy/issues/8151 |
| CuPy Jetson Orin Nano issues | https://github.com/cupy/cupy/issues/9216 |
| Numba CUDA on Jetson (JetsonHacks 2024) | https://jetsonhacks.com/2024/01/15/cuda-programming-in-python-with-numba/ |
| CUDA z-buffer efficiency forum | https://forums.developer.nvidia.com/t/efficient-z-buffer-in-cuda-/416079 |
| Isaac ROS Image Pipeline (no stitching) | https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_pipeline/index.html |
| Isaac ROS ESS stereo depth (108 FPS on Orin) | https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_stereo_depth/index.html |
| cuVSLAM multi-camera paper | https://arxiv.org/html/2506.04359v2 |
| RidgeRun BEV overview | https://www.ridgerun.com/post/birds-eye-view-on-nvidia-jetson-boards |
| RidgeRun BEV shop ($4,999) | https://shop.ridgerun.com/products/birds-eye-view |
| Cam2BEV open source | https://github.com/ika-rwth-aachen/Cam2BEV |
| IPM technique overview | https://www.emergentmind.com/topics/inverse-perspective-mapping-ipm |
| OpenCV Stitcher detail API | https://docs.opencv.org/4.x/d8/d19/tutorial_stitcher.html |
| Depth-seam-guided stitching MDPI 2022 | https://www.mdpi.com/2079-9292/11/12/1876 |
| Real-time panoramic surveillance Sensors 2026 | https://www.mdpi.com/1424-8220/26/1/186 |
| SEAGULL seam-guided parallax-tolerant stitching | http://publish.illinois.edu/visual-modeling-and-analytics/files/2016/08/Seagull.pdf |
| Depth-Supervised Fusion Network arXiv 2025 | https://arxiv.org/html/2510.21396v1 |
| ZED X and Jetson Orin review (Intermodalics) | https://www.intermodalics.ai/blog/zed-x-and-nvidia-applications-architecture-and-integrations |
| DRIVE AGX Orin developer blog | https://developer.nvidia.com/blog/now-available-drive-agx-orin-with-drive-os-6/ |
