# NovaPanoramaGPU — end-to-end GPU 360 panorama (C++/CUDA)

C++/CUDA port of `zed_360_panorama_gpu.py` + `reproject_gpu.py`. Same two-layer
result (rotation-only base + depth-corrected near overlay), but **every per-frame
stage runs on the GPU** and the final image is displayed through CUDA–OpenGL
interop, so nothing is copied to the CPU in the hot loop.

## What moved to the GPU (vs the Python version)

| Stage | Python (per frame) | Here |
|-------|--------------------|------|
| Base remap + blend | `cv2.remap` ×4 on CPU | `kAccumBase` CUDA kernel (bilinear + feather) |
| Cloud → rig → cylinder | torch on GPU | `kScatterOverlay` CUDA kernel |
| Z-buffer + color | torch `scatter_reduce_` + masked write | one `atomicMin` on packed `uint64` (`depth<<32 \| BGRA`) |
| Pinhole sealing | `cv2.morphologyEx` + `dilate` on CPU | in-kernel splat radius |
| Display | `pano.cpu().numpy()` + `cv2.imshow` | CUDA-GL PBO → texture → quad (zero copy) |
| Color source | XYZ + separate image subsample | single `XYZRGBA` retrieve |

## Prerequisites (Jetson rig)

```bash
sudo apt-get install -y libglew-dev freeglut3-dev libgl1-mesa-dev
```

ZED SDK + CUDA (JetPack) already provide the rest.

## Build

```bash
cd depth_reproject/cpp_panorama
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

If your GPU is not sm_87 (AGX Orin), pass e.g. `cmake .. -DCMAKE_CUDA_ARCHITECTURES=72`.

## Run

```bash
./NovaPanoramaGPU      # q / ESC to quit
```

Window title and terminal show **loop fps**, **per-frame ms**, and **min cam fps**.

## Tuning knobs (top of `src/main.cpp`)

| Constant | Meaning |
|----------|---------|
| `kScale` | pano pixels/radian (lower = smaller pano = faster) |
| `kNearMax` | meters; closer than this gets depth reprojection |
| `kPointStride` | subsample the cloud (2 = 4× fewer points) |
| `kSplat` | overlay splat radius; 1 = 3×3, seals pinholes |
| `kResolution` / `kFps` | ZED capture mode |

## Notes / parity

- Rig frame is **X-right, Y-down, Z-forward** (`COORDINATE_SYSTEM::IMAGE`), same
  as the Python rig. Extrinsics (`RIG[]`) match the Python `CAMERAS` table.
- Feather weights use an approximate (chamfer) distance transform computed once
  on the host — replaces `cv2.distanceTransform`. Close enough for seam feather.
- The z-buffer tie-break is "nearest depth wins"; ties resolve by min BGRA. If
  you ever see speckle at equal depths, lower `kPointStride` or raise `kSplat`.
- Capture threads pause until the GL thread consumes each frame (no GPU-mat race);
  the panorama refreshes at camera rate.

## Relationship to the other C++ folder

- `../cpp/` (`NovaFusedCloud`) renders the **fused 3D point cloud** (Depth-Viewer
  style) — good for verifying capture throughput.
- This folder renders the **stitched cylindrical panorama** — the actual product.
