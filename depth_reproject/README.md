# depth_reproject — 4× ZED 360° panorama with depth-corrected near field

## The goal

Stitch 4 outward-facing ZED X cameras (front / right / back / left, ~90° apart on
the car roof) into one continuous **360° panorama**, in real time on the Jetson Orin.

The hard part is **parallax**: the cameras sit ~70 cm apart, so a *close* object (a
car parked next to the rig) is seen in a different direction by each camera. A cheap
"rotation-only" stitch — which assumes all cameras share one optical center — works
for far scenery but makes close objects **smear / melt** at the seams.

The fix used here is a **two-layer** panorama:

- **BASE layer — rotation-only cylinder.** Fast, full 360°, correct for the far
  background (where parallax is negligible). A static lookup table built once.
- **OVERLAY layer — depth reprojection.** Only the *near* field (≤ `NEAR_MAX` m).
  Uses ZED depth to turn each pixel into a 3D point, re-renders everything from one
  common rig-center viewpoint (so the cameras agree), and lays the result on top of
  the base. This is what removes the melt.

The base guarantees no black holes (any pixel the overlay misses shows real
background); the overlay fixes close-range parallax. Far = base, near = overlay.

## The files

| File | Purpose |
|------|---------|
| **`reproject.py`** | The depth-reprojection **math**, pure NumPy (no cameras, no OpenCV — fully unit-testable). Five functions: `cam_points_to_rig` (move a camera's 3D points into the shared rig frame), `rig_to_cylinder` (project a 3D point to its panorama pixel + distance), `split_near_far` (keep the near points), `scatter_zbuffer` (paint points, keep the nearest per pixel — the z-buffer, via `np.minimum.at`), `composite` (lay the overlay on the base). |
| **`reproject_gpu.py`** | The **GPU port** of `reproject.py` — same math on PyTorch/CUDA tensors so it runs on the Orin GPU instead of the CPU. Mirrors the same five functions; the z-buffer becomes a packed-key `scatter_reduce_('amin')` that keeps the nearest point *and* recovers its color in one pass. Includes `build_overlay` (pool all cameras' near points → one global z-buffer) and a `__main__` self-test. Run `python3 reproject_gpu.py` first to confirm your torch build. |
| **`zed_360_panorama.py`** | The **CPU pipeline** — the full program. Opens the 4 ZEDs (threaded capture of color + XYZ point cloud), builds the rotation-only base table, and per frame blends the base and lays the depth overlay (calling `reproject.py`). Correct but ~1 fps on the Orin (NumPy on the ARM CPU), and its base-emptying step is what caused the indoor black holes. Kept as the readable reference / A-B baseline. |
| **`zed_360_panorama_gpu.py`** | The **GPU pipeline** — the cylinder panorama with the depth reprojection moved to the GPU and the base kept **full** (holes fix). Correct output, but still slow on the rig (see findings below). Has `TIMING`/`SCALE`/`POINT_STRIDE` knobs. |
| **`fused_pointcloud_viewer.py`** | Alternative renderer — **no cylinder, no stitching**. Fuses the 4 colored point clouds in 3D and renders with **Open3D**. Holes and all (depth-viewer style). Slow (~1 fps) due to Open3D's per-frame rebuild + GPU→CPU→GPU copies. |
| **`fused_pc_fast.py`** | Leaner version of the above using **pyqtgraph** (one VBO upload/frame, zero CUDA compute). Still ~1.4 fps — which is what proved the bottleneck is *capture*, not rendering (see findings). |

## How they relate

```
            math (per-pixel/point)         pipeline (cameras + loop)
  CPU   ->  reproject.py            <----  zed_360_panorama.py
  GPU   ->  reproject_gpu.py        <----  zed_360_panorama_gpu.py   <-- run this
```
The two pipelines share the same rig config, the same cylinder geometry, and the same
base-layer idea; they differ only in *where* the reprojection runs (CPU vs GPU) and
whether the base is emptied (CPU, buggy) or kept full (GPU, fixed).

## Running (on the rig)

```bash
# 1) confirm the GPU math works on your torch build
python3 reproject_gpu.py          # expect: "all GPU-math self-tests passed"

# 2) run the GPU pipeline
python3 zed_360_panorama_gpu.py   # press q to quit
```

Coordinate frame throughout: **X-right, Y-down, Z-forward** (matches the ZED point
cloud). Extrinsics (`t` per camera, in meters) are hand-measured ±1–2"; replacing them
with a calibrated set (e.g. LiDAR-hub calibration) is the main lever for tighter seams.

## Performance findings (as of Jun 2026)

Correctness is solved (clean 360 stitch, parallax fixed, holes fixed). **Framerate
is the open problem**, and a long investigation on the rig (Jetson AGX Orin,
JetPack 6.0 / CUDA 12.2, 4× ZED X, SVGA) narrowed it down:

What we ruled OUT as the bottleneck:
- **Algorithm** — int64 vs float `scatter_reduce('amin')`: no change.
- **Resolution / point count** — `SCALE` down, `POINT_STRIDE` up: no change.
- **Power throttling** — `nvpmodel -m 0` (MAXN) + `jetson_clocks`: no change.
- **The renderer** — cylinder+torch, Open3D, and pyqtgraph ALL land at ~1–1.4 fps.
  Three different renderers, same number → rendering is not the bottleneck.

What it actually is:
- **The capture/depth side.** 4 cameras each computing PERFORMANCE depth + the
  per-frame `get_data()` CPU copies, serialized through Python's GIL, caps the
  whole process at ~1–2 fps. The GPU is also largely occupied by ZED's 4-cam depth,
  which is why the single-camera ZED Depth Viewer is smooth and 4-cam fusion is not.
- Confirm with the 2-camera test: comment out 2 cameras → fps roughly doubles.

Gotchas hit along the way (don't relearn these):
- Generic `pip install torch` gives a CPU/wrong-CUDA wheel → `cuda.is_available()`
  is False → "GPU" path silently runs on CPU. Use the **Jetson wheel for your
  JetPack/CUDA** (JetPack 6.0 = CUDA 12.2).
- ZED X has **no HD720**; `NEURAL` depth needs a one-time model build (looks frozen).

## Levers that would actually help framerate (all capture-side)

- **Fewer cameras** (2 instead of 4), or **lower depth resolution / rate**.
- **GPU-resident capture** (`retrieve_measure(..., sl.MEM.GPU)`) to kill the big
  `get_data()` CPU copies — only pays off paired with **CUDA-OpenGL interop**
  rendering (Stereolabs' `ogl_viewer`, extended to 4 clouds with a per-cloud model
  matrix). This is the path to true depth-viewer speed.
- **Better extrinsics** (LiDAR-hub calibration) — fixes seam doubling, not fps.

## Honest status

The depth-corrected 360 **works and looks right**, but is **~1–2 fps on this Orin in
Python** — fine as an offline/research artifact, not yet real-time. Real-time needs
either fewer/lighter depth streams or the GPU-resident interop renderer above.
