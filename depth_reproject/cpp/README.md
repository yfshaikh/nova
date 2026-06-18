# NovaFusedCloud — GPU-resident 4× ZED viewer (C++)

Fused rig-center point cloud viewer using the same path as **ZED Depth Viewer**:
`retrieveMeasure(XYZRGBA, MEM::GPU)` → CUDA–OpenGL interop (`cudaMemcpyDeviceToDevice`).

No Python, no `get_data()` CPU copies, no Open3D/pyqtgraph upload loop.

## Prerequisites (Jetson rig)

- ZED SDK (same version as your ZED X driver)
- CUDA (ships with JetPack)
- GLEW + GLUT + OpenGL dev packages:

```bash
sudo apt-get install -y libglew-dev freeglut3-dev libgl1-mesa-dev
```

## Build

```bash
cd depth_reproject/cpp
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

If `find_package(ZED)` fails, ensure `/usr/local/zed` exists and you have sourced nothing special — the ZED SDK installer registers CMake paths.

## Run

```bash
./NovaFusedCloud
```

All four cameras must be connected (serials match `src/main.cpp` `RIG[]` table).

**Controls:** left-drag orbit · right-drag pan · wheel zoom · `q` / ESC quit

Window title shows **loop fps** (viewer refresh rate). Terminal also prints **min cam fps** from ZED SDK per-camera stats.

## vs Python `fused_pc_fast.py`

| | Python | This binary |
|---|--------|-------------|
| Point cloud memory | CPU `get_data()` copy | GPU `MEM::GPU` |
| Upload to GL | pyqtgraph VBO each frame | CUDA D2D into registered GL buffer |
| Capture threads | yes | yes |
| Rig transform | CPU numpy | GL model matrix (cam frame untouched on GPU) |

This does **not** render the cylindrical 360° panorama — it is the fused 3D cloud step that must be fast before any panorama compositor can run at depth-viewer speeds.

## Next steps (not implemented here)

- Cylinder panorama: GPU shader projecting the same GPU-resident clouds, or OpenGL FBO + depth reprojection kernel
- Optional `NEURAL_LIGHT` depth mode for lower GPU load with 4 cams
- LiDAR-hub calibrated extrinsics JSON loader instead of hard-coded `RIG[]`
