# 360° Stitching — Methods Tried (Post-Mortem)

A chronological record of every approach attempted for the Nova 4-camera ZED X 360°
panorama, and why each one failed or was rejected. Hardware: 4 ZED X (GMSL) cameras
facing outward ~90° apart on a car, Jetson Orin, ROS2 Humble. See also
`research-orin-360-stitch.md` (depth-reproj GPU path) and
`research-... calibration` findings for the LiDAR-hub calibration route.

---

| # | Method | What it was | Why it failed / verdict |
|---|--------|-------------|------------------------|
| 1 | **`cv2.Stitcher` (PANORAMA)** | OpenCV feature-matching panorama | Assumes all cameras share one nodal point and need feature matches in the overlaps. The ~70 cm baselines + dead zones produce parallax it can't model → stitch failures. Rejected early. |
| 2 | **ZED Fusion (`sl.Fusion`)** | ZED SDK multi-camera fusion | Wrong tool: fuses **metadata only** (object detection, body/skeleton tracking, positional+GNSS, spatial-map point cloud). `retrieveImage()` returns *one* camera's LEFT view — never a stitched RGB image. Confirmed from Stereolabs docs. |
| 3 | **ZED360 wizard** | Stereolabs calibration tool | Needs overlapping views to solve poses and is "primarily body-tracking fusion." The wizard **cannot converge on the dead-zone rig** (confirmed on hardware). Even if it ran, it outputs *extrinsics*, not an RGB stitch. |
| 4 | **Rotation-only cylindrical projection** (hand-measured extrinsics) — the **baseline** | Static per-pixel cylinder remap + distance-transform feather blend | **Partial success:** fast, full 360° coverage, far background stitches cleanly. **Failed on close objects:** rotation-only assumes infinity, so the nearby car **melts/ghosts at seams**. ±1–2" hand-measured extrinsics also cap accuracy. Bugs fixed along the way: upside-down image (Y-up vs Y-down), HD720 invalid on ZED X, seam-cut at canvas edge, and the "lost FOV" (really cylinder edge-compression + the parallax seam handoff). |
| 5 | **Depth reprojection, pure NumPy** | Unproject every pixel → transform to common rig origin → z-buffer scatter onto the cylinder; far/invalid falls back to the base | **Best-looking blend — the winner.** Failed only on *execution*: ~1 fps / "not responding" because ~2.3 M points/frame ran in NumPy on the **CPU** while the GPU sat idle; **black holes** because `NEAR_MAX=8 m` emptied the base indoors + sparse forward-scatter; NEURAL depth added a multi-minute first-run model-optimization hang. **Correct approach, wrong hardware path.** |
| 6 | **Depth-aware seam selection** (`fusion_seam.py`) | Pick the nearest camera in overlaps instead of feather-averaging | Didn't fix the melt (confirmed on hardware). Selection alone is too coarse — it doesn't align the two disagreeing views. |
| 7 | **BEV / ground-plane surround** (`fusion_bev.py`) | Top-down inverse-perspective-mapping homography composite | Looked bad: raised objects (car body, walls) smear radially — inherent to single-homography BEV — and it needs accurate `CAMERA_HEIGHT` + level cameras we don't have. |

---

## Also ruled out (from verified research)

- **No turnkey RGB stitcher exists** in the ZED ecosystem. Fusion's spatial mapping can
  build a fused *colored point cloud / mesh* across cameras, but that's a sparse offline
  3D reconstruction, not a live photographic panorama. With proper camera overlap a
  generic stitcher (OpenCV) would work; the **dead zones** rule those out and force
  custom stitching.

## Conclusion / chosen direction

**Method #5 (depth reprojection) is the one that actually worked visually.** The plan is
to keep it and fix its two execution problems:

1. **Lag → GPU.** Port the per-frame transform + z-buffer scatter to PyTorch CUDA
   (`scatter_reduce_('amin')` + `argmin`/`gather` for the winning point's color) on the
   Orin. NVIDIA ships prebuilt Jetson PyTorch wheels; effort ≈ an afternoon.
2. **Holes → base-layer composite.** Composite the sparse depth overlay **over** the
   dense rotation-only panorama (DIBR layered-depth pattern) so any missed pixel shows
   real RGB background instead of black. Plus small-disk splatting + an optional
   push-pull fill pass.

**Orthogonal improvement:** replace the ±1–2" hand-measured extrinsics with calibrated
ones via the **LiDAR-as-hub** approach (calibrate each camera→LiDAR, compose through the
LiDAR), using ROS2 tools (Autoware/TIER IV CalibrationTools, `ros2_calib`, Koide
`direct_visual_lidar_calibration`) that also emit a URDF — better extrinsics make the
depth-reproj seams align well.
