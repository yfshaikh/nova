# Depth Reprojection for the ZED 360° Panorama — Design

**Date:** 2026-06-03
**Status:** Approved design, pre-implementation
**Target file:** `fusion.py` on the rig (`~/Desktop/Navigator_Orin/`); local mirror `zed_360_panorama.py`

---

## Problem

The current pipeline stitches 4 outward-facing ZED cameras into a 360° cylindrical
panorama using a **rotation-only** projection (each pixel treated as a direction,
laid on a cylinder as if the scene were at infinity). This is correct for distant
scenery but fails on **close objects**: because the four cameras sit ~70 cm apart,
a nearby object (the UT Drive car parked beside the rig) projects to different
azimuths in adjacent cameras. At the right↔back seam this produces a melted,
ghosted blob, and at the feathered handoff some content disappears entirely.

Root cause, confirmed by diagnostics:
- The car edge **is** present in the right camera's projection (coverage overlay bright).
- It **vanishes in the blend** because the distance-to-edge feather drives the right
  camera's weight to ~0 at its edge, the right∩back overlap hands the pixel to the
  back camera (more interior → higher weight), and the back camera — due to parallax —
  is looking at the background behind the car at that azimuth.
- No rotation-only weighting scheme fixes this: the two cameras genuinely disagree
  about what is at that direction.

## Goal

Keep the **same cylindrical 360° panorama** output. Use ZED depth to re-render the
near-field from a single common origin so the cameras agree, eliminating the melt
and the vanishing edge. No new output format.

## Constraints / decisions

- **Output:** same cylindrical panorama; depth used purely to correct parallax.
- **Extrinsics:** hand-measured (±1–2", yaw-only). This sets the accuracy ceiling.
  Design includes an interactive nudge to fine-tune visually.
- **Performance:** correctness first. Pure NumPy + ZED depth, SVGA, 1–5 fps acceptable.
  No GPU work in this phase.

---

## Core idea

**Depth reprojection = re-rendering all four RGB-D cameras from one virtual viewpoint
at the rig center.** Parallax exists because the cameras occupy different positions;
re-rendering every pixel from one common origin using its true 3D position removes the
disagreement.

## Architecture: base + near-overlay (split by depth)

Each camera frame is split by depth and routed through two paths:

```
Each camera frame ──┬── FAR pixels (depth invalid OR range > NEAR_MAX)
                    │      → existing rotation-only cylinder remap → BASE layer
                    │        (far parallax negligible; ZED long-range depth never used)
                    │
                    └── NEAR pixels (valid depth ≤ NEAR_MAX)
                           → depth point-cloud scatter onto cylinder → OVERLAY layer
                             (where parallax lives; ZED depth is reliable here)

Final pano = BASE with OVERLAY composited on top (global z-buffer, feathered edges)
```

Rationale:
- Far background keeps using the already-trusted rotation-only code.
- Only near points get the expensive depth treatment (fewer points, no long-range
  depth dependence, and the car lands in the NEAR bucket).
- **No double-draw:** near pixels are removed from the base path and drawn *only* via
  the overlay, so the melted smear is never generated. The base is built from far
  pixels only.
- Incremental and testable: the base layer is the current pipeline; the overlay is
  added beside it.

`NEAR_MAX` (depth cutoff, e.g. 8 m) is the single knob defining the split.

---

## Reprojection math

ZED's default point cloud is already in **X-right, Y-down, Z-forward** per camera —
the same convention `cam_to_world` uses — so no axis remapping is needed.

**Rig frame:** X-right, Y-down, Z-forward; origin = the physical point the tape
measurements were taken from (MUST be confirmed and consistent across all four `t`).

**Per-camera extrinsics** (add `t` translation to `CAMERAS`, meters = inches × 0.0254):

| cam   | rotation | translation `t` (m)                        |
|-------|----------|--------------------------------------------|
| front | yaw 0°   | `(0, 0, +0.711)`  (28" fwd)                 |
| right | yaw +90° | `(+0.660, 0, −0.216)` (26" right, 8.5" back)|
| left  | yaw −90° | `(−0.660, 0, −0.216)` (26" left, 8.5" back) **← X-sign corrected** |
| back  | yaw 180° | `(0, 0, −1.422)` (56" back)                 |

Notes:
- The left/right X-sign was swapped in the original JSON; irrelevant for rotation-only
  but critical here.
- Vertical `Y` offsets assumed ≈ 0 (cameras roughly level); add measured values if available.

**Transform per valid NEAR pixel:**
```
P_cam = ZED XYZ at (u,v)               # meters, camera frame
P_rig = R_cam2rig @ P_cam + t_cam      # common rig frame (translation included)
az    = atan2(P_rig.X, P_rig.Z)
h     = P_rig.Y / hypot(P_rig.X, P_rig.Z)
rng   = ‖P_rig‖                        # z-buffer key
gx    = (az * scale + pano_w/2) mod pano_w
gy    =  h  * scale + pano_h/2
```
Identical cylinder parametrization to today (`ray = [sin az, h, cos az]`), but driven
by real 3D points from one origin. Nearest `rng` wins the pixel.

**Accuracy caveats baked in:**
- ±1–2", yaw-only extrinsics set the floor. Provide an **interactive nudge** (config
  deltas / arrow keys for per-camera `t` and yaw) to visually fine-tune on the car;
  pitch/roll can be added later if a seam needs it.
- Tape-measure origin must be the same point for all four cameras.

---

## Compositing

Two pano-sized overlay buffers: `overlay_color` and `overlay_zbuf` (range, init +∞).
All four cameras scatter NEAR points into the **same shared overlay with a global
z-buffer** — across cameras, the nearest consistent 3D surface wins each pixel. This
global z-buffer is what removes the melt (no averaging of disagreeing views).

```
final = base                                       # far background (current pano)
final = where(overlay has a point, blend(overlay, base), base)
```
Overlay edges get a short feather (a few px) so the near object does not hard-cut into
the background.

## Hole-fill (forward scatter leaves pinholes) — escalating, apply only as needed

1. **Splat** each point as a small disk (radius ~1–2 px), z-buffer still respected.
2. **Morphological close** on the overlay mask to seal hairline cracks.
3. Residual holes inside the near-object silhouette → `cv2.inpaint` or median fill.

## Depth settings

- `depth_mode`: `NONE` → `NEURAL` (quality; correctness-first). Fallback `PERFORMANCE`
  if too slow on the Orin.
- Retrieve `MEASURE.XYZ` (geometry) + `VIEW.LEFT` (color): same resolution,
  pixel-aligned, no registration needed.
- Mask out NaN/Inf and points beyond `NEAR_MAX`.
- Resolution: SVGA (valid for ZED X; less per-frame work).

---

## Diagnostics (additions to existing ISOLATE / TINT / COVERAGE / per-cam windows)

- `DEPTH_REPROJECT` — master switch. Off = exactly today's pano; On = depth-corrected.
  One-flag A/B to prove the fix.
- `SHOW_DEPTH_VALID` — per camera, valid-depth vs holes (e.g. car glass will be holes).
- `SHOW_NEAR_MASK` — which pano pixels came from the overlay vs the base.

## Performance posture

Pure NumPy scatter via `np.minimum.at` / `np.add.at`, SVGA, expect 1–5 fps. GPU
acceleration explicitly deferred to a later phase.

---

## Reused vs new

**Reused from current `fusion.py`:** threaded capture, `cam_to_world`, cylinder
parametrization + `scale`/`pano_w`/`pano_h`, `build_cylindrical_maps` (base layer),
distanceTransform feather, all existing diagnostic flags, startup banner.

**New:**
- `t` translations in `CAMERAS` (+ sign fix) and an extrinsics nudge.
- Depth enabled; per-thread `MEASURE.XYZ` retrieval alongside `VIEW.LEFT`.
- Depth split (NEAR/FAR) per frame.
- Point-cloud → rig-frame transform → cylinder scatter with global z-buffer + splat.
- Base-from-far-pixels-only path (mask near pixels out of the base).
- Overlay→base compositing with feathered edges and hole-fill.
- New diagnostics: `DEPTH_REPROJECT`, `SHOW_DEPTH_VALID`, `SHOW_NEAR_MASK`.

## Open items to confirm at implementation time

1. The physical reference point the tape measurements share (origin for all `t`).
2. Any measured vertical (`Y`) offsets between cameras (assumed 0 for now).
3. `NEAR_MAX` starting value (8 m proposed) — tune to scene.

## Out of scope (this phase)

- GPU/CUDA acceleration; real-time (15+ fps).
- New output formats (BEV, point-cloud ROS2 topic).
- Full 6-DoF calibration (only manual nudge on top of hand-measured extrinsics).
- Photometric exposure/WB matching beyond what already exists.
