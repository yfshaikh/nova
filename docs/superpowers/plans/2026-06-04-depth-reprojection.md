# Depth Reprojection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct close-range parallax in the ZED 360° cylindrical panorama by re-rendering the near field from a single rig-center viewpoint using ZED depth, leaving the far field on the existing rotation-only pipeline.

**Architecture:** Split each camera frame by depth. FAR pixels (invalid depth or range > `NEAR_MAX`) stay on the existing rotation-only cylinder remap → BASE layer. NEAR pixels become 3D points, are transformed into one common rig frame using full rotation+translation extrinsics, and are forward-scattered onto the cylinder with a global z-buffer → OVERLAY layer composited on top of BASE.

**Tech Stack:** Python 3.9, NumPy (pure-math module, unit-tested on the Mac), OpenCV + pyzed (capture/depth/hole-fill, verified visually on the Jetson Orin rig).

---

## Conventions for this plan

- **Two machines.** Pure-math module (`reproject.py`) + tests are written and run **locally on the Mac** (`~/Desktop/Projects/nova`). Integration into `fusion.py` happens on the **rig** (`~/Desktop/Navigator_Orin`), which has the ZED SDK and cameras. After Tasks 1–6 pass locally, sync `reproject.py` to the rig (Task 7) and wire it in (Task 8). Verify on the rig (Task 9).
- **Commits are user-initiated.** Per the project's standing rule, do **not** auto-commit. This directory isn't a git repo yet; if you want history, run `git init` first. Each task ends with a **Checkpoint** (run tests / visual verify, then pause for the user to review and commit). Suggested commit messages are provided.
- **Coordinate frame.** Rig frame = X-right, Y-down, Z-forward, origin = the physical reference point the tape measurements share. ZED's default point cloud is already in this convention per camera.

---

## File Structure

- **Create `reproject.py`** — pure NumPy reprojection math. No cv2, no pyzed, no I/O. Functions: `cam_points_to_rig`, `rig_to_cylinder`, `split_near_far`, `scatter_zbuffer`, `composite`. One responsibility: turn camera-frame 3D points + color into an overlay layer on the cylinder.
- **Create `test_reproject.py`** — pytest unit tests for every function in `reproject.py`, synthetic data only, runs on the Mac with no hardware.
- **Modify `fusion.py`** (rig; local mirror `zed_360_panorama.py`) — add translation extrinsics, enable depth, retrieve `MEASURE.XYZ`, wire in `reproject.py`, add `DEPTH_REPROJECT` / `NEAR_MAX` / `SHOW_DEPTH_VALID` / `SHOW_NEAR_MASK` flags and cv2 splat/hole-fill.

---

## Task 0: Local test environment

**Files:** none

- [ ] **Step 1: Install pytest locally**

Run: `python3 -m pip install --user pytest`
Expected: `Successfully installed pytest-...`

- [ ] **Step 2: Verify**

Run: `python3 -m pytest --version`
Expected: prints a `pytest 8.x` version line.

---

## Task 1: `cam_points_to_rig` — camera-frame points into the rig frame

**Files:**
- Create: `reproject.py`
- Test: `test_reproject.py`

- [ ] **Step 1: Write the failing test**

```python
# test_reproject.py
import numpy as np
import reproject as rp


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def test_cam_points_to_rig_translation_only():
    P_cam = np.array([[0.0, 0.0, 1.0]])          # 1 m straight ahead
    R = np.eye(3)
    t = np.array([1.0, 0.0, 0.0])                # rig sits 1 m to the +X
    out = rp.cam_points_to_rig(P_cam, R, t)
    np.testing.assert_allclose(out, [[1.0, 0.0, 1.0]], atol=1e-9)


def test_cam_points_to_rig_right_camera_yaw90():
    # Right camera (yaw +90): a point straight ahead in the camera maps to +X in rig.
    P_cam = np.array([[0.0, 0.0, 2.0]])
    R = _ry(np.radians(90))
    t = np.array([0.66, 0.0, -0.216])
    out = rp.cam_points_to_rig(P_cam, R, t)
    np.testing.assert_allclose(out, [[0.66 + 2.0, 0.0, -0.216]], atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest test_reproject.py -k cam_points_to_rig -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reproject'` (or `AttributeError`).

- [ ] **Step 3: Write minimal implementation**

```python
# reproject.py
"""Pure-NumPy reprojection math for depth-corrected 360 panorama.

No cv2 / pyzed / I/O — everything here is unit-testable on any machine.
Rig frame: X-right, Y-down, Z-forward; origin = shared tape-measure point.
"""
import numpy as np


def cam_points_to_rig(P_cam, R_cam2rig, t_cam):
    """Transform camera-frame points into the common rig frame.

    P_cam: (N, 3) points in camera frame (meters, X-right Y-down Z-forward).
    R_cam2rig: (3, 3) rotation. t_cam: (3,) translation (meters).
    Returns (N, 3) points in the rig frame.
    """
    P_cam = np.asarray(P_cam, dtype=np.float64)
    return P_cam @ np.asarray(R_cam2rig).T + np.asarray(t_cam)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest test_reproject.py -k cam_points_to_rig -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Checkpoint**

Run full file: `python3 -m pytest test_reproject.py -v` (expected: 2 passed). Pause for review.
Suggested commit: `feat(reproject): cam_points_to_rig point transform`

---

## Task 2: `rig_to_cylinder` — rig points to panorama pixels + range

**Files:**
- Modify: `reproject.py`
- Test: `test_reproject.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_reproject.py
def test_rig_to_cylinder_forward_center():
    # Point straight ahead lands at canvas center; h=0; range = Z.
    P = np.array([[0.0, 0.0, 5.0]])
    gx, gy, rng, valid = rp.rig_to_cylinder(P, scale=350.0, pano_w=2200, pano_h=512)
    assert valid[0]
    assert gx[0] == 1100               # pano_w/2
    assert gy[0] == 256                # pano_h/2
    np.testing.assert_allclose(rng[0], 5.0, atol=1e-6)


def test_rig_to_cylinder_right_is_positive_azimuth():
    # Point to the right (+X, Z=0) -> azimuth +90deg -> gx right of center.
    P = np.array([[5.0, 0.0, 0.0]])
    gx, gy, rng, valid = rp.rig_to_cylinder(P, scale=350.0, pano_w=2200, pano_h=512)
    assert valid[0]
    assert gx[0] == int(round(np.pi / 2 * 350.0 + 1100))


def test_rig_to_cylinder_invalid_at_origin():
    # Degenerate point at the origin -> invalid (no defined direction).
    P = np.array([[0.0, 0.0, 0.0]])
    gx, gy, rng, valid = rp.rig_to_cylinder(P, scale=350.0, pano_w=2200, pano_h=512)
    assert not valid[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest test_reproject.py -k rig_to_cylinder -v`
Expected: FAIL — `AttributeError: module 'reproject' has no attribute 'rig_to_cylinder'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to reproject.py
def rig_to_cylinder(P_rig, scale, pano_w, pano_h):
    """Project rig-frame points onto the cylinder (same parametrization as the
    base remap: ray = [sin az, h, cos az]).

    Returns (gx, gy, rng, valid), each shape (N,):
      gx, gy : int pano pixel (gx wraps mod pano_w)
      rng    : float distance from rig origin (z-buffer key)
      valid  : bool — finite, well-defined direction, gy in canvas
    """
    P = np.asarray(P_rig, dtype=np.float64)
    X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
    horiz = np.hypot(X, Z)
    rng = np.sqrt(X * X + Y * Y + Z * Z)
    az = np.arctan2(X, Z)
    h = np.divide(Y, horiz, out=np.zeros_like(Y), where=horiz > 1e-9)
    gx = np.mod(np.round(az * scale + pano_w / 2).astype(np.int64), pano_w)
    gy = np.round(h * scale + pano_h / 2).astype(np.int64)
    valid = (horiz > 1e-9) & np.isfinite(rng) & (gy >= 0) & (gy < pano_h)
    return gx, gy, rng, valid
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest test_reproject.py -k rig_to_cylinder -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Checkpoint**

Run: `python3 -m pytest test_reproject.py -v` (expected: 5 passed). Pause for review.
Suggested commit: `feat(reproject): rig_to_cylinder projection`

---

## Task 3: `split_near_far` — partition points by depth

**Files:**
- Modify: `reproject.py`
- Test: `test_reproject.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_reproject.py
def test_split_near_far():
    rng = np.array([1.0, 5.0, 20.0, np.inf])
    valid = np.array([True, True, True, False])
    near, far = rp.split_near_far(rng, valid, near_max=8.0)
    np.testing.assert_array_equal(near, [True, True, False, False])
    np.testing.assert_array_equal(far, [False, False, True, True])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest test_reproject.py -k split_near_far -v`
Expected: FAIL — attribute error.

- [ ] **Step 3: Write minimal implementation**

```python
# append to reproject.py
def split_near_far(rng, valid, near_max):
    """near = valid AND within near_max; far = everything else (invalid or distant).
    Returns (near_mask, far_mask) bool arrays."""
    valid = np.asarray(valid)
    near = valid & (np.asarray(rng) <= near_max)
    far = ~near
    return near, far
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest test_reproject.py -k split_near_far -v`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `python3 -m pytest test_reproject.py -v` (expected: 6 passed). Pause for review.
Suggested commit: `feat(reproject): split_near_far depth partition`

---

## Task 4: `scatter_zbuffer` — forward-scatter near points with global z-buffer

**Files:**
- Modify: `reproject.py`
- Test: `test_reproject.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_reproject.py
def test_scatter_zbuffer_nearest_wins():
    # Two points hit the same pixel; the nearer one's color must win.
    gx = np.array([2, 2])
    gy = np.array([1, 1])
    rng = np.array([5.0, 2.0])                     # second is nearer
    color = np.array([[10, 10, 10], [200, 200, 200]], dtype=np.uint8)
    oc, oz, om = rp.scatter_zbuffer(gx, gy, rng, color, pano_w=4, pano_h=3)
    assert om[1, 2]                                 # pixel marked covered
    np.testing.assert_array_equal(oc[1, 2], [200, 200, 200])
    np.testing.assert_allclose(oz[1, 2], 2.0)
    assert not om[0, 0]                             # untouched pixel empty


def test_scatter_zbuffer_empty_pixels_infinite():
    gx = np.array([0]); gy = np.array([0])
    rng = np.array([3.0]); color = np.array([[1, 2, 3]], dtype=np.uint8)
    oc, oz, om = rp.scatter_zbuffer(gx, gy, rng, color, pano_w=2, pano_h=2)
    assert np.isinf(oz[1, 1])
    assert not om[1, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest test_reproject.py -k scatter_zbuffer -v`
Expected: FAIL — attribute error.

- [ ] **Step 3: Write minimal implementation**

```python
# append to reproject.py
def scatter_zbuffer(gx, gy, rng, color, pano_w, pano_h):
    """Forward-scatter colored points onto the pano, keeping the nearest per pixel.

    gx, gy, rng: (N,). color: (N, 3) uint8. Returns:
      overlay_color: (H, W, 3) uint8
      overlay_zbuf : (H, W) float32, +inf where empty
      overlay_mask : (H, W) bool, True where any point landed
    """
    gx = np.asarray(gx, dtype=np.int64)
    gy = np.asarray(gy, dtype=np.int64)
    rng = np.asarray(rng, dtype=np.float32)
    color = np.asarray(color, dtype=np.uint8)

    flat = gy * pano_w + gx
    zbuf = np.full(pano_h * pano_w, np.inf, dtype=np.float32)
    np.minimum.at(zbuf, flat, rng)                  # nearest range per pixel

    # Write color for points that achieved (≈) the winning range at their pixel.
    win = rng <= zbuf[flat] + 1e-6
    out = np.zeros((pano_h * pano_w, 3), dtype=np.uint8)
    out[flat[win]] = color[win]

    mask = np.isfinite(zbuf)
    return (out.reshape(pano_h, pano_w, 3),
            zbuf.reshape(pano_h, pano_w),
            mask.reshape(pano_h, pano_w))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest test_reproject.py -k scatter_zbuffer -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Checkpoint**

Run: `python3 -m pytest test_reproject.py -v` (expected: 8 passed). Pause for review.
Suggested commit: `feat(reproject): scatter_zbuffer forward scatter`

---

## Task 5: `composite` — overlay onto base

**Files:**
- Modify: `reproject.py`
- Test: `test_reproject.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_reproject.py
def test_composite_overlay_replaces_base():
    base = np.full((3, 4, 3), 10, dtype=np.uint8)
    overlay = np.zeros((3, 4, 3), dtype=np.uint8)
    overlay[1, 2] = [200, 200, 200]
    mask = np.zeros((3, 4), dtype=bool)
    mask[1, 2] = True
    out = rp.composite(base, overlay, mask)
    np.testing.assert_array_equal(out[1, 2], [200, 200, 200])
    np.testing.assert_array_equal(out[0, 0], [10, 10, 10])   # base preserved
    assert out is not base                                    # does not mutate input
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest test_reproject.py -k composite -v`
Expected: FAIL — attribute error.

- [ ] **Step 3: Write minimal implementation**

```python
# append to reproject.py
def composite(base, overlay_color, overlay_mask):
    """Return base with overlay_color written where overlay_mask is True.
    Hard replace (edge feathering is applied at the integration layer with cv2).
    Does not mutate base."""
    out = np.array(base, dtype=np.uint8, copy=True)
    m = np.asarray(overlay_mask, dtype=bool)
    out[m] = np.asarray(overlay_color, dtype=np.uint8)[m]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest test_reproject.py -k composite -v`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `python3 -m pytest test_reproject.py -v` (expected: 9 passed). Pause for review.
Suggested commit: `feat(reproject): composite overlay onto base`

---

## Task 6: End-to-end pure-math integration test

**Files:**
- Test: `test_reproject.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_reproject.py
def test_end_to_end_two_cameras_agree_on_near_point():
    """A single physical point seen by two cameras at different positions must
    land on the SAME pano pixel after reprojection (this is the parallax fix)."""
    scale, pano_w, pano_h = 350.0, 2200, 512
    near_max = 8.0

    # Physical point at rig coords (1, 0, 3).
    P_world = np.array([1.0, 0.0, 3.0])

    # Camera A at origin, identity rotation: sees P in its own frame as P_world.
    Ra, ta = np.eye(3), np.zeros(3)
    Pa_cam = (P_world - ta) @ Ra            # inverse of cam_points_to_rig
    # Camera B shifted +0.5 in X, identity rotation.
    Rb, tb = np.eye(3), np.array([0.5, 0.0, 0.0])
    Pb_cam = (P_world - tb) @ Rb

    a_rig = rp.cam_points_to_rig(Pa_cam[None, :], Ra, ta)
    b_rig = rp.cam_points_to_rig(Pb_cam[None, :], Rb, tb)
    gxa, gya, _, _ = rp.rig_to_cylinder(a_rig, scale, pano_w, pano_h)
    gxb, gyb, _, _ = rp.rig_to_cylinder(b_rig, scale, pano_w, pano_h)

    assert gxa[0] == gxb[0]                  # same azimuth pixel -> no parallax split
    assert gya[0] == gyb[0]
```

- [ ] **Step 2: Run test to verify it fails, then passes**

Run: `python3 -m pytest test_reproject.py -k end_to_end -v`
Expected: PASS immediately (uses only already-implemented functions — it's a regression guard for the core property). If it FAILS, the bug is in `cam_points_to_rig`/`rig_to_cylinder`; fix there before proceeding.

- [ ] **Step 3: Checkpoint**

Run: `python3 -m pytest test_reproject.py -v` (expected: 10 passed). Pause for review.
Suggested commit: `test(reproject): end-to-end two-camera agreement guard`

---

## Task 7: Sync `reproject.py` to the rig

**Files:** none (transfer only)

- [ ] **Step 1: Copy the module to the rig**

From the Mac: `scp reproject.py nova@hailbopp-orin:~/Desktop/Navigator_Orin/`
(or git pull / USB — whatever you already use). Do **not** copy `test_reproject.py` necessarily; it's harmless if you do.

- [ ] **Step 2: Verify import on the rig**

On the rig: `cd ~/Desktop/Navigator_Orin && python3 -c "import reproject; print('ok', reproject.cam_points_to_rig.__doc__ is not None)"`
Expected: `ok True`

- [ ] **Step 3: Checkpoint** — pause.

---

## Task 8: Wire depth reprojection into `fusion.py` (rig)

**Files:**
- Modify: `~/Desktop/Navigator_Orin/fusion.py` (local mirror: `zed_360_panorama.py`)

This task is verified **visually on the rig**, not by unit test (it needs the cameras). Make the changes in order; run after the final step.

- [ ] **Step 1: Add translation extrinsics + new flags to the config block**

In the `CAMERAS` list, add a `"t"` (meters) to each entry, and add the new config flags near the other diagnostics:

```python
CAMERAS = [
    {"name": "front", "serial": 46108623, "yaw":   0.0, "pitch": 0.0, "roll": 0.0,
     "t": (0.0, 0.0,  0.711)},                       # 28" fwd
    {"name": "right", "serial": 47860268, "yaw":  90.0, "pitch": 0.0, "roll": 0.0,
     "t": (0.660, 0.0, -0.216)},                     # 26" right, 8.5" back
    {"name": "back",  "serial": 49004271, "yaw": 180.0, "pitch": 0.0, "roll": 0.0,
     "t": (0.0, 0.0, -1.422)},                       # 56" back
    {"name": "left",  "serial": 43765493, "yaw": -90.0, "pitch": 0.0, "roll": 0.0,
     "t": (-0.660, 0.0, -0.216)},                    # 26" left (X-sign fixed), 8.5" back
]

# --- Depth reprojection (Phase 2) ---
DEPTH_REPROJECT = False    # master switch: off = today's pano, on = depth-corrected
NEAR_MAX = 8.0             # meters; pixels closer than this get depth reprojection
SHOW_DEPTH_VALID = None    # camera name -> show its valid-depth mask, or None
SHOW_NEAR_MASK = False     # show which pano pixels came from the depth overlay
```

- [ ] **Step 2: Import the module, drop to SVGA, enable depth on open**

At the top, add `import reproject as rp`. Switch the resolution to SVGA when reprojecting (NEURAL depth at HD1080 is too slow on the Orin; ZED X supports SVGA = 960×600):

```python
RESOLUTION = sl.RESOLUTION.SVGA if DEPTH_REPROJECT else sl.RESOLUTION.HD1080
```

In `ZedThread.open()`, change the depth mode and set metric units:

```python
        init.depth_mode = sl.DEPTH_MODE.NEURAL if DEPTH_REPROJECT else sl.DEPTH_MODE.NONE
        init.coordinate_units = sl.UNIT.METER
```

- [ ] **Step 3: Retrieve the point cloud in the capture thread**

Add an `xyz` buffer to `ZedThread.__init__` (`self.xyz = None`) and retrieve it in `run()` alongside the color image:

```python
    def run(self):
        self.running = True
        rt = sl.RuntimeParameters()
        mat = sl.Mat()
        xyz_mat = sl.Mat()
        while self.running:
            if self.zed.grab(rt) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(mat, sl.VIEW.LEFT)
                bgr = cv2.cvtColor(mat.get_data(), cv2.COLOR_BGRA2BGR)
                xyz = None
                if DEPTH_REPROJECT:
                    self.zed.retrieve_measure(xyz_mat, sl.MEASURE.XYZ)
                    xyz = xyz_mat.get_data()[:, :, :3].copy()   # (H,W,3) meters
                with self.lock:
                    self.frame = bgr
                    self.xyz = xyz
```

Add a reader:

```python
    def read_xyz(self):
        with self.lock:
            return None if self.xyz is None else self.xyz
```

- [ ] **Step 4: Precompute per-camera rotation R and translation t at startup**

In `main()`, after maps/weights are built, build the extrinsics arrays parallel to `cams`:

```python
    Rs = [cam_to_world(c.cfg["yaw"], c.cfg["pitch"], c.cfg["roll"]) for c in cams]
    ts = [np.asarray(c.cfg["t"], dtype=np.float64) for c in cams]
    cam_ranges = [None] * len(cams)   # filled per frame, reused for far-mask
```

- [ ] **Step 5: Build the depth overlay each frame (the core wiring)**

Inside the runtime `while True:` loop, *after* the existing base panorama is computed into `pano` (uint8) but *before* `imshow`, add:

```python
            if DEPTH_REPROJECT:
                gxs, gys, rngs, cols = [], [], [], []
                for c, R, t in zip(cams, Rs, ts):
                    img = c.read(); xyz = c.read_xyz()
                    if img is None or xyz is None:
                        continue
                    P_cam = xyz.reshape(-1, 3)
                    col = img.reshape(-1, 3)
                    finite = np.isfinite(P_cam).all(axis=1)
                    P_cam, col = P_cam[finite], col[finite]
                    P_rig = rp.cam_points_to_rig(P_cam, R, t)
                    gx, gy, rng, valid = rp.rig_to_cylinder(P_rig, scale, pano_w, pano_h)
                    near, _ = rp.split_near_far(rng, valid, NEAR_MAX)
                    gxs.append(gx[near]); gys.append(gy[near])
                    rngs.append(rng[near]); cols.append(col[near])

                    if SHOW_DEPTH_VALID == c.cfg["name"]:
                        vis = (np.isfinite(xyz).all(axis=2) * 255).astype(np.uint8)
                        cv2.imshow(f"{c.cfg['name']} depth valid",
                                   cv2.resize(vis, (xyz.shape[1] // 2, xyz.shape[0] // 2)))

                if gxs:
                    gx = np.concatenate(gxs); gy = np.concatenate(gys)
                    rng = np.concatenate(rngs); col = np.concatenate(cols)
                    oc, oz, om = rp.scatter_zbuffer(gx, gy, rng, col, pano_w, pano_h)

                    # Splat + seal pinholes from the forward scatter.
                    k = np.ones((3, 3), np.uint8)
                    om_u8 = cv2.dilate((om * 255).astype(np.uint8), k)
                    oc = cv2.dilate(oc, k)
                    om_filled = cv2.morphologyEx(om_u8, cv2.MORPH_CLOSE, k) > 0

                    pano = rp.composite(pano, oc, om_filled)

                    if SHOW_NEAR_MASK:
                        cv2.imshow("near mask (white = depth overlay)",
                                   cv2.resize((om_filled * 255).astype(np.uint8),
                                              (pano_w // 2, pano_h // 2)))
```

- [ ] **Step 6: Run on the rig and visually verify**

Set `DEPTH_REPROJECT = False`, run — confirm it's byte-for-byte the old behavior (no regression).
Then set `DEPTH_REPROJECT = True`, run — confirm:
- the program opens (NEURAL depth mode is heavier; expect a slower start),
- the car region is no longer a melted blob,
- `SHOW_NEAR_MASK = True` shows the car silhouette as the overlay region.

Run: `python3 fusion.py`
Expected banner includes `DEPTH_REPROJECT = True`; the near car renders coherently. Note the fps (expected 1–5).

- [ ] **Step 7: Checkpoint** — pause for review.
Suggested commit: `feat(fusion): depth reprojection overlay (Phase 2)`

---

## Task 9: Far-only base (remove the residual smear) + extrinsics nudge

**Files:**
- Modify: `~/Desktop/Navigator_Orin/fusion.py`

After Task 8, the near car is drawn correctly by the overlay, but the BASE still contains the melted near pixels underneath. This task removes near pixels from the base so only the overlay draws them, and adds a manual nudge to fine-tune the hand-measured extrinsics.

- [ ] **Step 1: Drop near pixels from the base layer**

In the base-building loop (where `pano += warped * nw`), compute each camera's per-pixel range, remap it into pano space with the existing static maps, and zero the base weight where that pano-space range is near:

```python
            # inside the per-camera base loop, when DEPTH_REPROJECT and xyz available:
            if DEPTH_REPROJECT and (xyz := c.read_xyz()) is not None:
                cam_rng = np.linalg.norm(xyz, axis=2).astype(np.float32)   # (H,W), NaN where invalid
                pano_rng = cv2.remap(cam_rng, mx, my, cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
                far_w = nw.copy()
                far_w[(pano_rng < NEAR_MAX)] = 0.0    # near pixels handled by overlay only
                pano += warped.astype(np.float32) * far_w
            else:
                pano += warped.astype(np.float32) * (sw if ISOLATE_NAME is not None else nw)
```

(Adjust to match the exact variable names in your base loop — `warped`, `nw`, `mx`, `my`, the accumulation buffer.)

- [ ] **Step 2: Add a runtime extrinsics nudge (translation + yaw)**

Add config deltas near the depth flags:

```python
NUDGE = {        # per-camera fine-tuning on top of hand-measured extrinsics
    # "right": {"dt": (0.0, 0.0, 0.0), "dyaw": 0.0},   # meters, degrees
}
```

Apply them when building `Rs`/`ts` in `main()`:

```python
    Rs, ts = [], []
    for c in cams:
        n = NUDGE.get(c.cfg["name"], {})
        dt = np.asarray(n.get("dt", (0, 0, 0)), dtype=np.float64)
        Rs.append(cam_to_world(c.cfg["yaw"] + n.get("dyaw", 0.0),
                               c.cfg["pitch"], c.cfg["roll"]))
        ts.append(np.asarray(c.cfg["t"], dtype=np.float64) + dt)
```

- [ ] **Step 3: Run and tune on the rig**

Run with `DEPTH_REPROJECT = True`. The smear should be gone (base no longer draws the near car). If the car's two halves are slightly offset at the seam, adjust `NUDGE["right"]`/`NUDGE["back"]` `dt`/`dyaw` a few cm / degrees at a time until they line up. This is the hand-measured-extrinsics accuracy floor in action.

Run: `python3 fusion.py`
Expected: coherent near car, no melt, no smear; far background unchanged.

- [ ] **Step 4: Checkpoint** — pause for review.
Suggested commit: `feat(fusion): far-only base + extrinsics nudge`

---

## Notes / open items carried from the spec

- Confirm the **tape-measure origin** is the same physical point for all four `t` vectors. If not, recompute `t` against one shared origin before tuning `NUDGE`.
- Vertical `Y` offsets assumed 0; if cameras differ in height, fill the middle value of each `t`.
- `NEAR_MAX = 8.0` is a starting guess — lower it if distant clutter is being needlessly reprojected, raise it if the car's far edge falls into the (smearing) base.
- Hole-fill escalation: Task 8 applies splat (dilate) + morphological close. If holes still show *inside* the car silhouette (e.g. across glass with no depth), escalate to `cv2.inpaint(oc, (~om_filled & car_bbox).astype(uint8), 3, cv2.INPAINT_TELEA)` over just the overlay region — this is the spec's third-tier fill, added only if 1–2 are insufficient.
- If NEURAL depth is too slow to be usable even for inspection, drop to `sl.DEPTH_MODE.PERFORMANCE` in Step 2 of Task 8.
- Deferred (out of scope): GPU/CUDA scatter, real-time fps, photometric matching.
