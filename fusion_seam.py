#!/usr/bin/env python3
"""
fusion_seam.py — 4x ZED X -> real-time cylindrical 360° panorama with
                  depth-driven seam de-ghosting (NO per-point reprojection).

================================================================================
WHAT THIS FILE ADDS OVER zed_360_panorama.py
================================================================================
The rotation-only base pipeline (zed_360_panorama.py) feather-blends all
cameras with distance-to-edge weights.  That works well for distant scenery,
but close objects appear smeared / "melted" in overlap regions because the two
cameras see the object from different positions (parallax): the AVERAGE of two
shifted versions is a ghost.

The fix used here is:

  SEAM_SELECT — depth-driven winner-take-all in overlap regions.

  In any panorama pixel where:
    (a) two or more cameras both have valid depth readings, AND
    (b) a near object is present (min valid range < NEAR_MAX), AND
    (c) the cameras disagree substantially on range
        (max_valid_range - min_valid_range > RANGE_DISAGREE),

  stop averaging and instead PICK the single camera whose surface is NEAREST
  at that pixel.  The nearest camera is the one whose optical ray actually
  *hits* the object; the farther reading is the background behind it.

  Everywhere else — far scenery, single-camera regions, or close objects where
  both cameras agree — the normal feather blend is kept so sky/road/buildings
  remain seamless.

This stays real-time because:
  * The static cylindrical remap maps are built ONCE (same as the base).
  * Per-frame the depth path does: cv2.remap(range_img) × 4 (one per camera)
    → a (4, H, W) stack → pure NumPy min/argmin → boolean mask → np.where.
  * No per-point scatter, no millions-of-points Python loops.

================================================================================
HARDWARE  (baked-in; do not change without reading the comments)
================================================================================
Rig    : 4 ZED X (GMSL) cameras on a Jetson Orin.  CUDA available but this
         file is CPU NumPy + OpenCV only (no GPU code) so it runs anywhere.
Frame  : X-right, Y-down, Z-forward  (ZED SDK convention with Y-DOWN).
yaw    : +90 turns +Z look-dir toward +X (right).  front=0, right=+90,
         back=180, left=-90.  pitch=roll=0 (ideal mount assumed).
Resol. : ZED X supports HD1080 and SVGA only — HD720 is NOT a valid mode and
         will cause open() to fail.  SVGA is used here because depth is on and
         HD1080 + PERFORMANCE depth already saturates the ISP bandwidth.
Depth  : sl.DEPTH_MODE.PERFORMANCE.  NEURAL/NEURAL_PLUS require a one-time
         GPU-model optimization on first run (process appears frozen for minutes).
         Stay on PERFORMANCE unless you have pre-optimized with ZED_Diagnostic.
         coordinate_units = sl.UNIT.METER, so XYZ channels are already in metres.
================================================================================
"""

import os
import threading
import time

import cv2
import numpy as np
import pyzed.sl as sl

# ============================================================================
# CONFIG — hardware + seam-selection knobs
# ============================================================================

# --- Camera layout -----------------------------------------------------------
# (name, serial, yaw°).  pitch=roll=0 for all.
# Positions (t) are kept for documentation; not used in the rotation-only base.
CAMERAS = [
    {"name": "front", "serial": 46108623, "yaw":   0.0, "pitch": 0.0, "roll": 0.0,
     "t": ( 0.000,  0.0,  0.711)},   # 28" forward
    {"name": "right", "serial": 47860268, "yaw":  90.0, "pitch": 0.0, "roll": 0.0,
     "t": ( 0.660,  0.0, -0.216)},   # 26" right, 8.5" back
    {"name": "back",  "serial": 49004271, "yaw": 180.0, "pitch": 0.0, "roll": 0.0,
     "t": ( 0.000,  0.0, -1.422)},   # 56" back
    {"name": "left",  "serial": 43765493, "yaw": -90.0, "pitch": 0.0, "roll": 0.0,
     "t": (-0.660,  0.0, -0.216)},   # 26" left, 8.5" back
]

# --- Depth / seam-selection config -------------------------------------------
# SEAM_SELECT : master switch.
#   True  = depth-driven winner-take-all in disagree regions (the whole point).
#   False = pure feather blend (identical behaviour to zed_360_panorama.py).
#           Use False for A/B comparison.
SEAM_SELECT = True

# NEAR_MAX : pixels where the closest valid range is farther than this (metres)
#   are considered "background" and always use the normal feather blend.  Only
#   pixels with min_valid_range < NEAR_MAX are candidates for seam selection.
NEAR_MAX = 8.0          # metres

# RANGE_DISAGREE : within the near region, trigger winner-take-all only where
#   the SPREAD of valid ranges across cameras exceeds this threshold (metres).
#   A small spread means both cameras agree on the surface; no ghost exists.
#   A large spread means one camera sees the near object and another sees the
#   background behind it — that's the parallax ghost; select the nearest.
#   Tune downward if seams are still visible; upward if far feather breaks.
RANGE_DISAGREE = 0.5    # metres

# SHOW_WINNER_MASK : open a diagnostic window that lights up (white) every pixel
#   that was overridden by winner-take-all this frame.  Black = feather blend.
SHOW_WINNER_MASK = False

# --- ZED sensor config -------------------------------------------------------
# ZED X supports HD1080 and SVGA only.  Use SVGA when depth is on.
RESOLUTION  = sl.RESOLUTION.SVGA
FPS         = 15
# PERFORMANCE depth starts instantly; NEURAL requires slow first-run optimization.
DEPTH_MODE  = sl.DEPTH_MODE.PERFORMANCE

# --- Photometric harmonization -----------------------------------------------
# Lock exposure + WB across all cams so seams don't show brightness/colour jumps.
LOCK_EXPOSURE  = False
EXPOSURE_PCT   = 50     # 0..100
WB_KELVIN      = 4600   # 2800 (warm) .. 6500 (cool)

# --- Output ------------------------------------------------------------------
SAVE_VIDEO  = False
OUTPUT_PATH = "panorama_fusion.mp4"
SHOW_WINDOW = True      # False for headless operation

# --- Diagnostics -------------------------------------------------------------
# SHOW_PER_CAM_WINDOWS : for each camera open its raw frame AND its cylindrical
#   projection side by side (8 windows total).  Good for localising geometry bugs.
SHOW_PER_CAM_WINDOWS = False

# DEBUG_TINT : colour-tint each camera's contribution so you can see which
#   camera owns which region and where cross-fades happen.
DEBUG_TINT = False

# ISOLATE_NAME : render from a SINGLE camera only (e.g. "front").  None = blend.
ISOLATE_NAME = None

# SHOW_COVERAGE_NAME : paint which raw pixels survive into the panorama.
SHOW_COVERAGE_NAME = None

DEBUG_COLORS = {            # BGR, used only when DEBUG_TINT is True
    "front": (0,   255,   0),   # green
    "right": (0,     0, 255),   # red
    "back":  (0,   255, 255),   # yellow
    "left":  (255,   0,   0),   # blue
}


# ============================================================================
# Geometry helpers  (identical to zed_360_panorama.py)
# ============================================================================

def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]])

def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


def cam_to_world(yaw_deg, pitch_deg, roll_deg):
    """Camera->world rotation.  World frame: X-right, Y-down, Z-forward.
    Yaw about Y (down): +90 turns the +Z look-direction toward +X (right)."""
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    return _ry(y) @ _rx(p) @ _rz(r)


def build_cylindrical_maps(fx, fy, cx, cy, img_w, img_h, R_cw,
                            scale, pano_w, pano_h):
    """For each panorama pixel, find the source pixel in this camera.

    Returns (map_x, map_y, valid_mask):
      map_x[gy, gx], map_y[gy, gx]  -- source pixel in the raw frame.
      valid_mask                     -- uint8 255 where the camera sees this
                                        direction, 0 elsewhere.
    All three are static (computed once at startup).
    """
    R_wc = R_cw.T   # world->camera rotation

    gx, gy = np.meshgrid(np.arange(pano_w), np.arange(pano_h))
    phi = (gx - pano_w / 2.0) / scale      # azimuth, −π..π  (radians)
    hh  = (gy - pano_h / 2.0) / scale      # linear cylinder height

    # Ray on the unit cylinder (world coords) rotated into the camera frame.
    rays     = np.stack([np.sin(phi), hh, np.cos(phi)], axis=-1)  # (H,W,3)
    rays_cam = (rays.reshape(-1, 3) @ R_wc.T).reshape(pano_h, pano_w, 3)
    xc, yc, zc = rays_cam[..., 0], rays_cam[..., 1], rays_cam[..., 2]

    # Pinhole projection — rotation only (shared optical-centre approx).
    in_front = zc > 1e-6
    zc_safe  = np.where(in_front, zc, 1.0)
    map_x    = (fx * xc / zc_safe + cx).astype(np.float32)
    map_y    = (fy * yc / zc_safe + cy).astype(np.float32)

    valid = (in_front
             & (map_x >= 0) & (map_x <= img_w - 1)
             & (map_y >= 0) & (map_y <= img_h - 1))
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0
    return map_x, map_y, valid.astype(np.uint8) * 255


# ============================================================================
# Threaded ZED capture  (one thread per camera; grab() releases the GIL)
# ============================================================================

class ZedThread(threading.Thread):
    """Continuously grabs the freshest frame + XYZ point cloud from one camera.

    Call open() once, then start().  read() / read_xyz() return the latest
    data (None until the first successful grab).  stop() closes cleanly.
    """

    def __init__(self, cfg):
        super().__init__(daemon=True)
        self.cfg   = cfg
        self.zed   = sl.Camera()
        self.frame = None       # latest BGR image  (lock-guarded)
        self.xyz   = None       # latest XYZ cloud in METRES, or None
        self.lock  = threading.Lock()
        self.running = False
        self.fx = self.fy = self.cx = self.cy = None
        self.img_w = self.img_h = None

    def open(self):
        init = sl.InitParameters()
        init.set_from_serial_number(self.cfg["serial"])
        init.camera_resolution  = RESOLUTION     # SVGA — valid on ZED X with depth
        init.camera_fps         = FPS
        # Depth is always on in this file (needed for seam selection).
        # PERFORMANCE starts instantly; NEURAL requires one-time GPU optimization.
        init.depth_mode         = DEPTH_MODE
        init.coordinate_units   = sl.UNIT.METER  # XYZ already in metres
        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f"[{self.cfg['name']}] sl.Camera.open() failed: {err}  "
                f"(serial {self.cfg['serial']}).  "
                "Check GMSL cable, power, and that RESOLUTION=SVGA "
                "(ZED X does NOT support HD720).")

        if LOCK_EXPOSURE:
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, EXPOSURE_PCT)
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 0)
            self.zed.set_camera_settings(
                sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, WB_KELVIN)

        # ZED SDK 4.x: rectified LEFT view is an ideal pinhole — use left_cam
        # intrinsics for the cylindrical projection.
        info  = self.zed.get_camera_information()
        calib = info.camera_configuration.calibration_parameters.left_cam
        res   = info.camera_configuration.resolution
        self.fx, self.fy = calib.fx, calib.fy
        self.cx, self.cy = calib.cx, calib.cy
        self.img_w, self.img_h = res.width, res.height

    def run(self):
        self.running = True
        rt      = sl.RuntimeParameters()
        mat     = sl.Mat()
        xyz_mat = sl.Mat()
        while self.running:
            if self.zed.grab(rt) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(mat, sl.VIEW.LEFT)   # rectified RGB-A
                bgr = cv2.cvtColor(mat.get_data(), cv2.COLOR_BGRA2BGR)
                # XYZ measure is pixel-aligned with VIEW.LEFT.
                # Channels 0:3 are X, Y, Z in metres (camera frame).
                # Invalid pixels contain NaN or ±Inf.
                self.zed.retrieve_measure(xyz_mat, sl.MEASURE.XYZ)
                xyz = xyz_mat.get_data()[:, :, :3].copy()
                with self.lock:
                    self.frame = bgr
                    self.xyz   = xyz

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def read_xyz(self):
        with self.lock:
            return None if self.xyz is None else self.xyz.copy()

    def stop(self):
        self.running = False
        time.sleep(0.1)
        self.zed.close()


# ============================================================================
# Seam-selection blend  (the core contribution of this file)
# ============================================================================

def seam_select_blend(warped_list, pano_rng_list, norm_w_list):
    """Blend four cylindrical frames using depth-driven winner-take-all.

    Parameters
    ----------
    warped_list  : list of 4 (H, W, 3) uint8 arrays — each camera's colour
                   remapped into panorama space.
    pano_rng_list: list of 4 (H, W) float32 arrays — per-pixel range (metres)
                   for each camera in panorama space.  +inf where the camera
                   has no valid depth at that pixel.
    norm_w_list  : list of 4 (H, W, 1) float32 arrays — normalised feather
                   blend weights (pre-computed from distanceTransform, sum=1
                   across cameras at every pixel).

    Returns
    -------
    pano         : (H, W, 3) float32  — blended panorama
    winner_mask  : (H, W) bool        — True where winner-take-all was applied
                                        (diagnostic; used by SHOW_WINNER_MASK)

    Algorithm (fully vectorised — no Python pixel loops)
    ----------------------------------------------------
    1. Feather blend (default):
         pano_feather[y,x] = Σ_i  warped[i][y,x] * norm_w[i][y,x]
       This is identical to the original pipeline.

    2. Winner-take-all override (applied only where ALL three conditions hold):
         (a) num_valid[y,x] >= 2   — at least two cameras have depth here
         (b) min_rng[y,x] < NEAR_MAX  — a near object is present
         (c) rng_spread[y,x] > RANGE_DISAGREE
                 where spread = max_valid_rng - min_valid_rng
             — the cameras DISAGREE on depth → parallax ghost exists

       For those pixels we overwrite the feather result with the colour of
       argmin_rng[y,x] — the camera whose surface is NEAREST (i.e. the camera
       that actually hits the object, not the background behind it).

    3. Hard-edge flicker mitigation:
       The override boundary can flicker frame-to-frame when a range reading
       oscillates near the NEAR_MAX / RANGE_DISAGREE thresholds.  We soften
       this with a small morphological erosion of the winner_mask before
       application: pixels right at the boundary that are not stably "inside"
       the near-disagree region revert to feather.  The erosion kernel (3×3)
       removes isolated 1-pixel activations and boundary fringe.  This avoids
       a full temporal filter (which would need state) while eliminating the
       worst flicker.
    """
    n = len(warped_list)
    H, W = warped_list[0].shape[:2]

    # ------------------------------------------------------------------
    # Step 1 — feather blend (identical to original pipeline)
    # ------------------------------------------------------------------
    pano_feather = np.zeros((H, W, 3), dtype=np.float32)
    for warped, nw in zip(warped_list, norm_w_list):
        pano_feather += warped.astype(np.float32) * nw   # nw already (H,W,1)

    if not SEAM_SELECT:
        # A/B switch: return pure feather blend with an empty winner mask.
        return pano_feather, np.zeros((H, W), dtype=bool)

    # ------------------------------------------------------------------
    # Step 2 — build the (4, H, W) range stack
    #
    # pano_rng_list[i] is already +inf where the camera has no depth.
    # Stack into a single array so we can do axis=0 reductions in one call.
    # ------------------------------------------------------------------
    rng_stack = np.stack(pano_rng_list, axis=0).astype(np.float32)  # (4,H,W)

    # Valid = finite depth reading (not +inf we put there as sentinel).
    valid_stack = np.isfinite(rng_stack)                             # (4,H,W) bool

    # Count how many cameras have valid depth at each pixel.
    num_valid = valid_stack.sum(axis=0)                              # (H,W) int

    # Min and max valid range at each pixel.
    # Where a camera is invalid, substitute +inf (already done) so it never
    # contributes to min, and −inf would corrupt max, so substitute 0.
    rng_for_max = np.where(valid_stack, rng_stack, 0.0)
    min_rng     = rng_stack.min(axis=0)                              # (H,W) — inf where all invalid
    max_rng     = rng_for_max.max(axis=0)                            # (H,W) — 0 where all invalid

    rng_spread = max_rng - min_rng                                   # (H,W)

    # ------------------------------------------------------------------
    # Step 3 — build the winner-take-all condition mask
    # ------------------------------------------------------------------
    # All three conditions must be True simultaneously.
    cond_multi_valid  = num_valid >= 2                               # (H,W) bool
    cond_near         = min_rng < NEAR_MAX                           # (H,W) bool
    cond_disagree     = rng_spread > RANGE_DISAGREE                  # (H,W) bool

    raw_winner_mask = cond_multi_valid & cond_near & cond_disagree   # (H,W) bool

    # ------------------------------------------------------------------
    # Flicker mitigation: erode the mask by 1 pixel (3×3 kernel).
    # Pixels right on the boundary that toggle on/off frame-to-frame
    # are 1 pixel wide; eroding removes them while preserving the stable
    # interior of the override region.  The cost is one morphological op
    # per frame on a pano-sized binary mask — negligible vs. the remaps.
    # ------------------------------------------------------------------
    erode_kernel  = np.ones((3, 3), dtype=np.uint8)
    winner_mask   = cv2.erode(
        raw_winner_mask.astype(np.uint8), erode_kernel
    ).astype(bool)                                                    # (H,W) bool

    # ------------------------------------------------------------------
    # Step 4 — winner-take-all colour selection (vectorised)
    #
    # argmin across the range stack picks the nearest camera per pixel.
    # We use np.argmin which returns 0 for all-inf pixels — those pixels
    # are excluded from the override by the winner_mask anyway, so the
    # argmin value there is irrelevant.
    # ------------------------------------------------------------------
    # Stack colour: (4, H, W, 3) float32
    color_stack = np.stack(
        [w.astype(np.float32) for w in warped_list], axis=0          # (4,H,W,3)
    )

    # argmin_idx: (H, W) int — index 0..3 of nearest camera per pixel.
    argmin_idx = np.argmin(rng_stack, axis=0)                        # (H,W)

    # Gather the nearest-camera colour for every pixel.
    # np.take_along_axis requires the index to have the same ndim as the
    # array being indexed; we add axes to broadcast over H, W, and C.
    idx_expanded = argmin_idx[np.newaxis, :, :, np.newaxis]          # (1,H,W,1)
    idx_expanded = np.broadcast_to(idx_expanded,
                                   (1, H, W, 3))                     # (1,H,W,3)
    winner_color = np.take_along_axis(
        color_stack, idx_expanded, axis=0
    )[0]                                                              # (H,W,3)

    # ------------------------------------------------------------------
    # Step 5 — compose: feather everywhere; override where winner_mask
    # ------------------------------------------------------------------
    # Expand mask to (H,W,1) so it broadcasts over the colour channels.
    mask3 = winner_mask[:, :, np.newaxis]                            # (H,W,1)
    pano  = np.where(mask3, winner_color, pano_feather)              # (H,W,3)

    return pano, winner_mask


# ============================================================================
# Main
# ============================================================================

def main():
    # ---- evidence banner: prove WHICH file is running and WHAT flags are set.
    print("=" * 70)
    print(f"RUNNING FILE  : {os.path.abspath(__file__)}")
    print(f"SEAM_SELECT   = {SEAM_SELECT!r}  "
          f"(False = pure feather A/B mode)")
    print(f"NEAR_MAX      = {NEAR_MAX} m  "
          f"RANGE_DISAGREE = {RANGE_DISAGREE} m")
    print(f"SHOW_WINNER_MASK = {SHOW_WINNER_MASK!r}")
    print(f"DEPTH_MODE    = PERFORMANCE  RESOLUTION = SVGA")
    print(f"ISOLATE_NAME  = {ISOLATE_NAME!r}  "
          f"DEBUG_TINT = {DEBUG_TINT!r}")
    print("=" * 70)

    # ---- open cameras -------------------------------------------------------
    cams = [ZedThread(c) for c in CAMERAS]
    for c in cams:
        c.open()
        c.start()

    # Wait for the first frame from every camera.
    print("waiting for first frames...")
    t0 = time.time()
    while any(c.read() is None for c in cams):
        if time.time() - t0 > 10:
            raise RuntimeError("timed out waiting for camera frames (>10 s)")
        time.sleep(0.05)
    print("all cameras live.")

    # ---- build static warp maps + blend weights (ONCE) ---------------------
    # Use the first camera's intrinsics to fix the panorama geometry.
    # All four cameras on the same rig have identical SVGA intrinsics from the
    # ZED factory calibration; if they differ slightly the panorama scale is
    # set by cam[0] and is still correct to within a fraction of a percent.
    ref   = cams[0]
    scale = 350                                    # pano pixels per radian
    pano_w = int(round(2 * np.pi * scale))         # full 360° circumference
    pano_h = int(round(scale * ref.img_h / ref.fy))  # height from vertical FOV
    print(f"panorama size : {pano_w} x {pano_h}  (scale={scale} px/rad)")

    maps   = []   # per-camera (map_x, map_y)
    valids = []   # per-camera uint8 valid mask
    weights = []  # per-camera raw distanceTransform weights (unnormalised)
    covers = []   # per-camera coverage map on the RAW sensor frame

    for c in cams:
        R_cw = cam_to_world(c.cfg["yaw"], c.cfg["pitch"], c.cfg["roll"])
        mx, my, valid = build_cylindrical_maps(
            c.fx, c.fy, c.cx, c.cy, c.img_w, c.img_h,
            R_cw, scale, pano_w, pano_h)
        maps.append((mx, my))
        valids.append(valid)

        # Distance-to-edge feather weight: peaks at the centre of each camera's
        # FOV and tapers to 0 at its edges → smooth cross-fade in overlap zones.
        weights.append(cv2.distanceTransform(valid, cv2.DIST_L2, 5))

        # Back-projection coverage map (diagnostic / SHOW_COVERAGE_NAME).
        vm = valid > 0
        xs = np.clip(np.round(mx[vm]).astype(int), 0, c.img_w - 1)
        ys = np.clip(np.round(my[vm]).astype(int), 0, c.img_h - 1)
        cov = np.zeros((c.img_h, c.img_w), np.uint8)
        cov[ys, xs] = 255
        cov = cv2.dilate(cov, np.ones((9, 9), np.uint8))
        covers.append(cov)

        cols_used = int((valid.sum(axis=0) > 0).sum())
        px_used   = int((valid > 0).sum())
        print(f"[+] {c.cfg['name']:5s}  yaw={c.cfg['yaw']:+6.1f}°  "
              f"fx={c.fx:6.1f} fy={c.fy:6.1f}  "
              f"res={c.img_w}x{c.img_h}  "
              f"cols={cols_used}/{pano_w}  px={px_used}")

    # Normalise feather weights: Σ_i norm_w[i][y,x] == 1 for every (y,x) with
    # at least one valid camera.
    total = np.sum(weights, axis=0)       # (H,W)
    nz    = total > 0
    norm_w = []
    for w in weights:
        nw      = np.zeros_like(w)
        nw[nz]  = w[nz] / total[nz]
        norm_w.append(nw[..., None].astype(np.float32))   # (H,W,1)

    # Full-brightness alpha for ISOLATE_NAME (bypasses cross-fade).
    solo_w = [(v > 0).astype(np.float32)[..., None] for v in valids]

    # ---- optional video writer ----------------------------------------------
    writer = None
    if SAVE_VIDEO:
        writer = cv2.VideoWriter(
            OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"),
            FPS, (pano_w, pano_h))

    # =========================================================================
    # Real-time loop
    # =========================================================================
    print("running — press 'q' to quit.")
    frames, t_fps = 0, time.time()
    try:
        while True:
            warped_list   = []   # (H,W,3) uint8 per camera — colour in pano space
            pano_rng_list = []   # (H,W) float32 per camera — range in pano space

            for i, (c, (mx, my), nw, sw, cov) in enumerate(
                    zip(cams, maps, norm_w, solo_w, covers)):

                # ISOLATE: skip all cameras except the chosen one.
                if ISOLATE_NAME is not None and c.cfg["name"] != ISOLATE_NAME:
                    # Supply a black frame + +inf range as placeholders so the
                    # list lengths stay consistent (seam_select_blend expects 4).
                    warped_list.append(np.zeros((pano_h, pano_w, 3), np.uint8))
                    pano_rng_list.append(
                        np.full((pano_h, pano_w), np.inf, dtype=np.float32))
                    continue

                img = c.read()
                if img is None:
                    warped_list.append(np.zeros((pano_h, pano_w, 3), np.uint8))
                    pano_rng_list.append(
                        np.full((pano_h, pano_w), np.inf, dtype=np.float32))
                    continue

                # ---- coverage overlay diagnostic ----------------------------
                if SHOW_COVERAGE_NAME == c.cfg["name"]:
                    ov   = img.copy()
                    lost = cov == 0
                    ov[lost] = (ov[lost] * 0.2).astype(np.uint8)
                    cv2.imshow(
                        f"{c.cfg['name']} coverage (dark = NOT in pano)",
                        cv2.resize(ov, (img.shape[1] // 2, img.shape[0] // 2)))

                # ---- warp colour into pano space ----------------------------
                warped = cv2.remap(img, mx, my, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

                # ---- 8-window diagnostic ------------------------------------
                if SHOW_PER_CAM_WINDOWS:
                    name = c.cfg["name"]
                    cv2.imshow(f"{name} pano",
                               cv2.resize(warped, (pano_w // 3, pano_h // 3)))
                    cv2.imshow(f"{name} raw",
                               cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3)))

                # ---- per-camera colour tint ---------------------------------
                if DEBUG_TINT:
                    tint   = np.array(DEBUG_COLORS[c.cfg["name"]], np.float32) / 255.0
                    warped = (warped.astype(np.float32) * (0.5 + 0.5 * tint)
                              ).astype(np.uint8)

                warped_list.append(warped)

                # ---- remap range image into pano space ----------------------
                # cam_rng[y, x] = Euclidean distance from the camera to the
                # surface at that image pixel (metres); NaN/Inf where invalid.
                xyz = c.read_xyz()
                if xyz is not None and SEAM_SELECT:
                    # np.linalg.norm over the 3 XYZ channels gives per-pixel
                    # range.  Invalid pixels contain NaN so the norm is NaN.
                    cam_rng = np.linalg.norm(
                        xyz.astype(np.float32), axis=2)              # (H_cam, W_cam)

                    # Remap range into panorama space.
                    # INTER_NEAREST avoids interpolating across depth
                    # discontinuities (no averaging over object-boundary
                    # transitions).  borderValue=np.nan keeps "no camera"
                    # regions as NaN so we can distinguish "no data" from
                    # "near object".
                    pano_rng = cv2.remap(
                        cam_rng, mx, my, cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=np.nan)                           # (H,W) float32

                    # Replace NaN (invalid/border) with +inf so that:
                    #   • This camera is never selected as the "nearest" winner
                    #     at pixels it cannot see.
                    #   • np.isfinite() correctly identifies valid pixels.
                    pano_rng = np.where(
                        np.isfinite(pano_rng), pano_rng, np.inf
                    ).astype(np.float32)
                else:
                    # Depth unavailable or SEAM_SELECT is off: fill with +inf
                    # so this camera never triggers the winner-take-all branch.
                    pano_rng = np.full(
                        (pano_h, pano_w), np.inf, dtype=np.float32)

                pano_rng_list.append(pano_rng)

            # -----------------------------------------------------------------
            # Blend: feather + depth-driven seam selection
            # -----------------------------------------------------------------
            if ISOLATE_NAME is not None:
                # Single-camera isolation: use full-brightness weight for the
                # chosen camera, ignore the seam logic entirely.
                pano_f = np.zeros((pano_h, pano_w, 3), np.float32)
                cam_idx = next(
                    (j for j, c in enumerate(cams)
                     if c.cfg["name"] == ISOLATE_NAME), None)
                if cam_idx is not None:
                    pano_f += (warped_list[cam_idx].astype(np.float32)
                               * solo_w[cam_idx])
                pano = np.clip(pano_f, 0, 255).astype(np.uint8)
                winner_mask = np.zeros((pano_h, pano_w), dtype=bool)
            else:
                pano_f, winner_mask = seam_select_blend(
                    warped_list, pano_rng_list, norm_w)
                pano = np.clip(pano_f, 0, 255).astype(np.uint8)

            # ---- winner-mask diagnostic --------------------------------------
            if SHOW_WINNER_MASK and SEAM_SELECT:
                mask_vis = (winner_mask.astype(np.uint8) * 255)
                cv2.imshow(
                    "winner-take-all mask (white = seam-selected, black = feather)",
                    cv2.resize(mask_vis, (pano_w // 2, pano_h // 2)))

            # ---- output ------------------------------------------------------
            if writer is not None:
                writer.write(pano)

            if SHOW_WINDOW:
                cv2.imshow("360 panorama — fusion seam (q to quit)",
                           cv2.resize(pano, (pano_w // 2, pano_h // 2)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frames += 1
            if frames % 30 == 0:
                now = time.time()
                print(f"\r{30 / (now - t_fps):.1f} fps  "
                      f"winner_px={winner_mask.sum():,}", end="")
                t_fps = now

    finally:
        for c in cams:
            c.stop()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
