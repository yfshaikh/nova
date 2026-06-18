#!/usr/bin/env python3
"""
fusion_bev.py — 4x ZED X (GMSL) -> real-time bird's-eye-view (BEV) surround composite.

================================================================================
WHAT THIS DOES
================================================================================
Four ZED stereo cameras mounted on a Jetson Orin face outward ~90° apart
(front / right / back / left).  This script projects every camera's live RGB
feed onto the flat ground plane and composites the four into a single top-down
"surround view" image — the automotive-standard BEV stitched around the rig.

No GPU code.  Pure CPU: NumPy + OpenCV.  Depth mode is NONE (RGB only).

================================================================================
THE CORE IDEA: GROUND-PLANE HOMOGRAPHY VIA STATIC REMAP
================================================================================
The output image is a square top-down map covering ±EXTENT metres around the
rig origin.  Every output pixel (px, py) maps to a known ground point:

    P_world = (X,  CAMERA_HEIGHT,  Z)      [metres, world frame]

where X = (px - bev_cx) / PX_PER_M  and  Z = (py - bev_cy) / PX_PER_M.

Because the ground is fixed and the camera mount is (assumed) rigid, the
mapping from BEV pixel -> camera source pixel never changes.  We compute it
ONCE at startup (build_bev_maps) and use cv2.remap every frame — O(1) per
pixel per frame, regardless of scene content.

================================================================================
THE GROUND-TO-CAMERA PROJECTION (step by step)
================================================================================
Step 1 — Build a meshgrid of all BEV output pixels.
Step 2 — Convert each BEV pixel to its world ground point P_world:
             X   = (bev_px - bev_cx) / PX_PER_M
             Y   = CAMERA_HEIGHT          ← ground is at +CAMERA_HEIGHT because
                                            Y points DOWN and the camera is
                                            above the ground.
             Z   = (bev_py - bev_cy) / PX_PER_M
Step 3 — Express P_world relative to this camera's position:
             P_rel  = P_world - t_cam     [3-vector, same world frame]
Step 4 — Rotate into the camera's own coordinate frame:
             R_cam2rig = cam_to_world(yaw, pitch, roll)   [3x3 rotation]
             P_cam  = R_cam2rig.T @ P_rel                 [T = world->camera]
          In one vectorised call over all (H*W) BEV pixels:
             P_cam  = (P_rel_flat @ R_cam2rig)            [shape (N,3)]
Step 5 — Pinhole projection.  Only points with Zc > 0 (in front of the lens)
          are valid:
             u = fx * Xc / Zc + cx
             v = fy * Yc / Zc + cy
          Discard if (u,v) is outside the image frame.
Step 6 — Store map_x, map_y (float32) and a uint8 validity mask.
          Invalid pixels get map_x = map_y = -1.

Per frame: four cv2.remap calls + feather-weighted blend.

================================================================================
WHY THIS IS CORRECT ON THE GROUND — AND NOT ELSEWHERE
================================================================================
For a point ON the ground plane the math is exact: every camera that can see
that point produces consistent texture because the mapping is computed from
its known 3-D location.  There is no parallax on the ground.

For ANY point ABOVE the ground (car body, pedestrians, kerbs, walls) the model
collapses them to Y = CAMERA_HEIGHT (the ground).  That projects them radially
OUTWARD from the camera into the wrong BEV location — the taller the object,
the larger the displacement.  Seams between cameras will show for raised
objects.  This is inherent to single-homography BEV and is expected.

================================================================================
KEY TUNING UNKNOWNS
================================================================================
1. CAMERA_HEIGHT  — the single most important number.  Wrong by 10 cm ->
   distant ground texture shifts by ~0.5 camera-height/distance.  MUST be
   measured on the actual rig with a tape measure.

2. Pitch / roll   — assumed zero here.  A few degrees of pitch shifts the
   vanishing point of the ground projection noticeably.  Expose per-camera
   pitch in CAMERAS so it can be nudged without touching the math.

3. t (translation) — the vector from rig origin to each camera lens.  These
   come from zed_360_panorama.py and are baked-in hand measurements.  Re-
   measure or solve with a calibration target if seams on the ground shift.

================================================================================
COORDINATE CONVENTIONS
================================================================================
World frame: X-right, Y-DOWN, Z-forward.
  • The rig origin is the centre of the BEV output image.
  • Ground plane sits at world Y = +CAMERA_HEIGHT (positive because Y is down).
  • Yaw is about Y: front=0°, right=+90°, back=180°, left=-90°.
  • pitch/roll = 0 means the camera looks perfectly level (forward / outward).

================================================================================
HARDWARE
================================================================================
Rig  : 4 ZED X cameras on GMSL bus, Jetson Orin host.
Res  : SVGA (ZED X supports HD1080 / SVGA only — HD720 is NOT valid).
FPS  : 15.
Depth: NONE — BEV needs only RGB.
================================================================================
"""

import os
import threading
import time

import cv2
import numpy as np
import pyzed.sl as sl

# ----------------------------------------------------------------------------
# CONFIG  — tune for your rig
# ----------------------------------------------------------------------------

# *** CAMERA_HEIGHT is the single most important tuning knob. ***
# Set it to the measured vertical distance (metres) from the camera lens
# down to the ground surface.  Wrong values will misalign the ground texture
# between cameras.  1.5 m is a placeholder — MEASURE YOUR RIG.
CAMERA_HEIGHT = 1.5         # metres, camera above ground  ← MUST BE MEASURED

# BEV output covers ±EXTENT metres around the rig in X and Z.
EXTENT = 6.0                # metres each side of the rig

# Output resolution: PX_PER_M pixels per metre -> image is
#   (2*EXTENT*PX_PER_M) x (2*EXTENT*PX_PER_M)  ==  600 x 600 at defaults.
PX_PER_M = 50               # pixels per metre

# ZED X supports HD1080 and SVGA; HD720 is NOT a valid choice on ZED X.
RESOLUTION = sl.RESOLUTION.SVGA
FPS = 15

# Camera descriptors.  yaw/pitch/roll are in degrees.
# pitch=0 means level; add a few degrees (positive = nose-down) to correct
# a tilted mount.  Each camera's pitch can be nudged independently.
# Translations t=(X,Y,Z) are in metres from the rig origin (X-right,Y-down,
# Z-forward).  These match zed_360_panorama.py exactly.
CAMERAS = [
    # name     serial    yaw    pitch  roll  t (X,      Y,    Z)
    {"name": "front", "serial": 46108623, "yaw":   0.0, "pitch": 0.0, "roll": 0.0,
     "t": ( 0.000, 0.0,  0.711)},   # 28" forward
    {"name": "right", "serial": 47860268, "yaw":  90.0, "pitch": 0.0, "roll": 0.0,
     "t": ( 0.660, 0.0, -0.216)},   # 26" right, 8.5" back
    {"name": "back",  "serial": 49004271, "yaw": 180.0, "pitch": 0.0, "roll": 0.0,
     "t": ( 0.000, 0.0, -1.422)},   # 56" back
    {"name": "left",  "serial": 43765493, "yaw": -90.0, "pitch": 0.0, "roll": 0.0,
     "t": (-0.660, 0.0, -0.216)},   # 26" left, 8.5" back
]

# Photometric lock: set True and tune EXPOSURE_PCT / WB_KELVIN to prevent
# brightness / colour jumps at the seam between cameras.
LOCK_EXPOSURE  = False
EXPOSURE_PCT   = 50         # 0..100
WB_KELVIN      = 4600       # 2800 (warm) .. 6500 (cool)

SAVE_VIDEO     = False
OUTPUT_PATH    = "bev_surround.mp4"
SHOW_WINDOW    = True       # set False for headless / SSH sessions

# --- diagnostics ---
# SHOW_PER_CAM_BEV : open a separate window for each camera's warped BEV
#   contribution so you can see which camera covers which region.
SHOW_PER_CAM_BEV = False

# DEBUG_TINT : colour-tint each camera's region in the composite so you can
#   see camera boundaries and cross-fade zones at a glance.
DEBUG_TINT = False

DEBUG_COLORS = {            # BGR, only when DEBUG_TINT is on
    "front": (  0, 255,   0),   # green
    "right": (  0,   0, 255),   # red
    "back":  (  0, 255, 255),   # yellow
    "left":  (255,   0,   0),   # blue
}

# ----------------------------------------------------------------------------
# Derived BEV canvas dimensions (computed once here so helpers can use them)
# ----------------------------------------------------------------------------
BEV_SIZE   = int(round(2.0 * EXTENT * PX_PER_M))   # square side in pixels
BEV_CX     = BEV_SIZE / 2.0    # BEV pixel that maps to world X = 0
BEV_CY     = BEV_SIZE / 2.0    # BEV pixel that maps to world Z = 0

# ----------------------------------------------------------------------------
# Geometry helpers  (identical to zed_360_panorama.py)
# ----------------------------------------------------------------------------
def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

def cam_to_world(yaw_deg, pitch_deg, roll_deg):
    """Return the 3x3 camera->world rotation matrix.

    World frame: X-right, Y-down, Z-forward.
    Yaw  about Y (down): +90° rotates the camera's +Z look-direction toward +X.
    Pitch about X       : positive tilts the nose downward.
    Roll  about Z       : positive rolls the top of the camera to the right.
    Application order: yaw first, then pitch, then roll.
    """
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    return _ry(y) @ _rx(p) @ _rz(r)


# ----------------------------------------------------------------------------
# BEV warp-map builder  (the mathematical core — runs ONCE at startup)
# ----------------------------------------------------------------------------
def build_bev_maps(fx, fy, cx, cy, img_w, img_h,
                   R_cam2rig, t_cam,
                   bev_size=BEV_SIZE, bev_cx=BEV_CX, bev_cy=BEV_CY,
                   px_per_m=PX_PER_M, camera_height=CAMERA_HEIGHT):
    """Compute the static BEV remap tables for one camera.

    For each BEV output pixel we ask: "which source pixel should I sample?"
    The answer is found by:
      1. Converting the BEV pixel to its world ground-point P_world.
      2. Transforming P_world into the camera frame.
      3. Projecting through the camera's pinhole intrinsics.

    Parameters
    ----------
    fx, fy, cx, cy : float
        Pinhole intrinsics of the rectified LEFT view (from ZED SDK calibration).
    img_w, img_h : int
        Source image dimensions (pixels).
    R_cam2rig : ndarray (3,3)
        Camera->world (rig) rotation, i.e. cam_to_world(yaw, pitch, roll).
        Its transpose gives world->camera.
    t_cam : ndarray (3,)
        Camera origin in the world (rig) frame, metres.  Same convention as
        CAMERAS[i]["t"] — this is NOT the translation column of a pose matrix,
        it is the position of the camera expressed in world coordinates.
    bev_size : int
        Side length of the square BEV output image (pixels).
    bev_cx, bev_cy : float
        BEV pixel coordinates of the rig origin (world X=Z=0).
    px_per_m : float
        Pixels per metre in the BEV canvas.
    camera_height : float
        Vertical distance (metres) from the camera down to the ground.
        The ground is at world Y = +camera_height because Y points DOWN.

    Returns
    -------
    map_x, map_y : ndarray (bev_size, bev_size), float32
        Source pixel coordinates.  -1.0 where invalid (behind camera or
        outside the image).
    valid_mask : ndarray (bev_size, bev_size), uint8
        255 where this camera has a valid sample; 0 elsewhere.
    """
    # ------------------------------------------------------------------
    # Step 1: BEV pixel grid in output coordinates.
    # px_col runs left->right  (corresponds to world +X, right)
    # px_row runs top->bottom  (corresponds to world +Z, forward)
    # ------------------------------------------------------------------
    px_col, px_row = np.meshgrid(
        np.arange(bev_size, dtype=np.float64),   # X axis
        np.arange(bev_size, dtype=np.float64),   # Z axis
    )   # both shape (bev_size, bev_size)

    # ------------------------------------------------------------------
    # Step 2: Convert BEV pixel -> world ground point.
    #
    #   World X (right)   = (px_col - bev_cx) / px_per_m
    #   World Y (down)    = camera_height          ← ground level
    #   World Z (forward) = (px_row - bev_cy) / px_per_m
    #
    # Note: bev_cy maps to Z=0 (rig origin).  Rows BELOW bev_cy correspond
    # to NEGATIVE Z (behind the rig) in a standard Z-forward frame; BUT
    # we orient the BEV so that "top of image = in front of rig" which means
    # smaller row index -> larger Z value (more forward).
    # Therefore: Z = (bev_cy - px_row) / px_per_m  (Z increases upward in image)
    #
    # We use the convention: row 0 = +Z_max (front), row bev_size-1 = -Z_max (rear).
    # ------------------------------------------------------------------
    X_world = (px_col - bev_cx) / px_per_m                  # (H,W)
    Z_world = (bev_cy - px_row) / px_per_m                  # (H,W) — row 0 is front
    Y_world = np.full_like(X_world, camera_height)           # (H,W) constant ground plane

    # Stack into (H, W, 3) world points, then flatten to (N, 3) for the
    # vectorised rotation.
    N = bev_size * bev_size
    P_world_flat = np.stack(
        [X_world.ravel(), Y_world.ravel(), Z_world.ravel()], axis=1
    )   # shape (N, 3)

    # ------------------------------------------------------------------
    # Step 3: Translate to this camera's local position.
    #   P_rel = P_world - t_cam    (still in world/rig frame)
    # ------------------------------------------------------------------
    P_rel_flat = P_world_flat - t_cam[np.newaxis, :]         # (N, 3)

    # ------------------------------------------------------------------
    # Step 4: Rotate from world frame into the camera's own frame.
    #   R_cam2rig maps camera coords -> world coords.
    #   Its transpose (= inverse, since R is orthogonal) maps world -> camera.
    #
    #   P_cam = R_cam2rig.T @ P_rel  for each point.
    #
    #   Vectorised: P_cam_flat = P_rel_flat @ R_cam2rig   (equivalent to
    #               R_cam2rig.T @ P_rel for each column-vector, since
    #               (R.T @ v) == (v^T @ R)^T and numpy's matmul broadcasts
    #               row-vectors on the left.)
    # ------------------------------------------------------------------
    P_cam_flat = P_rel_flat @ R_cam2rig     # (N, 3); columns are Xc, Yc, Zc

    Xc = P_cam_flat[:, 0].reshape(bev_size, bev_size)
    Yc = P_cam_flat[:, 1].reshape(bev_size, bev_size)
    Zc = P_cam_flat[:, 2].reshape(bev_size, bev_size)

    # ------------------------------------------------------------------
    # Step 5: Pinhole projection.
    #   Only points in FRONT of the lens (Zc > 0) are physically visible.
    #   u = fx * Xc / Zc + cx
    #   v = fy * Yc / Zc + cy
    # ------------------------------------------------------------------
    in_front = Zc > 1e-6
    Zc_safe  = np.where(in_front, Zc, 1.0)   # avoid divide-by-zero

    map_x = (fx * Xc / Zc_safe + cx).astype(np.float32)
    map_y = (fy * Yc / Zc_safe + cy).astype(np.float32)

    # ------------------------------------------------------------------
    # Step 6: Validity: in front AND within image bounds.
    # ------------------------------------------------------------------
    valid = (
        in_front
        & (map_x >= 0) & (map_x <= img_w - 1)
        & (map_y >= 0) & (map_y <= img_h - 1)
    )

    # Mark invalid pixels so cv2.remap produces black (BORDER_CONSTANT=0)
    # for them rather than clamped edge pixels.
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0

    return map_x, map_y, valid.astype(np.uint8) * 255


# ----------------------------------------------------------------------------
# Threaded ZED capture  (one thread per camera; grab() releases the GIL)
# Adapted from zed_360_panorama.py — depth stripped out (BEV needs RGB only).
# ----------------------------------------------------------------------------
class ZedThread(threading.Thread):
    def __init__(self, cfg):
        super().__init__(daemon=True)
        self.cfg     = cfg
        self.zed     = sl.Camera()
        self.frame   = None          # latest BGR frame (lock-guarded)
        self.lock    = threading.Lock()
        self.running = False
        # Intrinsics — filled in open()
        self.fx = self.fy = self.cx = self.cy = None
        self.img_w = self.img_h = None

    def open(self):
        init = sl.InitParameters()
        init.set_from_serial_number(self.cfg["serial"])
        init.camera_resolution = RESOLUTION
        init.camera_fps        = FPS
        # BEV only needs colour — depth is off entirely.
        init.depth_mode        = sl.DEPTH_MODE.NONE
        init.coordinate_units  = sl.UNIT.METER

        if self.zed.open(init) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f"could not open {self.cfg['name']} (serial {self.cfg['serial']})"
            )

        # Optional photometric lock: keeps the four feeds consistent at seams.
        if LOCK_EXPOSURE:
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC,              0)
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE,              EXPOSURE_PCT)
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO,     0)
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, WB_KELVIN)

        # ZED SDK 4.x: intrinsics of the rectified LEFT view are an ideal
        # pinhole — exactly what the projection formula needs.
        info  = self.zed.get_camera_information()
        calib = info.camera_configuration.calibration_parameters.left_cam
        res   = info.camera_configuration.resolution
        self.fx, self.fy = calib.fx, calib.fy
        self.cx, self.cy = calib.cx, calib.cy
        self.img_w, self.img_h = res.width, res.height

    def run(self):
        self.running = True
        rt  = sl.RuntimeParameters()
        mat = sl.Mat()
        while self.running:
            if self.zed.grab(rt) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(mat, sl.VIEW.LEFT)   # rectified LEFT
                bgr = cv2.cvtColor(mat.get_data(), cv2.COLOR_BGRA2BGR)
                with self.lock:
                    self.frame = bgr

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        time.sleep(0.1)
        self.zed.close()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    # ---- startup evidence banner: confirm which file is actually running ----
    print("=" * 70)
    print(f"RUNNING FILE    : {os.path.abspath(__file__)}")
    print(f"EXTENT          = ±{EXTENT} m  ->  BEV canvas {BEV_SIZE}x{BEV_SIZE} px")
    print(f"PX_PER_M        = {PX_PER_M}")
    print(f"CAMERA_HEIGHT   = {CAMERA_HEIGHT} m   *** MUST BE MEASURED — see docstring ***")
    print(f"RESOLUTION      = {RESOLUTION}")
    print(f"FPS             = {FPS}")
    print(f"SHOW_PER_CAM_BEV= {SHOW_PER_CAM_BEV}")
    print(f"DEBUG_TINT      = {DEBUG_TINT}")
    print("=" * 70)

    # --- open cameras -------------------------------------------------------
    cams = [ZedThread(c) for c in CAMERAS]
    for c in cams:
        print(f"  opening {c.cfg['name']} (serial {c.cfg['serial']}) ...")
        c.open()
        c.start()

    # Wait for the first frame from every camera (10 s timeout).
    print("waiting for first frames...")
    t0 = time.time()
    while any(c.read() is None for c in cams):
        if time.time() - t0 > 10:
            raise RuntimeError("timed out waiting for camera frames")
        time.sleep(0.05)
    print("all cameras live.")

    # --- Build static BEV warp maps + feather weights (ONCE) ---------------
    # Each map tells cv2.remap "for BEV pixel (r,c), sample source pixel (u,v)".
    # The weights are a distance-transform of the valid mask, so cameras
    # cross-fade smoothly in the overlap zones — same idea as the cylinder pano.
    maps    = []   # list of (map_x, map_y)
    valids  = []   # list of uint8 validity masks (255/0)
    weights = []   # list of float32 feather weights

    for c in cams:
        R_cam2rig = cam_to_world(c.cfg["yaw"], c.cfg["pitch"], c.cfg["roll"])
        t_cam     = np.asarray(c.cfg["t"], dtype=np.float64)

        mx, my, valid_mask = build_bev_maps(
            c.fx, c.fy, c.cx, c.cy,
            c.img_w, c.img_h,
            R_cam2rig, t_cam,
        )
        maps.append((mx, my))
        valids.append(valid_mask)

        # distanceTransform on the valid mask: pixels near the valid-region
        # centre get high weight; pixels near the edge taper to 0.  This
        # produces a smooth cross-fade wherever two camera footprints overlap.
        w = cv2.distanceTransform(valid_mask, cv2.DIST_L2, 5)
        weights.append(w)

        # diagnostic: how much of the BEV canvas this camera covers
        px_used   = int((valid_mask > 0).sum())
        px_total  = BEV_SIZE * BEV_SIZE
        pct       = 100.0 * px_used / px_total
        print(f"[+] {c.cfg['name']:5s}  yaw={c.cfg['yaw']:+6.1f}°  "
              f"fx={c.fx:6.1f} fy={c.fy:6.1f}  "
              f"cx={c.cx:6.1f} cy={c.cy:6.1f}  "
              f"res={c.img_w}x{c.img_h}  "
              f"BEV_coverage={px_used}/{px_total} ({pct:.1f}%)")

    # Normalise weights so they sum to 1.0 at every pixel (including overlaps).
    total_w = np.sum(weights, axis=0)       # (H,W) float32
    nonzero = total_w > 0.0
    norm_weights = []
    for w in weights:
        nw = np.zeros_like(w)
        nw[nonzero] = w[nonzero] / total_w[nonzero]
        norm_weights.append(nw[..., np.newaxis].astype(np.float32))   # (H,W,1)

    # --- video writer -------------------------------------------------------
    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (BEV_SIZE, BEV_SIZE))
        print(f"recording to {OUTPUT_PATH}")

    # --- real-time loop -----------------------------------------------------
    # Per-frame cost: 4x cv2.remap (table lookup) + weighted accumulation.
    print("running. press 'q' to quit.")
    frame_count = 0
    t_fps = time.time()

    try:
        while True:
            bev = np.zeros((BEV_SIZE, BEV_SIZE, 3), dtype=np.float32)

            for c, (mx, my), nw in zip(cams, maps, norm_weights):
                img = c.read()
                if img is None:
                    continue

                # Warp this camera's frame into BEV space.
                # BORDER_CONSTANT=0: pixels that map_x/map_y = -1 (invalid)
                # produce black, not clamped edge colour.
                warped = cv2.remap(
                    img, mx, my,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )   # shape (BEV_SIZE, BEV_SIZE, 3), uint8

                # Optional per-camera tint (diagnostic only).
                if DEBUG_TINT:
                    tint = np.array(
                        DEBUG_COLORS[c.cfg["name"]], dtype=np.float32
                    ) / 255.0
                    warped = (warped.astype(np.float32) * (0.5 + 0.5 * tint)
                              ).astype(np.uint8)

                # Optional per-camera BEV window.
                if SHOW_PER_CAM_BEV:
                    cv2.imshow(
                        f"BEV {c.cfg['name']}",
                        cv2.resize(warped, (BEV_SIZE // 2, BEV_SIZE // 2)),
                    )

                # Accumulate: float32 weighted sum.
                bev += warped.astype(np.float32) * nw

            bev = np.clip(bev, 0, 255).astype(np.uint8)

            # Draw a small filled square at the rig centre so orientation is
            # instantly obvious.  The rig sits at BEV pixel (BEV_CX, BEV_CY).
            rig_cx = int(round(BEV_CX))
            rig_cy = int(round(BEV_CY))
            marker_half = max(4, BEV_SIZE // 120)   # ~5 px at 600x600
            cv2.rectangle(
                bev,
                (rig_cx - marker_half, rig_cy - marker_half),
                (rig_cx + marker_half, rig_cy + marker_half),
                (255, 255, 255),   # white fill
                thickness=-1,
            )
            cv2.rectangle(
                bev,
                (rig_cx - marker_half, rig_cy - marker_half),
                (rig_cx + marker_half, rig_cy + marker_half),
                (0, 0, 0),         # black border for contrast
                thickness=1,
            )

            if writer is not None:
                writer.write(bev)

            if SHOW_WINDOW:
                display = cv2.resize(bev, (BEV_SIZE, BEV_SIZE))
                cv2.imshow("BEV surround (q to quit)", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            if frame_count % 30 == 0:
                now = time.time()
                fps = 30.0 / (now - t_fps)
                print(f"\r{fps:.1f} fps", end="", flush=True)
                t_fps = now

    finally:
        print()   # newline after fps counter
        for c in cams:
            c.stop()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("done.")


if __name__ == "__main__":
    main()
