#!/usr/bin/env python3
"""
zed_360_panorama.py — 4x ZED -> real-time cylindrical 360° panorama (RGB POC).

================================================================================
WHAT THIS DOES
================================================================================
Four ZED stereo cameras are mounted on top of a car, facing outward, ~90° apart
(front / right / back / left). This script stitches their live RGB feeds into a
single horizontal 360° panorama in real time and (optionally) records it to mp4.

We deliberately do NOT use:
  - cv2.Stitcher : feature matching; assumes one nodal point — fails on a wide-
                   baseline rig with dead zones and parallax.
  - sl.Fusion    : fuses only metadata (skeletons/objects/tracking) — never
                   produces a stitched RGB image.
  - ZED360       : needs overlapping views to solve extrinsics; this rig has
                   dead zones, so extrinsics are set by hand (yaw/pitch/roll).

We DO use a custom **rotation-only cylindrical projection**:
  - OpenCV only for cv2.remap + cv2.imshow (NOT the Stitcher).
  - pyzed only for capture (NOT Fusion).

================================================================================
PIPELINE
================================================================================
  grab (4 threads) -> retrieve rectified LEFT view -> warp-to-cylinder
  (static precomputed maps) -> feather-blend -> show / save.

Threaded capture: one ZedThread per camera holds the latest frame. grab()
releases the GIL, so all four cameras stay current instead of serializing.

The expensive geometry (per-pixel source lookup + blend weights) is precomputed
ONCE in build_cylindrical_maps(). The per-frame cost is just cv2.remap (a table
lookup) + a weighted sum — that's what makes it real time.

================================================================================
HOW THE WARP WORKS
================================================================================
Picture a unit cylinder around the rig; the panorama is that cylinder unrolled:
    - horizontal axis (gx) -> azimuth angle  phi   (which direction you face)
    - vertical   axis (gy) -> linear HEIGHT  hh    (a true cylinder, NOT an
                              elevation angle — keeps vertical lines straight)

For every panorama pixel we:
  1. Build its 3D ray on the cylinder in WORLD coords: [sin(phi), hh, cos(phi)].
  2. Rotate that ray into a camera's frame using that camera's full yaw/pitch/
     roll rotation (world->camera).
  3. Project through the camera's pinhole intrinsics (fx, fy, cx, cy) to a
     source pixel; mark it valid only if it's in front of the camera and lands
     inside the frame.
  4. Store the source pixel in a remap table + a distance-to-edge blend weight.

================================================================================
COORDINATE CONVENTIONS  (get these wrong and the image flips / mirrors)
================================================================================
World frame: X-right, Y-DOWN, Z-forward (note: Y points down here, unlike the
             Y-up ZED default — so the cylinder height needs no sign flip).
yaw   : outward heading in degrees. front=0, right=+90, back=180, left=-90.
        +90 turns the +Z look-direction toward +X (right).
pitch : mount tilt, up(-) / down(+).   roll : camera-axis tilt.
        Start pitch/roll at 0; nudge if a seam is vertically misaligned/tilted.

If the panorama comes out left-right MIRRORED (right cam on the left), flip the
SIGN of every yaw below.

================================================================================
THE BIG ASSUMPTION (and its one real limitation)
================================================================================
This model assumes all four cameras share a single optical center and differ
only by rotation. They do NOT — they sit ~70 cm apart on the roof. We ignore
those translations on purpose: a translation's effect depends on how FAR each
point is, and this depth-free model only has ray *directions*, not distances.
So translations cannot be applied here.

Consequence: distant scenery stitches cleanly; CLOSE objects (a car right next
to the rig) ghost / double / look "half-built" in the overlap region, because
each camera sees them from a genuinely different position (PARALLAX). The only
real fix is Phase 2: turn on ZED depth, unproject each pixel to a 3D point, and
reproject onto the cylinder accounting for each camera's true position. Every
part of this file is reused by Phase 2 except the warp step.
================================================================================
"""

import os
import threading
import time

import cv2
import numpy as np
import pyzed.sl as sl

import reproject as rp          # Phase 2 depth-reprojection math (pure NumPy)

# ----------------------------------------------------------------------------
# CONFIG  — your rig
# ----------------------------------------------------------------------------
# Each camera's outward orientation AND its position (t) relative to the rig
# origin, in METERS. yaw/pitch/roll feed the rotation; t is only used by the
# depth reprojection (the rotation-only base ignores it). t = inches * 0.0254.
# NOTE: all four t vectors MUST be measured from the SAME physical origin point.
CAMERAS = [
    {"name": "front", "serial": 46108623, "yaw":   0.0, "pitch": 0.0, "roll": 0.0,
     "t": (0.0,    0.0,  0.711)},        # 28" forward
    {"name": "right", "serial": 47860268, "yaw":  90.0, "pitch": 0.0, "roll": 0.0,
     "t": (0.660,  0.0, -0.216)},        # 26" right, 8.5" back
    {"name": "back",  "serial": 49004271, "yaw": 180.0, "pitch": 0.0, "roll": 0.0,
     "t": (0.0,    0.0, -1.422)},        # 56" back
    {"name": "left",  "serial": 43765493, "yaw": -90.0, "pitch": 0.0, "roll": 0.0,
     "t": (-0.660, 0.0, -0.216)},        # 26" left (X-sign fixed), 8.5" back
]

# ----------------------------------------------------------------------------
# DEPTH REPROJECTION (Phase 2) — corrects close-range parallax
# ----------------------------------------------------------------------------
# DEPTH_REPROJECT: master switch. False = the original rotation-only pano (far
#   stitches fine, near objects melt). True = near field (<= NEAR_MAX) is
#   re-rendered from the rig origin using ZED depth so the cameras agree.
DEPTH_REPROJECT = False
# Depth mode. PERFORMANCE is light and starts instantly. NEURAL/NEURAL_PLUS are
# higher quality BUT require a one-time, multi-minute AI-model optimization on
# the FIRST run for a given GPU — during which the process looks frozen ("not
# responding"). Pre-optimize once with `ZED_Diagnostic` before using NEURAL, or
# just stay on PERFORMANCE for the POC.
DEPTH_MODE = sl.DEPTH_MODE.PERFORMANCE
NEAR_MAX = 8.0                  # meters; closer than this gets depth reprojection
SHOW_DEPTH_VALID = None         # camera name -> show its valid-depth mask, or None
SHOW_NEAR_MASK = False          # show which pano pixels came from the depth overlay
# Per-camera fine-tuning of the hand-measured extrinsics (visually null out seam
# misalignment on a near object). dt in meters, dyaw in degrees.
NUDGE = {
    # "right": {"dt": (0.0, 0.0, 0.0), "dyaw": 0.0},
}

RESOLUTION = (sl.RESOLUTION.SVGA if DEPTH_REPROJECT   # NEURAL depth at HD1080 is
              else sl.RESOLUTION.HD1080)              # too slow; SVGA is valid on
                                                      # ZED X. (HD720 is NOT.)
FPS = 15

# Photometric harmonization: lock exposure + white balance across ALL cams so
# seams don't show a brightness/color jump. Tune EXPOSURE/WB to your scene.
LOCK_EXPOSURE = False
EXPOSURE_PCT = 50      # 0..100
WB_KELVIN = 4600       # 2800 (warm) .. 6500 (cool)

SAVE_VIDEO = False
OUTPUT_PATH = "panorama.mp4"
SHOW_WINDOW = True     # set False if running headless

# ----------------------------------------------------------------------------
# DIAGNOSTICS  — set all off for the clean panorama
# ----------------------------------------------------------------------------
# SHOW_PER_CAM_WINDOWS : the "8-window diagnostic". For each camera, opens its
#   RAW frame and its cylindrical projection ("<name> pano") side by side.
#   4 cams x 2 = 8 windows. Compare per camera to localize missing content:
#     raw full  + pano full         -> camera & projection both fine
#     raw full  + pano broken        -> remap/geometry bug (trace projection)
#     raw already broken             -> camera coverage/occlusion (no SW fix)
#     object split across two panos  -> parallax (close object; no SW fix)
SHOW_PER_CAM_WINDOWS = False

# DEBUG_TINT : color-tint each camera's contribution in the final panorama so
#   you can see exactly which camera owns which region and where they cross-fade.
DEBUG_TINT = False

# ISOLATE_NAME : render the panorama from a SINGLE camera only (e.g. "back").
#   Shown at full brightness (its valid region, no cross-fade) so you can see
#   exactly what that one camera contributes. None = blend all cameras.
ISOLATE_NAME = None

# SHOW_COVERAGE_NAME : paint, ON the raw frame of the named camera, which pixels
#   actually survive into the panorama. Dark regions = NOT mapped into the pano.
#   This settles "are we losing FOV?" definitively: if the green car edge stays
#   bright, it's kept (just compressed at the cylinder edge); if it goes dark,
#   it's genuinely dropped. None = off.
SHOW_COVERAGE_NAME = None

DEBUG_COLORS = {                # BGR, used only when DEBUG_TINT is on
    "front": (0,   255, 0),     # green
    "right": (0,   0,   255),   # red
    "back":  (0,   255, 255),   # yellow
    "left":  (255, 0,   0),     # blue
}

# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def _rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def _ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def _rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def cam_to_world(yaw_deg, pitch_deg, roll_deg):
    """Camera->world rotation. World frame: X-right, Y-down, Z-forward.
    Yaw about Y (down): +90 turns the +Z look-direction toward +X (right)."""
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    return _ry(y) @ _rx(p) @ _rz(r)


def build_cylindrical_maps(fx, fy, cx, cy, img_w, img_h, R_cw,
                           scale, pano_w, pano_h):
    """For each panorama pixel, find the source pixel in this camera.
    Returns (map_x, map_y, valid_mask) — all static, computed once.

    map_x[gy,gx], map_y[gy,gx] = source pixel in the raw frame to sample for
    panorama pixel (gx,gy). valid_mask = 255 where this camera actually sees
    that direction (in front of the lens AND inside the frame), else 0.
    """
    R_wc = R_cw.T  # world->camera (rotate world rays into this camera's frame)

    # Panorama pixel grid -> cylinder ray angles/heights.
    gx, gy = np.meshgrid(np.arange(pano_w), np.arange(pano_h))
    phi = (gx - pano_w / 2.0) / scale          # azimuth, -pi..pi (radians)
    hh = (gy - pano_h / 2.0) / scale           # linear cylinder height

    # Ray on the unit cylinder, in world coords, then rotated into the camera.
    rays = np.stack([np.sin(phi), hh, np.cos(phi)], axis=-1)   # (H,W,3)
    rays_cam = rays.reshape(-1, 3) @ R_wc.T
    rays_cam = rays_cam.reshape(pano_h, pano_w, 3)
    xc, yc, zc = rays_cam[..., 0], rays_cam[..., 1], rays_cam[..., 2]

    # Pinhole projection; only rays in front of the camera (zc > 0) are valid.
    # NOTE: rotation only — no translation is applied (depth-free model; see the
    # module docstring). This is the shared-optical-center approximation.
    in_front = zc > 1e-6
    zc_safe = np.where(in_front, zc, 1.0)
    map_x = (fx * xc / zc_safe + cx).astype(np.float32)
    map_y = (fy * yc / zc_safe + cy).astype(np.float32)

    valid = (in_front
             & (map_x >= 0) & (map_x <= img_w - 1)
             & (map_y >= 0) & (map_y <= img_h - 1))
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0
    return map_x, map_y, valid.astype(np.uint8) * 255


# ----------------------------------------------------------------------------
# Threaded ZED capture (one thread per camera; grab() releases the GIL)
# ----------------------------------------------------------------------------
class ZedThread(threading.Thread):
    def __init__(self, cfg):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.zed = sl.Camera()
        self.frame = None              # latest BGR frame (shared, lock-guarded)
        self.xyz = None                # latest XYZ point cloud (meters), or None
        self.lock = threading.Lock()
        self.running = False
        self.fx = self.fy = self.cx = self.cy = None
        self.img_w = self.img_h = None

    def open(self):
        init = sl.InitParameters()
        init.set_from_serial_number(self.cfg["serial"])
        init.camera_resolution = RESOLUTION
        init.camera_fps = FPS
        # Depth only when reprojecting; DEPTH_MODE config picks the engine.
        init.depth_mode = DEPTH_MODE if DEPTH_REPROJECT else sl.DEPTH_MODE.NONE
        init.coordinate_units = sl.UNIT.METER
        if self.zed.open(init) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"could not open {self.cfg['name']} "
                               f"(serial {self.cfg['serial']})")

        # Optional: lock exposure/WB so the four feeds match at the seams.
        if LOCK_EXPOSURE:
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, EXPOSURE_PCT)
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 0)
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, WB_KELVIN)

        # ZED SDK 4.x calibration path. The rectified LEFT view is an ideal
        # pinhole, so these intrinsics are exactly what the projection needs.
        info = self.zed.get_camera_information()
        calib = info.camera_configuration.calibration_parameters.left_cam
        res = info.camera_configuration.resolution
        self.fx, self.fy, self.cx, self.cy = calib.fx, calib.fy, calib.cx, calib.cy
        self.img_w, self.img_h = res.width, res.height

    def run(self):
        # Continuously grab the freshest frame (and point cloud) into self.*.
        self.running = True
        rt = sl.RuntimeParameters()
        mat = sl.Mat()
        xyz_mat = sl.Mat()
        while self.running:
            if self.zed.grab(rt) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(mat, sl.VIEW.LEFT)   # rectified
                bgr = cv2.cvtColor(mat.get_data(), cv2.COLOR_BGRA2BGR)
                xyz = None
                if DEPTH_REPROJECT:
                    # XYZ measure is pixel-aligned with VIEW.LEFT; channels 0:3
                    # are X,Y,Z in meters (camera frame). Invalid = NaN/Inf.
                    self.zed.retrieve_measure(xyz_mat, sl.MEASURE.XYZ)
                    xyz = xyz_mat.get_data()[:, :, :3].copy()
                with self.lock:
                    self.frame = bgr
                    self.xyz = xyz

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame

    def read_xyz(self):
        with self.lock:
            return None if self.xyz is None else self.xyz

    def stop(self):
        self.running = False
        time.sleep(0.1)
        self.zed.close()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    # --- evidence banner: prove WHICH file is running and WHAT flags it sees.
    print("=" * 70)
    print(f"RUNNING FILE : {os.path.abspath(__file__)}")
    print(f"ISOLATE_NAME = {ISOLATE_NAME!r}")
    print(f"DEBUG_TINT   = {DEBUG_TINT!r}")
    print(f"SHOW_PER_CAM_WINDOWS = {SHOW_PER_CAM_WINDOWS!r}")
    print(f"DEPTH_REPROJECT = {DEPTH_REPROJECT!r}  NEAR_MAX = {NEAR_MAX}")
    print("=" * 70)

    cams = [ZedThread(c) for c in CAMERAS]
    for c in cams:
        c.open()
        c.start()

    # Wait for the first frame from every camera.
    print("waiting for first frames...")
    t0 = time.time()
    while any(c.read() is None for c in cams):
        if time.time() - t0 > 10:
            raise RuntimeError("timed out waiting for camera frames")
        time.sleep(0.05)

    # --- Build static warp maps + blend weights (ONCE) ---------------------
    ref = cams[0]
    scale = 350                                 # panorama pixels per radian
    pano_w = int(round(2 * np.pi * scale))      # full 360° around
    pano_h = int(round(scale * ref.img_h / ref.fy))   # height from vertical FOV
    print(f"panorama: {pano_w} x {pano_h}")

    maps, weights, valids, covers = [], [], [], []
    for c in cams:
        R_cw = cam_to_world(c.cfg["yaw"], c.cfg["pitch"], c.cfg["roll"])
        mx, my, valid = build_cylindrical_maps(
            c.fx, c.fy, c.cx, c.cy, c.img_w, c.img_h, R_cw, scale, pano_w, pano_h)
        maps.append((mx, my))
        valids.append(valid)
        # distance-to-edge -> smooth feather that cross-fades in overlaps
        weights.append(cv2.distanceTransform(valid, cv2.DIST_L2, 5))

        # Back-projection coverage: mark which RAW pixels some pano pixel samples.
        # The cylinder undersamples the compressed frame edges (several raw px per
        # pano column), so dilate to fill that sampling sparsity — what stays dark
        # after dilation is genuinely NOT in the panorama, not just undersampled.
        vm = valid > 0
        xs = np.clip(np.round(mx[vm]).astype(int), 0, c.img_w - 1)
        ys = np.clip(np.round(my[vm]).astype(int), 0, c.img_h - 1)
        cov = np.zeros((c.img_h, c.img_w), np.uint8)
        cov[ys, xs] = 255
        cov = cv2.dilate(cov, np.ones((9, 9), np.uint8))
        covers.append(cov)

        # --- coverage diagnostic: how much of the canvas this camera lights up.
        cols_used = int((valid.sum(axis=0) > 0).sum())
        px_used = int((valid > 0).sum())
        print(f"[+] {c.cfg['name']:5s}  yaw={c.cfg['yaw']:+6.1f}°  "
              f"fx={c.fx:6.1f} fy={c.fy:6.1f} cx={c.cx:6.1f} cy={c.cy:6.1f}  "
              f"res={c.img_w}x{c.img_h}  "
              f"cols_used={cols_used}/{pano_w}  px_used={px_used}")

    # Normalize feather weights so the four cameras sum to 1 at every pixel.
    total = np.sum(weights, axis=0)
    nz = total > 0
    norm_w = []
    for w in weights:
        nw = np.zeros_like(w)
        nw[nz] = w[nz] / total[nz]
        norm_w.append(nw[..., None].astype(np.float32))   # (H,W,1)

    # Full-brightness alpha (1 where the camera is valid) — used by ISOLATE_NAME
    # so a single camera shows its true content instead of a cross-faded share.
    solo_w = [(v > 0).astype(np.float32)[..., None] for v in valids]

    # Per-camera extrinsics for depth reprojection: rotation (cam->rig) and
    # translation (rig origin -> camera), with the optional manual NUDGE applied.
    Rs, ts = [], []
    for c in cams:
        n = NUDGE.get(c.cfg["name"], {})
        dt = np.asarray(n.get("dt", (0.0, 0.0, 0.0)), dtype=np.float64)
        Rs.append(cam_to_world(c.cfg["yaw"] + n.get("dyaw", 0.0),
                               c.cfg["pitch"], c.cfg["roll"]))
        ts.append(np.asarray(c.cfg["t"], dtype=np.float64) + dt)

    writer = None
    if SAVE_VIDEO:
        writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"),
                                 FPS, (pano_w, pano_h))

    # --- Real-time loop: per frame it's just remap + weighted sum ----------
    print("running. press 'q' to quit.")
    frames, t_fps = 0, time.time()
    try:
        while True:
            pano = np.zeros((pano_h, pano_w, 3), np.float32)
            gxs, gys, rngs, cols = [], [], [], []   # near points for the overlay
            for c, (mx, my), nw, sw, cov, R, t in zip(
                    cams, maps, norm_w, solo_w, covers, Rs, ts):
                # ISOLATE: skip every camera except the chosen one.
                if ISOLATE_NAME is not None and c.cfg["name"] != ISOLATE_NAME:
                    continue

                img = c.read()
                if img is None:
                    continue
                xyz = c.read_xyz() if DEPTH_REPROJECT else None

                # --- coverage overlay: darken raw pixels NOT used by the pano ---
                if SHOW_COVERAGE_NAME == c.cfg["name"]:
                    ov = img.copy()
                    lost = cov == 0
                    ov[lost] = (ov[lost] * 0.2).astype(np.uint8)
                    cv2.imshow(f"{c.cfg['name']} coverage (dark = NOT in pano)",
                               cv2.resize(ov, (img.shape[1] // 2, img.shape[0] // 2)))

                # --- depth-valid diagnostic: where this camera has depth --------
                if SHOW_DEPTH_VALID == c.cfg["name"] and xyz is not None:
                    vis = (np.isfinite(xyz).all(axis=2) * 255).astype(np.uint8)
                    cv2.imshow(f"{c.cfg['name']} depth valid",
                               cv2.resize(vis, (xyz.shape[1] // 2, xyz.shape[0] // 2)))
                # ----------------------------------------------------------------

                warped = cv2.remap(img, mx, my, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

                # --- 8-window diagnostic: raw frame vs. its projection -------
                if SHOW_PER_CAM_WINDOWS:
                    name = c.cfg["name"]
                    cv2.imshow(f"{name} pano",
                               cv2.resize(warped, (pano_w // 3, pano_h // 3)))
                    cv2.imshow(f"{name} raw",
                               cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3)))
                # -------------------------------------------------------------

                # --- per-camera color tint -----------------------------------
                if DEBUG_TINT:
                    tint = np.array(DEBUG_COLORS[c.cfg["name"]], np.float32) / 255.0
                    warped = warped.astype(np.float32) * (0.5 + 0.5 * tint)
                # -------------------------------------------------------------

                # --- BASE layer ----------------------------------------------
                if DEPTH_REPROJECT and ISOLATE_NAME is None and xyz is not None:
                    # Drop NEAR pixels from the base so the parallax smear is never
                    # generated — they're redrawn correctly by the depth overlay.
                    # Remap this camera's per-pixel range into pano space and zero
                    # the base weight where it's near. NaN (invalid/far) stays.
                    cam_rng = np.linalg.norm(xyz, axis=2).astype(np.float32)
                    pano_rng = cv2.remap(cam_rng, mx, my, cv2.INTER_NEAREST,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=np.nan)
                    far_w = nw.copy()
                    far_w[pano_rng < NEAR_MAX] = 0.0
                    pano += warped.astype(np.float32) * far_w
                else:
                    blend = sw if ISOLATE_NAME is not None else nw
                    pano += warped.astype(np.float32) * blend

                # --- gather NEAR points for the depth OVERLAY ----------------
                if DEPTH_REPROJECT and xyz is not None:
                    P_cam = xyz.reshape(-1, 3)
                    col = img.reshape(-1, 3)
                    finite = np.isfinite(P_cam).all(axis=1)
                    P_rig = rp.cam_points_to_rig(P_cam[finite], R, t)
                    gxp, gyp, rngp, validp = rp.rig_to_cylinder(
                        P_rig, scale, pano_w, pano_h)
                    near, _ = rp.split_near_far(rngp, validp, NEAR_MAX)
                    gxs.append(gxp[near]); gys.append(gyp[near])
                    rngs.append(rngp[near]); cols.append(col[finite][near])

            pano = np.clip(pano, 0, 255).astype(np.uint8)

            # --- composite the depth OVERLAY (global z-buffer across cameras) ---
            if DEPTH_REPROJECT and ISOLATE_NAME is None and gxs:
                gx = np.concatenate(gxs); gy = np.concatenate(gys)
                rng = np.concatenate(rngs); col = np.concatenate(cols)
                if gx.size:
                    oc, _oz, om = rp.scatter_zbuffer(gx, gy, rng, col, pano_w, pano_h)
                    # Splat (dilate) + seal pinholes (morphological close).
                    k = np.ones((3, 3), np.uint8)
                    om_filled = cv2.morphologyEx((om * 255).astype(np.uint8),
                                                 cv2.MORPH_CLOSE, k) > 0
                    oc = cv2.dilate(oc, k)
                    pano = rp.composite(pano, oc, om_filled)
                    if SHOW_NEAR_MASK:
                        cv2.imshow("near mask (white = depth overlay)",
                                   cv2.resize(om_filled.astype(np.uint8) * 255,
                                              (pano_w // 2, pano_h // 2)))

            if writer is not None:
                writer.write(pano)
            if SHOW_WINDOW:
                cv2.imshow("360 panorama (q to quit)",
                           cv2.resize(pano, (pano_w // 2, pano_h // 2)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frames += 1
            if frames % 30 == 0:
                now = time.time()
                print(f"\r{30 / (now - t_fps):.1f} fps", end="")
                t_fps = now
    finally:
        for c in cams:
            c.stop()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
