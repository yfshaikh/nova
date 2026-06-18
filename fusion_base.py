#!/usr/bin/env python3
"""
4x ZED -> real-time cylindrical 360 panorama (RGB POC, no depth).

Pipeline:  grab(4 threads) -> undistort(ZED rectified) -> warp-to-cylinder
           (static maps) -> feather-blend -> show/save.

Phase 1: parallax handled by a single implicit "scene at infinity" cylinder.
Near objects will ghost at seams; that's expected and gets fixed in Phase 2
(depth-aware reprojection). Everything here is reused by Phase 2 except the
warp step, so none of it is throwaway.
"""

import threading
import time

import cv2
import numpy as np
import pyzed.sl as sl

# ----------------------------------------------------------------------------
# CONFIG  — your rig
# ----------------------------------------------------------------------------
# yaw   = outward heading, degrees. front=0, right=+90, back=180, left=-90.
# pitch = mount tilt up(-)/down(+). roll = camera-axis tilt. Start at 0; nudge
#         if a seam is vertically misaligned or tilted.
#
# If the panorama comes out left-right MIRRORED (right cam on the left), flip
# the SIGN of every yaw below.
CAMERAS = [
    {"name": "front", "serial": 46108623, "yaw":   0.0, "pitch": 0.0, "roll": 0.0},
    {"name": "right", "serial": 47860268, "yaw":  90.0, "pitch": 0.0, "roll": 0.0},
    {"name": "back",  "serial": 49004271, "yaw": 180.0, "pitch": 0.0, "roll": 0.0},
    {"name": "left",  "serial": 43765493, "yaw": -90.0, "pitch": 0.0, "roll": 0.0},
]

RESOLUTION = sl.RESOLUTION.HD1080   # HD720 is the right tradeoff for 4 USB ZEDs
FPS = 15                            # 4 cams share bus bandwidth; 15 is safe

# Photometric harmonization: lock exposure + white balance across ALL cams so
# seams don't show a brightness/color jump. Tune EXPOSURE/WB to your scene.
LOCK_EXPOSURE = False
EXPOSURE_PCT = 50      # 0..100
WB_KELVIN = 4600       # 2800 (warm) .. 6500 (cool)

SAVE_VIDEO = False
OUTPUT_PATH = "panorama.mp4"
SHOW_WINDOW = True     # set False if running headless

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
    Returns (map_x, map_y, valid_mask) — all static, computed once."""
    R_wc = R_cw.T  # world->camera
    gx, gy = np.meshgrid(np.arange(pano_w), np.arange(pano_h))
    phi = (gx - pano_w / 2.0) / scale          # azimuth, -pi..pi
    hh = (gy - pano_h / 2.0) / scale           # cylinder height
    # Ray on the unit cylinder, in world coords:
    rays = np.stack([np.sin(phi), hh, np.cos(phi)], axis=-1)   # (H,W,3)
    rays_cam = rays.reshape(-1, 3) @ R_wc.T
    rays_cam = rays_cam.reshape(pano_h, pano_w, 3)
    xc, yc, zc = rays_cam[..., 0], rays_cam[..., 1], rays_cam[..., 2]

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
        self.frame = None              # latest BGR frame
        self.lock = threading.Lock()
        self.running = False
        self.fx = self.fy = self.cx = self.cy = None
        self.img_w = self.img_h = None

    def open(self):
        init = sl.InitParameters()
        init.set_from_serial_number(self.cfg["serial"])
        init.camera_resolution = RESOLUTION
        init.camera_fps = FPS
        init.depth_mode = sl.DEPTH_MODE.NONE   # RGB only -> light on the GPU
        if self.zed.open(init) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"could not open {self.cfg['name']} "
                               f"(serial {self.cfg['serial']})")

        if LOCK_EXPOSURE:
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, EXPOSURE_PCT)
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 0)
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, WB_KELVIN)

        # ZED SDK 4.x calibration path (rectified LEFT view = ideal pinhole).
        info = self.zed.get_camera_information()
        calib = info.camera_configuration.calibration_parameters.left_cam
        res = info.camera_configuration.resolution
        self.fx, self.fy, self.cx, self.cy = calib.fx, calib.fy, calib.cx, calib.cy
        self.img_w, self.img_h = res.width, res.height

    def run(self):
        self.running = True
        rt = sl.RuntimeParameters()
        mat = sl.Mat()
        while self.running:
            if self.zed.grab(rt) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(mat, sl.VIEW.LEFT)   # rectified
                bgr = cv2.cvtColor(mat.get_data(), cv2.COLOR_BGRA2BGR)
                with self.lock:
                    self.frame = bgr

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame

    def stop(self):
        self.running = False
        time.sleep(0.1)
        self.zed.close()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
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
    scale = 350                                 # panorama pixels/radian
    pano_w = int(round(2 * np.pi * scale))
    pano_h = int(round(scale * ref.img_h / ref.fy))
    print(f"panorama: {pano_w} x {pano_h}")

    maps, weights = [], []
    for c in cams:
        R_cw = cam_to_world(c.cfg["yaw"], c.cfg["pitch"], c.cfg["roll"])
        mx, my, valid = build_cylindrical_maps(
            c.fx, c.fy, c.cx, c.cy, c.img_w, c.img_h, R_cw, scale, pano_w, pano_h)
        maps.append((mx, my))
        # distance-to-edge -> smooth feather that cross-fades in overlaps
        weights.append(cv2.distanceTransform(valid, cv2.DIST_L2, 5))

    total = np.sum(weights, axis=0)
    nz = total > 0
    norm_w = []
    for w in weights:
        nw = np.zeros_like(w)
        nw[nz] = w[nz] / total[nz]
        norm_w.append(nw[..., None].astype(np.float32))   # (H,W,1)

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
            for c, (mx, my), nw in zip(cams, maps, norm_w):
                img = c.read()
                if img is None:
                    continue
                warped = cv2.remap(img, mx, my, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                pano += warped.astype(np.float32) * nw
            pano = np.clip(pano, 0, 255).astype(np.uint8)

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
