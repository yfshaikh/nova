#!/usr/bin/env python3
"""GPU-accelerated 4x ZED 360 panorama (depth reprojection on the Jetson GPU).

This is the GPU version of zed_360_panorama.py. Two changes vs the CPU file:

  1. SPEED: the per-frame depth reprojection (transform millions of points ->
     project -> z-buffer) runs on the GPU via reproject_gpu.py (torch/CUDA),
     instead of NumPy on the ARM CPU. That was the ~1 fps bottleneck.

  2. HOLES: the rotation-only BASE is kept FULL (we do NOT drop near pixels from
     it). The depth overlay is composited ON TOP. So any pixel the overlay
     misses (depth holes, dead zones) shows real background instead of black.
     (The CPU file emptied the base under near objects -> that caused the black
     holes indoors.)

Pipeline per frame:
  base  (CPU): 4x cv2.remap of the LEFT images -> feather-blended 360 background
  overlay (GPU): push each camera's XYZ+color to the GPU, transform to the rig
                 frame, project to the cylinder, one global z-buffer (nearest
                 wins across all cameras) -> the depth-corrected near field
  final: composite overlay over base

Run the GPU-math self-test first to confirm your torch build:
  python3 reproject_gpu.py        # expects "all GPU-math self-tests passed"
Then:
  python3 zed_360_panorama_gpu.py
"""
import os
import threading
import time

import cv2
import numpy as np
import pyzed.sl as sl
import torch

import reproject_gpu as rpg          # torch/CUDA depth-reprojection math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------------------------
# CONFIG (same rig as the CPU file)
# ----------------------------------------------------------------------------
CAMERAS = [
    {"name": "front", "serial": 46108623, "yaw":   0.0, "pitch": 0.0, "roll": 0.0,
     "t": (0.0,    0.0,  0.711)},
    {"name": "right", "serial": 47860268, "yaw":  90.0, "pitch": 0.0, "roll": 0.0,
     "t": (0.660,  0.0, -0.216)},
    {"name": "back",  "serial": 49004271, "yaw": 180.0, "pitch": 0.0, "roll": 0.0,
     "t": (0.0,    0.0, -1.422)},
    {"name": "left",  "serial": 43765493, "yaw": -90.0, "pitch": 0.0, "roll": 0.0,
     "t": (-0.660, 0.0, -0.216)},
]

DEPTH_REPROJECT = True              # this file's whole point; False = base only
DEPTH_MODE = sl.DEPTH_MODE.PERFORMANCE
NEAR_MAX = 8.0                      # meters; closer than this gets reprojected
RESOLUTION = sl.RESOLUTION.SVGA     # lighter than HD1080 for depth on the Orin
FPS = 15
SCALE = 240                         # pano pixels/radian. Lower = smaller pano =
                                    # less CPU base-remap + smaller GPU buffers.
                                    # ponytail: 240 ~halves pixel work vs 350.
POINT_STRIDE = 2                    # subsample the cloud (every Nth px) -> N^2
                                    # fewer points to transfer + scatter.
                                    # ponytail: 2 = 4x fewer points, minor near-field loss.
SHOW_NEAR_MASK = False
SHOW_WINDOW = True
TIMING = True                       # ponytail: print per-stage ms/frame to find the bottleneck; turn off once fixed


# ----------------------------------------------------------------------------
# Geometry helpers (identical to the CPU file)
# ----------------------------------------------------------------------------
def _rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def _ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def _rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def cam_to_world(yaw_deg, pitch_deg, roll_deg):
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    return _ry(y) @ _rx(p) @ _rz(r)


def build_cylindrical_maps(fx, fy, cx, cy, img_w, img_h, R_cw, scale, pano_w, pano_h):
    """Static inverse-warp table for the rotation-only base (see the CPU file)."""
    R_wc = R_cw.T
    gx, gy = np.meshgrid(np.arange(pano_w), np.arange(pano_h))
    phi = (gx - pano_w / 2.0) / scale
    hh = (gy - pano_h / 2.0) / scale
    rays = np.stack([np.sin(phi), hh, np.cos(phi)], axis=-1)
    rays_cam = (rays.reshape(-1, 3) @ R_wc.T).reshape(pano_h, pano_w, 3)
    xc, yc, zc = rays_cam[..., 0], rays_cam[..., 1], rays_cam[..., 2]
    in_front = zc > 1e-6
    zc_safe = np.where(in_front, zc, 1.0)
    map_x = (fx * xc / zc_safe + cx).astype(np.float32)
    map_y = (fy * yc / zc_safe + cy).astype(np.float32)
    valid = (in_front & (map_x >= 0) & (map_x <= img_w-1)
                      & (map_y >= 0) & (map_y <= img_h-1))
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0
    return map_x, map_y, valid.astype(np.uint8) * 255


# ----------------------------------------------------------------------------
# Threaded ZED capture (always retrieves XYZ — this is the depth pipeline)
# ----------------------------------------------------------------------------
class ZedThread(threading.Thread):
    def __init__(self, cfg):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.zed = sl.Camera()
        self.frame = None
        self.xyz = None
        self.lock = threading.Lock()
        self.running = False
        self.fx = self.fy = self.cx = self.cy = None
        self.img_w = self.img_h = None

    def open(self):
        init = sl.InitParameters()
        init.set_from_serial_number(self.cfg["serial"])
        init.camera_resolution = RESOLUTION
        init.camera_fps = FPS
        init.depth_mode = DEPTH_MODE if DEPTH_REPROJECT else sl.DEPTH_MODE.NONE
        init.coordinate_units = sl.UNIT.METER
        if self.zed.open(init) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"could not open {self.cfg['name']} ({self.cfg['serial']})")
        info = self.zed.get_camera_information()
        calib = info.camera_configuration.calibration_parameters.left_cam
        res = info.camera_configuration.resolution
        self.fx, self.fy, self.cx, self.cy = calib.fx, calib.fy, calib.cx, calib.cy
        self.img_w, self.img_h = res.width, res.height

    def run(self):
        self.running = True
        rt = sl.RuntimeParameters()
        mat, xyz_mat = sl.Mat(), sl.Mat()
        while self.running:
            if self.zed.grab(rt) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(mat, sl.VIEW.LEFT)
                bgr = cv2.cvtColor(mat.get_data(), cv2.COLOR_BGRA2BGR)
                xyz = None
                if DEPTH_REPROJECT:
                    self.zed.retrieve_measure(xyz_mat, sl.MEASURE.XYZ)
                    xyz = xyz_mat.get_data()[:, :, :3].copy()
                with self.lock:
                    self.frame, self.xyz = bgr, xyz

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
    print("=" * 70)
    print(f"RUNNING FILE : {os.path.abspath(__file__)}")
    print(f"DEVICE = {DEVICE}   DEPTH_REPROJECT = {DEPTH_REPROJECT}   NEAR_MAX = {NEAR_MAX}")
    print("=" * 70)

    cams = [ZedThread(c) for c in CAMERAS]
    for c in cams:
        c.open(); c.start()

    print("waiting for first frames...")
    t0 = time.time()
    while any(c.read() is None for c in cams):
        if time.time() - t0 > 15:
            raise RuntimeError("timed out waiting for camera frames")
        time.sleep(0.05)

    # --- static base maps + feather weights (CPU, once) --------------------
    ref = cams[0]
    scale = SCALE
    pano_w = int(round(2 * np.pi * scale))
    pano_h = int(round(scale * ref.img_h / ref.fy))
    print(f"panorama: {pano_w} x {pano_h}")

    maps, weights = [], []
    for c in cams:
        R_cw = cam_to_world(c.cfg["yaw"], c.cfg["pitch"], c.cfg["roll"])
        mx, my, valid = build_cylindrical_maps(
            c.fx, c.fy, c.cx, c.cy, c.img_w, c.img_h, R_cw, scale, pano_w, pano_h)
        maps.append((mx, my))
        weights.append(cv2.distanceTransform(valid, cv2.DIST_L2, 5))
    total = np.sum(weights, axis=0)
    nz = total > 0
    norm_w = []
    for w in weights:
        nw = np.zeros_like(w); nw[nz] = w[nz] / total[nz]
        norm_w.append(nw[..., None].astype(np.float32))

    # --- per-camera extrinsics as torch tensors on the GPU (once) ----------
    Rs_t = [torch.tensor(cam_to_world(c.cfg["yaw"], c.cfg["pitch"], c.cfg["roll"]),
                         dtype=torch.float32, device=DEVICE) for c in cams]
    ts_t = [torch.tensor(c.cfg["t"], dtype=torch.float32, device=DEVICE) for c in cams]

    print("running. press 'q' to quit.")
    frames, t_fps = 0, time.time()
    try:
        while True:
            # ---- BASE layer (CPU): full rotation-only background -----------
            t0 = time.perf_counter()
            pano = np.zeros((pano_h, pano_w, 3), np.float32)
            for c, (mx, my), nw in zip(cams, maps, norm_w):
                img = c.read()
                if img is None:
                    continue
                warped = cv2.remap(img, mx, my, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                pano += warped.astype(np.float32) * nw     # FULL base (no near-drop)
            pano = np.clip(pano, 0, 255).astype(np.uint8)
            t_base = time.perf_counter()

            # ---- OVERLAY layer (GPU): depth reprojection -------------------
            if DEPTH_REPROJECT:
                pts, cols, Rs_u, ts_u = [], [], [], []
                for c, R_t, t_t in zip(cams, Rs_t, ts_t):
                    img, xyz = c.read(), c.read_xyz()
                    if img is None or xyz is None:
                        continue
                    # subsample, then numpy -> torch on the GPU (one transfer/cam)
                    xyz_s = xyz[::POINT_STRIDE, ::POINT_STRIDE]
                    img_s = img[::POINT_STRIDE, ::POINT_STRIDE]
                    P = torch.from_numpy(np.ascontiguousarray(
                        xyz_s.reshape(-1, 3))).to(DEVICE)               # (N,3) float32
                    col = torch.from_numpy(np.ascontiguousarray(
                        img_s.reshape(-1, 3))).to(DEVICE)               # (N,3) uint8
                    finite = torch.isfinite(P).all(dim=1)              # drop NaN/Inf depth
                    pts.append(P[finite]); cols.append(col[finite])
                    Rs_u.append(R_t); ts_u.append(t_t)

                if pts:
                    oc, _oz, om = rpg.build_overlay(
                        pts, cols, Rs_u, ts_u, scale, pano_w, pano_h, NEAR_MAX)
                    if DEVICE == "cuda":
                        torch.cuda.synchronize()       # honest GPU timing
                    t_gpu = time.perf_counter()
                    # bring the overlay down once; splat + seal pinholes on CPU
                    oc_np = oc.cpu().numpy()
                    om_np = om.cpu().numpy()
                    k = np.ones((3, 3), np.uint8)
                    om_filled = cv2.morphologyEx(
                        (om_np * 255).astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
                    oc_np = cv2.dilate(oc_np, k)
                    pano[om_filled] = oc_np[om_filled]                 # composite on base
                    if TIMING:
                        t_end = time.perf_counter()
                        print(f"base {(t_base-t0)*1e3:5.0f}  transfer+gpu "
                              f"{(t_gpu-t_base)*1e3:5.0f}  back+morph "
                              f"{(t_end-t_gpu)*1e3:5.0f}  total {(t_end-t0)*1e3:5.0f} ms")
                    if SHOW_NEAR_MASK:
                        cv2.imshow("near mask", cv2.resize(
                            (om_filled * 255).astype(np.uint8), (pano_w // 2, pano_h // 2)))

            if SHOW_WINDOW:
                cv2.imshow("360 panorama GPU (q to quit)",
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
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
