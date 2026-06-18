#!/usr/bin/env python3
"""Depth-viewer-style prototype: fuse 4 ZED colored point clouds, render on GPU.

Unlike the cylinder pipeline, this does NO stitching/scatter/CPU-base. Each frame:
  - grab each camera's XYZ point cloud + LEFT color
  - transform points into the shared rig frame (P @ R.T + t)
  - merge all 4 and hand them to Open3D, which draws them on the GPU (OpenGL)

Holes (no depth: glass, far, low-texture) are just empty space, like the ZED
Depth Viewer. Orbit with the mouse. This avoids the seconds-per-frame Python
scatter; the only per-frame CPU work is a matmul on the (subsampled) points.

Install (rig):  pip install open3d        # aarch64 wheel exists for JetPack 6
Run:            python3 fused_pointcloud_viewer.py
"""
import threading
import time

import cv2
import numpy as np
import pyzed.sl as sl

try:
    import open3d as o3d
except ImportError:
    raise SystemExit("need Open3D:  pip install open3d")

# ----------------------------------------------------------------------------
CAMERAS = [
    {"serial": 46108623, "yaw":   0.0, "t": (0.0,    0.0,  0.711)},
    {"serial": 47860268, "yaw":  90.0, "t": (0.660,  0.0, -0.216)},
    {"serial": 49004271, "yaw": 180.0, "t": (0.0,    0.0, -1.422)},
    {"serial": 43765493, "yaw": -90.0, "t": (-0.660, 0.0, -0.216)},
]
RESOLUTION = sl.RESOLUTION.SVGA
FPS = 15
STRIDE = 2          # subsample the cloud (every Nth px). higher = faster, sparser
MAX_RANGE = 15.0    # meters; drop farther points (noisy + clutters the view)


def _ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])


class ZedThread(threading.Thread):
    def __init__(self, cfg):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.zed = sl.Camera()
        self.xyz = self.bgr = None
        self.lock = threading.Lock()
        self.running = False

    def open(self):
        init = sl.InitParameters()
        init.set_from_serial_number(self.cfg["serial"])
        init.camera_resolution = RESOLUTION
        init.camera_fps = FPS
        init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        init.coordinate_units = sl.UNIT.METER
        if self.zed.open(init) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"could not open {self.cfg['serial']}")

    def run(self):
        self.running = True
        rt = sl.RuntimeParameters()
        m, x = sl.Mat(), sl.Mat()
        while self.running:
            if self.zed.grab(rt) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(m, sl.VIEW.LEFT)
                self.zed.retrieve_measure(x, sl.MEASURE.XYZ)
                bgr = cv2.cvtColor(m.get_data(), cv2.COLOR_BGRA2BGR)
                xyz = x.get_data()[:, :, :3].copy()
                with self.lock:
                    self.bgr, self.xyz = bgr, xyz

    def read(self):
        with self.lock:
            return (None, None) if self.xyz is None else (self.xyz, self.bgr)

    def stop(self):
        self.running = False
        time.sleep(0.1)
        self.zed.close()


def main():
    cams = [ZedThread(c) for c in CAMERAS]
    for c in cams:
        c.open(); c.start()
    # cam->rig rotation (yaw only) and translation, as numpy
    Rs = [_ry(np.radians(c.cfg["yaw"])) for c in cams]
    ts = [np.asarray(c.cfg["t"]) for c in cams]

    print("waiting for first frames...")
    t0 = time.time()
    while any(c.read()[0] is None for c in cams):
        if time.time() - t0 > 15:
            raise RuntimeError("timed out waiting for frames")
        time.sleep(0.05)

    vis = o3d.visualization.Visualizer()
    vis.create_window("Fused ZED point cloud (orbit w/ mouse, q to quit)", 1280, 720)
    pcd = o3d.geometry.PointCloud()
    added = False

    frames, t_fps = 0, time.time()
    try:
        while True:
            pts_all, col_all = [], []
            for c, R, t in zip(cams, Rs, ts):
                xyz, bgr = c.read()
                if xyz is None:
                    continue
                xyz_s = xyz[::STRIDE, ::STRIDE].reshape(-1, 3)
                bgr_s = bgr[::STRIDE, ::STRIDE].reshape(-1, 3)
                ok = np.isfinite(xyz_s).all(axis=1) & (
                    np.linalg.norm(xyz_s, axis=1) < MAX_RANGE)
                P = xyz_s[ok] @ R.T + t                       # cam -> rig frame
                col = bgr_s[ok][:, ::-1].astype(np.float64) / 255.0   # BGR->RGB 0..1
                pts_all.append(P); col_all.append(col)

            if not pts_all:
                continue
            pcd.points = o3d.utility.Vector3dVector(np.concatenate(pts_all))
            pcd.colors = o3d.utility.Vector3dVector(np.concatenate(col_all))
            if not added:
                vis.add_geometry(pcd); added = True   # add once (sets the camera)
            else:
                vis.update_geometry(pcd)
            if not vis.poll_events():                 # window closed / q
                break
            vis.update_renderer()

            frames += 1
            if frames % 30 == 0:
                now = time.time()
                print(f"\r{30 / (now - t_fps):.1f} fps", end="")
                t_fps = now
    finally:
        for c in cams:
            c.stop()
        vis.destroy_window()


if __name__ == "__main__":
    main()
