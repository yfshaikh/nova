#!/usr/bin/env python3
"""Fast fused ZED point cloud viewer (no Open3D, no CUDA compute).

Why this beats the Open3D version's 1 fps: it skips Open3D's heavy per-frame
geometry rebuild AND does zero CUDA compute (our torch path was fighting ZED's
depth for the GPU). Per frame: grab 4 clouds, transform on the CPU (subsampled),
upload once to one GL vertex buffer (pyqtgraph), GPU draws. One upload, no CUDA.

This is NOT as fast as the ZED Depth Viewer -- that keeps the cloud GPU-resident
via CUDA-OpenGL interop (zero transfer). If this still isn't enough, that interop
is the only lever left, which means editing Stereolabs' ogl_viewer.viewer.

Install (rig):  pip install pyqtgraph PyQt5
Run:            python3 fused_pc_fast.py     # orbit with mouse, close window to quit
"""
import threading
import time

import cv2
import numpy as np
import pyzed.sl as sl
import pyqtgraph.opengl as gl
from PyQt5 import QtWidgets, QtCore

CAMERAS = [
    {"serial": 46108623, "yaw":   0.0, "t": (0.0,    0.0,  0.711)},
    {"serial": 47860268, "yaw":  90.0, "t": (0.660,  0.0, -0.216)},
    {"serial": 49004271, "yaw": 180.0, "t": (0.0,    0.0, -1.422)},
    {"serial": 43765493, "yaw": -90.0, "t": (-0.660, 0.0, -0.216)},
]
RESOLUTION = sl.RESOLUTION.SVGA
FPS = 15
STRIDE = 3          # subsample (every Nth px). up = faster/sparser
MAX_RANGE = 15.0    # m; drop farther (noisy) points


def _ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])


class ZedThread(threading.Thread):
    def __init__(self, cfg):
        super().__init__(daemon=True)
        self.cfg, self.zed = cfg, sl.Camera()
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
    Rs = [_ry(np.radians(c.cfg["yaw"])) for c in cams]
    ts = [np.asarray(c.cfg["t"], np.float32) for c in cams]

    print("waiting for first frames...")
    t0 = time.time()
    while any(c.read()[0] is None for c in cams):
        if time.time() - t0 > 15:
            raise RuntimeError("timed out waiting for frames")
        time.sleep(0.05)

    app = QtWidgets.QApplication([])
    w = gl.GLViewWidget()
    w.setWindowTitle("Fused ZED cloud (orbit w/ mouse)")
    w.setCameraPosition(distance=8)
    w.show()
    scatter = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), size=2.0, pxMode=True)
    w.addItem(scatter)

    st = {"n": 0, "t": time.time()}

    def update():
        pts, cols = [], []
        for c, R, t in zip(cams, Rs, ts):
            xyz, bgr = c.read()
            if xyz is None:
                continue
            xs = xyz[::STRIDE, ::STRIDE].reshape(-1, 3)
            bs = bgr[::STRIDE, ::STRIDE].reshape(-1, 3)
            ok = np.isfinite(xs).all(axis=1) & (np.linalg.norm(xs, axis=1) < MAX_RANGE)
            P = (xs[ok] @ R.T + t).astype(np.float32)              # cam -> rig
            rgb = (bs[ok][:, ::-1].astype(np.float32)) / 255.0      # BGR -> RGB
            rgba = np.concatenate([rgb, np.ones((len(rgb), 1), np.float32)], axis=1)
            pts.append(P); cols.append(rgba)
        if not pts:
            return
        scatter.setData(pos=np.concatenate(pts), color=np.concatenate(cols))
        st["n"] += 1
        if st["n"] % 30 == 0:
            now = time.time()
            print(f"\r{30 / (now - st['t']):.1f} fps", end="")
            st["t"] = now

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(0)        # as fast as the event loop allows
    try:
        app.exec_()
    finally:
        for c in cams:
            c.stop()


if __name__ == "__main__":
    main()
