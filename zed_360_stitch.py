"""
zed_360_stitch.py — Real-time 360° panorama from 4 outward-facing ZED cameras.

================================================================================
WHAT THIS DOES
================================================================================
Four ZED stereo cameras are mounted on top of a car, facing outward, roughly 90°
apart (front / right / left / back). This script stitches their live RGB feeds
into a single horizontal 360° panorama in real time.

We do NOT use:
  - cv2.Stitcher (feature matching; assumes one nodal point — fails on a wide-
    baseline rig with dead zones and parallax)
  - sl.Fusion (only fuses metadata: skeletons/objects/tracking — never produces
    a stitched RGB image)
  - ZED360 (needs overlapping views to solve extrinsics; this rig has dead zones)

We DO use a custom **rotation-only cylindrical projection**:
  - OpenCV is used only for cv2.remap + cv2.imshow (NOT the Stitcher).
  - pyzed is used only for capture (NOT Fusion).

================================================================================
HOW IT WORKS
================================================================================
Imagine a virtual cylinder around the rig. The output panorama is that cylinder
unrolled into a flat strip:
    - horizontal axis u  -> azimuth angle theta (which compass direction)
    - vertical axis   v  -> elevation angle phi (how far up/down)

For every output pixel (u, v) we:
  1. Convert (u, v) into a 3D ray direction (theta, phi) in WORLD space.
  2. Rotate that ray into a given camera's frame using only that camera's YAW.
  3. Project the rotated ray through the camera's pinhole intrinsics (fx, fy,
     cx, cy) to get a source pixel (x_src, y_src) in that camera's raw frame.
  4. Record (x_src, y_src) in a remap table, plus a feathered blend weight that
     fades toward the edges of each camera's coverage.

Steps 1-4 depend only on fixed geometry, so build_remap() runs ONCE per camera.
At runtime every frame is just: grab -> cv2.remap (cheap, table lookup) ->
weighted blend. That's what makes it real-time.

================================================================================
COORDINATE CONVENTIONS  (get these wrong and the image flips / mirrors)
================================================================================
World frame (ZED default): right-handed, +X right, +Y up, +Z forward.
Image pixels:              +x right, +y DOWN  (so world Y-up maps to image -y).
yaw:                       rotation about the vertical (Y) axis. 0 = front,
                           +90° = right, -90° = left, 180° = back.

================================================================================
THE BIG ASSUMPTION (and its one real limitation)
================================================================================
This model assumes all four cameras share a single optical center and differ
only by a yaw rotation. They do NOT — they sit ~70 cm apart on the roof. We
ignore those translations on purpose (see question 4 in the chat / README).

Consequence: for DISTANT scenery the shared-center approximation is excellent.
For CLOSE objects (e.g. a car parked right next to the rig) each camera sees the
object from a genuinely different position — that's PARALLAX. The cylinder can't
place a close object consistently for two cameras at once, so in the overlap
region you get ghosting / doubling / a "half-built" look. No amount of yaw
tuning fixes this; it would require per-pixel DEPTH (which the ZEDs can provide)
and a full 3D reprojection. This script is the depth-free version.
================================================================================
"""

import time
import cv2
import numpy as np
import pyzed.sl as sl

# ---- Config -----------------------------------------------------------------

CAMERAS = [
    # (serial, role, yaw_radians)
    (47860268, "right", +np.pi / 2),   # faces +90° (right)
    (46108623, "front",  0.0),         # faces 0°    (front, centered in canvas)
    (43765493, "left",  -np.pi / 2),   # faces -90°  (left)
    (49004271, "back",  +np.pi),       # faces 180°  (back)
]

# Panorama canvas size and field of view.
#   HFOV_TOTAL = 2*pi  -> exactly 360°. Clean, but whichever camera straddles
#                         the left/right seam (theta = ±180°) gets sliced: half
#                         its frame lands at the far-left edge, half at the
#                         far-right edge. With front centered, that victim is
#                         the BACK camera.
#   HFOV_TOTAL = 4*pi  -> 720°. Every direction is drawn twice, so no camera is
#                         ever cut by the seam (good for diagnosing coverage).
# Keep pixels-per-degree constant by scaling PANO_W with HFOV_TOTAL:
#   PANO_W = 3200 for 360°, 6400 for 720°.
PANO_W = 3200
PANO_H = 800
HFOV_TOTAL = 2 * np.pi          # 360° canvas (switch to 4*np.pi for the 720° test)
VFOV_TOTAL = np.pi / 3          # ±60° vertical coverage
CANVAS_YAW_OFFSET = 0.0         # 0 = front centered; np.pi = back centered
FEATHER_PX = 80                 # blend ramp width (px) at each camera's edge

# ---- Debug knobs (set all off for the clean panorama) -----------------------
DEBUG_TINT = False              # color-tint each camera's contribution
ISOLATE_SERIAL = None           # set to one serial to render ONLY that camera
SAVE_BACK_RAW = False           # save back_raw.png after 30 good back frames
SHOW_PER_CAM_WINDOWS = False    # the "8-window diagnostic" — see notes below

DEBUG_COLORS = {                # BGR, used only when DEBUG_TINT is on
    47860268: (0,   0,   255),  # right -> red
    46108623: (0,   255, 0),    # front -> green
    43765493: (255, 0,   0),    # left  -> blue
    49004271: (0,   255, 255),  # back  -> yellow
}


# ---- Per-camera remap + weight mask (built ONCE per camera) -----------------

def build_remap(yaw, fx, fy, cx, cy, src_w, src_h):
    """Precompute, for one camera, the (map_x, map_y, weight) tables that turn
    its raw frame into its slice of the cylindrical panorama.

    map_x[v,u], map_y[v,u] = the source pixel in the raw frame that output
    pixel (u,v) should sample. weight[v,u] = how much this camera contributes
    there (0 = nothing/out of view, 1 = full, ramped near frame edges so
    neighboring cameras cross-fade instead of showing a hard seam).
    """
    # Output pixel grid -> world ray angles.
    u, v = np.meshgrid(np.arange(PANO_W), np.arange(PANO_H))
    theta = (u - PANO_W / 2) * (HFOV_TOTAL / PANO_W) + CANVAS_YAW_OFFSET  # azimuth
    phi   = (v - PANO_H / 2) * (VFOV_TOTAL / PANO_H)                      # elevation

    # World ray direction (Y-up, +Z forward, +X right).
    dx =  np.sin(theta) * np.cos(phi)
    dy = -np.sin(phi)
    dz =  np.cos(theta) * np.cos(phi)

    # Rotate world -> camera by R_y(-yaw). NOTE: rotation only. No translation is
    # applied (we have no depth here, so a translation can't be resolved — see
    # the module docstring). This is the shared-optical-center approximation.
    c, s = np.cos(-yaw), np.sin(-yaw)
    cam_x =  c * dx + s * dz
    cam_y = -dy                  # world Y-up  ->  image Y-down
    cam_z = -s * dx + c * dz

    # Pinhole projection. Only rays in front of the camera (cam_z > 0) are valid.
    valid = cam_z > 0
    x_src = np.where(valid, fx * cam_x / cam_z + cx, -1.0)
    y_src = np.where(valid, fy * cam_y / cam_z + cy, -1.0)
    in_bounds = valid & (x_src >= 0) & (x_src < src_w) & (y_src >= 0) & (y_src < src_h)

    map_x = np.where(in_bounds, x_src, -1).astype(np.float32)
    map_y = np.where(in_bounds, y_src, -1).astype(np.float32)

    # Feathered weight: distance to the nearest frame edge, clipped to [0,1] over
    # FEATHER_PX. Cameras overlap-blend smoothly instead of butting at a seam.
    edge = np.minimum.reduce([x_src, src_w - 1 - x_src, y_src, src_h - 1 - y_src])
    weight = np.clip(edge / FEATHER_PX, 0, 1).astype(np.float32)
    weight = np.where(in_bounds, weight, 0.0)

    return map_x, map_y, weight


# ---- Open cameras + precompute their remap tables ---------------------------

zeds, remaps, roles = {}, {}, {}
for serial, role, yaw in CAMERAS:
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.NONE          # no depth needed for this model
    init.set_from_serial_number(serial)

    cam = sl.Camera()
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        print(f"[!] Failed to open {serial} ({role})")
        continue

    # Pull this camera's actual intrinsics (per-unit; don't hardcode).
    info = cam.get_camera_information()
    cp = info.camera_configuration.calibration_parameters.left_cam
    w  = info.camera_configuration.resolution.width
    h  = info.camera_configuration.resolution.height

    zeds[serial]   = cam
    roles[serial]  = role
    remaps[serial] = build_remap(yaw, cp.fx, cp.fy, cp.cx, cp.cy, w, h)

    # canvas_cols_used = how many output columns this camera lights up; a quick
    # sanity check that its full horizontal FOV made it onto the canvas.
    _, _, weight = remaps[serial]
    cols_used = (weight.sum(axis=0) > 0).sum()
    print(f"[+] {serial} ({role:5s})  yaw={np.degrees(yaw):+4.0f}°  "
          f"fx={cp.fx:6.1f}  fy={cp.fy:6.1f}  cx={cp.cx:6.1f}  cy={cp.cy:6.1f}  "
          f"res={w}x{h}  weight_sum={weight.sum():.0f}  canvas_cols_used={cols_used}")


# ---- Runtime loop -----------------------------------------------------------

runtime = sl.RuntimeParameters()
image   = sl.Mat()
accum   = np.zeros((PANO_H, PANO_W, 3), dtype=np.float32)   # weighted color sum
wsum    = np.zeros((PANO_H, PANO_W),    dtype=np.float32)    # weight sum (denom)

grab_stats = {s: {"ok": 0, "fail": 0} for s in zeds}
last_print = time.time()

while True:
    accum.fill(0); wsum.fill(0)

    for serial, cam in zeds.items():
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            grab_stats[serial]["fail"] += 1
            continue
        grab_stats[serial]["ok"] += 1

        # ISOLATE_SERIAL: render the panorama from a single camera (others skipped).
        if ISOLATE_SERIAL is not None and serial != ISOLATE_SERIAL:
            continue

        cam.retrieve_image(image, sl.VIEW.LEFT)
        frame = image.get_data()[:, :, :3]        # drop alpha -> BGR

        if SAVE_BACK_RAW and serial == 49004271 and grab_stats[serial]["ok"] == 30:
            cv2.imwrite("back_raw.png", frame)
            print("saved back_raw.png")

        # The whole per-frame cost: one table lookup.
        map_x, map_y, weight = remaps[serial]
        warped = cv2.remap(frame, map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # --- 8-window diagnostic ---------------------------------------------
        # For each camera, show its RAW frame and its cylindrical projection
        # (its "pano" contribution) side by side. 4 cams x 2 = 8 windows.
        # Compare per camera to localize any missing content:
        #   raw full  + pano full          -> camera & projection both fine
        #   raw full  + pano broken         -> remap bug (trace projection)
        #   raw already broken              -> camera coverage/occlusion (no SW fix)
        #   object split across two panos   -> parallax (close object; no SW fix)
        if SHOW_PER_CAM_WINDOWS:
            cv2.imshow(f"{roles[serial]} pano",
                       cv2.resize(warped, (PANO_W // 3, PANO_H // 3)))
            cv2.imshow(f"{roles[serial]} raw",
                       cv2.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3)))
        # ---------------------------------------------------------------------

        if DEBUG_TINT:
            tint = np.array(DEBUG_COLORS[serial], dtype=np.float32) / 255.0
            warped = warped.astype(np.float32) * (0.5 + 0.5 * tint)

        accum += warped.astype(np.float32) * weight[..., None]
        wsum  += weight

    # Normalize the weighted sum -> final panorama.
    denom = np.maximum(wsum[..., None], 1e-6)
    pano  = np.clip(accum / denom, 0, 255).astype(np.uint8)
    cv2.imshow("ZED 360", pano)

    if time.time() - last_print > 2.0:
        report = ", ".join(
            f'{roles[s]}={c["ok"]}/{c["ok"]+c["fail"]}'
            for s, c in grab_stats.items()
        )
        print(f"grabs: {report}")
        last_print = time.time()

    if cv2.waitKey(1) & 0xFF == 27:   # Esc to quit
        break

for cam in zeds.values():
    cam.close()
cv2.destroyAllWindows()
