"""Pure-NumPy reprojection math for depth-corrected 360 panorama.

No cv2 / pyzed / I/O — everything here is unit-testable on any machine.
Rig frame: X-right, Y-down, Z-forward; origin = shared tape-measure point.
"""
import numpy as np


def cam_points_to_rig(P_cam, R_cam2rig, t_cam):
    """Transform camera-frame points into the common rig frame.

    P_cam: (N, 3) or (3,) points in camera frame (meters, X-right Y-down Z-fwd).
    R_cam2rig: (3, 3) rotation. t_cam: (3,) translation (meters).
    Returns (N, 3) points in the rig frame (a single (3,) input returns (1, 3)).
    """
    P_cam = np.atleast_2d(np.asarray(P_cam, dtype=np.float64))
    return P_cam @ np.asarray(R_cam2rig).T + np.asarray(t_cam)


def rig_to_cylinder(P_rig, scale, pano_w, pano_h):
    """Project rig-frame points onto the cylinder (same parametrization as the
    base remap: ray = [sin az, h, cos az]).

    Returns (gx, gy, rng, valid), each shape (N,):
      gx, gy : int pano pixel (gx wraps mod pano_w)
      rng    : float distance from rig origin (z-buffer key); +inf for invalid
               points so they can never be classified as "near"
      valid  : bool — finite, well-defined direction, gy in canvas
    """
    P = np.asarray(P_rig, dtype=np.float64)
    X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
    horiz = np.hypot(X, Z)
    well_defined = horiz > 1e-9
    rng = np.sqrt(X * X + Y * Y + Z * Z)
    az = np.arctan2(X, Z)
    h = np.divide(Y, horiz, out=np.zeros_like(Y), where=well_defined)
    gx = np.mod(np.round(az * scale + pano_w / 2).astype(np.int64), pano_w)
    gy = np.round(h * scale + pano_h / 2).astype(np.int64)
    valid = well_defined & np.isfinite(rng) & (gy >= 0) & (gy < pano_h)
    rng = np.where(valid, rng, np.inf)               # invalid -> inf (never "near")
    return gx, gy, rng, valid


def split_near_far(rng, valid, near_max):
    """Partition points for the two render paths.

    near = valid AND within near_max (depth-reprojected overlay).
    far  = NOT near, which deliberately includes invalid points as well as
           distant ones. `far` is a convenience complement; the base/far render
           path does not consume this mask directly. Returns (near, far)."""
    valid = np.asarray(valid)
    near = valid & (np.asarray(rng) <= near_max)
    far = ~near
    return near, far


def scatter_zbuffer(gx, gy, rng, color, pano_w, pano_h):
    """Forward-scatter colored points onto the pano, keeping the nearest per pixel.

    gx, gy, rng: (N,). color: (N, 3) uint8. Returns:
      overlay_color: (H, W, 3) uint8
      overlay_zbuf : (H, W) float32, +inf where empty
      overlay_mask : (H, W) bool, True where any point landed

    Precondition: all gx in [0, pano_w) and gy in [0, pano_h). Filter points by
    the `valid`/`near` masks from rig_to_cylinder/split_near_far before calling;
    out-of-range coords raise ValueError (they would otherwise index the flat
    z-buffer out of bounds — negative indices silently wrap and corrupt output).
    """
    gx = np.asarray(gx, dtype=np.int64)
    gy = np.asarray(gy, dtype=np.int64)
    rng = np.asarray(rng, dtype=np.float32)
    color = np.asarray(color, dtype=np.uint8)

    if gx.size and not (gx.min() >= 0 and gx.max() < pano_w
                        and gy.min() >= 0 and gy.max() < pano_h):
        raise ValueError(
            "gx/gy out of range; filter points by `valid` before scatter_zbuffer")

    flat = gy * pano_w + gx                          # int64: safe even for huge panos
    zbuf = np.full(pano_h * pano_w, np.inf, dtype=np.float32)
    np.minimum.at(zbuf, flat, rng)                   # nearest range per pixel

    # Write color so the NEAREST point wins each pixel: process farthest-first so
    # the closest point makes the last fancy-index write at its pixel (otherwise,
    # among points within tolerance the last in array order would win, not the
    # nearest).
    order = np.argsort(rng, kind="stable")[::-1]
    flat_s, rng_s, color_s = flat[order], rng[order], color[order]
    win = rng_s <= zbuf[flat_s] + 1e-6
    out = np.zeros((pano_h * pano_w, 3), dtype=np.uint8)
    out[flat_s[win]] = color_s[win]

    mask = np.isfinite(zbuf)
    return (out.reshape(pano_h, pano_w, 3),
            zbuf.reshape(pano_h, pano_w),
            mask.reshape(pano_h, pano_w))


def composite(base, overlay_color, overlay_mask):
    """Return base with overlay_color written where overlay_mask is True.
    Hard replace (edge feathering is applied at the integration layer with cv2).
    Does not mutate base."""
    out = np.array(base, dtype=np.uint8, copy=True)
    m = np.asarray(overlay_mask, dtype=bool)
    out[m] = np.asarray(overlay_color, dtype=np.uint8)[m]
    return out
