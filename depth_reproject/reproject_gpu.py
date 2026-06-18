"""GPU (PyTorch/CUDA) depth-reprojection math — the fast port of reproject.py.

Same math, same rig frame (X-right, Y-down, Z-forward), but on torch tensors so
the heavy per-frame work runs on the Jetson Orin GPU instead of the CPU. The CPU
NumPy version (reproject.py) topped out ~1 fps because ~2.3M points/frame ran on
the ARM cores; this version runs the same arithmetic across the GPU's thousands
of cores.

CPU -> GPU correspondence (read reproject.py side by side):
  np.ndarray            -> torch.Tensor on `device` (cuda)
  np.minimum.at(...)    -> a packed-key scatter_reduce_('amin')  (the z-buffer)
  np.argsort + gather   -> folded INTO that one scatter via the packed key
Every function operates on whatever device its input tensors live on, so the
caller controls placement. Goal: ZED point cloud goes to the GPU once, the final
image comes back once — no per-op CPU<->GPU round trips.
"""
import torch


# ---------------------------------------------------------------------------
# Step 1 — move one camera's 3D points into the shared rig frame
#   (identical to reproject.cam_points_to_rig, but torch instead of numpy)
# ---------------------------------------------------------------------------
def cam_points_to_rig(P_cam, R_cam2rig, t_cam):
    """P_cam: (N,3) points in the camera frame (meters). R_cam2rig: (3,3).
    t_cam: (3,). Returns (N,3) points in the rig frame. P_rig = P_cam @ R.T + t.
    `@` rotates into the rig's orientation; `+ t` shifts by the camera's real
    position (the step that makes the four cameras agree -> kills parallax)."""
    return P_cam @ R_cam2rig.T + t_cam


# ---------------------------------------------------------------------------
# Step 2 — project rig points onto the cylinder -> pano pixels + range
#   (mirror of reproject.rig_to_cylinder)
# ---------------------------------------------------------------------------
def rig_to_cylinder(P_rig, scale, pano_w, pano_h):
    """Returns (gx, gy, rng, valid), each (N,):
      gx, gy : long pano pixel (gx wraps mod pano_w)
      rng    : distance from rig origin (z-buffer key); +inf for invalid points
      valid  : finite, well-defined direction, gy inside the canvas."""
    X, Y, Z = P_rig[:, 0], P_rig[:, 1], P_rig[:, 2]
    horiz = torch.hypot(X, Z)                      # horizontal dist from the axis
    well_defined = horiz > 1e-9
    rng = torch.sqrt(X * X + Y * Y + Z * Z)
    az = torch.atan2(X, Z)                         # azimuth  -> gx
    h = torch.where(well_defined, Y / horiz, torch.zeros_like(Y))   # height -> gy
    gx = torch.remainder(torch.round(az * scale + pano_w / 2).long(), pano_w)
    gy = torch.round(h * scale + pano_h / 2).long()
    valid = well_defined & torch.isfinite(rng) & (gy >= 0) & (gy < pano_h)
    rng = torch.where(valid, rng, torch.full_like(rng, float("inf")))
    return gx, gy, rng, valid


# ---------------------------------------------------------------------------
# Step 3 — keep only the NEAR points (mirror of reproject.split_near_far)
# ---------------------------------------------------------------------------
def split_near_far(rng, valid, near_max):
    """near = valid AND within near_max. Returns (near, far) bool tensors."""
    near = valid & (rng <= near_max)
    return near, ~near


# ---------------------------------------------------------------------------
# Step 4 — forward-scatter with a z-buffer (THE core; mirror of scatter_zbuffer)
#
# CPU version did: np.minimum.at(zbuf, flat, rng)  to keep nearest DEPTH, then a
# separate argsort pass to recover the winning point's COLOR. On the GPU we do
# both at once with the classic "packed key" trick:
#   pack each point as a single int64 = (quantized_depth << 32) | point_index
#   scatter_reduce_('amin') keeps the SMALLEST key per pixel
#   -> smallest key = smallest depth (high bits dominate) = nearest point,
#      and its low 32 bits ARE that point's index, so we recover the color by a
#      gather. One atomic-min pass, color included. (This is how production GPU
#      point rasterizers do z-buffering.)
# ---------------------------------------------------------------------------
def scatter_zbuffer(gx, gy, rng, color, pano_w, pano_h):
    """gx, gy: (N,) long. rng: (N,) float. color: (N,3). Returns:
      overlay_color (H,W,3), overlay_zbuf (H,W) float, overlay_mask (H,W) bool.
    Precondition: gx in [0,pano_w), gy in [0,pano_h) (filter by `valid`/`near`).

    ponytail: FLOAT scatter_reduce amin (fast CUDA path) for the depth test, then
    a masked color write. The int64 packed-key version was ~100x slower because
    integer amin has no good CUDA kernel. Tie-break (points within 1e-4 m at one
    pixel) = last writer wins; fine, they're the same surface. Upgrade to a
    sort-based exact-nearest only if ghosting at equal depths ever shows.
    """
    device = gx.device
    P = pano_h * pano_w
    flat = (gy * pano_w + gx).long()

    zbuf = torch.full((P,), float("inf"), device=device, dtype=torch.float32)
    zbuf.scatter_reduce_(0, flat, rng.float(), reduce="amin", include_self=True)

    win = rng <= zbuf[flat] + 1e-4                 # points that hit their pixel's min
    out = torch.zeros((P, 3), dtype=color.dtype, device=device)
    out[flat[win]] = color[win]                    # write winners (ties: last wins)

    mask = torch.isfinite(zbuf)
    return (out.view(pano_h, pano_w, 3),
            zbuf.view(pano_h, pano_w),
            mask.view(pano_h, pano_w))


# ---------------------------------------------------------------------------
# Step 5 — lay the overlay on top of the base (mirror of reproject.composite)
# ---------------------------------------------------------------------------
def composite(base, overlay_color, overlay_mask):
    """Return base with overlay_color written where overlay_mask is True.
    Does not mutate base."""
    out = base.clone()
    out[overlay_mask] = overlay_color[overlay_mask]
    return out


# ---------------------------------------------------------------------------
# Convenience — pool ALL cameras' near points and run ONE global z-buffer.
# Pooling first (then a single scatter) is what makes the cameras agree on a
# shared surface: the nearest point across ALL cameras wins each pixel.
# ---------------------------------------------------------------------------
def build_overlay(points_list, colors_list, Rs, ts, scale, pano_w, pano_h, near_max):
    """points_list[i]: (Ni,3) finite camera-frame points (tensor on `device`).
    colors_list[i]: (Ni,3). Rs[i]/ts[i]: (3,3)/(3,) on the same device.
    Returns (overlay_color, overlay_zbuf, overlay_mask)."""
    gxs, gys, rngs, cols = [], [], [], []
    for P_cam, color, R, t in zip(points_list, colors_list, Rs, ts):
        P_rig = cam_points_to_rig(P_cam, R, t)
        gx, gy, rng, valid = rig_to_cylinder(P_rig, scale, pano_w, pano_h)
        near, _ = split_near_far(rng, valid, near_max)
        gxs.append(gx[near]); gys.append(gy[near])
        rngs.append(rng[near]); cols.append(color[near])
    gx = torch.cat(gxs); gy = torch.cat(gys)
    rng = torch.cat(rngs); color = torch.cat(cols)
    return scatter_zbuffer(gx, gy, rng, color, pano_w, pano_h)


# ---------------------------------------------------------------------------
# Self-test — runs on CUDA if present, else CPU. Mirrors reproject.py's tests so
# you can confirm the GPU port matches the CPU math.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {dev}  (cuda available: {torch.cuda.is_available()})")
    f64 = torch.float64
    scale, pano_w, pano_h = 350.0, 2200, 512

    # (1) two cameras at different positions see the SAME world point -> same pixel
    P_world = torch.tensor([1.0, 0.0, 3.0], dtype=f64, device=dev)
    Ra = torch.eye(3, dtype=f64, device=dev); ta = torch.zeros(3, dtype=f64, device=dev)
    Rb = torch.eye(3, dtype=f64, device=dev); tb = torch.tensor([0.5, 0.0, 0.0], dtype=f64, device=dev)
    Pa = (P_world - ta) @ Ra
    Pb = (P_world - tb) @ Rb
    a = rig_to_cylinder(cam_points_to_rig(Pa[None], Ra, ta), scale, pano_w, pano_h)
    b = rig_to_cylinder(cam_points_to_rig(Pb[None], Rb, tb), scale, pano_w, pano_h)
    assert int(a[0]) == int(b[0]) and int(a[1]) == int(b[1]), "cameras disagree!"
    print("OK  two cameras agree on a near point's pixel")

    # (2) nearest point wins its pixel (z-buffer)
    gx = torch.tensor([2, 2], device=dev); gy = torch.tensor([1, 1], device=dev)
    rng = torch.tensor([5.0, 2.0], device=dev)            # 2nd is nearer
    color = torch.tensor([[10, 10, 10], [200, 200, 200]], dtype=torch.uint8, device=dev)
    oc, _oz, om = scatter_zbuffer(gx, gy, rng, color, pano_w=4, pano_h=3)
    assert bool(om[1, 2]) and int(oc[1, 2, 0]) == 200, "z-buffer picked wrong point!"
    print("OK  z-buffer keeps the nearest point's color")
    print("all GPU-math self-tests passed")
