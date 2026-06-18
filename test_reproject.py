import numpy as np
import reproject as rp


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def test_cam_points_to_rig_translation_only():
    P_cam = np.array([[0.0, 0.0, 1.0]])          # 1 m straight ahead
    R = np.eye(3)
    t = np.array([1.0, 0.0, 0.0])                # rig sits 1 m to the +X
    out = rp.cam_points_to_rig(P_cam, R, t)
    np.testing.assert_allclose(out, [[1.0, 0.0, 1.0]], atol=1e-9)


def test_cam_points_to_rig_right_camera_yaw90():
    # Right camera (yaw +90): a point straight ahead in the camera maps to +X in rig.
    P_cam = np.array([[0.0, 0.0, 2.0]])
    R = _ry(np.radians(90))
    t = np.array([0.66, 0.0, -0.216])
    out = rp.cam_points_to_rig(P_cam, R, t)
    np.testing.assert_allclose(out, [[0.66 + 2.0, 0.0, -0.216]], atol=1e-6)


def test_rig_to_cylinder_forward_center():
    # Point straight ahead lands at canvas center; h=0; range = Z.
    P = np.array([[0.0, 0.0, 5.0]])
    gx, gy, rng, valid = rp.rig_to_cylinder(P, scale=350.0, pano_w=2200, pano_h=512)
    assert valid[0]
    assert gx[0] == 1100               # pano_w/2
    assert gy[0] == 256                # pano_h/2
    np.testing.assert_allclose(rng[0], 5.0, atol=1e-6)


def test_rig_to_cylinder_right_is_positive_azimuth():
    # Point to the right (+X, Z=0) -> azimuth +90deg -> gx right of center.
    P = np.array([[5.0, 0.0, 0.0]])
    gx, gy, rng, valid = rp.rig_to_cylinder(P, scale=350.0, pano_w=2200, pano_h=512)
    assert valid[0]
    assert gx[0] == int(round(np.pi / 2 * 350.0 + 1100))


def test_rig_to_cylinder_invalid_at_origin():
    # Degenerate point at the origin -> invalid (no defined direction).
    P = np.array([[0.0, 0.0, 0.0]])
    gx, gy, rng, valid = rp.rig_to_cylinder(P, scale=350.0, pano_w=2200, pano_h=512)
    assert not valid[0]


def test_split_near_far():
    rng = np.array([1.0, 5.0, 20.0, np.inf])
    valid = np.array([True, True, True, False])
    near, far = rp.split_near_far(rng, valid, near_max=8.0)
    np.testing.assert_array_equal(near, [True, True, False, False])
    np.testing.assert_array_equal(far, [False, False, True, True])


def test_scatter_zbuffer_nearest_wins():
    # Two points hit the same pixel; the nearer one's color must win.
    gx = np.array([2, 2])
    gy = np.array([1, 1])
    rng = np.array([5.0, 2.0])                     # second is nearer
    color = np.array([[10, 10, 10], [200, 200, 200]], dtype=np.uint8)
    oc, oz, om = rp.scatter_zbuffer(gx, gy, rng, color, pano_w=4, pano_h=3)
    assert om[1, 2]                                 # pixel marked covered
    np.testing.assert_array_equal(oc[1, 2], [200, 200, 200])
    np.testing.assert_allclose(oz[1, 2], 2.0)
    assert not om[0, 0]                             # untouched pixel empty


def test_scatter_zbuffer_empty_pixels_infinite():
    gx = np.array([0]); gy = np.array([0])
    rng = np.array([3.0]); color = np.array([[1, 2, 3]], dtype=np.uint8)
    oc, oz, om = rp.scatter_zbuffer(gx, gy, rng, color, pano_w=2, pano_h=2)
    assert np.isinf(oz[1, 1])
    assert not om[1, 1]


def test_composite_overlay_replaces_base():
    base = np.full((3, 4, 3), 10, dtype=np.uint8)
    overlay = np.zeros((3, 4, 3), dtype=np.uint8)
    overlay[1, 2] = [200, 200, 200]
    mask = np.zeros((3, 4), dtype=bool)
    mask[1, 2] = True
    out = rp.composite(base, overlay, mask)
    np.testing.assert_array_equal(out[1, 2], [200, 200, 200])
    np.testing.assert_array_equal(out[0, 0], [10, 10, 10])   # base preserved
    assert out is not base                                    # does not mutate input


def test_end_to_end_two_cameras_agree_on_near_point():
    """A single physical point seen by two cameras at different positions must
    land on the SAME pano pixel after reprojection (this is the parallax fix)."""
    scale, pano_w, pano_h = 350.0, 2200, 512
    near_max = 8.0

    # Physical point at rig coords (1, 0, 3).
    P_world = np.array([1.0, 0.0, 3.0])

    # Camera A at origin, identity rotation: sees P in its own frame as P_world.
    Ra, ta = np.eye(3), np.zeros(3)
    Pa_cam = (P_world - ta) @ Ra            # inverse of cam_points_to_rig
    # Camera B shifted +0.5 in X, identity rotation.
    Rb, tb = np.eye(3), np.array([0.5, 0.0, 0.0])
    Pb_cam = (P_world - tb) @ Rb

    a_rig = rp.cam_points_to_rig(Pa_cam[None, :], Ra, ta)
    b_rig = rp.cam_points_to_rig(Pb_cam[None, :], Rb, tb)
    gxa, gya, _, _ = rp.rig_to_cylinder(a_rig, scale, pano_w, pano_h)
    gxb, gyb, _, _ = rp.rig_to_cylinder(b_rig, scale, pano_w, pano_h)

    assert gxa[0] == gxb[0]                  # same azimuth pixel -> no parallax split
    assert gya[0] == gyb[0]


def test_scatter_zbuffer_rejects_out_of_bounds():
    import pytest
    gx = np.array([0]); gy = np.array([10])          # gy >= pano_h
    rng = np.array([1.0]); color = np.array([[1, 2, 3]], dtype=np.uint8)
    with pytest.raises(ValueError):
        rp.scatter_zbuffer(gx, gy, rng, color, pano_w=2, pano_h=2)


def test_scatter_zbuffer_tie_nearest_first_still_wins():
    # Nearest listed FIRST this time -> ensures the result is order-independent.
    gx = np.array([0, 0]); gy = np.array([0, 0])
    rng = np.array([2.0, 5.0])                       # nearer is index 0
    color = np.array([[10, 10, 10], [200, 200, 200]], dtype=np.uint8)
    oc, _, _ = rp.scatter_zbuffer(gx, gy, rng, color, pano_w=2, pano_h=2)
    np.testing.assert_array_equal(oc[0, 0], [10, 10, 10])


def test_rig_to_cylinder_azimuth_seam_wraps():
    # Point directly behind (-Z) -> azimuth = pi -> wraps to gx 0.
    P = np.array([[0.0, 0.0, -5.0]])
    gx, gy, rng, valid = rp.rig_to_cylinder(P, scale=350.0, pano_w=2200, pano_h=512)
    assert valid[0]
    assert gx[0] == 0


def test_cam_points_to_rig_accepts_single_vector():
    out = rp.cam_points_to_rig(np.array([0.0, 0.0, 1.0]), np.eye(3),
                               np.array([1.0, 0.0, 0.0]))
    assert out.shape == (1, 3)
    np.testing.assert_allclose(out, [[1.0, 0.0, 1.0]], atol=1e-9)
