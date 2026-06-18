"""Nova 360 — one combined 3D explainer (Manim Community, ThreeDScene).

The whole story in a single video, in 3D:
  1. 4 cameras on the car, facing outward
  2. cylindrical projection: every direction -> a point on a cylinder
  3. unroll the cylinder -> the flat panorama
  4. parallax: far objects agree between cameras, near objects disagree ("melt")
  5. depth reprojection: rebuild 3D points, re-render from ONE center -> they agree
  6. z-buffer: many points on one ray -> keep the nearest
  7. recap: far = rotation-only base, near = depth overlay

Render:
  manim -pql nova360.py Nova360          # 480p preview (fast)
  manim -pqh nova360.py Nova360          # 1080p
Uses Text (Pango), so no LaTeX needed; needs manim + ffmpeg.

Note: 3D scenes sometimes need a small tweak once you SEE them render (a label
nudged, a camera angle, a run_time). If a label sits awkwardly, tell me which
beat and I'll adjust.
"""
from manim import *
import numpy as np


def pol(az_deg, r, z=0.0):
    """Polar (azimuth in the ground plane) -> a 3D point. az 0 = +X (right)."""
    a = np.deg2rad(az_deg)
    return np.array([r * np.cos(a), r * np.sin(a), z])


class Nova360(ThreeDScene):
    def construct(self):
        self._cap = None
        self.set_camera_orientation(phi=64 * DEGREES, theta=-50 * DEGREES, zoom=0.9)

        self.caption("Nova 360: 4 cameras -> one panorama")

        # ---- Beat 1: the car + 4 outward cameras --------------------------
        car = Prism(dimensions=[1.4, 2.2, 0.5])
        car.set_fill(GREY, opacity=0.65)
        car.set_stroke(WHITE, width=1)
        self.play(FadeIn(car))

        # az 0=+X(right), 90=+Y(front), 180=-X(left), 270=-Y(back)
        cam_specs = [("front", 90, GREEN), ("right", 0, RED),
                     ("back", 270, YELLOW), ("left", 180, BLUE)]
        cam_dots, cam_arrows, cam_labels = VGroup(), VGroup(), VGroup()
        for nm, az, c in cam_specs:
            base = pol(az, 1.0, 0.25)
            cam_dots.add(Dot3D(point=base, radius=0.09, color=c))
            cam_arrows.add(Arrow3D(start=base, end=pol(az, 2.6, 0.25), color=c))
            lbl = Text(nm, font_size=22, color=c).move_to(pol(az, 3.2, 0.25))
            self.add_fixed_orientation_mobjects(lbl)   # stays facing the camera
            cam_labels.add(lbl)
        self.play(LaggedStartMap(FadeIn, cam_dots, lag_ratio=0.2),
                  LaggedStartMap(FadeIn, cam_arrows, lag_ratio=0.2),  # GrowArrow is 2D-only
                  LaggedStartMap(FadeIn, cam_labels, lag_ratio=0.2))
        self.wait(1)

        # ---- Beat 2: the cylinder + rays ----------------------------------
        self.caption("Cylindrical projection: each direction -> a point on a cylinder")
        cyl = Cylinder(radius=3.0, height=3.0, direction=OUT, resolution=(16, 24))
        cyl.set_fill(BLUE_E, opacity=0.12)
        cyl.set_stroke(BLUE_E, width=1, opacity=0.5)
        self.play(FadeIn(cyl))

        self.begin_ambient_camera_rotation(rate=0.06)
        ray_specs = [(90, 0.0, GREEN), (0, 0.6, RED), (180, -0.5, BLUE),
                     (270, 0.3, YELLOW), (45, -0.8, ORANGE), (135, 0.8, PURPLE)]
        rays, hits = VGroup(), VGroup()
        for az, z, c in ray_specs:
            end = pol(az, 3.0, z)
            rays.add(Line3D(start=ORIGIN, end=end, color=c, thickness=0.012))
            hits.add(Dot3D(point=end, radius=0.08, color=c))
        self.play(LaggedStartMap(FadeIn, rays, lag_ratio=0.12),   # 3D: FadeIn not Create
                  LaggedStartMap(FadeIn, hits, lag_ratio=0.12))
        self.wait(3)
        self.stop_ambient_camera_rotation()

        # ---- Beat 3: unroll the cylinder -> flat panorama strip -----------
        self.caption("Unroll the cylinder -> the flat panorama")
        self.play(FadeOut(cyl), FadeOut(rays), FadeOut(car),
                  FadeOut(cam_dots), FadeOut(cam_arrows), FadeOut(cam_labels),
                  FadeOut(hits))

        strip = Rectangle(width=10.0, height=1.8, color=BLUE_E)
        strip.set_fill(BLUE_E, opacity=0.1)
        dots = VGroup()
        for az, z, c in ray_specs:
            x = -5.0 + (az / 360.0) * 10.0          # azimuth -> position along strip
            y = (z / 1.5) * 0.8                     # height -> up/down on the strip
            dots.add(Dot([x, y, 0], color=c, radius=0.09))
        axis_lbl = Text("azimuth 0 -> 360   (which way you face)",
                        font_size=22).next_to(strip, DOWN, buff=0.25)
        strip_grp = VGroup(strip, dots, axis_lbl).move_to(ORIGIN)
        self.add_fixed_in_frame_mobjects(strip_grp)
        self.play(Create(strip), LaggedStartMap(FadeIn, dots, lag_ratio=0.1),
                  Write(axis_lbl))
        self.wait(2)
        self.play(FadeOut(strip_grp))

        # ---- Beat 4: parallax (why near objects melt) ---------------------
        self.caption("Parallax: far objects agree, near objects disagree")
        camL = Dot3D(point=np.array([-2.2, -2.4, 0]), color=WHITE, radius=0.1)
        camR = Dot3D(point=np.array([2.2, -2.4, 0]), color=WHITE, radius=0.1)
        base = Line3D(camL.get_center(), camR.get_center(), color=GREY, thickness=0.02)
        self.play(FadeIn(camL), FadeIn(camR), FadeIn(base))

        far = Dot3D(point=np.array([0, 3.0, 0]), color=YELLOW, radius=0.12)
        fL = Line3D(camL.get_center(), far.get_center(), color=GREEN, thickness=0.012)
        fR = Line3D(camR.get_center(), far.get_center(), color=RED, thickness=0.012)
        self.play(FadeIn(far), FadeIn(fL), FadeIn(fR))
        self.caption("FAR: rays nearly parallel -> cameras AGREE")
        self.wait(1.5)
        self.play(FadeOut(far), FadeOut(fL), FadeOut(fR))

        near = Dot3D(point=np.array([0, 0.2, 0]), color=ORANGE, radius=0.12)
        nL = Line3D(camL.get_center(), near.get_center(), color=GREEN, thickness=0.012)
        nR = Line3D(camR.get_center(), near.get_center(), color=RED, thickness=0.012)
        self.play(FadeIn(near), FadeIn(nL), FadeIn(nR))
        self.caption("NEAR: rays diverge -> DISAGREE -> doubled / 'melt'")
        ghostL = Dot3D(point=np.array([0.5, 0.2, 0]), color=GREEN, radius=0.12).set_opacity(0.55)
        ghostR = Dot3D(point=np.array([-0.5, 0.2, 0]), color=RED, radius=0.12).set_opacity(0.55)
        self.play(FadeIn(ghostL), FadeIn(ghostR))
        self.wait(2)

        # ---- Beat 5: depth reprojection (re-render from one center) -------
        self.caption("Depth reprojection: re-render from ONE rig center")
        self.play(FadeOut(ghostL), FadeOut(ghostR), FadeOut(nL), FadeOut(nR))
        center = Dot3D(point=np.array([0, -2.4, 0]), color=WHITE, radius=0.12)
        ray = Line3D(center.get_center(), near.get_center(), color=WHITE, thickness=0.02)
        self.play(FadeIn(center), FadeIn(ray))
        self.caption("one viewpoint -> one direction -> cameras AGREE (no melt)")
        self.wait(2)

        # ---- Beat 6: z-buffer --------------------------------------------
        self.caption("z-buffer: keep the NEAREST point on each ray")
        far2 = Dot3D(point=np.array([0, 3.0, 0]), color=BLUE, radius=0.12)
        ray2 = Line3D(center.get_center(), np.array([0, 3.0, 0]), color=GREY, thickness=0.012)
        self.play(FadeOut(ray), FadeIn(ray2), FadeIn(far2))
        self.wait(0.5)
        self.play(far2.animate.set_opacity(0.2))     # the far one is hidden behind near
        self.caption("scatter_reduce('amin') keeps the orange (nearest) point")
        self.wait(2)

        # ---- Beat 7: recap ------------------------------------------------
        self.play(FadeOut(VGroup(camL, camR, base, near, far2, ray2, center)))
        recap = VGroup(
            Text("FAR  ->  rotation-only cylinder (fast, the base layer)", font_size=26),
            Text("NEAR ->  depth reprojection (corrects parallax, on top)", font_size=26),
        ).arrange(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(recap)
        self.caption("Two layers combined = the final 360")
        self.play(FadeIn(recap))
        self.wait(3)

    # ---- helper: a top-of-frame caption that swaps between beats ----------
    def caption(self, text):
        new = Text(text, font_size=28).to_edge(UP)
        self.add_fixed_in_frame_mobjects(new)
        if self._cap is not None:
            self.play(FadeOut(self._cap), FadeIn(new), run_time=0.8)
        else:
            self.play(FadeIn(new), run_time=0.8)
        self._cap = new
