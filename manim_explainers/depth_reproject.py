"""Manim Community explainers for parallax and depth reprojection.

Three scenes, in teaching order:
  Parallax          - far objects agree between cameras; near objects disagree (melt)
  DepthReprojection - depth -> 3D points -> re-render from ONE center -> they agree
  ZBuffer           - many points on one ray; keep the nearest (scatter_reduce amin)

Render:
  manim -pql depth_reproject.py Parallax
  manim -pql depth_reproject.py DepthReprojection
  manim -pql depth_reproject.py ZBuffer
All at once:  manim -pql depth_reproject.py -a
Uses Text (Pango), so no LaTeX needed.
"""
from manim import *
import numpy as np


class Parallax(Scene):
    """Why a CLOSE object melts: the two cameras see it in different directions."""

    def construct(self):
        title = Text("Parallax: why close objects melt", font_size=30).to_edge(UP)
        self.play(Write(title))

        camL = Dot(LEFT * 2.2 + DOWN * 2.4, color=WHITE)
        camR = Dot(RIGHT * 2.2 + DOWN * 2.4, color=WHITE)
        base = Line(camL.get_center(), camR.get_center(), color=GREY)
        base_lbl = Text("two cameras (~70 cm apart)", font_size=22).next_to(base, DOWN, buff=0.15)
        self.play(FadeIn(camL), FadeIn(camR), Create(base), Write(base_lbl))

        # FAR object: rays from both cameras are nearly parallel -> agreement.
        far = Dot(UP * 2.7, color=YELLOW)
        far_lbl = Text("FAR object", font_size=22, color=YELLOW).next_to(far, UP, buff=0.15)
        rL = Line(camL.get_center(), far.get_center(), color=GREEN)
        rR = Line(camR.get_center(), far.get_center(), color=RED)
        self.play(FadeIn(far), Write(far_lbl), Create(rL), Create(rR))
        agree = Text("rays nearly parallel -> same direction -> cameras AGREE",
                     font_size=24, color=GREEN).to_edge(DOWN)
        self.play(Write(agree))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in (rL, rR, far, far_lbl, agree)])

        # NEAR object: rays diverge strongly -> the cameras disagree.
        near = Dot(UP * 0.1, color=ORANGE)
        near_lbl = Text("NEAR object", font_size=22, color=ORANGE).next_to(near, RIGHT, buff=0.2)
        nL = Line(camL.get_center(), near.get_center(), color=GREEN)
        nR = Line(camR.get_center(), near.get_center(), color=RED)
        self.play(FadeIn(near), Write(near_lbl), Create(nL), Create(nR))
        disagree = Text("rays diverge -> different directions -> cameras DISAGREE",
                        font_size=24, color=RED).to_edge(DOWN)
        self.play(Write(disagree))
        self.wait(1.5)

        # The "melt": each camera places the object at a different spot; blending
        # them gives a doubled / smeared result.
        ghostL = near.copy().set_color(GREEN).set_opacity(0.55).shift(RIGHT * 0.45)
        ghostR = near.copy().set_color(RED).set_opacity(0.55).shift(LEFT * 0.45)
        melt = Text("blend the two -> doubled / smeared = 'melt'",
                    font_size=24).to_edge(DOWN)
        self.play(ReplacementTransform(disagree, melt),
                  FadeIn(ghostL), FadeIn(ghostR))
        self.wait(2)


class DepthReprojection(Scene):
    """The fix: rebuild 3D points and re-photograph from one center viewpoint."""

    def construct(self):
        title = Text("Depth reprojection: re-render from ONE viewpoint",
                     font_size=28).to_edge(UP)
        self.play(Write(title))

        camL = Dot(LEFT * 2.2 + DOWN * 2.4, color=WHITE)
        camR = Dot(RIGHT * 2.2 + DOWN * 2.4, color=WHITE)
        center = Dot(DOWN * 2.4, color=WHITE)
        cl = Text("L", font_size=20).next_to(camL, DOWN, buff=0.1)
        cr = Text("R", font_size=20).next_to(camR, DOWN, buff=0.1)
        self.play(FadeIn(camL), FadeIn(camR), Write(cl), Write(cr))

        near = Dot(UP * 0.6, color=ORANGE)
        near_lbl = Text("near object", font_size=22, color=ORANGE).next_to(near, RIGHT, buff=0.2)
        self.play(FadeIn(near), Write(near_lbl))

        step1 = Text("1) depth turns each pixel into a known 3D POINT (X, Y, Z)",
                     font_size=24).to_edge(DOWN)
        self.play(Write(step1))
        coords = Text("(X, Y, Z)", font_size=20, color=ORANGE).next_to(near, UP, buff=0.15)
        self.play(FadeIn(coords))
        self.wait(1.5)

        step2 = Text("2) place it in ONE shared frame using each camera's position",
                     font_size=24).to_edge(DOWN)
        self.play(ReplacementTransform(step1, step2))
        self.wait(1.5)

        step3 = Text("3) re-photograph from the rig CENTER -> one definite direction",
                     font_size=24).to_edge(DOWN)
        self.play(ReplacementTransform(step2, step3))
        ray = Line(center.get_center(), near.get_center(), color=WHITE, stroke_width=4)
        self.play(FadeIn(center), Create(ray))
        self.wait(1.5)

        done = Text("both cameras now agree -> no melt", font_size=28, color=GREEN).to_edge(DOWN)
        self.play(ReplacementTransform(step3, done))
        self.wait(2)


class ZBuffer(Scene):
    """Many points project to one pixel; keep the nearest (hide what's behind)."""

    def construct(self):
        title = Text("z-buffer: keep the nearest point per pixel", font_size=30).to_edge(UP)
        self.play(Write(title))

        center = Dot(LEFT * 4.5, color=WHITE)
        clbl = Text("rig center", font_size=22).next_to(center, DOWN, buff=0.15)
        self.play(FadeIn(center), Write(clbl))

        # One viewing direction (ray) with two points on it: near and far.
        far_pt = center.get_center() + np.array([9.0, 2.0, 0])
        near_pt = center.get_center() + np.array([4.2, 0.93, 0])
        ray = Line(center.get_center(), far_pt, color=GREY)
        self.play(Create(ray))

        near_dot = Dot(near_pt, color=ORANGE)
        near_l = Text("near", font_size=20, color=ORANGE).next_to(near_dot, UP, buff=0.1)
        far_dot = Dot(far_pt, color=BLUE)
        far_l = Text("far (behind)", font_size=20, color=BLUE).next_to(far_dot, UP, buff=0.1)
        self.play(FadeIn(near_dot), FadeIn(far_dot), Write(near_l), Write(far_l))

        note = Text("same direction -> same pixel.\nkeep the nearest; hide what's behind.",
                    font_size=24).to_edge(DOWN)
        self.play(Write(note))
        self.play(far_dot.animate.set_opacity(0.2), far_l.animate.set_opacity(0.2))

        keep = Text("scatter_reduce('amin') keeps the orange point",
                    font_size=24, color=ORANGE).to_edge(DOWN)
        self.play(ReplacementTransform(note, keep))
        self.wait(2)
