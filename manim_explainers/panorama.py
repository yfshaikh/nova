"""Manim Community explainers for the 360 cylindrical panorama.

Three scenes, in teaching order:
  PixelIsRay      - a photo stores direction+color per pixel, NOT distance
  FourCameraRig   - 4 outward cameras cover ~360 (rotation-only = one shared center)
  CylinderPanorama- directions -> points on a cylinder -> unrolled into a flat strip

Render (low quality preview):
  manim -pql panorama.py PixelIsRay
  manim -pql panorama.py FourCameraRig
  manim -pql panorama.py CylinderPanorama
Render all three at once:   manim -pql panorama.py -a
Higher quality:             swap -pql for -pqh
Uses Text (Pango), so no LaTeX needed.
"""
from manim import *
import numpy as np


class PixelIsRay(Scene):
    """A pixel records a DIRECTION + a COLOR. Distance along the ray is unknown."""

    def construct(self):
        title = Text("A pixel is a ray: direction + color, not distance",
                     font_size=30).to_edge(UP)
        self.play(Write(title))

        cam = Dot(LEFT * 4.5, color=WHITE)
        cam_label = Text("camera", font_size=24).next_to(cam, DOWN)
        self.play(FadeIn(cam), Write(cam_label))

        # A fan of colored rays leaving the camera (each = one pixel's direction).
        rays = VGroup()
        ray_colors = [RED, ORANGE, YELLOW, GREEN, BLUE]
        for c, ang in zip(ray_colors, np.linspace(-0.45, 0.45, 5)):
            end = cam.get_center() + 9 * np.array([np.cos(ang), np.sin(ang), 0])
            rays.add(Line(cam.get_center(), end, color=c, stroke_width=4))
        self.play(LaggedStartMap(Create, rays, lag_ratio=0.2))

        # Highlight that distance is unknown: a dot slides along one ray.
        mover = Dot(color=YELLOW).move_to(rays[2].get_start())
        self.play(FadeIn(mover))
        self.play(mover.animate.move_to(rays[2].point_from_proportion(0.55)))
        self.play(mover.animate.move_to(rays[2].point_from_proportion(0.95)))

        note = Text("each pixel = one direction;\nHOW FAR along the ray is unknown",
                    font_size=26).to_edge(DOWN)
        self.play(Write(note))
        self.wait(2)


class FourCameraRig(Scene):
    """Four outward-facing cameras tile ~360 of directions (top-down)."""

    def construct(self):
        title = Text("4 cameras -> ~360 coverage", font_size=30).to_edge(UP)
        self.play(Write(title))

        car = Rectangle(width=1.4, height=2.4, color=GREY,
                        fill_opacity=0.4).move_to(ORIGIN)
        car_lbl = Text("car (top-down)", font_size=22).next_to(car, DOWN, buff=0.15)
        self.play(FadeIn(car), Write(car_lbl))

        # FOV wedges, each ~105 deg, centered on a cardinal direction.
        names = ["front", "right", "back", "left"]
        center_deg = [90, 0, 270, 180]      # screen: up, right, down, left
        cols = [GREEN, RED, YELLOW, BLUE]
        wedges, labels = VGroup(), VGroup()
        for nm, cen, c in zip(names, center_deg, cols):
            cenr = np.deg2rad(cen)
            half = np.deg2rad(105 / 2)
            wedges.add(Sector(arc_center=ORIGIN, radius=2.8,
                              start_angle=cenr - half, angle=2 * half,
                              color=c, fill_opacity=0.18,
                              stroke_color=c, stroke_width=2))
            lp = 3.15 * np.array([np.cos(cenr), np.sin(cenr), 0])
            labels.add(Text(nm, font_size=22, color=c).move_to(lp))
        self.play(LaggedStartMap(FadeIn, wedges, lag_ratio=0.25),
                  LaggedStartMap(Write, labels, lag_ratio=0.25))

        note = Text("neighbors overlap (~15 deg)\nrotation-only pretends all 4 share ONE center",
                    font_size=24).to_edge(DOWN)
        self.play(Write(note))
        self.wait(2)


class CylinderPanorama(Scene):
    """Every direction hits a cylinder; unroll the cylinder -> the panorama strip."""

    def construct(self):
        title = Text("Cylindrical projection: directions -> an unrolled strip",
                     font_size=28).to_edge(UP)
        self.play(Write(title))

        # Left: viewer at center of a cylinder (top-down circle).
        center = LEFT * 3.7 + DOWN * 0.3
        viewer = Dot(center, color=WHITE)
        cyl = Circle(radius=2.1, color=BLUE_D).move_to(center)
        cyl_lbl = Text("cylinder\n(top-down)", font_size=20).next_to(cyl, DOWN, buff=0.1)
        self.play(FadeIn(viewer), Create(cyl), Write(cyl_lbl))

        # Rays at several azimuths; mark where each hits the cylinder.
        az_deg = [90, 25, 155, 250, 320]
        cols = [GREEN, RED, YELLOW, BLUE, ORANGE]
        rays, hits = VGroup(), VGroup()
        for a, c in zip(az_deg, cols):
            r = np.deg2rad(a)
            p = center + 2.1 * np.array([np.cos(r), np.sin(r), 0])
            rays.add(Line(center, p, color=c, stroke_width=3))
            hits.add(Dot(p, color=c))
        self.play(LaggedStartMap(Create, rays, lag_ratio=0.15),
                  LaggedStartMap(FadeIn, hits, lag_ratio=0.15))
        self.wait(1)

        # Right: the unrolled strip (the panorama). Place each hit by its azimuth.
        strip = Rectangle(width=6.0, height=1.6, color=BLUE_D).move_to(RIGHT * 3.5 + DOWN * 0.3)
        self.play(Create(strip))
        x0 = strip.get_left()[0]
        y0 = strip.get_center()[1]
        targets = VGroup(*[
            Dot([x0 + (a / 360.0) * 6.0, y0, 0], color=c)
            for a, c in zip(az_deg, cols)
        ])
        moving = hits.copy()
        self.play(Transform(moving, targets), run_time=2)

        axis = Text("azimuth 0 -> 360  (which compass way you face)",
                    font_size=22).next_to(strip, DOWN, buff=0.2)
        self.play(Write(axis))
        caption = Text("unroll the cylinder = the flat panorama",
                       font_size=22).to_edge(DOWN)
        self.play(Write(caption))
        self.wait(2)
