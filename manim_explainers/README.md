# Manim explainers — 360 panorama & depth reprojection

Animated explanations of the two core ideas, built with **Manim Community Edition**.
Docs: https://docs.manim.community/

## ⭐ The all-in-one 3D video

`nova360.py` is a single **3D** scene (`Nova360`) that tells the whole story end to
end — 4 cameras → cylinder → unroll → parallax → depth reprojection → z-buffer →
recap. This is the one to watch first.

```bash
cd manim_explainers
manim -pql nova360.py Nova360      # 480p preview (fast)
manim -pqh nova360.py Nova360      # 1080p
```

The `panorama.py` / `depth_reproject.py` files below are the original **2D**,
single-concept versions — keep them if you want shorter focused clips, or ignore
them in favor of `nova360.py`.

## What's here

| File | Scene | Explains |
|------|-------|----------|
| `panorama.py` | `PixelIsRay` | a photo stores direction + color per pixel, not distance |
| `panorama.py` | `FourCameraRig` | 4 outward cameras cover ~360 (rotation-only = one shared center) |
| `panorama.py` | `CylinderPanorama` | directions hit a cylinder; unroll it → the panorama strip |
| `depth_reproject.py` | `Parallax` | far objects agree between cameras; near objects disagree → "melt" |
| `depth_reproject.py` | `DepthReprojection` | depth → 3D points → re-render from one center → cameras agree |
| `depth_reproject.py` | `ZBuffer` | many points on one ray; keep the nearest (`scatter_reduce('amin')`) |

Watch them in the table's order — each builds on the previous.

## Install (one time)

Manim CE needs Python + **ffmpeg**. It does **not** need LaTeX here (these scenes
use `Text`/Pango, not `MathTex`).

```bash
# ffmpeg
#   macOS:        brew install ffmpeg
#   Ubuntu/Jetson: sudo apt-get install -y ffmpeg
pip install manim
manim --version        # confirm it installed
```

If you prefer not to touch your system Python:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install manim
```

## Render

```bash
cd manim_explainers

# one scene, low-quality preview (-p opens it, -ql = 480p fast)
manim -pql panorama.py CylinderPanorama

# every scene in a file
manim -pql panorama.py -a
manim -pql depth_reproject.py -a

# high quality (1080p) when you like them
manim -pqh panorama.py -a
```

Rendered videos land in `media/videos/<file>/<quality>/<SceneName>.mp4`.

## Tweaking

- Each scene is a `class <Name>(Scene)` with a `construct(self)` method — edit the
  body to change what's drawn.
- Common knobs: `font_size=`, color constants (`RED`, `BLUE`, `GREEN`, `ORANGE`,
  `YELLOW`, `GREY`, `WHITE`), `.shift(UP/DOWN/LEFT/RIGHT * n)`, `run_time=` on
  `self.play(...)`, and `self.wait(seconds)` for pacing.
- Animations used: `Create`, `Write`, `FadeIn`/`FadeOut`, `Transform`,
  `ReplacementTransform`, `LaggedStartMap`, and `mobject.animate.<method>()`.
