"""One visual theme for every AIRA window (live dashboard, demo, launcher).

Soft neumorphic look: everything sits on one warm light-grey surface; panels are
"raised" (light shadow top-left, dark shadow bottom-right) and wells are "pressed in".
Accent is a warm terracotta. Status colours keep their usual clinical meaning.
"""
import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---- palette ------------------------------------------------------------------------
HEX = dict(
    bg="#E7E3DD",       # the one surface everything is made of
    light="#FFFFFF",    # highlight (top-left)
    shadow="#C3BDB4",   # shadow (bottom-right)
    well="#DFDAD3",     # pressed-in areas
    text="#34302B",
    muted="#8C857B",
    faint="#B3ACA2",
    accent="#C0703F",   # terracotta
    ok="#4E9A64",       # breathing normally
    watch="#D49A2A",    # short pause
    pre="#DD7330",      # pre-alert
    alarm="#D0433D",    # alarm
    video="#2B2825",    # behind video frames
)


def rgb(name_or_hex):
    h = HEX.get(name_or_hex, name_or_hex).lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def bgr(name_or_hex):
    r, g, b = rgb(name_or_hex)
    return (b, g, r)


TIER_HEX = {0: HEX["ok"], 1: HEX["watch"], 2: HEX["pre"], 3: HEX["alarm"]}

# ---- fonts ------------------------------------------------------------------------
_FONT_DIRS = [os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts"), "/usr/share/fonts/truetype/dejavu", "/Library/Fonts"]
_FONT_FILES = {
    "regular": ["segoeui.ttf", "DejaVuSans.ttf", "Arial.ttf", "arial.ttf"],
    "semibold": ["seguisb.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf"],
    "bold": ["segoeuib.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf"],
    "light": ["segoeuil.ttf", "segoeui.ttf", "DejaVuSans.ttf", "arial.ttf"],
}


@lru_cache(maxsize=64)
def font(size, weight="regular"):
    for name in _FONT_FILES[weight]:
        for d in _FONT_DIRS:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return ImageFont.truetype(p, int(size))
    return ImageFont.load_default()


# ---- soft shapes (rendered once, cached) ------------------------------------------------
def _rr(draw, box, r, fill):
    draw.rounded_rectangle(box, radius=r, fill=fill)


@lru_cache(maxsize=256)
def raised(w, h, r=22, d=7, blur=9, face=None):
    """Raised panel as an RGB array of size (h + 2m, w + 2m); the face sits at offset m."""
    m = d + 2 * blur
    W, H = w + 2 * m, h + 2 * m
    img = Image.new("RGB", (W, H), rgb("bg"))
    sh = Image.new("RGB", (W, H), rgb("bg"))
    dr = ImageDraw.Draw(sh)
    _rr(dr, (m + d, m + d, m + w + d, m + h + d), r, rgb("shadow"))
    _rr(dr, (m - d, m - d, m + w - d, m + h - d), r, rgb("light"))
    img = sh.filter(ImageFilter.GaussianBlur(blur))
    _rr(ImageDraw.Draw(img), (m, m, m + w, m + h), r, rgb(face or "bg"))
    return np.asarray(img), m


@lru_cache(maxsize=256)
def pressed(w, h, r=16, d=4, blur=5, face=None, outer=None):
    """Pressed-in well as an RGB array of exactly (h, w)."""
    base = rgb(face or "well")
    o = 2 * d + blur
    size = (w + 2 * o, h + 2 * o)
    # dark inner shadow along the top-left edge
    dk = Image.new("RGB", size, rgb("shadow"))
    _rr(ImageDraw.Draw(dk), (o + d, o + d, o + w + d, o + h + d), r, base)
    dk = np.asarray(dk.filter(ImageFilter.GaussianBlur(blur)).crop((o, o, o + w, o + h)), np.int16)
    # light inner highlight along the bottom-right edge
    lt = Image.new("RGB", size, rgb("light"))
    _rr(ImageDraw.Draw(lt), (o - d, o - d, o + w - d, o + h - d), r, base)
    lt = np.asarray(lt.filter(ImageFilter.GaussianBlur(blur)).crop((o, o, o + w, o + h)), np.int16)
    b = np.array(base, np.int16)
    out = Image.fromarray(np.clip(dk + (lt - b) * 0.7, 0, 255).astype(np.uint8))
    mask = Image.new("L", (w, h), 0)
    _rr(ImageDraw.Draw(mask), (0, 0, w - 1, h - 1), r, 255)
    bg = Image.new("RGB", (w, h), rgb(outer or "bg"))
    bg.paste(out, (0, 0), mask)
    return np.asarray(bg)


@lru_cache(maxsize=64)
def round_mask(w, h, r):
    m = Image.new("L", (w, h), 0)
    _rr(ImageDraw.Draw(m), (0, 0, w - 1, h - 1), r, 255)
    return (np.asarray(m, np.float32) / 255.0)[..., None]


class Painter:
    """Collects text while an OpenCV (BGR) canvas is composed, then draws it all with
    real fonts in one pass (OpenCV's built-in fonts are hard to read)."""

    def __init__(self):
        self.ops = []

    @staticmethod
    def width(s, size, weight="regular"):
        return font(size, weight).getlength(s)

    def text(self, s, xy, size=16, color="text", weight="regular", anchor="la"):
        """anchor: PIL anchors, e.g. 'la' left, 'ra' right, 'ma' middle (a = top of ascender), 'lm' left-middle."""
        self.ops.append((s, xy, size, color, weight, anchor))

    def flush(self, canvas_bgr):
        if not self.ops:
            return canvas_bgr
        img = Image.fromarray(canvas_bgr[..., ::-1].copy())
        dr = ImageDraw.Draw(img)
        for s, xy, size, color, weight, anchor in self.ops:
            dr.text(xy, s, font=font(size, weight), fill=rgb(color), anchor=anchor)
        self.ops = []
        return np.ascontiguousarray(np.asarray(img)[..., ::-1])
