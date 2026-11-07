"""Draws the body + face mesh on the camera picture, in AIRA's own colours.

  * skeleton  - terracotta lines with a soft white halo and round joints
  * face      - face outline, eyes, eyebrows and lips as fine terracotta lines
  * breathing - the chest patch as a rounded outline that swells and shrinks with each breath
  * heart     - a tiny heart on the forehead that beats at the measured heart rate
"""
import time

import cv2
import numpy as np

from . import theme as T

POSE_LINKS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
              (24, 26), (26, 28), (15, 17), (15, 19), (17, 19), (16, 18), (16, 20), (18, 20), (27, 29), (29, 31),
              (28, 30), (30, 32)]
POSE_HEAD = [(0, 2), (0, 5), (2, 7), (5, 8), (9, 10)]          # only used if there is no face mesh
JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LEFT_BROW = [70, 63, 105, 66, 107]
RIGHT_BROW = [300, 293, 334, 296, 336]

_t = np.linspace(0, 2 * np.pi, 40)
HEART = np.stack([16 * np.sin(_t) ** 3, -(13 * np.cos(_t) - 5 * np.cos(2 * _t) - 2 * np.cos(3 * _t) - np.cos(4 * _t))], 1) / 17.0


def _line(img, a, b, col, w):
    cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), col, w, cv2.LINE_AA)


def _poly(img, pts, col, w, closed):
    cv2.polylines(img, [pts.astype(np.int32)], closed, col, w, cv2.LINE_AA)


def _rounded(img, x, y, w, h, r, col, thick):
    r = int(max(2, min(r, w / 2, h / 2)))
    x, y, w, h = int(x), int(y), int(w), int(h)
    for p0, p1 in (((x + r, y), (x + w - r, y)), ((x + r, y + h), (x + w - r, y + h)),
                   ((x, y + r), (x, y + h - r)), ((x + w, y + r), (x + w, y + h - r))):
        cv2.line(img, p0, p1, col, thick, cv2.LINE_AA)
    for (cx, cy), ang in (((x + r, y + r), 180), ((x + w - r, y + r), 270), ((x + w - r, y + h - r), 0), ((x + r, y + h - r), 90)):
        cv2.ellipse(img, (cx, cy), (r, r), ang, 0, 90, col, thick, cv2.LINE_AA)


def _halfwidth(v):
    """Torso silhouette, v=-1 (neck) .. 1 (belly): neck -> round shoulders -> chest -> soft waist."""
    v = np.asarray(v, float)
    neck = 0.22
    shoulder = neck + 0.72 * np.sqrt(np.clip((v + 1) / 0.35, 0, 1))      # neck flares into shoulders
    body = 1.0 - 0.3 * np.clip((v + 0.2) / 1.0, 0, 1) ** 1.5           # gentle taper to the waist
    return np.minimum(shoulder, body)


def _surface(img, x, y, w, h, b, col, halo, s):
    """Body-shaped wireframe torso over the breathing area; the chest rises and widens with each breath."""
    cx, cy = x + w / 2, y + h / 2
    f = np.linspace(-1, 1, 70)

    def project(U, V):
        chest = np.exp(-((V + 0.25) / 0.55) ** 2)                      # breathing mostly moves the chest
        hw = _halfwidth(V) * (1 + 0.07 * b * chest)
        X = cx + U * hw * w / 2
        bulge = np.sqrt(np.clip(1 - U ** 2, 0, 1))                     # rounded front of the body
        Y = cy + V * h / 2 + (0.09 + 0.03 * b * chest) * h * bulge - 0.04 * h * b * chest * bulge
        return np.stack([X, Y], 1)

    lines = [project(f, np.full_like(f, v)) for v in np.linspace(-0.75, 0.95, 9)]       # rib-like curves
    lines += [project(np.full_like(f, u), f) for u in np.linspace(-0.85, 0.85, 7)]      # lines down the body
    outline = np.vstack([project(np.ones_like(f), f), project(-np.ones_like(f), f[::-1])])
    lay = img.copy()
    thin = max(1, int(1.2 * s))
    for r in lines:
        _poly(lay, r, halo, thin + 2, False)
    _poly(lay, outline, halo, thin + 3, False)
    for r in lines:
        _poly(lay, r, col, thin, False)
    _poly(lay, outline, col, thin + 1, False)
    a = 0.6 + 0.25 * max(0.0, b)
    img[:] = cv2.addWeighted(lay, a, img, 1 - a, 0)


def draw(frame, poses=(), faces=(), zones=(), breath=(), heart_bpm=None, selected=None):
    """Camera picture with a soft see-through UI-coloured patch on each breathing area,
    which grows and glows on every in-breath."""
    img = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(zones):
        b = float(np.clip(breath[i] if i < len(breath) else 0.0, -1, 1))
        col = T.bgr("accent") if (selected is None or selected == i) else T.bgr("#F4D9C8")
        g = 0.05 * b * max(w, h)
        x0, y0, x1, y1 = int(x - g), int(y - g), int(x + w + g), int(y + h + g)
        r = int(0.3 * min(w, h))
        lay = img.copy()
        cv2.rectangle(lay, (x0 + r, y0), (x1 - r, y1), col, -1)
        cv2.rectangle(lay, (x0, y0 + r), (x1, y1 - r), col, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(lay, (cx, cy), r, col, -1, cv2.LINE_AA)
        a = 0.35 + 0.15 * b
        img = cv2.addWeighted(lay, a, img, 1 - a, 0)
        _rounded(img, x0, y0, x1 - x0, y1 - y0, r, (255, 255, 255), 2)
    return img
