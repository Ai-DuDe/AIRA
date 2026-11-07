"""Synthetic multi-baby video with exact ground truth, for measuring alarm latency
and false alarms before touching real data.

Each zone is a textured "blanket" whose chest region moves with breathing
(expansion for back-lying, sideways for side-lying, vertical for prone).
Scripted events: benign pauses (periodic breathing), true apneas, body movement.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Event:
    kind: str      # "pause" (benign, < 15 s), "apnea" (>= 20 s), "move"
    start: float
    dur: float


@dataclass
class BabySpec:
    pose: str = "back"            # back | side | prone
    bpm: float = 45.0
    amp_px: float = 1.0           # chest displacement amplitude in pixels
    noise: float = 3.0            # sensor noise sigma (grey levels)
    flicker: float = 0.0          # lighting gain flicker amplitude
    events: list = field(default_factory=list)


def schedule(rng, duration, pauses, apneas, moves, gap=(25, 55), lead=40.0):
    items = [("pause", d) for d in pauses] + [("apnea", d) for d in apneas] + [("move", d) for d in moves]
    rng.shuffle(items)
    ev, t = [], lead
    for kind, d in items:
        if t + d > duration - 10:
            break
        ev.append(Event(kind, t, d))
        t += d + rng.uniform(*gap)
    return ev


class SyntheticBaby:
    def __init__(self, spec: BabySpec, size: int, fps: float, seed: int):
        self.spec, self.size, self.fps = spec, size, fps
        self.rng = np.random.default_rng(seed)
        self.pad = 30
        S = size + 2 * self.pad
        r = self.rng
        fine = cv2.GaussianBlur(r.normal(0, 1, (S, S)).astype(np.float32), (0, 0), 1.6)
        coarse = cv2.GaussianBlur(r.normal(0, 1, (S, S)).astype(np.float32), (0, 0), 7)
        tex = fine / fine.std() + 1.5 * coarse / coarse.std()
        self.tex = cv2.normalize(tex, None, 50, 200, cv2.NORM_MINMAX)
        Y, X = np.mgrid[0:size, 0:size].astype(np.float32)
        self.X, self.Y = X, Y
        self.phase = 0.0
        self.bpm = spec.bpm
        self.off = np.zeros(2, np.float32)
        self.move_vel = np.zeros(2, np.float32)
        self.env = 1.0

    def _in(self, t, kind):
        return any(e.kind == kind and e.start <= t < e.start + e.dur for e in self.spec.events)

    def render(self, t):
        sp, dt = self.spec, 1.0 / self.fps
        # breathing rate wanders slowly
        self.bpm += self.rng.normal(0, 0.3) + 0.01 * (sp.bpm - self.bpm)
        self.phase += 2 * np.pi * self.bpm / 60 * dt
        stopped = self._in(t, "pause") or self._in(t, "apnea")
        target = 0.0 if stopped else 1.0
        self.env += np.clip(target - self.env, -dt / 0.7, dt / 0.7)
        amp = sp.amp_px * (1 + 0.1 * np.sin(0.05 * t)) * (1 + self.rng.normal(0, 0.03))
        d = self.env * amp * (1 - np.cos(self.phase)) / 2

        if self._in(t, "move"):
            self.move_vel = 0.85 * self.move_vel + self.rng.normal(0, 1.2, 2).astype(np.float32)
            self.off += self.move_vel * dt * 8
            self.off = np.clip(self.off, -self.pad + 2, self.pad - 2)
        size = self.size
        cx, cy = size / 2 + self.off[0], size / 2 + self.off[1]
        sx, sy = size * 0.18, size * 0.14
        W = np.exp(-(((self.X - cx) / sx) ** 2 + ((self.Y - cy) / sy) ** 2) / 2)
        if sp.pose == "back":      # chest rises toward overhead camera -> looks like expansion
            u = d * 1.6 * W * (self.X - cx) / sx
            v = d * 1.6 * W * (self.Y - cy) / sy + 0.3 * d * W
        elif sp.pose == "side":
            u, v = d * W, 0.0 * W
        else:
            u, v = 0.0 * W, -d * W
        mx = self.X - u - self.off[0] + self.pad
        my = self.Y - v - self.off[1] + self.pad
        img = cv2.remap(self.tex, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR)
        gain = 1 + sp.flicker * np.sin(2 * np.pi * 0.15 * t) + sp.flicker * 0.3 * self.rng.normal()
        img = img * gain + self.rng.normal(0, sp.noise, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)


class SyntheticWard:
    """Several babies side by side in one shared frame."""

    def __init__(self, specs, size=180, fps=15.0, seed=0):
        self.babies = [SyntheticBaby(s, size, fps, seed * 100 + i) for i, s in enumerate(specs)]
        self.size, self.fps = size, fps
        self.zones = [(i * size, 0, size, size) for i in range(len(specs))]  # x, y, w, h

    def frame(self, t):
        tiles = [b.render(t) for b in self.babies]
        return cv2.cvtColor(np.hstack(tiles), cv2.COLOR_GRAY2BGR)
