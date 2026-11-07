"""Step 7 - causal (real-time) filtering, motion-pattern projection and breath detection.

`filtfilt` (used in the methodology) is zero-phase but needs the FUTURE of the
signal, so it cannot be used live. Everything here is causal: one sample in,
one sample out, with < 1 s delay.
"""
from collections import deque

import numpy as np
from scipy import signal as sps


class CausalFilter:
    """Streaming Butterworth filter over `channels` parallel channels."""

    def __init__(self, sos, channels: int):
        self.sos = sos
        self.channels = channels
        self.reset()

    def reset(self):
        self.zi = np.zeros((self.sos.shape[0], 2, self.channels))

    def step(self, x):
        y, self.zi = sps.sosfilt(self.sos, np.asarray(x, float).reshape(1, -1), axis=0, zi=self.zi)
        return y[0]


def bandpass(fps, lo, hi, order, channels):
    return CausalFilter(sps.butter(order, [lo, hi], btype="band", fs=fps, output="sos"), channels)


def highpass(fps, hz, order, channels):
    return CausalFilter(sps.butter(order, hz, btype="high", fs=fps, output="sos"), channels)


class AdaptiveProjector:
    """PCA on the recent band-passed motion channels: finds the one motion pattern
    that carries the breathing, whatever direction the baby is lying in."""

    def __init__(self, n_channels: int, window: int, update_every: int):
        self.buf = deque(maxlen=window)
        self.update_every = max(1, update_every)
        self.count = 0
        self.w = np.ones(n_channels) / np.sqrt(n_channels)

    def reset(self):
        self.buf.clear()
        self.count = 0

    def step(self, y, learn: bool = True):
        if learn:
            self.buf.append(np.asarray(y, float))
            self.count += 1
            if self.count % self.update_every == 0 and len(self.buf) >= self.update_every * 2:
                X = np.asarray(self.buf)
                _, vecs = np.linalg.eigh(X.T @ X)
                w = vecs[:, -1]
                if w @ self.w < 0:  # keep the sign stable so the waveform does not flip
                    w = -w
                self.w = w
        return float(np.asarray(y) @ self.w)


class BreathDetector:
    """Causal peak/trough tracker with hysteresis h.

    A breath is confirmed when the signal has risen at least h above the previous
    trough and then fallen h below the peak. So h IS the minimum breath depth.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.rising = True
        self.ext = None
        self.ext_t = 0.0
        self.trough = None

    def step(self, s: float, t: float, h: float):
        if self.ext is None:
            self.ext, self.ext_t, self.trough = s, t, s
            return None
        if self.rising:
            if s > self.ext:
                self.ext, self.ext_t = s, t
            elif s < self.ext - h:
                peak_t, depth = self.ext_t, self.ext - self.trough
                self.rising, self.ext, self.ext_t = False, s, t
                if depth >= h:
                    return peak_t, depth
        else:
            if s < self.ext:
                self.ext, self.ext_t = s, t
            elif s > self.ext + h:
                self.trough = self.ext
                self.rising, self.ext, self.ext_t = True, s, t
        return None
