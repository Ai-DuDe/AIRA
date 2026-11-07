"""Step 6 - motion extraction with dense optical flow (Farneback).

Instead of the average flow *magnitude* (which is always positive, so one breath
gives two bumps and the rate comes out doubled), the ROI is split into a grid and
the signed mean flow (dx, dy) of every cell is returned. A later PCA step picks
whatever direction the chest actually moves in, so back, side and front lying all work.
"""
import cv2
import numpy as np


class MotionExtractor:
    def __init__(self, max_side: int = 160, grid: int = 4):
        self.max_side = max_side
        self.grid = grid
        self.prev = None
        self.n_channels = 2 * grid * grid

    def reset(self):
        self.prev = None

    def _prep(self, crop):
        g = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        s = min(1.0, self.max_side / max(h, w))
        if s < 1.0:
            g = cv2.resize(g, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
        g = g.astype(np.float32)
        # Normalize global gain/offset so lighting flicker / IR auto-exposure is not read as motion.
        g = (g - g.mean()) / (g.std() + 1e-6) * 40.0 + 128.0
        g = np.clip(g, 0, 255).astype(np.uint8)
        g = cv2.GaussianBlur(g, (5, 5), 1.0)
        return g, s

    def step(self, crop):
        """Returns (features[2*grid^2], motion_energy) or None on the first frame."""
        g, s = self._prep(crop)
        if self.prev is None or self.prev.shape != g.shape:
            self.prev = g
            return None
        flow = cv2.calcOpticalFlowFarneback(self.prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev = g
        flow /= s  # back to original-resolution pixels
        h, w = g.shape
        n = self.grid
        ys = np.linspace(0, h, n + 1).astype(int)
        xs = np.linspace(0, w, n + 1).astype(int)
        feats = np.empty(self.n_channels, np.float64)
        k = 0
        for i in range(n):
            for j in range(n):
                cell = flow[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
                feats[k] = cell[..., 0].mean()
                feats[k + 1] = cell[..., 1].mean()
                k += 2
        energy = float(np.sqrt((flow ** 2).sum(axis=2)).mean())
        return feats, energy
