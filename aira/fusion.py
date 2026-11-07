"""Max-SNR channel fusion (found by the 12-way breath experiment e03): weights the 32 motion
channels by breathing power vs >2.5 Hz noise power, instead of plain PCA."""
from collections import deque

import numpy as np

from .breathing import highpass


class FusionProjector:
    """Causal fusion of the 32 band-passed displacement channels.
    mode 'pca'  : top eigvec of Ryy (baseline).
    mode 'gevd' : max-SNR / MRC: top generalized eigvec of Ryy w = l (Rnn + reg*tr/n I) w,
                  Rnn = covariance of >2.5 Hz feats (noise; breathing has no power there).
    mode 'snrw' : diagonal SNR weighting of the PCA vector (w_i *= snr_i/(1+snr_i)).
    mode 'smooth': PCA with covariance EMA-smoothing across updates (more stable w)."""

    def __init__(self, n, window, every, fps, mode, reg):
        self.n, self.mode, self.reg = n, mode, reg
        self.buf = deque(maxlen=window); self.nbuf = deque(maxlen=window)
        self.every = max(1, every); self.count = 0
        self.w = np.ones(n) / np.sqrt(n)
        self.hp = highpass(fps, 2.5, 2, n); self.feat = np.zeros(n); self.C = None; self.Cn = None; self.alpha = 0.3

    def reset(self):
        self.buf.clear(); self.nbuf.clear(); self.count = 0

    def push_feats(self, f):
        self.feat = self.hp.step(f)

    def step(self, y, learn=True):
        y = np.asarray(y, float)
        if learn:
            self.buf.append(y); self.nbuf.append(self.feat); self.count += 1
            if self.count % self.every == 0 and len(self.buf) >= self.every * 2:
                X = np.asarray(self.buf); Ryy = X.T @ X / len(X)
                if self.mode == "pca":
                    w = np.linalg.eigh(Ryy)[1][:, -1]
                elif self.mode == "smooth":
                    self.C = Ryy if self.C is None else 0.7 * self.C + 0.3 * Ryy
                    w = np.linalg.eigh(self.C)[1][:, -1]
                else:
                    N = np.asarray(self.nbuf); Rnn = N.T @ N / len(N)
                    if self.mode == "gsm":
                        a = self.alpha
                        self.C = Ryy if self.C is None else (1 - a) * self.C + a * Ryy
                        self.Cn = Rnn if self.Cn is None else (1 - a) * self.Cn + a * Rnn
                        Ryy, Rnn = self.C, self.Cn
                    if self.mode in ("gevd", "gsm"):
                        Rnn = Rnn + self.reg * np.trace(Rnn) / self.n * np.eye(self.n) + 1e-12 * np.eye(self.n)
                        L = np.linalg.cholesky(Rnn); Li = np.linalg.inv(L)
                        v = np.linalg.eigh(Li @ Ryy @ Li.T)[1][:, -1]
                        w = Li.T @ v
                    else:
                        w = np.linalg.eigh(Ryy)[1][:, -1]
                        snr = np.diag(Ryy) / (np.diag(Rnn) + 1e-12)
                        w = w * snr / (1 + snr)
                    w = w / (np.linalg.norm(w) + 1e-12)
                if w @ self.w < 0:
                    w = -w
                self.w = w
        return float(y @ self.w)
