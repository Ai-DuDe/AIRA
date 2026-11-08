"""Presentation demo: real AIR-125 infant videos playing as a 3-cot ward.

    python demo.py                                   # one baby each from S01, S03, S05
    python demo.py --subjects S02 S04 S05
    python demo.py --fullscreen

On screen, per cot:
  * the real video, with the auto-detected chest zone (red box),
  * AIRA's breathing waveform,
  * GREEN ticks (top) = breaths marked by the dataset annotators,
    WHITE ticks (bottom) = breaths AIRA detected. They should line up.

One apnea case is scripted: at 40 s, Cot 2's video freezes for 35 s (labelled
SIMULATED APNEA on screen). Watch PAUSE (10 s), PRE-ALERT (15 s), ALARM (20 s).
Change with --apnea-cot N --apnea-at SECONDS, or --apnea-cot 0 for none.

Keys:
  1-3      select a cot
  a        an extra SIMULATED APNEA on the selected cot: the video freezes for 35 s
           (+ camera-like noise), so the chest stops moving. Watch PAUSE (10 s),
           PRE-ALERT (15 s), ALARM (20 s). Clearly labelled as simulated on screen.
  click    acknowledge an alarm
  f        full screen on/off
  Stop     the red button at the top right closes the demo

The AIR-125 clips contain no real apneas, which is why the apnea is simulated.
For a real apnea, use the live webcam demo:  python run_live.py --source 0 --auto-zones
"""
import argparse
import glob
import os
import time

import cv2
import h5py
import numpy as np

from aira.autoroi import find_breathing_zones
from aira.config import AiraConfig
from aira.dashboard import Dashboard
from aira.watchdog import follow_launcher
from aira.monitor import BabyMonitor
from aira.heart import HeartRateEstimator, HeartWorker, apply_to

ROOT = "data/air125/AIR_125"
CACHE = "data/cache"
WORK_W = 640
FREEZE_S = 35.0  # alarm needs 20 s without a breath; margin for the last breath before freezing


PRIVACY = os.environ.get("AIRA_PRIVACY") == "1"   # blur the babies (for screenshots / videos to share)


_pf = None


def privacy_blur(img, state):
    """Blur only the baby's head (found with MediaPipe body points; held between checks).
    Only the picture shown is blurred; AIRA still measures the original video."""
    global _pf
    if _pf is None:
        from aira.person import PersonFinder
        _pf = PersonFinder(1)
    H, W = img.shape[:2]
    state["n"] = state.get("n", 0) + 1
    if state["n"] % 5 == 1:
        for p in _pf.find(img):
            pts = p.pts[:11]
            ok = pts[:, 2] >= 0.2
            hp = pts[ok, :2] if ok.sum() >= 2 else pts[:, :2]
            c = hp.mean(0)
            r = max(2.3 * float(np.max(np.linalg.norm(hp - c, axis=1))), 0.09 * W)
            state["head"] = (c, r)
    out = img.copy()
    if "head" in state:
        (cx, cy), r = state["head"]
        soft = cv2.GaussianBlur(cv2.resize(cv2.resize(img, (W // 14, H // 14)), (W, H)), (0, 0), 5)
        m = np.zeros((H, W), np.uint8)
        cv2.circle(m, (int(cx), int(cy)), int(r), 255, -1, cv2.LINE_AA)
        m = cv2.GaussianBlur(m, (0, 0), r * 0.15)[..., None] / 255.0
        out = (soft * m + img * (1 - m)).astype(np.uint8)
        cv2.putText(out, "blurred for privacy", (int(cx - 60), int(max(14, cy - r - 6))), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (230, 230, 230), 1, cv2.LINE_AA)
    else:
        out = cv2.GaussianBlur(cv2.resize(cv2.resize(img, (W // 14, H // 14)), (W, H)), (0, 0), 5)   # not found yet
    return out


def prep(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (WORK_W, int(g.shape[0] * WORK_W / g.shape[1])), interpolation=cv2.INTER_AREA)
    h, w = g.shape
    g[: int(0.07 * h), : int(0.45 * w)] = 0  # timestamp overlay (ticks at 1 Hz) - same mask as validation
    return g


def zone_for(mp4, fps):
    """Auto-ROI zone: from the validation cache if present, else computed now from the first 20 s."""
    c = os.path.join(CACHE, mp4.split(os.sep)[-2] + "_" + os.path.basename(mp4)[:-4] + ".npz")
    if os.path.exists(c):
        return tuple(int(v) for v in np.load(c)["zone"])
    cap = cv2.VideoCapture(mp4)
    frames = []
    while len(frames) < int(20 * fps):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(prep(f))
    zs, dbg = find_breathing_zones(frames, fps, return_debug=True)
    if not zs:
        return (0, 0, frames[0].shape[1], frames[0].shape[0])
    b = max(range(len(zs)), key=lambda i: dbg["blobs"][i]["coherence"] * np.sqrt(dbg["blobs"][i]["blocks"]))
    return zs[b]


class Cot:
    """Plays all clips of one subject back to back through one BabyMonitor."""

    def __init__(self, name, subject):
        self.clips = sorted(c for c in glob.glob(os.path.join(ROOT, subject, "*.mp4")) if os.path.exists(c[:-4] + ".hdf5"))
        if not self.clips:
            raise SystemExit(f"no clips for {subject} in {ROOT} - download AIR-125 first (see README)")
        self.subject = subject
        cap = cv2.VideoCapture(self.clips[0])
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 15
        cap.release()
        self.mon = BabyMonitor(f"{name} ({subject})", AiraConfig(fps=self.fps))
        self.heart = HeartRateEstimator(self.fps, band_bpm=(80, 220))  # infant heart rates
        self.heart_worker = HeartWorker(self.heart)  # own thread + own skin region (not the breathing box)
        self.mon.demo_gt = []
        self.idx = -1
        self.cap = None
        self.clip_t0 = 0.0      # ward time at which the current clip's frame 0 was shown
        self.frame_i = 0
        self.next_t = 0.0
        self.freeze_until = None
        self.frozen = None
        self.rng = np.random.default_rng(len(name))
        self.display = None
        self.priv = {}
        self.snap = None

    def _open_next(self, t):
        self.idx = (self.idx + 1) % len(self.clips)
        mp4 = self.clips[self.idx]
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(mp4)
        self.zone = zone_for(mp4, self.fps)
        with h5py.File(mp4[:-4] + ".hdf5", "r") as h:
            gt = np.flatnonzero(np.asarray(h["impulse"]) > 0.5) / self.fps
        self.clip_t0 = t
        self.mon.demo_gt = list(t + gt)
        self.frame_i = 0
        self.priv = {}
        self.mon.motion.reset()  # new recording: different framing, don't compare across the cut
        self.clip_name = os.path.relpath(mp4, ROOT)

    def freeze(self, t):
        if self.freeze_until is None and self.frozen is not None:
            self.freeze_until = t + FREEZE_S
            # the recording (and its labelled breaths) pauses: no green ticks while the baby is "not breathing"
            self.mon.demo_gt = [g for g in self.mon.demo_gt if g < t] + [g + FREEZE_S for g in self.mon.demo_gt if g >= t]
            self.mon._log(t, "info", "SIMULATED APNEA started (video frozen)")

    def step(self, t):
        """Advance to ward time t. Returns True if a new frame was processed."""
        if t < self.next_t:
            return False
        self.next_t += 1.0 / self.fps
        if self.freeze_until is not None:
            if t >= self.freeze_until:
                # resume the recording (labels were already shifted when the freeze started)
                self.freeze_until = None
                self.mon.demo_banner = None
                self.clip_t0 += FREEZE_S
            else:
                g, disp = self.frozen
                noisy = np.clip(g.astype(np.float32) + self.rng.normal(0, 2.0, g.shape), 0, 255).astype(np.uint8)
                left = self.freeze_until - t
                self.mon.demo_banner = f"SIMULATED APNEA ({left:.0f}s left)"
                apply_to(self.mon, None, "Paused while the video is frozen")
                return self._process(noisy, disp, t)
        if self.cap is None:
            self._open_next(t)
        ok, frame = self.cap.read()
        if not ok:
            self._open_next(t)
            ok, frame = self.cap.read()
        self.frame_i += 1
        g = prep(frame)
        disp = cv2.resize(frame, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA)
        self.heart_worker.submit(disp, t)   # pulse from skin colour, before boxes are drawn on it
        apply_to(self.mon, self.heart)
        x, y, w, h = self.zone
        if PRIVACY:
            disp = privacy_blur(disp, self.priv)
        cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(disp, self.clip_name, (8, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        self.frozen = (g, disp)
        return self._process(g, disp, t)

    def _process(self, g, disp, t):
        x, y, w, h = self.zone
        self.snap = self.mon.update(g[y:y + h, x:x + w], t)
        self.display = disp
        return True


def main():
    follow_launcher()  # close this page if the launcher is stopped
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=["S01", "S03", "S05"])
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--apnea-cot", type=int, default=2, help="cot that gets the scripted apnea (0 = none)")
    ap.add_argument("--apnea-at", type=float, default=40.0, help="seconds after start")
    a = ap.parse_args()

    cots = [Cot(f"Cot {i + 1}", s) for i, s in enumerate(a.subjects)]
    dash = Dashboard([c.mon for c in cots], title="AIRA demo - real AIR-125 infant video")
    dash.subtitle = "Demo: real AIR-125 infant recordings" + (
        f"   -   Cot {a.apnea_cot} gets a simulated apnea at {a.apnea_at:.0f} s, press A to add one" if a.apnea_cot else "   -   press A for a simulated apnea")
    apnea_idx = a.apnea_cot - 1 if a.apnea_cot else None
    apnea_done = False
    if a.fullscreen:
        dash.toggle_fullscreen()

    def on_key(k, sel, t):
        if k == ord("a"):
            cots[sel].freeze(t)

    dash.on_key = on_key
    seen = [0] * len(cots)
    t_start = time.time()
    while True:
        t = time.time() - t_start
        for c in cots:
            c.step(t)
        # the one scripted apnea case: after every cot has finished learning its baseline
        if apnea_idx is not None and not apnea_done and t >= a.apnea_at:
            cots[apnea_idx].freeze(t)
            apnea_done = True
        if all(c.snap is not None for c in cots):
            if not dash.render([c.display for c in cots], [c.snap for c in cots], t):
                break
        for k, c in enumerate(cots):
            for ev in c.mon.events[seen[k]:]:
                print(f"[{t:6.1f}s] {ev[1].upper():9s} {ev[2]}")
            seen[k] = len(c.mon.events)
        time.sleep(0.003)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
