"""Run AIRA on a webcam, a video file, or the built-in synthetic ward.

    python run_live.py --source synthetic --auto-zones   # demo, babies found automatically
    python run_live.py --source 0 --auto-zones           # webcam, babies found automatically
    python run_live.py --source 0 --auto-zones --adult   # test it on yourself (adult breathing rates)
    python run_live.py --source 0 --select-zones         # webcam, draw a box per cot (Enter after each, Esc when done)
    python run_live.py --source clip.mp4 --zones zones.json

--auto-zones: watches the frame for ~12 s and finds every patch that moves with a
breathing rhythm (one patch = one baby). It rescans every 2 min in the background
and adds newly placed babies. Found zones are saved to zones.json.
All alerts are written to aira_events.csv.
"""
import argparse
import csv
import json
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from aira.autoroi import find_breathing_zones, new_zones
from aira.config import AiraConfig
from aira.dashboard import Dashboard
from aira.watchdog import follow_launcher
from aira.heart import HeartRateEstimator, HeartWorker, apply_to
from aira.tracker import LandmarkTracker, ZoneFollower
from aira.monitor import BabyMonitor
from aira.synthetic import BabySpec, SyntheticWard, schedule

SCAN_S = 12.0
RESCAN_EVERY_S = 120.0
SCAN_WIDTH = 480


def open_source(src, fps_arg):
    if src == "synthetic":
        import numpy as np
        rng = np.random.default_rng(7)
        specs = [BabySpec(pose=p, bpm=b, amp_px=a, noise=n, events=schedule(rng, 3600, [5, 8, 12], [22, 30], [4, 8], gap=(30, 50)))
                 for p, b, a, n in [("back", 45, 1.0, 3), ("side", 55, 0.6, 4), ("prone", 40, 0.8, 8)]]
        ward = SyntheticWard(specs, fps=fps_arg or 15)
        return ("synthetic", ward), ward.fps, ward.zones
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {src}")
    fps = fps_arg or cap.get(cv2.CAP_PROP_FPS) or 15
    if fps <= 1 or fps > 120:
        fps = 15
    return ("cap", cap), fps, None


class AutoZoner:
    """Keeps a short downscaled frame history and finds breathing zones in a background thread."""

    def __init__(self, fps, frame_shape, band, scan_s=SCAN_S, max_zones=0):
        self.fps = fps
        self.max_zones = max_zones
        self.band = band
        self.scan_s = scan_s
        h, w = frame_shape[:2]
        self.scale = min(1.0, SCAN_WIDTH / w)
        self.buf = deque(maxlen=int(scan_s * fps))
        self.result = None
        self.busy = False
        self.last_colour = None
        self.people_found = 0

    def add(self, frame):
        self.last_colour = frame
        f = frame if self.scale == 1.0 else cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        self.buf.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f)

    def ready(self):
        return len(self.buf) == self.buf.maxlen

    def start_scan(self):
        if self.busy or not self.ready():
            return
        self.busy = True
        frames = list(self.buf)
        colour = None if self.last_colour is None else self.last_colour.copy()

        def work():
            # breathing area = where something actually rises and falls like breathing
            # (motion-based; camera shimmer over the whole picture is rejected inside)
            z = find_breathing_zones(frames, self.fps, band=self.band, work_width=SCAN_WIDTH, max_zones=self.max_zones)
            k = 1.0 / self.scale
            self.result = [(int(x * k), int(y * k), int(w * k), int(h * k)) for x, y, w, h in z]
            self.busy = False

        threading.Thread(target=work, daemon=True).start()

    def take(self):
        r, self.result = self.result, None
        return r


def same_picture(a, b):
    """Many webcams report ~30 fps but only produce a NEW picture ~10 times a second and
    repeat the last one in between. Repeated pictures have zero motion and must be skipped,
    or breathing (and the movement detector) see a false 'nothing moves' most of the time."""
    if a is None or b is None or a.shape != b.shape:
        return False
    sa, sb = a[::8, ::8].astype(np.int16), b[::8, ::8].astype(np.int16)
    return float(np.abs(sa - sb).mean()) < 0.05


def attach_heart(heart, mons, zones):
    """Give the heart rate to the monitor whose zone is closest to the skin region it came from."""
    if not mons:
        return
    if heart.box is None or not heart.colour:
        for m in mons:
            apply_to(m, heart)
        return
    x, y, w, h = heart.box
    cx, cy = x + w / 2, y + h / 2
    k = min(range(len(zones)), key=lambda i: (zones[i][0] + zones[i][2] / 2 - cx) ** 2 + (zones[i][1] + zones[i][3] / 2 - cy) ** 2)
    for i, m in enumerate(mons):
        apply_to(m, heart if i == k else None, None if i == k else "Pulse is being read from another cot")


def main():
    follow_launcher()  # close this page if the launcher is stopped
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic")
    ap.add_argument("--zones", default="zones.json")
    ap.add_argument("--select-zones", action="store_true", help="draw cot boxes by hand")
    ap.add_argument("--auto-zones", action="store_true", help="find breathing babies automatically")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--adult", action="store_true",
                    help="self-test on an adult: breathing band down to 7 bpm (adults rest at ~12-20 bpm)")
    ap.add_argument("--name", default=None, help="label for the card(s), e.g. --name You")
    ap.add_argument("--no-heart", action="store_true", help="turn off the experimental heart rate from skin colour")
    a = ap.parse_args()

    (kind, src), fps, zones = open_source(a.source, a.fps)
    live = kind == "cap" and a.source.isdigit()
    first = None
    if kind == "cap":
        ok, first = src.read()
        if not ok:
            raise SystemExit("no frames from source")
        if live and a.fps is None:
            # webcams often report 30 fps but deliver less; rates are computed from fps, so measure it
            # count only NEW pictures: repeated frames are not real samples
            t0, n, prev = time.time(), 0, first
            while n < 30 and time.time() - t0 < 10:
                ok, f = src.read()
                if ok and not same_picture(f, prev):
                    n += 1
                    prev = f
            fps = max(5.0, min(60.0, n / (time.time() - t0)))
            first = prev
            print(f"camera delivers {fps:.1f} new pictures per second")
    if a.auto_zones:
        zones = []
    elif kind == "cap":
        if a.select_zones:
            zones = []          # drawn on the page itself (draw-cots mode), see below
        else:
            try:
                zones = [tuple(z) for z in json.load(open(a.zones))]
            except FileNotFoundError:
                raise SystemExit(f"{a.zones} not found - run with --auto-zones or --select-zones")
        if not zones and not a.select_zones:
            raise SystemExit("no zones")

    cfg = AiraConfig(fps=fps)
    if a.adult:
        # Infant settings cut off below 15 bpm, which is normal adult breathing. Same algorithm, wider band.
        cfg.band_low_hz = 0.12
        cfg.calibration_s = 16.0   # adults breathe slower: a little longer to see 4 breaths
    base_name = a.name or ("You" if a.adult else "Cot")
    label = lambda i: base_name if (a.name or a.adult) and i == 1 else f"{base_name} {i}"
    mons = [BabyMonitor(label(i + 1), cfg) for i in range(len(zones))]
    subject = "the person" if a.adult else "the baby"
    dash = Dashboard(mons)
    dash.subtitle = "Live camera" if live else ("Synthetic ward" if kind == "synthetic" else f"Video: {a.source}")
    zoner = None
    next_scan = 0.0
    auto = {"on": a.auto_zones}
    # landmark tracking (MediaPipe): find each person in ~1 s and keep the breathing zone ON their
    # chest as they move. Motion search stays as the fallback (blankets, IR, nobody found).
    tracker = LandmarkTracker(1 if a.adult else 4) if (kind == "cap" and a.auto_zones) else None
    if tracker is not None and not tracker.available:
        tracker = None
    follower = ZoneFollower()
    track = {"t0": None, "found": False}

    def back_to_auto():
        """'Draw cots myself' switched off: forget hand-drawn cots and search automatically again."""
        zones[:] = []
        mons[:] = []
        seen[:] = []
        auto["on"], auto["restart"] = True, True

    def set_zones(boxes):
        """Cots drawn by hand replace whatever was there; automatic search stops."""
        zones[:] = [tuple(int(v) for v in b) for b in boxes]
        mons[:] = []
        for i in range(len(zones)):
            m = BabyMonitor(label(i + 1), cfg)
            m.subject = subject
            m._log(None, "info", "cot drawn by hand")
            mons.append(m)
        seen[:] = [0] * len(mons)
        auto["on"] = False
        json.dump(zones, open(a.zones, "w"))

    if kind == "cap":                 # real camera or video file: the user may draw cots (not synthetic)
        dash.allow_draw = True
        dash.on_zones = set_zones
        dash.on_auto = back_to_auto
        if a.select_zones:
            dash.set_manual(True)
    heart = None if a.no_heart or kind == "synthetic" else HeartRateEstimator(fps, (42, 180) if a.adult else (80, 220))
    heart_worker = HeartWorker(heart) if heart is not None else None   # own thread, own skin region
    pool = ThreadPoolExecutor(max_workers=8)  # OpenCV/SciPy release the GIL -> real parallelism
    logf = open("aira_events.csv", "a", newline="")
    log = csv.writer(logf)
    seen = []
    i, t_start = 0, time.time()
    last_frame = None
    try:
        while True:
            if kind == "synthetic":
                t = i / fps
                frame = src.frame(t)
            else:
                if first is not None:
                    frame, first = first, None
                else:
                    ok, frame = src.read()
                    if not ok:
                        break
                    if live and same_picture(frame, last_frame):
                        continue               # repeated picture from the camera: not a new sample
                last_frame = frame
                t = (time.time() - t_start) if live else i / fps

            # ---- automatic zones ----
            # ---- landmark tracking: chest patches that follow each person ----
            if tracker is not None and not dash.manual:
                tracker.submit(frame)
                track["t0"] = t if track["t0"] is None else track["t0"]
                if tracker.updated and tracker.chests:
                    if not track["found"]:
                        track["found"] = True
                        auto["on"] = False             # people found: no need for the slow motion search
                        dash.scan_progress = None
                        zones[:] = [z for z in zones]  # (motion-search zones, if any, are kept)
                    for z in follower.update(zones, tracker.chests, t, 1 if a.adult else 0):
                        zones.append(z)
                        m = BabyMonitor(label(len(zones)), cfg)
                        m.subject = subject
                        m._log(t, "info", "auto-detected (person found)")
                        mons.append(m)
                    for i in follower.moved:            # zone jumped with its person: restart the motion
                        if i < len(mons):              # comparison so the jump itself isn't "movement"
                            mons[i].motion.reset()
                elif not track["found"] and t - track["t0"] > 6 and not auto["on"] and not zones:
                    auto["on"] = True                  # nobody found: fall back to the motion search
            if auto.pop("restart", False):
                zoner, next_scan = None, 0.0
            if auto["on"] and not dash.drawing:
                if zoner is None:
                    zoner = AutoZoner(fps, frame.shape, (cfg.band_low_hz, cfg.band_high_hz), 20.0 if a.adult else SCAN_S,
                                      max_zones=1 if a.adult else 0)  # self-test = one person
                zoner.add(frame)
                if not zones:
                    dash.scan_progress = (min(t, zoner.scan_s), zoner.scan_s)
                if t >= next_scan and zoner.ready():
                    zoner.start_scan()
                    next_scan = t + (RESCAN_EVERY_S if zones else 2.0)
                found = zoner.take()
                if found is not None:
                    added = new_zones(found, zones)
                    if a.adult:
                        added = added[: max(0, 1 - len(zones))]   # "Monitor me" = one person, never a 2nd zone
                    for z in added:
                        zones.append(z)
                        m = BabyMonitor(label(len(zones)), cfg)
                        m.subject = subject
                        m._log(t, "info", f"auto-detected at {z}")
                        mons.append(m)
                    if added:
                        json.dump(zones, open(a.zones, "w"))
                    elif not zones:
                        dash.scan_progress = None
                        dash.message = "No breathing found yet - still looking. Is the chest/belly in view?"

            crops = [frame[y:y + h, x:x + w] for (x, y, w, h) in zones]
            if heart_worker is not None:
                faces = tracker.faces if (tracker is not None and time.time() - tracker.updated < 1.0) else []
                heart_worker.submit(frame, t, faces[0] if faces else None)   # runs separately; never waits
            snaps = list(pool.map(lambda mc: mc[0].update(mc[1], t), zip(mons, crops)))
            if heart is not None:
                attach_heart(heart, mons, zones)
            dash.camera, dash.zones = frame, zones
            if tracker is not None:
                dash.full_view = not dash.manual           # whole picture + body mesh on the cards
                dash.mesh = (tracker.poses, tracker.face_pts) if time.time() - tracker.updated < 1.0 else ((), ())
            dash.skin_box = heart.box if heart is not None else None
            seen += [0] * (len(mons) - len(seen))
            for k, m in enumerate(mons):
                for ev in m.events[seen[k]:]:
                    log.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), f"{ev[0]:.1f}" if ev[0] is not None else "", ev[1], ev[2]])
                    print(f"[{t:7.1f}s] {ev[1].upper():9s} {ev[2]}")
                seen[k] = len(m.events)
            if not dash.render(crops, snaps, t):
                break
            i += 1
            if not live:  # play files / synthetic at real speed
                delay = t_start + i / fps - time.time()
                if delay > 0:
                    time.sleep(delay)
    finally:
        logf.close()
        pool.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
