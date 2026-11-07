"""Landmark tracker: finds people every frame and publishes the small patches AIRA reads.

Runs in its own thread on the NEWEST camera frame (never slows the camera loop):
  * MediaPipe Pose        -> per person a small CHEST patch just below the collarbones,
                             where breathing moves the body most (used for breathing);
  * MediaPipe Face Mesh   -> per face FOREHEAD + both CHEEK patches, placed on exact face
                             points so they follow the face (used for heart rate).

Patches are smoothed and only move when the person really moves, so the measurements
are not disturbed by landmark jitter. If the models are missing or nobody is found,
AIRA falls back to the motion-based search.
"""
import os
import threading
import time

import numpy as np

from .person import ROOT, PersonFinder, _quiet_stderr

FACE_MODEL = os.path.join(ROOT, "models", "face_landmarker.task")

# Face Mesh point groups (MediaPipe 468/478-point mesh)
FOREHEAD = (10, 109, 67, 103, 338, 297, 332, 151, 9, 108, 337)
LEFT_CHEEK = (116, 117, 118, 100, 126, 142, 203, 205, 50, 101, 36)
RIGHT_CHEEK = (345, 346, 347, 329, 355, 371, 423, 425, 280, 330, 266)


def _bbox(pts, W, H, shrink=0.0):
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    w, h = x1 - x0, y1 - y0
    x0, x1, y0, y1 = x0 + w * shrink, x1 - w * shrink, y0 + h * shrink, y1 - h * shrink
    x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
    return (x0, y0, x1 - x0, y1 - y0) if x1 - x0 > 3 and y1 - y0 > 3 else None


class FaceMesh:
    def __init__(self, max_faces=4):
        self.det = None
        if not os.path.exists(FACE_MODEL):
            return
        try:
            os.environ.setdefault("GLOG_minloglevel", "2")
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
            opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=FACE_MODEL), num_faces=max_faces,
                                                min_face_detection_confidence=0.4, min_face_presence_confidence=0.4)
            with _quiet_stderr():
                self.det = vision.FaceLandmarker.create_from_options(opts)
                self.det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((64, 64, 3), np.uint8)))
            self._mp = mp
        except Exception:
            self.det = None

    @property
    def available(self):
        return self.det is not None

    def patches(self, frame_bgr, with_points=False):
        """List (one per face) of [(name, box), ...] for forehead and both cheeks
        (+ the face mesh points per face if with_points)."""
        if self.det is None:
            return ([], []) if with_points else []
        H, W = frame_bgr.shape[:2]
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_bgr[..., ::-1]))
        out, all_pts = [], []
        for lm in self.det.detect(img).face_landmarks:
            pts = np.array([[p.x * W, p.y * H] for p in lm], np.float32)
            all_pts.append(pts)
            face = []
            for name, idx, shrink in (("forehead", FOREHEAD, 0.1), ("left cheek", LEFT_CHEEK, 0.15), ("right cheek", RIGHT_CHEEK, 0.15)):
                b = _bbox(pts[list(idx)], W, H, shrink)
                if b:
                    face.append((name, b))
            if face:
                out.append(face)
        return (out, all_pts) if with_points else out


def chest_patch(person, W, H):
    """Small patch just below the collarbones (between the shoulders), where breathing shows most."""
    pts = person.pts
    vis = pts[:, 2] >= 0.5
    if not (vis[11] and vis[12]):
        return None
    l, r = pts[11, :2], pts[12, :2]
    span = float(np.linalg.norm(l - r))
    if span < 20:
        return None
    sh = (l + r) / 2
    if vis[23] or vis[24]:
        # body axis shoulders -> hips: works whichever way the person lies (back, side, sitting)
        hips = np.mean([pts[i, :2] for i in (23, 24) if vis[i]], axis=0)
        axis = hips - sh
        length = float(np.linalg.norm(axis))
        cx, cy = sh + 0.33 * axis
        side = max(0.7 * span, 0.45 * length)
        w = h = side
    else:
        # hips out of view (e.g. sitting at a desk): straight down from the shoulder line
        cx, cy = sh
        w, h = 0.75 * span, 0.55 * span
        cy += 0.40 * span
    x0, y0 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
    x1, y1 = int(min(W, cx + w / 2)), int(min(H, cy + h / 2))
    return (x0, y0, x1 - x0, y1 - y0) if x1 - x0 > 16 and y1 - y0 > 16 else None


class LandmarkTracker:
    def __init__(self, max_people=4):
        self.pose = PersonFinder(max_people)
        self.mesh = FaceMesh(max_people)
        self.chests = []          # latest chest boxes (one per person)
        self.faces = []           # latest face patch lists
        self.poses = []           # latest body points per person: (33, 3) x, y, visibility
        self.face_pts = []        # latest face mesh points per face: (478, 2)
        self.updated = 0.0        # time of the latest result
        self._frame = None
        self._new = threading.Event()
        self._stop = False
        if self.available:
            threading.Thread(target=self._run, daemon=True, name="landmarks").start()

    @property
    def available(self):
        return self.pose.available or self.mesh.available

    def submit(self, frame_bgr):
        """Hand over the newest frame (the thread always works on the newest one)."""
        self._frame = frame_bgr
        self._new.set()

    def _run(self):
        while not self._stop:
            self._new.wait(0.5)
            self._new.clear()
            f = self._frame
            if f is None or f.ndim != 3:
                continue
            H, W = f.shape[:2]
            try:
                people = self.pose.find(f) if self.pose.available else []
                chests = [c for c in (chest_patch(p, W, H) for p in people) if c]
                faces, face_pts = self.mesh.patches(f, with_points=True) if self.mesh.available else ([], [])
            except Exception:
                continue
            self.chests, self.faces, self.updated = chests, faces, time.time()
            self.poses, self.face_pts = [p.pts for p in people], face_pts

    def stop(self):
        self._stop = True


class ZoneFollower:
    """Turns noisy per-frame chest boxes into steady zones that follow each person.

    * The zone SIZE is fixed when a person is first found (the breathing analysis needs the
      same crop size every frame).
    * A zone only moves when its person has REALLY moved (> 25% of the zone size, consistently
      for 1 s); then it jumps once. Landmark jitter never moves it.
    * A new person becomes a new zone only after being seen steadily for 1.5 s, and only if
      they don't overlap an existing zone (so one person never gets two cots)."""

    def __init__(self, move_frac=0.25, move_hold_s=1.0, new_hold_s=1.5):
        self.move_frac, self.move_hold_s, self.new_hold_s = move_frac, move_hold_s, new_hold_s
        self.pending = {}      # zone index -> (first time displaced, latest target box)
        self.newbies = []      # [(first seen t, box)]
        self.moved = []        # zone indices that jumped this update

    @staticmethod
    def _overlap(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        iy = max(0, min(ay + ah, by + bh) - max(ay, by))
        return ix * iy / float(max(1, min(aw * ah, bw * bh)))

    def update(self, zones, chests, t, max_zones=0):
        self.moved = []
        unmatched = list(chests)
        for i, (x, y, w, h) in enumerate(zones):
            if not unmatched:
                self.pending.pop(i, None)
                continue
            c = np.array([x + w / 2, y + h / 2])
            d = [np.hypot(cx + cw / 2 - c[0], cy + ch / 2 - c[1]) for cx, cy, cw, ch in unmatched]
            j = int(np.argmin(d))
            if d[j] > 1.5 * max(w, h):
                self.pending.pop(i, None)
                continue
            cx, cy, cw, ch = unmatched.pop(j)
            if d[j] > self.move_frac * max(w, h):
                first, _ = self.pending.get(i, (t, None))
                self.pending[i] = (first, (cx, cy, cw, ch))
                if t - first >= self.move_hold_s:                  # moved, and stayed moved: jump once
                    nx, ny = int(cx + cw / 2 - w / 2), int(cy + ch / 2 - h / 2)
                    zones[i] = (max(0, nx), max(0, ny), w, h)
                    self.pending.pop(i, None)
                    self.moved.append(i)
            else:
                self.pending.pop(i, None)
        # new people: steady for a while and not on top of an existing zone
        fresh = []
        for box in unmatched:
            if any(self._overlap(box, z) > 0.2 for z in zones):
                continue
            seen = [n for n in self.newbies if self._overlap(box, n[1]) > 0.3]
            if seen:
                first = seen[0][0]
                if t - first >= self.new_hold_s:
                    fresh.append(box)
                    self.newbies = [n for n in self.newbies if n is not seen[0]]
                else:
                    seen[0][1] = box
            else:
                self.newbies.append([t, box])
        self.newbies = [n for n in self.newbies if t - n[0] < 5.0]
        if max_zones:
            fresh = fresh[: max(0, max_zones - len(zones))]
        return fresh
