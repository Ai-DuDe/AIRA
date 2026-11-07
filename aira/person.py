"""Find people with MediaPipe Pose and turn their body points into the regions AIRA needs.

Works with partly visible bodies: every body point comes with a visibility score and
only points the model actually sees are used.

  torso   - chest + belly box (shoulders -> hips). If the hips are not visible it is
            extended down from the shoulders; if the shoulders are not visible, up from the hips.
            -> the BREATHING area.
  head    - box around the visible face points (nose, eyes, ears, mouth).
  skin    - named patches: forehead, cheeks (from the head), neck, hands -> where the PULSE is read.

Model: models/pose_landmarker_lite.task (Google MediaPipe). If it is missing, everything
here returns no people and AIRA falls back to motion-only search.
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSE_MODEL = os.path.join(ROOT, "models", "pose_landmarker_lite.task")

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, MOUTH_L, MOUTH_R = 0, 2, 5, 7, 8, 9, 10
L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24
FACE_PTS = (NOSE, 1, L_EYE, 3, 4, R_EYE, 6, L_EAR, R_EAR, MOUTH_L, MOUTH_R)
VIS = 0.5


class _quiet_stderr:
    def __enter__(self):
        import sys
        try:
            sys.stderr.flush()
            self.fd, self.null = os.dup(2), os.open(os.devnull, os.O_WRONLY)
            os.dup2(self.null, 2)
        except Exception:
            self.fd = None
        return self

    def __exit__(self, *a):
        if self.fd is not None:
            os.dup2(self.fd, 2)
            os.close(self.fd)
            os.close(self.null)


def _box(pts, pad, W, H, min_side=8):
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    w, h = x1 - x0, y1 - y0
    x0, x1 = x0 - w * pad, x1 + w * pad
    y0, y1 = y0 - h * pad, y1 + h * pad
    x0, y0 = int(max(0, x0)), int(max(0, y0))
    x1, y1 = int(min(W, x1)), int(min(H, y1))
    if x1 - x0 < min_side or y1 - y0 < min_side:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


class Person:
    def __init__(self, pts, W, H):
        self.pts = pts                     # (33, 3): x px, y px, visibility
        vis = pts[:, 2] >= VIS
        self.visible = int(vis.sum())
        P = lambda *ids: np.array([pts[i, :2] for i in ids if vis[i]])
        # --- torso (breathing area)
        sh, hp = P(L_SH, R_SH), P(L_HIP, R_HIP)
        self.torso = None
        if len(sh) == 2 and len(hp):
            self.torso = _box(np.vstack([sh, hp]), 0.12, W, H)
        elif len(sh) == 2:                              # hips out of view: extend down from the shoulders
            span = np.linalg.norm(sh[0] - sh[1])
            self.torso = _box(np.vstack([sh, sh + [0, 1.3 * span]]), 0.1, W, H)
        elif len(hp) == 2:                              # shoulders out of view: extend up from the hips
            span = np.linalg.norm(hp[0] - hp[1])
            self.torso = _box(np.vstack([hp, hp - [0, 1.6 * span]]), 0.1, W, H)
        # --- head + skin patches (pulse area): forehead, cheeks, neck, hands
        f = P(*FACE_PTS)
        self.head, self.skin = None, []
        span = np.linalg.norm(sh[0] - sh[1]) if len(sh) == 2 else None
        extra = []
        # neck: between the chin/mouth and the middle of the shoulders
        mouth = P(MOUTH_L, MOUTH_R)
        if len(sh) == 2 and (len(mouth) or len(f) >= 3):
            top = mouth.mean(axis=0) if len(mouth) else f.mean(axis=0)
            base = sh.mean(axis=0)
            nw = max(10, 0.28 * span)
            y0, y1 = top[1] + 0.25 * (base[1] - top[1]), base[1] - 0.15 * (base[1] - top[1])
            if y1 - y0 > 4:
                extra.append(("neck", (int(base[0] - nw / 2), int(y0), int(nw), int(y1 - y0))))
        # hands: around each visible wrist + its finger points
        for hand, wrist, fingers in (("left hand", 15, (17, 19, 21)), ("right hand", 16, (18, 20, 22))):
            pts_h = P(wrist, *fingers)
            if len(pts_h) >= 2:
                side = max(12, 0.22 * span) if span else max(12, 1.6 * np.ptp(pts_h, axis=0).max())
                c = pts_h.mean(axis=0)
                extra.append((hand, (int(c[0] - side / 2), int(c[1] - side / 2), int(side), int(side))))
        extra = [(n, (max(0, x), max(0, y), min(W - max(0, x), w), min(H - max(0, y), h))) for n, (x, y, w, h) in extra]
        self.skin_extra = [(n, b) for n, b in extra if b[2] > 4 and b[3] > 4]
        if len(f) >= 3:
            cx, cy = f[:, 0].mean(), f[:, 1].mean()
            spread = max(np.ptp(f[:, 0]), np.ptp(f[:, 1]), 12)
            s = spread * 1.25
            self.head = _box(np.array([[cx - s / 2, cy - s * 0.6], [cx + s / 2, cy + s * 0.55]]), 0.0, W, H)
            if self.head:
                x, y, w, h = self.head
                self.skin = [(n, b) for n, b in (
                    ("forehead", (x + int(w * 0.25), y + int(h * 0.05), int(w * 0.5), int(h * 0.22))),
                    ("left cheek", (x + int(w * 0.12), y + int(h * 0.45), int(w * 0.22), int(h * 0.22))),
                    ("right cheek", (x + int(w * 0.66), y + int(h * 0.45), int(w * 0.22), int(h * 0.22))),
                ) if b[2] > 2 and b[3] > 2]
        # named skin patches: forehead, cheeks, neck, hands (whatever is visible)
        self.skin = self.skin + self.skin_extra


class PersonFinder:
    """MediaPipe Pose (up to 4 people). `available` is False if the model file is missing."""

    def __init__(self, max_people=4):
        self.det = None
        if not os.path.exists(POSE_MODEL):
            return
        try:
            os.environ.setdefault("GLOG_minloglevel", "2")
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
            opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=POSE_MODEL), num_poses=max_people,
                                                min_pose_detection_confidence=0.4, min_pose_presence_confidence=0.4)
            with _quiet_stderr():
                self.det = vision.PoseLandmarker.create_from_options(opts)
                self.det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((64, 64, 3), np.uint8)))
            self._mp = mp
        except Exception:
            self.det = None

    @property
    def available(self):
        return self.det is not None

    def find(self, frame_bgr):
        if self.det is None or frame_bgr is None or frame_bgr.ndim != 3:
            return []
        H, W = frame_bgr.shape[:2]
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_bgr[..., ::-1]))
        res = self.det.detect(img)
        people = []
        for lm in res.pose_landmarks:
            pts = np.array([[p.x * W, p.y * H, getattr(p, "visibility", 1.0) or 0.0] for p in lm], np.float32)
            person = Person(pts, W, H)
            if person.visible >= 4:
                people.append(person)
        return people


def overlap(a, b):
    """Fraction of box a covered by box b."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return ix * iy / float(max(1, aw * ah))
