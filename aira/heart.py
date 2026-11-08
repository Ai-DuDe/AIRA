"""Experimental: heart rate from skin colour (remote photoplethysmography, rPPG).

Each heartbeat pushes a little more blood into the skin, which absorbs a little more
green light. The change is tiny (~0.1-1% of the pixel value) but rhythmic, so averaging
many skin pixels and filtering to the heart-rate band makes it measurable.

Where is the skin? First choice: the PERSON found by MediaPipe Pose (works with partly
visible bodies) - their forehead, cheeks, neck and hands are averaged into one signal.
Second: a MediaPipe face box. Last resort, when no person is found:
Colour alone is not enough (wood, beige walls, a pink sheet are
"skin-coloured"). So a region is used only if it is BOTH:
  * skin-coloured (YCrCb thresholds), and
  * actually PULSING: its own colour signal has a clear heartbeat rhythm,
    and that rhythm agrees with its neighbours.
Walls and sheets have the colour but no pulse, so they are rejected ("pulse map").
If a MediaPipe face-detector model is present (models/blaze_face_short_range.tflite),
the search is additionally limited to detected faces.
This never uses the breathing (apnea) box - it is a separate search.

Pipeline (all classical, explainable):
  1. every frame: shrink the image to a grid of small blocks (mean colour per block);
  2. once a second: per block, POS pulse signal (Wang et al., IEEE TBME 2017) over the
     last ~12 s -> spectrum -> "how clean is the heartbeat peak";
  3. keep skin-coloured blocks with a clean peak at the common rate (+ inside a face, if known);
  4. average those blocks -> POS -> band-pass -> spectrum peak = heart rate;
  5. quality = how much of the band's power sits in that one peak; low quality -> no number.

Limits (say these in the report):
  * needs COLOUR video - it does not work in IR night vision (grayscale);
  * needs steady lighting and little movement;
  * not validated against a reference ECG/oximeter here. Treat the number as a demo.
"""
import os
from collections import deque

import cv2
import numpy as np
from scipy import signal as sps

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "blaze_face_short_range.tflite")


class _quiet_stderr:
    """Temporarily silence native (C/C++) writes to stderr."""

    def __enter__(self):
        import sys
        try:
            sys.stderr.flush()
            self.fd = os.dup(2)
            self.null = os.open(os.devnull, os.O_WRONLY)
            os.dup2(self.null, 2)
        except Exception:
            self.fd = None
        return self

    def __exit__(self, *a):
        if self.fd is not None:
            os.dup2(self.fd, 2)
            os.close(self.fd)
            os.close(self.null)


class FaceFinder:
    """Optional MediaPipe face detector. Returns face boxes, or None if unavailable."""

    def __init__(self):
        self.det = None
        if not os.path.exists(MODEL):
            return
        try:
            os.environ.setdefault("GLOG_minloglevel", "2")       # keep MediaPipe's start-up logging quiet
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            import mediapipe as mp
            from mediapipe.tasks.python import vision, BaseOptions
            opts = vision.FaceDetectorOptions(base_options=BaseOptions(model_asset_path=MODEL), min_detection_confidence=0.5)
            with _quiet_stderr():   # MediaPipe's C++ engine prints a page of start-up logs
                self.det = vision.FaceDetector.create_from_options(opts)
                self.det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((64, 64, 3), np.uint8)))
            self._mp = mp
        except Exception:
            self.det = None

    @property
    def available(self):
        return self.det is not None

    def faces(self, frame_bgr):
        if self.det is None:
            return None
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_bgr[..., ::-1]))
        res = self.det.detect(img)
        return [(d.bounding_box.origin_x, d.bounding_box.origin_y, d.bounding_box.width, d.bounding_box.height) for d in res.detections]


def _pos(C, fps):
    """POS pulse signal for colour traces. C: (N, ..., 3) RGB means over time -> (N, ...)."""
    Cn = C / (C.mean(axis=0, keepdims=True) + 1e-9)
    R, G, B = Cn[..., 0], Cn[..., 1], Cn[..., 2]
    S1, S2 = G - B, G + B - 2 * R
    a = S1.std(axis=0, keepdims=True) / (S2.std(axis=0, keepdims=True) + 1e-12)
    h = S1 + a * S2
    return h - h.mean(axis=0, keepdims=True)


def _union(boxes):
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


class HeartRateEstimator:
    GRID_W = 40       # columns of blocks the frame is reduced to
    block_conc = 0.6  # a block must have a clean heartbeat peak (60% of its band power)
    min_blocks = 3    # ...and at least 3 touching blocks must agree on the rate

    def __init__(self, fps, band_bpm=(42, 180), window_s=12.0, min_quality=0.35, use_face=True):
        self.fps = fps
        self.band = (band_bpm[0] / 60.0, band_bpm[1] / 60.0)
        self.grids = deque(maxlen=int(window_s * fps))
        self.rgb = self.grids                 # kept for older callers (length = samples collected)
        self.min_quality = min_quality
        self.box = None          # region the pulse is read from, full-frame pixels (x, y, w, h)
        self.mask = None         # selected blocks (grid_h x grid_w bool)
        self.bpm = None
        self.quality = 0.0
        self.colour = True
        self.wave = None         # last few seconds of the pulse signal (for the on-screen trace)
        self.face_box = None
        self.faces = FaceFinder() if use_face else None
        from .person import PersonFinder
        self.people = PersonFinder() if use_face else None
        self.patches = []        # skin patches of the found person (forehead, cheeks, neck, hands)
        self.roi_trace = deque(maxlen=int(window_s * fps))   # mean skin colour over those patches
        self.source = None       # "person", "face" or "pulse map" - where the pulse is read
        self._last_patches_t = -1e9
        # LOCK: once a spot gives a clear pulse a few times in a row, stop searching and stay on it.
        self.locked = False
        self._good_streak = 0
        self._bad_since = None
        self.last_good_bpm, self.last_good_t = None, -1e9
        self._t = 0.0
        self._n = 0
        self._shape = None

    @staticmethod
    def _skin_blocks(grid_bgr):
        ycc = cv2.cvtColor(np.clip(grid_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2YCrCb)
        y, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
        return (y > 40) & (cr >= 133) & (cr <= 178) & (cb >= 80) & (cb <= 132)

    def update(self, frame_bgr, t=None, roi=None, patches=None):
        """Feed one full colour frame. `roi` (x, y, w, h) forces the region (used in tests)."""
        self._n += 1
        if frame_bgr.ndim == 2:
            self.colour = False
            return None
        sub = frame_bgr[::8, ::8].astype(np.int16)
        lab = cv2.cvtColor(np.clip(sub, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        if lab[:, 1].std() + lab[:, 2].std() < 4:                 # night vision / grey picture: no skin colour
            self.colour, self.bpm, self.box, self.wave = False, None, None, None   # IR / grayscale camera
            return None
        self.colour = True
        H, W = frame_bgr.shape[:2]
        gw = self.GRID_W
        gh = max(4, int(round(gw * H / W)))
        if self._shape != (H, W):
            self._shape, self.grids = (H, W), deque(maxlen=self.grids.maxlen)
            self.rgb = self.grids
        self.grids.append(cv2.resize(frame_bgr, (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32))
        t = (self._n / self.fps) if t is None else t
        self._t = t
        # 0) tracked face patches from the landmark tracker (Face Mesh) - the best source
        if patches:
            self._ext = list(patches)
            self._ext_t = t
        if getattr(self, "_ext", None) and t - getattr(self, "_ext_t", -1e9) <= 6.0:
            self.source = "face mesh"
            self.patches = [b for _, b in self._ext]
            self.box = _union(self.patches)
            self.roi_trace.append(self._ext_colour(frame_bgr))
            if len(self.roi_trace) >= self.roi_trace.maxlen * 0.75 and self._n % max(1, int(self.fps / 2)) == 0:
                self._estimate(np.asarray(self.roi_trace))
            return self.bpm
        if getattr(self, "_ext", None):           # face lost for > 3 s: start fresh with the fallbacks
            self._ext = None
            self.roi_trace.clear()
            self.patches = []
        # 1) search for skin (person / face) twice a second - but only until a good spot is LOCKED
        if roi is None and not self.locked and self._n % max(1, int(self.fps / 2)) == 0:
            self._find_patches(frame_bgr)
        if self.patches and roi is None:
            self._last_patches_t = t
            self.roi_trace.append(self._patch_colour(frame_bgr))
            self.box = _union(self.patches)
            if len(self.roi_trace) >= self.roi_trace.maxlen * 0.75 and self._n % max(1, int(self.fps / 2)) == 0:
                self._estimate(np.asarray(self.roi_trace))
            return self.bpm
        if t - self._last_patches_t > 2.0 and not self.locked:
            self.roi_trace.clear()           # person lost for a while: start that signal again
        self.source = "pulse map"
        if roi is not None:
            x, y, w, h = roi
            m = np.zeros((gh, gw), bool)
            m[int(y * gh / H):max(int(y * gh / H) + 1, int((y + h) * gh / H)), int(x * gw / W):max(int(x * gw / W) + 1, int((x + w) * gw / W))] = True
            self.mask, self.box = m, roi
        elif not self.locked and self._n % max(1, int(self.fps)) == 0 and len(self.grids) >= 6 * self.fps:
            if self.faces is not None and self.faces.available:
                try:
                    f = self.faces.faces(frame_bgr)
                    self.face_box = max(f, key=lambda b: b[2] * b[3]) if f else None
                except Exception:
                    self.face_box = None
            self._select(H, W, gh, gw)
        if self.mask is None:
            self.bpm, self.wave = None, None
            return None
        if len(self.grids) >= self.grids.maxlen * 0.75 and self._n % max(1, int(self.fps / 2)) == 0:
            self._estimate(np.asarray(self.grids)[..., ::-1][:, self.mask].mean(axis=1))
        return self.bpm

    # ---- person / face -> skin patches -------------------------------------------------
    def _find_patches(self, frame_bgr):
        patches, src = [], None
        if self.people is not None and self.people.available:
            try:
                people = self.people.find(frame_bgr)
                if people:
                    best = max(people, key=lambda p: (len(p.skin), p.visible))
                    patches, src = best.skin, "person"
            except Exception:
                pass
        if not patches and self.faces is not None and self.faces.available:
            try:
                f = self.faces.faces(frame_bgr)
                if f:
                    x, y, w, h = max(f, key=lambda b: b[2] * b[3])
                    patches = [("forehead", (x + int(w * 0.25), y + int(h * 0.05), int(w * 0.5), int(h * 0.22))),
                               ("left cheek", (x + int(w * 0.12), y + int(h * 0.45), int(w * 0.22), int(h * 0.22))),
                               ("right cheek", (x + int(w * 0.66), y + int(h * 0.45), int(w * 0.22), int(h * 0.22)))]
                    src = "face"
            except Exception:
                pass
        H, W = frame_bgr.shape[:2]
        found = {}
        for name, (x, y, w, h) in patches:
            x, y = max(0, x), max(0, y)
            w, h = min(w, W - x), min(h, H - y)
            if w > 3 and h > 3:
                found[name] = (x, y, w, h)
        if found:
            self._seen_n = self._n
            self.source = src
        elif self._n - getattr(self, "_seen_n", -10 ** 9) <= 3 * self.fps:
            return                                   # briefly lost: keep the last patches for up to ~3 s
        old = getattr(self, "_named", {})
        named = {}
        for name, box in found.items():
            prev = old.get(name)
            # keep the exact old box (and its skin mask) unless that patch really moved:
            # re-detection jitters by a few pixels, which would add a fake rhythm
            if prev is not None and max(abs(box[0] - prev["box"][0]), abs(box[1] - prev["box"][1])) < 0.15 * max(4, min(box[2], box[3])):
                named[name] = prev
                continue
            x, y, w, h = box
            crop = frame_bgr[y:y + h, x:x + w]
            ycc = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
            mask = (ycc[..., 1] >= 133) & (ycc[..., 1] <= 178) & (ycc[..., 2] >= 77) & (ycc[..., 2] <= 135)
            if mask.mean() < 0.3:
                continue                             # hardly any skin in this patch (sleeve, blanket...)
            named[name] = {"box": box, "mask": mask, "ref": None, "last": None}
        self._named = named
        self.patches = [v["box"] for v in named.values()]

    def _ext_colour(self, frame_bgr):
        """Mean skin colour over the tracked forehead/cheek patches (skin pixels only)."""
        tot, n = np.zeros(3), 0
        for _, (x, y, w, h) in self._ext:
            crop = frame_bgr[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            ycc = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
            m = (ycc[..., 1] >= 133) & (ycc[..., 1] <= 178) & (ycc[..., 2] >= 77) & (ycc[..., 2] <= 135)
            px = crop[m] if m.mean() >= 0.3 else crop.reshape(-1, 3)
            tot += px.reshape(-1, 3).sum(axis=0)
            n += px.reshape(-1, 3).shape[0]
        if not n:
            return self.roi_trace[-1] if self.roi_trace else np.array([128.0, 128.0, 128.0])
        return (tot / n)[::-1]

    def _patch_colour(self, frame_bgr):
        """One sample of skin colour, made continuous across patches coming and going:
        each named patch is tracked relative to its own level, then patches are averaged."""
        vals, weights = [], []
        for name, p in self._named.items():
            x, y, w, h = p["box"]
            crop = frame_bgr[y:y + h, x:x + w]
            if crop.shape[:2] != p["mask"].shape:
                continue
            rgb = crop[p["mask"]].reshape(-1, 3).mean(axis=0)[::-1].astype(np.float64)
            if p["ref"] is None:
                # new (or moved) patch: scale it so it joins the signal where it currently is
                p["ref"] = rgb / (self._level if getattr(self, "_level", None) is not None else rgb)
            vals.append(rgb / p["ref"])
            weights.append(p["mask"].sum())
        if not vals:
            return self.roi_trace[-1] if self.roi_trace else np.array([128.0, 128.0, 128.0])
        level = np.average(np.array(vals), axis=0, weights=np.array(weights, float))
        self._level = level
        return level

    # ---- the pulse map: which blocks are skin AND pulsing ------------------------------
    def _select(self, H, W, gh, gw):
        G = np.asarray(self.grids)[..., ::-1]                   # (N, gh, gw, RGB)
        skin = self._skin_blocks(G[-1][..., ::-1])
        if self.face_box is not None:
            fx, fy, fw, fh = self.face_box
            face = np.zeros_like(skin)
            face[max(0, int(fy * gh / H)):int((fy + fh) * gh / H) + 1, max(0, int(fx * gw / W)):int((fx + fw) * gw / W) + 1] = True
            skin &= face
        if skin.sum() < 2:
            self.mask, self.box = None, None
            return
        h = _pos(G, self.fps)                                    # (N, gh, gw)
        lo, hi = self.band[0], min(self.band[1], 0.45 * self.fps)
        sos = sps.butter(3, [lo, hi], btype="band", fs=self.fps, output="sos")
        x = sps.sosfiltfilt(sos, h, axis=0)
        N = x.shape[0]
        F = np.abs(np.fft.rfft(x * np.hanning(N)[:, None, None], n=4 * N, axis=0)) ** 2
        f = np.fft.rfftfreq(4 * N, 1 / self.fps)
        band = (f >= lo) & (f <= hi)
        Fb, fb = F[band], f[band]
        pk = Fb.argmax(axis=0)                                   # (gh, gw)
        width = max(1, int(0.1 / (fb[1] - fb[0])))
        k = np.arange(Fb.shape[0])[:, None, None]
        conc = (Fb * (np.abs(k - pk[None]) <= width)).sum(axis=0) / (Fb.sum(axis=0) + 1e-12)
        pulsing = skin & (conc >= self.block_conc)
        if pulsing.sum() < self.min_blocks:
            self.mask, self.box = None, None
            return
        # the heart rate most pulsing blocks agree on
        votes = np.bincount(pk[pulsing], weights=conc[pulsing], minlength=len(fb))
        common = int(np.convolve(votes, np.ones(2 * width + 1), "same").argmax())
        mask = pulsing & (np.abs(pk - common) <= width)
        # keep only the biggest connected group: a real face/arm is one patch, chance noise is scattered
        n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        if n > 1:
            sizes = np.bincount(lab.ravel())[1:]
            mask = lab == (1 + int(sizes.argmax()))
        if mask.sum() < self.min_blocks:
            self.mask, self.box = None, None
            return
        ys, xs = np.nonzero(mask)
        self.mask = mask
        self.box = (int(xs.min() * W / gw), int(ys.min() * H / gh), int((xs.max() - xs.min() + 1) * W / gw), int((ys.max() - ys.min() + 1) * H / gh))

    # ---- heart rate from the selected blocks -------------------------------------------
    def _estimate(self, C):
        """C: (N, 3) RGB over the chosen skin, oldest first."""
        N = C.shape[0]
        L = max(2, int(1.6 * self.fps))                          # POS sliding window
        Hs = np.zeros(N)
        for n in range(L, N + 1):
            h = _pos(C[n - L:n], self.fps)
            Hs[n - L:n] += h
        lo, hi = self.band[0], min(self.band[1], 0.45 * self.fps)
        sos = sps.butter(3, [lo, hi], btype="band", fs=self.fps, output="sos")
        f = np.fft.rfftfreq(8 * N, 1 / self.fps)
        band = (f >= lo) & (f <= hi)
        fb = f[band]

        def spectrum(sig):
            x = sps.sosfiltfilt(sos, sig - sig.mean())
            Fb = (np.abs(np.fft.rfft(x * np.hanning(N), n=8 * N)) ** 2)[band]
            k = int(np.argmax(Fb))
            q = Fb[np.abs(fb - fb[k]) <= 0.12].sum() / (Fb.sum() + 1e-12)
            return x, Fb, q

        # two classic rPPG methods; keep whichever gives the cleaner pulse this time
        Cn = C / (C.mean(axis=0, keepdims=True) + 1e-9)
        Xc = 3 * Cn[:, 0] - 2 * Cn[:, 1]
        Yc = 1.5 * Cn[:, 0] + Cn[:, 1] - 1.5 * Cn[:, 2]
        xc, yc = sps.sosfiltfilt(sos, Xc - Xc.mean()), sps.sosfiltfilt(sos, Yc - Yc.mean())
        chrom = xc - (xc.std() / (yc.std() + 1e-12)) * yc
        x, Fb, q = max((spectrum(Hs), spectrum(chrom)), key=lambda v: v[2])
        k = int(np.argmax(Fb))
        sm = getattr(self, "smooth", None)
        if sm is not None:                                       # prefer a strong peak near the current rate
            nearby = np.abs(fb * 60 - sm) <= 12
            if nearby.any():
                j = int(np.argmax(np.where(nearby, Fb, 0)))
                if Fb[j] >= 0.4 * Fb[k]:
                    k = j
        near = np.abs(fb - fb[k]) <= 0.12
        self.quality = float(Fb[near].sum() / (Fb.sum() + 1e-12))
        self.bpm = float(fb[k] * 60) if self.quality >= self.min_quality else None
        self._update_lock()
        w = x[-int(5 * self.fps):]
        self.wave = w / (np.abs(w).max() + 1e-12)

    def _update_lock(self):
        t = self._t
        # steady number: needs 3 agreeing estimates to start, ignores single spikes, eases toward changes
        if not hasattr(self, "smooth"):
            self.smooth, self._cand = None, []
        if self.bpm is not None:
            if self.smooth is None or abs(self.bpm - self.smooth) > 12:
                self._cand = [b for b in self._cand if abs(b - self.bpm) <= 6] + [self.bpm]
                if len(self._cand) >= (3 if self.smooth is None else 4):
                    self.smooth, self._cand = float(np.median(self._cand)), []
                else:
                    self.bpm = None                              # a spike: don't use it
            else:
                self._cand = []
                self.smooth = 0.8 * self.smooth + 0.2 * self.bpm
        good = self.bpm is not None and self.quality >= 0.5
        if self.bpm is not None:
            self.last_good_bpm, self.last_good_t = self.bpm, t
        if good:
            self._good_streak += 1
            self._bad_since = None
            if not self.locked and self._good_streak >= 4:      # ~1.5 s of clean pulse: lock this spot
                self.locked = True
        else:
            self._good_streak = 0
            if self.locked:
                self._bad_since = t if self._bad_since is None else self._bad_since
                if t - self._bad_since > 15.0:                   # lost for 15 s: search again
                    self.locked, self._bad_since = False, None

    @property
    def display_bpm(self):
        """What the screen shows: the current rate, or the last good one for up to 10 s of dropout."""
        sm = getattr(self, "smooth", None)
        if sm is not None and self._t - self.last_good_t <= 15.0:
            return sm
        return None

    @property
    def progress(self):
        """0..1 while the first ~9 s of skin colour are being collected."""
        n = len(self.roi_trace) if self.patches else len(self.grids)
        return min(1.0, n / (self.grids.maxlen * 0.75))


def apply_to(mon, est, note=None):
    """Copy what the dashboard's heart panel shows onto a monitor (None est = clear)."""
    mon.heart_bpm = mon.heart_wave = None
    mon.heart_locked = False
    mon.heart_quality, mon.heart_progress = 0.0, 0.0
    mon.heart_note = note
    if est is None:
        return
    if not est.colour:
        mon.heart_note = "Needs a colour camera (IR night vision has no colour)"
        return
    if est.box is None and not est.locked:
        mon.heart_note = note or "Looking for a pulse (face or skin in view)..."
        return
    mon.heart_progress = est.progress
    mon.heart_quality = est.quality
    mon.heart_wave = est.wave
    mon.heart_locked = est.locked
    if est.display_bpm:
        mon.heart_bpm = est.display_bpm
    else:
        mon.heart_note = note or ("Measuring..." if est.progress < 1 else "Pulse unclear - keep still, more light")


class HeartWorker:
    """Runs heart-rate detection in its own thread, on its own region (skin found by colour),
    completely separate from the breathing/apnea analysis and its breathing-area box.

    The camera loop only hands over frames (non-blocking); every frame is processed in order
    so the pulse signal keeps its timing. If the worker ever falls 3 s behind, the oldest
    frames are dropped rather than slowing the camera loop.
    """

    def __init__(self, estimator):
        import queue
        import threading
        self.est = estimator
        self.q = queue.Queue(maxsize=max(10, int(estimator.fps * 3)))
        self._stop = False
        threading.Thread(target=self._run, daemon=True, name="heart-rate").start()

    def submit(self, frame_bgr, t, patches=None):
        import queue
        item = (frame_bgr.copy(), t, patches)   # copy: the caller may draw on its frame afterwards
        try:
            self.q.put_nowait(item)
        except queue.Full:
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            self.q.put_nowait(item)

    def _run(self):
        while not self._stop:
            frame, t, patches = self.q.get()
            try:
                self.est.update(frame, t, patches=patches)
            except Exception:
                pass

    def stop(self):
        self._stop = True
