"""Automatic ROI: find every breathing chest in the frame, with no manual boxes and no trained model.

Idea: a breathing chest is the only thing in a cot that moves rhythmically at
breathing speed (18-90 bpm). So:
  1. watch the whole frame for a few seconds (dense optical flow),
  2. average the motion in small blocks,
  3. score each block by how much *rhythmic breathing-band* motion it has
     compared with the static background,
  4. group strong neighbouring blocks into blobs; one blob = one baby,
  5. pad each blob into a zone.

This works on swaddled / blanket-covered infants where pose models (MediaPipe)
fail, because it only needs the blanket to move with the chest.
Limitation: a baby that is NOT breathing during the scan is not found, so
zones are kept once found and the scan is repeated periodically to add new ones.
"""
import cv2
import numpy as np
from scipy import signal as sps


def _prep(frame, width):
    g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    s = min(1.0, width / g.shape[1])
    if s < 1:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(g, (5, 5), 1.0), s


def find_breathing_zones(frames, fps, band=(0.25, 1.5), block=12, work_width=480, integrate_tau_s=4.0,
                         noise_k=0.8, min_concentration=0.6, min_blocks=3, min_blob_concentration=0.7,
                         pad=0.4, min_side=64, max_zones=0, flicker_frac=0.3, min_rel_score=0.5, min_blob_blocks=5, rank_strength=True,
                         return_debug=False):
    """frames: list of BGR/gray frames (>= ~10 s). Returns list of (x, y, w, h) in full-res pixels."""
    prev, s = _prep(frames[0], work_width)
    H, W = prev.shape
    bh, bw = H // block, W // block
    series = []
    for f in frames[1:]:
        g, _ = _prep(f, work_width)
        flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prev = g
        fl = flow[:bh * block, :bw * block].reshape(bh, block, bw, block, 2).mean(axis=(1, 3))
        series.append(fl)
    X = np.asarray(series)                                  # (T, bh, bw, 2)
    X = X - X.mean(axis=0)
    # velocity -> displacement (same leaky integrator as the monitor); suppresses fast artifacts
    leak = np.exp(-1.0 / (integrate_tau_s * fps))
    D = sps.lfilter([1.0], [1.0, -leak], X, axis=0)
    sos = sps.butter(2, band, btype="band", fs=fps, output="sos")
    B = sps.sosfiltfilt(sos, D, axis=0)                     # offline scan, so zero-phase is fine here

    # dominant motion direction per block (2x2 PCA) -> 1-D breathing signal per block
    C = np.einsum("tijk,tijl->ijkl", B, B) / len(B)
    vals, vecs = np.linalg.eigh(C)
    v = vecs[..., :, -1]
    sig = np.einsum("tijk,ijk->tij", B, v)
    power = np.sqrt(vals[..., -1])                          # breathing-band motion strength

    # spectral concentration: is the motion one clean rhythm, or broadband noise?
    F = np.abs(np.fft.rfft(sig, axis=0)) ** 2
    freqs = np.fft.rfftfreq(len(sig), 1 / fps)
    inband = (freqs >= band[0]) & (freqs <= band[1])
    Fb = F[inband]
    peak = Fb.argmax(axis=0)
    k = np.arange(Fb.shape[0])[:, None, None]
    near = np.abs(k - peak[None]) <= max(1, int(0.08 * len(sig) / fps))
    conc = (Fb * near).sum(axis=0) / (Fb.sum(axis=0) + 1e-12)

    # per-block noise from the >2.5 Hz band (no breathing there), scaled to the breathing bandwidth.
    # Each block is judged against its OWN noise, so a strong baby can't hide a weak one.
    nyq = fps / 2
    hp_hz = min(2.5, 0.8 * nyq)
    hsos = sps.butter(2, hp_hz, btype="high", fs=fps, output="sos")
    Hn = sps.sosfiltfilt(hsos, X, axis=0)
    noise = np.sqrt((Hn ** 2).sum(axis=-1).mean(axis=0) / 2)
    # convert to "noise in the breathing signal" (white-noise gain of integrate+band-pass vs high-pass)
    wn = np.random.default_rng(0).normal(0, 1, int(fps * 600))
    g_in = sps.sosfiltfilt(sos, sps.lfilter([1.0], [1.0, -leak], wn))[int(fps * 30):-int(fps * 30)].std()
    g_hf = sps.sosfiltfilt(hsos, wn)[int(fps * 30):-int(fps * 30)].std()
    noise *= g_in / g_hf
    snr = power / (noise + 1e-9)
    mask = (snr > noise_k) & (conc >= min_concentration)
    # Camera flicker / auto-exposure / compression "breathing": the same rhythm shows up all
    # over the picture. A real chest is a small patch. Drop rhythms found in > flicker_frac of
    # ALL blocks (including the background).
    tol_bins = max(1, int(round(0.1 / (freqs[1] - freqs[0]))))
    strong = mask.copy()                      # only clear, strong rhythmic motion counts (not random noise)
    ring = np.zeros_like(mask)                # outer border of the picture (background, walls, cot edges)
    ring[:2, :] = ring[-2:, :] = ring[:, :2] = ring[:, -2:] = True
    for b in np.unique(peak[strong]):
        same = strong & (np.abs(peak.astype(int) - int(b)) <= tol_bins)
        # camera-wide shimmer: big share of the picture AND most of the border moves with it.
        # (a person close to the camera can fill a third of the picture, but not the whole border)
        if same.mean() > flicker_frac and same[ring].mean() > 0.4:
            mask &= ~same
    # Camera noise makes scattered single blocks look rhythmic; a chest/belly is a solid patch.
    # Keep only blocks with at least 2 rhythmic neighbours, so stray noise can't glue onto a chest.
    nb = cv2.filter2D(mask.astype(np.float32), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - mask
    mask &= nb >= 2
    # Group candidate blocks into blobs: neighbours (allowing a 1-block gap) join only if they
    # breathe at the same rate, so two babies lying side by side are never merged into one zone.
    df = freqs[1] - freqs[0]
    tol = max(1, int(round(0.1 / df)))                      # ~6 bpm
    labels = np.zeros(mask.shape, int)
    n = 1
    for a0, b0 in np.argwhere(mask):
        if labels[a0, b0]:
            continue
        labels[a0, b0] = n
        stack = [(a0, b0)]
        while stack:
            a, b = stack.pop()
            for da in range(-2, 3):
                for db in range(-2, 3):
                    na, nb = a + da, b + db
                    if (0 <= na < bh and 0 <= nb < bw and mask[na, nb] and not labels[na, nb]
                            and abs(int(peak[na, nb]) - int(peak[a, b])) <= tol):
                        labels[na, nb] = n
                        stack.append((na, nb))
        n += 1

    def concentration(x):
        P = np.abs(np.fft.rfft(x)) ** 2
        Pb = P[inband]
        p = Pb.argmax()
        w = max(1, int(0.08 * len(x) / fps))
        return Pb[max(0, p - w):p + w + 1].sum() / (Pb.sum() + 1e-12), freqs[inband][p]

    zones, blob_scores = [], []
    for i in range(1, n):
        members = np.argwhere(mask & (labels == i))
        if len(members) < min_blocks:
            continue
        # Coherence check: real breathing adds up across neighbouring blocks (same rhythm),
        # random noise that only looked rhythmic by chance does not.
        ref_i = max(members, key=lambda m: conc[m[0], m[1]] * snr[m[0], m[1]])
        ref = sig[:, ref_i[0], ref_i[1]]
        comb = np.zeros(len(sig))
        for a, b in members:
            x = sig[:, a, b]
            x = x / (x.std() + 1e-12)
            comb += x if (x @ ref) >= 0 else -x
        cc, f0 = concentration(comb)
        if cc < min_blob_concentration:
            continue
        blob_scores.append(dict(coherence=cc, bpm=f0 * 60, blocks=len(members),
                                strength=float(np.median(snr[members[:, 0], members[:, 1]]))))
        y0, x0 = members.min(axis=0)
        y1, x1 = members.max(axis=0) + 1
        zones.append((x0 * block / s, y0 * block / s, (x1 - x0) * block / s, (y1 - y0) * block / s))
    # Rank candidates (clean rhythm x strength x size) and drop the weak ones: crib bars, blanket
    # edges and camera noise can make small, faint "breathing" spots next to the real chest.
    if zones:
        score = [b["coherence"] * np.sqrt(b["blocks"]) * (b["strength"] if rank_strength else 1.0) for b in blob_scores]
        best = max(score)
        keep = [i for i, sc_ in enumerate(score) if sc_ >= min_rel_score * best and blob_scores[i]["blocks"] >= min_blob_blocks]
        zones = [zones[i] for i in keep]
        blob_scores = [blob_scores[i] for i in keep]
    zones = _pad_without_overlap(zones, pad, min_side, frames[0].shape[:2])
    # drop slivers (e.g. a strip at the frame edge): a chest zone is never this thin
    fh, fw = frames[0].shape[:2]
    keep = [i for i, (x, y, w, h) in enumerate(zones) if min(w, h) >= 0.5 * min_side and w * h >= 0.01 * fw * fh]
    zones = [zones[i] for i in keep]
    blob_scores = [blob_scores[i] for i in keep]
    if max_zones and len(zones) > max_zones:
        best = sorted(range(len(zones)), key=lambda i: -blob_scores[i]["coherence"] * blob_scores[i]["strength"] * np.sqrt(blob_scores[i]["blocks"]))[:max_zones]
        zones = [zones[i] for i in best]
        blob_scores = [blob_scores[i] for i in best]
    order = sorted(range(len(zones)), key=lambda i: (zones[i][1] // 100, zones[i][0]))  # reading order
    zones = [zones[i] for i in order]
    blob_scores = [blob_scores[i] for i in order]
    if return_debug:
        return zones, dict(power=power, snr=snr, conc=conc, mask=mask, block=block, scale=s, blobs=blob_scores)
    return zones


def _pad_without_overlap(raw, pad, min_side, shape):
    """Grow each blob box by `pad`, but never past the midline to a neighbouring blob."""
    fh, fw = shape
    out = []
    for i, (x, y, w, h) in enumerate(raw):
        cx, cy = x + w / 2, y + h / 2
        w2, h2 = max(min_side, w * (1 + pad)), max(min_side, h * (1 + pad))
        x0, x1, y0, y1 = cx - w2 / 2, cx + w2 / 2, cy - h2 / 2, cy + h2 / 2
        for j, (ox, oy, ow, oh) in enumerate(raw):
            if i == j:
                continue
            ocx, ocy = ox + ow / 2, oy + oh / 2
            # split along the axis on which the two blobs are further apart
            if abs(ocx - cx) >= abs(ocy - cy):
                mid = ((x + w) + ox) / 2 if ocx > cx else ((ox + ow) + x) / 2
                if ocx > cx:
                    x1 = min(x1, mid)
                else:
                    x0 = max(x0, mid)
            else:
                mid = ((y + h) + oy) / 2 if ocy > cy else ((oy + oh) + y) / 2
                if ocy > cy:
                    y1 = min(y1, mid)
                else:
                    y0 = max(y0, mid)
        x0, y0 = int(max(0, x0)), int(max(0, y0))
        x1, y1 = int(min(fw, x1)), int(min(fh, y1))
        out.append((x0, y0, x1 - x0, y1 - y0))
    return out


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(min(aw * ah, bw * bh) + 1e-9)


def _touch(a, b):
    return not (a[0] + a[2] < b[0] or b[0] + b[2] < a[0] or a[1] + a[3] < b[1] or b[1] + b[3] < a[1])


def merge_overlapping(zones, thr=0.3):
    zones = list(zones)
    merged = True
    while merged:
        merged = False
        for i in range(len(zones)):
            for j in range(i + 1, len(zones)):
                if _iou(zones[i], zones[j]) > thr or (thr == 0.0 and _touch(zones[i], zones[j])):
                    a, b = zones[i], zones[j]
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                    zones[i] = (x0, y0, x1 - x0, y1 - y0)
                    zones.pop(j)
                    merged = True
                    break
            if merged:
                break
    return zones


def new_zones(found, existing, thr=0.3):
    """Zones in `found` that don't overlap any existing zone (a newly placed baby)."""
    return [z for z in found if all(_iou(z, e) <= thr for e in existing)]
