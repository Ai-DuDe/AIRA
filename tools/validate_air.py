"""Validate AIRA on the real AIR-125 infant videos (Step 11 of the methodology).

    python tools/validate_air.py --root data/air125/AIR_125            # all clips with video
    python tools/validate_air.py --root data/air125/AIR_125 --limit 10

For each clip it reports:
  * breathing-rate error vs. the annotated breaths (MAE, bpm) - comparable to the
    published AIRFlowNet number (~2.9 bpm MAE on AIR-125);
  * whether AIRA raised any PRE-ALERT / ALARM. These clips are normal breathing,
    so any alarm where the labels show no long pause is a FALSE alarm.
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import h5py
import numpy as np

from aira.autoroi import find_breathing_zones
from aira.config import AiraConfig
from aira.monitor import TIER_ALARM, TIER_PREALERT, BabyMonitor

WORK_W = 640


def load_clip(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (WORK_W, int(g.shape[0] * WORK_W / g.shape[1])), interpolation=cv2.INTER_AREA)
        h, w = g.shape
        g[: int(0.07 * h), : int(0.45 * w)] = 0  # timestamp overlay ticks at 1 Hz = looks like 60 bpm
        frames.append(g)
    return frames, fps


def gt_breaths(h5path, fps):
    with h5py.File(h5path, "r") as h:
        imp = np.asarray(h["impulse"])
    return np.flatnonzero(imp > 0.5) / fps


def rate_from(times, t0, t1):
    t = np.asarray([x for x in times if t0 <= x <= t1])
    if len(t) < 3:
        return None
    return 60.0 / float(np.median(np.diff(t)))


def run_clip(args):
    mp4, mode = args
    frames, fps = load_clip(mp4)
    gt = gt_breaths(mp4[:-4] + ".hdf5", fps)
    dur = len(frames) / fps
    if mode == "auto":
        zs, dbg = find_breathing_zones(frames[: int(20 * fps)], fps, return_debug=True)
        if zs:
            best = max(range(len(zs)), key=lambda i: dbg["blobs"][i]["coherence"] * np.sqrt(dbg["blobs"][i]["blocks"]))
            zone = zs[best]
        else:
            zone = (0, 0, frames[0].shape[1], frames[0].shape[0])
    else:
        zone = (0, 0, frames[0].shape[1], frames[0].shape[0])
    x, y, w, h = zone
    cfg = AiraConfig(fps=fps)
    m = BabyMonitor(os.path.relpath(mp4), cfg)
    detected, tiers, statuses = [], [], []
    seen = 0
    for i, f in enumerate(frames):
        t = i / fps
        snap = m.update(f[y:y + h, x:x + w], t)
        if len(m.breath_times) and (seen == 0 or m.breath_times[-1] != detected[-1]):
            detected.append(m.breath_times[-1])
            seen += 1
        tiers.append(snap["tier"])
        statuses.append(snap["status"])
    tiers = np.asarray(tiers)
    ts = np.arange(len(tiers)) / fps
    onset = lambda lvl: ts[1:][(tiers[1:] >= lvl) & (tiers[:-1] < lvl)]
    gt_gap = float(np.max(np.diff(np.r_[0.0, gt, dur]))) if len(gt) else dur
    t0 = cfg.warmup_s + 1
    return dict(
        clip=os.path.relpath(mp4), fps=fps, zone=zone, dur=dur,
        gt_rate=rate_from(gt, t0, dur), est_rate=rate_from(detected, t0, dur),
        gt_max_gap=gt_gap, n_gt=len(gt), n_det=len(detected),
        prealerts=len(onset(TIER_PREALERT)), alarms=len(onset(TIER_ALARM)),
        poor=float(np.mean([s == "POOR SIGNAL" for s in statuses])),
        calibrated=m.calibrated,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/air125/AIR_125")
    ap.add_argument("--mode", choices=["auto", "full"], default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    clips = sorted(glob.glob(os.path.join(a.root, "S0*", "*.mp4")))
    clips = [c for c in clips if os.path.exists(c[:-4] + ".hdf5")]
    if a.limit:
        clips = clips[: a.limit]
    print(f"{len(clips)} clips, mode={a.mode}")
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run_clip, [(c, a.mode) for c in clips]))

    import csv
    out = f"air125_results_{a.mode}.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(res[0].keys()))
        wr.writeheader()
        wr.writerows(res)

    ok = [r for r in res if r["gt_rate"] and r["est_rate"]]
    err = np.array([abs(r["est_rate"] - r["gt_rate"]) for r in ok])
    no_rate = len(res) - len(ok)
    print(f"\nbreathing rate: MAE {err.mean():.2f} bpm, median error {np.median(err):.2f} bpm, "
          f"within 5 bpm: {np.mean(err <= 5) * 100:.0f}%  (n={len(ok)}, no estimate on {no_rate})")
    # per subject
    for s in sorted({r['clip'].split(os.sep)[-2] for r in ok}):
        e = [abs(r["est_rate"] - r["gt_rate"]) for r in ok if r["clip"].split(os.sep)[-2] == s]
        print(f"   {s}: MAE {np.mean(e):.2f} bpm over {len(e)} clips")
    clean = [r for r in res if r["gt_max_gap"] < 10]
    fa = sum(r["alarms"] > 0 for r in clean)
    pa = sum(r["prealerts"] > 0 for r in clean)
    mins = sum(r["dur"] for r in clean) / 60
    print(f"\nclips with no labelled pause >10 s: {len(clean)} ({mins:.0f} min)")
    print(f"   clips with a false ALARM: {fa}   with a PRE-ALERT: {pa}")
    print(f"   clips calibrated: {sum(r['calibrated'] for r in res)}/{len(res)}, "
          f"mean POOR SIGNAL time {np.mean([r['poor'] for r in res]) * 100:.0f}%")
    print(f"\nper-clip details -> {out}")


if __name__ == "__main__":
    main()
