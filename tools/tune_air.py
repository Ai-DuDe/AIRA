"""Tune on some AIR-125 subjects, test on the others (no peeking).

Step 1 (slow, once): auto-ROI + optical flow for every clip -> cached motion features.
Step 2 (fast): replay the monitor with different settings on the cache.

    python tools/tune_air.py cache
    python tools/tune_air.py sweep --subjects S01 S02          # pick settings here
    python tools/tune_air.py test  --subjects S03 S04 S05 --set noise_margin=3 gross_motion_ratio=10
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]

import argparse
import glob
import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from aira.autoroi import find_breathing_zones
from aira.config import AiraConfig
from aira.monitor import TIER_ALARM, TIER_PREALERT, BabyMonitor
from aira.motion import MotionExtractor
from validate_air import gt_breaths, load_clip, rate_from

ROOT = "data/air125/AIR_125"
CACHE = "data/cache"


def cache_clip(mp4):
    out = os.path.join(CACHE, mp4.split(os.sep)[-2] + "_" + os.path.basename(mp4)[:-4] + ".npz")
    if os.path.exists(out):
        return out
    frames, fps = load_clip(mp4)
    zs, dbg = find_breathing_zones(frames[: int(20 * fps)], fps, return_debug=True)
    if zs:
        b = max(range(len(zs)), key=lambda i: dbg["blobs"][i]["coherence"] * np.sqrt(dbg["blobs"][i]["blocks"]))
        zone = zs[b]
    else:
        zone = (0, 0, frames[0].shape[1], frames[0].shape[0])
    x, y, w, h = zone
    cfg = AiraConfig(fps=fps)
    me = MotionExtractor(cfg.max_roi_side, cfg.grid)
    feats, energy = [], []
    for f in frames:
        o = me.step(f[y:y + h, x:x + w])
        if o is not None:
            feats.append(o[0])
            energy.append(o[1])
    np.savez(out, feats=np.asarray(feats), energy=np.asarray(energy), fps=fps,
             gt=gt_breaths(mp4[:-4] + ".hdf5", fps), zone=np.asarray(zone), dur=len(frames) / fps)
    return out


def replay(npz, overrides):
    d = np.load(npz)
    fps = float(d["fps"])
    cfg = AiraConfig(fps=fps, **overrides)
    m = BabyMonitor(os.path.basename(npz), cfg)
    tiers, detected = [], []
    for i, (f, e) in enumerate(zip(d["feats"], d["energy"])):
        t = (i + 1) / fps
        snap = m.process_motion(f, float(e), t, 1 / fps)
        if m.breath_times and (not detected or m.breath_times[-1] != detected[-1]):
            detected.append(m.breath_times[-1])
        tiers.append(snap["tier"])
    tiers = np.asarray(tiers)
    gt, dur = d["gt"], float(d["dur"])
    t0 = cfg.warmup_s + 1
    gap = float(np.max(np.diff(np.r_[0.0, gt, dur]))) if len(gt) else dur
    return dict(gt=rate_from(gt, t0, dur), est=rate_from(detected, t0, dur), gap=gap,
                pre=int(np.any(tiers >= TIER_PREALERT)), alarm=int(np.any(tiers >= TIER_ALARM)),
                cal=m.calibrated, n_det=len(detected), n_gt=len(gt))


def score(results):
    ok = [r for r in results if r["gt"] and r["est"]]
    err = np.array([abs(r["est"] - r["gt"]) for r in ok]) if ok else np.array([np.nan])
    clean = [r for r in results if r["gap"] < 10]
    return dict(
        mae=float(np.nanmean(err)), within5=float(np.nanmean(err <= 5)) if ok else 0.0,
        rated=len(ok), n=len(results),
        false_alarm_clips=sum(r["alarm"] for r in clean), prealert_clips=sum(r["pre"] for r in clean),
        clean=len(clean), calibrated=sum(r["cal"] for r in results),
    )


def files_for(subjects):
    return sorted(f for s in subjects for f in glob.glob(os.path.join(CACHE, f"{s}_*.npz")))


def run_set(files, overrides, workers):
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(replay, files, [overrides] * len(files)))


def fmt(s):
    return (f"MAE {s['mae']:5.2f} bpm | within 5 bpm {s['within5'] * 100:3.0f}% | rate on {s['rated']}/{s['n']} | "
            f"false ALARM {s['false_alarm_clips']}/{s['clean']} clips | pre-alert {s['prealert_clips']}/{s['clean']} | "
            f"calibrated {s['calibrated']}/{s['n']}")


def parse_set(items):
    out = {}
    for it in items or []:
        k, v = it.split("=")
        cur = getattr(AiraConfig, k)
        if isinstance(cur, bool):
            out[k] = v in ("1", "True", "true")
        elif isinstance(cur, tuple):
            out[k] = tuple(float(x) for x in v.split(","))
        else:
            out[k] = type(cur)(float(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "sweep", "test"])
    ap.add_argument("--subjects", nargs="+", default=["S01", "S02"])
    ap.add_argument("--set", nargs="*")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    if a.cmd == "cache":
        clips = sorted(c for c in glob.glob(os.path.join(ROOT, "S0*", "*.mp4")) if os.path.exists(c[:-4] + ".hdf5"))
        with ProcessPoolExecutor(a.workers) as ex:
            for i, p in enumerate(ex.map(cache_clip, clips)):
                if i % 10 == 0:
                    print(f"{i + 1}/{len(clips)} {p}", flush=True)
        return
    files = files_for(a.subjects)
    if a.cmd == "test":
        print(f"{len(files)} clips {a.subjects}: {fmt(score(run_set(files, parse_set(a.set), a.workers)))}")
        return
    grid = dict(
        noise_margin=[1.5],
        gross_motion_ratio=[6.0, 8.0, 10.0],
        gross_motion_frames=[2, 3, 5],
        max_breath_gap_s=[7.0],
    )
    keys = list(grid)
    rows = []
    for vals in itertools.product(*grid.values()):
        ov = dict(zip(keys, vals))
        s = score(run_set(files, ov, a.workers))
        rows.append((ov, s))
        print(ov, fmt(s), flush=True)
    # safety first: fewest false alarms, then fewest pre-alerts, then best rate accuracy
    rows.sort(key=lambda r: (r[1]["false_alarm_clips"], r[1]["prealert_clips"], r[1]["mae"]))
    print("\nBEST:")
    for ov, s in rows[:5]:
        print(" ", ov, fmt(s))


if __name__ == "__main__":
    main()
