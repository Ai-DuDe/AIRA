"""Breath-by-breath accuracy on AIR-125: does AIRA find EVERY breath, and ONLY real breaths?

Each detected breath is matched to the nearest annotated breath (within 40% of that
clip's typical breath interval). Then:
  missed breaths  = annotated breaths with no detection      -> recall    = found / annotated
  false breaths   = detections with no annotated breath      -> precision = real / detected

    python tools/breath_eval.py --subjects S01 S02                 # tuning babies
    python tools/breath_eval.py --subjects S03 S04 S05             # held-out babies
    python tools/breath_eval.py --subjects S01 S02 --set breath_valid_ratio=0.25

Uses the motion cache made by `python tools/tune_air.py cache`.
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from aira.config import AiraConfig
from aira.monitor import BabyMonitor
from tune_air import CACHE, parse_set

SKIP_S = 6.0   # ignore the first seconds (filters settling)


def detected_breaths(npz, overrides):
    d = np.load(npz)
    fps = float(d["fps"])
    m = BabyMonitor(os.path.basename(npz), AiraConfig(fps=fps, **overrides))
    for i, (f, e) in enumerate(zip(d["feats"], d["energy"])):
        m.process_motion(f, float(e), (i + 1) / fps, 1 / fps)
    return np.asarray(m.accepted_peaks), d["gt"], float(d["dur"])


def match(det, gt, dur):
    det = det[(det >= SKIP_S) & (det <= dur - 1)]
    gt = gt[(gt >= SKIP_S) & (gt <= dur - 1)]
    if len(gt) < 2:
        return 0, 0, len(det)
    tol = 0.4 * float(np.median(np.diff(gt)))
    used, tp = np.zeros(len(det), bool), 0
    for g in gt:
        if len(det) == 0:
            break
        k = np.argmin(np.abs(det - g) + used * 1e9)
        if not used[k] and abs(det[k] - g) <= tol:
            used[k] = True
            tp += 1
    return tp, len(gt) - tp, len(det) - tp      # matched, missed, false


def run(args):
    npz, overrides = args
    det, gt, dur = detected_breaths(npz, overrides)
    return (os.path.basename(npz),) + match(det, gt, dur)


def evaluate(subjects, overrides, workers=None):
    files = sorted(f for s in subjects for f in glob.glob(os.path.join(CACHE, f"{s}_*.npz")))
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(run, [(f, overrides) for f in files]))
    tp = sum(r[1] for r in rows)
    fn = sum(r[2] for r in rows)
    fp = sum(r[3] for r in rows)
    return rows, tp, fn, fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=["S01", "S02"])
    ap.add_argument("--set", nargs="*")
    ap.add_argument("--worst", type=int, default=0, help="also list the N worst clips")
    a = ap.parse_args()
    rows, tp, fn, fp = evaluate(a.subjects, parse_set(a.set))
    rec = tp / max(1, tp + fn)
    prec = tp / max(1, tp + fp)
    print(f"{len(rows)} clips {a.subjects}: breaths annotated {tp + fn} | found {tp} | MISSED {fn} ({100 * (1 - rec):.1f}%) | "
          f"FALSE {fp} ({100 * (1 - prec):.1f}% of detections) | F1 {2 * prec * rec / max(1e-9, prec + rec):.3f}")
    if a.worst:
        for r in sorted(rows, key=lambda r: -(r[2] + r[3]))[: a.worst]:
            print(f"   {r[0]}: found {r[1]}, missed {r[2]}, false {r[3]}")


if __name__ == "__main__":
    main()
