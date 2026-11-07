
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]
"""Measures what matters for the alert: how fast real apneas alarm, and how many
false alarms happen (benign pauses, movement, normal breathing).

    python tools/evaluate_synthetic.py --minutes 15 --seeds 2
"""
import argparse
import time

import numpy as np

from aira.config import AiraConfig
from aira.monitor import TIER_ALARM, TIER_PREALERT, BabyMonitor
from aira.synthetic import BabySpec, SyntheticWard, schedule

SCENARIOS = {
    "back, clean":           dict(pose="back", bpm=45, amp_px=1.0, noise=3, flicker=0.0),
    "side, shallow":         dict(pose="side", bpm=55, amp_px=0.5, noise=4, flicker=0.0),
    "prone, IR-night noise": dict(pose="prone", bpm=40, amp_px=0.8, noise=9, flicker=0.04),
    "preterm fast, shallow": dict(pose="back", bpm=65, amp_px=0.5, noise=5, flicker=0.02),
}


# Deliberately bad conditions, to find where the camera stops being trustworthy.
STRESS = {
    "very shallow (0.3px)":     dict(pose="side", bpm=55, amp_px=0.3, noise=5, flicker=0.0),
    "tiny + noisy (0.3px, n12)": dict(pose="back", bpm=50, amp_px=0.3, noise=12, flicker=0.03),
    "heavy noise (n20)":        dict(pose="prone", bpm=45, amp_px=0.8, noise=20, flicker=0.05),
    "very fast (80 bpm)":       dict(pose="back", bpm=80, amp_px=0.5, noise=5, flicker=0.0),
}


def run(minutes, seeds, cfg):
    fps = cfg.fps
    dur = minutes * 60
    rows = {k: dict(apneas=0, caught=0, prealert_lat=[], alarm_lat=[], benign=0, benign_fa=0,
                    benign_pre=0, other_fa=0, hours=0.0, poor=0, frames=0) for k in SCENARIOS}
    t_proc, n_proc = 0.0, 0
    for seed in range(seeds):
        rng = np.random.default_rng(1000 + seed)
        specs, names = [], []
        for name, kw in SCENARIOS.items():
            ev = schedule(rng, dur, pauses=[3, 5, 6, 8, 10, 12], apneas=[20, 22, 25, 30, 40], moves=[2, 4, 6, 10])
            specs.append(BabySpec(events=ev, **kw))
            names.append(name)
        ward = SyntheticWard(specs, fps=fps, seed=seed)
        mons = [BabyMonitor(n, cfg) for n in names]
        tiers = [[] for _ in mons]
        n = int(dur * fps)
        for i in range(n):
            t = i / fps
            frame = ward.frame(t)
            t1 = time.perf_counter()
            for k, (x, y, w, h) in enumerate(ward.zones):
                snap = mons[k].update(frame[y:y + h, x:x + w], t)
                tiers[k].append(snap["tier"])
                rows[names[k]]["poor"] += snap["status"] == "POOR SIGNAL"
                rows[names[k]]["frames"] += 1
            t_proc += time.perf_counter() - t1
            n_proc += 1

        for k, name in enumerate(names):
            r = rows[name]
            r["hours"] += dur / 3600
            tr = np.asarray(tiers[k])
            ts = np.arange(len(tr)) / fps
            onset_alarm = ts[1:][(tr[1:] >= TIER_ALARM) & (tr[:-1] < TIER_ALARM)]
            onset_pre = ts[1:][(tr[1:] >= TIER_PREALERT) & (tr[:-1] < TIER_PREALERT)]
            explained = np.zeros(len(onset_alarm), bool)
            for e in specs[k].events:
                win = (e.start - 2, e.start + e.dur + 3)
                a = onset_alarm[(onset_alarm >= win[0]) & (onset_alarm <= win[1])]
                p = onset_pre[(onset_pre >= win[0]) & (onset_pre <= win[1])]
                explained |= (onset_alarm >= win[0]) & (onset_alarm <= win[1])
                if e.kind == "apnea":
                    r["apneas"] += 1
                    if len(a):
                        r["caught"] += 1
                        r["alarm_lat"].append(a[0] - e.start)
                    if len(p):
                        r["prealert_lat"].append(p[0] - e.start)
                elif e.kind == "pause":
                    r["benign"] += 1
                    r["benign_fa"] += int(len(a) > 0)
                    r["benign_pre"] += int(len(p) > 0)
            r["other_fa"] += int((~explained).sum())
            # ignore the first 30 s: monitor is still calibrating and has seen no breath yet
            r["other_fa"] -= int(((onset_alarm < 30) & ~explained).sum())

    print(f"\nprocessing time: {1000 * t_proc / n_proc:.1f} ms per frame for {len(SCENARIOS)} babies "
          f"(budget at {fps:.0f} fps = {1000 / fps:.0f} ms)\n")
    hdr = f"{'scenario':24s} {'apneas caught':>14s} {'pre-alert @':>12s} {'ALARM @':>9s} {'false ALARMs':>13s} {'benign-pause pre-alerts':>24s} {'POOR SIGNAL':>12s}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in rows.items():
        pl = f"{np.median(r['prealert_lat']):.1f}s" if r["prealert_lat"] else "-"
        al = f"{np.median(r['alarm_lat']):.1f}s" if r["alarm_lat"] else "-"
        fa = r["benign_fa"] + r["other_fa"]
        print(f"{name:24s} {r['caught']:>6d}/{r['apneas']:<7d} {pl:>12s} {al:>9s} "
              f"{fa:>5d} ({fa / r['hours']:.1f}/h) {r['benign_pre']:>12d}/{r['benign']:<11d} {100 * r['poor'] / max(r['frames'], 1):>5.0f}% of time")
    print("\n'@' = median seconds after breathing actually stopped. Alarm threshold is "
          f"{cfg.alarm_s:.0f}s (pre-alert {cfg.prealert_s:.0f}s), so ~{cfg.alarm_s:.0f}s is the ideal.")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--stress", action="store_true", help="run the deliberately-bad-conditions set")
    a = ap.parse_args()
    if a.stress:
        SCENARIOS.clear()
        SCENARIOS.update(STRESS)
    run(a.minutes, a.seeds, AiraConfig())
