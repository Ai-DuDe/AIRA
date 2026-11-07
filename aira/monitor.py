"""Steps 8-9 - signal quality, per-baby calibration and the apnea decision.

Core idea: apnea is defined clinically by DURATION of breathing cessation, so
the detector is a clock: "seconds since the last real breath". That is directly
explainable ("no breath for 21 s"), it adds almost no delay, and its threshold
is a clinical number (seconds) instead of a statistical one.

Important difference from the methodology's SQI: the SQI must NOT include
"is there a clean periodic signal", because a real apnea has no periodic
signal. If low SNR suppressed alerts, real apneas would be silently hidden.
Here, only things independent of breathing (visibility, gross body movement)
can pause the clock, and both have their own time limits / alarms.
"""
from collections import deque

import cv2
import numpy as np
from scipy.signal import find_peaks

from .breathing import AdaptiveProjector, BreathDetector, bandpass, highpass
from .config import AiraConfig
from .motion import MotionExtractor

TIER_OK, TIER_PAUSE, TIER_PREALERT, TIER_ALARM = 0, 1, 2, 3
TIER_NAMES = {0: "OK", 1: "PAUSE", 2: "PRE-ALERT", 3: "ALARM"}


class BabyMonitor:
    def __init__(self, name: str, cfg: AiraConfig | None = None):
        self.name = name
        self.cfg = cfg or AiraConfig()
        self.events = []  # (t, kind, message)
        self.vacated = False
        self._reset_state(t=None)

    # ------------------------------------------------------------------ control
    def recalibrate(self, t=None):
        self._reset_state(t)
        self._log(t, "info", "recalibrating")

    def mark_vacated(self, t=None):
        self.vacated = True
        self._log(t, "info", "zone marked vacated")

    def unvacate(self, t=None):
        self.vacated = False
        self.recalibrate(t)  # same position never implies same baby

    def acknowledge(self, t=None):
        if self.latched:
            self.latched = False
            self.acked_episode = True
            self._log(t, "ack", "alarm acknowledged")

    # ------------------------------------------------------------------ internals
    def _reset_state(self, t):
        c = self.cfg
        self.motion = MotionExtractor(c.max_roi_side, c.grid)
        n = self.motion.n_channels
        self.bp = bandpass(c.fps, c.band_low_hz, c.band_high_hz, c.filter_order, n)
        self.hp = highpass(c.fps, min(c.noise_hp_hz, 0.8 * c.fps / 2), 2, 1)
        from .fusion import FusionProjector
        self.proj = FusionProjector(n, int(10 * c.fps), int(c.fps), c.fps, "gsm", 0.05)
        self.det = BreathDetector()
        self.rate_hz = None           # current breathing rate (from recent breath intervals)
        self.track_f = None           # centre of the rate-tracking filter
        self.track_bp = None
        self.last_peak_t = None
        self.accepted_peaks = []
        self.disp = np.zeros(n)
        self.energy_hist = deque(maxlen=int(30 * c.fps))
        self.high_run = 0
        self.leak = float(np.exp(-1.0 / (c.integrate_tau_s * c.fps)))
        self.noise_gain = self._noise_gain()
        self.t0 = t
        self.last_t = t
        self.calibrated = False
        self.cal_buf = []
        self.cal_energy = []
        self.cal_failures = 0
        self.cal_amp = None
        self.baseline_amp = None
        self.energy_base = None
        self.noise_var = None
        self.noise_var_slow = None
        self.pause_anchor = None
        self.pending_breath_t = None
        self.breath_times = deque(maxlen=12)
        self.motion_until = -1.0
        self.motion_started = None
        self.was_moving = False
        self.invisible_since = None
        self.tier = TIER_OK
        self.latched = False
        self.acked_episode = False
        self.status = "STARTING"
        self.signal_hist = deque(maxlen=int(15 * c.fps))
        self.h = c.abs_floor
        self.amp_trend_low = False
        self.poor_signal = False
        self.quality_flip_since = None

    def _log(self, t, kind, msg):
        self.events.append((t, kind, f"{self.name}: {msg}"))

    def _hold_clock(self, dt):
        if self.pause_anchor is not None:
            self.pause_anchor += dt

    def _noise_gain(self):
        """How much white per-frame motion noise ends up in the breathing signal (integrated + band-passed)
        compared with the >2.5 Hz band we measure it in. Computed once by simulation, exact for white noise."""
        from scipy import signal as sps
        c = self.cfg
        rng = np.random.default_rng(0)
        v = rng.normal(0, 1, int(c.fps * 600))
        leak = np.exp(-1.0 / (c.integrate_tau_s * c.fps))
        d = sps.lfilter([1.0], [1.0, -leak], v)
        inband = sps.sosfilt(self.bp.sos, d)[int(c.fps * 30):]
        hf = sps.sosfilt(self.hp.sos, v)[int(c.fps * 30):]
        return float(inband.std() / hf.std())

    def _inband_noise(self, slow=False):
        var = self.noise_var_slow if slow else self.noise_var
        if var is None:
            return 0.0
        return float(np.sqrt(var) * self.noise_gain)

    def _try_calibrate(self, t):
        c = self.cfg
        x = np.asarray(self.cal_buf)
        spread = np.percentile(x, 95) - np.percentile(x, 5)
        peaks, props = find_peaks(x, prominence=0.25 * spread, distance=max(1, int(c.fps * 60 / c.max_bpm)))
        ok = len(peaks) >= c.min_calibration_breaths
        if ok:
            amp = float(np.median(props["prominences"]))
            xc = x - x.mean()
            ac = np.correlate(xc, xc, "full")[len(xc) - 1:]
            ac /= ac[0] + 1e-12
            lo, hi = int(c.fps * 60 / c.max_bpm), int(c.fps / c.band_low_hz)
            periodicity = float(ac[lo:hi].max()) if hi > lo else 0.0
            ok = periodicity >= c.min_periodicity and amp > c.noise_margin * self._inband_noise()
        if ok:
            self.calibrated = True
            self.cal_amp = self.baseline_amp = amp
            self.energy_base = float(np.median(self.cal_energy))
            rate = 60 * c.fps / np.median(np.diff(peaks))
            self.rate_hz = rate / 60.0
            self._log(t, "info", f"calibrated: depth={amp:.4f}, rate~{rate:.0f} bpm, periodicity={periodicity:.2f}")
        else:
            self.cal_failures += 1
            half = len(self.cal_buf) // 2  # slide the window and keep trying
            self.cal_buf = self.cal_buf[half:]
            self.cal_energy = self.cal_energy[half:]

    # ------------------------------------------------------------------ main step
    def update(self, crop, t: float) -> dict:
        c = self.cfg
        if self.t0 is None:
            self.t0 = t
        dt = 0.0 if self.last_t is None else max(0.0, t - self.last_t)
        self.last_t = t

        if self.vacated:
            self.status, self.tier, self.latched = "VACATED", TIER_OK, False
            self.motion.reset()
            return self.snapshot(t)

        # ---- visibility (independent of breathing) ----
        gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        m, sd = float(gray.mean()), float(gray.std())
        if sd < c.min_contrast or m < 12 or m > 245:
            if self.invisible_since is None:
                self.invisible_since = t
                self._log(t, "warn", "baby not visible")
            self._hold_clock(dt)
            self.motion.reset()
            self.status = "NOT VISIBLE"
            gone = t - self.invisible_since
            self._set_tier(t, TIER_ALARM if gone >= c.not_visible_alarm_s else TIER_PAUSE, "not visible")
            return self.snapshot(t)
        self.invisible_since = None

        out = self.motion.step(crop)
        if out is None:
            return self.snapshot(t)
        return self.process_motion(out[0], out[1], t, dt)

    def process_motion(self, feats, energy, t, dt):
        """Everything after optical flow. Split out so recorded motion can be replayed (fast tuning)."""
        self.proj.push_feats(np.asarray(feats, float))
        c = self.cfg
        if self.t0 is None:
            self.t0 = t
        # ---- gross body movement (independent of breathing) ----
        # Compared with the recent typical motion level (rolling median), which is robust on real video
        # where the per-frame motion level fluctuates a lot.
        # Real body movement lasts; single-frame spikes (video keyframes, IR exposure jumps) don't.
        # So: high for `gross_motion_frames` frames in a row, or one extreme spike.
        if len(self.energy_hist) >= c.fps * 5:
            med = float(np.median(self.energy_hist))
            # also an absolute floor: if the usual level is ~0 (a still scene), 6x ~0 is still ~0
            # and every tiny breath would count as "body movement"
            high = energy > max(c.gross_motion_ratio * med, c.gross_motion_floor)
            self.high_run = self.high_run + 1 if high else 0
            gross = self.high_run >= c.gross_motion_frames or energy > max(c.gross_spike_ratio * med, 3 * c.gross_motion_floor)
        else:
            gross = energy > c.gross_motion_abs
        # Always record (the median ignores short movements). Recording only "still" frames could
        # deadlock: if the start-up threshold is exceeded, history never fills and MOVING never ends.
        self.energy_hist.append(energy)
        if gross:
            self.motion_until = t + c.motion_settle_s
        moving = t < self.motion_until
        if moving and not self.was_moving:
            self.motion_started = t
        if self.was_moving and not moving:
            self.bp.reset()  # drop filter ringing from the movement
            if self.track_bp is not None:
                self.track_bp.reset()
            self.det.reset()
            self.disp[:] = 0.0
            self.motion_started = None
        self.was_moving = moving

        # chest velocity -> displacement: breathing is a slow displacement, flicker/compression
        # artifacts are fast, so integrating suppresses them in proportion to their frequency
        self.disp = self.leak * self.disp + feats
        y = self.bp.step(self.disp)
        s = self.proj.step(y, learn=not moving)
        self.signal_hist.append(0.0 if moving else s)

        warm = (t - self.t0) >= c.warmup_s
        if self.pause_anchor is None and warm:
            self.pause_anchor = t

        if moving:
            # Movement means we cannot see breathing, but the baby is clearly active.
            # Hold the clock - but only for a limited time.
            if t - self.motion_started <= c.max_motion_hold_s:
                self._hold_clock(dt)
            self.status = "MOVING"
            self._update_tier(t)
            return self.snapshot(t)

        # ---- noise floor (from the >2.5 Hz band, where breathing has no power) ----
        hf = self.hp.step([feats @ self.proj.w])[0]
        if warm:
            self.noise_var = hf * hf if self.noise_var is None else (1 - c.noise_alpha) * self.noise_var + c.noise_alpha * hf * hf
            qa = c.quality_noise_alpha
            self.noise_var_slow = hf * hf if self.noise_var_slow is None else (1 - qa) * self.noise_var_slow + qa * hf * hf

        # ---- minimum breath depth for this baby right now ----
        h = max(c.abs_floor, c.noise_margin * self._inband_noise())
        if self.calibrated:
            h = max(h, c.breath_valid_ratio * self.baseline_amp)
        self.h = h

        if not warm:
            self.status = "STARTING"
            return self.snapshot(t)

        # ---- breath detection ----
        # Rate-tracking filter: once the breathing rate is known, a narrower band around it removes
        # wiggles that are much faster/slower than this baby's breathing (fewer false + missed breaths).
        sd = s
        if c.track_rate and self.rate_hz:
            if self.track_f is None or abs(self.rate_hz - self.track_f) > 0.15 * self.track_f:
                lo = max(c.band_low_hz * 0.8, c.track_band[0] * self.rate_hz)
                hi = min(0.45 * c.fps, c.track_band[1] * self.rate_hz)
                if hi > lo * 1.2:
                    from scipy import signal as _sps
                    new = _sps.butter(2, [lo, hi], btype="band", fs=c.fps, output="sos")
                    if self.track_bp is None:
                        from .breathing import CausalFilter
                        self.track_bp = CausalFilter(new, 1)
                    else:
                        self.track_bp.sos = new           # keep the running state: no restart transient
                    self.track_f = self.rate_hz
            if self.track_bp is not None:
                sd = float(self.track_bp.step([s])[0])
        res = self.det.step(sd, t, h)
        if res is not None and self.last_peak_t is not None and self.rate_hz:
            if res[0] - self.last_peak_t < c.refractory / self.rate_hz:
                res = None                              # too soon after the last breath: same breath
        if res is not None:
            self.last_peak_t = res[0]
            self.accepted_peaks.append(res[0])     # every breath AIRA accepts (for evaluation)
        if res is not None:
            peak_t, depth = res
            prev = self.pending_breath_t
            self.pending_breath_t = peak_t
            if prev is not None and peak_t - prev <= c.max_breath_gap_s:
                # two real breaths close together -> breathing is present
                self.pause_anchor = max(self.pause_anchor, peak_t)
                self.breath_times.append(peak_t)
                if len(self.breath_times) >= 4:
                    self.rate_hz = 1.0 / float(np.median(np.diff(np.asarray(self.breath_times)[-6:])))
                if self.calibrated and depth <= c.max_breath_ratio * self.baseline_amp:
                    # (a "breath" far deeper than normal is body movement: it must not inflate the baseline,
                    #  or every normal breath afterwards would look too shallow -> false alarm)
                    b = (1 - c.baseline_alpha) * self.baseline_amp + c.baseline_alpha * depth
                    floor = c.baseline_drift_floor * self.cal_amp
                    if b < floor and not self.amp_trend_low:
                        self._log(t, "warn", "breathing depth trending low vs calibration - review")
                    self.amp_trend_low = b < floor
                    self.baseline_amp = max(b, floor)
                    self.energy_base = 0.98 * self.energy_base + 0.02 * energy

        # ---- calibration (runs on clean, non-moving data only) ----
        if not self.calibrated:
            self.cal_buf.append(s)
            self.cal_energy.append(energy)
            if len(self.cal_buf) >= c.calibration_s * c.fps:
                self._try_calibrate(t)
            self.status = "CALIBRATING" if self.cal_failures == 0 else "WEAK SIGNAL"
        else:
            self.status = "MONITORING"

        # ---- signal quality: can breaths be seen above the noise at all? ----
        noise_thr = c.noise_margin * self._inband_noise(slow=True)
        if self.calibrated:
            bad = self.baseline_amp < c.min_snr * noise_thr
        else:
            bad = self.cal_failures >= c.max_cal_failures
        if bad != self.poor_signal:
            if self.quality_flip_since is None:
                self.quality_flip_since = t
            if t - self.quality_flip_since >= c.poor_signal_hold_s:
                self.poor_signal = bad
                self.quality_flip_since = None
                self._log(t, "technical" if bad else "info",
                          "POOR SIGNAL - breathing not measurable, check camera/lighting/position" if bad
                          else "signal quality restored")
        else:
            self.quality_flip_since = None

        self._update_tier(t)
        if self.poor_signal:
            # Fail toward caution: the apnea clock keeps running (a real apnea must still alarm),
            # and on top of that a persistent technical alert says the reading is unreliable.
            self.status = "POOR SIGNAL"
            self.tier = max(self.tier, TIER_PREALERT)
        return self.snapshot(t)

    def _update_tier(self, t):
        c = self.cfg
        p = self.pause_seconds(t)
        if p >= c.alarm_s:
            tier = TIER_ALARM
        elif p >= c.prealert_s:
            tier = TIER_PREALERT
        elif p >= c.pause_notice_s:
            tier = TIER_PAUSE
        else:
            tier = TIER_OK
        self._set_tier(t, tier, f"no breath for {p:.0f}s")

    def _set_tier(self, t, tier, why):
        if tier > self.tier and tier >= TIER_PREALERT:
            self._log(t, TIER_NAMES[tier].lower(), why)
        if tier == TIER_ALARM and not self.acked_episode:
            self.latched = True
        if tier == TIER_OK:
            self.acked_episode = False
        self.tier = tier

    def pause_seconds(self, t):
        return 0.0 if self.pause_anchor is None else max(0.0, t - self.pause_anchor)

    def rate_bpm(self):
        if len(self.breath_times) < 4:
            return None
        iv = np.diff(np.asarray(self.breath_times)[-8:])
        return float(60.0 / np.median(iv))

    def snapshot(self, t):
        return {
            "name": self.name,
            "status": self.status,
            "tier": self.tier,
            "latched": self.latched,
            "pause_s": self.pause_seconds(t),
            "rate_bpm": self.rate_bpm() if self.pause_seconds(t) < 5 else None,
            "calibrated": self.calibrated,
            "amp_trend_low": self.amp_trend_low,
        }
