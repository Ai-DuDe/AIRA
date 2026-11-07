"""All tunable numbers live here, so every alert can be traced back to a named setting."""
from dataclasses import dataclass


@dataclass
class AiraConfig:
    fps: float = 15.0

    # --- Breathing band -------------------------------------------------------
    # Neonates breathe ~30-60 bpm, preterm infants up to ~70-80 bpm.
    # The methodology's 12-40 bpm (0.2-0.7 Hz) band would CUT OFF normal newborn
    # breathing, so the band here is 18-90 bpm.
    # 15-90 bpm. Lower edge chosen on real AIR-125 videos (subjects S01-S02): 0.15 Hz let slow posture drift win,
    # 0.3 Hz cut into older infants breathing ~18-20 bpm.
    band_low_hz: float = 0.25
    band_high_hz: float = 1.5
    integrate_tau_s: float = 4.0     # chest velocity -> displacement (leaky integrator); suppresses fast artifacts
    filter_order: int = 2
    max_bpm: float = 100.0

    # --- Motion extraction ----------------------------------------------------
    max_roi_side: int = 160          # ROI is downscaled to this before optical flow
    grid: int = 4                    # ROI split into grid x grid cells -> 2*grid^2 motion channels
    projection_window_s: float = 10.0

    # --- Calibration ----------------------------------------------------------
    warmup_s: float = 3.0            # let the integrator + causal filter settle (was 5 s)
    calibration_s: float = 15.0      # learn this person's normal breathing depth (was 20 s)
    min_calibration_breaths: int = 5
    min_periodicity: float = 0.3     # autocorrelation peak needed to trust a calibration

    # --- What counts as a breath ----------------------------------------------
    breath_valid_ratio: float = 0.3  # a breath must reach 30% of this baby's baseline depth
    noise_margin: float = 1.5        # ...and 1.5x the measured in-band noise level (tuned on AIR-125 S01-S02)
    abs_floor: float = 0.004         # ...and an absolute floor (px/frame units)
    max_breath_gap_s: float = 7.0    # 2 breaths within this gap end a pause (7 s: tolerates one missed breath at 17 bpm)
    baseline_alpha: float = 0.05     # EWMA weight for baseline depth
    track_rate: bool = False         # 2nd, narrower filter centred on this baby's current breathing rate
    track_band: tuple = (0.6, 1.7)   # ...from 0.6x to 1.7x the current rate
    refractory: float = 0.0          # (off) merge breaths closer than this x usual interval - tested: trades misses for false
    max_breath_ratio: float = 3.0    # deeper than 3x baseline = movement, not used to update the baseline
    baseline_drift_floor: float = 0.5  # baseline may not fall below 50% of calibration ("boiling frog" guard)
    noise_hp_hz: float = 2.5         # above this, the signal is noise (used to measure the noise floor)
    noise_alpha: float = 0.01

    # --- Alert tiers (seconds without a breath) --------------------------------
    # Pauses < ~10 s are normal in newborns (periodic breathing). Alarming on them
    # is the #1 cause of false alarms, so they only show on screen.
    pause_notice_s: float = 10.0     # tile turns yellow, no sound
    prealert_s: float = 15.0         # orange + soft beep: "watch this baby"
    alarm_s: float = 20.0            # red + loud alarm, latched until acknowledged (AAP apnea definition)

    # --- Movement / visibility --------------------------------------------------
    gross_motion_ratio: float = 6.0  # motion > 6x the recent median...
    gross_motion_frames: int = 2     # ...for 2 frames in a row = body movement
    gross_spike_ratio: float = 40.0  # or a single frame > 40x
    gross_motion_abs: float = 1.5    # px/frame, used before calibration
    gross_motion_floor: float = 0.6  # px/frame: smaller average motion is never "body movement"
    motion_settle_s: float = 1.0
    max_motion_hold_s: float = 30.0  # a movement can hold the pause clock for at most this long
    not_visible_alarm_s: float = 15.0

    # --- Signal quality (technical alert, like "leads off" on a bedside monitor) ---
    # Measured from the >2.5 Hz noise band, so it does NOT drop during a real apnea.
    max_cal_failures: int = 3         # ~40 s without a usable calibration -> POOR SIGNAL
    min_snr: float = 1.0              # POOR when typical breath depth < the noise-based breath threshold
    quality_noise_alpha: float = 0.002  # slow noise average (~30 s) for this decision, avoids flapping
    poor_signal_hold_s: float = 8.0   # SNR must be bad for this long before switching
    min_contrast: float = 4.0
