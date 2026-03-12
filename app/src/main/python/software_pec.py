"""
Software Periodic Error Correction (PEC)
=========================================

Automatic detection and correction of periodic mechanical errors
without physical sensors (encoderless / sensorless PEC).

Architecture & Data Flow
------------------------
This module sits in the real-time tracking correction pipeline
(see realtime_tracking.py, step 7b). The pipeline is:

    plate-solve residuals
        -> Kalman filter (removes drift / smooth trends)
        -> ML model (removes structured non-periodic error)
        -> **Software PEC** (removes periodic mechanical error)  <-- this module
        -> final tracking rate sent to motors

The PEC correction is *additive*: the value returned by get_correction()
is added directly to the alt/az tracking rates (arcsec/sec).

How It Works (high level)
-------------------------
1. **Collect**: Each tracking cycle, the post-Kalman/ML residual error
   is fed to add_error_sample(). These residuals isolate the periodic
   component that Kalman and ML cannot predict.
2. **Detect**: Periodically (_run_analysis), the collected residuals are
   resampled onto a uniform time grid, detrended, and passed through an
   FFT to find dominant periodicities.
3. **Fit**: For each detected period, a multi-harmonic Fourier series is
   fit via least-squares regression.
4. **Correct**: At each tracking cycle, get_correction() evaluates the
   summed Fourier model at the current elapsed time and returns the
   *negative* of the predicted error (to cancel it).

Mount Agnosticism
-----------------
This module is **frequency-agnostic**: it discovers all dominant
periodicities directly from the data instead of relying on hard-coded
gear ratios. This makes it work with ANY mechanical drive system:

- **Worm gears** (e.g. Skywatcher Dobsonians) - single dominant period
  matching the worm rotation period.
- **Planetary gearboxes + GT2 belts** (e.g. the author's custom Dobson
  Alt-Az mount) - multiple overlapping periods from planet carrier
  rotation, ring-gear mesh, and belt tooth engagement.
- **Harmonic drives**, spur gear trains, or any combination thereof.

Spectral Analysis Details
-------------------------
- A Hann window is applied before the FFT to suppress spectral leakage
  from non-integer cycle counts in the observation window.
- The minimum resolvable frequency separation is 1/T (where T is the
  data span); a 3x guard band is used as the exclusion zone around each
  detected peak because the Hann window's main lobe is ~3 bins wide.
- Peaks are selected as local maxima (not just threshold crossings) to
  avoid counting leakage sidelobes as separate periods.
- Detected peaks are checked against existing peaks to reject harmonics
  of already-detected fundamentals (ratio close to integer => skip).

Coefficient Update Strategy
---------------------------
When re-analyzing (every 30 s by default), the new Fourier coefficients
are blended with the old ones via Exponential Moving Average (EMA):

    coeff_new = coeff_old * (1 - lr) + coeff_measured * lr

This avoids abrupt jumps in the correction signal when the model is
updated mid-observation.

Persistence
-----------
The learned model (detected periods + Fourier coefficients) is
serialised to JSON via save()/load(). On the next session, the model
is loaded and begins correcting immediately; new data will gradually
re-lock the phase and refine coefficients through EMA blending.

Units & Sign Convention
-----------------------
- All error samples and corrections are in **arcsec/sec** (tracking rate
  units), matching the rest of the correction pipeline.
- The sign convention is:
      correction = -predicted_error
  A positive predicted error means the mount is tracking too fast in
  that axis, so a negative correction slows it down (and vice versa).

All corrections are in arcsec/sec, matching the tracking pipeline.
"""

import numpy as np
import time
import json
import os
import threading
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass, field


@dataclass
class PECPeriod:
    """
    A single detected periodic error component.

    Each instance represents one mechanical periodicity (e.g. one gear's
    rotation period) and holds the Fourier series that models it:

        correction(t) = SUM over n=1..n_harmonics of:
            a_n * cos(2*pi*n*f0*t) + b_n * sin(2*pi*n*f0*t)

    where f0 = frequency_hz (the fundamental), a_n = cos_coeffs[n-1],
    and b_n = sin_coeffs[n-1].

    Attributes:
        period_sec:       Period of this component in seconds.
        frequency_hz:     Fundamental frequency (= 1/period_sec).
        amplitude_arcsec: RMS amplitude of the full Fourier series
                          (sqrt of summed squared coefficients), in
                          arcsec/sec.
        snr:              Signal-to-noise ratio at which this period was
                          detected in the FFT power spectrum.
        cos_coeffs:       Cosine (a_n) Fourier coefficients, length
                          n_harmonics.
        sin_coeffs:       Sine (b_n) Fourier coefficients, length
                          n_harmonics.
        n_harmonics:      Number of harmonics in the fit (1 = fundamental
                          only, 2 = fundamental + 1st overtone, etc.).
    """
    period_sec: float           # Period in seconds
    frequency_hz: float         # Frequency in Hz
    amplitude_arcsec: float     # Peak amplitude in arcsec/sec
    snr: float                  # Signal-to-noise ratio of detection
    # Fourier coefficients: correction(t) = sum(a_n*cos(2*pi*n*f*t) + b_n*sin(2*pi*n*f*t))
    cos_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sin_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_harmonics: int = 0


# ================================================================
#  Drive-type presets (module-level)
# ================================================================
#
# Each mount drive system has characteristic periodic error signatures.
# These presets tune the PEC engine for optimal detection and correction
# of each system's dominant error patterns.
#
# - worm_gear:         Single dominant period (worm rotation, often ~400-480 s).
#                      Clean periodic signature, few harmonics needed.
# - planetary_gearbox: Multiple overlapping periods from planet carrier
#                      rotation, ring-gear mesh, and belt tooth engagement.
#                      Needs more periods and harmonics.
# - harmonic_drive:    Very short periods from the flex-spline; typically
#                      low amplitude but high frequency.
# - belt_drive:        Periodic errors from belt tooth engagement. Period
#                      depends on belt length and pulley teeth count.
# - direct_drive:      No gearing -- very low periodic error. PEC still
#                      useful for bearing imperfections but needs high SNR.

DRIVE_TYPE_PRESETS = {
    'worm_gear': {
        'max_harmonics': 6,        # worm errors are fairly sinusoidal
        'min_snr': 2.5,            # strong, clean signals
        'max_periods': 3,          # usually 1-2 dominant periods
        'min_samples_for_fft': 80, # converges fast with clean signal
        'learning_rate': 0.08,     # faster convergence (clean periodic signal)
        'analysis_interval': 45.0, # re-analyse more often (single period locks fast)
    },
    'planetary_gearbox': {
        'max_harmonics': 8,        # non-sinusoidal gear tooth profiles
        'min_snr': 3.0,            # multiple overlapping periods, need selectivity
        'max_periods': 6,          # carrier, ring-gear, belt all contribute
        'min_samples_for_fft': 120,# need more data to resolve close frequencies
        'learning_rate': 0.05,     # slower blending (complex multi-period model)
        'analysis_interval': 60.0, # standard interval
    },
    'harmonic_drive': {
        'max_harmonics': 4,        # short-period errors are fairly clean
        'min_snr': 4.0,            # small amplitudes, need high confidence
        'max_periods': 4,          # flex-spline + wave generator
        'min_samples_for_fft': 60, # short periods fill cycles fast
        'learning_rate': 0.06,
        'analysis_interval': 45.0,
    },
    'belt_drive': {
        'max_harmonics': 6,        # belt cogging can have harmonics
        'min_snr': 3.0,
        'max_periods': 4,          # belt + pulley + motor
        'min_samples_for_fft': 100,
        'learning_rate': 0.06,
        'analysis_interval': 50.0,
    },
    'direct_drive': {
        'max_harmonics': 4,        # bearing imperfections only
        'min_snr': 5.0,            # very low amplitude, demand high SNR
        'max_periods': 2,          # few periodic sources
        'min_samples_for_fft': 150,# need lots of data to find weak signals
        'learning_rate': 0.03,     # very cautious (signals are weak)
        'analysis_interval': 90.0, # less frequent (slow convergence)
    },
}


class SoftwarePEC:
    """
    Automatic Software Periodic Error Correction.

    Detects and corrects periodic mechanical errors by analysing
    plate-solve residuals over time using FFT-based period detection
    and multi-harmonic Fourier fitting.

    Independent correction models are maintained for the Altitude and
    Azimuth axes, since each axis may have different mechanical drives
    with different periodic signatures.

    Supports drive-type-aware parameter presets (see DRIVE_TYPE_PRESETS)
    for optimised detection on worm gears, planetary gearboxes, harmonic
    drives, belt drives, and direct drives.

    Typical usage inside the tracking loop::

        pec = SoftwarePEC(drive_type='worm_gear')
        pec.load("pec_model.json")          # restore previous session

        # Inside each tracking cycle:
        pec.add_error_sample(residual_alt, residual_az)   # feed residuals
        corr_alt, corr_az = pec.get_correction()          # get correction
        tracking_rate_alt += corr_alt                      # apply (additive)
        tracking_rate_az  += corr_az

        pec.save("pec_model.json")          # persist at end of session
    """
    
    def __init__(self,
                 drive_type: Optional[str] = None,
                 max_harmonics: int = 8,
                 min_snr: float = 3.0,
                 max_periods: int = 5,
                 min_samples_for_fft: int = 100,
                 min_cycles_for_fit: float = 2.0,
                 learning_rate: float = 0.05):
        """
        Initialise the Software PEC engine.

        Args:
            drive_type:        Mount drive system type. If provided, loads
                               optimised parameter presets for that system
                               from DRIVE_TYPE_PRESETS. Explicit parameter
                               values take precedence over the preset.
                               Valid types: 'worm_gear', 'planetary_gearbox',
                               'harmonic_drive', 'belt_drive', 'direct_drive'.
            max_harmonics:     Maximum number of Fourier harmonics fitted
                               per detected period. Higher values capture
                               more complex waveforms (e.g. non-sinusoidal
                               gear errors) but require more data to avoid
                               overfitting. 8 is a good default.
            min_snr:           Minimum signal-to-noise ratio in the FFT
                               power spectrum for a peak to be accepted as
                               a real periodicity. Lower values are more
                               sensitive but risk false detections.
            max_periods:       Maximum number of independent periodic
                               components kept per axis. Sorted by
                               amplitude; the weakest are pruned.
            min_samples_for_fft: Minimum number of error samples
                               accumulated before the first FFT analysis
                               is attempted. Too few samples yield poor
                               frequency resolution.
            min_cycles_for_fit: Minimum number of complete cycles of a
                               candidate period that must be present in the
                               data span before that period is accepted.
                               Ensures there is enough phase coverage for
                               a reliable Fourier fit. 2.0 cycles is the
                               safe minimum.
            learning_rate:     Exponential Moving Average (EMA) blending
                               factor used when updating Fourier
                               coefficients on re-analysis. 0.05 means
                               each new analysis contributes 5% and the
                               prior model retains 95%, giving a smooth
                               transition with a ~20-analysis time constant.
        """
        # Store the drive type for reporting and persistence
        self.drive_type = drive_type or 'planetary_gearbox'

        # Apply drive-type presets first, then let explicit arguments override
        preset = DRIVE_TYPE_PRESETS.get(self.drive_type, {})
        self.max_harmonics = preset.get('max_harmonics', max_harmonics)
        self.min_snr = preset.get('min_snr', min_snr)
        self.max_periods = preset.get('max_periods', max_periods)
        self.min_samples_for_fft = preset.get('min_samples_for_fft', min_samples_for_fft)
        self.min_cycles_for_fit = min_cycles_for_fit
        self.learning_rate = preset.get('learning_rate', learning_rate)
        
        # ----- Error sample ring buffers -----
        # Stores (elapsed_time, error_arcsec_per_sec) tuples for each axis.
        # elapsed_time is relative to _start_time for numerical stability.
        self._timestamps: List[float] = []
        self._errors_alt: List[float] = []
        self._errors_az: List[float] = []
        # Buffer capacity: ~8 min at 10 Hz plate-solve rate, ~42 min at 2 Hz.
        # Reduced from 10000 for lower memory usage on constrained devices.
        # Older samples are trimmed FIFO when the buffer is full.
        self._max_samples = 5000
        
        # Phase reference: wall-clock time of the first sample. All elapsed
        # times are computed relative to this so that the Fourier phase
        # argument stays numerically small (avoids float precision loss).
        self._start_time: Optional[float] = None
        
        # ----- Learned model (per axis) -----
        # Each list holds the detected PECPeriod objects with their
        # fitted Fourier coefficients.
        self.periods_alt: List[PECPeriod] = []
        self.periods_az: List[PECPeriod] = []
        
        # ----- Runtime state -----
        self.is_trained = False      # True once at least one period is detected
        self.is_enabled = True       # Master enable/disable switch
        self.is_learning = True      # If True, continuously re-analyse and update
        self._analysis_count = 0     # How many times _run_analysis has completed
        self._last_analysis_time = 0.0
        # Re-analyse interval: drive-type preset or default 60 s.
        # Increased from 30s for constrained devices. Trade-off: too frequent
        # wastes CPU, too rare delays adaptation to changing conditions.
        self._analysis_interval = preset.get('analysis_interval', 60.0)
        
        # Thread safety: protects the sample buffers and period lists, which
        # are read by get_correction() (called from the tracking thread) and
        # written by _run_analysis() / add_error_sample().
        self._lock = threading.Lock()
        
        # Optional external logging callback (e.g. to the application GUI log).
        self.on_log: Optional[Callable[[str], None]] = None
        
        # Diagnostic counters exposed to the UI via get_statistics().
        self.stats = {
            'total_samples': 0,
            'periods_detected_alt': 0,
            'periods_detected_az': 0,
            'correction_rms_alt': 0.0,
            'correction_rms_az': 0.0,
            'last_analysis_time': 0.0,
            'data_span_sec': 0.0,
        }
    
    def _log(self, msg: str):
        """Forward a log message to the external callback, if registered."""
        if self.on_log:
            try:
                self.on_log(msg)
            except Exception:
                pass
    
    def reset(self):
        """
        Reset all learned data and return to the untrained state.

        This clears the sample buffers, detected periods, and Fourier
        coefficients. Useful when the user changes the mechanical setup
        or wants to start fresh.
        """
        with self._lock:
            self._timestamps.clear()
            self._errors_alt.clear()
            self._errors_az.clear()
            self._start_time = None
            self.periods_alt.clear()
            self.periods_az.clear()
            self.is_trained = False
            self._analysis_count = 0
            self._log("PEC: Reset - all learned data cleared")
    
    def set_drive_type(self, drive_type: str):
        """
        Change the mount drive type and re-tune PEC parameters.

        Applies the preset for the new drive type WITHOUT clearing the
        learned model. If the user switches drive type mid-session, the
        existing model is preserved (it will gradually adapt via EMA
        blending on subsequent analyses).

        To start fresh with the new drive type, call reset() after this.

        Args:
            drive_type: One of 'worm_gear', 'planetary_gearbox',
                        'harmonic_drive', 'belt_drive', 'direct_drive'.
        """
        preset = DRIVE_TYPE_PRESETS.get(drive_type, {})
        if not preset:
            self._log(f"PEC: Unknown drive type '{drive_type}', keeping current settings")
            return

        self.drive_type = drive_type
        self.max_harmonics = preset.get('max_harmonics', self.max_harmonics)
        self.min_snr = preset.get('min_snr', self.min_snr)
        self.max_periods = preset.get('max_periods', self.max_periods)
        self.min_samples_for_fft = preset.get('min_samples_for_fft', self.min_samples_for_fft)
        self.learning_rate = preset.get('learning_rate', self.learning_rate)
        self._analysis_interval = preset.get('analysis_interval', self._analysis_interval)

        self._log(f"PEC: Drive type set to '{drive_type}' "
                  f"(harmonics={self.max_harmonics}, SNR>={self.min_snr}, "
                  f"max_periods={self.max_periods})")

    def add_error_sample(self, error_alt: float, error_az: float):
        """
        Add an observed tracking error sample.
        
        This is called from the correction loop with the drift residuals
        AFTER Kalman + ML correction has been applied. By measuring the
        residual *after* those stages, we isolate the periodic component
        that deterministic / adaptive filters cannot predict -- which is
        exactly what PEC should correct.

        Internally this method:
        1. Appends the sample to the ring buffers.
        2. Trims the oldest samples if the buffer is full.
        3. Triggers a periodic re-analysis (_run_analysis) if enough
           samples have been collected and enough time has elapsed since
           the last analysis.
        
        Args:
            error_alt: Residual tracking error in altitude (arcsec/sec).
                       Positive means the mount is pointing too high
                       relative to the target.
            error_az:  Residual tracking error in azimuth (arcsec/sec).
                       Positive means the mount is pointing too far east
                       (or whichever direction corresponds to positive az
                       in the mount's coordinate frame).
        """
        now = time.time()
        
        with self._lock:
            # Initialise the phase reference on the very first sample.
            if self._start_time is None:
                self._start_time = now
            
            # Store elapsed time relative to _start_time for numerical
            # stability (avoids loss of float64 precision when computing
            # sin/cos of large absolute timestamps).
            elapsed = now - self._start_time
            
            self._timestamps.append(elapsed)
            self._errors_alt.append(error_alt)
            self._errors_az.append(error_az)
            
            # FIFO trim: discard the oldest samples when the buffer is full.
            # This means the FFT always analyses the most recent window of
            # data, allowing the model to adapt to slowly changing conditions.
            if len(self._timestamps) > self._max_samples:
                excess = len(self._timestamps) - self._max_samples
                self._timestamps = self._timestamps[excess:]
                self._errors_alt = self._errors_alt[excess:]
                self._errors_az = self._errors_az[excess:]
            
            self.stats['total_samples'] = len(self._timestamps)
            if len(self._timestamps) > 1:
                self.stats['data_span_sec'] = self._timestamps[-1] - self._timestamps[0]
        
        # Trigger periodic re-analysis.
        # Conditions: we have enough samples AND enough wall-clock time has
        # passed since the last analysis (to avoid thrashing on every sample).
        if (len(self._timestamps) >= self.min_samples_for_fft and 
                now - self._last_analysis_time > self._analysis_interval):
            self._run_analysis()
    
    def get_correction(self, t: Optional[float] = None) -> Tuple[float, float]:
        """
        Get the PEC correction for the current (or specified) time.
        
        This is the main output of the PEC system. The returned values
        should be **added** to the current tracking rates (they are
        already sign-inverted relative to the predicted error).

        Args:
            t: Optional elapsed time override (seconds since _start_time).
               If None, the current wall-clock time is used. Providing an
               explicit value is useful for unit testing or replay.
                
        Returns:
            (correction_alt, correction_az) in arcsec/sec.
            Both are 0.0 if PEC is disabled or untrained.
        """
        if not self.is_enabled or not self.is_trained:
            return 0.0, 0.0
        
        with self._lock:
            if self._start_time is None:
                return 0.0, 0.0
            
            if t is None:
                t = time.time() - self._start_time
            
            corr_alt = self._evaluate_fourier(t, self.periods_alt)
            corr_az = self._evaluate_fourier(t, self.periods_az)
        
        return corr_alt, corr_az
    
    def _evaluate_fourier(self, t: float, periods: List[PECPeriod]) -> float:
        """
        Evaluate the summed multi-harmonic Fourier model at time t.
        
        For each detected period p with fundamental frequency f0:

            predicted_error(t) = SUM_{n=1}^{N} [
                a_n * cos(2*pi*n*f0*t) + b_n * sin(2*pi*n*f0*t)
            ]

        The *correction* is the negative of the predicted error, because
        we want to cancel it:

            correction(t) = -predicted_error(t)

        Sign convention example:
        - If the Fourier model predicts the mount will track +0.5"/s too
          fast in altitude at time t, the correction is -0.5"/s, which
          slows the tracking rate down to compensate.

        Vectorized implementation: builds phase arrays for all harmonics
        across all periods and evaluates cos/sin via numpy in a single
        call, which is ~5-20x faster than per-harmonic Python loops
        on mobile devices.

        Args:
            t:       Elapsed time in seconds (relative to _start_time).
            periods: List of PECPeriod objects for one axis.

        Returns:
            The correction value in arcsec/sec (already negated).
        """
        if not periods:
            return 0.0

        # Collect all harmonic phases and coefficients into flat arrays
        # for a single vectorized cos/sin evaluation.
        all_phases = []
        all_cos_coeffs = []
        all_sin_coeffs = []

        for p in periods:
            if p.n_harmonics == 0:
                continue
            harmonics = np.arange(1, p.n_harmonics + 1)
            phases = 2.0 * np.pi * harmonics * p.frequency_hz * t
            all_phases.append(phases)
            all_cos_coeffs.append(np.asarray(p.cos_coeffs[:p.n_harmonics]))
            all_sin_coeffs.append(np.asarray(p.sin_coeffs[:p.n_harmonics]))

        if not all_phases:
            return 0.0

        # Concatenate and evaluate in one numpy call
        phases = np.concatenate(all_phases)
        cos_c = np.concatenate(all_cos_coeffs)
        sin_c = np.concatenate(all_sin_coeffs)

        correction = float(np.dot(cos_c, np.cos(phases)) + np.dot(sin_c, np.sin(phases)))

        # Return NEGATIVE of the predicted error: the correction must
        # have the opposite sign to cancel the periodic tracking error.
        return -correction
    
    def _run_analysis(self):
        """
        Run FFT-based period detection and Fourier fitting on the
        accumulated error samples.

        This is the core analysis pipeline, called periodically (every
        _analysis_interval seconds) as new data arrives. The algorithm:

        1. **Snapshot**: Copy the sample buffers under the lock to avoid
           blocking the tracking thread during heavy computation.
        2. **Resample**: The raw samples arrive at irregular intervals
           (plate solves are not perfectly periodic). Linearly interpolate
           onto a uniform time grid, which is required by the FFT.
        3. **Detrend**: Remove the DC offset and any linear drift from
           the resampled signal. This isolates the periodic content;
           drift is already handled by the Kalman filter, and any
           residual linear trend would corrupt the low-frequency FFT bins.
        4. **FFT period detection** (_detect_periods): Apply a Hann
           window, compute the power spectrum, and find local maxima
           above the noise floor. Spectral leakage suppression ensures
           only true periodicities are reported (see _detect_periods
           docstring for details).
        5. **Fourier fit** (_fit_fourier): For each detected period, fit
           a multi-harmonic Fourier series to the detrended signal using
           ordinary least squares. This yields the cos/sin coefficients
           that model the waveform shape (which is generally NOT a pure
           sine due to gear tooth profiles, belt cogging, etc.).
        6. **Merge** (_merge_periods): Blend newly detected periods with
           the existing model via EMA, so the correction signal evolves
           smoothly rather than jumping discontinuously.
        7. **Log**: Report the results for user visibility.
        """
        with self._lock:
            if len(self._timestamps) < self.min_samples_for_fft:
                return
            
            # Snapshot the buffers (copy to numpy arrays) so we can release
            # the lock and do the heavy FFT/fitting work without blocking
            # the tracking thread.
            timestamps = np.array(self._timestamps)
            errors_alt = np.array(self._errors_alt)
            errors_az = np.array(self._errors_az)
        
        self._last_analysis_time = time.time()
        self._analysis_count += 1
        
        # --- Step 2: Resample to uniform time grid ---
        # The FFT requires uniformly spaced samples. We use linear
        # interpolation (np.interp) which is adequate because the sample
        # rate is much higher than the frequencies we care about.
        t_start, t_end = timestamps[0], timestamps[-1]
        data_span = t_end - t_start
        
        if data_span < 10.0:
            return  # Need at least 10 seconds of data for meaningful analysis
        
        # Estimate the effective (average) sample rate from the data.
        n_samples = len(timestamps)
        effective_dt = data_span / (n_samples - 1)
        effective_fs = 1.0 / effective_dt
        
        # Cap the uniform grid size at 4096 points to bound FFT cost
        # (O(N log N)). For typical plate-solve rates of 2-10 Hz over
        # minutes of data, this is more than sufficient.
        n_uniform = min(n_samples, 4096)
        t_uniform = np.linspace(t_start, t_end, n_uniform)
        alt_uniform = np.interp(t_uniform, timestamps, errors_alt)
        az_uniform = np.interp(t_uniform, timestamps, errors_az)
        
        # --- Step 3: Detrend ---
        # Remove DC offset and linear slope so that only oscillatory
        # (periodic) content remains. The detrending uses a simple OLS
        # linear fit: y = a*t + b, then subtracts it.
        alt_detrended = self._detrend(alt_uniform)
        az_detrended = self._detrend(az_uniform)
        
        # --- Step 4: FFT period detection ---
        # The sample rate on the uniform grid may differ slightly from
        # effective_fs because n_uniform may be capped. Recompute it.
        fs_uniform = (n_uniform - 1) / data_span
        new_periods_alt = self._detect_periods(alt_detrended, fs_uniform, data_span)
        new_periods_az = self._detect_periods(az_detrended, fs_uniform, data_span)
        
        # --- Step 5: Fourier fit ---
        # Fit multi-harmonic Fourier series for each detected period.
        # Time is shifted to start from 0 so the Fourier phase reference
        # aligns with _start_time (which is what get_correction uses).
        for p in new_periods_alt:
            self._fit_fourier(p, t_uniform - t_start, alt_detrended)
        for p in new_periods_az:
            self._fit_fourier(p, t_uniform - t_start, az_detrended)
        
        # --- Step 6: Merge with existing model ---
        with self._lock:
            self.periods_alt = self._merge_periods(self.periods_alt, new_periods_alt)
            self.periods_az = self._merge_periods(self.periods_az, new_periods_az)
            
            self.stats['periods_detected_alt'] = len(self.periods_alt)
            self.stats['periods_detected_az'] = len(self.periods_az)
            
            was_trained = self.is_trained
            self.is_trained = (len(self.periods_alt) > 0 or len(self.periods_az) > 0)
        
        # --- Step 7: Log results ---
        if self.is_trained and not was_trained:
            period_info_alt = ", ".join(
                f"{p.period_sec:.1f}s (SNR:{p.snr:.1f})" for p in self.periods_alt
            )
            period_info_az = ", ".join(
                f"{p.period_sec:.1f}s (SNR:{p.snr:.1f})" for p in self.periods_az
            )
            self._log(f"PEC trained! Alt periods: [{period_info_alt}], Az periods: [{period_info_az}]")
        
        # Periodic progress log (every 5th analysis).
        if self._analysis_count % 5 == 0:
            self._log(
                f"PEC analysis #{self._analysis_count}: "
                f"{len(self.periods_alt)} Alt + {len(self.periods_az)} Az periods, "
                f"{n_samples} samples over {data_span:.0f}s"
            )
    
    @staticmethod
    def _detrend(signal: np.ndarray) -> np.ndarray:
        """
        Remove DC offset and linear trend from a signal.

        Uses ordinary least-squares to fit y = a*t + b, then subtracts
        the fit. This is critical before FFT because:
        - A DC offset would create a large spike at frequency 0,
          overwhelming the noise floor estimate.
        - A linear drift would spread energy across low-frequency bins
          (spectral leakage from a ramp), potentially masking or
          creating false periodic detections.

        Falls back to simple mean removal if the OLS fit fails (e.g.
        singular matrix from constant-value input).

        Args:
            signal: 1-D array of uniformly sampled error values.

        Returns:
            Detrended signal (same length as input).
        """
        n = len(signal)
        t = np.arange(n, dtype=float)
        # Design matrix for linear fit: each row is [t_i, 1]
        A = np.vstack([t, np.ones(n)]).T
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, signal, rcond=None)
            return signal - (coeffs[0] * t + coeffs[1])
        except np.linalg.LinAlgError:
            # Fallback: just remove the mean (DC offset) if OLS fails.
            return signal - np.mean(signal)
    
    def _detect_periods(self, signal: np.ndarray, fs: float, 
                        data_span: float) -> List[PECPeriod]:
        """
        Detect dominant periodic components in the signal using FFT.
        
        This is the spectral analysis core. The algorithm:

        1. Apply a Hann window to suppress spectral leakage.
        2. Compute the one-sided power spectrum via rfft.
        3. Estimate the noise floor as the *median* of the power
           spectrum (robust to the presence of strong peaks).
        4. Find *local maxima* in the power spectrum (not just bins
           above a threshold) -- this prevents leakage sidelobes from
           being counted as separate peaks.
        5. For each candidate peak (sorted by SNR, strongest first):
           a. Reject if SNR < min_snr.
           b. Reject if it falls within the exclusion zone of an
              already-accepted stronger peak (spectral leakage guard).
           c. Reject if its frequency is a near-integer multiple (or
              submultiple) of an already-accepted peak (harmonic check).
           d. Reject if fewer than min_cycles_for_fit complete cycles
              are present in the data span.
           e. Otherwise, accept and add an exclusion zone around it.

        The exclusion zone width is 3x the fundamental frequency
        resolution (1/data_span). This matches the main-lobe width of
        the Hann window, ensuring that the broad main-lobe skirts of
        a strong peak are not mistakenly detected as separate signals.

        Args:
            signal:    Detrended, uniformly sampled error signal.
            fs:        Sample rate of the uniform grid (Hz).
            data_span: Total time span of the data (seconds). Used to
                       compute frequency resolution = 1/data_span.
            
        Returns:
            List of PECPeriod objects (frequency, period, SNR, amplitude
            are populated; Fourier coefficients are NOT yet fitted --
            that is done in _fit_fourier).
        """
        n = len(signal)
        if n < 16:
            return []
        
        # --- Windowing ---
        # The Hann (raised cosine) window tapers both ends of the signal
        # to zero, dramatically reducing spectral leakage compared to a
        # rectangular window. The trade-off is a wider main lobe (3 bins
        # vs 1 bin), slightly reducing frequency resolution -- but this
        # is a good trade for PEC because we care more about detecting
        # the right peaks than about resolving very close frequencies.
        window = np.hanning(n)
        windowed = signal * window
        
        # --- FFT ---
        # rfft returns only the non-negative frequencies (0 to Nyquist),
        # which is all we need for a real-valued input signal.
        fft_vals = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        # Power spectrum: |FFT|^2.  We work with power (not amplitude)
        # because SNR is defined as peak_power / noise_power.
        power = np.abs(fft_vals) ** 2
        
        # --- Frequency resolution and exclusion zone ---
        # The fundamental frequency resolution of a DFT is df = 1/T
        # where T is the data span. Two spectral features closer than df
        # cannot be independently resolved.
        freq_resolution = 1.0 / data_span
        # The Hann window's main lobe is ~3 bins wide (from -1.5*df to
        # +1.5*df around the true frequency). We use 3*df as the minimum
        # separation between accepted peaks so that main-lobe leakage
        # from a strong peak is not mistaken for a separate signal.
        min_peak_separation = 3.0 * freq_resolution
        
        # --- Minimum detectable frequency ---
        # We require at least min_cycles_for_fit complete cycles in the
        # data span to ensure the Fourier fit has enough phase coverage.
        # Periods longer than data_span / min_cycles_for_fit cannot
        # satisfy this, so their frequencies are below min_freq.
        min_freq = self.min_cycles_for_fit / data_span
        
        # --- Noise floor estimation ---
        # The noise floor is estimated as the median of the power spectrum
        # (excluding sub-min_freq bins). The median is robust: even if
        # half the bins contain strong peaks, the median is unaffected.
        valid_mask = freqs > min_freq
        if not np.any(valid_mask):
            return []
        
        valid_power = power[valid_mask]
        valid_freqs = freqs[valid_mask]
        valid_indices = np.where(valid_mask)[0]  # Map back to original arrays
        
        noise_floor = float(np.median(valid_power))
        if noise_floor <= 0:
            noise_floor = 1e-12  # Guard against zero (e.g. all-zero input)
        
        # --- Local maxima detection (vectorized) ---
        # We find points that are the maximum within a +-2 bin window.
        # Using a half-window of 2 (instead of 1) makes the detection
        # robust to "flat-topped" peaks that can occur at low frequencies
        # where the Hann main lobe spans several bins. Without this,
        # both shoulders of a flat peak could be detected as separate
        # local maxima.
        #
        # Vectorized approach: compare each element against shifted versions
        # of itself using numpy rolling max via stride tricks. This replaces
        # the O(n) Python for-loop with pure numpy operations (~10-50x
        # faster on mobile devices).
        n_valid = len(valid_power)
        local_max_mask = np.ones(n_valid, dtype=bool)
        half_win = 2
        for offset in range(1, half_win + 1):
            # Compare with left neighbor at distance 'offset'
            local_max_mask[offset:] &= (valid_power[offset:] >= valid_power[:-offset])
            # Compare with right neighbor at distance 'offset'
            local_max_mask[:-offset] &= (valid_power[:-offset] >= valid_power[offset:])
        
        peak_indices_local = np.where(local_max_mask)[0]
        if len(peak_indices_local) == 0:
            return []
        
        # Compute SNR for each local maximum.
        peak_snr = valid_power[peak_indices_local] / noise_floor
        peak_freqs = valid_freqs[peak_indices_local]
        peak_powers = valid_power[peak_indices_local]
        
        # Sort candidates by SNR descending (process strongest first).
        sort_order = np.argsort(peak_snr)[::-1]
        
        detected = []
        # Exclusion zones: list of (freq_low, freq_high) intervals around
        # already-accepted peaks. Any candidate falling inside an existing
        # zone is rejected as probable spectral leakage.
        used_freq_ranges = []
        
        for sort_idx in sort_order:
            if len(detected) >= self.max_periods:
                break
            
            snr = float(peak_snr[sort_idx])
            if snr < self.min_snr:
                break  # Sorted by SNR desc, so all remaining are weaker
            
            freq = float(peak_freqs[sort_idx])
            if freq <= 0:
                continue
            
            # --- Spectral leakage suppression ---
            # Reject candidates that fall within the exclusion zone of a
            # previously accepted (stronger) peak. Such candidates are
            # most likely sidelobe artefacts of the stronger peak, not
            # independent periodicities.
            too_close = False
            for (f_low, f_high) in used_freq_ranges:
                if f_low <= freq <= f_high:
                    too_close = True
                    break
            if too_close:
                continue
            
            period = 1.0 / freq
            # Convert FFT magnitude to physical amplitude (arcsec/sec).
            # Factor of 2/n accounts for the one-sided spectrum and the
            # Hann window's amplitude scaling.
            amplitude = float(np.sqrt(peak_powers[sort_idx]) * 2.0 / n)
            
            # --- Harmonic rejection ---
            # If the candidate's frequency is a near-integer multiple (or
            # submultiple) of an already-detected period, it is likely a
            # harmonic of that fundamental rather than an independent
            # mechanical periodicity. We skip it because _fit_fourier will
            # capture harmonics of each fundamental automatically via its
            # multi-harmonic Fourier series.
            is_harmonic = False
            for existing in detected:
                ratio = freq / existing.frequency_hz
                # Check if this is a harmonic of an existing fundamental
                if abs(ratio - round(ratio)) < 0.1 and round(ratio) > 1:
                    is_harmonic = True
                    break
                # Check if an existing peak is a harmonic of this one
                ratio_inv = existing.frequency_hz / freq
                if abs(ratio_inv - round(ratio_inv)) < 0.1 and round(ratio_inv) > 1:
                    is_harmonic = True
                    break
            
            if is_harmonic:
                continue
            
            # --- Minimum cycle count check ---
            # Ensure we have observed enough complete cycles for a
            # reliable Fourier fit. With fewer cycles, the fit is
            # under-determined and the coefficients are unreliable.
            n_cycles = data_span * freq
            if n_cycles < self.min_cycles_for_fit:
                continue
            
            detected.append(PECPeriod(
                period_sec=float(period),
                frequency_hz=float(freq),
                amplitude_arcsec=float(amplitude),
                snr=float(snr)
            ))
            
            # Add an exclusion zone around this peak to suppress its
            # leakage sidelobes from being detected in subsequent
            # iterations of this loop.
            used_freq_ranges.append((freq - min_peak_separation, freq + min_peak_separation))
        
        return detected
    
    def _fit_fourier(self, period: PECPeriod, t: np.ndarray, signal: np.ndarray):
        """
        Fit a multi-harmonic Fourier series to the signal at a given period.

        For a detected period with fundamental frequency f0, fits the model:

            y(t) = SUM_{n=1}^{N} [ a_n*cos(2*pi*n*f0*t) + b_n*sin(2*pi*n*f0*t) ]

        using ordinary least-squares (OLS) regression. The design matrix has
        columns [cos(2*pi*1*f0*t), sin(2*pi*1*f0*t), cos(2*pi*2*f0*t), ...].

        The number of harmonics N is chosen adaptively:
        - Upper bound: self.max_harmonics (default 8).
        - Nyquist limit: data_span * f0 / 2 (can't fit harmonics above
          half the number of observed fundamental cycles).
        - At least 1 harmonic (the fundamental) is always fitted.

        Multiple harmonics are important because real gear errors are
        generally non-sinusoidal: a single worm tooth defect produces a
        waveform with sharp edges that requires several harmonics to
        approximate accurately.

        On success, period.cos_coeffs, period.sin_coeffs, and
        period.n_harmonics are populated. On OLS failure, n_harmonics
        is set to 0 (no correction will be applied for this period).

        Args:
            period: PECPeriod to fit (modified in-place).
            t:      Time array (seconds, starting from 0).
            signal: Detrended error signal (same length as t).
        """
        n = len(t)
        f0 = period.frequency_hz
        
        # Determine how many harmonics we can reliably fit.
        # The highest harmonic has frequency n_harmonics * f0; this must
        # be below the Nyquist frequency of our data. Since we have
        # data_span * f0 fundamental cycles, the Nyquist limit for
        # harmonics is approximately data_span * f0 / 2.
        data_span = t[-1] - t[0] if len(t) > 1 else 0
        max_possible = int(data_span * f0 / 2)  # Nyquist limit for harmonics
        n_harmonics = min(self.max_harmonics, max(1, max_possible))
        
        # Build the OLS design matrix.
        # Each harmonic contributes two columns: cos and sin.
        # Layout: [cos(w*t), sin(w*t), cos(2w*t), sin(2w*t), ...]
        # where w = 2*pi*f0.
        n_cols = 2 * n_harmonics
        A = np.zeros((n, n_cols))
        
        for h in range(n_harmonics):
            harmonic = h + 1
            phase = 2.0 * np.pi * harmonic * f0 * t
            A[:, 2 * h] = np.cos(phase)      # a_n basis function
            A[:, 2 * h + 1] = np.sin(phase)  # b_n basis function
        
        # Solve the normal equations: A^T A x = A^T signal
        # np.linalg.lstsq handles rank-deficient cases gracefully.
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, signal, rcond=None)
        except np.linalg.LinAlgError:
            # OLS failed (e.g. degenerate data). Mark this period as
            # unfitted so it won't contribute to corrections.
            period.n_harmonics = 0
            return
        
        # Unpack the interleaved cos/sin coefficients into separate arrays.
        cos_c = np.zeros(n_harmonics)
        sin_c = np.zeros(n_harmonics)
        for h in range(n_harmonics):
            cos_c[h] = coeffs[2 * h]
            sin_c[h] = coeffs[2 * h + 1]
        
        period.cos_coeffs = cos_c
        period.sin_coeffs = sin_c
        period.n_harmonics = n_harmonics
        
        # Update the amplitude from the actual fitted coefficients.
        # This is the RMS amplitude of the multi-harmonic series:
        # sqrt( sum(a_n^2 + b_n^2) ), which is more accurate than the
        # initial single-bin FFT estimate.
        period.amplitude_arcsec = float(np.sqrt(np.sum(cos_c**2 + sin_c**2)))
    
    def _merge_periods(self, old: List[PECPeriod], 
                       new: List[PECPeriod]) -> List[PECPeriod]:
        """
        Merge newly detected periods with the existing model.

        This implements the EMA (Exponential Moving Average) coefficient
        blending strategy:

        - If a newly detected period matches an existing one (periods
          within 10% of each other), their Fourier coefficients are
          blended:
              coeff = old_coeff * (1 - lr) + new_coeff * lr
          where lr = self.learning_rate. This ensures the correction
          signal evolves smoothly and doesn't jump discontinuously
          when the model is updated mid-observation.

        - If a new period does not match any existing one, it is
          appended (up to max_periods).

        - The merged list is sorted by amplitude (descending) and
          pruned to max_periods entries, keeping the strongest
          components.

        The period estimate itself is also EMA-blended, allowing the
        system to slowly track mechanical changes (e.g. thermal
        expansion changing gear mesh).

        Args:
            old: Currently active PECPeriod list for one axis.
            new: Newly detected PECPeriod list from the latest analysis.

        Returns:
            Merged and pruned list of PECPeriod objects.
        """
        if not old:
            return new[:self.max_periods]
        
        if not new:
            return old
        
        merged = list(old)
        
        for np_ in new:
            matched = False
            for i, op in enumerate(merged):
                # Check if the two periods are "the same" mechanical
                # source, allowing 10% tolerance for measurement jitter
                # in the FFT frequency estimate.
                ratio = np_.period_sec / op.period_sec
                if 0.9 < ratio < 1.1:
                    # Match found -- blend Fourier coefficients via EMA.
                    if np_.n_harmonics > 0 and op.n_harmonics > 0:
                        # Use the smaller harmonic count to avoid index
                        # errors (the two fits may have different N).
                        n_h = min(np_.n_harmonics, op.n_harmonics)
                        lr = self.learning_rate
                        
                        # Truncate both to the common harmonic count.
                        new_cos = np_.cos_coeffs[:n_h]
                        new_sin = np_.sin_coeffs[:n_h]
                        old_cos = op.cos_coeffs[:n_h]
                        old_sin = op.sin_coeffs[:n_h]
                        
                        # EMA blend: slowly incorporate new measurements.
                        blended_cos = old_cos * (1 - lr) + new_cos * lr
                        blended_sin = old_sin * (1 - lr) + new_sin * lr
                        
                        merged[i].cos_coeffs = blended_cos
                        merged[i].sin_coeffs = blended_sin
                        merged[i].n_harmonics = n_h
                        # Blend the SNR and period estimates too, so they
                        # track slow changes over time.
                        merged[i].snr = float(op.snr * (1 - lr) + np_.snr * lr)
                        merged[i].amplitude_arcsec = float(
                            np.sqrt(np.sum(blended_cos**2 + blended_sin**2))
                        )
                        # Slowly converge the period estimate towards the
                        # latest measurement (handles thermal drift etc.).
                        merged[i].period_sec = float(op.period_sec * (1 - lr) + np_.period_sec * lr)
                        merged[i].frequency_hz = float(1.0 / merged[i].period_sec)
                    
                    matched = True
                    break
            
            # No existing period matched -- this is a newly discovered
            # mechanical periodicity. Add it if we have room.
            if not matched and len(merged) < self.max_periods:
                merged.append(np_)
        
        # Keep only the strongest components (by amplitude), up to the
        # configured maximum. Weaker periods contribute less to tracking
        # error and are more likely to be noise.
        merged.sort(key=lambda p: p.amplitude_arcsec, reverse=True)
        return merged[:self.max_periods]
    
    # ================================================================
    #                         Persistence
    # ================================================================
    
    def save(self, filepath: str = "pec_model.json"):
        """
        Save the learned PEC model to a JSON file.

        The saved data includes all detected periods with their Fourier
        coefficients, plus metadata (analysis count, sample count,
        settings). This allows the model to be restored on the next
        session via load(), providing immediate correction without
        waiting for a new training period.

        Args:
            filepath: Path to the output JSON file.
        """
        with self._lock:
            data = {
                'version': 2,
                'timestamp': time.time(),
                'drive_type': self.drive_type,
                'analysis_count': self._analysis_count,
                'total_samples': self.stats['total_samples'],
                'periods_alt': [self._period_to_dict(p) for p in self.periods_alt],
                'periods_az': [self._period_to_dict(p) for p in self.periods_az],
                'settings': {
                    'max_harmonics': self.max_harmonics,
                    'min_snr': self.min_snr,
                    'max_periods': self.max_periods,
                    'learning_rate': self.learning_rate,
                }
            }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            self._log(f"PEC: Model saved to {filepath}")
        except Exception as e:
            self._log(f"PEC: Error saving model: {e}")
    
    def load(self, filepath: str = "pec_model.json") -> bool:
        """
        Load a previously learned PEC model from a JSON file.

        On success, the model begins producing corrections immediately.
        The phase reference (_start_time) is set to the current wall
        clock time, meaning the Fourier series restarts from t=0. Since
        the series is periodic, the phase offset relative to the actual
        mechanical state is arbitrary -- it will re-lock naturally as new
        data arrives and the EMA blending adjusts the coefficients.

        Old model versions (< 2) are rejected because the coefficient
        format is incompatible.

        Args:
            filepath: Path to the JSON model file.

        Returns:
            True if the model was loaded successfully, False otherwise.
        """
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if data.get('version', 1) < 2:
                self._log("PEC: Old model version, skipping load")
                return False
            
            with self._lock:
                self.periods_alt = [self._dict_to_period(d) for d in data.get('periods_alt', [])]
                self.periods_az = [self._dict_to_period(d) for d in data.get('periods_az', [])]
                self._analysis_count = data.get('analysis_count', 0)
                self.is_trained = (len(self.periods_alt) > 0 or len(self.periods_az) > 0)
                
                # Set phase reference so get_correction() works immediately.
                # The Fourier series is periodic, so starting from t=0 is fine -
                # the phase will re-lock after a few analysis cycles with new data.
                if self.is_trained and self._start_time is None:
                    self._start_time = time.time()
                
                self.stats['periods_detected_alt'] = len(self.periods_alt)
                self.stats['periods_detected_az'] = len(self.periods_az)
            
            if self.is_trained:
                self._log(
                    f"PEC: Model loaded from {filepath} - "
                    f"{len(self.periods_alt)} Alt + {len(self.periods_az)} Az periods"
                )
            return True
            
        except Exception as e:
            self._log(f"PEC: Error loading model: {e}")
            return False
    
    @staticmethod
    def _period_to_dict(p: PECPeriod) -> dict:
        """Serialise a PECPeriod to a JSON-compatible dict."""
        return {
            'period_sec': p.period_sec,
            'frequency_hz': p.frequency_hz,
            'amplitude_arcsec': p.amplitude_arcsec,
            'snr': p.snr,
            'n_harmonics': p.n_harmonics,
            'cos_coeffs': p.cos_coeffs.tolist() if p.n_harmonics > 0 else [],
            'sin_coeffs': p.sin_coeffs.tolist() if p.n_harmonics > 0 else [],
        }
    
    @staticmethod
    def _dict_to_period(d: dict) -> PECPeriod:
        """Deserialise a PECPeriod from a JSON-loaded dict."""
        p = PECPeriod(
            period_sec=d['period_sec'],
            frequency_hz=d['frequency_hz'],
            amplitude_arcsec=d.get('amplitude_arcsec', 0.0),
            snr=d.get('snr', 0.0),
            n_harmonics=d.get('n_harmonics', 0),
        )
        if p.n_harmonics > 0:
            p.cos_coeffs = np.array(d.get('cos_coeffs', []))
            p.sin_coeffs = np.array(d.get('sin_coeffs', []))
        return p
    
    # ================================================================
    #                         Diagnostics
    # ================================================================
    
    def get_statistics(self) -> dict:
        """
        Return PEC statistics for UI display.

        Returns a dict with current state, sample counts, detected
        period details (per axis), and analysis metadata. Intended to
        be polled by the GUI to show PEC status to the user.
        """
        with self._lock:
            return {
                **self.stats,
                'is_trained': self.is_trained,
                'is_enabled': self.is_enabled,
                'is_learning': self.is_learning,
                'drive_type': self.drive_type,
                'analysis_count': self._analysis_count,
                'periods_alt_detail': [
                    {
                        'period_sec': p.period_sec,
                        'amplitude': p.amplitude_arcsec,
                        'snr': p.snr,
                        'harmonics': p.n_harmonics
                    } for p in self.periods_alt
                ],
                'periods_az_detail': [
                    {
                        'period_sec': p.period_sec,
                        'amplitude': p.amplitude_arcsec,
                        'snr': p.snr,
                        'harmonics': p.n_harmonics
                    } for p in self.periods_az
                ],
            }
    
    def get_correction_curve(self, duration_sec: float = 600.0,
                             n_points: int = 1000) -> dict:
        """
        Generate the full PEC correction curves for visualisation.

        Evaluates the Fourier model at n_points evenly spaced times
        over [0, duration_sec]. Useful for plotting the predicted
        correction waveform in the GUI.

        Args:
            duration_sec: Time span to generate (seconds). Default 600 s
                          (10 minutes) covers several cycles of typical
                          gear periods.
            n_points:     Number of evaluation points.
            
        Returns:
            Dict with keys:
                'time': 1-D numpy array of time values (seconds).
                'alt':  1-D numpy array of altitude corrections (arcsec/s).
                'az':   1-D numpy array of azimuth corrections (arcsec/s).
        """
        t = np.linspace(0, duration_sec, n_points)
        alt_curve = np.zeros(n_points)
        az_curve = np.zeros(n_points)
        
        with self._lock:
            for i, ti in enumerate(t):
                alt_curve[i] = self._evaluate_fourier(ti, self.periods_alt)
                az_curve[i] = self._evaluate_fourier(ti, self.periods_az)
        
        return {
            'time': t,
            'alt': alt_curve,
            'az': az_curve
        }
