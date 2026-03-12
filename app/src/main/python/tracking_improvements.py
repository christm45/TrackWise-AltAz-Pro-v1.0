"""
Long-term tracking improvements for the Dobson Alt-Az telescope controller.

This module provides tracking quality monitoring, alerting, error handling with
retry logic, and atmospheric/thermal correction functions. It is a supporting
module consumed by the real-time tracking pipeline and the main UI layer.

Architecture role
-----------------
- **realtime_tracking.py** calls ``atmospheric_refraction_correction()`` and
  ``temperature_drift_correction()`` every tracking cycle to refine the
  commanded Alt/Az position before sending motor commands.
- **main_realtime.py** instantiates ``LongTermTrackingMonitor`` to display
  live quality metrics (RMS error, solve success rate, drift) in the user
  interface and to decide when the Kalman filter should be reset.
- ``ErrorHandler`` wraps any unreliable I/O (serial communication with motor
  controllers, plate-solve HTTP calls) to provide automatic retry with
  linear back-off.

Data flow
---------
1. Each plate-solve result feeds ``LongTermTrackingMonitor.update_position()``
   with the measured Alt/Az errors (arcsec).
2. The monitor accumulates errors in a rolling window (up to 1 000 samples),
   computes RMS, classifies quality via ``TrackingQuality``, and fires
   callbacks (quality-change, alert, log) when thresholds are breached.
3. Once per hour (configurable) the monitor signals that a Kalman-filter
   reset is advisable to prevent long-term drift accumulation.

Classes
-------
- ``TrackingQuality``  -- Enum rating tracking accuracy from EXCELLENT to FAILED.
- ``TrackingMetrics``   -- Dataclass snapshot of all current tracking statistics.
- ``LongTermTrackingMonitor`` -- Stateful monitor that evaluates tracking health.
- ``ErrorHandler``      -- Generic retry-with-back-off wrapper.

Standalone functions
--------------------
- ``atmospheric_refraction_correction()`` -- Alt correction for atmospheric
  refraction (important below ~10 deg altitude).
- ``temperature_drift_correction()`` -- Estimated mechanical drift caused by
  thermal expansion of the telescope tube and mount.

Dependencies
------------
- Python standard library: ``time``, ``math``, ``threading``, ``collections``,
  ``dataclasses``, ``enum``, ``typing``.
- Third-party: ``numpy`` (used for vectorised RMS computation over the error
  history deque).
"""

import time
import math
import threading
from typing import Optional, Callable, Dict, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import numpy as np


class TrackingQuality(Enum):
    """Classification of real-time tracking accuracy.

    The boundaries are chosen to match common amateur-astronomy expectations
    for a Dobson Alt-Az mount with plate-solve feedback:

    - EXCELLENT (<1 arcsec) -- sub-arcsecond; suitable for long-exposure
      astrophotography on an equatorial platform.
    - GOOD (1-3 arcsec) -- adequate for short-exposure planetary imaging.
    - FAIR (3-10 arcsec) -- visually acceptable; object stays in eyepiece FOV.
    - POOR (>10 arcsec) -- noticeable drift; user may need to re-centre.
    - FAILED -- no valid data available (e.g. plate-solve is not returning).
    """
    EXCELLENT = "excellent"  # < 1 arcsec RMS total error
    GOOD = "good"           # 1-3 arcsec RMS total error
    FAIR = "fair"           # 3-10 arcsec RMS total error
    POOR = "poor"           # > 10 arcsec RMS total error
    FAILED = "failed"       # No valid measurement data available


@dataclass
class TrackingMetrics:
    """Snapshot of all tracking performance statistics at a given instant.

    Instances are created and continuously updated by
    ``LongTermTrackingMonitor``. The UI layer reads a copy via
    ``get_metrics()`` to refresh the on-screen display.

    Attributes:
        rms_error_alt: Root-mean-square error in altitude (arcsec).
        rms_error_az: Root-mean-square error in azimuth (arcsec).
        rms_error_total: Combined RMS error (Euclidean norm of alt and az).
        cumulative_drift_alt: Accumulated absolute altitude drift (arcsec),
            reset every periodic Kalman reset.
        cumulative_drift_az: Accumulated absolute azimuth drift (arcsec),
            reset every periodic Kalman reset.
        quality: Current ``TrackingQuality`` classification.
        tracking_duration: Elapsed time since tracking started (seconds).
        last_solve_time: Unix timestamp of the most recent plate-solve update.
        solve_success_rate: Fraction of successful plate-solves (0.0 to 1.0).
        total_corrections: Total number of correction updates received.
        avg_correction_magnitude: Average magnitude of applied corrections
            (currently unused but reserved for future averaging logic).
        temperature: Ambient temperature in degrees Celsius, if available.
        timestamp: Unix timestamp when this metrics snapshot was last updated.
    """
    # RMS error components (arcsec)
    rms_error_alt: float = 0.0
    rms_error_az: float = 0.0
    rms_error_total: float = 0.0
    
    # Cumulative drift since last Kalman reset (arcsec)
    cumulative_drift_alt: float = 0.0
    cumulative_drift_az: float = 0.0
    
    # Overall tracking quality classification
    quality: TrackingQuality = TrackingQuality.FAILED
    
    # Temporal statistics
    tracking_duration: float = 0.0  # seconds since tracking started
    last_solve_time: float = 0.0    # unix timestamp of last plate-solve
    solve_success_rate: float = 0.0 # ratio of successful solves (0.0-1.0)
    
    # Correction counters
    total_corrections: int = 0          # number of correction cycles applied
    avg_correction_magnitude: float = 0.0  # average correction size (arcsec)
    
    # Ambient temperature if a sensor is connected (degrees Celsius)
    temperature: Optional[float] = None
    
    # Unix timestamp when these metrics were last refreshed
    timestamp: float = field(default_factory=time.time)


class LongTermTrackingMonitor:
    """Monitors tracking quality over extended observation sessions.

    This class accumulates plate-solve error measurements in rolling-window
    deques, computes RMS statistics, classifies the overall tracking quality,
    and triggers callbacks when quality changes or alert thresholds are
    exceeded.

    It also implements a periodic reset mechanism: after a configurable
    interval (default 1 hour) the monitor signals that the Kalman filter
    in the tracking pipeline should be reinitialised to prevent unbounded
    drift accumulation from small systematic errors.

    Key features:
        - Rolling RMS error computation (last 1 000 samples).
        - Cumulative-drift detection over the last ~60 samples.
        - Periodic Kalman-filter reset recommendation (hourly by default).
        - Configurable alert callbacks for excessive drift, RMS, or low
          plate-solve success rate.
        - Thread-safe via an internal ``threading.Lock``.

    Typical usage::

        monitor = LongTermTrackingMonitor(
            on_quality_change=update_ui_quality_badge,
            on_alert=show_user_alert,
            on_log=write_to_logfile,
        )
        monitor.start_tracking()
        # ... in the tracking loop:
        monitor.update_position(alt, az, err_alt, err_az)
        if monitor.should_reset_kalman():
            kalman_filter.reset()
    """
    
    def __init__(self, 
                 on_quality_change: Optional[Callable[[TrackingQuality], None]] = None,
                 on_alert: Optional[Callable[[str, str], None]] = None,
                 on_log: Optional[Callable[[str], None]] = None):
        """Initialise the long-term tracking monitor.

        Args:
            on_quality_change: Callback invoked when the tracking quality
                classification changes (e.g. GOOD -> FAIR).  Receives the
                new ``TrackingQuality`` value.
            on_alert: Callback invoked when an alert condition is detected.
                Receives two strings: severity level ("warning" or "error")
                and a human-readable message.
            on_log: Callback for general informational log messages.
                Receives a single string.
        """
        self.on_quality_change = on_quality_change
        self.on_alert = on_alert
        self.on_log = on_log
        
        # Rolling history of per-axis errors for RMS computation.
        # maxlen=1000 keeps roughly the last ~17 minutes at 1 Hz solve rate,
        # which is long enough to smooth transient spikes but short enough
        # to remain responsive to real degradation.
        self.error_history_alt = deque(maxlen=1000)
        self.error_history_az = deque(maxlen=1000)
        
        # Position history for cumulative-drift detection.
        # maxlen=500 keeps ~8 minutes at 1 Hz -- enough to compute a
        # meaningful drift rate.
        self.position_history = deque(maxlen=500)
        
        # Current aggregate metrics (updated in-place on every call to
        # update_position).
        self.metrics = TrackingMetrics()
        self.last_quality = TrackingQuality.FAILED
        
        # ----- Configurable thresholds -----
        # Reset the Kalman filter periodically to flush accumulated bias.
        # 3600 s (1 hour) is a practical trade-off: short enough to bound
        # drift, long enough that the filter converges well between resets.
        self.reset_interval = 3600.0  # seconds between Kalman resets
        # Cumulative drift alert threshold.  60 arcsec (~1 arcmin) means
        # the object has wandered noticeably in a typical eyepiece FOV.
        self.max_cumulative_drift = 60.0  # arcsec
        # RMS error alert threshold.  10 arcsec is the FAIR/POOR boundary;
        # beyond this the user should investigate (clouds, vibration, etc.).
        self.max_rms_error = 10.0  # arcsec
        # Minimum acceptable plate-solve success rate.  Below 70 % the
        # Kalman filter is mostly predicting rather than correcting, so
        # tracking reliability is degraded.
        self.min_solve_success_rate = 0.7  # 70 %
        
        # Solve counters (cumulative since start_tracking)
        self.total_solves = 0
        self.successful_solves = 0
        self.tracking_start_time = None
        
        # Periodic reset bookkeeping
        self.last_reset_time = time.time()
        self.reset_counter = 0
        
        # Lock for thread-safety (UI thread reads metrics while the
        # tracking thread writes them).
        self._lock = threading.Lock()
    
    def start_tracking(self):
        """Reset all counters and begin a new monitoring session.

        Should be called once when the user starts a tracking session.
        """
        self.tracking_start_time = time.time()
        self.last_reset_time = time.time()
        self.total_solves = 0
        self.successful_solves = 0
        self.reset_counter = 0
        if self.on_log:
            self.on_log("📊 Long-term tracking monitoring started")
    
    def update_position(self, alt: float, az: float, 
                       error_alt: float, error_az: float,
                       solve_success: bool = True,
                       temperature: Optional[float] = None):
        """Record a new plate-solve measurement and refresh all metrics.

        This is the main entry point called by the tracking loop each time
        a plate-solve completes (successfully or not).  It updates error
        histories, recomputes RMS, evaluates quality, checks for alerts,
        and determines whether a Kalman reset is due.

        Args:
            alt: Current measured altitude (degrees).
            az: Current measured azimuth (degrees).
            error_alt: Signed pointing error in altitude (arcsec).
            error_az: Signed pointing error in azimuth (arcsec).
            solve_success: Whether the plate-solve succeeded.  Failed solves
                still increment the total count (affecting the success rate)
                but their error values are still recorded.
            temperature: Ambient temperature in degrees Celsius, or ``None``
                if no sensor is available.
        """
        with self._lock:
            current_time = time.time()
            
            # Append new error samples to rolling histories
            self.error_history_alt.append(error_alt)
            self.error_history_az.append(error_az)
            self.position_history.append((current_time, alt, az))
            
            # Update solve success/failure counters
            self.total_solves += 1
            if solve_success:
                self.successful_solves += 1
            
            # Compute RMS once we have a statistically meaningful sample.
            # 10 samples is the minimum to avoid noisy early estimates.
            if len(self.error_history_alt) > 10:
                # Vectorised RMS: sqrt(mean(x^2)) via numpy for efficiency
                self.metrics.rms_error_alt = math.sqrt(
                    np.mean(np.array(self.error_history_alt)**2)
                )
                self.metrics.rms_error_az = math.sqrt(
                    np.mean(np.array(self.error_history_az)**2)
                )
                # Total RMS is the Euclidean norm of the two axes
                self.metrics.rms_error_total = math.sqrt(
                    self.metrics.rms_error_alt**2 + self.metrics.rms_error_az**2
                )
            
            # Estimate cumulative drift from recent position samples.
            # We look at up to the last 60 entries (~1 minute at 1 Hz) to
            # compute an instantaneous drift rate (arcsec/s) and accumulate
            # the absolute drift.
            if len(self.position_history) > 10:
                recent_positions = list(self.position_history)[-60:]  # ~1 min at 1 Hz
                if len(recent_positions) > 1:
                    first = recent_positions[0]
                    last = recent_positions[-1]
                    dt = last[0] - first[0]  # elapsed time (seconds)
                    if dt > 0:
                        # Convert positional change from degrees to arcsec/s
                        drift_alt = (last[1] - first[1]) * 3600 / dt  # arcsec/sec
                        drift_az = (last[2] - first[2]) * 3600 / dt
                        # Accumulate absolute drift (always positive)
                        self.metrics.cumulative_drift_alt += abs(drift_alt) * dt
                        self.metrics.cumulative_drift_az += abs(drift_az) * dt
            
            # Re-evaluate the quality classification
            self._update_quality()
            
            # Refresh remaining scalar metrics
            if self.tracking_start_time:
                self.metrics.tracking_duration = current_time - self.tracking_start_time
            self.metrics.last_solve_time = current_time
            self.metrics.solve_success_rate = (
                self.successful_solves / self.total_solves 
                if self.total_solves > 0 else 0.0
            )
            self.metrics.total_corrections = self.total_solves
            self.metrics.temperature = temperature
            self.metrics.timestamp = current_time
            
            # Check whether the periodic Kalman reset interval has elapsed
            if current_time - self.last_reset_time > self.reset_interval:
                self._check_reset_needed()
            
            # Evaluate alert conditions (drift, RMS, success rate)
            self._check_alerts()
    
    def _update_quality(self):
        """Classify tracking quality based on the current total RMS error.

        The thresholds match the ``TrackingQuality`` enum boundaries:
        <1" EXCELLENT, 1-3" GOOD, 3-10" FAIR, >10" POOR, 0 (no data) FAILED.

        If the quality level has changed since the last call, the
        ``on_quality_change`` callback is invoked.
        """
        rms = self.metrics.rms_error_total
        
        # Classify based on total RMS error thresholds (arcsec)
        if rms < 1.0:
            new_quality = TrackingQuality.EXCELLENT
        elif rms < 3.0:
            new_quality = TrackingQuality.GOOD
        elif rms < 10.0:
            new_quality = TrackingQuality.FAIR
        elif rms > 0:
            new_quality = TrackingQuality.POOR
        else:
            # rms == 0.0 means no data has been processed yet
            new_quality = TrackingQuality.FAILED
        
        # Fire callback only on transitions to avoid spamming the UI
        if new_quality != self.last_quality:
            self.last_quality = new_quality
            self.metrics.quality = new_quality
            if self.on_quality_change:
                try:
                    self.on_quality_change(new_quality)
                except Exception:
                    pass  # Never let a callback failure crash the monitor
    
    def _check_reset_needed(self):
        """Determine whether a periodic Kalman-filter reset is due.

        If the configured ``reset_interval`` has elapsed since the last
        reset, the cumulative drift counters are zeroed and the log callback
        is invoked.

        Returns:
            bool: ``True`` if a reset was triggered, ``False`` otherwise.
        """
        current_time = time.time()
        
        # Periodic reset (default: every hour)
        if current_time - self.last_reset_time >= self.reset_interval:
            self.reset_counter += 1
            self.last_reset_time = current_time
            
            # Zero cumulative drift because the Kalman filter will be
            # reinitialised, so past drift is no longer meaningful.
            self.metrics.cumulative_drift_alt = 0.0
            self.metrics.cumulative_drift_az = 0.0
            
            if self.on_log:
                self.on_log(f"🔄 Periodic reset #{self.reset_counter} "
                           f"(RMS: {self.metrics.rms_error_total:.2f}\")")
            
            # Return True to signal the caller that a reset is needed
            return True
        
        return False
    
    def _check_alerts(self):
        """Evaluate alert conditions and fire callbacks as necessary.

        Three independent alert conditions are checked:

        1. **Cumulative drift** exceeds ``max_cumulative_drift`` (default
           60 arcsec / 1 arcmin).  Indicates systematic tracking error.
        2. **RMS error** exceeds ``max_rms_error`` (default 10 arcsec).
           Indicates generally poor tracking (clouds, vibration, etc.).
        3. **Plate-solve success rate** falls below
           ``min_solve_success_rate`` (default 70 %), checked only after
           50 solves to avoid premature alerts during ramp-up.
        """
        # Alert: excessive cumulative drift (Euclidean norm of both axes)
        total_drift = math.sqrt(
            self.metrics.cumulative_drift_alt**2 + 
            self.metrics.cumulative_drift_az**2
        )
        if total_drift > self.max_cumulative_drift:
            if self.on_alert:
                self.on_alert("warning", 
                    f"⚠️ High cumulative drift: {total_drift:.1f}\" "
                    f"(max: {self.max_cumulative_drift}\")")
        
        # Alert: RMS error too high
        if self.metrics.rms_error_total > self.max_rms_error:
            if self.on_alert:
                self.on_alert("error",
                    f"❌ Excessive RMS error: {self.metrics.rms_error_total:.1f}\" "
                    f"(max: {self.max_rms_error}\")")
        
        # Alert: plate-solve success rate too low (only after enough samples
        # to be statistically meaningful -- 50 solves minimum)
        if (self.total_solves > 50 and 
            self.metrics.solve_success_rate < self.min_solve_success_rate):
            if self.on_alert:
                self.on_alert("warning",
                    f"⚠️ Low plate-solve success rate: "
                    f"{self.metrics.solve_success_rate*100:.1f}% "
                    f"(min: {self.min_solve_success_rate*100:.0f}%)")
    
    def get_metrics(self) -> TrackingMetrics:
        """Return a snapshot of the current tracking metrics.

        The returned ``TrackingMetrics`` instance is the monitor's own
        object (not a copy), so callers should treat it as read-only or
        copy it if they need to store historical values.

        Returns:
            TrackingMetrics: The current metrics dataclass instance.
        """
        with self._lock:
            return self.metrics
    
    def should_reset_kalman(self) -> bool:
        """Check whether the Kalman filter should be reinitialised now.

        Delegates to ``_check_reset_needed()``.  Intended to be called by
        the tracking loop on every cycle; returns ``True`` at most once per
        ``reset_interval`` seconds.

        Returns:
            bool: ``True`` if a reset should be performed, ``False`` otherwise.
        """
        return self._check_reset_needed()


class ErrorHandler:
    """Robust error handler with automatic retry and linear back-off.

    Wraps any callable so that transient failures (e.g. serial timeouts,
    network errors during plate-solve requests) are retried a configurable
    number of times with increasing delay between attempts.

    It also keeps a running count of exceptions by type, which can be
    useful for diagnostics.

    Typical usage::

        handler = ErrorHandler(max_retries=3, retry_delay=0.5,
                               on_error=log_error)
        result = handler.execute_with_retry(serial_port.read, 128)
    """
    
    def __init__(self, 
                 max_retries: int = 3,
                 retry_delay: float = 0.5,
                 on_error: Optional[Callable[[str, Exception], None]] = None):
        """Initialise the error handler.

        Args:
            max_retries: Maximum number of attempts before giving up.
            retry_delay: Base delay between retries (seconds).  The actual
                delay on attempt *n* (0-indexed) is
                ``retry_delay * (n + 1)`` -- i.e. linear back-off.
            on_error: Optional callback invoked on every failure.  Receives
                a descriptive string and the caught ``Exception``.
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.on_error = on_error
        self.error_counts = {}  # Running count of exceptions keyed by type name
    
    def execute_with_retry(self, func: Callable, *args, **kwargs):
        """Execute *func* with automatic retry on exception.

        If *func* raises, the handler waits with linear back-off
        (``retry_delay * attempt_number``) and retries up to
        ``max_retries`` times.  The ``on_error`` callback is invoked after
        each failed attempt.

        Args:
            func: The callable to execute.
            *args: Positional arguments forwarded to *func*.
            **kwargs: Keyword arguments forwarded to *func*.

        Returns:
            The return value of *func* on success, or ``None`` if all
            attempts failed.
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_type = type(e).__name__
                # Increment the per-type error counter for diagnostics
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
                
                if attempt < self.max_retries - 1:
                    # Linear back-off: delay increases with each attempt
                    time.sleep(self.retry_delay * (attempt + 1))
                    if self.on_error:
                        self.on_error(f"Attempt {attempt + 1}/{self.max_retries} failed: {error_type}", e)
                else:
                    # Final attempt failed -- report total failure
                    if self.on_error:
                        self.on_error(f"Failed after {self.max_retries} attempts: {error_type}", e)
        
        return None
    
    def validate_data(self, value: float, min_val: float, max_val: float, 
                     name: str = "value") -> bool:
        """Validate that a numeric value falls within an expected range.

        Useful as a sanity check before feeding sensor readings or
        plate-solve results into the tracking pipeline.

        Args:
            value: The numeric value to validate.
            min_val: Minimum acceptable value (inclusive).
            max_val: Maximum acceptable value (inclusive).
            name: Human-readable label for the value (used in error messages).

        Returns:
            bool: ``True`` if the value is within [min_val, max_val],
            ``False`` otherwise (and ``on_error`` is called).
        """
        if not (min_val <= value <= max_val):
            if self.on_error:
                self.on_error(f"Invalid value: {name}={value} (expected: {min_val}-{max_val})", 
                             ValueError(f"Invalid {name}"))
            return False
        return True


class FlexureModel:
    """Position-dependent tube flexure learning model.

    Learns the relationship between telescope pointing direction (Alt, Az),
    ambient temperature, and the resulting mechanical flexure (pointing
    offset) by accumulating tracking residuals in a grid-based sky map
    and fitting a parametric model.

    Why this is needed
    ------------------
    Tube flexure (gravitational bending of the optical tube) changes
    continuously as the telescope moves across the sky:

    - At **low altitudes**, gravity acts perpendicular to the tube,
      creating maximum bending (the optical axis tilts downward).
    - At the **zenith**, gravity acts along the tube axis -- minimal
      bending.
    - **Temperature changes** alter the tube stiffness (thermal
      expansion, material modulus changes), modifying the flexure
      amount.
    - **Azimuth** matters because the tube is not perfectly symmetric:
      the focuser, finder, camera, and cables create an asymmetric
      mass distribution that shifts the flexure direction.

    The standard Saemundsson refraction formula already accounts for
    atmospheric bending, but it has systematic errors at low altitudes
    and does not account for mechanical flexure at all.  This model
    learns the *residual* after analytical refraction is applied.

    Architecture
    ------------
    The model maintains two representations:

    1. **Grid map** (alt_bins x az_bins): A lookup table of average
       residual corrections in (Alt, Az) bins.  Provides per-bin
       corrections with exponential moving average smoothing.  This
       captures the high-resolution spatial structure of the flexure.

    2. **Parametric model**: A compact set of physical coefficients
       that model flexure as:

           flexure_alt(alt, az, dT) =
               A * cos(alt)                    # gravity component
             + B * cos(alt) * cos(az)          # azimuthal asymmetry
             + C * cos(alt) * sin(az)          # azimuthal asymmetry (quadrature)
             + D * cos(alt) * dT               # thermal-gravity interaction
             + E * dT                           # pure thermal drift

           flexure_az(alt, az, dT) =
               F * sin(az) * cos(alt)
             + G * cos(az) * cos(alt)
             + H * dT

       The parametric model provides predictions in un-observed sky
       regions by extrapolation, and initialises new grid cells.

    Both are blended: in well-observed regions (many samples in the
    grid cell), the grid correction dominates; in poorly-observed
    regions, the parametric model fills in.

    Persistence
    -----------
    The entire model (grid + parametric coefficients + metadata) is
    serialised to JSON via save()/load(), allowing the flexure map to
    accumulate across observing sessions.  A fresh session loads the
    previous model and continues refining it.

    Units
    -----
    All corrections are in **arcsec/sec** (tracking rate units),
    matching the rest of the correction pipeline.
    """

    def __init__(self,
                 alt_bins: int = 18,
                 az_bins: int = 36,
                 learning_rate: float = 0.05):
        """
        Initialise the flexure learning model.

        Args:
            alt_bins:      Number of altitude bins (0-90 deg). Default 18
                           gives 5-degree resolution.
            az_bins:       Number of azimuth bins (0-360 deg). Default 36
                           gives 10-degree resolution.
            learning_rate: EMA blending factor for grid cell updates.
                           0.05 means each new sample contributes 5%.
        """
        self.alt_bins = alt_bins
        self.az_bins = az_bins
        self.learning_rate = learning_rate

        # Grid-based correction map: (alt_bins, az_bins) for each axis
        self.grid_alt = np.zeros((alt_bins, az_bins))
        self.grid_az = np.zeros((alt_bins, az_bins))
        self.grid_count = np.zeros((alt_bins, az_bins), dtype=int)

        # Parametric model coefficients
        # Alt flexure: [A, B, C, D, E] = gravity, az_cos, az_sin, thermal*grav, thermal
        self.param_alt = np.zeros(5)
        # Az flexure: [F, G, H] = az_sin*grav, az_cos*grav, thermal
        self.param_az = np.zeros(3)

        # Online learning state
        self.is_learning = True
        self.is_enabled = True
        self.total_samples = 0
        self._base_temperature: Optional[float] = None

        # Statistics
        self.stats = {
            'total_samples': 0,
            'grid_coverage_pct': 0.0,
            'param_rms_alt': 0.0,
            'param_rms_az': 0.0,
        }

    def add_residual(self, alt_deg: float, az_deg: float,
                     residual_alt: float, residual_az: float,
                     temperature: Optional[float] = None):
        """
        Record a tracking residual for flexure learning.

        This should be called with the drift residual AFTER Kalman + ML +
        PEC corrections have been applied, so the residual isolates the
        flexure and refraction-residual components.

        Args:
            alt_deg:      Current altitude in degrees (0-90).
            az_deg:       Current azimuth in degrees (0-360).
            residual_alt: Residual tracking error in Alt (arcsec/sec).
            residual_az:  Residual tracking error in Az (arcsec/sec).
            temperature:  Ambient temperature in Celsius (optional).
        """
        if not self.is_learning:
            return

        # Set base temperature on first sample
        if temperature is not None and self._base_temperature is None:
            self._base_temperature = temperature

        # Map to grid indices
        alt_idx = min(int(alt_deg / 90.0 * self.alt_bins), self.alt_bins - 1)
        az_idx = min(int(az_deg / 360.0 * self.az_bins), self.az_bins - 1)
        alt_idx = max(0, alt_idx)
        az_idx = max(0, az_idx)

        # Update grid with EMA
        count = self.grid_count[alt_idx, az_idx]
        if count == 0:
            # First sample in this cell: initialise from parametric model
            self.grid_alt[alt_idx, az_idx] = residual_alt
            self.grid_az[alt_idx, az_idx] = residual_az
        else:
            # EMA blend: use adaptive alpha (higher early, converges to learning_rate)
            alpha = min(self.learning_rate, 1.0 / (count + 1))
            self.grid_alt[alt_idx, az_idx] = (
                (1 - alpha) * self.grid_alt[alt_idx, az_idx] +
                alpha * residual_alt
            )
            self.grid_az[alt_idx, az_idx] = (
                (1 - alpha) * self.grid_az[alt_idx, az_idx] +
                alpha * residual_az
            )

        self.grid_count[alt_idx, az_idx] += 1
        self.total_samples += 1

        # Update parametric model via SGD every 10 samples
        if self.total_samples % 10 == 0:
            self._update_parametric(alt_deg, az_deg, residual_alt,
                                    residual_az, temperature)

        # Update statistics periodically
        if self.total_samples % 50 == 0:
            self._update_stats()

    def get_correction(self, alt_deg: float, az_deg: float,
                       temperature: Optional[float] = None) -> Tuple[float, float]:
        """
        Get the flexure correction for a given sky position.

        Blends the grid-based and parametric corrections weighted by the
        number of samples in the relevant grid cell.  In well-observed
        regions, the grid dominates; in un-observed regions, the
        parametric model provides the prediction.

        Args:
            alt_deg:     Current altitude in degrees (0-90).
            az_deg:      Current azimuth in degrees (0-360).
            temperature: Ambient temperature in Celsius (optional).

        Returns:
            (correction_alt, correction_az) in arcsec/sec.
            The corrections should be ADDED to the tracking rates.
        """
        if not self.is_enabled:
            return 0.0, 0.0

        # Parametric prediction (always available, even for un-observed cells)
        param_alt, param_az = self._parametric_predict(alt_deg, az_deg,
                                                        temperature)

        # Grid lookup with bilinear interpolation
        grid_alt, grid_az, confidence = self._grid_lookup(alt_deg, az_deg)

        # Blend: confidence 0.0 = purely parametric, 1.0 = purely grid
        corr_alt = confidence * grid_alt + (1.0 - confidence) * param_alt
        corr_az = confidence * grid_az + (1.0 - confidence) * param_az

        # Return NEGATIVE of the learned residual (correction opposes error)
        return -corr_alt, -corr_az

    def _grid_lookup(self, alt_deg: float, az_deg: float
                     ) -> Tuple[float, float, float]:
        """
        Bilinear interpolation on the grid map.

        Returns:
            (correction_alt, correction_az, confidence) where confidence
            is in [0, 1] based on sample count in surrounding cells.
        """
        # Continuous grid coordinates
        alt_f = alt_deg / 90.0 * self.alt_bins - 0.5
        az_f = az_deg / 360.0 * self.az_bins - 0.5

        # Clamp to valid range
        alt_f = max(0, min(alt_f, self.alt_bins - 1.001))
        az_f = max(0, min(az_f, self.az_bins - 1.001))

        # Integer and fractional parts
        a0 = int(alt_f)
        z0 = int(az_f)
        a1 = min(a0 + 1, self.alt_bins - 1)
        z1 = (z0 + 1) % self.az_bins  # wrap azimuth
        fa = alt_f - a0
        fz = az_f - z0

        # Bilinear interpolation weights
        w00 = (1 - fa) * (1 - fz)
        w01 = (1 - fa) * fz
        w10 = fa * (1 - fz)
        w11 = fa * fz

        # Interpolated corrections
        corr_alt = (w00 * self.grid_alt[a0, z0] + w01 * self.grid_alt[a0, z1] +
                    w10 * self.grid_alt[a1, z0] + w11 * self.grid_alt[a1, z1])
        corr_az = (w00 * self.grid_az[a0, z0] + w01 * self.grid_az[a0, z1] +
                   w10 * self.grid_az[a1, z0] + w11 * self.grid_az[a1, z1])

        # Confidence: based on sample count in surrounding cells
        # Saturates at 1.0 after ~20 samples per cell
        min_count = min(self.grid_count[a0, z0], self.grid_count[a0, z1],
                        self.grid_count[a1, z0], self.grid_count[a1, z1])
        confidence = min(1.0, min_count / 20.0)

        return corr_alt, corr_az, confidence

    def _parametric_predict(self, alt_deg: float, az_deg: float,
                            temperature: Optional[float] = None
                            ) -> Tuple[float, float]:
        """
        Evaluate the parametric flexure model.

        The model captures the physics of gravitational tube bending:
        - Flexure proportional to cos(alt) (gravity perpendicular to tube)
        - Azimuthal asymmetry from off-axis mass (focuser, camera)
        - Thermal coefficient coupling (flexure changes with temperature)
        """
        alt_rad = math.radians(max(1.0, alt_deg))  # clamp to avoid cos(0)=1 edge
        az_rad = math.radians(az_deg)

        cos_alt = math.cos(alt_rad)
        sin_az = math.sin(az_rad)
        cos_az = math.cos(az_rad)

        delta_temp = 0.0
        if temperature is not None and self._base_temperature is not None:
            delta_temp = temperature - self._base_temperature

        # Alt flexure: A*cos(alt) + B*cos(alt)*cos(az) + C*cos(alt)*sin(az)
        #            + D*cos(alt)*dT + E*dT
        p = self.param_alt
        flex_alt = (p[0] * cos_alt +
                    p[1] * cos_alt * cos_az +
                    p[2] * cos_alt * sin_az +
                    p[3] * cos_alt * delta_temp +
                    p[4] * delta_temp)

        # Az flexure: F*sin(az)*cos(alt) + G*cos(az)*cos(alt) + H*dT
        q = self.param_az
        flex_az = (q[0] * sin_az * cos_alt +
                   q[1] * cos_az * cos_alt +
                   q[2] * delta_temp)

        return float(flex_alt), float(flex_az)

    def _update_parametric(self, alt_deg: float, az_deg: float,
                           residual_alt: float, residual_az: float,
                           temperature: Optional[float] = None):
        """
        SGD update step for the parametric model.

        Computes the gradient of the squared error loss and updates
        coefficients with L2 regularization.
        """
        alt_rad = math.radians(max(1.0, alt_deg))
        az_rad = math.radians(az_deg)

        cos_alt = math.cos(alt_rad)
        sin_az = math.sin(az_rad)
        cos_az = math.cos(az_rad)

        delta_temp = 0.0
        if temperature is not None and self._base_temperature is not None:
            delta_temp = temperature - self._base_temperature

        # Alt features
        features_alt = np.array([
            cos_alt,
            cos_alt * cos_az,
            cos_alt * sin_az,
            cos_alt * delta_temp,
            delta_temp,
        ])

        # Az features
        features_az = np.array([
            sin_az * cos_alt,
            cos_az * cos_alt,
            delta_temp,
        ])

        # Current predictions
        pred_alt = np.dot(self.param_alt, features_alt)
        pred_az = np.dot(self.param_az, features_az)

        # Errors
        err_alt = residual_alt - pred_alt
        err_az = residual_az - pred_az

        # SGD update with L2 regularization
        lr = 0.005
        reg = 0.001
        self.param_alt += lr * (err_alt * features_alt - reg * self.param_alt)
        self.param_az += lr * (err_az * features_az - reg * self.param_az)

    def _update_stats(self):
        """Recompute diagnostic statistics."""
        filled = int(np.sum(self.grid_count > 0))
        total = self.alt_bins * self.az_bins
        self.stats['total_samples'] = self.total_samples
        self.stats['grid_coverage_pct'] = 100.0 * filled / total if total > 0 else 0.0
        self.stats['param_rms_alt'] = float(np.sqrt(np.mean(self.param_alt ** 2)))
        self.stats['param_rms_az'] = float(np.sqrt(np.mean(self.param_az ** 2)))

    def get_statistics(self) -> dict:
        """Return flexure model statistics for UI display."""
        self._update_stats()
        return {
            **self.stats,
            'is_enabled': self.is_enabled,
            'is_learning': self.is_learning,
            'param_alt_coeffs': self.param_alt.tolist(),
            'param_az_coeffs': self.param_az.tolist(),
            'grid_shape': [self.alt_bins, self.az_bins],
        }

    def save(self, filepath: str = "flexure_model.json"):
        """
        Persist the flexure model to a JSON file.

        Saves both the grid map and parametric coefficients, allowing
        the model to accumulate knowledge across observing sessions.
        """
        data = {
            'version': 1,
            'timestamp': time.time(),
            'alt_bins': self.alt_bins,
            'az_bins': self.az_bins,
            'total_samples': self.total_samples,
            'grid_alt': self.grid_alt.tolist(),
            'grid_az': self.grid_az.tolist(),
            'grid_count': self.grid_count.tolist(),
            'param_alt': self.param_alt.tolist(),
            'param_az': self.param_az.tolist(),
            'base_temperature': self._base_temperature,
        }
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass

    def load(self, filepath: str = "flexure_model.json") -> bool:
        """
        Load a previously saved flexure model.

        Returns:
            True if loaded successfully, False otherwise.
        """
        import os
        if not os.path.exists(filepath):
            return False
        try:
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)

            if data.get('version', 0) < 1:
                return False

            self.alt_bins = data['alt_bins']
            self.az_bins = data['az_bins']
            self.total_samples = data.get('total_samples', 0)
            self.grid_alt = np.array(data['grid_alt'])
            self.grid_az = np.array(data['grid_az'])
            self.grid_count = np.array(data['grid_count'], dtype=int)
            self.param_alt = np.array(data['param_alt'])
            self.param_az = np.array(data['param_az'])
            self._base_temperature = data.get('base_temperature')
            return True

        except Exception:
            return False

    def reset(self):
        """Reset all learned data and return to untrained state."""
        self.grid_alt = np.zeros((self.alt_bins, self.az_bins))
        self.grid_az = np.zeros((self.alt_bins, self.az_bins))
        self.grid_count = np.zeros((self.alt_bins, self.az_bins), dtype=int)
        self.param_alt = np.zeros(5)
        self.param_az = np.zeros(3)
        self.total_samples = 0
        self._base_temperature = None


def atmospheric_refraction_correction(alt_degrees: float, 
                                     temperature: Optional[float] = None,
                                     pressure: Optional[float] = None,
                                     weather_data: Optional[object] = None) -> float:
    """Compute the atmospheric refraction correction for a given altitude.

    Near the horizon the atmosphere bends light upward, making objects
    appear higher than their true geometric position.  This function
    returns a correction (in degrees) that should be *added* to the
    commanded altitude to compensate.

    Two formula variants are used:

    - **alt > 10 deg**: the simple Saemundsson (1986) approximation
      ``R = 1.02 / tan(alt)`` arcminutes, which is accurate to a few
      arcsec above 10 degrees.
    - **alt <= 10 deg**: an extended formula that adds a cubic tangent
      term ``-0.001927 / tan(alt)^3`` to improve accuracy closer to the
      horizon where refraction changes rapidly.

    Both variants are scaled by temperature and pressure factors relative
    to the standard atmosphere (10 deg C, 1013.25 hPa).

    Args:
        alt_degrees: Apparent (observed) altitude of the object in degrees.
        temperature: Ambient air temperature in degrees Celsius.  Defaults
            to 20 deg C if neither *temperature* nor *weather_data* is given.
        pressure: Atmospheric pressure in hPa (mbar).  Defaults to
            1013.25 hPa (standard sea-level pressure) if not provided.
        weather_data: An optional object with ``.temperature`` and
            ``.pressure`` attributes (e.g. a ``WeatherData`` instance).
            When provided, its values take priority over the explicit
            *temperature* and *pressure* parameters.

    Returns:
        float: Refraction correction to add to the altitude, in degrees.
        The value is negative (altitude must be lowered) because the
        observed position is already refracted upward.
    """
    # If a weather-data object is available, prefer its readings
    if weather_data:
        temp = weather_data.temperature
        press = weather_data.pressure
    else:
        # Fall back to explicit parameters or sensible defaults
        temp = temperature if temperature is not None else 20.0
        press = pressure if pressure is not None else 1013.25
    
    # Convert apparent altitude to radians for trigonometric functions
    alt_rad = math.radians(alt_degrees)
    
    # Temperature correction factor.
    # Standard reference temperature for the Saemundsson formula is 10 deg C
    # (283.15 K).  The factor scales refraction proportionally to the ratio
    # of air densities at the reference and actual temperatures.
    temp_factor = 283.15 / (273.15 + temp)
    # Pressure correction factor.
    # Refraction is proportional to air density, which scales linearly
    # with pressure at a given temperature.
    pressure_factor = press / 1013.25
    
    # Compute refraction in arcminutes
    if alt_degrees > 10:
        # Simple Saemundsson approximation -- accurate above 10 deg
        refraction_arcmin = 1.02 / math.tan(alt_rad) * temp_factor * pressure_factor
    else:
        # Extended formula for low altitudes where the simple 1/tan
        # approximation diverges.  The cubic term improves accuracy
        # down to about 1 degree altitude.
        refraction_arcmin = (
            1.02 / math.tan(alt_rad) - 
            0.001927 / (math.tan(alt_rad)**3)
        ) * temp_factor * pressure_factor
    
    # Convert from arcminutes to degrees and negate: the correction is
    # subtracted from the apparent altitude because objects appear higher
    # than they really are due to refraction.
    return -refraction_arcmin / 60.0


def temperature_drift_correction(temperature: float, 
                                 base_temperature: float = 20.0,
                                 thermal_coefficient: float = 0.1) -> Tuple[float, float]:
    """Estimate mechanical drift caused by thermal expansion of the telescope.

    Temperature changes cause the telescope tube and mount structure to
    expand or contract, shifting the optical axis.  This function returns
    a simple linear estimate of the resulting pointing error in both axes.

    The altitude axis is assumed to be more affected than azimuth because
    the tube length (and therefore the lever arm for flexure) changes
    directly with temperature.  The azimuth correction is set to half the
    altitude correction as a first-order approximation.

    Args:
        temperature: Current ambient temperature in degrees Celsius.
        base_temperature: Reference temperature at which the telescope was
            calibrated / aligned (degrees Celsius).  Defaults to 20 deg C.
        thermal_coefficient: Sensitivity of pointing error to temperature
            change, in arcsec per degree Celsius.  The default of 0.1
            arcsec/deg C is a conservative estimate for a typical Dobson
            tube; the actual value depends on tube material and length.

    Returns:
        tuple[float, float]: A pair ``(correction_alt, correction_az)``
        in arcsec.  Positive values indicate the commanded position should
        be increased by that amount to compensate for the drift.
    """
    # Temperature difference from the calibration baseline
    delta_temp = temperature - base_temperature
    # Linear drift model: correction = delta_T * coefficient
    correction = delta_temp * thermal_coefficient
    
    # Altitude is the primary affected axis (tube expansion changes the
    # balance point and flexure).  Azimuth is less affected; the 0.5
    # factor is an empirical estimate for typical Dobson mounts.
    return correction, correction * 0.5
