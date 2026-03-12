"""
Real-Time Tracking System with Fast Plate Solving for Alt-Az Dobson Mounts

This is the central module of the telescope correction pipeline. It orchestrates
multiple correction sources and sends custom tracking rates to the telescope mount
via OnStep's SXTR/SXTD commands.

Architecture Overview
=====================
The RealTimeTrackingController runs a 5Hz control loop (_control_loop) in a
background thread. On each iteration, the latest plate-solve position is used to
compute corrections from four independent sources, which are then fused into a
single pair of Alt/Az tracking rates.

Correction Sources and Weights
------------------------------
1. **Kalman Filter (45%)**
   - Smooths noisy plate-solve positions and estimates drift velocity.
   - Provided by `AdaptiveKalmanFilter` (see kalman_filter.py).

2. **ML Drift Predictor (55%)**
   - A position-dependent drift model trained online via SGD regression.
   - Learns systematic drift patterns as a function of (Alt, Az).
   - Provided by `DriftPredictor` (see drift_ml.py).

3. **Atmospheric Refraction (additive, Alt-axis only)**
   - Saemundsson formula corrects for atmosphere bending light upward.
   - Most significant below ~10 deg altitude.
   - Provided by `tracking_improvements.atmospheric_refraction_correction()`.

4. **Temperature Drift (additive, both axes)**
   - Linear thermal expansion model compensates for tube/mount flexure
     as temperature changes relative to the alignment baseline.
   - Provided by `tracking_improvements.temperature_drift_correction()`.

5. **Software PEC (additive, not weighted)**
   - FFT-based periodic error correction.
   - Learns repeating mechanical errors and predicts a Fourier correction.
   - Added on top of the weighted sum; not part of the weight budget.
   - Provided by `SoftwarePEC` (see software_pec.py).

Data Flow (per iteration at 5Hz)
---------------------------------
1. Plate-solve delivers RA/Dec -> converted to Alt/Az via _radec_to_altaz().
2. Kalman filter is updated with the new Alt/Az position.
3. ML predictor is fed the observed drift for online learning.
4. _calculate_and_apply_correction() runs:
   a. Query Kalman velocity estimate.
   b. Query ML drift prediction at current (Alt, Az).
   c. Fuse sources with adaptive weights.
   d. Compute base sidereal tracking rates in Alt/Az.
   e. Subtract the fused drift correction (negative feedback).
   f. Add PEC correction (if enabled).
   g. Record everything in correction_history for plotting.
5. The resulting Alt/Az rates are converted to RA/Dec offsets
   (_altaz_rate_to_radec_rate) and sent to the mount as SXTR/SXTD commands.

Why SXTR/SXTD Use RA/Dec
-------------------------
Even though the application works entirely in Alt/Az internally, the OnStep
mount firmware expects custom tracking rate offsets in equatorial coordinates
(RA offset from sidereal, Dec rate). The firmware then handles the conversion
to motor steps for whichever mount type is configured. Therefore,
_send_tracking_rate() converts the final Alt/Az rates to RA/Dec via the local
Jacobian transformation in _altaz_rate_to_radec_rate() before sending.

Key Dataclasses
---------------
- TrackingRate: holds the current Alt/Az tracking rates (arcsec/sec).
- CorrectionRecord: one snapshot of all correction contributions for plotting.
- PositionSample: a single plate-solve measurement with timestamp.

Timing
------
- Plate-solve target interval: 4.0s (depends on camera/ASTAP speed).
- Correction loop interval: 0.2s (5Hz).
- Prediction horizon: 1.0s ahead.
"""

import threading
import time
import math
import numpy as np
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
from collections import deque

from kalman_filter import AdaptiveKalmanFilter
from drift_ml import DriftPredictor
from software_pec import SoftwarePEC

# Atmospheric refraction, thermal drift, and flexure corrections
# (imported from the tracking improvements module)
try:
    from desktop_only.tracking_improvements import (
        atmospheric_refraction_correction,
        temperature_drift_correction,
        FlexureModel,
    )
    _HAS_TRACKING_IMPROVEMENTS = True
except ImportError:
    try:
        # Flat layout (Android / Chaquopy or same-directory)
        from tracking_improvements import (
            atmospheric_refraction_correction,
            temperature_drift_correction,
            FlexureModel,
        )
        _HAS_TRACKING_IMPROVEMENTS = True
    except ImportError:
        _HAS_TRACKING_IMPROVEMENTS = False


@dataclass
class TrackingRate:
    """Current tracking rate in Alt/Az coordinates.

    Both fields are expressed in arcseconds per second. These represent the
    instantaneous rates being sent to the mount (after all corrections are
    applied and before the final RA/Dec conversion for SXTR/SXTD).

    Attributes:
        alt_rate: Altitude tracking rate (arcsec/sec).
        az_rate:  Azimuth tracking rate (arcsec/sec).
    """
    alt_rate: float = 0.0
    az_rate: float = 0.0


@dataclass
class CorrectionRecord:
    """Snapshot of one correction cycle, used for the real-time graph overlay.

    All rate values are in arcsec/sec in the Alt/Az frame.  Each field records
    the contribution of a single correction source so the GUI can display
    stacked area charts showing how much each source contributes.

    Attributes:
        timestamp:       Unix epoch of this record.
        kalman_alt:      Kalman filter velocity estimate in Alt (arcsec/sec).
        kalman_az:       Kalman filter velocity estimate in Az (arcsec/sec).
        ml_alt:          ML drift prediction in Alt (arcsec/sec).
        ml_az:           ML drift prediction in Az (arcsec/sec).
        pec_alt:         Software PEC correction in Alt (arcsec/sec).
        pec_az:          Software PEC correction in Az (arcsec/sec).
        refraction_alt:  Atmospheric refraction correction in Alt (arcsec).
        thermal_alt:     Thermal drift correction in Alt (arcsec).
        thermal_az:      Thermal drift correction in Az (arcsec).
        total_alt:       Final combined tracking rate in Alt (arcsec/sec).
        total_az:        Final combined tracking rate in Az (arcsec/sec).
        error_alt:       Measured drift error in Alt (arcsec/sec).
        error_az:        Measured drift error in Az (arcsec/sec).
    """
    timestamp: float
    # Individual source contributions (arcsec/sec)
    kalman_alt: float = 0.0
    kalman_az: float = 0.0
    ml_alt: float = 0.0
    ml_az: float = 0.0
    pec_alt: float = 0.0
    pec_az: float = 0.0
    # Environmental corrections (arcsec -- applied as position offsets)
    refraction_alt: float = 0.0
    thermal_alt: float = 0.0
    thermal_az: float = 0.0
    # Flexure model corrections (arcsec/sec)
    flexure_alt: float = 0.0
    flexure_az: float = 0.0
    # Final combined correction (Alt/Az)
    total_alt: float = 0.0
    total_az: float = 0.0
    # Measured error
    error_alt: float = 0.0
    error_az: float = 0.0


@dataclass
class PositionSample:
    """A single plate-solve measurement with both coordinate representations.

    Stores the raw RA/Dec from the plate solver together with the derived
    Alt/Az and metadata about solve performance.

    Attributes:
        timestamp:    Unix epoch when the solve was received.
        ra_hours:     Right ascension in decimal hours [0, 24).
        dec_degrees:  Declination in decimal degrees [-90, +90].
        alt_degrees:  Altitude in decimal degrees [0, 90].
        az_degrees:   Azimuth in decimal degrees [0, 360).
        solve_time_ms: Wall-clock time the plate solver took (milliseconds).
        is_valid:     Whether the solve result is considered trustworthy.
    """
    timestamp: float
    ra_hours: float
    dec_degrees: float
    alt_degrees: float
    az_degrees: float
    solve_time_ms: float
    is_valid: bool = True


class RealTimeTrackingController:
    """Main real-time tracking controller that orchestrates the correction pipeline.

    This class is the heart of the telescope tracking system.  It fuses three
    independent correction sources (Kalman filter, ML drift predictor, and
    software PEC) into a single pair of tracking rates that are sent to the
    OnStep mount firmware at 5Hz.

    Lifecycle:
        1. Instantiate and configure (set latitude/longitude, callbacks, etc.).
        2. Call ``start()`` to begin the background control loop.
        3. Feed plate-solve results via ``update_from_plate_solve()`` as they
           arrive (typically every ~0.5s).
        4. The 5Hz control loop continuously computes corrections and sends
           SXTR/SXTD commands to the mount.
        5. Call ``stop()`` to halt tracking and zero out rates.

    Threading model:
        - ``_control_loop()`` runs in a dedicated daemon thread at ~5Hz.
        - ``update_from_plate_solve()`` is called from the plate-solve thread.
        - A threading.Lock (_lock) protects shared state accessed by both.

    Attributes:
        kalman:           AdaptiveKalmanFilter instance for position smoothing.
        ml_predictor:     DriftPredictor instance for learned drift compensation.
        pec:              SoftwarePEC for FFT-based periodic error correction.
        is_running:       Whether the controller is active.
        is_tracking:      Whether corrections are being applied.
        tracking_rate:    Current TrackingRate being sent to the mount.
        correction_history: Deque of CorrectionRecord for GUI graphing.
    """
    
    # Standard sidereal rate: 15.041 arcsec per SI second
    SIDEREAL_RATE = 15.041067  # arcsec/SI-sec

    # Ratio of SI second to sidereal second.  One sidereal day =
    # 23h 56m 04.0905s (86164.0905 SI seconds), so one sidereal second
    # = 86164.0905 / 86400 = 0.9972696 SI seconds.
    # OnStepX SXTR/SXTD units are arcsec per *sidereal* second, but the
    # internal pipeline computes rates in arcsec per SI second.  Multiply
    # by this factor to convert: rate_sid = rate_SI * SI_TO_SIDEREAL.
    SI_TO_SIDEREAL = 86400.0 / 86164.0905  # ~1.002738

    # Base RA tracking rates for each mode (arcsec per SI second).
    # SXTR is an offset from *the mount's current base rate*, so we must
    # subtract the correct base rate before sending.
    TRACKING_RATES = {
        'sidereal': 15.041067,   # Earth rotation rate
        'lunar':    14.685,      # Moon's apparent rate (~0.549"/s slower)
        'solar':    15.0,        # Sun's apparent rate (~0.041"/s slower)
        'king':     15.037,      # King rate (refraction-corrected sidereal)
    }
    
    def __init__(self):
        """Initialize all correction components, state variables, and configuration."""
        # --- Correction pipeline components ---
        self.kalman = AdaptiveKalmanFilter()
        self.ml_predictor = DriftPredictor()
        
        # --- Tracking state ---
        self.is_running = False
        self.is_tracking = False

        # --- Mount base tracking rate ---
        # Tracks which rate mode the mount firmware is currently using so
        # the SXTR offset is computed relative to the correct base rate.
        self._base_rate_mode = 'sidereal'  # 'sidereal', 'lunar', 'solar', 'king'
        
        # --- Current telescope position ---
        self.current_ra = 0.0       # hours [0, 24)
        self.current_dec = 0.0      # degrees [-90, +90]
        self.current_alt = 45.0     # degrees [0, 90]
        self.current_az = 180.0     # degrees [0, 360)
        
        # Position history for drift analysis (ring buffer, reduced for memory)
        self.position_history: deque = deque(maxlen=500)
        
        # Current tracking rates being sent to the mount
        self.tracking_rate = TrackingRate()
        
        # --- Timing configuration ---
        self.plate_solve_interval = 4.0   # seconds (target plate-solve cadence)
        self.correction_interval = 0.2    # seconds (5Hz correction loop)
        self.prediction_horizon = 1.0     # seconds ahead for Kalman prediction (increased for longer solve interval)
        
        # --- PID-like correction gains ---
        self.gain_p = 0.8      # Proportional gain
        self.gain_i = 0.1      # Integral gain
        self.gain_d = 0.05     # Derivative gain
        
        # Integral accumulators for PID error correction
        self.error_integral_alt = 0.0
        self.error_integral_az = 0.0
        self.last_error_alt = 0.0
        self.last_error_az = 0.0
        
        # --- Observatory location (defaults; overridden by config_manager) ---
        self.latitude = 48.8566   # see config_manager.DEFAULT_LATITUDE
        self.longitude = 2.3522   # see config_manager.DEFAULT_LONGITUDE
        
        # --- Trig cache for sidereal rate computation (Pi optimization) ---
        # Latitude trig values are precomputed when latitude is set (never
        # changes during a session). Alt/Az trig values are cached and only
        # recomputed when the position changes by more than the threshold.
        self._cached_lat_rad = math.radians(self.latitude)
        self._cached_cos_lat = math.cos(self._cached_lat_rad)
        self._cached_sin_lat = math.sin(self._cached_lat_rad)
        
        # Forward the initial latitude to the EKF sidereal model
        self.kalman.set_latitude(self.latitude)
        self._cached_sidereal_alt = None  # (alt, az) at last computation
        self._cached_sidereal_az = None
        self._cached_sidereal_result = (0.0, 0.0)
        self._sidereal_cache_threshold = 0.01  # degrees -- recompute only when position changes by this much
        
        # --- Callbacks (set by the GUI / application layer) ---
        self.on_rate_update: Optional[Callable[[float, float], None]] = None
        self.on_position_update: Optional[Callable[[float, float, float, float], None]] = None
        self.on_log: Optional[Callable[[str], None]] = None
        self.send_command: Optional[Callable[[str], str]] = None
        
        # --- Threading ---
        self._control_thread: Optional[threading.Thread] = None
        self._solve_thread: Optional[threading.Thread] = None
        
        # Lock protecting shared state between control loop and plate-solve thread
        self._lock = threading.Lock()
        
        # --- Runtime statistics ---
        self.stats = {
            'total_solves': 0,
            'successful_solves': 0,
            'avg_solve_time': 0.0,
            'avg_correction': 0.0,
            'total_corrections': 0
        }
        
        # Correction history for real-time graphs (~50 seconds at 5Hz)
        self.correction_history: deque = deque(maxlen=250)

        # --- Rate smoothing / debounce ---
        # Exponential moving average on the output rates prevents sudden jumps
        # caused by plate-solve noise from jolting the mount.  A small alpha
        # (close to 0) gives heavy smoothing; 1.0 disables smoothing entirely.
        self._rate_smooth_alpha = 0.15   # EMA blending factor (was 0.35)
        self._smooth_alt_rate = 0.0      # Smoothed alt rate (arcsec/s)
        self._smooth_az_rate = 0.0       # Smoothed az rate (arcsec/s)

        # --- Max rate clamp ---
        # Absolute ceiling on tracking correction rates sent to the mount.
        # Prevents runaway rates from ever reaching the motors.
        self._max_rate = 50.0            # arcsec/sec hard ceiling

        # --- Acceleration limiter ---
        # Limits how much the sent rate can change between consecutive cycles.
        # This prevents sudden jumps that cause star trailing on long exposures.
        # At 5 Hz (0.2s cycle), 2.5 "/s per cycle = 12.5 "/s^2 max accel.
        self._max_accel = 2.5            # arcsec/sec max change per cycle
        self._prev_sent_alt_rate = 0.0   # Last rate actually sent (alt)
        self._prev_sent_az_rate = 0.0    # Last rate actually sent (az)

        # --- Start/stop ramping ---
        # Gradually ramp correction authority from 0 to 1 over ~1.5 seconds
        # after tracking starts, and ramp down before stopping.  This avoids
        # the initial jolt when corrections kick in at full strength.
        self._ramp_factor = 0.0          # 0.0 = fully ramped down, 1.0 = full rate
        self._ramp_increment = 0.13      # Per cycle increment (~1.5s to reach 1.0 at 5Hz)

        # Software PEC (periodic error correction via FFT + Fourier synthesis)
        # drive_type is set later by the app layer via configure_drive_type()
        self.pec = SoftwarePEC()
        self.pec_enabled = True  # Enable PEC by default

        # Flexure learning model -- builds a sky map of position-dependent
        # tube flexure and refraction residuals from tracking data.
        # Persisted across sessions for cumulative learning.
        self.flexure_model: Optional['FlexureModel'] = None
        self.flexure_enabled = _HAS_TRACKING_IMPROVEMENTS
        if _HAS_TRACKING_IMPROVEMENTS:
            try:
                self.flexure_model = FlexureModel()
            except Exception:
                self.flexure_model = None
                self.flexure_enabled = False

        # --- Environmental corrections ---
        # Atmospheric refraction and thermal drift, imported from
        # tracking_improvements.py.  Enabled by default when the module
        # is available; can be toggled at runtime.
        self.refraction_enabled = _HAS_TRACKING_IMPROVEMENTS
        self.thermal_drift_enabled = _HAS_TRACKING_IMPROVEMENTS

        # Current weather / environment state (updated by the application
        # layer via ``update_weather()`` whenever new sensor data arrives).
        self._temperature_c: Optional[float] = None      # Ambient temp (Celsius)
        self._pressure_hpa: Optional[float] = None        # Barometric pressure (hPa)
        self._humidity_pct: Optional[float] = None        # Relative humidity (%)
        self._base_temperature_c: Optional[float] = None  # Calibration temp (set once)

        # Thermal coefficient: arcsec of pointing shift per degree Celsius
        # of temperature change.  0.1 is conservative; typical Dobson tubes
        # can be 0.05-0.3 depending on material and length.
        self.thermal_coefficient = 0.1

        # Previous refraction correction (degrees) for computing the rate of
        # change between cycles.
        self._prev_refraction_deg = 0.0

        # Protocol-aware tracking rate sending (set by HEADLESS_SERVER).
        # If set, this callback sends Alt/Az variable rates directly via
        # NexStar passthrough; if None, falls back to LX200 SXTR/SXTD.
        self.send_variable_rate_altaz: Optional[Callable[[float, float], None]] = None
    
    def set_latitude(self, latitude: float):
        """Update the observer latitude and refresh cached trig values.
        
        Call this instead of setting self.latitude directly so the
        sidereal rate trig cache stays in sync.
        
        Args:
            latitude: Observer latitude in decimal degrees.
        """
        self.latitude = latitude
        self._cached_lat_rad = math.radians(latitude)
        self._cached_cos_lat = math.cos(self._cached_lat_rad)
        self._cached_sin_lat = math.sin(self._cached_lat_rad)
        # Invalidate the sidereal rate cache
        self._cached_sidereal_alt = None
        self._cached_sidereal_az = None
        # Forward to the EKF sidereal model
        self.kalman.set_latitude(latitude)

    def update_weather(self, temperature_c: Optional[float] = None,
                        pressure_hpa: Optional[float] = None,
                        humidity_pct: Optional[float] = None):
        """Update weather / environment data for atmospheric corrections.

        Called by the application layer whenever new sensor readings are
        available (e.g. from the mount's weather sensor, an external BME280,
        or manual user input).

        The *first* temperature reading is automatically saved as the
        ``_base_temperature_c`` (calibration baseline for thermal drift).
        Subsequent readings are used to compute the delta.

        Args:
            temperature_c: Ambient air temperature in degrees Celsius.
            pressure_hpa: Barometric pressure in hPa (mbar).
            humidity_pct: Relative humidity in percent (0-100).
        """
        if temperature_c is not None:
            self._temperature_c = temperature_c
            # First reading sets the calibration baseline
            if self._base_temperature_c is None:
                self._base_temperature_c = temperature_c
                self._log(f"Thermal drift baseline set: {temperature_c:.1f} C")
            # Forward temperature to ML predictor for flexure features
            self.ml_predictor.update_temperature(
                temperature_c, self._base_temperature_c
            )
            # Forward temperature to flexure model
            if self.flexure_model is not None:
                if self.flexure_model._base_temperature is None:
                    self.flexure_model._base_temperature = temperature_c
        if pressure_hpa is not None:
            self._pressure_hpa = pressure_hpa
        if humidity_pct is not None:
            self._humidity_pct = humidity_pct

    def set_base_tracking_rate(self, mode: str):
        """Update the mount's base tracking rate mode.

        Must be called whenever the user changes the mount tracking rate
        (sidereal / lunar / solar / king) so that SXTR offsets are computed
        relative to the correct base rate.

        Args:
            mode: One of 'sidereal', 'lunar', 'solar', 'king'.
        """
        mode = mode.lower()
        if mode in self.TRACKING_RATES:
            self._base_rate_mode = mode
            self._log(f"Base tracking rate updated: {mode} "
                      f"({self.TRACKING_RATES[mode]:.3f}\"/s)")

    def configure_drive_type(self, drive_type: str):
        """Set the mount drive type and retune the PEC engine.

        This should be called during application initialisation after the
        config has been loaded (before start()).  It adjusts the PEC
        analysis parameters to match the periodic error characteristics
        of the specified drive system.

        Args:
            drive_type: One of 'worm_gear', 'planetary_gearbox',
                        'harmonic_drive', 'belt_drive', 'direct_drive'.
        """
        self.pec.set_drive_type(drive_type)
        self._log(f"Drive type configured: {drive_type}")

    def start(self):
        """Start real-time tracking.

        Initializes the Kalman filter at the current position, loads any
        previously trained PEC model, and spawns the background control thread
        that runs at 5Hz.

        If tracking is already running this method is a no-op.
        """
        if self.is_running:
            return
        
        self.is_running = True
        self.is_tracking = True

        # Reset smoothing state for a clean start
        self._ramp_factor = 0.0              # Will ramp up over ~1.5s
        self._smooth_alt_rate = 0.0
        self._smooth_az_rate = 0.0
        self._prev_sent_alt_rate = 0.0
        self._prev_sent_az_rate = 0.0
        
        # Initialize Kalman filter with current position as the prior
        self.kalman.initialize(self.current_alt, self.current_az)
        
        # Initialize PEC: attach logger and load previously learned model if available
        self.pec.on_log = self._log_wrapper
        self.pec.load()  # Load previously learned PEC if available

        # Load flexure model from previous session (if available)
        if self.flexure_model is not None:
            if self.flexure_model.load():
                self._log("Flexure model loaded from previous session")
        
        # Spawn the background control thread (daemon so it dies with the process)
        self._control_thread = threading.Thread(
            target=self._control_loop,
            daemon=True,
            name="TrackingControl"
        )
        self._control_thread.start()
        
        self._log("▶️ Real-time tracking started")
        self._log(f"   Plate solve interval: {self.plate_solve_interval}s")
        self._log(f"   Correction interval: {self.correction_interval}s")
    
    def stop(self):
        """Stop real-time tracking with a gradual ramp-down.

        Persists the PEC model if it was trained during this session, then
        gradually ramps the tracking rates to zero over ~1 second so the
        mount decelerates smoothly instead of jerking to a halt.
        Blocks up to 2 seconds for the control thread to exit.
        """
        self.is_running = False
        self.is_tracking = False
        
        # Save PEC model for next session
        if self.pec_enabled and self.pec.is_trained:
            self.pec.save()

        # Save flexure model for next session
        if self.flexure_model is not None and self.flexure_model.total_samples > 0:
            self.flexure_model.save()
        
        # Gradual ramp-down: reduce rates over ~1 second (5 steps at 0.2s)
        # instead of an abrupt zero which causes a mechanical jolt.
        ramp_steps = 5
        for step in range(ramp_steps, 0, -1):
            factor = step / ramp_steps  # 1.0, 0.8, 0.6, 0.4, 0.2
            try:
                self._send_tracking_rate(
                    self._prev_sent_alt_rate * factor,
                    self._prev_sent_az_rate * factor,
                )
            except Exception:
                break
            time.sleep(0.2)

        # Final zero to guarantee full stop
        self._send_tracking_rate(0.0, 0.0)
        self._prev_sent_alt_rate = 0.0
        self._prev_sent_az_rate = 0.0
        
        if self._control_thread:
            self._control_thread.join(timeout=2.0)
        
        self._log("⏹️ Tracking stopped (ramped down)")
    
    def update_from_plate_solve(self, ra_hours: float, dec_degrees: float,
                                 solve_time_ms: float):
        """Ingest a new plate-solve result and update all filters.

        Called from the plate-solve thread whenever ASTAP returns a successful
        solution.  This method:
        1. Converts RA/Dec to Alt/Az.
        2. Computes observed drift from the previous sample and feeds it to
           the ML predictor for online learning.
        3. Updates the Kalman filter with the new measurement.
        4. Stores the filtered position as the current telescope position.

        Args:
            ra_hours:     Right ascension in decimal hours [0, 24).
            dec_degrees:  Declination in decimal degrees [-90, +90].
            solve_time_ms: How long the plate solver took (milliseconds).
        """
        with self._lock:
            timestamp = time.time()
            
            # Convert equatorial RA/Dec to horizontal Alt/Az for the local observer
            alt, az = self._radec_to_altaz(ra_hours, dec_degrees)
            
            # Build a position sample for history
            sample = PositionSample(
                timestamp=timestamp,
                ra_hours=ra_hours,
                dec_degrees=dec_degrees,
                alt_degrees=alt,
                az_degrees=az,
                solve_time_ms=solve_time_ms
            )
            
            # Compute observed drift rate from previous sample (deg/sec)
            if len(self.position_history) > 0:
                last = self.position_history[-1]
                dt = timestamp - last.timestamp
                if dt > 0:
                    drift_alt = (alt - last.alt_degrees) / dt
                    drift_az = (az - last.az_degrees) / dt
                    
                    # Feed the observed drift to the ML predictor for online training
                    self.ml_predictor.add_sample(alt, az, drift_alt, drift_az)
            
            # Append to position history ring buffer
            self.position_history.append(sample)
            
            # Update Kalman filter with the raw measurement; returns filtered position
            filtered_alt, filtered_az = self.kalman.update(alt, az)
            
            # Store the Kalman-filtered position as the current best estimate
            self.current_ra = ra_hours
            self.current_dec = dec_degrees
            self.current_alt = filtered_alt
            self.current_az = filtered_az
            
            # Update solve statistics
            self.stats['total_solves'] += 1
            self.stats['successful_solves'] += 1
            self._update_avg_solve_time(solve_time_ms)
            
            # Notify the GUI of the new position
            if self.on_position_update:
                self.on_position_update(ra_hours, dec_degrees, filtered_alt, filtered_az)
    
    def _control_loop(self):
        """Background control loop running at ~5Hz.

        This is the main correction thread.  It wakes up every ~20ms (via a
        short sleep to avoid CPU saturation) and checks whether enough time
        has elapsed since the last correction (``correction_interval``, default
        0.2s).  When it fires, it calls ``_calculate_and_apply_correction()``
        to fuse all correction sources and send rates to the mount.

        Runs as a daemon thread; exits when ``self.is_running`` is set to False.
        """
        last_correction_time = time.time()
        
        while self.is_running:
            current_time = time.time()
            dt = current_time - last_correction_time
            
            if dt >= self.correction_interval:
                # Compute and apply the fused correction from all sources
                self._calculate_and_apply_correction(dt)
                last_correction_time = current_time
            
            # Short sleep to avoid busy-waiting (CPU throttle).
            # 20ms sleep is sufficient for the 5Hz (200ms) correction interval
            # and reduces CPU wake-ups on mobile devices.
            time.sleep(0.02)
    
    def _calculate_and_apply_correction(self, dt: float):
        """Compute the fused correction from all sources and send it to the mount.

        **This is the heart of the entire tracking system.**

        The correction pipeline proceeds in 9 numbered steps:

        1. **Kalman velocity** -- Ask the Kalman filter for its current velocity
           estimate (deg/sec in Alt/Az).  This captures the smoothed, filtered
           drift as measured by recent plate solves.

        2. **ML drift prediction** -- Query the SGD regression model for the
           expected drift at the current (Alt, Az) sky position.  This captures
           *learned* systematic drift that depends on where the telescope is
           pointing (e.g., gear eccentricity varies with altitude).

        3. **Weight fusion** -- Combine Kalman and ML predictions
           using fixed weights (Kalman 45%, ML 55%).  The weighted sum
           yields a single drift estimate in (Alt, Az) deg/sec.

        4. **Base sidereal rates** -- Compute the theoretical Alt/Az tracking
           rates needed to follow the sky's sidereal motion at the current
           position and latitude.  These are NOT constant for an Alt/Az mount
           (unlike an equatorial mount).

        5. **Apply drift correction** -- Subtract the fused drift from the base
           sidereal rates.  The correction is the *negative* of the observed
           drift: if the telescope is drifting north, we add a southward rate
           component.  Drift values are converted from deg/sec to arcsec/sec.

        6. **Atmospheric refraction** -- Compute the Saemundsson refraction
           correction for the current altitude/temperature/pressure.  The
           *rate of change* of the refraction offset is applied to the Alt
           tracking rate (important below ~10 deg altitude).

        6b. **Temperature drift** -- Estimate mechanical pointing shift from
            thermal expansion of the telescope tube relative to the
            calibration baseline temperature.  Applied as a small rate
            correction on both axes (Alt more than Az).

        7. **Add Software PEC** -- If PEC is enabled, feed the residual drift
            (what Kalman+ML could *not* predict) into PEC for learning, then
            query the PEC model for its Fourier-based correction and add it on
            top.  PEC is additive (not part of the weight budget).

        8. **Record for plotting** -- Save all individual contributions and the
           final rates into a CorrectionRecord appended to correction_history.

        9. **Update statistics** -- Compute the total correction magnitude and
           update the running exponential average.

        After all steps, the final Alt/Az rates are sent to the mount via
        ``_send_tracking_rate()``, which converts them to RA/Dec and issues
        SXTR/SXTD commands.

        Args:
            dt: Time elapsed since the last correction cycle (seconds).
                Used for PID derivative and integral terms (reserved for
                future use; current fusion is purely velocity-based).
        """
        try:
            with self._lock:
                # ----------------------------------------------------------------
                # Step 1: Kalman filter velocity estimate (deg/sec in Alt/Az)
                # The Kalman filter tracks position and velocity; here we only
                # need the velocity (drift rate).
                # ----------------------------------------------------------------
                v_alt_kalman, v_az_kalman = self.kalman.get_velocity()
                
                # ----------------------------------------------------------------
                # Step 2: ML drift prediction (deg/sec in Alt/Az)
                # The SGD regression model predicts the expected drift as a
                # function of the current sky position (Alt, Az).
                # ----------------------------------------------------------------
                v_alt_ml, v_az_ml = self.ml_predictor.predict(self.current_alt, self.current_az)
                
                # ----------------------------------------------------------------
                # Step 3: Weight fusion
                # Kalman 45%, ML 55% (total 100%).
                # ----------------------------------------------------------------
                kalman_weight = 0.45
                ml_weight = 0.55
                
                # Fused drift estimate in Alt/Az (deg/sec)
                drift_alt = v_alt_kalman * kalman_weight + v_alt_ml * ml_weight
                drift_az = v_az_kalman * kalman_weight + v_az_ml * ml_weight
                
                # ----------------------------------------------------------------
                # Step 4: Compute base sidereal tracking rates in Alt/Az
                # For an Alt/Az mount, the sidereal tracking rates are NOT
                # constant -- they depend on the current (Alt, Az) and the
                # observer's latitude.  This function returns (arcsec/sec).
                # ----------------------------------------------------------------
                base_alt_rate, base_az_rate = self._calculate_sidereal_altaz_rate()
                
                # ----------------------------------------------------------------
                # Step 5: Apply drift correction (negative feedback)
                # The tracking rate must OPPOSE the observed drift:
                #   tracking_rate = base_sidereal_rate - measured_drift
                # drift_alt/az are in deg/sec, so multiply by 3600 -> arcsec/sec.
                # ----------------------------------------------------------------
                total_alt_rate = base_alt_rate - drift_alt * 3600
                total_az_rate = base_az_rate - drift_az * 3600
                
                # ----------------------------------------------------------------
                # Step 6: Atmospheric refraction correction (Alt only)
                #
                # The atmosphere refracts light, making objects appear higher
                # than their true geometric position.  The refraction amount
                # depends on altitude, temperature, and pressure.  We compute
                # the correction at the current position and convert its *rate
                # of change* to an additional tracking rate component.
                #
                # The Saemundsson formula (tracking_improvements.py) returns
                # a position offset in degrees.  Since the telescope is moving
                # (sidereal tracking), the refraction offset changes over time.
                # We compute the difference from the previous cycle and convert
                # it to arcsec/sec.
                # ----------------------------------------------------------------
                refraction_corr_arcsec = 0.0
                if self.refraction_enabled and _HAS_TRACKING_IMPROVEMENTS:
                    try:
                        refraction_deg = atmospheric_refraction_correction(
                            self.current_alt,
                            temperature=self._temperature_c,
                            pressure=self._pressure_hpa,
                        )
                        # Rate of change: delta_refraction / dt (arcsec/sec)
                        # Clamped to ±5 "/s like thermal to prevent sudden jumps
                        # when atmospheric conditions change rapidly.
                        delta_refraction_deg = refraction_deg - self._prev_refraction_deg
                        if dt > 0:
                            refraction_rate = (delta_refraction_deg * 3600) / dt
                            refraction_rate = max(-5.0, min(5.0, refraction_rate))
                            total_alt_rate += refraction_rate
                        refraction_corr_arcsec = refraction_deg * 3600
                        self._prev_refraction_deg = refraction_deg
                    except Exception:
                        pass

                # ----------------------------------------------------------------
                # Step 6b: Temperature drift correction (Alt + Az)
                #
                # Temperature changes cause the telescope tube and mount to
                # expand/contract, shifting the optical axis.  The correction
                # is a position offset (arcsec) proportional to the temperature
                # delta from the calibration baseline.  We apply it as a rate
                # correction over the control interval.
                # ----------------------------------------------------------------
                thermal_corr_alt_arcsec = 0.0
                thermal_corr_az_arcsec = 0.0
                if (self.thermal_drift_enabled and _HAS_TRACKING_IMPROVEMENTS
                        and self._temperature_c is not None
                        and self._base_temperature_c is not None):
                    try:
                        corr_alt, corr_az = temperature_drift_correction(
                            self._temperature_c,
                            base_temperature=self._base_temperature_c,
                            thermal_coefficient=self.thermal_coefficient,
                        )
                        thermal_corr_alt_arcsec = corr_alt
                        thermal_corr_az_arcsec = corr_az
                        # Apply as a steady rate over the control interval:
                        # if the tube has drifted by X arcsec, nudge at X/interval
                        # to compensate within one cycle.  Clamp to avoid large
                        # jumps when temperature readings arrive for the first time.
                        if dt > 0:
                            thermal_rate_alt = max(-5.0, min(5.0, corr_alt / dt))
                            thermal_rate_az = max(-5.0, min(5.0, corr_az / dt))
                            total_alt_rate += thermal_rate_alt
                            total_az_rate += thermal_rate_az
                    except Exception:
                        pass

                # ----------------------------------------------------------------
                # Step 7: Software PEC -- periodic error correction (additive)
                # PEC learns repeating mechanical errors via FFT and synthesizes
                # a Fourier correction.  It is NOT part of the weight budget;
                # it is simply added on top of the weighted sum.
                # ----------------------------------------------------------------
                pec_corr_alt, pec_corr_az = 0.0, 0.0
                if self.pec_enabled:
                    # Feed the residual drift (what Kalman+ML could NOT predict)
                    # into PEC so it can learn periodic patterns in the residual.
                    if self.pec.is_learning:
                        self.pec.add_error_sample(drift_alt * 3600, drift_az * 3600)
                    
                    # Query PEC for its correction (arcsec/sec to ADD)
                    pec_corr_alt, pec_corr_az = self.pec.get_correction()
                    total_alt_rate += pec_corr_alt
                    total_az_rate += pec_corr_az

                # ----------------------------------------------------------------
                # Step 7b: Flexure model -- position-dependent tube flexure
                # and refraction-residual learning (additive, like PEC).
                #
                # The flexure model accumulates knowledge of how the
                # telescope's mechanical structure bends at different sky
                # positions and temperatures.  It captures:
                # - Gravitational flexure (altitude-dependent tube sag)
                # - Azimuthal asymmetry from off-axis mass (focuser, camera)
                # - Thermal-flexure interaction (tube stiffness vs temperature)
                # - Systematic errors in the analytical refraction formula
                #
                # The model is fed the final residual *after* all other
                # corrections (Kalman, ML, PEC, refraction, thermal) so it
                # isolates the position-dependent component that the other
                # systems cannot capture.
                # ----------------------------------------------------------------
                flexure_corr_alt, flexure_corr_az = 0.0, 0.0
                if (self.flexure_enabled and self.flexure_model is not None
                        and _HAS_TRACKING_IMPROVEMENTS):
                    try:
                        # Feed residual to the flexure model for learning
                        if self.flexure_model.is_learning:
                            self.flexure_model.add_residual(
                                self.current_alt, self.current_az,
                                drift_alt * 3600, drift_az * 3600,
                                temperature=self._temperature_c,
                            )

                        # Query the flexure model for its correction
                        flexure_corr_alt, flexure_corr_az = (
                            self.flexure_model.get_correction(
                                self.current_alt, self.current_az,
                                temperature=self._temperature_c,
                            )
                        )
                        total_alt_rate += flexure_corr_alt
                        total_az_rate += flexure_corr_az
                    except Exception:
                        pass

                # ----------------------------------------------------------------
                # Step 8: Record all contributions for the real-time graph
                # Each CorrectionRecord stores both the individual source
                # contributions and the final fused rates so the GUI can render
                # stacked area charts.
                # ----------------------------------------------------------------
                record = CorrectionRecord(
                    timestamp=time.time(),
                    kalman_alt=v_alt_kalman * 3600,  # Convert deg/sec -> arcsec/sec
                    kalman_az=v_az_kalman * 3600,
                    ml_alt=v_alt_ml * 3600,
                    ml_az=v_az_ml * 3600,
                    pec_alt=pec_corr_alt,
                    pec_az=pec_corr_az,
                    refraction_alt=refraction_corr_arcsec,
                    thermal_alt=thermal_corr_alt_arcsec,
                    thermal_az=thermal_corr_az_arcsec,
                    flexure_alt=flexure_corr_alt,
                    flexure_az=flexure_corr_az,
                    total_alt=total_alt_rate,
                    total_az=total_az_rate,
                    error_alt=drift_alt * 3600,
                    error_az=drift_az * 3600
                )
                self.correction_history.append(record)
                
                # ----------------------------------------------------------------
                # Step 9: Update running statistics
                # Compute the total correction magnitude (Euclidean norm in
                # arcsec/sec) and feed it to the exponential moving average.
                # ----------------------------------------------------------------
                correction_magnitude = math.sqrt(drift_alt**2 + drift_az**2) * 3600
                self._update_avg_correction(correction_magnitude)
                self.stats['total_corrections'] += 1
                
                # Save final rates for use outside the lock
                alt_rate_final = total_alt_rate
                az_rate_final = total_az_rate
        except Exception as e:
            if self.on_log:
                self.on_log(f"⚠️ Error computing correction: {e}")
            return
        
        # Apply EMA smoothing to the output rates to debounce plate-solve
        # jitter before it reaches the mount motors.
        a = self._rate_smooth_alpha
        self._smooth_alt_rate = a * alt_rate_final + (1 - a) * self._smooth_alt_rate
        self._smooth_az_rate = a * az_rate_final + (1 - a) * self._smooth_az_rate

        # --- Apply ramp factor (gradual start) ---
        # During the first ~1.5s after start(), corrections are ramped up
        # from zero to prevent the initial jolt.
        if self._ramp_factor < 1.0:
            self._ramp_factor = min(1.0, self._ramp_factor + self._ramp_increment)
        ramped_alt = self._smooth_alt_rate * self._ramp_factor
        ramped_az = self._smooth_az_rate * self._ramp_factor

        # --- Max rate clamp ---
        # Hard ceiling prevents runaway rates from ever reaching the mount.
        ramped_alt = max(-self._max_rate, min(self._max_rate, ramped_alt))
        ramped_az = max(-self._max_rate, min(self._max_rate, ramped_az))

        # --- Acceleration limiter ---
        # Limit how much the output rate can change per cycle so corrections
        # are imperceptibly smooth for long-exposure astrophotography.
        delta_alt = ramped_alt - self._prev_sent_alt_rate
        delta_az = ramped_az - self._prev_sent_az_rate
        delta_alt = max(-self._max_accel, min(self._max_accel, delta_alt))
        delta_az = max(-self._max_accel, min(self._max_accel, delta_az))
        send_alt = self._prev_sent_alt_rate + delta_alt
        send_az = self._prev_sent_az_rate + delta_az

        # Remember what we actually sent for next cycle's delta computation
        self._prev_sent_alt_rate = send_alt
        self._prev_sent_az_rate = send_az

        # Send the smoothed, ramped, clamped, acceleration-limited tracking
        # rates OUTSIDE the lock to avoid blocking the plate-solve thread
        # while waiting for serial/TCP I/O.
        try:
            self._send_tracking_rate(send_alt, send_az)
        except Exception as e:
            if self.on_log:
                self.on_log(f"Error sending tracking rate: {e}")
    
    def _calculate_sidereal_altaz_rate(self) -> Tuple[float, float]:
        """Compute the instantaneous sidereal tracking rates in Alt/Az.

        Unlike an equatorial mount where the sidereal rate is a constant 15"/s
        in RA, an Alt/Az mount must continuously vary both axes.  The rates
        depend on:
        - The current altitude and azimuth of the target.
        - The observer's geographic latitude.

        The derivation uses the time derivatives of the Alt/Az transformation
        under pure Earth rotation (angular velocity omega):

            dAlt/dt = omega * cos(lat) * sin(az)
            dAz/dt  = omega * (sin(lat) - tan(alt) * cos(lat) * cos(az))

        Near the zenith (alt -> 90 deg) the azimuth rate diverges (field
        rotation singularity); we clamp it to zero when cos(alt) < 0.01.

        **Optimization**: Uses cached latitude trig values (they
        never change during a session) and skips recomputation when the
        Alt/Az position has changed by less than the cache threshold (0.01 deg
        ~ 36 arcsec). This eliminates redundant trig calls in the 5Hz loop
        when the telescope is tracking smoothly.

        Returns:
            Tuple of (alt_rate, az_rate) in arcsec/sec.
        """
        # Check if the cached result is still valid (position hasn't moved enough)
        if (self._cached_sidereal_alt is not None and
            self._cached_sidereal_az is not None and
            abs(self.current_alt - self._cached_sidereal_alt) < self._sidereal_cache_threshold and
            abs(self.current_az - self._cached_sidereal_az) < self._sidereal_cache_threshold):
            return self._cached_sidereal_result

        # Convert current position to radians (latitude trig is pre-cached)
        alt_rad = math.radians(self.current_alt)
        az_rad = math.radians(self.current_az)
        
        # Earth's rotation rate (rad/sec) -- one rotation per sidereal day
        omega = 7.2921159e-5  # 2*pi / 86164.1, pre-computed constant
        
        cos_alt = math.cos(alt_rad)
        sin_alt = math.sin(alt_rad)
        sin_az = math.sin(az_rad)
        cos_az = math.cos(az_rad)
        # Use pre-cached latitude trig values
        cos_lat = self._cached_cos_lat
        sin_lat = self._cached_sin_lat
        
        # Altitude rate (rad/sec): dAlt/dt = omega * cos(lat) * sin(az)
        rate_alt_rad = omega * cos_lat * sin_az
        
        # Azimuth rate (rad/sec): dAz/dt = omega * (sin(lat) - tan(alt) * cos(lat) * cos(az))
        # Guard against division by zero near the zenith (cos_alt -> 0)
        if cos_alt > 0.01:
            tan_alt = sin_alt / cos_alt
            rate_az_rad = omega * (sin_lat - tan_alt * cos_lat * cos_az)
        else:
            rate_az_rad = 0.0
        
        # Convert from rad/sec to arcsec/sec (1 rad = 206264.806" ≈ degrees * 3600)
        rate_alt = math.degrees(rate_alt_rad) * 3600
        rate_az = math.degrees(rate_az_rad) * 3600
        
        # Cache the result
        self._cached_sidereal_alt = self.current_alt
        self._cached_sidereal_az = self.current_az
        self._cached_sidereal_result = (rate_alt, rate_az)
        
        return rate_alt, rate_az
    
    def _altaz_rate_to_radec_rate(self, alt_rate: float, az_rate: float,
                                   alt: float, az: float) -> Tuple[float, float]:
        """Convert Alt/Az tracking rates to RA/Dec rates for SXTR/SXTD commands.

        The OnStep firmware expects custom tracking rate offsets in equatorial
        coordinates:
        - SXTR: RA rate *offset* from the standard sidereal rate (arcsec/sec).
        - SXTD: Dec rate (arcsec/sec, absolute).

        This method applies the inverse Jacobian of the (RA, Dec) -> (Alt, Az)
        coordinate transformation at the current position, which is a local
        linear approximation valid for the small rate corrections we apply.

        The Jacobian partial derivatives are:
            dRA/dAlt  = -cos(lat) * sin(az) / (cos(dec) * cos(alt))
            dRA/dAz   = (sin(lat)*cos(alt) - cos(lat)*sin(alt)*cos(az)) / (cos(dec)*cos(alt))
            dDec/dAlt = (sin(lat)*cos(alt) - cos(lat)*sin(alt)*cos(az)) / cos(dec)
            dDec/dAz  = cos(lat) * cos(alt) * sin(az) / cos(dec)

        Near the celestial pole (cos(dec) -> 0) the RA rate diverges; in that
        case we return (0, alt_rate) as a safe fallback.

        Args:
            alt_rate: Altitude tracking rate (arcsec/sec).
            az_rate:  Azimuth tracking rate (arcsec/sec).
            alt:      Current altitude (degrees).
            az:       Current azimuth (degrees).

        Returns:
            Tuple of (ra_offset, dec_rate) in arcsec/sec, where ra_offset is
            the offset from the standard sidereal rate (for SXTR).
        """
        # Convert to radians
        alt_rad = math.radians(alt)
        az_rad = math.radians(az)
        lat_rad = math.radians(self.latitude)
        
        # Trig values for the Jacobian computation
        cos_alt = math.cos(alt_rad)
        sin_alt = math.sin(alt_rad)
        cos_az = math.cos(az_rad)
        sin_az = math.sin(az_rad)
        cos_lat = math.cos(lat_rad)
        sin_lat = math.sin(lat_rad)
        
        # Compute declination from Alt/Az (needed for the Jacobian denominator)
        sin_dec = sin_lat * sin_alt + cos_lat * cos_alt * cos_az
        cos_dec = math.sqrt(max(1 - sin_dec**2, 1e-12))
        
        # Near the celestial pole OR near zenith the Jacobian is singular;
        # return a safe fallback to avoid division-by-zero blow-up.
        if cos_dec < 0.01 or cos_alt < 0.01:
            return 0.0, alt_rate
        
        # Inverse Jacobian matrix elements (local linear approximation)
        # Maps (dAlt, dAz) -> (dRA, dDec)
        
        # dRA/dAlt and dRA/dAz
        dra_dalt = -cos_lat * sin_az / (cos_dec * cos_alt)
        dra_daz = (sin_lat * cos_alt - cos_lat * sin_alt * cos_az) / (cos_dec * cos_alt)
        
        # dDec/dAlt and dDec/dAz
        ddec_dalt = (sin_lat * cos_alt - cos_lat * sin_alt * cos_az) / cos_dec
        ddec_daz = cos_lat * cos_alt * sin_az / cos_dec
        
        # Apply the transformation: convert input rates from arcsec/sec to deg/sec
        alt_rate_deg = alt_rate / 3600
        az_rate_deg = az_rate / 3600
        
        # Compute RA and Dec rates (deg/sec) via the Jacobian
        ra_rate_deg = dra_dalt * alt_rate_deg + dra_daz * az_rate_deg
        dec_rate_deg = ddec_dalt * alt_rate_deg + ddec_daz * az_rate_deg
        
        # Convert back to arcsec per SI second
        ra_rate = ra_rate_deg * 3600
        dec_rate = dec_rate_deg * 3600

        # SXTR is an OFFSET from the mount's current base tracking rate.
        # Subtract whichever base rate the mount firmware is currently using
        # (sidereal / lunar / solar / king) so the mount only applies the
        # differential correction.
        base_rate = self.TRACKING_RATES.get(self._base_rate_mode, self.SIDEREAL_RATE)
        ra_offset = ra_rate - base_rate

        # OnStepX SXTR/SXTD units are arcsec per *sidereal* second, but
        # the pipeline computes in arcsec per SI second.  Apply the
        # conversion factor (1 SI sec = ~1.002738 sidereal seconds).
        ra_offset *= self.SI_TO_SIDEREAL
        dec_rate *= self.SI_TO_SIDEREAL

        return ra_offset, dec_rate
    
    def _radec_rate_to_altaz_rate(self, ra_rate: float, dec_rate: float,
                                   alt: float, az: float) -> Tuple[float, float]:
        """Convert RA/Dec correction rates to Alt/Az rates.

        This is the forward Jacobian of the (RA, Dec) -> (Alt, Az) transformation,
        i.e. the matrix inverse of ``_altaz_rate_to_radec_rate``.  Used to properly
        map RA/Dec correction rates into the Alt/Az frame used by the
        tracking pipeline.

        The forward Jacobian elements are:
            dAlt/dRA  = -cos(lat) * sin(az) * cos(dec)
            dAlt/dDec =  sin(lat)*cos(dec) - cos(lat)*sin(dec)*cos(az)  (simplified)
            dAz/dRA   =  cos(lat)*cos(az)*cos(dec) / cos(alt)
            dAz/dDec  = (cos(lat)*sin(dec)*sin(az)) / cos(alt)         (simplified)

        Rather than re-derive these, we invert the 2x2 inverse-Jacobian matrix
        from ``_altaz_rate_to_radec_rate`` which is already validated.

        Near the zenith (cos_alt -> 0) the Az term diverges; returns (ra_rate, 0)
        as a safe fallback.

        Args:
            ra_rate:  RA correction rate (arcsec/sec).
            dec_rate: Dec correction rate (arcsec/sec).
            alt:      Current altitude (degrees).
            az:       Current azimuth (degrees).

        Returns:
            Tuple of (alt_rate, az_rate) in arcsec/sec.
        """
        alt_rad = math.radians(alt)
        az_rad = math.radians(az)
        lat_rad = math.radians(self.latitude)

        cos_alt = math.cos(alt_rad)
        sin_alt = math.sin(alt_rad)
        cos_az = math.cos(az_rad)
        sin_az = math.sin(az_rad)
        cos_lat = math.cos(lat_rad)
        sin_lat = math.sin(lat_rad)

        # Declination from current Alt/Az (needed for the Jacobian)
        sin_dec = sin_lat * sin_alt + cos_lat * cos_alt * cos_az
        cos_dec = math.sqrt(max(1 - sin_dec ** 2, 1e-12))

        # Guard: near celestial pole the mapping is ill-conditioned
        if cos_dec < 0.01 or cos_alt < 0.01:
            return ra_rate, dec_rate

        # Inverse Jacobian matrix J^{-1}: maps (dAlt,dAz) -> (dRA,dDec)
        # (same as in _altaz_rate_to_radec_rate)
        j11 = -cos_lat * sin_az / (cos_dec * cos_alt)
        j12 = (sin_lat * cos_alt - cos_lat * sin_alt * cos_az) / (cos_dec * cos_alt)
        j21 = (sin_lat * cos_alt - cos_lat * sin_alt * cos_az) / cos_dec
        j22 = cos_lat * cos_alt * sin_az / cos_dec

        # The forward Jacobian J = (J^{-1})^{-1}.  For a 2x2 matrix:
        #   J = (1/det) * [[j22, -j12], [-j21, j11]]
        det = j11 * j22 - j12 * j21
        if abs(det) < 1e-12:
            return ra_rate, dec_rate  # Degenerate: fallback to identity

        # Convert input rates from arcsec/sec to deg/sec for the matrix multiply
        ra_deg = ra_rate / 3600
        dec_deg = dec_rate / 3600

        alt_rate_deg = (j22 * ra_deg - j12 * dec_deg) / det
        az_rate_deg = (-j21 * ra_deg + j11 * dec_deg) / det

        return alt_rate_deg * 3600, az_rate_deg * 3600

    def _send_tracking_rate(self, alt_rate: float, az_rate: float):
        """Send the computed Alt/Az tracking rates to the telescope mount.

        This method is the final step of the correction pipeline.  It:

        1. Stores the Alt/Az rates locally for GUI display.
        2. **If a NexStar-style variable-rate callback is registered**
           (``send_variable_rate_altaz``), sends Alt/Az rates directly via
           the NexStar passthrough 'P' command.  This is more accurate for
           Alt/Az mounts because it avoids the Jacobian approximation.
        3. **Otherwise**, converts to RA/Dec via ``_altaz_rate_to_radec_rate()``
           and sends SXTR/SXTD commands for OnStep/LX200 mounts.

        Args:
            alt_rate: Final altitude tracking rate (arcsec/sec).
            az_rate:  Final azimuth tracking rate (arcsec/sec).
        """
        try:
            # Store the Alt/Az rates locally (used by GUI and stats)
            self.tracking_rate.alt_rate = alt_rate
            self.tracking_rate.az_rate = az_rate

            # --- Path A: NexStar variable-rate passthrough (preferred for Alt/Az) ---
            if self.send_variable_rate_altaz is not None:
                try:
                    self.send_variable_rate_altaz(alt_rate, az_rate)
                except Exception as e:
                    if self.on_log:
                        self.on_log(f"Error sending NexStar variable rate: {e}")
            # --- Path B: LX200 / OnStep SXTR/SXTD (equatorial rate offsets) ---
            elif self.send_command:
                # Convert Alt/Az rates to RA/Dec for the SXTR/SXTD protocol
                # OnStep requires equatorial rate offsets regardless of mount type
                ra_offset, dec_offset = self._altaz_rate_to_radec_rate(
                    alt_rate, az_rate, self.current_alt, self.current_az
                )

                cmd_ra = f":SXTR,{ra_offset:.4f}#"    # RA rate offset from sidereal
                cmd_dec = f":SXTD,{dec_offset:.4f}#"  # Dec rate (absolute)

                try:
                    self.send_command(cmd_ra)
                    self.send_command(cmd_dec)
                except Exception as e:
                    if self.on_log:
                        self.on_log(f"Error sending tracking commands: {e}")
            
            # Notify the GUI with Alt/Az rates (not RA/Dec, since the GUI
            # displays the native Alt/Az frame)
            if self.on_rate_update:
                try:
                    self.on_rate_update(alt_rate, az_rate)
                except Exception as e:
                    if self.on_log:
                        self.on_log(f"Error in rate update callback: {e}")
        except Exception as e:
            if self.on_log:
                self.on_log(f"Error in _send_tracking_rate: {e}")
    
    def _radec_to_altaz(self, ra_hours: float, dec_deg: float) -> Tuple[float, float]:
        """Convert equatorial RA/Dec to horizontal Alt/Az for the local observer.

        Uses the standard spherical astronomy transformation:
        1. Compute the Local Sidereal Time (LST) from the system clock.
        2. Derive the Hour Angle: HA = LST - RA.
        3. Apply the rotation for the observer's latitude to get (Alt, Az).

        The azimuth convention is 0=North, 90=East, 180=South, 270=West,
        matching the standard astronomical convention.

        Args:
            ra_hours: Right ascension in decimal hours [0, 24).
            dec_deg:  Declination in decimal degrees [-90, +90].

        Returns:
            Tuple of (altitude_degrees, azimuth_degrees).
        """
        # Compute local sidereal time (hours)
        lst = self._local_sidereal_time()
        
        # Hour angle in degrees (LST and RA are both in hours; convert to degrees)
        ha = (lst - ra_hours) * 15
        ha_rad = math.radians(ha)
        dec_rad = math.radians(dec_deg)
        lat_rad = math.radians(self.latitude)
        
        # Altitude: sin(alt) = sin(lat)*sin(dec) + cos(lat)*cos(dec)*cos(ha)
        sin_alt = (math.sin(lat_rad) * math.sin(dec_rad) +
                   math.cos(lat_rad) * math.cos(dec_rad) * math.cos(ha_rad))
        alt = math.degrees(math.asin(max(-1, min(1, sin_alt))))
        
        # Azimuth: cos(az) = (sin(dec) - sin(lat)*sin(alt)) / (cos(lat)*cos(alt))
        cos_az = ((math.sin(dec_rad) - math.sin(lat_rad) * math.sin(math.radians(alt))) /
                  (math.cos(lat_rad) * math.cos(math.radians(alt))))
        cos_az = max(-1, min(1, cos_az))  # Clamp to [-1, 1] for numerical safety
        az = math.degrees(math.acos(cos_az))
        
        # Resolve the azimuth quadrant ambiguity: if sin(HA) > 0, az = 360 - az
        if math.sin(ha_rad) > 0:
            az = 360 - az
        
        return alt, az
    
    def _local_sidereal_time(self) -> float:
        """Compute the Local Sidereal Time (LST) in decimal hours.

        Uses the standard algorithm:
        1. Compute the Julian Date from the current UTC time.
        2. Compute Greenwich Sidereal Time (GST) from the Julian Date.
        3. Add the observer's longitude to get LST.

        The LST tells us which RA is currently on the local meridian, and is
        essential for the RA/Dec <-> Alt/Az conversions.

        Returns:
            Local Sidereal Time in decimal hours [0, 24).
        """
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        
        # Compute Julian Date
        year = now.year
        month = now.month
        day = now.day + now.hour/24 + now.minute/1440 + now.second/86400
        
        # Adjust for Jan/Feb (treat as months 13/14 of the previous year)
        if month <= 2:
            year -= 1
            month += 12
        
        # Julian Date formula (Meeus, Astronomical Algorithms)
        a = int(year / 100)
        b = 2 - a + int(a / 4)
        jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
        
        # Greenwich Sidereal Time (degrees) from Julian Date
        t = (jd - 2451545.0) / 36525.0
        gst = 280.46061837 + 360.98564736629 * (jd - 2451545.0)
        gst = gst % 360  # Normalize to [0, 360)
        
        # Local Sidereal Time = GST + observer longitude, converted to hours
        lst = (gst + self.longitude) / 15.0
        return lst % 24
    
    def _update_avg_solve_time(self, new_time_ms: float):
        """Update the cumulative average plate-solve time.

        Uses a simple running mean over all successful solves.

        Args:
            new_time_ms: The latest solve duration in milliseconds.
        """
        n = self.stats['successful_solves']
        if n == 1:
            self.stats['avg_solve_time'] = new_time_ms
        else:
            self.stats['avg_solve_time'] = (
                self.stats['avg_solve_time'] * (n - 1) + new_time_ms
            ) / n
    
    def _update_avg_correction(self, correction: float):
        """Update the exponential moving average of correction magnitude.

        Uses an EMA with alpha=0.1, giving a smoothing window of roughly
        10 samples (~2 seconds at 5Hz).

        Args:
            correction: The latest correction magnitude in arcsec/sec.
        """
        n = self.stats['total_corrections']
        if n == 1:
            self.stats['avg_correction'] = correction
        else:
            # Exponential moving average (alpha = 0.1)
            alpha = 0.1
            self.stats['avg_correction'] = (
                self.stats['avg_correction'] * (1 - alpha) + correction * alpha
            )
    
    def get_stats(self) -> dict:
        """Return a consolidated dictionary of tracking statistics.

        Aggregates statistics from all sub-components (Kalman, ML, PEC)
        into a single flat dictionary suitable for GUI display.

        Returns:
            Dictionary containing solve counts, timing, correction averages,
            sub-component status flags, and current tracking rates.
        """
        with self._lock:
            kalman_stats = self.kalman.get_statistics()
            ml_stats = self.ml_predictor.get_statistics()
            
            pec_stats = self.pec.get_statistics() if self.pec_enabled else {}
            
            return {
                **self.stats,
                'kalman_rms_arcsec': kalman_stats.get('rms_alt_arcsec', 0),
                'ml_samples': ml_stats.get('samples', 0),
                'ml_ready': ml_stats.get('model_ready', False),
                'current_alt_rate': self.tracking_rate.alt_rate,
                'current_az_rate': self.tracking_rate.az_rate,
                'pec_trained': pec_stats.get('is_trained', False),
                'pec_enabled': self.pec_enabled,
                'pec_periods_alt': pec_stats.get('periods_detected_alt', 0),
                'pec_periods_az': pec_stats.get('periods_detected_az', 0),
                'pec_samples': pec_stats.get('total_samples', 0),
                'pec_data_span': pec_stats.get('data_span_sec', 0.0),
                # Environmental corrections
                'refraction_enabled': self.refraction_enabled,
                'thermal_drift_enabled': self.thermal_drift_enabled,
                'temperature_c': self._temperature_c if self._temperature_c is not None else 0.0,
                'pressure_hpa': self._pressure_hpa if self._pressure_hpa is not None else 0.0,
                'base_temperature_c': self._base_temperature_c if self._base_temperature_c is not None else 0.0,
                'thermal_coefficient': self.thermal_coefficient,
                'refraction_offset_arcsec': self._prev_refraction_deg * 3600,
                # Flexure model
                'flexure_enabled': self.flexure_enabled,
                'flexure_samples': (
                    self.flexure_model.total_samples
                    if self.flexure_model else 0
                ),
                'flexure_coverage_pct': (
                    self.flexure_model.stats.get('grid_coverage_pct', 0.0)
                    if self.flexure_model else 0.0
                ),
                # Drive type
                'drive_type': getattr(self.pec, 'drive_type', 'unknown'),
            }
    
    def get_correction_history(self) -> list:
        """Return a snapshot of the correction history for graph rendering.

        Returns:
            List of CorrectionRecord instances (copied from the internal deque
            so the caller can iterate without holding the lock).
        """
        with self._lock:
            return list(self.correction_history)
    
    def get_graph_data(self) -> dict:
        """Return correction history formatted as numpy arrays for plotting.

        Converts the deque of CorrectionRecord objects into a dictionary of
        aligned numpy arrays, one per data series.  Timestamps are normalized
        to start at zero (relative to the first record).

        Returns:
            Dictionary with keys: 'timestamps', 'kalman_alt', 'kalman_az',
            'ml_alt', 'ml_az', 'pec_alt', 'pec_az',
            'total_alt', 'total_az', 'error_alt', 'error_az'.
            Each value is a 1D numpy array.  Returns empty lists if no data.
        """
        with self._lock:
            if len(self.correction_history) == 0:
                return {
                    'timestamps': [],
                    'kalman_alt': [], 'kalman_az': [],
                    'ml_alt': [], 'ml_az': [],
                    'pec_alt': [], 'pec_az': [],
                    'refraction_alt': [],
                    'thermal_alt': [], 'thermal_az': [],
                    'flexure_alt': [], 'flexure_az': [],
                    'total_alt': [], 'total_az': [],
                    'error_alt': [], 'error_az': []
                }
            
            # Convert deque to list for indexed access
            records = list(self.correction_history)
            t0 = records[0].timestamp if records else time.time()
            
            return {
                'timestamps': np.array([r.timestamp - t0 for r in records]),
                'kalman_alt': np.array([r.kalman_alt for r in records]),
                'kalman_az': np.array([r.kalman_az for r in records]),
                'ml_alt': np.array([r.ml_alt for r in records]),
                'ml_az': np.array([r.ml_az for r in records]),
                'pec_alt': np.array([r.pec_alt for r in records]),
                'pec_az': np.array([r.pec_az for r in records]),
                'refraction_alt': np.array([r.refraction_alt for r in records]),
                'thermal_alt': np.array([r.thermal_alt for r in records]),
                'thermal_az': np.array([r.thermal_az for r in records]),
                'flexure_alt': np.array([getattr(r, 'flexure_alt', 0.0) for r in records]),
                'flexure_az': np.array([getattr(r, 'flexure_az', 0.0) for r in records]),
                'total_alt': np.array([r.total_alt for r in records]),
                'total_az': np.array([r.total_az for r in records]),
                'error_alt': np.array([r.error_alt for r in records]),
                'error_az': np.array([r.error_az for r in records])
            }
    
    def _log(self, message: str):
        """Emit a log message via the registered callback (if any).

        Args:
            message: The log string to emit.
        """
        if self.on_log:
            self.on_log(message)
    
    def _log_wrapper(self, message: str):
        """Log wrapper passed to sub-components (e.g., PEC) as their on_log callback.

        Args:
            message: The log string to emit.
        """
        self._log(message)


class FastPlateSolver:
    """Lightweight wrapper around ASTAP for fast plate solving (target < 500ms).

    This class configures ASTAP with aggressive settings optimized for speed
    rather than accuracy, since the Kalman filter downstream will smooth out
    any noise.  The key optimizations are:

    - **Aggressive downsampling** (4x): reduces the image size so star detection
      and pattern matching run on far fewer pixels.
    - **Position hint**: provides the expected RA/Dec so ASTAP starts its
      search near the correct answer (reduces search from full-sky to 5 deg).
    - **Reduced star count** (max 100): limits the number of stars extracted
      for pattern matching.
    - **Short timeout** (2s): fails fast if the solve is taking too long
      (better to skip a frame than stall the pipeline).

    After each successful solve, the hint is automatically updated to the
    solved position so the next solve starts from the best prior.

    Attributes:
        astap_path:     Path to the ASTAP executable.
        downsample:     Image downsampling factor.
        search_radius:  Sky search radius in degrees.
        max_stars:      Maximum stars to extract.
        timeout:        Subprocess timeout in seconds.
        hint_ra:        RA hint for the next solve (hours).
        hint_dec:       Dec hint for the next solve (degrees).
        solve_times:    Deque of recent solve times (ms) for averaging.
    """
    
    def __init__(self, astap_path: str = "astap"):
        """Initialize the fast plate solver.

        Args:
            astap_path: Path to the ASTAP binary (default: "astap", assumes
                        it is on the system PATH).
        """
        self.astap_path = astap_path
        
        # Configuration tuned for fast solving (< 500ms)
        self.downsample = 4         # Aggressive downsampling (4x)
        self.search_radius = 5.0    # Only search within 5 degrees of the hint
        self.max_stars = 100        # Fewer stars = faster pattern matching
        self.timeout = 2.0          # Short timeout to fail fast
        self.fov_deg = 0.0          # Estimated FOV in degrees (0 = not set)
        
        # Position hint (updated after each successful solve)
        self.hint_ra = 0.0
        self.hint_dec = 0.0
        
        # Recent solve times for performance monitoring
        self.solve_times = deque(maxlen=100)
    
    def solve_fast(self, image_path: str) -> Optional[Tuple[float, float, float]]:
        """Perform a fast plate solve on the given image.

        Invokes ASTAP as a subprocess with speed-optimized parameters and
        parses the resulting WCS file for the solved RA/Dec.

        Args:
            image_path: Filesystem path to the FITS or image file to solve.

        Returns:
            A tuple of (ra_hours, dec_degrees, solve_time_ms) on success,
            or None if the solve failed or timed out.
        """
        import subprocess
        import os
        import re
        
        start_time = time.time()
        
        # Build the ASTAP command line with speed-optimized flags
        cmd = [
            self.astap_path,
            "-f", image_path,
            "-r", str(self.search_radius),    # Search radius (degrees)
            "-z", str(self.downsample),        # Downsample factor
            "-s", str(self.max_stars),         # Max stars to extract
            "-ra", str(self.hint_ra),          # RA hint (hours)
            "-spd", str(90 + self.hint_dec),   # South Pole Distance = 90 + Dec
            "-update"                          # Write WCS to file
        ]

        # Add FOV estimate if configured (helps ASTAP converge faster)
        if self.fov_deg > 0:
            cmd.extend(["-fov", f"{self.fov_deg:.2f}"])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=self.timeout
            )
            
            solve_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.solve_times.append(solve_time)
            
            if result.returncode == 0:
                # Parse the WCS output file for the solved coordinates
                wcs_path = os.path.splitext(image_path)[0] + ".wcs"
                if os.path.exists(wcs_path):
                    with open(wcs_path, 'r') as f:
                        content = f.read()
                    
                    # Extract CRVAL1 (RA in degrees) and CRVAL2 (Dec in degrees)
                    ra_match = re.search(r'CRVAL1\s*=\s*([-\d.]+)', content)
                    dec_match = re.search(r'CRVAL2\s*=\s*([-\d.]+)', content)
                    
                    if ra_match and dec_match:
                        ra_deg = float(ra_match.group(1))
                        ra_hours = ra_deg / 15.0  # Convert degrees to hours
                        dec_deg = float(dec_match.group(1))
                        
                        # Update the hint for the next solve (warm start)
                        self.hint_ra = ra_hours
                        self.hint_dec = dec_deg
                        
                        return ra_hours, dec_deg, solve_time
            
            return None
            
        except subprocess.TimeoutExpired:
            return None
        except Exception:
            return None
    
    def get_average_solve_time(self) -> float:
        """Return the average plate-solve time over recent solves.

        Returns:
            Average solve time in milliseconds, or 0.0 if no solves yet.
        """
        if not self.solve_times:
            return 0.0
        return sum(self.solve_times) / len(self.solve_times)
