"""
Extended Kalman Filter (EKF) for Alt-Azimuth Telescope Tracking.

Architecture Role:
    This module provides the non-linear state estimation layer in the
    correction pipeline.  It replaces the previous linear Kalman filter
    with a sidereal-aware Extended Kalman Filter that properly models the
    non-linear motion of celestial objects in Alt/Az coordinates.

    Data Flow:
        plate_solve (Alt/Az) --> EKF.update() --> smoothed position + drift velocity
                                              --> used by realtime_tracking.py

    The filter maintains a 4-dimensional state vector [alt, az, drift_alt, drift_az]:
      - alt, az:       True telescope position in degrees.
      - drift_alt, drift_az: Tracking *error* velocity in degrees/second
                             (deviation from ideal sidereal motion).

    Unlike the old linear filter that used a constant-velocity model (F*x),
    this EKF models the actual sidereal motion of the sky using spherical
    trigonometry, then linearises the model at each step via the Jacobian
    matrix.  This correctly separates the known sidereal rate from the
    unknown drift, yielding more accurate drift estimates at all sky
    positions and proper handling of the zenith singularity.

    The EKF contribution to the final correction is weighted at 35-45 %
    (see realtime_tracking.py for the full fusion pipeline).

Why EKF instead of linear?
    - The sidereal rate in Alt/Az is NOT constant -- it depends on the
      current position via sin/cos/tan functions.  A linear filter cannot
      model this and confuses sidereal motion with drift.
    - Near the zenith, the azimuth rate diverges (field rotation
      singularity).  The EKF's Jacobian captures this acceleration,
      preventing the filter from producing wildly wrong drift estimates.
    - The EKF also adapts its process noise Q near the zenith,
      acknowledging that azimuth predictions become unreliable there.

Classes:
    KalmanState          -- Dataclass holding filter state snapshot.
    AdaptiveKalmanFilter -- EKF with sidereal model, Jacobian, adaptive R,
                           and zenith-aware Q adaptation.  The class name
                           is kept for backward compatibility with existing
                           imports throughout the codebase.

Key Concepts:
    - Jacobian F: Partial derivatives of the non-linear state transition
      f(x), computed at every predict step.  This is the core of the EKF.
    - Q matrix (process noise): Models drift uncertainty.  Automatically
      inflated near the zenith where azimuth is ill-defined.
    - R matrix (measurement noise): Plate-solve accuracy.  Adaptively
      updated from observed innovation statistics.
    - Sidereal rates: Computed from the observer's latitude and the
      current (Alt, Az) position using the standard formulas:
          dAlt/dt = omega * cos(lat) * sin(az)
          dAz/dt  = omega * (sin(lat) - tan(alt) * cos(lat) * cos(az))

Dependencies:
    - numpy: Matrix operations for Kalman equations.
    - Used by: realtime_tracking.py (RealTimeTrackingController).
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from collections import deque
import time
import math

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Earth's sidereal rotation rate (rad/s) -- 2*pi / 86164.0905 s
OMEGA_EARTH = 7.2921159e-5

# Conversion factors
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


@dataclass
class KalmanState:
    """Snapshot of the EKF internal state.

    Used for debugging and telemetry; not consumed by the correction
    pipeline directly.

    Attributes:
        position:   [alt, az] in degrees.
        velocity:   [drift_alt, drift_az] in degrees/second.
        covariance: 4x4 error covariance matrix P.
        timestamp:  Unix timestamp of the last update.
    """
    position: np.ndarray      # [alt, az] in degrees
    velocity: np.ndarray      # [drift_alt, drift_az] in degrees/second
    covariance: np.ndarray    # 4x4 covariance matrix
    timestamp: float          # Unix timestamp of last update


class AdaptiveKalmanFilter:
    """Extended Kalman Filter with sidereal-aware state transition.

    This class replaces both the old ``TelescopeKalmanFilter`` and its
    ``AdaptiveKalmanFilter`` subclass.  The class name is kept as
    ``AdaptiveKalmanFilter`` so that existing imports in
    ``realtime_tracking.py``, ``crash_recovery.py``, ``web_server.py``,
    and ``tracking_improvements.py`` continue to work unchanged.

    State vector: x = [alt, az, drift_alt, drift_az]
        - alt, az:           Current position in degrees.
        - drift_alt, drift_az: Tracking error velocity in degrees/second.
                              This is the deviation from ideal sidereal
                              motion -- the quantity the correction pipeline
                              needs to counteract.

    Non-linear state transition f(x, dt):
        alt_new = alt + (sidereal_rate_alt(alt, az) + drift_alt) * dt
        az_new  = az  + (sidereal_rate_az(alt, az)  + drift_az)  * dt
        drift_alt_new = drift_alt    (random walk + process noise)
        drift_az_new  = drift_az

    Jacobian F = df/dx (computed at each step):
        F = [[1,                        d_sid_alt/d_az * dt,  dt, 0 ],
             [d_sid_az/d_alt * dt,  1 + d_sid_az/d_az * dt,  0,  dt],
             [0,                        0,                     1,  0 ],
             [0,                        0,                     0,  1 ]]

    Observation model (linear -- plate-solving gives Alt/Az directly):
        H = [[1, 0, 0, 0],
             [0, 1, 0, 0]]

    Typical usage:
        kf = AdaptiveKalmanFilter()
        kf.set_latitude(45.0)          # observer latitude
        kf.initialize(alt=45.0, az=180.0)
        # Each plate-solve result:
        smoothed_alt, smoothed_az = kf.update(measured_alt, measured_az)
        drift_alt, drift_az = kf.get_velocity()
    """

    def __init__(self):
        # --- Dimensions ---
        self.state_dim = 4
        self.measurement_dim = 2

        # --- State vector [alt, az, drift_alt, drift_az] ---
        self.x = np.zeros(self.state_dim)

        # --- Initial covariance: high uncertainty before first measurement ---
        self.P = np.eye(self.state_dim) * 100.0

        # --- Process noise Q ---
        # Position noise is lower than the old linear filter because
        # sidereal motion is now modeled explicitly (less model error).
        # Drift noise allows gradual drift changes from wind, gear
        # imperfections, thermal flexure, etc.
        self.Q = np.array([
            [0.005, 0,     0,     0    ],   # Position alt noise (deg^2)
            [0,     0.005, 0,     0    ],   # Position az noise  (deg^2)
            [0,     0,     0.001, 0    ],   # Drift alt noise    (deg^2/s^2)
            [0,     0,     0,     0.001],   # Drift az noise     (deg^2/s^2)
        ])
        # Base copy for zenith Q-adaptation (restored when away from zenith)
        self._Q_base = self.Q.copy()

        # --- Measurement noise R ---
        # 0.001 deg^2 corresponds to ~3.6 arcsec standard deviation,
        # matching typical ASTAP plate-solve accuracy.
        self.R = np.array([
            [0.001, 0    ],
            [0,     0.001],
        ])

        # --- Observation matrix (linear: Alt/Az measured directly) ---
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        # --- Pre-allocated work matrices ---
        self._F = np.eye(self.state_dim, dtype=np.float64)
        self._I = np.eye(self.state_dim, dtype=np.float64)

        # --- Observer latitude (defaults; overridden by set_latitude) ---
        self._lat_deg = 48.8566
        self._lat_rad = self._lat_deg * DEG2RAD
        self._cos_lat = math.cos(self._lat_rad)
        self._sin_lat = math.sin(self._lat_rad)

        # --- Timestamp tracking ---
        self.last_update_time: Optional[float] = None

        # --- History buffer for performance analysis ---
        self.max_history = 500
        self.history: deque = deque(maxlen=self.max_history)

        # --- Adaptive R settings ---
        self.window_size = 20
        self.adaptation_rate = 0.1
        self.residual_window: deque = deque(maxlen=self.window_size)

        # --- Filter state ---
        self.is_initialized = False

        # --- Zenith protection ---
        # Above this altitude the Q matrix is inflated for azimuth states.
        self._zenith_threshold_deg = 85.0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_latitude(self, lat_deg: float):
        """Set the observer's geographic latitude.

        MUST be called before tracking begins so the sidereal model uses
        the correct latitude.  The tracking controller calls this when the
        GPS fix arrives or when the user configures the location.

        Args:
            lat_deg: Observer latitude in decimal degrees (-90 to +90).
        """
        self._lat_deg = lat_deg
        self._lat_rad = lat_deg * DEG2RAD
        self._cos_lat = math.cos(self._lat_rad)
        self._sin_lat = math.sin(self._lat_rad)

    # ------------------------------------------------------------------
    # Sidereal model (non-linear)
    # ------------------------------------------------------------------

    def _sidereal_rates(self, alt_deg: float, az_deg: float) -> Tuple[float, float]:
        """Compute instantaneous sidereal tracking rates in Alt/Az.

        These are the rates at which a fixed point on the celestial sphere
        moves through local Alt/Az coordinates due to Earth's rotation.

            dAlt/dt = omega * cos(lat) * sin(az)          [rad/s]
            dAz/dt  = omega * (sin(lat) - tan(alt) * cos(lat) * cos(az))

        Returned values are in **degrees/second**.

        Near the zenith (cos(alt) -> 0) the azimuth rate diverges; we
        clamp it to avoid numerical blow-up.

        Args:
            alt_deg: Altitude in degrees.
            az_deg:  Azimuth in degrees.

        Returns:
            (rate_alt, rate_az) in degrees/second.
        """
        alt_rad = alt_deg * DEG2RAD
        az_rad  = az_deg  * DEG2RAD

        cos_alt = math.cos(alt_rad)
        sin_alt = math.sin(alt_rad)
        sin_az  = math.sin(az_rad)
        cos_az  = math.cos(az_rad)

        # dAlt/dt (rad/s)
        rate_alt_rad = OMEGA_EARTH * self._cos_lat * sin_az

        # dAz/dt (rad/s) -- guarded near zenith
        if cos_alt > 0.01:
            tan_alt = sin_alt / cos_alt
            rate_az_rad = OMEGA_EARTH * (self._sin_lat
                                         - tan_alt * self._cos_lat * cos_az)
        else:
            # Near zenith: rate diverges; clamp to zero.
            rate_az_rad = 0.0

        return rate_alt_rad * RAD2DEG, rate_az_rad * RAD2DEG

    # ------------------------------------------------------------------
    # Jacobian of the state transition  (THE core EKF operation)
    # ------------------------------------------------------------------

    def _compute_jacobian(self, alt_deg: float, az_deg: float,
                          dt: float) -> np.ndarray:
        """Compute the Jacobian F = df/dx of the state transition.

        This linearises the non-linear sidereal model at the current
        operating point, enabling proper covariance propagation through
        the non-linear dynamics.

        Partial derivatives of the sidereal rates (the DEG2RAD and
        RAD2DEG chain-rule factors cancel out, leaving clean formulas):

            d(rate_alt)/d(alt) = 0
            d(rate_alt)/d(az)  = omega * cos(lat) * cos(az)
            d(rate_az)/d(alt)  = -omega * cos(lat) * cos(az) / cos^2(alt)
            d(rate_az)/d(az)   = omega * tan(alt) * cos(lat) * sin(az)

        Args:
            alt_deg: Current altitude in degrees.
            az_deg:  Current azimuth in degrees.
            dt:      Time step in seconds.

        Returns:
            4x4 Jacobian matrix F (mutated in-place on pre-allocated array).
        """
        alt_rad = alt_deg * DEG2RAD
        az_rad  = az_deg  * DEG2RAD

        cos_alt = math.cos(alt_rad)
        sin_alt = math.sin(alt_rad)
        sin_az  = math.sin(az_rad)
        cos_az  = math.cos(az_rad)

        # --- Partial derivatives of sidereal rates ---

        # d(rate_alt)/d(alt) = 0  (rate_alt independent of alt)
        dra_dalt = 0.0

        # d(rate_alt)/d(az) = omega * cos(lat) * cos(az)
        dra_daz = OMEGA_EARTH * self._cos_lat * cos_az

        # d(rate_az)/d(alt) = -omega * cos(lat) * cos(az) / cos^2(alt)
        # Near zenith (cos_alt < 0.05 ~ alt > 87.1 deg) clamp denominator
        cos_alt_clamped = max(cos_alt, 0.05)
        draz_dalt = (-OMEGA_EARTH * self._cos_lat * cos_az
                     / (cos_alt_clamped * cos_alt_clamped))

        # d(rate_az)/d(az) = omega * tan(alt) * cos(lat) * sin(az)
        tan_alt = sin_alt / cos_alt_clamped
        draz_daz = OMEGA_EARTH * tan_alt * self._cos_lat * sin_az

        # --- Build the 4x4 Jacobian in-place ---
        F = self._F

        # Row 0: d(alt_new)/d(state)
        F[0, 0] = 1.0                    # d(alt_new)/d(alt)       = 1 + 0*dt
        F[0, 1] = dra_daz * dt           # d(alt_new)/d(az)        = d(rate_alt)/d(az) * dt
        F[0, 2] = dt                     # d(alt_new)/d(drift_alt) = dt
        F[0, 3] = 0.0                    # d(alt_new)/d(drift_az)  = 0

        # Row 1: d(az_new)/d(state)
        F[1, 0] = draz_dalt * dt         # d(az_new)/d(alt)        = d(rate_az)/d(alt) * dt
        F[1, 1] = 1.0 + draz_daz * dt   # d(az_new)/d(az)         = 1 + d(rate_az)/d(az) * dt
        F[1, 2] = 0.0                    # d(az_new)/d(drift_alt)  = 0
        F[1, 3] = dt                     # d(az_new)/d(drift_az)   = dt

        # Rows 2-3: drift is a random walk (identity)
        F[2, 0] = 0.0;  F[2, 1] = 0.0;  F[2, 2] = 1.0;  F[2, 3] = 0.0
        F[3, 0] = 0.0;  F[3, 1] = 0.0;  F[3, 2] = 0.0;  F[3, 3] = 1.0

        return F

    # ------------------------------------------------------------------
    # Zenith-adaptive process noise
    # ------------------------------------------------------------------

    def _adapt_Q_for_zenith(self, alt_deg: float):
        """Inflate azimuth process noise near the zenith.

        Near the zenith the azimuth rate diverges (field rotation
        singularity) and the azimuth coordinate itself becomes
        ill-defined.  We increase Q for the az-related states so the
        filter widens its uncertainty there instead of trusting an
        unreliable prediction.

        The scale factor grows as (cos(threshold) / cos(alt))^2,
        capped at 10 000x to prevent overflow.

        Args:
            alt_deg: Current altitude in degrees.
        """
        if alt_deg > self._zenith_threshold_deg:
            cos_alt = math.cos(alt_deg * DEG2RAD)
            cos_thr = math.cos(self._zenith_threshold_deg * DEG2RAD)
            if cos_alt > 1e-6:
                factor = (cos_thr / cos_alt) ** 2
                factor = min(factor, 10000.0)
            else:
                factor = 10000.0
            self.Q = self._Q_base.copy()
            self.Q[1, 1] *= factor   # az position noise
            self.Q[3, 3] *= factor   # az drift noise
        else:
            # Away from zenith: restore base Q
            np.copyto(self.Q, self._Q_base)

    # ------------------------------------------------------------------
    # Azimuth wraparound helper
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_az_residual(residual_az: float) -> float:
        """Wrap an azimuth residual into [-180, +180) degrees.

        Prevents a 359 deg -> 1 deg jump from being interpreted as a
        358 deg innovation.
        """
        if residual_az > 180.0:
            residual_az -= 360.0
        elif residual_az < -180.0:
            residual_az += 360.0
        return residual_az

    # ------------------------------------------------------------------
    # Public API (same signatures as the old filter)
    # ------------------------------------------------------------------

    def initialize(self, alt: float, az: float):
        """Set the filter to a known starting position with zero drift.

        Called once when tracking begins or after a reset.  Sets high
        initial covariance so the filter converges quickly from
        subsequent measurements.

        Args:
            alt: Initial altitude in degrees.
            az:  Initial azimuth in degrees.
        """
        self.x = np.array([alt, az, 0.0, 0.0])
        self.P = np.eye(self.state_dim) * 100.0
        self.last_update_time = time.time()
        self.is_initialized = True
        self.history.clear()
        self.residual_window.clear()

    def predict(self, dt: Optional[float] = None) -> Tuple[float, float]:
        """EKF prediction step: propagate state through the sidereal model.

        1. Compute Jacobian F at the current state.
        2. Propagate state through the non-linear transition f(x, dt).
        3. Propagate covariance: P = F @ P @ F^T + Q * dt.

        Args:
            dt: Time step in seconds.  If None, computed from wall-clock
                time since the last update.

        Returns:
            Predicted (alt, az) position in degrees.
        """
        if not self.is_initialized:
            return 0.0, 0.0

        current_time = time.time()
        if dt is None:
            if self.last_update_time is None:
                dt = 0.1
            else:
                dt = current_time - self.last_update_time

        # Clamp dt to avoid huge jumps after long pauses or negative dt
        dt = max(0.001, min(dt, 10.0))

        # Adapt process noise for zenith region
        self._adapt_Q_for_zenith(self.x[0])

        # --- Step 1: Compute Jacobian at current state ---
        F = self._compute_jacobian(self.x[0], self.x[1], dt)

        # --- Step 2: Non-linear state propagation ---
        sid_alt, sid_az = self._sidereal_rates(self.x[0], self.x[1])

        self.x[0] += (sid_alt + self.x[2]) * dt    # alt + (sidereal + drift) * dt
        self.x[1] += (sid_az  + self.x[3]) * dt    # az  + (sidereal + drift) * dt
        # x[2], x[3] unchanged (drift is a random walk)

        # Wrap azimuth to [0, 360)
        self.x[1] = self.x[1] % 360.0

        # --- Step 3: Covariance propagation via Jacobian ---
        self.P = F @ self.P @ F.T + self.Q * dt

        # Enforce symmetry (guard against numerical drift)
        self.P = 0.5 * (self.P + self.P.T)

        return self.x[0], self.x[1]

    def update(self, measured_alt: float, measured_az: float) -> Tuple[float, float]:
        """EKF update step: fuse a plate-solve measurement with the prediction.

        1. Collect pre-update innovation for adaptive R.
        2. Call predict() (non-linear propagation + Jacobian covariance).
        3. Compute innovation, Kalman gain, and correct state & covariance.

        On the very first call (before ``initialize()``), this auto-initialises
        the filter at the measured position.

        Args:
            measured_alt: Altitude from plate solving (degrees).
            measured_az:  Azimuth from plate solving (degrees).

        Returns:
            Filtered (alt, az) position in degrees.
        """
        if not self.is_initialized:
            self.initialize(measured_alt, measured_az)
            return measured_alt, measured_az

        # ---------------------------------------------------------------
        # Adaptive R: collect pre-update innovation
        # ---------------------------------------------------------------
        predicted = self.H @ self.x
        residual_pre = np.array([measured_alt - predicted[0],
                                 measured_az  - predicted[1]])
        residual_pre[1] = self._wrap_az_residual(residual_pre[1])
        self.residual_window.append(residual_pre)

        if len(self.residual_window) >= 5:
            residuals = np.array(list(self.residual_window))
            observed_variance = np.var(residuals, axis=0)
            # EMA blend to avoid abrupt R changes
            self.R = (self.R * (1.0 - self.adaptation_rate)
                      + np.diag(observed_variance) * self.adaptation_rate)
            # Floor R: prevent overconfidence
            self.R[0, 0] = max(self.R[0, 0], 1e-6)
            self.R[1, 1] = max(self.R[1, 1], 1e-6)

        # ---------------------------------------------------------------
        # EKF prediction (non-linear propagation + Jacobian covariance)
        # ---------------------------------------------------------------
        self.predict()

        # ---------------------------------------------------------------
        # Measurement update (standard Kalman equations)
        # ---------------------------------------------------------------
        z = np.array([measured_alt, measured_az])

        # Innovation (measurement residual)
        y = z - self.H @ self.x
        y[1] = self._wrap_az_residual(y[1])

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Analytical 2x2 inverse (avoids LAPACK overhead)
        det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        if abs(det) < 1e-15:
            return self.x[0], self.x[1]   # singular; skip update
        inv_det = 1.0 / det
        S_inv = np.array([
            [ S[1, 1] * inv_det, -S[0, 1] * inv_det],
            [-S[1, 0] * inv_det,  S[0, 0] * inv_det],
        ])

        # Kalman gain
        K = self.P @ self.H.T @ S_inv

        # State update
        self.x = self.x + K @ y

        # Wrap azimuth after update
        self.x[1] = self.x[1] % 360.0

        # Covariance update (Joseph form is more stable but the simple
        # form works fine for a 4-state system)
        self.P = (self._I - K @ self.H) @ self.P

        # Enforce symmetry
        self.P = 0.5 * (self.P + self.P.T)

        # Record for analysis
        self._record_history(measured_alt, measured_az)
        self.last_update_time = time.time()

        return self.x[0], self.x[1]

    def get_velocity(self) -> Tuple[float, float]:
        """Return the estimated drift velocity.

        In the EKF, x[2] and x[3] represent pure tracking drift
        (deviation from sidereal), NOT total velocity.  This is exactly
        the quantity the correction pipeline needs.

        Returns:
            (drift_alt, drift_az) in degrees/second.
        """
        return self.x[2], self.x[3]

    def get_drift_rate(self) -> float:
        """Return the total drift magnitude in arcseconds/second.

        Combines alt and az drift components via Euclidean norm.
        """
        d_alt, d_az = self.get_velocity()
        return math.sqrt(d_alt**2 + d_az**2) * 3600.0

    def get_correction(self) -> Tuple[float, float]:
        """Compute the correction needed to counteract predicted drift.

        Returns the negative of the drift over a 1-second horizon.

        Returns:
            (correction_alt, correction_az) in degrees.
        """
        if not self.is_initialized:
            return 0.0, 0.0
        d_alt, d_az = self.get_velocity()
        return -d_alt, -d_az

    def set_measurement_noise(self, noise_arcsec: float):
        """Configure measurement noise from plate-solve accuracy.

        Args:
            noise_arcsec: Estimated plate-solve RMS error in arcseconds
                         (typically 1-10 for ASTAP).
        """
        noise_deg = noise_arcsec / 3600.0
        self.R = np.array([
            [noise_deg**2, 0           ],
            [0,            noise_deg**2],
        ])

    def set_process_noise(self, position_noise: float, velocity_noise: float):
        """Configure process noise matrix Q.

        Also updates the base copy used for zenith Q-adaptation.

        Args:
            position_noise: Variance for position states (degrees^2).
            velocity_noise: Variance for drift states (degrees^2/s^2).
        """
        self.Q = np.array([
            [position_noise, 0, 0, 0],
            [0, position_noise, 0, 0],
            [0, 0, velocity_noise, 0],
            [0, 0, 0, velocity_noise],
        ])
        self._Q_base = self.Q.copy()

    def get_state(self) -> KalmanState:
        """Return a snapshot of the current EKF state.

        Returns:
            KalmanState dataclass with position, velocity, covariance,
            and timestamp.
        """
        return KalmanState(
            position=self.x[:2].copy(),
            velocity=self.x[2:4].copy(),
            covariance=self.P.copy(),
            timestamp=self.last_update_time or 0.0,
        )

    # ------------------------------------------------------------------
    # History & statistics
    # ------------------------------------------------------------------

    def _record_history(self, measured_alt: float, measured_az: float):
        """Store a measurement record for performance analysis.

        Uses deque with maxlen so old records are automatically discarded.
        """
        record = {
            'time': time.time(),
            'measured': (measured_alt, measured_az),
            'filtered': (self.x[0], self.x[1]),
            'velocity': (self.x[2], self.x[3]),
            'residual': (measured_alt - self.x[0], measured_az - self.x[1]),
        }
        self.history.append(record)

    def get_statistics(self) -> dict:
        """Compute filter performance statistics over the history buffer.

        Returns:
            Dict with RMS residuals (arcsec), mean drift rates (arcsec/s),
            and sample count.  Empty dict if fewer than 2 records.
        """
        if len(self.history) < 2:
            return {}

        hist_list = list(self.history)
        residuals  = np.array([h['residual']  for h in hist_list])
        velocities = np.array([h['velocity'] for h in hist_list])

        return {
            'rms_alt_arcsec':  float(np.std(residuals[:, 0]))  * 3600,
            'rms_az_arcsec':   float(np.std(residuals[:, 1]))  * 3600,
            'mean_drift_alt':  float(np.mean(velocities[:, 0])) * 3600,
            'mean_drift_az':   float(np.mean(velocities[:, 1])) * 3600,
            'samples':         len(hist_list),
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        """Reset the filter to uninitialised state.

        Called when tracking stops or the user requests a fresh start.
        All state, covariance, history, and adaptive R data are cleared.
        """
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 100.0
        np.copyto(self.Q, self._Q_base)
        self.last_update_time = None
        self.is_initialized = False
        self.history.clear()
        self.residual_window.clear()
