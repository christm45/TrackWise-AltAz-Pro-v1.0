"""
Machine Learning Drift Predictor for Telescope Tracking.

Architecture Role:
    This module provides position-dependent drift prediction using online
    linear regression. It is one of the correction sources in the fusion pipeline.

    Data Flow:
        plate_solve residuals --> DriftPredictor.add_sample() --> model learns
        current (Alt, Az)     --> DriftPredictor.predict()    --> predicted drift
                              --> weighted at 40-55% in realtime_tracking.py

    The predictor learns a mapping from telescope position (Alt, Az) to expected
    drift rates using polynomial + trigonometric features and Stochastic Gradient
    Descent (SGD). This captures systematic drift patterns caused by:
      - Mechanical imperfections that vary with pointing direction
      - Gravitational flexure (altitude-dependent)
      - Azimuthal bearing irregularities (periodic in azimuth)

Classes:
    DriftSample            -- Dataclass for one observed drift measurement.
    DriftPredictor         -- Online SGD linear regression with polynomial features.
    PeriodicErrorCorrector -- LEGACY: Simple position-binning PEC (superseded by
                             software_pec.py which uses FFT-based frequency detection).
                             Kept for backward compatibility but NOT integrated into
                             the active correction pipeline.

Dependencies:
    - numpy: Feature computation and linear algebra.
    - Used by: realtime_tracking.py (RealTimeTrackingController).

Model Persistence:
    The model weights are saved to / loaded from 'drift_model.json' so that
    learned drift patterns survive across application restarts.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import time
import json
import os

from telescope_logger import get_logger

_logger = get_logger(__name__)


@dataclass
class DriftSample:
    """A single observed drift measurement at a specific sky position.

    Attributes:
        timestamp:  Unix time when the sample was recorded.
        alt:        Altitude in degrees at the time of measurement.
        az:         Azimuth in degrees at the time of measurement.
        drift_alt:  Observed altitude drift rate (degrees/second).
        drift_az:   Observed azimuth drift rate (degrees/second).
        temperature: Ambient temperature in Celsius (optional, for future use).
    """
    timestamp: float
    alt: float          # Altitude in degrees
    az: float           # Azimuth in degrees
    drift_alt: float    # Drift in altitude (degrees/sec)
    drift_az: float     # Drift in azimuth (degrees/sec)
    temperature: float = 20.0  # Ambient temperature (optional)


class DriftPredictor:
    """Online linear regression model for telescope drift prediction.

    The model predicts drift rates (degrees/second) as a function of the
    telescope's current Alt/Az position. It uses 12 features:

        [1, alt_norm, az_norm, alt_norm^2, az_norm^2, alt*az,
         sin(az), cos(az),
         cos(alt_rad), sin(alt_rad)*cos(az_rad), sin(alt_rad)*sin(az_rad),
         cos(alt_rad)*dT]

    Feature rationale:
        - Bias (1.0): Captures constant systematic drift.
        - alt_norm, az_norm: Linear position dependence.
        - Quadratic terms: Captures nonlinear effects like gravitational flexure
          which increases roughly as sin(zenith_angle) ~ cos(alt).
        - Cross term (alt*az): Captures interaction effects.
        - sin(az), cos(az): Captures periodic azimuthal errors from bearings/gears.
        - cos(alt_rad): Explicit gravitational flexure term -- gravity perpendicular
          to the tube is proportional to cos(altitude).
        - sin(alt)*cos(az), sin(alt)*sin(az): Captures altitude-azimuth coupling
          in tube flexure due to asymmetric mass distribution (focuser, camera).
        - cos(alt)*dT: Thermal-gravity interaction -- flexure magnitude changes
          with temperature (tube stiffness, material expansion).

    Learning:
        Uses SGD (Stochastic Gradient Descent) with L2 regularization for online
        updates. Each new drift sample triggers a single SGD step, making the model
        continuously adaptive. Batch training is also available via train_batch().

    Normalization:
        Alt and Az are z-score normalized using fixed statistics (mean=45/180,
        std=30/100) to keep feature magnitudes balanced.

    Typical usage:
        predictor = DriftPredictor()
        # Feed observed drift samples:
        predictor.add_sample(alt=45, az=180, drift_alt=0.001, drift_az=-0.002)
        # Get drift prediction for correction:
        d_alt, d_az = predictor.predict(alt=46, az=181)
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the drift predictor.

        Args:
            model_path: File path for saving/loading the model (default: 'drift_model.json').
        """
        self.model_path = model_path or "drift_model.json"

        # Training data buffer (capped to prevent unbounded memory growth).
        # Reduced from 10000 for lower memory usage on constrained devices.
        self.samples: List[DriftSample] = []
        self.max_samples = 5000

        # Linear regression weights: one set per axis (alt and az)
        # Features: [1, alt, az, alt^2, az^2, alt*az, sin(az), cos(az),
        #            cos(alt_rad), sin(alt)*cos(az), sin(alt)*sin(az),
        #            cos(alt_rad)*dT]
        self.n_features = 12
        self.weights_alt = np.zeros(self.n_features)
        self.weights_az = np.zeros(self.n_features)

        # Temperature tracking for flexure features
        self._current_temperature: float = 20.0
        self._base_temperature: float = 20.0

        # SGD hyperparameters
        self.learning_rate = 0.01       # Step size for weight updates
        self.regularization = 0.001     # L2 penalty to prevent overfitting

        # Performance tracking
        self.total_predictions = 0
        self.total_error = 0.0

        # --- Output EMA smoothing ---
        # Smooths the raw dot-product predictions to prevent jumps when the
        # telescope crosses sky regions with different learned weights.
        self._ml_smooth_alpha = 0.2      # EMA blending factor for predictions
        self._smooth_alt = 0.0           # Smoothed alt prediction
        self._smooth_az = 0.0            # Smoothed az prediction

        # Normalization statistics (fixed, not learned from data)
        # alt ~ N(45, 30): typical Alt range is 15-75 degrees
        # az  ~ N(180, 100): typical Az range is 0-360 degrees
        self.mean_alt = 45.0
        self.std_alt = 30.0
        self.mean_az = 180.0
        self.std_az = 100.0

        # Load previously saved model weights (if any)
        self._load_model()

    def update_temperature(self, temperature: float,
                           base_temperature: Optional[float] = None):
        """Update ambient temperature for flexure feature computation.

        Args:
            temperature:      Current ambient temperature in Celsius.
            base_temperature: Calibration baseline temperature. If None,
                              the first temperature reading becomes the baseline.
        """
        self._current_temperature = temperature
        if base_temperature is not None:
            self._base_temperature = base_temperature
        elif self._base_temperature == 20.0 and temperature != 20.0:
            # Auto-set baseline on first real reading
            self._base_temperature = temperature

    def add_sample(self, alt: float, az: float,
                   drift_alt: float, drift_az: float,
                   temperature: float = 20.0):
        """Record an observed drift sample and update the model.

        Each call triggers a single SGD update step, so the model learns
        incrementally as new data arrives.

        Args:
            alt:         Current altitude in degrees.
            az:          Current azimuth in degrees.
            drift_alt:   Observed altitude drift rate (degrees/second).
            drift_az:    Observed azimuth drift rate (degrees/second).
            temperature: Ambient temperature in Celsius (used in flexure features).
        """
        self._current_temperature = temperature
        sample = DriftSample(
            timestamp=time.time(),
            alt=alt,
            az=az,
            drift_alt=drift_alt,
            drift_az=drift_az,
            temperature=temperature
        )

        self.samples.append(sample)

        # Cap buffer size to prevent unbounded memory growth
        if len(self.samples) > self.max_samples:
            self.samples = self.samples[-self.max_samples:]

        # Online SGD update with this new sample
        self._update_online(sample)

    def predict(self, alt: float, az: float) -> Tuple[float, float]:
        """Predict drift rates for a given sky position.

        Args:
            alt: Altitude in degrees.
            az:  Azimuth in degrees.

        Returns:
            (drift_alt, drift_az) EMA-smoothed predicted drift in degrees/second.
        """
        features = self._compute_features(alt, az)

        raw_alt = float(np.dot(self.weights_alt, features))
        raw_az = float(np.dot(self.weights_az, features))

        # Apply EMA smoothing to prevent jumps when crossing sky regions
        # with different learned weight contributions.
        a = self._ml_smooth_alpha
        self._smooth_alt = a * raw_alt + (1 - a) * self._smooth_alt
        self._smooth_az = a * raw_az + (1 - a) * self._smooth_az

        self.total_predictions += 1

        return self._smooth_alt, self._smooth_az

    def get_correction(self, alt: float, az: float,
                       lookahead: float = 1.0) -> Tuple[float, float]:
        """Compute the correction to counteract predicted drift.

        The correction is the negative of predicted drift scaled by the
        lookahead time horizon.

        Args:
            alt:       Current altitude in degrees.
            az:        Current azimuth in degrees.
            lookahead: Prediction horizon in seconds (default 1.0).

        Returns:
            (correction_alt, correction_az) in degrees.
        """
        drift_alt, drift_az = self.predict(alt, az)

        # Correction opposes the predicted drift
        return -drift_alt * lookahead, -drift_az * lookahead

    def _compute_features(self, alt: float, az: float) -> np.ndarray:
        """Compute the feature vector for regression.

        Returns 12 features that capture linear, quadratic, interaction,
        periodic (trigonometric), and gravitational flexure relationships
        between position, temperature, and drift.

        Args:
            alt: Altitude in degrees.
            az:  Azimuth in degrees.

        Returns:
            numpy array of shape (12,).
        """
        # Z-score normalization to keep features on similar scales
        alt_norm = (alt - self.mean_alt) / self.std_alt
        az_norm = (az - self.mean_az) / self.std_az

        alt_rad = np.radians(max(1.0, alt))  # clamp to avoid singularity at 0
        az_rad = np.radians(az)

        cos_alt = np.cos(alt_rad)
        sin_alt = np.sin(alt_rad)

        # Temperature delta from calibration baseline
        delta_temp = (self._current_temperature - self._base_temperature) / 10.0  # normalise

        return np.array([
            1.0,                    # [0] Bias term (constant offset)
            alt_norm,               # [1] Linear altitude
            az_norm,                # [2] Linear azimuth
            alt_norm ** 2,          # [3] Quadratic alt (captures flexure curvature)
            az_norm ** 2,           # [4] Quadratic az
            alt_norm * az_norm,     # [5] Interaction (cross-coupling effects)
            np.sin(az_rad),         # [6] Periodic azimuthal component (sine)
            np.cos(az_rad),         # [7] Periodic azimuthal component (cosine)
            # --- Explicit flexure features ---
            cos_alt,                # [8] Gravitational flexure: max at horizon, zero at zenith
            sin_alt * np.cos(az_rad),  # [9] Altitude-azimuth flexure coupling (N-S asymmetry)
            sin_alt * np.sin(az_rad),  # [10] Altitude-azimuth flexure coupling (E-W asymmetry)
            cos_alt * delta_temp,   # [11] Thermal-gravity interaction: flexure changes with temp
        ])

    def _update_online(self, sample: DriftSample):
        """Perform one SGD step using a single sample.

        Updates both weight vectors (alt and az) using the gradient of the
        squared error loss with L2 regularization:
            loss = (target - prediction)^2 + lambda * ||weights||^2

        Args:
            sample: The DriftSample to learn from.
        """
        features = self._compute_features(sample.alt, sample.az)

        # Current predictions
        pred_alt = np.dot(self.weights_alt, features)
        pred_az = np.dot(self.weights_az, features)

        # Prediction errors
        error_alt = sample.drift_alt - pred_alt
        error_az = sample.drift_az - pred_az

        # SGD weight update with L2 regularization
        self.weights_alt += self.learning_rate * (
            error_alt * features - self.regularization * self.weights_alt
        )
        self.weights_az += self.learning_rate * (
            error_az * features - self.regularization * self.weights_az
        )

        # Track cumulative error for diagnostics
        self.total_error += abs(error_alt) + abs(error_az)

    def train_batch(self, epochs: int = 100):
        """Batch training over all stored samples.

        Useful after collecting a large number of samples (e.g., after a long
        observation session) to fully optimize the model weights.

        Performs multiple passes (epochs) over randomly shuffled data.
        Requires at least 10 samples to be meaningful.

        Args:
            epochs: Number of full passes over the dataset.
        """
        if len(self.samples) < 10:
            return

        # Pre-compute feature matrix for all samples
        X = np.array([
            self._compute_features(s.alt, s.az) for s in self.samples
        ])
        y_alt = np.array([s.drift_alt for s in self.samples])
        y_az = np.array([s.drift_az for s in self.samples])

        for epoch in range(epochs):
            # Random shuffle for stochastic updates
            indices = np.random.permutation(len(self.samples))
            X_shuffled = X[indices]
            y_alt_shuffled = y_alt[indices]
            y_az_shuffled = y_az[indices]

            for i in range(len(self.samples)):
                pred_alt = np.dot(self.weights_alt, X_shuffled[i])
                pred_az = np.dot(self.weights_az, X_shuffled[i])

                error_alt = y_alt_shuffled[i] - pred_alt
                error_az = y_az_shuffled[i] - pred_az

                self.weights_alt += self.learning_rate * (
                    error_alt * X_shuffled[i] - self.regularization * self.weights_alt
                )
                self.weights_az += self.learning_rate * (
                    error_az * X_shuffled[i] - self.regularization * self.weights_az
                )

    def get_statistics(self) -> dict:
        """Return model performance statistics.

        Vectorized implementation: builds feature matrix for all recent samples
        at once and computes predictions via a single matrix-vector multiply
        (avoids the slow per-sample Python loop on mobile devices).

        Returns:
            Dict with keys:
                samples            -- Number of training samples collected.
                model_ready        -- True if >= 10 samples (minimum for meaningful predictions).
                mean_error_arcsec  -- Mean prediction error on recent samples (arcsec).
                total_predictions  -- Total number of predictions made.
                weights_alt        -- Current alt model weights (list).
                weights_az         -- Current az model weights (list).
        """
        if len(self.samples) == 0:
            return {'samples': 0, 'model_ready': False}

        # Evaluate on the 100 most recent samples -- vectorized
        recent = self.samples[-100:]
        # Build feature matrix (N x 8) in one pass
        X = np.array([self._compute_features(s.alt, s.az) for s in recent])
        actual_alt = np.array([s.drift_alt for s in recent])
        actual_az = np.array([s.drift_az for s in recent])

        # Vectorized prediction: (N x 8) @ (8,) -> (N,)
        pred_alt = X @ self.weights_alt
        pred_az = X @ self.weights_az

        # Vectorized error computation
        errors = np.sqrt((actual_alt - pred_alt) ** 2 + (actual_az - pred_az) ** 2)

        return {
            'samples': len(self.samples),
            'model_ready': len(self.samples) >= 10,
            'mean_error_arcsec': float(np.mean(errors)) * 3600,
            'total_predictions': self.total_predictions,
            'weights_alt': self.weights_alt.tolist(),
            'weights_az': self.weights_az.tolist()
        }

    def save_model(self):
        """Persist model weights to disk as JSON.

        Saves weights, normalization parameters, and metadata so the model
        can be reloaded across sessions without retraining from scratch.
        """
        data = {
            'weights_alt': self.weights_alt.tolist(),
            'weights_az': self.weights_az.tolist(),
            'mean_alt': self.mean_alt,
            'std_alt': self.std_alt,
            'mean_az': self.mean_az,
            'std_az': self.std_az,
            'n_samples': len(self.samples),
            'n_features': self.n_features,
            'base_temperature': self._base_temperature,
            'timestamp': time.time()
        }

        try:
            with open(self.model_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            _logger.error("Error saving drift model: %s", e)

    def _load_model(self):
        """Load previously saved model weights from disk.

        Called automatically during __init__. If no saved model exists,
        the predictor starts with zero weights (no prior knowledge).

        Handles backward compatibility: old models with 8 features are
        zero-padded to 12 features (the 4 new flexure features start
        at zero, meaning no flexure correction initially -- they will
        learn from new data).
        """
        if not os.path.exists(self.model_path):
            return

        try:
            with open(self.model_path, 'r') as f:
                data = json.load(f)

            loaded_alt = np.array(data['weights_alt'])
            loaded_az = np.array(data['weights_az'])

            # Backward compatibility: zero-pad old 8-feature models to 12
            if len(loaded_alt) < self.n_features:
                padded_alt = np.zeros(self.n_features)
                padded_alt[:len(loaded_alt)] = loaded_alt
                loaded_alt = padded_alt
            if len(loaded_az) < self.n_features:
                padded_az = np.zeros(self.n_features)
                padded_az[:len(loaded_az)] = loaded_az
                loaded_az = padded_az

            self.weights_alt = loaded_alt[:self.n_features]
            self.weights_az = loaded_az[:self.n_features]
            self.mean_alt = data.get('mean_alt', 45.0)
            self.std_alt = data.get('std_alt', 30.0)
            self.mean_az = data.get('mean_az', 180.0)
            self.std_az = data.get('std_az', 100.0)
            self._base_temperature = data.get('base_temperature', 20.0)

        except Exception as e:
            _logger.error("Error loading drift model: %s", e)

    def reset(self):
        """Reset the model to untrained state.

        Clears all samples, resets weights to zero, and resets counters.
        Does NOT delete the saved model file on disk.
        """
        self.samples = []
        self.weights_alt = np.zeros(self.n_features)
        self.weights_az = np.zeros(self.n_features)
        self.total_predictions = 0
        self.total_error = 0.0
        self._current_temperature = 20.0
        self._base_temperature = 20.0


class PeriodicErrorCorrector:
    """LEGACY: Simple position-binning Periodic Error Corrector.

    WARNING: This class is SUPERSEDED by SoftwarePEC in software_pec.py,
    which uses FFT-based frequency detection and Fourier series fitting.
    This class is retained for backward compatibility but is NOT integrated
    into the active correction pipeline.

    Approach:
        Divides the 360-degree position space into n_bins and maintains a
        lookup table of average errors per bin. Uses exponential moving average
        to smooth updates. Interpolates between adjacent bins for output.

    Limitations vs SoftwarePEC:
        - Assumes a fixed 360-degree period (cannot detect other periodicities).
        - No frequency detection (cannot find gear-specific periods automatically).
        - Requires full 360-degree coverage to be effective.
        - No multi-harmonic fitting capability.

    For Dobson Alt-Az mounts, the relevant periodic errors come from:
        - Gear tooth errors in the planetary gearbox
        - Belt pitch errors in GT2 belt drives
        - For Skywatcher mounts: worm gear periodic error (~8 min period)
    """

    def __init__(self, period_alt: float = 360.0, period_az: float = 360.0,
                 n_bins: int = 360):
        """Initialize the legacy PEC corrector.

        Args:
            period_alt: Period in altitude (degrees). Default 360 = full rotation.
            period_az:  Period in azimuth (degrees). Default 360 = full rotation.
            n_bins:     Number of lookup table bins per axis.
        """
        self.period_alt = period_alt
        self.period_az = period_az
        self.n_bins = n_bins

        # Correction lookup tables (one per axis)
        self.pec_table_alt = np.zeros(n_bins)
        self.pec_table_az = np.zeros(n_bins)
        self.sample_count_alt = np.zeros(n_bins)
        self.sample_count_az = np.zeros(n_bins)

        # EMA smoothing rate for bin updates
        self.alpha = 0.1

    def add_error(self, alt: float, az: float,
                  error_alt: float, error_az: float):
        """Record an observed error for learning.

        Maps the current position to a bin index and updates the bin value
        using an adaptive exponential moving average (alpha decreases with
        more samples in the bin for stability).

        Args:
            alt:       Current altitude in degrees.
            az:        Current azimuth in degrees.
            error_alt: Observed altitude error (degrees).
            error_az:  Observed azimuth error (degrees).
        """
        # Map position to bin index
        idx_alt = int((alt % self.period_alt) / self.period_alt * self.n_bins) % self.n_bins
        idx_az = int((az % self.period_az) / self.period_az * self.n_bins) % self.n_bins

        # Increment sample counts
        self.sample_count_alt[idx_alt] += 1
        self.sample_count_az[idx_az] += 1

        n_alt = self.sample_count_alt[idx_alt]
        n_az = self.sample_count_az[idx_az]

        # Adaptive alpha: starts at 1/n (simple mean) then converges to self.alpha
        # This gives more weight to early samples for fast convergence
        alpha_alt = min(self.alpha, 1.0 / n_alt)
        alpha_az = min(self.alpha, 1.0 / n_az)

        # Update bins with EMA
        self.pec_table_alt[idx_alt] = (
            (1 - alpha_alt) * self.pec_table_alt[idx_alt] +
            alpha_alt * error_alt
        )
        self.pec_table_az[idx_az] = (
            (1 - alpha_az) * self.pec_table_az[idx_az] +
            alpha_az * error_az
        )

    def get_correction(self, alt: float, az: float) -> Tuple[float, float]:
        """Return the PEC correction for a given position.

        Uses linear interpolation between adjacent bins for smoother output.
        The correction is the negative of the learned error pattern.

        Args:
            alt: Current altitude in degrees.
            az:  Current azimuth in degrees.

        Returns:
            (correction_alt, correction_az) in degrees.
        """
        idx_alt = int((alt % self.period_alt) / self.period_alt * self.n_bins) % self.n_bins
        idx_az = int((az % self.period_az) / self.period_az * self.n_bins) % self.n_bins

        # Fractional position within the bin (for linear interpolation)
        frac_alt = (alt % self.period_alt) / self.period_alt * self.n_bins - idx_alt
        frac_az = (az % self.period_az) / self.period_az * self.n_bins - idx_az

        next_idx_alt = (idx_alt + 1) % self.n_bins
        next_idx_az = (idx_az + 1) % self.n_bins

        # Linear interpolation between adjacent bins
        corr_alt = (1 - frac_alt) * self.pec_table_alt[idx_alt] + \
                   frac_alt * self.pec_table_alt[next_idx_alt]
        corr_az = (1 - frac_az) * self.pec_table_az[idx_az] + \
                  frac_az * self.pec_table_az[next_idx_az]

        # Correction opposes the learned error
        return -corr_alt, -corr_az

    def is_trained(self) -> bool:
        """Check if enough bins have data for meaningful corrections.

        Returns True when at least 50% of bins in both axes have at least
        one sample. Full coverage is needed for reliable PEC output.
        """
        min_coverage = self.n_bins * 0.5  # At least 50% of bins
        alt_ok = int(np.sum(self.sample_count_alt > 0)) > min_coverage
        az_ok = int(np.sum(self.sample_count_az > 0)) > min_coverage
        return alt_ok and az_ok
