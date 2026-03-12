"""
Crash Recovery Manager -- Periodic state checkpointing and session restore.

This module provides automatic crash recovery for the telescope controller.
It periodically saves a snapshot of the application's volatile state to a
JSON checkpoint file. On startup, if a checkpoint exists without a clean
shutdown marker, the previous session's state can be restored.

Architecture role:
    Application (main_realtime.py)
        |
        v
    >>> CrashRecoveryManager <<<   <-- this module
        |
        v
    .telescope_checkpoint.json  (checkpoint file on disk)

How it works:
    1. **During a session**: ``save_checkpoint()`` is called every N seconds
       (default 30) by a tkinter ``after()`` timer. It writes the current
       telescope position, tracking state, Kalman filter parameters, and
       session statistics to a JSON file.

    2. **On clean shutdown**: ``mark_clean_shutdown()`` is called from
       ``_on_close()``. This writes ``"clean_shutdown": true`` into the
       checkpoint, so the next startup knows the session ended normally.

    3. **On next startup**: ``has_crash_checkpoint()`` checks whether a
       checkpoint file exists with ``"clean_shutdown": false`` (meaning the
       process was killed or crashed). If so, ``load_checkpoint()`` returns
       the saved state for the UI to restore.

    4. **After restore (or skip)**: ``clear_checkpoint()`` deletes the
       checkpoint file so it doesn't trigger again.

What is checkpointed:
    - Telescope position (Alt, Az, RA, Dec) and target coordinates
    - Tracking controller state (is_running, correction rates, PID accumulators)
    - Kalman filter state vector and covariance matrix
    - Session statistics (solve counts, correction counts, RMS)
    - Connection type and simulator state
    - Timestamp and session duration

What is NOT checkpointed (already persisted elsewhere):
    - User settings → ``telescope_config.json`` (via ConfigManager)
    - ML drift model weights → ``drift_model.json`` (via DriftPredictor)
    - PEC Fourier model → ``pec_model.json`` (via SoftwarePEC)

Thread safety:
    ``save_checkpoint()`` acquires no external locks -- it reads attributes
    that are either atomic (floats, bools) or protected by the tracking
    controller's internal lock. The checkpoint is written atomically via
    write-to-temp + rename to avoid corruption from mid-write crashes.

Dependencies:
    Python standard library only (json, os, time, tempfile).
"""

import json
import os
import time
import tempfile
from typing import Any, Dict, Optional

from telescope_logger import get_logger

_logger = get_logger(__name__)

# Default checkpoint file path (same directory as the application)
DEFAULT_CHECKPOINT_FILE = ".telescope_checkpoint.json"

# How often to save checkpoints (seconds)
# 60s is sufficient -- crash recovery doesn't need sub-minute granularity
# and reducing writes extends storage lifespan.
DEFAULT_CHECKPOINT_INTERVAL_SEC = 60


class CrashRecoveryManager:
    """Manages periodic state checkpointing and crash recovery.

    Usage::

        recovery = CrashRecoveryManager()

        # On startup: check for crash
        if recovery.has_crash_checkpoint():
            state = recovery.load_checkpoint()
            # ... offer to restore state ...
            recovery.clear_checkpoint()

        # During session: periodic saves (called from tkinter after loop)
        recovery.save_checkpoint(app_state_dict)

        # On clean shutdown:
        recovery.mark_clean_shutdown()

    Attributes:
        checkpoint_file: Path to the JSON checkpoint file.
        interval_sec: How often checkpoints are saved (seconds).
        last_save_time: Timestamp of the most recent checkpoint write.
        session_start_time: When the current session began.
    """

    def __init__(self, checkpoint_file: str = DEFAULT_CHECKPOINT_FILE,
                 interval_sec: float = DEFAULT_CHECKPOINT_INTERVAL_SEC):
        """Initialize the crash recovery manager.

        Args:
            checkpoint_file: Path to the checkpoint JSON file.
            interval_sec: Minimum interval between checkpoint writes.
        """
        self.checkpoint_file = checkpoint_file
        self.interval_sec = interval_sec
        self.last_save_time: float = 0.0
        self.session_start_time: float = time.time()

    # ------------------------------------------------------------------
    # Checkpoint detection (startup)
    # ------------------------------------------------------------------

    def has_crash_checkpoint(self) -> bool:
        """Check whether a crash checkpoint exists from a previous session.

        A crash checkpoint is a checkpoint file where ``clean_shutdown`` is
        False, indicating the application did not exit normally.

        Returns:
            True if a recoverable crash checkpoint exists.
        """
        if not os.path.exists(self.checkpoint_file):
            return False
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # If clean_shutdown is True, this was a normal exit -- no recovery needed
            if data.get("clean_shutdown", False):
                return False
            # Checkpoint must have a timestamp to be valid
            if "timestamp" not in data:
                return False
            # Don't offer recovery for very old checkpoints (>24 hours)
            age_sec = time.time() - data.get("timestamp", 0)
            if age_sec > 86400:
                _logger.info("Stale checkpoint (%.1f hours old) -- ignoring", age_sec / 3600)
                return False
            return True
        except (json.JSONDecodeError, OSError, KeyError) as e:
            _logger.warning("Corrupt checkpoint file: %s", e)
            return False

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the crash checkpoint data.

        Returns:
            The checkpoint dictionary, or None if loading fails.
        """
        if not os.path.exists(self.checkpoint_file):
            return None
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            _logger.info("Loaded crash checkpoint from %s (age: %.0fs)",
                         self.checkpoint_file,
                         time.time() - data.get("timestamp", 0))
            return data
        except (json.JSONDecodeError, OSError) as e:
            _logger.warning("Failed to load checkpoint: %s", e)
            return None

    def clear_checkpoint(self):
        """Delete the checkpoint file after restore or skip.

        Safe to call even if the file doesn't exist.
        """
        try:
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
                _logger.info("Checkpoint file cleared")
        except OSError as e:
            _logger.warning("Could not remove checkpoint file: %s", e)

    # ------------------------------------------------------------------
    # Checkpoint writing (during session)
    # ------------------------------------------------------------------

    def should_save(self) -> bool:
        """Check whether enough time has elapsed for the next checkpoint.

        Returns:
            True if at least ``interval_sec`` seconds have passed since
            the last save.
        """
        return (time.time() - self.last_save_time) >= self.interval_sec

    def save_checkpoint(self, state: Dict[str, Any]) -> bool:
        """Write the application state to the checkpoint file.

        The write is atomic: data is written to a temporary file first,
        then renamed over the checkpoint file. This prevents corruption
        if the process is killed mid-write.

        The ``clean_shutdown`` flag is always set to False in the checkpoint.
        It is only set to True by ``mark_clean_shutdown()``.

        Args:
            state: Dictionary of application state to checkpoint.

        Returns:
            True if the checkpoint was written successfully.
        """
        state["timestamp"] = time.time()
        state["clean_shutdown"] = False
        state["session_duration_sec"] = time.time() - self.session_start_time

        try:
            # Atomic write: temp file + rename
            dir_name = os.path.dirname(os.path.abspath(self.checkpoint_file))
            fd, tmp_path = tempfile.mkstemp(
                suffix=".tmp", prefix=".checkpoint_", dir=dir_name
            )
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(state, f, indent=2, default=_json_serializer)
                # On Windows, os.rename fails if the target exists -- remove first
                if os.path.exists(self.checkpoint_file):
                    os.remove(self.checkpoint_file)
                os.rename(tmp_path, self.checkpoint_file)
            except Exception:
                # Clean up temp file on error
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise

            self.last_save_time = time.time()
            return True

        except Exception as e:
            _logger.warning("Checkpoint save failed: %s", e)
            return False

    def mark_clean_shutdown(self) -> bool:
        """Mark the checkpoint as a clean shutdown.

        Called from ``_on_close()`` to indicate the session ended normally.
        On next startup, ``has_crash_checkpoint()`` will return False.

        Returns:
            True if the marker was written successfully.
        """
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data["clean_shutdown"] = True
                data["shutdown_time"] = time.time()
                with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=_json_serializer)
            else:
                # No checkpoint file -- write a minimal clean shutdown marker
                data = {
                    "clean_shutdown": True,
                    "shutdown_time": time.time(),
                    "timestamp": time.time(),
                }
                with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            _logger.info("Clean shutdown marked")
            return True
        except Exception as e:
            _logger.warning("Failed to mark clean shutdown: %s", e)
            return False


# ---------------------------------------------------------------------------
# State collection helpers
# ---------------------------------------------------------------------------

def collect_app_state(app) -> Dict[str, Any]:
    """Collect the current application state for checkpointing.

    Reads volatile state from the app's subsystems and packs it into a
    serializable dictionary.

    Args:
        app: The RealTimeTelescopeApp instance.

    Returns:
        Dictionary of checkpoint-worthy state.
    """
    state: Dict[str, Any] = {}

    # --- Telescope position ---
    try:
        state["position"] = {
            "alt_degrees": getattr(app.protocol, 'alt_degrees', 0.0),
            "az_degrees": getattr(app.protocol, 'az_degrees', 0.0),
            "ra_hours": getattr(app.protocol, 'ra_hours', 0.0),
            "dec_degrees": getattr(app.protocol, 'dec_degrees', 0.0),
            "target_alt": getattr(app.protocol, 'target_alt', 0.0),
            "target_az": getattr(app.protocol, 'target_az', 0.0),
            "is_slewing": getattr(app.protocol, 'is_slewing', False),
            "is_tracking": getattr(app.protocol, 'is_tracking', False),
            "latitude": getattr(app.protocol, 'latitude', 48.8566),
            "longitude": getattr(app.protocol, 'longitude', 2.3522),
        }
    except Exception:
        state["position"] = {}

    # --- Tracking controller ---
    try:
        t = app.tracking
        state["tracking"] = {
            "is_running": getattr(t, 'is_running', False),
            "current_alt": getattr(t, 'current_alt', 0.0),
            "current_az": getattr(t, 'current_az', 0.0),
            "current_ra": getattr(t, 'current_ra', 0.0),
            "current_dec": getattr(t, 'current_dec', 0.0),
            "error_integral_alt": getattr(t, 'error_integral_alt', 0.0),
            "error_integral_az": getattr(t, 'error_integral_az', 0.0),
            "last_error_alt": getattr(t, 'last_error_alt', 0.0),
            "last_error_az": getattr(t, 'last_error_az', 0.0),
            "total_corrections": t.stats.get('total_corrections', 0),
            "total_solves": t.stats.get('total_solves', 0),
            "successful_solves": t.stats.get('successful_solves', 0),
            "avg_solve_time": t.stats.get('avg_solve_time', 0.0),
            "avg_correction": t.stats.get('avg_correction', 0.0),
        }
    except Exception:
        state["tracking"] = {}

    # --- Kalman filter ---
    try:
        k = app.tracking.kalman
        if getattr(k, 'is_initialized', False):
            state["kalman"] = {
                "is_initialized": True,
                "state_vector": k.x.tolist() if hasattr(k.x, 'tolist') else list(k.x),
                "covariance": k.P.tolist() if hasattr(k.P, 'tolist') else [list(r) for r in k.P],
                "R_matrix": k.R.tolist() if hasattr(k.R, 'tolist') else [list(r) for r in k.R],
            }
        else:
            state["kalman"] = {"is_initialized": False}
    except Exception:
        state["kalman"] = {"is_initialized": False}

    # --- Connection ---
    try:
        state["connection"] = {
            "simulator_active": getattr(app, '_simulator_active', False),
            "bridge_connected": getattr(app.telescope_bridge, 'is_connected', False),
            "connection_type": app.connection_type_var.get(),
        }
    except Exception:
        state["connection"] = {}

    # --- Session stats ---
    try:
        state["session"] = {
            "tracking_was_running": getattr(app.tracking, 'is_running', False),
            "solving_was_active": getattr(app, 'is_solving', False),
            "auto_solve_mode": getattr(app, 'auto_solve_mode', 'none'),
            "pec_enabled": getattr(app.tracking, 'pec_enabled', True),
        }
    except Exception:
        state["session"] = {}

    return state


def restore_app_state(app, state: Dict[str, Any]) -> list:
    """Restore application state from a crash checkpoint.

    Applies the saved state to the app's subsystems. Does NOT restart
    tracking or reconnect hardware -- those are left to the user.

    Args:
        app: The RealTimeTelescopeApp instance.
        state: The checkpoint dictionary from ``load_checkpoint()``.

    Returns:
        List of human-readable strings describing what was restored.
    """
    restored = []

    # --- Position ---
    pos = state.get("position", {})
    if pos:
        try:
            app.protocol.alt_degrees = pos.get("alt_degrees", app.protocol.alt_degrees)
            app.protocol.az_degrees = pos.get("az_degrees", app.protocol.az_degrees)
            app.protocol.ra_hours = pos.get("ra_hours", app.protocol.ra_hours)
            app.protocol.dec_degrees = pos.get("dec_degrees", app.protocol.dec_degrees)
            app.protocol.target_alt = pos.get("target_alt", 0.0)
            app.protocol.target_az = pos.get("target_az", 0.0)
            alt = pos.get("alt_degrees", 0.0)
            az = pos.get("az_degrees", 0.0)
            restored.append(f"Position: Alt={alt:.2f} Az={az:.2f}")
        except Exception as e:
            _logger.warning("Position restore failed: %s", e)

    # --- Tracking state ---
    trk = state.get("tracking", {})
    if trk:
        try:
            app.tracking.current_alt = trk.get("current_alt", 0.0)
            app.tracking.current_az = trk.get("current_az", 0.0)
            app.tracking.current_ra = trk.get("current_ra", 0.0)
            app.tracking.current_dec = trk.get("current_dec", 0.0)
            app.tracking.error_integral_alt = trk.get("error_integral_alt", 0.0)
            app.tracking.error_integral_az = trk.get("error_integral_az", 0.0)
            app.tracking.last_error_alt = trk.get("last_error_alt", 0.0)
            app.tracking.last_error_az = trk.get("last_error_az", 0.0)
            app.tracking.stats['total_corrections'] = trk.get("total_corrections", 0)
            app.tracking.stats['total_solves'] = trk.get("total_solves", 0)
            app.tracking.stats['successful_solves'] = trk.get("successful_solves", 0)
            app.tracking.stats['avg_solve_time'] = trk.get("avg_solve_time", 0.0)
            app.tracking.stats['avg_correction'] = trk.get("avg_correction", 0.0)
            restored.append(f"Tracking stats: {trk.get('total_solves', 0)} solves, "
                            f"{trk.get('total_corrections', 0)} corrections")
        except Exception as e:
            _logger.warning("Tracking state restore failed: %s", e)

    # --- Kalman filter ---
    kal = state.get("kalman", {})
    if kal.get("is_initialized", False):
        try:
            import numpy as np
            k = app.tracking.kalman
            k.x = np.array(kal["state_vector"], dtype=float)
            k.P = np.array(kal["covariance"], dtype=float)
            k.R = np.array(kal["R_matrix"], dtype=float)
            k.is_initialized = True
            k.last_update_time = time.time()
            vel = k.x[2:4] * 3600  # deg/s -> arcsec/s
            restored.append(f"Kalman filter: velocity Alt={vel[0]:.1f}\"/s Az={vel[1]:.1f}\"/s")
        except Exception as e:
            _logger.warning("Kalman restore failed: %s", e)

    # --- Session flags (informational only -- don't auto-restart) ---
    sess = state.get("session", {})
    if sess.get("tracking_was_running", False):
        restored.append("Tracking was active (restart manually)")
    if sess.get("pec_enabled", True):
        try:
            app.tracking.pec_enabled = True
        except Exception:
            pass

    duration = state.get("session_duration_sec", 0)
    if duration > 0:
        mins = duration / 60.0
        restored.append(f"Previous session: {mins:.1f} minutes")

    return restored


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------

def _json_serializer(obj):
    """Custom JSON serializer for numpy arrays and other non-standard types.

    Args:
        obj: Object to serialize.

    Returns:
        JSON-compatible representation.

    Raises:
        TypeError: If the object type is not handled.
    """
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
