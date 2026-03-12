"""Session data recorder and exporter for TrackWise-AltAzPro.

Collects tracking telemetry, plate-solve results, logs, and
configuration snapshots throughout an observing session and writes
them to a timestamped folder as CSV + JSON + log files.

Output structure (one folder per save)::

    sessions/
      2026-02-26_22-15-00/
        session_summary.json   -- compact overview
        telemetry.csv          -- correction history time-series
        solves.csv             -- plate-solve result history
        session.log            -- text log of the session

The module is intentionally self-contained with no heavy dependencies
beyond the Python standard library.
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
import logging
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # avoid circular imports at runtime

logger = logging.getLogger("TelescopeApp.session")

# Default base folder (relative to project root)
_DEFAULT_SESSION_DIR = "sessions"


class SessionRecorder:
    """Collects and exports observing session data.

    Typical usage::

        recorder = SessionRecorder(app)
        recorder.start()          # mark session start
        ...                        # observing happens
        folder = recorder.save()  # export everything
        recorder.stop()           # mark session end

    The ``save()`` method can be called multiple times during a session
    (e.g. on tracking stop and again on shutdown) — each call produces
    a new timestamped subfolder.

    Args:
        app: The ``HeadlessTelescopeApp`` instance (or any object
            exposing ``tracking``, ``auto_solver``, ``web_server``,
            ``config_manager``, ``telescope_bridge``, and the various
            ``*_var`` attributes).
        base_dir: Root folder for session exports.  Defaults to
            ``sessions/`` in the working directory.
    """

    def __init__(self, app: Any, base_dir: str = _DEFAULT_SESSION_DIR):
        self.app = app
        self.base_dir = Path(base_dir)
        self.session_start: float = 0.0
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Mark the beginning of a session."""
        self.session_start = time.time()
        self._started = True
        logger.info("Session recording started")

    def stop(self) -> None:
        """Mark the end of a session (does NOT auto-save)."""
        self._started = False
        logger.info("Session recording stopped")

    @property
    def is_started(self) -> bool:
        return self._started

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save(self, auto: bool = False) -> Optional[str]:
        """Export all session data to a timestamped folder.

        Args:
            auto: If ``True`` this is an automatic save (e.g. on
                tracking stop) — noted in the summary.

        Returns:
            The absolute path of the created folder, or ``None`` on
            failure.
        """
        now = time.time()
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = self.base_dir / stamp
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("Cannot create session folder %s: %s", folder, exc)
            return None

        saved: List[str] = []
        try:
            self._write_telemetry_csv(folder)
            saved.append("telemetry.csv")
        except Exception as exc:
            logger.error("Error writing telemetry CSV: %s", exc)

        try:
            self._write_solves_csv(folder)
            saved.append("solves.csv")
        except Exception as exc:
            logger.error("Error writing solves CSV: %s", exc)

        try:
            self._write_session_log(folder)
            saved.append("session.log")
        except Exception as exc:
            logger.error("Error writing session log: %s", exc)

        try:
            summary = self._build_summary(now, auto, saved)
            with open(folder / "session_summary.json", "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2, default=str)
            saved.append("session_summary.json")
        except Exception as exc:
            logger.error("Error writing session summary: %s", exc)

        abs_path = str(folder.resolve())
        logger.info("Session saved to %s  (%s)", abs_path, ", ".join(saved))
        return abs_path

    # ------------------------------------------------------------------
    # Internal writers
    # ------------------------------------------------------------------

    def _write_telemetry_csv(self, folder: Path) -> None:
        """Write correction_history from the tracking controller."""
        tracking = getattr(self.app, "tracking", None)
        if not tracking:
            return
        records = tracking.get_correction_history()
        if not records:
            return

        # Use dataclass field names as CSV header
        from realtime_tracking import CorrectionRecord
        header = [f.name for f in fields(CorrectionRecord)]

        with open(folder / "telemetry.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            writer.writeheader()
            for rec in records:
                writer.writerow(asdict(rec))

    def _write_solves_csv(self, folder: Path) -> None:
        """Write plate-solve history from auto_solver + tracking position_history."""
        rows: List[Dict[str, Any]] = []

        # Source 1: auto_solver.solve_history (SolveResult with RA/Dec/time)
        auto_solver = getattr(self.app, "auto_solver", None)
        if auto_solver:
            for sr in getattr(auto_solver, "solve_history", []):
                rows.append({
                    "timestamp": getattr(sr, "timestamp", ""),
                    "ra_hours": sr.ra_hours,
                    "dec_degrees": sr.dec_degrees,
                    "solve_time_ms": sr.solve_time_ms,
                    "success": sr.success,
                    "error": sr.error,
                })

        # Source 2: tracking.position_history (PositionSample with Alt/Az)
        tracking = getattr(self.app, "tracking", None)
        if tracking:
            for ps in getattr(tracking, "position_history", []):
                rows.append({
                    "timestamp": ps.timestamp,
                    "ra_hours": ps.ra_hours,
                    "dec_degrees": ps.dec_degrees,
                    "alt_degrees": ps.alt_degrees,
                    "az_degrees": ps.az_degrees,
                    "solve_time_ms": ps.solve_time_ms,
                    "is_valid": ps.is_valid,
                })

        if not rows:
            return

        # Merge all possible keys
        all_keys = list(dict.fromkeys(
            k for row in rows for k in row.keys()
        ))

        with open(folder / "solves.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            for row in sorted(rows, key=lambda r: r.get("timestamp", 0)):
                writer.writerow(row)

    def _write_session_log(self, folder: Path) -> None:
        """Dump the web-server log buffer to a text file."""
        ws = getattr(self.app, "web_server", None)
        if not ws:
            return
        buf = getattr(ws, "_log_buffer", None)
        if not buf:
            return

        with open(folder / "session.log", "w", encoding="utf-8") as fh:
            for entry in buf:
                ts = entry.get("ts", 0)
                msg = entry.get("msg", "")
                tag = entry.get("tag", "")
                dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "???"
                fh.write(f"[{dt}] [{tag:>7s}] {msg}\n")

    # ------------------------------------------------------------------
    # Summary builder
    # ------------------------------------------------------------------

    def _build_summary(self, now: float, auto: bool,
                       saved_files: List[str]) -> Dict[str, Any]:
        """Build a JSON-serializable session summary dict."""
        duration = now - self.session_start if self.session_start else 0
        summary: Dict[str, Any] = {
            "session_start": self.session_start,
            "session_start_utc": _epoch_to_iso(self.session_start),
            "session_end": now,
            "session_end_utc": _epoch_to_iso(now),
            "duration_seconds": round(duration, 1),
            "duration_human": _fmt_duration(duration),
            "auto_save": auto,
            "files": saved_files,
        }

        # Tracking stats
        tracking = getattr(self.app, "tracking", None)
        if tracking:
            summary["tracking"] = tracking.get_stats()
            summary["tracking"]["correction_history_length"] = len(
                getattr(tracking, "correction_history", [])
            )
            summary["tracking"]["position_history_length"] = len(
                getattr(tracking, "position_history", [])
            )

        # Plate-solve stats
        auto_solver = getattr(self.app, "auto_solver", None)
        if auto_solver:
            summary["plate_solve"] = auto_solver.get_statistics()

        # Weather snapshot
        weather_data: Dict[str, str] = {}
        for attr in ("weather_temp_var", "weather_humidity_var",
                      "weather_cloud_var", "weather_wind_var",
                      "weather_conditions_var", "weather_observing_var",
                      "weather_dew_risk_var", "weather_location_var"):
            var = getattr(self.app, attr, None)
            if var:
                try:
                    weather_data[attr.replace("weather_", "").replace("_var", "")] = var.get()
                except Exception:
                    pass
        if weather_data:
            summary["weather"] = weather_data

        # Configuration snapshot
        cfg = getattr(self.app, "config_manager", None)
        if cfg:
            summary["config"] = getattr(cfg, "config", {})

        # Connection info
        summary["connection"] = {
            "type": _safe_get(self.app, "connection_type_var"),
            "port": _safe_get(self.app, "usb_port_var"),
            "baudrate": _safe_get(self.app, "usb_baudrate_var"),
            "wifi_ip": _safe_get(self.app, "wifi_ip_var"),
            "wifi_port": _safe_get(self.app, "wifi_port_var"),
            "connected": _safe_get(self.app, "usb_connected_var"),
        }

        # Location
        summary["location"] = {
            "latitude": _safe_get(self.app, "lat_var"),
            "longitude": _safe_get(self.app, "lon_var"),
        }

        return summary


# ======================================================================
# Helpers
# ======================================================================

def _safe_get(app: Any, attr: str) -> Any:
    """Read a HeadlessVar's value, returning '' on any error."""
    var = getattr(app, attr, None)
    if var is None:
        return ""
    try:
        return var.get()
    except Exception:
        return ""


def _epoch_to_iso(ts: float) -> str:
    """Convert a Unix timestamp to ISO-8601 UTC string."""
    if not ts:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _fmt_duration(seconds: float) -> str:
    """Format seconds as 'Xh Ym Zs'."""
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)
