"""
Remote-access web server for the Telescope Controller.

Runs a Flask HTTP server in a background thread, exposing a REST API
and a mobile-friendly web UI for LAN control of the telescope.  The
server never touches tkinter directly -- it reads from a thread-safe
state snapshot updated by the main 10 Hz loop, and schedules commands
on the tkinter main thread via ``root.after()``.

Architecture
------------
* ``TelescopeWebServer`` owns a ``Flask`` app and a ``werkzeug`` server.
* ``update_state()`` is called every ~500 ms from the main thread to
  refresh the snapshot that API endpoints return.
* ``execute_command(callback)`` posts a callable to ``root.after(0, ...)``.
* A bounded ``collections.deque`` buffers the last 200 log lines for
  the ``/api/log`` endpoint.

Typical usage (inside ``main_realtime.py``)::

    from web_server import TelescopeWebServer

    self.web_server = TelescopeWebServer(self, host='0.0.0.0', port=8080)
    self.web_server.start()
    # In the 10 Hz update loop:
    self.web_server.update_state()
    # On shutdown:
    self.web_server.stop()
"""

from __future__ import annotations

import collections
import logging
import math
import os
import threading
import time
from typing import Any, Callable, Deque, Dict, Optional

from flask import Flask, Response, jsonify, request, send_file

from telescope_logger import get_logger

# Optional OpenCV for camera streaming (may not be installed on all systems)
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# Optional numpy for ASCOM frame processing (16-bit -> 8-bit stretch)
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# Optional ASCOM camera support (Windows only, requires pywin32 + ASCOM Platform).
# On Android the ASCOMCameraCapture class is importable (the stub modules prevent
# ImportError) but COM dispatch will never work, so force _HAS_ASCOM to False.
_IS_ANDROID = os.environ.get("TELESCOPE_PLATFORM") == "android"
try:
    from auto_platesolve import ASCOMCameraCapture
    _HAS_ASCOM = not _IS_ANDROID  # ASCOM is Windows-only
except ImportError:
    _HAS_ASCOM = False

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Standalone RA/Dec -> Alt/Az conversion (used by catalog endpoints)
# ---------------------------------------------------------------------------
# This is the same spherical-trig formula used in lx200_protocol.py and
# realtime_tracking.py, extracted here as a free function so catalog
# endpoints can compute Alt/Az without needing a protocol instance.

def _catalog_ra_dec_to_alt_az(
    ra_hours: float,
    dec_deg: float,
    latitude: float,
    longitude: float,
) -> tuple:
    """Convert equatorial (RA/Dec) to horizontal (Alt/Az) for *now*.

    Returns (alt_deg, az_deg) where azimuth is 0=North, 90=East.
    """
    from datetime import datetime, timezone

    # ── Julian Date ───────────────────────────────────────────────
    now = datetime.now(timezone.utc)
    year, month = now.year, now.month
    day = now.day + now.hour / 24 + now.minute / 1440 + now.second / 86400
    if month <= 2:
        year -= 1
        month += 12
    a = int(year / 100)
    b = 2 - a + int(a / 4)
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5

    # ── Greenwich & Local Sidereal Time ──────────────────────────
    gst = (280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360
    lst = ((gst + longitude) / 15.0) % 24  # hours

    # ── Hour Angle ───────────────────────────────────────────────
    ha = (lst - ra_hours) * 15  # degrees
    ha = ha % 360
    if ha > 180:
        ha -= 360

    lat_rad = math.radians(latitude)
    dec_rad = math.radians(dec_deg)
    ha_rad = math.radians(ha)

    # ── Altitude ─────────────────────────────────────────────────
    sin_alt = (math.sin(lat_rad) * math.sin(dec_rad) +
               math.cos(lat_rad) * math.cos(dec_rad) * math.cos(ha_rad))
    sin_alt = max(-1.0, min(1.0, sin_alt))
    alt = math.degrees(math.asin(sin_alt))

    # ── Azimuth (0=North, 90=East) ───────────────────────────────
    cos_alt = math.cos(math.radians(alt))
    if abs(cos_alt) < 1e-10:
        az = 0.0
    else:
        sin_az = -math.cos(dec_rad) * math.sin(ha_rad) / cos_alt
        cos_az = (math.sin(dec_rad) -
                  math.sin(lat_rad) * math.sin(math.radians(alt))) / (
                  math.cos(lat_rad) * cos_alt)
        az = math.degrees(math.atan2(sin_az, cos_az))
        az = az % 360
        if az < 0:
            az += 360

    return alt, az


def _safe_float(tkvar, default: float = 0.0) -> float:
    """Read a float from a tkinter StringVar (or any .get()-able), with fallback."""
    if tkvar is None:
        return default
    try:
        return float(tkvar.get())
    except (ValueError, TypeError, AttributeError):
        return default


# ---------------------------------------------------------------------------
# HTML template served at GET /
# ---------------------------------------------------------------------------
_INDEX_HTML: str  # assigned at bottom of file (keeps module readable)


# ===================================================================
# TelescopeWebServer
# ===================================================================

class TelescopeWebServer:
    """Background Flask server exposing telescope state and controls."""

    # Maximum log lines kept in memory for the web client (reduced for Pi memory)
    MAX_LOG_LINES = 500

    def __init__(
        self,
        app_ref: Any,
        host: str = "0.0.0.0",
        port: int = 8080,
    ) -> None:
        self.app_ref = app_ref
        self.host = host
        self.port = port

        # Thread-safe state snapshot refreshed by the main thread
        self._state: Dict[str, Any] = {}
        self._state_lock = threading.Lock()

        # Circular log buffer (newest at the right)
        self._log_buffer: Deque[Dict[str, Any]] = collections.deque(
            maxlen=self.MAX_LOG_LINES
        )
        self._log_lock = threading.Lock()

        # Monotonic counter so the web client can request only new lines
        self._log_seq = 0

        # --- Camera live-view streaming state ---
        self._camera: Optional[Any] = None       # cv2.VideoCapture (UVC) instance
        self._camera_lock = threading.Lock()
        self._camera_active = False               # True while streaming
        self._camera_index = 0                    # OpenCV device index
        self._camera_source = "uvc"               # "uvc" or "ascom"
        self._android_source = ""                 # "auto"/"zwo"/"uvc"/"phone" hint

        # --- ASCOM camera state (separate from plate-solving ASCOM camera) ---
        self._ascom_cam: Optional[Any] = None     # ASCOMCameraCapture instance
        self._ascom_exposure = 0.5                # Exposure in seconds for live view
        self._ascom_gain = 100                    # Gain setting
        self._ascom_binning = 2                   # Binning for live view (2x2 recommended)

        # Flask app + werkzeug server
        self._flask_app = self._create_flask_app()
        self._server: Optional[Any] = None  # werkzeug.serving.BaseWSGIServer
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Public helpers called from the main thread
    # ------------------------------------------------------------------

    def update_state(self) -> None:
        """Collect a state snapshot.  Called from the tkinter main thread."""
        snapshot = self._collect_state()
        with self._state_lock:
            self._state = snapshot

    def push_log(self, message: str, tag: str = "info") -> None:
        """Append a log line (called from any thread)."""
        with self._log_lock:
            self._log_seq += 1
            self._log_buffer.append(
                {"seq": self._log_seq, "ts": time.time(), "msg": message, "tag": tag}
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the web server in a daemon thread."""
        if self._running:
            return
        self._running = True
        from werkzeug.serving import make_server, BaseWSGIServer
        import socket as _socket

        # Ensure SO_REUSEADDR is set so we can rebind immediately after a
        # crash (avoids "Address already in use" on Android restart).
        _orig_server_bind = BaseWSGIServer.server_bind

        def _patched_server_bind(srv_self):
            srv_self.socket.setsockopt(
                _socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1
            )
            return _orig_server_bind(srv_self)

        BaseWSGIServer.server_bind = _patched_server_bind  # type: ignore
        try:
            self._server = make_server(
                self.host, self.port, self._flask_app, threaded=True
            )
        finally:
            # Restore original to avoid leaking the patch
            BaseWSGIServer.server_bind = _orig_server_bind  # type: ignore

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="TelescopeWebServer",
            daemon=True,
        )
        self._thread.start()
        _logger.info(
            "Web server started on http://%s:%d", self.host, self.port
        )

    def stop(self) -> None:
        """Shut down the server gracefully."""
        if not self._running:
            return
        self._running = False
        # Release camera if streaming
        self._close_camera()
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        _logger.info("Web server stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Command helper -- schedule on tkinter main thread
    # ------------------------------------------------------------------

    def execute_command(self, callback: Callable[[], Any]) -> None:
        """Run *callback* on the tkinter main thread (thread-safe)."""
        try:
            self.app_ref.root.after(0, callback)
        except Exception:
            pass  # root may have been destroyed during shutdown

    def _execute_and_wait(
        self, callback: Callable[[], Any], timeout: float = 5.0
    ) -> tuple:
        """Schedule *callback* on the main thread and block until it runs.

        Returns:
            (True, None) on success,
            (True, exception) if callback raised,
            (False, None) on timeout.
        """
        done = threading.Event()
        error_box: list = []

        def wrapper():
            try:
                callback()
            except Exception as exc:
                error_box.append(exc)
            finally:
                done.set()

        self.execute_command(wrapper)
        finished = done.wait(timeout=timeout)
        if not finished:
            return (False, None)
        if error_box:
            return (True, error_box[0])
        return (True, None)

    def _is_connected(self) -> bool:
        """Check if a telescope (real or simulator) is connected.

        Uses bridge.is_connected as the authoritative source of truth.
        The usb_connected_var HeadlessVar is a UI display flag and may
        lag behind when the bridge drops unexpectedly -- so we do NOT
        use it as a fallback.
        """
        app = self.app_ref
        # Real hardware: bridge.is_connected is ground truth
        bridge = getattr(app, "telescope_bridge", None)
        if bridge and getattr(bridge, "is_connected", False):
            return True
        # Simulator
        if getattr(app, "_simulator_active", False):
            sim = getattr(app, "telescope_simulator", None)
            if sim and getattr(sim, "is_connected", False):
                return True
        return False

    # ------------------------------------------------------------------
    # State snapshot collector
    # ------------------------------------------------------------------

    def _collect_state(self) -> Dict[str, Any]:
        """Build a JSON-serialisable state dict from the live app."""
        app = self.app_ref
        proto = getattr(app, "protocol", None)
        tracking = getattr(app, "tracking", None)

        # ---- Position ------------------------------------------------
        position: Dict[str, Any] = {}
        if proto is not None:
            position = {
                "ra_hours": round(proto.ra_hours, 6),
                "dec_degrees": round(proto.dec_degrees, 4),
                "alt_degrees": round(proto.alt_degrees, 4),
                "az_degrees": round(proto.az_degrees, 4),
                "is_slewing": getattr(proto, "is_slewing", False),
                "is_tracking": getattr(proto, "is_tracking", False),
            }
        # Formatted strings from StringVars (safe under GIL)
        # GoTo target info
        for attr, key in [
            ("goto_target_name_var", "goto_target"),
            ("goto_target_ra_var", "goto_ra"),
            ("goto_target_dec_var", "goto_dec"),
            ("goto_target_alt_var", "goto_alt"),
            ("goto_target_az_var", "goto_az"),
        ]:
            var = getattr(app, attr, None)
            if var is not None:
                try:
                    position[key] = var.get()
                except Exception:
                    position[key] = ""

        for attr, key in [
            ("ra_var", "ra_display"),
            ("dec_var", "dec_display"),
            ("alt_var", "alt_display"),
            ("az_var", "az_display"),
            ("rate_alt_var", "rate_alt"),
            ("rate_az_var", "rate_az"),
        ]:
            var = getattr(app, attr, None)
            if var is not None:
                try:
                    position[key] = var.get()
                except Exception:
                    position[key] = ""

        # ---- Connection ----------------------------------------------
        connection: Dict[str, Any] = {}
        for attr, key in [
            ("usb_connected_var", "connected"),
            ("sim_active_var", "simulator_active"),
        ]:
            var = getattr(app, attr, None)
            if var is not None:
                try:
                    connection[key] = var.get()
                except Exception:
                    connection[key] = False
        for attr, key in [
            ("connection_type_var", "type"),
            ("usb_port_var", "port"),
            ("usb_baudrate_var", "baudrate"),
            ("wifi_ip_var", "wifi_ip"),
            ("wifi_port_var", "wifi_port"),
            ("usb_status_var", "status"),
            ("mount_protocol_var", "protocol"),
        ]:
            var = getattr(app, attr, None)
            if var is not None:
                try:
                    connection[key] = var.get()
                except Exception:
                    connection[key] = ""

        # ---- Tracking ------------------------------------------------
        tracking_state: Dict[str, Any] = {
            "is_running": tracking.is_running if tracking else False,
        }
        for attr, key in [
            ("solve_time_var", "solve_time"),
            ("solve_rate_var", "solve_rate"),
            ("drift_var", "drift"),
            ("rms_var", "rms"),
            ("ml_samples_var", "ml_samples"),
            ("auto_solve_status_var", "auto_solve_status"),
        ]:
            var = getattr(app, attr, None)
            if var is not None:
                try:
                    tracking_state[key] = var.get()
                except Exception:
                    tracking_state[key] = ""
        # Auto-solve running flag
        auto_solver = getattr(app, "auto_solver", None)
        tracking_state["auto_solve_running"] = (
            auto_solver.is_running if auto_solver else False
        )

        # Last solved FOV from ASTAP (for live diagnostics in the UI)
        try:
            from android_bridge.local_solver import get_last_solved_fov
            tracking_state["last_solved_fov"] = get_last_solved_fov()
        except ImportError:
            tracking_state["last_solved_fov"] = 0.0

        # ---- Mount / PEC / Flexure ------------------------------------
        mount_info: Dict[str, Any] = {}
        dt_var = getattr(app, 'mount_drive_type_var', None)
        mount_info['drive_type'] = dt_var.get() if dt_var else 'unknown'
        fl_var = getattr(app, 'flexure_learning_var', None)
        mount_info['flexure_learning'] = fl_var.get() if fl_var else False
        # Flexure stats from tracking controller
        tracking_ctrl = getattr(app, 'tracking', None)
        if tracking_ctrl and hasattr(tracking_ctrl, 'flexure_model') and tracking_ctrl.flexure_model:
            fm = tracking_ctrl.flexure_model
            mount_info['flexure_enabled'] = fm.is_enabled
            mount_info['flexure_samples'] = fm.total_samples
            mount_info['flexure_coverage_pct'] = fm.stats.get('grid_coverage_pct', 0.0)
        else:
            mount_info['flexure_enabled'] = False
            mount_info['flexure_samples'] = 0
            mount_info['flexure_coverage_pct'] = 0.0

        pec: Dict[str, Any] = {}
        for attr, key in [
            ("pec_enabled_var", "enabled"),
        ]:
            var = getattr(app, attr, None)
            if var is not None:
                try:
                    pec[key] = var.get()
                except Exception:
                    pec[key] = False
        for attr, key in [
            ("pec_status_var", "status"),
            ("pec_periods_var", "periods"),
            ("pec_correction_var", "correction"),
        ]:
            var = getattr(app, attr, None)
            if var is not None:
                try:
                    pec[key] = var.get()
                except Exception:
                    pec[key] = ""
        # Include drive type in PEC data for UI display
        pec['drive_type'] = mount_info['drive_type']

        # ---- Weather -------------------------------------------------
        weather: Dict[str, Any] = {}
        for attr, key in [
            ("weather_temp_var", "temperature"),
            ("weather_pressure_var", "pressure"),
            ("weather_humidity_var", "humidity"),
            ("weather_cloud_var", "cloud_cover"),
            ("weather_wind_var", "wind_speed"),
            ("weather_wind_dir_var", "wind_direction"),
            ("weather_gusts_var", "gusts"),
            ("weather_dewpoint_var", "dew_point"),
            ("weather_conditions_var", "conditions"),
            ("weather_observing_var", "observing_quality"),
            ("weather_dew_risk_var", "dew_risk"),
            ("weather_status_var", "status"),
            ("weather_location_var", "location"),
        ]:
            var = getattr(app, attr, None)
            if var is not None:
                try:
                    weather[key] = var.get()
                except Exception:
                    weather[key] = ""

        # ---- Dock controls -------------------------------------------
        controls: Dict[str, Any] = {}
        for attr, key in [
            ("telescope_status_var", "telescope_status"),
            ("telescope_speed_var", "telescope_speed"),
            ("focuser_status_var", "focuser_status"),
            ("focuser_position_var", "focuser_position"),
            ("focuser_speed_var", "focuser_speed"),
            ("derotator_angle_var", "derotator_angle"),
            ("derotator_status_var", "derotator_status"),
            ("derotator_speed_var", "derotator_speed"),
        ]:
            var = getattr(app, attr, None)
            if var is not None:
                try:
                    controls[key] = var.get()
                except Exception:
                    controls[key] = ""

        # ---- Location ------------------------------------------------
        location: Dict[str, Any] = {}
        for attr, key in [
            ("lat_var", "latitude"),
            ("lon_var", "longitude"),
        ]:
            var = getattr(app, attr, None)
            if var is not None:
                try:
                    location[key] = var.get()
                except Exception:
                    location[key] = ""

        # ---- Camera / Solve settings ---------------------------------
        camera: Dict[str, Any] = {}
        for attr, key in [
            ("solve_mode_var", "solve_mode"),
            ("camera_index_var", "camera_index"),
            ("ascom_camera_name_var", "ascom_camera_name"),
            ("ascom_camera_id_var", "ascom_camera_id"),
            ("ascom_exposure_var", "ascom_exposure"),
            ("ascom_gain_var", "ascom_gain"),
            ("ascom_binning_var", "ascom_binning"),
            ("watch_folder_var", "watch_folder"),
            ("solve_interval_var", "solve_interval"),
            ("astap_path_var", "astap_path"),
            ("save_images_var", "save_images"),
            ("save_folder_var", "save_folder"),
            ("save_format_var", "save_format"),
        ]:
            var = getattr(app, attr, None)
            if var is not None:
                try:
                    camera[key] = var.get()
                except Exception:
                    camera[key] = ""

        # ---- Session -------------------------------------------------
        session: Dict[str, Any] = {"has_session": False}
        recorder = getattr(app, "session_recorder", None)
        if recorder and recorder.is_started:
            session["has_session"] = True

        # ---- OnStep Extended -----------------------------------------
        onstep: Dict[str, Any] = {}

        # Park state
        ps_var = getattr(app, 'park_state_var', None)
        onstep['park_state'] = ps_var.get() if ps_var else 'Unknown'

        # Tracking rate & enable
        tr_var = getattr(app, 'tracking_rate_var', None)
        onstep['tracking_rate'] = tr_var.get() if tr_var else 'Sidereal'
        te_var = getattr(app, 'tracking_enabled_var', None)
        onstep['tracking_enabled'] = te_var.get() if te_var else False

        # Mount-side PEC
        mpec_var = getattr(app, 'mount_pec_status_var', None)
        onstep['mount_pec_status'] = mpec_var.get() if mpec_var else '--'
        mpec_rec = getattr(app, 'mount_pec_recorded_var', None)
        onstep['mount_pec_recorded'] = mpec_rec.get() if mpec_rec else False

        # Firmware
        for attr, key in [
            ('firmware_name_var', 'firmware_name'),
            ('firmware_version_var', 'firmware_version'),
            ('firmware_mount_type_var', 'firmware_mount_type'),
        ]:
            var = getattr(app, attr, None)
            onstep[key] = var.get() if var else '--'

        # Backlash
        for attr, key in [
            ('backlash_ra_var', 'backlash_ra'),
            ('backlash_dec_var', 'backlash_dec'),
        ]:
            var = getattr(app, attr, None)
            onstep[key] = var.get() if var else '--'

        # Limits
        for attr, key in [
            ('horizon_limit_var', 'horizon_limit'),
            ('overhead_limit_var', 'overhead_limit'),
        ]:
            var = getattr(app, attr, None)
            onstep[key] = var.get() if var else '--'

        # Auxiliary features
        onstep['auxiliary_features'] = getattr(app, '_auxiliary_features', [])

        # Extended focuser
        for attr, key in [
            ('focuser_target_var', 'focuser_target'),
            ('focuser_temperature_var', 'focuser_temperature'),
            ('focuser_selected_var', 'focuser_selected'),
        ]:
            var = getattr(app, attr, None)
            onstep[key] = var.get() if var else '--'
        ftcf_var = getattr(app, 'focuser_tcf_var', None)
        onstep['focuser_tcf'] = ftcf_var.get() if ftcf_var else False

        # Rotator
        for attr, key in [
            ('rotator_angle_var', 'rotator_angle'),
            ('rotator_status_var', 'rotator_status'),
        ]:
            var = getattr(app, attr, None)
            onstep[key] = var.get() if var else '--'
        rdr_var = getattr(app, 'rotator_derotating_var', None)
        onstep['rotator_derotating'] = rdr_var.get() if rdr_var else False

        return {
            "timestamp": time.time(),
            "platform": "android" if _IS_ANDROID else "desktop",
            "position": position,
            "connection": connection,
            "tracking": tracking_state,
            "mount": mount_info,
            "pec": pec,
            "weather": weather,
            "controls": controls,
            "location": location,
            "camera": camera,
            "session": session,
            "onstep": onstep,
        }

    # ------------------------------------------------------------------
    # Camera live-view helpers
    # ------------------------------------------------------------------

    def _open_camera(self, index: int = 0, source: str = "uvc",
                     ascom_id: str = "", exposure: float = 0.5,
                     gain: int = 100, binning: int = 2,
                     android_source: str = "") -> bool:
        """Open a camera for MJPEG streaming.

        Args:
            index: OpenCV device index (UVC mode only).
            source: ``"uvc"`` for USB/webcam via OpenCV, ``"ascom"`` for
                ASCOM-compatible astronomy cameras (ZWO, QHY, etc.).
            ascom_id: ASCOM ProgID (e.g. ``"ASCOM.ASICamera2.Camera"``).
                Required when *source* is ``"ascom"``.
            exposure: ASCOM exposure time in seconds for live view frames.
            gain: ASCOM sensor gain.
            binning: ASCOM pixel binning factor.
            android_source: Android camera source hint — ``"auto"``,
                ``"zwo"``, ``"uvc"``, or ``"phone"``.  Passed through to
                the Android camera bridge monkey-patch.

        Returns:
            True if the camera was opened successfully.
        """
        # Close any existing camera first
        self._close_camera()

        self._camera_source = source
        # Store android_source hint for the monkey-patched open method
        self._android_source = android_source

        if source == "ascom":
            return self._open_ascom_camera(ascom_id, exposure, gain, binning)
        elif source == "asi":
            return self._open_asi_camera(index)
        else:
            return self._open_uvc_camera(index)

    def _open_uvc_camera(self, index: int = 0) -> bool:
        """Open a UVC (USB/webcam) camera via OpenCV."""
        if not _HAS_CV2:
            _logger.warning("OpenCV not installed — cannot open UVC camera")
            return False
        with self._camera_lock:
            import sys
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2  # type: ignore
            cap = cv2.VideoCapture(index, backend)  # type: ignore
            if not cap.isOpened():
                # Fallback without explicit backend
                cap = cv2.VideoCapture(index)  # type: ignore
            if cap.isOpened():
                # Set modest resolution for Pi streaming performance
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # type: ignore
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)  # type: ignore
                self._camera = cap
                self._camera_index = index
                self._camera_active = True
                _logger.info("UVC camera %d opened for live view", index)
                return True
            _logger.warning("Failed to open UVC camera %d", index)
            return False

    def _open_ascom_camera(self, camera_id: str, exposure: float = 0.5,
                           gain: int = 100, binning: int = 2) -> bool:
        """Open an ASCOM astronomy camera for live-view streaming.

        Creates a dedicated ASCOMCameraCapture instance (separate from
        the plate-solving camera) and connects to the specified device.

        Note:
            ASCOM cameras only allow one connection per device.  If the
            plate solver is using the same ASCOM camera, stop it first.
        """
        if not _HAS_ASCOM:
            _logger.warning("ASCOM support not available (Windows + pywin32 required)")
            return False
        if not _HAS_NUMPY:
            _logger.warning("numpy required for ASCOM frame processing")
            return False
        with self._camera_lock:
            try:
                cam = ASCOMCameraCapture()  # type: ignore
                cam.gain = gain
                cam.binning = binning
                cam.on_log = lambda msg: _logger.info("[ASCOM LiveView] %s", msg)
                if not cam.connect(camera_id):
                    _logger.warning("Failed to connect ASCOM camera: %s", camera_id)
                    return False
                self._ascom_cam = cam
                self._ascom_exposure = exposure
                self._ascom_gain = gain
                self._ascom_binning = binning
                self._camera_active = True
                _logger.info("ASCOM camera opened for live view: %s (exp=%.2fs, gain=%d, bin=%d)",
                             camera_id, exposure, gain, binning)
                return True
            except Exception as e:
                _logger.error("ASCOM camera open error: %s", e)
                return False

    def _open_asi_camera(self, index: int = 0) -> bool:
        """Open a ZWO ASI camera via the native SDK (Linux/RPi/Windows).

        Uses the ``asi_camera`` module which wraps the C SDK via ctypes.
        The camera runs a background thread that grabs frames and converts
        them to JPEG, accessible via ``asi_camera.get_jpeg_frame()``.
        """
        try:
            import asi_camera  # type: ignore[import-not-found]
        except ImportError:
            _logger.warning("asi_camera module not available")
            return False

        if not asi_camera.is_asi_sdk_available():
            _logger.warning("ASI SDK library not found — cannot open ASI camera")
            return False

        cameras = asi_camera.list_cameras()
        if not cameras:
            _logger.warning("No ASI cameras connected")
            return False

        cam_idx = min(index, len(cameras) - 1)
        _logger.info("Opening ASI camera #%d: %s", cam_idx, cameras[cam_idx].get("name", "?"))

        if asi_camera.open_asi_camera(cam_idx):
            self._camera_active = True
            self._camera_source = "asi"
            _logger.info("ASI camera opened via native SDK: %s",
                         asi_camera.get_active_source())
            return True
        _logger.warning("Failed to open ASI camera #%d", cam_idx)
        return False

    def _close_camera(self) -> None:
        """Release whichever camera is currently active (UVC, ASCOM, or ASI)."""
        with self._camera_lock:
            self._camera_active = False
            # Close UVC
            if self._camera is not None:
                self._camera.release()
                self._camera = None
            # Close ASCOM
            if self._ascom_cam is not None:
                try:
                    self._ascom_cam.disconnect()
                except Exception:
                    pass
                self._ascom_cam = None
            # Close ASI (native SDK)
            if self._camera_source == "asi":
                try:
                    import asi_camera  # type: ignore[import-not-found]
                    asi_camera.close_camera()
                except Exception:
                    pass
            _logger.info("Camera closed")

    # ------------------------------------------------------------------
    # ASCOM frame capture helper
    # ------------------------------------------------------------------

    def _capture_ascom_frame_jpeg(self) -> Optional[bytes]:
        """Capture one frame from the ASCOM camera and return JPEG bytes.

        Performs a short exposure, retrieves the 16-bit image array,
        auto-stretches to 8-bit, and JPEG-encodes for streaming.

        Returns:
            JPEG bytes on success, None on failure.
        """
        cam = self._ascom_cam
        if cam is None or not cam.is_connected or cam.camera is None:
            return None
        try:
            # Start exposure (True = light frame)
            cam.camera.StartExposure(self._ascom_exposure, True)

            # Poll until ready (with timeout)
            timeout = self._ascom_exposure + 10
            start = time.time()
            while not cam.camera.ImageReady:
                if time.time() - start > timeout:
                    try:
                        cam.camera.AbortExposure()
                    except Exception:
                        pass
                    return None
                if not self._camera_active:
                    return None  # Cancelled
                time.sleep(0.05)

            # Retrieve raw image array (COM SAFEARRAY -> numpy)
            image_data = cam.camera.ImageArray
            arr = np.array(image_data, dtype=np.uint16)  # type: ignore
            if arr.ndim == 2:
                arr = arr.T  # ASCOM returns column-major

            # Auto-stretch 16-bit to 8-bit
            arr_min = arr.min()
            arr_max = arr.max()
            if arr_max > arr_min:
                arr8 = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)  # type: ignore
            else:
                arr8 = np.zeros(arr.shape, dtype=np.uint8)  # type: ignore

            # Encode as JPEG via OpenCV (if available) or PIL
            if _HAS_CV2:
                ok, buf = cv2.imencode('.jpg', arr8, [cv2.IMWRITE_JPEG_QUALITY, 80])  # type: ignore
                if ok:
                    return buf.tobytes()
            else:
                # Fallback: use PIL
                try:
                    from PIL import Image
                    import io
                    img = Image.fromarray(arr8, mode='L')
                    bio = io.BytesIO()
                    img.save(bio, format='JPEG', quality=80)
                    return bio.getvalue()
                except ImportError:
                    pass
            return None
        except Exception as e:
            _logger.debug("ASCOM frame capture error: %s", e)
            return None

    # ------------------------------------------------------------------
    # MJPEG generator (supports both UVC and ASCOM sources)
    # ------------------------------------------------------------------

    def _generate_mjpeg(self):
        """Generator yielding MJPEG frames for the /api/camera/stream endpoint.

        Each frame is JPEG-encoded and wrapped in the multipart/x-mixed-replace
        boundary protocol that browsers natively support via ``<img src="...">``.

        For UVC cameras, targets ~10 fps.  For ASCOM cameras, frame rate
        depends on exposure time (typically 1-5 fps for short exposures).
        """
        while self._camera_active:
            if self._camera_source == "ascom":
                jpeg = self._capture_ascom_frame_jpeg()
                if jpeg is None:
                    time.sleep(0.2)
                    continue
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'
                    + jpeg
                    + b'\r\n'
                )
                # No extra sleep — exposure time already limits frame rate
            elif self._camera_source == "asi":
                # ASI SDK path — frames grabbed by background thread
                try:
                    import asi_camera  # type: ignore[import-not-found]
                    jpeg = asi_camera.get_jpeg_frame()
                except ImportError:
                    jpeg = None
                if jpeg is None:
                    time.sleep(0.1)
                    continue
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'
                    + jpeg
                    + b'\r\n'
                )
                time.sleep(0.05)  # ~20 fps cap (actual rate limited by exposure)
            else:
                # UVC path (OpenCV VideoCapture)
                frame = None
                with self._camera_lock:
                    if self._camera is not None and self._camera.isOpened():
                        ret, frame = self._camera.read()
                        if not ret:
                            frame = None
                if frame is None:
                    time.sleep(0.1)
                    continue
                # Encode as JPEG (quality 70 balances size vs quality for WiFi)
                ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])  # type: ignore
                if not ok:
                    time.sleep(0.1)
                    continue
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'
                    + buf.tobytes()
                    + b'\r\n'
                )
                time.sleep(0.09)  # ~10 fps cap

    # ------------------------------------------------------------------
    # Flask app factory
    # ------------------------------------------------------------------

    def _create_flask_app(self) -> Flask:
        flask_app = Flask(__name__)
        flask_app.config["JSON_SORT_KEYS"] = False

        # Silence werkzeug request logging (very noisy with 1 Hz polling)
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.WARNING)

        server = self  # closure ref

        # ============================================================
        # Optional API token authentication
        # ============================================================
        # If ``web.api_token`` is set in config (non-empty string), every
        # API request must include ``Authorization: Bearer <token>`` or
        # ``?token=<token>`` in the query string.  The HTML page and
        # manifest are always public so the browser can load the UI.
        _api_token = ""
        try:
            cfg = getattr(server.app_ref, "config_manager", None)
            if cfg:
                _api_token = cfg.get("web.api_token", "")
        except Exception:
            pass

        @flask_app.before_request
        def _check_api_token():
            if not _api_token:
                return None  # No token configured -- allow all
            path = request.path
            # Always allow the HTML page and manifest
            if path in ("/", "/manifest.json"):
                return None
            # Allow static-like paths
            if not path.startswith("/api/"):
                return None
            # Check Bearer token in header or query param
            auth = request.headers.get("Authorization", "")
            if auth == f"Bearer {_api_token}":
                return None
            if request.args.get("token") == _api_token:
                return None
            return jsonify({"ok": False, "error": "Unauthorized"}), 401

        # ============================================================
        # Page
        # ============================================================

        @flask_app.route("/")
        def index():
            return _INDEX_HTML

        @flask_app.route("/help")
        def help_page():
            """Serve the user guide HTML.  Searches several locations so it
            works on both desktop (file next to script or in android app/)
            and Android (bundled by Chaquopy alongside the .py modules)."""
            _candidates = [
                os.path.join(os.path.dirname(__file__), "TelescopeController_UserGuide.html"),
                os.path.join(os.path.dirname(__file__), "..", "android app", "TelescopeController_UserGuide.html"),
                os.path.join(os.path.dirname(__file__), "android app", "TelescopeController_UserGuide.html"),
            ]
            for _p in _candidates:
                if os.path.isfile(_p):
                    with open(_p, "r", encoding="utf-8") as _f:
                        return _f.read()
            return "<html><body style='background:#0d0d14;color:#e0e0f0;font-family:sans-serif;padding:40px'><h1>User Guide</h1><p>User guide file not found. Please ensure <code>TelescopeController_UserGuide.html</code> is bundled with the app.</p></body></html>", 404

        @flask_app.route("/static/chart.umd.min.js")
        def chart_js():
            """Serve bundled Chart.js so charts work offline (telescope WiFi)."""
            js_path = os.path.join(os.path.dirname(__file__), "chart.umd.min.js")
            if os.path.isfile(js_path):
                return send_file(js_path, mimetype="application/javascript")
            return "", 404

        @flask_app.route("/manifest.json")
        def manifest():
            return jsonify({
                "name": "TrackWise-AltAzPro",
                "short_name": "TrackWise",
                "description": "TrackWise-AltAzPro by CRACIUN BOGDAN - Intelligent Alt-Az Mount Tracking & Control",
                "start_url": "/",
                "display": "standalone",
                "orientation": "portrait",
                "background_color": "#0d0d0d",
                "theme_color": "#0d0d0d",
                "icons": []
            })

        # ============================================================
        # State endpoints (GET)
        # ============================================================

        @flask_app.route("/api/status")
        def api_status():
            with server._state_lock:
                state = server._state.copy()
            return jsonify(state)

        @flask_app.route("/api/log")
        def api_log():
            since = request.args.get("since", 0, type=int)
            with server._log_lock:
                lines = [
                    entry for entry in server._log_buffer if entry["seq"] > since
                ]
            return jsonify({"lines": lines})

        # ============================================================
        # Connection endpoints
        # ============================================================

        def _run(callback, label="Command"):
            """Run callback via _execute_and_wait and return proper JSON."""
            finished, exc = server._execute_and_wait(callback)
            if not finished:
                return jsonify({"ok": False, "error": f"{label}: timeout"}), 504
            if exc:
                return jsonify({"ok": False, "error": f"{label}: {exc}"}), 500
            return jsonify({"ok": True})

        def _require_connection():
            """Return error response if not connected, else None."""
            if not server._is_connected():
                return jsonify({"ok": False, "error": "Telescope not connected"}), 400
            return None

        @flask_app.route("/api/connect", methods=["POST"])
        def api_connect():
            app = server.app_ref
            if server._is_connected():
                return jsonify({"ok": False, "error": "Already connected"}), 400
            server.push_log("[WebUI] Connect requested", "server")
            return _run(app._toggle_usb_telescope, "Connect")

        @flask_app.route("/api/disconnect", methods=["POST"])
        def api_disconnect():
            app = server.app_ref
            if not server._is_connected():
                return jsonify({"ok": False, "error": "Not connected"}), 400
            server.push_log("[WebUI] Disconnect requested", "server")
            return _run(app._toggle_usb_telescope, "Disconnect")

        @flask_app.route("/api/simulator", methods=["POST"])
        def api_simulator():
            server.push_log("[WebUI] Simulator toggle requested", "server")
            return _run(server.app_ref._toggle_simulator, "Simulator")

        # ============================================================
        # Tracking endpoints
        # ============================================================

        @flask_app.route("/api/tracking/start", methods=["POST"])
        def api_tracking_start():
            app = server.app_ref
            err = _require_connection()
            if err: return err
            tracking = getattr(app, "tracking", None)
            if tracking and tracking.is_running:
                return jsonify({"ok": False, "error": "Already tracking"}), 400
            return _run(app._toggle_tracking, "Start tracking")

        @flask_app.route("/api/tracking/stop", methods=["POST"])
        def api_tracking_stop():
            err = _require_connection()
            if err: return err
            app = server.app_ref
            tracking = getattr(app, "tracking", None)
            if tracking and not tracking.is_running:
                return jsonify({"ok": False, "error": "Not tracking"}), 400
            return _run(app._toggle_tracking, "Stop tracking")

        @flask_app.route("/api/session/save", methods=["POST"])
        def api_session_save():
            """Manually save all session data to a timestamped folder."""
            app = server.app_ref
            recorder = getattr(app, "session_recorder", None)
            if not recorder:
                return jsonify({"ok": False, "error": "Session recorder not available"}), 500
            # Auto-start recording if not started (covers cases where
            # the user never started tracking but still wants to save).
            if not recorder.is_started:
                recorder.start()
            try:
                path = recorder.save(auto=False)
                if path:
                    return jsonify({"ok": True, "path": path})
                return jsonify({"ok": False, "error": "Failed to create session folder"}), 500
            except Exception as exc:
                return jsonify({"ok": False, "error": str(exc)}), 500

        @flask_app.route("/api/session/download")
        def api_session_download():
            """Download the latest session as a ZIP file."""
            import zipfile, io as _io
            app = server.app_ref
            recorder = getattr(app, "session_recorder", None)
            if not recorder:
                return jsonify({"ok": False, "error": "No recorder"}), 500
            # Save first
            if not recorder.is_started:
                recorder.start()
            path = recorder.save(auto=False)
            if not path:
                return jsonify({"ok": False, "error": "Save failed"}), 500
            # Build ZIP in memory
            buf = _io.BytesIO()
            import pathlib
            session_dir = pathlib.Path(path)
            with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                for f in session_dir.rglob('*'):
                    if f.is_file():
                        arcname = f"{session_dir.name}/{f.relative_to(session_dir)}"
                        zf.write(f, arcname)
            buf.seek(0)
            from flask import send_file
            return send_file(
                buf,
                mimetype='application/zip',
                as_attachment=True,
                download_name=f"session_{session_dir.name}.zip",
            )

        @flask_app.route("/api/solve/start", methods=["POST"])
        def api_solve_start():
            err = _require_connection()
            if err: return err
            app = server.app_ref
            solver = getattr(app, "auto_solver", None)
            if solver and solver.is_running:
                return jsonify({"ok": False, "error": "Already solving"}), 400
            return _run(app._toggle_auto_solve, "Start solving")

        @flask_app.route("/api/solve/stop", methods=["POST"])
        def api_solve_stop():
            err = _require_connection()
            if err: return err
            app = server.app_ref
            solver = getattr(app, "auto_solver", None)
            if solver and not solver.is_running:
                return jsonify({"ok": False, "error": "Not solving"}), 400
            return _run(app._toggle_auto_solve, "Stop solving")

        # ============================================================
        # GoTo
        # ============================================================

        @flask_app.route("/api/goto", methods=["POST"])
        def api_goto():
            err = _require_connection()
            if err: return err
            data = request.get_json(silent=True) or {}
            target = data.get("target", "").strip()
            ra = data.get("ra", "").strip()
            dec = data.get("dec", "").strip()
            server.push_log(f"[WebUI] GoTo requested: target={target or 'manual'} RA={ra} Dec={dec}", "server")

            app = server.app_ref

            # If a named target is given, try catalog lookup
            if target and not ra:
                try:
                    from catalog_loader import load_all_catalogs, get_solar_system_objects
                    catalogs = load_all_catalogs()
                    key = target.lower()
                    ra_h = dec_d = None
                    # Check solar system objects first (fresh positions)
                    try:
                        solar = get_solar_system_objects()
                        if key in solar:
                            ra_h, dec_d = solar[key]
                    except Exception:
                        pass
                    # Then check fixed catalogs
                    if ra_h is None and key in catalogs:
                        ra_h, dec_d = catalogs[key]
                    if ra_h is not None and dec_d is not None:
                        # Format to LX200 strings
                        ra_h_int = int(ra_h)
                        ra_m = (ra_h - ra_h_int) * 60
                        ra_m_int = int(ra_m)
                        ra_s = (ra_m - ra_m_int) * 60
                        ra = f"{ra_h_int:02d}:{ra_m_int:02d}:{ra_s:04.1f}"

                        sign = "+" if dec_d >= 0 else "-"
                        dec_abs = abs(dec_d)
                        dec_deg = int(dec_abs)
                        dec_m = (dec_abs - dec_deg) * 60
                        dec_m_int = int(dec_m)
                        dec_s = (dec_m - dec_m_int) * 60
                        dec = f"{sign}{dec_deg:02d}*{dec_m_int:02d}:{dec_s:04.1f}"
                    else:
                        return jsonify(
                            {"ok": False, "error": f"Target '{target}' not found in catalog"}
                        ), 404
                except ImportError:
                    return jsonify(
                        {"ok": False, "error": "Catalog loader not available"}
                    ), 500

            if not ra or not dec:
                return jsonify(
                    {"ok": False, "error": "Provide ra+dec or target name"}
                ), 400

            # Store target info for display in the Position card
            target_name_var = getattr(app, "goto_target_name_var", None)
            target_ra_var = getattr(app, "goto_target_ra_var", None)
            target_dec_var = getattr(app, "goto_target_dec_var", None)
            if target_name_var is not None:
                target_name_var.set(target if target else f"RA {ra}")
            if target_ra_var is not None:
                target_ra_var.set(ra)
            if target_dec_var is not None:
                target_dec_var.set(dec)

            # Parse RA/Dec to floats for live Alt/Az recomputation
            try:
                _parse = getattr(app, "_parse_lx200_dms", None)
                if _parse and ra and dec:
                    ra_h = _parse(ra)
                    dec_d = _parse(dec)
                    app._goto_target_ra_hours = ra_h
                    app._goto_target_dec_deg = dec_d
                    # Compute initial Alt/Az
                    proto = getattr(app, "protocol", None)
                    if proto and hasattr(proto, "_ra_dec_to_alt_az"):
                        t_alt, t_az = proto._ra_dec_to_alt_az(ra_h, dec_d)
                        if hasattr(app, "goto_target_alt_var"):
                            app.goto_target_alt_var.set(f"{t_alt:.1f}\u00b0")
                        if hasattr(app, "goto_target_az_var"):
                            app.goto_target_az_var.set(f"{t_az:.1f}\u00b0")
            except Exception:
                pass

            goto_error = [None]

            def do_goto():
                goto_error[0] = app._goto_altaz_from_radec(ra, dec)

            finished, exc = server._execute_and_wait(do_goto)
            if not finished:
                return jsonify({"ok": False, "error": "GoTo: timeout"}), 504
            if exc:
                return jsonify({"ok": False, "error": f"GoTo: {exc}"}), 500
            if goto_error[0]:
                return jsonify({"ok": False, "error": goto_error[0]}), 500
            return jsonify({"ok": True, "ra": ra, "dec": dec, "target": target or ""})

        # ============================================================
        # Slew / Park / Home
        # ============================================================

        @flask_app.route("/api/slew", methods=["POST"])
        def api_slew():
            err = _require_connection()
            if err: return err
            data = request.get_json(silent=True) or {}
            direction = data.get("direction", "").upper()
            speed = data.get("speed")
            if direction not in ("N", "S", "E", "W"):
                return jsonify({"ok": False, "error": "Invalid direction"}), 400

            app = server.app_ref
            if speed is not None:
                def set_speed():
                    try:
                        app.telescope_speed_var.set(str(int(speed)))
                    except Exception:
                        pass
                server._execute_and_wait(set_speed)

            return _run(lambda: app._slew_telescope(direction), "Slew")

        @flask_app.route("/api/slew/stop", methods=["POST"])
        def api_slew_stop():
            err = _require_connection()
            if err: return err
            return _run(server.app_ref._stop_telescope, "Slew stop")

        @flask_app.route("/api/park", methods=["POST"])
        def api_park():
            err = _require_connection()
            if err: return err
            return _run(server.app_ref._park_telescope, "Park")

        @flask_app.route("/api/home", methods=["POST"])
        def api_home():
            err = _require_connection()
            if err: return err
            finished, exc = server._execute_and_wait(
                server.app_ref._home_telescope
            )
            if not finished:
                return jsonify({"ok": False, "error": "Home: timeout"}), 504
            if exc:
                return jsonify({"ok": False, "error": f"Home: {exc}"}), 500
            return jsonify({
                "ok": True,
                "message": "Home reset complete. Position is now 0,0 (this is normal)."
            })

        # ============================================================
        # Focuser
        # ============================================================

        @flask_app.route("/api/focuser/move", methods=["POST"])
        def api_focuser_move():
            err = _require_connection()
            if err: return err
            data = request.get_json(silent=True) or {}
            direction = data.get("direction", "").upper()
            if direction not in ("IN", "OUT"):
                return jsonify({"ok": False, "error": "Invalid direction"}), 400
            speed = data.get("speed")
            app = server.app_ref
            if speed is not None:
                def set_speed():
                    try:
                        app.focuser_speed_var.set(str(int(speed)))
                    except Exception:
                        pass
                server._execute_and_wait(set_speed)
            return _run(lambda: app._move_focuser(direction), "Focuser")

        @flask_app.route("/api/focuser/stop", methods=["POST"])
        def api_focuser_stop():
            err = _require_connection()
            if err: return err
            return _run(server.app_ref._stop_focuser, "Focuser stop")

        # ============================================================
        # Derotator
        # ============================================================

        @flask_app.route("/api/derotator/rotate", methods=["POST"])
        def api_derotator_rotate():
            err = _require_connection()
            if err: return err
            data = request.get_json(silent=True) or {}
            direction = data.get("direction", "").upper()
            if direction not in ("CW", "CCW"):
                return jsonify({"ok": False, "error": "Invalid direction"}), 400
            speed = data.get("speed")
            app = server.app_ref
            if speed is not None:
                def set_speed():
                    try:
                        app.derotator_speed_var.set(str(float(speed)))
                    except Exception:
                        pass
                server._execute_and_wait(set_speed)
            return _run(lambda: app._rotate_derotator(direction), "Derotator")

        @flask_app.route("/api/derotator/stop", methods=["POST"])
        def api_derotator_stop():
            err = _require_connection()
            if err: return err
            return _run(server.app_ref._stop_derotator, "Derotator stop")

        @flask_app.route("/api/derotator/sync", methods=["POST"])
        def api_derotator_sync():
            err = _require_connection()
            if err: return err
            return _run(server.app_ref._sync_derotator, "Derotator sync")

        # ============================================================
        # PEC
        # ============================================================

        @flask_app.route("/api/pec/toggle", methods=["POST"])
        def api_pec_toggle():
            err = _require_connection()
            if err: return err
            return _run(server.app_ref._toggle_pec, "PEC toggle")

        @flask_app.route("/api/pec/save", methods=["POST"])
        def api_pec_save():
            err = _require_connection()
            if err: return err
            return _run(server.app_ref._pec_save, "PEC save")

        @flask_app.route("/api/pec/load", methods=["POST"])
        def api_pec_load():
            err = _require_connection()
            if err: return err
            return _run(server.app_ref._pec_load, "PEC load")

        @flask_app.route("/api/pec/reset", methods=["POST"])
        def api_pec_reset():
            err = _require_connection()
            if err: return err
            return _run(server.app_ref._pec_reset, "PEC reset")

        @flask_app.route("/api/mount/drive-type", methods=["POST"])
        def api_set_drive_type():
            """Set mount drive type. Body: {"drive_type": "worm_gear"|...}"""
            data = request.get_json(silent=True) or {}
            drive_type = data.get("drive_type", "")
            valid = ['worm_gear', 'planetary_gearbox', 'harmonic_drive',
                     'belt_drive', 'direct_drive']
            if drive_type not in valid:
                return jsonify(ok=False, error=f"Invalid type. Use: {valid}"), 400
            return _run(lambda: server.app_ref._set_drive_type(drive_type),
                        "Set drive type")

        @flask_app.route("/api/mount/drive-type")
        def api_get_drive_type():
            """Return current mount drive type."""
            app = server.app_ref
            dt = getattr(app, 'mount_drive_type_var', None)
            return jsonify(ok=True, drive_type=dt.get() if dt else 'unknown')

        @flask_app.route("/api/flexure/toggle", methods=["POST"])
        def api_flexure_toggle():
            """Toggle flexure learning model on/off."""
            return _run(server.app_ref._toggle_flexure_learning,
                        "Flexure toggle")

        @flask_app.route("/api/flexure/reset", methods=["POST"])
        def api_flexure_reset():
            """Reset the flexure learning model."""
            def _reset():
                tracking = getattr(server.app_ref, 'tracking', None)
                if tracking and hasattr(tracking, 'flexure_model') and tracking.flexure_model:
                    tracking.flexure_model.reset()
                    server.app_ref._log("Flexure model reset", "warning")
            return _run(_reset, "Flexure reset")

        @flask_app.route("/api/flexure/stats")
        def api_flexure_stats():
            """Return flexure model statistics."""
            tracking = getattr(server.app_ref, 'tracking', None)
            if tracking and hasattr(tracking, 'flexure_model') and tracking.flexure_model:
                return jsonify(ok=True, **tracking.flexure_model.get_statistics())
            return jsonify(ok=True, is_enabled=False, total_samples=0)

        # ============================================================
        # OnStep Extended: Park / Unpark
        # ============================================================

        @flask_app.route("/api/unpark", methods=["POST"])
        def api_unpark():
            err = _require_connection()
            if err:
                return err
            return _run(server.app_ref._unpark_telescope, "Unpark")

        @flask_app.route("/api/park/set", methods=["POST"])
        def api_park_set():
            err = _require_connection()
            if err:
                return err
            return _run(server.app_ref._set_park_position, "Set park position")

        # ============================================================
        # OnStep Extended: Tracking Rate
        # ============================================================

        @flask_app.route("/api/tracking/rate", methods=["POST"])
        def api_tracking_rate():
            err = _require_connection()
            if err:
                return err
            data = request.get_json(silent=True) or {}
            rate = data.get("rate", "sidereal")
            if rate not in ("sidereal", "lunar", "solar", "king"):
                return jsonify({"ok": False, "error": f"Invalid rate: {rate}"}), 400
            return _run(lambda: server.app_ref._set_tracking_rate(rate),
                        f"Tracking rate {rate}")

        @flask_app.route("/api/tracking/enable", methods=["POST"])
        def api_tracking_enable():
            err = _require_connection()
            if err:
                return err
            return _run(server.app_ref._enable_tracking, "Enable tracking")

        @flask_app.route("/api/tracking/disable", methods=["POST"])
        def api_tracking_disable():
            err = _require_connection()
            if err:
                return err
            return _run(server.app_ref._disable_tracking, "Disable tracking")

        # ============================================================
        # OnStep Extended: Tracking Configuration
        # ============================================================

        @flask_app.route("/api/tracking/axis_mode", methods=["POST"])
        def api_tracking_axis_mode():
            err = _require_connection()
            if err:
                return err
            data = request.get_json(silent=True) or {}
            mode = data.get("mode", 2)
            if mode not in (1, 2):
                return jsonify({"ok": False, "error": "Mode must be 1 or 2"}), 400
            return _run(lambda: server.app_ref._set_tracking_axis_mode(mode),
                        f"Tracking axis mode {mode}")

        @flask_app.route("/api/tracking/compensation", methods=["POST"])
        def api_tracking_compensation():
            err = _require_connection()
            if err:
                return err
            data = request.get_json(silent=True) or {}
            model = data.get("model", "full")
            if model not in ("full", "refraction", "none"):
                return jsonify({"ok": False, "error": f"Invalid model: {model}"}), 400
            return _run(lambda: server.app_ref._set_compensation_model(model),
                        f"Compensation {model}")

        @flask_app.route("/api/tracking/sidereal_clock", methods=["POST"])
        def api_tracking_sidereal_clock():
            err = _require_connection()
            if err:
                return err
            data = request.get_json(silent=True) or {}
            direction = data.get("direction", "reset")
            if direction not in ("+", "-", "reset"):
                return jsonify({"ok": False, "error": f"Invalid direction: {direction}"}), 400
            return _run(lambda: server.app_ref._adjust_sidereal_clock(direction),
                        f"Sidereal clock {direction}")

        # ============================================================
        # OnStep Extended: Mount-side PEC
        # ============================================================

        @flask_app.route("/api/mount/pec/play", methods=["POST"])
        def api_mount_pec_play():
            err = _require_connection()
            if err:
                return err
            return _run(server.app_ref._mount_pec_playback_start,
                        "PEC playback start")

        @flask_app.route("/api/mount/pec/stop", methods=["POST"])
        def api_mount_pec_stop():
            err = _require_connection()
            if err:
                return err
            return _run(server.app_ref._mount_pec_playback_stop,
                        "PEC playback stop")

        @flask_app.route("/api/mount/pec/record", methods=["POST"])
        def api_mount_pec_record():
            err = _require_connection()
            if err:
                return err
            return _run(server.app_ref._mount_pec_record_start,
                        "PEC record start")

        @flask_app.route("/api/mount/pec/record/stop", methods=["POST"])
        def api_mount_pec_record_stop():
            err = _require_connection()
            if err:
                return err
            return _run(server.app_ref._mount_pec_record_stop,
                        "PEC record stop")

        @flask_app.route("/api/mount/pec/clear", methods=["POST"])
        def api_mount_pec_clear():
            err = _require_connection()
            if err:
                return err
            return _run(server.app_ref._mount_pec_clear, "PEC clear")

        @flask_app.route("/api/mount/pec/save", methods=["POST"])
        def api_mount_pec_save():
            err = _require_connection()
            if err:
                return err
            return _run(server.app_ref._mount_pec_write_nv,
                        "PEC save to NV")

        @flask_app.route("/api/mount/pec/load", methods=["POST"])
        def api_mount_pec_load():
            err = _require_connection()
            if err:
                return err
            return _run(server.app_ref._mount_pec_read_nv,
                        "PEC load from NV")

        # ============================================================
        # OnStep Extended: Backlash
        # ============================================================

        @flask_app.route("/api/mount/backlash", methods=["POST"])
        def api_mount_backlash():
            err = _require_connection()
            if err:
                return err
            data = request.get_json(silent=True) or {}
            axis = data.get("axis", "ra")
            try:
                value = int(data.get("value", 0))
            except (ValueError, TypeError):
                return jsonify({"ok": False, "error": "Invalid value"}), 400
            if axis not in ("ra", "dec", "azm", "alt"):
                return jsonify({"ok": False, "error": f"Invalid axis: {axis}"}), 400
            return _run(lambda: server.app_ref._set_backlash(axis, value),
                        f"Backlash {axis}")

        @flask_app.route("/api/mount/backlash/get", methods=["GET"])
        def api_mount_backlash_get():
            err = _require_connection()
            if err:
                return err
            finished, exc = server._execute_and_wait(server.app_ref._get_backlash)
            if not finished:
                return jsonify({"ok": False, "error": "Timeout"}), 504
            if exc:
                return jsonify({"ok": False, "error": str(exc)}), 500
            return jsonify({
                "ok": True,
                "ra": server.app_ref.backlash_ra_var.get(),
                "dec": server.app_ref.backlash_dec_var.get(),
            })

        # ============================================================
        # OnStep Extended: Mount Limits
        # ============================================================

        @flask_app.route("/api/mount/limits", methods=["POST"])
        def api_mount_limits():
            err = _require_connection()
            if err:
                return err
            data = request.get_json(silent=True) or {}
            limit_type = data.get("type")  # 'horizon' or 'overhead'
            try:
                degrees = int(data.get("degrees", 0))
            except (ValueError, TypeError):
                return jsonify({"ok": False, "error": "Invalid degrees"}), 400
            if limit_type == "horizon":
                return _run(lambda: server.app_ref._set_horizon_limit(degrees),
                            "Horizon limit")
            elif limit_type == "overhead":
                return _run(lambda: server.app_ref._set_overhead_limit(degrees),
                            "Overhead limit")
            else:
                return jsonify({"ok": False, "error": "type must be 'horizon' or 'overhead'"}), 400

        @flask_app.route("/api/mount/limits/get", methods=["GET"])
        def api_mount_limits_get():
            err = _require_connection()
            if err:
                return err
            finished, exc = server._execute_and_wait(server.app_ref._get_limits)
            if not finished:
                return jsonify({"ok": False, "error": "Timeout"}), 504
            if exc:
                return jsonify({"ok": False, "error": str(exc)}), 500
            return jsonify({
                "ok": True,
                "horizon": server.app_ref.horizon_limit_var.get(),
                "overhead": server.app_ref.overhead_limit_var.get(),
            })

        # ============================================================
        # OnStep Extended: Auxiliary Features
        # ============================================================

        @flask_app.route("/api/mount/auxiliary/discover", methods=["POST"])
        def api_auxiliary_discover():
            err = _require_connection()
            if err:
                return err
            return _run(server.app_ref._discover_auxiliary_features,
                        "Discover auxiliary")

        @flask_app.route("/api/mount/auxiliary/set", methods=["POST"])
        def api_auxiliary_set():
            err = _require_connection()
            if err:
                return err
            data = request.get_json(silent=True) or {}
            try:
                slot = int(data.get("slot", 0))
                value = int(data.get("value", 0))
            except (ValueError, TypeError):
                return jsonify({"ok": False, "error": "Invalid slot/value"}), 400
            return _run(lambda: server.app_ref._set_auxiliary_value(slot, value),
                        f"Aux slot {slot}")

        @flask_app.route("/api/mount/auxiliary/refresh", methods=["POST"])
        def api_auxiliary_refresh():
            err = _require_connection()
            if err:
                return err
            return _run(server.app_ref._refresh_auxiliary_values,
                        "Refresh auxiliary")

        # ============================================================
        # OnStep Extended: Firmware Info
        # ============================================================

        @flask_app.route("/api/mount/firmware", methods=["GET"])
        def api_mount_firmware():
            app = server.app_ref
            return jsonify({
                "ok": True,
                "name": app.firmware_name_var.get(),
                "version": app.firmware_version_var.get(),
                "mount_type": app.firmware_mount_type_var.get(),
            })

        @flask_app.route("/api/mount/firmware/refresh", methods=["POST"])
        def api_mount_firmware_refresh():
            err = _require_connection()
            if err:
                return err
            return _run(server.app_ref._query_firmware_info,
                        "Firmware refresh")

        # ============================================================
        # OnStep Extended: Extended Focuser
        # ============================================================

        @flask_app.route("/api/focuser/goto", methods=["POST"])
        def api_focuser_goto():
            err = _require_connection()
            if err:
                return err
            data = request.get_json(silent=True) or {}
            try:
                position = int(data.get("position", 0))
            except (ValueError, TypeError):
                return jsonify({"ok": False, "error": "Invalid position"}), 400
            return _run(lambda: server.app_ref._focuser_goto(position),
                        "Focuser goto")

        @flask_app.route("/api/focuser/zero", methods=["POST"])
        def api_focuser_zero():
            err = _require_connection()
            if err:
                return err
            return _run(server.app_ref._focuser_zero, "Focuser zero")

        @flask_app.route("/api/focuser/home", methods=["POST"])
        def api_focuser_home():
            err = _require_connection()
            if err:
                return err
            return _run(server.app_ref._focuser_go_home, "Focuser home")

        @flask_app.route("/api/focuser/sethome", methods=["POST"])
        def api_focuser_sethome():
            err = _require_connection()
            if err:
                return err
            return _run(server.app_ref._focuser_set_home, "Focuser set home")

        @flask_app.route("/api/focuser/tcf", methods=["POST"])
        def api_focuser_tcf():
            err = _require_connection()
            if err:
                return err
            data = request.get_json(silent=True) or {}
            enabled = bool(data.get("enabled", False))
            return _run(lambda: server.app_ref._focuser_set_tcf(enabled),
                        "Focuser TCF")

        @flask_app.route("/api/focuser/select", methods=["POST"])
        def api_focuser_select():
            err = _require_connection()
            if err:
                return err
            data = request.get_json(silent=True) or {}
            try:
                num = int(data.get("focuser", 1))
            except (ValueError, TypeError):
                return jsonify({"ok": False, "error": "Invalid focuser number"}), 400
            if num < 1 or num > 6:
                return jsonify({"ok": False, "error": "Focuser must be 1-6"}), 400
            return _run(lambda: server.app_ref._focuser_select(num),
                        f"Select focuser {num}")

        # ============================================================
        # OnStep Extended: Rotator
        # ============================================================

        @flask_app.route("/api/rotator/move", methods=["POST"])
        def api_rotator_move():
            err = _require_connection()
            if err:
                return err
            data = request.get_json(silent=True) or {}
            direction = data.get("direction", "CW").upper()
            if direction not in ("CW", "CCW"):
                return jsonify({"ok": False, "error": "direction must be CW or CCW"}), 400
            return _run(lambda: server.app_ref._rotator_move(direction),
                        f"Rotator {direction}")

        @flask_app.route("/api/rotator/stop", methods=["POST"])
        def api_rotator_stop():
            err = _require_connection()
            if err:
                return err
            return _run(server.app_ref._rotator_stop, "Rotator stop")

        @flask_app.route("/api/rotator/goto", methods=["POST"])
        def api_rotator_goto():
            err = _require_connection()
            if err:
                return err
            data = request.get_json(silent=True) or {}
            try:
                angle = float(data.get("angle", 0))
            except (ValueError, TypeError):
                return jsonify({"ok": False, "error": "Invalid angle"}), 400
            return _run(lambda: server.app_ref._rotator_goto(angle),
                        "Rotator goto")

        @flask_app.route("/api/rotator/zero", methods=["POST"])
        def api_rotator_zero():
            err = _require_connection()
            if err:
                return err
            return _run(server.app_ref._rotator_zero, "Rotator zero")

        @flask_app.route("/api/rotator/derotate", methods=["POST"])
        def api_rotator_derotate():
            err = _require_connection()
            if err:
                return err
            return _run(server.app_ref._rotator_toggle_derotation,
                        "Rotator derotation toggle")

        @flask_app.route("/api/rotator/reverse", methods=["POST"])
        def api_rotator_reverse():
            err = _require_connection()
            if err:
                return err
            return _run(server.app_ref._rotator_reverse, "Rotator reverse")

        @flask_app.route("/api/rotator/parallactic", methods=["POST"])
        def api_rotator_parallactic():
            err = _require_connection()
            if err:
                return err
            return _run(server.app_ref._rotator_parallactic,
                        "Rotator parallactic")

        @flask_app.route("/api/rotator/rate", methods=["POST"])
        def api_rotator_rate():
            err = _require_connection()
            if err:
                return err
            data = request.get_json(silent=True) or {}
            try:
                rate = int(data.get("rate", 1))
            except (ValueError, TypeError):
                return jsonify({"ok": False, "error": "Invalid rate"}), 400
            return _run(lambda: server.app_ref._rotator_set_rate(rate),
                        "Rotator rate")

        # ============================================================
        # OnStep Extended: Reticle / LED
        # ============================================================

        @flask_app.route("/api/reticle/brighter", methods=["POST"])
        def api_reticle_brighter():
            err = _require_connection()
            if err:
                return err
            return _run(server.app_ref._reticle_brighter, "Reticle brighter")

        @flask_app.route("/api/reticle/dimmer", methods=["POST"])
        def api_reticle_dimmer():
            err = _require_connection()
            if err:
                return err
            return _run(server.app_ref._reticle_dimmer, "Reticle dimmer")

        # ============================================================
        # OnStep Extended: Home (improved)
        # ============================================================

        @flask_app.route("/api/home/find", methods=["POST"])
        def api_home_find():
            err = _require_connection()
            if err:
                return err
            return _run(server.app_ref._home_find, "Find home")

        # ============================================================
        # Initialize: Set Time/Date, Set Site, Send Weather (individual)
        # ============================================================

        @flask_app.route("/api/mount/set-time", methods=["POST"])
        def api_mount_set_time():
            """Send current local date + time to the mount."""
            err = _require_connection()
            if err:
                return err
            def _do():
                details = server.app_ref._send_time_to_telescope()
                return {"ok": True, "details": details}
            return _run(_do, "Set time/date")

        @flask_app.route("/api/mount/set-site", methods=["POST"])
        def api_mount_set_site():
            """Send just observer lat/lon + UTC offset to the mount."""
            err = _require_connection()
            if err:
                return err
            def _do():
                details = server.app_ref._send_site_to_telescope()
                return {"ok": True, "details": details}
            return _run(_do, "Set site")

        @flask_app.route("/api/mount/set-weather", methods=["POST"])
        def api_mount_set_weather():
            """Send weather data (temp/pressure/humidity) to the mount."""
            err = _require_connection()
            if err:
                return err
            def _do():
                details = server.app_ref._send_weather_to_telescope()
                return {"ok": True, "details": details}
            return _run(_do, "Set weather")

        # ============================================================
        # Logging
        # ============================================================

        @flask_app.route("/api/log/level", methods=["POST"])
        def api_log_level():
            """Change log level at runtime. Body: {"level": "DEBUG"|"INFO"|"WARNING"|"ERROR"}"""
            import logging as _logging
            from telescope_logger import set_log_level as _set_log_level
            data = request.get_json(silent=True) or {}
            level_str = data.get("level", "").upper()
            level_map = {"DEBUG": _logging.DEBUG, "INFO": _logging.INFO,
                         "WARNING": _logging.WARNING, "ERROR": _logging.ERROR}
            level = level_map.get(level_str)
            if level is None:
                return jsonify({"ok": False, "error": f"Unknown level '{level_str}'"}), 400
            _set_log_level(console_level=level, file_level=level)
            server.push_log(f"[WebUI] Log level changed to {level_str}", "server")
            return jsonify({"ok": True, "level": level_str})

        # ============================================================
        # Weather
        # ============================================================

        @flask_app.route("/api/weather/refresh", methods=["POST"])
        def api_weather_refresh():
            finished, exc = server._execute_and_wait(server.app_ref._update_weather)
            if not finished:
                return jsonify({"ok": False, "error": "Weather refresh: timeout"}), 504
            if exc:
                return jsonify({"ok": False, "error": f"Weather refresh: {exc}"}), 500
            return jsonify({"ok": True})

        # ============================================================
        # Location (GPS / observer position)
        # ============================================================

        @flask_app.route("/api/location", methods=["POST"])
        def api_location():
            data = request.get_json(silent=True) or {}
            lat = data.get("latitude")
            lon = data.get("longitude")
            if lat is None or lon is None:
                return jsonify({"ok": False, "error": "Provide latitude and longitude"}), 400
            try:
                lat = float(lat)
                lon = float(lon)
            except (ValueError, TypeError):
                return jsonify({"ok": False, "error": "Invalid coordinate values"}), 400
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                return jsonify({"ok": False, "error": "Coordinates out of range"}), 400

            app = server.app_ref

            def set_location():
                app.lat_var.set(str(lat))
                app.lon_var.set(str(lon))
                # Update protocol
                proto = getattr(app, "protocol", None)
                if proto:
                    proto.latitude = lat
                    proto.longitude = lon
                # Save to config
                cm = getattr(app, "config_manager", None)
                if cm:
                    try:
                        cm.set("location.latitude", lat)
                        cm.set("location.longitude", lon)
                        cm.save_config()
                    except Exception:
                        pass

            finished, exc = server._execute_and_wait(set_location)
            if not finished:
                return jsonify({"ok": False, "error": "Set location: timeout"}), 504
            if exc:
                return jsonify({"ok": False, "error": f"Set location: {exc}"}), 500

            # Refresh weather synchronously so the user sees data immediately
            # after saving location (the HTTP call takes ~1-2s max).
            weather_ok = False
            if hasattr(app, "_update_weather"):
                try:
                    w_done, w_exc = server._execute_and_wait(
                        app._update_weather, timeout=10
                    )
                    weather_ok = w_done and not w_exc
                except Exception:
                    pass

            return jsonify({
                "ok": True,
                "latitude": lat,
                "longitude": lon,
                "weather_refreshed": weather_ok,
            })

        # ============================================================
        # Solver settings (mode, API key, timeout)
        # ============================================================
        #
        # The solver settings control how plate solving works:
        #   - mode: "auto" (ASTAP first, cloud fallback), "astap" (local only),
        #           "cloud" (Astrometry.net only).  Legacy "local" → "astap".
        #   - cloud_api_key: Optional API key for nova.astrometry.net
        #                    (free, speeds up queue priority)
        #   - timeout: Max seconds for a single solve attempt (default 120).
        #              Applies to both ASTAP local and cloud solvers.
        #
        # Config keys in telescope_config.json:
        #   solver.mode, solver.cloud_api_key, solver.timeout

        @flask_app.route("/api/solver/settings", methods=["GET", "POST"])
        def api_solver_settings():
            """Get or set plate-solver settings (mode, API key, timeout, optics)."""
            import math
            app = server.app_ref
            cm = getattr(app, "config_manager", None)

            if request.method == "GET":
                api_key = ""
                solver_mode = "auto"
                solver_timeout = 120
                focal_length_mm = 0
                sensor_width_mm = 0
                if cm:
                    try:
                        api_key = cm.get("solver.cloud_api_key", "")
                        solver_mode = cm.get("solver.mode", "auto")
                        solver_timeout = int(cm.get("solver.timeout", 120))
                        focal_length_mm = float(cm.get("solver.focal_length_mm", 0))
                        sensor_width_mm = float(cm.get("solver.sensor_width_mm", 0))
                    except Exception:
                        pass

                # Calculate FOV from optics if configured
                calculated_fov = 0.0
                if focal_length_mm > 0 and sensor_width_mm > 0:
                    calculated_fov = round(
                        2.0 * math.degrees(math.atan(sensor_width_mm / (2.0 * focal_length_mm))),
                        4
                    )

                # Get last solved FOV and recommended database
                last_solved_fov = 0.0
                recommended_db = ""
                try:
                    from android_bridge.local_solver import get_last_solved_fov, recommend_database
                    last_solved_fov = get_last_solved_fov()
                    effective_fov = last_solved_fov if last_solved_fov > 0 else (calculated_fov if calculated_fov > 0 else 5.0)
                    recommended_db = recommend_database(effective_fov)
                except ImportError:
                    pass  # desktop -- no local_solver

                return jsonify({
                    "ok": True,
                    "cloud_api_key": api_key,
                    "mode": solver_mode,
                    "timeout": solver_timeout,
                    "focal_length_mm": focal_length_mm,
                    "sensor_width_mm": sensor_width_mm,
                    "calculated_fov": calculated_fov,
                    "last_solved_fov": last_solved_fov,
                    "recommended_db": recommended_db,
                })

            # POST -- save settings (with defensive null/type handling)
            data = request.get_json(silent=True) or {}
            raw_key = data.get("cloud_api_key")
            api_key = str(raw_key).strip() if raw_key is not None else ""
            raw_mode = data.get("mode")
            solver_mode = str(raw_mode).strip().lower() if raw_mode is not None else "auto"
            # Clamp timeout to sensible range: 10-600 seconds
            try:
                solver_timeout = max(10, min(600, int(data.get("timeout", 120))))
            except (ValueError, TypeError):
                solver_timeout = 120  # Default on invalid input
            if solver_mode not in ("auto", "cloud", "astap"):
                # Normalize legacy "local" to "astap"
                if solver_mode == "local":
                    solver_mode = "astap"
                else:
                    solver_mode = "auto"

            # Parse optics settings (0 = not configured)
            try:
                focal_length_mm = max(0, float(data.get("focal_length_mm", 0)))
            except (ValueError, TypeError):
                focal_length_mm = 0
            try:
                sensor_width_mm = max(0, float(data.get("sensor_width_mm", 0)))
            except (ValueError, TypeError):
                sensor_width_mm = 0

            if cm:
                try:
                    cm.set("solver.cloud_api_key", api_key)
                    cm.set("solver.mode", solver_mode)
                    cm.set("solver.timeout", solver_timeout)
                    cm.set("solver.focal_length_mm", focal_length_mm)
                    cm.set("solver.sensor_width_mm", sensor_width_mm)
                    cm.save_config()
                except Exception as e:
                    return jsonify({"ok": False, "error": str(e)}), 500

            # Hot-reload the API key into the running cloud solver singleton
            # (avoids requiring an app restart to apply a new API key)
            try:
                from android_bridge.cloud_solver import _solver
                import android_bridge.cloud_solver as cs
                if cs._solver is not None:
                    cs._solver.api_key = api_key
                    cs._solver.session_key = None  # force re-login with new key
            except ImportError:
                pass  # desktop -- no cloud solver loaded

            # Calculate the resulting FOV for the response
            calculated_fov = 0.0
            if focal_length_mm > 0 and sensor_width_mm > 0:
                calculated_fov = round(
                    2.0 * math.degrees(math.atan(sensor_width_mm / (2.0 * focal_length_mm))),
                    4
                )

            return jsonify({
                "ok": True,
                "cloud_api_key": api_key,
                "mode": solver_mode,
                "timeout": solver_timeout,
                "focal_length_mm": focal_length_mm,
                "sensor_width_mm": sensor_width_mm,
                "calculated_fov": calculated_fov,
            })

        # ============================================================
        # ASTAP star database management (Android only)
        # ============================================================
        #
        # These endpoints manage the ASTAP star databases required for
        # local (offline) plate solving.  Databases are downloaded from
        # SourceForge and extracted to the app's external files directory.
        #
        # Endpoints:
        #   GET  /api/solver/databases          -- List all databases + status
        #   POST /api/solver/databases/download -- Start downloading a database
        #   POST /api/solver/databases/delete   -- Delete an installed database
        #   GET  /api/solver/databases/progress -- SSE stream of download progress
        #
        # On desktop (non-Android), these return graceful "not available"
        # responses since ASTAP databases are Android-only.
        #
        # The SSE progress endpoint replaces the old polling approach
        # (which polled /api/solver/databases every 2 seconds) with a
        # real-time push stream using Server-Sent Events.  The browser
        # connects once via EventSource and receives progress updates
        # every 500ms until the download completes.

        @flask_app.route("/api/solver/databases", methods=["GET"])
        def api_solver_databases():
            """Get ASTAP star database status (installed, available, paths).

            Returns JSON:
              ok: bool
              databases: {name: {installed, size_mb, description, min_fov, max_fov, path}}
              installed: str or null (name of the first installed database)
            """
            try:
                from android_bridge.local_solver import get_database_status
                status = get_database_status()
                installed = None
                for name, info in status.items():
                    if info.get("installed"):
                        installed = name
                        break
                return jsonify({"ok": True, "databases": status, "installed": installed})
            except ImportError:
                # Desktop -- ASTAP native binary not available
                return jsonify({"ok": True, "databases": {}, "installed": None,
                                "message": "ASTAP databases only available on Android"})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500

        @flask_app.route("/api/solver/databases/download", methods=["POST"])
        def api_solver_database_download():
            """Start downloading an ASTAP star database (non-blocking).

            Request JSON: {"db_name": "d05"}  (d05, d20, d50, or w08)

            The download runs in a background thread.  Monitor progress via
            the SSE endpoint: GET /api/solver/databases/progress
            """
            data = request.get_json(silent=True) or {}
            db_name = str(data.get("db_name", "d05")).strip().lower()
            # Validate database name before attempting download
            valid_dbs = ("d05", "d20", "d50", "w08")
            if db_name not in valid_dbs:
                return jsonify({"ok": False, "error": f"Unknown database '{db_name}'. Valid: {', '.join(valid_dbs)}"}), 400
            try:
                from android_bridge.local_solver import download_database, is_database_installed
                if is_database_installed(db_name):
                    return jsonify({"ok": True, "message": f"{db_name} already installed"})
                started = download_database(db_name)
                if started:
                    return jsonify({"ok": True, "message": f"Downloading {db_name}..."})
                return jsonify({"ok": False, "error": "Failed to start download"}), 500
            except ImportError:
                return jsonify({"ok": False, "error": "ASTAP not available on this platform"}), 400
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500

        @flask_app.route("/api/solver/databases/delete", methods=["POST"])
        def api_solver_database_delete():
            """Delete an installed ASTAP star database to free storage.

            Request JSON: {"db_name": "d05"}
            """
            data = request.get_json(silent=True) or {}
            db_name = str(data.get("db_name", "")).strip().lower()
            if not db_name:
                return jsonify({"ok": False, "error": "db_name required"}), 400
            try:
                from android_bridge.local_solver import delete_database
                success = delete_database(db_name)
                return jsonify({"ok": success})
            except ImportError:
                return jsonify({"ok": False, "error": "ASTAP not available"}), 400
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500

        @flask_app.route("/api/solver/databases/progress")
        def api_solver_database_progress():
            """Server-Sent Events (SSE) stream for database download progress.

            The browser connects via:
              const evtSource = new EventSource('/api/solver/databases/progress');
              evtSource.onmessage = (e) => { const data = JSON.parse(e.data); ... };

            Each SSE event is a JSON object:
              {
                "state": "downloading"|"extracting"|"complete"|"error"|"idle",
                "bytes_downloaded": 12345678,
                "bytes_total": 47185920,
                "extracted_files": 5,
                "error": "",
                "db_name": "d05"
              }

            The stream sends events every ~500ms and automatically closes
            when the download completes, errors, or is idle.

            Architecture:
              Browser EventSource <-- SSE events <-- this Flask generator
                reads from --> local_solver.get_download_progress()
                  reads from --> AstapDatabaseManager.kt @Volatile fields
                    updated by --> download background thread
            """
            import json as _json

            def _generate_progress():
                try:
                    from android_bridge.local_solver import get_download_progress
                except ImportError:
                    # Desktop fallback -- no ASTAP available
                    yield "data: " + _json.dumps({
                        "state": "error",
                        "error": "ASTAP not available on this platform",
                        "bytes_downloaded": 0, "bytes_total": 0,
                        "extracted_files": 0, "db_name": ""
                    }) + "\n\n"
                    return

                import time as _time
                idle_count = 0
                while True:
                    progress = get_download_progress()
                    yield "data: " + _json.dumps(progress) + "\n\n"

                    state = progress.get("state", "idle")
                    # Stop streaming on terminal states
                    if state in ("complete", "error"):
                        break
                    # If idle for too long (10 seconds), stop streaming
                    # (client can reconnect if needed)
                    if state == "idle":
                        idle_count += 1
                        if idle_count > 20:
                            break
                    else:
                        idle_count = 0

                    _time.sleep(0.5)

            return Response(
                _generate_progress(),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",  # Disable nginx buffering
                    "Connection": "keep-alive",
                },
            )

        # ============================================================
        # IP-based geolocation (works over HTTP, no HTTPS needed)
        # ============================================================

        @flask_app.route("/api/gps")
        def api_gps():
            """Get approximate location via IP geolocation.

            Uses free ip-api.com service as a fallback when browser
            Geolocation API is blocked (requires HTTPS).
            """
            try:
                import requests as req
                resp = req.get("http://ip-api.com/json/?fields=status,lat,lon,city,regionName,country",
                               timeout=5)
                data = resp.json()
                if data.get("status") == "success":
                    return jsonify({
                        "ok": True,
                        "latitude": data["lat"],
                        "longitude": data["lon"],
                        "city": data.get("city", ""),
                        "region": data.get("regionName", ""),
                        "country": data.get("country", ""),
                        "source": "ip"
                    })
                return jsonify({"ok": False, "error": "IP geolocation failed"})
            except ImportError:
                return jsonify({"ok": False, "error": "requests module not installed"})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)})

        # ============================================================
        # ASCOM Camera listing and selection
        # ============================================================

        @flask_app.route("/api/ascom/cameras")
        def api_ascom_cameras():
            """List all ASCOM-registered camera devices."""
            try:
                from auto_platesolve import AutoPlateSolver
                cameras = AutoPlateSolver.list_ascom_cameras()
                return jsonify({"ok": True, "cameras": cameras})
            except ImportError:
                return jsonify({"ok": True, "cameras": []})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e), "cameras": []})

        @flask_app.route("/api/ascom/select", methods=["POST"])
        def api_ascom_select():
            """Select an ASCOM camera by ProgID."""
            data = request.get_json(silent=True) or {}
            camera_id = data.get("camera_id", "").strip()
            if not camera_id:
                return jsonify({"ok": False, "error": "No camera_id provided"}), 400

            app = server.app_ref
            name = camera_id.split('.')[-1] if '.' in camera_id else camera_id

            def set_camera():
                app.ascom_camera_id_var.set(camera_id)
                app.ascom_camera_name_var.set(name)

            finished, exc = server._execute_and_wait(set_camera)
            if not finished:
                return jsonify({"ok": False, "error": "ASCOM select: timeout"}), 504
            if exc:
                return jsonify({"ok": False, "error": f"ASCOM select: {exc}"}), 500
            return jsonify({"ok": True, "camera_id": camera_id, "name": name})

        # ============================================================
        # Camera / Solve settings
        # ============================================================

        @flask_app.route("/api/camera/settings", methods=["POST"])
        def api_camera_settings():
            data = request.get_json(silent=True) or {}
            app = server.app_ref

            def apply():
                if "solve_mode" in data:
                    mode = data["solve_mode"]
                    # Accept both desktop modes and Android camera sources
                    if mode in ("camera", "ascom", "folder", "manual",
                                "auto", "zwo", "uvc", "phone"):
                        # On Android, map source names to "camera" mode
                        # since the actual camera selection happens via
                        # android_source hint, not the solve_mode var.
                        if mode in ("auto", "zwo", "uvc", "phone"):
                            app.solve_mode_var.set("camera")
                        else:
                            app.solve_mode_var.set(mode)
                if "android_source" in data:
                    # Store the Android camera source hint on the app
                    # so _android_start_camera_mode can read it
                    app._android_camera_source = str(data["android_source"])
                if "camera_index" in data:
                    app.camera_index_var.set(str(data["camera_index"]))
                if "ascom_exposure" in data:
                    app.ascom_exposure_var.set(str(data["ascom_exposure"]))
                if "ascom_gain" in data:
                    app.ascom_gain_var.set(str(data["ascom_gain"]))
                if "ascom_binning" in data:
                    app.ascom_binning_var.set(str(data["ascom_binning"]))
                if "watch_folder" in data:
                    app.watch_folder_var.set(str(data["watch_folder"]))
                if "solve_interval" in data:
                    app.solve_interval_var.set(str(data["solve_interval"]))

            finished, exc = server._execute_and_wait(apply)
            if not finished:
                return jsonify({"ok": False, "error": "Camera settings: timeout"}), 504
            if exc:
                return jsonify({"ok": False, "error": f"Camera settings: {exc}"}), 500
            return jsonify({"ok": True})

        # ============================================================
        # Connection settings (type, port, baudrate, wifi)
        # ============================================================

        @flask_app.route("/api/connection/settings", methods=["POST"])
        def api_connection_settings():
            data = request.get_json(silent=True) or {}
            app = server.app_ref

            def apply():
                if "type" in data:
                    t = data["type"]
                    if t in ("USB", "WiFi"):
                        app.connection_type_var.set(t)
                if "port" in data:
                    app.usb_port_var.set(str(data["port"]))
                if "baudrate" in data:
                    app.usb_baudrate_var.set(str(data["baudrate"]))
                if "wifi_ip" in data:
                    app.wifi_ip_var.set(str(data["wifi_ip"]))
                if "wifi_port" in data:
                    app.wifi_port_var.set(str(data["wifi_port"]))
                if "protocol" in data:
                    proto = str(data["protocol"]).lower()
                    if proto in ("lx200", "nexstar", "ioptron", "audiostar", "alpaca", "indi"):
                        app.mount_protocol_var.set(proto)
                        # Apply to the bridge
                        app.telescope_bridge.set_protocol(proto)

            finished, exc = server._execute_and_wait(apply)
            if not finished:
                return jsonify({"ok": False, "error": "Connection settings: timeout"}), 504
            if exc:
                return jsonify({"ok": False, "error": f"Connection settings: {exc}"}), 500
            return jsonify({"ok": True})

        # ============================================================
        # Auto-connect preference
        # ============================================================

        @flask_app.route("/api/autoconnect", methods=["GET"])
        def api_autoconnect_get():
            """Return auto-connect preference and saved connection settings."""
            cfg = server.app_ref.config_manager
            return jsonify({
                "ok": True,
                "enabled": cfg.get("connection.auto_connect", False),
                "type": cfg.get("connection.type", "wifi"),
                "wifi_ip": cfg.get("connection.wifi_ip", ""),
                "wifi_port": cfg.get("connection.wifi_port", ""),
                "baudrate": cfg.get("connection.baudrate", 9600),
            })

        @flask_app.route("/api/autoconnect", methods=["POST"])
        def api_autoconnect_set():
            """Save auto-connect preference. Also saves current conn settings."""
            data = request.get_json(silent=True) or {}
            cfg = server.app_ref.config_manager
            enabled = bool(data.get("enabled", False))
            cfg.set("connection.auto_connect", enabled)
            # Save current connection settings alongside the preference
            if enabled:
                cfg.set("connection.type",
                        data.get("type", "wifi"))
                cfg.set("connection.wifi_ip",
                        data.get("wifi_ip", ""))
                cfg.set("connection.wifi_port",
                        data.get("wifi_port", ""))
                cfg.set("connection.baudrate",
                        int(data.get("baudrate", 9600)))
            return jsonify({"ok": True, "enabled": enabled})

        # ============================================================
        # Serial port enumeration
        # ============================================================

        @flask_app.route("/api/serial/ports")
        def api_serial_ports():
            """Return detected serial ports for the port dropdown."""
            bridge = getattr(server.app_ref, "telescope_bridge", None)
            if not bridge:
                return jsonify({"ok": True, "ports": [], "detected": []})
            all_ports = bridge.get_available_ports()
            # Also return just the auto-detected subset so the UI can
            # highlight them at the top of the dropdown.
            detected = getattr(bridge, "_last_detected_ports", [])
            return jsonify({
                "ok": True,
                "ports": all_ports,
                "detected": detected,
            })

        # ============================================================
        # Catalog search
        # ============================================================

        @flask_app.route("/api/catalog/search")
        def api_catalog_search():
            q = request.args.get("q", "").strip().lower()
            if len(q) < 1:
                return jsonify({"results": []})
            try:
                from catalog_loader import load_all_catalogs, get_solar_system_objects
                catalogs = load_all_catalogs()
            except ImportError:
                return jsonify({"results": []})

            # Merge solar system objects (computed fresh -- they move)
            try:
                solar = get_solar_system_objects()
            except Exception:
                solar = {}

            # Get observer location for Alt/Az computation
            app = server.app_ref
            lat = _safe_float(getattr(app, "lat_var", None), 0.0)
            lon = _safe_float(getattr(app, "lon_var", None), 0.0)

            matches = []
            # Search solar system first (higher priority for common names)
            for name, (ra_h, dec_d) in solar.items():
                if q in name:
                    alt, az = _catalog_ra_dec_to_alt_az(ra_h, dec_d, lat, lon)
                    matches.append({
                        "name": name,
                        "ra_hours": round(ra_h, 4),
                        "dec_degrees": round(dec_d, 4),
                        "alt_deg": round(alt, 1),
                        "az_deg": round(az, 1),
                    })
                    if len(matches) >= 20:
                        break
            # Then search fixed catalogs
            if len(matches) < 20:
                for name, (ra_h, dec_d) in catalogs.items():
                    if q in name:
                        alt, az = _catalog_ra_dec_to_alt_az(ra_h, dec_d, lat, lon)
                        matches.append({
                            "name": name,
                            "ra_hours": round(ra_h, 4),
                            "dec_degrees": round(dec_d, 4),
                            "alt_deg": round(alt, 1),
                            "az_deg": round(az, 1),
                        })
                        if len(matches) >= 20:
                            break
            return jsonify({"results": matches})

        # ============================================================
        # Catalog browse (by category)
        # ============================================================

        @flask_app.route("/api/catalog/browse")
        def api_catalog_browse():
            """Return catalog objects organized by category, or a single category."""
            category = request.args.get("cat", "").strip()
            try:
                from catalog_loader import get_catalog_categories, _build_solar_system_list
                categories = get_catalog_categories()
            except ImportError:
                return jsonify({"categories": [], "objects": []})

            # Inject "Solar System" category (always computed fresh)
            try:
                categories["Solar System"] = _build_solar_system_list()
            except Exception:
                pass

            if not category:
                # Return list of available categories with object counts
                # Put Solar System first
                cat_list = []
                if "Solar System" in categories:
                    cat_list.append({"name": "Solar System",
                                     "count": len(categories["Solar System"])})
                for name, objects in categories.items():
                    if name == "Solar System":
                        continue
                    cat_list.append({"name": name, "count": len(objects)})
                return jsonify({"categories": cat_list})

            # Return objects for the requested category
            if category not in categories:
                return jsonify({"error": f"Unknown category: {category}",
                                "categories": list(categories.keys())}), 404

            page = request.args.get("page", "1")
            try:
                page = max(1, int(page))
            except ValueError:
                page = 1
            per_page = 50
            objects = categories[category]
            start = (page - 1) * per_page
            end = start + per_page
            page_objects = objects[start:end]

            # Enrich with Alt/Az
            app = server.app_ref
            lat = _safe_float(getattr(app, "lat_var", None), 0.0)
            lon = _safe_float(getattr(app, "lon_var", None), 0.0)
            for obj in page_objects:
                alt, az = _catalog_ra_dec_to_alt_az(
                    obj["ra_hours"], obj["dec_degrees"], lat, lon
                )
                obj["alt_deg"] = round(alt, 1)
                obj["az_deg"] = round(az, 1)

            return jsonify({
                "category": category,
                "total": len(objects),
                "page": page,
                "per_page": per_page,
                "objects": page_objects,
            })

        # ============================================================
        # Sky Chart data
        # ============================================================

        _skychart_cache = {"data": None, "ts": 0}

        @flask_app.route("/api/skychart/data")
        def api_skychart_data():
            """Return full sky chart dataset (stars, DSOs, constellations, planets)."""
            import time as _time
            now = _time.time()
            # Cache for 30s (planets move slowly enough)
            if _skychart_cache["data"] and now - _skychart_cache["ts"] < 30:
                cached = _skychart_cache["data"]
            else:
                try:
                    from catalog_loader import get_skychart_data
                    cached = get_skychart_data()
                    _skychart_cache["data"] = cached
                    _skychart_cache["ts"] = now
                except Exception as exc:
                    _logger.warning("Sky chart data error: %s", exc)
                    return jsonify({"error": str(exc)}), 500

            # Always refresh: observer location + telescope position + planets
            app = server.app_ref
            lat = _safe_float(getattr(app, "lat_var", None), 0.0)
            lon = _safe_float(getattr(app, "lon_var", None), 0.0)

            # Current telescope position (from nested 'position' dict in _state)
            scope = None
            try:
                state = server._state
                pos = state.get("position", {})
                alt = pos.get("alt_degrees", 0)
                az = pos.get("az_degrees", 0)
                ra = pos.get("ra_hours", 0)
                dec = pos.get("dec_degrees", 0)
                if alt != 0 or az != 0 or ra != 0 or dec != 0:
                    scope = {"alt": alt, "az": az, "ra": ra, "dec": dec}
            except Exception:
                pass

            # Refresh planets (they move)
            planets = []
            try:
                from catalog_loader import get_solar_system_objects
                ss = get_solar_system_objects()
                for disp, key, spec in [
                    ("Sun","sun","G"),("Moon","moon",None),
                    ("Mercury","mercury",None),("Venus","venus",None),
                    ("Mars","mars","K"),("Jupiter","jupiter",None),
                    ("Saturn","saturn",None),("Uranus","uranus",None),
                    ("Neptune","neptune",None),
                ]:
                    if key in ss:
                        ra_h, dec_d = ss[key]
                        planets.append({'n':disp,'r':round(ra_h,4),'d':round(dec_d,2),'s':spec})
            except Exception:
                pass

            return jsonify({
                "stars": cached.get("stars", []),
                "ext_stars": cached.get("ext_stars", []),
                "dsos": cached.get("dsos", []),
                "planets": planets,
                "con_lines": cached.get("con_lines", []),
                "con_labels": cached.get("con_labels", []),
                "observer": {"lat": lat, "lon": lon},
                "telescope": scope,
            })

        # ============================================================
        # Auto-alignment
        # ============================================================

        @flask_app.route("/api/alignment/start", methods=["POST"])
        def api_alignment_start():
            """Start the alignment procedure (auto or manual mode)."""
            err = _require_connection()
            if err: return err
            data = request.get_json(silent=True) or {}
            num_stars = data.get("num_stars", 6)
            mode = data.get("mode", "auto")
            try:
                num_stars = int(num_stars)
            except (TypeError, ValueError):
                num_stars = 6
            if mode not in ("auto", "manual"):
                mode = "auto"
            app = server.app_ref
            aligner = getattr(app, "auto_alignment", None)
            if aligner is None:
                return jsonify({"ok": False, "error": "Alignment not available"}), 500
            if aligner.is_running():
                return jsonify({"ok": False, "error": "Alignment already running"}), 400
            ok = aligner.start(num_stars, mode=mode)
            return jsonify({"ok": ok, "num_stars": num_stars, "mode": mode})

        @flask_app.route("/api/alignment/abort", methods=["POST"])
        def api_alignment_abort():
            """Abort the running alignment procedure."""
            err = _require_connection()
            if err: return err
            app = server.app_ref
            aligner = getattr(app, "auto_alignment", None)
            if aligner is None:
                return jsonify({"ok": False, "error": "Alignment not available"}), 500
            aligner.abort()
            return jsonify({"ok": True})

        @flask_app.route("/api/alignment/status")
        def api_alignment_status():
            """Return current alignment progress."""
            app = server.app_ref
            aligner = getattr(app, "auto_alignment", None)
            if aligner is None:
                return jsonify({"running": False, "phase": "unavailable"})
            return jsonify(aligner.get_status())

        @flask_app.route("/api/alignment/manual/sync", methods=["POST"])
        def api_alignment_manual_sync():
            """Manual mode: user confirms star is centered, sync the mount."""
            err = _require_connection()
            if err: return err
            app = server.app_ref
            aligner = getattr(app, "auto_alignment", None)
            if aligner is None:
                return jsonify({"ok": False, "error": "Alignment not available"}), 500
            ok = aligner.manual_confirm_sync()
            if not ok:
                return jsonify({"ok": False, "error": "Not waiting for user action"}), 400
            return jsonify({"ok": True})

        @flask_app.route("/api/alignment/manual/recenter", methods=["POST"])
        def api_alignment_manual_recenter():
            """Manual mode: re-slew to the current alignment star."""
            err = _require_connection()
            if err: return err
            app = server.app_ref
            aligner = getattr(app, "auto_alignment", None)
            if aligner is None:
                return jsonify({"ok": False, "error": "Alignment not available"}), 500
            ok = aligner.manual_recenter()
            if not ok:
                return jsonify({"ok": False, "error": "Not waiting for user action"}), 400
            return jsonify({"ok": True})

        @flask_app.route("/api/alignment/manual/skip", methods=["POST"])
        def api_alignment_manual_skip():
            """Manual mode: skip the current alignment star."""
            err = _require_connection()
            if err: return err
            app = server.app_ref
            aligner = getattr(app, "auto_alignment", None)
            if aligner is None:
                return jsonify({"ok": False, "error": "Alignment not available"}), 500
            ok = aligner.manual_skip_star()
            if not ok:
                return jsonify({"ok": False, "error": "Not waiting for user action"}), 400
            return jsonify({"ok": True})

        # ============================================================
        # Camera live-view streaming
        # ============================================================

        @flask_app.route("/api/camera/start", methods=["POST"])
        def api_camera_start():
            """Open a camera (UVC, ASCOM, or ASI SDK) for MJPEG streaming.

            JSON body fields:
                source: ``"uvc"`` (default), ``"ascom"``, or ``"asi"``
                camera_index: Device index (UVC/ASI mode, default 0)
                ascom_id: ASCOM ProgID (ASCOM mode, required)
                exposure: ASCOM exposure seconds (default 0.5)
                gain: ASCOM gain (default 100)
                binning: ASCOM binning (default 2)
            """
            data = request.get_json(silent=True) or {}
            source = data.get("source", "uvc")

            if source == "ascom":
                ascom_id = data.get("ascom_id", "").strip()
                if not ascom_id:
                    return jsonify({"ok": False, "error": "No ascom_id provided"}), 400
                if not _HAS_ASCOM:
                    return jsonify({"ok": False, "error": "ASCOM not available (Windows + pywin32 required)"}), 500
                exposure = float(data.get("exposure", 0.5))
                gain = int(data.get("gain", 100))
                binning = int(data.get("binning", 2))
                ok = server._open_camera(
                    source="ascom", ascom_id=ascom_id,
                    exposure=exposure, gain=gain, binning=binning,
                )
                return jsonify({"ok": ok, "source": "ascom", "ascom_id": ascom_id})
            elif source == "asi":
                idx = int(data.get("camera_index", 0))
                ok = server._open_camera(index=idx, source="asi")
                return jsonify({"ok": ok, "source": "asi", "camera_index": idx})
            else:
                idx = int(data.get("camera_index", 0))
                android_src = data.get("android_source", "")
                if not _HAS_CV2 and not android_src:
                    return jsonify({"ok": False, "error": "OpenCV not installed"}), 500
                ok = server._open_camera(index=idx, source="uvc",
                                         android_source=android_src)
                return jsonify({"ok": ok, "source": "uvc",
                                "camera_index": idx,
                                "android_source": android_src})

        @flask_app.route("/api/camera/stop", methods=["POST"])
        def api_camera_stop():
            """Close the active camera stream (UVC or ASCOM)."""
            server._close_camera()
            return jsonify({"ok": True})

        @flask_app.route("/api/camera/stream")
        def api_camera_stream():
            """MJPEG stream (use as <img> src or fetch).

            Works for both UVC and ASCOM camera sources.
            """
            if not server._camera_active:
                return "Camera not active", 503
            return Response(
                server._generate_mjpeg(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @flask_app.route("/api/camera/snapshot")
        def api_camera_snapshot():
            """Return a single JPEG frame from the active camera (UVC or ASCOM)."""
            if not server._camera_active:
                return "Camera not active", 503

            jpeg = None
            if server._camera_source == "ascom":
                # ASCOM: capture one frame and return JPEG
                jpeg = server._capture_ascom_frame_jpeg()
                if jpeg is None:
                    return "Failed to capture ASCOM frame", 503
            elif server._camera_source == "asi":
                # ASI SDK: grab latest JPEG from streaming buffer
                asi = _get_asi_module()
                if asi and hasattr(asi, '_last_jpeg') and asi._last_jpeg:  # type: ignore
                    jpeg = asi._last_jpeg  # type: ignore
                elif asi and hasattr(asi, 'capture_for_solving'):
                    jpeg = asi.capture_for_solving()
                if jpeg is None:
                    return "Failed to capture ASI frame", 503
            else:
                # UVC: grab from OpenCV
                if not _HAS_CV2:
                    return "OpenCV not installed", 503
                with server._camera_lock:
                    if server._camera is None or not server._camera.isOpened():
                        return "Camera not active", 503
                    ret, frame = server._camera.read()
                if not ret or frame is None:
                    return "Failed to capture frame", 503
                ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])  # type: ignore
                if not ok:
                    return "Encoding failed", 500
                jpeg = buf.tobytes()
            # Return with Content-Disposition: attachment so Android
            # WebView DownloadListener triggers a proper file save.
            from datetime import datetime as _dt
            fname = f"telescope_{_dt.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            resp = Response(jpeg, mimetype="image/jpeg")
            resp.headers["Content-Disposition"] = f'attachment; filename="{fname}"'
            return resp

        @flask_app.route("/api/camera/status")
        def api_camera_status():
            """Return camera streaming state and capabilities."""
            result = {
                "active": server._camera_active,
                "source": server._camera_source,
                "camera_index": server._camera_index,
                "cv2_available": _HAS_CV2,
                "ascom_available": _HAS_ASCOM,
                "ascom_exposure": server._ascom_exposure,
                "ascom_gain": server._ascom_gain,
                "ascom_binning": server._ascom_binning,
            }
            android_src = getattr(server, '_android_source', '')
            if android_src:
                result["android_source"] = android_src
            return jsonify(result)

        @flask_app.route("/api/camera/ascom/settings", methods=["POST"])
        def api_camera_ascom_settings():
            """Update ASCOM live-view settings while streaming.

            JSON body fields (all optional):
                exposure: new exposure time in seconds
                gain: new gain value
                binning: new binning factor (requires reconnect)
            """
            data = request.get_json(silent=True) or {}
            try:
                if "exposure" in data:
                    server._ascom_exposure = float(data["exposure"])
                if "gain" in data:
                    server._ascom_gain = int(data["gain"])
                    with server._camera_lock:
                        if server._ascom_cam and server._ascom_cam.is_connected:
                            server._ascom_cam.set_gain(server._ascom_gain)
                if "binning" in data:
                    server._ascom_binning = int(data["binning"])
                    with server._camera_lock:
                        if server._ascom_cam and server._ascom_cam.is_connected:
                            server._ascom_cam.set_binning(server._ascom_binning)
            except Exception as exc:
                return jsonify({"ok": False, "error": f"ASCOM settings: {exc}"}), 500
            return jsonify({
                "ok": True,
                "exposure": server._ascom_exposure,
                "gain": server._ascom_gain,
                "binning": server._ascom_binning,
            })

        # ============================================================
        # ASI SDK camera controls (Android + Linux/RPi ZWO cameras)
        # ============================================================

        def _get_asi_module():
            """Get the ASI camera module (Android bridge or desktop ctypes)."""
            try:
                from android_bridge import camera_bridge
                return camera_bridge
            except ImportError:
                pass
            try:
                import asi_camera  # type: ignore[import-not-found]
                return asi_camera
            except ImportError:
                pass
            return None

        @flask_app.route("/api/camera/asi/settings", methods=["POST"])
        def api_camera_asi_settings():
            """Set ASI camera controls (exposure, gain, gamma, offset, etc).

            JSON body fields (all optional):
                exposure_ms: exposure in milliseconds (0.032 to 10000)
                gain: gain value (0 to ~300 depending on camera)
                gamma: gamma correction (0 to 100)
                offset: brightness offset (0 to 255)
                flip: 0=none, 1=horiz, 2=vert, 3=both
                bandwidth: USB bandwidth (40 to 100)
                high_speed: 0=off, 1=on (reduces data path overhead)
            """
            data = request.get_json(silent=True) or {}
            asi_mod = _get_asi_module()
            if asi_mod is None:
                return jsonify({"ok": False, "error": "ASI SDK not available"}), 400

            if not asi_mod.is_asi_sdk_active():
                return jsonify({"ok": False, "error": "ASI SDK camera not active"}), 400

            results = {}
            try:
                if "exposure_ms" in data:
                    exp_us = int(float(data["exposure_ms"]) * 1000)
                    asi_mod.set_asi_control(1, exp_us)  # CTRL_EXPOSURE
                    results["exposure_ms"] = float(data["exposure_ms"])

                if "gain" in data:
                    asi_mod.set_asi_control(0, int(data["gain"]))  # CTRL_GAIN
                    results["gain"] = int(data["gain"])

                if "gamma" in data:
                    asi_mod.set_asi_control(2, int(data["gamma"]))  # CTRL_GAMMA
                    results["gamma"] = int(data["gamma"])

                if "offset" in data:
                    asi_mod.set_asi_control(5, int(data["offset"]))  # CTRL_OFFSET
                    results["offset"] = int(data["offset"])

                if "flip" in data:
                    asi_mod.set_asi_control(9, int(data["flip"]))  # CTRL_FLIP
                    results["flip"] = int(data["flip"])

                if "bandwidth" in data:
                    asi_mod.set_asi_control(6, int(data["bandwidth"]))  # CTRL_BANDWIDTH
                    results["bandwidth"] = int(data["bandwidth"])

                if "high_speed" in data:
                    asi_mod.set_asi_control(14, int(data["high_speed"]))  # CTRL_HIGH_SPEED
                    results["high_speed"] = int(data["high_speed"])

            except Exception as exc:
                return jsonify({"ok": False, "error": str(exc)}), 500

            return jsonify({"ok": True, **results})

        @flask_app.route("/api/camera/asi/status")
        def api_camera_asi_status():
            """Get current ASI camera control values and info."""
            asi_mod = _get_asi_module()
            if asi_mod is None:
                return jsonify({"ok": False, "error": "ASI SDK not available"}), 400

            if not asi_mod.is_asi_sdk_active():
                return jsonify({"ok": False, "active": False})

            controls = asi_mod.get_asi_all_controls()
            info = asi_mod.get_asi_camera_info()

            # Convert exposure from µs to ms for the UI
            if "exposure_us" in controls:
                controls["exposure_ms"] = controls["exposure_us"] / 1000.0

            return jsonify({
                "ok": True,
                "active": True,
                "controls": controls,
                "info": info,
            })

        @flask_app.route("/api/camera/phone/sensor")
        def api_camera_phone_sensor():
            """Get the phone's rear camera physical sensor dimensions."""
            try:
                from android_bridge import camera_bridge as cam_mod
                info = cam_mod.get_phone_sensor_info()
                if not info:
                    return jsonify({"ok": False, "error": "Phone sensor info unavailable"})
                return jsonify({"ok": True, **info})
            except Exception as exc:
                return jsonify({"ok": False, "error": str(exc)}), 500

        # ============================================================
        # Telemetry endpoints (real-time charts & diagnostics)
        # ============================================================

        @flask_app.route("/api/telemetry/graph")
        def api_telemetry_graph():
            """Return correction history time-series for chart rendering.

            Returns 10 time-series (Kalman/ML/PEC/Total/Error for alt & az)
            as JSON arrays, plus timestamps relative to the first record.
            """
            tracking = getattr(server.app_ref, "tracking", None)
            if not tracking:
                return jsonify({"ok": False, "error": "Tracking not available"})
            try:
                data = tracking.get_graph_data()
                # Convert numpy arrays to lists for JSON serialization
                result = {}
                for key, val in data.items():
                    if hasattr(val, 'tolist'):
                        result[key] = val.tolist()
                    else:
                        result[key] = list(val) if val else []
                return jsonify({"ok": True, **result})
            except Exception as exc:
                return jsonify({"ok": False, "error": str(exc)}), 500

        @flask_app.route("/api/telemetry/stats")
        def api_telemetry_stats():
            """Return consolidated tracking statistics from all subsystems."""
            tracking = getattr(server.app_ref, "tracking", None)
            if not tracking:
                return jsonify({"ok": False, "error": "Tracking not available"})
            try:
                stats = tracking.get_stats()
                return jsonify({"ok": True, **stats})
            except Exception as exc:
                return jsonify({"ok": False, "error": str(exc)}), 500

        @flask_app.route("/api/telemetry/kalman")
        def api_telemetry_kalman():
            """Return EKF internals: state, sidereal model, confidence, history."""
            tracking = getattr(server.app_ref, "tracking", None)
            if not tracking:
                return jsonify({"ok": False, "error": "Tracking not available"})
            try:
                kf = tracking.kalman
                result = {
                    "ok": True,
                    "is_initialized": kf.is_initialized,
                }
                if kf.is_initialized:
                    import math

                    result["state"] = {
                        "alt": float(kf.x[0]),
                        "az": float(kf.x[1]),
                        "v_alt": float(kf.x[2]),
                        "v_az": float(kf.x[3]),
                    }
                    result["covariance_diag"] = {
                        "alt": float(kf.P[0, 0]),
                        "az": float(kf.P[1, 1]),
                        "v_alt": float(kf.P[2, 2]),
                        "v_az": float(kf.P[3, 3]),
                    }
                    result["R_matrix"] = {
                        "alt": float(kf.R[0, 0]),
                        "az": float(kf.R[1, 1]),
                    }

                    # --- EKF-specific: sidereal rates at current position ---
                    sid_alt, sid_az = kf._sidereal_rates(kf.x[0], kf.x[1])
                    result["sidereal_rate"] = {
                        "alt": round(sid_alt * 3600, 3),  # arcsec/s
                        "az":  round(sid_az * 3600, 3),
                    }

                    # --- EKF-specific: confidence (0-100) from P diagonal ---
                    # Position sigma in arcsec, mapped to 0-100 via exp decay
                    pos_sigma = math.sqrt((kf.P[0, 0] + kf.P[1, 1]) / 2) * 3600
                    confidence = max(0.0, min(100.0, 100.0 * math.exp(-pos_sigma / 20.0)))
                    result["confidence"] = round(confidence, 1)

                    # --- EKF-specific: zenith proximity ---
                    result["zenith_proximity"] = round(max(0.0, kf.x[0] - 80.0), 1)

                    # --- Drift magnitude (combined) ---
                    drift_mag = math.sqrt(kf.x[2]**2 + kf.x[3]**2) * 3600
                    result["drift_arcsec"] = round(drift_mag, 3)

                    # Statistics
                    stats = kf.get_statistics()
                    result["rms_alt_arcsec"] = stats.get("rms_alt_arcsec", 0)
                    result["rms_az_arcsec"] = stats.get("rms_az_arcsec", 0)
                    result["mean_drift_alt"] = stats.get("mean_drift_alt", 0)
                    result["mean_drift_az"] = stats.get("mean_drift_az", 0)
                    result["samples"] = stats.get("samples", 0)

                    # History for charting (last 50 points)
                    hist = list(kf.history)[-50:]
                    if hist:
                        t0 = hist[0]["time"]
                        result["residual_history"] = {
                            "timestamps":   [h["time"] - t0           for h in hist],
                            "measured_alt":  [h["measured"][0]  * 3600 for h in hist],
                            "measured_az":   [h["measured"][1]  * 3600 for h in hist],
                            "filtered_alt":  [h["filtered"][0]  * 3600 for h in hist],
                            "filtered_az":   [h["filtered"][1]  * 3600 for h in hist],
                            "residual_alt":  [h["residual"][0]  * 3600 for h in hist],
                            "residual_az":   [h["residual"][1]  * 3600 for h in hist],
                            "velocity_alt":  [h["velocity"][0]  * 3600 for h in hist],
                            "velocity_az":   [h["velocity"][1]  * 3600 for h in hist],
                        }
                    else:
                        result["residual_history"] = {}

                    # Adaptive R residual window
                    if hasattr(kf, "residual_window"):
                        rw = list(kf.residual_window)
                        result["innovation_window"] = {
                            "alt": [float(r[0]) * 3600 for r in rw],
                            "az":  [float(r[1]) * 3600 for r in rw],
                        }
                return jsonify(result)
            except Exception as exc:
                return jsonify({"ok": False, "error": str(exc)}), 500

        @flask_app.route("/api/telemetry/ml")
        def api_telemetry_ml():
            """Return ML drift predictor internals: weights, errors, features."""
            tracking = getattr(server.app_ref, "tracking", None)
            if not tracking:
                return jsonify({"ok": False, "error": "Tracking not available"})
            try:
                ml = tracking.ml_predictor
                stats = ml.get_statistics()
                result = {
                    "ok": True,
                    "samples": stats.get("samples", 0),
                    "model_ready": stats.get("model_ready", False),
                    "mean_error_arcsec": stats.get("mean_error_arcsec", 0),
                    "total_predictions": stats.get("total_predictions", 0),
                    "weights_alt": stats.get("weights_alt", []),
                    "weights_az": stats.get("weights_az", []),
                    "feature_names": [
                        "Bias", "Alt (lin)", "Az (lin)",
                        "Alt\u00b2", "Az\u00b2", "Alt\u00d7Az",
                        "sin(Az)", "cos(Az)"
                    ],
                    "learning_rate": ml.learning_rate,
                    "regularization": ml.regularization,
                }
                # Current prediction at current position (if tracking)
                if hasattr(tracking, "current_alt") and hasattr(tracking, "current_az"):
                    try:
                        pred = ml.predict(tracking.current_alt, tracking.current_az)
                        result["current_prediction"] = {
                            "alt": pred[0] * 3600 if pred else 0,
                            "az": pred[1] * 3600 if pred else 0,
                        }
                    except Exception:
                        pass
                return jsonify(result)
            except Exception as exc:
                return jsonify({"ok": False, "error": str(exc)}), 500

        @flask_app.route("/api/telemetry/pec")
        def api_telemetry_pec():
            """Return PEC internals: period details, correction curve, stats."""
            tracking = getattr(server.app_ref, "tracking", None)
            if not tracking:
                return jsonify({"ok": False, "error": "Tracking not available"})
            try:
                pec = tracking.pec
                stats = pec.get_statistics()
                result = {
                    "ok": True,
                    "is_trained": stats.get("is_trained", False),
                    "is_enabled": stats.get("is_enabled", False),
                    "is_learning": stats.get("is_learning", False),
                    "analysis_count": stats.get("analysis_count", 0),
                    "total_samples": stats.get("total_samples", 0),
                    "data_span_sec": stats.get("data_span_sec", 0),
                    "correction_rms_alt": stats.get("correction_rms_alt", 0),
                    "correction_rms_az": stats.get("correction_rms_az", 0),
                    "periods_alt_detail": stats.get("periods_alt_detail", []),
                    "periods_az_detail": stats.get("periods_az_detail", []),
                }
                # Correction curve for plotting (200 points over 600s to keep payload small)
                if stats.get("is_trained", False):
                    try:
                        curve = pec.get_correction_curve(duration_sec=600.0, n_points=200)
                        result["correction_curve"] = {
                            "time": curve["time"].tolist(),
                            "alt": curve["alt"].tolist(),
                            "az": curve["az"].tolist(),
                        }
                    except Exception:
                        pass
                return jsonify(result)
            except Exception as exc:
                return jsonify({"ok": False, "error": str(exc)}), 500

        return flask_app


# ===================================================================
# Embedded HTML / CSS / JavaScript for the web UI
# ===================================================================

_INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#0d0d0d">
<link rel="manifest" href="/manifest.json">
<title>TrackWise-AltAzPro</title>
<script src="/static/chart.umd.min.js" onerror="
  var s=document.createElement('script');s.src='https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js';
  s.onerror=function(){console.warn('Chart.js unavailable (offline)')};document.head.appendChild(s);
"></script>
<style>
:root{
  --bg:rgba(8,8,18,.65);--bg2:rgba(15,15,35,.45);--bg3:rgba(25,25,55,.40);--bg4:rgba(50,50,90,.35);
  --accent:#ff8c00;--accent2:#cc7000;--accent-glow:rgba(255,140,0,.3);
  --blue:#4a9eff;--green:#4caf50;--red:#ff4444;--yellow:#ffd700;
  --text:#e8e8f0;--dim:rgba(180,180,210,.7);
  --radius:14px;--glass:rgba(255,255,255,.04);--glass-border:rgba(255,255,255,.08);
}
*{margin:0;padding:0;box-sizing:border-box}

/* Starfield canvas */
#starfield{position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none}

body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#050510;color:var(--text);
  overflow-x:hidden;-webkit-tap-highlight-color:transparent;width:100%;min-height:100vh}
.container{max-width:900px;margin:0 auto;padding:8px;overflow-x:hidden;position:relative;z-index:1}

/* Animated accent line keyframes */
@keyframes accentSlide{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes subtleFloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-2px)}}
@keyframes glowPulse{0%,100%{box-shadow:0 0 8px var(--accent-glow)}50%{box-shadow:0 0 18px var(--accent-glow),0 0 40px rgba(255,140,0,.1)}}

/* Header */
header{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;
  background:var(--bg2);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
  border:1px solid var(--glass-border);margin-bottom:10px;border-radius:var(--radius);
  box-shadow:0 4px 30px rgba(0,0,0,.3),inset 0 1px 0 rgba(255,255,255,.05);
  position:relative;overflow:hidden}
header::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,var(--accent),var(--blue),var(--accent2),var(--accent));
  background-size:300% 100%;animation:accentSlide 6s ease infinite}
header h1{font-size:1.25em;color:var(--accent);font-weight:700;
  text-shadow:0 0 20px var(--accent-glow);letter-spacing:.5px}
.status-dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:6px;
  transition:all .3s}
.dot-green{background:var(--green);box-shadow:0 0 12px var(--green),0 0 30px rgba(76,175,80,.5);
  animation:glowPulse 2s ease infinite}
.dot-red{background:var(--red);box-shadow:0 0 12px var(--red),0 0 30px rgba(255,68,68,.5)}
.dot-yellow{background:var(--yellow);box-shadow:0 0 12px var(--yellow),0 0 30px rgba(255,215,0,.5)}
.header-status{font-size:.95em;display:flex;align-items:center}

/* Cards -- glassmorphism with enhanced glow on hover */
.card{background:var(--bg2);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);
  border-radius:var(--radius);padding:16px;margin-bottom:10px;
  border:1px solid var(--glass-border);overflow:hidden;position:relative;
  box-shadow:0 4px 24px rgba(0,0,0,.3),inset 0 1px 0 rgba(255,255,255,.04);
  transition:transform .3s ease,box-shadow .4s ease,border-color .4s ease}
.card:hover{
  box-shadow:0 8px 40px rgba(0,0,0,.5),
    0 0 30px rgba(255,140,0,.1),
    0 0 60px rgba(255,140,0,.04),
    inset 0 1px 0 rgba(255,255,255,.08);
  border-color:rgba(255,140,0,.25);
  transform:translateY(-2px)}
.card h2{font-size:.92em;color:var(--accent);margin-bottom:10px;text-transform:uppercase;
  letter-spacing:1.2px;padding-bottom:6px;position:relative;
  text-shadow:0 0 12px var(--accent-glow);transition:text-shadow .3s}
.card:hover h2{text-shadow:0 0 18px var(--accent-glow),0 0 35px rgba(255,140,0,.15)}
.card h2::after{content:'';position:absolute;bottom:0;left:0;width:100%;height:1px;
  background:linear-gradient(90deg,var(--accent-glow),transparent 70%);transition:opacity .3s;opacity:.6}
.card:hover h2::after{opacity:1}

/* Position display */
.pos-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.pos-item{background:var(--bg3);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);
  border-radius:10px;padding:12px 10px;text-align:center;border:1px solid var(--glass-border);
  transition:all .25s ease;position:relative;overflow:hidden}
.pos-item::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,var(--accent-glow),transparent);opacity:0;
  transition:opacity .3s}
.pos-item:hover{background:rgba(30,30,65,.7);border-color:rgba(255,140,0,.2);
  transform:translateY(-2px);
  box-shadow:0 6px 20px rgba(0,0,0,.25),0 0 20px rgba(255,140,0,.06),0 0 40px rgba(255,140,0,.03)}
.pos-item:hover::before{opacity:1}
.pos-item:hover .pos-value{text-shadow:0 0 14px rgba(255,140,0,.2)}
.pos-label{font-size:.72em;color:var(--dim);text-transform:uppercase;letter-spacing:2px;font-weight:600}
.pos-value{font-size:1.4em;font-family:'Consolas','Courier New',monospace;color:var(--text);
  margin-top:4px;font-weight:700;text-shadow:0 0 10px rgba(255,255,255,.08)}

/* Buttons */
.btn{display:inline-flex;align-items:center;justify-content:center;padding:10px 16px;
  border:1px solid var(--glass-border);border-radius:10px;font-size:.85em;font-weight:600;cursor:pointer;
  transition:all .2s ease;touch-action:manipulation;user-select:none;min-height:44px;
  backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);position:relative;overflow:hidden;
  letter-spacing:.3px}
.btn::after{content:'';position:absolute;top:0;left:-100%;width:100%;height:100%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.06),transparent);
  transition:left .4s ease;pointer-events:none}
.btn:hover::after{left:100%}
.btn:active{transform:scale(.95);opacity:.85}
.btn-accent{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#000;
  border-color:var(--accent);box-shadow:0 0 18px var(--accent-glow),0 0 40px rgba(255,140,0,.1);font-weight:700;
  text-shadow:0 1px 2px rgba(0,0,0,.3)}
.btn-accent:hover{box-shadow:0 0 30px var(--accent-glow),0 0 60px rgba(255,140,0,.15);filter:brightness(1.15)}
.btn-blue{background:linear-gradient(135deg,#4a9eff,#2a7edf);color:#000;border-color:#4a9eff;
  box-shadow:0 0 16px rgba(74,158,255,.35),0 0 40px rgba(74,158,255,.1);transition:all .25s ease;
  text-shadow:0 1px 2px rgba(0,0,0,.3)}
.btn-blue:hover{box-shadow:0 0 30px rgba(74,158,255,.5),0 0 60px rgba(74,158,255,.15);
  filter:brightness(1.15);transform:translateY(-1px)}
.btn-green{background:linear-gradient(135deg,#4caf50,#388e3c);color:#000;border-color:#4caf50;
  box-shadow:0 0 16px rgba(76,175,80,.35),0 0 40px rgba(76,175,80,.1);transition:all .25s ease;
  text-shadow:0 1px 2px rgba(0,0,0,.3)}
.btn-green:hover{box-shadow:0 0 30px rgba(76,175,80,.5),0 0 60px rgba(76,175,80,.15);
  filter:brightness(1.15);transform:translateY(-1px)}
.btn-red{background:linear-gradient(135deg,#ff4444,#cc2222);color:#fff;border-color:#ff4444;
  box-shadow:0 0 16px rgba(255,68,68,.35),0 0 40px rgba(255,68,68,.1);transition:all .25s ease;
  text-shadow:0 1px 2px rgba(0,0,0,.3)}
.btn-red:hover{box-shadow:0 0 30px rgba(255,68,68,.5),0 0 60px rgba(255,68,68,.15);
  filter:brightness(1.15);transform:translateY(-1px)}
.btn-dim{background:var(--bg3);color:var(--dim);border-color:var(--glass-border);transition:all .25s ease;
  opacity:.7}
.btn-dim:hover{background:var(--bg4);border-color:rgba(255,255,255,.18);opacity:1;color:var(--text);
  box-shadow:0 0 18px rgba(255,255,255,.08),0 0 40px rgba(255,255,255,.03);transform:translateY(-1px)}
.btn-sm{padding:6px 10px;font-size:.8em;min-height:36px}
.btn-block{width:100%}

/* Button rows */
.btn-row{display:flex;gap:6px;margin-top:6px}
.btn-row .btn{flex:1}

/* Slew pad - enlarged for better visibility and blind operation */
.slew-grid{display:grid;grid-template-columns:repeat(3,88px);grid-template-rows:repeat(3,78px);gap:6px;
  justify-content:center;width:282px;margin:0 auto;flex-shrink:0}
.slew-grid .btn{font-size:1.1em;padding:0;min-height:0;min-width:0;width:100%;height:100%;overflow:hidden;
  border-radius:12px;font-weight:700;display:flex;flex-direction:column;align-items:center;justify-content:center;line-height:1.1}
.slew-grid .btn .arrow{font-size:1.5em;display:block;line-height:1;opacity:.85}

/* Two-column layout for control panels on wider screens */
@media(min-width:500px){
  .ctrl-cols{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  .ctrl-cols .card{margin-bottom:0}
}

/* Inputs */
.input-row{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.input-row label{font-size:.8em;color:var(--dim);min-width:60px;font-weight:500}
.input-row input,.input-row select{background:rgba(15,15,40,.45);color:var(--text);
  border:1px solid var(--glass-border);border-radius:8px;padding:8px 10px;font-size:.85em;flex:1;min-height:38px;
  backdrop-filter:blur(4px);-webkit-backdrop-filter:blur(4px);transition:all .25s ease}
.input-row input:focus,.input-row select:focus{outline:none;border-color:var(--accent);
  box-shadow:0 0 12px var(--accent-glow),inset 0 0 4px rgba(255,140,0,.05);
  background:rgba(15,15,45,.85)}

/* Search */
.search-box{position:relative;margin-bottom:6px}
.search-box input{width:100%;background:rgba(15,15,40,.45);color:var(--text);
  border:1px solid var(--glass-border);border-radius:10px;padding:12px 14px;font-size:.9em;
  backdrop-filter:blur(4px);-webkit-backdrop-filter:blur(4px);transition:all .25s ease}
.search-box input:focus{outline:none;border-color:var(--accent);
  box-shadow:0 0 15px var(--accent-glow),inset 0 0 4px rgba(255,140,0,.05);
  background:rgba(15,15,45,.85)}
.search-results{position:absolute;top:100%;left:0;right:0;background:rgba(12,12,30,.80);
  backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
  border:1px solid var(--glass-border);border-radius:0 0 8px 8px;max-height:200px;overflow-y:auto;
  z-index:100;display:none}
.search-results .sr-item{padding:10px 12px;cursor:pointer;font-size:.85em;
  border-bottom:1px solid var(--glass-border);display:flex;justify-content:space-between;
  transition:background .15s}
.search-results .sr-item:hover,.search-results .sr-item:active{background:rgba(255,140,0,.08);
  box-shadow:inset 2px 0 0 var(--accent)}
.sr-name{color:var(--accent);font-weight:600}
.sr-coord{color:var(--dim);font-size:.8em}

/* Catalog browse */
.cat-chips{display:flex;flex-wrap:wrap;gap:5px;margin:8px 0}
.cat-chip{padding:6px 12px;font-size:.75em;font-weight:600;border-radius:16px;cursor:pointer;
  background:var(--bg3);color:var(--dim);border:1px solid var(--glass-border);transition:all .2s;
  white-space:nowrap;backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px)}
.cat-chip:hover,.cat-chip:active{background:var(--bg4);color:var(--text);
  border-color:rgba(255,255,255,.15);box-shadow:0 2px 12px rgba(0,0,0,.2),0 0 10px rgba(255,140,0,.04);
  transform:translateY(-1px)}
.cat-chip.active{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;
  border-color:var(--accent);box-shadow:0 0 12px var(--accent-glow)}
.cat-chip .chip-count{font-size:.85em;color:var(--dim);margin-left:3px}
.cat-chip.active .chip-count{color:rgba(255,255,255,.7)}
.browse-list{max-height:260px;overflow-y:auto;background:rgba(8,8,20,.45);
  border:1px solid var(--glass-border);border-radius:8px;display:none;
  backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px)}
.browse-item{padding:8px 12px;cursor:pointer;font-size:.85em;
  border-bottom:1px solid var(--glass-border);display:flex;justify-content:space-between;
  align-items:center;transition:background .15s}
.browse-item:last-child{border-bottom:none}
.browse-item:hover,.browse-item:active{background:rgba(255,140,0,.06);
  box-shadow:inset 2px 0 0 var(--accent)}
.browse-id{color:var(--accent);font-weight:600;min-width:70px}
.browse-coord{color:var(--dim);font-size:.8em}
.browse-goto{padding:4px 10px;font-size:.75em;background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:#fff;border:none;border-radius:6px;cursor:pointer;white-space:nowrap;font-weight:600;
  box-shadow:0 0 8px var(--accent-glow);transition:all .15s}
.browse-goto:hover{box-shadow:0 0 18px var(--accent-glow),0 0 40px rgba(255,140,0,.08);filter:brightness(1.1)}
.browse-goto:active{transform:scale(.95)}
.browse-pager{display:flex;justify-content:center;align-items:center;gap:8px;padding:6px;
  font-size:.8em;color:var(--dim)}
.browse-pager button{padding:4px 10px;font-size:.8em;background:var(--bg3);color:var(--text);
  border:1px solid var(--glass-border);border-radius:6px;cursor:pointer;
  backdrop-filter:blur(4px);-webkit-backdrop-filter:blur(4px);transition:all .15s}
.browse-pager button:hover{border-color:rgba(255,255,255,.2);
  box-shadow:0 2px 10px rgba(0,0,0,.2),0 0 8px rgba(255,140,0,.04);transform:translateY(-1px)}
.browse-pager button:disabled{opacity:.3;cursor:default}

/* ========== Enhanced Log System ========== */

/* Log toolbar row */
.log-toolbar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:8px}
.log-toolbar .btn-sm{font-size:.7em;padding:5px 10px;border-radius:8px;cursor:pointer;
  border:1px solid var(--glass-border);background:var(--bg3);color:var(--dim);
  transition:all .2s;white-space:nowrap;font-weight:600;letter-spacing:.3px}
.log-toolbar .btn-sm:hover{background:rgba(255,140,0,.12);color:var(--accent);border-color:rgba(255,140,0,.3)}
.log-toolbar .btn-sm.active{background:rgba(255,140,0,.2);color:var(--accent);
  border-color:rgba(255,140,0,.5);box-shadow:0 0 8px rgba(255,140,0,.15)}

/* Filter tag buttons — each tag type gets its own active color */
.log-toolbar .btn-sm.flt-error.active{background:rgba(255,68,68,.2);color:var(--red);border-color:rgba(255,68,68,.5)}
.log-toolbar .btn-sm.flt-warning.active{background:rgba(255,215,0,.2);color:var(--yellow);border-color:rgba(255,215,0,.5)}
.log-toolbar .btn-sm.flt-success.active{background:rgba(76,175,80,.2);color:var(--green);border-color:rgba(76,175,80,.5)}
.log-toolbar .btn-sm.flt-cmd.active{background:rgba(74,158,255,.2);color:var(--blue);border-color:rgba(74,158,255,.5)}
.log-toolbar .btn-sm.flt-info.active{background:rgba(180,180,210,.15);color:var(--dim);border-color:rgba(180,180,210,.4)}

/* Log search input */
.log-search{flex:1;min-width:120px;max-width:250px;padding:5px 10px;border-radius:8px;
  border:1px solid var(--glass-border);background:rgba(0,0,0,.3);color:var(--text);
  font-size:.75em;font-family:'Consolas',monospace;outline:none;transition:border-color .2s}
.log-search:focus{border-color:rgba(255,140,0,.4);box-shadow:0 0 8px rgba(255,140,0,.1)}
.log-search::placeholder{color:rgba(180,180,210,.4)}

/* Log level selector */
.log-level-select{padding:4px 8px;border-radius:8px;border:1px solid var(--glass-border);
  background:var(--bg3);color:var(--dim);font-size:.7em;cursor:pointer;outline:none}

/* Stats bar */
.log-stats{display:flex;gap:10px;flex-wrap:wrap;padding:6px 10px;margin-bottom:6px;
  background:rgba(0,0,0,.25);border-radius:8px;font-size:.68em;color:var(--dim);
  border:1px solid var(--glass-border);font-family:'Consolas',monospace}
.log-stats .stat{display:flex;align-items:center;gap:3px}
.log-stats .stat-dot{width:6px;height:6px;border-radius:50%;display:inline-block}

/* Log container */
.log-box{background:rgba(0,0,0,.4);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);
  border:1px solid var(--glass-border);border-radius:10px;padding:10px;
  font-family:'Consolas','Courier New',monospace;font-size:.73em;
  height:calc(100vh - 300px);min-height:250px;overflow-y:auto;line-height:1.65;
  position:relative}

/* Log lines — base */
.log-box .log-line{white-space:pre-wrap;word-break:break-all;padding:2px 4px;
  border-bottom:1px solid rgba(255,255,255,.02);display:flex;align-items:baseline;gap:6px;
  border-radius:3px;transition:background .15s}
.log-line:hover{background:rgba(255,255,255,.03)}
.log-line.hidden{display:none}

/* Timestamp */
.log-ts{color:rgba(180,180,210,.45);font-size:.9em;min-width:62px;flex-shrink:0;user-select:none}

/* Tag badges */
.log-tag{font-size:.7em;font-weight:700;padding:1px 5px;border-radius:4px;min-width:32px;
  text-align:center;flex-shrink:0;letter-spacing:.5px;user-select:none;text-transform:uppercase}

/* Message text */
.log-msg{flex:1;min-width:0}

/* 10 tag-type colors */
.log-line.error{color:#ff6b6b}.log-line.error .log-tag{background:rgba(255,68,68,.25);color:#ff6b6b;border:1px solid rgba(255,68,68,.4)}
.log-line.warning{color:#ffdd57}.log-line.warning .log-tag{background:rgba(255,215,0,.2);color:#ffdd57;border:1px solid rgba(255,215,0,.4)}
.log-line.success{color:#69db7c}.log-line.success .log-tag{background:rgba(76,175,80,.2);color:#69db7c;border:1px solid rgba(76,175,80,.4)}
.log-line.info{color:rgba(180,180,210,.75)}.log-line.info .log-tag{background:rgba(180,180,210,.1);color:rgba(180,180,210,.6);border:1px solid rgba(180,180,210,.2)}
.log-line.cmd{color:#74b9ff}.log-line.cmd .log-tag{background:rgba(74,158,255,.2);color:#74b9ff;border:1px solid rgba(74,158,255,.4)}
.log-line.rate{color:#00cec9}.log-line.rate .log-tag{background:rgba(0,206,201,.15);color:#00cec9;border:1px solid rgba(0,206,201,.3)}
.log-line.tracking{color:#a29bfe}.log-line.tracking .log-tag{background:rgba(162,155,254,.15);color:#a29bfe;border:1px solid rgba(162,155,254,.3)}
.log-line.server{color:#fdcb6e}.log-line.server .log-tag{background:rgba(253,203,110,.15);color:#fdcb6e;border:1px solid rgba(253,203,110,.3)}
.log-line.response{color:#55efc4}.log-line.response .log-tag{background:rgba(85,239,196,.12);color:#55efc4;border:1px solid rgba(85,239,196,.25)}
.log-line.usb{color:#fd79a8}.log-line.usb .log-tag{background:rgba(253,121,168,.15);color:#fd79a8;border:1px solid rgba(253,121,168,.3)}

/* Scroll-to-bottom floating button */
.log-scroll-btn{position:absolute;bottom:12px;right:16px;padding:5px 12px;border-radius:20px;
  background:rgba(255,140,0,.85);color:#fff;border:none;font-size:.72em;font-weight:700;
  cursor:pointer;z-index:5;opacity:0;visibility:hidden;transition:all .25s;
  box-shadow:0 2px 12px rgba(255,140,0,.4);letter-spacing:.3px}
.log-scroll-btn.visible{opacity:1;visibility:visible}
.log-scroll-btn:hover{background:rgba(255,140,0,1);transform:scale(1.05)}

/* Weather grid */
.weather-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.weather-grid .wi{background:var(--bg3);backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);
  border:1px solid var(--glass-border);border-radius:8px;padding:8px 10px;
  display:flex;justify-content:space-between;align-items:center;transition:all .25s ease}
.weather-grid .wi:hover{background:rgba(30,30,65,.7);border-color:rgba(255,140,0,.15);
  transform:translateY(-2px);box-shadow:0 4px 16px rgba(0,0,0,.2),0 0 16px rgba(255,140,0,.05),0 0 35px rgba(255,140,0,.025)}
.wi-label{font-size:.72em;color:var(--dim);font-weight:500;letter-spacing:.5px}
.wi-val{font-size:.85em;font-weight:700}

/* Tabs - enlarged for better visibility */
.tabs{display:flex;gap:4px;margin-bottom:12px;overflow-x:auto;-webkit-overflow-scrolling:touch;padding:4px 0}
.tab{padding:14px 22px;font-size:1em;font-weight:700;cursor:pointer;border-radius:12px 12px 0 0;
  background:rgba(200,130,50,.10);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);
  color:var(--accent);white-space:nowrap;transition:all .25s ease;
  border:1px solid rgba(200,130,50,.20);border-bottom:none;position:relative;
  text-shadow:0 0 10px var(--accent-glow);letter-spacing:.5px;min-height:50px;
  display:flex;align-items:center;justify-content:center}
.tab:hover{color:#ffb347;background:rgba(200,130,50,.14);transform:translateY(-2px);
  box-shadow:0 4px 16px rgba(0,0,0,.25),0 0 14px rgba(255,140,0,.1);
  border-color:rgba(220,150,60,.3);text-shadow:0 0 12px var(--accent-glow)}
.tab.active{background:rgba(200,130,50,.35);color:#ffcc66;border-color:rgba(220,150,60,.6);
  text-shadow:0 0 18px rgba(255,180,0,.6);box-shadow:0 -2px 20px rgba(255,140,0,.3),0 0 30px rgba(255,140,0,.12)}
.tab.active::after{content:'';position:absolute;bottom:0;left:5%;right:5%;height:3px;
  background:linear-gradient(90deg,transparent,var(--accent),transparent);border-radius:2px;box-shadow:0 0 16px var(--accent-glow),0 0 30px rgba(255,140,0,.15)}
.tab-content{display:none}.tab-content.active{display:block}

/* Sky Chart */
.sc-wrap{position:relative;width:100%;background:#000;border-radius:12px;overflow:hidden;touch-action:none}
.sc-canvas{display:block;width:100%;height:100%}
.sc-controls{position:absolute;top:8px;right:8px;display:flex;flex-direction:column;gap:4px;z-index:10}
.sc-controls button{width:36px;height:36px;border:1px solid rgba(255,180,80,.3);border-radius:8px;
  background:rgba(0,0,0,.55);backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);
  color:var(--accent);font-size:1.1em;cursor:pointer;display:flex;align-items:center;justify-content:center;
  transition:all .2s;touch-action:manipulation;-webkit-tap-highlight-color:rgba(255,180,80,.2);
  user-select:none;-webkit-user-select:none;position:relative;z-index:15}
.sc-controls button:hover,.sc-controls button.sc-active{background:rgba(200,130,50,.3);border-color:var(--accent)}
.sc-fov{position:absolute;top:8px;left:8px;background:rgba(0,0,0,.55);backdrop-filter:blur(6px);
  -webkit-backdrop-filter:blur(6px);border:1px solid rgba(255,180,80,.2);border-radius:8px;
  padding:4px 10px;color:var(--accent);font-size:.75em;font-weight:600;z-index:10;pointer-events:none}
.sc-info{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,.8);backdrop-filter:blur(10px);
  -webkit-backdrop-filter:blur(10px);border-top:1px solid rgba(255,180,80,.25);padding:10px 14px;
  z-index:20;display:none;animation:scInfoIn .25s ease}
@keyframes scInfoIn{from{transform:translateY(100%);opacity:0}to{transform:translateY(0);opacity:1}}
.sc-info-name{font-size:1.1em;font-weight:700;color:#ffcc66;text-shadow:0 0 8px rgba(255,180,0,.4)}
.sc-info-detail{font-size:.8em;color:var(--dim);margin-top:2px}
.sc-info-row{display:flex;justify-content:space-between;align-items:center;margin-top:6px}
.sc-info .btn{padding:6px 16px;font-size:.85em}

/* Misc */
.mt-1{margin-top:6px}.mt-2{margin-top:12px}
.text-center{text-align:center}
.text-dim{color:var(--dim);font-size:.8em}
.flex-between{display:flex;justify-content:space-between;align-items:center}
.badge{display:inline-block;padding:4px 12px;border-radius:14px;font-size:.75em;font-weight:700;
  backdrop-filter:blur(4px);-webkit-backdrop-filter:blur(4px);letter-spacing:.3px;
  transition:all .2s ease}
.badge-green{background:rgba(76,175,80,.22);color:#6fe87a;border:1px solid rgba(76,175,80,.45);
  box-shadow:0 0 10px rgba(76,175,80,.12)}
.badge-red{background:rgba(255,68,68,.22);color:#ff7b7b;border:1px solid rgba(255,68,68,.45);
  box-shadow:0 0 10px rgba(255,68,68,.12)}
.badge-yellow{background:rgba(255,215,0,.22);color:#ffe44d;border:1px solid rgba(255,215,0,.45);
  box-shadow:0 0 10px rgba(255,215,0,.12)}
.badge-blue{background:rgba(74,158,255,.22);color:#7bbfff;border:1px solid rgba(74,158,255,.45);
  box-shadow:0 0 14px rgba(74,158,255,.2)}

/* Active button indicator (glowing when feature is running) */
.btn.btn-active-green{background:rgba(76,175,80,.35)!important;color:#fff!important;
  border-color:rgba(76,175,80,.7)!important;
  box-shadow:0 0 20px rgba(76,175,80,.45),0 0 50px rgba(76,175,80,.15),inset 0 0 12px rgba(76,175,80,.15)!important;
  text-shadow:0 0 10px rgba(76,175,80,.6);
  animation:pulseGreen 2s ease-in-out infinite}
.btn.btn-active-yellow{background:rgba(255,215,0,.30)!important;color:#fff!important;
  border-color:rgba(255,215,0,.65)!important;
  box-shadow:0 0 20px rgba(255,215,0,.40),0 0 50px rgba(255,215,0,.12),inset 0 0 12px rgba(255,215,0,.12)!important;
  text-shadow:0 0 10px rgba(255,215,0,.5);
  animation:pulseYellow 2s ease-in-out infinite}
.btn.btn-active-blue{background:rgba(74,158,255,.35)!important;color:#fff!important;
  border-color:rgba(74,158,255,.7)!important;
  box-shadow:0 0 20px rgba(74,158,255,.45),0 0 50px rgba(74,158,255,.15),inset 0 0 12px rgba(74,158,255,.15)!important;
  text-shadow:0 0 10px rgba(74,158,255,.6);
  animation:pulseBlue 2s ease-in-out infinite}
@keyframes pulseGreen{0%,100%{box-shadow:0 0 20px rgba(76,175,80,.45),0 0 50px rgba(76,175,80,.15),inset 0 0 12px rgba(76,175,80,.15)}
  50%{box-shadow:0 0 35px rgba(76,175,80,.6),0 0 70px rgba(76,175,80,.2),inset 0 0 18px rgba(76,175,80,.2)}}
@keyframes pulseYellow{0%,100%{box-shadow:0 0 20px rgba(255,215,0,.40),0 0 50px rgba(255,215,0,.12),inset 0 0 12px rgba(255,215,0,.12)}
  50%{box-shadow:0 0 35px rgba(255,215,0,.55),0 0 70px rgba(255,215,0,.18),inset 0 0 18px rgba(255,215,0,.18)}}
@keyframes pulseBlue{0%,100%{box-shadow:0 0 20px rgba(74,158,255,.45),0 0 50px rgba(74,158,255,.15),inset 0 0 12px rgba(74,158,255,.15)}
  50%{box-shadow:0 0 35px rgba(74,158,255,.6),0 0 70px rgba(74,158,255,.2),inset 0 0 18px rgba(74,158,255,.2)}}

/* ---- Phone (< 400px) ---- */
@media(max-width:400px){
  .container{padding:4px}
  header{padding:8px;margin-bottom:6px}
  header h1{font-size:.95em}
  .card{padding:8px;margin-bottom:5px}
  .card h2{font-size:.78em}
  .pos-grid{gap:4px}
  .pos-value{font-size:1em}
  .pos-label{font-size:.65em}
  .tabs{gap:2px}
  .tab{padding:10px 14px;font-size:.88em;min-height:44px}
  .slew-grid{grid-template-columns:repeat(3,76px);grid-template-rows:repeat(3,68px);width:244px}
  .btn{padding:8px 12px;font-size:.82em;min-height:40px}
  .btn-row{gap:4px}
  .log-box{height:calc(100vh - 340px);min-height:200px;font-size:.68em}
  .log-toolbar{gap:4px}.log-toolbar .btn-sm{font-size:.65em;padding:4px 8px}
  .log-search{min-width:80px;max-width:160px}
  .log-stats{font-size:.62em;gap:6px;padding:4px 8px}
  .log-ts{min-width:52px;font-size:.85em}
  .input-row label{min-width:50px;font-size:.75em}
  .input-row input,.input-row select{padding:6px;font-size:.82em;min-height:34px}
}

/* ---- Tablet landscape (500-900px) ---- */
@media(min-width:500px) and (max-width:900px){
  .pos-grid{grid-template-columns:repeat(4,1fr)}
}

/* ---- Large tablet / desktop (> 900px) ---- */
@media(min-width:900px){
  .pos-grid{grid-template-columns:repeat(4,1fr)}
  .container{padding:12px}
}

/* Touch: prevent scroll/zoom while pressing slew buttons */
.slew-grid .btn{touch-action:none}
.btn{-webkit-user-select:none;user-select:none}

/* Fullscreen slew overlay - large cross for blind telescope operation */
.slew-fs-overlay{position:fixed;top:0;left:0;right:0;bottom:0;z-index:10000;background:#000;
  display:none;grid-template-columns:1fr 1fr 1fr;grid-template-rows:1fr 1fr 1fr}
.slew-fs-overlay.active{display:grid}
.slew-fs-btn{display:flex;flex-direction:column;align-items:center;justify-content:center;
  font-weight:800;color:rgba(255,255,255,.85);border:2px solid rgba(255,255,255,.08);
  touch-action:none;user-select:none;cursor:pointer;transition:background .15s,border-color .15s}
.slew-fs-btn:active,.slew-fs-btn.pressing{border-color:rgba(255,180,0,.6)}
.slew-fs-btn .fs-arrow{font-size:4em;line-height:1;display:block;pointer-events:none}
.slew-fs-btn .fs-label{font-size:1.8em;letter-spacing:3px;margin-top:4px;pointer-events:none}
.slew-fs-n{background:rgba(0,70,160,.25);grid-column:2;grid-row:1}
.slew-fs-n:active,.slew-fs-n.pressing{background:rgba(0,100,220,.5)}
.slew-fs-w{background:rgba(0,70,160,.25);grid-column:1;grid-row:2}
.slew-fs-w:active,.slew-fs-w.pressing{background:rgba(0,100,220,.5)}
.slew-fs-stop{background:rgba(180,20,20,.4);grid-column:2;grid-row:2;position:relative}
.slew-fs-stop:active,.slew-fs-stop.pressing{background:rgba(220,40,40,.7)}
.slew-fs-stop .fs-arrow{font-size:3em}
.slew-fs-stop .fs-label{font-size:2.2em;font-weight:900;color:#ff6666}
.slew-fs-stop::before{content:'';position:absolute;top:50%;left:0;right:0;height:2px;
  background:rgba(255,255,255,.15);pointer-events:none}
.slew-fs-stop::after{content:'';position:absolute;left:50%;top:0;bottom:0;width:2px;
  background:rgba(255,255,255,.15);pointer-events:none}
.slew-fs-e{background:rgba(0,70,160,.25);grid-column:3;grid-row:2}
.slew-fs-e:active,.slew-fs-e.pressing{background:rgba(0,100,220,.5)}
.slew-fs-s{background:rgba(0,70,160,.25);grid-column:2;grid-row:3}
.slew-fs-s:active,.slew-fs-s.pressing{background:rgba(0,100,220,.5)}
.slew-fs-corner{background:rgba(0,0,0,.95);display:flex;align-items:center;justify-content:center}
.slew-fs-exit{position:absolute;top:12px;right:12px;z-index:10001;background:rgba(60,60,60,.8);
  color:#fff;border:2px solid rgba(255,255,255,.3);border-radius:50%;width:56px;height:56px;
  font-size:1.6em;font-weight:700;cursor:pointer;display:flex;align-items:center;justify-content:center;
  touch-action:none;backdrop-filter:blur(4px)}
.slew-fs-exit:active{background:rgba(200,40,40,.8);border-color:rgba(255,100,100,.5)}
.slew-fs-speed{position:absolute;top:12px;left:12px;z-index:10001;display:flex;gap:6px}
.slew-fs-speed .fs-spd-btn{width:52px;height:52px;border-radius:50%;font-size:1.3em;font-weight:700;
  border:2px solid rgba(255,255,255,.2);background:rgba(40,40,40,.8);color:rgba(255,255,255,.7);
  cursor:pointer;touch-action:none;transition:all .15s}
.slew-fs-speed .fs-spd-btn.active{background:rgba(255,140,0,.4);border-color:rgba(255,180,0,.6);
  color:#ffcc66;box-shadow:0 0 15px rgba(255,140,0,.3)}
.slew-fs-crosshair-h{position:absolute;top:50%;left:0;right:0;height:1px;
  background:rgba(255,100,0,.2);pointer-events:none;z-index:10000}
.slew-fs-crosshair-v{position:absolute;left:50%;top:0;bottom:0;width:1px;
  background:rgba(255,100,0,.2);pointer-events:none;z-index:10000}

/* Alignment slew overlay - centering cross for manual alignment */
.align-slew-overlay{position:fixed;top:0;left:0;right:0;bottom:0;z-index:10002;background:#000;
  display:none;grid-template-columns:1fr 1fr 1fr;grid-template-rows:auto 1fr 1fr 1fr}
.align-slew-overlay.active{display:grid}
.align-slew-header{grid-column:1/4;grid-row:1;background:rgba(255,140,0,.10);
  border-bottom:1px solid rgba(255,180,0,.25);padding:10px 14px;display:flex;
  align-items:center;justify-content:space-between;gap:10px;min-height:0}
.align-slew-header .align-slew-star-info{flex:1;min-width:0}
.align-slew-header .align-slew-star-name{font-size:1.1em;font-weight:800;color:#ffcc66;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.align-slew-header .align-slew-star-coords{font-size:.78em;color:rgba(255,255,255,.55);margin-top:2px}
.align-slew-header .align-slew-hint{font-size:.72em;color:rgba(255,180,0,.7);font-weight:600;
  margin-top:3px}
.align-slew-btn{display:flex;flex-direction:column;align-items:center;justify-content:center;
  font-weight:800;color:rgba(255,255,255,.85);border:2px solid rgba(255,255,255,.08);
  touch-action:none;user-select:none;cursor:pointer;transition:background .15s,border-color .15s}
.align-slew-btn:active,.align-slew-btn.pressing{border-color:rgba(255,180,0,.6)}
.align-slew-btn .fs-arrow{font-size:3.5em;line-height:1;display:block;pointer-events:none}
.align-slew-btn .fs-label{font-size:1.5em;letter-spacing:3px;margin-top:4px;pointer-events:none}
.align-slew-n{background:rgba(0,70,160,.25);grid-column:2;grid-row:2}
.align-slew-n:active,.align-slew-n.pressing{background:rgba(0,100,220,.5)}
.align-slew-w{background:rgba(0,70,160,.25);grid-column:1;grid-row:3}
.align-slew-w:active,.align-slew-w.pressing{background:rgba(0,100,220,.5)}
.align-slew-sync{background:linear-gradient(135deg,rgba(0,160,60,.45),rgba(0,200,80,.3));
  grid-column:2;grid-row:3;position:relative;border:2px solid rgba(76,175,80,.4)}
.align-slew-sync:active,.align-slew-sync.pressing{background:linear-gradient(135deg,rgba(0,200,80,.7),rgba(0,255,100,.4));
  border-color:rgba(76,175,80,.8)}
.align-slew-sync .fs-arrow{font-size:2.5em;color:#66ff88}
.align-slew-sync .fs-label{font-size:1.8em;font-weight:900;color:#66ff88;letter-spacing:2px}
.align-slew-sync::before{content:'';position:absolute;top:50%;left:0;right:0;height:2px;
  background:rgba(76,175,80,.2);pointer-events:none}
.align-slew-sync::after{content:'';position:absolute;left:50%;top:0;bottom:0;width:2px;
  background:rgba(76,175,80,.2);pointer-events:none}
.align-slew-e{background:rgba(0,70,160,.25);grid-column:3;grid-row:3}
.align-slew-e:active,.align-slew-e.pressing{background:rgba(0,100,220,.5)}
.align-slew-s{background:rgba(0,70,160,.25);grid-column:2;grid-row:4}
.align-slew-s:active,.align-slew-s.pressing{background:rgba(0,100,220,.5)}
.align-slew-corner{background:rgba(0,0,0,.95);display:flex;align-items:center;justify-content:center}
.align-slew-exit{position:absolute;top:12px;right:12px;z-index:10003;background:rgba(60,60,60,.8);
  color:#fff;border:2px solid rgba(255,255,255,.3);border-radius:50%;width:52px;height:52px;
  font-size:1.4em;font-weight:700;cursor:pointer;display:flex;align-items:center;justify-content:center;
  touch-action:none;backdrop-filter:blur(4px)}
.align-slew-exit:active{background:rgba(200,40,40,.8);border-color:rgba(255,100,100,.5)}
.align-slew-speed{position:absolute;bottom:12px;left:12px;z-index:10003;display:flex;gap:6px}
.align-slew-speed .as-spd-btn{width:48px;height:48px;border-radius:50%;font-size:1.2em;font-weight:700;
  border:2px solid rgba(255,255,255,.2);background:rgba(40,40,40,.8);color:rgba(255,255,255,.7);
  cursor:pointer;touch-action:none;transition:all .15s}
.align-slew-speed .as-spd-btn.active{background:rgba(255,140,0,.4);border-color:rgba(255,180,0,.6);
  color:#ffcc66;box-shadow:0 0 15px rgba(255,140,0,.3)}
.align-slew-skip{position:absolute;bottom:12px;right:12px;z-index:10003;background:rgba(180,40,40,.6);
  color:#ff8888;border:2px solid rgba(255,80,80,.3);border-radius:10px;padding:10px 18px;
  font-size:.95em;font-weight:700;cursor:pointer;touch-action:none;backdrop-filter:blur(4px)}
.align-slew-skip:active{background:rgba(220,50,50,.8);border-color:rgba(255,100,100,.5);color:#fff}
.align-slew-crosshair-h{position:absolute;top:calc(50% + 20px);left:0;right:0;height:1px;
  background:rgba(76,175,80,.25);pointer-events:none;z-index:10002}
.align-slew-crosshair-v{position:absolute;left:50%;top:0;bottom:0;width:1px;
  background:rgba(76,175,80,.25);pointer-events:none;z-index:10002}

/* Safe area for notched phones */
@supports(padding:env(safe-area-inset-top)){
  body{padding-top:env(safe-area-inset-top);padding-bottom:env(safe-area-inset-bottom);
    padding-left:env(safe-area-inset-left);padding-right:env(safe-area-inset-right)}
}

/* Alignment */
.align-star-btns{display:flex;flex-wrap:wrap;gap:4px;margin:6px 0}
.align-star-btn{padding:6px 12px;font-size:.8em;font-weight:700;border-radius:8px;cursor:pointer;
  background:var(--bg3);color:var(--dim);border:1px solid var(--glass-border);transition:all .2s;
  backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);min-width:40px;text-align:center}
.align-star-btn:hover{background:var(--bg4);color:var(--text);border-color:rgba(255,255,255,.15);
  box-shadow:0 2px 12px rgba(0,0,0,.2),0 0 10px rgba(255,140,0,.04);transform:translateY(-1px)}
.align-star-btn.active{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;
  border-color:var(--accent);box-shadow:0 0 12px var(--accent-glow)}
.align-progress{margin:8px 0;font-size:.85em}
.align-progress .step{color:var(--accent);font-weight:600}
.align-stars-list{max-height:180px;overflow-y:auto;background:rgba(8,8,20,.5);
  border:1px solid var(--glass-border);border-radius:8px;margin-top:6px;
  backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px)}
.align-star-row{padding:6px 10px;font-size:.8em;display:flex;align-items:center;gap:6px;
  border-bottom:1px solid var(--glass-border)}
.align-star-row:last-child{border-bottom:none}
.align-star-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.align-dot-pending{background:var(--dim);opacity:.4}
.align-dot-active{background:var(--accent);box-shadow:0 0 8px var(--accent-glow);animation:pulse 1s infinite}
.align-dot-done{background:var(--green);box-shadow:0 0 6px rgba(76,175,80,.4)}
.align-dot-failed{background:var(--red);box-shadow:0 0 6px rgba(255,68,68,.4)}
.align-star-name{color:var(--text);font-weight:600;min-width:80px}
.align-star-info{color:var(--dim);font-size:.9em}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.align-dot-waiting{background:var(--accent);box-shadow:0 0 8px var(--accent-glow);animation:pulse 1.2s infinite}
.align-manual-panel-waiting{animation:manualPulse 2s ease-in-out infinite}
@keyframes manualPulse{0%,100%{border-color:rgba(255,180,0,.25)}50%{border-color:rgba(255,180,0,.6)}}

/* Camera live view */
.cam-container{position:relative;background:#000;border-radius:10px;overflow:hidden;
  border:1px solid var(--glass-border);box-shadow:0 4px 20px rgba(0,0,0,.4)}
.cam-img{display:block;width:100%;height:auto;min-height:200px;object-fit:contain;background:#000}
.cam-overlay{position:absolute;top:8px;right:8px;display:flex;gap:4px;z-index:10}
.cam-overlay .btn{padding:6px 10px;font-size:.75em;min-height:0;
  background:rgba(0,0,0,.6);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px)}
.cam-placeholder{display:flex;align-items:center;justify-content:center;height:250px;
  color:var(--dim);font-size:.9em;background:rgba(0,0,0,.4);border-radius:10px;
  border:1px dashed rgba(255,140,0,.2);transition:border-color .3s}
.cam-placeholder:hover{border-color:rgba(255,140,0,.4)}
/* Fullscreen camera */
.cam-fullscreen{position:fixed!important;top:0!important;left:0!important;
  width:100vw!important;height:100vh!important;z-index:9999!important;
  margin:0!important;border-radius:0!important;border:none!important}
.cam-fullscreen .cam-img{width:100vw;height:100vh;object-fit:contain}
.cam-fullscreen .cam-overlay{top:16px;right:16px}
.cam-fullscreen .cam-overlay .btn{padding:10px 16px;font-size:.9em}

/* Toast notification panel (left side) */
@keyframes toastIn{from{transform:translateX(-100%);opacity:0}to{transform:translateX(0);opacity:1}}
@keyframes toastOut{from{transform:translateX(0);opacity:1}to{transform:translateX(-100%);opacity:0}}
.toast-panel{position:fixed;bottom:16px;left:16px;z-index:2000;display:flex;flex-direction:column-reverse;
  gap:6px;max-height:60vh;overflow:hidden;pointer-events:none}
.toast{pointer-events:auto;display:flex;align-items:center;gap:8px;padding:10px 14px;
  border-radius:10px;font-size:.8em;font-weight:600;min-width:200px;max-width:340px;
  backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  border:1px solid rgba(255,255,255,.1);
  box-shadow:0 4px 20px rgba(0,0,0,.4);
  animation:toastIn .3s ease forwards;letter-spacing:.3px}
.toast.removing{animation:toastOut .3s ease forwards}
.toast-ok{background:rgba(20,60,30,.85);color:#6fcf7c;border-color:rgba(76,175,80,.25)}
.toast-err{background:rgba(60,15,15,.85);color:#ff7b7b;border-color:rgba(255,68,68,.25)}
.toast-info{background:rgba(15,30,60,.85);color:#7bb8ff;border-color:rgba(74,158,255,.25)}
.toast-icon{font-size:1.1em;flex-shrink:0}
.toast-msg{flex:1;line-height:1.3}

/* Control tab: copper-orange card accent */
#tab-control .card{border-left:3px solid rgba(200,120,40,.5)}
#tab-control .card:hover{border-left-color:rgba(220,140,50,.8)}
#tab-control .card h2{color:#d4935a;text-shadow:0 0 10px rgba(200,130,50,.3)}
#tab-control .card h2::after{background:linear-gradient(90deg,rgba(200,130,50,.4),transparent 70%)}
#tab-control .slew-grid .btn{border-color:rgba(200,130,50,.3);border-width:2px}
#tab-control .slew-grid .btn:hover{border-color:rgba(220,150,60,.5);
  box-shadow:0 0 14px rgba(200,130,50,.25)}
#tab-control .slew-grid .btn:active{background:rgba(200,130,50,.3);border-color:rgba(255,180,60,.6)}

/* Light theme -- applied to <html> via JS toggle */
html.lightmode{--bg:rgba(240,240,245,.95);--bg2:rgba(230,230,240,.9);--bg3:rgba(215,215,230,.85);--bg4:rgba(200,200,220,.8);
  --accent:#d47200;--accent2:#b86000;--accent-glow:rgba(200,110,0,.15);
  --blue:#2a7de1;--green:#2e8b3e;--red:#d42020;--yellow:#c5a000;
  --text:#1a1a2e;--dim:rgba(60,60,90,.65);
  --glass:rgba(0,0,0,.03);--glass-border:rgba(0,0,0,.1)}
html.lightmode body{background:#e8e8f0!important}
html.lightmode #starfield{display:none}
html.lightmode header{box-shadow:0 2px 16px rgba(0,0,0,.08),inset 0 -1px 0 rgba(0,0,0,.05)}
html.lightmode .card{background:rgba(255,255,255,.85)!important;
  box-shadow:0 2px 12px rgba(0,0,0,.06),inset 0 1px 0 rgba(255,255,255,.8)!important;
  border-color:rgba(0,0,0,.08)!important}
html.lightmode .card:hover{box-shadow:0 4px 20px rgba(0,0,0,.1),inset 0 1px 0 rgba(255,255,255,.9)!important}
html.lightmode .card h2{text-shadow:none}
html.lightmode header h1{text-shadow:none}
html.lightmode .btn-dim{background:rgba(0,0,0,.06);color:var(--text);border-color:rgba(0,0,0,.1)}
html.lightmode .btn-dim:hover{background:rgba(0,0,0,.1);box-shadow:0 2px 10px rgba(0,0,0,.08)!important;transform:translateY(-1px)}
html.lightmode .btn-blue:hover,html.lightmode .btn-green:hover,html.lightmode .btn-red:hover{
  box-shadow:0 4px 16px rgba(0,0,0,.12)!important;transform:translateY(-1px)}
html.lightmode .btn-accent{background:linear-gradient(135deg,#e88a00,#d47200)!important;color:#fff!important}
html.lightmode .tab{background:rgba(180,110,40,.06);color:#a06830;border-color:rgba(180,110,40,.15);
  text-shadow:none}
html.lightmode .tab:hover{background:rgba(180,110,40,.12);color:#8a5520;
  box-shadow:0 2px 10px rgba(0,0,0,.08)!important;border-color:rgba(180,110,40,.25)}
html.lightmode .tab.active{background:rgba(180,110,40,.16);color:#7a4a18;border-color:rgba(180,110,40,.35);
  box-shadow:0 -2px 10px rgba(180,110,40,.1)!important}
html.lightmode input,html.lightmode select{background:rgba(255,255,255,.9)!important;color:#1a1a2e!important;
  border-color:rgba(0,0,0,.12)!important}
html.lightmode .pos-item{background:rgba(255,255,255,.7);border-color:rgba(0,0,0,.08)}
html.lightmode .log-line{border-bottom-color:rgba(0,0,0,.05)}
html.lightmode .toolbar .btn{box-shadow:0 2px 8px rgba(0,0,0,.12);background:rgba(255,255,255,.9);
  border-color:rgba(0,0,0,.1);color:#1a1a2e}
html.lightmode .toolbar .btn:hover{box-shadow:0 4px 16px rgba(0,0,0,.15)!important;
  border-color:rgba(200,110,0,.25)!important}
html.lightmode .toolbar .btn.active{background:rgba(200,110,0,.1);border-color:rgba(200,110,0,.3)}
html.lightmode .weather-grid .wi{background:rgba(255,255,255,.7);border-color:rgba(0,0,0,.06)}
html.lightmode .weather-grid .wi:hover{box-shadow:0 2px 10px rgba(0,0,0,.08)!important;
  border-color:rgba(200,110,0,.15)!important}
html.lightmode .pos-item:hover{box-shadow:0 3px 12px rgba(0,0,0,.08)!important;
  border-color:rgba(200,110,0,.15)!important;background:rgba(255,255,255,.85)!important}
html.lightmode ::-webkit-scrollbar-thumb{background:rgba(0,0,0,.15)}
html.lightmode ::-webkit-scrollbar-thumb:hover{background:rgba(0,0,0,.25)}
html.lightmode .toast-ok{background:rgba(230,250,235,.95);color:#1a7a2e;border-color:rgba(76,175,80,.3)}
html.lightmode .toast-err{background:rgba(255,235,235,.95);color:#c02020;border-color:rgba(255,68,68,.3)}
html.lightmode .toast-info{background:rgba(235,245,255,.95);color:#1a5ea0;border-color:rgba(74,158,255,.3)}
html.lightmode .toast{box-shadow:0 4px 16px rgba(0,0,0,.1)}
html.lightmode #tab-control .card{border-left-color:rgba(180,110,40,.3)}
html.lightmode #tab-control .card h2{color:#a06830}
html.lightmode .dot-green{box-shadow:0 0 6px var(--green)}
html.lightmode .dot-red{box-shadow:0 0 6px var(--red)}
html.lightmode .dot-yellow{box-shadow:0 0 6px var(--yellow)}

/* Toolbar (bottom-right floating, collapsible) */
.toolbar{position:fixed;bottom:16px;right:16px;z-index:1000;display:flex;flex-direction:column;align-items:center;gap:8px}
.toolbar .btn{width:44px;height:44px;border-radius:50%;padding:0;font-size:1.1em;
  box-shadow:0 4px 16px rgba(0,0,0,.5);transition:all .25s ease;
  border:1px solid rgba(255,255,255,.1);background:rgba(15,15,35,.85);
  backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px)}
.toolbar .btn:hover{transform:scale(1.15);box-shadow:0 6px 24px rgba(0,0,0,.6),0 0 20px var(--accent-glow),0 0 45px rgba(255,140,0,.08);
  border-color:rgba(255,140,0,.35)}
.toolbar .btn.active{background:rgba(255,140,0,.30);border-color:rgba(255,140,0,.6);
  box-shadow:0 0 16px rgba(255,140,0,.3);
  box-shadow:0 4px 16px rgba(0,0,0,.5),0 0 14px var(--accent-glow)}
.toolbar-items{display:flex;flex-direction:column;gap:8px;overflow:hidden;
  max-height:0;opacity:0;transition:max-height .3s ease,opacity .25s ease;pointer-events:none}
.toolbar-items.open{max-height:200px;opacity:1;pointer-events:auto}
.toolbar-toggle{width:44px;height:44px;border-radius:50%;font-size:1.3em;
  transition:transform .3s ease,background .25s ease}
.toolbar-toggle.open{transform:rotate(135deg);background:rgba(255,140,0,.15);border-color:rgba(255,140,0,.4)}

/* Telemetry dashboard */
.telem-section{margin-bottom:8px}
.telem-section h3{font-size:.78em;color:var(--accent);text-transform:uppercase;letter-spacing:1px;
  margin-bottom:6px;padding-bottom:4px;border-bottom:1px solid var(--glass-border)}
.telem-chart-wrap{position:relative;width:100%;height:200px;margin-bottom:8px}
.telem-chart-wrap.tall{height:260px}
.telem-stats-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px}
.telem-stat{background:var(--bg3);border-radius:8px;padding:8px 10px;border:1px solid var(--glass-border)}
.telem-stat-label{font-size:.7em;color:var(--dim);text-transform:uppercase;letter-spacing:.5px}
.telem-stat-value{font-size:1em;font-weight:600;margin-top:2px}
.telem-stat-value.green{color:var(--green)}
.telem-stat-value.red{color:var(--red)}
.telem-stat-value.yellow{color:var(--yellow)}
.telem-stat-value.blue{color:var(--blue)}
.telem-stat-value.accent{color:var(--accent)}
.telem-axis-toggle{display:flex;gap:4px;margin-bottom:6px}
.telem-axis-btn{font-size:.75em;padding:3px 10px;border-radius:6px;cursor:pointer;
  background:var(--bg3);border:1px solid var(--glass-border);color:var(--dim);transition:all .2s}
.telem-axis-btn.active{background:rgba(255,140,0,.15);border-color:var(--accent);color:var(--accent)}
.telem-period-table{width:100%;font-size:.8em;border-collapse:collapse;margin-top:6px}
.telem-period-table th{text-align:left;color:var(--accent);font-weight:600;padding:4px 6px;
  border-bottom:1px solid var(--glass-border);font-size:.85em;text-transform:uppercase}
.telem-period-table td{padding:4px 6px;border-bottom:1px solid rgba(255,255,255,.03)}
.telem-weights-grid{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-top:6px}
.telem-weight-bar{background:var(--bg3);border-radius:6px;padding:6px 8px;border:1px solid var(--glass-border);
  display:flex;flex-direction:column;gap:2px}
.telem-weight-name{font-size:.7em;color:var(--dim)}
.telem-weight-vals{display:flex;gap:8px;font-size:.8em}
.telem-bar-track{height:4px;background:var(--bg4);border-radius:2px;margin-top:2px;overflow:hidden}
.telem-bar-fill{height:100%;border-radius:2px;transition:width .3s}
.telem-bar-fill.alt-bar{background:var(--accent)}
.telem-bar-fill.az-bar{background:var(--blue)}
.telem-no-data{text-align:center;color:var(--dim);padding:30px 0;font-size:.85em}

/* Scrollbar */
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(255,140,0,.3);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:rgba(255,140,0,.5)}

/* ============================================================
   ENHANCEMENT 1: Bottom Tab Bar Navigation
   ============================================================ */
.bottom-nav{position:fixed;bottom:0;left:0;right:0;z-index:900;
  display:flex;justify-content:space-around;align-items:stretch;
  background:rgba(8,8,18,.92);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  border-top:1px solid rgba(255,140,0,.15);
  padding:0;padding-bottom:env(safe-area-inset-bottom,0);
  box-shadow:0 -4px 24px rgba(0,0,0,.5)}
.bottom-nav .bnav-item{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;
  padding:6px 2px 8px;cursor:pointer;transition:all .2s;color:var(--dim);
  -webkit-tap-highlight-color:transparent;user-select:none;position:relative;min-height:56px}
.bottom-nav .bnav-item .bnav-icon{font-size:1.35em;line-height:1;margin-bottom:2px;transition:transform .2s}
.bottom-nav .bnav-item .bnav-label{font-size:.62em;font-weight:600;letter-spacing:.5px;text-transform:uppercase;
  transition:color .2s}
.bottom-nav .bnav-item.active{color:var(--accent)}
.bottom-nav .bnav-item.active .bnav-icon{transform:scale(1.15)}
.bottom-nav .bnav-item.active::after{content:'';position:absolute;top:0;left:20%;right:20%;height:2px;
  background:var(--accent);border-radius:0 0 2px 2px;box-shadow:0 0 8px var(--accent-glow)}
.bottom-nav .bnav-item:active{transform:scale(.92);opacity:.7}
/* Sub-tabs within each bottom nav section */
.sub-tabs{display:flex;gap:4px;margin-bottom:10px;overflow-x:auto;-webkit-overflow-scrolling:touch;padding:2px 0}
.sub-tab{padding:8px 14px;font-size:.82em;font-weight:600;cursor:pointer;border-radius:8px;
  background:var(--bg3);color:var(--dim);border:1px solid var(--glass-border);transition:all .2s;
  white-space:nowrap;backdrop-filter:blur(4px);min-height:36px;display:flex;align-items:center}
.sub-tab:hover{background:var(--bg4);color:var(--text);border-color:rgba(255,255,255,.15)}
.sub-tab.active{background:rgba(255,140,0,.15);color:var(--accent);border-color:rgba(255,140,0,.35);
  box-shadow:0 0 8px var(--accent-glow)}
/* Hide original top tabs (replaced by bottom nav) */
.tabs{display:none!important}
/* Add bottom padding to container so content isn't hidden behind bottom nav */
.container{padding-bottom:72px}
/* Sub-tabs bar visibility */
.sub-tabs:empty{display:none}

/* ============================================================
   ENHANCEMENT 2: Night Vision (Red) Mode
   ============================================================ */
html.nightmode{--bg:rgba(15,0,0,.75);--bg2:rgba(25,0,0,.65);--bg3:rgba(35,0,0,.55);--bg4:rgba(50,0,0,.5);
  --accent:#880000;--accent2:#660000;--accent-glow:rgba(140,0,0,.4);
  --blue:#660000;--green:#553300;--red:#880000;--yellow:#663300;
  --text:#991111;--dim:rgba(120,30,30,.7);
  --glass:rgba(80,0,0,.06);--glass-border:rgba(100,0,0,.15)}
html.nightmode body{background:#0a0000!important}
html.nightmode #starfield{display:none}
html.nightmode header{box-shadow:0 2px 16px rgba(80,0,0,.3)}
html.nightmode .card{background:rgba(20,0,0,.7)!important;border-color:rgba(80,0,0,.2)!important;
  box-shadow:0 2px 12px rgba(0,0,0,.5)!important}
html.nightmode .card:hover{box-shadow:0 4px 20px rgba(40,0,0,.4)!important;border-color:rgba(100,0,0,.3)!important;
  transform:none}
html.nightmode .card h2{color:#880000;text-shadow:0 0 6px rgba(100,0,0,.3)}
html.nightmode header h1{color:#880000;text-shadow:0 0 8px rgba(100,0,0,.3)}
html.nightmode header::before{background:linear-gradient(90deg,#440000,#660000,#440000)!important;
  background-size:300% 100%!important}
html.nightmode .btn{border-color:rgba(80,0,0,.3)}
html.nightmode .btn-accent{background:linear-gradient(135deg,#770000,#550000)!important;color:#cc3333!important;
  border-color:#660000!important;box-shadow:0 0 10px rgba(100,0,0,.3)!important}
html.nightmode .btn-blue{background:rgba(60,0,0,.5)!important;color:#993333!important;
  border-color:#550000!important;box-shadow:0 0 8px rgba(80,0,0,.2)!important}
html.nightmode .btn-green{background:rgba(40,30,0,.5)!important;color:#996633!important;
  border-color:#554400!important;box-shadow:0 0 8px rgba(60,40,0,.2)!important}
html.nightmode .btn-red{background:rgba(60,0,0,.5)!important;color:#993333!important;
  border-color:#660000!important;box-shadow:0 0 8px rgba(80,0,0,.2)!important}
html.nightmode .btn-dim{background:rgba(30,0,0,.5)!important;color:rgba(120,40,40,.7)!important;
  border-color:rgba(80,0,0,.2)!important}
html.nightmode .pos-item{background:rgba(25,0,0,.5);border-color:rgba(80,0,0,.15)}
html.nightmode .pos-value{color:#aa2222}
html.nightmode .pos-label{color:rgba(100,30,30,.6)}
html.nightmode input,html.nightmode select{background:rgba(20,0,0,.6)!important;color:#991111!important;
  border-color:rgba(80,0,0,.2)!important}
html.nightmode input:focus,html.nightmode select:focus{border-color:#660000!important;
  box-shadow:0 0 6px rgba(80,0,0,.3)!important}
html.nightmode .bottom-nav{background:rgba(10,0,0,.95)!important;border-top-color:rgba(80,0,0,.2)!important}
html.nightmode .bottom-nav .bnav-item{color:rgba(100,30,30,.6)}
html.nightmode .bottom-nav .bnav-item.active{color:#880000}
html.nightmode .bottom-nav .bnav-item.active::after{background:#660000}
html.nightmode .sub-tab{background:rgba(30,0,0,.5);color:rgba(100,30,30,.6);border-color:rgba(80,0,0,.15)}
html.nightmode .sub-tab.active{background:rgba(60,0,0,.3);color:#880000;border-color:rgba(100,0,0,.3)}
html.nightmode .dot-green{background:#553300!important;box-shadow:0 0 6px rgba(60,40,0,.5)!important}
html.nightmode .dot-red{background:#660000!important;box-shadow:0 0 6px rgba(80,0,0,.5)!important}
html.nightmode .dot-yellow{background:#664400!important;box-shadow:0 0 6px rgba(80,50,0,.5)!important}
html.nightmode .badge-green{background:rgba(50,30,0,.3);color:#996633;border-color:rgba(80,50,0,.3)}
html.nightmode .badge-red{background:rgba(60,0,0,.3);color:#993333;border-color:rgba(80,0,0,.3)}
html.nightmode .badge-yellow{background:rgba(50,30,0,.3);color:#996633;border-color:rgba(80,50,0,.3)}
html.nightmode .badge-blue{background:rgba(40,0,0,.3);color:#993333;border-color:rgba(60,0,0,.3)}
html.nightmode .toast{border-color:rgba(80,0,0,.2)!important}
html.nightmode .toast-ok{background:rgba(20,10,0,.9)!important;color:#886633!important}
html.nightmode .toast-err{background:rgba(30,0,0,.9)!important;color:#993333!important}
html.nightmode .toast-info{background:rgba(20,0,0,.9)!important;color:#884444!important}
html.nightmode .weather-grid .wi{background:rgba(25,0,0,.5);border-color:rgba(80,0,0,.12)}
html.nightmode .log-box{background:rgba(10,0,0,.6)!important;border-color:rgba(80,0,0,.15)!important}
html.nightmode .log-line{color:rgba(120,30,30,.7)!important;border-bottom-color:rgba(60,0,0,.1)!important}
html.nightmode .log-ts{color:rgba(100,20,20,.4)!important}
html.nightmode .toolbar .btn{background:rgba(15,0,0,.9)!important;border-color:rgba(80,0,0,.2)!important}
html.nightmode .emergency-stop{background:rgba(60,0,0,.8)!important;border-color:#550000!important;
  box-shadow:0 0 12px rgba(60,0,0,.4)!important}
/* Night mode brightness overlay */
.night-dimmer{position:fixed;top:0;left:0;width:100%;height:100%;
  background:rgba(0,0,0,0);pointer-events:none;z-index:9998;transition:background .3s}

/* ============================================================
   ENHANCEMENT 3: Larger Slew Buttons + Better Ergonomics
   ============================================================ */
.slew-grid{display:grid;grid-template-columns:repeat(3,100px)!important;grid-template-rows:repeat(3,90px)!important;
  gap:8px!important;justify-content:center;width:316px!important;margin:0 auto;flex-shrink:0}
.slew-grid .btn{font-size:1.2em!important;border-radius:14px!important;border-width:2px}
.slew-grid .btn .arrow{font-size:1.6em!important}
@media(max-width:400px){
  .slew-grid{grid-template-columns:repeat(3,86px)!important;grid-template-rows:repeat(3,78px)!important;
    gap:6px!important;width:274px!important}
}

/* ============================================================
   ENHANCEMENT 4: Persistent Status Bar
   ============================================================ */
.status-strip{position:sticky;top:0;z-index:800;
  display:flex;align-items:center;justify-content:space-around;
  background:rgba(8,8,18,.88);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  border:1px solid var(--glass-border);border-radius:10px;
  padding:6px 10px;margin-bottom:8px;gap:4px;min-height:40px;
  box-shadow:0 2px 16px rgba(0,0,0,.3)}
.status-strip .ss-item{display:flex;flex-direction:column;align-items:center;gap:1px;min-width:0}
.status-strip .ss-label{font-size:.55em;color:var(--dim);text-transform:uppercase;letter-spacing:1px;
  font-weight:600;white-space:nowrap}
.status-strip .ss-value{font-size:.82em;font-weight:700;font-family:'Consolas',monospace;
  color:var(--text);white-space:nowrap}
.status-strip .ss-divider{width:1px;height:24px;background:var(--glass-border);flex-shrink:0}
html.nightmode .status-strip{background:rgba(10,0,0,.92)!important;border-color:rgba(80,0,0,.15)!important}

/* ============================================================
   ENHANCEMENT 5: Camera Fullscreen Enhancements
   ============================================================ */
.cam-fullscreen .cam-hud{position:absolute;bottom:0;left:0;right:0;
  background:rgba(0,0,0,.7);backdrop-filter:blur(8px);
  padding:8px 12px;display:flex;justify-content:space-around;align-items:center;
  font-size:.78em;color:var(--text);z-index:10000}
.cam-fullscreen .cam-hud .hud-item{text-align:center}
.cam-fullscreen .cam-hud .hud-label{font-size:.7em;color:var(--dim);text-transform:uppercase}
.cam-fullscreen .cam-hud .hud-value{font-weight:700;font-family:monospace}
/* Solve progress ring */
.solve-progress{display:inline-block;width:28px;height:28px;position:relative}
.solve-progress svg{transform:rotate(-90deg)}
.solve-progress circle{fill:none;stroke-width:3}
.solve-progress .bg{stroke:rgba(255,255,255,.1)}
.solve-progress .fg{stroke:var(--accent);stroke-linecap:round;transition:stroke-dashoffset .3s}

/* ============================================================
   ENHANCEMENT 6: GoTo Recent Targets & Favorites
   ============================================================ */
.recent-targets{margin-top:8px}
.recent-targets h3{font-size:.78em;color:var(--dim);text-transform:uppercase;letter-spacing:1px;
  margin-bottom:4px}
.recent-item{display:flex;align-items:center;justify-content:space-between;
  padding:6px 10px;border-radius:6px;background:var(--bg3);border:1px solid var(--glass-border);
  margin-bottom:4px;cursor:pointer;transition:all .15s;font-size:.82em}
.recent-item:hover{background:rgba(255,140,0,.06);border-color:rgba(255,140,0,.15)}
.recent-item .ri-name{color:var(--accent);font-weight:600;flex:1}
.recent-item .ri-time{color:var(--dim);font-size:.8em}
.recent-item .ri-goto{padding:3px 8px;font-size:.72em;background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:#fff;border:none;border-radius:5px;cursor:pointer;font-weight:600;white-space:nowrap}

/* ============================================================
   ENHANCEMENT 7: Two-Column Responsive Layout for Tablets
   ============================================================ */
@media(min-width:768px) and (orientation:landscape){
  .container{max-width:1200px!important;padding:8px 16px}
  .dual-col{display:grid!important;grid-template-columns:1fr 1fr;gap:10px;align-items:start}
  .dual-col>.card{margin-bottom:0}
  .pos-grid{grid-template-columns:repeat(4,1fr)!important}
  .status-strip{max-width:1200px;margin-left:auto;margin-right:auto}
}
@media(min-width:1024px){
  .container{max-width:1400px!important}
  .triple-col{display:grid!important;grid-template-columns:1fr 1fr 1fr;gap:10px;align-items:start}
}

/* ============================================================
   ENHANCEMENT 8: Accessibility (Large Text + High Contrast)
   ============================================================ */
html.largetext{font-size:18px}
html.largetext .pos-value{font-size:1.6em}
html.largetext .btn{font-size:1em;min-height:48px;padding:12px 18px}
html.largetext .card h2{font-size:1.05em}
html.largetext .text-dim{font-size:.9em}
html.largetext .input-row label{font-size:.9em}
html.largetext .input-row input,html.largetext .input-row select{font-size:.95em;min-height:42px}
html.largetext .bottom-nav .bnav-item .bnav-label{font-size:.7em}
html.largetext .bottom-nav .bnav-item .bnav-icon{font-size:1.5em}
html.largetext .status-strip .ss-value{font-size:.95em}
html.largetext .sub-tab{font-size:.9em;padding:10px 16px}
/* High contrast mode */
html.highcontrast .card{border-width:2px!important;border-color:rgba(255,140,0,.35)!important;
  backdrop-filter:none!important;background:rgba(5,5,16,.9)!important}
html.highcontrast .pos-item{border-width:2px!important;background:rgba(5,5,16,.9)!important}
html.highcontrast .pos-value{text-shadow:none!important;font-weight:800}
html.highcontrast .btn{border-width:2px!important;backdrop-filter:none!important;font-weight:700}
html.highcontrast .card h2{text-shadow:none}
html.highcontrast header{backdrop-filter:none!important;background:rgba(5,5,16,.95)!important}

/* ============================================================
   ENHANCEMENT 10: Emergency Stop + Micro-Interactions
   ============================================================ */
.emergency-stop{position:fixed;bottom:68px;left:12px;z-index:950;
  width:56px;height:56px;border-radius:50%;
  background:linear-gradient(135deg,#cc0000,#990000);
  color:#fff;border:3px solid #ff3333;
  font-size:.65em;font-weight:900;letter-spacing:.5px;
  cursor:pointer;display:flex;align-items:center;justify-content:center;
  box-shadow:0 4px 20px rgba(200,0,0,.4),0 0 30px rgba(200,0,0,.15);
  transition:all .15s;touch-action:manipulation;user-select:none;
  text-transform:uppercase;line-height:1.1;text-align:center;
  animation:emergencyPulse 3s ease-in-out infinite}
.emergency-stop:active{transform:scale(.9);box-shadow:0 2px 10px rgba(200,0,0,.6)}
.emergency-stop:hover{box-shadow:0 6px 30px rgba(200,0,0,.5),0 0 50px rgba(200,0,0,.2)}
@keyframes emergencyPulse{0%,100%{box-shadow:0 4px 20px rgba(200,0,0,.4),0 0 30px rgba(200,0,0,.15)}
  50%{box-shadow:0 4px 20px rgba(200,0,0,.5),0 0 40px rgba(200,0,0,.25)}}
/* Connection dot enhanced animation */
@keyframes connPulse{0%,100%{transform:scale(1);opacity:1}50%{transform:scale(1.3);opacity:.7}}
.dot-green{animation:connPulse 2s ease-in-out infinite!important}
/* Toast improvements -- slide from bottom on mobile */
@media(max-width:600px){
  .toast-panel{bottom:72px!important;left:8px!important;right:8px!important;max-width:none}
  .toast{max-width:none}
}
/* Enhanced toolbar positioning to account for bottom nav */
.toolbar{bottom:72px!important}
</style>
</head>
<body>
<canvas id="starfield"></canvas>

<div class="container">

<!-- Header -->
<header>
  <div style="display:flex;flex-direction:column">
    <h1>TrackWise-AltAzPro</h1>
    <span style="font-size:.65em;color:var(--dim);font-weight:400;letter-spacing:.5px;margin-top:1px">by CRACIUN BOGDAN</span>
  </div>
  <div style="display:flex;align-items:center;gap:12px">
    <span id="hdr-clock" style="font-size:.82em;color:var(--dim);font-family:monospace;min-width:58px">--:--:--</span>
    <div class="header-status">
      <span class="status-dot dot-red" id="hdr-dot"></span>
      <span id="hdr-status">Disconnected</span>
    </div>
  </div>
</header>

<!-- ENHANCEMENT 4: Persistent Status Strip -->
<div class="status-strip" id="status-strip">
  <div class="ss-item"><span class="ss-label">RA</span><span class="ss-value" id="ss-ra">--</span></div>
  <div class="ss-divider"></div>
  <div class="ss-item"><span class="ss-label">Dec</span><span class="ss-value" id="ss-dec">--</span></div>
  <div class="ss-divider"></div>
  <div class="ss-item"><span class="ss-label">Alt</span><span class="ss-value" id="ss-alt">--</span></div>
  <div class="ss-divider"></div>
  <div class="ss-item"><span class="ss-label">Az</span><span class="ss-value" id="ss-az">--</span></div>
  <div class="ss-divider"></div>
  <div class="ss-item"><span class="ss-label">Tracking</span><span class="ss-value" id="ss-tracking" style="color:var(--red)">OFF</span></div>
  <div class="ss-divider"></div>
  <div class="ss-item"><span class="ss-label">RMS</span><span class="ss-value" id="ss-rms">--</span></div>
</div>

<!-- Position Card -->
<div class="card">
  <h2>Position</h2>
  <div class="pos-grid">
    <div class="pos-item"><div class="pos-label">Right Ascension</div><div class="pos-value" id="p-ra">--</div></div>
    <div class="pos-item"><div class="pos-label">Declination</div><div class="pos-value" id="p-dec">--</div></div>
    <div class="pos-item"><div class="pos-label">Altitude</div><div class="pos-value" id="p-alt">--</div></div>
    <div class="pos-item"><div class="pos-label">Azimuth</div><div class="pos-value" id="p-az">--</div></div>
  </div>
  <div class="pos-grid mt-1">
    <div class="pos-item"><div class="pos-label">Alt Rate</div><div class="pos-value" id="p-rate-alt" style="font-size:.9em">--</div></div>
    <div class="pos-item"><div class="pos-label">Az Rate</div><div class="pos-value" id="p-rate-az" style="font-size:.9em">--</div></div>
  </div>
  <div id="target-row" class="mt-1" style="display:none;padding:6px 10px;border-radius:8px;background:rgba(74,158,255,.08);border:1px solid rgba(74,158,255,.15)">
    <div class="flex-between">
      <span class="text-dim" style="font-size:.75em">TARGET</span>
      <span id="p-target-badge" class="badge badge-yellow" style="font-size:.65em;padding:2px 8px">Slewing</span>
    </div>
    <div style="font-size:.95em;font-weight:600;color:var(--blue);margin-top:2px" id="p-target-name">--</div>
    <div class="text-dim" style="font-size:.75em;margin-top:1px" id="p-target-coords">--</div>
  </div>
</div>

<!-- Legacy tabs (hidden, kept for JS compatibility) -->
<div class="tabs">
  <div class="tab active" data-tab="control">Control</div>
  <div class="tab" data-tab="camera">Camera</div>
  <div class="tab" data-tab="skychart">Sky Chart</div>
  <div class="tab" data-tab="goto">GoTo</div>
  <div class="tab" data-tab="tracking">Tracking</div>
  <div class="tab" data-tab="telemetry">Telemetry</div>
  <div class="tab" data-tab="weather">Location</div>
  <div class="tab" data-tab="log">Log</div>
  <div class="tab" data-tab="help">&#128214; Help</div>
</div>

<!-- ENHANCEMENT 1: Dynamic Sub-Tabs (populated by JS based on active bottom nav) -->
<div class="sub-tabs" id="sub-tabs-bar"></div>

<!-- Tab: Control -->
<div class="tab-content active" id="tab-control">
  <!-- Connection -->
  <div class="card">
    <h2>Connection</h2>
    <div class="flex-between">
      <span class="text-dim" id="conn-info">--</span>
      <span id="conn-badge" class="badge badge-red">Disconnected</span>
    </div>
    <div class="input-row mt-1">
      <label>Mode</label>
      <select id="conn-type" onchange="onConnTypeChange()">
        <option value="USB">USB / Serial</option>
        <option value="WiFi">WiFi / TCP</option>
      </select>
    </div>
    <div class="input-row">
      <label>Protocol</label>
      <select id="conn-protocol" onchange="onProtocolChange()">
        <option value="lx200">LX200 / OnStep</option>
        <option value="nexstar">NexStar / SynScan</option>
        <option value="ioptron">iOptron (AZ Mount Pro)</option>
        <option value="audiostar">Meade AudioStar</option>
        <option value="alpaca">ASCOM Alpaca</option>
        <option value="indi">INDI</option>
      </select>
    </div>
    <div id="conn-usb-fields">
      <div class="input-row">
        <label>Port</label>
        <div style="display:flex;gap:4px;flex:1">
          <select id="conn-port" style="flex:1"><option value="">-- scan ports --</option></select>
          <button class="btn btn-dim" onclick="refreshSerialPorts()" title="Refresh ports" style="padding:4px 8px;min-width:0">&#x21bb;</button>
        </div>
      </div>
      <div class="input-row">
        <label>Baud</label>
        <select id="conn-baud">
          <option value="9600" selected>9600</option>
          <option value="19200">19200</option>
          <option value="38400">38400</option>
          <option value="57600">57600</option>
          <option value="115200">115200</option>
        </select>
      </div>
    </div>
    <div id="conn-wifi-fields" style="display:none">
      <div class="input-row">
        <label>IP</label>
        <input type="text" id="conn-wifi-ip" placeholder="e.g. 192.168.0.1" value="">
      </div>
      <div class="input-row">
        <label>Port</label>
        <input type="number" id="conn-wifi-port" placeholder="e.g. 9996" value="">
      </div>
      <div class="text-dim" style="font-size:.72em;margin-top:4px;padding:4px 6px;background:rgba(255,200,0,.12);border-radius:4px">
        Note: Close the SmartWebServer web page (mount.htm) in your browser
        before connecting. It saturates the serial line and blocks commands.
      </div>
    </div>
    <div class="btn-row mt-1">
      <button class="btn btn-green" id="btn-connect" onclick="doConnect()">Connect</button>
      <button class="btn btn-red" id="btn-disconnect" onclick="doDisconnect()" disabled>Disconnect</button>
      <button class="btn btn-dim" id="btn-sim" onclick="doSimulator()">Simulator</button>
    </div>
    <div style="margin-top:6px;display:flex;align-items:center;gap:6px">
      <input type="checkbox" id="auto-connect-chk" onchange="saveAutoConnect()">
      <label for="auto-connect-chk" class="text-dim" style="font-size:.78em;cursor:pointer">
        Auto-connect on launch (remembers last settings)
      </label>
    </div>
  </div>

  <!-- Slew Pad -->
  <div class="card">
    <h2>Slew Control</h2>
    <div style="display:flex;align-items:start;gap:12px;flex-wrap:wrap;justify-content:center">
      <div>
        <div class="slew-grid">
          <div></div>
          <button class="btn btn-dim" ontouchstart="slewStartHaptic('N',event)" ontouchend="slewStopHaptic(event)" ontouchcancel="slewStopHaptic(event)" onmousedown="slewStart('N')" onmouseup="slewStop()"><span class="arrow">&#9650;</span>N</button>
          <div></div>
          <button class="btn btn-dim" ontouchstart="slewStartHaptic('W',event)" ontouchend="slewStopHaptic(event)" ontouchcancel="slewStopHaptic(event)" onmousedown="slewStart('W')" onmouseup="slewStop()"><span class="arrow">&#9664;</span>W</button>
          <button class="btn btn-red" onclick="slewStop()" style="font-size:.85em;font-weight:800">STOP</button>
          <button class="btn btn-dim" ontouchstart="slewStartHaptic('E',event)" ontouchend="slewStopHaptic(event)" ontouchcancel="slewStopHaptic(event)" onmousedown="slewStart('E')" onmouseup="slewStop()"><span class="arrow">&#9654;</span>E</button>
          <div></div>
          <button class="btn btn-dim" ontouchstart="slewStartHaptic('S',event)" ontouchend="slewStopHaptic(event)" ontouchcancel="slewStopHaptic(event)" onmousedown="slewStart('S')" onmouseup="slewStop()"><span class="arrow">&#9660;</span>S</button>
          <div></div>
        </div>
      </div>
      <div style="display:flex;flex-direction:column;gap:6px;min-width:120px">
        <div class="input-row" style="margin:0">
          <label>Speed</label>
          <select id="slew-speed">
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3">3</option>
            <option value="4" selected>4</option>
          </select>
        </div>
        <button class="btn btn-accent btn-sm" onclick="toggleFullscreenSlew()" style="font-size:.9em" title="Open fullscreen directional cross for blind operation">&#11017; Fullscreen</button>
        <button class="btn btn-dim btn-sm" onclick="confirmPark()">Park</button>
        <button class="btn btn-dim btn-sm" onclick="confirmHome()">Home</button>
        <div class="text-dim" style="font-size:.8em" id="slew-status">Stopped</div>
      </div>
    </div>
  </div>

  <!-- Focuser + Derotator side by side -->
  <div class="ctrl-cols">
    <div class="card">
      <h2>Focuser</h2>
      <div class="flex-between">
        <span class="text-dim" id="focus-pos">--</span>
        <span class="text-dim" id="focus-status">--</span>
      </div>
      <div class="btn-row mt-1">
        <button class="btn btn-dim btn-sm" ontouchstart="focusMove('IN')" ontouchend="focusStop()" ontouchcancel="focusStop()" onmousedown="focusMove('IN')" onmouseup="focusStop()">In</button>
        <button class="btn btn-red btn-sm" onclick="focusStop()">Stop</button>
        <button class="btn btn-dim btn-sm" ontouchstart="focusMove('OUT')" ontouchend="focusStop()" ontouchcancel="focusStop()" onmousedown="focusMove('OUT')" onmouseup="focusStop()">Out</button>
      </div>
      <!-- Extended Focuser (OnStepX) -->
      <div id="focuser-extended" style="border-top:1px solid #333;margin-top:8px;padding-top:8px">
        <div class="text-dim" style="font-size:.75em">
          Temp: <span id="focus-temp">--</span> &nbsp;|&nbsp;
          TCF: <span id="focus-tcf">Off</span>
        </div>
        <div class="input-row mt-1" style="margin:4px 0">
          <label style="font-size:.8em;min-width:40px">GoTo</label>
          <input type="number" id="focus-goto-pos" placeholder="position" style="max-width:80px;font-size:.8em">
          <button class="btn btn-dim btn-sm" onclick="focuserGoto()" style="font-size:.72em;padding:3px 6px">Go</button>
        </div>
        <div class="btn-row" style="margin-top:4px">
          <button class="btn btn-dim btn-sm" onclick="apiPost('/api/focuser/home')" style="font-size:.72em;padding:3px 6px">Home</button>
          <button class="btn btn-dim btn-sm" onclick="apiPost('/api/focuser/sethome')" style="font-size:.72em;padding:3px 6px">Set Home</button>
          <button class="btn btn-dim btn-sm" onclick="apiPost('/api/focuser/zero')" style="font-size:.72em;padding:3px 6px">Zero</button>
        </div>
        <div class="btn-row" style="margin-top:4px">
          <button class="btn btn-dim btn-sm" onclick="focuserToggleTCF()" style="font-size:.72em;padding:3px 6px" id="focus-tcf-btn">TCF On/Off</button>
          <select id="focuser-select" onchange="focuserSelect()" style="font-size:.8em;max-width:70px">
            <option value="1">F1</option><option value="2">F2</option>
            <option value="3">F3</option><option value="4">F4</option>
            <option value="5">F5</option><option value="6">F6</option>
          </select>
        </div>
      </div>
    </div>
    <div class="card">
      <h2>Derotator</h2>
      <div class="flex-between">
        <span class="text-dim" id="derot-angle">--</span>
        <span class="text-dim" id="derot-status">--</span>
      </div>
      <div class="btn-row mt-1">
        <button class="btn btn-dim btn-sm" ontouchstart="derotRotate('CCW')" ontouchend="derotStop()" ontouchcancel="derotStop()" onmousedown="derotRotate('CCW')" onmouseup="derotStop()">CCW</button>
        <button class="btn btn-red btn-sm" onclick="derotStop()">Stop</button>
        <button class="btn btn-dim btn-sm" ontouchstart="derotRotate('CW')" ontouchend="derotStop()" ontouchcancel="derotStop()" onmousedown="derotRotate('CW')" onmouseup="derotStop()">CW</button>
        <button class="btn btn-dim btn-sm" onclick="derotSync()">Sync</button>
      </div>
    </div>
  </div>

  <!-- Rotator (OnStepX hardware rotator) -->
  <div class="card" id="rotator-card">
    <h2>Rotator (OnStepX)</h2>
    <div class="flex-between">
      <span class="text-dim">Angle: <span id="rot-angle">--</span></span>
      <span class="text-dim" id="rot-status">Stopped</span>
    </div>
    <div class="btn-row mt-1">
      <button class="btn btn-dim btn-sm" ontouchstart="rotatorMove('CCW')" ontouchend="rotatorStop()" ontouchcancel="rotatorStop()" onmousedown="rotatorMove('CCW')" onmouseup="rotatorStop()">CCW</button>
      <button class="btn btn-red btn-sm" onclick="rotatorStop()">Stop</button>
      <button class="btn btn-dim btn-sm" ontouchstart="rotatorMove('CW')" ontouchend="rotatorStop()" ontouchcancel="rotatorStop()" onmousedown="rotatorMove('CW')" onmouseup="rotatorStop()">CW</button>
    </div>
    <div class="input-row mt-1" style="margin:4px 0">
      <label style="font-size:.8em;min-width:40px">GoTo</label>
      <input type="number" id="rot-goto-angle" placeholder="degrees" step="0.1" style="max-width:80px;font-size:.8em">
      <button class="btn btn-dim btn-sm" onclick="rotatorGoto()" style="font-size:.72em;padding:3px 6px">Go</button>
    </div>
    <div class="btn-row mt-1">
      <button class="btn btn-dim btn-sm" onclick="apiPost('/api/rotator/zero')" style="font-size:.72em;padding:3px 6px">Zero</button>
      <button class="btn btn-dim btn-sm" onclick="apiPost('/api/rotator/derotate')" id="rot-derotate-btn" style="font-size:.72em;padding:3px 6px">Derotate On/Off</button>
      <button class="btn btn-dim btn-sm" onclick="apiPost('/api/rotator/reverse')" style="font-size:.72em;padding:3px 6px">Reverse</button>
      <button class="btn btn-dim btn-sm" onclick="apiPost('/api/rotator/parallactic')" style="font-size:.72em;padding:3px 6px">PA</button>
    </div>
  </div>

  <!-- Park / Unpark / Home (OnStepX extended) -->
  <div class="card" id="park-card">
    <h2>Park / Home</h2>
    <div class="flex-between">
      <span class="text-dim">Park: <span id="park-state">Unknown</span></span>
    </div>
    <div class="btn-row mt-1">
      <button class="btn btn-dim btn-sm" onclick="confirmPark()">Park</button>
      <button class="btn btn-dim btn-sm" onclick="doUnpark()">Unpark</button>
      <button class="btn btn-dim btn-sm" onclick="apiPost('/api/park/set')">Set Park Pos</button>
    </div>
    <div class="btn-row mt-1">
      <button class="btn btn-dim btn-sm" onclick="confirmHome()">Home (Reset)</button>
      <button class="btn btn-dim btn-sm" onclick="doHomeFind()">Find Home</button>
    </div>
  </div>

  <!-- Firmware Info + Reticle -->
  <div class="ctrl-cols">
    <div class="card" id="firmware-card">
      <h2>Firmware</h2>
      <div class="text-dim" style="font-size:.82em">
        <div><strong id="fw-name">--</strong> v<span id="fw-version">--</span></div>
        <div>Mount: <span id="fw-mount-type">--</span></div>
      </div>
      <button class="btn btn-dim btn-sm mt-1" onclick="apiPost('/api/mount/firmware/refresh')" style="font-size:.72em;padding:3px 6px">Refresh</button>
    </div>
    <div class="card" id="reticle-card">
      <h2>Reticle / LED</h2>
      <div class="btn-row mt-1">
        <button class="btn btn-dim btn-sm" onclick="apiPost('/api/reticle/brighter')">Brighter</button>
        <button class="btn btn-dim btn-sm" onclick="apiPost('/api/reticle/dimmer')">Dimmer</button>
      </div>
    </div>
  </div>
</div>

<!-- Tab: Sky Chart -->
<div class="tab-content" id="tab-skychart">
  <div class="sc-wrap" id="sc-wrap">
    <canvas class="sc-canvas" id="sc-canvas"></canvas>
    <div class="sc-fov" id="sc-fov">FOV 90&deg;</div>
    <div class="sc-controls">
      <button onclick="scToggleFullscreen()" id="sc-btn-fullscreen" title="Fullscreen">&#x26F6;</button>
      <button onclick="scToggleGrid()" id="sc-btn-grid" title="Alt/Az grid">&#9783;</button>
      <button onclick="scToggleConst()" id="sc-btn-const" title="Constellations" class="sc-active">&#9734;</button>
      <button onclick="scToggleLabels()" id="sc-btn-labels" title="Labels" class="sc-active">Aa</button>
      <button onclick="scToggleDSOs()" id="sc-btn-dso" title="Deep sky objects" class="sc-active">&#11044;</button>
      <button onclick="scToggleEcliptic()" id="sc-btn-ecliptic" title="Ecliptic line">&#9788;</button>
      <button onclick="scToggleEquator()" id="sc-btn-equator" title="Celestial equator">&#8853;</button>
      <button onclick="scFollowScope()" id="sc-btn-follow" title="Follow telescope">&#8982;</button>
      <button onclick="scResetView()" title="Reset view">&#8634;</button>
    </div>
    <div class="sc-info" id="sc-info">
      <div class="sc-info-name" id="sc-info-name"></div>
      <div class="sc-info-detail" id="sc-info-detail"></div>
      <div class="sc-info-row">
        <span class="text-dim" id="sc-info-coords"></span>
        <div style="display:flex;gap:6px">
          <button class="btn btn-dim" onclick="scCloseInfo()">Close</button>
          <button class="btn btn-accent" onclick="scGotoSelected()">GoTo</button>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Tab: GoTo -->
<div class="tab-content" id="tab-goto">
  <div class="card">
    <h2>GoTo Target</h2>
    <div class="search-box">
      <input type="text" id="goto-search" placeholder="Search (M31, Polaris, Jupiter, Moon, NGC 7000)..."
             oninput="catalogSearch(this.value)" autocomplete="off">
      <div class="search-results" id="search-results"></div>
    </div>
    <!-- Browse by category -->
    <div class="text-dim mt-1" style="font-size:.8em">Or browse by catalog:</div>
    <div class="cat-chips" id="cat-chips"></div>
    <div class="browse-list" id="browse-list"></div>
    <div class="browse-pager" id="browse-pager" style="display:none">
      <button onclick="browsePage(-1)" id="browse-prev" disabled>&laquo; Prev</button>
      <span id="browse-page-info">1 / 1</span>
      <button onclick="browsePage(1)" id="browse-next" disabled>Next &raquo;</button>
    </div>

    <div class="text-dim mt-1" style="font-size:.8em">Or enter coordinates manually:</div>
    <div class="input-row mt-1">
      <label>RA</label>
      <input type="text" id="goto-ra" placeholder="HH:MM:SS">
    </div>
    <div class="input-row">
      <label>Dec</label>
      <input type="text" id="goto-dec" placeholder="+DD*MM:SS">
    </div>
    <button class="btn btn-accent btn-block mt-1" onclick="doGoto()">GoTo</button>
    <div class="text-dim text-center mt-1" id="goto-status"></div>

    <!-- ENHANCEMENT 6: Recent Targets -->
    <div class="recent-targets" id="recent-targets" style="display:none">
      <h3>Recent Targets</h3>
      <div id="recent-targets-list"></div>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════
     Nav: Imaging
     ═══════════════════════════════════════════════════════════════ -->
<!-- Tab: Camera (live focus view) -->
<div class="tab-content" id="tab-camera">
  <div class="card">
    <h2>Live Camera View</h2>
    <div class="input-row">
      <label>Source</label>
      <select id="lv-cam-source" onchange="onCamSourceChange()" style="max-width:130px">
        <option value="uvc" selected>UVC / USB</option>
        <option value="asi">ZWO ASI (SDK)</option>
        <option value="ascom">ASCOM</option>
      </select>
      <button class="btn btn-green btn-sm" id="btn-cam-start" onclick="camStart()">Start</button>
      <button class="btn btn-red btn-sm" id="btn-cam-stop" onclick="camStop()" disabled>Stop</button>
    </div>
    <!-- UVC settings (desktop only — hidden on Android) -->
    <div id="cam-uvc-fields">
      <div class="input-row mt-1">
        <label>Cam #</label>
        <input type="number" id="lv-cam-idx" value="0" min="0" max="9" style="max-width:60px">
      </div>
    </div>
    <!-- ASCOM settings -->
    <div id="cam-ascom-fields" style="display:none">
      <div class="input-row mt-1">
        <label>Camera</label>
        <select id="lv-ascom-id" style="max-width:200px;font-size:.85em">
          <option value="">-- select ASCOM camera --</option>
        </select>
        <button class="btn btn-dim btn-sm" onclick="camRefreshAscom()" title="Refresh list">&#x21bb;</button>
      </div>
      <div class="input-row mt-1">
        <label>Exp (s)</label>
        <input type="number" id="lv-ascom-exp" value="0.5" step="0.1" min="0.01" max="30" style="max-width:70px">
      </div>
      <div class="input-row mt-1">
        <label>Gain</label>
        <input type="number" id="lv-ascom-gain" value="100" step="10" min="0" max="600" style="max-width:70px">
      </div>
      <div class="input-row mt-1">
        <label>Bin</label>
        <select id="lv-ascom-bin" style="max-width:60px">
          <option value="1">1</option>
          <option value="2" selected>2</option>
          <option value="3">3</option>
          <option value="4">4</option>
        </select>
        <button class="btn btn-dim btn-sm" onclick="camApplyAscomSettings()">Apply</button>
      </div>
    </div>
    <!-- ASI SDK camera controls (shown when ASI camera active) -->
    <div id="cam-asi-controls" style="display:none">
      <div class="input-row mt-1">
        <label style="min-width:55px">Exp (ms)</label>
        <input type="range" id="asi-exp-slider" min="0" max="4000" step="1" value="100"
               oninput="document.getElementById('asi-exp-val').value=this.value" style="flex:1">
        <input type="number" id="asi-exp-val" min="0" max="600000" step="1" value="100"
               oninput="document.getElementById('asi-exp-slider').value=Math.min(this.value,4000)"
               style="max-width:70px">
      </div>
      <div class="input-row mt-1">
        <label style="min-width:55px">Gain</label>
        <input type="range" id="asi-gain-slider" min="0" max="300" step="1" value="50"
               oninput="document.getElementById('asi-gain-val').textContent=this.value" style="flex:1">
        <span id="asi-gain-val" style="min-width:30px;text-align:right">50</span>
      </div>
      <div class="input-row mt-1">
        <label style="min-width:55px">Gamma</label>
        <input type="range" id="asi-gamma-slider" min="0" max="100" step="1" value="50"
               oninput="document.getElementById('asi-gamma-val').textContent=this.value" style="flex:1">
        <span id="asi-gamma-val" style="min-width:30px;text-align:right">50</span>
      </div>
      <div class="input-row mt-1">
        <label style="min-width:55px">Offset</label>
        <input type="range" id="asi-offset-slider" min="0" max="255" step="1" value="0"
               oninput="document.getElementById('asi-offset-val').textContent=this.value" style="flex:1">
        <span id="asi-offset-val" style="min-width:30px;text-align:right">0</span>
      </div>
      <div class="input-row mt-1">
        <label style="min-width:55px">Flip</label>
        <select id="asi-flip" style="max-width:100px">
          <option value="0">None</option>
          <option value="1">Horiz</option>
          <option value="2">Vert</option>
          <option value="3">Both</option>
        </select>
        <button class="btn btn-green btn-sm" onclick="camApplyAsiSettings()">Apply</button>
        <button class="btn btn-dim btn-sm" onclick="camReadAsiSettings()" title="Read current values">&#x21bb;</button>
      </div>
      <div class="text-dim text-center" style="font-size:.75em" id="asi-info">--</div>
    </div>
    <div id="cam-view-wrap">
      <div class="cam-placeholder" id="cam-placeholder">Camera off &mdash; press Start to begin live view</div>
      <div class="cam-container" id="cam-container" style="display:none">
        <img class="cam-img" id="cam-img" alt="Live view">
        <div class="cam-overlay">
          <button class="btn btn-dim" onclick="camSnapshot()" title="Save snapshot">&#128247;</button>
          <button class="btn btn-dim" onclick="camFullscreen()" title="Fullscreen" id="btn-cam-fs">&#x26F6;</button>
        </div>
      </div>
    </div>
    <div class="text-dim text-center mt-1" id="cam-status">Camera inactive</div>
  </div>

  <!-- Focuser shortcut in camera tab for live focusing -->
  <div class="card">
    <h2>Focus Control</h2>
    <div class="flex-between">
      <span class="text-dim" id="cam-focus-pos">--</span>
      <span class="text-dim" id="cam-focus-status">--</span>
    </div>
    <div class="btn-row mt-1">
      <button class="btn btn-dim" style="min-height:52px;font-size:1.1em"
        ontouchstart="focusMove('IN')" ontouchend="focusStop()" ontouchcancel="focusStop()"
        onmousedown="focusMove('IN')" onmouseup="focusStop()">&#9664; Focus In</button>
      <button class="btn btn-red btn-sm" onclick="focusStop()">Stop</button>
      <button class="btn btn-dim" style="min-height:52px;font-size:1.1em"
        ontouchstart="focusMove('OUT')" ontouchend="focusStop()" ontouchcancel="focusStop()"
        onmousedown="focusMove('OUT')" onmouseup="focusStop()">Focus Out &#9654;</button>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════
     Nav: Tracking
     ═══════════════════════════════════════════════════════════════ -->
<!-- Tab: Tracking -->
<div class="tab-content" id="tab-tracking">
  <div class="card">
    <h2>Tracking Control</h2>
    <div class="btn-row">
      <button class="btn btn-green" id="btn-track-start" onclick="doTrackStart()">Start Tracking</button>
      <button class="btn btn-red" id="btn-track-stop" onclick="doTrackStop()" disabled>Stop Tracking</button>
      <button class="btn btn-dim" onclick="doSessionSave()" title="Save session data (telemetry, solves, logs, config)">Save Session</button>
      <button id="session-download-link" class="btn btn-dim" onclick="doSessionDownload()" style="display:none" title="Download session as ZIP">Download</button>
    </div>
    <div class="pos-grid mt-1">
      <div class="pos-item"><div class="pos-label">Drift</div><div class="pos-value" id="t-drift" style="font-size:.9em">--</div></div>
      <div class="pos-item"><div class="pos-label">RMS</div><div class="pos-value" id="t-rms" style="font-size:.9em">--</div></div>
      <div class="pos-item"><div class="pos-label">Solve Time</div><div class="pos-value" id="t-solve" style="font-size:.9em">--</div></div>
      <div class="pos-item"><div class="pos-label">Solve Rate</div><div class="pos-value" id="t-rate" style="font-size:.9em">--</div></div>
    </div>
    <div class="flex-between mt-1">
      <span class="text-dim">ML Samples: <span id="t-ml">--</span></span>
      <span class="text-dim">Auto-solve: <span id="t-auto">--</span></span>
    </div>
    <div class="flex-between" style="margin-top:.3em">
      <span class="text-dim" style="font-size:.78em">Solved FOV: <span id="t-solved-fov" style="font-weight:600">--</span></span>
    </div>
  </div>

  <!-- Auto Plate Solve -->
  <div class="card">
    <h2>Auto Plate Solve</h2>
    <div class="input-row">
      <label>Source</label>
      <select id="solve-mode" onchange="onSolveModeChange()">
        <option value="camera" selected>Camera</option>
        <option value="ascom">ASCOM Camera</option>
      </select>
    </div>
    <div id="solve-camera-fields">
      <div class="input-row">
        <label>Cam #</label>
        <input type="number" id="cam-index" value="0" min="0" max="9" style="max-width:60px">
      </div>
    </div>
    <div id="solve-ascom-fields" style="display:none">
      <div class="flex-between">
        <div class="text-dim" style="font-size:.8em">ASCOM: <span id="ascom-cam-name">No camera selected</span></div>
        <button class="btn btn-dim btn-sm" onclick="loadAscomCameras()" style="font-size:.75em;padding:4px 8px;min-height:0">Choose...</button>
      </div>
      <div id="ascom-cam-list" style="display:none" class="mt-1"></div>
      <div class="input-row mt-1">
        <label>Exp (s)</label>
        <input type="number" id="ascom-exp" value="0.5" step="0.1" min="0.1" style="max-width:70px">
      </div>
      <div class="input-row">
        <label>Gain</label>
        <input type="number" id="ascom-gain" value="100" min="0" style="max-width:70px">
      </div>
      <div class="input-row">
        <label>Bin</label>
        <select id="ascom-bin">
          <option value="1">1x1</option>
          <option value="2" selected>2x2</option>
          <option value="3">3x3</option>
          <option value="4">4x4</option>
        </select>
      </div>
    </div>
    <div class="input-row mt-1">
      <label>Interval</label>
      <input type="number" id="solve-interval" value="4.0" step="0.1" min="0.1" style="max-width:70px">
      <span class="text-dim">sec</span>
    </div>
    <div class="btn-row mt-1">
      <button class="btn btn-blue" id="btn-solve-start" onclick="doSolveStart()">Start Solving</button>
      <button class="btn btn-red" id="btn-solve-stop" onclick="doSolveStop()" disabled>Stop Solving</button>
    </div>
    <button class="btn btn-dim btn-sm btn-block mt-1" onclick="applyCameraSettings()">Apply Camera Settings</button>
  </div>

  <!-- Mount Drive & PEC -->
  <div class="card">
    <h2>Mount Drive & PEC</h2>
    <div class="input-row">
      <label style="font-size:.85em">Drive Type</label>
      <select id="mount-drive-type" onchange="setDriveType(this.value)" style="font-size:.85em">
        <option value="worm_gear">Worm Gear</option>
        <option value="planetary_gearbox">Planetary Gearbox</option>
        <option value="harmonic_drive">Harmonic Drive</option>
        <option value="belt_drive">Belt Drive</option>
        <option value="direct_drive">Direct Drive</option>
      </select>
    </div>
    <div class="flex-between mt-1">
      <span class="text-dim">PEC: <span id="pec-status">--</span></span>
      <span id="pec-badge" class="badge badge-red">Off</span>
    </div>
    <div class="text-dim mt-1" id="pec-periods" style="font-size:.8em">--</div>
    <div class="text-dim" id="pec-corr" style="font-size:.8em">--</div>
    <div class="btn-row mt-1">
      <button class="btn btn-dim btn-sm" onclick="apiPost('/api/pec/toggle')">Toggle PEC</button>
      <button class="btn btn-dim btn-sm" onclick="apiPost('/api/pec/save')">Save</button>
      <button class="btn btn-dim btn-sm" onclick="apiPost('/api/pec/load')">Load</button>
      <button class="btn btn-dim btn-sm" onclick="apiPost('/api/pec/reset')">Reset</button>
    </div>
    <hr style="border-color:#333;margin:.6em 0">
    <div class="flex-between">
      <span class="text-dim" style="font-size:.85em">Flexure Learning</span>
      <span id="flexure-badge" class="badge badge-red">Off</span>
    </div>
    <div class="text-dim" style="font-size:.8em">
      Samples: <span id="flexure-samples">0</span> |
      Sky coverage: <span id="flexure-coverage">0</span>%
    </div>
    <div class="btn-row mt-1">
      <button class="btn btn-dim btn-sm" onclick="apiPost('/api/flexure/toggle')">Toggle Flexure</button>
      <button class="btn btn-dim btn-sm" onclick="apiPost('/api/flexure/reset')">Reset Flexure</button>
    </div>
  </div>

  <!-- Mount-Side PEC (OnStepX hardware PEC) -->
  <div class="card" id="mount-pec-card">
    <h2>Mount PEC (OnStepX)</h2>
    <div class="flex-between">
      <span class="text-dim">Status: <span id="mpec-status">--</span></span>
      <span id="mpec-badge" class="badge badge-red">Idle</span>
    </div>
    <div class="text-dim" style="font-size:.8em">
      PEC Data Recorded: <span id="mpec-recorded">No</span>
    </div>
    <div class="btn-row mt-1">
      <button class="btn btn-dim btn-sm" onclick="apiPost('/api/mount/pec/play')">Play</button>
      <button class="btn btn-dim btn-sm" onclick="apiPost('/api/mount/pec/stop')">Stop</button>
      <button class="btn btn-dim btn-sm" onclick="apiPost('/api/mount/pec/record')">Record</button>
      <button class="btn btn-dim btn-sm" onclick="apiPost('/api/mount/pec/record/stop')">Rec Stop</button>
    </div>
    <div class="btn-row mt-1">
      <button class="btn btn-dim btn-sm" onclick="apiPost('/api/mount/pec/save')">Save NV</button>
      <button class="btn btn-dim btn-sm" onclick="apiPost('/api/mount/pec/load')">Load NV</button>
      <button class="btn btn-dim btn-sm" onclick="if(confirm('Clear PEC buffer?')) apiPost('/api/mount/pec/clear')">Clear</button>
    </div>
  </div>

  <!-- Tracking Rate (OnStepX) -->
  <div class="card" id="tracking-rate-card">
    <h2>Tracking Rate</h2>
    <div class="flex-between">
      <span class="text-dim">Rate: <span id="track-rate">Sidereal</span></span>
      <span id="track-enabled-badge" class="badge badge-red">Off</span>
    </div>
    <div class="btn-row mt-1">
      <button class="btn btn-dim btn-sm" onclick="setTrackRate('sidereal')">Sidereal</button>
      <button class="btn btn-dim btn-sm" onclick="setTrackRate('lunar')">Lunar</button>
      <button class="btn btn-dim btn-sm" onclick="setTrackRate('solar')">Solar</button>
      <button class="btn btn-dim btn-sm" onclick="setTrackRate('king')">King</button>
    </div>
    <div class="btn-row mt-1">
      <button class="btn btn-green btn-sm" onclick="apiPost('/api/tracking/enable')">Enable</button>
      <button class="btn btn-red btn-sm" onclick="apiPost('/api/tracking/disable')">Disable</button>
    </div>
  </div>

  <!-- Backlash Config (OnStepX) -->
  <div class="card" id="backlash-card">
    <h2>Backlash</h2>
    <div class="text-dim" style="font-size:.82em">
      RA/Azm: <span id="bl-ra">--</span>" &nbsp;|&nbsp; Dec/Alt: <span id="bl-dec">--</span>"
    </div>
    <div class="input-row mt-1" style="margin:4px 0">
      <label style="font-size:.8em;min-width:28px">RA</label>
      <input type="number" id="bl-ra-input" placeholder="arcsec" min="0" style="max-width:70px;font-size:.8em">
      <button class="btn btn-dim btn-sm" onclick="setBacklash('ra')" style="font-size:.72em;padding:3px 6px">Set</button>
    </div>
    <div class="input-row" style="margin:4px 0">
      <label style="font-size:.8em;min-width:28px">Dec</label>
      <input type="number" id="bl-dec-input" placeholder="arcsec" min="0" style="max-width:70px;font-size:.8em">
      <button class="btn btn-dim btn-sm" onclick="setBacklash('dec')" style="font-size:.72em;padding:3px 6px">Set</button>
    </div>
  </div>

  <!-- Mount Limits (OnStepX) -->
  <div class="card" id="limits-card">
    <h2>Mount Limits</h2>
    <div class="text-dim" style="font-size:.82em">
      Horizon: <span id="lim-horizon">--</span>&deg; &nbsp;|&nbsp; Overhead: <span id="lim-overhead">--</span>&deg;
    </div>
    <div class="input-row mt-1" style="margin:4px 0">
      <label style="font-size:.8em;min-width:52px">Horizon</label>
      <input type="number" id="lim-horizon-input" placeholder="deg" min="-30" max="30" style="max-width:60px;font-size:.8em">
      <button class="btn btn-dim btn-sm" onclick="setLimit('horizon')" style="font-size:.72em;padding:3px 6px">Set</button>
    </div>
    <div class="input-row" style="margin:4px 0">
      <label style="font-size:.8em;min-width:52px">Overhead</label>
      <input type="number" id="lim-overhead-input" placeholder="deg" min="60" max="91" style="max-width:60px;font-size:.8em">
      <button class="btn btn-dim btn-sm" onclick="setLimit('overhead')" style="font-size:.72em;padding:3px 6px">Set</button>
    </div>

  </div>

  <!-- Auxiliary Features (OnStepX) -->
  <div class="card" id="auxiliary-card">
    <h2>Auxiliary Features</h2>
    <div id="aux-features-list" class="text-dim" style="font-size:.82em">
      No auxiliary features discovered.
    </div>
    <div class="btn-row mt-1">
      <button class="btn btn-dim btn-sm" onclick="apiPost('/api/mount/auxiliary/discover')">Discover</button>
      <button class="btn btn-dim btn-sm" onclick="apiPost('/api/mount/auxiliary/refresh')">Refresh</button>
    </div>
  </div>

  <!-- Star Alignment -->
  <div class="card">
    <h2>Star Alignment</h2>

    <!-- Mode selector -->
    <div class="text-dim" style="font-size:.8em;margin-bottom:4px">Alignment mode:</div>
    <div class="align-star-btns" id="align-mode-btns" style="margin-bottom:8px">
      <span class="align-star-btn active" data-mode="auto" onclick="setAlignMode('auto')"
            title="Fully automatic: plate-solves each star, re-centers, and syncs without user interaction">Automatic</span>
      <span class="align-star-btn" data-mode="manual" onclick="setAlignMode('manual')"
            title="User-guided: telescope slews to each star, you center it visually, then confirm sync">Manual</span>
    </div>
    <div class="text-dim" style="font-size:.72em;margin-bottom:6px" id="align-mode-desc">
      Automatic: plate-solver centers each star and syncs automatically.
    </div>

    <div class="text-dim" style="font-size:.8em">Select the number of alignment stars:</div>
    <div class="align-star-btns" id="align-star-btns">
      <span class="align-star-btn" data-n="3">3</span>
      <span class="align-star-btn active" data-n="6">6</span>
      <span class="align-star-btn" data-n="9">9</span>
      <span class="align-star-btn" data-n="12">12</span>
      <span class="align-star-btn" data-n="16">16</span>
      <span class="align-star-btn" data-n="20">20</span>
      <span class="align-star-btn" data-n="24">24</span>
    </div>
    <div class="btn-row">
      <button class="btn btn-accent" id="btn-align-start" onclick="doAlignStart()">Start Alignment</button>
      <button class="btn btn-red" id="btn-align-abort" onclick="doAlignAbort()" disabled>Abort</button>
    </div>

    <!-- Progress display -->
    <div class="align-progress mt-1" id="align-progress" style="display:none">
      <div class="flex-between">
        <span>Phase: <span class="step" id="align-phase">--</span></span>
        <span>Star <span id="align-star-num">0</span>/<span id="align-star-total">0</span></span>
      </div>
      <div class="flex-between mt-1">
        <span>Step: <span class="step" id="align-step">--</span></span>
        <span>Attempt: <span id="align-attempt">0</span>/<span id="align-max-attempt">5</span></span>
      </div>
      <div class="mt-1" id="align-error-row">Error: <span id="align-error">--</span> arcsec</div>
      <div class="text-dim mt-1" id="align-msg">--</div>
    </div>

    <!-- Manual mode: user action panel (shown only when waiting for user) -->
    <div id="align-manual-panel" style="display:none;margin-top:8px;padding:10px 12px;
         background:rgba(255,180,0,.08);border:1px solid rgba(255,180,0,.25);border-radius:10px">
      <div style="font-size:.85em;font-weight:700;color:var(--accent);margin-bottom:6px">
        Center this star in your eyepiece / camera:
      </div>
      <div style="font-size:.7em;color:var(--dim);margin-bottom:4px">
        Sync if already centered, or use Slew &amp; Center to fine-tune with directional controls
      </div>
      <div style="font-size:.82em;margin-bottom:4px">
        <span style="color:var(--text);font-weight:600" id="align-manual-star">--</span>
        <span class="text-dim" id="align-manual-coords"></span>
      </div>
      <div style="font-size:.78em;color:var(--dim);margin-bottom:8px" id="align-manual-altaz"></div>
      <div class="btn-row" style="gap:6px">
        <button class="btn btn-accent" id="btn-align-sync" onclick="doAlignManualSync()"
                style="flex:1;font-weight:700">Sync</button>
        <button class="btn btn-dim" id="btn-align-recenter" onclick="doAlignManualRecenter()"
                style="flex:1">Slew &amp; Center</button>
        <button class="btn btn-red btn-sm" id="btn-align-skip" onclick="doAlignManualSkip()"
                style="flex:0 0 auto;font-size:.78em">Skip</button>
      </div>
    </div>

    <div class="align-stars-list" id="align-stars-list" style="display:none"></div>
  </div>
</div>

<!-- Tab: Telemetry -->
<div class="tab-content" id="tab-telemetry">

  <!-- Tracking Error Trend -->
  <div class="card">
    <h2>Tracking Error</h2>
    <div class="telem-section">
      <div class="telem-stats-grid">
        <div class="telem-stat">
          <div class="telem-stat-label">RMS Alt</div>
          <div class="telem-stat-value accent" id="tm-rms-alt">--</div>
        </div>
        <div class="telem-stat">
          <div class="telem-stat-label">RMS Az</div>
          <div class="telem-stat-value blue" id="tm-rms-az">--</div>
        </div>
        <div class="telem-stat">
          <div class="telem-stat-label">Mean Drift Alt</div>
          <div class="telem-stat-value" id="tm-drift-alt">--</div>
        </div>
        <div class="telem-stat">
          <div class="telem-stat-label">Mean Drift Az</div>
          <div class="telem-stat-value" id="tm-drift-az">--</div>
        </div>
      </div>
      <div class="telem-chart-wrap tall">
        <canvas id="chart-error"></canvas>
      </div>
    </div>
  </div>

  <!-- Correction Source Breakdown -->
  <div class="card">
    <h2>Correction Sources</h2>
    <div class="telem-section">
      <div class="telem-axis-toggle">
        <span class="telem-axis-btn active" data-axis="alt" onclick="toggleCorrAxis(this)">Altitude</span>
        <span class="telem-axis-btn" data-axis="az" onclick="toggleCorrAxis(this)">Azimuth</span>
      </div>
      <div class="telem-chart-wrap tall">
        <canvas id="chart-corrections"></canvas>
      </div>
      <div class="telem-stats-grid">
        <div class="telem-stat">
          <div class="telem-stat-label">Total Solves</div>
          <div class="telem-stat-value" id="tm-total-solves">--</div>
        </div>
        <div class="telem-stat">
          <div class="telem-stat-label">Avg Solve Time</div>
          <div class="telem-stat-value" id="tm-avg-solve">--</div>
        </div>
        <div class="telem-stat">
          <div class="telem-stat-label">Corrections</div>
          <div class="telem-stat-value" id="tm-total-corr">--</div>
        </div>
        <div class="telem-stat">
          <div class="telem-stat-label">Avg Correction</div>
          <div class="telem-stat-value" id="tm-avg-corr">--</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Extended Kalman Filter (EKF) -->
  <div class="card">
    <h2>Extended Kalman Filter (EKF)</h2>
    <div class="telem-section">
      <!-- Status bar: visual indicators instead of raw numbers -->
      <div class="telem-stats-grid">
        <div class="telem-stat">
          <div class="telem-stat-label">EKF Status</div>
          <div class="telem-stat-value" id="tm-kf-init">--</div>
        </div>
        <div class="telem-stat">
          <div class="telem-stat-label">Drift Rate</div>
          <div class="telem-stat-value accent" id="tm-kf-drift">--</div>
        </div>
        <div class="telem-stat">
          <div class="telem-stat-label">Confidence</div>
          <div class="telem-stat-value" id="tm-kf-confidence">--</div>
        </div>
        <div class="telem-stat">
          <div class="telem-stat-label">Samples</div>
          <div class="telem-stat-value" id="tm-kf-samples">--</div>
        </div>
      </div>
      <!-- Sidereal model info: shows what the EKF is computing -->
      <div class="telem-stats-grid" style="margin-top:6px">
        <div class="telem-stat">
          <div class="telem-stat-label">Sidereal Alt Rate</div>
          <div class="telem-stat-value" id="tm-kf-sid-alt" style="font-size:.85em">--</div>
        </div>
        <div class="telem-stat">
          <div class="telem-stat-label">Sidereal Az Rate</div>
          <div class="telem-stat-value" id="tm-kf-sid-az" style="font-size:.85em">--</div>
        </div>
        <div class="telem-stat">
          <div class="telem-stat-label">Drift Alt</div>
          <div class="telem-stat-value accent" id="tm-kf-valt" style="font-size:.85em">--</div>
        </div>
        <div class="telem-stat">
          <div class="telem-stat-label">Drift Az</div>
          <div class="telem-stat-value blue" id="tm-kf-vaz" style="font-size:.85em">--</div>
        </div>
      </div>
      <!-- Confidence bar -->
      <div style="margin:10px 0 4px;display:flex;align-items:center;gap:8px">
        <span style="font-size:.78em;color:var(--muted);min-width:62px">Confidence</span>
        <div style="flex:1;height:8px;background:var(--bg3);border-radius:4px;overflow:hidden">
          <div id="tm-kf-conf-bar" style="height:100%;width:0%;border-radius:4px;background:linear-gradient(90deg,#ff4444,#ffd700,#4caf50);transition:width .6s ease"></div>
        </div>
        <span id="tm-kf-conf-pct" style="font-size:.78em;color:var(--muted);min-width:32px;text-align:right">--%</span>
      </div>

      <!-- Chart 1: Measured vs Filtered (the main "what is the EKF doing?" chart) -->
      <h3 style="margin-top:14px">Tracking: Measured vs EKF Estimate</h3>
      <p style="font-size:.75em;color:var(--muted);margin:-4px 0 6px">
        Dots = raw plate-solve positions (noisy). Lines = EKF smoothed estimate.
        The smoother the lines, the better the filter is working.</p>
      <div class="telem-chart-wrap">
        <canvas id="chart-kalman-residuals"></canvas>
      </div>

      <!-- Chart 2: Drift velocity over time -->
      <h3>Drift Estimation</h3>
      <p style="font-size:.75em;color:var(--muted);margin:-4px 0 6px">
        How fast the telescope is drifting away from the target (arcsec/s).
        Flat near zero = stable tracking. Changing = filter is correcting.</p>
      <div class="telem-chart-wrap">
        <canvas id="chart-kalman-innovation"></canvas>
      </div>
    </div>
  </div>

  <!-- ML Drift Predictor -->
  <div class="card">
    <h2>ML Drift Predictor</h2>
    <div class="telem-section">
      <div class="telem-stats-grid">
        <div class="telem-stat">
          <div class="telem-stat-label">Training Samples</div>
          <div class="telem-stat-value" id="tm-ml-samples">--</div>
        </div>
        <div class="telem-stat">
          <div class="telem-stat-label">Model Ready</div>
          <div class="telem-stat-value" id="tm-ml-ready">--</div>
        </div>
        <div class="telem-stat">
          <div class="telem-stat-label">Mean Error</div>
          <div class="telem-stat-value" id="tm-ml-error">--</div>
        </div>
        <div class="telem-stat">
          <div class="telem-stat-label">Total Predictions</div>
          <div class="telem-stat-value" id="tm-ml-preds">--</div>
        </div>
        <div class="telem-stat">
          <div class="telem-stat-label">Current Pred Alt</div>
          <div class="telem-stat-value accent" id="tm-ml-curalt">--</div>
        </div>
        <div class="telem-stat">
          <div class="telem-stat-label">Current Pred Az</div>
          <div class="telem-stat-value blue" id="tm-ml-curaz">--</div>
        </div>
      </div>
      <h3>Model Weights</h3>
      <div class="telem-chart-wrap">
        <canvas id="chart-ml-weights"></canvas>
      </div>
      <div class="telem-weights-grid" id="tm-ml-weights-detail"></div>
    </div>
  </div>

  <!-- PEC Correction -->
  <div class="card">
    <h2>Periodic Error Correction</h2>
    <div class="telem-section">
      <div class="telem-stats-grid">
        <div class="telem-stat">
          <div class="telem-stat-label">Trained</div>
          <div class="telem-stat-value" id="tm-pec-trained">--</div>
        </div>
        <div class="telem-stat">
          <div class="telem-stat-label">Learning</div>
          <div class="telem-stat-value" id="tm-pec-learning">--</div>
        </div>
        <div class="telem-stat">
          <div class="telem-stat-label">Samples</div>
          <div class="telem-stat-value" id="tm-pec-samples">--</div>
        </div>
        <div class="telem-stat">
          <div class="telem-stat-label">Data Span</div>
          <div class="telem-stat-value" id="tm-pec-span">--</div>
        </div>
        <div class="telem-stat">
          <div class="telem-stat-label">Correction RMS Alt</div>
          <div class="telem-stat-value accent" id="tm-pec-rmsalt">--</div>
        </div>
        <div class="telem-stat">
          <div class="telem-stat-label">Correction RMS Az</div>
          <div class="telem-stat-value blue" id="tm-pec-rmsaz">--</div>
        </div>
      </div>
      <h3>Detected Periods</h3>
      <div id="tm-pec-periods-container">
        <div class="telem-no-data" id="tm-pec-no-periods">No periods detected yet</div>
        <table class="telem-period-table" id="tm-pec-periods-table" style="display:none">
          <thead><tr><th>Axis</th><th>Period</th><th>Amplitude</th><th>SNR</th><th>Harmonics</th></tr></thead>
          <tbody id="tm-pec-periods-body"></tbody>
        </table>
      </div>
      <h3>Correction Curve</h3>
      <div class="telem-chart-wrap tall">
        <canvas id="chart-pec-curve"></canvas>
      </div>
    </div>
  </div>

</div>
<!-- Tab: Weather / Location -->
<div class="tab-content" id="tab-weather">

  <!-- Observer Location -->
  <div class="card">
    <h2>Observer Location</h2>
    <div class="input-row">
      <label>Lat</label>
      <input type="number" id="loc-lat" step="0.0001" placeholder="48.8566">
    </div>
    <div class="input-row">
      <label>Lon</label>
      <input type="number" id="loc-lon" step="0.0001" placeholder="2.3522">
    </div>
    <div class="input-row">
      <label>Time</label>
      <span id="loc-time" style="font-family:monospace;font-size:.95em;color:var(--fg)">--:--:--</span>
      <span id="loc-utc" class="text-dim" style="font-size:.75em;margin-left:6px">UTC--</span>
    </div>
    <div class="btn-row mt-1">
      <button class="btn btn-accent btn-sm" onclick="getGPS()">Use GPS</button>
    </div>
    <div class="text-dim text-center mt-1" id="loc-status"></div>
  </div>

  <!-- Initialize -->
  <div class="card">
    <h2>Initialize</h2>
    <div class="text-dim" style="font-size:.78em;margin-bottom:.6em">
      Individual initialization commands for the mount controller.
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">
      <button class="btn btn-accent btn-sm" onclick="doSetTimeDate()" id="btn-init-time" style="padding:10px 6px;font-size:.82em">
        <span style="font-size:1.2em;display:block">&#128339;</span>Set Time &amp; Date
      </button>
      <button class="btn btn-accent btn-sm" onclick="doSetSite()" id="btn-init-site" style="padding:10px 6px;font-size:.82em">
        <span style="font-size:1.2em;display:block">&#127760;</span>Set Site Location
      </button>
      <button class="btn btn-dim btn-sm" onclick="doResetHome()" id="btn-init-reset-home" style="padding:10px 6px;font-size:.82em">
        <span style="font-size:1.2em;display:block">&#127968;</span>Reset Home
      </button>
      <button class="btn btn-dim btn-sm" onclick="doReturnHome()" id="btn-init-return-home" style="padding:10px 6px;font-size:.82em">
        <span style="font-size:1.2em;display:block">&#127793;</span>Return Home
      </button>
      <button class="btn btn-dim btn-sm" onclick="doUnpark()" id="btn-init-unpark" style="padding:10px 6px;font-size:.82em">
        <span style="font-size:1.2em;display:block">&#128275;</span>Unpark
      </button>
      <button class="btn btn-dim btn-sm" onclick="doPark()" id="btn-init-park" style="padding:10px 6px;font-size:.82em">
        <span style="font-size:1.2em;display:block">&#128274;</span>Park
      </button>
    </div>
    <div class="text-dim text-center mt-1" id="init-status" style="font-size:.78em"></div>
  </div>

  <!-- Auxiliary Features (OnStepX) - Smart UI -->
  <div class="card" id="auxiliary-card-loc">
    <h2>Auxiliary Features</h2>
    <div id="aux-features-loc" class="text-dim" style="font-size:.82em">
      Tap Discover to detect available features from the controller.
    </div>
    <div class="btn-row mt-1">
      <button class="btn btn-accent btn-sm" onclick="auxDiscover()">Discover</button>
      <button class="btn btn-dim btn-sm" onclick="auxRefresh()">Refresh</button>
    </div>
    <div class="text-dim text-center mt-1" id="aux-status" style="font-size:.78em"></div>
  </div>
  <div class="card">
    <h2>Plate Solver</h2>
    <div class="text-dim" style="font-size:.78em;margin-bottom:.5em">
      Plate solving identifies your telescope's exact pointing position from
      a sky photo. Choose a solver mode below.
    </div>
    <div class="input-row">
      <label>Mode</label>
      <select id="solver-mode">
        <option value="auto">Auto (ASTAP first, cloud fallback)</option>
        <option value="astap">ASTAP Local (offline, no internet)</option>
        <option value="cloud">Astrometry.net (cloud only)</option>
      </select>
    </div>
    <!-- Solver timeout configuration -->
    <div class="input-row" style="margin-top:.4em">
      <label>Timeout</label>
      <div style="display:flex;align-items:center;gap:6px;flex:1">
        <input type="range" id="solver-timeout-slider" min="10" max="300" value="120"
               style="flex:1" oninput="document.getElementById('solver-timeout-val').value=this.value">
        <input type="number" id="solver-timeout-val" min="10" max="600" value="120"
               style="width:52px;font-size:.82em;text-align:center"
               oninput="document.getElementById('solver-timeout-slider').value=Math.min(300,Math.max(10,this.value))">
        <span class="text-dim" style="font-size:.72em">sec</span>
      </div>
    </div>
    <div class="text-dim" style="font-size:.68em;margin-top:.15em;margin-left:70px">
      Max time per solve attempt. Typical: 30-120s. Increase for blind solves.
    </div>
    <!-- ASTAP Star Database section -->
    <div id="astap-db-section" style="margin-top:.6em;padding:.6em;border-radius:6px;background:rgba(0,180,255,.04);border:1px solid rgba(0,180,255,.1)">
      <div style="font-size:.82em;font-weight:600;margin-bottom:.4em;color:var(--accent)">ASTAP Star Database</div>
      <div id="astap-db-status" class="text-dim" style="font-size:.76em;margin-bottom:.4em">Checking...</div>
      <div class="input-row" style="margin-bottom:.3em">
        <label style="font-size:.78em">Database</label>
        <select id="astap-db-select" style="font-size:.82em">
          <option value="d05">D05 (45 MB, FOV &ge; 0.6&deg;)</option>
          <option value="d20">D20 (170 MB, FOV &ge; 0.3&deg;)</option>
          <option value="d50">D50 (500 MB, FOV &ge; 0.2&deg;)</option>
          <option value="w08">W08 (15 MB, FOV &ge; 20&deg; wide-field)</option>
        </select>
      </div>
      <div class="btn-row">
        <button class="btn btn-accent btn-sm" id="astap-db-download-btn" onclick="downloadAstapDb()">Download Database</button>
        <button class="btn btn-dim btn-sm" onclick="refreshAstapDbStatus()">Refresh</button>
        <button class="btn btn-dim btn-sm" id="astap-db-delete-btn" onclick="deleteAstapDb()" style="display:none">Delete</button>
      </div>
      <div id="astap-db-progress" style="display:none;margin-top:.4em">
        <div style="background:rgba(255,255,255,.05);border-radius:4px;overflow:hidden;height:6px">
          <div id="astap-db-progress-bar" style="width:0%;height:100%;background:var(--accent);transition:width .3s"></div>
        </div>
        <div id="astap-db-progress-text" class="text-dim" style="font-size:.72em;margin-top:.2em;text-align:center">Downloading...</div>
      </div>
    </div>
    <!-- Optics / FOV configuration -->
    <div id="solver-optics-section" style="margin-top:.6em;padding:.6em;border-radius:6px;background:rgba(0,180,255,.04);border:1px solid rgba(0,180,255,.1)">
      <div style="font-size:.82em;font-weight:600;margin-bottom:.4em;color:var(--accent)">Optics / FOV</div>
      <div class="text-dim" style="font-size:.68em;margin-bottom:.5em">
        Setting your focal length and sensor width lets ASTAP solve faster and
        more reliably by knowing the image scale. Leave at 0 if unknown.
      </div>
      <div class="input-row" style="margin-bottom:.3em">
        <label style="font-size:.78em">Focal Length</label>
        <div style="display:flex;align-items:center;gap:4px;flex:1">
          <input type="number" id="solver-focal-length" min="0" max="50000" value="0" step="1"
                 style="width:80px;font-size:.82em;text-align:center" oninput="updateFovPreview()">
          <span class="text-dim" style="font-size:.72em">mm</span>
        </div>
      </div>
      <div class="input-row" style="margin-bottom:.3em">
        <label style="font-size:.78em">Sensor Width</label>
        <div style="display:flex;align-items:center;gap:4px;flex:1">
          <input type="number" id="solver-sensor-width" min="0" max="100" value="0" step="0.1"
                 style="width:80px;font-size:.82em;text-align:center" oninput="updateFovPreview()">
          <span class="text-dim" style="font-size:.72em">mm</span>
        </div>
      </div>
      <div class="text-dim" style="font-size:.65em;margin-top:.1em;margin-left:0;padding:4px 6px;background:rgba(74,158,255,.06);border-radius:4px;line-height:1.4">
        Physical chip width in mm (not pixels). For ZWO cameras this is auto-detected
        when connected. For other cameras, use the calculator below.
      </div>
      <!-- Pixel Size Calculator for non-ZWO cameras -->
      <details id="sensor-calc-details" style="margin-top:.4em;padding:6px 8px;border-radius:6px;background:rgba(255,140,0,.04);border:1px solid rgba(255,140,0,.12)">
        <summary style="font-size:.74em;font-weight:600;color:var(--accent);cursor:pointer;user-select:none">
          Sensor Width Calculator
        </summary>
        <div style="margin-top:.4em;font-size:.68em" class="text-dim">
          Enter your camera's resolution (width in pixels) and pixel size (in &micro;m) from the spec sheet.
        </div>
        <div style="display:flex;gap:6px;align-items:center;margin-top:.35em;flex-wrap:wrap">
          <div style="display:flex;align-items:center;gap:3px">
            <label style="font-size:.72em;white-space:nowrap">Width (px)</label>
            <input type="number" id="calc-res-x" min="1" max="100000" placeholder="e.g. 4656"
                   style="width:72px;font-size:.78em;text-align:center" oninput="calcSensorWidth()">
          </div>
          <div style="display:flex;align-items:center;gap:3px">
            <label style="font-size:.72em;white-space:nowrap">Pixel (&micro;m)</label>
            <input type="number" id="calc-pixel-um" min="0.1" max="100" step="0.01" placeholder="e.g. 3.76"
                   style="width:66px;font-size:.78em;text-align:center" oninput="calcSensorWidth()">
          </div>
          <button class="btn btn-accent btn-sm" style="font-size:.7em;padding:3px 8px" onclick="applySensorCalc()">Apply</button>
        </div>
        <div id="calc-result" style="margin-top:.3em;font-size:.72em;min-height:1.2em" class="text-dim"></div>
      </details>
      <div id="solver-fov-preview" style="margin-top:.4em;padding:5px 8px;border-radius:5px;background:rgba(0,0,0,.2);font-size:.76em">
        <span class="text-dim">Calculated FOV: </span><span id="solver-fov-value" style="font-weight:600;color:var(--fg)">--</span>
        <span id="solver-fov-db-hint" class="text-dim" style="margin-left:6px"></span>
      </div>
      <div id="solver-fov-solved" style="margin-top:.3em;font-size:.72em" class="text-dim">
        <span>Last solved FOV: </span><span id="solver-fov-solved-val" style="font-weight:600">--</span>
      </div>
    </div>
    <!-- Cloud solver API key -->
    <div id="cloud-api-key-section" style="margin-top:.6em">
      <div class="input-row">
        <label>API Key</label>
        <input type="text" id="solver-api-key" placeholder="(anonymous if empty)"
               style="font-family:monospace;font-size:.82em">
      </div>
      <div class="text-dim" style="font-size:.72em;margin-top:.3em">
        Cloud key from <a href="https://nova.astrometry.net/api_help" target="_blank"
        style="color:var(--accent)">nova.astrometry.net/api_help</a> (optional, free).
      </div>
    </div>
    <div class="btn-row mt-1">
      <button class="btn btn-accent btn-sm" onclick="saveSolverSettings()">Save</button>
    </div>
    <div class="text-dim text-center mt-1" id="solver-status"></div>
  </div>
  <div class="card">
    <h2>Weather Conditions</h2>
    <div class="flex-between mb-1">
      <span class="text-dim" id="w-location">--</span>
      <button class="btn btn-dim btn-sm" onclick="apiPost('/api/weather/refresh')">Refresh</button>
    </div>
    <div id="weather-no-internet" class="text-dim text-center" style="display:none;padding:8px;margin:4px 0;border-radius:6px;background:rgba(255,215,0,.08);border:1px solid rgba(255,215,0,.12);font-size:.78em;color:var(--yellow)">
      No internet access (telescope WiFi). Weather data unavailable.
    </div>
    <div class="weather-grid mt-1">
      <div class="wi"><span class="wi-label">Temperature</span><span class="wi-val" id="w-temp">--</span></div>
      <div class="wi"><span class="wi-label">Humidity</span><span class="wi-val" id="w-hum">--</span></div>
      <div class="wi"><span class="wi-label">Pressure</span><span class="wi-val" id="w-press">--</span></div>
      <div class="wi"><span class="wi-label">Cloud Cover</span><span class="wi-val" id="w-cloud">--</span></div>
      <div class="wi"><span class="wi-label">Wind</span><span class="wi-val" id="w-wind">--</span></div>
      <div class="wi"><span class="wi-label">Gusts</span><span class="wi-val" id="w-gust">--</span></div>
      <div class="wi"><span class="wi-label">Dew Point</span><span class="wi-val" id="w-dew">--</span></div>
      <div class="wi"><span class="wi-label">Dew Risk</span><span class="wi-val" id="w-dewrisk">--</span></div>
      <div class="wi"><span class="wi-label">Conditions</span><span class="wi-val" id="w-cond">--</span></div>
      <div class="wi"><span class="wi-label">Observing</span><span class="wi-val" id="w-obs">--</span></div>
    </div>
    <div class="text-dim text-center mt-1" id="w-status">--</div>
  </div>
</div>
<!-- Tab: Log (Enhanced) -->
<div class="tab-content" id="tab-log">
  <div class="card">
    <div class="flex-between" style="margin-bottom:8px">
      <h2 style="margin:0">Application Log</h2>
      <div style="display:flex;gap:4px;align-items:center">
        <select class="log-level-select" id="log-level-sel" onchange="logSetLevel(this.value)" title="Server log verbosity">
          <option value="DEBUG">DEBUG</option>
          <option value="INFO" selected>INFO</option>
          <option value="WARNING">WARNING</option>
          <option value="ERROR">ERROR</option>
        </select>
        <button class="btn btn-sm" onclick="logClear()" title="Clear log display">Clear</button>
        <button class="btn btn-sm" onclick="logCopy()" title="Copy log to clipboard">Copy</button>
        <button class="btn btn-sm" onclick="logExport()" title="Download log as text file">Export</button>
      </div>
    </div>
    <!-- Filter buttons -->
    <div class="log-toolbar" id="log-toolbar">
      <button class="btn-sm active" data-filter="all" onclick="logToggleFilter(this,'all')">All</button>
      <button class="btn-sm flt-error active" data-filter="error" onclick="logToggleFilter(this,'error')">Errors</button>
      <button class="btn-sm flt-warning active" data-filter="warning" onclick="logToggleFilter(this,'warning')">Warnings</button>
      <button class="btn-sm flt-success active" data-filter="success" onclick="logToggleFilter(this,'success')">Success</button>
      <button class="btn-sm flt-cmd active" data-filter="cmd" onclick="logToggleFilter(this,'cmd')">Commands</button>
      <button class="btn-sm flt-info active" data-filter="info" onclick="logToggleFilter(this,'info')">Info</button>
      <input type="text" class="log-search" id="log-search" placeholder="Search logs..." oninput="logApplySearch()">
    </div>
    <!-- Stats bar -->
    <div class="log-stats" id="log-stats">
      <span class="stat">Total: <strong id="ls-total">0</strong></span>
      <span class="stat"><span class="stat-dot" style="background:#ff6b6b"></span> Errors: <strong id="ls-errors">0</strong></span>
      <span class="stat"><span class="stat-dot" style="background:#ffdd57"></span> Warnings: <strong id="ls-warnings">0</strong></span>
      <span class="stat"><span class="stat-dot" style="background:#69db7c"></span> Success: <strong id="ls-success">0</strong></span>
      <span class="stat"><span class="stat-dot" style="background:#74b9ff"></span> Commands: <strong id="ls-cmds">0</strong></span>
    </div>
    <!-- Log output -->
    <div style="position:relative">
      <div class="log-box" id="log-box"></div>
      <button class="log-scroll-btn" id="log-scroll-btn" onclick="logScrollToBottom()">&#8595; Jump to bottom</button>
    </div>
  </div>
</div>

<!-- Tab: Help (User Manual) -->
<div class="tab-content" id="tab-help">
  <div class="card" style="padding:0;overflow:hidden">
    <div style="display:flex;justify-content:space-between;align-items:center;padding:14px 16px;border-bottom:1px solid var(--glass-border)">
      <h2 style="margin:0">&#128214; User Manual</h2>
      <div style="display:flex;gap:8px">
        <button class="btn btn-sm" onclick="document.getElementById('help-frame').contentWindow.scrollTo({top:0,behavior:'smooth'})" title="Scroll to top">&#8679; Top</button>
        <button class="btn btn-sm" onclick="window.open('/help','_blank')" title="Open in new window">&#8599; Open</button>
      </div>
    </div>
    <iframe id="help-frame" src="/help" style="width:100%;height:calc(100vh - 280px);min-height:500px;border:none;background:#0d0d14"></iframe>
  </div>
</div>
</div><!-- /container -->

<!-- ENHANCEMENT 1: Bottom Navigation Bar -->
<nav class="bottom-nav" id="bottom-nav">
  <div class="bnav-item active" data-nav="control" onclick="switchNav('control')">
    <span class="bnav-icon">&#9783;</span>
    <span class="bnav-label">Control</span>
  </div>
  <div class="bnav-item" data-nav="imaging" onclick="switchNav('imaging')">
    <span class="bnav-icon">&#128247;</span>
    <span class="bnav-label">Camera</span>
  </div>
  <div class="bnav-item" data-nav="tracking" onclick="switchNav('tracking')">
    <span class="bnav-icon">&#127919;</span>
    <span class="bnav-label">Tracking</span>
  </div>
  <div class="bnav-item" data-nav="settings" onclick="switchNav('settings')">
    <span class="bnav-icon">&#9881;</span>
    <span class="bnav-label">Settings</span>
  </div>
</nav>

<!-- ENHANCEMENT 10: Emergency Stop Button (always visible) -->
<button class="emergency-stop" id="emergency-stop" onclick="emergencyStop()" title="Emergency Stop - halt all motion">STOP</button>

<!-- ENHANCEMENT 2: Night mode brightness overlay -->
<div class="night-dimmer" id="night-dimmer"></div>

<!-- Floating toolbar (collapsible) - ENHANCED with new toggles -->
<div class="toolbar">
  <div class="toolbar-items" id="toolbar-items">
    <button class="btn btn-dim" onclick="toggleNightMode()" id="btn-nightmode" title="Night vision (red) mode">&#9790;</button>
    <button class="btn btn-dim" onclick="toggleStarfield()" id="btn-starfield" title="Toggle starfield background">&#10026;</button>
    <button class="btn btn-dim" onclick="toggleLightTheme()" id="btn-lighttheme" title="Toggle light/dark theme">&#9788;</button>
    <button class="btn btn-dim" onclick="toggleLargeText()" id="btn-largetext" title="Toggle large text">Aa</button>
    <button class="btn btn-dim" onclick="toggleHighContrast()" id="btn-highcontrast" title="Toggle high contrast">&#9673;</button>
    <button class="btn btn-dim" onclick="toggleWakeLock()" id="btn-wakelock" title="Keep screen on">&#9728;</button>
  </div>
  <button class="btn btn-dim toolbar-toggle" id="toolbar-toggle" onclick="toggleToolbar()" title="Settings">&#9881;</button>
</div>

<!-- Toast notifications -->
<div class="toast-panel" id="toast-panel"></div>

<script>
// ============================================================
// State
// ============================================================
let lastLogSeq = 0;
let searchTimeout = null;

// ============================================================
// Tabs
// ============================================================
document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('tab-' + t.dataset.tab).classList.add('active');
  });
});

// ============================================================
// Toast notification system
// ============================================================
function toast(message, type) {
  // type: 'ok', 'err', 'info'
  type = type || 'info';
  const panel = document.getElementById('toast-panel');
  const el = document.createElement('div');
  el.className = 'toast toast-' + type;
  const icons = { ok: '\u2713', err: '\u2717', info: '\u2139' };
  el.innerHTML = '<span class="toast-icon">' + (icons[type] || '\u2139') + '</span>'
    + '<span class="toast-msg">' + message + '</span>';
  panel.appendChild(el);
  // Auto-remove after 3s
  setTimeout(() => {
    el.classList.add('removing');
    setTimeout(() => el.remove(), 300);
  }, 3000);
  // Limit to 6 visible
  while (panel.children.length > 6) panel.children[0].remove();
}

// Friendly labels for API endpoints
const _apiLabels = {
  '/api/connect': 'Connect',
  '/api/disconnect': 'Disconnect',
  '/api/simulator': 'Simulator',
  '/api/tracking/start': 'Start tracking',
  '/api/tracking/stop': 'Stop tracking',
  '/api/session/save': 'Save session',
  '/api/solve/start': 'Plate solve started',
  '/api/solve/stop': 'Plate solve stopped',
  '/api/park': 'Park telescope',
  '/api/home': 'Home telescope',
  '/api/goto': 'GoTo target',
  '/api/slew': 'Slew',
  '/api/slew/stop': 'Slew stop',
  '/api/focuser/move': 'Focuser move',
  '/api/focuser/stop': 'Focuser stop',
  '/api/derotator/rotate': 'Derotator rotate',
  '/api/derotator/stop': 'Derotator stop',
  '/api/derotator/sync': 'Derotator sync',
  '/api/pec/toggle': 'PEC toggle',
  '/api/pec/save': 'PEC save',
  '/api/pec/load': 'PEC load',
  '/api/pec/reset': 'PEC reset',
  '/api/mount/drive-type': 'Drive type',
  '/api/flexure/toggle': 'Flexure toggle',
  '/api/flexure/reset': 'Flexure reset',
  '/api/mount/set-time': 'Set time/date',
  '/api/mount/set-site': 'Set site location',
  '/api/mount/set-weather': 'Set weather',
  '/api/home/find': 'Find home',
  '/api/unpark': 'Unpark',
  '/api/mount/auxiliary/discover': 'Discover features',
  '/api/mount/auxiliary/refresh': 'Refresh features',
  '/api/mount/auxiliary/set': 'Set auxiliary',
  '/api/camera/start': 'Camera start',
  '/api/camera/stop': 'Camera stop',
  '/api/camera/ascom/settings': 'Camera settings',
  '/api/camera/asi/settings': 'ASI camera settings',
  '/api/connection/settings': 'Connection settings',
  '/api/autoconnect': 'Auto-connect settings',
  '/api/alignment/start': 'Alignment start',
  '/api/alignment/abort': 'Alignment abort',
  '/api/alignment/manual/sync': 'Manual sync',
  '/api/alignment/manual/recenter': 'Re-center star',
  '/api/alignment/manual/skip': 'Skip star',
  '/api/weather/refresh': 'Weather refresh',
};

// Endpoints that fire rapidly (no toast for these)
const _silentEndpoints = new Set(['/api/slew', '/api/slew/stop', '/api/focuser/move', '/api/focuser/stop',
  '/api/derotator/rotate', '/api/derotator/stop', '/api/camera/settings',
  '/api/mount/auxiliary/set', '/api/mount/auxiliary/refresh']);

// ============================================================
// API helpers (with automatic toast feedback + auth token)
// ============================================================
// Read token from URL ?token=... param (set once at page load)
const _apiToken = new URLSearchParams(window.location.search).get('token') || '';

function _authHeaders(extra) {
  const h = Object.assign({}, extra || {});
  if (_apiToken) h['Authorization'] = 'Bearer ' + _apiToken;
  return h;
}

async function apiGet(url) {
  try {
    const r = await fetch(url, {headers: _authHeaders()});
    return await r.json();
  } catch(e) { return null; }
}

async function apiPost(url, body) {
  const label = _apiLabels[url] || url.replace('/api/', '');
  try {
    const r = await fetch(url, {
      method: 'POST',
      headers: _authHeaders({'Content-Type': 'application/json'}),
      body: body ? JSON.stringify(body) : '{}'
    });
    const data = await r.json();
    if (!_silentEndpoints.has(url)) {
      if (data && data.ok === false) {
        toast(label + ': ' + (data.error || 'Failed'), 'err');
      } else {
        toast(data && data.message ? data.message : label + ': OK', 'ok');
      }
    }
    return data;
  } catch(e) {
    if (!_silentEndpoints.has(url)) {
      toast(label + ': Connection error', 'err');
    }
    return null;
  }
}

// Mount drive type setter
async function setDriveType(dt) {
  await apiPost('/api/mount/drive-type', {drive_type: dt});
}

// ============================================================
// Status polling (1 Hz)
// ============================================================
async function pollStatus() {
  const s = await apiGet('/api/status');
  if (!s) return;

  // Hide ASCOM UI elements on Android (first poll detects platform)
  if (!window._platformChecked && s.platform) {
    window._platformChecked = true;
    if (s.platform === 'android') {
      window._isAndroid = true;
      // Replace plate-solve source dropdown with Android camera options
      const solveSel = document.getElementById('solve-mode');
      if (solveSel) {
        solveSel.innerHTML = '';
        const solveOpts = [
          {v: 'auto',  t: 'Auto (best available)'},
          {v: 'zwo',   t: 'ZWO ASI Camera'},
          {v: 'uvc',   t: 'USB Camera'},
          {v: 'phone', t: 'Phone Camera'},
        ];
        for (const o of solveOpts) {
          const el = document.createElement('option');
          el.value = o.v; el.textContent = o.t;
          solveSel.appendChild(el);
        }
      }
      // Remove ASCOM fields entirely (not applicable on Android)
      const ascomFields = document.getElementById('solve-ascom-fields');
      if (ascomFields) ascomFields.remove();
      // Hide desktop "Cam #" field (Android uses source selector instead)
      const solveCamFields = document.getElementById('solve-camera-fields');
      if (solveCamFields) solveCamFields.style.display = 'none';
      // Remove ASCOM camera settings panel from live view
      const camAscom = document.getElementById('cam-ascom-fields');
      if (camAscom) camAscom.remove();
      // Hide desktop UVC "Cam #" field in live view (irrelevant on Android)
      const camUvcFields = document.getElementById('cam-uvc-fields');
      if (camUvcFields) camUvcFields.style.display = 'none';
      // Replace live camera source dropdown with Android options
      const lvSel = document.getElementById('lv-cam-source');
      if (lvSel) {
        lvSel.innerHTML = '';
        const opts = [
          {v: 'auto',  t: 'Auto (best)'},
          {v: 'zwo',   t: 'ZWO ASI'},
          {v: 'uvc',   t: 'USB Camera'},
          {v: 'phone', t: 'Phone Camera'},
        ];
        for (const o of opts) {
          const el = document.createElement('option');
          el.value = o.v; el.textContent = o.t;
          lvSel.appendChild(el);
        }
        lvSel.style.maxWidth = '130px';
      }
    }
  }

  const p = s.position || {};
  const c = s.connection || {};
  const t = s.tracking || {};
  const pec = s.pec || {};
  const w = s.weather || {};
  const ctrl = s.controls || {};

  // Header
  const connected = c.connected || false;
  const simActive = c.simulator_active || false;
  const dot = document.getElementById('hdr-dot');
  const hdrStatus = document.getElementById('hdr-status');
  if (connected || simActive) {
    const slewingNow = p.is_slewing || false;
    const trackingNow = t.is_running || false;
    dot.className = 'status-dot ' + (slewingNow ? 'dot-yellow' : (trackingNow ? 'dot-green' : 'dot-yellow'));
    let statusText = simActive ? 'Simulator' : (c.status || 'Connected');
    if (slewingNow) statusText += ' \u2022 Slewing';
    else if (trackingNow) statusText += ' \u2022 Tracking';
    hdrStatus.textContent = statusText;
  } else {
    dot.className = 'status-dot dot-red';
    hdrStatus.textContent = 'Disconnected';
  }

  // Position
  setText('p-ra', p.ra_display || '--');
  setText('p-dec', p.dec_display || '--');
  setText('p-alt', p.alt_display || '--');
  setText('p-az', p.az_display || '--');
  setText('p-rate-alt', p.rate_alt || '--');
  setText('p-rate-az', p.rate_az || '--');

  // Target display
  const targetRow = document.getElementById('target-row');
  const targetName = p.goto_target || '';
  if (targetRow) {
    if (targetName) {
      targetRow.style.display = '';
      setText('p-target-name', targetName);
      const tEq = (p.goto_ra && p.goto_dec) ? `RA ${p.goto_ra}  Dec ${p.goto_dec}` : '';
      const tHz = (p.goto_alt && p.goto_az) ? `  \u2502  Alt ${p.goto_alt}  Az ${p.goto_az}` : '';
      setText('p-target-coords', tEq + tHz);
      const tBadge = document.getElementById('p-target-badge');
      if (tBadge) {
        if (p.is_slewing) {
          tBadge.textContent = 'Slewing'; tBadge.className = 'badge badge-yellow';
          tBadge.style.cssText = 'font-size:.65em;padding:2px 8px';
        } else {
          tBadge.textContent = 'On Target'; tBadge.className = 'badge badge-green';
          tBadge.style.cssText = 'font-size:.65em;padding:2px 8px';
        }
      }
    } else {
      targetRow.style.display = 'none';
    }
  }

  // Connection
  setText('conn-info', c.type === 'WiFi'
    ? `WiFi ${c.wifi_ip||''}:${c.wifi_port||''}`
    : `USB ${c.port||''} @ ${c.baudrate||''}`);
  const cb = document.getElementById('conn-badge');
  if (connected || simActive) {
    cb.textContent = simActive ? 'Simulator' : 'Connected';
    cb.className = 'badge badge-green';
  } else {
    cb.textContent = 'Disconnected';
    cb.className = 'badge badge-red';
  }
  document.getElementById('btn-connect').disabled = connected || simActive;
  document.getElementById('btn-disconnect').disabled = !connected && !simActive;

  // Sync connection fields ONLY when connected -- while disconnected the user
  // is configuring and polling must never overwrite their input.
  if (connected) {
    if (c.type && document.activeElement !== document.getElementById('conn-type')) {
      document.getElementById('conn-type').value = c.type;
      updateConnTypeUI();
    }
    if (c.port && document.activeElement !== document.getElementById('conn-port')) {
      const sel = document.getElementById('conn-port');
      if (![...sel.options].some(o => o.value === c.port)) {
        const opt = document.createElement('option');
        opt.value = c.port; opt.textContent = c.port;
        sel.prepend(opt);
      }
      sel.value = c.port;
    }
    if (c.baudrate && document.activeElement !== document.getElementById('conn-baud'))
      document.getElementById('conn-baud').value = c.baudrate;
    if (c.wifi_ip && document.activeElement !== document.getElementById('conn-wifi-ip'))
      document.getElementById('conn-wifi-ip').value = c.wifi_ip;
    if (c.wifi_port && document.activeElement !== document.getElementById('conn-wifi-port'))
      document.getElementById('conn-wifi-port').value = c.wifi_port;
    if (c.protocol && document.activeElement !== document.getElementById('conn-protocol'))
      document.getElementById('conn-protocol').value = c.protocol;
  }

  // Location (skip if GPS coords are pending user confirmation)
  if (!window._gpsPending) {
    const loc = s.location || {};
    if (loc.latitude && document.activeElement !== document.getElementById('loc-lat'))
      document.getElementById('loc-lat').value = loc.latitude;
    if (loc.longitude && document.activeElement !== document.getElementById('loc-lon'))
      document.getElementById('loc-lon').value = loc.longitude;
  }

  // Camera / Solve
  const cam = s.camera || {};
  if (cam.solve_mode && document.activeElement !== document.getElementById('solve-mode')) {
    document.getElementById('solve-mode').value = cam.solve_mode;
    onSolveModeChange();
  }
  if (cam.camera_index && document.activeElement !== document.getElementById('cam-index'))
    document.getElementById('cam-index').value = cam.camera_index;
  // ASCOM fields (only present on desktop, removed on Android)
  if (document.getElementById('ascom-cam-name')) {
    if (cam.ascom_camera_name)
      setText('ascom-cam-name', cam.ascom_camera_name);
    if (cam.ascom_exposure && document.activeElement !== document.getElementById('ascom-exp'))
      document.getElementById('ascom-exp').value = cam.ascom_exposure;
    if (cam.ascom_gain && document.activeElement !== document.getElementById('ascom-gain'))
      document.getElementById('ascom-gain').value = cam.ascom_gain;
    if (cam.ascom_binning && document.activeElement !== document.getElementById('ascom-bin'))
      document.getElementById('ascom-bin').value = cam.ascom_binning;
  }
  if (cam.solve_interval && document.activeElement !== document.getElementById('solve-interval'))
    document.getElementById('solve-interval').value = cam.solve_interval;

  // Tracking
  const isTracking = t.is_running || false;
  const isSlewing = p.is_slewing || false;
  document.getElementById('btn-track-start').disabled = isTracking;
  document.getElementById('btn-track-stop').disabled = !isTracking;
  // Active glow indicators on tracking buttons
  const btnTrackStart = document.getElementById('btn-track-start');
  if (isTracking) { btnTrackStart.classList.add('btn-active-green'); }
  else { btnTrackStart.classList.remove('btn-active-green'); }
  const btnSolveStart = document.getElementById('btn-solve-start');
  const isSolvingNow = t.auto_solve_running || false;
  if (isSolvingNow) { btnSolveStart.classList.add('btn-active-blue'); }
  else { btnSolveStart.classList.remove('btn-active-blue'); }
  setText('t-drift', t.drift || '--');
  setText('t-rms', t.rms || '--');
  setText('t-solve', t.solve_time || '--');
  setText('t-rate', t.solve_rate || '--');
  setText('t-ml', t.ml_samples || '--');
  setText('t-auto', t.auto_solve_status || '--');
  // Solved FOV from ASTAP (live-updated each poll)
  const fovEl = document.getElementById('t-solved-fov');
  if (fovEl) {
    const sfov = t.last_solved_fov || 0;
    if (sfov > 0) {
      fovEl.textContent = sfov.toFixed(2) + '\u00b0';
      fovEl.style.color = 'var(--accent)';
    } else {
      fovEl.textContent = '--';
      fovEl.style.color = '';
    }
  }

  document.getElementById('btn-solve-start').disabled = isSolvingNow;
  document.getElementById('btn-solve-stop').disabled = !isSolvingNow;

  // PEC + Mount
  setText('pec-status', pec.status || '--');
  setText('pec-periods', pec.periods || '--');
  setText('pec-corr', pec.correction || '--');
  const pecBadge = document.getElementById('pec-badge');
  if (pec.enabled) {
    pecBadge.textContent = 'On'; pecBadge.className = 'badge badge-green';
  } else {
    pecBadge.textContent = 'Off'; pecBadge.className = 'badge badge-red';
  }
  // Mount drive type selector
  const mountInfo = s.mount || {};
  const dtSel = document.getElementById('mount-drive-type');
  if (dtSel && mountInfo.drive_type && dtSel.value !== mountInfo.drive_type) {
    dtSel.value = mountInfo.drive_type;
  }
  // Flexure model
  const flexBadge = document.getElementById('flexure-badge');
  if (flexBadge) {
    if (mountInfo.flexure_learning) {
      flexBadge.textContent = 'Learning'; flexBadge.className = 'badge badge-green';
    } else if (mountInfo.flexure_enabled) {
      flexBadge.textContent = 'Active'; flexBadge.className = 'badge badge-blue';
    } else {
      flexBadge.textContent = 'Off'; flexBadge.className = 'badge badge-red';
    }
  }
  setText('flexure-samples', mountInfo.flexure_samples || 0);
  setText('flexure-coverage', mountInfo.flexure_coverage_pct !== undefined
    ? mountInfo.flexure_coverage_pct.toFixed(1) : '0');

  // Weather
  setText('w-location', w.location || '--');
  setText('w-temp', w.temperature || '--');
  setText('w-hum', w.humidity || '--');
  setText('w-press', w.pressure || '--');
  setText('w-cloud', w.cloud_cover || '--');
  setText('w-wind', w.wind_speed || '--');
  setText('w-gust', w.gusts || '--');
  setText('w-dew', w.dew_point || '--');
  setText('w-dewrisk', w.dew_risk || '--');
  setText('w-cond', w.conditions || '--');
  setText('w-obs', w.observing_quality || '--');
  setText('w-status', w.status || '--');
  // Show "no internet" hint when weather data is all defaults
  const weatherNoInet = document.getElementById('weather-no-internet');
  if (weatherNoInet) {
    const wStatus = (w.status || '').toLowerCase();
    const allDash = (w.temperature || '--') === '--' && (w.humidity || '--') === '--';
    weatherNoInet.style.display = (allDash || wStatus.includes('error') || wStatus.includes('not configured')) ? '' : 'none';
  }

  // Controls
  setText('slew-status', ctrl.telescope_status || 'Stopped');
  setText('focus-pos', ctrl.focuser_position || '--');
  setText('focus-status', ctrl.focuser_status || '--');
  setText('derot-angle', ctrl.derotator_angle || '--');
  setText('derot-status', ctrl.derotator_status || '--');

  // OnStep extended state
  const os = s.onstep || {};

  // Park state
  setText('park-state', os.park_state || 'Unknown');

  // Firmware
  setText('fw-name', os.firmware_name || '--');
  setText('fw-version', os.firmware_version || '--');
  setText('fw-mount-type', os.firmware_mount_type || '--');

  // Tracking rate
  setText('track-rate', os.tracking_rate || 'Sidereal');
  const teBadge = document.getElementById('track-enabled-badge');
  if (teBadge) {
    if (os.tracking_enabled) {
      teBadge.textContent = 'Tracking'; teBadge.className = 'badge badge-green';
    } else {
      teBadge.textContent = 'Off'; teBadge.className = 'badge badge-red';
    }
  }

  // Mount PEC
  setText('mpec-status', os.mount_pec_status || '--');
  setText('mpec-recorded', os.mount_pec_recorded ? 'Yes' : 'No');
  const mpecBadge = document.getElementById('mpec-badge');
  if (mpecBadge) {
    const mps = (os.mount_pec_status || '').toLowerCase();
    if (mps.includes('play')) { mpecBadge.textContent = 'Playing'; mpecBadge.className = 'badge badge-green'; }
    else if (mps.includes('record')) { mpecBadge.textContent = 'Recording'; mpecBadge.className = 'badge badge-yellow'; }
    else { mpecBadge.textContent = 'Idle'; mpecBadge.className = 'badge badge-red'; }
  }

  // Backlash
  setText('bl-ra', os.backlash_ra || '--');
  setText('bl-dec', os.backlash_dec || '--');

  // Limits
  setText('lim-horizon', os.horizon_limit || '--');
  setText('lim-overhead', os.overhead_limit || '--');

  // Extended Focuser
  setText('focus-temp', os.focuser_temperature || '--');
  setText('focus-tcf', os.focuser_tcf ? 'On' : 'Off');
  const fSel = document.getElementById('focuser-select');
  if (fSel && os.focuser_selected && document.activeElement !== fSel) {
    fSel.value = os.focuser_selected;
  }

  // Rotator
  setText('rot-angle', os.rotator_angle || '--');
  setText('rot-status', os.rotator_status || 'Stopped');
  const rotDerotBtn = document.getElementById('rot-derotate-btn');
  if (rotDerotBtn) {
    rotDerotBtn.textContent = os.rotator_derotating ? 'Derotate Off' : 'Derotate On';
  }

  // Auxiliary features (Control tab - simple sliders)
  const auxList = document.getElementById('aux-features-list');
  if (auxList && os.auxiliary_features) {
    const feats = os.auxiliary_features;
    if (feats.length === 0) {
      auxList.innerHTML = 'No auxiliary features discovered.';
    } else {
      let html = '';
      for (const f of feats) {
        html += '<div style="display:flex;align-items:center;gap:6px;margin:3px 0">'
          + '<span style="flex:1">' + (f.name||'?') + ' (slot ' + f.slot + ')</span>'
          + '<input type="range" min="0" max="255" value="' + (f.value||0) + '" '
          + 'style="width:80px" id="aux-slider-' + f.slot + '" '
          + 'onchange="setAux(' + f.slot + ', this.value)">'
          + '<span style="min-width:28px;text-align:right" id="aux-val-' + f.slot + '">' + (f.value||0) + '</span>'
          + '</div>';
      }
      auxList.innerHTML = html;
    }
  }
  // Auxiliary features (Location tab - smart UI with feature-specific controls)
  // Only re-render when feature list structure changes (not every poll)
  if (os.auxiliary_features && typeof renderAuxFeatures === 'function') {
    const newKey = os.auxiliary_features.map(f => f.slot + ':' + f.name).join('|');
    if (window._auxFeatKey !== newKey) {
      window._auxFeatKey = newKey;
      renderAuxFeatures(os.auxiliary_features);
    }
    // Update values in-place without rebuilding UI
    auxUpdateValues(os.auxiliary_features);
  }

  // Camera tab focuser mirror
  updateCamFocuser(ctrl);

  // Session download link: persist across page reloads
  const sess = s.session || {};
  const dlLink = document.getElementById('session-download-link');
  if (dlLink && (sess.has_session || isTracking)) {
    dlLink.style.display = '';
  }
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

// ============================================================
// Enhanced Log System
// ============================================================
// Tag badge labels (short, uppercase)
const LOG_TAG_LABELS = {
  error: 'ERR', warning: 'WARN', success: 'OK', info: 'INFO',
  cmd: 'CMD', rate: 'RATE', tracking: 'TRK', server: 'SRV',
  response: 'RSP', usb: 'USB'
};

// Active tag filters (all on by default)
let logActiveFilters = new Set(['error','warning','success','info','cmd','rate','tracking','server','response','usb']);

// Log counters for stats bar
let logCounts = { total: 0, error: 0, warning: 0, success: 0, cmd: 0 };

// Auto-scroll state
let logAutoScroll = true;

// Detect when user scrolls up — pause auto-scroll
(function() {
  const box = document.getElementById('log-box');
  if (!box) return;
  box.addEventListener('scroll', function() {
    const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
    logAutoScroll = atBottom;
    const btn = document.getElementById('log-scroll-btn');
    if (btn) {
      if (atBottom) btn.classList.remove('visible');
      else btn.classList.add('visible');
    }
  });
})();

function logScrollToBottom() {
  const box = document.getElementById('log-box');
  if (box) { box.scrollTop = box.scrollHeight; logAutoScroll = true; }
  const btn = document.getElementById('log-scroll-btn');
  if (btn) btn.classList.remove('visible');
}

// Format Unix timestamp to HH:MM:SS
function logFormatTs(ts) {
  if (!ts) return '--:--:--';
  const d = new Date(ts * 1000);
  return d.getHours().toString().padStart(2,'0') + ':' +
         d.getMinutes().toString().padStart(2,'0') + ':' +
         d.getSeconds().toString().padStart(2,'0');
}

// Tag filter toggle
function logToggleFilter(btn, tag) {
  if (tag === 'all') {
    // Toggle all on/off
    const allBtn = btn;
    const allOn = allBtn.classList.contains('active');
    document.querySelectorAll('#log-toolbar .btn-sm').forEach(function(b) {
      if (allOn) b.classList.remove('active'); else b.classList.add('active');
    });
    if (allOn) logActiveFilters.clear();
    else logActiveFilters = new Set(['error','warning','success','info','cmd','rate','tracking','server','response','usb']);
  } else {
    btn.classList.toggle('active');
    if (btn.classList.contains('active')) logActiveFilters.add(tag);
    else logActiveFilters.delete(tag);
    // Update "All" button state
    const allBtn = document.querySelector('#log-toolbar [data-filter="all"]');
    if (allBtn) {
      if (logActiveFilters.size >= 6) allBtn.classList.add('active');
      else allBtn.classList.remove('active');
    }
  }
  logApplyVisibility();
}

// Apply filter + search visibility to all existing lines
function logApplyVisibility() {
  const searchTerm = (document.getElementById('log-search') || {}).value || '';
  const searchLower = searchTerm.toLowerCase();
  document.querySelectorAll('#log-box .log-line').forEach(function(line) {
    const tag = line.dataset.tag || 'info';
    const msg = (line.dataset.msg || '').toLowerCase();
    const tagOk = logActiveFilters.has(tag);
    const searchOk = !searchLower || msg.indexOf(searchLower) !== -1;
    line.classList.toggle('hidden', !(tagOk && searchOk));
  });
}

function logApplySearch() {
  clearTimeout(searchTimeout);
  searchTimeout = setTimeout(logApplyVisibility, 200);
}

// Update stats bar
function logUpdateStats() {
  var el;
  el = document.getElementById('ls-total'); if (el) el.textContent = logCounts.total;
  el = document.getElementById('ls-errors'); if (el) el.textContent = logCounts.error;
  el = document.getElementById('ls-warnings'); if (el) el.textContent = logCounts.warning;
  el = document.getElementById('ls-success'); if (el) el.textContent = logCounts.success;
  el = document.getElementById('ls-cmds'); if (el) el.textContent = logCounts.cmd;
}

// Set server log level
function logSetLevel(level) {
  apiPost('/api/log/level', { level: level }).then(function(r) {
    if (r && r.ok) toast('Log level set to ' + level, 'ok');
    else toast('Failed to set log level', 'err');
  });
}

// Clear log display (client-side only)
function logClear() {
  const box = document.getElementById('log-box');
  if (box) box.innerHTML = '';
  logCounts = { total: 0, error: 0, warning: 0, success: 0, cmd: 0 };
  logUpdateStats();
}

// Copy all visible log lines to clipboard
function logCopy() {
  const lines = [];
  document.querySelectorAll('#log-box .log-line:not(.hidden)').forEach(function(line) {
    lines.push(line.textContent);
  });
  if (!lines.length) { toast('No log lines to copy', 'info'); return; }
  navigator.clipboard.writeText(lines.join('\n')).then(function() {
    toast('Copied ' + lines.length + ' log lines', 'ok');
  }).catch(function() { toast('Copy failed', 'err'); });
}

// Export log as downloadable .txt file
function logExport() {
  const lines = [];
  document.querySelectorAll('#log-box .log-line:not(.hidden)').forEach(function(line) {
    lines.push(line.textContent);
  });
  if (!lines.length) { toast('No log lines to export', 'info'); return; }
  const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'telescope_log_' + new Date().toISOString().slice(0,19).replace(/:/g,'-') + '.txt';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  toast('Exported ' + lines.length + ' lines', 'ok');
}

// Main log polling (1 Hz)
async function pollLog() {
  const d = await apiGet('/api/log?since=' + lastLogSeq);
  if (!d || !d.lines || !d.lines.length) return;
  const box = document.getElementById('log-box');
  if (!box) return;
  const searchTerm = ((document.getElementById('log-search') || {}).value || '').toLowerCase();

  for (const l of d.lines) {
    lastLogSeq = l.seq;
    const tag = l.tag || 'info';
    const ts = logFormatTs(l.ts);
    const label = LOG_TAG_LABELS[tag] || tag.toUpperCase();

    // Build log line element
    const div = document.createElement('div');
    div.className = 'log-line ' + tag;
    div.dataset.tag = tag;
    div.dataset.msg = l.msg || '';

    // Timestamp span
    const tsSpan = document.createElement('span');
    tsSpan.className = 'log-ts';
    tsSpan.textContent = ts;
    div.appendChild(tsSpan);

    // Tag badge span
    const tagSpan = document.createElement('span');
    tagSpan.className = 'log-tag';
    tagSpan.textContent = label;
    div.appendChild(tagSpan);

    // Message span
    const msgSpan = document.createElement('span');
    msgSpan.className = 'log-msg';
    msgSpan.textContent = l.msg || '';
    div.appendChild(msgSpan);

    // Apply visibility based on filters and search
    const tagOk = logActiveFilters.has(tag);
    const searchOk = !searchTerm || (l.msg || '').toLowerCase().indexOf(searchTerm) !== -1;
    if (!(tagOk && searchOk)) div.classList.add('hidden');

    box.appendChild(div);

    // Update counters
    logCounts.total++;
    if (tag in logCounts) logCounts[tag]++;
  }

  // Auto-scroll only if user hasn't scrolled up
  if (logAutoScroll) box.scrollTop = box.scrollHeight;

  // Trim old DOM nodes
  while (box.children.length > 1500) box.removeChild(box.firstChild);

  // Update stats
  logUpdateStats();
}

// ============================================================
// Commands
// ============================================================
function doConnect() {
  // Save connection settings first, then connect
  const ct = document.getElementById('conn-type').value;
  const body = { type: ct };
  body.protocol = document.getElementById('conn-protocol').value;
  if (ct === 'USB') {
    body.port = document.getElementById('conn-port').value;
    body.baudrate = document.getElementById('conn-baud').value;
  } else {
    body.wifi_ip = document.getElementById('conn-wifi-ip').value;
    body.wifi_port = document.getElementById('conn-wifi-port').value;
  }
  apiPost('/api/connection/settings', body).then(() => apiPost('/api/connect'));
}
function doDisconnect() { apiPost('/api/disconnect'); }
function doSimulator()  { apiPost('/api/simulator'); }

// ── Auto-connect ─────────────────────────────────────────────
function saveAutoConnect() {
  const enabled = document.getElementById('auto-connect-chk').checked;
  const ct = document.getElementById('conn-type').value;
  const body = { enabled: enabled, type: ct };
  if (ct === 'WiFi') {
    body.wifi_ip = document.getElementById('conn-wifi-ip').value;
    body.wifi_port = document.getElementById('conn-wifi-port').value;
  } else {
    body.baudrate = document.getElementById('conn-baud').value;
  }
  apiPost('/api/autoconnect', body);
}

function checkAutoConnect() {
  apiGet('/api/autoconnect').then(d => {
    if (!d) return;
    const chk = document.getElementById('auto-connect-chk');
    if (chk) chk.checked = !!d.enabled;
    // Populate saved connection settings into the form
    if (d.type) {
      const sel = document.getElementById('conn-type');
      if (sel) { sel.value = d.type === 'USB' ? 'USB' : 'WiFi'; updateConnTypeUI(); }
    }
    if (d.wifi_ip) {
      const el = document.getElementById('conn-wifi-ip');
      if (el) el.value = d.wifi_ip;
    }
    if (d.wifi_port) {
      const el = document.getElementById('conn-wifi-port');
      if (el) el.value = d.wifi_port;
    }
    // Auto-connect if enabled and not already connected
    if (d.enabled) {
      setTimeout(() => {
        const badge = document.getElementById('conn-badge');
        if (badge && badge.textContent.toLowerCase().includes('disconnected')) {
          console.log('Auto-connecting...');
          doConnect();
        }
      }, 1500);  // small delay so the UI has time to render
    }
  });
}

function doTrackStart() { apiPost('/api/tracking/start'); }
function doTrackStop()  { apiPost('/api/tracking/stop'); }
async function doSessionSave() {
  const d = await apiPost('/api/session/save');
  if (d && d.ok) {
    document.getElementById('session-download-link').style.display = '';
    toast('Session saved: ' + (d.path || ''), 'ok');
  }
}
function doSessionDownload() {
  // Use an invisible iframe to trigger the download.
  // This works reliably in Android WebView with DownloadListener,
  // and also works in desktop browsers.
  const iframe = document.createElement('iframe');
  iframe.style.display = 'none';
  iframe.src = '/api/session/download';
  document.body.appendChild(iframe);
  // Clean up after a delay (download will have started by then)
  setTimeout(() => { document.body.removeChild(iframe); }, 10000);
  toast('Starting download...', 'ok');
}
async function doSolveStart() {
  try {
    await applyCameraSettings();
  } catch (e) {
    console.warn('applyCameraSettings error:', e);
  }
  await apiPost('/api/solve/start');
}
async function doSolveStop() { await apiPost('/api/solve/stop'); }
function doPark()       { apiPost('/api/park'); }
function doHome()       { apiPost('/api/home'); }
function confirmPark()  { if (confirm('Park the telescope? This will end tracking.')) doPark(); }
function confirmHome()  { if (confirm('Home the telescope?\\n\\nThis resets the mount position to 0,0 (Alt/Az).\\nThis is normal -- use GoTo to slew to a new target afterwards.')) doHome(); }
function doUnpark()     { apiPost('/api/unpark'); }
function doHomeFind()   { apiPost('/api/home/find'); }

// OnStep: Tracking rate
function setTrackRate(rate) { apiPost('/api/tracking/rate', { rate }); }

// OnStep: Mount PEC (handled inline in HTML with apiPost)

// OnStep: Backlash
function setBacklash(axis) {
  const el = document.getElementById('bl-' + axis + '-input');
  if (!el) return;
  const val = parseInt(el.value);
  if (isNaN(val) || val < 0) { alert('Enter a positive arcsec value'); return; }
  apiPost('/api/mount/backlash', { axis, value: val });
}

// OnStep: Limits
function setLimit(type) {
  const el = document.getElementById('lim-' + type + '-input');
  if (!el) return;
  const val = parseInt(el.value);
  if (isNaN(val)) { alert('Enter degrees'); return; }
  apiPost('/api/mount/limits', { type, degrees: val });
}

// OnStep: Auxiliary
function setAux(slot, value) {
  const valEl = document.getElementById('aux-val-' + slot);
  if (valEl) valEl.textContent = value;
  apiPost('/api/mount/auxiliary/set', { slot: parseInt(slot), value: parseInt(value) });
}

// OnStep: Extended Focuser
function focuserGoto() {
  const el = document.getElementById('focus-goto-pos');
  if (!el) return;
  const pos = parseInt(el.value);
  if (isNaN(pos)) { alert('Enter a position'); return; }
  apiPost('/api/focuser/goto', { position: pos });
}
function focuserToggleTCF() {
  const cur = document.getElementById('focus-tcf');
  const enabled = cur && cur.textContent === 'On';
  apiPost('/api/focuser/tcf', { enabled: !enabled });
}
function focuserSelect() {
  const sel = document.getElementById('focuser-select');
  if (sel) apiPost('/api/focuser/select', { focuser: parseInt(sel.value) });
}

// OnStep: Rotator
function rotatorMove(dir)  { apiPost('/api/rotator/move', { direction: dir }); }
function rotatorStop()     { apiPost('/api/rotator/stop'); }
function rotatorGoto() {
  const el = document.getElementById('rot-goto-angle');
  if (!el) return;
  const angle = parseFloat(el.value);
  if (isNaN(angle)) { alert('Enter angle in degrees'); return; }
  apiPost('/api/rotator/goto', { angle });
}

// Connection type toggle -- UI only (safe to call from polling/restore)
function updateConnTypeUI() {
  const isUSB = document.getElementById('conn-type').value === 'USB';
  document.getElementById('conn-usb-fields').style.display = isUSB ? 'block' : 'none';
  document.getElementById('conn-wifi-fields').style.display = isUSB ? 'none' : 'block';
}
// User-triggered mode change: update UI + sync to backend immediately
function onConnTypeChange() {
  updateConnTypeUI();
  apiPost('/api/connection/settings', { type: document.getElementById('conn-type').value });
}

// Protocol change: update default WiFi port + sync to backend immediately
function onProtocolChange() {
  const proto = document.getElementById('conn-protocol').value;
  const portEl = document.getElementById('conn-wifi-port');
  const defaults = { lx200: '9996', nexstar: '11882', ioptron: '8080', audiostar: '4030', alpaca: '11111', indi: '7624' };
  const oldDefaults = Object.values(defaults);
  const body = { protocol: proto };
  // Only auto-update port if the current value is a known default
  if (oldDefaults.includes(portEl.value) || portEl.value === '') {
    portEl.value = defaults[proto] || '9996';
    body.wifi_port = portEl.value;
  }
  // Sync to backend so polling (when connected) reads correct values
  apiPost('/api/connection/settings', body);
}

// Serial port dropdown refresh
async function refreshSerialPorts() {
  const sel = document.getElementById('conn-port');
  const prev = sel.value;
  sel.innerHTML = '<option value="">Scanning...</option>';
  const d = await apiGet('/api/serial/ports');
  if (!d || !d.ports || d.ports.length === 0) {
    sel.innerHTML = '<option value="">No ports found</option>';
    return;
  }
  const detected = new Set(d.detected || []);
  sel.innerHTML = '';
  for (const p of d.ports) {
    const opt = document.createElement('option');
    opt.value = p;
    opt.textContent = detected.has(p) ? p + ' *' : p;
    sel.appendChild(opt);
  }
  // Restore previous selection if still available, else pick first detected
  if (prev && d.ports.includes(prev)) {
    sel.value = prev;
  } else if (d.detected && d.detected.length > 0) {
    sel.value = d.detected[0];
  }
}

// Solve mode toggle
function onSolveModeChange() {
  const m = document.getElementById('solve-mode').value;
  // On Android, all modes use the same camera bridge -- no extra fields needed
  if (window._isAndroid) return;
  const camF = document.getElementById('solve-camera-fields');
  if (camF) camF.style.display = m === 'camera' ? 'block' : 'none';
  const ascomF = document.getElementById('solve-ascom-fields');
  if (ascomF) ascomF.style.display = m === 'ascom' ? 'block' : 'none';
}

// ASCOM camera list / select
function loadAscomCameras() {
  const box = document.getElementById('ascom-cam-list');
  box.innerHTML = '<div class="text-dim" style="font-size:.8em;padding:4px">Loading cameras...</div>';
  box.style.display = 'block';
  apiGet('/api/ascom/cameras').then(r => {
    if (!r || !r.cameras || !r.cameras.length) {
      box.innerHTML = '<div class="text-dim" style="font-size:.8em;padding:4px;color:var(--yellow)">No ASCOM cameras found. Check that ASCOM Platform and camera drivers are installed.</div>';
      return;
    }
    box.innerHTML = '';
    for (const cam of r.cameras) {
      const btn = document.createElement('button');
      btn.className = 'btn btn-dim btn-sm btn-block';
      btn.style.cssText = 'font-size:.8em;margin-bottom:3px;text-align:left;justify-content:flex-start';
      btn.textContent = cam.name + ' (' + cam.id + ')';
      btn.addEventListener('click', () => selectAscomCamera(cam.id, cam.name));
      box.appendChild(btn);
    }
  });
}

function selectAscomCamera(id, name) {
  apiPost('/api/ascom/select', { camera_id: id }).then(r => {
    if (r && r.ok) {
      setText('ascom-cam-name', r.name);
      document.getElementById('ascom-cam-list').style.display = 'none';
    }
  });
}

// Apply camera/solve settings
function applyCameraSettings() {
  const mode = document.getElementById('solve-mode').value;
  const body = {
    solve_mode: mode,
    camera_index: (document.getElementById('cam-index') || {}).value || '0',
    solve_interval: (document.getElementById('solve-interval') || {}).value || '4.0'
  };
  // ASCOM fields may have been removed on Android -- null-safe access
  const ascomExp = document.getElementById('ascom-exp');
  if (ascomExp) {
    body.ascom_exposure = ascomExp.value;
    body.ascom_gain = (document.getElementById('ascom-gain') || {}).value || '100';
    body.ascom_binning = (document.getElementById('ascom-bin') || {}).value || '2';
  }
  // Android camera source hint for plate solving
  if (window._isAndroid) {
    body.android_source = mode;  // mode is auto/zwo/uvc/phone on Android
  }
  return apiPost('/api/camera/settings', body);
}

// GPS: try browser geolocation first, fall back to IP-based lookup
function getGPS() {
  const el = document.getElementById('loc-status');
  el.textContent = 'Acquiring location...';
  el.style.color = 'var(--yellow)';

  // Fill lat/lon inputs from result
  function applyCoords(lat, lon, source, extra) {
    document.getElementById('loc-lat').value = lat.toFixed(6);
    document.getElementById('loc-lon').value = lon.toFixed(6);
    const label = source === 'gps' ? 'GPS acquired' : ('IP location: ' + (extra || ''));
    el.textContent = label + ' - click Set Location to apply';
    el.style.color = 'var(--green)';
    window._gpsPending = true;
  }

  // Fallback: server-side IP geolocation (works over HTTP)
  function ipFallback() {
    el.textContent = 'Browser GPS unavailable, trying IP location...';
    el.style.color = 'var(--yellow)';
    fetch('/api/gps', {headers: _authHeaders()}).then(r => r.json()).then(d => {
      if (d.ok) {
        const place = [d.city, d.country].filter(Boolean).join(', ');
        applyCoords(d.latitude, d.longitude, 'ip', place);
      } else {
        el.textContent = 'Location failed: ' + (d.error || 'unknown');
        el.style.color = 'var(--red)';
      }
    }).catch(e => {
      el.textContent = 'Location failed: ' + e.message;
      el.style.color = 'var(--red)';
    });
  }

  // Try browser Geolocation API first (needs HTTPS or localhost)
  if (navigator.geolocation && window.isSecureContext) {
    navigator.geolocation.getCurrentPosition(
      pos => applyCoords(pos.coords.latitude, pos.coords.longitude, 'gps', ''),
      () => ipFallback(),
      { enableHighAccuracy: true, timeout: 10000 }
    );
  } else {
    ipFallback();
  }
}

// ---- Initialize buttons ----
function _initStatus(msg, color) {
  const el = document.getElementById('init-status');
  if (el) { el.textContent = msg; el.style.color = color || 'var(--dim)'; }
}
function doSetTimeDate() {
  _initStatus('Sending time & date...', 'var(--yellow)');
  apiPost('/api/mount/set-time').then(r => {
    _initStatus(r && r.ok ? 'Time & date sent to mount' : (r && r.error || 'Failed'), r && r.ok ? 'var(--green)' : 'var(--red)');
  }).catch(() => _initStatus('Network error', 'var(--red)'));
}
function doSetSite() {
  const lat = parseFloat(document.getElementById('loc-lat').value);
  const lon = parseFloat(document.getElementById('loc-lon').value);
  if (isNaN(lat) || isNaN(lon)) {
    _initStatus('Enter valid coordinates first', 'var(--red)');
    return;
  }
  _initStatus('Saving & sending site...', 'var(--yellow)');
  // Save coords to app, then send to mount
  apiPost('/api/location', { latitude: lat, longitude: lon }).then(r => {
    if (!r || !r.ok) { _initStatus(r ? r.error : 'Failed', 'var(--red)'); return; }
    apiPost('/api/mount/set-site').then(r2 => {
      _initStatus(r2 && r2.ok ? 'Site saved & sent to mount' : (r2 && r2.error || 'Failed'), r2 && r2.ok ? 'var(--green)' : 'var(--red)');
    }).catch(() => _initStatus('Network error', 'var(--red)'));
  }).catch(() => _initStatus('Network error', 'var(--red)'));
}
function doResetHome() {
  _initStatus('Resetting home position...', 'var(--yellow)');
  apiPost('/api/home').then(r => {
    _initStatus(r && r.ok ? 'Home position reset (0,0)' : (r && r.error || 'Failed'), r && r.ok ? 'var(--green)' : 'var(--red)');
  }).catch(() => _initStatus('Network error', 'var(--red)'));
}
function doReturnHome() {
  _initStatus('Returning to home position...', 'var(--yellow)');
  apiPost('/api/home/find').then(r => {
    _initStatus(r && r.ok ? 'Finding home position...' : (r && r.error || 'Failed'), r && r.ok ? 'var(--green)' : 'var(--red)');
  }).catch(() => _initStatus('Network error', 'var(--red)'));
}

// ---- Auxiliary Features (smart UI) ----
function auxDiscover() {
  const el = document.getElementById('aux-status');
  if (el) { el.textContent = 'Discovering features...'; el.style.color = 'var(--yellow)'; }
  apiPost('/api/mount/auxiliary/discover').then(r => {
    if (el) { el.textContent = r && r.ok ? 'Discovery complete' : (r && r.error || 'Failed'); el.style.color = r && r.ok ? 'var(--green)' : 'var(--red)'; }
    // Immediately refresh to show results
    setTimeout(() => auxRefresh(), 500);
  }).catch(() => { if (el) { el.textContent = 'Network error'; el.style.color = 'var(--red)'; } });
}
function auxRefresh() {
  apiPost('/api/mount/auxiliary/refresh').then(() => {}).catch(() => {});
}
function auxSetValue(slot, value) {
  const valEl = document.getElementById('aux-val-loc-' + slot);
  if (valEl) valEl.textContent = value;
  apiPost('/api/mount/auxiliary/set', { slot: parseInt(slot), value: parseInt(value) });
}

// Update auxiliary feature values in-place without rebuilding HTML.
// Called every poll tick to keep displayed values fresh while
// preserving slider/input positions the user may have adjusted.
function auxUpdateValues(feats) {
  if (!feats || feats.length === 0) return;
  let intvMainSlotEl = document.getElementById('aux-intv-main-slot');
  let intvMainSlot = intvMainSlotEl ? parseInt(intvMainSlotEl.value) : -1;
  for (const f of feats) {
    const val = parseInt(f.value) || 0;
    // Update value display labels (both dashboard and location tab)
    const valEl = document.getElementById('aux-val-loc-' + f.slot);
    if (valEl) valEl.textContent = val;
    const valElDash = document.getElementById('aux-val-' + f.slot);
    if (valElDash) valElDash.textContent = val;
    // Update dashboard slider position (simple list on Control tab)
    const dashSlider = document.getElementById('aux-slider-' + f.slot);
    if (dashSlider && document.activeElement !== dashSlider) {
      dashSlider.value = val;
    }
    // Intervalometer: update Start/Stop button styling based on running state
    if (f.slot === intvMainSlot) {
      const running = val > 0;
      const startBtn = document.getElementById('aux-intv-start-btn');
      const stopBtn = document.getElementById('aux-intv-stop-btn');
      if (startBtn) {
        startBtn.style.background = !running ? 'rgba(200,40,40,.85)' : 'rgba(160,120,120,.25)';
        startBtn.style.color = !running ? 'rgba(255,255,255,.9)' : 'rgba(200,150,150,.55)';
      }
      if (stopBtn) {
        stopBtn.style.background = running ? 'rgba(200,40,40,.85)' : 'rgba(160,120,120,.25)';
        stopBtn.style.color = running ? 'rgba(255,255,255,.9)' : 'rgba(200,150,150,.55)';
      }
    }
  }
}

// Render smart auxiliary feature controls based on feature name/purpose
// Groups intervalometer-related slots into one panel
function renderAuxFeatures(feats) {
  const container = document.getElementById('aux-features-loc');
  if (!container) return;
  if (!feats || feats.length === 0) {
    container.innerHTML = '<div class="text-dim">No auxiliary features discovered. Tap Discover to detect.</div>';
    return;
  }
  // Group intervalometer-related features together
  const intvFeats = [];
  const otherFeats = [];
  for (const f of feats) {
    const n = (f.name || '').toLowerCase();
    if (n.includes('interval') || n.includes('intv') || n.includes('shutter') || n.includes('camera')) {
      intvFeats.push(f);
    } else {
      otherFeats.push(f);
    }
  }
  let html = '';
  // Render grouped intervalometer panel if any intervalometer slots found
  if (intvFeats.length > 0) {
    html += renderIntervalometerPanel(intvFeats);
  }
  // Render other features individually
  for (const f of otherFeats) {
    const name = f.name || 'Feature ' + f.slot;
    const val = parseInt(f.value) || 0;
    html += '<div style="border:1px solid rgba(255,180,80,.15);border-radius:8px;padding:8px 10px;margin-bottom:6px;background:rgba(0,0,0,.2)">';
    html += '<div style="font-weight:600;color:var(--accent);font-size:.88em;margin-bottom:4px">' + name + ' <span class="text-dim" style="font-size:.72em">(slot ' + f.slot + ')</span></div>';
    const nameLow = name.toLowerCase();
    if (nameLow.includes('dew') || nameLow.includes('heater')) {
      html += renderHeaterUI(f, val);
    } else if (nameLow.includes('switch') || nameLow.includes('relay') || nameLow.includes('light')) {
      html += renderSwitchUI(f, val);
    } else {
      html += renderGenericAuxUI(f, val);
    }
    html += '</div>';
  }
  container.innerHTML = html;
}

// ---- Intervalometer panel (OnStep-style clean UI) ----
function renderIntervalometerPanel(feats) {
  let mainSlot = feats[0].slot;
  let mainVal = parseInt(feats[0].value) || 0;
  let mainName = feats[0].name || 'CAMERA CTR';
  const running = mainVal > 0;

  let h = '<div style="border-radius:10px;padding:14px;margin-bottom:8px;background:rgba(0,0,0,.18)">';
  // Title - show feature name centered
  h += '<div style="text-align:center;font-size:1.15em;color:var(--fg);margin-bottom:10px;opacity:.8">' + mainName + '</div>';

  // Start / Stop segmented control
  h += '<div style="display:flex;border-radius:6px;overflow:hidden;margin-bottom:16px;border:1px solid rgba(255,255,255,.08)">';
  h += '<button id="aux-intv-start-btn" onclick="auxIntvStart()" style="flex:1;padding:12px;font-size:1.05em;font-weight:600;border:none;cursor:pointer;';
  h += !running ? 'background:rgba(200,40,40,.85);color:rgba(255,255,255,.9)' : 'background:rgba(160,120,120,.25);color:rgba(200,150,150,.55)';
  h += '">Start</button>';
  h += '<button id="aux-intv-stop-btn" onclick="auxIntvStop()" style="flex:1;padding:12px;font-size:1.05em;font-weight:600;border:none;cursor:pointer;';
  h += running ? 'background:rgba(200,40,40,.85);color:rgba(255,255,255,.9)' : 'background:rgba(160,120,120,.25);color:rgba(200,150,150,.55)';
  h += '">Stop</button>';
  h += '</div>';

  // Count slider
  h += '<div style="display:flex;align-items:center;margin-bottom:14px;gap:10px">';
  h += '<span style="font-size:.98em;color:var(--fg);min-width:140px;opacity:.7" id="aux-intv-count-label">Count (33X)</span>';
  h += '<input type="range" min="1" max="255" value="33" id="aux-intv-count-slider" style="flex:1" ';
  h += 'oninput="document.getElementById(\'aux-intv-count-label\').textContent=\'Count (\'+this.value+\'X)\'">';
  h += '</div>';

  // Exposure slider
  h += '<div style="display:flex;align-items:center;margin-bottom:14px;gap:10px">';
  h += '<span style="font-size:.98em;color:var(--fg);min-width:140px;opacity:.7" id="aux-intv-exp-label">Exp. (' + auxExpFromSlider(1) + ')</span>';
  h += '<input type="range" min="0" max="240" value="1" id="aux-intv-exp-slider" style="flex:1" ';
  h += 'oninput="document.getElementById(\'aux-intv-exp-label\').textContent=\'Exp. (\'+auxExpFromSlider(this.value)+\')\'">';
  h += '</div>';

  // Delay slider
  h += '<div style="display:flex;align-items:center;margin-bottom:6px;gap:10px">';
  h += '<span style="font-size:.98em;color:var(--fg);min-width:140px;opacity:.7" id="aux-intv-delay-label">Delay (1s)</span>';
  h += '<input type="range" min="0" max="60" value="1" id="aux-intv-delay-slider" style="flex:1" ';
  h += 'oninput="document.getElementById(\'aux-intv-delay-label\').textContent=\'Delay (\'+this.value+\'s)\'">';
  h += '</div>';

  // Hidden: main slot reference and value display
  h += '<input type="hidden" id="aux-intv-main-slot" value="' + mainSlot + '">';
  h += '<span style="display:none" id="aux-val-loc-' + mainSlot + '">' + mainVal + '</span>';
  h += '</div>';
  return h;
}

// Intervalometer: non-linear exposure slider mapping
// Slider 0-240 maps to 0.01s - 120s with fine control at low end
function auxExpFromSlider(v) {
  v = parseInt(v);
  if (v <= 0) return '0.01s';
  if (v <= 50) {
    // 0-50 -> 0.01s to 0.50s (0.01s steps)
    return (v * 0.01).toFixed(2) + 's';
  } else if (v <= 90) {
    // 50-90 -> 0.5s to 1.0s (0.0125s steps)
    return (0.5 + (v - 50) * 0.0125).toFixed(2) + 's';
  } else if (v <= 130) {
    // 90-130 -> 1.0s to 10.0s (0.225s steps)
    return (1.0 + (v - 90) * 0.225).toFixed(1) + 's';
  } else if (v <= 180) {
    // 130-180 -> 10s to 30s (0.4s steps)
    return Math.round(10 + (v - 130) * 0.4) + 's';
  } else if (v <= 220) {
    // 180-220 -> 30s to 60s (0.75s steps)
    return Math.round(30 + (v - 180) * 0.75) + 's';
  } else {
    // 220-240 -> 60s to 120s (3s steps)
    return Math.round(60 + (v - 220) * 3) + 's';
  }
}

// Convert slider position to seconds (for sending to mount)
function auxExpSliderToSec(v) {
  v = parseInt(v);
  if (v <= 0) return 0.01;
  if (v <= 50) return v * 0.01;
  if (v <= 90) return 0.5 + (v - 50) * 0.0125;
  if (v <= 130) return 1.0 + (v - 90) * 0.225;
  if (v <= 180) return 10 + (v - 130) * 0.4;
  if (v <= 220) return 30 + (v - 180) * 0.75;
  return 60 + (v - 220) * 3;
}

function auxIntvStart() {
  const slot = parseInt(document.getElementById('aux-intv-main-slot').value);
  auxSetValue(slot, 1);
  const startBtn = document.getElementById('aux-intv-start-btn');
  const stopBtn = document.getElementById('aux-intv-stop-btn');
  if (startBtn) { startBtn.style.background = 'rgba(160,120,120,.25)'; startBtn.style.color = 'rgba(200,150,150,.55)'; }
  if (stopBtn) { stopBtn.style.background = 'rgba(200,40,40,.85)'; stopBtn.style.color = 'rgba(255,255,255,.9)'; }
}

function auxIntvStop() {
  const slot = parseInt(document.getElementById('aux-intv-main-slot').value);
  auxSetValue(slot, 0);
  const startBtn = document.getElementById('aux-intv-start-btn');
  const stopBtn = document.getElementById('aux-intv-stop-btn');
  if (startBtn) { startBtn.style.background = 'rgba(200,40,40,.85)'; startBtn.style.color = 'rgba(255,255,255,.9)'; }
  if (stopBtn) { stopBtn.style.background = 'rgba(160,120,120,.25)'; stopBtn.style.color = 'rgba(200,150,150,.55)'; }
}

function renderHeaterUI(f, val) {
  const on = val > 0;
  let h = '<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">';
  h += '<button class="btn ' + (on ? 'btn-red' : 'btn-accent') + ' btn-sm" onclick="auxSetValue(' + f.slot + ',' + (on ? '0' : '128') + ')" style="min-width:60px">';
  h += on ? 'Off' : 'On';
  h += '</button>';
  h += '<span class="text-dim" style="font-size:.78em">' + (on ? 'Active' : 'Inactive') + '</span>';
  h += '</div>';
  h += '<label class="text-dim" style="font-size:.72em">Power</label>';
  h += '<div style="display:flex;align-items:center;gap:6px">';
  h += '<input type="range" min="0" max="255" value="' + val + '" style="flex:1" ';
  h += 'oninput="document.getElementById(\'aux-val-loc-' + f.slot + '\').textContent=this.value" ';
  h += 'onchange="auxSetValue(' + f.slot + ',this.value)">';
  h += '<span style="min-width:28px;text-align:right;font-size:.82em" id="aux-val-loc-' + f.slot + '">' + val + '</span>';
  h += '<span class="text-dim" style="font-size:.72em">/ 255</span>';
  h += '</div>';
  return h;
}

function renderSwitchUI(f, val) {
  const on = val > 0;
  let h = '<div style="display:flex;align-items:center;gap:10px">';
  h += '<button class="btn ' + (on ? 'btn-accent' : 'btn-dim') + ' btn-sm" onclick="auxSetValue(' + f.slot + ',' + (on ? '0' : '255') + ')" style="min-width:70px">';
  h += on ? 'ON' : 'OFF';
  h += '</button>';
  h += '<span style="font-size:.82em;color:' + (on ? 'var(--green)' : 'var(--dim)') + '">' + (on ? 'Active' : 'Inactive') + '</span>';
  h += '<span class="text-dim" style="font-size:.72em" id="aux-val-loc-' + f.slot + '">' + val + '</span>';
  h += '</div>';
  return h;
}

function renderGenericAuxUI(f, val) {
  let h = '<div style="display:flex;align-items:center;gap:6px">';
  h += '<input type="range" min="0" max="255" value="' + val + '" style="flex:1" ';
  h += 'oninput="document.getElementById(\'aux-val-loc-' + f.slot + '\').textContent=this.value" ';
  h += 'onchange="auxSetValue(' + f.slot + ',this.value)">';
  h += '<span style="min-width:28px;text-align:right;font-size:.82em" id="aux-val-loc-' + f.slot + '">' + val + '</span>';
  h += '<span class="text-dim" style="font-size:.72em">/ 255</span>';
  h += '</div>';
  return h;
}

function slewStart(dir) {
  const sp = document.getElementById('slew-speed').value;
  apiPost('/api/slew', { direction: dir, speed: parseInt(sp) });
}
function slewStop()     { apiPost('/api/slew/stop'); }

function focusMove(dir) { apiPost('/api/focuser/move', { direction: dir }); }
function focusStop()    { apiPost('/api/focuser/stop'); }

function derotRotate(dir) { apiPost('/api/derotator/rotate', { direction: dir }); }
function derotStop()    { apiPost('/api/derotator/stop'); }
function derotSync()    { apiPost('/api/derotator/sync'); }

// Prevent scrolling / zoom on touch-hold for slew, focuser, derotator buttons
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.slew-grid .btn, .ctrl-cols .btn, .slew-fs-btn').forEach(btn => {
    btn.addEventListener('touchstart', e => e.preventDefault(), {passive:false});
    btn.addEventListener('touchend', e => e.preventDefault(), {passive:false});
  });
  // Load auto-connect preference and trigger connection if enabled
  checkAutoConnect();
});

function doGoto() {
  const ra = document.getElementById('goto-ra').value.trim();
  const dec = document.getElementById('goto-dec').value.trim();
  const search = document.getElementById('goto-search').value.trim();
  const body = ra && dec ? { ra, dec } : { target: search };
  apiPost('/api/goto', body).then(r => {
    const el = document.getElementById('goto-status');
    if (r && r.ok) {
      el.textContent = 'GoTo sent: RA=' + (r.ra||'') + ' Dec=' + (r.dec||'');
      el.style.color = 'var(--green)';
      // Immediately show target in position card
      const tRow = document.getElementById('target-row');
      if (tRow) {
        tRow.style.display = '';
        setText('p-target-name', r.target || search || ('RA ' + (r.ra||'')));
        setText('p-target-coords', 'RA ' + (r.ra||'') + '  Dec ' + (r.dec||''));
        const tBadge = document.getElementById('p-target-badge');
        if (tBadge) { tBadge.textContent = 'Slewing'; tBadge.className = 'badge badge-yellow'; }
      }
    } else {
      el.textContent = r ? r.error : 'Request failed';
      el.style.color = 'var(--red)';
    }
  });
}

// ============================================================
// Catalog search
// ============================================================
function catalogSearch(q) {
  const box = document.getElementById('search-results');
  if (q.length < 1) { box.style.display = 'none'; return; }
  clearTimeout(searchTimeout);
  searchTimeout = setTimeout(async () => {
    const d = await apiGet('/api/catalog/search?q=' + encodeURIComponent(q));
    if (!d || !d.results.length) { box.style.display = 'none'; return; }
    box.innerHTML = '';
    for (const r of d.results) {
      const div = document.createElement('div');
      div.className = 'sr-item';
      const below = r.alt_deg < 0;
      const vis = below ? '<span style="color:var(--red)">\u2717</span> '
                        : '<span style="color:var(--green)">\u2713</span> ';
      div.innerHTML = `<span class="sr-name">${esc(r.name)}</span>` +
        `<span class="sr-coord">${vis}Alt ${r.alt_deg.toFixed(1)}\u00b0 Az ${r.az_deg.toFixed(1)}\u00b0</span>`;
      if (below) div.style.opacity = '0.5';
      div.addEventListener('click', () => {
        document.getElementById('goto-search').value = r.name;
        box.style.display = 'none';
      });
      box.appendChild(div);
    }
    box.style.display = 'block';
  }, 250);
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// Close search results on outside click
document.addEventListener('click', e => {
  if (!e.target.closest('.search-box')) {
    document.getElementById('search-results').style.display = 'none';
  }
});

// ============================================================
// Catalog browse
// ============================================================
let browseCurrentCat = '';
let browseCurrentPage = 1;
let browseTotalPages = 1;

async function loadCatalogCategories() {
  const d = await apiGet('/api/catalog/browse');
  if (!d || !d.categories) return;
  const box = document.getElementById('cat-chips');
  box.innerHTML = '';
  // Filter out "NGC (all)" from chips -- too large for casual browsing
  const cats = d.categories.filter(c => c.name !== 'NGC (all)');
  for (const c of cats) {
    const chip = document.createElement('span');
    chip.className = 'cat-chip';
    chip.dataset.cat = c.name;
    chip.innerHTML = esc(c.name) + ' <span class="chip-count">(' + c.count + ')</span>';
    chip.addEventListener('click', () => browseCategory(c.name));
    box.appendChild(chip);
  }
}

async function browseCategory(name) {
  // Toggle off if same category clicked again
  if (browseCurrentCat === name) {
    browseCurrentCat = '';
    document.getElementById('browse-list').style.display = 'none';
    document.getElementById('browse-pager').style.display = 'none';
    document.querySelectorAll('.cat-chip').forEach(c => c.classList.remove('active'));
    return;
  }
  browseCurrentCat = name;
  browseCurrentPage = 1;
  document.querySelectorAll('.cat-chip').forEach(c => {
    c.classList.toggle('active', c.dataset.cat === name);
  });
  await loadBrowsePage();
}

async function loadBrowsePage() {
  const url = '/api/catalog/browse?cat=' + encodeURIComponent(browseCurrentCat) + '&page=' + browseCurrentPage;
  const d = await apiGet(url);
  if (!d || !d.objects) return;

  const list = document.getElementById('browse-list');
  list.innerHTML = '';

  for (const obj of d.objects) {
    const row = document.createElement('div');
    row.className = 'browse-item';
    const below = obj.alt_deg < 0;
    const vis = below ? '<span style="color:var(--red)">\u2717</span> '
                      : '<span style="color:var(--green)">\u2713</span> ';
    row.innerHTML = '<span class="browse-id">' + esc(obj.id) + '</span>' +
      '<span class="browse-coord">' + vis + 'Alt ' + obj.alt_deg.toFixed(1) + '\u00b0 Az ' + obj.az_deg.toFixed(1) + '\u00b0</span>' +
      '<button class="browse-goto" data-name="' + esc(obj.id) + '">GoTo</button>';
    if (below) row.style.opacity = '0.5';
    row.querySelector('.browse-goto').addEventListener('click', e => {
      e.stopPropagation();
      document.getElementById('goto-search').value = obj.id;
      doGoto();
    });
    // Click row to fill search field
    row.addEventListener('click', () => {
      document.getElementById('goto-search').value = obj.id;
    });
    list.appendChild(row);
  }
  list.style.display = d.objects.length ? 'block' : 'none';

  // Pager
  browseTotalPages = Math.max(1, Math.ceil(d.total / d.per_page));
  const pager = document.getElementById('browse-pager');
  if (browseTotalPages > 1) {
    pager.style.display = 'flex';
    document.getElementById('browse-page-info').textContent = browseCurrentPage + ' / ' + browseTotalPages;
    document.getElementById('browse-prev').disabled = browseCurrentPage <= 1;
    document.getElementById('browse-next').disabled = browseCurrentPage >= browseTotalPages;
  } else {
    pager.style.display = 'none';
  }
}

function browsePage(delta) {
  browseCurrentPage = Math.max(1, Math.min(browseTotalPages, browseCurrentPage + delta));
  loadBrowsePage();
}

// Load categories on startup
loadCatalogCategories();

// ============================================================
// Solver settings + ASTAP database management
// ============================================================
//
// Solver settings include: mode (auto/astap/cloud), API key, and timeout.
// ASTAP database management uses Server-Sent Events (SSE) for real-time
// download progress instead of HTTP polling -- this gives accurate byte-level
// progress updates and reduces unnecessary network requests.

async function loadSolverSettings() {
  // Load solver settings from the backend and populate the UI controls
  const d = await apiGet('/api/solver/settings');
  if (!d) return;
  if (d.cloud_api_key !== undefined)
    document.getElementById('solver-api-key').value = d.cloud_api_key;
  if (d.mode) {
    let mode = d.mode === 'local' ? 'astap' : d.mode;
    document.getElementById('solver-mode').value = mode;
  }
  // Load timeout (default 120s, clamped to slider range)
  if (d.timeout !== undefined) {
    const t = Math.max(10, Math.min(600, d.timeout));
    document.getElementById('solver-timeout-slider').value = Math.min(300, t);
    document.getElementById('solver-timeout-val').value = t;
  }
  // Load optics / FOV settings
  if (d.focal_length_mm !== undefined) {
    document.getElementById('solver-focal-length').value = d.focal_length_mm || '';
  }
  if (d.sensor_width_mm !== undefined) {
    document.getElementById('solver-sensor-width').value = d.sensor_width_mm || '';
  }
  // Show calculated FOV and last solved FOV
  updateFovDisplay(d.calculated_fov, d.last_solved_fov, d.recommended_db);
  refreshAstapDbStatus();
}

function updateFovPreview() {
  // Live-calculate FOV as user edits focal length / sensor width
  const fl = parseFloat(document.getElementById('solver-focal-length').value) || 0;
  const sw = parseFloat(document.getElementById('solver-sensor-width').value) || 0;
  let fov = 0;
  if (fl > 0 && sw > 0) {
    fov = 2 * Math.atan(sw / (2 * fl)) * (180 / Math.PI);
  }
  const fovEl = document.getElementById('solver-fov-value');
  const hintEl = document.getElementById('solver-fov-db-hint');
  if (fov > 0) {
    fovEl.textContent = fov.toFixed(2) + '\u00b0';
    fovEl.style.color = 'var(--accent)';
    // Show recommended database
    let rec = fov >= 20 ? 'W08' : fov >= 0.6 ? 'D05' : fov >= 0.3 ? 'D20' : 'D50';
    hintEl.textContent = '(use ' + rec + ' database)';
    hintEl.style.color = 'var(--yellow)';
  } else {
    fovEl.textContent = '--';
    fovEl.style.color = 'var(--fg)';
    hintEl.textContent = '';
  }
}

// ------ Sensor Width Calculator (non-ZWO cameras) ------
function calcSensorWidth() {
  var resX = parseFloat(document.getElementById('calc-res-x').value) || 0;
  var pxUm = parseFloat(document.getElementById('calc-pixel-um').value) || 0;
  var el = document.getElementById('calc-result');
  if (resX > 0 && pxUm > 0) {
    var sw = (resX * pxUm / 1000).toFixed(2);
    el.innerHTML = 'Sensor width = ' + resX + ' &times; ' + pxUm + ' / 1000 = <span style="color:var(--accent);font-weight:600">' + sw + ' mm</span>';
  } else {
    el.textContent = '';
  }
}
function applySensorCalc() {
  var resX = parseFloat(document.getElementById('calc-res-x').value) || 0;
  var pxUm = parseFloat(document.getElementById('calc-pixel-um').value) || 0;
  if (resX <= 0 || pxUm <= 0) { toast('Enter both resolution and pixel size', 'warn'); return; }
  var sw = (resX * pxUm / 1000).toFixed(2);
  document.getElementById('solver-sensor-width').value = sw;
  updateFovPreview();
  toast('Sensor width set to ' + sw + ' mm', 'ok');
}

function updateFovDisplay(calcFov, solvedFov, recDb) {
  // Update FOV preview with server-calculated values
  const fovEl = document.getElementById('solver-fov-value');
  const hintEl = document.getElementById('solver-fov-db-hint');
  const solvedEl = document.getElementById('solver-fov-solved-val');
  if (calcFov && calcFov > 0) {
    fovEl.textContent = calcFov.toFixed(2) + '\u00b0';
    fovEl.style.color = 'var(--accent)';
    if (recDb) {
      hintEl.textContent = '(use ' + recDb.toUpperCase() + ' database)';
      hintEl.style.color = 'var(--yellow)';
    }
  } else {
    fovEl.textContent = 'Not configured';
    fovEl.style.color = 'var(--dim)';
    hintEl.textContent = '';
  }
  if (solvedFov && solvedFov > 0) {
    solvedEl.textContent = solvedFov.toFixed(2) + '\u00b0';
    solvedEl.style.color = 'var(--green)';
    // Warn if configured and solved FOV differ significantly
    if (calcFov && calcFov > 0) {
      const ratio = solvedFov / calcFov;
      if (ratio < 0.5 || ratio > 2.0) {
        solvedEl.textContent += ' \u26a0 Mismatch!';
        solvedEl.style.color = 'var(--red)';
      }
    }
  } else {
    solvedEl.textContent = 'No solve yet';
    solvedEl.style.color = 'var(--dim)';
  }
}

async function saveSolverSettings() {
  const key = document.getElementById('solver-api-key').value.trim();
  const mode = document.getElementById('solver-mode').value;
  const timeout = parseInt(document.getElementById('solver-timeout-val').value) || 120;
  const fl = parseFloat(document.getElementById('solver-focal-length').value) || 0;
  const sw = parseFloat(document.getElementById('solver-sensor-width').value) || 0;
  const r = await apiPost('/api/solver/settings', {
    cloud_api_key: key, mode: mode, timeout: timeout,
    focal_length_mm: fl, sensor_width_mm: sw
  });
  const el = document.getElementById('solver-status');
  if (r && r.ok) {
    const labels = {auto:'Auto (ASTAP + Cloud)', cloud:'Astrometry.net', astap:'ASTAP Local'};
    let msg = (labels[mode]||mode) + ' mode saved (timeout: ' + timeout + 's)';
    if (r.calculated_fov && r.calculated_fov > 0) {
      msg += ' | FOV: ' + r.calculated_fov.toFixed(2) + '\u00b0';
    }
    el.textContent = msg;
    el.style.color = 'var(--green)';
  } else {
    el.textContent = r ? r.error : 'Failed';
    el.style.color = 'var(--red)';
  }
}

// -- ASTAP database status --
async function refreshAstapDbStatus() {
  const d = await apiGet('/api/solver/databases');
  const statusEl = document.getElementById('astap-db-status');
  const dlBtn = document.getElementById('astap-db-download-btn');
  const delBtn = document.getElementById('astap-db-delete-btn');
  if (!d || !d.ok) {
    if (statusEl) statusEl.textContent = 'Could not check database status';
    return;
  }
  if (d.installed) {
    const db = d.databases[d.installed];
    statusEl.innerHTML = '<span style="color:var(--green)">&#10003;</span> <b>' +
      d.installed.toUpperCase() + '</b> installed' +
      (db ? ' (' + db.size_mb + ' MB)' : '');
    if (dlBtn) dlBtn.textContent = 'Download Another';
    if (delBtn) { delBtn.style.display = ''; delBtn.dataset.db = d.installed; }
  } else {
    statusEl.innerHTML = '<span style="color:var(--yellow)">&#9888;</span> ' +
      'No star database installed. Download one to enable ASTAP local solving.';
    if (dlBtn) dlBtn.textContent = 'Download Database';
    if (delBtn) delBtn.style.display = 'none';
  }
}

// -- ASTAP database download with SSE progress --
// Uses Server-Sent Events (EventSource) for real-time progress updates
// instead of polling.  The SSE endpoint (/api/solver/databases/progress)
// pushes JSON events every 500ms with byte-level download progress and
// extraction file counts.
let _sseSource = null;  // Active EventSource connection (if any)

async function downloadAstapDb() {
  const sel = document.getElementById('astap-db-select');
  const dbName = sel ? sel.value : 'd05';
  const progressDiv = document.getElementById('astap-db-progress');
  const progressBar = document.getElementById('astap-db-progress-bar');
  const progressText = document.getElementById('astap-db-progress-text');
  const dlBtn = document.getElementById('astap-db-download-btn');

  // Show progress UI
  if (progressDiv) progressDiv.style.display = '';
  if (progressBar) progressBar.style.width = '0%';
  if (progressText) progressText.textContent = 'Starting download of ' + dbName.toUpperCase() + '...';
  if (dlBtn) { dlBtn.disabled = true; dlBtn.textContent = 'Downloading...'; }

  // Start the download (non-blocking -- returns immediately)
  const r = await apiPost('/api/solver/databases/download', { db_name: dbName });
  if (!r || !r.ok) {
    if (progressText) progressText.textContent = (r && r.error) || 'Download failed';
    if (dlBtn) { dlBtn.disabled = false; dlBtn.textContent = 'Download Database'; }
    return;
  }

  // Connect to the SSE progress stream
  // Close any previous SSE connection first
  if (_sseSource) { _sseSource.close(); _sseSource = null; }
  _sseSource = new EventSource('/api/solver/databases/progress');

  _sseSource.onmessage = function(event) {
    try {
      const data = JSON.parse(event.data);

      if (data.state === 'downloading') {
        // Show real byte-level progress
        if (data.bytes_total > 0) {
          const pct = (data.bytes_downloaded / data.bytes_total * 100).toFixed(1);
          if (progressBar) progressBar.style.width = pct + '%';
          const dlMB = (data.bytes_downloaded / 1048576).toFixed(1);
          const totMB = (data.bytes_total / 1048576).toFixed(1);
          if (progressText) progressText.textContent = 'Downloading ' + dbName.toUpperCase() + ': ' + dlMB + ' / ' + totMB + ' MB (' + pct + '%)';
        } else {
          const dlMB = (data.bytes_downloaded / 1048576).toFixed(1);
          if (progressText) progressText.textContent = 'Downloading ' + dbName.toUpperCase() + ': ' + dlMB + ' MB...';
          if (progressBar) progressBar.style.width = '50%';
        }

      } else if (data.state === 'extracting') {
        if (progressBar) progressBar.style.width = '90%';
        if (progressText) progressText.textContent = 'Extracting star database files... (' + data.extracted_files + ' files)';

      } else if (data.state === 'complete') {
        if (progressBar) progressBar.style.width = '100%';
        if (progressText) progressText.textContent = dbName.toUpperCase() + ' installed successfully!';
        if (dlBtn) { dlBtn.disabled = false; dlBtn.textContent = 'Download Another'; }
        if (_sseSource) { _sseSource.close(); _sseSource = null; }
        refreshAstapDbStatus();

      } else if (data.state === 'error') {
        if (progressText) {
          progressText.textContent = 'Error: ' + (data.error || 'Unknown error');
          progressText.style.color = 'var(--red)';
        }
        if (dlBtn) { dlBtn.disabled = false; dlBtn.textContent = 'Download Database'; }
        if (_sseSource) { _sseSource.close(); _sseSource = null; }

      } else if (data.state === 'idle') {
        // Download may not have started yet, wait a moment
        if (progressText) progressText.textContent = 'Waiting for download to start...';
      }
    } catch (e) {
      console.warn('SSE parse error:', e);
    }
  };

  _sseSource.onerror = function() {
    // SSE connection lost -- fall back to simple polling
    if (_sseSource) { _sseSource.close(); _sseSource = null; }
    if (progressText) progressText.textContent = 'Progress stream lost. Checking status...';
    // Fall back: check database status after a delay
    setTimeout(async () => {
      const s = await apiGet('/api/solver/databases');
      if (s && s.installed) {
        if (progressText) progressText.textContent = s.installed.toUpperCase() + ' installed!';
        if (progressBar) progressBar.style.width = '100%';
        if (dlBtn) { dlBtn.disabled = false; dlBtn.textContent = 'Download Another'; }
        refreshAstapDbStatus();
      } else {
        if (progressText) progressText.textContent = 'Download may still be running. Click Refresh to check.';
        if (dlBtn) { dlBtn.disabled = false; dlBtn.textContent = 'Download Database'; }
      }
    }, 3000);
  };
}

async function deleteAstapDb() {
  const delBtn = document.getElementById('astap-db-delete-btn');
  const dbName = delBtn ? delBtn.dataset.db : '';
  if (!dbName) return;
  if (!confirm('Delete ' + dbName.toUpperCase() + ' star database? You can re-download it later.')) return;
  const r = await apiPost('/api/solver/databases/delete', { db_name: dbName });
  if (r && r.ok) refreshAstapDbStatus();
}

loadSolverSettings();

// ============================================================
// Star Alignment UI (Auto + Manual modes)
// ============================================================
let alignSelectedStars = 6;
let alignSelectedMode = 'auto';   // 'auto' or 'manual'

// Star count buttons (only bind to the #align-star-btns container)
document.querySelectorAll('#align-star-btns .align-star-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#align-star-btns .align-star-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    alignSelectedStars = parseInt(btn.dataset.n);
  });
});

function setAlignMode(mode) {
  alignSelectedMode = mode;
  document.querySelectorAll('#align-mode-btns .align-star-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.mode === mode);
  });
  const desc = document.getElementById('align-mode-desc');
  if (desc) {
    desc.textContent = mode === 'manual'
      ? 'Manual: telescope slews to each star. You center it visually, then press Sync.'
      : 'Automatic: plate-solver centers each star and syncs automatically.';
  }
}

function doAlignStart() {
  apiPost('/api/alignment/start', { num_stars: alignSelectedStars, mode: alignSelectedMode });
}
function doAlignAbort() {
  apiPost('/api/alignment/abort');
}

// Manual-mode user actions
function doAlignManualSync() {
  apiPost('/api/alignment/manual/sync');
}
function doAlignManualRecenter() {
  // Open the alignment slew cross overlay for fine manual centering
  openAlignSlewOverlay();
}
function doAlignManualSkip() {
  apiPost('/api/alignment/manual/skip');
}

/* ============================================================
   Alignment Slew Overlay - manual centering cross for alignment
   ============================================================ */
let _alignSlewSpeed = 3;

function openAlignSlewOverlay() {
  const overlay = document.getElementById('align-slew-overlay');
  if (!overlay) return;
  // Populate star info from the manual panel
  const starName = document.getElementById('align-manual-star');
  const starCoords = document.getElementById('align-manual-coords');
  const starAltaz = document.getElementById('align-manual-altaz');
  if (starName) {
    document.getElementById('align-slew-star-name').textContent = starName.textContent || '--';
  }
  if (starCoords && starAltaz) {
    document.getElementById('align-slew-star-coords').textContent =
      (starCoords.textContent || '') + '  |  ' + (starAltaz.textContent || '');
  }
  overlay.classList.add('active');
  updateAlignSlewSpeedBtns();
  if (navigator.vibrate) navigator.vibrate(50);
}

function closeAlignSlewOverlay() {
  const overlay = document.getElementById('align-slew-overlay');
  if (!overlay) return;
  overlay.classList.remove('active');
  // Stop any ongoing slew when closing
  apiPost('/api/slew/stop');
}

function alignSlewSync() {
  // User confirmed star is centered via the overlay SYNC button
  if (navigator.vibrate) navigator.vibrate([40, 30, 40]);
  // Send sync to backend, then close overlay
  apiPost('/api/alignment/manual/sync');
  closeAlignSlewOverlay();
}

function alignSlewSkip() {
  // User wants to skip this star from the overlay
  if (navigator.vibrate) navigator.vibrate(30);
  apiPost('/api/alignment/manual/skip');
  closeAlignSlewOverlay();
}

function alignSlewDir(dir, e) {
  if (e) e.preventDefault();
  if (navigator.vibrate) navigator.vibrate(25);
  apiPost('/api/slew', { direction: dir, speed: _alignSlewSpeed });
}

function alignSlewStop(e) {
  if (e) e.preventDefault();
  apiPost('/api/slew/stop');
}

function setAlignSlewSpeed(s) {
  _alignSlewSpeed = s;
  updateAlignSlewSpeedBtns();
  if (navigator.vibrate) navigator.vibrate(20);
}

function updateAlignSlewSpeedBtns() {
  document.querySelectorAll('#align-slew-speed .as-spd-btn').forEach(function(b) {
    b.classList.toggle('active', parseInt(b.dataset.speed) === _alignSlewSpeed);
  });
}

async function pollAlignment() {
  const s = await apiGet('/api/alignment/status');
  if (!s) return;

  const running = s.running || false;
  const phase = s.phase || 'idle';
  const mode = s.mode || 'auto';
  const waitingForUser = s.waiting_for_user || false;

  document.getElementById('btn-align-start').disabled = running;
  document.getElementById('btn-align-abort').disabled = !running;
  // Disable star count + mode buttons while running
  document.querySelectorAll('#align-star-btns .align-star-btn').forEach(b => {
    b.style.pointerEvents = running ? 'none' : 'auto';
    b.style.opacity = running ? '0.5' : '1';
  });
  document.querySelectorAll('#align-mode-btns .align-star-btn').forEach(b => {
    b.style.pointerEvents = running ? 'none' : 'auto';
    b.style.opacity = running ? '0.5' : '1';
  });

  const progDiv = document.getElementById('align-progress');
  const listDiv = document.getElementById('align-stars-list');
  const manualPanel = document.getElementById('align-manual-panel');
  const errorRow = document.getElementById('align-error-row');

  if (phase === 'idle' || phase === 'unavailable') {
    progDiv.style.display = 'none';
    listDiv.style.display = 'none';
    if (manualPanel) manualPanel.style.display = 'none';
    // Auto-close the alignment slew overlay if open
    var _aso = document.getElementById('align-slew-overlay');
    if (_aso && _aso.classList.contains('active')) _aso.classList.remove('active');
    return;
  }

  progDiv.style.display = 'block';

  // Show mode in phase label
  const phaseLabel = mode === 'manual' ? phase + ' (manual)' : phase;
  setText('align-phase', phaseLabel);
  setText('align-star-num', (s.stars_completed || 0) + (running && s.current_star_name ? 1 : 0));
  setText('align-star-total', s.stars_total || 0);

  // Step display -- show user-friendly text for manual waiting
  const step = s.current_step || '--';
  const stepLabel = (step === 'waiting_for_user')
    ? 'Waiting for you...'
    : step;
  setText('align-step', stepLabel);
  setText('align-attempt', s.attempt || 0);
  setText('align-max-attempt', s.max_attempts || 5);

  // In manual mode the plate-solve error is not applicable
  if (errorRow) {
    if (mode === 'manual') {
      errorRow.style.display = 'none';
    } else {
      errorRow.style.display = '';
      setText('align-error', s.last_solve_error || '--');
    }
  }
  setText('align-msg', s.message || '--');

  // Manual panel: show/hide based on waiting_for_user
  if (manualPanel) {
    if (mode === 'manual' && waitingForUser) {
      manualPanel.style.display = 'block';
      manualPanel.classList.add('align-manual-panel-waiting');
      setText('align-manual-star', s.current_star_name || '--');
      const coords = (s.manual_target_ra || '--') + ' / ' + (s.manual_target_dec || '--');
      setText('align-manual-coords', '  RA ' + coords);
      setText('align-manual-altaz',
        'Alt ' + (s.manual_target_alt || '--') + '\u00B0  Az ' + (s.manual_target_az || '--') + '\u00B0');
    } else {
      manualPanel.style.display = 'none';
      manualPanel.classList.remove('align-manual-panel-waiting');
      // Auto-close the alignment slew overlay when no longer waiting for user
      var _aso2 = document.getElementById('align-slew-overlay');
      if (_aso2 && _aso2.classList.contains('active')) _aso2.classList.remove('active');
    }
  }

  // Star list
  const stars = s.star_list || [];
  if (stars.length > 0) {
    listDiv.style.display = 'block';
    listDiv.innerHTML = '';
    for (let i = 0; i < stars.length; i++) {
      const st = stars[i];
      let dotClass;
      if (st.status === 'done') {
        dotClass = 'align-dot-done';
      } else if (st.status === 'failed') {
        dotClass = 'align-dot-failed';
      } else if (running && i === (s.current_star_index || 0) && st.status === 'pending') {
        dotClass = waitingForUser ? 'align-dot-waiting' : 'align-dot-active';
      } else {
        dotClass = 'align-dot-pending';
      }
      const row = document.createElement('div');
      row.className = 'align-star-row';
      row.innerHTML = '<span class="align-star-dot ' + dotClass + '"></span>' +
        '<span class="align-star-name">' + esc(st.name) + '</span>' +
        '<span class="align-star-info">mag ' + st.mag + ' | Alt ' + st.alt + ' Az ' + st.az + '</span>';
      listDiv.appendChild(row);
    }
  } else {
    listDiv.style.display = 'none';
  }
}

// ============================================================
// Camera live view (supports UVC + ASCOM sources)
// ============================================================
let camStreaming = false;

function onCamSourceChange() {
  const src = document.getElementById('lv-cam-source').value;
  // Desktop: show UVC cam# field only for uvc source
  const uvcF = document.getElementById('cam-uvc-fields');
  if (uvcF) uvcF.style.display = (src === 'uvc' && !window._isAndroid) ? 'block' : 'none';
  const ascomC = document.getElementById('cam-ascom-fields');
  if (ascomC) ascomC.style.display = src === 'ascom' ? 'block' : 'none';
  // Show ASI controls hint (actual controls appear after camera starts)
  const asiC = document.getElementById('cam-asi-controls');
  if (asiC && src !== 'asi' && src !== 'zwo' && src !== 'auto') asiC.style.display = 'none';
}

async function camRefreshAscom() {
  const sel = document.getElementById('lv-ascom-id');
  if (!sel) return;  // Element removed on Android (ASCOM is desktop-only)
  sel.innerHTML = '<option value="">Loading...</option>';
  try {
    const resp = await fetch('/api/ascom/cameras', {headers: _authHeaders()});
    const data = await resp.json();
    sel.innerHTML = '<option value="">-- select ASCOM camera --</option>';
    if (data.ok && data.cameras) {
      for (const c of data.cameras) {
        const opt = document.createElement('option');
        opt.value = c.id;
        opt.textContent = c.name || c.id;
        sel.appendChild(opt);
      }
    }
  } catch (e) {
    sel.innerHTML = '<option value="">Error loading cameras</option>';
  }
}

async function camStart() {
  const source = document.getElementById('lv-cam-source').value;
  let body, label;

  if (source === 'ascom') {
    const ascomId = document.getElementById('lv-ascom-id').value;
    if (!ascomId) {
      document.getElementById('cam-status').textContent = 'Select an ASCOM camera first';
      document.getElementById('cam-status').style.color = 'var(--red)';
      return;
    }
    const exp = parseFloat(document.getElementById('lv-ascom-exp').value) || 0.5;
    const gain = parseInt(document.getElementById('lv-ascom-gain').value) || 100;
    const bin = parseInt(document.getElementById('lv-ascom-bin').value) || 2;
    body = { source: 'ascom', ascom_id: ascomId, exposure: exp, gain: gain, binning: bin };
    label = ascomId.split('.').pop() || 'ASCOM';
  } else if (source === 'asi') {
    // Desktop/RPi ZWO ASI SDK camera
    const idx = parseInt(document.getElementById('lv-cam-idx').value) || 0;
    body = { source: 'asi', camera_index: idx };
    label = 'ZWO ASI (SDK)';
  } else if (['auto', 'zwo', 'phone'].includes(source)) {
    // Android camera sources
    body = { source: 'uvc', android_source: source };
    const names = {auto: 'Auto', zwo: 'ZWO ASI', phone: 'Phone'};
    label = names[source] || source;
  } else {
    const idx = parseInt(document.getElementById('lv-cam-idx').value) || 0;
    body = { source: 'uvc', camera_index: idx };
    label = 'UVC #' + idx;
  }

  document.getElementById('cam-status').textContent = 'Connecting...';
  document.getElementById('cam-status').style.color = 'var(--dim)';
  const r = await apiPost('/api/camera/start', body);

  if (r && r.ok) {
    camStreaming = true;
    document.getElementById('cam-placeholder').style.display = 'none';
    document.getElementById('cam-container').style.display = 'block';
    // Append timestamp to bust cache
    document.getElementById('cam-img').src = '/api/camera/stream?' + Date.now();
    document.getElementById('btn-cam-start').disabled = true;
    document.getElementById('btn-cam-stop').disabled = false;
    document.getElementById('cam-status').textContent = 'Streaming from ' + label;
    document.getElementById('cam-status').style.color = 'var(--green)';
    // Show ASI controls if this is a ZWO/ASI source (Android or desktop)
    if (['auto', 'zwo', 'asi'].includes(source)) {
      setTimeout(() => camShowAsiControls(true), 1500);
    } else {
      camShowAsiControls(false);
    }
    // Auto-detect phone camera sensor dimensions for plate solving
    if (source === 'phone') {
      setTimeout(async function() {
        try {
          var ps = await apiGet('/api/camera/phone/sensor');
          if (ps && ps.ok && ps.width_mm > 0) {
            var swEl = document.getElementById('solver-sensor-width');
            if (swEl && (!swEl.value || parseFloat(swEl.value) === 0)) {
              swEl.value = ps.width_mm.toFixed(2);
              updateFovPreview();
              toast('Phone sensor: ' + ps.width_mm.toFixed(2) + 'x' + ps.height_mm.toFixed(2)
                + ' mm (' + ps.resolution_x + 'x' + ps.resolution_y + 'px)', 'ok');
            }
          }
        } catch(e) { console.warn('Phone sensor detection failed:', e); }
      }, 500);
    }
  } else {
    document.getElementById('cam-status').textContent = (r && r.error) || 'Failed to open camera';
    document.getElementById('cam-status').style.color = 'var(--red)';
  }
}

async function camStop() {
  await apiPost('/api/camera/stop');
  camStreaming = false;
  camShowAsiControls(false);
  document.getElementById('cam-img').src = '';
  document.getElementById('cam-container').style.display = 'none';
  document.getElementById('cam-placeholder').style.display = 'flex';
  document.getElementById('btn-cam-start').disabled = false;
  document.getElementById('btn-cam-stop').disabled = true;
  document.getElementById('cam-status').textContent = 'Camera stopped';
  document.getElementById('cam-status').style.color = 'var(--dim)';
  // Exit fullscreen if active
  const ctr = document.getElementById('cam-container');
  ctr.classList.remove('cam-fullscreen');
}

async function camApplyAscomSettings() {
  const exp = parseFloat(document.getElementById('lv-ascom-exp').value) || 0.5;
  const gain = parseInt(document.getElementById('lv-ascom-gain').value) || 100;
  const bin = parseInt(document.getElementById('lv-ascom-bin').value) || 2;
  await apiPost('/api/camera/ascom/settings', { exposure: exp, gain: gain, binning: bin });
}

// ASI SDK camera controls
async function camApplyAsiSettings() {
  const expMs = parseFloat(document.getElementById('asi-exp-val').value) || 100;
  const gain = parseInt(document.getElementById('asi-gain-slider').value) || 0;
  const gamma = parseInt(document.getElementById('asi-gamma-slider').value) || 50;
  const offset = parseInt(document.getElementById('asi-offset-slider').value) || 0;
  const flip = parseInt(document.getElementById('asi-flip').value) || 0;
  const r = await apiPost('/api/camera/asi/settings', {
    exposure_ms: expMs, gain: gain, gamma: gamma, offset: offset, flip: flip
  });
  if (r && r.ok) {
    document.getElementById('asi-info').textContent =
      'Applied: ' + expMs + 'ms, gain=' + gain + ', gamma=' + gamma;
    document.getElementById('asi-info').style.color = 'var(--green)';
  } else {
    document.getElementById('asi-info').textContent = (r && r.error) || 'Failed';
    document.getElementById('asi-info').style.color = 'var(--red)';
  }
}

async function camReadAsiSettings() {
  const r = await apiGet('/api/camera/asi/status');
  if (r && r.ok && r.controls) {
    const c = r.controls;
    if (c.exposure_ms !== undefined) {
      document.getElementById('asi-exp-val').value = Math.round(c.exposure_ms);
      document.getElementById('asi-exp-slider').value = Math.min(Math.round(c.exposure_ms), 4000);
    }
    if (c.gain !== undefined && c.gain >= 0) {
      document.getElementById('asi-gain-slider').value = c.gain;
      document.getElementById('asi-gain-val').textContent = c.gain;
    }
    if (c.gamma !== undefined && c.gamma >= 0) {
      document.getElementById('asi-gamma-slider').value = c.gamma;
      document.getElementById('asi-gamma-val').textContent = c.gamma;
    }
    if (c.offset !== undefined && c.offset >= 0) {
      document.getElementById('asi-offset-slider').value = c.offset;
      document.getElementById('asi-offset-val').textContent = c.offset;
    }
    if (c.flip !== undefined && c.flip >= 0) {
      document.getElementById('asi-flip').value = c.flip;
    }
    let infoText = '';
    if (r.info && r.info.name) infoText += r.info.name;
    if (c.temperature_c !== undefined) infoText += ' | ' + c.temperature_c.toFixed(1) + ' C';
    if (r.info && r.info.sdk_version) infoText += ' | ' + r.info.sdk_version;
    // Show sensor dimensions in the info line
    if (r.info && r.info.sensor_width_mm > 0) {
      infoText += ' | ' + r.info.sensor_width_mm.toFixed(2) + 'x'
        + r.info.sensor_height_mm.toFixed(2) + 'mm';
    }
    document.getElementById('asi-info').textContent = infoText || 'ASI camera connected';
    document.getElementById('asi-info').style.color = 'var(--dim)';

    // Auto-populate solver sensor width from ASI camera (if not already set)
    if (r.info && r.info.sensor_width_mm > 0) {
      const swEl = document.getElementById('solver-sensor-width');
      if (swEl && (!swEl.value || parseFloat(swEl.value) === 0)) {
        swEl.value = r.info.sensor_width_mm.toFixed(2);
        updateFovPreview();
        toast('Sensor width auto-detected: ' + r.info.sensor_width_mm.toFixed(2) + ' mm ('
          + r.info.pixel_size_um.toFixed(2) + ' um/px, '
          + r.info.max_width + 'x' + r.info.max_height + ')', 'ok');
      }
    }
  }
}

function camShowAsiControls(show) {
  document.getElementById('cam-asi-controls').style.display = show ? 'block' : 'none';
  if (show) camReadAsiSettings();
}

// Auto-apply ASI settings on slider change (debounced)
let _asiApplyTimer = null;
function asiAutoApply() {
  if (_asiApplyTimer) clearTimeout(_asiApplyTimer);
  _asiApplyTimer = setTimeout(() => camApplyAsiSettings(), 300);
}
// Attach auto-apply to sliders after DOM load
document.addEventListener('DOMContentLoaded', () => {
  ['asi-exp-slider','asi-gain-slider','asi-gamma-slider','asi-offset-slider','asi-flip'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', asiAutoApply);
  });
  // Also auto-apply when exposure number input changes
  const expVal = document.getElementById('asi-exp-val');
  if (expVal) expVal.addEventListener('change', asiAutoApply);
});

function camFullscreen() {
  const ctr = document.getElementById('cam-container');
  if (ctr.classList.contains('cam-fullscreen')) {
    ctr.classList.remove('cam-fullscreen');
    // Try exiting browser fullscreen API too
    if (document.exitFullscreen) document.exitFullscreen().catch(()=>{});
  } else {
    ctr.classList.add('cam-fullscreen');
    // Request browser fullscreen for true immersive mode
    if (ctr.requestFullscreen) ctr.requestFullscreen().catch(()=>{});
    else if (ctr.webkitRequestFullscreen) ctr.webkitRequestFullscreen();
  }
}

// Exit camera fullscreen on Escape or when browser exits fullscreen
document.addEventListener('fullscreenchange', () => {
  if (!document.fullscreenElement) {
    document.getElementById('cam-container').classList.remove('cam-fullscreen');
  }
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    document.getElementById('cam-container').classList.remove('cam-fullscreen');
  }
});

function camSnapshot() {
  // Download current frame as JPEG.
  // Use an iframe to trigger the download -- works in Android WebView
  // (DownloadListener intercepts it) and in desktop browsers.
  const iframe = document.createElement('iframe');
  iframe.style.display = 'none';
  iframe.src = '/api/camera/snapshot?' + Date.now();
  document.body.appendChild(iframe);
  setTimeout(() => { document.body.removeChild(iframe); }, 10000);
  toast('Saving snapshot...', 'ok');
}

// Sync focuser state into camera tab too
function updateCamFocuser(ctrl) {
  setText('cam-focus-pos', ctrl.focuser_position || '--');
  setText('cam-focus-status', ctrl.focuser_status || '--');
}

// Auto-load ASCOM camera list when switching to ASCOM source
// (also load on page init in case user was previously using ASCOM)
setTimeout(camRefreshAscom, 2000);

// ============================================================
// Toolbar collapse/expand
// ============================================================
function toggleToolbar() {
  const items = document.getElementById('toolbar-items');
  const btn = document.getElementById('toolbar-toggle');
  if (!items || !btn) return;
  const isOpen = items.classList.toggle('open');
  btn.classList.toggle('open', isOpen);
}

// ============================================================
// Starfield + Theme state
// ============================================================
let _starfieldEnabled = true;
let _starfieldAnimId = null;

// ============================================================
// Light / Dark theme toggle
// ============================================================
let lightTheme = false;
function toggleLightTheme() {
  lightTheme = !lightTheme;
  document.documentElement.classList.toggle('lightmode', lightTheme);
  const btn = document.getElementById('btn-lighttheme');
  if (btn) btn.style.opacity = lightTheme ? '0.4' : '1';
  // Hide/show starfield
  const sf = document.getElementById('starfield');
  if (sf) sf.style.display = lightTheme ? 'none' : '';
  if (!lightTheme && _starfieldEnabled) {
    if (typeof _starfieldStart === 'function') _starfieldStart();
  }
}

// ============================================================
// Starfield background toggle
// ============================================================
function toggleStarfield() {
  _starfieldEnabled = !_starfieldEnabled;
  const canvas = document.getElementById('starfield');
  const btn = document.getElementById('btn-starfield');
  if (_starfieldEnabled) {
    if (canvas) canvas.style.display = '';
    if (btn) btn.style.opacity = '1';
    if (typeof _starfieldStart === 'function') _starfieldStart();
  } else {
    if (canvas) canvas.style.display = 'none';
    if (_starfieldAnimId) { cancelAnimationFrame(_starfieldAnimId); _starfieldAnimId = null; }
    if (btn) btn.style.opacity = '0.4';
  }
}

// ============================================================
// Screen wake lock (prevent screen off during long sessions)
// Works over HTTPS via Wake Lock API, falls back to a hidden
// video trick for plain HTTP (e.g. Pi on LAN).
// ============================================================
let wakeLock = null;
let _wakeLockVideo = null;
const _hasNativeWakeLock = ('wakeLock' in navigator);

function _createWakeLockVideo() {
  // Fallback: a tiny looping video keeps the screen awake on mobile browsers
  // even over plain HTTP.  Uses a minimal base64-encoded mp4.
  const v = document.createElement('video');
  v.setAttribute('playsinline', '');
  v.setAttribute('muted', '');
  v.setAttribute('loop', '');
  v.style.position = 'fixed';
  v.style.top = '-1px';
  v.style.left = '-1px';
  v.style.width = '1px';
  v.style.height = '1px';
  v.style.opacity = '0.01';
  // Smallest valid mp4 (blank 1x1 px, 1s, silent)
  v.src = 'data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAAABNtZGF0AAAA0AAAAG1vb3YAAABsbXZoZAAAAAAAAAAAAAAAAAAAA+gAAAAAAAEAAAEAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAABidWR0YQAAAFptZXRhAAAAAAAAACFoZGxyAAAAAAAAAABtZGlyYXBwbAAAAAAAAAAAAAAAAC1pbHN0AAAAJal0b28AAAAdZGF0YQAAAAEAAAAATGF2YzU4Ljkx';
  document.body.appendChild(v);
  return v;
}

async function toggleWakeLock() {
  const btn = document.getElementById('btn-wakelock');

  // --- Release if currently active ---
  if (wakeLock || (_wakeLockVideo && !_wakeLockVideo.paused)) {
    if (wakeLock) { try { wakeLock.release(); } catch(e){} wakeLock = null; }
    if (_wakeLockVideo) { _wakeLockVideo.pause(); }
    if (btn) { btn.style.background = ''; btn.title = 'Keep screen on'; }
    return;
  }

  // --- Acquire: try native API first, then video fallback ---
  if (_hasNativeWakeLock) {
    try {
      wakeLock = await navigator.wakeLock.request('screen');
      if (btn) { btn.style.background = 'var(--green)'; btn.title = 'Screen lock ON (tap to disable)'; }
      wakeLock.addEventListener('release', () => {
        wakeLock = null;
        if (btn) { btn.style.background = ''; btn.title = 'Keep screen on'; }
      });
      return;
    } catch(e) { /* fall through to video fallback */ }
  }

  // Video fallback (works over HTTP)
  try {
    if (!_wakeLockVideo) _wakeLockVideo = _createWakeLockVideo();
    await _wakeLockVideo.play();
    if (btn) { btn.style.background = 'var(--green)'; btn.title = 'Screen lock ON - video fallback (tap to disable)'; }
  } catch(e) {
    if (btn) btn.title = 'Wake lock not supported in this browser';
  }
}

// ============================================================
// Keyboard shortcuts (arrow keys = slew, Esc = stop)
// ============================================================
const keyToDir = { ArrowUp:'N', ArrowDown:'S', ArrowLeft:'W', ArrowRight:'E' };
const keysHeld = new Set();

document.addEventListener('keydown', e => {
  // Skip if user is typing in an input
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
  const dir = keyToDir[e.key];
  if (dir && !keysHeld.has(e.key)) {
    keysHeld.add(e.key);
    slewStart(dir);
    e.preventDefault();
  }
  if (e.key === ' ') { slewStop(); e.preventDefault(); }
});

document.addEventListener('keyup', e => {
  if (keyToDir[e.key] && keysHeld.has(e.key)) {
    keysHeld.delete(e.key);
    if (keysHeld.size === 0) slewStop();
  }
});

// ============================================================
// Local clock (always visible, updates every second)
// ============================================================
function updateClock() {
  const now = new Date();
  const hh = String(now.getHours()).padStart(2, '0');
  const mm = String(now.getMinutes()).padStart(2, '0');
  const ss = String(now.getSeconds()).padStart(2, '0');
  const timeStr = hh + ':' + mm + ':' + ss;
  const hdrClock = document.getElementById('hdr-clock');
  if (hdrClock) hdrClock.textContent = timeStr;
  const locTime = document.getElementById('loc-time');
  if (locTime) locTime.textContent = timeStr;
  // UTC offset label
  const off = -now.getTimezoneOffset();
  const offH = Math.floor(Math.abs(off) / 60);
  const offM = Math.abs(off) % 60;
  const offStr = 'UTC' + (off >= 0 ? '+' : '-') + String(offH).padStart(2, '0') + ':' + String(offM).padStart(2, '0');
  const locUtc = document.getElementById('loc-utc');
  if (locUtc) locUtc.textContent = offStr;
}
setInterval(updateClock, 1000);
updateClock();

// ============================================================
// Polling loops
// ============================================================
// Status polling (1 Hz for responsive position updates)
setInterval(pollStatus, 1000);
setInterval(pollLog, 1000);
setInterval(pollAlignment, 3000);
// Weather auto-refresh every 5 minutes (300s) -- also pushes to telescope
setInterval(() => fetch('/api/weather/refresh', {method:'POST', headers: _authHeaders()}), 300000);
pollStatus();
pollLog();
refreshSerialPorts();

// ============================================================
// Telemetry Dashboard
// ============================================================
let telemCharts = {};
let telemCorrAxis = 'alt';  // which axis to show in correction chart
let telemPollTimer = null;
let telemTabActive = false;

// Chart.js global config for dark theme
function chartDefaults() {
  if (typeof Chart === 'undefined') return;
  Chart.defaults.color = 'rgba(180,180,210,0.7)';
  Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
  Chart.defaults.font.family = "'Segoe UI',system-ui,sans-serif";
  Chart.defaults.font.size = 11;
  Chart.defaults.animation.duration = 300;
  Chart.defaults.plugins.legend.labels.boxWidth = 10;
  Chart.defaults.plugins.legend.labels.padding = 8;
}

function createLineChart(canvasId, datasets, xLabel, yLabel, opts) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  const defaults = {
    type: 'line',
    data: { labels: [], datasets: datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: datasets.length > 1, position: 'top' },
        tooltip: { enabled: true }
      },
      scales: {
        x: { title: { display: !!xLabel, text: xLabel || '' }, ticks: { maxTicksLimit: 8 } },
        y: { title: { display: !!yLabel, text: yLabel || '' }, ticks: { maxTicksLimit: 6 } }
      },
      elements: { point: { radius: 0, hitRadius: 6 }, line: { tension: 0.3, borderWidth: 1.5 } }
    }
  };
  if (opts) {
    if (opts.stacked) {
      defaults.options.scales.y.stacked = true;
      defaults.data.datasets.forEach(d => { d.fill = opts.fill !== false ? 'origin' : false; });
    }
    if (opts.legend === false) defaults.options.plugins.legend.display = false;
  }
  return new Chart(ctx, defaults);
}

function createBarChart(canvasId, labels, datasetsConfig) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'bar',
    data: { labels: labels, datasets: datasetsConfig },
    options: {
      responsive: true, maintainAspectRatio: false,
      indexAxis: 'y',
      plugins: { legend: { display: datasetsConfig.length > 1, position: 'top' } },
      scales: {
        x: { title: { display: true, text: 'Weight value' }, ticks: { maxTicksLimit: 6 } },
        y: { ticks: { font: { size: 10 } } }
      }
    }
  });
}

function initTelemCharts() {
  if (typeof Chart === 'undefined') return;
  chartDefaults();

  // Tracking Error chart (error_alt, error_az, total_alt, total_az)
  telemCharts.error = createLineChart('chart-error', [
    { label: 'Error Alt', borderColor: '#ff8c00', backgroundColor: 'rgba(255,140,0,0.08)', data: [] },
    { label: 'Error Az', borderColor: '#4a9eff', backgroundColor: 'rgba(74,158,255,0.08)', data: [] },
    { label: 'Total Alt', borderColor: 'rgba(255,140,0,0.4)', borderDash: [4,3], data: [] },
    { label: 'Total Az', borderColor: 'rgba(74,158,255,0.4)', borderDash: [4,3], data: [] }
  ], 'Time (s)', 'arcsec/s');

  // Correction Sources chart (stacked area for one axis at a time)
  telemCharts.corrections = createLineChart('chart-corrections', [
    { label: 'EKF', borderColor: '#ff8c00', backgroundColor: 'rgba(255,140,0,0.15)', data: [], fill: 'origin' },
    { label: 'ML', borderColor: '#4caf50', backgroundColor: 'rgba(76,175,80,0.15)', data: [], fill: 'origin' },
    { label: 'PEC', borderColor: '#ffd700', backgroundColor: 'rgba(255,215,0,0.15)', data: [], fill: 'origin' },
    { label: 'Total', borderColor: '#fff', backgroundColor: 'rgba(255,255,255,0.05)', data: [], fill: false, borderWidth: 2 }
  ], 'Time (s)', 'arcsec/s', { stacked: false });

  // EKF: Measured vs Filtered overlay chart
  telemCharts.kalmanResiduals = createLineChart('chart-kalman-residuals', [
    { label: 'Measured Alt', borderColor: 'rgba(255,140,0,0.4)', backgroundColor: 'rgba(255,140,0,0.06)', data: [], borderWidth: 1, borderDash: [3,3], pointRadius: 2, pointBackgroundColor: '#ff8c00' },
    { label: 'Measured Az', borderColor: 'rgba(74,158,255,0.4)', backgroundColor: 'rgba(74,158,255,0.06)', data: [], borderWidth: 1, borderDash: [3,3], pointRadius: 2, pointBackgroundColor: '#4a9eff' },
    { label: 'EKF Alt', borderColor: '#ff8c00', data: [], borderWidth: 2.5, tension: 0.4 },
    { label: 'EKF Az', borderColor: '#4a9eff', data: [], borderWidth: 2.5, tension: 0.4 }
  ], 'Time (s)', 'arcsec');

  // EKF: Drift velocity over time
  telemCharts.kalmanInnovation = createLineChart('chart-kalman-innovation', [
    { label: 'Drift Alt', borderColor: '#ff8c00', backgroundColor: 'rgba(255,140,0,0.1)', data: [], borderWidth: 2, fill: 'origin' },
    { label: 'Drift Az', borderColor: '#4a9eff', backgroundColor: 'rgba(74,158,255,0.1)', data: [], borderWidth: 2, fill: 'origin' }
  ], 'Time (s)', 'arcsec/s');

  // ML weights bar chart
  const featureNames = ['Bias', 'Alt', 'Az', 'Alt\u00b2', 'Az\u00b2', 'Alt\u00d7Az', 'sin(Az)', 'cos(Az)'];
  telemCharts.mlWeights = createBarChart('chart-ml-weights', featureNames, [
    { label: 'Alt weights', backgroundColor: 'rgba(255,140,0,0.6)', borderColor: '#ff8c00', borderWidth: 1, data: [] },
    { label: 'Az weights', backgroundColor: 'rgba(74,158,255,0.6)', borderColor: '#4a9eff', borderWidth: 1, data: [] }
  ]);

  // PEC correction curve
  telemCharts.pecCurve = createLineChart('chart-pec-curve', [
    { label: 'Alt correction', borderColor: '#ff8c00', backgroundColor: 'rgba(255,140,0,0.08)', data: [] },
    { label: 'Az correction', borderColor: '#4a9eff', backgroundColor: 'rgba(74,158,255,0.08)', data: [] }
  ], 'Time (s)', 'arcsec/s');
}

// Toggle correction chart axis
function toggleCorrAxis(btn) {
  document.querySelectorAll('.telem-axis-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  telemCorrAxis = btn.dataset.axis;
}

// Update chart data helper
function updateChartData(chart, labels, datasets) {
  if (!chart) return;
  chart.data.labels = labels;
  for (let i = 0; i < datasets.length; i++) {
    if (chart.data.datasets[i]) {
      chart.data.datasets[i].data = datasets[i];
    }
  }
  chart.update('none');
}

// Format number for display
function fmtNum(v, digits) {
  if (v === undefined || v === null || v === '--') return '--';
  return typeof v === 'number' ? v.toFixed(digits !== undefined ? digits : 2) : String(v);
}

// ---- Telemetry data fetchers ----

async function fetchTelemGraph() {
  const d = await apiGet('/api/telemetry/graph');
  if (!d || !d.ok) return;

  const ts = d.timestamps || [];
  const labels = ts.map(t => t.toFixed(1));

  // Error chart
  updateChartData(telemCharts.error, labels, [
    d.error_alt || [], d.error_az || [],
    d.total_alt || [], d.total_az || []
  ]);

  // Correction sources chart (axis-dependent)
  const ax = telemCorrAxis;
  updateChartData(telemCharts.corrections, labels, [
    d['kalman_' + ax] || [],
    d['ml_' + ax] || [],
    d['pec_' + ax] || [],
    d['total_' + ax] || []
  ]);
}

async function fetchTelemStats() {
  const d = await apiGet('/api/telemetry/stats');
  if (!d || !d.ok) return;

  setText('tm-total-solves', d.successful_solves !== undefined ? d.successful_solves + '/' + d.total_solves : '--');
  setText('tm-avg-solve', d.avg_solve_time !== undefined ? fmtNum(d.avg_solve_time, 2) + 's' : '--');
  setText('tm-total-corr', d.total_corrections !== undefined ? d.total_corrections : '--');
  setText('tm-avg-corr', d.avg_correction !== undefined ? fmtNum(d.avg_correction * 3600, 2) + '"/s' : '--');
}

async function fetchTelemKalman() {
  const d = await apiGet('/api/telemetry/kalman');
  if (!d || !d.ok) return;

  // --- Status badge ---
  const initEl = document.getElementById('tm-kf-init');
  if (initEl) {
    const samples = d.samples || 0;
    let label, cls;
    if (!d.is_initialized) { label = 'Waiting'; cls = 'red'; }
    else if (samples < 5)  { label = 'Converging'; cls = 'yellow'; }
    else                   { label = 'Active'; cls = 'green'; }
    initEl.textContent = label;
    initEl.className = 'telem-stat-value ' + cls;
  }

  // --- Drift rate (combined magnitude) ---
  setText('tm-kf-drift', d.drift_arcsec !== undefined ? fmtNum(d.drift_arcsec, 2) + '"/s' : '--');
  setText('tm-kf-samples', d.samples || '--');

  // --- Tracking Error stats (RMS + Mean Drift) ---
  setText('tm-rms-alt', d.rms_alt_arcsec !== undefined ? fmtNum(d.rms_alt_arcsec, 2) + '"' : '--');
  setText('tm-rms-az', d.rms_az_arcsec !== undefined ? fmtNum(d.rms_az_arcsec, 2) + '"' : '--');
  setText('tm-drift-alt', d.mean_drift_alt !== undefined ? fmtNum(d.mean_drift_alt * 3600, 3) + '"/s' : '--');
  setText('tm-drift-az', d.mean_drift_az !== undefined ? fmtNum(d.mean_drift_az * 3600, 3) + '"/s' : '--');

  // --- Per-axis drift velocities ---
  if (d.state) {
    setText('tm-kf-valt', fmtNum(d.state.v_alt * 3600, 3) + '"/s');
    setText('tm-kf-vaz', fmtNum(d.state.v_az * 3600, 3) + '"/s');
  }

  // --- Sidereal rates (EKF model output) ---
  if (d.sidereal_rate) {
    setText('tm-kf-sid-alt', fmtNum(d.sidereal_rate.alt, 2) + '"/s');
    setText('tm-kf-sid-az', fmtNum(d.sidereal_rate.az, 2) + '"/s');
  }

  // --- Confidence bar + percentage ---
  const conf = d.confidence !== undefined ? d.confidence : 0;
  setText('tm-kf-confidence', conf >= 80 ? 'High' : conf >= 40 ? 'Medium' : 'Low');
  const confEl = document.getElementById('tm-kf-confidence');
  if (confEl) confEl.className = 'telem-stat-value ' + (conf >= 80 ? 'green' : conf >= 40 ? 'yellow' : 'red');
  const bar = document.getElementById('tm-kf-conf-bar');
  if (bar) bar.style.width = conf.toFixed(0) + '%';
  setText('tm-kf-conf-pct', conf.toFixed(0) + '%');

  // --- Chart 1: Measured vs EKF Estimate ---
  if (d.residual_history && d.residual_history.timestamps) {
    const rh = d.residual_history;
    const labels = rh.timestamps.map(t => t.toFixed(1));
    // Compute position offsets relative to the first point for readability
    const m_alt = rh.measured_alt || [];
    const m_az  = rh.measured_az  || [];
    const f_alt = rh.filtered_alt || [];
    const f_az  = rh.filtered_az  || [];
    const ref_alt = m_alt.length > 0 ? m_alt[0] : 0;
    const ref_az  = m_az.length  > 0 ? m_az[0]  : 0;
    updateChartData(telemCharts.kalmanResiduals, labels, [
      m_alt.map(v => v - ref_alt),   // measured alt offset
      m_az.map(v => v - ref_az),     // measured az offset
      f_alt.map(v => v - ref_alt),   // EKF alt offset
      f_az.map(v => v - ref_az)      // EKF az offset
    ]);
  }

  // --- Chart 2: Drift velocity over time ---
  if (d.residual_history && d.residual_history.timestamps) {
    const rh = d.residual_history;
    const labels = rh.timestamps.map(t => t.toFixed(1));
    updateChartData(telemCharts.kalmanInnovation, labels, [
      rh.velocity_alt || [], rh.velocity_az || []
    ]);
  }
}

async function fetchTelemML() {
  const d = await apiGet('/api/telemetry/ml');
  if (!d || !d.ok) return;

  setText('tm-ml-samples', d.samples || 0);
  const readyEl = document.getElementById('tm-ml-ready');
  if (readyEl) {
    readyEl.textContent = d.model_ready ? 'Yes' : 'No (' + d.samples + '/10)';
    readyEl.className = 'telem-stat-value ' + (d.model_ready ? 'green' : 'yellow');
  }
  setText('tm-ml-error', d.mean_error_arcsec !== undefined ? fmtNum(d.mean_error_arcsec, 3) + '"' : '--');
  setText('tm-ml-preds', d.total_predictions || 0);

  if (d.current_prediction) {
    setText('tm-ml-curalt', fmtNum(d.current_prediction.alt, 3) + '"/s');
    setText('tm-ml-curaz', fmtNum(d.current_prediction.az, 3) + '"/s');
  }

  // Weights bar chart
  if (d.weights_alt && d.weights_az && telemCharts.mlWeights) {
    telemCharts.mlWeights.data.datasets[0].data = d.weights_alt;
    telemCharts.mlWeights.data.datasets[1].data = d.weights_az;
    telemCharts.mlWeights.update('none');
  }

  // Detailed weights grid
  const featureNames = d.feature_names || ['Bias','Alt','Az','Alt\u00b2','Az\u00b2','Alt\u00d7Az','sin(Az)','cos(Az)'];
  const container = document.getElementById('tm-ml-weights-detail');
  if (container && d.weights_alt && d.weights_az) {
    const maxW = Math.max(
      ...d.weights_alt.map(Math.abs),
      ...d.weights_az.map(Math.abs),
      0.001
    );
    let html = '';
    for (let i = 0; i < featureNames.length; i++) {
      const wa = d.weights_alt[i] || 0;
      const wz = d.weights_az[i] || 0;
      const pctA = Math.min(100, (Math.abs(wa) / maxW) * 100);
      const pctZ = Math.min(100, (Math.abs(wz) / maxW) * 100);
      html += '<div class="telem-weight-bar">' +
        '<div class="telem-weight-name">' + featureNames[i] + '</div>' +
        '<div class="telem-weight-vals">' +
          '<span style="color:var(--accent)">Alt: ' + wa.toFixed(4) + '</span>' +
          '<span style="color:var(--blue)">Az: ' + wz.toFixed(4) + '</span>' +
        '</div>' +
        '<div class="telem-bar-track"><div class="telem-bar-fill alt-bar" style="width:' + pctA + '%"></div></div>' +
        '<div class="telem-bar-track"><div class="telem-bar-fill az-bar" style="width:' + pctZ + '%"></div></div>' +
      '</div>';
    }
    container.innerHTML = html;
  }
}

async function fetchTelemPEC() {
  const d = await apiGet('/api/telemetry/pec');
  if (!d || !d.ok) return;

  const trainedEl = document.getElementById('tm-pec-trained');
  if (trainedEl) {
    trainedEl.textContent = d.is_trained ? 'Yes' : 'No';
    trainedEl.className = 'telem-stat-value ' + (d.is_trained ? 'green' : 'red');
  }
  const learnEl = document.getElementById('tm-pec-learning');
  if (learnEl) {
    learnEl.textContent = d.is_learning ? 'Active' : 'Idle';
    learnEl.className = 'telem-stat-value ' + (d.is_learning ? 'yellow' : '');
  }
  setText('tm-pec-samples', d.total_samples || 0);
  setText('tm-pec-span', d.data_span_sec !== undefined ? fmtNum(d.data_span_sec, 1) + 's' : '--');
  setText('tm-pec-rmsalt', d.correction_rms_alt !== undefined ? fmtNum(d.correction_rms_alt, 3) + '"/s' : '--');
  setText('tm-pec-rmsaz', d.correction_rms_az !== undefined ? fmtNum(d.correction_rms_az, 3) + '"/s' : '--');

  // Period details table
  const altP = d.periods_alt_detail || [];
  const azP = d.periods_az_detail || [];
  const allP = altP.map(p => ({...p, axis: 'Alt'})).concat(azP.map(p => ({...p, axis: 'Az'})));

  const table = document.getElementById('tm-pec-periods-table');
  const noData = document.getElementById('tm-pec-no-periods');
  const tbody = document.getElementById('tm-pec-periods-body');

  if (allP.length > 0) {
    table.style.display = 'table';
    noData.style.display = 'none';
    tbody.innerHTML = allP.map(p =>
      '<tr>' +
        '<td style="color:' + (p.axis === 'Alt' ? 'var(--accent)' : 'var(--blue)') + '">' + p.axis + '</td>' +
        '<td>' + fmtNum(p.period_sec, 1) + 's</td>' +
        '<td>' + fmtNum(p.amplitude, 3) + '"</td>' +
        '<td>' + fmtNum(p.snr, 1) + '</td>' +
        '<td>' + (p.harmonics || 0) + '</td>' +
      '</tr>'
    ).join('');
  } else {
    table.style.display = 'none';
    noData.style.display = 'block';
  }

  // PEC correction curve
  if (d.correction_curve) {
    const cc = d.correction_curve;
    const labels = (cc.time || []).map(t => t.toFixed(0));
    updateChartData(telemCharts.pecCurve, labels, [
      cc.alt || [], cc.az || []
    ]);
  }
}

// Main telemetry poll (only when tab is active)
async function pollTelemetry() {
  if (!telemTabActive) return;
  // Fire all requests in parallel
  await Promise.all([
    fetchTelemGraph(),
    fetchTelemStats(),
    fetchTelemKalman(),
    fetchTelemML(),
    fetchTelemPEC()
  ]);
}

// Watch for telemetry tab activation
function onTelemTabChange() {
  const tabEl = document.getElementById('tab-telemetry');
  const wasActive = telemTabActive;
  telemTabActive = tabEl && tabEl.classList.contains('active');

  if (telemTabActive && !wasActive) {
    // Initialize charts on first view
    if (!telemCharts.error && typeof Chart !== 'undefined') {
      initTelemCharts();
    }
    // Start polling (3s interval to avoid Pi overhead)
    pollTelemetry();
    if (!telemPollTimer) {
      telemPollTimer = setInterval(pollTelemetry, 3000);
    }
  }
}

// Hook into tab switching
document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => setTimeout(onTelemTabChange, 50));
});

// ============================================================
// Starfield + Comet animation
// ============================================================
(function(){
  const canvas = document.getElementById('starfield');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H;

  function resize() {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  window.addEventListener('resize', resize);
  resize();

  // --- Stars ---
  const NUM_STARS = 220;
  const stars = [];
  for (let i = 0; i < NUM_STARS; i++) {
    stars.push({
      x: Math.random() * W,
      y: Math.random() * H,
      r: Math.random() * 1.8 + 0.3,
      brightness: Math.random(),
      twinkleSpeed: Math.random() * 0.02 + 0.005,
      twinklePhase: Math.random() * Math.PI * 2
    });
  }

  function drawStars(t) {
    for (const s of stars) {
      const flicker = 0.5 + 0.5 * Math.sin(t * s.twinkleSpeed + s.twinklePhase);
      const alpha = 0.3 + 0.7 * s.brightness * flicker;
      // Color tint -- some stars are warm, some cool
      const hue = s.brightness > 0.7 ? 30 : (s.brightness < 0.3 ? 220 : 0);
      const sat = s.brightness > 0.7 ? '60%' : (s.brightness < 0.3 ? '40%' : '0%');
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fillStyle = hue === 0
        ? `rgba(255,255,255,${alpha})`
        : `hsla(${hue},${sat},85%,${alpha})`;
      ctx.fill();
      // Glow for bright stars
      if (s.r > 1.2) {
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r * 3, 0, Math.PI * 2);
        const g = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, s.r * 3);
        g.addColorStop(0, `rgba(255,255,255,${alpha * 0.15})`);
        g.addColorStop(1, 'rgba(255,255,255,0)');
        ctx.fillStyle = g;
        ctx.fill();
      }
    }
  }

  // --- Comets ---
  const comets = [];
  const MAX_COMETS = 3;

  function spawnComet() {
    if (comets.length >= MAX_COMETS) return;
    // Pick a random entry angle (mostly top-right to bottom-left)
    const side = Math.random();
    let x, y, angle;
    if (side < 0.5) {
      // Enter from top
      x = W * 0.3 + Math.random() * W * 0.7;
      y = -20;
      angle = Math.PI * 0.55 + Math.random() * 0.4; // downward-left
    } else {
      // Enter from right
      x = W + 20;
      y = Math.random() * H * 0.5;
      angle = Math.PI * 0.65 + Math.random() * 0.3;
    }
    const speed = 1.5 + Math.random() * 2.5;
    const size = 2.5 + Math.random() * 3;
    comets.push({
      x, y, angle, speed, size,
      tailLen: 60 + Math.random() * 120,
      life: 0,
      maxLife: 300 + Math.random() * 200,
      hue: Math.random() < 0.4 ? 200 : (Math.random() < 0.5 ? 30 : 180),
      trail: []
    });
  }

  function updateComets() {
    for (let i = comets.length - 1; i >= 0; i--) {
      const c = comets[i];
      c.x += Math.cos(c.angle) * c.speed;
      c.y += Math.sin(c.angle) * c.speed;
      c.life++;
      c.trail.unshift({ x: c.x, y: c.y });
      if (c.trail.length > c.tailLen) c.trail.pop();
      // Remove if off screen or expired
      if (c.life > c.maxLife || c.x < -100 || c.x > W + 100 || c.y > H + 100) {
        comets.splice(i, 1);
      }
    }
  }

  function drawComets() {
    for (const c of comets) {
      const fadeIn = Math.min(1, c.life / 30);
      const fadeOut = Math.min(1, (c.maxLife - c.life) / 40);
      const alpha = fadeIn * fadeOut;
      if (alpha <= 0) continue;

      // Tail (gradient line of particles)
      if (c.trail.length > 1) {
        for (let i = 0; i < c.trail.length - 1; i++) {
          const t = 1 - i / c.trail.length;
          const a = t * t * alpha * 0.6;
          const w = c.size * t;
          ctx.beginPath();
          ctx.moveTo(c.trail[i].x, c.trail[i].y);
          ctx.lineTo(c.trail[i + 1].x, c.trail[i + 1].y);
          ctx.strokeStyle = `hsla(${c.hue},70%,75%,${a})`;
          ctx.lineWidth = w;
          ctx.lineCap = 'round';
          ctx.stroke();
        }
      }

      // Head glow
      const headX = c.x, headY = c.y;
      const g1 = ctx.createRadialGradient(headX, headY, 0, headX, headY, c.size * 6);
      g1.addColorStop(0, `hsla(${c.hue},60%,90%,${alpha * 0.5})`);
      g1.addColorStop(0.3, `hsla(${c.hue},70%,70%,${alpha * 0.2})`);
      g1.addColorStop(1, `hsla(${c.hue},70%,50%,0)`);
      ctx.beginPath();
      ctx.arc(headX, headY, c.size * 6, 0, Math.PI * 2);
      ctx.fillStyle = g1;
      ctx.fill();

      // Bright core
      ctx.beginPath();
      ctx.arc(headX, headY, c.size * 0.8, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255,255,255,${alpha * 0.9})`;
      ctx.fill();
    }
  }

  // --- Nebula background glow ---
  function drawNebula() {
    // Subtle purple/blue nebula patches
    const g1 = ctx.createRadialGradient(W * 0.2, H * 0.3, 0, W * 0.2, H * 0.3, W * 0.4);
    g1.addColorStop(0, 'rgba(40,10,80,0.04)');
    g1.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g1;
    ctx.fillRect(0, 0, W, H);

    const g2 = ctx.createRadialGradient(W * 0.8, H * 0.7, 0, W * 0.8, H * 0.7, W * 0.35);
    g2.addColorStop(0, 'rgba(10,20,60,0.05)');
    g2.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g2;
    ctx.fillRect(0, 0, W, H);
  }

  // --- Animation loop ---
  let frame = 0;
  function animate() {
    if (!_starfieldEnabled) { _starfieldAnimId = null; return; }
    ctx.clearRect(0, 0, W, H);
    drawNebula();
    drawStars(frame);
    updateComets();
    drawComets();

    // Spawn comets randomly
    if (Math.random() < 0.006) spawnComet();

    frame++;
    _starfieldAnimId = requestAnimationFrame(animate);
  }

  // Expose start function so toggleStarfield can restart
  window._starfieldStart = function() {
    if (!_starfieldAnimId && _starfieldEnabled) animate();
  };

  // Start with a comet already visible
  setTimeout(spawnComet, 1000);
  animate();
})();
</script>

<!-- Alignment Slew Overlay - centering cross for manual alignment -->
<div class="align-slew-overlay" id="align-slew-overlay">
  <div class="align-slew-crosshair-h"></div>
  <div class="align-slew-crosshair-v"></div>
  <button class="align-slew-exit" onclick="closeAlignSlewOverlay()" title="Back to alignment">&#10005;</button>
  <div class="align-slew-speed" id="align-slew-speed">
    <button class="as-spd-btn" data-speed="1" onclick="setAlignSlewSpeed(1)">1</button>
    <button class="as-spd-btn" data-speed="2" onclick="setAlignSlewSpeed(2)">2</button>
    <button class="as-spd-btn active" data-speed="3" onclick="setAlignSlewSpeed(3)">3</button>
    <button class="as-spd-btn" data-speed="4" onclick="setAlignSlewSpeed(4)">4</button>
  </div>
  <button class="align-slew-skip" onclick="alignSlewSkip()">Skip Star</button>

  <!-- Header with star info -->
  <div class="align-slew-header">
    <div class="align-slew-star-info">
      <div class="align-slew-star-name" id="align-slew-star-name">--</div>
      <div class="align-slew-star-coords" id="align-slew-star-coords">RA -- / Dec --</div>
      <div class="align-slew-hint">Use arrows to center the star, then press SYNC</div>
    </div>
  </div>

  <!-- 3x3 directional grid -->
  <div class="align-slew-corner"></div>
  <div class="align-slew-btn align-slew-n"
       ontouchstart="alignSlewDir('N',event)" ontouchend="alignSlewStop(event)" ontouchcancel="alignSlewStop(event)"
       onmousedown="alignSlewDir('N',event)" onmouseup="alignSlewStop(event)">
    <span class="fs-arrow">&#9650;</span><span class="fs-label">N</span>
  </div>
  <div class="align-slew-corner"></div>
  <div class="align-slew-btn align-slew-w"
       ontouchstart="alignSlewDir('W',event)" ontouchend="alignSlewStop(event)" ontouchcancel="alignSlewStop(event)"
       onmousedown="alignSlewDir('W',event)" onmouseup="alignSlewStop(event)">
    <span class="fs-arrow">&#9664;</span><span class="fs-label">W</span>
  </div>
  <div class="align-slew-btn align-slew-sync" onclick="alignSlewSync()">
    <span class="fs-arrow">&#10003;</span><span class="fs-label">SYNC</span>
  </div>
  <div class="align-slew-btn align-slew-e"
       ontouchstart="alignSlewDir('E',event)" ontouchend="alignSlewStop(event)" ontouchcancel="alignSlewStop(event)"
       onmousedown="alignSlewDir('E',event)" onmouseup="alignSlewStop(event)">
    <span class="fs-arrow">&#9654;</span><span class="fs-label">E</span>
  </div>
  <div class="align-slew-corner"></div>
  <div class="align-slew-btn align-slew-s"
       ontouchstart="alignSlewDir('S',event)" ontouchend="alignSlewStop(event)" ontouchcancel="alignSlewStop(event)"
       onmousedown="alignSlewDir('S',event)" onmouseup="alignSlewStop(event)">
    <span class="fs-arrow">&#9660;</span><span class="fs-label">S</span>
  </div>
  <div class="align-slew-corner"></div>
</div>

<!-- Fullscreen Slew Overlay - large cross for blind telescope guidance -->
<div class="slew-fs-overlay" id="slew-fs-overlay">
  <div class="slew-fs-crosshair-h"></div>
  <div class="slew-fs-crosshair-v"></div>
  <button class="slew-fs-exit" onclick="toggleFullscreenSlew()" title="Exit fullscreen">&#10005;</button>
  <div class="slew-fs-speed" id="slew-fs-speed">
    <button class="fs-spd-btn" data-speed="1" onclick="setFsSpeed(1)">1</button>
    <button class="fs-spd-btn" data-speed="2" onclick="setFsSpeed(2)">2</button>
    <button class="fs-spd-btn" data-speed="3" onclick="setFsSpeed(3)">3</button>
    <button class="fs-spd-btn active" data-speed="4" onclick="setFsSpeed(4)">4</button>
  </div>
  <div class="slew-fs-corner"></div>
  <div class="slew-fs-btn slew-fs-n"
       ontouchstart="fsSlew('N',event)" ontouchend="fsStop(event)" ontouchcancel="fsStop(event)"
       onmousedown="fsSlew('N',event)" onmouseup="fsStop(event)">
    <span class="fs-arrow">&#9650;</span><span class="fs-label">N</span>
  </div>
  <div class="slew-fs-corner"></div>
  <div class="slew-fs-btn slew-fs-w"
       ontouchstart="fsSlew('W',event)" ontouchend="fsStop(event)" ontouchcancel="fsStop(event)"
       onmousedown="fsSlew('W',event)" onmouseup="fsStop(event)">
    <span class="fs-arrow">&#9664;</span><span class="fs-label">W</span>
  </div>
  <div class="slew-fs-btn slew-fs-stop" onclick="fsSlewStop()">
    <span class="fs-arrow">&#9632;</span><span class="fs-label">STOP</span>
  </div>
  <div class="slew-fs-btn slew-fs-e"
       ontouchstart="fsSlew('E',event)" ontouchend="fsStop(event)" ontouchcancel="fsStop(event)"
       onmousedown="fsSlew('E',event)" onmouseup="fsStop(event)">
    <span class="fs-arrow">&#9654;</span><span class="fs-label">E</span>
  </div>
  <div class="slew-fs-corner"></div>
  <div class="slew-fs-btn slew-fs-s"
       ontouchstart="fsSlew('S',event)" ontouchend="fsStop(event)" ontouchcancel="fsStop(event)"
       onmousedown="fsSlew('S',event)" onmouseup="fsStop(event)">
    <span class="fs-arrow">&#9660;</span><span class="fs-label">S</span>
  </div>
  <div class="slew-fs-corner"></div>
</div>

<script>
/* ============================================================
   Fullscreen slew overlay controls + Haptic feedback
   ============================================================ */
let _fsSpeed = 4;

function toggleFullscreenSlew() {
  const overlay = document.getElementById('slew-fs-overlay');
  overlay.classList.toggle('active');
  if (overlay.classList.contains('active')) {
    // Sync speed from main selector
    const mainSpeed = document.getElementById('slew-speed');
    if (mainSpeed) { _fsSpeed = parseInt(mainSpeed.value) || 4; }
    updateFsSpeedBtns();
    // Vibrate to confirm entry
    if (navigator.vibrate) navigator.vibrate(50);
  } else {
    slewStop();
  }
}

function setFsSpeed(s) {
  _fsSpeed = s;
  // Sync back to main speed selector
  const mainSpeed = document.getElementById('slew-speed');
  if (mainSpeed) mainSpeed.value = s;
  updateFsSpeedBtns();
  if (navigator.vibrate) navigator.vibrate(20);
}

function updateFsSpeedBtns() {
  document.querySelectorAll('#slew-fs-speed .fs-spd-btn').forEach(b => {
    b.classList.toggle('active', parseInt(b.dataset.speed) === _fsSpeed);
  });
}

function fsSlew(dir, e) {
  if (e) e.preventDefault();
  if (navigator.vibrate) navigator.vibrate(30);
  apiPost('/api/slew', { direction: dir, speed: _fsSpeed });
}

function fsStop(e) {
  if (e) e.preventDefault();
  apiPost('/api/slew/stop');
}

function fsSlewStop() {
  if (navigator.vibrate) navigator.vibrate([40, 30, 40]);
  apiPost('/api/slew/stop');
}

/* Haptic-enhanced slew for the normal (non-fullscreen) grid */
function slewStartHaptic(dir, e) {
  if (e) e.preventDefault();
  if (navigator.vibrate) navigator.vibrate(25);
  slewStart(dir);
}

function slewStopHaptic(e) {
  if (e) e.preventDefault();
  slewStop();
}

// ============================================================
// Sky Chart Engine
// ============================================================
const D2R = Math.PI/180, R2D = 180/Math.PI, H2R = Math.PI/12;

// Spectral type -> RGB color (realistic, Harre & Heller 2021)
const SC_COLORS = {
  O:'#9bb0ff', B:'#aabfff', A:'#cad7ff', F:'#f8f7ff',
  G:'#fff4ea', K:'#ffd2a1', M:'#ffb56c'
};

// Hex color to rgba string
function scHexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1,3), 16);
  const g = parseInt(hex.slice(3,5), 16);
  const b = parseInt(hex.slice(5,7), 16);
  return 'rgba('+r+','+g+','+b+','+alpha+')';
}

// Sky chart state
const sc = {
  canvas:null, ctx:null, W:0, H:0,
  stars:[], extStars:[], dsos:[], planets:[], conLines:[], conLabels:[],
  obs:{lat:0,lon:0}, scope:null,
  // View: azimuthal (alt/az) center
  vAz:180, vAlt:45, fov:90,
  // Toggles
  grid:false, constLines:true, constNames:true, dsoShow:true, labels:true, follow:false,
  ecliptic:false, equator:false, fullscreen:false,
  // Interaction
  dragging:false, lx:0, ly:0, pinchD:0,
  // Selection
  sel:null, selScreen:null, selType:null,
  // Animation
  raf:null, loaded:false,
  // Per-frame cache
  frameLST:0,
};

function scInit() {
  sc.canvas = document.getElementById('sc-canvas');
  if (!sc.canvas) return;
  sc.ctx = sc.canvas.getContext('2d');
  const wrap = document.getElementById('sc-wrap');
  // Size canvas to fill card width, ~65vh height (or full viewport in fullscreen)
  function resize() {
    const w = wrap.clientWidth || 360;
    const h = sc.fullscreen ? window.innerHeight : Math.max(320, Math.min(window.innerHeight * 0.65, 700));
    wrap.style.height = h + 'px';
    const dpr = window.devicePixelRatio || 1;
    sc.canvas.width = w * dpr;
    sc.canvas.height = h * dpr;
    sc.canvas.style.width = w + 'px';
    sc.canvas.style.height = h + 'px';
    sc.W = w * dpr; sc.H = h * dpr;
    sc.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    sc.W = w; sc.H = h;
    if (sc.loaded) scRender();
  }
  resize();
  window.addEventListener('resize', resize);
  // Pointer events (unified touch + mouse)
  sc.canvas.addEventListener('pointerdown', scPtrDown);
  sc.canvas.addEventListener('pointermove', scPtrMove);
  sc.canvas.addEventListener('pointerup', scPtrUp);
  sc.canvas.addEventListener('pointercancel', scPtrUp);
  sc.canvas.addEventListener('wheel', scWheel, {passive:false});
  // Touch for pinch zoom
  sc.canvas.addEventListener('touchstart', scTouchStart, {passive:false});
  sc.canvas.addEventListener('touchmove', scTouchMove, {passive:false});
  sc.canvas.addEventListener('touchend', scTouchEnd);
  scLoadData();
}

// ---- LST & coordinate transforms ----
function scLST() {
  const now = new Date();
  const y = now.getUTCFullYear(), mo = now.getUTCMonth()+1;
  let yr=y, mn=mo;
  const d = now.getUTCDate() + now.getUTCHours()/24 + now.getUTCMinutes()/1440 + now.getUTCSeconds()/86400;
  if (mn<=2){yr--;mn+=12;}
  const a = Math.floor(yr/100), b = 2-a+Math.floor(a/4);
  const jd = Math.floor(365.25*(yr+4716))+Math.floor(30.6001*(mn+1))+d+b-1524.5;
  const gst = (280.46061837+360.98564736629*(jd-2451545.0))%360;
  return (((gst+sc.obs.lon)/15)%24+24)%24;
}

function scRaDecToAltAz(ra_h, dec_d) {
  const lst = sc.frameLST || scLST();
  const ha = ((lst - ra_h)*15) % 360;
  const haR = (ha<0?ha+360:ha)*D2R;
  const latR = sc.obs.lat*D2R, decR = dec_d*D2R;
  const sinAlt = Math.sin(latR)*Math.sin(decR)+Math.cos(latR)*Math.cos(decR)*Math.cos(haR);
  const alt = Math.asin(Math.max(-1,Math.min(1,sinAlt)));
  const cosAlt = Math.cos(alt);
  let az = 0;
  if (Math.abs(cosAlt) > 1e-10) {
    const sinAz = -Math.cos(decR)*Math.sin(haR)/cosAlt;
    const cosAz = (Math.sin(decR)-Math.sin(latR)*sinAlt)/(Math.cos(latR)*cosAlt);
    az = Math.atan2(sinAz, cosAz);
  }
  return {alt: alt*R2D, az: ((az*R2D)%360+360)%360};
}

// Stereographic projection: (alt,az) -> screen (x,y) or null
function scProject(alt, az) {
  const a1=sc.vAlt*D2R, z1=sc.vAz*D2R;
  const a2=alt*D2R, z2=az*D2R;
  const cosC = Math.sin(a1)*Math.sin(a2)+Math.cos(a1)*Math.cos(a2)*Math.cos(z2-z1);
  if (cosC < -0.05) return null; // behind viewer
  const k = 2/(1+cosC);
  const x = k * Math.cos(a2)*Math.sin(z2-z1);
  const y = k * (Math.cos(a1)*Math.sin(a2)-Math.sin(a1)*Math.cos(a2)*Math.cos(z2-z1));
  const scale = Math.min(sc.W,sc.H) / (sc.fov*D2R*2);
  return {x: sc.W/2 + x*scale, y: sc.H/2 - y*scale};
}

// ---- Rendering ----
function scRender() {
  const c = sc.ctx;
  if (!c) return;
  sc.frameLST = scLST(); // Cache LST for entire frame
  // Background: deep space gradient
  const bg = c.createRadialGradient(sc.W/2,sc.H/2,0,sc.W/2,sc.H/2,sc.W*0.7);
  bg.addColorStop(0,'#0a0a1a'); bg.addColorStop(1,'#020208');
  c.fillStyle = bg; c.fillRect(0,0,sc.W,sc.H);

  scDrawMilkyWay(c);
  if (sc.grid) scDrawGrid(c);
  if (sc.equator) scDrawEquator(c);
  if (sc.ecliptic) scDrawEcliptic(c);
  scDrawHorizon(c);
  if (sc.constLines) scDrawConLines(c);
  if (sc.constNames) scDrawConLabels(c);
  scDrawStars(c);
  if (sc.dsoShow) scDrawDSOs(c);
  scDrawPlanets(c);
  scDrawScope(c);
  scDrawCompass(c);
  // Recompute selection screen pos each frame (fixes pan drift)
  if (sc.sel) {
    const aa = scRaDecToAltAz(sc.sel.r, sc.sel.d);
    sc.selScreen = scProject(aa.alt, aa.az);
  }
  if (sc.sel && sc.selScreen) scDrawSelection(c);
}

function scDrawMilkyWay(c) {
  // Enhanced Milky Way rendering with Gaussian profile, Great Rift dark lane,
  // bright star cloud patches, variable width, and warm color tones.
  // Galactic-to-equatorial: sin(dec) = sin(decGP)*sin(b) + cos(decGP)*cos(b)*sin(l-lNCP)
  const decGP=27.12825*D2R, raGP=192.85948*D2R, lNCP=122.93192*D2R;
  const sinDecGP = Math.sin(decGP), cosDecGP = Math.cos(decGP);
  c.save();
  c.lineCap = 'round'; c.lineJoin = 'round';
  const lStep = 3;   // 3-degree longitude steps for smooth appearance
  const baseA = 0.032; // base opacity per band layer

  // 21 latitude bands from -20 to +20 degrees (2-degree spacing)
  for (let bDeg = -20; bDeg <= 20; bDeg += 2) {
    const bR = bDeg * D2R;
    const sinB = Math.sin(bR), cosB = Math.cos(bR);
    const absBDeg = Math.abs(bDeg);

    // Project all points for this latitude ring
    const pts = [], lons = [];
    for (let l = 0; l < 360; l += lStep) {
      const lr = l * D2R;
      const sinLNCP = Math.sin(lr - lNCP), cosLNCP = Math.cos(lr - lNCP);
      const sinDec = sinDecGP*sinB + cosDecGP*cosB*sinLNCP;
      const dec = Math.asin(Math.max(-1, Math.min(1, sinDec)));
      const yy = cosB * cosLNCP;
      const xx = sinB*cosDecGP - cosB*sinDecGP*sinLNCP;
      const ra = raGP + Math.atan2(yy, xx);
      const ra_h = (((ra*R2D)%360+360)%360)/15;
      const aa = scRaDecToAltAz(ra_h, dec*R2D);
      pts.push(scProject(aa.alt, aa.az));
      lons.push(l);
    }

    for (let i = 1; i < pts.length; i++) {
      if (!pts[i] || !pts[i-1]) continue;
      if (Math.abs(pts[i].x - pts[i-1].x) > sc.W * 0.5) continue;
      const l = lons[i];

      // --- Longitude brightness: Sgr center (l~0/360) brightest ---
      const cosHalfL = Math.cos(l * D2R * 0.5);
      const lBright = 0.35 + 0.65 * cosHalfL * cosHalfL;

      // --- Variable width: MW wider near center (sigma 12 deg) vs anticenter (7 deg) ---
      const sigma = 7 + 5 * cosHalfL * cosHalfL;

      // --- Gaussian falloff from galactic plane ---
      const gauss = Math.exp(-(bDeg * bDeg) / (2 * sigma * sigma));

      // --- Great Rift dark lane (l ~ 10-90 deg, strongest at l~45, near plane) ---
      let rift = 1.0;
      if (l >= 5 && l <= 100 && absBDeg < 5) {
        const riftL = Math.exp(-((l - 45) * (l - 45)) / (30 * 30));
        const riftB = Math.exp(-(bDeg * bDeg) / (2.5 * 2.5));
        rift = 1.0 - 0.55 * riftL * riftB;
      }

      // --- Bright star-cloud patches ---
      let patch = 1.0;
      // Sagittarius Star Cloud (l ~ 5-15 deg)
      if (l < 20 && absBDeg < 7)
        patch += 0.35 * Math.exp(-((l-8)*(l-8))/36) * Math.exp(-(bDeg*bDeg)/20);
      // Scutum Star Cloud (l ~ 27 deg, slightly below plane)
      if (l >= 18 && l <= 38 && absBDeg < 6)
        patch += 0.25 * Math.exp(-((l-27)*(l-27))/25) * Math.exp(-((bDeg+2)*(bDeg+2))/12);
      // Cygnus Star Cloud (l ~ 78-82 deg)
      if (l >= 65 && l <= 95 && absBDeg < 6)
        patch += 0.25 * Math.exp(-((l-80)*(l-80))/36) * Math.exp(-(bDeg*bDeg)/14);
      // Carina bright region (l ~ 280-290 deg, southern sky)
      if (l >= 270 && l <= 300 && absBDeg < 6)
        patch += 0.2 * Math.exp(-((l-285)*(l-285))/40) * Math.exp(-(bDeg*bDeg)/14);

      // --- Combined alpha ---
      const alpha = baseA * lBright * gauss * rift * patch;
      if (alpha < 0.002) continue;

      // --- Line width scales with MW angular width at this longitude ---
      const lineW = Math.max(6, sc.W * 0.038 * sigma / 10);

      // --- Color: warmer near center, bluer at edges ---
      const warmth = lBright * gauss;
      const cr = Math.min(255, 130 + Math.round(50 * warmth));
      const cg = Math.min(255, 145 + Math.round(30 * warmth));
      const cb = Math.min(255, 190 - Math.round(15 * warmth));
      c.strokeStyle = 'rgb(' + cr + ',' + cg + ',' + cb + ')';
      c.lineWidth = lineW;
      c.globalAlpha = alpha;
      c.beginPath();
      c.moveTo(pts[i-1].x, pts[i-1].y);
      c.lineTo(pts[i].x, pts[i].y);
      c.stroke();
    }
  }
  c.restore();
}

function scDrawGrid(c) {
  c.save();
  c.lineWidth = 1.0;
  // Alt circles
  for (let alt=0; alt<=80; alt+=10) {
    c.beginPath();
    c.strokeStyle = (alt % 30 === 0) ? 'rgba(100,160,220,0.35)' : 'rgba(100,140,200,0.18)';
    let first=true;
    for (let az=0; az<360; az+=2) {
      const p = scProject(alt, az);
      if (p) { if(first){c.moveTo(p.x,p.y);first=false;}else c.lineTo(p.x,p.y); }
    }
    c.stroke();
  }
  // Alt labels
  c.font = Math.max(8,sc.W*0.018)+'px system-ui';
  c.fillStyle = 'rgba(120,170,220,0.45)';
  c.textAlign = 'left'; c.textBaseline = 'middle';
  for (let alt=10; alt<=80; alt+=10) {
    const p = scProject(alt, sc.vAz);
    if (p && p.x > 30 && p.x < sc.W-30 && p.y > 15 && p.y < sc.H-15) {
      c.fillText(alt+'\u00b0', p.x+4, p.y);
    }
  }
  // Az lines
  for (let az=0; az<360; az+=15) {
    c.beginPath();
    c.strokeStyle = (az % 45 === 0) ? 'rgba(100,160,220,0.30)' : 'rgba(100,140,200,0.15)';
    let first=true;
    for (let alt=-10; alt<=90; alt+=2) {
      const p = scProject(alt, az);
      if (p) { if(first){c.moveTo(p.x,p.y);first=false;}else c.lineTo(p.x,p.y); }
    }
    c.stroke();
  }
  // Az labels along horizon
  c.fillStyle = 'rgba(120,170,220,0.45)';
  for (let az=0; az<360; az+=15) {
    const p = scProject(2, az);
    if (p && p.x > 20 && p.x < sc.W-20 && p.y > 10 && p.y < sc.H-10) {
      c.textAlign = 'center';
      c.fillText(az+'\u00b0', p.x, p.y+12);
    }
  }
  c.restore();
}

function scDrawHorizon(c) {
  c.save();
  // Collect horizon points (alt=0) at fine azimuth steps
  const pts = [];
  for (let az=0; az<=360; az+=1) {
    const p = scProject(0, az);
    if (p) pts.push({x:p.x, y:p.y, az:az});
  }
  if (pts.length > 2) {
    // Sort points by x to build a proper fill polygon
    // Fill below horizon with opaque dark earth gradient
    c.beginPath();
    c.moveTo(pts[0].x, pts[0].y);
    for (let i=1; i<pts.length; i++) {
      if (Math.abs(pts[i].x - pts[i-1].x) < sc.W*0.6)
        c.lineTo(pts[i].x, pts[i].y);
      else
        c.moveTo(pts[i].x, pts[i].y);
    }
    // Close downward to fill entire area below horizon
    c.lineTo(sc.W+10, pts[pts.length-1].y);
    c.lineTo(sc.W+10, sc.H+10);
    c.lineTo(-10, sc.H+10);
    c.lineTo(-10, pts[0].y);
    c.closePath();
    // Dark earth gradient - opaque so nothing shows through
    const grdY = Math.min(...pts.map(p=>p.y));
    const earthGrad = c.createLinearGradient(0, grdY, 0, sc.H);
    earthGrad.addColorStop(0, 'rgba(25,18,12,0.95)');
    earthGrad.addColorStop(0.15, 'rgba(18,14,10,0.97)');
    earthGrad.addColorStop(0.5, 'rgba(12,10,8,0.98)');
    earthGrad.addColorStop(1, 'rgba(8,6,5,0.99)');
    c.fillStyle = earthGrad;
    c.fill();

    // Horizon glow line
    c.beginPath();
    c.moveTo(pts[0].x, pts[0].y);
    for (let i=1; i<pts.length; i++) {
      if (Math.abs(pts[i].x - pts[i-1].x) < sc.W*0.6)
        c.lineTo(pts[i].x, pts[i].y);
      else
        c.moveTo(pts[i].x, pts[i].y);
    }
    // Outer glow
    c.strokeStyle = 'rgba(180,120,50,0.15)';
    c.lineWidth = 6;
    c.stroke();
    // Main horizon line
    c.strokeStyle = 'rgba(200,130,50,0.65)';
    c.lineWidth = 1.5;
    c.stroke();
  }
  c.restore();
}

function scDrawConLines(c) {
  c.save();
  c.strokeStyle = 'rgba(70,130,210,0.40)';
  c.lineWidth = 1.2;
  for (const seg of sc.conLines) {
    const aa1 = scRaDecToAltAz(seg[0], seg[1]);
    const aa2 = scRaDecToAltAz(seg[2], seg[3]);
    const p1 = scProject(aa1.alt, aa1.az);
    const p2 = scProject(aa2.alt, aa2.az);
    if (p1 && p2 && Math.abs(p1.x-p2.x)<sc.W*0.5) {
      c.beginPath(); c.moveTo(p1.x,p1.y); c.lineTo(p2.x,p2.y); c.stroke();
    }
  }
  c.restore();
}

function scDrawConLabels(c) {
  c.save();
  c.fillStyle = 'rgba(70,130,210,0.50)';
  c.font = '600 ' + Math.max(9, sc.W*0.024) + 'px system-ui';
  c.textAlign = 'center'; c.textBaseline = 'middle';
  for (const lb of sc.conLabels) {
    const aa = scRaDecToAltAz(lb.ra, lb.dec);
    const p = scProject(aa.alt, aa.az);
    if (p && p.x > 0 && p.x < sc.W && p.y > 0 && p.y < sc.H) {
      c.fillText(lb.name, p.x, p.y);
    }
  }
  c.restore();
}

function scStarColor(star) {
  if (star.s && SC_COLORS[star.s]) return SC_COLORS[star.s];
  // Approximate spectral type from magnitude (rough B-V proxy)
  if (star.m < 0.5) return '#cad7ff';  // Bright → likely hot (A/B)
  if (star.m < 1.5) return '#dde4ff';
  if (star.m < 2.5) return '#f0ecff';
  if (star.m < 3.5) return '#fff8ee';  // Mid → likely F/G
  return '#ffeedd';                     // Faint → likely K
}

function scStarSize(mag) {
  // KStars-inspired: wider magnitude range for more dramatic size variation
  // Sirius(-1.46)~5.5px, Vega(0)~3.2px, mag3~1.0px, mag6~0.4px on 360px screen
  const base = Math.max(sc.W, sc.H) * 0.009;
  const sz = base * Math.pow(0.68, mag);
  return Math.max(0.4, Math.min(base * 1.8, sz));
}

function scDrawStars(c) {
  c.save();
  // Magnitude limit: wider FOV shows fewer stars, zoomed in reveals fainter
  // FOV 150->mag 5, FOV 90->mag 6.5, FOV 40->mag 7.5, FOV 10->mag 8
  const magLim = 5.0 + (150 - sc.fov) * 0.022;

  // --- Extended stars (24K+ faint stars, simple rendering) ---
  for (const s of sc.extStars) {
    if (s.m > magLim) continue;
    const aa = scRaDecToAltAz(s.r, s.d);
    const p = scProject(aa.alt, aa.az);
    if (!p || p.x < -5 || p.x > sc.W+5 || p.y < -5 || p.y > sc.H+5) continue;
    const sz = scStarSize(s.m);
    const col = scStarColor(s);
    // Faint stars: just a colored dot (fast)
    if (sz < 1.0) {
      c.globalAlpha = 0.4 + sz * 0.4;
      c.fillStyle = col;
      c.fillRect(p.x - sz*0.4, p.y - sz*0.4, sz*0.8, sz*0.8);
    } else {
      // Medium stars: small circle + subtle glow for brighter ones
      if (sz > 1.5) {
        const gr = sz * 2.5;
        const g = c.createRadialGradient(p.x,p.y,0,p.x,p.y,gr);
        g.addColorStop(0, scHexToRgba(col, 0.35));
        g.addColorStop(1, scHexToRgba(col, 0));
        c.globalAlpha = 1;
        c.fillStyle = g;
        c.beginPath(); c.arc(p.x,p.y,gr,0,Math.PI*2); c.fill();
      }
      c.globalAlpha = 0.6 + sz * 0.2;
      c.fillStyle = col;
      c.beginPath(); c.arc(p.x,p.y,Math.max(0.4,sz*0.45),0,Math.PI*2); c.fill();
    }
  }
  c.globalAlpha = 1;

  // --- Primary named stars (408 bright stars, full rendering) ---
  for (const s of sc.stars) {
    const aa = scRaDecToAltAz(s.r, s.d);
    const p = scProject(aa.alt, aa.az);
    if (!p || p.x < -20 || p.x > sc.W+20 || p.y < -20 || p.y > sc.H+20) continue;
    const sz = scStarSize(s.m);
    const col = scStarColor(s);
    // Glow halo for bright stars
    if (sz > 1.0) {
      const glowR = sz * 3.5;
      const g = c.createRadialGradient(p.x,p.y,0,p.x,p.y,glowR);
      g.addColorStop(0, scHexToRgba(col, 0.6));
      g.addColorStop(0.25, scHexToRgba(col, 0.15));
      g.addColorStop(1, scHexToRgba(col, 0));
      c.fillStyle = g;
      c.beginPath(); c.arc(p.x,p.y,glowR,0,Math.PI*2); c.fill();
    }
    // Colored core
    c.globalAlpha = Math.min(1, 0.6 + sz*0.25);
    c.fillStyle = col;
    c.beginPath(); c.arc(p.x,p.y,Math.max(0.4,sz*0.55),0,Math.PI*2); c.fill();
    // Bright star white center highlight
    if (sz > 2.0) {
      c.fillStyle = 'rgba(255,255,255,0.55)';
      c.beginPath(); c.arc(p.x,p.y,sz*0.18,0,Math.PI*2); c.fill();
    }
    c.globalAlpha = 1;
    // Label for bright named stars
    if (sc.labels && s.n && s.m < 2.5) {
      c.fillStyle = 'rgba(220,210,190,0.7)';
      c.font = Math.max(8, sc.W*0.02) + 'px system-ui';
      c.textAlign = 'left'; c.textBaseline = 'bottom';
      c.fillText(s.n, p.x + sz + 4, p.y - 3);
    }
  }
  c.restore();
}

function scDrawDSOs(c) {
  c.save();
  for (const d of sc.dsos) {
    const aa = scRaDecToAltAz(d.r, d.d);
    const p = scProject(aa.alt, aa.az);
    if (!p || p.x<-10||p.x>sc.W+10||p.y<-10||p.y>sc.H+10) continue;
    // Magnitude limit depends on zoom
    if (d.m > 8 + (90 - sc.fov)*0.075) continue;
    // Scale symbol size by magnitude (brighter = bigger)
    const baseSz = Math.max(3, sc.W*0.008);
    const sz = baseSz * Math.max(0.7, Math.min(1.8, (12 - d.m) / 8));
    c.lineWidth = 1;
    const t = d.t || '';
    if (t.includes('Galaxy')) {
      c.strokeStyle = 'rgba(255,180,80,0.50)';
      c.beginPath(); c.ellipse(p.x,p.y,sz*1.3,sz*0.6,0.3,0,Math.PI*2); c.stroke();
    } else if (t.includes('Globular')) {
      c.strokeStyle = 'rgba(255,220,100,0.50)';
      c.beginPath(); c.arc(p.x,p.y,sz,0,Math.PI*2); c.stroke();
      c.beginPath(); c.moveTo(p.x-sz,p.y); c.lineTo(p.x+sz,p.y); c.stroke();
      c.beginPath(); c.moveTo(p.x,p.y-sz); c.lineTo(p.x,p.y+sz); c.stroke();
    } else if (t.includes('Planetary')) {
      // MUST check before 'Nebula' (fixes 'Planetary Nebula' matching wrong branch)
      c.strokeStyle = 'rgba(100,220,255,0.55)';
      c.beginPath(); c.arc(p.x,p.y,sz,0,Math.PI*2); c.stroke();
      c.fillStyle = 'rgba(100,220,255,0.35)';
      c.beginPath(); c.arc(p.x,p.y,1.5,0,Math.PI*2); c.fill();
    } else if (t.includes('Nebula') || t.includes('SNR')) {
      c.strokeStyle = 'rgba(100,220,100,0.50)';
      c.strokeRect(p.x-sz, p.y-sz, sz*2, sz*2);
    } else {
      // Open cluster or other: dashed circle
      c.strokeStyle = 'rgba(255,255,150,0.40)';
      c.setLineDash([2,2]);
      c.beginPath(); c.arc(p.x,p.y,sz,0,Math.PI*2); c.stroke();
      c.setLineDash([]);
    }
    // Label: show for brighter DSOs (threshold scales with zoom)
    const labelMagLim = 6 + (90 - sc.fov)*0.04;
    if (sc.labels && d.m < labelMagLim) {
      c.fillStyle = 'rgba(255,200,100,0.55)';
      c.font = Math.max(7, sc.W*0.016) + 'px system-ui';
      c.textAlign = 'left'; c.textBaseline = 'bottom';
      const label = d.cn ? d.n + ' ' + d.cn : d.n;
      c.fillText(label, p.x+sz+3, p.y-2);
    }
  }
  c.restore();
}

function scDrawPlanets(c) {
  c.save();
  const pColors = {Sun:'#ffee44',Moon:'#e8e8d0',Mercury:'#bbbbbb',Venus:'#ffffcc',
    Mars:'#ff8844',Jupiter:'#ffddaa',Saturn:'#ffe8aa',Uranus:'#aaeeff',Neptune:'#7799ff'};
  for (const pl of sc.planets) {
    const aa = scRaDecToAltAz(pl.r, pl.d);
    const p = scProject(aa.alt, aa.az);
    if (!p) continue;
    const col = pColors[pl.n] || '#ffcc88';
    const sz = pl.n==='Sun'||pl.n==='Moon' ? sc.W*0.012 : sc.W*0.007;
    // Glow
    const g = c.createRadialGradient(p.x,p.y,0,p.x,p.y,sz*4);
    g.addColorStop(0,col); g.addColorStop(0.4,col.slice(0,7)+'60'); g.addColorStop(1,'transparent');
    c.fillStyle = g;
    c.beginPath(); c.arc(p.x,p.y,sz*4,0,Math.PI*2); c.fill();
    // Core
    c.fillStyle = col;
    c.beginPath(); c.arc(p.x,p.y,sz,0,Math.PI*2); c.fill();
    // Label
    c.fillStyle = col;
    c.font = '600 '+Math.max(9,sc.W*0.02)+'px system-ui';
    c.textAlign = 'left'; c.textBaseline = 'bottom';
    c.fillText(pl.n, p.x+sz+4, p.y-3);
  }
  c.restore();
}

function scDrawScope(c) {
  if (!sc.scope) return;
  const p = scProject(sc.scope.alt, sc.scope.az);
  if (!p) return;
  const sz = Math.max(14, sc.W*0.035);
  c.save();
  // Outer glow for visibility against any background
  c.shadowColor = 'rgba(255,50,50,0.6)';
  c.shadowBlur = 8;
  c.strokeStyle = '#ff3333';
  c.lineWidth = 2.5;
  c.globalAlpha = 0.9;
  // Outer circle
  c.beginPath(); c.arc(p.x,p.y,sz,0,Math.PI*2); c.stroke();
  // Inner circle
  c.shadowBlur = 0;
  c.lineWidth = 1.5;
  c.beginPath(); c.arc(p.x,p.y,sz*0.4,0,Math.PI*2); c.stroke();
  // Crosshair arms (extending beyond outer circle)
  c.lineWidth = 2;
  c.beginPath();
  c.moveTo(p.x-sz*1.6, p.y); c.lineTo(p.x-sz*1.05, p.y);
  c.moveTo(p.x+sz*1.05, p.y); c.lineTo(p.x+sz*1.6, p.y);
  c.moveTo(p.x, p.y-sz*1.6); c.lineTo(p.x, p.y-sz*1.05);
  c.moveTo(p.x, p.y+sz*1.05); c.lineTo(p.x, p.y+sz*1.6);
  c.stroke();
  // Center dot
  c.fillStyle = '#ff3333';
  c.beginPath(); c.arc(p.x,p.y,1.5,0,Math.PI*2); c.fill();
  // Label
  c.shadowBlur = 0;
  c.globalAlpha = 0.8;
  c.font = '600 '+Math.max(9,sc.W*0.018)+'px system-ui';
  c.fillStyle = '#ff5555';
  c.textAlign = 'left'; c.textBaseline = 'bottom';
  c.fillText('Scope', p.x+sz*1.2, p.y-sz*0.5);
  // Coordinate readout
  c.font = Math.max(8,sc.W*0.014)+'px system-ui';
  c.globalAlpha = 0.6;
  c.fillText('Alt '+sc.scope.alt.toFixed(1)+'\u00b0 Az '+sc.scope.az.toFixed(1)+'\u00b0', p.x+sz*1.2, p.y-sz*0.5+12);
  c.restore();
}

function scDrawCompass(c) {
  c.save();
  const dirs = [{az:0,l:'N'},{az:90,l:'E'},{az:180,l:'S'},{az:270,l:'W'}];
  c.font = '700 '+Math.max(10,sc.W*0.025)+'px system-ui';
  c.textAlign = 'center'; c.textBaseline = 'middle';
  for (const d of dirs) {
    const p = scProject(0, d.az);
    if (p && p.x>5 && p.x<sc.W-5) {
      c.fillStyle = d.l==='N'?'rgba(255,80,80,0.7)':'rgba(200,170,120,0.5)';
      c.fillText(d.l, p.x, p.y + 14);
    }
  }
  c.restore();
}

function scDrawSelection(c) {
  if (!sc.selScreen) return;
  const p = sc.selScreen;
  c.save();
  c.strokeStyle = '#ffcc44';
  c.lineWidth = 1.5;
  c.setLineDash([4,4]);
  c.beginPath(); c.arc(p.x,p.y,14,0,Math.PI*2); c.stroke();
  c.setLineDash([]);
  c.restore();
}

// ---- Ecliptic & Celestial Equator ----
function scDrawEcliptic(c) {
  c.save();
  c.strokeStyle = 'rgba(200,180,50,0.35)';
  c.lineWidth = 1;
  c.setLineDash([6,4]);
  const obliquity = 23.4393 * D2R;
  const pts = [];
  for (let lon = 0; lon < 360; lon += 2) {
    const lonR = lon * D2R;
    const dec = Math.asin(Math.sin(obliquity) * Math.sin(lonR));
    const ra = Math.atan2(Math.cos(obliquity)*Math.sin(lonR), Math.cos(lonR));
    const ra_h = (((ra*R2D)%360+360)%360)/15;
    const aa = scRaDecToAltAz(ra_h, dec*R2D);
    pts.push(scProject(aa.alt, aa.az));
  }
  c.beginPath();
  let prev = null;
  for (const p of pts) {
    if (!p) { prev = null; continue; }
    if (!prev || Math.abs(p.x - prev.x) > sc.W * 0.5) c.moveTo(p.x, p.y);
    else c.lineTo(p.x, p.y);
    prev = p;
  }
  c.stroke();
  c.setLineDash([]);
  // Label
  c.fillStyle = 'rgba(200,180,50,0.4)';
  c.font = Math.max(8, sc.W*0.016) + 'px system-ui';
  for (const p of pts) { if (p && p.x > sc.W*0.4 && p.x < sc.W*0.6) { c.fillText('Ecliptic', p.x+5, p.y-5); break; } }
  c.restore();
}

function scDrawEquator(c) {
  c.save();
  c.strokeStyle = 'rgba(200,80,80,0.3)';
  c.lineWidth = 1;
  c.setLineDash([6,4]);
  const pts = [];
  for (let ra = 0; ra < 24; ra += 0.1) {
    const aa = scRaDecToAltAz(ra, 0);
    pts.push(scProject(aa.alt, aa.az));
  }
  c.beginPath();
  let prev = null;
  for (const p of pts) {
    if (!p) { prev = null; continue; }
    if (!prev || Math.abs(p.x - prev.x) > sc.W * 0.5) c.moveTo(p.x, p.y);
    else c.lineTo(p.x, p.y);
    prev = p;
  }
  c.stroke();
  c.setLineDash([]);
  c.fillStyle = 'rgba(200,80,80,0.35)';
  c.font = Math.max(8, sc.W*0.016) + 'px system-ui';
  for (const p of pts) { if (p && p.x > sc.W*0.3 && p.x < sc.W*0.5) { c.fillText('Equator', p.x+5, p.y-5); break; } }
  c.restore();
}

// ---- Interaction ----
let _scPointers = {};
function scPtrDown(e) {
  _scPointers[e.pointerId] = {x:e.offsetX,y:e.offsetY};
  if (Object.keys(_scPointers).length === 1) {
    sc.dragging = false;
    sc.lx = e.offsetX; sc.ly = e.offsetY;
    sc._clickX = e.offsetX; sc._clickY = e.offsetY;
  }
  sc.canvas.setPointerCapture(e.pointerId);
}
function scPtrMove(e) {
  _scPointers[e.pointerId] = {x:e.offsetX,y:e.offsetY};
  const ids = Object.keys(_scPointers);
  if (ids.length === 1) {
    const dx = e.offsetX - sc.lx, dy = e.offsetY - sc.ly;
    if (Math.abs(dx)+Math.abs(dy) > 3) sc.dragging = true;
    if (sc.dragging) {
      const sens = sc.fov / sc.W;
      sc.vAz -= dx * sens;
      sc.vAlt += dy * sens;
      sc.vAz = ((sc.vAz % 360) + 360) % 360;
      sc.vAlt = Math.max(-10, Math.min(90, sc.vAlt));
      sc.lx = e.offsetX; sc.ly = e.offsetY;
      scRender();
    }
  } else if (ids.length === 2) {
    // Pinch zoom
    const ps = ids.map(id => _scPointers[id]);
    const dist = Math.hypot(ps[1].x-ps[0].x, ps[1].y-ps[0].y);
    if (sc.pinchD > 0) {
      const ratio = sc.pinchD / dist;
      sc.fov = Math.max(10, Math.min(150, sc.fov * ratio));
      scUpdateFov();
      scRender();
    }
    sc.pinchD = dist;
  }
}
function scPtrUp(e) {
  delete _scPointers[e.pointerId];
  if (Object.keys(_scPointers).length === 0) {
    sc.pinchD = 0;
    if (!sc.dragging && sc._clickX !== undefined) {
      scTapAt(sc._clickX, sc._clickY);
    }
    sc.dragging = false;
  }
}
function scTouchStart(e) {
  if (e.touches.length === 2) { e.preventDefault(); sc.pinchD = 0; }
}
function scTouchMove(e) {
  if (e.touches.length === 2) e.preventDefault();
}
function scTouchEnd(e) { sc.pinchD = 0; }

function scWheel(e) {
  e.preventDefault();
  sc.fov = Math.max(10, Math.min(150, sc.fov + e.deltaY * 0.08));
  scUpdateFov();
  scRender();
}

function scUpdateFov() {
  const el = document.getElementById('sc-fov');
  if (el) el.textContent = 'FOV ' + Math.round(sc.fov) + '\u00b0';
}

function scTapAt(x, y) {
  // Find nearest object
  let best = null, bestD = 25;
  function check(obj, type) {
    const aa = scRaDecToAltAz(obj.r, obj.d);
    const p = scProject(aa.alt, aa.az);
    if (!p) return;
    const d = Math.hypot(p.x-x, p.y-y);
    if (d < bestD) { bestD = d; best = {obj, type, screen: p}; }
  }
  for (const s of sc.stars) { if (s.n) check(s, 'star'); }
  for (const d of sc.dsos) check(d, 'dso');
  for (const p of sc.planets) check(p, 'planet');
  // Also check unnamed bright stars
  if (!best) { for (const s of sc.stars) { if (!s.n && s.m < 3) check(s, 'star'); } }
  if (best) {
    sc.sel = best.obj;
    sc.selScreen = best.screen;
    sc.selType = best.type;
    scShowInfo(best.obj, best.type);
    scRender();
  } else {
    scCloseInfo();
  }
}

function scShowInfo(obj, type) {
  const box = document.getElementById('sc-info');
  const nm = document.getElementById('sc-info-name');
  const dt = document.getElementById('sc-info-detail');
  const co = document.getElementById('sc-info-coords');
  let name = obj.n || 'Unnamed';
  if (obj.cn) name += '  (' + obj.cn + ')';
  nm.textContent = name;
  const aa = scRaDecToAltAz(obj.r, obj.d);
  let detail = '';
  if (type === 'dso') detail = (obj.t||'DSO') + '  Mag ' + obj.m;
  else if (type === 'planet') detail = 'Solar System';
  else detail = 'Star  Mag ' + obj.m;
  dt.textContent = detail;
  co.textContent = 'Alt ' + aa.alt.toFixed(1) + '\u00b0  Az ' + aa.az.toFixed(1) + '\u00b0';
  box.style.display = 'block';
}

function scCloseInfo() {
  document.getElementById('sc-info').style.display = 'none';
  sc.sel = null; sc.selScreen = null;
  scRender();
}

function scGotoSelected() {
  if (!sc.sel) return;
  const name = sc.sel.n || '';
  // Switch to GoTo tab and fill in the search
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelector('[data-tab="goto"]').classList.add('active');
  document.getElementById('tab-goto').classList.add('active');
  const search = document.getElementById('goto-search');
  if (search) { search.value = name; search.dispatchEvent(new Event('input')); }
  // Also trigger GoTo directly if we have coords
  if (sc.sel.r !== undefined) {
    const raH = Math.floor(sc.sel.r);
    const raM = Math.floor((sc.sel.r - raH) * 60);
    const raS = Math.round(((sc.sel.r - raH) * 60 - raM) * 60);
    const decSign = sc.sel.d >= 0 ? '+' : '-';
    const decAbs = Math.abs(sc.sel.d);
    const decD = Math.floor(decAbs);
    const decM = Math.floor((decAbs - decD) * 60);
    const decS = Math.round(((decAbs - decD) * 60 - decM) * 60);
    document.getElementById('goto-ra').value = String(raH).padStart(2,'0')+':'+String(raM).padStart(2,'0')+':'+String(raS).padStart(2,'0');
    document.getElementById('goto-dec').value = decSign+String(decD).padStart(2,'0')+'*'+String(decM).padStart(2,'0')+':'+String(decS).padStart(2,'0');
  }
  scCloseInfo();
}

// ---- Toggle buttons ----
// Stop pointer events on buttons from reaching the canvas (prevents setPointerCapture stealing)
document.querySelectorAll('.sc-controls button').forEach(function(btn) {
  btn.addEventListener('pointerdown', function(e) { e.stopPropagation(); }, true);
  btn.addEventListener('pointermove', function(e) { e.stopPropagation(); }, true);
  btn.addEventListener('pointerup', function(e) { e.stopPropagation(); }, true);
  btn.addEventListener('touchstart', function(e) { e.stopPropagation(); }, {passive:true, capture:true});
  btn.addEventListener('touchend', function(e) { e.stopPropagation(); }, {passive:true, capture:true});
});
function scToggleGrid() { sc.grid=!sc.grid; document.getElementById('sc-btn-grid').classList.toggle('sc-active',sc.grid); scRender(); }
function scToggleConst() { sc.constLines=!sc.constLines; sc.constNames=sc.constLines; document.getElementById('sc-btn-const').classList.toggle('sc-active',sc.constLines); scRender(); }
function scToggleLabels() { sc.labels=!sc.labels; document.getElementById('sc-btn-labels').classList.toggle('sc-active',sc.labels); scRender(); }
function scToggleDSOs() { sc.dsoShow=!sc.dsoShow; document.getElementById('sc-btn-dso').classList.toggle('sc-active',sc.dsoShow); scRender(); }
function scToggleEcliptic() { sc.ecliptic=!sc.ecliptic; document.getElementById('sc-btn-ecliptic').classList.toggle('sc-active',sc.ecliptic); scRender(); }
function scToggleEquator() { sc.equator=!sc.equator; document.getElementById('sc-btn-equator').classList.toggle('sc-active',sc.equator); scRender(); }
function scFollowScope() {
  sc.follow=!sc.follow;
  document.getElementById('sc-btn-follow').classList.toggle('sc-active',sc.follow);
  if (sc.follow && sc.scope) { sc.vAlt = sc.scope.alt; sc.vAz = sc.scope.az; }
  scRender();
}
function scResetView() { sc.vAz=180; sc.vAlt=45; sc.fov=90; scUpdateFov(); scRender(); }
function scToggleFullscreen() {
  sc.fullscreen = !sc.fullscreen;
  const wrap = document.getElementById('sc-wrap');
  const card = wrap.closest('.tab-content');
  if (sc.fullscreen) {
    wrap.style.position = 'fixed';
    wrap.style.top = '0'; wrap.style.left = '0';
    wrap.style.width = '100vw'; wrap.style.height = '100vh';
    wrap.style.zIndex = '9999'; wrap.style.borderRadius = '0';
    if (card) card.style.overflow = 'visible';
  } else {
    wrap.style.position = 'relative';
    wrap.style.top = ''; wrap.style.left = '';
    wrap.style.width = '100%'; wrap.style.height = '';
    wrap.style.zIndex = ''; wrap.style.borderRadius = '12px';
    if (card) card.style.overflow = '';
  }
  document.getElementById('sc-btn-fullscreen').classList.toggle('sc-active', sc.fullscreen);
  window.dispatchEvent(new Event('resize'));
}

// ---- Data loading ----
async function scLoadData() {
  try {
    const d = await apiGet('/api/skychart/data');
    if (!d || d.error) return;
    sc.stars = d.stars || [];
    sc.extStars = d.ext_stars || [];
    sc.dsos = d.dsos || [];
    sc.planets = d.planets || [];
    sc.conLines = d.con_lines || [];
    sc.conLabels = d.con_labels || [];
    if (d.observer) { sc.obs.lat = d.observer.lat||0; sc.obs.lon = d.observer.lon||0; }
    if (d.telescope) { sc.scope = d.telescope; }
    sc.loaded = true;
    scRender();
    // Periodic refresh (telescope position + planets)
    setInterval(scRefresh, 2000);
  } catch(e) { console.error('Sky chart load error', e); }
}

async function scRefresh() {
  // Only refresh if sky chart tab is visible
  const tab = document.getElementById('tab-skychart');
  if (!tab || !tab.classList.contains('active')) return;
  try {
    const d = await apiGet('/api/skychart/data');
    if (!d) return;
    if (d.planets) sc.planets = d.planets;
    if (d.telescope) {
      sc.scope = d.telescope;
      if (sc.follow) { sc.vAlt = sc.scope.alt; sc.vAz = sc.scope.az; }
    }
    if (d.observer) { sc.obs.lat = d.observer.lat||0; sc.obs.lon = d.observer.lon||0; }
    scRender();
  } catch(e) {}
}

// Init sky chart when tab is first shown (lazy init)
let _scInited = false;
const _origTabClick = null;
document.querySelectorAll('.tab').forEach(t => {
  const orig = t.onclick;
  t.addEventListener('click', () => {
    if (t.dataset.tab === 'skychart' && !_scInited) {
      _scInited = true;
      setTimeout(scInit, 100);
    }
  });
});

// ============================================================
// ENHANCEMENT 1: Bottom Navigation System
// ============================================================
// Maps bottom nav sections to their sub-tabs: {label, tabId}
const NAV_SUB_TABS = {
  control: [
    {label: 'Slew', tab: 'control'},
    {label: 'GoTo', tab: 'goto'},
    {label: 'Sky Chart', tab: 'skychart'}
  ],
  imaging: [
    {label: 'Live View', tab: 'camera'}
  ],
  tracking: [
    {label: 'Tracking', tab: 'tracking'},
    {label: 'Telemetry', tab: 'telemetry'}
  ],
  settings: [
    {label: 'Location', tab: 'weather'},
    {label: 'Log', tab: 'log'},
    {label: 'Help', tab: 'help'}
  ]
};

let _activeNav = 'control';

function switchNav(navId) {
  _activeNav = navId;
  // Update bottom nav active state
  document.querySelectorAll('.bnav-item').forEach(b => b.classList.toggle('active', b.dataset.nav === navId));
  // Build sub-tabs for this nav section
  const bar = document.getElementById('sub-tabs-bar');
  const subs = NAV_SUB_TABS[navId] || [];
  bar.innerHTML = '';
  if (subs.length > 1) {
    subs.forEach(function(s, i) {
      const el = document.createElement('div');
      el.className = 'sub-tab' + (i === 0 ? ' active' : '');
      el.textContent = s.label;
      el.dataset.subtab = s.tab;
      el.addEventListener('click', function() {
        bar.querySelectorAll('.sub-tab').forEach(function(x) { x.classList.remove('active'); });
        el.classList.add('active');
        switchToTab(s.tab);
      });
      bar.appendChild(el);
    });
  }
  // Switch to the first tab in this nav section
  if (subs.length > 0) {
    switchToTab(subs[0].tab);
  }
  // Haptic feedback
  if (navigator.vibrate) navigator.vibrate(15);
  // Scroll to top
  window.scrollTo({top: 0, behavior: 'smooth'});
}

function switchToTab(tabId) {
  // Handle special cases
  if (tabId === 'skychart2') tabId = 'skychart';
  // Show/hide legacy tab-content divs
  document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
  const target = document.getElementById('tab-' + tabId);
  if (target) target.classList.add('active');
  // Update legacy tab active states (for compatibility with existing JS)
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tabId));
  // Lazy-init sky chart
  if (tabId === 'skychart' && !_scInited) {
    _scInited = true;
    setTimeout(scInit, 100);
  }
}

// ============================================================
// ENHANCEMENT 2: Night Vision (Red) Mode
// ============================================================
let _nightMode = false;
let _nightBrightness = 0; // 0 = no dimming, 1 = fully black

function toggleNightMode() {
  _nightMode = !_nightMode;
  document.documentElement.classList.toggle('nightmode', _nightMode);
  // Disable light mode if night mode is on
  if (_nightMode) document.documentElement.classList.remove('lightmode');
  const btn = document.getElementById('btn-nightmode');
  if (btn) btn.classList.toggle('active', _nightMode);
  // Apply default dimming in night mode
  const dimmer = document.getElementById('night-dimmer');
  if (dimmer) dimmer.style.background = _nightMode ? 'rgba(0,0,0,0.15)' : 'rgba(0,0,0,0)';
  toast(_nightMode ? 'Night vision ON - red only' : 'Night vision OFF', 'info');
  // Save preference
  try { localStorage.setItem('tw_nightmode', _nightMode ? '1' : '0'); } catch(e) {}
}

// ENHANCEMENT 4: Status strip is updated via the pollStatus hook below.

// ============================================================
// ENHANCEMENT 6: Recent Targets
// ============================================================
let _recentTargets = [];
const MAX_RECENT = 10;

function addRecentTarget(name, ra, dec) {
  if (!name || name === '--') return;
  // Remove duplicate
  _recentTargets = _recentTargets.filter(t => t.name !== name);
  // Add to front
  _recentTargets.unshift({
    name: name,
    ra: ra || '',
    dec: dec || '',
    time: new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})
  });
  // Trim
  if (_recentTargets.length > MAX_RECENT) _recentTargets.pop();
  // Save
  try { localStorage.setItem('tw_recent_targets', JSON.stringify(_recentTargets)); } catch(e) {}
  renderRecentTargets();
}

function renderRecentTargets() {
  const container = document.getElementById('recent-targets');
  const list = document.getElementById('recent-targets-list');
  if (!container || !list) return;
  if (_recentTargets.length === 0) {
    container.style.display = 'none';
    return;
  }
  container.style.display = '';
  list.innerHTML = '';
  for (const t of _recentTargets) {
    const div = document.createElement('div');
    div.className = 'recent-item';
    div.innerHTML = '<span class="ri-name">' + esc(t.name) + '</span>'
      + '<span class="ri-time">' + t.time + '</span>'
      + '<button class="ri-goto">GoTo</button>';
    div.querySelector('.ri-goto').addEventListener('click', function(e) {
      e.stopPropagation();
      document.getElementById('goto-search').value = t.name;
      if (t.ra && t.dec) {
        document.getElementById('goto-ra').value = t.ra;
        document.getElementById('goto-dec').value = t.dec;
      }
      doGoto();
    });
    div.addEventListener('click', function() {
      document.getElementById('goto-search').value = t.name;
    });
    list.appendChild(div);
  }
}

// Load saved recent targets
try {
  const saved = localStorage.getItem('tw_recent_targets');
  if (saved) _recentTargets = JSON.parse(saved);
  renderRecentTargets();
} catch(e) {}

// Hook into doGoto to record recent targets
const _origDoGoto = window.doGoto;
window.doGoto = function() {
  const search = document.getElementById('goto-search').value.trim();
  const ra = document.getElementById('goto-ra').value.trim();
  const dec = document.getElementById('goto-dec').value.trim();
  const name = search || (ra && dec ? 'RA ' + ra : '');
  if (name) addRecentTarget(name, ra, dec);
  return _origDoGoto.apply(this, arguments);
};

// ============================================================
// ENHANCEMENT 8: Accessibility Toggles
// ============================================================
function toggleLargeText() {
  document.documentElement.classList.toggle('largetext');
  const active = document.documentElement.classList.contains('largetext');
  document.getElementById('btn-largetext').classList.toggle('active', active);
  toast(active ? 'Large text ON' : 'Large text OFF', 'info');
  try { localStorage.setItem('tw_largetext', active ? '1' : '0'); } catch(e) {}
}

function toggleHighContrast() {
  document.documentElement.classList.toggle('highcontrast');
  const active = document.documentElement.classList.contains('highcontrast');
  document.getElementById('btn-highcontrast').classList.toggle('active', active);
  toast(active ? 'High contrast ON' : 'High contrast OFF', 'info');
  try { localStorage.setItem('tw_highcontrast', active ? '1' : '0'); } catch(e) {}
}

// ============================================================
// ENHANCEMENT 9: Chart.js Offline Fallback
// ============================================================
if (typeof Chart === 'undefined') {
  // Chart.js failed to load from CDN (likely on telescope WiFi with no internet)
  // Show a warning but don't break the app
  console.warn('Chart.js not loaded (no internet?). Telemetry charts will be unavailable.');
  window._chartUnavailable = true;
  window.Chart = function() { return { destroy: function(){}, update: function(){}, data: {datasets:[]} }; };
  window.Chart.register = function() {};
}

// ============================================================
// ENHANCEMENT 10: Emergency Stop
// ============================================================
function emergencyStop() {
  // Immediately stop all motion
  if (navigator.vibrate) navigator.vibrate([100, 50, 100, 50, 200]);
  // Fire multiple stop commands in parallel for safety
  Promise.all([
    apiPost('/api/slew/stop'),
    apiPost('/api/focuser/stop'),
    apiPost('/api/derotator/stop'),
    apiPost('/api/rotator/stop')
  ]).then(function() {
    toast('EMERGENCY STOP - all motion halted', 'err');
  }).catch(function() {
    toast('EMERGENCY STOP sent (some commands may have failed)', 'err');
  });
}

// ============================================================
// Hook into existing pollStatus for status strip updates
// ============================================================
// We patch pollStatus to also update the status strip.
// The original pollStatus already fetches /api/status, so we
// intercept its result instead of making a redundant API call.
(function() {
  const _origPoll = window.pollStatus;
  window.pollStatus = async function() {
    // Call the original pollStatus
    await _origPoll.apply(this, arguments);
    // Then update the status strip from a lightweight read
    // (The original pollStatus already updated the DOM,
    //  so we just read the values it set.)
    try {
      const ssRa = document.getElementById('p-ra');
      const ssDec = document.getElementById('p-dec');
      const ssAlt = document.getElementById('p-alt');
      const ssAz = document.getElementById('p-az');
      const tDrift = document.getElementById('t-drift');
      const tRms = document.getElementById('t-rms');
      setText('ss-ra', ssRa ? ssRa.textContent : '--');
      setText('ss-dec', ssDec ? ssDec.textContent : '--');
      setText('ss-alt', ssAlt ? ssAlt.textContent : '--');
      setText('ss-az', ssAz ? ssAz.textContent : '--');
      setText('ss-rms', tRms ? tRms.textContent : '--');
      // Tracking state from the header dot
      const dot = document.getElementById('hdr-dot');
      const ssTrack = document.getElementById('ss-tracking');
      if (dot && ssTrack) {
        if (dot.classList.contains('dot-green')) {
          ssTrack.textContent = 'ON'; ssTrack.style.color = 'var(--green)';
        } else if (dot.classList.contains('dot-yellow')) {
          ssTrack.textContent = 'SLEW'; ssTrack.style.color = 'var(--yellow)';
        } else {
          ssTrack.textContent = 'OFF'; ssTrack.style.color = 'var(--red)';
        }
      }
    } catch(e) {}
  };
})();

// ============================================================
// Restore saved preferences on load
// ============================================================
(function restorePrefs() {
  try {
    if (localStorage.getItem('tw_nightmode') === '1') {
      _nightMode = true;
      document.documentElement.classList.add('nightmode');
      const btn = document.getElementById('btn-nightmode');
      if (btn) btn.classList.add('active');
      const dimmer = document.getElementById('night-dimmer');
      if (dimmer) dimmer.style.background = 'rgba(0,0,0,0.15)';
    }
    if (localStorage.getItem('tw_largetext') === '1') {
      document.documentElement.classList.add('largetext');
      const btn = document.getElementById('btn-largetext');
      if (btn) btn.classList.add('active');
    }
    if (localStorage.getItem('tw_highcontrast') === '1') {
      document.documentElement.classList.add('highcontrast');
      const btn = document.getElementById('btn-highcontrast');
      if (btn) btn.classList.add('active');
    }
  } catch(e) {}
})();

// ============================================================
// Initialize sub-tabs + status strip on page load
// ============================================================
// Build initial sub-tabs for the default nav (control)
switchNav('control');
// Set the control tab as active by default
switchToTab('control');

</script>

</body>
</html>"""
