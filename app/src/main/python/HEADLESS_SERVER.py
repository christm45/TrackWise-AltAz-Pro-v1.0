"""
TrackWise-AltAzPro -- Headless Telescope Controller.

Runs the full telescope control stack **without tkinter** -- controlled
entirely through the web UI at ``http://<host>:8080``.

Architecture
------------
Instead of tkinter ``StringVar``/``BooleanVar``, this module uses a plain
``HeadlessVar`` class with the same ``.get()`` / ``.set()`` API.  A fake
``root`` provides ``.after()`` via :func:`threading.Timer` so that the web
server's ``execute_command()`` works unchanged.

The update loop runs in a background thread at ~1 Hz.  Headless mode
skips tkinter, matplotlib, and camera-preview rendering -- saving
significant CPU compared to the desktop GUI.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Core subsystem imports (no tkinter)
# ---------------------------------------------------------------------------
from lx200_protocol import LX200Protocol
from telescope_bridge import TelescopeBridge
from mount_protocol import (
    MountProtocol, get_protocol, list_protocols, CommandResult,
)
from telescope_simulator import TelescopeSimulator
from realtime_tracking import RealTimeTrackingController, FastPlateSolver
from auto_platesolve import AutoPlateSolver
from crash_recovery import CrashRecoveryManager, collect_app_state
from telescope_logger import get_logger, setup_logging
from web_server import TelescopeWebServer
from auto_alignment import AutoAlignment
from session_recorder import SessionRecorder

try:
    from config_manager import ConfigManager, DEFAULT_LATITUDE, DEFAULT_LONGITUDE
except ImportError:
    ConfigManager = None  # type: ignore[assignment, misc]
    DEFAULT_LATITUDE = 48.8566
    DEFAULT_LONGITUDE = 2.3522

try:
    from weather_service import OpenMeteoService
except ImportError:
    OpenMeteoService = None  # type: ignore[assignment, misc]


_logger = get_logger(__name__)


# ===================================================================
# HeadlessVar -- drop-in replacement for tkinter StringVar / BooleanVar
# ===================================================================

class HeadlessVar:
    """Thread-safe variable with the same .get()/.set() API as tkinter vars."""

    __slots__ = ("_value", "_lock")

    def __init__(self, value: Any = ""):
        self._value = value
        self._lock = threading.Lock()

    def get(self) -> Any:
        with self._lock:
            return self._value

    def set(self, value: Any) -> None:
        with self._lock:
            self._value = value


# ===================================================================
# FakeRoot -- replacement for tk.Tk that provides .after()
# ===================================================================

class FakeRoot:
    """Minimal stand-in for ``tk.Tk`` used only for ``root.after()``."""

    def after(self, ms: int, callback) -> None:  # noqa: D401
        """Schedule *callback* to run (immediately, on the calling thread)."""
        try:
            callback()
        except Exception:
            _logger.debug("FakeRoot.after callback failed", exc_info=True)

    def destroy(self) -> None:
        pass


# ===================================================================
# HeadlessTelescopeApp
# ===================================================================

class HeadlessTelescopeApp:
    """Full telescope control stack without tkinter.

    Exposes the same attribute interface that :class:`TelescopeWebServer`
    reads (``protocol``, ``tracking``, ``*_var`` variables, action methods).
    """

    def __init__(
        self,
        connection_type: str = "USB",
        port: str = "",
        baudrate: int = 9600,
        wifi_ip: str = "",
        wifi_port: int = 0,
        simulator: bool = False,
        web_port: int = 8080,
        latitude: float = DEFAULT_LATITUDE,
        longitude: float = DEFAULT_LONGITUDE,
    ) -> None:
        self.root = FakeRoot()

        # ---- Core subsystems ----
        self.protocol = LX200Protocol()
        # RA/Dec are derived from Alt/Az (home = 0°/0° = North horizon).
        # Once connected, on_altaz_update overwrites with real mount position.
        self.telescope_bridge = TelescopeBridge()
        self.telescope_bridge.on_log = lambda msg: self._log(f"[Bridge] {msg}", "info")
        self.telescope_simulator = TelescopeSimulator()
        self._simulator_active = False
        self.crash_recovery = CrashRecoveryManager()
        self.tracking = RealTimeTrackingController()
        self.plate_solver = FastPlateSolver()
        self.auto_solver = AutoPlateSolver()

        # Config
        self.config_manager = None
        if ConfigManager is not None:
            try:
                self.config_manager = ConfigManager("telescope_config.json")
            except Exception:
                _logger.warning("ConfigManager init failed", exc_info=True)

        # Weather
        self.weather_service = None
        if OpenMeteoService is not None:
            try:
                self.weather_service = OpenMeteoService()
            except Exception:
                _logger.warning("WeatherService init failed", exc_info=True)

        # ---- Variables (HeadlessVar replaces tkinter vars) ----
        self._init_variables(
            connection_type=connection_type,
            port=port,
            baudrate=baudrate,
            wifi_ip=wifi_ip,
            wifi_port=wifi_port,
            latitude=latitude,
            longitude=longitude,
        )

        # ---- Threading state ----
        self._connection_lock = threading.Lock()
        self.is_solving = False
        self.auto_solve_mode = "none"
        self.solve_thread: Optional[threading.Thread] = None
        self._debug_counter = 0
        self._position_update_counter = 0

        # ---- Web server ----
        self.web_server = TelescopeWebServer(self, port=web_port)

        # ---- Auto-alignment engine ----
        self.auto_alignment = AutoAlignment(self)

        # ---- Session recorder ----
        self.session_recorder = SessionRecorder(self)

        # ---- Update loop ----
        self._running = False
        self._update_thread: Optional[threading.Thread] = None

        # Auto-connect if requested
        if simulator:
            self._toggle_simulator()

    # ------------------------------------------------------------------
    # Variable initialization
    # ------------------------------------------------------------------

    def _init_variables(self, **kw):
        """Create all HeadlessVar instances (mirrors main_realtime._init_variables)."""
        # Position
        self.ra_var = HeadlessVar("00h 00m 00s")
        self.dec_var = HeadlessVar("+00\u00b0 00' 00\"")
        self.alt_var = HeadlessVar("45.00\u00b0")
        self.az_var = HeadlessVar("180.00\u00b0")
        self.rate_alt_var = HeadlessVar("0.000 \"/s")
        self.rate_az_var = HeadlessVar("0.000 \"/s")

        # GoTo target (what we're slewing to / looking at)
        self.goto_target_name_var = HeadlessVar("")
        self.goto_target_ra_var = HeadlessVar("")
        self.goto_target_dec_var = HeadlessVar("")
        self.goto_target_alt_var = HeadlessVar("")
        self.goto_target_az_var = HeadlessVar("")
        # Store target RA/Dec as floats for live Alt/Az recomputation
        self._goto_target_ra_hours = None   # float or None
        self._goto_target_dec_deg = None    # float or None

        # Statistics
        self.solve_time_var = HeadlessVar("-- ms")
        self.solve_rate_var = HeadlessVar("-- Hz")
        self.drift_var = HeadlessVar("-- \"/s")
        self.rms_var = HeadlessVar("-- \"")
        self.ml_samples_var = HeadlessVar("0")
        self.auto_solve_status_var = HeadlessVar("Stopped")

        # ASTAP
        astap_default = "/usr/bin/astap" if sys.platform != "win32" else r"C:\Program Files\astap\astap.exe"
        self.astap_path_var = HeadlessVar(astap_default)
        self.solve_interval_var = HeadlessVar("4.0")
        self.solve_mode_var = HeadlessVar("camera")

        # Camera/capture
        self.watch_folder_var = HeadlessVar("")
        self.camera_index_var = HeadlessVar("0")
        self.ascom_camera_id_var = HeadlessVar("")
        self.ascom_camera_name_var = HeadlessVar("No camera selected")
        self.ascom_exposure_var = HeadlessVar("0.5")
        self.ascom_gain_var = HeadlessVar("100")
        self.ascom_binning_var = HeadlessVar("2")
        self.save_images_var = HeadlessVar(False)
        self.save_folder_var = HeadlessVar("")
        self.save_format_var = HeadlessVar("fits")

        # Mount drive type and PEC
        self.mount_drive_type_var = HeadlessVar(
            kw.get("drive_type", "planetary_gearbox"))
        self.pec_enabled_var = HeadlessVar(True)
        self.pec_status_var = HeadlessVar("Learning...")
        self.pec_periods_var = HeadlessVar("--")
        self.pec_correction_var = HeadlessVar("0.00 \"/s")
        self.flexure_learning_var = HeadlessVar(True)

        # Location
        self.lat_var = HeadlessVar(str(kw.get("latitude", DEFAULT_LATITUDE)))
        self.lon_var = HeadlessVar(str(kw.get("longitude", DEFAULT_LONGITUDE)))

        # Weather
        self.weather_location_var = HeadlessVar("Not configured")
        self.weather_temp_var = HeadlessVar("--\u00b0C")
        self.weather_pressure_var = HeadlessVar("-- hPa")
        self.weather_humidity_var = HeadlessVar("--%")
        self.weather_cloud_var = HeadlessVar("--%")
        self.weather_wind_var = HeadlessVar("-- km/h")
        self.weather_wind_dir_var = HeadlessVar("--")
        self.weather_gusts_var = HeadlessVar("-- km/h")
        self.weather_dewpoint_var = HeadlessVar("--\u00b0C")
        self.weather_conditions_var = HeadlessVar("--")
        self.weather_observing_var = HeadlessVar("--")
        self.weather_dew_risk_var = HeadlessVar("--")
        self.weather_status_var = HeadlessVar("Not configured")
        self.current_weather = None

        # Connection
        self.usb_port_var = HeadlessVar(kw.get("port", ""))
        self.usb_baudrate_var = HeadlessVar(str(kw.get("baudrate", 9600)))
        self.connection_type_var = HeadlessVar(kw.get("connection_type", "USB"))
        self.wifi_ip_var = HeadlessVar(kw.get("wifi_ip", ""))
        self.wifi_port_var = HeadlessVar(str(kw.get("wifi_port", "")))
        self.mount_protocol_var = HeadlessVar("lx200")  # 'lx200' or 'nexstar'
        self.usb_connected_var = HeadlessVar(False)
        self.usb_status_var = HeadlessVar("Disconnected")
        self.sim_active_var = HeadlessVar(False)

        # Dock controls
        self.telescope_speed_var = HeadlessVar("4")
        self.telescope_status_var = HeadlessVar("Stopped")
        self.focuser_position_var = HeadlessVar("--")
        self.focuser_speed_var = HeadlessVar("5")
        self.focuser_status_var = HeadlessVar("Stopped")
        self.derotator_angle_var = HeadlessVar("0.0\u00b0")
        self.derotator_speed_var = HeadlessVar("1.0")
        self.derotator_status_var = HeadlessVar("Stopped")

        # Derotator software angle tracking state
        self._derotator_rotating = False
        self._derotator_rate = 0.0       # degrees/sec (+ = CW, - = CCW)
        self._derotator_angle = 0.0      # accumulated angle in degrees [0, 360)
        self._derotator_last_time = None  # time.time() when last updated

        # ----- OnStep extended features -----

        # Park state
        self.park_state_var = HeadlessVar("Unknown")  # 'Parked', 'Not Parked', 'Parking', 'Park Failed', 'Unknown'

        # Tracking rate
        self.tracking_rate_var = HeadlessVar("Sidereal")  # 'Sidereal', 'Lunar', 'Solar', 'King'
        self.tracking_enabled_var = HeadlessVar(False)

        # Mount-side PEC (OnStep hardware PEC, separate from software PEC)
        self.mount_pec_status_var = HeadlessVar("--")  # Idle/Playing/Recording/Ready etc
        self.mount_pec_recorded_var = HeadlessVar(False)

        # Firmware info
        self.firmware_name_var = HeadlessVar("--")
        self.firmware_version_var = HeadlessVar("--")
        self.firmware_mount_type_var = HeadlessVar("--")

        # Backlash
        self.backlash_ra_var = HeadlessVar("--")  # arcsec or '--'
        self.backlash_dec_var = HeadlessVar("--")

        # Mount limits
        self.horizon_limit_var = HeadlessVar("--")  # degrees or '--'
        self.overhead_limit_var = HeadlessVar("--")

        # Auxiliary features (OnStepX slot-based, up to 8 slots)
        self._auxiliary_features = []  # list of dicts: {slot, name, purpose, value}

        # Extended focuser state
        self.focuser_target_var = HeadlessVar("--")
        self.focuser_temperature_var = HeadlessVar("--")
        self.focuser_tcf_var = HeadlessVar(False)
        self.focuser_selected_var = HeadlessVar("1")  # selected focuser 1-6

        # Rotator state (OnStepX hardware rotator)
        self.rotator_angle_var = HeadlessVar("--")
        self.rotator_status_var = HeadlessVar("Stopped")
        self.rotator_derotating_var = HeadlessVar(False)

        # NexStar/SynScan-specific features (app-side)
        self.guide_rate_var = HeadlessVar("7.5")  # arcsec/sec
        self.speed_comp_ppm_var = HeadlessVar("0.0")  # ppm
        self.hibernate_status_var = HeadlessVar("No saved position")

        # Firmware/status polling control
        self._firmware_queried = False  # query once on connect
        self._onstep_poll_counter = 0   # throttle polling to every N ticks

        # Load saved config
        if self.config_manager:
            try:
                lat = str(self.config_manager.get("location.latitude", DEFAULT_LATITUDE))
                lon = str(self.config_manager.get("location.longitude", DEFAULT_LONGITUDE))
                self.lat_var.set(lat)
                self.lon_var.set(lon)
                self.astap_path_var.set(
                    self.config_manager.get("astap.path", self.astap_path_var.get())
                )
                self.solve_interval_var.set(
                    str(self.config_manager.get("astap.solve_interval", 4.0))
                )
            except Exception:
                pass

        # Propagate observer location to the protocol for coordinate conversion
        # (Without this, protocol defaults to Paris 48.8566, 2.3522)
        try:
            self.protocol.latitude = float(self.lat_var.get())
            self.protocol.longitude = float(self.lon_var.get())
        except (ValueError, TypeError):
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_active_bridge(self):
        if self._simulator_active:
            return self.telescope_simulator
        return self.telescope_bridge

    @staticmethod
    def _parse_lx200_dms(s: str) -> float:
        """Parse an LX200 DMS string (sDD*MM:SS# or DDD*MM:SS#) to decimal degrees.

        Handles signed (+/-) and unsigned formats, with '*' or degree symbols
        as separators. Returns 0.0 if parsing fails.
        """
        s = s.strip().rstrip('#')
        if not s:
            return 0.0
        try:
            sign = 1.0
            if s[0] in ('+', '-'):
                sign = -1.0 if s[0] == '-' else 1.0
                s = s[1:]
            s = s.replace('\xb0', '*').replace('\xdf', '*').replace("'", ':')
            parts = s.replace('*', ':').split(':')
            d = float(parts[0])
            m = float(parts[1]) if len(parts) > 1 else 0.0
            sec = float(parts[2]) if len(parts) > 2 else 0.0
            return sign * (d + m / 60.0 + sec / 3600.0)
        except (ValueError, IndexError):
            return 0.0

    def _wire_position_callback(self, bridge):
        """Wire position callbacks on *bridge* so that position data flows
        from the bridge/simulator read-loop into ``self.protocol`` and the
        HeadlessVar display variables.

        The main callback receives 5 arguments:
            (alt_str, az_str, is_slewing, ra_str=None, dec_str=None)
        When ra_str/dec_str are provided (from mount :GR#/:GD#), they are
        used directly.  Otherwise RA/Dec are derived locally from Alt/Az.
        """
        app = self  # capture for closure

        def _on_altaz_update(alt_str, az_str, is_slewing=False,
                             ra_str=None, dec_str=None):
            # Parse Alt/Az if available (may be empty strings)
            if alt_str:
                alt_deg = app._parse_lx200_dms(alt_str)
                app.protocol.alt_degrees = alt_deg
            else:
                alt_deg = app.protocol.alt_degrees

            if az_str:
                az_deg = app._parse_lx200_dms(az_str)
                app.protocol.az_degrees = az_deg
            else:
                az_deg = app.protocol.az_degrees

            app.protocol.is_slewing = is_slewing

            # RA/Dec: prefer mount's own values if available
            got_mount_radec = False
            if ra_str and dec_str:
                try:
                    # :GR# returns HH:MM:SS.s# — parse as hours
                    ra_h = app._parse_lx200_dms(ra_str)  # d+m/60+s/3600 works for HH:MM:SS
                    # :GD# returns sDD*MM'SS# — parse as degrees
                    dec_d = app._parse_lx200_dms(dec_str)
                    if 0 <= ra_h < 24 and -90 <= dec_d <= 90:
                        app.protocol.ra_hours = ra_h
                        app.protocol.dec_degrees = dec_d
                        got_mount_radec = True
                except Exception:
                    pass

            # Fallback: derive RA/Dec from Alt/Az locally
            if not got_mount_radec:
                try:
                    ra_h, dec_d = app.protocol._alt_az_to_ra_dec(alt_deg, az_deg)
                    app.protocol.ra_hours = ra_h
                    app.protocol.dec_degrees = dec_d
                except Exception:
                    pass

            # Periodic debug log (every 10th callback)
            app._position_update_counter += 1
            if app._position_update_counter % 10 == 0:
                slew_tag = " [SLEWING]" if is_slewing else ""
                src = "mount" if got_mount_radec else "local"
                _logger.info(
                    "Position #%d: Alt=%.2f Az=%.2f RA=%.4fh Dec=%.2f [%s]%s",
                    app._position_update_counter, alt_deg, az_deg,
                    app.protocol.ra_hours, app.protocol.dec_degrees,
                    src, slew_tag,
                )

        bridge.on_altaz_update = _on_altaz_update

        # --- Focuser position callback ---
        def _on_focuser_position(pos_str: str):
            pos_str = pos_str.strip().rstrip('#')
            try:
                steps = int(pos_str)
                app.focuser_position_var.set(f"{steps}")
            except (ValueError, TypeError):
                app.focuser_position_var.set(pos_str)

        bridge.on_focuser_position = _on_focuser_position

    def _log(self, message: str, tag: str = "info"):
        """Log to web server + file."""
        if self.web_server:
            self.web_server.push_log(message, tag)
        level = {"error": _logger.error, "warning": _logger.warning}.get(
            tag, _logger.info
        )
        level(message)

    # ------------------------------------------------------------------
    # Update loop (background thread, ~2 Hz)
    # ------------------------------------------------------------------

    def _update_loop_thread(self):
        """Background update loop refreshing position and stats.

        Dynamically adjusts from 0.5 Hz to 2 Hz based on CPU load.
        Under light load we tick faster for snappier UI updates;
        when the device is busy (plate-solving, FFT analysis) we back off
        to keep the system responsive.
        """
        self._status_log_counter = 0
        self._loop_interval = 1.0       # Start at 1 Hz
        self._loop_interval_min = 0.5   # 2 Hz max
        self._loop_interval_max = 2.0   # 0.5 Hz min
        while self._running:
            tick_start = time.monotonic()
            try:
                self._tick()
            except Exception as e:
                _logger.error("Update loop error: %s", e)
            tick_elapsed = time.monotonic() - tick_start

            # Adaptive interval: if the tick itself takes > 40% of the
            # budget, back off; if it takes < 10%, speed up.
            load_ratio = tick_elapsed / self._loop_interval
            if load_ratio > 0.4:
                self._loop_interval = min(
                    self._loop_interval * 1.25, self._loop_interval_max
                )
            elif load_ratio < 0.1:
                self._loop_interval = max(
                    self._loop_interval * 0.85, self._loop_interval_min
                )

            sleep_time = max(0, self._loop_interval - tick_elapsed)
            time.sleep(sleep_time)

    def _tick(self):
        """Single update tick."""
        # Tracking stats
        if self.tracking.is_running:
            stats = self.tracking.get_stats()
            self.drift_var.set(f"{stats.get('avg_correction', 0):.2f} \"/s")
            self.rms_var.set(f"{stats.get('kalman_rms_arcsec', 0):.2f} \"")
            self.ml_samples_var.set(str(stats.get('ml_samples', 0)))

        # Auto-solver stats
        if self.auto_solver.is_running:
            solver_stats = self.auto_solver.get_statistics()
            self.auto_solve_status_var.set(
                f"{solver_stats['successful_solves']}/{solver_stats['total_attempts']} "
                f"({solver_stats['success_rate']:.0f}%)"
            )

        # Position display
        self._update_position_display()

        # Derotator angle tracking (software-based)
        if self._derotator_rotating and self._derotator_last_time is not None:
            self._derotator_flush_angle()
            self.derotator_angle_var.set(f"{self._derotator_angle:.1f}\u00b0")

        # OnStep extended status polling (every ~5 ticks = ~5 seconds)
        if self.usb_connected_var.get() or self.sim_active_var.get():
            self._onstep_poll_counter += 1
            # One-time firmware query on connect
            if not self._firmware_queried:
                self._firmware_queried = True
                try:
                    self._query_firmware_info()
                    self._get_backlash()
                    self._get_limits()
                    self._discover_auxiliary_features()
                except Exception as e:
                    _logger.debug("Initial OnStep query error: %s", e)
            # Periodic status poll (~every 5 seconds)
            if self._onstep_poll_counter % 5 == 0:
                try:
                    self._poll_onstep_status()
                except Exception:
                    pass
            # Focuser/rotator polling (~every 10 seconds)
            if self._onstep_poll_counter % 10 == 0:
                try:
                    self._poll_focuser_extended()
                    self._poll_rotator()
                except Exception:
                    pass
        else:
            # Reset firmware query flag when disconnected
            if self._firmware_queried:
                self._firmware_queried = False
                self._onstep_poll_counter = 0

        # Update target Alt/Az (changes with time as Earth rotates)
        if self._goto_target_ra_hours is not None and self._goto_target_dec_deg is not None:
            try:
                t_alt, t_az = self.protocol._ra_dec_to_alt_az(
                    self._goto_target_ra_hours, self._goto_target_dec_deg
                )
                self.goto_target_alt_var.set(f"{t_alt:.1f}\u00b0")
                self.goto_target_az_var.set(f"{t_az:.1f}\u00b0")
            except Exception:
                pass

        # Periodic status log (~every 10 seconds = 10 ticks at 1 Hz)
        self._status_log_counter = getattr(self, "_status_log_counter", 0) + 1
        if self._status_log_counter >= 10:
            self._status_log_counter = 0
            connected = self.usb_connected_var.get()
            sim = self.sim_active_var.get()
            tracking = self.tracking.is_running
            alt = self.protocol.alt_degrees
            az = self.protocol.az_degrees
            if sim:
                src = "SIM"
            elif connected:
                ct = getattr(self.telescope_bridge, 'connection_type', '')
                src = "WiFi" if ct == 'tcp' else "USB" if ct == 'serial' else "TCP"
            else:
                src = "---"
            slewing = getattr(self.protocol, 'is_slewing', False)
            trk = "SLEWING" if slewing else ("TRACKING" if tracking else "idle")
            self._log(
                f"[{src}] Alt={alt:.2f} Az={az:.2f}  {trk}  |  "
                f"RA={self.ra_var.get()}  Dec={self.dec_var.get()}",
                "info",
            )

        # Web server state snapshot
        if self.web_server:
            self.web_server.update_state()

        # Crash checkpoint
        if self.crash_recovery.should_save():
            try:
                self.crash_recovery.save_checkpoint(collect_app_state(self))
            except Exception:
                pass

    def _update_position_display(self):
        """Format position strings from protocol values.

        Uses a cache to avoid re-formatting unchanged values on every tick.
        Position is rounded to display precision before comparison so tiny
        floating-point jitter doesn't trigger unnecessary string allocations.
        """
        ra = round(self.protocol.ra_hours, 4)
        dec = round(self.protocol.dec_degrees, 3)
        alt = round(self.protocol.alt_degrees, 2)
        az = round(self.protocol.az_degrees, 2)

        cache = getattr(self, "_pos_cache", None)
        if cache is None:
            cache = {}
            self._pos_cache = cache

        if cache.get("ra") != ra:
            cache["ra"] = ra
            ra_h = int(ra)
            ra_m = int((ra - ra_h) * 60)
            ra_s = ((ra - ra_h) * 60 - ra_m) * 60
            self.ra_var.set(f"{ra_h:02d}h {ra_m:02d}m {ra_s:04.1f}s")

        if cache.get("dec") != dec:
            cache["dec"] = dec
            sign = "+" if dec >= 0 else "-"
            ad = abs(dec)
            dd = int(ad)
            dm = int((ad - dd) * 60)
            ds = ((ad - dd) * 60 - dm) * 60
            self.dec_var.set(f"{sign}{dd:02d}\u00b0 {dm:02d}' {ds:04.1f}\"")

        if cache.get("alt") != alt:
            cache["alt"] = alt
            self.alt_var.set(f"{alt:.2f}\u00b0")

        if cache.get("az") != az:
            cache["az"] = az
            self.az_var.set(f"{az:.2f}\u00b0")

    # ------------------------------------------------------------------
    # Connection actions
    # ------------------------------------------------------------------

    def _toggle_usb_telescope(self):
        """Connect or disconnect the telescope."""
        with self._connection_lock:
            if self.usb_connected_var.get():
                # Disconnect
                try:
                    bridge = self._get_active_bridge()
                    if bridge.is_connected:
                        bridge.disconnect()
                except Exception as e:
                    self._log(f"Disconnect error: {e}", "error")
                self.usb_connected_var.set(False)
                self.usb_status_var.set("Disconnected")
                self.protocol.is_slewing = False
                self._log("Telescope disconnected", "warning")
            else:
                # Connect
                conn_type = self.connection_type_var.get()
                try:
                    # Wire position callback BEFORE connecting so the
                    # bridge's _read_loop can fire it immediately.
                    self._wire_position_callback(self.telescope_bridge)

                    # Wire disconnection callback so usb_connected_var
                    # stays in sync when the bridge drops (socket error,
                    # cable unplug, etc.)
                    app_ref = self
                    def _on_bridge_disconnected():
                        app_ref.usb_connected_var.set(False)
                        app_ref.usb_status_var.set("Disconnected (connection lost)")
                        app_ref.protocol.is_slewing = False
                        app_ref._log("Telescope connection lost", "error")
                    self.telescope_bridge.on_disconnected = _on_bridge_disconnected

                    if conn_type == "WiFi":
                        ip = self.wifi_ip_var.get().strip()
                        port_str = self.wifi_port_var.get().strip()
                        if not ip:
                            raise RuntimeError(
                                "WiFi IP address not set. Enter your "
                                "telescope's IP address first."
                            )
                        if not port_str:
                            raise RuntimeError(
                                "WiFi port not set. Enter your "
                                "telescope's TCP port first."
                            )
                        port = int(port_str)
                        success = self.telescope_bridge.connect(
                            "", 0, connection_type='tcp',
                            tcp_ip=ip, tcp_port=port,
                        )
                    else:
                        port_name = self.usb_port_var.get()
                        baud = int(self.usb_baudrate_var.get())
                        success = self.telescope_bridge.connect(
                            port_name, baud, connection_type='serial',
                        )
                    if not success:
                        raise RuntimeError("Bridge returned False")
                    self.usb_connected_var.set(True)
                    self.usb_status_var.set(f"Connected ({conn_type})")
                    self._log(f"Telescope connected via {conn_type}", "success")
                except Exception as e:
                    self.usb_status_var.set(f"Error: {e}")
                    self._log(f"Connection failed: {e}", "error")

    def _toggle_simulator(self):
        """Activate/deactivate the telescope simulator."""
        if self._simulator_active:
            try:
                self.telescope_simulator.disconnect()
            except Exception:
                pass
            self._simulator_active = False
            self.sim_active_var.set(False)
            self.usb_connected_var.set(False)
            self.usb_status_var.set("Disconnected")
            self.protocol.is_slewing = False
            self._log("Simulator deactivated", "warning")
        else:
            # Disconnect real hardware first
            if self.telescope_bridge.is_connected:
                try:
                    self.telescope_bridge.disconnect()
                except Exception:
                    pass
            try:
                # Wire position callback BEFORE connecting so the
                # simulator's _read_loop can fire it immediately.
                self._wire_position_callback(self.telescope_simulator)

                self.telescope_simulator.connect()
                self._simulator_active = True
                self.sim_active_var.set(True)
                self.usb_connected_var.set(True)
                self.usb_status_var.set("Simulator active")
                self._log("Simulator activated", "success")
            except Exception as e:
                self._log(f"Simulator error: {e}", "error")

    # ------------------------------------------------------------------
    # Tracking actions
    # ------------------------------------------------------------------

    def _toggle_tracking(self):
        if self.tracking.is_running:
            self.tracking.stop()
            self._log("Tracking stopped", "warning")
            # Auto-save session data when tracking stops
            if self.session_recorder.is_started:
                try:
                    path = self.session_recorder.save(auto=True)
                    if path:
                        self._log(f"Session auto-saved to {path}", "success")
                except Exception as e:
                    self._log(f"Session auto-save error: {e}", "error")
        else:
            try:
                lat = float(self.lat_var.get())
                lon = float(self.lon_var.get())
                interval = float(self.solve_interval_var.get())
                self.tracking.set_latitude(lat)
                self.tracking.longitude = lon
                self.tracking.plate_solve_interval = interval

                # Wire up callbacks (same as core/callbacks.py does for GUI)
                bridge = self._get_active_bridge()

                def send_command(cmd: str) -> str:
                    try:
                        if bridge.is_connected:
                            return bridge.send_command(cmd) or ""
                        elif self._simulator_active:
                            return self.protocol.process_command(cmd) or ""
                        return ""
                    except Exception as e:
                        self._log(f"Tracking command error: {e}", "error")
                        return ""

                self.tracking.send_command = send_command
                self.tracking.on_log = lambda msg: self._log(msg, "info")

                # --- Wire variable-rate Alt/Az if the protocol supports it ---
                # Protocols like NexStar/SynScan and ASCOM Alpaca can send
                # Alt/Az rates directly instead of converting to RA/Dec.
                mp = self.telescope_bridge.mount_protocol
                if mp.supports_variable_rate_altaz and bridge.is_connected:
                    def _var_rate_altaz(alt_rate: float, az_rate: float):
                        try:
                            if bridge.is_connected:
                                mp.send_variable_rate_altaz(
                                    alt_rate, az_rate, bridge.send_command
                                )
                        except Exception as e:
                            self._log(f"Variable-rate Alt/Az error: {e}", "error")

                    self.tracking.send_variable_rate_altaz = _var_rate_altaz
                    self._log(f"Using {mp.name} variable-rate Alt/Az tracking", "info")
                else:
                    self.tracking.send_variable_rate_altaz = None

                # --- Seed weather data for atmospheric corrections ---
                if self.weather_service and self.weather_service.cached_data:
                    wd = self.weather_service.cached_data
                    self.tracking.update_weather(
                        temperature_c=wd.temperature,
                        pressure_hpa=wd.pressure,
                        humidity_pct=wd.humidity,
                    )
                    self._log(
                        f"Tracking weather: {wd.temperature:.1f}C, "
                        f"{wd.pressure:.0f}hPa", "info"
                    )

                def on_position_update(ra, dec, alt, az):
                    if alt != 0.0 or az != 0.0:
                        self.protocol.ra_hours, self.protocol.dec_degrees = \
                            self.protocol._alt_az_to_ra_dec(alt, az)
                    else:
                        self.protocol.ra_hours = ra
                        self.protocol.dec_degrees = dec
                    self.protocol.alt_degrees = alt
                    self.protocol.az_degrees = az

                self.tracking.on_position_update = on_position_update
                self.tracking.start()
                self._log("Tracking started", "success")
                # Start session recording
                if not self.session_recorder.is_started:
                    self.session_recorder.start()
            except Exception as e:
                self._log(f"Tracking start error: {e}", "error")

    def _toggle_auto_solve(self):
        if self.auto_solver.is_running:
            self.auto_solver.stop()
            self.auto_solve_status_var.set("Stopped")
            self._log("Auto plate solving stopped", "warning")
        else:
            try:
                # Configure the solver
                self.auto_solver.astap_path = self.astap_path_var.get()
                self.auto_solver.solve_interval = float(self.solve_interval_var.get())

                # Calculate FOV from configured optics (focal length + sensor width)
                import math
                try:
                    fl = float(self.config_manager.get("solver.focal_length_mm", 0))
                    sw = float(self.config_manager.get("solver.sensor_width_mm", 0))
                    if fl > 0 and sw > 0:
                        fov = 2.0 * math.degrees(math.atan(sw / (2.0 * fl)))
                        self.auto_solver.fov_deg = round(fov, 4)
                        self.plate_solver.fov_deg = round(fov, 4)
                        self._log(f"Solver FOV: {fov:.2f} deg (FL={fl}mm, sensor={sw}mm)", "info")
                except Exception:
                    pass

                # Solve-complete callback: feed results into tracking
                def on_solve(result):
                    if result.success:
                        self.tracking.update_from_plate_solve(
                            result.ra_hours,
                            result.dec_degrees,
                            result.solve_time_ms,
                        )
                        self.solve_time_var.set(f"{result.solve_time_ms:.0f} ms")
                        stats = self.auto_solver.get_statistics()
                        if stats.get('avg_solve_time', 0) > 0:
                            rate = 1000 / stats['avg_solve_time']
                            self.solve_rate_var.set(f"{rate:.2f} Hz")

                self.auto_solver.on_solve_complete = on_solve
                self.auto_solver.on_log = lambda msg: self._log(msg, "info")

                # Start based on the selected mode
                mode = self.solve_mode_var.get()
                success = False

                if mode == "camera":
                    camera_idx = int(self.camera_index_var.get())
                    success = self.auto_solver.start_camera_mode(camera_idx)
                elif mode == "ascom":
                    camera_id = self.ascom_camera_id_var.get()
                    if not camera_id:
                        raise RuntimeError("Select an ASCOM camera first")
                    exposure = float(self.ascom_exposure_var.get())
                    gain = int(self.ascom_gain_var.get())
                    binning = int(self.ascom_binning_var.get())
                    success = self.auto_solver.start_ascom_mode(
                        camera_id=camera_id, exposure_sec=exposure,
                        gain=gain, binning=binning,
                    )
                elif mode == "folder":
                    folder = self.watch_folder_var.get()
                    if folder:
                        success = self.auto_solver.start_folder_watch_mode(folder)
                    else:
                        raise RuntimeError("Select a folder to watch")
                else:
                    raise RuntimeError(f"Unsupported solve mode: {mode}")

                if success:
                    self.auto_solve_status_var.set("In progress...")
                    self._log(f"Auto plate solving started (mode: {mode})", "success")
                else:
                    raise RuntimeError("Failed to start plate solving — check camera connection")
            except Exception as e:
                self._log(f"Auto solve error: {e}", "error")

    # ------------------------------------------------------------------
    # GoTo
    # ------------------------------------------------------------------

    def _goto_altaz_from_radec(self, ra_str: str, dec_str: str) -> str:
        """Parse RA/Dec strings and send GoTo via the bridge/protocol.

        Returns:
            Empty string on success, or an error message string on failure.
        """
        bridge = self._get_active_bridge()
        if not bridge.is_connected:
            self._log("GoTo failed: not connected", "error")
            return "Telescope not connected"
        try:
            self._log(
                f"GoTo request: RA={ra_str} Dec={dec_str} "
                f"(current Alt={self.protocol.alt_degrees:.2f} Az={self.protocol.az_degrees:.2f})",
                "cmd",
            )
            # Set target coordinates on local protocol state
            self.protocol.process_command(f":Sr{ra_str}#")
            self.protocol.process_command(f":Sd{dec_str}#")

            # Send GoTo via mount protocol
            mp = self.telescope_bridge.mount_protocol
            result = mp.goto_radec(ra_str, dec_str, bridge.send_command)

            if result.success:
                self._log(
                    f"GoTo accepted: RA={ra_str} Dec={dec_str} (mount slewing)",
                    "success",
                )
                return ""
            else:
                msg = (
                    f"GoTo refused by mount: {result.message} "
                    f"(check target visibility)"
                )
                self._log(
                    f"GoTo refused: RA={ra_str} Dec={dec_str} -- {result.message}",
                    "warning",
                )
                return msg
        except Exception as e:
            self._log(f"GoTo error: {e}", "error")
            return f"GoTo error: {e}"

    # ------------------------------------------------------------------
    # Telescope motion
    # ------------------------------------------------------------------

    def _slew_telescope(self, direction: str):
        speed = int(self.telescope_speed_var.get())
        self.protocol.slew_speed = speed
        dirs = {"N": (1, 0), "S": (-1, 0), "E": (0, 1), "W": (0, -1)}
        if direction in dirs:
            self.protocol.slew_alt_az(*dirs[direction])
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.slew(direction, speed, bridge.send_command)
        self.telescope_status_var.set(f"Slew {direction} @ {speed}")
        self._log(f"Slew {direction} (speed {speed})", "cmd")

    def _stop_telescope(self):
        self.protocol._stop_slew()
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.stop(bridge.send_command)
        self.telescope_status_var.set("Stopped")
        self._log("Telescope stopped", "warning")

    def _park_telescope(self):
        self.protocol.park()
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.park(bridge.send_command)
        self.telescope_status_var.set("Parking...")
        self._log("Telescope parking", "info")

    def _home_telescope(self):
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.home(bridge.send_command)
        self.telescope_status_var.set("Home reset")
        self._log(
            "Telescope home reset sent (:hF#). "
            "Position is now 0,0 -- this is normal. "
            "Use GoTo to slew to a new target.",
            "info",
        )

    # ------------------------------------------------------------------
    # Site / Time / Weather -> Telescope
    # ------------------------------------------------------------------

    def _send_site_to_telescope(self) -> list:
        """Send observer latitude, longitude, and UTC offset to the mount.

        Uses the active mount protocol (LX200 or NexStar) to format
        and send location commands.

        Returns a list of (command, response) tuples for diagnostics.
        """
        import datetime as _dt

        bridge = self._get_active_bridge()
        if not bridge.is_connected:
            self._log("Cannot send site: telescope not connected", "warning")
            return []

        try:
            lat = float(self.lat_var.get())
            lon = float(self.lon_var.get())
        except (ValueError, TypeError):
            self._log("Cannot send site: invalid lat/lon", "error")
            return []

        # Compute UTC offset (west-positive for OnStep, but the protocol handles it)
        now = _dt.datetime.now(_dt.timezone.utc).astimezone()
        _utc_off = now.utcoffset()
        utc_offset_east_sec = _utc_off.total_seconds() if _utc_off else 0.0
        utc_offset_west_h = -utc_offset_east_sec / 3600.0

        mp = self.telescope_bridge.mount_protocol
        result = mp.set_site(lat, lon, utc_offset_west_h, bridge.send_command)
        for cmd, resp in result.details:
            self._log(f"Site -> {cmd} => {resp}", "cmd")
        return result.details

    def _send_time_to_telescope(self) -> list:
        """Send current local date and time to the mount.

        Uses the active mount protocol.

        Returns a list of (command, response) tuples for diagnostics.
        """
        import datetime as _dt

        bridge = self._get_active_bridge()
        if not bridge.is_connected:
            self._log("Cannot send time: telescope not connected", "warning")
            return []

        now = _dt.datetime.now()
        mp = self.telescope_bridge.mount_protocol
        result = mp.set_time(now, bridge.send_command)
        for cmd, resp in result.details:
            self._log(f"Time -> {cmd} => {resp}", "cmd")
        return result.details

    def _send_weather_to_telescope(self) -> list:
        """Send weather data (temperature, pressure, humidity) to the mount.

        Uses the active mount protocol (LX200/OnStep supports this;
        NexStar does not -- the protocol returns a no-op).

        Returns a list of (command, response) tuples for diagnostics.
        """
        bridge = self._get_active_bridge()
        if not bridge.is_connected:
            self._log("Cannot send weather: telescope not connected", "warning")
            return []

        if self.weather_service is None or self.weather_service.cached_data is None:
            self._log("No weather data available to send to telescope", "info")
            return []

        data = self.weather_service.cached_data
        mp = self.telescope_bridge.mount_protocol
        result = mp.set_weather(
            data.temperature, data.pressure, data.humidity,
            bridge.send_command,
        )
        for cmd, resp in result.details:
            self._log(f"Weather -> {cmd} => {resp}", "cmd")

        # Feed weather data into the tracking controller for atmospheric
        # refraction and thermal drift corrections (runs continuously).
        if self.tracking is not None:
            self.tracking.update_weather(
                temperature_c=data.temperature,
                pressure_hpa=data.pressure,
                humidity_pct=data.humidity,
            )

        return result.details

    def _setup_telescope_site(self) -> dict:
        """Full telescope site setup: location + time + weather + home reset.

        Called when the user hits "Set Location" in the web UI.
        Uses the active mount protocol for all commands.

        Returns a dict with results for each step.
        """
        report = {"site": [], "time": [], "weather": [], "home": False}

        bridge = self._get_active_bridge()
        if not bridge.is_connected:
            self._log("Site setup: telescope not connected", "warning")
            return report

        self._log("--- Site setup started ---", "info")

        # 1) Send location (lat, lon, UTC offset)
        report["site"] = self._send_site_to_telescope()

        # 2) Send current date & time
        report["time"] = self._send_time_to_telescope()

        # 3) Send weather (if available)
        report["weather"] = self._send_weather_to_telescope()

        # 4) Reset home position
        mp = self.telescope_bridge.mount_protocol
        mp.home(bridge.send_command)
        report["home"] = True
        self._log("Home position reset", "cmd")

        self._log("--- Site setup complete ---", "success")
        self.telescope_status_var.set("Site configured")
        return report

    # ------------------------------------------------------------------
    # Focuser
    # ------------------------------------------------------------------

    def _move_focuser(self, direction: str):
        speed = int(self.focuser_speed_var.get())
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.focuser_move(direction, speed, bridge.send_command)
        # Update local protocol state (LX200 format)
        lx200_speed = max(1, min(4, (speed - 1) // 5 + 1))
        self.protocol.process_command(f":F{lx200_speed}#")
        self.protocol.process_command(f":F+#" if direction == "IN" else ":F-#")
        self.focuser_status_var.set(f"Moving {direction} @ {lx200_speed}")
        self._log(f"Focuser {direction} (speed {lx200_speed})", "cmd")

    def _stop_focuser(self):
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.focuser_stop(bridge.send_command)
        self.protocol.process_command(":FQ#")
        self.focuser_status_var.set("Stopped")
        self._log("Focuser stopped", "warning")

    # ------------------------------------------------------------------
    # Derotator
    # ------------------------------------------------------------------

    def _rotate_derotator(self, direction: str):
        speed = float(self.derotator_speed_var.get())
        # Update local protocol state
        cmd = f":DR+{speed}#" if direction == "CW" else f":DR-{speed}#"
        self.protocol.process_command(cmd)
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.derotator_rotate(direction, speed, bridge.send_command)

        # Start software angle tracking
        self._derotator_flush_angle()
        self._derotator_rotating = True
        self._derotator_rate = speed if direction == "CW" else -speed
        self._derotator_last_time = time.time()

        self.derotator_status_var.set(f"Rotating {direction} @ {speed}\u00b0/s")
        self._log(f"Derotator {direction} ({speed}\u00b0/s)", "cmd")

    def _stop_derotator(self):
        self.protocol.process_command(":DRQ#")
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.derotator_stop(bridge.send_command)

        # Stop software angle tracking, flush accumulated angle
        self._derotator_flush_angle()
        self._derotator_rotating = False
        self._derotator_last_time = None
        self.derotator_angle_var.set(f"{self._derotator_angle:.1f}\u00b0")

        self.derotator_status_var.set("Stopped")
        self._log(f"Derotator stopped at {self._derotator_angle:.1f}\u00b0", "warning")

    def _sync_derotator(self):
        self.protocol.process_command(":DR0#")
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.derotator_sync(bridge.send_command)

        # Reset angle to 0
        self._derotator_angle = 0.0
        self._derotator_rotating = False
        self._derotator_last_time = None
        self.derotator_angle_var.set("0.0\u00b0")
        self.derotator_status_var.set("Synced 0\u00b0")
        self._log("Derotator synchronized to 0\u00b0", "success")

    def _derotator_flush_angle(self):
        """Accumulate elapsed angle from the last update time."""
        if self._derotator_rotating and self._derotator_last_time is not None:
            now = time.time()
            elapsed = now - self._derotator_last_time
            self._derotator_angle = (self._derotator_angle + self._derotator_rate * elapsed) % 360
            self._derotator_last_time = now

    # ------------------------------------------------------------------
    # PEC
    # ------------------------------------------------------------------

    def _toggle_pec(self):
        enabled = self.pec_enabled_var.get()
        self.pec_enabled_var.set(not enabled)
        # Sync to tracking controller
        if hasattr(self, 'tracking') and self.tracking:
            self.tracking.pec_enabled = not enabled
        self._log(f"PEC {'disabled' if enabled else 'enabled'}", "info")

    def _set_drive_type(self, drive_type: str):
        """Change mount drive type and retune PEC parameters."""
        valid = ['worm_gear', 'planetary_gearbox', 'harmonic_drive',
                 'belt_drive', 'direct_drive']
        if drive_type not in valid:
            self._log(f"Invalid drive type: {drive_type}", "error")
            return
        self.mount_drive_type_var.set(drive_type)
        if hasattr(self, 'tracking') and self.tracking and hasattr(self.tracking, 'pec'):
            self.tracking.pec.set_drive_type(drive_type)
        # Persist to config
        if hasattr(self, 'config_manager') and self.config_manager:
            self.config_manager.set("mount.drive_type", drive_type)
            self.config_manager.save_config()
        self._log(f"Mount drive type set to: {drive_type}", "success")

    def _toggle_flexure_learning(self):
        """Toggle the flexure learning model on/off."""
        enabled = self.flexure_learning_var.get()
        self.flexure_learning_var.set(not enabled)
        if hasattr(self, 'tracking') and self.tracking:
            if hasattr(self.tracking, 'flexure_model'):
                self.tracking.flexure_model.is_learning = not enabled
        self._log(f"Flexure learning {'disabled' if enabled else 'enabled'}", "info")

    def _pec_save(self):
        if hasattr(self.tracking, 'pec'):
            try:
                self.tracking.pec.save("pec_model.json")
                self._log("PEC model saved", "success")
            except Exception as e:
                self._log(f"PEC save error: {e}", "error")

    def _pec_load(self):
        if hasattr(self.tracking, 'pec'):
            try:
                self.tracking.pec.load("pec_model.json")
                self._log("PEC model loaded", "success")
            except Exception as e:
                self._log(f"PEC load error: {e}", "error")

    def _pec_reset(self):
        if hasattr(self.tracking, 'pec'):
            self.tracking.pec.reset()
            self.pec_status_var.set("Learning...")
            self._log("PEC reset", "warning")

    # ------------------------------------------------------------------
    # OnStep Extended: Park / Unpark
    # ------------------------------------------------------------------

    def _unpark_telescope(self):
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.unpark(bridge.send_command)
        self.park_state_var.set("Not Parked")
        self.telescope_status_var.set("Unparked")
        self._log("Telescope unparked", "success")

    def _set_park_position(self):
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.set_park_position(bridge.send_command)
        self._log("Park position set to current coordinates", "success")

    # ------------------------------------------------------------------
    # OnStep Extended: Tracking Rate
    # ------------------------------------------------------------------

    def _set_tracking_rate(self, rate: str):
        """Set tracking rate: 'sidereal', 'lunar', 'solar', 'king'.

        Sends the standard LX200/OnStep command to the mount firmware and
        also notifies the real-time tracking controller so that SXTR offsets
        are computed relative to the correct base rate.
        """
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            try:
                mp.set_tracking_rate(rate, bridge.send_command)
                self.tracking_rate_var.set(rate.capitalize())
                self._log(f"Tracking rate set to {rate}", "success")
            except Exception as e:
                self._log(f"Set tracking rate failed: {e}", "error")
                return
        else:
            self._log("Cannot set tracking rate: not connected", "warning")
            return

        # Notify the real-time tracking controller so SXTR/SXTD offsets
        # are computed relative to the new base rate (not always sidereal).
        if hasattr(self, 'tracking') and self.tracking is not None:
            self.tracking.set_base_tracking_rate(rate)

    def _enable_tracking(self):
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.enable_tracking(bridge.send_command)
        self.tracking_enabled_var.set(True)
        self._log("Mount tracking enabled", "success")

    def _disable_tracking(self):
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.disable_tracking(bridge.send_command)
        self.tracking_enabled_var.set(False)
        self._log("Mount tracking disabled", "warning")

    # ------------------------------------------------------------------
    # OnStep Extended: Tracking Configuration
    # ------------------------------------------------------------------

    def _set_tracking_axis_mode(self, mode: int):
        """Set tracking axis mode: 1=single (RA only), 2=dual (both axes)."""
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            try:
                mp.set_tracking_axis_mode(mode, bridge.send_command)
                label = "dual-axis" if mode == 2 else "single-axis"
                self._log(f"Tracking axis mode: {label}", "success")
            except Exception as e:
                self._log(f"Set axis mode failed: {e}", "error")

    def _set_compensation_model(self, model: str):
        """Set compensation model: 'full', 'refraction', 'none'."""
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            try:
                mp.set_compensation_model(model, bridge.send_command)
                self._log(f"Compensation model: {model}", "success")
            except Exception as e:
                self._log(f"Set compensation failed: {e}", "error")

    def _adjust_sidereal_clock(self, direction: str):
        """Adjust master sidereal clock: '+', '-', or 'reset'."""
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            try:
                mp.adjust_sidereal_clock(direction, bridge.send_command)
                label = {'+':"increased", '-':"decreased", 'reset':"reset"}.get(direction, direction)
                self._log(f"Sidereal clock {label}", "success")
            except Exception as e:
                self._log(f"Sidereal clock adjust failed: {e}", "error")

    def _set_backlash(self, axis: str, arcsec: float):
        """Set backlash for axis ('ra' or 'dec') in arcseconds."""
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            try:
                mp.set_backlash(axis, arcsec, bridge.send_command)
                self._log(f"Backlash {axis.upper()}: {arcsec:.0f} arcsec", "success")
            except Exception as e:
                self._log(f"Set backlash failed: {e}", "error")

    def _get_backlash(self, axis: str) -> float:
        """Get backlash for axis ('ra' or 'dec') in arcseconds."""
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            try:
                return mp.get_backlash(axis, bridge.send_command)
            except Exception:
                return 0.0
        return 0.0

    # ------------------------------------------------------------------
    # OnStep Extended: Mount-side PEC
    # ------------------------------------------------------------------

    def _mount_pec_playback_start(self):
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.pec_playback_start(bridge.send_command)
        self.mount_pec_status_var.set("Playing")
        self._log("Mount PEC playback started", "success")

    def _mount_pec_playback_stop(self):
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.pec_playback_stop(bridge.send_command)
        self.mount_pec_status_var.set("Idle")
        self._log("Mount PEC playback stopped", "warning")

    def _mount_pec_record_start(self):
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.pec_record_start(bridge.send_command)
        self.mount_pec_status_var.set("Recording")
        self._log("Mount PEC recording armed", "success")

    def _mount_pec_record_stop(self):
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.pec_record_stop(bridge.send_command)
        self.mount_pec_status_var.set("Idle")
        self._log("Mount PEC recording stopped", "warning")

    def _mount_pec_clear(self):
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.pec_clear(bridge.send_command)
        self.mount_pec_status_var.set("Idle")
        self.mount_pec_recorded_var.set(False)
        self._log("Mount PEC buffer cleared", "warning")

    def _mount_pec_write_nv(self):
        """Write PEC data to non-volatile memory."""
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.pec_write_eeprom(bridge.send_command)
        self._log("Mount PEC data saved to NV memory", "success")

    def _mount_pec_read_nv(self):
        """Read PEC data from non-volatile memory."""
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.pec_read_eeprom(bridge.send_command)
        self._log("Mount PEC data loaded from NV memory", "success")

    # ------------------------------------------------------------------
    # OnStep Extended: Backlash
    # ------------------------------------------------------------------

    def _set_backlash(self, axis: str, value: int):
        """Set backlash for given axis ('ra'/'dec') in arcseconds."""
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            result = mp.set_backlash(axis, value, bridge.send_command)
            if result.success:
                if axis.lower() in ('ra', 'azm'):
                    self.backlash_ra_var.set(str(value))
                else:
                    self.backlash_dec_var.set(str(value))
                self._log(f"Backlash {axis} set to {value} arcsec", "success")
            else:
                self._log(f"Set backlash failed: {result.message}", "error")
        else:
            self._log("Cannot set backlash: not connected", "warning")

    def _get_backlash(self):
        """Query current backlash values from mount."""
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            ra_val = mp.get_backlash('ra', bridge.send_command)
            dec_val = mp.get_backlash('dec', bridge.send_command)
            if ra_val is not None:
                self.backlash_ra_var.set(str(ra_val))
            if dec_val is not None:
                self.backlash_dec_var.set(str(dec_val))

    # ------------------------------------------------------------------
    # OnStep Extended: Mount Limits
    # ------------------------------------------------------------------

    def _set_horizon_limit(self, degrees: int):
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            result = mp.set_horizon_limit(degrees, bridge.send_command)
            if result.success:
                self.horizon_limit_var.set(str(degrees))
                self._log(f"Horizon limit set to {degrees}\u00b0", "success")
            else:
                self._log(f"Set horizon limit failed: {result.message}", "error")
        else:
            self._log("Cannot set limits: not connected", "warning")

    def _set_overhead_limit(self, degrees: int):
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            result = mp.set_overhead_limit(degrees, bridge.send_command)
            if result.success:
                self.overhead_limit_var.set(str(degrees))
                self._log(f"Overhead limit set to {degrees}\u00b0", "success")
            else:
                self._log(f"Set overhead limit failed: {result.message}", "error")
        else:
            self._log("Cannot set limits: not connected", "warning")

    def _get_limits(self):
        """Query current horizon/overhead limits from mount."""
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            h = mp.get_horizon_limit(bridge.send_command)
            o = mp.get_overhead_limit(bridge.send_command)
            if h is not None:
                self.horizon_limit_var.set(str(h))
            if o is not None:
                self.overhead_limit_var.set(str(o))

    # ------------------------------------------------------------------
    # OnStep Extended: Auxiliary Features
    # ------------------------------------------------------------------

    def _discover_auxiliary_features(self):
        """Discover available auxiliary feature slots from OnStepX."""
        bridge = self._get_active_bridge()
        if not bridge.is_connected:
            return
        mp = self.telescope_bridge.mount_protocol
        if not hasattr(mp, 'get_auxiliary_bitmap'):
            return
        try:
            bitmap = mp.get_auxiliary_bitmap(bridge.send_command)
            if not bitmap:
                self._auxiliary_features = []
                return
            features = []
            for i, ch in enumerate(bitmap):
                if ch == '1':
                    slot = i + 1
                    info = mp.get_auxiliary_info(slot, bridge.send_command)
                    val = mp.get_auxiliary(slot, bridge.send_command)
                    features.append({
                        'slot': slot,
                        'name': info.get('name', f'Feature {slot}') if info else f'Feature {slot}',
                        'purpose': info.get('purpose', '?') if info else '?',
                        'value': val or '0',
                    })
            self._auxiliary_features = features
            if features:
                self._log(f"Discovered {len(features)} auxiliary feature(s)", "info")
        except Exception as e:
            self._log(f"Auxiliary discovery error: {e}", "error")

    def _set_auxiliary_value(self, slot: int, value: int):
        """Set auxiliary feature slot value."""
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            result = mp.set_auxiliary(slot, value, bridge.send_command)
            if result.success:
                # Update cached value
                for f in self._auxiliary_features:
                    if f['slot'] == slot:
                        f['value'] = str(value)
                        break
                self._log(f"Auxiliary slot {slot} set to {value}", "success")
            else:
                self._log(f"Set auxiliary failed: {result.message}", "error")
        else:
            self._log("Cannot set auxiliary: not connected", "warning")

    def _refresh_auxiliary_values(self):
        """Refresh current values for all discovered auxiliary slots."""
        bridge = self._get_active_bridge()
        if not bridge.is_connected:
            return
        mp = self.telescope_bridge.mount_protocol
        for f in self._auxiliary_features:
            try:
                val = mp.get_auxiliary(f['slot'], bridge.send_command)
                if val is not None:
                    f['value'] = val
            except Exception:
                pass

    # ------------------------------------------------------------------
    # OnStep Extended: Firmware Info
    # ------------------------------------------------------------------

    def _query_firmware_info(self):
        """Query firmware info from mount (called once on connect)."""
        bridge = self._get_active_bridge()
        if not bridge.is_connected:
            return
        mp = self.telescope_bridge.mount_protocol
        try:
            info = mp.get_firmware_info(bridge.send_command)
            if info:
                self.firmware_name_var.set(info.get('product', '--'))
                self.firmware_version_var.set(info.get('version', '--'))
                self.firmware_mount_type_var.set(info.get('mount_type', '--'))
                # Extract park state from GU flags
                park = info.get('park_state', '')
                if park:
                    self.park_state_var.set(park)
                # Extract tracking state
                tracking = info.get('tracking', '')
                if tracking:
                    self.tracking_enabled_var.set(tracking == 'Tracking')
                # Extract tracking rate
                rate = info.get('tracking_rate', '')
                if rate:
                    self.tracking_rate_var.set(rate)
                # Extract PEC state
                pec = info.get('pec_state', '')
                if pec:
                    self.mount_pec_status_var.set(pec)
                pec_recorded = info.get('pec_recorded', False)
                self.mount_pec_recorded_var.set(pec_recorded)
                self._log(
                    f"Firmware: {info.get('product', '?')} v{info.get('version', '?')} "
                    f"({info.get('mount_type', '?')})",
                    "info",
                )
        except Exception as e:
            self._log(f"Firmware query error: {e}", "error")

    # ------------------------------------------------------------------
    # NexStar/SynScan: Guide Rate
    # ------------------------------------------------------------------

    def _set_guide_rate(self, rate_arcsec: float):
        """Set autoguide rate in arcsec/sec via NexStar passthrough."""
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            if hasattr(mp, 'set_guide_rate'):
                result = mp.set_guide_rate(rate_arcsec, bridge.send_command)
                if result.success:
                    self.guide_rate_var.set(f"{rate_arcsec:.1f}")
                    self._log(result.message, "success")
                else:
                    self._log(f"Set guide rate failed: {result.message}", "error")
            else:
                self._log("Guide rate not supported by this protocol", "warning")
        else:
            self._log("Cannot set guide rate: not connected", "warning")

    # ------------------------------------------------------------------
    # NexStar/SynScan: Hibernate (Position Save/Restore)
    # ------------------------------------------------------------------

    def _hibernate_save(self):
        """Save current mount position for later restoration."""
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            if hasattr(mp, 'hibernate_save'):
                result = mp.hibernate_save(bridge.send_command)
                if result.success:
                    self.hibernate_status_var.set("Position saved")
                    self._log(result.message, "success")
                    # Persist to config
                    pos = mp.get_hibernate_position()
                    if pos and self.config_manager:
                        self.config_manager.set("nexstar.hibernate_azm", pos['azm'])
                        self.config_manager.set("nexstar.hibernate_alt", pos['alt'])
                        self.config_manager.save()
                else:
                    self._log(f"Hibernate save failed: {result.message}", "error")
            else:
                self._log("Hibernate not supported by this protocol", "warning")
        else:
            self._log("Cannot save position: not connected", "warning")

    def _hibernate_restore(self):
        """Restore previously saved mount position."""
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            if hasattr(mp, 'hibernate_restore'):
                # Try to load from config if not in memory
                if mp.get_hibernate_position() is None and self.config_manager:
                    azm = self.config_manager.get("nexstar.hibernate_azm", None)
                    alt = self.config_manager.get("nexstar.hibernate_alt", None)
                    if azm is not None and alt is not None:
                        mp.set_hibernate_position(float(azm), float(alt))

                result = mp.hibernate_restore(bridge.send_command)
                if result.success:
                    self.hibernate_status_var.set("Restoring...")
                    self._log(result.message, "success")
                else:
                    self._log(f"Hibernate restore failed: {result.message}", "error")
            else:
                self._log("Hibernate not supported by this protocol", "warning")
        else:
            self._log("Cannot restore position: not connected", "warning")

    # ------------------------------------------------------------------
    # NexStar/SynScan: Speed Compensation (ppm)
    # ------------------------------------------------------------------

    def _set_speed_compensation(self, ppm: float):
        """Set tracking speed compensation in parts-per-million."""
        bridge = self._get_active_bridge()
        mp = self.telescope_bridge.mount_protocol
        if hasattr(mp, 'set_speed_compensation'):
            result = mp.set_speed_compensation(ppm, bridge.send_command if bridge.is_connected else None)
            if result.success:
                self.speed_comp_ppm_var.set(f"{ppm:.1f}")
                self._log(result.message, "success")
            else:
                self._log(f"Speed comp failed: {result.message}", "error")
        else:
            self._log("Speed compensation not supported by this protocol", "warning")

    # ------------------------------------------------------------------
    # OnStep Extended: Extended Focuser
    # ------------------------------------------------------------------

    def _focuser_goto(self, position: int):
        """Send focuser to absolute position (microns)."""
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            result = mp.focuser_goto(position, bridge.send_command)
            if result.success:
                self.focuser_target_var.set(str(position))
                self._log(f"Focuser goto {position}", "cmd")
            else:
                self._log(f"Focuser goto failed: {result.message}", "error")
        else:
            self._log("Cannot move focuser: not connected", "warning")

    def _focuser_zero(self):
        """Zero the focuser position."""
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.focuser_zero(bridge.send_command)
            self.focuser_position_var.set("0")
            self._log("Focuser position zeroed", "success")

    def _focuser_set_home(self):
        """Set current position as focuser home."""
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.focuser_set_home(bridge.send_command)
            self._log("Focuser home position set", "success")

    def _focuser_go_home(self):
        """Send focuser to home position."""
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.focuser_home(bridge.send_command)
            self._log("Focuser going home", "cmd")

    def _focuser_set_tcf(self, enabled: bool):
        """Enable/disable temperature compensation."""
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.focuser_set_tcf(enabled, bridge.send_command)
            self.focuser_tcf_var.set(enabled)
            self._log(f"Focuser TCF {'enabled' if enabled else 'disabled'}", "info")

    def _focuser_select(self, focuser_num: int):
        """Select active focuser (1-6)."""
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            result = mp.focuser_select(focuser_num, bridge.send_command)
            if result.success:
                self.focuser_selected_var.set(str(focuser_num))
                self._log(f"Selected focuser {focuser_num}", "success")
            else:
                self._log(f"Select focuser failed: {result.message}", "error")

    def _poll_focuser_extended(self):
        """Poll focuser temperature and TCF state."""
        bridge = self._get_active_bridge()
        if not bridge.is_connected:
            return
        mp = self.telescope_bridge.mount_protocol
        try:
            temp = mp.focuser_get_temperature(bridge.send_command)
            if temp is not None:
                self.focuser_temperature_var.set(f"{temp:.1f}\u00b0C")
        except Exception:
            pass
        try:
            tcf = mp.focuser_get_tcf_enabled(bridge.send_command)
            if tcf is not None:
                self.focuser_tcf_var.set(tcf)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # OnStep Extended: Rotator
    # ------------------------------------------------------------------

    def _rotator_move(self, direction: str):
        """Move rotator CW or CCW continuously."""
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            if direction == "CW":
                mp.rotator_move_cw(bridge.send_command)
            else:
                mp.rotator_move_ccw(bridge.send_command)
            self.rotator_status_var.set(f"Moving {direction}")
            self._log(f"Rotator moving {direction}", "cmd")

    def _rotator_stop(self):
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.rotator_stop(bridge.send_command)
        self.rotator_status_var.set("Stopped")
        self._log("Rotator stopped", "warning")

    def _rotator_goto(self, angle: float):
        """Go to absolute rotator angle in degrees."""
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            result = mp.rotator_goto(angle, bridge.send_command)
            if result.success:
                self._log(f"Rotator goto {angle:.1f}\u00b0", "cmd")
            else:
                self._log(f"Rotator goto failed: {result.message}", "error")

    def _rotator_zero(self):
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.rotator_zero(bridge.send_command)
            self.rotator_angle_var.set("0.0\u00b0")
            self._log("Rotator position zeroed", "success")

    def _rotator_toggle_derotation(self):
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            currently = self.rotator_derotating_var.get()
            if currently:
                mp.rotator_disable_derotation(bridge.send_command)
                self.rotator_derotating_var.set(False)
                self._log("Rotator derotation disabled", "warning")
            else:
                mp.rotator_enable_derotation(bridge.send_command)
                self.rotator_derotating_var.set(True)
                self._log("Rotator derotation enabled", "success")

    def _rotator_reverse(self):
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.rotator_reverse(bridge.send_command)
            self._log("Rotator direction reversed", "info")

    def _rotator_parallactic(self):
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.rotator_parallactic(bridge.send_command)
            self._log("Rotator set to parallactic angle", "info")

    def _rotator_set_rate(self, rate: int):
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.rotator_set_rate(rate, bridge.send_command)
            self._log(f"Rotator rate set to {rate}", "info")

    def _poll_rotator(self):
        """Poll rotator angle."""
        bridge = self._get_active_bridge()
        if not bridge.is_connected:
            return
        mp = self.telescope_bridge.mount_protocol
        try:
            angle = mp.rotator_get_angle(bridge.send_command)
            if angle is not None:
                self.rotator_angle_var.set(f"{angle:.1f}\u00b0")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # OnStep Extended: Reticle / LED
    # ------------------------------------------------------------------

    def _reticle_brighter(self):
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.reticle_brighter(bridge.send_command)
            self._log("Reticle brightness increased", "cmd")

    def _reticle_dimmer(self):
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            mp.reticle_dimmer(bridge.send_command)
            self._log("Reticle brightness decreased", "cmd")

    # ------------------------------------------------------------------
    # OnStep Extended: Home (improved)
    # ------------------------------------------------------------------

    def _home_find(self):
        """Find home position (:hC#)."""
        bridge = self._get_active_bridge()
        if bridge.is_connected:
            mp = self.telescope_bridge.mount_protocol
            # :hC# = find home
            bridge.send_command(":hC#")
        self.telescope_status_var.set("Finding home...")
        self._log("Finding home position...", "info")

    # ------------------------------------------------------------------
    # OnStep status polling (called from _tick)
    # ------------------------------------------------------------------

    def _poll_onstep_status(self):
        """Poll OnStepX status flags from :GU# (called every few ticks).

        Updates park state, tracking, pier side, PEC status, and
        tracking rate from the firmware-reported flags.
        """
        bridge = self._get_active_bridge()
        if not bridge.is_connected:
            return
        mp = self.telescope_bridge.mount_protocol
        try:
            info = mp.get_firmware_info(bridge.send_command)
            if not info:
                return
            # Park state
            park = info.get('park_state', '')
            if park:
                self.park_state_var.set(park)
            # Tracking
            tracking = info.get('tracking', '')
            if tracking:
                self.tracking_enabled_var.set(tracking == 'Tracking')
            # Rate
            rate = info.get('tracking_rate', '')
            if rate:
                self.tracking_rate_var.set(rate)
            # PEC
            pec = info.get('pec_state', '')
            if pec:
                self.mount_pec_status_var.set(pec)
            pec_rec = info.get('pec_recorded', False)
            self.mount_pec_recorded_var.set(pec_rec)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Weather
    # ------------------------------------------------------------------

    def _update_weather(self):
        if self.weather_service is None:
            self.weather_status_var.set("Weather service not available")
            self._log("Weather service not available", "warning")
            return
        try:
            lat = float(self.lat_var.get())
            lon = float(self.lon_var.get())

            # Skip fetch for unset coordinates (0,0 = middle of Atlantic)
            if lat == 0.0 and lon == 0.0:
                self.weather_status_var.set("Set location first")
                return

            # Reset backoff/cache when coordinates change significantly
            # so stale None cache from old position doesn't block the fetch.
            cached = self.weather_service.cached_data
            if cached is not None:
                if (abs(lat - cached.latitude) > 0.01 or
                        abs(lon - cached.longitude) > 0.01):
                    self.weather_service.cached_data = None
                    self.weather_service.last_update_time = 0
                    self.weather_service._consecutive_failures = 0
                    _logger.info("Weather cache invalidated -- coordinates changed")
            elif self.weather_service._consecutive_failures > 0:
                # No cached data but in backoff -- reset so we retry immediately
                self.weather_service.last_update_time = 0
                self.weather_service._consecutive_failures = 0

            self.weather_status_var.set("Fetching...")
            data = self.weather_service.get_weather(lat, lon)
            if data:
                self.weather_temp_var.set(f"{data.temperature:.1f}°C")
                self.weather_pressure_var.set(f"{data.pressure:.0f} hPa")
                self.weather_humidity_var.set(f"{data.humidity:.0f}%")
                self.weather_cloud_var.set(f"{data.cloud_cover:.0f}%")
                self.weather_wind_var.set(f"{data.wind_speed:.0f} km/h")
                self.weather_wind_dir_var.set(f"{data.wind_direction:.0f}°")
                self.weather_gusts_var.set(f"{data.wind_gusts:.0f} km/h")
                self.weather_dewpoint_var.set(f"{data.dew_point:.1f}°C")
                self.weather_location_var.set(data.location or "--")
                from weather_service import weather_code_description, assess_observing_conditions
                self.weather_conditions_var.set(
                    weather_code_description(data.weather_code) if data.weather_code else "--"
                )
                # Dew risk: margin between temperature and dew point
                dew_margin = data.temperature - data.dew_point
                if dew_margin < 2:
                    self.weather_dew_risk_var.set("HIGH")
                elif dew_margin < 5:
                    self.weather_dew_risk_var.set("Moderate")
                else:
                    self.weather_dew_risk_var.set("Low")
                # Observing conditions rating
                rating, _ = assess_observing_conditions(data)
                self.weather_observing_var.set(rating)
                self.weather_status_var.set("Data up to date")
                self._log("Weather data refreshed", "success")
                # Push weather to telescope for refraction correction
                try:
                    self._send_weather_to_telescope()
                except Exception as wx:
                    _logger.debug("Weather push to telescope failed: %s", wx)
            else:
                self.weather_status_var.set("Fetch failed -- will retry")
                self._log("Weather fetch returned no data (network error?)", "warning")
        except Exception as e:
            self.weather_status_var.set(f"Error: {e}")
            self._log(f"Weather error: {e}", "error")

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _save_config(self):
        if self.config_manager:
            try:
                self.config_manager.set("location.latitude", float(self.lat_var.get()))
                self.config_manager.set("location.longitude", float(self.lon_var.get()))
                self.config_manager.save_config()
                self._log("Configuration saved", "success")
            except Exception as e:
                self._log(f"Config save error: {e}", "error")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the web server and background update loop."""
        self._running = True

        # Start web server
        self.web_server.start()

        # Start update loop
        self._update_thread = threading.Thread(
            target=self._update_loop_thread,
            name="HeadlessUpdateLoop",
            daemon=True,
        )
        self._update_thread.start()

        # Fetch weather data on startup (in a daemon thread so it doesn't
        # block the caller -- the HTTP request can take a few seconds).
        def _initial_weather():
            try:
                self._update_weather()
            except Exception:
                _logger.debug("Initial weather fetch failed", exc_info=True)

        threading.Thread(
            target=_initial_weather,
            name="InitialWeatherFetch",
            daemon=True,
        ).start()

        self._log("=" * 50, "info")
        self._log("  TrackWise-AltAzPro - Headless Mode", "success")
        self._log(f"  Web UI: http://0.0.0.0:{self.web_server.port}", "info")
        self._log("  Press Ctrl+C to stop", "info")
        self._log("=" * 50, "info")

    def stop(self):
        """Clean shutdown."""
        self._log("Shutting down...", "warning")
        self._running = False

        # Save session data before tearing down subsystems
        if self.session_recorder.is_started:
            try:
                path = self.session_recorder.save(auto=True)
                if path:
                    self._log(f"Session saved on shutdown to {path}", "success")
                self.session_recorder.stop()
            except Exception as e:
                self._log(f"Session save on shutdown error: {e}", "error")

        # Stop tracking
        if self.tracking.is_running:
            try:
                self.tracking.stop()
            except Exception:
                pass

        # Save config
        self._save_config()

        # ML model save
        if hasattr(self.tracking, 'ml_predictor'):
            try:
                self.tracking.ml_predictor.save_model()
            except Exception:
                pass

        # Clean shutdown marker
        try:
            self.crash_recovery.mark_clean_shutdown()
        except Exception:
            pass

        # Disconnect hardware
        for bridge in [self.telescope_bridge, self.telescope_simulator]:
            try:
                if bridge.is_connected:
                    bridge.disconnect()
            except Exception:
                pass

        # Stop web server
        if self.web_server:
            self.web_server.stop()

        # Wait for update thread
        if self._update_thread:
            self._update_thread.join(timeout=3)

        _logger.info("Headless server stopped")

    def wait_forever(self):
        """Block until Ctrl+C or SIGTERM."""
        stop_event = threading.Event()

        def on_signal(sig, frame):
            stop_event.set()

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)
        stop_event.wait()
        self.stop()


# ===================================================================
# CLI entry point
# ===================================================================

def main():
    # Force unbuffered stdout so messages appear immediately in the terminal
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

    parser = argparse.ArgumentParser(
        description="TrackWise-AltAzPro - Headless Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python headless_server.py --simulator\n"
            "  python headless_server.py --port /dev/ttyUSB0\n"
            "  python headless_server.py --wifi 192.168.0.1 --wifi-port 9996\n"
            "  python headless_server.py --port COM3 --web-port 9090\n"
        ),
    )
    parser.add_argument("--port", default="", help="Serial port (e.g. /dev/ttyUSB0, COM3)")
    parser.add_argument("--baud", type=int, default=9600, help="Baud rate (default: 9600)")
    parser.add_argument("--wifi", default="", help="WiFi/TCP IP address")
    parser.add_argument("--wifi-port", type=int, default=9996, help="WiFi/TCP port (default: 9996)")
    parser.add_argument("--simulator", action="store_true", help="Use virtual telescope simulator")
    parser.add_argument("--web-port", type=int, default=8080, help="Web server port (default: 8080)")
    parser.add_argument("--lat", type=float, default=DEFAULT_LATITUDE,
                        help=f"Observer latitude (default: {DEFAULT_LATITUDE})")
    parser.add_argument("--lon", type=float, default=DEFAULT_LONGITUDE,
                        help=f"Observer longitude (default: {DEFAULT_LONGITUDE})")
    args = parser.parse_args()

    conn_type = "USB"
    if args.wifi:
        conn_type = "WiFi"

    print("Starting TrackWise-AltAzPro (Headless)...", flush=True)

    app = HeadlessTelescopeApp(
        connection_type=conn_type,
        port=args.port,
        baudrate=args.baud,
        wifi_ip=args.wifi,
        wifi_port=args.wifi_port,
        simulator=args.simulator,
        web_port=args.web_port,
        latitude=args.lat,
        longitude=args.lon,
    )

    # Wire logging config from config_manager into telescope_logger
    if app.config_manager:
        log_cfg = {
            "file_path": app.config_manager.get("logging.file_path", "telescope_app.log"),
            "max_file_size_mb": app.config_manager.get("logging.max_file_size_mb", 10),
            "backup_count": app.config_manager.get("logging.backup_count", 5),
        }
        setup_logging(
            log_file=log_cfg["file_path"],
            max_bytes=int(log_cfg["max_file_size_mb"]) * 1024 * 1024,
            backup_count=int(log_cfg["backup_count"]),
        )
    else:
        setup_logging()

    app.start()

    print(flush=True)
    print("=" * 55, flush=True)
    print("  TrackWise-AltAzPro - Headless Mode", flush=True)
    print(f"  Web UI: http://0.0.0.0:{args.web_port}", flush=True)
    print("  Press Ctrl+C to stop", flush=True)
    print("=" * 55, flush=True)
    print(flush=True)

    app.wait_forever()


if __name__ == "__main__":
    main()
