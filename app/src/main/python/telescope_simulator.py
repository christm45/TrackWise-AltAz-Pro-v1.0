"""
Telescope Simulator - Virtual telescope mount for testing without hardware.

This module provides a drop-in replacement for TelescopeBridge that simulates
a Dobson Alt-Az telescope mount. It generates realistic position data with
configurable drift, periodic errors, and random noise so the full tracking
pipeline (Kalman filter, ML predictor, Software PEC) can
be exercised end-to-end without a physical mount.

Architecture role:
    Application layers (UI, tracking, plate-solving)
        |
        v
    Protocol layer (lx200_protocol / onstep_protocol)
        |
        v
    >>> TelescopeSimulator <<<   <-- this module (replaces TelescopeBridge)
        |
        v
    [No hardware -- all positions are computed mathematically]

Drift model:
    The simulated mount applies three sources of pointing error:

    1. **Sidereal drift** -- Earth rotation causes the field to drift at
       ~15 arcsec/s in the azimuth axis (scaled by cos(altitude)). This is
       the dominant error source on an untracked Dobson.

    2. **Periodic error (PE)** -- Sinusoidal oscillation with configurable
       amplitude and period, simulating worm-gear mechanical errors. Default
       is +/-10 arcsec at a 480-second period (8 minutes, typical for a
       single-arm worm drive).

    3. **Random noise** -- Gaussian noise on each position read, simulating
       seeing, vibration, and encoder resolution limits. Default sigma is
       1 arcsec.

Slew simulation:
    GOTO commands set a target position and the mount "slews" toward it at
    a configurable speed (default 3 deg/s). During a slew the :D# command
    returns a '|' character; once the target is reached it returns '#'.

Plate-solve simulation:
    The simulator can optionally generate synthetic plate-solve results.
    When enabled, a background thread fires on_plate_solve callbacks at
    a configurable interval (default 2 seconds) with the current simulated
    position plus a small noise term, mimicking ASTAP output.

Command processing:
    The simulator responds to the same LX200 subset that TelescopeBridge
    uses in its _read_loop() and force_position_update():
        :GA# -> current altitude (sDD*MM:SS#)
        :GZ# -> current azimuth  (DDD*MM:SS#)
        :D#  -> slew status ('|' if slewing, '#' if stationary)
        :Sa, :Sz, :MA#  -> Alt-Az GOTO
        :Sr, :Sd, :MS#  -> RA/Dec GOTO (converted to Alt-Az internally)
        :CM# -> sync (resets position to target)
        :Q#  -> stop all motion
        :GVP# -> product name ("Simulator")
        :GVN# -> firmware version ("1.0")
    All other commands return "1#" (generic acknowledgement).

Thread safety:
    Internal state is protected by a threading.Lock. The background
    position-update thread and the main-thread command interface can
    operate concurrently without races.
"""

import math
import time
import random
import threading
from typing import Optional, Callable
from dataclasses import dataclass

from telescope_logger import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Simulation parameters (sensible defaults for a Dobson Alt-Az)
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Tunable parameters for the simulated telescope mount.

    All angular values are in arcseconds unless otherwise noted.

    Attributes:
        start_alt: Initial altitude in degrees.
        start_az: Initial azimuth in degrees.
        sidereal_drift_enabled: Whether to apply Earth-rotation drift.
        sidereal_rate_arcsec: Base sidereal rate in arcsec/s (15.0411).
        pe_enabled: Whether to apply periodic error oscillation.
        pe_amplitude_arcsec: Peak periodic error amplitude (arcsec).
        pe_period_sec: Periodic error cycle length (seconds).
        noise_enabled: Whether to add Gaussian position noise.
        noise_sigma_arcsec: Noise standard deviation (arcsec).
        slew_speed_deg_s: Slew speed for GOTO operations (deg/s).
        position_poll_interval: How often (seconds) the background thread
            fires on_altaz_update.  Matches real bridge (~2 s).
        latitude: Observer latitude in degrees (for sidereal projection).
    """
    start_alt: float = 45.0
    start_az: float = 180.0
    sidereal_drift_enabled: bool = True
    sidereal_rate_arcsec: float = 15.0411
    pe_enabled: bool = True
    pe_amplitude_arcsec: float = 10.0
    pe_period_sec: float = 480.0
    noise_enabled: bool = True
    noise_sigma_arcsec: float = 1.0
    slew_speed_deg_s: float = 3.0
    position_poll_interval: float = 2.0
    latitude: float = 48.8566  # Paris default -- overridden by app config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deg_to_dms(deg: float, signed: bool = False) -> str:
    """Convert decimal degrees to LX200 sDD*MM:SS or DDD*MM:SS string.

    Args:
        deg: Angle in decimal degrees.
        signed: If True, prefix with '+' or '-' (for altitude/declination).

    Returns:
        LX200-format string ending with '#'.
    """
    negative = deg < 0
    deg = abs(deg)
    d = int(deg)
    m = int((deg - d) * 60)
    s = int(((deg - d) * 60 - m) * 60 + 0.5)
    if s >= 60:
        s -= 60
        m += 1
    if m >= 60:
        m -= 60
        d += 1

    if signed:
        sign = '-' if negative else '+'
        return f"{sign}{d:02d}*{m:02d}:{s:02d}#"
    else:
        # Unsigned (azimuth): DDD*MM:SS
        return f"{d:03d}*{m:02d}:{s:02d}#"


def _hours_to_hms(hours: float) -> str:
    """Convert decimal hours to LX200 HH:MM:SS string.

    Args:
        hours: Angle in decimal hours (0..24).

    Returns:
        LX200-format string ending with '#'.
    """
    hours = hours % 24.0
    h = int(hours)
    m = int((hours - h) * 60)
    s = int(((hours - h) * 60 - m) * 60 + 0.5)
    if s >= 60:
        s -= 60
        m += 1
    if m >= 60:
        m -= 60
        h += 1
    return f"{h:02d}:{m:02d}:{s:02d}#"


def _parse_dms(s: str) -> Optional[float]:
    """Parse a DMS or signed-DMS LX200 string to decimal degrees.

    Accepts formats: sDD*MM:SS, sDD*MM, DDD*MM:SS, DDD*MM
    where s is '+' or '-' and '*' can also be a degree symbol.

    Returns:
        Decimal degrees, or None if parsing fails.
    """
    s = s.strip().rstrip('#')
    if not s:
        return None
    try:
        sign = 1.0
        if s[0] in ('+', '-'):
            sign = -1.0 if s[0] == '-' else 1.0
            s = s[1:]
        s = s.replace('°', '*').replace('\xdf', '*')
        parts = s.replace('*', ':').split(':')
        d = float(parts[0])
        m = float(parts[1]) if len(parts) > 1 else 0.0
        sec = float(parts[2]) if len(parts) > 2 else 0.0
        return sign * (d + m / 60.0 + sec / 3600.0)
    except (ValueError, IndexError):
        return None


def _parse_hms(s: str) -> Optional[float]:
    """Parse an HMS LX200 string (HH:MM:SS) to decimal hours.

    Returns:
        Decimal hours, or None if parsing fails.
    """
    s = s.strip().rstrip('#')
    if not s:
        return None
    try:
        parts = s.split(':')
        h = float(parts[0])
        m = float(parts[1]) if len(parts) > 1 else 0.0
        sec = float(parts[2]) if len(parts) > 2 else 0.0
        return h + m / 60.0 + sec / 3600.0
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# TelescopeSimulator -- drop-in replacement for TelescopeBridge
# ---------------------------------------------------------------------------

class TelescopeSimulator:
    """Simulated telescope mount that generates realistic drift and noise.

    Exposes the same public interface as TelescopeBridge so it can be used
    as a drop-in replacement. The application swaps between the real bridge
    and the simulator via a UI toggle button.

    Public interface mirrored from TelescopeBridge:
        Attributes: is_connected, telescope_info, last_error, is_onstep,
                    on_connected, on_disconnected, on_log, on_altaz_update
        Methods:    connect(), disconnect(), send_command(),
                    force_position_update(), goto_altaz(), goto(), stop(),
                    sync_altaz(), sync(), get_available_ports()
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """Initialize the simulator with default or custom configuration.

        Args:
            config: Optional simulation parameters. Uses sensible defaults
                    for a Dobson Alt-Az mount if not provided.
        """
        self.config = config or SimulationConfig()

        # --- Mirror TelescopeBridge attributes ---
        self.is_connected: bool = False
        self.telescope_info = None
        self.last_error: Optional[str] = None
        self.is_onstep: bool = True  # Pretend to be OnStep firmware
        self.connection_type: Optional[str] = 'simulator'

        # Callbacks (same signature as TelescopeBridge)
        self.on_connected: Optional[Callable] = None
        self.on_disconnected: Optional[Callable] = None
        self.on_log: Optional[Callable] = None
        self.on_altaz_update: Optional[Callable] = None

        # --- Internal simulation state ---
        self._lock = threading.Lock()

        # True mount position (degrees) -- updated every tick
        self._alt: float = self.config.start_alt
        self._az: float = self.config.start_az

        # Target position for GOTO slews
        self._target_alt: float = self._alt
        self._target_az: float = self._az
        self._is_slewing: bool = False

        # Timestamp bookkeeping
        self._start_time: float = 0.0
        self._last_tick: float = 0.0

        # Cumulative tracking rate corrections applied by the controller
        # (arcsec/s offsets that the tracking loop sends via :SXTR / :SXTD)
        self._correction_rate_alt: float = 0.0   # arcsec/s
        self._correction_rate_az: float = 0.0    # arcsec/s

        # Background thread
        self._read_thread: Optional[threading.Thread] = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Port enumeration (stub)
    # ------------------------------------------------------------------

    def get_available_ports(self):
        """Return a list of available ports (includes a simulator entry).

        Returns:
            A list containing the "SIMULATOR" pseudo-port plus any real
            COM ports that happen to be available.
        """
        return ["SIMULATOR"]

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, port=None, baudrate=None, connection_type=None,
                tcp_ip=None, tcp_port=None) -> bool:
        """Start the simulated telescope connection.

        All parameters are accepted for API compatibility but ignored --
        the simulator always connects instantly.

        Returns:
            True (always succeeds).
        """
        if self.is_connected:
            self.disconnect()

        with self._lock:
            self._start_time = time.time()
            self._last_tick = self._start_time
            self.is_connected = True
            self.connection_type = 'simulator'

            # Create a TelescopeInfo-compatible object
            from telescope_bridge import TelescopeInfo
            self.telescope_info = TelescopeInfo(
                port="SIMULATOR",
                baudrate=0,
                is_connected=True,
                model="Simulator (Virtual Dobson)"
            )

        self._log("Simulator connected -- virtual Dobson Alt-Az mount active")
        self._log(f"  Start position: Alt={self._alt:.2f} Az={self._az:.2f}")
        self._log(f"  Sidereal drift: {'ON' if self.config.sidereal_drift_enabled else 'OFF'}")
        self._log(f"  Periodic error: {'ON' if self.config.pe_enabled else 'OFF'} "
                  f"({self.config.pe_amplitude_arcsec}\" @ {self.config.pe_period_sec}s)")
        self._log(f"  Random noise:   {'ON' if self.config.noise_enabled else 'OFF'} "
                  f"(sigma={self.config.noise_sigma_arcsec}\")")

        # Start the background position-update thread
        self._running = True
        self._read_thread = threading.Thread(
            target=self._read_loop, daemon=True, name="SimulatorReadLoop"
        )
        self._read_thread.start()

        # Fire the on_connected callback
        if self.on_connected:
            self.on_connected(self.telescope_info)

        return True

    def disconnect(self):
        """Stop the simulator and clean up.

        Stops the background polling thread, resets state, and fires the
        on_disconnected callback.
        """
        self._running = False
        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join(timeout=3.0)

        with self._lock:
            self.is_connected = False
            self.telescope_info = None
            self.connection_type = None
            self._is_slewing = False
            self._correction_rate_alt = 0.0
            self._correction_rate_az = 0.0

        self._log("Simulator disconnected")
        if self.on_disconnected:
            self.on_disconnected()

    # ------------------------------------------------------------------
    # Physics simulation
    # ------------------------------------------------------------------

    def _tick(self):
        """Advance the simulation by one time step.

        Must be called with self._lock held. Updates the mount position
        based on elapsed time, applying sidereal drift, periodic error,
        tracking corrections, and slew motion.
        """
        now = time.time()
        dt = now - self._last_tick
        self._last_tick = now

        if dt <= 0 or dt > 10.0:
            # Skip unreasonable dt (clock skew or long pause)
            return

        elapsed = now - self._start_time

        # --- Slew motion ---
        if self._is_slewing:
            max_step = self.config.slew_speed_deg_s * dt
            d_alt = self._target_alt - self._alt
            d_az = self._target_az - self._az

            dist = math.sqrt(d_alt ** 2 + d_az ** 2)
            if dist <= max_step:
                # Arrived at target
                self._alt = self._target_alt
                self._az = self._target_az
                self._is_slewing = False
                self._log(f"Slew complete: Alt={self._alt:.2f} Az={self._az:.2f}")
            else:
                # Move toward target at slew speed
                ratio = max_step / dist
                self._alt += d_alt * ratio
                self._az += d_az * ratio
            return  # No drift while slewing

        # --- Sidereal drift (Earth rotation) ---
        if self.config.sidereal_drift_enabled:
            # Azimuth drifts at sidereal_rate * cos(alt) (field rotation)
            cos_alt = math.cos(math.radians(self._alt))
            az_drift = (self.config.sidereal_rate_arcsec * cos_alt * dt) / 3600.0
            # Altitude has a smaller drift component (depends on azimuth)
            sin_az = math.sin(math.radians(self._az))
            alt_drift = (self.config.sidereal_rate_arcsec * 0.3 * abs(sin_az) * dt) / 3600.0
            self._az += az_drift
            self._alt -= alt_drift  # Objects set toward horizon

        # --- Periodic error ---
        if self.config.pe_enabled:
            phase = (2.0 * math.pi * elapsed) / self.config.pe_period_sec
            pe_alt = (self.config.pe_amplitude_arcsec * math.sin(phase)) / 3600.0
            pe_az = (self.config.pe_amplitude_arcsec * 0.7 * math.cos(phase * 1.3)) / 3600.0
            self._alt += pe_alt * (dt / self.config.pe_period_sec) * 2 * math.pi
            self._az += pe_az * (dt / self.config.pe_period_sec) * 2 * math.pi

        # --- Tracking corrections (applied by the control loop) ---
        self._alt -= (self._correction_rate_alt * dt) / 3600.0
        self._az -= (self._correction_rate_az * dt) / 3600.0

        # --- Clamp to valid ranges ---
        self._alt = max(0.0, min(90.0, self._alt))
        self._az = self._az % 360.0

    def _get_noisy_position(self):
        """Return the current position with optional Gaussian noise.

        Returns:
            (alt_deg, az_deg) tuple with noise applied.
        """
        alt = self._alt
        az = self._az
        if self.config.noise_enabled:
            alt += random.gauss(0, self.config.noise_sigma_arcsec) / 3600.0
            az += random.gauss(0, self.config.noise_sigma_arcsec) / 3600.0
        return alt, az

    # ------------------------------------------------------------------
    # Background polling thread (mirrors TelescopeBridge._read_loop)
    # ------------------------------------------------------------------

    def _read_loop(self):
        """Background thread that periodically fires on_altaz_update.

        Matches the real TelescopeBridge polling cadence (~2 s). Each
        iteration advances the physics simulation and formats the position
        as LX200 strings before invoking the callback.

        Position is reported even during slewing so the UI shows real-time
        updates during GoTo. The callback receives three arguments:
        (alt_str, az_str, is_slewing).
        """
        time.sleep(0.5)  # Startup stabilization (shorter than real bridge)

        # Track slewing state transitions for logging
        was_slewing = False

        while self._running and self.is_connected:
            try:
                with self._lock:
                    self._tick()
                    is_slewing = self._is_slewing

                    # ALWAYS report position, even during slewing, so the
                    # UI shows real-time position updates during GoTo.
                    alt, az = self._get_noisy_position()
                    alt_str = _deg_to_dms(alt, signed=True)
                    az_str = _deg_to_dms(az, signed=False)

                # Log slew state transitions (outside lock)
                if is_slewing and not was_slewing:
                    self._log("Simulator is slewing...")
                elif not is_slewing and was_slewing:
                    self._log("Simulator slew complete - stationary")
                was_slewing = is_slewing

                if self.on_altaz_update:
                    self.on_altaz_update(alt_str, az_str, is_slewing)

                # Poll faster during slewing for responsive position updates
                poll_interval = 0.5 if is_slewing else 1.0
                time.sleep(poll_interval)

            except Exception as e:
                _logger.error(f"Simulator read loop error: {e}")
                break

    # ------------------------------------------------------------------
    # LX200 command interface (mirrors TelescopeBridge.send_command)
    # ------------------------------------------------------------------

    def send_command(self, command: str) -> str:
        """Process an LX200 command and return a simulated response.

        Handles the subset of commands that the tracking pipeline and
        bridge _read_loop actually use. Unknown commands return "1#".

        Args:
            command: LX200 command string (e.g. ":GA#").

        Returns:
            Simulated response string.
        """
        if not self.is_connected:
            return ""

        # Normalize: strip leading ':' and trailing '#'
        cmd = command.strip()
        if cmd.startswith(':'):
            cmd = cmd[1:]
        if cmd.endswith('#'):
            cmd = cmd[:-1]

        with self._lock:
            self._tick()
            return self._process(cmd)

    def _process(self, cmd: str) -> str:
        """Dispatch a normalized command. Must be called with _lock held.

        Args:
            cmd: Command without leading ':' or trailing '#'.

        Returns:
            Response string.
        """
        # --- Position queries ---
        if cmd == "GA":
            # Get Altitude
            alt, _ = self._get_noisy_position()
            return _deg_to_dms(alt, signed=True)

        if cmd == "GZ":
            # Get Azimuth
            _, az = self._get_noisy_position()
            return _deg_to_dms(az, signed=False)

        if cmd == "GR":
            # Get RA -- approximate conversion from Alt/Az
            ra_h = self._alt_az_to_ra(self._alt, self._az)
            return _hours_to_hms(ra_h)

        if cmd == "GD":
            # Get Dec -- approximate conversion from Alt/Az
            dec_d = self._alt_az_to_dec(self._alt, self._az)
            return _deg_to_dms(dec_d, signed=True)

        # --- Slew status ---
        if cmd == "D":
            return "|#" if self._is_slewing else "#"

        # --- Identification ---
        if cmd == "GVP":
            return "Simulator#"
        if cmd == "GVN":
            return "1.0#"

        # --- Stop ---
        if cmd == "Q":
            self._is_slewing = False
            self._target_alt = self._alt
            self._target_az = self._az
            self._correction_rate_alt = 0.0
            self._correction_rate_az = 0.0
            return "#"

        # --- Alt-Az GOTO components ---
        if cmd.startswith("Sa"):
            # Set altitude target: :Sa+45*30:00#
            val = _parse_dms(cmd[2:])
            if val is not None:
                self._target_alt = val
                return "1#"
            return "0#"

        if cmd.startswith("Sz"):
            # Set azimuth target: :Sz180*00:00#
            val = _parse_dms(cmd[2:])
            if val is not None:
                self._target_az = val
                return "1#"
            return "0#"

        if cmd == "MA":
            # Move to Alt-Az target
            self._is_slewing = True
            self._log(f"Simulator GOTO: Alt={self._target_alt:.2f} Az={self._target_az:.2f}")
            return "0#"  # 0 = success for :MA#

        # --- Equatorial GOTO components ---
        if cmd.startswith("Sr"):
            # Set RA target (store temporarily -- will be used by :MS#)
            val = _parse_hms(cmd[2:])
            if val is not None:
                self._pending_ra = val
                return "1#"
            return "0#"

        if cmd.startswith("Sd"):
            # Set Dec target
            val = _parse_dms(cmd[2:])
            if val is not None:
                self._pending_dec = val
                return "1#"
            return "0#"

        if cmd == "MS":
            # Slew to RA/Dec target -- convert to Alt/Az
            ra = getattr(self, '_pending_ra', None)
            dec = getattr(self, '_pending_dec', None)
            if ra is not None and dec is not None:
                alt, az = self._ra_dec_to_alt_az(ra, dec)
                self._target_alt = alt
                self._target_az = az
                self._is_slewing = True
                self._log(f"Simulator equatorial GOTO: RA={ra:.4f}h Dec={dec:.4f} -> "
                          f"Alt={alt:.2f} Az={az:.2f}")
            return "0#"

        # --- Sync ---
        if cmd == "CM":
            # Sync: jump position to target immediately (no slew)
            self._alt = self._target_alt
            self._az = self._target_az
            self._is_slewing = False
            self._log(f"Simulator synced to Alt={self._alt:.2f} Az={self._az:.2f}")
            return "Coordinates matched#"

        # --- Tracking rate commands (OnStep SXTR / SXTD) ---
        if cmd.startswith("SXTR,"):
            # Custom RA tracking rate offset (arcsec/s)
            try:
                rate = float(cmd[5:])
                # Map RA rate to azimuth correction (approximate)
                self._correction_rate_az = rate
                return "1#"
            except ValueError:
                return "0#"

        if cmd.startswith("SXTD,"):
            # Custom Dec tracking rate offset (arcsec/s)
            try:
                rate = float(cmd[5:])
                # Map Dec rate to altitude correction (approximate)
                self._correction_rate_alt = rate
                return "1#"
            except ValueError:
                return "0#"

        # --- Speed commands ---
        if cmd in ("RS", "RM", "RC", "RG"):
            return "#"

        # --- Move commands ---
        if cmd in ("Mn", "Ms", "Me", "Mw"):
            return "#"
        if cmd in ("Qn", "Qs", "Qe", "Qw"):
            return "#"

        # --- Default: acknowledge unknown commands ---
        return "1#"

    # ------------------------------------------------------------------
    # Coordinate conversion helpers (approximate, for simulation only)
    # ------------------------------------------------------------------

    def _alt_az_to_ra(self, alt: float, az: float) -> float:
        """Approximate Alt/Az to RA conversion using observer latitude.

        Args:
            alt: Altitude in degrees.
            az: Azimuth in degrees.

        Returns:
            Right Ascension in decimal hours.
        """
        lat = math.radians(self.config.latitude)
        alt_r = math.radians(alt)
        az_r = math.radians(az)

        # Hour angle
        sin_ha = -math.sin(az_r) * math.cos(alt_r)
        cos_ha = (math.sin(alt_r) - math.sin(lat) * self._sin_dec_from_altaz(alt, az)) / (
            math.cos(lat) * self._cos_dec_from_altaz(alt, az) + 1e-10
        )
        ha = math.atan2(sin_ha, cos_ha)
        ha_hours = math.degrees(ha) / 15.0

        # Approximate LST (simplified)
        import datetime
        now = datetime.datetime.utcnow()
        jd = (now - datetime.datetime(2000, 1, 1, 12, 0, 0)).total_seconds() / 86400.0 + 2451545.0
        lst = (280.46061837 + 360.98564736629 * (jd - 2451545.0) + self.config.latitude) % 360.0
        lst_hours = lst / 15.0

        ra = (lst_hours - ha_hours) % 24.0
        return ra

    def _alt_az_to_dec(self, alt: float, az: float) -> float:
        """Approximate Alt/Az to Dec conversion.

        Args:
            alt: Altitude in degrees.
            az: Azimuth in degrees.

        Returns:
            Declination in decimal degrees.
        """
        lat = math.radians(self.config.latitude)
        alt_r = math.radians(alt)
        az_r = math.radians(az)

        sin_dec = math.sin(lat) * math.sin(alt_r) + math.cos(lat) * math.cos(alt_r) * math.cos(az_r)
        dec = math.degrees(math.asin(max(-1.0, min(1.0, sin_dec))))
        return dec

    def _sin_dec_from_altaz(self, alt: float, az: float) -> float:
        """Helper: compute sin(Dec) from Alt/Az."""
        lat = math.radians(self.config.latitude)
        alt_r = math.radians(alt)
        az_r = math.radians(az)
        return math.sin(lat) * math.sin(alt_r) + math.cos(lat) * math.cos(alt_r) * math.cos(az_r)

    def _cos_dec_from_altaz(self, alt: float, az: float) -> float:
        """Helper: compute cos(Dec) from Alt/Az."""
        sin_dec = self._sin_dec_from_altaz(alt, az)
        return math.sqrt(max(0.0, 1.0 - sin_dec ** 2))

    def _ra_dec_to_alt_az(self, ra_hours: float, dec_degrees: float) -> tuple:
        """Approximate RA/Dec to Alt/Az conversion.

        Args:
            ra_hours: Right Ascension in decimal hours.
            dec_degrees: Declination in decimal degrees.

        Returns:
            (alt, az) in decimal degrees.
        """
        import datetime
        lat = math.radians(self.config.latitude)

        # Approximate LST
        now = datetime.datetime.utcnow()
        jd = (now - datetime.datetime(2000, 1, 1, 12, 0, 0)).total_seconds() / 86400.0 + 2451545.0
        lst = (280.46061837 + 360.98564736629 * (jd - 2451545.0) + self.config.latitude) % 360.0
        lst_hours = lst / 15.0

        ha = (lst_hours - ra_hours) * 15.0
        ha_r = math.radians(ha)
        dec_r = math.radians(dec_degrees)

        sin_alt = math.sin(dec_r) * math.sin(lat) + math.cos(dec_r) * math.cos(lat) * math.cos(ha_r)
        alt = math.degrees(math.asin(max(-1.0, min(1.0, sin_alt))))

        cos_az = (math.sin(dec_r) - math.sin(lat) * sin_alt) / (math.cos(lat) * math.cos(math.radians(alt)) + 1e-10)
        sin_az = -math.cos(dec_r) * math.sin(ha_r) / (math.cos(math.radians(alt)) + 1e-10)
        az = math.degrees(math.atan2(sin_az, cos_az)) % 360.0

        return alt, az

    # ------------------------------------------------------------------
    # High-level GOTO / sync / stop (mirrors TelescopeBridge interface)
    # ------------------------------------------------------------------

    def force_position_update(self):
        """Immediately fire on_altaz_update with current simulated position.

        Matches the TelescopeBridge.force_position_update() interface so the
        GUI can request an instant position refresh after sync or GOTO.
        Callback receives 3 args: (alt_str, az_str, is_slewing).
        """
        if not self.is_connected:
            return
        with self._lock:
            self._tick()
            alt, az = self._get_noisy_position()
            alt_str = _deg_to_dms(alt, signed=True)
            az_str = _deg_to_dms(az, signed=False)
            is_slewing = self._is_slewing
        if self.on_altaz_update:
            self.on_altaz_update(alt_str, az_str, is_slewing)

    def goto_altaz(self, alt_str: str, az_str: str) -> bool:
        """Initiate a GOTO to the specified Alt/Az coordinates.

        Args:
            alt_str: Target altitude in LX200 format (e.g. "+45*30:00").
            az_str: Target azimuth in LX200 format (e.g. "180*00:00").

        Returns:
            True if the command was accepted.
        """
        if not self.is_connected:
            return False
        self.send_command(f":Sa{alt_str}#")
        self.send_command(f":Sz{az_str}#")
        result = self.send_command(":MA#")
        return "0" in result

    def goto(self, ra: str, dec: str) -> bool:
        """Legacy equatorial GOTO.

        Args:
            ra: Target RA in LX200 format (e.g. "12:30:00").
            dec: Target Dec in LX200 format (e.g. "+45*30:00").

        Returns:
            True if the command was accepted.
        """
        if not self.is_connected:
            return False
        self.send_command(f":Sr{ra}#")
        self.send_command(f":Sd{dec}#")
        result = self.send_command(":MS#")
        return "0" in result

    def stop(self):
        """Stop all simulated motion."""
        if self.is_connected:
            self.send_command(":Q#")
            self._log("Simulator: all motion stopped")

    def sync_altaz(self, alt_str: str, az_str: str) -> bool:
        """Sync the simulated position to the given Alt/Az.

        Args:
            alt_str: True altitude in LX200 format.
            az_str: True azimuth in LX200 format.

        Returns:
            True if accepted.
        """
        if not self.is_connected:
            return False
        self.send_command(f":Sa{alt_str}#")
        self.send_command(f":Sz{az_str}#")
        result = self.send_command(":CM#")
        return bool(result)

    def sync(self, ra: str, dec: str) -> bool:
        """Sync via RA/Dec coordinates.

        Args:
            ra: True RA in LX200 format.
            dec: True Dec in LX200 format.

        Returns:
            True if accepted.
        """
        if not self.is_connected:
            return False
        self.send_command(f":Sr{ra}#")
        self.send_command(f":Sd{dec}#")
        result = self.send_command(":CM#")
        return bool(result)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, message: str):
        """Log a message through the callback or module logger.

        Args:
            message: Diagnostic message to log.
        """
        if self.on_log:
            self.on_log(message)
        else:
            _logger.info(message)
