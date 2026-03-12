"""
Mount Protocol Abstraction Layer - Multi-protocol telescope mount communication.

This module defines the abstract interface (``MountProtocol``) for communicating
with different telescope mount firmwares, and provides concrete implementations
for:

    - **LX200 / OnStep**  (``LX200MountProtocol``)  -- Meade LX200 ASCII
      command set as extended by OnStep.

    - **NexStar / SynScan**  (``NexStarMountProtocol``)  -- Celestron NexStar
      and Sky-Watcher SynScan hand-controller protocol.

    - **iOptron**  (``iOptronMountProtocol``)  -- iOptron Alt-Az mounts
      (AZ Mount Pro, HAE29, CubePro) using the iOptron Command Language.

    - **Meade AudioStar**  (``MeadeAudioStarMountProtocol``)  -- Meade
      AudioStar / AutoStar hand controllers (ETX, LX90, LX200GPS, etc.).

    - **ASCOM Alpaca**  (``ASCOMAlpacaMountProtocol``)  -- ASCOM Alpaca REST
      API for any ASCOM-compatible mount accessible over HTTP.

    - **INDI**  (``INDIClientMountProtocol``)  -- INDI XML-over-TCP client
      for any INDI-compatible mount driver.

Architecture
------------
The protocol layer is purely about **command building and response parsing**.
It does NOT own the transport (serial / TCP); instead, every method receives a
``send_fn`` callback from the caller (``TelescopeBridge``), which handles the
actual byte I/O.  This separation lets one bridge instance work with any
protocol, and makes testing easy (just provide a mock ``send_fn``).

::

    HEADLESS_SERVER / UI                  (high-level: "goto M42")
            |
            v
    MountProtocol.goto_radec(...)         (protocol: build commands, parse responses)
            |  uses send_fn(cmd) -> resp
            v
    TelescopeBridge._send_command_locked  (transport: serial / TCP byte I/O)
            |
            v
    Physical mount hardware


Coordinate encoding
-------------------
- LX200 / Meade: ASCII DMS strings (``+45*30:00#``).
- NexStar: 32-bit hex-encoded fractions of a full turn.
- iOptron: Integer arc-seconds * 100.
- ASCOM Alpaca: JSON floats (decimal degrees / hours).
- INDI: XML text (decimal degrees / hours).

Adding a new protocol
---------------------
1. Subclass ``MountProtocol``.
2. Implement all ``@abstractmethod`` members.
3. Register the new class in ``PROTOCOL_REGISTRY`` at the bottom.
"""

from __future__ import annotations

import socket
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from telescope_logger import get_logger

_logger = get_logger(__name__)


# ===================================================================
# Data classes shared by all protocols
# ===================================================================

@dataclass
class PositionData:
    """Result of a position poll from the mount.

    All angular fields are in decimal degrees (or hours for RA).
    The ``*_str`` fields carry the raw LX200-format strings so that
    existing callbacks (which expect DMS strings) keep working even
    when the source protocol is NexStar.
    """
    alt_deg: Optional[float] = None
    az_deg: Optional[float] = None
    ra_hours: Optional[float] = None
    dec_deg: Optional[float] = None
    is_slewing: bool = False
    focuser_position: Optional[str] = None

    # LX200-format strings for backward-compatible callbacks
    alt_str: Optional[str] = None
    az_str: Optional[str] = None
    ra_str: Optional[str] = None
    dec_str: Optional[str] = None


@dataclass
class CommandResult:
    """Result of a multi-step command sequence (goto, sync, site-setup, ...)."""
    success: bool
    message: str = ""
    details: List[Tuple[str, str]] = field(default_factory=list)


# ===================================================================
# SendFn type alias
# ===================================================================
# The bridge passes a callable with signature:
#   send_fn(command: str, timeout: float = 2.0) -> str
# The protocol calls it for every individual command exchange.
SendFn = Callable[..., str]


# ===================================================================
# MountProtocol ABC
# ===================================================================

class MountProtocol(ABC):
    """Abstract interface for telescope mount communication protocols.

    Subclasses encapsulate the wire format, command structure, response
    parsing, and interaction patterns for a specific mount firmware.
    """

    # --- Identity --------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable protocol name (e.g. 'LX200 / OnStep')."""

    @property
    @abstractmethod
    def default_baudrate(self) -> int:
        """Typical serial baud rate for this protocol."""

    @property
    @abstractmethod
    def default_tcp_port(self) -> int:
        """Typical TCP port for WiFi connections."""

    @property
    @abstractmethod
    def response_terminator(self) -> bytes:
        """Single byte that terminates every response (b'#' for both LX200 and NexStar)."""

    # --- Connection test -------------------------------------------

    @abstractmethod
    def test_connection(self, send_fn: SendFn) -> Tuple[bool, str, bool]:
        """Test connection and identify mount firmware.

        Args:
            send_fn: ``send_fn(cmd_bytes: bytes, timeout: float) -> str``

        Returns:
            ``(success, model_name, is_onstep)``
        """

    # --- Position polling ------------------------------------------

    @abstractmethod
    def poll_position(self, send_fn: SendFn) -> PositionData:
        """Poll current position (Alt/Az, RA/Dec, slew status, focuser).

        The implementation decides which commands to send and how to
        parse responses.  The bridge calls this from ``_read_loop``.
        """

    # --- GoTo ------------------------------------------------------

    @abstractmethod
    def goto_radec(self, ra_str: str, dec_str: str,
                   send_fn: SendFn) -> CommandResult:
        """GoTo by RA/Dec coordinates."""

    @abstractmethod
    def goto_altaz(self, alt_str: str, az_str: str,
                   send_fn: SendFn) -> CommandResult:
        """GoTo by horizontal coordinates (preferred for Alt-Az mounts)."""

    # --- Motion control --------------------------------------------

    @abstractmethod
    def slew(self, direction: str, speed: int,
             send_fn: SendFn) -> None:
        """Start slewing in *direction* ('N','S','E','W') at *speed* (1-4)."""

    @abstractmethod
    def stop(self, send_fn: SendFn) -> None:
        """Emergency stop -- halt all motion."""

    # --- Sync ------------------------------------------------------

    @abstractmethod
    def sync_radec(self, ra_str: str, dec_str: str,
                   send_fn: SendFn) -> CommandResult:
        """Sync mount position to RA/Dec."""

    @abstractmethod
    def sync_altaz(self, alt_str: str, az_str: str,
                   send_fn: SendFn) -> CommandResult:
        """Sync mount position to Alt/Az."""

    # --- Park / Home -----------------------------------------------

    @abstractmethod
    def park(self, send_fn: SendFn) -> None:
        """Park the mount."""

    @abstractmethod
    def home(self, send_fn: SendFn) -> None:
        """Home / reset the mount."""

    # --- Site / Time / Weather -------------------------------------

    @abstractmethod
    def set_site(self, lat: float, lon: float,
                 utc_offset_west_h: float,
                 send_fn: SendFn) -> CommandResult:
        """Send observer location + UTC offset to the mount."""

    @abstractmethod
    def set_time(self, dt: datetime,
                 send_fn: SendFn) -> CommandResult:
        """Send local date/time to the mount."""

    def set_weather(self, temp_c: float, pressure_hpa: float,
                    humidity_pct: float,
                    send_fn: SendFn) -> CommandResult:
        """Send weather data for atmospheric refraction (optional)."""
        return CommandResult(True, "Weather not supported by this protocol")

    # --- Focuser (optional) ----------------------------------------

    def focuser_move(self, direction: str, speed: int,
                     send_fn: SendFn) -> None:
        """Move focuser IN or OUT at *speed*."""

    def focuser_stop(self, send_fn: SendFn) -> None:
        """Stop the focuser."""

    # --- Derotator (optional) --------------------------------------

    def derotator_rotate(self, direction: str, speed: float,
                         send_fn: SendFn) -> None:
        """Rotate the field derotator CW or CCW."""

    def derotator_stop(self, send_fn: SendFn) -> None:
        """Stop the field derotator."""

    def derotator_sync(self, send_fn: SendFn) -> None:
        """Reset derotator angle to 0."""

    # --- Tracking (pass-through) -----------------------------------

    def send_tracking_command(self, cmd: str,
                              send_fn: SendFn) -> str:
        """Forward a tracking-rate command. Default: pass through."""
        return send_fn(cmd)

    def send_variable_rate_altaz(self, alt_rate_arcsec: float,
                                  az_rate_arcsec: float,
                                  send_fn: SendFn) -> None:
        """Send variable tracking rates directly in Alt/Az (arcsec/sec).

        This is the preferred method for the real-time tracking controller
        when operating on Alt/Az mounts.  The base class raises
        ``NotImplementedError``; protocols that support direct variable-rate
        slewing (NexStar passthrough) override this.

        For protocols that do NOT support variable-rate Alt/Az (e.g. LX200),
        the tracking controller falls back to its own RA/Dec conversion and
        sends SXTR/SXTD commands via ``send_tracking_command()``.
        """
        raise NotImplementedError(
            f"{self.name} does not support direct variable-rate Alt/Az slewing"
        )

    @property
    def supports_variable_rate_altaz(self) -> bool:
        """Whether the protocol can accept direct variable-rate Alt/Az commands."""
        return False

    # --- Unpark / Set park position (optional) ---------------------

    def unpark(self, send_fn: SendFn) -> None:
        """Unpark the mount (restore parked position)."""

    def set_park_position(self, send_fn: SendFn) -> None:
        """Set the current position as the park position."""

    # --- Tracking rate (optional) ----------------------------------

    def set_tracking_rate(self, rate: str,
                          send_fn: SendFn) -> None:
        """Set tracking rate: 'sidereal', 'lunar', 'solar', 'king'."""

    def enable_tracking(self, send_fn: SendFn) -> None:
        """Enable mount tracking."""

    def disable_tracking(self, send_fn: SendFn) -> None:
        """Disable mount tracking."""

    # --- Mount-side PEC (optional) ---------------------------------

    def pec_record_start(self, send_fn: SendFn) -> None:
        """Start PEC recording on the mount."""

    def pec_record_stop(self, send_fn: SendFn) -> None:
        """Stop PEC recording on the mount."""

    def pec_playback_start(self, send_fn: SendFn) -> None:
        """Start PEC playback on the mount."""

    def pec_playback_stop(self, send_fn: SendFn) -> None:
        """Stop PEC playback on the mount."""

    def pec_clear(self, send_fn: SendFn) -> None:
        """Clear PEC data on the mount."""

    def pec_write_eeprom(self, send_fn: SendFn) -> None:
        """Write PEC data to mount EEPROM."""

    def pec_read_eeprom(self, send_fn: SendFn) -> None:
        """Read PEC data from mount EEPROM."""

    def get_pec_status(self, send_fn: SendFn) -> Optional[str]:
        """Query mount-side PEC status. Returns None if unsupported."""
        return None

    # --- Mount configuration (optional) ----------------------------

    def set_backlash(self, axis: str, value: int,
                     send_fn: SendFn) -> CommandResult:
        """Set backlash for *axis* ('ra' or 'dec') in steps."""
        return CommandResult(True, "Backlash not supported by this protocol")

    def get_backlash(self, axis: str,
                     send_fn: SendFn) -> Optional[int]:
        """Get backlash value (steps) for *axis*."""
        return None

    def set_horizon_limit(self, degrees: int,
                          send_fn: SendFn) -> CommandResult:
        """Set horizon limit in degrees."""
        return CommandResult(True, "Limits not supported by this protocol")

    def set_overhead_limit(self, degrees: int,
                           send_fn: SendFn) -> CommandResult:
        """Set overhead limit in degrees."""
        return CommandResult(True, "Limits not supported by this protocol")

    def get_horizon_limit(self, send_fn: SendFn) -> Optional[int]:
        """Get horizon limit (degrees)."""
        return None

    def get_overhead_limit(self, send_fn: SendFn) -> Optional[int]:
        """Get overhead limit (degrees)."""
        return None

    # --- Auxiliary features (optional) -----------------------------

    def set_auxiliary(self, feature_id: int, value: int,
                     send_fn: SendFn) -> CommandResult:
        """Set auxiliary feature value (0-255)."""
        return CommandResult(True, "Auxiliary features not supported")

    def get_auxiliary(self, feature_id: int,
                     send_fn: SendFn) -> Optional[str]:
        """Get auxiliary feature current value."""
        return None

    # --- Firmware info (optional) ----------------------------------

    def get_firmware_info(self, send_fn: SendFn) -> Dict[str, str]:
        """Get firmware product name, version, and mount type."""
        return {}

    # --- Reticle / LED control (optional) --------------------------

    def reticle_brighter(self, send_fn: SendFn) -> None:
        """Increase reticle/LED brightness."""
        pass

    def reticle_dimmer(self, send_fn: SendFn) -> None:
        """Decrease reticle/LED brightness."""
        pass

    # --- Alignment / Pointing state (optional) ---------------------

    def is_aligned(self, send_fn: SendFn) -> Optional[bool]:
        """Query whether the mount alignment is complete. Returns None if unsupported."""
        return None

    def get_pointing_state(self, send_fn: SendFn) -> Optional[str]:
        """Query mount pointing state. Returns None (not used for Alt-Az mounts)."""
        return None

    # --- Command formatting helpers --------------------------------

    @abstractmethod
    def format_outgoing(self, command: str) -> str:
        """Add protocol-specific delimiters to a raw command string.

        For LX200: ensure leading ':' and trailing '#'.
        For NexStar: no-op (commands have no standard delimiters).
        """

    def get_command_delay(self, command: str) -> float:
        """Pre-send delay (seconds) for firmware to be ready."""
        return 0.15

    def get_read_timeout(self, command: str) -> float:
        """Read timeout (seconds) for the response."""
        return 1.0

    def normalize_response(self, command: str, response: str,
                           log_fn: Callable[[str], None] = lambda _: None) -> str:
        """Post-process a response from the mount (optional).

        The transport layer calls this after reading the raw response.
        Protocols can override to apply firmware-specific quirk handling
        (e.g. OnStep returns position strings instead of ACK for ``:Sr``/``:Sd``).

        Default: return *response* unchanged.
        """
        return response


# ===================================================================
# Helpers
# ===================================================================

def _parse_lx200_dms(s: str) -> float:
    """Parse an LX200 DMS / HMS string to a decimal value.

    Handles formats like ``+45*30:00#``, ``12:30:00.5#``, ``180*00#``.
    Returns 0.0 on failure.
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


def _deg_to_lx200_alt(deg: float) -> str:
    """Format decimal degrees as LX200 altitude: ``sDD*MM:SS``."""
    sign = '+' if deg >= 0 else '-'
    d = abs(deg)
    dd = int(d)
    mm = int((d - dd) * 60)
    ss = int(((d - dd) * 60 - mm) * 60 + 0.5)
    if ss >= 60:
        ss = 0
        mm += 1
    if mm >= 60:
        mm = 0
        dd += 1
    return f"{sign}{dd:02d}*{mm:02d}:{ss:02d}"


def _deg_to_lx200_az(deg: float) -> str:
    """Format decimal degrees as LX200 azimuth: ``DDD*MM:SS``."""
    deg = deg % 360
    dd = int(deg)
    mm = int((deg - dd) * 60)
    ss = int(((deg - dd) * 60 - mm) * 60 + 0.5)
    if ss >= 60:
        ss = 0
        mm += 1
    if mm >= 60:
        mm = 0
        dd += 1
    return f"{dd:03d}*{mm:02d}:{ss:02d}"


# ===================================================================
# LX200MountProtocol
# ===================================================================

class LX200MountProtocol(MountProtocol):
    """LX200 / OnStep protocol implementation.

    This is the existing protocol used by this application.  All the
    hardcoded ``:GA#``, ``:Sr...#`` command strings that were previously
    scattered across telescope_bridge.py, HEADLESS_SERVER.py, and
    android_bridge/main.py are now centralized here.
    """

    @property
    def name(self) -> str:
        return "LX200 / OnStep"

    @property
    def default_baudrate(self) -> int:
        return 9600

    @property
    def default_tcp_port(self) -> int:
        # Port 9996 is the PERSISTENT command channel on the OnStep
        # SmartWebServer.  Port 9999 (STANDARD) has a hard 1-second
        # connection timeout with no keep-alive, which forces constant
        # reconnection.  9996 has a 10-second timeout that resets on
        # every byte received, so a polling client stays connected
        # indefinitely.
        return 9996

    @property
    def response_terminator(self) -> bytes:
        return b'#'

    # --- Connection test -------------------------------------------

    def test_connection(self, send_fn: SendFn) -> Tuple[bool, str, bool]:
        test_cmds = [
            (b":GVP#", "product name"),
            (b":GVN#", "firmware version"),
            (b":GR#", "right ascension"),
            (b":GD#", "declination"),
        ]
        model = "Unknown"
        is_onstep = False

        for cmd_bytes, cmd_name in test_cmds:
            try:
                resp = send_fn(cmd_bytes, 1.5)
                if resp and resp.endswith('#'):
                    if cmd_bytes == b":GVP#":
                        model = resp.rstrip('#').strip()
                        if "On-Step" in model or "OnStep" in model:
                            is_onstep = True
                    else:
                        model = f"LX200 Telescope (responds to {cmd_name})"
                    return True, model, is_onstep
                elif resp:
                    # Partial but non-empty
                    if cmd_bytes == b":GVP#":
                        model = resp.rstrip('#').strip() or "LX200 Telescope"
                        is_onstep = "On" in model
                    return True, model, is_onstep
            except Exception:
                continue

        return False, model, is_onstep

    # --- Position polling ------------------------------------------

    @staticmethod
    def _is_dms_like(body: str) -> bool:
        """Return True if *body* (``#`` already stripped) looks like a
        DMS/HMS coordinate (contains ``*`` or has colons with digits).
        Used to guard against response-shift where a focuser value
        like ``25000`` is mistaken for a coordinate.
        """
        return '*' in body

    @staticmethod
    def _is_hms_like(body: str) -> bool:
        """Return True if *body* looks like HH:MM:SS (RA format).
        Must have colons separating digits but NO ``*``.
        """
        if '*' in body:
            return False
        parts = body.split(':')
        if len(parts) < 2:
            return False
        try:
            int(parts[0])
            int(parts[1])
            return True
        except (ValueError, IndexError):
            return False

    @staticmethod
    def _is_numeric(body: str) -> bool:
        """Return True if *body* is purely numeric (focuser position)."""
        try:
            int(body)
            return True
        except ValueError:
            return False

    def poll_position(self, send_fn: SendFn) -> PositionData:
        result = PositionData()

        # -------------------------------------------------------
        # Response validation helpers.
        #
        # The SmartWebServer ESP8266 has a shared serial line with
        # NO mutex.  Background tasks (:GU#, :A?#, etc.) and any
        # open web page (:GA#, :GD#, etc.) can inject commands
        # that cause "response shift" — our read picks up the
        # PREVIOUS command's answer.  To detect this we validate
        # every response against the EXPECTED format for the
        # command we actually sent.  Mismatches are discarded.
        #
        # Expected formats (from OnStep/LX200):
        #   :D#  -> '#' (idle) or content+'#' (slewing, may have '|')
        #           MUST NOT be purely numeric (that's :FG#)
        #           MUST NOT contain '*' or ':' (those are coords)
        #   :GU# -> flag chars like NpAT... ending with '#'
        #           MUST NOT contain '*' (not a coordinate)
        #   :GA# -> sDD*MM:SS# (altitude) — MUST have '*'
        #   :GZ# -> DDD*MM:SS# (azimuth) — MUST have '*'
        #   :GR# -> HH:MM:SS# (RA) — MUST have ':' and NO '*'
        #   :GD# -> sDD*MM:SS# (Dec) — MUST have '*'
        #   :FG# -> NNNNN# (focuser) — purely numeric
        # -------------------------------------------------------

        # --- Slew check: :D# and :GU# ---
        # NOTE: No sleeps between commands — OnStep responds in ~10 ms.
        # Extra idle time causes the ESP8266/ESP32 WiFi adapter to close
        # the TCP connection.  ConnectionError / BrokenPipeError are
        # allowed to propagate so the caller can detect a dead link.
        d_slewing = False
        gu_slewing = False

        try:
            d_resp = send_fn(":D#")
            if d_resp and d_resp.endswith('#'):
                body = d_resp.rstrip('#')
                # Valid :D# responses: empty body (idle) or body with
                # bar/space chars (slewing).  Reject if it looks like
                # a coordinate or focuser value (response shift).
                if body == '':
                    d_slewing = False
                elif self._is_dms_like(body) or self._is_hms_like(body):
                    pass  # Response shift — discard
                elif self._is_numeric(body):
                    pass  # Likely :FG# response — discard
                else:
                    d_slewing = True  # Genuinely slewing
        except (ConnectionError, BrokenPipeError, ConnectionResetError):
            raise
        except Exception:
            pass

        try:
            gu_resp = send_fn(":GU#")
            if gu_resp and gu_resp.endswith('#'):
                body = gu_resp.rstrip('#')
                # :GU# returns status flags (letters like NpAT...).
                # Reject if it looks like a coordinate or number.
                if not self._is_dms_like(body) and not self._is_numeric(body):
                    if 'n' in body and 'N' not in body:
                        gu_slewing = True
        except (ConnectionError, BrokenPipeError, ConnectionResetError):
            raise
        except Exception:
            pass

        result.is_slewing = d_slewing or gu_slewing

        # --- Alt/Az ---
        try:
            alt_resp = send_fn(":GA#")
            if alt_resp and alt_resp.endswith('#'):
                body = alt_resp.rstrip('#')
                # Altitude must be DMS-like (contains '*')
                if self._is_dms_like(body):
                    result.alt_str = alt_resp
                    result.alt_deg = _parse_lx200_dms(alt_resp)
        except (ConnectionError, BrokenPipeError, ConnectionResetError):
            raise
        except Exception:
            pass

        try:
            az_resp = send_fn(":GZ#")
            if az_resp and az_resp.endswith('#'):
                body = az_resp.rstrip('#')
                # Azimuth must be DMS-like (contains '*')
                if self._is_dms_like(body):
                    result.az_str = az_resp
                    result.az_deg = _parse_lx200_dms(az_resp)
        except (ConnectionError, BrokenPipeError, ConnectionResetError):
            raise
        except Exception:
            pass

        # --- RA/Dec ---
        try:
            ra_resp = send_fn(":GR#")
            if ra_resp and ra_resp.endswith('#'):
                body = ra_resp.rstrip('#')
                # RA must be HMS-like (HH:MM:SS, no '*')
                if self._is_hms_like(body):
                    result.ra_str = ra_resp
                    result.ra_hours = _parse_lx200_dms(ra_resp)
        except (ConnectionError, BrokenPipeError, ConnectionResetError):
            raise
        except Exception:
            pass

        try:
            dec_resp = send_fn(":GD#")
            if dec_resp and dec_resp.endswith('#'):
                body = dec_resp.rstrip('#')
                # Dec must be DMS-like (contains '*')
                if self._is_dms_like(body):
                    result.dec_str = dec_resp
                    result.dec_deg = _parse_lx200_dms(dec_resp)
        except (ConnectionError, BrokenPipeError, ConnectionResetError):
            raise
        except Exception:
            pass

        # --- Focuser ---
        try:
            fg_resp = send_fn(":FG#")
            if fg_resp and fg_resp.endswith('#'):
                body = fg_resp.rstrip('#').strip()
                # Focuser must be purely numeric
                if self._is_numeric(body):
                    result.focuser_position = body
        except (ConnectionError, BrokenPipeError, ConnectionResetError):
            raise
        except Exception:
            pass

        return result

    # --- GoTo ------------------------------------------------------

    def goto_radec(self, ra_str: str, dec_str: str,
                   send_fn: SendFn) -> CommandResult:
        details = []

        ra_resp = send_fn(f":Sr{ra_str}#")
        details.append((f":Sr{ra_str}#", ra_resp))
        dec_resp = send_fn(f":Sd{dec_str}#")
        details.append((f":Sd{dec_str}#", dec_resp))
        ms_resp = send_fn(":MS#")
        details.append((":MS#", ms_resp))

        ms_clean = (ms_resp or "").strip().rstrip('#')
        # "0" = LX200 success.  Empty string means no response from the
        # mount -- treat as failure (command was likely lost).
        if not ms_clean:
            return CommandResult(
                success=False,
                message="No response to :MS# -- mount may not be connected",
                details=details,
            )
        ok = ms_clean == "0"
        msg = "GoTo accepted" if ok else f"GoTo refused: {ms_resp}"
        return CommandResult(success=ok, message=msg, details=details)

    def goto_altaz(self, alt_str: str, az_str: str,
                   send_fn: SendFn) -> CommandResult:
        details = []

        alt_resp = send_fn(f":Sa{alt_str}#")
        details.append((f":Sa{alt_str}#", alt_resp))
        az_resp = send_fn(f":Sz{az_str}#")
        details.append((f":Sz{az_str}#", az_resp))
        ma_resp = send_fn(":MA#")
        details.append((":MA#", ma_resp))

        ma_clean = (ma_resp or "").strip().rstrip('#')
        # Empty response = mount didn't reply (command lost)
        if not ma_clean:
            return CommandResult(
                success=False,
                message="No response to :MA# -- mount may not be connected",
                details=details,
            )
        ok = ma_clean == "0" or "*" in ma_clean or ":" in ma_clean
        msg = "Alt/Az GoTo accepted" if ok else f"Alt/Az GoTo refused: {ma_resp}"
        return CommandResult(success=ok, message=msg, details=details)

    # --- Motion control --------------------------------------------

    def slew(self, direction: str, speed: int,
             send_fn: SendFn) -> None:
        speed_cmds = {1: ":RG#", 2: ":RC#", 3: ":RM#", 4: ":RS#"}
        dir_cmds = {"N": ":Mn#", "S": ":Ms#", "E": ":Me#", "W": ":Mw#"}
        send_fn(speed_cmds.get(speed, ":RS#"))
        if direction in dir_cmds:
            send_fn(dir_cmds[direction])

    def stop(self, send_fn: SendFn) -> None:
        send_fn(":Q#")

    # --- Sync ------------------------------------------------------

    def sync_radec(self, ra_str: str, dec_str: str,
                   send_fn: SendFn) -> CommandResult:
        details = []
        details.append((f":Sr{ra_str}#", send_fn(f":Sr{ra_str}#")))
        details.append((f":Sd{dec_str}#", send_fn(f":Sd{dec_str}#")))
        cm_resp = send_fn(":CM#")
        details.append((":CM#", cm_resp))
        ok = bool(cm_resp)
        return CommandResult(success=ok, message="Synced" if ok else "Sync failed",
                             details=details)

    def sync_altaz(self, alt_str: str, az_str: str,
                   send_fn: SendFn) -> CommandResult:
        details = []
        details.append((f":Sa{alt_str}#", send_fn(f":Sa{alt_str}#")))
        details.append((f":Sz{az_str}#", send_fn(f":Sz{az_str}#")))
        cm_resp = send_fn(":CM#")
        details.append((":CM#", cm_resp))
        ok = bool(cm_resp)
        return CommandResult(success=ok, message="Synced" if ok else "Sync failed",
                             details=details)

    # --- Park / Home -----------------------------------------------

    def park(self, send_fn: SendFn) -> None:
        send_fn(":hP#")

    def home(self, send_fn: SendFn) -> None:
        send_fn(":hF#")

    # --- Site / Time / Weather -------------------------------------

    def set_site(self, lat: float, lon: float,
                 utc_offset_west_h: float,
                 send_fn: SendFn) -> CommandResult:
        details = []

        # Latitude: :St sDD*MM#
        lat_sign = '+' if lat >= 0 else '-'
        lat_abs = abs(lat)
        lat_deg = int(lat_abs)
        lat_min = int((lat_abs - lat_deg) * 60 + 0.5)
        if lat_min >= 60:
            lat_deg += 1; lat_min = 0
        lat_cmd = f":St{lat_sign}{lat_deg:02d}*{lat_min:02d}#"
        resp = send_fn(lat_cmd)
        details.append((lat_cmd, resp))

        # Longitude: :Sg DDD*MM#  (OnStep: 0-360 west-positive)
        onstep_lon = (-lon) % 360
        lon_deg = int(onstep_lon)
        lon_min = int((onstep_lon - lon_deg) * 60 + 0.5)
        if lon_min >= 60:
            lon_deg += 1; lon_min = 0
        lon_cmd = f":Sg{lon_deg:03d}*{lon_min:02d}#"
        resp = send_fn(lon_cmd)
        details.append((lon_cmd, resp))

        # UTC offset: :SG sHH.H#
        sign = '+' if utc_offset_west_h >= 0 else '-'
        offset_cmd = f":SG{sign}{abs(utc_offset_west_h):04.1f}#"
        resp = send_fn(offset_cmd)
        details.append((offset_cmd, resp))

        return CommandResult(success=True, message="Site sent", details=details)

    def set_time(self, dt: datetime,
                 send_fn: SendFn) -> CommandResult:
        details = []

        time_cmd = f":SL{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}#"
        resp = send_fn(time_cmd)
        details.append((time_cmd, resp))

        date_cmd = f":SC{dt.month:02d}/{dt.day:02d}/{dt.year % 100:02d}#"
        resp = send_fn(date_cmd)
        details.append((date_cmd, resp))

        return CommandResult(success=True, message="Time sent", details=details)

    def set_weather(self, temp_c: float, pressure_hpa: float,
                    humidity_pct: float,
                    send_fn: SendFn) -> CommandResult:
        details = []
        for label, cmd in [
            ("temp", f":SX9A,{temp_c:.1f}#"),
            ("pressure", f":SX9B,{pressure_hpa:.1f}#"),
            ("humidity", f":SX9C,{humidity_pct:.0f}#"),
        ]:
            resp = send_fn(cmd)
            details.append((cmd, resp))
        return CommandResult(success=True, message="Weather sent", details=details)

    # --- Focuser (OnStepX extended) --------------------------------
    #
    # OnStepX focuser commands:
    #   :FA1#..:FA6# -- Select active focuser (returns 0/1)
    #   :F1#..:F9#   -- Set move/goto rate preset
    #   :F+#         -- Move inward
    #   :F-#         -- Move outward
    #   :FQ#         -- Stop
    #   :FG#         -- Get position in microns (returns sn#)
    #   :Fg#         -- Get position in steps (returns sn#)
    #   :FSn#        -- Absolute goto in microns (returns 0/1)
    #   :Fsn#        -- Absolute goto in steps (returns 0/1)
    #   :FZ#         -- Zero current position
    #   :FH#         -- Set current position as home
    #   :Fh#         -- Move to home
    #   :Ft#         -- Get temperature (returns n.n#)
    #   :Fc#         -- TCF enabled status (returns 0 or 1)
    #   :Fc0#/:Fc1#  -- Disable/Enable TCF
    #   :FC#         -- Get TCF coefficient (returns n.nnnnn#)
    #   :FCsn.n#     -- Set TCF coefficient (returns 0/1)
    #   :FB#         -- Get backlash in microns (returns n#)
    #   :FBn#        -- Set backlash in microns (returns 0/1)
    #   :FT#         -- Focuser status (returns M1#, S3#, etc.)

    def focuser_move(self, direction: str, speed: int,
                     send_fn: SendFn) -> None:
        lx200_speed = max(1, min(4, (speed - 1) // 5 + 1))
        send_fn(f":F{lx200_speed}#")
        send_fn(":F+#" if direction == "IN" else ":F-#")

    def focuser_stop(self, send_fn: SendFn) -> None:
        send_fn(":FQ#")

    def focuser_goto(self, position: int,
                     send_fn: SendFn) -> CommandResult:
        """Move focuser to absolute position in microns."""
        resp = send_fn(f":FS{position}#")
        ok = resp and '1' in resp
        return CommandResult(ok, f"Focuser goto {position}" if ok else f"Failed: {resp}")

    def focuser_zero(self, send_fn: SendFn) -> None:
        """Zero the focuser position counter."""
        send_fn(":FZ#")

    def focuser_home(self, send_fn: SendFn) -> None:
        """Move focuser to home position."""
        send_fn(":Fh#")

    def focuser_set_home(self, send_fn: SendFn) -> None:
        """Set current focuser position as home."""
        send_fn(":FH#")

    def focuser_get_temperature(self, send_fn: SendFn) -> Optional[float]:
        """Get focuser temperature in Celsius."""
        try:
            resp = send_fn(":Ft#")
            if resp:
                return float(resp.rstrip('#').strip())
        except Exception:
            pass
        return None

    def focuser_get_tcf_enabled(self, send_fn: SendFn) -> Optional[bool]:
        """Get TCF (temperature compensation) enabled status."""
        try:
            resp = send_fn(":Fc#")
            if resp:
                return resp.rstrip('#').strip() == '1'
        except Exception:
            pass
        return None

    def focuser_set_tcf(self, enabled: bool,
                        send_fn: SendFn) -> None:
        """Enable or disable temperature compensation."""
        send_fn(":Fc1#" if enabled else ":Fc0#")

    def focuser_select(self, focuser_num: int,
                       send_fn: SendFn) -> CommandResult:
        """Select active focuser (1-6)."""
        if 1 <= focuser_num <= 6:
            resp = send_fn(f":FA{focuser_num}#")
            ok = resp and '1' in resp
            return CommandResult(ok, f"Focuser {focuser_num} selected" if ok else f"Failed: {resp}")
        return CommandResult(False, f"Invalid focuser number: {focuser_num}")

    def focuser_get_status(self, send_fn: SendFn) -> Optional[str]:
        """Get focuser status string (e.g. 'M1', 'S3')."""
        try:
            resp = send_fn(":FT#")
            if resp:
                return resp.rstrip('#').strip()
        except Exception:
            pass
        return None

    # --- Rotator (OnStepX) -----------------------------------------
    #
    # OnStepX rotator commands:
    #   :rG#          -- Get current angle (sDDD*MM#)
    #   :rSsDDD*MM#   -- Absolute goto (returns 0/1)
    #   :rrsDDD*MM#   -- Relative goto
    #   :rQ#          -- Stop
    #   :r>#          -- Move clockwise
    #   :r<#          -- Move counter-clockwise
    #   :rZ#          -- Zero position
    #   :r+#          -- Enable derotation
    #   :r-#          -- Disable derotation
    #   :rR#          -- Toggle reverse direction
    #   :rP#          -- Move to parallactic angle
    #   :rT#          -- Status string
    #   :r1#..:r9#    -- Set rate preset

    def rotator_move_cw(self, send_fn: SendFn) -> None:
        """Start continuous clockwise rotation."""
        send_fn(":r>#")

    def rotator_move_ccw(self, send_fn: SendFn) -> None:
        """Start continuous counter-clockwise rotation."""
        send_fn(":r<#")

    def rotator_stop(self, send_fn: SendFn) -> None:
        """Stop rotator."""
        send_fn(":rQ#")

    def rotator_goto(self, angle_deg: float,
                     send_fn: SendFn) -> CommandResult:
        """Go to absolute angle. Format: sDDD*MM."""
        sign = '+' if angle_deg >= 0 else '-'
        deg = int(abs(angle_deg))
        mins = int((abs(angle_deg) - deg) * 60 + 0.5)
        if mins >= 60:
            deg += 1; mins = 0
        resp = send_fn(f":rS{sign}{deg:03d}*{mins:02d}#")
        ok = resp and '1' in resp
        return CommandResult(ok, f"Rotator goto {angle_deg}" if ok else f"Failed: {resp}")

    def rotator_get_angle(self, send_fn: SendFn) -> Optional[float]:
        """Get current rotator angle in degrees."""
        try:
            resp = send_fn(":rG#")
            if resp:
                body = resp.rstrip('#').strip()
                # Parse sDDD*MM format
                sign = -1 if body.startswith('-') else 1
                body = body.lstrip('+-')
                parts = body.split('*')
                if len(parts) == 2:
                    deg = int(parts[0])
                    mins = int(parts[1])
                    return sign * (deg + mins / 60.0)
        except Exception:
            pass
        return None

    def rotator_zero(self, send_fn: SendFn) -> None:
        """Zero the rotator position."""
        send_fn(":rZ#")

    def rotator_enable_derotation(self, send_fn: SendFn) -> None:
        """Enable field derotation."""
        send_fn(":r+#")

    def rotator_disable_derotation(self, send_fn: SendFn) -> None:
        """Disable field derotation."""
        send_fn(":r-#")

    def rotator_reverse(self, send_fn: SendFn) -> None:
        """Toggle rotator reverse direction."""
        send_fn(":rR#")

    def rotator_parallactic(self, send_fn: SendFn) -> None:
        """Move to parallactic angle."""
        send_fn(":rP#")

    def rotator_set_rate(self, rate: int,
                         send_fn: SendFn) -> None:
        """Set rotator rate preset (1-9)."""
        if 1 <= rate <= 9:
            send_fn(f":r{rate}#")

    def rotator_get_status(self, send_fn: SendFn) -> Optional[str]:
        """Get rotator status string."""
        try:
            resp = send_fn(":rT#")
            if resp:
                return resp.rstrip('#').strip()
        except Exception:
            pass
        return None

    # --- Derotator (legacy custom protocol) ------------------------

    def derotator_rotate(self, direction: str, speed: float,
                         send_fn: SendFn) -> None:
        cmd = f":DR+{speed}#" if direction == "CW" else f":DR-{speed}#"
        send_fn(cmd)

    def derotator_stop(self, send_fn: SendFn) -> None:
        send_fn(":DRQ#")

    def derotator_sync(self, send_fn: SendFn) -> None:
        send_fn(":DR0#")

    # --- Unpark / Set park position (OnStep) -----------------------

    def unpark(self, send_fn: SendFn) -> None:
        send_fn(":hR#")

    def set_park_position(self, send_fn: SendFn) -> None:
        send_fn(":hQ#")

    # --- Tracking rate (OnStep) ------------------------------------

    def set_tracking_rate(self, rate: str,
                          send_fn: SendFn) -> None:
        cmds = {
            "sidereal": ":TQ#",
            "lunar": ":TL#",
            "solar": ":TS#",
            "king": ":TK#",
        }
        cmd = cmds.get(rate.lower())
        if cmd:
            send_fn(cmd)

    def enable_tracking(self, send_fn: SendFn) -> None:
        send_fn(":Te#")

    def disable_tracking(self, send_fn: SendFn) -> None:
        send_fn(":Td#")

    def get_tracking_rate_hz(self, send_fn: SendFn) -> float:
        """Query tracking rate in Hz. Returns 0.0 if not tracking. :GT#"""
        resp = send_fn(":GT#")
        try:
            return float(resp.rstrip('#'))
        except (ValueError, AttributeError):
            return 0.0

    def set_tracking_rate_hz(self, freq: float,
                             send_fn: SendFn) -> None:
        """Set tracking rate in Hz. 0 stops tracking. :STn.n#"""
        send_fn(f":ST{freq:.5f}#")

    def set_tracking_axis_mode(self, mode: int,
                               send_fn: SendFn) -> None:
        """Set tracking axis mode: 1=single (RA), 2=dual (Alt-Az). :T1# / :T2#"""
        if mode in (1, 2):
            send_fn(f":T{mode}#")

    def set_compensation_model(self, model: str,
                               send_fn: SendFn) -> None:
        """Set compensation model: 'full' :To#, 'refraction' :Tr#, 'none' :Tn#."""
        cmds = {'full': ':To#', 'refraction': ':Tr#', 'none': ':Tn#'}
        cmd = cmds.get(model)
        if cmd:
            send_fn(cmd)

    def adjust_sidereal_clock(self, direction: str,
                              send_fn: SendFn) -> None:
        """Adjust master sidereal clock: '+' :T+#, '-' :T-#, 'reset' :TR#."""
        cmds = {'+': ':T+#', '-': ':T-#', 'reset': ':TR#'}
        cmd = cmds.get(direction)
        if cmd:
            send_fn(cmd)

    def set_backlash(self, axis: str, arcsec: float,
                     send_fn: SendFn) -> None:
        """Set backlash. axis='ra' :$BRn#, axis='dec' :$BDn#."""
        if axis == 'ra':
            send_fn(f":$BR{arcsec:.0f}#")
        elif axis == 'dec':
            send_fn(f":$BD{arcsec:.0f}#")

    def get_backlash(self, axis: str,
                     send_fn: SendFn) -> float:
        """Get backlash. axis='ra' :%BR#, axis='dec' :%BD#."""
        cmd = ':%BR#' if axis == 'ra' else ':%BD#'
        resp = send_fn(cmd)
        try:
            return float(resp.rstrip('#'))
        except (ValueError, AttributeError):
            return 0.0

    def get_tracking_rate_offsets(self,
                                 send_fn: SendFn) -> tuple:
        """Query SXTR/SXTD offsets. Returns (ra_offset, dec_offset)."""
        ra_resp = send_fn(":GXTR#")
        dec_resp = send_fn(":GXTD#")
        try:
            ra = float(ra_resp.rstrip('#'))
        except (ValueError, AttributeError):
            ra = 0.0
        try:
            dec = float(dec_resp.rstrip('#'))
        except (ValueError, AttributeError):
            dec = 0.0
        return ra, dec

    # --- Mount-side PEC (OnStepX) ----------------------------------
    #
    # OnStepX PEC commands (from COMMAND_REFERENCE.md):
    #   :$QZ+#  -- Enable PEC playback
    #   :$QZ-#  -- Disable PEC (stops both playback and recording)
    #   :$QZ/#  -- Arm PEC recording
    #   :$QZZ#  -- Clear PEC buffer
    #   :$QZ!#  -- Save PEC data to NV (EEPROM)
    #   :$QZ?#  -- PEC status: I=off, p=ready-play, P=playing,
    #                           r=ready-record, R=recording

    def pec_playback_start(self, send_fn: SendFn) -> None:
        send_fn(":$QZ+#")

    def pec_playback_stop(self, send_fn: SendFn) -> None:
        send_fn(":$QZ-#")

    def pec_record_start(self, send_fn: SendFn) -> None:
        send_fn(":$QZ/#")

    def pec_record_stop(self, send_fn: SendFn) -> None:
        send_fn(":$QZ-#")

    def pec_clear(self, send_fn: SendFn) -> None:
        send_fn(":$QZZ#")

    def pec_write_eeprom(self, send_fn: SendFn) -> None:
        send_fn(":$QZ!#")

    def pec_read_eeprom(self, send_fn: SendFn) -> None:
        """No-op: PEC data is read automatically from NV on boot."""
        pass

    def get_pec_status(self, send_fn: SendFn) -> Optional[str]:
        """Query PEC status via :$QZ?#.

        Returns status character(s):
            I  = off/ignore
            p  = ready to play
            P  = playing
            r  = ready to record
            R  = recording
            .  = index detected this second (appended)
        """
        try:
            resp = send_fn(":$QZ?#")
            if resp:
                return resp.rstrip('#').strip()
        except Exception:
            pass
        return None

    # --- Mount configuration (OnStepX) -----------------------------
    #
    # Backlash (arcsec):
    #   :$BRn#  -- Set RA/Azm backlash to n arcsec (returns 0/1)
    #   :$BDn#  -- Set Dec/Alt backlash to n arcsec (returns 0/1)
    #   :%BR#   -- Get RA/Azm backlash in arcsec (returns n#)
    #   :%BD#   -- Get Dec/Alt backlash in arcsec (returns n#)
    #
    # Limits:
    #   :ShsDD# -- Set horizon limit (returns 0/1)
    #   :SoDD#  -- Set overhead limit (returns 0/1)
    #   :Gh#    -- Get horizon limit (returns sDD*#)
    #   :Go#    -- Get overhead limit (returns DD*#)
    def set_backlash(self, axis: str, value: int,
                     send_fn: SendFn) -> CommandResult:
        if axis.lower() in ('ra', 'azm'):
            resp = send_fn(f":$BR{value}#")
        elif axis.lower() in ('dec', 'alt'):
            resp = send_fn(f":$BD{value}#")
        else:
            return CommandResult(False, f"Unknown axis: {axis}")
        ok = resp and '1' in resp
        return CommandResult(ok, f"Backlash {axis} set to {value} arcsec" if ok else f"Failed: {resp}")

    def get_backlash(self, axis: str,
                     send_fn: SendFn) -> Optional[int]:
        try:
            if axis.lower() in ('ra', 'azm'):
                resp = send_fn(":%BR#")
            elif axis.lower() in ('dec', 'alt'):
                resp = send_fn(":%BD#")
            else:
                return None
            if resp:
                val = resp.rstrip('#').strip()
                return int(val)
        except Exception:
            pass
        return None

    def set_horizon_limit(self, degrees: int,
                          send_fn: SendFn) -> CommandResult:
        sign = '+' if degrees >= 0 else '-'
        resp = send_fn(f":Sh{sign}{abs(degrees):02d}#")
        ok = resp and '1' in resp
        return CommandResult(ok, f"Horizon limit set to {degrees}" if ok else f"Failed: {resp}")

    def set_overhead_limit(self, degrees: int,
                           send_fn: SendFn) -> CommandResult:
        resp = send_fn(f":So{degrees:02d}#")
        ok = resp and '1' in resp
        return CommandResult(ok, f"Overhead limit set to {degrees}" if ok else f"Failed: {resp}")

    def get_horizon_limit(self, send_fn: SendFn) -> Optional[int]:
        try:
            resp = send_fn(":Gh#")
            if resp:
                # Response format: sDD*# (e.g. "+10*#" or "-05*#")
                body = resp.rstrip('#').strip().rstrip('*')
                return int(body)
        except Exception:
            pass
        return None

    def get_overhead_limit(self, send_fn: SendFn) -> Optional[int]:
        try:
            resp = send_fn(":Go#")
            if resp:
                # Response format: DD*# (e.g. "90*#")
                body = resp.rstrip('#').strip().rstrip('*')
                return int(body)
        except Exception:
            pass
        return None

    # --- Auxiliary features (OnStepX slot-based) -------------------
    #
    # Slot numbers 1-8. Commands:
    #   :GXY0#     -- Bitmap of active slots (8 chars, '1'=present)
    #   :GXYn#     -- Get slot n name and purpose (name,purpose#)
    #   :GXXn#     -- Get slot n current state
    #   :SXXn,Vv#  -- Set slot n value to v

    def get_auxiliary_bitmap(self, send_fn: SendFn) -> Optional[str]:
        """Get bitmap of active auxiliary feature slots (8 chars)."""
        try:
            resp = send_fn(":GXY0#")
            if resp:
                return resp.rstrip('#').strip()
        except Exception:
            pass
        return None

    def get_auxiliary_info(self, slot: int,
                          send_fn: SendFn) -> Optional[Dict[str, str]]:
        """Get auxiliary slot name and purpose code."""
        try:
            resp = send_fn(f":GXY{slot}#")
            if resp:
                body = resp.rstrip('#').strip()
                parts = body.split(',', 1)
                if len(parts) == 2:
                    return {'name': parts[0], 'purpose': parts[1]}
        except Exception:
            pass
        return None

    def set_auxiliary(self, feature_id: int, value: int,
                     send_fn: SendFn) -> CommandResult:
        cmd = f":SXX{feature_id},V{value}#"
        resp = send_fn(cmd)
        ok = resp and '1' in resp
        return CommandResult(ok, f"Aux {feature_id} set to {value}" if ok else f"Failed: {resp}")

    def get_auxiliary(self, feature_id: int,
                     send_fn: SendFn) -> Optional[str]:
        try:
            resp = send_fn(f":GXX{feature_id}#")
            if resp:
                return resp.rstrip('#').strip()
        except Exception:
            pass
        return None

    # --- Firmware info (OnStepX :GVP / :GVN / :GU#) ----------------

    def get_firmware_info(self, send_fn: SendFn) -> Dict[str, str]:
        """Query firmware product name, version, and mount status.

        Parses :GU# flags per OnStepX COMMAND_REFERENCE.md:
          Mount type:  E=GEM, K=Fork, A=AltAzm, L=AltAlt
          Pier side:   o=none, T=East, W=West
          Park state:  p=not parked, P=Parked, I=parking, F=park failed
          Tracking:    n=not tracking  (absence means tracking)
          PEC:         R=PEC recorded, /=ready-play, ,=playing,
                       ~=ready-record, ;=recording, ^=index detected
          Rates:       (=lunar, O=solar, k=King (absence=sidereal)
        """
        info: Dict[str, str] = {}
        try:
            resp = send_fn(":GVP#")
            if resp:
                info['product'] = resp.rstrip('#').strip()
        except Exception:
            pass
        try:
            resp = send_fn(":GVN#")
            if resp:
                info['version'] = resp.rstrip('#').strip()
        except Exception:
            pass
        try:
            resp = send_fn(":GU#")
            if resp:
                body = resp.rstrip('#').strip()
                info['status_flags'] = body
                # Mount type flags
                if 'E' in body:
                    info['mount_type'] = 'GEM'
                elif 'K' in body:
                    info['mount_type'] = 'Fork'
                elif 'L' in body:
                    info['mount_type'] = 'Alt-Alt'
                elif 'A' in body:
                    info['mount_type'] = 'Alt-Azimuth'
                # Park state: p=not parked, P=parked, I=parking, F=failed
                if 'P' in body and 'p' not in body:
                    info['park_state'] = 'Parked'
                elif 'I' in body:
                    info['park_state'] = 'Parking'
                elif 'F' in body:
                    info['park_state'] = 'Park Failed'
                elif 'p' in body:
                    info['park_state'] = 'Not Parked'
                # Tracking state
                if 'n' in body:
                    info['tracking'] = 'Off'
                else:
                    info['tracking'] = 'On'
                # Tracking rate
                if '(' in body:
                    info['tracking_rate'] = 'Lunar'
                elif 'O' in body:
                    info['tracking_rate'] = 'Solar'
                elif 'k' in body:
                    info['tracking_rate'] = 'King'
                else:
                    info['tracking_rate'] = 'Sidereal'
                # PEC state
                if ';' in body:
                    info['pec_state'] = 'Recording'
                elif ',' in body:
                    info['pec_state'] = 'Playing'
                elif '~' in body:
                    info['pec_state'] = 'Ready to Record'
                elif '/' in body:
                    info['pec_state'] = 'Ready to Play'
                else:
                    info['pec_state'] = 'Off'
                if 'R' in body:
                    info['pec_recorded'] = 'Yes'
                # At home
                if 'H' in body:
                    info['at_home'] = 'Yes'
        except Exception:
            pass
        return info

    # --- Reticle / LED control (LX200 :B+# / :B-#) ----------------

    def reticle_brighter(self, send_fn: SendFn) -> None:
        """Increase reticle/LED brightness (:B+#)."""
        try:
            send_fn(":B+#")
        except Exception:
            pass

    def reticle_dimmer(self, send_fn: SendFn) -> None:
        """Decrease reticle/LED brightness (:B-#)."""
        try:
            send_fn(":B-#")
        except Exception:
            pass

    # --- Command formatting ----------------------------------------

    def format_outgoing(self, command: str) -> str:
        if not command.startswith(':'):
            command = ':' + command
        if not command.endswith('#'):
            command = command + '#'
        return command

    def get_command_delay(self, command: str) -> float:
        if ':MS' in command:
            return 0.3
        if ':Sd' in command:
            return 0.3
        if ':Sr' in command:
            return 0.2
        return 0.15

    def get_read_timeout(self, command: str) -> float:
        if ':MS' in command or ':Sd' in command or ':Sr' in command:
            return 2.0
        if ':GR' in command or ':GD' in command or ':GA' in command or ':GZ' in command:
            return 1.5
        return 1.0

    def normalize_response(self, command: str, response: str,
                           log_fn: Callable[[str], None] = lambda _: None) -> str:
        """LX200/OnStep response normalization.

        Handles OnStep quirks:
        - ``:MS#`` responses: "0#" = success, "1..." = error.
        - ``:Sr#``/``:Sd#`` responses: empty or position strings treated as success "1#".
        """
        if not response:
            return response
        rc = response.strip()

        if command.startswith(':MS'):
            if rc == "0#":
                pass  # standard success
            elif rc.startswith("1"):
                log_fn(f"WARNING: GOTO refused by telescope: {rc}")
            elif len(rc) > 10:
                log_fn(f"WARNING: Unexpected response for :MS#: '{rc}' (long)")
                if not rc.startswith(("0", "1")):
                    log_fn("WARNING: Unexpected response format, but continuing")
            else:
                if rc == "0" or rc.startswith("0"):
                    response = "0#"
        elif command.startswith(':Sr') or command.startswith(':Sd'):
            if rc == "#" or rc == "":
                log_fn(f"INFO: Empty response '{rc}' for {command}, treated as success")
                response = "1#"
            elif rc in ("1#", "0#"):
                pass
            elif ':' in rc or '*' in rc:
                log_fn(f"INFO: OnStep returned a position '{rc}' instead of an ack, treated as success")
                response = "1#"
            elif len(rc) > 0:
                if rc[0] in ('0', '1'):
                    response = rc[0] + "#"
                else:
                    log_fn(f"INFO: Unexpected response for {command}: '{rc}', treated as success")
                    response = "1#"
            else:
                response = "1#"
                log_fn(f"INFO: Empty response for {command}, treated as success (OnStep)")

        return response


# ===================================================================
# NexStar helpers
# ===================================================================

def _nexstar_angle_to_hex32(deg: float) -> str:
    """Convert degrees (0-360) to NexStar 32-bit hex (8 uppercase chars)."""
    deg = deg % 360
    val = int(deg / 360.0 * 0x100000000) & 0xFFFFFFFF
    return f"{val:08X}"


def _nexstar_hex32_to_angle(h: str) -> float:
    """Convert NexStar 8-char hex string to degrees (0-360)."""
    val = int(h, 16)
    return val / 0x100000000 * 360.0


def _nexstar_angle_to_hex16(deg: float) -> str:
    """Convert degrees (0-360) to NexStar 16-bit hex (4 uppercase chars)."""
    deg = deg % 360
    val = int(deg / 360.0 * 65536) & 0xFFFF
    return f"{val:04X}"


def _nexstar_hex16_to_angle(h: str) -> float:
    """Convert NexStar 4-char hex string to degrees (0-360)."""
    val = int(h, 16)
    return val / 65536.0 * 360.0


def _nexstar_signed_angle(deg: float) -> float:
    """Convert NexStar angle (0-360) to signed (-180..+180).

    NexStar encodes negative declination / altitude as values > 180
    (e.g. -10 deg = 350 in the 0-360 range).
    """
    if deg > 180:
        return deg - 360.0
    return deg


# ===================================================================
# NexStarMountProtocol
# ===================================================================

class NexStarMountProtocol(MountProtocol):
    """Celestron NexStar / Sky-Watcher SynScan hand-controller protocol.

    Implements the NexStar serial/WiFi command set documented in:
        - Celestron NexStar Communication Protocol v1.2+
        - Open-source references: INDI synscandriver, Stellarium plugins

    Compatible with:
        - Celestron NexStar, StarSense, SkyPortal controllers
        - Sky-Watcher SynScan hand controllers and SynScan WiFi adapters
        - Sky-Watcher AZ-GTi, Virtuoso, Dobsonian GoTo mounts
        - Any mount using the NexStar HC command protocol

    The SynScan hand controller uses the **same** NexStar HC protocol
    because Sky-Watcher/Synta is the OEM manufacturer for Celestron.
    The SynScan WiFi adapter exposes this protocol over TCP (port 11882).

    Uses **32-bit precision** (lowercase commands: ``e``, ``z``, ``r``,
    ``b``, ``s``) for all position and goto operations.

    Key protocol differences from LX200:
        - Commands are 1-2 character ASCII, NOT ``:cmd#`` format.
        - Positions are hex-encoded fractions of a full turn.
        - Some commands (time, location) use raw binary bytes.
        - All responses are terminated by ``#``.
    """

    @property
    def name(self) -> str:
        return "NexStar / SynScan"

    @property
    def default_baudrate(self) -> int:
        return 9600

    @property
    def default_tcp_port(self) -> int:
        return 11882

    @property
    def response_terminator(self) -> bytes:
        return b'#'

    # --- Connection test -------------------------------------------

    def test_connection(self, send_fn: SendFn) -> Tuple[bool, str, bool]:
        """Test connection using echo command ``Kx`` and version ``V``.

        Returns (success, model_name, is_onstep=False).
        """
        model = "Unknown"

        # Test 1: Echo command 'K' + char -> should return char + '#'
        try:
            resp = send_fn(b"K\x55", 3.0)  # echo 0x55
            if resp and len(resp) >= 1:
                # Should be chr(0x55) + '#' = 'U#'
                if resp.endswith('#'):
                    model = "NexStar Mount"
                    # Try to get version
                    try:
                        ver_resp = send_fn(b"V", 3.0)
                        if ver_resp and ver_resp.endswith('#'):
                            model = self._parse_version(ver_resp)
                    except Exception:
                        pass

                    # Try to get mount model
                    try:
                        m_resp = send_fn(b"m", 3.0)
                        if m_resp and len(m_resp) >= 2 and m_resp.endswith('#'):
                            model_id = ord(m_resp[0])
                            model = self._identify_model(model_id, model)
                    except Exception:
                        pass

                    return True, model, False
        except Exception:
            pass

        # Test 2: Try version command directly
        try:
            ver_resp = send_fn(b"V", 3.0)
            if ver_resp and ver_resp.endswith('#'):
                ver_str = self._parse_version(ver_resp)
                return True, ver_str, False
        except Exception:
            pass

        return False, model, False

    @staticmethod
    def _parse_version(ver_resp: str) -> str:
        """Parse the ``V`` command response, handling both firmware formats.

        - **SynScan V3.38+/V4.38.06+** (per official SynScan serial protocol
          v3.3): Returns **6 ASCII hex digits** + ``#``, e.g. ``"042507#"``
          for firmware version 04.37.07 (0x25 = 37).
        - **Older Celestron/SynScan HC**: Returns **2 binary bytes** (major,
          minor) + ``#``.

        We distinguish the two by checking if the body is 6 printable ASCII
        hex characters (the v3.3 format) vs. 2 raw bytes (old format).
        """
        body = ver_resp.rstrip('#')
        if not body:
            return "NexStar HC (unknown version)"

        # SynScan v3.3 format: 6 ASCII hex digits, e.g. "042507"
        if len(body) == 6 and all(c in '0123456789ABCDEFabcdef' for c in body):
            try:
                major = int(body[0:2], 16)
                minor = int(body[2:4], 16)
                patch = int(body[4:6], 16)
                return f"SynScan HC v{major}.{minor}.{patch:02d}"
            except ValueError:
                pass

        # Old Celestron format: 2 raw bytes
        if len(body) >= 2:
            major = ord(body[0])
            minor = ord(body[1])
            return f"NexStar HC v{major}.{minor}"

        return f"NexStar HC ({body})"

    @staticmethod
    def _identify_model(model_id: int, fallback: str = "NexStar Mount") -> str:
        """Map the ``m`` command model byte to a human-readable name.

        Covers Celestron and Sky-Watcher Alt-Az mount IDs from the
        official SynScan serial protocol v3.3 document.  IDs for all
        models are kept for correct identification even if the mount
        is not a primary target of this Alt-Az app.
        """
        # Celestron Alt-Az models
        celestron_names = {
            3: "i-Series",
            4: "i-Series SE",
            7: "SLT",
            9: "CPC",
            10: "GT",
            11: "4/5 SE",
            12: "6/8 SE",
            15: "LCM",
            22: "NexStar Evolution",
        }
        # Sky-Watcher / SynScan models (official v3.3 doc + INDI)
        skywatcher_names = {
            160: "AllView GOTO",
            161: "Virtuoso",
            165: "AZ-GTi GOTO",
        }
        # IDs 128-143 = AZ GOTO Series, 144-159 = DOB GOTO Series
        if model_id in celestron_names:
            return f"Celestron {celestron_names[model_id]}"
        if 128 <= model_id <= 143:
            return "Sky-Watcher AZ GOTO Series"
        if 144 <= model_id <= 159:
            return "Sky-Watcher Dob GOTO Series"
        if model_id in skywatcher_names:
            return f"Sky-Watcher {skywatcher_names[model_id]}"
        return f"NexStar/SynScan Model {model_id}"

    # --- Position polling ------------------------------------------

    def poll_position(self, send_fn: SendFn) -> PositionData:
        result = PositionData()

        # --- Slew check: 'L' command -> '0#' (idle) or '1#' (slewing) ---
        # Per SynScan v3.3: response is ASCII "0" or "1" + "#"
        try:
            l_resp = send_fn("L", 3.0)
            if l_resp and l_resp.endswith('#'):
                body = l_resp.rstrip('#')
                result.is_slewing = (body == '1' or body == '\x01')
        except Exception:
            pass

        # --- Get Alt/Az (32-bit precision): 'z' ---
        # Per SynScan v3.3: only upper 24 bits carry data (0.08" precision),
        # but parsing all 32 bits is mathematically equivalent since the
        # lower byte is always 0x00.  Timeout 5s per developer notes.
        time.sleep(0.05)
        try:
            z_resp = send_fn("z", 5.0)
            if z_resp and z_resp.endswith('#') and ',' in z_resp:
                parts = z_resp.rstrip('#').split(',')
                if len(parts) == 2 and len(parts[0]) == 8 and len(parts[1]) == 8:
                    az_raw = _nexstar_hex32_to_angle(parts[0])
                    alt_raw = _nexstar_hex32_to_angle(parts[1])
                    result.az_deg = az_raw
                    result.alt_deg = _nexstar_signed_angle(alt_raw)
                    # Generate LX200-format strings for backward compatibility
                    result.alt_str = _deg_to_lx200_alt(result.alt_deg) + '#'
                    result.az_str = _deg_to_lx200_az(result.az_deg) + '#'
        except Exception:
            pass

        # --- Get RA/Dec (32-bit precision): 'e' ---
        time.sleep(0.05)
        try:
            e_resp = send_fn("e", 5.0)
            if e_resp and e_resp.endswith('#') and ',' in e_resp:
                parts = e_resp.rstrip('#').split(',')
                if len(parts) == 2 and len(parts[0]) == 8 and len(parts[1]) == 8:
                    ra_deg = _nexstar_hex32_to_angle(parts[0])
                    dec_deg = _nexstar_hex32_to_angle(parts[1])
                    result.ra_hours = ra_deg / 15.0  # degrees -> hours
                    result.dec_deg = _nexstar_signed_angle(dec_deg)
                    # LX200-format strings
                    ra_h = int(result.ra_hours)
                    ra_m = int((result.ra_hours - ra_h) * 60)
                    ra_s = ((result.ra_hours - ra_h) * 60 - ra_m) * 60
                    result.ra_str = f"{ra_h:02d}:{ra_m:02d}:{ra_s:04.1f}#"
                    dec_sign = '+' if result.dec_deg >= 0 else '-'
                    dec_abs = abs(result.dec_deg)
                    dec_d = int(dec_abs)
                    dec_m = int((dec_abs - dec_d) * 60)
                    dec_s = int(((dec_abs - dec_d) * 60 - dec_m) * 60)
                    result.dec_str = f"{dec_sign}{dec_d:02d}*{dec_m:02d}'{dec_s:02d}#"
        except Exception:
            pass

        return result

    # --- GoTo ------------------------------------------------------

    def goto_radec(self, ra_str: str, dec_str: str,
                   send_fn: SendFn) -> CommandResult:
        """GoTo using RA/Dec coordinates (32-bit precision).

        Parses RA (HH:MM:SS) and Dec (sDD*MM:SS) from LX200-format
        strings, converts to NexStar hex encoding, sends 'r' command.
        """
        details = []
        try:
            ra_hours = _parse_lx200_dms(ra_str)
            dec_deg = _parse_lx200_dms(dec_str)
            ra_hex = _nexstar_angle_to_hex32(ra_hours * 15.0)  # hours -> degrees
            dec_hex = _nexstar_angle_to_hex32(dec_deg % 360)  # handle negative
            cmd = f"r{ra_hex},{dec_hex}"
            resp = send_fn(cmd, 2.0)
            details.append((cmd, resp))
            ok = resp is not None and resp.endswith('#')
            return CommandResult(success=ok,
                                message="GoTo accepted" if ok else f"GoTo failed: {resp}",
                                details=details)
        except Exception as e:
            return CommandResult(success=False, message=f"GoTo error: {e}",
                                details=details)

    def goto_altaz(self, alt_str: str, az_str: str,
                   send_fn: SendFn) -> CommandResult:
        """GoTo using horizontal coordinates (32-bit precision).

        Preferred for Alt-Az mounts.  Accepts LX200-format strings
        or raw degree values.  Uses lowercase 'b' for 32-bit precision.
        """
        details = []
        try:
            alt_deg = _parse_lx200_dms(alt_str)
            az_deg = _parse_lx200_dms(az_str)
            az_hex = _nexstar_angle_to_hex32(az_deg)
            alt_hex = _nexstar_angle_to_hex32(alt_deg % 360)  # handle negative
            cmd = f"b{az_hex},{alt_hex}"
            resp = send_fn(cmd, 2.0)
            details.append((cmd, resp))
            ok = resp is not None and resp.endswith('#')
            return CommandResult(success=ok,
                                message="Alt/Az GoTo accepted" if ok else f"GoTo failed: {resp}",
                                details=details)
        except Exception as e:
            return CommandResult(success=False, message=f"GoTo error: {e}",
                                details=details)

    # --- Motion control --------------------------------------------

    def slew(self, direction: str, speed: int,
             send_fn: SendFn) -> None:
        """Slew using NexStar motor passthrough commands.

        Uses the 'P' passthrough command to set motor speed per axis.
        Axes: 16=AZM, 17=ALT.  Directions: 36=positive, 37=negative.
        Speed: 1-9 fixed rates.
        """
        # Map direction to (axis, dir_code)
        dir_map = {
            'N': (17, 36),   # ALT positive (up)
            'S': (17, 37),   # ALT negative (down)
            'E': (16, 36),   # AZM positive (CW)
            'W': (16, 37),   # AZM negative (CCW)
        }
        if direction not in dir_map:
            return

        axis, dir_code = dir_map[direction]
        # Map speed 1-4 to NexStar rate 1-9
        rate = min(9, max(1, speed * 2))

        # Build passthrough command: P + chr(2) + axis + dir + rate + 0 + 0 + 0
        cmd_bytes = bytes([ord('P'), 2, axis, dir_code, rate, 0, 0, 0])
        try:
            send_fn(cmd_bytes, 1.0)
        except Exception:
            pass

    def stop(self, send_fn: SendFn) -> None:
        """Stop all motion: set rate=0 on both axes + cancel goto."""
        # Stop AZM axis (rate 0)
        try:
            send_fn(bytes([ord('P'), 2, 16, 36, 0, 0, 0, 0]), 0.5)
        except Exception:
            pass
        # Stop ALT axis (rate 0)
        try:
            send_fn(bytes([ord('P'), 2, 17, 36, 0, 0, 0, 0]), 0.5)
        except Exception:
            pass
        # Cancel goto
        try:
            send_fn("M", 0.5)
        except Exception:
            pass

    # --- Sync ------------------------------------------------------

    def sync_radec(self, ra_str: str, dec_str: str,
                   send_fn: SendFn) -> CommandResult:
        """Sync using NexStar 's' command (32-bit precision).

        NexStar sync: 's' + RA_hex + ',' + Dec_hex -> '#'
        """
        details = []
        try:
            ra_hours = _parse_lx200_dms(ra_str)
            dec_deg = _parse_lx200_dms(dec_str)
            ra_hex = _nexstar_angle_to_hex32(ra_hours * 15.0)
            dec_hex = _nexstar_angle_to_hex32(dec_deg % 360)
            cmd = f"s{ra_hex},{dec_hex}"
            resp = send_fn(cmd, 2.0)
            details.append((cmd, resp))
            ok = resp is not None and resp.endswith('#')
            return CommandResult(success=ok,
                                message="Synced" if ok else "Sync failed",
                                details=details)
        except Exception as e:
            return CommandResult(success=False, message=f"Sync error: {e}",
                                details=details)

    def sync_altaz(self, alt_str: str, az_str: str,
                   send_fn: SendFn) -> CommandResult:
        """NexStar doesn't have a native alt/az sync.

        We convert alt/az to RA/Dec internally (requires observer location
        which we don't have at protocol level), so for now fall back to
        RA/Dec sync if we can, or report unsupported.
        """
        return CommandResult(success=False,
                             message="NexStar: Alt/Az sync not directly supported. Use RA/Dec sync.")

    # --- Park / Home -----------------------------------------------

    def park(self, send_fn: SendFn) -> None:
        """NexStar doesn't have a standard park command in the HC protocol.

        Some mounts support it via passthrough, but it's not universal.
        We'll slew to zenith (Alt=90, Az=0) as a soft park.
        """
        az_hex = _nexstar_angle_to_hex32(0.0)
        alt_hex = _nexstar_angle_to_hex32(90.0)
        try:
            send_fn(f"b{az_hex},{alt_hex}", 2.0)
        except Exception:
            pass

    def home(self, send_fn: SendFn) -> None:
        """NexStar doesn't have a standard home command.

        Some newer mounts support 'h' commands via passthrough.
        For now this is a no-op.
        """
        pass

    # --- Site / Time -----------------------------------------------

    def set_site(self, lat: float, lon: float,
                 utc_offset_west_h: float,
                 send_fn: SendFn) -> CommandResult:
        """Set location using NexStar 'W' command.

        ``W`` + 8 bytes: lat_deg, lat_min, lat_sec, lat_sign,
                          lon_deg, lon_min, lon_sec, lon_sign
        Sign bytes: 0 = positive (N/E), 1 = negative (S/W).

        Longitude convention: NexStar expects 0-180 range with a
        west flag.  Input ``lon`` is standard geographic longitude
        (-180..+180 or 0..360).  If lon > 180, it means west.
        """
        details = []
        try:
            # Latitude: negative = south
            lat_sign = 0 if lat >= 0 else 1
            lat_abs = abs(lat)
            lat_d = int(lat_abs)
            lat_m = int((lat_abs - lat_d) * 60)
            lat_s = int(((lat_abs - lat_d) * 60 - lat_m) * 60 + 0.5)
            if lat_s >= 60:
                lat_s = 0; lat_m += 1
            if lat_m >= 60:
                lat_m = 0; lat_d += 1

            # Longitude: NexStar wants 0-180 with west flag
            # Standard geographic: -180..+180 (negative = west)
            #   or 0..360 (> 180 = west, matching INDI convention)
            is_west = False
            if lon > 180:
                lon = 360.0 - lon
                is_west = True
            elif lon < 0:
                lon = -lon
                is_west = True
            lon_sign = 1 if is_west else 0
            lon_abs = abs(lon)
            lon_d = int(lon_abs)
            lon_m = int((lon_abs - lon_d) * 60)
            lon_s = int(((lon_abs - lon_d) * 60 - lon_m) * 60 + 0.5)
            if lon_s >= 60:
                lon_s = 0; lon_m += 1
            if lon_m >= 60:
                lon_m = 0; lon_d += 1

            cmd_bytes = bytes([ord('W'),
                               lat_d, lat_m, lat_s, lat_sign,
                               lon_d, lon_m, lon_s, lon_sign])
            resp = send_fn(cmd_bytes, 2.0)
            details.append(("W (set location)", resp))

            ok = resp is not None and resp.endswith('#')
            return CommandResult(success=ok,
                                message="Location set" if ok else f"Location failed: {resp}",
                                details=details)
        except Exception as e:
            return CommandResult(success=False, message=f"Set site error: {e}",
                                details=details)

    def set_time(self, dt: datetime,
                 send_fn: SendFn) -> CommandResult:
        """Set time using NexStar 'H' command.

        ``H`` + 8 bytes: hour, minute, second, month, day,
                          year(2-digit), gmt_offset(signed), dst(0/1)
        """
        details = []
        try:
            # Compute GMT offset: NexStar wants hours east of GMT as signed byte
            import datetime as _dt
            now_tz = _dt.datetime.now(_dt.timezone.utc).astimezone()
            utc_off = now_tz.utcoffset()
            gmt_off_h = int(utc_off.total_seconds() / 3600) if utc_off else 0
            # NexStar uses signed byte (two's complement for negative)
            gmt_byte = gmt_off_h & 0xFF
            dst = 0  # No reliable DST detection here

            cmd_bytes = bytes([ord('H'),
                               dt.hour, dt.minute, dt.second,
                               dt.month, dt.day, dt.year % 100,
                               gmt_byte, dst])
            resp = send_fn(cmd_bytes, 2.0)
            details.append(("H (set time)", resp))

            ok = resp is not None and resp.endswith('#')
            return CommandResult(success=ok,
                                message="Time set" if ok else f"Time set failed: {resp}",
                                details=details)
        except Exception as e:
            return CommandResult(success=False, message=f"Set time error: {e}",
                                details=details)

    # --- Tracking --------------------------------------------------

    def send_tracking_command(self, cmd: str,
                               send_fn: SendFn) -> str:
        """NexStar tracking: set mode via 'T' command.

        Tracking modes (byte value after 'T'):
            0 = Off
            1 = Alt/Az tracking

        Also supports 't' (lowercase) to query current tracking mode.
        Passes through LX200-style tracking commands gracefully (ignored).
        """
        if cmd.startswith('T') and len(cmd) == 2:
            return send_fn(cmd, 1.0)
        if cmd == 't':
            return send_fn(cmd, 1.0)
        # Ignore LX200-style tracking commands
        return ""

    def send_variable_rate_altaz(self, alt_rate_arcsec: float,
                                  az_rate_arcsec: float,
                                  send_fn: SendFn) -> None:
        """Send variable tracking rates in Alt/Az using NexStar passthrough.

        Uses the ``P`` passthrough command with **variable rate** format
        (per SynScan serial protocol v3.3, p.3):

            P + chr(3) + chr(axis) + chr(dir) + chr(rateHigh) + chr(rateLow) + chr(0) + chr(0)

        where:
            - axis: 16 = AZM/RA, 17 = ALT/DEC
            - dir: 6 = positive, 7 = negative
            - rate = int(desired_arcsec_per_sec * 4), split into high/low bytes

        This allows the tracking controller to set precise custom tracking
        rates on each axis independently -- essential for real-time plate-solve
        correction on Alt/Az mounts.

        Args:
            alt_rate_arcsec: Altitude tracking rate in arcsec/sec (positive = up).
            az_rate_arcsec: Azimuth tracking rate in arcsec/sec (positive = CW/east).
            send_fn: Transport callback.
        """
        for axis, rate in [(17, alt_rate_arcsec), (16, az_rate_arcsec)]:
            if abs(rate) < 0.001:
                # Rate effectively zero -- stop this axis
                direction = 6
                rate_val = 0
            else:
                direction = 6 if rate >= 0 else 7
                # Per SynScan v3.3 protocol: multiply desired rate by 4
                rate_val = min(65535, int(abs(rate) * 4 + 0.5))

            rate_high = (rate_val >> 8) & 0xFF
            rate_low = rate_val & 0xFF

            cmd_bytes = bytes([
                ord('P'), 3, axis, direction,
                rate_high, rate_low, 0, 0
            ])
            try:
                send_fn(cmd_bytes, 1.0)
            except Exception:
                pass

    @property
    def supports_variable_rate_altaz(self) -> bool:
        """NexStar supports direct variable-rate Alt/Az via passthrough."""
        return True

    # --- Alignment / Pointing state --------------------------------

    def is_aligned(self, send_fn: SendFn) -> Optional[bool]:
        """Query alignment status using ``J`` command.

        Returns ``True`` if mount is aligned, ``False`` if not, ``None``
        on communication error.  Per SynScan v3.3: ``J`` -> chr(0/1) + ``#``.
        """
        try:
            resp = send_fn("J", 3.0)
            if resp and resp.endswith('#'):
                body = resp.rstrip('#')
                if body:
                    return ord(body[0]) == 1 or body == '1'
        except Exception:
            pass
        return None

    # --- Command formatting ----------------------------------------

    def format_outgoing(self, command: str) -> str:
        """NexStar commands don't use delimiters -- return as-is."""
        return command

    def get_command_delay(self, command: str) -> float:
        return 0.05  # NexStar is generally faster than LX200

    def get_read_timeout(self, command: str) -> float:
        """Read timeout per SynScan v3.3 developer notes.

        The official document states: "Software drivers should be prepared
        to wait up to 5.0s (worst case scenario) for a hand control
        response."  We use 5s for position and goto commands, 3s for others.
        """
        if command and command[0] in ('r', 'b', 's', 'R', 'B', 'S'):
            return 5.0  # GoTo / sync: 5s per official doc
        if command and command[0] in ('e', 'z', 'E', 'Z', 'L'):
            return 5.0  # Position queries: hand control retries internally
        return 3.0  # Everything else: 3s (generous)


# ===================================================================
# iOptron helpers
# ===================================================================

def _ioptron_deg_to_counts(deg: float, is_dec: bool = False) -> str:
    """Convert degrees to iOptron command format (integer count string).

    iOptron uses integer arc-second * 100 for positions:
    - RA: 0 to 360*60*60*100 = 129600000 (centered at 0)
    - Dec: 0 to 180*60*60*100 = 64800000 (centered at 324000000 = 90 deg)
    For iOptron firmware v20140101+, positions use the 0x81 command format.
    """
    if is_dec:
        # Dec: -90..+90 -> 0..64800000 (offset by 324000000 = 90*3600*100)
        counts = int((deg + 90.0) * 3600.0 * 100.0 + 0.5)
        return f"{counts:09d}"
    else:
        # RA: 0..24 hours -> 0..360 deg -> counts
        counts = int(deg * 3600.0 * 100.0 + 0.5) % 129600000
        return f"{counts:09d}"


def _ioptron_counts_to_deg(s: str, is_dec: bool = False) -> float:
    """Convert iOptron integer count string back to degrees."""
    counts = int(s)
    if is_dec:
        return counts / (3600.0 * 100.0) - 90.0
    else:
        return counts / (3600.0 * 100.0)


# ===================================================================
# iOptronMountProtocol
# ===================================================================

class iOptronMountProtocol(MountProtocol):
    """iOptron mount protocol implementation (Alt-Az mounts).

    Supports iOptron Alt-Az mounts using the iOptron Command Language (iCL):
        - AZ Mount Pro (Alt-Az GoTo mount for visual + imaging)
        - HAE29 (Alt-Az mount)
        - CubePro (compact Alt-Az)
        - Cube II AA (Alt-Az version)

    The iOptron command protocol is shared across all iOptron mounts
    (both Alt-Az and EQ).  This app uses Alt-Az mode only.

    The iOptron protocol is ASCII-based with command format:
        ``:cmd[data]#``  (colon prefix, hash suffix -- similar to LX200)

    Key differences from LX200:
        - Responses end with ``#`` but do NOT start with ':'
        - Position data uses integer arc-second * 100 encoding
        - Status is queried with ``:GLS#`` (comprehensive) or ``:GAS#``
        - GoTo uses ``:SRA#``/``:SDE#`` + ``:MS1#`` for RA/Dec
        - Model identification via ``:MountInfo#``

    References:
        - iOptron RS-232 Command Language v2.5 / v3.1
        - INDI iOptron driver source code
    """

    @property
    def name(self) -> str:
        return "iOptron"

    @property
    def default_baudrate(self) -> int:
        return 115200  # iOptron mounts typically use 115200

    @property
    def default_tcp_port(self) -> int:
        return 8080  # iOptron WiFi adapter default port

    @property
    def response_terminator(self) -> bytes:
        return b'#'

    # --- Connection test -------------------------------------------

    def test_connection(self, send_fn: SendFn) -> Tuple[bool, str, bool]:
        """Test connection using ``:MountInfo#`` and ``:FW1#``.

        Returns (success, model_name, is_onstep=False).
        """
        model = "Unknown"

        # Try :MountInfo# — returns a 4-digit model code
        try:
            resp = send_fn(b":MountInfo#", 2.0)
            if resp and resp.endswith('#'):
                body = resp.rstrip('#').strip()
                model = self._identify_model(body)
                return True, model, False
        except Exception:
            pass

        # Fallback: try :FW1# (firmware date)
        try:
            resp = send_fn(b":FW1#", 2.0)
            if resp and resp.endswith('#'):
                model = f"iOptron Mount (FW: {resp.rstrip('#')})"
                return True, model, False
        except Exception:
            pass

        # Fallback: try :GLS# (get location + status)
        try:
            resp = send_fn(b":GLS#", 2.0)
            if resp and resp.endswith('#') and len(resp) > 10:
                return True, "iOptron Mount", False
        except Exception:
            pass

        return False, model, False

    @staticmethod
    def _identify_model(info: str) -> str:
        """Map MountInfo response to model name.

        All iOptron model IDs are included for correct identification.
        Alt-Az models (5xxx series) are the primary targets of this app.
        """
        models = {
            # Alt-Az models (primary targets)
            "5010": "Cube II AA",
            "5035": "AZ Mount Pro",
            "5045": "HAE29",
            # Other iOptron models (for identification only)
            "0010": "Cube II",
            "0011": "SmartEQ Pro+",
            "0025": "CEM25",
            "0026": "CEM26",
            "0028": "GEM28",
            "0030": "iEQ30 Pro",
            "0040": "CEM40",
            "0043": "GEM45",
            "0045": "iEQ45 Pro",
            "0060": "CEM60",
            "0070": "CEM70",
            "0120": "CEM120",
        }
        for code, name in models.items():
            if info.startswith(code):
                return f"iOptron {name}"
        if info:
            return f"iOptron Mount ({info})"
        return "iOptron Mount"

    # --- Position polling ------------------------------------------

    def poll_position(self, send_fn: SendFn) -> PositionData:
        """Poll position using ``:GEP#`` (Get Extended Position).

        Response: ``sDDDDDDDDDsDDDDDDDDDp#``
        - 9 digits for RA (arc-seconds * 100)
        - 9 digits for Dec (arc-seconds * 100, offset by +90 deg)
        - 1 digit status

        If ``:GEP#`` fails, fall back to ``:GEC#`` or ``:GLS#``.
        """
        result = PositionData()

        # Try :GEP# first (most comprehensive)
        try:
            resp = send_fn(":GEP#", 1.5)
            if resp and resp.endswith('#') and len(resp) >= 20:
                body = resp.rstrip('#')
                # Parse: first 9 chars = RA counts, next sign+9 = Dec counts
                # Format varies by firmware; try the common format
                ra_str_raw = body[0:9]
                dec_str_raw = body[9:18]
                try:
                    ra_deg = _ioptron_counts_to_deg(ra_str_raw, is_dec=False)
                    dec_deg = _ioptron_counts_to_deg(dec_str_raw, is_dec=True)
                    result.ra_hours = ra_deg / 15.0  # degrees -> hours
                    result.dec_deg = dec_deg
                    # Generate LX200-format strings
                    ra_h = int(result.ra_hours)
                    ra_m = int((result.ra_hours - ra_h) * 60)
                    ra_s = ((result.ra_hours - ra_h) * 60 - ra_m) * 60
                    result.ra_str = f"{ra_h:02d}:{ra_m:02d}:{ra_s:04.1f}#"
                    dec_sign = '+' if dec_deg >= 0 else '-'
                    dec_abs = abs(dec_deg)
                    dec_d = int(dec_abs)
                    dec_m = int((dec_abs - dec_d) * 60)
                    dec_s = int(((dec_abs - dec_d) * 60 - dec_m) * 60)
                    result.dec_str = f"{dec_sign}{dec_d:02d}*{dec_m:02d}:{dec_s:02d}#"
                except (ValueError, IndexError):
                    pass
        except Exception:
            pass

        # Alt/Az: try :GAC# (Get Alt/Az Counts)
        time.sleep(0.05)
        try:
            resp = send_fn(":GAC#", 1.5)
            if resp and resp.endswith('#') and len(resp) >= 18:
                body = resp.rstrip('#')
                alt_str_raw = body[0:9]
                az_str_raw = body[9:18]
                try:
                    result.alt_deg = _ioptron_counts_to_deg(alt_str_raw, is_dec=True)  # Same encoding as Dec
                    result.az_deg = _ioptron_counts_to_deg(az_str_raw, is_dec=False)
                    result.alt_str = _deg_to_lx200_alt(result.alt_deg) + '#'
                    result.az_str = _deg_to_lx200_az(result.az_deg) + '#'
                except (ValueError, IndexError):
                    pass
        except Exception:
            pass

        # Slew status: :GAS# (Get All Status)
        time.sleep(0.05)
        try:
            resp = send_fn(":GAS#", 1.0)
            if resp and resp.endswith('#') and len(resp) >= 2:
                body = resp.rstrip('#')
                # Status byte: 0=stopped, 1=tracking, 2=slewing, 3=guiding, ...
                if body and body[-1] in ('2', '5', '6'):
                    result.is_slewing = True
        except Exception:
            pass

        return result

    # --- GoTo ------------------------------------------------------

    def goto_radec(self, ra_str: str, dec_str: str,
                   send_fn: SendFn) -> CommandResult:
        details = []
        try:
            ra_hours = _parse_lx200_dms(ra_str)
            dec_deg = _parse_lx200_dms(dec_str)
            ra_counts = _ioptron_deg_to_counts(ra_hours * 15.0, is_dec=False)
            dec_counts = _ioptron_deg_to_counts(dec_deg, is_dec=True)

            # Set RA target
            cmd = f":SRA{ra_counts}#"
            resp = send_fn(cmd)
            details.append((cmd, resp))

            # Set Dec target
            cmd = f":SDE{dec_counts}#"
            resp = send_fn(cmd)
            details.append((cmd, resp))

            # Initiate slew: :MS1# (GoTo target)
            cmd = ":MS1#"
            resp = send_fn(cmd)
            details.append((cmd, resp))

            ok = resp is not None and ('1' in resp or resp.endswith('#'))
            return CommandResult(success=ok,
                                message="GoTo accepted" if ok else f"GoTo failed: {resp}",
                                details=details)
        except Exception as e:
            return CommandResult(success=False, message=f"GoTo error: {e}",
                                details=details)

    def goto_altaz(self, alt_str: str, az_str: str,
                   send_fn: SendFn) -> CommandResult:
        """GoTo by Alt/Az — iOptron supports this on AZ mounts via ``:MSS#``."""
        details = []
        try:
            alt_deg = _parse_lx200_dms(alt_str)
            az_deg = _parse_lx200_dms(az_str)
            alt_counts = _ioptron_deg_to_counts(alt_deg, is_dec=True)
            az_counts = _ioptron_deg_to_counts(az_deg, is_dec=False)

            cmd = f":SAL{alt_counts}#"
            resp = send_fn(cmd)
            details.append((cmd, resp))

            cmd = f":SAZ{az_counts}#"
            resp = send_fn(cmd)
            details.append((cmd, resp))

            cmd = ":MSS#"
            resp = send_fn(cmd)
            details.append((cmd, resp))

            ok = resp is not None and ('1' in resp or resp.endswith('#'))
            return CommandResult(success=ok,
                                message="Alt/Az GoTo accepted" if ok else f"GoTo failed: {resp}",
                                details=details)
        except Exception as e:
            return CommandResult(success=False, message=f"GoTo error: {e}",
                                details=details)

    # --- Motion control --------------------------------------------

    def slew(self, direction: str, speed: int,
             send_fn: SendFn) -> None:
        # iOptron speed: 1-9 (1=1x, 2=2x, ..., 9=max)
        rate = min(9, max(1, speed * 2))
        send_fn(f":SR{rate}#")
        dir_cmds = {"N": ":mn#", "S": ":ms#", "E": ":me#", "W": ":mw#"}
        if direction in dir_cmds:
            send_fn(dir_cmds[direction])

    def stop(self, send_fn: SendFn) -> None:
        send_fn(":Q#")

    # --- Sync ------------------------------------------------------

    def sync_radec(self, ra_str: str, dec_str: str,
                   send_fn: SendFn) -> CommandResult:
        details = []
        try:
            ra_hours = _parse_lx200_dms(ra_str)
            dec_deg = _parse_lx200_dms(dec_str)
            ra_counts = _ioptron_deg_to_counts(ra_hours * 15.0, is_dec=False)
            dec_counts = _ioptron_deg_to_counts(dec_deg, is_dec=True)

            cmd = f":SRA{ra_counts}#"
            resp = send_fn(cmd)
            details.append((cmd, resp))

            cmd = f":SDE{dec_counts}#"
            resp = send_fn(cmd)
            details.append((cmd, resp))

            cmd = ":CM#"
            resp = send_fn(cmd)
            details.append((cmd, resp))

            ok = resp is not None and resp.endswith('#')
            return CommandResult(success=ok,
                                message="Synced" if ok else "Sync failed",
                                details=details)
        except Exception as e:
            return CommandResult(success=False, message=f"Sync error: {e}",
                                details=details)

    def sync_altaz(self, alt_str: str, az_str: str,
                   send_fn: SendFn) -> CommandResult:
        return CommandResult(success=False,
                             message="iOptron: Alt/Az sync not directly supported. Use RA/Dec sync.")

    # --- Park / Home -----------------------------------------------

    def park(self, send_fn: SendFn) -> None:
        send_fn(":MP1#")  # Park at current park position

    def home(self, send_fn: SendFn) -> None:
        send_fn(":MH#")  # Go to home position

    # --- Site / Time -----------------------------------------------

    def set_site(self, lat: float, lon: float,
                 utc_offset_west_h: float,
                 send_fn: SendFn) -> CommandResult:
        details = []
        try:
            # iOptron latitude: sDDMMSS (sign + 6 digits)
            lat_sign = '+' if lat >= 0 else '-'
            lat_abs = abs(lat)
            lat_d = int(lat_abs)
            lat_m = int((lat_abs - lat_d) * 60)
            lat_s = int(((lat_abs - lat_d) * 60 - lat_m) * 60 + 0.5)
            cmd = f":SLO{lat_sign}{lat_d:02d}{lat_m:02d}{lat_s:02d}#"
            resp = send_fn(cmd)
            details.append((cmd, resp))

            # iOptron longitude: sDDDMMSS
            lon_sign = '+' if lon >= 0 else '-'
            lon_abs = abs(lon)
            lon_d = int(lon_abs)
            lon_m = int((lon_abs - lon_d) * 60)
            lon_s = int(((lon_abs - lon_d) * 60 - lon_m) * 60 + 0.5)
            cmd = f":SLA{lon_sign}{lon_d:03d}{lon_m:02d}{lon_s:02d}#"
            resp = send_fn(cmd)
            details.append((cmd, resp))

            # UTC offset: :SG sHHMM# (iOptron uses minutes offset)
            off_h = int(utc_offset_west_h)
            off_m = int(abs(utc_offset_west_h - off_h) * 60)
            off_sign = '+' if utc_offset_west_h >= 0 else '-'
            cmd = f":SG{off_sign}{abs(off_h):02d}{off_m:02d}#"
            resp = send_fn(cmd)
            details.append((cmd, resp))

            return CommandResult(success=True, message="Site sent", details=details)
        except Exception as e:
            return CommandResult(success=False, message=f"Set site error: {e}",
                                details=details)

    def set_time(self, dt: datetime,
                 send_fn: SendFn) -> CommandResult:
        details = []
        try:
            # iOptron time: :SL HH:MM:SS#
            cmd = f":SL{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}#"
            resp = send_fn(cmd)
            details.append((cmd, resp))

            # iOptron date: :SC MM/DD/YY#
            cmd = f":SC{dt.month:02d}/{dt.day:02d}/{dt.year % 100:02d}#"
            resp = send_fn(cmd)
            details.append((cmd, resp))

            return CommandResult(success=True, message="Time sent", details=details)
        except Exception as e:
            return CommandResult(success=False, message=f"Set time error: {e}",
                                details=details)

    # --- Command formatting ----------------------------------------

    def format_outgoing(self, command: str) -> str:
        """iOptron uses ``:cmd#`` format, same as LX200."""
        if not command.startswith(':'):
            command = ':' + command
        if not command.endswith('#'):
            command = command + '#'
        return command

    def get_command_delay(self, command: str) -> float:
        if ':MS' in command:
            return 0.2
        return 0.1

    def get_read_timeout(self, command: str) -> float:
        if ':MS' in command:
            return 2.0
        if ':GEP' in command or ':GAC' in command or ':GLS' in command:
            return 1.5
        return 1.0

    def normalize_response(self, command: str, response: str,
                           log_fn: Callable[[str], None] = lambda _: None) -> str:
        """iOptron responses are generally clean — minimal normalization."""
        return response


# ===================================================================
# MeadeAudioStarMountProtocol
# ===================================================================

class MeadeAudioStarMountProtocol(MountProtocol):
    """Meade AudioStar / AutoStar protocol implementation.

    Extends the standard LX200 command set with Meade-specific additions.
    Compatible with:
        - Meade ETX-60/70/80/90/105/125 with AutoStar/AudioStar
        - Meade LX90, LX200GPS, LX200ACF, LX600
        - Meade LX85, LX65
        - Any Meade mount with AutoStar II or AudioStar hand controllers

    The AudioStar/AutoStar protocol is nearly identical to standard LX200 with
    these key extensions:
        - ``:GVD#`` / ``:GVN#`` / ``:GVP#`` / ``:GVT#`` — version queries
        - ``:Me#``, ``:Mn#``, ``:Ms#``, ``:Mw#`` — directional slew (LX200-compat)
        - Alignment commands: ``:AL#`` (land), ``:AP#`` (polar), ``:AA#`` (altaz)
        - ``:U#`` — toggles between high-precision (HH:MM:SS) and low-precision
          (HH:MM.T) coordinate formats

    Most commands are standard LX200 and reuse the same wire format. The main
    differences are in model detection and a few Meade-specific extensions.
    """

    @property
    def name(self) -> str:
        return "Meade AudioStar"

    @property
    def default_baudrate(self) -> int:
        return 9600

    @property
    def default_tcp_port(self) -> int:
        return 4030  # Meade WiFi adapter typically uses 4030

    @property
    def response_terminator(self) -> bytes:
        return b'#'

    # --- Connection test -------------------------------------------

    def test_connection(self, send_fn: SendFn) -> Tuple[bool, str, bool]:
        """Test connection using standard LX200 product name query.

        Meade mounts respond to ``:GVP#`` with "ETX-125", "LX200GPS", etc.
        Also tries ``:GVN#`` for firmware version.
        """
        model = "Unknown"

        # Send :U# first to ensure high-precision mode
        try:
            send_fn(b":U#", 0.5)
        except Exception:
            pass

        for cmd_bytes, cmd_name in [
            (b":GVP#", "product name"),
            (b":GVN#", "firmware version"),
        ]:
            try:
                resp = send_fn(cmd_bytes, 2.0)
                if resp and resp.endswith('#'):
                    body = resp.rstrip('#').strip()
                    if cmd_bytes == b":GVP#" and body:
                        model = f"Meade {body}"
                    elif body:
                        model = f"Meade Mount (FW: {body})"
                    return True, model, False
            except Exception:
                continue

        # Fallback: try coordinate queries
        for cmd_bytes in [b":GR#", b":GD#"]:
            try:
                resp = send_fn(cmd_bytes, 2.0)
                if resp and resp.endswith('#'):
                    return True, "Meade Mount (responds to LX200)", False
            except Exception:
                continue

        return False, model, False

    # --- Position polling ------------------------------------------

    def poll_position(self, send_fn: SendFn) -> PositionData:
        """Poll position using standard LX200 commands.

        Meade mounts use the same :GA#, :GZ#, :GR#, :GD# as LX200.
        """
        result = PositionData()

        # Slew check: :D# (display status, '|' chars = slewing)
        try:
            d_resp = send_fn(":D#", 0.5)
            result.is_slewing = bool(d_resp and '|' in d_resp)
        except Exception:
            pass

        # Alt/Az
        time.sleep(0.1)
        try:
            alt_resp = send_fn(":GA#", 0.8)
            if alt_resp and alt_resp.endswith('#'):
                result.alt_str = alt_resp
                result.alt_deg = _parse_lx200_dms(alt_resp)
        except Exception:
            pass

        time.sleep(0.1)
        try:
            az_resp = send_fn(":GZ#", 0.8)
            if az_resp and az_resp.endswith('#'):
                result.az_str = az_resp
                result.az_deg = _parse_lx200_dms(az_resp)
        except Exception:
            pass

        # RA/Dec
        time.sleep(0.1)
        try:
            ra_resp = send_fn(":GR#", 0.8)
            if ra_resp and ra_resp.endswith('#'):
                result.ra_str = ra_resp
                result.ra_hours = _parse_lx200_dms(ra_resp)
        except Exception:
            pass

        time.sleep(0.1)
        try:
            dec_resp = send_fn(":GD#", 0.8)
            if dec_resp and dec_resp.endswith('#'):
                result.dec_str = dec_resp
                result.dec_deg = _parse_lx200_dms(dec_resp)
        except Exception:
            pass

        return result

    # --- GoTo ------------------------------------------------------

    def goto_radec(self, ra_str: str, dec_str: str,
                   send_fn: SendFn) -> CommandResult:
        details = []
        ra_resp = send_fn(f":Sr{ra_str}#")
        details.append((f":Sr{ra_str}#", ra_resp))
        dec_resp = send_fn(f":Sd{dec_str}#")
        details.append((f":Sd{dec_str}#", dec_resp))
        ms_resp = send_fn(":MS#")
        details.append((":MS#", ms_resp))
        ms_clean = (ms_resp or "").strip().rstrip('#')
        ok = ms_clean in ("0", "")
        msg = "GoTo accepted" if ok else f"GoTo refused: {ms_resp}"
        return CommandResult(success=ok, message=msg, details=details)

    def goto_altaz(self, alt_str: str, az_str: str,
                   send_fn: SendFn) -> CommandResult:
        details = []
        alt_resp = send_fn(f":Sa{alt_str}#")
        details.append((f":Sa{alt_str}#", alt_resp))
        az_resp = send_fn(f":Sz{az_str}#")
        details.append((f":Sz{az_str}#", az_resp))
        ma_resp = send_fn(":MA#")
        details.append((":MA#", ma_resp))
        ma_clean = (ma_resp or "").strip().rstrip('#')
        ok = ma_clean in ("0", "") or "*" in ma_clean or ":" in ma_clean
        msg = "Alt/Az GoTo accepted" if ok else f"Alt/Az GoTo refused: {ma_resp}"
        return CommandResult(success=ok, message=msg, details=details)

    # --- Motion control --------------------------------------------

    def slew(self, direction: str, speed: int,
             send_fn: SendFn) -> None:
        speed_cmds = {1: ":RG#", 2: ":RC#", 3: ":RM#", 4: ":RS#"}
        dir_cmds = {"N": ":Mn#", "S": ":Ms#", "E": ":Me#", "W": ":Mw#"}
        send_fn(speed_cmds.get(speed, ":RS#"))
        if direction in dir_cmds:
            send_fn(dir_cmds[direction])

    def stop(self, send_fn: SendFn) -> None:
        send_fn(":Q#")

    # --- Sync ------------------------------------------------------

    def sync_radec(self, ra_str: str, dec_str: str,
                   send_fn: SendFn) -> CommandResult:
        details = []
        details.append((f":Sr{ra_str}#", send_fn(f":Sr{ra_str}#")))
        details.append((f":Sd{dec_str}#", send_fn(f":Sd{dec_str}#")))
        cm_resp = send_fn(":CM#")
        details.append((":CM#", cm_resp))
        ok = bool(cm_resp)
        return CommandResult(success=ok, message="Synced" if ok else "Sync failed",
                             details=details)

    def sync_altaz(self, alt_str: str, az_str: str,
                   send_fn: SendFn) -> CommandResult:
        details = []
        details.append((f":Sa{alt_str}#", send_fn(f":Sa{alt_str}#")))
        details.append((f":Sz{az_str}#", send_fn(f":Sz{az_str}#")))
        cm_resp = send_fn(":CM#")
        details.append((":CM#", cm_resp))
        ok = bool(cm_resp)
        return CommandResult(success=ok, message="Synced" if ok else "Sync failed",
                             details=details)

    # --- Park / Home -----------------------------------------------

    def park(self, send_fn: SendFn) -> None:
        send_fn(":hP#")

    def home(self, send_fn: SendFn) -> None:
        send_fn(":hF#")

    # --- Site / Time -----------------------------------------------

    def set_site(self, lat: float, lon: float,
                 utc_offset_west_h: float,
                 send_fn: SendFn) -> CommandResult:
        details = []
        # Same as standard LX200
        lat_sign = '+' if lat >= 0 else '-'
        lat_abs = abs(lat)
        lat_deg = int(lat_abs)
        lat_min = int((lat_abs - lat_deg) * 60 + 0.5)
        if lat_min >= 60:
            lat_deg += 1; lat_min = 0
        cmd = f":St{lat_sign}{lat_deg:02d}*{lat_min:02d}#"
        resp = send_fn(cmd)
        details.append((cmd, resp))

        # Meade uses standard geographic longitude (east-positive)
        # Different from OnStep which uses west-positive
        lon_abs = abs(lon)
        lon_sign = '+' if lon >= 0 else '-'
        lon_deg = int(lon_abs)
        lon_min = int((lon_abs - lon_deg) * 60 + 0.5)
        if lon_min >= 60:
            lon_deg += 1; lon_min = 0
        cmd = f":Sg{lon_sign}{lon_deg:03d}*{lon_min:02d}#"
        resp = send_fn(cmd)
        details.append((cmd, resp))

        sign = '+' if utc_offset_west_h >= 0 else '-'
        cmd = f":SG{sign}{abs(utc_offset_west_h):04.1f}#"
        resp = send_fn(cmd)
        details.append((cmd, resp))

        return CommandResult(success=True, message="Site sent", details=details)

    def set_time(self, dt: datetime,
                 send_fn: SendFn) -> CommandResult:
        details = []
        cmd = f":SL{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}#"
        resp = send_fn(cmd)
        details.append((cmd, resp))
        cmd = f":SC{dt.month:02d}/{dt.day:02d}/{dt.year % 100:02d}#"
        resp = send_fn(cmd)
        details.append((cmd, resp))
        return CommandResult(success=True, message="Time sent", details=details)

    # --- Command formatting ----------------------------------------

    def format_outgoing(self, command: str) -> str:
        """Standard LX200 formatting."""
        if not command.startswith(':'):
            command = ':' + command
        if not command.endswith('#'):
            command = command + '#'
        return command

    def get_command_delay(self, command: str) -> float:
        if ':MS' in command:
            return 0.3
        if ':Sd' in command or ':Sr' in command:
            return 0.25
        return 0.15

    def get_read_timeout(self, command: str) -> float:
        if ':MS' in command or ':Sd' in command or ':Sr' in command:
            return 2.0
        if ':GR' in command or ':GD' in command or ':GA' in command or ':GZ' in command:
            return 1.5
        return 1.0

    def normalize_response(self, command: str, response: str,
                           log_fn: Callable[[str], None] = lambda _: None) -> str:
        """Meade response normalization — similar to LX200/OnStep."""
        if not response:
            return response
        rc = response.strip()

        if command.startswith(':MS'):
            if rc == "0#":
                pass
            elif rc.startswith("1"):
                log_fn(f"WARNING: GOTO refused: {rc}")
            elif rc == "0" or rc.startswith("0"):
                response = "0#"
        elif command.startswith(':Sr') or command.startswith(':Sd'):
            if rc in ("#", ""):
                response = "1#"
            elif rc in ("1#", "0#"):
                pass
            elif ':' in rc or '*' in rc:
                response = "1#"
            else:
                response = "1#"

        return response


# ===================================================================
# ASCOMAlpacaMountProtocol
# ===================================================================

class ASCOMAlpacaMountProtocol(MountProtocol):
    """ASCOM Alpaca REST API protocol for telescope mounts.

    ASCOM Alpaca is a platform-independent REST API that provides access to
    ASCOM-compatible astronomical devices over HTTP. Any mount with an ASCOM
    driver can be accessed via an Alpaca server running on the same network.

    Architecture:
        TrackWise app  <--HTTP-->  Alpaca Server (PC)  <--ASCOM-->  Mount driver

    The Alpaca API uses HTTP GET/PUT with JSON payloads on endpoints like:
        ``/api/v1/telescope/{device_number}/{method}``

    Key Alpaca telescope methods:
        - GET ``/slewtocoordinatesasync``  (GoTo RA/Dec)
        - GET ``/rightascension``, ``/declination`` (position query)
        - GET ``/altitude``, ``/azimuth`` (alt/az position)
        - PUT ``/tracking`` (tracking on/off)
        - PUT ``/synctocoordinates`` (sync)
        - PUT ``/abortslew`` (stop)

    This protocol uses the ``send_fn`` callback differently: instead of
    serial/TCP raw byte commands, it constructs Alpaca REST commands as
    specially-formatted strings that the bridge transmits and the server
    interprets. The bridge transport (TCP socket) connects to the Alpaca
    server's HTTP port.

    For this implementation, we encode Alpaca requests as pseudo-commands
    through the existing ``send_fn`` mechanism using a simple text protocol:
        ``ALPACA|METHOD|PATH|BODY``
    The bridge's _send_tcp_command detects the ALPACA| prefix and performs
    the HTTP request instead of raw socket I/O.

    Note: In practice, ASCOM Alpaca uses HTTP, not raw TCP. This protocol
    implementation works by directly making HTTP requests from within the
    protocol methods (using urllib), bypassing the serial/TCP transport.
    The ``send_fn`` is still available for status logging.

    Default port: 11111 (ASCOM Alpaca standard discovery port).
    """

    def __init__(self):
        self._base_url = ""  # Set during connection: "http://ip:port"
        self._device_number = 0
        self._client_id = 1
        self._client_transaction_id = 0

    @property
    def name(self) -> str:
        return "ASCOM Alpaca"

    @property
    def default_baudrate(self) -> int:
        return 0  # Not applicable for HTTP

    @property
    def default_tcp_port(self) -> int:
        return 11111  # ASCOM Alpaca standard port

    @property
    def response_terminator(self) -> bytes:
        return b'#'  # Not really used — HTTP responses are JSON

    def _alpaca_url(self, method: str) -> str:
        return f"{self._base_url}/api/v1/telescope/{self._device_number}/{method}"

    def _alpaca_get(self, method: str) -> Optional[dict]:
        """Perform an Alpaca GET request."""
        import urllib.request
        import json as _json
        self._client_transaction_id += 1
        url = self._alpaca_url(method)
        url += f"?ClientID={self._client_id}&ClientTransactionID={self._client_transaction_id}"
        try:
            req = urllib.request.Request(url, method='GET')
            req.add_header('Accept', 'application/json')
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = _json.loads(resp.read().decode())
                if data.get("ErrorNumber", 0) == 0:
                    return data
                _logger.debug("Alpaca GET %s error: %s", method, data.get("ErrorMessage"))
        except Exception as e:
            _logger.debug("Alpaca GET %s failed: %s", method, e)
        return None

    def _alpaca_put(self, method: str, params: Optional[Dict] = None) -> Optional[dict]:
        """Perform an Alpaca PUT request."""
        import urllib.request
        import urllib.parse
        import json as _json
        self._client_transaction_id += 1
        url = self._alpaca_url(method)
        form_data = {
            "ClientID": self._client_id,
            "ClientTransactionID": self._client_transaction_id,
        }
        if params:
            form_data.update(params)
        body = urllib.parse.urlencode(form_data).encode('utf-8')
        try:
            req = urllib.request.Request(url, data=body, method='PUT')
            req.add_header('Content-Type', 'application/x-www-form-urlencoded')
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = _json.loads(resp.read().decode())
                if data.get("ErrorNumber", 0) == 0:
                    return data
                _logger.debug("Alpaca PUT %s error: %s", method, data.get("ErrorMessage"))
        except Exception as e:
            _logger.debug("Alpaca PUT %s failed: %s", method, e)
        return None

    # --- Connection test -------------------------------------------

    def test_connection(self, send_fn: SendFn) -> Tuple[bool, str, bool]:
        """Test Alpaca connection by querying the telescope name.

        The ``send_fn`` is not directly used for Alpaca HTTP calls.
        Instead, we need the base URL which is derived from the bridge's
        TCP connection target (ip:port). We extract it from the first
        call context or use a fallback.
        """
        model = "Unknown"

        # Try to get the telescope description
        if self._base_url:
            data = self._alpaca_get("name")
            if data and "Value" in data:
                model = f"ASCOM: {data['Value']}"
                return True, model, False

            data = self._alpaca_get("description")
            if data and "Value" in data:
                model = f"ASCOM: {data['Value']}"
                return True, model, False

            # Just try connected status
            data = self._alpaca_get("connected")
            if data is not None:
                return True, "ASCOM Alpaca Telescope", False

        return False, model, False

    def set_base_url(self, ip: str, port: int):
        """Set the Alpaca server base URL. Called by the bridge after connect."""
        self._base_url = f"http://{ip}:{port}"

    # --- Position polling ------------------------------------------

    def poll_position(self, send_fn: SendFn) -> PositionData:
        result = PositionData()

        if not self._base_url:
            return result

        # RA/Dec
        try:
            ra_data = self._alpaca_get("rightascension")
            dec_data = self._alpaca_get("declination")
            if ra_data and "Value" in ra_data:
                result.ra_hours = float(ra_data["Value"])
                ra_h = int(result.ra_hours)
                ra_m = int((result.ra_hours - ra_h) * 60)
                ra_s = ((result.ra_hours - ra_h) * 60 - ra_m) * 60
                result.ra_str = f"{ra_h:02d}:{ra_m:02d}:{ra_s:04.1f}#"
            if dec_data and "Value" in dec_data:
                result.dec_deg = float(dec_data["Value"])
                dec_sign = '+' if result.dec_deg >= 0 else '-'
                dec_abs = abs(result.dec_deg)
                dec_d = int(dec_abs)
                dec_m = int((dec_abs - dec_d) * 60)
                dec_s = int(((dec_abs - dec_d) * 60 - dec_m) * 60)
                result.dec_str = f"{dec_sign}{dec_d:02d}*{dec_m:02d}:{dec_s:02d}#"
        except Exception:
            pass

        # Alt/Az
        try:
            alt_data = self._alpaca_get("altitude")
            az_data = self._alpaca_get("azimuth")
            if alt_data and "Value" in alt_data:
                result.alt_deg = float(alt_data["Value"])
                result.alt_str = _deg_to_lx200_alt(result.alt_deg) + '#'
            if az_data and "Value" in az_data:
                result.az_deg = float(az_data["Value"])
                result.az_str = _deg_to_lx200_az(result.az_deg) + '#'
        except Exception:
            pass

        # Slewing status
        try:
            slew_data = self._alpaca_get("slewing")
            if slew_data and "Value" in slew_data:
                result.is_slewing = bool(slew_data["Value"])
        except Exception:
            pass

        return result

    # --- GoTo ------------------------------------------------------

    def goto_radec(self, ra_str: str, dec_str: str,
                   send_fn: SendFn) -> CommandResult:
        details = []
        try:
            ra_hours = _parse_lx200_dms(ra_str)
            dec_deg = _parse_lx200_dms(dec_str)
            data = self._alpaca_put("slewtocoordinatesasync", {
                "RightAscension": ra_hours,
                "Declination": dec_deg,
            })
            details.append(("SlewToCoordinatesAsync", str(data)))
            ok = data is not None
            return CommandResult(success=ok,
                                message="GoTo accepted" if ok else "GoTo failed",
                                details=details)
        except Exception as e:
            return CommandResult(success=False, message=f"GoTo error: {e}",
                                details=details)

    def goto_altaz(self, alt_str: str, az_str: str,
                   send_fn: SendFn) -> CommandResult:
        details = []
        try:
            alt_deg = _parse_lx200_dms(alt_str)
            az_deg = _parse_lx200_dms(az_str)
            data = self._alpaca_put("slewtoaltazasync", {
                "Altitude": alt_deg,
                "Azimuth": az_deg,
            })
            details.append(("SlewToAltAzAsync", str(data)))
            ok = data is not None
            return CommandResult(success=ok,
                                message="Alt/Az GoTo accepted" if ok else "GoTo failed",
                                details=details)
        except Exception as e:
            return CommandResult(success=False, message=f"GoTo error: {e}",
                                details=details)

    # --- Motion control --------------------------------------------

    def slew(self, direction: str, speed: int,
             send_fn: SendFn) -> None:
        # Alpaca uses MoveAxis(axis, rate) — rate in deg/sec
        # Axis: 0 = primary (RA/Az), 1 = secondary (Dec/Alt)
        rate_degsec = speed * 1.0  # 1-4 deg/sec
        axis_map = {
            'N': (1, rate_degsec),    # Dec/Alt positive
            'S': (1, -rate_degsec),   # Dec/Alt negative
            'E': (0, rate_degsec),    # RA/Az positive
            'W': (0, -rate_degsec),   # RA/Az negative
        }
        if direction in axis_map:
            axis, rate = axis_map[direction]
            self._alpaca_put("moveaxis", {"Axis": axis, "Rate": rate})

    def stop(self, send_fn: SendFn) -> None:
        self._alpaca_put("abortslew")
        # Also stop any MoveAxis motion
        self._alpaca_put("moveaxis", {"Axis": 0, "Rate": 0})
        self._alpaca_put("moveaxis", {"Axis": 1, "Rate": 0})

    # --- Sync ------------------------------------------------------

    def sync_radec(self, ra_str: str, dec_str: str,
                   send_fn: SendFn) -> CommandResult:
        details = []
        try:
            ra_hours = _parse_lx200_dms(ra_str)
            dec_deg = _parse_lx200_dms(dec_str)
            data = self._alpaca_put("synctocoordinates", {
                "RightAscension": ra_hours,
                "Declination": dec_deg,
            })
            details.append(("SyncToCoordinates", str(data)))
            ok = data is not None
            return CommandResult(success=ok,
                                message="Synced" if ok else "Sync failed",
                                details=details)
        except Exception as e:
            return CommandResult(success=False, message=f"Sync error: {e}",
                                details=details)

    def sync_altaz(self, alt_str: str, az_str: str,
                   send_fn: SendFn) -> CommandResult:
        details = []
        try:
            alt_deg = _parse_lx200_dms(alt_str)
            az_deg = _parse_lx200_dms(az_str)
            data = self._alpaca_put("synctoaltaz", {
                "Altitude": alt_deg,
                "Azimuth": az_deg,
            })
            details.append(("SyncToAltAz", str(data)))
            ok = data is not None
            return CommandResult(success=ok,
                                message="Synced" if ok else "Sync failed",
                                details=details)
        except Exception as e:
            return CommandResult(success=False, message=f"Sync error: {e}",
                                details=details)

    # --- Park / Home -----------------------------------------------

    def park(self, send_fn: SendFn) -> None:
        self._alpaca_put("park")

    def home(self, send_fn: SendFn) -> None:
        self._alpaca_put("findhome")

    # --- Site / Time -----------------------------------------------

    def set_site(self, lat: float, lon: float,
                 utc_offset_west_h: float,
                 send_fn: SendFn) -> CommandResult:
        details = []
        try:
            data = self._alpaca_put("siteelevation", {"SiteElevation": 0})
            details.append(("SiteElevation", str(data)))
            data = self._alpaca_put("sitelatitude", {"SiteLatitude": lat})
            details.append(("SiteLatitude", str(data)))
            data = self._alpaca_put("sitelongitude", {"SiteLongitude": lon})
            details.append(("SiteLongitude", str(data)))
            return CommandResult(success=True, message="Site set via Alpaca",
                                details=details)
        except Exception as e:
            return CommandResult(success=False, message=f"Set site error: {e}",
                                details=details)

    def set_time(self, dt: datetime,
                 send_fn: SendFn) -> CommandResult:
        # Alpaca mounts typically get time from the host system
        return CommandResult(success=True,
                             message="Time managed by Alpaca server (host system)")

    # --- Tracking --------------------------------------------------

    @property
    def supports_variable_rate_altaz(self) -> bool:
        """Alpaca supports MoveAxis which can be used for variable rates."""
        return True

    def send_variable_rate_altaz(self, alt_rate_arcsec: float,
                                  az_rate_arcsec: float,
                                  send_fn: SendFn) -> None:
        """Use MoveAxis to set variable tracking rates."""
        alt_rate_degsec = alt_rate_arcsec / 3600.0
        az_rate_degsec = az_rate_arcsec / 3600.0
        self._alpaca_put("moveaxis", {"Axis": 1, "Rate": alt_rate_degsec})
        self._alpaca_put("moveaxis", {"Axis": 0, "Rate": az_rate_degsec})

    # --- Command formatting ----------------------------------------

    def format_outgoing(self, command: str) -> str:
        """Alpaca doesn't use wire-level commands — pass through."""
        return command

    def get_command_delay(self, command: str) -> float:
        return 0.0  # HTTP requests have their own latency

    def get_read_timeout(self, command: str) -> float:
        return 5.0  # HTTP timeout handled internally

    def normalize_response(self, command: str, response: str,
                           log_fn: Callable[[str], None] = lambda _: None) -> str:
        return response


# ===================================================================
# INDIClientMountProtocol
# ===================================================================

class INDIClientMountProtocol(MountProtocol):
    """INDI (Instrument-Neutral Distributed Interface) client protocol.

    INDI is an XML-based protocol over TCP for controlling astronomical
    instruments. An INDI server runs on a host (e.g. Raspberry Pi, StellarMate,
    AstroPC) and exposes device drivers. This protocol connects as a client
    to an INDI server and controls a telescope mount device.

    Architecture:
        TrackWise app  <--TCP/XML-->  INDI Server  <--driver-->  Mount

    INDI uses XML messages over a persistent TCP connection (port 7624):
        - ``<getProperties/>`` — discover devices and properties
        - ``<newNumberVector>`` — set numerical properties (RA, Dec, etc.)
        - ``<newSwitchVector>`` — toggle switches (connect, park, etc.)
        - ``<defNumberVector>`` / ``<setNumberVector>`` — server updates

    This implementation maintains a TCP connection to the INDI server and
    parses XML messages to extract position data and send commands.

    Default port: 7624 (INDI standard).
    """

    def __init__(self):
        self._indi_socket = None  # type: ignore[assignment]
        self._device_name = ""  # Auto-detected or user-specified
        self._ra_hours = 0.0
        self._dec_deg = 0.0
        self._alt_deg = 0.0
        self._az_deg = 0.0
        self._is_slewing = False
        self._is_connected = False

    @property
    def name(self) -> str:
        return "INDI"

    @property
    def default_baudrate(self) -> int:
        return 0  # Not applicable — uses TCP

    @property
    def default_tcp_port(self) -> int:
        return 7624  # INDI standard port

    @property
    def response_terminator(self) -> bytes:
        return b'#'  # Not really used — INDI uses XML

    def _send_indi_xml(self, xml_str: str, send_fn: SendFn) -> str:
        """Send an INDI XML message through the bridge's TCP connection.

        We encode the XML as a raw string and send it through send_fn.
        The response will be raw XML from the INDI server.
        """
        try:
            return send_fn(xml_str, 3.0)
        except Exception as e:
            _logger.debug("INDI send error: %s", e)
            return ""

    def _parse_indi_xml_value(self, xml_text: str, prop_name: str) -> Optional[str]:
        """Simple XML value extractor (avoids heavy XML parser dependency).

        Looks for ``<oneNumber name="prop_name">value</oneNumber>`` or
        ``<defNumber name="prop_name" ...>value</defNumber>`` patterns.
        """
        import re
        # Try <oneNumber name="...">value</oneNumber>
        pattern = rf'<(?:one|def)Number[^>]*name="{re.escape(prop_name)}"[^>]*>([^<]+)</(?:one|def)Number>'
        m = re.search(pattern, xml_text)
        if m:
            return m.group(1).strip()
        return None

    # --- Connection test -------------------------------------------

    def test_connection(self, send_fn: SendFn) -> Tuple[bool, str, bool]:
        """Test INDI connection by sending getProperties and looking for a telescope device."""
        model = "Unknown"

        # Send getProperties to discover all devices
        xml = '<getProperties version="1.7"/>'
        try:
            resp = send_fn(xml.encode('utf-8') if isinstance(xml, str) else xml, 5.0)
            if resp:
                # Look for a telescope device in the response
                import re
                devices = re.findall(r'device="([^"]+)"', resp)
                telescope_keywords = ['telescope', 'mount', 'az', 'dob', 'altaz',
                                      'lx200', 'nexstar', 'ioptron', 'synscan',
                                      'onstep', 'skywatcher', 'meade']
                for dev in devices:
                    dev_lower = dev.lower()
                    for kw in telescope_keywords:
                        if kw in dev_lower:
                            self._device_name = dev
                            model = f"INDI: {dev}"
                            return True, model, False

                # If no telescope keyword found, take the first device
                if devices:
                    self._device_name = devices[0]
                    model = f"INDI: {devices[0]}"
                    return True, model, False
        except Exception:
            pass

        return False, model, False

    # --- Position polling ------------------------------------------

    def poll_position(self, send_fn: SendFn) -> PositionData:
        """Poll INDI telescope position.

        Sends ``getProperties`` for the telescope device and parses the
        EQUATORIAL_EOD_COORD and HORIZONTAL_COORD property values.
        """
        result = PositionData()

        if not self._device_name:
            return result

        # Request current equatorial coordinates
        try:
            xml = f'<getProperties version="1.7" device="{self._device_name}" name="EQUATORIAL_EOD_COORD"/>'
            resp = send_fn(xml, 3.0)
            if resp:
                ra_val = self._parse_indi_xml_value(resp, "RA")
                dec_val = self._parse_indi_xml_value(resp, "DEC")
                if ra_val:
                    result.ra_hours = float(ra_val)
                    ra_h = int(result.ra_hours)
                    ra_m = int((result.ra_hours - ra_h) * 60)
                    ra_s = ((result.ra_hours - ra_h) * 60 - ra_m) * 60
                    result.ra_str = f"{ra_h:02d}:{ra_m:02d}:{ra_s:04.1f}#"
                if dec_val:
                    result.dec_deg = float(dec_val)
                    dec_sign = '+' if result.dec_deg >= 0 else '-'
                    dec_abs = abs(result.dec_deg)
                    dec_d = int(dec_abs)
                    dec_m = int((dec_abs - dec_d) * 60)
                    dec_s = int(((dec_abs - dec_d) * 60 - dec_m) * 60)
                    result.dec_str = f"{dec_sign}{dec_d:02d}*{dec_m:02d}:{dec_s:02d}#"
        except Exception:
            pass

        # Request horizontal coordinates
        time.sleep(0.05)
        try:
            xml = f'<getProperties version="1.7" device="{self._device_name}" name="HORIZONTAL_COORD"/>'
            resp = send_fn(xml, 3.0)
            if resp:
                alt_val = self._parse_indi_xml_value(resp, "ALT")
                az_val = self._parse_indi_xml_value(resp, "AZ")
                if alt_val:
                    result.alt_deg = float(alt_val)
                    result.alt_str = _deg_to_lx200_alt(result.alt_deg) + '#'
                if az_val:
                    result.az_deg = float(az_val)
                    result.az_str = _deg_to_lx200_az(result.az_deg) + '#'
        except Exception:
            pass

        return result

    # --- GoTo ------------------------------------------------------

    def goto_radec(self, ra_str: str, dec_str: str,
                   send_fn: SendFn) -> CommandResult:
        details = []
        try:
            ra_hours = _parse_lx200_dms(ra_str)
            dec_deg = _parse_lx200_dms(dec_str)

            # First, set the on-coord action to SLEW
            xml = (f'<newSwitchVector device="{self._device_name}" name="ON_COORD_SET">'
                   f'<oneSwitch name="TRACK">On</oneSwitch>'
                   f'<oneSwitch name="SLEW">Off</oneSwitch>'
                   f'<oneSwitch name="SYNC">Off</oneSwitch>'
                   f'</newSwitchVector>')
            resp = send_fn(xml, 2.0)
            details.append(("ON_COORD_SET=TRACK", resp))

            # Send target coordinates
            xml = (f'<newNumberVector device="{self._device_name}" name="EQUATORIAL_EOD_COORD">'
                   f'<oneNumber name="RA">{ra_hours:.6f}</oneNumber>'
                   f'<oneNumber name="DEC">{dec_deg:.6f}</oneNumber>'
                   f'</newNumberVector>')
            resp = send_fn(xml, 2.0)
            details.append(("EQUATORIAL_EOD_COORD", resp))

            return CommandResult(success=True, message="INDI GoTo sent",
                                details=details)
        except Exception as e:
            return CommandResult(success=False, message=f"GoTo error: {e}",
                                details=details)

    def goto_altaz(self, alt_str: str, az_str: str,
                   send_fn: SendFn) -> CommandResult:
        details = []
        try:
            alt_deg = _parse_lx200_dms(alt_str)
            az_deg = _parse_lx200_dms(az_str)

            xml = (f'<newNumberVector device="{self._device_name}" name="HORIZONTAL_COORD">'
                   f'<oneNumber name="ALT">{alt_deg:.6f}</oneNumber>'
                   f'<oneNumber name="AZ">{az_deg:.6f}</oneNumber>'
                   f'</newNumberVector>')
            resp = send_fn(xml, 2.0)
            details.append(("HORIZONTAL_COORD", resp))

            return CommandResult(success=True, message="INDI Alt/Az GoTo sent",
                                details=details)
        except Exception as e:
            return CommandResult(success=False, message=f"GoTo error: {e}",
                                details=details)

    # --- Motion control --------------------------------------------

    def slew(self, direction: str, speed: int,
             send_fn: SendFn) -> None:
        """Start motion using INDI TELESCOPE_MOTION_NS / TELESCOPE_MOTION_WE."""
        motion_map = {
            'N': ("TELESCOPE_MOTION_NS", "MOTION_NORTH", "MOTION_SOUTH"),
            'S': ("TELESCOPE_MOTION_NS", "MOTION_SOUTH", "MOTION_NORTH"),
            'E': ("TELESCOPE_MOTION_WE", "MOTION_EAST", "MOTION_WEST"),
            'W': ("TELESCOPE_MOTION_WE", "MOTION_WEST", "MOTION_EAST"),
        }
        if direction not in motion_map:
            return

        # Set slew rate first
        rate_names = {1: "1x", 2: "2x", 3: "3x", 4: "4x"}
        rate_name = rate_names.get(speed, "4x")
        try:
            xml = (f'<newSwitchVector device="{self._device_name}" name="TELESCOPE_SLEW_RATE">'
                   f'<oneSwitch name="{rate_name}">On</oneSwitch>'
                   f'</newSwitchVector>')
            send_fn(xml, 1.0)
        except Exception:
            pass

        prop, on_name, off_name = motion_map[direction]
        xml = (f'<newSwitchVector device="{self._device_name}" name="{prop}">'
               f'<oneSwitch name="{on_name}">On</oneSwitch>'
               f'<oneSwitch name="{off_name}">Off</oneSwitch>'
               f'</newSwitchVector>')
        try:
            send_fn(xml, 1.0)
        except Exception:
            pass

    def stop(self, send_fn: SendFn) -> None:
        """Abort slew via INDI TELESCOPE_ABORT_MOTION."""
        xml = (f'<newSwitchVector device="{self._device_name}" name="TELESCOPE_ABORT_MOTION">'
               f'<oneSwitch name="ABORT">On</oneSwitch>'
               f'</newSwitchVector>')
        try:
            send_fn(xml, 1.0)
        except Exception:
            pass

    # --- Sync ------------------------------------------------------

    def sync_radec(self, ra_str: str, dec_str: str,
                   send_fn: SendFn) -> CommandResult:
        details = []
        try:
            ra_hours = _parse_lx200_dms(ra_str)
            dec_deg = _parse_lx200_dms(dec_str)

            # Set coord action to SYNC
            xml = (f'<newSwitchVector device="{self._device_name}" name="ON_COORD_SET">'
                   f'<oneSwitch name="TRACK">Off</oneSwitch>'
                   f'<oneSwitch name="SLEW">Off</oneSwitch>'
                   f'<oneSwitch name="SYNC">On</oneSwitch>'
                   f'</newSwitchVector>')
            resp = send_fn(xml, 2.0)
            details.append(("ON_COORD_SET=SYNC", resp))

            # Send coordinates (in SYNC mode this syncs instead of slewing)
            xml = (f'<newNumberVector device="{self._device_name}" name="EQUATORIAL_EOD_COORD">'
                   f'<oneNumber name="RA">{ra_hours:.6f}</oneNumber>'
                   f'<oneNumber name="DEC">{dec_deg:.6f}</oneNumber>'
                   f'</newNumberVector>')
            resp = send_fn(xml, 2.0)
            details.append(("EQUATORIAL_EOD_COORD (sync)", resp))

            # Restore TRACK mode
            xml = (f'<newSwitchVector device="{self._device_name}" name="ON_COORD_SET">'
                   f'<oneSwitch name="TRACK">On</oneSwitch>'
                   f'<oneSwitch name="SLEW">Off</oneSwitch>'
                   f'<oneSwitch name="SYNC">Off</oneSwitch>'
                   f'</newSwitchVector>')
            send_fn(xml, 1.0)

            return CommandResult(success=True, message="INDI Synced",
                                details=details)
        except Exception as e:
            return CommandResult(success=False, message=f"Sync error: {e}",
                                details=details)

    def sync_altaz(self, alt_str: str, az_str: str,
                   send_fn: SendFn) -> CommandResult:
        return CommandResult(success=False,
                             message="INDI: Alt/Az sync not directly supported. Use RA/Dec sync.")

    # --- Park / Home -----------------------------------------------

    def park(self, send_fn: SendFn) -> None:
        xml = (f'<newSwitchVector device="{self._device_name}" name="TELESCOPE_PARK">'
               f'<oneSwitch name="PARK">On</oneSwitch>'
               f'</newSwitchVector>')
        try:
            send_fn(xml, 2.0)
        except Exception:
            pass

    def home(self, send_fn: SendFn) -> None:
        # INDI doesn't have a universal "home" command
        # Some drivers support TELESCOPE_HOME
        xml = (f'<newSwitchVector device="{self._device_name}" name="TELESCOPE_HOME">'
               f'<oneSwitch name="GO">On</oneSwitch>'
               f'</newSwitchVector>')
        try:
            send_fn(xml, 2.0)
        except Exception:
            pass

    # --- Site / Time -----------------------------------------------

    def set_site(self, lat: float, lon: float,
                 utc_offset_west_h: float,
                 send_fn: SendFn) -> CommandResult:
        details = []
        try:
            xml = (f'<newNumberVector device="{self._device_name}" name="GEOGRAPHIC_COORD">'
                   f'<oneNumber name="LAT">{lat:.6f}</oneNumber>'
                   f'<oneNumber name="LONG">{lon:.6f}</oneNumber>'
                   f'<oneNumber name="ELEV">0</oneNumber>'
                   f'</newNumberVector>')
            resp = send_fn(xml, 2.0)
            details.append(("GEOGRAPHIC_COORD", resp))
            return CommandResult(success=True, message="INDI site set",
                                details=details)
        except Exception as e:
            return CommandResult(success=False, message=f"Set site error: {e}",
                                details=details)

    def set_time(self, dt: datetime,
                 send_fn: SendFn) -> CommandResult:
        details = []
        try:
            # INDI uses ISO format for UTC time
            utc_str = dt.strftime("%Y-%m-%dT%H:%M:%S")
            xml = (f'<newTextVector device="{self._device_name}" name="TIME_UTC">'
                   f'<oneText name="UTC">{utc_str}</oneText>'
                   f'<oneText name="OFFSET">0</oneText>'
                   f'</newTextVector>')
            resp = send_fn(xml, 2.0)
            details.append(("TIME_UTC", resp))
            return CommandResult(success=True, message="INDI time set",
                                details=details)
        except Exception as e:
            return CommandResult(success=False, message=f"Set time error: {e}",
                                details=details)

    # --- Command formatting ----------------------------------------

    def format_outgoing(self, command: str) -> str:
        """INDI uses XML — no additional formatting needed."""
        return command

    def get_command_delay(self, command: str) -> float:
        return 0.0  # XML messages are self-delimited

    def get_read_timeout(self, command: str) -> float:
        return 3.0  # INDI responses can be slow


# ===================================================================
# Protocol registry
# ===================================================================

PROTOCOL_REGISTRY: Dict[str, type] = {
    "lx200": LX200MountProtocol,
    "nexstar": NexStarMountProtocol,
    "ioptron": iOptronMountProtocol,
    "audiostar": MeadeAudioStarMountProtocol,
    "alpaca": ASCOMAlpacaMountProtocol,
    "indi": INDIClientMountProtocol,
}


def get_protocol(name: str) -> MountProtocol:
    """Instantiate a protocol by name.

    Args:
        name: Protocol identifier (e.g. ``'lx200'``, ``'nexstar'``,
              ``'ioptron'``, ``'audiostar'``, ``'alpaca'``, ``'indi'``).

    Returns:
        A fresh ``MountProtocol`` instance.

    Raises:
        KeyError: If the name is not in the registry.
    """
    cls = PROTOCOL_REGISTRY[name.lower()]
    return cls()


def list_protocols() -> List[str]:
    """Return the registered protocol names."""
    return list(PROTOCOL_REGISTRY.keys())
