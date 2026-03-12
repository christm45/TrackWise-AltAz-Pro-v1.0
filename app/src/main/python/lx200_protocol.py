"""
LX200 Protocol Implementation - Virtual Meade LX200 Telescope Server

This module implements a virtual Meade LX200-compatible telescope server that allows
planetarium software (Stellarium, SkySafari, ASCOM Alpaca, etc.) to control a Dobson
Alt-Az telescope mount using the standard LX200 serial command protocol.

Architecture Role:
    This protocol layer sits between external planetarium clients (e.g. Stellarium)
    and the physical telescope hardware interface (telescope_bridge.py). It translates
    standardized LX200 commands into internal telescope state changes and notifies the
    main application via callbacks.

    Stellarium / SkySafari          (Planetarium software - sends LX200 commands)
            |
            v
    LX200Protocol                   (This module - command parsing & coordinate conversion)
            |
            v
    telescope_bridge.py             (Hardware interface - drives stepper motors)

Coordinate Systems:
    - RA/Dec (Right Ascension / Declination): The equatorial coordinate system used by
      planetarium software. RA is measured in hours (0-24h), Dec in degrees (-90 to +90).
    - Alt/Az (Altitude / Azimuth): The horizontal coordinate system native to the Dobson
      mount. Alt is elevation above horizon (0-90 deg), Az is compass bearing (0-360 deg).
    - Since a Dobson is an Alt-Az mount, Alt/Az is the "source of truth" for the physical
      telescope position. RA/Dec values are derived from Alt/Az via sidereal time and
      spherical trigonometry, and are only maintained for protocol compatibility.

Coordinate Conversion (RA/Dec <-> Alt/Az):
    Conversion relies on the observer's geographic location (latitude, longitude) and the
    Local Sidereal Time (LST). The Hour Angle (HA) bridges the two systems:
        HA = LST - RA
    The standard spherical trigonometry formulas are then applied:
        sin(Alt) = sin(Lat)*sin(Dec) + cos(Lat)*cos(Dec)*cos(HA)
        Az = atan2(-cos(Dec)*sin(HA), sin(Dec)*cos(Lat) - cos(Dec)*sin(Lat)*cos(HA))

Command Categories (LX200 Protocol):
    - Position queries:   :GR# (get RA), :GD# (get Dec), :GA# (get Alt), :GZ# (get Az)
    - Target setting:     :Sr# (set target RA), :Sd# (set target Dec)
    - Slewing:            :MS# (slew to target), :Q# (abort slew)
    - Manual movement:    :Me#/:Mw#/:Mn#/:Ms# (move E/W/N/S), :Qe#/:Qw#/:Qn#/:Qs# (stop)
    - Slew speed:         :RS# (max), :RM# (find), :RC# (center), :RG# (guide)
    - Site configuration: :Sg#/:St# (set long/lat), :Gg#/:Gt# (get long/lat)
    - Time/Date:          :GL#/:GS#/:GC# (get time/sidereal/date), :SL#/:SC# (set time/date)
    - Sync:               :CM# (sync current position to target)
    - Info:               :GVP# (product name), :GVN# (firmware version), :D# (slew status)
    - Precision:          :U# (toggle high/low precision), :P# (get precision mode)
    - Tracking rates:     :SXTR,n# / :SXTD,n# (set RA/Dec tracking rate offsets)
                          :GXTR# / :GXTD# (get current tracking rate offsets)
    - Alt-Az precise:     :GAL# (precise altitude), :GAZ# (precise azimuth)
    - Extended info:      :GX# (get all info in CSV format)
    - Derotator (custom): :DR+#/:DR-# (rotate CW/CCW), :DRQ# (stop), :DR0# (sync to 0)

Callback System:
    The protocol notifies the main application of state changes via optional callbacks:
    - on_goto(alt_deg, az_deg):             Called when a GOTO slew is initiated (:MS#)
    - on_position_update(alt_deg, az_deg):  Called when position changes (sync, slew progress)
    - on_stop():                            Called when all motion is stopped (:Q#)
    - on_tracking_rate_change(ra_rate, dec_rate): Called when tracking rates change (:SXTR#/:SXTD#)
                                            Used by realtime_tracking.py for drift correction
    - on_derotator_rotate(direction, speed): Called for derotator rotation commands
    - on_derotator_stop():                  Called to stop the derotator
    - on_derotator_sync():                  Called to reset derotator angle to 0

Usage:
    protocol = LX200Protocol()
    protocol.on_goto = my_goto_handler
    protocol.set_position_altaz(alt=45.0, az=180.0)
    response = protocol.process_command(":GR#")  # Returns current RA
"""

import math
from datetime import datetime, timezone
from typing import Tuple, Optional, Callable
import re

from telescope_logger import get_logger

_logger = get_logger(__name__)


class LX200Protocol:
    """
    Virtual Meade LX200-compatible telescope protocol handler.

    Implements the LX200 command set to present a Dobson Alt-Az mount as a standard
    LX200 telescope to planetarium software. All LX200 commands are received as strings
    (e.g. ":GR#"), parsed, and dispatched to internal handler methods that return the
    appropriate LX200 response string.

    The internal state maintains both RA/Dec and Alt/Az representations of the telescope
    position. Since this is an Alt-Az (Dobson) mount, Alt/Az is the authoritative
    coordinate system; RA/Dec values are computed from Alt/Az using the observer's
    location and Local Sidereal Time.

    Attributes:
        ra_hours (float):           Current right ascension in decimal hours (0-24).
        dec_degrees (float):        Current declination in decimal degrees (-90 to +90).
        alt_degrees (float):        Current altitude (elevation) in degrees (0-90).
        az_degrees (float):         Current azimuth in degrees (0-360).
        target_alt (float):         GOTO target altitude in degrees.
        target_az (float):          GOTO target azimuth in degrees.
        target_ra (float):          GOTO target RA in decimal hours (protocol compatibility).
        target_dec (float):         GOTO target Dec in decimal degrees (protocol compatibility).
        is_slewing (bool):          True if the telescope is currently slewing to a target.
        is_tracking (bool):         True if sidereal tracking corrections are active.
        slew_speed (int):           Current slew rate: 1=guide, 2=center, 3=find, 4=max.
        tracking_rate_alt (float):  Alt-axis tracking correction rate in arcsec/sec.
        tracking_rate_az (float):   Az-axis tracking correction rate in arcsec/sec.
        tracking_rate_ra (float):   RA tracking rate offset in arcsec/sec (protocol alias for SXTR).
        tracking_rate_dec (float):  Dec tracking rate offset in arcsec/sec (protocol alias for SXTD).
        latitude (float):           Observer latitude in decimal degrees.
        longitude (float):          Observer longitude in decimal degrees.
        utc_offset (int):           UTC timezone offset in hours.
        high_precision (bool):      If True, position responses include arc-seconds;
                                    if False, only degrees and arc-minutes.
        on_goto (Callable):         Callback: on_goto(alt_degrees, az_degrees).
        on_position_update (Callable): Callback: on_position_update(alt_degrees, az_degrees).
        on_stop (Callable):         Callback: on_stop().
        on_tracking_rate_change (Callable): Callback: on_tracking_rate_change(ra_rate, dec_rate).
        on_derotator_rotate (Callable): Callback: on_derotator_rotate(direction, speed).
        on_derotator_stop (Callable): Callback: on_derotator_stop().
        on_derotator_sync (Callable): Callback: on_derotator_sync().
    """

    def __init__(self):
        # --- Current telescope position (decimal degrees / hours) ---
        self.ra_hours = 0.0       # Right Ascension in hours (0-24h)
        self.dec_degrees = 0.0    # Declination in degrees (-90 to +90)

        # Alt-Az position (native coordinate system for Dobson mount)
        # Home position: horizon (0°) pointing North (0°)
        self.alt_degrees = 0.0    # Altitude / elevation in degrees (0-90)
        self.az_degrees = 0.0     # Azimuth in degrees (0-360, 0=North, 90=East)

        # --- GOTO target coordinates (Alt-Az is the native target system) ---
        self.target_alt = 0.0
        self.target_az = 0.0
        # RA/Dec targets maintained for LX200 protocol compatibility (set by :Sr#/:Sd#)
        self.target_ra = 0.0
        self.target_dec = 0.0

        # --- Telescope state ---
        self.is_slewing = False
        self.is_tracking = True
        self.slew_speed = 4       # Slew rate: 1=guide, 2=center, 3=find, 4=max
        self.tracking_rate_mode = 'sidereal'  # 'sidereal', 'lunar', 'solar', 'king'

        # --- OnStepX tracking configuration ---
        # Master sidereal clock frequency (Hz). Standard sidereal = 60.16427 Hz.
        # :T+#/:T-# adjust by 0.02 Hz, :TR# resets to default.
        self.SIDEREAL_FREQ_DEFAULT = 60.16427
        self.sidereal_clock_freq = self.SIDEREAL_FREQ_DEFAULT
        # Tracking rate frequencies for each mode (Hz)
        self._tracking_freqs = {
            'sidereal': 60.16427,
            'lunar':    58.74122,
            'solar':    60.0,
            'king':     60.15036,
        }
        # Tracking axis mode: 1 = RA-only (equatorial), 2 = dual-axis (Alt-Az)
        self.tracking_axis_mode = 2   # Default dual-axis for Alt-Az
        # Compensation model: 'full', 'refraction', 'none'
        self.compensation_model = 'full'
        # Backlash in arcseconds
        self.backlash_ra = 0.0    # RA / Azm backlash (arcsec)
        self.backlash_dec = 0.0   # Dec / Alt backlash (arcsec)

        # --- Custom tracking rate offsets (arcsec/sidereal-sec) ---
        # Set by SXTR/SXTD commands from the real-time correction pipeline.
        self.tracking_rate_alt = 0.0   # Altitude tracking rate (arcsec/sec)
        self.tracking_rate_az = 0.0    # Azimuth tracking rate (arcsec/sec)
        self.tracking_rate_ra = 0.0    # RA offset (SXTR) arcsec/sidereal-sec
        self.tracking_rate_dec = 0.0   # Dec offset (SXTD) arcsec/sidereal-sec

        # --- Observatory location (defaults; overridden by config_manager) ---
        self.latitude = 48.8566   # Default: Paris; see config_manager.DEFAULT_LATITUDE
        self.longitude = 2.3522   # Default: Paris; see config_manager.DEFAULT_LONGITUDE
        self.utc_offset = 1       # UTC offset in hours (UTC+1 for CET)

        # --- Event callbacks (set by the main application) ---
        self.on_goto: Optional[Callable] = None  # (alt_degrees, az_degrees)
        self.on_position_update: Optional[Callable] = None  # (alt_degrees, az_degrees) - Alt-Az only
        self.on_stop: Optional[Callable] = None
        self.on_tracking_rate_change: Optional[Callable] = None  # For SXTR/SXTD commands
        # Focuser callbacks
        self.on_focuser_move: Optional[Callable] = None  # (direction)
        self.on_focuser_stop: Optional[Callable] = None
        # Derotator callbacks (custom extension for field derotator hardware)
        self.on_derotator_rotate: Optional[Callable] = None  # (direction, speed)
        self.on_derotator_stop: Optional[Callable] = None
        self.on_derotator_sync: Optional[Callable] = None

        # Coordinate format precision: high = DD*MM'SS, low = DD*MM
        self.high_precision = True

    def process_command(self, command: str) -> str:
        """
        Parse and dispatch a single LX200 protocol command, returning the response.

        LX200 commands follow the format `:XX...#` where the leading colon and trailing
        hash are delimiters. This method strips those delimiters, identifies the command
        by its prefix, and routes it to the appropriate handler method.

        Some clients (e.g. ASCOM Alpaca) may send commands without the leading colon,
        or may send empty strings as keep-alive probes. These edge cases are handled
        gracefully by returning "0#".

        Args:
            command: Raw LX200 command string, e.g. ":GR#", ":Sr12:30:00#".

        Returns:
            LX200 protocol response string (always ends with '#').
        """
        if not command:
            return "0#"  # Reply even to empty commands (ASCOM Alpaca expects a response)

        # Strip whitespace from the raw command
        command = command.strip()

        # Handle commands with or without the leading ':' (some clients omit it)
        if command.startswith(':'):
            command = command[1:]
        if command.endswith('#'):
            command = command[:-1]

        # If the command is empty after stripping delimiters, still respond
        if not command:
            return "0#"

        # ===== Command Dispatch Table =====
        # Routes the cleaned command string to the appropriate handler method.
        # Commands are matched by prefix, checked in order. The first matching
        # prefix wins, so more specific prefixes (e.g. 'SXTR') must appear
        # before shorter overlapping ones (e.g. 'GX').
        #
        # Categories:
        #   G*  = Get (query) commands
        #   S*  = Set commands
        #   M*  = Movement commands
        #   Q*  = Quit/stop commands
        #   R*  = Rate (slew speed) commands
        #   DR* = Derotator commands (custom extension)

        # --- Position query commands ---
        # High-precision variants (H suffix) must be checked before standard
        if command.startswith('GRH'):
            return self._get_ra_high_precision()
        elif command.startswith('GDH'):
            return self._get_dec_high_precision()
        elif command.startswith('GAH'):
            return self._get_altitude_high_precision()
        elif command.startswith('GZH'):
            return self._get_azimuth_high_precision()
        elif command.startswith('GR'):
            return self._get_ra()
        elif command.startswith('GD'):
            return self._get_dec()
        elif command.startswith('GA'):
            return self._get_altitude()
        elif command.startswith('GZ'):
            return self._get_azimuth()
        elif command.startswith('GT'):
            return self._get_tracking_rate_hz()
        elif command.startswith('GL'):
            return self._get_local_time()
        elif command.startswith('GC'):
            return self._get_date()
        elif command.startswith('Gg'):
            return self._get_longitude()
        elif command.startswith('Gt'):
            return self._get_latitude()
        elif command.startswith('GS'):
            return self._get_sidereal_time()

        # --- Target coordinate setting commands ---
        elif command.startswith('Sr'):
            return self._set_target_ra(command[2:])
        elif command.startswith('Sd'):
            return self._set_target_dec(command[2:])

        # --- Slew and motion commands ---
        elif command.startswith('MS'):
            return self._slew_to_target()
        elif command.startswith('Q'):
            return self._stop_slew()
        elif command.startswith('Me'):
            return self._move_east()
        elif command.startswith('Mw'):
            return self._move_west()
        elif command.startswith('Mn'):
            return self._move_north()
        elif command.startswith('Ms'):
            return self._move_south()
        elif command.startswith('Qe') or command.startswith('Qw') or \
             command.startswith('Qn') or command.startswith('Qs'):
            return self._stop_move()

        # --- Slew rate commands ---
        elif command.startswith('RS'):
            return self._set_slew_rate_max()
        elif command.startswith('RM'):
            return self._set_slew_rate_find()
        elif command.startswith('RC'):
            return self._set_slew_rate_center()
        elif command.startswith('RG'):
            return self._set_slew_rate_guide()

        # --- Tracking rate / enable commands (OnStepX) ---
        # Rate presets
        elif command == 'TQ':
            self.tracking_rate_mode = 'sidereal'
            self.is_tracking = True
            return "1#"
        elif command == 'TL':
            self.tracking_rate_mode = 'lunar'
            self.is_tracking = True
            return "1#"
        elif command == 'TS':
            self.tracking_rate_mode = 'solar'
            self.is_tracking = True
            return "1#"
        elif command == 'TK':
            self.tracking_rate_mode = 'king'
            self.is_tracking = True
            return "1#"
        # Enable / disable
        elif command == 'Te':
            self.is_tracking = True
            return "1#"
        elif command == 'Td':
            self.is_tracking = False
            return "1#"
        # Compensation model
        elif command == 'To':
            self.compensation_model = 'full'
            return "1#"
        elif command == 'Tr':
            self.compensation_model = 'refraction'
            return "1#"
        elif command == 'Tn':
            self.compensation_model = 'none'
            return "1#"
        # Axis mode
        elif command == 'T1':
            self.tracking_axis_mode = 1
            return "1#"
        elif command == 'T2':
            self.tracking_axis_mode = 2
            return "1#"
        # Master sidereal clock adjustment
        elif command == 'T+':
            self.sidereal_clock_freq += 0.02
            return "1#"
        elif command == 'T-':
            self.sidereal_clock_freq -= 0.02
            return "1#"
        elif command == 'TR':
            self.sidereal_clock_freq = self.SIDEREAL_FREQ_DEFAULT
            return "1#"
        # :STn.n# - Set tracking rate in Hz (0 stops tracking)
        elif command.startswith('ST'):
            return self._set_tracking_rate_hz(command[2:])

        # --- Backlash commands (OnStepX) ---
        elif command.startswith('$BD'):
            return self._set_backlash_dec(command[3:])
        elif command.startswith('$BR'):
            return self._set_backlash_ra(command[3:])
        elif command.startswith('%BD'):
            return f"{self.backlash_dec:.0f}#"
        elif command.startswith('%BR'):
            return f"{self.backlash_ra:.0f}#"

        # --- Site configuration commands ---
        elif command.startswith('Sg'):
            return self._set_longitude(command[2:])
        elif command.startswith('St'):
            return self._set_latitude(command[2:])
        elif command.startswith('SL'):
            return self._set_local_time(command[2:])
        elif command.startswith('SC'):
            return self._set_date(command[2:])

        # --- Sync and precision commands ---
        elif command.startswith('CM'):
            return self._sync_to_target()
        elif command.startswith('U'):
            return self._toggle_precision()

        # --- Derotator commands (custom extension for field derotator hardware) ---
        # NOTE: These must be checked BEFORE the generic 'D' (slew status)
        # command, because 'DR+...' also starts with 'D'.
        elif command.startswith('DR+'):
            # Rotate derotator clockwise: :DR+{speed}#
            return self._rotate_derotator_cw(command[3:])
        elif command.startswith('DR-'):
            # Rotate derotator counter-clockwise: :DR-{speed}#
            return self._rotate_derotator_ccw(command[3:])
        elif command.startswith('DRQ'):
            # Stop derotator: :DRQ#
            return self._stop_derotator()
        elif command.startswith('DR0'):
            # Sync derotator angle to 0 degrees: :DR0#
            return self._sync_derotator()

        # --- Identification / info commands ---
        elif command.startswith('GVP'):
            return self._get_product_name()
        elif command.startswith('GVN'):
            return self._get_firmware_version()
        elif command.startswith('D'):
            return self._get_slew_status()
        elif command.startswith('P'):
            return self._get_high_precision()

        # --- Custom tracking rate commands (for real-time drift correction) ---
        # Used by realtime_tracking.py to send Alt-Az correction velocities
        elif command.startswith('SXTR'):
            return self._set_tracking_rate_ra(command[5:])  # Skip "SXTR," prefix
        elif command.startswith('SXTD'):
            return self._set_tracking_rate_dec(command[5:])  # Skip "SXTD," prefix
        elif command.startswith('GXTR'):
            return self._get_tracking_rate_ra()
        elif command.startswith('GXTD'):
            return self._get_tracking_rate_dec()

        # --- Alt-Az specific high-precision queries ---
        elif command.startswith('GAL'):
            return self._get_altitude_precise()
        elif command.startswith('GAZ'):
            return self._get_azimuth_precise()
        elif command.startswith('GX'):
            # Extended info: returns all coordinates and site data as CSV
            return self._get_extended_info()

        # --- Focuser commands (standard LX200 / OnStep) ---
        elif command.startswith('F+'):
            # Move focuser inward: :F+#
            return self._focuser_move("IN")
        elif command.startswith('F-'):
            # Move focuser outward: :F-#
            return self._focuser_move("OUT")
        elif command.startswith('FQ'):
            # Stop focuser: :FQ#
            return self._focuser_stop()
        elif command.startswith('F') and len(command) == 2 and command[1].isdigit():
            # Set focuser speed: :F{1-4}#
            return self._focuser_set_speed(int(command[1]))

        else:
            # Unknown command - return '0' to indicate "not supported"
            # Some clients expect a response even for unrecognized commands
            _logger.debug("Unknown LX200 command: '%s'", command)
            return "0#"

    # ========== Position Query Commands ==========

    def _get_ra(self) -> str:
        """
        Return the current Right Ascension in LX200 format.

        High precision format: HH:MM:SS.s (hours, minutes, seconds with tenths)
        Low precision format:  HH:MM.t    (hours, minutes with tenths of minutes)

        Responds to the :GR# command.

        Returns:
            Formatted RA string ending with '#'.
        """
        hours = int(self.ra_hours)
        # Convert fractional hours to minutes and seconds
        minutes_total = (self.ra_hours - hours) * 60
        minutes = int(minutes_total)
        seconds = (minutes_total - minutes) * 60

        if self.high_precision:
            return f"{hours:02d}:{minutes:02d}:{seconds:04.1f}#"
        else:
            # Low precision: tenths of minutes (seconds / 6 gives tenths of a minute)
            return f"{hours:02d}:{minutes:02d}.{int(seconds/6):01d}#"

    def _get_dec(self) -> str:
        """
        Return the current Declination in LX200 format.

        High precision format: sDD*MM'SS (sign, degrees, arcminutes, arcseconds)
        Low precision format:  sDD*MM    (sign, degrees, arcminutes)

        Responds to the :GD# command.

        Returns:
            Formatted Dec string ending with '#'.
        """
        sign = '+' if self.dec_degrees >= 0 else '-'
        dec_abs = abs(self.dec_degrees)
        degrees = int(dec_abs)
        # Convert fractional degrees to arcminutes and arcseconds
        minutes_total = (dec_abs - degrees) * 60
        minutes = int(minutes_total)
        seconds = (minutes_total - minutes) * 60

        if self.high_precision:
            return f"{sign}{degrees:02d}*{minutes:02d}'{seconds:02.0f}#"
        else:
            return f"{sign}{degrees:02d}*{minutes:02d}#"

    def _get_altitude(self) -> str:
        """
        Return the current altitude (elevation above horizon) in LX200 format.

        Computes Alt/Az from the current RA/Dec using the observer's location and
        Local Sidereal Time. Format: sDD*MM (sign, degrees, arcminutes).

        Responds to the :GA# command.

        Returns:
            Formatted altitude string ending with '#'.
        """
        alt, az = self._ra_dec_to_alt_az(self.ra_hours, self.dec_degrees)
        sign = '+' if alt >= 0 else '-'
        alt_abs = abs(alt)
        degrees = int(alt_abs)
        minutes = int((alt_abs - degrees) * 60)
        return f"{sign}{degrees:02d}*{minutes:02d}#"

    def _get_azimuth(self) -> str:
        """
        Return the current azimuth in LX200 format.

        Computes Alt/Az from the current RA/Dec using the observer's location and
        Local Sidereal Time. Format: DDD*MM (degrees, arcminutes).

        Responds to the :GZ# command.

        Returns:
            Formatted azimuth string ending with '#'.
        """
        alt, az = self._ra_dec_to_alt_az(self.ra_hours, self.dec_degrees)
        degrees = int(az)
        minutes = int((az - degrees) * 60)
        return f"{degrees:03d}*{minutes:02d}#"

    def _get_local_time(self) -> str:
        """
        Return the current local time.

        Format: HH:MM:SS (24-hour).

        Responds to the :GL# command.

        Returns:
            Formatted local time string ending with '#'.
        """
        now = datetime.now()
        return f"{now.hour:02d}:{now.minute:02d}:{now.second:02d}#"

    def _get_date(self) -> str:
        """
        Return the current date.

        Format: MM/DD/YY (month, day, 2-digit year).

        Responds to the :GC# command.

        Returns:
            Formatted date string ending with '#'.
        """
        now = datetime.now()
        return f"{now.month:02d}/{now.day:02d}/{now.year % 100:02d}#"

    def _get_longitude(self) -> str:
        """
        Return the observatory longitude.

        Format: DDD*MM (degrees, arcminutes). Sign is implicit in the degree value.

        Responds to the :Gg# command.

        Returns:
            Formatted longitude string ending with '#'.
        """
        degrees = int(abs(self.longitude))
        minutes = int((abs(self.longitude) - degrees) * 60)
        return f"{degrees:03d}*{minutes:02d}#"

    def _get_latitude(self) -> str:
        """
        Return the observatory latitude.

        Format: sDD*MM (sign, degrees, arcminutes).

        Responds to the :Gt# command.

        Returns:
            Formatted latitude string ending with '#'.
        """
        sign = '+' if self.latitude >= 0 else '-'
        degrees = int(abs(self.latitude))
        minutes = int((abs(self.latitude) - degrees) * 60)
        return f"{sign}{degrees:02d}*{minutes:02d}#"

    def _get_sidereal_time(self) -> str:
        """
        Return the Local Sidereal Time (LST).

        LST is the hour angle of the vernal equinox and is essential for converting
        between RA and hour angle: HA = LST - RA.

        Format: HH:MM:SS.

        Responds to the :GS# command.

        Returns:
            Formatted LST string ending with '#'.
        """
        lst = self._calculate_lst()
        hours = int(lst)
        minutes = int((lst - hours) * 60)
        seconds = int(((lst - hours) * 60 - minutes) * 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}#"

    # ========== High-Precision Position Queries (OnStepX) ==========

    def _get_ra_high_precision(self) -> str:
        """Return RA in highest precision: HH:MM:SS.SSSS#. Responds to :GRH#."""
        hours = int(self.ra_hours)
        minutes_total = (self.ra_hours - hours) * 60
        minutes = int(minutes_total)
        seconds = (minutes_total - minutes) * 60
        return f"{hours:02d}:{minutes:02d}:{seconds:07.4f}#"

    def _get_dec_high_precision(self) -> str:
        """Return Dec in highest precision: sDD*MM:SS.SSS#. Responds to :GDH#."""
        sign = '+' if self.dec_degrees >= 0 else '-'
        dec_abs = abs(self.dec_degrees)
        degrees = int(dec_abs)
        minutes_total = (dec_abs - degrees) * 60
        minutes = int(minutes_total)
        seconds = (minutes_total - minutes) * 60
        return f"{sign}{degrees:02d}*{minutes:02d}:{seconds:06.3f}#"

    def _get_altitude_high_precision(self) -> str:
        """Return altitude in highest precision: sDD*MM'SS.SSS#. Responds to :GAH#."""
        alt, _ = self._ra_dec_to_alt_az(self.ra_hours, self.dec_degrees)
        sign = '+' if alt >= 0 else '-'
        alt_abs = abs(alt)
        degrees = int(alt_abs)
        minutes_total = (alt_abs - degrees) * 60
        minutes = int(minutes_total)
        seconds = (minutes_total - minutes) * 60
        return f"{sign}{degrees:02d}*{minutes:02d}'{seconds:06.3f}#"

    def _get_azimuth_high_precision(self) -> str:
        """Return azimuth in highest precision: DDD*MM'SS.SSS#. Responds to :GZH#."""
        _, az = self._ra_dec_to_alt_az(self.ra_hours, self.dec_degrees)
        degrees = int(az)
        minutes_total = (az - degrees) * 60
        minutes = int(minutes_total)
        seconds = (minutes_total - minutes) * 60
        return f"{degrees:03d}*{minutes:02d}'{seconds:06.3f}#"

    # ========== Tracking Rate Commands (OnStepX) ==========

    def _get_tracking_rate_hz(self) -> str:
        """Return tracking rate in Hz, or 0 if not tracking. Responds to :GT#."""
        if not self.is_tracking:
            return "0#"
        freq = self._tracking_freqs.get(self.tracking_rate_mode,
                                        self.SIDEREAL_FREQ_DEFAULT)
        return f"{freq:.5f}#"

    def _set_tracking_rate_hz(self, value: str) -> str:
        """Set tracking rate in Hz. 0 stops tracking. Responds to :STn.n#."""
        try:
            freq = float(value.strip())
            if freq <= 0:
                self.is_tracking = False
                return "1#"
            # Find closest standard mode or accept as custom
            self.sidereal_clock_freq = freq
            self.is_tracking = True
            return "1#"
        except ValueError:
            return "0#"

    def _set_backlash_dec(self, value: str) -> str:
        """Set Dec/Alt backlash in arcsec. Responds to :$BDn#."""
        try:
            self.backlash_dec = float(value.strip())
            return "1#"
        except ValueError:
            return "0#"

    def _set_backlash_ra(self, value: str) -> str:
        """Set RA/Azm backlash in arcsec. Responds to :$BRn#."""
        try:
            self.backlash_ra = float(value.strip())
            return "1#"
        except ValueError:
            return "0#"

    # ========== Target Coordinate Setting Commands ==========

    def _set_target_ra(self, value: str) -> str:
        """
        Set the GOTO target Right Ascension.

        Accepts two LX200 formats:
          - High precision: HH:MM:SS (hours, minutes, seconds)
          - Low precision:  HH:MM.T  (hours, minutes, tenths of minutes)

        The parsed value is stored in self.target_ra as decimal hours.

        Responds to the :Sr# command.

        Args:
            value: RA string after the "Sr" prefix, e.g. "12:30:00" or "12:30.5".

        Returns:
            "1#" on success, "0#" on parse error.
        """
        try:
            # Split on ':' and '.' to extract numeric components
            parts = value.replace(':', ' ').replace('.', ' ').split()
            hours = int(parts[0])
            minutes = int(parts[1])
            if len(parts) > 2:
                if '.' in value:
                    # Low precision format HH:MM.T - tenths of minutes, convert to seconds
                    seconds = int(parts[2]) * 6
                else:
                    # High precision format HH:MM:SS
                    seconds = float(parts[2])
            else:
                seconds = 0

            # Convert to decimal hours: hours + minutes/60 + seconds/3600
            self.target_ra = hours + minutes/60 + seconds/3600
            return "1#"
        except (ValueError, IndexError):
            return "0#"

    def _set_target_dec(self, value: str) -> str:
        """
        Set the GOTO target Declination.

        Accepts LX200 formats:
          - High precision: sDD*MM:SS or sDD*MM'SS (sign, degrees, arcmin, arcsec)
          - Low precision:  sDD*MM (sign, degrees, arcmin)

        The parsed value is stored in self.target_dec as decimal degrees.

        Responds to the :Sd# command.

        Args:
            value: Dec string after the "Sd" prefix, e.g. "+45*30:00" or "-12*15".

        Returns:
            "1#" on success, "0#" on parse error.
        """
        try:
            # Extract sign (+ or -)
            sign = 1 if value[0] == '+' else -1
            value = value[1:] if value[0] in '+-' else value

            # Split on degree/minute/second separators: *, :, ', "
            parts = re.split(r'[*:\'\"]', value)
            degrees = int(parts[0])
            minutes = int(parts[1]) if len(parts) > 1 else 0
            seconds = float(parts[2]) if len(parts) > 2 else 0

            # Convert to decimal degrees: sign * (deg + min/60 + sec/3600)
            self.target_dec = sign * (degrees + minutes/60 + seconds/3600)
            return "1#"
        except (ValueError, IndexError):
            return "0#"

    # ========== Slew and Motion Commands ==========

    def _slew_to_target(self) -> str:
        """
        Initiate a GOTO slew to the target coordinates.

        Converts the target RA/Dec (set by :Sr#/:Sd#) to Alt/Az coordinates for the
        Dobson mount, checks that the target is above the horizon, and fires the
        on_goto callback to notify the hardware layer.

        Responds to the :MS# command.

        Returns:
            "0#" on success (slew initiated).
            "1Object below horizon#" if the target altitude is negative.
        """
        # Convert target RA/Dec to the mount's native Alt/Az coordinate system
        target_alt, target_az = self._ra_dec_to_alt_az(self.target_ra, self.target_dec)
        self.target_alt = target_alt
        self.target_az = target_az

        # Reject targets below the horizon (negative altitude)
        if target_alt < 0:
            return "1Object below horizon#"

        self.is_slewing = True

        # Notify the hardware layer to begin the physical slew
        if self.on_goto:
            self.on_goto(target_alt, target_az)

        return "0#"

    def _stop_slew(self) -> str:
        """
        Abort all telescope motion (emergency stop).

        Responds to the :Q# command.

        Returns:
            "#" (acknowledged).
        """
        self.is_slewing = False
        if self.on_stop:
            self.on_stop()
        return "#"

    def _move_east(self) -> str:
        """
        Start manual movement toward the east.

        Responds to the :Me# command. Currently a stub for manual directional movement.

        Returns:
            "#" (acknowledged).
        """
        return "#"

    def _move_west(self) -> str:
        """
        Start manual movement toward the west.

        Responds to the :Mw# command. Currently a stub for manual directional movement.

        Returns:
            "#" (acknowledged).
        """
        return "#"

    def _move_north(self) -> str:
        """
        Start manual movement toward the north.

        Responds to the :Mn# command. Currently a stub for manual directional movement.

        Returns:
            "#" (acknowledged).
        """
        return "#"

    def _move_south(self) -> str:
        """
        Start manual movement toward the south.

        Responds to the :Ms# command. Currently a stub for manual directional movement.

        Returns:
            "#" (acknowledged).
        """
        return "#"

    def _stop_move(self) -> str:
        """
        Stop manual movement in a specific direction.

        Responds to the :Qe#, :Qw#, :Qn#, :Qs# commands.

        Returns:
            "#" (acknowledged).
        """
        return "#"

    def _sync_to_target(self) -> str:
        """
        Synchronize the telescope's current position to the target coordinates.

        This tells the mount "you are actually pointing at the target coordinates",
        used for alignment calibration. Updates both RA/Dec and Alt/Az representations,
        with Alt/Az being the authoritative coordinate system for the Dobson mount.

        Responds to the :CM# command.

        Returns:
            "Synced#" to confirm the sync operation.
        """
        # Update RA/Dec coordinates for protocol compatibility
        self.ra_hours = self.target_ra
        self.dec_degrees = self.target_dec
        # Compute Alt/Az from the synced RA/Dec (Alt/Az is the source of truth)
        self.target_alt, self.target_az = self._ra_dec_to_alt_az(self.target_ra, self.target_dec)
        self.alt_degrees = self.target_alt
        self.az_degrees = self.target_az

        # Notify the main application of the position update
        if self.on_position_update:
            self.on_position_update(self.alt_degrees, self.az_degrees)

        return "Synced#"

    # ========== Slew Rate Commands ==========

    def _set_slew_rate_max(self) -> str:
        """Set slew speed to maximum (rate 4). Responds to :RS#."""
        self.slew_speed = 4
        return "#"

    def _set_slew_rate_find(self) -> str:
        """Set slew speed to find (rate 3). Responds to :RM#."""
        self.slew_speed = 3
        return "#"

    def _set_slew_rate_center(self) -> str:
        """Set slew speed to center (rate 2). Responds to :RC#."""
        self.slew_speed = 2
        return "#"

    def _set_slew_rate_guide(self) -> str:
        """Set slew speed to guide (rate 1, slowest). Responds to :RG#."""
        self.slew_speed = 1
        return "#"

    # ========== Site Configuration Commands ==========

    def _set_longitude(self, value: str) -> str:
        """
        Set the observatory longitude.

        Parses the format DDD*MM (degrees, arcminutes separated by '*' or ':').

        Responds to the :Sg# command.

        Args:
            value: Longitude string after the "Sg" prefix, e.g. "002*21".

        Returns:
            "1#" on success, "0#" on parse error.
        """
        try:
            parts = re.split(r'[*:]', value)
            degrees = int(parts[0])
            minutes = int(parts[1]) if len(parts) > 1 else 0
            self.longitude = degrees + minutes/60
            return "1#"
        except (ValueError, IndexError):
            return "0#"

    def _set_latitude(self, value: str) -> str:
        """
        Set the observatory latitude.

        Parses the format sDD*MM (sign, degrees, arcminutes).

        Responds to the :St# command.

        Args:
            value: Latitude string after the "St" prefix, e.g. "+48*51".

        Returns:
            "1#" on success, "0#" on parse error.
        """
        try:
            sign = 1 if value[0] == '+' else -1
            value = value[1:] if value[0] in '+-' else value
            parts = re.split(r'[*:]', value)
            degrees = int(parts[0])
            minutes = int(parts[1]) if len(parts) > 1 else 0
            self.latitude = sign * (degrees + minutes/60)
            return "1#"
        except (ValueError, IndexError):
            return "0#"

    def _set_local_time(self, value: str) -> str:
        """
        Set the local time (stub - uses system clock instead).

        Responds to the :SL# command.

        Args:
            value: Time string (currently ignored).

        Returns:
            "1#" (always succeeds).
        """
        return "1#"

    def _set_date(self, value: str) -> str:
        """
        Set the date (stub - uses system clock instead).

        Responds to the :SC# command.

        Args:
            value: Date string (currently ignored).

        Returns:
            "1Updating Planetary Data#" (standard LX200 success response).
        """
        return "1Updating Planetary Data#"

    def _toggle_precision(self) -> str:
        """
        Toggle between high and low coordinate precision mode.

        High precision includes arcseconds in position responses (HH:MM:SS / DD*MM'SS).
        Low precision only includes arcminutes (HH:MM.T / DD*MM).

        Responds to the :U# command.

        Returns:
            "#" (acknowledged).
        """
        self.high_precision = not self.high_precision
        return "#"

    # ========== Identification / Info Commands ==========

    def _get_product_name(self) -> str:
        """
        Return the telescope product name.

        Responds to the :GVP# command.

        Returns:
            Product name string ending with '#'.
        """
        return "LX200 TrackWise-AltAzPro#"

    def _get_firmware_version(self) -> str:
        """
        Return the firmware version string.

        Responds to the :GVN# command.

        Returns:
            Version string ending with '#'.
        """
        return "1.0#"

    # ========== Focuser Commands (Standard LX200 / OnStep) ==========

    def _focuser_move(self, direction: str) -> str:
        """Start focuser movement in the given direction.

        Responds to :F+# (inward) and :F-# (outward).
        Fires the ``on_focuser_move`` callback if set.

        Returns:
            ``"1#"`` (acknowledged).
        """
        if hasattr(self, 'on_focuser_move') and self.on_focuser_move:
            self.on_focuser_move(direction)
        return "1#"

    def _focuser_stop(self) -> str:
        """Halt focuser movement.

        Responds to the :FQ# command.
        Fires the ``on_focuser_stop`` callback if set.

        Returns:
            ``"1#"`` (acknowledged).
        """
        if hasattr(self, 'on_focuser_stop') and self.on_focuser_stop:
            self.on_focuser_stop()
        return "1#"

    def _focuser_set_speed(self, speed: int) -> str:
        """Set focuser speed (1=slowest .. 4=fastest).

        Responds to :F1# through :F4#.

        Returns:
            ``"1#"`` (acknowledged).
        """
        self._focuser_speed = max(1, min(4, speed))
        return "1#"

    # ========== Derotator Commands (Custom Extension) ==========

    def _rotate_derotator_cw(self, speed_str: str) -> str:
        """
        Rotate the field derotator clockwise at the specified speed.

        The derotator compensates for field rotation inherent to Alt-Az mounts.
        Speed is clamped to the range [0.1, 10.0] degrees/sec.

        Responds to the custom :DR+{speed}# command.

        Args:
            speed_str: Rotation speed as a string, in degrees/sec. Defaults to 1.0
                       if empty or missing.

        Returns:
            "1#" on success, "0#" on parse error.
        """
        try:
            speed = float(speed_str) if speed_str else 1.0
            # Clamp speed to a reasonable range (0.1 to 10 deg/sec)
            speed = max(0.1, min(10.0, speed))
            # Notify the derotator hardware via callback
            if hasattr(self, 'on_derotator_rotate') and self.on_derotator_rotate:
                self.on_derotator_rotate("CW", speed)
            return "1#"  # Success
        except (ValueError, IndexError):
            return "0#"  # Error

    def _rotate_derotator_ccw(self, speed_str: str) -> str:
        """
        Rotate the field derotator counter-clockwise at the specified speed.

        Speed is clamped to the range [0.1, 10.0] degrees/sec.

        Responds to the custom :DR-{speed}# command.

        Args:
            speed_str: Rotation speed as a string, in degrees/sec. Defaults to 1.0
                       if empty or missing.

        Returns:
            "1#" on success, "0#" on parse error.
        """
        try:
            speed = float(speed_str) if speed_str else 1.0
            # Clamp speed to a reasonable range (0.1 to 10 deg/sec)
            speed = max(0.1, min(10.0, speed))
            # Notify the derotator hardware via callback
            if hasattr(self, 'on_derotator_rotate') and self.on_derotator_rotate:
                self.on_derotator_rotate("CCW", speed)
            return "1#"  # Success
        except (ValueError, IndexError):
            return "0#"  # Error

    def _stop_derotator(self) -> str:
        """
        Stop the field derotator.

        Responds to the custom :DRQ# command.

        Returns:
            "1#" (always succeeds).
        """
        if hasattr(self, 'on_derotator_stop') and self.on_derotator_stop:
            self.on_derotator_stop()
        return "1#"  # Success

    def _sync_derotator(self) -> str:
        """
        Synchronize the derotator angle to 0 degrees (reset reference point).

        Responds to the custom :DR0# command.

        Returns:
            "1#" (always succeeds).
        """
        if hasattr(self, 'on_derotator_sync') and self.on_derotator_sync:
            self.on_derotator_sync()
        return "1#"  # Success

    def _get_slew_status(self) -> str:
        """
        Return the current slew status.

        Returns a pipe character '|' while slewing (as a visual activity indicator),
        or an empty response when the slew is complete.

        Responds to the :D# command.

        Returns:
            "|#" if slewing, "#" if idle.
        """
        return "|#" if self.is_slewing else "#"

    def _get_high_precision(self) -> str:
        """
        Return the current coordinate precision mode.

        Responds to the :P# command.

        Returns:
            "HIGH PRECISION#" or "LOW PRECISION#".
        """
        return "HIGH PRECISION#" if self.high_precision else "LOW PRECISION#"

    # ========== Utility / Astronomical Computation Methods ==========

    def _calculate_lst(self) -> float:
        """
        Calculate the Local Sidereal Time (LST) for the observer's longitude.

        LST is the fundamental link between the equatorial (RA/Dec) and horizontal
        (Alt/Az) coordinate systems. It represents the Right Ascension currently
        on the observer's meridian.

        The computation follows these steps:
          1. Compute the Julian Date (JD) for the current UTC time.
          2. Compute Julian centuries (T) since J2000.0 epoch (JD 2451545.0).
          3. Compute Greenwich Sidereal Time (GST) using the IAU formula.
          4. Add the observer's longitude to get Local Sidereal Time.

        Returns:
            Local Sidereal Time in decimal hours (0-24).
        """
        now = datetime.now(timezone.utc)
        jd = self._julian_date(now)
        # Julian centuries since the J2000.0 epoch (2000 Jan 1.5 TT)
        t = (jd - 2451545.0) / 36525.0

        # Greenwich Sidereal Time (GST) in degrees, using the IAU formula:
        #   GST = 280.46061837 + 360.98564736629*(JD - 2451545.0)
        #         + 0.000387933*T^2 - T^3/38710000
        # The linear term (360.985...) reflects the ~361 deg/day sidereal rotation rate.
        gst = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + \
              0.000387933 * t**2 - t**3 / 38710000.0
        gst = gst % 360  # Normalize to [0, 360) degrees

        # Local Sidereal Time = GST + observer's east longitude, converted to hours
        lst = (gst + self.longitude) / 15.0  # 15 degrees per hour
        return lst % 24  # Normalize to [0, 24) hours

    def _julian_date(self, dt: datetime) -> float:
        """
        Compute the Julian Date (JD) for a given datetime.

        Uses the standard algorithm for dates in the Gregorian calendar.
        The Julian Date is a continuous day count used in astronomical calculations,
        with JD 0 starting at noon on January 1, 4713 BC.

        Args:
            dt: A datetime object (should be in UTC for astronomical accuracy).

        Returns:
            Julian Date as a float.
        """
        year = dt.year
        month = dt.month
        # Convert time of day to fractional day: hour/24 + minute/1440 + second/86400
        day = dt.day + dt.hour/24 + dt.minute/1440 + dt.second/86400

        # For Jan/Feb, treat as months 13/14 of the previous year (algorithm requirement)
        if month <= 2:
            year -= 1
            month += 12

        # Gregorian calendar correction terms
        a = int(year / 100)
        b = 2 - a + int(a / 4)

        # Standard Julian Date formula
        jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
        return jd

    def _ra_dec_to_alt_az(self, ra_hours: float, dec_deg: float) -> Tuple[float, float]:
        """
        Convert equatorial coordinates (RA/Dec) to horizontal coordinates (Alt/Az).

        This is the core coordinate transformation that maps the sky coordinate system
        used by planetarium software (RA/Dec) to the mount's native system (Alt/Az).

        The conversion steps are:
          1. Compute the Hour Angle: HA = LST - RA (in degrees)
          2. Apply the spherical trigonometry altitude formula:
             sin(Alt) = sin(Lat)*sin(Dec) + cos(Lat)*cos(Dec)*cos(HA)
          3. Apply the azimuth formula using atan2 for correct quadrant:
             Az = atan2(-cos(Dec)*sin(HA),
                         sin(Dec)*cos(Lat) - cos(Dec)*sin(Lat)*cos(HA))
          4. Normalize azimuth to [0, 360) degrees.

        Args:
            ra_hours: Right Ascension in decimal hours (0-24).
            dec_deg: Declination in decimal degrees (-90 to +90).

        Returns:
            Tuple of (altitude_degrees, azimuth_degrees) where:
              - altitude is in degrees (-90 to +90, negative = below horizon)
              - azimuth is in degrees (0-360, 0=North, 90=East)
        """
        lst = self._calculate_lst()
        # Hour Angle = LST - RA, converted from hours to degrees (* 15 deg/hour)
        ha = (lst - ra_hours) * 15

        # Normalize Hour Angle to [-180, +180] degrees for correct azimuth computation
        ha = ha % 360
        if ha > 180:
            ha = ha - 360

        # Convert all angles to radians for trigonometric functions
        lat_rad = math.radians(self.latitude)
        dec_rad = math.radians(dec_deg)
        ha_rad = math.radians(ha)

        # === Altitude calculation ===
        # Formula: sin(Alt) = sin(Lat)*sin(Dec) + cos(Lat)*cos(Dec)*cos(HA)
        # This projects the celestial sphere onto the observer's horizon plane.
        sin_alt = math.sin(lat_rad) * math.sin(dec_rad) + \
                  math.cos(lat_rad) * math.cos(dec_rad) * math.cos(ha_rad)
        sin_alt = max(-1, min(1, sin_alt))  # Clamp to [-1, 1] to avoid asin domain errors
        alt = math.degrees(math.asin(sin_alt))

        # === Azimuth calculation ===
        # Uses the two-argument form for correct quadrant determination:
        #   sin(Az) = -cos(Dec) * sin(HA) / cos(Alt)
        #   cos(Az) = (sin(Dec) - sin(Lat)*sin(Alt)) / (cos(Lat)*cos(Alt))
        cos_alt = math.cos(math.radians(alt))
        if abs(cos_alt) < 1e-10:
            # At zenith or nadir, azimuth is undefined; default to 0
            az = 0.0
        else:
            sin_az = -math.cos(dec_rad) * math.sin(ha_rad) / cos_alt
            cos_az = (math.sin(dec_rad) - math.sin(lat_rad) * math.sin(math.radians(alt))) / \
                     (math.cos(lat_rad) * cos_alt)

            # atan2(sin, cos) gives the angle in the correct quadrant (-pi to +pi)
            az = math.degrees(math.atan2(sin_az, cos_az))

            # Normalize azimuth to [0, 360) degrees
            az = az % 360
            if az < 0:
                az += 360

        return alt, az

    def set_position(self, ra_hours: float, dec_degrees: float):
        """
        Update the telescope's current position using RA/Dec coordinates.

        This is a compatibility method for setting the position in equatorial coordinates.
        The RA/Dec values are stored and then converted to Alt/Az (the mount's native
        coordinate system) using the current observer location and sidereal time.

        The on_position_update callback is fired with the computed Alt/Az values.

        Args:
            ra_hours: Right Ascension in decimal hours (will be wrapped to 0-24).
            dec_degrees: Declination in decimal degrees (will be clamped to -90 to +90).
        """
        self.ra_hours = ra_hours % 24
        self.dec_degrees = max(-90, min(90, dec_degrees))
        # Compute Alt/Az from RA/Dec (Alt/Az is the authoritative coordinate system)
        self.alt_degrees, self.az_degrees = self._ra_dec_to_alt_az(self.ra_hours, self.dec_degrees)

        if self.on_position_update:
            self.on_position_update(self.alt_degrees, self.az_degrees)

    def update_slew_progress(self, progress: float = 0.1):
        """
        Incrementally update the telescope position during a GOTO slew.

        This should be called periodically (e.g. by a timer) to simulate smooth
        movement toward the target. Each call moves the position a fraction
        (given by `progress`) of the remaining distance to the target.

        When the position is within 0.001 of the target in both RA and Dec,
        the slew is considered complete and is_slewing is set to False.

        Args:
            progress: Fraction of remaining distance to move per call (0.0 to 1.0).
                      Default 0.1 means 10% of remaining distance per call,
                      producing an exponential ease-in to the target.
        """
        if not self.is_slewing:
            return

        # Compute remaining distance to target in RA and Dec
        ra_diff = self.target_ra - self.ra_hours
        dec_diff = self.target_dec - self.dec_degrees

        # Check if we've arrived (within threshold of 0.001 in both axes)
        if abs(ra_diff) < 0.001 and abs(dec_diff) < 0.001:
            self.ra_hours = self.target_ra
            self.dec_degrees = self.target_dec
            self.is_slewing = False
        else:
            # Apply fractional movement toward target (exponential approach)
            self.ra_hours += ra_diff * progress
            self.dec_degrees += dec_diff * progress

        # Recompute Alt/Az from the updated RA/Dec position
        self.alt_degrees, self.az_degrees = self._ra_dec_to_alt_az(self.ra_hours, self.dec_degrees)

        if self.on_position_update:
            self.on_position_update(self.alt_degrees, self.az_degrees)

    # ========== Custom Tracking Rate Commands (SXTR / SXTD) ==========

    def _set_tracking_rate_ra(self, value: str) -> str:
        """
        Set the RA tracking rate offset for real-time drift correction.

        Used by realtime_tracking.py to send continuous correction velocities to
        compensate for the non-uniform tracking inherent to Alt-Az mounts. The rate
        is specified in arcseconds per sidereal second and may be negative.

        The on_tracking_rate_change callback is fired with both RA and Dec rates,
        allowing the tracking system to apply simultaneous two-axis corrections.

        Responds to the custom :SXTR,value# command.

        Args:
            value: Rate offset string in arcsec/sec (may include leading comma from
                   the SXTR, prefix). Clamped to [-3600, +3600] arcsec/sec.

        Returns:
            "1#" on success, "0#" on parse error.
        """
        try:
            # Strip any leading comma left over from the "SXTR," prefix parsing
            value = value.lstrip(',').strip()
            rate = float(value)

            # Clamp to reasonable bounds (-3600 to +3600 arcsec/sec = +/-1 deg/sec)
            rate = max(-3600.0, min(3600.0, rate))

            self.tracking_rate_ra = rate

            # Notify the tracking system of the updated rates
            if self.on_tracking_rate_change:
                self.on_tracking_rate_change(self.tracking_rate_ra, self.tracking_rate_dec)

            return "1#"
        except ValueError:
            return "0#"

    def _set_tracking_rate_dec(self, value: str) -> str:
        """
        Set the Dec tracking rate offset for real-time drift correction.

        Companion to _set_tracking_rate_ra. Together, these two commands allow the
        realtime_tracking module to continuously adjust the mount's motion to keep
        a celestial object centered despite Alt-Az tracking errors.

        Responds to the custom :SXTD,value# command.

        Args:
            value: Rate offset string in arcsec/sec (may include leading comma from
                   the SXTD, prefix). Clamped to [-3600, +3600] arcsec/sec.

        Returns:
            "1#" on success, "0#" on parse error.
        """
        try:
            value = value.lstrip(',').strip()
            rate = float(value)

            # Clamp to reasonable bounds
            rate = max(-3600.0, min(3600.0, rate))

            self.tracking_rate_dec = rate

            if self.on_tracking_rate_change:
                self.on_tracking_rate_change(self.tracking_rate_ra, self.tracking_rate_dec)

            return "1#"
        except ValueError:
            return "0#"

    def _get_tracking_rate_ra(self) -> str:
        """
        Return the current RA tracking rate offset.

        Responds to the :GXTR# command.

        Returns:
            Current RA rate in arcsec/sec with 4 decimal places, ending with '#'.
        """
        return f"{self.tracking_rate_ra:.4f}#"

    def _get_tracking_rate_dec(self) -> str:
        """
        Return the current Dec tracking rate offset.

        Responds to the :GXTD# command.

        Returns:
            Current Dec rate in arcsec/sec with 4 decimal places, ending with '#'.
        """
        return f"{self.tracking_rate_dec:.4f}#"

    def _get_altitude_precise(self) -> str:
        """
        Return the current altitude with high precision.

        Computes Alt/Az from the current RA/Dec, updates the cached Alt/Az values,
        and returns altitude in the format sDD*MM.mm (degrees and fractional arcminutes).

        Responds to the custom :GAL# command.

        Returns:
            High-precision altitude string ending with '#'.
        """
        # Recompute Alt/Az from RA/Dec to ensure consistency
        alt, az = self._ra_dec_to_alt_az(self.ra_hours, self.dec_degrees)
        self.alt_degrees = alt
        self.az_degrees = az

        sign = '+' if alt >= 0 else '-'
        alt_abs = abs(alt)
        degrees = int(alt_abs)
        minutes = (alt_abs - degrees) * 60
        return f"{sign}{degrees:02d}*{minutes:05.2f}#"

    def _get_azimuth_precise(self) -> str:
        """
        Return the current azimuth with high precision.

        Computes Alt/Az from the current RA/Dec, updates the cached Alt/Az values,
        and returns azimuth in the format DDD*MM.mm (degrees and fractional arcminutes).

        Responds to the custom :GAZ# command.

        Returns:
            High-precision azimuth string ending with '#'.
        """
        alt, az = self._ra_dec_to_alt_az(self.ra_hours, self.dec_degrees)
        self.alt_degrees = alt
        self.az_degrees = az

        degrees = int(az)
        minutes = (az - degrees) * 60
        return f"{degrees:03d}*{minutes:05.2f}#"

    def _get_extended_info(self) -> str:
        """
        Return comprehensive telescope state as a comma-separated string.

        Aggregates RA, Dec, Alt, Az, local time, date, latitude, and longitude into
        a single CSV response. Useful for dashboard or monitoring applications.

        Responds to the :GX# command.

        Returns:
            CSV string of all position and site data, ending with '#'.
            Format: RA,Dec,Alt,Az,Time,Date,Lat,Lon#
        """
        # Recompute Alt/Az from RA/Dec to ensure consistency
        alt, az = self._ra_dec_to_alt_az(self.ra_hours, self.dec_degrees)
        self.alt_degrees = alt
        self.az_degrees = az

        # Collect formatted values from individual query methods (strip trailing '#')
        ra_str = self._get_ra().strip('#')
        dec_str = self._get_dec().strip('#')
        alt_str = self._get_altitude().strip('#')
        az_str = self._get_azimuth().strip('#')
        time_str = self._get_local_time().strip('#')
        date_str = self._get_date().strip('#')
        lat_str = self._get_latitude().strip('#')
        lon_str = self._get_longitude().strip('#')

        # Return all values as comma-separated fields
        return f"{ra_str},{dec_str},{alt_str},{az_str},{time_str},{date_str},{lat_str},{lon_str}#"

    def set_position_altaz(self, alt: float, az: float):
        """
        Update the telescope's current position using Alt-Az coordinates.

        This is the preferred method for setting position on a Dobson mount, since
        Alt/Az is the native coordinate system. The RA/Dec values are automatically
        computed from Alt/Az for protocol compatibility with planetarium software.

        The on_position_update callback is fired with the Alt/Az values.

        Args:
            alt: Altitude in degrees (will be clamped to -90 to +90).
            az: Azimuth in degrees (will be wrapped to 0-360).
        """
        self.alt_degrees = max(-90, min(90, alt))
        self.az_degrees = az % 360

        # Derive RA/Dec from Alt/Az for internal protocol compatibility
        self.ra_hours, self.dec_degrees = self._alt_az_to_ra_dec(alt, az)

        if self.on_position_update:
            self.on_position_update(self.alt_degrees, self.az_degrees)

    def _alt_az_to_ra_dec(self, alt: float, az: float) -> Tuple[float, float]:
        """
        Convert horizontal coordinates (Alt/Az) to equatorial coordinates (RA/Dec).

        This is the inverse of _ra_dec_to_alt_az. For a Dobson (Alt-Az) mount, the
        physical encoder positions give Alt/Az directly, and this method derives the
        corresponding RA/Dec for reporting to planetarium software.

        The conversion steps are:
          1. Compute Declination from:
             sin(Dec) = sin(Lat)*sin(Alt) + cos(Lat)*cos(Alt)*cos(Az)
          2. Compute Hour Angle from:
             HA = atan2(sin(Az), cos(Az)*sin(Lat) - tan(Alt)*cos(Lat))
          3. Convert Hour Angle to RA:
             RA = LST - HA

        Args:
            alt: Altitude in degrees (-90 to +90).
            az: Azimuth in degrees (0-360).

        Returns:
            Tuple of (ra_hours, dec_degrees) where:
              - ra_hours is in decimal hours (0-24)
              - dec_degrees is in decimal degrees (-90 to +90)
        """
        alt_rad = math.radians(alt)
        az_rad = math.radians(az)
        lat_rad = math.radians(self.latitude)

        # === Declination calculation ===
        # Formula: sin(Dec) = sin(Lat)*sin(Alt) + cos(Lat)*cos(Alt)*cos(Az)
        # This is the inverse altitude formula, solving for declination.
        sin_dec = math.sin(lat_rad) * math.sin(alt_rad) + \
                  math.cos(lat_rad) * math.cos(alt_rad) * math.cos(az_rad)
        dec = math.degrees(math.asin(max(-1, min(1, sin_dec))))  # Clamp for numerical safety

        # === Hour Angle calculation ===
        # Standard astronomical formulas:
        #   sin(HA) = -sin(Az) * cos(Alt) / cos(Dec)
        #   cos(HA) = (sin(Alt) - sin(Lat)*sin(Dec)) / (cos(Lat)*cos(Dec))
        # Using atan2(sin, cos) to resolve the correct quadrant.
        dec_rad = math.radians(dec)
        cos_dec = math.cos(dec_rad)

        if abs(cos_dec) < 1e-10:
            # At a celestial pole, HA is undefined; default to 0
            ha = 0.0
        else:
            sin_ha = -math.sin(az_rad) * math.cos(alt_rad) / cos_dec
            cos_ha = (math.sin(alt_rad) - math.sin(lat_rad) * math.sin(dec_rad)) / \
                     (math.cos(lat_rad) * cos_dec)
            ha_rad = math.atan2(sin_ha, cos_ha)
            ha = math.degrees(ha_rad)

        # === Convert Hour Angle to Right Ascension ===
        # RA = LST - HA (both in hours; HA must be converted from degrees to hours)
        lst = self._calculate_lst()
        ra = (lst - ha / 15.0) % 24  # 15 degrees per hour, normalize to [0, 24)

        return ra, dec

    def get_tracking_rates(self) -> Tuple[float, float]:
        """
        Return the current Alt-Az tracking correction rates.

        Returns:
            Tuple of (alt_rate, az_rate) in arcseconds per second.
        """
        return self.tracking_rate_alt, self.tracking_rate_az

    def apply_tracking_correction(self, dt: float):
        """
        Apply incremental tracking corrections based on the current Alt/Az rates.

        This method should be called periodically (e.g. every 100ms) by the tracking
        loop to apply continuous corrections to the telescope position. For an Alt-Az
        mount, corrections are applied directly in Alt and Az axes.

        After adjusting Alt/Az, the RA/Dec values are recomputed for protocol
        compatibility (so that :GR# and :GD# queries return correct values).

        Args:
            dt: Time interval since the last correction, in seconds.
                Used to convert rates (arcsec/sec) to angular displacement.
        """
        if not self.is_tracking:
            return

        # Convert tracking rates from arcsec/sec to degrees, scaled by elapsed time:
        #   displacement_deg = (rate_arcsec_per_sec / 3600) * dt_seconds
        alt_correction = (self.tracking_rate_alt / 3600.0) * dt
        az_correction = (self.tracking_rate_az / 3600.0) * dt

        # Apply corrections to the Alt/Az position, clamping/wrapping as needed
        self.alt_degrees = max(-90, min(90, self.alt_degrees + alt_correction))
        self.az_degrees = (self.az_degrees + az_correction) % 360

        # Recompute RA/Dec from the corrected Alt/Az (for protocol compatibility)
        self.ra_hours, self.dec_degrees = self._alt_az_to_ra_dec(
            self.alt_degrees, self.az_degrees
        )

    def slew_alt_az(self, alt_delta: float, az_delta: float):
        """
        Apply a relative movement to the telescope in Alt/Az coordinates.

        Used for manual nudging or joystick-style control. The deltas are applied
        directly to the current Alt/Az position, and RA/Dec is recomputed for
        protocol compatibility.

        Args:
            alt_delta: Change in altitude in degrees (positive = up toward zenith).
            az_delta: Change in azimuth in degrees (positive = eastward / clockwise).
        """
        # Apply the deltas to the native Alt/Az position, clamping altitude to [0, 90]
        self.alt_degrees = max(0, min(90, self.alt_degrees + alt_delta))
        self.az_degrees = (self.az_degrees + az_delta) % 360

        # Derive RA/Dec from the new Alt/Az for protocol compatibility
        self.ra_hours, self.dec_degrees = self._alt_az_to_ra_dec(
            self.alt_degrees, self.az_degrees
        )

        self.is_slewing = True

        if self.on_position_update:
            self.on_position_update(self.alt_degrees, self.az_degrees)

    def park(self):
        """
        Park the telescope at a safe resting position.

        For an Alt-Az (Dobson) mount, the park position is at the zenith:
        Alt=90 degrees (pointing straight up), Az=0 degrees (north).
        This is a safe position that avoids mechanical interference and is a
        well-defined reference point for startup alignment.

        Fires the on_goto callback to initiate the physical slew to park position.
        """
        self.target_alt = 90.0
        self.target_az = 0.0
        # Compute the corresponding RA/Dec for protocol compatibility
        self.target_ra, self.target_dec = self._alt_az_to_ra_dec(90.0, 0.0)
        self.is_slewing = True

        if self.on_goto:
            self.on_goto(self.target_alt, self.target_az)
