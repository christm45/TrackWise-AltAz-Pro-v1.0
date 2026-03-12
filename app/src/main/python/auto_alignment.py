"""
Automatic multi-star alignment procedure.

Selects the brightest stars visible above the horizon, slews to each one,
plate-solves to determine the actual pointing, re-centers until confirmed
on target, then syncs the mount.  After all stars are aligned the mount's
pointing model is significantly improved.

Usage (from web/headless):
    aligner = AutoAlignment(app)
    aligner.start(num_stars=6)   # runs in background thread
    aligner.abort()              # cancel at any time
    aligner.get_status()         # poll progress
"""

import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from telescope_logger import get_logger

_logger = get_logger(__name__)


# ===================================================================
# Alignment star database (with magnitudes)
# ===================================================================

@dataclass
class AlignmentStar:
    """A star candidate for alignment."""
    name: str
    ra_hours: float
    dec_degrees: float
    magnitude: float        # visual magnitude (lower = brighter)
    alt: float = 0.0        # current altitude (computed at runtime)
    az: float = 0.0         # current azimuth  (computed at runtime)


def load_alignment_stars(catalog_dir: str = "catalogs") -> List[AlignmentStar]:
    """
    Load named bright stars with their magnitudes from stars.h.

    The stars.h format is:
        { Has_name, Cons, BayerFlam, Has_subId, Obj_id, Mag, RA_deg, Dec_deg }

    Mag is stored as integer = magnitude * 100 (e.g. 76 = 0.76).
    RA is in degrees (must /15 to get hours).

    Returns:
        List of AlignmentStar sorted by magnitude (brightest first).
    """
    stars_file = os.path.join(catalog_dir, "data", "stars.h")
    if not os.path.exists(stars_file):
        _logger.warning("Stars catalog not found: %s", stars_file)
        return []

    try:
        with open(stars_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        _logger.warning("Error reading stars catalog: %s", e)
        return []

    # Extract names
    names_match = re.search(
        r'Cat_Stars_Names=(.*?);\s*const', content, re.DOTALL
    )
    names: List[str] = []
    if names_match:
        for m in re.finditer(r'"([^"]+)"', names_match.group(1)):
            first_name = m.group(1).split(";")[0].strip()
            if first_name:
                names.append(first_name)

    # Extract star data: Has_name, ..., Mag, RA_deg, Dec_deg
    pattern = (
        r'\{\s*(\d+)\s*,'       # Has_name
        r'\s*[^,]*,'             # Cons
        r'\s*[^,]*,'             # BayerFlam
        r'\s*[^,]*,'             # Has_subId
        r'\s*[^,]*,'             # Obj_id
        r'\s*(\d+)\s*,'          # Mag (int, *100)
        r'\s*([-\d.]+)\s*,'      # RA (degrees)
        r'\s*([-\d.]+)\s*\}'     # Dec (degrees)
    )
    matches = re.findall(pattern, content)

    stars: List[AlignmentStar] = []
    name_idx = 0
    for has_name, mag_str, ra_str, dec_str in matches:
        if int(has_name) == 1 and name_idx < len(names):
            mag = int(mag_str) / 100.0
            ra_hours = float(ra_str) / 15.0
            dec_deg = float(dec_str)
            stars.append(AlignmentStar(
                name=names[name_idx],
                ra_hours=ra_hours,
                dec_degrees=dec_deg,
                magnitude=mag,
            ))
            name_idx += 1

    # Sort by brightness (lowest magnitude = brightest)
    stars.sort(key=lambda s: s.magnitude)
    _logger.info("Loaded %d named alignment stars (brightest: %s mag=%.2f)",
                 len(stars), stars[0].name if stars else "?",
                 stars[0].magnitude if stars else 0)
    return stars


def select_alignment_stars(
    all_stars: List[AlignmentStar],
    num_stars: int,
    protocol,
    min_alt: float = 15.0,
) -> List[AlignmentStar]:
    """
    Select the best alignment stars from those currently visible.

    Enhanced strategy for optimal sky coverage (both azimuth AND altitude):

    1. Compute current Alt/Az for every star.
    2. Filter to stars above min_alt degrees.
    3. Divide the sky into azimuth sectors AND altitude bands:
       - Azimuth sectors: up to min(num_stars, 12) sectors for
         uniform azimuthal coverage.
       - Altitude bands: 3 bands (low 15-35, mid 35-60, high 60-90)
         to ensure stars at different elevations are included.
         This is critical for flexure model calibration, as tube
         flexure changes significantly with altitude.
    4. Pick the brightest star from each (sector, band) cell,
       prioritising diversity.
    5. Fill remaining slots with overall brightest unused stars,
       preferring stars that improve altitude diversity.

    Why altitude diversity matters:
    - Tube flexure (gravitational bending) is strongly altitude-dependent.
    - Atmospheric refraction varies with altitude.
    - Alignment models that only sample mid-altitudes will have poor
      corrections near the horizon and at the zenith.

    Args:
        all_stars: Full list of alignment stars.
        num_stars: How many to select (3/6/9/12/16/20/24).
        protocol: LX200Protocol instance (has _ra_dec_to_alt_az).
        min_alt: Minimum altitude in degrees.

    Returns:
        Selected stars with alt/az populated, ordered for efficient slew.
    """
    # Compute current alt/az for all stars
    visible: List[AlignmentStar] = []
    for s in all_stars:
        try:
            alt, az = protocol._ra_dec_to_alt_az(s.ra_hours, s.dec_degrees)
            if alt >= min_alt:
                s.alt = alt
                s.az = az
                visible.append(s)
        except Exception:
            continue

    if not visible:
        _logger.warning("No stars above %.0f deg altitude", min_alt)
        return []

    _logger.info("Visible alignment candidates: %d (above %.0f deg)",
                 len(visible), min_alt)

    if len(visible) <= num_stars:
        visible.sort(key=lambda s: s.az)
        return visible

    # ---- Phase 1: Azimuth + altitude grid selection ----
    # Altitude bands: low, mid, high (for flexure coverage)
    alt_bands = [(min_alt, 35.0), (35.0, 60.0), (60.0, 90.1)]
    num_sectors = min(num_stars, 12)
    sector_size = 360.0 / num_sectors

    # Build a 2D grid: (sector, band) -> list of stars
    grid: Dict[Tuple[int, int], List[AlignmentStar]] = {}
    for ai in range(len(alt_bands)):
        for si in range(num_sectors):
            grid[(si, ai)] = []

    for s in visible:
        si = int(s.az / sector_size) % num_sectors
        ai = 0
        for bi, (lo, hi) in enumerate(alt_bands):
            if lo <= s.alt < hi:
                ai = bi
                break
        grid[(si, ai)].append(s)

    # Pick brightest from each cell, distributing across alt bands first
    selected: List[AlignmentStar] = []
    used = set()

    # First pass: one per altitude band (ensures altitude diversity)
    for ai in range(len(alt_bands)):
        for si in range(num_sectors):
            if len(selected) >= num_stars:
                break
            cell = sorted(grid[(si, ai)], key=lambda s: s.magnitude)
            for star in cell:
                if star.name not in used:
                    selected.append(star)
                    used.add(star.name)
                    break
        if len(selected) >= num_stars:
            break

    # Second pass: fill from underrepresented altitude bands
    alt_counts = [0, 0, 0]
    for s in selected:
        for bi, (lo, hi) in enumerate(alt_bands):
            if lo <= s.alt < hi:
                alt_counts[bi] += 1
                break

    # Sort remaining by brightness, prefer underrepresented bands
    remaining = sorted(
        [s for s in visible if s.name not in used],
        key=lambda s: s.magnitude,
    )
    for s in remaining:
        if len(selected) >= num_stars:
            break
        # Determine which band this star falls in
        band = 0
        for bi, (lo, hi) in enumerate(alt_bands):
            if lo <= s.alt < hi:
                band = bi
                break
        # Prefer stars from bands with fewer representatives
        min_band_count = min(alt_counts)
        if alt_counts[band] <= min_band_count + 1:
            selected.append(s)
            used.add(s.name)
            alt_counts[band] += 1

    # Final fill with any remaining bright stars
    for s in remaining:
        if len(selected) >= num_stars:
            break
        if s.name not in used:
            selected.append(s)

    # Sort for efficient slewing: nearest-neighbour path
    selected = _optimize_slew_order(selected)

    _logger.info("Selected %d stars: alt bands [%d low, %d mid, %d high]",
                 len(selected), alt_counts[0], alt_counts[1], alt_counts[2])
    return selected


def _optimize_slew_order(stars: List[AlignmentStar]) -> List[AlignmentStar]:
    """
    Reorder stars for minimal total slew distance (nearest-neighbour heuristic).

    Instead of sorting by azimuth alone (which ignores altitude), this uses a
    simple greedy nearest-neighbour approach on (alt, az) angular distance.
    This reduces total slew time, especially for large star counts.
    """
    if len(stars) <= 2:
        return stars

    remaining = list(stars)
    ordered = [remaining.pop(0)]

    while remaining:
        last = ordered[-1]
        # Find nearest unvisited star (angular distance in alt/az space)
        best_idx = 0
        best_dist = float('inf')
        for i, s in enumerate(remaining):
            # Simple Euclidean on (alt, az) -- good enough for slew optimization
            dalt = s.alt - last.alt
            daz = s.az - last.az
            # Handle azimuth wrap-around
            if daz > 180:
                daz -= 360
            elif daz < -180:
                daz += 360
            dist = dalt * dalt + daz * daz
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        ordered.append(remaining.pop(best_idx))

    return ordered


# ===================================================================
# Alignment status
# ===================================================================

@dataclass
class AlignmentStatus:
    """Current state of the alignment procedure."""
    running: bool = False
    mode: str = "auto"          # "auto" (plate-solver) or "manual" (user-guided)
    phase: str = "idle"         # idle, homing, aligning, complete, aborted, error
    num_stars: int = 0
    current_star_index: int = 0
    current_star_name: str = ""
    stars_completed: int = 0
    stars_total: int = 0
    current_step: str = ""      # goto, solving, recentering, confirming, syncing
                                # manual mode adds: waiting_for_user
    attempt: int = 0
    max_attempts: int = 5
    last_solve_error: float = 0.0   # arcsec from target
    message: str = ""
    star_list: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    # Manual mode: target coordinates for the star the user must center
    manual_target_ra: str = ""      # HH:MM:SS display string
    manual_target_dec: str = ""     # +DD*MM:SS display string
    manual_target_alt: float = 0.0  # current altitude for user reference
    manual_target_az: float = 0.0   # current azimuth for user reference
    waiting_for_user: bool = False   # True when paused, waiting for user action

    def to_dict(self) -> Dict:
        return {
            "running": self.running,
            "mode": self.mode,
            "phase": self.phase,
            "num_stars": self.num_stars,
            "current_star_index": self.current_star_index,
            "current_star_name": self.current_star_name,
            "stars_completed": self.stars_completed,
            "stars_total": self.stars_total,
            "current_step": self.current_step,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "last_solve_error": round(self.last_solve_error, 1),
            "message": self.message,
            "star_list": self.star_list,
            "errors": self.errors[-10:],  # last 10 errors
            "manual_target_ra": self.manual_target_ra,
            "manual_target_dec": self.manual_target_dec,
            "manual_target_alt": round(self.manual_target_alt, 1),
            "manual_target_az": round(self.manual_target_az, 1),
            "waiting_for_user": self.waiting_for_user,
        }


# ===================================================================
# Auto-alignment engine
# ===================================================================

class AutoAlignment:
    """
    Automatic multi-star alignment procedure.

    The procedure:
    1. Home the telescope (goto home position).
    2. Select N brightest visible stars spread across the sky.
    3. For each star:
       a. GoTo the star's RA/Dec coordinates.
       b. Wait for slew to complete.
       c. Plate-solve to determine actual pointing.
       d. If error > threshold: compute correction, re-GoTo, repeat (max 5 attempts).
       e. Once confirmed on target (error < threshold): sync the mount.
    4. After all stars: alignment complete.

    The alignment syncs improve the mount's internal pointing model,
    reducing GoTo errors across the sky.
    """

    # Acceptable pointing error to confirm star is centered (arcsec)
    CONFIRM_THRESHOLD_ARCSEC = 60.0    # 1 arcminute
    # Max correction attempts per star before giving up
    MAX_ATTEMPTS = 5
    # Mechanical vibration settle time after mount stops slewing (seconds)
    SLEW_SETTLE_TIME = 2.0
    # Maximum time to wait for a slew to complete (seconds)
    SLEW_TIMEOUT = 120.0
    # Seconds to wait before assuming the slew never started (very short
    # slew that completed before the first poll, or :D# not supported)
    SLEW_START_GRACE = 5.0
    # Wait time between solve attempts (seconds)
    SOLVE_WAIT_TIME = 2.0
    # Timeout waiting for a plate solve result (seconds)
    SOLVE_TIMEOUT = 30.0

    def __init__(self, app):
        """
        Args:
            app: The telescope app instance (RealTimeTelescopeApp or
                 HeadlessTelescopeApp). Must have:
                 - protocol (LX200Protocol with _ra_dec_to_alt_az, etc.)
                 - _goto_altaz_from_radec(ra_str, dec_str)
                 - _home_telescope()
                 - auto_solver (AutoPlateSolver or None)
        """
        self.app = app
        self._thread: Optional[threading.Thread] = None
        self._abort_event = threading.Event()
        self._status = AlignmentStatus()
        self._lock = threading.Lock()
        self._all_stars: List[AlignmentStar] = []
        self._solve_result = None
        self._solve_event = threading.Event()

        # Manual-mode user interaction events
        self._user_sync_event = threading.Event()     # user pressed "Sync"
        self._user_recenter_event = threading.Event()  # user pressed "Re-center"
        self._user_skip_event = threading.Event()      # user pressed "Skip"

    # --- Public API ---

    def start(self, num_stars: int = 6, mode: str = "auto") -> bool:
        """
        Start the alignment procedure.

        Args:
            num_stars: Number of alignment stars (3/6/9/12/16/20/24).
            mode: "auto" for plate-solver alignment (fully automatic),
                  "manual" for user-guided alignment (user centers each
                  star visually through the eyepiece and confirms sync).

        Returns:
            True if started, False if already running.
        """
        if self._status.running:
            return False

        if num_stars not in (3, 6, 9, 12, 16, 20, 24):
            num_stars = max(3, min(24, num_stars))

        if mode not in ("auto", "manual"):
            mode = "auto"

        self._abort_event.clear()
        self._user_sync_event.clear()
        self._user_recenter_event.clear()
        self._user_skip_event.clear()

        mode_label = "automatic (plate-solver)" if mode == "auto" else "manual (user-guided)"
        self._status = AlignmentStatus(
            running=True,
            mode=mode,
            phase="starting",
            num_stars=num_stars,
            stars_total=num_stars,
            message=f"Starting {num_stars}-star {mode_label} alignment...",
        )

        self._thread = threading.Thread(
            target=self._run_alignment,
            args=(num_stars,),
            daemon=True,
            name="AutoAlignment",
        )
        self._thread.start()
        _logger.info("Alignment started: mode=%s, stars=%d", mode, num_stars)
        return True

    def abort(self):
        """Abort the alignment procedure."""
        if self._status.running:
            self._abort_event.set()
            self._update_status(phase="aborted", running=False,
                                message="Alignment aborted by user")
            _logger.warning("Auto-alignment aborted")

    def get_status(self) -> Dict:
        """Return current alignment status as a dict."""
        with self._lock:
            return self._status.to_dict()

    def is_running(self) -> bool:
        return self._status.running

    # --- Manual-mode user actions ---

    def manual_confirm_sync(self) -> bool:
        """
        User confirms the star is centered -- proceed with sync.

        Called from the UI when the user has visually centered the
        alignment star in the eyepiece/camera and is ready to sync.

        Returns:
            True if the event was accepted (alignment is waiting).
        """
        if not self._status.running or self._status.mode != "manual":
            return False
        if not self._status.waiting_for_user:
            return False
        self._user_sync_event.set()
        _logger.info("Manual alignment: user confirmed sync for %s",
                     self._status.current_star_name)
        return True

    def manual_recenter(self) -> bool:
        """
        User requests a re-center slew before syncing.

        The telescope will GoTo the target star again so the user can
        fine-tune centering.  After the slew, the alignment loop will
        pause and wait for the user again.

        Returns:
            True if the event was accepted.
        """
        if not self._status.running or self._status.mode != "manual":
            return False
        if not self._status.waiting_for_user:
            return False
        self._user_recenter_event.set()
        _logger.info("Manual alignment: user requested re-center for %s",
                     self._status.current_star_name)
        return True

    def manual_skip_star(self) -> bool:
        """
        User wants to skip the current alignment star.

        Returns:
            True if the event was accepted.
        """
        if not self._status.running or self._status.mode != "manual":
            return False
        if not self._status.waiting_for_user:
            return False
        self._user_skip_event.set()
        _logger.info("Manual alignment: user skipped %s",
                     self._status.current_star_name)
        return True

    # --- Internal ---

    def _update_status(self, **kwargs):
        """Thread-safe status update."""
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self._status, k):
                    setattr(self._status, k, v)

    def _log(self, msg: str, level: str = "info"):
        """Log and update status message."""
        self._update_status(message=msg)
        getattr(_logger, level, _logger.info)(msg)

    def _is_aborted(self) -> bool:
        return self._abort_event.is_set()

    def _format_ra(self, ra_hours: float) -> str:
        """Format RA hours as HH:MM:SS string."""
        h = int(ra_hours)
        m_frac = (ra_hours - h) * 60
        m = int(m_frac)
        s = (m_frac - m) * 60
        return f"{h:02d}:{m:02d}:{s:04.1f}"

    def _format_dec(self, dec_deg: float) -> str:
        """Format Dec degrees as +DD*MM:SS string."""
        sign = "+" if dec_deg >= 0 else "-"
        d_abs = abs(dec_deg)
        d = int(d_abs)
        m_frac = (d_abs - d) * 60
        m = int(m_frac)
        s = (m_frac - m) * 60
        return f"{sign}{d:02d}*{m:02d}:{s:04.1f}"

    def _angular_separation(self, ra1: float, dec1: float,
                            ra2: float, dec2: float) -> float:
        """
        Compute angular separation between two RA/Dec positions.

        Args:
            ra1, ra2: RA in decimal hours.
            dec1, dec2: Dec in decimal degrees.

        Returns:
            Separation in arcseconds.
        """
        # Convert to radians
        ra1_rad = math.radians(ra1 * 15)
        ra2_rad = math.radians(ra2 * 15)
        dec1_rad = math.radians(dec1)
        dec2_rad = math.radians(dec2)

        # Haversine formula
        cos_sep = (math.sin(dec1_rad) * math.sin(dec2_rad) +
                   math.cos(dec1_rad) * math.cos(dec2_rad) *
                   math.cos(ra1_rad - ra2_rad))
        cos_sep = max(-1.0, min(1.0, cos_sep))
        sep_rad = math.acos(cos_sep)
        return math.degrees(sep_rad) * 3600  # arcseconds

    def _wait_for_slew_complete(self, timeout: Optional[float] = None):
        """Poll the mount's slewing flag until the slew finishes.

        The telescope bridge's background loop already queries the mount
        every 0.5 s via the LX200 :D# command (distance-bar indicator)
        and the OnStep :GU# command (global status flags), plus a
        position-delta fallback.  The result is written into
        ``app.protocol.is_slewing``.

        This method watches that flag through three phases:

        1. **Start grace** -- wait up to ``SLEW_START_GRACE`` seconds for
           ``is_slewing`` to become True.  If it never does, the slew was
           either so short it completed between polls, or the mount does
           not support :D#.  Either way, we proceed.
        2. **Slewing** -- once we see ``is_slewing == True``, we wait for
           it to return to False (mount has arrived).
        3. **Settle** -- a fixed ``SLEW_SETTLE_TIME`` pause for mechanical
           vibrations to damp before plate solving.
        """
        if timeout is None:
            timeout = self.SLEW_TIMEOUT

        start = time.time()
        poll_interval = 0.5

        # Brief delay so the bridge's next poll cycle picks up the slew
        time.sleep(1.0)

        slewing_seen = False
        while time.time() - start < timeout:
            if self._abort_event.is_set():
                return

            is_slewing = getattr(self.app.protocol, 'is_slewing', False)

            if is_slewing:
                if not slewing_seen:
                    self._log("Mount is slewing...")
                slewing_seen = True
            elif slewing_seen:
                # Was slewing, now stopped -> slew complete
                self._log("Mount slew complete")
                break
            elif time.time() - start > self.SLEW_START_GRACE:
                # Never detected slewing after grace period -- the slew
                # was either very short or :D# is not supported
                self._log("Slew start not detected (short slew or "
                          ":D# unsupported) -- proceeding")
                break

            time.sleep(poll_interval)
        else:
            self._log(f"Slew timeout after {timeout:.0f}s -- proceeding "
                      "anyway", "warning")

        # Mechanical vibration settle time
        time.sleep(self.SLEW_SETTLE_TIME)

    def _do_goto(self, ra_hours: float, dec_degrees: float):
        """Send GoTo command and wait for the mount to finish slewing."""
        ra_str = self._format_ra(ra_hours)
        dec_str = self._format_dec(dec_degrees)
        try:
            self.app._goto_altaz_from_radec(ra_str, dec_str)
        except Exception as e:
            self._log(f"GoTo error: {e}", "error")
        # Wait for mount to actually finish slewing
        self._wait_for_slew_complete()

    def _do_sync(self, ra_hours: float, dec_degrees: float):
        """Sync the mount to the given RA/Dec position."""
        ra_str = self._format_ra(ra_hours)
        dec_str = self._format_dec(dec_degrees)
        try:
            # Set target coordinates
            self.app.protocol.process_command(f":Sr{ra_str}#")
            self.app.protocol.process_command(f":Sd{dec_str}#")
            # Execute sync
            result = self.app.protocol.process_command(":CM#")
            self._log(f"Sync result: {result}")

            # Also sync on hardware bridge if connected
            bridge = getattr(self.app, '_get_active_bridge', lambda: None)()
            if bridge and hasattr(bridge, 'send_command') and bridge.is_connected:
                bridge.send_command(f":Sr{ra_str}#")
                bridge.send_command(f":Sd{dec_str}#")
                bridge.send_command(":CM#")
        except Exception as e:
            self._log(f"Sync error: {e}", "error")

    def _plate_solve(self) -> Optional[Tuple[float, float]]:
        """
        Trigger a plate solve and wait for result.

        Returns:
            (ra_hours, dec_degrees) if successful, None otherwise.
        """
        solver = getattr(self.app, 'auto_solver', None)
        if solver is None:
            self._log("No plate solver available", "error")
            return None

        # Save original callback, install ours
        original_callback = getattr(solver, 'on_solve_complete', None)
        self._solve_result = None
        self._solve_event.clear()

        def on_solve(result):
            self._solve_result = result
            self._solve_event.set()
            # Also call original
            if original_callback:
                original_callback(result)

        try:
            solver.on_solve_complete = on_solve

            # If solver isn't running, try a single-shot solve
            if hasattr(solver, 'solve_single'):
                solver.solve_single()
            elif hasattr(solver, '_running') and not solver._running:
                # Start solver briefly
                solver.start()
                self._solve_event.wait(timeout=self.SOLVE_TIMEOUT)
                solver.stop()
            else:
                # Solver is already running, just wait for next result
                self._solve_event.wait(timeout=self.SOLVE_TIMEOUT)

        finally:
            solver.on_solve_complete = original_callback

        if self._solve_result and self._solve_result.success:
            return (self._solve_result.ra_hours,
                    self._solve_result.dec_degrees)
        return None

    def _align_one_star(self, star: AlignmentStar, index: int) -> bool:
        """
        Align on a single star: goto, solve, recenter, confirm, sync.

        Returns:
            True if alignment succeeded, False if failed/aborted.
        """
        self._update_status(
            current_star_index=index,
            current_star_name=star.name,
            current_step="goto",
            attempt=0,
        )
        self._log(f"Star {index+1}/{self._status.stars_total}: "
                   f"{star.name} (mag {star.magnitude:.1f}, "
                   f"Alt {star.alt:.0f} Az {star.az:.0f})")

        target_ra = star.ra_hours
        target_dec = star.dec_degrees

        for attempt in range(self.MAX_ATTEMPTS):
            if self._is_aborted():
                return False

            self._update_status(attempt=attempt + 1)

            # Step 1: GoTo target
            self._update_status(current_step="goto")
            self._log(f"  Attempt {attempt+1}: GoTo {star.name}...")
            self._do_goto(target_ra, target_dec)

            if self._is_aborted():
                return False

            # Step 2: Plate solve
            self._update_status(current_step="solving")
            self._log(f"  Plate solving...")
            time.sleep(self.SOLVE_WAIT_TIME)

            solved = self._plate_solve()
            if solved is None:
                err_msg = f"  Plate solve failed on attempt {attempt+1}"
                self._log(err_msg, "warning")
                with self._lock:
                    self._status.errors.append(
                        f"{star.name}: solve failed (attempt {attempt+1})")
                continue

            solved_ra, solved_dec = solved
            error_arcsec = self._angular_separation(
                solved_ra, solved_dec, target_ra, target_dec
            )
            self._update_status(last_solve_error=error_arcsec)
            self._log(f"  Solved: RA={solved_ra:.4f}h Dec={solved_dec:.2f} "
                       f"error={error_arcsec:.0f}\"")

            # Step 3: Check if centered
            if error_arcsec <= self.CONFIRM_THRESHOLD_ARCSEC:
                # Confirm with one more solve
                self._update_status(current_step="confirming")
                self._log(f"  Within threshold, confirming...")
                time.sleep(self.SOLVE_WAIT_TIME)

                confirm = self._plate_solve()
                if confirm:
                    confirm_ra, confirm_dec = confirm
                    confirm_error = self._angular_separation(
                        confirm_ra, confirm_dec, target_ra, target_dec
                    )
                    self._update_status(last_solve_error=confirm_error)

                    if confirm_error <= self.CONFIRM_THRESHOLD_ARCSEC:
                        # Confirmed! Sync the mount
                        self._update_status(current_step="syncing")
                        self._log(f"  Confirmed on {star.name} "
                                   f"(error={confirm_error:.0f}\"). Syncing.")
                        self._do_sync(target_ra, target_dec)
                        return True
                    else:
                        self._log(f"  Confirmation failed "
                                   f"(error={confirm_error:.0f}\"), retrying")
                else:
                    self._log("  Confirmation solve failed, retrying", "warning")

            # Step 4: Recenter -- GoTo again using the offset
            self._update_status(current_step="recentering")
            # Compute correction: we need to move by (target - solved)
            # Apply this as a corrected GoTo
            ra_correction = target_ra - solved_ra
            dec_correction = target_dec - solved_dec
            # Apply correction to current target
            corrected_ra = (target_ra + ra_correction) % 24
            corrected_dec = max(-90, min(90, target_dec + dec_correction))
            self._log(f"  Recentering (dRA={ra_correction*3600:.0f}s "
                       f"dDec={dec_correction*3600:.0f}\")")

            # Use corrected position for next GoTo attempt
            # (we GoTo the corrected position, not the original)
            self._do_goto(corrected_ra, corrected_dec)
            time.sleep(self.SOLVE_WAIT_TIME)

            if self._is_aborted():
                return False

        # Exhausted attempts
        self._log(f"  Failed to center on {star.name} after "
                   f"{self.MAX_ATTEMPTS} attempts", "warning")
        with self._lock:
            self._status.errors.append(
                f"{star.name}: failed after {self.MAX_ATTEMPTS} attempts")
        return False

    def _align_one_star_manual(self, star: AlignmentStar, index: int) -> bool:
        """
        Manual alignment on a single star: goto, wait for user, sync.

        In manual mode the user visually centers the star through the
        eyepiece (or camera live view) and presses "Sync" when ready.
        If not perfectly centered, the user can press "Re-center" to
        slew again, adjust, then sync.

        Flow per star:
        1. GoTo the star's coordinates.
        2. Wait for mount to finish slewing.
        3. Show star info and wait for user action:
           a. "Sync" -> sync the mount to this star's known coordinates,
              mark success, proceed to next star.
           b. "Re-center" -> GoTo the star again (user may have nudged
              the mount or wants a fresh slew), then wait again.
           c. "Skip" -> skip this star (no sync), proceed to next.
        4. Repeat from step 3 until user syncs or skips.

        Returns:
            True if user synced, False if skipped/aborted.
        """
        self._update_status(
            current_star_index=index,
            current_star_name=star.name,
            current_step="goto",
            attempt=0,
            manual_target_ra=self._format_ra(star.ra_hours),
            manual_target_dec=self._format_dec(star.dec_degrees),
            manual_target_alt=star.alt,
            manual_target_az=star.az,
        )
        self._log(f"Star {index+1}/{self._status.stars_total}: "
                   f"{star.name} (mag {star.magnitude:.1f}, "
                   f"Alt {star.alt:.0f} Az {star.az:.0f})")

        target_ra = star.ra_hours
        target_dec = star.dec_degrees

        # Initial GoTo
        self._log(f"  Slewing to {star.name}...")
        self._do_goto(target_ra, target_dec)

        if self._is_aborted():
            return False

        # Enter the user-interaction loop
        recenter_count = 0
        while not self._is_aborted():
            # Clear all user events
            self._user_sync_event.clear()
            self._user_recenter_event.clear()
            self._user_skip_event.clear()

            # Signal that we're waiting for the user
            self._update_status(
                current_step="waiting_for_user",
                waiting_for_user=True,
                message=f"Waiting for user: center {star.name} "
                        f"(Alt {star.alt:.0f} Az {star.az:.0f}) "
                        f"then press Sync, or Re-center to slew again",
            )
            self._log(f"  Waiting for user action on {star.name}...")

            # Wait until the user presses one of the buttons
            # (poll every 0.5s so we can also check abort)
            while not self._is_aborted():
                if self._user_sync_event.is_set():
                    break
                if self._user_recenter_event.is_set():
                    break
                if self._user_skip_event.is_set():
                    break
                time.sleep(0.5)

            self._update_status(waiting_for_user=False)

            if self._is_aborted():
                return False

            # Handle user's choice
            if self._user_skip_event.is_set():
                self._log(f"  User skipped {star.name}")
                return False

            if self._user_recenter_event.is_set():
                recenter_count += 1
                self._update_status(
                    current_step="recentering",
                    attempt=recenter_count,
                )
                self._log(f"  Re-centering on {star.name} "
                           f"(attempt {recenter_count})...")
                self._do_goto(target_ra, target_dec)
                if self._is_aborted():
                    return False
                continue  # back to waiting

            if self._user_sync_event.is_set():
                # User confirmed the star is centered -- sync!
                self._update_status(current_step="syncing")
                self._log(f"  User confirmed {star.name} centered. Syncing.")
                self._do_sync(target_ra, target_dec)
                self._log(f"  Synced on {star.name} successfully")
                return True

        return False

    def _run_alignment(self, num_stars: int):
        """Main alignment thread."""
        try:
            # Load star database
            self._update_status(phase="loading", message="Loading star catalog...")
            self._all_stars = load_alignment_stars()
            if not self._all_stars:
                self._update_status(phase="error", running=False,
                                    message="No alignment stars found in catalog")
                return

            # Step 1: Home the telescope
            if self._is_aborted():
                return
            self._update_status(phase="homing", message="Homing telescope...")
            self._log("Homing telescope...")
            try:
                self.app._home_telescope()
                self._wait_for_slew_complete()
            except Exception as e:
                self._log(f"Home failed: {e}", "warning")

            if self._is_aborted():
                return

            # Step 2: Select alignment stars
            self._update_status(phase="selecting",
                                message=f"Selecting {num_stars} alignment stars...")
            selected = select_alignment_stars(
                self._all_stars, num_stars, self.app.protocol
            )
            if not selected:
                self._update_status(phase="error", running=False,
                                    message="No visible alignment stars")
                return

            actual_count = len(selected)
            self._update_status(
                stars_total=actual_count,
                star_list=[{
                    "name": s.name,
                    "mag": round(s.magnitude, 1),
                    "alt": round(s.alt, 0),
                    "az": round(s.az, 0),
                    "status": "pending",
                } for s in selected],
            )
            self._log(f"Selected {actual_count} alignment stars: "
                       f"{', '.join(s.name for s in selected)}")

            # Step 3: Align on each star
            is_manual = self._status.mode == "manual"
            self._update_status(phase="aligning")
            completed = 0
            for i, star in enumerate(selected):
                if self._is_aborted():
                    return

                if is_manual:
                    success = self._align_one_star_manual(star, i)
                else:
                    success = self._align_one_star(star, i)

                # Update star status in the list
                with self._lock:
                    if i < len(self._status.star_list):
                        self._status.star_list[i]["status"] = (
                            "done" if success else "failed"
                        )

                if success:
                    completed += 1
                    self._update_status(stars_completed=completed)

            # Step 4: Verification pass (auto mode only)
            # In manual mode the user already confirmed each star visually,
            # so an automated verification pass is not meaningful without
            # a plate solver.
            if (not is_manual and not self._is_aborted()
                    and completed >= 3 and len(selected) >= 3):
                self._update_status(phase="verifying",
                                    message="Verification pass...")
                verify_count = min(2, completed)
                for vi in range(verify_count):
                    if self._is_aborted():
                        break
                    star = selected[vi]
                    self._log(f"Verification {vi+1}/{verify_count}: "
                              f"re-checking {star.name}...")
                    self._update_status(current_step="verify_solve",
                                        current_star_name=star.name)
                    self._do_goto(star.ra_hours, star.dec_degrees)
                    time.sleep(self.SOLVE_WAIT_TIME)
                    solved = self._plate_solve()
                    if solved:
                        verify_err = self._angular_separation(
                            solved[0], solved[1],
                            star.ra_hours, star.dec_degrees
                        )
                        self._log(f"  Verification error: {verify_err:.0f}\"")
                        # If the error is worse than 2x threshold, re-sync
                        if verify_err <= self.CONFIRM_THRESHOLD_ARCSEC * 2:
                            self._log(f"  {star.name} verified OK")
                        elif verify_err <= self.CONFIRM_THRESHOLD_ARCSEC * 5:
                            self._log(f"  {star.name} re-syncing (error "
                                      f"{verify_err:.0f}\")")
                            self._do_sync(star.ra_hours, star.dec_degrees)
                        else:
                            self._log(f"  {star.name} verification: large "
                                      f"error {verify_err:.0f}\" (skipping)",
                                      "warning")

            # Step 5: Complete
            if self._is_aborted():
                return

            mode_label = "Manual" if is_manual else "Auto"
            self._update_status(
                phase="complete",
                running=False,
                waiting_for_user=False,
                message=f"{mode_label} alignment complete: {completed}/{actual_count} "
                        f"stars aligned successfully",
            )
            _logger.info("%s alignment complete: %d/%d stars",
                         mode_label, completed, actual_count)

        except Exception as e:
            self._update_status(phase="error", running=False,
                                message=f"Alignment error: {e}")
            _logger.error("Auto-alignment error: %s", e, exc_info=True)
