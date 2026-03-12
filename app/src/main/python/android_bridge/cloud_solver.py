"""
Plate-solving dispatcher for the Android telescope controller.

=============================================================================
 Module: cloud_solver.py
 Location: android app/app/src/main/python/android_bridge/cloud_solver.py
 Purpose: Dispatch plate-solve requests to ASTAP (local) or Astrometry.net
          (cloud) based on user configuration, with automatic fallback.
=============================================================================

Solver Modes
------------
The user selects a mode in the web UI (Plate Solver card, Location tab):

  "auto" (default):  Try ASTAP local solver first.  If it fails or is
                     unavailable, fall back to Astrometry.net cloud solver.
                     Best for most users — works offline with ASTAP when a
                     star database is installed, and degrades gracefully.

  "astap":           ASTAP local solver ONLY.  No internet access needed.
                     Fails immediately if ASTAP binary or star database
                     is missing.  Best for field use without cellular.

  "cloud":           Astrometry.net cloud solver ONLY.  Requires internet
                     (routed through cellular network on Android, since the
                     phone's WiFi is connected to the telescope mount).
                     Best as a fallback or for users without storage space
                     for star databases.

  "local" (legacy):  Automatically normalized to "astap".  This was the old
                     name for the local solver before the ASTAP rewrite.

Cloud Solver: nova.astrometry.net
---------------------------------
The cloud solver uploads the image to nova.astrometry.net's free API, which
does the heavy computation on their servers and returns the solved coordinates.

API docs: https://nova.astrometry.net/api/

Advantages:
  - No local star database needed (saves 15-500 MB of storage)
  - Works for any FOV and sky position
  - Free (anonymous submissions allowed, API key optional for faster priority)

Disadvantages:
  - Requires internet (cellular) — ~30-60s solve time
  - Unreliable in remote locations with poor cellular coverage
  - Rate-limited (especially without an API key)

Network Routing (Android)
-------------------------
On Android, the phone's WiFi is connected to the telescope mount's WiFi
hotspot (e.g., OnStep Access Point at 192.168.0.1), which has NO internet.
All HTTP calls to nova.astrometry.net are routed through the cellular
(4G/5G) network via network_bridge.py -> CellularHttpClient.kt.

This routing is transparent to this module — the _cellular_get/post helper
functions handle the network selection automatically.

Solver Timeout Configuration
-----------------------------
The solve timeout (in seconds) is stored in the config as `solver.timeout`
and can be adjusted in the web UI.  It applies to both ASTAP local solves
and cloud solve total wait time.  Default: 120 seconds.

Editing Notes
-------------
This file lives ONLY in the Android source tree.  It is NOT synced from
the root TelescopeController/ directory.  All edits should be made here.
"""

import os
import time
import json
import logging
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger("cloud_solver")

# Try to import requests (used for cloud solving on desktop and as fallback)
try:
    import requests
except ImportError:
    requests = None
    logger.warning("requests library not available -- cloud solving disabled")

# ═══════════════════════════════════════════════════════════════════════
#  Cellular network helpers
# ═══════════════════════════════════════════════════════════════════════
#
# On Android, HTTP requests must be routed through the cellular network
# because the WiFi interface is connected to the telescope mount (no internet).
#
# These helpers try the cellular path first (via CellularHttpClient.kt),
# then fall back to standard requests (which works on desktop or if
# cellular routing is unavailable).


def _is_android() -> bool:
    """Check if running on Android (Chaquopy environment)."""
    return os.environ.get("TELESCOPE_PLATFORM") == "android"


def _cellular_get(url: str, timeout_s: int = 10) -> Optional[str]:
    """HTTP GET preferring cellular network on Android.

    On Android: Routes through CellularHttpClient.kt which binds the
    request to the cellular network interface (bypassing the mount WiFi).

    On desktop: Falls back to requests.get() (standard networking).

    Returns:
        Response body as string, or None on failure.
    """
    if _is_android():
        try:
            from android_bridge.network_bridge import cellular_get, is_cellular_available
            if is_cellular_available():
                result = cellular_get(url, timeout_ms=timeout_s * 1000)
                if result is not None:
                    logger.debug("CellularGET %s -> %d bytes", url.split("?")[0], len(result))
                    return result
                logger.warning("Cellular GET returned None for %s", url)
        except Exception as e:
            logger.warning("Cellular GET failed, falling back to requests: %s", e)

    # Fallback to standard requests library
    if requests is not None:
        r = requests.get(url, timeout=timeout_s)
        return r.text
    return None


def _cellular_post_form(url: str, form_data: dict, timeout_s: int = 15) -> Optional[str]:
    """HTTP POST with form-urlencoded body, preferring cellular on Android.

    Used for the Astrometry.net login endpoint (simple key-value form data).

    Returns:
        Response body as string, or None on failure.
    """
    if _is_android():
        try:
            from android_bridge.network_bridge import cellular_post, is_cellular_available
            if is_cellular_available():
                result = cellular_post(url, form_data, timeout_ms=timeout_s * 1000)
                if result is not None:
                    logger.debug("CellularPOST %s -> %d bytes", url.split("?")[0], len(result))
                    return result
                logger.warning("Cellular POST returned None for %s", url)
        except Exception as e:
            logger.warning("Cellular POST failed, falling back to requests: %s", e)

    if requests is not None:
        r = requests.post(url, data=form_data, timeout=timeout_s)
        return r.text
    return None


def _cellular_post_multipart(
    url: str,
    form_data: dict,
    file_path: str,
    file_field: str = "file",
    timeout_s: int = 30,
) -> Optional[str]:
    """HTTP POST with multipart/form-data (file upload), preferring cellular.

    Used for uploading sky images to Astrometry.net for plate solving.
    The image file is sent as a multipart form field.

    Returns:
        Response body as string, or None on failure.
    """
    if _is_android():
        try:
            from android_bridge.network_bridge import cellular_post_multipart, is_cellular_available
            if is_cellular_available():
                result = cellular_post_multipart(
                    url, form_data, file_path,
                    file_field=file_field,
                    file_mime="application/octet-stream",
                    timeout_ms=timeout_s * 1000,
                )
                if result is not None:
                    logger.debug("CellularMultipart %s -> %d bytes", url.split("?")[0], len(result))
                    return result
                logger.warning("Cellular multipart POST returned None for %s", url)
        except Exception as e:
            logger.warning("Cellular multipart POST failed, falling back to requests: %s", e)

    if requests is not None:
        with open(file_path, "rb") as f:
            r = requests.post(
                url,
                data=form_data,
                files={file_field: (os.path.basename(file_path), f)},
                timeout=timeout_s,
            )
        return r.text
    return None


# ═══════════════════════════════════════════════════════════════════════
#  Cloud solver configuration constants
# ═══════════════════════════════════════════════════════════════════════

NOVA_API_URL = "https://nova.astrometry.net/api/"
DEFAULT_TIMEOUT = 60       # Max seconds to wait for a cloud solve
POLL_INTERVAL = 3          # Seconds between status polls (cloud solver)
MAX_FIELD_SIZE = 180.0     # Accept any field size from the solver
DOWNSCALE_FACTOR = 2       # Shrink image before uploading (saves bandwidth)


@dataclass
class CloudSolveResult:
    """Result from a cloud plate-solve attempt.

    Fields:
        success:      True if the solve produced valid coordinates
        ra_hours:     Right Ascension in hours (0-24)
        dec_degrees:  Declination in degrees (-90 to +90)
        solve_time_ms: Wall-clock time for the entire solve process
        field_w:      Solved field width in degrees
        field_h:      Solved field height in degrees
        error:        Human-readable error message (if success=False)
        source:       "cloud" for Astrometry.net, "offline" if network unavailable
    """
    success: bool
    ra_hours: float = 0.0
    dec_degrees: float = 0.0
    solve_time_ms: float = 0.0
    field_w: float = 0.0
    field_h: float = 0.0
    error: str = ""
    source: str = "cloud"


class CloudPlateSolver:
    """Plate solver client for the nova.astrometry.net API.

    Implements the full Astrometry.net API workflow:
    1. Login (get session key, cached for reuse)
    2. Upload image (with optional RA/Dec hint and search radius)
    3. Wait for job creation (poll submission status)
    4. Wait for job completion (poll job status)
    5. Retrieve WCS calibration (RA, Dec, field size)

    The API is free for anonymous use, but an API key provides:
    - Faster queue priority
    - Higher rate limits
    - Access to your solve history on the website

    Get an API key at: https://nova.astrometry.net/api_help

    Usage:
        solver = CloudPlateSolver(api_key="your_key_here")
        result = solver.solve("/path/to/image.jpg", hint_ra=12.5, hint_dec=45.0)
        if result.success:
            print(f"Solved: RA={result.ra_hours}h Dec={result.dec_degrees}d")
    """

    def __init__(self, api_key: str = "", timeout: int = DEFAULT_TIMEOUT):
        self.api_key = api_key
        self.timeout = timeout
        self.session_key: Optional[str] = None
        self._lock = threading.Lock()
        self._online = True  # Assume online until proven otherwise

    # ── Public API ──────────────────────────────────────────────────────

    def solve(
        self,
        image_path: str,
        hint_ra: Optional[float] = None,
        hint_dec: Optional[float] = None,
        search_radius: float = 10.0,
    ) -> CloudSolveResult:
        """Solve an image using the Astrometry.net cloud API.

        Args:
            image_path:    Path to the image file (PNG, JPEG, or FITS)
            hint_ra:       Optional RA hint in hours (0-24).  Speeds up solving.
            hint_dec:      Optional Dec hint in degrees (-90 to +90).
            search_radius: Search radius in degrees around the hint position.

        Returns:
            CloudSolveResult with coordinates on success, error info on failure.
        """
        if requests is None and not _is_android():
            return CloudSolveResult(
                success=False,
                error="requests library not available and not on Android"
            )

        if not self._online:
            return CloudSolveResult(
                success=False,
                error="Device is offline",
                source="offline"
            )

        start_time = time.time()

        try:
            # Step 1: Get or reuse a session key (authentication)
            session = self._get_session()
            if not session:
                return CloudSolveResult(
                    success=False,
                    error="Failed to obtain API session"
                )

            # Step 2: Upload the image for solving
            submission_id = self._upload_image(
                session, image_path, hint_ra, hint_dec, search_radius
            )
            if not submission_id:
                return CloudSolveResult(
                    success=False,
                    error="Image upload failed"
                )

            # Step 3: Wait for the server to create a job and solve it
            job_id = self._wait_for_job(submission_id)
            if not job_id:
                return CloudSolveResult(
                    success=False,
                    error="Solve timed out or failed",
                    solve_time_ms=(time.time() - start_time) * 1000
                )

            # Step 4: Retrieve the astrometric calibration (RA/Dec/field size)
            result = self._get_calibration(job_id)
            result.solve_time_ms = (time.time() - start_time) * 1000
            return result

        except Exception as e:
            # Detect network errors to switch to offline mode
            if requests is not None and isinstance(e, requests.ConnectionError):
                self._online = False
                logger.warning("Network unreachable -- switching to offline mode")
                return CloudSolveResult(
                    success=False,
                    error="Network unreachable",
                    source="offline"
                )
            logger.error(f"Cloud solve error: {e}")
            return CloudSolveResult(
                success=False,
                error=str(e),
                solve_time_ms=(time.time() - start_time) * 1000
            )

    def check_online(self) -> bool:
        """Probe whether the Astrometry.net API is reachable.

        Updates the internal _online flag.  Used to detect when cellular
        connectivity is restored after a period of being offline.
        """
        try:
            result = _cellular_get(NOVA_API_URL, timeout_s=5)
            self._online = result is not None
        except Exception:
            self._online = False
        return self._online

    # ── Internal API calls ──────────────────────────────────────────────

    def _get_session(self) -> Optional[str]:
        """Get or reuse an API session key.

        Session keys are cached for reuse across solves.  If no API key
        is configured, an anonymous session is obtained (works but may
        have lower priority in the solve queue).
        """
        with self._lock:
            if self.session_key:
                return self.session_key

        payload = {"apikey": self.api_key} if self.api_key else {}
        resp_text = _cellular_post_form(
            NOVA_API_URL + "login",
            {"request-json": json.dumps(payload)},
            timeout_s=15,
        )

        if resp_text is None:
            logger.error("Login request failed (no response)")
            return None

        data = json.loads(resp_text)

        if data.get("status") == "success":
            self.session_key = data["session"]
            logger.info("Cloud solver session obtained")
            return self.session_key
        else:
            logger.error(f"Login failed: {data}")
            return None

    def _upload_image(
        self,
        session: str,
        image_path: str,
        hint_ra: Optional[float],
        hint_dec: Optional[float],
        search_radius: float,
    ) -> Optional[int]:
        """Upload an image for plate solving.  Returns submission ID.

        The submission data includes privacy settings (no public sharing)
        and optional position hints.  Hints dramatically speed up solving
        because the server can narrow its search to a small sky area.
        """
        submit_data: dict = {
            "session": session,
            "allow_commercial_use": "n",
            "allow_modifications": "n",
            "publicly_visible": "n",
        }

        # Add position hint if available (speeds up solving from ~60s to ~10s)
        if hint_ra is not None and hint_dec is not None:
            submit_data["center_ra"] = hint_ra * 15.0   # hours -> degrees
            submit_data["center_dec"] = hint_dec
            submit_data["radius"] = search_radius

        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return None

        file_size = os.path.getsize(image_path)
        logger.info(f"Uploading {image_path} ({file_size / 1024:.0f} KB)")

        resp_text = _cellular_post_multipart(
            NOVA_API_URL + "upload",
            {"request-json": json.dumps(submit_data)},
            image_path,
            file_field="file",
            timeout_s=30,
        )

        if resp_text is None:
            logger.error("Upload request failed (no response)")
            return None

        data = json.loads(resp_text)
        if data.get("status") == "success":
            sub_id = data["subid"]
            logger.info(f"Uploaded, submission ID: {sub_id}")
            return sub_id
        else:
            logger.error(f"Upload failed: {data}")
            return None

    def _wait_for_job(self, submission_id: int) -> Optional[int]:
        """Poll until the submission produces a job, then wait for solve completion.

        The Astrometry.net workflow is:
        1. Upload → creates a "submission"
        2. Submission gets assigned to a "job" (may take seconds to minutes)
        3. Job runs the solver (typically 10-60 seconds)
        4. Job status becomes "success" or "failure"

        We poll both the submission (for job creation) and the job (for
        completion) with POLL_INTERVAL second intervals.
        """
        deadline = time.time() + self.timeout
        job_id = None

        # Phase 1: Wait for job creation
        while time.time() < deadline:
            resp_text = _cellular_get(
                f"{NOVA_API_URL}submissions/{submission_id}",
                timeout_s=10,
            )
            if resp_text is None:
                time.sleep(POLL_INTERVAL)
                continue

            data = json.loads(resp_text)
            jobs = data.get("jobs", [])
            if jobs and jobs[0] is not None:
                job_id = jobs[0]
                break

            time.sleep(POLL_INTERVAL)

        if not job_id:
            logger.warning(f"No job produced for submission {submission_id}")
            return None

        # Phase 2: Wait for job completion
        while time.time() < deadline:
            resp_text = _cellular_get(
                f"{NOVA_API_URL}jobs/{job_id}",
                timeout_s=10,
            )
            if resp_text is None:
                time.sleep(POLL_INTERVAL)
                continue

            data = json.loads(resp_text)
            status = data.get("status")

            if status == "success":
                logger.info(f"Job {job_id} solved")
                return job_id
            elif status == "failure":
                logger.warning(f"Job {job_id} failed to solve")
                return None

            time.sleep(POLL_INTERVAL)

        logger.warning(f"Job {job_id} timed out after {self.timeout}s")
        return None

    def _get_calibration(self, job_id: int) -> CloudSolveResult:
        """Retrieve the WCS calibration from a completed job.

        The calibration contains:
        - ra, dec: center coordinates in degrees
        - width_arcdeg, height_arcdeg: field size
        - orientation: position angle
        - pixscale: arcsec/pixel
        """
        resp_text = _cellular_get(
            f"{NOVA_API_URL}jobs/{job_id}/calibration/",
            timeout_s=10,
        )

        if resp_text is None:
            return CloudSolveResult(success=False, error="Failed to get calibration")

        data = json.loads(resp_text)

        ra_deg = data.get("ra", 0.0)
        dec_deg = data.get("dec", 0.0)
        ra_hours = ra_deg / 15.0   # degrees -> hours

        return CloudSolveResult(
            success=True,
            ra_hours=ra_hours,
            dec_degrees=dec_deg,
            field_w=data.get("width_arcdeg", 0.0),
            field_h=data.get("height_arcdeg", 0.0),
            source="cloud"
        )


# ═══════════════════════════════════════════════════════════════════════
#  Module-level singleton and configuration reader
# ═══════════════════════════════════════════════════════════════════════
#
# The cloud solver is a stateful object (maintains session key, online
# status).  A module-level singleton avoids re-creating it on every call.
# It's lazily initialized with the API key from the config file.

_solver: Optional[CloudPlateSolver] = None


def get_solver() -> CloudPlateSolver:
    """Get or create the global cloud solver instance.

    Reads the API key from telescope_config.json on first call.
    The API key can be set in the web UI: Location tab → Plate Solver → API Key.
    """
    global _solver
    if _solver is None:
        api_key = ""
        data_dir = os.environ.get("TELESCOPE_DATA_DIR", "")
        config_path = os.path.join(data_dir, "telescope_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                api_key = cfg.get("solver", {}).get("cloud_api_key", "")
            except Exception:
                pass
        _solver = CloudPlateSolver(api_key=api_key)
    return _solver


def _get_solver_mode() -> str:
    """Read the solver mode from the config file.

    Valid modes: "auto", "astap", "cloud"
    Legacy "local" is normalized to "astap" for backward compatibility.

    Config key: solver.mode in telescope_config.json
    """
    data_dir = os.environ.get("TELESCOPE_DATA_DIR", "")
    config_path = os.path.join(data_dir, "telescope_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            mode = cfg.get("solver", {}).get("mode", "auto")
            if mode == "local":
                mode = "astap"
            return mode
        except Exception:
            pass
    return "auto"


def _get_solver_timeout() -> int:
    """Read the solver timeout from the config file.

    Config key: solver.timeout in telescope_config.json
    Default: 120 seconds

    This timeout applies to:
    - ASTAP local solve (passed to AstapSolver.solve() timeoutSec parameter)
    - Cloud solve total wait time (CloudPlateSolver.timeout)
    """
    data_dir = os.environ.get("TELESCOPE_DATA_DIR", "")
    config_path = os.path.join(data_dir, "telescope_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            return int(cfg.get("solver", {}).get("timeout", 120))
        except Exception:
            pass
    return 120


def _get_solver_fov() -> float:
    """Calculate FOV from configured focal length and sensor width.

    Reads solver.focal_length_mm and solver.sensor_width_mm from the
    config file.  If both are set (> 0), calculates:

        FOV = 2 * atan(sensor_width / (2 * focal_length))  [degrees]

    Returns 0.0 if the optics are not configured (caller should use a
    sensible default like 5.0 degrees).
    """
    import math
    data_dir = os.environ.get("TELESCOPE_DATA_DIR", "")
    config_path = os.path.join(data_dir, "telescope_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            solver = cfg.get("solver", {})
            fl = float(solver.get("focal_length_mm", 0))
            sw = float(solver.get("sensor_width_mm", 0))
            if fl > 0 and sw > 0:
                fov = 2.0 * math.degrees(math.atan(sw / (2.0 * fl)))
                return round(fov, 4)
        except Exception:
            pass
    return 0.0


# ═══════════════════════════════════════════════════════════════════════
#  Main plate-solving dispatcher
# ═══════════════════════════════════════════════════════════════════════
#
# cloud_solve() is the primary entry point for ALL plate solving on Android.
# It is called by the monkey-patched _solve_image and solve_fast methods
# (see main.py _patch_plate_solvers).
#
# The function respects the user's solver mode preference and routes
# accordingly:
#
#   "auto" mode:
#     1. Try ASTAP local solver (fast, offline)
#     2. If ASTAP fails → fall back to cloud solver (slow, needs internet)
#
#   "astap" mode:
#     1. Try ASTAP local solver only
#     2. If ASTAP fails → return None (no fallback)
#
#   "cloud" mode:
#     1. Try cloud solver only (skip ASTAP entirely)


def cloud_solve(
    image_path: str,
    hint_ra: Optional[float] = None,
    hint_dec: Optional[float] = None,
    search_radius: float = 10.0,
    fov_deg: float = 0.0,
) -> Optional[Tuple[float, float, float]]:
    """Main plate-solving dispatcher.

    Routes the solve request to the appropriate backend(s) based on
    the user's configured solver mode.

    Args:
        image_path:    Path to the sky image (JPEG, PNG, or FITS)
        hint_ra:       Optional RA hint in hours (0-24)
        hint_dec:      Optional Dec hint in degrees (-90 to +90)
        search_radius: Search radius in degrees (for cloud solver)
        fov_deg:       Estimated field of view in degrees (for ASTAP).
                       0 = auto-detect from config (focal length + sensor).
                       Falls back to 5.0 if optics not configured.

    Returns:
        (ra_hours, dec_degrees, solve_time_ms) on success, None on failure.
        This tuple format matches FastPlateSolver.solve_fast() and is
        expected by the tracking pipeline (realtime_tracking.py).
    """
    mode = _get_solver_mode()
    timeout = _get_solver_timeout()

    # Resolve FOV: caller override > config calculation > default 5.0
    if fov_deg <= 0:
        fov_deg = _get_solver_fov()
    if fov_deg <= 0:
        fov_deg = 5.0

    # ── ASTAP local solver (primary for "astap" and "auto" modes) ─────
    if mode in ("astap", "auto"):
        try:
            from android_bridge.local_solver import solve as astap_solve, is_available
            if is_available():
                result = astap_solve(
                    image_path,
                    hint_ra=hint_ra,
                    hint_dec=hint_dec,
                    search_radius=search_radius,
                    fov_deg=fov_deg,
                    timeout_sec=timeout,
                )
                if result is not None:
                    logger.info(
                        f"ASTAP local solve OK: RA={result[0]:.4f}h "
                        f"Dec={result[1]:.2f}d ({result[2]:.0f}ms)"
                    )
                    return result
                logger.info("ASTAP local solve returned no result")
            else:
                logger.info("ASTAP solver not available (no binary or no database)")
        except Exception as e:
            logger.warning(f"ASTAP solver error: {e}")

        # In astap-only mode, don't fall back to cloud
        if mode == "astap":
            logger.info("ASTAP-only mode: no fallback to cloud")
            return None

    # ── Cloud solve (Astrometry.net) ──────────────────────────────────
    solver = get_solver()
    solver.timeout = timeout  # Apply user-configured timeout
    result = solver.solve(image_path, hint_ra, hint_dec, search_radius)

    if result.success:
        return (result.ra_hours, result.dec_degrees, result.solve_time_ms)
    else:
        logger.info(f"Cloud solve failed: {result.error}")
        return None
