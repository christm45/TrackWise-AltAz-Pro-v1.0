"""
ASTAP-based local (offline) plate solver for the Android telescope controller.

=============================================================================
 Module: local_solver.py
 Location: android app/app/src/main/python/android_bridge/local_solver.py
 Purpose: Bridge between the Python backend and the Kotlin ASTAP solver
=============================================================================

Architecture
------------
This module sits between the Python plate-solving pipeline and the Kotlin
wrappers around the ASTAP CLI binary:

  Python (web_server.py / cloud_solver.py / auto_platesolve.py)
    |
    | calls local_solver.solve(), get_database_status(), etc.
    v
  THIS MODULE (local_solver.py)
    |
    | Chaquopy Java interop: from java import jclass
    v
  AstapSolver.kt          -- Runs astap_cli native binary via ProcessBuilder
  AstapDatabaseManager.kt -- Manages star database download/extraction/lifecycle

ASTAP (Astrometric STAcking Program)
-------------------------------------
ASTAP is a professional-grade plate solver that uses hash-based star pattern
recognition against the Gaia DR3 catalog.  It replaces the previous custom
scipy/numpy-based triangle matcher that was unreliable and slow.

Key advantages:
  - Sub-arcsecond accuracy (vs ~30+ arcsecond for the old solver)
  - Solves in 1-30 seconds (with position hint: ~1-5s)
  - Handles a wide range of FOVs (0.2 to 180 degrees)
  - Works offline (no internet needed after database download)
  - Professional tool used by astrophotographers worldwide

Star Databases
--------------
ASTAP requires star index files (.1476 format) stored locally:

  D05: ~45 MB  (FOV >= 0.6 deg) -- good for most telescopes
  D20: ~170 MB (FOV >= 0.3 deg) -- better for narrow fields
  D50: ~500 MB (FOV >= 0.2 deg) -- best for very narrow fields
  W08: ~15 MB  (FOV >= 20 deg)  -- phone cameras / wide-field

The database must be downloaded once (on first use) and is stored in
the app's external files directory.  The UI provides a download button,
database selection dropdown, and real-time progress bar (via SSE).

SSE Progress Reporting
----------------------
Download progress is exposed via get_download_progress(), which reads
@Volatile fields from AstapDatabaseManager.kt.  The web server streams
these as Server-Sent Events (SSE) to the browser for real-time updates
without WebSocket overhead.  See web_server.py /api/solver/databases/progress.

Thread Safety
-------------
All public functions in this module are thread-safe.  The Kotlin objects
(_solver, _db_manager) handle their own synchronization.  The Python-side
initialization uses a threading.Lock to prevent double-init races.

Editing Notes
-------------
This file lives ONLY in the Android source tree.  It is NOT synced from
the root TelescopeController/ directory (unlike web_server.py, etc.).
All edits should be made directly to this file.
"""

import os
import time
import logging
import threading
from typing import Optional, Tuple

logger = logging.getLogger("local_solver")

# ═══════════════════════════════════════════════════════════════════════
#  ASTAP solver initialization via Chaquopy Java interop
# ═══════════════════════════════════════════════════════════════════════
#
# These module-level singletons hold references to the Kotlin objects.
# They are lazily initialized on first use (via _init_astap()) because:
#   1. The Android Context is not available at import time
#   2. Java interop imports (from java import jclass) only work on Android
#   3. Lazy init avoids startup cost if ASTAP is never used

_solver = None         # AstapSolver Java object (wraps the native binary)
_db_manager = None     # AstapDatabaseManager Java object (manages star databases)
_init_lock = threading.Lock()
_initialized = False

# Last solved FOV from ASTAP (degrees).  Updated after each successful solve.
# Exposed via get_last_solved_fov() so the web UI can display it and detect
# FOV mismatches (configured vs actual).
_last_solved_fov = 0.0


def _init_astap():
    """Initialize the ASTAP solver and database manager via Java interop.

    Called lazily on first use.  Uses Chaquopy's ``jclass()`` to instantiate
    the Kotlin wrapper classes (AstapSolver, AstapDatabaseManager) with the
    Android application context.

    The initialization is guarded by a threading.Lock to prevent races when
    multiple threads (e.g., Flask request handlers) try to use the solver
    simultaneously.

    Returns:
        True if initialization succeeded and the ASTAP binary is available.
        False if not on Android, binary not found, or any error occurred.
    """
    global _solver, _db_manager, _initialized

    with _init_lock:
        if _initialized:
            return _solver is not None

        try:
            from java import jclass

            # Get the Android application context via ActivityThread.
            # This works because Chaquopy runs inside the app's process,
            # so ActivityThread.currentApplication() returns our Application.
            activity_thread = jclass("android.app.ActivityThread")
            context = activity_thread.currentApplication()

            if context is None:
                logger.error("Cannot get Android application context")
                _initialized = True
                return False

            # Instantiate the Kotlin solver and database manager
            AstapSolver = jclass("com.telescopecontroller.solver.AstapSolver")
            AstapDbMgr = jclass("com.telescopecontroller.solver.AstapDatabaseManager")

            _solver = AstapSolver(context)
            _db_manager = AstapDbMgr(context)

            # Verify the ASTAP binary exists in nativeLibraryDir
            if AstapSolver.isAvailable(context):
                logger.info("ASTAP solver initialized (binary found)")
            else:
                logger.warning("ASTAP binary not found in native library dir")
                _solver = None

            _initialized = True
            return _solver is not None

        except ImportError:
            # Not running on Android (desktop tests) — Java interop unavailable
            logger.info("Not running on Android (no Java interop)")
            _initialized = True
            return False
        except Exception as e:
            logger.error(f"Failed to initialize ASTAP solver: {e}")
            _initialized = True
            return False


# ═══════════════════════════════════════════════════════════════════════
#  Database management API (exposed to web UI via Flask endpoints)
# ═══════════════════════════════════════════════════════════════════════
#
# These functions are called by the Flask API handlers in web_server.py:
#   GET  /api/solver/databases          -> get_database_status()
#   POST /api/solver/databases/download -> download_database()
#   POST /api/solver/databases/delete   -> delete_database()
#   GET  /api/solver/databases/progress -> get_download_progress() (SSE)

def get_database_status() -> dict:
    """Get status of all ASTAP star databases.

    Returns a dict with database names as keys, each containing:
      installed: bool     -- Whether the database is fully extracted and usable
      size_mb: int        -- Approximate download size in megabytes
      description: str    -- Human-readable description for the UI
      min_fov: float      -- Minimum supported field of view (degrees)
      max_fov: float      -- Maximum practical field of view (degrees)
      path: str           -- Absolute filesystem path to the database directory

    On desktop (not Android), returns an empty dict.

    Called by: web_server.py api_solver_databases() handler
    """
    if not _init_astap() or _db_manager is None:
        return {}

    try:
        # Convert Java Map to Python dict.
        # Chaquopy does NOT make Java's LinkedHashMap$LinkedKeySet directly
        # iterable in Python, so we must use the Java Iterator interface
        # explicitly (iterator() / hasNext() / next()) instead of a for-loop.
        java_status = _db_manager.getDatabaseStatus()
        result = {}
        key_iter = java_status.keySet().iterator()
        while key_iter.hasNext():
            name = key_iter.next()
            info = java_status.get(name)
            result[str(name)] = {
                "installed": bool(info.get("installed")),
                "size_mb": int(info.get("size_mb")),
                "description": str(info.get("description")),
                "min_fov": float(info.get("min_fov")),
                "max_fov": float(info.get("max_fov")),
                "path": str(info.get("path")),
            }
        return result
    except Exception as e:
        logger.error(f"Error getting database status: {e}")
        return {}


def get_installed_database() -> Optional[str]:
    """Get the name of the first installed database, or None.

    Checks in order: d05 -> d20 -> d50 -> w08 (prefers general-purpose).
    """
    if not _init_astap() or _db_manager is None:
        return None

    try:
        result = _db_manager.getInstalledDatabase()
        return str(result) if result is not None else None
    except Exception as e:
        logger.error(f"Error checking installed database: {e}")
        return None


def is_database_installed(db_name: str = "d05") -> bool:
    """Check if a specific star database is installed and ready to use.

    Checks for the existence of the ready.txt marker file in the database
    directory.  This marker is only written after all .1476 files are
    successfully extracted.
    """
    if not _init_astap() or _db_manager is None:
        return False

    try:
        return bool(_db_manager.isDatabaseInstalled(db_name))
    except Exception as e:
        logger.error(f"Error checking database {db_name}: {e}")
        return False


def download_database(db_name: str = "d05") -> bool:
    """Start downloading a star database (non-blocking).

    Launches a background thread in the Kotlin layer that:
    1. Downloads the .deb archive from SourceForge
    2. Extracts .1476 star index files
    3. Writes a ready.txt marker on success

    Progress can be monitored via get_download_progress() (which feeds
    the SSE endpoint in web_server.py).

    Args:
        db_name: Database to download ("d05", "d20", "d50", or "w08")

    Returns:
        True if the download was started, False on error.
    """
    if not _init_astap() or _db_manager is None:
        logger.error("ASTAP not initialized, cannot download database")
        return False

    try:
        _db_manager.downloadDatabase(db_name)
        logger.info(f"Database download started: {db_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to start database download: {e}")
        return False


def delete_database(db_name: str) -> bool:
    """Delete an installed database to free storage.

    Recursively removes the database directory and all .1476 files.
    The user can re-download later via the UI.
    """
    if not _init_astap() or _db_manager is None:
        return False

    try:
        return bool(_db_manager.deleteDatabase(db_name))
    except Exception as e:
        logger.error(f"Failed to delete database {db_name}: {e}")
        return False


def get_database_path(db_name: str = "d05") -> Optional[str]:
    """Get the absolute filesystem path to a database directory.

    This path is passed to ASTAP's -d flag when running a solve.
    The path exists even if the database isn't installed yet.
    """
    if not _init_astap() or _db_manager is None:
        return None

    try:
        return str(_db_manager.getDatabasePath(db_name))
    except Exception as e:
        logger.error(f"Error getting database path: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
#  Real-time download progress (for SSE streaming)
# ═══════════════════════════════════════════════════════════════════════
#
# The AstapDatabaseManager exposes @Volatile fields that are updated
# by the download background thread.  We read them here for the SSE
# endpoint in web_server.py.
#
# This approach avoids the complexity of implementing a Java interface
# from Python (which would require Chaquopy's static_proxy mechanism).
# Instead, simple field reads via Chaquopy getters are sufficient.

def get_download_progress() -> dict:
    """Get real-time download progress for the SSE endpoint.

    Returns a dict with:
      state: str            -- "idle", "downloading", "extracting", "complete", "error"
      bytes_downloaded: int  -- Bytes downloaded so far
      bytes_total: int       -- Total expected bytes (-1 if unknown)
      extracted_files: int   -- Number of .1476 files extracted so far
      error: str             -- Error message (if state == "error")
      db_name: str           -- Database being downloaded (e.g., "d05")

    This is called by the SSE generator in web_server.py
    (GET /api/solver/databases/progress) every ~500ms.

    On desktop (not Android), returns {"state": "idle"}.
    """
    if not _init_astap() or _db_manager is None:
        return {"state": "idle", "bytes_downloaded": 0, "bytes_total": 0,
                "extracted_files": 0, "error": "", "db_name": ""}

    try:
        return {
            "state": str(_db_manager.getDownloadState()),
            "bytes_downloaded": int(_db_manager.getDownloadProgress()),
            "bytes_total": int(_db_manager.getDownloadTotal()),
            "extracted_files": int(_db_manager.getExtractedFiles()),
            "error": str(_db_manager.getDownloadError()),
            "db_name": str(_db_manager.getDownloadingDbName()),
        }
    except Exception as e:
        logger.error(f"Error reading download progress: {e}")
        return {"state": "error", "bytes_downloaded": 0, "bytes_total": 0,
                "extracted_files": 0, "error": str(e), "db_name": ""}


# ═══════════════════════════════════════════════════════════════════════
#  Public API: main solve function
# ═══════════════════════════════════════════════════════════════════════
#
# The solve() function is the primary entry point.  It is called by:
#   - cloud_solver.py cloud_solve() in "astap" or "auto" modes
#   - Indirectly by auto_platesolve.py and realtime_tracking.py (after
#     monkey-patching in main.py)
#
# Return format: (ra_hours, dec_degrees, solve_time_ms) or None
# This matches the signature expected by the tracking pipeline.

def is_available() -> bool:
    """Check if the ASTAP local solver is available and has a database.

    Both conditions must be met:
    1. The ASTAP CLI binary exists (libastapcli.so in nativeLibraryDir)
    2. At least one star database is installed (has ready.txt marker)

    Called by: cloud_solver.py to decide whether to attempt a local solve
    before falling back to cloud.
    """
    if not _init_astap():
        return False
    if _solver is None:
        return False
    return get_installed_database() is not None


def solve(
    image_path: str,
    hint_ra: Optional[float] = None,
    hint_dec: Optional[float] = None,
    search_radius: float = 10.0,
    fov_deg: float = 5.0,
    data_dir: str = "",
    db_name: str = "",
    timeout_sec: int = 120,
) -> Optional[Tuple[float, float, float]]:
    """Solve an image using the ASTAP plate solver.

    This is the main plate-solving entry point for local (offline) solving.
    It delegates to the Kotlin AstapSolver which runs the native astap_cli
    binary via ProcessBuilder.

    Args:
        image_path:     Absolute path to the image (JPEG, PNG, FITS, TIFF)
        hint_ra:        Approximate RA in hours (0-24).  None = blind solve.
                        Providing a hint dramatically speeds up solving
                        (~2s vs ~30s) by narrowing the search space.
        hint_dec:       Approximate Dec in degrees (-90 to +90).  None = blind.
        search_radius:  Search radius in degrees (used by cloud fallback only)
        fov_deg:        Estimated field of view in degrees (helps solver converge)
        data_dir:       Unused — kept for backward API compatibility
        db_name:        Specific database to use (e.g., "d05").  Empty = auto-select.
        timeout_sec:    Maximum solve time in seconds (default 120, configurable
                        via the solver.timeout config key in the web UI)

    Returns:
        (ra_hours, dec_degrees, solve_time_ms) on success, None on failure.

    Examples:
        # Blind solve (slower, ~10-30 seconds)
        result = solve("/tmp/sky.jpg", fov_deg=2.5)

        # Hinted solve (fast, ~1-5 seconds)
        result = solve("/tmp/sky.jpg", hint_ra=5.57, hint_dec=22.0, fov_deg=2.5)
    """
    if not _init_astap() or _solver is None:
        logger.warning("ASTAP solver not available")
        return None

    start = time.time()

    # Auto-select database if not specified
    # Prefers d05 -> d20 -> d50 -> w08 (see getInstalledDatabase)
    if not db_name:
        detected = get_installed_database()
        if not detected:
            logger.warning("No ASTAP star database installed. "
                          "Download one via the Plate Solver settings.")
            return None
        db_name = detected

    # Get the filesystem path to the database directory
    db_path = get_database_path(db_name)
    if not db_path:
        logger.error(f"Cannot find database path for {db_name}")
        return None

    # Verify the database is complete (ready.txt exists)
    if not is_database_installed(db_name):
        logger.warning(f"Database {db_name} is not installed")
        return None

    # Verify the image file exists
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return None

    # Convert Python hint values to ASTAP conventions
    # ASTAP uses -1 for "no RA hint" and -999 for "no Dec hint"
    hint_ra_hours = hint_ra if hint_ra is not None else -1.0
    hint_dec_deg = hint_dec if hint_dec is not None else -999.0

    logger.info(
        f"ASTAP solve: image={os.path.basename(image_path)} "
        f"db={db_name} fov={fov_deg:.1f}d timeout={timeout_sec}s "
        f"hint=({hint_ra_hours:.2f}h, {hint_dec_deg:.1f}d)"
    )

    try:
        # Call the Kotlin solver via Chaquopy Java interop.
        # AstapSolver.solve() runs the native binary via ProcessBuilder,
        # waits for completion, parses the .wcs output file, and returns
        # a SolveResult data class.
        result = _solver.solve(
            image_path,
            db_path,
            float(fov_deg),
            float(hint_ra_hours),
            float(hint_dec_deg),
            int(timeout_sec)
        )

        elapsed_ms = (time.time() - start) * 1000

        if result is None:
            logger.info(f"ASTAP returned null result after {elapsed_ms:.0f}ms")
            return None

        # Access SolveResult fields via Chaquopy-generated getters
        success = bool(result.getSuccess())
        if not success:
            error_msg = str(result.getErrorMessage())
            logger.info(f"ASTAP solve failed after {elapsed_ms:.0f}ms: {error_msg}")
            return None

        global _last_solved_fov

        ra_hours = float(result.getRaHours())
        dec_deg = float(result.getDecDeg())
        fov = float(result.getFovDeg())
        rotation = float(result.getRotation())
        solve_time = float(result.getSolveTimeMs())

        # Store the actual solved FOV for diagnostics.
        # This allows the UI to compare configured vs actual FOV and
        # warn the user if they have the wrong database or optics config.
        if fov > 0:
            _last_solved_fov = fov

        logger.info(
            f"ASTAP solve OK: RA={ra_hours:.4f}h Dec={dec_deg:.3f}d "
            f"FOV={fov:.2f}d rot={rotation:.1f}d ({solve_time:.0f}ms)"
        )

        return (ra_hours, dec_deg, solve_time)

    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        logger.error(f"ASTAP solve error after {elapsed_ms:.0f}ms: {e}")
        return None


def get_last_solved_fov() -> float:
    """Return the FOV (degrees) from the most recent successful ASTAP solve.

    Returns 0.0 if no solve has succeeded yet.  The web UI uses this to:
    - Display the actual FOV next to the configured/estimated FOV
    - Warn if the configured optics don't match the actual image scale
    - Suggest the correct star database based on the real FOV
    """
    return _last_solved_fov


def recommend_database(fov_deg: float) -> str:
    """Recommend the best ASTAP star database for a given FOV.

    Delegates to AstapDatabaseManager.recommendDatabase() in Kotlin:
        FOV >= 20 deg  -> w08 (wide-field, phone cameras)
        FOV >= 0.6 deg -> d05 (most telescopes)
        FOV >= 0.3 deg -> d20 (narrow-field)
        FOV < 0.3 deg  -> d50 (very narrow, planetary)

    Falls back to a Python implementation if Java interop is unavailable.
    """
    if _init_astap() and _db_manager is not None:
        try:
            from java import jclass
            AstapDbMgr = jclass("com.telescopecontroller.solver.AstapDatabaseManager")
            return str(AstapDbMgr.recommendDatabase(float(fov_deg)))
        except Exception:
            pass

    # Python fallback (same logic as Kotlin)
    if fov_deg >= 20.0:
        return "w08"
    elif fov_deg >= 0.6:
        return "d05"
    elif fov_deg >= 0.3:
        return "d20"
    else:
        return "d50"
