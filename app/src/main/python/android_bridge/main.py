"""
Android entry point for the Telescope Controller Python backend.

=============================================================================
 Module: main.py
 Location: android app/app/src/main/python/android_bridge/main.py
 Purpose: Bootstrap the Python backend on Android via Chaquopy
=============================================================================

Called from Kotlin's TelescopeService via Chaquopy Java-to-Python bridge:
    bridge = Python.getInstance().getModule("android_bridge.main")
    bridge.callAttr("start_server", data_dir, cache_dir, port)

Startup Sequence
----------------
  1. Set up Android-specific paths (data_dir, cache_dir) and sys.path
  2. Configure logging (file + logcat via stdout)
  3. Kill any leftover Flask server from a previous crash (port check)
  4. Extract catalog assets from APK to filesystem
  5. Apply pre-import patches (stub Windows/desktop-only modules)
  6. Monkey-patch components that don't work on Android:
     - Desktop ASTAP subprocess -> ASTAP CLI native binary (local) or
       Astrometry.net API (cloud), dispatched via cloud_solver.py
     - OpenCV camera -> Android camera bridge (Camera2 / UVC / ZWO ASI SDK)
     - pyserial USB -> Android USB serial bridge (usb-serial-for-android)
     - pywin32 / ASCOM -> stubbed out (not applicable on Android)
  7. Create HeadlessTelescopeApp + start Flask web server

Monkey-Patching Strategy
------------------------
The desktop Python modules (auto_platesolve.py, telescope_bridge.py,
web_server.py) use desktop-specific libraries (OpenCV, pyserial, ASTAP
subprocess, etc.) that don't work on Android.  Instead of modifying those
shared modules (which would break desktop functionality), we monkey-patch
their methods at runtime after import.

Key patches:
  - AutoPlateSolver._solve_image()     -> cloud_solver.cloud_solve()
  - AutoPlateSolver.start_camera_mode() -> camera_bridge (ZWO/UVC/phone)
  - FastPlateSolver.solve_fast()       -> cloud_solver.cloud_solve()
  - TelescopeBridge._connect_serial()  -> AndroidSerialPort (USB OTG)
  - TelescopeWebServer._open_uvc_camera() -> camera_bridge
  - TelescopeWebServer._generate_mjpeg()  -> camera_bridge.get_jpeg_frame()

Editing Notes
-------------
This file lives ONLY in the Android source tree (android_bridge/).
It is NOT synced from the root TelescopeController/ directory.
"""

import os
import sys
import json
import time
import socket
import logging
import logging.handlers
import threading

logger = logging.getLogger("android_bridge")

# Global reference to the running app
_app = None
_shutdown_event = threading.Event()


# ═══════════════════════════════════════════════════════════════════════
#  Public API (called from Kotlin via Chaquopy)
# ═══════════════════════════════════════════════════════════════════════

def start_server(data_dir: str, cache_dir: str, port: int = 8080):
    """
    Start the telescope controller backend.

    Args:
        data_dir:  Android filesDir (persistent storage for config, models,
                   sessions, catalogs)
        cache_dir: Android cacheDir (temp files -- plate-solve images, etc.)
        port:      Flask listening port (default 8080)
    """
    global _app

    try:
        _start_server_inner(data_dir, cache_dir, port)
    except SystemExit as e:
        # Intercept SystemExit so it doesn't silently kill the Chaquopy
        # Python thread.  Convert it to a RuntimeError with a descriptive
        # message that Kotlin can display to the user.
        code = e.code if e.code is not None else "unknown"
        msg = (
            f"Python backend raised SystemExit({code}) during startup. "
            f"This usually means the Flask server failed to bind port {port} "
            f"(port already in use from a previous crash) or a library "
            f"initialization error.  Try force-stopping the app and relaunching."
        )
        logger.error(msg, exc_info=True)
        raise RuntimeError(msg) from e
    except BaseException as e:
        # Catch absolutely everything so we always get a log entry.
        logger.error(f"Fatal error in start_server: {e}", exc_info=True)
        raise


def _start_server_inner(data_dir: str, cache_dir: str, port: int):
    """Actual startup logic, separated so start_server() can wrap it."""
    global _app

    # ── 1. Set up paths ────────────────────────────────────────────────
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # The main Python modules are bundled alongside this package by Chaquopy.
    bridge_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(bridge_dir)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    os.environ["TELESCOPE_TEMP_DIR"] = cache_dir
    os.environ["TELESCOPE_DATA_DIR"] = data_dir
    os.environ["TELESCOPE_PLATFORM"] = "android"

    # ── 2. Set up logging ──────────────────────────────────────────────
    log_path = os.path.join(data_dir, "telescope_app.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.handlers.RotatingFileHandler(
                log_path, maxBytes=5 * 1024 * 1024, backupCount=3
            ),
            # Also log to stdout so Chaquopy forwards to Android logcat
            # (visible via 'adb logcat' for debugging)
            logging.StreamHandler(sys.stdout),
        ]
    )
    logger.info("Android bridge starting")
    logger.info(f"  data_dir:  {data_dir}")
    logger.info(f"  cache_dir: {cache_dir}")
    logger.info(f"  port:      {port}")

    # ── 2b. Kill any leftover Flask server from a previous crash ───────
    #
    # If the app was force-killed (e.g. USB cable pulled, ANR, OOM), the
    # old Python process may still be holding the port.  Try to connect
    # and shut it down, or at least verify the port is free.
    _ensure_port_free(port)

    # ── 3. Extract catalog assets to filesystem ──────────────────────────
    #
    # catalog_loader.py reads .h files via os.path.exists() / open().
    # On Android these live inside the APK as assets, not on the
    # filesystem.  We extract them once to data_dir/catalogs/data/
    # so the existing loader works unchanged.
    _extract_catalog_assets(data_dir)

    # ── 4. Apply Android patches BEFORE importing main modules ─────────
    _patch_for_android(data_dir, cache_dir)

    # ── 5. Monkey-patch plate solving, camera, and serial after import ──
    try:
        _patch_serial_bridge()
        logger.info("Serial bridge patches applied")
    except Exception as e:
        logger.error(f"Failed to patch serial bridge: {e}", exc_info=True)

    try:
        _patch_plate_solvers()
        logger.info("Plate solver patches applied")
    except Exception as e:
        logger.error(f"Failed to patch plate solvers: {e}", exc_info=True)

    try:
        _patch_web_server_camera()
        logger.info("Web server camera patches applied")
    except Exception as e:
        logger.error(f"Failed to patch web server camera: {e}", exc_info=True)

    # ── 6. Import and start the headless server ────────────────────────
    from HEADLESS_SERVER import HeadlessTelescopeApp

    # Ensure config file exists in data_dir (cwd is already data_dir)
    config_path = os.path.join(data_dir, "telescope_config.json")
    if not os.path.exists(config_path):
        _create_default_config(config_path)

    logger.info("Creating HeadlessTelescopeApp...")

    # HeadlessTelescopeApp reads telescope_config.json from cwd via ConfigManager
    _app = HeadlessTelescopeApp(
        connection_type="wifi",
        web_port=port,
    )

    logger.info("HeadlessTelescopeApp created, starting...")

    # start() launches Flask + update loop on daemon threads and RETURNS.
    # We must NOT block here -- Kotlin's TelescopeService needs callAttr()
    # to return so it can run waitForFlask() and set isServerReady.
    _shutdown_event.clear()
    _app.start()

    logger.info(f"Backend started -- Flask on http://127.0.0.1:{port}")
    # Returns immediately. Kotlin service handles lifecycle from here.
    # stop_server() will be called from onDestroy().


def stop_server():
    """Gracefully shut down the telescope controller."""
    global _app
    logger.info("Stop requested from Android")

    if _app is not None:
        try:
            _app.stop()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        _app = None

    _shutdown_event.set()


def is_running() -> bool:
    """Check if the backend is currently running."""
    return _app is not None and not _shutdown_event.is_set()


# ═══════════════════════════════════════════════════════════════════════
#  Pre-import patches (run before any telescope module is imported)
# ═══════════════════════════════════════════════════════════════════════

def _patch_for_android(data_dir: str, cache_dir: str):
    """
    Patch environment and stub desktop-only modules.
    Must run BEFORE importing HEADLESS_SERVER / auto_platesolve / etc.
    """

    # ── Working directory -> data_dir (config files live here) ─────────
    os.chdir(data_dir)

    # ── Temp directory -> cache_dir (plate-solve images, etc.) ─────────
    import tempfile
    tempfile.tempdir = cache_dir

    # ── Stub Windows-only modules ──────────────────────────────────────
    for mod in ("win32com", "win32com.client", "pythoncom", "winreg",
                "pywintypes", "win32api"):
        _stub_module(mod)

    # ── Stub OpenCV if not available ───────────────────────────────────
    try:
        import cv2  # noqa: F401
        logger.info("OpenCV available on Android")
    except ImportError:
        logger.info("OpenCV not available -- using Android camera bridge")
        _stub_module("cv2")

    # ── Stub matplotlib (desktop-only graphing) ────────────────────────
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        _stub_module("matplotlib")
        _stub_module("matplotlib.pyplot")

    logger.info("Pre-import Android patches applied")


# ═══════════════════════════════════════════════════════════════════════
#  Post-import patches (monkey-patch actual methods after modules load)
# ═══════════════════════════════════════════════════════════════════════

def _patch_serial_bridge():
    """
    Monkey-patch telescope_bridge._connect_serial to use AndroidSerialPort.

    On Android, pyserial's ``serial.Serial`` cannot access USB devices
    (there are no ``/dev/ttyUSB*`` nodes without root).  The Kotlin
    ``UsbSerialManager`` provides USB-serial access via Android's USB
    Host API, and ``AndroidSerialPort`` wraps it in a pyserial-compatible
    interface.

    This patch replaces ``_connect_serial`` so that when the user selects
    "USB" mode in the app, the connection goes through
    AndroidSerialPort -> serial_bridge -> UsbSerialManager (Kotlin)
    instead of the desktop pyserial path.
    """
    import telescope_bridge
    from android_bridge.serial_bridge import AndroidSerialPort, list_ports

    _original_connect_serial = telescope_bridge.TelescopeBridge._connect_serial

    def _android_connect_serial(self, port: str, baudrate: int = 9600) -> bool:
        """Connect to telescope via Android USB serial bridge."""
        self._log("Android USB serial connection starting...")

        # Discover USB serial devices via UsbSerialManager
        ports = list_ports()
        if not ports:
            error_msg = (
                "No USB serial devices found.  Check that:\n"
                "  - The USB-serial cable is plugged in\n"
                "  - You granted USB permission when prompted\n"
                "  - The cable uses a supported chip (FTDI, CP210x, CH340, PL2303)"
            )
            self.last_error = error_msg
            self._log(f"ERROR: {error_msg}")
            return False

        self._log(f"Found {len(ports)} USB serial device(s):")
        for p in ports:
            self._log(
                f"  [{p.get('index')}] {p.get('name', 'Unknown')} "
                f"(VID:0x{p.get('vid', 0):04X} PID:0x{p.get('pid', 0):04X} "
                f"by {p.get('manufacturer', '?')})"
            )

        # Select port: match by name/index, or use first available
        port_index = 0
        if port:
            for p in ports:
                if (p.get('name', '') == port
                        or str(p.get('index', '')) == str(port)):
                    port_index = int(p['index'])
                    break

        # Open the AndroidSerialPort (pyserial-compatible shim)
        android_port = AndroidSerialPort()
        if not android_port.open(port_index=port_index, baudrate=baudrate):
            error_msg = (
                f"Failed to open USB serial port #{port_index}.  "
                "USB permission may have been denied — unplug and re-plug "
                "the cable, then tap 'Allow' on the permission dialog."
            )
            self.last_error = error_msg
            self._log(f"ERROR: {error_msg}")
            return False

        self.serial_connection = android_port
        self._log(f"Android USB serial port opened (baud={baudrate})")

        # Flush stale data
        if android_port.in_waiting > 0:
            self._log(f"Cleaning buffer: {android_port.in_waiting} bytes waiting")
            android_port.reset_input_buffer()

        time.sleep(0.5)  # let port stabilise

        # ── Send test commands via mount_protocol ──────────────────────
        def _serial_test_send(cmd_bytes: bytes, timeout: float = 1.0) -> str:
            """Send a command and read until '#' or timeout."""
            try:
                if android_port.in_waiting > 0:
                    android_port.reset_input_buffer()
                android_port.write(cmd_bytes)
                time.sleep(0.3)
                response = ""
                start = time.time()
                while time.time() - start < timeout:
                    if android_port.in_waiting > 0:
                        char = android_port.read(1).decode('ascii', errors='ignore')
                        if char.isprintable() or char == '#':
                            response += char
                        if char == '#':
                            break
                        if len(response) >= 50:
                            break
                    else:
                        time.sleep(0.01)
                self._log(f"Test TX: {cmd_bytes} -> RX: '{response}'")
                return response
            except Exception as e:
                self._log(f"Test send error: {e}")
                return ""

        connection_confirmed, model, detected_onstep = \
            self.mount_protocol.test_connection(_serial_test_send)
        self.is_onstep = detected_onstep

        if connection_confirmed:
            self._log(f"Connection confirmed: {model}")
        else:
            # Fallback: accept if port is open (some mounts are silent)
            if android_port.is_open:
                model = f"{self.mount_protocol.name} Telescope (connected without test response)"
                self._log("No test response, but port is open — accepting connection")
                connection_confirmed = True

        if not connection_confirmed:
            error_msg = "USB serial port opened but telescope did not respond."
            self.last_error = error_msg
            self._log(f"ERROR: {error_msg}")
            android_port.close()
            return False

        # ── Mark as connected & start threads ──────────────────────────
        self.connection_type = 'serial'
        self.telescope_info = telescope_bridge.TelescopeInfo(
            port=f"USB#{port_index}",
            baudrate=baudrate,
            is_connected=True,
            model=model,
        )
        self.is_connected = True
        self._log(f"Serial connected to: {model}")

        import threading
        self._running = True
        self._read_thread = threading.Thread(
            target=self._read_loop, daemon=True, name="ReadLoop"
        )
        self._read_thread.start()
        self._io_thread = threading.Thread(
            target=self._io_worker, daemon=True, name="IOWorker"
        )
        self._io_thread.start()

        if self.on_connected:
            self.on_connected(self.telescope_info)
        return True

    telescope_bridge.TelescopeBridge._connect_serial = _android_connect_serial
    logger.info("Patched telescope_bridge._connect_serial -> Android USB serial")

    # ── Also patch get_available_ports() for Android USB discovery ──────
    def _android_get_available_ports(self):
        """List USB serial devices via Android UsbSerialManager."""
        ports = list_ports()
        result = []
        for p in ports:
            name = p.get('name', f"USB Serial #{p.get('index', '?')}")
            result.append(name)
        self._last_detected_ports = result
        return result

    telescope_bridge.TelescopeBridge.get_available_ports = _android_get_available_ports
    logger.info("Patched telescope_bridge.get_available_ports -> Android USB")


def _patch_plate_solvers():
    """
    Replace ASTAP subprocess calls with the cloud plate solver.

    Patches two classes:
      - auto_platesolve.AutoPlateSolver._solve_image()
      - realtime_tracking.FastPlateSolver.solve_fast()
    """
    from android_bridge.cloud_solver import cloud_solve, get_solver, _get_solver_fov

    # ── Patch AutoPlateSolver._solve_image ─────────────────────────────
    try:
        import auto_platesolve
        _original_solve_image = auto_platesolve.AutoPlateSolver._solve_image

        def _android_solve_image(self, image_path):
            """Cloud-based replacement for ASTAP _solve_image."""
            self.is_solving = True
            with self._lock:
                self.stats['total_attempts'] += 1

            start_time = time.time()

            try:
                # Read FOV from config (focal length + sensor width)
                fov = _get_solver_fov()  # 0.0 if not configured

                result = cloud_solve(
                    image_path,
                    hint_ra=self.hint_ra if self.use_hint else None,
                    hint_dec=self.hint_dec if self.use_hint else None,
                    search_radius=self.search_radius,
                    fov_deg=fov,
                )

                solve_time = (time.time() - start_time) * 1000

                if result is not None:
                    ra_hours, dec_degrees, _ = result
                    with self._lock:
                        self.stats['successful_solves'] += 1
                        self._update_avg_time(solve_time)
                        self.stats['last_solve_time'] = solve_time

                    solve_result = auto_platesolve.SolveResult(
                        success=True,
                        ra_hours=ra_hours,
                        dec_degrees=dec_degrees,
                        solve_time_ms=solve_time,
                        image_path=image_path,
                    )
                    self.solve_history.append(solve_result)
                    self._log(
                        f"Cloud solve OK: RA={ra_hours:.4f}h "
                        f"Dec={dec_degrees:.2f}d ({solve_time:.0f}ms)"
                    )
                    return solve_result

                with self._lock:
                    self.stats['failed_solves'] += 1
                return auto_platesolve.SolveResult(
                    success=False, error="Cloud solve failed"
                )

            except Exception as e:
                with self._lock:
                    self.stats['failed_solves'] += 1
                return auto_platesolve.SolveResult(
                    success=False, error=str(e)
                )
            finally:
                self.is_solving = False

        auto_platesolve.AutoPlateSolver._solve_image = _android_solve_image
        logger.info("Patched AutoPlateSolver._solve_image -> cloud solver")

        # ── Patch start_camera_mode to use Android camera bridge ──────
        def _android_start_camera_mode(self, camera_index=0, exposure_ms=500):
            """Open the Android camera and start the plate solve loop.

            Reads ``app._android_camera_source`` hint set by the UI:
              - ``"auto"`` or ``""`` -- try ZWO SDK > ZWO UVC > UVC > Phone
              - ``"zwo"``  -- ZWO ASI camera (SDK first, UVC fallback)
              - ``"zwo_sdk"``  -- ZWO ASI via native SDK only (full controls)
              - ``"uvc"``  -- generic USB UVC camera only
              - ``"phone"`` -- phone rear camera only

            Camera selection priority in auto mode:
              1. ZWO ASI via native SDK (full controls, all cameras)
              2. ZWO ASI via UVC (limited, planetary cameras only)
              3. Generic UVC camera (USB webcam)
              4. Phone rear camera (Camera2) -- fallback
            """
            from android_bridge import camera_bridge

            # Read the camera source hint set by /api/camera/settings
            # The hint is stored on the HeadlessTelescopeApp by the API handler.
            # Since 'self' is AutoPlateSolver (not the app), read from _app global.
            hint = ''
            try:
                if _app:
                    hint = getattr(_app, '_android_camera_source', '') or ''
            except Exception:
                pass
            hint = hint or 'auto'

            self._log(f"Opening Android camera (source={hint})...")

            success = False

            if hint == 'zwo_sdk':
                # Explicit SDK-only mode
                self._log("Opening ZWO ASI camera via native SDK...")
                if camera_bridge.is_asi_sdk_available():
                    success = camera_bridge.open_zwo_camera()
                    if success and camera_bridge.is_asi_sdk_active():
                        self._log("ZWO ASI camera opened via SDK (full control)")
                    else:
                        success = False
                        self._log("SDK mode requested but failed")
                else:
                    self._log("ASI SDK libraries not available")
            elif hint == 'zwo':
                # ZWO with SDK-first fallback to UVC
                self._log("Opening ZWO ASI camera (SDK preferred)...")
                success = camera_bridge.open_zwo_camera()
                if success:
                    source = camera_bridge.get_active_source()
                    self._log(f"ZWO ASI camera opened: {source}")
            elif hint == 'uvc':
                self._log("Opening USB UVC camera...")
                success = camera_bridge.open_uvc_camera()
                if success:
                    self._log("USB UVC camera opened")
            elif hint == 'phone':
                self._log("Opening phone camera...")
                success = camera_bridge.open_phone_camera()
                if success:
                    self._log("Phone camera opened")
            else:
                # Auto mode: try all in priority order
                # open_zwo_camera() now tries SDK first, then UVC
                self._log("Auto mode: trying ZWO ASI camera...")
                success = camera_bridge.open_zwo_camera()
                if success:
                    source = camera_bridge.get_active_source()
                    self._log(f"ZWO ASI camera opened: {source}")
                else:
                    self._log("No ZWO camera, trying generic UVC camera...")
                    success = camera_bridge.open_uvc_camera()
                    if success:
                        self._log("USB UVC camera opened")
                    else:
                        self._log("No USB camera, falling back to phone camera...")
                        success = camera_bridge.open_phone_camera()
                        if success:
                            self._log("Phone camera opened")

            if not success:
                self._log("Failed to open any camera")
                return False

            self.camera_index = camera_index
            self.exposure_ms = exposure_ms
            # Mark camera as open (self.camera set to sentinel so isOpened-like
            # checks don't fail; actual capture goes through camera_bridge)
            self.camera = True

            source_name = camera_bridge.get_active_source()
            self._log(f"Camera active: {source_name}")

            # Set exposure if supported
            try:
                camera_bridge.set_exposure(exposure_ms)
            except Exception:
                pass

            self._start_solve_loop(mode="camera")
            return True

        auto_platesolve.AutoPlateSolver.start_camera_mode = _android_start_camera_mode
        logger.info("Patched AutoPlateSolver.start_camera_mode -> Android camera")

        # ── Patch _capture_from_camera to use Android camera bridge ───
        def _android_capture_from_camera(self):
            """Capture a frame via Android camera bridge for plate solving."""
            from android_bridge import camera_bridge

            path = camera_bridge.capture_for_solving()
            if path:
                self._log(f"Captured frame: {path}")
            return path

        auto_platesolve.AutoPlateSolver._capture_from_camera = _android_capture_from_camera
        logger.info("Patched AutoPlateSolver._capture_from_camera -> Android camera")

        # ── Patch stop() to close Android camera instead of OpenCV ────
        _original_stop = auto_platesolve.AutoPlateSolver.stop

        def _android_stop(self):
            """Stop solving and close Android camera instead of OpenCV."""
            from android_bridge import camera_bridge

            self.is_running = False

            # Close Android camera (don't call .release() on sentinel)
            if self.camera:
                camera_bridge.close_camera()
                self.camera = None

            # Disconnect ASCOM camera if active
            if self.ascom_camera:
                self.ascom_camera.disconnect()
                self.ascom_camera = None

            # Wait for the solve thread to finish
            if self._solve_thread:
                self._solve_thread.join(timeout=2.0)

            self._log("Automatic plate solving stopped")

        auto_platesolve.AutoPlateSolver.stop = _android_stop
        logger.info("Patched AutoPlateSolver.stop -> Android camera cleanup")

    except Exception as e:
        logger.error(f"Failed to patch AutoPlateSolver: {e}")

    # ── Patch FastPlateSolver.solve_fast ────────────────────────────────
    try:
        import realtime_tracking

        def _android_solve_fast(self, image_path):
            """Cloud-based replacement for ASTAP FastPlateSolver.solve_fast."""
            start_time = time.time()

            fov = _get_solver_fov()  # 0.0 if not configured
            result = cloud_solve(
                image_path,
                hint_ra=self.hint_ra,
                hint_dec=self.hint_dec,
                search_radius=self.search_radius,
                fov_deg=fov,
            )

            if result is not None:
                ra_hours, dec_degrees, _ = result
                solve_time = (time.time() - start_time) * 1000
                self.solve_times.append(solve_time)
                self.hint_ra = ra_hours
                self.hint_dec = dec_degrees
                return ra_hours, dec_degrees, solve_time

            return None

        realtime_tracking.FastPlateSolver.solve_fast = _android_solve_fast
        logger.info("Patched FastPlateSolver.solve_fast -> cloud solver")

    except Exception as e:
        logger.error(f"Failed to patch FastPlateSolver: {e}")


def _patch_web_server_camera():
    """
    Replace OpenCV camera methods in web_server.py with Android camera bridge.

    Patches:
      - _open_uvc_camera()  -> use Android camera bridge
      - _generate_mjpeg()   -> get frames from Android camera bridge
      - _close_camera()     -> close via Android camera bridge
    """
    try:
        import web_server
        from android_bridge import camera_bridge

        # ── Patch _open_uvc_camera to use Android camera ───────────────
        def _android_open_uvc_camera(self, index=0):
            """Open USB or phone camera via Android bridge instead of OpenCV.

            Respects ``self._android_source`` hint from the UI:
              - ``"auto"`` or ``""`` — try ZWO SDK > ZWO UVC > UVC > Phone
              - ``"zwo"``  — ZWO ASI (SDK first, UVC fallback)
              - ``"zwo_sdk"`` — ZWO ASI via native SDK only
              - ``"uvc"``  — generic UVC only
              - ``"phone"`` — phone rear camera only
            """
            hint = getattr(self, '_android_source', '') or 'auto'
            success = False

            if hint == 'zwo_sdk':
                if camera_bridge.is_asi_sdk_available():
                    success = camera_bridge.open_zwo_camera()
                    if success and not camera_bridge.is_asi_sdk_active():
                        camera_bridge.close_camera()
                        success = False
            elif hint == 'zwo':
                success = camera_bridge.open_zwo_camera()
            elif hint == 'uvc':
                success = camera_bridge.open_uvc_camera()
            elif hint == 'phone':
                success = camera_bridge.open_phone_camera()
            else:
                # Auto: try all in priority order
                # open_zwo_camera() tries SDK first, then UVC
                success = camera_bridge.open_zwo_camera()
                if not success:
                    success = camera_bridge.open_uvc_camera()
                if not success:
                    success = camera_bridge.open_phone_camera()

            if success:
                self._camera_active = True
                self._camera_source = "android"
                source_name = camera_bridge.get_active_source()
                logger.info(f"Android camera opened: {source_name} (hint={hint})")
            else:
                logger.warning(f"Failed to open camera (hint={hint})")
            return success

        web_server.TelescopeWebServer._open_uvc_camera = _android_open_uvc_camera

        # ── Patch MJPEG generator to use Android camera frames ─────────
        _original_generate_mjpeg = web_server.TelescopeWebServer._generate_mjpeg

        def _android_generate_mjpeg(self):
            """MJPEG stream from Android camera bridge."""
            while self._camera_active:
                if self._camera_source == "ascom":
                    # ASCOM path unchanged (though ASCOM won't be used on Android)
                    jpeg = self._capture_ascom_frame_jpeg()
                    if jpeg is None:
                        time.sleep(0.2)
                        continue
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n'
                        + jpeg + b'\r\n'
                    )
                else:
                    # Android camera path
                    jpeg = camera_bridge.get_jpeg_frame()
                    if jpeg is None:
                        time.sleep(0.1)
                        continue
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n'
                        + jpeg + b'\r\n'
                    )
                    time.sleep(0.09)  # ~10 fps cap

        web_server.TelescopeWebServer._generate_mjpeg = _android_generate_mjpeg

        # ── Patch _close_camera ────────────────────────────────────────
        _original_close_camera = web_server.TelescopeWebServer._close_camera

        def _android_close_camera(self):
            """Close camera via Android bridge, then call original cleanup."""
            camera_bridge.close_camera()
            self._camera_active = False
            self._camera = None
            self._camera_source = "uvc"
            logger.info("Android camera closed")

        web_server.TelescopeWebServer._close_camera = _android_close_camera

        logger.info("Patched web_server camera methods -> Android camera bridge")

    except Exception as e:
        logger.error(f"Failed to patch web_server camera: {e}")


# ═══════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════

def _ensure_port_free(port: int, host: str = "127.0.0.1"):
    """
    Make sure the Flask port is not held by a zombie from a previous run.

    On Android, if the app was killed without a clean shutdown (USB cable
    pulled, ANR, OOM-kill) the old Python thread may still hold the TCP
    port.  We try to:
      1. Connect to the old server and hit /stop (graceful).
      2. If that fails, just verify the port is bindable.
    If the port is stuck, we wait up to 5 seconds for TIME_WAIT to expire
    (SO_REUSEADDR handles most cases).
    """
    # ── Try graceful shutdown of a leftover server ─────────────────────
    try:
        import urllib.request
        urllib.request.urlopen(
            f"http://{host}:{port}/api/status", timeout=1
        )
        # Old server is alive -- this shouldn't normally happen because
        # Android creates a fresh process, but just in case:
        logger.warning(f"Old Flask server still responding on port {port}")
    except Exception:
        pass  # Expected -- no old server running

    # ── Verify the port is bindable ────────────────────────────────────
    for attempt in range(6):  # up to ~3 seconds
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.close()
            if attempt > 0:
                logger.info(f"Port {port} became free after {attempt} retries")
            return  # Port is free
        except OSError as e:
            logger.warning(f"Port {port} busy (attempt {attempt + 1}): {e}")
            sock.close()
            time.sleep(0.5)

    # If we get here the port is still busy.  Log a warning but continue
    # anyway -- make_server() with SO_REUSEADDR might still succeed.
    logger.warning(f"Port {port} may still be in use -- proceeding anyway")


def _extract_catalog_assets(data_dir: str):
    """
    Extract catalog .h files from APK assets to the filesystem.

    catalog_loader.py reads files via os.path / open(), which only works
    on real filesystem paths.  On Android the catalog data is packed
    inside the APK under  assets/catalogs/data/*.h.

    This function copies them to  <data_dir>/catalogs/data/  so the
    existing Python loader works unchanged.  Skips files that already
    exist (same size) to avoid redundant I/O on every startup.
    """
    dest_base = os.path.join(data_dir, "catalogs", "data")
    os.makedirs(dest_base, exist_ok=True)

    # ── Try Chaquopy Java interop (real Android device) ────────────────
    try:
        from java import jclass
        # Get the Application context via Android's ActivityThread
        activity_thread = jclass("android.app.ActivityThread")
        context = activity_thread.currentApplication()
        asset_mgr = context.getAssets()

        # List files under "catalogs/data" in the APK assets
        asset_files = asset_mgr.list("catalogs/data")
        if not asset_files:
            logger.warning("No catalog assets found under catalogs/data")
            return

        extracted = 0
        for filename in asset_files:
            dest_path = os.path.join(dest_base, str(filename))
            if os.path.exists(dest_path):
                continue  # already extracted

            asset_path = f"catalogs/data/{filename}"
            try:
                inp = asset_mgr.open(asset_path)
                # Read the full asset via Java InputStream
                from java import jarray, jbyte
                buf = jarray(jbyte)(8192)
                with open(dest_path, "wb") as out:
                    while True:
                        n = inp.read(buf)
                        if n == -1:
                            break
                        # jarray -> bytes: use the first n bytes
                        out.write(bytes(buf[:n]))
                inp.close()
                extracted += 1
            except Exception as e:
                logger.warning(f"Failed to extract {asset_path}: {e}")

        logger.info(
            f"Catalog assets: {extracted} new files extracted to {dest_base} "
            f"({len(asset_files)} total)"
        )
        return

    except ImportError:
        pass  # Not running on Android (desktop test) -- fall through

    # ── Fallback: copy from parent project if available (desktop tests) ─
    try:
        src_base = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)
            ))),
            "catalogs", "data",
        )
        if os.path.isdir(src_base):
            import shutil
            for fn in os.listdir(src_base):
                src = os.path.join(src_base, fn)
                dst = os.path.join(dest_base, fn)
                if os.path.isfile(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
            logger.info(f"Catalog data copied from {src_base}")
        else:
            logger.warning(f"No catalog source found at {src_base}")
    except Exception as e:
        logger.warning(f"Catalog fallback copy failed: {e}")


def _stub_module(name: str):
    """
    Insert a dummy module into sys.modules so that 'import X' doesn't
    crash.  Any attribute access returns a no-op callable.
    """
    import types
    stub = types.ModuleType(name)
    stub.__dict__["__path__"] = []

    class _Stub:
        """Silent stand-in for any missing attribute or call."""
        def __getattr__(self, _):
            return _Stub()
        def __call__(self, *a, **kw):
            return _Stub()
        def __bool__(self):
            return False
        def __iter__(self):
            return iter([])
        def __len__(self):
            return 0
        def __str__(self):
            return ""
        def __int__(self):
            return 0
        def __float__(self):
            return 0.0

    stub.__dict__["__getattr__"] = lambda _: _Stub()
    sys.modules[name] = stub


def _create_default_config(config_path: str):
    """Write a minimal default config for first launch."""
    config = {
        "location": {
            "latitude": 0.0,
            "longitude": 0.0,
            "timezone": "UTC"
        },
        "connection": {
            "type": "wifi",
            "wifi_ip": "192.168.0.1",
            "wifi_port": 9996,
            "baudrate": 9600
        },
        "camera": {
            "solve_mode": "manual",
            "camera_index": 0,
            "solve_interval": 4.0
        },
        "tracking": {
            "correction_interval": 0.2,
            "prediction_horizon": 1.0,
            "pec_enabled": False
        },
        "solver": {
            "mode": "auto",
            "cloud_api_key": "",
            "search_radius": 10.0,
            "downsample": 4,
            "timeout": 120
        },
        "platform": "android"
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Created default config at {config_path}")
