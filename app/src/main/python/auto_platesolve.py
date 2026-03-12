"""
Automated Plate Solving Module for Dobson Alt-Az Telescope Controller

This module implements continuous automated plate solving using ASTAP (Astrometric
STAcking Program) as the external solver engine. It captures sky images from various
sources, submits them to ASTAP for astrometric solving, and returns precise celestial
coordinates (RA/Dec) for each solved frame.

Architecture Overview:
    - AutoPlateSolver: Main orchestrator class that manages the solve loop across
      multiple image acquisition modes (OpenCV webcam, folder watch, single image,
      or ASCOM camera). Runs the solve loop in a background daemon thread.
    - ASCOMCameraCapture: Handles direct camera control via the ASCOM/COM interface
      using pywin32, supporting astronomy cameras (ZWO ASI, QHY, etc.) on Windows.
    - SolveResult: Dataclass holding plate solve results including RA (hours),
      Dec (degrees), field width, solve time, and error information.

Plate Solving Pipeline:
    1. Image acquisition (camera capture or file monitoring)
    2. ASTAP CLI invocation with optional position hint for faster solving
    3. WCS (World Coordinate System) output file parsing for RA/Dec extraction
    4. Callback invocation with SolveResult for downstream consumers
    5. Temporary file cleanup

Integration with Telescope Controller:
    - Plate solve results (RA/Dec) are forwarded to realtime_tracking.py for
      conversion to Alt/Az coordinates appropriate for the Dobson mount.
    - Results feed into the correction pipeline: plate_solve -> Kalman filter ->
      tracking rate adjustments.
    - The on_solve_complete callback is the primary integration point.

Dependencies:
    - ASTAP must be installed with star database files (e.g., H18 or D50)
    - OpenCV (opencv-python) for webcam capture mode
    - pywin32 for ASCOM camera mode (Windows only)
    - astropy (optional) for FITS file writing; falls back to PNG via Pillow
    - numpy for image array manipulation
    - Pillow (PIL) for PNG export fallback
"""

import os
import sys
import time
import threading
import tempfile
import shutil
from typing import Optional, Callable, Tuple
from dataclasses import dataclass
from collections import deque
from pathlib import Path
import subprocess

from telescope_logger import get_logger

_logger = get_logger(__name__)


@dataclass
class SolveResult:
    """Holds the result of a single plate solve attempt.

    Each instance represents one ASTAP solve invocation, whether successful
    or failed. Successful results contain the solved celestial coordinates;
    failed results contain an error description.

    Attributes:
        success: True if ASTAP successfully solved the image.
        ra_hours: Right Ascension of the image center in decimal hours (0-24).
                  Converted from ASTAP's degree output (CRVAL1 / 15).
        dec_degrees: Declination of the image center in decimal degrees (-90 to +90).
                     Directly from ASTAP's CRVAL2 WCS keyword.
        solve_time_ms: Wall-clock time for the solve in milliseconds, including
                       ASTAP process startup and WCS file parsing.
        image_path: Filesystem path to the image that was solved.
        error: Human-readable error message if the solve failed; empty on success.
    """
    success: bool
    ra_hours: float = 0.0
    dec_degrees: float = 0.0
    solve_time_ms: float = 0.0
    image_path: str = ""
    error: str = ""


class AutoPlateSolver:
    """Automated plate solving system using ASTAP as the astrometric engine.

    Manages continuous plate solving in a background thread, acquiring images
    from one of several configurable sources and solving them in a timed loop.
    Each successful solve updates the internal position hint to accelerate
    subsequent solves (narrow search radius around the last known position).

    Supported acquisition modes:
        1. Camera mode (OpenCV/DirectShow) - Captures frames from a USB webcam
           or similar video device via OpenCV's VideoCapture.
        2. Folder watch mode - Monitors a directory for new FITS/image files and
           solves each new file as it appears.
        3. Single image mode - Repeatedly solves the same image file; useful for
           testing and development.
        4. ASCOM camera mode - Controls an astronomy camera (ASI, QHY, etc.) via
           the ASCOM COM interface for high-quality exposures.

    Thread Safety:
        The solve loop runs in a daemon thread. Statistics access is protected
        by a threading lock. The solver sets is_solving = True while ASTAP is
        running to allow external status queries.

    Typical Usage:
        solver = AutoPlateSolver(astap_path=r"C:\\Program Files\\astap\\astap.exe")
        solver.on_solve_complete = my_callback  # Called with SolveResult on each solve
        solver.set_hint(ra_hours=12.5, dec_degrees=45.0)  # Initial position hint
        solver.start_camera_mode(camera_index=0, exposure_ms=500)
        # ... later ...
        solver.stop()
        solver.cleanup()
    """

    def __init__(self, astap_path: str = r"C:\Program Files\astap\astap.exe"):
        """Initialize the automated plate solver.

        Args:
            astap_path: Full filesystem path to the ASTAP executable. ASTAP must
                        be installed with at least one star database (e.g., H18 or
                        D50) for solving to work.
        """
        self.astap_path = astap_path

        # --- Solve configuration ---
        self.solve_interval = 4.0       # Minimum interval between solve attempts (seconds)
        self.downsample = 8             # Image downsampling factor (ASTAP -z flag)
        self.search_radius = 10.0       # Sky search radius in degrees (ASTAP -r flag)
        self.timeout = 5.0              # ASTAP process timeout in seconds
        self.fov_deg = 0.0              # Estimated FOV in degrees (ASTAP -fov flag, 0 = not set)

        # Position hint: updated after each successful solve to narrow search area.
        # Providing a hint dramatically speeds up ASTAP by limiting the sky area to search.
        self.hint_ra = 0.0              # Hint Right Ascension in decimal hours
        self.hint_dec = 0.0             # Hint Declination in decimal degrees
        self.use_hint = True            # Whether to pass the hint to ASTAP

        # --- Runtime state ---
        self.is_running = False         # True while the solve loop is active
        self.is_solving = False         # True while ASTAP subprocess is running

        # --- OpenCV camera state ---
        self.camera = None              # cv2.VideoCapture instance (or None)
        self.camera_index = 0           # OpenCV camera device index
        self.exposure_ms = 500          # Exposure time in milliseconds for OpenCV camera

        # --- Folder watch state ---
        self.watch_folder: Optional[str] = None   # Directory path to monitor for new images
        self.last_image_time = 0                   # mtime of the last processed image file

        # --- Single image mode state ---
        self._single_image_path: Optional[str] = None  # Path to the single image to re-solve

        # --- ASCOM camera state ---
        self.ascom_camera: Optional[ASCOMCameraCapture] = None  # ASCOM camera wrapper instance
        self.ascom_exposure_sec = 0.5   # Exposure time in seconds for ASCOM camera

        # Temporary directory for captured images and ASTAP output files.
        # On Linux, prefer /dev/shm (RAM-backed tmpfs) to avoid
        # slow disk I/O for the frequent capture/solve/delete cycle.
        _tmpdir_base = None
        if sys.platform != "win32" and os.path.isdir("/dev/shm"):
            _tmpdir_base = "/dev/shm"
        self.temp_dir = tempfile.mkdtemp(prefix="astap_", dir=_tmpdir_base)

        # --- Image saving configuration ---
        self.save_images = False                    # If True, keep captured images after solving
        self.save_folder: Optional[str] = None      # Save directory (None defaults to temp_dir)
        self.save_format = "fits"                   # Save format: "fits" or "png"

        # --- Callbacks ---
        self.on_solve_complete: Optional[Callable[[SolveResult], None]] = None  # Called on each successful solve
        self.on_log: Optional[Callable[[str], None]] = None                    # Called for log messages

        # --- Threading ---
        self._solve_thread: Optional[threading.Thread] = None  # Background solve loop thread
        self._lock = threading.Lock()                          # Protects statistics access

        # --- Solve statistics ---
        self.stats = {
            'total_attempts': 0,        # Total number of solve attempts
            'successful_solves': 0,     # Number of successful solves
            'failed_solves': 0,         # Number of failed solves
            'avg_solve_time': 0.0,      # Running average solve time in ms
            'last_solve_time': 0.0      # Most recent solve time in ms
        }

        # Circular buffer of recent SolveResult objects for history/analysis
        self.solve_history: deque = deque(maxlen=50)

        # --- ROI / center-crop ---
        # Cropping the center of large images before solving reduces ASTAP's
        # workload.  The crop_ratio (0-1) controls how much of the frame to
        # keep (e.g. 0.5 = central 50% in each axis = 25% total area).
        # Set crop_ratio to 1.0 (or crop_enabled=False) to disable.
        self.crop_enabled = True
        self.crop_ratio = 0.6               # Keep central 60% of each axis

        # --- Adaptive solve interval ---
        self._adaptive_interval = True      # Shorten interval when errors high
        self._min_solve_interval = 2.0      # Fastest: every 2s
        self._max_solve_interval = 10.0     # Slowest: every 10s
        self._adaptive_error_threshold = 5.0  # arcsec: high error -> faster solves
        self._last_error_arcsec = 0.0       # Cached from last solve
        self._astap_warmed_up = False       # Has ASTAP been run once to load DB?

    def start_camera_mode(self, camera_index: int = 0, exposure_ms: int = 500) -> bool:
        """Start plate solving with automatic webcam capture via OpenCV.

        Opens the specified camera device using DirectShow (Windows) backend,
        configures exposure, and starts the background solve loop in "camera" mode.

        Args:
            camera_index: OpenCV camera device index (0 = default/first camera).
            exposure_ms: Desired exposure time in milliseconds. Converted to
                         OpenCV's logarithmic exposure scale as -exposure_ms/100.

        Returns:
            True if the camera was opened successfully and the solve loop started.
            False if OpenCV is not installed, the camera cannot be opened, or
            another error occurs.
        """
        try:
            import cv2

            self._log(f"Connecting to camera {camera_index}...")

            # Use DirectShow backend on Windows for better device compatibility
            self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

            if not self.camera.isOpened():
                self._log("Failed to open camera")
                return False

            # Set exposure: OpenCV uses a logarithmic scale where negative values
            # represent shorter exposures (e.g., -5 = 2^(-5) seconds)
            self.camera.set(cv2.CAP_PROP_EXPOSURE, -int(exposure_ms / 100))

            self.camera_index = camera_index
            self.exposure_ms = exposure_ms

            self._log(f"Camera {camera_index} connected")

            # Start the background solve loop
            self._start_solve_loop(mode="camera")

            return True

        except ImportError:
            self._log("OpenCV not installed. Run: pip install opencv-python")
            return False
        except Exception as e:
            self._log(f"Camera error: {e}")
            return False

    def start_folder_watch_mode(self, folder_path: str) -> bool:
        """Start plate solving by monitoring a folder for new image files.

        Watches the specified directory for new FITS or image files. When a file
        with a modification time newer than the last processed file is found,
        it is automatically submitted for plate solving.

        Supported file extensions: .fits, .fit, .fts, .jpg, .jpeg, .png, .tiff, .tif

        Args:
            folder_path: Absolute path to the directory to monitor.

        Returns:
            True if the folder exists and the solve loop was started.
            False if the folder does not exist.
        """
        if not os.path.isdir(folder_path):
            self._log(f"Folder not found: {folder_path}")
            return False

        self.watch_folder = folder_path
        self._log(f"Watching folder: {folder_path}")

        self._start_solve_loop(mode="folder")
        return True

    def start_single_image_mode(self, image_path: str) -> bool:
        """Start plate solving by repeatedly solving a single image file.

        This mode is primarily useful for testing and development. The same image
        is solved on each loop iteration, allowing verification of solve accuracy
        and performance without requiring a live camera.

        Args:
            image_path: Absolute path to the image file to solve repeatedly.

        Returns:
            True if the image file exists and the solve loop was started.
            False if the image file does not exist.
        """
        if not os.path.isfile(image_path):
            self._log(f"Image not found: {image_path}")
            return False

        self._single_image_path = image_path
        self._log(f"Single image mode: {image_path}")

        self._start_solve_loop(mode="single")
        return True

    def start_ascom_mode(self, camera_id: Optional[str] = None, exposure_sec: float = 0.5,
                         gain: int = 100, binning: int = 2) -> bool:
        """Start plate solving with an ASCOM-compatible astronomy camera.

        Initializes and connects to an ASCOM camera (e.g., ZWO ASI, QHY) via
        the COM interface. If no camera_id is provided, the ASCOM Chooser dialog
        is displayed to let the user select a camera interactively.

        Args:
            camera_id: ASCOM ProgID of the camera (e.g., "ASCOM.ASICamera2.Camera").
                       If None, opens the ASCOM Chooser dialog for interactive selection.
            exposure_sec: Exposure duration in seconds per capture.
            gain: Camera gain setting (sensor-dependent, typically 0-300).
            binning: Pixel binning factor (1 = no binning, 2 = 2x2, 4 = 4x4).
                     Higher binning reduces resolution but improves sensitivity
                     and download speed, which is beneficial for plate solving.

        Returns:
            True if the camera was connected successfully and the solve loop started.
            False if connection failed or ASCOM is not available.
        """
        try:
            self._log("Initializing ASCOM camera...")

            # Create the ASCOM camera wrapper instance
            self.ascom_camera = ASCOMCameraCapture()
            self.ascom_camera.on_log = self._log
            self.ascom_camera.gain = gain
            self.ascom_camera.binning = binning
            self.ascom_exposure_sec = exposure_sec

            # Connect to the camera (may open Chooser dialog if camera_id is None)
            if not self.ascom_camera.connect(camera_id):
                self._log("Failed to connect ASCOM camera")
                self.ascom_camera = None
                return False

            self._log(f"Exposure: {exposure_sec}s, Gain: {gain}, Binning: {binning}x{binning}")

            # Start the background solve loop
            self._start_solve_loop(mode="ascom")

            return True

        except Exception as e:
            self._log(f"ASCOM initialization error: {e}")
            return False

    def choose_ascom_camera(self) -> Optional[str]:
        """Open the ASCOM Chooser dialog for interactive camera selection.

        Creates a temporary ASCOMCameraCapture instance to display the standard
        ASCOM device chooser. Does not connect to the camera.

        Returns:
            The ASCOM ProgID string of the selected camera (e.g.,
            "ASCOM.ASICamera2.Camera"), or None if the user cancelled
            or an error occurred.
        """
        try:
            temp_cam = ASCOMCameraCapture()
            if temp_cam.choose_camera():
                return temp_cam.camera_id
            return None
        except Exception as e:
            self._log(f"Camera selection error: {e}")
            return None

    @staticmethod
    def list_ascom_cameras() -> list:
        """List all ASCOM-registered camera devices on this system.

        Returns:
            A list of dicts, each with 'id' (ASCOM ProgID) and 'name' keys.
            Returns an empty list if no cameras are registered or ASCOM is
            not available.
        """
        return ASCOMCameraCapture.list_cameras()

    def stop(self):
        """Stop the automatic plate solving loop and release all resources.

        Signals the background thread to stop, releases the OpenCV camera
        (if any), disconnects the ASCOM camera (if any), and waits up to
        2 seconds for the solve thread to finish.
        """
        self.is_running = False

        # Release OpenCV camera if active
        if self.camera:
            self.camera.release()
            self.camera = None

        # Disconnect ASCOM camera if active
        if self.ascom_camera:
            self.ascom_camera.disconnect()
            self.ascom_camera = None

        # Wait for the solve thread to finish (with timeout to avoid hanging)
        if self._solve_thread:
            self._solve_thread.join(timeout=2.0)

        self._log("Automatic plate solving stopped")

    def warm_up_astap(self):
        """Run ASTAP with a dummy invocation to pre-load its star database.

        The first real solve is typically slower because
        ASTAP reads its star database from storage.  Calling this at
        startup makes the first real solve complete at normal speed.
        """
        if self._astap_warmed_up:
            return
        try:
            import subprocess
            self._log("Warming up ASTAP (pre-loading star database)...")
            # Just ask ASTAP for its version -- this forces it to load the DB
            subprocess.run(
                [self.astap_path, "-h"],
                capture_output=True, timeout=10.0,
            )
            self._astap_warmed_up = True
            self._log("ASTAP warm-up complete")
        except Exception as e:
            self._log(f"ASTAP warm-up skipped: {e}")

    def _start_solve_loop(self, mode: str):
        """Start the background plate solving loop thread.

        Creates and starts a daemon thread that runs _solve_loop() with the
        specified acquisition mode.

        Args:
            mode: Acquisition mode string, one of "camera", "ascom", "folder",
                  or "single".
        """
        self.is_running = True

        # Warm up ASTAP to pre-load star database (avoids slow first solve)
        if not self._astap_warmed_up:
            self.warm_up_astap()

        self._solve_thread = threading.Thread(
            target=self._solve_loop,
            args=(mode,),
            daemon=True,
            name="AutoPlateSolve"
        )
        self._solve_thread.start()

        self._log(f"Automatic plate solving started (mode: {mode}, interval: {self.solve_interval}s)")

    def _solve_loop(self, mode: str):
        """Main plate solving loop running in a background thread.

        Continuously acquires images, submits them to ASTAP for solving,
        processes results, and cleans up temporary files. The loop respects
        self.solve_interval as the minimum time between iterations.

        Args:
            mode: Acquisition mode - determines which image source method is called:
                  "camera" -> _capture_from_camera() (OpenCV webcam)
                  "ascom"  -> _capture_from_ascom() (ASCOM astronomy camera)
                  "folder" -> _get_latest_from_folder() (directory monitoring)
                  "single" -> uses _single_image_path directly (static image)
        """
        while self.is_running:
            loop_start = time.time()

            try:
                # Acquire an image based on the current mode
                image_path = None

                if mode == "camera":
                    image_path = self._capture_from_camera()
                elif mode == "ascom":
                    image_path = self._capture_from_ascom()
                elif mode == "folder":
                    image_path = self._get_latest_from_folder()
                elif mode == "single":
                    image_path = self._single_image_path

                if image_path and os.path.exists(image_path):
                    # Run ASTAP plate solve on the acquired image
                    result = self._solve_image(image_path)

                    if result.success:
                        # Update position hint for the next solve iteration.
                        # This narrows ASTAP's search area, dramatically improving
                        # solve speed for subsequent frames.
                        self.hint_ra = result.ra_hours
                        self.hint_dec = result.dec_degrees

                        # Notify the registered callback (e.g., realtime_tracking.py)
                        if self.on_solve_complete:
                            self.on_solve_complete(result)

                    # Clean up temporary files for camera modes (OpenCV and ASCOM).
                    # Folder and single modes use externally-managed files.
                    if mode in ("camera", "ascom") and image_path != self._single_image_path:
                        if not self.save_images:
                            # Delete the captured image and all ASTAP output files
                            try:
                                os.remove(image_path)
                                # ASTAP generates .wcs (WCS solution), .ini (solve parameters),
                                # and .bak (backup) files alongside the input image
                                base = os.path.splitext(image_path)[0]
                                for ext in ['.wcs', '.ini', '.bak']:
                                    f = base + ext
                                    if os.path.exists(f):
                                        os.remove(f)
                            except OSError:
                                pass
                        else:
                            # When saving images, only delete ASTAP's temporary metadata
                            # files (.ini, .bak) but keep the image and .wcs solution
                            try:
                                base = os.path.splitext(image_path)[0]
                                for ext in ['.ini', '.bak']:
                                    f = base + ext
                                    if os.path.exists(f):
                                        os.remove(f)
                            except OSError:
                                pass

            except Exception as e:
                self._log(f"Error in solve loop: {e}")

            # Adaptive solve interval: shorten when tracking error is high,
            # lengthen when stable (saves CPU).
            effective_interval = self.solve_interval
            if self._adaptive_interval and self._last_error_arcsec > 0:
                if self._last_error_arcsec > self._adaptive_error_threshold:
                    # High error: solve faster (down to min)
                    effective_interval = max(
                        self._min_solve_interval,
                        self.solve_interval * 0.5,
                    )
                elif self._last_error_arcsec < self._adaptive_error_threshold * 0.3:
                    # Very stable: solve less often (up to max)
                    effective_interval = min(
                        self._max_solve_interval,
                        self.solve_interval * 1.5,
                    )

            # Enforce minimum interval between solve iterations.
            # Wait in small increments for responsive stopping.
            elapsed = time.time() - loop_start
            wait_time = max(0, effective_interval - elapsed)
            while wait_time > 0 and self.is_running:
                sleep_chunk = min(0.25, wait_time)
                time.sleep(sleep_chunk)
                wait_time -= sleep_chunk

    def _get_capture_folder(self) -> str:
        """Return the directory to save captured images to.

        If image saving is enabled and a valid save folder is configured,
        returns that folder. Otherwise returns the temporary directory.

        Returns:
            Absolute path to the capture output directory.
        """
        if self.save_images and self.save_folder and os.path.isdir(self.save_folder):
            return self.save_folder
        return self.temp_dir

    def _crop_center(self, frame):
        """Crop the central region of an image numpy array.

        Used to reduce the number of pixels ASTAP must process, which on
        this can cut solve time by 40-60% for large sensor images.
        The center crop is astrophotographically sound because the optical
        axis (and thus the least-distorted region) is at the frame center.

        Args:
            frame: numpy ndarray of shape (H, W) or (H, W, C).

        Returns:
            Cropped numpy ndarray, or the original frame if cropping is
            disabled or the crop ratio is >= 1.0.
        """
        if not self.crop_enabled or self.crop_ratio >= 1.0:
            return frame
        h, w = frame.shape[:2]
        new_h = max(int(h * self.crop_ratio), 64)
        new_w = max(int(w * self.crop_ratio), 64)
        y0 = (h - new_h) // 2
        x0 = (w - new_w) // 2
        return frame[y0:y0 + new_h, x0:x0 + new_w]

    def _capture_from_camera(self) -> Optional[str]:
        """Capture a single frame from the OpenCV webcam.

        Reads and discards several frames first to flush the camera's internal
        buffer and get the most recent frame (USB webcams often buffer 2-3 frames).
        Converts the captured frame to grayscale (ASTAP expects monochrome input
        for optimal performance) and saves it as a PNG file.

        Returns:
            Absolute path to the saved PNG image, or None if capture failed.
        """
        if not self.camera or not self.camera.isOpened():
            return None

        try:
            import cv2

            # Discard 3 buffered frames to get the most recent one.
            # USB webcams typically buffer a few frames internally, so reading
            # and discarding ensures we get a fresh exposure.
            for _ in range(3):
                self.camera.read()

            ret, frame = self.camera.read()

            if not ret:
                return None

            # Generate a unique filename using millisecond timestamp
            timestamp = int(time.time() * 1000)
            folder = self._get_capture_folder()
            image_path = os.path.join(folder, f"capture_{timestamp}.png")

            # Convert color image to grayscale for plate solving.
            # ASTAP works best with monochrome data; color channels add
            # noise without improving astrometric accuracy.
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ROI center crop: reduces pixel count for ASTAP (Pi optimization)
            frame = self._crop_center(frame)

            cv2.imwrite(image_path, frame)

            return image_path

        except Exception as e:
            self._log(f"Capture error: {e}")
            return None

    def _capture_from_ascom(self) -> Optional[str]:
        """Capture a single frame from the ASCOM astronomy camera.

        Takes an exposure using the configured duration and saves the result
        to disk in the configured format (FITS or PNG).

        Returns:
            Absolute path to the saved image file, or None if capture failed
            or the camera is not connected.
        """
        if not self.ascom_camera or not self.ascom_camera.is_connected:
            return None

        try:
            # Generate a human-readable timestamp for the filename
            # Format: YYYYMMDD_HHMMSS_mmm (date_time_milliseconds)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Trim to milliseconds
            folder = self._get_capture_folder()

            # Choose file extension based on configured save format
            ext = ".fits" if self.save_format == "fits" else ".png"
            image_path = os.path.join(folder, f"ascom_{timestamp}{ext}")

            # Capture the image via the ASCOM camera interface
            if self.ascom_camera.capture(self.ascom_exposure_sec, image_path):
                if self.save_images:
                    self._log(f"Image saved: {os.path.basename(image_path)}")
                return image_path
            else:
                return None

        except Exception as e:
            self._log(f"ASCOM capture error: {e}")
            return None

    def set_save_folder(self, folder: str):
        """Set the directory where captured images are saved.

        Only takes effect when self.save_images is True.

        Args:
            folder: Absolute path to an existing directory. If the directory
                    does not exist, the setting is rejected with a log message.
        """
        if os.path.isdir(folder):
            self.save_folder = folder
            self._log(f"Save folder set: {folder}")
        else:
            self._log(f"Invalid folder: {folder}")

    def _get_latest_from_folder(self) -> Optional[str]:
        """Find the newest image file in the watched folder.

        Scans the watch folder for files with supported image extensions
        and returns the one with the most recent modification time, but only
        if it is newer than the last file we already processed.

        Supported extensions: .fits, .fit, .fts, .jpg, .jpeg, .png, .tiff, .tif

        Returns:
            Absolute path to the newest unprocessed image file, or None if no
            new files are available.
        """
        if not self.watch_folder:
            return None

        try:
            # Supported image and FITS file extensions
            extensions = ('.fits', '.fit', '.fts', '.jpg', '.jpeg', '.png', '.tiff', '.tif')

            # Find the most recently modified file newer than the last processed one
            latest_file = None
            latest_time = self.last_image_time

            for f in os.listdir(self.watch_folder):
                if f.lower().endswith(extensions):
                    path = os.path.join(self.watch_folder, f)
                    mtime = os.path.getmtime(path)

                    if mtime > latest_time:
                        latest_time = mtime
                        latest_file = path

            # Only return a file if it's strictly newer than the last one we processed
            if latest_file and latest_time > self.last_image_time:
                self.last_image_time = latest_time
                return latest_file

            return None

        except Exception as e:
            self._log(f"Folder read error: {e}")
            return None

    def _solve_image(self, image_path: str) -> SolveResult:
        """Run ASTAP plate solve on a single image.

        Constructs the ASTAP CLI command, executes it as a subprocess with
        a timeout, and parses the resulting WCS file for RA/Dec coordinates.

        ASTAP CLI arguments used:
            -f <path>       : Input image file path
            -r <radius>     : Sky search radius in degrees (limits search area)
            -z <factor>     : Downsampling factor (reduces image resolution for speed)
            -update         : Write WCS solution back to the image header
            -ra <hours>     : Hint RA in decimal hours (optional, speeds up solving)
            -spd <degrees>  : Hint South Pole Distance = 90 + Dec (optional)
                              SPD is used instead of Dec because it avoids negative values

        Args:
            image_path: Absolute path to the image file to solve.

        Returns:
            SolveResult with success=True and coordinates if solved, or
            success=False with an error message if solving failed or timed out.
        """
        self.is_solving = True
        with self._lock:
            self.stats['total_attempts'] += 1

        start_time = time.time()

        try:
            # Build the ASTAP command line
            cmd = [
                self.astap_path,
                "-f", image_path,       # Input image file
                "-r", str(self.search_radius),  # Search radius in degrees
                "-z", str(self.downsample),     # Downsampling factor (1=none, 2=half, 4=quarter)
                "-update"               # Write WCS keywords to output file
            ]

            # Add FOV estimate if configured (from focal length + sensor width).
            # This dramatically helps ASTAP converge by narrowing the scale search.
            # Without it, ASTAP must try many scale options (slower, more failures).
            if self.fov_deg > 0:
                cmd.extend(["-fov", f"{self.fov_deg:.2f}"])

            # Add position hint if enabled and a previous position is known.
            # The hint dramatically reduces solve time by limiting the sky area
            # ASTAP needs to search. ASTAP uses South Pole Distance (SPD = 90 + Dec)
            # instead of Declination directly.
            if self.use_hint and (self.hint_ra != 0 or self.hint_dec != 0):
                cmd.extend(["-ra", str(self.hint_ra)])          # Hint RA in decimal hours
                cmd.extend(["-spd", str(90 + self.hint_dec)])   # Hint SPD = 90 + Declination

            # Execute ASTAP as a subprocess with a timeout to prevent hangs
            result = subprocess.run(
                cmd,
                capture_output=True,    # Capture stdout/stderr for debugging
                timeout=self.timeout    # Kill process if it exceeds the timeout
            )

            solve_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            with self._lock:
                self.stats['last_solve_time'] = solve_time

            # ASTAP returns exit code 0 on successful solve
            if result.returncode == 0:
                # Parse the WCS output file to extract RA/Dec coordinates
                ra, dec = self._parse_wcs(image_path)

                if ra is not None and dec is not None:
                    with self._lock:
                        self.stats['successful_solves'] += 1
                        self._update_avg_time(solve_time)

                    solve_result = SolveResult(
                        success=True,
                        ra_hours=ra,
                        dec_degrees=dec,
                        solve_time_ms=solve_time,
                        image_path=image_path
                    )

                    self.solve_history.append(solve_result)
                    self._log(f"Solve OK: RA={ra:.4f}h Dec={dec:.2f} deg ({solve_time:.0f}ms)")

                    return solve_result

            # Solve failed (non-zero exit code or WCS parsing failed)
            with self._lock:
                self.stats['failed_solves'] += 1
            return SolveResult(success=False, error="Solve failed")

        except subprocess.TimeoutExpired:
            with self._lock:
                self.stats['failed_solves'] += 1
            return SolveResult(success=False, error="Timeout")
        except Exception as e:
            with self._lock:
                self.stats['failed_solves'] += 1
            return SolveResult(success=False, error=str(e))
        finally:
            self.is_solving = False

    def _parse_wcs(self, image_path: str) -> Tuple[Optional[float], Optional[float]]:
        """Parse the WCS (World Coordinate System) file generated by ASTAP.

        ASTAP writes a .wcs file alongside the input image containing FITS-style
        WCS keywords. The two critical keywords are:
            - CRVAL1: Reference point RA in degrees (0-360)
            - CRVAL2: Reference point Dec in degrees (-90 to +90)

        These represent the celestial coordinates of the image center (the
        reference pixel defined by CRPIX1/CRPIX2).

        Args:
            image_path: Path to the solved image. The WCS file is expected at
                        the same path with a .wcs extension instead of the
                        original extension.

        Returns:
            A tuple of (ra_hours, dec_degrees) where RA has been converted
            from degrees to hours (divided by 15). Returns (None, None) if
            the WCS file doesn't exist or cannot be parsed.
        """
        import re

        # ASTAP writes the WCS file with the same basename but .wcs extension
        wcs_path = os.path.splitext(image_path)[0] + ".wcs"

        if not os.path.exists(wcs_path):
            return None, None

        try:
            with open(wcs_path, 'r') as f:
                content = f.read()

            # Extract CRVAL1 (RA in degrees) and CRVAL2 (Dec in degrees)
            # from FITS-style header lines like: "CRVAL1  =    123.456789"
            ra_match = re.search(r'CRVAL1\s*=\s*([-\d.]+)', content)
            dec_match = re.search(r'CRVAL2\s*=\s*([-\d.]+)', content)

            if ra_match and dec_match:
                ra_deg = float(ra_match.group(1))    # RA in degrees (0-360)
                dec_deg = float(dec_match.group(1))   # Dec in degrees (-90 to +90)
                # Convert RA from degrees to hours (360 deg = 24 hours, so divide by 15)
                return ra_deg / 15.0, dec_deg

            return None, None

        except (ValueError, KeyError):
            return None, None

    def _update_avg_time(self, new_time: float):
        """Update the running average solve time using cumulative averaging.

        Uses the formula: new_avg = (old_avg * (n-1) + new_value) / n
        This avoids storing all historical values while maintaining accuracy.

        Args:
            new_time: The solve time in milliseconds to incorporate into the average.
        """
        n = self.stats['successful_solves']
        if n == 1:
            self.stats['avg_solve_time'] = new_time
        else:
            self.stats['avg_solve_time'] = (
                self.stats['avg_solve_time'] * (n - 1) + new_time
            ) / n

    def get_statistics(self) -> dict:
        """Return a snapshot of solve statistics and current state.

        Thread-safe: acquires the internal lock before reading statistics.

        Returns:
            A dictionary containing:
                - total_attempts: Total solve attempts
                - successful_solves: Number of successful solves
                - failed_solves: Number of failed solves
                - avg_solve_time: Average solve time in ms
                - last_solve_time: Most recent solve time in ms
                - success_rate: Success percentage (0-100)
                - is_running: Whether the solve loop is active
                - is_solving: Whether a solve is currently in progress
                - hint_ra: Current RA hint in hours
                - hint_dec: Current Dec hint in degrees
        """
        with self._lock:
            success_rate = 0
            if self.stats['total_attempts'] > 0:
                success_rate = (self.stats['successful_solves'] /
                               self.stats['total_attempts']) * 100

            return {
                **self.stats,
                'success_rate': success_rate,
                'is_running': self.is_running,
                'is_solving': self.is_solving,
                'hint_ra': self.hint_ra,
                'hint_dec': self.hint_dec
            }

    def set_hint(self, ra_hours: float, dec_degrees: float):
        """Set the position hint used to accelerate ASTAP solving.

        Providing an approximate position allows ASTAP to search a small area
        of the sky instead of the entire celestial sphere, reducing solve time
        from seconds to fractions of a second.

        Args:
            ra_hours: Approximate Right Ascension in decimal hours (0-24).
            dec_degrees: Approximate Declination in decimal degrees (-90 to +90).
        """
        self.hint_ra = ra_hours
        self.hint_dec = dec_degrees

    def _log(self, message: str):
        """Send a log message to the registered callback.

        Args:
            message: The log message string to forward.
        """
        if self.on_log:
            self.on_log(message)

    def cleanup(self):
        """Remove the temporary directory and all files within it.

        Should be called when the solver is no longer needed to free disk space
        used by temporary captured images and ASTAP output files.
        """
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except OSError:
            pass


class ASCOMCameraCapture:
    """ASCOM camera control wrapper for astronomy cameras on Windows.

    Provides a simplified interface to ASCOM-compatible astronomy cameras
    (ZWO ASI, QHY, Atik, etc.) via the Windows COM interface using pywin32.
    Handles camera connection, configuration (gain, binning), image capture,
    and saving in FITS or PNG format.

    ASCOM (Astronomy Common Object Model) is a Windows-standard interface
    that allows astronomy software to communicate with cameras, mounts,
    focusers, and other devices through a uniform COM API.

    Requirements:
        - Windows OS with ASCOM Platform installed
        - pywin32 Python package (pip install pywin32)
        - ASCOM driver for the specific camera installed and registered

    Typical Usage:
        cam = ASCOMCameraCapture()
        cam.gain = 200
        cam.binning = 2
        cam.connect("ASCOM.ASICamera2.Camera")
        cam.capture(exposure_sec=1.0, save_path="image.fits")
        cam.disconnect()
    """

    def __init__(self):
        """Initialize the ASCOM camera wrapper with default settings."""
        self.camera = None              # COM camera object (win32com dispatch)
        self.is_connected = False       # True when successfully connected
        self.camera_id = ""             # ASCOM ProgID (e.g., "ASCOM.ASICamera2.Camera")
        self.camera_name = ""           # Human-readable camera name from the driver
        self.on_log: Optional[Callable[[str], None]] = None  # Optional log callback

        # Camera configuration parameters
        self.gain = 100                 # Sensor gain (higher = more sensitive, more noise)
        self.binning = 1                # Pixel binning factor (1=no binning, 2=2x2, etc.)
        self.width = 0                  # Sensor width in pixels (populated on connect)
        self.height = 0                 # Sensor height in pixels (populated on connect)

    def _log(self, message: str):
        """Send a log message to the registered callback, or print to console.

        Args:
            message: The log message string to forward.
        """
        if self.on_log:
            self.on_log(message)
        else:
            _logger.info(message)

    @staticmethod
    def list_cameras() -> list:
        """List all ASCOM-registered camera devices on this system.

        Queries the ASCOM Profile to enumerate all camera drivers that have
        been registered on the system. This does not require the cameras to
        be physically connected.

        Returns:
            A list of dicts, each containing:
                - 'id': ASCOM ProgID string (e.g., "ASCOM.ASICamera2.Camera")
                - 'name': Human-readable device name
            Returns an empty list if ASCOM is not installed or no cameras
            are registered.
        """
        cameras = []
        try:
            import win32com.client

            # Use the ASCOM Profile utility to enumerate registered camera drivers
            profile = win32com.client.Dispatch("ASCOM.Utilities.Profile")
            profile.DeviceType = "Camera"

            # RegisteredDevices returns a list of (ProgID, Name) tuples
            registered = profile.RegisteredDevices("Camera")
            for item in registered:
                cameras.append({
                    'id': item[0],
                    'name': item[1]
                })
        except Exception:
            pass

        return cameras

    def choose_camera(self) -> bool:
        """Open the standard ASCOM Chooser dialog for interactive camera selection.

        Displays the ASCOM Chooser UI which lists all registered camera drivers
        and allows the user to select one. The selected camera's ProgID is stored
        in self.camera_id.

        Returns:
            True if the user selected a camera, False if cancelled or on error.
        """
        try:
            import win32com.client

            # The ASCOM Chooser provides a standard Windows dialog for device selection
            chooser = win32com.client.Dispatch("ASCOM.Utilities.Chooser")
            chooser.DeviceType = "Camera"
            # Pass the current camera_id to pre-select it in the dialog
            camera_id = chooser.Choose(self.camera_id)

            if camera_id:
                self.camera_id = camera_id
                return True
            return False

        except ImportError:
            self._log("pywin32 not installed. Run: pip install pywin32")
            return False
        except Exception as e:
            self._log(f"Camera selection error: {e}")
            return False

    def connect(self, camera_id: Optional[str] = None) -> bool:
        """Connect to an ASCOM camera.

        Creates a COM object for the specified camera driver, sets Connected=True,
        reads sensor dimensions, and applies gain/binning settings.

        Args:
            camera_id: ASCOM ProgID of the camera (e.g., "ASCOM.ASICamera2.Camera").
                       If None and no camera_id is stored, opens the Chooser dialog.

        Returns:
            True if the connection was established successfully.
            False if the connection failed, pywin32 is not installed, or the
            user cancelled the Chooser dialog.
        """
        try:
            import win32com.client

            if camera_id:
                self.camera_id = camera_id

            if not self.camera_id:
                # No camera ID specified; open the interactive chooser dialog
                if not self.choose_camera():
                    return False

            self._log(f"Connecting to {self.camera_id}...")

            # Create the COM object and establish connection
            self.camera = win32com.client.Dispatch(self.camera_id)
            self.camera.Connected = True

            # Read camera sensor information
            self.camera_name = self.camera.Name
            self.width = self.camera.CameraXSize
            self.height = self.camera.CameraYSize

            # Configure binning if the camera supports it.
            # Binning combines adjacent pixels, increasing sensitivity
            # and reducing download time at the cost of resolution.
            try:
                if self.camera.CanSetBinning:
                    self.camera.BinX = self.binning
                    self.camera.BinY = self.binning
            except Exception:
                pass

            # Configure gain if the camera supports it.
            # Not all ASCOM camera drivers expose a Gain property.
            try:
                if hasattr(self.camera, 'Gain'):
                    self.camera.Gain = self.gain
            except Exception:
                pass

            self.is_connected = True
            self._log(f"Camera connected: {self.camera_name}")
            self._log(f"   Resolution: {self.width}x{self.height}")

            return True

        except ImportError:
            self._log("pywin32 not installed. Run: pip install pywin32")
            return False
        except Exception as e:
            self._log(f"ASCOM error: {e}")
            return False

    def set_gain(self, gain: int):
        """Set the camera sensor gain.

        Updates the stored gain value and applies it immediately if the camera
        is currently connected and supports gain adjustment.

        Args:
            gain: Gain value (sensor-dependent, typically 0-300). Higher values
                  increase sensitivity but also increase read noise.
        """
        self.gain = gain
        if self.is_connected and self.camera:
            try:
                if hasattr(self.camera, 'Gain'):
                    self.camera.Gain = gain
            except Exception:
                pass

    def set_binning(self, binning: int):
        """Set the camera pixel binning factor.

        Updates the stored binning value and applies it immediately if the camera
        is currently connected and supports binning. Binning combines adjacent
        pixels (e.g., 2x2 binning combines 4 pixels into 1), which increases
        sensitivity and reduces image download time at the cost of resolution.

        Args:
            binning: Binning factor (1 = no binning, 2 = 2x2, 4 = 4x4).
        """
        self.binning = binning
        if self.is_connected and self.camera:
            try:
                if self.camera.CanSetBinning:
                    self.camera.BinX = binning
                    self.camera.BinY = binning
            except Exception:
                pass

    def capture(self, exposure_sec: float, save_path: str) -> bool:
        """Capture a single image from the camera and save it to disk.

        Starts an exposure, waits for it to complete (with timeout), retrieves
        the image data from the camera, and saves it in FITS or PNG format.

        The ASCOM ImageArray returns a 2D array that is often transposed relative
        to the expected row-major orientation, so we transpose it before saving.

        Args:
            exposure_sec: Exposure duration in seconds.
            save_path: Absolute path where the image will be saved. The file
                       extension determines the format: .fits/.fit for FITS
                       (preserving 16-bit data), anything else for PNG (8-bit).

        Returns:
            True if the image was captured and saved successfully.
            False if the camera is not connected, the exposure timed out,
            or an error occurred.
        """
        if not self.is_connected or not self.camera:
            return False

        try:
            # Start the camera exposure. The second argument (True) indicates
            # a "light" frame (as opposed to a dark/bias frame).
            self.camera.StartExposure(exposure_sec, True)

            # Wait for the exposure to complete with a generous timeout.
            # The 10-second margin accounts for camera download time and overhead.
            timeout = exposure_sec + 10
            start = time.time()

            while not self.camera.ImageReady:
                if time.time() - start > timeout:
                    self._log("Capture timeout")
                    self.camera.AbortExposure()
                    return False
                time.sleep(0.05)  # Poll every 50ms to avoid busy-waiting

            # Retrieve the raw image data from the camera's buffer
            image_data = self.camera.ImageArray

            # Convert to numpy array for processing and saving
            import numpy as np

            # ASCOM ImageArray is typically column-major (Fortran order),
            # so we transpose to get standard row-major (C order) layout.
            arr = np.array(image_data, dtype=np.uint16)
            if arr.ndim == 2:
                arr = arr.T  # Transpose from column-major to row-major

            # Save as FITS to preserve full 16-bit dynamic range (preferred for astrometry)
            if save_path.lower().endswith(('.fits', '.fit')):
                try:
                    from astropy.io import fits
                    hdu = fits.PrimaryHDU(arr)
                    hdu.writeto(save_path, overwrite=True)
                except ImportError:
                    # Fallback to PNG if astropy is not available
                    save_path = save_path.rsplit('.', 1)[0] + '.png'
                    self._save_as_png(arr, save_path)
            else:
                self._save_as_png(arr, save_path)

            return True

        except Exception as e:
            self._log(f"Capture error: {e}")
            return False

    def _save_as_png(self, arr, save_path: str):
        """Save a numpy array as an 8-bit PNG image.

        Performs min-max normalization to scale the full dynamic range of the
        16-bit input array to the 0-255 range suitable for PNG.

        Args:
            arr: 2D numpy array (typically uint16) containing the image data.
            save_path: Absolute path where the PNG file will be written.
        """
        import numpy as np
        from PIL import Image

        # Normalize from the full data range to 8-bit (0-255).
        # This stretches the image contrast to use the full 8-bit range.
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max > arr_min:
            arr = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
        else:
            # Uniform image (all pixels same value) - output black
            arr = np.zeros_like(arr, dtype=np.uint8)

        img = Image.fromarray(arr)
        img.save(save_path)

    def abort(self):
        """Abort an exposure currently in progress.

        Safe to call even if no exposure is active; errors are silently ignored.
        """
        if self.is_connected and self.camera:
            try:
                self.camera.AbortExposure()
            except Exception:
                pass

    def disconnect(self):
        """Disconnect from the ASCOM camera and release the COM object.

        Sets Connected=False on the ASCOM driver and clears the internal
        camera reference. Safe to call even if not connected.
        """
        if self.camera:
            try:
                if self.is_connected:
                    self.camera.Connected = False
                    self._log("Camera disconnected")
            except Exception:
                pass
        self.camera = None
        self.is_connected = False
