"""
Comprehensive tests for the Android bridge layer.

Validates that:
  1. All Python modules import from the Android source tree
  2. Monkey-patching replaces ASTAP subprocess with cloud solver
  3. Monkey-patching replaces OpenCV camera with Android camera bridge
  4. Cloud solver handles online/offline/error states correctly
  5. Camera bridge API is complete and safe when no Java backend exists
  6. Serial bridge API is complete and safe when no Java backend exists
  7. HeadlessTelescopeApp can start with Android patches applied
  8. Stub modules behave correctly (no crashes on attribute access)
  9. Config generation works for first-launch
  10. All bridge modules are JSON-serializable (for Chaquopy Java<->Python)

Run from project root:
    python -m pytest "android app/app/src/test/python/test_android_bridge.py" -v
Or without pytest:
    python -m unittest discover -s "android app/app/src/test/python" -v
"""

import os
import sys
import json
import time
import types
import shutil
import tempfile
import unittest
import threading
from unittest.mock import MagicMock, patch
from dataclasses import asdict

# ── Set up paths ──────────────────────────────────────────────────────
# The Android Python source tree and the project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ANDROID_PYTHON = os.path.normpath(
    os.path.join(_SCRIPT_DIR, "..", "..", "main", "python")
)
_PROJECT_ROOT = os.path.normpath(
    os.path.join(_SCRIPT_DIR, "..", "..", "..", "..", "..", "..")
)

# Insert Android source tree FIRST so android_bridge is found there
if _ANDROID_PYTHON not in sys.path:
    sys.path.insert(0, _ANDROID_PYTHON)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(1, _PROJECT_ROOT)

# ── Pre-stub Windows-only modules ─────────────────────────────────────
for _mod in ("win32com", "win32com.client", "pythoncom", "winreg",
             "pywintypes", "win32api"):
    if _mod not in sys.modules:
        _stub = types.ModuleType(_mod)
        _stub.__path__ = []
        sys.modules[_mod] = _stub


# ═══════════════════════════════════════════════════════════════════════
#  Test 1: All Python modules importable from Android source tree
# ═══════════════════════════════════════════════════════════════════════
class TestAndroidImports(unittest.TestCase):
    """Every Python module in the Android source tree must import."""

    def test_bridge_package_imports(self):
        from android_bridge import cloud_solver, camera_bridge, serial_bridge
        self.assertIsNotNone(cloud_solver)
        self.assertIsNotNone(camera_bridge)
        self.assertIsNotNone(serial_bridge)

    def test_bridge_main_imports(self):
        from android_bridge.main import start_server, stop_server, is_running
        self.assertTrue(callable(start_server))
        self.assertTrue(callable(stop_server))
        self.assertTrue(callable(is_running))

    def test_core_modules_import(self):
        """All shared Python modules must import without error."""
        modules = [
            "kalman_filter", "drift_ml", "software_pec",
            "lx200_protocol", "config_manager", "session_recorder",
            "telescope_logger", "telescope_simulator", "weather_service",
            "catalog_loader", "crash_recovery",
        ]
        for mod_name in modules:
            with self.subTest(module=mod_name):
                mod = __import__(mod_name)
                self.assertIsNotNone(mod, f"{mod_name} imported as None")

    def test_heavy_modules_import(self):
        """Modules with complex dependencies (numpy, etc.) must import."""
        from realtime_tracking import RealTimeTrackingController, FastPlateSolver
        from auto_platesolve import AutoPlateSolver, SolveResult
        from HEADLESS_SERVER import HeadlessTelescopeApp
        self.assertIsNotNone(RealTimeTrackingController)
        self.assertIsNotNone(AutoPlateSolver)
        self.assertIsNotNone(HeadlessTelescopeApp)

    def test_web_server_imports(self):
        """web_server must import (depends on flask)."""
        from web_server import TelescopeWebServer
        self.assertIsNotNone(TelescopeWebServer)


# ═══════════════════════════════════════════════════════════════════════
#  Test 2: Cloud Solver
# ═══════════════════════════════════════════════════════════════════════
class TestCloudSolver(unittest.TestCase):
    """Test the cloud plate solver replacement for ASTAP."""

    def test_solver_instantiation(self):
        from android_bridge.cloud_solver import CloudPlateSolver
        solver = CloudPlateSolver(api_key="test_key", timeout=15)
        self.assertEqual(solver.api_key, "test_key")
        self.assertEqual(solver.timeout, 15)
        self.assertTrue(solver._online)

    def test_offline_returns_failure(self):
        from android_bridge.cloud_solver import CloudPlateSolver
        solver = CloudPlateSolver()
        solver._online = False
        result = solver.solve("/tmp/fake.png")
        self.assertFalse(result.success)
        self.assertEqual(result.source, "offline")
        self.assertIn("offline", result.error.lower())

    def test_result_dataclass_fields(self):
        from android_bridge.cloud_solver import CloudSolveResult
        r = CloudSolveResult(
            success=True, ra_hours=12.5, dec_degrees=45.0,
            solve_time_ms=2500.0, field_w=1.2, field_h=0.9,
            source="cloud"
        )
        self.assertEqual(r.ra_hours, 12.5)
        self.assertEqual(r.dec_degrees, 45.0)
        self.assertEqual(r.solve_time_ms, 2500.0)
        self.assertTrue(r.success)

    def test_result_json_serializable(self):
        from android_bridge.cloud_solver import CloudSolveResult
        from dataclasses import asdict
        r = CloudSolveResult(success=True, ra_hours=6.0, dec_degrees=-30.0)
        d = asdict(r)
        s = json.dumps(d)
        self.assertIn('"ra_hours": 6.0', s)

    def test_cloud_solve_function_returns_none_when_offline(self):
        from android_bridge.cloud_solver import cloud_solve, get_solver
        solver = get_solver()
        solver._online = False
        result = cloud_solve("/tmp/fake.png")
        self.assertIsNone(result)
        # Reset for other tests
        solver._online = True

    def test_cloud_solve_function_signature(self):
        """cloud_solve returns (ra, dec, time_ms) or None -- matches FastPlateSolver."""
        import inspect
        from android_bridge.cloud_solver import cloud_solve
        sig = inspect.signature(cloud_solve)
        params = list(sig.parameters.keys())
        self.assertIn("image_path", params)
        self.assertIn("hint_ra", params)
        self.assertIn("hint_dec", params)
        self.assertIn("search_radius", params)

    def test_check_online_sets_flag(self):
        from android_bridge.cloud_solver import CloudPlateSolver
        solver = CloudPlateSolver()
        # Without network, check_online should set _online to False
        # (we can't guarantee network, but the method shouldn't crash)
        result = solver.check_online()
        self.assertIsInstance(result, bool)
        self.assertEqual(result, solver._online)


# ═══════════════════════════════════════════════════════════════════════
#  Test 3: Monkey-patching AutoPlateSolver
# ═══════════════════════════════════════════════════════════════════════
class TestAutoSolverPatching(unittest.TestCase):
    """Verify _solve_image can be replaced with cloud solver."""

    def test_original_method_exists(self):
        from auto_platesolve import AutoPlateSolver
        self.assertTrue(hasattr(AutoPlateSolver, '_solve_image'))

    def test_patch_replaces_method(self):
        from auto_platesolve import AutoPlateSolver, SolveResult

        original = AutoPlateSolver._solve_image

        def fake_solve(self, image_path):
            return SolveResult(success=True, ra_hours=10.0, dec_degrees=20.0,
                               solve_time_ms=500.0, image_path=image_path)

        try:
            AutoPlateSolver._solve_image = fake_solve
            solver = AutoPlateSolver()
            result = solver._solve_image("/tmp/test.png")
            self.assertTrue(result.success)
            self.assertEqual(result.ra_hours, 10.0)
            self.assertEqual(result.dec_degrees, 20.0)
        finally:
            AutoPlateSolver._solve_image = original

    def test_full_patch_via_bridge(self):
        """The actual _patch_plate_solvers applies correctly."""
        from auto_platesolve import AutoPlateSolver
        from android_bridge.main import _patch_plate_solvers

        original = AutoPlateSolver._solve_image

        try:
            _patch_plate_solvers()
            # After patching, _solve_image should NOT be the original
            self.assertIsNot(AutoPlateSolver._solve_image, original,
                             "Patch did not replace _solve_image")

            # The patched method should be callable
            solver = AutoPlateSolver()
            # It will fail (no network / no image) but should not crash
            result = solver._solve_image("/tmp/nonexistent.png")
            self.assertIsNotNone(result)
            self.assertFalse(result.success)  # No image to solve
        finally:
            AutoPlateSolver._solve_image = original


# ═══════════════════════════════════════════════════════════════════════
#  Test 4: Monkey-patching FastPlateSolver
# ═══════════════════════════════════════════════════════════════════════
class TestFastSolverPatching(unittest.TestCase):
    """Verify FastPlateSolver.solve_fast can be replaced."""

    def test_original_method_exists(self):
        from realtime_tracking import FastPlateSolver
        self.assertTrue(hasattr(FastPlateSolver, 'solve_fast'))

    def test_patch_replaces_method(self):
        from realtime_tracking import FastPlateSolver

        original = FastPlateSolver.solve_fast

        def fake_fast(self, image_path):
            return (12.0, 45.0, 200.0)

        try:
            FastPlateSolver.solve_fast = fake_fast
            solver = FastPlateSolver()
            result = solver.solve_fast("/tmp/test.png")
            self.assertEqual(result, (12.0, 45.0, 200.0))
        finally:
            FastPlateSolver.solve_fast = original

    def test_full_patch_via_bridge(self):
        from realtime_tracking import FastPlateSolver
        from android_bridge.main import _patch_plate_solvers

        original = FastPlateSolver.solve_fast

        try:
            _patch_plate_solvers()
            self.assertIsNot(FastPlateSolver.solve_fast, original,
                             "Patch did not replace solve_fast")

            solver = FastPlateSolver()
            # Will return None (offline / no image) but shouldn't crash
            result = solver.solve_fast("/tmp/nonexistent.png")
            self.assertIsNone(result)
        finally:
            FastPlateSolver.solve_fast = original


# ═══════════════════════════════════════════════════════════════════════
#  Test 5: Camera Bridge
# ═══════════════════════════════════════════════════════════════════════
class TestCameraBridge(unittest.TestCase):
    """Test the camera bridge Python interface."""

    def test_no_manager_returns_none(self):
        from android_bridge import camera_bridge
        camera_bridge._camera_manager = None
        self.assertIsNone(camera_bridge.capture_for_solving())
        self.assertIsNone(camera_bridge.get_jpeg_frame())
        self.assertEqual(camera_bridge.list_cameras(), [])

    def test_set_manager_stores_reference(self):
        from android_bridge import camera_bridge
        mock_manager = MagicMock()
        camera_bridge.set_camera_manager(mock_manager)
        self.assertIs(camera_bridge._camera_manager, mock_manager)
        # Cleanup
        camera_bridge._camera_manager = None

    def test_capture_calls_java(self):
        from android_bridge import camera_bridge
        mock_manager = MagicMock()
        mock_manager.captureImage.return_value = "/tmp/captured.png"

        camera_bridge._camera_manager = mock_manager

        # Create a fake file so the existence check passes
        tmp = tempfile.mktemp(suffix=".png")
        mock_manager.captureImage.return_value = tmp
        open(tmp, 'w').close()

        try:
            result = camera_bridge.capture_for_solving(save_dir="/tmp")
            mock_manager.captureImage.assert_called_once()
            self.assertEqual(result, tmp)
        finally:
            os.unlink(tmp)
            camera_bridge._camera_manager = None

    def test_get_jpeg_frame_returns_bytes(self):
        from android_bridge import camera_bridge
        mock_manager = MagicMock()
        mock_manager.getJpegFrame.return_value = b"\xff\xd8\xff\xe0"

        camera_bridge._camera_manager = mock_manager
        result = camera_bridge.get_jpeg_frame()
        self.assertIsInstance(result, bytes)
        self.assertEqual(result, b"\xff\xd8\xff\xe0")
        camera_bridge._camera_manager = None

    def test_open_phone_camera_no_manager(self):
        from android_bridge import camera_bridge
        camera_bridge._camera_manager = None
        self.assertFalse(camera_bridge.open_phone_camera())

    def test_close_camera_no_crash_without_manager(self):
        from android_bridge import camera_bridge
        camera_bridge._camera_manager = None
        # Should not raise
        camera_bridge.close_camera()

    def test_set_exposure_no_crash_without_manager(self):
        from android_bridge import camera_bridge
        camera_bridge._camera_manager = None
        camera_bridge.set_exposure(500)
        camera_bridge.set_gain(100)


# ═══════════════════════════════════════════════════════════════════════
#  Test 6: Serial Bridge
# ═══════════════════════════════════════════════════════════════════════
class TestSerialBridge(unittest.TestCase):
    """Test the USB serial bridge Python interface."""

    def test_no_manager_returns_defaults(self):
        from android_bridge import serial_bridge
        serial_bridge._serial_manager = None
        self.assertEqual(serial_bridge.list_ports(), [])
        self.assertFalse(serial_bridge.connect())
        self.assertEqual(serial_bridge.send(":GR#"), "")
        self.assertFalse(serial_bridge.is_connected())

    def test_disconnect_no_crash_without_manager(self):
        from android_bridge import serial_bridge
        serial_bridge._serial_manager = None
        serial_bridge.disconnect()  # Should not raise

    def test_set_manager(self):
        from android_bridge import serial_bridge
        mock = MagicMock()
        serial_bridge.set_serial_manager(mock)
        self.assertIs(serial_bridge._serial_manager, mock)
        serial_bridge._serial_manager = None

    def test_connect_calls_java(self):
        from android_bridge import serial_bridge
        mock = MagicMock()
        mock.connect.return_value = True
        serial_bridge._serial_manager = mock

        result = serial_bridge.connect(0, 9600)
        mock.connect.assert_called_once_with(0, 9600)
        self.assertTrue(result)
        serial_bridge._serial_manager = None

    def test_send_calls_java(self):
        from android_bridge import serial_bridge
        mock = MagicMock()
        mock.send.return_value = "12:30:00#"
        serial_bridge._serial_manager = mock

        response = serial_bridge.send(":GR#")
        mock.send.assert_called_once_with(":GR#")
        self.assertEqual(response, "12:30:00#")
        serial_bridge._serial_manager = None

    def test_android_serial_port_shim(self):
        from android_bridge.serial_bridge import AndroidSerialPort
        port = AndroidSerialPort()
        self.assertFalse(port.is_open)
        # write/close should not crash when not connected
        self.assertEqual(port.write(b":GR#"), 0)
        port.close()

    def test_android_serial_port_context_manager(self):
        from android_bridge.serial_bridge import AndroidSerialPort
        with AndroidSerialPort() as port:
            self.assertFalse(port.is_open)


# ═══════════════════════════════════════════════════════════════════════
#  Test 7: Stub Modules
# ═══════════════════════════════════════════════════════════════════════
class TestStubModules(unittest.TestCase):
    """Test that stubbed Windows modules don't crash on access."""

    def test_stub_module_creation(self):
        from android_bridge.main import _stub_module
        _stub_module("test_fake_module_12345")
        import test_fake_module_12345
        self.assertIsNotNone(test_fake_module_12345)
        del sys.modules["test_fake_module_12345"]

    def test_stub_attribute_access(self):
        from android_bridge.main import _stub_module
        _stub_module("test_stub_attr_123")
        import test_stub_attr_123
        # Any attribute access should return a stub, not crash
        result = test_stub_attr_123.Dispatch("SomeCamera")
        self.assertIsNotNone(result)
        # Chained access
        result2 = test_stub_attr_123.client.Dispatch("X").Connected
        self.assertIsNotNone(result2)
        del sys.modules["test_stub_attr_123"]

    def test_stub_bool_is_false(self):
        from android_bridge.main import _stub_module
        _stub_module("test_stub_bool_123")
        import test_stub_bool_123
        obj = test_stub_bool_123.SomeClass()
        self.assertFalse(bool(obj))
        del sys.modules["test_stub_bool_123"]

    def test_stub_iteration(self):
        from android_bridge.main import _stub_module
        _stub_module("test_stub_iter_123")
        import test_stub_iter_123
        obj = test_stub_iter_123.comports()
        self.assertEqual(list(obj), [])
        del sys.modules["test_stub_iter_123"]


# ═══════════════════════════════════════════════════════════════════════
#  Test 8: Config Generation
# ═══════════════════════════════════════════════════════════════════════
class TestConfigGeneration(unittest.TestCase):
    """Test default config file creation for first launch."""

    def test_creates_valid_json(self):
        from android_bridge.main import _create_default_config
        tmp = tempfile.mktemp(suffix=".json")
        try:
            _create_default_config(tmp)
            with open(tmp) as f:
                config = json.load(f)

            self.assertIn("location", config)
            self.assertIn("connection", config)
            self.assertIn("camera", config)
            self.assertIn("tracking", config)
            self.assertIn("solver", config)
            self.assertEqual(config["platform"], "android")
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_config_has_wifi_defaults(self):
        from android_bridge.main import _create_default_config
        tmp = tempfile.mktemp(suffix=".json")
        try:
            _create_default_config(tmp)
            with open(tmp) as f:
                config = json.load(f)

            self.assertEqual(config["connection"]["type"], "wifi")
            self.assertEqual(config["connection"]["wifi_port"], 9996)
            self.assertEqual(config["solver"]["mode"], "auto")
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_config_solver_section(self):
        from android_bridge.main import _create_default_config
        tmp = tempfile.mktemp(suffix=".json")
        try:
            _create_default_config(tmp)
            with open(tmp) as f:
                config = json.load(f)

            solver = config["solver"]
            self.assertEqual(solver["mode"], "auto")
            self.assertIn("cloud_api_key", solver)
            self.assertIn("search_radius", solver)
            self.assertIn("timeout", solver)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


# ═══════════════════════════════════════════════════════════════════════
#  Test 9: HeadlessTelescopeApp Integration
# ═══════════════════════════════════════════════════════════════════════
class TestHeadlessIntegration(unittest.TestCase):
    """Test that HeadlessTelescopeApp can be created with Android patches."""

    def test_app_instantiation_with_wifi(self):
        """HeadlessTelescopeApp creates successfully in WiFi mode."""
        from HEADLESS_SERVER import HeadlessTelescopeApp
        app = HeadlessTelescopeApp(
            connection_type="wifi",
            wifi_ip="192.168.1.1",
            wifi_port=9999,
            web_port=18080,
        )
        self.assertIsNotNone(app)
        self.assertIsNotNone(app.tracking)
        self.assertIsNotNone(app.plate_solver)
        self.assertIsNotNone(app.auto_solver)
        self.assertIsNotNone(app.protocol)

    def test_tracking_controller_works(self):
        """RealTimeTrackingController is functional after import."""
        from realtime_tracking import RealTimeTrackingController
        ctrl = RealTimeTrackingController()
        ctrl.send_command = lambda cmd: "1"
        ctrl.on_log = lambda msg: None
        ctrl.kalman.initialize(45.0, 180.0)
        ctrl.current_alt = 45.0
        ctrl.current_az = 180.0
        stats = ctrl.get_stats()
        self.assertIn("total_corrections", stats)
        self.assertIn("kalman_rms_arcsec", stats)

    def test_kalman_filter_works(self):
        """Kalman filter is functional (numpy dependency)."""
        from kalman_filter import TelescopeKalmanFilter
        kf = TelescopeKalmanFilter()
        kf.initialize(45.0, 180.0)
        state = kf.update(45.001, 180.001)
        self.assertIsNotNone(state)
        vel = kf.get_velocity()
        self.assertEqual(len(vel), 2)

    def test_ml_predictor_works(self):
        """ML drift predictor is functional (numpy dependency)."""
        from drift_ml import DriftPredictor
        pred = DriftPredictor()
        for i in range(50):
            pred.add_sample(
                alt=45.0 + i * 0.001,
                az=180.0,
                drift_alt=0.5,
                drift_az=-0.3,
            )
        result = pred.predict(45.025, 180.0)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    def test_software_pec_works(self):
        """Software PEC is functional (numpy FFT)."""
        from software_pec import SoftwarePEC
        pec = SoftwarePEC()
        stats = pec.get_statistics()
        self.assertIn("total_samples", stats)


# ═══════════════════════════════════════════════════════════════════════
#  Test 10: Web Server Camera Patching
# ═══════════════════════════════════════════════════════════════════════
class TestWebServerPatching(unittest.TestCase):
    """Test that web_server camera methods can be patched."""

    def test_web_server_has_patchable_methods(self):
        from web_server import TelescopeWebServer
        self.assertTrue(hasattr(TelescopeWebServer, '_open_uvc_camera'))
        self.assertTrue(hasattr(TelescopeWebServer, '_generate_mjpeg'))
        self.assertTrue(hasattr(TelescopeWebServer, '_close_camera'))

    def test_patch_applies(self):
        from web_server import TelescopeWebServer
        from android_bridge.main import _patch_web_server_camera

        original_open = TelescopeWebServer._open_uvc_camera
        original_mjpeg = TelescopeWebServer._generate_mjpeg
        original_close = TelescopeWebServer._close_camera

        try:
            _patch_web_server_camera()
            self.assertIsNot(TelescopeWebServer._open_uvc_camera, original_open)
            self.assertIsNot(TelescopeWebServer._generate_mjpeg, original_mjpeg)
            self.assertIsNot(TelescopeWebServer._close_camera, original_close)
        finally:
            TelescopeWebServer._open_uvc_camera = original_open
            TelescopeWebServer._generate_mjpeg = original_mjpeg
            TelescopeWebServer._close_camera = original_close


# ═══════════════════════════════════════════════════════════════════════
#  Test 11: End-to-End Bridge Startup Simulation
# ═══════════════════════════════════════════════════════════════════════
class TestBridgeStartup(unittest.TestCase):
    """Simulate the Android bridge startup sequence without Flask."""

    def test_patch_for_android(self):
        """_patch_for_android completes without errors."""
        from android_bridge.main import _patch_for_android
        tmp_data = tempfile.mkdtemp(prefix="tc_test_data_")
        tmp_cache = tempfile.mkdtemp(prefix="tc_test_cache_")
        original_cwd = os.getcwd()

        try:
            _patch_for_android(tmp_data, tmp_cache)

            # Verify cwd changed to data_dir
            self.assertEqual(os.path.realpath(os.getcwd()),
                             os.path.realpath(tmp_data))

            # Verify tempdir redirect
            import tempfile as tf
            self.assertEqual(tf.tempdir, tmp_cache)

            # Verify Windows-only modules are stubbed
            self.assertIn("win32com", sys.modules)
            self.assertIn("pythoncom", sys.modules)
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(tmp_data, ignore_errors=True)
            shutil.rmtree(tmp_cache, ignore_errors=True)
            # Reset tempdir
            import tempfile as tf
            tf.tempdir = None

    def test_is_running_initially_false(self):
        from android_bridge.main import is_running
        self.assertFalse(is_running())


if __name__ == "__main__":
    unittest.main()
