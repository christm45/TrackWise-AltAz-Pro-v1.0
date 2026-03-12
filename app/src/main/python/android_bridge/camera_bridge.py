"""
Camera bridge -- Python interface to the Android CameraManager.

Provides the same capture interface that auto_platesolve.py expects,
but delegates to the Kotlin CameraManager via Chaquopy's Java bridge.

On desktop:
    auto_platesolve uses cv2.VideoCapture or ASCOMCameraCapture
On Android:
    auto_platesolve uses this bridge, which calls CameraManager.captureImage()

The Kotlin side supports:
    - Phone camera (Camera2 API)
    - USB UVC webcams (UVCCamera library)
    - ZWO ASI cameras (ZWO Android SDK)

Integration:
    The android_bridge.main module patches auto_platesolve to use this
    bridge instead of OpenCV / ASCOM when running on Android.
"""

import os
import time
import logging
import tempfile
from typing import Optional

logger = logging.getLogger("camera_bridge")

# The Kotlin CameraManager instance, set by the Android service at startup
_camera_manager = None


def set_camera_manager(manager):
    """
    Called from Kotlin at startup to inject the CameraManager instance.

    In Kotlin:
        val py = Python.getInstance()
        val bridge = py.getModule("android_bridge.camera_bridge")
        bridge.callAttr("set_camera_manager", cameraManager)
    """
    global _camera_manager
    _camera_manager = manager
    logger.info("Camera manager set from Android")


def capture_for_solving(
    save_dir: Optional[str] = None,
    filename: str = "plate_solve.png"
) -> Optional[str]:
    """
    Capture a single frame for plate solving.

    Returns the path to the saved image, or None on failure.
    This is the function that auto_platesolve._capture_from_camera()
    is patched to call on Android.
    """
    if _camera_manager is None:
        logger.warning("No camera manager available")
        return None

    if save_dir is None:
        save_dir = os.environ.get(
            "TELESCOPE_TEMP_DIR",
            tempfile.gettempdir()
        )

    save_path = os.path.join(save_dir, filename)

    try:
        result = _camera_manager.captureImage(save_path)
        if result is not None:
            # Chaquopy returns Java String -- convert to Python str
            path = str(result)
            if os.path.exists(path):
                size_kb = os.path.getsize(path) / 1024
                logger.info(f"Captured frame: {path} ({size_kb:.0f} KB)")
                return path
            else:
                logger.warning(f"Capture returned path but file missing: {path}")
                return None
        else:
            logger.warning("Camera capture returned null")
            return None
    except Exception as e:
        logger.error(f"Camera capture failed: {e}")
        return None


def get_jpeg_frame() -> Optional[bytes]:
    """
    Get the latest camera frame as JPEG bytes for MJPEG live streaming.

    Called by web_server.py's MJPEG generator on Android instead of
    cv2.VideoCapture.read().
    """
    if _camera_manager is None:
        return None

    try:
        jpeg = _camera_manager.getJpegFrame()
        if jpeg is not None:
            # Chaquopy returns Java byte[] -- convert to Python bytes
            return bytes(jpeg)
        return None
    except Exception as e:
        logger.error(f"Error getting JPEG frame: {e}")
        return None


def list_cameras() -> list:
    """
    List available cameras.

    Returns a list of dicts: [{"id": "0", "name": "Rear Camera", "source": "phone"}, ...]
    """
    if _camera_manager is None:
        return []

    try:
        cameras = _camera_manager.listCameras()
        # Convert Java List<Map> to Python list of dicts
        return [dict(cam) for cam in cameras]
    except Exception as e:
        logger.error(f"Error listing cameras: {e}")
        return []


def open_phone_camera() -> bool:
    """Open the phone's rear camera for plate solving."""
    if _camera_manager is None:
        return False
    try:
        return bool(_camera_manager.openPhoneCamera())
    except Exception as e:
        logger.error(f"Error opening phone camera: {e}")
        return False


def open_uvc_camera(vendor_id: int = 0, product_id: int = 0) -> bool:
    """
    Open a USB UVC camera for plate solving / live view.

    Works with generic USB webcams and ZWO ASI planetary cameras that
    expose a UVC interface (ASI120, ASI224, ASI290, ASI385, ASI462, etc.).

    Args:
        vendor_id:  USB vendor ID (0 = first available UVC device)
        product_id: USB product ID (0 = any product from that vendor)

    Returns:
        True if the camera was successfully opened.
    """
    if _camera_manager is None:
        logger.warning("No camera manager -- cannot open UVC camera")
        return False
    try:
        result = _camera_manager.openUvcCamera(int(vendor_id), int(product_id))
        success = bool(result)
        if success:
            name = str(_camera_manager.getActiveSourceName())
            logger.info(f"UVC camera opened: {name}")
        else:
            logger.warning(f"Failed to open UVC camera (vendor={vendor_id}, product={product_id})")
        return success
    except Exception as e:
        logger.error(f"Error opening UVC camera: {e}")
        return False


def open_zwo_camera() -> bool:
    """
    Open the first ZWO ASI camera found.

    Strategy:
      1. Try native ASI SDK (preferred for ALL ASI cameras)
         - Full camera control (exposure, gain, ROI, binning, cooler)
         - Works with all ASI cameras: ASI120MC, ASI224MC, ASI294MC Pro, etc.
      2. Fall back to UVC mode only if SDK init fails
         - Limited control (MJPEG only, no RAW, no fine exposure)

    Returns:
        True if a ZWO camera was found and opened.
    """
    if _camera_manager is None:
        logger.warning("No camera manager -- cannot open ZWO camera")
        return False
    try:
        # Try native ASI SDK first (full control)
        if bool(_camera_manager.isAsiSdkAvailable()):
            logger.info("ASI SDK available, attempting SDK mode...")
            result = _camera_manager.openZwoCameraSDK()
            if bool(result):
                name = str(_camera_manager.getActiveSourceName())
                logger.info(f"ZWO ASI camera opened via SDK: {name}")
                return True
            logger.info("ASI SDK open failed, trying UVC fallback...")

        # Fall back to UVC mode
        result = _camera_manager.openZwoCamera()
        success = bool(result)
        if success:
            name = str(_camera_manager.getActiveSourceName())
            logger.info(f"ZWO ASI camera opened via UVC: {name}")
        else:
            logger.warning("No ZWO ASI camera found")
        return success
    except Exception as e:
        logger.error(f"Error opening ZWO camera: {e}")
        return False


def get_active_source() -> str:
    """
    Get the name of the currently active camera source.

    Returns one of: "None", "Phone Camera", "USB Camera (...)",
    "ZWO ASI (0x...)"
    """
    if _camera_manager is None:
        return "None"
    try:
        return str(_camera_manager.getActiveSourceName())
    except Exception as e:
        logger.error(f"Error getting active source: {e}")
        return "None"


def set_exposure(ms: int):
    """Set camera exposure time in milliseconds."""
    if _camera_manager is not None:
        try:
            _camera_manager.setExposureMs(int(ms))
        except Exception as e:
            logger.error(f"Error setting exposure: {e}")


def set_gain(value: int):
    """Set camera gain/ISO."""
    if _camera_manager is not None:
        try:
            _camera_manager.setGain(int(value))
        except Exception as e:
            logger.error(f"Error setting gain: {e}")


def close_camera():
    """Close the active camera."""
    if _camera_manager is not None:
        try:
            _camera_manager.close()
        except Exception as e:
            logger.error(f"Error closing camera: {e}")


# ═══════════════════════════════════════════════════════════════════════
#  ASI SDK-specific controls (only work when camera is opened via SDK)
# ═══════════════════════════════════════════════════════════════════════

def is_asi_sdk_available() -> bool:
    """Check if the ASI SDK native libraries are loaded."""
    if _camera_manager is None:
        return False
    try:
        return bool(_camera_manager.isAsiSdkAvailable())
    except Exception:
        return False


def is_asi_sdk_active() -> bool:
    """Check if the current camera is using the ASI SDK (not UVC)."""
    if _camera_manager is None:
        return False
    try:
        source = str(_camera_manager.getActiveSourceName())
        return "(SDK)" in source
    except Exception:
        return False


def get_asi_temperature() -> float:
    """
    Get the sensor temperature in degrees Celsius.
    Returns -999 if not available (no cooled camera or not SDK mode).
    """
    if _camera_manager is None:
        return -999.0
    try:
        return float(_camera_manager.getAsiTemperature())
    except Exception:
        return -999.0


def set_asi_roi(width: int, height: int, bin: int = 1, img_type: int = 0) -> bool:
    """
    Set the ASI SDK Region of Interest.

    Args:
        width:    Width in pixels (must be multiple of 8)
        height:   Height in pixels (must be multiple of 8)
        bin:      Binning factor (1, 2, 3, or 4)
        img_type: 0=RAW8, 1=RGB24, 2=RAW16, 3=Y8

    Returns:
        True on success. Stops/restarts video capture internally.
    """
    if _camera_manager is None:
        return False
    try:
        return bool(_camera_manager.setAsiROI(int(width), int(height), int(bin), int(img_type)))
    except Exception as e:
        logger.error(f"Error setting ASI ROI: {e}")
        return False


def set_asi_cooler_target(temp_c: int) -> bool:
    """
    Set the cooler target temperature (cooled cameras only).
    Also enables the cooler.

    Args:
        temp_c: Target temperature in degrees Celsius (e.g. -20)
    """
    if _camera_manager is None:
        return False
    try:
        return bool(_camera_manager.setAsiCoolerTarget(int(temp_c)))
    except Exception as e:
        logger.error(f"Error setting cooler target: {e}")
        return False


def set_asi_control(control_type: int, value: int) -> bool:
    """
    Set any ASI SDK control value.

    Control types (from ASICameraSDK.kt):
        0  = GAIN           6  = BANDWIDTH_OVERLOAD
        1  = EXPOSURE (µs)  8  = TEMPERATURE (read-only)
        2  = GAMMA          9  = FLIP
        3  = WB_R           14 = HIGH_SPEED_MODE
        4  = WB_B           15 = COOLER_POWER (read-only)
        5  = OFFSET         16 = TARGET_TEMP
                            17 = COOLER_ON

    Args:
        control_type: ASI control type ID
        value: Control value (int)

    Returns:
        True on success
    """
    if _camera_manager is None:
        return False
    try:
        sdk = _camera_manager.getAsiSdk()
        if sdk is None:
            return False
        return bool(sdk.setControlValue(int(control_type), int(value), False))
    except Exception as e:
        logger.error(f"Error setting ASI control {control_type}={value}: {e}")
        return False


def get_asi_control(control_type: int) -> int:
    """
    Get an ASI SDK control value.

    Args:
        control_type: ASI control type ID (see set_asi_control for list)

    Returns:
        Current control value, or -1 on error.
    """
    if _camera_manager is None:
        return -1
    try:
        sdk = _camera_manager.getAsiSdk()
        if sdk is None:
            return -1
        return int(sdk.getControlValue(int(control_type)))
    except Exception as e:
        logger.error(f"Error getting ASI control {control_type}: {e}")
        return -1


# ASI control type constants (mirror ASICameraSDK.kt)
ASI_CTRL_GAIN = 0
ASI_CTRL_EXPOSURE = 1       # microseconds
ASI_CTRL_GAMMA = 2
ASI_CTRL_WB_R = 3
ASI_CTRL_WB_B = 4
ASI_CTRL_OFFSET = 5
ASI_CTRL_BANDWIDTH = 6
ASI_CTRL_TEMPERATURE = 8    # read-only, value / 10 = °C
ASI_CTRL_FLIP = 9
ASI_CTRL_HIGH_SPEED = 14
ASI_CTRL_COOLER_POWER = 15  # read-only
ASI_CTRL_TARGET_TEMP = 16
ASI_CTRL_COOLER_ON = 17


def get_asi_all_controls() -> dict:
    """
    Read all major ASI control values at once.

    Returns dict with keys: exposure_us, gain, gamma, offset, flip,
    bandwidth, high_speed, temperature, wb_r, wb_b.
    """
    if _camera_manager is None:
        return {}
    try:
        sdk = _camera_manager.getAsiSdk()
        if sdk is None:
            return {}
        result = {}
        for name, ctrl_id in [
            ("exposure_us", ASI_CTRL_EXPOSURE),
            ("gain", ASI_CTRL_GAIN),
            ("gamma", ASI_CTRL_GAMMA),
            ("offset", ASI_CTRL_OFFSET),
            ("flip", ASI_CTRL_FLIP),
            ("bandwidth", ASI_CTRL_BANDWIDTH),
            ("high_speed", ASI_CTRL_HIGH_SPEED),
            ("temperature", ASI_CTRL_TEMPERATURE),
            ("wb_r", ASI_CTRL_WB_R),
            ("wb_b", ASI_CTRL_WB_B),
        ]:
            try:
                result[name] = int(sdk.getControlValue(ctrl_id))
            except Exception:
                result[name] = -1
        # Convert temperature: raw value / 10 = °C
        if result.get("temperature", -1) != -1:
            result["temperature_c"] = result["temperature"] / 10.0
        return result
    except Exception as e:
        logger.error(f"Error getting all ASI controls: {e}")
        return {}


def get_phone_sensor_info() -> dict:
    """
    Get the phone's rear camera physical sensor dimensions via Camera2 API.

    Returns a dict with:
        width_mm, height_mm, pixel_size_um, resolution_x, resolution_y

    Useful for auto-populating the ASTAP plate solver sensor width field
    when using the phone camera for plate solving.
    """
    if _camera_manager is None:
        return {}
    try:
        info = _camera_manager.getPhoneSensorInfo()
        if info is None or info.isEmpty():
            return {}
        return {
            "width_mm": float(info.get("width_mm")),
            "height_mm": float(info.get("height_mm")),
            "pixel_size_um": float(info.get("pixel_size_um")),
            "resolution_x": int(info.get("resolution_x")),
            "resolution_y": int(info.get("resolution_y")),
        }
    except Exception as e:
        logger.error(f"Error getting phone sensor info: {e}")
        return {}


def get_asi_camera_info() -> dict:
    """
    Get ASI SDK camera information.

    Returns a dict with camera properties:
        name, max_width, max_height, pixel_size_um, sensor_width_mm,
        sensor_height_mm, is_color, is_cooled, sdk_version

    The sensor dimensions are computed as:
        sensor_width_mm  = max_width  * pixel_size_um / 1000
        sensor_height_mm = max_height * pixel_size_um / 1000
    These are the physical chip dimensions needed for FOV calculation
    in the ASTAP plate solver.
    """
    if _camera_manager is None:
        return {}
    try:
        sdk = _camera_manager.getAsiSdk()
        if sdk is None:
            return {}
        return {
            "name": str(sdk.cameraName),
            "max_width": int(sdk.maxWidth),
            "max_height": int(sdk.maxHeight),
            "pixel_size_um": float(sdk.pixelSizeUm),
            "sensor_width_mm": float(sdk.sensorWidthMm),
            "sensor_height_mm": float(sdk.sensorHeightMm),
            "is_color": bool(sdk.isColor),
            "is_cooled": bool(sdk.isCooled),
            "is_open": bool(sdk.isOpen),
            "is_capturing": bool(sdk.isCapturing),
            "sdk_version": str(sdk.getSDKVersion()),
        }
    except Exception as e:
        logger.error(f"Error getting ASI camera info: {e}")
        return {}
