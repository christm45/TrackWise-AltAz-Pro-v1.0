"""
Network bridge -- Python interface to Android CellularHttpClient.

Allows Python code (e.g. weather_service.py) to make HTTP GET/POST requests
over the cellular network (4G/5G) even when WiFi is connected to a
local-only network like a telescope mount hotspot.

The Kotlin CellularHttpClient is injected by TelescopeService at startup
via set_cellular_client().

Usage:
    from android_bridge.network_bridge import cellular_get, cellular_post, cellular_post_multipart
    json_string = cellular_get("https://api.open-meteo.com/v1/forecast?...")
    result = cellular_post("https://example.com/api", {"key": "value"})
    result = cellular_post_multipart("https://nova.astrometry.net/api/upload", {"apikey": "..."}, "/path/to/image.fits")
"""

import logging
import os

logger = logging.getLogger("network_bridge")

# The Kotlin CellularHttpClient instance, set by the Android service
_cellular_client = None


def set_cellular_client(client):
    """
    Called from Kotlin at startup to inject the CellularHttpClient.

    In Kotlin:
        val py = Python.getInstance()
        val bridge = py.getModule("android_bridge.network_bridge")
        bridge.callAttr("set_cellular_client", cellularHttpClient)
    """
    global _cellular_client
    _cellular_client = client
    if client is not None:
        logger.info("CellularHttpClient injected")
        # Request the cellular network immediately
        try:
            client.requestCellularNetwork()
            logger.info("Cellular network requested")
        except Exception as e:
            logger.warning("Failed to request cellular network: %s", e)
    else:
        logger.warning("CellularHttpClient set to None")


def cellular_get(url: str, timeout_ms: int = 5000) -> str:
    """
    Make an HTTP GET request over the cellular (4G/5G) network.

    Args:
        url:        Full URL with query parameters.
        timeout_ms: Connection + read timeout in milliseconds.

    Returns:
        Response body as a string, or None if unavailable/failed.
    """
    if _cellular_client is None:
        return None
    try:
        result = _cellular_client.get(url, timeout_ms)
        # Chaquopy returns Java String which auto-converts to Python str
        return str(result) if result is not None else None
    except Exception as e:
        logger.warning("Cellular GET failed: %s", e)
        return None


def cellular_post(url: str, form_data: dict, timeout_ms: int = 10000) -> str:
    """
    Make an HTTP POST request with form-urlencoded body over the cellular network.

    Args:
        url:         Full URL.
        form_data:   Dict of key-value pairs to send as form data.
        timeout_ms:  Connection + read timeout in milliseconds.

    Returns:
        Response body as a string, or None if unavailable/failed.
    """
    if _cellular_client is None:
        return None
    try:
        from java import jclass
        HashMap = jclass("java.util.HashMap")
        java_map = HashMap()
        for k, v in form_data.items():
            java_map.put(str(k), str(v))

        result = _cellular_client.post(url, java_map, timeout_ms)
        return str(result) if result is not None else None
    except ImportError:
        logger.error("Java imports not available -- not running on Android/Chaquopy")
        return None
    except Exception as e:
        logger.warning("Cellular POST failed: %s", e)
        return None


def cellular_post_multipart(
    url: str,
    form_fields: dict,
    file_path: str,
    file_field: str = "file",
    file_mime: str = "application/octet-stream",
    timeout_ms: int = 30000,
) -> str:
    """
    Make an HTTP POST request with multipart/form-data body over the cellular
    network.  Supports uploading a binary file alongside text form fields.

    Args:
        url:          Full URL.
        form_fields:  Dict of text key-value pairs to include.
        file_path:    Path to the file to upload.
        file_field:   The form field name for the file (default "file").
        file_mime:    MIME type of the file.
        timeout_ms:   Connection + read timeout in milliseconds.

    Returns:
        Response body as a string, or None if unavailable/failed.
    """
    if _cellular_client is None:
        return None
    try:
        from java import jclass, jarray, jbyte

        # Read file bytes
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        # Convert Python bytes to Java byte[]
        java_bytes = jarray(jbyte)(file_bytes)

        # Convert Python dict to Java HashMap
        HashMap = jclass("java.util.HashMap")
        java_map = HashMap()
        for k, v in form_fields.items():
            java_map.put(str(k), str(v))

        file_name = os.path.basename(file_path)

        result = _cellular_client.postMultipart(
            url, java_map, file_name, file_field, java_bytes, file_mime, timeout_ms
        )
        return str(result) if result is not None else None
    except ImportError:
        logger.error("Java imports not available -- not running on Android/Chaquopy")
        return None
    except Exception as e:
        logger.warning("Cellular multipart POST failed: %s", e)
        return None


def is_cellular_available() -> bool:
    """Check if the cellular network bridge is available and connected."""
    if _cellular_client is None:
        return False
    try:
        return bool(_cellular_client.isAvailable())
    except Exception:
        return False
