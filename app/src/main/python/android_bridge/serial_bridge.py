"""
USB serial bridge -- Python interface to the Android UsbSerialManager.

Provides a pyserial-compatible interface so that telescope_bridge.py can
talk to a USB-connected mount without code changes.

On desktop:
    telescope_bridge.py uses serial.Serial (pyserial)
On Android:
    telescope_bridge.py uses this bridge, which calls UsbSerialManager
    via Chaquopy's Java bridge.

The Kotlin UsbSerialManager supports FTDI, CP210x, CH340, PL2303 --
the same chips used in telescope serial cables.

NOTE: For v1.0, WiFi TCP mode is the recommended and zero-change path.
USB serial is here for mounts without WiFi (older OnStep boards, etc.).
"""

import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger("serial_bridge")

# Kotlin UsbSerialManager instance, injected by TelescopeService at startup
_serial_manager = None


def set_serial_manager(manager):
    """
    Called from Kotlin at startup to inject the UsbSerialManager.

    In Kotlin:
        val py = Python.getInstance()
        val bridge = py.getModule("android_bridge.serial_bridge")
        bridge.callAttr("set_serial_manager", usbSerialManager)
    """
    global _serial_manager
    _serial_manager = manager
    logger.info("USB serial manager set from Android")


def list_ports() -> List[Dict[str, Any]]:
    """
    List available USB serial ports.
    Returns a list of dicts matching telescope_bridge.py's port enumeration.
    """
    if _serial_manager is None:
        return []
    try:
        java_list = _serial_manager.listPorts()
        result = []
        # Chaquopy returns a Java ArrayList<Map<String,Object>>.
        # Iterate by index and convert each Java Map to a Python dict.
        for i in range(java_list.size()):
            java_map = java_list.get(i)
            py_dict = {}
            for key in java_map.keySet().toArray():
                py_dict[str(key)] = java_map.get(key)
            result.append(py_dict)
        return result
    except Exception as e:
        logger.error(f"Error listing USB ports: {e}")
        return []


def connect(port_index: int = 0, baudrate: int = 9600) -> bool:
    """Connect to a USB serial port by index."""
    if _serial_manager is None:
        logger.warning("No USB serial manager available")
        return False
    try:
        return bool(_serial_manager.connect(port_index, baudrate))
    except Exception as e:
        logger.error(f"USB connect error: {e}")
        return False


def send(command: str) -> str:
    """
    Send an LX200 command and read the response.
    This is the key function called by telescope_bridge.py's _io_worker.
    """
    if _serial_manager is None:
        return ""
    try:
        result = _serial_manager.send(command)
        return str(result) if result else ""
    except Exception as e:
        logger.error(f"USB send error: {e}")
        return ""


def disconnect():
    """Disconnect from the USB serial port."""
    if _serial_manager is None:
        return
    try:
        _serial_manager.disconnect()
    except Exception as e:
        logger.error(f"USB disconnect error: {e}")


def is_connected() -> bool:
    """Check if a USB serial port is currently open."""
    if _serial_manager is None:
        return False
    try:
        return bool(_serial_manager.isConnected())
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════
#  pyserial-compatible shim
# ═══════════════════════════════════════════════════════════════════════

class AndroidSerialPort:
    """
    Full pyserial-compatible wrapper around UsbSerialManager.

    Drop-in replacement for ``serial.Serial`` that telescope_bridge.py
    uses when running on Android.  All byte-level I/O is delegated to the
    Kotlin ``UsbSerialManager`` via Chaquopy's Java bridge.

    The wrapper maintains an internal read buffer so that
    ``in_waiting`` and single-byte ``read(1)`` calls (used heavily by
    ``_read_serial_response``) work correctly even though the USB-serial
    driver delivers data in variable-size chunks.
    """

    def __init__(self):
        self._is_open = False
        self._port_index = 0
        self._baudrate = 9600
        self._read_buffer = bytearray()

    # ── pyserial properties ────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        return self._is_open and is_connected()

    @property
    def in_waiting(self) -> int:
        """Number of bytes available to read without blocking."""
        if not self.is_open or _serial_manager is None:
            return len(self._read_buffer)
        # Pull any hardware-buffered data with a non-blocking read
        try:
            data = bytes(_serial_manager.rawRead(512, 0))
            if data:
                self._read_buffer.extend(data)
        except Exception:
            pass
        return len(self._read_buffer)

    # ── open / close ───────────────────────────────────────────────────

    def open(self, port_index: int = 0, baudrate: int = 9600) -> bool:
        self._port_index = port_index
        self._baudrate = baudrate
        self._read_buffer = bytearray()
        self._is_open = connect(port_index, baudrate)
        return self._is_open

    def close(self):
        disconnect()
        self._is_open = False
        self._read_buffer = bytearray()

    # ── write ──────────────────────────────────────────────────────────

    def write(self, data: bytes) -> int:
        """Write raw bytes to the serial port."""
        if not self.is_open or _serial_manager is None:
            return 0
        try:
            return int(_serial_manager.rawWrite(data))
        except Exception as e:
            logger.error(f"AndroidSerialPort.write error: {e}")
            return 0

    def flush(self):
        """Flush write buffer (no-op — USB writes are synchronous)."""
        pass

    # ── read ───────────────────────────────────────────────────────────

    def read(self, size: int = 1) -> bytes:
        """Read up to *size* bytes, blocking briefly if needed."""
        if not self.is_open or _serial_manager is None:
            return b""

        # Serve from internal buffer first
        if len(self._read_buffer) >= size:
            result = bytes(self._read_buffer[:size])
            del self._read_buffer[:size]
            return result

        # Not enough buffered — do a blocking read from hardware
        try:
            data = bytes(_serial_manager.rawRead(max(size, 64), 100))
            if data:
                self._read_buffer.extend(data)
        except Exception:
            pass

        if len(self._read_buffer) >= size:
            result = bytes(self._read_buffer[:size])
            del self._read_buffer[:size]
            return result
        elif self._read_buffer:
            result = bytes(self._read_buffer)
            self._read_buffer = bytearray()
            return result

        return b""

    # ── buffer management ──────────────────────────────────────────────

    def reset_input_buffer(self):
        """Discard any buffered input data."""
        self._read_buffer = bytearray()
        if _serial_manager is None:
            return
        try:
            _serial_manager.purgeBuffers()
        except Exception:
            pass
        # Drain any remaining hardware data
        try:
            while True:
                data = bytes(_serial_manager.rawRead(1024, 0))
                if not data:
                    break
        except Exception:
            pass

    # ── context manager ────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
