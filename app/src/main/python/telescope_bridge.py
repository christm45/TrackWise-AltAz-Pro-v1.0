"""
Telescope Bridge - Hardware communication layer for telescope mount control.

This module provides the TelescopeBridge class, which manages the physical
communication link between the application and a real telescope mount. It sits
between the high-level protocol layer (mount_protocol.py) and the mount
hardware, abstracting away the transport details so the rest of the
application can issue commands without caring whether the mount is connected
via USB serial or WiFi TCP.

Architecture role:
    Application layers (UI, tracking, plate-solving)
        |
        v
    Protocol layer (mount_protocol.py: LX200, NexStar, iOptron, etc.)
        |
        v
    >>> TelescopeBridge <<<   <-- this module
        |
        v
    Physical mount (serial USB or WiFi TCP)

Connection types:
    - USB Serial: Uses pyserial to open a COM port (Windows) or /dev/ttyUSB*
      (Linux). Typical baud rates: 9600 (LX200) or 19200/115200 (OnStep).
    - WiFi TCP: Opens a raw TCP socket to the mount's IP address and port
       (port 9996 for OnStep SmartWebServer persistent channel).

Command flow:
    1. Caller invokes send_command(cmd_string).
    2. send_command() delegates to _send_serial_command() or _send_tcp_command()
       depending on the active connection type.
    3. The transport method delegates formatting to mount_protocol.format_outgoing(),
       writes bytes to the serial port or TCP socket, waits a protocol-specific
       delay (mount_protocol.get_command_delay()), then reads the response with
       a protocol-specific timeout (mount_protocol.get_read_timeout()).
    4. Response normalization is delegated to mount_protocol.normalize_response().
    5. The response string is returned to the caller.

Thread safety:
    All serial/TCP I/O is serialized through _io_lock. User-initiated commands
    go through _cmd_queue -> _io_worker, and the _read_loop uses _poll_send
    which also acquires _io_lock. This prevents interleaved commands from
    corrupting the byte stream.

Callbacks:
    - on_altaz_update(alt, az): Fired each time the _read_loop() successfully
      reads valid altitude and azimuth from the mount.
    - on_connected(telescope_info): Fired when a connection is established.
    - on_disconnected(): Fired when the connection is lost or closed.
    - on_log(message): Fired for diagnostic log messages.

Auto-detection:
    During connection, the bridge uses mount_protocol.test_connection() to
    identify the mount firmware. For LX200, if the response contains "OnStep"
    or "On-Step", the is_onstep flag is set to True.
"""

import threading
import time
import socket
import queue
from concurrent.futures import Future
from typing import Optional, Callable, List
from dataclasses import dataclass

from telescope_logger import get_logger
from mount_protocol import (
    MountProtocol, LX200MountProtocol, NexStarMountProtocol,
    PositionData, CommandResult, get_protocol, list_protocols,
    _parse_lx200_dms,
)

_logger = get_logger(__name__)


@dataclass
class TelescopeInfo:
    """Information about a connected telescope.

    Attributes:
        port: Connection endpoint identifier. For serial connections this is
              the port name (e.g. "COM3"); for TCP connections it is "ip:port".
        baudrate: Serial baud rate. Set to 0 for TCP connections.
        is_connected: Whether the telescope is currently connected.
        model: Human-readable model/product name as reported by the firmware,
               or a descriptive fallback string if auto-detection failed.
    """
    port: str
    baudrate: int
    is_connected: bool = False
    model: str = "Unknown"


class TelescopeBridge:
    """
    Hardware communication bridge for telescope mounts.

    Protocol-agnostic transport layer that supports multiple mount protocols
    (LX200, NexStar, iOptron, ASCOM Alpaca, INDI, etc.) via the
    ``MountProtocol`` abstraction.  Command formatting, timing, and response
    normalization are delegated to the active protocol instance.

    Supports two transport layers:
        - USB serial (via pyserial) for direct cable connections.
        - WiFi TCP sockets for wireless mount adapters.

    On connection, the bridge uses ``mount_protocol.test_connection()`` to
    identify the mount firmware.

    After a successful connection, a background daemon thread (_read_loop) is
    started. This thread continuously polls the mount for its current position
    via ``mount_protocol.poll_position()`` and fires the on_altaz_update
    callback with each new reading.

    Typical usage::

        bridge = TelescopeBridge()
        bridge.set_protocol("nexstar")
        bridge.on_altaz_update = lambda alt, az: print(f"Alt={alt} Az={az}")
        bridge.connect("COM3", baudrate=9600)
        # ... later ...
        bridge.disconnect()
    """
    
    def __init__(self):
        """Initialize the TelescopeBridge with default state.

        All connection handles are None, callbacks are unset, and the
        background read thread is not running. Call connect() to establish
        a link to the mount.
        """
        self.serial_connection = None
        self.socket_connection = None  # TCP socket connection
        self.connection_type = None  # 'serial' or 'tcp'
        self.is_connected = False
        self.telescope_info: Optional[TelescopeInfo] = None
        self.last_error: Optional[str] = None  # Store the last error message
        self.is_onstep = False  # Automatic OnStep detection flag

        # TCP reconnection state: stored by _connect_tcp() so that
        # _ensure_tcp_connection() can transparently create a fresh socket
        # when the ESP8266/ESP32 drops the connection (1-second timeout).
        self._tcp_ip: Optional[str] = None
        self._tcp_port: Optional[int] = None
        self._tcp_reconnect_lock = threading.Lock()
        self._tcp_reconnect_count = 0  # For logging
        self._poll_log_count = 0  # Diagnostic poll counter

        # Mount communication protocol (LX200 default, NexStar optional)
        self.mount_protocol: MountProtocol = LX200MountProtocol()
        
        # Callbacks
        self.on_connected: Optional[Callable] = None
        self.on_disconnected: Optional[Callable] = None
        self.on_log: Optional[Callable] = None
        self.on_altaz_update: Optional[Callable] = None  # Alt/Az + RA/Dec (source of truth)
        self.on_focuser_position: Optional[Callable] = None  # Focuser position from :FG#
        
        # I/O serialization: all serial/TCP reads and writes go through this lock
        # to prevent interleaved commands from multiple threads (tracking loop,
        # _read_loop, UI thread, plate-solve) corrupting the byte stream.
        self._io_lock = threading.Lock()

        # Command queue: callers submit (command_str, Future) tuples.
        # A dedicated _io_worker thread drains the queue and executes each
        # command under _io_lock, writing the result into the Future.
        self._cmd_queue: queue.Queue = queue.Queue()
        self._io_thread: Optional[threading.Thread] = None

        # Read thread
        self._read_thread: Optional[threading.Thread] = None
        self._running = False
        
    def _safe_log(self, message: str):
        """Log a message safely, falling back to print if _log is unavailable.

        This wrapper exists because _log() relies on the on_log callback which
        may not yet be set during early initialization or port enumeration.

        Args:
            message: The diagnostic message to log.
        """
        try:
            self._log(message)
        except Exception:
            # If _log callback is not available, use the module logger
            _logger.debug(message)
    
    def set_protocol(self, protocol_name: str) -> None:
        """Switch the mount communication protocol.

        Args:
            protocol_name: ``'lx200'`` or ``'nexstar'``.
        """
        if self.is_connected:
            self._safe_log("Cannot change protocol while connected")
            return
        self.mount_protocol = get_protocol(protocol_name)
        self._safe_log(f"Protocol set to: {self.mount_protocol.name}")

    def _poll_send(self, cmd, timeout: float = 0.5) -> str:
        """Low-level send/receive for the read-loop polling thread.

        Acquires ``_io_lock`` for each command-response exchange so that
        the ``_io_worker`` thread can interleave between poll reads.

        **TCP auto-reconnect**: if a ``ConnectionError`` occurs during a
        TCP exchange, the lock is released, ``_ensure_tcp_connection()``
        is called to create a fresh socket, and the command is retried
        once.  If the retry also fails, the error is re-raised so that
        ``_read_loop`` can handle a true disconnect.

        For serial connections, errors are still raised immediately.

        Args:
            cmd: Command as ``str`` or ``bytes``.
            timeout: Read timeout in seconds (default 0.5 s).

        Returns:
            Response string (may be empty on timeout).

        Raises:
            ConnectionError, BrokenPipeError: on dead TCP connection
            after reconnect retry has also failed.
        """
        if isinstance(cmd, bytes):
            raw_bytes = cmd
        else:
            raw_bytes = cmd.encode('ascii')

        for attempt in range(2):  # attempt 0 = normal, attempt 1 = after reconnect
            with self._io_lock:
                if not self.is_connected:
                    return ""
                try:
                    if self.connection_type == 'serial' and self.serial_connection:
                        # Flush stale data before sending
                        try:
                            if self.serial_connection.in_waiting > 0:
                                self.serial_connection.reset_input_buffer()
                        except OSError:
                            pass
                        self.serial_connection.write(raw_bytes)
                        self.serial_connection.flush()
                        return self._read_serial_response(timeout=timeout) or ""

                    elif self.connection_type == 'tcp' and self.socket_connection:
                        self.socket_connection.sendall(raw_bytes)
                        resp = self._read_tcp_response(timeout=timeout) or ""
                        # Debug-level poll logging (visible only with DEBUG log level)
                        if _logger.isEnabledFor(10):  # logging.DEBUG == 10
                            self._poll_log_count += 1
                            cmd_str = raw_bytes.decode('ascii', errors='replace')
                            _logger.debug("POLL#%d TX:%s -> RX:'%s'",
                                          self._poll_log_count, cmd_str, resp)
                        return resp

                except (ConnectionError, BrokenPipeError,
                        ConnectionResetError, ConnectionAbortedError) as e:
                    if self.connection_type != 'tcp' or attempt == 1:
                        raise  # Serial or second attempt: give up
                    # TCP first attempt failed — will reconnect below
                except Exception as e:
                    self._log(f"_poll_send error ({cmd}): {e}")
                    return ""

            # --- TCP reconnect (outside _io_lock to avoid deadlock) ---
            if not self._ensure_tcp_connection():
                raise ConnectionError("TCP reconnect failed in _poll_send")

        return ""

    # ------------------------------------------------------------------
    # TCP auto-reconnect
    # ------------------------------------------------------------------

    def _ensure_tcp_connection(self) -> bool:
        """Transparently reconnect the TCP socket if the ESP has closed it.

        The ESP8266/ESP32 WiFi-to-serial bridge closes TCP connections
        after approximately 1 second regardless of activity.  This
        method is called before (or after failure of) every TCP
        send/recv cycle.  It:

            1. Probes the socket with non-blocking ``recv(1, MSG_PEEK)``
               to detect remote FIN (CLOSE_WAIT).
            2. If alive, returns True immediately (fast path).
            3. If dead, closes the old socket, waits 300 ms for the ESP
               to tear down, then creates a fresh connection.
            4. **Verifies** the new connection by sending ``:GVP#`` and
               checking for a non-empty response.  If the ESP bridge
               is not yet forwarding data, retries up to 3 times with
               increasing stabilisation delays (500 ms, 1 s, 2 s).
            5. On total failure returns False.

        Thread safety: Uses ``_tcp_reconnect_lock`` to serialize
        reconnect attempts from multiple threads (_read_loop polling
        and _io_worker commands).

        **Must be called WITHOUT _io_lock held** to avoid deadlock.

        Returns:
            True if a **verified** TCP socket is available after this call.
        """
        if not self._tcp_ip or not self._tcp_port:
            return False
        if not self._running:
            return False

        def _socket_alive(sock) -> bool:
            """True only if the socket is connected AND the remote
            has NOT sent FIN.  getpeername() is NOT sufficient because
            it returns True on CLOSE_WAIT sockets (FIN received, not
            yet locally closed).

            We use a non-blocking recv(1, MSG_PEEK) to probe:
              - socket.timeout / BlockingIOError → alive (no data yet)
              - b""  → remote sent FIN → dead
              - data → alive (data waiting)
              - OSError → dead
            """
            if sock is None:
                return False
            try:
                sock.getpeername()
            except (OSError, socket.error):
                return False
            try:
                old_timeout = sock.gettimeout()
                sock.settimeout(0)  # Non-blocking
                try:
                    data = sock.recv(1, socket.MSG_PEEK)
                    if not data:
                        return False  # FIN received
                    return True  # Data waiting — alive
                except (BlockingIOError, socket.timeout):
                    return True  # No data yet — alive
                except (OSError, socket.error):
                    return False
                finally:
                    sock.settimeout(old_timeout)
            except Exception:
                return False

        # Fast path: socket is still alive
        if _socket_alive(self.socket_connection):
            return True

        with self._tcp_reconnect_lock:
            # Double-check inside lock (another thread may have reconnected)
            if _socket_alive(self.socket_connection):
                return True

            # Close the dead socket and wait for ESP to tear down
            if self.socket_connection:
                try:
                    self.socket_connection.close()
                except OSError:
                    pass
                self.socket_connection = None
                # Give the ESP time to fully close its side and reset
                # the serial bridge before we reconnect.
                time.sleep(0.3)

            # Create a fresh TCP socket WITH verification.
            # The ESP serial bridge sometimes accepts a connection but
            # does NOT forward data to OnStep (empty responses).  This
            # was observed on the 2nd reconnect even with 500 ms delay.
            # Solution: send :GVP# after connecting and verify we get a
            # real response.  Retry with increasing delays if not.
            _MAX_RETRIES = 3
            _delays = [0.5, 1.0, 2.0]  # post-connect stabilisation (s)

            for _attempt in range(_MAX_RETRIES):
                new_sock = None
                try:
                    new_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    new_sock.settimeout(3.0)
                    new_sock.connect((self._tcp_ip, self._tcp_port))
                    new_sock.setsockopt(
                        socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    new_sock.settimeout(2.0)

                    time.sleep(_delays[_attempt])

                    # -- Verification: send :GVP# and expect a response --
                    try:
                        new_sock.sendall(b':GVP#')
                        resp_bytes = b''
                        deadline = time.time() + 1.0
                        while time.time() < deadline:
                            try:
                                b = new_sock.recv(1)
                                if not b:
                                    break          # FIN
                                resp_bytes += b
                                if b == b'#':
                                    break          # full response
                            except socket.timeout:
                                break
                        resp_str = resp_bytes.decode(
                            'ascii', errors='replace').rstrip('#')
                    except (OSError, socket.error) as ve:
                        resp_str = ''
                        self._log(
                            f"TCP reconnect attempt {_attempt+1}: "
                            f"verification send/recv error: {ve}")

                    if resp_str:
                        # Verification passed — bridge is forwarding data
                        self.socket_connection = new_sock
                        self._tcp_reconnect_count += 1
                        if (self._tcp_reconnect_count <= 5
                                or self._tcp_reconnect_count % 10 == 0):
                            self._log(
                                f"TCP reconnected #{self._tcp_reconnect_count}"
                                f" to {self._tcp_ip}:{self._tcp_port}"
                                f" (verified: '{resp_str}',"
                                f" attempt {_attempt+1},"
                                f" delay {_delays[_attempt]}s)")
                        return True

                    # Empty response — ESP bridge not forwarding yet
                    self._log(
                        f"TCP reconnect attempt {_attempt+1}: connected "
                        f"but :GVP# returned empty "
                        f"(delay {_delays[_attempt]}s)")
                    try:
                        new_sock.close()
                    except OSError:
                        pass
                    new_sock = None
                    # Extra gap before next retry so ESP can fully reset
                    if _attempt < _MAX_RETRIES - 1:
                        time.sleep(0.5)

                except (socket.timeout, socket.error, OSError) as e:
                    self._log(
                        f"TCP reconnect attempt {_attempt+1} FAILED: {e}")
                    if new_sock is not None:
                        try:
                            new_sock.close()
                        except OSError:
                            pass
                    new_sock = None
                    if _attempt < _MAX_RETRIES - 1:
                        time.sleep(0.5)

            # All retries exhausted
            self._log(
                f"TCP reconnect FAILED after {_MAX_RETRIES} attempts")
            self.socket_connection = None
            return False

    def get_available_ports(self) -> List[str]:
        """Enumerate available serial ports on the system.

        Uses two complementary detection methods:
            1. pyserial's ``list_ports.comports()`` — the most reliable and
               cross-platform approach.
            2. Windows Registry (HKLM\\HARDWARE\\DEVICEMAP\\SERIALCOMM) — a
               supplementary method that can find ports not reported by pyserial.

        On **Windows** the result always includes COM1–COM30 as a fallback
        (the UI combobox is editable so the user can also type manually).
        On **Linux** only the auto-detected ports are returned (typically
        ``/dev/ttyUSB*`` and ``/dev/ttyACM*``).

        The auto-detected subset is also stored in ``_last_detected_ports``
        so the web UI can highlight truly present ports.

        Returns:
            A sorted list of port name strings.
        """
        detected_ports: List[str] = []
        detected_set: set = set()
        import sys

        # Method 1: pyserial list_ports (cross-platform)
        try:
            import serial.tools.list_ports
            available_ports = serial.tools.list_ports.comports()
            for port in available_ports:
                if port.device not in detected_set:
                    detected_set.add(port.device)
                    detected_ports.append(port.device)
                    self._safe_log(f"Port detected (pyserial): {port.device} - {port.description}")
        except ImportError:
            self._safe_log("pyserial not installed")
        except Exception as e:
            self._safe_log(f"Error in list_ports.comports(): {e}")

        # Method 2: Windows registry supplement
        if sys.platform == 'win32':
            try:
                import winreg
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DEVICEMAP\SERIALCOMM"
                )
                try:
                    i = 0
                    while True:
                        try:
                            name, value, _ = winreg.EnumValue(key, i)
                            if value.startswith('COM'):
                                if value not in detected_set:
                                    detected_set.add(value)
                                    detected_ports.append(value)
                                    self._safe_log(f"Port detected (registry): {value}")
                            i += 1
                        except OSError:
                            break
                finally:
                    winreg.CloseKey(key)
            except Exception as e:
                self._safe_log(f"Error reading Windows registry: {e}")

        # Sort helper: numeric for COMx, alphabetical for /dev paths
        def port_sort_key(port_name: str):
            if port_name.startswith('COM'):
                try:
                    return (0, int(port_name[3:]))
                except ValueError:
                    return (0, 9999)
            # /dev paths: ttyACM before ttyUSB for consistency
            return (1, port_name)

        detected_ports.sort(key=port_sort_key)
        # Save for the web API to highlight
        self._last_detected_ports = list(detected_ports)

        if sys.platform == 'win32':
            # On Windows include COM1-COM30 fallback for manual entry
            merged_set = set(detected_ports)
            all_ports = list(detected_ports)  # detected first
            for i in range(1, 31):
                name = f"COM{i}"
                if name not in merged_set:
                    all_ports.append(name)
            all_ports.sort(key=port_sort_key)
        else:
            # On Linux/macOS only return detected ports — COM ports are
            # meaningless. If nothing is detected, offer common defaults.
            if detected_ports:
                all_ports = detected_ports
            else:
                all_ports = ["/dev/ttyUSB0", "/dev/ttyACM0"]
                self._safe_log("No ports detected; showing Linux defaults")

        if detected_ports:
            self._safe_log(f"{len(detected_ports)} port(s) detected: {', '.join(detected_ports)}")
        else:
            self._safe_log("No ports detected automatically")

        return all_ports
    
    def connect(self, port: str, baudrate: int = 9600, connection_type: str = 'serial', tcp_ip: Optional[str] = None, tcp_port: Optional[int] = None) -> bool:
        """
        Connect to the telescope via serial port or WiFi TCP.

        This is the main entry point for establishing a connection. It
        delegates to _connect_serial() or _connect_tcp() based on the
        connection_type parameter. If already connected, the existing
        connection is closed first.

        Args:
            port: Serial port name (e.g. "COM3", "/dev/ttyUSB0"). Used when
                  connection_type is 'serial'.
            baudrate: Serial communication speed in baud (default: 9600). Used
                      when connection_type is 'serial'.
            connection_type: Transport type - 'serial' for USB or 'tcp' for
                             WiFi.
            tcp_ip: IP address for TCP connection (e.g. '192.168.0.1'). Required
                    when connection_type is 'tcp'.
            tcp_port: TCP port number (e.g. 9999). Required when connection_type
                      is 'tcp'.
            
        Returns:
            True if the connection was successfully established.
        """
        # Disconnect first if already connected
        if self.is_connected:
            self.disconnect()
        
        if connection_type == 'tcp':
            if tcp_ip is None or tcp_port is None:
                self._safe_log("TCP connection requires both ip and port")
                return False
            return self._connect_tcp(tcp_ip, tcp_port)
        else:
            return self._connect_serial(port, baudrate)
    
    def _connect_tcp(self, ip: str, port: int) -> bool:
        """
        Connect to the telescope via TCP socket (WiFi).

        Connection sequence:
            1. Create a TCP socket with a 5-second connection timeout.
            2. Connect to ip:port.
            3. Disable Nagle's algorithm (TCP_NODELAY) for low-latency command
               exchange, set read/write timeout to 2 seconds.
            4. Wait 500ms for OnStep firmware to stabilize after accept.
            5. Send :GVP# (Get Version Product) to identify the controller.
            6. If no response, retry with :GR# (Get RA) as a fallback.
            7. If still no response but the socket remains open, accept the
               connection anyway (OnStep may be in silent mode).
            8. On success, start the background _read_loop() thread and fire
               the on_connected callback.

        Args:
            ip: IP address of the telescope (e.g. '192.168.0.1').
            port: TCP port number (e.g. 9999).
            
        Returns:
            True if the connection was successfully established.
        """
        try:
            import socket
            
            # Store for auto-reconnect (_ensure_tcp_connection)
            self._tcp_ip = ip
            self._tcp_port = port
            self._tcp_reconnect_count = 0
            
            self._log(f"WiFi connection to {ip}:{port}...")
            
            # Create a TCP socket (no ping test to avoid blocking)
            self.socket_connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_connection.settimeout(5.0)  # Connection timeout: 5 seconds (reasonable)
            
            # Connect
            self._log(f"Attempting TCP connection to {ip}:{port}...")
            self.socket_connection.connect((ip, port))
            
            # Configure the socket to reduce latency
            self.socket_connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.socket_connection.settimeout(2.0)  # Timeout for read/write operations
            
            self._log(f"TCP socket connected to {ip}:{port}")
            
            # Pause after TCP accept so the ESP8266/ESP32 WiFi module
            # can finish setting up its serial bridge.  300 ms is the
            # minimum reliable value — shorter sleeps cause the first
            # command to be lost (OnStep never sees it).
            time.sleep(0.3)
            
            # Test the connection using the mount protocol
            self._log(f"Testing WiFi connection with protocol: {self.mount_protocol.name}")

            model = "Unknown"
            connection_confirmed = False

            def _tcp_test_send(cmd_bytes, timeout=2.0):
                """Send/receive for connection test (TCP)."""
                try:
                    if isinstance(cmd_bytes, str):
                        cmd_bytes = cmd_bytes.encode('ascii')
                    self.socket_connection.sendall(cmd_bytes)
                    # No sleep needed — _read_tcp_response() already
                    # waits up to `timeout` seconds for data.
                    response = self._read_tcp_response(timeout=timeout)
                    self._log(f"WiFi test TX: {cmd_bytes} -> RX: '{response}'")
                    return response or ""
                except (ConnectionResetError, ConnectionAbortedError, OSError) as e:
                    error_code = getattr(e, 'winerror', None) or getattr(e, 'errno', None)
                    if error_code in (10053, 10054):
                        self._log(f"Connection closed during test (code {error_code})")
                    else:
                        self._log(f"WiFi test error: {e}")
                    return ""
                except Exception as e:
                    self._log(f"WiFi test error: {e}")
                    return ""

            connection_confirmed, model, detected_onstep = \
                self.mount_protocol.test_connection(_tcp_test_send)
            self.is_onstep = detected_onstep

            if connection_confirmed:
                self._log(f"WiFi connection confirmed: {model}")
            else:
                # LX200 fallback: try :GR# directly
                if isinstance(self.mount_protocol, LX200MountProtocol):
                    try:
                        self._log("Trying alternative command :GR# (RA)...")
                        self.socket_connection.sendall(b":GR#")
                        alt_response = self._read_tcp_response(timeout=2.0)
                        if alt_response and len(alt_response) > 0:
                            self._log(f"Response received with :GR#: '{alt_response[:20]}...'")
                            model = "WiFi Telescope (OnStep compatible)"
                            self.is_onstep = True
                            connection_confirmed = True
                    except Exception as e2:
                        self._log(f"Alternative command failed: {e2}")
            
            # If no confirmation but the connection is still open,
            # accept anyway (some OnStep units don't respond to test commands)
            if not connection_confirmed:
                try:
                    # Verify the socket is still valid
                    self.socket_connection.getpeername()
                    self._log(f"No response to test commands, but connection still open")
                    self._log(f"Accepting connection anyway (OnStep may be in silent mode)")
                    model = "WiFi Telescope (OnStep)"
                    self.is_onstep = True
                    connection_confirmed = True
                except (OSError, socket.error):
                    # Socket closed, do not accept
                    pass
            
            if connection_confirmed:
                self.connection_type = 'tcp'
                self.telescope_info = TelescopeInfo(
                    port=f"{ip}:{port}",
                    baudrate=0,  # No baud rate for TCP
                    is_connected=True,
                    model=model
                )
                self.is_connected = True
                
                self._log(f"Connected via WiFi to: {model}")
                
                # Start the read thread and I/O worker IMMEDIATELY.
                # DO NOT sleep here — any idle gap lets the ESP8266/ESP32
                # WiFi adapter close the TCP connection.
                self._running = True
                self._read_thread = threading.Thread(target=self._read_loop, daemon=True, name="ReadLoop")
                self._read_thread.start()
                self._io_thread = threading.Thread(target=self._io_worker, daemon=True, name="IOWorker")
                self._io_thread.start()
                
                if self.on_connected:
                    self.on_connected(self.telescope_info)
                
                return True
            else:
                error_msg = f"Unable to confirm WiFi connection to {ip}:{port}. Check:\n- That the telescope is powered on\n- That the IP address {ip} is correct\n- That the port {port} is correct\n- That the telescope accepts TCP connections"
                self.last_error = error_msg
                self._log(f"ERROR: {error_msg}")
                if self.socket_connection:
                    try:
                        self.socket_connection.close()
                    except OSError:
                        pass
                    self.socket_connection = None
                return False
                
        except socket.timeout:
            error_msg = f"Timeout connecting via WiFi to {ip}:{port}. The telescope is not responding."
            self.last_error = error_msg
            self._log(f"ERROR: {error_msg}")
            self._log(f"Troubleshooting checklist:")
            self._log(f"   1. Is the telescope powered on and connected to WiFi?")
            self._log(f"   2. Is the IP address {ip} correct?")
            self._log(f"   3. Is port {port} correct? (OnStep persistent channel is 9996)")
            self._log(f"   4. Is your computer on the same WiFi network?")
            self._log(f"   5. Is a firewall blocking the connection?")
            if self.socket_connection:
                try:
                    self.socket_connection.close()
                except OSError:
                    pass
                self.socket_connection = None
            return False
        except socket.error as e:
            error_code = getattr(e, 'winerror', None) or getattr(e, 'errno', None)
            error_str = str(e)
            
            # Clearer error messages based on error code
            if error_code == 10061:  # Connection refused
                error_msg = f"Connection refused at {ip}:{port}. The port may be closed or no service is listening."
                self._log(f"ERROR: {error_msg}")
                self._log(f"The telescope may be powered on but the TCP service is not active on port {port}")
            elif error_code == 10051:  # Network unreachable
                error_msg = f"Network unreachable. Cannot reach {ip}:{port}."
                self._log(f"ERROR: {error_msg}")
                self._log(f"Verify that you are on the same WiFi network as the telescope")
            elif error_code == 10060:  # Connection timed out
                error_msg = f"Connection timed out at {ip}:{port}. The telescope is not responding."
                self._log(f"ERROR: {error_msg}")
                self._log(f"Verify the IP address and that the telescope is powered on")
            else:
                error_msg = f"WiFi connection error at {ip}:{port}: {error_str}"
                self._log(f"ERROR: {error_msg}")
            
            self.last_error = error_msg
            if self.socket_connection:
                try:
                    self.socket_connection.close()
                except OSError:
                    pass
                self.socket_connection = None
            return False
        except Exception as e:
            error_msg = f"WiFi connection error: {str(e)}"
            self.last_error = error_msg
            self._log(f"ERROR: {error_msg}")
            if self.socket_connection:
                try:
                    self.socket_connection.close()
                except OSError:
                    pass
                self.socket_connection = None
            return False
    
    def _connect_serial(self, port: str, baudrate: int = 9600) -> bool:
        """
        Connect to the telescope via serial port (USB).

        Connection sequence:
            1. Open the serial port with the specified baud rate, 2-second
               read/write timeouts.
            2. Flush any stale data from the input buffer.
            3. Wait 500ms for the port to become ready.
            4. Send a series of test commands to identify the controller:
               - :GVP# (Get Version Product name)
               - :GVN# (Get Version Number)
               - :GR#  (Get Right Ascension)
               - :GD#  (Get Declination)
            5. For each test command, flush the input buffer first, send the
               command, wait 300ms, then read the response character-by-character
               until '#' terminator or 1-second timeout.
            6. If :GVP# response contains "OnStep"/"On-Step", set is_onstep=True.
            7. If no test command gets a valid response but raw data is received,
               accept the connection with a warning.
            8. If the serial port is at least open (even without any response),
               accept the connection as some telescopes don't respond to test
               commands but work fine for normal operations.
            9. On success, start the background _read_loop() thread and fire
               the on_connected callback.

        Args:
            port: Serial port name (e.g. "COM3", "/dev/ttyUSB0").
            baudrate: Communication speed in baud (default: 9600).
            
        Returns:
            True if the connection was successfully established.
        """
        try:
            import serial
            
            self._log(f"Serial connection to {port} ({baudrate} baud)...")
            
            self.serial_connection = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=2.0,
                write_timeout=2.0
            )
            
            # Clean the input buffer (may contain residual data)
            if self.serial_connection.in_waiting > 0:
                self._log(f"Cleaning buffer: {self.serial_connection.in_waiting} bytes waiting")
                self.serial_connection.reset_input_buffer()
            
            # Wait for the port to be ready
            time.sleep(0.5)
            
            # Test the connection using the mount protocol
            self._log(f"Testing connection with protocol: {self.mount_protocol.name}")

            def _serial_test_send(cmd_bytes, timeout=1.5):
                """Send/receive for connection test (serial)."""
                try:
                    if isinstance(cmd_bytes, str):
                        cmd_bytes = cmd_bytes.encode('ascii')
                    # Clean buffer before sending
                    if self.serial_connection.in_waiting > 0:
                        self.serial_connection.reset_input_buffer()
                    self.serial_connection.write(cmd_bytes)
                    self.serial_connection.flush()
                    time.sleep(0.3)
                    # Read response char by char
                    response = ""
                    start_time = time.time()
                    while time.time() - start_time < timeout:
                        if self.serial_connection.in_waiting > 0:
                            char = self.serial_connection.read(1).decode('ascii', errors='ignore')
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
                # Fallback: check if the port is at least open and has data
                try:
                    time.sleep(0.2)
                    if self.serial_connection.in_waiting > 0:
                        raw_data = self.serial_connection.read(self.serial_connection.in_waiting)
                        self._log(f"Data received (unformatted): {raw_data}")
                        connection_confirmed = True
                        model = f"{self.mount_protocol.name} Telescope (non-standard response)"
                        self._log(f"Non-standard response detected, connection accepted anyway")
                except Exception as e:
                    self._log(f"Error checking for data: {e}")
            
            if connection_confirmed or self.serial_connection.is_open:
                # Connection accepted (either confirmed by response, or port is open)
                # If no response, accept anyway because some telescopes
                # may not respond to tests but work correctly for normal operations
                if not connection_confirmed:
                    model = f"{self.mount_protocol.name} Telescope (connected without test response)"
                    self._log(f"No response to test commands, but connection accepted")
                    self._log(f"The telescope may work even without test responses")
                
                self.connection_type = 'serial'
                self.telescope_info = TelescopeInfo(
                    port=port,
                    baudrate=baudrate,
                    is_connected=True,
                    model=model
                )
                self.is_connected = True
                
                self._log(f"Serial connected to: {model}")
                
                # Start the read thread and I/O worker
                self._running = True
                self._read_thread = threading.Thread(target=self._read_loop, daemon=True, name="ReadLoop")
                self._read_thread.start()
                self._io_thread = threading.Thread(target=self._io_worker, daemon=True, name="IOWorker")
                self._io_thread.start()
                
                if self.on_connected:
                    self.on_connected(self.telescope_info)
                
                return True
            else:
                # Port not open or error
                error_msg = f"Unable to establish connection on {port}. Check:\n- That the telescope is powered on\n- That the baud rate ({baudrate}) is correct\n- That the telescope is in LX200 mode\n- That the USB cable is working\n- That port {port} exists"
                self.last_error = error_msg
                self._log(f"ERROR: {error_msg}")
                if self.serial_connection:
                    try:
                        self.serial_connection.close()
                    except OSError:
                        pass
                return False
                
        except ImportError:
            error_msg = "pyserial is not installed. Run: pip install pyserial"
            self.last_error = error_msg
            self._log(f"ERROR: {error_msg}")
            return False
        except Exception as e:
            # Check if this is a SerialException (if serial is available)
            error_str = str(e)
            error_type = type(e).__name__
            
            # Serial port specific errors
            if "SerialException" in error_type or "serial" in error_str.lower():
                if "could not open port" in error_str.lower() or "access is denied" in error_str.lower():
                    error_msg = f"Cannot open port {port}. The port may be in use by another program or you lack permissions."
                elif "no such file or directory" in error_str.lower() or "device not found" in error_str.lower():
                    error_msg = f"Port {port} not found. Verify that the telescope is connected and the port is correct."
                elif "timeout" in error_str.lower():
                    error_msg = f"Timeout connecting to port {port}. The telescope is not responding."
                else:
                    error_msg = f"Serial port error: {error_str}"
            else:
                # Other errors
                error_msg = f"Connection error: {error_str}"
            
            self.last_error = error_msg
            self._log(f"ERROR: {error_msg}")
            if self.serial_connection:
                try:
                    self.serial_connection.close()
                except OSError:
                    pass
            return False
    
    def disconnect(self):
        """Disconnect from the telescope and clean up all resources.

        Stops the background _read_loop() and _io_worker threads, drains
        the command queue (failing pending futures), closes the serial port
        or TCP socket, resets all connection state, and fires the
        on_disconnected callback.
        """
        self._running = False

        # Drain command queue so _io_worker can exit and callers don't hang
        while not self._cmd_queue.empty():
            try:
                _, fut = self._cmd_queue.get_nowait()
                fut.set_result("")
            except (queue.Empty, Exception):
                break
        # Sentinel so _io_worker wakes up and exits
        self._cmd_queue.put(None)
        
        if self.serial_connection:
            try:
                self.serial_connection.close()
            except OSError:
                pass
            self.serial_connection = None
        
        if self.socket_connection:
            try:
                self.socket_connection.close()
            except OSError:
                pass
            self.socket_connection = None
        
        self.connection_type = None
        self.is_connected = False
        self.telescope_info = None
        self._tcp_ip = None
        self._tcp_port = None
        
        self._log("Telescope disconnected")
        
        if self.on_disconnected:
            self.on_disconnected()
    
    def send_command(self, command: str, timeout: float = 5.0) -> str:
        """
        Send a command to the telescope and return the response.

        This is the primary method used by the protocol layer and application
        code to communicate with the mount. Commands are enqueued to a
        dedicated I/O worker thread that serializes all serial/TCP access,
        preventing concurrent writes from multiple threads (tracking loop,
        _read_loop, UI thread) from corrupting the byte stream.

        The command string is normalized (ensuring ':' prefix and '#' suffix)
        by the underlying transport method before transmission.

        Args:
            command: LX200/OnStep command string (with or without ':' and '#'
                     delimiters). Examples: ":GA#", "GA", ":GVP#".
            timeout: Maximum time in seconds to wait for the response
                     (includes queue wait + I/O time). Defaults to 5s.
            
        Returns:
            The '#'-terminated response string from the telescope, or an empty
            string if not connected, if an error occurred, or if the timeout
            expired.
        """
        if not self.is_connected:
            return ""

        fut: Future = Future()
        self._cmd_queue.put((command, fut))
        try:
            return fut.result(timeout=timeout)
        except Exception:
            _logger.debug("send_command(%r) timed out or failed", command)
            return ""

    def _io_worker(self):
        """Dedicated I/O thread that drains the command queue.

        Runs as a daemon thread alongside _read_loop. For each
        ``(command, future)`` item pulled from the queue, it acquires
        ``_io_lock``, executes the command via the appropriate transport,
        and writes the response into the future so the caller unblocks.

        A ``None`` sentinel on the queue signals this thread to exit.
        """
        while self._running:
            try:
                item = self._cmd_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break  # Sentinel: shutdown
            command, fut = item
            try:
                response = self._send_command_locked(command)
                if not fut.done():
                    fut.set_result(response)
            except Exception as e:
                if not fut.done():
                    fut.set_result("")
                _logger.debug(f"IO worker error: {e}")

    def _send_command_locked(self, command: str) -> str:
        """Execute a command under the I/O lock (called only by _io_worker).

        Delegates to _send_serial_command() or _send_tcp_command() depending
        on the active connection type. The _io_lock is held for the entire
        send-receive cycle to prevent _read_loop from interleaving.

        **TCP auto-reconnect**: on connection error, releases _io_lock,
        calls _ensure_tcp_connection(), then retries once under _io_lock.

        Args:
            command: LX200 command string.

        Returns:
            The response string, or empty string on error.
        """
        for attempt in range(2):
            with self._io_lock:
                if not self.is_connected:
                    return ""
                if self.connection_type == 'serial':
                    return self._send_serial_command(command)
                elif self.connection_type == 'tcp':
                    try:
                        return self._send_tcp_command(command)
                    except (ConnectionError, BrokenPipeError,
                            ConnectionResetError, ConnectionAbortedError,
                            OSError) as e:
                        if attempt == 1:
                            # Second attempt failed — real disconnect
                            error_code = getattr(e, 'winerror', None) or getattr(e, 'errno', None)
                            if self._running:
                                self._log(f"WARNING: TCP send failed after reconnect "
                                          f"(code {error_code}): {e}")
                            self.is_connected = False
                            self._running = False
                            if self.on_disconnected:
                                self.on_disconnected()
                            return ""
                        # First attempt — will reconnect below
                else:
                    return ""

            # --- TCP reconnect (outside _io_lock) ---
            if not self._ensure_tcp_connection():
                self._log("TCP reconnect failed in _send_command_locked")
                self.is_connected = False
                self._running = False
                if self.on_disconnected:
                    self.on_disconnected()
                return ""
        return ""
    
    def _send_serial_command(self, command: str) -> str:
        """Send a command via serial port and return the response.

        Delegates command formatting, timing delays, and response normalization
        to the active ``mount_protocol`` instance, so this method works
        correctly for any registered protocol (LX200, NexStar, iOptron, etc.).

        Args:
            command: Command string (protocol-specific format).

        Returns:
            The normalized response string, or empty string on error.
        """
        if not self.serial_connection:
            return ""
        
        try:
            # Delegate command formatting to the active protocol
            command = self.mount_protocol.format_outgoing(command)
            
            # Clean the input buffer before sending (to avoid mixed responses)
            try:
                if self.serial_connection.in_waiting > 0:
                    self.serial_connection.reset_input_buffer()
            except OSError:
                pass
            
            # Send the command with detailed logging
            command_bytes = command.encode('ascii')
            self._log(f"TX serial: {command} (bytes: {command_bytes.hex()})")
            self.serial_connection.write(command_bytes)
            self.serial_connection.flush()  # Ensure the command is sent
            
            # Wait for the telescope to process the command
            # Timing is protocol-specific
            delay = self.mount_protocol.get_command_delay(command)
            time.sleep(delay)
            
            # Read the response with protocol-specific timeout
            timeout = self.mount_protocol.get_read_timeout(command)
            response = self._read_serial_response(timeout=timeout)
            
            # Detailed response logging
            if response:
                self._log(f"RX serial: '{response}' (length: {len(response)} bytes)")
            else:
                self._log(f"WARNING: No response received for {command}")
            
            # Protocol-specific response normalization
            if response:
                response = self.mount_protocol.normalize_response(
                    command, response, self._log)
            
            return response
            
        except Exception as e:
            self._log(f"Send error: {e}")
            import traceback
            self._log(f"Traceback: {traceback.format_exc()}")
            return ""
    
    def _send_tcp_command(self, command: str) -> str:
        """Send a command via TCP socket and return the response.

        Mirrors the behavior of _send_serial_command() but uses a TCP socket
        instead of a serial port.  Connection errors are **raised** so the
        caller (``_send_command_locked``) can attempt a transparent reconnect
        before giving up.

        Delegates command formatting, timing, and response normalization to
        the active ``mount_protocol`` instance.

        Args:
            command: Command string (protocol-specific format).

        Returns:
            The normalized response string, or empty string on timeout.

        Raises:
            ConnectionError, BrokenPipeError, OSError: on dead TCP socket.
        """
        if not self.socket_connection or not self.is_connected:
            return ""

        # Delegate command formatting to the active protocol
        command = self.mount_protocol.format_outgoing(command)

        # Send the command
        command_bytes = command.encode('ascii')
        self._log(f"TX WiFi: {command}")
        self.socket_connection.sendall(command_bytes)

        # Wait for the telescope to process the command (protocol-specific)
        delay = self.mount_protocol.get_command_delay(command)
        time.sleep(delay)

        # Read the response with protocol-specific timeout
        timeout = self.mount_protocol.get_read_timeout(command)
        response = self._read_tcp_response(timeout=timeout)

        # Detailed response logging
        if response:
            self._log(f"RX WiFi: '{response}' (length: {len(response)} bytes)")
        else:
            self._log(f"WARNING: No WiFi response received for {command}")

        # Protocol-specific response normalization
        if response:
            response = self.mount_protocol.normalize_response(
                command, response, self._log)

        return response
    
    def _read_tcp_response(self, timeout: float = 0.5) -> str:
        """Read a '#'-terminated response from the TCP socket.

        Reads one byte at a time (required — the ESP8266/ESP32 serial-
        to-TCP bridge delivers data byte-by-byte and recv(N>1) may
        block until the full N bytes arrive or the timeout expires).

        **Two-phase timeout**: the SmartWebServer runs a background
        ``status.update()`` task every ~1 s that sends 5-7 commands to
        OnStep via the shared serial line.  While that task is running,
        our command is queued and the first response byte can be delayed
        by up to ~200 ms (or more if commands time out).  To avoid
        reading a LATE response as the wrong command's response (which
        causes a persistent shift), we use a generous wait for the
        **first byte** and a short timeout for subsequent bytes:

            * ``timeout`` (default 0.5 s) — max wait for the first
              byte.  Covers the SmartWebServer background processing
              time (~70 ms for 5-7 commands at ~10 ms each) with
              generous margin.  Kept under 1 s so that if the SWS web
              page is open (saturating the serial line), a 7-command
              poll cycle completes in ~3.5 s instead of ~10.5 s.
            * 100 ms — max gap between consecutive bytes once the
              response starts arriving.

        **Connection-close detection**: if ``recv()`` returns ``b""``,
        the remote end has closed the connection (TCP FIN).  This is
        raised as ``ConnectionError`` so the caller can react
        immediately rather than silently returning empty strings.

        Args:
            timeout: Max seconds to wait for the first response byte.
                     Default 0.5 s.

        Returns:
            The '#'-terminated response string, or ``""`` on timeout.

        Raises:
            ConnectionError: if recv returns b"" (remote closed).
        """
        if not self.socket_connection:
            return ""

        response = ""
        max_len = 50

        # Time limit for the FIRST byte (generous — covers SWS
        # background serial traffic).
        first_byte_timeout = timeout
        # Time limit between SUBSEQUENT bytes (once the response
        # starts, bytes arrive in quick succession at serial speed).
        inter_byte_timeout = 0.1

        try:
            old_timeout = self.socket_connection.gettimeout()
            self.socket_connection.settimeout(first_byte_timeout)

            got_first_byte = False

            while True:
                try:
                    data = self.socket_connection.recv(1)
                    if not data:
                        # Remote sent FIN — connection is closed.
                        self.socket_connection.settimeout(old_timeout)
                        raise ConnectionError(
                            "recv returned empty (remote closed)")

                    if not got_first_byte:
                        got_first_byte = True
                        # Switch to short inter-byte timeout now
                        # that the response has started arriving.
                        self.socket_connection.settimeout(
                            inter_byte_timeout)

                    char = data.decode('ascii', errors='ignore')
                    if char.isprintable() or char == '#':
                        response += char
                    if char == '#':
                        break
                    if len(response) >= max_len:
                        break
                except socket.timeout:
                    break

            self.socket_connection.settimeout(old_timeout)

        except ConnectionError:
            raise  # propagate to caller
        except Exception as e:
            self._log(f"TCP read error: {e}")
            raise ConnectionError(f"TCP read failed: {e}")

        # Return responses that end with '#' (normal LX200 responses).
        # Also return short non-'#' responses (1-2 chars like "0" or "1")
        # because OnStep SET commands (:St, :Sg, :Sr, :Sd, :SL, :SC,
        # :SX, :MS, etc.) return just "0" or "1" without '#' terminator.
        if response:
            if response.endswith('#'):
                return response
            if len(response) <= 2:
                # Likely a SET command ack ("0" or "1") — return as-is
                return response
        return ""
    
    def _read_serial_response(self, timeout: float = 0.5) -> str:
        """
        Read a response from the telescope via serial port with timeout.

        Reads one byte at a time from the serial port until a '#' terminator
        is received, the timeout expires, or the maximum response length (50
        characters) is reached. Non-printable characters (except '#') are
        discarded.

        Important: This method does NOT flush the input buffer before reading.
        OnStep may respond very quickly, and calling reset_input_buffer() here
        would discard the response that has already arrived. Buffer management
        is the caller's responsibility.

        If the response does not end with '#' after the initial read, an
        additional 100ms wait is performed and any remaining buffered data is
        read. If a '#' is found in the combined data, the response is trimmed
        to end at the first '#'. Otherwise, an empty string is returned.

        Args:
            timeout: Maximum time in seconds to wait for a complete response.
                     Defaults to 0.5s.

        Returns:
            The '#'-terminated response string, or empty string if no valid
            response was received within the timeout. Responses that do not
            end with '#' after the retry are discarded (empty string returned).
        """
        if not self.serial_connection:
            return ""
        
        # DO NOT clean the buffer here - the response may have already arrived.
        # OnStep can respond very quickly, and reset_input_buffer() would delete the response.
        
        response = ""
        start_time = time.time()
        max_length = 50  # Limit length to avoid reading too much corrupted data
        
        # Brief initial wait for the response to arrive (especially for OnStep)
        time.sleep(0.05)  # 50ms initial wait
        
        try:
            while time.time() - start_time < timeout:
                if self.serial_connection.in_waiting > 0:
                    char = self.serial_connection.read(1).decode('ascii', errors='ignore')
                    # Ignore non-printable control characters except '#'
                    if char.isprintable() or char == '#':
                        response += char
                    if char == '#':
                        break
                    # Limit length to avoid infinite responses
                    if len(response) >= max_length:
                        break
                else:
                    # If no data available, wait a bit longer
                    time.sleep(0.01)
        except Exception as e:
            self._log(f"Response read error: {e}")
        
        # Validate that the response ends with '#' (standard LX200 format)
        if response and not response.endswith('#'):
            # Incomplete response - maybe there is more data
            # Wait a bit more and retry
            try:
                time.sleep(0.1)
                if self.serial_connection.in_waiting > 0:
                    additional = self.serial_connection.read(self.serial_connection.in_waiting).decode('ascii', errors='ignore')
                    response += additional
                    # Look for '#' in the complete response
                    if '#' in response:
                        response = response[:response.index('#') + 1]
            except OSError:
                pass
        
        # If the response still doesn't end with '#', return empty
        if response and not response.endswith('#'):
            return ""
        
        return response
    
    def _validate_position_response(self, response: str, coord_type: str = "RA") -> bool:
        """Validate that a position response is in a valid format.

        Checks the response string against expected coordinate formats for
        each coordinate type to filter out corrupted or garbled data.

        Validation rules by coord_type:
            - "RA": Must contain ':', parsed as HH:MM[:SS]. Hours must be
              0-23, minutes 0-59, seconds (if present) 0-59.
            - "Dec": Must contain '*', parsed as sDD*MM[:SS]. Degrees must
              be 0-90, minutes 0-59.
            - "Alt"/"Az": May contain '*' or ':', or be a plain decimal
              number. Alt must be -90 to +90, Az must be 0 to 360.

        Args:
            response: The raw '#'-terminated response string from the mount.
            coord_type: One of "RA", "Dec", "Alt", or "Az".

        Returns:
            True if the response is valid for the given coordinate type.
        """
        if not response or not response.endswith('#'):
            return False
        
        response_clean = response.rstrip('#').strip()
        
        if coord_type == "RA":
            # RA should be in format HH:MM:SS or HH:MM.SS
            # Verify it contains at least one ':' and valid digits
            if ':' not in response_clean:
                return False
            parts = response_clean.split(':')
            if len(parts) < 2:
                return False
            try:
                hours = int(parts[0])
                minutes = int(parts[1])
                # Validate reasonable ranges
                if hours < 0 or hours >= 24 or minutes < 0 or minutes >= 60:
                    return False
                if len(parts) > 2:
                    seconds = float(parts[2])
                    if seconds < 0 or seconds >= 60:
                        return False
            except (ValueError, IndexError):
                return False
        elif coord_type == "Dec":
            # Dec should be in format sDD*MM:SS or sDD*MM
            if '*' not in response_clean:
                return False
            parts = response_clean.split('*')
            if len(parts) < 2:
                return False
            try:
                degrees = int(parts[0].lstrip('+-'))
                if degrees < 0 or degrees > 90:
                    return False
                if ':' in parts[1]:
                    min_sec = parts[1].split(':')
                    minutes = int(min_sec[0])
                    if minutes < 0 or minutes >= 60:
                        return False
                else:
                    minutes = int(parts[1])
                    if minutes < 0 or minutes >= 60:
                        return False
            except (ValueError, IndexError):
                return False
        elif coord_type in ["Alt", "Az"]:
            # Alt/Az format similar to Dec
            if '*' not in response_clean and ':' not in response_clean:
                # May be in plain decimal degrees
                try:
                    value = float(response_clean.lstrip('+-'))
                    if coord_type == "Alt" and (value < -90 or value > 90):
                        return False
                    if coord_type == "Az" and (value < 0 or value > 360):
                        return False
                except ValueError:
                    return False
        
        return True
    
    def _read_loop(self):
        """
        Background thread that continuously polls the mount for position updates.

        This method runs in a daemon thread started after a successful connection.
        It implements a polling state machine with the following sequence on each
        iteration.  All I/O is performed under ``_io_lock`` to prevent
        interleaving with commands sent by the ``_io_worker`` thread.

        Polling sequence (repeated every ~2 seconds when stable):
            1. SLEW CHECK: Send :D# command to query slew status.
               - Response contains '|' -> mount is slewing, wait 1 second and
                 skip to next iteration (avoids reading unstable intermediate
                 positions).
               - Response is '#' only or no '|' -> mount is stationary, proceed.

            2. BUFFER CLEANUP (serial only): Flush any stale data from the
               serial input buffer to ensure clean reads.

            3. READ ALTITUDE: Send :GA# (Get Altitude). Wait 100ms between
               commands. Read response with 500ms timeout.

            4. READ AZIMUTH: Send :GZ# (Get Azimuth). Wait 100ms between
               commands. Read response with 500ms timeout.

            5. VALIDATE & NOTIFY: If both Alt and Az responses are valid
               (end with '#' and pass format validation), fire the
               on_altaz_update(alt_response, az_response) callback.

            6. SLEEP: Wait 2 seconds before the next polling cycle. This
               interval is intentionally long to avoid overwhelming OnStep
               with too-frequent commands and to reduce position jitter.

        Error recovery:
            - Serial errors: Any exception breaks out of the loop, effectively
              stopping the polling thread.
            - TCP connection errors (Windows error codes 10053/10054 for
              connection aborted/reset): The bridge is marked as disconnected,
              _running is set to False, on_disconnected callback is fired, and
              the thread exits.
            - Other TCP errors: Logged once, then the thread exits.
            - The thread also exits if _running is set to False (e.g. by
              disconnect()) or if is_connected becomes False.

        Transport differences:
            - Serial: Directly writes to self.serial_connection and uses
              _read_serial_response().
            - TCP: Verifies socket liveness via getpeername() before each
              cycle. Uses socket_connection.sendall() and _read_tcp_response().
              Wraps the entire TCP polling block in an additional try/except
              for connection-specific exceptions.
        """
        # Connection-lost error codes: Windows (10053, 10054) + Linux (103, 104)
        _CONN_LOST_CODES = {10053, 10054, 103, 104}

        # Track slewing state transitions for logging
        was_slewing = False

        # Position-based slew detection (fallback).  If position changes
        # by more than _SLEW_THRESHOLD_DEG between two successive polls the
        # mount is slewing.
        _SLEW_THRESHOLD_DEG = 0.3
        _prev_alt = None
        _prev_az = None

        def _do_disconnect(reason: str):
            """Helper: mark disconnected and fire callback."""
            if self._running:
                self._log(f"WARNING: {reason}")
            self.is_connected = False
            self._running = False
            if self.on_disconnected:
                self.on_disconnected()

        # Track consecutive poll failures.  A single failure is expected
        # each time the ESP closes the socket; _poll_send will auto-
        # reconnect and retry.  If _poll_send's retry also fails it
        # raises ConnectionError which we catch here — but we give the
        # outer loop one more chance with _ensure_tcp_connection()
        # before giving up entirely.
        _MAX_CONSECUTIVE_FAILURES = 3
        _consecutive_failures = 0

        self._log("Read loop started")

        while self._running and self.is_connected:
            try:
                # --- Poll position via the mount protocol ---
                # _poll_send() handles auto-reconnect internally for TCP.
                # If reconnect fails it raises ConnectionError.
                try:
                    pos: PositionData = self.mount_protocol.poll_position(self._poll_send)
                    _consecutive_failures = 0  # Reset on success
                except (ConnectionError, BrokenPipeError,
                        ConnectionResetError, ConnectionAbortedError) as e:
                    if not self._running:
                        break
                    _consecutive_failures += 1
                    if _consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                        _do_disconnect(f"Connection lost after {_consecutive_failures} "
                                       f"consecutive failures: {e}")
                        break
                    # One more reconnect attempt at the read-loop level
                    if self.connection_type == 'tcp':
                        self._log(f"Poll failed ({_consecutive_failures}/{_MAX_CONSECUTIVE_FAILURES}), "
                                  f"attempting reconnect...")
                        if self._ensure_tcp_connection():
                            time.sleep(0.1)
                            continue  # Retry the poll
                        else:
                            _do_disconnect(f"TCP reconnect failed after poll error: {e}")
                            break
                    else:
                        _do_disconnect(f"Connection lost during poll: {e}")
                        break
                except Exception as e:
                    error_code = getattr(e, 'winerror', None) or getattr(e, 'errno', None)
                    if error_code in _CONN_LOST_CODES:
                        _consecutive_failures += 1
                        if _consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                            _do_disconnect(f"Connection lost (code {error_code}): {e}")
                            break
                        if self.connection_type == 'tcp' and self._ensure_tcp_connection():
                            continue
                        _do_disconnect(f"Connection lost (code {error_code}): {e}")
                        break
                    elif self._running and self.is_connected:
                        self._log(f"Poll error: {e}")
                    continue

                is_slewing = pos.is_slewing

                # --- Method C: Position-based slew detection (fallback) ---
                if pos.alt_deg is not None and pos.az_deg is not None:
                    cur_alt = pos.alt_deg
                    cur_az = pos.az_deg
                    if _prev_alt is not None and _prev_az is not None:
                        delta_alt = abs(cur_alt - _prev_alt)
                        delta_az = abs(cur_az - _prev_az)
                        if delta_az > 180:
                            delta_az = 360 - delta_az
                        if (delta_alt > _SLEW_THRESHOLD_DEG or
                                delta_az > _SLEW_THRESHOLD_DEG):
                            is_slewing = True
                    _prev_alt = cur_alt
                    _prev_az = cur_az

                # Log slew state transitions
                if is_slewing and not was_slewing:
                    self._log("Telescope is slewing...")
                elif not is_slewing and was_slewing:
                    self._log("Slew complete - telescope stationary")

                # --- CALLBACK (outside lock) ---
                # Fire whenever we have ANY position data.  Alt-Az mounts
                # need alt/az; if only RA/Dec is available the callback
                # still fires so the UI updates.
                has_altaz = bool(pos.alt_str and pos.az_str)
                has_radec = bool(pos.ra_str and pos.dec_str)
                if has_altaz or has_radec:
                    if self.on_altaz_update:
                        self.on_altaz_update(
                            pos.alt_str or "", pos.az_str or "", is_slewing,
                            pos.ra_str, pos.dec_str,
                        )

                # Focuser position callback
                if pos.focuser_position is not None:
                    if self.on_focuser_position:
                        self.on_focuser_position(pos.focuser_position + '#')

                was_slewing = is_slewing

                # Pause between poll cycles.  The 7-command poll itself
                # takes ~70-100 ms when the serial line is clear (longer
                # if the SmartWebServer is busy).  A 200-500 ms pause
                # gives the ESP8266 breathing room and reduces serial
                # contention with the SmartWebServer's own background
                # status polling (every 1 s).  Net rate: ~2-3 Hz during
                # slew, ~1.5-2 Hz when idle — plenty for smooth UI.
                poll_interval = 0.20 if is_slewing else 0.50
                time.sleep(poll_interval)

            except (ConnectionError, ConnectionResetError,
                    ConnectionAbortedError, BrokenPipeError, OSError) as e:
                if not self._running:
                    break
                error_code = getattr(e, 'winerror', None) or getattr(e, 'errno', None)
                _do_disconnect(f"Connection error in read loop (code {error_code}): {e}")
                break
            except Exception as e:
                if not self._running:
                    break
                error_code = getattr(e, 'winerror', None) or getattr(e, 'errno', None)
                if error_code in _CONN_LOST_CODES:
                    _do_disconnect(f"Connection lost (code {error_code}): {e}")
                else:
                    _do_disconnect(f"Unexpected read error (code {error_code}): {e}")
                break
    
    def goto_altaz(self, alt_str: str, az_str: str) -> bool:
        """Slew the mount to the specified Alt/Az coordinates.

        Delegates to the active mount protocol for command building.
        This is the preferred GOTO method for Dobson Alt-Az mounts.

        Args:
            alt_str: Target altitude in LX200 format (e.g. "+45*30:00").
            az_str: Target azimuth in LX200 format (e.g. "180*00:00").

        Returns:
            True if the GOTO command was accepted by the mount.
        """
        if not self.is_connected:
            return False

        result = self.mount_protocol.goto_altaz(alt_str, az_str, self.send_command)
        if result.success:
            self._log(f"Alt/Az GOTO started towards Alt:{alt_str} Az:{az_str}")
        else:
            self._log(f"Alt/Az GOTO refused: {result.message}")
        return result.success

    def goto(self, ra: str, dec: str) -> bool:
        """GOTO using equatorial RA/Dec coordinates.

        Delegates to the active mount protocol. For Alt-Az mounts,
        prefer ``goto_altaz()`` when possible.

        Args:
            ra: Target Right Ascension in LX200 format (e.g. "12:30:00").
            dec: Target Declination in LX200 format (e.g. "+45*30:00").

        Returns:
            True if the GOTO command was accepted by the mount.
        """
        if not self.is_connected:
            return False

        result = self.mount_protocol.goto_radec(ra, dec, self.send_command)
        if result.success:
            self._log("GOTO started (via RA/Dec)")
        else:
            self._log(f"GOTO refused: {result.message}")
        return result.success

    def stop(self):
        """Stop all mount movement immediately."""
        if self.is_connected:
            self.mount_protocol.stop(self.send_command)
            self._log("Movement stopped")
    
    def _read_response(self, timeout: float = 0.5) -> str:
        """Read a response using the appropriate transport method.

        Convenience wrapper that delegates to _read_tcp_response() or
        _read_serial_response() based on the current connection_type.

        Args:
            timeout: Maximum time in seconds to wait for a response.

        Returns:
            The response string, or empty string if not connected or on error.
        """
        if self.connection_type == 'tcp':
            return self._read_tcp_response(timeout)
        elif self.connection_type == 'serial':
            return self._read_serial_response(timeout)
        else:
            return ""
    
    def force_position_update(self):
        """Force an immediate position read from the telescope.

        Uses the mount protocol to poll position, bypassing the normal
        polling interval.  Thread-safe via ``_poll_send``.
        """
        if not self.is_connected:
            return

        try:
            pos = self.mount_protocol.poll_position(self._poll_send)
            if pos.alt_str and pos.az_str:
                if self.on_altaz_update:
                    self.on_altaz_update(
                        pos.alt_str, pos.az_str, False,
                        pos.ra_str, pos.dec_str,
                    )
        except Exception as e:
            self._log(f"Error during forced position update: {e}")
    
    def sync_altaz(self, alt_str: str, az_str: str) -> bool:
        """Synchronize the mount's position to given Alt/Az coordinates.

        Delegates to the active mount protocol.

        Args:
            alt_str: Current true altitude in LX200 format (e.g. "+45*30:00").
            az_str: Current true azimuth in LX200 format (e.g. "180*00:00").

        Returns:
            True if the sync was accepted.
        """
        if not self.is_connected:
            return False

        result = self.mount_protocol.sync_altaz(alt_str, az_str, self.send_command)
        if result.success:
            self._log(f"Synced to Alt:{alt_str} Az:{az_str}")
        else:
            self._log(f"Sync failed: {result.message}")
        return result.success

    def sync(self, ra: str, dec: str) -> bool:
        """Synchronize via RA/Dec coordinates.

        Delegates to the active mount protocol.

        Args:
            ra: Current true Right Ascension in LX200 format.
            dec: Current true Declination in LX200 format.

        Returns:
            True if the sync was accepted.
        """
        if not self.is_connected:
            return False

        result = self.mount_protocol.sync_radec(ra, dec, self.send_command)
        if result.success:
            self._log("Synced (via RA/Dec)")
        else:
            self._log(f"Sync failed: {result.message}")
        return result.success
    
    def _log(self, message: str):
        """Log a diagnostic message via the on_log callback or stdout.

        If the on_log callback is set, the message is passed to it (typically
        routed to the application's log panel). Otherwise, the message is
        printed to stdout as a fallback.

        Args:
            message: The diagnostic/status message to log.
        """
        if self.on_log:
            self.on_log(message)
        else:
            _logger.debug(message)


class VirtualTelescope:
    """
    Virtual telescope for testing without physical hardware.

    Simulates an LX200-compatible telescope mount with basic state tracking
    (current RA/Dec, slewing status, target coordinates). This class is
    primarily used during development and testing when no real mount is
    available.

    Note: The process_command() method is a stub; actual command simulation
    is delegated to LX200Protocol in main.py.
    """
    
    def __init__(self):
        """Initialize the virtual telescope at coordinates (0, 0)."""
        self.ra_hours = 0.0
        self.dec_degrees = 0.0
        self.is_slewing = False
        self.target_ra = 0.0
        self.target_dec = 0.0
        
    def process_command(self, command: str) -> Optional[str]:
        """Process an LX200 command and return a simulated response.

        This method is a stub. Actual command processing is delegated to
        the LX200Protocol class in main.py.

        Args:
            command: An LX200 command string (e.g. ":GR#").

        Returns:
            None (stub implementation).
        """
        # This class delegates to LX200Protocol in main.py
        return None
