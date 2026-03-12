package com.telescopecontroller.serial

import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import android.os.Build
import android.util.Log
import com.hoho.android.usbserial.driver.UsbSerialDriver
import com.hoho.android.usbserial.driver.UsbSerialPort
import com.hoho.android.usbserial.driver.UsbSerialProber
import java.io.IOException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * USB serial bridge for telescope mount communication.
 *
 * Wraps the usb-serial-for-android library to provide a simple read/write
 * interface that the Python telescope_bridge.py can call through Chaquopy.
 *
 * Supported chips (same ones used in telescope serial cables):
 *   - FTDI FT232R/FT2232H
 *   - Silicon Labs CP2102/CP2104
 *   - WCH CH340/CH341
 *   - Prolific PL2303
 *
 * The Python side calls:
 *   serial_bridge.list_ports()             -> [{"name": "...", "vid": 1234, ...}]
 *   serial_bridge.connect(port_index, baud) -> True/False
 *   serial_bridge.send(command_str)        -> response_str
 *   serial_bridge.disconnect()
 *
 * NOTE: For v1.0, WiFi TCP mode is recommended (zero changes needed in Python).
 * USB serial is provided for setups without WiFi-enabled mounts (e.g. older
 * OnStep boards without ESP32).
 */
class UsbSerialManager(private val context: Context) {

    companion object {
        private const val TAG = "UsbSerial"
        private const val ACTION_USB_PERMISSION = "com.telescopecontroller.USB_PERMISSION"
        private const val DEFAULT_BAUD = 9600
        private const val READ_TIMEOUT_MS = 1000
        private const val WRITE_TIMEOUT_MS = 1000
        private const val PERMISSION_TIMEOUT_SEC = 30L
    }

    private var port: UsbSerialPort? = null
    private val readBuffer = ByteArray(4096)

    /**
     * List all connected USB serial devices.
     * Returns a list of maps describing each device.
     */
    fun listPorts(): List<Map<String, Any>> {
        val usbManager = context.getSystemService(Context.USB_SERVICE) as UsbManager
        val drivers = UsbSerialProber.getDefaultProber().findAllDrivers(usbManager)

        return drivers.mapIndexed { index, driver ->
            val device: UsbDevice = driver.device
            mapOf(
                "index" to index,
                "name" to (device.productName ?: "USB Serial ${device.deviceId}"),
                "vid" to device.vendorId,
                "pid" to device.productId,
                "manufacturer" to (device.manufacturerName ?: "Unknown"),
                "port_count" to driver.ports.size
            )
        }
    }

    /**
     * Request USB permission for a device and wait for the user response.
     * Returns true if permission was granted (or was already granted).
     */
    private fun ensurePermission(usbManager: UsbManager, device: UsbDevice): Boolean {
        if (usbManager.hasPermission(device)) {
            Log.d(TAG, "USB permission already granted for ${device.productName}")
            return true
        }

        Log.i(TAG, "Requesting USB permission for ${device.productName}...")

        val granted = AtomicBoolean(false)
        val latch = CountDownLatch(1)

        val receiver = object : BroadcastReceiver() {
            override fun onReceive(ctx: Context, intent: Intent) {
                if (ACTION_USB_PERMISSION == intent.action) {
                    val permDevice = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                        intent.getParcelableExtra(UsbManager.EXTRA_DEVICE, UsbDevice::class.java)
                    } else {
                        @Suppress("DEPRECATION")
                        intent.getParcelableExtra(UsbManager.EXTRA_DEVICE)
                    }
                    if (permDevice?.deviceId == device.deviceId) {
                        granted.set(intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false))
                        Log.i(TAG, "USB permission ${if (granted.get()) "GRANTED" else "DENIED"} for ${device.productName}")
                        latch.countDown()
                    }
                }
            }
        }

        val filter = IntentFilter(ACTION_USB_PERMISSION)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            context.registerReceiver(receiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            context.registerReceiver(receiver, filter)
        }

        val pendingIntent = PendingIntent.getBroadcast(
            context, 0,
            Intent(ACTION_USB_PERMISSION),
            PendingIntent.FLAG_MUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        usbManager.requestPermission(device, pendingIntent)

        // Block until the user taps Allow/Deny or timeout
        val answered = latch.await(PERMISSION_TIMEOUT_SEC, TimeUnit.SECONDS)

        try {
            context.unregisterReceiver(receiver)
        } catch (_: Exception) {}

        if (!answered) {
            Log.w(TAG, "USB permission request timed out after ${PERMISSION_TIMEOUT_SEC}s")
            return false
        }

        return granted.get()
    }

    /**
     * Connect to a USB serial port by driver index.
     * Automatically requests USB permission if not yet granted.
     * Returns true on success.
     */
    fun connect(driverIndex: Int, baudRate: Int = DEFAULT_BAUD): Boolean {
        val usbManager = context.getSystemService(Context.USB_SERVICE) as UsbManager
        val drivers = UsbSerialProber.getDefaultProber().findAllDrivers(usbManager)

        if (driverIndex < 0 || driverIndex >= drivers.size) {
            Log.e(TAG, "Invalid driver index: $driverIndex (${drivers.size} available)")
            return false
        }

        val driver = drivers[driverIndex]

        // Request permission if needed (shows system dialog, blocks until user responds)
        if (!ensurePermission(usbManager, driver.device)) {
            Log.e(TAG, "USB permission denied for ${driver.device.productName}")
            return false
        }

        val connection = usbManager.openDevice(driver.device)
        if (connection == null) {
            Log.e(TAG, "Failed to open USB device ${driver.device.productName} (openDevice returned null)")
            return false
        }

        try {
            port = driver.ports[0].apply {
                open(connection)
                setParameters(baudRate, 8, UsbSerialPort.STOPBITS_1, UsbSerialPort.PARITY_NONE)
                dtr = true   // Data Terminal Ready
                rts = true   // Request To Send
            }
            Log.i(TAG, "Connected to ${driver.device.productName} at $baudRate baud")
            return true
        } catch (e: IOException) {
            Log.e(TAG, "Failed to open port: ${e.message}")
            port = null
            return false
        }
    }

    /**
     * Send an LX200 command string and read the response.
     *
     * LX200 commands end with '#' and responses also end with '#'.
     * This method handles the read loop until '#' is seen or timeout.
     */
    fun send(command: String): String {
        val serialPort = port ?: run {
            Log.w(TAG, "Not connected")
            return ""
        }

        return try {
            // Write command
            serialPort.write(command.toByteArray(Charsets.US_ASCII), WRITE_TIMEOUT_MS)

            // Read response until '#' terminator or timeout
            val response = StringBuilder()
            val deadline = System.currentTimeMillis() + READ_TIMEOUT_MS

            while (System.currentTimeMillis() < deadline) {
                val len = serialPort.read(readBuffer, 100)
                if (len > 0) {
                    val chunk = String(readBuffer, 0, len, Charsets.US_ASCII)
                    response.append(chunk)
                    if (chunk.contains('#')) break
                }
            }

            response.toString()
        } catch (e: IOException) {
            Log.e(TAG, "I/O error: ${e.message}")
            ""
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Raw I/O methods (pyserial-compatible -- used by AndroidSerialPort)
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Write raw bytes to the serial port.
     * Returns the number of bytes written, or 0 on error.
     */
    fun rawWrite(data: ByteArray): Int {
        val serialPort = port ?: return 0
        return try {
            serialPort.write(data, WRITE_TIMEOUT_MS)
            data.size
        } catch (e: IOException) {
            Log.e(TAG, "rawWrite error: ${e.message}")
            0
        }
    }

    /**
     * Read raw bytes from the serial port with a timeout.
     *
     * @param maxBytes  Maximum number of bytes to read.
     * @param timeoutMs Timeout in milliseconds (0 = non-blocking poll).
     * @return The bytes that were read (may be shorter than maxBytes),
     *         or an empty array if nothing was available / on error.
     */
    fun rawRead(maxBytes: Int, timeoutMs: Int): ByteArray {
        val serialPort = port ?: return ByteArray(0)
        return try {
            val buffer = ByteArray(maxBytes)
            val n = serialPort.read(buffer, timeoutMs)
            if (n > 0) buffer.copyOf(n) else ByteArray(0)
        } catch (e: IOException) {
            // timeout=0 (non-blocking) legitimately returns 0; don't log that
            if (timeoutMs > 0) Log.e(TAG, "rawRead error: ${e.message}")
            ByteArray(0)
        }
    }

    /**
     * Flush hardware receive and transmit buffers.
     * Not all USB-serial chips support this; failures are silently ignored.
     */
    fun purgeBuffers() {
        try {
            port?.purgeHwBuffers(true, true)
        } catch (_: Exception) {
            // purgeHwBuffers not supported by every driver -- that's OK
        }
    }

    /**
     * Disconnect from the USB serial port.
     */
    fun disconnect() {
        try {
            port?.close()
        } catch (e: IOException) {
            Log.w(TAG, "Error closing port: ${e.message}")
        }
        port = null
        Log.i(TAG, "Disconnected")
    }

    /**
     * Check if a port is currently open.
     */
    fun isConnected(): Boolean = port != null
}
