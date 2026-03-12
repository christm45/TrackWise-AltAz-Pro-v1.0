package com.telescopecontroller.service

import android.app.Notification
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import com.chaquo.python.Python
import com.telescopecontroller.MainActivity
import com.telescopecontroller.R
import com.telescopecontroller.TelescopeApp
import com.telescopecontroller.camera.CameraManager
import com.telescopecontroller.network.CellularHttpClient
import com.telescopecontroller.serial.UsbSerialManager
import java.net.HttpURLConnection
import java.net.URL
import kotlin.concurrent.thread

/**
 * Foreground service that runs the Python backend (HEADLESS_SERVER + Flask).
 *
 * Why a foreground service?
 *   - Android kills background processes aggressively.  A telescope tracking
 *     loop must survive screen-off, app switching, and doze mode.
 *   - The persistent notification tells the user tracking is active and
 *     provides a quick-return tap target.
 *
 * Architecture:
 *   1. Start the Chaquopy Python interpreter
 *   2. Call android_bridge.main.start_server(data_dir, cache_dir)
 *   3. Inject CameraManager and UsbSerialManager into Python bridges
 *   4. That launches HEADLESS_SERVER + Flask on 127.0.0.1:8080
 *   5. MainActivity's WebView connects to that local URL
 */
class TelescopeService : Service() {

    companion object {
        private const val TAG = "TelescopeService"
        private const val NOTIFICATION_ID = 1
        private const val FLASK_PORT = 8080
    }

    // Binder for MainActivity to query state
    inner class LocalBinder : Binder() {
        fun getService(): TelescopeService = this@TelescopeService
    }

    private val binder = LocalBinder()
    private var pythonThread: Thread? = null
    private var wakeLock: PowerManager.WakeLock? = null

    // Hardware managers -- created here, injected into Python bridge
    private var cameraManager: CameraManager? = null
    private var usbSerialManager: UsbSerialManager? = null
    private var cellularHttpClient: CellularHttpClient? = null

    @Volatile
    var isServerReady = false
        private set

    /** True if the Python backend failed to start. */
    @Volatile
    var isServerFailed = false
        private set

    /** Error message if the server failed. */
    @Volatile
    var failureMessage: String? = null
        private set

    private val readyCallbacks = mutableListOf<() -> Unit>()

    // ── Service lifecycle ──────────────────────────────────────────────

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "Service created")

        // Acquire partial wake lock to keep CPU alive during tracking
        val pm = getSystemService(POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "TelescopeController::TrackingLock"
        ).apply {
            acquire(12 * 60 * 60 * 1000L)  // 12-hour max (full night session)
        }

        // Create hardware managers
        cameraManager = CameraManager(this)
        usbSerialManager = UsbSerialManager(this)
        cellularHttpClient = CellularHttpClient(this).also {
            it.requestCellularNetwork()
        }

        // Android 14 (API 34) requires the foreground service type to be
        // passed explicitly to startForeground().  Without it the system
        // throws MissingForegroundServiceTypeException -- which was the
        // root cause of the "SystemExit:1" crash when launching without ADB.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(
                NOTIFICATION_ID,
                buildNotification("Starting..."),
                ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE
            )
        } else {
            startForeground(NOTIFICATION_ID, buildNotification("Starting..."))
        }
        startPythonBackend()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // If the system kills us, restart automatically
        return START_STICKY
    }

    /**
     * Called when the user swipes the app away from the recents screen.
     * Ensures we shut down the Python backend and release the Flask port
     * so the next launch doesn't get "Address already in use".
     */
    override fun onTaskRemoved(rootIntent: Intent?) {
        Log.i(TAG, "Task removed -- cleaning up")
        stopSelf()
        super.onTaskRemoved(rootIntent)
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.i(TAG, "Service destroyed -- shutting down Python backend")

        // Tell the Python server to shut down gracefully
        try {
            val py = Python.getInstance()
            val bridge = py.getModule("android_bridge.main")
            bridge.callAttr("stop_server")
        } catch (e: Exception) {
            Log.w(TAG, "Error stopping Python server: ${e.message}")
        }

        // Close hardware managers
        try {
            cameraManager?.close()
        } catch (e: Exception) {
            Log.w(TAG, "Error closing camera: ${e.message}")
        }
        try {
            usbSerialManager?.disconnect()
        } catch (e: Exception) {
            Log.w(TAG, "Error closing serial: ${e.message}")
        }
        try {
            cellularHttpClient?.release()
        } catch (e: Exception) {
            Log.w(TAG, "Error releasing cellular client: ${e.message}")
        }

        wakeLock?.let {
            if (it.isHeld) it.release()
        }

        pythonThread?.interrupt()
    }

    // ── Python backend ─────────────────────────────────────────────────

    private fun startPythonBackend() {
        pythonThread = thread(name = "PythonBackend", isDaemon = true) {
            try {
                Log.i(TAG, "Starting Python backend...")
                updateNotification("Initializing Python...")

                // Wait for Python interpreter to finish initializing
                // (started on background thread in TelescopeApp.onCreate)
                val app = application as TelescopeApp
                if (!app.awaitPython(60)) {
                    val err = app.pythonInitError ?: "Python init timed out"
                    throw RuntimeException("Python failed to start: $err")
                }

                val py = Python.getInstance()
                val bridge = py.getModule("android_bridge.main")

                // Pass Android-specific paths to Python
                val dataDir = filesDir.absolutePath       // persistent app storage
                val cacheDir = cacheDir.absolutePath       // temp files (plate-solve images)

                Log.i(TAG, "Data dir: $dataDir")
                Log.i(TAG, "Cache dir: $cacheDir")

                // start_server() applies Android patches, creates HeadlessTelescopeApp,
                // calls app.start() (which launches Flask + update loop on daemon
                // threads), then returns immediately.
                updateNotification("Starting server...")
                Log.i(TAG, "Calling start_server...")

                bridge.callAttr("start_server", dataDir, cacheDir, FLASK_PORT)

                Log.i(TAG, "start_server returned, injecting hardware managers...")

                // ── Inject CameraManager into Python camera_bridge ─────────
                try {
                    val cameraBridge = py.getModule("android_bridge.camera_bridge")
                    cameraBridge.callAttr("set_camera_manager", cameraManager)
                    Log.i(TAG, "CameraManager injected into Python bridge")
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to inject CameraManager: ${e.message}")
                }

                // ── Inject UsbSerialManager into Python serial_bridge ──────
                try {
                    val serialBridge = py.getModule("android_bridge.serial_bridge")
                    serialBridge.callAttr("set_serial_manager", usbSerialManager)
                    Log.i(TAG, "UsbSerialManager injected into Python bridge")
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to inject UsbSerialManager: ${e.message}")
                }

                // ── Inject CellularHttpClient into Python network_bridge ──
                try {
                    val networkBridge = py.getModule("android_bridge.network_bridge")
                    networkBridge.callAttr("set_cellular_client", cellularHttpClient)
                    Log.i(TAG, "CellularHttpClient injected into Python bridge")
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to inject CellularHttpClient: ${e.message}")
                }

                Log.i(TAG, "Hardware managers injected, polling Flask...")
                updateNotification("Waiting for Flask...")

                // Poll until Flask is responding on localhost
                val flaskReady = waitForFlask()

                if (!flaskReady) {
                    Log.e(TAG, "Flask never responded -- marking server as failed")
                    isServerFailed = true
                    failureMessage = "Flask web server did not start within 30 seconds"
                    updateNotification("Error: Flask timeout")
                    // Fire callbacks anyway so Activity can show an error
                }

                // Notify waiting callbacks (inside synchronized to avoid race)
                synchronized(readyCallbacks) {
                    isServerReady = true
                    readyCallbacks.forEach { it() }
                    readyCallbacks.clear()
                }

                if (!isServerFailed) {
                    updateNotification("Tracking ready")
                    Log.i(TAG, "Python backend ready on port $FLASK_PORT")
                }

            } catch (e: Exception) {
                // Extract the most useful error message.  Chaquopy wraps
                // Python exceptions as PyException; their message already
                // includes the Python traceback.
                val rawMsg = e.message ?: "Unknown error"
                val shortMsg = when {
                    rawMsg.contains("SystemExit") ->
                        "Python SystemExit during startup -- port $FLASK_PORT may be busy. " +
                        "Force-stop the app and relaunch."
                    rawMsg.contains("Address already in use") ->
                        "Port $FLASK_PORT already in use. Force-stop the app and retry."
                    rawMsg.contains("MissingForegroundServiceType") ->
                        "Foreground service type error (Android 14). Please update the app."
                    else -> rawMsg
                }
                Log.e(TAG, "Python backend failed: $shortMsg", e)
                isServerFailed = true
                failureMessage = shortMsg
                updateNotification("Error: $shortMsg")

                // Fire callbacks so Activity knows about the failure
                synchronized(readyCallbacks) {
                    isServerReady = true  // technically "done starting" (failed)
                    readyCallbacks.forEach { it() }
                    readyCallbacks.clear()
                }

                // Stop service after brief delay so notification is visible
                thread {
                    Thread.sleep(5000)
                    stopSelf()
                }
            }
        }
    }

    /**
     * Poll Flask until it responds on localhost.
     * Returns true if Flask is ready, false if timed out.
     */
    private fun waitForFlask(): Boolean {
        val maxWait = 30_000L  // 30 seconds max
        val start = System.currentTimeMillis()

        while (System.currentTimeMillis() - start < maxWait) {
            try {
                val url = URL("http://127.0.0.1:$FLASK_PORT/api/status")
                val conn = url.openConnection() as HttpURLConnection
                conn.connectTimeout = 500
                conn.readTimeout = 500
                conn.requestMethod = "GET"
                val code = conn.responseCode
                conn.disconnect()
                if (code == 200) {
                    Log.i(TAG, "Flask ready after ${System.currentTimeMillis() - start}ms")
                    return true
                }
            } catch (_: Exception) {
                // Not ready yet
            }
            Thread.sleep(250)
        }

        Log.w(TAG, "Flask did not respond within ${maxWait}ms")
        return false
    }

    // ── Callbacks ──────────────────────────────────────────────────────

    /**
     * Register a callback to be invoked when the server is ready (or has
     * finished attempting to start).  Thread-safe.
     */
    fun waitForReady(callback: () -> Unit) {
        synchronized(readyCallbacks) {
            if (isServerReady) {
                callback()
            } else {
                readyCallbacks.add(callback)
            }
        }
    }

    // ── Notifications ──────────────────────────────────────────────────

    private fun buildNotification(text: String): Notification {
        val tapIntent = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        return NotificationCompat.Builder(this, TelescopeApp.CHANNEL_TRACKING)
            .setContentTitle("TrackWise-AltAzPro")
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_telescope)
            .setContentIntent(tapIntent)
            .setOngoing(true)
            .setSilent(true)
            .build()
    }

    private fun updateNotification(text: String) {
        try {
            val nm = getSystemService(NOTIFICATION_SERVICE) as android.app.NotificationManager
            nm.notify(NOTIFICATION_ID, buildNotification(text))
        } catch (e: Exception) {
            Log.w(TAG, "Failed to update notification: ${e.message}")
        }
    }
}
