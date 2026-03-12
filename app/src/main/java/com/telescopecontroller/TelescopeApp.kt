package com.telescopecontroller

import android.app.NotificationChannel
import android.app.NotificationManager
import android.os.Build
import android.util.Log
import com.chaquo.python.Python
import com.chaquo.python.android.PyApplication
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Application class -- initializes Python interpreter and notification channels.
 *
 * Extends [PyApplication] so that Chaquopy's native libraries and Python runtime
 * are properly loaded before any Python code is invoked.  The interpreter is then
 * started on a background thread to avoid ANR on first launch (Python extraction
 * can take several seconds).
 * [awaitPython] lets callers block until the interpreter is ready.
 */
class TelescopeApp : PyApplication() {

    companion object {
        const val CHANNEL_TRACKING = "tracking_channel"
        const val CHANNEL_ALERTS  = "alerts_channel"
        private const val TAG = "TelescopeApp"
    }

    /** Latch that gates access to the Python interpreter. */
    private val pythonReady = CountDownLatch(1)

    /** Non-null if Python initialization failed. */
    @Volatile
    var pythonInitError: String? = null
        private set

    override fun onCreate() {
        super.onCreate()

        // ── Notification channels (required on Android 8+) ────────────
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val nm = getSystemService(NotificationManager::class.java)
            nm?.let { notifManager ->
                notifManager.createNotificationChannel(
                    NotificationChannel(
                        CHANNEL_TRACKING,
                        "Tracking Service",
                        NotificationManager.IMPORTANCE_LOW
                    ).apply {
                        description = "Keeps the telescope tracking loop alive"
                    }
                )

                notifManager.createNotificationChannel(
                    NotificationChannel(
                        CHANNEL_ALERTS,
                        "Telescope Alerts",
                        NotificationManager.IMPORTANCE_HIGH
                    ).apply {
                        description = "Connection loss, tracking errors, weather warnings"
                    }
                )
            }
        }

        // ── Python is started by PyApplication.onCreate() (super call) ──
        // The interpreter is already available via Python.getInstance().
        // Signal readiness to any threads waiting on awaitPython().
        try {
            if (Python.isStarted()) {
                Log.i(TAG, "Python interpreter ready (started by PyApplication)")
            } else {
                Log.w(TAG, "Python not started after super.onCreate() -- should not happen")
                pythonInitError = "Python interpreter was not started by PyApplication"
            }
        } catch (e: Exception) {
            Log.e(TAG, "FATAL: Python initialization failed", e)
            pythonInitError = e.message ?: "Unknown Python init error"
        } finally {
            pythonReady.countDown()
        }
    }

    /**
     * Block the calling thread until the Python interpreter is ready.
     * Returns true if Python started successfully, false on failure.
     *
     * @param timeoutSec Maximum time to wait (default 60s).
     */
    fun awaitPython(timeoutSec: Long = 60): Boolean {
        return try {
            pythonReady.await(timeoutSec, TimeUnit.SECONDS) && pythonInitError == null
        } catch (e: InterruptedException) {
            Log.w(TAG, "Interrupted while waiting for Python")
            false
        }
    }
}
