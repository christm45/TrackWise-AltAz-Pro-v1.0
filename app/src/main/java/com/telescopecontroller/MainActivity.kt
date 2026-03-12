package com.telescopecontroller

import android.Manifest
import android.annotation.SuppressLint
import android.app.DownloadManager
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.IBinder
import android.util.Log
import android.view.View
import android.view.WindowInsetsController
import android.webkit.ConsoleMessage
import android.webkit.CookieManager
import android.webkit.URLUtil
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.telescopecontroller.service.TelescopeService

/**
 * Main activity -- hosts a full-screen WebView that loads the Flask UI
 * served by the embedded Python backend.
 *
 * Lifecycle:
 *   onCreate  -> start TelescopeService (foreground, keeps Python alive)
 *   onResume  -> bind to service, wait for Flask ready, load WebView
 *   onPause   -> unbind (service stays alive in foreground)
 *   onDestroy -> destroy WebView, if finishing stop service
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "TelescopeMain"
        private const val FLASK_URL = "http://127.0.0.1:8080"
        private const val PERMISSION_REQUEST_CODE = 1001
    }

    private lateinit var webView: WebView
    private lateinit var splashStatus: TextView
    private lateinit var splashProgress: ProgressBar

    private var service: TelescopeService? = null
    private var bound = false
    private var flaskLoaded = false

    // ── Permissions required at runtime ────────────────────────────────
    private val requiredPermissions: Array<String>
        get() = buildList {
            add(Manifest.permission.CAMERA)
            add(Manifest.permission.ACCESS_FINE_LOCATION)
            // Android 12+ requires COARSE alongside FINE
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                add(Manifest.permission.ACCESS_COARSE_LOCATION)
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                add(Manifest.permission.POST_NOTIFICATIONS)
            }
        }.toTypedArray()

    // ── Service connection ─────────────────────────────────────────────
    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            val localBinder = binder as? TelescopeService.LocalBinder
            if (localBinder == null) {
                Log.e(TAG, "Failed to bind: binder is null or wrong type")
                return
            }
            service = localBinder.getService()
            bound = true
            Log.i(TAG, "Bound to TelescopeService")

            // Wait for Flask to be ready, then load the WebView
            service?.waitForReady {
                runOnUiThread {
                    if (service?.isServerFailed == true) {
                        showError(service?.failureMessage ?: "Server failed to start")
                    } else {
                        loadFlaskUI()
                    }
                }
            }
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            service = null
            bound = false
            Log.w(TAG, "TelescopeService disconnected unexpectedly")
        }
    }

    // ── Activity lifecycle ─────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        webView = findViewById(R.id.webview)
        splashStatus = findViewById(R.id.splash_status)
        splashProgress = findViewById(R.id.splash_progress)

        setupWebView()
        setupBackNavigation()
        requestPermissionsIfNeeded()

        // Start the Python backend as a foreground service
        val intent = Intent(this, TelescopeService::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent)
        } else {
            startService(intent)
        }
    }

    override fun onResume() {
        super.onResume()
        // Bind so we can query service state
        if (!bound) {
            Intent(this, TelescopeService::class.java).also {
                bindService(it, serviceConnection, Context.BIND_AUTO_CREATE)
            }
        }
        hideSystemUI()
    }

    override fun onPause() {
        super.onPause()
        if (bound) {
            unbindService(serviceConnection)
            bound = false
        }
    }

    override fun onDestroy() {
        // Destroy WebView to prevent memory leaks
        try {
            val parent = webView.parent as? android.view.ViewGroup
            parent?.removeView(webView)
            webView.stopLoading()
            webView.clearCache(true)
            webView.destroy()
        } catch (e: Exception) {
            Log.w(TAG, "Error destroying WebView: ${e.message}")
        }

        if (isFinishing) {
            // User explicitly closed the app -- shut down the backend
            stopService(Intent(this, TelescopeService::class.java))
        }
        super.onDestroy()
    }

    // ── Back navigation (modern API) ───────────────────────────────────

    private fun setupBackNavigation() {
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (webView.canGoBack()) {
                    webView.goBack()
                } else {
                    // Let the system handle it (exit activity)
                    isEnabled = false
                    onBackPressedDispatcher.onBackPressed()
                }
            }
        })
    }

    // ── WebView setup ──────────────────────────────────────────────────

    @SuppressLint("SetJavaScriptEnabled")
    private fun setupWebView() {
        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            mediaPlaybackRequiresUserGesture = false
            allowFileAccess = false  // not needed; Flask serves everything over HTTP
            // Allow HTTP on localhost (Flask)
            mixedContentMode = android.webkit.WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
            // Disable caching to always get fresh Flask UI
            cacheMode = android.webkit.WebSettings.LOAD_NO_CACHE
            // Enable geolocation for the "Use GPS" button in the Location tab
            setGeolocationEnabled(true)
        }

        webView.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(
                view: WebView?, request: WebResourceRequest?
            ): Boolean {
                // Keep all navigation inside the WebView
                val url = request?.url?.toString() ?: return false
                if (url.startsWith(FLASK_URL)) return false
                // Block external navigation
                return true
            }

            override fun onPageFinished(view: WebView?, url: String?) {
                super.onPageFinished(view, url)
                // Hide splash, show WebView
                findViewById<View>(R.id.splash_container).visibility = View.GONE
                webView.visibility = View.VISIBLE
                flaskLoaded = true
                Log.i(TAG, "Flask UI loaded: $url")
            }

            override fun onReceivedError(
                view: WebView?,
                request: WebResourceRequest?,
                error: WebResourceError?
            ) {
                super.onReceivedError(view, request, error)
                // Only handle main frame errors
                if (request?.isForMainFrame == true) {
                    Log.e(TAG, "WebView error: ${error?.description} (${error?.errorCode})")
                    runOnUiThread {
                        showSplashWithRetry("Connection error. Retrying...")
                    }
                }
            }
        }

        // Handle file downloads (session ZIP, camera snapshot, etc.)
        webView.setDownloadListener { url, userAgent, contentDisposition, mimetype, contentLength ->
            Log.i(TAG, "Download request: url=$url mime=$mimetype len=$contentLength")
            try {
                val request = DownloadManager.Request(Uri.parse(url))
                // Guess filename from Content-Disposition header or URL
                val filename = URLUtil.guessFileName(url, contentDisposition, mimetype)
                request.setTitle(filename)
                request.setDescription("TrackWise-AltAzPro download")
                request.setNotificationVisibility(
                    DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED
                )
                request.setDestinationInExternalPublicDir(
                    Environment.DIRECTORY_DOWNLOADS, filename
                )
                request.setMimeType(mimetype)
                // Forward cookies (Flask session)
                val cookie = CookieManager.getInstance().getCookie(url)
                if (cookie != null) {
                    request.addRequestHeader("Cookie", cookie)
                }
                request.addRequestHeader("User-Agent", userAgent)

                val dm = getSystemService(DOWNLOAD_SERVICE) as DownloadManager
                dm.enqueue(request)

                Toast.makeText(
                    this@MainActivity,
                    "Downloading $filename ...",
                    Toast.LENGTH_SHORT
                ).show()
                Log.i(TAG, "Download enqueued: $filename")
            } catch (e: Exception) {
                Log.e(TAG, "Download failed: ${e.message}", e)
                Toast.makeText(
                    this@MainActivity,
                    "Download failed: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
            }
        }

        // Forward JS console.log to Android logcat + enable geolocation
        webView.webChromeClient = object : WebChromeClient() {
            override fun onConsoleMessage(msg: ConsoleMessage?): Boolean {
                msg?.let {
                    Log.d("WebView", "${it.sourceId()}:${it.lineNumber()} ${it.message()}")
                }
                return true
            }

            // Auto-grant geolocation permission to our localhost Flask UI.
            // The "Use GPS" button calls navigator.geolocation.getCurrentPosition()
            // which triggers this callback in the WebView.
            override fun onGeolocationPermissionsShowPrompt(
                origin: String?,
                callback: android.webkit.GeolocationPermissions.Callback?
            ) {
                // Only grant to our own Flask server on localhost
                if (origin?.contains("127.0.0.1") == true || origin?.contains("localhost") == true) {
                    callback?.invoke(origin, true, true)  // allow, remember
                    Log.i(TAG, "Geolocation granted to $origin")
                } else {
                    callback?.invoke(origin, false, false)
                    Log.w(TAG, "Geolocation denied for external origin: $origin")
                }
            }
        }
    }

    private fun loadFlaskUI() {
        if (flaskLoaded) return  // already loaded; avoid double-load on rebind
        splashStatus.text = "Loading interface..."
        webView.loadUrl(FLASK_URL)
    }

    private fun showError(message: String) {
        splashStatus.text = "Error: $message"
        splashProgress.visibility = View.GONE
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }

    private fun showSplashWithRetry(message: String) {
        if (!flaskLoaded) return  // still on splash, no action needed
        // Show splash again and retry after delay
        findViewById<View>(R.id.splash_container).visibility = View.VISIBLE
        webView.visibility = View.GONE
        splashStatus.text = message
        splashProgress.visibility = View.VISIBLE
        flaskLoaded = false
        webView.postDelayed({
            webView.loadUrl(FLASK_URL)
        }, 3000)
    }

    // ── Permissions ────────────────────────────────────────────────────

    private fun requestPermissionsIfNeeded() {
        val needed = requiredPermissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (needed.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, needed.toTypedArray(), PERMISSION_REQUEST_CODE)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            val denied = permissions.zip(grantResults.toTypedArray()).filter {
                it.second != PackageManager.PERMISSION_GRANTED
            }.map { it.first }

            if (denied.isNotEmpty()) {
                Log.w(TAG, "Permissions denied: $denied")
                // Camera and location are optional; app still works for WiFi + manual coords
                if (Manifest.permission.CAMERA in denied) {
                    Toast.makeText(this,
                        "Camera permission denied. Plate solving via camera won't work.",
                        Toast.LENGTH_LONG).show()
                }
                if (Manifest.permission.ACCESS_FINE_LOCATION in denied) {
                    Toast.makeText(this,
                        "Location denied. Enter site coordinates manually.",
                        Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    // ── Immersive mode ─────────────────────────────────────────────────

    private fun hideSystemUI() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.insetsController?.apply {
                hide(android.view.WindowInsets.Type.statusBars())
                systemBarsBehavior =
                    WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            }
        } else {
            @Suppress("DEPRECATION")
            window.decorView.systemUiVisibility = (
                View.SYSTEM_UI_FLAG_FULLSCREEN
                    or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                    or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
            )
        }
    }
}
