package com.telescopecontroller.camera

import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.SurfaceTexture
import android.graphics.YuvImage
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CaptureRequest
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import android.media.ImageReader
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.util.Log
import android.util.SizeF
import com.jiangdg.usb.USBMonitor
import com.jiangdg.uvc.IFrameCallback
import com.jiangdg.uvc.UVCCamera
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Unified camera manager for the telescope controller.
 *
 * Supports three camera sources, exposed to the Python backend through
 * Chaquopy's Java bridge:
 *
 *   1. PHONE CAMERA (Camera2 API)
 *      - Quick visual alignment, field identification
 *      - Used when no external camera is connected
 *
 *   2. USB UVC CAMERA (via libuvc native library)
 *      - Standard USB webcams connected via OTG
 *      - Same workflow as the desktop OpenCV capture
 *
 *   3. ZWO ASI CAMERA (via UVC protocol -- most planetary ASI cameras)
 *      - ASI120, ASI224, ASI290, ASI385, ASI462, etc.
 *      - Connected via USB OTG, detected by vendor ID 0x03C3
 *      - Uses the same UVC protocol as generic webcams
 *      - Note: Cooled deep-sky cameras (ASI294 Pro, ASI2600, etc.)
 *        do NOT support UVC and require the proprietary ZWO SDK
 *        (which has no Android version).
 *
 * The Python side calls:
 *   camera_manager.captureImage(save_path)   -> captures a single frame
 *   camera_manager.getJpegFrame()            -> returns JPEG bytes for MJPEG stream
 *   camera_manager.setExposureMs(ms)         -> sets exposure time
 *   camera_manager.setGain(value)            -> sets gain/ISO
 *   camera_manager.listCameras()             -> lists available cameras
 *   camera_manager.openUvcCamera(vendorId, productId) -> open specific USB camera
 *   camera_manager.openZwoCamera()           -> open first ZWO ASI camera found
 *   camera_manager.getActiveSourceName()     -> name of active camera
 */
class CameraManager(private val context: Context) {

    companion object {
        private const val TAG = "CameraManager"

        // ZWO vendor ID: 0x03C3 = 963 decimal
        const val ZWO_VENDOR_ID = 963

        // Default capture resolution for plate solving
        private const val DEFAULT_WIDTH = 1280
        private const val DEFAULT_HEIGHT = 960
    }

    init {
        // Initialize ZWO ASI USB manager if SDK is available
        // Must be called before any ASI camera operations
        if (ASICameraSDK.isNativeLoaded) {
            ASICameraSDK.initContext(context)
        }
    }

    enum class CameraSource {
        NONE,
        PHONE_CAMERA,
        USB_UVC,
        ZWO_ASI,         // ZWO via UVC protocol (legacy fallback if SDK fails)
        ZWO_ASI_SDK      // ZWO via native ASI SDK (preferred for ALL ASI cameras)
    }

    var activeSource: CameraSource = CameraSource.NONE
        private set

    // ── Phone camera (Camera2) ─────────────────────────────────────────
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null
    private var cameraThread: HandlerThread? = null
    private var cameraHandler: Handler? = null

    // ── UVC camera (libuvc via JNI) ────────────────────────────────────
    private var usbMonitor: USBMonitor? = null
    private var uvcCamera: UVCCamera? = null
    private var uvcDevice: UsbDevice? = null
    // Dummy SurfaceTexture for preview (UVCCamera requires a preview target)
    private var previewTexture: SurfaceTexture? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    // Latch for waiting on async UVC open
    @Volatile
    private var openLatch: CountDownLatch? = null
    @Volatile
    private var openSuccess = false

    // Target device for filtering in callbacks
    @Volatile
    private var targetVendorId: Int = 0
    @Volatile
    private var targetProductId: Int = 0

    // ── ASI SDK (native ZWO SDK) ──────────────────────────────────────
    private var asiSdk: ASICameraSDK? = null

    // ── Settings ───────────────────────────────────────────────────────
    // Use _exposure / _gain backing fields to avoid JVM signature clash
    // with setExposureMs() / setGain() methods called from Python
    private var _exposureMs: Long = 500
    private var _gain: Int = 100
    var binning: Int = 1

    // ── Latest captured frame (for MJPEG streaming) ────────────────────
    @Volatile
    private var latestJpeg: ByteArray? = null

    // ── ASI SDK video thread ──────────────────────────────────────────
    @Volatile
    private var asiVideoThread: Thread? = null
    @Volatile
    private var asiVideoRunning = false

    // ══════════════════════════════════════════════════════════════════
    //  Phone Camera (Camera2)
    // ══════════════════════════════════════════════════════════════════

    /**
     * Open the phone's rear camera for capturing plate-solve images.
     * Returns true on success.
     */
    fun openPhoneCamera(): Boolean {
        try {
            val manager = context.getSystemService(Context.CAMERA_SERVICE)
                as android.hardware.camera2.CameraManager

            // Find rear-facing camera
            val cameraId = manager.cameraIdList.firstOrNull { id ->
                val chars = manager.getCameraCharacteristics(id)
                chars.get(CameraCharacteristics.LENS_FACING) ==
                    CameraCharacteristics.LENS_FACING_BACK
            } ?: run {
                Log.e(TAG, "No rear camera found")
                return false
            }

            cameraThread = HandlerThread("CameraThread").apply { start() }
            cameraHandler = Handler(cameraThread!!.looper)

            // Set up ImageReader for JPEG capture
            val chars = manager.getCameraCharacteristics(cameraId)
            val sizes = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                ?.getOutputSizes(ImageFormat.JPEG)
            // Pick a resolution close to 1280x960 for plate solving
            val targetSize = sizes?.minByOrNull {
                Math.abs(it.width - DEFAULT_WIDTH) + Math.abs(it.height - DEFAULT_HEIGHT)
            } ?: sizes?.first()

            imageReader = ImageReader.newInstance(
                targetSize?.width ?: DEFAULT_WIDTH,
                targetSize?.height ?: DEFAULT_HEIGHT,
                ImageFormat.JPEG,
                2
            )

            imageReader?.setOnImageAvailableListener({ reader ->
                val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                try {
                    val buffer: ByteBuffer = image.planes[0].buffer
                    val bytes = ByteArray(buffer.remaining())
                    buffer.get(bytes)
                    latestJpeg = bytes
                } finally {
                    image.close()
                }
            }, cameraHandler)

            // Open the camera
            manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraDevice = camera
                    activeSource = CameraSource.PHONE_CAMERA
                    createCaptureSession()
                    Log.i(TAG, "Phone camera opened: $cameraId")
                }

                override fun onDisconnected(camera: CameraDevice) {
                    camera.close()
                    cameraDevice = null
                    activeSource = CameraSource.NONE
                    Log.w(TAG, "Phone camera disconnected")
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    camera.close()
                    cameraDevice = null
                    activeSource = CameraSource.NONE
                    Log.e(TAG, "Phone camera error: $error")
                }
            }, cameraHandler)

            return true
        } catch (e: SecurityException) {
            Log.e(TAG, "Camera permission denied", e)
            return false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to open phone camera", e)
            return false
        }
    }

    private fun createCaptureSession() {
        val camera = cameraDevice ?: return
        val reader = imageReader ?: return

        val stateCallback = object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(session: CameraCaptureSession) {
                captureSession = session
                val request = camera.createCaptureRequest(
                    CameraDevice.TEMPLATE_PREVIEW
                ).apply {
                    addTarget(reader.surface)
                    set(CaptureRequest.CONTROL_MODE,
                        CaptureRequest.CONTROL_MODE_AUTO)
                }.build()
                session.setRepeatingRequest(request, null, cameraHandler)
                Log.i(TAG, "Camera capture session configured")
            }

            override fun onConfigureFailed(session: CameraCaptureSession) {
                Log.e(TAG, "Capture session configuration failed")
            }
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            // API 28+: use non-deprecated SessionConfiguration API
            val outputConfig = android.hardware.camera2.params.OutputConfiguration(reader.surface)
            val sessionConfig = android.hardware.camera2.params.SessionConfiguration(
                android.hardware.camera2.params.SessionConfiguration.SESSION_REGULAR,
                listOf(outputConfig),
                context.mainExecutor,
                stateCallback
            )
            camera.createCaptureSession(sessionConfig)
        } else {
            // API 26-27: use legacy API
            @Suppress("DEPRECATION")
            camera.createCaptureSession(
                listOf(reader.surface),
                stateCallback,
                cameraHandler
            )
        }
    }

    // ══════════════════════════════════════════════════════════════════
    //  UVC Camera (libuvc via JNI) -- for USB webcams and ZWO ASI
    // ══════════════════════════════════════════════════════════════════

    /**
     * Open a UVC camera by USB vendor and product ID.
     *
     * This works for generic USB webcams and ZWO ASI planetary cameras
     * that expose a UVC interface (ASI120, ASI224, ASI290, etc.).
     *
     * Must be called from a background thread (blocks up to 15s waiting
     * for USB permission and camera open).
     *
     * @param vendorId  USB vendor ID (0 = first available UVC device)
     * @param productId USB product ID (0 = any product from vendor)
     * @return true if the camera was successfully opened
     */
    fun openUvcCamera(vendorId: Int = 0, productId: Int = 0): Boolean {
        try {
            val usbManager = context.getSystemService(Context.USB_SERVICE) as UsbManager
            val deviceList = usbManager.deviceList
            if (deviceList.isEmpty()) {
                Log.w(TAG, "No USB devices connected")
                return false
            }

            // Log all USB devices
            for ((name, dev) in deviceList) {
                Log.d(TAG, "USB device: $name vendor=0x${Integer.toHexString(dev.vendorId)} " +
                    "product=0x${Integer.toHexString(dev.productId)} " +
                    "class=${dev.deviceClass} ifaces=${dev.interfaceCount}")
            }

            // Find matching device
            val targetDevice = findUsbDevice(usbManager, vendorId, productId)
            if (targetDevice == null) {
                Log.w(TAG, "No matching UVC device found (vendor=$vendorId, product=$productId)")
                return false
            }

            Log.i(TAG, "Target UVC device: ${targetDevice.deviceName} " +
                "vendor=0x${Integer.toHexString(targetDevice.vendorId)} " +
                "product=0x${Integer.toHexString(targetDevice.productId)}")

            // Close any existing UVC camera
            closeUvcCamera()

            targetVendorId = targetDevice.vendorId
            targetProductId = targetDevice.productId
            uvcDevice = targetDevice

            activeSource = if (targetDevice.vendorId == ZWO_VENDOR_ID) {
                CameraSource.ZWO_ASI
            } else {
                CameraSource.USB_UVC
            }

            // Create latch for async open
            openLatch = CountDownLatch(1)
            openSuccess = false

            // USBMonitor must be created on main thread
            val clientCreatedLatch = CountDownLatch(1)
            mainHandler.post {
                try {
                    val monitor = USBMonitor(context, object : USBMonitor.OnDeviceConnectListener {
                        override fun onAttach(device: UsbDevice?) {
                            Log.d(TAG, "USB device attached: ${device?.deviceName}")
                        }

                        override fun onDetach(device: UsbDevice?) {
                            Log.d(TAG, "USB device detached: ${device?.deviceName}")
                        }

                        override fun onConnect(
                            device: UsbDevice?,
                            ctrlBlock: USBMonitor.UsbControlBlock?,
                            createNew: Boolean
                        ) {
                            if (device == null || ctrlBlock == null) {
                                Log.w(TAG, "onConnect: null device or ctrlBlock")
                                openLatch?.countDown()
                                return
                            }

                            // Filter for our target device
                            if (device.vendorId != targetVendorId) return
                            if (targetProductId != 0 && device.productId != targetProductId) return

                            Log.i(TAG, "USB permission granted, opening UVC camera: ${device.deviceName}")

                            try {
                                val camera = UVCCamera()
                                camera.open(ctrlBlock)
                                Log.i(TAG, "UVCCamera.open() succeeded")

                                // Set preview size -- try MJPEG first, fall back to YUYV
                                try {
                                    camera.setPreviewSize(
                                        DEFAULT_WIDTH, DEFAULT_HEIGHT,
                                        1, 30,  // min/max fps
                                        UVCCamera.FRAME_FORMAT_MJPEG,
                                        UVCCamera.DEFAULT_BANDWIDTH
                                    )
                                    Log.i(TAG, "Preview set: ${DEFAULT_WIDTH}x${DEFAULT_HEIGHT} MJPEG")
                                } catch (e: Exception) {
                                    Log.w(TAG, "MJPEG format failed, trying YUYV: ${e.message}")
                                    try {
                                        camera.setPreviewSize(
                                            640, 480,
                                            1, 30,
                                            UVCCamera.FRAME_FORMAT_YUYV,
                                            UVCCamera.DEFAULT_BANDWIDTH
                                        )
                                        Log.i(TAG, "Preview set: 640x480 YUYV")
                                    } catch (e2: Exception) {
                                        Log.e(TAG, "Failed to set any preview size: ${e2.message}")
                                        camera.close()
                                        camera.destroy()
                                        openLatch?.countDown()
                                        return
                                    }
                                }

                                // Set frame callback to receive NV21 data
                                camera.setFrameCallback(IFrameCallback { frame ->
                                    frame ?: return@IFrameCallback
                                    try {
                                        frame.position(0)
                                        val data = ByteArray(frame.capacity())
                                        frame.get(data)
                                        // Convert NV21 to JPEG
                                        val previewW = camera.previewSize?.width ?: DEFAULT_WIDTH
                                        val previewH = camera.previewSize?.height ?: DEFAULT_HEIGHT
                                        val jpeg = nv21ToJpeg(data, previewW, previewH)
                                        if (jpeg != null) {
                                            latestJpeg = jpeg
                                        }
                                    } catch (e: Exception) {
                                        // Frame conversion error -- skip frame
                                    }
                                }, UVCCamera.PIXEL_FORMAT_YUV420SP)

                                // Create a dummy SurfaceTexture for preview
                                // (UVCCamera requires a preview target even for headless capture)
                                val texture = SurfaceTexture(0)
                                camera.setPreviewTexture(texture)
                                camera.startPreview()

                                uvcCamera = camera
                                previewTexture = texture
                                openSuccess = true
                                openLatch?.countDown()

                                Log.i(TAG, "UVC camera streaming: ${device.deviceName}")

                            } catch (e: Exception) {
                                Log.e(TAG, "Failed to open UVC camera: ${e.message}", e)
                                openSuccess = false
                                openLatch?.countDown()
                            }
                        }

                        override fun onDisconnect(
                            device: UsbDevice?,
                            ctrlBlock: USBMonitor.UsbControlBlock?
                        ) {
                            Log.d(TAG, "USB device disconnected: ${device?.deviceName}")
                            if (device?.vendorId == targetVendorId) {
                                closeUvcCameraInternal()
                            }
                        }

                        override fun onCancel(device: UsbDevice?) {
                            Log.w(TAG, "USB permission denied for: ${device?.deviceName}")
                            openSuccess = false
                            openLatch?.countDown()
                        }
                    })

                    usbMonitor = monitor
                    monitor.register()
                    Log.i(TAG, "USBMonitor registered")

                    // Request permission for the target device
                    monitor.requestPermission(targetDevice)

                    clientCreatedLatch.countDown()
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to create USBMonitor: ${e.message}", e)
                    clientCreatedLatch.countDown()
                    openLatch?.countDown()
                }
            }

            // Wait for monitor creation
            clientCreatedLatch.await(5, TimeUnit.SECONDS)

            // Wait for camera open (async via USB permission + UVC init)
            val opened = openLatch?.await(15, TimeUnit.SECONDS) ?: false
            if (!opened) {
                Log.w(TAG, "UVC camera open timed out (15s)")
                activeSource = CameraSource.NONE
                return false
            }

            if (!openSuccess) {
                Log.w(TAG, "UVC camera open failed")
                activeSource = CameraSource.NONE
                return false
            }

            Log.i(TAG, "UVC camera ready: ${targetDevice.deviceName} (source=$activeSource)")
            return true

        } catch (e: Exception) {
            Log.e(TAG, "Failed to open UVC camera", e)
            activeSource = CameraSource.NONE
            return false
        }
    }

    /**
     * Open the first ZWO ASI camera found via UVC.
     * This is a legacy fallback -- prefer openZwoCameraSDK() for full control.
     */
    fun openZwoCamera(): Boolean {
        Log.i(TAG, "Looking for ZWO ASI camera via UVC (vendor 0x03C3)...")
        return openUvcCamera(vendorId = ZWO_VENDOR_ID)
    }

    // ══════════════════════════════════════════════════════════════════
    //  ZWO ASI Camera via Native SDK (full camera control)
    // ══════════════════════════════════════════════════════════════════

    /**
     * Check if the ASI SDK native libraries are available.
     * Returns false if the user hasn't placed libASICamera2.so in jniLibs.
     */
    fun isAsiSdkAvailable(): Boolean = ASICameraSDK.isNativeLoaded

    /**
     * Open a ZWO ASI camera using the official ZWO Android SDK.
     *
     * The SDK handles USB enumeration and permissions internally
     * via ASIUSBManager -- no need to pass UsbDevice or file descriptors.
     *
     * Benefits over UVC mode:
     *   - Full camera control (exposure, gain, ROI, binning, flip, etc.)
     *   - RAW image data (not compressed MJPEG)
     *   - Full exposure range (32us to 2000s)
     *   - Hardware binning
     *   - Cooler control (cooled cameras)
     *   - Temperature readout
     *   - Works with ALL ASI cameras (USB 2.0 and 3.0, planetary and deep-sky)
     *
     * Must be called from a background thread (blocks during USB init).
     *
     * @return true if a ZWO camera was opened via the SDK
     */
    fun openZwoCameraSDK(): Boolean {
        if (!ASICameraSDK.isNativeLoaded) {
            Log.w(TAG, "ASI SDK not available -- falling back to UVC")
            return false
        }

        // ── Pre-grant USB permission for the ZWO device ──────────────
        // The ZWO SDK's ASIUSBManager.getUsbFD() internally calls
        // PendingIntent.getBroadcast(ctx, 0, intent, 0) with flags=0.
        // On Android 12+ (API 31+) this crashes with:
        //   IllegalArgumentException: Targeting S+ requires
        //   FLAG_IMMUTABLE or FLAG_MUTABLE
        // The resulting pending JNI exception causes abort() when the
        // native code calls FindClass().
        //
        // Fix: request USB permission ourselves with correct flags
        // BEFORE the SDK tries to. Once granted, the SDK's internal
        // requestPermission path is skipped entirely.
        val usbManager = context.getSystemService(Context.USB_SERVICE) as UsbManager
        val zwoDevice = usbManager.deviceList.values
            .firstOrNull { it.vendorId == ZWO_VENDOR_ID }

        if (zwoDevice == null) {
            Log.w(TAG, "No ZWO USB device found (vendor 0x03C3)")
            return false
        }

        if (!usbManager.hasPermission(zwoDevice)) {
            Log.i(TAG, "Requesting USB permission for ZWO device: ${zwoDevice.productName}")

            val permGranted = AtomicBoolean(false)
            val permLatch = CountDownLatch(1)
            val ACTION_USB_PERM = "com.telescopecontroller.ZWO_USB_PERMISSION"

            val receiver = object : BroadcastReceiver() {
                override fun onReceive(ctx: Context, intent: Intent) {
                    if (ACTION_USB_PERM == intent.action) {
                        val permDevice = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                            intent.getParcelableExtra(UsbManager.EXTRA_DEVICE, UsbDevice::class.java)
                        } else {
                            @Suppress("DEPRECATION")
                            intent.getParcelableExtra(UsbManager.EXTRA_DEVICE)
                        }
                        if (permDevice?.deviceId == zwoDevice.deviceId) {
                            permGranted.set(
                                intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false)
                            )
                            Log.i(TAG, "ZWO USB permission ${if (permGranted.get()) "GRANTED" else "DENIED"}")
                            permLatch.countDown()
                        }
                    }
                }
            }

            val filter = IntentFilter(ACTION_USB_PERM)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                context.registerReceiver(receiver, filter, Context.RECEIVER_NOT_EXPORTED)
            } else {
                context.registerReceiver(receiver, filter)
            }

            // FLAG_MUTABLE is required: the system fills EXTRA_DEVICE
            // and EXTRA_PERMISSION_GRANTED into the intent
            val pendingIntent = PendingIntent.getBroadcast(
                context, 0,
                Intent(ACTION_USB_PERM),
                PendingIntent.FLAG_MUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
            )
            usbManager.requestPermission(zwoDevice, pendingIntent)

            // Block until user taps Allow/Deny (or timeout)
            val answered = permLatch.await(30, TimeUnit.SECONDS)

            try { context.unregisterReceiver(receiver) } catch (_: Exception) {}

            if (!answered) {
                Log.w(TAG, "ZWO USB permission request timed out (30s)")
                return false
            }
            if (!permGranted.get()) {
                Log.w(TAG, "ZWO USB permission DENIED by user")
                return false
            }
        } else {
            Log.i(TAG, "ZWO USB permission already granted for: ${zwoDevice.productName}")
        }

        // ── Open camera on fresh thread (JNI safety) ─────────────────
        // The ZWO SDK's native JNI code (libzwo_camera.so) calls FindClass()
        // without clearing pending JNI exceptions. When called from a
        // Chaquopy/Python thread, the cross-language bridge may leave a
        // pending exception that causes FindClass to trigger
        // AssertNoPendingException -> abort(). Running on a fresh thread
        // guarantees a clean JNI environment.
        val resultLatch = CountDownLatch(1)
        val successHolder = booleanArrayOf(false)

        val initThread = Thread({
            try {
                // Close any existing camera
                closeAsiSdk()
                closeUvcCamera()

                // Create SDK instance -- it handles USB internally
                val sdk = ASICameraSDK(context)
                val opened = sdk.openCamera()

                if (opened) {
                    asiSdk = sdk
                    activeSource = CameraSource.ZWO_ASI_SDK

                    // Also find the UsbDevice for name display
                    val usbMgr = context.getSystemService(Context.USB_SERVICE) as UsbManager
                    uvcDevice = usbMgr.deviceList.values
                        .firstOrNull { it.vendorId == ZWO_VENDOR_ID }

                    Log.i(TAG, "ASI SDK camera opened: ${sdk.cameraName} " +
                        "(${sdk.maxWidth}x${sdk.maxHeight})")

                    // Set default exposure and gain
                    sdk.setExposure(_exposureMs * 1000)  // ms -> us
                    sdk.setGain(_gain)

                    // Start video capture and frame conversion thread
                    sdk.startCapture()
                    startAsiVideoThread(sdk)

                    successHolder[0] = true
                } else {
                    Log.e(TAG, "ASI SDK openCamera failed")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to open ZWO camera via SDK: ${e.message}", e)
            }
            resultLatch.countDown()
        }, "ASI-SDK-Init")

        initThread.start()

        return try {
            val completed = resultLatch.await(15, TimeUnit.SECONDS)
            if (!completed) {
                Log.w(TAG, "ASI SDK init timed out (15s)")
            }
            successHolder[0]
        } catch (e: InterruptedException) {
            Log.e(TAG, "ASI SDK init interrupted")
            false
        }
    }

    /**
     * Background thread that reads raw frames from the ASI SDK and
     * converts them to JPEG for the latestJpeg buffer (used by
     * MJPEG streaming and plate solving).
     */
    private fun startAsiVideoThread(sdk: ASICameraSDK) {
        asiVideoRunning = true
        // The ROI is set to full resolution in openCamera(), so use maxWidth/maxHeight
        val w = sdk.maxWidth
        val h = sdk.maxHeight
        asiVideoThread = Thread({
            Log.i(TAG, "ASI video thread started (${w}x${h})")

            while (asiVideoRunning && sdk.isOpen) {
                try {
                    val rawData = sdk.getFrame(2000) ?: continue

                    // Convert RAW8 grayscale to JPEG
                    val jpeg = raw8ToJpeg(rawData, w, h)
                    if (jpeg != null) {
                        latestJpeg = jpeg
                    }
                } catch (e: Exception) {
                    if (asiVideoRunning) {
                        Log.w(TAG, "ASI video frame error: ${e.message}")
                    }
                }
            }
            Log.i(TAG, "ASI video thread stopped")
        }, "ASI-Video").apply {
            isDaemon = true
            start()
        }
    }

    private fun stopAsiVideoThread() {
        asiVideoRunning = false
        asiVideoThread?.let { thread ->
            try {
                thread.join(3000)
            } catch (e: InterruptedException) {
                // ignore
            }
        }
        asiVideoThread = null
    }

    /**
     * Convert RAW8 (grayscale) pixel data to JPEG.
     * Creates a Bitmap, fills it with gray pixels, and compresses to JPEG.
     */
    private fun raw8ToJpeg(raw: ByteArray, width: Int, height: Int): ByteArray? {
        return try {
            val expectedSize = width * height
            if (raw.size < expectedSize) {
                Log.w(TAG, "RAW8 buffer too small: ${raw.size} < $expectedSize")
                return null
            }

            // Convert grayscale to ARGB_8888
            val pixels = IntArray(expectedSize)
            for (i in 0 until expectedSize) {
                val gray = raw[i].toInt() and 0xFF
                pixels[i] = (0xFF shl 24) or (gray shl 16) or (gray shl 8) or gray
            }

            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            bitmap.setPixels(pixels, 0, width, 0, 0, width, height)

            val out = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.JPEG, 85, out)
            bitmap.recycle()

            out.toByteArray()
        } catch (e: Exception) {
            Log.e(TAG, "RAW8 to JPEG conversion failed: ${e.message}")
            null
        }
    }

    private fun closeAsiSdk() {
        stopAsiVideoThread()
        try {
            asiSdk?.close()
        } catch (e: Exception) {
            Log.w(TAG, "Error closing ASI SDK: ${e.message}")
        }
        asiSdk = null
    }

    // ── ASI SDK-specific getters (called from Python via camera_bridge) ─

    /**
     * Get the ASI SDK instance (or null if not using SDK mode).
     * Used by camera_bridge.py for advanced controls.
     */
    fun getAsiSdk(): ASICameraSDK? = asiSdk

    /**
     * Get sensor temperature from ASI SDK (degrees C).
     * Returns -999 if not available.
     */
    fun getAsiTemperature(): Float {
        return asiSdk?.getTemperature() ?: -999f
    }

    /**
     * Set the ASI SDK ROI (Region of Interest).
     */
    fun setAsiROI(width: Int, height: Int, bin: Int, imgType: Int): Boolean {
        val sdk = asiSdk ?: return false
        // Need to stop capture, change ROI, restart
        stopAsiVideoThread()
        sdk.stopCapture()
        val ok = sdk.setROI(width, height, bin, imgType)
        if (ok) {
            binning = bin
            sdk.startCapture()
            startAsiVideoThread(sdk)
        }
        return ok
    }

    /**
     * Set ASI SDK cooler target temperature.
     */
    fun setAsiCoolerTarget(tempC: Int): Boolean {
        return asiSdk?.setCoolerTarget(tempC) ?: false
    }

    private fun findUsbDevice(
        usbManager: UsbManager,
        vendorId: Int,
        productId: Int
    ): UsbDevice? {
        val deviceList = usbManager.deviceList

        if (vendorId != 0) {
            return deviceList.values.firstOrNull { dev ->
                dev.vendorId == vendorId &&
                    (productId == 0 || dev.productId == productId)
            }
        }

        // No vendor specified -- find first UVC-class device
        return deviceList.values.firstOrNull { dev ->
            isUvcDevice(dev)
        }
    }

    private fun isUvcDevice(device: UsbDevice): Boolean {
        if (device.deviceClass == 14) return true
        for (i in 0 until device.interfaceCount) {
            if (device.getInterface(i).interfaceClass == 14) return true
        }
        return false
    }

    private fun closeUvcCameraInternal() {
        try {
            uvcCamera?.stopPreview()
            uvcCamera?.close()
            uvcCamera?.destroy()
        } catch (e: Exception) {
            Log.w(TAG, "Error closing UVC camera: ${e.message}")
        }
        uvcCamera = null

        try {
            previewTexture?.release()
        } catch (e: Exception) {
            // ignore
        }
        previewTexture = null
    }

    private fun closeUvcCamera() {
        closeUvcCameraInternal()

        try {
            usbMonitor?.unregister()
            usbMonitor?.destroy()
        } catch (e: Exception) {
            Log.w(TAG, "Error destroying USBMonitor: ${e.message}")
        }
        usbMonitor = null
        uvcDevice = null
    }

    // ══════════════════════════════════════════════════════════════════
    //  Frame format conversion
    // ══════════════════════════════════════════════════════════════════

    private fun nv21ToJpeg(nv21: ByteArray, width: Int, height: Int): ByteArray? {
        return try {
            val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, width, height), 85, out)
            out.toByteArray()
        } catch (e: Exception) {
            null
        }
    }

    // ══════════════════════════════════════════════════════════════════
    //  Capture interface (called from Python via Chaquopy)
    // ══════════════════════════════════════════════════════════════════

    fun captureImage(savePath: String): String? {
        return when (activeSource) {
            CameraSource.PHONE_CAMERA -> captureFromLatestJpeg(savePath, "phone")
            CameraSource.USB_UVC -> captureFromLatestJpeg(savePath, "UVC")
            CameraSource.ZWO_ASI -> captureFromLatestJpeg(savePath, "ZWO-UVC")
            CameraSource.ZWO_ASI_SDK -> captureFromLatestJpeg(savePath, "ZWO-SDK")
            CameraSource.NONE -> {
                Log.w(TAG, "No camera active")
                null
            }
        }
    }

    fun getJpegFrame(): ByteArray? = latestJpeg

    fun listCameras(): List<Map<String, String>> {
        val cameras = mutableListOf<Map<String, String>>()

        // Phone cameras
        try {
            val manager = context.getSystemService(Context.CAMERA_SERVICE)
                as android.hardware.camera2.CameraManager
            for (id in manager.cameraIdList) {
                val chars = manager.getCameraCharacteristics(id)
                val facing = chars.get(CameraCharacteristics.LENS_FACING)
                val name = when (facing) {
                    CameraCharacteristics.LENS_FACING_BACK -> "Rear Camera"
                    CameraCharacteristics.LENS_FACING_FRONT -> "Front Camera"
                    else -> "Camera $id"
                }
                cameras.add(mapOf("id" to id, "name" to name, "source" to "phone"))
            }
        } catch (e: Exception) {
            Log.w(TAG, "Error listing phone cameras: ${e.message}")
        }

        // USB cameras
        try {
            val usbManager = context.getSystemService(Context.USB_SERVICE) as UsbManager
            val sdkAvailable = ASICameraSDK.isNativeLoaded
            for ((_, device) in usbManager.deviceList) {
                if (isUvcDevice(device) || device.vendorId == ZWO_VENDOR_ID) {
                    val isZwo = device.vendorId == ZWO_VENDOR_ID
                    val source = if (isZwo) {
                        if (sdkAvailable) "zwo_sdk" else "zwo"
                    } else {
                        "uvc"
                    }
                    val name = if (isZwo) {
                        val mode = if (sdkAvailable) "SDK" else "UVC"
                        "ZWO ASI ($mode, 0x${Integer.toHexString(device.productId)})"
                    } else {
                        device.productName ?: "USB Camera (0x${Integer.toHexString(device.vendorId)}:" +
                            "0x${Integer.toHexString(device.productId)})"
                    }
                    cameras.add(mapOf(
                        "id" to "${device.vendorId}:${device.productId}",
                        "name" to name,
                        "source" to source,
                        "vendor_id" to device.vendorId.toString(),
                        "product_id" to device.productId.toString(),
                        "sdk_available" to sdkAvailable.toString()
                    ))
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Error listing USB cameras: ${e.message}")
        }

        return cameras
    }

    fun setExposureMs(ms: Long) {
        _exposureMs = ms
        // Update ASI SDK exposure if active (SDK uses microseconds)
        if (activeSource == CameraSource.ZWO_ASI_SDK) {
            asiSdk?.setExposure(ms * 1000)
        }
        Log.d(TAG, "Exposure set to ${ms}ms")
    }

    fun setGain(value: Int) {
        _gain = value
        try {
            uvcCamera?.gain = value
        } catch (e: Exception) {
            Log.w(TAG, "Could not set UVC gain: ${e.message}")
        }
        // Update ASI SDK gain if active
        if (activeSource == CameraSource.ZWO_ASI_SDK) {
            asiSdk?.setGain(value)
        }
        Log.d(TAG, "Gain set to $value")
    }

    fun getActiveSourceName(): String {
        return when (activeSource) {
            CameraSource.NONE -> "None"
            CameraSource.PHONE_CAMERA -> "Phone Camera"
            CameraSource.USB_UVC -> {
                uvcDevice?.productName ?: "USB Camera"
            }
            CameraSource.ZWO_ASI -> {
                val pid = uvcDevice?.productId?.let { "0x${Integer.toHexString(it)}" } ?: "?"
                "ZWO ASI UVC ($pid)"
            }
            CameraSource.ZWO_ASI_SDK -> {
                val name = asiSdk?.cameraName ?: "ZWO ASI"
                "$name (SDK)"
            }
        }
    }

    /**
     * Return the phone's rear camera physical sensor dimensions in mm.
     * Uses Camera2 SENSOR_INFO_PHYSICAL_SIZE.
     * Returns a Map with "width_mm", "height_mm", "pixel_size_um",
     * "resolution_x", "resolution_y", or empty map if unavailable.
     */
    fun getPhoneSensorInfo(): Map<String, Double> {
        try {
            val manager = context.getSystemService(Context.CAMERA_SERVICE)
                as android.hardware.camera2.CameraManager
            val cameraId = manager.cameraIdList.firstOrNull { id ->
                val chars = manager.getCameraCharacteristics(id)
                chars.get(CameraCharacteristics.LENS_FACING) ==
                    CameraCharacteristics.LENS_FACING_BACK
            } ?: return emptyMap()

            val chars = manager.getCameraCharacteristics(cameraId)

            // Physical sensor size (mm)
            val physSize: SizeF = chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                ?: return emptyMap()

            // Active pixel array size (full resolution)
            val pixelArray: android.graphics.Rect? =
                chars.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)

            val widthMm = physSize.width.toDouble()
            val heightMm = physSize.height.toDouble()
            val resX = pixelArray?.width()?.toDouble() ?: 0.0
            val resY = pixelArray?.height()?.toDouble() ?: 0.0

            // Derive pixel size in micrometers
            val pixelSizeUm = if (resX > 0) (widthMm / resX) * 1000.0 else 0.0

            Log.i(TAG, "Phone sensor: ${widthMm}x${heightMm}mm, " +
                "${resX.toInt()}x${resY.toInt()}px, pixel=${String.format("%.3f", pixelSizeUm)}um")

            return mapOf(
                "width_mm" to widthMm,
                "height_mm" to heightMm,
                "pixel_size_um" to pixelSizeUm,
                "resolution_x" to resX,
                "resolution_y" to resY
            )
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get phone sensor info: ${e.message}")
            return emptyMap()
        }
    }

    // ══════════════════════════════════════════════════════════════════
    //  Source-specific capture
    // ══════════════════════════════════════════════════════════════════

    private fun captureFromLatestJpeg(savePath: String, sourceName: String): String? {
        val jpeg = latestJpeg ?: run {
            Log.w(TAG, "No frame available from $sourceName camera")
            return null
        }

        return try {
            val bitmap = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size) ?: run {
                Log.e(TAG, "Failed to decode JPEG from $sourceName")
                return null
            }

            // Center crop to 60% (matches desktop auto_platesolve.py)
            val cropRatio = 0.6
            val cropW = (bitmap.width * cropRatio).toInt()
            val cropH = (bitmap.height * cropRatio).toInt()
            val x = (bitmap.width - cropW) / 2
            val y = (bitmap.height - cropH) / 2
            val cropped = Bitmap.createBitmap(bitmap, x, y, cropW, cropH)

            FileOutputStream(File(savePath)).use { out ->
                cropped.compress(Bitmap.CompressFormat.PNG, 100, out)
            }

            cropped.recycle()
            bitmap.recycle()

            Log.d(TAG, "Captured $sourceName frame -> $savePath")
            savePath
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save $sourceName capture: ${e.message}")
            null
        }
    }

    // ══════════════════════════════════════════════════════════════════
    //  Cleanup
    // ══════════════════════════════════════════════════════════════════

    fun close() {
        // Close phone camera
        captureSession?.close()
        cameraDevice?.close()
        imageReader?.close()
        cameraThread?.quitSafely()
        captureSession = null
        cameraDevice = null
        imageReader = null

        // Close UVC camera
        closeUvcCamera()

        // Close ASI SDK camera
        closeAsiSdk()

        activeSource = CameraSource.NONE
        latestJpeg = null
        Log.i(TAG, "Camera manager closed")
    }
}
