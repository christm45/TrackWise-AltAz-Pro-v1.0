package com.telescopecontroller.camera

import android.content.Context
import android.util.Log
import com.zwo.ASIConstants
import com.zwo.ASICameraProperty
import com.zwo.ASIImageBuffer
import com.zwo.ASIUSBManager
import com.zwo.ZwoCamera
import java.nio.ByteBuffer

/**
 * Kotlin wrapper around the official ZWO ASI Camera Android SDK (zwocamera.jar).
 *
 * The SDK provides:
 *   - ASIUSBManager: static USB enumeration and context init
 *   - ZwoCamera: per-camera instance with full control
 *   - Pre-built native libs: libASICamera2.so + libzwo_camera.so
 *
 * Thread safety: All SDK calls are synchronized on [lock].
 */
class ASICameraSDK(private val context: Context) {

    companion object {
        private const val TAG = "ASICameraSDK"

        /** ASI Image type constants */
        const val IMG_RAW8  = 0
        const val IMG_RGB24 = 1
        const val IMG_RAW16 = 2
        const val IMG_Y8    = 3

        /** ASI Exposure status constants */
        const val EXP_IDLE    = 0
        const val EXP_WORKING = 1
        const val EXP_SUCCESS = 2
        const val EXP_FAILED  = 3

        /** ASI Control type constants */
        const val CTRL_GAIN         = 0   // ASI_GAIN
        const val CTRL_EXPOSURE     = 1   // ASI_EXPOSURE (microseconds)
        const val CTRL_GAMMA        = 2
        const val CTRL_WB_R         = 3
        const val CTRL_WB_B         = 4
        const val CTRL_OFFSET       = 5   // ASI_BRIGHTNESS/OFFSET
        const val CTRL_BANDWIDTH    = 6
        const val CTRL_TEMPERATURE  = 8   // returns 10x degrees C
        const val CTRL_FLIP         = 9
        const val CTRL_HIGH_SPEED   = 14
        const val CTRL_COOLER_POWER = 15
        const val CTRL_TARGET_TEMP  = 16
        const val CTRL_COOLER_ON    = 17

        /** ASI_BOOL int constants */
        private const val ASI_FALSE = 0
        private const val ASI_TRUE  = 1

        /** ASI_SUCCESS error code value */
        private const val SUCCESS = ASIConstants.ASI_ERROR_CODE.ASI_SUCCESS  // 0

        /**
         * Whether the native libraries loaded successfully.
         */
        @Volatile
        var isNativeLoaded = false
            private set

        @Volatile
        private var isContextInitialized = false

        init {
            try {
                System.loadLibrary("ASICamera2")
                System.loadLibrary("zwo_camera")
                isNativeLoaded = true
                Log.i(TAG, "ZWO ASI SDK native libraries loaded")
            } catch (e: UnsatisfiedLinkError) {
                isNativeLoaded = false
                Log.w(TAG, "ZWO ASI SDK native libs not available: ${e.message}")
            }
        }

        /**
         * Initialize the USB manager with Android context.
         * Must be called before any camera operations.
         */
        fun initContext(context: Context) {
            if (!isNativeLoaded) return
            if (isContextInitialized) return
            try {
                ASIUSBManager.initContext(context.applicationContext)
                isContextInitialized = true
                Log.i(TAG, "ASIUSBManager context initialized")
            } catch (e: Exception) {
                Log.e(TAG, "ASIUSBManager.initContext failed: ${e.message}", e)
            }
        }

        /** Helper: check if an ASI_ERROR_CODE represents success */
        private fun isSuccess(code: ASIConstants.ASI_ERROR_CODE?): Boolean {
            return code != null && code.intVal == SUCCESS
        }
    }

    // ── Internal state ────────────────────────────────────────────────
    private var camera: ZwoCamera? = null
    private var cameraProperty: ASICameraProperty? = null
    private val lock = Object()

    var cameraName: String = "Not connected"
        private set
    var maxWidth: Int = 0
        private set
    var maxHeight: Int = 0
        private set
    var pixelSizeUm: Double = 0.0   // Pixel size in micrometers (from SDK)
        private set
    var isColor: Boolean = false
        private set
    var isCooled: Boolean = false
        private set
    var isOpen: Boolean = false
        private set
    var isCapturing: Boolean = false
        private set

    /** Sensor physical width in mm = maxWidth * pixelSize_um / 1000 */
    val sensorWidthMm: Double
        get() = if (maxWidth > 0 && pixelSizeUm > 0) maxWidth * pixelSizeUm / 1000.0 else 0.0

    /** Sensor physical height in mm = maxHeight * pixelSize_um / 1000 */
    val sensorHeightMm: Double
        get() = if (maxHeight > 0 && pixelSizeUm > 0) maxHeight * pixelSizeUm / 1000.0 else 0.0

    // ── Frame buffer ──────────────────────────────────────────────────
    private var frameBuffer: ASIImageBuffer? = null
    private var frameBufferSize: Int = 0

    // ══════════════════════════════════════════════════════════════════
    //  Public API
    // ══════════════════════════════════════════════════════════════════

    fun isAvailable(): Boolean = isNativeLoaded

    /**
     * Open the first connected ASI camera.
     * The SDK handles USB enumeration internally.
     */
    fun openCamera(): Boolean {
        if (!isNativeLoaded) {
            Log.e(TAG, "Cannot open: native libs not loaded")
            return false
        }

        initContext(context)

        synchronized(lock) {
            close()

            try {
                val numCameras = ZwoCamera.getNumOfConnectedCameras()
                Log.i(TAG, "Connected ASI cameras: $numCameras")

                if (numCameras <= 0) {
                    Log.w(TAG, "No ASI cameras found")
                    return false
                }

                // Get first camera properties
                val propRet = ZwoCamera.getCameraProperty(0)
                if (propRet == null) {
                    Log.e(TAG, "getCameraProperty returned null")
                    return false
                }
                if (!isSuccess(propRet.errorCode)) {
                    Log.e(TAG, "getCameraProperty failed: ${propRet.errorCode?.intVal}")
                    return false
                }

                val prop = propRet.obj as? ASICameraProperty
                if (prop == null) {
                    Log.e(TAG, "getCameraProperty returned null property object")
                    return false
                }

                cameraProperty = prop
                cameraName = prop.name ?: "ZWO ASI"
                maxWidth = prop.maxWidth.toInt()
                maxHeight = prop.maxHeight.toInt()
                pixelSizeUm = prop.pixelSize   // micrometers
                isColor = prop.isColorCam != 0
                isCooled = prop.isCoolerCam != 0

                Log.i(TAG, "Camera: $cameraName (ID=${prop.cameraID}, " +
                    "${maxWidth}x${maxHeight}, pixel=${pixelSizeUm}um, " +
                    "sensor=${String.format("%.2f", sensorWidthMm)}x${String.format("%.2f", sensorHeightMm)}mm, " +
                    "color=$isColor, cooled=$isCooled)")

                // Create and open camera
                val cam = ZwoCamera(prop.cameraID)

                var rc = cam.openCamera()
                if (!isSuccess(rc)) {
                    Log.e(TAG, "openCamera failed: ${rc?.intVal}")
                    return false
                }

                rc = cam.initCamera()
                if (!isSuccess(rc)) {
                    Log.e(TAG, "initCamera failed: ${rc?.intVal}")
                    cam.closeCamera()
                    return false
                }

                camera = cam
                isOpen = true

                // Set default ROI: full resolution, bin1, RAW8
                setROI(maxWidth, maxHeight, 1, IMG_RAW8)

                Log.i(TAG, "ASI camera ready: $cameraName")
                return true

            } catch (e: Exception) {
                Log.e(TAG, "openCamera exception: ${e.message}", e)
                return false
            }
        }
    }

    /**
     * Set capture ROI. Width must be multiple of 8, height multiple of 2.
     */
    fun setROI(width: Int, height: Int, bin: Int = 1, imgType: Int = IMG_RAW8): Boolean {
        synchronized(lock) {
            val cam = camera ?: return false
            val w = (width / 8) * 8
            val h = (height / 2) * 2

            val rc = cam.setRoiFormat(w, h, bin, imgType)
            if (!isSuccess(rc)) {
                Log.e(TAG, "setRoiFormat($w, $h, bin$bin, fmt$imgType) failed: ${rc?.intVal}")
                return false
            }

            // Allocate frame buffer
            val pw = w / bin
            val ph = h / bin
            val bufSize = when (imgType) {
                IMG_RAW16 -> pw * ph * 2
                IMG_RGB24 -> pw * ph * 3
                else -> pw * ph  // RAW8, Y8
            }
            frameBuffer = ASIImageBuffer.allocate(bufSize)
            frameBufferSize = bufSize

            Log.i(TAG, "ROI set: ${w}x${h} bin$bin fmt$imgType (buf=$bufSize)")
            return true
        }
    }

    fun setExposure(microseconds: Long): Boolean {
        synchronized(lock) {
            val cam = camera ?: return false
            val rc = cam.setControlValue(CTRL_EXPOSURE, microseconds, ASI_FALSE)
            return rc == SUCCESS
        }
    }

    fun setGain(gain: Int): Boolean {
        synchronized(lock) {
            val cam = camera ?: return false
            val rc = cam.setControlValue(CTRL_GAIN, gain.toLong(), ASI_FALSE)
            return rc == SUCCESS
        }
    }

    fun getTemperature(): Float {
        synchronized(lock) {
            val cam = camera ?: return -999f
            val ret = cam.getControlValue(CTRL_TEMPERATURE) ?: return -999f
            if (!isSuccess(ret.errorCode)) return -999f
            return ret.extraLongVal1.toFloat() / 10f
        }
    }

    fun setCoolerTarget(tempC: Int): Boolean {
        synchronized(lock) {
            val cam = camera ?: return false
            if (!isCooled) return false
            cam.setControlValue(CTRL_COOLER_ON, 1L, ASI_FALSE)
            val rc = cam.setControlValue(CTRL_TARGET_TEMP, tempC.toLong(), ASI_FALSE)
            return rc == SUCCESS
        }
    }

    fun setControlValue(controlType: Int, value: Long, isAuto: Boolean = false): Boolean {
        synchronized(lock) {
            val cam = camera ?: return false
            val rc = cam.setControlValue(controlType, value, if (isAuto) ASI_TRUE else ASI_FALSE)
            return rc == SUCCESS
        }
    }

    fun getControlValue(controlType: Int): Long {
        synchronized(lock) {
            val cam = camera ?: return -1
            val ret = cam.getControlValue(controlType) ?: return -1
            return if (isSuccess(ret.errorCode)) ret.extraLongVal1 else -1
        }
    }

    // ── Video capture ─────────────────────────────────────────────────

    fun startCapture(): Boolean {
        synchronized(lock) {
            val cam = camera ?: return false
            if (isCapturing) cam.stopVideoCapture()
            val rc = cam.startVideoCapture()
            if (!isSuccess(rc)) {
                Log.e(TAG, "startVideoCapture failed: ${rc?.intVal}")
                return false
            }
            isCapturing = true
            Log.i(TAG, "Video capture started")
            return true
        }
    }

    /**
     * Get one frame. getVideoData(buffer, bufferSize, waitMs)
     */
    fun getFrame(timeoutMs: Int = 2000): ByteArray? {
        synchronized(lock) {
            val cam = camera ?: return null
            val buf = frameBuffer ?: return null
            if (!isCapturing) return null

            val rc = cam.getVideoData(buf, frameBufferSize, timeoutMs)
            if (!isSuccess(rc)) {
                if (rc?.intVal != ASIConstants.ASI_ERROR_CODE.ASI_ERROR_TIMEOUT) {
                    Log.w(TAG, "getVideoData failed: ${rc?.intVal}")
                }
                return null
            }

            val byteBuffer: ByteBuffer = buf.getmByteBuffer() ?: return null
            byteBuffer.position(0)
            val data = ByteArray(frameBufferSize)
            byteBuffer.get(data, 0, frameBufferSize)
            return data
        }
    }

    fun stopCapture() {
        synchronized(lock) {
            val cam = camera ?: return
            if (isCapturing) {
                cam.stopVideoCapture()
                isCapturing = false
                Log.i(TAG, "Video capture stopped")
            }
        }
    }

    // ── Single exposure ───────────────────────────────────────────────

    fun startExposure(isDark: Boolean = false): Boolean {
        synchronized(lock) {
            val cam = camera ?: return false
            val rc = cam.startExposure(isDark)
            return isSuccess(rc)
        }
    }

    fun getExposureStatus(): Int {
        synchronized(lock) {
            val cam = camera ?: return EXP_IDLE
            val ret = cam.expStatus ?: return EXP_IDLE
            return if (isSuccess(ret.errorCode)) {
                ret.extraLongVal1.toInt()
            } else EXP_IDLE
        }
    }

    fun getExposureData(): ByteArray? {
        synchronized(lock) {
            val cam = camera ?: return null
            val buf = frameBuffer ?: return null
            val rc = cam.getDataAfterExp(buf, frameBufferSize)
            if (!isSuccess(rc)) {
                Log.e(TAG, "getDataAfterExp failed: ${rc?.intVal}")
                return null
            }
            val byteBuffer: ByteBuffer = buf.getmByteBuffer() ?: return null
            byteBuffer.position(0)
            val data = ByteArray(frameBufferSize)
            byteBuffer.get(data, 0, frameBufferSize)
            return data
        }
    }

    // ── Utility ───────────────────────────────────────────────────────

    fun getSDKVersion(): String {
        return if (isNativeLoaded) "Android SDK v1.1" else "not loaded"
    }

    fun close() {
        synchronized(lock) {
            if (isCapturing) {
                camera?.stopVideoCapture()
                isCapturing = false
            }
            camera?.closeCamera()
            camera = null
            cameraProperty = null
            isOpen = false
            cameraName = "Not connected"
            maxWidth = 0
            maxHeight = 0
            frameBuffer = null
            Log.i(TAG, "ASI camera closed")
        }
    }
}
