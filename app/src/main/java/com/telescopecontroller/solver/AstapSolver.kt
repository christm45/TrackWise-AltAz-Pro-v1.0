package com.telescopecontroller.solver

import android.content.Context
import android.os.Build
import android.util.Log
import java.io.*
import java.util.concurrent.TimeUnit
import java.util.regex.Pattern

/**
 * Kotlin wrapper around the ASTAP (Astrometric STAcking Program) command-line
 * plate solver.
 *
 * ## Architecture Overview
 *
 * The ASTAP CLI binary (`astap_cli`) is a standalone ELF executable compiled
 * for Android ABIs (arm64-v8a, armeabi-v7a, x86_64).  It is bundled as
 * `libastapcli.so` inside `jniLibs/<abi>/` — the "lib" prefix + ".so" suffix
 * trick makes Android's PackageManager extract it to `nativeLibraryDir` with
 * execute permissions at install time.
 *
 * Source binaries: https://sourceforge.net/projects/astap-program/files/android/
 * Each binary is ~300-1000 KB depending on the ABI.
 *
 * ## Invocation Details
 *
 * On real Android devices, native libraries are marked as `ET_DYN` (shared
 * objects) even though `astap_cli` is a PIE executable.  The system linker
 * (`/system/bin/linker64` for 64-bit, `/system/bin/linker` for 32-bit) must
 * be used to execute them.  On emulators this isn't needed.
 *
 * ## ASTAP Command-Line Arguments
 *
 * Key flags used:
 *   -f <image>     Input image file (JPEG, PNG, FITS, TIFF)
 *   -d <dbdir>     Path to star database directory (contains .1476 files)
 *   -fov <degrees> Estimated field of view (helps the solver converge faster)
 *   -ra <hours>    Optional RA hint (0-24h) — dramatically speeds up solving
 *   -spd <degrees> South Pole Distance = Dec + 90 (ASTAP convention)
 *   -z <factor>    Downsample factor (2 = half resolution, faster solving)
 *
 * ## Output
 *
 * On success (exit code 0), ASTAP writes a `.wcs` file next to the input
 * image.  This file contains FITS-style key=value pairs:
 *   CRVAL1 = RA center in degrees
 *   CRVAL2 = Dec center in degrees
 *   CDELT1 = pixel scale X (deg/pixel)
 *   CDELT2 = pixel scale Y (deg/pixel)
 *   CROTA2 = image rotation angle in degrees
 *   CD1_1, CD1_2, CD2_1, CD2_2 = CD matrix (alternative to CDELT+CROTA)
 *
 * ## Usage from Python (via Chaquopy Java Interop)
 *
 * ```python
 * from java import jclass
 * AstapSolver = jclass("com.telescopecontroller.solver.AstapSolver")
 * solver = AstapSolver(context)
 * result = solver.solve("/path/to/image.jpg", "/path/to/stardb", 2.5, 12.5, 45.0, 120)
 * if result is not None and result.getSuccess():
 *     ra_h = result.getRaHours()   # RA in hours (0-24)
 *     dec_d = result.getDecDeg()   # Dec in degrees (-90 to +90)
 * ```
 *
 * ## Thread Safety
 *
 * Each solve() call is self-contained and thread-safe.  Multiple solves can
 * run concurrently (though on Android this would be wasteful).  The abort()
 * method can be called from any thread to forcibly kill a running solve.
 *
 * @param context  Android Context, used to locate the native library directory
 *
 * @see AstapDatabaseManager  For downloading and managing star databases
 * @see <a href="https://www.hnsky.org/astap.htm">ASTAP Official Documentation</a>
 */
class AstapSolver(private val context: Context) {

    companion object {
        private const val TAG = "AstapSolver"

        /** Filename of the ASTAP CLI binary in the native library directory. */
        private const val ASTAP_LIB_NAME = "libastapcli.so"

        /**
         * Default timeout for a plate solve operation in seconds.
         * 120s is generous — most solves complete in 2-30 seconds with a
         * position hint.  Blind solves of wide-field images can take longer.
         * This can be overridden per-solve and via user config (solver.timeout).
         */
        private const val DEFAULT_TIMEOUT_SEC = 120L

        /**
         * Check whether the ASTAP CLI binary is available on this device.
         *
         * This checks for the existence of the `libastapcli.so` file in the
         * app's native library directory.  It does NOT check whether a star
         * database is installed (use AstapDatabaseManager for that).
         *
         * @param context  Android Context
         * @return true if the ASTAP binary exists and can potentially be executed
         */
        @JvmStatic
        fun isAvailable(context: Context): Boolean {
            val path = getAstapPath(context)
            return path != null && path.exists()
        }

        /**
         * Get the filesystem path to the ASTAP CLI binary.
         *
         * The binary is stored as `libastapcli.so` in the app's native library
         * directory (e.g., `/data/app/.../lib/arm64/libastapcli.so`).  Android
         * extracts it there at install time because `android:extractNativeLibs="true"`
         * is set in AndroidManifest.xml.
         *
         * @param context  Android Context
         * @return File object pointing to the binary, or null if not found
         */
        @JvmStatic
        fun getAstapPath(context: Context): File? {
            val nativeDir = File(context.applicationInfo.nativeLibraryDir)
            val astap = File(nativeDir, ASTAP_LIB_NAME)
            return if (astap.exists()) astap else null
        }

        /**
         * Detect if we're running in an emulator vs. a real device.
         *
         * This affects how the ASTAP binary is invoked:
         * - Real device: Must use `/system/bin/linker64` as the executable
         *   (because Android marks native libs as ET_DYN shared objects)
         * - Emulator: Can invoke the binary directly (more permissive execution)
         *
         * Detection uses Build.FINGERPRINT, Build.MODEL, and other system
         * properties that differ between emulators and real hardware.
         */
        private fun isEmulator(): Boolean {
            return (Build.FINGERPRINT.startsWith("generic")
                    || Build.FINGERPRINT.startsWith("unknown")
                    || Build.MODEL.contains("Emulator")
                    || Build.MODEL.contains("Android SDK built for x86")
                    || Build.MANUFACTURER.contains("Genymotion")
                    || Build.BRAND.startsWith("generic")
                    || Build.DEVICE.startsWith("generic")
                    || "google_sdk" == Build.PRODUCT)
        }
    }

    /**
     * Data class holding the result of an ASTAP plate solve attempt.
     *
     * On success, contains the solved sky coordinates and image metadata.
     * On failure, contains an error message and any captured stdout/stderr
     * from the ASTAP process for debugging.
     *
     * All coordinate values use standard astronomical conventions:
     * - RA in degrees (0-360) and hours (0-24)
     * - Dec in degrees (-90 to +90)
     * - FOV and pixel scale in degrees
     * - Rotation in degrees (position angle, N through E)
     *
     * Python access via Chaquopy getters: result.getRaDeg(), result.getDecDeg(), etc.
     */
    data class SolveResult(
        val success: Boolean,
        val raDeg: Double = 0.0,       // RA of image center in degrees (0-360)
        val decDeg: Double = 0.0,      // Dec of image center in degrees (-90..+90)
        val raHours: Double = 0.0,     // RA in hours (raDeg / 15.0)
        val fovDeg: Double = 0.0,      // Actual field of view in degrees
        val rotation: Double = 0.0,    // Image rotation / position angle in degrees
        val cdelt1: Double = 0.0,      // Pixel scale X (deg/pixel)
        val cdelt2: Double = 0.0,      // Pixel scale Y (deg/pixel)
        val solveTimeMs: Long = 0,     // Wall-clock time taken in milliseconds
        val stdout: String = "",       // ASTAP process stdout (for debugging)
        val stderr: String = "",       // ASTAP process stderr (for debugging)
        val errorMessage: String = ""  // Human-readable error if success=false
    )

    /**
     * Run ASTAP plate solve on an image file.
     *
     * This is the main entry point.  It:
     * 1. Validates the image path and database directory
     * 2. Locates the ASTAP binary and ensures execute permission
     * 3. Builds the command line (with linker64 prefix on real devices)
     * 4. Runs the process with stdout/stderr capture in background threads
     * 5. Waits up to `timeoutSec` for completion
     * 6. Parses the output .wcs file for the solution coordinates
     * 7. Cleans up temporary files (.wcs, .ini)
     *
     * @param imagePath   Absolute path to the image (JPEG, PNG, FITS, TIFF supported)
     * @param starDbDir   Absolute path to the star database dir (contains .1476 files)
     * @param fovDeg      Estimated field of view in degrees (helps solver converge)
     * @param hintRaHours Optional RA hint in hours (0-24). Pass -1 for blind solve.
     * @param hintDecDeg  Optional Dec hint in degrees (-90..+90). Pass -999 for blind.
     * @param timeoutSec  Maximum solve time in seconds (default 120, configurable via UI)
     * @return SolveResult with coordinates on success, or error info on failure
     */
    @JvmOverloads
    fun solve(
        imagePath: String,
        starDbDir: String,
        fovDeg: Double = 5.0,
        hintRaHours: Double = -1.0,
        hintDecDeg: Double = -999.0,
        timeoutSec: Long = DEFAULT_TIMEOUT_SEC
    ): SolveResult {
        val startTime = System.currentTimeMillis()

        // ── Input validation ─────────────────────────────────────────────
        val imageFile = File(imagePath)
        if (!imageFile.exists()) {
            return SolveResult(
                success = false,
                errorMessage = "Image file not found: $imagePath",
                solveTimeMs = System.currentTimeMillis() - startTime
            )
        }

        val dbDir = File(starDbDir)
        if (!dbDir.exists() || !dbDir.isDirectory) {
            return SolveResult(
                success = false,
                errorMessage = "Star database directory not found: $starDbDir",
                solveTimeMs = System.currentTimeMillis() - startTime
            )
        }

        // ── Locate ASTAP binary ──────────────────────────────────────────
        val astapPath = getAstapPath(context)
        if (astapPath == null || !astapPath.exists()) {
            return SolveResult(
                success = false,
                errorMessage = "ASTAP CLI binary not found in native library directory",
                solveTimeMs = System.currentTimeMillis() - startTime
            )
        }

        // ── Ensure execute permission ────────────────────────────────────
        // The binary is stored as a .so file (Android packaging trick).
        // Even though Android extracts it with execute permission, some
        // devices/ROMs strip the +x bit.  We try to set it explicitly.
        if (!astapPath.canExecute()) {
            try {
                val chmod = ProcessBuilder("chmod", "+x", astapPath.absolutePath)
                    .start()
                chmod.waitFor(5, TimeUnit.SECONDS)
                Log.i(TAG, "Set execute permission on ${astapPath.name}")
            } catch (e: Exception) {
                Log.w(TAG, "chmod failed (may still work via linker): ${e.message}")
            }
        }

        // ── Build command line ───────────────────────────────────────────
        // On real devices, we must prefix with the system linker because
        // Android marks all files in nativeLibraryDir as ET_DYN (shared
        // objects), not ET_EXEC.  The linker can load and execute them.
        val cmdline = mutableListOf<String>()

        if (!isEmulator()) {
            // Choose 32-bit or 64-bit linker based on device ABI support
            val linker = if (Build.SUPPORTED_64_BIT_ABIS.isNotEmpty()) {
                "/system/bin/linker64"
            } else {
                "/system/bin/linker"
            }
            cmdline.add(linker)
        }

        cmdline.add(astapPath.absolutePath)

        // Required arguments
        cmdline.add("-f")
        cmdline.add(imagePath)
        cmdline.add("-d")
        cmdline.add(starDbDir)
        cmdline.add("-fov")
        cmdline.add("%.2f".format(fovDeg))

        // Optional position hint (dramatically speeds up solving from ~30s to ~2s)
        // ASTAP uses "South Pole Distance" (SPD) instead of Declination:
        //   SPD = Dec + 90  (so Dec=0 -> SPD=90, Dec=-90 -> SPD=0, Dec=+90 -> SPD=180)
        if (hintRaHours >= 0 && hintDecDeg > -999.0) {
            cmdline.add("-ra")
            cmdline.add("%.6f".format(hintRaHours))    // RA in hours (0-24)
            cmdline.add("-spd")
            cmdline.add("%.6f".format(hintDecDeg + 90.0))  // South Pole Distance
        }

        // Downsample large images for faster solving (2x = half resolution)
        // This is a good trade-off: most images have more than enough stars
        // at half resolution, and solving speed roughly scales with pixel count
        cmdline.add("-z")
        cmdline.add("2")

        Log.i(TAG, "ASTAP command: ${cmdline.joinToString(" ")}")

        // ── Run the ASTAP process ────────────────────────────────────────
        // ProcessBuilder starts the process, and we read stdout/stderr in
        // separate daemon threads to prevent pipe buffer deadlocks.
        var process: Process? = null
        try {
            process = ProcessBuilder(cmdline)
                .directory(astapPath.parentFile)   // Working dir = binary location
                .redirectErrorStream(false)         // Keep stdout and stderr separate
                .start()

            // Track for abort() support
            currentProcess = process

            // Read stdout and stderr concurrently to avoid blocking
            // (if a pipe buffer fills up, the process would hang)
            val stdoutReader = StreamReader(process.inputStream)
            val stderrReader = StreamReader(process.errorStream)
            stdoutReader.start()
            stderrReader.start()

            // Wait for completion with user-configurable timeout
            val completed = process.waitFor(timeoutSec, TimeUnit.SECONDS)
            if (!completed) {
                process.destroyForcibly()
                return SolveResult(
                    success = false,
                    errorMessage = "ASTAP solve timed out after ${timeoutSec}s",
                    stdout = stdoutReader.getOutput(),
                    stderr = stderrReader.getOutput(),
                    solveTimeMs = System.currentTimeMillis() - startTime
                )
            }

            // Wait for stream readers to finish (they may still be reading
            // after the process exits due to buffering)
            stdoutReader.join(2000)
            stderrReader.join(2000)

            val exitCode = process.exitValue()
            val stdout = stdoutReader.getOutput()
            val stderr = stderrReader.getOutput()
            val elapsed = System.currentTimeMillis() - startTime

            Log.i(TAG, "ASTAP exit code: $exitCode, time: ${elapsed}ms")
            if (stdout.isNotBlank()) Log.d(TAG, "ASTAP stdout: $stdout")
            if (stderr.isNotBlank()) Log.d(TAG, "ASTAP stderr: $stderr")

            // ASTAP returns non-zero exit code when it fails to solve
            // (no matching star pattern found in the database)
            if (exitCode != 0) {
                return SolveResult(
                    success = false,
                    errorMessage = "ASTAP failed (exit $exitCode): ${stderr.ifBlank { stdout }}",
                    stdout = stdout,
                    stderr = stderr,
                    solveTimeMs = elapsed
                )
            }

            // ── Parse the .wcs output file ───────────────────────────────
            // ASTAP writes the solution to a .wcs file alongside the input
            // image (same name, different extension).  Two naming conventions:
            //   1. Replace extension: image.jpg -> image.wcs
            //   2. Append extension: image.jpg -> image.jpg.wcs
            val wcsPath = imagePath.replaceAfterLast('.', "wcs")
            val wcsFile = File(wcsPath)
            if (!wcsFile.exists()) {
                val wcsPath2 = "$imagePath.wcs"
                val wcsFile2 = File(wcsPath2)
                if (wcsFile2.exists()) {
                    return parseWcsFile(wcsFile2, stdout, stderr, elapsed)
                }
                return SolveResult(
                    success = false,
                    errorMessage = "ASTAP exited OK but no .wcs file found at $wcsPath",
                    stdout = stdout,
                    stderr = stderr,
                    solveTimeMs = elapsed
                )
            }

            return parseWcsFile(wcsFile, stdout, stderr, elapsed)

        } catch (e: Exception) {
            Log.e(TAG, "ASTAP execution failed", e)
            return SolveResult(
                success = false,
                errorMessage = "ASTAP execution error: ${e.message}",
                solveTimeMs = System.currentTimeMillis() - startTime
            )
        } finally {
            currentProcess = null
            process?.destroyForcibly()
        }
    }

    // ── Abort support ────────────────────────────────────────────────────
    // abort() can be called from any thread (e.g. a UI cancel button) to
    // forcibly kill a running ASTAP process.  The @Volatile annotation
    // ensures visibility across threads without explicit synchronization.

    @Volatile
    private var currentProcess: Process? = null

    /**
     * Abort a currently running plate solve.
     *
     * Safe to call from any thread.  If no solve is running, this is a no-op.
     * The solve() call that was interrupted will return a failure result.
     */
    fun abort() {
        currentProcess?.destroyForcibly()
        Log.i(TAG, "ASTAP solve aborted")
    }

    // ═════════════════════════════════════════════════════════════════════
    //  WCS file parser
    // ═════════════════════════════════════════════════════════════════════
    //
    // ASTAP .wcs files use a simplified FITS header format:
    //   KEY     = value           (numeric)
    //   KEY     = 'string value'  (string with single quotes)
    //
    // We extract the standard WCS keywords that define the astrometric
    // solution (CRVAL1/2, CDELT1/2, CROTA, CD matrix).
    //
    // Two coordinate systems may be present:
    // 1. CDELT + CROTA: Simple scale + rotation (older convention)
    // 2. CD matrix (CD1_1, CD1_2, CD2_1, CD2_2): General affine transform
    //    Encodes scale, rotation, shear, and flip in one 2x2 matrix.
    //
    // We prefer the CD matrix when available and fall back to CDELT+CROTA.

    private fun parseWcsFile(
        wcsFile: File,
        stdout: String,
        stderr: String,
        elapsedMs: Long
    ): SolveResult {
        try {
            val wcsData = mutableMapOf<String, String>()

            // Regex patterns for numeric and string WCS key-value pairs
            val numPattern = Pattern.compile("^([A-Z0-9_-]+)\\s*=\\s*([0-9.+\\-eE]+)")
            val strPattern = Pattern.compile("^([A-Z0-9_-]+)\\s*=\\s*'([^']+)'")

            wcsFile.forEachLine { line ->
                val numMatch = numPattern.matcher(line.trim())
                val strMatch = strPattern.matcher(line.trim())
                when {
                    numMatch.find() -> wcsData[numMatch.group(1)!!] = numMatch.group(2)!!
                    strMatch.find() -> wcsData[strMatch.group(1)!!] = strMatch.group(2)!!
                }
            }

            // ── Extract WCS values ───────────────────────────────────────
            val crval1 = wcsData["CRVAL1"]?.toDoubleOrNull() ?: 0.0  // RA center (degrees)
            val crval2 = wcsData["CRVAL2"]?.toDoubleOrNull() ?: 0.0  // Dec center (degrees)
            val cdelt1 = wcsData["CDELT1"]?.toDoubleOrNull() ?: 0.0  // Pixel scale X
            val cdelt2 = wcsData["CDELT2"]?.toDoubleOrNull() ?: 0.0  // Pixel scale Y
            val crota2 = wcsData["CROTA2"]?.toDoubleOrNull()
                ?: wcsData["CROTA1"]?.toDoubleOrNull() ?: 0.0  // Rotation angle

            // CD matrix elements (more general than CDELT+CROTA)
            val cd11 = wcsData["CD1_1"]?.toDoubleOrNull()
            val cd12 = wcsData["CD1_2"]?.toDoubleOrNull()
            val cd21 = wcsData["CD2_1"]?.toDoubleOrNull()
            val cd22 = wcsData["CD2_2"]?.toDoubleOrNull()

            // ── Derive FOV from stdout ────────────────────────────────────
            // ASTAP prints "FOV=X.XX" in its stdout on successful solves
            var fov = 0.0
            val fovPattern = Pattern.compile("FOV=([0-9.]+)")
            val fovMatch = fovPattern.matcher(stdout)
            if (fovMatch.find()) {
                fov = fovMatch.group(1)?.toDoubleOrNull() ?: 0.0
            }

            // ── Compute effective pixel scale from CD matrix ─────────────
            // The CD matrix encodes scale + rotation.  The pixel scale along
            // each axis is the magnitude of the corresponding matrix row:
            //   scale_x = sqrt(CD1_1^2 + CD1_2^2)
            //   scale_y = sqrt(CD2_1^2 + CD2_2^2)
            val effectiveCdelt1 = if (cd11 != null && cd12 != null) {
                Math.sqrt(cd11 * cd11 + cd12 * cd12)
            } else {
                Math.abs(cdelt1)
            }
            val effectiveCdelt2 = if (cd21 != null && cd22 != null) {
                Math.sqrt(cd21 * cd21 + cd22 * cd22)
            } else {
                Math.abs(cdelt2)
            }

            // ── Compute rotation from CD matrix if CROTA not explicit ────
            // rotation = atan2(CD1_2, CD1_1) gives the position angle
            val effectiveRotation = if (cd11 != null && cd12 != null && crota2 == 0.0) {
                Math.toDegrees(Math.atan2(cd12, cd11))
            } else {
                crota2
            }

            // Convert RA from degrees to hours (astronomical convention)
            val raHours = crval1 / 15.0

            Log.i(TAG, "WCS solution: RA=${raHours}h Dec=${crval2}d FOV=${fov}d rot=${effectiveRotation}d")

            // ── Clean up ASTAP output files ──────────────────────────────
            // ASTAP creates .wcs and sometimes .ini files next to the image.
            // We clean them up to avoid accumulating temp files.
            try {
                wcsFile.delete()
                val iniFile = File(wcsFile.absolutePath.replaceAfterLast('.', "ini"))
                if (iniFile.exists()) iniFile.delete()
            } catch (e: Exception) {
                // Non-critical — don't fail the solve for cleanup errors
            }

            return SolveResult(
                success = true,
                raDeg = crval1,
                decDeg = crval2,
                raHours = raHours,
                fovDeg = fov,
                rotation = effectiveRotation,
                cdelt1 = effectiveCdelt1,
                cdelt2 = effectiveCdelt2,
                solveTimeMs = elapsedMs,
                stdout = stdout,
                stderr = stderr
            )

        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse WCS file", e)
            return SolveResult(
                success = false,
                errorMessage = "WCS parse error: ${e.message}",
                stdout = stdout,
                stderr = stderr,
                solveTimeMs = elapsedMs
            )
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    //  StreamReader: read process output in a background thread
    // ═════════════════════════════════════════════════════════════════════
    //
    // ProcessBuilder pipes (stdout/stderr) have a fixed OS buffer (~64KB).
    // If the process writes more than the buffer can hold before we read,
    // the process blocks and appears to hang.  Reading in separate daemon
    // threads prevents this deadlock.

    private class StreamReader(private val stream: InputStream) : Thread() {
        private val buffer = StringBuilder()

        init {
            isDaemon = true  // Don't prevent JVM shutdown
        }

        override fun run() {
            try {
                BufferedReader(InputStreamReader(stream)).use { reader ->
                    var line = reader.readLine()
                    while (line != null) {
                        buffer.appendLine(line)
                        line = reader.readLine()
                    }
                }
            } catch (e: Exception) {
                buffer.appendLine("[stream read error: ${e.message}]")
            }
        }

        fun getOutput(): String = buffer.toString().trim()
    }
}
