package com.telescopecontroller.solver

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.Handler
import android.os.Looper
import android.util.Log
import java.io.*
import java.net.UnknownHostException
import java.util.zip.ZipInputStream
import org.tukaani.xz.XZInputStream

/**
 * Manages ASTAP star database lifecycle: download, extraction, status, and deletion.
 *
 * ## Star Databases
 *
 * ASTAP uses proprietary `.1476` star index files derived from the Gaia DR3 catalog.
 * Four database tiers are available, trading size for sky coverage depth:
 *
 * | Name | Size   | Min FOV | Best For                                    |
 * |------|--------|---------|---------------------------------------------|
 * | D05  | ~45 MB | 0.6 deg | Most telescopes (SCT, refractor, Newtonians) |
 * | D20  | ~170MB | 0.3 deg | Narrow-field (long focal length)             |
 * | D50  | ~500MB | 0.2 deg | Very narrow fields (planetary cameras)       |
 * | W08  | ~15 MB | 20 deg  | Wide-field (phone cameras, DSLR lenses)      |
 *
 * Download source: https://sourceforge.net/projects/astap-program/files/star_databases/
 *
 * ## Storage Layout
 *
 * Databases are stored in the app's external files directory:
 * ```
 * <externalFilesDir>/astap_databases/
 *   d05/
 *     d05_0000.1476
 *     d05_0100.1476
 *     ...
 *     ready.txt          <-- Marker file: database fully extracted and usable
 *   d20/
 *     ...
 * ```
 *
 * The `ready.txt` marker file is written ONLY after all `.1476` files are
 * successfully extracted.  Its presence indicates the database is complete
 * and safe to use.  If the app crashes mid-extraction, the marker won't
 * exist and the user can re-download.
 *
 * ## Archive Format
 *
 * SourceForge hosts databases as `.deb` files (Debian packages), which are:
 * ```
 * .deb file (Unix "ar" archive)
 *   ├── debian-binary        (version string "2.0")
 *   ├── control.tar.xz       (package metadata)
 *   └── data.tar.xz          (actual files: ./opt/astap/{name}.1476)
 * ```
 *
 * Extraction chain: .deb -> ar parse -> data.tar.xz -> xz decompress -> tar extract
 *
 * XZ decompression uses the `org.tukaani:xz` pure Java library (Apache 2.0, ~130KB),
 * since Android's stdlib lacks XZ/LZMA2 support.  Multi-strategy extraction:
 * 1. Parse the ar archive, find data.tar, check compression → extract via XZ/Gzip/raw
 * 2. If xz compressed → try the device's `xz` command-line tool (available on some ROMs)
 * 3. Fall back to trying the whole file as a zip archive (some mirrors provide .zip)
 *
 * ## Real-Time Progress Tracking (SSE)
 *
 * Download progress is exposed via volatile fields that the Python bridge
 * polls.  The web UI connects to a Server-Sent Events (SSE) endpoint
 * (`GET /api/solver/databases/progress`) which reads these fields and
 * pushes real-time updates to the browser — no WebSocket needed.
 *
 * Progress flow:
 * ```
 * AstapDatabaseManager (Kotlin, background thread)
 *   updates @Volatile fields: downloadState, downloadProgress, downloadTotal, ...
 *     ↓
 * local_solver.py get_download_progress() (Python, reads via Chaquopy interop)
 *     ↓
 * web_server.py /api/solver/databases/progress (SSE generator, yields JSON events)
 *     ↓
 * Browser EventSource (JavaScript, updates progress bar in real time)
 * ```
 *
 * ## Usage from Python (via Chaquopy)
 *
 * ```python
 * from java import jclass
 * DbMgr = jclass("com.telescopecontroller.solver.AstapDatabaseManager")
 * mgr = DbMgr(context)
 * if not mgr.isDatabaseInstalled("d05"):
 *     mgr.downloadDatabase("d05")
 *     # Poll mgr.getDownloadState() for progress...
 * dbPath = mgr.getDatabasePath("d05")
 * ```
 *
 * @param context  Android Context, used for getExternalFilesDir()
 *
 * @see AstapSolver  For the plate solve execution wrapper
 */
class AstapDatabaseManager(private val context: Context) {

    companion object {
        private const val TAG = "AstapDbManager"

        /** Marker filename written to a database dir after successful extraction. */
        private const val READY_MARKER = "ready.txt"

        /**
         * Database catalog: name → download URL, size, FOV range, description.
         *
         * URLs point to SourceForge's /download redirect endpoint which
         * redirects through multiple hops (SF mirror selection) to the
         * actual download server.  We follow redirects manually because
         * HttpURLConnection.setInstanceFollowRedirects doesn't handle
         * cross-domain redirects reliably.
         */
        val DATABASES = mapOf(
            "d05" to DatabaseInfo(
                url = "https://sourceforge.net/projects/astap-program/files/star_databases/d05_star_database.deb/download",
                sizeMb = 45,
                description = "Small database (45 MB) -- FOV >= 0.6 degrees",
                minFovDeg = 0.6,
                maxFovDeg = 6.0
            ),
            "d20" to DatabaseInfo(
                url = "https://sourceforge.net/projects/astap-program/files/star_databases/d20_star_database.deb/download",
                sizeMb = 170,
                description = "Medium database (170 MB) -- FOV >= 0.3 degrees",
                minFovDeg = 0.3,
                maxFovDeg = 6.0
            ),
            "d50" to DatabaseInfo(
                url = "https://sourceforge.net/projects/astap-program/files/star_databases/d50_star_database.deb/download",
                sizeMb = 500,
                description = "Large database (500 MB) -- FOV >= 0.2 degrees",
                minFovDeg = 0.2,
                maxFovDeg = 6.0
            ),
            "w08" to DatabaseInfo(
                url = "https://sourceforge.net/projects/astap-program/files/star_databases/w08_star_database_mag08_astap.deb/download",
                sizeMb = 15,
                description = "Wide-field database (15 MB) -- FOV >= 20 degrees (phone cameras)",
                minFovDeg = 20.0,
                maxFovDeg = 180.0
            )
        )

        /**
         * Metadata for a single ASTAP star database.
         *
         * @property url        SourceForge download URL (follows redirects to actual mirror)
         * @property sizeMb     Approximate download size in megabytes
         * @property description Human-readable description for the UI
         * @property minFovDeg  Minimum field of view this database supports (degrees)
         * @property maxFovDeg  Maximum practical field of view (degrees)
         */
        data class DatabaseInfo(
            val url: String,
            val sizeMb: Int,
            val description: String,
            val minFovDeg: Double,
            val maxFovDeg: Double
        )

        /**
         * Recommend the best database for a given field of view.
         *
         * Selection logic:
         *   FOV >= 20° → W08 (wide-field, phone cameras)
         *   FOV >= 0.6° → D05 (most telescopes)
         *   FOV >= 0.3° → D20 (narrow-field)
         *   FOV < 0.3° → D50 (very narrow, planetary)
         *
         * @param fovDeg  Estimated field of view in degrees
         * @return Recommended database name ("d05", "d20", "d50", or "w08")
         */
        @JvmStatic
        fun recommendDatabase(fovDeg: Double): String {
            return when {
                fovDeg >= 20.0 -> "w08"
                fovDeg >= 0.6 -> "d05"
                fovDeg >= 0.3 -> "d20"
                else -> "d50"
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Progress callback interface (Java-side listeners)
    // ═════════════════════════════════════════════════════════════════════

    /**
     * Callback interface for monitoring download/extraction progress.
     *
     * Implementors receive events on the download background thread (NOT
     * the main thread).  If updating UI, post to the main looper.
     *
     * NOTE: The Python bridge does NOT use this interface directly.
     * Instead, it reads the @Volatile progress fields below.
     */
    interface ProgressCallback {
        fun onDownloadStarted(dbName: String)
        fun onDownloadProgress(dbName: String, bytesDownloaded: Long, totalBytes: Long)
        fun onDownloadComplete(dbName: String)
        fun onExtractionStarted(dbName: String)
        fun onExtractionProgress(dbName: String, filesExtracted: Int, totalFiles: Int)
        fun onExtractionComplete(dbName: String, success: Boolean, error: String?)
    }

    @Volatile
    private var progressCallback: ProgressCallback? = null
    private var currentDownloadId: Long = -1
    private val handler = Handler(Looper.getMainLooper())

    fun setProgressCallback(callback: ProgressCallback?) {
        this.progressCallback = callback
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Real-time progress fields (read by Python SSE bridge)
    // ═════════════════════════════════════════════════════════════════════
    //
    // These @Volatile fields are updated by the download background thread
    // and read by the Python bridge (via Chaquopy getters) for SSE progress
    // streaming.  @Volatile ensures cross-thread visibility without locks.
    //
    // State machine:
    //   "idle" → "downloading" → "extracting" → "complete"
    //                                         → "error" (on failure)
    //
    // The Python bridge polls these every ~500ms from the SSE generator.

    /** Current download state: "idle", "downloading", "extracting", "complete", "error" */
    @Volatile
    var downloadState: String = "idle"
        private set

    /** Bytes downloaded so far (updated during "downloading" state) */
    @Volatile
    var downloadProgress: Long = 0
        private set

    /** Total expected bytes (-1 if unknown, updated when Content-Length is received) */
    @Volatile
    var downloadTotal: Long = -1
        private set

    /** Number of .1476 files extracted so far (updated during "extracting" state) */
    @Volatile
    var extractedFiles: Int = 0
        private set

    /** Error message if downloadState == "error" */
    @Volatile
    var downloadError: String = ""
        private set

    /** Name of the database currently being downloaded (e.g., "d05") */
    @Volatile
    var downloadingDbName: String = ""
        private set

    // ═════════════════════════════════════════════════════════════════════
    //  Database status queries
    // ═════════════════════════════════════════════════════════════════════

    /**
     * Check if a database is fully installed and ready to use.
     *
     * Checks for the `ready.txt` marker file, which is only created
     * after all .1476 files are successfully extracted.
     */
    fun isDatabaseInstalled(dbName: String): Boolean {
        val readyFile = File(getDatabaseDir(dbName), READY_MARKER)
        return readyFile.exists()
    }

    /**
     * Get the absolute path to a database directory.
     *
     * This path is passed to ASTAP's `-d` flag.  The directory may or may
     * not exist yet (it's created during download).
     *
     * @return Absolute path string (e.g., "/storage/emulated/0/Android/data/.../astap_databases/d05")
     */
    fun getDatabasePath(dbName: String): String {
        return getDatabaseDir(dbName).absolutePath
    }

    /**
     * Get the first installed database, preferring smaller ones.
     *
     * Check order: d05 → d20 → d50 → w08.  This prefers the "general
     * purpose" database (D05) over more specialized ones.
     *
     * @return Database name string, or null if no database is installed
     */
    fun getInstalledDatabase(): String? {
        for (name in listOf("d05", "d20", "d50", "w08")) {
            if (isDatabaseInstalled(name)) return name
        }
        return null
    }

    /**
     * Get info about all databases and their installation status.
     *
     * Returns a map compatible with JSON serialization for the web API.
     * Each entry contains: installed, size_mb, description, min_fov, max_fov, path
     *
     * Called by Python's `local_solver.get_database_status()` via Chaquopy.
     */
    fun getDatabaseStatus(): Map<String, Map<String, Any>> {
        val result = mutableMapOf<String, Map<String, Any>>()
        for ((name, info) in DATABASES) {
            result[name] = mapOf(
                "installed" to isDatabaseInstalled(name),
                "size_mb" to info.sizeMb,
                "description" to info.description,
                "min_fov" to info.minFovDeg,
                "max_fov" to info.maxFovDeg,
                "path" to getDatabasePath(name)
            )
        }
        return result
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Database download and extraction
    // ═════════════════════════════════════════════════════════════════════

    /**
     * Download and install a star database (non-blocking).
     *
     * Launches a background thread that:
     * 1. Downloads the .deb archive from SourceForge (following redirects)
     * 2. Extracts .1476 star index files from the archive
     * 3. Writes a ready.txt marker on success
     *
     * Progress is reported via:
     * - The ProgressCallback interface (for Java listeners)
     * - The @Volatile progress fields (for Python SSE bridge)
     *
     * Safe to call multiple times — returns immediately if already installed.
     *
     * @param dbName  Database name: "d05", "d20", "d50", or "w08"
     */
    fun downloadDatabase(dbName: String) {
        val info = DATABASES[dbName]
        if (info == null) {
            Log.e(TAG, "Unknown database: $dbName")
            downloadState = "error"
            downloadError = "Unknown database: $dbName"
            progressCallback?.onExtractionComplete(dbName, false, "Unknown database: $dbName")
            return
        }

        if (isDatabaseInstalled(dbName)) {
            Log.i(TAG, "Database $dbName is already installed")
            downloadState = "complete"
            progressCallback?.onExtractionComplete(dbName, true, null)
            return
        }

        val dbDir = getDatabaseDir(dbName)
        dbDir.mkdirs()

        // Reset progress tracking fields
        downloadState = "downloading"
        downloadProgress = 0
        downloadTotal = info.sizeMb.toLong() * 1024 * 1024  // Estimate until we get Content-Length
        extractedFiles = 0
        downloadError = ""
        downloadingDbName = dbName

        Log.i(TAG, "Starting download of $dbName (${info.sizeMb} MB)")
        progressCallback?.onDownloadStarted(dbName)

        // Download in a background daemon thread
        // (DownloadManager has issues with SourceForge's multi-hop redirects,
        // so we use HttpURLConnection with manual redirect following)
        Thread {
            try {
                downloadAndExtract(dbName, info)
            } catch (e: UnknownHostException) {
                // DNS resolution failed -- most likely the phone is on
                // the telescope controller's WiFi which has no internet
                val userMsg = "No internet connection. Connect to a WiFi " +
                    "network with internet access or enable mobile data, " +
                    "then try again."
                Log.e(TAG, "DNS resolution failed for $dbName: ${e.message}")
                downloadState = "error"
                downloadError = userMsg
                progressCallback?.onExtractionComplete(dbName, false, userMsg)
            } catch (e: Exception) {
                Log.e(TAG, "Database download failed", e)
                downloadState = "error"
                downloadError = "Download failed: ${e.message}"
                progressCallback?.onExtractionComplete(
                    dbName, false, "Download failed: ${e.message}"
                )
            }
        }.apply {
            isDaemon = true
            name = "AstapDbDownload-$dbName"
            start()
        }
    }

    /**
     * Delete an installed database to free storage space.
     *
     * Recursively deletes the database directory and all .1476 files.
     * The user can re-download later if needed.
     *
     * @return true if deletion succeeded, false on error
     */
    fun deleteDatabase(dbName: String): Boolean {
        val dbDir = getDatabaseDir(dbName)
        return try {
            dbDir.deleteRecursively()
            Log.i(TAG, "Deleted database $dbName")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to delete database $dbName", e)
            false
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Internal: directory helpers
    // ═════════════════════════════════════════════════════════════════════

    /**
     * Get the directory for a specific database.
     *
     * Uses getExternalFilesDir(null) which gives us:
     *   /storage/emulated/0/Android/data/com.telescopecontroller/files/
     *
     * This directory is:
     * - Readable/writable without storage permissions (app-private external)
     * - Cleared when the app is uninstalled
     * - Visible via USB file transfer (unlike internal storage)
     */
    private fun getDatabaseDir(dbName: String): File {
        return File(context.getExternalFilesDir(null), "astap_databases/$dbName")
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Internal: download and extraction pipeline
    // ═════════════════════════════════════════════════════════════════════

    /**
     * Download the database archive and extract .1476 star index files.
     *
     * This is the main download+extract pipeline, running on a background thread.
     *
     * ## SourceForge Redirect Chain
     *
     * SourceForge download URLs go through multiple redirects:
     *   1. sourceforge.net/projects/.../download  (302 → SF mirror selection)
     *   2. downloads.sourceforge.net/...           (302 → actual mirror)
     *   3. mirror.example.com/...                  (200 → actual file)
     *
     * We follow up to 10 redirects manually because HttpURLConnection's
     * setInstanceFollowRedirects doesn't handle cross-domain redirects
     * (different hosts) which SourceForge uses extensively.
     *
     * ## Archive Extraction
     *
     * The downloaded file is a .deb (ar archive) containing data.tar.xz.
     * We try multiple extraction strategies (see extractDeb docs).
     */
    private fun downloadAndExtract(dbName: String, info: DatabaseInfo) {
        val dbDir = getDatabaseDir(dbName)
        dbDir.mkdirs()

        val tempFile = File(dbDir, "$dbName.download")

        try {
            // ── Phase 1: Download ────────────────────────────────────────
            Log.i(TAG, "Downloading $dbName from ${info.url}")

            // Follow SourceForge redirects manually (up to 10 hops)
            var redirectUrl = info.url
            var redirectCount = 0
            var conn: java.net.HttpURLConnection

            while (redirectCount < 10) {
                conn = java.net.URL(redirectUrl).openConnection() as java.net.HttpURLConnection
                conn.instanceFollowRedirects = false  // We handle redirects ourselves
                conn.connectTimeout = 15000
                conn.readTimeout = 60000
                conn.setRequestProperty("User-Agent", "TelescopeController/1.0")

                val code = conn.responseCode
                if (code in 300..399) {
                    // Follow redirect
                    redirectUrl = conn.getHeaderField("Location") ?: break
                    conn.disconnect()
                    redirectCount++
                    Log.d(TAG, "Redirect #$redirectCount -> $redirectUrl")
                    continue
                }

                if (code != 200) {
                    conn.disconnect()
                    throw IOException("HTTP $code downloading $dbName from $redirectUrl")
                }

                // ── Streaming download with progress updates ─────────────
                val totalBytes = conn.contentLengthLong
                if (totalBytes > 0) {
                    downloadTotal = totalBytes  // Update with actual Content-Length
                }
                var downloaded = 0L

                conn.inputStream.use { input ->
                    FileOutputStream(tempFile).use { output ->
                        val buffer = ByteArray(65536)  // 64KB read buffer
                        var read: Int
                        while (input.read(buffer).also { read = it } != -1) {
                            output.write(buffer, 0, read)
                            downloaded += read
                            // Update progress fields for SSE streaming
                            downloadProgress = downloaded
                            if (totalBytes > 0) {
                                progressCallback?.onDownloadProgress(
                                    dbName, downloaded, totalBytes
                                )
                            }
                        }
                    }
                }
                conn.disconnect()

                Log.i(TAG, "Downloaded $dbName: ${downloaded / 1024}KB")
                downloadState = "extracting"
                progressCallback?.onDownloadComplete(dbName)
                break
            }

            if (!tempFile.exists() || tempFile.length() < 1000) {
                throw IOException("Download produced empty or missing file")
            }

            // ── Phase 2: Extract .1476 files from archive ────────────────
            progressCallback?.onExtractionStarted(dbName)

            val extracted = extractDeb(tempFile, dbDir, dbName)

            if (extracted > 0) {
                // Write the ready marker (contains db name and timestamp for diagnostics)
                File(dbDir, READY_MARKER).writeText(
                    "ready\n$dbName\n${System.currentTimeMillis()}"
                )
                Log.i(TAG, "Database $dbName installed: $extracted files extracted")
                downloadState = "complete"
                progressCallback?.onExtractionComplete(dbName, true, null)
            } else {
                throw IOException("No star database files extracted from archive")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Download/extract failed for $dbName", e)
            downloadState = "error"
            downloadError = "Failed: ${e.message}"
            progressCallback?.onExtractionComplete(
                dbName, false, "Failed: ${e.message}"
            )
        } finally {
            // Clean up temp download file (the actual .1476 files stay)
            tempFile.delete()
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Archive extraction: .deb / ar / tar / zip
    // ═════════════════════════════════════════════════════════════════════

    /**
     * Extract .1476 star index files from a downloaded archive.
     *
     * Tries multiple formats in order:
     * 1. Unix ar archive (.deb format) → parse headers → find data.tar → extract
     * 2. Zip archive (.zip) → ZipInputStream → extract .1476 entries
     * 3. Raw scan → try xz decompression via shell command → tar extract
     *
     * @return Number of .1476 files successfully extracted
     */
    private fun extractDeb(debFile: File, destDir: File, dbName: String): Int {
        var extracted = 0

        try {
            RandomAccessFile(debFile, "r").use { raf ->
                // Check for Unix ar archive magic: "!<arch>\n"
                val magic = ByteArray(8)
                raf.readFully(magic)
                val magicStr = String(magic)

                if (magicStr == "!<arch>\n") {
                    // Standard .deb format (ar archive)
                    extracted = extractFromAr(raf, destDir, dbName)
                } else {
                    // Not ar — try as zip (some mirrors provide .zip alternatives)
                    Log.i(TAG, "Not an ar archive, trying as zip...")
                    extracted = extractFromZip(debFile, destDir, dbName)
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Structured extraction failed: ${e.message}, trying raw scan...")
            // Last resort: attempt xz decompression via shell + tar extract
            extracted = extractByRawScan(debFile, destDir, dbName)
        }

        return extracted
    }

    /**
     * Extract .1476 files from a Unix ar archive (the standard .deb format).
     *
     * ## ar Archive Format
     *
     * An ar archive consists of:
     * - 8-byte global header: "!<arch>\n"
     * - Sequence of entries, each with:
     *   - 60-byte entry header:
     *     - name[16] + mtime[12] + uid[6] + gid[6] + mode[8] + size[10] + magic[2]
     *   - Entry data (padded to 2-byte alignment)
     *
     * A .deb file typically contains three entries:
     *   1. debian-binary     (text: "2.0\n")
     *   2. control.tar.xz    (package metadata, we skip this)
     *   3. data.tar.xz       (actual file contents — this has our .1476 files)
     *
     * We scan for the "data.tar" entry and extract its contents.
     */
    private fun extractFromAr(raf: RandomAccessFile, destDir: File, dbName: String): Int {
        var extracted = 0

        while (raf.filePointer < raf.length()) {
            // Each ar entry has a 60-byte header
            if (raf.length() - raf.filePointer < 60) break

            val header = ByteArray(60)
            raf.readFully(header)
            val headerStr = String(header)

            // Parse entry name (first 16 chars) and size (bytes 48-57, decimal)
            val name = headerStr.substring(0, 16).trim()
            val sizeStr = headerStr.substring(48, 58).trim()
            val entrySize = sizeStr.toLongOrNull() ?: 0

            Log.d(TAG, "ar entry: name='$name' size=$entrySize")

            if (name.startsWith("data.tar")) {
                // Found the data archive — read it to a temp file for extraction
                val dataTar = File(destDir, "data.tar.tmp")
                try {
                    FileOutputStream(dataTar).use { fos ->
                        val buf = ByteArray(65536)
                        var remaining = entrySize
                        while (remaining > 0) {
                            val toRead = minOf(remaining, buf.size.toLong()).toInt()
                            raf.readFully(buf, 0, toRead)
                            fos.write(buf, 0, toRead)
                            remaining -= toRead
                        }
                    }

                    // Extract .1476 files from the tar (possibly xz/gzip compressed)
                    extracted = extractFromTar(dataTar, destDir, dbName)

                } finally {
                    dataTar.delete()
                }
                break  // We found and processed the data archive
            } else {
                // Skip this entry's data
                raf.seek(raf.filePointer + entrySize)
            }

            // ar entries are padded to 2-byte alignment
            if (entrySize % 2 != 0L && raf.filePointer < raf.length()) {
                raf.seek(raf.filePointer + 1)
            }
        }

        return extracted
    }

    /**
     * Extract .1476 files from a tar archive (possibly compressed).
     *
     * Handles:
     * - Uncompressed .tar → parse directly
     * - .tar.gz (gzip) → decompress via GZIPInputStream → parse
     * - .tar.xz (xz) → fall back to extractByRawScan (Android lacks XZ in stdlib)
     *
     * ## tar File Format
     *
     * A tar file is a sequence of 512-byte header blocks followed by file data:
     * - Header: name[100] + mode[8] + uid[8] + gid[8] + size[12] + mtime[12] + ...
     * - Size is stored in OCTAL (base 8) ASCII
     * - Data is padded to 512-byte boundaries
     * - Two consecutive zero blocks mark end-of-archive
     *
     * ASTAP database tar entries have paths like:
     *   ./opt/astap/d05_0000.1476
     *   ./opt/astap/d05_0100.1476
     *
     * We extract only files ending in ".1476" and strip the directory prefix.
     */
    private fun extractFromTar(tarFile: File, destDir: File, dbName: String): Int {
        var extracted = 0

        // Detect compression by reading magic bytes
        val isGzip = try {
            FileInputStream(tarFile).use { fis ->
                val b1 = fis.read()
                val b2 = fis.read()
                b1 == 0x1f && b2 == 0x8b  // Gzip magic: 1F 8B
            }
        } catch (e: Exception) { false }

        val isXz = try {
            FileInputStream(tarFile).use { fis ->
                val magic = ByteArray(6)
                fis.read(magic)
                // XZ magic: FD 37 7A 58 5A 00
                magic[0] == 0xFD.toByte() && magic[1] == 0x37.toByte() &&
                        magic[2] == 0x7A.toByte() && magic[3] == 0x58.toByte() &&
                        magic[4] == 0x5A.toByte() && magic[5] == 0x00.toByte()
            }
        } catch (e: Exception) { false }

        // Open the stream with the appropriate decompression wrapper.
        // XZ uses org.tukaani:xz pure Java library (Android stdlib lacks XZ).
        // Gzip uses java.util.zip.GZIPInputStream from the standard library.
        val inputStream: InputStream = if (isXz) {
            Log.i(TAG, "Data archive is xz compressed, using XZInputStream (tukaani)...")
            XZInputStream(FileInputStream(tarFile))
        } else if (isGzip) {
            java.util.zip.GZIPInputStream(FileInputStream(tarFile))
        } else {
            FileInputStream(tarFile)
        }

        try {
            inputStream.use { stream ->
                val headerBuf = ByteArray(512)
                while (true) {
                    // Read 512-byte tar header block
                    val bytesRead = readFully(stream, headerBuf)
                    if (bytesRead < 512) break

                    // Two zero blocks = end of archive
                    if (headerBuf.all { it == 0.toByte() }) break

                    // Parse header: name at offset 0 (100 bytes), size at offset 124 (12 bytes, octal)
                    val entryName = String(headerBuf, 0, 100).trim('\u0000', ' ')
                    val sizeOctal = String(headerBuf, 124, 12).trim('\u0000', ' ')
                    val entrySize = try {
                        sizeOctal.toLong(8)  // Octal to decimal
                    } catch (e: Exception) { 0L }

                    if (entryName.endsWith(".1476")) {
                        // This is a star database file — extract it
                        val outName = entryName.substringAfterLast('/')
                        val outFile = File(destDir, outName)

                        FileOutputStream(outFile).use { fos ->
                            val buf = ByteArray(65536)
                            var remaining = entrySize
                            while (remaining > 0) {
                                val toRead = minOf(remaining, buf.size.toLong()).toInt()
                                val n = stream.read(buf, 0, toRead)
                                if (n <= 0) break
                                fos.write(buf, 0, n)
                                remaining -= n
                            }
                        }
                        extracted++
                        extractedFiles = extracted  // Update SSE progress field
                        Log.d(TAG, "Extracted: $outName (${entrySize / 1024}KB)")

                        progressCallback?.onExtractionProgress(dbName, extracted, -1)
                    } else {
                        // Skip non-.1476 entry data
                        var remaining = entrySize
                        val skipBuf = ByteArray(65536)
                        while (remaining > 0) {
                            val toSkip = minOf(remaining, skipBuf.size.toLong()).toInt()
                            val n = stream.read(skipBuf, 0, toSkip)
                            if (n <= 0) break
                            remaining -= n
                        }
                    }

                    // tar entries are padded to 512-byte boundaries
                    val padding = (512 - (entrySize % 512)) % 512
                    if (padding > 0) {
                        val skipBuf = ByteArray(padding.toInt())
                        readFully(stream, skipBuf)
                    }
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Tar extraction error: ${e.message}")
        }

        return extracted
    }

    /**
     * Extract .1476 files from a zip archive.
     *
     * This handles the case where the database is provided as a .zip file
     * (some alternative download mirrors use this format).
     */
    private fun extractFromZip(zipFile: File, destDir: File, dbName: String): Int {
        var extracted = 0

        ZipInputStream(FileInputStream(zipFile)).use { zis ->
            var entry = zis.nextEntry
            while (entry != null) {
                if (!entry.isDirectory && entry.name.endsWith(".1476")) {
                    val outName = entry.name.substringAfterLast('/')
                    val outFile = File(destDir, outName)

                    FileOutputStream(outFile).use { fos ->
                        zis.copyTo(fos, 65536)
                    }
                    extracted++
                    extractedFiles = extracted  // Update SSE progress field
                    Log.d(TAG, "Extracted from zip: $outName")
                    progressCallback?.onExtractionProgress(dbName, extracted, -1)
                }
                zis.closeEntry()
                entry = zis.nextEntry
            }
        }

        return extracted
    }

    /**
     * Last-resort extraction: decompress a raw XZ file using the Tukaani library.
     *
     * This is called when the structured ar/tar parsing fails but the file
     * might still be a raw .tar.xz or .xz archive.  Decompresses with
     * XZInputStream, writes a .tar file, then delegates to extractFromTar().
     *
     * Uses org.tukaani:xz pure Java library (no native binaries needed).
     */
    private fun extractByRawScan(file: File, destDir: File, dbName: String): Int {
        try {
            // Decompress XZ to a plain .tar file using the Tukaani Java library
            val tarFile = File(destDir, "data.tar")
            Log.i(TAG, "Raw scan: decompressing ${file.name} with XZInputStream...")

            XZInputStream(FileInputStream(file)).use { xzIn ->
                FileOutputStream(tarFile).use { tarOut ->
                    xzIn.copyTo(tarOut, 65536)
                }
            }

            if (tarFile.exists() && tarFile.length() > 0) {
                val extracted = extractFromTar(tarFile, destDir, dbName)
                tarFile.delete()
                return extracted
            }
            tarFile.delete()
        } catch (e: Exception) {
            Log.w(TAG, "XZ decompression failed: ${e.message}")
        }

        Log.w(TAG, "Could not extract database from archive format")
        return 0
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Utility: read exactly N bytes from a stream
    // ═════════════════════════════════════════════════════════════════════

    /**
     * Read exactly `buf.size` bytes from a stream, handling partial reads.
     *
     * InputStream.read() may return fewer bytes than requested even when
     * more data is available (due to buffering, network packets, etc.).
     * This method loops until the buffer is full or the stream ends.
     *
     * @return Number of bytes actually read (may be less than buf.size at EOF)
     */
    private fun readFully(stream: InputStream, buf: ByteArray): Int {
        var total = 0
        while (total < buf.size) {
            val n = stream.read(buf, total, buf.size - total)
            if (n <= 0) break
            total += n
        }
        return total
    }
}
