package com.telescopecontroller.network

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.util.Log
import java.io.BufferedReader
import java.io.ByteArrayOutputStream
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Makes HTTP GET requests over the cellular (4G/5G) network, even when
 * WiFi is connected to a local-only network (e.g. telescope hotspot).
 *
 * Android routes all traffic through WiFi by default when WiFi is connected.
 * This class uses ConnectivityManager.requestNetwork() to obtain a reference
 * to the cellular network and binds HTTP connections to it explicitly.
 *
 * Usage from Python (via Chaquopy):
 *     client = CellularHttpClient(context)
 *     json_string = client.get("https://api.open-meteo.com/v1/forecast?lat=48&lon=2&...")
 */
class CellularHttpClient(context: Context) {

    companion object {
        private const val TAG = "CellularHttp"
        private const val CELLULAR_TIMEOUT_SEC = 10L
    }

    private val connectivityManager =
        context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager

    // Cached cellular network reference
    @Volatile
    private var cellularNetwork: Network? = null

    private val networkCallback = object : ConnectivityManager.NetworkCallback() {
        override fun onAvailable(network: Network) {
            Log.i(TAG, "Cellular network available: $network")
            cellularNetwork = network
        }

        override fun onLost(network: Network) {
            Log.w(TAG, "Cellular network lost: $network")
            if (cellularNetwork == network) {
                cellularNetwork = null
            }
        }
    }

    @Volatile
    private var registered = false

    /**
     * Request access to the cellular network.
     * Call this once at startup; the callback will keep the reference updated.
     */
    fun requestCellularNetwork() {
        if (registered) return
        try {
            val request = NetworkRequest.Builder()
                .addTransportType(NetworkCapabilities.TRANSPORT_CELLULAR)
                .addCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
                .build()
            connectivityManager.requestNetwork(request, networkCallback)
            registered = true
            Log.i(TAG, "Cellular network requested")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to request cellular network: ${e.message}")
        }
    }

    /**
     * Release the cellular network request.
     * Call this when the service is destroyed.
     */
    fun release() {
        if (registered) {
            try {
                connectivityManager.unregisterNetworkCallback(networkCallback)
            } catch (_: Exception) {}
            registered = false
            cellularNetwork = null
        }
    }

    /**
     * Perform an HTTP GET request over the cellular network.
     *
     * @param urlString  Full URL including query parameters.
     * @param timeoutMs  Connection + read timeout in milliseconds.
     * @return Response body as a String, or null on failure.
     *
     * Called from Python via Chaquopy:
     *     result = cellular_client.get("https://...", 5000)
     */
    fun get(urlString: String, timeoutMs: Int = 5000): String? {
        val network = cellularNetwork
        if (network == null) {
            Log.w(TAG, "No cellular network available -- trying to acquire...")
            // Try to get it synchronously with a short wait
            val acquired = acquireCellularSync()
            if (!acquired || cellularNetwork == null) {
                Log.e(TAG, "Could not acquire cellular network")
                return null
            }
        }

        return try {
            val url = URL(urlString)
            // Open connection bound to the cellular network
            val conn = cellularNetwork!!.openConnection(url) as HttpURLConnection
            conn.connectTimeout = timeoutMs
            conn.readTimeout = timeoutMs
            conn.requestMethod = "GET"
            conn.setRequestProperty("User-Agent", "TelescopeController/1.0")

            val code = conn.responseCode
            if (code != 200) {
                Log.w(TAG, "HTTP $code from $urlString")
                conn.disconnect()
                return null
            }

            val reader = BufferedReader(InputStreamReader(conn.inputStream))
            val body = reader.readText()
            reader.close()
            conn.disconnect()

            Log.d(TAG, "GET $urlString -> ${body.length} bytes via cellular")
            body
        } catch (e: Exception) {
            Log.e(TAG, "Cellular HTTP GET failed: ${e.message}")
            null
        }
    }

    /**
     * Perform an HTTP POST request with form-urlencoded body over the cellular network.
     *
     * @param urlString  Full URL.
     * @param formData   Key-value pairs to send as form data.
     * @param timeoutMs  Connection + read timeout in milliseconds.
     * @return Response body as a String, or null on failure.
     */
    fun post(urlString: String, formData: Map<String, String>, timeoutMs: Int = 10000): String? {
        val network = cellularNetwork
        if (network == null) {
            Log.w(TAG, "No cellular network available -- trying to acquire...")
            val acquired = acquireCellularSync()
            if (!acquired || cellularNetwork == null) {
                Log.e(TAG, "Could not acquire cellular network")
                return null
            }
        }

        return try {
            val url = URL(urlString)
            val conn = cellularNetwork!!.openConnection(url) as HttpURLConnection
            conn.connectTimeout = timeoutMs
            conn.readTimeout = timeoutMs
            conn.requestMethod = "POST"
            conn.doOutput = true
            conn.setRequestProperty("User-Agent", "TelescopeController/1.0")
            conn.setRequestProperty("Content-Type", "application/x-www-form-urlencoded")

            // Build URL-encoded form body
            val body = formData.entries.joinToString("&") { (k, v) ->
                "${URLEncoder.encode(k, "UTF-8")}=${URLEncoder.encode(v, "UTF-8")}"
            }

            val writer = OutputStreamWriter(conn.outputStream)
            writer.write(body)
            writer.flush()
            writer.close()

            val code = conn.responseCode
            if (code !in 200..299) {
                Log.w(TAG, "HTTP $code from POST $urlString")
                conn.disconnect()
                return null
            }

            val reader = BufferedReader(InputStreamReader(conn.inputStream))
            val responseBody = reader.readText()
            reader.close()
            conn.disconnect()

            Log.d(TAG, "POST $urlString -> ${responseBody.length} bytes via cellular")
            responseBody
        } catch (e: Exception) {
            Log.e(TAG, "Cellular HTTP POST failed: ${e.message}")
            null
        }
    }

    /**
     * Perform an HTTP POST request with multipart/form-data body over the cellular network.
     * Supports uploading a binary file alongside text form fields.
     *
     * @param urlString    Full URL.
     * @param formFields   Key-value text fields to include in the multipart body.
     * @param fileName     The filename to report in the Content-Disposition header.
     * @param fileField    The form field name for the file part.
     * @param fileBytes    Raw bytes of the file to upload.
     * @param fileMimeType MIME type of the file (e.g. "application/fits").
     * @param timeoutMs    Connection + read timeout in milliseconds.
     * @return Response body as a String, or null on failure.
     */
    fun postMultipart(
        urlString: String,
        formFields: Map<String, String>,
        fileName: String,
        fileField: String,
        fileBytes: ByteArray,
        fileMimeType: String,
        timeoutMs: Int = 30000
    ): String? {
        val network = cellularNetwork
        if (network == null) {
            Log.w(TAG, "No cellular network available -- trying to acquire...")
            val acquired = acquireCellularSync()
            if (!acquired || cellularNetwork == null) {
                Log.e(TAG, "Could not acquire cellular network")
                return null
            }
        }

        val boundary = "----CellularBoundary" + java.util.UUID.randomUUID().toString().replace("-", "").substring(0, 16)
        val crlf = "\r\n"

        return try {
            // Build the multipart body in memory
            val baos = ByteArrayOutputStream()

            // Text form fields
            for ((key, value) in formFields) {
                baos.write("--$boundary$crlf".toByteArray())
                baos.write("Content-Disposition: form-data; name=\"$key\"$crlf".toByteArray())
                baos.write(crlf.toByteArray())
                baos.write(value.toByteArray())
                baos.write(crlf.toByteArray())
            }

            // File part
            baos.write("--$boundary$crlf".toByteArray())
            baos.write("Content-Disposition: form-data; name=\"$fileField\"; filename=\"$fileName\"$crlf".toByteArray())
            baos.write("Content-Type: $fileMimeType$crlf".toByteArray())
            baos.write(crlf.toByteArray())
            baos.write(fileBytes)
            baos.write(crlf.toByteArray())

            // Closing boundary
            baos.write("--$boundary--$crlf".toByteArray())

            val multipartBody = baos.toByteArray()

            val url = URL(urlString)
            val conn = cellularNetwork!!.openConnection(url) as HttpURLConnection
            conn.connectTimeout = timeoutMs
            conn.readTimeout = timeoutMs
            conn.requestMethod = "POST"
            conn.doOutput = true
            conn.setRequestProperty("User-Agent", "TelescopeController/1.0")
            conn.setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
            conn.setRequestProperty("Content-Length", multipartBody.size.toString())
            conn.setFixedLengthStreamingMode(multipartBody.size)

            conn.outputStream.write(multipartBody)
            conn.outputStream.flush()
            conn.outputStream.close()

            val code = conn.responseCode
            if (code !in 200..299) {
                Log.w(TAG, "HTTP $code from multipart POST $urlString")
                conn.disconnect()
                return null
            }

            val reader = BufferedReader(InputStreamReader(conn.inputStream))
            val responseBody = reader.readText()
            reader.close()
            conn.disconnect()

            Log.d(TAG, "Multipart POST $urlString -> ${responseBody.length} bytes via cellular")
            responseBody
        } catch (e: Exception) {
            Log.e(TAG, "Cellular HTTP multipart POST failed: ${e.message}")
            null
        }
    }

    /**
     * Check if the cellular network is currently available.
     */
    fun isAvailable(): Boolean = cellularNetwork != null

    /**
     * Try to acquire cellular network synchronously (blocking, max 5s).
     */
    private fun acquireCellularSync(): Boolean {
        if (cellularNetwork != null) return true

        val latch = CountDownLatch(1)
        val tempCallback = object : ConnectivityManager.NetworkCallback() {
            override fun onAvailable(network: Network) {
                cellularNetwork = network
                latch.countDown()
            }
        }

        return try {
            val request = NetworkRequest.Builder()
                .addTransportType(NetworkCapabilities.TRANSPORT_CELLULAR)
                .addCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
                .build()
            connectivityManager.requestNetwork(request, tempCallback)
            val got = latch.await(CELLULAR_TIMEOUT_SEC, TimeUnit.SECONDS)
            try {
                connectivityManager.unregisterNetworkCallback(tempCallback)
            } catch (_: Exception) {}
            got
        } catch (e: Exception) {
            Log.e(TAG, "acquireCellularSync failed: ${e.message}")
            false
        }
    }
}
