package com.jiangdg.utils;

import android.app.Application;
import android.util.Log;

/**
 * Simplified XLog wrapper -- uses android.util.Log directly.
 *
 * Replaces the original AUSBC XLogWrapper which depended on
 * com.elvishew:xlog.  We don't need file logging for the UVC
 * camera -- Android logcat is sufficient.
 *
 * Original author: jiangdg (2022/7/19)
 * Simplified for TelescopeController.
 */
public class XLogWrapper {
    private static final String TAG = "UVCCamera";

    public static void init(Application application, String folderPath) {
        // No-op: using android.util.Log directly
    }

    public static void v(String tag, String msg) {
        Log.v(tag, msg != null ? msg : "");
    }

    public static void i(String tag, String msg) {
        Log.i(tag, msg != null ? msg : "");
    }

    public static void d(String tag, String msg) {
        Log.d(tag, msg != null ? msg : "");
    }

    public static void w(String tag, String msg) {
        Log.w(tag, msg != null ? msg : "");
    }

    public static void w(String tag, String msg, Throwable throwable) {
        Log.w(tag, msg != null ? msg : "", throwable);
    }

    public static void w(String tag, Throwable throwable) {
        Log.w(tag, "", throwable);
    }

    public static void e(String tag, String msg) {
        Log.e(tag, msg != null ? msg : "");
    }

    public static void e(String tag, String msg, Throwable throwable) {
        Log.e(tag, msg != null ? msg : "", throwable);
    }
}
