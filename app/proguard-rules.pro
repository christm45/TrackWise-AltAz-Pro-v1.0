# Telescope Controller ProGuard rules
# NOTE: minifyEnabled is false because Chaquopy's Python code
# breaks with R8/ProGuard.  These rules are here for future use
# if selective minification is enabled.

# Keep Chaquopy Python bridge classes
-keep class com.chaquo.python.** { *; }
-keep class com.telescopecontroller.** { *; }

# Keep USB serial driver classes (reflection-based detection)
-keep class com.hoho.android.usbserial.driver.** { *; }
