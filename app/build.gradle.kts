import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.chaquo.python")
}

android {
    namespace = "com.telescopecontroller"
    compileSdk = 34

    signingConfigs {
        create("release") {
            val localPropsFile = project.rootProject.file("local.properties")
            val localProps = Properties()
            if (localPropsFile.exists()) {
                localPropsFile.inputStream().use { localProps.load(it) }
            }
            storeFile = file(localProps.getProperty("RELEASE_STORE_FILE") ?: "../telescope-release.jks")
            storePassword = localProps.getProperty("RELEASE_STORE_PASSWORD") ?: ""
            keyAlias = localProps.getProperty("RELEASE_KEY_ALIAS") ?: "telescope"
            keyPassword = localProps.getProperty("RELEASE_KEY_PASSWORD") ?: ""
        }
    }

    defaultConfig {
        applicationId = "com.telescopecontroller"
        minSdk = 26          // Android 8.0+ (Camera2 mature, USB host stable)
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"

        // Chaquopy: embed Python 3.10 interpreter (required for scipy)
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false   // Python code breaks with R8/ProGuard
            signingConfig = signingConfigs.getByName("release")
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        viewBinding = true
    }

    // Chaquopy's native libraries (libpython3.10.so, libchaquopy_java.so) must be
    // extracted to the filesystem so that dlopen() works reliably on all devices.
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

// ---------------------------------------------------------------------------
//  Chaquopy (embedded Python)
// ---------------------------------------------------------------------------
chaquopy {
    defaultConfig {
        // Python 3.10 required for scipy (Chaquopy has cp310 wheels for scipy)
        // Python 3.11 only has numpy; no scipy/tetra3 support yet.
        version = "3.10"

        // Python packages installed into the APK at build time
        // --ignore-requires-python is needed because the BUILD python (3.14)
        // doesn't satisfy scipy's metadata constraint (<3.11), but the TARGET
        // python (3.10) does.  The package works fine at runtime.
        pip {
            // --ignore-requires-python is needed because the BUILD python
            // (3.14) doesn't satisfy some metadata constraints, but the
            // TARGET python (3.10) does.  Packages work fine at runtime.
            //
            // Note: scipy 1.8.1 declares numpy<1.25, but Chaquopy provides
            // numpy 1.26.2 as its pre-built wheel for cp310.  The version
            // mismatch is cosmetic -- numpy 1.26 is backward-compatible with
            // scipy 1.8.1 at runtime.  We cannot pin numpy to an older
            // version because Chaquopy only ships specific pre-built wheels.
            options("--ignore-requires-python")
            install("numpy")
            install("flask")
            install("requests")
            install("scipy")
            install("Pillow")
        }

        // Root of the Python source tree -- contains our bridge + copies of
        // the main Python modules (HEADLESS_SERVER, web_server, etc.)
        pyc {
            src = false   // Ship .py not .pyc for easier debugging
        }
    }

    sourceSets {
        getByName("main") {
            srcDir("src/main/python")
        }
    }
}

// ---------------------------------------------------------------------------
//  Python sources
// ---------------------------------------------------------------------------
//  All Python modules (shared + android_bridge) live directly in
//  src/main/python/.  Chaquopy bundles them into the APK automatically.
//  No sync from a parent directory is needed -- this is a standalone project.
// ---------------------------------------------------------------------------

val syncCatalogDest = file("src/main/assets/catalogs")

dependencies {
    // AndroidX core
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.lifecycle:lifecycle-service:2.7.0")
    implementation("androidx.webkit:webkit:1.10.0")

    // USB serial (FTDI, CP210x, CH340 -- same chips as telescope cables)
    implementation("com.github.mik3y:usb-serial-for-android:3.7.3")

    // UVC camera (USB webcams + ZWO ASI planetary cameras via UVC protocol)
    // Source: AndroidUSBCamera/libuvc (https://github.com/jiangdongguo/AndroidUSBCamera)
    // Java sources in com.jiangdg.{uvc,usb,utils} + pre-compiled native .so in jniLibs/
    // Provides: USBMonitor (USB device management), UVCCamera (UVC protocol via JNI)
    // No external dependency needed -- source + native libs are included directly.

    // ASTAP plate solver: libastapcli.so (native ELF binary) in jniLibs/
    // Source: https://sourceforge.net/projects/astap-program/files/android/
    // Not a shared library -- it's a standalone executable invoked via ProcessBuilder.
    // Star databases (D05/D20/D50/W08) are downloaded at runtime to externalFilesDir.
    // Kotlin wrappers: AstapSolver.kt (execution + WCS parsing), AstapDatabaseManager.kt

    // XZ decompression (pure Java) -- needed to extract ASTAP star databases.
    // ASTAP databases are distributed as .deb files (ar archive → data.tar.xz).
    // Android's stdlib lacks XZ/LZMA2 support, so we use the Tukaani library.
    // Source: https://tukaani.org/xz/java.html  (~130 KB, Apache 2.0 license)
    implementation("org.tukaani:xz:1.9")

    // CameraX (modern Camera2 wrapper for phone camera)
    val cameraxVersion = "1.3.1"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    // ZWO ASI Camera SDK (official Android SDK v1.1)
    // Java API: zwocamera.jar in app/libs/
    // Native libs: libASICamera2.so + libzwo_camera.so in jniLibs/<abi>/
    // Source: https://www.zwoastro.com/software/ -> ASI Camera SDK
    implementation(files("libs/zwocamera.jar"))

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
}
