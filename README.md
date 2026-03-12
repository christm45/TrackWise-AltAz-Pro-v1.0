ANDROID APK APPLICATION AND USER GUIDE HERE:
https://mega.nz/folder/2oogUbYY#ecUwWwXyywuhZNYOKDAr9Q

# TrackWise AltAz Pro

**Advanced Android telescope controller for Alt-Azimuth mounts**

*By Craciun Bogdan*

---

## Overview

TrackWise AltAz Pro is a full-featured telescope controller Android app designed exclusively for **Alt-Azimuth mounts**. It communicates with your mount over WiFi or USB serial using standard telescope protocols, and provides a rich browser-style UI served from an embedded web server.

### Supported Mounts

| Mount / Protocol | Connection |
|---|---|
| **OnStep / OnStepX** | WiFi (TCP) |
| **SynScan / Sky-Watcher** | WiFi (TCP) + Extended Features |
| **Celestron NexStar** | WiFi / USB Serial + Extended Features |
| **Meade LX200-compatible** | WiFi / USB Serial |

### Architecture

```
Android App (Kotlin)
  +-- Chaquopy (embedded Python 3.10)
       +-- Flask web server on localhost:8080
       +-- Mount protocol handlers
       +-- Tracking engine, Kalman filter, ML predictor
       +-- Plate solving integration (ASTAP / Astrometry.net)
  +-- WebView (Single Page Application)
       +-- Full UI (~10,000+ lines HTML/CSS/JS)
       +-- Real-time telemetry charts (Chart.js)
       +-- Interactive sky chart
```

---

## Features

### Telescope Control
- **GoTo** with 90,000+ object catalog (Messier, NGC, IC, Caldwell, Herschel 400, double stars, solar system)
- **Directional slew** with adjustable speed (Guide / Center / Move / Slew)
- **Fullscreen slew mode** for blind operation at the eyepiece
- **Park / Home / Emergency Stop**
- **Focuser and field derotator control**

### Tracking System
- **5 Hz closed-loop tracking** with real-time corrections
- **Kalman filter** (45% weight) for optimal state estimation
- **ML drift predictor** (55% weight) using gradient-boosted trees
- **Periodic Error Correction (PEC)** with auto-learning
- **Environmental corrections** (temperature, pressure, humidity via weather API)
- **Flexure model** with sky-map learning

### Plate Solving
- **ASTAP local solver** (offline, works on Raspberry Pi)
- **Astrometry.net cloud solver** (API key based)
- **Phone camera sensor detection** for automatic FOV calculation
- **Pixel size calculator** for manual optics configuration

### Camera & Live View
- ZWO ASI camera support (native SDK)
- UVC camera support (via OpenCV)
- ASCOM camera support (Windows)
- Phone camera sensor info (Android Camera2 API)
- MJPEG streaming with snapshot capture

### Sky Chart
- Interactive planetarium with real-time telescope position overlay
- Tap-to-GoTo from the chart
- Constellation lines, deep sky objects, planets

### Telemetry & Monitoring
- Real-time tracking error chart (RA/Dec RMS)
- Kalman filter state visualization
- ML prediction confidence display
- PEC training progress
- Session recording and CSV export

### Auto-Alignment
- Multi-star alignment workflow
- Plate-solve assisted alignment
- Pointing model with flexure corrections

### UI & Accessibility
- Bottom tab bar navigation (Control, Imaging, Tracking, Settings)
- Night vision (red) mode for dark adaptation
- Light / dark theme toggle
- Large text and high contrast modes
- Persistent status strip (RA | Dec | Alt | Az | Tracking | RMS)
- Emergency STOP button (always visible)
- Screen wake lock
- Responsive two-column layout

### Deployment Options
- **Android phone/tablet** (primary)
- **Raspberry Pi** with headless server mode
- **Desktop** (Windows/Linux) via Python

---

## Installation

### Android (Primary)

1. Download the APK from the [Releases](https://github.com/christm45/TrackWise-AltAz-Pro-v1.0/releases) page
2. Enable "Install from unknown sources" in Android settings
3. Install the APK
4. Launch TrackWise AltAz Pro
5. Connect to your mount's WiFi network
6. Enter the mount's IP address and port, then tap Connect

### Building from Source

**Requirements:**
- Android Studio (Hedgehog or later)
- JDK 17+
- Android SDK 34
- Chaquopy Gradle plugin (configured in `build.gradle.kts`)

**Build:**
```bash
git clone https://github.com/christm45/TrackWise-AltAz-Pro-v1.0.git
cd TrackWise-AltAz-Pro-v1.0
./gradlew assembleDebug
```

The APK will be at `app/build/outputs/apk/debug/app-debug.apk`.

### Raspberry Pi

See Chapter 14 of the [User Guide](TelescopeController_UserGuide.html) for headless deployment on Raspberry Pi.

---

## User Guide

The complete user guide is included in this repository:

**[TelescopeController_UserGuide.html](TelescopeController_UserGuide.html)**

### Table of Contents

1. **Installation & First Launch**
2. **Connecting to Your Telescope** - WiFi/USB setup for OnStep, SynScan, NexStar, LX200
3. **Understanding the Position Display** - Alt/Az, RA/Dec coordinates
4. **GoTo & Catalog Navigation** - 90,000+ objects, search, browse, solar system ephemeris
5. **Telescope Controls** - Slew, speed, focuser, derotator, park, keyboard shortcuts
6. **Telemetry Tab** - Tracking error charts, Kalman filter, ML predictor, PEC
7. **Plate Solving** - ASTAP, Astrometry.net, FOV configuration
8. **Tracking System** - 5 Hz control loop, correction pipeline, rate delivery
9. **Camera & Live View** - ASI, UVC, ASCOM, MJPEG streaming
10. **Sky Chart** - Interactive planetarium with GoTo integration
11. **Log Tab** - Event monitoring and troubleshooting
12. **Location & Weather** - GPS, weather service, atmospheric corrections
13. **Auto-Alignment System** - Multi-star alignment, pointing model
14. **Session Recording & Export** - CSV export, crash recovery
15. **Raspberry Pi Deployment** - Headless server setup
16. **OnStep Extended Features** - Auxiliary ports, interval control
16b. **NexStar/SynScan Extended Features** - Backlash compensation, altitude limits, guide rate, hibernate, speed compensation, tracking rate
17. **Troubleshooting** - Common issues and solutions
18. **Hardware Compatibility** - Tested mounts, cameras, platforms
19. **UI & Accessibility** - Night mode, themes, large text, high contrast

---

## Project Structure

```
app/src/main/
  java/com/telescopecontroller/
    MainActivity.kt              # Single activity, WebView host
    service/TelescopeService.kt  # Foreground service
    camera/CameraManager.kt      # Android Camera2 integration
    camera/ASICameraSDK.kt       # ZWO ASI native wrapper
  python/
    web_server.py                # Flask server + entire SPA UI (~10,000 lines)
    mount_protocol.py            # LX200/OnStep/SynScan/NexStar protocols
    HEADLESS_SERVER.py           # Raspberry Pi headless mode
    config_manager.py            # Persistent configuration
    auto_platesolve.py           # Plate solving engine
    catalog_loader.py            # Object catalog (90,000+ objects)
    chart.umd.min.js             # Bundled Chart.js v4.4.7
    android_bridge/
      camera_bridge.py           # Python<->Kotlin camera bridge
      cloud_solver.py            # Astrometry.net API client
    TelescopeController_UserGuide.html  # Bundled in APK
```

---

## License

All rights reserved. Copyright (c) 2026 Craciun Bogdan.

---

## Contact

For questions, bug reports, or feature requests, please open an [Issue](https://github.com/christm45/TrackWise-AltAz-Pro-v1.0/issues).
