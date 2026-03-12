# TrackWise AltAz Pro - Development Info

**Last updated:** 2026-03-12
**Author:** Craciun Bogdan
**Repo:** https://github.com/christm45/TrackWise-AltAz-Pro-v1.0

---

## Goal

Build a telescope control Android app (TrackWise-AltAzPro) that supports multiple mount protocols (OnStep/LX200, NexStar/SynScan, iOptron, Meade AudioStar, ASCOM Alpaca, INDI) with:

1. **Adaptive UI**: GUI dynamically hides/shows panels based on the connected controller type (e.g., hide OnStep-specific panels when connected to a Sky-Watcher mount).
2. **Feature parity for NexStar/SynScan protocol**: Add missing features from the SynScan App Pro to the NexStar protocol implementation, bringing it closer to feature parity with OnStep support.

---

## Architecture Overview

### Layers

- **Protocol layer** (`mount_protocol.py`): Abstract `MountProtocol` base class with 6 implementations: LX200, NexStar, iOptron, MeadeAudioStar, ASCOMAlpaca, INDIClient. Protocol registry at line ~4056.
- **Bridge layer** (`telescope_bridge.py`): Handles serial/TCP connections, stores `is_onstep` flag (line 139). Android variant in `android_bridge/main.py`.
- **Application layer** (`HEADLESS_SERVER.py`): Main app class `HeadlessTelescopeApp`. Feature probing happens in `_tick()` method (lines 524-625). Post-connection burst probes firmware, backlash, limits, auxiliary features on first tick. Periodic polls every 5/10 ticks for status, focuser, rotator.
- **Web UI** (`web_server.py`): Flask-based, single-file HTML/CSS/JS embedded in Python. State collected in `_collect_state()` (line 350), polled by frontend `pollStatus()` JS function at 1 Hz.
- **Simulator** (`telescope_simulator.py`): Has `is_onstep = True` (line 254), pretends to be OnStep.

### Build System

- Android Studio with Chaquopy (embedded Python 3.10)
- Gradle 8.5, JDK 17+, Android SDK 34
- Build command: `powershell -Command "& { Set-Location 'C:\Users\Bogdan\Desktop\android app\android app'; .\gradlew.bat assembleDebug 2>&1 }"`
- Clean build: replace `assembleDebug` with `clean assembleDebug`
- APK output: `app/build/outputs/apk/debug/app-debug.apk`
- Device install: `powershell -Command "adb install -r 'app\build\outputs\apk\debug\app-debug.apk' 2>&1"`
- Push files to device: `powershell -Command "adb push '<file>' '/sdcard/Download/<name>' 2>&1"`

### Device

- Connected device ID: `RF8M20K2G0K`
- App package: `com.telescopecontroller`

---

## Key Files

| File | Lines | Description |
|---|---|---|
| `app/src/main/python/mount_protocol.py` | ~4085 | All 6 protocol implementations. NexStarMountProtocol at lines 1593+. Base ABC at lines 126-437. PROTOCOL_REGISTRY at line 4056. |
| `app/src/main/python/web_server.py` | ~10,300 | Flask server + entire SPA UI (HTML/CSS/JS). `_collect_state()` at line 350. Adaptive UI JS in `pollStatus()` at lines 5933+. API routes at lines 1668+. |
| `app/src/main/python/HEADLESS_SERVER.py` | ~2,300 | Main app class. `_tick()` at line 524, feature probing burst at lines 553-561. HeadlessVar state at lines 297-340. NexStar methods at lines 1694+. |
| `app/src/main/python/telescope_bridge.py` | ~1,916 | Bridge with `is_onstep` flag (line 139), OnStep detection (lines 677-707). |
| `app/src/main/python/telescope_simulator.py` | ~889 | Simulator with `is_onstep = True` (line 254). |
| `app/src/main/python/android_bridge/main.py` | ~1,011 | Android bridge, `is_onstep` set at line 377. |
| `TelescopeController_UserGuide.html` | ~5,900+ | Complete user guide (HTML, bundled in APK). Chapter 15b = NexStar features. |
| `app/src/main/python/TelescopeController_UserGuide.html` | same | Copy bundled inside APK (must be kept in sync with root copy). |
| `README.md` | ~209 | GitHub readme with feature list, install instructions, TOC. |
| `Notice_SynScanAPP-min.pdf` | 48 pages | SynScan App Pro manual (French). Reference for NexStar features. Not committed to git. |

---

## Protocol Details

### NexStar Protocol (lines 1593-2213 in mount_protocol.py)

- **Connection test**: `K\x55` echo, `V` version, `m` model (lines 1672-1716)
- **Version parsing**: `_parse_version()` handles 2-byte binary and 6-char hex SynScan v3.3+ format (lines 1718-1751)
- **Model identification**: `_identify_model()` maps byte codes to Celestron/Sky-Watcher models (lines 1753-1789)
- **Passthrough command** `P` used for slewing, variable-rate tracking, guide rate
- **32-bit precision** for all position/goto (lowercase commands: `e`, `z`, `r`, `b`, `s`)
- **`supports_variable_rate_altaz = True`** - direct Alt/Az rate commands via passthrough

### NexStar Instance State (`__init__`, line 1620)

```python
self._fw_version_str = ""       # cached from V command
self._fw_model_name = ""        # cached from m command
self._fw_model_id = -1          # raw model byte
self._backlash_azm = 0          # arcsec (app-side)
self._backlash_alt = 0          # arcsec (app-side)
self._last_dir_azm = 0          # +1 or -1 for direction tracking
self._last_dir_alt = 0
self._horizon_limit = -5        # min altitude degrees
self._overhead_limit = 90       # max altitude degrees
self._guide_rate = 7.5          # arcsec/sec (0.5x sidereal)
self._speed_comp_ppm = 0.0      # parts per million
self._hibernate_azm = None      # saved azimuth degrees
self._hibernate_alt = None      # saved altitude degrees
```

### Protocol Registry (line 4056)

```python
PROTOCOL_REGISTRY = {
    "lx200": LX200MountProtocol,
    "nexstar": NexStarMountProtocol,
    "ioptron": iOptronMountProtocol,
    "audiostar": MeadeAudioStarMountProtocol,
    "alpaca": ASCOMAlpacaMountProtocol,
    "indi": INDIClientMountProtocol,
}
```

### Feature Discovery Gaps (other protocols)

- **OnStep** is the only protocol with real feature discovery (`:GU#` status flags, `:GXY0#` auxiliary bitmap, implicit `:FG#`/`:rG#` polling)
- **iOptron, AudioStar, ASCOM Alpaca, INDI** all have NO feature discovery beyond model identification
- The SynScan App Pro itself doesn't discover accessories (focuser/rotator) -- it's purely mount control

---

## Accomplished

### 1. Adaptive UI (OnStep panel hiding) -- DONE

Two changes in `web_server.py`:

**Backend** (`_collect_state()`, lines 426-434): Added `is_onstep` and `protocol_name` to the `connection` section of the status JSON response. Uses `_get_active_bridge()` to get the correct bridge.

**Frontend** (`pollStatus()` JS, lines 5933+): Three-tier adaptive UI:
- **OnStep-only panels**: focuser-extended, rotator-card, park-card, mount-pec-card, auxiliary-card, auxiliary-card-loc
- **Shared panels** (OnStep + NexStar): firmware-card, backlash-card, limits-card, tracking-rate-card
- **NexStar-only panels**: nexstar-guide-rate-card, nexstar-hibernate-card, nexstar-speed-comp-card
- **OnStep-only buttons**: btn-init-unpark, btn-init-return-home

### 2. NexStar/SynScan Extended Features -- DONE (all 6)

#### 2a. Firmware Version Display
- **Protocol** (`mount_protocol.py`): `get_firmware_info()` override returns cached firmware data from `test_connection()`. Re-queries `V`/`m` commands if cache is empty.
- **App** (`HEADLESS_SERVER.py`): `_query_firmware_info()` already existed and now works for NexStar.
- **UI** (`web_server.py`): Firmware card (shared panel) shows product name, version, mount type.

#### 2b. App-Side Backlash Compensation
- **Protocol**: `set_backlash()` / `get_backlash()` store per-axis values. `slew()` method overridden to detect direction reversal and inject a corrective motor pulse before the normal slew.
- **Pulse duration**: `backlash_arcsec / (60 * rate)`, capped at 2 seconds.
- **App**: Uses existing `_set_backlash()` / `_get_backlash()` methods which dispatch to protocol.
- **UI**: Backlash card (shared panel) with RA/Azm and Dec/Alt inputs.

#### 2c. Altitude Safety Limits
- **Protocol**: `set_horizon_limit()` / `set_overhead_limit()` / `get_horizon_limit()` / `get_overhead_limit()` store limits. `_check_altitude_limits()` helper. `goto_altaz()` overridden to check limits before sending GoTo -- rejects with error message if out of bounds.
- **Defaults**: Horizon = -5 deg, Overhead = 90 deg.
- **App**: Uses existing `_set_horizon_limit()` / `_set_overhead_limit()` / `_get_limits()`.
- **UI**: Limits card (shared panel) with horizon and overhead inputs.

#### 2d. Autoguide Rate
- **Protocol**: `set_guide_rate(rate_arcsec, send_fn)` sends NexStar passthrough `P` command with byte 0x46 to both motor axes (0x10 AZM, 0x11 ALT). Rate is converted to % of sidereal (1-99).
- **App** (`HEADLESS_SERVER.py`): `_set_guide_rate(rate_arcsec)` method, `guide_rate_var` state variable.
- **API**: `POST /api/mount/guide_rate` with `{"rate": 7.5}`, `GET /api/mount/guide_rate/get`.
- **UI**: Guide Rate card (NexStar-only) with presets: 0.25x (3.75"/s), 0.5x (7.5"/s), 1x (15.04"/s).

#### 2e. Hibernate / Position Save-Restore
- **Protocol**: `hibernate_save(send_fn)` reads current Alt/Az via `z` command and stores. `hibernate_restore(send_fn)` issues `b` GoTo to saved position. `get_hibernate_position()` / `set_hibernate_position()` for persistence.
- **App**: `_hibernate_save()` persists to config (`nexstar.hibernate_azm/alt` keys). `_hibernate_restore()` loads from config if not in memory. `hibernate_status_var` state.
- **API**: `POST /api/mount/hibernate/save`, `POST /api/mount/hibernate/restore`, `GET /api/mount/hibernate/status`.
- **UI**: Hibernate card (NexStar-only) with Save Position / Restore Position buttons.

#### 2f. Speed Compensation (ppm)
- **Protocol**: `set_speed_compensation(ppm, send_fn)` stores ppm correction. `get_speed_compensation()` returns current value. Applied app-side by adjusting effective tracking rate.
- **App**: `_set_speed_compensation(ppm)` method, `speed_comp_ppm_var` state.
- **API**: `POST /api/mount/speed_comp` with `{"ppm": 5.0}`, `GET /api/mount/speed_comp/get`.
- **UI**: Speed Compensation card (NexStar-only) with ppm input.

#### 2g. Tracking Rate Control (bonus)
- **Protocol**: `set_tracking_rate(rate, send_fn)` sends NexStar `T` command. Supports: off (0), sidereal (1), lunar (2), solar (3). `enable_tracking()` / `disable_tracking()` convenience methods.
- **UI**: Tracking Rate card (shared panel, already existed).

### 3. User Guide Updated -- DONE

- Added **Chapter 15b: NexStar/SynScan Extended Features** with 9 sub-sections (firmware, backlash, limits, guide rate, hibernate, speed comp, tracking rate, adaptive UI, API reference).
- Updated both sidebar TOC and inline TOC.
- Updated SynScan-Specific Features section in Chapter 2 with new feature list and cross-reference.
- Synced both copies (root + `app/src/main/python/`).
- Updated `README.md` TOC and supported mounts table.

---

## API Endpoints

### NexStar-Specific (new)

| Endpoint | Method | Body | Description |
|---|---|---|---|
| `/api/mount/guide_rate` | POST | `{"rate": 7.5}` | Set guide rate (arcsec/s) |
| `/api/mount/guide_rate/get` | GET | -- | Get current guide rate |
| `/api/mount/hibernate/save` | POST | -- | Save current position |
| `/api/mount/hibernate/restore` | POST | -- | Restore saved position |
| `/api/mount/hibernate/status` | GET | -- | Get hibernate status |
| `/api/mount/speed_comp` | POST | `{"ppm": 5.0}` | Set speed compensation |
| `/api/mount/speed_comp/get` | GET | -- | Get current ppm value |

### Shared (OnStep + NexStar, already existed)

| Endpoint | Method | Body | Description |
|---|---|---|---|
| `/api/mount/backlash` | POST | `{"axis":"ra","value":30}` | Set backlash |
| `/api/mount/backlash/get` | GET | -- | Get backlash values |
| `/api/mount/limits` | POST | `{"type":"horizon","degrees":-5}` | Set limit |
| `/api/mount/limits/get` | GET | -- | Get limits |
| `/api/mount/firmware` | GET | -- | Get firmware info |
| `/api/mount/firmware/refresh` | POST | -- | Re-query firmware |

### Status Response (`/api/status`)

The status JSON now includes:
- `connection.is_onstep` (bool) - whether connected to OnStep
- `connection.protocol_name` (string) - e.g. "nexstar", "lx200"
- `onstep` (object) - OnStep extended state (firmware, backlash, limits, PEC, focuser, rotator, auxiliary)
- `nexstar` (object) - NexStar-specific state: `guide_rate`, `speed_comp_ppm`, `hibernate_status`

---

## State Variables (HEADLESS_SERVER.py)

### OnStep Extended (lines 297-339)

```python
self.park_state_var = HeadlessVar("Unknown")
self.tracking_rate_var = HeadlessVar("Sidereal")
self.tracking_enabled_var = HeadlessVar(False)
self.mount_pec_status_var = HeadlessVar("--")
self.mount_pec_recorded_var = HeadlessVar(False)
self.firmware_name_var = HeadlessVar("--")
self.firmware_version_var = HeadlessVar("--")
self.firmware_mount_type_var = HeadlessVar("--")
self.backlash_ra_var = HeadlessVar("--")
self.backlash_dec_var = HeadlessVar("--")
self.horizon_limit_var = HeadlessVar("--")
self.overhead_limit_var = HeadlessVar("--")
self._auxiliary_features = []
self.focuser_target_var = HeadlessVar("--")
self.focuser_temperature_var = HeadlessVar("--")
self.focuser_tcf_var = HeadlessVar(False)
self.focuser_selected_var = HeadlessVar("1")
self.rotator_angle_var = HeadlessVar("--")
self.rotator_status_var = HeadlessVar("Stopped")
self.rotator_derotating_var = HeadlessVar(False)
```

### NexStar-Specific (line 340)

```python
self.guide_rate_var = HeadlessVar("7.5")
self.speed_comp_ppm_var = HeadlessVar("0.0")
self.hibernate_status_var = HeadlessVar("No saved position")
```

---

## Commit History

```
e90e33a Add NexStar/SynScan extended features: backlash, altitude limits, guide rate, hibernate, speed compensation, firmware display
4bd90d6 Update README.md
7ec71b1 Update README.md
c33901d Add README with project overview, features, and user guide table of contents
5bef71e Remove meridian/pier side (Alt-Az only), move STOP button to bottom-left, fix all LSP type errors, add Az to status strip, update User Guide v2.0
b1b0773 Initial commit: TrackWise-AltAzPro telescope controller
```

---

## Known LSP Warnings (pre-existing, safe to ignore)

- `Argument of type "bool | Literal['']"` errors in LX200/iOptron protocols -- caused by `resp and resp.endswith('#')` pattern used as `success` arg
- `Method declaration "set_backlash" is obscured` in LX200 -- duplicate method definitions (overloaded signatures)
- `Cannot access attribute "X" for class "MountProtocol"` in HEADLESS_SERVER -- accessing subclass-specific methods via `hasattr()` guards (correct at runtime)
- `"cv2" is possibly unbound` / `"np" is possibly unbound` in web_server -- optional imports guarded by try/except

---

## Not Yet Done / Future Work

- **NexStar accessory discovery**: NexStar protocol supports passthrough to device IDs (0x12=Focuser, 0xB0=GPS, 0xB2=RTC) but discovery is not implemented
- **NexStar PEC**: Not relevant for Alt-Az mounts (PPEC is for equatorial)
- **iOptron extended features**: No feature discovery beyond model identification
- **AudioStar extended features**: No feature discovery
- **ASCOM Alpaca extended features**: Could query ASCOM capabilities via REST
- **INDI extended features**: Could query INDI property vectors
- **Speed compensation integration with tracking controller**: The ppm value is stored but the tracking controller (`realtime_tracking.py`) doesn't yet read it when computing correction rates
- **Hibernate auto-restore on connect**: Could automatically restore position if saved data exists and protocol is NexStar
