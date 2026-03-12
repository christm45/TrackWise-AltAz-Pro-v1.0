"""
Configuration Manager -- JSON-based persistent application settings.

Architecture Role:
    Provides centralized configuration storage for all application modules.
    Settings are persisted as a JSON file (telescope_config.json) and loaded
    at startup with automatic migration: new default keys are added, but
    existing user values are preserved.

    Data Flow:
        telescope_config.json <--> ConfigManager <--> all other modules
        (on disk)                  (in memory)        (via get/set)

    Key consumers:
        - main_realtime.py: loads UI prefs, telescope connection params, ASTAP path
        - realtime_tracking.py: tracking intervals, component enable flags
        - auto_platesolve.py: camera/ASCOM settings, solve parameters
        - telescope_bridge.py: telescope communication

Classes:
    ConfigManager -- Load, save, get, set configuration values with dot-path keys.

Configuration Sections:
    location    -- GPS coordinates (latitude, longitude, UTC offset)
    telescope   -- Connection type (USB/WiFi), port, baud rate, IP, TCP port
    astap       -- ASTAP solver path, interval, downsample, search radius, timeout
    auto_solve  -- Plate solving mode, camera index, watch folder, save settings
    ascom_camera-- ASCOM camera ID, exposure, gain, binning
    tracking    -- Plate solve interval, correction interval, component enable flags
    ui          -- Window dimensions, theme
    logging     -- File logging settings, max size, backup count

Dependencies:
    - json: Config serialization.
    - pathlib: Cross-platform file path handling.
    - Used by: main_realtime.py (primary consumer).
"""

import json
import os
import tempfile
import threading
import time
from typing import Dict, Any, Optional
from pathlib import Path

from telescope_logger import get_logger

_logger = get_logger(__name__)

# ── Default observer location (Paris, France) ─────────────────────────
# Used across the entire project when no GPS fix or manual entry is
# available.  Import these constants instead of hardcoding the values.
DEFAULT_LATITUDE = 48.8566
DEFAULT_LONGITUDE = 2.3522


class ConfigManager:
    """JSON configuration manager with dot-path key access.

    Handles loading, saving, and accessing application settings stored in
    a JSON file. Supports nested key access via dot notation
    (e.g., "telescope.usb_port") and automatic merging with defaults when
    new configuration keys are added in code updates.

    Merge strategy (_merge_config):
        - Keys present in both default and loaded: loaded value wins.
        - Keys only in default (new keys): default value is added.
        - Keys only in loaded (removed keys): preserved (no data loss).
        - Nested dicts are merged recursively.

    Typical usage:
        config = ConfigManager("telescope_config.json")
        port = config.get("telescope.usb_port", "COM1")
        config.set("telescope.usb_port", "COM4")
        config.save_config()
    """

    def __init__(self, config_file: str = "telescope_config.json"):
        """Initialize the configuration manager.

        Loads the config file if it exists, or creates one with defaults.
        Merges loaded config with defaults to pick up any new keys added
        in code updates.

        Args:
            config_file: Path to the JSON configuration file.
        """
        self.config_file = Path(config_file).resolve()
        self.config: Dict[str, Any] = {}
        self._save_lock = threading.Lock()
        self.load_config()

    def get_default_config(self) -> Dict[str, Any]:
        """Return the full default configuration dictionary.

        This serves as the template for all available settings. When a new
        setting is needed, add it here with a sensible default value.
        The merge logic ensures existing user configs pick up new keys
        without losing their customized values.

        Returns:
            Complete default configuration dictionary.
        """
        return {
            # Geographic location for coordinate conversions (Alt/Az <-> RA/Dec)
            "location": {
                "latitude": DEFAULT_LATITUDE,
                "longitude": DEFAULT_LONGITUDE,
                "utc_offset": 1          # Hours from UTC
            },

            # Telescope connection settings
            "telescope": {
                "connection_type": "USB",  # "USB" or "WiFi"
                "usb_port": "COM1",        # Serial port name
                "usb_baudrate": 9600,      # Serial baud rate (OnStep default)
                "wifi_ip": "192.168.0.1",  # TCP/IP address for WiFi mode
                "wifi_port": 9996          # TCP port for WiFi mode (persistent channel)
            },

            # ASTAP plate solver settings
            "astap": {
                "path": r"C:\Program Files\astap\astap.exe",
                "solve_interval": 4.0,    # Seconds between solves
                "downsample": 8,          # Image downsampling factor
                "search_radius": 10.0,    # Search radius in degrees
                "timeout": 5.0            # Max solve time in seconds
            },

            # Plate solver configuration (mode, FOV, cloud settings)
            "solver": {
                "mode": "auto",             # auto, astap, cloud
                "cloud_api_key": "",        # Astrometry.net API key (optional)
                "timeout": 120,             # Max solve time in seconds
                "focal_length_mm": 0,       # Telescope focal length in mm (0 = not set)
                "sensor_width_mm": 0,       # Camera sensor width in mm (0 = not set)
            },

            # Automatic plate solving mode
            "auto_solve": {
                "mode": "manual",          # manual, camera, ascom, folder
                "camera_index": 0,         # OpenCV camera index
                "watch_folder": "",        # Folder to watch for new images
                "save_images": False,      # Save captured images
                "save_folder": "",         # Where to save images
                "save_format": "fits"      # Image format: fits, png, jpg
            },

            # ASCOM camera settings (for direct camera control via COM)
            "ascom_camera": {
                "camera_id": "",
                "camera_name": "No camera selected",
                "exposure": 0.5,           # Exposure time in seconds
                "gain": 100,               # Camera gain
                "binning": 2               # Pixel binning factor
            },

            # Mount drive system configuration
            "mount": {
                "drive_type": "planetary_gearbox",  # worm_gear, planetary_gearbox, harmonic_drive, belt_drive, direct_drive
                "pec_enabled": True,                # Enable software PEC (periodic error correction)
                "pec_learning": True,               # Allow PEC to learn new periodic patterns
                "flexure_learning": True,           # Enable position-dependent flexure learning
                "flexure_model_path": "flexure_model.json",  # Persistence file for flexure map
            },

            # Real-time tracking pipeline settings
            "tracking": {
                "plate_solve_interval": 4.0,   # Seconds between plate solves
                "correction_interval": 0.2,    # Seconds between correction updates (5 Hz)
                "enable_kalman": True,         # Enable Kalman filter component
                "enable_ml": True,             # Enable ML drift predictor component
            },

            # User interface settings
            "ui": {
                "window_width": 1100,
                "window_height": 750,
                "theme": "dark"
            },

            # Application logging
            "logging": {
                "file_enabled": True,
                "file_path": "telescope_app.log",
                "max_file_size_mb": 10,
                "backup_count": 5
            }
        }

    def load_config(self) -> bool:
        """Load configuration from the JSON file.

        If the file exists, loads it and merges with defaults (to pick up
        new keys). If not, creates the file with default values.

        Returns:
            True if loaded successfully, False on error (defaults used).
        """
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults to add any new keys from code updates
                    default_config = self.get_default_config()
                    self.config = self._merge_config(default_config, loaded_config)
                    # --- One-time migrations ---
                    self._apply_migrations()
                    return True
            else:
                # First run: create config file with defaults
                self.config = self.get_default_config()
                self.save_config()
                return True
        except Exception as e:
            _logger.warning("Config load error: %s", e)
            # Fall back to defaults on any error (corrupted JSON, permission, etc.)
            self.config = self.get_default_config()
            return False

    def _apply_migrations(self) -> None:
        """Apply one-time config migrations for breaking changes.

        Called after merging loaded config with defaults.  Each migration
        checks a condition and, if matched, updates the value and saves.
        """
        dirty = False

        # Migration: port 9999 → 9996 (SmartWebServer persistent channel)
        # Port 9999 has a hard 1-second connection timeout in the ESP
        # firmware.  Port 9996 is the persistent channel (10 s, extends
        # on activity).  Users who saved 9999 need automatic migration.
        # Check both config paths (telescope.* and connection.*).
        for section in ("telescope", "connection"):
            key = f"{section}.wifi_port"
            val = self.get(key)
            if val == 9999 or val == "9999":
                _logger.info("Migrating %s 9999 -> 9996 "
                             "(persistent command channel)", key)
                self.config.setdefault(section, {})["wifi_port"] = 9996
                dirty = True

        if dirty:
            self.save_config()

    def save_config(self) -> bool:
        """Save current configuration to the JSON file.

        Uses atomic write (temp file + rename) with retry logic to handle
        Windows file-locking issues (antivirus, concurrent access).
        A threading lock prevents multiple threads from saving at the
        same time.

        Returns:
            True if saved successfully, False on error.
        """
        if not self._save_lock.acquire(timeout=5):
            _logger.warning("Config save skipped: lock timeout")
            return False
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)

            # Serialize once
            data = json.dumps(self.config, indent=2, ensure_ascii=False)

            # Atomic write: write to temp file in same directory, then rename
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    fd, tmp_path = tempfile.mkstemp(
                        suffix=".tmp",
                        dir=str(self.config_file.parent),
                    )
                    try:
                        os.write(fd, data.encode("utf-8"))
                    finally:
                        os.close(fd)

                    # On Windows, os.replace can fail if the target is locked
                    # by antivirus or another reader. Retry after a short wait.
                    os.replace(tmp_path, str(self.config_file))
                    return True

                except PermissionError:
                    # Clean up temp file if rename failed
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    if attempt < max_attempts - 1:
                        time.sleep(0.2 * (attempt + 1))
                    else:
                        raise
            return False  # unreachable but satisfies type checker
        except Exception as e:
            _logger.warning("Config save error: %s", e)
            return False
        finally:
            self._save_lock.release()

    def _merge_config(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge loaded config into defaults.

        Strategy:
            - Start with a copy of the default config.
            - For each key in loaded config:
              - If both default and loaded have a dict at that key: recurse.
              - Otherwise: loaded value overwrites default.
            - Keys only in default are preserved (new settings get defaults).
            - Keys only in loaded are preserved (no data loss on downgrade).

        Args:
            default: Default configuration dictionary.
            loaded:  User's saved configuration dictionary.

        Returns:
            Merged configuration dictionary.
        """
        result = default.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result

    def get(self, key_path: str, default: Any = None) -> Any:
        """Retrieve a configuration value using dot-path notation.

        Example: config.get("telescope.usb_port") navigates to
        config["telescope"]["usb_port"].

        Args:
            key_path: Dot-separated key path (e.g., "telescope.usb_port").
            default:  Value to return if the key path doesn't exist.

        Returns:
            The configuration value, or default if not found.
        """
        keys = key_path.split('.')
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any) -> bool:
        """Set a configuration value using dot-path notation.

        Creates intermediate dictionaries as needed. Does NOT auto-save;
        call save_config() afterward to persist.

        Args:
            key_path: Dot-separated key path (e.g., "telescope.usb_port").
            value:    Value to set.

        Returns:
            True if set successfully, False on error.
        """
        keys = key_path.split('.')
        config = self.config
        try:
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
            config[keys[-1]] = value
            return True
        except Exception as e:
            _logger.warning("Config set error for %s: %s", key_path, e)
            return False

    def save_section(self, section: str, values: Dict[str, Any]) -> bool:
        """Update an entire configuration section and save to disk.

        Convenience method for updating multiple keys at once (e.g., all
        telescope connection settings after a dialog).

        Args:
            section: Top-level section name (e.g., "telescope").
            values:  Dictionary of key-value pairs to merge into the section.

        Returns:
            True if saved successfully, False on error.
        """
        try:
            if section not in self.config:
                self.config[section] = {}
            self.config[section].update(values)
            return self.save_config()
        except Exception as e:
            _logger.warning("Config section save error for %s: %s", section, e)
            return False
