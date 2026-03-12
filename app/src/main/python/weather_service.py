"""
Weather Service -- Open-Meteo API Integration for Atmospheric Corrections.

Architecture Role:
    Provides real-time weather data (temperature, pressure, humidity, cloud
    cover, wind, dew point) used by tracking_improvements.py to compute
    atmospheric refraction corrections and by main_realtime.py to display
    comprehensive observing conditions.

    Atmospheric refraction bends starlight, causing apparent position shifts
    that increase near the horizon. Accurate temperature and pressure data
    improve the refraction model accuracy. Cloud cover, wind speed, and dew
    point help the observer assess whether conditions are suitable for
    imaging or visual observation.

    Data Flow:
        GPS coordinates --> OpenMeteoService.get_weather() --> WeatherData
                       --> tracking_improvements.atmospheric_refraction_correction()
                       --> applied to Alt coordinate before sending tracking rates
                       --> main_realtime.py weather panel (full conditions display)

    The service uses a 5-minute cache to minimize API calls. Open-Meteo is a
    free API that does not require an API key.

Classes:
    WeatherData      -- Dataclass holding all weather fields.
    OpenMeteoService -- HTTP client for the Open-Meteo forecast API.

Functions:
    calculate_dew_point     -- Magnus formula dew point from temp + humidity.
    assess_observing_conditions -- Qualitative sky rating from weather data.
    get_gps_from_browser    -- Stub for future browser-based GPS location.

Dependencies:
    - requests: HTTP client for API calls.
    - math: dew point calculation.
    - Used by: main_realtime.py (weather display), tracking_improvements.py (refraction).
"""

import math
import os
import requests
import json
import time
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field

from telescope_logger import get_logger

_logger = get_logger(__name__)

def _is_android() -> bool:
    """Lazy Android detection (env var set by android_bridge.main at runtime)."""
    return os.environ.get("TELESCOPE_PLATFORM") == "android"


def _cellular_get(url: str, params: dict, timeout: float = 5.0) -> Optional[requests.Response]:
    """Try to make an HTTP GET via the Android cellular network bridge.

    On Android, when connected to a telescope WiFi hotspot with no internet,
    this routes the request through 4G/5G mobile data instead.

    Returns a fake Response-like object with .json() and .status_code,
    or None if cellular is unavailable or the request fails.
    """
    if not _is_android():
        return None
    try:
        from android_bridge.network_bridge import cellular_get, is_cellular_available
        if not is_cellular_available():
            _logger.debug("Cellular network not available")
            return None

        # Build full URL with query params
        from urllib.parse import urlencode
        full_url = url + "?" + urlencode(params)
        body = cellular_get(full_url, int(timeout * 1000))
        if body is None:
            return None

        # Create a minimal response-like object
        class CellularResponse:
            status_code = 200
            text = body
            def json(self):
                return json.loads(self.text)
            def raise_for_status(self):
                pass

        _logger.info("Weather fetched via cellular network (%d bytes)", len(body))
        return CellularResponse()
    except ImportError:
        return None
    except Exception as e:
        _logger.warning("Cellular weather fetch failed: %s", e)
        return None


@dataclass
class WeatherData:
    """Current weather conditions at the observation site.

    Attributes:
        temperature:    Air temperature in degrees Celsius.
        pressure:       Surface atmospheric pressure in hectopascals (hPa).
        humidity:       Relative humidity as a percentage (0-100).
        timestamp:      Unix timestamp when the data was fetched.
        location:       Human-readable location string (e.g., "Paris, France").
        latitude:       GPS latitude of the observation site.
        longitude:      GPS longitude of the observation site.
        cloud_cover:    Total cloud cover as a percentage (0-100).
        wind_speed:     Wind speed at 10 m above ground in km/h.
        wind_direction: Wind direction in degrees (0 = North, 90 = East).
        wind_gusts:     Maximum wind gusts at 10 m in km/h.
        dew_point:      Dew point temperature in degrees Celsius.
        weather_code:   WMO weather interpretation code (0 = clear, etc.).
        is_day:         Whether it is currently daytime (True) or night (False).
    """
    temperature: float      # Celsius
    pressure: float         # hPa (hectopascals)
    humidity: float         # Percentage (0-100)
    timestamp: float
    location: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    cloud_cover: float = 0.0        # Percentage (0-100)
    wind_speed: float = 0.0         # km/h
    wind_direction: float = 0.0     # Degrees (0-360)
    wind_gusts: float = 0.0         # km/h
    dew_point: float = 0.0          # Celsius
    weather_code: int = 0           # WMO code
    is_day: bool = True


def calculate_dew_point(temperature: float, humidity: float) -> float:
    """Compute dew point using the Magnus formula.

    The dew point is the temperature at which air becomes saturated and
    moisture begins to condense.  For astronomers this indicates dew risk
    on optics -- when the ambient temperature is within a few degrees of
    the dew point, dew shields or heaters are recommended.

    Args:
        temperature: Air temperature in degrees Celsius.
        humidity:    Relative humidity as a percentage (0-100).

    Returns:
        Dew point temperature in degrees Celsius.
    """
    if humidity <= 0:
        return temperature - 30.0  # Extremely dry air fallback
    # Magnus coefficients (valid for -45 to +60 C)
    a = 17.27
    b = 237.7
    alpha = (a * temperature) / (b + temperature) + math.log(humidity / 100.0)
    return (b * alpha) / (a - alpha)


# WMO weather interpretation codes -> human-readable descriptions
WMO_CODES: Dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight showers",
    81: "Moderate showers",
    82: "Violent showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm + slight hail",
    99: "Thunderstorm + heavy hail",
}


def weather_code_description(code: int) -> str:
    """Return a human-readable description for a WMO weather code.

    Args:
        code: WMO weather interpretation code.

    Returns:
        Short English description string.
    """
    return WMO_CODES.get(code, f"Code {code}")


def assess_observing_conditions(weather: WeatherData) -> Tuple[str, str]:
    """Rate the current sky conditions for astronomical observation.

    Evaluates cloud cover, wind, humidity, and dew risk to produce a
    qualitative rating. This helps the observer decide whether to open
    the session or wait for better conditions.

    Args:
        weather: A populated WeatherData instance.

    Returns:
        A (rating, color_key) tuple where rating is one of
        "Excellent", "Good", "Fair", "Poor", or "Bad" and color_key is
        a CSS/Tk-compatible color string for UI display.
    """
    score = 100  # Start with a perfect score and subtract penalties

    # Cloud cover is the dominant factor for astronomy
    if weather.cloud_cover > 80:
        score -= 60
    elif weather.cloud_cover > 50:
        score -= 35
    elif weather.cloud_cover > 25:
        score -= 15
    elif weather.cloud_cover > 10:
        score -= 5

    # High wind degrades seeing and shakes the telescope
    if weather.wind_speed > 40:
        score -= 30
    elif weather.wind_speed > 25:
        score -= 15
    elif weather.wind_speed > 15:
        score -= 5

    # Wind gusts are worse than steady wind for telescope stability
    if weather.wind_gusts > 50:
        score -= 15
    elif weather.wind_gusts > 35:
        score -= 8

    # Dew risk: when temp is close to dew point, optics fog up
    dew_margin = weather.temperature - weather.dew_point
    if dew_margin < 2:
        score -= 20
    elif dew_margin < 4:
        score -= 10
    elif dew_margin < 6:
        score -= 3

    # Very high humidity degrades transparency
    if weather.humidity > 90:
        score -= 10
    elif weather.humidity > 80:
        score -= 5

    # Precipitation makes observing impossible
    if weather.weather_code >= 51:  # Any form of precipitation
        score -= 40

    # Map score to rating
    if score >= 80:
        return ("Excellent", "#00ff88")
    elif score >= 60:
        return ("Good", "#88ff00")
    elif score >= 40:
        return ("Fair", "#ffaa00")
    elif score >= 20:
        return ("Poor", "#ff6600")
    else:
        return ("Bad", "#ff3333")


class OpenMeteoService:
    """HTTP client for the Open-Meteo weather forecast API.

    Open-Meteo (https://open-meteo.com/) is a free, open-source weather API
    that does not require registration or an API key. It provides current
    conditions and forecasts from multiple weather models.

    Caching Strategy:
        Results are cached for 5 minutes (cache_duration). A cached result
        is reused if:
        1. Less than 5 minutes have elapsed since the last fetch, AND
        2. The requested coordinates are within 0.01 degrees (~1 km) of
           the cached coordinates.
        This prevents excessive API calls during normal operation where the
        observation site does not change.

    Error Handling:
        Returns None on any network error, timeout, or parsing failure.
        The caller (main_realtime.py) handles None gracefully by showing
        "N/A" in the weather display panel.
    """

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self):
        self.last_update_time = 0.0
        self.cache_duration = 600.0  # 10 minutes cache (reduced API traffic)
        self.cached_data: Optional[WeatherData] = None
        # Offline backoff: after consecutive failures, increase wait time
        self._consecutive_failures = 0
        self._max_backoff = 900.0  # 15 min max backoff when offline

    def get_weather(self, latitude: float, longitude: float) -> Optional[WeatherData]:
        """Fetch current weather conditions for a GPS location.

        Uses the Open-Meteo API to retrieve temperature, pressure, humidity,
        cloud cover, wind, and weather code.  Dew point is computed locally
        via the Magnus formula so it does not require an extra API field.
        Returns cached data if the cache is still valid.

        Args:
            latitude:  Observation site latitude in degrees.
            longitude: Observation site longitude in degrees.

        Returns:
            WeatherData object, or None if the API call fails.
        """
        # Check cache validity: same location + within cache duration
        current_time = time.time()
        # If we're in backoff due to consecutive failures, extend the cache window
        effective_cache = self.cache_duration
        if self._consecutive_failures > 0:
            effective_cache = min(
                self.cache_duration * (2 ** self._consecutive_failures),
                self._max_backoff,
            )
        if (self.cached_data and
            current_time - self.last_update_time < effective_cache and
            abs(self.cached_data.latitude - latitude) < 0.01 and
            abs(self.cached_data.longitude - longitude) < 0.01):
            return self.cached_data

        try:
            # Request current conditions from Open-Meteo (expanded field set)
            params = {
                'latitude': latitude,
                'longitude': longitude,
                'current': (
                    'temperature_2m,relative_humidity_2m,surface_pressure,'
                    'cloud_cover,wind_speed_10m,wind_direction_10m,'
                    'wind_gusts_10m,weather_code,is_day'
                ),
                'timezone': 'auto',
                'forecast_days': 1
            }

            # On Android, try cellular network first (WiFi may be telescope hotspot)
            response = None
            if _is_android():
                response = _cellular_get(self.BASE_URL, params, timeout=5.0)

            # Fall back to default network (works on desktop or when on normal WiFi)
            if response is None:
                response = requests.get(self.BASE_URL, params=params, timeout=5.0)
                response.raise_for_status()

            data = response.json()

            if 'current' in data:
                current = data['current']

                temp = current.get('temperature_2m', 20.0)
                hum = current.get('relative_humidity_2m', 50.0)

                weather = WeatherData(
                    temperature=temp,
                    pressure=current.get('surface_pressure', 1013.25),
                    humidity=hum,
                    timestamp=current_time,
                    latitude=latitude,
                    longitude=longitude,
                    location=f"{latitude:.4f}N, {longitude:.4f}E",
                    cloud_cover=current.get('cloud_cover', 0.0),
                    wind_speed=current.get('wind_speed_10m', 0.0),
                    wind_direction=current.get('wind_direction_10m', 0.0),
                    wind_gusts=current.get('wind_gusts_10m', 0.0),
                    dew_point=calculate_dew_point(temp, hum),
                    weather_code=int(current.get('weather_code', 0)),
                    is_day=bool(current.get('is_day', 1)),
                )

                # Update cache and reset failure counter
                self.cached_data = weather
                self.last_update_time = current_time
                self._consecutive_failures = 0

                return weather
            else:
                self._consecutive_failures += 1
                return self.cached_data  # Return stale data instead of None

        except requests.exceptions.RequestException as e:
            self._consecutive_failures += 1
            _logger.warning("Weather fetch error (attempt %d): %s",
                            self._consecutive_failures, e)
            return self.cached_data  # Return stale data instead of None
        except (KeyError, ValueError) as e:
            self._consecutive_failures += 1
            _logger.warning("Weather data parsing error: %s", e)
            return self.cached_data  # Return stale data instead of None

    def get_location_name(self, latitude: float, longitude: float) -> str:
        """Reverse-geocode GPS coordinates to a human-readable place name.

        Uses the Nominatim (OpenStreetMap) reverse geocoding API.
        Falls back to a coordinate string if the lookup fails.

        Args:
            latitude:  GPS latitude in degrees.
            longitude: GPS longitude in degrees.

        Returns:
            Location name string (e.g., "Paris, France") or coordinate fallback.
        """
        try:
            url = "https://nominatim.openstreetmap.org/reverse"
            params = {
                'lat': latitude,
                'lon': longitude,
                'format': 'json',
                'addressdetails': 1
            }
            # Nominatim requires a User-Agent header to identify the application
            headers = {'User-Agent': 'TrackWiseAltAzPro/1.0'}

            # On Android, try cellular network first
            response = None
            if _is_android():
                # Add User-Agent to params isn't possible, but cellular_get
                # already sets User-Agent in Kotlin, so this works fine
                response = _cellular_get(url, params, timeout=3.0)

            if response is None:
                response = requests.get(url, params=params, headers=headers, timeout=3.0)
                response.raise_for_status()

            data = response.json()
            if 'address' in data:
                addr = data['address']
                city = addr.get('city') or addr.get('town') or addr.get('village')
                country = addr.get('country')
                if city and country:
                    return f"{city}, {country}"
                elif city:
                    return city
                elif country:
                    return country

            return f"{latitude:.4f}N, {longitude:.4f}E"
        except Exception:
            return f"{latitude:.4f}N, {longitude:.4f}E"


def get_gps_from_browser() -> Optional[Tuple[float, float]]:
    """Stub: Attempt to get GPS position from the browser.

    Currently unimplemented. Would require a web-based interface or
    OS-level location API to determine the user's position automatically.

    Returns:
        None (not yet implemented).
    """
    return None
