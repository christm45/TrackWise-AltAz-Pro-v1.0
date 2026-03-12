"""
Celestial object catalog loader from .h header files.

Parses C/C++ header files and extracts objects with their coordinates.
"""

import re
import os
from typing import Dict, Tuple, List

from telescope_logger import get_logger

_logger = get_logger(__name__)


def parse_messier_catalog(catalog_dir: str) -> Dict[str, Tuple[float, float]]:
    """
    Parse the Messier catalog from messier.h (optimized version).
    
    Returns:
        Dict: {name: (ra_hours, dec_degrees)}
    """
    catalog = {}
    messier_file = os.path.join(catalog_dir, "data", "messier.h")
    
    if not os.path.exists(messier_file):
        return catalog
    
    try:
        # Read the file in one pass
        with open(messier_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract ONLY coordinates (faster)
        # Format: { flags, type, subId, dup, id, mag, ra, dec }
        coords_pattern = r'\{[^}]*,\s*(\d+),\s*[\d.]+,\s*([\d.]+),\s*([-\d.]+)\s*\}'
        coords_matches = re.findall(coords_pattern, content)
        
        # Extract names (simplified)
        names_match = re.search(r'Cat_Messier_Names=(.*?);\s*const char \*Cat_Messier_SubId', content, re.DOTALL)
        names = []
        if names_match:
            names_str = names_match.group(1)
            names = [n.rstrip(';').strip() for n in re.findall(r'"([^"]+)"', names_str)]
        
        # Process all coordinate entries (names are optional)
        max_objects = min(110, len(coords_matches))
        for i in range(max_objects):
            obj_id, ra_str, dec_str = coords_matches[i]
            ra = float(ra_str)
            dec = float(dec_str)
            
            # Always add "m" + number
            catalog[f"m{obj_id}"] = (ra, dec)
            catalog[f"messier {obj_id}"] = (ra, dec)
            
            # If we have a name for this object
            if i < len(names):
                name = names[i].lower().rstrip(';').strip()
                if name:
                    catalog[name] = (ra, dec)
                    
                    # Common alternative names (only the most important ones)
                    if "andromeda" in name:
                        catalog["andromeda"] = (ra, dec)
                    if "orion" in name:
                        catalog["orion nebula"] = (ra, dec)
                        catalog["orion"] = (ra, dec)
                    if "pleiades" in name:
                        catalog["pleiades"] = (ra, dec)
                    if "whirlpool" in name:
                        catalog["whirlpool"] = (ra, dec)
                    if "ring" in name:
                        catalog["ring nebula"] = (ra, dec)
                    if "dumbbell" in name:
                        catalog["dumbbell"] = (ra, dec)
                    if "crab" in name:
                        catalog["crab nebula"] = (ra, dec)
    
    except Exception as e:
        _logger.warning("Error parsing Messier catalog: %s", e)
        import traceback
        traceback.print_exc()
    
    return catalog


def parse_ngc_catalog(catalog_dir: str, limit: int = 300, min_dec: float = -30.0) -> Dict[str, Tuple[float, float]]:
    """
    Parse the NGC catalog (filtered for northern hemisphere).
    
    Args:
        limit: Maximum number of objects to load.
        min_dec: Minimum declination (default: -30 deg for northern hemisphere).
    
    Returns:
        Dict: {name: (ra_hours, dec_degrees)}
    """
    catalog = {}
    ngc_file = os.path.join(catalog_dir, "data", "ngc.h")
    
    if not os.path.exists(ngc_file):
        return catalog
    
    try:
        with open(ngc_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract coordinates from dso_t structs
        # Format: { flags, type, subId, dup, id, mag, ra, dec }
        # RA is in decimal hours, Dec in decimal degrees (verified against comments)
        coords_pattern = r'\{[^}]*,\s*(\d+),\s*[\d.]+,\s*([\d.]+),\s*([-\d.]+)\s*\}'
        coords_matches = re.findall(coords_pattern, content)
        
        count = 0
        for obj_id, ra_str, dec_str in coords_matches:
            if count >= limit:
                break
            ra_hours = float(ra_str)  # Already in decimal hours
            dec = float(dec_str)
            
            # Filter by declination
            if dec >= min_dec:
                # Add "ngc" + number
                catalog[f"ngc {obj_id}"] = (ra_hours, dec)
                catalog[f"ngc{obj_id}"] = (ra_hours, dec)
                catalog[f"n{obj_id}"] = (ra_hours, dec)
                count += 1
    
    except Exception as e:
        _logger.warning("Error parsing NGC catalog: %s", e)
    
    return catalog


def parse_ic_catalog(catalog_dir: str, limit: int = 100, min_dec: float = -30.0) -> Dict[str, Tuple[float, float]]:
    """
    Parse the IC catalog (Index Catalogue, filtered for northern hemisphere).
    
    Args:
        limit: Maximum number of objects to load.
        min_dec: Minimum declination (default: -30 deg for northern hemisphere).
    """
    catalog = {}
    ic_file = os.path.join(catalog_dir, "data", "ic.h")
    
    if not os.path.exists(ic_file):
        return catalog
    
    try:
        with open(ic_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract coordinates (dso_t format like Messier)
        # Format: { flags, type, subId, dup, id, mag, ra, dec }
        # Example: { 0, 61, 3, 0, 1, 9990, 0.14085, 27.71767}
        coords_pattern = r'\{[^}]*,\s*(\d+),\s*[\d.]+,\s*([\d.]+),\s*([-\d.]+)\s*\}'
        coords_matches = re.findall(coords_pattern, content)
        
        count = 0
        for obj_id, ra_str, dec_str in coords_matches:
            if count >= limit:
                break
            ra = float(ra_str)  # RA in decimal hours
            dec = float(dec_str)  # Dec in decimal degrees
            
            # Filter for northern hemisphere (declination > min_dec)
            if dec >= min_dec:
                # Add "ic" + number
                catalog[f"ic {obj_id}"] = (ra, dec)
                catalog[f"ic{obj_id}"] = (ra, dec)
                catalog[f"i{obj_id}"] = (ra, dec)
                count += 1
    
    except Exception as e:
        _logger.warning("Error parsing IC catalog: %s", e)
    
    return catalog


def parse_stars_catalog(catalog_dir: str, limit: int = 200) -> Dict[str, Tuple[float, float]]:
    """
    Parse the bright star catalog (optimized).

    Format: { Has_name, Cons, BayerFlam, Has_subId, Obj_id, Mag, RA, DE }
    
    Args:
        limit: Maximum number of stars to load (default: 200).
    """
    catalog = {}
    stars_file = os.path.join(catalog_dir, "data", "stars.h")
    
    if not os.path.exists(stars_file):
        return catalog
    
    try:
        with open(stars_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract names -- the C format is a single semicolon-delimited string
        # spread across multiple quoted segments (C string concatenation).
        # e.g. "Ain;Alcyone;" is TWO names, not one.
        names_match = re.search(r'Cat_Stars_Names=(.*?);\s*const', content, re.DOTALL)
        if not names_match:
            return catalog
        
        names_str = names_match.group(1)
        # Concatenate all quoted segments into one string, then split by semicolons
        all_text = ''.join(re.findall(r'"([^"]+)"', names_str))
        names = [n.strip() for n in all_text.split(';') if n.strip()]
        if limit and len(names) > limit:
            names = names[:limit]
        
        # Extract coordinates (gen_star_t format)
        # Format: { Has_name, Cons, BayerFlam, Has_subId, Obj_id, Mag, RA, DE }
        # First number is Has_name (1 = has a name, 0 = no name)
        # RA and DE are the 7th and 8th elements (the last two numbers)
        star_pattern = r'\{\s*(\d+)\s*,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*([-\d.]+),\s*([-\d.]+)\s*\}'
        star_matches = re.findall(star_pattern, content)
        
        # Associate names and coordinates -- only for stars with Has_name=1
        name_index = 0
        count = 0
        for has_name, ra_str, dec_str in star_matches:
            if count >= limit:
                break
            if int(has_name) == 1 and name_index < len(names):
                name = names[name_index].lower()
                ra_hours = float(ra_str)  # Already in decimal hours
                dec_deg = float(dec_str)
                catalog[name] = (ra_hours, dec_deg)
                name_index += 1
                count += 1
    
    except Exception as e:
        _logger.warning("Error parsing Stars catalog: %s", e)
    
    return catalog


def parse_caldwell_catalog(catalog_dir: str) -> Dict[str, Tuple[float, float]]:
    """
    Parse the Caldwell catalog from caldwell.h.

    Returns:
        Dict: {name: (ra_hours, dec_degrees)}
    """
    catalog = {}
    caldwell_file = os.path.join(catalog_dir, "data", "caldwell.h")

    if not os.path.exists(caldwell_file):
        return catalog

    try:
        with open(caldwell_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Extract coordinates
        # Format: { flags, type, subId, dup, id, mag, ra, dec }
        coords_pattern = r'\{[^}]*,\s*(\d+),\s*[\d.]+,\s*([\d.]+),\s*([-\d.]+)\s*\}'
        coords_matches = re.findall(coords_pattern, content)

        # Extract names
        names_match = re.search(
            r'Cat_Caldwell_Names=(.*?);\s*\n',
            content, re.DOTALL
        )
        names = []
        if names_match:
            names_str = names_match.group(1)
            names = [n.rstrip(';').strip() for n in re.findall(r'"([^"]+)"', names_str)]

        max_objects = min(109, len(coords_matches))
        for i in range(max_objects):
            obj_id, ra_str, dec_str = coords_matches[i]
            ra = float(ra_str)
            dec = float(dec_str)

            catalog[f"c{obj_id}"] = (ra, dec)
            catalog[f"caldwell {obj_id}"] = (ra, dec)

            # Add common name if available
            if i < len(names):
                name = names[i].lower().rstrip(';').strip()
                if name and name.strip():
                    catalog[name] = (ra, dec)

    except Exception as e:
        _logger.warning("Error parsing Caldwell catalog: %s", e)

    return catalog


# ===================================================================
# Category browsing support
# ===================================================================

_CATEGORY_CACHE = None


def _build_category_list(
    catalog: Dict[str, Tuple[float, float]],
    prefix: str,
    id_format: str = "{prefix}{num}",
) -> List[Dict]:
    """
    Build a sorted list of objects from the catalog matching a prefix.

    Args:
        catalog: The flat catalog dict.
        prefix: Key prefix to match (e.g. 'm', 'ngc ', 'c').
        id_format: Format string for display ID.

    Returns:
        List of dicts with id, name, ra_hours, dec_degrees.
    """
    seen_ids = set()
    items = []
    for key, (ra, dec) in catalog.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        if not suffix.strip().isdigit():
            continue
        num = int(suffix.strip())
        if num in seen_ids:
            continue
        seen_ids.add(num)
        items.append({
            "id": f"{prefix.strip().upper()}{num}",
            "num": num,
            "ra_hours": round(ra, 4),
            "dec_degrees": round(dec, 4),
        })
    items.sort(key=lambda x: x["num"])
    # Remove the 'num' helper key
    for it in items:
        del it["num"]
    return items


def _build_star_list(catalog_dir: str) -> List[Dict]:
    """
    Build a list of named bright stars from the stars catalog.

    Returns:
        List of dicts with id (star name), ra_hours, dec_degrees.
    """
    stars = parse_stars_catalog(catalog_dir, limit=500)
    items = []
    for name, (ra, dec) in stars.items():
        items.append({
            "id": name.title(),
            "ra_hours": round(ra, 4),
            "dec_degrees": round(dec, 4),
            "sort_key": name,
        })
    items.sort(key=lambda x: x["sort_key"])
    for it in items:
        del it["sort_key"]
    return items


def get_catalog_categories(catalog_dir: str = "catalogs") -> Dict[str, List[Dict]]:
    """
    Return catalog objects organized by browsable category.

    Categories: Messier, Caldwell, NGC (popular subset), IC, Stars.

    Returns:
        Dict keyed by category name, each value is a list of
        {id, ra_hours, dec_degrees} dicts sorted by catalog number.
    """
    global _CATEGORY_CACHE
    if _CATEGORY_CACHE is not None:
        return _CATEGORY_CACHE

    # Ensure the flat catalog is loaded first
    all_cat = load_all_catalogs(catalog_dir)

    # Also load Caldwell into the flat cache if not already present
    if "c1" not in all_cat:
        caldwell = parse_caldwell_catalog(catalog_dir)
        all_cat.update(caldwell)

    categories = {}

    # Messier (M1-M110)
    categories["Messier"] = _build_category_list(all_cat, "m")

    # Caldwell (C1-C109)
    categories["Caldwell"] = _build_category_list(all_cat, "c")

    # Popular NGC (limit to objects with NGC numbers from a curated set,
    # or just all NGC sorted by number -- but that's 10k+ objects).
    # For browsing, show top 200 NGC objects sorted by number.
    ngc_list = _build_category_list(all_cat, "ngc ")
    categories["NGC"] = ngc_list[:200]
    categories["NGC (all)"] = ngc_list

    # IC
    ic_list = _build_category_list(all_cat, "ic ")
    categories["IC"] = ic_list[:100]

    # Bright stars
    categories["Stars"] = _build_star_list(catalog_dir)

    _CATEGORY_CACHE = categories
    return categories


# Global cache to avoid reloading every time
_CATALOG_CACHE = None

def load_all_catalogs(catalog_dir: str = "catalogs", use_cache: bool = True, fast_mode: bool = True) -> Dict[str, Tuple[float, float]]:
    """
    Load all available catalogs (with caching).
    
    Args:
        catalog_dir: Directory containing the catalog header files.
        use_cache: Use cached data if available (default: True).
        fast_mode: Fast mode -- loads only Messier + stars + NGC/IC (default: True).
    
    Returns:
        Combined dict of all objects: {name: (ra_hours, dec_degrees)}
    """
    global _CATALOG_CACHE
    
    # Return cache if available
    if use_cache and _CATALOG_CACHE is not None:
        return _CATALOG_CACHE
    
    combined = {}
    
    _logger.info("Loading catalogs from: %s", catalog_dir)
    
    # Messier (always loaded -- fast and most commonly used)
    try:
        messier = parse_messier_catalog(catalog_dir)
        combined.update(messier)
        _logger.info("  Messier: %d objects", len(messier))
    except Exception as e:
        _logger.warning("  Messier: Error (%s)", e)
    
    # Fast mode: Messier + stars + NGC/IC (all visible objects)
    if fast_mode:
        # Load bright stars (up to 500)
        try:
            stars = parse_stars_catalog(catalog_dir, limit=500)
            combined.update(stars)
            _logger.info("  Stars: %d stars", len(stars))
        except Exception as e:
            _logger.warning("  Stars: Error (%s)", e)
        
        # Load all NGC objects (includes NGC 7000 etc.)
        try:
            ngc = parse_ngc_catalog(catalog_dir, limit=99999, min_dec=-90.0)
            combined.update(ngc)
            _logger.info("  NGC: %d objects", len(ngc))
        except Exception as e:
            _logger.warning("  NGC: Error (%s)", e)
        
        # Load all IC objects
        try:
            ic = parse_ic_catalog(catalog_dir, limit=99999, min_dec=-90.0)
            combined.update(ic)
            _logger.info("  IC: %d objects", len(ic))
        except Exception as e:
            _logger.warning("  IC: Error (%s)", e)

        # Load Caldwell catalog
        try:
            caldwell = parse_caldwell_catalog(catalog_dir)
            combined.update(caldwell)
            _logger.info("  Caldwell: %d objects", len(caldwell))
        except Exception as e:
            _logger.warning("  Caldwell: Error (%s)", e)

        _logger.info("Total: %d objects loaded", len(combined))
        if use_cache:
            _CATALOG_CACHE = combined
        return combined
    
    # NGC (limited to 50 for performance -- only in non-fast mode)
    try:
        ngc = parse_ngc_catalog(catalog_dir, limit=50)
        combined.update(ngc)
        _logger.info("  NGC: %d objects (limited)", len(ngc))
    except Exception as e:
        _logger.warning("  NGC: Error (%s)", e)
    
    # Stars -- skipped for now (too slow for full catalog)
    
    _logger.info("Total: %d objects loaded", len(combined))
    
    # Store in cache
    if use_cache:
        _CATALOG_CACHE = combined
    
    return combined


# ===================================================================
# Solar system objects -- real-time ephemeris (low-precision)
# ===================================================================
# Uses simplified Keplerian elements + perturbation terms from
# Jean Meeus "Astronomical Algorithms" and the USNO/AA low-precision
# formulae.  Accuracy: ~1' for planets, ~2' for the Moon, ~1' for Sun.
# This is more than sufficient for GoTo slewing.
# ===================================================================

import math as _math

def _jd_now():
    """Current Julian Date (UTC)."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    y, m = now.year, now.month
    d = now.day + now.hour / 24.0 + now.minute / 1440.0 + now.second / 86400.0
    if m <= 2:
        y -= 1
        m += 12
    a = int(y / 100)
    b = 2 - a + int(a / 4)
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5


def _norm360(x):
    return x % 360.0


def _norm180(x):
    x = x % 360.0
    if x > 180:
        x -= 360
    return x


def _sun_ra_dec(jd):
    """Low-precision Sun RA/Dec (equatorial, J2000-ish).

    Returns (ra_hours, dec_degrees).
    """
    T = (jd - 2451545.0) / 36525.0  # Julian centuries from J2000

    # Mean longitude and mean anomaly of the Sun
    L0 = _norm360(280.46646 + 36000.76983 * T + 0.0003032 * T * T)
    M = _norm360(357.52911 + 35999.05029 * T - 0.0001537 * T * T)
    M_rad = _math.radians(M)

    # Equation of center
    C = ((1.914602 - 0.004817 * T - 0.000014 * T * T) * _math.sin(M_rad) +
         (0.019993 - 0.000101 * T) * _math.sin(2 * M_rad) +
         0.000289 * _math.sin(3 * M_rad))

    # Sun's true longitude and true anomaly
    sun_lon = _norm360(L0 + C)

    # Obliquity of the ecliptic
    eps = 23.439291 - 0.0130042 * T
    eps_rad = _math.radians(eps)
    lon_rad = _math.radians(sun_lon)

    # Convert ecliptic -> equatorial
    ra_rad = _math.atan2(_math.cos(eps_rad) * _math.sin(lon_rad), _math.cos(lon_rad))
    dec_rad = _math.asin(_math.sin(eps_rad) * _math.sin(lon_rad))

    ra_deg = _math.degrees(ra_rad) % 360.0
    ra_hours = ra_deg / 15.0
    dec_deg = _math.degrees(dec_rad)

    return ra_hours, dec_deg


def _moon_ra_dec(jd):
    """Low-precision Moon RA/Dec.

    Uses the simplified lunar theory from Meeus Ch. 47 (truncated to
    the dominant terms).  Accuracy ~0.3 deg, good enough for GoTo.

    Returns (ra_hours, dec_degrees).
    """
    T = (jd - 2451545.0) / 36525.0

    # Fundamental arguments (degrees)
    Lp = _norm360(218.3164477 + 481267.88123421 * T
                   - 0.0015786 * T * T + T * T * T / 538841.0)  # Moon mean longitude
    D = _norm360(297.8501921 + 445267.1114034 * T
                  - 0.0018819 * T * T + T * T * T / 545868.0)   # Mean elongation
    M = _norm360(357.5291092 + 35999.0502909 * T
                  - 0.0001536 * T * T)                            # Sun mean anomaly
    Mp = _norm360(134.9633964 + 477198.8675055 * T
                   + 0.0087414 * T * T + T * T * T / 69699.0)    # Moon mean anomaly
    F = _norm360(93.2720950 + 483202.0175233 * T
                  - 0.0036539 * T * T)                             # Moon argument of latitude

    # Convert to radians
    D_r = _math.radians(D)
    M_r = _math.radians(M)
    Mp_r = _math.radians(Mp)
    F_r = _math.radians(F)

    # Longitude (sum of the largest periodic terms, in degrees)
    l = (6.288774 * _math.sin(Mp_r)
         + 1.274027 * _math.sin(2 * D_r - Mp_r)
         + 0.658314 * _math.sin(2 * D_r)
         + 0.213618 * _math.sin(2 * Mp_r)
         - 0.185116 * _math.sin(M_r)
         - 0.114332 * _math.sin(2 * F_r)
         + 0.058793 * _math.sin(2 * D_r - 2 * Mp_r)
         + 0.057066 * _math.sin(2 * D_r - M_r - Mp_r)
         + 0.053322 * _math.sin(2 * D_r + Mp_r)
         + 0.045758 * _math.sin(2 * D_r - M_r)
         - 0.040923 * _math.sin(M_r - Mp_r)
         - 0.034720 * _math.sin(D_r)
         - 0.030383 * _math.sin(M_r + Mp_r)
         + 0.015327 * _math.sin(2 * D_r - 2 * F_r)
         - 0.012528 * _math.sin(Mp_r + 2 * F_r)
         + 0.010980 * _math.sin(Mp_r - 2 * F_r))

    # Latitude
    b = (5.128122 * _math.sin(F_r)
         + 0.280602 * _math.sin(Mp_r + F_r)
         + 0.277693 * _math.sin(Mp_r - F_r)
         + 0.173237 * _math.sin(2 * D_r - F_r)
         + 0.055413 * _math.sin(2 * D_r - Mp_r + F_r)
         + 0.046271 * _math.sin(2 * D_r - Mp_r - F_r)
         + 0.032573 * _math.sin(2 * D_r + F_r)
         + 0.017198 * _math.sin(2 * Mp_r + F_r)
         + 0.009266 * _math.sin(2 * D_r + Mp_r - F_r)
         + 0.008822 * _math.sin(2 * Mp_r - F_r))

    moon_lon = Lp + l
    moon_lat = b

    # Obliquity
    eps = 23.439291 - 0.0130042 * T
    eps_r = _math.radians(eps)
    lon_r = _math.radians(moon_lon)
    lat_r = _math.radians(moon_lat)

    # Ecliptic -> equatorial
    ra_rad = _math.atan2(
        _math.sin(lon_r) * _math.cos(eps_r) - _math.tan(lat_r) * _math.sin(eps_r),
        _math.cos(lon_r)
    )
    dec_rad = _math.asin(
        _math.sin(lat_r) * _math.cos(eps_r) +
        _math.cos(lat_r) * _math.sin(eps_r) * _math.sin(lon_r)
    )

    ra_hours = (_math.degrees(ra_rad) % 360.0) / 15.0
    dec_deg = _math.degrees(dec_rad)

    return ra_hours, dec_deg


# Keplerian orbital elements at J2000 + rates per century
# Source: Standish (1992) via JPL / Meeus
# (a_au, e, I_deg, L_deg, w_deg, Om_deg) + rates per century
_PLANET_ELEMENTS = {
    "mercury": (
        (0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593),
        (0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081),
    ),
    "venus": (
        (0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255),
        (0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418),
    ),
    "mars": (
        (1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891),
        (0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343),
    ),
    "jupiter": (
        (5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909),
        (-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106),
    ),
    "saturn": (
        (9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448),
        (-0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794),
    ),
    "uranus": (
        (19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503),
        (-0.00196176, -0.00004397, -0.00242939, 428.48202785, 0.40805281, 0.04240589),
    ),
    "neptune": (
        (30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574),
        (0.00026291, 0.00005105, 0.00035372, 218.45945325, -0.32241464, -0.00508664),
    ),
}


def _planet_ra_dec(name, jd):
    """Compute geocentric RA/Dec for a planet using Keplerian elements.

    Returns (ra_hours, dec_degrees) or None if not found.
    """
    name_l = name.lower()
    if name_l not in _PLANET_ELEMENTS:
        return None

    T = (jd - 2451545.0) / 36525.0  # centuries from J2000

    elem0, rate = _PLANET_ELEMENTS[name_l]
    # Elements at current epoch
    a = elem0[0] + rate[0] * T
    e = elem0[1] + rate[1] * T
    I = elem0[2] + rate[2] * T
    L = _norm360(elem0[3] + rate[3] * T)
    w_bar = _norm360(elem0[4] + rate[4] * T)  # longitude of perihelion
    Om = _norm360(elem0[5] + rate[5] * T)      # longitude of ascending node

    w = w_bar - Om  # argument of perihelion
    M_deg = _norm180(L - w_bar)  # mean anomaly
    M_rad = _math.radians(M_deg)

    # Solve Kepler's equation: E - e*sin(E) = M  (Newton-Raphson)
    E = M_rad
    for _ in range(20):
        dE = (E - e * _math.sin(E) - M_rad) / (1 - e * _math.cos(E))
        E -= dE
        if abs(dE) < 1e-12:
            break

    # Heliocentric position in orbital plane
    xp = a * (_math.cos(E) - e)
    yp = a * _math.sqrt(1 - e * e) * _math.sin(E)

    # Rotate to ecliptic (heliocentric ecliptic coordinates)
    w_r = _math.radians(w)
    Om_r = _math.radians(Om)
    I_r = _math.radians(I)

    cos_w, sin_w = _math.cos(w_r), _math.sin(w_r)
    cos_Om, sin_Om = _math.cos(Om_r), _math.sin(Om_r)
    cos_I, sin_I = _math.cos(I_r), _math.sin(I_r)

    x_ecl = ((cos_w * cos_Om - sin_w * sin_Om * cos_I) * xp +
             (-sin_w * cos_Om - cos_w * sin_Om * cos_I) * yp)
    y_ecl = ((cos_w * sin_Om + sin_w * cos_Om * cos_I) * xp +
             (-sin_w * sin_Om + cos_w * cos_Om * cos_I) * yp)
    z_ecl = (sin_w * sin_I * xp + cos_w * sin_I * yp)

    # Earth's heliocentric position (use simplified Earth elements)
    earth_elem0 = (1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0)
    earth_rate = (0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0)

    a_e = earth_elem0[0] + earth_rate[0] * T
    e_e = earth_elem0[1] + earth_rate[1] * T
    L_e = _norm360(earth_elem0[3] + earth_rate[3] * T)
    w_bar_e = _norm360(earth_elem0[4] + earth_rate[4] * T)
    M_e_deg = _norm180(L_e - w_bar_e)
    M_e_rad = _math.radians(M_e_deg)

    E_e = M_e_rad
    for _ in range(20):
        dE = (E_e - e_e * _math.sin(E_e) - M_e_rad) / (1 - e_e * _math.cos(E_e))
        E_e -= dE
        if abs(dE) < 1e-12:
            break

    xe = a_e * (_math.cos(E_e) - e_e)
    ye = a_e * _math.sqrt(1 - e_e * e_e) * _math.sin(E_e)

    # Earth in ecliptic -- simplified (Om~0, I~0 for ecliptic plane)
    w_e = _math.radians(w_bar_e)  # for Earth, Om~0 so w ≈ w_bar
    x_earth = _math.cos(w_e) * xe - _math.sin(w_e) * ye
    y_earth = _math.sin(w_e) * xe + _math.cos(w_e) * ye
    z_earth = 0.0  # Earth in ecliptic plane by definition

    # Geocentric ecliptic
    xg = x_ecl - x_earth
    yg = y_ecl - y_earth
    zg = z_ecl - z_earth

    # Ecliptic -> equatorial
    eps = _math.radians(23.439291 - 0.0130042 * T)
    xeq = xg
    yeq = yg * _math.cos(eps) - zg * _math.sin(eps)
    zeq = yg * _math.sin(eps) + zg * _math.cos(eps)

    ra_rad = _math.atan2(yeq, xeq)
    dec_rad = _math.atan2(zeq, _math.sqrt(xeq * xeq + yeq * yeq))

    ra_hours = (_math.degrees(ra_rad) % 360.0) / 15.0
    dec_deg = _math.degrees(dec_rad)

    return ra_hours, dec_deg


def get_solar_system_objects() -> Dict[str, Tuple[float, float]]:
    """Compute current RA/Dec for Sun, Moon, and planets.

    Returns a dict: {name: (ra_hours, dec_degrees)}.
    These are always computed fresh (never cached) because they move.
    """
    objects = {}
    try:
        jd = _jd_now()

        # Sun
        ra, dec = _sun_ra_dec(jd)
        objects["sun"] = (ra, dec)
        objects["the sun"] = (ra, dec)

        # Moon
        ra, dec = _moon_ra_dec(jd)
        objects["moon"] = (ra, dec)
        objects["the moon"] = (ra, dec)
        objects["luna"] = (ra, dec)

        # Planets
        for planet in _PLANET_ELEMENTS:
            result = _planet_ra_dec(planet, jd)
            if result:
                objects[planet] = result
                # Common aliases
                if planet == "mars":
                    objects["the red planet"] = result
                elif planet == "jupiter":
                    objects["jup"] = result
                elif planet == "saturn":
                    objects["sat"] = result

    except Exception as e:
        _logger.warning("Error computing solar system positions: %s", e)

    return objects


# Browsable list for the "Solar System" category
def _build_solar_system_list() -> List[Dict]:
    """Build the Solar System browsable category list."""
    objs = get_solar_system_objects()
    # Display order
    display_names = [
        ("Sun", "sun"),
        ("Moon", "moon"),
        ("Mercury", "mercury"),
        ("Venus", "venus"),
        ("Mars", "mars"),
        ("Jupiter", "jupiter"),
        ("Saturn", "saturn"),
        ("Uranus", "uranus"),
        ("Neptune", "neptune"),
    ]
    items = []
    for display, key in display_names:
        if key in objs:
            ra, dec = objs[key]
            items.append({
                "id": display,
                "ra_hours": round(ra, 4),
                "dec_degrees": round(dec, 4),
            })
    return items


# ===================================================================
# Sky Chart data extraction
# ===================================================================

# DSO type codes used in .h catalog files
_DSO_TYPE_NAMES = {
    0: 'Galaxy', 1: 'Open Cluster', 2: 'Star', 3: 'Double Star',
    5: 'Galaxy', 8: 'Globular Cluster', 9: 'Planetary Nebula', 10: 'Nebula',
    13: 'Star Association', 15: 'Supernova Remnant', 19: 'Duplicate',
    61: 'Galaxy', 77: 'SNR',
}

# Constellation stick figure patterns: {cons_index: {'name','abbr','lines':[[b1,b2],...]}}
# Lines connect stars by their Bayer/Flamsteed index within the constellation.
# Cross-constellation lines are in _CROSS_CONSTELLATION_LINES.
_CONSTELLATION_PATTERNS = {
    # Ursa Major - Big Dipper bowl + handle
    82: {'name': 'Ursa Major', 'abbr': 'UMa',
         'lines': [[0,1],[1,2],[2,3],[0,3],[3,4],[4,5],[5,6]]},
    # Ursa Minor - Little Dipper
    83: {'name': 'Ursa Minor', 'abbr': 'UMi',
         'lines': [[0,2],[2,1]]},
    # Orion
    59: {'name': 'Orion', 'abbr': 'Ori',
         'lines': [[0,10],[10,2],[0,5],[2,3],[3,4],[4,5],[5,9],[3,1],[1,9]]},
    # Cassiopeia - W shape
    16: {'name': 'Cassiopeia', 'abbr': 'Cas',
         'lines': [[0,1],[1,2],[2,3],[3,4]]},
    # Cygnus - Northern Cross
    30: {'name': 'Cygnus', 'abbr': 'Cyg',
         'lines': [[0,2],[2,1],[2,3],[2,4]]},
    # Leo - Sickle + body
    46: {'name': 'Leo', 'abbr': 'Leo',
         'lines': [[0,6],[6,2],[2,3],[3,1],[0,7],[7,1]]},
    # Gemini - Twins
    37: {'name': 'Gemini', 'abbr': 'Gem',
         'lines': [[0,1],[0,9],[9,3],[3,10],[1,8],[8,4],[4,11],[11,6]]},
    # Lyra - parallelogram + Vega
    51: {'name': 'Lyra', 'abbr': 'Lyr',
         'lines': [[0,1],[1,2],[0,2]]},
    # Bootes - kite shape
    8: {'name': 'Bootes', 'abbr': 'Boo',
        'lines': [[0,4],[4,3],[3,1],[0,6],[6,2],[2,16],[16,1]]},
    # Scorpius - full scorpion
    71: {'name': 'Scorpius', 'abbr': 'Sco',
         'lines': [[1,3],[3,15],[15,0],[0,17],[17,18],[18,4],[4,11],[11,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,19]]},
    # Sagittarius - Teapot
    76: {'name': 'Sagittarius', 'abbr': 'Sgr',
         'lines': [[4,3],[3,10],[10,20],[20,5],[5,17],[17,18],[18,5],[4,6],[6,3],[10,17]]},
    # Aquila - eagle
    3: {'name': 'Aquila', 'abbr': 'Aql',
        'lines': [[1,0],[0,2]]},
    # Pegasus - Great Square (partial)
    61: {'name': 'Pegasus', 'abbr': 'Peg',
         'lines': [[0,1],[0,2],[1,6],[4,7]]},
    # Andromeda - chain
    0: {'name': 'Andromeda', 'abbr': 'And',
        'lines': [[0,3],[3,1],[1,2]]},
    # Taurus - V shape + horns
    77: {'name': 'Taurus', 'abbr': 'Tau',
         'lines': [[0,7],[7,2],[0,4],[4,3],[0,10],[10,13],[1,5]]},
    # Canis Major
    9: {'name': 'Canis Major', 'abbr': 'CMa',
        'lines': [[0,1],[0,6],[6,3],[3,4],[0,14]]},
    # Canis Minor
    10: {'name': 'Canis Minor', 'abbr': 'CMi',
         'lines': [[0,1]]},
    # Perseus
    62: {'name': 'Perseus', 'abbr': 'Per',
         'lines': [[0,3],[3,4],[4,5],[0,2],[2,6],[0,1]]},
    # Auriga - pentagon
    7: {'name': 'Auriga', 'abbr': 'Aur',
        'lines': [[0,1],[1,7],[7,8],[8,4],[4,0]]},
    # Draco - winding dragon
    33: {'name': 'Draco', 'abbr': 'Dra',
         'lines': [[2,1],[1,13],[13,12],[12,5],[5,6],[6,8],[8,0],[0,10],[2,3],[3,4]]},
    # Corona Borealis - arc
    26: {'name': 'Corona Borealis', 'abbr': 'CrB',
         'lines': [[0,1],[0,2]]},
    # Aries
    6: {'name': 'Aries', 'abbr': 'Ari',
        'lines': [[0,1]]},
    # Virgo
    85: {'name': 'Virgo', 'abbr': 'Vir',
         'lines': [[0,2],[2,3],[3,4],[4,1],[2,5]]},
    # Libra
    48: {'name': 'Libra', 'abbr': 'Lib',
         'lines': [[0,1],[0,17]]},
    # Ophiuchus
    58: {'name': 'Ophiuchus', 'abbr': 'Oph',
         'lines': [[0,1],[0,9],[1,2],[3,4],[3,5],[5,6]]},
    # Hercules - Keystone + limbs
    39: {'name': 'Hercules', 'abbr': 'Her',
         'lines': [[5,6],[6,15],[15,3],[3,5],[0,3],[1,5],[6,8]]},
    # Cepheus - house shape
    18: {'name': 'Cepheus', 'abbr': 'Cep',
         'lines': [[0,1],[1,2],[0,3],[3,6],[6,0],[6,5]]},
    # Corvus - trapezoid
    29: {'name': 'Corvus', 'abbr': 'Crv',
         'lines': [[1,2],[2,3],[3,4],[4,1]]},
    # Crux - Southern Cross
    28: {'name': 'Crux', 'abbr': 'Cru',
         'lines': [[0,2],[1,3]]},
    # Centaurus
    17: {'name': 'Centaurus', 'abbr': 'Cen',
         'lines': [[0,1],[1,4],[4,5],[5,6]]},
    # Triangulum
    80: {'name': 'Triangulum', 'abbr': 'Tri',
         'lines': [[0,1]]},
    # Eridanus (partial)
    35: {'name': 'Eridanus', 'abbr': 'Eri',
         'lines': [[1,2],[2,3],[3,4]]},
    # Capricornus
    14: {'name': 'Capricornus', 'abbr': 'Cap',
         'lines': [[0,1],[1,2],[2,3]]},
    # Aquarius
    4: {'name': 'Aquarius', 'abbr': 'Aqr',
        'lines': [[0,1],[1,3]]},
    # Delphinus
    31: {'name': 'Delphinus', 'abbr': 'Del',
         'lines': [[0,1]]},
    # Lepus
    47: {'name': 'Lepus', 'abbr': 'Lep',
         'lines': [[0,1],[0,2],[2,3],[1,4]]},
    # Carina (partial)
    15: {'name': 'Carina', 'abbr': 'Car',
         'lines': [[0,1],[1,4]]},
    # Vela
    84: {'name': 'Vela', 'abbr': 'Vel',
         'lines': [[2,3],[3,9],[9,10]]},
    # Puppis
    67: {'name': 'Puppis', 'abbr': 'Pup',
         'lines': [[5,15],[15,17],[17,18]]},
    # Grus
    38: {'name': 'Grus', 'abbr': 'Gru',
         'lines': [[0,1],[0,2]]},
    # Pavo
    60: {'name': 'Pavo', 'abbr': 'Pav',
         'lines': [[0,1]]},
    # Sagittae
    75: {'name': 'Sagitta', 'abbr': 'Sge',
         'lines': [[2,3]]},
    # Lupus
    49: {'name': 'Lupus', 'abbr': 'Lup',
         'lines': [[0,1],[1,2],[0,3],[3,4]]},
    # Triangulum Australe
    79: {'name': 'Triangulum Australe', 'abbr': 'TrA',
         'lines': [[0,1],[1,2],[2,0]]},
    # Phoenix
    63: {'name': 'Phoenix', 'abbr': 'Phe',
         'lines': [[0,1],[1,2]]},
    # Hydra (head)
    41: {'name': 'Hydra', 'abbr': 'Hya',
         'lines': [[4,5],[5,0]]},
    # Ara - Southern altar arc
    5: {'name': 'Ara', 'abbr': 'Ara',
        'lines': [[5,1],[1,2],[2,3],[0,7],[5,6]]},
    # Cetus - Whale body + head
    19: {'name': 'Cetus', 'abbr': 'Cet',
         'lines': [[0,2],[1,5],[5,7],[7,8],[7,6],[6,18],[18,1]]},
    # Columba - Dove
    23: {'name': 'Columba', 'abbr': 'Col',
         'lines': [[0,1]]},
    # Dorado - Swordfish
    32: {'name': 'Dorado', 'abbr': 'Dor',
         'lines': [[0,1]]},
    # Hydrus - Water snake triangle
    42: {'name': 'Hydrus', 'abbr': 'Hyi',
         'lines': [[0,1],[1,2],[2,0]]},
    # Indus - Indian
    43: {'name': 'Indus', 'abbr': 'Ind',
         'lines': [[0,1]]},
    # Musca - Southern fly
    55: {'name': 'Musca', 'abbr': 'Mus',
         'lines': [[0,1],[0,3],[3,2],[2,10]]},
    # Pisces - Two fish connected
    66: {'name': 'Pisces', 'abbr': 'Psc',
         'lines': [[6,0],[0,2]]},
    # Reticulum - Reticle
    69: {'name': 'Reticulum', 'abbr': 'Ret',
         'lines': [[0,1]]},
    # Serpens - Snake (head: alp-bet, body: alp-del-eps)
    73: {'name': 'Serpens', 'abbr': 'Ser',
         'lines': [[0,1],[0,3],[3,4],[4,11],[5,13]]},
    # Volans - Flying fish
    86: {'name': 'Volans', 'abbr': 'Vol',
         'lines': [[1,2]]},
    # Piscis Austrinus - labeled only (1 star: Fomalhaut)
    65: {'name': 'Piscis Austrinus', 'abbr': 'PsA',
         'lines': []},
}

# Cross-constellation lines: [(cons1,bayer1, cons2,bayer2), ...]
_CROSS_CONSTELLATION_LINES = [
    (0, 0, 61, 1),   # Alpheratz (And) - Scheat (Peg) = Great Square side
    (0, 0, 61, 2),   # Alpheratz (And) - Algenib (Peg) = Great Square side
    (77, 1, 7, 0),   # Alnath (Tau) shared with Auriga
]

# Approximate spectral type for named stars (for color rendering on sky chart)
# Sources: SIMBAD, Yale BSC, Hipparcos spectral classifications
_STAR_SPECTRAL = {
    # M-class (red): T < 3700K
    'Betelgeuse': 'M', 'Antares': 'M', 'Mirach': 'M', 'Menkab': 'M',
    'Gacrux': 'M', 'Scheat': 'M', 'Rasalgethi': 'M', 'Mira': 'M',
    # K-class (orange): 3700-5200K
    'Aldebaran': 'K', 'Arcturus': 'K', 'Pollux': 'K', 'Dubhe': 'K',
    'Kochab': 'K', 'Unukalhai': 'K', 'Albireo': 'K', 'Almach': 'K',
    'Hamal': 'K', 'Diphda': 'K', 'Eltanin': 'K', 'Enif': 'K',
    'Suhail': 'K', 'Avior': 'K', 'Kaus Media': 'K', 'Atria': 'K',
    'Alsephina': 'K', 'Yed Post': 'K', 'Cebalrai': 'K', 'Schedar': 'K',
    'Rastaban': 'K', 'Kocab': 'K', 'Rasalhague': 'K',
    # G-class (yellow): 5200-6000K -- Sun-like
    'Capella': 'G', 'Vindemiatrix': 'G', 'Sadalsuud': 'G', 'Sadalmelik': 'G',
    'Mebsuta': 'G', 'Nashira': 'G', 'Kraz': 'G',
    # F-class (yellow-white): 6000-7500K
    'Procyon': 'F', 'Canopus': 'F', 'Polaris': 'F', 'Mirfak': 'F',
    'Wezen': 'F', 'Sargas': 'F', 'Turais': 'F', 'Miaplacidus': 'F',
    'Porrima': 'F', 'Zubenelgenubi': 'F', 'Caph': 'F', 'Phakt': 'F',
    # A-class (white): 7500-10000K
    'Sirius': 'A', 'Vega': 'A', 'Altair': 'A', 'Deneb': 'A',
    'Fomalhaut': 'A', 'Castor': 'A', 'Mizar': 'A', 'Alphecca': 'A',
    'Markab': 'A', 'Algenib': 'A', 'Denebola': 'A',
    'Alioth': 'A', 'Alkaid': 'A', 'Merak': 'A', 'Phecda': 'A',
    'Megrez': 'A', 'Algol': 'A', 'Alhena': 'A', 'Menkent': 'A',
    'Aspidiske': 'A', 'Alphard': 'A', 'Thuban': 'A', 'Sabik': 'A',
    'Zubeneschamali': 'A', 'Ascella': 'A', 'Cor Caroli': 'A',
    'Al Nair': 'A', 'Mothallah': 'A',
    # B-class (blue-white): 10000-30000K
    'Rigel': 'B', 'Spica': 'B', 'Regulus': 'B', 'Achernar': 'B',
    'Mimosa': 'B', 'Hadar': 'B', 'Acrux': 'B', 'Bellatrix': 'B',
    'Alnilam': 'B', 'Alnitak': 'B', 'Shaula': 'B', 'Adhara': 'B',
    'Saiph': 'B', 'Mintaka': 'B', 'Peacock': 'B', 'Navi': 'B',
    "Al Na'ir": 'B', 'Nunki': 'B', 'Kaus Australis': 'B',
    'Dschubba': 'B', 'Aludra': 'B', 'Mirzam': 'B', 'Alpheratz': 'B',
    'Sheratan': 'B', 'Alnath': 'B', 'Wazn': 'B', 'Pherkab': 'B',
    'Algieba': 'K', 'Tejat': 'M',
    # O-class (blue): > 30000K
    'Naos': 'O', 'Regor': 'O',
}


def get_skychart_stars(catalog_dir: str) -> list:
    """Parse ALL stars from stars.h with full data for sky chart rendering.

    Returns list of dicts: {name, ra, dec, mag, cons, bayer, spectral}
    """
    stars_list = []
    stars_file = os.path.join(catalog_dir, "data", "stars.h")

    if not os.path.exists(stars_file):
        return stars_list

    try:
        with open(stars_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Extract names
        names_match = re.search(r'Cat_Stars_Names=(.*?);\s*const', content, re.DOTALL)
        names = []
        if names_match:
            all_text = ''.join(re.findall(r'"([^"]+)"', names_match.group(1)))
            names = [n.strip() for n in all_text.split(';') if n.strip()]

        # Extract all star structs
        star_pattern = (
            r'\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*\d+\s*,\s*\d+\s*,'
            r'\s*([-\d]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\}'
        )
        matches = re.findall(star_pattern, content)

        name_idx = 0
        for has_name, cons, bayer, mag_str, ra_str, dec_str in matches:
            name = None
            spectral = None
            if int(has_name) == 1 and name_idx < len(names):
                name = names[name_idx]
                name_idx += 1
                spectral = _STAR_SPECTRAL.get(name)

            mag = int(mag_str) / 100.0
            stars_list.append({
                'n': name,          # name (None if unnamed)
                'r': round(float(ra_str), 5),   # RA hours
                'd': round(float(dec_str), 4),   # Dec degrees
                'm': round(mag, 2),              # magnitude
                'c': int(cons),     # constellation index
                'b': int(bayer),    # Bayer/Flamsteed index
                's': spectral,      # spectral type letter
            })

    except Exception as e:
        _logger.warning("Error parsing stars for sky chart: %s", e)

    return stars_list


def get_skychart_dsos(catalog_dir: str) -> list:
    """Parse Messier, Caldwell, NGC, and IC DSOs with types for sky chart.

    NGC/IC objects with unknown magnitudes (mag=9990) are excluded.
    Deduplication removes NGC/IC entries that overlap Messier/Caldwell by position.

    Returns list of dicts: {name, common, ra, dec, mag, type}
    """
    dsos = []

    # --- Messier ---
    messier_file = os.path.join(catalog_dir, "data", "messier.h")
    if os.path.exists(messier_file):
        try:
            with open(messier_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            names_match = re.search(
                r'Cat_Messier_Names=(.*?);\s*const char \*Cat_Messier_SubId',
                content, re.DOTALL
            )
            m_names = []
            if names_match:
                m_names = [n.rstrip(';').strip()
                           for n in re.findall(r'"([^"]+)"', names_match.group(1))]

            pattern = (
                r'\{\s*\d+\s*,\s*\d+\s*,\s*(\d+)\s*,\s*(\d+)\s*,'
                r'\s*(\d+)\s*,\s*(\d+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\}'
            )
            matches = re.findall(pattern, content)

            for i, (obj_type, dup, obj_id, mag_str, ra_str, dec_str) in enumerate(matches):
                if int(dup) == 1 and int(obj_type) == 19:
                    continue  # skip duplicates
                mag = int(mag_str) / 100.0
                if mag > 90:
                    mag = 12.0  # unknown magnitude -> faint
                common = m_names[i] if i < len(m_names) else None
                type_code = int(obj_type)
                dsos.append({
                    'n': f"M{obj_id}",
                    'cn': common,
                    'r': round(float(ra_str), 5),
                    'd': round(float(dec_str), 4),
                    'm': round(mag, 1),
                    't': _DSO_TYPE_NAMES.get(type_code, 'Unknown'),
                })
        except Exception as e:
            _logger.warning("Error parsing Messier for sky chart: %s", e)

    # --- Caldwell ---
    caldwell_file = os.path.join(catalog_dir, "data", "caldwell.h")
    if os.path.exists(caldwell_file):
        try:
            with open(caldwell_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            names_match = re.search(
                r'Cat_Caldwell_Names=(.*?);\s*\n', content, re.DOTALL
            )
            c_names = []
            if names_match:
                c_names = [n.rstrip(';').strip()
                           for n in re.findall(r'"([^"]+)"', names_match.group(1))]

            pattern = (
                r'\{\s*\d+\s*,\s*\d+\s*,\s*(\d+)\s*,\s*(\d+)\s*,'
                r'\s*(\d+)\s*,\s*(\d+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\}'
            )
            matches = re.findall(pattern, content)

            for i, (obj_type, dup, obj_id, mag_str, ra_str, dec_str) in enumerate(matches):
                if int(dup) == 1 and int(obj_type) == 19:
                    continue
                mag = int(mag_str) / 100.0
                if mag > 90:
                    mag = 12.0
                common = c_names[i] if i < len(c_names) else None
                type_code = int(obj_type)
                dsos.append({
                    'n': f"C{obj_id}",
                    'cn': common,
                    'r': round(float(ra_str), 5),
                    'd': round(float(dec_str), 4),
                    'm': round(mag, 1),
                    't': _DSO_TYPE_NAMES.get(type_code, 'Unknown'),
                })
        except Exception as e:
            _logger.warning("Error parsing Caldwell for sky chart: %s", e)

    # Build position set for deduplication (Messier/Caldwell already added)
    # Round RA to 2 decimal places (~0.5 min), Dec to 1 (~6 arcmin)
    existing_pos = set()
    for d in dsos:
        existing_pos.add((round(d['r'], 2), round(d['d'], 1)))

    # Types to skip: 2=Star, 3=Double Star, 4=Other, 19=Duplicate
    _SKIP_TYPES = {2, 3, 4, 19}

    # --- NGC ---
    ngc_file = os.path.join(catalog_dir, "data", "ngc.h")
    if os.path.exists(ngc_file):
        try:
            with open(ngc_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            pattern = (
                r'\{\s*\d+\s*,\s*\d+\s*,\s*(\d+)\s*,\s*(\d+)\s*,'
                r'\s*(\d+)\s*,\s*(\d+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\}'
            )
            matches = re.findall(pattern, content)

            for obj_type, dup, obj_id, mag_str, ra_str, dec_str in matches:
                type_code = int(obj_type)
                if type_code in _SKIP_TYPES:
                    continue
                mag = int(mag_str) / 100.0
                if mag > 90:
                    continue  # skip unknown magnitude NGC objects
                ra_val = round(float(ra_str), 5)
                dec_val = round(float(dec_str), 4)
                # Deduplicate against Messier/Caldwell
                pos_key = (round(ra_val, 2), round(dec_val, 1))
                if pos_key in existing_pos:
                    continue
                existing_pos.add(pos_key)
                dsos.append({
                    'n': f"N{obj_id}",
                    'cn': None,
                    'r': ra_val,
                    'd': dec_val,
                    'm': round(mag, 1),
                    't': _DSO_TYPE_NAMES.get(type_code, 'Galaxy'),
                })
        except Exception as e:
            _logger.warning("Error parsing NGC for sky chart: %s", e)

    # --- IC ---
    ic_file = os.path.join(catalog_dir, "data", "ic.h")
    if os.path.exists(ic_file):
        try:
            with open(ic_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            pattern = (
                r'\{\s*\d+\s*,\s*\d+\s*,\s*(\d+)\s*,\s*(\d+)\s*,'
                r'\s*(\d+)\s*,\s*(\d+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\}'
            )
            matches = re.findall(pattern, content)

            for obj_type, dup, obj_id, mag_str, ra_str, dec_str in matches:
                type_code = int(obj_type)
                if type_code in _SKIP_TYPES:
                    continue
                mag = int(mag_str) / 100.0
                if mag > 90:
                    continue  # skip unknown magnitude IC objects
                ra_val = round(float(ra_str), 5)
                dec_val = round(float(dec_str), 4)
                pos_key = (round(ra_val, 2), round(dec_val, 1))
                if pos_key in existing_pos:
                    continue
                existing_pos.add(pos_key)
                dsos.append({
                    'n': f"I{obj_id}",
                    'cn': None,
                    'r': ra_val,
                    'd': dec_val,
                    'm': round(mag, 1),
                    't': _DSO_TYPE_NAMES.get(type_code, 'Galaxy'),
                })
        except Exception as e:
            _logger.warning("Error parsing IC for sky chart: %s", e)

    return dsos


def get_skychart_constellation_data(stars_list: list) -> Tuple[list, list]:
    """Resolve constellation stick-figure lines from star data.

    Args:
        stars_list: Output of get_skychart_stars().

    Returns:
        (lines, labels) where:
        - lines: [[ra1,dec1,ra2,dec2], ...]
        - labels: [{'name','abbr','ra','dec'}, ...]
    """
    # Build lookup: (cons, bayer) -> (ra, dec)
    lookup: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for s in stars_list:
        key = (s['c'], s['b'])
        # Keep the brightest star for each (cons, bayer) pair
        if key not in lookup or s['m'] < lookup[key][2]:
            lookup[key] = (s['r'], s['d'], s['m'])

    lines = []
    labels = []

    for cons_id, info in _CONSTELLATION_PATTERNS.items():
        con_points = []
        for b1, b2 in info['lines']:
            p1 = lookup.get((cons_id, b1))
            p2 = lookup.get((cons_id, b2))
            if p1 and p2:
                lines.append([p1[0], p1[1], p2[0], p2[1]])
                con_points.extend([(p1[0], p1[1]), (p2[0], p2[1])])

        if con_points:
            avg_ra = sum(p[0] for p in con_points) / len(con_points)
            avg_dec = sum(p[1] for p in con_points) / len(con_points)
            labels.append({
                'name': info['name'],
                'abbr': info['abbr'],
                'ra': round(avg_ra, 3),
                'dec': round(avg_dec, 2),
            })
        elif not info['lines']:
            # Constellation with no lines (e.g., single star) — use brightest star as label pos
            brightest = None
            for s in stars_list:
                if s['c'] == cons_id:
                    if brightest is None or s['m'] < brightest['m']:
                        brightest = s
            if brightest:
                labels.append({
                    'name': info['name'],
                    'abbr': info['abbr'],
                    'ra': brightest['r'],
                    'dec': brightest['d'],
                })

    # Cross-constellation lines
    for c1, b1, c2, b2 in _CROSS_CONSTELLATION_LINES:
        p1 = lookup.get((c1, b1))
        p2 = lookup.get((c2, b2))
        if p1 and p2:
            lines.append([p1[0], p1[1], p2[0], p2[1]])

    return lines, labels


def get_skychart_extended_stars(catalog_dir: str) -> list:
    """Parse extended star catalog (24K+ stars from KStars database).

    Returns list of compact dicts: {r: RA_hours, d: Dec_deg, m: mag, s: spectral}
    Deduplicates against positions that would overlap the primary 408 stars.
    """
    ext_stars = []
    ext_file = os.path.join(catalog_dir, "data", "extended_stars.h")
    if not os.path.exists(ext_file):
        return ext_stars

    try:
        with open(ext_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Extract the float array entries: {ra, dec, mag100}
        pattern = r'\{([-\d.]+),([-\d.]+),([-\d]+)\}'
        matches = re.findall(pattern, content)

        # Extract spectral type string
        spec_match = re.search(r'Cat_ExtStars_Spec\[.*?\]\s*=\s*"([^"]+)"', content)
        spec_str = spec_match.group(1) if spec_match else ''

        for i, (ra_str, dec_str, mag_str) in enumerate(matches):
            mag = int(mag_str) / 100.0
            spec = spec_str[i] if i < len(spec_str) else None
            ext_stars.append({
                'r': round(float(ra_str), 4),
                'd': round(float(dec_str), 3),
                'm': round(mag, 2),
                's': spec,
            })
    except Exception as e:
        _logger.warning("Error parsing extended stars for sky chart: %s", e)

    return ext_stars


def get_skychart_data(catalog_dir: str = "catalogs") -> dict:
    """Build the complete sky chart dataset.

    Returns dict with: stars, ext_stars, dsos, planets, constellation_lines,
    constellation_labels.
    """
    stars = get_skychart_stars(catalog_dir)
    dsos = get_skychart_dsos(catalog_dir)
    ext_stars = get_skychart_extended_stars(catalog_dir)
    con_lines, con_labels = get_skychart_constellation_data(stars)

    # Solar system
    planets = []
    try:
        ss = get_solar_system_objects()
        display = [
            ("Sun", "sun", "G"), ("Moon", "moon", None),
            ("Mercury", "mercury", None), ("Venus", "venus", None),
            ("Mars", "mars", "K"), ("Jupiter", "jupiter", None),
            ("Saturn", "saturn", None), ("Uranus", "uranus", None),
            ("Neptune", "neptune", None),
        ]
        for disp, key, spec in display:
            if key in ss:
                ra, dec = ss[key]
                planets.append({
                    'n': disp,
                    'r': round(ra, 4),
                    'd': round(dec, 2),
                    's': spec,
                })
    except Exception:
        pass

    return {
        'stars': stars,
        'ext_stars': ext_stars,
        'dsos': dsos,
        'planets': planets,
        'con_lines': con_lines,
        'con_labels': con_labels,
    }


if __name__ == "__main__":
    # Test
    catalogs = load_all_catalogs()
    print(f"\nSample objects loaded:")
    for name, (ra, dec) in list(catalogs.items())[:10]:
        print(f"  {name}: RA={ra:.4f}h, Dec={dec:.2f}deg")

    # Test solar system
    print(f"\nSolar system objects:")
    ss = get_solar_system_objects()
    for name, (ra, dec) in ss.items():
        if name in ("sun", "moon", "mercury", "venus", "mars",
                     "jupiter", "saturn", "uranus", "neptune"):
            print(f"  {name:10s}: RA={ra:7.4f}h, Dec={dec:+8.4f}deg")

