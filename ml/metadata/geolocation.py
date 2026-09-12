import math
from typing import Optional, Tuple

# Official NOAA Thunder Bay National Marine Sanctuary coordinates for benchmark shipwrecks
SHIPWRECK_REGISTRY = {
    "EB_Allen": (45.0315, -83.1878, "Lake Huron / Thunder Bay Sanctuary"),
    "DR_Hanna": (45.0841, -83.0845, "Lake Huron / Thunder Bay Sanctuary"),
    "Montana": (45.0682, -83.1812, "Lake Huron / Thunder Bay Sanctuary"),
    "Pewabic": (45.0392, -83.2514, "Lake Huron / Thunder Bay Sanctuary"),
    "Grecian": (45.1965, -83.2185, "Lake Huron / Thunder Bay Sanctuary"),
    "Isaac_M_Scott": (45.0645, -83.0422, "Lake Huron / Thunder Bay Sanctuary"),
    "Lucinda_van_Valkenburg": (45.0543, -83.2081, "Lake Huron / Thunder Bay Sanctuary"),
    "Viator": (45.1124, -83.3105, "Lake Huron / Thunder Bay Sanctuary"),
    "Monohansett": (45.0321, -83.1954, "Lake Huron / Thunder Bay Sanctuary"),
    "Monrovia": (44.9125, -83.0182, "Lake Huron / Thunder Bay Sanctuary"),
    "WP_Thew": (45.0456, -83.2871, "Lake Huron / Thunder Bay Sanctuary"),
    "Corsican": (45.1325, -83.2451, "Lake Huron / Thunder Bay Sanctuary")
}

def lookup_shipwreck_coordinates(filename_or_site: str) -> Optional[Tuple[float, float, str]]:
    """
    Lookup official sanctuary coordinates if a known shipwreck site is identified.
    """
    for site_name, coords in SHIPWRECK_REGISTRY.items():
        if site_name.lower() in filename_or_site.lower():
            return coords
    return None

def local_xy_to_wgs84(
    x_m: float, 
    y_m: float, 
    origin_lat: float = 45.0000, 
    origin_lon: float = -83.0000
) -> Tuple[float, float]:
    """
    Project local cartesian metric coordinates (Easting/Northing in meters)
    to WGS84 Latitude and Longitude via flat-Earth approximation (valid for local <10 km AUV tracks).
    """
    # 1 degree latitude ~ 111,139 meters
    # 1 degree longitude ~ 111,139 * cos(lat) meters
    lat_rad = math.radians(origin_lat)
    delta_lat = y_m / 111139.0
    delta_lon = x_m / (111139.0 * math.cos(lat_rad))
    return round(origin_lat + delta_lat, 6), round(origin_lon + delta_lon, 6)

