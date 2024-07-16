# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""Provides utilities for geographic information system (GIS) operations, 
including coordinate transformations, EPSG code handling, UTM zone conversions, 
and validation of geographic data."""

import numpy as np
from .._gofastlog import gofastlog
from ..decorators import Deprecated ,CheckGDALData # noqa

from ..exceptions import GISError # noqa
try : 
    from ._set_gdal import HAS_GDAL, EPSG_DICT, NEW_GDAL # noqa
    if HAS_GDAL:
        from osgeo import osr
        from osgeo.ogr import OGRERR_NONE # noqa
    else:
        import pyproj # noqa
except :
    HAS_GDAL =False
    pass 

_logger = gofastlog.get_gofast_logger(__name__)


__all__= [
    'adjust_time_component',
     'assert_lat_lon_values',
     'assert_xy_coordinate_system',
     'convert_lat_lon_to_utm',
     'convert_position_float2str',
     'convert_position_str2float',
     'convert_position_str2float2',
     'convert_utm_to_lat_lon',
     'get_utm_string_from_sr',
     'get_utm_zone_2',
     'll_to_utm',
     'utm_to_ll',
     'validate_elevation'
 ]

_deg2rad = np.pi / 180.0
_rad2deg = 180.0 / np.pi
_equatorial_radius = 2
_eccentricity_squared = 3

_ellipsoid = [
    #  id, Ellipsoid name, Equatorial Radius, square of eccentricity
    # first once is a placeholder only, To allow array indices to match id
    # numbers
    [-1, "Placeholder", 0, 0],
    [1, "Airy", 6377563, 0.00667054],
    [2, "Australian National", 6378160, 0.006694542],
    [3, "Bessel 1841", 6377397, 0.006674372],
    [4, "Bessel 1841 (Nambia] ", 6377484, 0.006674372],
    [5, "Clarke 1866", 6378206, 0.006768658],
    [6, "Clarke 1880", 6378249, 0.006803511],
    [7, "Everest", 6377276, 0.006637847],
    [8, "Fischer 1960 (Mercury] ", 6378166, 0.006693422],
    [9, "Fischer 1968", 6378150, 0.006693422],
    [10, "GRS 1967", 6378160, 0.006694605],
    [11, "GRS 1980", 6378137, 0.00669438],
    [12, "Helmert 1906", 6378200, 0.006693422],
    [13, "Hough", 6378270, 0.00672267],
    [14, "International", 6378388, 0.00672267],
    [15, "Krassovsky", 6378245, 0.006693422],
    [16, "Modified Airy", 6377340, 0.00667054],
    [17, "Modified Everest", 6377304, 0.006637847],
    [18, "Modified Fischer 1960", 6378155, 0.006693422],
    [19, "South American 1969", 6378160, 0.006694542],
    [20, "WGS 60", 6378165, 0.006693422],
    [21, "WGS 66", 6378145, 0.006694542],
    [22, "WGS-72", 6378135, 0.006694318],
    [23, "WGS-84", 6378137, 0.00669438]
]
# http://spatialreference.org/ref/epsg/28350/proj4/
epsg_dict = {
    28350: ['+proj=utm +zone=50 +south +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs', 50],
    28351: ['+proj=utm +zone=51 +south +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs', 51],
    28352: ['+proj=utm +zone=52 +south +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs', 52],
    28353: ['+proj=utm +zone=53 +south +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs', 53],
    28354: ['+proj=utm +zone=54 +south +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs', 54],
    28355: ['+proj=utm +zone=55 +south +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs', 55],
    28356: ['+proj=utm +zone=56 +south +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs', 56],
    3112: [
        '+proj=lcc +lat_1=-18 +lat_2=-36 +lat_0=0 +lon_0=134 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs',
        0],
    4326: ['+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs', 0],
    32609: ['+proj=utm +zone=9 +ellps=WGS84 +datum=WGS84 +units=m +no_defs', 9],
    32610: ['+proj=utm +zone=10 +ellps=WGS84 +datum=WGS84 +units=m +no_defs', 10],
    32611: ['+proj=utm +zone=11 +ellps=WGS84 +datum=WGS84 +units=m +no_defs', 11],
    32612: ['+proj=utm +zone=12 +ellps=WGS84 +datum=WGS84 +units=m +no_defs', 12],
    32613: ['+proj=utm +zone=13 +ellps=WGS84 +datum=WGS84 +units=m +no_defs', 13],
    32614: ['+proj=utm +zone=14 +ellps=WGS84 +datum=WGS84 +units=m +no_defs', 14],
    32615: ['+proj=utm +zone=15 +ellps=WGS84 +datum=WGS84 +units=m +no_defs', 15],
    32616: ['+proj=utm +zone=16 +ellps=WGS84 +datum=WGS84 +units=m +no_defs', 16],
    32617: ['+proj=utm +zone=17 +ellps=WGS84 +datum=WGS84 +units=m +no_defs', 17],
    32618: ['+proj=utm +zone=18 +ellps=WGS84 +datum=WGS84 +units=m +no_defs', 18],
    32619: ['+proj=utm +zone=19 +ellps=WGS84 +datum=WGS84 +units=m +no_defs', 19]
}


def ll_to_utm(reference_ellipsoid, lat, lon):
    """
    Convert latitude and longitude to UTM coordinates.

    Parameters
    ----------
    reference_ellipsoid : int
        The index of the reference ellipsoid from the `_ellipsoid` list.
    lat : float or str
        Latitude in decimal degrees or DD:MM:SS.ms format.
    lon : float or str
        Longitude in decimal degrees or DD:MM:SS.ms format.

    Returns
    -------
    tuple
        UTM zone, easting, and northing coordinates.

    Examples
    --------
    >>> ll_to_utm(23, 42.0, -71.0)
    ('18T', 321875.5348340456, 4654642.366709269)

    >>> ll_to_utm(1, "51:28:38.5", "-0:0:24.6")
    ('30U', 699745.948, 5706456.731)
    """
    a = _ellipsoid[reference_ellipsoid][_equatorial_radius]
    ecc_squared = _ellipsoid[reference_ellipsoid][_eccentricity_squared]
    k0 = 0.9996
    lat, lon = _convert_lat_lon(lat, lon)
    long_temp = _normalize_longitude(lon)

    lat_rad = lat * _deg2rad
    long_rad = long_temp * _deg2rad

    zone_number = _calculate_zone_number(long_temp, lat)

    if _is_svalbard(lat):
        zone_number = _calculate_svalbard_zone(long_temp)

    long_origin = _calculate_long_origin(zone_number)
    long_origin_rad = long_origin * _deg2rad

    utm_zone = _get_utm_zone(zone_number, lat)

    ecc_prime_squared = ecc_squared / (1 - ecc_squared)
    N = _calculate_N(a, ecc_squared, lat_rad)
    T = np.tan(lat_rad) ** 2
    C = ecc_prime_squared * np.cos(lat_rad) ** 2
    A = np.cos(lat_rad) * (long_rad - long_origin_rad)
    M = _calculate_M(a, ecc_squared, lat_rad)
    utm_easting = _calculate_easting(k0, N, A, T, C, ecc_prime_squared)
    utm_northing = _calculate_northing(k0, M, N, lat_rad, A, T, C, ecc_prime_squared, lat)

    if lat < 0:
        utm_northing += 10000000.0

    return utm_zone, utm_easting, utm_northing


def _convert_lat_lon(lat, lon):
    """
    Convert latitude and longitude from string representations to float 
    values if needed.

    Parameters:
        lat (float or str): Latitude in decimal degrees or DD:MM:SS.ms format.
        lon (float or str): Longitude in decimal degrees or DD:MM:SS.ms format.

    Returns:
        tuple: Tuple containing the converted latitude and longitude as
        float values.
    """
    if isinstance(lat, str):
        lat = convert_position_str2float(lat)
    if isinstance(lon, str):
        lon = convert_position_str2float(lon)
    return lat, lon

def _normalize_longitude(lon):
    """
    Normalize a longitude value to the range -180.00 to 179.9 degrees.

    Parameters:
        lon (float): Longitude value in degrees.

    Returns:
        float: The normalized longitude value.
    """
    return (lon + 180) - int((lon + 180) / 360) * 360 - 180

def _calculate_zone_number(long_temp, lat):
    """
    Calculate the UTM zone number based on the longitude and latitude.

    Parameters:
        long_temp (float): Normalized longitude value.
        lat (float): Latitude value in degrees.

    Returns:
        int: The UTM zone number.
    """

    zone_number = int((long_temp + 180) / 6) + 1
    if 56.0 <= lat < 64.0 and 3.0 <= long_temp < 12.0:
        zone_number = 32
    return zone_number

def _is_svalbard(lat):
    """
    Check if the given latitude is within the Svalbard region.

    Parameters:
        lat (float): Latitude value in degrees.

    Returns:
        bool: True if the latitude is within the Svalbard region, otherwise False.
    """
    return 72.0 <= lat < 84.0

def _calculate_svalbard_zone(long_temp):
    """
    Calculate the UTM zone number for the Svalbard region based on the longitude.

    Parameters:
        long_temp (float): Normalized longitude value.

    Returns:
        int: The UTM zone number for the Svalbard region.
    """
    if 0.0 <= long_temp < 9.0:
        return 31
    elif 9.0 <= long_temp < 21.0:
        return 33
    elif 21.0 <= long_temp < 33.0:
        return 35
    elif 33.0 <= long_temp < 42.0:
        return 37


def _calculate_long_origin(zone_number):
    """
    Calculate the longitude of the UTM zone's origin.

    Parameters:
        zone_number (int): UTM zone number.

    Returns:
        float: The longitude of the UTM zone's origin.
    """
    return (zone_number - 1) * 6 - 180 + 3

def _get_utm_zone(zone_number, lat):
    """
    Generate the UTM zone designation.

    Parameters:
        zone_number (int): UTM zone number.
        lat (float): Latitude value in degrees.

    Returns:
        str: The UTM zone designation.
    """
    utm_zone = "%d%c" % (zone_number, _utm_letter_designator(lat))
    return utm_zone

def _calculate_N(a, ecc_squared, lat_rad):
    """
    Calculate the radius of curvature in the prime vertical (N) for a given ellipsoid.

    Parameters:
        a (float): Equatorial radius of the ellipsoid.
        ecc_squared (float): Square of eccentricity of the ellipsoid.
        lat_rad (float): Latitude in radians.

    Returns:
        float: The radius of curvature in the prime vertical (N).
    """
    return a / np.sqrt(1 - ecc_squared * np.sin(lat_rad) ** 2)


def _calculate_M(a, ecc_squared, lat_rad):
    """
    Calculate the meridional arc length (M) for a given ellipsoid.

    Parameters:
        a (float): Equatorial radius of the ellipsoid.
        ecc_squared (float): Square of eccentricity of the ellipsoid.
        lat_rad (float): Latitude in radians.

    Returns:
        float: The meridional arc length (M).
    """
    return a * (
        (1 - ecc_squared / 4 - 3 * ecc_squared ** 2 / 64 - 5 * ecc_squared ** 3 / 256) * lat_rad
        - (3 * ecc_squared / 8 + 3 * ecc_squared ** 2 / 32 + 45 * ecc_squared ** 3 / 1024) * np.sin(2 * lat_rad)
        + (15 * ecc_squared ** 2 / 256 + 45 * ecc_squared ** 3 / 1024) * np.sin(4 * lat_rad)
        - (35 * ecc_squared ** 3 / 3072) * np.sin(6 * lat_rad)
    )

def _calculate_easting(k0, N, A, T, C, ecc_prime_squared):
    """
    Calculate the UTM easting coordinate.

    Parameters:
        k0 (float): Scale factor.
        N (float): Radius of curvature in the prime vertical.
        A (float): Difference in longitude from the UTM zone's origin.
        T (float): Tangent squared of latitude.
        C (float): Square of eccentricity prime.
        ecc_prime_squared (float): Square of eccentricity prime.

    Returns:
        float: The UTM easting coordinate.
    """
    return (
        k0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T ** 2 + 72 * C - 58 * ecc_prime_squared) * A ** 5 / 120)
        + 500000.0
    )

def _calculate_northing(k0, M, N, lat_rad, A, T, C, ecc_prime_squared, lat):
    """
    Calculate the UTM northing coordinate.

    Parameters:
        k0 (float): Scale factor.
        M (float): Meridional arc length.
        N (float): Radius of curvature in the prime vertical.
        lat_rad (float): Latitude in radians.
        A (float): Difference in longitude from the UTM zone's origin.
        T (float): Tangent squared of latitude.
        C (float): Square of eccentricity prime.
        ecc_prime_squared (float): Square of eccentricity prime.
        lat (float): Latitude value in degrees.

    Returns:
        float: The UTM northing coordinate.
    """
    return (
        k0
        * (M + N * np.tan(lat_rad) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24 + (61 - 58 * T + T ** 2 + 600 * C - 330 * ecc_prime_squared) * A ** 6 / 720))
        if lat >= 0
        else k0
        * (M + N * np.tan(lat_rad) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24 + (61 - 58 * T + T ** 2 + 600 * C - 330 * ecc_prime_squared) * A ** 6 / 720))
        + 10000000.0
    )


def _utm_letter_designator2(lat):
    """
    Determine the UTM letter designator for a given latitude.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees.

    Returns
    -------
    str
        UTM letter designator.

    Example
    -------
    >>> _utm_letter_designator(42.0)
    'T'

    >>> _utm_letter_designator(-23.5)
    'J'
    """
    if -80 <= lat <= 84:
        return "CDEFGHJKLMNPQRSTUVWX"[int((lat + 80) / 8)]
    else:
        return ""


def _get_zone_letter(zone):
    """
    Get the zone letter from the UTM zone.

    Parameters:
        zone (str): UTM zone in the format '##X' or '##Y'.

    Returns:
        str: The zone letter ('N' for Northern Hemisphere, 'S' for Southern Hemisphere).
    """
    zone_letter = zone[-1]
    return zone_letter

def _is_northern_hemisphere(zone_letter):
    """
    Check if the point is in the Northern Hemisphere based on the UTM zone letter.

    Parameters:
        zone_letter (str): UTM zone letter.

    Returns:
        int: 1 for the Northern Hemisphere, 0 for the Southern Hemisphere.
    """
    return 1 if zone_letter >= 'N' else 0

def _remove_offset_for_southern_hemisphere(northing, is_northern_hemisphere):
    """
    Remove the offset for the Southern Hemisphere from the northing value.

    Parameters:
        northing (float): Northing value.
        is_northern_hemisphere (int): 1 for the Northern Hemisphere, 0 for 
        the Southern Hemisphere.

    Returns:
        float: Adjusted northing value.
    """
    if is_northern_hemisphere == 0:
        northing -= 10000000.0
    return northing

def _compute_longitude_origin(zone_number):
    """
    Compute the longitude origin for the UTM zone.

    Parameters:
        zone_number (int): UTM zone number.

    Returns:
        float: Longitude origin in degrees.
    """
    return (zone_number - 1) * 6 - 180 + 3

# import numpy as np

# # Global dictionary for ellipsoid parameters (placeholder, fill with actual values)
# _ellipsoid = {
#     'reference_ellipsoid_name': {'_equatorial_radius': 0, '_eccentricity_squared': 0}
# }

# _rad2deg = 180.0 / np.pi

def _calculate_eccentricity_prime_squared(ecc_squared):
    """
    Calculate the prime squared eccentricity.

    Parameters
    ----------
    ecc_squared : float
        Eccentricity squared of the ellipsoid.

    Returns
    -------
    float
        Prime squared eccentricity.

    Example
    -------
    >>> _calculate_eccentricity_prime_squared(0.00669438)
    0.006739496742276437
    """
    return ecc_squared / (1 - ecc_squared)

def _calculate_mu(y, k0, a, ecc_squared):
    """
    Calculate the mu parameter used in lat/lon calculations.

    Parameters
    ----------
    y : float
        Northing.
    k0 : float
        Scale factor.
    a : float
        Equatorial radius of the ellipsoid.
    ecc_squared : float
        Eccentricity squared of the ellipsoid.

    Returns
    -------
    float
        Mu parameter.

    Example
    -------
    >>> _calculate_mu(5000000, 0.9996, 6378137, 0.00669438)
    44.88888896130864
    """
    M = y / k0
    return M / (a * (1 - ecc_squared / 4 - 3 * ecc_squared ** 2 / 64 - 5 * ecc_squared ** 3 / 256))

def _calculate_footprint_latitude(mu, e1):
    """
    Calculate the footprint latitude.

    Parameters
    ----------
    mu : float
        Mu parameter.
    e1 : float
        E1 parameter calculated from eccentricity.

    Returns
    -------
    float
        Footprint latitude in radians.

    Example
    -------
    >>> _calculate_footprint_latitude(44.88888896130864, 0.0016792203863838463)
    0.7853981633974483
    """
    phi1_rad = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu)
                + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
                + (151 * e1 ** 3 / 96) * np.sin(6 * mu))
    return phi1_rad

def _calculate_lat_lon(phi1_rad, ecc_squared, n1, t1, c1, r1, d, long_origin):
    """
    Calculate latitude and longitude from intermediate parameters.

    Parameters
    ----------
    phi1_rad : float
        Footprint latitude in radians.
    ecc_squared : float
        Eccentricity squared of the ellipsoid.
    n1 : float
        N1 parameter.
    t1 : float
        T1 parameter.
    c1 : float
        C1 parameter.
    r1 : float
        R1 parameter.
    d : float
        D parameter.
    long_origin : float
        Longitude origin of the UTM zone.

    Returns
    -------
    tuple
        Latitude and longitude in decimal degrees.

    Example
    -------
    >>> _calculate_lat_lon(0.7853981633974483, 0.00669438, 6399593.625758449, 0.0, 0.0, 6367449.145823415, 500000, -177)
    (0.0, -177.0)
    """
    # Calculate ecc_prime_squared based on ecc_squared
    ecc_prime_squared = ecc_squared / (1 - ecc_squared)

    lat = phi1_rad - (n1 * np.tan(phi1_rad) / r1) * (
        d ** 2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * ecc_prime_squared) * d ** 4 / 24
        + (
            61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * ecc_prime_squared - 3 * c1 ** 2) * d ** 6 / 720)
    lat = lat * _rad2deg

    lon = (d - (1 + 2 * t1 + c1) * d ** 3 / 6 + (
        5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * ecc_prime_squared + 24 * t1 ** 2)
           * d ** 5 / 120) / np.cos(phi1_rad)
    lon = long_origin + lon * _rad2deg
    return lat, lon

def utm_to_ll(reference_ellipsoid, northing, easting, zone):
    """
    Converts UTM coordinates to latitude and longitude.

    Parameters
    ----------
    reference_ellipsoid : str
        Name of the reference ellipsoid.
    northing : float
        Northing coordinate.
    easting : float
        Easting coordinate.
    zone : str
        UTM zone.

    Returns
    -------
    tuple
        Latitude and longitude in decimal degrees.

    Example
    -------
    >>> utm_to_ll('WGS84', 5000000, 400000, '33N')
    (0.0, 16.0)
    """
    k0 = 0.9996
    a = _ellipsoid[reference_ellipsoid][_equatorial_radius]
    ecc_squared = _ellipsoid[reference_ellipsoid][_eccentricity_squared]
    e1 = (1 - np.sqrt(1 - ecc_squared)) / (1 + np.sqrt(1 - ecc_squared))

    x = easting - 500000.0  # Remove 500,000 meter offset for longitude.
    y = northing
    zone_letter = zone[-1]
    NorthernHemisphere = 1 if zone_letter >= 'N' else 0
    if not NorthernHemisphere:
        y -= 10000000.0  # Remove offset for southern hemisphere.

    long_origin = (int(zone[:-1]) - 1) * 6 - 180 + 3
    ecc_prime_squared = _calculate_eccentricity_prime_squared(ecc_squared)
    mu = _calculate_mu(y, k0, a, ecc_squared)
    phi1_rad = _calculate_footprint_latitude(mu, e1)

    n1 = a / np.sqrt(1 - ecc_squared * np.sin(phi1_rad) ** 2)
    t1 = np.tan(phi1_rad) ** 2
    c1 = ecc_prime_squared * np.cos(phi1_rad) ** 2
    r1 = a * (1 - ecc_squared) / np.power(1 - ecc_squared * np.sin(phi1_rad) ** 2, 1.5)
    d = x / (n1 * k0)

    return _calculate_lat_lon(phi1_rad, ecc_squared, n1, t1, c1, r1, d, long_origin)


def assert_xy_coordinate_system(x, y):
    """
    Determine the coordinate system of the given x and y coordinates.

    Parameters
    ----------
    x, y : array_like
        1D arrays of position coordinates x and y.

    Returns
    -------
    str
        Coordinate system identified ('utm', 'dms', or 'll').

    Examples
    --------
    >>> x, y = np.random.rand(7), np.arange(7)
    >>> assert_xy_coordinate_system(x, y)
    'll'
    >>> x, y = np.random.rand(7), np.random.rand(7) * 180
    >>> assert_xy_coordinate_system(x, y)
    'utm'
    >>> x = ['28:24:43.08', '28:24:42.69', '28:24:42.31']
    >>> y = ['109:19:58.34', '109:19:58.93', '109:19:59.51']
    >>> assert_xy_coordinate_system(x, y)
    'dms'
    """
    def _is_dms(coords):
        """Check if the given coordinates are in DMS format."""
        return all(':' in str(coord) for coord in coords)

    x, y = np.array(x), np.array(y)  # Ensure numpy array for consistency

    if _is_dms(x) or _is_dms(y):
        return 'dms'
    elif ((x < 90).all() and (y < 180).all()) or ((x < 180).all() and (y < 90).all()):
        return 'll'
    else:
        return 'utm'

def convert_position_str2float(position_str: str) -> float:
    """
    Converts a position from DMS (Degrees:Minutes:Seconds) format to decimal degrees.

    Parameters
    ----------
    position_str : str
        The position in DMS format, e.g., "118:34:56.3".

    Returns
    -------
    float
        The position in decimal degrees.

    Raises
    ------
    ValueError
        If the input string is not in the correct DMS format.

    Example
    -------
    >>> convert_position_str2float("118:34:56.3")
    118.58230555555556
    """
    parts = position_str.split(':')
    if len(parts) != 3:
        raise ValueError(f'{position_str} is not in correct DMS format. Expected DD:MM:SS')

    degrees, minutes, seconds = map(float, parts)
    sign = -1 if degrees < 0 else 1
    return sign * (abs(degrees) + minutes / 60 + seconds / 3600)

def assert_lat_lon_values(value: float, value_type: str = 'latitude') -> float:
    """
    Validates and ensures latitude or longitude values are within acceptable 
    ranges.

    Parameters
    ----------
    value : float
        The latitude or longitude value to validate.
    value_type : str, optional
        Specifies the type of value being validated ('latitude' or 'longitude'). 
        Default is 'latitude'.

    Returns
    -------
    float
        The validated latitude or longitude value.

    Raises
    ------
    ValueError
        If the value is outside the acceptable range for latitudes (-90 to 90)
        or longitudes (-180 to 180).

    Example
    -------
    >>> assert_lat_lon_values(45.0, 'latitude')
    45.0
    >>> assert_lat_lon_values(-190.0, 'longitude')
    ValueError: |Longitude| > 180, unacceptable!
    """
    if value_type == 'latitude' and not -90 <= value <= 90:
        raise ValueError('|Latitude| > 90, unacceptable!')
    elif value_type == 'longitude' and not -180 <= value <= 180:
        raise ValueError('|Longitude| > 180, unacceptable!')
    return value

def convert_position_float2str(position: float) -> str:
    """
    Converts a position from decimal degrees to DMS (Degrees:Minutes:Seconds) 
    format.

    Parameters
    ----------
    position : float
        The position in decimal degrees to convert.

    Returns
    -------
    str
        The position in DMS format.

    Example
    -------
    >>> convert_position_float2str(118.58230555555556)
    '118:34:56.30'
    """
    sign = -1 if position < 0 else 1
    position = abs(position)
    degrees = int(position)
    minutes = int((position - degrees) * 60)
    seconds = (position - degrees - minutes / 60) * 3600
    return f"{'-' if sign < 0 else ''}{degrees}:{minutes:02}:{seconds:05.2f}"

def adjust_time_component(deg_or_min, value):
    """
    Adjust the time component for overflow, converting excess minutes or 
    seconds into hours or minutes respectively.

    This function is designed to handle cases where the seconds or minutes 
    exceed 60, indicating an overflow that should be carried over into the
    next higher unit (minutes to hours, or seconds to minutes).

    Parameters
    ----------
    deg_or_min : int
        The current degrees or minutes value, before adjustment.
    value : int or float
        The seconds or minutes value to adjust for overflow. This value is 
        expected to be in the same unit as `deg_or_min`.

    Returns
    -------
    tuple of (int, float)
        A tuple containing two elements:
        1. The adjusted degrees or minutes, incremented if `value` causes an
           overflow.
        2. The remainder of `value` after adjusting for any overflow, ensuring
        it is less than 60.

    Examples
    --------
    >>> adjust_time_component(15, 75)
    (16, 15)

    >>> adjust_time_component(2, 59.5)
    (2, 59.5)

    >>> adjust_time_component(0, 120)
    (2, 0)
    """
    if value >= 60:
        overflow, remainder = divmod(value, 60)
        return (deg_or_min + int(overflow), remainder)
    else:
        return (deg_or_min, value)

def convert_position_str2float2(position_str):
    """
    Convert a position string in DD:MM:SS format to decimal degrees.

    Parameters
    ----------
    position_str : str
        Position in 'DD:MM:SS.ms' format.

    Returns
    -------
    float
        Position in decimal degrees.

    Examples
    --------
    >>> convert_position_str2float('118:34:56.3')
    -118.58230555555556
    """
    if position_str.lower() == 'none':
        return None

    try:
        degrees, minutes, seconds = [float(part) for part in position_str.split(':')]
        degrees, minutes = adjust_time_component(degrees, minutes)
        degrees, seconds = adjust_time_component(degrees, seconds)
        sign = -1 if degrees < 0 else 1
        return sign * (abs(degrees) + minutes / 60 + seconds / 3600)
    except ValueError:
        raise ValueError(f"Invalid format: '{position_str}', expected 'DD:MM:SS.ms'")

def _validate_time_component(value, component_name='value'):
    """
    Validate that a time component (minutes or seconds) is within the valid range [0, 60).

    Parameters
    ----------
    value : float or int
        The time component to validate.
    component_name : str, optional
        The name of the component being validated (e.g., 'minutes', 'seconds').

    Returns
    -------
    float or int
        The validated time component.

    Raises
    ------
    ValueError
        If the time component is not within the valid range.

    Examples
    --------
    >>> _validate_time_component(30, 'minutes')
    30

    >>> _validate_time_component(60, 'seconds')
    ValueError: seconds needs to be < 60 and >= 0, currently 60.
    """
    if not 0 <= value < 60:
        raise ValueError(f'{component_name} needs to be < 60 and >= 0, currently {value:.3f}.')
    return value

# Refactor _assert_minutes and _assert_seconds to use _validate_time_component
_assert_minutes = lambda minutes: _validate_time_component(minutes, 'minutes')
_assert_seconds = lambda seconds: _validate_time_component(seconds, 'seconds')


def validate_elevation(elevation):
    """
    Validate and convert the elevation input to a floating-point number.

    If the input is not a valid number, logs a warning and defaults to 0.0.

    Parameters
    ----------
    elevation : str, int, float
        The elevation value to validate and convert.

    Returns
    -------
    float
        The elevation as a floating-point number. Defaults to 0.0 
        if the input is invalid.

    Examples
    --------
    >>> validate_elevation('123.45')
    123.45

    >>> validate_elevation('not a number')
    0.0
    """
    try:
        return float(elevation)
    except (ValueError, TypeError):
        _logger.warning(f"{elevation} is not a number, setting elevation to 0.0")
        return 0.0
    
@Deprecated("NATO UTM zone is used in other parts of gofast; this function "
            "is for Standard UTM")
def get_utm_string_from_sr(spatialreference):
    """
    Return UTM zone string from a spatial reference instance.

    Parameters
    ----------
    spatialreference : osr.SpatialReference
        GDAL/osr spatial reference object.

    Returns
    -------
    str
        UTM zone string.

    Examples
    --------
    >>> from osgeo import osr
    >>> from gofast.geo.gisutils import get_utm_string_from_sr

    >>> sr = osr.SpatialReference()
    >>> sr.ImportFromEPSG(32633)  # Example EPSG code for UTM zone 33N
    >>> print(get_utm_string_from_sr(sr))
    ```
    """
    zone_number = spatialreference.GetUTMZone()
    if zone_number > 0:
        return f"{zone_number}N"
    elif zone_number < 0:
        return f"{abs(zone_number)}S"
    else:
        return str(zone_number)

def get_utm_zone_2(latitude, longitude):
    """
    Get UTM zone from a given latitude and longitude.

    Parameters
    ----------
    latitude : float
        Latitude of the point.
    longitude : float
        Longitude of the point.

    Returns
    -------
    tuple
        A tuple containing the zone number, a boolean indicating 
        if it is northern, and the UTM zone as a string.

    Examples
    --------
    ```python
    from gofast.geo.gisutils import get_utm_zone_2

    latitude, longitude = 40.7128, -74.0060
    print(get_utm_zone_2(latitude, longitude))
    ```
    """
    zone_number = int(1 + (longitude + 180.0) / 6.0)
    is_northern = latitude >= 0
    n_str = _utm_letter_designator(latitude)

    return zone_number, is_northern, f"{zone_number:02}{n_str}"

def _utm_letter_designator(lat):
    """
    Determine the correct UTM letter designator for the given latitude.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees.

    Returns
    -------
    str
        UTM letter designator.

    Notes
    -----
    Returns 'Z' if latitude is outside the UTM limits of 84N to 80S.
    """
    letter_ranges = {
        range(72, 85): 'X',
        range(64, 72): 'W',
        range(56, 64): 'V',
        range(48, 56): 'U',
        range(40, 48): 'T',
        range(32, 40): 'S',
        range(24, 32): 'R',
        range(16, 24): 'Q',
        range(8, 16): 'P',
        range(0, 8): 'N',
        range(-8, 0): 'M',
        range(-16, -8): 'L',
        range(-24, -16): 'K',
        range(-32, -24): 'J',
        range(-40, -32): 'H',
        range(-48, -40): 'G',
        range(-56, -48): 'F',
        range(-64, -56): 'E',
        range(-72, -64): 'D',
        range(-80, -72): 'C',
    }

    for key_range, letter in letter_ranges.items():
        if lat in key_range:
            return letter

    return 'Z'  # If the Latitude is outside the UTM limits


def _convert_lat_lon_to_decimal(lat, lon):
    """
    Convert latitude and longitude to decimal degrees.

    Parameters
    ----------
    lat : float or str
        Latitude in decimal degrees or DD:MM:SS.ms format.
    lon : float or str
        Longitude in decimal degrees or DD:MM:SS.ms format.

    Returns
    -------
    float
        Latitude and Longitude in decimal degrees.
    """
    if isinstance(lat, str):
        # Convert DD:MM:SS.ms format to decimal degrees
        lat_deg, lat_min, lat_sec = map(float, lat.split(':'))
        lat = lat_deg + lat_min / 60 + lat_sec / 3600

    if isinstance(lon, str):
        # Convert DD:MM:SS.ms format to decimal degrees
        lon_deg, lon_min, lon_sec = map(float, lon.split(':'))
        lon = lon_deg + lon_min / 60 + lon_sec / 3600

    return float(lat), float(lon)

def _set_coordinate_system(datum):
    """
    Set the spatial reference coordinate system based on the datum.

    Parameters
    ----------
    datum : str or int
        Well-known datum (e.g., 'WGS84', 'NAD27') or EPSG code.

    Returns
    -------
    osr.SpatialReference
        Spatial reference coordinate system.
    """
    if isinstance(datum, int):
        spatial_ref = osr.SpatialReference()
        ogrerr = spatial_ref.ImportFromEPSG(datum)
        if ogrerr != osr.OGRERR_NONE:
            raise Exception("GDAL/osgeo ogr error code: {}".format(ogrerr))
    elif isinstance(datum, str):
        spatial_ref = osr.SpatialReference()
        ogrerr = spatial_ref.SetWellKnownGeogCS(datum)
        if ogrerr != osr.OGRERR_NONE:
            raise Exception("GDAL/osgeo ogr error code: {}".format(ogrerr))
    else:
        raise ValueError("Datum not understood. Use EPSG code or well-known"
                         " datum as a string.")

    return spatial_ref

def _convert_lat_lon_to_utm(lat, lon, datum, utm_zone, epsg=None):
    """
    Convert latitude and longitude to UTM coordinates.

    Parameters
    ----------
    lat : float or str
        Latitude in decimal degrees or DD:MM:SS.ms format.
    lon : float or str
        Longitude in decimal degrees or DD:MM:SS.ms format.
    datum : str or int
        Well-known datum (e.g., 'WGS84', 'NAD27') or EPSG code.
    utm_zone : str
        UTM zone in the form of number and 'N' or 'S' (e.g., '55S').
    epsg : int, optional
        EPSG number defining projection.

    Returns
    -------
    tuple
        Projected UTM point (easting, northing, zone).
    """
    lat, lon = _convert_lat_lon_to_decimal(lat, lon)
    spatial_ref = _set_coordinate_system(datum)

    if epsg is not None:
        utm_spatial_ref = osr.SpatialReference()
        ogrerr = utm_spatial_ref.ImportFromEPSG(epsg)
        if ogrerr != osr.OGRERR_NONE:
            raise Exception("GDAL/osgeo ogr error code: {}".format(ogrerr))
    else:
        utm_spatial_ref = osr.SpatialReference()
        utm_zone_number = int(utm_zone[:-1])
        is_northern = utm_zone[-1] == 'N'
        utm_spatial_ref.SetUTM(utm_zone_number, is_northern)

    transformation = osr.CoordinateTransformation(spatial_ref, utm_spatial_ref)
    utm_point = transformation.TransformPoint(lon, lat)

    return round(utm_point[0], 6), round(utm_point[1], 6), utm_zone

def _convert_utm_to_lat_lon(easting, northing, utm_zone, datum, epsg=None):
    """
    Convert UTM coordinates to latitude and longitude.

    Parameters
    ----------
    easting : float
        Easting coordinate in meters.
    northing : float
        Northing coordinate in meters.
    utm_zone : str
        UTM zone in the form of number and 'N' or 'S' (e.g., '55S').
    datum : str or int
        Well-known datum (e.g., 'WGS84', 'NAD27') or EPSG code.
    epsg : int, optional
        EPSG number defining projection.

    Returns
    -------
    tuple
        Projected latitude and longitude in decimal degrees.
    """
    easting, northing = float(easting), float(northing)
    spatial_ref = _set_coordinate_system(datum)

    if epsg is not None:
        utm_spatial_ref = osr.SpatialReference()
        ogrerr = utm_spatial_ref.ImportFromEPSG(epsg)
        if ogrerr != osr.OGRERR_NONE:
            raise Exception("GDAL/osgeo ogr error code: {}".format(ogrerr))
    else:
        utm_spatial_ref = osr.SpatialReference()
        utm_zone_number = int(utm_zone[:-1])
        is_northern = utm_zone[-1] == 'N'
        utm_spatial_ref.SetUTM(utm_zone_number, is_northern)

    transformation = osr.CoordinateTransformation(utm_spatial_ref, spatial_ref)
    lat_lon_point = transformation.TransformPoint(easting, northing)

    return round(lat_lon_point[1], 6), round(lat_lon_point[0], 6)

def convert_lat_lon_to_utm(lat, lon, datum='WGS84', utm_zone=None, epsg=None):
    """
    Convert latitude and longitude to UTM coordinates.

    Parameters
    ----------
    lat : float or str or list-like
        Latitude in decimal degrees, DD:MM:SS.ms format, or list of latitudes.
    lon : float or str or list-like
        Longitude in decimal degrees, DD:MM:SS.ms format, or list of longitudes.
    datum : str or int, optional
        Well-known datum (e.g., 'WGS84', 'NAD27') or EPSG code.
    utm_zone : str, optional
        UTM zone in the form of number and 'N' or 'S' (e.g., '55S').
        Defaults to the center point of the provided latitudes and longitudes.
    epsg : int, optional
        EPSG number defining projection.

    Returns
    -------
    tuple or np.recarray
        If a single point is provided, returns a tuple (easting, northing, zone).
        If multiple points are provided, returns an np.recarray with fields:
            easting, northing, elev, utm_zone.
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)

    if lat.size != lon.size:
        raise ValueError("Latitude and longitude arrays must have the same length.")

    if lat.size == 1:
        return _convert_lat_lon_to_utm(lat[0], lon[0], datum, utm_zone, epsg)
    else:
        easting = np.empty_like(lat)
        northing = np.empty_like(lat)
        elev = np.empty_like(lat)
        utm_zone_result = np.empty_like(lat, dtype='U4')

        for i in range(lat.size):
            easting[i], northing[i], elev[i], utm_zone_result[i] = _convert_lat_lon_to_utm(
                lat[i], lon[i], datum, utm_zone, epsg)

        return np.rec.array([easting, northing, elev, utm_zone_result], 
                            names=['easting', 'northing', 'elev', 'utm_zone'])

def convert_utm_to_lat_lon(easting, northing, utm_zone, datum='WGS84', epsg=None):
    """
    Convert UTM coordinates to latitude and longitude.

    Parameters
    ----------
    easting : float or list-like
        Easting coordinate in meters or list of easting values.
    northing : float or list-like
        Northing coordinate in meters or list of northing values.
    utm_zone : str or list-like
        UTM zone in the form of number and 'N' or 'S' (e.g., '55S') or list
        of utm_zone values.
    datum : str or int, optional
        Well-known datum (e.g., 'WGS84', 'NAD27') or EPSG code.
    epsg : int, optional
        EPSG number defining projection.

    Returns
    -------
    tuple or np.recarray
        If a single point is provided, returns a tuple (
            latitude, longitude, utm_zone).
        If multiple points are provided, returns an np.recarray with fields:
            latitude, longitude, utm_zone.
    """
    easting = np.asarray(easting)
    northing = np.asarray(northing)
    utm_zone = np.asarray(utm_zone)

    if easting.size != northing.size or easting.size != utm_zone.size:
        raise ValueError("Easting, northing, and utm_zone arrays"
                         " must have the same length.")

    if easting.size == 1:
        return _convert_utm_to_lat_lon(easting[0], northing[0], utm_zone[0],
                                       datum, epsg)
    else:
        latitude = np.empty_like(easting)
        longitude = np.empty_like(easting)

        for i in range(easting.size):
            latitude[i], longitude[i], _ = _convert_utm_to_lat_lon(
                easting[i], northing[i], utm_zone[i], datum, epsg)

        return np.rec.array([latitude, longitude, utm_zone], names=[
            'latitude', 'longitude', 'utm_zone'])


# # Example usage of refactored functions
# if __name__ == "__main__":
#     print(convert_position_str2float("118:34:56.3"))
#     print(assert_lat_lon_values(91, 'latitude'))  # This will raise a ValueError
#     print(convert_position_float2str(118.58230555555556))
# >>> from gofast.geo.utils import convert_lat_lon_to_utm, convert_utm_to_lat_lon

# # Convert latitude and longitude to UTM
# >>> easting, northing, zone = convert_lat_lon_to_utm(37.7749, -122.4194)
# >>> print(easting, northing, zone)
# # Output: 551705.062, 4163838.791, 10S

# # Convert UTM coordinates to latitude and longitude
# >>> lat, lon, utm_zone = convert_utm_to_lat_lon(551705.062, 4163838.791, '10S')
# >>> print(lat, lon, utm_zone)
# # Output: 37.7749, -122.4194, 10S
