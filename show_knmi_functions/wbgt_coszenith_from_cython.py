# version = "20260604120000"
# Description:
# calculate:
#     cosine zenith angle:
#     instantaneous cosine zenith angle: cosz
#     average cosine zenith angle during each interval (e.g. 3-hourly interval): cosza
#     average cosine zenith angle during only the sunlit part of each interval: coszda

# Reference:
#     Di Napoli, C., Hogan, R. J. & Pappenberger, F. Mean radiant temperature from global-scale numerical weather
#     prediction models. Int J Biometeorol 64, 1233–1245 (2020).

# Original Cython code by Qinqin Kong (07-04-2021)
# Kong, Qinqin, and Matthew Huber. 
# “Explicit Calculations of Wet Bulb Globe Temperature Compared with 
# Approximations and Why It Matters for Labor Productivity.” 
# Earth’s Future, January 31, 2022. https://doi.org/10.1029/2021EF002334
#

# Translated to pure Python by Claude (2026-06-04)

import math
import numpy as np

# Constants
PI = 3.1415926535897932
DECL1 = 0.006918
DECL2 = 0.399912
DECL3 = 0.070257
DECL4 = 0.006758
DECL5 = 0.000907
DECL6 = 0.002697
DECL7 = 0.00148


def _hourangel(hour: float, lon: float) -> float:
    """
    hour: hour of the day in UTC time
    lon: longitude (radian)
    return hour angle (radian)
    """
    lon = lon if lon <= PI else lon - 2 * PI
    ha = (hour - 12) * 15 * PI / 180.0 + lon
    if ha < -PI:
        return ha + 2 * PI
    elif ha >= PI:
        return ha - 2 * PI
    else:
        return ha


def _hstart(h: float, interval: float) -> float:
    """
    h: hour angle (radian)
    interval: length of interval (e.g. 3 for 3-hourly interval)
    return hour angle of the starting point of each interval (radian)
    """
    k = interval / 2.0
    if (h + k * 15 * PI / 180) >= PI:
        return h - k * 15 * PI / 180
    elif (h - k * 15 * PI / 180) < -PI:
        return h - k * 15 * PI / 180 + 2 * PI
    else:
        return h - k * 15 * PI / 180


def _hend(h: float, interval: float) -> float:
    """
    h: hour angle (radian)
    interval: length of interval (e.g. 3 for 3-hourly interval)
    return hour angle of the end point of each interval (radian)
    """
    k = interval / 2.0
    if (h + k * 15 * PI / 180) >= PI:
        return h + k * 15 * PI / 180 - 2 * PI
    elif (h - k * 15 * PI / 180) < -PI:
        return h + k * 15 * PI / 180
    else:
        return h + k * 15 * PI / 180


def _czda(h_start: float, h_end: float, h_sunrise: float, h_sunset: float,
          lat: float, Decl: float, interval: float) -> float:
    """
    h_start: hour angle of the starting point of each interval (radian)
    h_end: hour angle of the end point of each interval (radian)
    h_sunrise: hour angle at sunrise (radian)
    h_sunset: hour angle at sunset (radian)
    lat: latitude (radian)
    Decl: solar declination angle (radian)
    interval: length of interval (e.g. 3 for 3-hourly interval)
    return: cosine zenith angle during only the sunlit part of each interval
    """
    if math.isnan(h_sunrise) and lat * Decl > 0:
        h_min = h_start
        h_max = h_end
        return (math.sin(Decl) * math.sin(lat)
                + math.cos(Decl) * math.cos(lat)
                * (math.sin(h_max) - math.sin(h_min))
                / (interval * 15.0 / 180.0 * PI))
    elif math.isnan(h_sunrise) and lat * Decl < 0:
        return 0.0
    elif ((h_start > h_sunset and h_end < h_sunrise)
          or (h_start < h_sunrise and h_end < h_sunrise)
          or (h_start > h_sunset and h_end > h_sunset)):
        return 0.0
    elif h_start > h_sunset and h_end < 0 and h_end > h_sunrise:
        h_min = h_sunrise
        h_max = h_end
        return (math.sin(Decl) * math.sin(lat)
                + math.cos(Decl) * math.cos(lat)
                * (math.sin(h_max) - math.sin(h_min))
                / (h_max - h_min))
    elif h_start > 0 and h_start < h_sunset and h_end < h_sunrise:
        h_min = h_start
        h_max = h_sunset
        return (math.sin(Decl) * math.sin(lat)
                + math.cos(Decl) * math.cos(lat)
                * (math.sin(h_max) - math.sin(h_min))
                / (h_max - h_min))
    elif h_start > 0 and h_start < h_sunset and h_end < 0 and h_end > h_sunrise:
        h_min1, h_max1 = h_start, h_sunset
        h_min2, h_max2 = h_sunrise, h_end
        return ((math.sin(Decl) * math.sin(lat) * (h_max1 - h_min1)
                 + math.cos(Decl) * math.cos(lat) * (math.sin(h_max1) - math.sin(h_min1))
                 + math.sin(Decl) * math.sin(lat) * (h_max2 - h_min2)
                 + math.cos(Decl) * math.cos(lat) * (math.sin(h_max2) - math.sin(h_min2)))
                / (h_max1 - h_min1 + h_max2 - h_min2))
    else:
        h_min = max(h_sunrise, h_start)
        h_max = min(h_sunset, h_end)
        return (math.sin(Decl) * math.sin(lat)
                + math.cos(Decl) * math.cos(lat)
                * (math.sin(h_max) - math.sin(h_min))
                / (h_max - h_min))


def _cza(h_start: float, h_end: float, lat: float, Decl: float, interval: float) -> float:
    """
    h_start: hour angle of the starting point of each interval (radian)
    h_end: hour angle of the end point of each interval (radian)
    lat: latitude (radian)
    Decl: solar declination angle (radian)
    interval: length of interval (e.g. 3 for 3-hourly interval)
    return: cosine zenith angle during each interval
    """
    if h_start > 0 and h_end < 0:
        h_min1, h_max1 = h_start, PI
        h_min2, h_max2 = -PI, h_end
        return ((math.sin(Decl) * math.sin(lat) * (h_max1 - h_min1)
                 + math.cos(Decl) * math.cos(lat) * (math.sin(h_max1) - math.sin(h_min1))
                 + math.sin(Decl) * math.sin(lat) * (h_max2 - h_min2)
                 + math.cos(Decl) * math.cos(lat) * (math.sin(h_max2) - math.sin(h_min2)))
                / (h_max1 - h_min1 + h_max2 - h_min2))
    else:
        h_min = h_start
        h_max = h_end
        return (math.sin(Decl) * math.sin(lat)
                + math.cos(Decl) * math.cos(lat)
                * (math.sin(h_max) - math.sin(h_min))
                / (h_max - h_min))


def _declination(doy: float, hour: float, tod: float) -> float:
    """Compute solar declination angle (radian) from day-of-year and hour."""
    g = (360.0 / tod) * (doy + hour / 24.0) * (PI / 180.0)
    return (DECL1
            - DECL2 * math.cos(g) + DECL3 * math.sin(g)
            - DECL4 * math.cos(2 * g) + DECL5 * math.sin(2 * g)
            - DECL6 * math.cos(3 * g) + DECL7 * math.sin(3 * g))


def _days_in_year(date_scalar) -> int:
    """Return number of days in the year of a numpy datetime64 scalar."""
    y = str(date_scalar.astype('datetime64[Y]'))
    return int((np.datetime64(y + '-12-31') - np.datetime64(y + '-01-01') + 1)
               / np.timedelta64(1, 'D'))


def cosz(date: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    date: date and time series (numpy datetime64 array, shape (T,))
    lat: latitude in radians (2-D array, shape (Y, X))
    lon: longitude in radians (2-D array, shape (Y, X))
    return: instantaneous cosine zenith angle, shape (T, Y, X)
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    doy = ((date - date.astype('datetime64[Y]')).astype('timedelta64[D]')
           / np.timedelta64(1, 'D'))
    hour = ((date - date.astype('datetime64[D]')).astype('timedelta64[h]')
            / np.timedelta64(1, 'h'))

    T = date.shape[0]
    Y, X = lat.shape
    result = np.zeros((T, Y, X), dtype=np.float64)

    for i in range(T):
        tod = _days_in_year(date[i])
        Decl = _declination(doy[i], hour[i], tod)
        for j in range(Y):
            for k in range(X):
                h = _hourangel(hour[i], lon[j, k])
                result[i, j, k] = (math.sin(Decl) * math.sin(lat[j, k])
                                   + math.cos(Decl) * math.cos(lat[j, k]) * math.cos(h))
    return result


def cosza(date: np.ndarray, lat: np.ndarray, lon: np.ndarray, interval: float) -> np.ndarray:
    """
    date: date and time series (numpy datetime64 array, shape (T,))
    lat: latitude in radians (2-D array, shape (Y, X))
    lon: longitude in radians (2-D array, shape (Y, X))
    interval: length of interval in hours (e.g. 3 for 3-hourly)
    return: average cosine zenith angle during each interval, shape (T, Y, X)
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    doy = ((date - date.astype('datetime64[Y]')).astype('timedelta64[D]')
           / np.timedelta64(1, 'D'))
    hour = ((date - date.astype('datetime64[D]')).astype('timedelta64[h]')
            / np.timedelta64(1, 'h'))

    T = date.shape[0]
    Y, X = lat.shape
    result = np.zeros((T, Y, X), dtype=np.float64)

    for i in range(T):
        tod = _days_in_year(date[i])
        Decl = _declination(doy[i], hour[i], tod)
        for j in range(Y):
            for k in range(X):
                h = _hourangel(hour[i], lon[j, k])
                h_s = _hstart(h, interval)
                h_e = _hend(h, interval)
                result[i, j, k] = _cza(h_s, h_e, lat[j, k], Decl, interval)
    return result


def coszda(date: np.ndarray, lat: np.ndarray, lon: np.ndarray, interval: float) -> np.ndarray:
    """
    date: date and time series (numpy datetime64 array, shape (T,))
    lat: latitude in radians (2-D array, shape (Y, X))
    lon: longitude in radians (2-D array, shape (Y, X))
    interval: length of interval in hours (e.g. 3 for 3-hourly)
    return: average cosine zenith angle during only the sunlit period of each interval,
            shape (T, Y, X)
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    doy = ((date - date.astype('datetime64[Y]')).astype('timedelta64[D]')
           / np.timedelta64(1, 'D'))
    hour = ((date - date.astype('datetime64[D]')).astype('timedelta64[h]')
            / np.timedelta64(1, 'h'))

    T = date.shape[0]
    Y, X = lat.shape
    result = np.zeros((T, Y, X), dtype=np.float64)

    for i in range(T):
        tod = _days_in_year(date[i])
        Decl = _declination(doy[i], hour[i], tod)
        for j in range(Y):
            for k in range(X):
                h = _hourangel(hour[i], lon[j, k])
                h_s = _hstart(h, interval)
                h_e = _hend(h, interval)
                cos_arg = -math.tan(Decl) * math.tan(lat[j, k])
                if abs(cos_arg) > 1.0:
                    # Polar day or night: acos undefined → use NaN to signal polar day
                    h_sunrise = float('nan')
                    h_sunset = float('nan')
                else:
                    h_sunrise = -math.acos(cos_arg)
                    h_sunset = math.acos(cos_arg)
                result[i, j, k] = _czda(h_s, h_e, h_sunrise, h_sunset,
                                        lat[j, k], Decl, interval)
    return result
