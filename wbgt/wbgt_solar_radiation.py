"""
Solar position and clear-sky radiation calculator.
No external dependencies — only Python stdlib (math, datetime).

Based on:
- Spencer (1971) for solar declination
- Iqbal (1983) / Bird & Hulstrom (1981) for clear-sky GHI
"""

# version : 20260527-120000 - Initial version: solar position + clear-sky radiation
# version : 20260527-121000 - Replaced dataclass return with plain dict
current_version = "20260527-121000"

import math
from datetime import datetime, timezone


def solar_radiation(
    dt: datetime,
    lat: float,
    lon: float,
    pressure_hpa: float = 1013.25,
    temperature_c: float = 15.0,
    turbidity: float = 2.5,
) -> dict:
    """
    Bereken zonnepositie en clear-sky straling voor een gegeven locatie en tijdstip.

    Parameters
    ----------
    dt : datetime
        Tijdstip (timezone-aware of naive wordt behandeld als UTC).
    lat : float
        Breedtegraad in decimale graden (-90 t/m 90, N positief).
    lon : float
        Lengtegraad in decimale graden (-180 t/m 180, O positief).
    pressure_hpa : float
        Luchtdruk in hPa (default: 1013.25 = zeeniveau).
    temperature_c : float
        Temperatuur in °C (voor refractiecorrectie, default: 15).
    turbidity : float
        Linke turbiditeitscoëfficiënt (2 = zeer helder, 3 = gemiddeld, 5 = wazig).

    Returns
    -------
    dict met keys:
        datetime_utc, latitude, longitude,
        solar_elevation_deg, solar_azimuth_deg, zenith_deg,
        extraterrestrial_irradiance,
        clear_sky_ghi, clear_sky_dni, clear_sky_dhi,
        day_of_year, equation_of_time_min,
        solar_noon_utc, sunrise_utc, sunset_utc
    """
    # --- Tijdsconversie naar UTC decimaal uur ---
    if dt.tzinfo is not None:
        dt_utc = dt.astimezone(timezone.utc).replace(tzinfo=None)
    else:
        dt_utc = dt  # behandel als UTC

    doy = dt_utc.timetuple().tm_yday
    utc_hour = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0

    # --- Zonne-geometrie (Spencer 1971) ---
    B = math.radians((360 / 365) * (doy - 1))

    eot = 229.18 * (
        0.000075
        + 0.001868 * math.cos(B)
        - 0.032077 * math.sin(B)
        - 0.014615 * math.cos(2 * B)
        - 0.04089 * math.sin(2 * B)
    )

    declination_deg = math.degrees(
        0.006918
        - 0.399912 * math.cos(B)
        + 0.070257 * math.sin(B)
        - 0.006758 * math.cos(2 * B)
        + 0.000907 * math.sin(2 * B)
        - 0.002697 * math.cos(3 * B)
        + 0.00148 * math.sin(3 * B)
    )
    decl = math.radians(declination_deg)
    lat_rad = math.radians(lat)

    time_offset = eot / 60.0 + lon / 15.0
    solar_time = utc_hour + time_offset
    hour_angle_deg = (solar_time - 12.0) * 15.0
    ha = math.radians(hour_angle_deg)

    sin_elev = (
        math.sin(lat_rad) * math.sin(decl)
        + math.cos(lat_rad) * math.cos(decl) * math.cos(ha)
    )
    sin_elev = max(-1.0, min(1.0, sin_elev))
    elevation_rad = math.asin(sin_elev)
    elevation_deg = math.degrees(elevation_rad)

    # Refractiecorrectie (Bennett 1982)
    if elevation_deg > 0.0:
        elev_corr = elevation_deg + 10.3 / (elevation_deg + 5.11)
        refraction = (
            pressure_hpa / 1013.25
            * (283.0 / (273.0 + temperature_c))
            * 1.02
            / (60.0 * math.tan(math.radians(elev_corr)))
        )
        elevation_deg += refraction

    zenith_deg = 90.0 - elevation_deg

    # Azimut (graden vanaf Noord, met de klok mee)
    cos_az = (
        math.sin(decl) - math.sin(lat_rad) * sin_elev
    ) / (math.cos(lat_rad) * math.cos(elevation_rad) + 1e-10)
    cos_az = max(-1.0, min(1.0, cos_az))
    azimuth_deg = math.degrees(math.acos(cos_az))
    if hour_angle_deg > 0:
        azimuth_deg = 360.0 - azimuth_deg

    # --- Zonsopkomst / zonsondergang ---
    cos_omega0 = -math.tan(lat_rad) * math.tan(decl)
    solar_noon_utc = 12.0 - time_offset

    if cos_omega0 < -1.0:
        sunrise_utc = sunset_utc = None  # middernachtzon
    elif cos_omega0 > 1.0:
        sunrise_utc = sunset_utc = None  # poolnacht
    else:
        omega0_deg = math.degrees(math.acos(cos_omega0))
        sunrise_utc = solar_noon_utc - omega0_deg / 15.0
        sunset_utc = solar_noon_utc + omega0_deg / 15.0

    # --- Extraterrestrische straling ---
    E0 = (1.000110 + 0.034221 * math.cos(B) + 0.001280 * math.sin(B)
          + 0.000719 * math.cos(2 * B) + 0.000077 * math.sin(2 * B))
    I_ext = 1361.0 * E0

    if elevation_deg <= 0.0:
        return {
            "datetime_utc": dt_utc,
            "latitude": lat,
            "longitude": lon,
            "solar_elevation_deg": round(elevation_deg, 3),
            "solar_azimuth_deg": round(azimuth_deg, 3),
            "zenith_deg": round(zenith_deg, 3),
            "extraterrestrial_irradiance": round(I_ext, 2),
            "clear_sky_ghi": 0.0,
            "clear_sky_dni": 0.0,
            "clear_sky_dhi": 0.0,
            "day_of_year": doy,
            "equation_of_time_min": round(eot, 3),
            "solar_noon_utc": round(solar_noon_utc, 4),
            "sunrise_utc": round(sunrise_utc, 4) if sunrise_utc is not None else None,
            "sunset_utc": round(sunset_utc, 4) if sunset_utc is not None else None,
        }

    # --- Clear-sky straling (Bird & Hulstrom 1981) ---
    zenith_rad = math.radians(zenith_deg)
    cos_z = math.cos(zenith_rad)

    air_mass = 1.0 / (cos_z + 0.50572 * (96.07995 - zenith_deg) ** (-1.6364))
    air_mass_pressure = air_mass * pressure_hpa / 1013.25

    tau_r = math.exp(-0.0903 * air_mass_pressure ** 0.84 * (
        1.0 + air_mass_pressure - air_mass_pressure ** 1.01
    ))

    ozone_cm = 0.35
    tau_o = 1.0 - (
        0.1611 * ozone_cm * air_mass * (1.0 + 139.48 * ozone_cm * air_mass) ** (-0.3035)
        - 0.002715 * ozone_cm * air_mass / (
            1.0 + 0.044 * ozone_cm * air_mass + 0.0003 * (ozone_cm * air_mass) ** 2
        )
    )

    water_cm = 1.5
    tau_w = 1.0 - 2.4959 * water_cm * air_mass / (
        (1.0 + 79.034 * water_cm * air_mass) ** 0.6828 + 6.385 * water_cm * air_mass
    )

    beta = max(0.02, turbidity * 0.04 - 0.02)
    tau_a = math.exp(-beta * air_mass ** 0.873 * (1.0 + beta - beta ** 0.7088) * 0.547)

    dni = 0.9751 * I_ext * tau_r * tau_o * tau_w * tau_a
    ghi_direct = dni * cos_z

    tau_total = tau_r * tau_o * tau_w * tau_a
    dhi = max(0.0, I_ext * cos_z * (0.2710 - 0.2939 * tau_total) * 0.79
              / (1.0 - air_mass + air_mass ** 1.02))
    ghi = ghi_direct + dhi

    return {
        "datetime_utc": dt_utc,
        "latitude": lat,
        "longitude": lon,
        "solar_elevation_deg": round(elevation_deg, 3),
        "solar_azimuth_deg": round(azimuth_deg, 3),
        "zenith_deg": round(zenith_deg, 3),
        "extraterrestrial_irradiance": round(I_ext, 2),
        "clear_sky_ghi": round(max(0.0, ghi), 2),
        "clear_sky_dni": round(max(0.0, dni), 2),
        "clear_sky_dhi": round(dhi, 2),
        "day_of_year": doy,
        "equation_of_time_min": round(eot, 3),
        "solar_noon_utc": round(solar_noon_utc, 4),
        "sunrise_utc": round(sunrise_utc, 4) if sunrise_utc is not None else None,
        "sunset_utc": round(sunset_utc, 4) if sunset_utc is not None else None,
    }


# ---------------------------------------------------------------------------
# Gebruiksvoorbeeld
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = solar_radiation(
        dt=datetime(2026, 5, 27, 10, 0, 0),
        lat=16.047,
        lon=108.206,
    )
    for k, v in result.items():
        print(f"{k:<35} {v}")
