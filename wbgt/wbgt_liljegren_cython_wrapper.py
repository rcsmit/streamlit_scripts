# version = "20260605120000"


# Wrapper om WBGT_Liljegren aan te roepen met intuïtieve scalaire eenheden
# (°C, %, m/s, W/m², lat/lon in graden, datetime).

# Interne stappen:
#   1. Bereken cosz (instantaan) via wbgt_coszenith_from_cython.cosz()
#   2. Bereken cosza en coszda (intervalgemiddelden) voor fdir
#   3. Bereken fdir via wbgt_liljegren_from_cython.fdir()
#   4. Roep WBGT_Liljegren() aan
#   5. Retourneer WBGT in °C


# Original Cython code by Qinqin Kong (07-04-2021)
# Kong, Qinqin, and Matthew Huber. 
# “Explicit Calculations of Wet Bulb Globe Temperature Compared with 
# Approximations and Why It Matters for Labor Productivity.” 
# Earth’s Future, January 31, 2022. https://doi.org/10.1029/2021EF002334
#

import math
import numpy as np
from datetime import datetime


try:
    import wbgt_liljegren_from_cython as lilj
    import wbgt_coszenith_from_cython as coszmod
except:
    import show_knmi_functions.wbgt_liljegren_from_cython as lilj
    import show_knmi_functions.wbgt_coszenith_from_cython as coszmod
    
# Standaard KNMI-stationscoördinaten (station → (lat, lon, hoogte_m))
KNMI_STATIONS: dict[int, tuple[float, float, float]] = {
    210: (52.165,  4.419,  -0.2),   # Valkenburg
    235: (52.924,  4.781,   1.2),   # De Kooy
    240: (52.318,  4.790,  -3.3),   # Schiphol
    242: (53.241,  4.897,  10.8),   # Vlieland
    249: (52.644,  4.979,   2.4),   # Berkhout
    251: (53.392,  5.346,   0.7),   # Hoorn (Terschelling)
    257: (52.506,  5.747,   1.3),   # Wijk aan Zee
    258: (52.047,  5.177,   1.9),   # De Bilt (hoofdstation)
    260: (52.100,  5.180,   2.0),   # De Bilt (uurdata)
    265: (52.130,  5.274,   1.9),   # Soesterberg
    267: (52.898,  5.384,  -1.3),   # Stavoren
    269: (52.458,  5.520,  -3.7),   # Lelystad
    270: (53.224,  5.752,   1.2),   # Leeuwarden
    273: (52.703,  5.888,  -3.3),   # Marknesse
    275: (52.056,  5.888,  48.2),   # Deelen
    277: (53.413,  6.200,   2.9),   # Lauwersoog
    278: (52.435,  6.259,   3.6),   # Heino
    279: (52.750,  6.574,  15.8),   # Hoogeveen
    280: (53.125,  6.585,   5.2),   # Eelde
    283: (52.644,  6.657,  29.1),   # Hupsel
    286: (53.198,  7.150,   0.0),   # Nieuw Beerta
    290: (52.274,  6.891,  34.8),   # Twenthe
    310: (51.442,  3.596,   8.0),   # Vlissingen
    319: (51.226,  3.861,   1.7),   # Westdorpe
    323: (51.527,  3.884,   1.4),   # Wilhelminadorp
    330: (51.992,  4.122,  11.9),   # Hoek van Holland
    340: (51.449,  4.342,  19.2),   # Woensdrecht
    344: (51.962,  4.447,  -4.3),   # Rotterdam
    348: (51.971,  4.926,  -0.7),   # Cabauw
    350: (51.566,  4.936,  14.9),   # Gilze-Rijen
    356: (51.859,  5.146,   7.5),   # Herwijnen
    370: (51.451,  5.377,  22.6),   # Eindhoven
    375: (51.659,  5.707,  26.8),   # Volkel
    377: (51.198,  5.762,  30.0),   # Ell
    380: (50.906,  5.762, 114.3),   # Maastricht
    391: (51.499,  6.197,  19.5),   # Arcen
}

def wbgt_liljegren_from_station_cython(
    temp_c: float,
    rh_pct: float,
    wind_ms: float,
    q_wm2: float,
    stn: int,
    dt: datetime,
    pressure_hpa: float = 1013.25,
    fdir_mode: str = "knmi_obs",
) -> float:
    """Zelfde als wbgt_liljegren maar met KNMI-stationnummer als locatie-input.

    Bereken WBGT (°C) via de Liljegren-methode op basis van
    menselijk-leesbare invoerwaarden. Dit is een wrapper voor de code vanuit cython

    Parameters
    ----------
        temp_c       : luchttemperatuur (°C)
        rh_pct       : relatieve vochtigheid (%)
        wind_ms      : windsnelheid (m/s)
        q_wm2        : neerwaartse kortgolvige straling aan het oppervlak (W/m²)
        stn:       KNMI-stationnummer (bijv. 260 voor De Bilt).
        dt           : datum en tijdstip (timezone-naïef, UTC verondersteld)
        pressure_hpa : luchtdruk (hPa); standaard 1013.25
        fdir_mode: Zie wbgt_liljegren(); default 'knmi_obs'.

    Returns
    -------
        float  : WBGT (°C)

    Raises:
        KeyError: als het stationnummer niet bekend is.
    """
    if stn not in KNMI_STATIONS:
        raise KeyError(
            f"Stationnummer {stn} niet in KNMI_STATIONS. "
            f"Bekende stations: {sorted(KNMI_STATIONS.keys())}"
        )
    lat, lon, _ = KNMI_STATIONS[stn]
    return wbgt_liljegren_from_cython(temp_c, rh_pct, wind_ms, q_wm2, lat, lon, dt, pressure_hpa)


def wbgt_liljegren_from_cython(
    temp_c: float,
    rh_pct: float,
    wind_ms: float,
    q_wm2: float,
    lat: float,
    lon: float,
    dt: datetime,
    pressure_hpa: float = 1013.25,
    interval_h: float = 1.0,
    is2mwind: bool = True,
) -> float:
    """
    Bereken WBGT (°C) via de Liljegren-methode op basis van
    menselijk-leesbare invoerwaarden. Dit is een wrapper voor de code vanuit cython

    Parameters
    ----------
    temp_c       : luchttemperatuur (°C)
    rh_pct       : relatieve vochtigheid (%)
    wind_ms      : windsnelheid (m/s)
    q_wm2        : neerwaartse kortgolvige straling aan het oppervlak (W/m²)
    lat          : breedtegraad (decimale graden, N positief)
    lon          : lengtegraad (decimale graden, O positief)
    dt           : datum en tijdstip (timezone-naïef, UTC verondersteld)
    pressure_hpa : luchtdruk (hPa); standaard 1013.25
    interval_h   : tijdsinterval voor cosza/coszda (uren); standaard 1
    is2mwind     : True = windmeting op 2 m, False = 10 m; standaard True

    Returns
    -------
    float  : WBGT (°C)
    """
    # --- eenheden omzetten ------------------------------------------------
    tas_K  = temp_c + 273.15
    ps_Pa  = pressure_hpa * 100.0
    lat_r  = math.radians(lat)
    lon_r  = math.radians(lon)

    # numpy datetime64 (nanoseconde-resolutie) voor de coszenith-functies
    date_np = np.array([np.datetime64(dt.isoformat())], dtype='datetime64[ns]')

    # lat/lon moeten 2D zijn (Y, X) voor de coszenith-functies
    lat2d = np.array([[lat_r]])
    lon2d = np.array([[lon_r]])

    # --- cosinus-zenithoek ------------------------------------------------
    cosz_val  = coszmod.cosz(date_np, lat2d, lon2d)          # (1, 1, 1)
    cosza_val = coszmod.cosza(date_np, lat2d, lon2d, interval_h)
    coszda_val= coszmod.coszda(date_np, lat2d, lon2d, interval_h)

    # --- fdir: aandeel directe straling -----------------------------------
    rsds_arr = np.array([[[q_wm2]]])                          # (1, 1, 1)
    fdir_val = lilj.fdir(cosza_val, coszda_val, rsds_arr, date_np)

    # --- WBGT Liljegren ---------------------------------------------------
    wbgt_K = lilj.WBGT_Liljegren(
        tas      = np.array([[[tas_K]]]),
        hurs     = np.array([[[rh_pct]]]),
        ps       = np.array([[[ps_Pa]]]),
        sfcwind  = np.array([[[wind_ms]]]),
        rsds     = rsds_arr,
        fdir_arr = fdir_val,
        cosz_arr = cosz_val,
        is2mwind = is2mwind,
    )

    return float(wbgt_K) - 273.15