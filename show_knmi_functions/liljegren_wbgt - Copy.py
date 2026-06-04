"""
Liljegren WBGT solver — volledige implementatie van Liljegren et al. (2008).

Vervangt de vereenvoudigde _nat_bol_temp() en _globe_temp() functies in
wbgt_knmi.py door iteratieve warmte/massatransfer-oplossers die ook de
directe/diffuse opsplitsing van zonnestraling correct modelleren.

Gebruik:
    from liljegren_wbgt import wbgt_liljegren, solar_zenith_angle

    wbgt = wbgt_liljegren(
        temp_c=30.0, rh_pct=60.0, wind_ms=2.0, q_wm2=800.0,
        lat=52.10, lon=5.18,
        dt=datetime(2025, 7, 15, 13, 0),
        pressure_hpa=1013.25,
    )
"""

# version : 20260528-120000 - Initial version: full Liljegren 2008 WBGT solver with solar geometry
# version : 20260528-130000 - Fix math domain error: guard log(RH=0) and globe T^0.25 against negative base

from __future__ import annotations

import math
from datetime import datetime #, timezone
from typing import Optional
from datetime import datetime, timedelta
current_version = "20260528-130000"

# ---------------------------------------------------------------------------
# Fysische constanten
# ---------------------------------------------------------------------------
SIGMA      = 5.6704e-8   # Stefan-Boltzmann  [W m⁻² K⁻⁴]
S0         = 1367.0      # Zonsconstante      [W m⁻²]
M_WATER    = 0.018016    # Molmassa water     [kg mol⁻¹]
M_AIR      = 0.028966    # Molmassa droge lucht [kg mol⁻¹]
R_GAS      = 8.314472    # Gasconstante       [J mol⁻¹ K⁻¹]

# Wick-geometrie (QUESTemp°34, Liljegren 2008 Fig. 1)
D_WICK     = 0.007       # diameter wick  [m]
L_WICK     = 0.0254      # lengte wick    [m]

# Globe-geometrie
D_GLOBE    = 0.0508      # diameter globe [m]
# D_GLOBE    = 0.15      # diameter globe [m]

# Gekalibreerde albedo/emissiviteiten uit Liljegren (2008), pp. 648–649
EPS_WICK   = 0.95        # emissiviteit wick
ALPHA_WICK = 0.40        # albedo wick
EPS_GLOBE  = 0.95        # emissiviteit globe
ALPHA_GLOBE= 0.05        # albedo globe (zwarte bol)
ALPHA_SFC  = 0.20        # oppervlakte-albedo (incl. instrument-effect)

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


# ---------------------------------------------------------------------------
# Zonnegeometrie
# ---------------------------------------------------------------------------

def _day_of_year(dt: datetime) -> int:
    """Dag van het jaar (1–366)."""
    return dt.timetuple().tm_yday


def _earth_sun_distance(doy: int) -> float:
    """Aarde-zon afstand in AU (vereenvoudigde formule, nauwkeurig op ±0.5%)."""
    B = 2.0 * math.pi * (doy - 1) / 365.0
    return 1.0 / (
        1.000110
        + 0.034221 * math.cos(B)
        + 0.001280 * math.sin(B)
        + 0.000719 * math.cos(2 * B)
        + 0.000077 * math.sin(2 * B)
    ) ** 0.5


def _solar_declination(doy: int) -> float:
    """Zonsdeclinatie in radialen (Spencer 1971)."""
    B = 2.0 * math.pi * (doy - 1) / 365.0
    return (
        0.006918
        - 0.399912 * math.cos(B)
        + 0.070257 * math.sin(B)
        - 0.006758 * math.cos(2 * B)
        + 0.000907 * math.sin(2 * B)
        - 0.002697 * math.cos(3 * B)
        + 0.00148  * math.sin(3 * B)
    )


def _equation_of_time(doy: int) -> float:
    """Tijdsvergelijking in uren."""
    B = 2.0 * math.pi * (doy - 1) / 365.0
    return (
        0.000075
        + 0.001868 * math.cos(B)
        - 0.032077 * math.sin(B)
        - 0.014615 * math.cos(2 * B)
        - 0.04089  * math.sin(2 * B)
    ) * (12.0 / math.pi)


def solar_zenith_angle(dt: datetime, lat_deg: float, lon_deg: float) -> float:
    """Zonzenithoek θ in radialen.

    Args:
        dt:       UTC datetime (timezone-aware of naive wordt als UTC behandeld).
        lat_deg:  Geografische breedte [°N].
        lon_deg:  Geografische lengte  [°E].

    Returns:
        Zonzenithoek θ ∈ [0, π].  θ > π/2 betekent zon onder horizon.
    """
    doy  = _day_of_year(dt)
    decl = _solar_declination(doy)
    eot  = _equation_of_time(doy)

    # Lokale ware zonnetijd in uren
    utc_hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    solar_time = utc_hour + lon_deg / 15.0 + eot

    # Uurhoek [rad]  (0 = solar noon, negatief = ochtend)
    hour_angle = math.radians((solar_time - 12.0) * 15.0)

    lat = math.radians(lat_deg)
    cos_theta = (
        math.sin(lat) * math.sin(decl)
        + math.cos(lat) * math.cos(decl) * math.cos(hour_angle)
    )
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.acos(cos_theta)


# ---------------------------------------------------------------------------
# Luchteigenschappen (geëvalueerd op filmtemperatuur)
# ---------------------------------------------------------------------------

def _saturation_vapor_pressure(T_K: float) -> float:
    """Verzadigingsdampdruk [Pa] via Buck (1981), omgezet naar Kelvin."""
    T_C = T_K - 273.15
    return 100.0 * 6.1121 * math.exp(17.368 * T_C / (238.88 + T_C))


def _air_properties(T_film_K: float, P_Pa: float) -> tuple[float, float, float, float, float, float]:
    """Lucht­eigenschappen op filmtemperatuur.

    Returns:
        (rho, mu, k, cp, Pr, D_wv)
        rho  : dichtheid           [kg m⁻³]
        mu   : dynamische viscositeit [Pa s]
        k    : warmtegeleidingsvermogen [W m⁻¹ K⁻¹]
        cp   : soortelijke warmte  [J kg⁻¹ K⁻¹]
        Pr   : Prandtl-getal       [-]
        D_wv : diffusiviteit waterdamp in lucht [m² s⁻¹]
    """
    # Dichtheid ideaal gas
    rho = P_Pa * M_AIR / (R_GAS * T_film_K)

    # Sutherland-viscositeit
    mu = 1.458e-6 * T_film_K**1.5 / (T_film_K + 110.4)

    # Warmtegeleidingsvermogen (empirisch)
    k = 0.02624 * (T_film_K / 300.0) ** 0.8646

    # Soortelijke warmte [J kg⁻¹ K⁻¹]  (vrijwel constant voor lucht)
    cp = 1005.0 + ((T_film_K - 250.0) / 125.0) ** 2 * 0.0

    # Prandtl
    Pr = cp * mu / k

    # Diffusiviteit waterdamp in lucht (Chapman-Enskog benadering)
    D_wv = 2.471e-5 * (T_film_K / 313.15) ** 1.81 * (101325.0 / P_Pa)

    return rho, mu, k, cp, Pr, D_wv


# ---------------------------------------------------------------------------
# Hulpfuncties: fdir-opsplitsing (Liljegren 2008, Eq. 13–14)
# ---------------------------------------------------------------------------

def _fdir(S: float, theta: float, d_AU: float) -> float:
    """Fractie directe straling in totale horizontale straling.

    Args:
        S:      Totale horizontale straling [W m⁻²].
        theta:  Zonzenithoek [rad].
        d_AU:   Aarde-zon afstand [AU].

    Returns:
        fdir ∈ [0, 1].
    """
    if theta >= math.radians(89.5) or S <= 0.0:
        return 0.0
    S_max = S0 * math.cos(theta) / d_AU**2
    if S_max <= 0.0:
        return 0.0
    S_star = min(S / S_max, 1.0)
    if S_star <= 0.0:
        return 0.0
    # Liljegren Eq. 13
    fd = math.exp(3.0 - 1.34 * S_star - 1.65 / S_star)
    return max(0.0, min(1.0, fd))


# ---------------------------------------------------------------------------
# Globe-temperatuur (Liljegren 2008, Eq. 15–17)
# ---------------------------------------------------------------------------

def _globe_temp_liljegren(
    T_a: float,
    wind_ms: float,
    S: float,
    fdir: float,
    theta: float,
    P_Pa: float,
    eps_a: float,
) -> float:
    """Iteratieve globe-temperatuur solver.

    Args:
        T_a:    Luchttemperatuur [K].
        wind_ms: Windsnelheid [m s⁻¹].
        S:      Totale horizontale straling [W m⁻²].
        fdir:   Directe fractie [-].
        theta:  Zonzenithoek [rad].
        P_Pa:   Luchtdruk [Pa].
        eps_a:  Atmosferische emissiviteit [-].

    Returns:
        Globe-temperatuur [K].
    """
    T_g = T_a  # beginschatting

    for _ in range(100):
        T_film = 0.5 * (T_g + T_a)
        rho, mu, k, cp, Pr, _ = _air_properties(T_film, P_Pa)

        # Convectieve warmteoverdrachtscoëfficiënt — bol (Nusselt, Liljegren Eq. 16)
        Re = rho * max(wind_ms, 0.5) * D_GLOBE / mu
        Nu = 2.0 + 0.6 * Re**0.5 * Pr**(1.0 / 3.0)
        h  = k / D_GLOBE * Nu

        # Stralingsterm (Liljegren Eq. 17, herschikt)
        cos_theta = math.cos(theta) if theta < math.radians(89.5) else 0.0

        # Directe stralingsterm begrenzen: cos(θ) minimaal 0.087 (= 85°)
        # zodat 1/(2*cos) maximaal ~5.7 wordt — fysisch realistisch
        cos_theta_clamped = max(cos_theta, math.cos(math.radians(85.0)))
        # Begrens 1/(2cosθ) tegen de geometrische explosie bij lage zon.
        # cos_theta_clamped = max(cos_theta, math.cos(math.radians(80.0)))

        solar_term = (S / (2.0 * EPS_GLOBE * SIGMA)) * (1.0 - ALPHA_GLOBE) * (
            1.0 + (1.0 / (2.0 * cos_theta_clamped) - 1.0) * fdir + ALPHA_SFC
        )

        T_g_base = (
            0.5 * (1.0 + eps_a) * T_a**4
            - h / (EPS_GLOBE * SIGMA) * (T_g - T_a)
            + solar_term
        )
        T_g_new = max(T_g_base, 200.0**4) ** 0.25  # minimaal 200 K (-73 °C)

        # Relaxatie (Liljegren: 0.9 oud + 0.1 nieuw)
        T_g_next = 0.9 * T_g + 0.1 * T_g_new

        if abs(T_g_next - T_g) < 0.02:
            T_g = T_g_next
            break
        T_g = T_g_next

    return T_g


# ---------------------------------------------------------------------------
# Nat-bol temperatuur (Liljegren 2008, Eq. 9, 12)
# ---------------------------------------------------------------------------

def _nat_bol_temp_liljegren(
    T_a: float,
    RH: float,
    wind_ms: float,
    S: float,
    fdir: float,
    theta: float,
    P_Pa: float,
    eps_a: float,
) -> float:
    """Iteratieve nat-bol temperatuur solver.

    Args:
        T_a:    Luchttemperatuur [K].
        RH:     Relatieve vochtigheid [0–1].
        wind_ms: Windsnelheid [m s⁻¹].
        S:      Totale horizontale straling [W m⁻²].
        fdir:   Directe fractie [-].
        theta:  Zonzenithoek [rad].
        P_Pa:   Luchtdruk [Pa].
        eps_a:  Atmosferische emissiviteit [-].

    Returns:
        Nat-bol temperatuur [K].
    """
    # Beginschatting = dauwpunttemperatuur (benadering)
    # Clamp RH zodat e_a altijd > 0 (voorkomt math domain error bij RH=0)
    RH = max(RH, 0.001)
    e_a = RH * _saturation_vapor_pressure(T_a)
    # Magnus-inversie voor Td
    e_a_safe = max(e_a, 1.0)   # minimaal 1 Pa, voorkomt log(0) of log(negatief)
    ln_e = math.log(e_a_safe / 611.2)
    T_dew = 273.15 + 238.88 * ln_e / (17.368 - ln_e)
    T_w = max(T_dew, T_a - 30.0)

    for _ in range(200):
        T_film = 0.5 * (T_w + T_a)
        rho, mu, k, cp, Pr, D_wv = _air_properties(T_film, P_Pa)

        # Schmidt-getal
        Sc = mu / (rho * D_wv)

        # Convectieve warmte/massaoverdracht — cilinder in dwarsstroming
        # Bedingfield & Drew (Liljegren Eq. 10): a=0.56, b=0.281, c=0.4
        Re = rho * max(wind_ms, 0.5) * D_WICK / mu
        h  = k / D_WICK * 0.281 * Re**0.6 * Pr**0.44   # (1-c=0.6, 1-a=0.44)

        # Verdampingswarmte (temperatuurafhankelijk) [J kg⁻¹]
        H_vap = (2.501 - 0.00237 * (T_w - 273.15)) * 1e6

        # Dampdrukken
        e_w = _saturation_vapor_pressure(T_w)
        e_a_val = RH * _saturation_vapor_pressure(T_a)

        # Stralingsbalans op de wick (Liljegren Eq. 12)
        # Thermische straling
        rad_therm = SIGMA * EPS_WICK * (
            0.5 * (1.0 + eps_a) * T_a**4 - T_w**4
        )

        # Zonnestraling op wick
        # A1 = πDL + πD²/4  (diffuse, zijkant + bovenkant)
        # A2 = DL sin(θ) + πD²/4 cos(θ)  (direct, projectie)
        # Fnet/A = stralingsterm per eenheidoppervlak
        tan_theta = math.tan(min(theta, math.radians(85.0))) if theta < math.radians(89.5) else 0.0
        cos_theta  = math.cos(theta) if theta < math.radians(89.5) else 0.0
        D_over_4L  = D_WICK / (4.0 * L_WICK)

        rad_solar = (1.0 - ALPHA_WICK) * S * (
            (1.0 - fdir) * (1.0 + D_over_4L)          # diffuse
            + fdir * (tan_theta / math.pi + D_over_4L) # direct
            + ALPHA_SFC                                 # gereflecteerd
        )

        Fnet_over_A = rad_therm + rad_solar

        # Nat-bol vergelijking (Liljegren Eq. 9)
        psychro_term = (H_vap * M_WATER) / (cp * M_AIR) * (Pr / Sc)**0.56
        T_w_new = T_a - psychro_term * (e_w - e_a_val) / (P_Pa - e_w) + Fnet_over_A / h

        # Relaxatie
        T_w_next = 0.9 * T_w + 0.1 * T_w_new

        if abs(T_w_next - T_w) < 0.02:
            T_w = T_w_next
            break
        T_w = T_w_next

    return T_w


# ---------------------------------------------------------------------------
# Publieke interface
# ---------------------------------------------------------------------------

   
def wbgt_liljegren(
    temp_c: float,
    rh_pct: float,
    wind_ms: float,
    q_wm2: float,
    lat: float,
    lon: float,
    dt: datetime,
    pressure_hpa: float = 1013.25,
    use_solar_geometry: bool = False,
) -> float:
    """Bereken WBGT (buiten, zon) via de volledige Liljegren (2008) methode.

    WBGT = 0.7·Tw + 0.2·Tg + 0.1·Ta

    Args:
        temp_c:       Luchttemperatuur [°C].
        rh_pct:       Relatieve vochtigheid [%] (0–100).
        wind_ms:      Windsnelheid op 2 m hoogte [m s⁻¹].
        q_wm2:        Totale horizontale straling [W m⁻²].
        lat:          Breedtegraad [°N].
        lon:          Lengtegraad  [°E].
        dt:           UTC datum/tijd.
        pressure_hpa: Luchtdruk [hPa], standaard 1013.25.
        use_solar_geometry: True als het berekend moet worden met tijd, False als de gegeven waarde wordt genomen

    Returns:
        WBGT [°C].
    """
    T_a   = temp_c + 273.15
    RH    = rh_pct / 100.0
    P_Pa  = pressure_hpa * 100.0
    S     = max(q_wm2, 0.0)
    wind_ms = max(wind_ms, 0.5)   # Liljegren: methode niet geldig < 0.5 m/s
    
    # Zonnegeometrie
    doy   = _day_of_year(dt)
    d_AU  = _earth_sun_distance(doy)
   
    
    # theta = solar_zenith_angle(dt, lat, lon)
    # gebruik het midden van het uurinterval
    # KNMI labelt HH als einde van het uurinterval (HH=6 → 05:00-06:00).
    # Neem het midden van het interval; data en solver zijn beide in UTC.
    # dt = dt - timedelta(minutes=30) # 120 minutes for utc vs est. 30 min to be in the middle of the hour

    theta = solar_zenith_angle(dt, lat, lon)
    # theta = theta - math.radians(2)
    # if theta >= math.radians(89.5):
    #     S = 0.0   # zon onder horizon: negeer eventuele reststraling

    fd    = _fdir(S, theta, d_AU)
    # fd=1

    # S_max_ref = S0 / d_AU**2  # maximale straling bij θ=0, geen atmosfeer
    # S_star = S / S_max_ref    # 0..1 onafhankelijk van tijdstip
    # fd = math.exp(3 - 1.34*S_star - 1.65/S_star) if 0 < S_star < 1 else (1.0 if S_star >= 1 else 0.0)

    # Atmosferische emissiviteit (Liljegren 2008, p. 648)
    e_a   = RH * _saturation_vapor_pressure(T_a)          # [Pa]
    e_a_hPa = e_a / 100.0                                  # [hPa]
    eps_a = 0.575 * e_a_hPa**0.143

    # Submodellen
    T_g = _globe_temp_liljegren(T_a, wind_ms, S, fd, theta, P_Pa, eps_a)
    T_w = _nat_bol_temp_liljegren(T_a, RH, wind_ms, S, fd, theta, P_Pa, eps_a)

    # WBGT (Liljegren Eq. 1)
    wbgt_K = 0.7 * T_w + 0.2 * T_g + 0.1 * T_a
    return wbgt_K - 273.15


def wbgt_liljegren_from_station(
    temp_c: float,
    rh_pct: float,
    wind_ms: float,
    q_wm2: float,
    stn: int,
    dt: datetime,
    pressure_hpa: float = 1013.25,
) -> float:
    """Zelfde als wbgt_liljegren maar met KNMI-stationnummer als locatie-input.

    Args:
        stn: KNMI-stationnummer (bijv. 260 voor De Bilt).

    Raises:
        KeyError: als het stationnummer niet bekend is.
    """
    if stn not in KNMI_STATIONS:
        raise KeyError(
            f"Stationnummer {stn} niet in KNMI_STATIONS. "
            f"Bekende stations: {sorted(KNMI_STATIONS.keys())}"
        )
    lat, lon, _ = KNMI_STATIONS[stn]
    return wbgt_liljegren(temp_c, rh_pct, wind_ms, q_wm2, lat, lon, dt, pressure_hpa)


# ---------------------------------------------------------------------------
# Verificatie / snelle selftest
# ---------------------------------------------------------------------------

def _selftest() -> None:
    """Vergelijk uitvoer met Liljegren paper (Blue Grass Depot, zomermiddag)."""
    from datetime import datetime

    # Typische warme zomermiddag, Blue Grass KY (37.73°N, 84.19°W)
    dt = datetime(2007, 7, 19, 18, 0)  # 18:00 UTC ≈ 14:00 lokaal
    result = wbgt_liljegren(
        temp_c=32.0,
        rh_pct=55.0,
        wind_ms=2.0,
        q_wm2=700.0,
        lat=37.73,
        lon=-84.19,
        dt=dt,
    )
    print(f"WBGT (Blue Grass zomermiddag): {result:.2f} °C  (verwacht ~31–33 °C)")

    # Nacht: WBGT moet ≈ psychrometrische nat-bol zijn (geen straling)
    dt_nacht = datetime(2007, 7, 20, 2, 0)  # 02:00 UTC
    result_nacht = wbgt_liljegren(
        temp_c=22.0,
        rh_pct=80.0,
        wind_ms=1.5,
        q_wm2=0.0,
        lat=37.73,
        lon=-84.19,
        dt=dt_nacht,
    )
    print(f"WBGT (nacht, geen zon):         {result_nacht:.2f} °C  (verwacht ~20–22 °C)")

    # De Bilt (stn 260), warme dag
    dt_bilt = datetime(2019, 7, 25, 13, 0)  # hittegolf Nederland
    result_bilt = wbgt_liljegren_from_station(
        temp_c=40.7,
        rh_pct=25.0,
        wind_ms=3.0,
        q_wm2=850.0,
        stn=260,
        dt=dt_bilt,
    )
    print(f"WBGT (De Bilt hittegolf 25-7-2019): {result_bilt:.2f} °C")


if __name__ == "__main__":
    _selftest()