"""Liljegren (2008) outdoor Wet Bulb Globe Temperature (WBGT) in pure Python.

Vertaling van de C-code van James C. Liljegren (Argonne National Laboratory,
WBGT v1.1, 2008), met de R/ANSI-aanpassingen van Max Lieblich (2016).

Referentie:
    Liljegren, J. C., R. A. Carhart, P. Lawday, S. Tschopp & R. Sharp (2008):
    "Modeling the Wet Bulb Globe Temperature Using Standard Meteorological
    Measurements." Journal of Occupational and Environmental Hygiene, 5:10,
    645-655.

WBGT (buiten, met zon) = 0.7 * Tnwb + 0.2 * Tg + 0.1 * Tair
"""

# version : 20260604-000000 - Initiële vertaling van Liljegren WBGT C-code naar Python
current_version = "20260604-000000"

import math
from datetime import datetime

# ---------------------------------------------------------------------------
# Fysische constanten (uit wbgt.h)
# ---------------------------------------------------------------------------
SOLAR_CONST = 1367.0          # zonneconstante, W m-2
GRAVITY = 9.807               # m s-2
STEFANB = 5.6696e-8           # Stefan-Boltzmann, W m-2 K-4
Cp = 1003.5                   # soortelijke warmte droge lucht, J kg-1 K-1
M_AIR = 28.97                 # molgewicht droge lucht, g mol-1
M_H2O = 18.015                # molgewicht water, g mol-1
R_GAS = 8314.34               # universele gasconstante, J kmol-1 K-1
R_AIR = R_GAS / M_AIR         # gasconstante droge lucht, J kg-1 K-1
Pr = Cp / (Cp + 1.25 * R_AIR)  # Prandtl-getal

# Eigenschappen van de wick (natteboltemperatuur-sensor)
EMIS_WICK = 0.95
ALB_WICK = 0.4
D_WICK = 0.007                # diameter, m
L_WICK = 0.0254               # lengte, m

# Eigenschappen van de globe
EMIS_GLOBE = 0.95
ALB_GLOBE = 0.05
D_GLOBE = 0.0508              # diameter, m

# Eigenschappen van het oppervlak
EMIS_SFC = 0.999
ALB_SFC = 0.45

# Fysische / numerieke parameters
RATIO = Cp * M_AIR / M_H2O
CZA_MIN = 0.00873             # cos van zenithoek-ondergrens
NORMSOLAR_MAX = 0.85
REF_HEIGHT = 2.0              # referentiehoogte windsnelheid, m
MIN_SPEED = 0.13              # minimum windsnelheid, m s-1
CONVERGENCE = 0.02            # convergentiecriterium iteraties
MAX_ITER = 50

# Hoek-conversies
DEG_RAD = math.pi / 180.0
RAD_DEG = 180.0 / math.pi
TWOPI = 2.0 * math.pi

TRUE = 1
FALSE = 0


# ---------------------------------------------------------------------------
# Thermofysische eigenschappen van lucht en water
# ---------------------------------------------------------------------------
def esat(tk: float, phase: int) -> float:
    """Verzadigingsdampdruk (mb) boven water (phase=0) of ijs (phase=1).

    Referentie: Buck (1981), benadering van Wexler (1976).
    """
    if phase == 0:  # boven vloeibaar water
        y = (tk - 273.15) / (tk - 32.18)
        es = 6.1121 * math.exp(17.502 * y)
    else:           # boven ijs
        y = (tk - 273.15) / (tk - 0.6)
        es = 6.1115 * math.exp(22.452 * y)
    # correctie voor vochtige lucht (druk > 800 mb)
    es = 1.004 * es
    return es


def dew_point(e: float, phase: int) -> float:
    """Dauwpunt (phase=0) of vriespunt (phase=1) in K."""
    if phase == 0:  # dauwpunt
        z = math.log(e / (6.1121 * 1.004))
        tdk = 273.15 + 240.97 * z / (17.502 - z)
    else:           # vriespunt
        z = math.log(e / (6.1115 * 1.004))
        tdk = 273.15 + 272.55 * z / (22.452 - z)
    return tdk


def viscosity(Tair: float) -> float:
    """Viscositeit van lucht, kg/(m s). Referentie: BSL p. 23."""
    sigma = 3.617
    eps_kappa = 97.0
    Tr = Tair / eps_kappa
    omega = (Tr - 2.9) / 0.4 * (-0.034) + 1.048
    return 2.6693e-6 * math.sqrt(M_AIR * Tair) / (sigma * sigma * omega)


def thermal_cond(Tair: float) -> float:
    """Warmtegeleidingscoëfficiënt van lucht, W/(m K). Referentie: BSL p. 257."""
    return (Cp + 1.25 * R_AIR) * viscosity(Tair)


def diffusivity(Tair: float, Pair: float) -> float:
    """Diffusiviteit van waterdamp in lucht, m2/s. Referentie: BSL p. 505."""
    Pcrit_air = 36.4
    Pcrit_h2o = 218.0
    Tcrit_air = 132.0
    Tcrit_h2o = 647.3
    a = 3.640e-4
    b = 2.334

    Pcrit13 = (Pcrit_air * Pcrit_h2o) ** (1.0 / 3.0)
    Tcrit512 = (Tcrit_air * Tcrit_h2o) ** (5.0 / 12.0)
    Tcrit12 = math.sqrt(Tcrit_air * Tcrit_h2o)
    Mmix = math.sqrt(1.0 / M_AIR + 1.0 / M_H2O)
    Patm = Pair / 1013.25  # mb -> atm

    return a * (Tair / Tcrit12) ** b * Pcrit13 * Tcrit512 * Mmix / Patm * 1e-4


def evap(Tair: float) -> float:
    """Verdampingswarmte, J/(kg K), geldig in bereik 283-313 K.

    Referentie: Van Wylen & Sonntag, Tabel A.1.1.
    """
    return (313.15 - Tair) / 30.0 * (-71100.0) + 2.4073e6


def emis_atm(Tair: float, rh: float) -> float:
    """Atmosferische emissiviteit. Referentie: Oke (2e ed.), p. 373."""
    e = rh * esat(Tair, 0)
    return 0.575 * e ** 0.143


# ---------------------------------------------------------------------------
# Convectieve warmteoverdrachtscoëfficiënten
# ---------------------------------------------------------------------------
def h_sphere_in_air(diameter: float, Tair: float, Pair: float, speed: float) -> float:
    """Convectieve warmteoverdracht W/(m2 K) voor bol. Referentie: BSL p. 409."""
    density = Pair * 100.0 / (R_AIR * Tair)
    Re = max(speed, MIN_SPEED) * density * diameter / viscosity(Tair)
    Nu = 2.0 + 0.6 * math.sqrt(Re) * Pr ** 0.3333
    return Nu * thermal_cond(Tair) / diameter


def h_cylinder_in_air(
    diameter: float, length: float, Tair: float, Pair: float, speed: float
) -> float:
    """Convectieve warmteoverdracht W/(m2 K) voor cilinder in dwarsstroming.

    Referentie: Bedingfield & Drew, vgl. 32. (`length` wordt niet gebruikt,
    behouden t.b.v. trouwe vertaling.)
    """
    a = 0.56
    b = 0.281
    c = 0.4
    density = Pair * 100.0 / (R_AIR * Tair)
    Re = max(speed, MIN_SPEED) * density * diameter / viscosity(Tair)
    Nu = b * Re ** (1.0 - c) * Pr ** (1.0 - a)
    return Nu * thermal_cond(Tair) / diameter


# ---------------------------------------------------------------------------
# Zonnepositie (Larson, PNNL) + zonneparameters
# ---------------------------------------------------------------------------
def daynum(year: int, month: int, day: int) -> int:
    """Volgnummer van de dag binnen een Gregoriaans kalenderjaar (1=1 jan)."""
    begmonth = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    if year < 1:
        return -1
    leapyr = 1 if ((year % 4 == 0 and year % 100 != 0) or year % 400 == 0) else 0
    dnum = begmonth[month] + day
    if leapyr and month > 2:
        dnum += 1
    return dnum


def solarposition(
    year: int, month: int, day: float, days_1900: float, latitude: float, longitude: float
) -> tuple[int, float, float, float, float, float, float]:
    """Lage-precisie zonnepositie volgens Astronomical Almanac 1990.

    Retourneert ``(rc, ap_ra, ap_dec, altitude, refraction, azimuth, distance)``.
    ``rc`` is -1 bij ongeldige invoer, anders 0. Hoeken in graden, afstand in AU.
    """
    if latitude < -90.0 or latitude > 90.0 or longitude < -180.0 or longitude > 180.0:
        return (-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    pressure = 1013.25  # mb
    temp = 15.0         # degC

    if year != 0:
        if year < 1950 or year > 2049:
            return (-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if month != 0:
            if month < 1 or month > 12 or day < 0.0 or day > 33.0:
                return (-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            daynumber = daynum(year, month, int(day))
        else:
            if day < 0.0 or day > 368.0:
                return (-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            daynumber = int(day)

        delta_years = year - 2000
        delta_days = delta_years * 365 + delta_years // 4 + daynumber
        if year > 2000:
            delta_days += 1
        days_J2000 = delta_days - 1.5
        cent_J2000 = days_J2000 / 36525.0

        ut, integral = math.modf(day)
        days_J2000 += ut
        ut *= 24.0
    else:
        if days_1900 < 18262.0 or days_1900 > 54788.0:
            return (-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        days_J2000 = days_1900 - 36525.5
        frac, integral = math.modf(days_1900)
        ut = frac * 24.0
        cent_J2000 = (integral - 36525.5) / 36525.0

    # Zonneparameters (A.A. 1990, C24)
    mean_anomaly = 357.528 + 0.9856003 * days_J2000
    mean_longitude = 280.460 + 0.9856474 * days_J2000

    frac, _ = math.modf(mean_anomaly / 360.0)
    mean_anomaly = frac * TWOPI
    frac, _ = math.modf(mean_longitude / 360.0)
    mean_longitude = frac * TWOPI

    mean_obliquity = (23.439 - 4.0e-7 * days_J2000) * DEG_RAD
    ecliptic_long = (
        (1.915 * math.sin(mean_anomaly)) + (0.020 * math.sin(2.0 * mean_anomaly))
    ) * DEG_RAD + mean_longitude

    distance = 1.00014 - 0.01671 * math.cos(mean_anomaly) - 0.00014 * math.cos(
        2.0 * mean_anomaly
    )

    ap_ra = math.atan2(
        math.cos(mean_obliquity) * math.sin(ecliptic_long), math.cos(ecliptic_long)
    )
    if ap_ra < 0.0:
        ap_ra += TWOPI
    frac, _ = math.modf(ap_ra / TWOPI)
    ap_ra = frac * 24.0

    ap_dec = math.asin(math.sin(mean_obliquity) * math.sin(ecliptic_long))

    # Lokale gemiddelde sterrentijd (A.A. 1990, B6-B7)
    gmst0h = 24110.54841 + cent_J2000 * (
        8640184.812866 + cent_J2000 * (0.093104 - cent_J2000 * 6.2e-6)
    )
    frac, _ = math.modf(gmst0h / 3600.0 / 24.0)
    gmst0h = frac * 24.0
    if gmst0h < 0.0:
        gmst0h += 24.0

    lmst = gmst0h + (ut * 1.00273790934) + longitude / 15.0
    frac, _ = math.modf(lmst / 24.0)
    lmst = frac * 24.0
    if lmst < 0.0:
        lmst += 24.0

    local_ha = lmst - ap_ra
    if local_ha < -12.0:
        local_ha += 24.0
    elif local_ha > 12.0:
        local_ha -= 24.0

    latitude_rad = latitude * DEG_RAD
    local_ha = local_ha / 24.0 * TWOPI

    cos_apdec = math.cos(ap_dec)
    sin_apdec = math.sin(ap_dec)
    cos_lat = math.cos(latitude_rad)
    sin_lat = math.sin(latitude_rad)
    cos_lha = math.cos(local_ha)

    altitude = math.asin(sin_apdec * sin_lat + cos_apdec * cos_lha * cos_lat)
    cos_alt = math.cos(altitude)

    if abs(altitude) < 1.57079615:
        tan_alt = math.tan(altitude)
    else:
        tan_alt = 6.0e6

    cos_az = (sin_apdec * cos_lat - cos_apdec * cos_lha * sin_lat) / cos_alt
    sin_az = -(cos_apdec * math.sin(local_ha) / cos_alt)
    azimuth = math.acos(max(-1.0, min(1.0, cos_az)))

    if math.atan2(sin_az, cos_az) < 0.0:
        azimuth = TWOPI - azimuth

    ap_dec *= RAD_DEG
    altitude *= RAD_DEG
    azimuth *= RAD_DEG

    # Refractiecorrectie
    if altitude < -1.0 or tan_alt == 6.0e6:
        refraction = 0.0
    else:
        if altitude < 19.225:
            refraction = (
                0.1594 + altitude * (0.0196 + 0.00002 * altitude)
            ) * pressure
            refraction /= (
                1.0 + altitude * (0.505 + 0.0845 * altitude)
            ) * (273.0 + temp)
        else:
            refraction = 0.00452 * (pressure / (273.0 + temp)) / tan_alt

    altitude = altitude + refraction
    return (0, ap_ra, ap_dec, altitude, refraction, azimuth, distance)


def calc_solar_parameters(
    year: int, month: int, day: float, lat: float, lon: float, solar: float
) -> tuple[float, float, float]:
    """Bereken cos(zenithoek), aangepaste straling en directe-bundelfractie.

    Retourneert ``(solar, cza, fdir)``.
    """
    days_1900 = 0.0
    rc, ap_ra, ap_dec, elev, refr, azim, soldist = solarposition(
        year, month, day, days_1900, lat, lon
    )
    cza = math.cos((90.0 - elev) * DEG_RAD)
    toasolar = SOLAR_CONST * max(0.0, cza) / (soldist * soldist)

    # zon niet volledig boven horizon -> top-of-atmosphere straling = 0
    if cza < CZA_MIN:
        toasolar = 0.0

    if toasolar > 0.0:
        normsolar = min(solar / toasolar, NORMSOLAR_MAX)
        solar = normsolar * toasolar
        if normsolar > 0.0:
            fdir = math.exp(3.0 - 1.34 * normsolar - 1.65 / normsolar)
            fdir = max(min(fdir, 0.9), 0.0)
        else:
            fdir = 0.0
    else:
        fdir = 0.0

    return (solar, cza, fdir)


# ---------------------------------------------------------------------------
# Globe- en (natte)boltemperaturen
# ---------------------------------------------------------------------------
def Tglobe(
    Tair: float, rh: float, Pair: float, speed: float, solar: float, fdir: float, cza: float
) -> float:
    """Globetemperatuur in °C; -9999 als de iteratie niet convergeert."""
    Tsfc = Tair
    Tglobe_prev = Tair
    converged = False
    it = 0
    Tglobe_new = Tair
    while not converged and it < MAX_ITER:
        it += 1
        Tref = 0.5 * (Tglobe_prev + Tair)
        h = h_sphere_in_air(D_GLOBE, Tref, Pair, speed)
        Tglobe_new = (
            0.5 * (emis_atm(Tair, rh) * Tair ** 4.0 + EMIS_SFC * Tsfc ** 4.0)
            - h / (STEFANB * EMIS_GLOBE) * (Tglobe_prev - Tair)
            + solar / (2.0 * STEFANB * EMIS_GLOBE)
            * (1.0 - ALB_GLOBE)
            * (fdir * (1.0 / (2.0 * cza) - 1.0) + 1.0 + ALB_SFC)
        ) ** 0.25
        if abs(Tglobe_new - Tglobe_prev) < CONVERGENCE:
            converged = True
        Tglobe_prev = 0.9 * Tglobe_prev + 0.1 * Tglobe_new

    if converged:
        return Tglobe_new - 273.15
    return -9999.0


def Twb(
    Tair: float,
    rh: float,
    Pair: float,
    speed: float,
    solar: float,
    fdir: float,
    cza: float,
    rad: int,
) -> float:
    """Natteboltemperatuur in °C.

    ``rad=1`` -> natuurlijke natbol (incl. straling); ``rad=0`` -> psychrometrische.
    Retourneert -9999 als de iteratie niet convergeert.
    """
    a = 0.56  # Bedingfield & Drew
    Tsfc = Tair
    sza = math.acos(cza)
    eair = rh * esat(Tair, 0)
    Tdew = dew_point(eair, 0)
    Twb_prev = Tdew  # eerste schatting = dauwpunt
    converged = False
    it = 0
    Twb_new = Twb_prev
    while not converged and it < MAX_ITER:
        it += 1
        Tref = 0.5 * (Twb_prev + Tair)
        h = h_cylinder_in_air(D_WICK, L_WICK, Tref, Pair, speed)
        Fatm = STEFANB * EMIS_WICK * (
            0.5 * (emis_atm(Tair, rh) * Tair ** 4.0 + EMIS_SFC * Tsfc ** 4.0)
            - Twb_prev ** 4.0
        ) + (1.0 - ALB_WICK) * solar * (
            (1.0 - fdir) * (1.0 + 0.25 * D_WICK / L_WICK)
            + fdir * ((math.tan(sza) / math.pi) + 0.25 * D_WICK / L_WICK)
            + ALB_SFC
        )
        ewick = esat(Twb_prev, 0)
        density = Pair * 100.0 / (R_AIR * Tref)
        Sc = viscosity(Tref) / (density * diffusivity(Tref, Pair))
        Twb_new = (
            Tair
            - evap(Tref) / RATIO * (ewick - eair) / (Pair - ewick) * (Pr / Sc) ** a
            + (Fatm / h * rad)
        )
        if abs(Twb_new - Twb_prev) < CONVERGENCE:
            converged = True
        Twb_prev = 0.9 * Twb_prev + 0.1 * Twb_new

    if converged:
        return Twb_new - 273.15
    return -9999.0


# ---------------------------------------------------------------------------
# Windschatting en stabiliteitsklasse (alleen nodig als z != 2 m)
# ---------------------------------------------------------------------------
def stab_srdt(daytime: int, speed: float, solar: float, dT: float) -> int:
    """Stabiliteitsklasse (1-6). Referentie: EPA-454/5-99-005 (2000), §6.2.5."""
    lsrdt = [
        [1, 1, 2, 4, 0, 5, 6, 0],
        [1, 2, 3, 4, 0, 5, 6, 0],
        [2, 2, 3, 4, 0, 4, 4, 0],
        [3, 3, 4, 4, 0, 0, 0, 0],
        [3, 4, 4, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    if daytime:
        if solar >= 925.0:
            j = 0
        elif solar >= 675.0:
            j = 1
        elif solar >= 175.0:
            j = 2
        else:
            j = 3
        if speed >= 6.0:
            i = 4
        elif speed >= 5.0:
            i = 3
        elif speed >= 3.0:
            i = 2
        elif speed >= 2.0:
            i = 1
        else:
            i = 0
    else:
        j = 6 if dT >= 0.0 else 5
        if speed >= 2.5:
            i = 2
        elif speed >= 2.0:
            i = 1
        else:
            i = 0
    return lsrdt[i][j]


def est_wind_speed(speed: float, zspeed: float, stability_class: int, urban: int) -> float:
    """Schat 2-m windsnelheid. Referentie: EPA-454/5-99-005 (2000), §6.2.5."""
    urban_exp = [0.15, 0.15, 0.20, 0.25, 0.30, 0.30]
    rural_exp = [0.07, 0.07, 0.10, 0.15, 0.35, 0.55]
    exponent = (
        urban_exp[stability_class - 1] if urban else rural_exp[stability_class - 1]
    )
    est = speed * (REF_HEIGHT / zspeed) ** exponent
    return max(est, MIN_SPEED)


# ---------------------------------------------------------------------------
# Volledige WBGT-berekening (trouwe vertaling van calc_wbgt)
# ---------------------------------------------------------------------------
def calc_wbgt(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    gmt: int,
    avg: int,
    lat: float,
    lon: float,
    solar: float,
    pres: float,
    Tair: float,
    relhum: float,
    speed: float,
    zspeed: float,
    dT: float,
    urban: int,
) -> dict:
    """Buiten-WBGT volgens Liljegren.

    Argumenten komen overeen met de originele C-functie. ``gmt`` is het verschil
    in uren tussen lokale tijd en GMT; ``avg`` is de middelingsperiode in minuten;
    ``zspeed`` is de hoogte (m) van de windmeting; ``dT`` is het verticale
    temperatuurverschil (voor stabiliteit); ``urban`` is 1 (stedelijk) of 0.

    Retourneert een dict met ``est_speed``, ``Tg``, ``Tnwb``, ``Tpsy``,
    ``Twbg`` en ``rc`` (0 bij succes, -1 bij niet-convergentie).
    """
    hour_gmt = hour - gmt + (minute - 0.5 * avg) / 60.0
    dday = day + hour_gmt / 24.0

    solar, cza, fdir = calc_solar_parameters(year, month, dday, lat, lon, solar)

    est_speed = speed
    if zspeed != REF_HEIGHT:
        daytime = TRUE if cza > 0.0 else FALSE
        stability_class = stab_srdt(daytime, speed, solar, dT)
        est_speed = est_wind_speed(speed, zspeed, stability_class, urban)
        speed = est_speed

    tk = Tair + 273.15
    rh = 0.01 * relhum

    Tg = Tglobe(tk, rh, pres, speed, solar, fdir, cza)
    Tnwb = Twb(tk, rh, pres, speed, solar, fdir, cza, 1)
    Tpsy = Twb(tk, rh, pres, speed, solar, fdir, cza, 0)
    Twbg = 0.1 * Tair + 0.2 * Tg + 0.7 * Tnwb

    if Tg == -9999 or Tnwb == -9999:
        return {
            "est_speed": est_speed,
            "Tg": Tg,
            "Tnwb": Tnwb,
            "Tpsy": Tpsy,
            "Twbg": -9999.0,
            "rc": -1,
        }
    return {
        "est_speed": est_speed,
        "Tg": Tg,
        "Tnwb": Tnwb,
        "Tpsy": Tpsy,
        "Twbg": Twbg,
        "rc": 0,
    }


# ---------------------------------------------------------------------------
# Publieke wrapper met de gevraagde signature
# ---------------------------------------------------------------------------
def wbgt_liljegren_opus(
    temp_c: float,
    rh_pct: float,
    wind_ms: float,
    q_wm2: float,
    lat: float,
    lon: float,
    dt: datetime,
    pressure_hpa: float = 1013.25,
) -> float:
    """Bereken WBGT (buiten, zon) via de volledige Liljegren (2008) methode, 
    geconverteerd vanuit de oorspronkelijke C-routine
    https://github.com/mdljts/wbgt/blob/master/src/wbgt.c

    WBGT = 0.7·Tw + 0.2·Tg + 0.1·Ta

    Args:
        temp_c:       Luchttemperatuur [°C].
        rh_pct:       Relatieve vochtigheid [%] (0–100).
        wind_ms:      Windsnelheid op 2 m hoogte [m s⁻¹].
        q_wm2:        Totale horizontale straling [W m⁻²].
        lat:          Breedtegraad [°N].
        lon:          Lengtegraad  [°E].
        dt:           UTC datum/tijd (einde-uur, KNMI HH-conventie).
        pressure_hpa: Luchtdruk [hPa], standaard 1013.25.

    Returns:
        WBGT [°C].

    Raises:
        ValueError: als de Liljegren-iteratie niet convergeert (bijv. bij
            fysisch onmogelijke invoer).
    """
    result = calc_wbgt(
        year=dt.year,
        month=dt.month,
        day=dt.day,
        hour=dt.hour,
        minute=dt.minute,
        gmt=0,            # dt is reeds UTC
        avg=60,           # KNMI uurwaarden: middeling over 60 min
        lat=lat,
        lon=lon,
        solar=q_wm2,
        pres=pressure_hpa,
        Tair=temp_c,
        relhum=rh_pct,
        speed=wind_ms,
        zspeed=REF_HEIGHT,  # wind reeds op 2 m -> geen extrapolatie
        dT=0.0,
        urban=0,
    )
    if result["rc"] != 0:
        raise ValueError("Liljegren WBGT-iteratie convergeerde niet voor deze invoer.")
    return result["Twbg"]



def wbgt_liljegren_from_station(
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

    Args:
        stn:       KNMI-stationnummer (bijv. 260 voor De Bilt).
        fdir_mode: Zie wbgt_liljegren(); default 'knmi_obs'.

    Raises:
        KeyError: als het stationnummer niet bekend is.
    """
    if stn not in KNMI_STATIONS:
        raise KeyError(
            f"Stationnummer {stn} niet in KNMI_STATIONS. "
            f"Bekende stations: {sorted(KNMI_STATIONS.keys())}"
        )
    lat, lon, _ = KNMI_STATIONS[stn]
    return wbgt_liljegren_opus(temp_c, rh_pct, wind_ms, q_wm2, lat, lon, dt, pressure_hpa)


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

if __name__ == "__main__":
    # Validatievoorbeeld: warme zomermiddag, Nederland.
    demo = wbgt_liljegren(
        temp_c=30.0,
        rh_pct=50.0,
        wind_ms=1.0,
        q_wm2=800.0,
        lat=52.1,
        lon=5.2,
        dt=datetime(2024, 7, 1, 13, 0),  # UTC
        pressure_hpa=1013.25,
    )
    print(f"WBGT = {demo:.2f} °C")