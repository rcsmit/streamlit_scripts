# version = "20260604120000"
# Description:
# Calculate wet bulb globe temperature (WBGT) using the Liljegren method.
#
# Based on the original C code by Liljegren:
# https://github.com/mdljts/wbgt/blob/master/src/wbgt.c.original
#
# Reference:
#     Liljegren, J. C., Carhart, R. A., Lawday, P., Tschopp, S. & Sharp, R.
#     Modeling the Wet Bulb Globe Temperature Using Standard Meteorological Measurements.
#     Journal of Occupational and Environmental Hygiene 5, 645–655 (2008).

# Original Cython code by Qinqin Kong (07-04-2021)
# Kong, Qinqin, and Matthew Huber. 
# “Explicit Calculations of Wet Bulb Globe Temperature Compared with 
# Approximations and Why It Matters for Labor Productivity.” 
# Earth’s Future, January 31, 2022. https://doi.org/10.1029/2021EF002334
#

# Translated to pure Python by Claude (2026-06-04)

# GCM = Global Climate Model (ook wel General Circulation Model). 
# Het is een alternatieve manier om de boltemperatuur (Tg) en de 
# natte-bol-temperatuur (Tnwb) te berekenen wanneer je volledige 
# stralingsbalans-uitvoer van een weermodel beschikbaar hebt.

import math
import numpy as np
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
mair = 28.97           # molecular weight of dry air (g/mol)
mh2o = 18.015          # molecular weight of water vapour (g/mol)
rgas = 8314.34         # ideal gas constant (J / kg·mol·K)
cp = 1003.5            # specific heat capacity of air at constant pressure (J/kg/K)
stefanb = 5.6696e-8    # Stefan–Boltzmann constant (W/m²/K⁴)
ratio = cp * mair / mh2o
rair = rgas / mair
Pr = cp / (cp + 1.25 * rair)   # Prandtl number

# Globe constants
diamglobe = 0.0508     # diameter of globe (m)
emisglobe = 0.95       # emissivity of globe
albglobe = 0.05        # albedo of globe

# Wick constants
emiswick = 0.95        # emissivity of the wick
albwick = 0.4          # albedo of the wick
diamwick = 0.007       # diameter of the wick (m)
lenwick = 0.0254       # length of the wick (m)

# Surface constant
albsfc = 0.45

PI = 3.1415926535897932

# ---------------------------------------------------------------------------
# Pasquill–Gifford stability class lookup table (lsrdt) and urban exponents
# ---------------------------------------------------------------------------
_lsrdt = [
    [1, 1, 2, 4, 0, 5, 6, 0],
    [1, 2, 3, 4, 0, 5, 6, 0],
    [2, 2, 3, 4, 0, 4, 4, 0],
    [3, 3, 4, 4, 0, 0, 0, 0],
    [3, 4, 4, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]

_urban_exp = [0.15, 0.15, 0.20, 0.25, 0.30, 0.30]


# ---------------------------------------------------------------------------
# Helper / physics functions
# ---------------------------------------------------------------------------

def _sunearth(date) -> float:
    """
    date: single numpy datetime64 value
    Return sun–earth distance (astronomical units).
    """
    days_J2000 = (
        (date - np.datetime64('2000-01-01T12:00:00.000000000'))
        .astype('timedelta64[m]') / np.timedelta64(1, 'm')
    ) / (60 * 24)
    g = ((357.528 + 0.9856003 * days_J2000) % 360) * PI / 180
    return 1.00014 - 0.01671 * math.cos(g) - 0.00014 * math.cos(2.0 * g)


def _viscosity(tas: float) -> float:
    """Air viscosity (kg/m/s).  tas in K."""
    omega = 1.2945 - tas / 1141.176470588
    return 0.0000026693 * math.sqrt(28.97 * tas) / (13.082689 * omega)


def _thermcond(tas: float) -> float:
    """Thermal conductivity of air (W/m/K).  tas in K."""
    return (cp + 1.25 * rair) * _viscosity(tas)


def _esat(tas: float, ps: float) -> float:
    """
    Saturation vapour pressure (Pa).
    tas: air temperature (K); ps: surface pressure (Pa).
    """
    if tas > 273.15:
        es = 611.21 * math.exp(17.502 * (tas - 273.15) / (tas - 32.18))
        es = (1.0007 + 3.46e-6 * ps / 100) * es
    else:
        es = 611.15 * math.exp(22.452 * (tas - 273.15) / (tas - 0.6))
        es = (1.0003 + 4.18e-6 * ps / 100) * es
    return es


def _emisatm(tas: float, hurs: float, ps: float) -> float:
    """Atmospheric emissivity."""
    e = hurs * 0.01 * (_esat(tas, ps) * 0.01)
    return 0.575 * (e ** 0.143)


def _diffusivity(tas: float, ps: float) -> float:
    """Diffusivity of water vapour in air (m²/s)."""
    return 2.471773765165648e-05 * ((tas * 0.0034210563748421257) ** 2.334) / (ps / 101325)


def _h_evap(tas: float) -> float:
    """Heat of evaporation (J/kg).  tas in K."""
    return 1665134.5 + 2370.0 * tas


def _stab_srdt(cosz: float, sfcwind: float, rsds: float) -> int:
    """Pasquill–Gifford stability class index."""
    if cosz > 0:
        if rsds >= 925.0:
            j = 0
        elif rsds >= 675.0:
            j = 1
        elif rsds >= 175.0:
            j = 2
        else:
            j = 3
        if sfcwind >= 6.0:
            i = 4
        elif sfcwind >= 5.0:
            i = 3
        elif sfcwind >= 3.0:
            i = 2
        elif sfcwind >= 2.0:
            i = 1
        else:
            i = 0
    else:
        j = 5
        if sfcwind >= 2.5:
            i = 2
        elif sfcwind >= 2.0:
            i = 1
        else:
            i = 0
    return _lsrdt[i][j]


def _wind2m(sfcwind: float, cosz: float, rsds: float) -> float:
    """Convert 10 m wind speed to 2 m wind speed."""
    stability_class = _stab_srdt(cosz, sfcwind, rsds)
    return max(sfcwind * math.pow(2.0 / 10.0, _urban_exp[stability_class - 1]), 0.13)


def _h_sphere_in_air(tas: float, ps: float, sfcwind: float) -> float:
    """
    Convective heat transfer coefficient for flow around a sphere (W/m²/K).
    tas: K; ps: Pa; sfcwind: m/s.
    """
    thermcon = _thermcond(tas)
    density = ps / (rair * tas)
    Re = sfcwind * density * diamglobe / _viscosity(tas)
    Nu = 2 + 0.6 * math.sqrt(Re) * math.pow(Pr, 0.3333)
    return Nu * thermcon / diamglobe


def _h_cylinder_in_air(tas: float, ps: float, sfcwind: float) -> float:
    """
    Convective heat transfer coefficient for a long cylinder (W/m²/K).
    tas: K; ps: Pa; sfcwind: m/s.
    """
    thermcon = _thermcond(tas)
    density = ps / (rair * tas)
    Re = sfcwind * density * diamwick / _viscosity(tas)
    Nu = 0.281 * (Re ** 0.6) * (Pr ** 0.44)
    return Nu * thermcon / diamwick


# ---------------------------------------------------------------------------
# Root-finding equations
# ---------------------------------------------------------------------------

def _fTg(x: float, C0: float, C1: float, C2: float, C3: float) -> float:
    """Equation for globe temperature Tg (to be solved by iteration)."""
    h = _h_sphere_in_air(0.5 * (C1 + x), C2, C3)
    return C0 - (1.0 / (emisglobe * stefanb)) * h * (x - C1) - x ** 4


def _fTnwb(x: float, D0: float, D1: float, D2: float, D3: float, D4: float) -> float:
    """Equation for natural wet-bulb temperature Tnwb (to be solved by iteration)."""
    evap = _h_evap(0.5 * (x + D0))
    es = _esat(x, D1)
    density = D1 / (0.5 * (D0 + x) * rair)
    Sc = _viscosity(0.5 * (D0 + x)) / (density * _diffusivity(0.5 * (D0 + x), D1))
    h = _h_cylinder_in_air(0.5 * (D0 + x), D1, D3)
    Fatm = D4 - emiswick * stefanb * (x ** 4)
    return (D0
            - evap / ratio * (es - D2) / (D1 - es) * (Pr / Sc) ** 0.56
            + Fatm / h
            - x)


# ---------------------------------------------------------------------------
# Scalar solvers
# ---------------------------------------------------------------------------

def _solve_Tg(C0, C1, C2, C3, xtol=0.01, rtol=4*np.finfo(float).eps, mitr=1000):
    xa = C1 - 50
    xb = C1 + 90
    return brentq(_fTg, xa, xb, args=(C0, C1, C2, C3),
                  xtol=xtol, rtol=rtol, maxiter=mitr)

def _solve_Tnwb(D0, D1, D2, D3, D4, hurs, xtol=0.001, rtol=4*np.finfo(float).eps, mitr=1_000_000):
    xa = D0 - (100 - hurs) / 5.0 - 50
    xb = min(D0 + 70, 340.0)
    return brentq(_fTnwb, xa, xb, args=(D0, D1, D2, D3, D4),
                  xtol=xtol, rtol=rtol, maxiter=mitr)

# De standaardwaarde rtol=0.0 is niet toegestaan in scipy — scipy eist rtol >= 4*eps ≈ 8.88e-16.
# 4*np.finfo(float).eps is exact de ondergrens die scipy intern gebruikt, dus dit is de minimale geldige waarde. 
# De Cython-versie gebruikte rtol=0.0 omdat die een eigen Brent-implementatie had zonder die beperking.
def _solve_Tg_original(C0: float, C1: float, C2: float, C3: float,
              xtol: float = 0.01, rtol: float = 0.0, mitr: int = 1000) -> float:
    """Solve for globe temperature (scalar)."""
    xa = C1 - 50
    xb = C1 + 90
    return brentq(_fTg, xa, xb, args=(C0, C1, C2, C3),
                  xtol=xtol, rtol=rtol, maxiter=mitr)


def _solve_Tnwb_original(D0: float, D1: float, D2: float, D3: float, D4: float,
                hurs: float,
                xtol: float = 0.001, rtol: float = 0.0, mitr: int = 1_000_000) -> float:
    """Solve for natural wet-bulb temperature (scalar)."""
    xa = D0 - (100 - hurs) / 5.0 - 50
    xb = min(D0 + 70, 340.0)
    return brentq(_fTnwb, xa, xb, args=(D0, D1, D2, D3, D4),
                  xtol=xtol, rtol=rtol, maxiter=mitr)


# ---------------------------------------------------------------------------
# Array-level private helpers (3-D loops)
# ---------------------------------------------------------------------------

def _Tg_GCM_core(tas, ps, sfcwind, rsds, rsus, rlds, rlus, fdir_arr, cosz_arr,
                 use_2m_wind: bool,
                 xtol: float = 0.01, rtol: float = 0.0, mitr: int = 1000) -> np.ndarray:
    """Compute globe temperature using GCM radiation variables (3-D arrays)."""
    shape = tas.shape
    result = np.full(shape, np.nan, dtype=np.float64)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if any(math.isnan(v) for v in [tas[i, j, k], ps[i, j, k],
                                               sfcwind[i, j, k], rsds[i, j, k],
                                               rsus[i, j, k], rlds[i, j, k],
                                               rlus[i, j, k], fdir_arr[i, j, k],
                                               cosz_arr[i, j, k]]):
                    continue
                w = sfcwind[i, j, k] if use_2m_wind else _wind2m(sfcwind[i, j, k],
                                                                   cosz_arr[i, j, k],
                                                                   rsds[i, j, k])
                C0 = (0.5 / stefanb * (rlds[i, j, k] + rlus[i, j, k])
                      + rsds[i, j, k] / (2 * emisglobe * stefanb)
                      * (1 - albglobe)
                      * (1 - fdir_arr[i, j, k] + 0.5 * fdir_arr[i, j, k] / cosz_arr[i, j, k])
                      + (1 - albglobe) / (2 * emisglobe * stefanb) * rsus[i, j, k])
                result[i, j, k] = _solve_Tg(C0, tas[i, j, k], ps[i, j, k], w,
                                             xtol=xtol,  mitr=mitr)
    return result


def _Tg_Liljegren_core(tas, hurs, ps, sfcwind, rsds, fdir_arr, cosz_arr,
                       use_2m_wind: bool,
                       xtol: float = 0.001, rtol: float = 0.0,
                       mitr: int = 1_000_000) -> np.ndarray:
    """Compute globe temperature using Liljegren parameterisation (3-D arrays)."""
    shape = tas.shape
    result = np.full(shape, np.nan, dtype=np.float64)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if any(math.isnan(v) for v in [tas[i, j, k], hurs[i, j, k],
                                               ps[i, j, k], sfcwind[i, j, k],
                                               rsds[i, j, k], fdir_arr[i, j, k],
                                               cosz_arr[i, j, k]]):
                    continue
                w = sfcwind[i, j, k] if use_2m_wind else _wind2m(sfcwind[i, j, k],
                                                                   cosz_arr[i, j, k],
                                                                   rsds[i, j, k])
                C0 = (0.5 * (1 + _emisatm(tas[i, j, k], hurs[i, j, k], ps[i, j, k]))
                      * tas[i, j, k] ** 4
                      + rsds[i, j, k] / (2 * emisglobe * stefanb)
                      * (1 - albglobe)
                      * (1 + (0.5 / cosz_arr[i, j, k] - 1) * fdir_arr[i, j, k] + albsfc))
                result[i, j, k] = _solve_Tg(C0, tas[i, j, k], ps[i, j, k], w,
                                             xtol=xtol, mitr=mitr)
    return result


def _Tnwb_GCM_core(tas, hurs, ps, sfcwind, rsds, rsus, rlds, rlus, fdir_arr, cosz_arr,
                   use_2m_wind: bool,
                   xtol: float = 0.01, rtol: float = 0.0,
                   mitr: int = 100_000_000) -> np.ndarray:
    """Compute natural wet-bulb temperature using GCM radiation variables (3-D arrays)."""
    shape = tas.shape
    result = np.full(shape, np.nan, dtype=np.float64)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if any(math.isnan(v) for v in [tas[i, j, k], hurs[i, j, k],
                                               ps[i, j, k], sfcwind[i, j, k],
                                               rsds[i, j, k], rsus[i, j, k],
                                               rlds[i, j, k], rlus[i, j, k],
                                               fdir_arr[i, j, k], cosz_arr[i, j, k]]):
                    continue
                w = sfcwind[i, j, k] if use_2m_wind else _wind2m(sfcwind[i, j, k],
                                                                   cosz_arr[i, j, k],
                                                                   rsds[i, j, k])
                D0 = tas[i, j, k]
                D1 = ps[i, j, k]
                D2 = hurs[i, j, k] * 0.01 * _esat(D0, D1)
                D3 = w
                D4 = (emiswick * 0.5 * (rlds[i, j, k] + rlus[i, j, k])
                      + (1 + diamwick / (4 * lenwick)) * (1 - albwick)
                      * (1 - fdir_arr[i, j, k]) * rsds[i, j, k]
                      + (math.tan(math.acos(cosz_arr[i, j, k])) / PI
                         + diamwick / (4 * lenwick))
                      * (1 - albwick) * fdir_arr[i, j, k] * rsds[i, j, k]
                      + (1 - albwick) * rsus[i, j, k])
                result[i, j, k] = _solve_Tnwb(D0, D1, D2, D3, D4,
                                               hurs[i, j, k],
                                               xtol=xtol, mitr=mitr)
    return result


def _Tnwb_Liljegren_core(tas, hurs, ps, sfcwind, rsds, fdir_arr, cosz_arr,
                         use_2m_wind: bool,
                         xtol: float = 0.001, rtol: float = 0.0,
                         mitr: int = 1_000_000) -> np.ndarray:
    """Compute natural wet-bulb temperature using Liljegren parameterisation (3-D arrays)."""
    shape = tas.shape
    result = np.full(shape, np.nan, dtype=np.float64)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if any(math.isnan(v) for v in [tas[i, j, k], hurs[i, j, k],
                                               ps[i, j, k], sfcwind[i, j, k],
                                               rsds[i, j, k], fdir_arr[i, j, k],
                                               cosz_arr[i, j, k]]):
                    continue
                w = sfcwind[i, j, k] if use_2m_wind else _wind2m(sfcwind[i, j, k],
                                                                   cosz_arr[i, j, k],
                                                                   rsds[i, j, k])
                D0 = tas[i, j, k]
                D1 = ps[i, j, k]
                D2 = hurs[i, j, k] * 0.01 * _esat(D0, D1)
                D3 = w
                D4 = (emiswick * 0.5 * stefanb * D0 ** 4
                      * (_emisatm(D0, hurs[i, j, k], D1) + 1)
                      + (1 - albwick) * rsds[i, j, k]
                      * ((1 + diamwick / (4 * lenwick)) * (1 - fdir_arr[i, j, k])
                         + (math.tan(math.acos(cosz_arr[i, j, k])) / PI
                            + diamwick / (4 * lenwick)) * fdir_arr[i, j, k]
                         + albsfc))
                result[i, j, k] = _solve_Tnwb(D0, D1, D2, D3, D4,
                                               hurs[i, j, k],
                                               xtol=xtol,  mitr=mitr)
    return result


# ---------------------------------------------------------------------------
# fdir (ratio of direct solar radiation)
# ---------------------------------------------------------------------------

def fdir(cza_arr, czda_arr, rsds, date) -> np.ndarray:
    """
    cza_arr: temporal average cosine zenith angle during each interval
    czda_arr: temporal average cosine zenith angle during only the sunlit part of each interval
    rsds: surface downward solar radiation (W/m²)
    date: date and time series
    Return the ratio of direct solar radiation.
    """
    cza_arr = np.asarray(cza_arr, dtype=np.float64)
    czda_arr = np.asarray(czda_arr, dtype=np.float64)
    rsds = np.asarray(rsds, dtype=np.float64)

    if cza_arr.ndim != 3:
        cza3 = np.atleast_3d(cza_arr).transpose(2, 0, 1)
        czda3 = np.atleast_3d(czda_arr).transpose(2, 0, 1)
        rsds3 = np.atleast_3d(rsds).transpose(2, 0, 1)
        return _fdir_3d(cza3, czda3, rsds3, date).squeeze()
    return _fdir_3d(cza_arr, czda_arr, rsds, date)


def _fdir_3d(cosza_arr, coszda_arr, rsds, date) -> np.ndarray:
    shape = rsds.shape
    f = np.zeros(shape, dtype=np.float64)
    cos895 = math.cos(89.5 / 180 * PI)
    for i in range(shape[0]):
        d = _sunearth(date[i])
        for j in range(shape[1]):
            for k in range(shape[2]):
                if cosza_arr[i, j, k] <= cos895 or rsds[i, j, k] <= 0:
                    f[i, j, k] = 0.0
                else:
                    s_star = min(rsds[i, j, k] / (1367 * coszda_arr[i, j, k] * (d ** (-2))), 0.85)
                    val = math.exp(3 - 1.34 * s_star - 1.65 / s_star)
                    f[i, j, k] = max(min(val, 0.9), 0.0)
    return f


# ---------------------------------------------------------------------------
# Public API — mirrors the original Cython module interface exactly
# ---------------------------------------------------------------------------

def _prepare_3d(*arrays):
    """Ensure all arrays are float64 and 3-D (T, Y, X)."""
    return [np.asarray(a, dtype=np.float64) for a in arrays]


def Tg_GCM(tas, ps, sfcwind, rsds, rsus, rlds, rlus, fdir_arr, cosz_arr, is2mwind):
    """
    Compute outdoor black globe temperature (K) — GCM method.

    Parameters
    ----------
    tas       : air temperature (K)
    ps        : surface pressure (Pa)
    sfcwind   : wind speed (m/s)
    rsds      : surface downward solar radiation (W/m²)
    rsus      : surface reflected solar radiation (W/m²)
    rlds      : surface downward long-wave radiation (W/m²)
    rlus      : surface upwelling long-wave radiation (W/m²)
    fdir_arr  : ratio of direct solar radiation
    cosz_arr  : cosine zenith angle
    is2mwind  : True for 2 m wind, False for 10 m wind
    """
    tas = np.asarray(tas, dtype=np.float64)
    if tas.ndim != 3:
        arrs = [np.atleast_3d(a) for a in [tas, ps, sfcwind, rsds, rsus, rlds, rlus,
                                            fdir_arr, cosz_arr]]
        return _Tg_GCM_core(*arrs, use_2m_wind=is2mwind).squeeze()
    return _Tg_GCM_core(
        *_prepare_3d(tas, ps, sfcwind, rsds, rsus, rlds, rlus, fdir_arr, cosz_arr),
        use_2m_wind=is2mwind)


def Tnwb_GCM(tas, hurs, ps, sfcwind, rsds, rsus, rlds, rlus, fdir_arr, cosz_arr, is2mwind):
    """
    Compute outdoor natural wet-bulb temperature (K) — GCM method.

    Parameters
    ----------
    tas       : air temperature (K)
    hurs      : relative humidity (%)
    ps        : surface pressure (Pa)
    sfcwind   : wind speed (m/s)
    rsds      : surface downward solar radiation (W/m²)
    rsus      : surface reflected solar radiation (W/m²)
    rlds      : surface downward long-wave radiation (W/m²)
    rlus      : surface upwelling long-wave radiation (W/m²)
    fdir_arr  : ratio of direct solar radiation
    cosz_arr  : cosine zenith angle
    is2mwind  : True for 2 m wind, False for 10 m wind
    """
    tas = np.asarray(tas, dtype=np.float64)
    if tas.ndim != 3:
        arrs = [np.atleast_3d(a) for a in [tas, hurs, ps, sfcwind, rsds, rsus, rlds, rlus,
                                            fdir_arr, cosz_arr]]
        return _Tnwb_GCM_core(*arrs, use_2m_wind=is2mwind).squeeze()
    return _Tnwb_GCM_core(
        *_prepare_3d(tas, hurs, ps, sfcwind, rsds, rsus, rlds, rlus, fdir_arr, cosz_arr),
        use_2m_wind=is2mwind)


def Tg_Liljegren(tas, hurs, ps, sfcwind, rsds, fdir_arr, cosz_arr, is2mwind):
    """
    Compute outdoor black globe temperature (K) — Liljegren method.

    Parameters
    ----------
    tas       : air temperature (K)
    hurs      : relative humidity (%)
    ps        : surface pressure (Pa)
    sfcwind   : wind speed (m/s)
    rsds      : surface downward solar radiation (W/m²)
    fdir_arr  : ratio of direct solar radiation
    cosz_arr  : cosine zenith angle
    is2mwind  : True for 2 m wind, False for 10 m wind
    """
    tas = np.asarray(tas, dtype=np.float64)
    if tas.ndim != 3:
        arrs = [np.atleast_3d(a) for a in [tas, hurs, ps, sfcwind, rsds, fdir_arr, cosz_arr]]
        return _Tg_Liljegren_core(*arrs, use_2m_wind=is2mwind).squeeze()
    return _Tg_Liljegren_core(
        *_prepare_3d(tas, hurs, ps, sfcwind, rsds, fdir_arr, cosz_arr),
        use_2m_wind=is2mwind)


def Tnwb_Liljegren(tas, hurs, ps, sfcwind, rsds, fdir_arr, cosz_arr, is2mwind):
    """
    Compute outdoor natural wet-bulb temperature (K) — Liljegren method.

    Parameters
    ----------
    tas       : air temperature (K)
    hurs      : relative humidity (%)
    ps        : surface pressure (Pa)
    sfcwind   : wind speed (m/s)
    rsds      : surface downward solar radiation (W/m²)
    fdir_arr  : ratio of direct solar radiation
    cosz_arr  : cosine zenith angle
    is2mwind  : True for 2 m wind, False for 10 m wind
    """
    tas = np.asarray(tas, dtype=np.float64)
    if tas.ndim != 3:
        arrs = [np.atleast_3d(a) for a in [tas, hurs, ps, sfcwind, rsds, fdir_arr, cosz_arr]]
        return _Tnwb_Liljegren_core(*arrs, use_2m_wind=is2mwind).squeeze()
    return _Tnwb_Liljegren_core(
        *_prepare_3d(tas, hurs, ps, sfcwind, rsds, fdir_arr, cosz_arr),
        use_2m_wind=is2mwind)


def WBGT_Liljegren(tas, hurs, ps, sfcwind, rsds, fdir_arr, cosz_arr, is2mwind):
    """
    Compute outdoor wet-bulb globe temperature (K) — Liljegren method.
    WBGT = 0.7 * Tnwb + 0.2 * Tg + 0.1 * Tair

    Parameters
    ----------
    tas       : air temperature (K)
    hurs      : relative humidity (%)
    ps        : surface pressure (Pa)
    sfcwind   : wind speed (m/s)
    rsds      : surface downward solar radiation (W/m²)
    fdir_arr  : ratio of direct solar radiation
    cosz_arr  : cosine zenith angle
    is2mwind  : True for 2 m wind, False for 10 m wind
    """
    tg = Tg_Liljegren(tas, hurs, ps, sfcwind, rsds, fdir_arr, cosz_arr, is2mwind)
    tnwb = Tnwb_Liljegren(tas, hurs, ps, sfcwind, rsds, fdir_arr, cosz_arr, is2mwind)
    return 0.7 * tnwb + 0.2 * tg + 0.1 * np.asarray(tas, dtype=np.float64)


def WBGT_GCM(tas, hurs, ps, sfcwind, rsds, rsus, rlds, rlus, fdir_arr, cosz_arr, is2mwind):
    """
    Compute outdoor wet-bulb globe temperature (K) — GCM method.
    WBGT = 0.7 * Tnwb + 0.2 * Tg + 0.1 * Tair

    Parameters
    ----------
    tas       : air temperature (K)
    hurs      : relative humidity (%)
    ps        : surface pressure (Pa)
    sfcwind   : wind speed (m/s)
    rsds      : surface downward solar radiation (W/m²)
    rsus      : surface reflected solar radiation (W/m²)
    rlds      : surface downward long-wave radiation (W/m²)
    rlus      : surface upwelling long-wave radiation (W/m²)
    fdir_arr  : ratio of direct solar radiation
    cosz_arr  : cosine zenith angle
    is2mwind  : True for 2 m wind, False for 10 m wind
    """
    tg = Tg_GCM(tas, ps, sfcwind, rsds, rsus, rlds, rlus, fdir_arr, cosz_arr, is2mwind)
    tnwb = Tnwb_GCM(tas, hurs, ps, sfcwind, rsds, rsus, rlds, rlus, fdir_arr, cosz_arr, is2mwind)
    return 0.7 * tnwb + 0.2 * tg + 0.1 * np.asarray(tas, dtype=np.float64)
