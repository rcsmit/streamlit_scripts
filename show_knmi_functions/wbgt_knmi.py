from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import pandas as pd
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import plotly.graph_objects as go

# try:
# if 1==1:
    from utils import get_data, getdata_wrapper, check_from_until, calculate_heat_index, calculate_wind_chill, celsius_to_fahrenheit, fahrenheit_to_celsius
    from solar_app import solar_wrapper
    from liljegren_wbgt import wbgt_liljegren_from_station, KNMI_STATIONS, wbgt_liljegren
    from select_time_place import select_time_place
    from replicate_knmi_wbgt import show_historical_data
# except:
#     from show_knmi_functions.utils import calculate_heat_index, calculate_wind_chill, celsius_to_fahrenheit, fahrenheit_to_celsius
#     from show_knmi_functions.solar_app import solar_wrapper
#     from show_knmi_functions.liljegren_wbgt import wbgt_liljegren_from_station, KNMI_STATIONS, wbgt_liljegren
#     from show_knmi_functions.select_time_place import select_time_place
#     from show_knmi_functions.replicate_knmi_wbgt import show_historical_data
# version : 20260526-120000 - Initial version: WBGT berekening met KNMI dagdata
current_version = "20260526-120000"



# ---------------------------------------------------------------------------
# Hulpfuncties
# ---------------------------------------------------------------------------

# Function to determine the feels-like temperature
def feels_like_temperature(T_C: float, RH: float, wind_ms: Optional[float]) -> float:
    # Dictionary to map temp_type to the corresponding column

    # Get the temperature based on the temp_type
   
    # Calculate average relative humidity, considering missing values
    
    # Convert wind speed from m/s to mph (default to 0 if wind_max is missing)
    V_mph = wind_ms * 2.23694  # Default to 0 if wind_max is missing

    # Convert Celsius to Fahrenheit
    T_F = celsius_to_fahrenheit(T_C)

    # Determine feels-like temperature
    if T_F >= 80 and RH is not None:
        # Calculate Heat Index
        feels_like_F = calculate_heat_index(T_F, RH)
    elif T_F <= 50 and V_mph >= 3:
        # Calculate Wind Chill
        feels_like_F = calculate_wind_chill(T_F, V_mph)
    else:
        feels_like_F = T_F  # No adjustment

    # Convert back to Celsius
    feels_like_C = fahrenheit_to_celsius(feels_like_F)
    return feels_like_C

def _dampdruk_hpa(temp_c: float, rh_pct: float) -> float:
    """Actuele dampdruk in hPa via Magnus-formule.

    Args:
        temp_c:  Luchttemperatuur in °C.
        rh_pct:  Relatieve vochtigheid in % (0–100).

    Returns:
        Dampdruk in hPa.
    """
    # Verzadigingsdampdruk (hPa) — Magnus
    e_sat = 6.1078 * math.exp(17.27 * temp_c / (temp_c + 237.3))
    return (rh_pct / 100.0) * e_sat


def _nat_bol_temp(temp_c: float, rh_pct: float) -> float:
    """Natuurlijke nat-bol temperatuur (Tnw) benadering.

    Stull (2011) psychrometrische benadering:
      Tnw ≈ T * atan(0.151977 * (RH + 8.313659)^0.5)
           + atan(T + RH) - atan(RH - 1.676331)
           + 0.00391838 * RH^1.5 * atan(0.023101 * RH) - 4.686035

    Args:
        temp_c:  Luchttemperatuur in °C.
        rh_pct:  Relatieve vochtigheid in % (0–100).

    Returns:
        Nat-bol temperatuur in °C.
    """
    t = temp_c
    rh = rh_pct
    tnw = (
        t * math.atan(0.151977 * (rh + 8.313659) ** 0.5)
        + math.atan(t + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh)
        - 4.686035
    )
    return tnw

def _globe_temp(temp_c: float, wind_ms: float, q_wm2: float) -> float:
    """Schatting van de globe-temperatuur (Tg) op basis van stralingsbalans.

    Gebaseerd op Liljegren et al. (2008), vereenvoudigd voor daggemiddelden:

      Tg ≈ T + 17.5 * (Q_W/m² / 1000)^0.6 / (1 + 1.1 * v^0.6)

    (waarbij Q_W/m² = Q_J/cm² * 10000 / 3600  (J/cm² per uur → W/m²) al eerder berekend)

       Args:
        temp_c:   Luchttemperatuur in °C.
        wind_ms:  Windsnelheid in m/s.
        q_wm2:   Globale straling in W/m² 

    Returns:
        Geschatte globe-temperatuur in °C.
    """
    v = max(wind_ms, 0.1)
   
    delta_tg = (
        17.5
        * (q_wm2 / 1000.0) ** 0.6
        / (1.0 + 1.1 * v ** 0.6)
    )

    return temp_c + delta_tg
    



# ---------------------------------------------------------------------------
# Hoofd-WBGT functies
# ---------------------------------------------------------------------------



def wbgt_buiten(
    temp_c: float,
    rh_pct: float,
    wind_ms: float,
    q_wm2: float,
    stn: int = 260,           # KNMI-stationnummer (default De Bilt)
    dt: "datetime | None" = None,   # UTC datetime; None → gebruik nu
    pressure_hpa: float = 1013.25,
) -> float:
    """WBGT buiten (zon) — volledige Liljegren et al. (2008) methode.
 
    WBGT = 0.7·Tw + 0.2·Tg + 0.1·Ta
 
    Inputs:
        temp_c:       Droge-bol temperatuur [°C].
        rh_pct:       Relatieve vochtigheid [%].
        wind_ms:      Windsnelheid [m s⁻¹].
        q_wm2:        Globale straling [W m⁻²].
        stn:          KNMI-stationnummer voor lat/lon lookup.
        dt:           UTC datum/tijd van de meting.
        pressure_hpa: Luchtdruk [hPa].
 
    Returns:
        WBGT [°C].
    """
    from datetime import datetime as _dt
    if dt is None:
        dt = _dt.utcnow()
        #dt = _dt.now(datetime.timezone.utc)
    return wbgt_liljegren_from_station(
        temp_c, rh_pct, wind_ms, q_wm2, stn, dt, pressure_hpa
    )
 
def wbgt_buiten_oud(temp_c: float, rh_pct: float, wind_ms: float, q_wm2: float) -> float:
    """WBGT buiten (zon) — Liljegren/ISO 7243.

      WBGT = 0.7 * Tnw + 0.2 * Tg + 0.1 * Td

    Args:
        temp_c:   Droge-bol temperatuur in °C.
        rh_pct:   Relatieve vochtigheid in %.
        wind_ms:  Windsnelheid in m/s.
        # q_jcm2:   Globale straling in J/cm² (KNMI kolom Q).
        q_wm2       Globale straling in W/m² (KNMI kolom Q).
    Returns:
        WBGT in °C.
    """
    tnw = _nat_bol_temp(temp_c, rh_pct)
    tg  = _globe_temp(temp_c, wind_ms, q_wm2)
    td  = temp_c
    return 0.7 * tnw + 0.2 * tg + 0.1 * td


def wbgt_schaduw(temp_c: float, rh_pct: float) -> float:
    """WBGT in de schaduw / binnenshuis — ISO 7243.

      WBGT = 0.7 * Tnw + 0.3 * Tg  (Tg ≈ Td in schaduw)

    Args:
        temp_c:  Droge-bol temperatuur in °C.
        rh_pct:  Relatieve vochtigheid in %.

    Returns:
        WBGT in °C.
    """
    tnw = _nat_bol_temp(temp_c, rh_pct)
    tg  = temp_c   # In schaduw: globe-temp ≈ luchttemperatuur
    return 0.7 * tnw + 0.3 * tg


def wbgt_bernard(temp_c: float, rh_pct: float) -> float:
    """Vereenvoudigde WBGT-benadering (Bernard & Pourmoghani, 1999).

    Handig als snelle check zonder stralingsdata:
      WBGT ≈ 0.567 * T + 0.393 * e + 3.94
    waarbij e = dampdruk in hPa.

    Args:
        temp_c:  Luchttemperatuur in °C.
        rh_pct:  Relatieve vochtigheid in %.

    Returns:
        WBGT in °C.
    """
    e = _dampdruk_hpa(temp_c, rh_pct)
    return 0.567 * temp_c + 0.393 * e + 3.94


# ---------------------------------------------------------------------------
# Risicoclassificatie (ISO 7243 / OSHA richtlijnen)
# ---------------------------------------------------------------------------

WBGT_DREMPELWAARDEN = [
    (18.0, "Laag",      "Geen beperkingen voor gezonde personen"),
    (23.0, "Matig",     "Let op bij zware arbeid of sport"),
    (28.0, "Hoog",      "Beperk inspanning; regelmatige rustpauzes"),
    (32.0, "Zeer hoog", "Alleen lichte arbeid; goede acclimatisatie vereist"),
    (float("inf"), "Gevaarlijk", "Vermijd fysieke inspanning buitenshuis"),
]

KNMI_DREMPELWAARDEN = [
    
    (14.0, "HK 0", "Laag risico"),
    (16.0, "HK 1", "Laag risico"),
    (18.0, "HK 2", "Laag risico"),
    (20.0, "HK 3", "Matig risico"),
    (22.0, "HK 4", "Matig risico"),
    (24.0, "HK 5", "Matig risico"),
    (26.0, "HK 6", "Hoog"),
    (28.0, "HK 7", "Hoog"),
    (30.0, "HK 8", "Zeer hoog"),
    (32.0, "HK 9", "Zeer hoog"),
    (float("inf"), "HK 10", "Gevaarlijk"),
]



def wbgt_risico(wbgt: float) -> tuple[str, str]:
    """Geeft risiconiveau en advies bij een WBGT-waarde.

    Args:
        wbgt: WBGT in °C.

    Returns:
        Tuple (risiconiveau: str, advies: str).
    """
    for grens, niveau, advies in KNMI_DREMPELWAARDEN:
        if wbgt < grens:
            return niveau, advies
    return "Gevaarlijk", "Vermijd fysieke inspanning buitenshuis"


# ---------------------------------------------------------------------------
# Vectorised versie voor een Pandas DataFrame
# ---------------------------------------------------------------------------

# def wbgt_bereken_df(df: pd.DataFrame) -> pd.DataFrame:
def wbgt_bereken_df(df: pd.DataFrame, stn: int = 260) -> pd.DataFrame:
    """Bereken WBGT voor een KNMI-dagdata DataFrame.

    Verwachte kolommen (KNMI ruwe eenheden):
      TG  : gemiddelde temperatuur  (0.1 °C)  → gedeeld door 10 → °C
      U  : relatieve vochtigheid   (%)
      F  : windsnelheid            (0.1 m/s) → gedeeld door 10 → m/s
      Q   : globale straling        (J/cm²)

    Optioneel (worden gebruikt als aanwezig):
      TX  : maximumtemperatuur      (0.1 °C)
      UX  : maximale vochtigheid    (%)

    Voegt toe aan het DataFrame:
      temp_c, wind_ms, wbgt_buiten, wbgt_schaduw, wbgt_bernard,
      wbgt_risico_niveau, wbgt_risico_advies

    Args:
        df: Pandas DataFrame met KNMI etmaalwaarden.

    Returns:
        Kopie van het DataFrame met extra WBGT-kolommen.
    """
    result = df.copy()

    # Converteer KNMI-eenheden
    result["temp_c"]   = result["T"] / 10.0
    result["wind_ms"]  = result["F"] / 10.0
    result["rh_pct"]   = result["U"].clip(0, 100)
    result["q_wm2"] = result["Q"].clip(lower=0) * 10_000 / 3600  # J/cm²/uur → W/m²

    # Converteer KNMI HH (1–24) naar correcte UTC datetime
    # HH=24 → volgende dag 00:00
    result["dt_utc"] = pd.to_datetime(
        result["YYYYMMDD"].astype(str), format="%Y-%m-%d"
    ) + pd.to_timedelta((result["HH"].clip(1, 24) - 1), unit="h") + pd.Timedelta(hours=1)
    # Uitleg: HH=1 = 00:00–01:00 → labelen als 01:00 (einde uur, KNMI-conventie)
    # HH=24 = 23:00–24:00 → labelen als volgende dag 00:00  ✓
    # WBGT-varianten (rij voor rij via apply voor leesbaarheid)

    # simpele versie    
    # result["wbgt_buiten"] = result.apply(
    #     lambda r: wbgt_buiten(r["temp_c"], r["rh_pct"], r["wind_ms"], r["q_wm2"]),
    #     axis=1,
    # ).round(1)

    
    # def _wbgt_bereken_df_patch(result, stn: int = 260):
    #     """Drop-in vervanging voor het wbgt_buiten-blok in wbgt_bereken_df()."""
    #     from datetime import datetime
 
    def _row_wbgt(r):
        dt = r["dt_utc"].to_pydatetime()
        return wbgt_buiten(r["temp_c"], r["rh_pct"], r["wind_ms"], r["q_wm2"],
                        stn=stn, dt=dt)
    def _row_wbgt_oud(r):
        try:
            # Bouw UTC datetime uit YYYYMMDD + HH (KNMI uurdata kolom HH = 1–24)
            
            hh = int(r.get("HH", 12))
            if hh == 24:          # KNMI codeert middernacht soms als uur 24
                hh = 0
            dt = datetime(
                int(str(r["YYYYMMDD"])[:4]),
                int(str(r["YYYYMMDD"])[4:6]),
                int(str(r["YYYYMMDD"])[6:8]),
                hh,
            )
        except Exception:
            dt = datetime.utcnow()
        return wbgt_buiten(
            r["temp_c"], r["rh_pct"], r["wind_ms"], r["q_wm2"],
            stn=stn, dt=dt,
        )
 
    result["wbgt_buiten"] = result.apply(_row_wbgt, axis=1).round(1)
    # return result

    result["wbgt_schaduw"] = result.apply(
        lambda r: wbgt_schaduw(r["temp_c"], r["rh_pct"]),
        axis=1,
    ).round(1)

    result["wbgt_bernard"] = result.apply(
        lambda r: wbgt_bernard(r["temp_c"], r["rh_pct"]),
        axis=1,
    ).round(1)

    result["feels_like_temperature"] = result.apply(
        lambda r: feels_like_temperature(r["temp_c"], r["rh_pct"], r["wind_ms"]),
        axis=1,
    ).round(1)
    # Risicoclassificatie op basis van buiten-WBGT
    risico = result["wbgt_buiten"].apply(wbgt_risico)
    result["wbgt_risico_niveau"] = risico.apply(lambda x: x[0])
    result["wbgt_risico_advies"] = risico.apply(lambda x: x[1])

    return result


# ---------------------------------------------------------------------------
# Risicozones (WBGT-drempelwaarden, ISO 7243)
# ---------------------------------------------------------------------------

RISICO_ZONES_WBGT = [
    {"label": "Laag",       "y_min":  0,   "y_max": 18,        "color": "rgba(144,238,144,0.20)"},  # lichtgroen
    {"label": "Matig",      "y_min": 18,   "y_max": 23,        "color": "rgba(255,255,102,0.25)"},  # geel
    {"label": "Hoog",       "y_min": 23,   "y_max": 28,        "color": "rgba(255,178,102,0.30)"},  # oranje
    {"label": "Zeer hoog",  "y_min": 28,   "y_max": 32,        "color": "rgba(255,102,102,0.35)"},  # rood
    {"label": "Gevaarlijk", "y_min": 32,   "y_max": 50,        "color": "rgba(180,  0,  0,0.30)"},  # donkerrood
]

RISICO_ZONES_KNMI = [
    {"label": "HK 1",  "y_min":  0,  "y_max": 16,  "color": "rgba(143,209, 79,0.20)"},  # lichtgroen
    {"label": "HK 2",  "y_min": 16,  "y_max": 18,  "color": "rgba( 74,138, 42,0.25)"},  # donkergroen
    {"label": "HK 3",  "y_min": 18,  "y_max": 20,  "color": "rgba(245,230, 66,0.30)"},  # geel
    {"label": "HK 4",  "y_min": 20,  "y_max": 22,  "color": "rgba(245,184,  0,0.35)"},  # goudgeel
    {"label": "HK 5",  "y_min": 22,  "y_max": 24,  "color": "rgba(240,128,  0,0.35)"},  # oranje
    {"label": "HK 6",  "y_min": 24,  "y_max": 26,  "color": "rgba(200, 90,  0,0.35)"},  # donkeroranje
    {"label": "HK 7",  "y_min": 26,  "y_max": 28,  "color": "rgba(160, 48,  0,0.35)"},  # roodbruin
    {"label": "HK 8",  "y_min": 28,  "y_max": 30,  "color": "rgba(122, 26, 26,0.40)"},  # donkerrood
    {"label": "HK 9",  "y_min": 30,  "y_max": 32,  "color": "rgba( 74, 10, 10,0.40)"},  # zeer donkerrood
    {"label": "HK 10", "y_min": 32,  "y_max": 50,  "color": "rgba(  0,  0,  0,0.35)"},  # zwart
]


BADGE_KLEUREN_WBGT = {
    "Laag":       "green",
    "Matig":      "orange",
    "Hoog":       "orange",
    "Zeer hoog":  "red",
    "Gevaarlijk": "red",
}

BADGE_KLEUREN_KNMI = {
    "HK 0":  "#8FD14F",   
    "HK 1":  "#8FD14F",
    "HK 2":  "#4A8A2A",
    "HK 3":  "#F5E642",
    "HK 4":  "#F5B800",
    "HK 5":  "#F08000",
    "HK 6":  "#C85A00",
    "HK 7":  "#A03000",
    "HK 8":  "#7A1A1A",
    "HK 9":  "#4A0A0A",
    "HK 10": "#000000",
}

ZONE_KLEUREN_WBGT = {z["label"]: z["color"] for z in RISICO_ZONES_WBGT}
ZONE_KLEUREN_KNMI = {z["label"]: z["color"] for z in RISICO_ZONES_KNMI}

# ---------------------------------------------------------------------------
# Plotly figuur
# ---------------------------------------------------------------------------

def maak_wbgt_figuur(df: pd.DataFrame, toon_temp: bool = True) -> go.Figure:
    """Maak een Plotly tijdreeksgrafiek met WBGT-lijnen en risicozones.

    Args:
        df:         DataFrame met kolommen YYYYMMDD, temp_c, wbgt_buiten,
                    wbgt_schaduw, wbgt_bernard.
        toon_temp:  Of de T-lijn (temp_c) getoond wordt.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    x = df["YYYYMMDD"]
    # x = df["YYYYMMDD"].astype(str) + " " + df["HH"].astype(str).str.zfill(2) + ":00"
    # # VOOR:
    # x = df["YYYYMMDD"].astype(str) + " " + df["HH"].astype(str).str.zfill(2) + ":00"

    # NA:
    x = df["dt_utc"]   # proper datetime → Plotly plot dit correct
    y_max_data = max(
        df[["wbgt_buiten", "wbgt_schaduw", "wbgt_bernard", "temp_c"]].max()
    )
    y_axis_max = max(y_max_data + 3, 35)

    # --- Risicozones als achtergrond (shapes) ---
    for zone in RISICO_ZONES_KNMI:
        y0 = zone["y_min"]
        y1 = min(zone["y_max"], y_axis_max)
        if y0 >= y_axis_max:
            continue
        fig.add_hrect(
            y0=y0, y1=y1,
            fillcolor=zone["color"],
            line_width=0,
            layer="below",
            annotation_text=zone["label"],
            annotation_position="right",
            annotation=dict(
                font_size=11,
                font_color="rgba(80,80,80,0.8)",
                xanchor="right",
                x=1.0,
            ),
        )

    # --- Drempellijnen (stippel) ---
    for zone in RISICO_ZONES_KNMI[1:]:  # sla "Laag" ondergrens over
        fig.add_hline(
            y=zone["y_min"],
            line_dash="dot",
            line_color="rgba(120,120,120,0.5)",
            line_width=1,
        )

    # --- TX lijn ---
    if toon_temp:
        fig.add_trace(go.Scatter(
            x=x, y=df["temp_c"],
            name="Temperatuur (T)",
            mode="lines",
            line=dict(color="#4a90d9", width=1.5, dash="dot"),
            opacity=0.7,
            hovertemplate="%{x|%d %b %Y}<br>Temp: %{y:.1f} °C<extra></extra>",
        ))

    # --- WBGT buiten ---
    fig.add_trace(go.Scatter(
        x=x, y=df["wbgt_buiten"],
        name="WBGT buiten (zon)",
        mode="lines",
        line=dict(color="#e05c00", width=2.5),
        hovertemplate="%{x|%d %b %Y}<br>WBGT buiten: %{y:.1f} °C<extra></extra>",
    ))

    # --- WBGT schaduw ---
    fig.add_trace(go.Scatter(
        x=x, y=df["wbgt_schaduw"],
        name="WBGT schaduw",
        mode="lines",
        line=dict(color="#9b59b6", width=2),
        hovertemplate="%{x|%d %b %Y}<br>WBGT schaduw: %{y:.1f} °C<extra></extra>",
    ))

    # --- WBGT Bernard ---
    fig.add_trace(go.Scatter(
        x=x, y=df["wbgt_bernard"],
        name="WBGT Bernard",
        mode="lines",
        line=dict(color="#27ae60", width=2, dash="dash"),
        hovertemplate="%{x|%d %b %Y}<br>WBGT Bernard: %{y:.1f} °C<extra></extra>",
    ))


    # --- feels like ---
    fig.add_trace(go.Scatter(
        x=x, y=df["feels_like_temperature"],
        name="feels_like_temperature",
        mode="lines",
        line=dict(color="#ae3e6e", width=2, dash="dash"),
        hovertemplate="%{x|%d %b %Y}<br>Feels like: %{y:.1f} °C<extra></extra>",
    ))


    fig.update_layout(
        title=dict(
            text="Wet Bulb Globe Temperature met risicozones",
            font_size=16,
            x=0.0,
        ),
        xaxis=dict(
            title="Datum",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            # tickformat="%b %Y",
        ),
        yaxis=dict(
            title="Temperatuur (°C)",
            range=[min(df["temp_c"].min() - 2, 0), y_axis_max],
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
        ),
        # legend=dict(
        #     orientation="h",
        #     yanchor="bottom",
        #     y=1.02,
        #     xanchor="left",
        #     x=0,
        #     font_size=12,
        # ),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=140, t=80, b=60),
        height=520,
    )

    return fig

def maak_wbgt_barchart(df: pd.DataFrame, datum: str | None = None) -> go.Figure:
    """Staafdiagram van uurlijkse WBGT met hittekracht-kleuren, zoals KNMI-rapport Fig.

    Args:
        df:    DataFrame zoals geproduceerd door wbgt_bereken_df(), met kolom
               dt_utc, wbgt_buiten, wbgt_risico_niveau, HH.
        datum: 'YYYY-MM-DD' string om één dag te selecteren. Als None: laatste dag.

    Returns:
        Plotly Figure.
    """
    # --- Daginschnitt ---
    df = df.copy()
    df["_date_str"] = df["dt_utc"].dt.strftime("%Y-%m-%d")

    if datum is None:
        datum = df["_date_str"].iloc[-1]

    dag = df[df["_date_str"] == datum].copy()
    if dag.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"Geen data voor {datum}", showarrow=False,
                           font_size=16, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig

    # Lokale tijd (CEST/CET) voor x-as label — gebruik HH direct uit KNMI
    # HH is al het einduur van het interval (1=00–01 → label "1")
    dag["hh_label"] = dag["HH"].astype(int)

    # Kleur per balk op basis van wbgt_risico_niveau
    dag["kleur"] = dag["wbgt_risico_niveau"].map(BADGE_KLEUREN_KNMI).fillna("#cccccc")

    # HK-getal uit niveau ("HK 7" → "7")
    dag["hk_getal"] = dag["wbgt_risico_niveau"].str.replace("HK ", "", regex=False)
    
    fig = go.Figure()

    # Één trace per HK-niveau zodat de legenda klopt
    for niveau, kleur in BADGE_KLEUREN_KNMI.items():
        subset = dag[dag["wbgt_risico_niveau"] == niveau]
        if subset.empty:
            continue
        fig.add_trace(go.Bar(
            x=subset["hh_label"],
            y=subset["wbgt_buiten"],
            name=niveau,
            marker_color=kleur,
            text=subset["hk_getal"],
            textposition="outside",
            textfont=dict(size=11, color="#333333"),
            hovertemplate=(
                "Uur: %{x}<br>"
                "WBGT: %{y:.1f} °C<br>"
                f"Niveau: {niveau}<extra></extra>"
            ),
            width=0.7,
        ))

    # Datum in titel — ook lokale datum tonen
    fig.update_layout(
        title=dict(
            text=f"Uurlijkse WBGT en Hittekracht — {datum}",
            font_size=15,
            x=0.0,
        ),
        xaxis=dict(
            title="Tijd (uur, lokale tijd CEST)",
            tickmode="array",
            tickvals=dag["hh_label"].tolist(),
            ticktext=[str(h) for h in dag["hh_label"].tolist()],
            showgrid=False,
        ),
        yaxis=dict(
            title="WBGT (°C)",
            range=[0, max(dag["wbgt_buiten"].max() + 4, 35)],
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
        ),
        barmode="overlay",
        bargap=0.1,
        legend=dict(
            title="Hittekracht",
            traceorder="normal",
            font_size=11,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=40, t=60, b=60),
        height=450,
        showlegend=True,
    )

    return fig

def maak_wbgt_maand_barchart(df: pd.DataFrame) -> go.Figure:
    """Gestapeld 100%-staafdiagram van hittekracht per maand.

    Toont het aandeel uren per hittekracht-niveau per maand,
    zoals in het KNMI-rapport.

    Args:
        df: DataFrame zoals geproduceerd door wbgt_bereken_df(),
            met kolommen dt_utc en wbgt_risico_niveau.

    Returns:
        Plotly Figure.
    """
    df = df.copy()
    df["maand"] = df["dt_utc"].dt.month

    MAAND_LABELS = {
        1: "Jan", 2: "Feb", 3: "Mrt", 4: "Apr",
        5: "Mei", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Okt", 11: "Nov", 12: "Dec",
    }

    # Alle HK-niveaus in volgorde
    alle_niveaus = [f"HK {i}" for i in range(1, 11)]
    alle_maanden = list(range(1, 13))

    # Tel uren per maand per niveau
    counts = (
        df.groupby(["maand", "wbgt_risico_niveau"])
        .size()
        .reset_index(name="n")
    )

    # Draai naar matrix (maand × niveau)
    pivot = counts.pivot(index="maand", columns="wbgt_risico_niveau", values="n").fillna(0)

    # Zorg dat alle niveaus aanwezig zijn
    for niveau in alle_niveaus:
        if niveau not in pivot.columns:
            pivot[niveau] = 0
    pivot = pivot[alle_niveaus]  # juiste volgorde

    # Normaliseer naar 100%
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig = go.Figure()

    for niveau in alle_niveaus:
        kleur = BADGE_KLEUREN_KNMI.get(niveau, "#cccccc")
        hk_getal = niveau.replace("HK ", "")

        y_vals = [pivot_pct.loc[m, niveau] if m in pivot_pct.index else 0
                  for m in alle_maanden]

        # Label alleen tonen als aandeel > 4% (anders te krap)
        tekst = [hk_getal if v > 4 else "" for v in y_vals]
        tekstkleur = "black" if niveau in ("HK 1", "HK 2", "HK 3") else "white"
            
        fig.add_trace(go.Bar(
            name=niveau,
            x=[MAAND_LABELS[m] for m in alle_maanden],
            y=y_vals,
            marker_color=kleur,
            text=tekst,
            textposition="inside",
            insidetextanchor="middle",
            # textfont=dict(size=11, color=tekstkleur),
            textfont=dict(size=11, color="white"),
            hovertemplate=(
                "%{x}<br>"
                f"{niveau}<br>"
                "Aandeel: %{y:.1f}%<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="stack",
        title=dict(
            text="Verdeling Hittekracht per maand",
            font_size=15,
            x=0.0,
        ),
        xaxis=dict(
            title="Maand",
            showgrid=False,
        ),
        yaxis=dict(
            title="Aandeel (%)",
            range=[0, 100],
            ticksuffix="%",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
        ),
        legend=dict(
            title="Hittekracht",
            traceorder="normal",
            font_size=11,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=40, t=60, b=60),
        height=450,
        showlegend=True,
    )
    

    return fig
# ---------------------------------------------------------------------------
# KPI-berekeningen
# ---------------------------------------------------------------------------

def _bereken_kpis(df: pd.DataFrame) -> dict:
    """Bereken samenvattende statistieken voor de KPI-row."""
    wbgt = df["wbgt_buiten"]
    counts = df["wbgt_risico_niveau"].value_counts()
    gevaarlijke_dagen = int(
        counts.get("Gevaarlijk", 0) + counts.get("Zeer hoog", 0)
    )
    return {
        "max_wbgt":          round(wbgt.max(), 1),
        "gem_wbgt":          round(wbgt.mean(), 1),
        "dagen_hoog_plus":   int((wbgt >= 23).sum()),
        "gevaarlijke_dagen": gevaarlijke_dagen,
        "meest_voorkomend":  df["wbgt_risico_niveau"].mode().iloc[0]
                             if not df.empty else "–",
    }


# ---------------------------------------------------------------------------
# Publieke render-functie
# ---------------------------------------------------------------------------

def render_wbgt_chart(df: pd.DataFrame) -> None:
    """Render de volledige WBGT-sectie in Streamlit.

    Args:
        df: DataFrame zoals geproduceerd door wbgt_bereken_df().
    """
    # --- Filters in sidebar ---
    with st.sidebar:
        st.subheader(":material/thermostat: Filters")

        jaren = sorted(df["YYYY"].unique()) if "YYYY" in df.columns else []
        if jaren:
            jaar_keuze = st.multiselect(
                "Jaar", jaren, default=jaren[-3:] if len(jaren) >= 3 else jaren
            )
            df = df[df["YYYY"].isin(jaar_keuze)] if jaar_keuze else df

        maanden = list(range(1, 13))
        maand_labels = {
            1: "Jan", 2: "Feb", 3: "Mrt", 4: "Apr",
            5: "Mei", 6: "Jun", 7: "Jul", 8: "Aug",
            9: "Sep", 10: "Okt", 11: "Nov", 12: "Dec",
        }
        maand_keuze = st.multiselect(
            "Maand",
            options=maanden,
            default=[1,2,3,4,5, 6, 7, 8, 9,10,11,12],
            format_func=lambda m: maand_labels[m],
        )
        if maand_keuze and "MM" in df.columns:
            df = df[df["MM"].isin(maand_keuze)]

        toon_temp = st.toggle("Toon luchttemperatuur", value=True)

        st.caption(f"v{current_version}")

    if df.empty:
        st.warning("Geen data voor de geselecteerde filters.")
        return

    # --- KPI row ---
    kpis = _bereken_kpis(df)
    badge_kleur = BADGE_KLEUREN_KNMI.get(kpis["meest_voorkomend"], "gray")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        ":material/thermometer: Max WBGT buiten",
        f"{kpis['max_wbgt']} °C",
        border=True,
    )
    c2.metric(
        ":material/bar_chart: Gem. WBGT buiten",
        f"{kpis['gem_wbgt']} °C",
        border=True,
    )
    c3.metric(
        ":material/warning: Dagen WBGT ≥ 23 (Hoog+)",
        kpis["dagen_hoog_plus"],
        border=True,
    )
    c4.metric(
        ":material/dangerous: Dagen Zeer hoog/Gevaarlijk",
        kpis["gevaarlijke_dagen"],
        border=True,
    )

    # Meest voorkomend risiconiveau als badge
    st.markdown(
        f"""
        <div style="
            background-color: {badge_kleur};
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            display: inline-block;
        ">
            🌡️ Meest voorkomend risiconiveau: {kpis['meest_voorkomend']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.space("small")

    # --- Grafiek ---
    fig = maak_wbgt_figuur(df, toon_temp=toon_temp)
    st.plotly_chart(fig, width="stretch")

    with st.expander(":material/science: Drie WBGT-methoden"):
        st.dataframe(
            pd.DataFrame([
                {"Functie": "wbgt_buiten()",  "Methode": "Liljegren/ISO 7243 (0.7·Tnw + 0.2·Tg + 0.1·Td)", "Wanneer": "Buiten in de zon — meest nauwkeurig"},
                {"Functie": "wbgt_schaduw()", "Methode": "ISO 7243 (0.7·Tnw + 0.3·Tg)",                     "Wanneer": "In schaduw of binnen"},
                {"Functie": "wbgt_bernard()", "Methode": "Bernard 1999 (dampdruk-formule)",                  "Wanneer": "Snelle check zonder stralingsdata"},
                  {"Functie": "feels_like_temperature", "Methode": "-",                  "Wanneer": "-"},
            ]),
            hide_index=True,
            width="stretch",
        )
        st.write("Tnw = nat bol temperatuur (natural wet bulb temperature) — vochtigheid + luchtbeweging")
        st.write("Tg = globe temperatuur — zwarte bol, meet stralingseffect")
        st.write("Td = droge bol temperatuur — gewone luchttemperatuur")
    # --- Risico-uitleg ---
    with st.expander(":material/info: Uitleg risicozones (ISO 7243)", expanded=False):
        uitleg = pd.DataFrame([
            {"Zone": "Laag",       "WBGT": "< 18 °C",  "Advies": "Geen beperkingen voor gezonde personen"},
            {"Zone": "Matig",      "WBGT": "18–23 °C", "Advies": "Let op bij zware arbeid of sport"},
            {"Zone": "Hoog",       "WBGT": "23–28 °C", "Advies": "Beperk inspanning; regelmatige rustpauzes"},
            {"Zone": "Zeer hoog",  "WBGT": "28–32 °C", "Advies": "Alleen lichte arbeid; goede acclimatisatie vereist"},
            {"Zone": "Gevaarlijk", "WBGT": "> 32 °C",  "Advies": "Vermijd fysieke inspanning buitenshuis"},
        ])
        st.dataframe(uitleg, hide_index=True, width="stretch")

    # --- Dagtabel (inklapbaar) ---
    with st.expander(":material/table: Dagdata", expanded=False):
        tabel_cols = ["id",
            "YYYYMMDD","HH", "temp_c", "rh_pct", "wind_ms","q_wm2",
            "wbgt_buiten", "wbgt_schaduw", "wbgt_bernard",
            "wbgt_risico_niveau",
        ]
        beschikbaar = [c for c in tabel_cols if c in df.columns]
        st.dataframe(
            df[beschikbaar].sort_values("YYYYMMDD", ascending=False),
            hide_index=True,
            width="stretch",
            column_config={
                "YYYYMMDD":           st.column_config.DateColumn("Datum", format="DD-MM-YYYY"),
                "temp_c":             st.column_config.NumberColumn("Temp (°C)", format="%.1f"),
                "rh_pct":             st.column_config.NumberColumn("RV (%)", format="%d"),
                "wind_ms":            st.column_config.NumberColumn("Wind (m/s)", format="%.1f"),
                "q_wm2":              st.column_config.NumberColumn("Q (W/m2)", format="%.1f"),
                "wbgt_buiten":        st.column_config.NumberColumn("WBGT buiten", format="%.1f"),
                "wbgt_schaduw":       st.column_config.NumberColumn("WBGT schaduw", format="%.1f"),
                "wbgt_bernard":       st.column_config.NumberColumn("WBGT Bernard", format="%.1f"),
                "wbgt_risico_niveau": st.column_config.TextColumn("Risico"),
                "feels_like_temperature":    st.column_config.NumberColumn("Feels like", format="%.1f"),
            },
        )

    # --- Dagselectie voor barchart ---
    beschikbare_dagen = sorted(df["dt_utc"].dt.strftime("%Y-%m-%d").unique())
    gekozen_dag = st.selectbox(
        "Dag voor uurdiagram",
        beschikbare_dagen,
        index=len(beschikbare_dagen) - 2,
    )
    fig_bar = maak_wbgt_barchart(df, datum=gekozen_dag)
    st.plotly_chart(fig_bar, width="stretch")

    fig_maand = maak_wbgt_maand_barchart(df)
    st.plotly_chart(fig_maand, width="stretch")

def info():
    st.info("""
Selectie van de bronnen:
  - Liljegren et al. (2008): Modeling the Wet Bulb Globe Temperature
    Using Standard Meteorological Measurements. J. Occup. Environ. Hyg.
  - Bernard & Pourmoghani (1999): Apparent Temperature and the WBGT
  - ISO 7243:2017 (Heat stress index)
  - KNMI daggegevens API: https://daggegevens.knmi.nl/klimatologie/daggegevens
""")

def scatterplots(df_dagmax, titel):
    import plotly.graph_objects as go

    fig = go.Figure()

    for kolom, kleur, naam in [
        ("wbgt_buiten",  "#e05c00", "WBGT buiten"),
        ("wbgt_schaduw", "#9b59b6", "WBGT schaduw"),
        ("wbgt_bernard", "#27ae60", "WBGT Bernard"),
        ("feels_like_temperature", "#ae2760", "Feels like"),
    ]:
        fig.add_trace(go.Scatter(
            x=df_dagmax["temp_c"],
            y=df_dagmax[kolom],
            mode="markers",
            name=naam,
            marker=dict(color=kleur, size=5, opacity=0.6),
            hovertemplate="temp: %{x:.1f} °C<br>" + naam + ": %{y:.1f} °C<extra></extra>",
        ))

    x_min = df_dagmax["temp_c"].min()
    x_max = df_dagmax["temp_c"].max()

    fig.add_trace(go.Scatter(
        x=[x_min, x_max],
        y=[x_min, x_max],
        mode="lines",
        name="x = y",
        line=dict(color="black", width=1, dash="dash"),
        hoverinfo="skip",
    ))
    fig.update_layout(
        title=f"WBGT vs luchttemperatuur - {titel}",
        xaxis_title="Temperatuur (°C)",
        yaxis_title="WBGT (°C)",
        hovermode="closest",
        plot_bgcolor="white",
        height=500,
    )

    st.plotly_chart(fig, width="stretch")

def referentie_tabel(lat,lon,dt_ref):
    """Replicatie van https://arielschecklist.com/wbgt-chart/ """
    import numpy as np
    import plotly.graph_objects as go

    temps = list(range(20, 36, 2))
    rhs   = list(range(0, 105, 5))
    col1,col2,col3=st.columns(3)
    with col1:
        wind  = st.number_input("Wind (m/s)", 0.50,100.0,2.0)
    with col2:
        q     = st.number_input("Straling (W/m2)", 0.0, 1500.0, 870.0) # 870.0  # W/m² — heldere hemel, hoogste waarde 2025
    with col3:
        what = st.selectbox("Welke index?", ["wbgt", "knmi"], index=1, format_func=lambda s: "WBGT" if s=="wbgt" else "KNMI hitte-index")
    
   
    # col1,col2,col3=st.columns(3)
    # with col1:
    #     datum_str = st.text_input("Datum referentietabel", "2026-05-15")
    # with col2:
    #     tijd_str  = st.text_input("Tijd (lokaal, NL)", "15:00")
    
    #     lokaal = datetime.strptime(f"{datum_str} {tijd_str}", "%Y-%m-%d %H:%M")
    #     lokaal_tz = lokaal.replace(tzinfo=ZoneInfo("Europe/Amsterdam"))
    #     dt_ref = lokaal_tz.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    #     st.write(f"UTC:{dt_ref}")
        
    # with col3:
    #     lat=st.number_input("Lat", -180.0, 180.0, 52.047 )
    #     lon=st.number_input("Lon", -180.0, 180.0,  5.177 )
        # Door:
        # dt_ref = datetime(2024, 7, 15, 11, 0)  # 13:00 lokaal ≈ 11:00 UTC
    # dt_ref is nu naive UTC datetime, klaar voor de solver
    # Bereken WBGT-matrix
    z = []
    for rh in rhs:
        rij = []
        for t in temps:
            
           
            # waarde = wbgt_buiten(float(t), float(rh), wind, q, stn=260, dt=dt_ref)
            waarde = wbgt_liljegren(float(t), float(rh), wind, q, lat,lon, dt_ref)
            

            
            
            rij.append(round(waarde, 1))
        z.append(rij)

    
    if what =="wbgt":
        zmin, zmax = 14, 32
    else:
        zmin, zmax = 14, 32

    def naar_schaal(v):
        return (v - zmin) / (zmax - zmin)
    
    
    colorscale_wbgt = [
        [0.0,                    "#ffffff"],
        [naar_schaal(18),        "#90ee90"],  # Laag
        [naar_schaal(18),        "#ffff66"],  # Matig
        [naar_schaal(23),        "#ffff66"],
        [naar_schaal(23),        "#ffb266"],  # Hoog
        [naar_schaal(28),        "#ffb266"],
        [naar_schaal(28),        "#ff6666"],  # Zeer hoog
        [naar_schaal(32),        "#ff6666"],
        [naar_schaal(32),        "#b40000"],  # Gevaarlijk
        [1.0,                    "#b40000"],
    ]
    


    colorscale_knmi = [
        [0.0,                    "#ffffff"],   # HK 0 : < 14
        [naar_schaal(14),        "#ffffff"],
        [naar_schaal(16),        "#8FD14F"],   # HK 1: 14-16
        [naar_schaal(18),        "#8FD14F"],
        [naar_schaal(18),        "#4A8A2A"],   # HK 2: 16–18
        [naar_schaal(20),        "#4A8A2A"],
        [naar_schaal(20),        "#F5E642"],   # HK 3: 18–20
        [naar_schaal(22),        "#F5E642"],
        [naar_schaal(22),        "#F5B800"],   # HK 4: 20–22
        [naar_schaal(24),        "#F5B800"],
        [naar_schaal(24),        "#F08000"],   # HK 5: 22–24
        [naar_schaal(26),        "#F08000"],
        [naar_schaal(26),        "#C85A00"],   # HK 6: 24–26
        [naar_schaal(28),        "#C85A00"],
        [naar_schaal(28),        "#A03000"],   # HK 7: 26–28
        [naar_schaal(30),        "#A03000"],
        [naar_schaal(30),        "#7A1A1A"],   # HK 8: 28–30
        [naar_schaal(32),        "#7A1A1A"],
        [naar_schaal(32),        "#4A0A0A"],   # HK 9: 30–32
        [naar_schaal(32),        "#000000"],   # HK 10: ≥ 32
        [1.0,                    "#000000"],
    ]

    colorscale_knmi = [
        [0.0,                    "#8FD14F"],   # < 14
        # [naar_schaal(14),        "#ffffff"],
        [naar_schaal(14),        "#8FD14F"],   # HK 1: 14-16
        # [naar_schaal(16),        "#8FD14F"],
        [naar_schaal(16),        "#4A8A2A"],   # HK 2: 16–18
        [naar_schaal(18),        "#4A8A2A"],
        [naar_schaal(18),        "#F5E642"],   # HK 3: 18–20
        [naar_schaal(20),        "#F5E642"],
        [naar_schaal(20),        "#F5B800"],   # HK 4: 20–22
        [naar_schaal(22),        "#F5B800"],
        [naar_schaal(22),        "#F08000"],   # HK 5: 22–24
        [naar_schaal(24),        "#F08000"],
        [naar_schaal(24),        "#C85A00"],   # HK 6: 24–26
        [naar_schaal(26),        "#C85A00"],
        [naar_schaal(26),        "#A03000"],   # HK 7: 26–28
        [naar_schaal(28),        "#A03000"],
        [naar_schaal(28),        "#7A1A1A"],   # HK 8: 28–30
        [naar_schaal(30),        "#7A1A1A"],
        [naar_schaal(30),        "#4A0A0A"],   # HK 9: 30–32
        [naar_schaal(32),        "#000000"],   # HK 10: ≥ 32
        [1.0,                    "#000000"],
    ]

    if what =="wbgt":
        zmin, zmax = 14, 32
        colorscale = colorscale_wbgt
        title = f"Wet Bulb Globe Temperature (WBGT) | wind = {wind} m/s, Q = {q} W/m², heldere hemel | lat:{lat},lon:{lon},UTC:{dt_ref}"
    else:
        zmin, zmax = 14, 32
        colorscale = colorscale_knmi
        title = f"KNMI Hitte Kracht | wind = {wind} m/s, Q = {q} W/m², heldere hemel | lat:{lat},lon:{lon},UTC:{dt_ref}"
    

    fig = go.Figure(go.Heatmap(
        z=z,
        x=temps,
        y=rhs,
        text=[[str(v) for v in rij] for rij in z],
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        showscale=False,
        xgap=2,
        ygap=2,
    ))

    fig.update_layout(
        title=dict(
            text=title,
            x=0,y=0,
            font_size=14,
        ),
        xaxis=dict(title="Temperatuur (°C)", tickvals=temps, tickmode="array",  side="top",),
        yaxis=dict(title="Relatieve vochtigheid (%)", tickvals=rhs, tickmode="array", autorange="reversed"),
        plot_bgcolor="white",
        height=600,
        margin=dict(l=80, r=40, t=100, b=60),
    )

    st.plotly_chart(fig, width="stretch")

def feels_like_all(temp_c: float, rh_pct: float, wind_ms: float, q_wm2: float, lat: float, lon: float, utc_dt: datetime) -> dict:
    """Bereken alle vier temperatuurindices voor gegeven omstandigheden.

    Args:
        temp_c:   Luchttemperatuur in °C
        rh_pct:   Relatieve vochtigheid in % (0–100)
        wind_ms:  Windsnelheid in m/s
        q_wm2:    Globale straling in W/m²

    Returns:
        Dictionary met wbgt_buiten, wbgt_schaduw, wbgt_bernard, feels_like
    """
    return {
        "wbgt_buiten":  round(wbgt_liljegren(temp_c, rh_pct, wind_ms, q_wm2, lat, lon, utc_dt), 1),
        "wbgt_schaduw": round(wbgt_schaduw(temp_c, rh_pct), 1),
        "wbgt_bernard": round(wbgt_bernard(temp_c, rh_pct), 1),
        "feels_like":   round(feels_like_temperature(temp_c, rh_pct, wind_ms), 1),
    }
    
def feels_like_calculator(lat,lon,utc_dt):
    st.subheader(":material/thermostat: Voeltemperatuur calculator")

    c1, c2, c3, c4 = st.columns(4)
    temp_c  = c1.number_input("Temperatuur (°C)",  value=28.0, min_value=-20.0, max_value=50.0, step=0.5)
    rh_pct  = c2.number_input("Luchtvochtigheid (%)", value=50.0, min_value=0.0,  max_value=100.0, step=1.0)
    wind_ms = c3.number_input("Windsnelheid (m/s)", value=2.0,  min_value=0.5,  max_value=30.0, step=0.5)
    q_wm2   = c4.number_input("Straling (W/m²)",   value=800.0, min_value=0.0,  max_value=1400.0, step=10.0)

    # col1,col2,col3=st.columns(3)
    # with col1:
    #     datum_str = st.text_input("Datum referentietabel", "2024-07-15")
    # with col2:
    #     tijd_str  = st.text_input("Tijd (lokaal, NL)", "13:00")
    # with col3:
    #     lokaal = datetime.strptime(f"{datum_str} {tijd_str}", "%Y-%m-%d %H:%M")
    #     lokaal_tz = lokaal.replace(tzinfo=ZoneInfo("Europe/Amsterdam"))
    #     dt_ref = lokaal_tz.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    #     st.write(f"UTC:{dt_ref}")
    result = feels_like_all(temp_c, rh_pct, wind_ms, q_wm2, lat,lon,utc_dt)

    niveau, _ = wbgt_risico(result["wbgt_buiten"])
    badge_kleur = BADGE_KLEUREN_KNMI.get(niveau, "gray")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(":material/wb_sunny: WBGT buiten",  f"{result['wbgt_buiten']} °C",  border=True)
    c2.metric(":material/home: WBGT schaduw",     f"{result['wbgt_schaduw']} °C", border=True)
    c3.metric(":material/science: WBGT Bernard",  f"{result['wbgt_bernard']} °C", border=True)
    c4.metric(":material/air: Feels like",         f"{result['feels_like']} °C",   border=True)

    # st.markdown(f"Risiconiveau: :{badge_kleur}-badge[{niveau} {_}]")
    st.markdown(f"Risiconiveau: : {niveau} {_}")
    st.write(f"lat:{lat},lon:{lon},UTC:{utc_dt}")

# -------
# --------------------------------------------------------------------
# CLI-demo / quick test
# ---------------------------------------------------------------------------
def main_():

    stn = 260
    start_ = "2026-01-01"
    today = datetime.today().strftime("%Y-%m-%d")
    from__ = st.sidebar.text_input("start datum (yyyy-mm-dd) from 1-1-1900", start_, key="wbgt1")
    until__ = st.sidebar.text_input("end datum (yyyy-mm-dd)", today, key="wbgt2")

    #fromx, until = check_from_until(from_, until_)
    fromx = from__.replace("-", "")
    until = until__.replace("-", "")
    only_dagmax = st.sidebar.toggle("Alleen maximale HK's per dag",value=True)
    
    url = f"https://www.daggegevens.knmi.nl/klimatologie/uurgegevens?stns={stn}&vars=T:U:FH:Q&start={fromx}00&end={until}23"
    try:
        df = pd.read_csv(
                url,
                delimiter=",",
                header= None,
                comment="#",
                low_memory=False,
            )
    except:
        st.error(f"Fout bij het inlezen van de gegevens. URL: {url}")
        st.stop()
    
    column_replacements = [
        [0, "STN"],
        [1, "YYYYMMDD"],

        [2, "HH"],
        [3, "T"],
        [4, "U"],
        [5, "F"],
        [6, "Q"],
        
    ]

    for c in column_replacements:
        df = df.rename(columns={c[0]: c[1]})
    
    df["YYYYMMDD"] = pd.to_datetime(df["YYYYMMDD"].astype(str))
    df["YYYY"] = df["YYYYMMDD"].dt.year
    df["MM"] = df["YYYYMMDD"].dt.month
    df["DD"] = df["YYYYMMDD"].dt.day
    df["dayofyear"] = df["YYYYMMDD"].dt.dayofyear
    df["id"] =  range(1, len(df) + 1)
    # df_result = wbgt_bereken_df(df)
    df_result = wbgt_bereken_df(df, stn=260)
    if only_dagmax:
        df_dagmax = df_result.loc[df_result.groupby("YYYYMMDD")["wbgt_buiten"].idxmax()].reset_index(drop=True)
    else:
        df_dagmax = df_result
    # print (df_result.dtypes)
    render_wbgt_chart(df_dagmax)
  
    for w in ["wbgt_risico_niveau", "wbgt_risico_advies"]:
        counts = df_dagmax.groupby(w).size().reset_index(name="aantal")
        counts["pct"] = (counts["aantal"] / len(df_dagmax) * 100).round(2)
        with st.expander(f"Opgeslitst naar {w}"):
            st.write (counts)
    scatterplots(df_dagmax, "webgt_buiten-max")
    scatterplots(df_result, "alle waardes")

def show_info():
    st.subheader("Info")
    st.info("Voor uitleg, achtergrond informatie en referenties: https://rene-smit.com/hitte-meet-je-niet-met-een-thermometer/")
    st.info("Script : https://github.com/rcsmit/streamlit_scripts/blob/main/show_knmi_functions/wbgt_knmi.py (zie ook de scripts bij imports, in dezelfde directory)")

def wbgt_knmi():
    with st.sidebar:
        lat,lon,utc_dt, loc_name, selected_date, selected_time,tz,LOCATIONS = select_time_place()

    tab1,tab2, tab3,tab4,tab5,tab6=st.tabs(["Main", "Tabel", "Calculator", "Solarinfo","1991-2025","INFO"])
    with tab2:
        referentie_tabel(lat,lon,utc_dt)
    with tab3:
        feels_like_calculator(lat,lon,utc_dt)
    with tab4:
        solar_wrapper(lat,lon,utc_dt, loc_name, selected_date, selected_time,tz,LOCATIONS)
    with tab5:
        show_historical_data()
    with tab6:
        show_info()
        info()
    with tab1:
        main_()
    
def main():
    wbgt_knmi()
if __name__=="__main__":
    main()