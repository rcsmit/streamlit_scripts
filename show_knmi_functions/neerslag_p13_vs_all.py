"""Streamlit-app: gemiddelde neerslag van de P13-stations vs. alle KNMI-stations.

Leest de KNMI-exports neerslag1.csv t/m neerslag6.csv, verwijdert duplicaten en
vergelijkt het P13-gemiddelde met het gemiddelde over alle beschikbare stations:
per dag en als cumulatieve som per jaar over een instelbaar seizoen (standaard
1 april t/m 30 september, zoals het KNMI voor het neerslagtekort gebruikt).
"""

# version : 20260728-131500 - Initiele Streamlit-versie: inlezen, dedupliceren, dagelijkse gemiddeldes, scatterplot met R2, tabblad met stationsinfo
# version : 20260728-143000 - Tabblad Weerstations toegevoegd: pydeck-kaart met kleurcodering P13
# version : 20260728-161500 - P13-koppeling gecorrigeerd naar de officiele KNMI-mapping (Vlissingen, Eindhoven, Ell); alle 13 stations doen mee; overeenstemmingsmaten (bias, MAE, RMSE, NSE) en tabblad met seizoenscumulatieven toegevoegd
# version : 20260728-174500 - Originele P13-neerslagstations met coordinaten op de kaart, verbindingslijnen naar het vervangende weerstation en afstandsberekening
current_version = "20260728-174500"

import datetime
import glob
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
import io
import re
import urllib.error
import urllib.request


# =============================================================================
# CONFIGURATIE
# =============================================================================
DEFAULT_INPUT_DIR = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\knmi"
DEFAULT_INPUT_DIR = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/knmi"
DEFAULT_FILE_PATTERN = "neerslag*.csv"

# Bij een URL bestaat er geen directory listing, dus worden neerslag1.csv t/m
# neerslag<REMOTE_MAX_FILES>.csv geprobeerd; ontbrekende bestanden (404) worden
# overgeslagen.
REMOTE_MAX_FILES = 12
REMOTE_TIMEOUT = 30

# Kolomnamen zoals ze in de KNMI-export staan (na het strippen van '# ')
COL_STN = "STN"
COL_DATE = "YYYYMMDD"
COL_RH = "RH"

# RH staat in 0.1 mm; -1 betekent "< 0.05 mm" (spoor van neerslag)
RH_SCALE = 0.1
RH_TRACE_VALUE = -1
RH_TRACE_REPLACEMENT = 0.0

# Standaardseizoen voor de cumulatieven: KNMI rekent het neerslagtekort over
# 1 april t/m 30 september (groeiseizoen).
DEFAULT_SEASON_START = (4, 1)
DEFAULT_SEASON_END = (9, 30)
# Minimaal percentage van de seizoensdagen dat aanwezig moet zijn om een jaar
# als "volledig" te bestempelen.
COMPLETENESS_THRESHOLD = 0.98

MARKER_COLOR = "#1f77b4"
YEAR_MARKER_COLOR = "#6a3d9a"
REGRESSION_COLOR = "#d62728"
ONE_TO_ONE_COLOR = "#7f7f7f"
PLOT_HEIGHT = 700

# -----------------------------------------------------------------------------
# De 13 officiele P13-neerslagstations (handmatige regenmeters, nummers op _N),
# met de automatische weerstations die het KNMI er zelf aan koppelt. Bron:
# KNMI, achtergrondinformatie klimaatdashboard / neerslagtekort.
# Let op: de P13 zijn neerslagstations, geen automatische weerstations. Kerkwerve
# en Oudenbosch meten wel degelijk, maar leveren geen uurdata; daarom staan hier
# de door het KNMI gebruikte vervangende weerstations.
# -----------------------------------------------------------------------------
DATA_P13 = {
    "stationsnr": [260, 235, 280, 278, 240, 249, 310, 370, 377, 286, 251, 319, 283],
    "P13-neerslagstation": [
        "De Bilt", "De Kooy", "Groningen", "Heerde", "Hoofddorp", "Hoorn",
        "Kerkwerve", "Oudenbosch", "Roermond", "Ter Apel", "West-Terschelling",
        "Westdorpe", "Winterswijk",
    ],
    "neerslagstation": [
        "550_N", "25_N", "139_N", "328_N", "438_N", "222_N", "737_N", "828_N",
        "961_N", "144_N", "11_N", "770_N", "666_N",
    ],
    "gebruikt weerstation": [
        "De Bilt", "De Kooy", "Eelde", "Heino", "Schiphol", "Berkhout",
        "Vlissingen", "Eindhoven", "Ell", "Nieuw Beerta", "Hoorn Terschelling",
        "Westdorpe", "Hupsel",
    ],
}

# --- Kaart met weerstations --------------------------------------------------
WEERSTATIONS_URL = (
    "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/"
    "show_knmi_functions/img_knmi/weerstations.csv"
)
GOOGLE_MAPS_URL = (
    "https://www.google.com/maps/d/u/0/edit?mid=1ePEzqJ4_aNyyTwF5FyUM6XiqhLZPSBjN"
    "&ll=52.17534745851063%2C5.197922250000001&z=7"
)
MAP_STYLE = "light"  # Carto-stijl, geen Mapbox-token nodig
MAP_ZOOM = 6.5
MAP_HEIGHT = 700
MARKER_RADIUS = 4000
LABEL_SIZE_SCALE = 0.3

CAT_OVERIG = "Overig weerstation"
CAT_P13_WEERSTATION = "Vervangend weerstation"
CAT_P13_ORIGINEEL = "Origineel P13-neerslagstation"

CATEGORY_COLORS = {
    CAT_OVERIG: [150, 150, 150, 140],
    CAT_P13_WEERSTATION: [0, 140, 70, 200],
    CAT_P13_ORIGINEEL: [230, 120, 20, 220],
}
LINE_COLOR = [230, 120, 20, 160]
LINE_WIDTH = 0.1

# Locaties van de P13-neerslagstations die door een ander station vervangen zijn.
# De Bilt, De Kooy en Westdorpe staan hier niet bij: daar valt het neerslagstation
# samen met het gebruikte weerstation. Coordinaten zijn de plaatscoordinaten en
# dus bij benadering; de exacte regenmeterpositie kan enkele honderden meters
# afwijken. Dat is verwaarloosbaar ten opzichte van de verplaatsingsafstanden.
P13_NEERSLAGSTATIONS = [
    {"original_Name": "Groningen (139_N)", "lat": 53.2194, "lon": 6.5665, "vervangen_door": 280},
    {"original_Name": "Heerde (328_N)", "lat": 52.3909, "lon": 6.0496, "vervangen_door": 278},
    {"original_Name": "Hoofddorp (438_N)", "lat": 52.3061, "lon": 4.6907, "vervangen_door": 240},
    {"original_Name": "Hoorn (222_N)", "lat": 52.6424, "lon": 5.0602, "vervangen_door": 249},
    {"original_Name": "Kerkwerve (737_N)", "lat": 51.6856, "lon": 3.8995, "vervangen_door": 310},
    {"original_Name": "Oudenbosch (828_N)", "lat": 51.5833, "lon": 4.5276, "vervangen_door": 370},
    {"original_Name": "Roermond (961_N)", "lat": 51.1913, "lon": 5.9878, "vervangen_door": 377},
    {"original_Name": "Ter Apel (144_N)", "lat": 52.8772, "lon": 7.0592, "vervangen_door": 286},
    {"original_Name": "West-Terschelling (11_N)", "lat": 53.3627, "lon": 5.2169, "vervangen_door": 251},
    {"original_Name": "Winterswijk (666_N)", "lat": 51.9713, "lon": 6.7205, "vervangen_door": 283},
]

ACHTERGROND_TEKST = """
### Stations

Het KNMI berekent het neerslagtekort uit de gemiddelde neerslag van 13
referentiestations (de P13) en de referentieverdamping, berekend uit de
zonneschijnduur in De Bilt (tot 2001) of de globale straling bij de P13-stations
(vanaf 2001).

De P13-stations zijn: De Bilt, De Kooy, Groningen, Heerde, Hoofddorp, Hoorn,
Kerkwerve, Oudenbosch, Roermond, Ter Apel, West-Terschelling, Westdorpe en
Winterswijk.

**Belangrijk:** de P13 zijn *neerslagstations* — handmatige regenmeters met een
nummer op `_N` — en geen automatische weerstations. Alle dertien meten dus gewoon,
ook Kerkwerve en Oudenbosch. Ze leveren alleen geen uurdata via de API die dit
script gebruikt. Het KNMI koppelt daarom zelf aan elk P13-station een automatisch
weerstation in de buurt; die koppeling wordt hier aangehouden, inclusief Vlissingen
voor Kerkwerve, Eindhoven voor Oudenbosch en Ell voor Roermond.

Er is dus geen enkel P13-station dat buiten de berekening valt: alle dertien doen
mee via het gekoppelde weerstation.

De reden dat het er precies dertien zijn is historisch: de reeks moet vergelijkbaar
blijven met de metingen vanaf 1906. Dertien is niet het optimale aantal voor een zo
nauwkeurig mogelijk landelijk gemiddelde, maar het aantal waarvoor een homogene
reeks van meer dan een eeuw bestaat.

De koppeling die dit script gebruikt:
"""

CUMULATIEF_TOELICHTING = """
Een dagwaarde uit dertien puntmetingen heeft een flinke ruimtelijke
bemonsteringsfout, zeker bij lokale zomerbuien. Die fout is echter grotendeels
willekeurig en middelt uit zodra je optelt: over een seizoen groeit de som met het
aantal dagen, terwijl de fout ongeveer met de wortel daarvan groeit. Een
seizoenstotaal — en dus ook een neerslagtekort — is daarom een aanzienlijk
robuustere grootheid dan een dagtotaal. Deze grafiek laat zien hoeveel de
overeenstemming verbetert bij aggregatie.
"""

# =============================================================================


def _read_single_file(path: str) -> pd.DataFrame:
    """Lees een KNMI-neerslagbestand in.

    De export begint met commentaarregels ('# ...'); de laatste commentaarregel met
    komma's bevat de kolomnamen en wordt als header gebruikt.
    """
    header_names = None
    skiprows = 0

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for i, line in enumerate(handle):
            if not line.lstrip().startswith("#"):
                skiprows = i
                break
            stripped = line.lstrip("# ").strip()
            if stripped and "," in stripped:
                header_names = [part.strip() for part in stripped.split(",")]

    if header_names is None:
        header_names = [COL_STN, COL_DATE, COL_RH]

    return pd.read_csv(
        path,
        skiprows=skiprows,
        header=None,
        names=header_names,
        skipinitialspace=True,
        na_values=["", " "],
    )



def _is_url(path: str) -> bool:
    """True als het pad een http(s)-URL is in plaats van een lokaal pad."""
    return str(path).lower().startswith(("http://", "https://"))


def _read_text(path: str) -> str:
    """Lees de inhoud van een lokaal bestand of een URL als tekst."""
    if _is_url(path):
        with urllib.request.urlopen(path, timeout=REMOTE_TIMEOUT) as response:
            return response.read().decode("utf-8", errors="replace")
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def _list_paths(input_dir: str, pattern: str) -> list[str]:
    """Stel de lijst met te lezen bestanden samen.

    Lokaal gebeurt dat met glob. Bij een URL kan een map niet worden uitgelezen,
    dus worden de namen afgeleid uit het patroon: `neerslag*.csv` wordt
    `neerslag1.csv` t/m `neerslag<REMOTE_MAX_FILES>.csv`. Een URL die al naar een
    los csv-bestand wijst wordt als sjabloon gebruikt.
    """
    if not _is_url(input_dir):
        return sorted(glob.glob(os.path.join(input_dir, pattern)))

    if input_dir.lower().endswith(".csv"):
        base, naam = input_dir.rsplit("/", 1)
    else:
        base, naam = input_dir.rstrip("/"), pattern

    # 'neerslag*.csv' en 'neerslag1.csv' worden allebei 'neerslag{}.csv'
    sjabloon = re.sub(r"(\*|\d+)(?=\.csv$)", "{}", naam, flags=re.IGNORECASE)
    if "{}" not in sjabloon:
        return [f"{base}/{naam}"]
    return [f"{base}/{sjabloon.format(i)}" for i in range(1, REMOTE_MAX_FILES + 1)]


def _read_single_file(path: str) -> pd.DataFrame:
    """Lees een KNMI-neerslagbestand in vanaf schijf of URL."""
    tekst = _read_text(path)
    regels = tekst.splitlines()

    header_names = None
    skiprows = 0
    for i, line in enumerate(regels):
        if not line.lstrip().startswith("#"):
            skiprows = i
            break
        stripped = line.lstrip("# ").strip()
        if stripped and "," in stripped:
            header_names = [part.strip() for part in stripped.split(",")]

    if header_names is None:
        header_names = [COL_STN, COL_DATE, COL_RH]

    return pd.read_csv(
        io.StringIO(tekst),
        skiprows=skiprows,
        header=None,
        names=header_names,
        skipinitialspace=True,
        na_values=["", " "],
    )


@st.cache_data(ttl="1h", max_entries=8, show_spinner=False)
def load_data(input_dir: str, pattern: str) -> tuple[pd.DataFrame, dict]:
    """Lees alle neerslagbestanden, plak ze aan elkaar en verwijder duplicaten."""
    paths = _list_paths(input_dir, pattern)
    if not paths:
        raise FileNotFoundError(
            f"Geen bestanden gevonden met patroon '{pattern}' in '{input_dir}'"
        )

    frames, gelezen, overgeslagen = [], [], []
    for path in paths:
        try:
            frames.append(_read_single_file(path))
            gelezen.append(path)
        except (urllib.error.HTTPError, urllib.error.URLError, FileNotFoundError, OSError):
            # Bij een URL bestaan de hogere nummers vaak niet; die slaan we over.
            overgeslagen.append(path)

    if not frames:
        raise FileNotFoundError(
            f"Geen leesbare bestanden gevonden voor patroon '{pattern}' in '{input_dir}'"
        )

    df = pd.concat(frames, ignore_index=True)

    n_before = len(df)
    df = df.drop_duplicates(subset=[COL_STN, COL_DATE], keep="first").reset_index(drop=True)
    n_after = len(df)

    df[COL_STN] = pd.to_numeric(df[COL_STN], errors="coerce").astype("Int64")
    df[COL_RH] = pd.to_numeric(df[COL_RH], errors="coerce")
    df["datum"] = pd.to_datetime(df[COL_DATE].astype(str), format="%Y%m%d", errors="coerce")

    df["neerslag_mm"] = df[COL_RH].replace(RH_TRACE_VALUE, RH_TRACE_REPLACEMENT) * RH_SCALE
    df = df.dropna(subset=["datum", COL_STN]).reset_index(drop=True)

    info = {
        "bestanden": [p.rsplit("/", 1)[-1] if _is_url(p) else os.path.basename(p) for p in gelezen],
        "overgeslagen": len(overgeslagen),
        "bron": "URL" if _is_url(input_dir) else "schijf",
        "rijen_voor": n_before,
        "duplicaten": n_before - n_after,
        "rijen_na": n_after,
        "stations": int(df[COL_STN].nunique()),
    }
    return df, info

def p13_station_numbers() -> list[int]:
    """De stationsnummers van de weerstations die aan de P13 gekoppeld zijn."""
    return [int(stn) for stn in DATA_P13["stationsnr"]]


def build_daily_means(
    df: pd.DataFrame,
    require_complete_p13: bool,
    min_stations_all: int,
    exclude_p13_from_all: bool,
) -> pd.DataFrame:
    """Bouw per dag het P13-gemiddelde en het gemiddelde van alle stations."""
    p13_stations = p13_station_numbers()
    df_valid = df.dropna(subset=["neerslag_mm"])

    df_all = df_valid
    if exclude_p13_from_all:
        df_all = df_valid[~df_valid[COL_STN].isin(p13_stations)]

    all_mean = (
        df_all.groupby("datum")["neerslag_mm"]
        .agg(neerslag_all="mean", n_stations_all="count")
        .reset_index()
    )

    df_p13 = df_valid[df_valid[COL_STN].isin(p13_stations)]
    p13_mean = (
        df_p13.groupby("datum")["neerslag_mm"]
        .agg(neerslag_P13="mean", n_stations_P13="count")
        .reset_index()
    )

    result = pd.merge(p13_mean, all_mean, on="datum", how="inner")

    if require_complete_p13:
        result = result[result["n_stations_P13"] == len(p13_stations)]
    result = result[result["n_stations_all"] >= min_stations_all]

    return result.sort_values("datum").reset_index(drop=True)


def _season_mask(dates: pd.Series, start_md: tuple[int, int], end_md: tuple[int, int]) -> pd.Series:
    """True voor datums binnen het seizoen (dag/maand-venster), jaaroverschrijdend toegestaan."""
    md = list(zip(dates.dt.month, dates.dt.day))
    start, end = start_md, end_md
    if start <= end:
        return pd.Series([start <= item <= end for item in md], index=dates.index)
    return pd.Series([item >= start or item <= end for item in md], index=dates.index)


def _season_year(dates: pd.Series, start_md: tuple[int, int], end_md: tuple[int, int]) -> pd.Series:
    """Het seizoensjaar; bij een jaaroverschrijdend venster telt het startjaar."""
    if start_md <= end_md:
        return dates.dt.year
    md = list(zip(dates.dt.month, dates.dt.day))
    years = [
        year if item >= start_md else year - 1
        for item, year in zip(md, dates.dt.year)
    ]
    return pd.Series(years, index=dates.index)


def _season_length(start_md: tuple[int, int], end_md: tuple[int, int]) -> int:
    """Aantal dagen in het seizoensvenster, gerekend over een schrikkeljaar."""
    ref = 2000
    start = datetime.date(ref, *start_md)
    end = datetime.date(ref, *end_md)
    if start <= end:
        return (end - start).days + 1
    return (datetime.date(ref, 12, 31) - start).days + 1 + (end - datetime.date(ref, 1, 1)).days + 1


def build_season_totals(
    daily: pd.DataFrame,
    start_md: tuple[int, int],
    end_md: tuple[int, int],
    only_complete_years: bool,
) -> pd.DataFrame:
    """Tel per seizoensjaar de neerslag op voor P13 en voor alle stations."""
    if daily.empty:
        return daily.head(0).assign(jaar=[], dagen=[], volledig=[])

    subset = daily[_season_mask(daily["datum"], start_md, end_md)].copy()
    if subset.empty:
        return subset.assign(jaar=[], dagen=[], volledig=[])

    subset["jaar"] = _season_year(subset["datum"], start_md, end_md)

    totals = (
        subset.groupby("jaar")
        .agg(
            cum_P13=("neerslag_P13", "sum"),
            cum_all=("neerslag_all", "sum"),
            dagen=("datum", "count"),
            eerste_dag=("datum", "min"),
            laatste_dag=("datum", "max"),
        )
        .reset_index()
    )

    verwacht = _season_length(start_md, end_md)
    totals["dekking"] = totals["dagen"] / verwacht
    totals["volledig"] = totals["dekking"] >= COMPLETENESS_THRESHOLD

    if only_complete_years:
        totals = totals[totals["volledig"]]

    return totals.sort_values("jaar").reset_index(drop=True)


def calculate_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Bereken helling, intercept en R^2 van de lineaire regressie y = a*x + b."""
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def agreement_metrics(x: np.ndarray, y: np.ndarray) -> dict:
    """Maten voor overeenstemming tussen twee schattingen van dezelfde grootheid.

    x = P13-schatting, y = referentie (alle stations). Naast R^2 en Pearson r
    worden bias, MAE, RMSE en de Nash-Sutcliffe-efficiency (R^2 ten opzichte van
    de 1:1-lijn) berekend; die laatste drie zeggen iets over overeenstemming in
    plaats van alleen over samenhang.
    """
    slope, intercept, r2 = calculate_r2(x, y)
    residu = x - y
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    nse = 1.0 - float(np.sum(residu ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_bias = float(np.mean(residu) / np.mean(y) * 100) if np.mean(y) != 0 else float("nan")
    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "r": float(np.corrcoef(x, y)[0, 1]),
        "bias": float(np.mean(residu)),
        "rel_bias": rel_bias,
        "mae": float(np.mean(np.abs(residu))),
        "rmse": float(np.sqrt(np.mean(residu ** 2))),
        "nse": nse,
    }


def _scatter_figure(
    x: np.ndarray,
    y: np.ndarray,
    labels: pd.Series,
    metrics: dict,
    x_title: str,
    y_title: str,
    marker_color: str,
    marker_size: int,
    mode: str = "markers",
) -> go.Figure:
    """Generieke scatterplot met regressielijn en 1:1-lijn."""
    lo = float(min(np.nanmin(x), np.nanmin(y)))
    hi = float(max(np.nanmax(x), np.nanmax(y)))
    x_line = np.linspace(lo, hi, 100)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode=mode, name="waarnemingen",
            marker=dict(size=marker_size, color=marker_color, opacity=0.7),
            text=labels,
            textposition="top center",
            hovertemplate="%{text}<br>P13: %{x:.2f} mm<br>Alle: %{y:.2f} mm<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line, y=metrics["slope"] * x_line + metrics["intercept"], mode="lines",
            name=f"regressie (R² = {metrics['r2']:.4f})",
            line=dict(color=REGRESSION_COLOR, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line, y=x_line, mode="lines", name="1:1-lijn",
            line=dict(color=ONE_TO_ONE_COLOR, width=1, dash="dash"),
        )
    )
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        height=PLOT_HEIGHT,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=60, r=30, t=30, b=60),
    )
    return fig


def render_metric_row(metrics: dict, eenheid: str = "mm") -> None:
    """Toon de kern- en overeenstemmingsmaten."""
    with st.container(horizontal=True):
        st.metric("R²", f"{metrics['r2']:.4f}", border=True)
        st.metric("Pearson r", f"{metrics['r']:.4f}", border=True)
        st.metric("NSE (t.o.v. 1:1)", f"{metrics['nse']:.4f}", border=True)
    with st.container(horizontal=True):
        st.metric("Bias", f"{metrics['bias']:+.2f} {eenheid}", border=True)
        st.metric("Relatieve bias", f"{metrics['rel_bias']:+.2f} %", border=True)
        st.metric("MAE", f"{metrics['mae']:.2f} {eenheid}", border=True)
        st.metric("RMSE", f"{metrics['rmse']:.2f} {eenheid}", border=True)


def stations_dataframe() -> pd.DataFrame:
    """De P13-koppeltabel."""
    stations = pd.DataFrame(DATA_P13)
    stations["stationsnr"] = stations["stationsnr"].astype("Int64")
    return stations


def get_weerstations() -> list[list]:
    """Stationsnummers en -namen van de automatische KNMI-weerstations."""
    weer_stations = [
        [209, "IJmond"],
        [210, "Valkenburg Zh"],
        [215, "Voorschoten"],
        [225, "IJmuiden"],
        [235, "De Kooy"],
        [240, "Schiphol"],
        [242, "Vlieland"],
        [248, "Wijdenes"],
        [249, "Berkhout"],
        [251, "Hoorn Terschelling"],
        [257, "Wijk aan Zee"],
        [258, "Houtribdijk"],
        [260, "De Bilt"],
        [265, "Soesterberg"],
        [267, "Stavoren"],
        [269, "Lelystad"],
        [270, "Leeuwarden"],
        [273, "Marknesse"],
        [275, "Deelen"],
        [277, "Lauwersoog"],
        [278, "Heino"],
        [279, "Hoogeveen"],
        [280, "Eelde"],
        [283, "Hupsel"],
        [285, "Huibertgat"],
        [286, "Nieuw Beerta"],
        [290, "Twenthe"],
        [308, "Cadzand"],
        [310, "Vlissingen"],
        [311, "Hoofdplaat"],
        [312, "Oosterschelde"],
        [313, "Vlakte van De Raan"],
        [315, "Hansweert"],
        [316, "Schaar"],
        [319, "Westdorpe"],
        [323, "Wilhelminadorp"],
        [324, "Stavenisse"],
        [330, "Hoek van Holland"],
        [331, "Tholen"],
        [340, "Woensdrecht"],
        [343, "Rotterdam Geulhaven"],
        [344, "Rotterdam"],
        [348, "Cabauw Mast"],
        [350, "Gilze-Rijen"],
        [356, "Herwijnen"],
        [370, "Eindhoven"],
        [375, "Volkel"],
        [377, "Ell"],
        [380, "Maastricht"],
        [391, "Arcen"],
    ]
    return weer_stations


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Afstand over het aardoppervlak tussen twee punten, in kilometers."""
    radius = 6371.0088
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return float(2 * radius * np.arcsin(np.sqrt(a)))


@st.cache_data(ttl="24h", max_entries=4, show_spinner=False)
def load_weerstations(url: str = WEERSTATIONS_URL) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Haal de stationslocaties op, categoriseer ze en bereken de verplaatsingen.

    Levert twee dataframes: alle punten voor de kaart, en de verbindingen tussen
    de originele P13-neerslagstations en de weerstations die ze vervangen.
    """
    df_map = pd.read_csv(url, comment="#", delimiter=",", low_memory=False)
    df_map = df_map[["original_Name", "station_nr", "lat", "lon"]].copy()
    df_map["station_nr"] = pd.to_numeric(df_map["station_nr"], errors="coerce").astype("Int64")

    # Namen uit get_weerstations() aanhouden waar het stationsnummer bekend is
    namen = {int(nr): naam for nr, naam in get_weerstations()}
    df_map["original_Name"] = [
        namen.get(int(nr), naam) if pd.notna(nr) else naam
        for nr, naam in zip(df_map["station_nr"], df_map["original_Name"])
    ]

    koppeling = {
        int(nr): naam
        for nr, naam in zip(DATA_P13["stationsnr"], DATA_P13["P13-neerslagstation"])
    }

    # Verbindingen origineel -> vervanger opbouwen, met afstand
    coords = {
        int(nr): (float(lat), float(lon))
        for nr, lat, lon in zip(df_map["station_nr"], df_map["lat"], df_map["lon"])
        if pd.notna(nr)
    }
    verbindingen = []
    for punt in P13_NEERSLAGSTATIONS:
        doel = coords.get(int(punt["vervangen_door"]))
        if doel is None:
            continue
        verbindingen.append(
            {
                "P13-neerslagstation": punt["original_Name"],
                "vervangen door": namen.get(int(punt["vervangen_door"]), ""),
                "weerstation nr": int(punt["vervangen_door"]),
                "afstand_km": haversine_km(punt["lat"], punt["lon"], doel[0], doel[1]),
                "van": [punt["lon"], punt["lat"]],
                "naar": [doel[1], doel[0]],
            }
        )
    df_lines = pd.DataFrame(verbindingen)

    # Originele P13-neerslagstations als eigen punten toevoegen
    p13_punten = pd.DataFrame(
        [
            {
                "original_Name": punt["original_Name"],
                "station_nr": pd.NA,
                "lat": punt["lat"],
                "lon": punt["lon"],
            }
            for punt in P13_NEERSLAGSTATIONS
        ]
    )
    df_map = pd.concat([df_map, p13_punten], ignore_index=True)
    df_map["station_nr"] = df_map["station_nr"].astype("Int64")

    origineel_namen = {punt["original_Name"] for punt in P13_NEERSLAGSTATIONS}

    def _categorie(row: pd.Series) -> str:
        if row["original_Name"] in origineel_namen:
            return CAT_P13_ORIGINEEL
        if pd.notna(row["station_nr"]) and int(row["station_nr"]) in koppeling:
            return CAT_P13_WEERSTATION
        return CAT_OVERIG

    df_map["categorie"] = df_map.apply(_categorie, axis=1)
    df_map["staat_voor"] = [
        koppeling.get(int(nr), "") if pd.notna(nr) else ""
        for nr in df_map["station_nr"]
    ]

    afstanden = dict(zip(df_lines["P13-neerslagstation"], df_lines["afstand_km"])) if not df_lines.empty else {}
    df_map["afstand_km"] = [
        afstanden.get(naam, np.nan) for naam in df_map["original_Name"]
    ]
    df_map["color"] = df_map["categorie"].map(CATEGORY_COLORS)

    # P13-punten bovenop de rest tekenen
    volgorde = {CAT_OVERIG: 0, CAT_P13_WEERSTATION: 1, CAT_P13_ORIGINEEL: 2}
    df_map = (
        df_map.assign(_z=df_map["categorie"].map(volgorde))
        .sort_values("_z")
        .drop(columns="_z")
        .reset_index(drop=True)
    )
    return df_map, df_lines


def make_stations_deck(
    df_map: pd.DataFrame, df_lines: pd.DataFrame, show_labels: bool, show_lines: bool
) -> pdk.Deck:
    """Bouw de pydeck-kaart met gekleurde stations en verplaatsingslijnen."""
    midpoint = (float(np.average(df_map["lat"])), float(np.average(df_map["lon"])))

    layers = []
    if show_lines and not df_lines.empty:
        layers.append(
            pdk.Layer(
                "LineLayer",
                df_lines,
                get_source_position="van",
                get_target_position="naar",
                get_color=LINE_COLOR,
                get_width=LINE_WIDTH,
                width_units="meters",
                pickable=False,
            )
        )

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            df_map,
            get_position=["lon", "lat"],
            auto_highlight=True,
            get_radius=MARKER_RADIUS,
            get_fill_color="color",
            pickable=True,
        )
    )

    if show_labels:
        layers.append(
            pdk.Layer(
                "TextLayer",
                data=df_map,
                pickable=False,
                get_position=["lon", "lat"],
                get_text="original_Name",
                get_color=[0, 0, 0],
                get_angle=0,
                sizeScale=LABEL_SIZE_SCALE,
                getTextAnchor='"middle"',
                get_alignment_baseline='"bottom"',
            )
        )

    tooltip = {
        "html": "<b>{original_Name}</b><br/>station {station_nr}<br/>"
                "{categorie}<br/>staat voor: {staat_voor}"
    }

    return pdk.Deck(
        map_style=MAP_STYLE,
        initial_view_state=pdk.ViewState(
            longitude=midpoint[1],
            latitude=midpoint[0],
            pitch=0,
            zoom=MAP_ZOOM,
        ),
        layers=layers,
        tooltip=tooltip,
    )


def render_sidebar() -> dict:
    """Invoerinstellingen in de sidebar."""
    with st.sidebar:
        st.header("Instellingen")
        input_dir = st.text_input("Map met de csv-bestanden", value=DEFAULT_INPUT_DIR)
        pattern = st.text_input("Bestandspatroon", value=DEFAULT_FILE_PATTERN)
        require_complete_p13 = st.checkbox(
            "Alleen dagen met alle 13 P13-stations", value=True,
            help="Sluit dagen uit waarop niet alle gekoppelde weerstations een meting hebben.",
        )
        exclude_p13_from_all = st.checkbox(
            "P13-stations uitsluiten uit 'alle stations'", value=False,
            help="Standaard zitten de P13-stations ook in het landelijke gemiddelde.",
        )
        min_stations_all = st.number_input(
            "Minimum aantal stations voor landelijk gemiddelde",
            min_value=1, max_value=100, value=1, step=1,
        )

        st.subheader("Seizoen voor de cumulatieven")
        st.caption("Het jaartal is niet van belang, alleen dag en maand.")
        start_date = st.date_input(
            "Startdatum", value=datetime.date(2000, *DEFAULT_SEASON_START), format="DD-MM-YYYY"
        )
        end_date = st.date_input(
            "Einddatum", value=datetime.date(2000, *DEFAULT_SEASON_END), format="DD-MM-YYYY"
        )
        only_complete_years = st.checkbox(
            "Alleen volledige seizoenen", value=True,
            help=f"Jaren met minder dan {COMPLETENESS_THRESHOLD:.0%} van de seizoensdagen weglaten.",
        )

    return {
        "input_dir": input_dir,
        "pattern": pattern,
        "require_complete_p13": require_complete_p13,
        "exclude_p13_from_all": exclude_p13_from_all,
        "min_stations_all": int(min_stations_all),
        "start_md": (start_date.month, start_date.day),
        "end_md": (end_date.month, end_date.day),
        "only_complete_years": only_complete_years,
    }


def render_analyse_tab(daily: pd.DataFrame) -> None:
    """Tabblad met kerncijfers en de dagelijkse scatterplot."""
    if daily.empty:
        st.warning("Geen overlappende dagen gevonden met de huidige instellingen.")
        return

    x = daily["neerslag_P13"].to_numpy(dtype=float)
    y = daily["neerslag_all"].to_numpy(dtype=float)
    metrics = agreement_metrics(x, y)

    render_metric_row(metrics)

    st.caption(
        f"{len(daily)} dagen, {daily['datum'].min():%d-%m-%Y} t/m {daily['datum'].max():%d-%m-%Y} · "
        f"regressie: neerslag_all = {metrics['slope']:.4f} × neerslag_P13 + {metrics['intercept']:.4f}"
    )

    fig = _scatter_figure(
        x, y, daily["datum"].dt.strftime("%d-%m-%Y"), metrics,
        "Gemiddelde dagneerslag P13 (mm)",
        "Gemiddelde dagneerslag alle stations (mm)",
        MARKER_COLOR, 5,
    )
    st.plotly_chart(fig, width="stretch")


def render_cumulatief_tab(daily: pd.DataFrame, settings: dict) -> None:
    """Tabblad met de seizoenscumulatieven per jaar."""
    start_md, end_md = settings["start_md"], settings["end_md"]
    periode = f"{start_md[1]:02d}-{start_md[0]:02d} t/m {end_md[1]:02d}-{end_md[0]:02d}"
    st.caption(f"Seizoen {periode} · {_season_length(start_md, end_md)} dagen · in te stellen in de sidebar")

    totals = build_season_totals(daily, start_md, end_md, settings["only_complete_years"])

    if totals.empty:
        st.warning("Geen seizoenen gevonden met de huidige instellingen.")
        return
    if len(totals) < 3:
        st.info(
            f"Slechts {len(totals)} volledig seizoen(en) beschikbaar; de maten hieronder "
            "zijn daarmee weinig informatief.",
            icon=":material/info:",
        )

    x = totals["cum_P13"].to_numpy(dtype=float)
    y = totals["cum_all"].to_numpy(dtype=float)

    if len(totals) >= 2:
        metrics = agreement_metrics(x, y)
        render_metric_row(metrics)
        fig = _scatter_figure(
            x, y, totals["jaar"].astype(str), metrics,
            f"Cumulatieve neerslag P13, {periode} (mm)",
            f"Cumulatieve neerslag alle stations, {periode} (mm)",
            YEAR_MARKER_COLOR, 12, mode="markers+text",
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown(CUMULATIEF_TOELICHTING)

    st.dataframe(
        totals, hide_index=True, width="stretch",
        column_config={
            "jaar": st.column_config.NumberColumn("Jaar", format="%d"),
            "cum_P13": st.column_config.NumberColumn("Cumulatief P13 (mm)", format="%.1f"),
            "cum_all": st.column_config.NumberColumn("Cumulatief alle (mm)", format="%.1f"),
            "dagen": st.column_config.NumberColumn("Dagen"),
            "eerste_dag": st.column_config.DateColumn("Eerste dag", format="DD-MM-YYYY"),
            "laatste_dag": st.column_config.DateColumn("Laatste dag", format="DD-MM-YYYY"),
            "dekking": st.column_config.ProgressColumn(
                "Dekking", min_value=0.0, max_value=1.0, format="%.0f%%"
            ),
            "volledig": st.column_config.CheckboxColumn("Volledig"),
        },
    )

    st.download_button(
        ":material/download: Download cumulatieven als csv",
        data=totals.to_csv(index=False).encode("utf-8"),
        file_name="neerslag_cumulatief_per_jaar.csv",
        mime="text/csv",
    )


def render_data_tab(daily: pd.DataFrame, info: dict) -> None:
    """Tabblad met het dagelijkse dataframe en de inleesstatistieken."""
    with st.container(horizontal=True):
        st.metric("Bestanden", len(info["bestanden"]), border=True)
        st.metric("Rijen ingelezen", f"{info['rijen_voor']:,}".replace(",", "."), border=True)
        st.metric("Duplicaten verwijderd", f"{info['duplicaten']:,}".replace(",", "."), border=True)
        st.metric("Unieke stations", info["stations"], border=True)

    st.caption("Ingelezen bestanden: " + ", ".join(info["bestanden"]))

    st.subheader("Dagelijkse gemiddeldes")
    st.dataframe(
        daily, hide_index=True, width="stretch",
        column_config={
            "datum": st.column_config.DateColumn("Datum", format="DD-MM-YYYY"),
            "neerslag_P13": st.column_config.NumberColumn("Neerslag P13 (mm)", format="%.2f"),
            "n_stations_P13": st.column_config.NumberColumn("Stations P13"),
            "neerslag_all": st.column_config.NumberColumn("Neerslag alle (mm)", format="%.2f"),
            "n_stations_all": st.column_config.NumberColumn("Stations alle"),
        },
    )

    st.download_button(
        ":material/download: Download dagwaarden als csv",
        data=daily.to_csv(index=False).encode("utf-8"),
        file_name="neerslag_p13_vs_all.csv",
        mime="text/csv",
    )


def render_stations_tab() -> None:
    """Tabblad met de achtergrondinformatie en de koppeltabel."""
    st.markdown(ACHTERGROND_TEKST)

    st.dataframe(
        stations_dataframe(), hide_index=True, width="stretch",
        column_config={
            "stationsnr": st.column_config.NumberColumn("Weerstation nr", format="%d"),
            "P13-neerslagstation": st.column_config.TextColumn("P13-neerslagstation", pinned=True),
            "neerslagstation": st.column_config.TextColumn("Nr neerslagstation"),
            "gebruikt weerstation": st.column_config.TextColumn("Gebruikt weerstation"),
        },
    )

    st.caption(
        "Alle 13 P13-stations doen mee via het door het KNMI gekoppelde weerstation. "
        "Kerkwerve en Oudenbosch meten wel, maar leveren geen uurdata."
    )


def render_weerstations_tab() -> None:
    """Tabblad met de kaart van alle KNMI-stations."""
    try:
        with st.spinner("Stationslocaties ophalen..."):
            df_map, df_lines = load_weerstations()
    except Exception as exc:  # netwerk- of parsefout
        st.error(f"Kon de stationslijst niet ophalen: {exc}", icon=":material/error:")
        return

    st.markdown(
        f":orange-badge[{CAT_P13_ORIGINEEL}] "
        f":green-badge[{CAT_P13_WEERSTATION}] "
        f":gray-badge[{CAT_OVERIG}]"
    )
    st.caption(
        "Oranje: de oorspronkelijke P13-neerslagstations die door een ander station "
        "vervangen zijn, op hun eigen locatie. Groen: het weerstation dat het KNMI "
        "ervoor in de plaats gebruikt. De lijn verbindt beide. De Bilt, De Kooy en "
        "Westdorpe ontbreken in het oranje: daar valt het neerslagstation samen met "
        "het weerstation."
    )

    with st.container(horizontal=True):
        show_labels = st.checkbox("Stationsnamen tonen", value=True)
        show_lines = st.checkbox("Verplaatsingen tonen", value=True)

    st.pydeck_chart(
        make_stations_deck(df_map, df_lines, show_labels, show_lines), height=MAP_HEIGHT
    )

    if not df_lines.empty:
        with st.container(horizontal=True):
            st.metric("Vervangen stations", len(df_lines), border=True)
            st.metric("Mediane afstand", f"{df_lines['afstand_km'].median():.1f} km", border=True)
            st.metric("Grootste afstand", f"{df_lines['afstand_km'].max():.1f} km", border=True)

        st.subheader("Afstand tussen origineel en vervanger")
        st.dataframe(
            df_lines.drop(columns=["van", "naar"]).sort_values("afstand_km", ascending=False),
            hide_index=True, width="stretch",
            column_config={
                "P13-neerslagstation": st.column_config.TextColumn(
                    "Origineel P13-station", pinned=True
                ),
                "vervangen door": st.column_config.TextColumn("Vervangen door"),
                "weerstation nr": st.column_config.NumberColumn("Weerstation nr", format="%d"),
                "afstand_km": st.column_config.NumberColumn("Afstand (km)", format="%.1f"),
            },
        )
        st.caption(
            "Coordinaten van de originele stations zijn plaatscoordinaten en dus bij "
            "benadering; de afwijking van enkele honderden meters valt weg tegen deze afstanden."
        )

    st.subheader("Alle punten op de kaart")
    st.dataframe(
        df_map.drop(columns=["color"]), hide_index=True, width="stretch",
        column_config={
            "original_Name": st.column_config.TextColumn("Station", pinned=True),
            "station_nr": st.column_config.NumberColumn("Stationsnr", format="%d"),
            "lat": st.column_config.NumberColumn("Breedtegraad", format="%.4f"),
            "lon": st.column_config.NumberColumn("Lengtegraad", format="%.4f"),
            "categorie": st.column_config.TextColumn("Categorie"),
            "staat_voor": st.column_config.TextColumn("Staat voor P13-station"),
            "afstand_km": st.column_config.NumberColumn("Afstand tot vervanger (km)", format="%.1f"),
        },
    )

    st.link_button(":material/open_in_new: KNMI-stations op Google Maps", GOOGLE_MAPS_URL)


def main() -> None:
    st.set_page_config(
        page_title="Neerslag P13 vs. alle stations",
        page_icon=":material/water_drop:",
        layout="wide",
    )
    st.title("Neerslag: P13-stations vs. alle KNMI-stations")

    settings = render_sidebar()

    try:
        with st.spinner("Bestanden inlezen..."):
            df, info = load_data(settings["input_dir"], settings["pattern"])
    except FileNotFoundError as exc:
        st.error(str(exc), icon=":material/error:")
        st.stop()

    daily = build_daily_means(
        df,
        require_complete_p13=settings["require_complete_p13"],
        min_stations_all=settings["min_stations_all"],
        exclude_p13_from_all=settings["exclude_p13_from_all"],
    )

    tab_dag, tab_cum, tab_data, tab_stations, tab_kaart = st.tabs(
        [
            ":material/scatter_plot: Dagwaarden",
            ":material/show_chart: Cumulatief per jaar",
            ":material/table: Data",
            ":material/pin_drop: Stations en achtergrond",
            ":material/map: Weerstations",
        ]
    )
    with tab_stations:
        render_stations_tab()
    with tab_kaart:
        render_weerstations_tab()

    with tab_dag:
        render_analyse_tab(daily)
    with tab_cum:
        render_cumulatief_tab(daily, settings)
    with tab_data:
        render_data_tab(daily, info)
    

if __name__ == "__main__":
    main()