"""Streamlit-app: gemiddelde etmaalneerslag van de P13-stations vs. alle KNMI-stations.

Leest de KNMI-exports neerslag1.csv t/m neerslag6.csv, verwijdert duplicaten,
berekent per dag het gemiddelde van de P13-stations en van alle stations en
toont een scatterplot met regressielijn en R^2.
"""

# version : 20260728-131500 - Initiele Streamlit-versie: inlezen, dedupliceren, dagelijkse gemiddeldes, scatterplot met R2, tabblad met stationsinfo
# version : 20260728-143000 - Tabblad Weerstations toegevoegd: pydeck-kaart met kleurcodering P13 (gebruikt / niet opgenomen / overig)
current_version = "20260728-143000"

import glob
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

# =============================================================================
# CONFIGURATIE
# =============================================================================
DEFAULT_INPUT_DIR = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\knmi"
DEFAULT_FILE_PATTERN = "neerslag*.csv"

# Kolomnamen zoals ze in de KNMI-export staan (na het strippen van '# ')
COL_STN = "STN"
COL_DATE = "YYYYMMDD"
COL_RH = "RH"

# RH staat in 0.1 mm; -1 betekent "< 0.05 mm" (spoor van neerslag)
RH_SCALE = 0.1
RH_TRACE_VALUE = -1
RH_TRACE_REPLACEMENT = 0.0

MARKER_COLOR = "#1f77b4"
REGRESSION_COLOR = "#d62728"
ONE_TO_ONE_COLOR = "#7f7f7f"
PLOT_HEIGHT = 700

# De 13 officiele P13-stations. `stationsnr` is None waar de KNMI-API geen
# bruikbaar alternatief levert; die stations doen niet mee in de berekening.
DATA_P13 = {
    "stationsnr": [260, 235, 280, 278, 240, 249, None, None, 391, 286, 251, 319, 283],
    "genoemd in tekst": [
        "De Bilt", "De Kooy", "Groningen", "Heerde", "Hoofddorp", "Hoorn",
        "Kerkwerve", "Oudenbosch", "Roermond", "Ter Apel", "West-Terschelling",
        "Westdorpe", "Winterswijk",
    ],
    "gebruikte data": [
        "De Bilt", "De Kooy", "Eelde", "Heino", "Schiphol", "Berkhout",
        "- NIET OPGENOMEN -", "- NIET OPGENOMEN -", "Arcen", "Nieuw Beerta",
        "Hoorn Terschelling", "Westdorpe", "Hupsel",
    ],
}

# --- Kaart met weerstations -------------------------------------------------
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
LABEL_SIZE_SCALE = 0.6

CAT_OVERIG = "Overig station"
CAT_P13_GEBRUIKT = "P13 - gebruikt"
CAT_P13_ONTBREEKT = "P13 - niet opgenomen"

CATEGORY_COLORS = {
    CAT_OVERIG: [150, 150, 150, 140],
    CAT_P13_GEBRUIKT: [0, 140, 70, 200],
    CAT_P13_ONTBREEKT: [220, 60, 40, 220],
}

# De twee P13-stations zonder KNMI-data; coordinaten handmatig toegevoegd
P13_ONTBREKENDE_STATIONS = [
    {"station_nr": pd.NA, "original_Name": "Kerkwerve", "lat": 51.686, "lon": 3.898},
    {"station_nr": pd.NA, "original_Name": "Oudenbosch", "lat": 51.5833, "lon": 4.5276},
]

ACHTERGROND_TEKST = """
### Stations

De KNMI-berekening gebruikt de gemiddelde neerslag van 13 referentiestations in
Nederland (de zogenoemde P13-stations) en de referentieverdamping, berekend op basis
van de zonneschijnduur in De Bilt (tot 2001) of de globale straling nabij de
P13-stations (vanaf 2001).

De P13-stations zijn: De Bilt, De Kooy, Groningen, Heerde, Hoofddorp, Hoorn,
Kerkwerve, Oudenbosch, Roermond, Ter Apel, West-Terschelling, Westdorpe en
Winterswijk.

De P13-stations zijn niet allemaal beschikbaar via de API van het KNMI, dus is het
dichtstbijzijnde alternatief gebruikt. Voor twee stations was er geen alternatief of
ontbraken de meetwaarden.

De globale straling van het gekozen station is meegenomen in de berekening. De
zonneschijnduur maakt echter geen deel uit van de aangeleverde formule. Daarnaast
lijkt de temperatuur, hoewel onderdeel van de gebruikte formule, door het KNMI niet
te worden gebruikt.

De stations die het script gebruikt:
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


@st.cache_data(ttl="1h", max_entries=8, show_spinner=False)
def load_data(input_dir: str, pattern: str) -> tuple[pd.DataFrame, dict]:
    """Lees alle neerslagbestanden, plak ze aan elkaar en verwijder duplicaten."""
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"Geen bestanden gevonden met patroon '{pattern}' in '{input_dir}'"
        )

    frames = [_read_single_file(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)

    n_before = len(df)
    df = df.drop_duplicates(subset=[COL_STN, COL_DATE], keep="first").reset_index(drop=True)
    n_after = len(df)

    df[COL_STN] = pd.to_numeric(df[COL_STN], errors="coerce").astype("Int64")
    df[COL_RH] = pd.to_numeric(df[COL_RH], errors="coerce")
    df["datum"] = pd.to_datetime(df[COL_DATE].astype(str), format="%Y%m%d", errors="coerce")

    # RH omzetten naar mm; -1 (spoor van neerslag) telt als 0 mm
    df["neerslag_mm"] = df[COL_RH].replace(RH_TRACE_VALUE, RH_TRACE_REPLACEMENT) * RH_SCALE
    df = df.dropna(subset=["datum", COL_STN]).reset_index(drop=True)

    info = {
        "bestanden": [os.path.basename(p) for p in paths],
        "rijen_voor": n_before,
        "duplicaten": n_before - n_after,
        "rijen_na": n_after,
        "stations": int(df[COL_STN].nunique()),
    }
    return df, info


def p13_station_numbers() -> list[int]:
    """De stationsnummers uit DATA_P13 die daadwerkelijk data hebben."""
    return [int(stn) for stn in DATA_P13["stationsnr"] if stn is not None]


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


def calculate_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Bereken helling, intercept en R^2 van de lineaire regressie y = a*x + b."""
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def make_scatterplot(daily: pd.DataFrame, slope: float, intercept: float, r2: float) -> go.Figure:
    """Scatterplot met regressielijn en 1:1-lijn."""
    x = daily["neerslag_P13"].to_numpy(dtype=float)
    y = daily["neerslag_all"].to_numpy(dtype=float)
    x_line = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="markers", name="dagwaarden",
            marker=dict(size=5, color=MARKER_COLOR, opacity=0.55),
            text=daily["datum"].dt.strftime("%Y-%m-%d"),
            hovertemplate="%{text}<br>P13: %{x:.2f} mm<br>Alle: %{y:.2f} mm<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line, y=slope * x_line + intercept, mode="lines",
            name=f"regressie (R² = {r2:.4f})",
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
        xaxis_title="Gemiddelde neerslag P13-stations (mm)",
        yaxis_title="Gemiddelde neerslag alle stations (mm)",
        template="plotly_white",
        height=PLOT_HEIGHT,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=60, r=30, t=30, b=60),
    )
    return fig


def stations_dataframe() -> pd.DataFrame:
    """De P13-stationstabel zoals in de tekst beschreven."""
    stations = pd.DataFrame(DATA_P13)
    stations["stationsnr"] = stations["stationsnr"].astype("Int64")
    return stations


def get_weerstations() -> list[list]:
    """Stationsnummers en -namen van de KNMI-weerstations."""
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


@st.cache_data(ttl="24h", max_entries=4, show_spinner=False)
def load_weerstations(url: str = WEERSTATIONS_URL) -> pd.DataFrame:
    """Haal de coordinaten van de weerstations op en voeg de P13-categorie toe."""
    df_map = pd.read_csv(url, comment="#", delimiter=",", low_memory=False)
    df_map = df_map[["original_Name", "station_nr", "lat", "lon"]].copy()
    df_map["station_nr"] = pd.to_numeric(df_map["station_nr"], errors="coerce").astype("Int64")

    # Namen uit get_weerstations() aanhouden waar het stationsnummer bekend is
    namen = {int(nr): naam for nr, naam in get_weerstations()}
    df_map["original_Name"] = [
        namen.get(int(nr), naam) if pd.notna(nr) else naam
        for nr, naam in zip(df_map["station_nr"], df_map["original_Name"])
    ]

    # De twee P13-stations zonder KNMI-data erbij zetten
    df_map = pd.concat([df_map, pd.DataFrame(P13_ONTBREKENDE_STATIONS)], ignore_index=True)
    df_map["station_nr"] = df_map["station_nr"].astype("Int64")

    p13_stations = p13_station_numbers()
    ontbrekend = {row["original_Name"] for row in P13_ONTBREKENDE_STATIONS}

    def _categorie(row: pd.Series) -> str:
        if row["original_Name"] in ontbrekend:
            return CAT_P13_ONTBREEKT
        if pd.notna(row["station_nr"]) and int(row["station_nr"]) in p13_stations:
            return CAT_P13_GEBRUIKT
        return CAT_OVERIG

    df_map["categorie"] = df_map.apply(_categorie, axis=1)
    df_map["color"] = df_map["categorie"].map(CATEGORY_COLORS)

    # P13-stations bovenop de rest tekenen
    volgorde = {CAT_OVERIG: 0, CAT_P13_GEBRUIKT: 1, CAT_P13_ONTBREEKT: 2}
    df_map = (
        df_map.assign(_z=df_map["categorie"].map(volgorde))
        .sort_values("_z")
        .drop(columns="_z")
        .reset_index(drop=True)
    )
    return df_map


def make_stations_deck(df_map: pd.DataFrame, show_labels: bool) -> pdk.Deck:
    """Bouw de pydeck-kaart met gekleurde stations."""
    midpoint = (float(np.average(df_map["lat"])), float(np.average(df_map["lon"])))

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            df_map,
            get_position=["lon", "lat"],
            auto_highlight=True,
            get_radius=MARKER_RADIUS,
            get_fill_color="color",
            pickable=True,
        )
    ]
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

    tooltip = {"html": "<b>{original_Name}</b><br/>station {station_nr}<br/>{categorie}"}

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


def render_weerstations_tab() -> None:
    """Tabblad met de kaart van alle KNMI-weerstations."""
    try:
        with st.spinner("Stationslocaties ophalen..."):
            df_map = load_weerstations()
    except Exception as exc:  # netwerk- of parsefout
        st.error(f"Kon de stationslijst niet ophalen: {exc}", icon=":material/error:")
        return

    st.markdown(
        f":green-badge[{CAT_P13_GEBRUIKT}] "
        f":red-badge[{CAT_P13_ONTBREEKT}] "
        f":gray-badge[{CAT_OVERIG}]"
    )
    st.caption(
        "Kerkwerve en Oudenbosch zijn handmatig toegevoegd; voor deze twee P13-stations "
        "levert de KNMI-API geen bruikbare reeks."
    )

    show_labels = st.checkbox("Stationsnamen op de kaart tonen", value=True)
    st.pydeck_chart(make_stations_deck(df_map, show_labels), height=MAP_HEIGHT)

    st.dataframe(
        df_map.drop(columns=["color"]),
        hide_index=True,
        width="stretch",
        column_config={
            "original_Name": st.column_config.TextColumn("Station", pinned=True),
            "station_nr": st.column_config.NumberColumn("Stationsnr", format="%d"),
            "lat": st.column_config.NumberColumn("Breedtegraad", format="%.4f"),
            "lon": st.column_config.NumberColumn("Lengtegraad", format="%.4f"),
            "categorie": st.column_config.TextColumn("Categorie"),
        },
    )

    st.link_button(":material/open_in_new: KNMI-stations op Google Maps", GOOGLE_MAPS_URL)


def render_sidebar() -> dict:
    """Invoerinstellingen in de sidebar."""
    with st.sidebar:
        st.header("Instellingen")
        input_dir = st.text_input("Map met de csv-bestanden", value=DEFAULT_INPUT_DIR)
        pattern = st.text_input("Bestandspatroon", value=DEFAULT_FILE_PATTERN)
        require_complete_p13 = st.checkbox(
            "Alleen dagen met alle P13-stations", value=True,
            help="Sluit dagen uit waarop niet alle beschikbare P13-stations een meting hebben.",
        )
        exclude_p13_from_all = st.checkbox(
            "P13-stations uitsluiten uit 'alle stations'", value=False,
            help="Standaard zitten de P13-stations ook in het landelijke gemiddelde.",
        )
        min_stations_all = st.number_input(
            "Minimum aantal stations voor landelijk gemiddelde",
            min_value=1, max_value=100, value=1, step=1,
        )
    return {
        "input_dir": input_dir,
        "pattern": pattern,
        "require_complete_p13": require_complete_p13,
        "exclude_p13_from_all": exclude_p13_from_all,
        "min_stations_all": int(min_stations_all),
    }


def render_analyse_tab(daily: pd.DataFrame) -> None:
    """Tabblad met kerncijfers en scatterplot."""
    if daily.empty:
        st.warning("Geen overlappende dagen gevonden met de huidige instellingen.")
        return

    x = daily["neerslag_P13"].to_numpy(dtype=float)
    y = daily["neerslag_all"].to_numpy(dtype=float)
    slope, intercept, r2 = calculate_r2(x, y)
    corr = float(np.corrcoef(x, y)[0, 1])

    with st.container(horizontal=True):
        st.metric("R²", f"{r2:.4f}", border=True)
        st.metric("Pearson r", f"{corr:.4f}", border=True)
        st.metric("Aantal dagen", f"{len(daily):,}".replace(",", "."), border=True)
        st.metric("Gem. P13", f"{x.mean():.2f} mm", border=True)
        st.metric("Gem. alle", f"{y.mean():.2f} mm", border=True)

    st.caption(
        f"Periode {daily['datum'].min():%d-%m-%Y} t/m {daily['datum'].max():%d-%m-%Y} · "
        f"regressie: neerslag_all = {slope:.4f} × neerslag_P13 + {intercept:.4f}"
    )

    st.plotly_chart(make_scatterplot(daily, slope, intercept, r2), width="stretch")


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
        ":material/download: Download als csv",
        data=daily.to_csv(index=False).encode("utf-8"),
        file_name="neerslag_p13_vs_all.csv",
        mime="text/csv",
    )


def render_stations_tab() -> None:
    """Tabblad met de achtergrondinformatie en de stationstabel."""
    st.markdown(ACHTERGROND_TEKST)

    stations = stations_dataframe()
    st.dataframe(
        stations, width="stretch",
        column_config={
            "stationsnr": st.column_config.NumberColumn("stationsnr", format="%d"),
            "genoemd in tekst": st.column_config.TextColumn("genoemd in tekst"),
            "gebruikte data": st.column_config.TextColumn("gebruikte data"),
        },
    )

    gebruikt = len(p13_station_numbers())
    st.caption(
        f"{gebruikt} van de {len(stations)} P13-stations hebben data en tellen mee in het "
        "P13-gemiddelde. Kerkwerve en Oudenbosch zijn niet opgenomen."
    )


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

    tab_analyse, tab_data, tab_stations, tab_kaart = st.tabs(
        [
            ":material/scatter_plot: Analyse",
            ":material/table: Data",
            ":material/pin_drop: Stations en achtergrond",
            ":material/map: Weerstations",
        ]
    )

    with tab_analyse:
        render_analyse_tab(daily)
    with tab_data:
        render_data_tab(daily, info)
    with tab_stations:
        render_stations_tab()
    with tab_kaart:
        render_weerstations_tab()


if __name__ == "__main__":
    main()