from __future__ import annotations

import streamlit as st
import pandas as pd

from datetime import datetime, timezone
from zoneinfo import ZoneInfo   

import math
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import pandas as pd

import numpy as np
import plotly.graph_objects as go

try:
# if 1==1:
   
    from utils import get_data, getdata_wrapper, check_from_until, calculate_heat_index, calculate_wind_chill, celsius_to_fahrenheit, fahrenheit_to_celsius
    from solar_app import solar_wrapper
    from liljegren_wbgt import wbgt_liljegren_from_station, KNMI_STATIONS, wbgt_liljegren
    from select_time_place import select_time_place
except:

    from show_knmi_functions.utils import get_data, calculate_heat_index, calculate_wind_chill, celsius_to_fahrenheit, fahrenheit_to_celsius
    from show_knmi_functions.solar_app import solar_wrapper
    from show_knmi_functions.liljegren_wbgt import wbgt_liljegren_from_station, KNMI_STATIONS, wbgt_liljegren
    from show_knmi_functions.select_time_place import select_time_place

# ======================== DUBBELOP maar zorgt voor kring-imports
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

# =======================================

def prepare_data():
    """ Van ruwe data naar een opgeslagn DataFrame met alle benodigde kolommen voor analyse en visualisatie.
    Zodat je niet steeds opnieuw gegevens hoeeft op te halen bij het KNMI.
    De link geeft waarschijnlijk rond de 10 jaar aan data door. Het gebruikte bestand is handmatig opgehaald, maar
    het zou waarschijnlijk ook automatisch kunnen. """
    # https://www.daggegevens.knmi.nl/klimatologie/uurgegevens?stns=260&vars=T:U:FH:Q&start=2011010100&end=2025070323
    # url = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\show_knmi_functions\data_wbgt_1991_2026.csv"
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/show_knmi_functions/wbgt_results_1990_2026.csv"
    df = pd.read_csv(url, delimiter=",",
                header= None,
                comment="#",
                low_memory=False,)
    st.write(df)
    column_replacements = [
            [0, "STN"],
            [1, "YYYYMMDD"],
            [2, "HH"],
            [3, "F"],
            [4, "T"],
            [5, "Q"],
            [6, "U"],
            
        ]

    for c in column_replacements:
        df = df.rename(columns={c[0]: c[1]})
    
    df["YYYYMMDD"] = pd.to_datetime(df["YYYYMMDD"].astype(str))
    df["YYYY"] = df["YYYYMMDD"].dt.year
    df["MM"] = df["YYYYMMDD"].dt.month
    df["DD"] = df["YYYYMMDD"].dt.day
    df["dayofyear"] = df["YYYYMMDD"].dt.dayofyear
    df["id"] =  range(1, len(df) + 1)
    st.write (df)
    # df_result = wbgt_bereken_df(df)
    df_result = wbgt_bereken_df(df, stn=260)
    st.write(df_result)
    
    df_result.to_csv("wbgt_results_1990_2026.csv", index=True)
    st.write("DONE")

def referentie_tabel_based_on_history(df):
    """ We maken een referentie tabel, gebaseerd op de gegevens 1991-2025. Voor elke temperatuur/RH combinatie
    wordt de gemiddelde wbgt_buiten berekend en in de tabel gezet. Hierdoor kan de gebruiker een inschatting maken
    van de hittekracht zonder de straling en de wind te hoeven te weten. De inschatting is veelal  +/- 1 zoals in een andere
    grafiek op deze pagina te zien is"""


    temps = list(range(16, 36, 2))
    rhs   = list(range(20, 105, 5))
    z = []
    z_sd =[]
    z_median=[]
    z_aantal=[]
    for rh in rhs:

        rij = []
        rij_sd = []
        rij_median=[]
        rij_aantal=[]
        for temp in temps:
    
            df_temp=df[(df["temp_c"] > temp-1) & (df["temp_c"] < temp+1) & (df["rh_pct"] > rh-2.5) & (df["rh_pct"] < rh+2.5)]
            if len(df_temp)>0:
                waarde = df_temp["wbgt_buiten"].mean()
                waarde_median = df_temp["wbgt_buiten"].median()
                waarde_stdev = df_temp["wbgt_buiten"].std()
                waarde_aantal = len(df_temp)
            else:
                waarde = 0
                waarde_stdev = 0
                waarde_median=0
                waarde_aantal=0
            rij.append(round(waarde, 1))
            rij_sd.append(round(waarde_stdev, 1))
            rij_median.append(round(waarde_median, 1))
            rij_aantal.append(round(waarde_aantal, 0))

        z.append(rij)
        z_sd.append(rij_sd)
        z_median.append(rij_median)
        z_aantal.append(rij_aantal)
    

    zmin, zmax = 14, 32

    def naar_schaal(v):
        return (v - zmin) / (zmax - zmin)
    
    

    colorscale_knmi = [
        [0.0,                    "#FFFFFF"],   # None
     
        # [0.0000001,        "#E7FFCF"],   # HK 0: <14
        [naar_schaal(14),        "#E7FFCF"],   # HK 0: <14
        [naar_schaal(14),        "#8FD14F"],   # HK 1: 14-16
        [naar_schaal(16),        "#8FD14F"],
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


    zmin, zmax = 14, 32
    colorscale = colorscale_knmi
    title = f"KNMI Hitte Kracht, de gemiddelde waardes 1991-2025 voor een bepaalde T and RH combinatie"

    def _maak_heatmap(z, temps, rhs, title, colorscale=None, zmin=None, zmax=None) -> go.Figure:
        fig = go.Figure(go.Heatmap(
            z=z,
            x=temps,
            y=rhs,
            text=[[str(v) if v != 0 else "" for v in rij] for rij in z],
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            showscale=False,
            xgap=1,
            ygap=1,
        ))
        fig.update_layout(
            title=dict(text=title,  font_size=14), # x=0, y=0,
            xaxis=dict(showgrid=True, zeroline=True, title="Temperatuur (°C)",
                    tickvals=temps, tickmode="array", side="bottom"),
            yaxis=dict(showgrid=True, zeroline=True, title="Relatieve vochtigheid (%)",
                    tickvals=rhs, tickmode="array", autorange="reversed"),
            plot_bgcolor="#777777",
            height=600,
            margin=dict(l=80, r=40, t=100, b=60, pad=0),
        )
        return fig
    number_of_charts=3
    if number_of_charts==4:
        # Aanroepen:
        panels = [
            (z,        title,    colorscale, zmin, zmax),
            (z_median, "Median", colorscale, zmin, zmax),
            (z_sd,     "SD",     None,       None, None),
            (z_aantal, "Aantal", None,       None, None),
        ]

        for (za, zb), (z1, t1, cs1, mn1, mx1), (z2, t2, cs2, mn2, mx2) in zip(
            [st.columns(2), st.columns(2)],
            panels[0::2],
            panels[1::2],
        ):
            with za:
                st.plotly_chart(_maak_heatmap(z1, temps, rhs, t1, cs1, mn1, mx1), width="stretch")
            with zb:
                st.plotly_chart(_maak_heatmap(z2, temps, rhs, t2, cs2, mn2, mx2), width="stretch")
    else:
        panels = [
            (z,        title,  colorscale, zmin, zmax),
            (z_sd,     "Standard deviatie van de wbgt_buiten-waardes",   None,       None, None),
            (z_aantal, "Aantal van een bepaalde Temp en RH combinatie", None,     None, None),
        ]

        c1, c2, c3 = st.columns(3)
        for col, (z_, t_, cs_, mn_, mx_) in zip([c1, c2, c3], panels):
            with col:
                st.plotly_chart(_maak_heatmap(z_, temps, rhs, t_, cs_, mn_, mx_), width="stretch")
    
def histogram_risico(df: pd.DataFrame) -> None:
    """Toon histogram van wbgt_risico_niveau voor gegeven temp_c en rh_pct."""
    
    volgorde = ["HK 0", "HK 1", "HK 2", "HK 3", "HK 4", "HK 5",
                "HK 6", "HK 7", "HK 8", "HK 9", "HK 10"]
    kleuren = BADGE_KLEUREN_KNMI  # jouw bestaande dict

    counts = (
        df["wbgt_risico_niveau"]
        .value_counts()
        .reindex(volgorde, fill_value=0)
        .reset_index()
    )
    counts.columns = ["niveau", "aantal"]

    fig = go.Figure(go.Bar(
        x=counts["niveau"],
        y=counts["aantal"],
        marker_color=[kleuren.get(n, "#aaaaaa") for n in counts["niveau"]],
        text=counts["aantal"],
        textposition="outside",
    ))

    fig.update_layout(
        title="Verdeling WBGT risiconiveaus",
        xaxis_title="Risiconiveau",
        yaxis_title="Aantal uren",
        plot_bgcolor="white",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig, width="stretch")

def histogram_wbgt(df_risico, gemiddeld_wbgt) -> None:
    # Histogram
    fig = go.Figure(go.Histogram(
        x=df_risico["wbgt_buiten"],
        xbins=dict(start=df_risico["wbgt_buiten"].min(), size=0.5),
        marker_color="#e05c00",
        opacity=0.85,
    ))

    # Verticale lijn op gemiddelde
    fig.add_vline(x=gemiddeld_wbgt, line_dash="dash", line_color="black",
                annotation_text=f"gem. {gemiddeld_wbgt:.1f}°C", annotation_position="top right")

    fig.update_layout(
        title="Verdeling WBGT buiten (bins 0.5 °C)",
        xaxis_title="WBGT (°C)",
        yaxis_title="Aantal uren",
        plot_bgcolor="white",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig, width="stretch")
    
def make_histogram_temp_rh(df_risico, what):
    fig_t = go.Figure(go.Histogram(
        x=df_risico["temp_c"],
        xbins=dict(size=0.5),
        marker_color="#4a90d9",
        opacity=0.85,
    ))
    fig_t.update_layout(
        title=f"Verdeling temperatuur {what}",
        xaxis_title="Temp (°C)", yaxis_title="Aantal uren",
        plot_bgcolor="white", height=350,
    )

    fig_rh = go.Figure(go.Histogram(
        x=df_risico["rh_pct"],
        xbins=dict(size=2),
        marker_color="#27ae60",
        opacity=0.85,
    ))
    fig_rh.update_layout(
        title=f"Verdeling relatieve vochtigheid - {what}",
        xaxis_title="RV (%)", yaxis_title="Aantal uren",
        plot_bgcolor="white", height=350,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_t, width="stretch")
    with c2:
        st.plotly_chart(fig_rh, width="stretch")

def make_histogram_wind_q(df_risico, what):
    fig_t = go.Figure(go.Histogram(
        x=df_risico["wind_ms"],
        xbins=dict(size=0.5),
        marker_color="#4a90d9",
        opacity=0.85,
    ))
    fig_t.update_layout(
        title=f"Verdeling wind {what}",
        xaxis_title="wind (m/s)", yaxis_title="Aantal uren",
        plot_bgcolor="white", height=350,
    )

    fig_rh = go.Figure(go.Histogram(
        x=df_risico["q_wm2"],
        xbins=dict(size=2),
        marker_color="#27ae60",
        opacity=0.85,
    ))
    fig_rh.update_layout(
        title=f"straling - {what}",
        xaxis_title="Q (w/m2)", yaxis_title="Aantal uren",
        plot_bgcolor="white", height=350,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_t, width="stretch")
    with c2:
        st.plotly_chart(fig_rh, width="stretch")

@st.cache_data()
def get_data():
    # url=r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\show_knmi_functions\wbgt_results_1990_2026.csv"
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/show_knmi_functions/wbgt_results_1990_2026.csv"
    
    df = pd.read_csv(url, delimiter=",",
               
                comment="#",
                low_memory=False,)
    # st.write(df)
    # df = df[df["dt_utc"] <= pd.Timestamp("2025-07-03")]
    df = df[df["dt_utc"] >= "1991-01-01 00:00:01"]
    df = df[df["dt_utc"] <= "2025-07-03 23:59:59"]
    df_dagmax = df.loc[df.groupby("YYYYMMDD")["wbgt_buiten"].idxmax()].reset_index(drop=True)
    
    return df, df_dagmax
def show_historical_data():
    st.subheader("Informatie over data 1991-2025")
    st.write("We repliceren enkele grafieken van het RIVM rapport over hittekracht en voegen enkele nieuwe inzichten toe")
    
    
    _KNMI_DREMPELWAARDEN = [
        
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

    df,df_dagmax= get_data()

    referentie_tabel_based_on_history(df)
    # tabel B1
    col1,col2=st.columns(2)
    with col1:
        for w in ["wbgt_risico_niveau", "wbgt_risico_advies"]:
            counts = df.groupby(w).size().reset_index(name="aantal")
            counts["pct"] = (counts["aantal"] / len(df) * 100).round(2)
            # with st.expander(f"Opgeslitst naar {w}"):
            st.write(f"{w} Alle waardes")
            st.write (counts)
    with col2:
        for w in ["wbgt_risico_niveau", "wbgt_risico_advies"]:
            counts = df_dagmax.groupby(w).size().reset_index(name="aantal")
            counts["pct"] = (counts["aantal"] / len(df_dagmax) * 100).round(2)
            # with st.expander(f"Opgeslitst naar {w}"):
            st.write(f"{w} (dagmax)")
            st.write (counts)

    
    make_histogram_temp_rh(df, "alle waardes")
    
    make_histogram_temp_rh(df_dagmax, "dagmax")

    col1,col2=st.columns(2)
    with col1:
        temp = st.number_input("temperatuur", 0,100,25)
    with col2:

        rh = st.number_input("RH",0,100,50)

    df_risico=df[(df["temp_c"] > temp-.5) & (df["temp_c"] < temp+.5) & (df["rh_pct"] > rh-2.5) & (df["rh_pct"] < rh+2.5)]
    make_histogram_wind_q(df_risico, f"[selected temp={temp}, rh={rh}]")
    
    gemiddeld_wbgt = df_risico["wbgt_buiten"].mean()
    
    
    # Klasse bepalen
    niveau, advies = wbgt_risico(gemiddeld_wbgt)
    badge_kleur = BADGE_KLEUREN_KNMI.get(niveau, "#aaaaaa")

    st.markdown(
        f'<div style="background-color:{badge_kleur};color:white;padding:10px 16px;'
        f'border-radius:8px;display:inline-block;font-weight:bold;">'
        f'Gemiddelde WBGT: {gemiddeld_wbgt:.1f} °C — {niveau} ({advies})</div>',
        unsafe_allow_html=True,
    )
    col1,col2=st.columns(2)
    with col1:
        histogram_wbgt(df_risico,gemiddeld_wbgt)
    with col2:
        histogram_risico(df_risico)
    
def main():
    show_historical_data()
if __name__=="__main__":
    main()