from __future__ import annotations

import streamlit as st
import pandas as pd

from datetime import datetime, timezone
from zoneinfo import ZoneInfo   
from wbgt_knmi import *

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


def prepare_data():
    # https://www.daggegevens.knmi.nl/klimatologie/uurgegevens?stns=260&vars=T:U:FH:Q&start=2011010100&end=2025070323
    url = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\show_knmi_functions\data_wbgt_1991_2026.csv"
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
    url=r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\show_knmi_functions\wbgt_results_1990_2026.csv"

    df = pd.read_csv(url, delimiter=",",
               
                comment="#",
                low_memory=False,)
    st.write(df)
    # df = df[df["dt_utc"] <= pd.Timestamp("2025-07-03")]
    df = df[df["dt_utc"] >= "1991-01-01 00:00:01"]
    df = df[df["dt_utc"] <= "2025-07-03 23:59:59"]
    df_dagmax = df.loc[df.groupby("YYYYMMDD")["wbgt_buiten"].idxmax()].reset_index(drop=True)
    
    return df, df_dagmax
def show_historical_data():
    df,df_dagmax= get_data()
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
    make_histogram_wind_q(df_risico, f"selected {temp} {rh}")
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