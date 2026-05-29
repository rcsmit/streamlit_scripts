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

# FUNCTIES EN CONSTANTEN DIE IN MEERDERE BESTANDEN WORDEN GEBRUIKT



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


ZONE_KLEUREN_WBGT = {z["label"]: z["color"] for z in RISICO_ZONES_WBGT}
ZONE_KLEUREN_KNMI = {z["label"]: z["color"] for z in RISICO_ZONES_KNMI}


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

BADGE_KLEUREN_WBGT = {
    "Laag":       "green",
    "Matig":      "orange",
    "Hoog":       "orange",
    "Zeer hoog":  "red",
    "Gevaarlijk": "red",
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

def maak_wbgt_barchart(df: pd.DataFrame, datum: str | None = None) -> go.Figure:
    """Staafdiagram van uurlijkse WBGT met hittekracht-kleuren, zoals KNMI-rapport Fig.b2

    Args:
        df:    DataFrame zoals geproduceerd door wbgt_bereken_df(), met kolom
               dt_utc, wbgt_buiten, wbgt_risico_niveau, HH.
        datum: 'YYYY-MM-DD' string om één dag te selecteren. Als None: laatste dag.

    Returns:
        Plotly Figure.
    """
    # --- Daginschnitt ---
    
    df = df.copy()
    
    try:
        df["_date_str"] = df["dt_utc"].dt.strftime("%Y-%m-%d")

        if datum is None:
            datum = df["_date_str"].iloc[-1]

        dag = df[df["_date_str"] == datum].copy()
    except:
        # df["_date_str"] = df["dt_utc"].dt.strftime("%Y-%m-%d")

        if datum is None:
            datum = df["YYYYMMDD"].iloc[-1]

        dag = df[df["YYYYMMDD"] == datum].copy()
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