import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress
import streamlit as st


def verschoven_decennium(jaar):
    return ((jaar - 6) // 10) * 10 + 10  # bijvoorbeeld: 2016–2025 → 2020


def show_warme_dagen_(df, gekozen_weerstation, what_to_show_, afkapgrens,titel):
    """
    Toon het aantal tropische dagen per decennium in De Bilt.
    Geinspireerd door https://x.com/deheij/status/1941006395190423584

    Args:
        df (pd.DataFrame): DataFrame met de weerdata.
        gekozen_weerstation (str): Naam van het gekozen weerstation.
        what_to_show_ (str): Wat er getoond moet worden, hier niet gebruikt.
        afkapgrens (float): De temperatuurgrens voor tropische dagen, standaard 30.0°C.
        titel (str): warm/tropisch/hitte
    """
    
   # Zet de datumkolom om naar datetime
    
    # Data inladen
   
    df['datum'] = df['YYYYMMDD']
    df['jaar'] = df['datum'].dt.year
    #df['decennium'] = (df['jaar'] // 10) * 10
    df['decennium'] = df['jaar'].apply(verschoven_decennium)
    # Filter tropische dagen
    df_tropisch = df[df['temp_max'] > afkapgrens]

    # Aantal tropische dagen per jaar
    tropisch_per_jaar = df_tropisch.groupby('jaar').size().reindex(range(df['jaar'].min(), df['jaar'].max()+1), fill_value=0)

    # Dataframe bouwen
    df_tropisch_per_jaar = pd.DataFrame({
        'jaar': tropisch_per_jaar.index,
        'tropische_dagen': tropisch_per_jaar.values
    })
    #df_tropisch_per_jaar['decennium'] = (df_tropisch_per_jaar['jaar'] // 10) * 10


    df_tropisch_per_jaar['decennium'] = df_tropisch_per_jaar['jaar'].apply(verschoven_decennium)
    #df_tropisch_per_jaar['decennium'] = df_tropisch_per_jaar['jaar'].apply(aangepaste_decennium)
    # Gemiddelde en standaarddeviatie per decennium
    samenvatting = df_tropisch_per_jaar.groupby('decennium')['tropische_dagen'].agg(['mean', 'std']).reset_index()

    # Trendlijn
    slope, intercept, *_ = linregress(samenvatting['decennium'], samenvatting['mean'])
    samenvatting['trend'] = intercept + slope * samenvatting['decennium']


# # Samenvatting: gemiddelde + std per decennium
# samenvatting = df_tropisch_per_jaar.groupby('decennium')['tropische_dagen'].agg(['mean', 'std']).reset_index()

    # Mooie labels: bijv. 1996–2005
    samenvatting['label'] = (samenvatting['decennium'] - 4).astype(str) + '–' + (samenvatting['decennium']+5).astype(str)
    df_tropisch_per_jaar['label'] = df_tropisch_per_jaar['decennium'].apply(lambda d: f"{d-4}–{d+5}")

    # Trendlijn
    slope, intercept, *_ = linregress(samenvatting['decennium'], samenvatting['mean'])
    samenvatting['trend'] = intercept + slope * samenvatting['decennium']

    # Plotly-figuur
    fig = go.Figure()

    # Barplot met foutmarges
    fig.add_trace(go.Bar(
        x=samenvatting['label'],
        y=samenvatting['mean'],
        error_y=dict(type='data', array=samenvatting['std'], visible=True),
        name='Gemiddelde per (verschoven) decennium',
        marker_color='red',
        opacity=0.7
    ))

    # Scatterplot met individuele jaren
    fig.add_trace(go.Scatter(
        x=df_tropisch_per_jaar['label'],
        y=df_tropisch_per_jaar['tropische_dagen'],
        mode='markers',
        name='Jaarlijkse waarden',
        marker=dict(size=4, color='blue', opacity=0.6),
        # hovertext=f"{df_tropisch_per_jaar['jaar'].astype(str)} | {df_tropisch_per_jaar['tropische_dagen'].astype(str)} dagen (> {afkapgrens}°C)",
       
    ))

    # Trendlijn
    fig.add_trace(go.Scatter(
        x=samenvatting['label'],
        y=samenvatting['trend'],
        mode='lines',
        name=f"Trend: +{slope:.2f} dagen per (verschoven) decennium",
        line=dict(dash='dash', color='darkred')
    ))

    # Layout
    fig.update_layout(
        title="Gemiddeld aantal tropische dagen (>30.0°C) per verschoven decennium in De Bilt",
        xaxis_title="Decennium",
        yaxis_title="Gemiddeld aantal dagen per jaar",
    template="simple_white"
)
    # Layout
    fig.update_layout(
        title=f"Gemiddeld aantal {titel} (>{afkapgrens}°C) per (verschoven) decennium in De Bilt",
        xaxis_title="Decennium",
        yaxis_title="Gemiddeld aantal dagen per jaar",
        template="simple_white"
    )

 
    st.plotly_chart(fig)
    with st.expander("Data"):
        st.write(df_tropisch_per_jaar)
        st.write(df_tropisch)
    
def  show_warme_dagen(df, gekozen_weerstation, what_to_show_):
    afkapgrenzen = [25.0, 30.0, 35.0]  # Lijst van afkapgrenzen voor warme dagen
    for i in afkapgrenzen:
        if i == 25.0:
            titel = "warme dagen"
        elif i == 30.0:
            titel = "tropische dagen"
        elif i == 35.0:
            titel = "hitte dagen"
        else:
            st.error("Ongeldige keuze voor afkapgrens.")
        show_warme_dagen_(df, gekozen_weerstation, what_to_show_, i, titel)

    st.info("Geinspireerd door https://x.com/deheij/status/1941006395190423584")

