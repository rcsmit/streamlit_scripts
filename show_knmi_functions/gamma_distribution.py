import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gamma
import plotly.graph_objects as go
import streamlit as st
import sys

try:
    from show_knmi_functions.utils import get_data, loess_skmisc
except:
    from utils import get_data, loess_skmisc


def gamma_distribution(df, what_to_show_, start_year, special_year):
    """    Analyseer de temperatuurdata en pas een gamma-verdeling toe per dag van het jaar.
    Args:
        df (pd.DataFrame): DataFrame met de weerdata.   
        what_to_show (str): De kolomnaam van de temperatuurdata die geanalyseerd moet worden.
        start_year (int): Het jaar vanaf wanneer de data geanalyseerd moet worden.
        special_year (int): Het jaar waarvoor de speciale trend getoond moet worden.
    """

    if st.button("GO"):
        what_to_show = what_to_show_[0]
        st.title(f"Gamma-verdeling van {what_to_show} per dag van het jaar (start {start_year})")
        placeholder = st.empty()
        df['date'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')
        df['day_of_year'] = df['date'].dt.dayofyear
        df['year'] = df['date'].dt.year
        df = df[df["year"] >= start_year]
        df = df[['day_of_year', 'year', what_to_show]].dropna()

        daily_stats = []

        for day, group in df.groupby('day_of_year'):
            placeholder.info(f"Processing day {day}...\r")
            #sys.stdout.flush()
            temps = group[what_to_show]
            if len(temps) < 5:
                continue
            mean = temps.mean()
            shape, loc, scale = gamma.fit(temps)
            p025 = gamma.ppf(0.025, shape, loc=loc, scale=scale)
            p975 = gamma.ppf(0.975, shape, loc=loc, scale=scale)
            p16 = gamma.ppf(0.16, shape, loc=loc, scale=scale)
            p84 = gamma.ppf(0.84, shape, loc=loc, scale=scale)
            daily_stats.append({
                'day_of_year': day,
                'mean': mean,
                'p025': p025,
                'p975': p975,
                'p16': p16,
                'p84': p84,
            })
        placeholder.empty()
        stats_df = pd.DataFrame(daily_stats)
        stats_df['date'] = pd.to_datetime(stats_df['day_of_year'], format='%j')

        df_special_year = df[df['year'] == special_year][['day_of_year', what_to_show]].dropna()
        df_special_year['date'] = pd.to_datetime(df_special_year['day_of_year'], format='%j')

        # Loess smoothing
        def loess_with_dates(x, y):
            x_int = x.astype(int)
            x_dates = pd.to_datetime(x_int, format='%j')
            _, y_hat, _, _ = loess_skmisc(x_int, y)
            return x_dates, y_hat

        x_mean, y_mean = loess_with_dates(stats_df['day_of_year'], stats_df['mean'])
        x_025, y_025 = loess_with_dates(stats_df['day_of_year'], stats_df['p025'])
        x_975, y_975 = loess_with_dates(stats_df['day_of_year'], stats_df['p975'])
        x_16, y_16 = loess_with_dates(stats_df['day_of_year'], stats_df['p16'])
        x_84, y_84 = loess_with_dates(stats_df['day_of_year'], stats_df['p84'])
        x_special, y_special = loess_with_dates(df_special_year['day_of_year'], df_special_year[what_to_show])

        fig = go.Figure()

        # 2.5–97.5% vlak
        fig.add_trace(go.Scatter(
            x=list(x_025) + list(x_975[::-1]),
            y=list(y_025) + list(y_975[::-1]),
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            name='2.5%-97.5%'
        ))

        # 16–84% vlak
        fig.add_trace(go.Scatter(
            x=list(x_16) + list(x_84[::-1]),
            y=list(y_16) + list(y_84[::-1]),
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            name='16%-84%'
        ))

        # Gemiddelde
        fig.add_trace(go.Scatter(
            x=x_mean,
            y=y_mean,
            mode='lines',
            name='Gemiddelde',
            line=dict(color='black')
        ))

    # Special year trend line
        fig.add_trace(go.Scatter(
            x=x_special,
            y=y_special,
            mode='lines',
            name=f"{special_year} (trend)",
            line=dict(color='blue')
        ))

        # Special year raw data line
        fig.add_trace(go.Scatter(
            x=df_special_year['date'],
            y=df_special_year[what_to_show],
            mode='lines+markers',
            name=f'{special_year} (raw data)',
            line=dict(color='lightblue', width=1),
            marker=dict(size=3, color='lightblue', opacity=0.6)
        ))


        fig.update_layout(
            title=f"Temperatuurverloop per dag ({what_to_show}) met gamma-verdeling (start {start_year}) en LOESS-trend",
            xaxis_title="Datum",
            yaxis_title=f"{what_to_show} (°C)",
            xaxis=dict(
                tickformat="%d-%m",
                dtick="M1"
            ),
            template="plotly_white"
        )

        st.plotly_chart(fig)


def main():
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result_1901.csv"
    df = get_data(url)
    gamma_distribution(df, ["temp_max"],1900, 2025)
    #gamma_distribution(df, ["temp_max"],2020, 2025)

if __name__ == "__main__":
    main()
