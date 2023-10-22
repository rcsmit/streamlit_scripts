from imghdr import what
import pandas as pd
import numpy as np
import streamlit as st
#from streamlit import caching
import datetime as dt
import scipy.stats as stats
import math
from show_knmi_functions.utils import show_weerstations, help
from datetime import datetime
import matplotlib.pyplot as plt
# import matplotlib
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

_lock = RendererAgg.lock
import sys # for the progressbar
import shutil # for the progressbar

import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go

import platform
import streamlit.components.v1 as components
import time
import imageio
import os
import webbrowser

def plot_percentiles(df, gekozen_weerstation, what_to_show, wdw, centersmooth):
    if len(what_to_show)!=1 :
        st.warning("Choose (only) 1 thing to show")
        st.stop()

    df_quantile = pd.DataFrame(
        {"date": [],  "q10": [], "q25": [], "q50":[] ,"avg": [], "q75": [], "q90": []}    )
    year_to_show = st.sidebar.number_input("Year to highlight (2100 for nothing)", 1900, 2100, 2021)

    (month_from,month_until) = st.sidebar.slider("Months (from/until (incl.))", 1, 12, (1,12))
    if month_from > month_until:
        st.warning("Make sure that the end month is not before the start month")
        st.stop()
    df = df[
        (df["YYYYMMDD"].dt.month >= month_from) & (df["YYYYMMDD"].dt.month <= month_until)
    ]

    for month in list(range(1,13)):
        for day in list(range(1,32)):
            if month==2 and day==29:
                pass
            else:
                df_ = df[
                        (df["YYYYMMDD"].dt.month == month) & (df["YYYYMMDD"].dt.day == day)
                    ]

                df__ = df[
                        (df["YYYYMMDD"].dt.year == year_to_show) & (df["YYYYMMDD"].dt.month == month) & (df["YYYYMMDD"].dt.day == day)
                    ]

                if len(df__)>0:
                    value_in_year_ = df__[what_to_show].iloc[0]
                    value_in_year = value_in_year_[0]
                else:
                    value_in_year = None
                if len(df_)>0:
                    data = df_[what_to_show] #.tolist()
                    #st.write(data)

                    date_ = "1900-" +  str(month).zfill(2) + '-' + str(day).zfill(2)

                    q10 = np.percentile(data, 10)
                    q25 = np.percentile(data, 25)
                    q50 = np.percentile(data, 50)
                    q75 = np.percentile(data, 75)
                    q90 = np.percentile(data, 90)
                    avg = data.mean()

                                    
                    # Data for the new row
                    new_row = {
                        "date_": date_,
                        "q10": q10,
                        "q25": q25,
                        "q50": q50,
                        "avg": avg,
                        "q75": q75,
                        "q90": q90,
                        "value_in_year": value_in_year
                    }

                    # Append the new row to the DataFrame
                    df_quantile = pd.concat([df_quantile, pd.DataFrame([new_row])], ignore_index=True)


    df_quantile['date'] = pd.to_datetime(df_quantile.date_, format='%Y-%m-%d',  errors='coerce')

    columns = ["q10", "q25", "avg", "q50", "q75", "q90", "value_in_year"]
    for c in columns:
        df_quantile[c] = df_quantile[c].rolling(window=wdw, center=centersmooth).mean()
        df_quantile[c] = round(df_quantile[c],1)
    colors = ["red", "blue", ["yellow"]]
    title = (f" {what_to_show[0]} in {gekozen_weerstation} (percentiles (10/25/avg/75/90/))")
    graph_type = "plotly"
    if graph_type == "pyplot":

        with _lock:
            fig1x = plt.figure()
            ax = fig1x.add_subplot(111)
            idx = 0
            df_quantile.plot(x='date',y='avg', ax=ax, linewidth=0.75,
                            color=colors[idx],
                            label="avg")
            # df_quantile.plot(x='date',y='q50', ax=ax, linewidth=0.75,
            #                 color="yellow",
            #                 label="mediaan",  alpha=0.75)
            df_quantile.plot(x='date',y='value_in_year', ax=ax,
                            color="black",  linewidth=0.75,
                            label=f"value in {year_to_show}")
            ax.fill_between(df_quantile['date'],
                            y1=df_quantile['q25'],
                            y2=df_quantile['q75'],
                            alpha=0.30, facecolor=colors[idx])
            ax.fill_between(df_quantile['date'],
                            y1=df_quantile['q10'],
                            y2=df_quantile['q90'],
                            alpha=0.15, facecolor=colors[idx])

            ax.set_xticks(df_quantile["date"].index)
            # if datefield == "YYYY":
            #     ax.set_xticklabels(df[datefield], fontsize=6, rotation=90)
            # else:
            ax.set_xticklabels(df_quantile["date"], fontsize=6, rotation=90)
            xticks = ax.xaxis.get_major_ticks()
            for i, tick in enumerate(xticks):
                if i % 10 != 0:
                    tick.label1.set_visible(False)

            # plt.xticks()
            plt.grid(which="major", axis="y")
            plt.title(title)
            plt.legend()
            st.pyplot(fig1x)
    else:
        fig = go.Figure()
        q10 = go.Scatter(
            name='q10',
            x=df_quantile["date"],
            y=df_quantile['q10'] ,
            mode='lines',
            line=dict(width=0.5,
                    color="rgba(255, 188, 0, 0.5)"),
            fillcolor='rgba(68, 68, 68, 0.1)', fill='tonexty')

        q25 = go.Scatter(
            name='q25',
            x=df_quantile["date"],
            y=df_quantile['q25'] ,
            mode='lines',
            line=dict(width=0.5,
                    color="rgba(255, 188, 0, 0.5)"),
            fillcolor='rgba(68, 68, 68, 0.2)',
            fill='tonexty')

        avg = go.Scatter(
            name=what_to_show[0],
            x=df_quantile["date"],
            y=df_quantile["avg"],
            mode='lines',
            line=dict(width=0.75,color='rgba(68, 68, 68, 0.8)'),
            )

        value_in_year__ = go.Scatter(
            name=year_to_show,
            x=df_quantile["date"],
            y=df_quantile["value_in_year"],
            mode='lines',
            line=dict(width=0.75,color='rgba(255, 0, 0, 0.8)'),
            )

        q75 = go.Scatter(
            name='q75',
            x=df_quantile["date"],
            y=df_quantile['q75'] ,
            mode='lines',
            line=dict(width=0.5,
                    color="rgba(255, 188, 0, 0.5)"),
            fillcolor='rgba(68, 68, 68, 0.1)',
            fill='tonexty')

        q90 = go.Scatter(
            name='q90',
            x=df_quantile["date"],
            y=df_quantile['q90'],
            mode='lines',
            line=dict(width=0.5,
                    color="rgba(255, 188, 0, 0.5)"),
            fillcolor='rgba(68, 68, 68, 0.1)'
        )

        data = [q90, q75, q25, q10,avg, value_in_year__ ]

        layout = go.Layout(
            yaxis=dict(title=what_to_show[0]),
            title=title,)
            #, xaxis=dict(tickformat="%d-%m")
        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(xaxis=dict(tickformat="%d-%m"))
        st.plotly_chart(fig, use_container_width=True)
        # fig.show()