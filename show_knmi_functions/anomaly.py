import pandas as pd
import numpy as np
import streamlit as st
#from streamlit import caching
import matplotlib.pyplot as plt
# import matplotlib

try:
    from show_knmi_functions.utils import get_data
except:
    from utils import get_data
#_lock = RendererAgg.lock
import sys # for the progressbar
import shutil # for the progressbar

import plotly.express as px
import plotly.graph_objects as go


def anomaly(df, what_):
    wdw=st.sidebar.number_input ("Window moving average", 1,365,31)
    one_color = st.sidebar.selectbox("One color for anomaly graph", [True,False], 1)
    calculate_last_year_with_avg =  st.sidebar.selectbox("Include last year in average", [True,False], 0)
    smooth_before_distracting = =  st.sidebar.selectbox("Smooth before distracting", [True,False], 0)
    for what in what_:
        st.subheader(what)
        df['date_1900'] = pd.to_datetime(df['YYYYMMDD'].dt.strftime('%m-%d-2000'), format='mixed')
        # Identify the most recent date in the dataset
        most_recent_date = df['YYYYMMDD'].max()
        if calculate_last_year_with_avg:
            average_temps = df.groupby('date_1900')[what].mean().reset_index()
        else:
            all_but_last_year = df[df['YYYYMMDD'] < (most_recent_date - pd.Timedelta(days=365))]
            average_temps = all_but_last_year.groupby('date_1900')[what].mean().reset_index()
       
        # Filter the data for the last 365 days from the most recent date
        last_year = df[df['YYYYMMDD'] >= (most_recent_date - pd.Timedelta(days=365+31))]
        df_anomalie = pd.merge(average_temps, last_year, on="date_1900")
        df_anomalie = df_anomalie.sort_values(by='YYYYMMDD')
        min_date= df['YYYYMMDD'].min().strftime("%d-%m-%Y")
        max_date = df['YYYYMMDD'].max().strftime("%d-%m-%Y")
        
        if smooth_before_distracting:
            df_anomalie[f"{what}_x"] = df_anomalie[f"{what}_x"] .rolling(wdw, center=False).mean()
            df_anomalie[f"{what}_y"] = df_anomalie[f"{what}_y"] .rolling(wdw, center=False).mean()
            df_anomalie["verschil"] = df_anomalie[f"{what}_y"]-  df_anomalie[f"{what}_x"]
        else:
                
            df_anomalie["verschil"] = df_anomalie[f"{what}_y"]-  df_anomalie[f"{what}_x"]
            df_anomalie["verschil"] = df_anomalie["verschil"] .rolling(wdw, center=False).mean()

            df_anomalie[f"{what}_x"] = df_anomalie[f"{what}_x"] .rolling(wdw, center=False).mean()
            df_anomalie[f"{what}_y"] = df_anomalie[f"{what}_y"] .rolling(wdw, center=False).mean()
            
        # Select the last 365 rows
        df_anomalie = df_anomalie.tail(366)

        plot_lines (df_anomalie, what, wdw)
        plot_anomalie_really (df_anomalie, what,wdw, one_color, min_date, max_date)

def plot_anomalie_really (df_anomalie, what,wdw, one_color, min_date, max_date):
    fig = go.Figure()
    if one_color:
        #Add average temperature trace
        fig.add_trace(go.Scatter(
            x=df_anomalie['YYYYMMDD'],
            y=df_anomalie["verschil"],
            mode='lines',
            fill='tozeroy',  # Fill to y=0
            name='difference'
        ))
    else:

        # Separate positive and negative values
        df_anomalie['pos_diff'] = df_anomalie['verschil'].apply(lambda x: x if x > 0 else 0)
        df_anomalie['neg_diff'] = df_anomalie['verschil'].apply(lambda x: x if x < 0 else 0)


        # Trace for positive differences
        fig.add_trace(go.Scatter(
            x=df_anomalie['YYYYMMDD'],
            y=df_anomalie['pos_diff'],
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.2)',  # Green fill with transparency
            line=dict(color='green'),
            name='Positive Difference'
        ))

        # Trace for negative differences
        fig.add_trace(go.Scatter(
            x=df_anomalie['YYYYMMDD'],
            y=df_anomalie['neg_diff'],
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)',  # Red fill with transparency
            line=dict(color='red'),
            name='Negative Difference'
        ))
    

    # Add a horizontal line at y=0
    fig.add_shape(type='line',
                x0=df_anomalie['YYYYMMDD'].min(),
                y0=0,
                x1=df_anomalie['YYYYMMDD'].max(),
                y1=0,
                line=dict(color='Black', width=2,))


    fig.update_layout(
            xaxis=dict(title="date",tickformat="%d-%m-%Y"),
            yaxis=dict(title=what),
            title=f"Anomaly of {what}, sma({wdw}) - last year compared ith average of {min_date} - {max_date} " ,)
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    
    st.plotly_chart(fig)

def plot_lines (df_anomalie, what, wdw):
    fig = go.Figure()

    #    Add average temperature trace
    fig.add_trace(go.Scatter(
        x=df_anomalie['YYYYMMDD'],
        y=df_anomalie[f"{what}_x"],
        mode='lines',
        name=f'Average {what}'
    ))

    # Add last year's temperature trace
    fig.add_trace(go.Scatter(
        x=df_anomalie['YYYYMMDD'],
        y=df_anomalie[f"{what}_y"],
        mode='lines',
        name=f'Last Year {what}'
    ))


    fig.update_layout(
            xaxis=dict(title="date",tickformat="%d-%m"),
            yaxis=dict(title=what),
            title=f"{what}" ,)
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    
    #fig.update_traces(hovertemplate=None)  # Disable hover info for faster rendering
    #fig.update_layout(showlegend=False)   # Disable legend for faster rendering

    # Create a spaghetti line plot
   
    #fig.update_layout(xaxis=dict(tickformat="%d-%m"))
    st.plotly_chart(fig)

def main():
   
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result.csv" 
    df = get_data(url)
    df["YYYY"] = df["YYYYMMDD"].dt.year
    df["MM"] = df["YYYYMMDD"].dt.month
    df["DD"] = df["YYYYMMDD"].dt.day
    df["dayofyear"] = df["YYYYMMDD"].dt.dayofyear
    df["year"] = df["YYYY"].astype(str)
    df["month"] = df["month"].astype(str)
    df["day"] = df["DD"].astype(str)
    df["month_year"] = df["month"] + " - " + df["year"]
    df["year_month"] = df["year"] + " - " +  df["MM"].astype(str).str.zfill(2)
    df["month_day"] = df["month"] + " - " + df["day"]
    anomaly(df, "temp_avg")



if __name__ == "__main__":
    main()
