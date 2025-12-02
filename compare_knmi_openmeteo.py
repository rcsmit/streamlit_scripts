# Scrapping Weather Data from open-meteo
#
# Based on code of @orwel2022

# To install the needed packages : 
# pip install pandas requests beautifulsoup4 html5lib


import pandas as pd
from datetime import datetime, timedelta
# import requests
# from bs4 import BeautifulSoup

# import plotly.express as px
# import plotly.graph_objects as go
import streamlit as st
# import platform
# import os
# from io import StringIO
# import matplotlib.pyplot as plt
# from show_knmi_functions.show_calender_heatmap import show_calender_heatmap
# from show_knmi_functions.show_year_heatmap import show_year_heatmap

#from matplotlib.backends.backend_agg import RendererAgg
# from matplotlib.colors import ListedColormap
# #_lock = RendererAgg.lock
# import numpy as np
# # when using without Streamlit, to avoid 127.0.0.1 refused to connect :
# # plotly.offline.init_notebook_mode(connected=True)


try:
    st.set_page_config(layout='wide')
except:
    pass


from weather_open_meteo import get_data_open_meteo, prepare_dataframe, check_from_until
from show_knmi_functions.utils import  get_data
from show_knmi import getdata_wrapper


def interface(locations):
    location_names = [loc["name"] for loc in locations]
    
    start_ = "1996-01-01"
    today = datetime.today().strftime("%Y-%m-%d")
    from_ = st.sidebar.text_input("startdatum (yyyy-mm-dd) from 1-1-1900", start_)
    until_ = st.sidebar.text_input("enddatum (yyyy-mm-dd)", today)
    FROM, UNTIL = check_from_until(from_, until_)
    start_month, end_month = st.sidebar.slider("Months (incl)", 1,12,(9,11))
    # Convert FROM and UNTIL to datetime
    
    where = st.sidebar.selectbox("Location to show", location_names, index=0)
    #to_show = st.sidebar.selectbox("What to show x", ["T_Max","T_Min","T_Mean","Hr_Med","Wind_Max","Wind_Mean","SLP","STN","Vis","Prec","Diary", "Feels_Like"],0)
    


    
    month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    month = month_names.index(st.sidebar.selectbox("Month", month_names, index=0)) + 1
    
    return FROM,UNTIL,start_month,end_month,where,month_names,month

@st.cache_data()
def show_open_meteo(where,locations, FROM, UNTIL,start_month,end_month):
    if start_month == 9 and end_month ==11:
        print ("Statisch bestand ivm API limiet")
        #url = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\de_bilt_ sep_nov_1996_2025.csv"
        url= "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/de_bilt_ sep_nov_1996_2025.csv"
        df_open_meteo_ = pd.read_csv(url)
        
    else:
        df_open_meteo_,_ = get_data_open_meteo(where,locations, FROM, UNTIL)


    df_open_meteo_seizoen, _ = prepare_dataframe(start_month,end_month, df_open_meteo_)
    
    df_open_meteo_seizoen_2025 = df_open_meteo_seizoen[(df_open_meteo_seizoen["YYYY"]>=2025) ]
    

    df_open_meteo_seizoen = (
        df_open_meteo_seizoen
        .groupby("YYYY", as_index=False)
        .agg({
            "temp_mean": "mean",
            "sunshine_duration": "sum",
            "rain_sum": "sum",
        })
)
    st.subheader("Open meteo")
    st.write(f"2025 Temp mean🌡️ - {df_open_meteo_seizoen_2025['temp_mean'].mean():.1f} ")
    st.write(f"2025 Zon☀️ - {df_open_meteo_seizoen_2025['sunshine_duration'].sum():.1f} ")
    st.write(f"2025 Neerslag 🌧️ - {df_open_meteo_seizoen_2025['rain_sum'].sum():.1f} ")

    st.write(f"1996-2025 Temp mean🌡️ - {df_open_meteo_seizoen['temp_mean'].mean():.1f} ")
    st.write(f"1996-2025 Zon☀️ - {df_open_meteo_seizoen['sunshine_duration'].mean():.1f} ")
    st.write(f"1996-2025 Neerslag 🌧️ - {df_open_meteo_seizoen['rain_sum'].mean():.1f} ")
    st.write(df_open_meteo_)

@st.cache_data()
def show_knmi(FROM, UNTIL,start_month,end_month):
    df_knmi, url = getdata_wrapper(260, FROM.strftime("%Y%m%d"), UNTIL.strftime("%Y%m%d"))
    
    df_knmi_seizoen = (
        df_knmi[df_knmi["MM"].between(start_month,end_month)]
        .groupby("YYYY", as_index=False)
        .agg({
            "temp_avg": "mean",
            "zonneschijnduur": "sum",
            "neerslag_etmaalsom": "sum",
        })
)
    df_knmi_seizoen_2025 = df_knmi[(df_knmi["YYYY"]==2025)]
    
    st.subheader("KNMI")
    st.write(f"2025 Temp mean🌡️ - {df_knmi_seizoen_2025['temp_avg'].mean():.1f} ")
    st.write(f"2025 Zon☀️ - {df_knmi_seizoen_2025['zonneschijnduur'].sum():.1f} ")
    st.write(f"2025 Neerslag 🌧️ - {df_knmi_seizoen_2025['neerslag_etmaalsom'].sum():.1f} ")

    st.write(f"1996-2025 Temp mean🌡️ - {df_knmi_seizoen['temp_avg'].mean():.1f} ")
    st.write(f"1996-2025 Zon☀️ - {df_knmi_seizoen['zonneschijnduur'].mean():.1f} ")
    st.write(f"1996-2025 Neerslag 🌧️ - {df_knmi_seizoen['neerslag_etmaalsom'].mean():.1f} ")
    st.write(df_knmi)
def main():
    st.info("Replicating https://x.com/HansV_16/status/1996605222899716180/photo/1")
    locations = [
         {"name":"De Bilt","lat": 52.1017011, "lon":5.1783331, "timezone": "Europe/Amsterdam"},
       
    ]
    FROM,UNTIL,start_month,end_month,where,month_names,month = interface(locations)
    
    col1,col2=st.columns(2)
    with col1:
        show_open_meteo(where,locations, FROM, UNTIL,start_month,end_month)
    with col2:
        show_knmi(FROM, UNTIL,start_month,end_month)

if __name__ == "__main__":
    #read_ogimet()
    main()
    