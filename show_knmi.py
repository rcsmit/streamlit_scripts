import pandas as pd
import streamlit as st
from datetime import datetime
import platform

from show_knmi_functions.utils import show_weerstations, help,  list_to_text,check_from_until, find_date_for_title, download_button,get_weerstations, get_data
from show_knmi_functions.does_rain_predict_rain import does_rain_predict_rain
from show_knmi_functions.polar_plot import polar_plot
from show_knmi_functions.show_warmingstripes import show_warmingstripes
from show_knmi_functions.show_plot import show_plot
from show_knmi_functions.plot_percentiles import plot_percentiles
from show_knmi_functions.show_aantal_keren import show_aantal_keren
from show_knmi_functions.show_per_maand import show_per_maand
from show_knmi_functions.spaghetti_plot import spaghetti_plot

# INSPRIATION : https://weatherspark.com/m/52666/10/Average-Weather-in-October-in-Utrecht-Netherlands
# https://radumas.info/blog/tutorial/2017/04/17/percentile-test.html

@st.cache_data (ttl=60 * 60 * 24)
def getdata_wrapper(stn, fromx, until):
    #url=r"C:\Users\rcxsm\Downloads\df_knmi_de_bilt_01011901_27072023.csv"
    #url = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\knmi_nw_beerta_no_header.csv"
    url = f"https://www.daggegevens.knmi.nl/klimatologie/daggegevens?stns={stn}&vars=TEMP:SQ:SP:Q:DR:RH:UN:UX&start={fromx}&end={until}"
    #url = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\de_bilt_1901_2023_no_header.csv"
    #url = url_local if platform.processor() else url_knmi
    #header = 0  if platform.processor() else None
    df = get_data(url)
    
    if platform.processor():
        df = df[(df["YYYYMMDD"] >= fromx) & (df["YYYYMMDD"] <= until)]
    return df, url


    
def interface():
    """Kies het weerstation, de begindatum en de einddatum

    Returns:
        df, het weerstation, begindatum en einddatum (laatste drie als string)
    """
    mode = st.sidebar.selectbox(
        "Modus (kies HELP voor hulp)", ["doorlopend per dag", "aantal keren", "specifieke dag", "jaargemiddelde", "maandgemiddelde", "per dag in div jaren", "spaghetti plot", "per maand in div jaren", "percentiles", "polar plot/radar chart", "does rain predict rain", "show weerstations", "help"], index=0
    )
   
    weer_stations = get_weerstations()
    weerstation_namen = []
    for w in weer_stations:
        weerstation_namen.append(w[1])
    weerstation_namen.sort()

    gekozen_weerstation = st.sidebar.selectbox(
        "Weerstation", weerstation_namen, index=4
    )
    for w in weer_stations:
        if gekozen_weerstation == w[1]:
            stn = w[0]

    DATE_FORMAT = "%m/%d/%Y"
    start_ = "2019-01-01"
    today = datetime.today().strftime("%Y-%m-%d")
    from_ = st.sidebar.text_input("startdatum (yyyy-mm-dd) from 1-1-1900", start_)
    until_ = st.sidebar.text_input("enddatum (yyyy-mm-dd)", today)

    if mode != "does rain predict rain":

        show_options = [
            "temp_min",
            "temp_avg",
            "temp_max",
            "T10N",
            "zonneschijnduur",
            "perc_max_zonneschijnduur",
            "glob_straling",
            "neerslag_duur",
            "neerslag_etmaalsom","RH_min","RH_max","spec_humidity_knmi_derived","abs_humidity_knmi_derived","globale_straling_log10",
        ]

        what_to_show = st.sidebar.multiselect("Wat weer te geven", show_options, "temp_max")
        #if len(what_to_show)==1:
        graph_type = st.sidebar.selectbox("Graph type (plotly=interactive)", ["pyplot", "plotly"], index=1)
        #else:
        #    graph_type = "pyplot"

        wdw = st.sidebar.slider("Window smoothing curves", 1, 45, 7)
        wdw2 = st.sidebar.number_input("Window smoothing curves 2 (999 for none)", 1, 999, 999)
        if wdw2 != 999:
            sma2_how = st.sidebar.selectbox("SMA2 How", ["mean", "median"], 0)
        else:
            sma2_how = None
        centersmooth =  st.sidebar.selectbox(
            "Smooth in center", [True, False], index=0
            )
        st.sidebar.write("Smoothing niet altijd aanwezig")
        show_ci =  st.sidebar.selectbox(
            "Show CI", [True, False], index=1
            )
        if show_ci:
            wdw_ci = st.sidebar.slider("Window confidence intervals", 1, 100, 20)
        else:
            wdw_ci = 1
        show_loess =  st.sidebar.selectbox(
            "Show loess", [True, False], index=1
            )
        show_parts =  st.sidebar.selectbox(
            "Show parts", [True, False], index=0
            )
        if show_parts:
            no_of_parts = st.sidebar.slider("Number of parts", 1, 10, 5)
        else:
            no_of_parts = None
        groupby_ = st.sidebar.selectbox("Groupby", [True, False], index=1)
    else:
        wdw, wdw2,sma2_how, what_to_show, gekozen_weerstation, centersmooth, graph_type,show_ci,show_loess, wdw_ci,show_parts, no_of_parts, groupby_ = None,None,None,"neerslag_etmaalsom",None,None,None,None,None,None,None, None, None

    return stn, from_, until_, mode, groupby_, wdw, wdw2,sma2_how, what_to_show, gekozen_weerstation, centersmooth, graph_type,show_ci, show_loess, wdw_ci,show_parts, no_of_parts
    
def action(stn, from_, until_, mode,groupby_, wdw, wdw2, sma2_how, what_to_show, gekozen_weerstation, centersmooth, graph_type, show_ci, show_loess, wdw_ci,show_parts, no_of_parts):
    what_to_show_as_txt = list_to_text(what_to_show)
    FROM, UNTIL = check_from_until(from_, until_)

    df_getdata, url = getdata_wrapper(stn, FROM.strftime("%Y%m%d"), UNTIL.strftime("%Y%m%d"))
    df = df_getdata.copy(deep=False)
    
    if groupby_:
        groupby_how = st.sidebar.selectbox("Groupby", ["year", "year_month"], index=1)
        groupby_what = st.sidebar.selectbox("Groupby",["sum", "mean"], index=1)
        if groupby_what == "sum":
            df = df.groupby([df[groupby_how]], sort = True).sum(numeric_only = True).reset_index()
        elif groupby_what == "mean":
            df = df.groupby([df[groupby_how]], sort = True).mean(numeric_only = True).reset_index()
        datefield = groupby_how
    else:
        datefield = "YYYYMMDD"
    with st.expander("Dataframe"):
        st.write(df)
        download_button(df)
    
    if mode == "help":
        help()
    elif mode == "does rain predict rain":
        does_rain_predict_rain(df)
    elif mode == "show weerstations":
        show_weerstations()

    elif mode == "per dag in div jaren":
        show_per_maand(df, gekozen_weerstation, what_to_show, "per_dag", graph_type)
        datefield = None
        title = f"{what_to_show_as_txt} van {from_} - {until_} in {gekozen_weerstation}"
    elif mode == "spaghetti plot":
        wdw = st.sidebar.number_input("window for smoothing the 95% interval",1,len(df),9)
        spaghetti_plot(df, what_to_show, wdw)
    elif mode == "per maand in div jaren":
        show_per_maand(df, gekozen_weerstation, what_to_show, "maandgem", graph_type)
        datefield = None
        title = f"{what_to_show_as_txt} van {from_} - {until_} in {gekozen_weerstation}"
    elif mode == "aantal keren":
        show_aantal_keren(df, gekozen_weerstation, what_to_show)
    elif mode == "percentiles":
        plot_percentiles(df,  gekozen_weerstation, what_to_show, wdw, centersmooth)
    elif mode == "polar plot/radar chart":
        how = st.sidebar.selectbox(
            "Scatter / line", ["scatter", "line"], index=0
            )
        polar_plot(df,  what_to_show, how)
        
    else:
        if mode == "doorlopend per dag":
            # datefield = "YYYYMMDD"
            title = f"{what_to_show_as_txt} van {from_} - {until_} in {gekozen_weerstation}"
            #graph_type = "plotly"
            # graph_type = "pyplot" #too slow

        else:
            #graph_type = "pyplot"
            datefield = "YYYY"
            if mode == "jaargemiddelde":

                #graph_type = "plotly"
                df = df.groupby(["YYYY"], sort=True).mean(numeric_only = True).reset_index()
                st.write(df)
                title = f"Jaargemiddelde {what_to_show_as_txt}  van {from_[:4]} - {until_[:4]} in {gekozen_weerstation}"
                st.sidebar.write(
                    "Zorg ervoor dat de einddatum op 31 december valt voor het beste resultaat "
                )

            elif mode == "maandgemiddelde":
                
                month_min = st.sidebar.number_input("Maand",1,12,6)
                month_max = st.sidebar.number_input("Maand",1,12,8)
                if month_min > month_max:
                    st.error("Max must be higher or equal than min")
                    st.stop()

                df = df[(df["MM"] >= month_min) & (df["MM"] <= month_max)].reset_index().copy(deep=True)
                df = df.groupby(by=["year"]).mean(numeric_only=True).reset_index() # werkt maar geeft geen 0 waardes weer 
                title = f"Maandgemiddelde (maand = {month_min}-{month_max}) - {what_to_show} in {gekozen_weerstation}"
               
            elif mode == "specifieke dag":
                #graph_type = "plotly"
                day = st.sidebar.number_input("Dag", 1, 31, 1, None, format="%i")
                months = [
                    "januari",
                    "februari",
                    "maart",
                    "april",
                    "mei",
                    "juni",
                    "juli",
                    "augustus",
                    "september",
                    "oktober",
                    "november",
                    "december",
                ]
                month = months.index(st.sidebar.selectbox("Maand", months, index=0)) + 1
                df = df[
                    (df["YYYYMMDD"].dt.month == month) & (df["YYYYMMDD"].dt.day == day)
                ]
                title = f"{what_to_show_as_txt}  op {find_date_for_title(day,month)} van {from_[:4]} - {until_[:4]} in {gekozen_weerstation}"
                st.sidebar.write(
                    "Zorg ervoor dat de datum in de gekozen tijdrange valt voor het beste resultaat "
                )
        show_plot(df, datefield, title, wdw, wdw2, sma2_how, what_to_show, graph_type, centersmooth, show_ci, show_loess, wdw_ci, show_parts, no_of_parts)
        
        show_warmingstripes(df, title)

    st.sidebar.write(f"URL to get data: {url}")

def main():
    stn, from_, until_, mode,groupby_, wdw, wdw2, sma2_how, what_to_show, gekozen_weerstation, centersmooth, graph_type, show_ci, show_loess, wdw_ci,show_parts, no_of_parts = interface()
    action(stn, from_, until_, mode, groupby_,  wdw, wdw2,sma2_how,  what_to_show, gekozen_weerstation, centersmooth, graph_type, show_ci, show_loess,wdw_ci,show_parts, no_of_parts)

if __name__ == "__main__":
    main()
