import pandas as pd
import streamlit as st
from datetime import datetime
import platform

from show_knmi_functions.utils import show_weerstations, help, rh2q, rh2ah, log10, list_to_text,check_from_until, find_date_for_title, download_button,get_weerstations
from show_knmi_functions.does_rain_predict_rain import does_rain_predict_rain
from show_knmi_functions.polar_plot import polar_plot
from show_knmi_functions.show_warmingstripes import show_warmingstripes
from show_knmi_functions.show_plot import show_plot
from show_knmi_functions.plot_percentiles import plot_percentiles
from show_knmi_functions.show_aantal_keren import show_aantal_keren
from show_knmi_functions.show_per_maand import show_per_maand

# INSPRIATION : https://weatherspark.com/m/52666/10/Average-Weather-in-October-in-Utrecht-Netherlands
# https://radumas.info/blog/tutorial/2017/04/17/percentile-test.html

@st.cache_data (ttl=60 * 60 * 24)
def getdata(stn, fromx, until):
    #url=r"C:\Users\rcxsm\Downloads\df_knmi_de_bilt_01011901_27072023.csv"
    #url = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\knmi_nw_beerta_no_header.csv"
    url = f"https://www.daggegevens.knmi.nl/klimatologie/daggegevens?stns={stn}&vars=TEMP:SQ:SP:Q:DR:RH:UN:UX&start={fromx}&end={until}"
    #url = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\de_bilt_1901_2023_no_header.csv"
    #url = url_local if platform.processor() else url_knmi
    #header = 0  if platform.processor() else None
    header = None
    with st.spinner(f"GETTING ALL DATA ... {url}"):

        # url =  "https://www.daggegevens.knmi.nl/klimatologie/daggegevens?stns=251&vars=TEMP&start=18210301&end=20210310"
        # https://www.knmi.nl/kennis-en-datacentrum/achtergrond/data-ophalen-vanuit-een-script
        # url = f"https://www.daggegevens.knmi.nl/klimatologie/daggegevens?stns={stn}&vars=ALL&start={fromx}&end={until}"
        
        try:
            df = pd.read_csv(
                url,
                delimiter=",",
                header=header,
                comment="#",
                low_memory=False,
            )

        except:
            st.write("FOUT BIJ HET INLADEN.")
            st.stop()
        
        # TG        : Etmaalgemiddelde temperatuur (in 0.1 graden Celsius) / Daily mean temperature in (0.1 degrees Celsius)
        # TN        : Minimum temperatuur (in 0.1 graden Celsius) / Minimum temperature (in 0.1 degrees Celsius)
        # TNH       : Uurvak waarin TN is gemeten / Hourly division in which TN was measured
        # TX        : Maximum temperatuur (in 0.1 graden Celsius) / Maximum temperature (in 0.1 degrees Celsius)
        # TXH       : Uurvak waarin TX is gemeten / Hourly division in which TX was measured
        # T10N      : Minimum temperatuur op 10 cm hoogte (in 0.1 graden Celsius) / Minimum temperature at 10 cm above surface (in 0.1 degrees Celsius)
        # T10NH     : 6-uurs tijdvak waarin T10N is gemeten / 6-hourly division in which T10N was measured; 6=0-6 UT; 12=6-12 UT; 18=12-18 UT; 24=18-24 UT
        # SQ        : Zonneschijnduur (in 0.1 uur) berekend uit de globale straling (-1 voor <0.05 uur) / Sunshine duration (in 0.1 hour) calculated from global radiation (-1 for <0.05 hour)
        # SP        : Percentage van de langst mogelijke zonneschijnduur / Percentage of maximum potential sunshine duration
        # Q         : Globale straling (in J/cm2) / Global radiation (in J/cm2)
        # DR        : Duur van de neerslag (in 0.1 uur) / Precipitation duration (in 0.1 hour)
        # RH        : Etmaalsom van de neerslag (in 0.1 mm) (-1 voor <0.05 mm) / Daily precipitation amount (in 0.1 mm) (-1 for <0.05 mm)
        # UN        : Minimale relatieve vochtigheid (in procenten)
        # UX        : Maximale relatieve vochtigheid (in procenten)
        #  0  1          2    3    4  5   6     7   8   9    10  11  12 3   4  5  16     7    8  9 20  
        # STN,YYYYMMDD,DDVEC,FHVEC,FG,FHX,FHXH,FHN,FHNH,FXX,FXXH,TG,TN,TNH,TX,TXH,T10N,T10NH,SQ,SP,Q,
        # 21 22  3   4    5   6 7  8  9   30   1    2   3   4  5  6
        # DR,RH,RHX,RHXH,PG,PX,PXH,PN,PNH,VVN,VVNH,VVX,VVXH,NG,UG,UX,UXH,UN,UNH,EV24
        column_replacements_knmi_nw_beerta = [
            [0, "STN"],
            [1, "YYYYMMDD"],
            [11, "temp_avg"],
            [12, "temp_min"],
            [14, "temp_max"],
            [16, "T10N"],
            [18, "zonneschijnduur"],
            [19, "perc_max_zonneschijnduur"],
            [20, "glob_straling"],
            [21, "neerslag_duur"],
            [22, "neerslag_etmaalsom"],
            [38, "RH_min"],
            [36, "RH_max"]
        ]
        # 0   1           2     3     4    5       6     7      8    9    100    11   12
        # STN,YYYYMMDD,   TG,   TN,   TX, T10N,   SQ,   SP,    Q,   DR,   RH,   UN,   UX
        column_replacements_knmi = [
            [0, "STN"],
            [1, "YYYYMMDD"],
            [2, "temp_avg"],
            [3, "temp_min"],
            [4, "temp_max"],
            [5, "T10N"],
            [6, "zonneschijnduur"],
            [7, "perc_max_zonneschijnduur"],
            [8, "glob_straling"],
            [9, "neerslag_duur"],
            [10, "neerslag_etmaalsom"],
            [11, "RH_min"],
            [12, "RH_max"]
        ]
        column_replacements_local = [
            [0, "ID"],
            [1, "STN"],
            [2, "YYYYMMDD"],
            [3, "temp_avg"],
            [4, "temp_min"],
            [5, "temp_max"],
            [6, "T10N"],
            [7, "zonneschijnduur"],
            [8, "perc_max_zonneschijnduur"],
            [9, "glob_straling"],
            [10, "neerslag_duur"],
            [11, "neerslag_etmaalsom"],
            [12, "RH_min"],
            [13, "RH_max"]
        ]

        #column_replacements = column_replacements_local if platform.processor() else column_replacements_knmi
        column_replacements = column_replacements_knmi
        for c in column_replacements:
            df = df.rename(columns={c[0]: c[1]})
        # if platform.processor(): 
        #     df["YYYYMMDD"] = pd.to_datetime(df["YYYYMMDD"], format="%Y-%m-%d")
        # else:
        print (df.dtypes)
        print (df)
        
        df["YYYYMMDD"] = pd.to_datetime(df["YYYYMMDD"].astype(str))
        df["YYYY"] = df["YYYYMMDD"].dt.year
        df["MM"] = df["YYYYMMDD"].dt.month
        df["DD"] = df["YYYYMMDD"].dt.day
        df["dayofyear"] = df["YYYYMMDD"].dt.dayofyear
        df["count"] = 1
        month_long_to_short = {
            "January": "Jan",
            "February": "Feb",
            "March": "Mar",
            "April": "Apr",
            "May": "May",
            "June": "Jun",
            "July": "Jul",
            "August": "Aug",
            "September": "Sep",
            "October": "Oct",
            "November": "Nov",
            "December": "Dec",
        }
        month_number_to_short = {
            "1": "Jan",
            "2": "Feb",
            "3": "Mar",
            "4": "Apr",
            "5": "May",
            "6": "Jun",
            "7": "Jul",
            "8": "Aug",
            "9": "Sep",
            "10": "Oct",
            "11": "Nov",
            "12": "Dec",
        }
        df["month"] = df["MM"].astype(str).map(month_number_to_short)
        df["year"] = df["YYYY"].astype(str)
        df["month"] = df["month"].astype(str)
        df["day"] = df["DD"].astype(str)
        df["month_year"] = df["month"] + " - " + df["year"]
        df["year_month"] = df["year"] + " - " +  df["MM"].astype(str).str.zfill(2)
        df["month_day"] = df["month"] + " - " + df["day"]
        
        to_divide_by_10 = [
            "temp_avg",
            "temp_min",
            "temp_max",
            "zonneschijnduur",
            "neerslag_duur",
            "neerslag_etmaalsom",
        ]
        
        #divide_by_10 = False if platform.processor() else True
        divide_by_10 = True
        if divide_by_10:
            for d in to_divide_by_10:
                try:
                    df[d] = df[d] / 10
                except:
                    df[d] = df[d]

    df["spec_humidity_knmi_derived"] = df.apply(lambda x: rh2q(x['RH_min'],x['temp_max'], 1020),axis=1)
    df["abs_humidity_knmi_derived"] =df.apply(lambda x: rh2ah(x['RH_min'],x['temp_max']),axis=1)
    df["globale_straling_log10"] =df.apply(lambda x: log10(x['glob_straling']),axis=1) #  np.log10(df["glob_straling"])
    if platform.processor():
        df = df[(df["YYYYMMDD"] >= fromx) & (df["YYYYMMDD"] <= until)]
    
  
    return df, url




    
def interface():
    """Kies het weerstation, de begindatum en de einddatum

    Returns:
        df, het weerstation, begindatum en einddatum (laatste drie als string)
    """
    mode = st.sidebar.selectbox(
        "Modus (kies HELP voor hulp)", ["doorlopend per dag", "aantal keren", "specifieke dag", "jaargemiddelde", "maandgemiddelde", "per dag in div jaren", "per maand in div jaren", "percentiles", "polar plot/radar chart", "does rain predict rain", "show weerstations", "help"], index=0
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

        show_parts =  st.sidebar.selectbox(
            "Show parts", [True, False], index=0
            )
        if show_parts:
            no_of_parts = st.sidebar.slider("Number of parts", 1, 10, 5)
        else:
            no_of_parts = None
        groupby_ = st.sidebar.selectbox("Groupby", [True, False], index=1)
    else:
        wdw, wdw2,sma2_how, what_to_show, gekozen_weerstation, centersmooth, graph_type,show_ci, wdw_ci,show_parts, no_of_parts, groupby_ = None,None,None,"neerslag_etmaalsom",None,None,None,None,None,None,None, None

    return stn, from_, until_, mode, groupby_, wdw, wdw2,sma2_how, what_to_show, gekozen_weerstation, centersmooth, graph_type,show_ci, wdw_ci,show_parts, no_of_parts
    


def action(stn, from_, until_, mode,groupby_, wdw, wdw2, sma2_how, what_to_show, gekozen_weerstation, centersmooth, graph_type, show_ci, wdw_ci,show_parts, no_of_parts):
    what_to_show_as_txt = list_to_text(what_to_show)
    FROM, UNTIL = check_from_until(from_, until_)

    df_getdata, url = getdata(stn, FROM.strftime("%Y%m%d"), UNTIL.strftime("%Y%m%d"))
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
        show_plot(df, datefield, title, wdw, wdw2, sma2_how, what_to_show, graph_type, centersmooth, show_ci, wdw_ci, show_parts, no_of_parts)
        #try:
        show_warmingstripes(df, title)
        # except:
        #     pass
    st.sidebar.write(f"URL to get data: {url}")
  




def main():
    stn, from_, until_, mode,groupby_, wdw, wdw2, sma2_how, what_to_show, gekozen_weerstation, centersmooth, graph_type, show_ci, wdw_ci,show_parts, no_of_parts = interface()
    action(stn, from_, until_, mode, groupby_,  wdw, wdw2,sma2_how,  what_to_show, gekozen_weerstation, centersmooth, graph_type, show_ci, wdw_ci,show_parts, no_of_parts)

if __name__ == "__main__":
    main()
