from imghdr import what
import pandas as pd
import numpy as np

import streamlit as st
from streamlit import caching
import datetime as dt

from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg

_lock = RendererAgg.lock
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.dates as mdates

import plotly.express as px
import plotly.graph_objects as go


# INSPRIATION : https://weatherspark.com/m/52666/10/Average-Weather-in-October-in-Utrecht-Netherlands
# https://radumas.info/blog/tutorial/2017/04/17/percentile-test.html
def select_period_oud(df, field, show_from, show_until):
    """Shows two inputfields (from/until and Select a period in a df (helpers.py).

    Args:
        df (df): dataframe
        field (string): Field containing the date

    Returns:
        df: filtered dataframe
    """

    if show_from is None:
        show_from = "2020-01-01"

    if show_until is None:
        show_until = "2030-01-01"
    # "Date_statistics"
    mask = (df[field].dt.date >= show_from) & (df[field].dt.date <= show_until)
    df = df.loc[mask]
    df = df.reset_index()
    return df


@st.cache(ttl=60 * 60 * 24, suppress_st_warning=True)
def getdata(stn, fromx, until):
    with st.spinner(f"GETTING ALL DATA ..."):
        # url =  "https://www.daggegevens.knmi.nl/klimatologie/daggegevens?stns=251&vars=TEMP&start=18210301&end=20210310"
        # https://www.knmi.nl/kennis-en-datacentrum/achtergrond/data-ophalen-vanuit-een-script
        # url = f"https://www.daggegevens.knmi.nl/klimatologie/daggegevens?stns={stn}&vars=ALL&start={fromx}&end={until}"
        url = f"https://www.daggegevens.knmi.nl/klimatologie/daggegevens?stns={stn}&vars=TEMP:SQ:SP:Q:DR:RH&start={fromx}&end={until}"
        try:
            df = pd.read_csv(
                url,
                delimiter=",",
                header=None,
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

        column_replacements = [
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
        ]

        for c in column_replacements:
            df = df.rename(columns={c[0]: c[1]})

        df["YYYYMMDD"] = pd.to_datetime(df["YYYYMMDD"], format="%Y%m%d")
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
        df["month_day"] = df["month"] + " - " + df["day"]
        
        to_divide_by_10 = [
            "temp_avg",
            "temp_min",
            "temp_max",
            "zonneschijnduur",
            "neerslag_duur",
            "neerslag_etmaalsom",
        ]
        for d in to_divide_by_10:
            try:
                df[d] = df[d] / 10
            except:
                df[d] = None

    download_button(df)
    return df, url

def download_button(df):    
    csv = convert_df(df)

    st.sidebar.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='df_knmi.csv',
        mime='text/csv',
    )

@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

def show_aantal_kerend(df, gekozen_weerstation, what_to_show_):

    what_to_show_ = what_to_show_ if type(what_to_show_) == list else [what_to_show_]

    df.set_index("YYYYMMDD")
    (month_min,month_max) = st.sidebar.slider("Maanden (van/tot en met)", 1, 12, (1,12))

    (value_min,value_max) = st.sidebar.slider("Waarde (van/tot en met)", -99, 99, (0,99))


    jaren = df["YYYY"].tolist()
    for what_to_show in what_to_show_:
        df = df[(df["MM"] >= month_min) & (df["MM"] <= month_max)]
        df = df[(df[what_to_show] >= value_min) & (df[what_to_show] <= value_max)]

        df_grouped = df.groupby(by=["year"]).sum().reset_index()
        df_grouped = df_grouped[["year", "count"]]

        fig, ax = plt.subplots()
        plt.title(f"Aantal keren dat { what_to_show} in {gekozen_weerstation} tussen {value_min} en {value_max} ligt")
        plt.bar(df_grouped["year"], df_grouped["count"])
        plt.grid()
        plt.xticks(rotation=270)
        st.pyplot(fig)
        st.write(df_grouped)
        st.write(df)

def show_per_maand(df, gekozen_weerstation, what_to_show_, groeperen, graph_type):
    what_to_show_ = what_to_show_ if type(what_to_show_) == list else [what_to_show_]
    df.set_index("YYYYMMDD")
    (month_min,month_max) = st.sidebar.slider("Maanden (van/tot en met)", 1, 12, (1,12))

    jaren = df["YYYY"].tolist()
    df = df[(df["MM"] >= month_min) & (df["MM"] <= month_max)]
    df['DD'] = df['DD'].astype(str).str.zfill(2)
    df['MM'] = df['MM'].astype(str).str.zfill(2)
    df["mmdd"] = df["MM"] +"-" + df["DD"]

    df["year"] = df["year"].astype(str)
            
    for what_to_show in what_to_show_:
        if groeperen == "maandgem":
            df_grouped = df.groupby(["year", "MM"]).mean().reset_index()
     
            df_grouped ["year"] = df_grouped ["year"].astype(str)
     
            df_pivoted = df_grouped.pivot(
                index="MM", columns="year", values=what_to_show
            ).reset_index()
        elif groeperen == "per_dag":
            df["MD"] = df["month_day"]
            df_grouped = df.groupby(["year", "mmdd"]).mean().reset_index()
            df_grouped ["year"] = df_grouped ["year"].astype(str)
            df_pivoted = df_grouped.pivot(                index="mmdd", columns="year", values=what_to_show            ).reset_index()
        
    
        if graph_type == "plotly":
            fig = go.Figure()
            #df["sma"] = df_pivoted[what_to_show].rolling(window=wdw, center=centersmooth).mean()
            #st.write(df_pivoted.columns)
            # for c in df_pivoted.columns:
            #     print (c)
            #     print ("linne 224")
            #     print (pd.Series(df_pivoted.index.values))
            if groeperen == "maandgem":
                sma = [go.Scatter(x=pd.Series(df_pivoted.MM), y=df_pivoted[c],  
                   mode='lines', name=f'{c}')
                   for c in df_pivoted.columns[1:]]
            else:
                sma = [go.Scatter(x=[pd.Series(df_pivoted.index.values),df_pivoted.mmdd], y=df_pivoted[c],  
                   mode='lines',  line=dict(width=.7), name=f'{c}')
                    for c in df_pivoted.columns[1:]]
        

            data = sma
            title = (f"{ what_to_show} -  {gekozen_weerstation}")
            layout = go.Layout(
                # xaxis=dict(label=df_pivoted["mm-dd"]),
                yaxis=dict(title=what_to_show), 
                title=title,)
                #, xaxis=dict(tickformat="%d-%m")
            fig = go.Figure(data=data, layout=layout)
            #fig.update_layout(xaxis=dict(tickformat="%m-%d"))
            st.plotly_chart(fig, use_container_width=True)
        
        
        
        else:
            st.warning ("Under construction")

            # fig, ax = plt.subplots()
            # plt.title(f"{ what_to_show} - gemiddeld per maand in {gekozen_weerstation}")

           
            # if groeperen == "per_dag":
            #     major_format = mdates.DateFormatter("%b")
            #     ax.xaxis.set_major_formatter(major_format)
            # plt.grid()
            # ax.plot(df_pivoted)
            # plt.legend(df_pivoted.columns, title=df_pivoted.columns.name)

            # st.pyplot(fig)
            # st.subheader(f"Data of {what_to_show}")
            # st.write(df_pivoted)


def get_weerstations():
    weer_stations = [
        [209, "IJmond"],
        [210, "Valkenburg Zh"],
        [215, "Voorschoten"],
        [225, "IJmuiden"],
        [235, "De Kooy"],
        [240, "Schiphol"],
        [242, "Vlieland"],
        [248, "Wijdenes"],
        [249, "Berkhout"],
        [251, "Hoorn Terschelling"],
        [257, "Wijk aan Zee"],
        [258, "Houtribdijk"],
        [260, "De Bilt"],
        [265, "Soesterberg"],
        [267, "Stavoren"],
        [269, "Lelystad"],
        [270, "Leeuwarden"],
        [273, "Marknesse"],
        [275, "Deelen"],
        [277, "Lauwersoog"],
        [278, "Heino"],
        [279, "Hoogeveen"],
        [280, "Eelde"],
        [283, "Hupsel"],
        [285, "Huibertgat"],
        [286, "Nieuw Beerta"],
        [290, "Twenthe"],
        [308, "Cadzand"],
        [310, "Vlissingen"],
        [311, "Hoofdplaat"],
        [312, "Oosterschelde"],
        [313, "Vlakte van De Raan"],
        [315, "Hansweert"],
        [316, "Schaar"],
        [319, "Westdorpe"],
        [323, "Wilhelminadorp"],
        [324, "Stavenisse"],
        [330, "Hoek van Holland"],
        [331, "Tholen"],
        [340, "Woensdrecht"],
        [343, "Rotterdam Geulhaven"],
        [344, "Rotterdam"],
        [348, "Cabauw Mast"],
        [350, "Gilze-Rijen"],
        [356, "Herwijnen"],
        [370, "Eindhoven"],
        [375, "Volkel"],
        [377, "Ell"],
        [380, "Maastricht"],
        [391, "Arcen"],
    ]
    return weer_stations


def interface():
    """Kies het weerstation, de begindatum en de einddatum

    Returns:
        df, het weerstation, begindatum en einddatum (laatste drie als string)
    """
    mode = st.sidebar.selectbox(
        "Modus (kies HELP voor hulp)", ["doorlopend per dag", "aantal keren", "specifieke dag", "jaargemiddelde", "per dag in div jaren", "per maand in div jaren", "percentiles", "show weerstations", "help"], index=0
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

    

    show_options = [
        "temp_min",
        "temp_avg",
        "temp_max",
        "T10N",
        "zonneschijnduur",
        "perc_max_zonneschijnduur",
        "glob_straling",
        "neerslag_duur",
        "neerslag_etmaalsom",
    ]
    what_to_show = st.sidebar.multiselect("Wat weer te geven", show_options, "temp_max")
    #if len(what_to_show)==1:
    graph_type = st.sidebar.selectbox("Graph type (plotly=interactive)", ["pyplot", "plotly"], index=1)
    #else:
    #    graph_type = "pyplot"

    wdw = st.sidebar.slider("Window smoothing curves", 1, 45, 7)
    centersmooth =  st.sidebar.selectbox(
        "Smooth in center", [True, False], index=0
        )
    st.sidebar.write("Smoothing niet aanwezig bij alle niet altijd")
    return stn, from_, until_, mode, wdw, what_to_show, gekozen_weerstation, centersmooth, graph_type
    
def check_from_until(from_, until_):
    """Checks whethe start- and enddate are valid.

    Args:
        from_ (string): start date
        until_ (string): end date

    Returns:
        FROM, UNTIL : start- and end date in datetime

    """
    try:
        FROM = dt.datetime.strptime(from_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the startdate is in format yyyy-mm-dd")
        st.stop()

    try:
        UNTIL = dt.datetime.strptime(until_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the enddate is in format yyyy-mm-dd")
        st.stop()

    if FROM >= UNTIL:
        st.warning("Make sure that the end date is not before the start date")
        st.stop()

    return FROM, UNTIL

def list_to_text(what_to_show_):
    """converts list to text to use in plottitle

    Args:
        what_to_show_ (list with strings): list with the fields

    Returns:
        string: text to use in plottitle
    """
    what_to_show_ = what_to_show_ if type(what_to_show_) == list else [what_to_show_]
    w = ""
    for w_ in what_to_show_:
        if w_ == what_to_show_[-1]:
            w += w_
        elif w_ == what_to_show_[-2]:
            w += w_ + " & "
        else:
            w += w_ + ", "

    return w

def action(stn, from_, until_, mode, wdw, what_to_show, gekozen_weerstation, centersmooth, graph_type):
    what_to_show_as_txt = list_to_text(what_to_show)
    FROM, UNTIL = check_from_until(from_, until_)

    df_getdata, url = getdata(stn, FROM.strftime("%Y%m%d"), UNTIL.strftime("%Y%m%d"))

    df = df_getdata.copy(deep=False)
    if mode == "help":
        help()
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
            show_aantal_kerend(df, gekozen_weerstation, what_to_show)
    elif mode == "percentiles":
        plot_percentiles(df,  gekozen_weerstation, what_to_show, wdw, centersmooth)
    else:
        if mode == "doorlopend per dag":
            datefield = "YYYYMMDD"
            title = f"{what_to_show_as_txt} van {from_} - {until_} in {gekozen_weerstation}"
            #graph_type = "plotly"
            # graph_type = "pyplot" #too slow

        else:
            #graph_type = "pyplot"
            datefield = "YYYY"
            if mode == "jaargemiddelde":

                #graph_type = "plotly"
                df = df.groupby(["YYYY"], sort=True).mean().reset_index()

                title = f"Jaargemiddelde {what_to_show_as_txt}  van {from_[:4]} - {until_[:4]} in {gekozen_weerstation}"
                st.sidebar.write(
                    "Zorg ervoor dat de einddatum op 31 december valt voor het beste resultaat "
                )
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
        show_plot(df, datefield, title, wdw, what_to_show, graph_type, centersmooth)

        show_warmingstripes(df, title)
    st.sidebar.write(f"URL to get data: {url}")
  

def plot_percentiles(df, gekozen_weerstation, what_to_show, wdw, centersmooth):
    if len(what_to_show)!=1 :
        st.warning("Choose (only) 1 thing to show")
        st.stop()

    df_quantile = pd.DataFrame(
        {"date": [],  "q10": [], "q25": [], "q50":[] ,"avg": [], "q75": [], "q90": []}    )
    year_to_show = st.sidebar.number_input("Year to show (2100 for nothing)", 1900, 2100, 2021)

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


                    df_quantile = df_quantile.append(
                        {
                            "date_": date_,
                            "q10": q10,
                            "q25": q25,
                            "q50": q50,
                            "avg": avg,

                            "q75": q75,

                            "q90": q90,
                            "value_in_year" : value_in_year
                            },
                        ignore_index=True,
                    )

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
            name="2021",
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
def show_plot(df, datefield, title, wdw, what_to_show_, graph_type, centersmooth):
    what_to_show_ = what_to_show_ if type(what_to_show_) == list else [what_to_show_]
    color_list = [
        "#02A6A8",
        "#4E9148",
        "#F05225",
        "#024754",
        "#FBAA27",
        "#302823",
        "#F07826",
        "#ff6666",
    ]
    if len(df) == 1 and datefield == "YYYY":
        st.warning("Selecteer een grotere tijdsperiode")
        st.stop()

    if graph_type=="pyplot"  :
        with _lock:
            fig1x = plt.figure()
            ax = fig1x.add_subplot(111)
            for i, what_to_show in enumerate(what_to_show_):
                sma = df[what_to_show].rolling(window=wdw, center=centersmooth).mean()
                ax = df[what_to_show].plot(
                    label="_nolegend_",
                    linestyle="dotted",
                    color=color_list[i],
                    linewidth=0.5,
                )
                ax = sma.plot(label=what_to_show, color=color_list[i], linewidth=0.75)
            
            #ax.set_xticks(df[datefield]) #TOFIX : this gives an strange graph
            if datefield == "YYYY":
                ax.set_xticklabels(df[datefield], fontsize=6, rotation=90)
            else:
                ax.set_xticklabels(df[datefield].dt.date, fontsize=6, rotation=90)
            xticks = ax.xaxis.get_major_ticks()
            for i, tick in enumerate(xticks):
                if i % 10 != 0:
                    tick.label1.set_visible(False)

            plt.xticks()
            plt.grid(which="major", axis="y")
            plt.title(title)
            plt.legend()
            st.pyplot(fig1x)
    else:
        for what_to_show_x in what_to_show_:
            fig = go.Figure()
            df["sma"] = df[what_to_show_x].rolling(window=wdw, center=centersmooth).mean()

            sma = go.Scatter(
                name=what_to_show_x,
                x=df[datefield],
                y= df["sma"],
                mode='lines',
                line=dict(width=1,color='rgba(0, 0, 168, 0.8)'),
                )
            points = go.Scatter(
                name="",
                x=df[datefield],
                y= df[what_to_show_x],
                mode='markers',
                showlegend=False,marker=dict(
                color='LightSkyBlue',
                size=2))


            data = [sma,points]

            layout = go.Layout(
                yaxis=dict(title=what_to_show_x),
                title=title,)
                #, xaxis=dict(tickformat="%d-%m")
            fig = go.Figure(data=data, layout=layout)
            fig.update_layout(xaxis=dict(tickformat="%d-%m-%Y"))
            st.plotly_chart(fig, use_container_width=True)
    df =df[[datefield,what_to_show_[0]]]
    st.write(df)

def find_date_for_title(day, month):
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
    # ["January", "February",  "March", "April", "May", "June", "July", "August", "September", "Oktober", "November", "December"]
    return str(day) + " " + months[month - 1]


def show_warmingstripes(df, title):
    # Based on code of Sebastian Beyer
    # https://github.com/sebastianbeyer/warmingstripes/blob/master/warmingstripes.py

    # the colors in this colormap come from http://colorbrewer2.org
    # the 8 more saturated colors from the 9 blues / 9 reds
    # https://matplotlib.org/matplotblog/posts/warming-stripes/
    cmap = ListedColormap(
        [
            "#08306b",
            "#08519c",
            "#2171b5",
            "#4292c6",
            "#6baed6",
            "#9ecae1",
            "#c6dbef",
            "#deebf7",
            "#fee0d2",
            "#fcbba1",
            "#fc9272",
            "#fb6a4a",
            "#ef3b2c",
            "#cb181d",
            "#a50f15",
            "#67000d",
        ]
    )
    # https://github.com/sebastianbeyer/warmingstripes/blob/master/warmingstripes.py
    temperatures = df["temp_avg"].tolist()
    stacked_temps = np.stack((temperatures, temperatures))
    with _lock:
        # plt.figure(figsize=(4,18))
        fig, ax = plt.subplots()

        fig = ax.imshow(
            stacked_temps,
            cmap=cmap,
            aspect=40,
        )
        plt.gca().set_axis_off()

        plt.title(title)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.show()
        # st.pyplot(fig) - gives an error
        st.set_option("deprecation.showPyplotGlobalUse", False)
        st.pyplot()

def show_weerstations():
    MAPBOX = "pk.eyJ1IjoicmNzbWl0IiwiYSI6Ii1IeExqOGcifQ.EB6Xcz9f-ZCzd5eQMwSKLQ"
    # original_Name
    df_map=  pd.read_csv(
        "img_knmi/weerstations.csv",
        #"img_knmi/leeg.csv",
        comment="#",
        delimiter=",",
        low_memory=False,
    )
    df_map = df_map[["original_Name", "lat", "lon"]]
    #df_map = df_map.head(1)
    
    #st.map(data=df_map, zoom=None, use_container_width=True)

    # Adding code so we can have map default to the center of the data
    midpoint = (np.average(df_map['lat']), np.average(df_map['lon']))
    import pydeck as pdk
    tooltip = {
            "html":
                "{original_Name} <br/>"
            }
        
    layer1= pdk.Layer(
            'ScatterplotLayer',     # Change the `type` positional argument here
                df_map,
                get_position=['lon', 'lat'],
                auto_highlight=True,
                get_radius=4000,          # Radius is given in meters
                get_fill_color=[180, 0, 200, 140],  # Set an RGBA value for fill
                pickable=True)
    layer2 =  pdk.Layer(
                    type="TextLayer",
                    data=df_map,
                    pickable=False,
                    get_position=["lon", "lat"],
                    get_text="original_Name",
                    get_color=[0, 0, 0],
                    get_angle=0,
                    sizeScale= 0.5,
                    # Note that string constants in pydeck are explicitly passed as strings
                    # This distinguishes them from columns in a data set
                    getTextAnchor= '"middle"',
                    get_alignment_baseline='"bottom"'
                )

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
             longitude=midpoint[1],
            latitude=midpoint[0],
            pitch=0,
            zoom=6,
        ),
        layers=[layer1, layer2
            
        ],tooltip = tooltip
    ))


    st.write(df_map)

    st.sidebar.write("Link to map with KNMI stations on Google Maps https://www.google.com/maps/d/u/0/edit?mid=1ePEzqJ4_aNyyTwF5FyUM6XiqhLZPSBjN&ll=52.17534745851063%2C5.197922250000001&z=7")



def help():
    st.header("Help")
    st.write ("Hier zijn de verschillende mogelijkheden")
    st.subheader("Doorlopend per dag")
    st.write("Wat was de temperatuur in de loop van de tijd?")
    st.image("img_knmi/doorlopend_per_dag.png")

    st.subheader("Aantal keren")
    st.write("Hoeveel tropische dagen hebben we gehad in een bepaaalde periode?")
    st.image("img_knmi/aantal_keren.png")

    st.subheader("Specifieke dag")
    st.write("Welke temperatuur was het op nieuwjaarsdag door de loop van de tijd?")
    st.image("img_knmi/specifieke_dag.png")

    st.subheader("Jaargemiddelde")
    st.write("Wat was het jaargemiddelde?")
    st.image("img_knmi/jaargemiddelde.png")
    st.write("Kies hier volledige jaren als periode")

    st.subheader("Per dag in div jaren")
    st.write("Kan ik 2021 met 2021 per dag vergelijken?")
    st.image("img_knmi/per_dag_div_jaren_2020_2021.png")


    st.subheader("Per maand in diverse jaren")
    st.write("Kan ik 2021 met 2021 per maaand vergelijken?")
    st.image("img_knmi/per_maand_div_jaren_2020_2021.png")




    st.subheader("Percentiles")
    st.write("Wat zijn de uitschieters in het jaar? - kies hiervoor een lange periode")
    st.image("img_knmi/percentiles.png")
    st.subheader("Weerstations")
    st.write("Link to map with KNMI stations https://www.google.com/maps/d/u/0/edit?mid=1ePEzqJ4_aNyyTwF5FyUM6XiqhLZPSBjN&ll=52.17534745851063%2C5.197922250000001&z=7")
    st.image("img_knmi/weerstations.png")

                
    st.image(
            "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/buymeacoffee.png"
        )

    st.markdown(
        '<a href="https://www.buymeacoffee.com/rcsmit" target="_blank">If you are happy with this dashboard, you can buy me a coffee</a>',
        unsafe_allow_html=True,
    )


def main():
    stn, from_, until_, mode, wdw, what_to_show, gekozen_weerstation, centersmooth, graph_type = interface()
    action(stn, from_, until_, mode, wdw, what_to_show, gekozen_weerstation, centersmooth, graph_type)


if __name__ == "__main__":
    main()
