from imghdr import what
import pandas as pd

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


@st.cache(ttl=60 * 60 * 24)
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
            df[d] = df[d] / 10

    return df, url


def show_per_maand(df, gekozen_weerstation, what_to_show_):
    what_to_show_ = what_to_show_ if type(what_to_show_) == list else [what_to_show_]

    df.set_index("YYYYMMDD")
    month_min = st.sidebar.number_input("Beginmaand (van)", 1, 12, 1, None, format="%i")
    month_max = st.sidebar.number_input(
        "Eindmaand (tot en met)", 1, 12, 12, None, format="%i"
    )
    groeperen = st.sidebar.selectbox(
        "Per dag/maandgemiddelde", ["per_dag", "maandgem"], index=1
    )

    jaren = df["YYYY"].tolist()
    df = df[(df["MM"] >= month_min)]
    df = df[(df["MM"] <= month_max)]
    for what_to_show in what_to_show_:

        fig, ax = plt.subplots()
        plt.title(f"{ what_to_show} - gemiddeld per maand in {gekozen_weerstation}")

        if groeperen == "maandgem":
            df_grouped = df.groupby(["year", "month"]).mean()
            df_pivoted = df_grouped.pivot(
                index="MM", columns="YYYY", values=what_to_show
            )
        elif groeperen == "per_dag":
            df["MD"] = df["month_day"]
            df_grouped = df.groupby(["year", "month_day"]).mean()
            df_pivoted = df_grouped.pivot(
                index="dayofyear", columns="YYYY", values=what_to_show
            )
            major_format = mdates.DateFormatter("%b")

            ax.xaxis.set_major_formatter(major_format)
        plt.grid()
        ax.plot(df_pivoted)
        plt.legend(df_pivoted.columns, title=df_pivoted.columns.name)

        st.pyplot(fig)
        st.subheader(f"Data of {what_to_show}")
        st.write(df_pivoted)


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
    from_ = st.sidebar.text_input("startdatum (yyyy-mm-dd)", start_)
    until_ = st.sidebar.text_input("enddatum (yyyy-mm-dd)", today)

    mode = st.sidebar.selectbox(
        "Modus", ["per dag", "specifieke dag", "jaargemiddelde", "per maand"], index=3
    )
    wdw = st.sidebar.slider("Window smoothing curves", 1, 45, 7)

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
    what_to_show = st.sidebar.multiselect("Wat weer te geven", show_options, "temp_avg")

    return stn, from_, until_, mode, wdw, what_to_show, gekozen_weerstation


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

def action(stn, from_, until_, mode, wdw, what_to_show, gekozen_weerstation):
    what_to_show_as_txt = list_to_text(what_to_show)
    FROM, UNTIL = check_from_until(from_, until_)

    df_getdata, url = getdata(stn, FROM.strftime("%Y%m%d"), UNTIL.strftime("%Y%m%d"))

    df = df_getdata.copy(deep=False)

    if mode == "per maand":
        show_per_maand(df, gekozen_weerstation, what_to_show)
        datefield = None
        title = f"{what_to_show_as_txt} van {from_} - {until_} in {gekozen_weerstation}"
    else:
        if mode == "per dag":
            datefield = "YYYYMMDD"
            title = f"{what_to_show_as_txt} van {from_} - {until_} in {gekozen_weerstation}"
        else:
            datefield = "YYYY"
            if mode == "jaargemiddelde":
                df = df.groupby(["YYYY"], sort=True).mean().reset_index()

                title = f"{what_to_show_as_txt}  van {from_[:4]} - {until_[:4]} in {gekozen_weerstation}"
                st.sidebar.write(
                    "Zorg ervoor dat de einddatum op 31 december valt voor het beste resultaat "
                )
            elif mode == "specifieke dag":
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
        show_plot(df, datefield, title, wdw, what_to_show)

    show_warmingstripes(df, title)
    st.sidebar.write(f"URL to get data: {url}")


def show_plot(df, datefield, title, wdw, what_to_show_):
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
    with _lock:
        fig1x = plt.figure()
        ax = fig1x.add_subplot(111)
        for i, what_to_show in enumerate(what_to_show_):
            sma = df[what_to_show].rolling(window=wdw, center=True).mean()
            ax = df[what_to_show].plot(
                label="_nolegend_",
                linestyle="dotted",
                color=color_list[i],
                linewidth=0.5,
            )
            ax = sma.plot(label=what_to_show, color=color_list[i], linewidth=0.75)

        ax.set_xticks(df[datefield].index)
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


def main():
    stn, from_, until_, mode, wdw, what_to_show, gekozen_weerstation = interface()
    action(stn, from_, until_, mode, wdw, what_to_show, gekozen_weerstation)


if __name__ == "__main__":
    main()
