from os import supports_follow_symlinks
import pandas as pd

import streamlit as st
from streamlit import caching
import datetime as dt

from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

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
    #"Date_statistics"
    mask = (df[field].dt.date >= show_from) & (df[field].dt.date <= show_until)
    df = df.loc[mask]
    df = df.reset_index()
    return df

@st.cache(ttl=60 * 60 * 24)
def getdata(stn, fromx, until):
    url =  "https://www.daggegevens.knmi.nl/klimatologie/daggegevens?stns=251&vars=ALL&start=20210301&end=20210310"

    url = f"https://www.daggegevens.knmi.nl/klimatologie/daggegevens?stns={stn}&vars=ALL&start={fromx}&end={until}"
    try:
        df = pd.read_csv(url, delimiter=",", header=None,  comment="#",low_memory=False,)
    except:
        st.write("ERROR")
        st.stop()

    column_replacements = [[0, 'STN'],
                        [1, 'YYYYMMDD'],
                        [2, 'DDVEC'],
                        [3, 'FHVEC'],
                        [4, 'FG'],
                        [5, 'FHX'],
                        [6, 'FHXH'],
                        [7, 'FHN'],
                        [8, 'FHNH'],
                        [9, 'FXX'],
                        [10, 'FXXH'],
                        [11, 'TG'],
                        [12, 'TN'],
                        [13, 'TNH'],
                        [14, 'TX'],
                        [15, 'TXH'],
                        [16, 'T10N'],
                        [17, 'T10NH'],
                        [18, 'SQ'],
                        [19, 'SP'],
                        [20, 'Q'],
                        [21, 'DR'],
                        [22, 'RH'],
                        [23, 'RHX'],
                        [24, 'RHXH'],
                        [25, 'PG'],
                        [26, 'PX'],
                        [27, 'PXH'],
                        [28, 'PN'],
                        [29, 'PNH'],
                        [30, 'VVN'],
                        [31, 'VVNH'],
                        [32, 'VVX'],
                        [33, 'VVXH'],
                        [34, 'NG'],
                        [35, 'UG'],
                        [36, 'UX'],
                        [37, 'UXH'],
                        [38, 'UN'],
                        [39, 'UNH'],
                        [40, 'EV24']]

    for c in column_replacements:
        df = df.rename(columns={c[0]:c[1]})

    df["YYYYMMDD"] = pd.to_datetime(df["YYYYMMDD"], format="%Y%m%d")
    to_divide_by_10 = ["TG", "TX", "TN"]
    for d in to_divide_by_10:
        df[d] = df[d]/10

    return df

def interface():

    weer_stations = [[209, "IJmond"],
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
                    [391, "Arcen"]]
    weerstation_namen = []
    for w in weer_stations:
        weerstation_namen.append(w[1])
    weerstation_namen.sort()

    gekozen_weerstation = st.sidebar.selectbox("Weerstation", weerstation_namen, index=4)
    for w in weer_stations:
        if gekozen_weerstation == w[1]:
            stn = w[0]




    DATE_FORMAT = "%m/%d/%Y"
    start_ = "2021-01-01"
    today = datetime.today().strftime("%Y-%m-%d")
    from_ = st.sidebar.text_input("startdate (yyyy-mm-dd)", start_)

    try:
        FROM = dt.datetime.strptime(from_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the startdate is in format yyyy-mm-dd")
        st.stop()

    until_ = st.sidebar.text_input("enddate (yyyy-mm-dd)", today)

    try:
        UNTIL = dt.datetime.strptime(until_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the enddate is in format yyyy-mm-dd")
        st.stop()

    if FROM >= UNTIL:
        st.warning("Make sure that the end date is not before the start date")
        st.stop()

    df_getdata = getdata(stn, FROM.strftime("%Y%m%d"), UNTIL.strftime("%Y%m%d"))

    df = df_getdata.copy(deep=False)
    #st.write(df)
    return df, gekozen_weerstation, from_, until_

def show_plot(df, gekozen_weerstation, from_, until_):
    to_show = [["TG", "Gem. dagtemp", "g"],
                ["TN", "Min. dagtemp", "b"],
                ["TX", "Max. dagtemp", "r"],
                ]

    with _lock:
        fig1x = plt.figure()
        ax = fig1x.add_subplot(111)

        for show in to_show:
            sma = df[show[0]].rolling(window=7, center=True).mean()
            ax = df[show[0]].plot( label = "_nolegend_",  linestyle="dotted",color = show[2],  linewidth = 0.5)
            ax = sma.plot( label = show[1], color = show[2], linewidth = 0.75)

        ax.set_xticks(df["YYYYMMDD"].index)
        ax.set_xticklabels(df["YYYYMMDD"].dt.date, fontsize=6, rotation=90)
        xticks = ax.xaxis.get_major_ticks()
        for i, tick in enumerate(xticks):
                if i % 10 != 0:
                    tick.label1.set_visible(False)

        plt.xticks()
        plt.grid(which='major', axis = 'y')
        #matplotlib.pyplot.grid(b=None, which='major', axis='both', **kwargs)
        plt.title (f"Dagtemperatuur van {from_} - {until_} in {gekozen_weerstation}")
        plt.legend()
        st.pyplot(fig1x)

def main():
    df, gekozen_weerstation, from_, until_ = interface()
    #print (df)
    show_plot (df, gekozen_weerstation, from_, until_)
main()

# TO DO :

# OPTIES TOEVEOGEN
# * gemiddelde jaar tempeatuur over een periode
# * temperatuur op een bepaalde dag/ ve bepaalde maand door de tijd heen


# if mode == "year_avg":
#     plt.ylabel("Average Temperature of the year\nin degrees Celcius")
# elif mode == "spec_day":
#     plt.ylabel(f"Temperatures at {day_in_words} in {PLAATS}\nin degrees Celcius")
# if mode == "year_avg":
#     plt.title(f"average year-temperature {FIRST_YEAR} - {LAST_YEAR-1}")
# elif mode == "spec_day":
#     plt.title(f"Daytemperature of {day_in_words} in {PLAATS} {FIRST_YEAR} - {LAST_YEAR}")
# elif mode == "period":
#     plt.title(f"Average daytemperature of {months[month1-1]} to {months[month2-1]} in {PLAATS} {FIRST_YEAR} - {LAST_YEAR}")
# elif mode == "month":
#     plt.title(f"Average daytemperature of {months[month1-1]} {year} in {PLAATS}")
# elif mode == "year":
#     plt.title(f"Average daytemperature of {year} in {PLAATS}")