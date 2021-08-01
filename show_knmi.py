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
    #"Date_statistics"
    mask = (df[field].dt.date >= show_from) & (df[field].dt.date <= show_until)
    df = df.loc[mask]
    df = df.reset_index()
    return df

@st.cache(ttl=60 * 60 * 24)
def getdata(stn, fromx, until):
    with st.spinner(f"GETTING ALL DATA ..."):
        #url =  "https://www.daggegevens.knmi.nl/klimatologie/daggegevens?stns=251&vars=TEMP&start=18210301&end=20210310"
        # https://www.knmi.nl/kennis-en-datacentrum/achtergrond/data-ophalen-vanuit-een-script
        #url = f"https://www.daggegevens.knmi.nl/klimatologie/daggegevens?stns={stn}&vars=ALL&start={fromx}&end={until}"
        url = f"https://www.daggegevens.knmi.nl/klimatologie/daggegevens?stns={stn}&vars=TEMP&start={fromx}&end={until}"
        try:
            df = pd.read_csv(url, delimiter=",", header=None,  comment="#",low_memory=False,)

        except:
            st.write("FOUT BIJ HET INLADEN.")
            st.stop()


        column_replacements =  [[0, 'STN'],
                             [1, 'YYYYMMDD'],
                             [2, 'TG'],
                            [3, 'TN'],
                            [4, 'TX'],
                            [5, 'T10N']]
        # column_replacements = [[0, 'STN'],
        #                     [1, 'YYYYMMDD'],
        #                     [2, 'DDVEC'],
        #                     [3, 'FHVEC'],
        #                     [4, 'FG'],
        #                     [5, 'FHX'],
        #                     [6, 'FHXH'],
        #                     [7, 'FHN'],
        #                     [8, 'FHNH'],
        #                     [9, 'FXX'],
        #                     [10, 'FXXH'],
        #                     [11, 'TG'],
        #                     [12, 'TN'],
        #                     [13, 'TNH'],
        #                     [14, 'TX'],
        #                     [15, 'TXH'],
        #                     [16, 'T10N'],
        #                     [17, 'T10NH'],
        #                     [18, 'SQ'],
        #                     [19, 'SP'],
        #                     [20, 'Q'],
        #                     [21, 'DR'],
        #                     [22, 'RH'],
        #                     [23, 'RHX'],
        #                     [24, 'RHXH'],
        #                     [25, 'PG'],
        #                     [26, 'PX'],
        #                     [27, 'PXH'],
        #                     [28, 'PN'],
        #                     [29, 'PNH'],
        #                     [30, 'VVN'],
        #                     [31, 'VVNH'],
        #                     [32, 'VVX'],
        #                     [33, 'VVXH'],
        #                     [34, 'NG'],
        #                     [35, 'UG'],
        #                     [36, 'UX'],
        #                     [37, 'UXH'],
        #                     [38, 'UN'],
        #                     [39, 'UNH'],
        #                     [40, 'EV24']]

        for c in column_replacements:
            df = df.rename(columns={c[0]:c[1]})

        df["YYYYMMDD"] = pd.to_datetime(df["YYYYMMDD"], format="%Y%m%d")
        df["YYYY"]= df['YYYYMMDD'].dt.year
        df["MM"]= df['YYYYMMDD'].dt.month
        df["DD"]= df['YYYYMMDD'].dt.day
        df["dayofyear"]= df['YYYYMMDD'].dt.dayofyear
        month_long_to_short = {"January": "Jan",
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
                       "December": "Dec"
                      }
        month_number_to_short = {"1": "Jan",
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
                       "12": "Dec"
                      }
        df['month'] = df['MM'].astype(str).map(month_number_to_short)
        df['year'] = df['YYYY'].astype(str)
        df['month'] = df['month'].astype(str)
        df['day'] = df['DD'].astype(str)
        df['month_year'] = df['month'] + " - " + df['year']
        df['month_day'] = df['month'] + " - " + df['day']

        to_divide_by_10 = ["TG", "TX", "TN"]
        for d in to_divide_by_10:
            df[d] = df[d]/10

    return df

def show_per_maand(df, gekozen_weerstation, what_to_show):
    df.set_index('YYYYMMDD')
    month_min= st.sidebar.number_input("Beginmaand (van)", 1,12,1,None,format ="%i"  )
    month_max= st.sidebar.number_input("Eindmaand (tot en met)", 1,12,12,None,format ="%i"  )
    groeperen = st.sidebar.selectbox("Per dag/maandgemiddelde", ["per_dag", "maandgem"], index=1)

    jaren = df["YYYY"].tolist()
    df = df[(df['MM'] >= month_min)]
    df = df[(df['MM'] <= month_max)]

    fig, ax = plt.subplots()
    plt.title(f"Maximale dagtemperaturen - gemiddeld per maand in {gekozen_weerstation}")

    if groeperen == "maandgem":
        df_grouped = df.groupby(["year", "month"] ).mean()
        df_pivoted = df_grouped.pivot(index='MM', columns='YYYY', values=what_to_show)
    elif groeperen == "per_dag":
        df["MD"] = df["month_day"]
        df_grouped = df.groupby(["year", "month_day"] ).mean()
        df_pivoted = df_grouped.pivot(index='dayofyear', columns='YYYY', values=what_to_show)
        major_format = mdates.DateFormatter('%b')

        ax.xaxis.set_major_formatter(major_format)
    plt.grid()
    ax.plot(df_pivoted)
    plt.legend(df_pivoted.columns, title=df_pivoted.columns.name)

    st.pyplot(fig)
    st.write(df_pivoted)

def interface():
    """Kies het weerstation, de begindatum en de einddatum

    Returns:
        df, het weerstation, begindatum en einddatum (laatste drie als string)
    """
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
    start_ = "2019-01-01"
    today = datetime.today().strftime("%Y-%m-%d")
    from_ = st.sidebar.text_input("startdatum (yyyy-mm-dd)", start_)
    until_ = st.sidebar.text_input("enddatum (yyyy-mm-dd)", today)

    mode = st.sidebar.selectbox("Modus", ["per dag", "specifieke dag", "jaargemiddelde", "per maand"], index=3)
    wdw = st.sidebar.slider("Window smoothing curves", 1, 45, 7)
        #st.write(df)
    what_to_show = st.sidebar.selectbox("Wat weer te geven", ["temp"], index=0)
    action(stn, from_, until_, mode, wdw, what_to_show, gekozen_weerstation)

def test_from_until(from_, until_):

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

    return FROM,UNTIL

def action(stn, from_, until_, mode, wdw, what_to_show, gekozen_weerstation):
    FROM, UNTIL = test_from_until(from_, until_)
    st.write (mode)
    df_getdata = getdata(stn, FROM.strftime("%Y%m%d"), UNTIL.strftime("%Y%m%d"))

    df = df_getdata.copy(deep=False)

    if mode == "per maand":
        what_to_show = 'TX'
        show_per_maand(df, gekozen_weerstation, what_to_show)
        datefield = None
        title = (f"Dagtemperatuur van {from_} - {until_} in {gekozen_weerstation}")
    else:
        if mode == "per dag":
            datefield = "YYYYMMDD"
            title = (f"Dagtemperatuur van {from_} - {until_} in {gekozen_weerstation}")
        else:
            if mode == "jaargemiddelde":
                df = df.groupby(["YYYY"], sort=True).mean().reset_index()

                title = (f"Gemiddelde jaartemperatuur  van {from_[:4]} - {until_[:4]} in {gekozen_weerstation}")
                st.sidebar.write("Zorg ervoor dat de einddatum op 31 december valt voor het beste resultaat ")


            elif mode == "specifieke dag":
                day =  st.sidebar.number_input("Dag",   1,31,1,None,format ="%i" )
                months = ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober", "november", "december"]
                month = months.index(st.sidebar.selectbox("Maand", months, index=0))+1
                # month= st.sidebar.number_input("Maand", 1,12,1,None,format ="%i"  )

                df = df[(df['YYYYMMDD'].dt.month==month) & (df['YYYYMMDD'].dt.day==day)]
                title = (f"Gemiddelde temperatuur op {find_date_for_title(day,month)} van {from_[:4]} - {until_[:4]} in {gekozen_weerstation}")
                st.sidebar.write("Zorg ervoor dat de datum in de gekozen tijdrange valt voor het beste resultaat ")
            datefield = "YYYY"
        show_plot (df,  datefield, title, wdw, what_to_show)


    show_warmingstripes (df, title)


def show_plot(df, datefield, title,wdw, what_to_show):
    if what_to_show == "temp":
        to_show = [["TG", "Gem. dagtemp", "g"],
                    ["TN", "Min. dagtemp", "b"],
                    ["TX", "Max. dagtemp", "r"],
                    ]
    if len(df) == 1 and datefield =="YYYY":
        st.warning("Selecteer een grotere tijdsperiode")
        st.stop()
    with _lock:
        fig1x = plt.figure()
        ax = fig1x.add_subplot(111)

        for show in to_show:
            sma = df[show[0]].rolling(window=wdw, center=True).mean()
            ax = df[show[0]].plot( label = "_nolegend_",  linestyle="dotted",color = show[2],  linewidth = 0.5)
            ax = sma.plot( label = show[1], color = show[2], linewidth = 0.75)

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
        plt.grid(which='major', axis = 'y')
        plt.title(title)
        plt.legend()
        st.pyplot(fig1x)

def find_date_for_title(day,month):
    months = ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober", "november", "december"]
    # ["January", "February",  "March", "April", "May", "June", "July", "August", "September", "Oktober", "November", "December"]
    return (str(day) + " " + months[month-1])

def show_warmingstripes (df, title):
    # Based on code of Sebastian Beyer
    # https://github.com/sebastianbeyer/warmingstripes/blob/master/warmingstripes.py

    # the colors in this colormap come from http://colorbrewer2.org
    # the 8 more saturated colors from the 9 blues / 9 reds
    # https://matplotlib.org/matplotblog/posts/warming-stripes/
    cmap = ListedColormap([
        '#08306b', '#08519c', '#2171b5', '#4292c6',
        '#6baed6', '#9ecae1', '#c6dbef', '#deebf7',
        '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a',
        '#ef3b2c', '#cb181d', '#a50f15', '#67000d',
    ])
    # https://github.com/sebastianbeyer/warmingstripes/blob/master/warmingstripes.py
    temperatures = df['TG'].tolist()
    stacked_temps = np.stack((temperatures, temperatures))
    with _lock:
        # plt.figure(figsize=(4,18))
        fig, ax = plt.subplots()

        fig = ax.imshow(stacked_temps, cmap=cmap, aspect=40, )
        plt.gca().set_axis_off()

        plt.title(title)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        #plt.show()
        #st.pyplot(fig) - gives an error
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

def main():
    interface()


if __name__ == "__main__":
    main()