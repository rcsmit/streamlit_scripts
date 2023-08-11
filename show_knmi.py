from imghdr import what
import pandas as pd
import numpy as np

import streamlit as st
#from streamlit import caching
import datetime as dt
import scipy.stats as stats
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg

_lock = RendererAgg.lock
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.dates as mdates

import plotly.express as px
import math
import plotly.graph_objects as go
import platform

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

def rh2q(rh, t, p ):
    """[summary]

    Args:
        rh ([type]): rh min
        t ([type]): temp max

    Returns:
        [type]: [description]
    """
    # https://archive.eol.ucar.edu/projects/ceop/dm/documents/refdata_report/eqns.html

    #Td = math.log(e/6.112)*243.5/(17.67-math.log(e/6.112))
    try:
        es = 6.112 * math.exp((17.67 * t)/(t + 243.5))
        e = es * (rh / 100)
        q_ = (0.622 * e)/(p - (0.378 * e)) * 1000
        x  = round(q_,2)
    except:
        x= None

    return x  

def rh2ah(rh, t ):
    """[summary]

    Args:
        rh ([type]): rh min
        t ([type]): temp max

    Returns:
        [type]: [description]
    """
    # return (6.112 * ((17.67 * t) / (math.exp(t) + 243.5)) * rh * 2.1674) / (273.15 + t ) # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7831640/
    try: 
        x= (6.112 * math.exp((17.67 * t) / (t + 243.5)) * rh * 2.1674) / (273.15 + t )
    except:
        x= None
    return  x

def log10(t):
    try:
        x = np.log10(t)
    except:
        x = None
    return x

    
@st.cache_data (ttl=60 * 60 * 24)
def getdata(stn, fromx, until):
    #url=r"C:\Users\rcxsm\Downloads\df_knmi_de_bilt_01011901_27072023.csv"
    #url = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\knmi_nw_beerta_no_header.csv"
    url = f"https://www.daggegevens.knmi.nl/klimatologie/daggegevens?stns={stn}&vars=TEMP:SQ:SP:Q:DR:RH:UN:UX&start={fromx}&end={until}"
    
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

def download_button(df):    
    csv = convert_df(df)

    st.sidebar.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='df_knmi.csv',
        mime='text/csv',
    )

@st.cache_data 
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')



def does_rain_predict_rain(df):
    """reproducing 
    
    https://medium.com/towards-data-science/does-rain-predict-rain-us-weather-data-and-the-correlation-of-rain-today-and-tomorrow-3a62eda6f7f7

    Args:
        df (_type_): Dataframe with information. 
        STN = codenumber of the staiton
        neerslag_etmaalsom = total amount of percipation per 24h
        YYYYMMDD = the date, already made as pd.datetime elsewhere 
                            df["YYYYMMDD"] = pd.to_datetime(df["YYYYMMDD"], format="%Y%m%d")
        
        RAINY (float, optional): How much percipation is need to consider  a day  as rainy. 
                                 Defaults to 0.5.
    """   
    RAINY = st.sidebar.number_input("treshold(mm percipation)",0.0,100.0,0.5) 
    NR_DAYS = st.sidebar.number_input("Number of days to consider",0,100,10) 
    df.fillna(0) # there is no data before 1906 and april 1945 is missing.
    st.write (df)
    st.write ("reproducing https://medium.com/towards-data-science/does-rain-predict-rain-us-weather-data-and-the-correlation-of-rain-today-and-tomorrow-3a62eda6f7f7")
    stationDF = df.rename({"STN":"STATION", "YYYYMMDD":"DATE", "neerslag_etmaalsom":'DlySumToday'}, axis='columns') 
    stationDF = stationDF[["STATION","DATE","DlySumToday"]]  # keep just what we need
    

    stationCopyDF = df[["STN","YYYYMMDD","neerslag_etmaalsom"]] # keep just what we need
    stationCopyDF = stationCopyDF.rename({"STN":"STATION","neerslag_etmaalsom":"DlySumOther", "YYYYMMDD":"DATEother"}, axis='columns')  
 
    # Add in some other dates, for which we will pull in rainfall.
    for n in range(1,NR_DAYS+1):
        stationDF[f"DATE_minus{n}"] = stationDF["DATE"] - pd.offsets.Day(n)
    stationDF["DATE_plus1"] = stationDF["DATE"] + pd.offsets.Day(1)
   
    # Join other rainfall onto base record. Adjust column names to make clear what we did.
    for n in range(1,NR_DAYS+1):
        stationDF = stationDF.merge(stationCopyDF, how='inner', left_on=["STATION",f"DATE_minus{n}"], right_on = ["STATION","DATEother"])
        ago_n = f"DlySum{n}DaysAgo"

        stationDF = stationDF.rename({"DlySumOther":ago_n}, axis='columns')  
        stationDF = stationDF.drop(columns=["DATEother"])

    stationDF = stationDF.merge(stationCopyDF, how='inner', left_on=["STATION","DATE_plus1"], right_on = ["STATION","DATEother"])
    stationDF = stationDF.rename({"DlySumOther":"DlySumTomorrow"}, axis='columns')  
    stationDF = stationDF.drop(columns=["DATEother"])
    stationDF["DaysOfRain"] = 0

    stationDF.loc[(stationDF["DlySumToday"] >= RAINY), "DaysOfRain"] = 1
    for i in range(1, NR_DAYS+1):
        conditions = [
            stationDF[f'DlySum{i - j}DaysAgo'] >= RAINY for j in range(i)
        ]
        combined_condition = stationDF['DlySumToday'] >= RAINY
        for cond in conditions:
            combined_condition &= cond
        stationDF.loc[combined_condition, 'DaysOfRain'] = i+1

    stationDF = stationDF[["STATION","DATE","DlySumToday", 'DaysOfRain']]      
    stationDF = stationDF.sort_values(by='DATE')
    stationDF['DlySumToday_tomorrow'] = stationDF['DlySumToday'].shift(-1)
    stationDF['does_it_rain_tomorrow'] = stationDF['DlySumToday_tomorrow'] > RAINY
    stationDF.drop('DlySumToday_tomorrow', axis=1, inplace=True)

    st.header("Total period")
    rain_probabilities = stationDF.groupby('DaysOfRain')['does_it_rain_tomorrow'].mean().reset_index()
    st.write(rain_probabilities)

    num_rainy_days = (stationDF['DlySumToday'] > RAINY).sum()
    total_days = len(stationDF)
    fraction_rainy_days = num_rainy_days / total_days
    st.write(f"Fraction of days with DlySumToday > {RAINY}: {fraction_rainy_days}")

    st.header("Per decade")
    #Calculate the number and fraction of days where 'DlySumToday' > 0.1 for each decade
    stationDF['decade'] = stationDF['DATE'].dt.year // 10 * 10
    rainy_days_per_decade = stationDF[stationDF['DlySumToday'] > RAINY].groupby('decade')['DlySumToday'].count()
    total_days_per_decade = stationDF.groupby('decade')['DlySumToday'].count()
    fraction_rainy_days_per_decade = rainy_days_per_decade / total_days_per_decade
    st.subheader("Fraction of rainy days per decade:")
    st.write(fraction_rainy_days_per_decade)

    # RAIN PROBABILITIES PER [CONSECUTIVE DAYS OF RAIN] PER DECADE
    stationDF['decade'] = stationDF['DATE'].dt.year // 10 * 10
    rain_probabilities_by_decade = []

    for decade, decade_df in stationDF.groupby('decade'):
        rain_probabilities = decade_df.groupby('DaysOfRain')['does_it_rain_tomorrow'].mean().reset_index()
        rain_probabilities['decade'] = decade
        rain_probabilities_by_decade.append(rain_probabilities)

    rain_probabilities_combined = pd.concat(rain_probabilities_by_decade, ignore_index=True)
    result_table = rain_probabilities_combined.pivot(index='decade', columns='DaysOfRain', values='does_it_rain_tomorrow')
    st.subheader("RAIN PROBABILITIES PER [CONSECUTIVE DAYS OF RAIN] PER DECADE")
    st.write(result_table)

    # MAKE A PLOT (rain_probabilities islike result_table, but every value on a row)
    rain_probabilities = stationDF.groupby(['decade', 'DaysOfRain'])['does_it_rain_tomorrow'].mean().reset_index()
    fig = px.line(rain_probabilities, x='decade', y='does_it_rain_tomorrow', color='DaysOfRain', title='Rain Probabilities by Decade and Days of Rain')
    fig.update_xaxes(tickmode='linear', dtick=10)
    st.plotly_chart(fig)
    
    # Create a custom colorscale from light to dark
    colorscale = [
        [0, 'rgb(0, 256, 256)'],
        [1, 'rgb(0,128, 128)']
        
    ]

    result_table = result_table.iloc[::-1] # reverse the order, 1900 on top, 2020 bottom
    # Create a heatmap using plotly.graph_objs
    heatmap = go.Figure(data=go.Heatmap(
        z=result_table.values,
        x=result_table.columns,
        y=result_table.index,
        colorscale=colorscale   # You can choose a different color scale if desired, was 'Viridis'
    ))

    
    heatmap.update_layout(
        xaxis_title="Days of Rain",
        yaxis_title="Decade",
        title="Rain Probabilities Heatmap by Decade and Days of Rain"
    )

    # Display the heatmap
    st.plotly_chart(heatmap)

    # AVERAGE RAINGFALL PER DECADE
    st.subheader("AVERAGE RAINFALL PER DECADE")
    average_rainfall_per_decade = stationDF.groupby('decade')['DlySumToday'].mean().reset_index()
    fig = px.bar(average_rainfall_per_decade, x='decade', y='DlySumToday', 
             title='Average Rainfall per Decade', labels={'DlySumToday': 'Average Rainfall'})
    fig.update_xaxes(tickmode='linear', dtick=10)
    st.plotly_chart(fig)










def show_aantal_kerend(df_, gekozen_weerstation, what_to_show_):
    # TODO : stacked bargraphs met meerdere condities
    # https://twitter.com/Datagraver/status/1535200978814869504/photo/1
    months = {
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
    what_to_show_ = what_to_show_ if type(what_to_show_) == list else [what_to_show_]

    df_.set_index("YYYYMMDD")
    (month_min,month_max) = st.sidebar.slider("Maanden (van/tot en met)", 1, 12, (1,12))

    value_min = st.sidebar.number_input("Waarde vanaf", -99, 99, 0)
    value_max = st.sidebar.number_input("Waarde tot en met", -99, 99, 99)


    #jaren = df["YYYY"].tolist()
    for what_to_show in what_to_show_:
        st.subheader(what_to_show)
        
        df = df_[(df_["MM"] >= month_min) & (df_["MM"] <= month_max)].reset_index().copy(deep=True)
     
        # TODO :  this should be easier: 
        for i in range(len(df)):
            #if ((df.loc[i, what_to_show]  >= value_min) & (df.loc[i,what_to_show] <= value_max)):
            if ((df[what_to_show].iloc[i]  >= value_min) & (df[what_to_show].iloc[i] <= value_max)):
                df.loc[i,"count_"] = 1
            else:
                df.loc[i,"count_"] = 0
        df = df[df["count_"] == 1]
        
        # aantal keren
        df_grouped_aantal_keren = df.groupby(by=["year"]).sum(numeric_only=True).reset_index() # werkt maar geeft geen 0 waardes weer 
        title = (f"Aantal keren dat { what_to_show} in {gekozen_weerstation} tussen {value_min} en {value_max} ligt\n")
        
        plot_df_grouped( months, month_min, month_max, df_grouped_aantal_keren, "count_", title)

        
        # per maand
        table_per_month = pd.pivot_table(df, values="count_", index='MM', columns='year', aggfunc='sum', fill_value=0)
        all_months = range(month_min, month_max+1)
        all_years = df['year'].unique()
        table_per_month = table_per_month.reindex(index=all_months, columns=all_years, fill_value=0)
      
        fig_ = px.imshow(table_per_month, title=title)
        #fig.show()
        st.plotly_chart(fig_)
        
        # Som
        df_grouped_som = df.groupby(by=["year"]).sum(numeric_only=True).reset_index() # werkt maar geeft geen 0 waardes weer 
        title = (f"Som van {what_to_show} in {gekozen_weerstation} tussen {value_min} en {value_max}")
        plot_df_grouped( months, month_min, month_max, df_grouped_som, what_to_show, title)

        # Gemiddelde
        df_grouped_mean = df.groupby(by=["year"]).mean(numeric_only=True).reset_index() # werkt maar geeft geen 0 waardes weer 
        title = (f"Gemiddelde van {what_to_show} in {gekozen_weerstation} tussen {value_min} en {value_max}")
        plot_df_grouped( months, month_min, month_max,  df_grouped_mean, what_to_show, title)


def plot_df_grouped(months, month_min, month_max,  df_grouped_, veldnaam, title):
    # fig, ax = plt.subplots()
    # plt.set_loglevel('WARNING') #Avoid : Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.
    df_grouped = df_grouped_[["year", veldnaam]]

    if month_min ==1 & month_max ==12:
        st.write("compleet jaar") # FIXIT : werkt niet

    else:
        title += f" in de maanden {months.get(str(month_min))} tot en met {months.get(str(month_max))}"

    fig = px.bar(df_grouped, x='year', y=veldnaam, title=title)
    st.plotly_chart(fig)
 

    # HEATMAP
    # Create a 2D array from the dataframe for the heatmap
    heatmap_data = pd.pivot_table(df_grouped, values=veldnaam, index='year', columns=None)

    # Create the heatmap using plotly
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,  # Use the values from the 2D array
        x=heatmap_data.columns,  # X-axis (in this case, the count)
        y=heatmap_data.index,    # Y-axis (in this case, the year)
        colorscale='Viridis'     # Choose a colorscale (you can change it to another if you prefer)
    ))

    # Customize the heatmap layout
    fig.update_layout(
        title=title,
        xaxis_title='_',
        yaxis_title='Year'
    )

    # Show the heatmap
    st.plotly_chart(fig)

    # plt.title(title)
    # plt.bar(df_grouped["year"], df_grouped[veldnaam])
    # plt.grid()
    # xticks = ax.xaxis.get_major_ticks()
    # if len(xticks)>10:
    #     for i, tick in enumerate(xticks):
    #             if i % int(len(xticks)/10) != 0:
    #                 tick.label1.set_visible(False)
    # plt.xticks(rotation=270)
        
    # st.pyplot(fig)
    
    # df_ = df[(df["count_"] >0)].copy(deep=True)
    # st.write(df_)

    
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
            df_grouped = df.groupby(["year", "MM"]).mean(numeric_only = True).reset_index()
     
            df_grouped ["year"] = df_grouped ["year"].astype(str)
     
            df_pivoted = df_grouped.pivot(
                index="MM", columns="year", values=what_to_show
            ).reset_index()
        elif groeperen == "per_dag":
            df["MD"] = df["month_day"]
            df_grouped = df.groupby(["year", "mmdd"]).mean(numeric_only = True).reset_index()
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
                # sma = [go.Scatter(x=[pd.Series(df_pivoted.index.values),df_pivoted.mmdd], y=df_pivoted[c],  
                #    mode='lines',  line=dict(width=.7), name=f'{c}')
                #     for c in df_pivoted.columns[1:]]
        

                        # create the traces
                sma = []
                list_years = df_pivoted.columns[1:]
                try:
                    highlight_year = st.sidebar.selectbox("Highlight year", list_years,len(list_years)-1)
                except:
                    pass
                for i, col in enumerate(df_pivoted.columns[1:]):
                    line_width = 0.7 if col != highlight_year  else 3  # make last column thicker
                    trace = go.Scatter(x=[df_pivoted.index, df_pivoted.mmdd], y=df_pivoted[col],
                                    mode='lines', line=dict(width=line_width), name=col)
                    sma.append(trace)

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
        "Modus (kies HELP voor hulp)", ["doorlopend per dag", "aantal keren", "specifieke dag", "jaargemiddelde", "maandgemiddelde", "per dag in div jaren", "per maand in div jaren", "percentiles", "polar_plot", "does rain predict rain", "show weerstations", "help"], index=0
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
        show_aantal_kerend(df, gekozen_weerstation, what_to_show)
    elif mode == "percentiles":
        plot_percentiles(df,  gekozen_weerstation, what_to_show, wdw, centersmooth)
    elif mode == "polar_plot":
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




def calculate_loess(X, y, alpha, deg, all_x = True, num_points = 100):
    # from scipy.linalg import qr, pinv   
    # from scipy.linalg import solve_triangular
    '''
    NOT IN USE
    https://simplyor.netlify.app/loess-from-scratch-in-python-animation.en-us/

    Parameters
    ----------
    X : numpy array 1D
        Explanatory variable.
    y : numpy array 1D
        Response varible.
    alpha : double
        Proportion of the samples to include in local regression.
    deg : int
        Degree of the polynomial to fit. Option 1 or 2 only.
    all_x : boolean, optional
        Include all x points as target. The default is True.
    num_points : int, optional
        Number of points to include if all_x is false. The default is 100.

    Returns
    -------
    y_hat : numpy array 1D
        Y estimations at each focal point.
    x_space : numpy array 1D
        X range used to calculate each estimation of y.

    '''
    
    assert (deg == 1) or (deg == 2), "Deg has to be 1 or 2"
    assert (alpha > 0) and (alpha <=1), "Alpha has to be between 0 and 1"
    assert len(X) == len(y), "Length of X and y are different"
    
    if all_x:
        X_domain = X
    else:
        minX = min(X)
        maxX = max(X)
        X_domain = np.linspace(start=minX, stop=maxX, num=num_points)
        
    n = len(X)
    span = int(np.ceil(alpha * n))
    #y_hat = np.zeros(n)
    #x_space = np.zeros_like(X)
    
    y_hat = np.zeros(len(X_domain))
    x_space = np.zeros_like(X_domain)
    
    for i, val in enumerate(X_domain):
    #for i, val in enumerate(X):
        distance = abs(X - val)
        sorted_dist = np.sort(distance)
        ind = np.argsort(distance)
        
        Nx = X[ind[:span]]
        Ny = y[ind[:span]]
        
        delx0 = sorted_dist[span-1]
        
        u = distance[ind[:span]] / delx0
        w = (1 - u**3)**3
        
        W = np.diag(w)
        A = np.vander(Nx, N=1+deg)
        
        V = np.matmul(np.matmul(A.T, W), A)
        Y = np.matmul(np.matmul(A.T, W), Ny)
        Q, R = qr(V)
        p = solve_triangular(R, np.matmul(Q.T, Y))
        #p = np.matmul(pinv(R), np.matmul(Q.T, Y))
        #p = np.matmul(pinv(V), Y)
        y_hat[i] = np.polyval(p, val)
        x_space[i] = val
        
    return y_hat, x_space

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

def climatrend(t, y, p=None, t1=None, t2=None, ybounds=None, drawplot=False, draw30=False):

    """
    Fit a trendline to an annually sampled time-series by local linear regression (LOESS)

    Parameters:
    t : numpy array of shape (n,)
        Years, increasing by 1.
    y : numpy array of shape (n,)
        Annual values; missing values as blanks are allowed near the beginning and end.
    p : float, optional
        Confidence level for error bounds (default: 0.95).
    t1 : float, optional
        First year for which trendline value is compared in the test.
    t2 : float, optional
        Second year (see t1) for which trendline value is compared in the test.
    ybounds : list or array-like, optional
        Lower/upper bound on the value range of y (default: [-Inf, Inf]).
    drawplot : bool or str, optional
        If True, a plot will be drawn. If a string is provided, it will be used as the label on the y-axis.
        (default: False).
    draw30 : bool, optional
        If True, add 30-year moving averages to the plot (default: False).

    Returns:
    pandas DataFrame or dictionary
        A DataFrame or dictionary with the following columns/values:
            't': years,
            'trend': trendline in y for years in t,
            'p': confidence level,
            'trendubound': lower confidence limit,
            'trendlbound': upper confidence limit,
            'averaget': central value of t in a 30-year interval,
            'averagey': 30-year average of y,
            't1': first year for which trendline value is compared in the test,
            't2': second year for which trendline value is compared in the test,
            'pvalue': p-value of the test of no long-term change,
            'ybounds': bounds on the value range of y.

    Details:
    The trendline can be regarded as an approximation of a 30-year average, which has a smooth appearance
    and is extended toward the beginning and end of the time-series.

    It is based on linear local regression, computed using the statsmodels library. It uses a bicubic weight
    function over a 42-year window. In the central part of the time-series, the variance of the trendline
    estimate is approximately equal to the variance of a 30-year average.

    To test the proposition of no long-term change between the years t1 and t2, these years need to be supplied.
    The result is the p-value: the probability (under the proposition) that the estimated trendline values in
    t2 and t1 differ more than observed.

    Version: 09-Mar-2021

    References:
    KNMI Technical report TR-389 (see http://bibliotheek.knmi.nl/knmipubTR/TR389.pdf)

    Author: Cees de Valk (cees.de.valk@knmi.nl)

    # https://gitlab.com/cees.de.valk/trend_knmi/-/blob/master/R/climatrend.R?ref_type=heads
    # translated from R to Python by ChatGPT and adapted by Rene Smit
    # not tested 100%

    """
    

    # Fixed parameters
    width = 42
    
    # Check input -> gives error

    # if t is None or y is None or len(t) < 3 or len(t) != len(y):
    #     raise ValueError("t and y arrays must have equal lengths greater than 2.")
    # if np.isnan(t).any() or np.isnan(y).sum() < 3:
    #     raise ValueError("t or y contain too many NA.")
    
    # Set default values for p, t1, and t2
    if p is None:
        p = 0.95  # default confidence level
    if t1 is None or t2 is None:
        t1 = np.inf
        t2 = -np.inf

    # Set default value for ybounds
    if ybounds is None:
        ybounds = [-np.inf, np.inf]
    elif len(ybounds) != 2:
        ybounds = [-np.inf, np.inf]

    ybounds = sorted(ybounds)

    # Dimensions and checks
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dt = np.diff(t)[0]
    n = len(y)
    ig = ~np.isnan(y)
    yg = y[ig]
    tg = t[ig]
    ng = sum(ig)

    if ng <= 29:
        raise ValueError("Insufficient valid data (less than 30 observations).")

    # Check values of bounds
    if np.any(yg < ybounds[0]) or np.any(yg > ybounds[1]):
        raise ValueError("Stated bounds are not correct: y takes values beyond bounds.")

    # Averages over 30 time-steps
    avt, avy, avysd = None, None, None
    if ng > 29:
        avt = tg + dt / 2  # time (end of time-step, for 30-year averages)
        avy = np.convolve(yg, np.ones(30) / 30, mode='valid')
        avy2 = np.convolve(yg**2, np.ones(30) / 30, mode='valid')
        avysd = np.sqrt(avy2 - avy**2)
        ind = slice(15, ng - 15)
        avt = avt[ind]
        avy = avy[ind]
        avysd = avysd[ind]

    # Linear LOESS trendline computation
    span = width / ng
    loess_model = sm.nonparametric.lowess(yg, tg, frac=span, return_sorted=False)
    trend = loess_model

    # Confidence limits (normal approximation)
    trendsd = np.std(yg - trend)
    z = 1.96  # 1.96 corresponds to a 95% confidence interval (z-score)
    trendub = trend + z * trendsd
    trendlb = trend - z * trendsd

    # Apply bounds
    trend = np.clip(trend, ybounds[0], ybounds[1])
    trendub = np.clip(trendub, ybounds[0], ybounds[1])
    trendlb = np.clip(trendlb, ybounds[0], ybounds[1])

    # p-value for trend
    pvalue = None
    if t2 in t and t1 in t and t2 >= t1 + 30:
        y1 = trend[t1 == t][0]
        y2 = trend[t2 == t][0]
        y1sd = trendsd[t1 == t][0]
        y2sd = trendsd[t2 == t][0]
        # Two-sided test for absence of trend
        pvalue = (1 - norm.cdf(abs(y2 - y1), scale=np.sqrt(y1sd**2 + y2sd**2))) * 2

    # Plotting
    if drawplot:
        plt.figure(figsize=(8, 6))
        ylim = [np.min([np.min(y), np.min(trendlb)]), np.max([np.max(y), np.max(trendub)])]
        ylim[1] = ylim[0] + (ylim[1] - ylim[0]) * 1.0
        plt.plot(t, y, 'b-', label='Temperature Data')
        plt.plot(t, trend, 'r-', lw=2, label='Trendline')
        plt.fill_between(t, trendlb, trendub, color='grey', alpha=0.5, label='Confidence Interval')
        
        if draw30:
            plt.plot(avt, avy, 'ko', markersize=3, label='30-yr Average')

        plt.xlabel('Year')
        plt.ylabel('Temperature')
        plt.grid()
        plt.legend()
        plt.show()

    # results_df = pd.DataFrame({
    #     't': t,
    #     'trend': trend,
    #     'p': p,
    #     'trendubound': trendub,
    #     'trendlbound': trendlb,
    #     'averaget': avt,
    #     'averagey': avy,
    #     't1': t1,
    #     't2': t2,
    #     'pvalue': pvalue,
    #     'ybounds': ybounds
    # })
    

    # return {'t': t, 'trend': trend, 'p': p, 'trendubound': trendub, 'trendlbound': trendlb,
    #         'averaget': avt, 'averagey': avy, 't1': t1, 't2': t2, 'pvalue': pvalue,
    #         'ybounds': ybounds}
    return t, trend, trendlb, trendub

def show_plot(df, datefield, title, wdw, wdw2, sma2_how, what_to_show_, graph_type, centersmooth, show_ci, wdw_ci, show_parts, no_of_parts):
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
        fig = go.Figure()
        data=[]
        for what_to_show_x in what_to_show_:
            #fig = go.Figure()
            avg = round(df[what_to_show_x].mean(),1)
            std = round(df[what_to_show_x].std(),1)
            sem = df[what_to_show_x].sem()

            lower68 = round(df[what_to_show_x].quantile(0.16),1)
            upper68 = round(df[what_to_show_x].quantile(0.84),1)


            lower95 = round(df[what_to_show_x].quantile(0.025),1)
            upper95 = round(df[what_to_show_x].quantile(0.975),1)

            # Calculate the moving confidence interval for the mean using the last 25 values
            moving_ci_lower_95 = df[what_to_show_x].rolling(window=wdw_ci).mean() - df[what_to_show_x].rolling(window=wdw_ci).std() * 2
            moving_ci_upper_95 = df[what_to_show_x].rolling(window=wdw_ci).mean() + df[what_to_show_x].rolling(window=wdw_ci).std() * 2

            moving_ci_lower_68 = df[what_to_show_x].rolling(window=wdw_ci).mean() - df[what_to_show_x].rolling(window=wdw_ci).std() * 1
            moving_ci_upper_68 = df[what_to_show_x].rolling(window=wdw_ci).mean() + df[what_to_show_x].rolling(window=wdw_ci).std() * 1

            

          
            # Quantiles and (mean + 2*std) are two different measures of dispersion, which can be used to understand the distribution of a dataset.
 
            # Quantiles divide a dataset into equal-sized groups, based on the values of the dataset. For example, the median is the 50th percentile, which divides the dataset into two equal-sized groups. Similarly, the 25th percentile divides the dataset into two groups, with 25% of the values below the 25th percentile and 75% of the values above the 25th percentile.

            # On the other hand, (mean + 2*std) represents a range of values that are within two standard deviations of the mean. This is sometimes used as a rule of thumb to identify outliers, since values that are more than two standard deviations away from the mean are relatively rare.

            # The main difference between quantiles and (mean + 2std) is that quantiles divide the dataset into equal-sized groups based on the values, while (mean + 2std) represents a range of values based on the mean and standard deviation. In other words, quantiles are based on the actual values of the dataset, while (mean + 2*std) is based on the mean and standard deviation, which are summary statistics of the dataset.

            # It's also worth noting that (mean + 2std) assumes that the data is normally distributed, while quantiles can be used for any distribution. Therefore, if the data is not normally distributed, (mean + 2std) may not be a reliable measure of dispersion.
            # confidence interval for the mean
            ci = stats.t.interval(0.95, len(df[what_to_show_x])-1, loc=df[what_to_show_x].mean(), scale=sem)

            # print confidence interval
          
            n_parts = no_of_parts
            rows_per_part = len(df) // n_parts
            # Step 2: Calculate the average temperature for each part
            average_values = [df.iloc[i * rows_per_part:(i + 1) * rows_per_part][what_to_show_x].mean() for i in range(n_parts)]
            X_array = df[datefield].values
            Y_array = df[what_to_show_x].values
            if len(X_array)>30:
                #y_hat2, x_space2 = calculate_loess(X_array, Y_array, 0.05, 1, all_x = True, num_points = 200)
                x_space2, y_hat2, trendlb, trendub  = climatrend(X_array, Y_array)

                loess = go.Scatter(
                    name=f"{what_to_show_x} Loess",
                    x=x_space2,
                    y= y_hat2,
                    mode='lines',
                    line=dict(width=1,
                    color='rgba(255, 0, 255, 1)'
                    ),
                    )
                loess_low = go.Scatter(
                    name=f"{what_to_show_x} Loess low",
                    x=x_space2,
                    y= trendlb,
                    mode='lines',
                    line=dict(width=.7,
                    color='rgba(255, 0, 255, 0.5)'
                    ),
                    )
                loess_high = go.Scatter(
                    name=f"{what_to_show_x} Loess high",
                    x=x_space2,
                    y= trendub,
                    mode='lines',
                    line=dict(width=0.7,
                    color='rgba(255, 0, 255, 0.5)'
                    ),
                    )
            df["sma"] = df[what_to_show_x].rolling(window=wdw, center=centersmooth).mean()
            if (wdw2 != 999):
                if (sma2_how == "mean"):
                    df["sma2"] = df[what_to_show_x].rolling(window=wdw2, center=centersmooth).mean()
                elif (sma2_how == "median"):
                    df["sma2"] = df[what_to_show_x].rolling(window=wdw2, center=centersmooth).median()

                sma2 = go.Scatter(
                    name=f"{what_to_show_x} SMA ({wdw2})",
                    x=df[datefield],
                    y= df["sma2"],
                    mode='lines',
                    line=dict(width=2,
                    color='rgba(0, 168, 255, 0.8)'
                    ),
                    )
            if wdw ==1:
                name_sma = f"{what_to_show_x}"
            else:
                name_sma = f"{what_to_show_x} SMA ({wdw})"
            sma = go.Scatter(
                name=name_sma,
                x=df[datefield],
                y= df["sma"],
                mode='lines',
                line=dict(width=1,
                color='rgba(0, 0, 255, 0.6)'
                ),
                )
            if wdw != 1:
                points = go.Scatter(
                    name="",
                    x=df[datefield],
                    y= df[what_to_show_x],
                    mode='markers',
                    showlegend=False,
                    marker=dict(
                    #color='LightSkyBlue',
                    size=3))
            # Create traces for the moving confidence interval as filled areas
            ci_area_trace_95 = go.Scatter(
                name=f"{what_to_show_x} 95% CI",
                x=df[datefield],
                y=pd.concat([moving_ci_lower_95, moving_ci_upper_95[::-1]]),  # Concatenate lower and upper CI for the fill
                fill='tozerox',  # Fill the area to the x-axis
                fillcolor='rgba(211, 211, 211, 0.3)',  # Set the fill color to grey (adjust the opacity as needed)
                line=dict(width=0),  # Set line width to 0 to hide the line of the area trace
            )
             # Create traces for the moving confidence interval
            ci_lower_trace_95 = go.Scatter(
                name=f"{what_to_show_x} 95% CI Lower",
                x=df[datefield],
                y=moving_ci_lower_95,
                mode='lines',
                line=dict(width=1, dash='dash'),
            )
            ci_upper_trace_95 = go.Scatter(
                name=f"{what_to_show_x} 95% CI Upper",
                x=df[datefield],
                y=moving_ci_upper_95,
                mode='lines',
                line=dict(width=1, dash='dash'),
            )
            ci_area_trace_68 = go.Scatter(
                name=f"{what_to_show_x} 68% CI",
                x=df[datefield].to_list(),
                y=moving_ci_lower_68+ moving_ci_upper_68[::-1],  # Concatenate lower and upper CI for the fill
                fill='tozerox',  # Fill the area to the x-axis
                fillcolor='rgba(211, 211, 211, 0.5)',  # Set the fill color to grey (adjust the opacity as needed)
                line=dict(width=0),  # Set line width to 0 to hide the line of the area trace
            )
            ci_lower_trace_68 = go.Scatter(
                name=f"{what_to_show_x} 68% CI Lower",
                x=df[datefield],
                y=moving_ci_lower_68,
                mode='lines',
                line=dict(width=1, dash='dash'),
            )
            ci_upper_trace_68 = go.Scatter(
                name=f"{what_to_show_x} 68% CI Upper",
                x=df[datefield],
                y=moving_ci_upper_68,
                mode='lines',
                line=dict(width=1, dash='dash'),
            )

           
            #data = [sma,points]
            data.append(sma)
            if len(X_array)>30:
                data.append(loess)
                data.append(loess_low)
                data.append(loess_high)
            if wdw2 != 999:
                data.append(sma2)
            if wdw != 1:
                data.append(points)
            if show_ci:
                # Append the moving confidence interval traces to the data list
                data.append(ci_lower_trace_95)
                data.append(ci_upper_trace_95)
                data.append(ci_lower_trace_68)
                data.append(ci_upper_trace_68)
                #data.append(ci_area_trace_95)
                #data.append(ci_area_trace_68)


            layout = go.Layout(
                yaxis=dict(title=what_to_show_x),
                title=title,)
                #, xaxis=dict(tickformat="%d-%m")
            # fig = go.Figure(data=data, layout=layout)
            # fig.update_layout(xaxis=dict(tickformat="%d-%m-%Y"))
            # st.plotly_chart(fig, use_container_width=True)
        fig = go.Figure(data=data, layout=layout)
        # Add horizontal lines for average values
        if show_parts:
            for i, avg_val in enumerate(average_values):
                if i != (len(average_values) -1):
                    fig.add_trace(go.Scatter(x=[df[datefield].iloc[i * rows_per_part], df[datefield].iloc[min((i + 1) * rows_per_part - 1, len(df) - 1)]],
                                            y=[avg_val, avg_val],
                                            mode='lines', line=dict(color='red'),showlegend=False, name=f'Avg Part {i + 1}'))
                else:    
                    fig.add_trace(go.Scatter(x=[df[datefield].iloc[i * rows_per_part], df[datefield].iloc[len(df) - 1]],
                                            y=[avg_val, avg_val],
                                            mode='lines', line=dict(color='red'),showlegend=False, name=f'Avg Part {i + 1}'))
                
               
   
        fig.update_layout(xaxis=dict(tickformat="%d-%m-%Y"))
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"{what_to_show_x} | mean = {avg} | std= {std} | quantiles (68%) [{lower68}, {upper68}] | quantiles (95%) [{lower95}, {upper95}]")
            
    #df =df[[datefield,what_to_show_[0]]]
    #st.write(df)

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

def  polar_plot(df2,   what_to_show, how):
    # https://scipython.com/blog/visualizing-the-temperature-in-cambridge-uk/
    # import numpy as np
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # from matplotlib import cm
    # from matplotlib.colors import Normalize
    # from mpl_toolkits.mplot3d import Axes3D
    # plt.rcParams['text.usetex'] = True
 
    for w in what_to_show:   
        st.subheader(w)
        df2["YYYYMMDD_"] = pd.to_datetime(df2["YYYYMMDD"], format="%Y%m%d")
        # Convert the timestamp to the number of seconds since the start of the year.
        df2['secs'] = (df2.YYYYMMDD_ - pd.to_datetime(df2.YYYYMMDD.dt.year, format='%Y')).dt.total_seconds()
     
        # Approximate the angle as the number of seconds for the timestamp divide by
        # the number of seconds in an average year.
        df2['angle'] = df2['secs'] / (365.25 * 86400) *360  #   * 2 * np.pi

        def plot_polar():
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
            
            if how == "line":
                fig = px.line_polar(df2, r=w, color='YYYY', theta='angle',color_discrete_sequence=px.colors.sequential.Plasma_r, line_close=False, hover_data=['YYYYMMDD'])
            elif how == "scatter":
                fig = px.scatter_polar(df2, r=w, color='YYYY', theta='angle', hover_data=['YYYYMMDD'])

            else:
                st.error("Error in HOW")
            fig.update_layout(coloraxis={"colorbar":{"dtick":1}}) #only integers in legeenda
            labelevery = 6
            fig.update_layout(
                polar={
                    "angularaxis": {
                        "tickmode": "array",
                        "tickvals": list(range(0, 360, 180 // labelevery)),
                        "ticktext": months,
                    }
                }
            )
            st.plotly_chart(fig)

            # MATPLOTLIB
            # For the colourmap, the minimum is the largest multiple of 5 not greater than
            # the smallest value of T; the maximum is the smallest multiple of 5 not less
            # than the largest value of T, e.g. (-3.2, 40.2) -> (-5, 45).
            Tmin = 5 * np.floor(df2[w].min() / 5)
            Tmax = 5 * np.ceil(df2[w].max() / 5)
            # Normalization of the colourmap.
            # norm = Normalize(vmin=Tmin, vmax=Tmax)
            # c = norm(df2[w])

    
            # fig = plt.figure()
            # ax = fig.add_subplot(projection='polar')
            # # We prefer 1 January (0 deg) on the left, but the default is to the
            # # right, so offset by 180 deg.
            # ax.set_theta_offset(np.pi)
            # cmap = cm.turbo
            # ax.scatter(df2['angle'], df2[w], c=cmap(c), s=2)

            # # Tick labels.
            # ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 6))
            # ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            #                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            # ax.set_yticks([])

            # # Add and title the colourbar.
            # # cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
            # #                     ax=ax, orientation='vertical', pad=0.1)
            # # cbar.ax.set_title(r'Temp')

            # st.pyplot(fig)

        def plot_3d():
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            cmap = cm.turbo
            X = df2[w] * np.cos(df2['angle'] + np.pi)
            Y = df2[w] * np.sin(df2['angle'] + np.pi)
            z = df2.YYYYMMDD.dt.year
            ax.scatter(X, Y, z, c=cmap(c), s=2)

            ax.set_xticks([])
            ax.set_yticks([])
            cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                                ax=ax, orientation='horizontal', pad=-0.02, shrink=0.6)
            #cbar.ax.set_title(r'$T\;/^\circ\mathrm{C}$')
            st.pyplot(fig)

        plot_polar()
        #plot_3d()

    

def show_warmingstripes(df_, title):
    print (df_)
    df = df_.groupby(df_["YYYY"], sort=True).mean(numeric_only = True).reset_index()
    #df_grouped = df.groupby([df[valuefield]], sort=True).sum().reset_index()
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
        show_warmingstripes_matplotlib(df_, title)

def show_warmingstripes_matplotlib(df_, title):
    # https://matplotlib.org/matplotblog/posts/warming-stripes/
    st.subheader("Code from Matplotlib site")
    df = df_.groupby(df_["YYYY"], sort=True).mean(numeric_only = True).reset_index()
    avg_temperature = df["temp_avg"].mean()
    df["anomaly"] = df["temp_avg"]-avg_temperature

    #stacked_temps = np.stack((temperatures, temperatures))
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import ListedColormap
    import pandas as pd
    # Then we define our time limits, our reference period for the neutral color and the range around it for maximum saturation.

    FIRST = int( df["YYYY"].min())
    LAST = int(df["YYYY"].max())  # inclusive


    # Reference period for the center of the color scale

    FIRST_REFERENCE = FIRST
    LAST_REFERENCE = LAST
    LIM = 2 # degrees

    #Here we use pandas to read the fixed width text file, only the first two columns, which are the year and the deviation from the mean from 1961 to 1990.

    # data from

    # https://www.metoffice.gov.uk/hadobs/hadcrut4/data/current/time_series/HadCRUT.4.6.0.0.annual_ns_avg.txt

 
    anomaly = df['anomaly'].tolist()
  
    reference = sum(anomaly)/len(anomaly)
    # This is our custom colormap, we could also use one of the colormaps that come with matplotlib, e.g. coolwarm or RdBu.

    # the colors in this colormap come from http://colorbrewer2.org

    # the 8 more saturated colors from the 9 blues / 9 reds

    cmap = ListedColormap([
        '#08306b', '#08519c', '#2171b5', '#4292c6',
        '#6baed6', '#9ecae1', '#c6dbef', '#deebf7',
        '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a',
        '#ef3b2c', '#cb181d', '#a50f15', '#67000d',
    ])
    # We create a figure with a single axes object that fills the full area of the figure and does not have any axis ticks or labels.

    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    # Finally, we create bars for each year, assign the data, colormap and color limits and add it to the axes.

    # create a collection with a rectangle for each year

    col = PatchCollection([
        Rectangle((y, 0), 1, 1)
        for y in range(FIRST, LAST + 1)
    ])

    # set data, colormap and color limits

    col.set_array(anomaly)
    col.set_cmap(cmap)
    col.set_clim(reference - LIM, reference + LIM)
    ax.add_collection(col)
    #Make sure the axes limits are correct and save the figure.

    ax.set_ylim(0, 1)
    ax.set_xlim(FIRST, LAST + 1)

    fig.savefig('warming-stripes.png')
    st.pyplot(fig)
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
    stn, from_, until_, mode,groupby_, wdw, wdw2, sma2_how, what_to_show, gekozen_weerstation, centersmooth, graph_type, show_ci, wdw_ci,show_parts, no_of_parts = interface()
    action(stn, from_, until_, mode, groupby_,  wdw, wdw2,sma2_how,  what_to_show, gekozen_weerstation, centersmooth, graph_type, show_ci, wdw_ci,show_parts, no_of_parts)


if __name__ == "__main__":
    main()
