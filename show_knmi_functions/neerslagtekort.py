import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px  # For easy colormap generation
# import numpy as np  # For linspace to distribute sampling
import math
from datetime import datetime
try:
    from show_knmi_functions.spaghetti_plot import spaghetti_plot
    from show_knmi_functions.utils import get_data
except:
    from spaghetti_plot import spaghetti_plot
    from utils import get_data
# https://www.knmi.nl/nederland-nu/klimatologie/geografische-overzichten/historisch-neerslagtekort
# http://grondwaterformules.nl/index.php/vuistregels/neerslag-en-verdamping/langjarige-grondwateraanvulling


def calculate_s(temp):
    """s = de afgeleide naar temperatuur van de verzadigingsdampspanning
       (mbar/°C)
       https://nl.wikipedia.org/wiki/Referentie-gewasverdamping
    Args:
        temp (_type_): _description_

    Returns:
        _type_: _description_
    """

    a = 6.1078
    b = 17.294
    c = 237.73
    s = ((a*b*c)/((c+temp)**2))*math.exp((b*temp)/(c+temp))
    return s
def makkink(temp,  straling):
    """_summary_
    Referentiegewasverdamping is de hoeveelheid water die verdampt uit een grasveld dat goed voorzien is van water en nutriënten. Deze waarde wordt in de hydrologie gebruikt als basis om te kunnen berekenen hoeveel water verdampt uit oppervlaktes grond met diverse soorten gewassen.
    Het KNMI berekent sinds 1 april 1987 de referentie-gewasverdamping met de formule van Makkink.
    https://nl.wikipedia.org/wiki/Referentie-gewasverdamping

    Args:
        temp (_type_): _description_
        straling (_type_): _description_
    """    
    s = calculate_s(temp)
    lambdaa = 2.45*10**6
    c1 = 0.65
    c2 = 0
    kin = straling
    gamma = 0.66
    eref = (c1 * (s/(s+gamma)) * kin + c2)/lambdaa
    return eref
def neerslagtekort_(df):
    """Functie die het neerslagtekort berekent

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """    
    df=df.fillna(0)
    df['neerslag_etmaalsom'].replace(-0.1, 0, inplace=True)
    for what in ["temp_avg",  "neerslag_etmaalsom", "glob_straling"]: 
        try:  
            df[f"{what}_sma"] = df[what].rolling(1, center=True).mean()
        except:
            st.error(f"Missing values in {what}")
            st.stop()
    
    df['glob_straling_Wm2_sma'] = (df['glob_straling'] * 10**4) / 86400
    
    df["YYYYMMDD"] = pd.to_datetime(df["YYYYMMDD"].astype(str))

    df['year'] = df['YYYYMMDD'].dt.year
    df['month'] = df['YYYYMMDD'].dt.month
    df = df[(df['month'] >= 4) & (df['month'] <= 9)]
    
    # Applying the function
    df["eref"] = df.apply(lambda row: makkink(row["temp_avg_sma"], row["glob_straling_Wm2_sma"]), axis=1)
    # Conversion factor for kg/(m²·s) to mm/day
    conversion_factor = 864  # Assuming 1 kg/m² of water is equivalent to 86.4 mm of water depth over 24 hours

    # Convert referentiegewasverdamping from kg/(m²·s) to mm/day
    df['referentiegewasverdamping_mm_day'] = df['eref'] * conversion_factor
    df["neerslagtekort"] =   df["neerslag_etmaalsom"] - df["referentiegewasverdamping_mm_day"] 
    df['cumulative_neerslagtekort'] = df.groupby('year')['neerslagtekort'].cumsum()
 
    return df   

def plot_neerslagtekort(df):
    df['date_1900'] = pd.to_datetime(df['YYYYMMDD'].dt.strftime('%d-%m-1900'), format='%d-%m-%Y')

    # Create spaghetti plot with Plotly
    fig = go.Figure()

    for year, data in df.groupby('year'):
        fig.add_trace(go.Scatter(x=data['date_1900'], y=data['cumulative_neerslagtekort'], mode='lines', name=str(year)))

    fig.update_layout(
        title='Cumulative Precipitation Deficit (Neerslagtekort) from April to September - Spaghetti Plot',
        xaxis_title='Date',
        yaxis_title='Cumulative Precipitation Deficit'
    )

    fig.update_layout(
            xaxis=dict(title="date",tickformat="%d-%m"),
            yaxis=dict(title="Neerslagtekort"),
            title=f"Neerslagtekort" ,)
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    st.plotly_chart(fig)

def neerslagtekort(df):
    """deze routine wordt aangeroepen vanuit de algemene knmi menu

    """    
    df = neerslagtekort_(df)
    
    plot_neerslagtekort(df)
    spaghetti_plot(df, ['neerslag_etmaalsom'], 7, 7, False, False, True, False, True, False, "Greys", False)
    spaghetti_plot(df, ['neerslag_etmaalsom'], 7, 7, False, False, True, False, True, False, "Greys", True)

def get_dataframe(FROM, UNTIL):
    
    stations = [260,235,290,278,240,249,391,286,251,319,283]
    #  De Bilt,  260
    # De Kooy, 235
    # Groningen, 280
    # Heerde, 278
    # Hoofddorp, 240
    # Hoorn, 249
    # Kerkwerve, 312 en 324 geven lege results)
    # Oudenbosch, 340 (heeft geen neerslagetmaalsom)
    # Roermond, 391
    # Ter Apel, 286
    # West-Terschelling, 251
    # Westdorpe  319
    # Winterswijk. 283
    df_master = pd.DataFrame()  
    for stn in stations:
        fromx, until =  FROM.strftime("%Y%m%d"), UNTIL.strftime("%Y%m%d")
        # fromx="20230101"

        # until="20231231"     
        url = f"https://www.daggegevens.knmi.nl/klimatologie/daggegevens?stns={stn}&vars=TEMP:SQ:SP:Q:DR:RH:UN:UX&start={fromx}&end={until}"
     
        df_s = get_data(url)

        df= neerslagtekort_(df_s)
        
        df_master = pd.concat([df_master, df])  # Concatenate data for each station to df_master
    
    daily_avg_cumulative_neerslagtekort = df_master.groupby('YYYYMMDD')['cumulative_neerslagtekort'].mean().reset_index()

    return df_master, daily_avg_cumulative_neerslagtekort


def main():

    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result.csv" 
    
    df = get_data(url)   
    df = neerslagtekort_(df)
    st.write (df)
    plot_neerslagtekort(df)
    spaghetti_plot(df, ['neerslag_etmaalsom'], 7, 7, False, False, True, False, True, False, "Greys")
   
def neerslagtekort_meerdere_stations(FROM, UNTIL):
    
    df_master, daily_avg_cumulative_neerslagtekort = get_dataframe(FROM, UNTIL)
    print (df_master)
    # Pivot and calculate statistics
    pivot_table = df_master.pivot(index='YYYYMMDD', columns='STN', values="cumulative_neerslagtekort")
    print (daily_avg_cumulative_neerslagtekort)


    # Create a line plot using Plotly
    fig = go.Figure()
    for column in pivot_table.columns:
        fig.add_trace(go.Scatter(x=pivot_table.index, y=pivot_table[column], mode='lines', name=str(column)))

    fig.update_layout(
        title='Pivot Table Values - Line Plot',
        xaxis_title='Date',
        yaxis_title='Value')
    st.plotly_chart(fig)
if __name__ == "__main__":
    #main()
    fromx = datetime.strptime("2023-01-01", "%Y-%m-%d").date()
    until = datetime.strptime("2023-12-31", "%Y-%m-%d").date()
    neerslagtekort_meerdere_stations(fromx, until)
    