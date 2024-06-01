import pandas as pd
import streamlit as st
import plotly.graph_objects as go
# import plotly.express as px  # For easy colormap generation
import numpy as np  # For linspace to distribute sampling
import math
from datetime import datetime


from scipy.stats import linregress
import statsmodels.api as sm
from scipy import stats

try:
    from show_knmi_functions.spaghetti_plot import spaghetti_plot
    from show_knmi_functions.utils import get_data
except:
    from spaghetti_plot import spaghetti_plot
    from utils import get_data
# https://www.knmi.nl/nederland-nu/klimatologie/geografische-overzichten/historisch-neerslagtekort
# http://grondwaterformules.nl/index.php/vuistregels/neerslag-en-verdamping/langjarige-grondwateraanvulling

# Nog implementeren "   In deze grafiek wordt de berekening stopgezet indien het doorlopend 
#                       tekort op nul uitkomt en weer hervat zodra er een tekort optreedt."
def calculate_s(temp):
    """s = de afgeleide naar temperatuur van de verzadigingsdampspanning
       (mbar/°C)
       https://nl.wikipedia.org/wiki/Referentie-gewasverdamping
    Args:
        temp (float):temperatuur

    Returns:
        float: de afgeleide naar temperatuur van de verzadigingsdampspanning
    """

    a = 6.1078
    b = 17.294
    c = 237.73
    s = ((a*b*c)/((c+temp)**2))*math.exp((b*temp)/(c+temp))
    return s
def makkink(temp_avg, temp_max,  straling):
    """ Berekening van referentie-gewasverdamping met de formule van Makkink.
    Referentiegewasverdamping is de hoeveelheid water die verdampt uit een grasveld dat goed voorzien is van water en nutriënten. Deze waarde wordt in de hydrologie gebruikt als basis om te kunnen berekenen hoeveel water verdampt uit oppervlaktes grond met diverse soorten gewassen.
    Het KNMI berekent sinds 1 april 1987 de referentie-gewasverdamping met de formule van Makkink.
    https://nl.wikipedia.org/wiki/Referentie-gewasverdamping


    lambdaa = verdampingswarmte van water (2.45E6 J/kg bij 20 graden Celsius (°C))
    eref  = referentiegewasverdamping (kg/m^{2}*s)
    c_1 = constante (De Bruin (1981) vond hiervoor een waarde van ca. 0.65)
    c_2 = constante (De Bruin (1981) vond hiervoor een waarde van ca. 0)
    K_in = kortgolvig inkomende straling (W/m^{2})
    gamma = psychrometerconstante (ca 0.66 mbar/°C op zeeniveau)
    s = de afgeleide naar temperatuur van de verzadigingsdampspanning (mbar/°C)
    T = temperatuur (°C)

    temp_avg geeft te lage waardes (110 ipv 130)
    temp_max geeft te hoge waardes (150 ipv 130)

    CONVERSION STRALING
    globale straling van in J/cm2 naar Wm2
    Joules (J)** is a unit of energy.
    Watts (W)** is a unit of power, defined as energy per unit time (1 W = 1 J/s).
    10^4 cm2 = 1 m2

    Convert energy (J/m²) to power (W/m²).  Power is energy per unit time.
    If the radiation is measured over a specific time period, say 1 second, then:
    1 J/m² over 1 second = 1 W/m²
    For example, if the global radiation is given in J/cm² per day, you would 
    first convert it to J/m² and then divide by the number of seconds in a day (86400 seconds) to get W/m².

    CONVERSION EREF
    Conversion factor for kg/(m²·s) to mm/day
    Convert referentiegewasverdamping from kg/(m²·s) to mm/day
    Assuming 1 kg/m² of water is equivalent to 86.4 mm of water depth over 24 hours

    Args:
        temp (_type_): _description_
        straling (_type_): _description_
    """    
    temp_x = (temp_avg + temp_max)/2
    s = calculate_s(temp_x)
    lambdaa = 2.45*10**6
    c1 = 0.65
    c2 = 0
    conversion_factor_straling =  10**4 / 86400
    kin = straling * conversion_factor_straling

    gamma = 0.66

    conversion_factor_eref = 86400 
    eref = (c1 * (s/(s+gamma)) * kin + c2)/lambdaa * conversion_factor_eref
    return eref

def neerslagtekort_(df):
    """Functie die het neerslagtekort berekent

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """    
    df=df.fillna(0.0)
    df['neerslag_etmaalsom'].replace(-0.1, 0, inplace=True)
    
    wdwx =1
    if wdwx>1:
        for what in ["temp_avg", "temp_max", "neerslag_etmaalsom", "glob_straling"]: 
            try:  
                df[f"{what}_sma"] = df[what].rolling(7, center=True).mean()
            except:
                st.error(f"Missing values in {what}")
                st.stop()
    else:
        for what in ["temp_avg", "temp_max", "neerslag_etmaalsom", "glob_straling"]: 
            df[f"{what}_sma"] = df[f"{what}"] 
        
    
    df["YYYYMMDD"] = pd.to_datetime(df["YYYYMMDD"].astype(str))
    df['year'] = df['YYYYMMDD'].dt.year
    df['month'] = df['YYYYMMDD'].dt.month
    df = df[(df['month'] >= 4) & (df['month'] <= 9)]
   

    # Applying the function
    df["eref"] = df.apply(lambda row: makkink(row["temp_avg_sma"],row["temp_max_sma"], row["glob_straling"]), axis=1)
    df["neerslagtekort"] =     df["eref"] -df["neerslag_etmaalsom"]
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
    spaghetti_plot(df, ['neerslag_etmaalsom'], 3, 3, False, False, True, False, True, False, "Pubu", False)
    spaghetti_plot(df, ['neerslag_etmaalsom'], 3, 3, False, False, True, False, True, False, "Pubu", True)
    spaghetti_plot(df, ['temp_avg'], 3, 3, False, False, True, False, True, False, "Pubu", False)
    spaghetti_plot(df, ['temp_max'], 3, 3, False, False, True, False, True, False, "Pubu", False)
    
    spaghetti_plot(df, ['eref'], 1,1, False, False, True, False, True, False, "Pubu", False)
    spaghetti_plot(df, ['eref'], 1,1, False, False, True, False, True, False, "Pubu", True)

#@st.cache_data
def get_dataframe_multiple_(stations,FROM, UNTIL):
    """Get the dataframe with info from multiple stations

    Args:
        FROM (str): date as datetime.strptime("2023-01-01", "%Y-%m-%d").date()
        UNTIL (str): date as datetime.strptime("2023-01-01", "%Y-%m-%d").date()

    Returns:
        _type_: _description_
    """    
  
    df_master_ = pd.DataFrame()  
    for i, stn in enumerate(stations):
        if stn !=None:
            print (f"Downloading {i+1}/{len(stations)}")
            fromx, until =  FROM.strftime("%Y%m%d"), UNTIL.strftime("%Y%m%d")    
            url = f"https://www.daggegevens.knmi.nl/klimatologie/daggegevens?stns={stn}&vars=TEMP:SQ:SP:Q:DR:RH:UN:UX&start={fromx}&end={until}"
        
            df_s = get_data(url)
            df_s = df_s.fillna(0)
            df= neerslagtekort_(df_s)
            df_master_ = pd.concat([df_master_, df])  # Concatenate data for each station to df_master
    
    data = {
        "STN":       [260, 235, 280, 278, 240, 249, 391, 286, 251, 319, 283],
        "stn_in_txt": ["De Bilt", "De Kooy", "Groningen", "Heerde", "Hoofddorp", "Hoorn",  "Roermond", "Ter Apel", "West-Terschelling", "Westdorpe", "Winterswijk"],
        "stn_data": ["De Bilt", "De Kooy", "Eelde", "Heino", "Schiphol", "Berkhout",  "Arcen", "Nieuw Beerta", "Hoorn Terschilling", "Westdorpe", "Hupsel"],
        
    }

    # Create the DataFrame
    df_station = pd.DataFrame(data)
    df_master=pd.merge(df_master_,df_station,how="left",on="STN")
 
    return df_master


def main():

    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result.csv" 
    
    df = get_data(url)   
    df = neerslagtekort_(df)
    st.write (df)
    plot_neerslagtekort(df)
    spaghetti_plot(df, ['neerslag_etmaalsom'], 7, 7, False, False, True, False, True, False, "Pubu")
    # fromx = datetime.strptime("2000-01-01", "%Y-%m-%d").date()
    # until = datetime.strptime("2023-12-31", "%Y-%m-%d").date()
    # neerslagtekort_meerdere_stations(fromx, until)
    
def neerslagtekort_meerdere_stations(FROM, UNTIL):
    dropdown = False
    if dropdown:
        # dit dropdown menu komt niet tevoorschijn bij share.streamlit.io
        data = {
        "STN": [260, 235, 280, 278, 240, 249, 391, 286, 251, 319, 283],
        "stn_in_txt": ["De Bilt", "De Kooy", "Groningen", "Heerde", "Hoofddorp", "Hoorn", "Roermond", "Ter Apel", "West-Terschelling", "Westdorpe", "Winterswijk"],
        "stn_data": ["De Bilt", "De Kooy", "Eelde", "Heino", "Schiphol", "Berkhout", "Arcen", "Nieuw Beerta", "Hoorn Terschilling", "Westdorpe", "Hupsel"],
        }
        stnxx = ["De Bilt", "De Kooy"]
        # Create a dictionary to map station names to STN values
        stn_dict = dict(zip(data["stn_data"], data["STN"]))

        # Create a dropdown menu with the station names
        #selected_stations = st.sidebar.multiselect("Select stations:", options=data["stn_data"], default=data["stn_data"])
        selected_stations = st.sidebar.multiselect("Select stations:", options=data["stn_data"], default=stnxx)
        if len(selected_stations)==0:
            st.error("Select at least one station")
            st.stop()
        # Map the selected names to their corresponding STN values
        stations = [stn_dict[name] for name in selected_stations]
    else:
        stations = [260,235,280,278,240,249,391,286,251,319,283]
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
    print (selected_stations)
    print (stations)
    df_master = get_dataframe_multiple_(stations, FROM, UNTIL)


    daily_avg_cumulative_neerslagtekort = df_master.groupby('YYYYMMDD')['cumulative_neerslagtekort'].mean().reset_index()
    df_master['year'] = df_master['YYYYMMDD'].dt.year
    
    # # Pivot and calculate statistics
    df_master["cumm_neerslag_etmaalsom"] = df_master.groupby(['STN', 'year'])["neerslag_etmaalsom"].cumsum()
    #df_grouped = df.groupby(['STN', 'year'])['value'].cumsum()
  

    make_spaggetti(df_master,  "cumulative_neerslagtekort")
    make_spaggetti(df_master,  "neerslag_etmaalsom")
    make_spaggetti(df_master,  "cumm_neerslag_etmaalsom")
    plot_daily_cumm_neerslagtekort(daily_avg_cumulative_neerslagtekort)
    plot_average_various_years(daily_avg_cumulative_neerslagtekort)
    max_value_each_year(daily_avg_cumulative_neerslagtekort)
    first_day_of_dryness(df_master)
    multiple_lineair_regression(df_master)
   
    show_stations()
def first_day_of_dryness(df_master):
    afkapwaarde = st.sidebar.number_input("Treshold",1,365, 150)
    df = (
        df_master[df_master.cumulative_neerslagtekort >= afkapwaarde]
        .groupby(['stn_data', 'year'])['YYYYMMDD']
        .min()
        .reset_index()
    )
    
    pivot_table = df.pivot(index='year', columns='stn_data', values='YYYYMMDD')
    # Calculate the mean across columns
    pivot_table['mean'] = pivot_table.mean(axis=1)
    fig = go.Figure()
    for column in pivot_table.columns:
        # Extract day of year
        pivot_table[column] = pivot_table[column].dt.dayofyear
        y_values = pivot_table[column]
        x_values = pivot_table.index

        fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name=str(column)))

    fig.update_layout(
        title=f'First day of dryness. Day 1 is 1st of Jan. Treshold = {afkapwaarde}',
        xaxis_title='Year',
        yaxis_title='Date')
    st.plotly_chart(fig)
    with st.expander("values"):
        st.write(pivot_table)
    
def max_value_each_year(df):
    # Create scatter plot with Plotly
    max_cumulative_neerslagtekort = df.groupby('year')['cumulative_neerslagtekort'].max()
    avg_cumulative_neerslagtekort = df.groupby('year')['cumulative_neerslagtekort'].mean()
    last_cumulative_neerslagtekort = df.groupby('year')['cumulative_neerslagtekort'].last()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=max_cumulative_neerslagtekort.index, y=max_cumulative_neerslagtekort.values, name='max',mode='lines'))
    fig.add_trace(go.Scatter(x=avg_cumulative_neerslagtekort.index, y=avg_cumulative_neerslagtekort.values,  name='avg',mode='lines'))
    fig.add_trace(go.Scatter(x=last_cumulative_neerslagtekort.index, y=last_cumulative_neerslagtekort.values,  name='last',mode='lines'))
                  
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Maximum and mean Cumulative Precipitation Deficit',
        title='Maximale, gemiddelde en laatste neerslagtekort over 11 meetstations'
    )

    st.plotly_chart(fig)

    
def plot_average_various_years(daily_avg_cumulative_neerslagtekort):
    daily_avg_cumulative_neerslagtekort['date_1900'] = pd.to_datetime(daily_avg_cumulative_neerslagtekort['YYYYMMDD'].dt.strftime('%d-%m-1900'), format='%d-%m-%Y')

    pivot_daily_avg_cumulative_neerslagtekort = daily_avg_cumulative_neerslagtekort.pivot(index='date_1900', columns='year', values='cumulative_neerslagtekort')
    # Create a line plot using Plotly
    fig = go.Figure()
    for column in pivot_daily_avg_cumulative_neerslagtekort.columns:
        fig.add_trace(go.Scatter(x=pivot_daily_avg_cumulative_neerslagtekort.index, y=pivot_daily_avg_cumulative_neerslagtekort[column], mode='lines', name=str(column)))

    fig.update_layout(
        title=f'Landelijk gemiddelde cumm. neerslagtekort over 11 stations / verschillende jaren',
        xaxis_title='Date',
        xaxis=dict(title="date",tickformat="%d-%m"),
        yaxis_title="Cum. neerslagtekort")
   
    st.plotly_chart(fig)

def plot_daily_cumm_neerslagtekort(daily_avg_cumulative_neerslagtekort):
    daily_avg_cumulative_neerslagtekort['year'] = daily_avg_cumulative_neerslagtekort['YYYYMMDD'].dt.year
     # Create a line plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_avg_cumulative_neerslagtekort['YYYYMMDD'], y=daily_avg_cumulative_neerslagtekort["cumulative_neerslagtekort"], mode='markers',  name="cumulative_neerslagtekort"))

    fig.update_layout(
        title='Landelijk gemiddelde cumm. neerslagtekort over 11 stations door de tijd heen',
        xaxis_title='Date',
        yaxis_title='Value')
    st.plotly_chart(fig)
    
def show_stations():
    # Define the data
    data = {
        "stationsnr":       [260, 235, 280, 278, 240, 249, None, None, 391, 286, 251, 319, 283],
        "genoemd in tekst": ["De Bilt", "De Kooy", "Groningen", "Heerde", "Hoofddorp", "Hoorn", "Kerkwerve", "Oudenbosch", "Roermond", "Ter Apel", "West-Terschelling", "Westdorpe", "Winterswijk"],
        "gebruikte data": ["De Bilt", "De Kooy", "Eelde", "Heino", "Schiphol", "Berkhout", "- NIET OPGENOMEN : (312 en 324 geven lege results)", "Niet opgenomen (340 heeft geen neerslagetmaalsom)", "Arcen", "Nieuw Beerta", "Hoorn Terschilling", "Westdorpe", "Hupsel"],
        
    }

    # Create the DataFrame
    df = pd.DataFrame(data)

    # Display the DataFrame
    st.subheader("Gebruikte weerstations")
    st.write(df)

def make_spaggetti(df_master, values):
    min_date = df_master['YYYYMMDD'].min()
    max_date = df_master['YYYYMMDD'].max()
    date_range = pd.date_range(min_date, max_date, freq='D')

    # Reindex the pivot table to include all the dates in the date range
    pivot_table = df_master.pivot(index='YYYYMMDD', columns='stn_in_txt', values=values)
    pivot_table = pivot_table.reindex(date_range)

    
    # Create a line plot using Plotly
    fig = go.Figure()
    for column in pivot_table.columns:

        y_values = pivot_table[column]
        x_values = pivot_table.index
        y_values[(x_values.month < 4) | (x_values.month > 9)] = np.nan
        
        # x_values = x_values[(x_values.month >= 4) & (x_values.month <= 9)]
        # y_values = y_values[(x_values.month >= 4) & (x_values.month <= 9)]
        fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name=str(column)))
                      


        # fig.add_trace(go.Scatter(x=pivot_table.index, y=pivot_table[column], mode='lines', name=str(column)))

    fig.update_layout(
        title=f'{values} from various stations',
        xaxis_title='Date',
        yaxis_title=values)
    st.plotly_chart(fig)


def multiple_lineair_regression(df):
    """Calculates multiple lineair regression. User can choose the Y value and the X values
    Args:
        df_ (df): df with info
    """    
    st.subheader("Multiple Lineair Regression")
    y_value_ = "eref" 
    x_values_options =  ["temp_avg","temp_max","glob_straling","neerslag_etmaalsom"]
    x_values_default =  ["temp_avg","temp_max","glob_straling","neerslag_etmaalsom"]
    x_values = st.multiselect("X values", x_values_options, x_values_default)
    intercept=  st.checkbox("Intercept", False)
    df =df[["STN","YYYYMMDD"]+[y_value_]+ x_values]
    x = df[x_values]
    y = df[y_value_]
    
    # with statsmodels
    if intercept:
        x= sm.add_constant(x) # adding a constant
    
    model = sm.OLS(y, x, missing='drop').fit()

    st.write("**OUTPUT ORDINARY LEAST SQUARES**")
    print_model = model.summary()
    st.write(print_model)

if __name__ == "__main__":
    #main()
    fromx = datetime.strptime("2000-01-01", "%Y-%m-%d").date()
    until = datetime.strptime("2023-12-31", "%Y-%m-%d").date()
    neerslagtekort_meerdere_stations(fromx, until)
    