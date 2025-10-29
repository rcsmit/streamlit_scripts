# OLD USE WEATHER_OPEN_METEO.PY

# Scrapping Weather Data from Ogimet
#
# Based on this R-script
# https://www.kaggle.com/code/ahmadabuhamour/scrapping-weather-data-from-ogimet
# converted to python by chatGPT
# adapted by Rene Smit @rcsmit

# To install the needed packages : 
# pip install pandas requests beautifulsoup4 html5lib

# "Ogimet is a simple website and doesn't have great processing capabilities,
# such scrapping algorithim could take up to 15 mins to run if requesting 20+ years of data.
# Hence, in order not to overwhelm Ogimet's servers, I encourage you to use it only as a
# last resort for large amounts of data.""
#
# THIS IS THE REASON THAT THE SCRAPING FUNCTION IS NOT INTEGRATED IN THE SHOW-FUNCTION


import pandas as pd
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import platform
import os
from io import StringIO
import matplotlib.pyplot as plt
from show_knmi_functions.show_calender_heatmap import show_calender_heatmap
from show_knmi_functions.show_year_heatmap import show_year_heatmap

#from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.colors import ListedColormap
#_lock = RendererAgg.lock
import numpy as np
# when using without Streamlit, to avoid 127.0.0.1 refused to connect :
# plotly.offline.init_notebook_mode(connected=True)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import datetime as dt


def read_ogimet():
    """Read the data from Ogimet and save it to a CSV file. Reading the data while running the script every time
       is not encouraged, see above.
    """


    # find station codes here https://www.ogimet.com/indicativos.phtml.en
    station_code,location_str = "485500-99999", "ko_samui"  
                                   
    # station_code,location_str = "16242","Rome Fiumicino"
    #station_code,location_str = "48327","chiang_mai"
    #station_code,location_str = "16105","Venezia"
    station_code,location_str = "72202","Miami"
    start_date = datetime(2015, 1,1 )
    #start_date = datetime(1900, 1, 1)

    end_date = datetime(2019, 12, 31)
    end_date = datetime.today()  # You could use the desired end date
    number_of_days = (end_date - start_date).days 
    batches = int(number_of_days / 50)+1 # number of batches
    counter = 1
    request_date = start_date

    observations = pd.DataFrame() 

    while request_date < end_date:
        request_date += timedelta(days=50)

        year = request_date.year
        month = request_date.month
        day = request_date.day

        #url = f"https://www.ogimet.com/cgi-bin/gsynres?lang=en&ind={station_code}&ndays=50&ano={year}&mes={month}&day={day}&hora=06&ord=REV&Send=Send"
        # https://www.ogimet.com/cgi-bin/gsynres?lang=en&ord=REV&ndays=30&ano=2023&mes=07&day=24&hora=06&ind=48550 seems to work # Daily summary at 12:00 UTC
        # url = f"https://www.ogimet.com/cgi-bin/gsynres?lang=en&ord=REV&ndays=50&ano={year}&mes={month}&day={day}&hora=12&ind={station_code}" soup.find_all("table")[2]

        url = f"https://ogimet.com/cgi-bin/gsodres?lang=en&ind={station_code}&ord=DIR&ano={year}&mes={month}&day={day}&ndays=50" # Global Summary Of the Day (GSOD), is some days behind # soup.find_all("table")[3]
        print (f"Retreiving {url} {counter} / {batches}")
        st.write(f"Retreiving {url} {counter} / {batches}")
        counter += 1
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html5lib")
        #temp_table = pd.read_html(str(soup.find_all("table")[3]), encoding="utf-8")[0]


        # Assuming soup is already defined and contains your parsed HTML
        html_string = str(soup.find_all("table")[3])
        html_io = StringIO(html_string)

        # Now pass the StringIO object to read_html
        temp_table = pd.read_html(html_io, encoding="utf-8")[0]

        
        if temp_table.empty:
            
            continue
        else:
            print(temp_table)

        #temp_table = temp_table.iloc[1:]  # Remove the first two rows, which usually contains units

        date_vec = pd.date_range(end=request_date - timedelta(days=1), periods=len(temp_table), freq="1D")
        temp_table["Date"] = date_vec
        observations = pd.concat([temp_table, observations])
        

    # observations = observations[observations["Date"] <= end_date] # gives an error. I just delete the last rows in the CSV file
    observations = observations.sort_values(by=('Date','Date'))
    observations.to_csv(f"weather_{location_str}_b.csv", index=False)
    # You have to replace ---- with [nothing]. (Don't use [None], since it will turn the column into a text/object column) 
    print(observations)

def show_warmingstripes(df_, to_show, where):
    """_summary_

    # Based on code of Sebastian Beyer
    # https://github.com/sebastianbeyer/warmingstripes/blob/master/warmingstripes.py

    # the colors in this colormap come from http://colorbrewer2.org
    # the 8 more saturated colors from the 9 blues / 9 reds
    # https://matplotlib.org/matplotblog/posts/warming-stripes/
    

    Args:
        df_ (_type_): _description_
        to_show (_type_): _description_
        where (_type_): _description_
    """ 
    
    df = df_.groupby(df_["Year"], sort=True).mean(numeric_only = True).reset_index()
    #df_grouped = df.groupby([df[valuefield]], sort=True).sum().reset_index()
    

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
    temperatures = df[to_show].tolist()
    stacked_temps = np.stack((temperatures, temperatures))
    #with _lock:
    # plt.figure(figsize=(4,18))
    fig, ax = plt.subplots()

    fig = ax.imshow(
        stacked_temps,
        cmap=cmap,
        aspect=40,
    )
    plt.gca().set_axis_off()

    plt.title(f"{to_show} in {where}")
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.show()
    # st.pyplot(fig) - gives an error
    #st.set_option("deprecation.showPyplotGlobalUse", False)
    st.pyplot()
def show_month(df, to_show, day_min, day_max, month, month_names, where):
    """Show graph with to_show in different lines for each years for a certain (period of) a month
       Show a frequency table of this data
       Show a histogram of this data 

    Args:
        df (_type_): _description_
        to_show (_type_): _description_
        day_min (_type_): _description_
        day_max (_type_): _description_
        month (_type_): _description_
        month_names (_type_): _description_
        where (_type_): _description_
    """   
    
    df_month = df[(df['Month'] == month) & (df['Day']>=day_min) & (df['Day']<=day_max)]             
    fig = px.line(df_month, x='Day', y=to_show, color='Year', labels={'temp': 'Temperature (°C)'}, title=f'{to_show} for {month_names[month-1]} in {where}')
    st.plotly_chart(fig)

    frequency_table = df_month[to_show].value_counts().reset_index()
    frequency_table.columns = [to_show, 'Frequency']
    # Sort the frequency table by the variable 'to_show'
    frequency_table = frequency_table.sort_values(by=to_show, ascending=False)
    # Calculate cumulative absolute frequency
    frequency_table['Cumulative Absolute Frequency'] = frequency_table['Frequency'].cumsum()

    # Calculate cumulative percentage
    total_absolute_frequency = frequency_table['Frequency'].sum()
    frequency_table['Cumulative Percentage'] = (frequency_table['Cumulative Absolute Frequency'] / total_absolute_frequency) * 100

    # Display the frequency table
    st.write(frequency_table)

    # Histogram using Plotly Express
    fig = px.histogram(df_month, x=to_show, title=f'Histogram of {to_show}', labels={to_show: f'{to_show}', 'count': 'Frequency'})
    st.plotly_chart(fig)

    # Frequency table
    df_month['Temp_Bin'] = pd.cut(df_month[to_show], bins=range(int(df_month[to_show].min()), int(df_month[to_show].max()) + 2))
    frequency_table = df_month.pivot_table(index='Temp_Bin', columns='Year', aggfunc='size', fill_value=0)

    # Reset the index for plotting
    frequency_table = frequency_table.reset_index()
    frequency_table['Temp_Bin'] = frequency_table['Temp_Bin'].apply(lambda x: x.mid)

    # Display the frequency table
    st.write(frequency_table)

    # Frequency line graph
    frequency_table_melted = frequency_table.melt(id_vars='Temp_Bin', var_name='Year', value_name='Frequency')
    fig = px.line(frequency_table_melted, x='Temp_Bin', y='Frequency', color='Year',
                  labels={'Temp_Bin': 'Temperature (°C)', 'Frequency': 'Frequency'},
                  title=f'Frequency of {to_show} for {month_names[month-1]} in {where}')
    st.plotly_chart(fig)
    nbins = len(range(int(df_month[to_show].min()), int(df_month[to_show].max()) + 2))
    # Histogram of the to_show variable
    fig = px.histogram(df_month, x=to_show, nbins=nbins, title=f'Histogram of {to_show}', 
                       labels={to_show: f'{to_show}', 'count': 'Frequency'})
    st.plotly_chart(fig)


    #show_warmingstripes(df_month, to_show, where)

def cross_table_montly_avg(df, to_show, where, y_axis_zero):  
    """_summary_

    Args:
        df (_type_): _description_
        to_show (_type_): _description_
        where (_type_): _description_
        y_axis_zero (_type_): _description_
    """
    # CROSS TABLE WITH MONTLY AVERAGES
    st.subheader (f"Monthly averages of {to_show} - {where}")
    crosstable = pd.pivot_table(df, values=to_show, index='Month', columns='Year', aggfunc='mean').round(1)
    st.write (crosstable)

    st.subheader (f"Monthly averages of {to_show} - {where}")
    # Create the heatmap using Plotly Express
    fig = px.imshow(crosstable)
    #fig.show()
    st.plotly_chart(fig)

    #  # SHOW MONTLY AVERAGES THROUGH THE YEARS
    # transposed_df = crosstable.T
    # fig_x = go.Figure()

    # for column in transposed_df.columns:
    #     fig_x.add_trace(go.Scatter(x=transposed_df.index, y=transposed_df[column], mode='lines', name=column))

    # fig_x.update_layout(title=f'Monthly averages of {to_show} through time - {where}',
    #                 xaxis_title='Years',
    #                 yaxis_title=f'Montly average of {to_show}')
    # if y_axis_zero:
    #     fig_x.update_layout(yaxis_range=[0, max(transposed_df.max())])
    # st.plotly_chart(fig_x)

    transposed_df = crosstable.T

    # Initialize a new Plotly figure
    fig_x = go.Figure()

    # Create a scatter plot with a trendline for each month
    for column in transposed_df.columns:
        # Extract the data for the current month
        x = transposed_df.index.values.reshape(-1, 1)  # Years
        y = transposed_df[column].values  # Monthly averages


        mask = ~np.isnan(y)
        x_filtered = x[mask].reshape(-1, 1)
        y_filtered = y[mask]
        
        # Create scatter plot for the current month
        fig_x.add_trace(go.Scatter(x=x_filtered.flatten(), y=y_filtered, mode='markers', name=f'{column} Scatter'))

        # Fit a linear regression model
        if len(y_filtered) > 1:  # Ensure there are enough data points to fit a model
            model = LinearRegression()
            model.fit(x_filtered, y_filtered)
            trendline = model.predict(x_filtered)

            # Calculate the equation and R² value
            slope = model.coef_[0]
            intercept = model.intercept_
            r_squared = r2_score(y_filtered, trendline)

            # Add the trendline to the plot
            fig_x.add_trace(go.Scatter(x=x_filtered.flatten(), y=trendline, mode='lines', name=f'{column} Trendline'))

            # Add annotations for the equation and R² value
            equation_text = f'y = {slope:.2f}x + {intercept:.2f} | R² = {r_squared:.2f}'
            st.write(f"{column} - {equation_text}")
            #fig_x.add_annotation(x=x_filtered.flatten()[-1], y=trendline[-1], text=equation_text, showarrow=True, arrowhead=1)

    # Update layout
    fig_x.update_layout(
        title=f'Monthly averages of {to_show} through time - {where}',
        xaxis_title='Years',
        yaxis_title=f'Monthly average of {to_show}'
    )

    if y_axis_zero:
        fig_x.update_layout(yaxis_range=[0, max(transposed_df.max())])

    # Display the plot using Streamlit
    st.plotly_chart(fig_x)
def show_info():
    st.info("Source weather info: https://ogimet.com/")

    ''''
    Data until mid july 2023, not automaticall updated
    T_max = Max. temperature taken from explicit Tmax. report (°C)
    T_Min = Min. temperature taken from explicit Tmax. report (°C)
    T_Mean = Mean temperature derived from 8 observations (°C)
    Hr_Med = Mean relative humidity derived from 8 dew point observations (%)
    Wind_Max = Max wind speed computed from 8 observations (km/h)
    Wind_Mean =Mean wind speed computed from 8 observations (km/h) 
    SLP = Mean sea level pressure computed from 8 observations (mb)
    STN = Mean pressure at station level computed from 8 observations (mb)
    Vis = Mean visibility computed from 8 observations (km)
    Prec = Prec. computed from 1 report of 24-hour precipitation amount (mm)
    Diary = (images, not in use)
    Source https://ogimet.com/cgi-bin/gsodres?lang=en&ind=485500-99999&ord=DIR&ano=2000&mes=01&day=1&ndays=500
    '''

def show_treshold(where, to_show, treshold_value, above_under, df):
    """_summary_

    Args:
        where (_type_): _description_
        to_show (_type_): _description_
        treshold_value (_type_): _description_
        above_under (_type_): _description_
        df (_type_): _description_
    """    
    if above_under =="above":
    # Filter the DataFrame to include only the rows where Temperature is above 30 degrees
        df_above_30 = df[df[to_show] >= treshold_value]
        au_txt = ">="
    elif above_under =="equal":
        df_above_30 = df[df[to_show] == treshold_value]
        au_txt = "="
    elif above_under =="above":
        df_above_30 = df[df[to_show] <= treshold_value]
        au_txt = "<="
    else:
        st.error("ERROR")
        st.stop()
    
    # st.subheader(f"Numbers of days per month that {to_show} was {au_txt} {treshold_value} - {where}")
    # # Create a pivot table to count the occurrences of temperatures above 30 degrees per month and year
    # table = pd.pivot_table(df_above_30, values=to_show, index='Month', columns='Year', aggfunc='count', fill_value=0)
    # all_months = range(1, 13)
    # all_years = df['Year'].unique()
    # table = table.reindex(index=all_months, columns=all_years, fill_value=0)
    # st.write(table)


    # Generate the subheader
    st.subheader(f"Numbers of days per month that {to_show} was {au_txt} {treshold_value} - {where}")

    # Create a pivot table to count the occurrences of temperatures above the threshold per month and year
    table = pd.pivot_table(df_above_30, values=to_show, index='Month', columns='Year', aggfunc='count', fill_value=0)
    all_months = range(1, 13)
    all_years = df['Year'].unique()
    table = table.reindex(index=all_months, columns=all_years, fill_value=0)
    st.write(table)

    # Transpose the table for easier plotting
    transposed_table = table.T

    # Initialize a new Plotly figure
    fig_x = go.Figure()

    # Create a scatter plot with a trendline for each month
    for month in transposed_table.columns:
        # Extract the data for the current month and filter out NaN values
        x = transposed_table.index.values
        y = transposed_table[month].values

        mask = ~np.isnan(y)
        x_filtered = x[mask].reshape(-1, 1)
        y_filtered = y[mask]

        # Create scatter plot for the current month
        fig_x.add_trace(go.Scatter(x=x_filtered.flatten(), y=y_filtered, mode='markers', name=f'Month {month} Scatter'))

        # Fit a linear regression model
        if len(y_filtered) > 1:  # Ensure there are enough data points to fit a model
            model = LinearRegression()
            model.fit(x_filtered, y_filtered)
            trendline = model.predict(x_filtered)

            # Calculate the equation and R² value
            slope = model.coef_[0]
            intercept = model.intercept_
            r_squared = r2_score(y_filtered, trendline)

            # Add the trendline to the plot
            fig_x.add_trace(go.Scatter(x=x_filtered.flatten(), y=trendline, mode='lines', name=f'Month {month} Trendline'))

            # Add annotations for the equation and R² value
            equation_text = f'y = {slope:.2f}x + {intercept:.2f} | R² = {r_squared:.2f}'
            st.write(f"{month} - {equation_text}")
            
            # fig_x.add_annotation(x=x_filtered.flatten()[-1], y=trendline[-1], text=equation_text, showarrow=True, arrowhead=1)

    # Update layout
    fig_x.update_layout(
        title=f'Numbers of days per month that {to_show} was {au_txt} {treshold_value} - {where}',
        xaxis_title='Years',
        yaxis_title=f'Number of days {to_show} was {au_txt} {treshold_value}'
    )

    # Display the plot using Streamlit
    st.plotly_chart(fig_x)

    st.subheader(f"Numbers of days per month that {to_show} was {au_txt} {treshold_value} - {where}")
    fig = px.imshow(table)
    #fig.show()
    st.plotly_chart(fig)

def line_graph(to_show, window_size, y_axis_zero, df):
    """_summary_

    Args:
        to_show (_type_): _description_
        window_size (_type_): _description_
        y_axis_zero (_type_): _description_
        df (_type_): _description_
    """    
    df[f'{to_show}_SMA'] = df[to_show].rolling(window=window_size).mean()

    fig = px.line(df, x='Date', y=[to_show, f'{to_show}_SMA'],
                title=f'{to_show} over Time with SMA',
                labels={'value':to_show},
                line_shape='linear')
    # Set the range of the y-axis to start from 0
    if y_axis_zero:
        fig.update_layout(yaxis_range=[0, max(df[to_show])])

    # fig.show()
    # plotly.offline.plot(fig)
    st.plotly_chart(fig)
#@st.cache_data(ttl=24*60*60)
def get_data(where):
    # load_local = True if platform.processor() else False
    load_local = False

    # Define the base directory where the CSV files are stored
    base_dir = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input"
    github_base_url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input"

    # Map locations to their respective CSV files
    locations = {
        "Koh Samui": "weather_ko_samui",
        "Chiang Mai": "weather_chiang_mai",
        "Rome Fiumicino": "weather_rome_fiumicino",
        "Venezia": "weather_venezia"
    }

    # Check if the 'where' value is valid
    if where not in locations:
        st.error("Error in WHERE")
        st.stop()

    # Build the URL based on the location
    url = os.path.join(base_dir, locations[where]) if load_local else f"{github_base_url}/{locations[where]}.csv"
    url2 = os.path.join(base_dir, locations[where]) if load_local else f"{github_base_url}/{locations[where]}_b.csv"
    
    st.write(url)
    df_ = pd.read_csv(url)
   
    try:
        df_2 = pd.read_csv(url2)
        
    except:
        df_2 = pd.DataFrame()  # Create an empty DataFrame if the second file is not found

    # Align columns of df_2 to match df_, filling missing columns with None
    df_2 = df_2.reindex(columns=df_.columns, fill_value=None)

    # Concatenate the DataFrames
    df_combined = pd.concat([df_, df_2], ignore_index=True)



    return df_combined



# Function to convert Celsius to Fahrenheit
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

# Function to convert Fahrenheit to Celsius
def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9

# Function to calculate Heat Index
def calculate_heat_index(T, RH):
    # Formula for heat index calculation in Fahrenheit
    HI = (-42.379 + 2.04901523 * T + 10.14333127 * RH 
          - 0.22475541 * T * RH - 0.00683783 * T**2 
          - 0.05481717 * RH**2 + 0.00122874 * T**2 * RH 
          + 0.00085282 * T * RH**2 - 0.00000199 * T**2 * RH**2)
    return HI

# Function to calculate Wind Chill
def calculate_wind_chill(T, V):
    # Formula for wind chill calculation in Fahrenheit
    WC = 35.74 + 0.6215 * T - 35.75 * (V**0.16) + 0.4275 * T * (V**0.16)
    return WC

# Function to determine the feels-like temperature
def feels_like_temperature(row):
    T_C = row['T_Mean']
    RH = row['Hr_Med']
    V_mph = float(row['Wind_Max']) * 0.621371  # Converting km/h to mph
    
    T_F = celsius_to_fahrenheit(T_C)
    
    if T_F >= 80:
        # Calculate Heat Index
        feels_like_F = calculate_heat_index(T_F, RH)
    elif T_F <= 50 and V_mph >= 3:
        # Calculate Wind Chill
        feels_like_F = calculate_wind_chill(T_F, V_mph)
    else:
        feels_like_F = T_F  # No adjustment
    
    feels_like_C = fahrenheit_to_celsius(feels_like_F)
    return feels_like_C


def check_from_until(from_, until_):
    """Checks whether the start- and enddate are valid.

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

def main():
    """Show the data from Ogimet in a graph, and average values per month per year
    """    
                         

    start_ = "2019-01-01"
    today = datetime.today().strftime("%Y-%m-%d")
    from_ = st.sidebar.text_input("startdatum (yyyy-mm-dd) from 1-1-1900", start_)
    until_ = st.sidebar.text_input("enddatum (yyyy-mm-dd)", today)
    FROM, UNTIL = check_from_until(from_, until_)
    # Convert FROM and UNTIL to datetime
    FROM = pd.to_datetime(FROM)
    UNTIL = pd.to_datetime(UNTIL)
    where = st.sidebar.selectbox("Location to show", ["Koh Samui", "Chiang Mai", "Rome Fiumicino","Venezia"])
    to_show = st.sidebar.selectbox("What to show x", ["T_Max","T_Min","T_Mean","Hr_Med","Wind_Max","Wind_Mean","SLP","STN","Vis","Prec","Diary", "Feels_Like"],0)
    window_size =  st.sidebar.slider("Window for SMA",1,365,7) 
    y_axis_zero = st.sidebar.selectbox("Y axis start at zero", [True,False],1)
    multiply_minus_one = st.sidebar.selectbox("Multiply by -1", [True,False],1)
    treshold_value = st.sidebar.number_input("Treshold value (incl.)")
    above_under = st.sidebar.selectbox("Above or below", ["above", "equal", "below"],0)
    percentile_colomap_max = st.sidebar.number_input("percentile_colomap_max",1,100,100)
    
    month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    month = month_names.index(st.sidebar.selectbox("Month", month_names, index=0)) + 1
    day_min, day_max  = st.sidebar.slider("days",1,31,(1,31))
    df_ = get_data(where)
   
    if multiply_minus_one:
        # needed for for ex. visability 
        # Make a copy of the DataFrame without the "Date" column
        df_copy = df_.drop(columns=['Date']).copy()

        # Multiply all values by -1
        df_copy = df_copy * -1

        # Combine the "Date" column back with the modified values
        df = pd.concat([df_['Date'], df_copy], axis=1)
    else:
        df = df_
    df['Date'] = pd.to_datetime(df['Date'])

    # Filter the DataFrame based on the start and end dates
    df = df[(df['Date'] >= FROM) & (df['Date'] <= UNTIL)]



    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df = df.sort_values(by='Date')

    
    df["Date"] = pd.to_datetime(df["Date"].astype(str))
    df["YYYY"] = df["Date"].dt.year
    df["MM"] = df["Date"].dt.month
    df["DD"] = df["Date"].dt.day
    
    # Convert all columns except 'Date' to appropriate data types
    df[df.columns.difference(['Date'])] = df[df.columns.difference(['Date'])].apply(pd.to_numeric, errors='coerce')
    # Apply the feels_like_temperature function to each row in the DataFrame
    df['Feels_Like'] = df.apply(feels_like_temperature, axis=1)

    
    st.title(f"Weather info from {where}")
    st.write(df)
    line_graph(to_show, window_size, y_axis_zero, df)
    show_calender_heatmap(df,"Date", [to_show], percentile_colomap_max)
    show_year_heatmap(df,"Date", [to_show])
    
    cross_table_montly_avg(df, to_show, where, y_axis_zero)   
    show_treshold(where, to_show, treshold_value, above_under, df)
    show_warmingstripes(df, to_show, where) 
    show_month(df, to_show, day_min, day_max,month, month_names,where)
    show_info()

if __name__ == "__main__":
    #read_ogimet()
    main()
    