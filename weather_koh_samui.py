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
import plotly
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import platform
import os

# when using without Streamlit, to avoid 127.0.0.1 refused to connect :
# plotly.offline.init_notebook_mode(connected=True)
    
def read_ogimet():
    """Read the data from Ogimet and save it to a CSV file. Reading the data while running the script every time
       is not encouraged, see above.
    """

    # find station codes here https://www.ogimet.com/indicativos.phtml.en
    # station_code,location_str = "485500-99999", "Koh_Samui"  
                                   
    # station_code,location_str = "16242","Rome Fiumicino"
    station_code,location_str = "48327","Chiang_mai"
    
    start_date = datetime(2000, 1, 1)
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
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html5lib")
        temp_table = pd.read_html(str(soup.find_all("table")[3]), encoding="utf-8")[0]
        print(temp_table)
        if temp_table.empty:
            continue

        temp_table = temp_table.iloc[2:]  # Remove the first two rows, which usually contains units

        date_vec = pd.date_range(end=request_date - timedelta(days=1), periods=len(temp_table), freq="1D")
        temp_table["Date"] = date_vec
        observations = pd.concat([temp_table, observations])
        counter += 1

    # observations = observations[observations["Date"] <= end_date] # gives an error. I just delete the last rows in the CSV file

    observations.to_csv(f"irbid_weather_{location_str}_2023.csv", index=False)
    # You have to replace ---- with [nothing]. (Don't use [None], since it will turn the column into a text/object column) 
    print(observations)

def main():
    """Show the data from Ogimet in a graph, and average values per month per year
    """    
    where = st.sidebar.selectbox("Location to show", ["Koh Samui", "Chiang Mai", "Rome Fiumicino"]
                         )
    st.title(f"Weather info from {where}")
    
    
    df_ = get_data(where)

    to_show = st.sidebar.selectbox("What to show x", ["T_Max","T_Min","T_Mean","Hr_Med","Wind_Max","Wind_Mean","SLP","STN","Vis","Prec","Diary"],0)
    window_size =  st.sidebar.slider("Window for SMA",1,365,7) 
    y_axis_zero = st.sidebar.selectbox("Y axis start at zero", [True,False],1)
    multiply_minus_one = st.sidebar.selectbox("Multiply by -1", [True,False],1)
    
    
    if multiply_minus_one:
        # Make a copy of the DataFrame without the "Date" column
        df_copy = df_.drop(columns=['Date']).copy()

        # Multiply all values by -1
        df_copy = df_copy * -1

        # Combine the "Date" column back with the modified values
        df = pd.concat([df_['Date'], df_copy], axis=1)
    else:
        df = df_


    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df = df.sort_values(by='Date')
  
   
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
    
    # CROSS TABLE WITH MONTLY AVERAGES
    st.subheader (f"Monthly averages of {to_show} - {where}")
    crosstable = pd.pivot_table(df, values=to_show, index='Month', columns='Year', aggfunc='mean').round(1)
    st.write (crosstable)

    st.subheader (f"Monthly averages of {to_show} - {where}")
    # Create the heatmap using Plotly Express
    fig = px.imshow(crosstable)
    #fig.show()
    st.plotly_chart(fig)
    
    # SHOW MONTLY AVERAGES THROUGH THE YEARS
    transposed_df = crosstable.T
    fig_x = go.Figure()

    for column in transposed_df.columns:
        fig_x.add_trace(go.Scatter(x=transposed_df.index, y=transposed_df[column], mode='lines', name=column))

    fig_x.update_layout(title=f'Monthly averages of {to_show} through time - {where}',
                    xaxis_title='Years',
                    yaxis_title=f'Montly average of {to_show}')
    if y_axis_zero:
        fig_x.update_layout(yaxis_range=[0, max(transposed_df.max())])


    st.plotly_chart(fig_x)

    treshold_value = st.sidebar.number_input("Treshold value (incl.)")
    above_under = st.sidebar.selectbox("Above or below", ["above", "equal", "below"],0)

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
    
    st.subheader(f"Numbers of days per month that {to_show} was {au_txt} {treshold_value} - {where}")
    # Create a pivot table to count the occurrences of temperatures above 30 degrees per month and year
    table = pd.pivot_table(df_above_30, values=to_show, index='Month', columns='Year', aggfunc='count', fill_value=0)
    all_months = range(1, 13)
    all_years = df['Year'].unique()
    table = table.reindex(index=all_months, columns=all_years, fill_value=0)
    st.write(table)

    st.subheader(f"Numbers of days per month that {to_show} was {au_txt} {treshold_value} - {where}")
    fig = px.imshow(table)
    #fig.show()
    st.plotly_chart(fig)

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
@st.cache_data(ttl=24*60*60)
def get_data(where):
    load_local = True if platform.processor() else False


    # Define the base directory where the CSV files are stored
    base_dir = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input"
    github_base_url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input"

    # Map locations to their respective CSV files
    locations = {
        "Koh Samui": "weather_ko_samui.csv",
        "Chiang Mai": "weather_chiang_mai.csv",
        "Rome Fiumicino": "weather_rome_fiumicino.csv"
    }

    # Check if the 'where' value is valid
    if where not in locations:
        st.error("Error in WHERE")
        st.stop()

    # Build the URL based on the location
    url = os.path.join(base_dir, locations[where]) if load_local else f"{github_base_url}/{locations[where]}"
    df_ = pd.read_csv(url)
    return df_


if __name__ == "__main__":
    #read_ogimet()
    main()
    