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

import streamlit as st

# to avoid 127.0.0.1 refused to connect :
# plotly.offline.init_notebook_mode(connected=True)
    
def read_ogimet():
    """Read the data from Ogimet and save it to a CSV file. Reading the data while running the script every time
       is not encouraged, see above.
    """
    station_code = "485500-99999"   # this WMO code is for a station in Koh Samui, Thailand
                                    # find station codes here https://www.ogimet.com/indicativos.phtml.en
    
    start_date = datetime(2023, 5, 28)
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
        url = f"https://ogimet.com/cgi-bin/gsodres?lang=en&ind={station_code}&ord=DIR&ano={year}&mes={month}&day={day}&ndays=50"
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

    observations.to_csv("irbid_weather_ko_samui_2023.csv", index=False)
    print(observations)

def main():
    """Show the data from Ogimet in a graph, and average values per month per year
    """    
    st.title("Weather info from Koh Samui")
    url = r"C:\Users\rcxsm\Documents\python_scripts\weather_ko_samui.csv"
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/weather_ko_samui.csv"
    # 
    # Step 1: Read the CSV file into a DataFrame
    df = pd.read_csv(url)

    # Step 2: Convert the 'Date' column to datetime type (if it's not already in datetime format)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    df = df.sort_values(by='Date')
  
    # Step 4: Calculate the Simple Moving Average (SMA) of 'T_max' using pandas rolling mean
    to_show = st.sidebar.selectbox("What to show", ["T_Max","T_Min","T_Mean","Hr_Med","Wind_Max","Wind_Mean","SLP","STN","Vis","Prec","Diary"],0)
    window_size = st.sidebar.slider("Window for SMA",1,365,7) 
     # Change this to adjust the number of days for the SMA window
    df[f'{to_show}_SMA'] = df[to_show].rolling(window=window_size).mean()

    # Step 5: Create the Plotly plot
    fig = px.line(df, x='Date', y=[to_show, f'{to_show}_SMA'],
                title=f'{to_show} over Time with SMA',
                labels={'value':to_show},
                line_shape='linear')

    # Step 4: Show the plot
    # fig.show()
    #plotly.offline.plot(fig)
    st.plotly_chart(fig)
    # Step 4: Use pivot_table to create the crosstable with the average of rainfall per month
    crosstable = pd.pivot_table(df, values=to_show, index='Month', columns='Year', aggfunc='mean').round(1)

    # Step 5: Display the crosstable
    st.write (crosstable)

        
   
    # Step 5: Create the heatmap using Plotly Express
    fig = px.imshow(crosstable)

    # Step 6: Show the plot
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


if __name__ == "__main__":
    # read_ogimet()
    main()
    