import pandas as pd
import streamlit as st
from plotly_calplot import calplot
from weather_koh_samui import get_data
import plotly.express as px
from datetime import datetime, timedelta
import requests


st.set_page_config(layout="wide")

def show_calender_heatmap(df,what_to_show="pm25"):
    """_summary_

    Args:
        df (_type_): _description_
        what_to_show (str, optional):  [pm25|pm10|o3|no2|so2|co]. Defaults to "pm25".
    """    
    # https://python.plainenglish.io/interactive-calendar-heatmaps-with-plotly-the-easieast-way-youll-find-5fc322125db7
    # creating the plot
    #colorscales = ["aggrnyl", "agsunset", "blackbody", "bluered", "blues", "blugrn", "bluyl", "brwnyl", "bugn", "bupu", "burg", "burgyl", "cividis", "darkmint", "electric", "emrld", "gnbu", "greens", "greys", "hot", "inferno", "jet", "magenta", "magma", "mint", "orrd", "oranges", "oryel", "peach", "pinkyl", "plasma", "plotly3", "pubu", "pubugn", "purd", "purp", "purples", "purpor", "rainbow", "rdbu", "rdpu", "redor", "reds", "sunset", "sunsetdark", "teal", "tealgrn", "turbo", "viridis", "ylgn", "ylgnbu", "ylorbr", "ylorrd", "algae", "amp", "deep", "dense", "gray", "haline", "ice", "matter", "solar", "speed", "tempo", "thermal", "turbid", "armyrose", "brbg", "earth", "fall", "geyser", "prgn", "piyg", "picnic", "portland", "puor", "rdgy", "rdylbu", "rdylgn", "spectral", "tealrose", "temps", "tropic", "balance", "curl", "delta", "oxy", "edge", "hsv", "icefire", "phase", "twilight", "mrybm", "mygbm"]
    # https://plotly.com/python/builtin-colorscales/
    print(df)
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
    df=df.sort_values(by="date")
    
    years = df["date"].dt.year.unique()
    df["Vis"]=df["Vis"]*-1
    # Loop through each year and each what_to_show value
    for year in years:
        df_year = df[df["date"].dt.year == year]  
     
        if df_year[what_to_show].sum() > 0: #dont show empty years
                
            # Assuming df[what_to_show] contains the values you want to process

            # Fill missing values with interpolation   
            #df_year[what_to_show]=df_year[what_to_show].fillna(0)
                

            def normalize(value):
                # normalizes the value to the nearest 25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 1000
                # problem is that the original value isnt show as mouseover
                for n in[25,50,75,100,125,150,175,200,300,400,1000]:
                    if value < n:
                        return n
                return 1000
            col1,col2=st.columns(2)
            with col1:
            #df_year[what_to_show] = df_year[what_to_show].apply(normalize)
                fig = calplot(
                        df_year,
                        x="date",
                        y=what_to_show,
                        years_title=True,
                        name=what_to_show,
                        cmap_min=0,
                        cmap_max=1000,
                
                        # Color scale from https://aqicn.org/historical/#city:chiang-mai
                        colorscale=[(0.00,      "#00787e"), 
                                        (0.025, "#059a65"),
                                        (0.05,  "#85bd4b"),
                                        (0.075, "#ffdd33"),
                                        (0.1,   "#ffba33"),
                                        (0.125, "#fe9633"),
                                        (0.150, "#e44933"),
                                        (0.175, "#ca0035"),
                                        (0.2,   "#970068"),
                                        (0.3,   "#78003f"),
                                        (0.4,   "#4e0016"),
                                        (1,   "#4e0016")],
                    
                        gap=2,
                        month_lines_width=2,
                        month_lines_color="black",
                        space_between_plots=0.15
                )
                st.plotly_chart(fig)
            with col2:
                fig2 = calplot(
                        df_year,
                        x="date",
                        y="Vis",
                        years_title=True,
                        name="Visibility",
                        colorscale="blues",
                        gap=2,
                        month_lines_width=2,
                        month_lines_color="black",
                        space_between_plots=0.15
                )
                st.plotly_chart(fig2)



@st.cache_data()
def get_data_open_meteo():
    """Get the weather data for a specific location and date range from Open Meteo archive API.
    https://open-meteo.com/en/docs/historical-weather-api?daily=rain_sum,temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_hours,precipitation_sum,visibility_mean,visibility_min,visibility_max,relative_humidity_2m_mean,relative_humidity_2m_max,relative_humidity_2m_min,&latitude=9.755106899960907&longitude=99.9609068&start_date=2025-01-01&end_date=2025-07-15&hourly=&timezone=Asia%2FBangkok
    Args:
        location_name(str):Location
        start_date(str): start date yyyy-mm-dd
        end_date(str): end date yyyy-mm-dd
    returns:
        pd.DataFrame: DataFrame with weather data including date, temperature, rain, humidity, visibility, etc.
    """
    

    # start_date = "2025-01-01"
    # end_date = "2025-07-15"
    start = "2015-01-01"
    today = datetime.today().strftime("%Y-%m-%d")
   
    daily_vars = ",".join([
      
        "visibility_mean",
        "visibility_min",
        "visibility_max",
      
    ])

    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude=18.7931784"
        f"&longitude=98.9774429"
        f"&start_date={start}"
        f"&end_date={today}"
        f"&daily={daily_vars}"
        f"&timezone=Asia%2FBangkok"
   
    )


    print(f"Fetching weather for Chiang Mai: {url}")
    response = requests.get(url)
    response.raise_for_status()
    data = response.json().get("daily")
    if not data or "time" not in data:
        raise KeyError("Invalid weather API response structure")
    
    df = pd.DataFrame({
        "date": pd.to_datetime(data["time"]),
        "Date": pd.to_datetime(data["time"]), #some external scripts use Date
        "visibility_mean": data["visibility_mean"],
        "visibility_min": data["visibility_min"],
        "visibility_max": data["visibility_max"],
      
    })
    df["year"] = df["date"].dt.isocalendar().year
    df["week"] = df["date"].dt.isocalendar().week
    df = df.dropna(subset=["year", "week"])
    
    print(df)

    return df


def main():
    #url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/input/chiang-mai-air-quality.csv" 
    #url = "https://github.com/rcsmit/streamlit_scripts/blob/main/input/chiang-mai-air-quality.csv"
    #url =r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\chiang-mai-air-quality.csv"
    url="https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/chiang-mai-air-quality.csv"

    df = pd.read_csv(url)
    #   https://aqicn.org/city/chiang-mai/
    #     https://aqicn.org/historical/#city:chiang-mai
    

    df['date'] = pd.to_datetime(df['date'])
    #df_vis = get_data_open_meteo()
    df_vis = get_data("Chiang Mai")
    df_vis['Date'] = pd.to_datetime(df_vis['Date'])
    # df_vis['Vis'] = df_vis['visibility_mean'] / 1000 # convert to km    
    # Merge the DataFrames on 'date'
    merged_df = pd.merge(df, df_vis, left_on='date',right_on='Date', how="outer")
    merged_df['Year'] = merged_df['Date'].dt.year
    # Create a scatter plot with Plotly
    show_calender_heatmap(merged_df)

 
    fig = px.scatter(merged_df, y='Vis', x='pm25',  title='Visibility vs PM2.5', labels={'vis': 'Visibility', 'pm2.5': 'PM2.5'},hover_data={'date': True, 'Vis': True, 'pm25': True})

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    st.info("Source PM2.5: https://aqicn.org/historical/#city:chiang-mai")
    # st.info("Source air visibility : https://www.ogimet.com")
    st.info("Source air visibility : https://open-meteo.com/en/docs/historical-weather-api")
if __name__ == "__main__":
    main()
 
