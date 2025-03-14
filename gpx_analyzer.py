import gpxpy
import folium
import numpy as np
import pandas as pd
import plotly.express as px
from geopy import distance
from geopy.distance import geodesic
import streamlit as st
import branca.colormap as cm

from streamlit_folium import st_folium
st.set_page_config(layout="wide")
# https://chatgpt.com/c/67d29ecd-b0e8-8004-beb6-c0e31f3534f3

def gpx_to_df(gpx):
    
    # Extract latitude, longitude, and elevation
    data = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                data.append([point.latitude, point.longitude, point.elevation,point.time])

    df = pd.DataFrame(data, columns=["latitude", "longitude", "elevation","time"])
    
    df["time"] = pd.to_datetime(df["time"], format='%Y-%m-%dT%H:%M:%SZ')
    df["delta_time"] = None
    df.at[0,"delta_time"] = 0
    for i in range(1, len(df)):
        df.at[i,"delta_time"] = df.at[i,"time"] - df.at[i - 1,"time"]
    
    return df


def effort_based_distance_calculator(gradient):
    """
    Calculates the effort-based distance multiplier using Naismith's Rule, including both uphill and downhill adjustments.

    Parameters:
        gradient (float): The gradient percentage (elevation change / distance * 100).

    Returns:
        float: Effort multiplier based on gradient.
    """
    if gradient > 0:  # Uphill
        if gradient < 5:
            effort_multiplier = 1.10  # +10% effort for slight incline
        elif gradient < 12:
            effort_multiplier = 1.20  # +20% effort for moderate incline
        elif gradient < 20:
            effort_multiplier = 1.40  # +40% effort for steep incline
        else:
            effort_multiplier = 2.00  # Very steep uphill, doubling effort
    elif gradient < 0:  # Downhill
        abs_gradient = abs(gradient)


        if abs_gradient < 10:  # Moderate downhill (easier)
            effort_multiplier = 1 - 0.5 * (abs_gradient/100)
        else:  # Steep downhill (harder)
            effort_multiplier= 1 + 0.2 * (abs_gradient/100)
    else:
        effort_multiplier = 1.00  # Flat terrain

    return effort_multiplier

def process_df(df):
    """
    Processes a DataFrame to compute distances, slopes, gradients, and effort-based distances using Naismith's Rule.

    Parameters:
        df (pd.DataFrame): DataFrame with 'latitude', 'longitude', and 'elevation' columns.

    Returns:
        pd.DataFrame: DataFrame with additional computed columns.
    """
    
    distances = [0]
    elevation_changes = [0]
    slopes = [0]
    elevation_gains = [0]
    difficulty_based_distance_multipliers = [0]
    difficulty_based_distances = [0]
    difficulties = [0]

    for i in range(1, len(df)):
        point1 = (df.iloc[i - 1]["latitude"], df.iloc[i - 1]["longitude"])
        point2 = (df.iloc[i]["latitude"], df.iloc[i]["longitude"])
        distance_ = geodesic(point1, point2).meters
        distances.append(distance_)

        elevation_diff = df.iloc[i]["elevation"] - df.iloc[i - 1]["elevation"]
        elevation_changes.append(elevation_diff)

        elevation_gain = max(elevation_diff, 0)
        elevation_gains.append(elevation_gain)

        slope = elevation_diff / distance_ if distance_ > 0 else 0
        slopes.append(slope)

        gradient = (elevation_diff / distance_) * 100 if distance_ > 0 else 0
        difficulty_based_distance_multiplier = effort_based_distance_calculator(gradient)
        difficulty_based_distance = difficulty_based_distance_multiplier * distance_
        difficulty = 0 if gradient <= 0 else abs(gradient) * distance_

        difficulty_based_distance_multipliers.append(difficulty_based_distance_multiplier)
        difficulty_based_distances.append(difficulty_based_distance)
        difficulties.append(difficulty)

    df["distance_m"] = distances
    df["elevation_change_m"] = elevation_changes
    df["slopes"] = slopes
    df["gradient"] = df["slopes"] *100
    df["elevation_gain"] = elevation_gains
    df["difficulty_based_distance_multiplier"] = difficulty_based_distance_multipliers
    df["difficulty_based_distance"] = difficulty_based_distances
    df["difficulty"] = difficulties

    df["distance_cumm"] = df["distance_m"].cumsum()
    df["difficulty_based_distance_cumm"] = df["difficulty_based_distance"].cumsum()
    #df["delta_time_cumm"] = df["delta_time"].cumsum()
    # df["delta_time"] = pd.to_timedelta(df["delta_time"])
    df["delta_time"] = pd.to_timedelta(df["delta_time"])
    df["delta_time_cumm"] = df["delta_time"].cumsum()

    # Convert cumulative delta_time to datetime object
    start_time = df["time"].iloc[0]
    df["delta_time_cumm"] = start_time + df["delta_time_cumm"]
    #df["delta_time_cumm"] = df["delta_time"].cumsum()

    # Convert cumulative delta_time to time object
    #df["delta_time_cumm"] = df["delta_time_cumm"].apply(lambda x: (pd.Timestamp(0) + x).time())
    st.write(df)
   
    return df

def show_map(df, what_to_display_):
    # Normalize what_to_display for coloring (convert steepness to color intensity)
    if what_to_display_ =="slopes":
        what_to_display = df["slopes"].values
    else:
        what_to_display = df["difficulty"].values
        

    normalized_what_to_display = (np.array(what_to_display) - min(what_to_display)) / (max(what_to_display) - min(what_to_display))

    # Sort what_to_display to avoid threshold sorting issues
    sorted_values = sorted(what_to_display)

    # Create a colormap (green = easy, red = hard)
    colormap = cm.LinearColormap(
        ["green", "yellow", "red"], 
        vmin=min(sorted_values), 
        vmax=max(sorted_values)
    )
    # Create a colormap (green = easy, red = hard)
    
    #colormap = cm.LinearColormap(["green", "yellow", "red"], vmin=min(what_to_display), vmax=max(what_to_display))

    # Create folium map centered at the start
    start_location = [df.iloc[0]["latitude"], df.iloc[0]["longitude"]]
    m = folium.Map(location=start_location, zoom_start=14, tiles="OpenStreetMap")

    # Add gradient polyline to the map
    for i in range(len(df) - 1):
        folium.PolyLine(
            locations=[(df.iloc[i]["latitude"], df.iloc[i]["longitude"]),
                    (df.iloc[i + 1]["latitude"], df.iloc[i + 1]["longitude"])],
            color=colormap(what_to_display[i]), weight=4
        ).add_to(m)
    route = list(zip(df["latitude"], df["longitude"]))

    # Add start and end markers
    folium.Marker(route[0], popup="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(route[-1], popup="End", icon=folium.Icon(color="red")).add_to(m)

    # Add colormap legend
    colormap.caption = "Route Difficulty (Slope)"
    m.add_child(colormap)

    # call to render Folium map in Streamlit
    st_data = st_folium(m, width=725)
   

def show_scatterplots(df):
    # Create scatter plots
    fig4 = px.line(df, x="distance_cumm", y="gradient", title="Distance vs Gradient")
    fig4 = px.line(df, x="distance_cumm", y="elevation", title="Distance vs Elevation")
    
    fig5 = px.line(df, x="distance_cumm", y="gradient", title="Distance vs Gradient")
    fig6 = px.line(df, x="distance_cumm", y="difficulty", title="Distance vs Difficulty")
    fig7 = px.line(df, x="delta_time_cumm", y=["distance_cumm", "difficulty_based_distance_cumm"], 
              labels={"value": "Distance", "variable": "Type"}, 
              title="Time vs Cumulative Distance and Cumulative Difficulty-Based Distance")
    
    df["distance_sma"]= df["distance_m"].rolling(window=10).mean()
    df["difficulty_based_distance_sma"]= df["difficulty_based_distance"].rolling(window=10).mean()
    
    fig8 = px.line(df, x="delta_time_cumm", y=["distance_sma", "difficulty_based_distance_sma"], 
              labels={"value": "Distance", "variable": "Type"}, 
              title="Time vs  Distance(sma10) and  Difficulty-Based Distance (sma10)")

    fig9 = px.line(df, x="distance_cumm", y=["distance_sma", "difficulty_based_distance_sma"], 
              labels={"value": "Distance", "variable": "Type"}, 
              title="Distance vs  Distance and  Difficulty-Based Distance")

    col4,col5,col6=st.columns(3)
    with col4:
        st.plotly_chart(fig4)
    with col5:
        st.plotly_chart(fig5)
    with col6:
        st.plotly_chart(fig6)
    st.plotly_chart(fig7)
    st.plotly_chart(fig8)
    st.plotly_chart(fig9)
    st.info (f"Sum of difficulaty : {df['difficulty'].sum()}")
    st.info (f"Sum of gradient : {df['gradient'].sum()}")
    percentage_gradient_positive = (df['gradient'] > 0).mean() * 100
    percentage_gradient_negative = (df['gradient'] < 0).mean() * 100

    st.info (f"Percentage of positive gradient : {percentage_gradient_positive}")
    st.info (f"Percentage of negative gradient : {percentage_gradient_negative}")
  
    # Compute total values
    total_distance = df["distance_m"].sum() / 1000  # Convert to km
    total_elevation_gain = df["elevation_gain"].sum()
    
    # Compute average gradient
    average_gradient = (total_elevation_gain / (total_distance * 1000)) * 100

    # Compute difficulty score (higher = harder)
    difficulty_score = (total_elevation_gain / total_distance) + average_gradient

    
    # Print results
    st.info(f"Total Distance: {total_distance:.2f} km")
    st.info(f"Total Elevation Gain: {total_elevation_gain:.0f} m")
    
    st.info(f"Average Gradient: {average_gradient:.1f} %")
    st.info(f"Trail Difficulty Score: {difficulty_score:.1f}")
    st.info(f"Difficulty based distance {round(df["difficulty_based_distance_cumm"].iloc[-1]/1000,2)} km")
def main():
    # Load GPX file
    gpx = get_gpx()
    df= gpx_to_df(gpx)
    reversed = st.checkbox("Reverse the route")
    if not reversed:
        df = process_df(df)
        col11,col12=st.columns(2)
        with col11:
        
            show_map(df, "slopes")
        with col12:
        
            show_map(df, "diffulties")
        show_scatterplots(df)
    else:
        df_reversed = df.iloc[::-1].reset_index(drop=True)
    
        df_reversed = process_df(df_reversed)
        col13,col14=st.columns(2)
        with col13:
        
            show_map(df_reversed, "slopes")
        with col14:
        
            show_map(df_reversed, "diffulties")
        show_scatterplots(df_reversed)

def get_gpx():
    sample_data = st.checkbox("Use sample data")
    if sample_data:
        #gpx_file_path = r"C:\Users\rcxsm\Downloads\RK_gpx _2025-03-13_1317.gpx"
        gpx_file_path = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/input/RK_gpx _2025-03-13_1317.gpx"
        #gpx_file_path = r"C:\Users\rcxsm\Downloads\test.gpx"

        with open(gpx_file_path, "r") as gpx_file:
            gpx = gpxpy.parse(gpx_file)
    else:
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
                gpx = gpxpy.parse(uploaded_file)
        else:
            st.warning("You need to upload a gpx file. Files are not stored anywhere after the processing of this script")
            st.stop()
    return gpx


    
if __name__ == "__main__":
    
    #19959.6792674
    main()  