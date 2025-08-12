import gpxpy
import folium
import numpy as np
import pandas as pd
import plotly.express as px
from geopy import distance
from geopy.distance import geodesic
import streamlit as st
import branca.colormap as cm
from scipy.interpolate import interp1d
import requests
import pandas as pd
import time  # To avoid hitting rate limits
import plotly.graph_objects as go
import ast
from streamlit_folium import st_folium
import folium
from streamlit_folium import folium_static
import re
try:
    st.set_page_config(layout="wide")
except:
    pass
# https://chatgpt.com/c/67d29ecd-b0e8-8004-beb6-c0e31f3534f3

def gpx_to_df(gpx):
    """
    Converts a GPX object to a pandas DataFrame.

    Args:
        gpx (gpxpy.gpx.GPX): A GPX object containing tracks, segments, and points.

    Returns:
        pd.DataFrame: A DataFrame with columns for latitude, longitude, elevation, and time.
    """   
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

def effort_based_distance_calculator (gradient):
   

    """
    Calculate the effort based on gradient using interpolation of the fitted points.
    Using graph from https://medium.com/strava-engineering/an-improved-gap-model-8b07ae8886c3
    Points fitted with https://www.graphreader.com/

    Args:
        gradient (float): The gradient in percent.

    Returns:
        float: The interpolated effort based on the fitted points.
    """
    fitted_points = [
        (-32, 1.6), (-31, 1.545), (-30, 1.49), (-29, 1.435), (-28, 1.38), (-27, 1.343), (-26, 1.307), (-25, 1.27),
        (-24, 1.234), (-23, 1.196), (-22, 1.158), (-21, 1.12), (-20, 1.083), (-19, 1.047), (-18, 1.01), (-17, 0.979),
        (-16, 0.954), (-15, 0.93), (-14, 0.912), (-13, 0.893), (-12, 0.88), (-11, 0.877), (-10, 0.873), (-9, 0.865),
        (-8, 0.874), (-7, 0.882), (-6, 0.889), (-5, 0.902), (-4, 0.923), (-3, 0.937), (-2, 0.955), (-1, 0.98),
        (0, 0.996), (1, 1.012), (2, 1.053), (3, 1.096), (4, 1.14), (5, 1.18), (6, 1.216), (7, 1.281), (8, 1.338),
        (9, 1.407), (10, 1.484), (11, 1.551), (12, 1.617), (13, 1.693), (14, 1.771), (15, 1.855), (16, 1.946),
        (17, 2.037), (18, 2.124), (19, 2.21), (20, 2.297), (21, 2.376), (22, 2.455), (23, 2.535), (24, 2.615),
        (25, 2.712), (26, 2.81), (27, 2.908), (28, 3), (29, 3.085), (30, 3.17), (31, 3.254), (32, 3.339),
        (33, 3.404), (34, 3.469)
    ]

    gradients, efforts = zip(*fitted_points)
    interpolation_function = interp1d(gradients, efforts, kind='linear', fill_value='extrapolate')
    return interpolation_function(gradient)

def get_elevation_batch(coords):
    """Get elevation for multiple latitude/longitude pairs."""
    google=False
    if google:
        API_KEY = "SECRET" 
        
        for lat, lon in coords:
            url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={lat}%2C-{lon}&key={API_KEY}"
        
        response = requests.get(url)
        
    else:
        #url = "https://api.open-elevation.com/api/v1/lookup" #ssl rerror
        #surl="https://api.opentopodata.org/v1/aster30m"
        url ="https://api.opentopodata.org/v1/mapzen"
        locations = "|".join([f"{lat},{lon}" for lat, lon in coords])
        params = {"locations": locations}
        try:
            response = requests.get(url, params=params)
        except Exception as e:
            st.info(f"Error {e}")
            st.stop()

    if response.status_code == 200:
        
        return [result["elevation"] for result in response.json()["results"]]
    else:
        return [None] * len(coords)  # Handle errors gracefully

#@st.cache_data()
def process_df(df,wdw_input, wdw_output, lookup_elevations, total_right_distance):
    """
    Processes a DataFrame to compute distances, slopes, gradients, and effort-based distances using Naismith's Rule.

    Parameters:
        df (pd.DataFrame): DataFrame with 'latitude', 'longitude', and 'elevation' columns.
        wdw_input (int): Window size for moving average of distance and elevation.
        wdw_output (int): Window size for moving average of the output data.
        lookup_elevations (bool): Whether to look up elevations using an API.
        total_right_distance (float): The total distance to normalize the computed distances.

    Returns:
        pd.DataFrame: DataFrame with additional computed columns.
    """
   
    
    if lookup_elevations:
        df['elevation_original'] = df['elevation']
        
        # Define batch size (adjust based on API limits)
        batch_size = 50  
        elevations = []

        # Process DataFrame in batches
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            coords = list(zip(batch['latitude'], batch['longitude']))
            elevations.extend(get_elevation_batch(coords))
            print (f"{i} / {len(df)}")
            time.sleep(1)  # Pause to prevent hitting rate limits

        # Add elevation data to DataFrame
        
        df['elevation'] = elevations

    else:
        df['elevation_original'] = 0.0

    

    distances = [0]
    elevation_changes = [0]
    slopes = [0]
    elevation_gains = [0]
    elevation_losses = [0]
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
        elevation_loss = max(-elevation_diff, 0)
        elevation_losses.append(elevation_loss)
        slope = elevation_diff / distance_ if distance_ > 0 else 0
        if slope >.34:
            slope = .34
        if slope < -.34:
            slope = -.34
        slopes.append(slope)

        gradient = (elevation_diff / distance_) * 100 if distance_ > 0 else 0
        if gradient >34:
            gradient = 34
        if gradient < -34:
            gradient = -34
        difficulty_based_distance_multiplier = effort_based_distance_calculator(gradient)
        difficulty_based_distance = difficulty_based_distance_multiplier * distance_
        difficulty = 0 if gradient <= 0 else abs(gradient) * distance_

        difficulty_based_distance_multipliers.append(difficulty_based_distance_multiplier)
        difficulty_based_distances.append(difficulty_based_distance)
        difficulties.append(difficulty)

    df["distance_"] = distances
    df["elevation_change_m"] = elevation_changes
    
    df["elevation_change_m"] = df["elevation_change_m"].rolling(window=wdw_input).mean()
    df["distance_"] = df["distance_"].rolling(window=wdw_input).mean()
    
    df["distance_m"] = np.sqrt(df["distance_"]**2 + df["elevation_change_m"]**2)
    total_measured_distance=df["distance_m"].sum()
    
    correction_distance = total_right_distance / total_measured_distance if total_right_distance > 0 else 1.0
    df["distance_m"] = df["distance_m"]*correction_distance
    df["slopes"] = slopes
    df["gradient"] = df["slopes"] *100
    df["elevation_gain"] = elevation_gains
    df["elevation_loss"] = elevation_losses
    df["difficulty_based_distance_multiplier"] = difficulty_based_distance_multipliers
    df["difficulty_based_distance"] = difficulty_based_distances
    df["difficulty"] = difficulties

    df["distance_cumm"] = df["distance_m"].cumsum()

    df["difficulty_based_distance_cumm"] = df["difficulty_based_distance"].cumsum()
    #df["delta_time_cumm"] = df["delta_time"].cumsum()
    # df["delta_time"] = pd.to_timedelta(df["delta_time"])
    df["delta_time"] = pd.to_timedelta(df["delta_time"])
    df["delta_time_cumm"] = df["delta_time"].cumsum()

    df["delta_time_s"] = df["delta_time"].dt.total_seconds()
    df['speed_kmh'] = (df['distance_m'] / df['delta_time_s']) * 3.6
    
    # Convert cumulative delta_time to datetime object
    start_time = df["time"].iloc[0]
    df["delta_time_cumm"] = start_time + df["delta_time_cumm"]
    #df["delta_time_cumm"] = df["delta_time"].cumsum()
   
    #Calculate total elapsed seconds since start
    df["delta_time_cumm_s"] = (df["delta_time_cumm"] - start_time).dt.total_seconds()

    # Now compute cumulative speed
    df['speed_kmh_cumm'] = (df['distance_cumm'] / df['delta_time_cumm_s']) * 3.6
    # Convert cumulative delta_time to time object
    #df["delta_time_cumm"] = df["delta_time_cumm"].apply(lambda x: (pd.Timestamp(0) + x).time())
    df["gradient_sma"] = df["gradient"].rolling(window=wdw_output).mean()
    df["elevation_sma"] = df["elevation"].rolling(window=wdw_output).mean() 
    df["distance_sma"]= df["distance_m"].rolling(window=wdw_output).mean()
    df["difficulty_based_distance_sma"]= df["difficulty_based_distance"].rolling(window=wdw_output).mean()
 
    #df= df[df["distance_cumm"] <=5000]
    return df

def show_map(df,what_to_show_colors):
    """
    Creates and displays a folium map with the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the processed GPX data.

        what_to_show_colors (str): "gradient", "speed_kmh", "elevation_sma", "elevation_original"
    """     
    
    # what_to_display = df["gradient"].values
    what_to_display = df[what_to_show_colors].fillna(0).values

    # Sort what_to_display to avoid threshold sorting issues
    sorted_values = sorted(what_to_display)
    if what_to_show_colors =="speed_kmh":
        #take away the 10% slowest values to prevent that the scale starts at 0

        sorted_values = sorted_values[int(len(sorted_values)/10):]
   
    # Create a colormap (green = easy, red = hard)
    colormap = cm.LinearColormap(
        ["green", "yellow", "red"], 
        vmin=min(sorted_values), 
        vmax=max(sorted_values)
    )
 
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
    
    # Add markers for each kilometer
    for i in range(1, int(df["distance_cumm"].max() // 1000) + 1):
        km_marker = df[df["distance_cumm"] >= i * 1000].iloc[0]
        folium.Marker(
            location=[km_marker["latitude"], km_marker["longitude"]],
            popup=f"{i} km",
            icon=folium.Icon(color="blue")
            
        ).add_to(m)
    # Add colormap legend
    
    colormap.caption = what_to_show_colors

    
    m.add_child(colormap)

    # call to render Folium map in Streamlit
    #st_folium(m,  use_container_width=True)
    st_data = folium_static(m, width=500, height=500)


def speed_by_interval(df, mode='distance', interval=100):
    """
    Calculate and plot speed per fixed interval.
    
    Parameters:
    - df: DataFrame with 'distance_cumm', 'distance_m', and 'delta_time'
    - mode: 'distance' or 'time'
    - interval: size of each segment in meters or seconds
    """

    if mode == 'distance':
        st.subheader(f"Speed per {interval} meter segment")
        df['bin'] = ((df['distance_cumm'] // interval) + 1) * interval

    elif mode == 'time':
        st.subheader(f"Speed per {interval} second segment")
        # Ensure delta_time is timedelta and create a cumulative time in seconds
        df['delta_time'] = pd.to_timedelta(df['delta_time'])
        df['time_cumm'] = df['delta_time'].cumsum().dt.total_seconds()
        df['bin'] = ((df['time_cumm'] // interval) + 1) * interval

    else:
        st.error("Mode must be 'distance' or 'time'")
        return

    # Group and calculate speed
    grouped = df.groupby('bin').agg(
        total_distance_m=('distance_m', 'sum'),
        total_time_s=('delta_time', lambda x: x.dt.total_seconds().sum())
    ).reset_index()

    grouped['speed_kmh'] = (grouped['total_distance_m'] / grouped['total_time_s']) * 3.6
    grouped['group'] = ((grouped['bin'] - 1) // (1000 if mode == 'distance' else 60)).astype(int)

    st.dataframe(grouped[['bin', 'speed_kmh']])

    fig = px.bar(
        grouped,
        x='bin',
        y='speed_kmh',
        color='group',
        text=grouped['speed_kmh'].round(1),
        labels={
            'bin': f"End {'Distance (m)' if mode == 'distance' else 'Time (s)'}",
            'speed_kmh': 'Speed (km/h)',
            'group': 'Group'
        },
        title=f"Speed per {interval} {'meters' if mode == 'distance' else 'seconds'}"
    )

    fig.update_traces(textposition='outside', textangle=-90)
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='show')

    st.plotly_chart(fig)

def show_plots(df,wdw):
    """
    Creates and displays various plots for the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the processed GPX data.
        wdw (int): Window size for smoothing (for the plot titles).
    """   
    # Create scatter plots
    
   
    fig4 = px.line(df, x="distance_cumm", y=["elevation_sma","elevation_original"], title=f"Distance vs Elevation (sma{wdw})", 
                    labels={"distance_cumm": "Distance (m)", "elevation_sma": "Elevation (m)", "gradient": "Gradient (%)"}, 
                    #hover_data={"distance_cumm": True, "elevation_sma": True,"elevation_api": True, "gradient": True}
                    )
    
    fig5 = px.line(df, x="distance_cumm", y="gradient", title=f"Distance vs Gradient (sma{wdw})")
   
    # Add a horizontal line at y=0
    fig5.add_shape(
        type="line",
        x0=df["distance_cumm"].min(),
        y0=0,
        x1=df["distance_cumm"].max(),
        y1=0,
        line=dict(color="Red", width=2)
    )
    fig7 = px.line(df, x="delta_time_cumm", y=["distance_cumm", "difficulty_based_distance_cumm"], 
              labels={"value": "Distance", "variable": "Type"}, 
              title="Time vs Cumulative Distance and Cumulative Difficulty-Based Distance")
    
    df_=df.iloc[1:-1]
    fig8 = px.line(df_, x="delta_time_cumm", y=["distance_sma", "difficulty_based_distance_sma"], 
              labels={"value": "Distance", "variable": "Type"}, 
              title=f"Time vs  Distance(sma{wdw}) and  Difficulty-Based Distance (sma{wdw})")

    fig9 = px.line(df_, x="distance_cumm", y=["distance_sma", "difficulty_based_distance_sma"], 
              labels={"value": "Distance", "variable": "Type"}, 
              title=f"Distance vs  Distance (sma{wdw}) and  Difficulty-Based Distance(sma{wdw})")

    col4,col5=st.columns(2)
    with col4:
        st.plotly_chart(fig4)
        
    with col5:
        st.plotly_chart(fig5)
   
    st.plotly_chart(fig7)
    st.plotly_chart(fig8)
    st.plotly_chart(fig9)

def show_stats(df):
    """
    Calculates and displays various stats for the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the processed GPX data.
    """ 
    # Compute total values
    total_distance = df["distance_m"].sum() / 1000  # Convert to km
    total_distance_m = df["distance_m"].sum() 
    total_elevation_gain = df["elevation_gain"].sum()
    total_elevation_loss = df["elevation_loss"].sum()
    
    total_distance_with_elevation_gain = df[df["elevation_gain"] != 0]["distance_m"].sum()
    total_distance_with_elevation_loss = df[df["elevation_loss"] != 0]["distance_m"].sum()
    total_distance_with_even_elevation = df[df["elevation_change_m"] == 0]["distance_m"].sum()

    st.info(f"Distance gain : {total_distance_with_elevation_gain:.1f} m ({total_distance_with_elevation_gain/total_distance_m*100:.1f} %) | Even track : {total_distance_with_even_elevation:.1f} m  ({total_distance_with_even_elevation/total_distance_m*100:.1f} %)|  Distance loss : {total_distance_with_elevation_loss:.1f} m ({total_distance_with_elevation_loss/total_distance_m*100:.1f} %) ")
    st.info(f"Total Elevation Gain: {total_elevation_gain:.1f} m  (average gradient: {total_elevation_gain/total_distance_with_elevation_gain*100:.1f} %) | Total Elevation Loss: {total_elevation_loss:.1f} m  (average gradient: {total_elevation_loss/total_distance_with_elevation_loss*100:.1f} %)")
    st.info(f"Total Distance: {total_distance:.2f} km | Difficulty based distance {round(df['difficulty_based_distance_cumm'].iloc[-1]/1000,2)} km")


def show_info(df):
    """
    Displays info how the effeort mutiplier is calculated.
    """ 

    st.info("""
    Effort Multiplier Calculation:
    Gradient - Elevation change / Distance * 100
    - **Uphill**:
    - Gradient < 5%: +10% effort for slight incline
    - Gradient 5 - 12%: +20% effort for moderate incline
    - Gradient 12 -19%: +40% effort for steep incline
    - Gradient >= 20%: Doubling effort for very steep uphill
    - **Downhill**:
    - Gradient < 10%: Moderate downhill, effort decreases by 0.5% per gradient percentage
    - Gradient >= 10%: Steep downhill, effort increases by 0.2% per gradient percentage
    - **Flat terrain**: No additional effort
    """)
   
    
    # 12 beach hike https://www.strava.com/activities/13887499750
    # 55 km kpg https://www.strava.com/activities/12642202735

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def interval_analyse(df, segments, mode):
    """
    Analyseer snelheid per segment op basis van afstands- of tijdsintervallen.

    Parameters:
    - df: DataFrame met kolommen distance_cumm, delta_time, distance_m
    - segments: lijst van [snel, langzaam] intervallen
    - mode: 'distance' of 'time'
    """

    # Zorg dat tijdskolom bruikbaar is
    df['delta_time'] = pd.to_timedelta(df['delta_time'])

    if mode == 'time':
        df['time_cumm'] = df['delta_time'].cumsum().dt.total_seconds()
        position_col = 'time_cumm'
        x_label = 'Tijd (s)'
    else:
        position_col = 'distance_cumm'
        x_label = f'Afstand (m)'

    # Genereer segmentgrenzen
    boundaries = []
    start = 0
    for fast, slow in segments:
        if fast > 0:
            boundaries.append((start, start + fast, 'snel'))
            start += fast
        if slow > 0:
            boundaries.append((start, start + slow, 'langzaam'))
            start += slow

    # Bereken snelheid per segment
    result = []
    for start_p, end_p, type_ in boundaries:
        segment_df = df[(df[position_col] >= start_p) & (df[position_col] < end_p)]
        if not segment_df.empty:
            total_distance = segment_df['distance_m'].sum()
            total_time = segment_df['delta_time'].dt.total_seconds().sum()
            speed_kmh = (total_distance / total_time) * 3.6 if total_time > 0 else 0
            result.append({
                'segment': f'{start_p}-{end_p}',
                'start': start_p,
                'end': end_p,
                'mid': (start_p + end_p) / 2,
                'width': end_p - start_p,
                'speed_kmh': speed_kmh,
                'type': type_
            })

    df_segments = pd.DataFrame(result)
    # st.write(df_segments)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_segments['mid'],
        y=df_segments['speed_kmh'],
        width=df_segments['width'],
        text=df_segments['width'].astype(str) + ("s" if mode == 'time' else "m") +
             " | " + df_segments['speed_kmh'].round(1).astype(str) + " km/h",
        textposition='outside',
        marker_color=df_segments['type'].map({'snel': 'blue', 'langzaam': 'lightblue'}),
        hovertext=df_segments['segment'],
    ))

    fig.update_layout(
        title=f'Gemiddelde snelheid per segment ({x_label.lower()})',
        xaxis_title=x_label,
        yaxis_title='Snelheid (km/h)',
        bargap=0.0,
        showlegend=False
    )

    st.plotly_chart(fig)

def get_gpx():
    """
    Loads a GPX file from the user's input.
    """

    sample_data = False # st.checkbox("Use sample data")
    if sample_data:
        #gpx_file_path = r"C:\Users\rcxsm\Downloads\RK_gpx _2025-03-13_1317.gpx"
        #gpx_file_path = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/input/RK_gpx_2025-03-13_1317.gpx"
        gpx_file_path = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/RK_gpx_2025-03-13_1317.gpx"
        #gpx_file_path = r"C:\Users\rcxsm\Downloads\test.gpx"

        with open(gpx_file_path, "r") as gpx_file:
            gpx = gpxpy.parse(gpx_file)
    else:
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            try:
                gpx = gpxpy.parse(uploaded_file)
                if not gpx.tracks:
                    st.error("The uploaded GPX file doesn't contain any tracks. Please upload a valid GPX file.")
                    st.stop()
            except Exception as e:
                st.error(f"Error loading / parsing the GPX file: {str(e)}")
                st.stop()
        else:
            st.warning("You need to upload a gpx file. Files are not stored anywhere after the processing of this script")
            st.stop()
    return gpx

def parse_segments(input_str,total_time_gpx):
    """
    Parse input like:
    "[[0,60], 10x[30,30], [40,20]]"
    → [[0,60], [30,30], ..., [30,30], [40,20]]
    """
    pattern = r'(\d+)x\[(\d+),\s*(\d+)\]'
    expanded = input_str

    for match in re.finditer(pattern, input_str):
        count, a, b = match.groups()
        repeated = ', '.join([f'[{a},{b}]'] * int(count))
        expanded = expanded.replace(match.group(0), repeated)
    segments = ast.literal_eval(expanded)
    total_time_segments = sum(f + s for f, s in segments)

    
    # Rest time
    rest_time = total_time_gpx - total_time_segments
    st.write(f"rest time {rest_time}")
    if rest_time > 0:
        segments.append([0, rest_time])  # Add rest as final "slow" segment

    return segments,total_time_segments
def format_seconds_to_mmss(seconds):
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    return f"{minutes:02d}:{sec:02d}"
def find_fastest_5k(df,distance):
    df = df.copy()
    df['delta_time'] = pd.to_timedelta(df['delta_time'])
    df['time_seconds'] = df['delta_time'].dt.total_seconds()
    df['cum_time'] = df['time_seconds'].cumsum()
    #df['distance_cumm']=round(df['distance_cumm'])
    best_time = float('inf')
    best_start = best_end = None

    start_idx = 0
    for end_idx in range(len(df)):
        # Move start_idx until the distance is >= 5km
        while df.loc[end_idx, 'distance_cumm'] - df.loc[start_idx, 'distance_cumm'] >= distance:
            time_taken = df.loc[end_idx, 'cum_time'] - df.loc[start_idx, 'cum_time']
            if time_taken < best_time:
                best_time = time_taken
                best_start = start_idx
                best_end = end_idx
            start_idx += 1

    if best_start is not None:
        fastest_segment = df.loc[best_start:best_end].copy()
        duration_min = best_time / 60
        pace_per_km = (best_time / 5) / 60  # minutes per km
                
        distance_start = df.loc[best_start, 'distance_cumm']
        distance_end = df.loc[best_end, 'distance_cumm']
        total_distance_km = (distance_end - distance_start) / 1000
        speed_kmh = total_distance_km / (best_time / 3600)

        result= {
            'segment': df.loc[best_start:best_end].copy(),
            'duration_sec': best_time,
            'duration_min': best_time / 60,
            'pace_min_per_km': (best_time / 5) / 60,
            'distance_start': distance_start,
            'distance_end': distance_end,
            'speed_kmh': speed_kmh
        }

        if result:
            st.write(f"📍 **Startafstand:** {result['distance_start']:.1f} m")
            st.write(f"🏁 **Eindafstand:** {result['distance_end']:.1f} m")
            st.write(f"🏁 **Afstand:** {round(result['distance_end']-result['distance_start'],2)} m")
            
            st.write(f"⏱️ **Tijd:** {format_seconds_to_mmss(result['duration_sec'])}")
            st.write(f"⚡ **Gemiddelde snelheid:** {result['speed_kmh']:.2f} km/h")
            st.write(f"🦶 **Tempo:** {result['pace_min_per_km']:.2f} min/km")
            
        else:
            st.warning("Geen segment gevonden.")
                
        return 
    else:
        st.info("No fast segment found")
        return None, None, None, None

def main():
    st.title("GPX Analyzer")
    st.write("This script analyzes the data from a GPX file to provide insights on the route's difficulty, elevation gain, and more.")

    gpx = get_gpx()
    df= gpx_to_df(gpx)
    
    if df.empty:
        st.error("Could not extract data from the GPX file. Please ensure it contains valid track data.")
        st.stop()
    
    reversed = st.checkbox("Reverse the route")
    lookup_elevations = st.checkbox("Lookup elevations (may be slow for large files)")
    
    cola,colb,colc,cold=st.columns(4)
    with cola:
        what_to_show_colors = st.selectbox("What to show on the map", ["gradient", "speed_kmh", "elevation_sma", "elevation_original"])
    
    with colb:
        if len(df) < 10000:
            default_filter_factor = 1
        else:
            default_filter_factor = len(df) // 1000
        filter_factor = st.number_input("Filter factor",min_value=1,max_value=10000,value = default_filter_factor , help="The higher the number, the more points will be filtered out. Default value keeps 1000 points") 
    with colc:
        if filter_factor>1:
            wdw_input_default = 9
        else:
            wdw_input_default = 1
        if filter_factor >1:
            label_wdw_input ="Window size for moving average of the distance and elevation (before filtering)"
        else:
            label_wdw_input ="Window size for moving average of the distance and elevation"
            
        wdw_input =st.number_input(label_wdw_input,min_value=1,max_value=100,value=wdw_input_default, help="The higher the number, the smoother the data")
        
    with cold:
        if wdw_input ==1:
            wdw_output_default =9
        else:
            wdw_output_default =1
        wdw_output =st.number_input("Window size for moving average",min_value=1,max_value=100,value=wdw_output_default, help="The higher the number, the smoother the plot")
    
    
    st.write(f"GPX converted. {len(df)} waypoints")
    # df = df.iloc[1::filter_factor]
    if filter_factor > 1:
        df = pd.concat([df.iloc[[0]], df.iloc[1::filter_factor], df.iloc[[-1]]]).drop_duplicates()
    st.write(f"Data filtered. Result {len(df)} waypoints")

    if reversed:
        df = df.iloc[::-1].reset_index(drop=True)
    
    

    cola,colb,colc,cold=st.columns(4)
    with cola:    
        right_distance = st.number_input("Distance (m)",min_value=0.0,max_value=10_000_000.0,value=5000.0,  help=".")
                
        # Default input
    df = process_df(df,wdw_input,wdw_output, lookup_elevations,right_distance)
        
    with colb:
        mode = st.selectbox("Select mode", ['distance', 'time'])
        unit = "m" if mode=="distance" else "s"
        
    with colc:
        
        interval = st.number_input(f"Interval ({unit})", min_value=10, max_value=1000, step=10, value=100)

    with cold:
        if unit =="m":
            default_segments = "[[0,1000],[500,500],[600,400],[700,300],[500,500]]"
        else:
            default_segments = "[[0,60], 10x[30,30], [40,20], [50,10], [30,30]]"
        # User input via text area
        segment_input = st.text_area("Voer segmenten in (zoals [[snel,langzaam],[...]]):", default_segments)
       
        if unit =="m":
            # Try to parse
            try:
                segments = ast.literal_eval(segment_input)
                #st.success(f"Segments geladen: {segments}")
            except Exception as e:
                st.error(f"Fout bij het laden van segmenten: {e}")
                st.stop()
        else:
            try:
            # if 1==1:
                total_time_gpx = df["delta_time_cumm_s"].max()
                segments,total_time_segments = parse_segments(segment_input, total_time_gpx)
                st.write(f"Total time segments:{total_time_segments}")
                st.write(f"Total time training {df["delta_time_cumm_s"].max()}")
            except Exception as e:
                st.error(f"Fout bij het laden van segmenten: {e}")
                st.stop()
    show_map(df, what_to_show_colors)
    find_fastest_5k(df,5000)
    speed_by_interval(df, mode=mode, interval=interval)

    interval_analyse(df, segments, mode)
    show_plots(df,wdw_output)
    show_stats(df)
    show_info(df)
    st.write(df)
    
if __name__ == "__main__":
    main()  