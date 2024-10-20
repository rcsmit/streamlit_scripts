import os
import datetime
import platform
import polars as pl
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from show_knmi_functions.utils import loess_skmisc
import pandas as pd
from scipy import stats
from plotly.subplots import make_subplots
from scipy import stats
from pymannkendall import original_test,  seasonal_test
import numpy as np


def get_data_regen():
    """read the data with Polars, all in one file.

    Returns:
        df: polars dataframe
    """  
    st.subheader("The data")  
    st.write("THESE DATA CAN BE USED FREELY PROVIDED THAT THE FOLLOWING SOURCE IS ACKNOWLEDGED: ROYAL NETHERLANDS METEOROLOGICAL INSTITUTE")
    url = r"C:\Users\rcxsm\Documents\python_scripts\fritsander_knmi\combined_fritz.csv"
    
    df = pl.read_csv(url)
    df = df.with_columns(
        pl.col('YYYYMMDD').cast(pl.String).str.strptime(pl.Datetime, "%Y%m%d"))
    df = df.drop(pl.last())
    df = df.rename({"0TN": "STN"})
   
    st.write(df)

    # IF YOU WANT TO CHECK WHETHER ALL STATIONS ARE INCLUDED

    # # Assuming df is your Polars DataFrame
    # stn_count = df.select(pl.col("STN").n_unique()).item()
    # st.write(f"{stn_count}")# stations are used in the file")

    # getting info from a specific station
    # df_filtered = df.filter(pl.col('STN') == 1)

    # # Show the filtered result
    # st.write(df_filtered)
    return df


    
@st.cache_data()
def load_and_combine_data():

    """Loads multiple CSV files and combines them into a single dataframe.

    Args:
        files (list): List of file paths to load.

    Returns:
        pl.DataFrame: Combined dataframe.
    """
    #files= ['input/stations_1_to_194.csv', 
            # 'input/stations_195_to_344.csv', 
            # 'input/stations_345_to_530.csv', 
            # 'input/stations_531_to_716.csv', 
            # 'input/stations_718_to_910.csv', 
            # 'input/stations_911_to_983.csv']

    if platform.processor() != "":
        files= [r'C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\stations_1_to_194.csv', 
                r'C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\stations_195_to_344.csv', 
                r'C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\stations_345_to_530.csv', 
                r'C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\stations_531_to_716.csv', 
                r'C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\stations_718_to_910.csv', 
                r'C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\stations_911_to_983.csv']


    else:        
        files= [r'https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/stations_1_to_194.csv', 
                r'https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/stations_195_to_344.csv', 
                r'https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/stations_345_to_530.csv', 
                r'https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/stations_531_to_716.csv', 
                r'https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/stations_718_to_910.csv', 
                r'https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/stations_911_to_983.csv']



    

    #dataframes = [pl.read_csv(file) for file in files]
    dataframes = []

    # Read and process each file individually before concatenation
    for file in files:
        print (file)
        df = pl.read_csv(file)
        
        # Cast YYYYMMDD to string and then parse as Datetime
        df = df.with_columns(
            pl.col('YYYYMMDD').cast(pl.Utf8).str.strptime(pl.Datetime, "%Y%m%d")
        )
        # Ensure 'SX' is of integer type (change 'SX' to the actual column name if different)
        df = df.with_columns(
            pl.col('SX').cast(pl.Int64)  # Change to appropriate integer type if necessary
        )
        
        # Drop the last row
        df = df.head(df.height - 1)
        
        # Append the processed dataframe to the list
        dataframes.append(df)

    # Concatenate all processed dataframes
    df_combined = pl.concat(dataframes)
    return df_combined

def date_range(df):
    # Assuming your Polars dataframe is named df
    #
    # VERY SLOW. 
    # 
    # Just use the csv
    df_summary = df.group_by('STN').agg([
        pl.col('YYYYMMDD').min().alias('first_date'),  # Get the first date (min)
        pl.col('YYYYMMDD').max().alias('last_date')    # Get the last date (max)
    ])
    df_summary_pd = df_summary.to_pandas()
    # Show the result
    st.write(df_summary_pd)
    #df_summary_pd.to_csv("daterages_neerslagstations.csv")
    print ("done")


def show_station_data():
    """Show a table with the information about the measurement stations
    """
    st.subheader("Measurement station info")
    url_dateranges =  r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\daterages_neerslagstations.csv"
    url_stations =  r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\stations.csv"
    df_dateranges = pd.read_csv(url_dateranges)
    df_stations = pd.read_csv(url_stations)
    #df_dateranges = df_dateranges["STN"].astype(int)
    station_info=pd.merge(df_dateranges,df_stations, on="STN", how="outer")
    station_info = station_info.drop("Unnamed: 0", axis=1)
        
    # Convert columns to datetime
    station_info["last_date"] = pd.to_datetime(station_info["last_date"])
    station_info["first_date"] = pd.to_datetime(station_info["first_date"])

    # Calculate the difference in days
    station_info["days"] = (station_info["last_date"] - station_info["first_date"]).dt.days

    st.write(station_info)
    st.info( "Kaart: https://cdn.knmi.nl/knmi/map/page/additional/kaart_neerslagstations_201802.jpg")
   

    # Define the cutoff date
    cut_off = "2024-09-30"
    cutoff_date = pd.to_datetime(cut_off)

    # Count stations with last_date before the cutoff date
    count_stations = station_info[station_info["last_date"] == cutoff_date].shape[0]

    st.write(f"{count_stations} out of {len(df_stations)} have info until {cut_off}" )
    df_station_info_pl = pl.from_pandas(station_info)
    return df_station_info_pl


def plot_number_of_measurements_per_day(df):
    """Counting and plotting the number of measurements (stations used) per day

    Args:
        df (df): polars dataframe
    """
    # Assuming your Polars dataframe is named df
    # Group by 'YYYYMMDD' and count the number of measurements per day
    df_grouped = df.group_by('YYYYMMDD').agg([
        pl.col('STN').count().alias('count_per_day')
    ])

    # Convert to pandas for easy plotting with Plotly
    df_grouped_pandas = df_grouped.to_pandas()

    # Create a bar chart using Plotly
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_grouped_pandas['YYYYMMDD'],  # Date
        y=df_grouped_pandas['count_per_day'],  # Count of measurements
        name='Measurements per day'
    ))

    # Add layout details
    fig.update_layout(
        title='Number of Measurements per Day',
        xaxis_title='Date',
        yaxis_title='Count of Measurements',
        xaxis_tickformat='%Y-%m-%d'  # Format the x-axis for dates
    )

    # Show the plot
    
    st.plotly_chart(fig)


def calculate_average_per_day_year(df):
    """Calculating average per day and per year 

    Args:
        df (_type_): _description_

    Returns:
        df_grouped: average precipitation by day

        df_grouped_year: average precipitation by year
    """    
    
    # Group by YYYYMMDD, calculate mean of RD, and extract year and month in one chain
    df_grouped = (
        df
        .group_by("YYYYMMDD")
        .agg(pl.col("RD").mean().alias("mean_RD"))
        .with_columns([
            pl.col("YYYYMMDD").dt.year().alias("year"),
            pl.col("YYYYMMDD").dt.month().alias("month")
        ])
        .group_by(["year", "month"])
        .agg(pl.col("mean_RD").mean().alias("mean_RD"))  # Taking mean RD by year and month
        .with_columns([
            pl.concat_str([
                pl.col("year").cast(pl.Utf8),
                pl.lit("-"),
                pl.col("month").cast(pl.Utf8).str.zfill(2)
            ]).str.strptime(pl.Date, format="%Y-%m").alias("date_new")
        ])
        .sort("date_new")
        .with_columns(
            rolling_mean=pl.col("mean_RD").rolling_mean(window_size=365)
        )
    )

    # Group by year to calculate mean RD and sort in one chain
    df_grouped_year = (
        df
        .with_columns(pl.col("YYYYMMDD").dt.year().alias("year"))
        .group_by("year")
        .agg(pl.col("RD").mean().alias("mean_RD"))  # Taking mean RD by year
        .with_columns(
            rolling_mean=pl.col("mean_RD").rolling_mean(window_size=7).alias("mean_RD_sma_7")
        )
        .sort("year")
    )

    df_grouped_pd = df_grouped.to_pandas()
    df_grouped_year_pd = df_grouped_year.to_pandas()
    return df_grouped_pd, df_grouped_year_pd

def plot_grouped_by_month(df_grouped):
    # Create the plot
    fig = px.line(df_grouped, x=["date_new"], y='mean_RD', title='RD over time - grouped by month, sma 365')
    st.plotly_chart(fig)
    # Show the plot

def plot_graph_with_loess(df_grouped_year_pd):
    """Plot a graph with loess average, eq. to 30 years SMA
    See https://rene-smit.com/loess-is-more/

    Args:
        df (pandas df): df
    """  
    X_array =  df_grouped_year_pd["year"].values
    Y_array =  df_grouped_year_pd["mean_RD"].values
    t, loess_values, ll, ul = loess_skmisc(X_array, Y_array)

    fig = go.Figure()

    # Add traces for temp_loess and temp
    fig.add_trace(go.Scatter(x=t, y=Y_array, name='Actual Neerslag'))
    fig.add_trace(go.Scatter(x=t, y=loess_values, mode='lines', name='LOESS Neerslag'))
   
    # Update layout
    fig.update_layout(title='Neerslag per jaar met LOESS gemiddelde',
                xaxis_title='Jaar',
                yaxis_title='gem Neerslag ')
    st.plotly_chart(fig)

def plot_met_horizontale_lijnen(df):
    """Plot a graph with horizontal lines for every era of 20 years

    Args:
        df (pandas df): df
    """   
    # Maak een nieuwe kolom voor 20-jarige intervallen
    df['year_group'] = (df['year'] // 20) * 20

    # Bereken het gemiddelde per 20 jaar groep
    mean_per_20_years = df.groupby('year_group')['mean_RD'].mean().reset_index()

    # Maak een figuur aan
    fig = go.Figure()

    # Plot de originele data
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['mean_RD'],
        mode='lines+markers',
        name='mean_RD'
    ))

    # Voeg horizontale lijnen toe per 20 jaar interval
    for index, row in mean_per_20_years.iterrows():
        start_year = row['year_group']
        end_year = start_year + 19
        fig.add_trace(go.Scatter(
            x=[start_year, end_year],
            y=[row['mean_RD'], row['mean_RD']],
            mode='lines',
            line=dict(color='red', width=2),
            name=f"{start_year}-{end_year}" if index == 0 else "",  # Alleen 1 keer de naam tonen
            showlegend=False if index > 0 else True  # Legenda slechts 1 keer tonen
        ))

    # Layout details
    fig.update_layout(
        title='Mean_RD per 20 jaar gemiddelde',
        xaxis_title='Jaar',
        yaxis_title='mean_RD',
        showlegend=True
    )

    # Toon de grafiek
    st.plotly_chart(fig)

    st.info("Data: https://www.knmi.nl/nederland-nu/klimatologie/monv/reeksen")

def plot_heavy_rain(df, cut_off=500):

    """
    Plot heavy rain
    Args:
        df (polars df): df
        cut_off = cutt off, in tenths of mm

    Extreme neerslag kan zowel worden gedefinieerd op grond van de hoeveelheid 
    (neerslag boven een bepaalde drempelwaarde) als in termen van herhalingstijd 
    (de neerslaghoeveelheid die eens per zoveel jaar overschreden wordt). 
    Omdat neerslag vele karakteristieken heeft - grootte, intensiteit en duur - 
    is er geen eenduidige definitie van een extreem. Plaatselijke neerslag van 
    meer dan 25 millimeter in een uur noemen we een hoosbui, terwijl meer dan 
    50 millimeter in één dag wordt aangeduid met 'een dag met zware neerslag'. 
    Waardes boven de 50 millimeter in een uur en 100 millimeter in een dag komen 
    in Nederland ongeveer één keer per eeuw of iets vaker voor wanneer men zich 
    op een vaste locatie bevindt.
    https://www.knmi.nl/kennis-en-datacentrum/uitleg/extreme-neerslag

    Calculate the daily ratio of measurements above cut_off to unique stations from a Polars DataFrame.

    This script performs the following steps:
    1. Computes the number of unique stations (`a`) for each day.
    2. Counts the number of measurements (`b`) that exceed cut_off for each day.
    3. Calculates the ratio `c = b / a` for each day, where:
    - `b` is the count of measurements above cut_off.
    - `a` is the number of unique stations.
    4. Plots the resulting ratio (`c`) as a line graph using Plotly.
    """
    st.subheader("Heavy rain")
    st.write(f"This is the ratio of the stations where the rainfall was more than {cut_off/10}mm. So .3 means that at 30% of the stations there was more than this amount of rain for that day. Numbers before 1870 are excluded due the limited number of rainstations") 
    st.write("""
1. Computes the number of unique stations (`a`) for each day.
2. Counts the number of measurements (`b`) that exceed cut_off for each day.
3. Calculates the ratio `c = b / a` for each day,""")
    # Step 1: Calculate the number of unique stations (a) per day
    stations_per_day = (
        df.group_by("YYYYMMDD")
        .agg(pl.col("STN").n_unique().alias("a"))  # Count unique stations
    )
    #st.write(stations_per_day)
    # Step 2: Count the number of measurements (b) above cut_off per day
    measurements_above_cut_off = (
        df.filter(pl.col("RD") > cut_off)
        .group_by("YYYYMMDD")
        .agg(pl.len().alias("b"))  # Count measurements above cut_off
    )
    #st.write(measurements_above_cut_off)
   

    # Step 3:   Combine and calculate c = b/a, and extract year in one chain. sort by year
    #           we take values after 1870, before we get "extreme" values 
    #           probably bc the lack of measurementpoints
    result = (
        stations_per_day
        .join(measurements_above_cut_off, on="YYYYMMDD", how="full")  # Outer join to keep all dates
        
        .with_columns(
            pl.col("b").fill_null(0).alias("b"),  # Fill missing values in b with 0
            (pl.col("b") / pl.col("a")).alias("c"),  # Calculate c = b/a
            pl.col("YYYYMMDD").dt.year().alias("year")  # Extract year directly from YYYYMMDD
        ) .filter(pl.col("year") >= 1880) 
        .sort("YYYYMMDD")
    )

    # Step 4: Plot the result using Plotly
    fig = px.scatter(
        result.to_pandas(),
        x='YYYYMMDD',
        y='c',
        title=f'Ratio of Measurements Above {cut_off/10} mm to Unique Stations Per Day',
        labels={'c': 'Ratio (c = b/a)', 'YYYYMMDD': 'Date'},
    )


    st.plotly_chart(fig)
    
     # Step 5 : Calculate mean ratio per year
    yearly_result = result.group_by("year").agg(
        pl.col("c").mean().alias("mean_c").fill_null(0)  # Calculate mean of c for each year
    ).sort("year")  # Sort by year
    X_array = yearly_result['year']
    Y_array = yearly_result['mean_c']
    t, loess_values, ll, ul = loess_skmisc(X_array, Y_array)
   
    # Step 6: Plot the mean ratio per year using Plotly
    fig = go.Figure()

    # Add a line trace
    fig.add_trace(
        go.Scatter(
            x=yearly_result['year'],        # X-axis values
            y=yearly_result['mean_c'],      # Y-axis values
            mode='lines+markers',               # Show both lines and markers
            name='Mean Ratio',                  # Legend name for the trace
            line=dict(width=2),                 # Line width
            marker=dict(size=3)                 # Marker size
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,        # X-axis values
            y=loess_values,      # Y-axis values
            mode='lines',               # Show both lines and markers
            name='LOESS',                  # Legend name for the trace
            line=dict(width=1),                 # Line width
            
        )
    )

    # Update layout for title and labels
    fig.update_layout(
        title=f'Mean Ratio of Measurements Above {cut_off/10} mm to Unique Stations Per Year',
        xaxis_title='Year',
        yaxis_title='Mean Ratio (c = b/a)',
        template='plotly'                     # Optional: set a template
    )
    st.plotly_chart(fig)
    


    # Step 7: Create a decade column
    result = result.with_columns(
        (pl.col("year") // 10 * 10).alias("decade")  # Floor the year to the nearest decade
    )

    # Step 8: Group by decade and calculate the mean ratio per decade
    decade_result = result.group_by("decade").agg(
        pl.col("c").mean().alias("mean_c")  # Calculate mean of c for each decade
    ).sort("decade")  # Sort by decade

    # Step 9: Plot the mean ratio per decade using Plotly
    fig = px.bar(
        decade_result.to_pandas(),
        x='decade',
        y='mean_c',
        title=f'Mean Ratio of Measurements Above {cut_off/10} mm to Unique Stations Per Decade (2020s until 2024)',
        labels={'mean_c': 'Mean Ratio (c = b/a)', 'decade': 'Decade'},
    )

    st.plotly_chart(fig)
    
def extreme_claude_ai(df, how, treshold):

    """_summary_

    Returns:
        _type_: _description_
    """    
    st.subheader(f"Yearly  {treshold}")
    df = df.with_columns([
     
        pl.col('YYYYMMDD').dt.year().alias('year'),
        pl.col('YYYYMMDD').dt.month().alias('month'),
        pl.col('YYYYMMDD').dt.quarter().alias('season')
    ])

    # 2. Extreme regenval definiëren (bijvoorbeeld: 99e percentiel)
    extreme_thresholds = df.group_by('STN').agg(
        pl.col('RD').quantile(treshold).alias('threshold')
    )

        # 2. Extreme regenval definiëren (> 500 mm)
    def count_extreme_events_mm(df):
        """Count extreme events, based on quantile

        Args:
        df (_type_): _description_

        Returns:
        _type_: _description_
        """        
        return df.filter(pl.col('RD') > treshold).group_by(['year', 'STN']).agg(pl.len().alias('extreme_count'))
    
    # 3. Tijdsreeksen analyseren
    
    def count_extreme_events_quantile(df):
        """Count extreme events, based on rainfall in mm

        Args:
            df (_type_): _description_

        Returns:
            _type_: _description_
        """        
        return df.join(
            extreme_thresholds, on='STN'
        ).filter(
            pl.col('RD') > pl.col('threshold')
        ).group_by(['year', 'STN']).agg(
            pl.len().alias('extreme_count')
        )


    if how == "mm":
        extreme_counts = count_extreme_events_mm(df)
    elif how =="quantile":
        extreme_counts = count_extreme_events_quantile(df)
    else:
        st.error("Error in how")
        st.stop()
    # Gemiddeld aantal extreme gebeurtenissen per jaar (genormaliseerd voor aantal stations)
    yearly_extremes = extreme_counts.group_by('year').agg([
        pl.sum('extreme_count').alias('total_extreme_count'),
        pl.n_unique('STN').alias('station_count')
    ]).with_columns(
        (pl.col('total_extreme_count') / pl.col('station_count')).alias('normalized_extremes')
    )

    # 4. Statistische test (bijvoorbeeld: Mann-Kendall trend test)
    
    mk_result = original_test(yearly_extremes['normalized_extremes'].to_numpy())
    st.write(f"Mann-Kendall test result: trend = {mk_result.trend}, p-value = {mk_result.p}")
    
    # Linear regression
    x = yearly_extremes['year'].to_numpy()
    y = yearly_extremes['normalized_extremes'].to_numpy()
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    st.write(f"Linear regression: slope = {slope}, p-value = {p_value}, R-squared = {r_value**2}")
    plot_yearly_extremes(yearly_extremes, x, y, slope, intercept, mk_result,  r_value, p_value)

def plot_yearly_extremes(yearly_extremes, x, y,slope, intercept, mk_result,  r_value, p_value):
    # 5. Visualisatie met Plotly
    fig = px.scatter(
        yearly_extremes.to_pandas(), 
        x='year', 
        y='normalized_extremes',
        trendline='ols',
        title='Genormaliseerd aantal extreme regenval gebeurtenissen per jaar'
        )


    # Trendlijn (linear regression)
    fig.add_trace(go.Scatter(
        x=yearly_extremes['year'],
        y=slope * x + intercept,
        mode='lines',
        name='Linear trend',
        line=dict(color='red')
    ))

    # LOWESS smoother
    from statsmodels.nonparametric.smoothers_lowess import lowess
    lowess_result = lowess(y, x, frac=0.6)
    fig.add_trace(go.Scatter(
        x=lowess_result[:, 0],
        y=lowess_result[:, 1],
        mode='lines',
        name='LOWESS trend',
        line=dict(color='green')
    ))

    fig.update_layout(
        title='Genormaliseerd aantal extreme regenval gebeurtenissen per jaar',
        xaxis_title='Jaar',
        yaxis_title='Genormaliseerd aantal extreme gebeurtenissen'
    )

    # Voeg annotaties toe met testresultaten
    fig.add_annotation(
        x=0.05, y=0.95, xref='paper', yref='paper',
        text=f"Mann-Kendall:<br>trend = {mk_result.trend}<br>p-value = {mk_result.p:.4f}",
        showarrow=False, font=dict(size=10), align='left',
        bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1
    )
    fig.add_annotation(
        x=0.05, y=0.80, xref='paper', yref='paper',
        text=f"Linear regression:<br>slope = {slope:.4f}<br>p-value = {p_value:.4f}<br>R-squared = {r_value**2:.4f}",
        showarrow=False, font=dict(size=10), align='left',
        bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1
    )

    fig.update_layout(
        xaxis_title='Jaar',
        yaxis_title='Genormaliseerd aantal extreme gebeurtenissen'
    )
    st.plotly_chart(fig)

def extreme_claude_ai_seasonal(df, how, treshold):
    st.subheader(f"Seasonal {treshold}")
    df = df.with_columns([
     
        pl.col('YYYYMMDD').dt.year().alias('year'),
        pl.col('YYYYMMDD').dt.month().alias('month'),
        pl.col('YYYYMMDD').dt.quarter().alias('season')
    ])
    # 2. Extreme regenval definiëren (99e percentiel per station en seizoen)
    extreme_thresholds_seasonal = df.group_by(['STN', 'season']).agg(
        pl.col('RD').quantile(treshold).alias('threshold')
    )

    # 2. Extreme regenval definiëren (> 500 mm)
    def count_extreme_events_seasonal_mm(df):
        return df.filter(
            pl.col('RD') > treshold
        ).group_by(['year', 'season', 'STN']).agg(
            pl.len().alias('extreme_count')
        )
    
    # 3. Tijdsreeksen analyseren
    def count_extreme_events_seasonal_quantile(df):
        return df.join(
            extreme_thresholds_seasonal, on=['STN', 'season']
        ).filter(
            pl.col('RD') > pl.col('threshold')
        ).group_by(['year', 'season', 'STN']).agg(
            pl.len().alias('extreme_count')
        )

    
    if how == "mm":
        extreme_counts_seasonal = count_extreme_events_seasonal_mm(df)
    elif how =="quantile":
        extreme_counts_seasonal = count_extreme_events_seasonal_quantile(df)
    else:
        st.error("Error in how")
        st.stop()

    # Gemiddeld aantal extreme gebeurtenissen per jaar en seizoen (genormaliseerd voor aantal stations)
    yearly_seasonal_extremes = extreme_counts_seasonal.group_by(['year', 'season']).agg([
        pl.sum('extreme_count').alias('total_extreme_count'),
        pl.n_unique('STN').alias('station_count')
    ]).with_columns(
        (pl.col('total_extreme_count') / pl.col('station_count')).alias('normalized_extremes')
    ).sort(['year', 'season'])
    for p,s in zip([1,2,3,4],["Winter", "Spring", "Summer", "Autumn"]):
        # 4. Statistische tests
        # Seasonal Mann-Kendall test
        data = yearly_seasonal_extremes.pivot(index='year', on='season', values='normalized_extremes').sort('year')
        smk_result = seasonal_test(data.to_numpy(), period=p)
        st.write(f"Seasonal Mann-Kendall test result - season {s}: trend = {smk_result.trend}, p-value = {smk_result.p}")

    plot_yearly_extremes_seasonal(yearly_seasonal_extremes)
    # Print samenvattende statistieken
    st.write(yearly_seasonal_extremes.group_by('season').agg([
        pl.col('normalized_extremes').mean().alias('mean'),
        pl.col('normalized_extremes').std().alias('std'),
        pl.col('normalized_extremes').min().alias('min'),
        pl.col('normalized_extremes').max().alias('max')
    ]))

def plot_yearly_extremes_seasonal(yearly_seasonal_extremes):
    # 5. Visualisatie
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Winter", "Spring", "Summer", "Autumn"))

    seasons = [1, 2, 3, 4]
    row_col = [(1,1), (1,2), (2,1), (2,2)]

    for season, (row, col) in zip(seasons, row_col):
        season_data = yearly_seasonal_extremes.filter(pl.col('season') == season)
        x = season_data['year'].to_numpy()
        y = season_data['normalized_extremes'].to_numpy()
        
        # Scatter plot
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=f'Season {season}'), row=row, col=col)
        
        # Trendlijn
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        fig.add_trace(go.Scatter(x=x, y=slope*x + intercept, mode='lines', name=f'Trend S{season}'), row=row, col=col)

    fig.update_layout(height=800, title_text="Seizoensgebonden trends in extreme regenval")
    st.plotly_chart(fig)
    

def main():
    df_neerslag_data_pl = load_and_combine_data()
    col1,col2=st.columns(2)
    with col1:
        st.subheader("MM RAIN")
        extreme_claude_ai(df_neerslag_data_pl, "mm", 500)
        extreme_claude_ai_seasonal(df_neerslag_data_pl,"mm", 500)
    with col2:
        st.subheader("QUANTILE")
        extreme_claude_ai(df_neerslag_data_pl, "quantile", .995)
        extreme_claude_ai_seasonal(df_neerslag_data_pl,"quantile", .995)
    
    if 1==1:
        #date_range(df) #very slow even with polars
        
        df_station_info_pl = show_station_data()
        
        plot_number_of_measurements_per_day(df_neerslag_data_pl)
        
        df_avg_month, df_avg_year = calculate_average_per_day_year(df_neerslag_data_pl)
        
        
        plot_grouped_by_month(df_avg_month)
        plot_graph_with_loess(df_avg_year)
        plot_met_horizontale_lijnen(df_avg_year)
    col1,col2=st.columns(2)
    with col1:
        plot_heavy_rain(df_neerslag_data_pl, 500)
    with col2:
        plot_heavy_rain(df_neerslag_data_pl, 1000)

if __name__ == "__main__":
    #os.system('cls')
    import time
    s1 = int(time.time())
    print(f"--------------{datetime.datetime.now()}-------------------------")
    main()
    s2 = int(time.time())
    s2x = s2 - s1
    print(" ")  # to compensate the  sys.stdout.flush()

    print(f"Processing  took {str(s2x)} seconds ....)")
