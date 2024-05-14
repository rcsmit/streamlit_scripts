import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
# from show_knmi_functions.utils import get_data

def last_day(df, gekozen_weerstation, what_to_show_, value):
    """Make a plot that shows the last day that the minimum temperature was
    0 degrees.

    Inspired by a plot in the Volkskrant 14th May 2024
    

    Args:
        df (_type_): _description_
        gekozen_weerstation (_type_): _description_
        what_to_show_ (_type_): _description_
        value (_type_): _description_
    """    
    import matplotlib.pyplot as plt

    # Assuming 'df' is your DataFrame with 'date' and what_to_show_ columns
    # Convert 'date' to datetime and extract year and day of year
    
    df['date'] = pd.to_datetime(df["YYYYMMDD"].astype(str))
    df['year_'] = df['date'].dt.year
    df['day_of_year'] = df['date'].dt.dayofyear
    print (df)
    # Filter rows where temperature reaches the value
    df = df.dropna(subset=what_to_show_)
    #zero_temp_df = df[df[what_to_show_] >= 0.0]
    zero_temp_df = df.query('temp_min<=0.0 & day_of_year <200' )
    print ("zero_temp_df")
    print (zero_temp_df)
    # Find the first occurrence of 0 temperature for each year
    first_zero_temp = zero_temp_df.groupby('year_')['day_of_year'].max().reset_index()
    print (first_zero_temp)
    # Plotting
    
    # Calculate the 30-year moving average
    moving_avg = first_zero_temp['day_of_year'].rolling(window=30, center=False).mean()

    # Create a line plot
    fig = go.Figure()

    # Add a scatter plot trace
    fig.add_trace(go.Scatter(x=first_zero_temp['year_'],
                            y=moving_avg,
                            mode='lines',  # Use markers for scatter plot
                            marker=dict(color='black'),  # Set marker color to blue
                            name='SMA 30 years'))

    fig.add_trace(go.Scatter(x=first_zero_temp['year_'],
                            y=first_zero_temp['day_of_year'],
                            mode='lines',  # Use markers for scatter plot
                            marker=dict(color='blue'),  # Set marker color to blue
                            name='Last Day of Zero Temperature'))

    # Add a horizontal bar between y = 141 and y = 144
    # "Ijsheiligen"
    fig.add_shape(
        type="rect",
        xref="paper", yref="y",
        x0=0, y0=131,
        x1=1, y1=134,
        fillcolor="gray",
        opacity=0.3,
        layer="below",
        line=dict(width=0),
        name ='IJsheiligen',
    )
    # Update layout
    fig.update_layout(title='Last Day of Zero Temperature Each Year (1900-2023)',
                    xaxis_title='Year',
                    yaxis_title='Day of the Year',
                    showlegend=True,
                    xaxis=dict(tickmode='linear'))  # Ensure linear tick mode for x-axis

    # Show plot
    st.plotly_chart(fig)
