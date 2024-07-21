import streamlit as st
import plotly.express as px
import pandas as pd

import plotly.graph_objs as go

# Function to convert Celsius to Fahrenheit
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

# Function to convert Fahrenheit to Celsius
def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9

# Function to calculate Heat Index
def calculate_heat_index(T, RH):
    # https://wonder.cdc.gov/wonder/help/Climate/ta_htindx.PDF
    HI = (-42.379 + 2.04901523 * T + 10.14333127 * RH 
          - 0.22475541 * T * RH - 0.00683783 * T**2 
          - 0.05481717 * RH**2 + 0.00122874 * T**2 * RH 
          + 0.00085282 * T * RH**2 - 0.00000199 * T**2 * RH**2)
    return HI

# Function to calculate Wind Chill
def calculate_wind_chill(T, V):
    # https://unidata.github.io/MetPy/v0.10/_static/FCM-R19-2003-WindchillReport.pdf

    WC = 35.74 + 0.6215 * T - 35.75 * (V**0.16) + 0.4275 * T * (V**0.16)
    return WC

# Wind speed conversion to Beaufort scale
def beaufort_to_kmh(beaufort):
    scale = [0, 1, 4, 8, 13, 19, 25, 32, 39, 47, 55, 64, 73]
    return scale[beaufort]

def main():
    # Streamlit App
    st.title("Feels-Like Temperature Calculator")

    # User inputs
    temp_celsius = st.sidebar.number_input("Select temperature in degrees celcius", min_value=-10, max_value=50, value=30)

    T_F = celsius_to_fahrenheit(temp_celsius)

    data = []
    if T_F >= 80:
        # Calculate Heat Index
        RH_input = st.sidebar.number_input("Relative humidity", min_value=0, max_value=100, value=70)
        feels_like_F_ = calculate_heat_index(T_F, RH_input)
        feels_like_C_low_ = fahrenheit_to_celsius(feels_like_F_-1.3)  
        feels_like_C_high_ =  fahrenheit_to_celsius(feels_like_F_+1.3)  
        feels_like_C_ = fahrenheit_to_celsius(feels_like_F_)
        st.sidebar.info(f"Feels like : {round(feels_like_C_,1)} [{round(feels_like_C_low_,1)}-{round(feels_like_C_high_,1)}] ")
        for RH in range (0,101):
            feels_like_F = calculate_heat_index(T_F, RH)
            feels_like_C = round(fahrenheit_to_celsius(feels_like_F),2)
            feels_like_C_low = round(fahrenheit_to_celsius(feels_like_F-1.7),2)  
            feels_like_C_high =  round(fahrenheit_to_celsius(feels_like_F+1.7),2)   
            data.append({"RH": RH,  "Feels-Like Temperature (°C)": feels_like_C, "low": feels_like_C_low, "high": feels_like_C_high})
        df = pd.DataFrame(data)
        
       
            
        # Create the line for Feels-Like Temperature
        line = go.Scatter(
            x=df['RH'], 
            y=df['Feels-Like Temperature (°C)'], 
            mode='lines', 
            name='Feels-Like Temperature (°C)'
        )

        # Create the filled area between low and high
        fill = go.Scatter(
            x=df['RH'].tolist() + df['RH'][::-1].tolist(), 
            y=df['high'].tolist() + df['low'][::-1].tolist(), 
            fill='toself', 
            fillcolor='rgba(128, 128, 128, 0.2)', 
            line=dict(color='rgba(255, 255, 255, 0)'), 
            showlegend=False,
            name='Range'
        )

        # Create the layout
        layout = go.Layout(
            title='Feels-Like Temperature vs. Relative Humidity',
            xaxis=dict(title='Relative Humidity (%)'),
            yaxis=dict(title='Feels-Like Temperature (°C)'),
            showlegend=True,
           
        
        )

        # Create the figure and add traces
        fig = go.Figure(data=[fill, line], layout=layout)

        # Show the plot

        st.plotly_chart(fig)
    elif T_F <= 50:
        # Calculate Wind Chill

        for beaufort in range(0, 13):
            wind_speed_kmh = beaufort_to_kmh(beaufort)
            V_mph = wind_speed_kmh * 0.621371
            if  V_mph >= 3:
                feels_like_F = calculate_wind_chill(T_F, V_mph)
                feels_like_C = fahrenheit_to_celsius(feels_like_F)
            else:
                feels_like_C = temp_celsius

            data.append({"Beaufort": beaufort,  "Feels-Like Temperature (°C)": feels_like_C})

        df = pd.DataFrame(data)
        
        # Plotting with Plotly
        fig = px.line(df, x="Beaufort", y="Feels-Like Temperature (°C)", title="Feels-Like Temperature vs. Wind Speed (Beaufort Scale)", labels={"Beaufort": "Wind Speed (Beaufort Scale)", "Feels-Like Temperature (°C)": "Feels-Like Temperature (°C)"})
        st.plotly_chart(fig)
    
    else:
        st.info("Temperature is in ragnge [10-26.7°C]. Feel like temperarure is same as actual temperature ")
if __name__ == "__main__":
    main()
    