import streamlit as st
import plotly.express as px
import pandas as pd

# Function to convert Celsius to Fahrenheit
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

# Function to convert Fahrenheit to Celsius
def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9

# Function to calculate Heat Index
def calculate_heat_index(T, RH):
    HI = (-42.379 + 2.04901523 * T + 10.14333127 * RH 
          - 0.22475541 * T * RH - 0.00683783 * T**2 
          - 0.05481717 * RH**2 + 0.00122874 * T**2 * RH 
          + 0.00085282 * T * RH**2 - 0.00000199 * T**2 * RH**2)
    return HI

# Function to calculate Wind Chill
def calculate_wind_chill(T, V):
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
    temp_celsius = st.sidebar.slider("Select temperature in degrees celcius", min_value=-10, max_value=50, value=25)

    T_F = celsius_to_fahrenheit(temp_celsius)

    data = []
    if T_F >= 80:
        # Calculate Heat Index
        for RH in range (0,101):
            feels_like_F = calculate_heat_index(T_F, RH)
            feels_like_C = fahrenheit_to_celsius(feels_like_F)
        
            data.append({"RH": RH,  "Feels-Like Temperature (°C)": feels_like_C})
        df = pd.DataFrame(data)
        
        # Plotting with Plotly
        fig = px.line(df, x="RH", y="Feels-Like Temperature (°C)", title=f"Feels-Like Temperature vs. Relatively humidity, temp = {temp_celsius} ", labels={"RH": "RH", "Feels-Like Temperature (°C)": "Feels-Like Temperature (°C)"})
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
    