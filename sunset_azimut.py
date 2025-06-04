import math
from datetime import date, timedelta
import pytz
import plotly.graph_objects as go
import streamlit as st
from astral import LocationInfo
from astral.sun import sun, azimuth

# --- Config ---
VENLO = LocationInfo("Venlo", "Netherlands", "Europe/Amsterdam", 51.37, 6.17)
TIMEZONE = pytz.timezone(VENLO.timezone)
LAT, LON = 51.37, 6.17
AFSTAND_KM = 50

# --- Functions ---
def bereken_jaarlijkse_azimuten(startjaar=2025):
    datums = [date(startjaar, 1, 1) + timedelta(days=i) for i in range(365)]
    azimuten = []
    for d in datums:
        s = sun(VENLO.observer, date=d, tzinfo=TIMEZONE)
        az = azimuth(VENLO.observer, s["sunset"])
        azimuten.append((d, az))
    return azimuten

def plot_azimut_grafiek(azimuth_data):
    datums = [d for d, _ in azimuth_data]
    azimuten = [a for _, a in azimuth_data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datums, y=azimuten, mode='lines', name='Zonsondergang azimut'))
    fig.update_layout(
        title='Zonsondergang Azimut in Venlo (2025)',
        xaxis_title='Datum',
        yaxis_title='Azimut (graden vanaf noorden)',
        yaxis=dict(range=[200, 340]),
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    return datums, azimuten

def toon_extremen(datums, azimuten):
    min_index = azimuten.index(min(azimuten))
    max_index = azimuten.index(max(azimuten))
    datum_min = datums[min_index]
    datum_max = datums[max_index]
    st.write(f"Minimum azimut: {min(azimuten):.2f}° op {datum_min.strftime('%d %B %Y')}")
    st.write(f"Maximum azimut: {max(azimuten):.2f}° op {datum_max.strftime('%d %B %Y')}")
    return min(azimuten), max(azimuten)

def bereken_eindpunt(lat, lon, azimut_graden, afstand_km):
    R = 6371
    azimut_rad = math.radians(azimut_graden)
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lat2 = math.asin(math.sin(lat_rad) * math.cos(afstand_km / R) +
                     math.cos(lat_rad) * math.sin(afstand_km / R) * math.cos(azimut_rad))
    lon2 = lon_rad + math.atan2(math.sin(azimut_rad) * math.sin(afstand_km / R) * math.cos(lat_rad),
                                math.cos(afstand_km / R) - math.sin(lat_rad) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

def plot_richtingenkaart(min_az, max_az):
    lat_min, lon_min = bereken_eindpunt(LAT, LON, min_az, AFSTAND_KM)
    lat_max, lon_max = bereken_eindpunt(LAT, LON, max_az, AFSTAND_KM)
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(lon=[LON], lat=[LAT], mode='markers', marker=dict(size=8, color='red'), name='Venlo'))
    fig.add_trace(go.Scattergeo(lon=[LON, lon_min], lat=[LAT, lat_min], mode='lines', line=dict(width=2, color='blue'), name=f'Min azimut {min_az:.1f}°'))
    fig.add_trace(go.Scattergeo(lon=[LON, lon_max], lat=[LAT, lat_max], mode='lines', line=dict(width=2, color='green'), name=f'Max azimut {max_az:.1f}°'))
    fig.update_layout(
        title='Richting van zonsondergang vanuit Venlo (min/max azimut)',
        geo=dict(scope='europe', projection_type="natural earth", showland=True, landcolor="rgb(240, 240, 240)", showcountries=True, countrycolor="gray")
    )
    st.plotly_chart(fig, use_container_width=True)

def toon_gespiegelde_data(azimuth_data):
    targets = [date(2025, m, 1) for m in range(5, 13)] + [date(2026, 1, 1)]
    for t in targets:
        s_target = sun(VENLO.observer, date=t, tzinfo=TIMEZONE)
        az_target = azimuth(VENLO.observer, s_target["sunset"])
        best = min([(d, az) for (d, az) in azimuth_data if d < date(2025, 7, 1)], key=lambda x: abs(x[1] - az_target))
        st.write(f"{t} ({round(az_target, 1)}°) ≈ {best[0]} ({round(best[1], 1)}°)")

def main():
# --- App Flow ---
    //st.set_page_config(page_title="Zonsondergang Azimut Venlo 2025", layout="wide")
    st.title("Zonsondergang Azimut in Venlo (2025)")
    st.write("Deze app toont de azimut van zonsondergangen in Venlo voor het jaar 2025.")
    azimuth_data = bereken_jaarlijkse_azimuten()
    datums, azimuten = plot_azimut_grafiek(azimuth_data)
    min_az, max_az = toon_extremen(datums, azimuten)
    plot_richtingenkaart(min_az, max_az)
    toon_gespiegelde_data(azimuth_data)
    st.info("https://rene-smit.com/where-does-the-sun-set-a-map-a-chart-and-a-surprise/")
if __name__ == "__main__":
    
    main()