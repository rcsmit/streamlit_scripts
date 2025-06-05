import math
from datetime import date, timedelta
import pytz
import plotly.graph_objects as go
import streamlit as st
from astral import LocationInfo
from astral.sun import sun, azimuth
from astral import Observer
# --- Config ---
# LOCATION = LocationInfo("Venlo", "Netherlands", "Europe/Amsterdam", 51.37, 6.17)
LOCATION = LocationInfo("Koh Phangan", "Thailand", "Asia/Bangkok", 9.731, 99.994)
location_name = "Koh Phangan, Thailand"
TIMEZONE = pytz.timezone(LOCATION.timezone)
LAT, LON = 51.37, 6.17
AFSTAND_KM = 50

# --- Functions ---
def bereken_jaarlijkse_azimuten(startjaar=2025):
    datums = [date(startjaar, 1, 1) + timedelta(days=i) for i in range(365)]
    azimuten = []
    for d in datums:
        s = sun(LOCATION.observer, date=d, tzinfo=TIMEZONE)
        az = azimuth(LOCATION.observer, s["sunset"])
        azimuten.append((d, az))
    return azimuten

def plot_azimut_grafiek(azimuth_data):
    datums = [d for d, _ in azimuth_data]
    azimuten = [a for _, a in azimuth_data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datums, y=azimuten, mode='lines', name='Zonsondergang azimut'))
    fig.update_layout(
        title=f'Zonsondergang Azimut in {location_name} (2025)',
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
    fig.add_trace(go.Scattergeo(lon=[LON], lat=[LAT], mode='markers', marker=dict(size=8, color='red'), name=f'{location_name}'))
    fig.add_trace(go.Scattergeo(lon=[LON, lon_min], lat=[LAT, lat_min], mode='lines', line=dict(width=2, color='blue'), name=f'Min azimut {min_az:.1f}°'))
    fig.add_trace(go.Scattergeo(lon=[LON, lon_max], lat=[LAT, lat_max], mode='lines', line=dict(width=2, color='green'), name=f'Max azimut {max_az:.1f}°'))
    fig.update_layout(
        title=f'Richting van zonsondergang vanuit {location_name} (min/max azimut)',
        geo=dict(scope='europe', projection_type="natural earth", showland=True, landcolor="rgb(240, 240, 240)", showcountries=True, countrycolor="gray")
    )
    st.plotly_chart(fig, use_container_width=True)

def toon_gespiegelde_data(azimuth_data):
    targets = [date(2025, m, 1) for m in range(5, 13)] + [date(2026, 1, 1)]
    for t in targets:
        s_target = sun(LOCATION.observer, date=t, tzinfo=TIMEZONE)
        az_target = azimuth(LOCATION.observer, s_target["sunset"])
        best = min([(d, az) for (d, az) in azimuth_data if d < date(2025, 7, 1)], key=lambda x: abs(x[1] - az_target))
        st.write(f"{t} ({round(az_target, 1)}°) ≈ {best[0]} ({round(best[1], 1)}°)")

def plot_multiple_jaarlijnen():
   
    # Bereken zonsondergang-azimuten per jaar
    def bereken_azimuten_per_jaar(jaar):
        datums = [date(jaar, 1, 1) + timedelta(days=i) for i in range(365)]
        azimuten = []
        for d in datums:
            s = sun(LOCATION.observer, date=d, tzinfo=TIMEZONE)
            az = azimuth(LOCATION.observer, s["sunset"])
            azimuten.append((d.timetuple().tm_yday, az))  # gebruik dagnummer voor x-as
        return azimuten

    # Plot setup
    fig = go.Figure()

    # Genereer lijnen voor 2020 t/m 2029
    for jaar in range(2020, 2030):
        data = bereken_azimuten_per_jaar(jaar)
        dagen = [d for d, _ in data]
        azimuten = [a for _, a in data]
        fig.add_trace(go.Scatter(x=dagen, y=azimuten, mode='lines', name=str(jaar)))

    # Layout
    fig.update_layout(
        title=f"Sunset Azimuth in {location_name} (2020–2029)",
        xaxis_title="Day of Year",
        yaxis_title="Azimuth (degrees from north)",
        yaxis=dict(range=[230, 320]),
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
def bepaalde_dag(DAG, MAAND,startjaar, eindjaar):
        
    # Instellingen: welke datum wil je volgen over de jaren?
    MAAND = 9   # September
    DAG = 1     # 1e dag van de maand
    
    # Verzamel azimuten voor dezelfde dag in verschillende jaren
    jaren = list(range(startjaar, eindjaar + 1))
    azimuthen = []
    for jaar in jaren:
        d = date(jaar, MAAND, DAG)
        s = sun(LOCATION.observer, date=d, tzinfo=TIMEZONE)
        az = azimuth(LOCATION.observer, s["sunset"])
        azimuthen.append(az)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=jaren, y=azimuthen, mode='lines+markers', name='Sunset Azimuth'))

    fig.update_layout(
        title=f'Sunset Azimuth in {location_name} on {DAG:02d}-{MAAND:02d} ({startjaar}–{eindjaar})',
        xaxis_title='Year',
        yaxis_title='Azimuth (degrees from north)',
        yaxis=dict(range=[min(azimuthen) - 1, max(azimuthen) + 1]),
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)


def verschillende_lengtegraden():
        
    summer_date = date(2025, 6, 21)
    winter_date = date(2025, 12, 21)
    latitudes = list(range(-60, 61))  # from -60° to 60°

    azimuth_differences = []

    for lat in latitudes:
        observer = Observer(latitude=lat, longitude=0)
        tz = pytz.UTC
        try:
            summer_az = azimuth(observer, sun(observer, summer_date, tzinfo=tz)["sunset"])
            winter_az = azimuth(observer, sun(observer, winter_date, tzinfo=tz)["sunset"])
            diff = abs(summer_az - winter_az)
            azimuth_differences.append(diff)
        except:
            azimuth_differences.append(None)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=latitudes, y=azimuth_differences, mode='lines+markers'))
    fig.update_layout(
        title="Annual Sunset Azimuth Difference vs Latitude",
        xaxis_title="Latitude (°)",
        yaxis_title="Azimuth Difference (°)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
def main():
# --- App Flow ---
    # st.set_page_config(page_title="Zonsondergang Azimut Venlo 2025", layout="wide")
    st.title(f"Zonsondergang Azimut in {location_name} (2025)")
    st.write(f"Deze app toont de azimut van zonsondergangen in {location_name} voor het jaar 2025.")
    azimuth_data = bereken_jaarlijkse_azimuten()
    datums, azimuten = plot_azimut_grafiek(azimuth_data)
    min_az, max_az = toon_extremen(datums, azimuten)
    plot_richtingenkaart(min_az, max_az)
    toon_gespiegelde_data(azimuth_data)
    plot_multiple_jaarlijnen()
    bepaalde_dag(21, 6, 2000,2030)  # 1 september
    verschillende_lengtegraden()


    st.info("Bekijk ook de [blogpost](https://rene-smit.com/where-does-the-sun-set-a-map-a-chart-and-a-surprise/) voor meer uitleg.")

if __name__ == "__main__":
    
    main()