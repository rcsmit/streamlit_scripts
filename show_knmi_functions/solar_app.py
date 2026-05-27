"""
Streamlit wrapper voor de solar radiation calculator.
"""

# version : 20260527-130000 - Initial version: Streamlit UI voor solar_radiation()
# version : 20260527-131000 - Vervangen pytz door zoneinfo (stdlib, Python 3.9+)
current_version = "20260527-131000"

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import streamlit as st

try:
    from solar_radiation import solar_radiation
except:
    from show_knmi_functions.solar_radiation import solar_radiation
# ---------------------------------------------------------------------------
# Configuratie
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Solar radiation calculator",
    page_icon=":material/wb_sunny:",
    layout="wide",
)


def utc_decimal_to_local_str(utc_decimal: float | None, tz: ZoneInfo, ref_date: date) -> str:
    """Converteer decimaal UTC-uur naar lokale tijd string."""
    if utc_decimal is None:
        return "—"
    total_min = round(utc_decimal * 60)
    ref_dt = datetime(ref_date.year, ref_date.month, ref_date.day, tzinfo=timezone.utc)
    local_dt = (ref_dt + timedelta(minutes=total_min)).astimezone(tz)
    return local_dt.strftime("%H:%M")


def compute_day_curve(lat: float, lon: float, selected_date: date, tz: ZoneInfo) -> pd.DataFrame:
    """Bereken GHI/DNI/DHI per kwartier over de dag (UTC)."""
    rows = []
    for hour in range(24):
        for minute in (0, 15, 30, 45):
            dt = datetime(selected_date.year, selected_date.month, selected_date.day, hour, minute)
            r = solar_radiation(dt, lat, lon)
            local_dt = datetime(selected_date.year, selected_date.month, selected_date.day,
                                hour, minute, tzinfo=timezone.utc).astimezone(tz)
            rows.append({
                "tijd (lokaal)": local_dt.strftime("%H:%M"),
                "GHI (W/m²)": r["clear_sky_ghi"],
                "DNI (W/m²)": r["clear_sky_dni"],
                "DHI (W/m²)": r["clear_sky_dhi"],
                "elevatie (°)": max(0.0, r["solar_elevation_deg"]),
            })
    return pd.DataFrame(rows)


def solar_wrapper():
    LOCATIONS = [
        {"name": "Koh Phangan",      "lat":   9.7551,  "lon":  99.9609, "timezone": "Asia/Bangkok"},
        {"name": "Koh Chang",        "lat":  12.1036,  "lon": 102.3519, "timezone": "Asia/Bangkok"},
        {"name": "Chiang Mai",       "lat":  18.7932,  "lon":  98.9774, "timezone": "Asia/Bangkok"},
        {"name": "Da Nang",          "lat":  16.0471,  "lon": 108.2062, "timezone": "Asia/Ho_Chi_Minh"},
        {"name": "Ubud",             "lat":  -8.4960,  "lon": 115.2248, "timezone": "Asia/Makassar"},
        {"name": "Amsterdam",        "lat":  52.3676,  "lon":   4.9041, "timezone": "Europe/Amsterdam"},
        {"name": "Lisbon",           "lat":  38.7169,  "lon":  -9.1399, "timezone": "Europe/Lisbon"},
        {"name": "Rome",             "lat":  41.9102,  "lon":  12.3712, "timezone": "Europe/Rome"},
        {"name": "Venezia",          "lat":  45.4408,  "lon":  12.3155, "timezone": "Europe/Rome"},
        {"name": "Hoi An",           "lat":  15.8801,  "lon": 108.3380, "timezone": "Asia/Ho_Chi_Minh"},
        {"name": "Ho Chi Minh City", "lat":  10.7769,  "lon": 106.7009, "timezone": "Asia/Ho_Chi_Minh"},
        {"name": "Hanoi",            "lat":  21.0285,  "lon": 105.8542, "timezone": "Asia/Bangkok"},
        {"name": "Manila",           "lat":  14.5995,  "lon": 120.9842, "timezone": "Asia/Manila"},
        {"name": "Taipei",           "lat":  25.0330,  "lon": 121.5654, "timezone": "Asia/Taipei"},
        {"name": "Kathmandu",        "lat":  27.7172,  "lon":  85.3240, "timezone": "Asia/Kathmandu"},
        {"name": "Colombo",          "lat":   6.9271,  "lon":  79.8612, "timezone": "Asia/Colombo"},
        {"name": "London",           "lat":  51.5072,  "lon":  -0.1276, "timezone": "Europe/London"},
        {"name": "New York",         "lat":  40.7128,  "lon": -74.0060, "timezone": "America/New_York"},
        {"name": "Tokyo",            "lat":  35.6762,  "lon": 139.6503, "timezone": "Asia/Tokyo"},
        {"name": "Sydney",           "lat": -33.8688,  "lon": 151.2093, "timezone": "Australia/Sydney"},
        {"name": "Cape Town",        "lat": -33.9249,  "lon":  18.4241, "timezone": "Africa/Johannesburg"},
        {"name": "São Paulo",        "lat": -23.5505,  "lon": -46.6333, "timezone": "America/Sao_Paulo"},
        {"name": "Istanbul",         "lat":  41.0082,  "lon":  28.9784, "timezone": "Europe/Istanbul"},
        {"name": "— custom —",       "lat":  52.3676,  "lon":   4.9041, "timezone": "UTC"},
    ]

    LOC_NAMES = [loc["name"] for loc in LOCATIONS]

    # ---------------------------------------------------------------------------
    # Sidebar — invoer
    # ---------------------------------------------------------------------------
    st.title(":material/wb_sunny: Solar radiation")

    col1,col2=st.columns(2)

    with col1:
        
        loc_name = st.selectbox("Locatie", LOC_NAMES, index=5)
        loc = next(l for l in LOCATIONS if l["name"] == loc_name)
        if loc_name == "— custom —":
            lat = st.number_input("Breedtegraad", value=52.3676, min_value=-90.0, max_value=90.0, step=0.0001, format="%.4f")
            lon = st.number_input("Lengtegraad",  value=4.9041,  min_value=-180.0, max_value=180.0, step=0.0001, format="%.4f")
            tz_name = st.text_input("Tijdzone (bijv. Europe/Amsterdam)", value="UTC")
        else:
            lat = loc["lat"]
            lon = loc["lon"]
            tz_name = loc["timezone"]
            st.caption(f"📍 {lat:.4f}°, {lon:.4f}°")
            st.caption(f"🕐 {tz_name}")
    with col2:
        selected_date = st.date_input("Datum", value=date.today())
        selected_time = st.time_input("Tijd (lokaal)", value=time(12, 0), step=900)

    with st.expander(":material/tune: Atmosfeer"):
        pressure    = st.slider("Luchtdruk (hPa)", 800, 1050, 1013)
        temperature = st.slider("Temperatuur (°C)", -20, 50, 20)
        turbidity   = st.slider("Turbiditeit (Linke)", 1.0, 6.0, 2.5, 0.1,
                                help="2 = kristalhelder, 3 = gemiddeld, 5 = wazig/vervuild")

    show_curve = st.toggle(":material/show_chart: Dagcurve tonen", value=True)
    st.caption(f"v{current_version}")

    # ---------------------------------------------------------------------------
    # Tijdzone laden
    # ---------------------------------------------------------------------------
    try:
        tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        st.error(f"Onbekende tijdzone: `{tz_name}`")
        st.stop()

    # ---------------------------------------------------------------------------
    # Berekening
    # ---------------------------------------------------------------------------
    local_dt = datetime(
        selected_date.year, selected_date.month, selected_date.day,
        selected_time.hour, selected_time.minute,
        tzinfo=tz,
    )
    utc_dt = local_dt.astimezone(timezone.utc).replace(tzinfo=None)

    result = solar_radiation(utc_dt, lat, lon, pressure_hpa=pressure,
                            temperature_c=temperature, turbidity=turbidity)

    # ---------------------------------------------------------------------------
    # Hoofdpagina — uitvoer
    # ---------------------------------------------------------------------------
    st.title(f":material/wb_sunny: {loc_name}")
    st.caption(
        f"{selected_date.strftime('%d %B %Y')} · "
        f"{selected_time.strftime('%H:%M')} lokaal · "
        f"{utc_dt.strftime('%H:%M')} UTC · "
        f"dag {result['day_of_year']}"
    )

    # --- KPI rij ---
    zon_op   = utc_decimal_to_local_str(result["sunrise_utc"],    tz, selected_date)
    zon_on   = utc_decimal_to_local_str(result["sunset_utc"],     tz, selected_date)
    zon_noon = utc_decimal_to_local_str(result["solar_noon_utc"], tz, selected_date)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Zonne-elevatie", f"{result['solar_elevation_deg']:.1f}°",
                delta=f"azimut {result['solar_azimuth_deg']:.0f}°",
                delta_color="off", border=True)
    with col2:
        st.metric("Clear-sky GHI", f"{result['clear_sky_ghi']:.0f} W/m²",
                delta=f"DNI {result['clear_sky_dni']:.0f} · DHI {result['clear_sky_dhi']:.0f}",
                delta_color="off", border=True)
    with col3:
        st.metric("Zonsopkomst / -ondergang", f"{zon_op} → {zon_on}",
                delta=f"zonsmiddag {zon_noon}", delta_color="off", border=True)
    with col4:
        st.metric("Extraterrestr. straling", f"{result['extraterrestrial_irradiance']:.0f} W/m²",
                delta=f"tijdsvergelijking {result['equation_of_time_min']:.1f} min",
                delta_color="off", border=True)

    st.space("small")

    # --- Dagcurve ---
    if show_curve:
        with st.spinner("Dagcurve berekenen..."):
            df_curve = compute_day_curve(lat, lon, selected_date, tz)

        tab1, tab2 = st.tabs([":material/wb_sunny: Straling (W/m²)", ":material/north: Zonne-elevatie (°)"])
        with tab1:
            st.line_chart(
                df_curve.set_index("tijd (lokaal)")[["GHI (W/m²)", "DNI (W/m²)", "DHI (W/m²)"]],
                color=["#f59e0b", "#ef4444", "#60a5fa"],
            )
        with tab2:
            st.area_chart(
                df_curve.set_index("tijd (lokaal)")[["elevatie (°)"]],
                color=["#f59e0b"],
            )
        st.space("small")

    # --- Detailtabel ---
    with st.expander(":material/table: Alle waarden"):
        detail = {"Waarde": {
            "Breedtegraad":            f"{lat:.4f}°",
            "Lengtegraad":             f"{lon:.4f}°",
            "Datum/tijd UTC":          str(result["datetime_utc"]),
            "Dag van het jaar":        result["day_of_year"],
            "Zonne-elevatie":          f"{result['solar_elevation_deg']:.3f}°",
            "Azimut (N CW)":           f"{result['solar_azimuth_deg']:.3f}°",
            "Zenith":                  f"{result['zenith_deg']:.3f}°",
            "Tijdsvergelijking":       f"{result['equation_of_time_min']:.2f} min",
            "Zonsmiddag (lokaal)":     zon_noon,
            "Zonsopkomst (lokaal)":    zon_op,
            "Zonsondergang (lokaal)":  zon_on,
            "Extraterrestr. straling": f"{result['extraterrestrial_irradiance']:.2f} W/m²",
            "Clear-sky GHI":           f"{result['clear_sky_ghi']:.2f} W/m²",
            "Clear-sky DNI":           f"{result['clear_sky_dni']:.2f} W/m²",
            "Clear-sky DHI":           f"{result['clear_sky_dhi']:.2f} W/m²",
        }}
        st.dataframe(pd.DataFrame(detail), width="stretch")
    with st.expander("Uitleg"):
        st.info("GHI — Global Horizontal Irradiance. Gebruikt voor WBGT. Totale zonnestraling op een horizontaal vlak: direct + diffuus. Dit is wat een plat oppervlak (grond, dak) ontvangt.")
        st.info("DNI — Direct Normal Irradiance. Alleen de directe zonnestraling, gemeten loodrecht op de zon. Relevant voor zonnepanelen die de zon volgen, of voor het berekenen van schaduwen.")
        st.info("DHI — Diffuse Horizontal Irradiance. Alleen het verstrooide hemelslicht (wolken, atmosfeer), zonder de directe zonnestraling. GHI = DNI × cos(zenith) + DHI.)")
    # --- Vergelijking alle locaties ---
    with st.expander(":material/compare: Vergelijk alle locaties (zelfde datum/tijd)"):
        rows = []
        for loc_i in LOCATIONS:
            if loc_i["name"] == "— custom —":
                continue
            tz_i = ZoneInfo(loc_i["timezone"])
            local_i = datetime(selected_date.year, selected_date.month, selected_date.day,
                            selected_time.hour, selected_time.minute, tzinfo=tz_i)
            utc_i = local_i.astimezone(timezone.utc).replace(tzinfo=None)
            r = solar_radiation(utc_i, loc_i["lat"], loc_i["lon"],
                                pressure_hpa=pressure, temperature_c=temperature, turbidity=turbidity)
            rows.append({
                "Locatie":       loc_i["name"],
                "Elevatie (°)":  r["solar_elevation_deg"],
                "GHI (W/m²)":   r["clear_sky_ghi"],
                "DNI (W/m²)":   r["clear_sky_dni"],
                "DHI (W/m²)":   r["clear_sky_dhi"],
                "Zonsopkomst":   utc_decimal_to_local_str(r["sunrise_utc"], tz_i, selected_date),
                "Zonsondergang": utc_decimal_to_local_str(r["sunset_utc"],  tz_i, selected_date),
            })

        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Elevatie (°)": st.column_config.NumberColumn(format="%.1f°"),
                "GHI (W/m²)":  st.column_config.ProgressColumn(min_value=0, max_value=1200, format="%.0f"),
                "DNI (W/m²)":  st.column_config.NumberColumn(format="%.0f"),
                "DHI (W/m²)":  st.column_config.NumberColumn(format="%.0f"),
            },
        )

def main():
    solar_wrapper()

if __name__ == "__main__":
    main()