
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import streamlit as st

def select_time_place() -> tuple[float, float, datetime]:
    """Streamlit UI widget voor locatie- en tijdselectie.

    Toont twee kolommen:
    - Links: locatieselectie via dropdown (23 vooringestelde steden + custom).
              Bij custom: vrije invoer van breedtegraad, lengtegraad en tijdzone.
    - Rechts: datum- en tijdinvoer (lokale tijd, stapgrootte 15 min).

    Converteert de gekozen lokale datum/tijd naar een naive UTC datetime.

    Returns:
        lat:    Geografische breedte [°N].
        lon:    Geografische lengte [°E].
        utc_dt: Naive UTC datetime van de gekozen lokale datum/tijd.

    Raises:
        st.stop(): als de opgegeven tijdzone onbekend is.
    """
    LOCATIONS = [
        {"name": "De Bilt",      "lat":  52.047,    "lon":  5.177, "timezone": "Europe/Amsterdam"},
    
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

   
    col1,col2=st.columns(2)

    with col1:
        
        loc_name = st.selectbox("Locatie", LOC_NAMES, index=0)
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
    
    return lat,lon,utc_dt, loc_name, selected_date, selected_time,tz,LOCATIONS