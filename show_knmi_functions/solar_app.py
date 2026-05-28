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


def solar_wrapper(lat,lon,utc_dt, loc_name, selected_date, selected_time,tz,LOCATIONS):
    
    # ---------------------------------------------------------------------------
    # Sidebar — invoer
    # ---------------------------------------------------------------------------
    st.title(":material/wb_sunny: Solar radiation")
    st.write(f"{loc_name} - lat:{lat},lon:{lon},UTC:{utc_dt}")
    with st.expander(":material/tune: Atmosfeer"):
        pressure    = st.slider("Luchtdruk (hPa)", 800, 1050, 1013)
        temperature = st.slider("Temperatuur (°C)", -20, 50, 20)
        turbidity   = st.slider("Turbiditeit (Linke)", 1.0, 6.0, 2.5, 0.1,
                                help="2 = kristalhelder, 3 = gemiddeld, 5 = wazig/vervuild")

    show_curve = st.toggle(":material/show_chart: Dagcurve tonen", value=True)
    st.caption(f"v{current_version}")

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
            # utc_i is je gegeven UTC tijd (naive)
            utc_i = datetime(selected_date.year, selected_date.month, selected_date.day,
                            selected_time.hour, selected_time.minute, tzinfo=timezone.utc)
            # local_time_i = utc_i.astimezone(tz_i)

                        # utc_dt komt al uit select_time_place() als naive UTC
            utc_aware = utc_dt.replace(tzinfo=timezone.utc)
            local_time_i = utc_aware.astimezone(tz_i)
            r = solar_radiation(utc_i, loc_i["lat"], loc_i["lon"],
                                pressure_hpa=pressure, temperature_c=temperature, turbidity=turbidity)
            rows.append({
                "Locatie":       loc_i["name"],
                
                "Lokale tijd":       local_time_i.strftime("%H:%M"),
                
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