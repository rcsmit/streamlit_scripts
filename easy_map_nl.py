import polars as pl
import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np

try:
    st.set_page_config(
        page_title="EasyMap",
        layout="wide"
    )
except:
    pass
def generate_data():
    """
    group the cities and take the average of the lat and lon
    https://github.com/LJPc-solutions/Nederlandse-adressen-en-postcodes

    returns
        df  : df with the data
    """
    # CSV inladen
    df = pl.read_csv(r"C:\Users\rcxsm\Downloads\adressen.csv")

    # Groeperen op woonplaats en gemiddelde latitude/longitude nemen
    df_grouped = (
        df.group_by("woonplaats")
        .agg([
            pl.col("latitude").mean().alias("avg_latitude"),
            pl.col("longitude").mean().alias("avg_longitude")
        ])
    )

    # Wegschrijven naar CSV
    df_grouped.write_csv("woonplaats_lat_lon.csv")

    print(df_grouped)

def hex_to_rgba(hex_code: str, alpha: int = 255) -> list[int]:
    """Convert hex code (#RRGGBB) to [R, G, B, A]."""
    hex_code = hex_code.lstrip("#")
    if len(hex_code) != 6:
        raise ValueError("Hex code must be in format #RRGGBB")
    r = int(hex_code[0:2], 16)
    g = int(hex_code[2:4], 16)
    b = int(hex_code[4:6], 16)
    return [r, g, b, alpha]

@st.cache_data()
def get_data():
    #file= r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\woonplaats_lat_lon.csv"
    file = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/woonplaats_lat_lon.csv"
    df = pl.read_csv(file )
    return df


def main():
    data = get_data()
    
    # Stabiele options-lijst
    options = sorted(data.get_column("woonplaats").unique().to_list())

    # Init default één keer
    woonplaatsen_gekozen = ["Amsterdam", "Rotterdam"]

    woonplaatsen_gekozen = st.sidebar.multiselect(
        "Voer plaatsen in",
        options=options,
        default=woonplaatsen_gekozen,
        
    )
    map_style = st.sidebar.selectbox("Kaartstijl", ["light", "dark", "road", "satellite", "dark_no_labels","light_no_labels"],5)
    col1,col2=st.sidebar.columns(2)
    with col1:
        color_dots = hex_to_rgba( st.sidebar.color_picker("Stippen kleur", "#FF0000"))
    with col2:
        color_txt = hex_to_rgba( st.sidebar.color_picker("Tekst kleur", "#000000"))
    angle_txt = st.sidebar.number_input("Hoek tekst", 0,360,0)
    font_family = st.sidebar.selectbox("Lettertype", ["serif","sans-serif","monospace","cursive","fantasy","system-ui"],2)
    size_scale = st.sidebar.number_input("Text grootte", .1,10.,.5,.1)
    text_anchor = st.sidebar.selectbox(
        "Horizontale uitlijning",
        options=["start", "middle", "end"],
        index=1  # default = "middle"
    )

    # # Word break mode
    # word_break = st.selectbox(
    #     "Word Break",
    #     options=["break-word", "break-all"],
    #     index=0
    # )

    # # Max width (unitless, multiplied with font size)
    # max_width = st.slider(
    #     "Max Width (relative to font size)",
    #     min_value=-1,   # -1 = geen limiet
    #     max_value=100,
    #     value=-1,
    #     step=1
    # )

    # Alignment baseline options
    alignment_baseline = st.sidebar.selectbox(
        "Verticale uitlijning",
        options=["top", "center", "bottom"],
        index=2  # default = "bottom"
    )

    # Filter in Polars
    df_map = data.filter(pl.col("woonplaats").is_in(woonplaatsen_gekozen))

    # Niks gekozen of lege set
    if df_map.height == 0:
        st.warning("Geen resultaten voor de gekozen plaatsen")
        st.stop()


    # Zorg voor kolommen 'lat' en 'lon' voor pydeck

   
    # Converteer naar pandas voor pydeck
    pdf = df_map.to_pandas()

    # Midpoint
    midpoint = (np.average(pdf["lat"]), np.average(pdf["lon"]))

    # Optioneel: Mapbox token
    # pdk.settings.mapbox_api_key = "YOUR_MAPBOX_TOKEN"

    tooltip = {"html": "{woonplaats}"}

    layer_points = pdk.Layer(
        "ScatterplotLayer",
        pdf,
        get_position=["lon", "lat"],
        auto_highlight=True,
        get_radius=4000,
        get_fill_color=color_dots,
        pickable=True,
    )

    layer_text = pdk.Layer(
        "TextLayer",
        pdf,
        pickable=False,
        get_position=["lon", "lat"],
        get_text="woonplaats",
        get_color=color_txt,
        get_angle=angle_txt,
        fontFamily = f'"{font_family}"',
        sizeScale=size_scale,
        getTextAnchor=f'"{text_anchor}"',
        get_alignment_baseline=f'"{alignment_baseline}"',
        # wordBreak=word_break,
        # maxWidth=max_width,
    )

    deck = pdk.Deck(
            map_style=map_style,
            initial_view_state=pdk.ViewState(
                longitude=midpoint[1],
                latitude=midpoint[0],
                pitch=0,
                zoom=6,
            ),
            layers=[layer_points, layer_text],
            tooltip=tooltip,
        )
    
    # In Streamlit tonen
    st.pydeck_chart(deck)


    if st.button("Opslaan als HTML"):
        
        # Opslaan als HTML
        deck.to_html("map.html", notebook_display=False)

    st.info("Gebruik [WIN]-[Shift]-[S] om een gebied te selecteren en naar het klembord te kopieeren")
    st.info("Bron coördinaten: https://github.com/LJPc-solutions/Nederlandse-adressen-en-postcodes")

if __name__ == "__main__":
    main()
