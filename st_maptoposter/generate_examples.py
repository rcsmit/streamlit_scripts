import streamlit as st

from create_map_poster import get_available_themes, generate_poster


def generate_examples():
    city, country = "Stadskanaal", "Netherlands"
    lat, lon = "52.996700", "6.895670"
    distance = 1000

    available_themes = get_available_themes()

    if not available_themes:
        st.error("⚠️ No themes found! Please add theme JSON files to the 'themes' directory.")
        st.info("Create a file like `themes/noir.json` with color definitions.")
        st.stop()

    number_of_cols = 3
    cols = st.columns(number_of_cols)
    for i, theme_name in enumerate(available_themes):
        with cols[i % number_of_cols]:
            st.subheader(theme_name)
            fig, _ = generate_poster(
                city=city,
                country=country,
                theme=theme_name,
                distance=distance,
                latitude=lat,
                longitude=lon,
            )
            st.pyplot(fig)

if __name__=="main":
    generate_examples()