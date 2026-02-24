"""
Streamlit wrapper for the City Map Poster Generator.

Run with:
    streamlit run app.py
"""

import io
import json
import os

import matplotlib.pyplot as plt
import streamlit as st

from create_map_poster import get_available_themes,get_available_themes_with_subdirs, load_theme,  generate_poster


def app():
    # ── Pa`ge config ──────────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="City Map Poster Generator",
        page_icon="🗺️",
        layout="centered",
    )

    st.title("🗺️ City Map Poster Generator")
    st.caption("Generate beautiful, minimalist map posters for any city in the world.")

    # ── Sidebar – all inputs ─────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        city = st.text_input("City *", "Paris", placeholder="e.g. Paris")
        country = st.text_input("Country *", "France", placeholder="e.g. France")

        # Theme picker
        #available_themes = get_available_themes_with_subdirs()
        available_themes = get_available_themes()
        if available_themes:
            theme = st.selectbox("Theme", available_themes, index=0)

            # Show theme description if available
            try:
                theme_data = load_theme(theme)
                if "description" in theme_data:
                    st.caption(theme_data["description"])
            except Exception:
                pass
        else:
            st.warning("No themes found in `themes/` directory.")
            theme = "terracotta"

        distance = st.slider(
            "Map radius (meters)",
            min_value=2000,
            max_value=30000,
            value=18000,
            step=500,
            help="Small cities: 4000–6000 m · Medium: 8000–12000 m · Large: 15000–20000 m",
        )

        st.subheader("Canvas size")
        col_w, col_h = st.columns(2)
        with col_w:
            width = st.number_input("Width (in)", min_value=4.0, max_value=20.0, value=12.0, step=0.5)
        with col_h:
            height = st.number_input("Height (in)", min_value=4.0, max_value=20.0, value=16.0, step=0.5)

        output_format = st.selectbox("Output format", ["png", "pdf", "svg"], index=0)

        with st.expander("Advanced options"):
            latitude = st.text_input("Override latitude", placeholder="e.g. 48.8566")
            longitude = st.text_input("Override longitude", placeholder="e.g. 2.3522")
            display_city = st.text_input("Display city name", placeholder="Leave blank to use City")
            display_country = st.text_input("Display country name", placeholder="Leave blank to use Country")
            font_family = st.text_input(
                "Google Font family",
                placeholder='e.g. "Noto Sans JP"',
                help="If blank, uses local Roboto fonts.",
            )

        generate_btn = st.button("🎨 Generate Poster", type="primary", use_container_width=True)

    # ── Main area ─────────────────────────────────────────────────────────────────
    if generate_btn:
        if not city or not country:
            st.error("Please fill in both **City** and **Country** fields.")
        elif not available_themes:
            st.error("No themes found. Make sure the `themes/` directory contains `.json` files.")
        else:
            with st.spinner(f"Generating poster for **{city}, {country}** — this may take a minute…"):
                try:
                    fig, output_file = generate_poster(
                        city=city,
                        country=country,
                        theme=theme,
                        distance=distance,
                        width=width,
                        height=height,
                        output_format=output_format,
                        latitude=latitude or None,
                        longitude=longitude or None,
                        display_city=display_city or None,
                        display_country=display_country or None,
                        font_family=font_family or None,
                    )

                    # ── Display the poster ──
                    st.success("✅ Poster generated!")
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    # ── Download button ──
                    mime_map = {"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml"}
                    buf = io.BytesIO()

                    if output_format == "png":
                        fig_dl, ax_dl = plt.subplots(figsize=(width, height))
                        # Re-read the saved file for download to avoid re-rendering
                        with open(output_file, "rb") as f:
                            buf.write(f.read())
                        plt.close(fig_dl)
                    else:
                        with open(output_file, "rb") as f:
                            buf.write(f.read())

                    buf.seek(0)
                    filename = os.path.basename(output_file)
                    st.download_button(
                        label=f"⬇️ Download {output_format.upper()}",
                        data=buf,
                        file_name=filename,
                        mime=mime_map[output_format],
                        use_container_width=True,
                    )

                except ValueError as e:
                    st.error(f"❌ Configuration error: {e}")
                except RuntimeError as e:
                    st.error(f"❌ Map data error: {e}")
                except Exception as e:
                    st.exception(e)

    else:
        # Placeholder / instructions
        st.info(
            "👈 Fill in the city and country in the sidebar, pick a theme and radius, "
            "then click **Generate Poster**."
        )

        with st.expander("📐 Distance guide"):
            st.markdown(
                """
    | Radius | Best for |
    |--------|----------|
    | 4 000 – 6 000 m | Small / dense cities (Venice, Amsterdam old centre) |
    | 8 000 – 12 000 m | Medium cities, focused downtown (Paris, Barcelona) |
    | 15 000 – 20 000 m | Large metros, full city view (Tokyo, Mumbai) |
    """
            )

        with st.expander("🎨 Example cities to try"):
            st.markdown(
                """
    - **New York** · USA · theme `noir` · 12 000 m  
    - **Tokyo** · Japan · theme `midnight_blue` · 15 000 m  
    - **Venice** · Italy · theme `blueprint` · 4 000 m  
    - **Paris** · France · theme `pastel_dream` · 10 000 m  
    - **Marrakech** · Morocco · theme `terracotta` · 5 000 m  
    """
            )

if __name__ == "__main__":
    app()