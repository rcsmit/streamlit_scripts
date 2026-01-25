import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
import numpy as np
import time
import json
import os
from datetime import datetime
import streamlit as st
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from show_posters import show_posters
from theme_editor import theme_editor
from organize_svg import organize_svg_with_theme


# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
THEMES_DIR = SCRIPT_DIR / "themes"
FONTS_DIR = SCRIPT_DIR / "fonts"
POSTERS_DIR = SCRIPT_DIR / "posters"

# Ensure directories exist
THEMES_DIR.mkdir(exist_ok=True)
FONTS_DIR.mkdir(exist_ok=True)
POSTERS_DIR.mkdir(exist_ok=True)

# Pre-defined city coordinates (no API needed!)
CITY_COORDINATES = {
    "Da Nang, Vietnam": (16.06778, 108.22083),
    "Hoi An, Vietnam": (15.88006, 108.33804),
    "Stadskanaal, Netherlands":(52.996700, 6.895670),
    "Amsterdam, Netherlands": (52.3676, 4.9041),
    "New York, USA": (40.7128, -74.0060),
    "Paris, France": (48.8566, 2.3522),
    "London, UK": (51.5074, -0.1278),
    "Tokyo, Japan": (35.6762, 139.6503),
    "Barcelona, Spain": (41.3851, 2.1734),
    "Berlin, Germany": (52.5200, 13.4050),
    "Rome, Italy": (41.9028, 12.4964),
    "Sydney, Australia": (-33.8688, 151.2093),
    "Dubai, UAE": (25.2048, 55.2708),
    "Singapore, Singapore": (1.3521, 103.8198),
    "Istanbul, Turkey": (41.0082, 28.9784),
    "Moscow, Russia": (55.7558, 37.6173),
    "Mumbai, India": (19.0760, 72.8777),
    "San Francisco, USA": (37.7749, -122.4194),
    "Los Angeles, USA": (34.0522, -118.2437),
    "Chicago, USA": (41.8781, -87.6298),
    "Toronto, Canada": (43.6532, -79.3832),
    "Mexico City, Mexico": (19.4326, -99.1332),
    "São Paulo, Brazil": (-23.5505, -46.6333),
    "Buenos Aires, Argentina": (-34.6037, -58.3816),
    "Cairo, Egypt": (30.0444, 31.2357),
    "Cape Town, South Africa": (-33.9249, 18.4241),
    "Seoul, South Korea": (37.5665, 126.9780),
    "Bangkok, Thailand": (13.7563, 100.5018),
    "Hong Kong, China": (22.3193, 114.1694),
    "Shanghai, China": (31.2304, 121.4737),
    "Beijing, China": (39.9042, 116.4074),
    "Melbourne, Australia": (-37.8136, 144.9631),
    "Vienna, Austria": (48.2082, 16.3738),
    "Prague, Czech Republic": (50.0755, 14.4378),
    "Copenhagen, Denmark": (55.6761, 12.5683),
    "Stockholm, Sweden": (59.3293, 18.0686),
    "Oslo, Norway": (59.9139, 10.7522),
    "Helsinki, Finland": (60.1699, 24.9384),
    "Athens, Greece": (37.9838, 23.7275),
    "Lisbon, Portugal": (38.7223, -9.1393),
    "Madrid, Spain": (40.4168, -3.7038),
    "Brussels, Belgium": (50.8503, 4.3517),
    "Zurich, Switzerland": (47.3769, 8.5417),
    "Warsaw, Poland": (52.2297, 21.0122),
    "Budapest, Hungary": (47.4979, 19.0402),
    "Dublin, Ireland": (53.3498, -6.2603),
    "Edinburgh, UK": (55.9533, -3.1883),
    "Manchester, UK": (53.4808, -2.2426),
    "Lyon, France": (45.7640, 4.8357),
    "Venice, Italy": (45.4408, 12.3155),
    "Florence, Italy": (43.7696, 11.2558),
    "Milan, Italy": (45.4642, 9.1900),
    "Munich, Germany": (48.1351, 11.5820),
    "Hamburg, Germany": (53.5511, 9.9937),
}

# Timeout configuration
DEFAULT_TIMEOUT = 30  # seconds

def fetch_with_timeout(fetch_func, timeout_seconds, *args, **kwargs):
    """
    Wrapper to fetch data with a timeout using ThreadPoolExecutor.
    Returns None if timeout occurs or fetch fails.
    """
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fetch_func, *args, **kwargs)
            try:
                result = future.result(timeout=timeout_seconds)
                return result
            except FuturesTimeoutError:
                st.warning(f"⏱️ Timeout after {timeout_seconds}s: Skipping this data layer...")
                return None
    except Exception as e:
        st.warning(f"⚠️ Error fetching data: {str(e)}")
        return None

@st.cache_resource
def load_fonts():
    """Load Roboto fonts from the fonts directory."""
    fonts = {
        'bold': FONTS_DIR / 'Roboto-Bold.ttf',
        'regular': FONTS_DIR / 'Roboto-Regular.ttf',
        'light': FONTS_DIR / 'Roboto-Light.ttf'
    }
    
    missing_fonts = []
    for weight, path in fonts.items():
        if not path.exists():
            missing_fonts.append(f"{weight}: {path}")
    
    if missing_fonts:
        st.warning(f"⚠️ Fonts not found. Using system fonts.")
        return None
    
    return {k: str(v) for k, v in fonts.items()}

#@st.cache_data
def get_available_themes():
    """Scans the themes directory and returns a list of available theme names."""
    if not THEMES_DIR.exists():
        return []
    
    themes = []
    for file in sorted(THEMES_DIR.glob("*.json")):
        theme_name = file.stem
        themes.append(theme_name)
    
    return themes

#@st.cache_data
def load_theme(theme_name="feature_based"):
    """Load theme from JSON file in themes directory."""
    theme_file = THEMES_DIR / f"{theme_name}.json"
    
    if not theme_file.exists():
        st.warning(f"⚠️ Theme file not found. Using default theme.")
        return {
            "name": "Feature-Based Shading",
            "bg": "#FFFFFF",
            "text": "#000000",
            "gradient_color": "#FFFFFF",
            "water": "#C0C0C0",
            "parks": "#F0F0F0",
            "road_motorway": "#0A0A0A",
            "road_primary": "#1A1A1A",
            "road_secondary": "#2A2A2A",
            "road_tertiary": "#3A3A3A",
            "road_residential": "#4A4A4A",
            "road_default": "#3A3A3A"
        }
    
    try:
        with open(theme_file, 'r') as f:
            theme = json.load(f)
            return theme
    except Exception as e:
        st.error(f"Error loading theme: {e}")
        return None

def generate_output_filename(city, theme_name, file_format="png"):
    """Generate unique output filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    city_slug = city.lower().replace(' ', '_').replace(',', '')
    filename = f"{city_slug}_{theme_name}_{timestamp}.{file_format}"
    return POSTERS_DIR / filename

def create_gradient_fade(ax, color, location='bottom', zorder=10):
    """Creates a fade effect at the top or bottom of the map."""
    vals = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack((vals, vals))
    
    rgb = mcolors.to_rgb(color)
    my_colors = np.zeros((256, 4))
    my_colors[:, 0] = rgb[0]
    my_colors[:, 1] = rgb[1]
    my_colors[:, 2] = rgb[2]
    
    if location == 'bottom':
        my_colors[:, 3] = np.linspace(1, 0, 256)
        extent_y_start = 0
        extent_y_end = 0.25
    else:
        my_colors[:, 3] = np.linspace(0, 1, 256)
        extent_y_start = 0.75
        extent_y_end = 1.0

    custom_cmap = mcolors.ListedColormap(my_colors)
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    
    y_bottom = ylim[0] + y_range * extent_y_start
    y_top = ylim[0] + y_range * extent_y_end
    
    ax.imshow(gradient, extent=[xlim[0], xlim[1], y_bottom, y_top], 
              aspect='auto', cmap=custom_cmap, zorder=zorder, origin='lower')

def get_edge_colors_by_type(G, theme):
    """Assigns colors to edges based on road type hierarchy."""
    edge_colors = []
    
    for u, v, data in G.edges(data=True):
        highway = data.get('highway', 'unclassified')
        
        if isinstance(highway, list):
            highway = highway[0] if highway else 'unclassified'
        
        if highway in ['motorway', 'motorway_link']:
            color = theme['road_motorway']
        elif highway in ['trunk', 'trunk_link', 'primary', 'primary_link']:
            color = theme['road_primary']
        elif highway in ['secondary', 'secondary_link']:
            color = theme['road_secondary']
        elif highway in ['tertiary', 'tertiary_link']:
            color = theme['road_tertiary']
        elif highway in ['residential', 'living_street', 'unclassified']:
            color = theme['road_residential']
        else:
            color = theme['road_default']
        
        edge_colors.append(color)
    
    return edge_colors

def get_edge_widths_by_type(G):
    """Assigns line widths to edges based on road type."""
    edge_widths = []
    
    for u, v, data in G.edges(data=True):
        highway = data.get('highway', 'unclassified')
        
        if isinstance(highway, list):
            highway = highway[0] if highway else 'unclassified'
        
        if highway in ['motorway', 'motorway_link']:
            width = 1.2
        elif highway in ['trunk', 'trunk_link', 'primary', 'primary_link']:
            width = 1.0
        elif highway in ['secondary', 'secondary_link']:
            width = 0.8
        elif highway in ['tertiary', 'tertiary_link']:
            width = 0.6
        else:
            width = 0.4
        
        edge_widths.append(width)
    
    return edge_widths

def create_poster(city_label, point, dist, theme, fonts, gradient_fade, timeout=DEFAULT_TIMEOUT):
    """Generate the map poster."""
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 1. Fetch Street Network
        status_text.text("📡 Downloading street network...")
        progress_bar.progress(10)
        
        G = fetch_with_timeout(
            ox.graph_from_point,
            timeout,
            point,
            dist=dist,
            dist_type='bbox',
            network_type='all'
        )
        
        if G is None:
            st.error("❌ Failed to download street network. Try reducing the distance or choosing another city.")
            progress_bar.empty()
            status_text.empty()
            return None
            
        progress_bar.progress(40)
        
        # 2. Fetch Water Features
        status_text.text("💧 Downloading water features...")
        water = fetch_with_timeout(
            ox.features_from_point,
            timeout,
            point,
            tags={'natural': 'water', 'waterway': 'riverbank'},
            dist=dist
        )
        progress_bar.progress(60)
        
        # 3. Fetch Parks
        status_text.text("🌳 Downloading parks/green spaces...")
        parks = fetch_with_timeout(
            ox.features_from_point,
            timeout,
            point,
            tags={'leisure': 'park', 'landuse': 'grass'},
            dist=dist
        )
        progress_bar.progress(80)
        
        status_text.text("🎨 Rendering map...")
        
        # Setup Plot
        fig, ax = plt.subplots(figsize=(12, 16), facecolor=theme['bg'])
        ax.set_facecolor(theme['bg'])
        ax.set_position([0, 0, 1, 1])
        
        # Plot Layers
        if water is not None and not water.empty:
            water.plot(ax=ax, facecolor=theme['water'], edgecolor='none', zorder=1)
        if parks is not None and not parks.empty:
            parks.plot(ax=ax, facecolor=theme['parks'], edgecolor='none', zorder=2)
        
        # Roads with hierarchy coloring
        edge_colors = get_edge_colors_by_type(G, theme)
        edge_widths = get_edge_widths_by_type(G)
        
        ox.plot_graph(
            G, ax=ax, bgcolor=theme['bg'],
            node_size=0,
            edge_color=edge_colors,
            edge_linewidth=edge_widths,
            show=False, close=False
        )
        if gradient_fade:
            # Gradients
            create_gradient_fade(ax, theme['gradient_color'], location='bottom', zorder=10)
            create_gradient_fade(ax, theme['gradient_color'], location='top', zorder=10)
        
        # Typography
        if fonts:
            font_main = FontProperties(fname=fonts['bold'], size=60)
            font_sub = FontProperties(fname=fonts['light'], size=22)
            font_coords = FontProperties(fname=fonts['regular'], size=14)
            font_attr = FontProperties(fname=fonts['light'], size=8)
        else:
            font_main = FontProperties(family='sans-serif', weight='bold', size=60)
            font_sub = FontProperties(family='sans-serif', weight='300', size=22)
            font_coords = FontProperties(family='sans-serif', size=14)
            font_attr = FontProperties(family='sans-serif', size=8)
        
        # Parse city and country from label
        parts = city_label.split(',')
        city = parts[0].strip()
        country = parts[1].strip() if len(parts) > 1 else ""
        
        spaced_city = "  ".join(list(city.upper()))
        
        # Bottom text
        ax.text(0.5, 0.14, spaced_city, transform=ax.transAxes,
                color=theme['text'], ha='center', fontproperties=font_main, zorder=11)
        
        ax.text(0.5, 0.10, country.upper(), transform=ax.transAxes,
                color=theme['text'], ha='center', fontproperties=font_sub, zorder=11)
        
        lat, lon = point
        coords = f"{lat:.4f}° N / {abs(lon):.4f}° E" if lat >= 0 else f"{abs(lat):.4f}° S / {abs(lon):.4f}° E"
        if lon < 0:
            coords = coords.replace("E", "W")
        
        ax.text(0.5, 0.07, coords, transform=ax.transAxes,
                color=theme['text'], alpha=0.7, ha='center', fontproperties=font_coords, zorder=11)
        
        ax.plot([0.4, 0.6], [0.125, 0.125], transform=ax.transAxes, 
                color=theme['text'], linewidth=1, zorder=11)
        
        # Attribution
        ax.text(0.98, 0.02, "© OpenStreetMap contributors", transform=ax.transAxes,
                color=theme['text'], alpha=0.5, ha='right', va='bottom', 
                fontproperties=font_attr, zorder=11)
        
        progress_bar.progress(100)
        status_text.text("✅ Complete!")
        
        return fig
        
    except Exception as e:
        status_text.text("")
        progress_bar.empty()
        raise e

# Streamlit App
def main_():
    st.set_page_config(page_title="City Map Poster Generator", page_icon="🗺️", layout="wide")
    
    st.title("🗺️ City Map Poster Generator")
    st.markdown("Generate beautiful minimalist map posters for any city")
    
    # Load resources
    fonts = load_fonts()
    available_themes = get_available_themes()
    
    if not available_themes:
        st.error("⚠️ No themes found! Please add theme JSON files to the 'themes' directory.")
        st.info("Create a file like `themes/noir.json` with color definitions.")
        st.stop()
    
    # Sidebar inputs
    with st.sidebar:
        st.header("⚙️ Settings")
        use_custom = st.checkbox("Use custom coordinates", False, help="Check to enter latitude and longitude manually")
        if not use_custom:
            # City selection from predefined list
            city_label = st.selectbox(
                "Select City",
                options=sorted(CITY_COORDINATES.keys()),
                index=0,
                help="Choose from pre-loaded cities"
            )
        else:
                    # Or enter custom coordinates
            with st.expander("🌍 Use Custom Coordinates"):
                custom_lat = st.number_input("Latitude", -90.0, 90.0, 52.3676, format="%.4f")
                custom_lon = st.number_input("Longitude", -180.0, 180.0, 4.9041, format="%.4f")
                custom_city = st.text_input("Custom City Name", "Custom Location")
                
        
        
        
        theme_name = st.selectbox(
            "Theme", 
            available_themes,
            help="Select a visual theme for your poster"
        )
        
        distance = st.number_input(
            "Distance (meters)", 
            min_value=50, 
            max_value=50000, 
            value=1000, 
            step=1000,
            help="Map radius from city center"
        )
        
        timeout = st.number_input(
            "Timeout (seconds)",
            min_value=5,
            max_value=120,
            value=30,
            step=5,
            help="Maximum time to wait for data downloads"
        )

        gradient_fade = st.checkbox(
            "Add Gradient Fade",
            value=True,
            help="Add gradient fade effect at top and bottom of the poster"
        )
        output_format = st.radio(
            "Output Format",
            options=["PNG", "SVG"], #, "Both"], Both gives a problem because the download buttons dissapear when you download one of them
            index=0,
            help="Choose file format for your poster"
        )
        generate_btn = st.button("🎨 Generate Poster", type="primary")
    
        st.markdown("---")
        st.markdown("**Distance Guide:**")
        st.markdown("- 4,000-6,000m: Small cities")
        st.markdown("- 8,000-12,000m: Medium cities")
        st.markdown("- 15,000-20,000m: Large metros")
        
        
    # Show selected city info
    if not use_custom:
        coords = CITY_COORDINATES[city_label]
        st.info(f"📍 **{city_label}** - Coordinates: {coords[0]:.4f}°, {coords[1]:.4f}°")
    else:
        coords = (custom_lat, custom_lon)
        city_label = f"{custom_city}, Custom"
        st.info(f"📍 **Custom Location** - Coordinates: {custom_lat:.4f}°, {custom_lon:.4f}°")
    
    # Main content area
    if generate_btn:
        try:
            # Load theme
            theme = load_theme(theme_name)
            if theme is None:
                st.stop()
            
            # Generate poster
            st.write(f"🗺️ Generating map for **{city_label}**...")
            fig = create_poster(city_label, coords, distance, theme, fonts,  gradient_fade, timeout)
            
            if fig is None:
                st.stop()
            
            # Display
            st.pyplot(fig)

            # Save files based on selected format
            saved_files = []
            
            if output_format in ["PNG", "Both"]:
                output_file_png = generate_output_filename(city_label.split(',')[0], theme_name, "png")
                fig.savefig(output_file_png, dpi=300, facecolor=theme['bg'], bbox_inches='tight')
                saved_files.append(("PNG", output_file_png, "image/png"))
                st.success(f"✅ PNG saved: {output_file_png.name}")
            
            if output_format in ["SVG", "Both"]:
                # output_file_svg = generate_output_filename(city_label.split(',')[0], theme_name, "svg")

                # fig.savefig(output_file_svg, format='svg', facecolor=theme['bg'], bbox_inches='tight')
                # saved_files.append(("SVG", output_file_svg, "image/svg+xml"))
                # st.success(f"✅ SVG saved: {output_file_svg.name}")


                # Then in your code:
                output_file_svg = generate_output_filename(city_label.split(',')[0], theme_name, "svg")

                fig.savefig(output_file_svg, format='svg', facecolor=theme['bg'], bbox_inches='tight')
                #saved_files.append(("SVG", output_file_svg, "image/svg+xml"))
                #st.success(f"✅ SVG saved: {output_file_svg.name}")

                # Organize the SVG by colors
                output_file_svg_organized = output_file_svg.with_stem(f"{output_file_svg.stem}_organized")
                organized = organize_svg_with_theme(str(output_file_svg), str(output_file_svg_organized), theme)
               
                if organized is not None:
                    saved_files.append(("SVG Organized", output_file_svg_organized, "image/svg+xml"))
                    st.success(f"✅ Organized SVG saved: {output_file_svg_organized.name}")
                else:
                    st.warning("⚠️ Failed to organize SVG by colors")
            
            plt.close(fig)
            
            # Download buttons for all saved files
            if len(saved_files) == 1:
                # Single download button
                file_type, file_path, mime_type = saved_files[0]
                with open(file_path, "rb") as file:
                    st.download_button(
                        label=f"⬇️ Download {file_type}",
                        data=file,
                        file_name=file_path.name,
                        mime=mime_type,
                    
                    )
            else:
                # Multiple download buttons
                col1, col2 = st.columns(2)
                for idx, (file_type, file_path, mime_type) in enumerate(saved_files):
                    with (col1 if idx == 0 else col2):
                        with open(file_path, "rb") as file:
                            st.download_button(
                                label=f"⬇️ Download {file_type}",
                                data=file,
                                file_name=file_path.name,
                                mime=mime_type,
                                use_container_width=True
                            )
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())
    
    else:
        st.info("👈 Configure your settings in the sidebar and click 'Generate Poster'")
        st.sidebar.info("Based on Map to Poster by Ankur Gupta. MIT License. Data from OpenStreetMap. Streamlit app by Rene Smit https://github.com/rcsmit/streamlit_scripts/blob/main/st_maptoposter/st_create_map_poster.py")
       
        # Show available cities
        with st.expander(f"📍 Available Cities ({len(CITY_COORDINATES)})"):
            cols = st.columns(3)
            for idx, city in enumerate(sorted(CITY_COORDINATES.keys())):
                with cols[idx % 3]:
                    st.text(city)

def generate_examples():
    city_label, coords, distance,  gradient_fade, timeout = "Stadskanaal, Netherlands",(52.996700, 6.895670), 1000,False,30
    fonts = load_fonts()
    available_themes = get_available_themes()
    
    if not available_themes:
        st.error("⚠️ No themes found! Please add theme JSON files to the 'themes' directory.")
        st.info("Create a file like `themes/noir.json` with color definitions.")
        st.stop()

    number_of_cols=3    
    cols = st.columns(number_of_cols)
    for i,theme_name in enumerate(available_themes):
       
        with cols[i % number_of_cols]:
            theme = load_theme(theme_name)
            if theme is None:
                st.stop()
            
            # Generate poster
            st.subheader(theme_name)
            fig = create_poster(city_label, coords, distance, theme, fonts,  gradient_fade, timeout)
            
            if fig is None:
                st.stop()
            
            # Display
            st.pyplot(fig)
def main():
    tab1,tab2,tab3,tab4=st.tabs(["Start", "Examples","Galery","Theme Editor"])
    
    with tab1:
        main_()
   
    with tab2:
        st.header("Examples")
        st.info("The examples are made with a small town in the Netherlands due to efficiency reasons")
        if st.button("Show examples"):
            generate_examples()
    with tab3:
        st.header("Galery")
        if st.button("Show Galery"):
            show_posters()
    with tab4:
        theme_editor()

if __name__ == "__main__":
    main()