import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
import numpy as np
from geopy.geocoders import Nominatim
from tqdm import tqdm
import time
import json
import os
from datetime import datetime
import streamlit as st
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
THEMES_DIR = SCRIPT_DIR / "themes"
FONTS_DIR = SCRIPT_DIR / "fonts"
POSTERS_DIR = SCRIPT_DIR / "posters"

# Ensure directories exist
THEMES_DIR.mkdir(exist_ok=True)
FONTS_DIR.mkdir(exist_ok=True)
POSTERS_DIR.mkdir(exist_ok=True)

@st.cache_resource
def load_fonts():
    """
    Load Roboto fonts from the fonts directory.
    Returns dict with font paths for different weights.
    """
    fonts = {
        'bold': FONTS_DIR / 'Roboto-Bold.ttf',
        'regular': FONTS_DIR / 'Roboto-Regular.ttf',
        'light': FONTS_DIR / 'Roboto-Light.ttf'
    }
    
    # Verify fonts exist
    missing_fonts = []
    for weight, path in fonts.items():
        if not path.exists():
            missing_fonts.append(f"{weight}: {path}")
    
    if missing_fonts:
        st.warning(f"⚠️ Fonts not found:\n" + "\n".join(missing_fonts))
        st.info("The app will use fallback system fonts. To use Roboto fonts, add the .ttf files to the 'fonts' directory.")
        return None
    
    # Convert paths to strings for FontProperties
    return {k: str(v) for k, v in fonts.items()}

@st.cache_data
def get_available_themes():
    """
    Scans the themes directory and returns a list of available theme names.
    """
    if not THEMES_DIR.exists():
        st.error(f"Themes directory not found: {THEMES_DIR}")
        return []
    
    themes = []
    for file in sorted(THEMES_DIR.glob("*.json")):
        theme_name = file.stem  # filename without extension
        themes.append(theme_name)
    
    if not themes:
        st.warning("No theme files found in themes directory. Please add .json theme files.")
    
    return themes

@st.cache_data
def load_theme(theme_name="feature_based"):
    """
    Load theme from JSON file in themes directory.
    """
    theme_file = THEMES_DIR / f"{theme_name}.json"
    
    if not theme_file.exists():
        st.warning(f"⚠️ Theme file '{theme_file}' not found. Using default theme.")
        # Fallback to embedded default theme
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
            st.success(f"✓ Loaded theme: {theme.get('name', theme_name)}")
            if 'description' in theme:
                st.info(f"📝 {theme['description']}")
            return theme
    except Exception as e:
        st.error(f"Error loading theme: {e}")
        return None

def generate_output_filename(city, theme_name):
    """
    Generate unique output filename with city, theme, and datetime.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    city_slug = city.lower().replace(' ', '_')
    filename = f"{city_slug}_{theme_name}_{timestamp}.png"
    return POSTERS_DIR / filename

def create_gradient_fade(ax, color, location='bottom', zorder=10):
    """
    Creates a fade effect at the top or bottom of the map.
    """
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
    """
    Assigns colors to edges based on road type hierarchy.
    """
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
    """
    Assigns line widths to edges based on road type.
    """
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

def get_coordinates(city, country):
    """
    Fetches coordinates for a given city and country using geopy.
    """
    with st.spinner("🌍 Looking up coordinates..."):
        # geolocator = Nominatim(user_agent="city_map_poster_streamlit")
        # time.sleep(1)
        
        #location = geolocator.geocode(f"{city}, {country}")
        return 52.3676, 4.9041  # Amsterdam coordinates for testing
        # if location:
        #     st.success(f"✓ Found: {location.address}")
        #     st.info(f"📍 Coordinates: {location.latitude:.4f}, {location.longitude:.4f}")
        #     return (location.latitude, location.longitude)
        # else:
        #     raise ValueError(f"Could not find coordinates for {city}, {country}")

def create_poster(city, country, point, dist, theme, fonts):
    """
    Generate the map poster.
    """
    st.write(f"\n🗺️ Generating map for {city}, {country}...")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 1. Fetch Street Network
    status_text.text("📡 Downloading street network...")
    progress_bar.progress(10)
    G = ox.graph_from_point(point, dist=dist, dist_type='bbox', network_type='all')
    
    progress_bar.progress(40)
    time.sleep(0.5)
    
    # 2. Fetch Water Features
    status_text.text("💧 Downloading water features...")
    try:
        water = ox.features_from_point(point, tags={'natural': 'water', 'waterway': 'riverbank'}, dist=dist)
    except:
        water = None
    progress_bar.progress(60)
    time.sleep(0.3)
    
    # 3. Fetch Parks
    # status_text.text("🌳 Downloading parks/green spaces...")
    # try:
    #     parks = ox.features_from_point(point, tags={'leisure': 'park', 'landuse': 'grass'}, dist=dist)
    # except:
    #     parks = None
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
    
    spaced_city = "  ".join(list(city.upper()))
    
    # Bottom text
    ax.text(0.5, 0.14, spaced_city, transform=ax.transAxes,
            color=theme['text'], ha='center', fontproperties=font_main, zorder=11)
    
    ax.text(0.5, 0.10, country.upper(), transform=ax.transAxes,
            color=theme['text'], ha='center', fontproperties=font_sub, zorder=11)
    
    lat, lon = point
    coords = f"{lat:.4f}° N / {lon:.4f}° E" if lat >= 0 else f"{abs(lat):.4f}° S / {lon:.4f}° E"
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

# Streamlit App
def main():
    st.set_page_config(page_title="City Map Poster Generator", page_icon="🗺️", layout="wide")
    
    st.title("🗺️ City Map Poster Generator")
    st.markdown("Generate beautiful minimalist map posters for any city in the world")
    
    # Load resources
    fonts = load_fonts()
    available_themes = get_available_themes()
    
    if not available_themes:
        st.error("⚠️ No themes found! Please add theme JSON files to the 'themes' directory.")
        st.stop()
    
    # Sidebar inputs
    with st.sidebar:
        st.header("⚙️ Settings")
        
        city = st.text_input("City", "Amsterdam", help="Enter the city name")
        country = st.text_input("Country", "The Netherlands", help="Enter the country name")
        
        theme_name = st.selectbox(
            "Theme", 
            available_themes,
            help="Select a visual theme for your poster"
        )
        
        distance = st.number_input(
            "Distance (meters)", 
            min_value=1000, 
            max_value=50000, 
            value=10000, 
            step=1000,
            help="Map radius from city center"
        )
        
        st.markdown("---")
        st.markdown("**Distance Guide:**")
        st.markdown("- 4,000-6,000m: Small cities")
        st.markdown("- 8,000-12,000m: Medium cities")
        st.markdown("- 15,000-20,000m: Large metros")
        
        generate_btn = st.button("🎨 Generate Poster", type="primary", use_container_width=True)
    
    # Main content area
    if generate_btn:
        if not city or not country:
            st.error("Please enter both city and country names")
            st.stop()
        
        try:
            # Load theme
            theme = load_theme(theme_name)
            if theme is None:
                st.stop()
            
            # Get coordinates
            coords = get_coordinates(city, country)
            
            # Generate poster
            fig = create_poster(city, country, coords, distance, theme, fonts)
            
            # Display
            st.pyplot(fig)
            
            # Save option
            output_file = generate_output_filename(city, theme_name)
            fig.savefig(output_file, dpi=300, facecolor=theme['bg'], bbox_inches='tight')
            plt.close(fig)
            
            st.success(f"✅ Poster saved to: {output_file}")
            
            # Download button
            with open(output_file, "rb") as file:
                st.download_button(
                    label="⬇️ Download Poster",
                    data=file,
                    file_name=output_file.name,
                    mime="image/png"
                )
            
            st.info("Based on Map to Poster by Ankur Gupta. MIT License.")
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())
    
    else:
        st.info("👈 Configure your settings in the sidebar and click 'Generate Poster'")
        
        # Show example
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📋 Instructions")
            st.markdown("""
            1. Enter a city and country name
            2. Choose a theme
            3. Set the map radius (distance)
            4. Click 'Generate Poster'
            5. Download your custom map!
            """)
        
        with col2:
            st.subheader("🎨 Theme Requirements")
            st.markdown(f"""
            Themes directory: `{THEMES_DIR}`
            
            Found **{len(available_themes)}** themes
            
            Each theme should be a JSON file with colors for:
            - Background, text, water, parks
            - Road hierarchy (motorway, primary, etc.)
            """)

if __name__ == "__main__":
    main()