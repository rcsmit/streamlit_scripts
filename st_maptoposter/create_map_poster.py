#!/usr/bin/env python3
"""
City Map Poster Generator

This module generates beautiful, minimalist map posters for any city in the world.
It fetches OpenStreetMap data using OSMnx, applies customizable themes, and creates
high-quality poster-ready images with roads, water features, and parks.

CHANGES:
added get_available_themes_with_subdirs()

THEMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "themes")
FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
POSTERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "posters")

main() -> generate_poster()

if __name__ == "__main__":
    main()

the "free code" is now main()
"""

import argparse
import asyncio
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from geopandas import GeoDataFrame
from geopy.geocoders import Nominatim
from lat_lon_parser import parse
from matplotlib.font_manager import FontProperties
from networkx import MultiDiGraph
from shapely.geometry import Point
from tqdm import tqdm

from font_management import load_fonts


class CacheError(Exception):
    """Raised when a cache operation fails."""


CACHE_DIR_PATH = os.environ.get("CACHE_DIR", "cache")
CACHE_DIR = Path(CACHE_DIR_PATH)
CACHE_DIR.mkdir(exist_ok=True)

# THEMES_DIR =  os.path.join("themes")
# FONTS_DIR = os.path.join("fonts")
# POSTERS_DIR = os.path.join("posters")

FILE_ENCODING = "utf-8"

FONTS = load_fonts()

# After
THEMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "themes")
FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
POSTERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "posters")

def _cache_path(key: str) -> str:
    safe = key.replace(os.sep, "_")
    return os.path.join(CACHE_DIR, f"{safe}.pkl")


def cache_get(key: str):
    try:
        path = _cache_path(key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CacheError(f"Cache read failed: {e}") from e


def cache_set(key: str, value):
    try:
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        path = _cache_path(key)
        with open(path, "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise CacheError(f"Cache write failed: {e}") from e


def is_latin_script(text):
    if not text:
        return True
    latin_count = 0
    total_alpha = 0
    for char in text:
        if char.isalpha():
            total_alpha += 1
            if ord(char) < 0x250:
                latin_count += 1
    if total_alpha == 0:
        return True
    return (latin_count / total_alpha) > 0.8


def generate_output_filename(city, theme_name, output_format):
    if not os.path.exists(POSTERS_DIR):
        os.makedirs(POSTERS_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    city_slug = city.lower().replace(" ", "_")
    ext = output_format.lower()
    filename = f"{city_slug}_{theme_name}_{timestamp}.{ext}"
    return os.path.join(POSTERS_DIR, filename)

def get_available_themes_with_subdirs():
    """Scans the themes directory and returns a list of available theme names from folders."""
    
    if not THEMES_DIR.exists():
        print("THEMES_DIR doesnt exist")
        return []
    
    themes = []
    for file in sorted(THEMES_DIR.iterdir()):
        if file.is_file() and file.suffix.lower() == '.json':
            theme_name = file.stem
            themes.append(f"{theme_name}")
    
    for folder in sorted(THEMES_DIR.iterdir()):
        if folder.is_dir():
            for file in folder.iterdir(): 
                theme_file = folder / f"{file.stem}.json"
                if theme_file.exists():
                    themes.append(f"{folder.name}/{file.stem}")
    return themes

def get_available_themes():
    print (THEMES_DIR)
    if not os.path.exists(THEMES_DIR):
        os.makedirs(THEMES_DIR)
        return []
    themes = []
    for file in sorted(os.listdir(THEMES_DIR)):
        if file.endswith(".json"):
            theme_name = file[:-5]
            themes.append(theme_name)
    return themes


def load_theme(theme_name="terracotta"):
    theme_file = os.path.join(THEMES_DIR, f"{theme_name}.json")
    if not os.path.exists(theme_file):
        print(f"⚠ Theme file '{theme_file}' not found. Using default terracotta theme.")
        return {
            "name": "Terracotta",
            "description": "Mediterranean warmth - burnt orange and clay tones on cream",
            "bg": "#F5EDE4",
            "text": "#8B4513",
            "gradient_color": "#F5EDE4",
            "water": "#A8C4C4",
            "parks": "#E8E0D0",
            "road_motorway": "#A0522D",
            "road_primary": "#B8653A",
            "road_secondary": "#C9846A",
            "road_tertiary": "#D9A08A",
            "road_residential": "#E5C4B0",
            "road_default": "#D9A08A",
        }
    with open(theme_file, "r", encoding=FILE_ENCODING) as f:
        theme = json.load(f)
        print(f"✓ Loaded theme: {theme.get('name', theme_name)}")
        if "description" in theme:
            print(f"  {theme['description']}")
        return theme


THEME = dict[str, str]()


def create_gradient_fade(ax, color, location="bottom", zorder=10):
    vals = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack((vals, vals))
    rgb = mcolors.to_rgb(color)
    my_colors = np.zeros((256, 4))
    my_colors[:, 0] = rgb[0]
    my_colors[:, 1] = rgb[1]
    my_colors[:, 2] = rgb[2]
    if location == "bottom":
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
    ax.imshow(
        gradient,
        extent=[xlim[0], xlim[1], y_bottom, y_top],
        aspect="auto",
        cmap=custom_cmap,
        zorder=zorder,
        origin="lower",
    )


def get_edge_colors_by_type(g):
    edge_colors = []
    for _u, _v, data in g.edges(data=True):
        highway = data.get('highway', 'unclassified')
        if isinstance(highway, list):
            highway = highway[0] if highway else 'unclassified'
        if highway in ["motorway", "motorway_link"]:
            color = THEME["road_motorway"]
        elif highway in ["trunk", "trunk_link", "primary", "primary_link"]:
            color = THEME["road_primary"]
        elif highway in ["secondary", "secondary_link"]:
            color = THEME["road_secondary"]
        elif highway in ["tertiary", "tertiary_link"]:
            color = THEME["road_tertiary"]
        elif highway in ["residential", "living_street", "unclassified"]:
            color = THEME["road_residential"]
        else:
            color = THEME['road_default']
        edge_colors.append(color)
    return edge_colors


def get_edge_widths_by_type(g):
    edge_widths = []
    for _u, _v, data in g.edges(data=True):
        highway = data.get('highway', 'unclassified')
        if isinstance(highway, list):
            highway = highway[0] if highway else 'unclassified'
        if highway in ["motorway", "motorway_link"]:
            width = 1.2
        elif highway in ["trunk", "trunk_link", "primary", "primary_link"]:
            width = 1.0
        elif highway in ["secondary", "secondary_link"]:
            width = 0.8
        elif highway in ["tertiary", "tertiary_link"]:
            width = 0.6
        else:
            width = 0.4
        edge_widths.append(width)
    return edge_widths


def get_coordinates(city, country):
    coords = f"coords_{city.lower()}_{country.lower()}"
    cached = cache_get(coords)
    if cached:
        print(f"✓ Using cached coordinates for {city}, {country}")
        return cached
    print("Looking up coordinates...")
    geolocator = Nominatim(user_agent="city_map_poster", timeout=10)
    time.sleep(1)
    try:
        location = geolocator.geocode(f"{city}, {country}")
    except Exception as e:
        raise ValueError(f"Geocoding failed for {city}, {country}: {e}") from e
    if asyncio.iscoroutine(location):
        try:
            location = asyncio.run(location)
        except RuntimeError as exc:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError(
                    "Geocoder returned a coroutine while an event loop is already running."
                ) from exc
            location = loop.run_until_complete(location)
    if location:
        addr = getattr(location, "address", None)
        if addr:
            print(f"✓ Found: {addr}")
        else:
            print("✓ Found location (address not available)")
        print(f"✓ Coordinates: {location.latitude}, {location.longitude}")
        try:
            cache_set(coords, (location.latitude, location.longitude))
        except CacheError as e:
            print(e)
        return (location.latitude, location.longitude)
    raise ValueError(f"Could not find coordinates for {city}, {country}")


def get_crop_limits(g_proj, center_lat_lon, fig, dist):
    lat, lon = center_lat_lon
    center = (
        ox.projection.project_geometry(
            Point(lon, lat),
            crs="EPSG:4326",
            to_crs=g_proj.graph["crs"]
        )[0]
    )
    center_x, center_y = center.x, center.y
    fig_width, fig_height = fig.get_size_inches()
    aspect = fig_width / fig_height
    half_x = dist
    half_y = dist
    if aspect > 1:
        half_y = half_x / aspect
    else:
        half_x = half_y * aspect
    return (
        (center_x - half_x, center_x + half_x),
        (center_y - half_y, center_y + half_y),
    )


def fetch_graph(point, dist) -> MultiDiGraph | None:
    lat, lon = point
    graph = f"graph_{lat}_{lon}_{dist}"
    cached = cache_get(graph)
    if cached is not None:
        print("✓ Using cached street network")
        return cast(MultiDiGraph, cached)
    try:
        g = ox.graph_from_point(point, dist=dist, dist_type='bbox', network_type='all', truncate_by_edge=True)
        time.sleep(0.5)
        try:
            cache_set(graph, g)
        except CacheError as e:
            print(e)
        return g
    except Exception as e:
        print(f"OSMnx error while fetching graph: {e}")
        return None


def fetch_features(point, dist, tags, name) -> GeoDataFrame | None:
    lat, lon = point
    tag_str = "_".join(tags.keys())
    features = f"{name}_{lat}_{lon}_{dist}_{tag_str}"
    cached = cache_get(features)
    if cached is not None:
        print(f"✓ Using cached {name}")
        return cast(GeoDataFrame, cached)
    try:
        data = ox.features_from_point(point, tags=tags, dist=dist)
        time.sleep(0.3)
        try:
            cache_set(features, data)
        except CacheError as e:
            print(e)
        return data
    except Exception as e:
        print(f"OSMnx error while fetching features: {e}")
        return None


def create_poster(
    city,
    country,
    point,
    dist,
    output_file,
    output_format,
    width=12,
    height=16,
    country_label=None,
    name_label=None,
    display_city=None,
    display_country=None,
    fonts=None,
):
    """
    Generate a complete map poster with roads, water, parks, and typography.
    Returns the matplotlib Figure object.
    """
    global THEME

    display_city = display_city or name_label or city
    display_country = display_country or country_label or country

    print(f"\nGenerating map for {city}, {country}...")

    with tqdm(
        total=3,
        desc="Fetching map data",
        unit="step",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
    ) as pbar:
        pbar.set_description("Downloading street network")
        compensated_dist = dist * (max(height, width) / min(height, width)) / 4
        g = fetch_graph(point, compensated_dist)
        if g is None:
            raise RuntimeError("Failed to retrieve street network data.")
        pbar.update(1)

        pbar.set_description("Downloading water features")
        water = fetch_features(
            point,
            compensated_dist,
            tags={"natural": ["water", "bay", "strait"], "waterway": "riverbank"},
            name="water",
        )
        pbar.update(1)

        pbar.set_description("Downloading parks/green spaces")
        parks = fetch_features(
            point,
            compensated_dist,
            tags={"leisure": "park", "landuse": "grass"},
            name="parks",
        )
        pbar.update(1)

    print("✓ All data retrieved successfully!")
    print("Rendering map...")

    fig, ax = plt.subplots(figsize=(width, height), facecolor=THEME["bg"])
    ax.set_facecolor(THEME["bg"])
    ax.set_position((0.0, 0.0, 1.0, 1.0))

    g_proj = ox.project_graph(g)

    if water is not None and not water.empty:
        water_polys = water[water.geometry.type.isin(["Polygon", "MultiPolygon"])]
        if not water_polys.empty:
            try:
                water_polys = ox.projection.project_gdf(water_polys)
            except Exception:
                water_polys = water_polys.to_crs(g_proj.graph['crs'])
            water_polys.plot(ax=ax, facecolor=THEME['water'], edgecolor='none', zorder=0.5)

    if parks is not None and not parks.empty:
        parks_polys = parks[parks.geometry.type.isin(["Polygon", "MultiPolygon"])]
        if not parks_polys.empty:
            try:
                parks_polys = ox.projection.project_gdf(parks_polys)
            except Exception:
                parks_polys = parks_polys.to_crs(g_proj.graph['crs'])
            parks_polys.plot(ax=ax, facecolor=THEME['parks'], edgecolor='none', zorder=0.8)

    print("Applying road hierarchy colors...")
    edge_colors = get_edge_colors_by_type(g_proj)
    edge_widths = get_edge_widths_by_type(g_proj)

    crop_xlim, crop_ylim = get_crop_limits(g_proj, point, fig, compensated_dist)
    ox.plot_graph(
        g_proj, ax=ax, bgcolor=THEME['bg'],
        node_size=0,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        show=False,
        close=False,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(crop_xlim)
    ax.set_ylim(crop_ylim)

    create_gradient_fade(ax, THEME['gradient_color'], location='bottom', zorder=10)
    create_gradient_fade(ax, THEME['gradient_color'], location='top', zorder=10)

    scale_factor = min(height, width) / 12.0
    base_main = 60
    base_sub = 22
    base_coords = 14
    base_attr = 8

    active_fonts = fonts or FONTS
    if active_fonts:
        font_sub = FontProperties(fname=active_fonts["light"], size=base_sub * scale_factor)
        font_coords = FontProperties(fname=active_fonts["regular"], size=base_coords * scale_factor)
        font_attr = FontProperties(fname=active_fonts["light"], size=base_attr * scale_factor)
    else:
        font_sub = FontProperties(family="monospace", weight="normal", size=base_sub * scale_factor)
        font_coords = FontProperties(family="monospace", size=base_coords * scale_factor)
        font_attr = FontProperties(family="monospace", size=base_attr * scale_factor)

    if is_latin_script(display_city):
        spaced_city = "  ".join(list(display_city.upper()))
    else:
        spaced_city = display_city

    base_adjusted_main = base_main * scale_factor
    city_char_count = len(display_city)
    if city_char_count > 10:
        length_factor = 10 / city_char_count
        adjusted_font_size = max(base_adjusted_main * length_factor, 10 * scale_factor)
    else:
        adjusted_font_size = base_adjusted_main

    if active_fonts:
        font_main_adjusted = FontProperties(fname=active_fonts["bold"], size=adjusted_font_size)
    else:
        font_main_adjusted = FontProperties(family="monospace", weight="bold", size=adjusted_font_size)

    ax.text(0.5, 0.14, spaced_city, transform=ax.transAxes, color=THEME["text"],
            ha="center", fontproperties=font_main_adjusted, zorder=11)
    ax.text(0.5, 0.10, display_country.upper(), transform=ax.transAxes, color=THEME["text"],
            ha="center", fontproperties=font_sub, zorder=11)

    lat, lon = point
    coords = (
        f"{lat:.4f}° N / {lon:.4f}° E"
        if lat >= 0
        else f"{abs(lat):.4f}° S / {lon:.4f}° E"
    )
    if lon < 0:
        coords = coords.replace("E", "W")

    ax.text(0.5, 0.07, coords, transform=ax.transAxes, color=THEME["text"],
            alpha=0.7, ha="center", fontproperties=font_coords, zorder=11)
    ax.plot([0.4, 0.6], [0.125, 0.125], transform=ax.transAxes,
            color=THEME["text"], linewidth=1 * scale_factor, zorder=11)

    if FONTS:
        font_attr = FontProperties(fname=FONTS["light"], size=8)
    else:
        font_attr = FontProperties(family="monospace", size=8)

    ax.text(0.98, 0.02, "© OpenStreetMap contributors", transform=ax.transAxes,
            color=THEME["text"], alpha=0.5, ha="right", va="bottom",
            fontproperties=font_attr, zorder=11)

    print(f"Saving to {output_file}...")
    fmt = output_format.lower()
    save_kwargs = dict(facecolor=THEME["bg"], bbox_inches="tight", pad_inches=0.05)
    if fmt == "png":
        save_kwargs["dpi"] = 300
    plt.savefig(output_file, format=fmt, **save_kwargs)
    print(f"✓ Done! Poster saved as {output_file}")

    return fig


def generate_poster(
    city: str,
    country: str,
    theme: str = "terracotta",
    distance: int = 18000,
    width: float = 12,
    height: float = 16,
    output_format: str = "png",
    latitude: str | None = None,
    longitude: str | None = None,
    country_label: str | None = None,
    display_city: str | None = None,
    display_country: str | None = None,
    font_family: str | None = None,
    all_themes: bool = False,
) -> tuple[plt.Figure, str]:
    """
    Generate a city map poster programmatically.

    Args:
        city: City name (required)
        country: Country name (required)
        theme: Theme name (default: "terracotta")
        distance: Map radius in meters (default: 18000)
        width: Image width in inches (default: 12, max: 20)
        height: Image height in inches (default: 16, max: 20)
        output_format: File format — 'png', 'svg', or 'pdf' (default: 'png')
        latitude: Override latitude center point (optional)
        longitude: Override longitude center point (optional)
        country_label: Override country text displayed on poster (optional)
        display_city: Custom display name for city (optional)
        display_country: Custom display name for country (optional)
        font_family: Google Fonts family name (optional)
        all_themes: If True, generate posters for all available themes (default: False)

    Returns:
        Tuple of (matplotlib Figure, output file path).
        When all_themes=True, returns the last generated (Figure, path).

    Raises:
        ValueError: If theme not found or coordinates cannot be resolved.
        RuntimeError: If street network data cannot be retrieved.
    """
    global THEME

    # Enforce maximum dimensions
    width = min(width, 20.0)
    height = min(height, 20.0)

    available_themes = get_available_themes_with_subdirs()
    if not available_themes:
        raise ValueError("No themes found in 'themes/' directory.")

    if all_themes:
        themes_to_generate = available_themes
    else:
        if theme not in available_themes:
            raise ValueError(
                f"Theme '{theme}' not found. Available themes: {', '.join(available_themes)}"
            )
        themes_to_generate = [theme]

    # Load custom fonts if specified
    custom_fonts = None
    if font_family:
        custom_fonts = load_fonts(font_family)
        if not custom_fonts:
            print(f"⚠ Failed to load '{font_family}', falling back to Roboto")

    # Resolve coordinates
    if latitude and longitude:
        coords = (parse(latitude), parse(longitude))
        print(f"✓ Coordinates: {coords[0]}, {coords[1]}")
    else:
        coords = get_coordinates(city, country)

    fig = None
    output_file = None

    for theme_name in themes_to_generate:
        THEME = load_theme(theme_name)
        output_file = generate_output_filename(city, theme_name, output_format)
        fig = create_poster(
            city,
            country,
            coords,
            distance,
            output_file,
            output_format,
            width,
            height,
            country_label=country_label,
            display_city=display_city,
            display_country=display_country,
            fonts=custom_fonts,
        )

    print("\n" + "=" * 50)
    print("✓ Poster generation complete!")
    print("=" * 50)

    return fig, output_file


def list_themes():
    """List all available themes with descriptions."""
    available_themes = get_available_themes_with_subdirs()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        return
    print("\nAvailable Themes:")
    print("-" * 60)
    for theme_name in available_themes:
        theme_path = os.path.join(THEMES_DIR, f"{theme_name}.json")
        try:
            with open(theme_path, "r", encoding=FILE_ENCODING) as f:
                theme_data = json.load(f)
                display_name = theme_data.get('name', theme_name)
                description = theme_data.get('description', '')
        except (OSError, json.JSONDecodeError):
            display_name = theme_name
            description = ""
        print(f"  {theme_name}")
        print(f"    {display_name}")
        if description:
            print(f"    {description}")
        print()


def print_examples():
    """Print usage examples."""
    print("""
City Map Poster Generator
=========================

Usage:
  python create_map_poster.py --city <city> --country <country> [options]

Examples:
  python create_map_poster.py -c "New York" -C "USA" -t noir -d 12000
  python create_map_poster.py -c "Paris" -C "France" -t pastel_dream -d 10000
  python create_map_poster.py --list-themes

Options:
  --city, -c        City name (required)
  --country, -C     Country name (required)
  --theme, -t       Theme name (default: terracotta)
  --distance, -d    Map radius in meters (default: 18000)
  --list-themes     List all available themes
""")

def main():
    parser = argparse.ArgumentParser(
        description="Generate beautiful map posters for any city",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--city", "-c", type=str, help="City name")
    parser.add_argument("--country", "-C", type=str, help="Country name")
    parser.add_argument("--latitude", "-lat", dest="latitude", type=str)
    parser.add_argument("--longitude", "-long", dest="longitude", type=str)
    parser.add_argument("--country-label", dest="country_label", type=str)
    parser.add_argument("--theme", "-t", type=str, default="terracotta")
    parser.add_argument("--all-themes", dest="all_themes", action="store_true")
    parser.add_argument("--distance", "-d", type=int, default=18000)
    parser.add_argument("--width", "-W", type=float, default=12)
    parser.add_argument("--height", "-H", type=float, default=16)
    parser.add_argument("--list-themes", action="store_true")
    parser.add_argument("--display-city", "-dc", type=str)
    parser.add_argument("--display-country", "-dC", type=str)
    parser.add_argument("--font-family", type=str)
    parser.add_argument("--format", "-f", default="png", choices=["png", "svg", "pdf"])
    args = parser.parse_args()

    if len(sys.argv) == 1:
        print_examples()
        sys.exit(0)

    if args.list_themes:
        list_themes()
        sys.exit(0)

    if not args.city or not args.country:
        print("Error: --city and --country are required.\n")
        print_examples()
        sys.exit(1)

    try:
        generate_poster(
            city=args.city,
            country=args.country,
            theme=args.theme,
            distance=args.distance,
            width=args.width,
            height=args.height,
            output_format=args.format,
            latitude=args.latitude,
            longitude=args.longitude,
            country_label=args.country_label,
            display_city=args.display_city,
            display_country=args.display_country,
            font_family=args.font_family,
            all_themes=args.all_themes,
        )
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()