"""
Netherlands Land Cover Map
Replicates the 3D perspective land cover visualization
Data: Sentinel-2 10m Land Use/Land Cover – Esri, Impact Observatory, and Microsoft
Original visualization style by Milos Popovic
"""

import io
import math

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
from PIL import Image

try:
    st.set_page_config(
        page_title="Netherlands Land Cover",
        layout="wide",
        page_icon="🌍",
    )
except Exception:
    pass

# ── Land cover legend (Esri 10m Annual Land Use/Land Cover colour scheme) ──
LAND_COVER_LEGEND = [
    ("Water", "#419BDF"),
    ("Trees", "#397D49"),
    ("Flooded Vegetation", "#7A87C6"),
    ("Crops", "#E49635"),
    ("Built Area", "#C4281B"),
    ("Bare Ground", "#A59B8F"),
    ("Snow / Ice", "#A8EBFF"),
    ("Rangeland", "#E3D4AE"),
]

# Netherlands bounding box (WGS-84)
NL_LAT_MIN, NL_LAT_MAX = 50.75, 53.55
NL_LON_MIN, NL_LON_MAX = 3.36, 7.22


# ── Tile helpers ───────────────────────────────────────────────────────────
def _latlon_to_tile(lat: float, lon: float, z: int) -> tuple[int, int]:
    n = 2**z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_tile(z: int, x: int, y: int) -> Image.Image | None:
    """Download one 256×256 PNG tile from the Esri Annual Land Cover service."""
    url = (
        "https://tiles.arcgis.com/tiles/P3ePLMYs2RVChkJx/arcgis/rest/services"
        f"/EsriAnthromeAnnualLandCover/MapServer/tile/{z}/{y}/{x}"
    )
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            return Image.open(io.BytesIO(r.content)).convert("RGBA")
    except Exception:
        pass
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def assemble_map(zoom: int) -> Image.Image | None:
    """Fetch and stitch all tiles covering the Netherlands at *zoom*."""
    x_min, y_max = _latlon_to_tile(NL_LAT_MIN, NL_LON_MIN, zoom)
    x_max, y_min = _latlon_to_tile(NL_LAT_MAX, NL_LON_MAX, zoom)

    tile_size = 256
    ncols = x_max - x_min + 1
    nrows = y_max - y_min + 1
    total = ncols * nrows

    canvas = Image.new("RGBA", (ncols * tile_size, nrows * tile_size), (0, 0, 0, 0))
    bar = st.progress(0, text="Downloading land cover tiles…")

    for ci, tx in enumerate(range(x_min, x_max + 1)):
        for ri, ty in enumerate(range(y_min, y_max + 1)):
            tile = _fetch_tile(zoom, tx, ty)
            if tile:
                canvas.paste(tile, (ci * tile_size, ri * tile_size))
            done = ci * nrows + ri + 1
            bar.progress(done / total, text=f"Tile {done}/{total}")

    bar.empty()
    return canvas


# ── Perspective / 3-D transform ────────────────────────────────────────────
def apply_perspective(img: Image.Image, tilt: float = 0.45) -> Image.Image:
    """
    Simulate a bird's-eye perspective by mapping the image through a
    quadrilateral → rectangle transform (PIL QUAD).

    *tilt* ∈ [0, 1]: 0 = flat, higher values = more extreme angle.
    """
    w, h = img.size
    squeeze = int(tilt * w * 0.30)          # how much the top edge narrows
    new_h = int(h * (1.0 - tilt * 0.25))   # compressed height

    # PIL QUAD data: source quad corners in order
    # upper-left, lower-left, lower-right, upper-right  (source coords)
    # These map to the output rectangle corners.
    quad_data = (
        squeeze,     0,       # upper-left  → top compressed inward
        0,           h,       # lower-left  → stays at original edge
        w,           h,       # lower-right → stays at original edge
        w - squeeze, 0,       # upper-right → top compressed inward
    )
    return img.transform((w, new_h), Image.QUAD, quad_data, Image.BICUBIC)


# ── Final figure ───────────────────────────────────────────────────────────
def build_figure(
    map_img: Image.Image,
    tilt: float,
    show_3d: bool,
    bg_hex: str,
) -> plt.Figure:
    """Compose the poster-style figure: dark background + map + legend."""
    if show_3d:
        map_img = apply_perspective(map_img, tilt)

    arr = np.array(map_img)
    map_h, map_w = arr.shape[:2]

    # Figure size: add space above (title) and below (legend / caption)
    padding_top_px = 160
    padding_bot_px = 20
    dpi = 120
    fig_w = map_w / dpi
    fig_h = (map_h + padding_top_px + padding_bot_px) / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=bg_hex)
    ax = fig.add_axes([0, padding_bot_px / (map_h + padding_top_px + padding_bot_px),
                       1, map_h / (map_h + padding_top_px + padding_bot_px)])
    ax.set_facecolor(bg_hex)
    ax.imshow(arr)
    ax.axis("off")

    # ── Title block (top-left, in figure coordinates) ──────────────────
    fig.text(0.03, 0.97, "NETHERLANDS", color="white",
             fontsize=26, fontweight="bold", va="top", ha="left",
             fontfamily="DejaVu Sans")
    fig.text(0.03, 0.93, "LAND COVER", color="white",
             fontsize=20, fontweight="bold", va="top", ha="left",
             fontfamily="DejaVu Sans")
    fig.text(0.03, 0.905,
             "Sentinel-2 10 m Land Use / Land Cover  –  Esri · Impact Observatory · Microsoft",
             color="#aaaaaa", fontsize=7, va="top", ha="left")

    # ── Legend (bottom-right corner) ───────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor=color, edgecolor="none", label=name)
        for name, color in LAND_COVER_LEGEND
    ]
    legend = ax.legend(
        handles=legend_patches,
        loc="lower right",
        frameon=True,
        framealpha=0.25,
        facecolor="#111111",
        edgecolor="none",
        labelcolor="white",
        fontsize=8,
        title="Land Cover",
        title_fontsize=9,
    )
    legend.get_title().set_color("white")

    return fig


# ── Streamlit UI ───────────────────────────────────────────────────────────
st.title("Netherlands — Land Cover")
st.caption(
    "Sentinel-2 10 m Land Use / Land Cover  ·  Esri · Impact Observatory · Microsoft"
)

with st.sidebar:
    st.header("⚙️ Settings")
    zoom_level = st.slider(
        "Zoom level", min_value=6, max_value=9, value=8,
        help="Higher zoom = more detail, more tiles to download."
    )
    show_3d = st.checkbox("3 D perspective", value=True)
    tilt_val = 0.0
    if show_3d:
        tilt_val = st.slider(
            "Tilt amount", min_value=0.0, max_value=0.75, value=0.45, step=0.05
        )
    bg_color = st.color_picker("Background colour", "#0F0F19")

    st.divider()
    st.markdown("**Land cover classes**")
    for name, color in LAND_COVER_LEGEND:
        st.markdown(
            f'<span style="display:inline-block;width:14px;height:14px;'
            f'background:{color};border-radius:50%;vertical-align:middle;'
            f'margin-right:8px;"></span>{name}',
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption(
        "Tile source: Esri ArcGIS Living Atlas\n\n"
        "Visualization inspired by Milos Popovic"
    )

# ── Load & render ──────────────────────────────────────────────────────────
with st.spinner("Assembling map…"):
    nl_map = assemble_map(zoom_level)

if nl_map is None:
    st.error("Could not load any tiles. Check your internet connection and try again.")
    st.stop()

fig = build_figure(nl_map, tilt_val, show_3d, bg_color)
st.pyplot(fig, use_container_width=True)

st.caption(
    "Tip: Use the sidebar to adjust zoom, perspective tilt, and background colour."
)
