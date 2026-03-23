import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import geopandas as gpd
from scipy import stats

st.set_page_config(
    page_title="Hoe gewoon is jouw verjaardag?",
    page_icon="🎂",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=Source+Sans+3:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
    h1, h2, h3 { font-family: 'Source Serif 4', serif; }
    .stApp { background-color: #f5f5f0; }
    .main .block-container { padding-top: 2rem; max-width: 1200px; }
    .subtitle { color: #555; font-size: 1rem; margin-bottom: 1rem; }
    .source-note { color: #888; font-size: 0.8rem; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════
MONTH_NAMES   = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
CONCEPTION_DAYS = 266
NL_COL = "Nederland"

COLORSCALE_BLUE = [
    [0.0,"#0a1628"],[0.08,"#0d2244"],[0.15,"#1a3a6b"],[0.25,"#1e5085"],
    [0.40,"#2878b8"],[0.55,"#4a9fd4"],[0.70,"#80c0e8"],[0.85,"#b0d8f0"],[1.0,"#d8eef8"],
]
COLORSCALE_GREEN = [
    [0.0,"#071a0e"],[0.08,"#0c2e16"],[0.15,"#155227"],[0.25,"#1a6b32"],
    [0.40,"#2a8c45"],[0.55,"#4aad65"],[0.70,"#7fcc96"],[0.85,"#b0e4bf"],[1.0,"#d8f5e0"],
]
COLORSCALE_DIV = [
    [0.0,"#8b0000"],[0.2,"#d73027"],[0.4,"#fc8d59"],
    [0.47,"#fee090"],[0.5,"#ffffbf"],[0.53,"#e0f3a1"],
    [0.6,"#91cf60"],[0.8,"#1a9850"],[1.0,"#005a23"],
]

GEOJSON_URL = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/gemeente_2026.geojson"
# https://cartomap.github.io/nl/wgs84/gemeente_2026.geojson
# Name fixes: birth-CSV name → GeoJSON statnaam
GEMEENTE_FIX = {
    "Hengelo (O.)":                  "Hengelo",
    "Middelburg (Z.)":              "Middelburg",
    "Beek (L.)":                     "Beek",
    "Rijswijk (ZH.)":               "Rijswijk",
    "Stein (L.)":                    "Stein",
    "Laren (NH.)":                   "Laren",
}
#   "Nuenen, Gerwen en Nederwetten": "Nuenen",
  
# "'s-Gravenhage":                 "Den Haag",
#     "'s-Hertogenbosch":              "Den Bosch",
#    "Bergen (L.)":                   "Bergen (L)",
#     "Bergen (NH.)":                  "Bergen (NH)",

@st.cache_data(show_spinner=False)
def get_name_mismatches(gemeente_cols: list) -> tuple[list, list]:
    """
    Compare gemeente names in the birth CSV vs. GeoJSON (after applying GEMEENTE_FIX).
    Returns (in_csv_not_geo, in_geo_not_csv).
    """
    import requests

    try:
        r = requests.get(GEOJSON_URL, timeout=15)
        r.raise_for_status()
        geo_names = {f["properties"]["statnaam"] for f in r.json()["features"]}
    except Exception as e:
        st.warning(f"GeoJSON kon niet geladen worden: {e}")
        return [], []

    csv_fixed = {GEMEENTE_FIX.get(n, n) for n in gemeente_cols}

    in_csv_not_geo = sorted(csv_fixed - geo_names)
    in_geo_not_csv = sorted(geo_names - csv_fixed)
    return in_csv_not_geo, in_geo_not_csv



# ══════════════════════════════════════════════════════════════════
# Data loading  –  robust NaN/None handling
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def load_full() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/verjaardagen_2024.csv"
    for path in [
        "C:/Users/rcxsm/Documents/python_scripts/streamlit_scripts/input/verjaardagen_2024.csv",
        "/mnt/user-data/uploads/verjaardagen_2024.csv",
        url,
    ]:
        try:
            df = pd.read_csv(path, parse_dates=["date"], dayfirst=True)
            break
        except FileNotFoundError:
            continue
    else:
        st.error("CSV niet gevonden. Pas het pad aan in load_full().")
        st.stop()

    # ── strip whitespace from column names ──
    df.columns = df.columns.str.strip()

    # ── drop leap day ──
    df = df[~((df["date"].dt.month == 2) & (df["date"].dt.day == 29))].copy()

    # ── numeric columns: coerce, fill NaN with 0 ──
    num_cols = [c for c in df.columns if c != "date"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    df["month"] = df["date"].dt.month
    df["day"]   = df["date"].dt.day
    return df


df_full       = load_full()
gemeente_cols = [c for c in df_full.columns if c not in ("date", "month", "day", NL_COL)]

# ══════════════════════════════════════════════════════════════════
# Statistics: chi-square per gemeente
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def compute_deviations(df_full: pd.DataFrame, gemeente_cols: list, nl_col: str):
    nl_counts = df_full[nl_col].values.astype(float)
    nl_prop   = nl_counts / nl_counts.sum()

    rows = []
    for gem in gemeente_cols:
        obs   = df_full[gem].values.astype(float)
        total = obs.sum()
        if total < 100:
            continue
        expected = nl_prop * total
        mask = expected > 0
        chi2 = float(np.sum((obs[mask] - expected[mask]) ** 2 / expected[mask]))
        dof  = int(mask.sum()) - 1
        p    = float(stats.chi2.sf(chi2, dof))
        rows.append({
            "gemeente": gem, "total": int(total),
            "chi2": round(chi2, 1), "dof": dof,
            "p_value": p, "significant": p < 0.05,
        })

    return pd.DataFrame(rows).sort_values("p_value"), nl_prop


summary_df, nl_prop = compute_deviations(df_full, gemeente_cols, NL_COL)


def get_gemeente_zscores(gemeente: str) -> pd.DataFrame:
    obs   = df_full[gemeente].values.astype(float)
    total = obs.sum()
    exp   = nl_prop * total
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(exp > 0, (obs - exp) / np.sqrt(exp), 0.0)
    out = df_full[["date", "month", "day"]].copy()
    out["observed"] = obs.astype(int)
    out["expected"] = np.round(exp, 1)
    out["z_score"]  = np.round(z, 2)
    return out

# ══════════════════════════════════════════════════════════════════
# Matrix builders  –  safe against NaN / None everywhere
# ══════════════════════════════════════════════════════════════════
def _mask_invalid(matrix, text_matrix):
    """Set cells beyond each month's real day-count to NaN / ''."""
    for m_idx, max_day in enumerate(DAYS_IN_MONTH):
        for d_idx in range(max_day, 31):
            matrix[d_idx, m_idx]      = np.nan
            text_matrix[d_idx, m_idx] = ""
    return matrix, text_matrix


def _safe_int(v) -> int:
    """Convert value to int; return 0 for NaN / None / non-numeric."""
    try:
        f = float(v)
        if np.isnan(f):
            return 0
        return int(f)
    except (TypeError, ValueError):
        return 0


def build_rank_matrices(agg_df: pd.DataFrame, month_col: str, day_col: str, value_col: str):
    """Return (rank_matrix, text_matrix, ranked_df)."""
    agg = agg_df.copy()
    # Fill any NaN in the value column before ranking
    agg[value_col] = pd.to_numeric(agg[value_col], errors="coerce").fillna(0)
    agg["rank"] = agg[value_col].rank(ascending=False, method="first").astype(int)

    rm = np.full((31, 12), np.nan)
    tm = np.full((31, 12), "", dtype=object)

    for _, row in agg.iterrows():
        m = _safe_int(row[month_col]) - 1
        d = _safe_int(row[day_col])   - 1
        if 0 <= m < 12 and 0 <= d < 31:
            rm[d, m] = _safe_int(row["rank"])
            tm[d, m] = str(_safe_int(row["rank"]))

    rm, tm = _mask_invalid(rm, tm)
    return rm, tm, agg


def build_zscore_matrices(zdf: pd.DataFrame):
    """Return (z_matrix, text_matrix, hover_matrix)."""
    zm = np.full((31, 12), np.nan)
    tm = np.full((31, 12), "", dtype=object)
    hm = np.full((31, 12), "", dtype=object)

    for _, row in zdf.iterrows():
        m = _safe_int(row["month"]) - 1
        d = _safe_int(row["day"])   - 1
        if 0 <= m < 12 and 0 <= d < 31:
            z = row["z_score"]
            if pd.isna(z):
                z = 0.0
            zm[d, m] = float(z)
            tm[d, m] = f"{z:+.1f}"
            hm[d, m] = (
                f"obs={_safe_int(row['observed'])} "
                f"exp={float(row['expected']):.0f} "
                f"z={z:+.2f}"
            )

    zm, tm = _mask_invalid(zm, tm)
    return zm, tm, hm

# ══════════════════════════════════════════════════════════════════
# Figure factories
# ══════════════════════════════════════════════════════════════════
def _apply_layout(fig: go.Figure, height: int = 820):
    fig.update_layout(
        xaxis=dict(
            tickvals=list(range(12)), ticktext=MONTH_NAMES, side="top",
            tickfont=dict(size=12, family="Source Sans 3", color="#333"),
            showgrid=False, zeroline=False,
        ),
        yaxis=dict(
            tickvals=list(range(31)),
            ticktext=[str(d) for d in range(1, 32)],
            autorange="reversed",
            tickfont=dict(size=11, family="Source Sans 3", color="#333"),
            showgrid=False, zeroline=False,
        ),
        plot_bgcolor="#f5f5f0", paper_bgcolor="#f5f5f0",
        margin=dict(t=60, l=40, r=50, b=20),
        height=height,
    )


def make_rank_heatmap(rank_matrix, text_matrix, colorscale, highlight_row, highlight_color):
    fig = go.Figure(data=go.Heatmap(
        z=rank_matrix, text=text_matrix, texttemplate="%{text}",
        colorscale=colorscale, zmin=1, zmax=365,
        showscale=False, xgap=2, ygap=2,
        hovertemplate="<b>%{x} %{y}</b><br>Rang: %{z:.0f}<extra></extra>",
    ))
    try:
        r1_m = _safe_int(highlight_row["month"]) - 1
        r1_d = _safe_int(highlight_row["day"])   - 1
        fig.add_shape(type="rect",
            x0=r1_m-.5, x1=r1_m+.5, y0=r1_d-.5, y1=r1_d+.5,
            fillcolor=highlight_color, line=dict(color=highlight_color), layer="above")
        fig.add_annotation(x=r1_m, y=r1_d, text="<b>1</b>", showarrow=False,
            font=dict(color="white", size=11, family="Source Sans 3"))
    except Exception:
        pass  # no highlight if data missing
    fig.update_traces(textfont=dict(size=11, family="Source Sans 3", color="white"))
    _apply_layout(fig)
    return fig


def make_zscore_heatmap(z_matrix, text_matrix, hover_matrix):
    clipped = np.where(np.isnan(z_matrix), np.nan, np.clip(z_matrix, -4, 4))
    fig = go.Figure(data=go.Heatmap(
        z=clipped, text=text_matrix, texttemplate="%{text}",
        customdata=hover_matrix,
        colorscale=COLORSCALE_DIV, zmin=-4, zmax=4,
        showscale=True,
        colorbar=dict(
            title="z-score", tickvals=[-4,-2,0,2,4],
            ticktext=["≤-4","-2","0","+2","≥+4"],
            thickness=12, len=0.6,
        ),
        xgap=2, ygap=2,
        hovertemplate="<b>%{x} %{y}</b><br>%{customdata}<extra></extra>",
    ))
    fig.update_traces(textfont=dict(size=9, family="Source Sans 3", color="#333"))
    _apply_layout(fig)
    return fig

# ══════════════════════════════════════════════════════════════════
# Precompute national matrices (birth + conception)
# ══════════════════════════════════════════════════════════════════
nl_birth_df = df_full[["date","month","day", NL_COL]].copy()
rank_matrix_b, text_matrix_b, nl_birth_df = build_rank_matrices(
    nl_birth_df, "month", "day", NL_COL)
rank1_birth = nl_birth_df[nl_birth_df["rank"] == 1].iloc[0]

# Conception
concept_df = nl_birth_df.copy()
concept_df["concept_date"] = concept_df["date"] - pd.Timedelta(days=CONCEPTION_DAYS)
concept_df["con_month"]    = concept_df["concept_date"].dt.month
concept_df["con_day"]      = concept_df["concept_date"].dt.day
concept_agg = (concept_df
    .groupby(["con_month","con_day"])[NL_COL].sum()
    .reset_index())
concept_agg = concept_agg[
    ~((concept_agg["con_month"]==2) & (concept_agg["con_day"]==29))
]
rank_matrix_c, text_matrix_c, concept_agg = build_rank_matrices(
    concept_agg, "con_month", "con_day", NL_COL)
rank1_concept = concept_agg[concept_agg["rank"] == 1].iloc[0]
rank1_concept_row = pd.Series({
    "month": rank1_concept["con_month"],
    "day":   rank1_concept["con_day"],
})

fig_birth   = make_rank_heatmap(rank_matrix_b, text_matrix_b, COLORSCALE_BLUE,  rank1_birth,        "#f5a623")
fig_concept = make_rank_heatmap(rank_matrix_c, text_matrix_c, COLORSCALE_GREEN, rank1_concept_row,  "#e8a020")

# ══════════════════════════════════════════════════════════════════
# Helper: build gemeente rank heatmaps on demand
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def build_gemeente_figs(gemeente: str):
    """Return (fig_birth_gem, fig_concept_gem, fig_z) for a gemeente."""
    gem_df = df_full[["date","month","day", gemeente]].copy()
    gem_df[gemeente] = pd.to_numeric(gem_df[gemeente], errors="coerce").fillna(0)

    # Birth rank
    rm_b, tm_b, gem_ranked = build_rank_matrices(gem_df, "month", "day", gemeente)
    r1 = gem_ranked[gem_ranked["rank"] == 1].iloc[0]
    fb = make_rank_heatmap(rm_b, tm_b, COLORSCALE_BLUE,
                           pd.Series({"month": r1["month"], "day": r1["day"]}), "#f5a623")

    # Conception rank
    gc = gem_ranked.copy()
    gc["concept_date"] = gc["date"] - pd.Timedelta(days=CONCEPTION_DAYS)
    gc["con_month"]    = gc["concept_date"].dt.month
    gc["con_day"]      = gc["concept_date"].dt.day
    c_agg = gc.groupby(["con_month","con_day"])[gemeente].sum().reset_index()
    c_agg = c_agg[~((c_agg["con_month"]==2) & (c_agg["con_day"]==29))]
    rm_c, tm_c, c_agg = build_rank_matrices(c_agg, "con_month", "con_day", gemeente)
    r1c = c_agg[c_agg["rank"] == 1].iloc[0]
    fc = make_rank_heatmap(rm_c, tm_c, COLORSCALE_GREEN,
                           pd.Series({"month": r1c["con_month"], "day": r1c["con_day"]}), "#e8a020")

    # Z-score
    zdf        = get_gemeente_zscores(gemeente)
    zm, tm, hm = build_zscore_matrices(zdf)
    fz         = make_zscore_heatmap(zm, tm, hm)

    return fb, fc, fz, zdf

# ══════════════════════════════════════════════════════════════════
# Choropleth map: Chi² per gemeente
# ══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_geojson_gemeenten():
    return gpd.read_file(GEOJSON_URL)

def make_chi2_map(summary_df: pd.DataFrame, value_col: str = "chi2", title: str = "Chi² per gemeente"):
    """
    Choropleth map colored by chi2 (or p_value) using binned Blues palette,
    matching the pattern from tweedekamer.py.
    """
    gdf = load_geojson_gemeenten()

    # Prepare data – apply name fixes so CSV names match GeoJSON statnaam
    df_map = summary_df[["gemeente", value_col, "significant", "p_value", "total"]].copy()
    df_map["Gemeente"] = df_map["gemeente"].replace(GEMEENTE_FIX)

    # Merge so we know which municipalities matched
    gdf["Gemeente"] = gdf["statnaam"]
    merged = gdf[["Gemeente", "geometry"]].merge(df_map, on="Gemeente", how="left")

    #val_max = df_map[value_col].quantile(0.95)   # clip top 5% to avoid one outlier dominating
    val_max = df_map[value_col].max()
    # Bins: 8 equal-width steps up to val_max
    pct_edges = [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100]
    #edges = [val_max * p / 100 for p in pct_edges]
    edges = [df_map[value_col].quantile(p/100) for p in pct_edges]
    def fmt(x):
        return f"{x:.0f}" if x >= 10 else f"{x:.1f}"

    labels = (
        [f"< {fmt(edges[1])}"] +
        [f"{fmt(edges[i])}–{fmt(edges[i+1])}" for i in range(1, len(edges) - 1)]
    )

    df_map["klasse"] = pd.cut(
        df_map[value_col].clip(upper=val_max),
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
        ordered=True,
    )
    # Also put klasse onto the merged gdf for hover
    df_map_klasse = df_map[["Gemeente","klasse","significant","p_value","total", value_col]]

    # Use px.choropleth_mapbox with the geopandas GeoDataFrame
    palette = px.colors.sequential.Blues[1:9]   # 8 light→dark steps

    fig = px.choropleth_mapbox(
        df_map_klasse,
        geojson=gdf.__geo_interface__,
        locations="Gemeente",
        featureidkey="properties.statnaam",
        color="klasse",
        category_orders={"klasse": labels},
        color_discrete_sequence=palette,
        hover_data={
            "Gemeente":    True,
            value_col:     ":.1f",
            "p_value":     ":.4f",
            "total":       ":,",
            "significant": True,
            "klasse":      False,
        },
        mapbox_style="carto-positron",
        zoom=6,
        center={"lat": 52.2, "lon": 5.3},
        opacity=0.85,
    )
    fig.update_layout(
    legend=dict(traceorder="normal"),
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        height=620,
        title=dict(text=title, font=dict(size=14, family="Source Serif 4")),
        legend_title_text=f"{value_col} (klasse)",
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════
st.markdown("## Hoe gewoon is jouw verjaardag?")
st.markdown(
    '<p class="subtitle">Rang van geboorte- en conceptiedata in Nederland (2024). '
    'Rang 1 = meest voorkomend, rang 365 = minst voorkomend.</p>',
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["🇳🇱 Nederland", "🏘️ Gemeente"])

# ── Tab 1: national ──────────────────────────────────────────────
with tab1:
    col_b, col_c = st.columns(2)
    with col_b:
        st.markdown("### 🎂 Geboortedatum")
        st.plotly_chart(fig_birth, use_container_width=True)
    with col_c:
        st.markdown("### 🌱 Conceptiedatum")
        st.markdown(f'<p class="subtitle">Geboortedatum − {CONCEPTION_DAYS} dagen</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_concept, use_container_width=True)
    st.markdown('<p class="source-note">Bron: CBS / Gemeentelijke basisadministratie 2024</p>',
                unsafe_allow_html=True)

# ── Tab 2: gemeente ──────────────────────────────────────────────
with tab2:
    st.markdown("### Geboortepatroon per gemeente")

    # ── paste this inside tab2, e.g. just above the overview table ──
    with st.expander("🔍 Naam-mismatch diagnose: CSV ↔ GeoJSON"):
        in_csv_not_geo, in_geo_not_csv = get_name_mismatches(gemeente_cols)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**In CSV maar NIET op kaart** ({len(in_csv_not_geo)})")
            if in_csv_not_geo:
                st.dataframe(pd.DataFrame({"Gemeente (na fix)": in_csv_not_geo}),
                            hide_index=True, use_container_width=True)
            else:
                st.success("Geen mismatches ✅")
        with col_b:
            st.markdown(f"**Op kaart maar NIET in CSV** ({len(in_geo_not_csv)})")
            if in_geo_not_csv:
                st.dataframe(pd.DataFrame({"statnaam (GeoJSON)": in_geo_not_csv}),
                            hide_index=True, use_container_width=True)
            else:
                st.success("Geen mismatches ✅")

        # Also show the current fix dict so it's easy to extend
        st.markdown("**Huidige `GEMEENTE_FIX` (CSV-naam → GeoJSON-naam)**")
        fix_df = pd.DataFrame(
            [{"CSV-naam": k, "GeoJSON-naam": v} for k, v in GEMEENTE_FIX.items()]
        )
        st.dataframe(fix_df, hide_index=True, use_container_width=True)
    st.markdown(
        '<p class="subtitle">'
        'Selecteer een gemeente om de rang-heatmaps (blauw/groen) én de afwijking t.o.v. '
        'het landelijk gemiddelde (rood–groen z-score) te zien. '
        '<b style="color:#1a9850">Groen</b> = meer dan verwacht, '
        '<b style="color:#d73027">rood</b> = minder dan verwacht. '
        'Gemeentes met ★ wijken statistisch significant af (chi-kwadraat, p&nbsp;&lt;&nbsp;0.05).</p>',
        unsafe_allow_html=True
    )

    sig_count = int(summary_df["significant"].sum())
    st.markdown(
        f"**{sig_count} van {len(summary_df)} gemeentes** wijken significant af "
        f"van de landelijke verdeling (p < 0.05)"
    )

    # Dropdown: significant first, then rest, alphabetically within groups
    sig_gems  = sorted(summary_df[summary_df["significant"]]["gemeente"].tolist())
    nsig_gems = sorted(summary_df[~summary_df["significant"]]["gemeente"].tolist())
    all_options = ["— kies een gemeente —"] + sig_gems + nsig_gems

    gemeente_sel = st.selectbox(
        "Selecteer gemeente  (★ = significant afwijkend)",
        all_options,
        format_func=lambda x: f"★ {x}" if x in sig_gems else x,
    )

    if gemeente_sel != "— kies een gemeente —":
        gem_row = summary_df[summary_df["gemeente"] == gemeente_sel].iloc[0]

        # Stats bar
        sig_label = "✅ Significant" if gem_row["significant"] else "❌ Niet significant"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Geboorten (2024)", f"{gem_row['total']:,}")
        c2.metric("Chi²", f"{gem_row['chi2']:.1f}")
        c3.metric("p-waarde",
                  f"{gem_row['p_value']:.4f}" if gem_row["p_value"] >= 0.0001 else "< 0.0001")
        c4.metric("Significantie", sig_label)

        st.markdown("---")

        # Build all three figures (cached)
        with st.spinner(f"Berekenen voor {gemeente_sel}…"):
            fig_b_gem, fig_c_gem, fig_z, zdf = build_gemeente_figs(gemeente_sel)

        # Row 1: rank heatmaps
        col_b2, col_c2 = st.columns(2)
        with col_b2:
            st.markdown(f"#### 🎂 Geboortedatum — {gemeente_sel}")
            st.plotly_chart(fig_b_gem, use_container_width=True)
        with col_c2:
            st.markdown(f"#### 🌱 Conceptiedatum — {gemeente_sel}")
            st.markdown(f'<p class="subtitle">Geboortedatum − {CONCEPTION_DAYS} dagen</p>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_c_gem, use_container_width=True)

        # Row 2: z-score heatmap
        st.markdown(f"#### 📊 Afwijking t.o.v. landelijk — {gemeente_sel}")
        st.plotly_chart(fig_z, use_container_width=True)

        # Top-10 outlier days
        zdf_out = zdf.copy()
        zdf_out["datum"] = zdf_out.apply(
            lambda r: f"{_safe_int(r['day'])} {MONTH_NAMES[_safe_int(r['month'])-1]}", axis=1)
        zdf_out["richting"] = zdf_out["z_score"].apply(
            lambda z: "📈 meer" if (pd.notna(z) and z > 0) else "📉 minder")
        zdf_out["z_abs"] = zdf_out["z_score"].abs()
        top_out = zdf_out.nlargest(10, "z_abs")

        st.markdown("#### Top 10 meest afwijkende dagen")
        st.dataframe(
            top_out[["datum","observed","expected","z_score","richting"]]
            .rename(columns={
                "datum":     "Datum",
                "observed":  "Waargenomen",
                "expected":  "Verwacht",
                "z_score":   "Z-score",
                "richting":  "Richting",
            })
            .reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

    # Chi² map (always visible)
    st.markdown("---")
    st.markdown("#### 🗺️ Kaart: Chi² per gemeente")
    st.markdown(
        '<p class="subtitle">Donkerder blauw = grotere afwijking van de landelijke geboorteverdeling. '
        'Klik op een gemeente voor details. Gemeentes zonder data zijn grijs.</p>',
        unsafe_allow_html=True,
    )

    map_col_choice = st.radio(
        "Kleur op basis van",
        ["chi2", "p_value"],
        horizontal=True,
        format_func=lambda x: "Chi² (absolute afwijking)" if x == "chi2" else "p-waarde (significantie)",
        key="map_value_col",
    )
    map_title = (
        "Chi² per gemeente — afwijking geboorteverdeling t.o.v. Nederland (2024)"
        if map_col_choice == "chi2"
        else "p-waarde per gemeente — chi-kwadraattoets geboorteverdeling (2024)"
    )
    with st.spinner("Kaart laden…"):
        fig_map = make_chi2_map(summary_df, value_col=map_col_choice, title=map_title)
    st.plotly_chart(fig_map, use_container_width=True)

    # Overview table (always visible)
    st.markdown("---")
    st.markdown("#### Ranglijst: meest afwijkende gemeentes (top 30)")
    disp = summary_df.head(30).copy()
    disp["significant"] = disp["significant"].map({True: "✅ ja", False: "❌ nee"})
    disp["p_value"]     = disp["p_value"].apply(
        lambda p: f"{p:.4f}" if p >= 0.0001 else "< 0.0001")
    st.dataframe(
        disp[["gemeente","total","chi2","p_value","significant"]]
        .rename(columns={
            "gemeente":    "Gemeente",
            "total":       "Geboorten",
            "chi2":        "Chi²",
            "p_value":     "p-waarde",
            "significant": "Significant",
        })
        .reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

# ── Sidebar: birthday lookup ─────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎂 Zoek jouw verjaardag")
    month_sel = st.selectbox("Maand", MONTH_NAMES, index=0)
    m_idx     = MONTH_NAMES.index(month_sel) + 1
    day_sel   = st.number_input("Dag", min_value=1, max_value=DAYS_IN_MONTH[m_idx-1], value=1)

    result = nl_birth_df[(nl_birth_df["month"] == m_idx) & (nl_birth_df["day"] == day_sel)]
    if not result.empty:
        r     = result.iloc[0]
        rank  = _safe_int(r["rank"])
        count = _safe_int(r[NL_COL])
        st.markdown("---")
        st.markdown(f"**{day_sel} {month_sel}**")
        st.metric("Rang", f"#{rank} van 365")
        st.metric("Geboorten (2024)", f"{count:,}")
        verdict = (
            "🔥 Zeer populair!"  if rank <= 30  else
            "😊 Bovengemiddeld"  if rank <= 100 else
            "😐 Gemiddeld"       if rank <= 250 else
            "❄️ Zeldzame datum"
        )
        st.info(verdict)

    st.markdown("---")
    st.markdown("### Top 5 populairst")
    for _, row in nl_birth_df.nsmallest(5, "rank").iterrows():
        mn = MONTH_NAMES[_safe_int(row["month"]) - 1]
        st.markdown(f"**#{_safe_int(row['rank'])}** {_safe_int(row['day'])} {mn} ({_safe_int(row[NL_COL]):,})")

    st.markdown("### Top 5 zeldzaamst")
    for _, row in nl_birth_df.nlargest(5, "rank").iterrows():
        mn = MONTH_NAMES[_safe_int(row["month"]) - 1]
        st.markdown(f"**#{_safe_int(row['rank'])}** {_safe_int(row['day'])} {mn} ({_safe_int(row[NL_COL]):,})")
    
    st.info("Inspired by https://x.com/Globalstats11/status/2034256404119482460/photo/1")
    st.info("Data: https://www.cbs.nl/nl-nl/maatwerk/2024/39/verjaardagen-in-nederland-2024")
