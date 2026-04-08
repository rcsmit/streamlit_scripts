"""
PALANTIR FOUNDRY-STYLE DASHBOARD TEMPLATE
==========================================
Dark tactical ops dashboard inspired by Palantir Foundry / AIP.
Features: KPI status cards, Gantt timeline, geo map, live feeds,
correlation charts, and threat-level indicators.

Replace the synthetic data with your own data sources.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
import random

# ─── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="FOUNDRY — OPS DASHBOARD",
    page_icon=":material/shield:",
    layout="wide",
)

# ─── Dark Palantir theme via custom CSS ────────────────────────
PALANTIR_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root variables ── */
:root {
    --pal-bg: #0B0E14;
    --pal-surface: #111620;
    --pal-surface-2: #161D2A;
    --pal-surface-3: #1C2536;
    --pal-border: #1E2A3D;
    --pal-border-bright: #2A3A55;
    --pal-text: #C8D6E5;
    --pal-text-dim: #6B7D95;
    --pal-text-bright: #E8F0FA;
    --pal-accent: #00D4AA;
    --pal-accent-dim: rgba(0,212,170,0.15);
    --pal-warning: #FF9F43;
    --pal-danger: #EE5A5A;
    --pal-info: #54A0FF;
    --pal-purple: #7C5CFC;
}

/* ── Global overrides ── */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: var(--pal-bg) !important;
    color: var(--pal-text) !important;
    font-family: 'Inter', sans-serif !important;
}

header[data-testid="stHeader"] {
    background-color: var(--pal-bg) !important;
}

[data-testid="stSidebar"] {
    background-color: var(--pal-surface) !important;
    border-right: 1px solid var(--pal-border) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, var(--pal-surface) 0%, var(--pal-surface-2) 100%) !important;
    border: 1px solid var(--pal-border) !important;
    border-radius: 6px !important;
    padding: 16px 20px !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--pal-text-dim) !important;
}

[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    color: var(--pal-text-bright) !important;
}

/* ── Containers ── */
[data-testid="stVerticalBlock"] > div > [data-testid="stContainer"][class*="border"] {
    background: var(--pal-surface) !important;
    border: 1px solid var(--pal-border) !important;
    border-radius: 6px !important;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--pal-border) !important;
    border-radius: 6px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    background: var(--pal-surface) !important;
    border-radius: 6px 6px 0 0;
    border-bottom: 1px solid var(--pal-border);
}

.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--pal-text-dim) !important;
    padding: 10px 20px !important;
}

.stTabs [aria-selected="true"] {
    color: var(--pal-accent) !important;
    border-bottom-color: var(--pal-accent) !important;
}

/* ── Section headers (via st.markdown with section-hdr class) ── */
.section-hdr p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: var(--pal-text-dim) !important;
    padding: 8px 0 4px 0 !important;
    border-bottom: 1px solid var(--pal-border) !important;
    margin-bottom: 12px !important;
}

/* ── Status badges ── */
.status-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 3px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.status-active { background: rgba(0,212,170,0.15); color: #00D4AA; border: 1px solid rgba(0,212,170,0.3); }
.status-warning { background: rgba(255,159,67,0.15); color: #FF9F43; border: 1px solid rgba(255,159,67,0.3); }
.status-danger { background: rgba(238,90,90,0.15); color: #EE5A5A; border: 1px solid rgba(238,90,90,0.3); }
.status-info { background: rgba(84,160,255,0.15); color: #54A0FF; border: 1px solid rgba(84,160,255,0.3); }

/* ── Title bar ── */
.title-bar {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 12px 0;
    margin-bottom: 8px;
}
.title-bar h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--pal-text-bright);
    margin: 0;
}
.title-bar .subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--pal-text-dim);
    letter-spacing: 0.06em;
}

/* ── KPI banner cards ── */
.kpi-card {
    background: linear-gradient(135deg, var(--pal-surface) 0%, var(--pal-surface-2) 100%);
    border: 1px solid var(--pal-border);
    border-radius: 6px;
    padding: 14px 18px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.kpi-card.accent::before { background: var(--pal-accent); }
.kpi-card.warning::before { background: var(--pal-warning); }
.kpi-card.danger::before { background: var(--pal-danger); }
.kpi-card.info::before { background: var(--pal-info); }

.kpi-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--pal-text-dim);
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--pal-text-bright);
    line-height: 1.2;
}
.kpi-value.accent { color: var(--pal-accent); }
.kpi-value.warning { color: var(--pal-warning); }
.kpi-value.danger { color: var(--pal-danger); }

/* ── Feed items ── */
.feed-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid rgba(30,42,61,0.5);
    font-size: 0.78rem;
}
.feed-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    margin-top: 6px;
    flex-shrink: 0;
}
.feed-dot.green { background: var(--pal-accent); box-shadow: 0 0 6px rgba(0,212,170,0.4); }
.feed-dot.orange { background: var(--pal-warning); box-shadow: 0 0 6px rgba(255,159,67,0.4); }
.feed-dot.red { background: var(--pal-danger); box-shadow: 0 0 6px rgba(238,90,90,0.4); }
.feed-dot.blue { background: var(--pal-info); box-shadow: 0 0 6px rgba(84,160,255,0.4); }
.feed-time {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--pal-text-dim);
    min-width: 48px;
}
.feed-text {
    color: var(--pal-text);
    line-height: 1.4;
}

/* ── Gantt bars ── */
.gantt-row {
    display: flex;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid rgba(30,42,61,0.3);
}
.gantt-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--pal-text-dim);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    width: 140px;
    flex-shrink: 0;
}
.gantt-track {
    flex: 1;
    height: 20px;
    position: relative;
    background: rgba(30,42,61,0.3);
    border-radius: 3px;
}
.gantt-bar {
    position: absolute;
    height: 100%;
    border-radius: 3px;
    opacity: 0.85;
}
.gantt-bar.teal { background: linear-gradient(90deg, #00D4AA, #00B89C); }
.gantt-bar.blue { background: linear-gradient(90deg, #54A0FF, #3D7FE6); }
.gantt-bar.orange { background: linear-gradient(90deg, #FF9F43, #E68A30); }
.gantt-bar.purple { background: linear-gradient(90deg, #7C5CFC, #6244D9); }
.gantt-bar.red { background: linear-gradient(90deg, #EE5A5A, #D44444); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--pal-surface); }
::-webkit-scrollbar-thumb { background: var(--pal-border-bright); border-radius: 3px; }

/* ── Selectbox / inputs ── */
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--pal-text-dim) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--pal-surface) !important;
    border: 1px solid var(--pal-border) !important;
    border-radius: 6px !important;
}
</style>
"""
st.markdown(PALANTIR_CSS, unsafe_allow_html=True)


# ─── Synthetic data generators ─────────────────────────────────

def generate_time_series(days=30, base=100, volatility=5):
    dates = [datetime.now() - timedelta(days=days - i) for i in range(days)]
    values = [base]
    for _ in range(days - 1):
        values.append(values[-1] + random.gauss(0.5, volatility))
    return pd.DataFrame({"date": dates, "value": values})


def generate_gantt_data():
    return [
        {"task": "RECON PHASE", "start": 5, "end": 30, "color": "teal"},
        {"task": "DATA INGEST", "start": 15, "end": 50, "color": "blue"},
        {"task": "MODEL TRAIN", "start": 35, "end": 70, "color": "purple"},
        {"task": "VALIDATION", "start": 55, "end": 80, "color": "orange"},
        {"task": "DEPLOYMENT", "start": 72, "end": 95, "color": "teal"},
        {"task": "MONITORING", "start": 80, "end": 100, "color": "red"},
    ]


def generate_scatter_data(n=60):
    x = np.random.uniform(0, 100, n)
    y = 0.5 * x + np.random.normal(0, 12, n)
    cluster = np.where(x < 40, "ALPHA", np.where(x < 70, "BRAVO", "CHARLIE"))
    return pd.DataFrame({"x_metric": x, "y_metric": y, "cluster": cluster})


def generate_asset_data():
    assets = ["NODE-A1", "NODE-A2", "NODE-B1", "NODE-B3", "NODE-C1", "NODE-C2", "NODE-D1"]
    statuses = ["ACTIVE", "ACTIVE", "ACTIVE", "WARNING", "ACTIVE", "DEGRADED", "ACTIVE"]
    loads = [87, 64, 92, 78, 45, 31, 88]
    uptimes = [99.97, 99.94, 99.88, 98.12, 99.99, 95.67, 99.91]
    return pd.DataFrame({
        "asset": assets,
        "status": statuses,
        "load_pct": loads,
        "uptime_pct": uptimes,
        "throughput": [random.randint(1200, 4500) for _ in assets],
    })


def generate_feed_events():
    return [
        ("green", "14:32", "Pipeline ALPHA-7 completed — 2.4M records processed"),
        ("blue", "14:28", "Model v3.2.1 deployed to production cluster"),
        ("orange", "14:15", "NODE-B3 latency spike detected (p99 > 450ms)"),
        ("green", "14:02", "Anomaly detection sweep passed — 0 flags"),
        ("red", "13:48", "NODE-C2 disk usage critical — 94% capacity"),
        ("blue", "13:31", "Scheduled backup completed — snapshot ID: SNP-8821"),
        ("green", "13:15", "Ingestion rate normalized — 12.4k events/sec"),
        ("orange", "12:58", "Certificate renewal pending — 3 days remaining"),
    ]


def section_header(icon: str, label: str):
    """Render a styled section header with a Material icon.
    Uses st.markdown (which processes :material/...: syntax) plus a CSS class."""
    st.markdown(
        f":material/{icon}: {label}",
        help=None,
    )
    # Apply styling via a container wrapper — use raw HTML divider
    st.markdown(
        '<hr style="margin:-10px 0 12px 0;border:none;border-top:1px solid #1E2A3D;">',
        unsafe_allow_html=True,
    )


def main():
    # ─── TITLE BAR ─────────────────────────────────────────────────
    st.markdown("""
    <div class="title-bar">
        <h1>⬡ OPS DASHBOARD — FOUNDRY</h1>
        <span class="subtitle">LAST SYNC: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC") + """</span>
        <span style="margin-left:auto;">
            <span class="status-badge status-active">● SYSTEM ONLINE</span>
        </span>
    </div>
    """, unsafe_allow_html=True)


    # ─── TOP KPI BANNER ────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown("""
        <div class="kpi-card accent">
            <div class="kpi-label">PIPELINE STATUS <span class="status-badge status-active">ACT</span></div>
            <div class="kpi-value accent">ACTIVE</div>
        </div>""", unsafe_allow_html=True)

    with k2:
        st.markdown("""
        <div class="kpi-card info">
            <div class="kpi-label">DATA THROUGHPUT <span class="status-badge status-info">LIVE</span></div>
            <div class="kpi-value">12.4K evt/s</div>
        </div>""", unsafe_allow_html=True)

    with k3:
        st.markdown("""
        <div class="kpi-card warning">
            <div class="kpi-label">THREAT ASSESSMENT <span class="status-badge status-warning">HIGH</span></div>
            <div class="kpi-value warning">7 ANOMALIES</div>
        </div>""", unsafe_allow_html=True)

    with k4:
        st.markdown("""
        <div class="kpi-card danger">
            <div class="kpi-label">SYSTEM INTEGRITY <span class="status-badge status-danger">ALERT</span></div>
            <div class="kpi-value danger">DEGRADED (87%)</div>
        </div>""", unsafe_allow_html=True)

    st.space("small")


    # ─── MAIN CONTENT: 3-column layout ────────────────────────────
    col_left, col_mid, col_right = st.columns([3, 4, 3])


    # ── LEFT: Gantt + Asset Table ──────────────────────────────────
    with col_left:
        section_header('timeline', 'TIMELINE / GANTT')

        gantt = generate_gantt_data()
        gantt_html = ""
        for item in gantt:
            gantt_html += f"""
            <div class="gantt-row">
                <div class="gantt-label">{item['task']}</div>
                <div class="gantt-track">
                    <div class="gantt-bar {item['color']}" style="left:{item['start']}%;width:{item['end']-item['start']}%"></div>
                </div>
            </div>"""
        st.markdown(gantt_html, unsafe_allow_html=True)

        st.space("medium")
        section_header('dns', 'ASSET REGISTRY')

        asset_df = generate_asset_data()
        st.dataframe(
            asset_df,
            column_config={
                "asset": st.column_config.TextColumn("ASSET ID", width="small"),
                "status": st.column_config.TextColumn("STATUS", width="small"),
                "load_pct": st.column_config.ProgressColumn(
                    "LOAD %", min_value=0, max_value=100,
                ),
                "uptime_pct": st.column_config.NumberColumn(
                    "UPTIME %", format="%.2f",
                ),
                "throughput": st.column_config.NumberColumn(
                    "THROUGHPUT", format="%d ops/s",
                ),
            },
            hide_index=True,
            width="stretch",
            height=280,
        )


    # ── MID: Charts ────────────────────────────────────────────────
    with col_mid:
        section_header('monitoring', 'TELEMETRY')

        tab1, tab2, tab3 = st.tabs(["TIME SERIES", "CORRELATION", "DISTRIBUTION"])

        with tab1:
            ts_data = generate_time_series(days=60, base=1000, volatility=25)

            # Area chart with gradient feel
            area = alt.Chart(ts_data).mark_area(
                line={"color": "#00D4AA", "strokeWidth": 2},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[
                        alt.GradientStop(color="rgba(0,212,170,0.3)", offset=0),
                        alt.GradientStop(color="rgba(0,212,170,0.02)", offset=1),
                    ],
                    x1=1, x2=1, y1=1, y2=0,
                ),
            ).encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(
                    labelColor="#6B7D95", gridColor="#1E2A3D", domainColor="#1E2A3D",
                    format="%b %d", labelFont="JetBrains Mono", labelFontSize=10,
                )),
                y=alt.Y("value:Q", title="EVENTS / MIN", axis=alt.Axis(
                    labelColor="#6B7D95", gridColor="#1E2A3D", domainColor="#1E2A3D",
                    titleColor="#6B7D95", labelFont="JetBrains Mono", labelFontSize=10,
                    titleFont="JetBrains Mono", titleFontSize=10,
                )),
                tooltip=[
                    alt.Tooltip("date:T", title="DATE", format="%Y-%m-%d"),
                    alt.Tooltip("value:Q", title="VALUE", format=".1f"),
                ],
            ).properties(
                height=260,
                background="transparent",
            ).configure_view(
                stroke=None,
            )
            st.altair_chart(area, width="stretch")

        with tab2:
            scatter_df = generate_scatter_data()
            color_scale = alt.Scale(
                domain=["ALPHA", "BRAVO", "CHARLIE"],
                range=["#00D4AA", "#54A0FF", "#FF9F43"],
            )
            scatter = alt.Chart(scatter_df).mark_circle(size=50, opacity=0.7).encode(
                x=alt.X("x_metric:Q", title="INPUT SIGNAL", axis=alt.Axis(
                    labelColor="#6B7D95", gridColor="#1E2A3D", domainColor="#1E2A3D",
                    labelFont="JetBrains Mono", labelFontSize=10,
                    titleFont="JetBrains Mono", titleFontSize=10, titleColor="#6B7D95",
                )),
                y=alt.Y("y_metric:Q", title="OUTPUT RESPONSE", axis=alt.Axis(
                    labelColor="#6B7D95", gridColor="#1E2A3D", domainColor="#1E2A3D",
                    labelFont="JetBrains Mono", labelFontSize=10,
                    titleFont="JetBrains Mono", titleFontSize=10, titleColor="#6B7D95",
                )),
                color=alt.Color("cluster:N", scale=color_scale, legend=alt.Legend(
                    titleColor="#6B7D95", labelColor="#6B7D95",
                    titleFont="JetBrains Mono", labelFont="JetBrains Mono",
                    titleFontSize=10, labelFontSize=10,
                )),
                tooltip=["cluster:N", "x_metric:Q", "y_metric:Q"],
            ).properties(
                height=260,
                background="transparent",
            ).configure_view(stroke=None)
            st.altair_chart(scatter, width="stretch")

        with tab3:
            dist_data = pd.DataFrame({
                "value": np.concatenate([
                    np.random.normal(40, 10, 200),
                    np.random.normal(75, 8, 150),
                ]),
            })
            hist = alt.Chart(dist_data).mark_bar(
                color="#7C5CFC",
                opacity=0.7,
                cornerRadiusTopLeft=2,
                cornerRadiusTopRight=2,
            ).encode(
                x=alt.X("value:Q", bin=alt.Bin(maxbins=30), title="METRIC VALUE", axis=alt.Axis(
                    labelColor="#6B7D95", gridColor="#1E2A3D", domainColor="#1E2A3D",
                    labelFont="JetBrains Mono", labelFontSize=10,
                    titleFont="JetBrains Mono", titleFontSize=10, titleColor="#6B7D95",
                )),
                y=alt.Y("count()", title="FREQUENCY", axis=alt.Axis(
                    labelColor="#6B7D95", gridColor="#1E2A3D", domainColor="#1E2A3D",
                    labelFont="JetBrains Mono", labelFontSize=10,
                    titleFont="JetBrains Mono", titleFontSize=10, titleColor="#6B7D95",
                )),
            ).properties(
                height=260,
                background="transparent",
            ).configure_view(stroke=None)
            st.altair_chart(hist, width="stretch")

        # Secondary metrics row under chart
        st.space("small")
        with st.container(horizontal=True):
            st.metric("LATENCY P99", "142ms", "-8ms", border=True,
                    chart_data=[180, 165, 158, 152, 148, 145, 142], chart_type="line")
            st.metric("ERROR RATE", "0.03%", "-0.01%", border=True,
                    chart_data=[0.08, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03], chart_type="line")
            st.metric("THROUGHPUT", "12.4K/s", "+1.2K", border=True,
                    chart_data=[9.8, 10.2, 10.8, 11.1, 11.6, 12.0, 12.4], chart_type="bar")


    # ── RIGHT: Live Feed + Threat Matrix ──────────────────────────
    with col_right:
        section_header('rss_feed', 'LIVE EVENT FEED')

        events = generate_feed_events()
        feed_html = ""
        for dot_color, time, text in events:
            feed_html += f"""
            <div class="feed-item">
                <div class="feed-dot {dot_color}"></div>
                <div class="feed-time">{time}</div>
                <div class="feed-text">{text}</div>
            </div>"""
        st.markdown(feed_html, unsafe_allow_html=True)

        st.space("medium")
        section_header('security', 'THREAT MATRIX')

        # Heatmap-style threat matrix
        threat_categories = ["INTRUSION", "MALWARE", "ANOMALY", "EXFIL", "MISCONFIG"]
        threat_sources = ["EXTERNAL", "INTERNAL", "PARTNER", "UNKNOWN"]
        threat_data = pd.DataFrame(
            np.random.choice([0, 0, 0, 1, 1, 2, 3], size=(len(threat_categories), len(threat_sources))),
            index=threat_categories,
            columns=threat_sources,
        )

        heatmap_df = threat_data.reset_index().melt(id_vars="index", var_name="source", value_name="level")
        heatmap_df.columns = ["category", "source", "level"]

        heatmap = alt.Chart(heatmap_df).mark_rect(
            cornerRadius=3,
        ).encode(
            x=alt.X("source:N", title=None, axis=alt.Axis(
                labelColor="#6B7D95", domainColor="#1E2A3D",
                labelFont="JetBrains Mono", labelFontSize=9,
                labelAngle=0,
            )),
            y=alt.Y("category:N", title=None, axis=alt.Axis(
                labelColor="#6B7D95", domainColor="#1E2A3D",
                labelFont="JetBrains Mono", labelFontSize=9,
            )),
            color=alt.Color("level:Q", scale=alt.Scale(
                domain=[0, 1, 2, 3],
                range=["#111620", "#1C3A2E", "#3D5A1E", "#EE5A5A"],
            ), legend=None),
            tooltip=["category:N", "source:N", "level:Q"],
        ).properties(
            height=180,
            background="transparent",
        ).configure_view(stroke=None)

        st.altair_chart(heatmap, width="stretch")


    # ─── BOTTOM: Multi-series + Breakdown ─────────────────────────
    st.space("small")
    bot_left, bot_right = st.columns([5, 5])

    with bot_left:
        section_header('stacked_line_chart', 'MULTI-SIGNAL OVERLAY')

        ts1 = generate_time_series(30, 500, 15)
        ts2 = generate_time_series(30, 480, 20)
        ts3 = generate_time_series(30, 520, 10)

        multi_df = pd.DataFrame({
            "date": ts1["date"],
            "SIGNAL-A": ts1["value"],
            "SIGNAL-B": ts2["value"],
            "SIGNAL-C": ts3["value"],
        })

        multi_long = multi_df.melt("date", var_name="signal", value_name="value")
        multi_chart = alt.Chart(multi_long).mark_line(strokeWidth=2).encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(
                labelColor="#6B7D95", gridColor="#1E2A3D", domainColor="#1E2A3D",
                format="%b %d", labelFont="JetBrains Mono", labelFontSize=10,
            )),
            y=alt.Y("value:Q", title="AMPLITUDE", axis=alt.Axis(
                labelColor="#6B7D95", gridColor="#1E2A3D", domainColor="#1E2A3D",
                titleColor="#6B7D95", labelFont="JetBrains Mono", labelFontSize=10,
                titleFont="JetBrains Mono", titleFontSize=10,
            )),
            color=alt.Color("signal:N", scale=alt.Scale(
                domain=["SIGNAL-A", "SIGNAL-B", "SIGNAL-C"],
                range=["#00D4AA", "#54A0FF", "#FF9F43"],
            ), legend=alt.Legend(
                titleColor="#6B7D95", labelColor="#6B7D95",
                titleFont="JetBrains Mono", labelFont="JetBrains Mono",
                titleFontSize=10, labelFontSize=10, orient="top",
            )),
            tooltip=["signal:N", alt.Tooltip("value:Q", format=".1f"), "date:T"],
        ).properties(
            height=220,
            background="transparent",
        ).configure_view(stroke=None)
        st.altair_chart(multi_chart, width="stretch")


    with bot_right:
        section_header('donut_large', 'RESOURCE ALLOCATION')

        alloc_data = pd.DataFrame({
            "resource": ["COMPUTE", "STORAGE", "NETWORK", "MEMORY", "GPU"],
            "usage_pct": [78, 54, 42, 89, 67],
            "allocated": [200, 500, 100, 128, 8],
            "unit": ["vCPU", "TB", "Gbps", "GB", "A100"],
        })

        alloc_data["color_tier"] = alloc_data["usage_pct"].apply(
            lambda x: "CRITICAL" if x > 80 else ("WARNING" if x > 60 else "NORMAL")
        )

        bars = alt.Chart(alloc_data).mark_bar(
            cornerRadiusTopRight=3,
            cornerRadiusBottomRight=3,
            height=18,
        ).encode(
            x=alt.X("usage_pct:Q", title="UTILIZATION %",
                    scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(
                        labelColor="#6B7D95", gridColor="#1E2A3D", domainColor="#1E2A3D",
                        labelFont="JetBrains Mono", labelFontSize=10,
                        titleFont="JetBrains Mono", titleFontSize=10, titleColor="#6B7D95",
                    )),
            y=alt.Y("resource:N", title=None, sort="-x", axis=alt.Axis(
                labelColor="#6B7D95", domainColor="#1E2A3D",
                labelFont="JetBrains Mono", labelFontSize=10,
            )),
            color=alt.Color("color_tier:N", scale=alt.Scale(
                domain=["NORMAL", "WARNING", "CRITICAL"],
                range=["#00D4AA", "#FF9F43", "#EE5A5A"],
            ), legend=None),
            tooltip=["resource:N", "usage_pct:Q", "allocated:Q", "unit:N"],
        ).properties(
            height=220,
            background="transparent",
        ).configure_view(stroke=None)

        st.altair_chart(bars, width="stretch")


    # ─── GEO-INTEL MAP: Asset positions ───────────────────────────
    st.space("small")
    map_left, map_right = st.columns([6, 4])

    with map_left:
        section_header('map', 'GEO-INTEL — STRATEGIC ASSET LOCATIONS')

        import pydeck as pdk

        # Fictional forward operating bases / sensor stations
        map_data = pd.DataFrame({
            "lat": [
                48.8566, 52.5200, 41.9028, 38.7223, 59.3293,
                50.0755, 47.4979, 44.4268, 40.4168, 45.4642,
                51.5074, 55.7558, 37.9838, 46.2044, 53.3498,
            ],
            "lon": [
                2.3522, 13.4050, 12.4964, -9.1393, 18.0686,
                14.4378, 19.0402, 26.1025, -3.7038, 9.1900,
                -0.1278, 37.6173, 23.7275, 6.1432, -6.2603,
            ],
            "label": [
                "STATION ÉTOILE", "OUTPOST BRANDENBURG", "NODE COLOSSEUM",
                "RELAY TEJO", "BEACON NORDIC", "SENSOR VLTAVA",
                "ARRAY DANUBE", "POST DÂMBOVIȚA", "HUB CASTILLA",
                "NEXUS LOMBARDY", "COMMAND THAMES", "FORWARD MOSKVA",
                "WATCH AEGEAN", "DEPOT LÉMAN", "LINK LIFFEY",
            ],
            "status": [
                "ACTIVE", "ACTIVE", "STANDBY", "ACTIVE", "ACTIVE",
                "WARNING", "ACTIVE", "DEGRADED", "ACTIVE", "ACTIVE",
                "ACTIVE", "STANDBY", "ACTIVE", "ACTIVE", "WARNING",
            ],
            "signal_strength": [
                92, 88, 45, 91, 87, 63, 94, 31, 89, 96,
                99, 42, 85, 90, 58,
            ],
        })

        # Color per status: ACTIVE=teal, WARNING=orange, STANDBY=blue, DEGRADED=red
        status_colors = {
            "ACTIVE":   [0, 212, 170, 200],
            "WARNING":  [255, 159, 67, 200],
            "STANDBY":  [84, 160, 255, 200],
            "DEGRADED": [238, 90, 90, 200],
        }
        map_data["color"] = map_data["status"].map(status_colors)
        map_data["radius"] = map_data["signal_strength"].apply(lambda s: 8000 + s * 400)

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_data,
            get_position=["lon", "lat"],
            get_radius="radius",
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
            highlight_color=[255, 255, 255, 80],
        )

        view_state = pdk.ViewState(
            latitude=49.5,
            longitude=10.0,
            zoom=3.3,
            pitch=0,
        )

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={
                "html": (
                    "<div style='font-family:JetBrains Mono,monospace;font-size:12px;padding:4px 8px;'>"
                    "<b>{label}</b><br/>"
                    "STATUS: {status}<br/>"
                    "SIGNAL: {signal_strength}%"
                    "</div>"
                ),
                "style": {
                    "backgroundColor": "#161D2A",
                    "color": "#E8F0FA",
                    "border": "1px solid #2A3A55",
                    "borderRadius": "4px",
                },
            },
            map_style=None,
        )

        st.pydeck_chart(
            deck,
            height=380,
            width="stretch",
        
        )

    with map_right:
        section_header('cell_tower', 'STATION REGISTRY')

        st.dataframe(
            map_data[["label", "status", "signal_strength"]],
            column_config={
                "label": st.column_config.TextColumn("STATION ID", width="medium"),
                "status": st.column_config.TextColumn("STATUS", width="small"),
                "signal_strength": st.column_config.ProgressColumn(
                    "SIGNAL %", min_value=0, max_value=100,
                ),
            },
            hide_index=True,
            width="stretch",
            height=380,
        )


    # ─── FOOTER ────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 16px 0 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--pal-text-dim);
        border-top: 1px solid var(--pal-border);
        margin-top: 16px;
    ">
        FOUNDRY OPS v2.4.1 — FULL PALANTIR / AIP — {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()