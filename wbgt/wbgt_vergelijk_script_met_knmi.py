# version : 20260603140000 - Initial version: Streamlit + Plotly omzetting van vergelijk_wbgt script

current_version = "20260603140000"

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from wbgt_utils import wbgt_bereken_df
from datetime import datetime, timedelta
st.set_page_config(
    page_title="Vergelijk WBGT",
    page_icon=":material/thermostat:",
    layout="wide",
)

FOLDER_DEFAULT = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\show_knmi_functions"


def download_voor_vergelijking():
    """Laad daggegevens en bereken WBGT.
    Returns:
        df: daggegevens + wbgt + hk
    """
    

    stn = 260
    start_ = "2026-01-01"
    start_ = "2026-05-20"
    
    today = datetime.today().strftime("%Y-%m-%d")
    from__ =  start_
    until__ =  today

    fromx = from__.replace("-", "")
    until = until__.replace("-", "")

    url = f"https://www.daggegevens.knmi.nl/klimatologie/uurgegevens?stns={stn}&vars=T:U:FF:Q:P&start={fromx}00&end={until}23"
    try:
        df = pd.read_csv(
                url,
                delimiter=",",
                header= None,
                comment="#",
                low_memory=False,
            )
    except:
        print ("Error")
    
    column_replacements = [
        [0, "STN"],
        [1, "YYYYMMDD"],

        [2, "HH"],
        [3, "T"],
        [4, "U"],
        [5, "F"],
        [6, "Q"],
        [7, "P"],        
    ]

    for c in column_replacements:
        df = df.rename(columns={c[0]: c[1]})
    
    df["YYYYMMDD"] = pd.to_datetime(df["YYYYMMDD"].astype(str))
    df["YYYY"] = df["YYYYMMDD"].dt.year
    df["MM"] = df["YYYYMMDD"].dt.month
    df["DD"] = df["YYYYMMDD"].dt.day
    df["dayofyear"] = df["YYYYMMDD"].dt.dayofyear
    df["id"] =  range(1, len(df) + 1)
   
    df_result = wbgt_bereken_df(df, stn=260)
   
    return df_result
        
# ── helpers ──────────────────────────────────────────────────────────────────

# @st.cache_data
def laad_en_merge( pad_knmi: str) -> pd.DataFrame:
    df_script = download_voor_vergelijking()
    df_knmi   = pd.read_csv(pad_knmi)
    # df_knmi["uur"] = df_knmi["uur"] + 1  # KNMI-data is UTC, script-data is lokale tijd (UTC+1) — dit corrigeren voor juiste merge  
    # script-df normaliseren
    df_script = df_script.rename(columns={
        "YYYYMMDD":    "datum",
        "HH":          "uur",
        "wbgt_buiten": "wbgt_script",
    })
    df_script["hk_script"] = (
        df_script["wbgt_risico_niveau"]
        .str.extract(r"(\d+)")[0]
        .astype(float)
    )
    df_script["datum"] = df_script["datum"].astype(str)
    df_script["uur"]   = df_script["uur"].astype(int)

    # knmi-df normaliseren
    df_knmi = df_knmi.rename(columns={"wbgt": "wbgt_knmi", "hk": "hk_knmi"})
    df_knmi["datum"] = df_knmi["datum"].astype(str)
    df_knmi["uur"]   = df_knmi["uur"].astype(int)

    merged = pd.merge(df_script, df_knmi, on=["datum", "uur"], how="inner")
    # merged["wbgt_knmi"] = merged["wbgt_knmi"].shift(2)
    # merged["wbgt_script"] = merged["wbgt_script"].shift(-2)
    merged["wbgt_pct_diff"] = (
        (merged["wbgt_script"] - merged["wbgt_knmi"]) / merged["wbgt_knmi"] * 100
    )
    merged["hk_abs_diff"] = (
        (merged["hk_script"] - merged["hk_knmi"])
    )

    merged["wbgt_pct_diff_abs"] = merged["wbgt_pct_diff"].abs()
    merged["hk_abs_diff_abs"] = merged["hk_abs_diff"].abs()
    from wbgt_liljegren import solar_zenith_angle
    import math
    from datetime import datetime

    def _elevatie(row):
        dt = datetime.strptime(f"{row['datum']} {int(row['uur']):02d}:00", "%Y-%m-%d %H:%M")
        dt = dt - timedelta(minutes=30)   # zelfde midden-uur correctie als de solver
        theta = solar_zenith_angle(dt, lat_deg=52.10, lon_deg=5.18)
        return round(90.0 - math.degrees(theta), 1)

    merged["solar_elevation"] = merged.apply(_elevatie, axis=1)
    # merged = merged[(merged["uur"]>=8) & (merged["uur"]<=20)]
    return merged, df_knmi


def scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, titel: str) -> go.Figure:
    mask = df[x_col].notna() & df[y_col].notna()
    d = df[mask]
    r2 = r2_score(d[y_col], d[x_col])

    lim_min = min(d[x_col].min(), d[y_col].min()) - 0.5
    lim_max = max(d[x_col].max(), d[y_col].max()) + 0.5

    hover = ["datum", "uur"] + [c for c in ["temp_c", "wind_ms", "rh_pct", "q_wm2"] if c in d.columns]

    fig = px.scatter(
        d, x=x_col, y=y_col,
        hover_data=hover,
        opacity=0.7,
        title=f"{titel} — R² = {r2:.4f}",
        labels={x_col: x_col.replace("_", " "), y_col: y_col.replace("_", " ")},
    )
    fig.add_shape(
        type="line",
        x0=lim_min, y0=lim_min, x1=lim_max, y1=lim_max,
        line=dict(dash="dash", color="black", width=1),
    )
    fig.update_layout(
        xaxis=dict(range=[lim_min, lim_max]),
        yaxis=dict(range=[lim_min, lim_max]),
        height=450,
    )
    return fig


def regressie_plot(df: pd.DataFrame) -> tuple[go.Figure, dict]:
    features = [c for c in ["temp_c", "wind_ms", "rh_pct", "q_wm2"] if c in df.columns]
    df_reg = df[features + ["wbgt_pct_diff"]].dropna()

    X = df_reg[features].values
    y = df_reg["wbgt_pct_diff"].values

    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)

    stats = {
        "intercept": model.intercept_,
        "coef":      dict(zip(features, model.coef_)),
        "r2":        r2,
        "n":         len(df_reg),
    }

    colors = ["steelblue" if c >= 0 else "tomato" for c in model.coef_]
    fig = go.Figure(go.Bar(
        x=features,
        y=model.coef_,
        marker_color=colors,
        text=[f"{c:.4f}" for c in model.coef_],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(
        title=f"Regressiecoëfficiënten — R² = {r2:.4f}  |  N = {len(df_reg)}",
        yaxis_title="Coëfficiënt",
        height=400,
    )
    return fig, stats

def in_de_tijd_plot(df):
    df["temp_c"] = df["temp_c"].shift(-1)
    df["wbgt_knmi_min_tempc"] = df["wbgt_knmi"] - df["temp_c"]
    df["wbgt_script_min_tempc"] = df["wbgt_script"] - df["temp_c"]
    df["tijdstip"] = pd.to_datetime(df["datum"]) + pd.to_timedelta(df["uur"], unit="h")
    
    fig = go.Figure(go.Scatter(
        x=df["tijdstip"], y=df["wbgt_knmi"],
        mode="lines+markers", name="wbgt_knmi",) 
    )
    fig.add_trace(go.Scatter(
        x=df["tijdstip"], y=df["temp_c"],
        mode="lines+markers", name="temp_c",) 
    )
    fig.add_trace(go.Scatter(
        x=df["tijdstip"], y=df["wbgt_script"],
        mode="lines+markers", name="wbgt_script",) 
    )
    st.plotly_chart(fig, width="stretch")
    for w in ["wbgt_knmi_min_tempc", "wbgt_script_min_tempc"]:
        fig = px.line(
            df, x="tijdstip", y=w,
            title=f"{w} minus werkelijke temperatuur vs tijd",markers=True
        )
        st.plotly_chart(fig, width="stretch")

# ── UI ───────────────────────────────────────────────────────────────────────
def vergelijk_script_met_knmi_download():
    st.title("Vergelijk WBGT: script vs KNMI download")

    # with st.sidebar:
    #     st.subheader(":material/folder_open: Bestanden")
        # folder = st.text_input("Map", value=FOLDER_DEFAULT)

        # csv_files = []
        # if os.path.isdir(folder):
        #     csv_files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])

        # if csv_files:
        #     file_knmi   = st.selectbox("KNMI CSV",   csv_files,
        #                                index=next((i for i, f in enumerate(csv_files) if "script" not in f and "wbgt_knmi" in f), 0))
        # else:
        #     st.warning("Geen CSV-bestanden gevonden in deze map.")
        #     st.stop()

    # pad_script = os.path.join(folder, file_script)
    # pad_knmi   = os.path.join(folder, file_knmi)
    version = st.sidebar.selectbox("version", ["2.0", "3.0"], 1)

    if version=="2.0":
        pad_knmi = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/show_knmi_functions/wbgt_knmi_20260520_20260603_v20.csv"
    else:
        pad_knmi = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/show_knmi_functions/wbgt_knmi_20260520_20260603_v30.csv"

    with st.spinner("Bestanden laden en mergen..."):
        try:
            merged, df_knmi = laad_en_merge(pad_knmi)
        except Exception as e:
            st.error(f"Fout bij laden: {e}")
            st.stop()

    st.success(f":material/check_circle: {len(merged)} rijen gemerged", icon=":material/check:")

    # ── KPI-rij ──────────────────────────────────────────────────────────────────
    with st.container(horizontal=True):
        st.metric("Rijen", len(merged), border=True)
        st.metric("Gem. WBGT script", f"{merged['wbgt_script'].mean():.2f} °C", border=True)
        st.metric("Gem. WBGT KNMI",   f"{merged['wbgt_knmi'].mean():.2f} °C", border=True)
        st.metric("Gem. % verschil",  f"{merged['wbgt_pct_diff_abs'].mean():.2f} %", border=True)

    st.space("small")


    with st.expander( ":material/scatter_plot: WBGT scatter"):
        
        fig_wbgt = scatter_plot(merged, "wbgt_script", "wbgt_knmi", "WBGT script vs KNMI-value")
        st.plotly_chart(fig_wbgt, width="stretch")

        fig_wbgt = scatter_plot(merged, "temp_c", "wbgt_knmi", "KNMI-value vs real tempeature")
        st.plotly_chart(fig_wbgt, width="stretch")

        fig_wbgt = scatter_plot(merged, "temp_c","wbgt_script", "WBGT script vs real temperature")
        st.plotly_chart(fig_wbgt, width="stretch")

        fig_wbgt = px.scatter(
                merged,
                x="uur",
                y="wbgt_pct_diff")
        
        st.plotly_chart(fig_wbgt, width="stretch")

    with st.expander(":material/scatter_plot: HK scatter"):
        fig_hk = scatter_plot(merged, "hk_script", "hk_knmi", "Hittekracht (HK) script vs KNMI")
        st.plotly_chart(fig_hk, width="stretch")

    with st.expander(":material/scatter_plot: Uur vs verschil"):
        for w in ["wbgt_pct_diff", "hk_abs_diff"]:
            fig_uur = px.scatter(
                merged,
                x="uur",
                y=w,
                color="solar_elevation",          # ← dit toevoegen
                color_continuous_scale="RdYlBu_r",
                range_color=[-10, 60],
            
                hover_data=["datum"] + [c for c in ["temp_c", "wind_ms", "rh_pct", "q_wm2"] if c in merged.columns],
                opacity=0.6,
                title=f"{w} vs Uur van de dag",
                labels={"uur": "Uur (UTC)", "wbgt_pct_diff": "WBGT verschil (%)"},
            )
            fig_uur.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
            fig_uur.update_layout(height=450, xaxis=dict(tickmode="linear", dtick=1))
            st.plotly_chart(fig_uur, width="stretch")

        
        st.info("Negatief : KNMI hoger")
        for w in ["temp_c","wind_ms", "rh_pct", "q_wm2","solar_elevation" ]:
            fig_uur = px.scatter(
                    merged,
                    x=w,
                    y="wbgt_pct_diff",
                    
                    title=f" wbgt pct diff vs {w} ",
                    labels={"uur": "Uur (UTC)", "wbgt_pct_diff": "WBGT verschil (%)"},
                )
                
            st.plotly_chart(fig_uur, width="stretch")
        for col, label in [("wbgt_pct_diff", "WBGT % verschil"), ("hk_abs_diff", "HK absoluut verschil")]:
            fig = px.histogram(
                merged, x=col,
                nbins=50,
                title=f"Verdeling {label}",
                labels={col: label},
            )
            fig.add_vline(x=0, line_dash="dash", line_color="black")
            fig.add_vline(x=merged[col].mean(), line_dash="dot", line_color="red",
                        annotation_text=f"gem={merged[col].mean():.2f}", annotation_position="top right")
            fig.update_layout(height=400)
            st.plotly_chart(fig, width="stretch")
    with st.expander(":material/bar_chart: Regressie"):
        if not all(c in merged.columns for c in ["temp_c", "wind_ms", "rh_pct", "q_wm2"]):
            st.warning("Meteo-kolommen (temp_c, wind_ms, rh_pct, q_wm2) niet gevonden in script CSV.")
        else:
            fig_reg, stats = regressie_plot(merged)
            st.plotly_chart(fig_reg, width="stretch")

            with st.expander(":material/info: Regressie-details"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("R²",        f"{stats['r2']:.4f}", border=True)
                    st.metric("N",         stats["n"],           border=True)
                    st.metric("Intercept", f"{stats['intercept']:.4f}", border=True)
                with col_b:
                    st.dataframe(
                        pd.DataFrame.from_dict(stats["coef"], orient="index", columns=["coëfficiënt"]).round(4),
                        hide_index=False,
                        width="stretch",
                    )

    with st.expander(":material/table: Data"):
        st.dataframe(
            merged.round(3),
            hide_index=True,
            width="stretch",
            column_config={
                "wbgt_pct_diff": st.column_config.NumberColumn("% verschil WBGT", format="%.2f %%"),
                "wbgt_script":   st.column_config.NumberColumn("WBGT script",     format="%.2f"),
                "wbgt_knmi":     st.column_config.NumberColumn("WBGT KNMI",       format="%.2f"),
            },
        )
        st.download_button(
            label=":material/download: Download CSV",
            data=merged.to_csv(index=False).encode("utf-8"),
            file_name="wbgt_vergelijking.csv",
            mime="text/csv",
        )

    with st.expander(":material/scatter_plot: KNMI Data"):
        in_de_tijd_plot(merged)

    with st.expander(":material/scatter_plot: Liljegren vs Kong"):
        st.write(merged)
        fig_uur = scatter_plot(
            merged,
            "wbgt_script",
            "wbgt_buiten_cython",
            f"Values Kong (Cython) vs values Liljegren (C)",
            
        )
        st.plotly_chart(fig_uur, width="stretch")

        fig_wbgt = scatter_plot(merged, "wbgt_knmi", "wbgt_buiten_cython", "WBGT Kong vs KNMI-value")
        st.plotly_chart(fig_wbgt, width="stretch")



if __name__=="__main__":
    vergelijk_script_met_knmi_download()      