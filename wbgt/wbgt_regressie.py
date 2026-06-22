# version : 20260606-120000 - Initial version: WBGT multiple regressie analyse

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from io import StringIO
from wbgt_utils import wbgt_buiten_simpel
st.set_page_config(
    page_title="WBGT Regressie Analyse",
    page_icon=":material/thermostat:",
    layout="wide",
)

PREDICTOREN = [ "F", "T", "Q", "U"]
TARGET = "wbgt_buiten"
NAAM_VOORSPELD = "wbgt_buiten_voorspeld"
NAAM_SIMPEL = "wbgt_buiten_simpel"

@st.cache_data
def laad_data_original(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.read().decode("utf-8")
    df = pd.read_csv(StringIO(content), index_col=0)
    return df

@st.cache_data()
# def get_data():
def laad_data()-> pd.DataFrame:
    """"Laad de berekende data"""
    # url=r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\show_knmi_functions\wbgt_results_1990_2026.csv"
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/wbgt/wbgt_results_1990_2026.csv"
    
    # 0,260,1990-01-01,1,5,4,0,93,1990,1,1,1,1,0.4,0.5,93,0.0,1990-01-01 01:00:00,-0.9,-0.1,6.5,0.4,HK 0,Laag risico
    # tot
    # 319079,260,2026-05-26,24,30,183,0,79,2026,5,26,146,319080,18.3,3.0,79,0.0,2026-05-27 00:00:00,16.4,16.5,20.8,18.3,HK 2,Laag risico

    df = pd.read_csv(url, delimiter=",",
               
                comment="#",
                low_memory=False,)

    # dit zijn de afkappunten zoals in het KNMI rapport (WR02-2026)
    df = df[df["dt_utc"] >= "1991-01-01 00:00:01"]
    df = df[df["dt_utc"] <= "2025-07-03 23:59:59"]
    # st.write(f"Lengte na selectie {len(df)}")
    
    # Per dag de rij selecteren waarop wbgt_buiten het hoogst is (doorgaans vroege middag).
    # Hierdoor bevat df_dagmax één rij per dag, met alle bijbehorende waarden (T, RH, wind, Q)
    # op het moment van de dagelijkse piek — niet alleen de piekwaarde zelf.
    # df_dagmax = df.loc[df.groupby("YYYYMMDD")["wbgt_buiten"].idxmax()].reset_index(drop=True)
    
    return df


def valideer_kolommen(df: pd.DataFrame) -> tuple[bool, list[str]]:
    benodigde_kolommen = PREDICTOREN + [TARGET]
    ontbrekend = [k for k in benodigde_kolommen if k not in df.columns]
    return len(ontbrekend) == 0, ontbrekend


def voer_regressie_uit(df: pd.DataFrame) -> tuple[pd.DataFrame, object, object]:
    df_clean = df[PREDICTOREN + [TARGET]].dropna()

    X = df_clean[PREDICTOREN]
    y = df_clean[TARGET]

    # sklearn voor R² en voorspellingen
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    df_clean = df_clean.copy()
    df_clean[NAAM_VOORSPELD] = y_pred

    # statsmodels voor volledige statistieken (p-waarden, SE, etc.)
    X_sm = sm.add_constant(X)
    model_sm = sm.OLS(y, X_sm).fit()

    return df_clean, model, model_sm


def beoordeel_model(r2: float, model_sm) -> tuple[str, str]:
    """Geef een kwalitatieve beoordeling van het model."""
    p_waarden = model_sm.pvalues[1:]  # zonder intercept
    n_significant = (p_waarden < 0.05).sum()
    n_totaal = len(p_waarden)

    if r2 >= 0.90:
        kwaliteit = "Uitstekend"
        kleur = "green"
    elif r2 >= 0.75:
        kwaliteit = "Goed"
        kleur = "green"
    elif r2 >= 0.50:
        kwaliteit = "Matig"
        kleur = "orange"
    else:
        kwaliteit = "Zwak"
        kleur = "red"

    advies_delen = [
        f"**R² = {r2:.4f}** — {kwaliteit} model: de predictoren verklaren "
        f"{r2 * 100:.1f}% van de variantie in `{TARGET}`.",
        f"{n_significant} van de {n_totaal} predictoren zijn statistisch significant (p < 0.05).",
    ]

    if r2 >= 0.75 and n_significant >= 3:
        advies_delen.append(
            "✅ De vergelijking is **betrouwbaar** voor gebruik als benadering van WBGT buiten."
        )
    elif r2 >= 0.50:
        advies_delen.append(
            "⚠️ De vergelijking is **bruikbaar** maar wees voorzichtig bij gebruik buiten het trainingsbereik."
        )
    else:
        advies_delen.append(
            "❌ De vergelijking is **onbetrouwbaar** — overweeg extra predictoren of een niet-lineair model."
        )

    return "\n\n".join(advies_delen), kleur

def wbgt_regressie():
    # ── UI ──────────────────────────────────────────────────────────────────────

    st.title("WBGT meervoudige regressie analyse")
    st.caption("Voorspel `wbgt_buiten` op basis van uurlijkse KNMI-variabelen")

    st.header(":material/settings: Opties")
    toon_residuen = st.checkbox("Toon residu-plot", value=True)
    toon_tabel = st.checkbox("Toon coëfficiëntentabel", value=True)
    toon_data = st.checkbox("Toon ruwe data", value=False)


    with st.spinner("Data laden en regressie uitvoeren..."):
        df = laad_data()
        ok, ontbrekend = valideer_kolommen(df)

    if not ok:
        st.error(
            f"Ontbrekende kolommen: **{', '.join(ontbrekend)}**. "
            "Controleer of het juiste bestand is geüpload.",
            icon=":material/error:",
        )
        st.stop()

    df_result, model_sk, model_sm = voer_regressie_uit(df)

    y_echt = df_result[TARGET]
    y_pred = df_result[NAAM_VOORSPELD]
    r2 = r2_score(y_echt, y_pred)
    residuen = y_echt - y_pred
    rmse = np.sqrt(np.mean(residuen**2))
    mae = np.mean(np.abs(residuen))

    # ── KPI-balk ────────────────────────────────────────────────────────────────
    with st.container(horizontal=True):
        st.metric("R²", f"{r2:.4f}", border=True)
        st.metric("RMSE (°C)", f"{rmse:.3f}", border=True)
        st.metric("MAE (°C)", f"{mae:.3f}", border=True)
        st.metric("Aantal rijen", f"{len(df_result):,}", border=True)

    st.divider()

    # ── Vergelijkingsformule ─────────────────────────────────────────────────────
    intercept = model_sk.intercept_
    coefs = dict(zip(PREDICTOREN, model_sk.coef_))

    formule_delen = [f"{intercept:.4f}"]
    for naam, coef in coefs.items():
        teken = "+" if coef >= 0 else "−"
        formule_delen.append(f"{teken} {abs(coef):.4f} × {naam}")

    formule_str = "wbgt\\_buiten\\_voorspeld = " + " ".join(formule_delen)

    with st.expander(":material/functions: Regressievergelijking", expanded=True):
        st.latex(
            "\\widehat{\\text{wbgt\\_buiten}} = "
            + f"{intercept:.4f} "
            + " ".join(
                f"{'+ ' if c >= 0 else '- '}{abs(c):.4f}\\,\\text{{{p}}}"
                for p, c in coefs.items()
            )
        )

    # ── Coëfficiëntentabel ───────────────────────────────────────────────────────
    if toon_tabel:
        st.subheader(":material/table_chart: Coëfficiënten en statistieken")

        params = model_sm.params
        se = model_sm.bse
        tval = model_sm.tvalues
        pval = model_sm.pvalues
        conf = model_sm.conf_int()

        rijen = []
        labels = ["Intercept"] + PREDICTOREN
        for label in labels:
            sleutel = "const" if label == "Intercept" else label
            rijen.append(
                {
                    "Predictor": label,
                    "Coëfficiënt": round(params[sleutel], 5),
                    "Std. fout": round(se[sleutel], 5),
                    "t-waarde": round(tval[sleutel], 3),
                    "p-waarde": round(pval[sleutel], 5),
                    "95% CI laag": round(conf.loc[sleutel, 0], 5),
                    "95% CI hoog": round(conf.loc[sleutel, 1], 5),
                    "Significant": "✅" if pval[sleutel] < 0.05 else "❌",
                }
            )

        df_coef = pd.DataFrame(rijen)
        st.dataframe(df_coef, hide_index=True, width="stretch")

    # ── Scatterplot: voorspeld vs echt ──────────────────────────────────────────
    st.subheader(":material/scatter_plot: Voorspeld vs. gemeten WBGT")

    col1, col2 = st.columns([3, 2])

    with col1:
        fig_scatter = px.scatter(
            df_result,
            x=TARGET,
            y=NAAM_VOORSPELD,
            opacity=0.4,
            labels={
                TARGET: "Gemeten WBGT buiten (°C)",
                NAAM_VOORSPELD: "Voorspelde WBGT buiten (°C)",
            },
            color_discrete_sequence=["#1f77b4"],
        )

        min_val = min(y_echt.min(), y_pred.min())
        max_val = max(y_echt.max(), y_pred.max())
        fig_scatter.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Perfecte voorspelling (y = x)",
                line=dict(color="red", dash="dash", width=2),
            )
        )
        fig_scatter.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=450,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        # Coëfficiënten bar chart
        df_bar = pd.DataFrame(
            {"Predictor": PREDICTOREN, "Coëfficiënt": model_sk.coef_}
        ).sort_values("Coëfficiënt")
        fig_bar = px.bar(
            df_bar,
            x="Coëfficiënt",
            y="Predictor",
            orientation="h",
            color="Coëfficiënt",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            title="Relatieve bijdrage predictoren",
        )
        fig_bar.update_layout(showlegend=False, height=450)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Residu-plot ──────────────────────────────────────────────────────────────
    if toon_residuen:
        st.subheader(":material/analytics: Residu-analyse")
        st.info("""**Residu-analyse**

De residuen zijn de verschillen tussen de gemeten en de voorspelde waarden. Een goed model heeft residuen die:

- **willekeurig verspreid** zijn rond de nul-lijn — zonder zichtbaar patroon
- **normaal verdeeld** zijn, zichtbaar als een klokvorm in het histogram
- **geen trechtervorm** vertonen (heteroscedasticiteit): als de spreiding toeneemt bij hogere voorspelde waarden, schat het model bepaalde bereiken systematisch slechter in

**Wat te verwachten bij WBGT:**
Bij een goed lineair model liggen de meeste residuen binnen ±1–2 °C. Grotere afwijkingen bij hoge WBGT-waarden zijn niet ongewoon, omdat extreme hitte-omstandigheden moeilijker te voorspellen zijn met lineaire combinaties van de invoervariabelen.

**Vuistregels:**
- Symmetrisch histogram rond nul → geen systematische over- of onderschatting
- Patroon in de residu-plot (boog, trechter) → overweeg een niet-lineair model of interactietermen
- Uitschieters > 3× de standaarddeviatie → controleer of dit meetfouten of extreme weersomstandigheden zijn""")
        col3, col4 = st.columns(2)

        with col3:
            fig_res = px.scatter(
                x=y_pred,
                y=residuen,
                opacity=0.4,
                labels={"x": "Voorspelde waarde (°C)", "y": "Residu (°C)"},
                title="Residuen vs. voorspelde waarden",
                color_discrete_sequence=["#ff7f0e"],
            )
            fig_res.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_res.update_layout(height=350)
            st.plotly_chart(fig_res, use_container_width=True)

        with col4:
            fig_hist = px.histogram(
                x=residuen,
                nbins=50,
                labels={"x": "Residu (°C)", "y": "Frequentie"},
                title="Verdeling residuen",
                color_discrete_sequence=["#2ca02c"],
            )
            fig_hist.update_layout(height=350)
            st.plotly_chart(fig_hist, use_container_width=True)

    # ── Modelbeoordeling ─────────────────────────────────────────────────────────
    st.subheader(":material/fact_check: Modelbeoordeling")
    beoordeling, kleur = beoordeel_model(r2, model_sm)

    if kleur == "green":
        st.success(beoordeling, icon=":material/check_circle:")
    elif kleur == "orange":
        st.warning(beoordeling, icon=":material/warning:")
    else:
        st.error(beoordeling, icon=":material/cancel:")

    with st.expander(":material/info: Volledige statsmodels samenvatting"):
        st.write(model_sm.summary().as_text())

    # ── Ruwe data ────────────────────────────────────────────────────────────────
    if toon_data:
        st.subheader(":material/table: Ruwe data met voorspellingen")
        kolommen_tonen = PREDICTOREN + [TARGET, NAAM_VOORSPELD]
        beschikbare_kolommen = [k for k in kolommen_tonen if k in df_result.columns]
        st.dataframe(
            df_result[beschikbare_kolommen].round(3),
            hide_index=True,
            width="stretch",
        )

if __name__=="__main__":
    wbgt_regressie()