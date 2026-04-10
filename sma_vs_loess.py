import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import pearsonr
from skmisc.loess import loess as skmisc_loess
from typing import Optional

# ── CONFIG ──────────────────────────────────────────────
URL = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/de_bilt_jaargem_1901_2022.csv"
LOESS_RANGE = range(1, 100)
SMA_RANGE = range(3, 58)
# ────────────────────────────────────────────────────────


def centered_sma(y: np.ndarray, window: int) -> np.ndarray:
    """Bereken centered Simple Moving Average.

    Args:
        y: Array met jaarlijkse waarden.
        window: Venstergrootte in jaren.

    Returns:
        Array met SMA-waarden; NaN aan de randen.
    """
    return pd.Series(y).rolling(window, center=True, min_periods=window).mean().values


def loess_smooth(
    t: np.ndarray, y: np.ndarray, span_noemer: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bereken LOESS-trendlijn via scikit-misc (exacte KNMI TR-389 methode).

    Gebruikt surface='direct' en statistics='exact' conform het R-script.
    span = span_noemer / len(t), wat het 30-jarig lopend gemiddelde benadert.

    Args:
        t: Array met jaren, oplopend met stap 1.
        y: Array met jaarlijkse waarden.
        span_noemer: Venster in jaren (span = noemer / n).

    Returns:
        Tuple van (trendwaarden, ondergrens confidence interval,
        bovengrens confidence interval).

    Raises:
        ValueError: Als er onvoldoende geldige datapunten zijn.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    span = span_noemer / len(t)

    l = skmisc_loess(t, y)
    l.model.span = span
    l.model.degree = 1
    l.control.iterations = 1
    l.control.surface = "direct"
    l.control.statistics = "exact"
    l.fit()

    pred = l.predict(t, stderror=True)
    conf = pred.confidence()

    return pred.values, conf.lower, conf.upper


def load_data_from_url(url: str) -> pd.DataFrame:
    """Laad tijdreeksdata van een URL.

    Args:
        url: URL naar het CSV-bestand.

    Returns:
        DataFrame met originele kolommen, gesorteerd op eerste kolom.

    Raises:
        ValueError: Als het bestand niet minstens twee kolommen bevat.
    """
    try:
        df = pd.read_csv(url)
    except Exception:
        df = pd.read_csv(url, header=None)

    if df.shape[1] < 2:
        raise ValueError(
            f"CSV moet minstens 2 kolommen hebben, gevonden: {df.shape[1]}"
        )

    return df.sort_values(df.columns[0]).reset_index(drop=True)


def load_data_from_upload(uploaded_file: object) -> pd.DataFrame:
    """Laad tijdreeksdata van een geüpload Streamlit bestand.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        DataFrame met originele kolommen, gesorteerd op eerste kolom.

    Raises:
        ValueError: Als het bestand niet minstens twee kolommen bevat.
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, header=None)

    if df.shape[1] < 2:
        raise ValueError(
            f"CSV moet minstens 2 kolommen hebben, gevonden: {df.shape[1]}"
        )

    return df.sort_values(df.columns[0]).reset_index(drop=True)


def load_data_ui(sidebar_key: str) -> Optional[pd.DataFrame]:
    """Toon upload-widget en laad data van upload of standaard URL.

    Args:
        sidebar_key: Unieke sleutel voor de file_uploader widget.

    Returns:
        DataFrame als laden lukt, anders None.
    """
    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Upload CSV", type=["csv"], key=sidebar_key
        )

    if uploaded_file is not None:
        try:
            df = load_data_from_upload(uploaded_file)
            st.sidebar.success(f"Geüpload: {uploaded_file.name}")
            return df
        except ValueError as e:
            st.sidebar.error(str(e))
            return None
    else:
        try:
            df = load_data_from_url(URL)
            st.sidebar.info("Standaard URL gebruikt")
            return df
        except Exception as e:
            st.sidebar.error(f"Kan URL niet laden: {e}")
            return None

import pandas as pd
import numpy as np

def to_year_float(series: pd.Series) -> np.ndarray:
    """Convert a series of dates, datetimes, or numeric years to float.
    
    Handles:
      - datetime64 / Timestamp  → fractional year (e.g. 2023.5)
      - strings that look like dates (e.g. "2023-06-15") → fractional year
      - strings that look like years (e.g. "2023")        → float year
      - numeric int/float (already a year)                → float as-is
      - anything else                                     → row index (0, 1, 2, …)
    """
    # Try datetime conversion first (works for date strings too)
    try:
        dt = pd.to_datetime(series)
        year         = dt.dt.year.astype(float)
        day_of_year  = dt.dt.day_of_year.astype(float)
        days_in_year = dt.dt.is_leap_year.map({True: 366.0, False: 365.0})
        return (year + (day_of_year - 1) / days_in_year).values
    except (TypeError, ValueError, pd.errors.ParserError):
        pass

    # Try numeric year
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(float).values

    # Fallback: row numbers
    return np.arange(len(series), dtype=float)

def select_columns(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Laat de gebruiker kolommen kiezen voor jaar en waarde, en de reeks trimmen.

    Standaard worden de eerste en tweede kolom geselecteerd.

    Args:
        df: DataFrame met de geladen data.

    Returns:
        Tuple van (jaar-array, waarde-array) als floats, getrimd op gekozen bereik.
    """
    cols = df.columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        jaar_col = st.selectbox(
            "Kolom voor jaar / x-as", options=cols, index=0
        )
    with col2:
        waarde_col = st.selectbox(
            "Kolom voor waarde / y-as",
            options=cols,
            index=min(1, len(cols) - 1),
        )

    df_sel = df[[jaar_col, waarde_col]].dropna()
    t_all = df_sel[jaar_col].values.astype(float)
    y_all = df_sel[waarde_col].values.astype(float)

    # df_sel = df[[jaar_col, waarde_col]].dropna()
    # t_all = to_year_float(df_sel[jaar_col])
    # y_all = df_sel[waarde_col].values.astype(float)

    jaar_min = int(t_all.min())
    jaar_max = int(t_all.max())
   
    trim_min, trim_max = st.slider(
        "Reeks trimmen (jaren)",
        min_value=jaar_min,
        max_value=jaar_max,
        value=(jaar_min, jaar_max),
        step=1,
    )
    st.caption(
        f"Geselecteerd: {trim_min}–{trim_max}  "
        f"({trim_max - trim_min + 1} jaar)"
    )

    mask = (t_all >= trim_min) & (t_all <= trim_max)
    return t_all[mask], y_all[mask]


def find_best_sma(
    loess_y: np.ndarray,
    y: np.ndarray,
    sma_range: range,
    min_overlap: int = 10,
    max_decline: int = 5,
) -> tuple[Optional[int], float]:
    """Zoek de SMA-venstergrootte met hoogste correlatie t.o.v. een LOESS-curve.

    Stopt vroegtijdig als de correlatie max_decline keer achter elkaar daalt.

    Args:
        loess_y: LOESS-trendwaarden.
        y: Originele jaarlijkse waarden.
        sma_range: Te doorzoeken SMA-venstergroottes.
        min_overlap: Minimum aantal niet-NaN punten voor een geldige correlatie.
        max_decline: Aantal opeenvolgende dalingen waarna gestopt wordt.

    Returns:
        Tuple van (beste SMA-venster, bijbehorende Pearson r).
        Venster is None als geen geldige match gevonden werd.
    """
    best_r: float = -1.0
    best_sma: Optional[int] = None
    prev_r: float = -1.0
    decline_count: int = 0

    for sma_w in sma_range:
        if sma_w >= len(y):
            continue
        sma_y = centered_sma(y, sma_w)
        mask = ~np.isnan(sma_y)
        if mask.sum() < min_overlap:
            continue
        r, _ = pearsonr(loess_y[mask], sma_y[mask])
        if r > best_r:
            best_r = r
            best_sma = sma_w

        if r < prev_r:
            decline_count += 1
        else:
            decline_count = 0
        prev_r = r

        if decline_count >= max_decline:
            break

    return best_sma, best_r


def find_best_loess(
    t: np.ndarray,
    y: np.ndarray,
    sma_y: np.ndarray,
    loess_range: range,
    min_overlap: int = 10,
    max_decline: int = 5,
) -> tuple[Optional[int], float]:
    """Zoek het LOESS-venster met hoogste correlatie t.o.v. een SMA-curve.

    Stopt vroegtijdig als de correlatie max_decline keer achter elkaar daalt.

    Args:
        t: Array met jaren.
        y: Array met jaarlijkse waarden.
        sma_y: Voorberekende SMA-waarden.
        loess_range: Te doorzoeken LOESS-venstergroottes.
        min_overlap: Minimum aantal niet-NaN punten voor een geldige correlatie.
        max_decline: Aantal opeenvolgende dalingen waarna gestopt wordt.

    Returns:
        Tuple van (beste LOESS-venster, bijbehorende Pearson r).
        Venster is None als geen geldige match gevonden werd.
    """
    mask = ~np.isnan(sma_y)
    if mask.sum() < min_overlap:
        return None, -1.0

    best_r: float = -1.0
    best_loess: Optional[int] = None
    prev_r: float = -1.0
    decline_count: int = 0

    for loess_w in loess_range:
        if loess_w >= len(t):
            continue
        try:
            loess_y, _, _ = loess_smooth(t, y, loess_w)
        except ValueError:
            continue
        r, _ = pearsonr(loess_y[mask], sma_y[mask])
        if r > best_r:
            best_r = r
            best_loess = loess_w

        if r < prev_r:
            decline_count += 1
        else:
            decline_count = 0
        prev_r = r

        if decline_count >= max_decline:
            break

    return best_loess, best_r


def tab_batch_analyse(t: np.ndarray, y: np.ndarray, max_decline: int) -> None:
    """Tab 1: batch-analyse van beste LOESS per SMA-venster.

    Voor elk SMA-venster in SMA_RANGE wordt het beste LOESS-venster gezocht
    op basis van Pearson-correlatie. Early-stopping binnen de LOESS-loop
    versnelt de berekening.

    Args:
        t: Array met jaren.
        y: Array met jaarlijkse waarden.
        max_decline: Aantal opeenvolgende dalingen waarna de LOESS-loop stopt.
    """
    st.subheader("Batch analyse: beste LOESS per SMA-venster")
    st.write(
        "Voor elk SMA-venster wordt gezocht welk LOESS-venster de hoogste "
        "correlatie geeft. Dit kan even duren."
    )

    if st.button("▶ Start batch analyse", type="primary"):
        results: list[tuple[int, int, float]] = []
        progress = st.progress(0, text="Bezig...")
        sma_list = [w for w in SMA_RANGE if w < len(t)]

        for i, sma_w in enumerate(sma_list):
            sma_y = centered_sma(y, sma_w)
            best_loess, best_r = find_best_loess(
                t, y, sma_y, LOESS_RANGE, max_decline=max_decline
            )
            if best_loess is not None:
                results.append((sma_w, best_loess, best_r))
            progress.progress(
                (i + 1) / len(sma_list),
                text=f"SMA venster {sma_w} jaar...",
            )

        progress.empty()

        res_df = pd.DataFrame(
            results, columns=["sma_venster", "beste_loess", "pearson_r"]
        )

        # lineaire fit: x = SMA, y = beste LOESS
        coeffs = np.polyfit(
            res_df["sma_venster"].values,
            res_df["beste_loess"].values,
            1,
        )
        st.success(
            f"Formule: beste LOESS ≈ **{coeffs[0]:.3f}** × SMA_venster "
            f"+ **{coeffs[1]:.3f}**"
        )

        fit_line = np.polyval(coeffs, res_df["sma_venster"].values)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=res_df["sma_venster"], y=res_df["beste_loess"],
            mode="markers+lines", name="Beste LOESS venster",
            line=dict(color="royalblue"),
            marker=dict(size=4),
        ))
        fig.add_trace(go.Scatter(
            x=res_df["sma_venster"], y=fit_line,
            mode="lines", name="Lineaire fit",
            line=dict(color="firebrick", dash="dash"),
        ))
        fig.update_layout(
            title="Beste LOESS-venster per SMA-venster",
            xaxis_title="SMA venster (jaren)",
            yaxis_title="Beste LOESS venster (jaren)",
            hovermode="x unified",
            height=400,
        )
        st.plotly_chart(fig, width="stretch")

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(
            x=res_df["sma_venster"], y=res_df["pearson_r"],
            mode="markers+lines", name="Pearson r",
            line=dict(color="green"),
            marker=dict(size=4),
        ))
        fig_r.update_layout(
            title="Kwaliteit van de beste match per SMA-venster",
            xaxis_title="SMA venster (jaren)",
            yaxis_title="Pearson r",
            yaxis=dict(range=[0.9, 1.01]),
            hovermode="x unified",
            height=300,
        )
        st.plotly_chart(fig_r, width="stretch")

        st.dataframe(
            res_df,
            hide_index=True,
            width="stretch",
            column_config={
                "sma_venster": st.column_config.NumberColumn("SMA venster"),
                "beste_loess": st.column_config.NumberColumn("Beste LOESS"),
                "pearson_r": st.column_config.NumberColumn(
                    "Pearson r", format="%.4f"
                ),
            },
        )
        res_df.to_csv("sma_vs_loess_resultaten.csv", index=False)
        st.caption("Opgeslagen als sma_vs_loess_resultaten.csv")


def tab_vergelijk(
    t: np.ndarray, y: np.ndarray, sma_window: int, span_noemer: int
) -> None:
    """Tab 2: interactieve vergelijking van SMA en LOESS met Plotly.

    De gebruiker kiest het SMA-venster en de LOESS span-noemer via sliders
    in de gedeelde sidebar. Toont grafiek met confidence interval,
    Pearson r en datatabel.

    Args:
        t: Array met jaren.
        y: Array met jaarlijkse waarden.
        sma_window: SMA-venstergrootte gekozen in de sidebar.
        span_noemer: LOESS span-noemer gekozen in de sidebar.
    """
    st.subheader("Interactieve vergelijking SMA vs LOESS")

    sma_y = centered_sma(y, sma_window)

    try:
        loess_y, loess_ll, loess_ul = loess_smooth(t, y, span_noemer)
    except ValueError as e:
        st.error(f"LOESS fout: {e}")
        return

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([loess_ul, loess_ll[::-1]]),
        fill="toself",
        fillcolor="rgba(178,34,34,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="LOESS 95% CI",
        showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=t, y=y,
        mode="lines",
        name="Ruwe data",
        line=dict(color="lightgray", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=t, y=sma_y,
        mode="lines",
        name=f"SMA ({sma_window} jaar)",
        line=dict(color="royalblue", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=t, y=loess_y,
        mode="lines",
        name=f"LOESS (noemer={span_noemer})",
        line=dict(color="firebrick", width=2, dash="dash"),
    ))
    fig.update_layout(
        title="SMA vs LOESS trendlijn",
        xaxis_title="Jaar",
        yaxis_title="Waarde",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        hovermode="x unified",
        height=500,
    )
    st.plotly_chart(fig, width="stretch")

    mask = ~np.isnan(sma_y)
    if mask.sum() >= 10:
        r, _ = pearsonr(loess_y[mask], sma_y[mask])
        c1, c2, c3 = st.columns(3)
        c1.metric("SMA venster", f"{sma_window} jaar")
        c2.metric("LOESS noemer", span_noemer)
        c3.metric("Pearson r", f"{r:.4f}")

    with st.expander("Bekijk data"):
        df_view = pd.DataFrame({
            "jaar": t.astype(int),
            "waarde": y,
            f"SMA ({sma_window}j)": sma_y,
            f"LOESS (n={span_noemer})": loess_y,
            "LOESS ll": loess_ll,
            "LOESS ul": loess_ul,
        })
        st.dataframe(
            df_view,
            hide_index=True,
            width="stretch",
            column_config={
                "waarde": st.column_config.NumberColumn(
                    "Waarde", format="%.2f"
                ),
                f"SMA ({sma_window}j)": st.column_config.NumberColumn(
                    format="%.2f"
                ),
                f"LOESS (n={span_noemer})": st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "LOESS ll": st.column_config.NumberColumn(
                    "LOESS ondergrens", format="%.2f"
                ),
                "LOESS ul": st.column_config.NumberColumn(
                    "LOESS bovengrens", format="%.2f"
                ),
            },
        )


def tab_loess_correlatie(
    t: np.ndarray, y: np.ndarray, sma_window: int
) -> None:
    """Tab 3: correlatie tussen LOESS-venster en Pearson r bij een gegeven SMA-venster.

    Voor elk LOESS-venster in LOESS_RANGE wordt de Pearson-correlatie berekend
    t.o.v. de SMA met het in de sidebar gekozen venster. Toont het optimale
    LOESS-venster als verticale markering.

    Args:
        t: Array met jaren.
        y: Array met jaarlijkse waarden.
        sma_window: SMA-venstergrootte gekozen in de sidebar.
    """
    st.subheader(
        f"Correlatie LOESS-venster vs Pearson r  (SMA = {sma_window} jaar)"
    )

    sma_y = centered_sma(y, sma_window)
    mask = ~np.isnan(sma_y)
    if mask.sum() < 10:
        st.error("Te weinig geldige SMA-punten voor correlatie.")
        return

    results: list[tuple[int, float]] = []

    with st.spinner("Berekenen..."):
        for loess_w in LOESS_RANGE:
            if loess_w >= len(t):
                continue
            try:
                loess_y, _, _ = loess_smooth(t, y, loess_w)
            except ValueError:
                continue
            r, _ = pearsonr(loess_y[mask], sma_y[mask])
            results.append((loess_w, r))

    if not results:
        st.error("Geen resultaten beschikbaar.")
        return

    res_df = pd.DataFrame(results, columns=["loess_venster", "pearson_r"])
    best = res_df.loc[res_df["pearson_r"].idxmax()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=res_df["loess_venster"],
        y=res_df["pearson_r"],
        mode="markers+lines",
        name="Pearson r",
        line=dict(color="royalblue", width=2),
        marker=dict(size=5),
    ))
    fig.add_vline(
        x=best["loess_venster"],
        line=dict(color="firebrick", dash="dash", width=1.5),
        annotation_text=(
            f"Max r = {best['pearson_r']:.4f}  "
            f"@ venster = {int(best['loess_venster'])} jaar"
        ),
        annotation_position="top right",
        annotation_font=dict(color="firebrick"),
    )
    fig.update_layout(
        title=f"Pearson r per LOESS-venster  (SMA = {sma_window} jaar)",
        xaxis_title="LOESS venster (jaren)",
        yaxis_title="Pearson r",
        hovermode="x unified",
        height=420,
    )
    st.plotly_chart(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("SMA venster", f"{sma_window} jaar")
    c2.metric("Beste LOESS venster", f"{int(best['loess_venster'])} jaar")
    c3.metric("Pearson r (max)", f"{best['pearson_r']:.4f}")

    with st.expander("Bekijk data"):
        st.dataframe(
            res_df,
            hide_index=True,
            width="stretch",
            column_config={
                "loess_venster": st.column_config.NumberColumn("LOESS venster"),
                "pearson_r": st.column_config.NumberColumn(
                    "Pearson r", format="%.4f"
                ),
            },
        )


def main() -> None:
    """Hoofdfunctie: laad data en toon de drie-tabben Streamlit app."""
    st.set_page_config(page_title="LOESS vs SMA", layout="wide")
    st.title("LOESS vs SMA analyse")

    with st.sidebar:
        st.header("Data")

    df = load_data_ui(sidebar_key="upload")
    if df is None:
        return

    with st.container(border=True):
        st.subheader("Kolommen kiezen")
        t, y = select_columns(df)

    if len(t) < 10:
        st.error("Te weinig datapunten (minimaal 10).")
        return

    # ── Gedeelde sidebar-instellingen ──────────────────────────────────────
    with st.sidebar:
        st.header("Instellingen")
        sma_window = st.slider(
            label="SMA venster (jaren)",
            min_value=3,
            max_value=min(100, len(t) - 1),
            value=30,
            step=1,
        )
        span_noemer = st.slider(
            label="LOESS span-noemer",
            min_value=3,
            max_value=min(100, len(t) - 1),
            value=42,
            step=1,
        )
        st.caption(f"Reekslengte n = {len(t)}")
        st.caption(
            f"LOESS span = {span_noemer}/{len(t)} = {span_noemer / len(t):.3f}"
        )
        st.caption(f"Effectief LOESS venster ≈ {span_noemer} jaar")

        st.divider()
        st.subheader("Batch analyse")
        max_decline = st.slider(
            label="Vroegtijdig stoppen na x dalingen",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help=(
                "Stop de batch analyse als de Pearson r dit aantal keren "
                "achter elkaar daalt. Zet op 20 om altijd door te lopen."
            ),
        )
    # ───────────────────────────────────────────────────────────────────────

    tab1, tab2, tab3 = st.tabs([
        "📊 Batch analyse",
        "📈 Correlatie per LOESS-venster",
         "🔍 Vergelijk SMA vs LOESS",
       
    ])

    with tab1:
        tab_batch_analyse(t, y, max_decline)


    with tab2:
        tab_loess_correlatie(t, y, sma_window)

    with tab3:
        tab_vergelijk(t, y, sma_window, span_noemer)


if __name__ == "__main__":
    main()