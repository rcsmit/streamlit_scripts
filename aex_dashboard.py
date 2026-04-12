"""AEX Meltdown – een 'Software Meltdown'-stijl dashboard voor AEX-fondsen.

Haalt live koersdata op via yfinance en toont prijs, dagrendement, YTD,
52-weeks high delta, en een sparkline per fonds.

Usage:
    streamlit run aex_meltdown.py

Requirements:
    pip install streamlit yfinance pandas
"""

from __future__ import annotations

import datetime
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuratie
# ---------------------------------------------------------------------------

AEX_TICKERS: dict[str, str] = {
    "ASML.AS": "ASML",
    "HEIA.AS": "Heineken",
    "ADYEN.AS": "Adyen",
    "INGA.AS": "ING Groep",
    "PHIA.AS": "Philips",
    "UNA.AS": "Unilever",
    "ABN.AS": "ABN AMRO",
    "AD.AS": "Ahold Delhaize",
    "AKZA.AS": "AkzoNobel",
    "ASM.AS": "ASM International",
    "ASRNL.AS": "ASR Nederland",
    "NN.AS": "NN Group",
    "RAND.AS": "Randstad",
    "RELX.AS": "RELX",
    "REN.AS": "IMCD",
    "MT.AS": "ArcelorMittal",
    "WKL.AS": "Wolters Kluwer",
    "BESI.AS": "BE Semiconductor",
    "KPN.AS": "KPN",
    "SHELL.AS": "Shell",
    "IMCD.AS": "IMCD",
    "PRX.AS": "Prosus",
    "DSMF.AS": "dsm-firmenich",
    "EXOR.AS": "EXOR",
    "AGN.AS": "Aegon",
}

# Verwijder dubbele namen (IMCD staat twee keer)
_seen: set[str] = set()
AEX_TICKERS_CLEAN: dict[str, str] = {}
for ticker, name in AEX_TICKERS.items():
    if name not in _seen:
        _seen.add(name)
        AEX_TICKERS_CLEAN[ticker] = name

YEAR_START = datetime.date(datetime.date.today().year, 1, 1)
ONE_YEAR_AGO = datetime.date.today() - datetime.timedelta(days=365)


# ---------------------------------------------------------------------------
# Data ophalen
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(tickers: list[str]) -> dict[str, Any]:
    """Haal prijsdata op voor de opgegeven tickers.

    Args:
        tickers: Lijst van Yahoo Finance ticker-symbolen.

    Returns:
        Dict met per ticker: naam, prijs, pct_1d, pct_ytd, delta_52w_high,
        sparkline_1y (lijst van slotkoersen).
    """
    result: dict[str, Any] = {}
    ticker_str = " ".join(tickers)

    # Bulk download voor sparklines (1 jaar dagelijks)
    raw = yf.download(
        ticker_str,
        start=ONE_YEAR_AGO.isoformat(),
        auto_adjust=True,
        progress=False,
    )

    close: pd.DataFrame
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]]
        close.columns = tickers[:1]

    for ticker in tickers:
        try:
            series = close[ticker].dropna()
            if len(series) < 2:
                continue

            price_today = float(series.iloc[-1])
            price_prev = float(series.iloc[-2])
            pct_1d = (price_today / price_prev - 1) * 100

            # YTD: koers op 1 jan vs. vandaag
            ytd_series = series[series.index >= pd.Timestamp(YEAR_START)]
            if len(ytd_series) >= 2:
                pct_ytd = (ytd_series.iloc[-1] / ytd_series.iloc[0] - 1) * 100
            else:
                pct_ytd = float("nan")

            # Delta t.o.v. 52w high
            high_52w = float(series.max())
            delta_52w = (price_today / high_52w - 1) * 100

            # Sparkline: laatste 52 wekelijkse slotkoersen (elke vrijdag)
            spark = series.resample("W").last().tail(52).tolist()

            result[ticker] = {
                "price": price_today,
                "pct_1d": pct_1d,
                "pct_ytd": pct_ytd,
                "delta_52w_high": delta_52w,
                "sparkline": spark,
            }
        except Exception:
            continue

    return result


# ---------------------------------------------------------------------------
# Hulpfuncties
# ---------------------------------------------------------------------------

def color_pct(val: float, inverse: bool = False) -> str:
    """Geef een rode of groene kleurcode terug op basis van het teken van val.

    Args:
        val: Percentagewaarde.
        inverse: Keer kleuren om (negatief = goed).

    Returns:
        Hex kleurstring.
    """
    positive_color = "#22c55e"
    negative_color = "#ef4444"
    if pd.isna(val):
        return "#6b7280"
    if inverse:
        return positive_color if val >= 0 else negative_color
    return positive_color if val >= 0 else negative_color


def fmt_pct(val: float) -> str:
    """Formatteer een getal als percentage met één decimaal.

    Args:
        val: Getal om te formatteren.

    Returns:
        Geformatteerde string, bijv. '+3.45%'.
    """
    if pd.isna(val):
        return "n/a"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}%"


def bar_html(pct: float, max_abs: float = 80) -> str:
    """Genereer een mini horizontale balk als HTML.

    Args:
        pct: Percentagewaarde (negatief = rode balk naar links).
        max_abs: Maximale absolute waarde voor schaalbreedte.

    Returns:
        HTML string met een gekleurde balk.
    """
    if pd.isna(pct):
        return ""
    width = min(abs(pct) / max_abs * 100, 100)
    color = "#ef4444" if pct < 0 else "#22c55e"
    return (
        f'<div style="background:{color};width:{width:.0f}%;'
        f'height:14px;border-radius:2px;display:inline-block;'
        f'min-width:4px;"></div>'
    )


# ---------------------------------------------------------------------------
# HTML tabel builder
# ---------------------------------------------------------------------------

def _spark_svg(values: list[float], width: int = 120, height: int = 28) -> str:
    """Genereer een inline SVG-sparkline.

    Args:
        values: Lijst van slotkoersen (chronologisch).
        width: Breedte van de SVG in pixels.
        height: Hoogte van de SVG in pixels.

    Returns:
        SVG-string.
    """
    if not values or len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    rng = hi - lo or 1
    n = len(values)
    pts = []
    for i, v in enumerate(values):
        x = i / (n - 1) * width
        y = height - (v - lo) / rng * (height - 4) - 2
        pts.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(pts)
    color = "#22c55e" if values[-1] >= values[0] else "#ef4444"
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="1.5"/>'
        f"</svg>"
    )


def _pct_cell(val: float, highlight: bool = True) -> str:
    """Genereer een <td>-cel met heatmap achtergrondkleur.

    Rood: licht roze bij kleine dalingen, diep rood bij grote dalingen.
    Groen: licht groen bij kleine stijgingen, diep groen bij grote stijgingen.
    Schaal: 0% = wit, 70%+ = maximale kleur (net als voorbeeld).

    Args:
        val: Percentagewaarde.
        highlight: Of de cel een gekleurde achtergrond krijgt.

    Returns:
        HTML <td> string.
    """
    if pd.isna(val):
        return '<td style="color:#999;text-align:right;padding:4px 10px;">n/a</td>'
    sign = "+" if val >= 0 else ""
    text = f"{sign}{val:.2f}%"
    if not highlight:
        color = "#16a34a" if val >= 0 else "#dc2626"
        return f'<td style="color:{color};text-align:right;padding:4px 10px;font-weight:600;">{text}</td>'
    # Heatmap: intensiteit 0..1 over bereik 0..70%
    intensity = min(abs(val) / 70.0, 1.0)
    if val < 0:
        # Wit (255,255,255) → diep rood (180,20,20)
        r = int(255 - (255 - 180) * intensity)
        g = int(255 - (255 - 20) * intensity)
        b = int(255 - (255 - 20) * intensity)
    else:
        # Wit (255,255,255) → diep groen (20,140,60)
        r = int(255 - (255 - 20) * intensity)
        g = int(255 - (255 - 140) * intensity)
        b = int(255 - (255 - 60) * intensity)
    bg = f"rgb({r},{g},{b})"
    # Tekstkleur: donker bij lichte achtergrond, wit bij donkere
    fg = "#111" if intensity < 0.55 else "#fff"
    return (
        f'<td style="background:{bg};color:{fg};text-align:right;'
        f'padding:4px 10px;font-weight:600;">{text}</td>'
    )


def _build_html_table(df: pd.DataFrame) -> str:
    """Bouw een HTML-tabel met gekleurde cellen.

    Args:
        df: DataFrame met kolommen Ticker, Naam, Prijs (€), % 1D, % YTD,
            Δ 52w High, _spark.

    Returns:
        HTML-string van de volledige tabel.
    """
    hs = (
        "background:#f0f0f0;color:#555;font-size:0.70rem;"
        "text-transform:uppercase;letter-spacing:0.06em;"
        "padding:6px 10px;text-align:right;border-bottom:1px solid #2d2d2d;"
    )
    hs_left = hs.replace("text-align:right", "text-align:left")
    headers = ["Ticker", "Naam", "Prijs (€)", "% 1D", "% YTD", "Δ 52w High", "Grafiek 1Y"]
    aligns = ["left", "left", "right", "right", "right", "right", "left"]
    head_html = "".join(
        f'<th style="{hs_left if a == "left" else hs}">{h}</th>'
        for h, a in zip(headers, aligns)
    )
    td_base = "padding:4px 10px;font-size:0.78rem;font-family:'IBM Plex Mono',monospace;"
    rows_html = ""
    for idx, row in df.reset_index(drop=True).iterrows():
        bg = "#ffffff" if idx % 2 == 0 else "#f9f9f9"
        spark = _spark_svg(row["_spark"])
        pct1d = row["% 1D"]
        pctytd = row["% YTD"]
        delta52 = row["Δ 52w High"]
        prijs = row["Prijs (€)"]
        rows_html += (
            f'<tr style="background:{bg};">'
            f'<td style="{td_base}text-align:left;color:#111;font-weight:600;">{row["Ticker"]}</td>'
            f'<td style="{td_base}text-align:left;color:#444;">{row["Naam"]}</td>'
            f'<td style="{td_base}text-align:right;color:#111;">€{prijs:.2f}</td>'
            + _pct_cell(pct1d, highlight=False)
            + _pct_cell(pctytd, highlight=True)
            + _pct_cell(delta52, highlight=True)
            + f'<td style="{td_base}text-align:left;">{spark}</td>'
            f"</tr>"
        )
    return f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap');
      body {{ margin:0; background:#ffffff; color:#111; }}
      .aex-table {{ width:100%;border-collapse:collapse; }}
      .aex-table tr:hover td {{ filter:brightness(1.25); transition:filter 0.1s; }}
    </style>
    <table class="aex-table">
      <thead><tr>{head_html}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    """Hoofdfunctie voor de Streamlit-app."""
    st.set_page_config(
        page_title="AEX Meltdown",
        page_icon="🇳🇱",
        layout="wide",
    )

    # Custom CSS
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
            background-color: #0f0f0f;
            color: #e5e5e5;
        }
        .block-container { padding-top: 2rem; }
        .stDataFrame { background: transparent; }
        thead tr th {
            background: #1a1a1a !important;
            color: #9ca3af !important;
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.72rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        tbody tr td {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.78rem !important;
        }
        h1 {
            font-family: 'IBM Plex Sans', sans-serif !important;
            font-weight: 700 !important;
            font-size: 2.4rem !important;
            letter-spacing: -0.02em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown(
        """
        <h1 style="text-align:center; color:#f5f5f5; margin-bottom:0.2rem;">
            🇳🇱 AEX Meltdown
        </h1>
        <p style="text-align:center; color:#6b7280; font-size:0.85rem; margin-top:0;">
            Live AEX-fondsen • yfinance • ververs elke 5 min
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Sorteerkeuze
    col_sort, col_filter, _ = st.columns([2, 2, 4])
    with col_sort:
        sort_by = st.selectbox(
            "Sorteren op",
            ["% YTD", "% 1D", "Δ 52w High", "Naam"],
            index=0,
        )
    with col_filter:
        filter_neg = st.checkbox("Alleen dalers", value=False)

    tickers = list(AEX_TICKERS_CLEAN.keys())

    with st.spinner("Koersen ophalen…"):
        data = fetch_data(tickers)

    if not data:
        st.error("Geen data beschikbaar. Controleer de verbinding.")
        return

    # Bouw DataFrame
    rows = []
    for ticker, info in data.items():
        name = AEX_TICKERS_CLEAN.get(ticker, ticker)
        rows.append(
            {
                "Ticker": ticker.replace(".AS", ""),
                "Naam": name,
                "Prijs (€)": round(info["price"], 2),
                "% 1D": round(info["pct_1d"], 2),
                "% YTD": round(info["pct_ytd"], 2),
                "Δ 52w High": round(info["delta_52w_high"], 2),
                "_spark": info["sparkline"],
            }
        )

    df = pd.DataFrame(rows)

    # Filter
    if filter_neg:
        df = df[df["% YTD"] < 0]

    # Sortering
    sort_map = {
        "% YTD": "% YTD",
        "% 1D": "% 1D",
        "Δ 52w High": "Δ 52w High",
        "Naam": "Naam",
    }
    df = df.sort_values(sort_map[sort_by], ascending=(sort_by == "Naam"))

    # Gemiddelde YTD
    avg_ytd = df["% YTD"].mean()

    # Toon tabel als gekleurde HTML
    tabel_html = _build_html_table(df)
    n_rows = len(df)
    components.html(tabel_html, height=44 * n_rows + 60, scrolling=False)

    # Footer
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between;
             color:#6b7280; font-size:0.75rem; margin-top:1rem;
             font-family:'IBM Plex Mono',monospace;">
            <span>Gem. YTD: <b style="color:#e5e5e5;">{fmt_pct(avg_ytd)}</b>
            &nbsp;|&nbsp; {len(df)} fondsen</span>
            <span>{datetime.date.today().strftime("%d %B %Y").lstrip("0")}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Data: Yahoo Finance via yfinance · Geen beleggingsadvies. Geinspireerd door https://x.com/shiri_shh/status/2042601476699361432")


if __name__ == "__main__":
    main()