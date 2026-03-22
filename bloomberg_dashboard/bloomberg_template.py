"""
bloomberg_template.py
─────────────────────
Herbruikbaar Streamlit-startpunt in Bloomberg Terminal-stijl.
Kopieer dit bestand, hernoem het, en vervang de voorbeelddata
door je eigen logica.

Bevat kant-en-klare componenten:
  • Volledige CSS (desktop + mobile responsive)
  • Header met badge en timestamp
  • Scrollende ticker
  • KPI-rij (grote cijfers bovenaan)
  • SVG ring-gauge
  • Panelen met metrics en interpretatie-labels
  • Horizontale balk-heatmap
  • Score-breakdown balk
  • Alert-banner
  • Streamlit-widget overrides (selectbox, button, slider)

Gebruik:
  pip install streamlit
  streamlit run bloomberg_template.py
"""

import streamlit as st
from datetime import datetime

from theme import apply_template_theme
# ── PAGINA-CONFIG ────────────────────────────────────────────────────
# Altijd als allereerste Streamlit-aanroep
try:
  st.set_page_config(
      page_title="Mijn Dashboard",   # ← aanpassen
      page_icon="📊",                 # ← aanpassen
      layout="wide",
      initial_sidebar_state="collapsed",
  )
except:
  pass

def css():
     # ════════════════════════════════════════════════════════════════════
    # CSS
    # ════════════════════════════════════════════════════════════════════
   
    st.markdown("""
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

    /* ── Design tokens ── */
    :root {
      /* Achtergronden */
      --bg:    #050a0f;   /* pagina-achtergrond      */
      --bg2:   #070d14;   /* iets lichter bg         */
      --pan:   #0a1520;   /* paneel-achtergrond       */
      --pan2:  #0d1c2a;   /* secundair paneel         */

      /* Randen */
      --bdr:   #0e2233;   /* subtiele rand            */
      --bdr2:  #1a3a52;   /* zichtbare rand           */

      /* Groen — positief / goed */
      --g:     #00ff9d;
      --gd:    #00cc7a;   /* dimmer groen             */
      --gk:    #003d25;   /* donker groen (achtergrond) */

      /* Rood — negatief / slecht */
      --r:     #ff3a3a;
      --rd:    #cc2222;
      --rk:    #3d0000;

      /* Amber — waarschuwing / neutraal */
      --a:     #ffb700;
      --ad:    #cc9200;
      --ak:    #3d2c00;

      /* Blauw — accentkleur / titels */
      --b:     #00b4ff;
      --bd:    #007ab5;

      /* Tekst */
      --t1:    #c8dce8;   /* primaire tekst           */
      --t2:    #6a8fa8;   /* secundaire tekst         */
      --t3:    #3a5a6e;   /* gedimde labels           */

      /* Typografie */
      --mono:  'Share Tech Mono', monospace;
      --sans:  'Rajdhani', sans-serif;
    }

    /* ── Reset & basisstijlen ── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    .stApp { background: var(--bg) !important; font-family: var(--mono) !important; color: var(--t1) !important; }

    /* ── Streamlit chrome verbergen ── */
    #MainMenu, footer, header { visibility: hidden; }
    .stApp > header { display: none; }
    section[data-testid="stSidebar"] { display: none; }

    /* ── Hoofdcontainer ── */
    .block-container { padding: 0 !important; max-width: 100% !important; }

    /* ── Animaties ── */
    @keyframes pulse  { 0%,100%{opacity:1} 50%{opacity:.5} }
    @keyframes scroll { 0%{transform:translateX(0)} 100%{transform:translateX(-50%)} }
    @keyframes glow   { 0%,100%{box-shadow:0 0 6px currentColor} 50%{box-shadow:0 0 18px currentColor} }

    /* ════════════════════════════════════════════════
      TICKER
      ════════════════════════════════════════════════ */
    .ticker-bar {
      background: #000;
      border-bottom: 1px solid var(--bdr2);
      padding: 5px 0;
      overflow: hidden;
      white-space: nowrap;
    }
    .ticker-scroll {
      display: inline-block;
      animation: scroll 80s linear infinite;
      font-size: 11px;
    }
    /* Ticker pauzeren bij hover (optioneel — verwijder als niet gewenst) */
    .ticker-bar:hover .ticker-scroll { animation-play-state: paused; }

    .ticker-item  { display: inline-block; margin-right: 28px; color: var(--t2); }
    .ticker-item .sym { color: var(--b); margin-right: 4px; }
    .up  { color: var(--g); }
    .dn  { color: var(--r); }
    .neu { color: var(--t2); }

    /* ════════════════════════════════════════════════
      HEADER
      ════════════════════════════════════════════════ */
    .dash-header {
      background: linear-gradient(90deg, #060e18, #0a1825);
      border-bottom: 1px solid var(--bdr2);
      padding: 10px 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 8px;
    }
    .dash-title {
      font-family: var(--sans);
      font-size: 20px;
      font-weight: 700;
      color: var(--b);
      letter-spacing: .12em;
      text-transform: uppercase;
    }
    .dash-subtitle { font-size: 9px; color: var(--t3); letter-spacing: .15em; margin-top: 2px; }
    .dash-meta     { font-size: 9px; color: var(--t3); text-align: right; }

    /* Badges (live/demo/status) */
    .badge {
      display: inline-block;
      padding: 2px 9px;
      font-size: 9px;
      letter-spacing: .2em;
      border-radius: 2px;
      font-family: var(--mono);
    }
    .badge-green  { background: var(--gk); color: var(--g);  border: 1px solid var(--gd); animation: pulse 2s infinite; }
    .badge-amber  { background: var(--ak); color: var(--a);  border: 1px solid var(--ad); }
    .badge-red    { background: var(--rk); color: var(--r);  border: 1px solid var(--rd); }
    .badge-blue   { background: #001d30;   color: var(--b);  border: 1px solid var(--bd); }
    .badge-gray   { background: var(--pan2); color: var(--t2); border: 1px solid var(--bdr2); }

    /* ════════════════════════════════════════════════
      BODY WRAPPER
      ════════════════════════════════════════════════ */
    .dash-body { padding: 12px 16px; }

    /* ════════════════════════════════════════════════
      ALERT BANNER
      ════════════════════════════════════════════════ */
    .alert-banner {
      background: var(--ak);
      border: 1px solid var(--ad);
      padding: 7px 14px;
      margin-bottom: 10px;
      font-size: 11px;
      color: var(--a);
      letter-spacing: .07em;
      animation: pulse 1.8s infinite;
    }
    .alert-banner.alert-red   { background: var(--rk); border-color: var(--rd); color: var(--r); }
    .alert-banner.alert-green { background: var(--gk); border-color: var(--gd); color: var(--g); animation: none; }

    /* ════════════════════════════════════════════════
      KPI-RIJ (grote cijfers)
      ════════════════════════════════════════════════ */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 9px;
      margin-bottom: 12px;
    }
    .kpi-card {
      background: var(--pan);
      border: 1px solid var(--bdr2);
      padding: 14px 16px;
      position: relative;
      overflow: hidden;
    }
    /* Gekleurde toprand per status */
    .kpi-card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 2px;
    }
    .kpi-green::before { background: var(--g); box-shadow: 0 0 8px var(--g); }
    .kpi-amber::before { background: var(--a); box-shadow: 0 0 8px var(--a); }
    .kpi-red::before   { background: var(--r); box-shadow: 0 0 8px var(--r); }
    .kpi-blue::before  { background: var(--b); box-shadow: 0 0 8px var(--b); }

    .kpi-label { font-size: 9px; letter-spacing: .2em; color: var(--t3); margin-bottom: 6px; text-transform: uppercase; }
    .kpi-value { font-family: var(--sans); font-size: 32px; font-weight: 700; line-height: 1; margin-bottom: 4px; }
    .kpi-green .kpi-value { color: var(--g); }
    .kpi-amber .kpi-value { color: var(--a); }
    .kpi-red   .kpi-value { color: var(--r); }
    .kpi-blue  .kpi-value { color: var(--b); }
    .kpi-sub   { font-size: 9px; color: var(--t2); line-height: 1.5; }

    /* ════════════════════════════════════════════════
      RING GAUGE (SVG cirkeldiagram)
      ════════════════════════════════════════════════ */
    .ring-wrap  {
      position: relative;
      width: 106px;
      height: 106px;
      margin: 0 auto 8px;
    }
    .ring-svg   { transform: rotate(-90deg); }
    .ring-num   {
      position: absolute;
      top: 50%; left: 50%;
      transform: translate(-50%, -50%);
      font-size: 22px;
      font-weight: bold;
      font-family: var(--mono);
    }
    .ring-card  {
      background: var(--pan);
      border: 1px solid var(--bdr2);
      padding: 16px;
      text-align: center;
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    .ring-title { font-size: 9px; letter-spacing: .2em; color: var(--t3); margin-bottom: 8px; text-transform: uppercase; }
    .ring-sub   { font-size: 9px; color: var(--t2); }

    /* ════════════════════════════════════════════════
      PANELEN (metriekenkaartjes)
      ════════════════════════════════════════════════ */
    .panel-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 9px;
      margin-bottom: 10px;
    }
    .panel {
      background: var(--pan);
      border: 1px solid var(--bdr);
      padding: 11px;
      transition: border-color .15s;
    }
    .panel:hover { border-color: var(--bdr2); }

    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 7px;
      padding-bottom: 5px;
      border-bottom: 1px solid var(--bdr);
    }
    .panel-title  { font-size: 9px; letter-spacing: .2em; color: var(--t3); text-transform: uppercase; }

    /* Score badge in paneel-hoek */
    .score-badge  { font-size: 10px; padding: 1px 5px; border-radius: 2px; }
    .sb-green { background: var(--gk); color: var(--g); border: 1px solid var(--gd); }
    .sb-amber { background: var(--ak); color: var(--a); border: 1px solid var(--ad); }
    .sb-red   { background: var(--rk); color: var(--r); border: 1px solid var(--rd); }

    /* Metriekregen */
    .metric-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 2px 0;
      border-bottom: 1px solid #0a1820;
      font-size: 10px;
    }
    .metric-row:last-of-type { border-bottom: none; }
    .metric-key   { color: var(--t2); font-size: 9px; }
    .metric-val   { color: var(--t1); }
    .metric-up    { color: var(--g); }
    .metric-dn    { color: var(--r); }
    .metric-neu   { color: var(--t2); }

    /* Interpretatie-label onderaan paneel */
    .interp {
      margin-top: 6px;
      font-size: 9px;
      padding: 3px 6px;
      border-left: 2px solid;
      letter-spacing: .04em;
      line-height: 1.4;
    }
    .interp-green  { border-color: var(--g); background: var(--gk); color: var(--gd); }
    .interp-amber  { border-color: var(--a); background: var(--ak); color: var(--ad); }
    .interp-red    { border-color: var(--r); background: var(--rk); color: var(--rd); }

    /* ════════════════════════════════════════════════
      SECTIELABEL (boven een blok)
      ════════════════════════════════════════════════ */
    .section-label {
      font-size: 9px;
      letter-spacing: .25em;
      color: var(--b);
      border-bottom: 1px solid var(--bdr2);
      padding-bottom: 6px;
      margin-bottom: 9px;
      text-transform: uppercase;
    }

    /* ════════════════════════════════════════════════
      HORIZONTALE BALK-HEATMAP
      ════════════════════════════════════════════════ */
    .heatmap-row  { display: flex; align-items: center; margin-bottom: 4px; gap: 7px; }
    .hm-label     { color: var(--t2); width: 40px; flex-shrink: 0; font-size: 9.5px; }
    .hm-track     { flex: 1; height: 11px; background: var(--bg2); position: relative; border: 1px solid var(--bdr); overflow: hidden; }
    .hm-center    { position: absolute; top: 0; bottom: 0; left: 50%; width: 1px; background: var(--bdr2); }
    .hm-value     { color: var(--t1); width: 52px; text-align: right; flex-shrink: 0; font-size: 9.5px; }

    /* ════════════════════════════════════════════════
      SCORE-BREAKDOWN BALK
      ════════════════════════════════════════════════ */
    .breakdown-row   { display: flex; align-items: center; margin-bottom: 6px; gap: 6px; }
    .bd-label        { color: var(--t2); width: 74px; font-size: 9.5px; flex-shrink: 0; }
    .bd-weight       { color: var(--t3); width: 26px; font-size: 9px; flex-shrink: 0; }
    .bd-track        { flex: 1; height: 8px; background: var(--bg2); position: relative; border: 1px solid var(--bdr); }
    .bd-fill         { position: absolute; top: 0; left: 0; height: 100%; }
    .bd-score        { color: var(--t1); width: 26px; text-align: right; font-size: 9.5px; flex-shrink: 0; }

    /* ════════════════════════════════════════════════
      GENERIEKE KAART / BLOK
      ════════════════════════════════════════════════ */
    .card {
      background: var(--pan);
      border: 1px solid var(--bdr);
      padding: 12px;
    }
    .card-title {
      font-size: 9px;
      letter-spacing: .2em;
      color: var(--b);
      margin-bottom: 8px;
      padding-bottom: 5px;
      border-bottom: 1px solid var(--bdr);
      text-transform: uppercase;
    }

    /* ════════════════════════════════════════════════
      DIVIDER
      ════════════════════════════════════════════════ */
    .divider { border: none; border-top: 1px solid var(--bdr2); margin: 14px 0; }

    /* ════════════════════════════════════════════════
      STREAMLIT WIDGET OVERRIDES
      ════════════════════════════════════════════════ */
    /* Kolom-spacing */
    div[data-testid="stHorizontalBlock"] { gap: 0 !important; }
    div[data-testid="column"]            { padding: 0 4px !important; }

    /* Selectbox */
    .stSelectbox > div > div {
      background:   var(--pan)  !important;
      border:       1px solid var(--bdr2) !important;
      color:        var(--t1)   !important;
      font-family:  var(--mono) !important;
      font-size:    10.5px      !important;
      border-radius: 0          !important;
    }
    .stSelectbox > div > div:focus-within { border-color: var(--b) !important; }

    /* Button */
    .stButton > button {
      background:    var(--pan)  !important;
      color:         var(--b)    !important;
      border:        1px solid var(--bd) !important;
      font-family:   var(--mono) !important;
      font-size:     10px        !important;
      letter-spacing:.15em       !important;
      padding:       6px 14px    !important;
      border-radius: 0           !important;
      text-transform:uppercase   !important;
      width:         100%        !important;
      transition:    background .15s, color .15s;
    }
    .stButton > button:hover {
      background: var(--bd) !important;
      color: #fff !important;
    }

    /* Slider */
    .stSlider > div > div > div { background: var(--bdr2) !important; }
    .stSlider > div > div > div > div { background: var(--b) !important; }

    /* Number input */
    .stNumberInput > div > div > input {
      background:   var(--pan)  !important;
      border:       1px solid var(--bdr2) !important;
      color:        var(--t1)   !important;
      font-family:  var(--mono) !important;
      font-size:    11px        !important;
      border-radius: 0          !important;
    }

    /* Text input */
    .stTextInput > div > div > input {
      background:   var(--pan)  !important;
      border:       1px solid var(--bdr2) !important;
      color:        var(--t1)   !important;
      font-family:  var(--mono) !important;
      font-size:    11px        !important;
      border-radius: 0          !important;
    }

    /* Date input */
    .stDateInput > div > div > input {
      background:   var(--pan)  !important;
      border:       1px solid var(--bdr2) !important;
      color:        var(--t1)   !important;
      font-family:  var(--mono) !important;
      font-size:    11px        !important;
      border-radius: 0          !important;
    }

    /* DataFrame / tabel */
    div[data-testid="stDataFrame"] { border: 1px solid var(--bdr2) !important; }

    /* Spinner */
    .stSpinner > div { border-top-color: var(--b) !important; }

    /* Metric widget */
    div[data-testid="stMetric"] {
      background:    var(--pan)  !important;
      border:        1px solid var(--bdr2) !important;
      padding:       12px        !important;
      border-radius: 0           !important;
    }
    div[data-testid="stMetricLabel"]  { color: var(--t3) !important; font-size: 9px !important; letter-spacing: .15em !important; }
    div[data-testid="stMetricValue"]  { color: var(--t1) !important; font-family: var(--sans) !important; }
    div[data-testid="stMetricDelta"]  { font-size: 10px !important; }

    /* ════════════════════════════════════════════════
      MOBILE — responsive aanpassingen
      ════════════════════════════════════════════════ */
    @media (max-width: 768px) {
      /* Meer ademruimte op small screens */
      .dash-body { padding: 8px 10px; }

      /* Header: stapel verticaal */
      .dash-header { flex-direction: column; align-items: flex-start; gap: 6px; }
      .dash-title  { font-size: 16px; }
      .dash-meta   { text-align: left; }

      /* KPI-rij: 2 kolommen op mobile */
      .kpi-grid { grid-template-columns: repeat(2, 1fr); gap: 6px; }
      .kpi-value { font-size: 24px; }

      /* Paneel-grid: 1 kolom op mobile */
      .panel-grid { grid-template-columns: 1fr; gap: 6px; }

      /* Ring-gauges kleiner */
      .ring-wrap { width: 80px; height: 80px; }
      .ring-num  { font-size: 17px; }

      /* Ticker iets groter voor leesbaarheid */
      .ticker-scroll { font-size: 12px; }

      /* Streep body-padding van Streamlit-kolommen */
      div[data-testid="column"] { padding: 0 2px !important; }

      /* Buttons groter aanraakgebied */
      .stButton > button { padding: 10px 14px !important; font-size: 11px !important; }

      /* Heatmap labels smaller */
      .hm-label { width: 32px; font-size: 9px; }
      .hm-value { width: 44px; font-size: 9px; }
    }

    @media (max-width: 480px) {
      /* Telefoon: KPI ook 2 kolommen, value kleiner */
      .kpi-grid  { grid-template-columns: repeat(2, 1fr); }
      .kpi-value { font-size: 20px; }

      /* Ticker stopt met scrollen op heel kleine schermen (te snel) */
      .ticker-scroll { animation-duration: 120s; }
    }
    </style>
    """, unsafe_allow_html=True)


def bloomberg_template():
    apply_template_theme()

    
    # ════════════════════════════════════════════════════════════════════
    # HELPER-FUNCTIES
    # Kopieer deze naar je eigen app of importeer ze als module.
    # ════════════════════════════════════════════════════════════════════

    def fmt(v, dec=2, prefix="", suffix=""):
        """Formatteer een getal; geeft 'N/A' als None."""
        if v is None:
            return "N/A"
        return f"{prefix}{v:.{dec}f}{suffix}"

    def score_color(s: float) -> str:
        """Geeft een hex-kleur op basis van een score 0–100."""
        if s >= 70: return "#00ff9d"
        if s >= 50: return "#ffb700"
        return "#ff3a3a"

    def score_badge_cls(s: float) -> str:
        """CSS-klasse voor score-badge in paneel-hoek."""
        if s >= 70: return "sb-green"
        if s >= 50: return "sb-amber"
        return "sb-red"

    def interp_cls(positive_keywords, negative_keywords, text: str) -> str:
        """
        Bepaal de CSS-klasse voor een interpretatie-label.
        Geef lijsten van trefwoorden die 'goed' of 'slecht' betekenen.
        """
        t = text.lower()
        if any(k in t for k in positive_keywords): return "interp-green"
        if any(k in t for k in negative_keywords): return "interp-red"
        return "interp-amber"

    def ring_gauge(score: float, color: str, title: str, subtitle: str = "") -> str:
        """
        Genereert HTML voor een SVG ring-gauge.

        Parameters
        ----------
        score    : 0–100
        color    : hex-kleur van de boog
        title    : label boven de ring
        subtitle : label onder de ring
        """
        R  = 47
        C  = 2 * 3.14159 * R
        fl = C * max(0, min(100, score)) / 100
        gp = C - fl
        return f"""
    <div class="ring-card">
      <div class="ring-title">{title}</div>
      <div class="ring-wrap">
        <svg class="ring-svg" viewBox="0 0 108 108" width="106" height="106">
          <circle cx="54" cy="54" r="{R}" fill="none" stroke="var(--bdr)" stroke-width="8"/>
          <circle cx="54" cy="54" r="{R}" fill="none" stroke="{color}" stroke-width="8"
            stroke-linecap="round" stroke-dasharray="{fl:.1f} {gp:.1f}"/>
        </svg>
        <div class="ring-num" style="color:{color}">{score:.0f}</div>
      </div>
      <div class="ring-sub">{subtitle}</div>
    </div>"""

    def kpi_card(label: str, value: str, sub: str = "", color: str = "blue") -> str:
        """
        Genereert HTML voor een KPI-kaart.

        Parameters
        ----------
        label  : kleine label bovenaan
        value  : grote waarde
        sub    : kleine tekst eronder
        color  : 'green' | 'amber' | 'red' | 'blue'
        """
        return f"""
    <div class="kpi-card kpi-{color}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>"""

    def heatmap_row(label: str, value: float, max_abs: float,
                    pos_color: str = "linear-gradient(90deg,#005533,#00ff9d)",
                    neg_color: str = "linear-gradient(270deg,#5a0000,#ff3a3a)",
                    suffix: str = "%") -> str:
        """Genereert één rij van een horizontale balk-heatmap (gecentreerd op nul)."""
        pct   = min(abs(value) / max(max_abs, 0.01) * 47, 47)
        vc    = "metric-up" if value > 0 else ("metric-dn" if value < 0 else "metric-neu")
        sign  = "+" if value > 0 else ""
        if value >= 0:
            fill = f"position:absolute;top:0;bottom:0;left:50%;width:{pct:.1f}%;background:{pos_color}"
        else:
            fill = f"position:absolute;top:0;bottom:0;right:50%;width:{pct:.1f}%;background:{neg_color}"
        return f"""
    <div class="heatmap-row">
      <span class="hm-label">{label}</span>
      <div class="hm-track">
        <div class="hm-center"></div>
        <div style="{fill}"></div>
      </div>
      <span class="hm-value {vc}">{sign}{value:.2f}{suffix}</span>
    </div>"""

    def breakdown_bar(label: str, weight: float, score: float) -> str:
        """Genereert één rij van de score-breakdown."""
        color = score_color(score)
        return f"""
    <div class="breakdown-row">
      <span class="bd-label">{label}</span>
      <span class="bd-weight">{weight*100:.0f}%</span>
      <div class="bd-track">
        <div class="bd-fill" style="width:{score}%;background:{color};opacity:.75"></div>
      </div>
      <span class="bd-score" style="color:{color}">{score:.0f}</span>
    </div>"""

    def ticker_item(symbol: str, price=None, change=None, invert: bool = False) -> str:
        """
        Genereert één ticker-item.

        Parameters
        ----------
        symbol  : bijv. 'SPY'
        price   : huidige prijs (optioneel)
        change  : verandering als float; positief = stijging
        invert  : als True, is positieve change slecht (bijv. VIX)
        """
        if change is None:
            cls, arrow = "neu", "—"
        else:
            up = change > 0
            if invert:
                up = not up
            cls   = "up" if up else "dn"
            arrow = "▲" if change > 0 else "▼"
        price_str  = f" {price:.2f}" if price is not None else ""
        change_str = f" {change:+.2f}%" if change is not None else ""
        return (f'<span class="ticker-item">'
                f'<span class="sym">{symbol}</span>{price_str}'
                f'<span class="{cls}"> {arrow}{change_str}</span></span>')


    # ════════════════════════════════════════════════════════════════════
    # DEMO-INHOUD
    # Alles hieronder is voorbeeldcode — vervang door jouw eigen data.
    # ════════════════════════════════════════════════════════════════════

    now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    # ── Fictieve demo-data ────────────────────────────────────────────
    demo_items = [
        ("CAT A", 142.50, +1.23),
        ("CAT B", 89.10, -0.87),
        ("CAT C", 310.00, +0.05),
        ("IDX X", 4812.3, +0.61),
        ("IDX Y", 16340.0, -1.12),
        ("RATE",  4.28, +0.03),
    ]
    demo_kpis = [
        ("Score",     "74",  "van 100",  "blue"),
        ("Trend",     "+2.1%","5 daags",  "green"),
        ("Volatility","22.4", "hoog",     "amber"),
        ("Signaal",   "CAUTION","halve positie","amber"),
    ]
    demo_panels = [
        {
            "title": "⚡ Factor A",
            "score": 68,
            "metrics": [
                ("Waarde 1",  "142.5",  "metric-up"),
                ("Waarde 2",  "−0.82",  "metric-dn"),
                ("Waarde 3",  "56.3%",  "metric-val"),
                ("Waarde 4",  "1.04",   "metric-val"),
            ],
            "interp": "STABIEL — NORMALE OMSTANDIGHEDEN",
            "iclass": "interp-amber",
        },
        {
            "title": "📈 Factor B",
            "score": 82,
            "metrics": [
                ("Trend",     "+1.9%",  "metric-up"),
                ("vs MA50",   "+2.3%",  "metric-up"),
                ("vs MA200",  "+5.1%",  "metric-up"),
                ("RSI",       "62.4",   "metric-val"),
            ],
            "interp": "BULLISH — TREND INTACT",
            "iclass": "interp-green",
        },
        {
            "title": "🌊 Factor C",
            "score": 45,
            "metrics": [
                ("Positief",  "5/11",   "metric-val"),
                ("Ratio",     "1.04",   "metric-val"),
                ("Oscillator","-12.3",  "metric-dn"),
                ("Schatting", "48%",    "metric-val"),
            ],
            "interp": "MIXED — SELECTIEF BLIJVEN",
            "iclass": "interp-amber",
        },
        {
            "title": "🚀 Factor D",
            "score": 55,
            "metrics": [
                ("5d Chg",    "+0.8%",  "metric-up"),
                ("Spreiding", "4.2%",   "metric-val"),
                ("Leaders",   "3",      "metric-val"),
                ("Kracht",    "62",     "metric-up"),
            ],
            "interp": "MATIG — BEPERKTE ENERGIE",
            "iclass": "interp-amber",
        },
        {
            "title": "🏦 Factor E",
            "score": 38,
            "metrics": [
                ("Rente",     "4.28%",  "metric-dn"),
                ("Slope 5d",  "+0.022", "metric-dn"),
                ("Dollar",    "103.8",  "metric-val"),
                ("Stance",    "NEUTRAL","metric-val"),
            ],
            "interp": "HAWKISH — TEGENWIND",
            "iclass": "interp-red",
        },
    ]
    demo_heatmap = [
        ("Cat 1",  +3.2),
        ("Cat 2",  +1.8),
        ("Cat 3",  +0.9),
        ("Cat 4",  -0.4),
        ("Cat 5",  -1.7),
        ("Cat 6",  -2.9),
    ]
    demo_breakdown = [
        ("Factor A", .25, 68),
        ("Factor B", .20, 82),
        ("Factor C", .20, 45),
        ("Factor D", .25, 55),
        ("Factor E", .10, 38),
    ]
    demo_total = sum(s * w for _, w, s in demo_breakdown)


    # ════════════════════════════════════════════════════════════════════
    # STREAMLIT CONTROLS  (bovenaan, vóór de HTML-blokken)
    # ════════════════════════════════════════════════════════════════════
    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 7])
    with ctrl1:
        mode = st.selectbox(
            "m", ["modus A", "modus B"],
            label_visibility="collapsed",
            format_func=lambda x: f"◈ {x.upper()}"
        )
    with ctrl2:
        if st.button("⟳ REFRESH", key="uitleg"):
            st.cache_data.clear()
            st.rerun()


    # ════════════════════════════════════════════════════════════════════
    # RENDER
    # ════════════════════════════════════════════════════════════════════

    # ── 1. Ticker ────────────────────────────────────────────────────────
    ticker_html = "".join(ticker_item(s, p, c) for s, p, c in demo_items)
    st.markdown(f"""
    <div class="ticker-bar">
      <div class="ticker-scroll">{ticker_html * 3}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── 2. Header ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="dash-header">
      <div>
        <div class="dash-title">◈ Mijn Dashboard</div>
        <div class="dash-subtitle">OMSCHRIJVING VAN HET DASHBOARD &nbsp;·&nbsp; {mode.upper()}</div>
      </div>
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
        <span class="badge badge-green">● LIVE</span>
        <span class="dash-meta">UPDATED {now}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 3. Body wrapper open ──────────────────────────────────────────────
    st.markdown('<div class="dash-body">', unsafe_allow_html=True)

    # ── 4. Alert banner (optioneel — verwijder als niet nodig) ────────────
    st.markdown("""
    <div class="alert-banner">
      ⚠ VOORBEELD ALERTBANNER — vervang door eigen conditie of verwijder dit blok
    </div>
    """, unsafe_allow_html=True)

    # ── 5. KPI-rij ────────────────────────────────────────────────────────
    kpi_html = "".join(kpi_card(lbl, val, sub, col) for lbl, val, sub, col in demo_kpis)
    st.markdown(f"""
    <div class="section-label">◈ Kerncijfers</div>
    <div class="kpi-grid">{kpi_html}</div>
    """, unsafe_allow_html=True)

    # ── 6. Ring gauges + context (hero) ──────────────────────────────────
    ring1 = ring_gauge(demo_total, score_color(demo_total), "TOTAALSCORE", "Gewogen gemiddelde")
    ring2 = ring_gauge(62,         "#ffb700",               "UITVOERINGSVENSTER", "Setupkwaliteit")

    st.markdown(f"""
    <div class="section-label">◈ Scorenoverzicht</div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px">
      {ring1}
      {ring2}
      <div class="ring-card" style="align-items:flex-start;text-align:left;flex:1;min-width:160px">
        <div class="ring-title">CONTEXT</div>
        <div style="font-family:var(--sans);font-size:28px;font-weight:700;color:#ffb700;margin-bottom:10px">CAUTION</div>
        <div style="font-size:10px;color:var(--t2);line-height:2.1;font-family:var(--mono)">
          Modus: <span style="color:var(--t1)">{mode.upper()}</span><br/>
          Extra veld 1: <span style="color:var(--t1)">waarde</span><br/>
          Extra veld 2: <span style="color:var(--g)">POSITIEF</span><br/>
          Extra veld 3: <span style="color:var(--r)">NEGATIEF</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 7. Vijf detailpanelen ─────────────────────────────────────────────
    panels_html = ""
    for p in demo_panels:
        badge = f'<span class="score-badge {score_badge_cls(p["score"])}">{p["score"]}</span>'
        rows  = "".join(
            f'<div class="metric-row"><span class="metric-key">{k}</span>'
            f'<span class="metric-val {cls}">{v}</span></div>'
            for k, v, cls in p["metrics"]
        )
        panels_html += f"""
    <div class="panel">
      <div class="panel-header">
        <span class="panel-title">{p["title"]}</span>{badge}
      </div>
      {rows}
      <div class="interp {p['iclass']}">{p['interp']}</div>
    </div>"""

    st.markdown(f"""
    <div class="section-label">◈ Detailpanelen</div>
    <div class="panel-grid">{panels_html}</div>
    """, unsafe_allow_html=True)

    # ── 8. Onderste rij: heatmap + breakdown + terminal ───────────────────
    max_abs = max(abs(v) for _, v in demo_heatmap)
    hm_html = "".join(heatmap_row(lbl, val, max_abs) for lbl, val in demo_heatmap)

    bd_html = "".join(breakdown_bar(lbl, w, s) for lbl, w, s in demo_breakdown)
    bd_html += f"""
    <div style="border-top:1px solid var(--bdr2);margin-top:8px;padding-top:7px">
      <div class="breakdown-row">
        <span class="bd-label" style="color:var(--t1)">TOTAAL</span>
        <span class="bd-weight"></span>
        <div class="bd-track">
          <div class="bd-fill" style="width:{demo_total:.0f}%;background:{score_color(demo_total)}"></div>
        </div>
        <span class="bd-score" style="color:{score_color(demo_total)};font-size:13px">{demo_total:.0f}</span>
      </div>
    </div>"""

    terminal_lines = [
        "Initialisatie succesvol. Alle datafactors geladen.",
        "Factor B sterk bullish — trend intact boven alle gemiddelden.",
        "Factor C zwak — beperkte participatiebreedte, selectief blijven.",
        "Factor E toont tegenwind door stijgende rente.",
        "Conclusie: CAUTION. Halve positiegrootte, alleen sterkste setups.",
    ]
    term_html = "".join(
        f'<div style="margin-bottom:3px"><span style="color:var(--t3)">▸▸▸</span> '
        f'<span style="color:var(--gd)">{l}</span></div>'
        for l in terminal_lines
    )

    st.markdown(f"""
    <div class="section-label">◈ Analyse</div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1.1fr;gap:9px">

      <div class="card">
        <div class="card-title">◈ Heatmap — relatieve verandering</div>
        {hm_html}
      </div>

      <div class="card">
        <div class="card-title">◈ Score breakdown</div>
        {bd_html}
        <div style="margin-top:10px;font-size:9px;color:var(--t3);line-height:1.8;
                    border-top:1px solid var(--bdr);padding-top:6px">
          Drempels: GOED ≥ 70 &nbsp;|&nbsp; MATIG ≥ 50 &nbsp;|&nbsp; SLECHT &lt; 50
        </div>
      </div>

      <div class="card">
        <div class="card-title">◈ Terminal output — {now}</div>
        <div style="font-family:var(--mono);font-size:10px;line-height:1.9">{term_html}</div>
        <div style="margin-top:8px;border-top:1px solid var(--bdr);padding-top:6px;
                    font-size:9px;color:var(--t3)">
          ⚠ UITSLUITEND VOOR EDUCATIEF GEBRUIK — GEEN FINANCIEEL ADVIES
        </div>
      </div>

    </div>
    """, unsafe_allow_html=True)

    # ── 9. Body wrapper sluiten ──────────────────────────────────────────
    st.markdown('</div>', unsafe_allow_html=True)

    # ── 10. Optioneel: Streamlit-native widgets (Plotly, dataframe, etc.) ─
    # st.plotly_chart(fig, use_container_width=True)
    # st.dataframe(df, use_container_width=True)
    # Zet ze ná de HTML-blokken — Streamlit en raw HTML mengen goed
    # zolang de HTML-blokken gesloten zijn.

    # ── 11. Optioneel: auto-refresh (verwijder als niet nodig) ────────────
    # st.markdown('<script>setTimeout(()=>window.location.reload(),45000)</script>',
    #             unsafe_allow_html=True)

def main():
    bloomberg_template()


if __name__ == "__main__":
    main()
