"""
theme.py — Centraal thema voor alle Bloomberg-dashboard scripts.

Gebruik in elk script:
    from theme import apply_theme, apply_article_theme

apply_theme()         → dashboard-stijl (app.py, backtest.py)
apply_article_theme() → artikel-stijl (uitleg.py, uitleg_app.py)

Pagina-specifieke CSS (bijv. block-container breedte) voeg je daarna
toe met een tweede st.markdown() in het script zelf.
"""

import streamlit as st

# ════════════════════════════════════════════════════════════════════
# DESIGN TOKENS — pas hier aan, werkt overal door
# ════════════════════════════════════════════════════════════════════
TOKENS_BLOOMBERG = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Crimson+Pro:ital,wght@0,400;0,600;1,400&display=swap');
:root {
  /* Achtergronden */
  --bg:   #050a0f;
  --bg2:  #070d14;
  --pan:  #0a1520;
  --pan2: #0d1c2a;

  /* Randen */
  --bdr:  #0e2233;
  --bdr2: #1a3a52;

  /* Groen */
  --g:  #00ff9d;
  --gd: #00cc7a;
  --gk: #003d25;

  /* Rood */
  --r:  #ff3a3a;
  --rd: #cc2222;
  --rk: #3d0000;

  /* Amber */
  --a:  #ffb700;
  --ad: #cc9200;
  --ak: #3d2c00;

  /* Blauw */
  --b:  #00b4ff;
  --bd: #007ab5;

  /* Tekst */
  --t1: #c8dce8;
  --t2: #6a8fa8;
  --t3: #3a5a6e;

  /* Fonts */
  --mono:  'Share Tech Mono', monospace;
  --sans:  'Rajdhani', sans-serif;
  --serif: 'Crimson Pro', Georgia, serif;
}
"""

TOKENS_LICHT = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Crimson+Pro:ital,wght@0,400;0,600;1,400&display=swap');
:root {
  /* Achtergronden */
  --bg:   #f5f0e8;
  --bg2:  #ede8df;
  --pan:  #fffdf7;
  --pan2: #f0ebe0;

  /* Randen */
  --bdr:  #e0d8c8;
  --bdr2: #c8bda8;

  /* Groen */
  --g:  #007a47;
  --gd: #005c35;
  --gk: #d4f5e5;

  /* Rood */
  --r:  #cc2222;
  --rd: #a01a1a;
  --rk: #fde8e8;

  /* Amber */
  --a:  #8a6200;
  --ad: #6b4c00;
  --ak: #fef3d4;

  /* Blauw */
  --b:  #0070a8;
  --bd: #005580;

  /* Tekst */
  --t1: #2c2416;
  --t2: #6b5d45;
  --t3: #9c8b72;

  /* Fonts */
  --mono:  'Share Tech Mono', monospace;
  --sans:  'Rajdhani', sans-serif;
  --serif: 'Crimson Pro', Georgia, serif;
}
"""
TOKENS = TOKENS_LICHT


# ════════════════════════════════════════════════════════════════════
# BASIS — gedeeld door alle scripts
# ════════════════════════════════════════════════════════════════════

# * { box-sizing: border-box; margin: 0; padding: 0; }

BASE_CSS = """
.stApp { background: var(--bg) !important; font-family: var(--mono) !important; color: var(--t1) !important; }
#MainMenu, footer, header { visibility: hidden; }
.stApp > header { display: none; }
section[data-testid="stSidebar"] { display: none; }

/* Animaties */
@keyframes pulse  { 0%,100%{opacity:1} 50%{opacity:.5} }
@keyframes scroll { 0%{transform:translateX(0)} 100%{transform:translateX(-50%)} }

/* Streamlit widget overrides */
div[data-testid="stHorizontalBlock"] { gap: 0 !important; }
div[data-testid="column"]            { padding: 0 4px !important; }

.stSelectbox > div > div {
  background: var(--pan) !important; border: 1px solid var(--bdr2) !important;
  color: var(--t1) !important; font-family: var(--mono) !important;
  font-size: 10.5px !important; border-radius: 0 !important;
}
.stButton > button {
  background: var(--pan) !important; color: var(--b) !important;
  border: 1px solid var(--bd) !important; font-family: var(--mono) !important;
  font-size: 10px !important; letter-spacing: .15em !important;
  padding: 5px 12px !important; border-radius: 0 !important;
  text-transform: uppercase !important; width: 100%;
}
.stButton > button:hover { background: var(--bd) !important; color: #fff !important; }

div[data-testid="stMetric"] {
  background: var(--pan) !important; border: 1px solid var(--bdr2) !important;
  padding: 12px !important; border-radius: 0 !important;
}
"""

# ════════════════════════════════════════════════════════════════════
# DASHBOARD CSS — app.py en backtest.py
# ════════════════════════════════════════════════════════════════════
DASHBOARD_CSS = """
.block-container { padding: 0 !important; max-width: 100% !important; }

/* Ticker */
.tbr  { background: #000; border-bottom: 1px solid var(--bdr2); padding: 5px 0; overflow: hidden; white-space: nowrap; }
.tsc  { display: inline-block; animation: scroll 70s linear infinite; font-size: 11px; }
.tbr:hover .tsc { animation-play-state: paused; }
.ti   { display: inline-block; margin-right: 26px; color: var(--t2); }
.ti .sy { color: var(--b); margin-right: 3px; }
.up { color: var(--g); } .dn { color: var(--r); } .fl { color: var(--t2); }

/* Header */
.hdr { background: linear-gradient(90deg,#060e18,#0a1825); border-bottom: 1px solid var(--bdr2);
       padding: 9px 18px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
.ht  { font-family: var(--sans); font-size: 21px; font-weight: 700; color: var(--b); letter-spacing: .12em; text-transform: uppercase; }
.hs  { font-size: 9px; color: var(--t3); letter-spacing: .15em; margin-top: 2px; }
.ts  { font-size: 9px; color: var(--t3); }
.lb  { display: inline-block; background: var(--gk); color: var(--g); border: 1px solid var(--gd);
       padding: 2px 8px; font-size: 9px; letter-spacing: .2em; border-radius: 2px; animation: pulse 2s ease-in-out infinite; }
.db  { display: inline-block; background: var(--ak); color: var(--a); border: 1px solid var(--ad);
       padding: 2px 8px; font-size: 9px; letter-spacing: .2em; border-radius: 2px; }

/* Body wrapper */
.body { padding: 12px 16px; }

/* Alert */
.al { background: var(--ak); border: 1px solid var(--ad); padding: 6px 14px; margin-bottom: 9px;
      font-size: 10.5px; color: var(--a); letter-spacing: .07em; animation: pulse 1.8s infinite; }

/* Hero */
.hero  { display: flex; gap: 12px; margin-bottom: 12px; align-items: stretch; min-height: 170px; }
.hdec  { flex: 0 0 230px; background: var(--pan); border: 1px solid var(--bdr2); padding: 18px;
          display: flex; flex-direction: column; align-items: center; justify-content: center;
          text-align: center; position: relative; overflow: hidden; }
.hdec::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; }
.hy::before { background: var(--g); box-shadow: 0 0 12px var(--g); }
.hc::before { background: var(--a); box-shadow: 0 0 12px var(--a); }
.hn::before { background: var(--r); box-shadow: 0 0 12px var(--r); }
.dl  { font-size: 9px; letter-spacing: .3em; color: var(--t3); margin-bottom: 7px; }
.db2 { font-family: var(--sans); font-size: 50px; font-weight: 700; letter-spacing: .05em; line-height: 1; margin-bottom: 7px; }
.dy  { color: var(--g); text-shadow: 0 0 20px rgba(0,255,157,.35); }
.dc  { color: var(--a); text-shadow: 0 0 20px rgba(255,183,0,.35); }
.dn2 { color: var(--r); text-shadow: 0 0 20px rgba(255,58,58,.35); }
.dd  { font-size: 9.5px; color: var(--t2); line-height: 1.6; max-width: 190px; }

/* Ring gauges */
.sr  { display: flex; gap: 12px; flex: 1; }
.sc  { flex: 1; background: var(--pan); border: 1px solid var(--bdr2); padding: 16px;
        display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; }
.rc  { position: relative; width: 106px; height: 106px; margin: 0 auto 9px; }
.rs  { transform: rotate(-90deg); }
.rt  { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); font-size: 21px; font-weight: bold; font-family: var(--mono); }
.sct { font-size: 9px; letter-spacing: .2em; color: var(--t3); margin-bottom: 3px; }
.scs { font-size: 9px; color: var(--t2); }

/* Panels */
.pg  { display: grid; grid-template-columns: repeat(5,1fr); gap: 9px; margin-bottom: 10px; }
.pn  { background: var(--pan); border: 1px solid var(--bdr); padding: 11px; }
.pn:hover { border-color: var(--bdr2); }
.ph  { display: flex; justify-content: space-between; align-items: center; margin-bottom: 7px; border-bottom: 1px solid var(--bdr); padding-bottom: 5px; }
.pt  { font-size: 9px; letter-spacing: .2em; color: var(--t3); text-transform: uppercase; }
.pb  { font-size: 10px; padding: 1px 5px; border-radius: 2px; }
.bg  { background: var(--gk); color: var(--g); border: 1px solid var(--gd); }
.ba  { background: var(--ak); color: var(--a); border: 1px solid var(--ad); }
.br2 { background: var(--rk); color: var(--r); border: 1px solid var(--rd); }
.mr  { display: flex; justify-content: space-between; align-items: center; padding: 2px 0; border-bottom: 1px solid #0a1820; font-size: 10px; }
.mr:last-of-type { border-bottom: none; }
.mn  { color: var(--t2); font-size: 9px; }
.mv  { color: var(--t1); }
.mu  { color: var(--g); } .md { color: var(--r); } .mn2 { color: var(--t2); }
.ip  { margin-top: 6px; font-size: 9px; padding: 3px 6px; border-left: 2px solid; letter-spacing: .04em; line-height: 1.4; }
.ih  { border-color: var(--g); background: var(--gk); color: var(--gd); }
.in  { border-color: var(--a); background: var(--ak); color: var(--ad); }
.ir  { border-color: var(--r); background: var(--rk); color: var(--rd); }

/* Bottom row */
.br  { display: grid; grid-template-columns: 1fr 1fr 1.1fr; gap: 9px; }
.bp  { background: var(--pan); border: 1px solid var(--bdr); padding: 11px; }
.bpt { font-size: 9px; letter-spacing: .2em; color: var(--b); margin-bottom: 8px; border-bottom: 1px solid var(--bdr); padding-bottom: 5px; text-transform: uppercase; }

/* Sector heatmap */
.sbr { display: flex; align-items: center; margin-bottom: 3px; gap: 6px; }
.sl2 { color: var(--t2); width: 36px; flex-shrink: 0; font-size: 9.5px; }
.st  { flex: 1; height: 11px; background: var(--bg2); position: relative; border: 1px solid var(--bdr); overflow: hidden; }
.sm  { position: absolute; top: 0; bottom: 0; left: 50%; width: 1px; background: var(--bdr2); }
.sp  { color: var(--t1); width: 48px; text-align: right; flex-shrink: 0; font-size: 9.5px; }

/* Score breakdown */
.sbrow { display: flex; align-items: center; margin-bottom: 6px; gap: 6px; }
.sbl   { color: var(--t2); width: 70px; font-size: 9.5px; flex-shrink: 0; }
.sbw   { color: var(--t3); width: 26px; font-size: 9px; flex-shrink: 0; }
.sbt   { flex: 1; height: 8px; background: var(--bg2); position: relative; border: 1px solid var(--bdr); }
.sbf   { position: absolute; top: 0; left: 0; height: 100%; }
.sbv   { color: var(--t1); width: 26px; text-align: right; font-size: 9.5px; flex-shrink: 0; }

/* Terminal */
.th { font-size: 9px; letter-spacing: .2em; color: var(--b); margin-bottom: 7px; border-bottom: 1px solid var(--bdr); padding-bottom: 5px; }
.tt { font-family: var(--mono); font-size: 10px; color: var(--gd); line-height: 1.9; }
.tp { color: var(--t3); }

/* Backtest KPI / sec-hdr */
.kpi   { background: var(--pan); border: 1px solid var(--bdr2); padding: 14px; text-align: center; border-radius: 2px; }
.kpi-v { font-size: 28px; font-weight: 700; font-family: var(--sans); margin-bottom: 3px; }
.kpi-l { font-size: 9px; letter-spacing: .2em; color: var(--t3); }
.sec-hdr { font-size: 9px; letter-spacing: .25em; color: var(--b); border-bottom: 1px solid var(--bdr2);
           padding-bottom: 6px; margin: 18px 0 10px; text-transform: uppercase; }
"""

# ════════════════════════════════════════════════════════════════════
# ARTIKEL CSS — uitleg.py en uitleg_app.py (extra bovenop BASE)
# ════════════════════════════════════════════════════════════════════
ARTICLE_CSS = """
/* Typografie */
.art-title { font-family: var(--sans); font-size: 38px; font-weight: 700; color: var(--b);
             letter-spacing: .06em; text-transform: uppercase; margin-bottom: 6px; line-height: 1.1; }
.art-sub   { font-family: var(--mono); font-size: 11px; color: var(--t3); letter-spacing: .2em; margin-bottom: 32px; }
.art-lead  { font-family: var(--serif); font-size: 19px; color: var(--t1); line-height: 1.75;
             margin-bottom: 28px; border-left: 3px solid var(--b); padding-left: 18px; }

h2 { font-family: var(--sans) !important; font-size: 20px !important; font-weight: 700 !important;
     color: var(--b) !important; letter-spacing: .12em !important; text-transform: uppercase !important;
     border-bottom: 1px solid var(--bdr2) !important; padding-bottom: 8px !important;
     margin-top: 42px !important; margin-bottom: 16px !important; }
h3 { font-family: var(--sans) !important; font-size: 15px !important; font-weight: 600 !important;
     color: var(--a) !important; letter-spacing: .1em !important; text-transform: uppercase !important;
     margin-top: 28px !important; margin-bottom: 10px !important; }
p, li  { font-family: var(--serif) !important; font-size: 16px !important; color: var(--t1) !important; line-height: 1.8 !important; }
li     { margin-bottom: 4px !important; }
strong { color: var(--t1) !important; font-family: var(--mono) !important; font-size: 13px !important; }
code   { background: var(--pan2) !important; color: var(--g) !important; font-family: var(--mono) !important;
         font-size: 12px !important; padding: 1px 5px !important; border-radius: 2px !important;
         border: 1px solid var(--bdr) !important; }

/* Callouts */
.callout        { padding: 14px 18px; border-radius: 2px; margin: 18px 0; font-family: var(--mono); font-size: 12px; line-height: 1.7; }
.callout-green  { background: var(--gk); border-left: 3px solid var(--g); color: var(--gd); }
.callout-yellow { background: var(--ak); border-left: 3px solid var(--a); color: var(--ad); }
.callout-red    { background: var(--rk); border-left: 3px solid var(--r); color: var(--rd); }
.callout-blue   { background: #001d30;   border-left: 3px solid var(--b); color: var(--b); }

/* Score-tabel */
.score-table    { width: 100%; border-collapse: collapse; margin: 16px 0; font-family: var(--mono); font-size: 12px; }
.score-table th { background: var(--pan2); color: var(--t3); letter-spacing: .15em; padding: 8px 12px;
                  text-align: left; border-bottom: 1px solid var(--bdr2); font-size: 10px; }
.score-table td { padding: 8px 12px; border-bottom: 1px solid var(--bdr); color: var(--t1); vertical-align: top; }
.score-table tr:last-child td { border-bottom: none; }
.score-table .cat { color: var(--a); font-weight: 600; }
.score-table .wt  { color: var(--b); }
.score-table .pos { color: var(--gd); }
.score-table .neg { color: var(--rd); }

/* Formule box */
.formula     { background: var(--pan); border: 1px solid var(--bdr2); padding: 16px 20px;
               font-family: var(--mono); font-size: 13px; color: var(--g); margin: 16px 0; line-height: 2; border-radius: 2px; }
.formula .dim { color: var(--t3); }

/* Divider */
.div { border: none; border-top: 1px solid var(--bdr2); margin: 36px 0; }

/* Disclaimer */
.disclaimer { background: var(--pan); border: 1px solid var(--bdr); padding: 14px 18px;
              font-family: var(--mono); font-size: 10px; color: var(--t3); line-height: 1.8;
              margin-top: 48px; border-radius: 2px; }

/* Demo-blokken (uitleg_app replica's) */
.demo-wrap  { background: var(--pan); border: 1px solid var(--bdr2); border-radius: 3px; padding: 0; margin: 18px 0; overflow: hidden; }
.demo-label { background: var(--pan2); border-bottom: 1px solid var(--bdr2);
              font-family: var(--mono); font-size: 9px; color: var(--t3); letter-spacing: .2em; padding: 6px 14px; }

/* Mini-dashboard replica's */
.hero-mini  { display: flex; gap: 10px; padding: 14px; align-items: stretch; flex-wrap: wrap; }
.dec-card   { background: var(--pan2); border: 1px solid var(--bdr2); padding: 16px 20px;
              min-width: 160px; text-align: center; position: relative; overflow: hidden; }
.dec-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; }
.dc-yes::before { background: var(--g); box-shadow: 0 0 12px var(--g); }
.dc-cau::before { background: var(--a); box-shadow: 0 0 12px var(--a); }
.dc-no::before  { background: var(--r); box-shadow: 0 0 12px var(--r); }
.dec-lbl { font-size: 9px; letter-spacing: .25em; color: var(--t3); margin-bottom: 8px; font-family: var(--mono); }
.dec-val { font-family: var(--sans); font-size: 42px; font-weight: 700; line-height: 1; margin-bottom: 8px; }
.dec-yes { color: var(--g); text-shadow: 0 0 20px rgba(0,255,157,.3); }
.dec-cau { color: var(--a); text-shadow: 0 0 20px rgba(255,183,0,.3); }
.dec-no  { color: var(--r); text-shadow: 0 0 20px rgba(255,58,58,.3); }
.dec-sub { font-size: 10px; color: var(--t2); font-family: var(--mono); line-height: 1.5; }

.ring-card  { background: var(--pan2); border: 1px solid var(--bdr2); padding: 16px; flex: 1;
              min-width: 140px; display: flex; flex-direction: column; align-items: center; }
.ring-title { font-size: 9px; letter-spacing: .18em; color: var(--t3); margin-bottom: 10px;
              font-family: var(--mono); text-align: center; }
.ring-wrap  { position: relative; width: 90px; height: 90px; margin: 0 auto 8px; }
.ring-num   { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
              font-size: 22px; font-weight: 700; font-family: var(--mono); }
.ring-sub   { font-size: 9px; color: var(--t2); font-family: var(--mono); text-align: center; }

.regime-card { background: var(--pan2); border: 1px solid var(--bdr2); padding: 16px; flex: 1; min-width: 160px; }
.regime-val  { font-family: var(--sans); font-size: 26px; font-weight: 700; margin-bottom: 10px; }
.regime-row  { display: flex; justify-content: space-between; font-size: 10px; font-family: var(--mono);
               padding: 3px 0; border-bottom: 1px solid var(--bdr); }
.regime-row:last-child { border-bottom: none; }
.regime-k { color: var(--t2); }
.regime-v { color: var(--t1); }

.panel-grid  { display: grid; grid-template-columns: repeat(5,1fr); gap: 8px; padding: 14px; }
.panel       { background: var(--pan2); border: 1px solid var(--bdr); padding: 10px; }
.panel-hdr   { display: flex; justify-content: space-between; align-items: center;
               margin-bottom: 6px; padding-bottom: 5px; border-bottom: 1px solid var(--bdr); }
.panel-title { font-size: 9px; letter-spacing: .15em; color: var(--t3); font-family: var(--mono); }
.panel-score { font-size: 10px; padding: 1px 5px; border-radius: 2px; font-family: var(--mono); }
.ps-g { background: var(--gk); color: var(--g); border: 1px solid var(--gd); }
.ps-a { background: var(--ak); color: var(--a); border: 1px solid var(--ad); }
.ps-r { background: var(--rk); color: var(--r); border: 1px solid var(--rd); }
.metric         { display: flex; justify-content: space-between; padding: 2px 0;
                  border-bottom: 1px solid #0a1820; font-size: 10px; font-family: var(--mono); }
.metric:last-of-type { border-bottom: none; }
.mk  { color: var(--t2); font-size: 9px; }
.mv  { color: var(--t1); }
.mup { color: var(--g); }
.mdn { color: var(--r); }
.interp { margin-top: 6px; font-size: 9px; padding: 3px 6px; border-left: 2px solid;
          letter-spacing: .04em; line-height: 1.4; font-family: var(--mono); }
.ih  { border-color: var(--g); background: var(--gk); color: var(--gd); }
.in2 { border-color: var(--a); background: var(--ak); color: var(--ad); }
.ir  { border-color: var(--r); background: var(--rk); color: var(--rd); }

.sect-bar-wrap { padding: 14px; }
.sbr2 { display: flex; align-items: center; margin-bottom: 4px; gap: 6px; }
.sl   { color: var(--t2); width: 38px; flex-shrink: 0; font-size: 9.5px; font-family: var(--mono); }
.st2  { flex: 1; height: 11px; background: var(--bg); position: relative; border: 1px solid var(--bdr); overflow: hidden; }
.sm2  { position: absolute; top: 0; bottom: 0; left: 50%; width: 1px; background: var(--bdr2); }
.sp2  { color: var(--t1); width: 50px; text-align: right; flex-shrink: 0; font-size: 9.5px; font-family: var(--mono); }

/* Nummer-kaarten */
.num-card  { background: var(--pan); border: 1px solid var(--bdr2); padding: 12px 16px;
             margin-bottom: 8px; display: flex; align-items: flex-start; gap: 12px; }
.num-n     { font-family: var(--sans); font-size: 28px; font-weight: 700; color: var(--b);
             line-height: 1; flex-shrink: 0; min-width: 36px; }
.num-title { font-family: var(--mono); font-size: 11px; color: var(--a); letter-spacing: .1em; margin-bottom: 4px; }
.num-text  { font-family: var(--serif); font-size: 14px; color: var(--t2); line-height: 1.6; }

.sec-badge { display: inline-block; background: var(--pan2); border: 1px solid var(--bdr2);
             font-family: var(--mono); font-size: 9px; letter-spacing: .2em; color: var(--t3);
             padding: 3px 10px; margin-bottom: 14px; border-radius: 2px; }

.bodytext { padding: 100px; max-width: 800px; margin: 0 auto; }
"""


# ════════════════════════════════════════════════════════════════════
# PUBLIEKE FUNCTIES
# ════════════════════════════════════════════════════════════════════

def apply_theme():
    """
    Dashboard-stijl voor app.py en backtest.py.
    Voeg daarna pagina-specifieke CSS toe via een tweede st.markdown().
    """
    st.markdown(f"<style>{TOKENS}{BASE_CSS}{DASHBOARD_CSS}</style>",
                unsafe_allow_html=True)


def apply_article_theme():
    """
    Artikel-stijl voor uitleg.py en uitleg_app.py.
    Voeg daarna pagina-specifieke CSS toe (bijv. block-container breedte).
    """
    st.markdown(f"<style>{TOKENS}{BASE_CSS}{ARTICLE_CSS}</style>",
                unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TEMPLATE CSS — bloomberg_template.py klassenamen
# Alle klassen die de template gebruikt maar niet in DASHBOARD_CSS staan.
# ════════════════════════════════════════════════════════════════════
TEMPLATE_CSS = """
/* Animaties */
@keyframes glow { 0%,100%{box-shadow:0 0 6px currentColor} 50%{box-shadow:0 0 18px currentColor} }

/* Ticker */
.ticker-bar  { background: var(--bg2); border-bottom: 1px solid var(--bdr2); padding: 5px 0; overflow: hidden; white-space: nowrap; }
.ticker-scroll { display: inline-block; animation: scroll 80s linear infinite; font-size: 11px; }
.ticker-bar:hover .ticker-scroll { animation-play-state: paused; }
.ticker-item { display: inline-block; margin-right: 28px; color: var(--t2); }
.ticker-item .sym { color: var(--b); margin-right: 4px; }
.up  { color: var(--g); } .dn { color: var(--r); } .neu { color: var(--t2); }

/* Header */
.dash-header   { background: linear-gradient(90deg, var(--pan2), var(--pan)); border-bottom: 1px solid var(--bdr2);
                 padding: 10px 18px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
.dash-title    { font-family: var(--sans); font-size: 20px; font-weight: 700; color: var(--b); letter-spacing: .12em; text-transform: uppercase; }
.dash-subtitle { font-size: 9px; color: var(--t3); letter-spacing: .15em; margin-top: 2px; }
.dash-meta     { font-size: 9px; color: var(--t3); text-align: right; }
.dash-body     { padding: 12px 16px; }

/* Badges */
.badge        { display: inline-block; padding: 2px 9px; font-size: 9px; letter-spacing: .2em; border-radius: 2px; font-family: var(--mono); }
.badge-green  { background: var(--gk); color: var(--g);   border: 1px solid var(--gd); animation: pulse 2s infinite; }
.badge-amber  { background: var(--ak); color: var(--a);   border: 1px solid var(--ad); }
.badge-red    { background: var(--rk); color: var(--r);   border: 1px solid var(--rd); }
.badge-blue   { background: var(--pan2); color: var(--b); border: 1px solid var(--bd); }
.badge-gray   { background: var(--pan2); color: var(--t2); border: 1px solid var(--bdr2); }

/* Alert banner */
.alert-banner            { background: var(--ak); border: 1px solid var(--ad); padding: 7px 14px; margin-bottom: 10px;
                           font-size: 11px; color: var(--a); letter-spacing: .07em; animation: pulse 1.8s infinite; }
.alert-banner.alert-red  { background: var(--rk); border-color: var(--rd); color: var(--r); }
.alert-banner.alert-green { background: var(--gk); border-color: var(--gd); color: var(--g); animation: none; }

/* KPI-rij */
.kpi-grid  { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 9px; margin-bottom: 12px; }
.kpi-card  { background: var(--pan); border: 1px solid var(--bdr2); padding: 14px 16px; position: relative; overflow: hidden; }
.kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; }
.kpi-green::before { background: var(--g); box-shadow: 0 0 8px var(--g); }
.kpi-amber::before { background: var(--a); box-shadow: 0 0 8px var(--a); }
.kpi-red::before   { background: var(--r); box-shadow: 0 0 8px var(--r); }
.kpi-blue::before  { background: var(--b); box-shadow: 0 0 8px var(--b); }
.kpi-label { font-size: 9px; letter-spacing: .2em; color: var(--t3); margin-bottom: 6px; text-transform: uppercase; }
.kpi-value { font-family: var(--sans); font-size: 32px; font-weight: 700; line-height: 1; margin-bottom: 4px; }
.kpi-green .kpi-value { color: var(--g); } .kpi-amber .kpi-value { color: var(--a); }
.kpi-red   .kpi-value { color: var(--r); } .kpi-blue  .kpi-value { color: var(--b); }
.kpi-sub   { font-size: 9px; color: var(--t2); line-height: 1.5; }

/* Ring gauge */
.ring-wrap  { position: relative; width: 106px; height: 106px; margin: 0 auto 8px; }
.ring-svg   { transform: rotate(-90deg); }
.ring-num   { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); font-size: 22px; font-weight: bold; font-family: var(--mono); }
.ring-card  { background: var(--pan); border: 1px solid var(--bdr2); padding: 16px; text-align: center; display: flex; flex-direction: column; align-items: center; }
.ring-title { font-size: 9px; letter-spacing: .2em; color: var(--t3); margin-bottom: 8px; text-transform: uppercase; }
.ring-sub   { font-size: 9px; color: var(--t2); }

/* Panels */
.panel-grid   { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 9px; margin-bottom: 10px; }
.panel        { background: var(--pan); border: 1px solid var(--bdr); padding: 11px; transition: border-color .15s; }
.panel:hover  { border-color: var(--bdr2); }
.panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 7px; padding-bottom: 5px; border-bottom: 1px solid var(--bdr); }
.panel-title  { font-size: 9px; letter-spacing: .2em; color: var(--t3); text-transform: uppercase; }
.score-badge  { font-size: 10px; padding: 1px 5px; border-radius: 2px; }
.sb-green { background: var(--gk); color: var(--g); border: 1px solid var(--gd); }
.sb-amber { background: var(--ak); color: var(--a); border: 1px solid var(--ad); }
.sb-red   { background: var(--rk); color: var(--r); border: 1px solid var(--rd); }

/* Metric rijen */
.metric-row           { display: flex; justify-content: space-between; align-items: center; padding: 2px 0; border-bottom: 1px solid var(--bdr); font-size: 10px; }
.metric-row:last-of-type { border-bottom: none; }
.metric-key { color: var(--t2); font-size: 9px; } .metric-val { color: var(--t1); }
.metric-up  { color: var(--g); } .metric-dn { color: var(--r); } .metric-neu { color: var(--t2); }

/* Interpretatie-label */
.interp       { margin-top: 6px; font-size: 9px; padding: 3px 6px; border-left: 2px solid; letter-spacing: .04em; line-height: 1.4; }
.interp-green { border-color: var(--g); background: var(--gk); color: var(--gd); }
.interp-amber { border-color: var(--a); background: var(--ak); color: var(--ad); }
.interp-red   { border-color: var(--r); background: var(--rk); color: var(--rd); }

/* Sectielabel */
.section-label { font-size: 9px; letter-spacing: .25em; color: var(--b); border-bottom: 1px solid var(--bdr2); padding-bottom: 6px; margin-bottom: 9px; text-transform: uppercase; }

/* Heatmap */
.heatmap-row { display: flex; align-items: center; margin-bottom: 4px; gap: 7px; }
.hm-label    { color: var(--t2); width: 40px; flex-shrink: 0; font-size: 9.5px; }
.hm-track    { flex: 1; height: 11px; background: var(--bg2); position: relative; border: 1px solid var(--bdr); overflow: hidden; }
.hm-center   { position: absolute; top: 0; bottom: 0; left: 50%; width: 1px; background: var(--bdr2); }
.hm-value    { color: var(--t1); width: 52px; text-align: right; flex-shrink: 0; font-size: 9.5px; }

/* Score breakdown */
.breakdown-row { display: flex; align-items: center; margin-bottom: 6px; gap: 6px; }
.bd-label  { color: var(--t2); width: 74px; font-size: 9.5px; flex-shrink: 0; }
.bd-weight { color: var(--t3); width: 26px; font-size: 9px; flex-shrink: 0; }
.bd-track  { flex: 1; height: 8px; background: var(--bg2); position: relative; border: 1px solid var(--bdr); }
.bd-fill   { position: absolute; top: 0; left: 0; height: 100%; }
.bd-score  { color: var(--t1); width: 26px; text-align: right; font-size: 9.5px; flex-shrink: 0; }

/* Generieke kaart */
.card       { background: var(--pan); border: 1px solid var(--bdr); padding: 12px; }
.card-title { font-size: 9px; letter-spacing: .2em; color: var(--b); margin-bottom: 8px; padding-bottom: 5px; border-bottom: 1px solid var(--bdr); text-transform: uppercase; }

/* Divider */
.divider { border: none; border-top: 1px solid var(--bdr2); margin: 14px 0; }

/* Extra Streamlit overrides */
.stSlider > div > div > div { background: var(--bdr2) !important; }
.stSlider > div > div > div > div { background: var(--b) !important; }
.stNumberInput > div > div > input, .stTextInput > div > div > input, .stDateInput > div > div > input {
  background: var(--pan) !important; border: 1px solid var(--bdr2) !important;
  color: var(--t1) !important; font-family: var(--mono) !important;
  font-size: 11px !important; border-radius: 0 !important;
}
div[data-testid="stDataFrame"] { border: 1px solid var(--bdr2) !important; }
.stSpinner > div { border-top-color: var(--b) !important; }

/* Mobile */
@media (max-width: 768px) {
  .dash-header { flex-direction: column; align-items: flex-start; gap: 6px; }
  .dash-title  { font-size: 16px; }
  .kpi-grid    { grid-template-columns: repeat(2, 1fr); gap: 6px; }
  .kpi-value   { font-size: 24px; }
  .panel-grid  { grid-template-columns: 1fr; gap: 6px; }
  .ring-wrap   { width: 80px; height: 80px; }
  .ring-num    { font-size: 17px; }
  .stButton > button { padding: 10px 14px !important; }
  .hm-label    { width: 32px; font-size: 9px; }
  .hm-value    { width: 44px; font-size: 9px; }
}
@media (max-width: 480px) {
  .kpi-grid  { grid-template-columns: repeat(2, 1fr); }
  .kpi-value { font-size: 20px; }
}
"""


def apply_template_theme():
    """
    Volledige stijl voor bloomberg_template.py.
    Bevat TOKENS + BASE_CSS + TEMPLATE_CSS.
    """
    st.markdown(f"<style>{TOKENS}{BASE_CSS}{TEMPLATE_CSS}</style>",
                unsafe_allow_html=True)