"""
uitleg_app.py — "Should I Be Trading?" — Gebruikshandleiding voor app.py
Een interactief artikel dat elke sectie van het live dashboard uitlegt,
met live data als illustratie.

Gebruik:
  streamlit run uitleg_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
try:
    st.set_page_config(
        page_title="Dashboard Uitleg — Should I Be Trading?",
        page_icon="📖",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
except:
    pass

def uitleg_app():

    # ── CSS ──────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Crimson+Pro:ital,wght@0,400;0,600;1,400&display=swap');
    :root{
      --bg:#050a0f; --pan:#0a1520; --pan2:#0d1c2a; --bdr:#0e2233; --bdr2:#1a3a52;
      --g:#00ff9d; --gd:#00cc7a; --gk:#003d25;
      --r:#ff3a3a; --rd:#cc2222; --rk:#3d0000;
      --a:#ffb700; --ad:#cc9200; --ak:#3d2c00;
      --b:#00b4ff; --bd:#007ab5;
      --t1:#c8dce8; --t2:#6a8fa8; --t3:#3a5a6e;
      --mono:'Share Tech Mono',monospace;
      --sans:'Rajdhani',sans-serif;
      --serif:'Crimson Pro',Georgia,serif;
    }
    * { box-sizing: border-box; }
    .stApp { background: var(--bg) !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 2rem 4rem !important; max-width: 960px !important; margin: 0 auto; }
    .stApp > header { display: none; }
    section[data-testid="stSidebar"] { display: none; }

    /* Typografie */
    .art-title { font-family:var(--sans); font-size:38px; font-weight:700; color:var(--b);
      letter-spacing:.06em; text-transform:uppercase; margin-bottom:6px; line-height:1.1; }
    .art-sub { font-family:var(--mono); font-size:11px; color:var(--t3);
      letter-spacing:.2em; margin-bottom:32px; }
    .art-lead { font-family:var(--serif); font-size:19px; color:var(--t1); line-height:1.75;
      margin-bottom:28px; border-left:3px solid var(--b); padding-left:18px; }
    h2 { font-family:var(--sans)!important; font-size:20px!important; font-weight:700!important;
      color:var(--b)!important; letter-spacing:.12em!important; text-transform:uppercase!important;
      border-bottom:1px solid var(--bdr2)!important; padding-bottom:8px!important;
      margin-top:42px!important; margin-bottom:16px!important; }
    h3 { font-family:var(--sans)!important; font-size:15px!important; font-weight:600!important;
      color:var(--a)!important; letter-spacing:.1em!important; text-transform:uppercase!important;
      margin-top:28px!important; margin-bottom:10px!important; }
    p, li { font-family:var(--serif)!important; font-size:16px!important; color:var(--t1)!important;
      line-height:1.8!important; }
    li { margin-bottom:4px!important; }
    strong { color:var(--t1)!important; font-family:var(--mono)!important; font-size:13px!important; }
    code { background:var(--pan2)!important; color:var(--g)!important; font-family:var(--mono)!important;
      font-size:12px!important; padding:1px 5px!important; border-radius:2px!important;
      border:1px solid var(--bdr)!important; }

    /* Callouts */
    .callout { padding:14px 18px; border-radius:2px; margin:18px 0;
      font-family:var(--mono); font-size:12px; line-height:1.7; }
    .callout-green  { background:var(--gk); border-left:3px solid var(--g); color:var(--gd); }
    .callout-yellow { background:var(--ak); border-left:3px solid var(--a); color:var(--ad); }
    .callout-red    { background:var(--rk); border-left:3px solid var(--r); color:var(--rd); }
    .callout-blue   { background:#001d30;   border-left:3px solid var(--b); color:var(--b);  }

    /* Sectie-indicator */
    .sec-badge {
      display:inline-block; background:var(--pan2); border:1px solid var(--bdr2);
      font-family:var(--mono); font-size:9px; letter-spacing:.2em; color:var(--t3);
      padding:3px 10px; margin-bottom:14px; border-radius:2px;
    }

    /* Live data demo blok */
    .demo-wrap {
      background:var(--pan); border:1px solid var(--bdr2); border-radius:3px;
      padding:0; margin:18px 0; overflow:hidden;
    }
    .demo-label {
      background:var(--pan2); border-bottom:1px solid var(--bdr2);
      font-family:var(--mono); font-size:9px; color:var(--t3);
      letter-spacing:.2em; padding:6px 14px;
    }

    /* Mini-dashboard elementen (replica's van app.py) */
    .tbr{background:#000;border-bottom:1px solid var(--bdr2);padding:6px 14px;
        overflow:hidden;white-space:nowrap;font-size:11px;}
    .ti{display:inline-block;margin-right:22px;color:var(--t2)}
    .ti .sy{color:var(--b);margin-right:3px}
    .up{color:var(--g)} .dn{color:var(--r)} .fl{color:var(--t2)}

    .hero-mini{display:flex;gap:10px;padding:14px;align-items:stretch;flex-wrap:wrap}
    .dec-card{background:var(--pan2);border:1px solid var(--bdr2);padding:16px 20px;
              min-width:160px;text-align:center;position:relative;overflow:hidden}
    .dec-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
    .dc-yes::before{background:var(--g);box-shadow:0 0 12px var(--g)}
    .dc-cau::before{background:var(--a);box-shadow:0 0 12px var(--a)}
    .dc-no::before{background:var(--r);box-shadow:0 0 12px var(--r)}
    .dec-lbl{font-size:9px;letter-spacing:.25em;color:var(--t3);margin-bottom:8px;font-family:var(--mono)}
    .dec-val{font-family:var(--sans);font-size:42px;font-weight:700;line-height:1;margin-bottom:8px}
    .dec-yes{color:var(--g);text-shadow:0 0 20px rgba(0,255,157,.3)}
    .dec-cau{color:var(--a);text-shadow:0 0 20px rgba(255,183,0,.3)}
    .dec-no {color:var(--r);text-shadow:0 0 20px rgba(255,58,58,.3)}
    .dec-sub{font-size:10px;color:var(--t2);font-family:var(--mono);line-height:1.5}

    .ring-card{background:var(--pan2);border:1px solid var(--bdr2);padding:16px;
                flex:1;min-width:140px;display:flex;flex-direction:column;align-items:center}
    .ring-title{font-size:9px;letter-spacing:.18em;color:var(--t3);margin-bottom:10px;
                font-family:var(--mono);text-align:center}
    .ring-wrap{position:relative;width:90px;height:90px;margin:0 auto 8px}
    .ring-num{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
              font-size:22px;font-weight:700;font-family:var(--mono)}
    .ring-sub{font-size:9px;color:var(--t2);font-family:var(--mono);text-align:center}

    .regime-card{background:var(--pan2);border:1px solid var(--bdr2);padding:16px;flex:1;min-width:160px}
    .regime-val{font-family:var(--sans);font-size:26px;font-weight:700;margin-bottom:10px}
    .regime-row{display:flex;justify-content:space-between;font-size:10px;
                font-family:var(--mono);padding:3px 0;border-bottom:1px solid var(--bdr)}
    .regime-row:last-child{border-bottom:none}
    .regime-k{color:var(--t2)} .regime-v{color:var(--t1)}

    .panel-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;padding:14px}
    .panel{background:var(--pan2);border:1px solid var(--bdr);padding:10px}
    .panel-hdr{display:flex;justify-content:space-between;align-items:center;
                margin-bottom:6px;padding-bottom:5px;border-bottom:1px solid var(--bdr)}
    .panel-title{font-size:9px;letter-spacing:.15em;color:var(--t3);font-family:var(--mono)}
    .panel-score{font-size:10px;padding:1px 5px;border-radius:2px;font-family:var(--mono)}
    .ps-g{background:var(--gk);color:var(--g);border:1px solid var(--gd)}
    .ps-a{background:var(--ak);color:var(--a);border:1px solid var(--ad)}
    .ps-r{background:var(--rk);color:var(--r);border:1px solid var(--rd)}
    .metric{display:flex;justify-content:space-between;padding:2px 0;
            border-bottom:1px solid #0a1820;font-size:10px;font-family:var(--mono)}
    .metric:last-of-type{border-bottom:none}
    .mk{color:var(--t2);font-size:9px} .mv{color:var(--t1)}
    .mup{color:var(--g)} .mdn{color:var(--r)}
    .interp{margin-top:6px;font-size:9px;padding:3px 6px;border-left:2px solid;
            letter-spacing:.04em;line-height:1.4;font-family:var(--mono)}
    .ih{border-color:var(--g);background:var(--gk);color:var(--gd)}
    .in2{border-color:var(--a);background:var(--ak);color:var(--ad)}
    .ir{border-color:var(--r);background:var(--rk);color:var(--rd)}

    .sect-bar-wrap{padding:14px}
    .sbr{display:flex;align-items:center;margin-bottom:4px;gap:6px}
    .sl{color:var(--t2);width:38px;flex-shrink:0;font-size:9.5px;font-family:var(--mono)}
    .st2{flex:1;height:11px;background:var(--bg);position:relative;border:1px solid var(--bdr);overflow:hidden}
    .sm2{position:absolute;top:0;bottom:0;left:50%;width:1px;background:var(--bdr2)}
    .sp2{color:var(--t1);width:50px;text-align:right;flex-shrink:0;font-size:9.5px;font-family:var(--mono)}

    .formula{background:var(--pan);border:1px solid var(--bdr2);padding:16px 20px;
              font-family:var(--mono);font-size:13px;color:var(--g);margin:16px 0;
              line-height:2;border-radius:2px;}
    .formula .dim{color:var(--t3)}

    .div{border:none;border-top:1px solid var(--bdr2);margin:36px 0}
    .disclaimer{background:var(--pan);border:1px solid var(--bdr);padding:14px 18px;
                font-family:var(--mono);font-size:10px;color:var(--t3);line-height:1.8;
                margin-top:48px;border-radius:2px}

    /* Nummer-annotaties */
    .ann{display:inline-block;background:var(--b);color:#000;font-family:var(--mono);
          font-size:9px;font-weight:700;width:16px;height:16px;border-radius:50%;
          text-align:center;line-height:16px;margin-right:6px;flex-shrink:0}
    .ann-row{display:flex;align-items:flex-start;margin:10px 0;font-family:var(--serif);
              font-size:15px;color:var(--t1);line-height:1.6}

    /* Nummer-kaarten */
    .num-card{background:var(--pan);border:1px solid var(--bdr2);padding:12px 16px;
              margin-bottom:8px;display:flex;align-items:flex-start;gap:12px}
    .num-n{font-family:var(--sans);font-size:28px;font-weight:700;color:var(--b);
            line-height:1;flex-shrink:0;min-width:36px}
    .num-body{}
    .num-title{font-family:var(--mono);font-size:11px;color:var(--a);letter-spacing:.1em;margin-bottom:4px}
    .num-text{font-family:var(--serif);font-size:14px;color:var(--t2);line-height:1.6}
                .bodytext {padding:100px; max-width:800px; margin:0 auto;}
    </style>
    """, unsafe_allow_html=True)


    # ── DATA laden ────────────────────────────────────────────────────────
    @st.cache_data(ttl=60)
    def load():
        try:
            from market_data import get_market_data
            return get_market_data("swing")
        except Exception:
            # Fallback demo zonder yfinance/market_data
            np.random.seed(42)
            j = lambda b, s: float(b + np.random.randn() * s)
            sp = j(562.4, 0); m20 = j(569.1, 0); m50 = j(575.8, 0); m200 = j(542.2, 0)
            sc = {"XLK": -2.8, "XLF": -1.4, "XLE": 0.9, "XLV": 1.1, "XLI": -1.0,
                  "XLY": -2.1, "XLP": 1.3, "XLU": 1.6, "XLB": -0.5, "XLRE": 0.7, "XLC": -1.7}
            pos = sum(1 for v in sc.values() if v > 0)
            ss  = sorted(sc.items(), key=lambda x: x[1], reverse=True)
            d = {
                "is_live": False, "spy_price": sp, "spy_ma20": m20, "spy_ma50": m50,
                "spy_ma200": m200, "spy_rsi14": 44.2, "spy_1d_chg": -0.38,
                "spy_5d_chg": -1.95, "spy_vs_ma20": (sp/m20-1)*100,
                "spy_vs_ma50": (sp/m50-1)*100, "spy_vs_ma200": (sp/m200-1)*100,
                "regime": "CHOP", "qqq_price": 476.2, "qqq_vs_ma50": -3.2, "qqq_1d_chg": -0.6,
                "iwm_price": 204.1, "iwm_vs_ma50": -6.8,
                "vix": 22.8, "vix_slope5d": 0.18, "vix_pct1y": 62.0, "vvix": 112.4,
                "pc_ratio_est": 1.05, "tnx": 4.31, "tnx_slope5d": 0.022,
                "dxy": 103.8, "dxy_slope5d": -0.08, "fed_stance": "NEUTRAL",
                "sector_chg": sc, "top3_sectors": ss[:3], "bottom3_sectors": ss[-3:],
                "sector_spread": ss[0][1] - ss[-1][1], "sectors_positive": pos,
                "pct_sectors_pos": pos/11*100, "ad_ratio_est": pos/max(11-pos,1),
                "nasdaq_nh_est": 78, "nasdaq_nl_est": 142, "est_pct_above_50d": 38.4,
                "mclellan_est": -42.5,
                "market_quality_score": 46, "execution_window_score": 35,
                "decision": "NO",
                "scores": {
                    "volatility": {"score": 42, "weight": .25},
                    "trend":      {"score": 43, "weight": .20},
                    "breadth":    {"score": 50, "weight": .20},
                    "momentum":   {"score": 45, "weight": .25},
                    "macro":      {"score": 60, "weight": .10},
                },
                "summary": "SPY in choppy range — no clear directional trend. Volatility moderate (VIX 22.8). 4/11 sectors advancing. Execution poor — setups failing. → Stand aside. Preserve capital. Build watchlist.",
            }
            return d

    r = load()

    def f(v, dec=2): return "N/A" if v is None else f"{v:.{dec}f}"
    def sc_color(s): return "#00ff9d" if s >= 70 else ("#ffb700" if s >= 50 else "#ff3a3a")
    def ps_cls(s):   return "ps-g" if s >= 70 else ("ps-a" if s >= 50 else "ps-r")

    mqs      = r.get("market_quality_score", 50)
    ews      = r.get("execution_window_score", 50)
    decision = r.get("decision", "CAUTION")
    scores   = r.get("scores", {})
    sc2      = r.get("sector_chg", {})
    is_live  = r.get("is_live", False)
    now      = datetime.now().strftime("%Y-%m-%d  %H:%M")

    dec_cls  = {"YES": "dc-yes", "CAUTION": "dc-cau", "NO": "dc-no"}[decision]
    dec_val  = {"YES": "dec-yes","CAUTION": "dec-cau","NO": "dec-no"}[decision]
    dec_txt  = {"YES": "Volledige positiegrootte. Wind waait mee.",
                "CAUTION": "Halfgrootte. Alleen A+ setups. Strakke stops.",
                "NO": "Zijlijn. Kapitaal bewaren. Watchlist opbouwen."}[decision]

    def ring_svg(score, color):
        R = 38; C = 2*3.14159*R; fl = C*score/100; gp = C-fl
        return (
            f'<div class="ring-wrap">'
            f'<svg viewBox="0 0 90 90" width="90" height="90" style="transform:rotate(-90deg)">'
            f'<circle cx="45" cy="45" r="{R}" fill="none" stroke="var(--bdr)" stroke-width="7"/>'
            f'<circle cx="45" cy="45" r="{R}" fill="none" stroke="{color}" stroke-width="7"'
            f' stroke-linecap="round" stroke-dasharray="{fl:.1f} {gp:.1f}"/>'
            f'</svg>'
            f'<div class="ring-num" style="color:{color}">{score}</div>'
            f'</div>'
        )


    # ════════════════════════════════════════════════════════════════════
    # TITEL
    # ════════════════════════════════════════════════════════════════════
    live_badge = "● LIVE DATA" if is_live else "◌ DEMO DATA"
    live_color = "#00ff9d" if is_live else "#ffb700"

    data_source = "live Yahoo Finance." if is_live else "demodata (yfinance niet gevonden)."

    st.markdown(f"""
    <div class="bodytext">
    <div class="art-title">Dashboard Gebruikshandleiding</div>
    <div class="art-sub">◈ SHOULD I BE TRADING? — APP.PY — ELKE SECTIE UITGELEGD &nbsp;·&nbsp;
    <span style="color:{live_color}">{live_badge}</span> &nbsp;·&nbsp; {now}</div>
    <div class="art-lead">
      Het live dashboard geeft in één oogopslag antwoord op de vraag: <em>is dit een goed moment
      om te handelen?</em> Deze handleiding loopt van boven naar beneden door elk onderdeel van
      de pagina. De voorbeelden hieronder zijn gebaseerd op de meest recente data —
      {data_source}
    </div>
    <hr class="div"/>
    """, unsafe_allow_html=True)


    # ════════════════════════════════════════════════════════════════════
    # SECTIE 1 — TICKER
    # ════════════════════════════════════════════════════════════════════
    st.markdown("## 1 — De scrollende ticker")

    st.markdown("""
    Bovenaan de pagina loopt een zwarte balk continu van rechts naar links. Hij toont de meest
    recente koers en dagverandering van de belangrijkste instrumenten.
    """)

    # Live ticker replica
    TICKER_ITEMS = [
        ("SPY",  r.get("spy_price"),    r.get("spy_1d_chg")),
        ("QQQ",  r.get("qqq_price"),    r.get("qqq_1d_chg")),
        ("IWM",  r.get("iwm_price"),    None),
        ("VIX",  r.get("vix"),          r.get("vix_slope5d")),
        ("TNX",  r.get("tnx"),          r.get("tnx_slope5d")),
        ("DXY",  r.get("dxy"),          r.get("dxy_slope5d")),
    ] + [(e, None, sc2.get(e, 0)) for e in sc2]

    ticker_html = ""
    for sym, price, chg in TICKER_ITEMS:
        c = ("up" if (chg or 0) > 0 else "dn") if chg is not None else "fl"
        mk = "▲" if c == "up" else ("▼" if c == "dn" else "—")
        ps = f" {price:.2f}" if price else ""
        cs = f" {chg:+.2f}%" if chg is not None else ""
        ticker_html += f'<span class="ti"><span class="sy">{sym}</span>{ps}<span class="{c}"> {mk}{cs}</span></span>'

    st.markdown(f"""
    <div class="demo-wrap">
      <div class="demo-label">◈ LIVE VOORBEELD — TICKER BALK</div>
      <div class="tbr">{ticker_html}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Wat staat er in de ticker?**

    - `SPY` — S&P 500 ETF, dagprijs en 1-daagse verandering
    - `QQQ` — Nasdaq-100 ETF
    - `IWM` — Russell 2000 (small caps)
    - `VIX` — Angstindex (volatiliteit); de verandering hier is de 5-daagse slope, niet de dagverandering
    - `TNX` — 10-jaars Amerikaanse staatsrente in procent
    - `DXY` — Dollar Index
    - `XLK` t/m `XLC` — De 11 sector-ETFs van de S&P 500; verandering = 5-daags rendement

    **Kleurcode:** groen ▲ = positief, rood ▼ = negatief, grijs — = neutraal.

    Voor VIX geldt omgekeerd: een dalende VIX (rood ▼) is *goed nieuws* voor traders — minder angst.
    """)

    st.markdown("""
    <div class="callout callout-blue">
      ◈ De ticker ververst automatisch elke 45 seconden. Buiten beurstijden blijven de waarden
      staan op de laatste slotkoers.
    </div>
    """, unsafe_allow_html=True)


    # ════════════════════════════════════════════════════════════════════
    # SECTIE 2 — HEADER & MODUS
    # ════════════════════════════════════════════════════════════════════
    st.markdown("## 2 — De header en de moduskeuze")

    st.markdown("""
    Direct onder de ticker zie je twee bedieningselementen:
    """)

    st.markdown("""
    <div class="num-card">
      <div class="num-n">①</div>
      <div class="num-body">
        <div class="num-title">SWING / DAY MODE</div>
        <div class="num-text">
          Kies je handelsstijl. Dit bepaalt de drempelwaarden voor YES/CAUTION/NO.<br/>
          <strong>Swing:</strong> YES ≥ 80 · CAUTION 60–79 · NO &lt; 60<br/>
          <strong>Day:&nbsp;&nbsp;</strong> YES ≥ 75 · CAUTION 55–74 · NO &lt; 55<br/>
          Day trading heeft lagere drempels: je hoeft minder lang in een positie te blijven,
          dus de omgeving mag iets minder ideaal zijn.
        </div>
      </div>
    </div>
    <div class="num-card">
      <div class="num-n">②</div>
      <div class="num-body">
        <div class="num-title">⟳ REFRESH KNOP</div>
        <div class="num-text">
          Wist de cache en herlaadt de data direct. Normaal ververst het dashboard automatisch
          elke 45 seconden. De knop gebruik je als je direct de nieuwste stand wilt zien,
          bijvoorbeeld na een marktontwikkeling.
        </div>
      </div>
    </div>
    <div class="num-card">
      <div class="num-n">③</div>
      <div class="num-body">
        <div class="num-title">LIVE / DEMO BADGE</div>
        <div class="num-text">
          Rechts in de header staat een badge: groen <strong>● LIVE</strong> als Yahoo Finance
          bereikbaar is, geel <strong>◌ DEMO</strong> als de verbinding mislukt en het dashboard
          op gesimuleerde data draait. In demo-modus zijn alle berekeningen intact maar de
          cijfers zijn niet actueel.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


    # ════════════════════════════════════════════════════════════════════
    # SECTIE 3 — HERO / BESLISSEKCTIE
    # ════════════════════════════════════════════════════════════════════
    st.markdown("## 3 — De beslissectie (hero)")

    st.markdown("""
    Dit is de kern van het dashboard. Vier elementen naast elkaar geven de volledige samenvatting
    van de marktomgeving.
    """)

    # Replica hero
    reg     = r.get("regime", "CHOP")
    reg_col = {"UPTREND": "#00ff9d", "DOWNTREND": "#ff3a3a", "CHOP": "#ffb700"}.get(reg, "#ffb700")
    fed     = r.get("fed_stance", "NEUTRAL")
    fed_col = {"DOVISH": "#00ff9d", "HAWKISH": "#ff3a3a"}.get(fed, "#6a8fa8")

    # Bouw de hero-HTML volledig vooraf als één aaneengesloten string.
    # Functieresultaten met newlines (ring_svg) mogen NOOIT inline in een
    # multiline f-string zitten — Streamlit breekt de HTML dan per regel.
    v_rsi14   = f(r.get("spy_rsi14"), 1)
    v_vix_pct = f(r.get("vix_pct1y"), 0)
    v_sec_pos = str(r.get("sectors_positive", "—"))
    ring_mqs  = ring_svg(mqs, sc_color(mqs))
    ring_ews  = ring_svg(ews, sc_color(ews))

    hero_html = (
        '<div class="demo-wrap">'
        '<div class="demo-label">◈ LIVE VOORBEELD — BESLISSECTIE</div>'
        '<div class="hero-mini">'
        f'<div class="dec-card {dec_cls}" style="min-width:155px">'
        '<div class="dec-lbl">▸ SHOULD I TRADE TODAY?</div>'
        f'<div class="dec-val {dec_val}">{decision}</div>'
        f'<div class="dec-sub">{dec_txt}</div>'
        '</div>'
        '<div class="ring-card">'
        '<div class="ring-title">MARKET QUALITY SCORE</div>'
        + ring_mqs +
        '<div class="ring-sub">Gewogen 5-factor score</div>'
        '</div>'
        '<div class="ring-card">'
        '<div class="ring-title">EXECUTION WINDOW</div>'
        + ring_ews +
        '<div class="ring-sub">Setupkwaliteit huidig</div>'
        '</div>'
        f'<div class="regime-card">'
        '<div class="ring-title">REGIME &amp; CONTEXT</div>'
        f'<div class="regime-val" style="color:{reg_col}">{reg}</div>'
        f'<div class="regime-row"><span class="regime-k">RSI14</span><span class="regime-v">{v_rsi14}</span></div>'
        f'<div class="regime-row"><span class="regime-k">VIX %ile</span><span class="regime-v">{v_vix_pct}%</span></div>'
        f'<div class="regime-row"><span class="regime-k">Fed</span><span class="regime-v" style="color:{fed_col}">{fed}</span></div>'
        f'<div class="regime-row"><span class="regime-k">Sectoren+</span><span class="regime-v">{v_sec_pos}/11</span></div>'
        '</div>'
        '</div>'
        '</div>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)

    st.markdown("### ① Het YES / CAUTION / NO besluit")
    st.markdown(f"""
    Het grote woord linksboven is het eindbesluit. De kleur en de gloeiende bovenlijn geven
    het signaal:

    - 🟢 **YES** — Gunstige omgeving. Alle of de meeste factoren wijzen positief.
      Handel met volledige positiegrootte.
    - 🟡 **CAUTION** — Gemengde signalen. Alleen de sterkste setups, halfgrootte, strakke stops.
    - 🔴 **NO** — Ongunstige omgeving. Zijlijn houden.

    *Huidig signaal: **{decision}***
    """)

    st.markdown("### ② Market Quality Score (MQS)")
    st.markdown(f"""
    De linkerring toont de **MQS** — het gewogen totaalcijfer (0–100). Dit is de score die het
    YES/NO-besluit bepaalt. Groen boven 70, geel tussen 50 en 70, rood onder 50.

    De formule:

    ```
    MQS = (Volatiliteit × 25%) + (Trend × 20%) + (Breedte × 20%)
        + (Momentum × 25%) + (Macro × 10%)
    ```

    *Huidige MQS: **{mqs}***
    """)

    st.markdown("### ③ Execution Window Score")
    st.markdown(f"""
    De rechterring is een *aparte* score die meet hoe goed setups op dit moment uitwerken —
    ongeacht het YES/NO-signaal. Het is een aanvullende context, geen onderdeel van het besluit.

    Een MQS van 75 (YES) met een lage Execution Window (30) betekent: de omgeving is prima, maar
    breakouts falen op dit moment veel. Wees selectiever dan normaal.

    *Huidig: **{ews}***
    """)

    st.markdown(f"""
    ### ④ Regime & Context

    Vier kerncijfers in de rechterkaard:

    - **Regime:** UPTREND (SPY boven MA20 > MA50 > MA200), DOWNTREND, of CHOP. *Nu: {reg}*
    - **RSI14:** Relative Strength Index over 14 dagen. Boven 60 = kracht, onder 40 = zwakte,
      onder 30 = oversold.
    - **VIX %ile:** Hoe hoog is de VIX vergeleken met het afgelopen jaar? 80%ile = angst is hoog
      voor dit jaar.
    - **Fed:** Fed-stance afgeleid uit renteniveau en -richting. DOVISH = gunstig voor aandelen.
    - **Sectoren+:** Hoeveel van de 11 S&P-sectoren stegen de afgelopen 5 handelsdagen?
    """)


    # ════════════════════════════════════════════════════════════════════
    # SECTIE 4 — VIJF PANELEN
    # ════════════════════════════════════════════════════════════════════
    st.markdown("## 4 — De vijf detailpanelen")

    st.markdown("""
    Onder de beslissectie staan vijf panelen zij aan zij — één per factor. Elk paneel toont
    de ruwe getallen die de sub-score bepalen, plus een interpretatie-label.
    """)

    # Panels replica
    vix_v   = r.get("vix", 20) or 20
    vix_int = ("HEALTHY — LOW NOISE" if vix_v < 17 else
              "MODERATE — STANDARD" if vix_v < 22 else
              "ELEVATED — REDUCE SIZE" if vix_v < 28 else "EXTREME — RISK OFF")
    vix_ic  = "ih" if vix_v < 17 else ("in2" if vix_v < 22 else "ir")

    reg_int = ("BULLISH MA STACK" if reg == "UPTREND" else
              "BEARISH BREAKDOWN" if reg == "DOWNTREND" else "CHOPPY RANGE — NO TREND")
    reg_ic  = "ih" if reg == "UPTREND" else ("ir" if reg == "DOWNTREND" else "in2")

    ps_v    = r.get("sectors_positive", 5)
    br_int  = ("BROAD PARTICIPATION" if ps_v >= 7 else
              "MIXED — STAY SELECTIVE" if ps_v >= 5 else "NARROW / DETERIORATING")
    br_ic   = "ih" if ps_v >= 7 else ("in2" if ps_v >= 5 else "ir")

    spread  = r.get("sector_spread", 0) or 0
    mo_int  = "STRONG LEADERSHIP" if spread > 6 else ("MODERATE ROTATION" if spread > 3 else "FLAT — NO ENERGY")
    mo_ic   = "ih" if spread > 6 else ("in2" if spread > 3 else "ir")

    tnx_v   = r.get("tnx", 4.5) or 4.5
    mac_int = ("FAVORABLE — YIELDS LOW" if tnx_v < 4.0 else ("NEUTRAL" if tnx_v < 5.0 else "HAWKISH HEADWIND"))
    mac_ic  = "ih" if tnx_v < 4.0 else ("in2" if tnx_v < 5.0 else "ir")

    def score_badge(cat):
        s = scores.get(cat, {}).get("score", 50)
        return f'<span class="panel-score {ps_cls(s)}">{s:.0f}</span>'

    qqv = r.get("qqq_vs_ma50", 0) or 0
    mc  = r.get("mclellan_est", 0) or 0

    # ── Bereken alle conditionele klassen vooraf ─────────────────────────
    # Inline ternaries met aanhalingstekens in een HTML-attribuut breken
    # de Streamlit markdown-parser. Variabelen werken altijd correct.
    c_vix_sl = "mdn" if (r.get("vix_slope5d") or 0) > 0 else "mup"
    c_ma20   = "mup" if (r.get("spy_vs_ma20") or 0) > 0 else "mdn"
    c_ma50   = "mup" if (r.get("spy_vs_ma50") or 0) > 0 else "mdn"
    c_ma200  = "mup" if (r.get("spy_vs_ma200")or 0) > 0 else "mdn"
    c_qqv    = "mup" if qqv > 0 else "mdn"
    c_mc     = "mup" if mc  > 0 else "mdn"
    c_spy1d  = "mup" if (r.get("spy_1d_chg")  or 0) > 0 else "mdn"
    c_spy5d  = "mup" if (r.get("spy_5d_chg")  or 0) > 0 else "mdn"
    c_iwm    = "mup" if (r.get("iwm_vs_ma50") or 0) > 0 else "mdn"
    c_tnx    = "mdn" if (r.get("tnx_slope5d") or 0) > 0 else "mup"

    # ── Alle waarden als strings (vooraf geformatteerd) ──────────────────
    v_vix_sl  = f(r.get("vix_slope5d"), 3)
    v_vix_pct = f(r.get("vix_pct1y"), 1)
    v_vvix    = f(r.get("vvix"), 1)
    v_pc      = f(r.get("pc_ratio_est"), 2)
    v_spy     = f(r.get("spy_price"), 2)
    v_ma20    = f(r.get("spy_vs_ma20"), 2)
    v_ma50    = f(r.get("spy_vs_ma50"), 2)
    v_ma200   = f(r.get("spy_vs_ma200"), 2)
    v_qqv     = f(qqv, 2)
    v_rsi     = f(r.get("spy_rsi14"), 1)
    v_pctpos  = f(r.get("pct_sectors_pos"), 1)
    v_ad      = f(r.get("ad_ratio_est"), 2)
    v_nh      = str(r.get("nasdaq_nh_est", "—"))
    v_nl      = str(r.get("nasdaq_nl_est", "—"))
    v_mc      = f(mc, 1)
    v_pctma   = f(r.get("est_pct_above_50d"), 1)
    v_spy1d   = f(r.get("spy_1d_chg"), 2)
    v_spy5d   = f(r.get("spy_5d_chg"), 2)
    v_spread  = f(spread, 2)
    v_iwm     = f(r.get("iwm_vs_ma50"), 2)
    v_tnx     = f(tnx_v, 3)
    v_tnx_sl  = f(r.get("tnx_slope5d"), 4)
    v_dxy     = f(r.get("dxy"), 2)
    v_dxy_sl  = f(r.get("dxy_slope5d"), 3)
    ews_col   = sc_color(ews)

    panels_html = (
        '<div class="demo-wrap">'
        '<div class="demo-label">◈ LIVE VOORBEELD — VIJF DETAILPANELEN</div>'
        '<div class="panel-grid">'
    )
    panels_html += (
        f'<div class="panel">'
        f'<div class="panel-hdr"><span class="panel-title">⚡ VOLATILITY</span>{score_badge("volatility")}</div>'
        f'<div class="metric"><span class="mk">VIX Level</span><span class="mv {c_vix_sl}">{f(vix_v,2)}</span></div>'
        f'<div class="metric"><span class="mk">VIX 5d Slope</span><span class="mv">{v_vix_sl}</span></div>'
        f'<div class="metric"><span class="mk">VIX 1Y Pctile</span><span class="mv">{v_vix_pct}%</span></div>'
        f'<div class="metric"><span class="mk">VVIX</span><span class="mv">{v_vvix}</span></div>'
        f'<div class="metric"><span class="mk">P/C Ratio Est</span><span class="mv">{v_pc}</span></div>'
        f'<div class="interp {vix_ic}">{vix_int}</div></div>'
    )
    panels_html += (
        f'<div class="panel">'
        f'<div class="panel-hdr"><span class="panel-title">📈 TREND</span>{score_badge("trend")}</div>'
        f'<div class="metric"><span class="mk">SPY Price</span><span class="mv">{v_spy}</span></div>'
        f'<div class="metric"><span class="mk">vs MA20</span><span class="mv {c_ma20}">{v_ma20}%</span></div>'
        f'<div class="metric"><span class="mk">vs MA50</span><span class="mv {c_ma50}">{v_ma50}%</span></div>'
        f'<div class="metric"><span class="mk">vs MA200</span><span class="mv {c_ma200}">{v_ma200}%</span></div>'
        f'<div class="metric"><span class="mk">QQQ vs MA50</span><span class="mv {c_qqv}">{v_qqv}%</span></div>'
        f'<div class="metric"><span class="mk">RSI 14</span><span class="mv">{v_rsi}</span></div>'
        f'<div class="interp {reg_ic}">{reg_int}</div></div>'
    )
    panels_html += (
        f'<div class="panel">'
        f'<div class="panel-hdr"><span class="panel-title">🌊 BREADTH</span>{score_badge("breadth")}</div>'
        f'<div class="metric"><span class="mk">Sectors Pos</span><span class="mv">{ps_v}/11</span></div>'
        f'<div class="metric"><span class="mk">% Positive 5d</span><span class="mv">{v_pctpos}%</span></div>'
        f'<div class="metric"><span class="mk">A/D Ratio Est</span><span class="mv">{v_ad}</span></div>'
        f'<div class="metric"><span class="mk">NH / NL (NQ)</span>'
        f'<span class="mv"><span class="mup">{v_nh}</span>/<span class="mdn">{v_nl}</span></span></div>'
        f'<div class="metric"><span class="mk">McClellan Est</span><span class="mv {c_mc}">{v_mc}</span></div>'
        f'<div class="metric"><span class="mk">% &gt; MA50 Est</span><span class="mv">{v_pctma}%</span></div>'
        f'<div class="interp {br_ic}">{br_int}</div></div>'
    )
    panels_html += (
        f'<div class="panel">'
        f'<div class="panel-hdr"><span class="panel-title">🚀 MOMENTUM</span>{score_badge("momentum")}</div>'
        f'<div class="metric"><span class="mk">SPY 1d Chg</span><span class="mv {c_spy1d}">{v_spy1d}%</span></div>'
        f'<div class="metric"><span class="mk">SPY 5d Chg</span><span class="mv {c_spy5d}">{v_spy5d}%</span></div>'
        f'<div class="metric"><span class="mk">Sector Spread</span><span class="mv">{v_spread}%</span></div>'
        f'<div class="metric"><span class="mk">IWM vs MA50</span><span class="mv {c_iwm}">{v_iwm}%</span></div>'
        f'<div class="metric"><span class="mk">Exec Window</span>'
        f'<span class="mv" style="color:{ews_col}">{ews}/100</span></div>'
        f'<div class="interp {mo_ic}">{mo_int}</div></div>'
    )
    panels_html += (
        f'<div class="panel">'
        f'<div class="panel-hdr"><span class="panel-title">🏦 MACRO</span>{score_badge("macro")}</div>'
        f'<div class="metric"><span class="mk">10Y Yield</span><span class="mv {c_tnx}">{v_tnx}%</span></div>'
        f'<div class="metric"><span class="mk">Yield 5d Slope</span><span class="mv">{v_tnx_sl}</span></div>'
        f'<div class="metric"><span class="mk">DXY</span><span class="mv">{v_dxy}</span></div>'
        f'<div class="metric"><span class="mk">DXY 5d Slope</span><span class="mv">{v_dxy_sl}</span></div>'
        f'<div class="metric"><span class="mk">Fed Stance</span>'
        f'<span class="mv" style="color:{fed_col}">{fed}</span></div>'
        f'<div class="interp {mac_ic}">{mac_int}</div></div>'
    )
    panels_html += '</div></div>'
    st.markdown(panels_html, unsafe_allow_html=True)

    st.markdown("### Hoe lees je een paneel?")

    st.markdown("""
    Elk paneel heeft dezelfde structuur:

    1. **Titel** (links) — de factoornaam
    2. **Sub-score badge** (rechts) — de berekende score voor deze factor (0–100), groen/geel/rood
    3. **Metriekenlijst** — de ruwe getallen die de score bepalen, met kleurpijlen
    4. **Interpretatie-label** onderaan — een tekstvertaling van de score naar marktomschrijving

    De pijlen (↑ groen, ↓ rood) geven de richting van de 5-daagse verandering aan —
    *maar bij VIX is dit omgekeerd:* een dalende VIX is goed nieuws.
    """)

    st.markdown("### De vijf metrics per paneel uitgelegd")

    for title, items in [
        ("⚡ Volatiliteit", [
            ("VIX Level", "De CBOE Volatility Index. Meet hoeveel beweeglijkheid de markt verwacht in de komende 30 dagen, afgeleid uit optieprijzen op de S&P 500. Onder 17 = rust. Boven 28 = stress."),
            ("VIX 5d Slope", "Stijgt of daalt de VIX de laatste 5 handelsdagen? Positief = toenemende angst (slecht). Negatief = kalmering (goed)."),
            ("VIX 1Y Percentiel", "Hoe hoog is de huidige VIX vergeleken met alle dagwaarden van het afgelopen jaar? Een percentiel van 80% betekent dat de VIX 80% van het jaar lager was — relatief hoge angst."),
            ("VVIX", "Volatiliteit van de VIX zelf — hoe grillig beweegt de angstindex? Boven 120 duidt op extreem onzekere marktomstandigheden."),
            ("P/C Ratio Est", "Geschatte Put/Call ratio: verhouding beschermende puts t.o.v. speculatieve calls. Boven 1.0 = meer vraag naar bescherming = bearish sentiment."),
        ]),
        ("📈 Trend", [
            ("SPY Price", "Huidige slotkoers van SPY (S&P 500 ETF). Referentie voor de MA-berekeningen."),
            ("vs MA20/50/200", "Percentage waarmee SPY boven (+) of onder (−) zijn 20-, 50- en 200-daags voortschrijdend gemiddelde staat. Positief = price strength. Alle drie positief met MA20 > MA50 > MA200 = bullish stapel."),
            ("QQQ vs MA50", "Staat de Nasdaq-100 (tech-zwaar) boven zijn 50-daags gemiddelde? Tech leidt de markt vaak — als QQQ achterblijft, is de brede markttrend zwakker dan hij lijkt."),
            ("RSI 14", "Relative Strength Index. Meet of SPY oververkocht (<30) of overkocht (>70) is op basis van de laatste 14 handelsdagen. Waarden tussen 40–60 zijn neutraal."),
        ]),
        ("🌊 Marktbreedte", [
            ("Sectors Positive", "Van de 11 S&P-sectoren: hoeveel stegen de afgelopen 5 handelsdagen? Boven 7 = brede participatie (goed). Onder 4 = smalle markt (gevaarlijk)."),
            ("A/D Ratio Est", "Schatting van de Advance/Decline-ratio — stijgende vs. dalende sectoren. Boven 2.0 is positief. Onder 0.8 is negatief. Gebaseerd op de 11 sector-ETFs als proxy voor de volledige markt."),
            ("McClellan Est", "Schatting van de McClellan Oscillator: het verschil tussen de 10-daagse en 19-daagse exponentiële gemiddelden van SPY, geschaald. Positief = brede markt versnelt. Negatief = vertraging."),
            ("NH/NL (NQ)", "Schatting nieuwe highs (groen) vs. nieuwe lows (rood) op de Nasdaq. Een groot overschot aan new highs is bullish. Hier geschat op basis van sectorparticipatie."),
            ("% > MA50 Est", "Schatting van het percentage aandelen dat boven zijn 50-daags gemiddelde staat. Boven 60% = gezonde markt. Onder 40% = verzwakte markt."),
        ]),
        ("🚀 Momentum", [
            ("SPY 1d / 5d Chg", "Hoeveel procent bewoog SPY de afgelopen dag en de afgelopen week? Positief = wind mee. Een sterke 5-daagse stijging geeft +15 punten in de score."),
            ("Sector Spread", "Verschil in 5-daags rendement tussen de best en slechtst presterende sector. Een grote spreiding (>8%) betekent duidelijke rotatie en energie in de markt — gunstig. Vlak (<2%) = richtingloze markt."),
            ("IWM vs MA50", "Small caps (Russell 2000) t.o.v. hun 50-daags gemiddelde. Small caps reageren sterker op economisch sentiment. Als IWM achterloopt op large caps, is er ongezonde divergentie."),
            ("Exec Window", "De Execution Window Score — zie sectie 3."),
        ]),
        ("🏦 Macro", [
            ("10Y Yield", "De 10-jaars Amerikaanse staatsrente. Hoge rentes (>5%) zijn duur voor bedrijven en concurreren met aandelen als beleggingsoptie. De kleur toont de richting van de 5-daagse slope: dalend = goed (groen), stijgend = slecht (rood)."),
            ("Yield 5d Slope", "Hoe snel beweegt de 10-jaarsrente? Een snelle stijging (>0.03 per dag) geeft −10 punten in de macro-score."),
            ("DXY", "De Dollar Index — gewogen gemiddelde van de dollar t.o.v. zes andere valuta's. Een dalende dollar is doorgaans gunstig voor multinationals en grondstoffen."),
            ("Fed Stance", "DOVISH = Fed verlaagt rente of is soepel (goed voor aandelen). HAWKISH = Fed verhoogt of dreigt te verhogen (slecht). NEUTRAL = tussenpositie. Afgeleid uit het renteniveau en de 5-daagse slope."),
        ]),
    ]:
        st.markdown(f"**{title}**")
        for metric, uitleg in items:
            st.markdown(f"- **`{metric}`** — {uitleg}")


    # ════════════════════════════════════════════════════════════════════
    # SECTIE 5 — SECTOR HEATMAP
    # ════════════════════════════════════════════════════════════════════
    st.markdown("## 5 — De sector heatmap")

    st.markdown("""
    De heatmap toont alle 11 S&P-sectoren op een horizontale as, gesorteerd van beste naar
    slechtste 5-daags rendement. De middenlijn is nul.
    """)

    # Heatmap replica
    SECTOR_NAMES = {"XLK":"Tech","XLF":"Fin","XLE":"Energy","XLV":"Health","XLI":"Indust",
                    "XLY":"Discret","XLP":"Staples","XLU":"Util","XLB":"Material","XLRE":"RE","XLC":"Comms"}
    ss_sorted = sorted(sc2.items(), key=lambda x: x[1], reverse=True)
    maxabs = max(abs(v) for v in sc2.values()) if sc2 else 5.0; maxabs = max(maxabs, 0.01)

    bars_html = ""
    for e, cv in ss_sorted:
        pct  = min(abs(cv) / maxabs * 47, 47)
        vc   = "mup" if cv > 0 else ("mdn" if cv < 0 else "")
        fill = (f"position:absolute;top:0;bottom:0;left:50%;width:{pct:.1f}%;"
                f"background:{'linear-gradient(90deg,#005533,#00ff9d)' if cv >= 0 else 'linear-gradient(270deg,#5a0000,#ff3a3a)'}"
                if cv >= 0 else
                f"position:absolute;top:0;bottom:0;right:50%;width:{pct:.1f}%;"
                f"background:linear-gradient(270deg,#5a0000,#ff3a3a)")
        bars_html += f"""
    <div class="sbr">
      <span class="sl">{e}</span>
      <div class="st2"><div class="sm2"></div><div style="{fill}"></div></div>
      <span class="sp2 {vc}">{cv:+.2f}%</span>
    </div>"""

    st.markdown(f"""
    <div class="demo-wrap">
      <div class="demo-label">◈ LIVE VOORBEELD — SECTOR HEATMAP (5-DAAGS RENDEMENT)</div>
      <div class="sect-bar-wrap">{bars_html}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Hoe lees je de heatmap?**

    - Groene balken naar rechts = sector steeg de afgelopen 5 handelsdagen
    - Rode balken naar links = sector daalde
    - De lengte is relatief: de langste balk is de grootste absolute verandering

    **Wat betekent de sectorrotatie?**

    De samenstelling van welke sectoren leiden en welke achterblijven vertelt een verhaal:

    - **Defensief leiderschap** (XLU Utilities, XLP Staples, XLV Health voorop) met dalende groeisectoren
      = risicoaversie. Beleggers vluchten naar veiligheid. *Slechte omgeving voor actief handelen.*

    - **Groei-leiderschap** (XLK Tech, XLY Discretionary, XLF Financials voorop)
      = risicoappetijt. Markt zoekt rendement. *Gunstige omgeving.*

    - **Energie en materialen** leiden = vaak een commodities-gedreven markt, soms inflatiegedreven.
      Beperktere relevantie voor tech-aandelen.
    """)

    top3   = r.get("top3_sectors", [])
    bot3   = r.get("bottom3_sectors", [])
    if top3 and bot3:
        t_str = ", ".join(f"{e} ({SECTOR_NAMES.get(e,e)}) {v:+.1f}%" for e,v in top3)
        b_str = ", ".join(f"{e} ({SECTOR_NAMES.get(e,e)}) {v:+.1f}%" for e,v in bot3)
        st.markdown(f"""
    <div class="callout callout-blue">
      ◈ Huidig: leiders = {t_str}<br/>
      ◈ Achterblijvers = {b_str}
    </div>
    """, unsafe_allow_html=True)


    # ════════════════════════════════════════════════════════════════════
    # SECTIE 6 — SCORE BREAKDOWN
    # ════════════════════════════════════════════════════════════════════
    st.markdown("## 6 — De score breakdown")

    st.markdown("""
    Het middelste paneel van de drie onderste blokken toont precies hoe de MQS is opgebouwd.
    Voor elke factor zie je drie dingen: de sub-score als balk, het gewicht, en de gewogen bijdrage.
    """)

    sb_html = ""
    for key, label in [("volatility","Volatiliteit"),("trend","Trend"),("breadth","Breedte"),
                      ("momentum","Momentum"),("macro","Macro")]:
        s  = scores.get(key, {}).get("score", 50)
        wt = scores.get(key, {}).get("weight", .2)
        c  = sc_color(s)
        sb_html += f"""
    <div style="display:flex;align-items:center;margin-bottom:8px;gap:8px;font-family:var(--mono);font-size:11px">
      <span style="color:var(--t2);width:80px">{label}</span>
      <span style="color:var(--t3);width:28px">{wt*100:.0f}%</span>
      <div style="flex:1;height:9px;background:var(--bg);border:1px solid var(--bdr);position:relative">
        <div style="position:absolute;top:0;left:0;height:100%;width:{s}%;background:{c};opacity:.75"></div>
      </div>
      <span style="color:{c};width:28px;text-align:right">{s:.0f}</span>
      <span style="color:var(--t3);width:42px;text-align:right">=&nbsp;{s*wt:.1f}</span>
    </div>"""

    sb_html += f"""
    <div style="border-top:1px solid var(--bdr2);margin-top:10px;padding-top:8px;
      display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:12px">
      <span style="color:var(--t1);width:80px;font-weight:600">TOTAAL</span>
      <span style="width:28px"></span>
      <div style="flex:1;height:10px;background:var(--bg);border:1px solid var(--bdr2);position:relative">
        <div style="position:absolute;top:0;left:0;height:100%;width:{mqs}%;background:{sc_color(mqs)}"></div>
      </div>
      <span style="color:{sc_color(mqs)};width:28px;text-align:right;font-size:14px">{mqs}</span>
      <span style="width:42px"></span>
    </div>"""

    st.markdown(f"""
    <div class="demo-wrap">
      <div class="demo-label">◈ LIVE VOORBEELD — SCORE BREAKDOWN</div>
      <div style="padding:16px 20px">{sb_html}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Hoe lees je de breakdown?**

    Elke rij: `factor  gewicht  [balk]  sub-score  =  bijdrage`

    De bijdrage is sub-score × gewicht. De som van alle bijdragen is de MQS.

    Dit panel is nuttig om te begrijpen *waarom* het signaal YES of NO is. Als de MQS net
    onder de YES-drempel zit, zie je direct welke factor het meest drukt.
    """)


    # ════════════════════════════════════════════════════════════════════
    # SECTIE 7 — TERMINAL ANALYSIS
    # ════════════════════════════════════════════════════════════════════
    st.markdown("## 7 — De terminal analysis")

    st.markdown("""
    Het rechter-onderpaneel geeft een automatisch gegenereerde tekstsamenvatting in
    terminal-stijl. Dit is geen AI — het zijn vaste zinnen die worden ingevuld op basis
    van de score-uitkomsten.
    """)

    summary = r.get("summary", "")
    lines   = [l.strip() for l in summary.split(". ") if l.strip()]
    term_html = "".join(
        f'<div style="margin-bottom:4px"><span style="color:var(--t3)">▸▸▸</span> '
        f'<span style="color:var(--gd)">{l}.</span></div>'
        for l in lines
    )

    st.markdown(f"""
    <div class="demo-wrap">
      <div class="demo-label">◈ LIVE VOORBEELD — TERMINAL ANALYSIS</div>
      <div style="padding:16px 20px;font-family:var(--mono);font-size:11px;line-height:1.9">
        {term_html}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    De samenvatting bestaat altijd uit dezelfde zes onderdelen in vaste volgorde:

    1. **Regime-omschrijving** — trend van SPY t.o.v. zijn gemiddelden
    2. **Volatiliteitsoordeel** — VIX-niveau vertaald naar risicoadvies
    3. **Breedte** — hoeveel sectoren doen mee?
    4. **Leiders** — welke sectoren stegen het meest?
    5. **Execution window** — werken setups op dit moment?
    6. **Actie-aanbeveling** — wat te doen (volledige positie / halfgrootte / zijlijn)
    """)


    # ════════════════════════════════════════════════════════════════════
    # SECTIE 8 — FOMC
    # ════════════════════════════════════════════════════════════════════
    st.markdown("## 8 — De FOMC-alertbanner")

    st.markdown("""
    Als er binnen 3 handelsdagen een FOMC-vergadering is (Fed rentebesluit), verschijnt er
    een oranje knipperende banner boven de beslissectie:

    - **Vandaag** → `⚡ FOMC DECISION TODAY — Elevated volatility expected`
    - **Morgen** → `⚠ FOMC TOMORROW — Consider reducing overnight exposure`
    - **2–3 dagen** → `📅 FOMC in N days — Monitor risk appetite closely`

    FOMC-dagen zijn bijzonder: de markt beweegt sterk en onvoorspelbaar rond het moment van
    aankondiging (doorgaans 14:00 ET). Zelfs een YES-signaal verdient extra voorzichtigheid
    als er een FOMC in aantocht is.

    **Wanneer is de volgende?** Het dashboard bevat een vaste lijst met FOMC-data voor
    2025–2026. Deze lijst staat in `app.py` in de functie `fomc_alert()` en is eenvoudig
    uit te breiden.
    """)

    st.markdown("""
    <div class="callout callout-yellow">
      ◈ Vuistregel: geen nieuwe swing-posities openen in de 24 uur voor een FOMC-besluit.
      Bestaande posities beschermen met bredere stops of gedeeltelijk sluiten.
    </div>
    """, unsafe_allow_html=True)


    # ════════════════════════════════════════════════════════════════════
    # SECTIE 9 — WORKFLOW
    # ════════════════════════════════════════════════════════════════════
    st.markdown("## 9 — Aanbevolen dagelijkse workflow")

    st.markdown("""
    Hoe gebruik je dit dashboard in de praktijk? Een voorstel:
    """)

    for stap, titel, tekst in [
        ("①", "Check het signaal (30 seconden)",
        "Open app.py. Kijk naar het grote YES/CAUTION/NO-besluit. Als het NO is: sluit de tab en doe iets anders. "
        "Je hoeft de rest niet te lezen."),
        ("②", "Lees de regime-kaart (1 minuut)",
        "Is het UPTREND, DOWNTREND of CHOP? In een UPTREND werken long-setups beter. "
        "In CHOP werken breakouts slechter — overweeg mean-reversion setups. In DOWNTREND "
        "alleen shorts of zijlijn."),
        ("③", "Check de sector heatmap (2 minuten)",
        "Welke sectoren leiden? Als je een setup hebt in een aandeel uit een achterblijvende sector, "
        "is de kans kleiner dat de setup werkt. Zoek setups in leidende sectoren."),
        ("④", "Controleer de Execution Window (10 seconden)",
        "Een YES-signaal met Execution Window onder 40 betekent: omgeving is prima maar "
        "setups falen nu veel. Wees extra selectief. Alleen de allersterkste setups."),
        ("⑤", "Check de FOMC-banner",
        "Staat er een FOMC-alert? Pas je positiegrootte aan of open geen nieuwe posities."),
    ]:
        st.markdown(f"""
    <div class="num-card">
      <div class="num-n">{stap}</div>
      <div class="num-body">
        <div class="num-title">{titel}</div>
        <div class="num-text">{tekst}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


    # ════════════════════════════════════════════════════════════════════
    # DISCLAIMER
    # ════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="disclaimer">
      ⚠ EDUCATIEF GEBRUIK ALLEEN — GEEN BELEGGINGSADVIES.<br/>
      Dit dashboard en deze handleiding zijn uitsluitend bedoeld voor educatieve en informatieve
      doeleinden. Niets hierin vormt beleggingsadvies of een aanbeveling om te kopen of verkopen.
      Handel altijd op basis van eigen analyse en risicotolerantie. Verleden resultaten bieden
      geen garantie voor de toekomst.
    </div></div>
    """, unsafe_allow_html=True)


def main():
    uitleg_app()



if __name__ == "__main__":
    main()
