"""
Should I Be Trading? — Bloomberg Terminal-Style Market Dashboard
A live, auto-refreshing market environment analyzer for swing traders.

Usage:
  pip install streamlit yfinance pandas numpy
  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from theme import apply_theme

def dashboard():
  try:
      st.set_page_config(page_title="Should I Be Trading?", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")
  except:
      pass
  apply_theme()
 
  # ── GEDEELDE MODULES ────────────────────────────────────────────────
  from scoring import SECTORS, compute_scores, THRESHOLDS
  from market_data import get_market_data


  @st.cache_data(ttl=30)
  def build_analysis(mode="swing"):
      return get_market_data(mode=mode)


  def fomc_alert():
      dates=["2025-03-19","2025-05-07","2025-06-18","2025-07-30","2025-09-17","2025-10-29","2025-12-10",
            "2026-01-28","2026-03-18","2026-04-29","2026-06-17"]
      today=datetime.now().date()
      for s in dates:
          d=(datetime.strptime(s,"%Y-%m-%d").date()-today).days
          if d==0: return f"⚡ FOMC DECISION TODAY ({s}) — Elevated volatility expected"
          if d==1: return f"⚠ FOMC TOMORROW ({s}) — Consider reducing overnight exposure"
          if 1<d<=3: return f"📅 FOMC in {d} days ({s}) — Monitor risk appetite closely"
      return None


  # ── HELPERS ──────────────────────────────────────────────────────
  def f(v,dec=2,sx=""): return "N/A" if v is None else f"{v:.{dec}f}{sx}"
  def arr(v,inv=False):
      if v is None: return "→","mn2"
      p=v>.005; n=v<-.005
      if inv: p,n=n,p
      return ("↑","mu") if p else (("↓","md") if n else ("→","mn2"))
  def sc(s): return "#00ff9d" if s>=70 else("#ffb700" if s>=50 else "#ff3a3a")
  def bc(s): return "bg" if s>=70 else("ba" if s>=50 else "br2")
  def ic(l):
      l2=l.lower()
      if any(k in l2 for k in ["bull","healthy","strong","broad","favor","low noise","flat below"]): return "ih"
      if any(k in l2 for k in ["bear","risk","danger","extreme","narrow","poor","fail","extreme"]): return "ir"
      return "in"
  def ring(score,color,title,sub):
      R=47; C=2*3.14159*R; fl=C*score/100; gp=C-fl
      return f"""<div class="sc"><div class="rc">
  <svg class="rs" viewBox="0 0 108 108" width="108" height="108">
    <circle cx="54" cy="54" r="{R}" fill="none" stroke="var(--bdr)" stroke-width="8"/>
    <circle cx="54" cy="54" r="{R}" fill="none" stroke="{color}" stroke-width="8"
      stroke-linecap="round" stroke-dasharray="{fl:.1f} {gp:.1f}"/>
  </svg><div class="rt" style="color:{color}">{score}</div></div>
  <div class="sct">{title}</div><div class="scs">{sub}</div></div>"""


  # ── SESSION STATE ────────────────────────────────────────────────
  if "last_refresh" not in st.session_state: st.session_state.last_refresh=time.time()
  if "mode" not in st.session_state: st.session_state.mode="swing"

  # ── CONTROLS ─────────────────────────────────────────────────────
  c1,c2,c3=st.columns([2,1,6])
  with c1:
      mode=st.selectbox("m",["swing","day"],index=0 if st.session_state.mode=="swing" else 1,
                        label_visibility="collapsed",
                        format_func=lambda x:f"{'🔄 SWING' if x=='swing' else '⚡ DAY'} TRADING MODE")
      st.session_state.mode=mode
  with c2:
      import random
      if st.button("⟳ REFRESH", key=f"uitleg{random.randint(1,9999)}"):
          st.cache_data.clear(); st.session_state.last_refresh=time.time(); st.rerun()

  # ── FETCH ─────────────────────────────────────────────────────────
  with st.spinner(""):
      r=build_analysis(mode=st.session_state.mode)

  decision=r.get("decision","CAUTION"); mqs=r.get("market_quality_score",50)
  ews=r.get("execution_window_score",50); scores=r.get("scores",{})
  is_live=r.get("is_live",False); sc2=r.get("sector_chg",{})
  now=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"); elapsed=int(time.time()-st.session_state.last_refresh)

  # ── TICKER ────────────────────────────────────────────────────────
  def build_ticker():
      pairs=[("SPY",r.get("spy_price"),r.get("spy_1d_chg")),("QQQ",r.get("qqq_price"),r.get("qqq_1d_chg")),
            ("IWM",r.get("iwm_price"),None),("VIX",r.get("vix"),r.get("vix_slope5d")),
            ("TNX",r.get("tnx"),r.get("tnx_slope5d")),("DXY",r.get("dxy"),r.get("dxy_slope5d"))]
      for e in SECTORS: pairs.append((e,None,sc2.get(e,0)))
      h=""
      for sym,price,chg in pairs:
          c=(("up" if (chg or 0)>0 else "dn") if chg is not None else "fl")
          mk="▲" if c=="up" else("▼" if c=="dn" else "—")
          ps=f" {price:.2f}" if price else ""
          cs=f" {chg:+.2f}%" if chg is not None else ""
          h+=f'<span class="ti"><span class="sy">{sym}</span>{ps}<span class="{c}"> {mk}{cs}</span></span>'
      return f'<div class="tbr"><div class="tsc">{h*2}</div></div>'
  st.markdown(build_ticker(),unsafe_allow_html=True)

  # ── HEADER ───────────────────────────────────────────────────────
  badge='<span class="lb">● LIVE</span>' if is_live else '<span class="db">◌ DEMO</span>'
  st.markdown(f"""
  <div class="hdr">
    <div><div class="ht">◈ Should I Be Trading?</div>
    <div class="hs">MARKET INTELLIGENCE TERMINAL &nbsp;·&nbsp; {mode.upper()} MODE &nbsp;·&nbsp;
      {'LIVE DATA — Yahoo Finance / yfinance' if is_live else '⚠ DEMO MODE — pip install yfinance for live data'}</div></div>
    <div style="display:flex;align-items:center;gap:12px">
      {badge}<span class="ts">UPDATED {elapsed}s AGO &nbsp;·&nbsp; {now}</span></div>
  </div>""",unsafe_allow_html=True)

  # ── FOMC ALERT ───────────────────────────────────────────────────
  fa=fomc_alert()
  if fa: st.markdown(f'<div class="al">{fa}</div>',unsafe_allow_html=True)

  st.markdown('<div class="body">',unsafe_allow_html=True)

  # ── HERO ─────────────────────────────────────────────────────────
  ddesc={"YES":"Full size. All factors aligned. Favor A+ setups with clear catalysts.",
        "CAUTION":"Half size only. Mixed signals. A+ setups, tight stops, stay selective.",
        "NO":"Stand aside. Conditions unfavorable. Preserve capital. Build watchlist."}
  reg=r.get("regime","CHOP")
  rc2={"UPTREND":"#00ff9d","DOWNTREND":"#ff3a3a","CHOP":"#ffb700"}.get(reg,"#ffb700")
  dc2={"YES":"dy","CAUTION":"dc","NO":"dn2"}[decision]
  hc2={"YES":"hy","CAUTION":"hc","NO":"hn"}[decision]
  fed=r.get("fed_stance","NEUTRAL")
  fc=("mu" if fed=="DOVISH" else("md" if fed=="HAWKISH" else "mn2"))

  st.markdown(f"""
  <div class="hero">
    <div class="hdec {hc2}">
      <div class="dl">▸ SHOULD I TRADE TODAY?</div>
      <div class="db2 {dc2}">{decision}</div>
      <div class="dd">{ddesc[decision]}</div>
    </div>
    <div class="sr">
      {ring(mqs,sc(mqs),"MARKET QUALITY SCORE","Weighted 5-factor composite")}
      {ring(ews,sc(ews),"EXECUTION WINDOW","Breakout follow-through quality")}
      <div class="sc" style="align-items:flex-start;text-align:left">
        <div class="sct" style="margin-bottom:7px">REGIME &amp; CONTEXT</div>
        <div style="font-size:25px;font-weight:700;color:{rc2};margin-bottom:9px;font-family:var(--sans)">{reg}</div>
        <div style="font-size:9.5px;color:var(--t2);line-height:2.1">
          RSI14: <span style="color:var(--t1)">{f(r.get('spy_rsi14'),1)}</span><br>
          VIX %ile: <span style="color:var(--t1)">{f(r.get('vix_pct1y'),0)}%</span><br>
          Fed: <span class="{fc}">{fed}</span><br>
          Sectors+: <span style="color:var(--t1)">{r.get('sectors_positive','—')}/11</span>
        </div>
      </div>
    </div>
  </div>""",unsafe_allow_html=True)

  # ── FIVE PANELS ──────────────────────────────────────────────────
  def pb(cat):
      s=scores.get(cat,{}).get("score",50); return f'<span class="pb {bc(s)}">{s:.0f}</span>'

  vv=r.get("vix",20) or 20
  vi="HEALTHY — LOW NOISE" if vv<17 else("MODERATE — STANDARD" if vv<22 else("ELEVATED — REDUCE SIZE" if vv<28 else "EXTREME — RISK OFF"))
  vd2,vdc=arr(r.get("vix_slope5d"),inv=True)

  spy_reg=r.get("regime","CHOP")
  ti2="BULLISH MA STACK" if spy_reg=="UPTREND" else("BEARISH BREAKDOWN" if spy_reg=="DOWNTREND" else "CHOPPY RANGE — NO TREND")
  ps2=r.get("sectors_positive",5)
  bi="BROAD PARTICIPATION" if ps2>=7 else("MIXED — STAY SELECTIVE" if ps2>=5 else "NARROW / DETERIORATING")
  spread=r.get("sector_spread",0)
  mi="STRONG LEADERSHIP" if spread>6 else("MODERATE ROTATION" if spread>3 else "FLAT — NO ENERGY")
  tnv=r.get("tnx",4.5) or 4.5
  maci="FAVORABLE — YIELDS LOW" if tnv<4.0 else("NEUTRAL" if tnv<5.0 else "HAWKISH HEADWIND")

  ad_a,ad_c=arr(r.get("ad_ratio_est",1)-1)
  vd_a,vd_c=arr(r.get("vix_slope5d"),inv=True)
  tn_a,tn_c=arr(r.get("tnx_slope5d"),inv=True)
  dx_a,dx_c=arr(r.get("dxy_slope5d"))
  v20a,v20c=arr(r.get("spy_vs_ma20"))
  v50a,v50c=arr(r.get("spy_vs_ma50"))
  v200a,v200c=arr(r.get("spy_vs_ma200"))
  s5a,s5c=arr(r.get("spy_5d_chg"))
  iwa,iwc=arr(r.get("iwm_vs_ma50"))

  qqv=r.get("qqq_vs_ma50",0); qqc="mu" if qqv>0 else "md"
  mc=r.get("mclellan_est",0); mcc="mu" if mc>0 else "md"
  s1c="mu" if r.get("spy_1d_chg",0)>0 else "md"

  st.markdown(f"""
  <div class="pg">
  <div class="pn">
    <div class="ph"><span class="pt">⚡ Volatility</span>{pb('volatility')}</div>
    <div class="mr"><span class="mn">VIX Level</span><span class="mv {vd_c}">{f(r.get('vix'),2)} {vd_a}</span></div>
    <div class="mr"><span class="mn">VIX 5d Slope</span><span class="mv">{f(r.get('vix_slope5d'),3)}</span></div>
    <div class="mr"><span class="mn">VIX 1Y Pctile</span><span class="mv">{f(r.get('vix_pct1y'),1)}%</span></div>
    <div class="mr"><span class="mn">VVIX</span><span class="mv">{f(r.get('vvix'),1)}</span></div>
    <div class="mr"><span class="mn">P/C Ratio Est</span><span class="mv">{f(r.get('pc_ratio_est'),2)}</span></div>
    <div class="ip {ic(vi)}">{vi}</div>
  </div>
  <div class="pn">
    <div class="ph"><span class="pt">📈 Trend</span>{pb('trend')}</div>
    <div class="mr"><span class="mn">SPY Price</span><span class="mv">{f(r.get('spy_price'),2)}</span></div>
    <div class="mr"><span class="mn">vs MA20</span><span class="mv {v20c}">{f(r.get('spy_vs_ma20'),2)}% {v20a}</span></div>
    <div class="mr"><span class="mn">vs MA50</span><span class="mv {v50c}">{f(r.get('spy_vs_ma50'),2)}% {v50a}</span></div>
    <div class="mr"><span class="mn">vs MA200</span><span class="mv {v200c}">{f(r.get('spy_vs_ma200'),2)}% {v200a}</span></div>
    <div class="mr"><span class="mn">QQQ vs MA50</span><span class="mv {qqc}">{f(r.get('qqq_vs_ma50'),2)}%</span></div>
    <div class="mr"><span class="mn">RSI 14</span><span class="mv">{f(r.get('spy_rsi14'),1)}</span></div>
    <div class="ip {ic(ti2)}">{ti2}</div>
  </div>
  <div class="pn">
    <div class="ph"><span class="pt">🌊 Breadth</span>{pb('breadth')}</div>
    <div class="mr"><span class="mn">Sectors Pos</span><span class="mv">{r.get('sectors_positive','—')}/11</span></div>
    <div class="mr"><span class="mn">% Positive 5d</span><span class="mv">{f(r.get('pct_sectors_pos'),1)}%</span></div>
    <div class="mr"><span class="mn">A/D Ratio Est</span><span class="mv {ad_c}">{f(r.get('ad_ratio_est'),2)} {ad_a}</span></div>
    <div class="mr"><span class="mn">NH / NL (NQ)</span><span class="mv"><span class="mu">{r.get('nasdaq_nh_est','—')}</span>/<span class="md">{r.get('nasdaq_nl_est','—')}</span></span></div>
    <div class="mr"><span class="mn">McClellan Est</span><span class="mv {mcc}">{f(r.get('mclellan_est'),1)}</span></div>
    <div class="mr"><span class="mn">% > MA50 Est</span><span class="mv">{f(r.get('est_pct_above_50d'),1)}%</span></div>
    <div class="ip {ic(bi)}">{bi}</div>
  </div>
  <div class="pn">
    <div class="ph"><span class="pt">🚀 Momentum</span>{pb('momentum')}</div>
    <div class="mr"><span class="mn">SPY 1d Chg</span><span class="mv {s1c}">{f(r.get('spy_1d_chg'),2)}%</span></div>
    <div class="mr"><span class="mn">SPY 5d Chg</span><span class="mv {s5c}">{f(r.get('spy_5d_chg'),2)}% {s5a}</span></div>
    <div class="mr"><span class="mn">Sector Spread</span><span class="mv">{f(spread,2)}%</span></div>
    <div class="mr"><span class="mn">IWM vs MA50</span><span class="mv {iwc}">{f(r.get('iwm_vs_ma50'),2)}% {iwa}</span></div>
    <div class="mr"><span class="mn">Exec Window</span><span class="mv" style="color:{sc(ews)}">{ews}/100</span></div>
    <div class="ip {ic(mi)}">{mi}</div>
  </div>
  <div class="pn">
    <div class="ph"><span class="pt">🏦 Macro</span>{pb('macro')}</div>
    <div class="mr"><span class="mn">10Y Yield</span><span class="mv {tn_c}">{f(r.get('tnx'),3)}% {tn_a}</span></div>
    <div class="mr"><span class="mn">Yield 5d Slope</span><span class="mv">{f(r.get('tnx_slope5d'),4)}</span></div>
    <div class="mr"><span class="mn">DXY</span><span class="mv {dx_c}">{f(r.get('dxy'),2)} {dx_a}</span></div>
    <div class="mr"><span class="mn">DXY 5d Slope</span><span class="mv">{f(r.get('dxy_slope5d'),3)}</span></div>
    <div class="mr"><span class="mn">Fed Stance</span><span class="mv {fc}">{fed}</span></div>
    <div class="ip {ic(maci)}">{maci}</div>
  </div>
  </div>""",unsafe_allow_html=True)

  # ── BOTTOM ROW ───────────────────────────────────────────────────
  maxabs=max(abs(v) for v in sc2.values()) if sc2 else 5.; maxabs=max(maxabs,.01)
  sh=""
  for e,name in SECTORS.items():
      cv=sc2.get(e,0); pct=min(abs(cv)/maxabs*47,47)
      vc="mu" if cv>0 else("md" if cv<0 else "mn2")
      fill_style=f"position:absolute;top:0;bottom:0;{'left:50%' if cv>=0 else f'right:50%;'};width:{pct:.1f}%;background:{'linear-gradient(90deg,#005533,#00ff9d)' if cv>=0 else 'linear-gradient(90deg,#ff3a3a,#5a0000)'}"
      sh+=f"""<div class="sbr"><span class="sl2">{e}</span>
  <div class="st"><div class="sm"></div><div style="{fill_style}"></div></div>
  <span class="sp {vc}">{cv:+.2f}%</span></div>"""

  sb=""
  for key,label in [("volatility","Volatility"),("trend","Trend"),("breadth","Breadth"),("momentum","Momentum"),("macro","Macro")]:
      s3=scores.get(key,{}).get("score",50); wt=scores.get(key,{}).get("weight",.2); c=sc(s3)
      sb+=f"""<div class="sbrow"><span class="sbl">{label}</span><span class="sbw">{wt*100:.0f}%</span>
  <div class="sbt"><div class="sbf" style="width:{s3}%;background:{c};opacity:.75"></div></div>
  <span class="sbv" style="color:{c}">{s3:.0f}</span></div>"""

  sm=r.get("summary","")
  th2="".join(f'<span class="tp">&gt;&gt;&gt;</span> {l.strip()}.<br>' for l in sm.split(". ") if l.strip())

  ty=75 if mode=="day" else 80; tc2=55 if mode=="day" else 60
  st.markdown(f"""
  <div class="br">
  <div class="bp"><div class="bpt">◈ Sector Heatmap — 5-Day %</div>{sh}</div>
  <div class="bp">
    <div class="bpt">◈ Score Breakdown</div>{sb}
    <div style="border-top:1px solid var(--bdr2);margin-top:9px;padding-top:7px">
      <div class="sbrow"><span class="sbl" style="color:var(--t1)">TOTAL</span><span class="sbw"></span>
      <div class="sbt"><div class="sbf" style="width:{mqs}%;background:{sc(mqs)}"></div></div>
      <span class="sbv" style="color:{sc(mqs)};font-size:12px">{mqs}</span></div>
    </div>
    <div style="margin-top:9px;font-size:9px;color:var(--t3);line-height:1.9;border-top:1px solid var(--bdr);padding-top:6px">
      YES ≥ {ty} &nbsp;|&nbsp; CAUTION ≥ {tc2} &nbsp;|&nbsp; NO &lt; {tc2}<br>
      {'Live — yfinance / Yahoo Finance' if is_live else 'DEMO mode — pip install yfinance'} &nbsp;·&nbsp; Cache 30s &nbsp;·&nbsp; Auto-refresh 45s
    </div>
  </div>
  <div class="bp">
    <div class="th">◈ Terminal Analysis — {now}</div>
    <div class="tt">{th2}</div>
    <div style="margin-top:9px;border-top:1px solid var(--bdr);padding-top:6px;font-size:9px;color:var(--t3)">
      ⚠ FOR EDUCATIONAL USE ONLY. NOT FINANCIAL ADVICE.
    </div>
  </div>
  </div>""",unsafe_allow_html=True)

  st.markdown('</div>',unsafe_allow_html=True)

  # Auto-refresh 45s
  st.markdown('<script>setTimeout(()=>window.location.reload(),45000)</script>',unsafe_allow_html=True)

def main():
  dashboard()


if __name__ == "__main__":
    main()
