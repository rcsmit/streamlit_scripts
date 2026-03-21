"""
Should I Be Trading? — 5-Year Backtest Simulation
Runs the scoring engine on every trading day for the past 5 years
and shows how the YES/CAUTION/NO signals correlated with subsequent
SPY returns.

Usage:
  pip install streamlit yfinance pandas numpy plotly
  streamlit run backtest.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

try:
    st.set_page_config(
        page_title="Should I Be Trading? — Backtest",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
except:
    pass

def backtest():


# ── CSS ─────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');
    :root{
    --bg:#050a0f;--pan:#0a1520;--bdr:#0e2233;--bdr2:#1a3a52;
    --g:#00ff9d;--gd:#00cc7a;--gk:#003d25;
    --r:#ff3a3a;--rd:#cc2222;--rk:#3d0000;
    --a:#ffb700;--ad:#cc9200;--ak:#3d2c00;
    --b:#00b4ff;
    --t1:#c8dce8;--t2:#6a8fa8;--t3:#3a5a6e;
    --mono:'Share Tech Mono',monospace;--sans:'Rajdhani',sans-serif;
    }
    *{box-sizing:border-box}
    .stApp{background:var(--bg)!important;font-family:var(--mono)!important;color:var(--t1)!important}
    #MainMenu,footer,header{visibility:hidden}
    .block-container{padding:1rem 1.5rem!important;max-width:100%!important}
    .stApp>header{display:none}
    section[data-testid="stSidebar"]{display:none}
    .hdr{background:linear-gradient(90deg,#060e18,#0a1825);border-bottom:1px solid var(--bdr2);
        padding:12px 18px;margin-bottom:16px;border-radius:4px}
    .ht{font-family:var(--sans);font-size:22px;font-weight:700;color:var(--b);letter-spacing:.12em;text-transform:uppercase}
    .hs{font-size:10px;color:var(--t3);letter-spacing:.12em;margin-top:3px}
    .kpi{background:var(--pan);border:1px solid var(--bdr2);padding:14px;text-align:center;border-radius:2px}
    .kpi-v{font-size:28px;font-weight:700;font-family:var(--sans);margin-bottom:3px}
    .kpi-l{font-size:9px;letter-spacing:.2em;color:var(--t3)}
    .sec-hdr{font-size:9px;letter-spacing:.25em;color:var(--b);border-bottom:1px solid var(--bdr2);
            padding-bottom:6px;margin:18px 0 10px;text-transform:uppercase}
    div[data-testid="stMetric"]{background:var(--pan);border:1px solid var(--bdr2);padding:12px;border-radius:2px}
    .stSelectbox>div>div{background:var(--pan)!important;border:1px solid var(--bdr2)!important;
        color:var(--t1)!important;font-family:var(--mono)!important;font-size:10.5px!important;border-radius:0!important}
    .stButton>button{background:var(--pan)!important;color:var(--b)!important;border:1px solid var(--b)!important;
        font-family:var(--mono)!important;font-size:10px!important;letter-spacing:.15em!important;
        padding:6px 14px!important;border-radius:0!important;text-transform:uppercase!important}
    .stButton>button:hover{background:var(--b)!important;color:#000!important}
    </style>
    """, unsafe_allow_html=True)

    from scoring import SECTORS, compute_scores
    from market_data import load_historical


    # ── APPLY SCORING ────────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def run_backtest(mode: str) -> pd.DataFrame:
        df = load_historical()
        results = df.apply(lambda row: pd.Series(compute_scores(row, mode)), axis=1)
        # Flatten scores dict into separate columns
        for cat in ["volatility", "trend", "breadth", "momentum", "macro"]:
            results[f"{cat}_score"] = results["scores"].apply(lambda s: s[cat]["score"])
        results["mqs"]         = results["market_quality_score"]
        results["exec_window"] = results["execution_window_score"]
        results = results.drop(columns=["scores", "summary"], errors="ignore")
        return pd.concat([df, results], axis=1)


    # ── PLOT HELPERS ─────────────────────────────────────────────────────
    COLORS = {"YES": "#00ff9d", "CAUTION": "#ffb700", "NO": "#ff3a3a"}
    BG = "#050a0f"; PAN = "#0a1520"; GRID = "#0e2233"; TXT = "#6a8fa8"

    def apply_style(fig, title=""):
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=PAN, plot_bgcolor=BG,
            font=dict(family="Share Tech Mono", color=TXT, size=10),
            title=dict(text=title, font=dict(color="#00b4ff", size=12), x=0),
            margin=dict(l=8, r=8, t=36, b=8),
            xaxis=dict(gridcolor=GRID, zeroline=False, tickfont=dict(size=9)),
            yaxis=dict(gridcolor=GRID, zeroline=False, tickfont=dict(size=9)),
            legend=dict(bgcolor=PAN, bordercolor=GRID, font=dict(size=9)),
            hovermode="x unified",
        )
        return fig


    # ── MAIN APP ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hdr">
    <div class="ht">◈ Should I Be Trading? — 5-Year Backtest</div>
    <div class="hs">HOW WELL DID THE SCORING ENGINE PREDICT FAVORABLE TRADING CONDITIONS?</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 2, 6])
    with c1:
        mode = st.selectbox("Mode", ["swing", "day"],
                            format_func=lambda x: "🔄 SWING MODE" if x == "swing" else "⚡ DAY MODE")
    with c2:
        fwd_period = st.selectbox("Forward return", [1, 5, 10, 20],
                                index=1,
                                format_func=lambda x: f"Next {x} trading day{'s' if x > 1 else ''}")

    with st.spinner("Downloading 6 years of data and running backtest…"):
        df = run_backtest(mode)

    st.markdown('<div class="sec-hdr">◈ Loading complete</div>', unsafe_allow_html=True)

    fwd_col = f"fwd_{fwd_period}d"
    df_valid = df.dropna(subset=[fwd_col, "mqs", "decision"])

    # ── KPI ROW ─────────────────────────────────────────────────────────
    yes_df = df_valid[df_valid["decision"] == "YES"]
    cau_df = df_valid[df_valid["decision"] == "CAUTION"]
    no_df  = df_valid[df_valid["decision"] == "NO"]

    def pct_pos(d): return f"{(d[fwd_col] > 0).mean() * 100:.1f}%" if len(d) else "—"
    def avg_ret(d): return f"{d[fwd_col].mean():+.2f}%" if len(d) else "—"
    def med_ret(d): return f"{d[fwd_col].median():+.2f}%" if len(d) else "—"

    st.markdown('<div class="sec-hdr">◈ Signal Performance Summary</div>', unsafe_allow_html=True)
    cols = st.columns(9)
    kpis = [
        ("YES days", f"{len(yes_df)}", "#00ff9d"),
        ("YES avg return", avg_ret(yes_df), "#00ff9d"),
        ("YES % positive", pct_pos(yes_df), "#00ff9d"),
        ("CAUTION days", f"{len(cau_df)}", "#ffb700"),
        ("CAUTION avg", avg_ret(cau_df), "#ffb700"),
        ("NO days", f"{len(no_df)}", "#ff3a3a"),
        ("NO avg return", avg_ret(no_df), "#ff3a3a"),
        ("NO % positive", pct_pos(no_df), "#ff3a3a"),
        ("Total trading days", f"{len(df_valid)}", "#00b4ff"),
    ]
    for col, (label, value, color) in zip(cols, kpis):
        with col:
            st.markdown(f"""<div class="kpi">
    <div class="kpi-v" style="color:{color}">{value}</div>
    <div class="kpi-l">{label}</div></div>""", unsafe_allow_html=True)

    # ── €1000 INVESTMENT CALCULATOR ─────────────────────────────────────
    st.markdown('<div class="sec-hdr">◈ Beleggingscalculator — €X op datum Y is nu €Z</div>', unsafe_allow_html=True)

    inv_c1, inv_c2, inv_c3 = st.columns([2, 2, 6])
    with inv_c1:
        start_capital = st.number_input("Startbedrag (€)", min_value=100, max_value=1_000_000,
                                        value=1000, step=100)
    with inv_c2:
        min_date = df.index[0].date()
        max_date = df.index[-1].date()
        default_date = max(min_date, (df.index[-1] - pd.DateOffset(years=5) + pd.Timedelta(days=1)).date())
        start_date = st.date_input("Startdatum", value=default_date,
                                    min_value=min_date, max_value=max_date)

    # Find nearest trading day >= start_date
    start_ts = pd.Timestamp(start_date)
    df_from = df[df.index >= start_ts].copy()

    if len(df_from) < 5:
        st.warning("Te weinig data vanaf deze datum.")
    else:
        # Build daily equity curves from start_date
        df_from = df_from.dropna(subset=["spy_1d_chg", "decision"]).copy()
        df_from["daily_ret_spy"] = df_from["spy_1d_chg"] / 100

        # Het signaal van dag T is gebaseerd op slotdata van dag T.
        # Je kunt dus pas op dag T+1 handelen (kopen bij open of close T+1).
        # shift(1) schuift het signaal één dag op zodat het rendement van
        # dag T+1 wordt meegenomen — geen look-ahead bias.
        sig_yes = (df_from["decision"].shift(1) == "YES")
        sig_yc  = df_from["decision"].shift(1).isin(["YES", "CAUTION"])

        df_from["ret_yes_only"] = np.where(sig_yes, df_from["daily_ret_spy"], 0)
        df_from["ret_yes_cau"]  = np.where(sig_yc,  df_from["daily_ret_spy"], 0)

        df_from["eq_bah"]      = start_capital * (1 + df_from["daily_ret_spy"]).cumprod()
        df_from["eq_yes_only"] = start_capital * (1 + df_from["ret_yes_only"]).cumprod()
        df_from["eq_yes_cau"]  = start_capital * (1 + df_from["ret_yes_cau"]).cumprod()

        final_bah  = df_from["eq_bah"].iloc[-1]
        final_yes  = df_from["eq_yes_only"].iloc[-1]
        final_yc   = df_from["eq_yes_cau"].iloc[-1]
        actual_start = df_from.index[0].strftime("%d %b %Y")
        actual_end   = df_from.index[-1].strftime("%d %b %Y")
        n_years = (df_from.index[-1] - df_from.index[0]).days / 365.25

        def cagr(final, start, years):
            if years <= 0: return 0
            return ((final / start) ** (1 / years) - 1) * 100

        def max_dd(eq):
            roll_max = eq.cummax()
            dd = (eq - roll_max) / roll_max * 100
            return dd.min()

        # Big result cards
        st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:12px">
    <div style="background:#0a1520;border:2px solid #1a3a52;padding:20px;text-align:center;position:relative">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;background:#6a8fa8"></div>
        <div style="font-size:9px;letter-spacing:.25em;color:#3a5a6e;margin-bottom:8px">BUY &amp; HOLD SPY</div>
        <div style="font-size:11px;color:#6a8fa8;margin-bottom:4px">€{start_capital:,.0f} → </div>
        <div style="font-size:38px;font-weight:700;font-family:'Rajdhani',sans-serif;color:#6a8fa8">
        €{final_bah:,.0f}
        </div>
        <div style="font-size:10px;color:#3a5a6e;margin-top:6px">
        {(final_bah/start_capital-1)*100:+.1f}% totaal &nbsp;·&nbsp; CAGR {cagr(final_bah,start_capital,n_years):.1f}%<br>
        Max drawdown: {max_dd(df_from["eq_bah"]):.1f}%
        </div>
    </div>
    <div style="background:#0a1520;border:2px solid #cc9200;padding:20px;text-align:center;position:relative">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;background:#ffb700"></div>
        <div style="font-size:9px;letter-spacing:.25em;color:#3a5a6e;margin-bottom:8px">TRADE OP YES + CAUTION</div>
        <div style="font-size:11px;color:#ffb700;margin-bottom:4px">€{start_capital:,.0f} → </div>
        <div style="font-size:38px;font-weight:700;font-family:'Rajdhani',sans-serif;color:#ffb700">
        €{final_yc:,.0f}
        </div>
        <div style="font-size:10px;color:#cc9200;margin-top:6px">
        {(final_yc/start_capital-1)*100:+.1f}% totaal &nbsp;·&nbsp; CAGR {cagr(final_yc,start_capital,n_years):.1f}%<br>
        Max drawdown: {max_dd(df_from["eq_yes_cau"]):.1f}%
        </div>
    </div>
    <div style="background:#0a1520;border:2px solid #00cc7a;padding:20px;text-align:center;position:relative">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;background:#00ff9d;box-shadow:0 0 8px #00ff9d"></div>
        <div style="font-size:9px;letter-spacing:.25em;color:#3a5a6e;margin-bottom:8px">TRADE ALLEEN OP YES</div>
        <div style="font-size:11px;color:#00ff9d;margin-bottom:4px">€{start_capital:,.0f} → </div>
        <div style="font-size:38px;font-weight:700;font-family:'Rajdhani',sans-serif;color:#00ff9d">
        €{final_yes:,.0f}
        </div>
        <div style="font-size:10px;color:#00cc7a;margin-top:6px">
        {(final_yes/start_capital-1)*100:+.1f}% totaal &nbsp;·&nbsp; CAGR {cagr(final_yes,start_capital,n_years):.1f}%<br>
        Max drawdown: {max_dd(df_from["eq_yes_only"]):.1f}%
        </div>
    </div>
    </div>
    <div style="font-size:9px;color:#3a5a6e;margin-bottom:4px;letter-spacing:.05em">
    Periode: {actual_start} → {actual_end} &nbsp;·&nbsp; {len(df_from)} handelsdagen &nbsp;·&nbsp;
    {n_years:.1f} jaar &nbsp;·&nbsp;
    YES: {(df_from['decision']=='YES').sum()} dagen ({(df_from['decision']=='YES').mean()*100:.0f}%) &nbsp;·&nbsp;
    CAUTION: {(df_from['decision']=='CAUTION').sum()} &nbsp;·&nbsp;
    NO: {(df_from['decision']=='NO').sum()}
    </div>
    """, unsafe_allow_html=True)

        # Equity chart
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=df_from.index, y=df_from["eq_bah"], mode="lines",
            name="Buy & Hold", line=dict(color="#6a8fa8", width=1.5, dash="dot"),
            hovertemplate="B&H: €%{y:,.0f}"
        ))
        fig_eq.add_trace(go.Scatter(
            x=df_from.index, y=df_from["eq_yes_cau"], mode="lines",
            name="YES + CAUTION", line=dict(color="#ffb700", width=1.3),
            hovertemplate="YES+CAU: €%{y:,.0f}"
        ))
        fig_eq.add_trace(go.Scatter(
            x=df_from.index, y=df_from["eq_yes_only"], mode="lines",
            name="YES only", line=dict(color="#00ff9d", width=2),
            fill="tonexty", fillcolor="rgba(0,255,157,0.04)",
            hovertemplate="YES only: €%{y:,.0f}"
        ))
        fig_eq.add_hline(y=start_capital, line_color="#0e2233", line_dash="dot")

        # ── Koop/verkoop-pijlen op equity-grafiek (alle drie strategieën) ──
        # Pijl staat op dag T+1 (de dag waarop je daadwerkelijk handelt,
        # nadat het signaal van dag T bekend is).
        prev_from    = df_from["decision"].shift(1, fill_value="NO")
        in_yc_from   = df_from["decision"].isin(["YES", "CAUTION"])
        prev_yc_from = in_yc_from.shift(1, fill_value=False)

        # Signaalwisseling vindt plaats op dag T, handel op dag T+1
        signal_buy_yes  = (df_from["decision"] == "YES") & (prev_from != "YES")
        signal_sell_yes = (df_from["decision"] != "YES") & (prev_from == "YES")
        signal_buy_yc   = in_yc_from  & ~prev_yc_from
        signal_sell_yc  = ~in_yc_from &  prev_yc_from

        # Verschuif één dag: pijl op de dag van de daadwerkelijke transactie
        buy_eq_yes  = df_from[signal_buy_yes.shift(1,  fill_value=False)]
        sell_eq_yes = df_from[signal_sell_yes.shift(1, fill_value=False)]
        buy_eq_yc   = df_from[signal_buy_yc.shift(1,   fill_value=False)]
        sell_eq_yc  = df_from[signal_sell_yc.shift(1,  fill_value=False)]

        # YES-only pijlen (groen/rood, groot)
        fig_eq.add_trace(go.Scatter(
            x=buy_eq_yes.index,
            y=buy_eq_yes["eq_yes_only"] * 0.988,
            mode="markers",
            marker=dict(symbol="triangle-up", color="#00ff9d", size=12,
                        line=dict(color="#003d25", width=1)),
            name="BUY — YES only",
            hovertemplate="▲ BUY (YES only)<br>%{x|%Y-%m-%d}<br>€%{text}<extra></extra>",
            text=[f"{v:,.0f}" for v in buy_eq_yes["eq_yes_only"]],
        ))
        fig_eq.add_trace(go.Scatter(
            x=sell_eq_yes.index,
            y=sell_eq_yes["eq_yes_only"] * 1.012,
            mode="markers",
            marker=dict(symbol="triangle-down", color="#ff3a3a", size=12,
                        line=dict(color="#3d0000", width=1)),
            name="SELL — YES only",
            hovertemplate="▼ SELL (YES only)<br>%{x|%Y-%m-%d}<br>€%{text}<extra></extra>",
            text=[f"{v:,.0f}" for v in sell_eq_yes["eq_yes_only"]],
        ))

        # YES+CAUTION pijlen (amber, iets kleiner, op eigen equity-lijn)
        fig_eq.add_trace(go.Scatter(
            x=buy_eq_yc.index,
            y=buy_eq_yc["eq_yes_cau"] * 0.978,
            mode="markers",
            marker=dict(symbol="triangle-up", color="#ffb700", size=8,
                        line=dict(color="#3d2c00", width=1)),
            name="BUY — YES+CAUTION",
            hovertemplate="▲ BUY (YES+CAU)<br>%{x|%Y-%m-%d}<br>€%{text}<extra></extra>",
            text=[f"{v:,.0f}" for v in buy_eq_yc["eq_yes_cau"]],
        ))
        fig_eq.add_trace(go.Scatter(
            x=sell_eq_yc.index,
            y=sell_eq_yc["eq_yes_cau"] * 1.022,
            mode="markers",
            marker=dict(symbol="triangle-down", color="#ffb700", size=8,
                        line=dict(color="#3d2c00", width=1)),
            name="SELL — YES+CAUTION",
            hovertemplate="▼ SELL (YES+CAU)<br>%{x|%Y-%m-%d}<br>€%{text}<extra></extra>",
            text=[f"{v:,.0f}" for v in sell_eq_yc["eq_yes_cau"]],
        ))

        # Annotate final values
        for name, val, color in [
            ("Buy & Hold", final_bah, "#6a8fa8"),
            ("YES+CAUTION", final_yc, "#ffb700"),
            ("YES only", final_yes, "#00ff9d"),
        ]:
            fig_eq.add_annotation(
                x=df_from.index[-1], y=val,
                text=f"  {name}: €{val:,.0f}",
                showarrow=False, xanchor="left",
                font=dict(color=color, size=9)
            )

        apply_style(fig_eq, f"€{start_capital:,.0f} geïnvesteerd op {actual_start} — waarde per dag")
        fig_eq.update_layout(height=360, yaxis_title="Portefeuillewaarde (€)",
                            yaxis_tickprefix="€", yaxis_tickformat=",.0f")
        st.plotly_chart(fig_eq, use_container_width=True)

    # ── CHART 1: SPY price with signal overlay ─────────────────────────
    st.markdown(f'<div class="sec-hdr">◈ SPY Price + Daily Signal (last 5 years)</div>', unsafe_allow_html=True)

    fig1 = go.Figure()

    # SPY line
    fig1.add_trace(go.Scatter(
        x=df.index, y=df["spy_price"], mode="lines",
        line=dict(color="#00b4ff", width=1.5), name="SPY", hovertemplate="%{y:.2f}"
    ))

    # MA200
    fig1.add_trace(go.Scatter(
        x=df.index, y=df["spy_ma200"], mode="lines",
        line=dict(color="#3a5a6e", width=1, dash="dot"), name="MA200"
    ))

    # Signal scatter per decision
    for dec, color in COLORS.items():
        mask = df["decision"] == dec
        sub  = df[mask]
        fig1.add_trace(go.Scatter(
            x=sub.index, y=sub["spy_price"],
            mode="markers",
            marker=dict(color=color, size=3, opacity=0.4),
            name=dec, hovertemplate=f"{dec}: %{{y:.2f}}"
        ))

    # ── Koop/verkoop-pijlen voor alle drie strategieën ──────────────────
    # Signaal op dag T → handel (pijl) op dag T+1.
    # De wisseling wordt gedetecteerd op T, de pijl staat op T+1.
    df["in_yes"] = df["decision"] == "YES"
    df["in_yc"]  = df["decision"].isin(["YES", "CAUTION"])

    prev    = df["decision"].shift(1, fill_value="NO")
    prev_yc = df["in_yc"].shift(1, fill_value=False)

    # Signaalwisselingen op dag T
    sig_buy_yes  = (df["decision"] == "YES") & (prev != "YES")
    sig_sell_yes = (df["decision"] != "YES") & (prev == "YES")
    sig_buy_yc   = df["in_yc"]  & ~prev_yc
    sig_sell_yc  = ~df["in_yc"] &  prev_yc

    # Pijlen op dag T+1 (daadwerkelijke handeldag)
    buy_yes  = df[sig_buy_yes.shift(1,  fill_value=False)]
    sell_yes = df[sig_sell_yes.shift(1, fill_value=False)]
    buy_yc   = df[sig_buy_yc.shift(1,   fill_value=False)]
    sell_yc  = df[sig_sell_yc.shift(1,  fill_value=False)]

    # ── Pijlen YES-only (groen/rood, groot) ──────────────────────────────
    fig1.add_trace(go.Scatter(
        x=buy_yes.index,
        y=buy_yes["spy_price"] * 0.990,
        mode="markers",
        marker=dict(symbol="triangle-up", color="#00ff9d", size=12,
                    line=dict(color="#003d25", width=1)),
        name="BUY — YES only",
        hovertemplate="▲ BUY (YES only)<br>%{x|%Y-%m-%d}  SPY %{text}<extra></extra>",
        text=[f"{v:.2f}" for v in buy_yes["spy_price"]],
    ))
    fig1.add_trace(go.Scatter(
        x=sell_yes.index,
        y=sell_yes["spy_price"] * 1.010,
        mode="markers",
        marker=dict(symbol="triangle-down", color="#ff3a3a", size=12,
                    line=dict(color="#3d0000", width=1)),
        name="SELL — YES only",
        hovertemplate="▼ SELL (YES only)<br>%{x|%Y-%m-%d}  SPY %{text}<extra></extra>",
        text=[f"{v:.2f}" for v in sell_yes["spy_price"]],
    ))

    # ── Pijlen YES+CAUTION (amber/oranje, iets kleiner) ──────────────────
    fig1.add_trace(go.Scatter(
        x=buy_yc.index,
        y=buy_yc["spy_price"] * 0.981,
        mode="markers",
        marker=dict(symbol="triangle-up", color="#ffb700", size=8,
                    line=dict(color="#3d2c00", width=1)),
        name="BUY — YES+CAUTION",
        hovertemplate="▲ BUY (YES+CAU)<br>%{x|%Y-%m-%d}  SPY %{text}<extra></extra>",
        text=[f"{v:.2f}" for v in buy_yc["spy_price"]],
    ))
    fig1.add_trace(go.Scatter(
        x=sell_yc.index,
        y=sell_yc["spy_price"] * 1.019,
        mode="markers",
        marker=dict(symbol="triangle-down", color="#ffb700", size=8,
                    line=dict(color="#3d2c00", width=1)),
        name="SELL — YES+CAUTION",
        hovertemplate="▼ SELL (YES+CAU)<br>%{x|%Y-%m-%d}  SPY %{text}<extra></extra>",
        text=[f"{v:.2f}" for v in sell_yc["spy_price"]],
    ))

    n_yes = min(len(buy_yes), len(sell_yes))
    n_yc  = min(len(buy_yc),  len(sell_yc))
    apply_style(fig1, f"SPY — Koop▲ / Verkoop▼  |  Groen=YES-only ({n_yes}×)  Amber=YES+CAUTION ({n_yc}×)")
    fig1.update_layout(height=420,
        legend=dict(orientation="h", y=-0.15, font=dict(size=9)))
    st.plotly_chart(fig1, use_container_width=True)

    # ── CHART 2: MQS over time ─────────────────────────────────────────
    st.markdown('<div class="sec-hdr">◈ Market Quality Score (MQS) over time</div>', unsafe_allow_html=True)

    fig2 = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35], shared_xaxes=True, vertical_spacing=0.04)

    fig2.add_trace(go.Scatter(
        x=df.index, y=df["mqs"], mode="lines", fill="tozeroy",
        fillcolor="rgba(0,180,255,0.06)", line=dict(color="#00b4ff", width=1.2),
        name="MQS", hovertemplate="MQS: %{y:.1f}"
    ), row=1, col=1)

    # Threshold lines
    for val, color, label in [(80, "#00ff9d", "YES≥80"), (60, "#ffb700", "CAUTION≥60")]:
        fig2.add_hline(y=val, line=dict(color=color, width=0.7, dash="dot"),
                    annotation_text=label, annotation_font_size=8,
                    annotation_font_color=color, row=1, col=1)

    # VIX
    fig2.add_trace(go.Scatter(
        x=df.index, y=df["vix"], mode="lines",
        line=dict(color="#ff3a3a", width=1), name="VIX"
    ), row=2, col=1)

    apply_style(fig2, "Market Quality Score + VIX")
    fig2.update_layout(height=420)
    fig2.update_yaxes(title_text="MQS", row=1, col=1, title_font_size=9)
    fig2.update_yaxes(title_text="VIX", row=2, col=1, title_font_size=9)
    st.plotly_chart(fig2, use_container_width=True)

    # ── CHART 3: Forward return distribution by signal ─────────────────
    st.markdown(f'<div class="sec-hdr">◈ Distribution of {fwd_period}-Day Forward Returns by Signal</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        fig3 = go.Figure()
        for dec, color in COLORS.items():
            sub = df_valid[df_valid["decision"] == dec][fwd_col].dropna()
            if len(sub) < 10:
                continue
            fig3.add_trace(go.Violin(
                y=sub, name=dec, box_visible=True, meanline_visible=True,
                line_color=color, fillcolor=color, #.replace("#", "rgba(") + ",0.12)",
                hoverinfo="y+name"
            ))
        apply_style(fig3, f"Return Distribution — Next {fwd_period}d")
        fig3.update_layout(height=350, yaxis_title="Return (%)")
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        # Avg return per MQS bucket (10-point bins)
        df_valid2 = df_valid.copy()
        df_valid2["mqs_bucket"] = (df_valid2["mqs"] // 10 * 10).astype(int)
        bucket_stats = df_valid2.groupby("mqs_bucket")[fwd_col].agg(["mean", "median", "count"]).reset_index()
        bucket_stats.columns = ["bucket", "mean", "median", "count"]

        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=bucket_stats["bucket"].astype(str) + "-" + (bucket_stats["bucket"] + 10).astype(str),
            y=bucket_stats["mean"],
            name="Avg return",
            marker_color=[
                "#00ff9d" if v >= 0 else "#ff3a3a" for v in bucket_stats["mean"]
            ],
            text=[f"{v:+.2f}%" for v in bucket_stats["mean"]],
            textposition="outside",
            textfont=dict(size=8, color="#c8dce8"),
            customdata=bucket_stats[["count", "median"]].values,
            hovertemplate="MQS %{x}<br>Avg: %{y:.2f}%<br>Median: %{customdata[1]:.2f}%<br>Days: %{customdata[0]}<extra></extra>"
        ))
        apply_style(fig4, f"Avg {fwd_period}d Return by MQS Bucket")
        fig4.update_layout(height=350, yaxis_title="Avg Return (%)", xaxis_title="MQS Range")
        fig4.add_hline(y=0, line_color=GRID)
        st.plotly_chart(fig4, use_container_width=True)

    # ── CHART 4: Sub-score evolution ──────────────────────────────────
    st.markdown('<div class="sec-hdr">◈ Sub-Score Components over Time</div>', unsafe_allow_html=True)

    score_cols = {
        "volatility_score":     ("Volatility 25%", "#00b4ff"),
        "trend_score":   ("Trend 20%",      "#00ff9d"),
        "breadth_score": ("Breadth 20%",    "#ffb700"),
        "momentum_score":     ("Momentum 25%",   "#ff9d00"),
        "macro_score":   ("Macro 10%",      "#bb88ff"),
    }

    fig5 = go.Figure()
    for col, (label, color) in score_cols.items():
        fig5.add_trace(go.Scatter(
            x=df.index, y=df[col].rolling(10).mean(),
            mode="lines", name=label,
            line=dict(color=color, width=1.2),
            hovertemplate=f"{label}: %{{y:.1f}}"
        ))

    apply_style(fig5, "Sub-Scores — 10-Day Rolling Mean")
    fig5.update_layout(height=300)
    fig5.add_hline(y=50, line_color=GRID, line_dash="dot")
    st.plotly_chart(fig5, use_container_width=True)

    # ── CHART 5: Signal frequency & win-rate heatmap by year/month ─────
    st.markdown('<div class="sec-hdr">◈ Monthly YES% (share of YES days per month)</div>',
                unsafe_allow_html=True)

    df_cal = df_valid.copy()
    df_cal["year"]  = df_cal.index.year
    df_cal["month"] = df_cal.index.month
    df_cal["is_yes"] = (df_cal["decision"] == "YES").astype(int)

    pivot = df_cal.pivot_table(values="is_yes", index="year", columns="month", aggfunc="mean") * 100
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    pivot.columns = [month_names.get(c, c) for c in pivot.columns]

    fig6 = go.Figure(data=go.Heatmap(
        z=pivot.values, x=list(pivot.columns), y=[str(y) for y in pivot.index],
        colorscale=[[0,"#3d0000"],[0.35,"#3d2c00"],[0.65,"#003d25"],[1,"#00ff9d"]],
        text=[[f"{v:.0f}%" if not np.isnan(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}", textfont=dict(size=9),
        showscale=True, zmin=0, zmax=100,
        colorbar=dict(tickfont=dict(size=8, color=TXT), len=0.8)
    ))
    apply_style(fig6, "% of Days Rated YES per Month")
    fig6.update_layout(height=280)
    st.plotly_chart(fig6, use_container_width=True)

    # ── CHART 6: Cumulative equity – only trade on YES days ─────────────
    st.markdown('<div class="sec-hdr">◈ Hypothetical Cumulative Return — Buy on YES, Flat on CAUTION/NO</div>',
                unsafe_allow_html=True)

    # Use 1-day forward return as proxy for being "in the market" that day
    df_eq = df_valid[["decision", "fwd_1d", "spy_1d_chg"]].dropna().copy()
    df_eq["strategy_ret"] = np.where(df_eq["decision"] == "YES", df_eq["fwd_1d"] / 100, 0)
    df_eq["bah_ret"]      = df_eq["spy_1d_chg"] / 100

    df_eq["strategy_equity"] = (1 + df_eq["strategy_ret"]).cumprod()
    df_eq["bah_equity"]      = (1 + df_eq["bah_ret"]).cumprod()

    # Also: trade on YES + CAUTION
    df_eq["yc_ret"]      = np.where(df_eq["decision"].isin(["YES","CAUTION"]), df_eq["fwd_1d"] / 100, 0)
    df_eq["yc_equity"]   = (1 + df_eq["yc_ret"]).cumprod()

    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(
        x=df_eq.index, y=df_eq["bah_equity"], mode="lines",
        name="Buy & Hold SPY", line=dict(color="#6a8fa8", width=1.5, dash="dot")
    ))
    fig7.add_trace(go.Scatter(
        x=df_eq.index, y=df_eq["yc_equity"], mode="lines",
        name="Trade YES + CAUTION", line=dict(color="#ffb700", width=1.2)
    ))
    fig7.add_trace(go.Scatter(
        x=df_eq.index, y=df_eq["strategy_equity"], mode="lines",
        name="Trade YES only", line=dict(color="#00ff9d", width=1.5),
        fill="tonexty", fillcolor="rgba(0,255,157,0.04)"
    ))

    apply_style(fig7, "Cumulative Growth of $1 — Signal-Filtered vs Buy & Hold")
    fig7.update_layout(height=350, yaxis_title="Portfolio Value ($1 start)")

    # Annotate final values
    for name, col, color in [
        ("B&H",       "bah_equity",      "#6a8fa8"),
        ("YES+CAU",   "yc_equity",       "#ffb700"),
        ("YES only",  "strategy_equity", "#00ff9d"),
    ]:
        final = df_eq[col].iloc[-1]
        fig7.add_annotation(
            x=df_eq.index[-1], y=final,
            text=f"  {name}: {final:.2f}x",
            showarrow=False, font=dict(color=color, size=9), xanchor="left"
        )

    st.plotly_chart(fig7, use_container_width=True)

    # ── DATA TABLE ──────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">◈ Raw Signal Table (most recent 120 days)</div>',
                unsafe_allow_html=True)

    display_cols = ["spy_price", "vix", "tnx", "mqs", "decision", "exec_window",
                    "vol_score", "trend_score", "breadth_score", "mom_score", "macro_score",
                    "fwd_1d", "fwd_5d", "fwd_10d", "fwd_20d"]
    display_cols = [c for c in display_cols if c in df_valid.columns]
    recent = df_valid[display_cols].tail(120).sort_index(ascending=False)

    def color_decision(val):
        c = {"YES": "#003d25", "CAUTION": "#3d2c00", "NO": "#3d0000"}.get(val, "")
        tc = {"YES": "#00ff9d", "CAUTION": "#ffb700", "NO": "#ff3a3a"}.get(val, "#c8dce8")
        return f"background-color:{c};color:{tc}" if c else ""

    def color_return(val):
        if isinstance(val, float):
            return f"color:{'#00ff9d' if val > 0 else '#ff3a3a' if val < 0 else '#6a8fa8'}"
        return ""

    styled = (
        recent.style
        .format({
            "spy_price": "{:.2f}", "vix": "{:.2f}", "tnx": "{:.3f}",
            "mqs": "{:.1f}", "exec_window": "{:.0f}",
            "vol_score": "{:.0f}", "trend_score": "{:.0f}",
            "breadth_score": "{:.0f}", "mom_score": "{:.0f}", "macro_score": "{:.0f}",
            "fwd_1d": "{:+.2f}%", "fwd_5d": "{:+.2f}%",
            "fwd_10d": "{:+.2f}%", "fwd_20d": "{:+.2f}%",
        }, na_rep="—")
        .applymap(color_decision, subset=["decision"])
        .applymap(color_return, subset=[c for c in ["fwd_1d","fwd_5d","fwd_10d","fwd_20d"] if c in recent.columns])
        .set_properties(**{"background-color": "#070d14", "color": "#c8dce8",
                            "font-size": "10px", "font-family": "Share Tech Mono"})
        .set_table_styles([
            {"selector": "th", "props": [("background-color", "#0a1520"), ("color", "#6a8fa8"),
                                        ("font-size", "9px"), ("letter-spacing", ".1em"),
                                        ("border", "1px solid #0e2233")]}
        ])
    )
    st.dataframe(styled, height=420, use_container_width=True)

    st.markdown("""
    <div style="margin-top:18px;padding:8px 12px;background:#0a1520;border:1px solid #0e2233;
                font-size:9px;color:#3a5a6e;letter-spacing:.07em;border-radius:2px">
    ⚠ EDUCATIONAL PURPOSES ONLY — NOT FINANCIAL ADVICE.<br>
    Past signal performance does not guarantee future results.
    The scoring engine uses derived/estimated metrics (McClellan, NH/NL, A/D ratio)
    that approximate but do not replicate professional-grade breadth data.
    Forward returns shown are gross; no transaction costs, slippage, or taxes applied.
    </div>
    """, unsafe_allow_html=True)


def main():
    backtest()


if __name__ == "__main__":
    main()
