"""
market_data.py — Shared data layer voor Should I Be Trading?
Importeer dit in zowel app.py als backtest.py.
"""

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

from scoring import SECTORS, compute_scores


# ── Live data (voor app.py — huidige dag) ────────────────────────────
def fetch_live() -> dict:
    """
    Download de meest recente marktdata via yfinance en retourneer
    een flat dict geschikt voor compute_scores().
    Gooit een exception als er onvoldoende data is.
    """
    import yfinance as yf

    tickers = ["SPY", "QQQ", "IWM", "^VIX", "^VVIX", "^TNX", "DX-Y.NYB", "^GSPC"] + list(SECTORS.keys())
    raw = yf.download(
        tickers, period="1y", interval="1d",
        group_by="ticker", progress=False, auto_adjust=True, timeout=15
    )

    def gc(t):
        try:
            s = raw[t]["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
            return s.dropna()
        except Exception:
            return pd.Series(dtype=float)

    def last(s):
        try: return float(s.dropna().iloc[-1])
        except: return None

    def chg1(s):
        s = s.dropna()
        return float((s.iloc[-1] / s.iloc[-2] - 1) * 100) if len(s) > 1 else 0.0

    def chg5(s):
        s = s.dropna()
        return float((s.iloc[-1] / s.iloc[-6] - 1) * 100) if len(s) > 5 else 0.0

    def sl5(s):
        s = s.dropna()
        if len(s) < 5: return 0.0
        return float(np.polyfit(np.arange(5), s.iloc[-5:].values, 1)[0])

    def rsi14(s, p=14):
        d = s.diff()
        g = d.clip(lower=0).rolling(p).mean()
        l = (-d.clip(upper=0)).rolling(p).mean()
        rs = g / l.replace(0, np.nan)
        return float((100 - 100 / (1 + rs)).dropna().iloc[-1])

    def prank(s, v):
        s = s.dropna()
        return float((s < v).mean() * 100) if len(s) else 50.0

    spy  = gc("SPY");  qqq = gc("QQQ"); iwm = gc("IWM")
    vix  = gc("^VIX"); vvix = gc("^VVIX")
    tnx  = gc("^TNX"); dxy = gc("DX-Y.NYB")

    if len(spy) < 20:
        raise ValueError("Onvoldoende SPY data")

    d = {"is_live": True}

    # SPY
    p    = last(spy)
    m20  = last(spy.rolling(20).mean())
    m50  = last(spy.rolling(50).mean())
    m200 = last(spy.rolling(200).mean())
    d.update({
        "spy_price": p, "spy_ma20": m20, "spy_ma50": m50, "spy_ma200": m200,
        "spy_rsi14": rsi14(spy), "spy_1d_chg": chg1(spy), "spy_5d_chg": chg5(spy),
        "spy_vs_ma20":  (p / m20  - 1) * 100 if p and m20  else 0,
        "spy_vs_ma50":  (p / m50  - 1) * 100 if p and m50  else 0,
        "spy_vs_ma200": (p / m200 - 1) * 100 if p and m200 else 0,
    })
    d["regime"] = (
        "UPTREND"   if p and m20 and m50 and m200 and p > m20 > m50 > m200 else
        "DOWNTREND" if p and m200 and m50 and p < m200 and p < m50          else
        "CHOP"
    )

    # QQQ
    qp = last(qqq); qm50 = last(qqq.rolling(50).mean())
    d.update({
        "qqq_price": qp, "qqq_ma50": qm50, "qqq_1d_chg": chg1(qqq),
        "qqq_vs_ma50": (qp / qm50 - 1) * 100 if qp and qm50 else 0,
    })

    # IWM
    ip = last(iwm); im50 = last(iwm.rolling(50).mean())
    d.update({"iwm_price": ip, "iwm_vs_ma50": (ip / im50 - 1) * 100 if ip and im50 else 0})

    # VIX
    v = last(vix) or 20
    d.update({
        "vix": v, "vix_slope5d": sl5(vix), "vix_pct1y": prank(vix.iloc[-252:], v),
        "vvix": last(vvix),
        "pc_ratio_est": 0.70 if v < 15 else (0.82 if v < 20 else (1.00 if v < 25 else (1.18 if v < 30 else 1.38))),
    })

    # TNX / DXY
    t  = last(tnx) or 4.5
    ts = sl5(tnx)
    d.update({
        "tnx": t, "tnx_slope5d": ts,
        "dxy": last(dxy), "dxy_slope5d": sl5(dxy),
        "fed_stance": (
            "HAWKISH" if t > 5 or (t > 4 and ts > 0) else
            "DOVISH"  if t < 3.5 else
            "NEUTRAL"
        ),
    })

    # Sectoren
    sector_chg = {e: chg5(gc(e)) for e in SECTORS}
    d["sector_chg"] = sector_chg
    ss  = sorted(sector_chg.items(), key=lambda x: x[1], reverse=True)
    pos = sum(1 for val in sector_chg.values() if val > 0)
    d.update({
        "top3_sectors":    ss[:3],
        "bottom3_sectors": ss[-3:],
        "sector_spread":   ss[0][1] - ss[-1][1] if ss else 0,
        "sectors_positive":  pos,
        "pct_sectors_pos":   pos / 11 * 100,
        "ad_ratio_est":      pos / max(11 - pos, 1),
        "nasdaq_nh_est":     int(120 + pos * 15),
        "nasdaq_nl_est":     int(160 - pos * 12),
        "est_pct_above_50d": pos / 11 * 85,
    })

    # McClellan
    if len(spy) > 20:
        e10 = spy.ewm(span=10).mean()
        e19 = spy.ewm(span=19).mean()
        d["mclellan_est"] = float((e10.iloc[-1] - e19.iloc[-1]) / spy.iloc[-1] * 1000)
    else:
        d["mclellan_est"] = 0

    return d


def demo_data() -> dict:
    """Realistische demodata als yfinance niet beschikbaar is."""
    np.random.seed(int(time.time()) // 45)
    j = lambda b, s: float(b + np.random.randn() * s)

    d = {"is_live": False}
    sp = j(560.4, 1.2); m20 = j(567.1, .3); m50 = j(574.8, .2); m200 = j(541.2, .1)
    d.update({
        "spy_price": sp, "spy_ma20": m20, "spy_ma50": m50, "spy_ma200": m200,
        "spy_rsi14": j(44.2, 1.5), "spy_1d_chg": j(-.38, .25), "spy_5d_chg": j(-2.15, .4),
        "regime": "CHOP",
        "spy_vs_ma20":  (sp / m20  - 1) * 100,
        "spy_vs_ma50":  (sp / m50  - 1) * 100,
        "spy_vs_ma200": (sp / m200 - 1) * 100,
    })
    qp = j(475.2, 1.); qm = j(491.3, .2)
    d.update({"qqq_price": qp, "qqq_ma50": qm, "qqq_vs_ma50": (qp / qm - 1) * 100, "qqq_1d_chg": j(-.62, .3)})
    ip = j(203.5, .8); im = j(218.4, .2)
    d.update({"iwm_price": ip, "iwm_vs_ma50": (ip / im - 1) * 100})
    v = j(22.8, .6)
    d.update({
        "vix": v, "vix_slope5d": j(.18, .08), "vix_pct1y": j(62., 3.),
        "vvix": j(112.4, 2.), "pc_ratio_est": 1.05,
    })
    t = j(4.31, .03); ts = j(.022, .008)
    d.update({"tnx": t, "tnx_slope5d": ts, "dxy": j(103.8, .2), "dxy_slope5d": j(-.08, .04), "fed_stance": "NEUTRAL"})

    sc = {
        "XLK": j(-3.12, .4), "XLF": j(-1.88, .3), "XLE": j(.45, .4),
        "XLV": j(.92, .3),   "XLI": j(-1.42, .3), "XLY": j(-2.65, .4),
        "XLP": j(1.14, .2),  "XLU": j(1.38, .3),  "XLB": j(-.78, .3),
        "XLRE": j(.62, .3),  "XLC": j(-1.95, .4),
    }
    d["sector_chg"] = sc
    ss  = sorted(sc.items(), key=lambda x: x[1], reverse=True)
    pos = sum(1 for val in sc.values() if val > 0)
    d.update({
        "top3_sectors":    ss[:3],
        "bottom3_sectors": ss[-3:],
        "sector_spread":   ss[0][1] - ss[-1][1],
        "sectors_positive":  pos,
        "pct_sectors_pos":   pos / 11 * 100,
        "ad_ratio_est":      pos / max(11 - pos, 1),
        "nasdaq_nh_est":     int(j(78, 8)),
        "nasdaq_nl_est":     int(j(142, 10)),
        "est_pct_above_50d": j(38.4, 2.),
        "mclellan_est":      j(-42.5, 5.),
    })
    return d


def get_market_data(mode: str = "swing") -> dict:
    """
    Haal live data op (of demo als fallback), voeg scores toe en retourneer
    een volledig verrijkt dict klaar voor de UI.
    """
    try:
        d = fetch_live()
    except Exception as e:
        d = demo_data()
        d["fetch_error"] = str(e)

    result = compute_scores(d, mode)
    d.update(result)
    return d


# ── Historische data (voor backtest.py) ─────────────────────────────
def load_historical() -> pd.DataFrame:
    """
    Download ~6 jaar dagelijkse data en bereken alle features die
    compute_scores() nodig heeft. Retourneert een DataFrame met
    één rij per handelsdag plus forward returns.
    """
    import yfinance as yf

    tickers = ["SPY", "QQQ", "IWM", "^VIX", "^VVIX", "^TNX", "DX-Y.NYB"] + list(SECTORS.keys())
    raw = yf.download(
        tickers, period="6y", interval="1d",
        group_by="ticker", progress=False, auto_adjust=True, timeout=60
    )

    def gc(t):
        try:
            s = raw[t]["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
            return s.dropna()
        except Exception:
            return pd.Series(dtype=float)

    spy  = gc("SPY");  qqq = gc("QQQ");  iwm = gc("IWM")
    vix  = gc("^VIX"); tnx = gc("^TNX"); dxy = gc("DX-Y.NYB")
    vvix = gc("^VVIX")
    sec  = {e: gc(e) for e in SECTORS}

    idx = spy.index
    df  = pd.DataFrame(index=idx)
    df["spy_price"] = spy

    # Moving averages & changes
    df["spy_ma20"]  = spy.rolling(20).mean()
    df["spy_ma50"]  = spy.rolling(50).mean()
    df["spy_ma200"] = spy.rolling(200).mean()
    df["spy_1d_chg"] = spy.pct_change() * 100
    df["spy_5d_chg"] = spy.pct_change(5) * 100
    df["spy_vs_ma20"]  = (spy / df["spy_ma20"]  - 1) * 100
    df["spy_vs_ma50"]  = (spy / df["spy_ma50"]  - 1) * 100
    df["spy_vs_ma200"] = (spy / df["spy_ma200"] - 1) * 100

    # RSI 14
    delta = spy.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["spy_rsi14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # Regime
    def _regime(row):
        p, m20, m50, m200 = row["spy_price"], row["spy_ma20"], row["spy_ma50"], row["spy_ma200"]
        if pd.isna(m200): return "CHOP"
        if p > m20 > m50 > m200: return "UPTREND"
        if p < m200 and p < m50: return "DOWNTREND"
        return "CHOP"
    df["regime"] = df.apply(_regime, axis=1)

    # QQQ / IWM
    qqq_a = qqq.reindex(idx, method="ffill")
    df["qqq_vs_ma50"] = (qqq_a / qqq_a.rolling(50).mean() - 1) * 100
    iwm_a = iwm.reindex(idx, method="ffill")
    df["iwm_vs_ma50"] = (iwm_a / iwm_a.rolling(50).mean() - 1) * 100

    # VIX
    vix_a = vix.reindex(idx, method="ffill")
    df["vix"] = vix_a
    df["vix_slope5d"] = vix_a.diff(5) / 5

    # VIX 1-jaar percentielrank (rolling)
    vix_pct = pd.Series(index=vix_a.index, dtype=float)
    for i in range(len(vix_a)):
        window = vix_a.iloc[max(0, i - 252): i + 1]
        vix_pct.iloc[i] = (window < vix_a.iloc[i]).mean() * 100
    df["vix_pct1y"] = vix_pct

    df["vvix"] = vvix.reindex(idx, method="ffill")
    df["pc_ratio_est"] = df["vix"].map(
        lambda v: 0.70 if v < 15 else (0.82 if v < 20 else (1.00 if v < 25 else (1.18 if v < 30 else 1.38)))
    )

    # TNX / DXY
    tnx_a = tnx.reindex(idx, method="ffill")
    dxy_a = dxy.reindex(idx, method="ffill")
    df["tnx"] = tnx_a
    df["tnx_slope5d"] = tnx_a.diff(5) / 5
    df["dxy"] = dxy_a
    df["dxy_slope5d"] = dxy_a.diff(5) / 5
    df["fed_stance"] = df.apply(
        lambda r: "HAWKISH" if (r["tnx"] > 5 or (r["tnx"] > 4 and r["tnx_slope5d"] > 0))
        else ("DOVISH" if r["tnx"] < 3.5 else "NEUTRAL"), axis=1
    )

    # Sector metrics
    for e in SECTORS:
        df[f"sec_{e}_5d"] = sec[e].reindex(idx, method="ffill").pct_change(5) * 100

    sec_cols = [f"sec_{e}_5d" for e in SECTORS]
    df["sectors_positive"]  = df[sec_cols].gt(0).sum(axis=1)
    df["pct_sectors_pos"]   = df["sectors_positive"] / 11 * 100
    df["ad_ratio_est"]      = df["sectors_positive"] / (11 - df["sectors_positive"]).clip(lower=1)
    df["sector_spread"]     = df[sec_cols].max(axis=1) - df[sec_cols].min(axis=1)
    df["nasdaq_nh_est"]     = (120 + df["sectors_positive"] * 15).astype(int)
    df["nasdaq_nl_est"]     = (160 - df["sectors_positive"] * 12).astype(int)
    df["est_pct_above_50d"] = df["sectors_positive"] / 11 * 85

    # McClellan estimate
    e10 = spy.ewm(span=10).mean().reindex(idx)
    e19 = spy.ewm(span=19).mean().reindex(idx)
    df["mclellan_est"] = (e10 - e19) / spy.reindex(idx) * 1000

    # Forward returns
    for n in [1, 5, 10, 20]:
        df[f"fwd_{n}d"] = spy.pct_change(n).shift(-n).reindex(idx) * 100

    # Trim tot 5 jaar (6e jaar was warm-up voor MA200)
    cutoff = df.index[-1] - pd.DateOffset(years=5)
    df = df[df.index >= cutoff].copy()
    df.dropna(subset=["spy_price", "spy_ma200"], inplace=True)

    return df
