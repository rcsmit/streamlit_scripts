"""
scoring.py — Shared scoring engine for Should I Be Trading?
Importeer dit in zowel app.py als backtest.py.
"""

SECTORS = {
    "XLK": "Tech",   "XLF": "Fin",    "XLE": "Energy", "XLV": "Health",
    "XLI": "Indust", "XLY": "Discret","XLP": "Staples", "XLU": "Util",
    "XLB": "Material","XLRE": "RE",   "XLC": "Comms",
}

THRESHOLDS = {
    "swing": {"yes": 80, "caution": 60},
    "day":   {"yes": 75, "caution": 55},
}


def compute_scores(d: dict, mode: str = "swing") -> dict:
    """
    Bereken alle sub-scores en het eindbesluit vanuit een flat dict met marktdata.
    Werkt zowel op live data (app.py) als op een DataFrame-rij (backtest.py).

    Parameters
    ----------
    d    : dict-achtig object met marktwaarden (ook pd.Series werkt via .get())
    mode : 'swing' of 'day'

    Returns
    -------
    dict met keys: scores, market_quality_score, decision, execution_window_score, summary
    """

    # ── Volatility 25% ──────────────────────────────────────────────
    v   = d.get("vix", 20) or 20
    sl  = d.get("vix_slope5d", 0) or 0
    pct = d.get("vix_pct1y", 50) or 50

    vs = 90 if v < 15 else (75 if v < 18 else (60 if v < 22 else (42 if v < 26 else (22 if v < 32 else 8))))
    vs += -10 if sl > .25 else (-5 if sl > .1 else (8 if sl < -.15 else 0))
    vs += -15 if pct > 80 else (-5 if pct > 65 else (10 if pct < 25 else 0))
    vol_score = max(0, min(100, vs))

    # ── Trend 20% ───────────────────────────────────────────────────
    p    = d.get("spy_price", 450) or 450
    m20  = d.get("spy_ma20", p) or p
    m50  = d.get("spy_ma50", p) or p
    m200 = d.get("spy_ma200", p) or p
    rsi  = d.get("spy_rsi14", 50) or 50

    ts = 50
    ts += 18 if p > m200 else -22
    ts += 13 if p > m50  else -13
    ts +=  9 if p > m20  else -9
    ts += 10 if (m20 > m50 > m200) else (-10 if (m20 < m50 < m200) else 0)
    ts +=  5 if rsi > 60 else (-8 if rsi < 40 else (-16 if rsi < 30 else 2))
    ts +=  5 if (d.get("qqq_vs_ma50", 0) or 0) > 0 else -5
    trend_score = max(0, min(100, ts))

    # ── Breadth 20% ─────────────────────────────────────────────────
    pp = d.get("pct_sectors_pos", 50) or 50
    ar = d.get("ad_ratio_est", 1) or 1
    mc = d.get("mclellan_est", 0) or 0
    nn = (d.get("nasdaq_nh_est", 100) or 100) - (d.get("nasdaq_nl_est", 80) or 80)

    bs = 50
    bs += 20 if pp > 70 else (8 if pp > 55 else (-15 if pp < 40 else (-25 if pp < 25 else 0)))
    bs += 10 if ar > 2  else (5 if ar > 1.2 else (-10 if ar < .8 else (-20 if ar < .5 else 0)))
    bs +=  8 if mc > 50 else (-10 if mc < -50 else 0)
    bs +=  8 if nn > 100 else (-10 if nn < -50 else 0)
    breadth_score = max(0, min(100, bs))

    # ── Momentum 25% ────────────────────────────────────────────────
    sp2 = d.get("sector_spread", 0) or 0
    s5  = d.get("spy_5d_chg", 0) or 0
    ps  = d.get("sectors_positive", 5) or 5

    ms = 50
    ms += 15 if sp2 > 8 else (5 if sp2 > 4 else (-10 if sp2 < 2 else 0))
    ms += 15 if s5 > 2  else (5 if s5 > 0  else (-15 if s5 < -2 else -5))
    ms += 15 if ps >= 8 else (8 if ps >= 6 else (-15 if ps <= 3 else (-5 if ps <= 5 else 0)))
    mom_score = max(0, min(100, ms))

    # ── Macro 10% ───────────────────────────────────────────────────
    fed = d.get("fed_stance", "NEUTRAL") or "NEUTRAL"
    tsl = d.get("tnx_slope5d", 0) or 0
    dsl = d.get("dxy_slope5d", 0) or 0
    tv  = d.get("tnx", 4.5) or 4.5

    mac = 50
    mac += 20 if fed == "DOVISH" else (5 if fed == "NEUTRAL" else -15)
    mac +=  8 if tsl < -.02 else (-10 if tsl > .03 else 0)
    mac +=  5 if dsl < -.1  else (-5  if dsl > .15 else 0)
    mac +=  5 if tv < 3.8   else (-8  if tv > 5.2  else 0)
    macro_score = max(0, min(100, mac))

    # ── Totaalscore & besluit ────────────────────────────────────────
    scores = {
        "volatility": {"score": vol_score,    "weight": .25},
        "trend":      {"score": trend_score,  "weight": .20},
        "breadth":    {"score": breadth_score,"weight": .20},
        "momentum":   {"score": mom_score,    "weight": .25},
        "macro":      {"score": macro_score,  "weight": .10},
    }
    total  = round(sum(v2["score"] * v2["weight"] for v2 in scores.values()), 1)
    thresh = THRESHOLDS.get(mode, THRESHOLDS["swing"])
    decision = "YES" if total >= thresh["yes"] else ("NO" if total < thresh["caution"] else "CAUTION")

    # ── Execution window ────────────────────────────────────────────
    reg = d.get("regime", "CHOP") or "CHOP"
    ew  = 50
    ew += 20 if reg == "UPTREND" else (-20 if reg == "DOWNTREND" else 0)
    ew += 15 if s5 > 1.5 else (5 if s5 > 0 else (-15 if s5 < -1.5 else -5))
    ew += (ps - 5) * 3
    exec_window = max(0, min(100, ew))

    # ── Tekstsamenvatting ────────────────────────────────────────────
    reg_desc = (
        "SPY in confirmed uptrend — MA stack aligned bullish." if reg == "UPTREND" else
        "SPY in downtrend — below key moving averages." if reg == "DOWNTREND" else
        "SPY in choppy range — no clear directional trend."
    )
    vd = (
        f"Volatility compressed (VIX {v:.1f}) — low-noise conditions." if v < 17 else
        f"Volatility moderate (VIX {v:.1f}) — standard swing environment." if v < 22 else
        f"Volatility elevated (VIX {v:.1f}) — reduce size, widen stops." if v < 28 else
        f"Volatility extreme (VIX {v:.1f}) — capital preservation first."
    )
    bd = (
        f"{int(ps)}/11 sectors advancing — " + (
            "broad participation supports risk-taking." if ps >= 7 else
            "mixed participation; stay selective."     if ps >= 5 else
            "narrow breadth, avoid chasing."
        )
    )
    top3 = d.get("top3_sectors", [])
    tnames = ", ".join(f"{e}({SECTORS.get(e,e)})" for e, _ in top3[:3]) if top3 else ""
    ld = f"Leaders: {tnames}." if tnames else ""
    ed = (
        "Execution window strong — breakouts holding, setups following through." if exec_window >= 70 else
        "Execution mixed — some setups working, stay selective."                 if exec_window >= 45 else
        "Execution poor — setups failing; wait for better window."
    )
    act = (
        "Full position sizing. Press risk on A+ setups."                                     if decision == "YES"     else
        "Half-size only. A+ setups with strong catalysts. Tight stops."                      if decision == "CAUTION" else
        "Stand aside. Preserve capital. Build watchlist."
    )
    summary = f"{reg_desc} {vd} {bd} {ld} {ed} → {act}"

    return {
        "scores":                scores,
        "market_quality_score":  total,
        "decision":              decision,
        "execution_window_score": exec_window,
        "summary":               summary,
    }
