import streamlit as st

    
#---------------------------------------------------------------------------
# welcome.py  —  Landing page for rcsmit.streamlit.app
# options + CATEGORIES are kept in sync with menu_streamlit.py
# ---------------------------------------------------------------------------

options = [
    ["[0] welcome",                         "welcome",                     "Landing page and overview of all available scripts"],
    ["[1] newagebullshitgenerator",         "newagebullshitgenerator",     "Generates random new age sentences using word lists"],
    ["[2] KNMI grafieken",                  "show_knmi",                   "Dutch weather data visualization with multiple analysis modes"],
    ["[3] Text generator",                  "txt_generator_streamlit",     "Markov chain text generator from meditation scripts"],
    ["[4] YT transcriber",                  "YoutubeTranscriber_streamlit","YouTube video transcriber with translation support"],
    ["[5] Schoonmaaktijden",                "schoonmaaktijden",            "Cleaning time analysis fitted to Weibull distribution"],
    ["[6] Show sportactivities",            "show_sportactivities",        "Garmin activity tracker with running/sport statistics"],
    ["[7] YFinance info",                   "yfinance_info",               "Stock price Bollinger Bands trading strategy simulator"],
    ["[8] Crypto portfolio",                "crypto_portfolio",            "Bitcoin/crypto portfolio tracker with cost tracking"],
    ["[9] Yepcharts",                       "yepcharts",                   "iTunes hitlist fetcher and chart viewer for various countries"],
    ["[10] Zonnepanelen",                   "zonnepanelen",                "Solar panel yield analysis vs meteorological factors"],
    ["[11] Occupation Camping",             "read_bezetting_test",         "Accommodation occupancy analysis from planning sheets"],
    ["[12] Whatsapp analyzer",              "whatsapp_analyzer",           "WhatsApp chat analysis with statistics and visualization"],
    ["[13] Breezertaal converter",          "make_breezer_taal",           "Text converter that alternates case and replaces vowels"],
    ["[14] Fit to data",                    "fit_to_data_streamlit",       "Curve fitting for Dutch housing allowance formula"],
    ["[15] Inkomstenbelasting",             "inkomstenbelasting",          "Dutch income tax calculator with scenario simulations"],
    ["[16] Exchange money",                 "exchange_money",              "Thailand EUR-THB exchange rate comparison tool"],
    ["[17] Conscious communities",          "conscious_communities",       "Interactive map of ecovillages and intentional communities worldwide"],
    ["[18] How Much Month I need to work",  "balansprognose",              "Financial balance forecast for part-time work scenarios"],
    ["[19] Tarot Symbols",                  "tarot_symbols",               "Tarot card symbol meanings with images and GPT analysis"],
    ["[20] Dollar Cost Average",            "dollar_cost_average",         "Bitcoin DCA strategy calculator vs lump sum investing"],
    ["[21] Midjourney prompt generator",    "midjourney_prompt_generator", "Generates Midjourney AI art prompts from random data"],
    ["[22] Historical weather",             "weather_open_meteo",          "Historical weather data from Open-Meteo API with visualizations"],
    ["[23] KNMI trendline LOESS",           "loess",                       "KNMI temperature trend analysis using LOESS (statsmodels)"],
    ["[24] KNMI trendline LOESS - scikit",  "loess_scikitmisc",            "KNMI temperature trend analysis using LOESS (scikit-misc)"],
    ["[25] Yield management",               "yield_management",            "Hotel yield management and pricing optimization tool"],
    ["[26] Meat consumption",               "meat_consumption",            "Global meat consumption and health data correlation analysis"],
    ["[27] Waiting times",                  "waiting_times_disney_OO",     "Disney theme park queue/waiting time simulation (SimPy)"],
    ["[28] Extra pension",                  "extra_pension_OOP",           "Retirement planning calculator with windfalls and projections"],
    ["[29] Studyloan",                      "studyloan",                   "Dutch student loan repayment calculator and visualization"],
    ["[30] Gasverbruik",                    "gas_stand_vs_temp",           "Gas usage vs outdoor temperature analysis with LOESS smoothing"],
    ["[31] Sterfte vs temp",                "sterfte_temperatuur",         "Mortality and temperature correlation analyses (NL data)"],
    ["[32] Tax PvdA GL",                    "tax_pvda_gl",                 "Dutch tax bracket calculator with editable progressive rates"],
    ["[33] Momentum currency",              "momentum_koersen",            "Stock momentum trading strategy with Bollinger Bands"],
    ["[34] Acroyoga groups",                "acroyoga_groups",             "AcroYoga communities map with data from Google Sheets"],
    ["[35] Life Expectancy NL",             "life_expectancy_nl",          "Dutch life expectancy calculator using Monte Carlo simulations"],
    ["[36] Weblogs",                        "weblogs",                     "Blog entry reader from Excel with statistics and visualization"],
    ["[37] Yepcheck",                       "yepcheck",                    "SQLite database viewer displaying tables and their contents"],
    ["[38] Bollinger Bot",                  "bollinger_bot",               "Bollinger Bands trading bot with backtesting strategy"],
    ["[39] No shows",                       "no_shows",                    "Airline overbooking simulation with show-up probability analysis"],
    ["[40] Rome infomap",                   "rome",                        "Rome attractions map with Google Sheets data and filtering"],
    ["[41] Leisurewords",                   "leisurewords",                "Vocabulary lookup with multilingual translation & CSV export"],
    ["[42] Feels like temp.",               "feels_like_temperature",      "Temperature perception calculator: heat index & wind chill"],
    ["[43] Cleaning simulation",            "cleaning_simulation",         "Monte Carlo house cleaning simulation with probability analysis"],
    ["[44] CPI vs loon",                    "lonen_inflatie",              "Dutch wage and inflation analyzer using CBS statistics (= [65])"],
    ["[45] Crypto Dashboard",               "crypto_dashboard",            "Cryptocurrency trading dashboard with technical indicators"],
    ["[46] Transcript Thai",                "transcript_thai",             "Thai language transcription with phonetic conversion & translation"],
    ["[47] Levensverw. in tijd",            "levensverw_door_tijd_heen",   "Dutch life expectancy trends by age/gender using CBS data"],
    ["[48] Images2pdf",                     "images_to_pdf",               "Image/PDF processor for rotation, contrast and resizing"],
    ["[49] Viager calculation",             "viager_calculation",          "Real estate viager calculator using mortality tables & simulation"],
    ["[50] Neerslaganalyse",                "neerslaganalyse",             "Netherlands precipitation trend analysis with Mann-Kendall test"],
    ["[51] Seasonality",                    "seasonality",                 "Time series seasonality detector with decomposition & statistical tests"],
    ["[52] Vix Bot",                        "vix_bot",                     "VIX-based algorithmic trading bot with multiple strategy backtesting"],
    ["[53] GPX analyzer",                   "gpx_analyzer",                "GPS route analyzer calculating elevation profiles & effort distances"],
    ["[54] Nog in leven over 40 jaar",      "nog_in_leven_over_40_jaar",   "Dutch population survival projector for 85+ age group"],
    ["[55] WOO PDF to table",               "pdf_to_table",                "Dutch WOO document PDF parser extracting text to CSV/Excel"],
    ["[56] Read ChatGPT-json",              "read_conversations_json",     "ChatGPT conversations.json reader with display & export modes"],
    ["[57] Compound ROI",                   "compound_roi_calculator",     "Investment return calculator (ROI/CAGR) for stocks & crypto"],
    ["[58] Doorsigns",                      "public_doorsigns",            "PDF door sign generator with custom fonts & batch processing"],
    ["[59] Sunset Azimut",                  "sunset_azimut",               "Sunset direction calculator showing azimuth by latitude & year"],
    ["[60] Strudel Drumsequencer",          "strudel_drumbox",             "16-step drum sequencer for Strudel live coding with HTML embed"],
    ["[61] Vrijkomende woningen",           "aantal_vrijkomende_woningen", "Dutch housing market simulator modeling supply & demand"],
    ["[62] Tabs2notes",                     "tabs2notes",                  "Guitar tablature to musical notes converter with ASCII notation"],
    ["[63] Streamlit layouts",              "streamlit_layouts",           "Streamlit UI component showcase: cards, timelines, metrics"],
    ["[64] Analyze reviews",                "analyze_reviews_dummy",       "Customer satisfaction analyzer with NPS, heatmaps & trends"],
    ["[65] Lonen vs inflatie",              "lonen_inflatie",              "Dutch wage and inflation analyzer using CBS statistics"],
    ["[66] Generate monopoly",              "generate_monopoly",           "SVG Monopoly board editor with placeholder replacement & mapping"],
    ["[67] EasyMapNL",                      "easy_map_nl",                 "Make a custom map of the Netherlands"],
    ["[68] 2e kamer",                       "tweedekamer",                 "Verkiezings analyses 2e kamer"],
    ["[69] knmi vs openmeteo",              "compare_knmi_openmeteo",      "Seizoensinfo: KNMI vs Open-Meteo weather data comparison"],
    ["[70] Year timeline",                  "how_the_year_went",           "Timeline of the year with subjective length of months"],
    ["[71] Airquality Chiang Mai",          "airquality_chiangmai",        "Historical airquality in Chiang Mai"],
    ["[72] Letter Frequentie Analyse",      "letter_count",                "Welk woord heeft de meeste van elke letter?"],
    ["[73] Streamlit dashboard apps",       "streamlit_demo_apps",         "Demo templates made by Streamlit"],
    ["[74] Birthday heatmap",               "birthday_heatmap",            "Heatmap geboortedatums+afwijkingen per gemeente"],
    ["[75] Bloomberg Dashboard",            "bb_dashboard",                "Bloomberg dashboard + advice"],
    ["[76] Pensioencalculator2",            "pensioencalculator2",         "Another pension calculator"],
    ["[77] ZZP tarieven vs bullshitgehalte","zzp_bullshit",               "Hoe hoger de bullshit, des te hoger het tarief"],
    ["[78] Backtest Fathers rules",         "backtest_fathers_rules",      "Is the wisdom about trading stocks of Father right?"],
    ["[79] Berlin",                         "berlin",                      "POIs in Berlin"],
]

# Same as menu_streamlit.py but with an extra color field appended
CATEGORIES = [
    ("A", "🏠  Home",                  [0],                                      "#6C8EBF"),
    ("B", "🗺️  Maps & Travel",         [17, 34, 40, 67, 79],                     "#4ECDC4"),
    ("B", "🔤  Language & Text",       [3, 12, 13, 46, 72],                      "#F7B731"),
    ("D", "🎨  Fun & Creative",        [1, 9, 19, 21, 60, 62, 66, 70],           "#FF6B9D"),
    ("E", "🏃  Health & Lifestyle",    [6, 36, 53],                               "#55EFC4"),
    ("F", "🏕️  Camping & Rep life",   [5, 11, 25, 41, 43],                      "#E17055"),
    ("G", "🌦️  Weather & Nature",      [2, 10, 22, 30, 42, 50, 59, 69, 71],     "#74B9FF"),
    ("H", "📊  Data & Analysis",       [14, 23, 24, 26, 27, 31, 39, 51, 61, 64, 68, 74], "#A29BFE"),
    ("I", "🛠️  Tools & Utilities",     [4, 37, 48, 55, 56, 58],                  "#B2BEC3"),
    ("J", "👴  Life expectancy",       [35, 47, 49, 54],                          "#FDCB6E"),
    ("K", "💰  Finance & Income",      [15, 16, 18, 28, 29, 32, 44, 65, 76, 77], "#27AE60"),
    ("L", "📈  Trading",               [7, 8, 20, 33, 38, 45, 52, 57, 78],       "#E67E22"),
    ("M", "🎨  Streamlit layouts",     [75, 63, 73],                              "#9B59B6"),
]


def main():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8f4f8 100%);
        border: 1px solid #d1dce8;
        border-radius: 16px;
        padding: 2.6rem 2.4rem 2rem;
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -50px; right: -50px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(78,205,196,0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #1a2340;
        margin: 0 0 0.2rem;
        letter-spacing: -1px;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: #5a6a82;
        margin: 0 0 1.2rem;
        font-weight: 300;
    }
    .hero-tag {
        display: inline-block;
        background: #ffffff;
        border: 1px solid #c7d4e8;
        color: #3b4f72;
        border-radius: 20px;
        padding: 3px 11px;
        font-size: 0.73rem;
        font-family: 'Space Mono', monospace;
        margin: 0 4px 5px 0;
    }
    .stats-row {
        display: flex; gap: 0.8rem; flex-wrap: wrap; margin-top: 1.4rem;
    }
    .stat-pill {
        background: #ffffff;
        border: 1px solid #d1dce8;
        border-radius: 10px;
        padding: 0.55rem 1rem;
        text-align: center;
        min-width: 80px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stat-number {
        font-family: 'Space Mono', monospace;
        font-size: 1.25rem;
        font-weight: 700;
        color: #2563eb;
    }
    .stat-label {
        font-size: 0.65rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* category cards */
    .cat-card {
        background: #ffffff;
        border: 1px solid #e4eaf2;
        border-radius: 12px;
        padding: 1.15rem 1.2rem 0.7rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .cat-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid currentColor;
        opacity: 1;
    }
    .script-row {
        display: flex;
        align-items: flex-start;
        gap: 0.55rem;
        padding: 0.32rem 0;
        border-bottom: 1px solid #f1f5f9;
    }
    .script-row:last-child { border-bottom: none; }
    .script-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        margin-top: 6px;
        flex-shrink: 0;
    }
    .script-num {
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        color: #b0bec5;
        margin-top: 2px;
        flex-shrink: 0;
        width: 20px;
    }
    .script-name {
        font-size: 0.845rem;
        font-weight: 600;
        color: #1e293b;
        line-height: 1.3;
    }
    .script-desc {
        font-size: 0.74rem;
        color: #94a3b8;
        line-height: 1.35;
    }

    .about-box {
        background: #f8faff;
        border: 1px solid #dce5f5;
        border-radius: 12px;
        padding: 1.6rem 1.8rem;
        margin-top: 2rem;
    }
    .about-box h3 {
        font-family: 'Space Mono', monospace;
        color: #2563eb;
        margin-top: 0;
        font-size: 0.95rem;
    }
    .about-box p { color: #475569; line-height: 1.7; font-size: 0.875rem; margin: 0 0 0.6rem; }
    .about-box a { color: #2563eb; text-decoration: none; font-weight: 500; }
    .about-box a:hover { text-decoration: underline; }

    footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero ─────────────────────────────────────────────────────────────
    tags = ["Python", "Streamlit", "Plotly", "CBS data", "KNMI", "Finance", "Open source"]
    tag_html = "".join(f'<span class="hero-tag">{t}</span>' for t in tags)

    n_scripts = len(options) - 1  # exclude welcome itself
    n_cats    = len(CATEGORIES) - 1
    stats_html = "".join(
        f'<div class="stat-pill"><div class="stat-number">{v}</div><div class="stat-label">{k}</div></div>'
        for k, v in [("Scripts", str(n_scripts)), ("Categories", str(n_cats)), ("Years active", "5+")]
    )

    st.markdown(f"""
    <div class="hero">
        <div class="hero-title">René Smit</div>
        <div class="hero-sub">Data analyst · Streamlit builder · Nomadic freelancer</div>
        {tag_html}
        <div class="stats-row">{stats_html}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("👈 **Use the sidebar** to open any script, or browse the full catalogue below.")
    st.markdown("---")

    # ── Category grid — driven entirely by CATEGORIES + options ──────────
    cols = st.columns(2, gap="medium")
    col_idx = 0
    readme_txt =""
    for letter, cat_name, cat_indices, color in CATEGORIES:
        if cat_indices == [0]:   # skip Home meta-entry
            continue

        scripts_html = ""
        readme_txt_cat=""
        for idx in cat_indices:
            label = options[idx][0]
            desc  = options[idx][2]
            num   = label.split("]")[0].replace("[", "").strip()
            name  = label.split("] ", 1)[-1]
            scripts_html += f"""
<div class='script-row'>
<div class='script-dot' style='background:{color}'></div>
<div class='script-num'>{num}</div>
<div>
<div class='script-name'>{name}</div>
<div class='script-desc'>{desc}</div>
</div>
</div>"""
            readme_txt_cat  += f"""
| {num} | [{name}](https://rcsmit.streamlit.app/?choice={num}) | {desc} |"""
        cols[col_idx % 2].markdown(f"""
<div class='cat-card'>
<div class='cat-header' style='color:{color}; border-color:{color}'>
{cat_name}
</div>
{scripts_html}
</div>
        """, unsafe_allow_html=True)
        readme_txt  +=f"""
### {cat_name}
| # | Script | Description |
|---|--------|-------------|{readme_txt_cat}
"""
        col_idx += 1

    # ── About ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="about-box">
        <h3>👋 About</h3>
        <p>
            I'm René — a Dutch multidisciplinary freelancer with a nomadic lifestyle across
            Da Nang, Chiang Mai, Bali, and the Netherlands.
            This portfolio is a living collection of Streamlit apps built over the years:
            data explorations, financial tools, trading backtests, Dutch open data, maps, and the occasional bit of fun.
        </p>
        <p>
            Most scripts pull from Dutch open data (CBS, KNMI, Eurostat) or public APIs and are written
            in Python with Streamlit, Plotly, and pandas. Everything is open source.
        </p>
        <p>
            🔗 <a href="https://github.com/rcsmit/streamlit_scripts" target="_blank">GitHub</a> &nbsp;·&nbsp;
            🌐 <a href="https://rene-smit.com" target="_blank">rene-smit.com</a> &nbsp;·&nbsp;
            📊 <a href="https://rcsmit.streamlit.app" target="_blank">rcsmit.streamlit.app</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Contents for readme.MD"):
        st.code(readme_txt)
if __name__ == "__main__":
    main()