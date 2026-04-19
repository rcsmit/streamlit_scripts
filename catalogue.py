# catalogue.py
# ---------------------------------------------------------------------------
# Single source of truth for the script catalogue and category definitions.
# Both menu_streamlit.py and welcome.py import from here.
#
# Usage:
#   from catalogue import options, CATEGORIES, give_options_categories
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Script catalogue  [display_label, module_name, description]
# ---------------------------------------------------------------------------

import streamlit as st
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
    ["[77] ZZP tarieven vs bullshitgehalte","zzp_bullshit",                "Hoe hoger de bullshit, des te hoger het tarief"],
    ["[78] Backtest Fathers rules",         "backtest_fathers_rules",      "Is the wisdom about trading stocks of Father right?"],
    ["[79] Berlin",                         "berlin",                      "POI's in Berlin"],
    ["[80] Palentir style dashboard",       "palantir_dashboard",          "Palantir-style dashboard"],
    ["[81] From SMA to LOESS",              "sma_vs_loess",                "Which LOESS-span for a given SMA?"],
    ["[82] AEX dashboard",                  "aex_dashboard",               "AEX fund performance dashboard"],
    ["[83] Huis model",                     "huis_model",                  "Aankoop huis als belegging?"],
       ["[84] Loess viewer","loess_viewer", "Plot LOESS lines in a dataset "],
]

# ---------------------------------------------------------------------------
# Categories  — (letter_key, display_name, [choice_indices], hex_color)
#
# Rules:
#   - letter_key must be unique across all entries.
#   - Each script index may appear in at most one category.
#   - Scripts absent from every category are auto-collected into "Z".
# ---------------------------------------------------------------------------
CATEGORIES = [
    ("A", "🏠  Home",                  [0],                                           "#6C8EBF"),
    ("B", "🗺️  Maps & Travel",         [17, 34, 40, 67, 79],                          "#4ECDC4"),
    ("C", "🔤  Language & Text",       [3, 12, 13, 46, 72],                           "#F7B731"),
    ("D", "🎨  Fun & Creative",        [1, 9, 19, 21, 60, 62, 66, 70],                "#FF6B9D"),
    ("E", "🏃  Health & Lifestyle",    [6, 36, 53],                                    "#55EFC4"),
    ("F", "🏕️  Camping & Rep life",   [5, 11, 25, 41, 43],                           "#E17055"),
    ("G", "🌦️  Weather & Nature",      [2, 10, 22, 30, 42, 50, 59, 69, 71],          "#74B9FF"),
    ("H", "📊  Data & Analysis",       [14, 23, 24, 26, 27, 31, 39, 51, 61, 64, 68, 74, 81,84], "#A29BFE"),
    ("I", "🛠️  Tools & Utilities",     [4, 37, 48, 55, 56, 58],                       "#B2BEC3"),
    ("J", "👴  Life expectancy",       [35, 47, 49, 54],                               "#FDCB6E"),
    ("K", "💰  Finance & Income",      [15, 16, 18, 28, 29, 32, 44, 65, 76, 77, 83],  "#27AE60"),
    ("L", "📈  Trading",               [7, 8, 20, 33, 38, 45, 52, 57, 78, 82],        "#E67E22"),
    ("M", "🎨  Streamlit layouts",     [75, 63, 73, 80],                               "#9B59B6"),
]


# ---------------------------------------------------------------------------
# Safety check 1: duplicate category letter keys
# Raises ValueError at import time so the error surfaces immediately,
# whether running under Streamlit or plain Python.
# ---------------------------------------------------------------------------
def _check_no_duplicate_letters() -> None:
    seen: dict[str, str] = {}
    errors: list[str] = []
    for letter, cat_name, _idxs, _color in CATEGORIES:
        key = letter.upper()
        if key in seen:
            errors.append(
                f"  letter '{key}' used by '{seen[key]}' and '{cat_name}'"
            )
        else:
            seen[key] = cat_name
    if errors:
        # raise ValueError(
        #     "catalogue.py — duplicate category letters:\n" + "\n".join(errors)
        # )
        st.error(
            "catalogue.py — duplicate category letters:\n" + "\n".join(errors)
        )

_check_no_duplicate_letters()


# ---------------------------------------------------------------------------
# Safety check 2: script index appears in more than one category
# ---------------------------------------------------------------------------
def _check_no_duplicate_scripts() -> None:
    seen: dict[int, str] = {}
    errors: list[str] = []
    for _letter, cat_name, idxs, _color in CATEGORIES:
        for idx in idxs:
            if idx in seen:
                label = options[idx][0] if 0 <= idx < len(options) else str(idx)
                errors.append(
                    f"  [{idx}] '{label}' in '{seen[idx]}' and '{cat_name}'"
                )
            else:
                seen[idx] = cat_name
    if errors:
        print(
             "catalogue.py — scripts listed in multiple categories:\n" + "\n".join(errors)
        )
        
        # raise ValueError(
            # "catalogue.py — scripts listed in multiple categories:\n" + "\n".join(errors)
        # )


_check_no_duplicate_scripts()


# ---------------------------------------------------------------------------
# Auto-collect scripts not assigned to any category → category Z
# ---------------------------------------------------------------------------
_all_categorised: set[int] = {
    idx
    for _letter, _name, idxs, _color in CATEGORIES
    for idx in idxs
}
_uncategorised: list[int] = [i for i in range(len(options)) if i not in _all_categorised]

if _uncategorised:
    CATEGORIES.append(
        ("Z", "❓  Uncategorised", _uncategorised, "#95A5A6")
    )


# ---------------------------------------------------------------------------
# Derived lookups — built once at import time, re-exported for convenience
# ---------------------------------------------------------------------------
# choice_index → index in CATEGORIES
CHOICE_TO_CAT_I: dict[int, int] = {}
# "B" → index in CATEGORIES
LETTER_TO_CAT_I: dict[str, int] = {}

for _cat_i, (_letter, _name, _idxs, _color) in enumerate(CATEGORIES):
    LETTER_TO_CAT_I[_letter.upper()] = _cat_i
    for _idx in _idxs:
        CHOICE_TO_CAT_I[_idx] = _cat_i


def give_options_categories():
    """Convenience accessor kept for backwards compatibility."""
    return options, CATEGORIES
