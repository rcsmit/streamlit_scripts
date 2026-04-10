import importlib
import traceback
import os
import platform
import streamlit as st
import sys
sys.path.append("bloomberg_dashboard")

st.set_page_config(page_title="Streamlit scripts of René Smit")

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")


def dynamic_import(module):
    """Import a module stored in a variable."""
    return importlib.import_module(module)


# ---------------------------------------------------------------------------
# Script catalogue  [display_label, module_name, description]
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
    ["[79] Berlin",         "berlin",      "POI's in Berlin"],
    ["[80] Palentir style dashboard",         "palantir_dashboard",      "Palantir-style dashboard"],
    ["[81] From SMA to LOESS",         "sma_vs_loess",      "Which LOESS-span for a given SMA?"],
]

# ---------------------------------------------------------------------------
# Categories  — (letter_key, display_name, [choice_indices])
#
# letter_key is exposed as  ?cat=A  …  ?cat=I  in the URL.
# ---------------------------------------------------------------------------
CATEGORIES = [
    ("A", "🏠  Home",                  [0],                                      "#6C8EBF"),
    ("B", "🗺️  Maps & Travel",         [17, 34, 40, 67, 79],                     "#4ECDC4"),
    ("B", "🔤  Language & Text",       [3, 12, 13, 46, 72],                      "#F7B731"),
    ("D", "🎨  Fun & Creative",        [1, 9, 19, 21, 60, 62, 66, 70],           "#FF6B9D"),
    ("E", "🏃  Health & Lifestyle",    [6, 36, 53],                               "#55EFC4"),
    ("F", "🏕️  Camping & Rep life",   [5, 11, 25, 41, 43],                      "#E17055"),
    ("G", "🌦️  Weather & Nature",      [2, 10, 22, 30, 42, 50, 59, 69, 71],     "#74B9FF"),
    ("H", "📊  Data & Analysis",       [14, 23, 24, 26, 27, 31, 39, 51, 61, 64, 68, 74,81], "#A29BFE"),
    ("I", "🛠️  Tools & Utilities",     [4, 37, 48, 55, 56, 58],                  "#B2BEC3"),
    ("J", "👴  Life expectancy",       [35, 47, 49, 54],                          "#FDCB6E"),
    ("K", "💰  Finance & Income",      [15, 16, 18, 28, 29, 32, 44, 65, 76, 77], "#27AE60"),
    ("L", "📈  Trading",               [7, 8, 20, 33, 38, 45, 52, 57, 78],       "#E67E22"),
    ("M", "🎨  Streamlit layouts",     [75, 63, 73,80],                              "#9B59B6"),
]



# Derived lookups (built once at import time)
_choice_to_cat_i = {}          # choice_index  → index in CATEGORIES
_letter_to_cat_i = {}          # "B"           → index in CATEGORIES
for _cat_i, (_letter, _, _idxs, color) in enumerate(CATEGORIES):
    _letter_to_cat_i[_letter.upper()] = _cat_i
    for _idx in _idxs:
        _choice_to_cat_i[_idx] = _cat_i

def give_options_categories():
    return options, CATEGORIES

def main():
    # -----------------------------------------------------------------------
    # 1.  Read query params
    #     Supported:
    #       ?choice=42          → open script 42 (and its category)
    #       ?cat=C              → open category C, show its first script
    #       ?choice=42&cat=C    → choice wins; cat is kept in sync
    # -----------------------------------------------------------------------
    raw_choice = st.query_params.get("choice", None)
    raw_cat    = st.query_params.get("cat",    None)

    active_choice = 0
    if raw_choice is not None:
        try:
            c = int(raw_choice)
            if 0 <= c < len(options):
                active_choice = c
        except ValueError:
            pass

    # Derive active category
    if raw_choice is not None:
        # script deeplink → its owning category
        active_cat_i = _choice_to_cat_i.get(active_choice, 0)
    elif raw_cat is not None:
        # category deeplink → open category, land on first script in it
        active_cat_i = _letter_to_cat_i.get(raw_cat.upper(), 0)
        active_choice = CATEGORIES[active_cat_i][2][0]
    else:
        active_cat_i = 0

    # -----------------------------------------------------------------------
    # 2.  Sidebar — accordion categories with script buttons
    # -----------------------------------------------------------------------
    selected_choice = active_choice
    selected_cat_i  = active_cat_i

    with st.sidebar:
        st.markdown("### 📂 René's Scripts")
        st.caption("Pick a category, then a script. Options/parameters are below the menu")
        st.markdown("---")

        for cat_i, (letter, cat_name, cat_indices, color) in enumerate(CATEGORIES):
            is_open = (cat_i == active_cat_i)

            with st.expander(f"**[{letter}]** {cat_name}", expanded=is_open):
                for idx in cat_indices:
                    label     = options[idx][0]
                    desc      = options[idx][2]
                    is_active = (idx == active_choice)

                    # Strip the original [N] number — category is navigation now
                    short     = label.split("] ", 1)[-1]
                    number =  label.split(" ", 1)[0]
                    btn_label = ("▶ " if is_active else "") + short

                    if st.button(
                        btn_label,
                        key=f"btn_{idx}",
                        use_container_width=True,
                        help=f"{desc} {number}",
                        type="primary" if is_active else "secondary",
                    ):
                        selected_choice = idx
                        selected_cat_i  = cat_i

        st.markdown("---")
        # st.caption(
        #     "**Deeplinks**  \n"
        #     "`?choice=42` → script 42  \n"
        #     "`?cat=C` → category C  \n"
        #     "`?choice=42&cat=C` → both"
        # )

    # -----------------------------------------------------------------------
    # 3.  Keep both query params in sync so every URL is a valid deeplink
    # -----------------------------------------------------------------------
    current_letter = CATEGORIES[selected_cat_i][0]
    st.query_params["choice"] = str(selected_choice)
    st.query_params["cat"]    = current_letter

    # -----------------------------------------------------------------------
    # 4.  Dynamically import and run the selected module
    # -----------------------------------------------------------------------
    m = options[selected_choice][1].replace(" ", "_")

    try:
        module = dynamic_import(m)
    except Exception as e:
        st.error(f"Module '{m}' not found or error in the script")
        st.warning(str(e))
        st.warning(traceback.format_exc())
        st.stop()

    try:
        module.main()
    except Exception as e:
        st.error(f"Function 'main()' in module '{m}' not found or error in the script")
        st.warning(str(e))
        st.warning(traceback.format_exc())
        st.stop()

if __name__ == "__main__":
    main()
