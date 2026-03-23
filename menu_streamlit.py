import importlib
import traceback
import os
import platform
import streamlit as st

st.set_page_config(page_title="Streamlit scripts of René Smit")

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

def dynamic_import(module):
    """Import a module stored in a variable

    Args:
        module (string): The module to import

    Returns:
        the module you want
    """
    return importlib.import_module(module)

def main():
    """_summary_
    """    
    if platform.processor() != "":
        arr = os.listdir("C:\\Users\\rcxsm\\Documents\\python_scripts\\streamlit_scripts")
    else:
        arr = os.listdir()

    counter = 1
    options = [["[0] welcome","welcome","Landing page and overview of all available scripts"],
                ["[1] newagebullshitgenerator","newagebullshitgenerator","Generates random new age sentences using word lists"],
                ["[2] KNMI grafieken","show_knmi","Dutch weather data visualization with multiple analysis modes"],
                ["[3] Text generator","txt_generator_streamlit","Markov chain text generator from meditation scripts"],
                ["[4] YT transcriber","YoutubeTranscriber_streamlit","YouTube video transcriber with translation support"],
                ["[5] Schoonmaaktijden", "schoonmaaktijden","Cleaning time analysis fitted to Weibull distribution"],
                ["[6] Show sportactivities", "show_sportactivities","Garmin activity tracker with running/sport statistics"],
                ["[7] YFinance info", "yfinance_info","Stock price Bollinger Bands trading strategy simulator"],
                ["[8] Crypto portfolio", "crypto_portfolio","Bitcoin/crypto portfolio tracker with cost tracking"],
                ["[9] Yepcharts", "yepcharts","iTunes hitlist fetcher and chart viewer for various countries"],
                ["[10] Zonnepanelen", "zonnepanelen","Solar panel yield analysis vs meteorological factors"],
                ["[11] Occupation Camping", "read_bezetting_test","Accommodation occupancy analysis from planning sheets"], #was occupation_camping
                ["[12] Whatsapp analyzer", "whatsapp_analyzer","WhatsApp chat analysis with statistics and visualization"],
                ["[13] Breezertaal converter", "make_breezer_taal","Text converter that alternates case and replaces vowels"],
                ["[14] Fit to data", "fit_to_data_streamlit","Curve fitting for Dutch housing allowance formula"],
                ["[15] Inkomstenbelasting", "inkomstenbelasting","Dutch income tax calculator with scenario simulations"],
                ["[16] Exchange money", "exchange_money","Thailand EUR-THB exchange rate comparison tool"],
                ["[17] Conscious communities", "conscious_communities","Interactive map of ecovillages and intentional communities worldwide"],
                ["[18] How Much Month I need to work", "balansprognose","Financial balance forecast for part-time work scenarios"],
                ["[19] Tarot Symbols", "tarot_symbols","Tarot card symbol meanings with images and GPT analysis"],
                ["[20] Dollar Cost Average", "dollar_cost_average","Bitcoin DCA strategy calculator vs lump sum investing"],
                ["[21] Midjourney prompt generator", "midjourney_prompt_generator","Generates Midjourney AI art prompts from random data"],
                ["[22] Historical weather", "weather_open_meteo","Historical weather data from Open-Meteo API with visualizations"],
                ["[23] KNMI trendline LOESS", "loess","KNMI temperature trend analysis using LOESS (statsmodels)"],
                ["[24] KNMI trendline LOESS - scikit-misc", "loess_scikitmisc","KNMI temperature trend analysis using LOESS (scikit-misc)"],
                ["[25] Yield management", "yield_management","Hotel yield management and pricing optimization tool"],
                ["[26] Meat consumption", "meat_consumption","Global meat consumption and health data correlation analysis"],
                ["[27] Waiting times", "waiting_times_disney_OO","Disney theme park queue/waiting time simulation (SimPy)"],
                ["[28] Extra pension", "extra_pension_OOP","Retirement planning calculator with windfalls and projections"],
                ["[29] Studyloan", "studyloan","Dutch student loan repayment calculator and visualization"],
                ["[30] Gasverbruik", "gas_stand_vs_temp","Gas usage vs outdoor temperature analysis with LOESS smoothing"],
                ["[31] Sterfte vs temp", "sterfte_temperatuur","Mortality and temperature correlation analyses (NL data)"],
                ["[32] Tax PvdA GL", "tax_pvda_gl","Dutch tax bracket calculator with editable progressive rates"],
                ["[33] Momentum currency", "momentum_koersen","Stock momentum trading strategy with Bollinger Bands"],
                ["[34] Acroyoga groups", "acroyoga_groups","AcroYoga communities map with data from Google Sheets"],
                ["[35] Life Expectancy NL","life_expectancy_nl","Dutch life expectancy calculator using Monte Carlo simulations"],
                ["[36] Weblogs","weblogs","Blog entry reader from Excel with statistics and visualization"],
                ["[37] Yepcheck","yepcheck","SQLite database viewer displaying tables and their contents"],
                ["[38] Bollinger Bot","bollinger_bot","Bollinger Bands trading bot with backtesting strategy"],
                ["[39] No shows","no_shows","Airline overbooking simulation with show-up probability analysis"],
                ["[40] Rome infomap","rome","Rome attractions map with Google Sheets data and filtering"],
                ["[41] Leisurewords","leisurewords","Vocabulary lookup with multilingual translation & CSV export"],
                ["[42] Feels like temp.","feels_like_temperature","Temperature perception calculator: heat index & wind chill"],
                ["[43] Cleaning simulation","cleaning_simulation","Monte Carlo house cleaning simulation with probability analysis"],
                ["[44] CPI vs loon","lonen_inflatie","Dutch wage and inflation analyzer using CBS statistics (= [65])"], #was cpi_loon
                ["[45] Crypto Dashboard","crypto_dashboard","Cryptocurrency trading dashboard with technical indicators"],
                ["[46] Transcript Thai","transcript_thai","Thai language transcription with phonetic conversion & translation"],
                ["[47] Levensverw. in tijd ", "levensverw_door_tijd_heen","Dutch life expectancy trends by age/gender using CBS data"],
                ["[48] Images2pdf","images_to_pdf","Image/PDF processor for rotation, contrast and resizing"],
                ["[49] Viager calculation ","viager_calculation","Real estate viager calculator using mortality tables & simulation"],
                ["[50] Neerslaganalyse ","neerslaganalyse","Netherlands precipitation trend analysis with Mann-Kendall test"],
                ["[51] Seasonality ","seasonality","Time series seasonality detector with decomposition & statistical tests"],
                ["[52] Vix Bot ","vix_bot","VIX-based algorithmic trading bot with multiple strategy backtesting"],
                ["[53] GPX analyzer ","gpx_analyzer","GPS route analyzer calculating elevation profiles & effort distances"],
                ["[54] Nog in leven over 40 jaar ","nog_in_leven_over_40_jaar","Dutch population survival projector for 85+ age group"],
                ["[55] WOO PDF to table ","pdf_to_table","Dutch WOO document PDF parser extracting text to CSV/Excel"],
                ["[56] Read ChatGPT-json ","read_conversations_json","ChatGPT conversations.json reader with display & export modes"],
                ["[57] Compound ROI ","compound_roi_calculator","Investment return calculator (ROI/CAGR) for stocks & crypto"],
                ["[58] Doorsigns ","public_doorsigns","PDF door sign generator with custom fonts & batch processing"],
                ["[59] Sunset Azimut ","sunset_azimut","Sunset direction calculator showing azimuth by latitude & year"],
                ["[60] Strudel Drumsequencer ","strudel_drumbox","16-step drum sequencer for Strudel live coding with HTML embed"],
                ["[61] Vrijkomende woningen ","aantal_vrijkomende_woningen","Dutch housing market simulator modeling supply & demand"],
                ["[62] Tabs2notes ","tabs2notes","Guitar tablature to musical notes converter with ASCII notation"],
                ["[63] Streamlit layouts ","streamlit_layouts","Streamlit UI component showcase: cards, timelines, metrics"],
                ["[64] Analyze reviews ","analyze_reviews_dummy","Customer satisfaction analyzer with NPS, heatmaps & trends"],
                ["[65] Lonen vs inflatie ","lonen_inflatie","Dutch wage and inflation analyzer using CBS statistics"],
                ["[66] Generate monopoply ","generate_monopoly","SVG Monopoly board editor with placeholder replacement & mapping"],
                ["[67] EasyMapNL ","easy_map_nl","Make a custom map of the Netherlands"],
                ["[68] 2e kamer ","tweedekamer","Verkiezings analyses 2e kamer"],
                ["[69] knmi vs openmeteo ","compare_knmi_openmeteo", "seizoensinfo knmi vs openmeteo"],
                ["[69] knmi vs openmeteo ","compare_knmi_openmeteo", "Seizoensinfo: KNMI vs Open-Meteo weather data comparison"],
                ["[70] Year timeline ","how_the_year_went", "Timeline of the year with subjective length of months"],
                ["[71] Airquality Chiang Mai","airquality_chiangmai", "Historical airquality in Chiang Mai"],
                ["[72] Letter Frequentie Analyse","letter_count", "Welk woord heeft de meeste van elke letter?"],
                ["[73] Streamlit dashboard apps","streamlit_demo_apps", "Demo templates made by Streamlit"],
                ["[74] Birthday heatmap","birthday_heatmap", "Heatmap geboortedatums+afwijkingen per gemeente"],
                ["[75] Bloomberg Dashboard","bloomberg_dashboard.bb_dashboard", "Heatmap geboortedatums+afwijkingen per gemeente"],

       ]


    try:
        choice = int(st.query_params["choice"]) 
    except:
        choice = 0                                             
    menuchoicelist = [options[n][0] for n, l in enumerate(options)]
    menuhelplist = [options[n][2] for n, l in enumerate(options)]

    if choice !=0: 
        expanded = False
    else:
        expanded = True
    with st.sidebar.expander('MENU: Choose a script | scroll down for options/parameters',  expanded=expanded):
        # try:
        if 1==1:
            menu_choice = st.radio(".",menuchoicelist, index=choice, captions=menuhelplist, label_visibility='hidden')

    st.sidebar.markdown("- - - - - - - - - - - - - - - - - - ", unsafe_allow_html=True)
    st.query_params.choice = menuchoicelist.index(menu_choice)
    
    for n, l in enumerate(options):
        if menu_choice == options[n][0]:
            if platform.processor() != "":
                #m = "C:\\Users\\rcxsm\\Documents\\python_scripts\\streamlit_scripts\\" + options[n][1].replace(" ","_") 
                m =  options[n][1].replace(" ","_") 
                # I was too lazy to change it in the list
                #st.write (f"{m }")
            else:
                m = options[n][1].replace(" ","_") # I was too lazy to change it in the list
            try:
                module = dynamic_import(m)
            except Exception as e:
                st.error(f"Module '{m}' not found or error in the script\n")
                st.warning(f"{e}")
                st.warning(traceback.format_exc())

                st.stop()
            try:
                module.main()
            except Exception as e:
                st.error(f"Function 'main()' in module '{m}' not found or error in the script")
                st.warning(f"{e}")
                st.warning(traceback.format_exc())
                st.stop()

if __name__ == "__main__":
    main()
