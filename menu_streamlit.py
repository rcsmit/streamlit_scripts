import streamlit as st
import importlib
import traceback
import os
import platform


st.set_page_config(page_title="Streamlit scripts of René Smit")


def dynamic_import(module):
    """Import a module stored in a variable

    Args:
        module (string): The module to import

    Returns:
        the module you want
    """
    return importlib.import_module(module)

def main():
    if platform.processor() != "":
        arr = os.listdir("C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\streamlit_scripts")
    else:
        arr = os.listdir()

    counter = 1
    options = [["0. welcome","welcome"],
                ["1. newagebullshitgenerator","newagebullshitgenerator"],
                ["2. KNMI grafieken","show_knmi"],
                ["3. Text generator","txt_generator_streamlit"],
                ["4. YT transcriber","YoutubeTranscriber_streamlit"],
                ["5. Schoonmaaktijden", "schoonmaaktijden"],
                ["6. Show sportactivities", "show_sportactivities"],
                ["7. YFinance info", "yfinance_info"],
                ["8. Crypto portfolio", "crypto_portfolio"],
                ["9. Yepcharts", "yepcharts"],
                ["10. Zonnepanelen", "zonnepanelen"],
                ["11. Occupation Camping", "read_bezetting_test"], #was occupation_camping
                ["12. Whatsapp analyzer", "whatsapp_analyzer"],
                ["13. Breezertaal converter", "make_breezer_taal"],
                ["14. Fit to data", "fit_to_data_streamlit"],
                ["15. Inkomstenbelasting", "inkomstenbelasting"],
                ["16. Exchange money", "exchange_money"],
                ["17. Get Forex data", "get_forex_data_st"]
                
                ]

    



    # for file in arr:
    #     if file[-2:] =="py" and ( file != "welcome.py" and file !="menu_streamlit.py"):
    #         menutext = f"{counter}. {file}"
    #         menutext = menutext.replace("_"," ") # I was too lazy to change it in the list
    #         menutext = menutext.replace(".py","") # I was too lazy to change it in the list
    #         file_ = file.replace(".py","") # I was too lazy to change it in the list

    #         options.append([menutext, file_])
    #         counter +=1

    query_params = st.experimental_get_query_params() # reading  the choice from the URL..

    choice = int(query_params["choice"][0]) if "choice" in query_params else 0 # .. and make it the default value

    menuchoicelist = [options[n][0] for n, l in enumerate(options)]

    with st.sidebar.expander('MENU: Choose a script | scroll down for options/parameters',  expanded=True):
        menu_choice = st.radio("",menuchoicelist, index=choice)

    st.sidebar.markdown("- - - - - - - - - - - - - - - - - - ", unsafe_allow_html=True)
    st.experimental_set_query_params(choice=menuchoicelist.index(menu_choice)) # setting the choice in the URL

    for n, l in enumerate(options):
        if menu_choice == options[n][0]:
            if platform.processor() != "":
                m = "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\streamlit_scripts\\" + options[n][1].replace(" ","_") # I was too lazy to change it in the list
                st.write (f"{m }")
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