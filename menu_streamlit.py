import streamlit as st
import importlib
import traceback
import os

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
    st.title ("Streamlit scripts of René Smit")


    arr = os.listdir()
    #arr = os.listdir("C:\\Users\\rcxsm\\Documents\\phyton_scripts\\streamlit_scripts")

    counter = 1
    options = [["0. welcome","welcome"]]
    for file in arr:
        if file[-2:] =="py" and ( file != "welcome.py" and file !="menu_streamlit.py"):
            menutext = f"{counter}. {file}"
            menutext = menutext.replace("_"," ") # I was too lazy to change it in the list
            menutext = menutext.replace(".py","") # I was too lazy to change it in the list

            options.append([menutext, file])
            counter +=1
    query_params = st.experimental_get_query_params() # reading  the choice from the URL..

    choice = int(query_params["choice"][0]) if "choice" in query_params else 0 # .. and make it the default value

    menuchoicelist = [options[n][0] for n, l in enumerate(options)]

    with st.sidebar.beta_expander('MENU: Choose a script | scroll down for options/parameters',  expanded=True):
        menu_choice = st.radio("",menuchoicelist, index=choice)

    st.sidebar.markdown("<h1>- - - - - - - - - - - - - - - - - - </h1>", unsafe_allow_html=True)
    st.experimental_set_query_params(choice=menuchoicelist.index(menu_choice)) # setting the choice in the URL

    for n, l in enumerate(options):
        if menu_choice == options[n][0]:
            m = options[n][1].replace(" ","_") # I was too lazy to change it in the list
            try:
                module = dynamic_import(m)
            except Exception as e:
                st.error(f"Module '{m}' not found or error in the script\n")
                st.warning(f"{e}")
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