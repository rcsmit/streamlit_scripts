import streamlit as st

from sterfte_temperatuur_orwell import main_orwell
from sterfte_temperatuur_orwell_esp2013 import main_orwell_esp2013
from sterfte_temperatuur_rcsmit import main_rcsmit


def main():

        
    # if 'active_tab' not in st.session_state:
    #     st.session_state.active_tab = "Tab 1"

    # with st.sidebar:
    #     if st.session_state.active_tab == "Tab 1":
    #         st.write("Sidebar content for Tab 1")
    #         # Add any widgets or elements specific to Tab 1
    #     elif st.session_state.active_tab == "Tab 2":
    #         st.write("Sidebar content for Tab 2")
    #         # Add any widgets or elements specific to Tab 2
    #     else:
    #         st.write("Default sidebar content")

    # tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])

    # with tab1:
    #     st.session_state.active_tab = "Tab 1"
    #     st.write("Content of Tab 1")

    # with tab2:
    #     st.session_state.active_tab = "Tab 2"
    #     st.write("Content of Tab 2")
    tab1, tab2, tab3 = st.tabs(["rcsmit", "orwell", "orwell_esp2013"])

    with tab1:
        main_rcsmit()
        # st.info("rcsmit")
    with tab2:
        main_orwell()
        #st.info("orwell")
    with tab3:
        main_orwell_esp2013()
if __name__ == "__main__":
    main()
