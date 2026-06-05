import streamlit as st
try:
    from wbgt_utils import show_info, info, referentie_tabel, feels_like_calculator, solar_wrapper, main_
    from select_time_place import select_time_place
    from replicate_knmi_wbgt import show_historical_data
    from wbgt_vergelijk_script_met_knmi import vergelijk_script_met_knmi_download

except:
    from show_knmi_functions.wbgt_utils import show_info, info, referentie_tabel, feels_like_calculator, solar_wrapper, main_
    from show_knmi_functions.select_time_place import select_time_place
    from show_knmi_functions.replicate_knmi_wbgt import show_historical_data
    from show_knmi_functions.wbgt_vergelijk_script_met_knmi import vergelijk_script_met_knmi_download
def wbgt_knmi():
    with st.sidebar:
        lat,lon,utc_dt, loc_name, selected_date, selected_time,tz,LOCATIONS = select_time_place()

    tab1,tab2, tab3,tab4,tab5,tab6,tab7=st.tabs(["Main", "Tabel", "Calculator", "Solarinfo","1991-2025","script vs knmi","INFO"])

    with tab7:
        show_info()
        info()

    with tab2:
        referentie_tabel(lat,lon,utc_dt)
    with tab3:
        feels_like_calculator(lat,lon,utc_dt)
    with tab4:
        solar_wrapper(lat,lon,utc_dt, loc_name, selected_date, selected_time,tz,LOCATIONS)
    with tab1:
        main_()
    
    with tab5:
        show_historical_data()
    with tab6:
        vergelijk_script_met_knmi_download()    
def main():
    wbgt_knmi()
if __name__=="__main__":
    main()