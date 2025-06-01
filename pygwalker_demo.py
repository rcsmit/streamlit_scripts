
from pygwalker.api.streamlit import StreamlitRenderer
import pandas as pd
import streamlit as st
 
def main():
    # Adjust the width of the Streamlit page
    try:
        st.set_page_config(
            page_title="Use Pygwalker In Streamlit",
            layout="wide"
        )
    except:
        pass
    # Import your data
    df = pd.read_csv(r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\gapminder_data_graphs.csv")
    
    pyg_app = StreamlitRenderer(df)
    
    pyg_app.explorer()

if __name__ == "__main__":
    main()
    # Run the Streamlit app