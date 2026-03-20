import streamlit as st

from app import *
from backtest import *

tab1,tab2,tab3=st.tabs(["dashboard", "backtest","info"])

with tab1:
    dashboard()
    
with tab2:
    backtest()
with tab3:
    st.info("Vibecoded with ClaudeAI, prompt by @investingluc: https://x.com/investingluc/status/2034403688106615050")