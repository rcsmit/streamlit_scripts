import streamlit as st

from app import *
from backtest import *
from uitleg_app import *
from uitleg_backtest import *
from bloomberg_template import *

tab1,tab2,tab3, tab4,tab5,tab6=st.tabs(["dashboard", "backtest","uitleg app", "uitleg backtest", "info","template"])

with tab3:
    uitleg_app()
    
with tab4:
    uitleg_backtest()


with tab6:
    bloomberg_template()

with tab1:
    dashboard()
    
with tab2:
    backtest()

with tab5:
    st.info("Vibecoded with ClaudeAI, prompt by @investingluc: https://x.com/investingluc/status/2034403688106615050")
