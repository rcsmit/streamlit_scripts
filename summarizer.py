from keys import * # secret file with the prices
import openai


openai.api_key = OPENAI_API_KEY

import streamlit as st

st.title("Text Summarizer")

input_text = st.text_area(label='Enter full text:', value="", height=250)

st.button("submit")

output_text = st.text_area(label='Summarized text:', value='', height=250)