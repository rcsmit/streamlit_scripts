import pandas as pd
import plotly.express as px
import streamlit as st
import math

def line_function(input_x, a,b,c,d,e,f,g,h):
    if input_x <a:
        return 100
    elif input_x>=a and input_x <= c:
        k, x, l, y = a,b,c,d
    elif input_x>c and input_x <= e:
        k, x, l, y = c,d,e,f
    elif input_x>e :
        k, x, l, y = e,f,g,h
    A = x
    B = math.log(x / y) / (  l-k)
    return A * math.exp(-B * (input_x - k))

def main():
    st.title("Simple Yield management tool")
    st.info("Enter below the 4 datapoints. (which occupation given a certain price")
    with st.expander("Values"):
        col1,col2 = st.columns(2)
        with col1:
            st.write("Price")
            a = st.number_input("a",0,1000, 200)
            c = st.number_input("b",0,1000, 300)
            e = st.number_input("c",0,1000, 400)
            g = st.number_input("d",0,1000, 500)
        with col2:
            st.write("Occupation")
            b = st.number_input("e (fixed)",100,100, 100)
            d = st.number_input("f",0,1000, 75)
            f = st.number_input("g",0,1000, 50)
            h = st.number_input("h",0,1000, 10)
        
    x_values = list(range(1001))
    df = pd.DataFrame({'price': x_values})
    df['occupation'] = df['price'].apply(lambda input_x: line_function(input_x,  a,b,c,d,e,f,g,h))
    df['turn_over'] = df['price'] * df['occupation']
    fig_y = px.line(df, x='price', y='occupation', title='occupation', labels={'x': 'price', 'y': 'occupation'})
    fig_y_times_x = px.line(df, x='price', y='turn_over', title='turn_over (price x occupation)', labels={'x': 'price', 'y': 'turn_over'})
    col3,col4 = st.columns(2)
    with col3:
        st.plotly_chart (fig_y, use_container_width=True)
    with col4:
        st.plotly_chart (fig_y_times_x, use_container_width=True)

if __name__ == "__main__":
    main()