# REPRODUCING https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8881926/

import pandas as pd
import streamlit as st
import numpy as np

import plotly.express as px

from scipy.stats import linregress

from sklearn import linear_model

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy import stats

def get_data():
    """Gets the data and merges the dataframes

    Args:
        join_how (str): "inner" or "outer"

    Returns:
        df: the complete dataframe
    """    
    #url_meat = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/meat_consumption/meat_consumption.csv"
    url_meat = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/meat_consumption/meat_consumption_simple.csv"
    #url_meat = r"C:\Users\rcxsm\Downloads\meat_consumption_simple.csv"
    df_meat =  pd.read_csv(url_meat, delimiter=',')
    return df_meat


def multiple_lineair_regression_statsmodels(x,y):
    """Calculates multiple lineair regression. User can choose the Y value and the X values
        https://www.blog.dailydoseofds.com/p/statsmodel-regression-summary-will
    Args:
        df_ (df): df with info
    """    
    st.subheader("Multiple Lineair Regression")
   
   
   
    model = sm.OLS(y, x).fit()
    #predictions = model.predict(x) 
    st.write("**OUTPUT ORDINARY LEAST SQUARES**")
    print_model = model.summary()
    st.write(print_model)
    

def manipulate_df(df, show_log_x, show_log_y, standard, y_value, x_values):
    df = df.dropna(subset=x_values)
    df = df.dropna(subset=y_value)
    df =df[["country"]+[y_value]+ x_values]
    if show_log_x:
        for c in x_values:
            df[c] = np.log(df[c])
       
    if show_log_y:
        df[y_value] = np.log(df[y_value])
    
    if standard:
        # https://stackoverflow.com/questions/50842397/how-to-get-standardised-beta-coefficients-for-multiple-linear-regression-using
        df = df.select_dtypes(include=[np.number]).dropna().apply(stats.zscore)
      
    st.write("**DATA**")
    st.write(df)
    st.write(f"Length : {len(df)}")
    x = df[x_values]
    y = df[y_value]
    return x,y

def interface():
    show_log_x = st.sidebar.checkbox('Show logarithmic values X', True)
    show_log_y= st.sidebar.checkbox('Show logarithmic values Y', True)
    standard=  st.sidebar.checkbox("Standardizing dataframe", True)
    intercept=  st.sidebar.checkbox("Intercept", False)
    y_value = st.selectbox("Y value", ['life_exp',"life_exp_birth","life_exp_5","mort_under_5"],1)
    x_values_options =  ["meat_cons","cal_day","gdpppp _2011","urban_pop","bmi_over_30","cho_crops"]#,"prim_educ_over_25"]
    x_values_default = ['meat_cons',"cal_day","gdpppp _2011","urban_pop","bmi_over_30","cho_crops"]#, "prim_educ_over_25"]
  
    x_values = st.multiselect("X values", x_values_options, x_values_default)
    return show_log_x,show_log_y,standard,intercept,y_value,x_values
    

def main():
    """Main function
    """    
    st.header("Meat consumption vs life expectancy")
    st.info("REPRODUCING https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8881926/")
 

    df = get_data()
    show_log_x, show_log_y, standard, intercept, y_value, x_values = interface()
   

    x, y = manipulate_df(df, show_log_x, show_log_y, standard, y_value, x_values)
  
    if intercept:
        x= sm.add_constant(x) # adding a constant
    

    multiple_lineair_regression_statsmodels(x,y)

if __name__ == "__main__":
    main()
    #prepare_data()