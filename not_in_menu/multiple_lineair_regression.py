# REPRODUCING https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8881926/

import pandas as pd
import streamlit as st
import numpy as np

import plotly.express as px

from scipy.stats import linregress

from sklearn import linear_model

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

def get_data():
    """Gets the data and merges the dataframes

    Args:
        join_how (str): "inner" or "outer"

    Returns:
        df: the complete dataframe
    """    
    url_meat = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/meat_consumption.csv"
    url_meat = r"C:\Users\rcxsm\Downloads\meat_consumption_simple.csv"
    df_meat =  pd.read_csv(url_meat, delimiter=',')
    return df_meat


def multiple_lineair_regression_statsmodels(df_, x_values, y_value, show_log_x, show_log_y):
    """Calculates multiple lineair regression. User can choose the Y value and the X values
        https://www.blog.dailydoseofds.com/p/statsmodel-regression-summary-will
    Args:
        df_ (df): df with info
    """    
    st.subheader("Multiple Lineair Regression")
   
   
    df = df_.dropna(subset=x_values)
    df = df.dropna(subset=y_value)
    df =df[["country"]+[y_value]+ x_values]
    if show_log_x:
        for c in x_values:
            df[c] = np.log(df[c])
       
    if show_log_y:
        df[y_value] = np.log(df[y_value])
      
    st.write("**DATA**")
    st.write(df)
    st.write(f"Length : {len(df)}")
    x = df[x_values]
    y = df[y_value]
  
    #w = df["population"]
    
    # with statsmodels
    x = sm.add_constant(x) # adding a constant
    
    model = sm.OLS(y, x).fit()
    #predictions = model.predict(x) 
    st.write("**OUTPUT ORDINARY LEAST SQUARES**")
    print_model = model.summary()
    st.write(print_model)
    
    gamma_model = sm.GLM(y,x)#, family=sm.families.Gamma())

    gamma_results = gamma_model.fit()

    st.write(gamma_results.summary())

def multiple_lineair_regression_sklearn(df_,x_values, y_value,  show_log_x, show_log_y):
    """Calculates multiple lineair regression. User can choose the Y value and the X values

    Args:
        df_ (df): df with info
    """    
    st.subheader("Multiple Lineair Regression")
    
    
    df = df_.dropna(subset=x_values)
    df = df.dropna(subset=y_value)
    df =df[["country"]+[y_value]+ x_values]
    if show_log_x:
        for c in x_values:
            df[c] = np.log(df[c])
       
    if show_log_y:
        df[y_value] = np.log(df[y_value])
    from sklearn.model_selection import train_test_split
    x = df[x_values]
    y = df[y_value]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)
    #Fitting the Multiple Linear Regression model
    mlr = LinearRegression()  
    mlr.fit(x,y)
    #Intercept and Coefficient
    st.write("Intercept: ", mlr.intercept_)
    st.write("Coefficients:")
    st.write(list(zip(x, mlr.coef_)))

    #Model Evaluation
    from sklearn import metrics
    # meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
    # meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
    # rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
    st.write('R squared: {:.2f}'.format(mlr.score(x,y)*100))
    # st.write('Mean Absolute Error:', meanAbErr)
    # st.write('Mean Square Error:', meanSqErr)
    # st.write('Root Mean Square Error:', rootMeanSqErr)

def main():
    """Main function
    """    
    st.header("Meat consumption vs life expectancy")
    st.info("REPRODUCING https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8881926/")
 

    df = get_data()
    show_log_x = st.sidebar.checkbox('Show logarithmic values X')
    show_log_y= st.sidebar.checkbox('Show logarithmic values Y')
 
    y_value = st.selectbox("Y value", ['life_exp',"life_exp_birth","life_exp_5","mort_under_5"],1)
    x_values_options =  ["meat_cons","cal_day","gdpppp _2011","urban_pop","bmi_over_30","cho_crops"]#,"prim_educ_over_25"]
    x_values_default = ['meat_cons',"cal_day","gdpppp _2011","urban_pop","bmi_over_30","cho_crops"]#, "prim_educ_over_25"]
  
    x_values = st.multiselect("X values", x_values_options, x_values_default)
    
    multiple_lineair_regression_statsmodels(df, x_values, y_value, show_log_x, show_log_y)
    multiple_lineair_regression_sklearn(df,x_values, y_value,  show_log_x, show_log_y)

if __name__ == "__main__":
    main()
    #prepare_data()