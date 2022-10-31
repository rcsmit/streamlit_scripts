
import pandas as pd

# id	STN	YYYYMMDD	temp_avg	temp_min	temp_max	T10N	zonneschijnduur	perc_max_zonneschijnduur	
# glob_straling	neerslag_duur	neerslag_etmaalsom	YYYY	MM	DD	dayofyear	count	month	year	
# day	month_year	month_day	date	value_kwh

import pandas as pd
import numpy as np

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg

from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import scipy

_lock = RendererAgg.lock

import numpy as np


import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, metrics

def get_data():
    file = "input\\zonnepanelen.csv"
    #file = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\data\zonnepanelen.csv"
    file = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/zonnepanelen.csv"
    #st.write(file)
    try:
        df = pd.read_csv(
            file,
            delimiter=";",
            
            low_memory=False,
        )
        df["YYYYMMDD"] = pd.to_datetime(df["YYYYMMDD"], format="%d/%m/%Y")
        df["YYYY"] = df["YYYYMMDD"].dt.year
        df["MM"] = df["YYYYMMDD"].dt.month
        df["DD"] = df["YYYYMMDD"].dt.day
        df["dayofyear"] = df["YYYYMMDD"].dt.dayofyear
        df["count"] = 1

        # to_divide_by_10 = [
        #     "temp_avg",
        #     "temp_min",
        #     "temp_max",
        #     "zonneschijnduur",
        #     "neerslag_duur",
        #     "neerslag_etmaalsom",
        # ]
        # for d in to_divide_by_10:
        #     try:
        #         df[d] = df[d] / 10
        #     except:
        #         df[d] = None
    except:
        st.error ("Error loading data")
        st.stop()
    return df

@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

def download_button(df):    
    csv = convert_df(df)

    st.sidebar.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='df_knmi.csv',
        mime='text/csv',
    )

def make_plot(df, x_axis, y_axis, regression):  
        title = (f"{y_axis} vs {x_axis}")
        fig = px.scatter(df, x=x_axis, y=y_axis, trendline="ols", title=title, hover_data=["date",x_axis, y_axis ])
        
        st.plotly_chart(fig, use_container_width=True)
        if regression:
            model = px.get_trendline_results(fig)
            alpha = model.iloc[0]["px_fit_results"].params[0]
            beta = model.iloc[0]["px_fit_results"].params[1]
            # st.write (f"Alfa {alpha} - beta {beta}")
            st.write (f"y =  {round(alpha,4)} *x + {round(beta,4)}")
            r2 = px.get_trendline_results(fig).px_fit_results.iloc[0].rsquared
            st.write(f"R2 = {r2}")
            try:
                c = round(df[x_axis].corr(df[y_axis]), 3)
                st.write(f"Correlatie {x_axis} vs {y_axis}= {c}")
            except:
                st.write("_")

def find_correlations(df):
    factors =  ["temp_avg","temp_min","temp_max","T10N","zonneschijnduur","perc_max_zonneschijnduur",
          "glob_straling","neerslag_duur","neerslag_etmaalsom"]
    result = "value_kwh"
    st.header("Correlaties")
    for f in factors:
        c = round(df[f].corr(df[result]), 3)
        st.write(f"Correlatie {f} vs {result} = {c}")
        
def regression(df):
    st.header("The Negative Binomial Regression Model")
    st.write("https://timeseriesreasoning.com/contents/negative-binomial-regression-model/")

    mask = np.random.rand(len(df)) < 0.8
    df_train = df[mask]
    df_test = df[~mask]
    print('Training data set length='+str(len(df_train)))
    print('Testing data set length='+str(len(df_test)))
    st.subheader("STEP 1: We will now configure and fit the Poisson regression model on the training data set.")
    expr = """value_kwh ~  temp_max + T10N + zonneschijnduur + perc_max_zonneschijnduur + glob_straling + neerslag_duur + neerslag_etmaalsom"""
    #Set up the X and y matrices for the training and testing data sets. patsy makes this really simple.

    y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
    y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')

    poisson_training_results = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    st.write (poisson_training_results.summary())
    st.subheader("STEP 2: We will now fit the auxiliary OLS regression model on the data set and use the fitted model to get the value of α.")
    df_train['BB_LAMBDA'] = poisson_training_results.mu
    df_train['AUX_OLS_DEP'] = df_train.apply(lambda x: ((x['value_kwh'] - x['BB_LAMBDA'])**2 - x['BB_LAMBDA']) / x['BB_LAMBDA'], axis=1)
    ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA - 1"""
    aux_olsr_results = smf.ols(ols_expr, df_train).fit()
    st.write("Print the regression params ( coefficient is the α):")
    alfa = aux_olsr_results.params
    st.write (alfa)
    st.write("The OLSResults object contains the t-score of the regression coefficient α. Let’s print it out:")
    st.write(aux_olsr_results.tvalues)

    q = 0.99
    degr_freedom = (len(df)-1)
    t_value = scipy.stats.t.ppf(q, degr_freedom)
    st.write (f"t-value = {t_value}")
    if t_value<alfa[0] :
        st.write(f"alfa {alfa[0]} is statistically significantly.")
    else:
        st.write(f"alfa {alfa[0]} is NOT statistically significantly.")

    
    st.subheader("STEP 3: We supply the value of alpha found in STEP 2 into the statsmodels.genmod.families.family.NegativeBinomial class, and train the NB2 model on the training data set.")
    nb2_training_results = sm.GLM(y_train, X_train,family=sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0])).fit()
    st.write("As before, we’ll print the training summary:")

    st.write(nb2_training_results.summary())  

    st.subheader("STEP 4: Let’s make some predictions using our trained NB2 model.")
    nb2_predictions = nb2_training_results.get_prediction(X_test)
    st.write ("Let’s print out the predictions:")

    predictions_summary_frame = nb2_predictions.summary_frame()
    st.write(predictions_summary_frame)
    predicted_counts=predictions_summary_frame['mean']
    actual_counts = y_test['value_kwh']

    fig1x = plt.figure()
    fig1x.suptitle('Predicted versus actual value_kwh')
    predicted, = plt.plot(X_test.index, predicted_counts, 'g', linewidth=1, label='Predicted value_kwh')
    actual, = plt.plot(X_test.index, actual_counts, 'r', linewidth=1, label='Actual value_kwh')
    plt.legend(handles=[predicted, actual])
    st.pyplot(fig1x)

def sklearn(df):
    #factors = ["temp_avg","temp_min","temp_max","T10N","zonneschijnduur","perc_max_zonneschijnduur",
    #        "glob_straling","neerslag_duur","neerslag_etmaalsom","value_kwh"]
    factors = ["temp_max","zonneschijnduur","glob_straling","neerslag_etmaalsom","value_kwh"]
    df = df[factors]
    st.header("Lineaire regressie met sklearn")
    st.write("https://www.geeksforgeeks.org/linear-regression-python-implementation/")

    # load the boston dataset
    # boston = datasets.load_boston(return_X_y=False)
    
    # defining feature matrix(X) and response vector(y)
    # X = boston.data
    # y = boston.target
    
    # splitting X and y into training and testing sets
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(df.drop(["value_kwh"], axis=1), df["value_kwh"], test_size=1 / 3)

    
    # create linear regression object
    reg = linear_model.LinearRegression()
    
    # train the model using the training sets
    reg.fit(X_train, y_train)
    
    # regression coefficients
    st.write ('Factors:', factors)
    st.write('Coefficients: ', reg.coef_)
    
    # variance score: 1 means perfect prediction
    st.write('Variance score: {}'.format(reg.score(X_test, y_test)))
    
    # plot for residual error
    
    ## setting plot style
    fig1x = plt.figure()
    plt.style.use('fivethirtyeight')
    
    ## plotting residual errors in training data
    plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
                color = "green", s = 10, label = 'Train data')
    
    ## plotting residual errors in test data
    plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
                color = "blue", s = 10, label = 'Test data')
    
    ## plotting line for zero residual error
    plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
    
    ## plotting legend
    plt.legend(loc = 'upper right')
    
    ## plot title
    plt.title("Residual errors")
    
    ## method call for showing the plot
    #plt.show()
    st.pyplot(fig1x)
def main():

    st.title("De relatie tussen Zonnepanelenopbrengst en meteorologische omstandigheden")
    df = get_data()

    #print (df)
    fields=["id","STN","YYYYMMDD","temp_avg","temp_min","temp_max","T10N","zonneschijnduur","perc_max_zonneschijnduur",
            "glob_straling","neerslag_duur","neerslag_etmaalsom","YYYY","MM","DD","dayofyear","count","month","year",
            "day","month_year","month_day","date","value_kwh"]
    
    
    x_axis = st.sidebar.selectbox("X-as scatter",fields, index = 3)
    y_axis = st.sidebar.selectbox("Y-as door de tijd/scatter",fields, index=23)
    st.subheader("Door de tijd")
    make_plot(df, "YYYYMMDD", y_axis, False)
    st.subheader("Scatter")
    make_plot(df, x_axis, y_axis, True)
    find_correlations(df)
    regression(df)

    sklearn(df)
    download_button(df)
    st.sidebar.write("KNMI data is van STN286, Nieuw Beerta")


    # https://www.weerstationhaaksbergen.nl/weather/index.php/Weblog/zonnestraling-en-zonnepanelen.html
main()
