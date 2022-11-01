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
from plotly.subplots import make_subplots
import plotly.graph_objects as go
_lock = RendererAgg.lock

import numpy as np


import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, metrics

import numpy as np

def daylength_brock(dayOfYear, lat):
    """Computes the length of the day (the time between sunrise and
    sunset) given the day of the year and latitude of the location.
    Function uses the Brock model for the computations.
    For more information see, for example,
    Forsythe et al., "A model comparison for daylength as a
    function of latitude and day of year", Ecological Modelling,
    1995.
    https://gist.github.com/anttilipp/ed3ab35258c7636d87de6499475301ce
    https://sci-hub.se/https://doi.org/10.1016/0304-3800(94)00034-F

    Parameters
    ----------
    dayOfYear : int
        The day of the year. 1 corresponds to 1st of January
        and 365 to 31st December (on a non-leap year).
    lat : float
        Latitude of the location in degrees. Positive values
        for north and negative for south.
    Returns
    -------
    d : float
        Daylength in hours.
    """
    latInRad = np.deg2rad(lat)
    declinationOfEarth = 23.45*np.sin(np.deg2rad(360.0*(283.0+dayOfYear)/365.0))
    if -np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth)) <= -1.0:
        return 24.0
    elif -np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth)) >= 1.0:
        return 0.0
    else:
        hourAngle = np.rad2deg(np.arccos(-np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth))))
        return 2.0*hourAngle/15.0

    import math

def daylength_CBM(day_of_year, latitude):
    # https://www.dataliftoff.com/plotting-hours-of-daylight-in-python-with-matplotlib/
    # formula per Ecological Modeling, volume 80 (1995) pp. 87-95, called "A Model Comparison for Daylength as a Function of Latitude and Day of the Year."
    # see more details - http://mathforum.org/library/drmath/view/56478.html
    # Latitude in degrees, postive for northern hemisphere, negative for southern
    # Day 1 = Jan 1
    import math
    P = math.asin(0.39795 * math.cos(0.2163108 + 2 * math.atan(0.9671396 * math.tan(.00860 * (day_of_year - 186)))))
    pi = math.pi
    day_light_hours = 24 - (24 / pi) * math.acos((math.sin(0.8333 * pi / 180) + math.sin(latitude * pi / 180) * math.sin(P)) / (math.cos(latitude * pi / 180) * math.cos(P)))
    return  day_light_hours

def calculate_zonne_energie(temp_avg, temp_max, glob_straling, windsnelheid_avg,dayOfYear):
    # https://twitter.com/karin_vdwiel/status/1516393097101512712
    # https://www.knmi.nl/over-het-knmi/nieuws/van-weersverwachting-naar-energieverwachting
    # https://www.sciencedirect.com/science/article/pii/S1364032119302862?via%3Dihub
    # https://www.nrel.gov/docs/fy03osti/35645.pdf
    lat = 52.9268737
    daglengte = daylength_CBM(dayOfYear, lat)
    gamma = -0.005
    Tref = 25
    c1 = 4.3
    c2 = 0.943
    c3 = 0.028
    c4 = -1.528

    T_a_day_t = (temp_avg + temp_max) / 2
    Gt= glob_straling   #/10000# (van cm2 naar m2)
    Gstc = 1000
    Vt = windsnelheid_avg
    Tcell_t = c1 + c2* T_a_day_t + c3*Gt + c4 *Vt
    Pr_t = 1 + (gamma*(Tcell_t-Tref))
    PVpot_t = Pr_t*(Gt/Gstc) * daglengte
  
    return PVpot_t
@st.cache
def get_data():

    # file = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\knmi_nw_beerta.csv"
    file = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/knmi_nw_beerta.csv"
    df_nw_beerta=  pd.read_csv(
            file,
            delimiter=",",
            
            low_memory=False,
        )

   
    df_nw_beerta["YYYYMMDD"] = pd.to_datetime(df_nw_beerta["YYYYMMDD"], format="%Y%m%d")
    df_nw_beerta["windsnelheid_avg"] = df_nw_beerta["FG"] /10 # (in 0.1 m/s) / Daily mean windspeed (in 0.1 m/s)
    df_nw_beerta["temp_avg"] =df_nw_beerta["TG"]/10 # Etmaalgemiddelde temperatuur (in 0.1 graden Celsius) / Daily mean temperature in (0.1 degrees Celsius)
    df_nw_beerta["temp_min"] =df_nw_beerta["TN"]/10#Minimum temperatuur (in 0.1 graden Celsius) / Minimum temperature (in 0.1 degrees Celsius)
    df_nw_beerta["temp_max"]  =df_nw_beerta["TX"]/10  #    = Maximum temperatuur (in 0.1 graden Celsius) / Maximum temperature (in 0.1 degrees Celsius)
    df_nw_beerta["zonneschijnduur"] = df_nw_beerta["SQ"]/10#        = Zonneschijnduur (in 0.1 uur) berekend uit de globale straling (-1 voor <0.05 uur) / Sunshine duration (in 0.1 hour) calculated from global radiation (-1 for <0.05 hour)
    df_nw_beerta["perc_max_zonneschijnduur"] =df_nw_beerta["SP"]#       = Percentage van de langst mogelijke zonneschijnduur / Percentage of maximum potential sunshine duration
    df_nw_beerta["glob_straling"] =df_nw_beerta["Q"]  #      = Globale straling (in J/cm2) / Global radiation (in J/cm2)
    df_nw_beerta["neerslag_duur"] =df_nw_beerta["DR"]/10#      = Duur van de neerslag (in 0.1 uur) / Precipitation duration (in 0.1 hour)
    df_nw_beerta["neerslag_etmaalsom"] =df_nw_beerta["RH"]/10#       = Etmaalsom van de neerslag (in 0.1 mm) (-1 voor <0.05 mm) / Daily precipitation amount (in 0.1 mm) (-1 for <0.05 mm)
    df_nw_beerta["dayofyear"] =  df_nw_beerta["YYYYMMDD"].dt.dayofyear
    lat = 52.9268737
    df_nw_beerta["daglengte"]  = df_nw_beerta.apply(lambda x: daylength_CBM(x["dayofyear"], lat), axis=1)
    df_nw_beerta['zonne_energie_theoretisch'] = df_nw_beerta.apply(lambda x: calculate_zonne_energie(x["temp_avg"],     x["temp_max"], x["glob_straling"] , x["windsnelheid_avg"], x["dayofyear"]), axis=1)
    file = "input\\zonnepanelen.csv"
    #st.write (df_nw_beerta)
   
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
        df["value_kwh_gemeten"] = df["value_kwh"]
        df["year"] = df["YYYY"].astype(str)
        df["month"] = df["month"].astype(str)
        df["day"] = df["DD"].astype(str)
        df["month_year"] = df["month"] + " - " + df["year"]
        #df["year_month"] = df["year"] + " - " +  df["MM"].astype(str).str.zfill(2)
        df["year_month"] =   df["MM"].astype(str).str.zfill(2)  + "/" +  df["year"]
        df["year_month"] = pd.to_datetime(df["year_month"], format="%m/%Y")
        df= df[["YYYYMMDD", "YYYY","MM","DD","dayofyear","count","month","year",
            "day","month_year","month_day","year_month","date","value_kwh_gemeten"]]
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

    df_ = pd.merge(df_nw_beerta, df, how="inner", on = "YYYYMMDD")
    st.write(df_)
    print (df_)
    return df_

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

def make_plot(df, x_axis, y_axis, regression,y_axis2, datefield,how):
        
        if y_axis2 == None:
            title = (f"{y_axis} vs {x_axis}")
            if how=="line":
                fig = px.line(df, x=x_axis, y=y_axis, title=title,) #  hover_data=[datefield,x_axis, y_axis ]
            else:
                fig = px.scatter(df, x=x_axis, y=y_axis, trendline="ols", title=title,) #  hover_data=[datefield,x_axis, y_axis ]
            fig.layout.xaxis.title=x_axis
            fig.layout.yaxis.title=y_axis
            st.plotly_chart(fig, use_container_width=True)
        else:

            # https://stackoverflow.com/questions/62853539/plotly-how-to-plot-on-secondary-y-axis-with-plotly-express
            title = (f"{y_axis} & {y_axis2} vs {x_axis}")

                        
            fig = go.Figure()
            if how =="line":
                fig.add_trace(go.Scatter(
                    x=df[x_axis], y=df[y_axis],
                    name=y_axis,
                    mode='lines',
                    marker_color='rgba(152, 0, 0, .8)'
                ))

                fig.add_trace(go.Scatter(
                    x=df[x_axis], y=df[y_axis2],
                    name=y_axis2,
                    mode='lines',
                    marker_color='rgba(255, 182, 193, .9)'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=df[x_axis], y=df[y_axis],
                    name=y_axis,
                    mode='markers',
                    marker_color='rgba(152, 0, 0, .8)'
                ))

                fig.add_trace(go.Scatter(
                    x=df[x_axis], y=df[y_axis2],
                    name=y_axis2,
                    mode='markers',
                    marker_color='rgba(255, 182, 193, .9)'
                ))


            # subfig = make_subplots(specs=[[{"secondary_y": True}]])  
            # fig = px.scatter(df, x=x_axis, y=y_axis, trendline="ols", title=title,  marker_color='rgba(152, 0, 0, .8)', hover_data=["date",x_axis, y_axis ])
            
            # fig2 = px.scatter(df, x=x_axis, y=y_axis2, trendline="ols", title=title,  hover_data=["date",x_axis, y_axis2 ])

            # fig2.update_traces(yaxis="y2")

            # subfig.add_traces(fig.data + fig2.data)
            # subfig.layout.xaxis.title=x_axis
            # subfig.layout.yaxis.title=y_axis
           
            # subfig.layout.yaxis2.title=y_axis2

            # subfig.for_each_trace(lambda t: t.update(marker=dict(color=t.marker.color)))
            st.plotly_chart(fig, use_container_width=True)
            #subfig.show()
            
        if regression and y_axis2 == None:
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
    result = "value_kwh_gemeten"
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
    expr = """value_kwh_gemeten ~  temp_max + T10N + zonneschijnduur + perc_max_zonneschijnduur + glob_straling + neerslag_duur + neerslag_etmaalsom"""
    #Set up the X and y matrices for the training and testing data sets. patsy makes this really simple.

    y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
    y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')

    poisson_training_results = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    st.write (poisson_training_results.summary())
    st.subheader("STEP 2: We will now fit the auxiliary OLS regression model on the data set and use the fitted model to get the value of α.")
    df_train['BB_LAMBDA'] = poisson_training_results.mu
    df_train['AUX_OLS_DEP'] = df_train.apply(lambda x: ((x['value_kwh_gemeten'] - x['BB_LAMBDA'])**2 - x['BB_LAMBDA']) / x['BB_LAMBDA'], axis=1)
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
    actual_counts = y_test['value_kwh_gemeten']

    fig1x = plt.figure()
    fig1x.suptitle('Predicted versus actual value_kwh_gemeten')
    predicted, = plt.plot(X_test.index, predicted_counts, 'g', linewidth=1, label='Predicted value_kwh_gemeten')
    actual, = plt.plot(X_test.index, actual_counts, 'r', linewidth=1, label='Actual value_kwh_gemeten')
    plt.legend(handles=[predicted, actual])
    st.pyplot(fig1x)

def sklearn(df):
    #factors = ["temp_avg","temp_min","temp_max","T10N","zonneschijnduur","perc_max_zonneschijnduur",
    #        "glob_straling","neerslag_duur","neerslag_etmaalsom","value_kwh"]
    factors = ["temp_max","zonneschijnduur","glob_straling","neerslag_etmaalsom","value_kwh_gemeten"]
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
    X_train, X_test, y_train, y_test = train_test_split(df.drop(["value_kwh_gemeten"], axis=1), df["value_kwh_gemeten"], test_size=1 / 3)

    
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
    groupby_ = st.sidebar.selectbox("Groupby", [True, False], index=1)
    if groupby_:
        groupby_how = st.sidebar.selectbox("Groupby", ["year", "year_month"], index=1)
        groupby_what = st.sidebar.selectbox("Groupby",["sum", "mean"], index=1)
        if groupby_what == "sum":
            df = df.groupby([df[groupby_how]], sort = True).sum().reset_index()
        elif groupby_what == "mean":
            df = df.groupby([df[groupby_how]], sort = True).mean().reset_index()
        datefield = groupby_how
    else:
        datefield = "YYYYMMDD"
    st.write(df)
    
    #print (df)
    fields=[None,"id","STN",datefield,"temp_avg","temp_min","temp_max","T10N","zonneschijnduur","perc_max_zonneschijnduur",
            "glob_straling","zonne_energie_theoretisch", "neerslag_duur","neerslag_etmaalsom","YYYY","MM","DD","dayofyear","count","month","year",
            "day","month_year","month_day","date","value_kwh_gemeten", "daglengte"]
    
    
    x_axis = st.sidebar.selectbox("X-as scatter",fields, index = 4)
    y_axis = st.sidebar.selectbox("Y-as door de tijd/scatter",fields, index=25)
    y_axis2 = st.sidebar.selectbox("Sec. Y-as  door de tijd/scatter",fields, index=0)
    st.subheader("Door de tijd")
    if groupby_:
        make_plot(df, datefield, y_axis, False,  y_axis2, datefield, "line")
    else:
        make_plot(df, datefield, y_axis, False,  y_axis2, datefield, "scatter")

    st.subheader("Scatter")
    make_plot(df, x_axis, y_axis, True,  y_axis2, datefield, "scatter")
    find_correlations(df)
    regression(df)

    sklearn(df)
    download_button(df)
    st.sidebar.write("KNMI data is van STN286, Nieuw Beerta")
    st.sidebar.write("CODE: https://github.com/rcsmit/streamlit_scripts/blob/main/zonnepanelen.py")

    # https://www.weerstationhaaksbergen.nl/weather/index.php/Weblog/zonnestraling-en-zonnepanelen.html
main()
