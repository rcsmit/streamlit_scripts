import streamlit as st

import numpy as np
import pandas as pd
import statsmodels.api as sm

# import scipy.stats as stats
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.stats import sem


import plotly.graph_objects as go
import plotly.express as px
from scipy.linalg import qr, pinv   
from scipy.linalg import solve_triangular

import time
from math import ceil
import numpy as np
from scipy import linalg
from skmisc.loess import loess

def show_info():
    """Show an introduction text for the page
    """    
    st.header("Yearly temperature data")
    st.subheader("Trendline with local lineair regression - LOESS")

    st.info(
        """
    The trendline can be regarded as an approximation of a 30-year average, which has a smooth appearance
    and is extended toward the beginning and end of the time-series.

    It is based on linear local regression, computed using the statsmodels library. It uses a bicubic weight
    function over a 42-year window. In the central part of the time-series, the variance of the trendline
    estimate is approximately equal to the variance of a 30-year average.

    To test the proposition of no long-term change between the years t1 and t2, these years need to be supplied.
    The result is the p-value: the probability (under the proposition) that the estimated trendline values in
    t2 and t1 differ more than observed.

    Version: 09-Mar-2021

    References:
    https://www.knmi.nl/kennis-en-datacentrum/achtergrond/standaardmethode-voor-berekening-van-een-trend
    KNMI Technical report TR-389 (see http://bibliotheek.knmi.nl/knmipubTR/TR389.pdf)

    Author: Cees de Valk (cees.de.valk@knmi.nl)

    Original Source code:
    https://gitlab.com/cees.de.valk/trend_knmi/-/blob/master/R/climatrend.R?ref_type=heads
    translated from R to Python by ChatGPT and adapted by Rene Smit
    not tested 100%, p-value for trend doesnt work
    """
    )



def drawplot_matplotlib(t, y, draw30, avt, avy, trend, trendub, trendlb):
    """Draws the plot with matplotlib, like in the original R script


    Args:
        t (_type_): _description_
        y (_type_): _description_
        draw30 (_type_): _description_
        avt (_type_): _description_
        avy (_type_): _description_
        trend (_type_): _description_
        trendub (_type_): _description_
        trendlb (_type_): _description_

    """
    fig = plt.figure(figsize=(8, 6))
    ylim = [np.min([np.min(y), np.min(trendlb)]), np.max([np.max(y), np.max(trendub)])]
    ylim[1] = ylim[0] + (ylim[1] - ylim[0]) * 1.0
    plt.plot(t, y, "b-", label="Temperature Data")
    plt.plot(t, trend, "r-", lw=2, label="Trendline")
    plt.fill_between(
        t, trendlb, trendub, color="grey", alpha=0.5, label="Confidence Interval"
    )

    if draw30:
        plt.plot(avt, avy, "ko", markersize=3, label="30-yr Average")

    plt.xlabel("Year")
    plt.ylabel("Temperature")
    plt.grid()
    plt.legend()
    plt.show()
    st.pyplot(fig)


def show_plot_plotly(title, what_to_show, t, values_, trend, trendlb, trendub, avt=None, avy=None,  draw30=False):
    """_summary_

    Args:
        df (_type_): _description_
        what_to_show (_type_): _description_
        t (_type_): _description_
        trend (_type_): _description_
        trendlb (_type_): _description_
        trendub (_type_): _description_
        avt (_type_, optional): _description_. Defaults to None.
        avy (_type_, optional): _description_. Defaults to None.
        draw30 (bool, optional): _description_. Defaults to False.
        draw_ci (bool, optional): _description_. Defaults to True.
    """
    if draw30:
        av = go.Scatter(
            name=f"{what_to_show} avg 30 jaar",
            x=avt,
            y=avy,
            # mode='lines',
            line=dict(width=1, color="rgba(0, 0, 0, 1)"),
        )
    loess = go.Scatter(
        name=f"{what_to_show} Loess",
        x=t,
        y=trend,
        mode="lines",
        line=dict(width=1, color="rgba(255, 0, 255, 1)"),
    )
    
    loess_low = go.Scatter(
        name=f"{what_to_show} Loess low",
        x=t,
        y=trendlb,
        mode="lines",
        line=dict(width=0.7, color="rgba(255, 0, 255, 0.5)"),
    )
    loess_high = go.Scatter(
        name=f"{what_to_show} Loess high",
        x=t,
        y=trendub,
        mode="lines",
        line=dict(width=0.7, color="rgba(255, 0, 255, 0.5)"),
    )
    # Create a filled area plot for confidence interval
    confidence_trace = go.Scatter(x=np.concatenate([t, t[::-1]]),
                            y=np.concatenate([trendub, trendlb[::-1]]),
                                fill='tozeroy',
                                fillcolor='rgba(0, 128, 0, 0.2)',
                                line=dict(color='dimgrey', width=.5),
                                showlegend=True,
                                name="CI of the trendline")

    values = go.Scatter(
        name=what_to_show,
        x=t,
        y=values_,
        mode="lines",
        line=dict(width=1, color="rgba(0, 0, 255, 0.6)"),
    )

    data = [values, loess]
   
    data.append(confidence_trace)
    if draw30:
        data.append(av)
    values_np = np.array(values_)
    A_1d = np.ravel(values_)
    B_1d = np.ravel(trendub)
    C_1d = np.ravel(trendlb)
    #Y_values =  values_ + trendub.to_list() + trendlb.to_list() # trendlb #
    try:
        Y_values = A_1d.tolist() + B_1d.tolist() + C_1d.tolist()
        y_lower_bound = min(Y_values)    
        y_upper_bound = max(Y_values)
    except:
        y_lower_bound = values_.min()    
        y_upper_bound = values_.max()
    
    
    
    layout = go.Layout(
        yaxis=dict(title=what_to_show, range=[y_lower_bound, y_upper_bound ]), title=f"Year averages of {what_to_show} - {title}"
    )
 

    fig = go.Figure(data=data, layout=layout)

    fig.update_layout(xaxis=dict(tickformat="%d-%m-%Y"))
    st.plotly_chart(fig, use_container_width=True)
    
    
def getdata(url):
    try:
        df = pd.read_csv(
            url,
            delimiter=",",
            header=0,
            comment="#",
            low_memory=False,
        )

    except:
        st.error("FOUT BIJ HET INLADEN.")
        st.stop()
    return df





def main_output_R_script(draw30):
    df3 = getdata("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/trend_de_bilt_jaargem_1901_2022.csv" )
    output = False
    if output:
        st.subheader("Results KNMI script in R")
        st.write("These are the values in the output of the script in R, and seen as 'standard")
        title = "Output R script KNMI (golden standard)"
        show_plot_plotly(title, "temp_avg", df3["YYYY"],df3["temp_avg"],df3["knmi_R_script_loess"],
                                    df3["knmi_R_script_low"],df3["knmi_R_script_high"],  df3["YYYY"],
                                    df3["30_yr_average"],  draw30, True)
    return df3

def main_skmisc(t, y, t1,t2, draw30, p=None, ybounds=None, it=1):
    """Make a plot and calculate p-value with scikit-misc

    Args:
        t : list of Years, increasing by 1.
        y : list of Annual values
        t1 : float, optional
            First year for which trendline value is compared in the test.
        t2 : float, optional
             year (see t1) for which trendline value is compared in the test. Must be >30 higher than t1
        it : number of iterations
    Returns:
        _type_: _description_

    Relevant code in R
        # fixed parameters
        width <- 42
        control <- loess.control(surface = "direct", statistics= "exact",
                           iterations= 1)
        # linear LOESS trendline computation
        span <- width/ng
        mdl <- loess(y ~ t, data= data.frame(t= tg, y=yg), span= span,
                    degree= 1, control= control)
        # mdl <- loess(y ~ t, data= data.frame(t= t, y=y), span= span, degree= 1)
        pre <- predict(mdl, newdata= data.frame(t= t), se= TRUE)
        trend <- pre$fit          # trendline
        trendsd <- pre$se.fit     # standard deviation of trendline
    """

    # https://has2k1.github.io/scikit-misc/stable/generated/skmisc.loess.loess.html
    # https://stackoverflow.com/questions/31104565/confidence-interval-for-lowess-in-python

    st.subheader("Lowess with SciKit-Misc")
  


    # Set default values for p, t1, and t2
    if p is None:
        p = 0.95  # default confidence level
    if t1 is None or t2 is None:
        t1 = np.inf
        t2 = -np.inf

    # Set default value for ybounds
    if ybounds is None:
        ybounds = [-np.inf, np.inf]
    elif len(ybounds) != 2:
        ybounds = [-np.inf, np.inf]

    ybounds = sorted(ybounds)

    # Dimensions and checks
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dt = np.diff(t)[0]
    n = len(y)
    ig = ~np.isnan(y)
    yg = y[ig]
    tg = t[ig]
    ng = sum(ig)

    if ng <= 29:
        raise ValueError("Insufficient valid data (less than 30 observations).")

    # Check values of bounds
    if np.any(yg < ybounds[0]) or np.any(yg > ybounds[1]):
        raise ValueError("Stated bounds are not correct: y takes values beyond bounds.")

    # Averages over 30 time-steps
    avt, avy, avysd = None, None, None
    if ng > 29:
        avt = tg + dt / 2  # time (end of time-step, for 30-year averages)
        avy = np.convolve(yg, np.ones(30) / 30, mode="valid")
        avy2 = np.convolve(yg**2, np.ones(30) / 30, mode="valid")
        avysd = np.sqrt(avy2 - avy**2)
        ind = slice(
            15, ng - 14
        )  # was (15, ng-15) but gives an error, whether the df has an even or uneven length
        # [ValueError: x and y must have same first dimension, but have shapes (92,) and (93,)]
        avt = avt[ind]
        # avy = avy[ind]            # takes away y values, gives error
        # [ValueError: x and y must have same first dimension, but have shapes (93,) and (78,)]
        avysd = avysd[ind]


    span = 42/len(y)
    l = loess(t,y)
    
    
    # MODEL and CONTROL. Essential for replicating the results from the R script.
    #
    # https://has2k1.github.io/scikit-misc/stable/generated/skmisc.loess.loess_model.html#skmisc.loess.loess_model
    # https://has2k1.github.io/scikit-misc/stable/generated/skmisc.loess.loess_control.html#skmisc.loess.loess_control
   
    l.model.span = span
    l.model.degree = 1
    l.control.iterations = it # must be 1 for replicating the R-script
    l.control.surface = "direct"
    l.control.statistics = "exact"

    l.fit()
    pred = l.predict(t, stderror=True)
    conf = pred.confidence()


    ste = pred.stderr
    lowess = pred.values
    ll = conf.lower
    ul = conf.upper
    
    title,what_to_show ="scikit-misc", "temp_avg"
    show_plot_plotly(title, what_to_show, t, y, lowess, ll, ul, avt, avy,  draw30=draw30)
   
    if t2 in t and t1 in t and t2 >= t1 + 30:
        idx_t1 = np.where(t == t1)[0][0]
        idx_t2 = np.where(t == t2)[0][0]
        y1 = lowess[idx_t1]#[0]
        y2 = lowess[idx_t2]#[0]
        y1sd = ste[idx_t1]#[0]
        y2sd = ste[idx_t2]#[0]
        # Two-sided test for absence of trend
        pvalue = (1 - norm.cdf(abs(y2 - y1), scale=np.sqrt(y1sd**2 + y2sd**2))) * 2


        if pvalue != None:
            st.write(f"pvalue: {round(pvalue,4)}")
            if pvalue <0.05:
                st.info(f"The data indicates a long time change between {t1} and {t2}.")
            else:
                st.info(f"The data does not indicate (or a little) a long time change between {t1} and {t2}.")

    data = {'YYYY': t, 'skmisc_loess': lowess, 'skmisc_low': ll, 'skmisc_high':ul}

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)
    return df

def show_footer():
    st.write("A good introduction about LOWESS/LOESS can be found here :https://aitechtrend.com/smoothing-out-the-noise-analyzing-data-with-lowess-regression-in-python/")

    st.write("The difference between the two acronyms or names is mostly superficial, but there is an actual difference in R–there are two different functions, lowess() and loess(). Lowess was implemented first, while the latter (loess) is more flexible and powerful. The loess() function creates an object that contains the results, and the predict() function retrieves the fitted values.[1]")
    st.write("[1] https://www.ime.unicamp.br/~dias/loess.pdf")
    st.info("Source: https://github.com/rcsmit/streamlit_scripts/blob/main/loess.py")
        

def interface():
    
    what_to_show = "temp_avg" 
   
    drawplot = st.sidebar.selectbox("Show Matplotlib plot", [True, False], 1)
    draw30 = st.sidebar.selectbox("Show 30 year SMA", [True, False], 1)
    test_trend= st.sidebar.selectbox("Two-sided test for absence of trend", [True, False], 1)

    if test_trend:
        t1 = st.sidebar.number_input("t1",1901,2050, 1950)
        t2 = st.sidebar.number_input("t1", 1901,2050,2000)
        if t2<(t1+30): # or t1 not in year_list  or t2 not in year_list:
            st.error("t2 must be 30 years later than t1 and both years have to be in the dataframe")
            st.stop()
    else:
        t1,t2 = None, None
   
    
    return what_to_show,drawplot,draw30,t1,t2

def main():
    show_info()

    what_to_show, drawplot, draw30, t1, t2,   = interface()
   

     #url = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\de_bilt_jaargem_1901_2022.csv"
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/de_bilt_jaargem_1901_2022.csv" 
    df = getdata(url)
    show_options = [
        "temp_min",
        "temp_avg",
        "temp_max",
    ]
    what_to_show = "temp_avg"

    X_array = df["YYYY"].values
    Y_array = df[what_to_show].values
    df1 = main_output_R_script(draw30)
    
    
    df2 = main_skmisc(X_array, Y_array,t1,t2,draw30)

   
    df_m = pd.merge(df1, df2, on='YYYY')
    
    
    st.subheader("All the results")
    new_column_order = ["YYYY", "temp_avg", "knmi_R_script_loess","skmisc_loess",
                                "knmi_R_script_low",  "skmisc_low",
                                "knmi_R_script_high",  "skmisc_high", "30_yr_average"]
    result = df_m[new_column_order]
    st.write(result)
    st.write(result.round(2))
    
    compare_values_in_df_m( df_m)
   
    show_footer()

    # Return the result as a list of dictionaries
    return # result

def compare_values_in_df_m(df_m, compare_1 =  "knmi_R_script", compare_2 = "skmisc"):
    """ Compare two different methods to calculate LOESS and its CI's
        Shows values and an absolute and relative graph

    Args:
        df_m (df): df with the results of various methos
        compare_1 (str): first method used
        compare_2 (str): second method used


    """    

    for a in ["loess", "high", "low"]:
        df_m[f"diff_{a}_rel_{compare_1}_{compare_2}"] = (df_m[f"{compare_1}_{a}"]-df_m[f"{compare_2}_{a}"]) / df_m[f"{compare_2}_{a}"]*100
        df_m[f"diff_{a}_abs_{compare_1}_{compare_2}"] = df_m[f"{compare_1}_{a}"] - df_m[f"{compare_2}_{a}"]
   

    st.subheader(f"The values of the {compare_1} script compared to the {compare_2} script")
    
    for a in ["loess", "high", "low"]:
        st.write(f"**{a}**")
        st.write(f'Average values - Relatieve {df_m[f"diff_{a}_rel_{compare_1}_{compare_2}"].mean()}')
        st.write(f'Average values - Absolute {df_m[f"diff_{a}_abs_{compare_1}_{compare_2}"].mean()}')

        st.write(f'Max absolute value {a} - {df_m[f"diff_{a}_rel_{compare_1}_{compare_2}"].abs().max()}')
        st.write(f'Max absolute value {a} - {df_m[f"diff_{a}_abs_{compare_1}_{compare_2}"].abs().max()}')
        

    # Select all columns except 'YYYY' for plotting
    columns_to_plot = df_m.columns[(df_m.columns != 'YYYY') & (df_m.columns != '30_yr_average')]
    print (columns_to_plot)
    # Create the Plotly Express graph
    fig = px.line(df_m, x='YYYY', y=columns_to_plot, title='Line Plot of Columns except YYYY',
                labels={'value': 'Values', 'variable': 'Column'})

    st.plotly_chart(fig)

    # Line plot
    a= df_m[f"diff_high_abs_{compare_1}_{compare_2}"]
    b = df_m[f"diff_low_abs_{compare_1}_{compare_2}"]
    c = df_m[f"diff_loess_abs_{compare_1}_{compare_2}"]
    
    to_show = [a,b,c]

    fig_line = px.line(df_m, x='YYYY', y=to_show, title=f'Absolute value of the difference of the {compare_1} script and the {compare_2} script of the CI-intervalborders')
    
    fig_line.add_shape(
        type='line',
        x0=df_m['YYYY'].iloc[0],
        x1=df_m['YYYY'].iloc[-1],
        y0=0,
        y1=0,
        line=dict(color='red', dash='dash')
    )
    st.plotly_chart(fig_line)


    # Line plot
    a= df_m[f"diff_high_rel_{compare_1}_{compare_2}"]
    b = df_m[f"diff_low_rel_{compare_1}_{compare_2}"]
    c = df_m[f"diff_loess_rel_{compare_1}_{compare_2}"]
    
    to_show = [a,b,c]
   
    fig_line = px.line(df_m, x='YYYY', y=to_show, title=f'Relative  difference (%) of the {compare_1} script and the {compare_2} script of the CI-intervalborders')
    
    fig_line.add_shape(
        type='line',
        x0=df_m['YYYY'].iloc[0],
        x1=df_m['YYYY'].iloc[-1],
        y0=0,
        y1=0,
        line=dict(color='red', dash='dash')
    )
    st.plotly_chart(fig_line)


    
    return
    


if __name__ == "__main__":
    main()