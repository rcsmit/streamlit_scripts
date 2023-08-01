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

def calculate_loess_simply_with_CI(X, y, alpha, deg, N):

    smoothed = calculate_loess_simply(X, y, alpha, deg)
    t = np.asarray(X, dtype=np.float64)
    eval_x = np.linspace(t.min(), t.max(), 1*len(t))
    
    # Perform bootstrap resamplings of the data
    # and  evaluate the smoothing at a fixed set of points
    smoothed_values = np.empty((N, len(eval_x)))
   
    for i in range(N):
        if i % (N/10) == 0:
            print (f"Running {i}/{N}")
        sample_size = int(len(X) *1)
        sample_ = np.random.choice(len(X), sample_size, replace=True)
        sample = np.sort(sample_)
        sampled_x = X[sample]
        sampled_y = y[sample]
        
        
        smoothed_values[i] = calculate_loess_simply(sampled_x, sampled_y, alpha, deg,)
        
    

    # Get the confidence interval
    conf_interval = 0.95
    sorted_values = np.sort(smoothed_values, axis=0)
   
    bound = int(N * (1 - conf_interval) / 2)
    top = sorted_values[bound - 1]
    bottom = sorted_values[-bound]
    sd = (top - bottom)/(1.96 *2)
  
    return smoothed, X,bottom,top, sd


def calculate_loess_simply(X, y, alpha, deg, all_x = True, num_points = 100):
    '''
    Calculate LOESS like explained at the site Simply OR 
    https://simplyor.netlify.app/loess-from-scratch-in-python-animation.en-us/

    Parameters
    ----------
    X : numpy array 1D
        Explanatory variable.
    y : numpy array 1D
        Response varible.
    alpha : double
        Proportion of the samples to include in local regression.
    deg : int
        Degree of the polynomial to fit. Option 1 or 2 only.
    all_x : boolean, optional
        Include all x points as target. The default is True.
    num_points : int, optional
        Number of points to include if all_x is false. The default is 100.

    Returns
    -------
    y_hat : numpy array 1D
        Y estimations at each focal point.
    x_space : numpy array 1D
        X range used to calculate each estimation of y.
    '''
    
    assert (deg == 1) or (deg == 2), "Deg has to be 1 or 2"
    assert (alpha > 0) and (alpha <=1), "Alpha has to be between 0 and 1"
    assert len(X) == len(y), "Length of X and y are different"
    
    if all_x:
        X_domain = X
    else:
        minX = min(X)
        maxX = max(X)
        X_domain = np.linspace(start=minX, stop=maxX, num=num_points)
        
    n = len(X)
    span = int(np.ceil(alpha * n))
    #y_hat = np.zeros(n)
    #x_space = np.zeros_like(X)
    
    y_hat = np.zeros(len(X_domain))
    x_space = np.zeros_like(X_domain)
    
    for i, val in enumerate(X_domain):
    #for i, val in enumerate(X):
        distance = abs(X - val)
        sorted_dist = np.sort(distance)
        ind = np.argsort(distance)
        
        Nx = X[ind[:span]]
        Ny = y[ind[:span]]
        
        delx0 = sorted_dist[span-1]
        
        u = distance[ind[:span]] / delx0
        w = (1 - u**3)**3
        
        W = np.diag(w)
        A = np.vander(Nx, N=1+deg)
        
        V = np.matmul(np.matmul(A.T, W), A)
        Y = np.matmul(np.matmul(A.T, W), Ny)
        Q, R = qr(V)
        p = solve_triangular(R, np.matmul(Q.T, Y))
        #p = np.matmul(pinv(R), np.matmul(Q.T, Y))
        #p = np.matmul(pinv(V), Y)
        y_hat[i] = np.polyval(p, val)
        x_space[i] = val

    trend = y_hat
        
    return y_hat

def lowess_alexandre_gramfort(x, y, f=2. / 3., iter=3):
    '''....the number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations
    https://gist.github.com/agramfort/850437

    # Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
    #
    # License: BSD (3-clause)

    '''
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2
    
    return yest

def lowess_james_brennan(x, y, f=1./3.):
    # https://james-brennan.github.io/posts/lowess_conf/
    """
    Basic LOWESS smoother with uncertainty. 
    Note:
        - Not robust (so no iteration) and
             only normally distributed errors. 
        - No higher order polynomials d=1 
            so linear smoother.
    """
    # get some paras
    xwidth = f*(x.max()-x.min()) # effective width after reduction factor
    N = len(x) # number of obs
    # Don't assume the data is sorted
    order = np.argsort(x)
    # storage
    y_sm = np.zeros_like(y)
    y_stderr = np.zeros_like(y)
    bottom = np.zeros_like(y)
    top = np.zeros_like(y)
    # define the weigthing function -- clipping too!
    tricube = lambda d : np.clip((1- np.abs(d)**3)**3, 0, 1)
    # run the regression for each observation i
    for i in range(N):
        dist = np.abs((x[order][i]-x[order]))/xwidth
        w = tricube(dist)
        # form linear system with the weights
        A = np.stack([w, x[order]*w]).T
        b = w * y[order]
        ATA = A.T.dot(A)
        ATb = A.T.dot(b)
        # solve the syste
        sol = np.linalg.solve(ATA, ATb)
        # predict for the observation only
        yest = A[i].dot(sol)# equiv of A.dot(yest) just for k
        place = order[i]
        y_sm[place]=yest 
        sigma2 = (np.sum((A.dot(sol) -y [order])**2)/N )
        # Calculate the standard error
        y_stderr[place] = np.sqrt(sigma2 * 
                                A[i].dot(np.linalg.inv(ATA)
                                                    ).dot(A[i]))
        
        bottom[place] =  y_sm[place] -1.96*  y_stderr[place]
        top[place] =  y_sm[place] +  1.96* y_stderr[place]
    return y_sm, bottom, top, y_stderr 


def lowess_with_confidence_bounds(
    x, y, eval_x, N, conf_interval=0.95, lowess_kw=None
):
    """
    Perform Lowess regression and determine a confidence interval by bootstrap resampling
    https://www.statsmodels.org/devel/examples/notebooks/generated/lowess.html
    Use the same methode as the KNMI-R script, (translated by ChatGPT)

    statsmodels.api.nonparametric.lowess is used

    Relevant code in R
     # linear LOESS trendline computation
        span <- width/ng
        mdl <- loess(y ~ t, data= data.frame(t= tg, y=yg), span= span,
                    degree= 1, control= control)
        # mdl <- loess(y ~ t, data= data.frame(t= t, y=y), span= span, degree= 1)
        pre <- predict(mdl, newdata= data.frame(t= t), se= TRUE)
        trend <- pre$fit          # trendline
        trendsd <- pre$se.fit     # standard deviation of trendline
        
    """
    # Lowess smoothing
    smoothed = sm.nonparametric.lowess(exog=x, endog=y, xvals=eval_x, it=1, **lowess_kw)


    # Perform bootstrap resamplings of the data
    # and  evaluate the smoothing at a fixed set of points
    smoothed_values = np.empty((N, len(eval_x)))
    for i in range(N):
        if i % (N/10) == 0:
            print (f"Running {i}/{N}")
        sample = np.random.choice(len(x), len(x), replace=True)
        sampled_x = x[sample]
        sampled_y = y[sample]

        smoothed_values[i] = sm.nonparametric.lowess(
            exog=sampled_x, endog=sampled_y, xvals=eval_x, it=1, **lowess_kw
        )

    # Get the confidence interval
    sorted_values = np.sort(smoothed_values, axis=0)
    
    bound = int(N * (1 - conf_interval) / 2)
    bottom = sorted_values[bound - 1]
    top = sorted_values[-bound]
    
    sd_values = (bottom-top)/(1.96 *2)
    
    # #sd = sem(smoothed_values, axis=1)
    # sd = np.nanstd(smoothed_values, axis=1, ddof=0)
    # bottom = smoothed -1.96*sd
    # top =  smoothed +1.96*sd

    mean = np.nanmean(smoothed_values, axis=1)
    sd = sem(smoothed_values, axis=1)
    stderr = np.nanstd(smoothed_values, axis=1, ddof=0)
    return smoothed, bottom, top,sd, 



def climatrend(
    t, y,N, p=None, t1=None, t2=None, ybounds=None, drawplot=False, draw30=False
):

    """
    Fit a trendline to an annually sampled time-series by local linear regression (LOESS)

    Parameters:
    t : numpy array of shape (n,)
        Years, increasing by 1.
    y : numpy array of shape (n,)
        Annual values; missing values as blanks are allowed near the beginning and end.
    p : float, optional
        Confidence level for error bounds (default: 0.95).
    t1 : float, optional
        First year for which trendline value is compared in the test.
    t2 : float, optional
        Second year (see t1) for which trendline value is compared in the test. Must be >30 higher than t1
    ybounds : list or array-like, optional
        Lower/upper bound on the value range of y (default: [-Inf, Inf]).
    drawplot : bool or str, optional
        If True, a plot will be drawn.
        // If a string is provided, it will be used as the label on the y-axis. - - Not in the Python script //
        (default: False).
    draw30 : bool, optional
        If True, add 30-year moving averages to the plot (default: False).

    Returns:
    pandas DataFrame or dictionary
        A DataFrame or dictionary with the following columns/values:
            't': years,
            'trend': trendline in y for years in t,
            'p': confidence level,
            'trendubound': lower confidence limit,
            'trendlbound': upper confidence limit,
            'averaget': central value of t in a 30-year interval,
            'averagey': 30-year average of y,
            't1': first year for which trendline value is compared in the test,
            't2': second year for which trendline value is compared in the test,
            'pvalue': p-value of the test of no long-term change,
            'ybounds': bounds on the value range of y.

    Details:
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
    KNMI Technical report TR-389 (see http://bibliotheek.knmi.nl/knmipubTR/TR389.pdf)

    Author: Cees de Valk (cees.de.valk@knmi.nl)

    # https://gitlab.com/cees.de.valk/trend_knmi/-/blob/master/R/climatrend.R?ref_type=heads
    # translated from R to Python by ChatGPT and adapted by Rene Smit
    # not tested 100%
    # P value for trend doesnt work

    """

    # Fixed parameters
    width = 42

    # Check input -> gives error

    if t is None or y is None or len(t) < 3 or len(t) != len(y):
        raise ValueError("t and y arrays must have equal lengths greater than 2.")

    if np.isnan(t).any() or np.isnan(y).sum() > 3:
        # raise ValueError("t or y contain too many NA.")
        st.error("t or y contain too many NA.")
        st.stop()

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

    span = width / ng
    eval_x = np.linspace(t.min(), t.max(), len(t))
    trend, trendlb, trendub, trendsd = lowess_with_confidence_bounds(
        tg, y, eval_x,N, lowess_kw={"frac": span}
    )

    #trend, trendlb, trendub, trendsd = lowess_james_brennan( tg, y, f=span)
    # Apply bounds
    trend = np.clip(trend, ybounds[0], ybounds[1])
    trendub = np.clip(trendub, ybounds[0], ybounds[1])
    trendlb = np.clip(trendlb, ybounds[0], ybounds[1])

    # p-value for trend
    pvalue = None
    if t2 in t and t1 in t and t2 >= t1 + 30:
        idx_t1 = np.where(t == t1)[0][0]
        idx_t2 = np.where(t == t2)[0][0]
        y1 = trend[idx_t1]#[0]
        y2 = trend[idx_t2]#[0]
        y1sd = trendsd[idx_t1]#[0]
        y2sd = trendsd[idx_t2]#[0]
        # Two-sided test for absence of trend
        pvalue = (1 - norm.cdf(abs(y2 - y1), scale=np.sqrt(y1sd**2 + y2sd**2))) * 2
        

# 
#       pvalue <- (1-pnorm(abs(y2-y1)/sqrt(y1sd^2+y2sd^2)))*2
# 
    # Plotting, left it here to stay close to the original script. More logical is to call it from main()
    if drawplot:
        drawplot_matplotlib(t, y, draw30, avt, avy, trend, trendub, trendlb)

    return t, trend, trendlb, trendub, avt, avy, p, t1, t2, pvalue


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


def show_plot_plotly(title, what_to_show, t, values_, trend, trendlb, trendub, avt=None, avy=None,  draw30=False, draw_ci=True):
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
    if draw_ci:
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
    if draw_ci:
        # data.append(loess_low)
        # data.append(loess_high )
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
    #Y_values = np.concatenate([values_np, trendub , trendlb]) 
    
    
    layout = go.Layout(
        yaxis=dict(title=what_to_show, range=[y_lower_bound, y_upper_bound ]), title=f"Year averages of {what_to_show} - {title}"
    )
 

    fig = go.Figure(data=data, layout=layout)

    fig.update_layout(xaxis=dict(tickformat="%d-%m-%Y"))
    st.plotly_chart(fig, use_container_width=True)

def show_returned_values(t, trend, trendlb, trendub, avt, avy, p, t1, t2, pvalue):
    """Show the returned values from the function climatrend()

    Args:
        t (_type_): _description_
        trend (_type_): _description_
        trendlb (_type_): _description_
        trendub (_type_): _description_
        avt (_type_): _description_
        avy (_type_): _description_
        p (_type_): _description_
        t1 (_type_): _description_
        t2 (_type_): _description_
        pvalue (_type_): _description_
    """    
    st.subheader("Returned values")

    st.write(f"p: {p}")
    st.write(f"t1: {t1}")
    st.write(f"t2: {t2}")
    if pvalue != None:
        st.write(f"pvalue: {round(pvalue,4)}")
        if pvalue <0.05:
            st.info(f"The data indicates a long time change between {t1} and {t2}.")
        else:
            st.info(f"The data does not indicate (or a little) a long time change between {t1} and {t2}.")

    with st.expander("Arrays"):
        st.write(f"t: {t} - lengte:{len(t)}")
        st.write(f"trend: {trend} - lengte:{len(trend)}")
        st.write(f"trendubound: {trendub} - lengte:{len(trendub)}")
        st.write(f"trendlbound: {trendlb} - lengte:{len(trendlb)}")
        st.write(f"averaget: {avt} - lengte:{len(avt)}")
        st.write(f"averagey: {avy} - lengte:{len(avy)}")


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


def main_translated_script(df, N, X_array, Y_array, what_to_show, drawplot, draw30, t1, t2 ):
    """Calculation with script from KNMI

    """    
 
    t, trend, trendlb, trendub, avt, avy, p, t1_, t2_, pvalue = climatrend(
        X_array,
        Y_array,
        N,
        p=None,
        t1=t1,
        t2=t2,
        
        ybounds=None,
        drawplot=drawplot,
        draw30=draw30,
    )
    st.subheader("Results KNMI script translated in python - uses [statsmodels.api.nonparametric.lowess]")
    st.write("https://gitlab.com/cees.de.valk/trend_knmi/-/blob/master/R/climatrend.R?ref_type=heads")
    title = "KNMI script translated in python / statsmodels"
    show_plot_plotly(title, what_to_show, t, Y_array, trend, trendlb, trendub, None, None,  draw30, draw_ci=True)

    
    show_returned_values(t, trend, trendlb, trendub, avt, avy, p, t1_, t2_, pvalue)
    # Create a dictionary with column names as keys and lists as values
    data = {'YYYY': t, 'statsmodel_loess': trend, 'statsmodel_low': trendlb, 'statsmodel_high':trendub}

    # Create a DataFrame from the dictionary
    df1 = pd.DataFrame(data)
    return df1

def interface():
    show_options = [
        "temp_min",
        "temp_avg",
        "temp_max",
    ]
    #year_list = df["YYYY"].to_list()
    what_to_show = st.sidebar.selectbox("Wat weer te geven", show_options, 1)
    N = st.sidebar.number_input("Number of iterations (bootstrapping)",100,100000, 100)
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
    compare_1 = st.sidebar.selectbox("Compare 1",["knmi_R_script",  "statsmodel","skmisc","simply", "james"],0)
    compare_2 = st.sidebar.selectbox("Compare 2",["knmi_R_script",  "statsmodel","skmisc","simply", "james"],1)
    
    return what_to_show,drawplot,draw30,t1,t2, N, compare_1, compare_2


def main_alex(N,what_to_show, X_array, Y_array):
    
    """Calculation with script from the internet

    """   
    deg=1
    alpha = 42/len(X_array)
  
    y_hat  = lowess_alexandre_gramfort(X_array, Y_array, f=alpha, iter=3)
    st.subheader("Results script found on internet from Alexandre Gramfort")
    st.write("https://gist.github.com/agramfort/850437")
    st.write("The trendline is quit a bit off, I dint make a bootstrapping for it.")
    trendlb, trendub = None, None
    title = "Alexandre Gramfort"
    show_plot_plotly(title, what_to_show,  X_array,Y_array,  y_hat,trendlb, trendub, avt=None, avy=None,  draw30=False, draw_ci=False)
    
   # Create a dictionary with column names as keys and lists as values
    data = {'YYYY': X_array, 'alex_loess': y_hat}

    # Create a DataFrame from the dictionary
    df2 = pd.DataFrame(data)
    return df2


def main_james(N,what_to_show, X_array, Y_array ):
    
    """Calculation with script from the internet

    """   
   
    deg=1
    alpha = 42/len(X_array)
    
    y_hat, trendlb, trendub, sd  = lowess_james_brennan(X_array, Y_array, f=alpha)

    st.subheader("Results script found on internet from James Brennan")
    st.write("https://james-brennan.github.io/posts/lowess_conf/")
    st.write("The trendline is very close, but there seems to be a problem with the confidence intervals :)")
    title ="James Brennan"
    show_plot_plotly(title, what_to_show,  X_array,Y_array,  y_hat,trendlb, trendub, avt=None, avy=None,  draw30=False, draw_ci=True)
    
   # Create a dictionary with column names as keys and lists as values
    data = {'YYYY': X_array, 'james_loess': y_hat, 'james_low': trendlb, 'james_high':trendub}

    # Create a DataFrame from the dictionary
    df2 = pd.DataFrame(data)
    return df2


def main_simply(N, what_to_show, X_array, Y_array):
    """Calculation with script from the internet

    """   
   
    deg=1
    alpha = 42/len(X_array)
    
    y_hat, x_space, trendub,trendlb, sd  = calculate_loess_simply_with_CI(X_array, Y_array, alpha, deg,  N)

    st.subheader("Results script found on internet from simply OR")
    st.write("https://simplyor.netlify.app/loess-from-scratch-in-python-animation.en-us/")
    st.write("The trendline is exactly like the output of the R-script of KNMI, but the confidence intervals are much bigger esp. between 1970-2000")
    title = "Simply OR"
    show_plot_plotly(title, what_to_show,  X_array,Y_array,  y_hat,trendlb, trendub, avt=None, avy=None,  draw30=False, draw_ci=True)
    
   # Create a dictionary with column names as keys and lists as values
    data = {'YYYY': X_array, 'simply_loess': y_hat, 'simply_low': trendlb, 'simply_high':trendub}

    # Create a DataFrame from the dictionary
    df2 = pd.DataFrame(data)
    return df2

def main_output_R_script(draw30):
    df3 = getdata("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/trend_de_bilt_jaargem_1901_2022.csv" )
    calculate_sd_values = False
    if calculate_sd_values:
        # Calculate the window size for the standard deviation
        window_size = 15
        # Calculate the standard deviation for each point using a rolling window
        df3['Std_Dev'] = df3['temp_avg'].rolling(window=2*window_size+1, center=True, min_periods=1).std()
        # Calculate the confidence interval for the values
        df3["knmi_R_script_high"] = df3["knmi_R_script_loess"] + 1.96 * df3['Std_Dev']
        df3["knmi_R_script_low"] = df3["knmi_R_script_loess"] - 1.96 * df3['Std_Dev']


  
    st.subheader("Results KNMI script in R")
    st.write("These are the values in the output of the script in R, and seen as 'standard")
    title = "Output R script KNMI (golden standard)"
    show_plot_plotly(title, "temp_avg", df3["YYYY"],df3["temp_avg"],df3["knmi_R_script_loess"],
                                df3["knmi_R_script_low"],df3["knmi_R_script_high"],  df3["YYYY"],
                                  df3["30_yr_average"],  draw30, True)
    return df3

def main_skmisc(X_array, Y_array, t1,t2):
    """Make a plot and calculate p-value with scikit-misc

    Args:
        X_array : list of Years, increasing by 1.
        Ya_array : list of Annual values
        t1 : float, optional
            First year for which trendline value is compared in the test.
        t2 : float, optional
             year (see t1) for which trendline value is compared in the test. Must be >30 higher than t1

    Returns:
        _type_: _description_
    """

    # https://has2k1.github.io/scikit-misc/stable/generated/skmisc.loess.loess.html
    # https://stackoverflow.com/questions/31104565/confidence-interval-for-lowess-in-python
    st.subheader("Lowess with SciKit-Misc")
    st.write("https://has2k1.github.io/scikit-misc/stable/generated/skmisc.loess.loess.html")
    st.write("The only one who gives standard error and CI's; without bootstrapping")
  
    l = loess(X_array, Y_array)
    l.fit()
    pred = l.predict(X_array, stderror=True)
    conf = pred.confidence()


    ste = pred.stderr
    lowess = pred.values
    ll = conf.lower
    ul = conf.upper
    
    show_plot_plotly("Scikit-misc", "temp_avg", X_array, Y_array, lowess, ll, ul, None, None, False, True)

    # # Create a scatter plot for the data points
    # scatter_trace = go.Scatter(x=X_array, y=Y_array, mode='lines', line=dict(width=0.7, color="rgba(255, 0, 255, 0.5)"), name='Data Points')

    # # Create a line plot for lowess
    # lowess_trace = go.Scatter(x=X_array, y=lowess, mode='lines',  name='Lowess')

    # # Create a filled area plot for confidence interval
    # confidence_trace = go.Scatter(x=np.concatenate([X_array, X_array[::-1]]),
    #                             y=np.concatenate([ul, ll[::-1]]),
    #                             #fill='tozeroy',
    #                             fillcolor='rgba(0, 128, 0, 0.2)',
    #                             line=dict(color='dimgrey', width=.5),
    #                             showlegend=False)

    # # Combine the traces into a single figure
    # fig = go.Figure([scatter_trace, lowess_trace, confidence_trace])

    # # Update layout settings
    # fig.update_layout(
    #     title='Lowess with Confidence Interval',
    #     xaxis_title='X Values',
    #     yaxis_title='Y Values',
    #     showlegend=True,
    #     hovermode='x'
    # )

    # # Show the Plotly figure
    # st.plotly_chart(fig)
    t = np.asarray(X_array, dtype=np.float64)
    y = np.asarray(Y_array, dtype=np.float64)
    dt = np.diff(t)[0]
    n = len(y)
    ig = ~np.isnan(y)
    yg = y[ig]
    tg = t[ig]
    ng = sum(ig)
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


    data = {'YYYY': X_array, 'skmisc_loess': lowess, 'skmisc_low': ll, 'skmisc_high':ul}

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)
    return df

def show_footer():
    st.write("A good introduction about LOWESS/LOESS can be found here :https://aitechtrend.com/smoothing-out-the-noise-analyzing-data-with-lowess-regression-in-python/")

    st.write("The difference between the two acronyms or names is mostly superficial, but there is an actual difference in R–there are two different functions, lowess() and loess(). Lowess was implemented first, while the latter (loess) is more flexible and powerful. The loess() function creates an object that contains the results, and the predict() function retrieves the fitted values.[1]")
    st.write("[1] https://www.ime.unicamp.br/~dias/loess.pdf")
    st.info("Source: https://github.com/rcsmit/streamlit_scripts/blob/main/loess.py")
        
def main():
    show_info()

    what_to_show, drawplot, draw30, t1, t2, N, compare_1, compare_2  = interface()
    n_values = [N]

    # Initialize an empty list to store the results for each N
    all_results = []

    # Calculate the values for each N and store them in the list
    for N in n_values:
        s1 = int(time.time())
        # st.header(f" ----- {N} -----")
        # print (f" ----- {N} -----")
        # results = 
        main_calculations(N, what_to_show, drawplot, draw30, t1, t2, compare_1, compare_2)
        # all_results.extend(results)
        s2 = int(time.time())
        s2x = s2 - s1
        print(" ")  # to compensate the  sys.stdout.flush()

        print(f"{N} Iterations  took {str(s2x)} seconds ....)")
    show_footer()

    # Display the results in a table using Streamlit
    # df_results = pd.DataFrame(all_results)
    # st.subheader("All the results")
    # st.table(df_results)
    # df_results.to_csv(f"comparison_of_N.csv", index=False)

def main_calculations(N, what_to_show, drawplot, draw30, t1, t2, compare_1, compare_2):
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
    
    df2 = main_translated_script(df, N,X_array, Y_array, what_to_show, drawplot, draw30, t1, t2)
        
    df3 = main_simply(N,what_to_show, X_array, Y_array)
    
    df6 = main_skmisc(X_array, Y_array,t1,t2)

    df4= main_james(N,what_to_show,X_array, Y_array)
    df5 = main_alex(N,what_to_show,X_array, Y_array)
    df_m = pd.merge(df1, df2, on='YYYY').merge(df6, on='YYYY').merge(df3, on='YYYY').merge(df4, on='YYYY').merge(df5, on='YYYY')
    #df_m = pd.merge(df1, df2, on='YYYY').merge(df6, on='YYYY')
   
    st.subheader("All the results")
    st.write("temp_avg = the real average temperatures. Trend = values from R script, statsmodel = translated to python, (statsmodel), skmisc = scikit-misc, and the others are simply, james & alex")
    new_column_order = ["YYYY", "temp_avg", "knmi_R_script_loess", "statsmodel_loess","skmisc_loess","simply_loess",  "james_loess","alex_loess","knmi_R_script_low",  "statsmodel_low","skmisc_low","simply_low", "james_low", "knmi_R_script_high",  "statsmodel_high",   "skmisc_high",  "simply_high", "james_high",  "30_yr_average"]
    result = df_m[new_column_order]
    st.write(result)
    st.write(result.round(2))
    print (result.dtypes)
    compare_values_in_df_m(N, df_m, compare_1, compare_2)
    #result = output_df_m(N, df_m)

    # Return the result as a list of dictionaries
    return # result

def compare_values_in_df_m(N, df_m, compare_1, compare_2):
    
    
  

    for a in ["loess", "high", "low"]:
        df_m[f"diff_{a}_rel_{compare_1}_{compare_2}"] = df_m[f"{compare_1}_{a}"] / df_m[f"{compare_2}_{a}"]*100
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
    print (to_show)
    fig_line = px.line(df_m, x='YYYY', y=to_show, title=f'Absolute value of the difference of the {compare_1} script and the {compare_2} script of the CI-intervalborders')

    st.plotly_chart(fig_line)

    # result = {
    #     'N': N,
    #     'Average value diff_val rel_simply': df_m['diff_val_rel_simply'].mean(),    
    #     'Average value diff_up rel_simply': df_m['diff_up_rel_simply'].mean(),
    #     'Average value diff_low rel_simply': df_m['diff_low_rel_simply'].mean(),
    #     'Average value diff_up abs_simply': df_m['diff_up_abs_simply'].mean(),
    #     'Average value diff_low abs_simply': df_m['diff_low_abs_simply'].mean(),
    #     'Max absolute value diff_up_simply': df_m['diff_up_abs_simply'].abs().max(),
    #     'Max absolute value diff_low_simply': df_m['diff_low_abs_simply'].abs().max(),
    #     'Average value diff_val rel_sm': df_m['diff_val_rel_sm'].mean(),    
    #     'Average value diff_up rel_sm': df_m['diff_up_rel_sm'].mean(),
    #     'Average value diff_low rel_sm': df_m['diff_low_rel_sm'].mean(),
    #     'Average value diff_up abs_sm': df_m['diff_up_abs_sm'].mean(),
    #     'Average value diff_low abs_sm': df_m['diff_low_abs_sm'].mean(),
    #     'Max absolute value diff_up_sm': df_m['diff_up_abs_sm'].abs().max(),
    #     'Max absolute value diff_low_sm': df_m['diff_low_abs_sm'].abs().max()
    # }
    
    return #result
    



    # Calculate LOESS from scatch: https://simplyor.netlify.app/loess-from-scratch-in-python-animation.en-us/
if __name__ == "__main__":
    main()