from imghdr import what
import pandas as pd
import numpy as np
import streamlit as st
#from streamlit import caching
import datetime as dt
import scipy.stats as stats
import math
from show_knmi_functions.utils import show_weerstations, help
from datetime import datetime
import matplotlib.pyplot as plt
# import matplotlib
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

_lock = RendererAgg.lock
import sys # for the progressbar
import shutil # for the progressbar

import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go

import platform
import streamlit.components.v1 as components
import time
import imageio
import os
import webbrowser

def climatrend(t, y, p=None, t1=None, t2=None, ybounds=None, drawplot=False, draw30=False):

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
        Second year (see t1) for which trendline value is compared in the test.
    ybounds : list or array-like, optional
        Lower/upper bound on the value range of y (default: [-Inf, Inf]).
    drawplot : bool or str, optional
        If True, a plot will be drawn. If a string is provided, it will be used as the label on the y-axis.
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

    """
    

    # Fixed parameters
    width = 42
    
    # Check input -> gives error

    # if t is None or y is None or len(t) < 3 or len(t) != len(y):
    #     raise ValueError("t and y arrays must have equal lengths greater than 2.")
    # if np.isnan(t).any() or np.isnan(y).sum() < 3:
    #     raise ValueError("t or y contain too many NA.")
    
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
        avy = np.convolve(yg, np.ones(30) / 30, mode='valid')
        avy2 = np.convolve(yg**2, np.ones(30) / 30, mode='valid')
        avysd = np.sqrt(avy2 - avy**2)
        ind = slice(15, ng - 15)
        avt = avt[ind]
        avy = avy[ind]
        avysd = avysd[ind]

    # Linear LOESS trendline computation
    span = width / ng
    loess_model = sm.nonparametric.lowess(yg, tg, frac=span, return_sorted=False)
    trend = loess_model

    # Confidence limits (normal approximation)
    trendsd = np.std(yg - trend)
    z = 1.96  # 1.96 corresponds to a 95% confidence interval (z-score)
    trendub = trend + z * trendsd
    trendlb = trend - z * trendsd

    # Apply bounds
    trend = np.clip(trend, ybounds[0], ybounds[1])
    trendub = np.clip(trendub, ybounds[0], ybounds[1])
    trendlb = np.clip(trendlb, ybounds[0], ybounds[1])

    # p-value for trend
    pvalue = None
    if t2 in t and t1 in t and t2 >= t1 + 30:
        y1 = trend[t1 == t][0]
        y2 = trend[t2 == t][0]
        y1sd = trendsd[t1 == t][0]
        y2sd = trendsd[t2 == t][0]
        # Two-sided test for absence of trend
        pvalue = (1 - norm.cdf(abs(y2 - y1), scale=np.sqrt(y1sd**2 + y2sd**2))) * 2

    # Plotting
    if drawplot:
        plt.figure(figsize=(8, 6))
        ylim = [np.min([np.min(y), np.min(trendlb)]), np.max([np.max(y), np.max(trendub)])]
        ylim[1] = ylim[0] + (ylim[1] - ylim[0]) * 1.0
        plt.plot(t, y, 'b-', label='Temperature Data')
        plt.plot(t, trend, 'r-', lw=2, label='Trendline')
        plt.fill_between(t, trendlb, trendub, color='grey', alpha=0.5, label='Confidence Interval')
        
        if draw30:
            plt.plot(avt, avy, 'ko', markersize=3, label='30-yr Average')

        plt.xlabel('Year')
        plt.ylabel('Temperature')
        plt.grid()
        plt.legend()
        plt.show()

    # results_df = pd.DataFrame({
    #     't': t,
    #     'trend': trend,
    #     'p': p,
    #     'trendubound': trendub,
    #     'trendlbound': trendlb,
    #     'averaget': avt,
    #     'averagey': avy,
    #     't1': t1,
    #     't2': t2,
    #     'pvalue': pvalue,
    #     'ybounds': ybounds
    # })
    

    # return {'t': t, 'trend': trend, 'p': p, 'trendubound': trendub, 'trendlbound': trendlb,
    #         'averaget': avt, 'averagey': avy, 't1': t1, 't2': t2, 'pvalue': pvalue,
    #         'ybounds': ybounds}
    return t, trend, trendlb, trendub

def show_plot(df, datefield, title, wdw, wdw2, sma2_how, what_to_show_, graph_type, centersmooth, show_ci, wdw_ci, show_parts, no_of_parts):
    what_to_show_ = what_to_show_ if type(what_to_show_) == list else [what_to_show_]
    color_list = [
        "#02A6A8",
        "#4E9148",
        "#F05225",
        "#024754",
        "#FBAA27",
        "#302823",
        "#F07826",
        "#ff6666",
    ]
    if len(df) == 1 and datefield == "YYYY":
        st.warning("Selecteer een grotere tijdsperiode")
        st.stop()

    if graph_type=="pyplot"  :
        with _lock:
            fig1x = plt.figure()
            ax = fig1x.add_subplot(111)
            for i, what_to_show in enumerate(what_to_show_):
                sma = df[what_to_show].rolling(window=wdw, center=centersmooth).mean()
                ax = df[what_to_show].plot(
                    label="_nolegend_",
                    linestyle="dotted",
                    color=color_list[i],
                    linewidth=0.5,
                )
                ax = sma.plot(label=what_to_show, color=color_list[i], linewidth=0.75)
            
            #ax.set_xticks(df[datefield]) #TOFIX : this gives an strange graph
            if datefield == "YYYY":
                ax.set_xticklabels(df[datefield], fontsize=6, rotation=90)
            else:
                ax.set_xticklabels(df[datefield].dt.date, fontsize=6, rotation=90)
            xticks = ax.xaxis.get_major_ticks()
            for i, tick in enumerate(xticks):
                if i % 10 != 0:
                    tick.label1.set_visible(False)

            plt.xticks()
            plt.grid(which="major", axis="y")
            plt.title(title)
            plt.legend()
            st.pyplot(fig1x)
    else:
        fig = go.Figure()
        data=[]
        for what_to_show_x in what_to_show_:
            #fig = go.Figure()
            avg = round(df[what_to_show_x].mean(),1)
            std = round(df[what_to_show_x].std(),1)
            sem = df[what_to_show_x].sem()

            lower68 = round(df[what_to_show_x].quantile(0.16),1)
            upper68 = round(df[what_to_show_x].quantile(0.84),1)

            lower95 = round(df[what_to_show_x].quantile(0.025),1)
            upper95 = round(df[what_to_show_x].quantile(0.975),1)

            # Calculate the moving confidence interval for the mean using the last 25 values
            moving_ci_lower_95 = df[what_to_show_x].rolling(window=wdw_ci).mean() - df[what_to_show_x].rolling(window=wdw_ci).std() * 2
            moving_ci_upper_95 = df[what_to_show_x].rolling(window=wdw_ci).mean() + df[what_to_show_x].rolling(window=wdw_ci).std() * 2

            moving_ci_lower_68 = df[what_to_show_x].rolling(window=wdw_ci).mean() - df[what_to_show_x].rolling(window=wdw_ci).std() * 1
            moving_ci_upper_68 = df[what_to_show_x].rolling(window=wdw_ci).mean() + df[what_to_show_x].rolling(window=wdw_ci).std() * 1

            

          
            # Quantiles and (mean + 2*std) are two different measures of dispersion, which can be used to understand the distribution of a dataset.
 
            # Quantiles divide a dataset into equal-sized groups, based on the values of the dataset. For example, the median is the 50th percentile, which divides the dataset into two equal-sized groups. Similarly, the 25th percentile divides the dataset into two groups, with 25% of the values below the 25th percentile and 75% of the values above the 25th percentile.

            # On the other hand, (mean + 2*std) represents a range of values that are within two standard deviations of the mean. This is sometimes used as a rule of thumb to identify outliers, since values that are more than two standard deviations away from the mean are relatively rare.

            # The main difference between quantiles and (mean + 2std) is that quantiles divide the dataset into equal-sized groups based on the values, while (mean + 2std) represents a range of values based on the mean and standard deviation. In other words, quantiles are based on the actual values of the dataset, while (mean + 2*std) is based on the mean and standard deviation, which are summary statistics of the dataset.

            # It's also worth noting that (mean + 2std) assumes that the data is normally distributed, while quantiles can be used for any distribution. Therefore, if the data is not normally distributed, (mean + 2std) may not be a reliable measure of dispersion.
            # confidence interval for the mean
            ci = stats.t.interval(0.95, len(df[what_to_show_x])-1, loc=df[what_to_show_x].mean(), scale=sem)

            # print confidence interval
          
            n_parts = no_of_parts
            rows_per_part = len(df) // n_parts
            # Step 2: Calculate the average temperature for each part
            average_values = [df.iloc[i * rows_per_part:(i + 1) * rows_per_part][what_to_show_x].mean() for i in range(n_parts)]
            X_array = df[datefield].values
            Y_array = df[what_to_show_x].values
            if len(X_array)>30:
                #y_hat2, x_space2 = calculate_loess(X_array, Y_array, 0.05, 1, all_x = True, num_points = 200)
                x_space2, y_hat2, trendlb, trendub  = climatrend(X_array, Y_array)

                loess = go.Scatter(
                    name=f"{what_to_show_x} Loess",
                    x=x_space2,
                    y= y_hat2,
                    mode='lines',
                    line=dict(width=1,
                    color='rgba(255, 0, 255, 1)'
                    ),
                    )
                loess_low = go.Scatter(
                    name=f"{what_to_show_x} Loess low",
                    x=x_space2,
                    y= trendlb,
                    mode='lines',
                    line=dict(width=.7,
                    color='rgba(255, 0, 255, 0.5)'
                    ),
                    )
                loess_high = go.Scatter(
                    name=f"{what_to_show_x} Loess high",
                    x=x_space2,
                    y= trendub,
                    mode='lines',
                    line=dict(width=0.7,
                    color='rgba(255, 0, 255, 0.5)'
                    ),
                    )
            df["sma"] = df[what_to_show_x].rolling(window=wdw, center=centersmooth).mean()
            if (wdw2 != 999):
                if (sma2_how == "mean"):
                    df["sma2"] = df[what_to_show_x].rolling(window=wdw2, center=centersmooth).mean()
                elif (sma2_how == "median"):
                    df["sma2"] = df[what_to_show_x].rolling(window=wdw2, center=centersmooth).median()

                sma2 = go.Scatter(
                    name=f"{what_to_show_x} SMA ({wdw2})",
                    x=df[datefield],
                    y= df["sma2"],
                    mode='lines',
                    line=dict(width=2,
                    color='rgba(0, 168, 255, 0.8)'
                    ),
                    )
            if wdw ==1:
                name_sma = f"{what_to_show_x}"
            else:
                name_sma = f"{what_to_show_x} SMA ({wdw})"
            sma = go.Scatter(
                name=name_sma,
                x=df[datefield],
                y= df["sma"],
                mode='lines',
                line=dict(width=1,
                color='rgba(0, 0, 255, 0.6)'
                ),
                )
            if wdw != 1:
                points = go.Scatter(
                    name="",
                    x=df[datefield],
                    y= df[what_to_show_x],
                    mode='markers',
                    showlegend=False,
                    marker=dict(
                    #color='LightSkyBlue',
                    size=3))
            # Create traces for the moving confidence interval as filled areas
            ci_area_trace_95 = go.Scatter(
                name=f"{what_to_show_x} 95% CI",
                x=df[datefield],
                y=pd.concat([moving_ci_lower_95, moving_ci_upper_95[::-1]]),  # Concatenate lower and upper CI for the fill
                fill='tozerox',  # Fill the area to the x-axis
                fillcolor='rgba(211, 211, 211, 0.3)',  # Set the fill color to grey (adjust the opacity as needed)
                line=dict(width=0),  # Set line width to 0 to hide the line of the area trace
            )
             # Create traces for the moving confidence interval
            ci_lower_trace_95 = go.Scatter(
                name=f"{what_to_show_x} 95% CI Lower",
                x=df[datefield],
                y=moving_ci_lower_95,
                mode='lines',
                line=dict(width=1, dash='dash'),
            )
            ci_upper_trace_95 = go.Scatter(
                name=f"{what_to_show_x} 95% CI Upper",
                x=df[datefield],
                y=moving_ci_upper_95,
                mode='lines',
                line=dict(width=1, dash='dash'),
            )
            ci_area_trace_68 = go.Scatter(
                name=f"{what_to_show_x} 68% CI",
                x=df[datefield].to_list(),
                y=moving_ci_lower_68+ moving_ci_upper_68[::-1],  # Concatenate lower and upper CI for the fill
                fill='tozerox',  # Fill the area to the x-axis
                fillcolor='rgba(211, 211, 211, 0.5)',  # Set the fill color to grey (adjust the opacity as needed)
                line=dict(width=0),  # Set line width to 0 to hide the line of the area trace
            )
            ci_lower_trace_68 = go.Scatter(
                name=f"{what_to_show_x} 68% CI Lower",
                x=df[datefield],
                y=moving_ci_lower_68,
                mode='lines',
                line=dict(width=1, dash='dash'),
            )
            ci_upper_trace_68 = go.Scatter(
                name=f"{what_to_show_x} 68% CI Upper",
                x=df[datefield],
                y=moving_ci_upper_68,
                mode='lines',
                line=dict(width=1, dash='dash'),
            )

           
            #data = [sma,points]
            data.append(sma)
            if len(X_array)>30:
                data.append(loess)
                data.append(loess_low)
                data.append(loess_high)
            if wdw2 != 999:
                data.append(sma2)
            if wdw != 1:
                data.append(points)
            if show_ci:
                # Append the moving confidence interval traces to the data list
                data.append(ci_lower_trace_95)
                data.append(ci_upper_trace_95)
                data.append(ci_lower_trace_68)
                data.append(ci_upper_trace_68)
                #data.append(ci_area_trace_95)
                #data.append(ci_area_trace_68)

            layout = go.Layout(
                yaxis=dict(title=what_to_show_x),
                title=title,)
                #, xaxis=dict(tickformat="%d-%m")
            # fig = go.Figure(data=data, layout=layout)
            # fig.update_layout(xaxis=dict(tickformat="%d-%m-%Y"))
            # st.plotly_chart(fig, use_container_width=True)
        fig = go.Figure(data=data, layout=layout)
        # Add horizontal lines for average values
        if show_parts:
            for i, avg_val in enumerate(average_values):
                if i != (len(average_values) -1):
                    fig.add_trace(go.Scatter(x=[df[datefield].iloc[i * rows_per_part], df[datefield].iloc[min((i + 1) * rows_per_part - 1, len(df) - 1)]],
                                            y=[avg_val, avg_val],
                                            mode='lines', line=dict(color='red'),showlegend=False, name=f'Avg Part {i + 1}'))
                else:    
                    fig.add_trace(go.Scatter(x=[df[datefield].iloc[i * rows_per_part], df[datefield].iloc[len(df) - 1]],
                                            y=[avg_val, avg_val],
                                            mode='lines', line=dict(color='red'),showlegend=False, name=f'Avg Part {i + 1}'))
                
               
   
        fig.update_layout(xaxis=dict(tickformat="%d-%m-%Y"))
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"{what_to_show_x} | mean = {avg} | std= {std} | quantiles (68%) [{lower68}, {upper68}] | quantiles (95%) [{lower95}, {upper95}]")
            
    #df =df[[datefield,what_to_show_[0]]]
    #st.write(df)