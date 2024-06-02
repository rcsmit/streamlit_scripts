import pandas as pd
import numpy as np
import streamlit as st
import scipy.stats as stats
from show_knmi_functions.utils import get_data, loess_skmisc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
#_lock = RendererAgg.lock
import plotly.graph_objects as go
import plotly.express as px

def show_plot(df, datefield, title, wdw, wdw2, sma2_how, what_to_show_, graph_type, centersmooth, show_ci, show_loess, wdw_ci, show_parts, no_of_parts):
    """_summary_

    Args:
        df (_type_): _description_
        datefield (_type_): _description_
        title (_type_): _description_
        wdw (_type_): _description_
        wdw2 (_type_): _description_
        sma2_how (_type_): _description_
        what_to_show_ (_type_): _description_
        graph_type (_type_): _description_
        centersmooth (_type_): _description_
        show_ci (_type_): _description_
        show_loess (_type_): _description_
        wdw_ci (_type_): _description_
        show_parts (_type_): _description_
        no_of_parts (_type_): _description_
    """    
    
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
        # with _lock:
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
            if show_parts:
                n_parts = no_of_parts
                rows_per_part = len(df) // n_parts
                # Step 2: Calculate the average temperature for each part
                average_values = [df.iloc[i * rows_per_part:(i + 1) * rows_per_part][what_to_show_x].mean() for i in range(n_parts)]
            X_array = df[datefield].values
            Y_array = df[what_to_show_x].values
            if len(X_array)>42:
                #y_hat2, x_space2 = calculate_loess(X_array, Y_array, 0.05, 1, all_x = True, num_points = 200)
                x_space2, y_hat2, trendlb, trendub  = loess_skmisc(X_array, Y_array)
               
                if len(y_hat2) >0:
                    loess = go.Scatter(
                        name=f"{what_to_show_x} Loess",
                        x=X_array,
                        y= y_hat2,
                        mode='lines',
                        line=dict(width=1,
                        color='rgba(255, 0, 255, 1)'
                        ),
                        )
                    loess_low = go.Scatter(
                        name=f"{what_to_show_x} Loess low",
                        x=X_array,
                        y= trendlb,
                        mode='lines',
                        line=dict(width=.7,
                        color='rgba(255, 0, 255, 0.5)'
                        ),
                        )
                    loess_high = go.Scatter(
                        name=f"{what_to_show_x} Loess high",
                        x=X_array,
                        y= trendub,
                        mode='lines',
                        line=dict(width=0.7,
                        color='rgba(255, 0, 255, 0.5)'
                        ),
                        )
                    # Create a filled area plot for confidence interval
                    confidence_trace = go.Scatter(x=np.concatenate([X_array, X_array[::-1]]),
                            y=np.concatenate([trendub, trendlb[::-1]]),
                                fill='tozeroy',
                                fillcolor='rgba(0, 128, 0, 0.2)',
                                line=dict(color='dimgrey', width=.5),
                                showlegend=True,
                                name="CI of the trendline")
                else:
                    loess = None
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
            if len(X_array)>42 and loess !=None and show_loess:
                data.append(loess)
                data.append(loess_low)
                data.append(loess_high)
                data.append(confidence_trace)
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

       
        fig = px.histogram(df, x=what_to_show_x, title=f'Histogram of Column {what_to_show_x}')
        st.plotly_chart(fig)  
    #df =df[[datefield,what_to_show_[0]]]
    #st.write(df)
def main():
   
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result.csv" 
    df = get_data(url)
    
if __name__ == "__main__":
    # main()
    print ("")