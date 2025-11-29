# Scrapping Weather Data from open-meteo
#
# Based on code of @orwel2022

# To install the needed packages : 
# pip install pandas requests beautifulsoup4 html5lib


import pandas as pd
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import platform
import os
from io import StringIO
import matplotlib.pyplot as plt
from show_knmi_functions.show_calender_heatmap import show_calender_heatmap
from show_knmi_functions.show_year_heatmap import show_year_heatmap

#from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.colors import ListedColormap
#_lock = RendererAgg.lock
import numpy as np
# when using without Streamlit, to avoid 127.0.0.1 refused to connect :
# plotly.offline.init_notebook_mode(connected=True)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import datetime as dt
import requests

from scipy.stats import linregress
import statsmodels.api as sm
import pandas as pd
from scipy.stats import kendalltau
import pydeck as pdk
try:
    st.set_page_config(layout='wide')
except:
    pass


import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress, kendalltau
import statsmodels.api as sm


@st.cache_data()
def get_data(location_name,locations, start_date, end_date):
    """Get the weather data for a specific location and date range from Open Meteo archive API.
    https://open-meteo.com/en/docs/historical-weather-api?daily=rain_sum,temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_hours,precipitation_sum,visibility_mean,visibility_min,visibility_max,relative_humidity_2m_mean,relative_humidity_2m_max,relative_humidity_2m_min,&latitude=9.755106899960907&longitude=99.9609068&start_date=2025-01-01&end_date=2025-07-15&hourly=&timezone=Asia%2FBangkok
    Args:
        location_name(str):Location
        start_date(str): start date yyyy-mm-dd
        end_date(str): end date yyyy-mm-dd
    returns:
        pd.DataFrame: DataFrame with weather data including date, temperature, rain, humidity, visibility, etc.
    """
    

    # start_date = "2025-01-01"
    # end_date = "2025-07-15"
    daily_vars = ",".join([
        "rain_sum",
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_hours",
        "precipitation_sum",
        "visibility_mean",
        "visibility_min",
        "visibility_max",
        "relative_humidity_2m_mean",
        "relative_humidity_2m_max",
        "relative_humidity_2m_min",
        "wind_speed_10m_max"
    ])

    # Find the matching location
    loc = next((l for l in locations if l["name"] == location_name), None)
    if loc is None:
        raise ValueError(f"Location '{location_name}' not found.")

    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={loc['lat']}"
        f"&longitude={loc['lon']}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        f"&daily={daily_vars}"
        f"&timezone={loc['timezone'].replace('/', '%2F')}"
    )


    print(f"Fetching weather for {location_name}: {url}")
    response = requests.get(url)
    response.raise_for_status()
    data = response.json().get("daily")
    if not data or "time" not in data:
        raise KeyError("Invalid weather API response structure")
    
    df = pd.DataFrame({
        "date": pd.to_datetime(data["time"]),
        "Date": pd.to_datetime(data["time"]), #some external scripts use Date
        "temp_max": data["temperature_2m_max"],
        "temp_min": data["temperature_2m_min"],
        "temp_mean": data["temperature_2m_mean"],
        "rain_sum": data["rain_sum"],
        "precipitation_hours": data["precipitation_hours"],
        "precipitation_sum": data["precipitation_sum"],
        "visibility_mean": data["visibility_mean"],
        "visibility_min": data["visibility_min"],
        "visibility_max": data["visibility_max"],
        "rel_humidity_mean": data["relative_humidity_2m_mean"],
        "rel_humidity_max": data["relative_humidity_2m_max"],
        "rel_humidity_min": data["relative_humidity_2m_min"],
        "wind_max": data["wind_speed_10m_max"]
    })

    df["year"] = df["date"].dt.isocalendar().year
    df["week"] = df["date"].dt.isocalendar().week
    df = df.dropna(subset=["year", "week"])
    
    if 1==2:

        url_hourly = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={loc['lat']}"
            f"&longitude={loc['lon']}"
            f"&start_date={start_date}"
            f"&end_date={end_date}"
            f"&hourly=rain,temperature_2m,is_day,sunshine_duration"
            f"&timezone={loc['timezone'].replace('/', '%2F')}"
        )

        print(f"Fetching weather for {location_name}: {url_hourly}")
        response = requests.get(url)
        response.raise_for_status()
        data = response.json().get("hourly")
        st.write(data)

        # if not data or "time" not in data:
        #     raise KeyError("Invalid weather API response structure")
        
        df_hourly = pd.DataFrame({
            "date": pd.to_datetime(data["time"]),
            "Date": pd.to_datetime(data["time"]), #some external scripts use Date
            "temp": data["temperature_2m"],
            "rain": data["rain"],
            "is_day": data["is_day"],
            "sunshine_duration": data["sunshine_duration"]  
            
            })

        df_hourly["year"] = df_hourly["date"].dt.isocalendar().year
        df_hourly["time"] = df_hourly["date"].dt.isocalendar().time
        df_hourly["week"] = df_hourly["date"].dt.isocalendar().week
        df_hourly = df_hourly.dropna(subset=["year", "week"])
        
    else:
        df_hourly=None

    return df,df_hourly


def show_warmingstripes(df_, to_show, where):
    """_summary_

    # Based on code of Sebastian Beyer
    # https://github.com/sebastianbeyer/warmingstripes/blob/master/warmingstripes.py

    # the colors in this colormap come from http://colorbrewer2.org
    # the 8 more saturated colors from the 9 blues / 9 reds
    # https://matplotlib.org/matplotblog/posts/warming-stripes/
    

    Args:
        df_ (_type_): _description_
        to_show (_type_): _description_
        where (_type_): _description_
    """ 
    
    df = df_.groupby(df_["Year"], sort=True).mean(numeric_only = True).reset_index()
    #df_grouped = df.groupby([df[valuefield]], sort=True).sum().reset_index()
    

    cmap = ListedColormap(
        [
            "#08306b",
            "#08519c",
            "#2171b5",
            "#4292c6",
            "#6baed6",
            "#9ecae1",
            "#c6dbef",
            "#deebf7",
            "#fee0d2",
            "#fcbba1",
            "#fc9272",
            "#fb6a4a",
            "#ef3b2c",
            "#cb181d",
            "#a50f15",
            "#67000d",
        ]
    )
    temperatures = df[to_show].tolist()
    stacked_temps = np.stack((temperatures, temperatures))
    #with _lock:
    # plt.figure(figsize=(4,18))
    fig, ax = plt.subplots()

    fig = ax.imshow(
        stacked_temps,
        cmap=cmap,
        aspect=40,
    )
    plt.gca().set_axis_off()

    plt.title(f"{to_show} in {where}")
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.show()
    # st.pyplot(fig) - gives an error
    st.set_option("deprecation.showPyplotGlobalUse", False)
    st.pyplot(fig)
    plt.close()
def show_month(df, to_show, day_min, day_max, month, month_names, where, verbose=True):
    """Show graph with to_show in different lines for each years for a certain (period of) a month
       Show a frequency table of this data
       Show a histogram of this data 

    Args:
        df (_type_): _description_
        to_show (_type_): _description_
        day_min (_type_): _description_
        day_max (_type_): _description_
        month (_type_): _description_
        month_names (_type_): _description_
        where (_type_): _description_
    """   
    if verbose:
        st.header(f"Show month {to_show} in {where} in {month_names[month-1]}")
    else:
        if to_show=="precipitation_hours":
            st.subheader(f"prec hours")
        else:
            st.subheader(to_show)

    df_month = df[(df['Month'] == month) & (df['Day']>=day_min) & (df['Day']<=day_max)]             
    if verbose:
        fig = px.line(df_month, x='Day', y=to_show, color='Year', labels={'temp': f'{to_show}'}, title=f'{to_show} for {month_names[month-1]} in {where}')
        st.plotly_chart(fig)
    crosstable_daily = pd.pivot_table(df_month, values=to_show, index='Year', columns='Day', aggfunc='mean').round(1)
    avg_value = np.nanmean(crosstable_daily.to_numpy())
    if verbose:
        st.metric(
            label=f"Average daily value {to_show} - {month_names[month-1]} - {where}",
            value=f"{avg_value:.1f}"
        )
    else:
        
        if to_show in ["dry_day","rain_day"]:
            st.metric(
                label=f"{to_show}",
                value=f"{avg_value*100:.0f} %"
            )
        elif to_show=="precipitation_hours":
            st.metric(
            label=f"prec hours",
            value=f"{avg_value:.1f} hrs"
            )

        else:
            st.metric(
            label=f"{to_show}",
            value=f"{avg_value:.1f}"
            )
    #st.write (crosstable_daily)
    if verbose:
        title=f"Daily values  of {to_show} - {month_names[month-1]} - {where}"
    else:
        title=f"{to_show}"
    # fig=px.imshow(crosstable_daily, title=title, text_auto='.1f')
    fig = px.imshow(
        crosstable_daily,
        title=title,
        text_auto=".0f",
        color_continuous_scale=[(0.0, "white"), (1.0, "darkblue")],
        zmin=0,  # make sure 0 is the lowest value
        
    )
    if not verbose:
        fig.update_coloraxes(showscale=False)

    st.plotly_chart(fig, key=f'{title}{datetime.now().strftime("%H:%M:%S.%f")[:-3]}')
        # Average of all cells (ignores NaNs)
    if verbose:  
        
        # is this table used?

        frequency_table = df_month[to_show].value_counts().reset_index()
        frequency_table.columns = [to_show, 'Frequency']
        # Sort the frequency table by the variable 'to_show'
        frequency_table = frequency_table.sort_values(by=to_show, ascending=True)
        # Calculate cumulative absolute frequency
        frequency_table['Cumulative Absolute Frequency'] = frequency_table['Frequency'].cumsum()

        # Calculate cumulative percentage
        total_absolute_frequency = frequency_table['Frequency'].sum()
        frequency_table['Cumulative Percentage'] = (frequency_table['Cumulative Absolute Frequency'] / total_absolute_frequency) * 100

        # # Display the frequency table
    
        fig=px.line(frequency_table, x=to_show, y='Cumulative Percentage', title=f'Frequency of {to_show} for {month_names[month-1]} in {where}',)
        st.plotly_chart(fig)
        
        # Frequency table
        df_month['Temp_Bin'] = pd.cut(df_month[to_show], bins=range(int(df_month[to_show].min()), int(df_month[to_show].max()) + 2))
        frequency_table = df_month.pivot_table(index='Temp_Bin', columns='Year', aggfunc='size', fill_value=0)

        # Reset the index for plotting
        frequency_table = frequency_table.reset_index()
        frequency_table['Temp_Bin'] = frequency_table['Temp_Bin'].apply(lambda x: x.mid)

        # # Display the frequency table
        # st.write(frequency_table)

        # Frequency line graph
        frequency_table_melted = frequency_table.melt(id_vars='Temp_Bin', var_name='Year', value_name='Frequency')
        fig = px.line(frequency_table_melted, x='Temp_Bin', y='Frequency', color='Year',
                    labels={'Temp_Bin': f'{to_show}', 'Frequency': 'Frequency'},
                    title=f'Frequency of {to_show} for {month_names[month-1]} in {where}')
        st.plotly_chart(fig)
        nbins = len(range(int(df_month[to_show].min()), int(df_month[to_show].max()) + 2))
        # Histogram of the to_show variable
        fig = px.histogram(df_month, x=to_show, nbins=nbins, title=f'Histogram of of {to_show} for {month_names[month-1]} in {where}', 
                        labels={to_show: f'{to_show}', 'count': 'Frequency'})
        st.plotly_chart(fig)


    #show_warmingstripes(df_month, to_show, where)

def cross_table_montly_avg(df, to_show, where, y_axis_zero, selected_month):  
    """_summary_

    Args:
        df (_type_): _description_
        to_show (_type_): _description_
        where (_type_): _description_
        y_axis_zero (_type_): _description_
    """
    # CROSS TABLE WITH MONTLY AVERAGES
    st.subheader (f"Monthly averages of {to_show} - {where}")
    crosstable = pd.pivot_table(df, values=to_show, index='Month', columns='Year', aggfunc='mean').round(1)
    st.write (crosstable)
    
    title= f"Monthly averages of {to_show} - {where}"
    # Create the heatmap using Plotly Express
    fig = px.imshow(crosstable, title=title)  
    st.plotly_chart(fig)


    transposed_df = crosstable.T

    # Initialize a new Plotly figure
    fig_x = go.Figure()

    # Create a scatter plot with a trendline for each month
    for column in transposed_df.columns:
        # Extract the data for the current month
        x = transposed_df.index.values.reshape(-1, 1)  # Years
        y = transposed_df[column].values  # Monthly averages


        mask = ~np.isnan(y)
        x_filtered = x[mask].reshape(-1, 1)
        y_filtered = y[mask]
        
        # Create scatter plot for the current month
        fig_x.add_trace(go.Scatter(x=x_filtered.flatten(), y=y_filtered, mode='markers', name=f'{column} Scatter'))

        # Fit a linear regression model
        if len(y_filtered) > 1:  # Ensure there are enough data points to fit a model
            model = LinearRegression()
            model.fit(x_filtered, y_filtered)
            trendline = model.predict(x_filtered)

            # Calculate the equation and R² value
            slope = model.coef_[0]
            intercept = model.intercept_
            r_squared = r2_score(y_filtered, trendline)

            # Add the trendline to the plot
            fig_x.add_trace(go.Scatter(x=x_filtered.flatten(), y=trendline, mode='lines', name=f'{column} Trendline'))

            # Add annotations for the equation and R² value
            equation_text = f'y = {slope:.2f}x + {intercept:.2f} | R² = {r_squared:.2f}'
            st.write(f"{column} - {equation_text}")
            #fig_x.add_annotation(x=x_filtered.flatten()[-1], y=trendline[-1], text=equation_text, showarrow=True, arrowhead=1)

    # Update layout
    fig_x.update_layout(
        title=f'Monthly averages of {to_show} through time - {where}',
        xaxis_title='Years',
        yaxis_title=f'Monthly average of {to_show}'
    )

    if y_axis_zero:
        fig_x.update_layout(yaxis_range=[0, max(transposed_df.max())])

    # Display the plot using Streamlit
    st.plotly_chart(fig_x)
def show_info():
    st.info("Source weather info: https://open-meteo.com/en/docs/historical-weather-api/")

    '''
Zippenfenig, P. (2023). Open-Meteo.com Weather API [Computer software]. Zenodo. https://doi.org/10.5281/ZENODO.7970649

Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., Thépaut, J-N. (2023). ERA5 hourly data on single levels from 1940 to present [Data set]. ECMWF. https://doi.org/10.24381/cds.adbb2d47

Muñoz Sabater, J. (2019). ERA5-Land hourly data from 2001 to present [Data set]. ECMWF. https://doi.org/10.24381/CDS.E2161BAC

Schimanke S., Ridal M., Le Moigne P., Berggren L., Undén P., Randriamampianina R., Andrea U., Bazile E., Bertelsen A., Brousseau P., Dahlgren P., Edvinsson L., El Said A., Glinton M., Hopsch S., Isaksson L., Mladek R., Olsson E., Verrelle A., Wang Z.Q. (2021). CERRA sub-daily regional reanalysis data for Europe on single levels from 1984 to present [Data set]. ECMWF. https://doi.org/10.24381/CDS.622A565A
    '''
    st.info("Data retrieval script based on code of @orwell2022")
def show_treshold_old(where, to_show, treshold_value, above_under, df):
    """_summary_

    Args:
        where (_type_): _description_
        to_show (_type_): _description_
        treshold_value (_type_): _description_
        above_under (_type_): _description_
        df (_type_): _description_
    """    
    if above_under =="above":
    # Filter the DataFrame to include only the rows where Temperature is above 30 degrees
        df_above_30 = df[df[to_show] >= treshold_value]
        au_txt = ">="
    elif above_under =="equal":
        df_above_30 = df[df[to_show] == treshold_value]
        au_txt = "="
    elif above_under =="below":
        df_above_30 = df[df[to_show] <= treshold_value]
        au_txt = "<="
    else:
        st.error("ERROR")
        st.stop()
    
    # st.subheader(f"Numbers of days per month that {to_show} was {au_txt} {treshold_value} - {where}")
    # # Create a pivot table to count the occurrences of temperatures above 30 degrees per month and year
    # table = pd.pivot_table(df_above_30, values=to_show, index='Month', columns='Year', aggfunc='count', fill_value=0)
    # all_months = range(1, 13)
    # all_years = df['Year'].unique()
    # table = table.reindex(index=all_months, columns=all_years, fill_value=0)
    # st.write(table)
    # Create a pivot table to count the occurrences of temperatures above the threshold per month and year
    # Generate the subheader
    st.subheader(f"Numbers of days per year that {to_show} was {au_txt} {treshold_value} - {where}")

    # aantal hete dagen per jaar
    table_year = pd.pivot_table(
        df_above_30, values=to_show, index='Year', aggfunc='count', fill_value=0
    ).sort_index()

    st.write(table_year)

    # Zorg dat data schoon is
    table_year = table_year.sort_index()
    x = table_year.index.to_numpy()
    y = table_year[to_show].to_numpy()

    # 1) Parametrisch: lineaire regressie
    lin = linregress(x, y)
    slope = lin.slope
    p_lin = lin.pvalue
    r2 = lin.rvalue ** 2

    richting = "stijgend" if slope > 0 else "dalend" if slope < 0 else "vlak"
    significant = p_lin < 0.05

    st.write(
        f"**Lineair** trend: slope={slope:.3f}, R²={r2:.3f}, p={p_lin:.4f} → "
        f"**{richting}** en **{'significant' if significant else 'niet significant'}**"
    )

    # # 95% CI voor de slope met statsmodels
    # import statsmodels.api as sm
    # X = sm.add_constant(x)
    # ols = sm.OLS(y, X).fit()
    # ci_low, ci_high = ols.conf_int(alpha=0.05).iloc[1]
    # st.write(f"95% CI slope [{ci_low:.3f}, {ci_high:.3f}]")

    # 2) Niet-parametrisch: Kendall tau
    tau, p_kendall = kendalltau(x, y)
    richting_np = "stijgend" if tau > 0 else "dalend" if tau < 0 else "vlak"
    significant_np = p_kendall < 0.05
    st.write(
        f"**Kendall tau**: tau={tau:.3f}, p={p_kendall:.4f} → "
        f"**{richting_np}** en **{'significant' if significant_np else 'niet significant'}**"
    )

    # X moet 2D zijn, y 1D
    X = table_year.index.to_numpy().reshape(-1, 1)
    y = table_year[to_show].to_numpy()

    # fit lineair model
    model = LinearRegression()
    model.fit(X, y)
    trendline = model.predict(X)

    # metrics
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r_squared = float(r2_score(y, trendline))

    # plot
    fig_year = go.Figure()
    fig_year.add_trace(go.Bar(x=table_year.index, y=table_year[to_show], name="Number of days"))
    fig_year.add_trace(go.Scatter(x=table_year.index, y=trendline, mode='lines', name="Trendline"))

    fig_year.update_layout(
        title=f'Numbers of days per year that {to_show} was {au_txt} {treshold_value} - {where}',
        xaxis_title='Years',
        yaxis_title=f'Number of days {to_show} was {au_txt} {treshold_value}'
    )
     # Optioneel label in de grafiek
    label = f"trend {richting}, p={p_lin:.3f}"
    fig_year.add_annotation(x=x[-1], y=float(np.max(y)), text=label, showarrow=False)
    st.plotly_chart(fig_year)

    
    # Generate the subheader
    st.subheader(f"Numbers of days per month that {to_show} was {au_txt} {treshold_value} - {where}")

    # Create a pivot table to count the occurrences of temperatures above the threshold per month and year
    table = pd.pivot_table(df_above_30, values=to_show, index='Month', columns='Year', aggfunc='count', fill_value=0)
    all_months = range(1, 13)
    all_years = df['Year'].unique()
    table = table.reindex(index=all_months, columns=all_years, fill_value=0)
    st.write(table)
    
    st.subheader(f"Effect van jaar op het {to_show}")
    df_counts = table.stack().reset_index()
    df_counts.columns = ['Month', 'Year', 'Count']

    X = sm.add_constant(df_counts['Year'])
    y = df_counts['Count']
    model = sm.OLS(y, X).fit()
    st.write(model.summary())
    st.write ("Als p < 0.05 → significant stijgende of dalende trend.")
    tau, p_value = kendalltau(df_counts['Year'], df_counts['Count'])
    if p_value  < 0.05:
        trend = "significant"
    else:
        trend = "not significant"

    st.write(f"If there is no normal distribution : Kendall tau={tau:.3f}, p={p_value:.3f} [{trend}]")

    # Transpose the table for easier plotting
    transposed_table = table.T

    # Initialize a new Plotly figure
    fig_x = go.Figure()

    # Create a scatter plot with a trendline for each month
    for month in transposed_table.columns:

        y_ = transposed_table[month].values
        x_ = transposed_table.index.values
        mask = ~np.isnan(y_)
        slope, intercept, r_value, p_value, std_err = linregress(x_[mask], y_[mask])
        
        if p_value  < 0.05:
            trend = "significant"
        else:
            trend = "not significant"

        st.write(f"Maand {month}: slope={slope:.3f}, R²={r_value**2:.2f}, p={p_value:.3f} [{trend}]")
        
        # Extract the data for the current month and filter out NaN values
        x = transposed_table.index.values
        y = transposed_table[month].values

        mask = ~np.isnan(y)
        x_filtered = x[mask].reshape(-1, 1)
        y_filtered = y[mask]

        # Create scatter plot for the current month
        fig_x.add_trace(go.Scatter(x=x_filtered.flatten(), y=y_filtered, mode='markers', name=f'Month {month} Scatter'))

        # Fit a linear regression model
        if len(y_filtered) > 1:  # Ensure there are enough data points to fit a model
            model = LinearRegression()
            model.fit(x_filtered, y_filtered)
            trendline = model.predict(x_filtered)

            # Calculate the equation and R² value
            slope = model.coef_[0]
            intercept = model.intercept_
            r_squared = r2_score(y_filtered, trendline)

            # Add the trendline to the plot
            fig_x.add_trace(go.Scatter(x=x_filtered.flatten(), y=trendline, mode='lines', name=f'Month {month} Trendline'))

            # Add annotations for the equation and R² value
            # equation_text = f'y = {slope:.2f}x + {intercept:.2f} | R² = {r_squared:.2f}'
            # st.write(f"{month} - {equation_text}")
            
            # fig_x.add_annotation(x=x_filtered.flatten()[-1], y=trendline[-1], text=equation_text, showarrow=True, arrowhead=1)

    # Update layout
    fig_x.update_layout(
        title=f'Numbers of days per month that {to_show} was {au_txt} {treshold_value} - {where}',
        xaxis_title='Years',
        yaxis_title=f'Number of days {to_show} was {au_txt} {treshold_value}'
    )

    # Display the plot using Streamlit
    st.plotly_chart(fig_x)

    st.subheader(f"Numbers of days per month that {to_show} was {au_txt} {treshold_value} - {where}")
    fig = px.imshow(table)
    #fig.show()
    st.plotly_chart(fig)




# ------------------ Helpers: filtering ------------------ #
def filter_by_threshold(df: pd.DataFrame,
                        to_show: str,
                        treshold_value: float,
                        above_under: str) -> tuple[pd.DataFrame, str]:
    if above_under == "above":
        df_thr = df[df[to_show] >= treshold_value]
        au_txt = ">="
    elif above_under == "equal":
        df_thr = df[df[to_show] == treshold_value]
        au_txt = "="
    elif above_under == "below":
        df_thr = df[df[to_show] <= treshold_value]
        au_txt = "<="
    else:
        st.error("above_under must be 'above', 'equal' or 'below'")
        st.stop()

    return df_thr, au_txt


# ------------------ Helpers: statistiek jaar ------------------ #
def analyze_year_trend(table_year: pd.DataFrame, to_show: str) -> dict | None:
    x = table_year.index.to_numpy()
    y = table_year[to_show].to_numpy()

    if len(x) < 2:
        st.write("Te weinig jaren voor trendanalyse.")
        return None

    # lineaire trend
    lin = linregress(x, y)
    slope = lin.slope
    intercept = lin.intercept
    r2 = lin.rvalue ** 2
    p_lin = lin.pvalue

    richting = "stijgend" if slope > 0 else "dalend" if slope < 0 else "vlak"
    significant = p_lin < 0.05

    st.write(
        f"**Lineair** trend: slope={slope:.3f}, R²={r2:.3f}, p={p_lin:.4f} → "
        f"**{richting}** en **{'significant' if significant else 'niet significant'}**"
    )

    # Kendall tau
    tau, p_kendall = kendalltau(x, y)
    richting_np = "stijgend" if tau > 0 else "dalend" if tau < 0 else "vlak"
    significant_np = p_kendall < 0.05

    st.write(
        f"**Kendall tau**: tau={tau:.3f}, p={p_kendall:.4f} → "
        f"**{richting_np}** en **{'significant' if significant_np else 'niet significant'}**"
    )

    return {
        "x": x,
        "y": y,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "p_lin": p_lin,
        "richting": richting,
    }


# ------------------ Helpers: statistiek maand totaal ------------------ #
def analyze_global_month_trend(table: pd.DataFrame, to_show: str) -> pd.DataFrame:
    st.subheader(f"Effect van jaar op het {to_show}")

    df_counts = table.stack().rename("Count").reset_index()  # Month, Year, Count

    X = sm.add_constant(df_counts["Year"])
    y_counts = df_counts["Count"]

    model = sm.OLS(y_counts, X).fit()
    st.write(model.summary())
    st.write("Als p < 0.05 → significant stijgende of dalende trend.")

    tau_all, p_all = kendalltau(df_counts["Year"], df_counts["Count"])
    trend_all = "significant" if p_all < 0.05 else "not significant"
    st.write(f"If there is no normal distribution : Kendall tau={tau_all:.3f}, p={p_all:.3f} [{trend_all}]")

    return df_counts


# ------------------ Helpers: statistiek per maand ------------------ #
def analyze_month_trends(table: pd.DataFrame) -> dict:
    transposed = table.T  # index = Year, columns = Month
    month_stats: dict = {}

    for month in transposed.columns:
        y_m = transposed[month].values
        x_m = transposed.index.values

        mask = ~np.isnan(y_m)
        x_m = x_m[mask]
        y_m = y_m[mask]

        if len(y_m) < 2:
            continue

        lin_m = linregress(x_m, y_m)
        slope_m = lin_m.slope
        intercept_m = lin_m.intercept
        r2_m = lin_m.rvalue ** 2
        p_m = lin_m.pvalue
        trend_m = "significant" if p_m < 0.05 else "not significant"

        st.write(f"Maand {month}: slope={slope_m:.3f}, R²={r2_m:.2f}, p={p_m:.3f} [{trend_m}]")

        month_stats[month] = {
            "x": x_m,
            "y": y_m,
            "slope": slope_m,
            "intercept": intercept_m,
            "r2": r2_m,
            "p": p_m,
        }

    return month_stats


# ------------------ Helpers: plots ------------------ #
def plot_year_trend(table_year: pd.DataFrame,
                    to_show: str,
                    au_txt: str,
                    treshold_value: float,
                    where: str,
                    stats: dict | None) -> None:
    if stats is None:
        return

    x = stats["x"]
    y = stats["y"]
    slope = stats["slope"]
    intercept = stats["intercept"]
    p_lin = stats["p_lin"]
    richting = stats["richting"]

    trendline = slope * x + intercept

    fig_year = go.Figure()
    fig_year.add_trace(
        go.Bar(
            x=table_year.index,
            y=table_year[to_show],
            name="Number of days",
        )
    )
    fig_year.add_trace(
        go.Scatter(
            x=table_year.index,
            y=trendline,
            mode="lines",
            name="Trendline",
        )
    )

    fig_year.update_layout(
        title=f"Numbers of days per year that {to_show} was {au_txt} {treshold_value} - {where}",
        xaxis_title="Years",
        yaxis_title=f"Number of days {to_show} was {au_txt} {treshold_value}",
    )

    label = f"trend {richting}, p={p_lin:.3f}"
    fig_year.add_annotation(
        x=x[-1],
        y=float(np.max(y)),
        text=label,
        showarrow=False,
    )

    st.plotly_chart(fig_year)


def plot_month_trends(to_show: str,
                      au_txt: str,
                      treshold_value: float,
                      where: str,
                      month_stats: dict) -> None:
    fig = go.Figure()

    for month, stats in month_stats.items():
        x = stats["x"]
        y = stats["y"]
        slope = stats["slope"]
        intercept = stats["intercept"]

        trendline = slope * x + intercept

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=f"Month {month} Scatter",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=trendline,
                mode="lines",
                name=f"Month {month} Trendline",
            )
        )

    fig.update_layout(
        title=f"Numbers of days per month that {to_show} was {au_txt} {treshold_value} - {where}",
        xaxis_title="Years",
        yaxis_title=f"Number of days {to_show} was {au_txt} {treshold_value}",
    )
    st.plotly_chart(fig)


def plot_month_heatmap(table: pd.DataFrame,
                       to_show: str,
                       au_txt: str,
                       treshold_value: float,
                       where: str) -> None:
    st.subheader(f"Numbers of days per month that {to_show} was {au_txt} {treshold_value} - {where}")
    fig_heat = px.imshow(table)
    st.plotly_chart(fig_heat)


# ------------------ Hoofdfunctie ------------------ #
def show_treshold(where: str,
                  to_show: str,
                  treshold_value: float,
                  above_under: str,
                  df: pd.DataFrame) -> None:
    """Toon counts en trends voor dagen boven/onder/equal aan een drempel."""
    treshold_value = round(treshold_value,2)
    # Filter
    df_thr, au_txt = filter_by_threshold(df, to_show, treshold_value, above_under)

    # Per jaar
    st.subheader(f"Numbers of days per year that {to_show} was {au_txt} {treshold_value} - {where}")
    table_year = (
        pd.pivot_table(
            df_thr,
            values=to_show,
            index="Year",
            aggfunc="count",
            fill_value=0,
        )
        .sort_index()
    )
    table_year = table_year.rename(columns={"rainsum": "number_days"})

    st.write(table_year)

    year_stats = analyze_year_trend(table_year, to_show)
    plot_year_trend(table_year, to_show, au_txt, treshold_value, where, year_stats)

    # Per maand
    st.subheader(f"Numbers of days per month that {to_show} was {au_txt} {treshold_value} - {where}")
    table = pd.pivot_table(
        df_thr,
        values=to_show,
        index="Month",
        columns="Year",
        aggfunc="count",
        fill_value=0,
    )
    all_months = range(1, 13)
    all_years = df["Year"].unique()
    table = table.reindex(index=all_months, columns=all_years, fill_value=0)
    st.write(table)

    # Statistiek op totaal aantal dagen per jaar over alle maanden
    _ = analyze_global_month_trend(table, to_show)

    # Statistiek per maand en bijbehorende plot
    month_stats = analyze_month_trends(table)
    plot_month_trends(to_show, au_txt, treshold_value, where, month_stats)

    # Heatmap
    plot_month_heatmap(table, to_show, au_txt, treshold_value, where)

def plot_year_value_trend(table_year: pd.DataFrame,
                          to_show: str,
                          where: str,
                          stats: dict | None) -> None:
    if stats is None:
        return

    x = stats["x"]
    y = stats["y"]
    slope = stats["slope"]
    intercept = stats["intercept"]

    trendline = slope * x + intercept

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=table_year.index,
            y=table_year[to_show],
            mode="markers+lines",
            name=f"{to_show}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=table_year.index,
            y=trendline,
            mode="lines",
            name="Trendline",
        )
    )

    fig.update_layout(
        title=f"Yearly trend in {to_show} value - {where}",
        xaxis_title="Year",
        yaxis_title=to_show,
    )

    st.plotly_chart(fig)


def plot_month_value_trends(to_show: str,
                            where: str,
                            month_stats: dict) -> None:
    if not month_stats:
        return

    fig = go.Figure()

    for month, stats in month_stats.items():
        x = stats["x"]
        y = stats["y"]
        slope = stats["slope"]
        intercept = stats["intercept"]
        trendline = slope * x + intercept

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=f"Month {month} values",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=trendline,
                mode="lines",
                name=f"Month {month} trend",
            )
        )

    fig.update_layout(
        title=f"Monthly trend in {to_show} value - {where}",
        xaxis_title="Year",
        yaxis_title=to_show,
    )

    st.plotly_chart(fig)


def show_value_trend(where: str,
                     to_show: str,
                     df: pd.DataFrame) -> None:
    """Trend in de waarde zelf, per jaar en per maand."""

    # Jaarlijke gemiddelde waarde
    st.subheader(f"Yearly {to_show} value - {where}")
    table_year = (
        pd.pivot_table(
            df,
            values=to_show,
            index="Year",
            aggfunc="sum",
        )
        .sort_index()
    )
    
    year_stats = analyze_year_trend(table_year, to_show)
    plot_year_value_trend(table_year, to_show, where, year_stats)
    st.write(table_year)

    # Maandgemiddelde waarde
    st.subheader(f"Monthly {to_show} value - {where}")
    table = pd.pivot_table(
        df,
        values=to_show,
        index="Month",
        columns="Year",
        aggfunc="mean",
    )

    all_months = range(1, 13)
    all_years = df["Year"].unique()
    table = table.reindex(index=all_months, columns=all_years, fill_value=np.nan)
    st.write(table)

    # Hergebruik bestaande statistiek helpers
    _ = analyze_global_month_trend(table, to_show)
    month_stats = analyze_month_trends(table)
    plot_month_value_trends(to_show, where, month_stats)


def line_graph(to_show, window_size, y_axis_zero, df):
    """_summary_

    Args:
        to_show (_type_): _description_
        window_size (_type_): _description_
        y_axis_zero (_type_): _description_
        df (_type_): _description_
    """    
    df[f'{to_show}_SMA'] = df[to_show].rolling(window=window_size).mean()

    fig = px.line(df, x='date', y=[to_show, f'{to_show}_SMA'],
                title=f'{to_show} over Time with SMA',
                labels={'value':to_show},
                line_shape='linear')
    # Set the range of the y-axis to start from 0
    if y_axis_zero:
        fig.update_layout(yaxis_range=[0, max(df[to_show])])

    # fig.show()
    # plotly.offline.plot(fig)
    st.plotly_chart(fig)



# Function to convert Celsius to Fahrenheit
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

# Function to convert Fahrenheit to Celsius
def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9

# Function to calculate Heat Index
def calculate_heat_index(T, RH):
    # Formula for heat index calculation in Fahrenheit
    HI = (-42.379 + 2.04901523 * T + 10.14333127 * RH 
          - 0.22475541 * T * RH - 0.00683783 * T**2 
          - 0.05481717 * RH**2 + 0.00122874 * T**2 * RH 
          + 0.00085282 * T * RH**2 - 0.00000199 * T**2 * RH**2)
    return HI

# Function to calculate Wind Chill
def calculate_wind_chill(T, V):
    # Formula for wind chill calculation in Fahrenheit
    WC = 35.74 + 0.6215 * T - 35.75 * (V**0.16) + 0.4275 * T * (V**0.16)
    return WC

# Function to determine the feels-like temperature
def feels_like_temperature(row):
    T_C = row['temp_mean']
    RH = row['rel_humidity_mean']
    V_mph = float(row['wind_max']) * 0.621371  # Converting km/h to mph
    
    T_F = celsius_to_fahrenheit(T_C)
    
    if T_F >= 80:
        # Calculate Heat Index
        feels_like_F = calculate_heat_index(T_F, RH)
    elif T_F <= 50 and V_mph >= 3:
        # Calculate Wind Chill
        feels_like_F = calculate_wind_chill(T_F, V_mph)
    else:
        feels_like_F = T_F  # No adjustment
    
    feels_like_C = fahrenheit_to_celsius(feels_like_F)

    return feels_like_C


def check_from_until(from_, until_):
    """Checks whether the start- and enddate are valid.

    Args:
        from_ (string): start date
        until_ (string): end date

    Returns:
        FROM, UNTIL : start- and end date in datetime

    """
    
    try:
        FROM = dt.datetime.strptime(from_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the startdate is in format yyyy-mm-dd")
        st.stop()

    try:
        UNTIL = dt.datetime.strptime(until_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the enddate is in format yyyy-mm-dd")
        st.stop()

    if FROM >= UNTIL:
        st.warning("Make sure that the end date is not before the start date")
        st.stop()

    return FROM, UNTIL
def show_locations_2(locations):

    # Convert to DataFrame
    df_map = pd.DataFrame(locations)

    # Plot map
    st.title("🌍 World Locations Map")
    #st.map(df_map.rename(columns={"lat": "latitude", "lon": "longitude"}))

    # MAPBOX = "pk.eyJ1IjoicmNzbWl0IiwiYSI6Ii1IeExqOGcifQ.EB6Xcz9f-ZCzd5eQMwSKLQ"
    # # original_Name
    # df_map = df_map.rename(columns={"name": "original_Name", "lat": "lat", "lon": "lon"})
    # # Ensure the DataFrame has the correct columns for pydeck
    # if "original_Name" not in df_map.columns or "lat" not in df_map.columns or "lon" not in df_map.columns:
    #     st.error("DataFrame must contain 'original_Name', 'lat', and 'lon' columns.")
    #     return
    
    # df_map = df_map[["original_Name", "lat", "lon"]]

    # # Adding code so we can have map default to the center of the data
    # midpoint = (np.average(df_map['lat']), np.average(df_map['lon']))
    # 
    # tooltip = {
    #         "html":
    #             "{original_Name} <br/>"
    #         }
        
    # layer1= pdk.Layer(
    #         'ScatterplotLayer',     # Change the `type` positional argument here
    #             df_map,
    #             get_position=['lon', 'lat'],
    #             auto_highlight=True,
    #             get_radius=4000,          # Radius is given in meters
    #             get_fill_color=[180, 0, 200, 140],  # Set an RGBA value for fill
    #             pickable=True)
    # layer2 =  pdk.Layer(
    #                 type="TextLayer",
    #                 data=df_map,
    #                 pickable=False,
    #                 get_position=["lon", "lat"],
    #                 get_text="original_Name",
    #                 get_color=[0, 0, 0],
    #                 get_angle=0,
    #                 sizeScale= 0.75,
    #                 # Note that string constants in pydeck are explicitly passed as strings
    #                 # This distinguishes them from columns in a data set
    #                 getTextAnchor= '"middle"',
    #                 get_alignment_baseline='"bottom"'
    #             )

    # st.pydeck_chart(pdk.Deck(
    #     map_style='mapbox://styles/mapbox/light-v9',
    #     initial_view_state=pdk.ViewState(
    #          longitude=midpoint[1],
    #         latitude=midpoint[0],
    #         pitch=0,
    #         zoom=3,
    #     ),
    #     layers=[layer1, layer2
            
    #     ],tooltip = tooltip
    # ))

    col1,col2,col3=st.columns(3)
    with col1:
        map_style = st.selectbox("Kaartstijl", ["light", "dark", "road", "satellite", "dark_no_labels","light_no_labels"],5)
    
        angle_txt = st.number_input("Hoek tekst", 0,360,0)
    with col2:
        font_family = st.selectbox("Lettertype", ["serif","sans-serif","monospace","cursive","fantasy","system-ui"],2)
        size_scale = st.number_input("Text grootte", .1,10.,.5,.1)
    with col3:
        text_anchor = st.selectbox(
            "Horizontale uitlijning",
            options=["start", "middle", "end"],
            index=1  # default = "middle"
        )

    
        # Alignment baseline options
        alignment_baseline = st.selectbox(
            "Verticale uitlijning",
            options=["top", "center", "bottom"],
            index=2  # default = "bottom"
        )

    # Filter in Polars
    pdf = pd.DataFrame(locations)

    # Midpoint
    midpoint = (np.average(pdf["lat"]), np.average(pdf["lon"]))

    # Optioneel: Mapbox token
    # pdk.settings.mapbox_api_key = "YOUR_MAPBOX_TOKEN"

    tooltip = {"html": "{name}"}

    layer_points = pdk.Layer(
        "ScatterplotLayer",
        pdf,
        get_position=["lon", "lat"],
        auto_highlight=True,
        get_radius=4000,
        pickable=True,
    )

    layer_text = pdk.Layer(
        "TextLayer",
        pdf,
        pickable=False,
        get_position=["lon", "lat"],
        get_text="name",
     
        get_angle=angle_txt,
        fontFamily = f'"{font_family}"',
        sizeScale=size_scale,
        getTextAnchor=f'"{text_anchor}"',
        get_alignment_baseline=f'"{alignment_baseline}"',
        # wordBreak=word_break,
        # maxWidth=max_width,
    )

    deck = pdk.Deck(
            map_style=map_style,
            initial_view_state=pdk.ViewState(
                longitude=midpoint[1],
                latitude=midpoint[0],
                pitch=0,
                zoom=1,
            ),
            layers=[layer_points, layer_text],
            tooltip=tooltip,
        )
    
    # In Streamlit tonen
    st.pydeck_chart(deck)


  
def main_(df, n_years, locations,FROM, UNTIL, start_month, end_month, where, to_show, window_size, y_axis_zero, multiply_minus_one, treshold_value, above_under, percentile_colomap_max, month_names, month, selected_month, day_min, day_max, number_of_columns):
    """Show the data from OpenMeteo in a graph, and average values per month per year
    """    

    
  

    st.title(f"Weather info from {where}")
 
    line_graph(to_show, window_size, y_axis_zero, df)
    if n_years>11:
        st.info("Too much years to show heatmaps")
    elif n_years>5:

        with st.expander ("Year heatmaps", expanded = False):
            show_calender_heatmap(df,"date", [to_show], where, percentile_colomap_max,number_of_columns)
            show_year_heatmap(df,"date", [to_show])
    else:
        show_calender_heatmap(df,"date", [to_show], where, percentile_colomap_max,number_of_columns)
        show_year_heatmap(df,"date", [to_show])

    cross_table_montly_avg(df, to_show, where, y_axis_zero, selected_month)   
    
    
    
def prepare_dataframe(start_month, end_month, df):
    df['date'] = pd.to_datetime(df['date'])
    df['Day'] = df['date'].dt.day
    df['Month'] = df['date'].dt.month
    df['Year'] = df['date'].dt.year
    df = df.sort_values(by='date') 
    df["date"] = pd.to_datetime(df["date"].astype(str))
    df["YYYY"] = df["date"].dt.year
    df["MM"] = df["date"].dt.month
    df["DD"] = df["date"].dt.day
    n_years = df['Year'].nunique()
    df=df[(df['MM']>=start_month) & (df['MM']<=end_month)]

    # Convert all columns except 'date' to appropriate data types
    df[df.columns.difference(['date'])] = df[df.columns.difference(['date'])].apply(pd.to_numeric, errors='coerce')
    # Apply the feels_like_temperature function to each row in the DataFrame
    df['Feels_Like'] = df.apply(feels_like_temperature, axis=1)
    return df,n_years

def interface(locations):
    location_names = [loc["name"] for loc in locations]
    
    start_ = "2019-01-01"
    today = datetime.today().strftime("%Y-%m-%d")
    from_ = st.sidebar.text_input("startdatum (yyyy-mm-dd) from 1-1-1900", start_)
    until_ = st.sidebar.text_input("enddatum (yyyy-mm-dd)", today)
    FROM, UNTIL = check_from_until(from_, until_)
    start_month, end_month = st.sidebar.slider("Months (incl)", 1,12,(1,12))
    # Convert FROM and UNTIL to datetime
    FROM_ = pd.to_datetime(FROM)
    UNTIL_ = pd.to_datetime(UNTIL)

    where = st.sidebar.selectbox("Location to show", location_names, index=0)
    #to_show = st.sidebar.selectbox("What to show x", ["T_Max","T_Min","T_Mean","Hr_Med","Wind_Max","Wind_Mean","SLP","STN","Vis","Prec","Diary", "Feels_Like"],0)
    to_show = st.sidebar.selectbox("What to show x", ["rain_sum",
        "temp_mean",
        "temp_max",
        "temp_min",
        "rain_sum",
        "precipitation_hours",
        "precipitation_sum",
         "visibility_mean",
        "visibility_min",
        "visibility_max",
        "rel_humidity_mean",
        "rel_humidity_max",
        "rel_humidity_min",
        "wind_max",
        "Feels_Like"], index=0)


    window_size =  st.sidebar.slider("Window for SMA",1,365,7) 
    y_axis_zero = st.sidebar.selectbox("Y axis start at zero", [True,False],1)
    multiply_minus_one = st.sidebar.selectbox("Multiply by -1", [True,False],1)
    treshold_value = st.sidebar.number_input("Treshold value (incl.)",-100.0,100.0,.1)
    above_under = st.sidebar.selectbox("Above or below", ["above", "equal", "below"],0)
    percentile_colomap_max = st.sidebar.number_input("percentile_colomap_max",1,100,100)
    
    month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    month = month_names.index(st.sidebar.selectbox("Month", month_names, index=0)) + 1
    selected_month = month_names[month -1]
    day_min, day_max  = st.sidebar.slider("days",1,31,(1,31))
    number_of_columns = st.sidebar.number_input("Number of columns", 1,10,2)
    return FROM,UNTIL,start_month,end_month,where,to_show,window_size,y_axis_zero,multiply_minus_one,treshold_value,above_under,percentile_colomap_max,month_names,month,selected_month,day_min,day_max,number_of_columns
def legenda():
    # https://open-meteo.com/en/docs/historical-weather-api
    # Define the data for hourly and daily parameters

    st.info("Source: https://open-meteo.com/en/docs/historical-weather-api")
    st.info("Example: https://open-meteo.com/en/docs/historical-weather-api?daily=rain_sum,temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_hours,precipitation_sum,visibility_mean,visibility_min,visibility_max,relative_humidity_2m_mean,relative_humidity_2m_max,relative_humidity_2m_min,&latitude=9.755106899960907&longitude=99.9609068&start_date=2025-01-01&end_date=2025-07-15&hourly=&timezone=Asia%2FBangko")
    hourly_data = [
        ("temperature_2m", "Instant", "°C (°F)", "Air temperature at 2 meters above ground"),
        ("relative_humidity_2m", "Instant", "%", "Relative humidity at 2 meters above ground"),
        ("dew_point_2m", "Instant", "°C (°F)", "Dew point temperature at 2 meters above ground"),
        ("apparent_temperature", "Instant", "°C (°F)", "Perceived feels-like temperature combining wind chill, humidity, and solar radiation"),
        ("surface_pressure", "Instant", "hPa", "Air pressure at surface or mean sea level"),
        ("precipitation", "Preceding hour sum", "mm (inch)", "Total precipitation of the preceding hour"),
        ("rain", "Preceding hour sum", "mm (inch)", "Only liquid precipitation of the preceding hour"),
        ("snowfall", "Preceding hour sum", "cm (inch)", "Snowfall amount of the preceding hour"),
        ("cloud_cover", "Instant", "%", "Total cloud cover as area fraction"),
        ("cloud_cover_low", "Instant", "%", "Low level clouds up to 2 km"),
        ("cloud_cover_mid", "Instant", "%", "Mid level clouds 2–6 km"),
        ("cloud_cover_high", "Instant", "%", "High level clouds from 6 km"),
        ("shortwave_radiation", "Preceding hour mean", "W/m²", "Average shortwave solar radiation"),
        ("direct_normal_irradiance", "Preceding hour mean", "W/m²", "Direct solar radiation on normal plane"),
        ("diffuse_radiation", "Preceding hour mean", "W/m²", "Diffuse solar radiation average"),
        ("global_tilted_irradiance", "Preceding hour mean", "W/m²", "Total radiation on tilted pane"),
        ("sunshine_duration", "Preceding hour sum", "Seconds", "Seconds of sunshine per hour"),
        ("wind_speed_10m", "Instant", "km/h (mph, m/s, knots)", "Wind speed at 10 meters"),
        ("wind_speed_100m", "Instant", "km/h (mph, m/s, knots)", "Wind speed at 100 meters"),
        ("wind_direction_10m", "Instant", "°", "Wind direction at 10 meters"),
        ("wind_direction_100m", "Instant", "°", "Wind direction at 100 meters"),
        ("wind_gusts_10m", "Instant", "km/h (mph, m/s, knots)", "Wind gusts at 10 meters"),
        ("et0_fao_evapotranspiration", "Preceding hour sum", "mm (inch)", "Reference Evapotranspiration FAO-56"),
        ("weather_code", "Instant", "WMO code", "Weather condition as numeric WMO code"),
        ("snow_depth", "Instant", "meters", "Snow depth on the ground"),
        ("vapour_pressure_deficit", "Instant", "kPa", "Vapor Pressure Deficit"),
        ("soil_temperature_0_to_7cm", "Instant", "°C (°F)", "Soil temperature at 0–7 cm depth"),
        ("soil_temperature_7_to_28cm", "Instant", "°C (°F)", "Soil temperature at 7–28 cm depth"),
        ("soil_temperature_28_to_100cm", "Instant", "°C (°F)", "Soil temperature at 28–100 cm depth"),
        ("soil_temperature_100_to_255cm", "Instant", "°C (°F)", "Soil temperature at 100–255 cm depth"),
        ("soil_moisture_0_to_7cm", "Instant", "m³/m³", "Soil moisture at 0–7 cm depth"),
        ("soil_moisture_7_to_28cm", "Instant", "m³/m³", "Soil moisture at 7–28 cm depth"),
        ("soil_moisture_28_to_100cm", "Instant", "m³/m³", "Soil moisture at 28–100 cm depth"),
        ("soil_moisture_100_to_255cm", "Instant", "m³/m³", "Soil moisture at 100–255 cm depth")
    ]

    daily_data = [
        ("weather_code", "", "WMO code", "Most severe weather condition on a given day"),
        ("temperature_2m_max", "", "°C (°F)", "Maximum daily air temperature at 2 meters"),
        ("temperature_2m_min", "", "°C (°F)", "Minimum daily air temperature at 2 meters"),
        ("apparent_temperature_max", "", "°C (°F)", "Maximum daily apparent temperature"),
        ("apparent_temperature_min", "", "°C (°F)", "Minimum daily apparent temperature"),
        ("precipitation_sum", "", "mm", "Sum of daily precipitation"),
        ("rain_sum", "", "mm", "Sum of daily rain"),
        ("snowfall_sum", "", "cm", "Sum of daily snowfall"),
        ("precipitation_hours", "", "hours", "Number of hours with rain"),
        ("sunrise", "", "iso8601", "Sunrise time"),
        ("sunset", "", "iso8601", "Sunset time"),
        ("sunshine_duration", "", "seconds", "Total sunshine duration per day"),
        ("daylight_duration", "", "seconds", "Total daylight duration per day"),
        ("wind_speed_10m_max", "", "km/h (mph, m/s, knots)", "Max wind speed on a day"),
        ("wind_gusts_10m_max", "", "km/h (mph, m/s, knots)", "Max wind gusts on a day"),
        ("wind_direction_10m_dominant", "", "°", "Dominant wind direction"),
        ("shortwave_radiation_sum", "", "MJ/m²", "Daily solar radiation sum"),
        ("et0_fao_evapotranspiration", "", "mm", "Daily evapotranspiration (ET₀)")
    ]

    # Create DataFrames
    #df_hourly = pd.DataFrame(hourly_data, columns=["Variable", "Valid time", "Unit", "Description"])
    df_daily = pd.DataFrame(daily_data, columns=["Variable", "Valid time", "Unit", "Description"])

    st.dataframe(df_daily)

def location_dashboard(df, df_hourly, where, day_min, day_max, month, month_names, treshold_value):

   
    df["rain_day"] = (df["rain_sum"] >= treshold_value).astype(int)
    df["dry_day"] = (df["rain_sum"] < treshold_value).astype(int)
    values_to_show = ["temp_max","dry_day","rain_day","rain_sum","precipitation_hours"]
    
    for month in range (1,13):
        st.subheader(f"Location dashboard for {where} - {month_names[month-1]}")  
        columns = st.columns(len(values_to_show))  
        for i,to_show in enumerate(values_to_show):
            with columns[i]:
                show_month(df, to_show, day_min, day_max,month, month_names,where, False)
def main():
    locations = [
        {"name": "Koh Phangan", "lat": 9.755106899960907, "lon": 99.9609068, "timezone": "Asia/Bangkok"},
        {"name": "Koh Chang", "lat":  12.10361, "lon": 102.35194, "timezone": "Asia/Bangkok"},
        {"name": "Chiang Mai", "lat": 18.7931784, "lon": 98.9774429, "timezone": "Asia/Bangkok"},
        {"name": "Amsterdam", "lat": 52.3676, "lon": 4.9041, "timezone": "Europe/Amsterdam"},
        {"name": "Lisbon", "lat": 38.7169, "lon": -9.1399, "timezone": "Europe/Lisbon"},
        {"name": "Rome", "lat": 41.9102088, "lon": 12.371185, "timezone": "Europe/Rome"},
        {"name": "Venezia", "lat": 45.4408, "lon": 12.3155, "timezone": "Europe/Rome"},
        {"name": "Hoi An", "lat": 15.8801, "lon": 108.3380, "timezone": "Asia/Ho_Chi_Minh"},
        {"name": "Da Nang", "lat": 16.047079, "lon":108.206230, "timezone": "Asia/Ho_Chi_Minh"},
        {"name": "Ho Chi Minh City", "lat": 10.7769, "lon": 106.7009, "timezone": "Asia/Ho_Chi_Minh"},
        {"name": "Hanoi", "lat": 21.0285, "lon": 105.8542, "timezone": "Asia/Bangkok"},
        {"name": "Manila", "lat": 14.5995, "lon": 120.9842, "timezone": "Asia/Manila"},
        {"name": "Taipei", "lat": 25.0330, "lon": 121.5654, "timezone": "Asia/Taipei"},
        {"name": "Kathmandu", "lat": 27.7172, "lon": 85.3240, "timezone": "Asia/Kathmandu"},
        {"name": "Colombo", "lat": 6.9271, "lon": 79.8612, "timezone": "Asia/Colombo"},
        {"name": "London", "lat": 51.5072, "lon": -0.1276, "timezone": "Europe/London"},
        {"name": "New York", "lat": 40.7128, "lon": -74.0060, "timezone": "America/New_York"},
        {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503, "timezone": "Asia/Tokyo"},
        {"name": "Sydney", "lat": -33.8688, "lon": 151.2093, "timezone": "Australia/Sydney"},
        {"name": "Cape Town", "lat": -33.9249, "lon": 18.4241, "timezone": "Africa/Johannesburg"},
        {"name": "São Paulo", "lat": -23.5505, "lon": -46.6333, "timezone": "America/Sao_Paulo"},
        {"name": "Istanbul", "lat": 41.0082, "lon": 28.9784, "timezone": "Europe/Istanbul"},
    ]
    FROM, UNTIL, start_month, end_month, where, to_show, window_size, y_axis_zero, multiply_minus_one, treshold_value, above_under, percentile_colomap_max, month_names, month, selected_month, day_min, day_max, number_of_columns = interface(locations)
    
    df_,df_hourly = get_data(where,locations, FROM, UNTIL)
   
    if multiply_minus_one:
        # needed for for ex. visability 
        # Make a copy of the DataFrame without the "date" column
        df_copy = df_.drop(columns=['date']).copy()

        # Multiply values by -1
        # df_copy = df_copy * -1
        df_[to_show] = -df_[to_show]
        # Combine the "date" column back with the modified values
        df = pd.concat([df_['date'], df_copy], axis=1)
    else:
        df = df_
    df, n_years = prepare_dataframe(start_month, end_month, df)

  
    tab1,tab2, tab3,tab4,tab5,tab6,tab7 = st.tabs(["Data",  "Specific month","Trends","Treshold","Location dashboard","Locations","Source and Legenda"])
    
    with tab6:
        show_locations_2(locations)
    with tab7:
        show_info()
        legenda()
    with tab1:
        main_(df, n_years, locations,FROM, UNTIL, start_month, end_month, where, to_show, window_size, y_axis_zero, multiply_minus_one, treshold_value, above_under, percentile_colomap_max, month_names, month, selected_month, day_min, day_max, number_of_columns )
    with tab3:
    
        show_value_trend(where, to_show,df)
    with tab5:
        location_dashboard(df, df_hourly, where, day_min, day_max, month, month_names,treshold_value)
    with tab4:
        show_treshold(where, to_show, treshold_value, above_under, df)
    with tab2:
        #show_warmingstripes(df, to_show, where) 
        if month >= start_month and month <= end_month:
            show_month(df, to_show, day_min, day_max,month, month_names,where)
        else:
            st.info("Selected month is not in the indicated range for months to show")
    
if __name__ == "__main__":
    #read_ogimet()
    main()
    