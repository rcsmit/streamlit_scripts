import pandas as pd
import streamlit as st
import numpy as np

import plotly.express as px

from scipy.stats import linregress


# REPRODUCING https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8881926/

def get_data():
    #url = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\meat_consumption.csv"
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/meat_consumption.csv"
    df =  pd.read_csv(url, delimiter=',')


    # gapminder dataset 
    # https://www.kaggle.com/datasets/albertovidalrod/gapminder-dataset?resource=download
    # Country (country). Describes the country name
    # Continent (continent). Describes the continent to which the country belongs
    # Year (year). Describes the year to which the data belongs
    # Life expectancy (life_exp). Describes the life expectancy for a given country in a given year
    # Human Development Index (hdi_index). Describes the HDI index value for a given country in a given year
    # CO2 emissions per person(co2_consump). Describes the CO2 emissions in tonnes per person for a given country in a given year
    # Gross Domestic Product per capita (gdp). Describes the GDP per capita in dollars for a given country in a given year
    # % Service workers (services). Describes the the % of service workers for a given country in a given year
    
    #url_gm = r"C:\Users\rcxsm\AppData\Local\Temp\Temp73593f11-a6e0-472f-84a0-1505a4bd8c70_archive.zip\gapminder_data_graphs.csv"
    url_gm = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/gapminder_data_graphs.csv"
    df_gm_2018 = pd.read_csv(url_gm, delimiter=',')
    df_gm_2018 = df_gm[df_gm["year"] == 2018]
    
    merged_df = pd.merge(df, df_gm_2018, how='outer', on='country')


    # # Find countries in df but not in df_gm
    # countries_in_df_not_in_df_gm = df[~df['country'].isin(df_gm_2018['country'])]

    # # Find countries in df_gm but not in df
    # countries_in_df_gm_not_in_df = df_gm_2018[~df_gm_2018['country'].isin(df['country'])]

    # # Display the differences
    # st.write("Countries in df but not in df_gm:")
    # st.write(countries_in_df_not_in_df_gm)

    # st.write("Countries in df_gm but not in df:")
    # st.write(countries_in_df_gm_not_in_df)
    return merged_df

def make_scatterplot(df_, x, y, show_log_x,show_log_y,trendline_per_continent):
    df = df_.dropna(subset=[x,y])
    # Create a DataFrame (df) using your data
    # Calculate linear regression

    
    # Calculate log values if selected
    if show_log_x:
        df['log_' + x] = np.log(df[x])
        x = 'log_' + x

    if show_log_y:
        df['log_' + y] = np.log(df[y])
        y = 'log_' + y

    slope, intercept, r_value, p_value, std_err = linregress(df[x], df[y])
    r_squared = r_value ** 2
    # Calculate correlation coefficient
    correlation_coefficient = np.corrcoef(df[x], df[y])[0, 1]

    r_sq_corr = f'R-squared = {r_squared:.2f} / Correlation coeff = {correlation_coefficient:.2f}'
    if trendline_per_continent:
        fig = px.scatter(df, x=x, y=y,  hover_data=['country'],trendline='ols',  color='continent', title=f'{x} vs {y}')
    else:
        fig = px.scatter(df, x=x, y=y,  hover_data=['country'],  color='continent', title=f'{x} vs {y} || {r_sq_corr}')
        fig.add_trace(px.line(x=df[x], y=slope * df[x] + intercept, line_shape='linear').data[0])

    # Show the plot
    st.plotly_chart(fig)
    

def main():
    df = get_data()
   
    x = st.sidebar.selectbox('Select a column X', df.columns[1:], 0)
    y = st.sidebar.selectbox('Select a column Y', df.columns[1:], 0)
    
    df["continent"].fillna("UNKNOWN", inplace=True)
    df = df.fillna(0)
   
    # Get unique continents from the DataFrame
    continents = df["continent"].unique()
    all_countries = st.sidebar.checkbox('All countries')
    if all_countries == False:  
        selected_continents = st.sidebar.multiselect('Select continents', list(continents),list(continents))
        df = df[df['continent'].isin(selected_continents)]
    # Create a checkbox to determine whether to display log values
    show_log_x = st.sidebar.checkbox('Show logarithmic values X')
    show_log_y= st.sidebar.checkbox('Show logarithmic values Y')
    trendline_per_continent= st.sidebar.checkbox('Trendline per continent')
    make_scatterplot(df, x, y, show_log_x,show_log_y,trendline_per_continent)
    st.info("Meat consumption etc. : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8881926/ (appendix 1)\n            Gapminder data set, values from 2018 : https://www.kaggle.com/datasets/albertovidalrod/gapminder-dataset?resource=download")

main()