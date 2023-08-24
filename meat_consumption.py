# REPRODUCING https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8881926/

import pandas as pd
import streamlit as st
import numpy as np

import plotly.express as px

from scipy.stats import linregress

# from sklearn import linear_model

import statsmodels.api as sm


def get_data(join_how):
    """Gets the data and merges the dataframes

    Args:
        join_how (str): "inner" or "outer"

    Returns:
        df: the complete dataframe
    """    
    url_meat = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/meat_consumption.csv"
    url_gm = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/gapminder_data_graphs.csv"
    url_health =  "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/health_efficiency_index.csv"
    url_education_mean = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/mean-years-of-schooling-long-run.csv"
    url_education_expected = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/expected-years-of-schooling.csv"
    
    # url_meat = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\meat_consumption.csv"
    # url_gm = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\gapminder_data_graphs.csv"
    # url_health = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\health_efficiency_index.csv"
    # url_education_mean = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\mean-years-of-schooling-long-run.csv"
    # url_education_expected = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\expected-years-of-schooling.csv"
    
    df_meat =  pd.read_csv(url_meat, delimiter=',')

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
    
    df_gm = pd.read_csv(url_gm, delimiter=',')
    df_gm_2018 = df_gm[df_gm["year"] == 2018]
    
    # https://web.archive.org/web/20200313135813/https://www.who.int/healthinfo/paper30.pdf

    df_health = pd.read_csv(url_health, delimiter=',')
    df_health = df_health[["health_eff_index_rank" ,"country","health_eff_index"]]
   
    df_education_mean =  pd.read_csv(url_education_mean, delimiter=',')
    df_education_mean = df_education_mean[df_education_mean["year"] == 2020]
    df_education_expected =  pd.read_csv(url_education_expected, delimiter=',')
  
    df_education_expected = df_education_expected[df_education_expected["year"] == 2020]
    
    merged_df_1 = pd.merge(df_meat, df_gm_2018, how=join_how, on='country')
    merged_df_2 = merged_df_1.merge(df_health, on="country", how=join_how)
    merged_df_3 = merged_df_2.merge(df_education_mean, on="country", how=join_how)
    df = merged_df_3.merge(df_education_expected, on="country", how=join_how)
    df["education_index"] = ((df["schooling_expected"] / 18) + (df["schooling_mean"] /15) )/2

    # # Find countries in df but not in df_gm
    # countries_in_df_not_in_df_gm = df[~df['country'].isin(df_health['country'])]

    # # Find countries in df_gm but not in df
    # countries_in_df_gm_not_in_df = df_health[~df_health['country'].isin(df['country'])]

    # # Display the differences
    # st.write("Countries in df but not in df_health:")
    # st.write(countries_in_df_not_in_df_gm)

    # st.write("Countries in df_health but not in df:")
    st.subheader("DATA")
    st.write(df)
    st.write(f"Lengte {len(df)}")
    return df

def make_scatterplot(df_, x, y, show_log_x,show_log_y,trendline_per_continent):
    """Makes a scatterplot

    Args:
        df_ (df): the dataframe
        x (str): column used for x axis
        y (str): column used for y axis
        show_log_x (bool): _description_
        show_log_y (bool): _description_
        trendline_per_continent (bool): Do we want a trendline per continent
    """    
    st.subheader("Scatterplot")
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
    title_ = f"{x} vs {y} [{len(df)} countries]"
    r_sq_corr = f'R-squared = {r_squared:.2f} / Correlation coeff = {correlation_coefficient:.2f}'
    if trendline_per_continent:
        fig = px.scatter(df, x=x, y=y,  hover_data=['country'],trendline='ols',  color='continent', title=f'')
    else:
        fig = px.scatter(df, x=x, y=y,  hover_data=['country'],  color='continent', title=f'{title_} || {r_sq_corr}')
        fig.add_trace(px.line(x=df[x], y=slope * df[x] + intercept, line_shape='linear').data[0])

    # Show the plot
    st.plotly_chart(fig)
    
def correlation_matrix(df):
    """Generates and shows a correlation matrix and a heatmap

    Args:
        df (_type_): _description_
    """
    columns_meat =   ["meat_cons","life_exp_birth","life_exp_5","mort_under_5","cal_day","gdpppp _2011","urban_pop","bmi_over_30","cho_crops","prim_educ_over_25"]
    columns_health = ["health_eff_index_rank" ,"health_eff_index"]
    columns_gm = ["life_exp","hdi_index","co2_consump","gdp","services"]
    columns_educ = ["schooling_mean", "schooling_expected","education_index"]
    columns = columns_meat + columns_health + columns_gm +columns_educ

    df = df[columns]
    corrMatrix = df.corr()
    st.subheader("Correlation matrix")
    st.write(corrMatrix)
    fig = px.imshow(corrMatrix.abs()) 
    st.subheader("Correlation heatmap (absolute values)")
    
    st.plotly_chart(fig)


def multiple_lineair_regression(df_):
    """Calculates multiple lineair regression. User can choose the Y value and the X values

    Args:
        df_ (df): df with info
    """    
    st.subheader("Multiple Lineair Regression")
    y_value = st.selectbox("Y value", ['life_exp',"life_exp_birth","life_exp_5","mort_under_5"],0)
    x_values_options =  ["meat_cons","cal_day","gdpppp _2011","urban_pop","bmi_over_30","cho_crops","prim_educ_over_25","health_eff_index_rank" ,"health_eff_index","hdi_index","co2_consump","gdp","services", 'education_index', 'schooling_mean', 'schooling_expected']
    x_values_default = ['meat_cons','health_eff_index',"urban_pop","bmi_over_30","cho_crops","prim_educ_over_25", 'education_index']
    x_values = st.multiselect("X values", x_values_options, x_values_default)
    
    df = df_.dropna(subset=x_values)
    df = df.dropna(subset=y_value)
    df =df[["country"]+[y_value]+ x_values]
    st.write("**DATA**")
    st.write(df)
    st.write(f"Length : {len(df)}")
    x = df[x_values]
    y = df[y_value]
    
    # with statsmodels
    x = sm.add_constant(x) # adding a constant
    
    model = sm.OLS(y, x).fit()
    #predictions = model.predict(x) 
    st.write("**OUTPUT**")
    print_model = model.summary()
    st.write(print_model)
    
def main():
    """Main function
    """    
    join_how = "outer"
    df = get_data(join_how)
    df["continent"].fillna("UNKNOWN", inplace=True)
    df, x,y, show_log_x, show_log_y, trendline_per_continent = interface(df)
    
    make_scatterplot(df, x, y, show_log_x,show_log_y,trendline_per_continent)
    correlation_matrix(df)
    multiple_lineair_regression(df)
    show_footer()

def show_footer():
    """Shows the footer
    """    
    st.subheader("Data Sources")
    st.info("Meat consumption etc. : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8881926/ (appendix 1)")
    st.write("* newborn life expectancy (e(0)), life expectancy at 5 years of life (e(5)) and intakes of meat, and carbohydrate crops, respectively. The established risk factors to life expectancy – caloric intake, urbanization, obesity and education levels – were included as the potential confounders.")
    
    st.info("Gapminder data set, values from 2018 : https://www.kaggle.com/datasets/albertovidalrod/gapminder-dataset?resource=download")
    st.write("* Country (country). Describes the country name")
    st.write("* Continent (continent). Describes the continent to which the country belongs")
    st.write("* Year (year). Describes the year to which the data belongs")
    st.write("* Life expectancy (life_exp). Describes the life expectancy for a given country in a given year")
    st.write("* Human Development Index (hdi_index). Describes the HDI index value for a given country in a given year")
    st.write("* --- The Human Development Index (HDI) is a statistical composite index of life expectancy, education (mean years of schooling completed and expected years of schooling upon entering the education system), and per capita income indicators, which is used to rank countries into four tiers of human development. ")
    st.write("* CO2 emissions per person(co2_consump). Describes the CO2 emissions in tonnes per person for a given country in a given year")
    st.write("* Gross Domestic Product per capita (gdp). Describes the GDP per capita in dollars for a given country in a given year")
    st.write("* % Service workers (services). Describes the the % of service workers for a given country in a given year")
    st.info("Health Efficiency Index : https://web.archive.org/web/20200313135813/https://www.who.int/healthinfo/paper30.pdf")
    st.write("* Factors : 25% for health (DALE), 25% for health inequality, 12.5% for the level of responsiveness, 12.5% for the distribution of responsiveness, and 25% for fairness in financing.")
    st.info("Education: https://ourworldindata.org/human-development-index")
    st.write("* Mean years of schooling estimates the average number of years of total schooling adults aged 25 years and older have received. This data extends back to the year 1870 and is based on the combination of data from Lee and Lee (2016); Barro-Lee (2018); and the UN Development Programme. Fifteen is the projected maximum of this indicator for 2025.")
    st.write("* Expected years of schooling measures the number of years of schooling that a child of school entrance age can expect to receive if the current age-specific enrollment rates persist throughout the child’s life by country. Eighteen is equivalent to achieving a master's degree in most countries.")
    st.write("* education_index = ((schooling_expected / 18) + (schooling_mean / 15) )/2 " )
    st.subheader("Discussion")
    st.write("* Data is from different years")
    st.write("* Data is not weighted (The Netherlands is counting as much as India or China)")
    
def interface(df):
    """Makes the interface

    Args:
        df (df): dataframe

    Returns:
        
        df,x,y,show_log_x,show_log_y,trendline_per_continent : the various values
    """    
    x = st.sidebar.selectbox('Select a column X', df.columns[1:], 0)
    y = st.sidebar.selectbox('Select a column Y', df.columns[1:], 1)
    continents = df["continent"].unique()
    all_countries = st.sidebar.checkbox('All countries')
    if all_countries == False:  
        selected_continents = st.sidebar.multiselect('Select continents', list(continents),list(continents))
        df = df[df['continent'].isin(selected_continents)]
    # Create a checkbox to determine whether to display log values
    #join_how = st.sidebar.selectbox("How to join", ["inner", "outer"],0)
    show_log_x = st.sidebar.checkbox('Show logarithmic values X')
    show_log_y= st.sidebar.checkbox('Show logarithmic values Y')
    trendline_per_continent= st.sidebar.checkbox('Trendline per continent')
    return df,x,y,show_log_x,show_log_y,trendline_per_continent #,join_how

if __name__ == "__main__":
    main()