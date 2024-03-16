# REPRODUCING https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8881926/

import pandas as pd
import streamlit as st
import numpy as np

import plotly.express as px

from scipy.stats import linregress
import statsmodels.api as sm
from scipy import stats

def prepare_data():

    """Function to compare the country names in the various files. URL_COUNTRY is leading.
    """

    url_meat = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\meat_consumption.csv"
    url_gm = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\gapminder_data_graphs.csv"
    url_health = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\health_efficiency_index.csv"
    url_education_mean = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\mean-years-of-schooling-long-run.csv"
    url_education_expected = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\expected-years-of-schooling.csv"
    url_country = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\country_codes.csv"
    url_country_wikipedia = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\country_wikipedia.csv"
    df =     pd.read_csv(url_education_expected, delimiter=',')
    df_country =  pd.read_csv(url_country, delimiter=',')
    
    df_country = df_country.fillna(0)
    

    for v in ["population","area"]:
        df_country[v] = df_country[v].str.replace('.', '').astype(int)

    df_country["gdpppp"] =df_country["gdp"] * df_country["gdp_unit"] /  df_country["population"]
    st.write(df_country)

    check_field = "country"
    # Find countries in df but not in df_country
    countries_in_df_not_in_df_country = df[~df[check_field].isin(df_country[check_field])]

    # Find countries in df_country but not in df
    countries_in_df_country_not_in_df = df_country[~df_country[check_field].isin(df[check_field])]
    col1,col2= st.columns(2)
    # Display the differences
    with col1:
        st.write("Countries in df_country but not in df_target")
        st.write(countries_in_df_country_not_in_df)
        st.write(len(countries_in_df_country_not_in_df))
    with col2:
        st.write("Countries in df_target but not in df_country - CHANGE")
        
        st.write(countries_in_df_not_in_df_country)
        st.write(len(countries_in_df_not_in_df_country))

def get_data(join_how):
    """Gets the data and merges the dataframes

    Args:
        join_how (str): "inner" or "outer"

    Returns:
        df: the complete dataframe
    """    
    #url_country_wikipedia = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\country_wikipedia.csv"
    url_country = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/country_codes.csv"
    url_meat = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/meat_consumption.csv"
    url_gm = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/gapminder_data_graphs.csv"
    url_health =  "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/health_efficiency_index.csv"
    url_education_mean = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/mean-years-of-schooling-long-run.csv"
    url_education_expected = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/expected-years-of-schooling.csv"
    url_length = 'https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/length_male.csv'
   
    # url_country = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\country_codes.csv"
    # url_meat = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\meat_consumption.csv"
    # url_gm = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\gapminder_data_graphs.csv"
    # url_health = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\health_efficiency_index.csv"
    # url_education_mean = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\mean-years-of-schooling-long-run.csv"
    # url_education_expected = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\expected-years-of-schooling.csv"
    
    df_country =  pd.read_csv(url_country, delimiter=',')
    df_country = df_country.fillna(0)
    for v in ["population","area"]:
        df_country[v] = df_country[v].str.replace('.', '').astype(int)
    df_country["gdpppp"] =df_country["gdp"] * df_country["gdp_unit"] /  df_country["population"]
    

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
   
    # https://ourworldindata.org/grapher/mean-years-of-schooling-long-run
    # https://ourworldindata.org/grapher/expected-years-of-schooling-vs-share-in-extreme-poverty
   
    df_education_mean =  pd.read_csv(url_education_mean, delimiter=',')
    df_education_mean = df_education_mean[df_education_mean["year"] == 2020]
    df_education_expected =  pd.read_csv(url_education_expected, delimiter=',')
  
    df_education_expected = df_education_expected[df_education_expected["year"] == 2020]
 
    df_education_expected = df_education_expected.dropna(subset=['iso_3'])
    df_education_mean = df_education_mean.dropna(subset=['iso_3'])
   
    df_length = pd.read_csv(url_length, delimiter=',')
    merged_df_0 = pd.merge(df_meat, df_country, how="outer", on='country')
    merged_df_1 = merged_df_0.merge(df_gm_2018, how=join_how, on='country')
    merged_df_2 = merged_df_1.merge(df_health,  how=join_how, on="country")
    merged_df_3 = merged_df_2.merge(df_education_mean, on="iso_3", how=join_how)
    merged_df_4 = merged_df_3.merge(df_length, on="iso_3", how=join_how)
    
    df = merged_df_4.merge(df_education_expected, on="iso_3", how=join_how)
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
def m(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)

def cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)
def wcorr(x, y, w):
    """Weighted Correlation https://stackoverflow.com/questions/38641691/weighted-correlation-coefficient-with-pandas"""
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))

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
    df = df_.dropna(subset=[x,y,"population"])
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
    
    weighted_correlation_coefficient = wcorr(df[x], df[y], df["population"])

    title_ = f"{x} vs {y} [n = {len(df)}]"
    r_sq_corr = f'R2 = {r_squared:.2f} / Corr coeff = {correlation_coefficient:.2f} |  W R2 = {weighted_correlation_coefficient*weighted_correlation_coefficient:.2f} / W Corr coeff = {weighted_correlation_coefficient:.2f}'
    if trendline_per_continent:
        fig = px.scatter(df, x=x, y=y,  hover_data=['country'],trendline='ols',  color='continent', title=f'')
    else:
        fig = px.scatter(df, x=x, y=y,  hover_data=['country','population'],  size='population',color='continent', title=f'{title_} || {r_sq_corr}')
        fig.add_trace(px.line(x=df[x], y=slope * df[x] + intercept, line_shape='linear').data[0])

    # Show the plot
    st.plotly_chart(fig)
    # health eff index vs life exp. Weighted geeft andere waarde dan normaal, tenzij je Azie verwijderd. India en China zijn wellicht grote outliers
def correlation_matrix(df,show_log_x, show_log_y):
    """Generates and shows a correlation matrix and a heatmap

    Args:
        df (_type_): _description_
    """
    columns_meat =   ["meat_cons","life_exp_birth","life_exp_5","mort_under_5","cal_day","gdpppp _2011","urban_pop","bmi_over_30","cho_crops","prim_educ_over_25"]
    columns_health = ["health_eff_index_rank" ,"health_eff_index"]
    columns_gm = ["life_exp","hdi_index","co2_consump","gdp_y","services"]
    columns_educ = ["schooling_mean", "schooling_expected","education_index"]
    columns = columns_meat + columns_health + columns_gm +columns_educ

    df_corr = df[columns].copy(deep=True)
    if show_log_x and show_log_y:
        for c in columns: 
            df_corr[c] = np.log(df_corr[c])
            x = "(Log transformed variables)"
    else:
        x = ""
    
    # Calculate Pearson's correlation matrix
    pearson_corr = df_corr.corr(method='pearson')

    # Calculate Spearman's rank-order correlation matrix
    spearman_corr = df_corr.corr(method='spearman')

    st.write(f"Correlation matrix {x}")
    st.write("Pearson's above the diagonal and Spearman's rank below")
    combined_corr_matrix = pd.DataFrame(index=df_corr.columns, columns=df_corr.columns)

    # Fill the upper triangle with Pearson's and lower triangle with Spearman's
    for i in range(len(df_corr.columns)):
        for j in range(i, len(df_corr.columns)):
            if i == j:
                combined_corr_matrix.iloc[i, j] = 1.0  # Diagonal elements are 1
            else:
                combined_corr_matrix.iloc[i, j] = pearson_corr.iloc[i, j]
                combined_corr_matrix.iloc[j, i] = spearman_corr.iloc[j, i]

    st.write(combined_corr_matrix)
    #In this code, we manually create an empty DataFrame combined_corr_matrix with column and index labels. Then, we loop through the upper triangle of the matrix and fill in the values based on Pearson's and Spearman's correlation matrices. This approach ensures that both column names and values are retained correctly in the final combined correlation matrix.

    fig = px.imshow(combined_corr_matrix.abs()) 
    st.subheader(f"Correlation heatmap (absolute values) {x}")
    
    st.plotly_chart(fig)

def multiple_lineair_regression(df, show_log_x, show_log_y):
    """Calculates multiple lineair regression. User can choose the Y value and the X values

    Args:
        df_ (df): df with info
    """    
    st.subheader("Multiple Lineair Regression")
    y_value_ = st.selectbox("Y value", ['life_exp',"life_exp_birth","life_exp_5","mort_under_5"],1)
    x_values_options =  ["meat_cons","cal_day","gdpppp _2011","urban_pop","bmi_over_30","cho_crops","prim_educ_over_25","health_eff_index_rank" ,"health_eff_index","hdi_index","co2_consump","gdp_y","services", 'education_index', 'schooling_mean', 'schooling_expected']
    x_values_default = ['meat_cons',"cal_day","gdpppp _2011","urban_pop","bmi_over_30","cho_crops", 'health_eff_index','education_index']
    x_values = st.multiselect("X values", x_values_options, x_values_default)
    standard=  st.sidebar.checkbox("Standardizing dataframe", True)
    intercept=  st.sidebar.checkbox("Intercept", False)
    only_complete = st.sidebar.checkbox("Only complete rows", False)
    if only_complete:
        df=df.dropna()
    else:
        df = df.dropna(subset=x_values)
        df = df.dropna(subset=y_value_)
    df =df[["country","population"]+[y_value_]+ x_values]
   
    if show_log_x:
        if 'health_eff_index' in x_values:
            df['health_eff_index'] = df['health_eff_index'] * 100 # not possible to make z score from neg values -> log 0.1 gives -1
            df = df[df['country'] != 'Sierra Leone'] # has a value of 0, ln(0) is indefinite
        for c in x_values:
            df[c] = np.log(df[c])
       
    if show_log_y:
        df[y_value_] = np.log(df[y_value_])
    if standard:
        # https://stackoverflow.com/questions/50842397/how-to-get-standardised-beta-coefficients-for-multiple-linear-regression-using
        #df = df.select_dtypes(include=[np.number]).dropna().apply(stats.zscore)
        #df = df[x_values_default].dropna().apply(stats.zscore)
        # numeric_columns = df.select_dtypes(include=[np.number]).drop(columns=['population'])
        
        # # Apply Z-score normalization to the selected columns
        # df = numeric_columns.apply(stats.zscore)

                
        # Select numeric columns for Z-score normalization
        numeric_columns = df.select_dtypes(include=[np.number])
    
        # Exclude 'country' and 'population' from Z-score normalization
        columns_to_exclude = ['population']
        numeric_columns = numeric_columns.drop(columns=columns_to_exclude)
  
        # Apply Z-score normalization to the selected columns
        z_scored_df = numeric_columns.apply(stats.zscore)
   
        # Add 'country' and 'population' columns back to the Z-scored DataFrame
        df_standardized = pd.concat([df[['country', 'population']], z_scored_df], axis=1)
    st.write("**DATA**")
    st.write(df_standardized)
    st.write(f"Length : {len(df_standardized)}")
    x = df_standardized[x_values]
    y = df_standardized[y_value_]
  
    w = df_standardized["population"]
    
    # with statsmodels
    if intercept:
        x= sm.add_constant(x) # adding a constant
    
    model = sm.OLS(y, x).fit()
    #predictions = model.predict(x) 
    st.write("**OUTPUT ORDINARY LEAST SQUARES**")
    print_model = model.summary()
    st.write(print_model)
    df_standardized = df_standardized.dropna(subset="population")
    st.write("**OUTPUT WEIGHTED LEAST SQUARES (weightfactor = population)**")
    wls_model = sm.WLS(y,x, weights=w).fit()
    print_wls_model = wls_model.summary()
    st.write(print_wls_model)

def multiple_lineair_regression_sklearn(df_, show_log_x, show_log_y):
    """Calculates multiple lineair regression. User can choose the Y value and the X values

    Args:
        df_ (df): df with info
    """    
    st.subheader("Multiple Lineair Regression")
    y_value = st.selectbox("Y value", ['life_exp',"life_exp_birth","life_exp_5","mort_under_5"],1)
    x_values_options =  ["meat_cons","cal_day","gdpppp _2011","urban_pop","bmi_over_30","cho_crops","prim_educ_over_25","health_eff_index_rank" ,"health_eff_index","hdi_index","co2_consump","gdp_y","services", 'education_index', 'schooling_mean', 'schooling_expected']
    x_values_default = ['meat_cons',"cal_day","gdpppp _2011","urban_pop","bmi_over_30","cho_crops", 'health_eff_index','education_index']
    x_values = st.multiselect("X values", x_values_options, x_values_default)
    
   
    df = df_.dropna(subset=x_values)
    df = df.dropna(subset=y_value)
    df =df[["country","population"]+[y_value]+ x_values]
    if show_log_x:
        for c in x_values:
            df[c] = np.log(df[c])
       
    if show_log_y:
        df[y_value] = np.log(df[y_value])

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
    st.write("Male height: https://www.researchgate.net/publication/295394901_Major_correlates_of_male_height_A_study_of_105_countries")
    st.info("Country codes and info: https://countrycode.org/")
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
    show_log_x = st.sidebar.checkbox('Show logarithmic values X', True)
    show_log_y= st.sidebar.checkbox('Show logarithmic values Y', True)
    trendline_per_continent= st.sidebar.checkbox('Trendline per continent')
    return df,x,y,show_log_x,show_log_y,trendline_per_continent #,join_how

   
def main():
    """Main function
    """    
    st.header("Meat consumption vs life expectancy")
    st.info("REPRODUCING https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8881926/")
    st.info("https://rcsmit.medium.com/longer-living-due-to-eating-meat-or-something-else-7225f0055c1f")
    join_how = "outer"

    df = get_data(join_how)
    df["continent"].fillna("UNKNOWN", inplace=True)
    df = df.dropna(subset="iso_2")
    df, what_x, what_y, show_log_x, show_log_y, trendline_per_continent = interface(df)
  
    make_scatterplot(df, what_x, what_y, show_log_x,show_log_y,trendline_per_continent)
    correlation_matrix(df,show_log_x, show_log_y)
    multiple_lineair_regression(df, show_log_x, show_log_y)
    #multiple_lineair_regression_sklearn(df, show_log_x, show_log_y)
    show_footer()

if __name__ == "__main__":
    main()
    #prepare_data()