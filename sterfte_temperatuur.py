from datetime import datetime
import math
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.stats import linregress, pearsonr
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from patsy import dmatrices
import numpy as np
import random
from sterfte_temperatuur_orwell import main_orwell
from sterfte_temperatuur_orwell_esp2013 import main_orwell_esp2013
# st.set_page_config(layout="wide")
# WAT IS DE INVLOED VAN DE TEMPERATUUR OP DE STERFTE
#
# https://medium.com/@marc.jacobs012/oversterfte-en-temperatuur-238a54881493
def get_weather_info():
    """_summary_

    Returns:
        _type_: _description_
    """
    current_datetime = datetime.now()
    formatted_date = current_datetime.strftime("%Y%m%d")

    url_nw_beerta = f"https://www.daggegevens.knmi.nl/klimatologie/daggegevens?stns=260&vars=TEMP:SQ:SP:Q:DR:RH:UN:UX&start=20000101&end={formatted_date}"
    #url_nw_beerta = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/nw_beerta.csv"
    df_nw_beerta = pd.read_csv(
        url_nw_beerta,
        delimiter=",",
        header=None,
        comment="#",
        low_memory=False,
    )
    column_replacements_knmi = [
        [0, "STN"],
        [1, "YYYYMMDD"],
        [2, "temp_avg"],
        [3, "temp_min"],
        [4, "temp_max"],
        [5, "T10N"],
        [6, "zonneschijnduur"],
        [7, "perc_max_zonneschijnduur"],
        [8, "glob_straling"],
        [9, "neerslag_duur"],
        [10, "neerslag_etmaalsom"],
        [11, "RH_min"],
        [12, "RH_max"],
    ]
    column_replacements = column_replacements_knmi
    for c in column_replacements:
        df_nw_beerta = df_nw_beerta.rename(columns={c[0]: c[1]})
    to_divide_by_10 = [
        "temp_avg",
        "temp_min",
        "temp_max",
        "zonneschijnduur",
        "neerslag_duur",
        "neerslag_etmaalsom",
    ]

    # divide_by_10 = False if platform.processor() else True
    divide_by_10 = True
    if divide_by_10:
        for d in to_divide_by_10:
            try:
                df_nw_beerta[d] = df_nw_beerta[d] / 10
            except:
                df_nw_beerta[d] = df_nw_beerta[d]
    df_nw_beerta["YYYYMMDD"] = pd.to_datetime(df_nw_beerta["YYYYMMDD"].astype(str))
    df_nw_beerta["week_number"] = df_nw_beerta["YYYYMMDD"].dt.isocalendar().week
    df_nw_beerta["year_number"] = df_nw_beerta["YYYYMMDD"].dt.isocalendar().year

    result = (
        df_nw_beerta.groupby(["week_number", "year_number"])
        .agg(
            {
                "temp_avg": "mean",
                "temp_min": "mean",
                "temp_max": "mean",
                "T10N": "mean",
                "zonneschijnduur": "mean",
                "perc_max_zonneschijnduur": "mean",
                "glob_straling": "mean",
                "neerslag_duur": "mean",
                "neerslag_etmaalsom": "mean",
                "RH_min": "mean",
                "RH_max": "mean",
            }
        )
        .reset_index()
    )
    # Display the result
    result = result[["year_number", "week_number", "temp_min", "temp_max", "temp_avg"]]

    df_week3_10 = result[(result["week_number"] >= 3) & (result["week_number"] <= 10)]
    mean_3_10 = result["temp_avg"].mean()
    st.write(f"Gmiddelde temperatuur week 3-10 : {round(mean_3_10,1)}")
    return result

@st.cache_data()
def get_sterfte(per_100k, age_group="TOTAL_T"):
    """_summary_
    Arguments:
        per_100k (bool): if True, calculate per 100k inhabitants
        age_group (list): list with age groups, e.g. ['TOTAL_T', 'Y_LT5_T', 'Y_GE90_T']

    Returns:
        _type_: _description_
    """
    # Data from https://ec.europa.eu/eurostat/databrowser/product/view/demo_r_mwk_05?lang=en
    # https://ec.europa.eu/eurostat/databrowser/bookmark/fbd80cd8-7b96-4ad9-98be-1358dd80f191?lang=en
    # https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/dataflow/ESTAT/DEMO_R_MWK_05/1.0?references=descendants&detail=referencepartial&format=sdmx_2.1_generic&compressed=true
    file = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/sterfte_eurostats_weekly__2000_01__2023_41.csv"
    # file = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\sterfte_eurostats_weekly__2000_01__2025_14.csv"
    file = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/sterfte_eurostats_weekly__2000_01__2025_14.csv"
    
    #file = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/demo_r_mwk_05/1.0/*.*.*.*.*?c[freq]=W&c[age]=TOTAL,Y_LT5,Y5-9,Y10-14,Y15-19,Y20-24,Y25-29,Y30-34,Y35-39,Y40-44,Y45-49,Y50-54,Y55-59,Y60-64,Y65-69,Y70-74,Y75-79,Y80-84,Y85-89,Y_GE90,UNK&c[sex]=T,M,F&c[unit]=NR&c[geo]=NL&c[TIME_PERIOD]=ge:2000-W01&compress=false&format=csvdata&formatVersion=1.0&lang=en&labels=both"
    df_ = pd.read_csv(
        file,
        delimiter=",",
        low_memory=False,
    )
    file_bevolking = "https://raw.githubusercontent.com/rcsmit/COVIDcases/refs/heads/main/input/bevolking_leeftijd_NL.csv"
    # file_bevolking = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\bevolking_leeftijd_NL.csv"
    df_bevolking = pd.read_csv(
        file_bevolking,
        delimiter=";",
        low_memory=False,
    )

        
    # Prepare population data
    df_bevolking = df_bevolking.rename(columns={"leeftijd": "age", "geslacht": "sex", "jaar": "year_number", "aantal": "population"})

    df_bevolking["sex"] = df_bevolking["sex"].map({"F": "F", "M": "M", "T": "T"})
    #df_bevolking["age"] = df_bevolking["age"].astype(str)
    # Convert age to numeric
    
    df_bevolking["age"] = pd.to_numeric(df_bevolking["age"], errors="coerce")
    
    # Define age binning
    bins = [-1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
            50, 55, 60, 65, 70, 75, 80, 85, 90, 120]
    labels = [
        "Y_LT5", "Y5-9", "Y10-14", "Y15-19", "Y20-24", "Y25-29", "Y30-34",
        "Y35-39", "Y40-44", "Y45-49", "Y50-54", "Y55-59", "Y60-64",
        "Y65-69", "Y70-74", "Y75-79", "Y80-84", "Y85-89", "Y_GE90"
    ]

    # Assign Eurostat group
    df_bevolking["age_group"] = pd.cut(df_bevolking["age"], bins=bins, labels=labels, right=False)
    df_bevolking["age_sex"] = df_bevolking["age_group"].astype(str) + "_" + df_bevolking["sex"]
    # Sum population by Eurostat group, sex, and year
    df_bevolking_grouped = (
        df_bevolking
        .groupby(["year_number", "sex", "age_group"], as_index=False)
        .agg({"population": "sum"})
    )


    # Create age_sex column to match df_["age_sex"]
    df_bevolking_grouped["age_sex"] = df_bevolking_grouped["age_group"].astype(str) + "_" + df_bevolking_grouped["sex"]

    # Create TOTAL rows by summing over all age groups per year and sex
    total_rows = []

    for sex in ["T", "F", "M"]:
        df_sex = df_bevolking[df_bevolking["sex"] == sex]
        total = (
            df_sex.groupby("year_number", as_index=False)
            .agg({"population": "sum"})
            .assign(age_group="TOTAL", sex=sex)
        )
        total_rows.append(total)

    # Combine all TOTAL_* rows
    df_total = pd.concat(total_rows, ignore_index=True)
    df_total["age_sex"] = df_total["age_group"] + "_" + df_total["sex"]

    # Merge TOTAL rows with main grouped dataframe
    df_bevolking_grouped = pd.concat([df_bevolking_grouped, df_total], ignore_index=True)

    
    for col in ['freq', 'age', 'sex', 'unit', 'geo']:
        df_[[col, f'{col}_desc']] = df_[col].str.split(':', n=1, expand=True)
 
    df_["age_sex"] = df_["age"] + "_" + df_["sex"]
    df_ = df_[df_["sex"] == "T"]
    

    
    # st.write(df_["age_sex"].unique())
    df_["year_number"] = (df_["TIME_PERIOD"].str[:4]).astype(int)
    df_["week_number"] = (df_["TIME_PERIOD"].str[6:]).astype(int)
    df_["year_week"] = (
        df_["year_number"].astype(str).str.zfill(2)
        + "_"
        + df_["week_number"].astype(str).str.zfill(2)
    )
   
    df_ = df_[["year_number", "year_week", "week_number", "OBS_VALUE", "age_sex"]]
    df_ = df_[df_["age_sex"].isin(age_group)]
   
    # Merge population with mortality
    df_ = df_.groupby(["year_week", "year_number", "week_number", "age_sex"], as_index=False)["OBS_VALUE"].sum()
    df_merged = pd.merge(df_, df_bevolking_grouped, how="left", on=["year_number", "age_sex"])
    
    # Calculate OBS per 100k
    if per_100k:
        df_merged["OBS_VALUE"] = round(df_merged["OBS_VALUE"] / df_merged["population"] * 100_000,3)

    
    # We hebben ook de gemiddelde temperatuur tussen week 3 en 10 berekend (te weten 10,85 graad) en de daarbijbehorende verwachte sterfte genomen vanuit grafiek 2 (3042) en deze op 100 gesteld
    # We hebben de gemiddelde sterfte genomen van alle weken tussen week 3 tot en met en 10 (te weten 3055 personen) en dit op 100 gesteld
    df_merged["OBS_sma"] = df_["OBS_VALUE"].rolling(window=5).mean()
    df_merged_week3_10 = df_merged[(df_["week_number"] >= 3) & (df_merged["week_number"] <= 10)]

    mean_week3_10 = df_merged_week3_10["OBS_VALUE"].mean()
    st.write(f"Gemiddelde sterfte week 3-10 : {int(mean_week3_10)}")
    df_merged["obs_mean_3_10"] = round(df_merged["OBS_VALUE"] / mean_week3_10 * 100, 1)
    st.write(df_merged)
    return df_merged

def make_scatter(x, y, df, title):
    """make a scatter plot, and show correlation and the equation
    Args:
        x (str): x values-field
        y (str): y values-field
        merged_df (str): df
    returns:
        slope, 
        intercept
    """
    st.subheader(title)
    fig_ = px.scatter(
        df,
        x=x,
        y=y,
        hover_data=["year_number", "week_number"],
        color="year_number",
        title=title,
        trendline="ols",
        trendline_scope="overall",
        
    )  #  trendline_scope="overall", labels={'datum': 'Date', 'verbruik': 'Verbruik'})
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color="year_number",
        hover_data=["year_number", "week_number"],
        title=title,
    )  #  trendline_scope="overall", labels={'datum': 'Date', 'verbruik': 'Verbruik'})
    st.plotly_chart(fig_, use_container_width=True, key=f"scatter_plot_{random.randint(0,1000000)}")  # Plotly scatter plot with trendline
    # Calculate the correlation
    try:
        correlation = df[x].corr(df[y])

        # Perform linear regression to get the equation of the line
        slope, intercept, r_value, p_value, std_err = linregress(df[x], df[y])

        # Print the correlation and equation of the line
        st.write(f"Correlation: {correlation:.2f}")
        st.write(f"Equation of the line: y = {slope:.2f} * x + {intercept:.2f}")
    except:
        pass
    return slope, intercept


def make_line(x, y, df, title):
    """make a scatter plot, and show correlation and the equation

    Args:
        x (str): x values-field
        y (str): y values-field
        merged_df (str): df
    """
    st.subheader(title)
    fig = px.line(
        df,
        x=x,
        y=y,
        hover_data=["year_number", "week_number"],
        color="year_number",
        title=title,
    )  # ,  trendline='ols')#  trendline_scope="overall", labels={'datum': 'Date', 'verbruik': 'Verbruik'})
    st.plotly_chart(fig, use_container_width=True)  # Calculate the correlation

def multiple_lineair_regression(df_, x_values, y_value, afkap):
    """Calculates multiple lineair regression. User can choose the Y value and the X values


        A t-statistic with an absolute value greater than 2 suggests that the coefficient is
        statistically significant.

        The p-value associated with each coefficient tests the null hypothesis that the coefficient 
        is zero (i.e., it has no effect). A small p-value (typically less than 0.05) suggests 
        that the coefficient         is statistically significant.

        The F-statistic tests the overall significance of the regression model.
        A small p-value for the F-statistic indicates that at least one independent variable
        has a statistically significant effect on the dependent variable.

    Args:
        df_ (df): df with info
        x_values (list): list with the x values
        y_value (str): the result variable
        afkap(float) : afkapwaarde
    """

    df = df_.dropna(subset=x_values)
    df = df.dropna(subset=y_value)
    x = df[x_values]
    y = df[y_value]
    # with statsmodels
    x = sm.add_constant(x)  # adding a constant
    model = sm.OLS(y, x).fit()
    st.write(f"**OUTPUT ORDINARY LEAST SQUARES** afkapwaarde = {afkap}")
    print_model = model.summary()
    st.write(print_model)


def decomposed(merged_df):

    """https://chat.openai.com/c/95cc75cf-a3f6-45aa-97db-57958bf38b42"""

    # Assuming df_temp and df_deaths are your temperature and mortality DataFrames
    # and 'common_time_column' is the common column representing time
    # Seasonal decomposition for temperature

    st.subheader("Seasonal decomposition for temperature")
    merged_df["common_time_column"] = pd.to_datetime(
        merged_df["year_week"] + "-0", format="%Y_%W-%w"
    )

    merged_df.set_index("common_time_column", inplace=True)
    merged_df = merged_df.fillna(0)
    decomposition = seasonal_decompose(
        merged_df["temp_avg"], model="additive", period=52
    )
    seasonal_component = decomposition.seasonal

    # Correlation analysis with seasonally adjusted data
    merged_df["temp_avg_adj"] = merged_df["temp_avg"] - seasonal_component
    correlation_coefficient, p_value = pearsonr(
        merged_df["temp_avg_adj"], merged_df["OBS_VALUE"]
    )
   
    make_scatter("temp_avg_adj", "OBS_VALUE", merged_df, "all")
    st.write(f"Correlation Coefficient: {correlation_coefficient:.2f}")
    st.write(f"P-value: {p_value:.4f}")


def poisson_regression(df,afkap):
    """_summary_

    Args:
        df (_type_): _description_
        afkap(float): afkapwaarde

    """
    # used in Huynen, 2001.  The Impact of Heat Waves and Cold Spells on Mortality Rates
    #  in the Dutch Population [uses Poisson loglinear regression analyses]
    # https://timeseriesreasoning.com/contents/poisson-regression-model/
    # https://education.illinois.edu/docs/default-source/carolyn-anderson/edpsy589/lectures/4_glm/4glm_3_beamer_post.pdf

    #  Regression coefficients for individual lag periods were transformed by using
    #  the formula 100 * (exp beta - 1), to the percent change
    # in mortality associated with a 1°C increase in the average value of cold or heat within
    # the respective lag period ("percent effect").
    # Summing the transformed regression coefficients over individual lag periods yields an
    # estimate of the percent change in mortality associated with a 1°C increase in the average
    # value of cold or heat during the last month ("aggregate effect").
    # Kunst 1993 - https://sci-hub.se/https://doi.org/10.1093/oxfordjournals.aje.a116680

    st.subheader("Poisson regression")
    df["log_OBS_VALUE"] = np.log(df["OBS_VALUE"])

    optimum_value = afkap  # Replace with your actual optimum value

    # Create "heat" column
    df["heat"] = df["temp_avg"].apply(lambda x: max(0, x - optimum_value))

    # Create "cold" column
    df["cold"] = df["temp_avg"].apply(lambda x: max(0, optimum_value - x))

    # expr = """OBS_VALUE ~ week_number +  temp_avg + heat + cold"""
    # expr = """OBS_VALUE ~  heat + cold"""
    expr = """OBS_VALUE ~  temp_avg"""

    # Set up the X and y matrices
    y_train, x_train = dmatrices(expr, df, return_type="dataframe")

    # Using the statsmodels GLM class, train the Poisson regression model on the training data set.
    poisson_training_results = sm.GLM(
        y_train, x_train, family=sm.families.Poisson()
    ).fit()

    # Print the training summary.
    st.write(poisson_training_results.summary())
    coef = poisson_training_results.params["temp_avg"]
    change = (math.exp(coef) - 1) * 100
    st.write(f"For each change of 1 degrees, the mortality changes {round(change,1)}%")


def main_rcsmit():
    """_summary_"""
    st.header("Invloed van temperatuur op sterfte")
    st.info("https://rcsmit.medium.com/sterfte-vs-temperatuur-b65770af76d3")
    ages= ['TOTAL_T', 'UNK_T', 'Y10-14_T', 'Y15-19_T', 'Y20-24_T', 'Y25-29_T', 'Y30-34_T','Y35-39_T', 'Y40-44_T', 'Y45-49_T', 'Y5-9_T', 'Y50-54_T', 'Y55-59_T','Y60-64_T', 'Y65-69_T', 'Y70-74_T', 'Y75-79_T', 'Y80-84_T', 'Y85-89_T', 'Y_GE90_T', 'Y_LT5_T']
  
    age_group = st.multiselect("Select ages", ages, ["TOTAL_T"], key="age_group")
    if not age_group:
        st.error("Select agegroup(s)")
        st.stop()
    what = st.selectbox("what", ["temp_min", "temp_avg", "temp_max"],1)

    (min_year,max_year) = st.slider("Select year range (incl.)", 2000, 2025, (2015, 2020), key="year_range")
    afkap = round(st.number_input("afkapwaarde temp_avg", 16.5),2) # https://www.researchgate.net/publication/11937466_The_Impact_of_Heat_Waves_and_Cold_Spells_on_Mortality_Rates_in_the_Dutch_Population
    lag = st.number_input("lag in weeks", 1, 10, 1, 1) # lag in weeks
    per_100k = st.checkbox("per 100k", True, key="per_100k")
    if per_100k:
        title_100k = "/ [per 100k]"
    else:
        title_100k=""
    if min_year > max_year:
        st.error("Error in year range")
        st.stop()
    df_sterfte = get_sterfte(per_100k, age_group)
    df_temperature = get_weather_info()
    df = df_sterfte.merge(df_temperature, on=["year_number", "week_number"])
    df = df[(df["year_number"] >=min_year) & (df["year_number"] <= max_year)]

    # calculate the average for each week for all 20 years and merge it
    df_avg_week = df_sterfte.groupby("week_number", as_index=False).mean(numeric_only=True)

    df_avg_week.rename(columns={"OBS_VALUE": "OBS_MEAN"}, inplace=True)
    df_avg_week = df_avg_week[["week_number", "OBS_MEAN"]]
    df = df.merge(df_avg_week, on="week_number", how="outer")

    df = df.sort_values(by="year_week")
    # df=df[df["year_number"] == 2003]

    # Create a new column 'AVG_OBS_VALUE' with the average of OBS_VALUE in row n and row n-1
    df["AVG_OBS_VALUE"] = (df["OBS_VALUE"] + df["OBS_VALUE"].shift(1)) / 2
    
    df[f"OBS_VALUE_lag_{lag}_week"] = df["OBS_VALUE"].shift(lag)

    # Fill the NaN value in the last row of 'AVG_OBS_VALUE' with the corresponding value
    # from 'OBS_VALUE'
    df["AVG_OBS_VALUE"].iloc[-1] = df["OBS_VALUE"].iloc[-1]
    df["AVG_OBS_VALUE_YEAR"] = df.groupby("year_number")["OBS_VALUE"].transform("mean")

    # Create a new column 'DISTANCE_FROM_AVG' with the distance between OBS_VALUE and
    # the average for each year
    df["DISTANCE_FROM_AVG"] = df["OBS_VALUE"] - df["AVG_OBS_VALUE_YEAR"]
    col1, col2 = st.columns(2)
    with col1:
        slope_max,intercept_max = make_scatter("temp_avg", "temp_max", df, "temp_max vs temp_avg")
    with col2:
        slope_min,intercept_min = make_scatter("temp_avg", "temp_min", df, "temp_min vs temp_avg")
  
    
    if what=="temp_avg":
        afkap = afkap 
    elif what=="temp_min":
        afkap =  round(slope_min * 16.5,1) + intercept_min # based on Linear regression equation for the correlation between  temp_avg - temp_min
    
    elif what=="temp_max":
        afkap = round(slope_max * 16.5 + intercept_max,1)  #  [ 1.14 * temp_avg + 2.49] Linear regression equation for the correlation between temp_avg and temp_max
    else:
        st.error("Error in [what]")
        st.stop()
    df_lower = df[df[what] <= afkap].copy(deep=True)
    df_higher = df[df[what] > afkap]
    st.write(df)
    make_scatter(what, "OBS_VALUE", df, f"Deaths vs temperature {title_100k}")
    col1, col2 = st.columns(2)
    with col1:
        slope_x, intercept_x = make_scatter(what, "OBS_VALUE", df_lower, f"{what} <= {afkap} {title_100k}")

        with st.expander("Lagged values"):
            _,_ = make_scatter(what, "AVG_OBS_VALUE", df_lower, f"{what} > {afkap} / lag 3 days {title_100k}")
            _,_ = make_scatter(
                what,
                f"OBS_VALUE_lag_{lag}_week",
                df_lower,
                f"{what} <= {afkap} / lag {lag} week {title_100k}",
            )
        _,_ = make_scatter(
            what,
            "DISTANCE_FROM_AVG",
            df_lower,
            f"{what} <= {afkap} / distance from yearavg {title_100k}",
        )
        value = int(slope_x * afkap + intercept_x)
        df_lower[f"obs_{value}"] = round(df_lower["OBS_VALUE"] / value * 100, 1)
  
        _,_ = make_scatter(what, f"obs_{value}", df_lower, f"{what} <= {afkap} /  {value} = 100 {title_100k}")
        _,_ = make_scatter(
            what,
            "obs_mean_3_10",
            df_lower,
            f"{what} <= {afkap} / mean temp(week 3-10) = 100 {title_100k}",
        )

    with col2:
        slope_y, intercept_y = make_scatter(what, "OBS_VALUE", df_higher, f"{what} > {afkap} {title_100k}")
        with st.expander("Lagged values"):
            

            _,_ = make_scatter(what, "AVG_OBS_VALUE", df_higher, f"{what} > {afkap} / lag 3 days {title_100k}")
            _,_ = make_scatter(
                what,
                f"OBS_VALUE_lag_{lag}_week",
                df_higher,
                f"{what} > {afkap} / lag {lag} week {title_100k}",
            )
        _,_ = make_scatter(
            what,
            "DISTANCE_FROM_AVG",
            df_higher,
            f"{what} > {afkap} / distance from yearavg {title_100k}"
        )
        value = int(slope_y * afkap + intercept_y)
        df_higher[f"obs_{value}"] = round(df_higher["OBS_VALUE"] / value * 100, 1)
  
        _,_ = make_scatter(what, f"obs_{value}", df_higher, f"{what} > {afkap} /  {value} = 100 {title_100k}")
  
        _,_ = make_scatter(
            what,
            "obs_mean_3_10",
            df_higher,
            f"{what} > {afkap} / mean temp(week 3-10) = 100 {title_100k}",
        )
    # decomposed(df) #gives errors

    col1, col2 = st.columns(2)
    with col1:
        make_line("year_week", what, df, "Temperatuur in de tijd")
    with col2:
        make_line("year_week", "OBS_VALUE", df, f"Sterfte in de tijd {title_100k}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"t<={afkap} {title_100k}")
        multiple_lineair_regression(df_lower, ["temp_avg"], "OBS_VALUE", afkap)
        poisson_regression(df_lower, afkap)
    with col2:
        st.subheader(f"t>{afkap} {title_100k}")
        multiple_lineair_regression(df_higher, ["temp_avg"], "OBS_VALUE",afkap)
        poisson_regression(df_higher, afkap)
def main():

        
    # if 'active_tab' not in st.session_state:
    #     st.session_state.active_tab = "Tab 1"

    # with st.sidebar:
    #     if st.session_state.active_tab == "Tab 1":
    #         st.write("Sidebar content for Tab 1")
    #         # Add any widgets or elements specific to Tab 1
    #     elif st.session_state.active_tab == "Tab 2":
    #         st.write("Sidebar content for Tab 2")
    #         # Add any widgets or elements specific to Tab 2
    #     else:
    #         st.write("Default sidebar content")

    # tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])

    # with tab1:
    #     st.session_state.active_tab = "Tab 1"
    #     st.write("Content of Tab 1")

    # with tab2:
    #     st.session_state.active_tab = "Tab 2"
    #     st.write("Content of Tab 2")
    tab1, tab2, tab3 = st.tabs(["rcsmit", "orwell", "orwell_esp2013"])

    with tab1:
        main_rcsmit()
    with tab2:
        main_orwell()
    with tab3:
        main_orwell_esp2013()
if __name__ == "__main__":
    main()
