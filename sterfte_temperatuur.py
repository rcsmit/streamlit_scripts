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
    # url_nw_beerta = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/nw_beerta.csv"
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


def get_sterfte():
    """_summary_

    Returns:
        _type_: _description_
    """
    # Data from https://ec.europa.eu/eurostat/databrowser/product/view/demo_r_mwk_05?lang=en
    # https://ec.europa.eu/eurostat/databrowser/bookmark/fbd80cd8-7b96-4ad9-98be-1358dd80f191?lang=en
    # https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/dataflow/ESTAT/DEMO_R_MWK_05/1.0?references=descendants&detail=referencepartial&format=sdmx_2.1_generic&compressed=true
    file = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/sterfte_eurostats_weekly__2000_01__2023_41.csv"

    df_ = pd.read_csv(
        file,
        delimiter=",",
        low_memory=False,
    )

    df_["age_sex"] = df_["age"] + "_" + df_["sex"]
    df_ = df_[df_["age_sex"] == "TOTAL_T"]
    # st.write(df_["age_sex"].unique())
    df_["year_number"] = (df_["TIME_PERIOD"].str[:4]).astype(int)
    df_["week_number"] = (df_["TIME_PERIOD"].str[6:]).astype(int)
    df_["year_week"] = (
        df_["year_number"].astype(str).str.zfill(2)
        + "_"
        + df_["week_number"].astype(str).str.zfill(2)
    )
    df_ = df_[["year_number", "year_week", "week_number", "OBS_VALUE"]]
    df_["obs_2461"] = round(df_["OBS_VALUE"] / 2461 * 100, 1)
    df_["obs_3042"] = round(df_["OBS_VALUE"] / 3042 * 100, 1)
    df_["OBS_sma"] = df_["OBS_VALUE"].rolling(window=5).mean()
    df_week3_10 = df_[(df_["week_number"] >= 3) & (df_["week_number"] <= 10)]

    mean_week3_10 = df_week3_10["OBS_VALUE"].mean()
    st.write(f"Gemiddelde sterfte week 3-10 : {int(mean_week3_10)}")
    df_["obs_mean_3_10"] = round(df_["OBS_VALUE"] / mean_week3_10 * 100, 1)
    return df_

def make_scatter(x, y, df, title):
    """make a scatter plot, and show correlation and the equation
    Args:
        x (str): x values-field
        y (str): y values-field
        merged_df (str): df
    """
    st.subheader(title)
    fig = px.scatter(
        df,
        x=x,
        y=y,
        hover_data=["year_number", "week_number"],
        color="year_number",
        title=title,
        trendline="ols",
    )  #  trendline_scope="overall", labels={'datum': 'Date', 'verbruik': 'Verbruik'})
    st.plotly_chart(fig, use_container_width=True)
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

def multiple_lineair_regression(df_, x_values, y_value):
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
    """

    df = df_.dropna(subset=x_values)
    df = df.dropna(subset=y_value)
    x = df[x_values]
    y = df[y_value]
    # with statsmodels
    x = sm.add_constant(x)  # adding a constant
    model = sm.OLS(y, x).fit()
    st.write("**OUTPUT ORDINARY LEAST SQUARES**")
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


def poisson_regression(df):
    """_summary_

    Args:
        df (_type_): _description_
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

    optimum_value = 16.5  # Replace with your actual optimum value

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


def main():
    """_summary_"""
    st.header("Invloed van temperatuur op sterfte")
    st.info("https://rcsmit.medium.com/sterfte-vs-temperatuur-b65770af76d3")
    df_sterfte = get_sterfte()
    df_temperature = get_weather_info()
    df = df_sterfte.merge(df_temperature, on=["year_number", "week_number"])
    df = df[df["year_number"] < 2020]

    # calculate the average for each week for all 20 years and merge it
    df_avg_week = df_sterfte.groupby("week_number").mean().reset_index()
    df_avg_week.rename(columns={"OBS_VALUE": "OBS_MEAN"}, inplace=True)
    df_avg_week = df_avg_week[["week_number", "OBS_MEAN"]]
    df = df.merge(df_avg_week, on="week_number", how="outer")

    df = df.sort_values(by="year_week")
    # df=df[df["year_number"] == 2003]

    # Create a new column 'AVG_OBS_VALUE' with the average of OBS_VALUE in row n and row n-1
    df["AVG_OBS_VALUE"] = (df["OBS_VALUE"] + df["OBS_VALUE"].shift(1)) / 2
    lag = 1
    df[f"OBS_VALUE_lag_{lag}_week"] = df["OBS_VALUE"].shift(lag)

    # Fill the NaN value in the last row of 'AVG_OBS_VALUE' with the corresponding value
    # from 'OBS_VALUE'
    df["AVG_OBS_VALUE"].iloc[-1] = df["OBS_VALUE"].iloc[-1]
    df["AVG_OBS_VALUE_YEAR"] = df.groupby("year_number")["OBS_VALUE"].transform("mean")

    # Create a new column 'DISTANCE_FROM_AVG' with the distance between OBS_VALUE and
    # the average for each year
    df["DISTANCE_FROM_AVG"] = df["OBS_VALUE"] - df["AVG_OBS_VALUE_YEAR"]

    st.write(df)
    afkap = 16.5
    what = "temp_avg"
    df_lower = df[df[what] <= afkap].copy(deep=True)
    df_higher = df[df[what] > afkap]

    make_scatter(what, "OBS_VALUE", df, "all")
    col1, col2 = st.columns(2)
    with col1:
        make_scatter(what, "OBS_VALUE", df_lower, f"{what} <= {afkap}")
        make_scatter(what, "AVG_OBS_VALUE", df_lower, f"{what} > {afkap} / lag 3 days")
        make_scatter(
            what,
            f"OBS_VALUE_lag_{lag}_week",
            df_lower,
            f"{what} <= {afkap} / lag {lag} week",
        )
        make_scatter(
            what,
            "DISTANCE_FROM_AVG",
            df_lower,
            f"{what} <= {afkap} / distance from yearavg",
        )
        make_scatter(what, "obs_3042", df_lower, f"{what} <= {afkap} /  3042 = 100")
        make_scatter(
            what,
            "obs_mean_3_10",
            df_lower,
            f"{what} <= {afkap} / mean temp(week 3-10) = 100",
        )

    with col2:
        make_scatter(what, "OBS_VALUE", df_higher, f"{what} > {afkap}")
        make_scatter(what, "AVG_OBS_VALUE", df_higher, f"{what} > {afkap} / lag 3 days")
        make_scatter(
            what,
            f"OBS_VALUE_lag_{lag}_week",
            df_higher,
            f"{what} > {afkap} / lag {lag} week",
        )
        make_scatter(
            what,
            "DISTANCE_FROM_AVG",
            df_higher,
            f"{what} > {afkap} / distance from yearavg",
        )
        make_scatter(what, "obs_2461", df_higher, f"{what} > {afkap} / 2461=100")
    decomposed(df)

    col1, col2 = st.columns(2)
    with col1:
        make_line("year_week", what, df, "Temperatuur in de tijd")
    with col2:
        make_line("year_week", "OBS_VALUE", df, "Sterfte in de tijd")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("t<=16.5")
        multiple_lineair_regression(df_lower, ["temp_avg"], "OBS_VALUE")
        poisson_regression(df_lower)
    with col2:
        st.subheader("t>16.5")
        multiple_lineair_regression(df_higher, ["temp_avg"], "OBS_VALUE")
        poisson_regression(df_higher)

if __name__ == "__main__":
    main()
