import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from scipy.fftpack import fft
from statsmodels.stats.diagnostic import acorr_ljungbox 
import plotly.express as px
from statsmodels.tsa.seasonal import STL

def get_data_knmi():
    """
    Retrieve KNMI weather data from a CSV file and preprocess it.

    Returns:
        pd.DataFrame: Preprocessed KNMI data with 'Date' as the index and 'Month' column added.
    """
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result.csv" 
    df = pd.read_csv(url, delimiter=",", header=None, comment="#", low_memory=False)

    column_replacements = [
        [0, "STN"], [1, "YYYYMMDD"], [2, "temp_avg"], [3, "temp_min"], [4, "temp_max"],
        [5, "T10N"], [6, "zonneschijnduur"], [7, "perc_max_zonneschijnduur"], [8, "glob_straling"],
        [9, "neerslag_duur"], [10, "neerslag_etmaalsom"], [11, "RH_min"], [12, "RH_max"],
        [13, "EV24"], [14, "wind_max"]
    ]

    df.rename(columns=dict(column_replacements), inplace=True)
    df["Date"] = pd.to_datetime(df["YYYYMMDD"].astype(str))
    
    return df

@st.cache_data()
def get_data(choice, interval):
    """
    Retrieve financial data from Yahoo Finance.

    Args:
        choice (str): Ticker symbol for the financial data.
        interval (str): Data interval (e.g., '1d', '1wk', '1mo').

    Returns:
        pd.DataFrame: Preprocessed financial data with 'Date' as the index and 'Month' column added.
    """
    data = yf.download(tickers=choice, start="2015-01-01", interval=interval, group_by='ticker', auto_adjust=True, prepost=False)
    df = pd.DataFrame(data)

    if df.empty:
        print(f"No data or wrong input - {choice}")
        return None
    st.write(df)
    
    df['rownumber'] = np.arange(len(df))
    df.columns = ['_'.join(col) for col in df.columns]
    df["Close"] = df[f"{fieldname}_Close"]  
    st.write(df)
    
    df["Koers"] = df["Close"]
    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df.get("Datetime", df["Date"]))
    
    return df

def generate_seasonal_data():
    """
    Generate a DataFrame with 'Date' and a perfect seasonal pattern peaking on the 1st of July.

    Returns:
        pd.DataFrame: DataFrame with 'Date' and 'Value' columns.
    """
    date_range = pd.date_range(start='2005-01-01', end='2023-12-31', freq='D')
    days_in_year = 365.25
    seasonal_pattern = 50 + 50 * np.sin(2 * np.pi * (date_range.dayofyear - 182) / days_in_year)
    df = pd.DataFrame({'Date': date_range, 'Value': seasonal_pattern})
    
    return df

def generate_random_data():
    """
    Generate a DataFrame with 'Date' from 1/1/2005 and 'Koers' with random numbers between 0 and 100.

    Returns:
        pd.DataFrame: DataFrame with 'Date' and 'Koers' columns.
    """
    date_range = pd.date_range(start='2005-01-01', end='2023-12-31', freq='D')
    koers_values = np.random.randint(0, 100, size=len(date_range))
    df = pd.DataFrame({'Date': date_range, 'Koers': koers_values})
   
    return df
def plot_plotly_chart(df, fieldname, what):
    """
    Plot a Plotly chart for the given fieldname.

    Args:
        df (pd.DataFrame): DataFrame containing the data to plot.
        fieldname (str): The column name to plot.
        what (str): Description of the data being plotted.
    """
    st.subheader(f"Plotly Chart - {what}")
    fig = px.line(df, x=df.index, y=fieldname, title=f'{what} over Time')
    try:
        for year in df.index.year.unique():
            fig.add_vline(x=pd.Timestamp(f'{year}-01-01'), line_dash="dash", line_color="red")
            fig.add_vline(x=pd.Timestamp(f'{year}-07-01'), line_dash="dash", line_color="yellow")
    except:
        #df.set_index(['Year', 'Month'], inplace=True)
        # for year in df['Year'].unique():
        #     fig.add_vline(x=pd.Timestamp(f'{year}'), line_dash="dash", line_color="red")
        

        # monthly data
        nr_of_years = int(len(df)/12)+1
        for n in range(0, nr_of_years):
            m = (n*12)+6    
            fig.add_vline(x=n*12, line_dash="dash", line_color="red")
            fig.add_vline(x=m, line_dash="dash", line_color="yellow")
    st.plotly_chart(fig)

def plot_boxplot(df, fieldname, what):
    """
    Plot a boxplot for the given fieldname per month.

    Args:
        df (pd.DataFrame): DataFrame containing the data to plot.
        fieldname (str): The column name to plot.
    """
    st.subheader(f"Boxplot per month - {what}")
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Month', y=fieldname, data=df)
    plt.title('Distribution per month')
    st.pyplot(plt)

def plot_seasonal_decomposition(df, fieldname, what):
    """
    Plot seasonal decomposition for the given fieldname.

    Args:
        df (pd.DataFrame): DataFrame containing the data to decompose.
        fieldname (str): The column name to decompose.
    """
    
    
    st.subheader(f"Seizoensdecompositie - {what}")
    
    # the results are too regular
    # result = seasonal_decompose(df[fieldname], model='additive', period=12)
    # # Applying STL decomposition
    # fig = result.plot()
    # st.pyplot(fig)


    if isinstance(df, pd.Series):
        df = df.to_frame()

    stl = STL(df[fieldname], period=12)
    result = stl.fit()
    ig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    ax1.plot(df[fieldname])
    ax1.set_title(f'{what}')
    ax2.plot(result.trend)
    ax2.set_title('Trend Component')
    ax3.plot(result.seasonal)
    ax3.set_title('Seasonal Component')
    ax4.plot(result.resid)
    ax4.set_title('Residual (Noise) Component')

    plt.tight_layout()
    st.pyplot(plt)

    st.info("If the seasonal decomposition shows a clear annual pattern: confirmation.")

def plot_autocorrelation(df, fieldname, what):
    """
    Plot autocorrelation for the given fieldname.

    Args:
        df (pd.DataFrame): DataFrame containing the data to analyze.
        fieldname (str): The column name to analyze.
    """
    st.subheader(f"Autocorrelation analyse - {what}")
    fig, ax = plt.subplots()
    plot_acf(df[fieldname], lags=12, ax=ax)
    st.pyplot(fig)
    st.info("Autocorrelations: Peaks at  lag 12, 24, etc., suggest season influence.")

def perform_anova(df, fieldname, what):
    """
    Perform ANOVA for monthly differences for the given fieldname.

    Args:
        df (pd.DataFrame): DataFrame containing the data to analyze.
        fieldname (str): The column name to analyze.
    """
    st.subheader(f"ANOVA voor monthly differences - {what}")
    model = ols(f"{fieldname} ~ C(Month)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    st.write("ANOVA Results:\n", anova_table)

    groepen = [df[df["Month"] == m][fieldname].dropna() for m in range(1, 13)]
    stat, p = stats.f_oneway(*groepen)
    st.write(f"ANOVA p-value: {p}")

def perform_kruskal_wallis(df, fieldname, what):
    """
    Perform Kruskal-Wallis test for monthly differences for the given fieldname.

    Args:
        df (pd.DataFrame): DataFrame containing the data to analyze.
        fieldname (str): The column name to analyze.
    """
    st.subheader(f"Kruskal-Wallis-test (non-parametrisch) - {what}")
    groups = [df[df['Month'] == month][fieldname] for month in range(1, 13)]
    h_stat, p_value = stats.kruskal(*groups)
    st.write(f"Kruskal-Wallis p-waarde: {p_value:.4f}")
    st.info("ANOVA/Kruskal-Wallis: A p-value < 0.05 indicates significant differences between months, i.e. seasonal influence")

def perform_ljung_box(df, fieldname, what):
    """
    Perform Ljung-Box test for the given fieldname.

    Args:
        df (pd.DataFrame): DataFrame containing the data to analyze.
        fieldname (str): The column name to analyze.
    """
    st.subheader(f"Ljung-Box-test - {what}")
    result = acorr_ljungbox(df[fieldname], lags=[12], return_df=True)
    st.write(result)

def perform_regression(df, fieldname, what):
    """
    Perform regression analysis for seasonality detection.

    Args:
        df (pd.DataFrame): DataFrame containing the data to analyze.
        fieldname (str): The column name to analyze.
    """
    st.subheader(f"Regression - {what}")
    df["day_of_the_year"] = df.index.dayofyear
    value = 365.25
    df["sin_j"] = np.sin(2 * np.pi * df["day_of_the_year"] / value)
    df["cos_j"] = np.cos(2 * np.pi * df["day_of_the_year"] / value)

    X = sm.add_constant(df[["sin_j", "cos_j"]])
    y = df[fieldname]

    model = sm.OLS(y, X).fit()
    st.write(model.summary())
    st.info("If the coefficients of sin_j and cos_j are significant (p-value < 0.05), this indicates a seasonal pattern. A high R² means that a large part of the exchange rate variation is explained by these cyclical components.")


def rescale_yearly(df, fieldname):
    """
    Rescale the data year by year so that the average of each year is 100.

    Args:
        df (pd.DataFrame): DataFrame containing the data to rescale.
        fieldname (str): The column name to rescale.

    Returns:
        pd.DataFrame: DataFrame with an additional column for rescaled values.
    """
    df['Year'] = df.index.year
    df[fieldname] = df.groupby('Year')[fieldname].transform(lambda x: (x / x.mean()) * 100)
    return df

def find_seasonality(df, fieldname, what):
    """
    Analyze seasonality in the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data to analyze.
        fieldname (str): The column name to analyze for seasonality.

    Displays:
        Various plots and statistical test results to analyze seasonality.
    """
    df.set_index('Date', inplace=True) 
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    df_grouped = df.groupby(['Year', 'Month'])[fieldname].mean().reset_index()
    #df_grouped.set_index(['Year', 'Month'], inplace=True)
    
    plot_plotly_chart(df, fieldname, what)
    plot_plotly_chart(df_grouped, fieldname, what)
    plot_boxplot(df, fieldname, what)
    plot_seasonal_decomposition(df_grouped, fieldname, what)
    plot_autocorrelation(df_grouped, fieldname, what)
    perform_anova(df, fieldname, what)
    perform_kruskal_wallis(df, fieldname, what)
    perform_ljung_box(df_grouped, fieldname, what)
    perform_regression(df, fieldname, what)



def main():
    """
    Main function to run the Streamlit app.
    """
    st.header("Seasonal patterns")
    st.info("This script is designed to perform several analyses to detect seasonal patterns in time series data using various statistical and visualization techniques. It retrieves financial data from Yahoo Finance (EURO/Thai Baht). It is compared with weather data from KNMI (very seasonal), a random number generator (no seasonality), and a sinus curve (perfect seasonal).")

    
    
    for choice in ["EURTHB=X", "EURUSD=X"]:
        df = get_data(choice, "1d")
        find_seasonality(df, "Koers", choice)

    st.subheader("Max Temperature Data")
    
    df = get_data_knmi()
    find_seasonality(df, "temp_max", "Max Temperature")

    st.subheader("Random Data")
    random_df = generate_random_data()
    find_seasonality(random_df, "Koers", "Random")

    st.subheader("Seasonal Data")
    seasonal_df = generate_seasonal_data()
    find_seasonality(seasonal_df, "Value", "Seasonal Pattern")

if __name__ == "__main__":
    main()