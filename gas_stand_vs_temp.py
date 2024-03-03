import pandas as pd
import plotly.express as px

import streamlit as st
from scipy.stats import linregress
import statsmodels.api as sm
from datetime import datetime
import scipy.stats as stats
import numpy as np
from skmisc.loess import loess


def interface():
    what = st.sidebar.selectbox("What to show",['temp_min','temp_avg','temp_max','graad_dagen',  'T10N', 'zonneschijnduur', 'perc_max_zonneschijnduur', 'glob_straling', 'neerslag_duur', 'neerslag_etmaalsom', 'RH_min', 'RH_max' ],1)
    window_size = st.sidebar.number_input("Window size",1,100,3)
    if what =="graad_dagen":
        afkap_def = 999
    else:
        afkap_def = 18
    afkapgrens_scatter = st.sidebar.number_input("Afkapgrens scatter ",1,999,afkap_def)
    return what,window_size,afkapgrens_scatter

def calculate_graad_dagen(df_nw_beerta, what):

    """ CaLculate graaddagen. 
        https://www.olino.org/blog/nl/articles/2009/12/14/het-rekenen-met-graaddagen/
  
    Returns:
        _type_: _description_
    """    
    def calculate_graad_dagen_(row):
        month = row["YYYYMMDD"].month
        if 4 <= month <= 9:
            factor = 0.8
        elif month in [3, 10]:
            factor = 1.0
        else:
            factor = 1.1

        return factor * row['graad_dagen']
    
    if what == "graad_dagen":
        what = "temp_avg"
    df_nw_beerta["graad_dagen"] = 18 - df_nw_beerta[what]
    df_nw_beerta["graad_dagen"] = df_nw_beerta["graad_dagen"].apply(lambda x: max(0, x))

    df_nw_beerta["graad_dagen"] = df_nw_beerta.apply(calculate_graad_dagen_, axis=1)
    return df_nw_beerta

def get_verbruiks_data():
    google_sheets = False
    if google_sheets:
        sheet_id_verbruik = "1j9V-otA53UWaI7-pDS4owU_qtZtjgMK8K5DLaN9kgqk"
        sheet_name_verbruik = "verbruik"
        url_verbruik = f"https://docs.google.com/spreadsheets/d/{sheet_id_verbruik}/gviz/tq?tqx=out:csv&sheet={sheet_name_verbruik}"
        
        try:
            # df = pd.read_csv(csv_export_url, delimiter=",", header=0)
            df = pd.read_csv(url_verbruik, delimiter=",")
            df["datum"] = pd.to_datetime(df["datum"].astype(str),  format='%d/%m/%Y')
    
        except:
            st.error("Error reading verbruik")
            st.stop()

    else:
        excel_file_path = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/gasstanden95xxCN5.xlsx" # r"C:\Users\rcxsm\Documents\xls\gasstanden95xxCN5.xlsx"
        df = pd.read_excel(excel_file_path)
        df["datum"] = pd.to_datetime(df["datum"].astype(str),  format='%Y-%m-%d')
    
    df['week_number'] = df['datum'].dt.isocalendar().week
    df['year_number'] = df['datum'].dt.isocalendar().year
    return df

def get_weather_info(what):
    current_datetime = datetime.now()
    formatted_date = current_datetime.strftime("%Y%m%d")

    #url_nw_beerta = f"https://www.daggegevens.knmi.nl/klimatologie/daggegevens?stns=260&vars=TEMP:SQ:SP:Q:DR:RH:UN:UX&start=20190202&end={formatted_date}"
    url_nw_beerta = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/nw_beerta.csv"
    df_nw_beerta =  pd.read_csv(
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
            [12, "RH_max"]
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
        
    #divide_by_10 = False if platform.processor() else True
    divide_by_10 = True
    if divide_by_10:
        for d in to_divide_by_10:
            try:
                df_nw_beerta[d] = df_nw_beerta[d] / 10
            except:
                df_nw_beerta[d] = df_nw_beerta[d]
    df_nw_beerta["YYYYMMDD"] = pd.to_datetime(df_nw_beerta["YYYYMMDD"].astype(str))
    df_nw_beerta['week_number'] = df_nw_beerta['YYYYMMDD'].dt.isocalendar().week
    df_nw_beerta['year_number'] = df_nw_beerta['YYYYMMDD'].dt.isocalendar().year

    df_nw_beerta = calculate_graad_dagen(df_nw_beerta, what)

    # Group by 'week_number' and 'year_number' and calculate the average of 'temp_avg'
    if what =="graad_dagen":
        result = df_nw_beerta.groupby(['week_number', 'year_number'])[what].sum().reset_index()

    else:

        result = df_nw_beerta.groupby(['week_number', 'year_number']).agg({
                'temp_avg': 'mean',
                'temp_min': 'mean',
                'temp_max': 'mean',
                'graad_dagen': 'sum',
                
                'T10N': 'mean',
                'zonneschijnduur': 'mean',
                'perc_max_zonneschijnduur': 'mean',
                'glob_straling': 'mean',
                'neerslag_duur': 'mean',
                'neerslag_etmaalsom': 'mean',
                'RH_min': 'mean',
                'RH_max': 'mean' 
                  }).reset_index()
    # Display the result
    print(result)
    return result


def make_scatter(x,y, df):
    """make a scatter plot, and show correlation and the equation

    Args:
        x (str): x values-field
        y (str): y values-field
        merged_df (str): df
    """   
    st.subheader("Met trendlijnen")
   
    fig = px.scatter(df, x=x, y=y, hover_data=['year_week'], color='year_number',  trendline='ols')#  trendline_scope="overall", labels={'datum': 'Date', 'verbruik': 'Verbruik'})
    st.plotly_chart(fig)
    # Calculate the correlation
    correlation = df[x].corr(df[y])

    # Perform linear regression to get the equation of the line
    slope, intercept, r_value, p_value, std_err = linregress(df[x], df[y])

    # Print the correlation and equation of the line
    st.write(f"Correlation: {correlation:.2f}")
    st.write(f"Equation of the line: y = {slope:.2f} * x + {intercept:.2f}")

    st.subheader("Met betrouwbaarheidsintervallen")
  
    # Create temperature bins (1-degree bins)
    bin_width = 0.5
    df['temp_bin'] = pd.cut(df['temp_min'], bins=np.arange(df['temp_min'].min(), df['temp_min'].max() + bin_width, bin_width))

    # Group by temperature bins and calculate mean and standard deviation
    grouped = df.groupby('temp_bin')['verbruik'].agg([np.mean, np.std])

    # Calculate the number of data points in each group
    grouped['count'] = df.groupby('temp_bin')['verbruik'].count()

    # Define the confidence level (e.g., 95%)
    confidence_level = 0.95

    # Calculate the margin of error
    grouped['margin_error'] = grouped['std'] / np.sqrt(grouped['count']) * stats.t.ppf((1 + confidence_level) / 2, grouped['count'] - 1)

    # Calculate confidence intervals
    grouped['lower_ci'] = grouped['mean'] - grouped['margin_error']
    grouped['upper_ci'] = grouped['mean'] + grouped['margin_error']

    # Reset the index to make it more readable
    grouped.reset_index(inplace=True)
    
    # Delete rows where margin_error is NaN
    grouped = grouped.dropna(subset=['mean'])
    grouped = grouped.dropna(subset=['std'])
    #Calculate the average temperature for each temp_bin and add it as a new column
    grouped['average_temp'] = grouped['temp_bin'].apply(lambda x: (x.left + x.right) / 2)

    # Print the results
    print(grouped)
    
    # Create a scatter plot using Plotly Express
    df_= df.merge(grouped, left_on = 'temp_min', right_on = 'mean', how='outer')
    print (df_)
    print (df_.dtypes)
    import plotly.graph_objects as go

    fig = px.scatter(df_, x='temp_min', y='verbruik', hover_data=['year_week'], color='year_number', )
    # Add a line for the mean
    
    for m in ['mean', 'lower_ci', 'upper_ci']:
        df_[f"{m}_sma"] = df_[m].rolling(window=3).mean()

    fig.add_trace(go.Scatter(x=df_['average_temp'], y=df_['mean_sma'], ))
    fig.add_trace(go.Scatter(x=df_['average_temp'], y=df_['lower_ci_sma'],mode='lines', fill='tonexty', 
                                            fillcolor='rgba(120, 128, 0, 0.0)',
                                            line=dict(width=0.7, 
                                            color="rgba(0, 0, 255, 0.5)"),  name="lower CI", ))
    fig.add_trace(go.Scatter(x=df_['average_temp'], y=df_['upper_ci_sma'],fill='tonexty',
                                fillcolor='rgba(255, 0, 0, 0.2)',
                                line=dict(color='dimgrey', width=.5),
                                name="upper ci", ))

  
    # Update axis labels
    fig.update_xaxes(title_text='Temperature (°C)')
    fig.update_yaxes(title_text='Verbruik')

    # Show the plot
    st.plotly_chart(fig)







def multiple_lin_regr(merged_df):
    """wrapper for the multiple lineair regression

    Args:
        merged_df (df): df
    """    
    st.subheader("Multiple Lineair Regression")
    y_value, x_values = interface_mulitple_lineair_regression()
    multiple_lineair_regression(merged_df, x_values, y_value)

def make_plots(what, afkapgrens_scatter, merged_df):
    """Makes various plots

    Args:
        df (df): df
        what (str): what to show
        afkapgrens_scatter (int): days with a temperature above this, are ignored
        merged_df (df): df with info
    """    
    fig = px.scatter(merged_df, x='year_week', y=[what,'verbruik'], title=f"verbruik en {what} in de tijd")
    st.plotly_chart(fig)

    fig = px.line(merged_df, x='year_week', y=[f"{what}_sma",'verbruik_sma'], title=f"gladgestreken verbruik en {what} in de tijd")
    st.plotly_chart(fig)

    df_pivot = merged_df.pivot(index='week_number', columns='year_number', values='verbruik')
    fig = px.line(df_pivot, labels={'week_number': 'Week Number', 'value': 'Verbruik'}, title="verbruik in verschillende jaren")
    st.plotly_chart(fig)
 
    df_pivot = merged_df.pivot(index='week_number', columns='year_number', values=what)
    fig = px.line(df_pivot, labels={'week_number': 'Week Number', 'value': f"{what}"}, title=f"{what} in verschillende jaren")
    st.plotly_chart(fig)

    merged_df_ = merged_df[merged_df[what] < afkapgrens_scatter]
    make_scatter(what, 'verbruik', merged_df_)
    st.subheader("Lopend gemiddelde")
    make_scatter(f"{what}_sma", 'verbruik_sma', merged_df_)

def merge_dataframes(df, what, window_size, df_nw_beerta):
    """Merge the verbruik dataframe with the weather info of Nieuw Beerta

    Args:
        df (_type_): the dataframe with verbruiks info
        what (_type_): what to show
        window_size (_type_): window size for smooth moving average
        df_nw_beerta (_type_): the dataframe with weather ifno

    Returns:
        df: merged df
    """    
    merged_df = df.merge(df_nw_beerta, on=['year_number', 'week_number'], how='outer')
    merged_df["year_week"] = merged_df["year_number"].astype(str) +"_" +  merged_df["week_number"].astype(str)
    merged_df['verbruik_sma'] = merged_df['verbruik'].rolling(window=window_size).mean()
    merged_df[f"{what}_sma"] = merged_df[what].rolling(window=window_size).mean()
    return merged_df


def interface_mulitple_lineair_regression():
    """interface for the MLR

    Returns:
       y_value :
       x_values :
    """    
    y_value = st.selectbox("Y value", ['verbruik'],0)
    x_values_options = ['temp_avg', 'temp_min','temp_max','graad_dagen', 'T10N', 'zonneschijnduur', 'perc_max_zonneschijnduur', 'glob_straling', 'neerslag_duur', 'neerslag_etmaalsom', 'RH_min', 'RH_max']
    x_values_default = ['temp_min', 'zonneschijnduur', 'perc_max_zonneschijnduur']
    x_values = st.multiselect("X values", x_values_options, x_values_default)
    return y_value,x_values

def multiple_lineair_regression(df_,  x_values, y_value):
    """Calculates multiple lineair regression. User can choose the Y value and the X values

 
        A t-statistic with an absolute value greater than 2 suggests that the coefficient is 
        statistically significant.

        The p-value associated with each coefficient tests the null hypothesis that the coefficient is zero 
        (i.e., it has no effect). A small p-value (typically less than 0.05) suggests that the coefficient 
        is statistically significant.
        
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
    #df =df[["country","population"]+[y_value]+ x_values]
      
    # st.write("**DATA**")
    # st.write(df)
    # st.write(f"Length : {len(df)}")
    x = df[x_values]
    y = df[y_value]
  
    # with statsmodels
    x = sm.add_constant(x) # adding a constant
    model = sm.OLS(y, x).fit()
    st.write("**OUTPUT ORDINARY LEAST SQUARES**")
    print_model = model.summary()
    st.write(print_model)
    
def main():
    df = get_verbruiks_data()
    what, window_size, afkapgrens_scatter = interface()
    df_nw_beerta = get_weather_info(what)
    merged_df = merge_dataframes(df, what, window_size, df_nw_beerta)
    make_plots(what, afkapgrens_scatter, merged_df)
    if what !="graad_dagen":
        multiple_lin_regr(merged_df)

if __name__ == "__main__":
    main()