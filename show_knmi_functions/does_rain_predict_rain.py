import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import get_data

def does_rain_predict_rain(df):
    """reproducing 
    
    https://medium.com/towards-data-science/does-rain-predict-rain-us-weather-data-and-the-correlation-of-rain-today-and-tomorrow-3a62eda6f7f7

    Args:
        df (_type_): Dataframe with information. 
        STN = codenumber of the staiton
        neerslag_etmaalsom = total amount of percipation per 24h
        YYYYMMDD = the date, already made as pd.datetime elsewhere 
                            df["YYYYMMDD"] = pd.to_datetime(df["YYYYMMDD"], format="%Y%m%d")
        
        RAINY (float, optional): How much percipation is need to consider  a day  as rainy. 
                                 Defaults to 0.5.
    """   
    RAINY = st.sidebar.number_input("treshold(mm percipation)",0.0,100.0,0.5) 
    NR_DAYS = st.sidebar.number_input("Number of days to consider",0,100,10) 
    df = df.replace(r'^\s*$', None, regex=True)
    df = df.fillna(0) # there is no data before 1906 and april 1945 is missing.
    df['neerslag_etmaalsom'] = df['neerslag_etmaalsom'].astype(int)
    
    st.write ("reproducing https://medium.com/towards-data-science/does-rain-predict-rain-us-weather-data-and-the-correlation-of-rain-today-and-tomorrow-3a62eda6f7f7")
    stationDF = df.rename({"STN":"STATION", "YYYYMMDD":"DATE", "neerslag_etmaalsom":'DlySumToday'}, axis='columns') 
    stationDF = stationDF[["STATION","DATE","DlySumToday"]]  # keep just what we need
    

    stationCopyDF = df[["STN","YYYYMMDD","neerslag_etmaalsom"]] # keep just what we need
    stationCopyDF = stationCopyDF.rename({"STN":"STATION","neerslag_etmaalsom":"DlySumOther", "YYYYMMDD":"DATEother"}, axis='columns')  
 
    # Add in some other dates, for which we will pull in rainfall.
    for n in range(1,NR_DAYS+1):
        stationDF[f"DATE_minus{n}"] = stationDF["DATE"] - pd.offsets.Day(n)
    stationDF["DATE_plus1"] = stationDF["DATE"] + pd.offsets.Day(1)
   
    # Join other rainfall onto base record. Adjust column names to make clear what we did.
    for n in range(1,NR_DAYS+1):
        stationDF = stationDF.merge(stationCopyDF, how='inner', left_on=["STATION",f"DATE_minus{n}"], right_on = ["STATION","DATEother"])
        ago_n = f"DlySum{n}DaysAgo"

        stationDF = stationDF.rename({"DlySumOther":ago_n}, axis='columns')  
        stationDF = stationDF.drop(columns=["DATEother"])

    stationDF = stationDF.merge(stationCopyDF, how='inner', left_on=["STATION","DATE_plus1"], right_on = ["STATION","DATEother"])
    stationDF = stationDF.rename({"DlySumOther":"DlySumTomorrow"}, axis='columns')  
    stationDF = stationDF.drop(columns=["DATEother"])
    stationDF["DaysOfRain"] = 0

    stationDF.loc[(stationDF["DlySumToday"] >= RAINY), "DaysOfRain"] = 1
    for i in range(1, NR_DAYS+1):
        conditions = [
            stationDF[f'DlySum{i - j}DaysAgo'] >= RAINY for j in range(i)
        ]
        combined_condition = stationDF['DlySumToday'] >= RAINY
        for cond in conditions:
            combined_condition &= cond
        stationDF.loc[combined_condition, 'DaysOfRain'] = i+1

    stationDF = stationDF[["STATION","DATE","DlySumToday", 'DaysOfRain']]      
    stationDF = stationDF.sort_values(by='DATE')
    stationDF['DlySumToday_tomorrow'] = stationDF['DlySumToday'].shift(-1)
    stationDF['DaysOfRain_next'] = stationDF['DaysOfRain'].shift(-1)
    stationDF['does_it_rain_tomorrow'] = stationDF['DlySumToday_tomorrow'] > RAINY
    # Create the new column based on the condition
    stationDF['rain_period'] = stationDF['DaysOfRain'].where(stationDF['DaysOfRain_next'] == 0, other=0)

    stationDF.drop('DlySumToday_tomorrow', axis=1, inplace=True)
    stationDF.drop('DaysOfRain_next', axis=1, inplace=True)

    st.header("Total period")
    rain_probabilities = stationDF.groupby('DaysOfRain')['does_it_rain_tomorrow'].mean().reset_index()
    st.write(rain_probabilities)
    fig = px.bar(rain_probabilities, x='DaysOfRain', y='does_it_rain_tomorrow', 
             title='Wil it rain the next day given .. days of rain')
    st.plotly_chart(fig)

        
    # Calculate the frequencies of the 'NewColumn' values
    frequency_data = stationDF['rain_period'].value_counts().reset_index()
    frequency_data.columns = ['rain_period', 'frequency']
    frequency_data_filtered = frequency_data[frequency_data['rain_period'] != 0]

    # Create a bar graph using Plotly Express
    fig = px.bar(frequency_data_filtered, x='rain_period', y='frequency', 
                title='Frequency of Rain Periods', labels={'rain_period': 'Rain Period', 'frequency': 'Frequency'})

    st.plotly_chart(fig)

    num_rainy_days = (stationDF['DlySumToday'] > RAINY).sum()
    total_days = len(stationDF)
    fraction_rainy_days = num_rainy_days / total_days
    st.write(f"Fraction of days with DlySumToday > {RAINY}: {fraction_rainy_days}")

    st.header("Per decade")
    #Calculate the number and fraction of days where 'DlySumToday' > 0.1 for each decade
    stationDF['decade'] = stationDF['DATE'].dt.year // 10 * 10
    rainy_days_per_decade = stationDF[stationDF['DlySumToday'] > RAINY].groupby('decade')['DlySumToday'].count()
    total_days_per_decade = stationDF.groupby('decade')['DlySumToday'].count()
    fraction_rainy_days_per_decade = rainy_days_per_decade / total_days_per_decade
    fraction_rainy_days_per_decade = fraction_rainy_days_per_decade.rename({"DlySumToday":"fraction_rainy_days"}).reset_index()
    st.subheader("Fraction of rainy days per decade:")
    st.write(fraction_rainy_days_per_decade)
    fig = px.bar(fraction_rainy_days_per_decade, x='decade', y='DlySumToday', 
             title='Fraction rainy days', labels={'DlySumToday': 'Fraction rainy days'})
    st.plotly_chart(fig)

    # RAIN PROBABILITIES PER [CONSECUTIVE DAYS OF RAIN] PER DECADE
    stationDF['decade'] = stationDF['DATE'].dt.year // 10 * 10
    rain_probabilities_by_decade = []

    for decade, decade_df in stationDF.groupby('decade'):
        rain_probabilities = decade_df.groupby('DaysOfRain')['does_it_rain_tomorrow'].mean().reset_index()
        rain_probabilities['decade'] = decade
        rain_probabilities_by_decade.append(rain_probabilities)

    rain_probabilities_combined = pd.concat(rain_probabilities_by_decade, ignore_index=True)
    result_table = rain_probabilities_combined.pivot(index='decade', columns='DaysOfRain', values='does_it_rain_tomorrow')
    st.subheader("RAIN PROBABILITIES PER [CONSECUTIVE DAYS OF RAIN] PER DECADE")
    st.write(result_table)

    # MAKE A PLOT (rain_probabilities islike result_table, but every value on a row)
    rain_probabilities = stationDF.groupby(['decade', 'DaysOfRain'])['does_it_rain_tomorrow'].mean().reset_index()
    fig = px.line(rain_probabilities, x='decade', y='does_it_rain_tomorrow', color='DaysOfRain', title='Rain Probabilities by Decade and Days of Rain')
    fig.update_xaxes(tickmode='linear', dtick=10)
    st.plotly_chart(fig)
    
    # Create a custom colorscale from light to dark
    colorscale = [
        [0, 'rgb(0, 256, 256)'],
        [1, 'rgb(0,128, 128)']
        
    ]

    result_table = result_table.iloc[::-1] # reverse the order, 1900 on top, 2020 bottom
    # Create a heatmap using plotly.graph_objs
    heatmap = go.Figure(data=go.Heatmap(
        z=result_table.values,
        x=result_table.columns,
        y=result_table.index,
        colorscale=colorscale   # You can choose a different color scale if desired, was 'Viridis'
    ))

    
    heatmap.update_layout(
        xaxis_title="Days of Rain",
        yaxis_title="Decade",
        title="Rain Probabilities Heatmap by Decade and Days of Rain"
    )

    # Display the heatmap
    st.plotly_chart(heatmap)

    # AVERAGE RAINGFALL PER DECADE
    st.subheader("AVERAGE RAINFALL per day (mm) / PER DECADE")
    average_rainfall_per_decade = stationDF.groupby('decade')['DlySumToday'].mean().reset_index()
    fig = px.bar(average_rainfall_per_decade, x='decade', y='DlySumToday', 
             title='Average Rainfall per Decade', labels={'DlySumToday': 'Average Rainfall per day (mm)'})
    fig.update_xaxes(tickmode='linear', dtick=10)
    st.plotly_chart(fig)

def main():
   
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result.csv" 
    df = get_data(url)
    
if __name__ == "__main__":
    # main()
    print ("")