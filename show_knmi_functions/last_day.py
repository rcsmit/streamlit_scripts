import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    from show_knmi_functions.utils import get_data, loess_skmisc
except:
    from utils import get_data, loess_skmisc

def last_day(df, gekozen_weerstation, what_to_show_):
    """Make a plot that shows the last day that the minimum temperature was
    0 degrees.

    Inspired by a plot in the Volkskrant 14th May 2024
    

    Args:
        df (_type_): _description_
        gekozen_weerstation (_type_): _description_
        what_to_show_ (_type_): _description_
        value (_type_): _description_
    """
   
    
    #ijsheiligen = st.sidebar.selectbox("Show ijsheiligen", [True, False],0)

    first_last =st.sidebar.selectbox("first | last | extremes | ijsheiligen", ["first", "last", "extremes", "ijsheiligen"],2)
        
    if first_last == "ijsheiligen":
        min_month, max_month = 1,6
        value, first_last = 0, "ijsheiligen"
        what_to_show_ = ['temp_min']
        shift=False
        st.info("Make a plot that shows the last day that the minimum temperature was 0 degrees. Inspired by a plot in the Volkskrant 14th May 2024.Oange is April. Grey area is IJsheiligen (Ice Saints), 11 to 15 May")
        st.info ("https://www.volkskrant.nl/binnenland/laatste-vorst-van-het-jaar-valt-steeds-vaker-lang-voor-ijsheiligen~bf189dbf/")
    
    else:
        value = st.sidebar.number_input("Waarde", -99.0,99.0,0.0)
        min_month = st.sidebar.number_input("Maand minimaal (incl)", 1,12,1)
        max_month = st.sidebar.number_input("Maand maximaal (incl)", 1,12,12)
        shift= st.sidebar.selectbox("Shift 180 days", [False, True],0)  
    df['date'] = pd.to_datetime(df["YYYYMMDD"].astype(str))
    df['year'] = df['date'].dt.year
    df['MM'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
     
    df["year_"] =  df["year"].shift(180)
    # df['day_of_year_shifted'] = df['day_of_year'].shift(180)
    if shift:
        df["year_"] = df.apply(lambda row: row["year"] if row["MM"] >= 7 else row["year"] - 1, axis=1)

    # df=df[df['day_of_year_shifted'].notnull()]
    # df = df[~df['day_of_year_shifted'].isna()]
    # df['day_of_year_shifted'] = df['day_of_year'].astype(int)
    first_year = df["year"].min()
    fill_value = first_year - 1
    df["year_"] = df["year_"].fillna(fill_value)
    df = df[(df['MM'] >= min_month) & (df['MM'] <= max_month)]
    print (df)
    # Filter rows where temperature reaches the value
    df = df.dropna(subset=what_to_show_)
    marker_line = st.sidebar.selectbox("Graphtype (markers | lines)", ["markers","lines"],1)
   
    for what_to_show in what_to_show_:   
        
        # Find the first occurrence of 0 temperature for each year
        if first_last == "extremes":
            show_extremes(df, what_to_show)
        else:
            if (first_last =="last") or (first_last =="ijsheiligen"):
                first_zero_temp,title = find_last(df, first_last, value, what_to_show)
            else:
                first_zero_temp, title = find_first(df, first_last, value, what_to_show)
            # Convert day of year to date in 1900
            if len(first_zero_temp)==0:
                st.error("No values found")
                st.stop()  
            # Apply the transformation
            # first_zero_temp['day_of_year'] = first_zero_temp['day_of_year_shifted'].apply(
            #     lambda x: x - 180 if x - 180 > 0 else 365 + (x - 180)
            # )
            #first_zero_temp['date_1900'] = first_zero_temp.apply(lambda row: pd.Timestamp(year=1900, month=1, day=1) + pd.Timedelta(days=row['day_of_year'] - 1), axis=1)
            if shift:
                first_zero_temp['date_1900'] = first_zero_temp.apply(lambda row: pd.Timestamp(year=1900, month=7, day=1) + pd.Timedelta(days=row['day_of_year'] - 183) if row['day_of_year'] > 183 else pd.Timestamp(year=1900, month=1, day=1) + pd.Timedelta(days=row['day_of_year'] + 182), axis=1)
            else:
                first_zero_temp['date_1900'] = first_zero_temp.apply(lambda row: pd.Timestamp(year=1900, month=1, day=1) + pd.Timedelta(days=row['day_of_year'] - 1), axis=1)
            
            st.write(first_zero_temp)
            
            # Plotting
            sma =False # True if len(first_zero_temp) >= 42 else False
            
            if sma:
                # Calculate the 30-year moving average
                moving_avg = first_zero_temp['day_of_year'].rolling(window=30, center=False).mean()
            else:
                moving_avg = None

                # Apply LOESS smoothing (placeholder, actual implementation needed)
                # moving_avg_loess = loess_skmisc(first_zero_temp['year_'], first_zero_temp['day_of_year'])
            show_plot(sma, first_zero_temp, moving_avg, marker_line, first_last, value, what_to_show, title, shift)
def show_plot(sma, first_zero_temp, moving_avg, marker_line, first_last, value, what_to_show, title, shift):
    # Create a line plot
    fig = go.Figure()

    # Add a scatter plot trace
    if sma:
        fig.add_trace(go.Scatter(x=first_zero_temp['year_'],
                                y=moving_avg,
                                mode='lines',  # Use markers for scatter plot
                                marker=dict(color='black'),  # Set marker color to black
                                name='SMA 30 years'))
        
        # fig.add_trace(go.Scatter(x=first_zero_temp['year_'],
        #                         y=moving_avg_loess,
        #                         mode='lines',  # Use markers for scatter plot
        #                         marker=dict(color='red'),  # Set marker color to red
        #                         name='SMA 30 years (LOESS)'))

    fig.add_trace(go.Scatter(x=first_zero_temp['year_'],
                            y=first_zero_temp['date_1900'],
                            mode=marker_line,  # Use markers for scatter plot
                            marker=dict(color='blue'),  # Set marker color to blue
                            name=f'{first_last} Day of {value} {what_to_show}'))
    if first_last == "ijsheiligen":
        # Add a horizontal bar for "IJsheiligen" from 11 May to 15 May
        ijsheiligen_start = pd.Timestamp(year=1900, month=5, day=11)
        ijsheiligen_end = pd.Timestamp(year=1900, month=5, day=15)

        fig.add_shape(
            type="rect",
            xref="paper", yref="y",
            x0=0, y0=ijsheiligen_start,
            x1=1, y1=ijsheiligen_end,
            fillcolor="gray",
            opacity=0.3,
            layer="below",
            line=dict(width=0),
            name='IJsheiligen',
        )
    show_april = False
    if show_april:
        # Add a horizontal bar for April
        april_start = pd.Timestamp(year=1900, month=4, day=1)
        april_end = pd.Timestamp(year=1900, month=4, day=30)

        fig.add_shape(
            type="rect",
            xref="paper", yref="y",
            x0=0, y0=april_start,
            x1=1, y1=april_end,
            fillcolor="orange",
            opacity=0.3,
            layer="below",
            line=dict(width=0),
            name='April',
        )

    # Update layout
    fig.update_layout(title=title,
                    xaxis_title='Year',
                    yaxis_title='Date',
                    showlegend=True,
                    xaxis=dict(tickmode='linear', dtick=10))  # Ensure linear tick mode for x-axis.
    # Show plot
   
    if first_last == "ijsheiligen":
        fig.update_layout(yaxis=dict(tickformat="%d-%m", range=[pd.Timestamp(year=1900, month=3, day=1), pd.Timestamp(year=1900, month=6, day=15)]))
    else:
            
        if shift:
            
            fig.update_layout(yaxis=dict(tickformat="%d-%m", range=[pd.Timestamp(year=1900, month=7, day=1), pd.Timestamp(year=1901, month=6, day=30)]))  # Ensure linear tick mode for x-axis.
        else:
            fig.update_layout(yaxis=dict(tickformat="%d-%m", range=[pd.Timestamp(year=1900, month=1, day=1), pd.Timestamp(year=1900, month=12, day=31)]))
        
# Show plot
    st.plotly_chart(fig)

def find_first(df, first_last, value, what_to_show):
    zero_temp_df = df.query(f'{what_to_show}>={value}' )
            
    first_zero_temp = zero_temp_df.groupby('year_')['day_of_year'].min().reset_index()
    title = f'{first_last} day of {value} {what_to_show} each year'
    return first_zero_temp,title

def find_last(df, first_last, value, what_to_show):
    zero_temp_df = df.query(f'({what_to_show}<={value})' )                             # &{what_to_show}>{value}-1' )
            
    print (zero_temp_df)
    first_zero_temp = zero_temp_df.groupby('year_')['day_of_year'].max().reset_index()
    title = f'{first_last} day of {what_to_show} between {value} and {value-1} for each year'
    return first_zero_temp,title
def show_extremes(df, what_to_show):
    years = df["year_"].unique()
           
            # Display hottest and coldest days
    hottest_dates, hottest_temps = [],[]
    coldest_dates, coldest_temps =[],[]
    for y in years:
        df_year = df[df["year_"] == y].copy(deep=True)
        hottest_day = df_year.loc[df_year[what_to_show].idxmax()]
        hottest_date = hottest_day['date'].strftime('%d-%m-%Y')
        hottest_temp = hottest_day[what_to_show]
        coldest_day = df_year.loc[df_year[what_to_show].idxmin()]
        coldest_date = coldest_day['date'].strftime('%d-%m-%Y')
        coldest_temp = coldest_day[what_to_show]


        hottest_dates.append(hottest_date)
        hottest_temps.append(hottest_temp)
        coldest_dates.append(coldest_date)
        coldest_temps.append(coldest_temp)
                # st.subheader(f"{y}")
                # st.write(f"In {y}, the lowest value is on {coldest_date} with a {what_to_show} of {coldest_temp}°C.")
                # st.write(f"In {y}, the highest value is on {hottest_date} with a {what_to_show} of {hottest_temp}°C.")
                
     
    plot_extremes(years, hottest_dates, hottest_temps, coldest_dates, coldest_temps)

def plot_extremes(years, hottest_dates, hottest_temps, coldest_dates, coldest_temps):
      

            # Convert list of string dates to datetime
    h_dates = pd.to_datetime(hottest_dates, dayfirst=True)

            # Change year to 1900 and format to dd-mm-1900
    formatted_h_dates = [(date.replace(year=1900)).strftime('%Y-%m-%d') for date in h_dates]
    c_dates = pd.to_datetime(coldest_dates, dayfirst=True)


            # Change year to 1900 and format to dd-mm-1900
    formatted_c_dates = [(date.replace(year=1900)).strftime('%Y-%m-%d') for date in c_dates]



    fig = go.Figure()

                        
    fig.add_trace(go.Scatter(
                x=years,
                y=formatted_h_dates,
                mode='lines+markers',
                marker=dict(color='red'),
                name='Hottest',
                text=[f"{temp}°C" for temp in hottest_temps],
                hovertext=[f"Date: {date}<br>Temp: {temp}°C" for date, temp in zip(hottest_dates, hottest_temps)],
                hoverinfo='text'
            ))

    fig.add_trace(go.Scatter(
                x=years,
                y=formatted_c_dates,
                mode='lines+markers',
                marker=dict(color='blue'),
                name='Coldest',
                text=[f"{temp}°C" for temp in coldest_temps],
                hovertext=[f"Date: {date}<br>Temp: {temp}°C" for date, temp in zip(coldest_dates, coldest_temps)],
                hoverinfo='text'
            ))

    fig.update_layout(
                title="Hottest and Coldest Days Each Year",
                xaxis_title='Year',
                yaxis_title='Date',
                showlegend=True,
                xaxis=dict(tickmode='linear', dtick=10),
                yaxis=dict(tickformat="%d-%m")
            )

    fig.update_layout(yaxis=dict(tickformat="%d-%m"))
    st.plotly_chart(fig)

    
def main():
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result1900.csv" 
    df = get_data(url)
    what_to_show_ = st.sidebar.multiselect("What to show", ["temp_min", "temp_avg", "temp_max"], ["temp_avg"])
                                           
    
    last_day(df, "De Bilt", what_to_show_)
if __name__ == "__main__":
    main()