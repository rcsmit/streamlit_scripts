import cbsodata
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import life_expectancy_nl 
# https://chatgpt.com/share/670b0590-e6f0-8004-9d7c-aed0e63214b4


# THIS USES CBS DATA


@st.cache_data
def get_data():
    """Gets the data from Statistics Netherlands

    Returns:
        df: df
    """    
    data = pd.DataFrame(cbsodata.get_data('37360ned'))

    return data


def calculate_average_year(period):
    """ Calculates the average year if there is written 1901 tot 1906. 
        Returns 9999 if it is after 1950 and None if there already a single year 

    Args:
        period (str) : period

    Returns:
        value : average of the both years, 9999 or None
    """    
    if "tot" in period:
        start, end = period.split(" tot ")
        start_year = int(start.strip())
        end_year = int(end.strip())
        if end_year <= 1951:
            return int((start_year + end_year) / 2)
        else:
            return 9999  # We'll handle years beyond 1950 separately
    else:
        return None


def make_graph(data_combined, what, log,window, sex, complete):
    """make the graphs

    Args:
        data_combined (df): df with data

    Result:
        three graphs
    """    
    
    # Function to add traces for each figure
    def add_trace(fig, data, x_col, y_col, label,i):
        y_values = data[y_col].rolling(window=window).mean()
        fig.add_trace(go.Scatter(
            x=data[x_col],
            y=y_values,
            mode='lines',
            name=label,
            line=dict(color=rainbow_colors[i % len(rainbow_colors)])
        ))

    # Function to update layout for each figure
    def update_layout(fig, title, log):
        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title=what
        )

        if log==True:
            fig.update_yaxes(type="log")
    rainbow_colors = [
        
            "#000000",  # Black
            "#FF0000",  # Red
            "#FFD700",  # Gold
            '#FFFF00',  # Yellow
            "#99E619",  # Yellow-green
            "#32CD32",  # Lime green
            "#20B2AA",  # Light sea green
            "#00FF00",  # Green (note: uncommented for consistency)
            "#00BFFF",  # Deep sky blue
            "#000080",  # Navy blue
            "#800080",  # Purple
            "#999999",   # Light gray
        
            '#FF0000',  # Red
            '#FF7F00',  # Orange
            '#FFFF00',  # Yellow
            '#00FF00',  # Green
            '#0000FF',  # Blue
            '#4B0082',  # Indigo
            '#8B00FF',  # Violet
            '#FF1493',  # Deep Pink
            '#00FFFF',  # Cyan
            '#FFD700',  # Gold
            '#8B4513',  # Saddle Brown
            '#A52A2A'   # Brown
        ]
    # Initialize the figures
    
    fig = go.Figure()
    

    # List of target ages
    if complete:
        leeftijden = range(0,101)
        header = "alle leeftijden"
    else:
        leeftijden = [0, 5, 10, 20, 30, 40,47, 50, 60, 70, 80, 90, 100]
        header = "Tien jaars bins"
    st.subheader(header)
    min = int(data_combined["average_year_x"].min())
    max = int(data_combined["average_year_x"].max())
    # Loop through the target ages and add traces
    
    for i,leeftijd in enumerate(leeftijden):
        data_combined_ = data_combined[data_combined["LeeftijdOp31December"] == leeftijd].copy(deep=True)
        data_combined_=data_combined_.sort_values(by=['average_year_x'])
        if sex=="T":
            add_trace(fig, data_combined_, "average_year_x", f"{what}_totaal", leeftijd,i)
        elif sex=="M":
            add_trace(fig, data_combined_, "average_year_x", f"{what}_mannen", leeftijd,i)
        elif sex=="V":    
            add_trace(fig, data_combined_, "average_year_x", f"{what}_vrouwen", leeftijd,i)
        else:
            st.write("error")      
            st.stop()
    # Update layout for all figures
    update_layout(fig, f"{what} door de tijd - {sex} ({min}-{max})",log)
    

    # Display the plots
    st.plotly_chart(fig)
    


def process_genders(data, what):
    """ Make a pivot-ish table with seperate columns for each gender. 
        Calculate the expected age for totals in the period before 1950
     
    """    
    # Create a mapping of gender to the new column names
    gender_mapping = {
        'Mannen': f'{what}_mannen',
        'Vrouwen': f'{what}_vrouwen',
        'Totaal mannen en vrouwen': f'{what}_totaal'
    }

    # Function to filter data by gender and rename the column
    def filter_and_rename(data, gender, new_col_name):
        df_filtered = data[data["Geslacht"] == gender].copy()
        df_filtered[new_col_name] = df_filtered[what]
        return df_filtered
    
    # Apply the function to each gender
    data_mannen = filter_and_rename(data, 'Mannen', gender_mapping['Mannen'])
    data_vrouwen = filter_and_rename(data, 'Vrouwen', gender_mapping['Vrouwen'])
    data_totaal = filter_and_rename(data, 'Totaal mannen en vrouwen', gender_mapping['Totaal mannen en vrouwen'])
    
    # Merge the dataframes
    data_combined = pd.merge(data_mannen, data_vrouwen, on=["LeeftijdOp31December", "Perioden"])
    data_combined = pd.merge(data_combined, data_totaal, on=["LeeftijdOp31December", "Perioden"])
   
    # Calculate average and fill missing values
    #@if what=="Te_bereiken_leeftijd":
    data_combined[f'{what}_totaal_'] = (data_combined[f'{what}_vrouwen'] + data_combined[f'{what}_mannen']) / 2
    data_combined[f'{what}_totaal'] = data_combined[f'{what}_totaal'].fillna(data_combined[f'{what}_totaal_'])
    return data_combined
   
def process_data(data):
    """(pre)process the data.

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """    
    
    data['LeeftijdOp31December'] = data['LeeftijdOp31December'].str.replace(' jaar of ouder', '')
    data['LeeftijdOp31December'] = data['LeeftijdOp31December'].str.replace(' jaar', '')
    data["Te_bereiken_leeftijd"] = data["LeeftijdOp31December"].astype(float) + data["Levensverwachting_4"]
    # Apply the function to the 'Perioden' column to create a new column with the average year
    data['average_year'] = data['Perioden'].apply(calculate_average_year)
    data = data[data['average_year'] != 9999] # take away the 5 years data after 1950
    data.loc[:,'average_year'] = data['average_year'].fillna(data['Perioden']).astype(int)
    data.loc[:,"LeeftijdOp31December"] = data["LeeftijdOp31December"].astype(int)
    return data

def main():
    
    
    
    what = st.sidebar.selectbox("What", ["Te_bereiken_leeftijd","Sterftekans_1",
                                 "LevendenTafelbevolking_2",
                                 "OverledenenTafelbevolking_3",
                                 "Levensverwachting_4"])

    complete = st.sidebar.checkbox("All ages")
    after_1950 = st.sidebar.checkbox("After 1950")
    sex = st.sidebar.selectbox("Sexe", ["T","F","M"] )
    if what=="Sterftekans_1":
        log = st.sidebar.checkbox("Log at Y axis", False)
    else:
        log = False
    window = st.sidebar.number_input("SMA", 0,30,1)
    

    data = get_data()
    
    data = process_data(data)
    data = process_genders(data,what)
    if after_1950:
        data= data[data["average_year"]>=1950]
    make_graph(data, what, log,window,sex,complete)
    
    st.info("Source: CBS (tabel 37360ned https://opendata.cbs.nl/statline/#/CBS/nl/dataset/37360ned/table?fromstatweb).")
    st.info("Inspired by https://x.com/ActuaryByDay/status/1845129341362905148")
    life_expectancy_nl.main()

if __name__ == "__main__":
    main()