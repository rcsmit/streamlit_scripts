import cbsodata
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# https://chatgpt.com/share/670b0590-e6f0-8004-9d7c-aed0e63214b4

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
        if end_year <= 1950:
            return int((start_year + end_year) / 2)
        else:
            return 9999  # We'll handle years beyond 1950 separately
    else:
        return None


def make_graph(data_combined):
    """make the graphs

    Args:
        data_combined (df): df with data

    Result:
        three graphs
    """    
    # Function to add traces for each figure
    def add_trace(fig, data, x_col, y_col, label,i):
        fig.add_trace(go.Scatter(
            x=data[x_col],
            y=data[y_col],
            mode='lines',
            name=label,
            line=dict(color=rainbow_colors[i % len(rainbow_colors)])
        ))

    # Function to update layout for each figure
    def update_layout(fig, title):
        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title="Levensverwachting"
        )

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
    fig_totaal = go.Figure()
    fig_mannen = go.Figure()
    fig_vrouwen = go.Figure()

    # List of target ages
    leeftijden1 = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    leeftijden2 = range(0,101)
    for leeftijden_,header in zip([leeftijden1,leeftijden2],["Tien jaars bins", "alle leeftijden"]):
        st.subheader(header)
        min = int(data_combined["average_year_x"].min())
        max = int(data_combined["average_year_x"].max())
        # Loop through the target ages and add traces
        for i,leeftijd in enumerate(leeftijden_):
            data_combined_ = data_combined[data_combined["LeeftijdOp31December"] == leeftijd].copy(deep=True)
            
            add_trace(fig_totaal, data_combined_, "average_year_x", "Te_bereiken_leeftijd_totaal", leeftijd,i)
            add_trace(fig_mannen, data_combined_, "average_year_x", "Te_bereiken_leeftijd_mannen", leeftijd,i)
            add_trace(fig_vrouwen, data_combined_, "average_year_x", "Te_bereiken_leeftijd_vrouwen", leeftijd,i)

        # Update layout for all figures
        update_layout(fig_totaal, f"Levensverwachting door de tijd - mannen en vrouwen ({min}-{max})")
        update_layout(fig_mannen, f"Levensverwachting door de tijd - mannen ({min}-{max})")
        update_layout(fig_vrouwen, f"Levensverwachting door de tijd - vrouwen ({min}-{max})")

        # Display the plots
        st.plotly_chart(fig_totaal)
        st.plotly_chart(fig_mannen)
        st.plotly_chart(fig_vrouwen)


def process_genders(data):
    """ Make a pivot-ish table with seperate columns for each gender. 
        Calculate the expected age for totals in the period before 1950
     
    """    
    # Create a mapping of gender to the new column names
    gender_mapping = {
        'Mannen': 'Te_bereiken_leeftijd_mannen',
        'Vrouwen': 'Te_bereiken_leeftijd_vrouwen',
        'Totaal mannen en vrouwen': 'Te_bereiken_leeftijd_totaal'
    }

    # Function to filter data by gender and rename the column
    def filter_and_rename(data, gender, new_col_name):
        df_filtered = data[data["Geslacht"] == gender].copy()
        df_filtered[new_col_name] = df_filtered["Te_bereiken_leeftijd"]
        return df_filtered

    # Apply the function to each gender
    data_mannen = filter_and_rename(data, 'Mannen', gender_mapping['Mannen'])
    data_vrouwen = filter_and_rename(data, 'Vrouwen', gender_mapping['Vrouwen'])
    data_totaal = filter_and_rename(data, 'Totaal mannen en vrouwen', gender_mapping['Totaal mannen en vrouwen'])

    # Merge the dataframes
    data_combined = pd.merge(data_mannen, data_vrouwen, on=["LeeftijdOp31December", "Perioden"])
    data_combined = pd.merge(data_combined, data_totaal, on=["LeeftijdOp31December", "Perioden"])

    # Calculate average and fill missing values
    data_combined['Te_bereiken_leeftijd_totaal_'] = (data_combined["Te_bereiken_leeftijd_vrouwen"] + data_combined["Te_bereiken_leeftijd_mannen"]) / 2
    data_combined['Te_bereiken_leeftijd_totaal'] = data_combined['Te_bereiken_leeftijd_totaal'].fillna(data_combined['Te_bereiken_leeftijd_totaal_'])
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
    
    data = get_data()
    data = process_data(data)
    data = process_genders(data)
    data_1950 = data[data["average_year"]>=1950]
    make_graph(data)
    st.header("Na 1950")
    make_graph(data_1950)
    st.info("Source: CBS (tabel 37360ned). Inspired by https://x.com/ActuaryByDay/status/1845129341362905148")

if __name__ == "__main__":
    main()