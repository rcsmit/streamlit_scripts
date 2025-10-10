import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

# nieuwe versie lonen_inflatie.py

def get_df():
    """Get the DF with the possibilities.
       Loading from Google Sheets gives the 2 first rows as column header.

    Returns:
        df: the dataframe with the choices
    """    
    
    # Google Sheet details (make it publicly accessible)
    sheet_id = "11bCLM4-lLZ56-XJjBjvXyXJ11P3PiNjV6Yl96x-tEnM"
    sheet_name = "data"
    # https://docs.google.com/spreadsheets/d/11bCLM4-lLZ56-XJjBjvXyXJ11P3PiNjV6Yl96x-tEnM/gviz/tq?tqx=out:csv&sheet=data
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    df=pd.read_csv(url, delimiter=',')
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col])
    #df['datum'] = pd.to_datetime(df['datum'])

    return df

def calculate_indexes(df, reference_date):
    # Reference date (1/1/2015) values for loon_40, loon_38, loon_36
    
    ref_loon_40 = df.loc[df['datum'] == reference_date, 'loon_40'].values[0]
    ref_loon_38 = df.loc[df['datum'] == reference_date, 'loon_38'].values[0]
    ref_loon_36 = df.loc[df['datum'] == reference_date, 'loon_36'].values[0]

    # Calculate indexed values
    df['loon_40_index'] = (df['loon_40'] / ref_loon_40) * 100
    df['loon_38_index'] = (df['loon_38'] / ref_loon_38) * 100
    df['loon_36_index'] = (df['loon_36'] / ref_loon_36) * 100


    # Reference date (2021-01-01) values for CPI, CPI_afgeleid, loon_40, loon_38, loon_36

    ref_CPI = df.loc[df['datum'] == reference_date, 'CPI'].values[0]
    ref_CPI_afgeleid = df.loc[df['datum'] == reference_date, 'CPI_afgeleid'].values[0]
    ref_loon_40 = df.loc[df['datum'] == reference_date, 'loon_40'].values[0]
    ref_loon_38 = df.loc[df['datum'] == reference_date, 'loon_38'].values[0]
    ref_loon_36 = df.loc[df['datum'] == reference_date, 'loon_36'].values[0]

    # Calculate indexed values
    df['CPI_index_'] = (df['CPI'] / ref_CPI) * 100
    df['CPI_afgeleid_index_'] = (df['CPI_afgeleid'] / ref_CPI_afgeleid) * 100
    df['loon_40_index_'] = (df['loon_40'] / ref_loon_40) * 100
    df['loon_38_index_'] = (df['loon_38'] / ref_loon_38) * 100
    df['loon_36_index_'] = (df['loon_36'] / ref_loon_36) * 100
    
    df['loon_40_index_div_CPI_index_'] =  df['loon_40_index_']/df['CPI_index_'] *100
    df['loon_38_index_div_CPI_index_'] =  df['loon_38_index_']/df['CPI_index_'] *100
    df['loon_36_index_div_CPI_index_'] =  df['loon_36_index_']/df['CPI_index_'] *100
    return df


def plot_loon(df, reference_date, yaxis_range):
    fig = go.Figure()

    # Add traces for CPI and CPI_afgeleid on primary y-axis
    fig.add_trace(go.Scatter(x=df['datum'], y=df['CPI_index_'], mode='lines', name='CPI Index_'))
    fig.add_trace(go.Scatter(x=df['datum'], y=df['CPI_afgeleid_index_'], mode='lines', name='CPI Afgeleid Index_'))

    # Add traces for loon_40, loon_38, loon_36 on secondary y-axis
    fig.add_trace(go.Scatter(x=df['datum'], y=df['loon_40'], mode='lines', name='Loon 40', yaxis='y2'))
    fig.add_trace(go.Scatter(x=df['datum'], y=df['loon_38'], mode='lines', name='Loon 38', yaxis='y2'))
    fig.add_trace(go.Scatter(x=df['datum'], y=df['loon_36'], mode='lines', name='Loon 36', yaxis='y2'))

    # Update layout for dual y-axis
    fig.update_layout(
        title=f'Indexed Values with Dual Y-Axis ({reference_date} = 100)',
        xaxis_title='Date',
        yaxis=dict(
            title=f'CPI Index (reference_date = 100)',
            side='left',
             range=[yaxis_range[0], yaxis_range[1]] 
        ),
        yaxis2=dict(
            title='Loon',
            overlaying='y',
            side='right',
            
        ),
        legend=dict(
            x=1.1,
            y=1
        )
    )

    # Show the plot
    st.plotly_chart(fig)
def plot(df, reference_date):
    yaxis_range = [df[['CPI_index_', 'CPI_afgeleid_index_', 'loon_40_index_', 'loon_38_index_', 'loon_36_index_', 'loon_40_index_div_CPI_index_','loon_38_index_div_CPI_index_','loon_36_index_div_CPI_index_']].min().min(), df[['CPI_index_', 'CPI_afgeleid_index_', 'loon_40_index_', 'loon_38_index_', 'loon_36_index_', 'loon_40_index_div_CPI_index_','loon_38_index_div_CPI_index_','loon_36_index_div_CPI_index_']].max().max()]

    # Create a line plot for each indexed column using plotly.express
    fig = px.line(df, x='datum', y=['CPI_index_', 'CPI_afgeleid_index_', 'loon_40_index_', 'loon_38_index_', 'loon_36_index_', 'loon_40_index_div_CPI_index_','loon_38_index_div_CPI_index_','loon_36_index_div_CPI_index_'],
                labels={
                    'value': f'Index ({reference_date} = 100)',
                    'datum': 'Date'
                },
                title=f'Indexed Values ({reference_date} = 100)')

    # Update layout for better visualization
    fig.update_layout(
        yaxis_title=f'Index ({reference_date} = 100)',
        xaxis_title='Date',
        legend_title_text='Variables',
        yaxis=dict(
            title=f'CPI Index (reference_date = 100)',
            side='left',
             range=[yaxis_range[0], yaxis_range[1]] 
        ),
    )
 
 
    # Show the plot
    st.plotly_chart(fig)
    return yaxis_range
def main():
    st.header("Minimumloon vs prijsindex")
    df = get_df()
    dates = df["datum"].tolist()
    reference_date = st.selectbox("Reference date = 100", dates, 3)

    df = calculate_indexes(df, reference_date)
    yaxis_range = plot (df, reference_date)
    plot_loon(df, reference_date, yaxis_range)
    st.info("Bron CPI: https://www.cbs.nl/nl-nl/cijfers/detail/83131NED")
    
    st.info("Deze tabel bevat cijfers over het prijsverloop van een pakket goederen en diensten dat een gemiddeld Nederlands huishouden aanschaft. Dit wordt de consumentenprijsindex (CPI) genoemd. In de tabel staat ook de afgeleide consumentenprijsindex: dit is de CPI waarin het effect van veranderingen in de tarieven van productgebonden belastingen (bijvoorbeeld btw en accijns op alcohol en tabak) en subsidies en van consumptiegebonden belastingen (bijvoorbeeld motorrijtuigenbelasting) is verwijderd.")
    
    st.info("Bron minimumloon: https://nl.wikipedia.org/wiki/Minimumloon")

if __name__ == "__main__":
    main()