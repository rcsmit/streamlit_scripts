import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# stacked range bar chart
# as proposed by https://www.youtube.com/watch?v=5zBg9pH_6bE

# by Rene Smit (@rcsmit), with help from Github Co Pilot (GPT 4o)


def plotly_stacked_range_bar_chart(df,ranges, colors):
    """
    Create and display a stacked range bar chart using Plotly.

    Parameters:
    df (DataFrame): DataFrame containing the data.
    ranges (list): List of tuples representing the ranges.
    colors (list): List of colors for the ranges.
    """
    # Create figure
    fig = go.Figure()

    # Add traces for each range
    for i, (start, end) in enumerate(ranges):
        for j, (index, row) in enumerate(df.iterrows()):
            fig.add_trace(go.Bar(
                x=[index],
                y=[row[end] - row[start]],
                base=row[start],
                name=f"{start} to {end}" if j == 0 else None,
                marker_color=colors[i],
                showlegend=j == 0
            ))

    # Customize layout
    fig.update_layout(
        title='Stacked Range Bar Chart',
        xaxis_title='Location',
        yaxis_title='Temperature (°C)',
        barmode='stack'
    )

    # Show figure
    st.plotly_chart(fig)

def matplotlib_stacked_range_bar_chart(df,ranges, colors):
    """
    Create and display a stacked range bar chart using Matplotlib.

    Parameters:
    df (DataFrame): DataFrame containing the data.
    ranges (list): List of tuples representing the ranges.
    colors (list): List of colors for the ranges.
    """
    # Create figure and axis
    fig, ax = plt.subplots()

    # Add bars for each range
    for i, (start, end) in enumerate(ranges):
        for j, (index, row) in enumerate(df.iterrows()):
            ax.bar(index, row[end] - row[start], bottom=row[start], color=colors[i], label=f"{start} to {end}" if j == 0 else "")

    # Customize layout
    ax.set_title('Stacked Range Bar Chart')
    ax.set_xlabel('Location')
    ax.set_ylabel('Temperature (°C)')
    ax.legend()

    # Show plot
    st.pyplot(fig)


def plotly_stacked_range_bar_chart_rotated(df, ranges, colors):
    """
    Create and display a stacked range bar chart using Plotly with temperature 
    on the x-axis and locations on the y-axis. Can be used as a gantt chart.

    Parameters:
    df (DataFrame): DataFrame containing the data.
    ranges (list): List of tuples representing the ranges.
    colors (list): List of colors for the ranges.
    """
    # Create figure
    fig = go.Figure()

    # Add traces for each range
    for i, (start, end) in enumerate(ranges):
        for j, (index, row) in enumerate(df.iterrows()):
            fig.add_trace(go.Bar(
                y=[index],
                x=[row[end] - row[start]],
                base=row[start],
                orientation='h',
                name=f"{start} to {end}" if j == 0 else None,
                marker_color=colors[i],
                showlegend=j == 0
            ))

    # Customize layout
    fig.update_layout(
        title='Stacked Range Bar Chart',
        xaxis_title='Temperature (°C)',
        yaxis_title='Location',
        barmode='stack'
    )

    # Show figure
    st.plotly_chart(fig)

def matplotlib_stacked_range_bar_chart_rotated(df, ranges, colors):
    """
    Create and display a stacked range bar chart using Matplotlib with temperature 
    on the x-axis and locations on the y-axis. Can be used as a gantt chart.

    Parameters:
    df (DataFrame): DataFrame containing the data.
    ranges (list): List of tuples representing the ranges.
    colors (list): List of colors for the ranges.
    """
    # Create figure and axis
    fig, ax = plt.subplots()

    # Add bars for each range
    for i, (start, end) in enumerate(ranges):
        for j, (index, row) in enumerate(df.iterrows()):
            ax.barh(index, row[end] - row[start], left=row[start], color=colors[i], label=f"{start} to {end}" if j == 0 else "")

    # Customize layout
    ax.set_title('Stacked Range Bar Chart')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Location')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    # Show plot
    st.pyplot(fig)


def provide_data():
    """Get the data for the stacked range bar chart from a dictionary or a list.

    Returns:
        df: dataframe with the data
    """
    # YOU CAN USE THIS FORMAT

    data_1 = {
        "Ontario": {
            "Winter mean low": -9,
            "Annual mean low": 3,
            "Annual mean": 8,
            "Annual mean high": 13,
            "Summer mean high": 27,
        },
        "England": {
            "Winter mean low": 3,
            "Annual mean low": 8,
            "Annual mean": 12,
            "Annual mean high": 16,
            "Summer mean high": 24,
        },
        "Kentucky": {
            "Winter mean low": -3,
            "Annual mean low": 8,
            "Annual mean": 14,
            "Annual mean high": 20,
            "Summer mean high": 30,
        },
    }
    df = pd.DataFrame(data_1).T

    # OR USE THIS FORMAT
   
    data_2 = {
        "": ["Winter mean low", "Annual mean low", "Annual mean", "Annual mean high", "Summer mean high"],
         "Ontario": [-9,3,8,13,27],
        "England": [3,8,12,16,24],
        "Kentucky": [-3,8,14,20,30]
        }
    # Convert data to DataFrame
    df = pd.DataFrame(data_2)
    df = df.set_index("").transpose()
    return df


def main():

    df = provide_data()
   
    # Extract column names and create dynamic ranges
    columns = df.columns.tolist()
    ranges = [(columns[i], columns[i + 1]) for i in range(len(columns) - 1)]
    colors = ['orange', '#FFD580', '#ADD8E6', 'blue']

    plotly_stacked_range_bar_chart(df,ranges, colors)
    matplotlib_stacked_range_bar_chart(df,ranges, colors)

    plotly_stacked_range_bar_chart_rotated(df,ranges, colors)
    matplotlib_stacked_range_bar_chart_rotated(df,ranges, colors)

if __name__ == "__main__":
    main()
