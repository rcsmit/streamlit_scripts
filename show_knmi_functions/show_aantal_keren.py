import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from show_knmi_functions.utils import get_data

def show_aantal_keren(df_, gekozen_weerstation, what_to_show_):
    # TODO : stacked bargraphs met meerdere condities
    # https://twitter.com/Datagraver/status/1535200978814869504/photo/1
    months = {
            "1": "Jan",
            "2": "Feb",
            "3": "Mar",
            "4": "Apr",
            "5": "May",
            "6": "Jun",
            "7": "Jul",
            "8": "Aug",
            "9": "Sep",
            "10": "Oct",
            "11": "Nov",
            "12": "Dec",
        }
    what_to_show_ = what_to_show_ if type(what_to_show_) == list else [what_to_show_]

    df_.set_index("YYYYMMDD")
    (month_min,month_max) = st.sidebar.slider("Maanden (van/tot en met)", 1, 12, (1,12))

    value_min = st.sidebar.number_input("Waarde vanaf", -99, 99, 0)
    value_max = st.sidebar.number_input("Waarde tot en met", -99, 99, 99)

    #jaren = df["YYYY"].tolist()
    for what_to_show in what_to_show_:
        st.subheader(what_to_show)
        
        df = df_[(df_["MM"] >= month_min) & (df_["MM"] <= month_max)].reset_index().copy(deep=True)
     
        # TODO :  this should be easier: 
        for i in range(len(df)):
            #if ((df.loc[i, what_to_show]  >= value_min) & (df.loc[i,what_to_show] <= value_max)):
            if ((df[what_to_show].iloc[i]  >= value_min) & (df[what_to_show].iloc[i] <= value_max)):
                df.loc[i,"count_"] = 1
            else:
                df.loc[i,"count_"] = 0
        df = df[df["count_"] == 1]
        
        # aantal keren
        df_grouped_aantal_keren = df.groupby(by=["year"]).sum(numeric_only=True).reset_index() # werkt maar geeft geen 0 waardes weer 
        title = (f"Aantal keren dat { what_to_show} in {gekozen_weerstation} tussen {value_min} en {value_max} ligt\n")
        
        plot_df_grouped( months, month_min, month_max, df_grouped_aantal_keren, "count_", title)

        
        # per maand
        table_per_month = pd.pivot_table(df, values="count_", index='MM', columns='year', aggfunc='sum', fill_value=0)
        all_months = range(month_min, month_max+1)
        all_years = df['year'].unique()
        table_per_month = table_per_month.reindex(index=all_months, columns=all_years, fill_value=0)
      
        fig_ = px.imshow(table_per_month, title=title)
        #fig.show()
        st.plotly_chart(fig_)
        
        # Som
        df_grouped_som = df.groupby(by=["year"]).sum(numeric_only=True).reset_index() # werkt maar geeft geen 0 waardes weer 
        title = (f"Som van {what_to_show} in {gekozen_weerstation} tussen {value_min} en {value_max}")
        plot_df_grouped( months, month_min, month_max, df_grouped_som, what_to_show, title)

        # Gemiddelde
        df_grouped_mean = df.groupby(by=["year"]).mean(numeric_only=True).reset_index() # werkt maar geeft geen 0 waardes weer 
        title = (f"Gemiddelde van {what_to_show} in {gekozen_weerstation} tussen {value_min} en {value_max}")
        plot_df_grouped( months, month_min, month_max,  df_grouped_mean, what_to_show, title)

def plot_df_grouped(months, month_min, month_max,  df_grouped_, veldnaam, title):
    # fig, ax = plt.subplots()
    # plt.set_loglevel('WARNING') #Avoid : Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.
    df_grouped = df_grouped_[["year", veldnaam]]

    if month_min ==1 & month_max ==12:
        st.write("compleet jaar") # FIXIT : werkt niet

    else:
        title += f" in de maanden {months.get(str(month_min))} tot en met {months.get(str(month_max))}"

    fig = px.bar(df_grouped, x='year', y=veldnaam, title=title)
    st.plotly_chart(fig)
 

    # HEATMAP
    # Create a 2D array from the dataframe for the heatmap
    heatmap_data = pd.pivot_table(df_grouped, values=veldnaam, index='year', columns=None)

    # Create the heatmap using plotly
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,  # Use the values from the 2D array
        x=heatmap_data.columns,  # X-axis (in this case, the count)
        y=heatmap_data.index,    # Y-axis (in this case, the year)
        colorscale='Viridis'     # Choose a colorscale (you can change it to another if you prefer)
    ))

    # Customize the heatmap layout
    fig.update_layout(
        title=title,
        xaxis_title='_',
        yaxis_title='Year'
    )

    # Show the heatmap
    st.plotly_chart(fig)

   
def main():
   
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result.csv" 
    df = get_data(url)
    
if __name__ == "__main__":
    # main()
    print ("")