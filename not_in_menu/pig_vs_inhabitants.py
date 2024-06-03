import pandas as pd
import plotly.express as px
import streamlit as st
# reactie op https://twitter.com/Focusscience1/status/1797644870267625801
# https://ec.europa.eu/eurostat/databrowser/view/apro_mt_lspig/default/table?lang=en
# https://en.wikipedia.org/wiki/List_of_European_countries_by_population (UN estimate 2023)
# https://en.wikipedia.org/wiki/List_of_European_countries_by_area


def plot(x,y):
    fig = px.scatter(df, x=x, y=y)

    # Add annotations
    for i, row in df.iterrows():
        fig.add_annotation(
            x=row[x],
            y=row[y],
            text=row["Country"],
            showarrow=False,
            font=dict(
                color="black",
                size=12,
            )
        )
    st.plotly_chart(fig)

data = {
    'Country': ['Turkey', 'Malta', 'Luxembourg', 'North Macedonia', 'Slovenia', 'Estonia', 'Latvia', 'Cyprus', 'Slovakia', 'Lithuania', 'Bulgaria', 'Greece', 'Croatia', 'Finland', 'Switzerland', 'Sweden', 'Czech Republic', 'Ireland', 'Serbia', 'Portugal', 'Austria', 'Hungary', 'Romania', 'Belgium', 'Italy', 'Poland', 'Netherlands', 'Denmark', 'France', 'Germany', 'Spain'],
    'Inhabitants': [85816199, 535065, 654768, 2085679, 2119675, 1322766, 1830212, 1260138, 5795199, 2718352, 6687717, 10341277, 4008617, 5545475, 8796669, 10612086, 10495295, 5056935, 7149077, 10247605, 8958961, 9604000, 19892812, 11686140, 58870763, 41026068, 17618299, 5910913, 64756584, 83294633, 47519628],
    'Pigs': [1.67, 35.79, 65.20, 193.00, 196.14, 274.96, 289.40, 309.76, 403.04, 497.09, 726.94, 736.80, 847.00, 978.42, 1315.50, 1326.44, 1362.28, 1407.61, 2140.99, 2174.82, 2516.46, 2607.70, 3200.10, 5404.14, 9171.00, 9769.70, 10471.00, 11368.00, 11794.05, 21223.70, 33803.04],
    'Surface': [23757, 315, 2586, 25713, 20273, 45399, 64594, 9251, 49035, 65286, 110372, 131957, 56594, 336884, 41291, 438574, 78871, 69825, 77589, 92230, 83878, 93025, 238298, 30528, 301958, 312679, 41543, 42947, 543941, 357581, 498485 ]
}

df = pd.DataFrame(data)
df["Pigs"] = df["Pigs"]*1000
def plot(x,y):
    st.subheader(f"{y} vs {x}")
    fig = px.scatter(df, x=x, y=y)

    # Add annotations
    for i, row in df.iterrows():
        fig.add_annotation(
            x=row[x],
            y=row[y],
            text=row["Country"],
            showarrow=False,
            font=dict(
                color="black",
                size=12,
            )
        )
    st.plotly_chart(fig) 
    
df["pig_sqm"]    = round(df["Pigs"]/df["Surface"],1)

df["inh_sqm"] = round(df["Inhabitants"]/df["Surface"],1)

plot("Inhabitants", "Pigs" )
plot ("inh_sqm", "pig_sqm")
plot("Surface", "Pigs")
st.write(df)

st.subheader("Sources")
st.write("https://ec.europa.eu/eurostat/databrowser/view/apro_mt_lspig/default/table?lang=en")
st.write("https://en.wikipedia.org/wiki/List_of_European_countries_by_population (UN estimate 2023)")
st.write("https://en.wikipedia.org/wiki/List_of_European_countries_by_area)")
