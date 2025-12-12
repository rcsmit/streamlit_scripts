import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data(ttl=60 * 60 * 24)
def read():
    # https://docs.google.com/spreadsheets/d/1JLioUSN7i--8eiyUHlxOXY9Y0jWSMYKPzu7neyvRc8Y/edit?usp=sharing
    sheet_id = "1JLioUSN7i--8eiyUHlxOXY9Y0jWSMYKPzu7neyvRc8Y"
    sheet_name = "ouder"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

    #url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&sheet={sheet_name}"

    df = pd.read_csv(url, delimiter=',')

    df["datum"] = pd.to_datetime(df["datum"], format="%d-%m-%Y")

    for c in ["afstand totaal", "prijs", "prijs unused"]:    
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["prijs_per_1000km"] = df["prijs"] / df["afstand totaal"] * 1000

    return df



def get_top5_counts(df, countvalue,aantal_rijen):
    return (
        df.groupby(countvalue)
          .size()
          .sort_values(ascending=False)
          .head(aantal_rijen)
          .reset_index(name="aantal")
    )


def get_top5_sum(df, grouper, sumvalue,aantal_rijen):
    # df_x =  df.groupby(grouper)[sumvalue].sum().astype(int).sort_values(ascending=False)
    # st.write(df_x)
    return (
        df.groupby(grouper)[sumvalue]
          .sum().astype(int)
          .sort_values(ascending=False)
          .head(aantal_rijen)
          .reset_index(name=sumvalue)
          
    )


def plot_bar(data, x, y, title):
    fig = px.bar(data, x=x, y=y, text=y, title=title, template="plotly_white")
    fig.update_traces(textposition="outside")
    return fig

def wrapper_top5_sum(df, group, value,aantal_rijen):
    data = get_top5_sum(df, group, value,aantal_rijen)
    st.subheader(f"Top {aantal_rijen} {group} op basis van {value}")
    st.plotly_chart(plot_bar(data, group, value, f"Top {aantal_rijen} {group} – {value}"))

def wrapper_top5_count(df, grouper,aantal_rijen): 
    top5_count = get_top5_counts(df,grouper,aantal_rijen)
    st.subheader(f"Top {aantal_rijen} count op basis van {grouper}")
    st.plotly_chart(plot_bar(top5_count, grouper, "aantal", f"Aantal {grouper}"))
def plot_scatter(df, x, y, title):
    fig = px.scatter(
        df,
        x=x,
        y=y,
        trendline="ols",
        title=title,
        template="plotly_white",
        color="maatschappij",
        hover_data=[
            "maatschappij",
            "bestemming",
            "herkomst",
            "datum",
            
        ]
    )
    return fig

# -----------------------------
# STREAMLIT APP
# -----------------------------

   
def main():
    df = read()
    
    df_used = df[df["DONE OR NOT"] ==1].copy()
    df_unused = df[df["DONE OR NOT"] ==0].copy()
   
    df_used = df_used.sort_values("datum")
    df_used["days_since_last"] = df_used["datum"].diff().dt.days
     # Top 5 vluchten
    wrapper_top5_count(df_used, "maatschappij",50)

    analyses = [
        (df_used, "maatschappij", "afstand totaal",50),
        (df_used, "jaar", "afstand totaal",50),
        (df_used, "jaar", "prijs",50),
        (df_unused, "jaar", "prijs unused",50),
    ]

    for d, g, v,n in analyses:
        wrapper_top5_sum(d, g, v,n)
        
    st.plotly_chart(
        plot_scatter(df_used, "jaar", "prijs_per_1000km", "Prijs per 1000 km per jaar")
    )

    st.subheader("Gemiddelde prijs per 1000 km per maatschappij")
    avg = df_used.groupby("maatschappij")["prijs_per_1000km"].mean().sort_values()
    st.bar_chart(avg)

    st.subheader("Afstand versus prijs scatter")
    fig = px.scatter(
        df_used,
        x="afstand totaal",
        y="prijs",
        trendline="ols",
        hover_data=["maatschappij", "bestemming"],
        template="plotly_white"
    )
    st.plotly_chart(fig)

    st.subheader("Afstand versus prijs per 1000 km scatter")
    fig = px.scatter(
        df_used,
        x="afstand totaal",
        y="prijs_per_1000km",
        trendline="ols",
        hover_data=["maatschappij", "bestemming"],
        template="plotly_white"
    )
    st.plotly_chart(fig)

    st.subheader("Heatmap: prijs per 1000 km vs maatschappij vs jaar")
    pivot = df_used.pivot_table(
        index="maatschappij",
        columns="jaar",
        values="prijs_per_1000km",
        aggfunc="mean"
    )

    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="viridis")
    st.plotly_chart(fig)

    st.subheader("Bestemmingen die je het vaakst deed")
    wrapper_top5_count(df_used, "bestemming", 10)
    wrapper_top5_sum(df_used, "bestemming", "afstand totaal", 10)

    st.subheader("Welke vluchten waren extreem duur per km?")
    st.write(df_used.nlargest(5, "prijs_per_1000km"))

    st.subheader("Boxplots per airline")
    fig = px.box(df_used, x="maatschappij", y="prijs_per_1000km", title="Prijs per 1000 km")
    st.plotly_chart(fig)

    st.subheader("Tijd tussen vluchten")
    

    st.line_chart(
        df_used.set_index("datum")["days_since_last"]
    )

    st.write(df)
    st.write(df_used)
if __name__ == "__main__":
    #caching.clear_cache()
    main()