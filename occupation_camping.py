import pandas as pd
import streamlit as st
import numpy as np
def read_csv():
    df = pd.read_csv(
        "masterfinance.csv",
        names=["id","bron","datum","bedrag",
                "tegenpartij","hoofdrub","rubriek"],

        dtype={
            "bron": "category",
            "hoofdrub": "category",
            "rubriek": "category",

        },
        delimiter=';',
        parse_dates=["datum"],
        encoding='latin-1'  ,
        dayfirst=True
    )
def read():
    file = r"C:\Users\rcxsm\Downloads\planning 2019-2022.xlsm"
    sheet = "EXPORT_dummy"
    try:
        df = pd.read_excel(
            file,
            sheet_name=sheet,
            header=0,
            usecols="a,c,d,g,h,i,r",
            names=[
                "datum",
                "number_of_acco",
                "bezet",
                "vertrek_totaal",
                "wissel",
                "new_arrival",
                "acco_type",
            ],
        )
        # df["datum"] = pd.to_datetime(df["datum"], format="%Y-%m-%d")
        df.datum = pd.to_datetime(df.datum, errors="coerce", dayfirst=True)
        df["in_house"] = df["bezet"] + df["wissel"] + df["new_arrival"]
    except Exception as e:
        print("error reading xls file")
    df = make_date_columns(df)
    print(df)
    return df

def make_date_columns(df):
    df["jaar"] = df["datum"].dt.strftime("%Y")
    df["maand"] = df["datum"].dt.strftime("%m").astype(str).str.zfill(2)
    df["dag"] = df["datum"].dt.strftime("%d").astype(str).str.zfill(2)
    df["maand_dag"] = df["maand"] + "-" + df["dag"]
    df["dag_maand"] = df["dag"] + "-" + df["maand"]
    return df

def group_data(df):
    """First we groupby the date. After we make a pivot table with the days on the rows and the different years
    in the columns
           2010  2011  2012
    1-1     .     .     .
    2-1     .     .     .
    3-1     .     .     .
    ..
    31-12   .     .     .

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """    
    df = df[(df["number_of_acco"] != 0)]
    for y in ["2019", "2021", "2022"]:
        df_a = df[(df["jaar"] == y)]
    for c in ["number_of_acco", "bezet", "vertrek_totaal", "wissel", "new_arrival"]:
        df[c] = df[c].astype(int)
    df_grouped_date = df.groupby(["datum"]).sum().reset_index()


    df_grouped_date = make_date_columns(df_grouped_date)
    
    df_grouped_date["in_house"] = (
        df_grouped_date["bezet"]
        + df_grouped_date["wissel"]
        + df_grouped_date["new_arrival"]
    )

    df_grouped_date["arrivals"] = (
        + df_grouped_date["wissel"]
        + df_grouped_date["new_arrival"]
    )

    df_grouped_date["departures"] = (
        df_grouped_date["vertrek_totaal"]
        + df_grouped_date["wissel"]
     
    )

    df_grouped_date["occupation"] = round(
        (df_grouped_date["in_house"] / df_grouped_date["number_of_acco"] * 100).astype(
            float
        ),
        2,
    )
    
    df_pivot = pd.pivot_table(
        df_grouped_date,
        values="occupation",
        index=["maand_dag"],
        columns=["jaar"],
        aggfunc=np.sum,
        fill_value=0,
    ).reset_index()

    df_pivot_in_house = pd.pivot_table(
        df_grouped_date,
        values="in_house",
        index=["maand_dag"],
        columns=["jaar"],
        aggfunc=np.sum,
        fill_value=0,
    ).reset_index()
    

    df_pivot_number_of_acco = pd.pivot_table(
        df_grouped_date,
        values="number_of_acco",
        index=["maand_dag"],
        columns=["jaar"],
        aggfunc=np.sum,
        fill_value=0,
    ).reset_index()

    df_pivot_arrivals = pd.pivot_table(
        df_grouped_date,
        values="arrivals",
        index=["maand_dag"],
        columns=["jaar"],
        aggfunc=np.sum,
        fill_value=0,
    ).reset_index()
    

    df_pivot = df_pivot.sort_values(by=["maand_dag"])
    print(df_pivot)
    return df_pivot, df_pivot_in_house, df_pivot_number_of_acco, df_pivot_arrivals 

def make_graph(df, df_grouped_in_house, df_pivot_number_of_acco, df_pivot_arrivals,  what_to_show_, datefield, acco_type):
    # doesnt work well
    df[datefield] = df[datefield].astype(str)
    df["datum_"] = pd.to_datetime(df[datefield], format="%m-%d")
    data = []
    import plotly.graph_objects as go

    fig = go.Figure()
    title = f"Bezetting {acco_type}"
    for what_to_show_x in what_to_show_:
        try:
            number_of_acco = df_pivot_number_of_acco[what_to_show_x].sum()
            aantal_nachten_bezet = df_grouped_in_house[what_to_show_x].sum()
            aantal_acco = number_of_acco / len(df_pivot_number_of_acco)
            number_of_arrivals = df_pivot_arrivals[what_to_show_x].sum()
            gemiddelde_verblijfsduur = aantal_nachten_bezet / number_of_arrivals
            st.write(f"{what_to_show_x} - Aantal acco's {int(aantal_acco)} - Gemiddelde bezetting {round((aantal_nachten_bezet/number_of_acco*100),1)}% - Gemiddelde verblijfsduur {round(gemiddelde_verblijfsduur,1)} nachten")
            # Aantal nachten bezet {aantal_in_house}
            if what_to_show_x == "2022":
                width = 2
                opacity = 1
            else:
                width = 0.7
                opacity = 0.8
           
            points = go.Scatter(
                name=what_to_show_x,
                x=df["datum_"],
                y=df[what_to_show_x],
                line=dict(width=width),
                opacity=opacity,
                mode="lines",
            )

            data.append(points)

        except:
            # this acco wasnt there in that year
            pass

    layout = go.Layout(
        yaxis=dict(title=f"Bezetting {acco_type} (%)"),
        title=title,
    )
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(xaxis=dict(tickformat="%d-%m"))
    st.plotly_chart(fig, use_container_width=True)

def show_graph_for_selection(df_, choice):
    st.subheader(f"Bezetting voor {choice}")
    if choice != "ALLES":
        df_ = df_[df_["acco_type"] == choice]
    df_grouped, df_grouped_in_house, df_pivot_number_of_acco, df_pivot_arrivals  = group_data(df_)
    make_graph(
        df_grouped, df_grouped_in_house, df_pivot_number_of_acco, df_pivot_arrivals, ["2019", "2021", "2022"], "maand_dag", choice
    )
def select_months(df_):
    (month_from,month_until) = st.sidebar.slider("Months (from/until (incl.))", 1, 12, (1,12))
    if month_from > month_until:
        st.warning("Make sure that the end month is not before the start month")
        st.stop()
    df_ = df_[
        (df_["datum"].dt.month >= month_from) & (df_["datum"].dt.month <= month_until)
             ]
    
    return df_

def main():
    df_ = read()
    
    # df = df_[(df_["maand"] == '07')].copy(deep=True)
    a = ["ALLES"]

    #acco_types = df_["acco_type"].drop_duplicates().to_list()
    acco_types = [  "WAIKIKI",   "BALI",  "SERENGETTI XL",  "SERENGETTI L",  "KALAHARI1",  "KALAHARI2",  "SAHARA",]
 
    acco_types_list = a + acco_types

    choice = st.sidebar.selectbox("Accotype", acco_types_list, index=0)

    df_ = select_months(df_)
    if choice != "ALLES":
        show_graph_for_selection(df_, choice)
    else:
        for a in acco_types_list:
            show_graph_for_selection(df_, a)



if __name__ == "__main__":
    main()
