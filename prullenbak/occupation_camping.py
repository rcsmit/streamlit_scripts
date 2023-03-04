# VEROUDERD. GEBRUIK "read_bezetting.py"
# DIT BESTAND GEBRUIKT DE TOTALEN DIE DOOR EXCEL WORDEN GEGENEREERD (adh het optellen van de kleuren in een kolom)


import pandas as pd
import streamlit as st
import numpy as np
def read_csv():

    url="https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/planning_2019-2022_dummy.csv"
    df_ = pd.read_csv(
        url,


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
    df_ = make_date_columns(df_)
    df_["in_house"] = df_["bezet"] + df_["wissel"] + df_["new_arrival"]
  
    url_prijzen = f"https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/prijzen_dummy.csv"

    df_prijzen = pd.read_csv(url_prijzen, delimiter=';')
    
    #df_prijzen_stacked = df_prijzen.stack()
    df_prijzen_stacked = df_prijzen.melt('acco_type', var_name='maand_int', value_name='price_per_night')
  
    #.set_index('acco_type').stack().rename(columns={'price_per_night':'month'})
    #df_["maand_str"] = df_["maand_int"].astype(str)
    df_prijzen_stacked["maand_str"] = df_prijzen_stacked["maand_int"].astype(str)
    
    df = pd.merge(df_, df_prijzen_stacked,how="outer", left_on=["acco_type", "maand"], right_on=["acco_type","maand_int"])

    df["omzet"] = df["in_house"] * df["price_per_night"]
   

    return df

def read_google_sheets():
    sheet_id = st.secrets["google_sheet_occupation"]
    sheet_name = "EXPORT"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

    df_ = pd.read_csv(url, delimiter=',')
    df_["in_house"] = df_["bezet"] + df_["wissel"] + df_["new_arrival"]
    df_ = make_date_columns(df_)
    df_["maand_int"] = df_["datum"].dt.strftime("%m").astype(int)

    sheet_name_prijzen = "prijzen"
    url_prijzen = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name_prijzen}"

    df_prijzen = pd.read_csv(url_prijzen, delimiter=',')
    #df_prijzen_stacked = df_prijzen.stack()
    df_prijzen_stacked = df_prijzen.melt('acco_type', var_name='maand_int', value_name='price_per_night')
    #.set_index('acco_type').stack().rename(columns={'price_per_night':'month'})
    df_["maand_str"] = df_["maand_int"].astype(str)
    df_prijzen_stacked["maand_str"] = df_prijzen_stacked["maand_int"].astype(str)
    df = pd.merge(df_, df_prijzen_stacked,how="outer", on=["acco_type","maand_str"])

    df["omzet"] = df["in_house"] * df["price_per_night"]
   
    
    
    return df



def read():
    file = r"C:\Users\rcxsm\Downloads\planning 2019-2022.xlsm"
    sheet = "EXPORT"
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

    df['datum'] = pd.to_datetime(df.datum, format='%d-%m-%Y')
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
    df =  make_date_columns(df) 
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
    
    df_pivot_omzet = pd.pivot_table(
        df_grouped_date,
        values="omzet",
        index=["maand_dag"],
        columns=["jaar"],
        aggfunc=np.sum,
        fill_value=0,
    ).reset_index()
    

    df_pivot = df_pivot.sort_values(by=["maand_dag"])
    print(df_pivot)
    return df_pivot, df_pivot_in_house, df_pivot_number_of_acco, df_pivot_arrivals, df_pivot_omzet, df_grouped_date 

def make_graph(df, df_grouped_in_house, df_pivot_number_of_acco, df_pivot_arrivals,df_pivot_omzet,  what_to_show_, datefield, acco_type, prijs_per_nacht_fixed):
    # doesnt work well
    df[datefield] = df[datefield].astype(str)
    df["datum_"] = pd.to_datetime(df[datefield], format="%m-%d")
    data = []
    import plotly.graph_objects as go

    fig = go.Figure()
    title = f"Bezetting {acco_type}"
    df_info =  pd.DataFrame()
    for what_to_show_x in what_to_show_:
        try:
            number_of_acco = df_pivot_number_of_acco[what_to_show_x].sum()
            aantal_nachten_bezet = df_grouped_in_house[what_to_show_x].sum()
            totale_omzet = df_pivot_omzet[what_to_show_x].sum()
            omzet_fixed_price =round(aantal_nachten_bezet * prijs_per_nacht_fixed)
            aantal_acco = number_of_acco / len(df_pivot_number_of_acco)
            number_of_arrivals = df_pivot_arrivals[what_to_show_x].sum()
            gem_prijs_per_nacht = totale_omzet / aantal_nachten_bezet
            gemiddelde_verblijfsduur = aantal_nachten_bezet / number_of_arrivals
            what_to_show_x_int = int(what_to_show_x)
            df__ =  pd.DataFrame([ {
                                "year" : what_to_show_x_int,
                                "Aantal_accos": int(aantal_acco),
                                "Gemiddelde_bezetting_(%)" : round((aantal_nachten_bezet/number_of_acco*100),1),
                                "Gemiddelde_verblijfsduur_(nacht)" : round(gemiddelde_verblijfsduur,1),
                                "Aantal_nachten_bezet" : aantal_nachten_bezet,
                               
                                "gemiddelde_prijs_per_nacht_(euro)" : round(gem_prijs_per_nacht,2),
                                "omzet_(euro)" : totale_omzet, 

                                 }]
                            )           

            df_info = pd.concat([df_info, df__],axis = 0)   
            
            
            
            # st.write(f"{what_to_show_x} - Aantal acco's {int(aantal_acco)} - Gemiddelde bezetting {round((aantal_nachten_bezet/number_of_acco*100),1)}% - Gemiddelde verblijfsduur {round(gemiddelde_verblijfsduur,1)} nachten")
            # st.write(f"Aantal nachten bezet {aantal_nachten_bezet} - omzet (via prijslijst) {totale_omzet:_}  - gemiddelde prijs per nacht (via prijslijst) {round(gem_prijs_per_nacht,2)}")
            # # - omzet (via fixed price) {omzet_fixed_price}
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
   
    df_info = df_info.set_index("year").transpose()
    st.write (df_info.style.format("{:.2f}"))

    layout = go.Layout(
        yaxis=dict(title=f"Bezetting {acco_type} (%)"),
        title=title,
    )
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(xaxis=dict(tickformat="%d-%m"))
    st.plotly_chart(fig, use_container_width=True)

def show_graph_for_selection(df_, choice,prijs_per_nacht_fixed):
    st.subheader(f"Bezetting voor {choice}")
    if choice != "ALLES":
        df_ = df_[df_["acco_type"] == choice]
    df_grouped, df_grouped_in_house, df_pivot_number_of_acco, df_pivot_arrivals,df_pivot_omzet,df_grouped_date  = group_data(df_)
    make_graph(
        df_grouped, df_grouped_in_house, df_pivot_number_of_acco, df_pivot_arrivals,df_pivot_omzet, ["2019", "2021", "2022"], "maand_dag", choice, prijs_per_nacht_fixed
    )
    if choice == "ALLES":
        col1,col2= st.columns(2)
        with col1:
            year ="2019"
            show_count( df_grouped_date,"arrivals", year)
        with col2:
            year ="2021"
            show_count( df_grouped_date,"arrivals", year)
     
        year ="2022"
        show_count( df_grouped_date,"arrivals", year)
        col1,col2= st.columns(2)
        with col1:
            year ="2019"
            show_count( df_grouped_date,"departures", year)
        with col2:
            year ="2021"
            show_count( df_grouped_date,"departures", year)
        year ="2022"
        show_count( df_grouped_date,"departures", year)
     
 
def show_count( df_grouped_date, what, year):
    print(df_grouped_date)
    if year != None:
        df_grouped_date_selection = df_grouped_date[df_grouped_date["jaar"]==year]
        title=f"Days with most {what} in {year}"
    else:
        df_grouped_date_selection = df_grouped_date
        title=f"Days with most {what}"
    what_df = df_grouped_date_selection[["dag_maand", what]].sort_values(what, ascending=False)
    what_df =  what_df[what_df[what]!=0]
    st.subheader(title)
    st.write(what_df.head(30))
    import plotly.express as px
    max_value =int(what_df[what].max())
    sum = what_df[what].sum()
    st.write (f"Total {sum}")
    fig = px.histogram(what_df, x=what, nbins=max_value)
    fig.update_layout(
    title_text=title, # title of plot
    xaxis_title_text='Value', # xaxis label
    yaxis_title_text='Count', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1 # gap between bars of the same location coordinates
)

    #fig.show()
    st.plotly_chart(fig, use_container_width=True)
    

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
    pw = st.sidebar.text_input("Password", "****", type="password")
    if pw == st.secrets["PASSWORD"]:
        st.sidebar.write("Pasword ok")
        df_ = read_google_sheets()
    else:
        st.sidebar.write("Enter the right password. Showing dummy data.")
        df_ = read_csv()
    
    # df = df_[(df_["maand"] == '07')].copy(deep=True)
    a = ["ALLES"]

    #acco_types = df_["acco_type"].drop_duplicates().to_list()
    acco_types = [  "WAIKIKI",   "BALI",  "SERENGETTI XL",  "SERENGETTI L",  "KALAHARI1",  "KALAHARI2",  "SAHARA",]
 
    acco_types_list = a + acco_types

    choice = st.sidebar.selectbox("Accotype", acco_types_list, index=0)
    prijs_per_nacht_fixed = st.sidebar.number_input("Prijs per nacht", 0,1000,120)
    df_ = select_months(df_)
    if choice != "ALLES":
        show_graph_for_selection(df_, choice, prijs_per_nacht_fixed)
    else:
        for a in acco_types_list:
            show_graph_for_selection(df_, a, prijs_per_nacht_fixed)



if __name__ == "__main__":
    main()