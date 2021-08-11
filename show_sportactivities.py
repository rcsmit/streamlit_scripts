import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn
import streamlit as st
from helpers import *



def get_data():
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/garminactivities_new.csv"
    url = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\in\\garminactivities_new.csv"
    df = pd.read_csv(url, delimiter=';')
    df["Datum"] = pd.to_datetime(df["Datum"], format="%d-%m-%Y")
    df = df.sort_values(by=['Datum'])
    df["YYYY"] = df["Datum"].dt.year
    df["MM"] = df["Datum"].dt.month
    df["DD"] = df["Datum"].dt.day
    df["count"] = 1
    df = df[df["Activiteittype"] == "Hardlopen"].copy(deep=False)
    df = df[["Datum","Titel", "Afstand","Tijd", "gem_snelh", "count", "MM", "YYYY"]]
    return df

def select(df, select_field, van, tot):
    #df = df[(df[select_field] >= round(van)) & ( df[select_field] <= round(tot) )].copy(deep=False)
    df = df[(df[select_field] >= (van)) & ( df[select_field] <= (tot) )].copy(deep=False)
    return df

def select_maand(df, maand, jaar):
    df = df[(df["MM"] == maand) & ( df["YYYY"] == jaar )].copy(deep=False)
    return df

def show_bar(df, x, what,  title):
    fig, ax = plt.subplots()
    plt.bar(df[x], df[what])
    if title == None:
        plt.title(f"{x} - {what}")
    else:
        plt.title(title)
    plt.grid()
    st.pyplot(fig)
def show_scatter(df, x, what, cat, title):
    s=10 if len(df)<=20 else 3
    #seaborn.set(style='ticks')
    fig, ax = plt.subplots()
    if cat == True:
        cat_ = df['YYYY'].to_numpy()
        #we converting it into categorical data
        cat_col = df['YYYY'].astype('category')
        #we are getting codes for it
        cat_col_ = cat_col.cat.codes
        scatter = plt.scatter(df[x], df[what], s=s, c = cat_col_, label=cat_)
        legend1 = ax.legend(*scatter.legend_elements(),
                 bbox_to_anchor=(1.1, 1), loc='upper left', ncol=1)
        ax.add_artist(legend1)
    else:
        plt.scatter(df[x], df[what], s=s)
    if title == None:
        plt.title(f"{x} - {what}")
    else:
        plt.title(title)
    plt.grid()
    # plt.show()
    st.pyplot(fig)

def show_df(df, heatmap):
    max_value = df.max()

    # #if heatmap == True:
    # st.write(df.style.format(None, na_rep="-").applymap(lambda x:  cell_background_helper(x,"lineair", max_value,None)).set_precision(2))
    # st.write(df.style .background_gradient( vmin=0, vmax=max_value))
    # #else:
    #st.write(df.style.format(None, na_rep="-").set_precision(2))
    st.write(df)


def find_fastest_per_distance(df_):
    fields = ["Datum","Titel", "Afstand","Tijd", "gem_snelh", "YYYY"]
    new_table_list = []
    for y in range (1,30):
        df_temp = select(df_,"Afstand", y-0.1,y+0.1)
        df_temp = df_temp.sort_values(by=['gem_snelh'],ascending= False).reset_index(drop=True)
        my_dict = {"Datum":None,"Titel":None,"Afstand":None,"Tijd":None,"gem_snelh":None, "YYYY":None};
        try:
            for f in fields:
                my_dict[f] = (df_temp.at[0, f])
            new_table_list.append(my_dict)
        except:
            #st.write (f"Nothing for {y}")
            pass # no activities with this distance in this year
    df_pr_of_year = pd.DataFrame(data=new_table_list)
    show_scatter(df_pr_of_year, "Afstand", "gem_snelh", False, "Beste gemiddelde tijd voor de afstand")
    show_df(df_pr_of_year, True)



def find_fastest_per_year(df):
    distance = st.sidebar.slider("Distance", 0,30,5)
    margin = st.sidebar.slider("Distance", 0.0,0.5,0.2,0.05)
    fields = ["Datum","Titel", "Afstand","Tijd", "gem_snelh", "YYYY"]
    df = select(df, "Afstand", distance-margin,distance+margin)
    new_table_list = []
    for y in range (2010,2022):
        df_temp = select(df,"YYYY", y, y)
        df_temp = df_temp.sort_values(by=['gem_snelh'],ascending= False).reset_index(drop=True)
        my_dict = {"Datum":None,"Titel":None,"Afstand":None,"Tijd":None,"gem_snelh":None, "YYYY":None};
        try:
            for f in fields:
                my_dict[f] = (df_temp.at[0, f])
            new_table_list.append(my_dict)
        except:
            pass # no activities with this distance in this year
    df_pr_of_year = pd.DataFrame(data=new_table_list)


    title = (f"Snelste snelheid door de jaren heen op {distance} km")
    show_bar(df_pr_of_year, "YYYY", "gem_snelh", title)

    show_scatter(df, "YYYY", "gem_snelh", False, None)
    show_df(df_pr_of_year, True)


def find_fastest_activities(df):
    # Snelste activiteiten
    df = df.sort_values(by=['gem_snelh'], ascending = False)
    show_df(df.head(25), True)
    #show_df(df_legenda.style.format(None, na_rep="-").applymap(lambda x:  cell_background_helper(x,"lineair", max_value,None)).set_precision(2))


def find_km_per_year(df):
    # Aantal kilometers per jaar
    df_afstand_jaar = df.groupby(["YYYY"]).sum().reset_index()
    #df_afstand_jaar = df_afstand_jaar[["Afstand"]]


    show_bar(df_afstand_jaar, "YYYY", "Afstand", "Afstand per jaar")
    show_bar(df_afstand_jaar, "YYYY", "count", "Aantal per jaar")
    show_df(df_afstand_jaar[["Afstand"]], True)
    show_df(df_afstand_jaar[["count"]], True)

def find_km_per_month_per_year(df):
    # Aantal activiteiten per maand (per jaar)
    df_pivot = df.pivot_table(index='MM', columns='YYYY', values='Afstand',  aggfunc='sum', fill_value=0, margins = True)
    show_df(df_pivot, True)

def find_nr_activities_per_month_per_year(df):
    # Aantal activiteiten per maand (per jaar)
    df_pivot = df.pivot_table(index='MM', columns='YYYY', values='count',  aggfunc='sum', fill_value=0, margins = True)
    st.write(df_pivot.dtypes)
    show_df(df_pivot, True)


def find_avg_km_avg_speed_per_year(df):
    # Gemiddelde afstand en snelheid per jaar
    df_mean = df.groupby(["YYYY"]).mean()
    df_mean = df_mean[["Afstand","gem_snelh"]]
    show_scatter(df_mean, "Afstand", "gem_snelh", False, "Gemiddelde afstand en snelheid per jaar")
    show_df(df_mean, True)

def find_activities_in_month(df):
    # Activeitein in een bepaalde maand
    month = st.sidebar.slider("Maand", 7,12,1)
    year = st.sidebar.slider("Jaar", 2010,2021,2021)


    df_maand_jaar = select_maand(df, month, year)
    show_df(df_maand_jaar, True)
def find_biggest_distances(df):
    # Verste activiteiten
    df = df.sort_values(by=['Afstand'], ascending = False)
    show_df(df, True)

def show_various_scatters(df):
    # Verschillende scatterplots
    show_scatter(df, "Datum", "gem_snelh", False, None)
    show_scatter(df, "Datum", "Afstand", False, None)
    show_scatter(df, "Afstand", "gem_snelh", True, None)

def main():
    df = get_data().copy(deep=False)
    lijst = ["find km per year",
            "find fastest per distance",
            "find fastest per year",
            "find fastest activities",       
            #"find km per month per year",
            #"find nr activities per month per year",
            "find avg km avg speed per year",
            "find activities in month",
            "show various scatters",
            "find biggest distances"]

    functies = [ find_fastest_per_distance ,
        find_km_per_year ,
        find_fastest_per_year ,
        find_fastest_activities ,     
        #find_km_per_month_per_year ,
        #find_nr_activities_per_month_per_year,
        find_avg_km_avg_speed_per_year ,
        find_activities_in_month ,
        show_various_scatters ,
        find_biggest_distances ]
    st.sidebar.subheader("Menu")
    menu_choice = st.sidebar.radio("",lijst, index=0)
    for i, choice in enumerate(lijst):
        if menu_choice == choice:
            st.header(lijst[i])
            functies[i](df)

if __name__ == "__main__":
    main()
