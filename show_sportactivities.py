import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn
import streamlit as st
from helpers import *
#from streamlit import caching
import numpy as np
import matplotlib.animation as animation

import plotly.express as px

def get_data(who):
    if who == "Rene":
        url_oud = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/garminactivities_new.csv"
        url_2022 = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/garminactivities_2022.csv"
        url_2023a = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/garminactivities_2023a.csv"
        
        #url = "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\streamlit_scripts\\input\\garminactivities_new.csv"
        df_oud = pd.read_csv(url_oud, delimiter=';')
        df_2022 = pd.read_csv(url_2022, delimiter=',')
        df_2023a = pd.read_csv(url_2023a, delimiter=',')
        
       
        for d in [df_oud, df_2022,df_2023a]:
            create_extra_date_time_columns(d)
       
        
        for d in [df_2022,df_2023a]:
            # todo : rename the columns in df_oud
            df_2022 = rename_columns(df_2022)
            df_2023a = rename_columns(df_2023a)
        
       
        
        df_tm_2022 = df_2022.append(df_oud, ignore_index=False)

        df = df_20223a.append(df_tm_2022, ignore_index=False)

       

        df = filter_df(df, "Activiteittype",1).copy(deep=False)
        
    elif who == "Didier":
        url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/activities_didier.csv"
        #url = "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\streamlit_scripts\\input\\activities_didier.csv"

        df = pd.read_csv(url, delimiter=',')
        df = filter_df(df, "Activity Type",5).copy(deep=False)
        prepare_df_didier(df)
    else:
        st.error("Error in who")
        st.stop()
    df = last_manipulations_df(df).copy(deep=False)

    return df

def create_extra_date_time_columns(df):
    df['Datum_x'] = pd.to_datetime(df['Datum']).dt.date
    df["Datum_xy"] = pd.to_datetime(df["Datum"], format="%Y-%m-%d %H:%M:%S")
        
    df["Tijd_xy"] = pd.to_datetime(df["Datum"], format="%Y-%m-%d %H:%M:%S")
    df['Tijd_x'] = pd.to_datetime(df['Tijd_xy']).dt.strftime('%H:%M:%S')
    df['Tijd_y'] = pd.to_datetime(df['Tijd']).dt.time
    df['Tijd_h'] = pd.to_datetime(df['Tijd']).dt.hour
    df['Tijd_m'] = pd.to_datetime(df['Tijd']).dt.minute
    df['Tijd_s'] = pd.to_datetime(df['Tijd']).dt.second
    df['Tijd_seconds'] = (df['Tijd_h']*3600) + (df['Tijd_m']*60) + df['Tijd_s'] 
    df['gem_snelh'] = df['Afstand'] / df['Tijd_seconds'] * 3600.0 

def rename_columns(df):
    df['Gem_HS'] = df['Gem. HS']
    df['Max_HS'] = df['Max. HS']
    df["Gem_loopcadans"] = df['Gem. loopcadans']
    df["Max_loopcadans"] = df['Max. loopcadans']
    df["Gemiddeld_tempo"] = df['Gemiddeld tempo']
    df["gem_snelh"] = None
    df = df[["Activiteittype","Leeg","Datum_xy","Tijd_x","Titel","Afstand","Tijd","gem_snelh","Gem_HS","Max_HS","Gem_loopcadans","Max_loopcadans","Gemiddeld_tempo"]]
    return df

def last_manipulations_df(df):
    print (df.dtypes)
    df = df.sort_values(by=['Datum_xy'])
    df["YYYY"] = df["Datum_xy"].dt.year
    df["MM"] = df["Datum_xy"].dt.month
    df["DD"] = df["Datum_xy"].dt.day
    df["count"] = 1
    df = df[["Datum_xy","Titel", "Tijd_h","Tijd_m","Tijd_s","Tijd_seconds", "Afstand","Tijd", "gem_snelh", "count", "MM", "YYYY"]]

 
    return df

def prepare_df_didier(df):
    df["Datum"] = pd.to_datetime(df["Activity Date"], format="%b %d, %Y, %H:%M:%S %p")
    df["gem_snelh"] = df["Distance"].astype(float) / df["Elapsed Time"]*3600
    df["Afstand"] = df["Distance"].astype(float)
    df["hh"] = (df["Elapsed Time"]/3600).astype(int)
    df["mm"] = ((df["Elapsed Time"] - (df['hh']*3600))/60).astype(int)
    df["ss"] = (df["Elapsed Time"] - (df['hh']*3600) - (df['mm']*60)).astype(int)
    for xx in ["hh", "mm", "ss"]:
        df[xx] = (df[xx]).astype(int).astype(str).str.zfill(2)
    df["Tijd"]  = df['hh'] + ":"+ df['mm'] +":"+ df['ss']
    df["Titel"] = df [ "Activity Name"]

def filter_df(df, veldnaam, default):
    act_type_list =  df[veldnaam].drop_duplicates().sort_values().tolist()
    act_type = st.sidebar.selectbox("Welke activiteitssoort",act_type_list, default)
    df = df[df[veldnaam] == act_type]

    return df




def calculate_average_speed(df):
    # CALCULATE AVERAGE SPEED, not used atm
    df["Tijd"]= df["Tijd"].str.zfill(8)
    df["hh"] = df["Tijd"].str[:2].astype(int)
    df["mm"] = df["Tijd"].str[3:5].astype(int)
    df["ss"] = df["Tijd"].str[-2:].astype(int)
    df["snelh_new"] = round(((3600 / (df["hh"] * 3600 + df["mm"] *60 + df["ss"]))*df[ "Afstand"]),2)
    # df = df[round(df["snelh_new"]) != round(df["gem_snelh"])] #CHECK IF THERE ARE DIFFERNCES WITH THE GIVEN SPEED
    return df

def in_between(df):
    print (df.dtypes)
    what = st.sidebar.selectbox("Wat", [ "Tijd_h","Tijd_m","Tijd_seconds", "Afstand","Tijd", "gem_snelh","YYYY", "MM" ], index=3) 
    min = df[what].min().astype(int) -1
    max = df[what].max().astype(int) +1
    van = st.sidebar.number_input("From", 0, max, 0)
    tot = st.sidebar.number_input("Until (incl.)", 0, max, max)
    # st.write(min,max)
    # (van,tot)=  st.sidebar.slider("van", 0,100,value = (0,100))
    # tot=  st.sidebar.slider("tot", 0,9999,1)
    if what == "Tijd_m":
        df = df[(df["Tijd_h"] == 0)].copy(deep=False)
        
    df = select(df, what, van, tot)
    df = df.sort_values(by=[what])
    st.write(f"Aantal activiteiten {len(df)}")
    
    st.write(f"Totale afstand {df['Afstand'].sum()}" )
    
    seconds= df["Tijd_seconds"].sum()
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    ts = f'{h:d}:{m:02d}:{s:02d}'


    st.write(f"Totale tijd {ts}" )
    st.write(df)

def select(df, select_field, van, tot):
    #df = df[(df[select_field] >= round(van)) & ( df[select_field] <= round(tot) )].copy(deep=False)
    df = df[(df[select_field] >= (van)) & ( df[select_field] <= (tot) )].copy(deep=False)
    return df

def select_maand(df, maand, jaar):
    df = df[(df["MM"] == maand) & ( df["YYYY"] == jaar )].copy(deep=False)
    return df

def show_bar(df, x, what,  title):
    if title == None:
        title = (f"{x} - {what}")
    fig = px.bar(df, x=x, y=what,title=title)
    st.plotly_chart(fig)

   
def show_scatter(df, x, what, cat, title):
    s=10 if len(df)<=20 else 3
    if title == None:
        title = (f"{x} - {what}")
    #seaborn.set(style='ticks')
    
    if cat == True:
        fig, ax = plt.subplots()
        cat_ = df['YYYY'].to_numpy()
        #we converting it into categorical data
        cat_col = df['YYYY'].astype('category')
        #we are getting codes for it
        cat_col_ = cat_col.cat.codes
        scatter = plt.scatter(df[x], df[what], s=s, c = cat_col_, label=cat_)
        legend1 = ax.legend(*scatter.legend_elements(),
                 bbox_to_anchor=(1.1, 1), loc='upper left', ncol=1)
        ax.add_artist(legend1)
        plt.title(title)
        plt.grid()
        # plt.show()
        st.pyplot(fig)
    else:
        fig = px.scatter(df, x=x, y=what,title=title)
        st.plotly_chart(fig)

def show_df(df, heatmap, title):
    max_value = df.max()
    st.write(title)
    st.write(df.style.format(None, na_rep="-", precision=2))

    # if heatmap == True:
    #     st.write(f"Heatmap {max_value}")
    #     st.write(df.style.format(None, na_rep="-", precision=2).applymap(lambda x:  cell_background_helper(x,"lineair", max_value,None)))
    # else:
    #     st.write(df.style.format(None, na_rep="-", precision=2))


def find_fastest_per_distance(df_):
    
    fields = ["Datum_xy","Titel", "Afstand","Tijd", "gem_snelh", "YYYY"]
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
            pass # no activities with this distance
    df_pr_of_year = pd.DataFrame(data=new_table_list)

    
    show_scatter(df_pr_of_year, "Afstand", "gem_snelh", False, "Beste gemiddelde snelheid voor de afstand")
    show_df(df_pr_of_year, True, "Beste gemiddelde snelheid voor de afstand")

def find_pr_of_year(df, field):
    fields = ["Datum_xy","Titel", "Afstand","Tijd", "gem_snelh", "YYYY"]
    new_table_list = []
    for y in range (2010,2025):
        df_temp = select(df,"YYYY", y, y)
        df_temp = df_temp.sort_values(by=[field],ascending= False).reset_index(drop=True)
        my_dict = {"Datum":None,"Titel":None,"Afstand":None,"Tijd":None,"gem_snelh":None, "YYYY":None};
        try:
            for f in fields:
                my_dict[f] = (df_temp.at[0, f])
            new_table_list.append(my_dict)
        except:
            pass # no activities with this distance in this year
    df_pr_of_year = pd.DataFrame(data=new_table_list)
    title = (f"Beste van {field} door de jaren heen")
    show_bar(df_pr_of_year, "YYYY", field, title)
    show_scatter(df, "YYYY", field, False, None)
    show_df(df_pr_of_year, True, title)


def find_fastest_per_year(df):
    distance = st.sidebar.slider("Distance", 0,30,5)
    margin = st.sidebar.slider("Distance", 0.0,0.5,0.2,0.05)
    fields = ["Datum","Titel", "Afstand","Tijd", "gem_snelh", "YYYY"]
    df = select(df, "Afstand", distance-margin,distance+margin)
    find_pr_of_year(df, "gem_snelh")


def find_fastest_activities(df):
    # Snelste activiteiten
    df = df.sort_values(by=['gem_snelh'], ascending = False)
    show_df(df.head(25), True, "Snelste activiteiten")
    #show_df(df_legenda.style.format(None, na_rep="-").applymap(lambda x:  cell_background_helper(x,"lineair", max_value,None)).set_precision(2))
    find_pr_of_year(df, "gem_snelh")


def find_km_per_year(df):
    # Aantal kilometers per jaar
    df_afstand_jaar = df.groupby(["YYYY"]).sum().reset_index()
    #df_afstand_jaar = df_afstand_jaar[["Afstand"]]
    df_afstand_jaar["afstand_per_keer_per_jaar"] = df_afstand_jaar["Afstand"] / df_afstand_jaar["count"]


    show_bar(df_afstand_jaar, "YYYY", "Afstand", "Afstand per jaar")
    show_bar(df_afstand_jaar, "YYYY", "count", "Aantal per jaar")
    show_bar(df_afstand_jaar, "YYYY", "afstand_per_keer_per_jaar", "Km per activiteit per jaar")

    show_df(df_afstand_jaar[["YYYY", "Afstand"]], True, "Afstand per jaar")
    show_df(df_afstand_jaar[["YYYY", "count"]], True, "Aantal keren per jaar")

    show_df(df_afstand_jaar[["YYYY", "afstand_per_keer_per_jaar"]], True, "Afstand per keer per jaar")


def find_km_per_month_per_year(df):
    # Aantal activiteiten per maand (per jaar)
    df["MM"] = df["MM"].astype(str).str.zfill(2)

    df_pivot = df.pivot_table(index='MM', columns='YYYY', values='Afstand',  aggfunc='sum', fill_value=0, margins = True)
    show_df(df_pivot, True, "km per maand per jaar")

def find_nr_activities_per_month_per_year(df):
    # Aantal activiteiten per maand (per jaar)
    df["MM"] = df["MM"].astype(str).str.zfill(2)
    df_pivot = df.pivot_table(index='MM', columns='YYYY', values='count',  aggfunc='sum', fill_value=0, margins = True)

    show_df(df_pivot, True, "Activiteiten per maand per jaar")


def find_avg_km_avg_speed_per_year(df):
    # Gemiddelde afstand en snelheid per jaar
    df_mean = df.groupby(["YYYY"]).mean()
    df_mean = df_mean[["Afstand","gem_snelh"]]
    show_scatter(df_mean, "Afstand", "gem_snelh", False, "Gemiddelde afstand en snelheid per jaar")
    show_df(df_mean, True, "gemiddelde afstand vs gem snelheid per jaar")
def show_all(df):
    st.write(df)
def find_activities_in_month(df):
    # Activeitein in een bepaalde maand
    month = st.sidebar.slider("Maand", 7,12,1)
    year = st.sidebar.slider("Jaar", 2010,2021,2021)


    df_maand_jaar = select_maand(df, month, year)
    months = [
                    "januari",
                    "februari",
                    "maart",
                    "april",
                    "mei",
                    "juni",
                    "juli",
                    "augustus",
                    "september",
                    "oktober",
                    "november",
                    "december",
                ]
    show_df(df_maand_jaar, True, f"Activiteiten in {months[month-1]} {year}")
def find_biggest_distances(df):
    # Verste activiteiten
    df = df.sort_values(by=['Afstand'], ascending = False)
    show_df(df, True, "Grootste afstand")
    find_pr_of_year(df, "Afstand")

def show_various_scatters(df):
    # Verschillende scatterplots
    show_scatter(df, "Datum_xy", "gem_snelh", False, None)
    show_scatter(df, "Datum_xy", "Afstand", False, None)
    show_scatter(df, "Afstand", "gem_snelh", True, None)

def plot_histogram_distance_year(df):
    y = st.sidebar.slider("Jaar", 2010,2021,2021)
    df_temp = select(df,"YYYY", y,y)
    bins = np.arange(min(df_temp["Afstand"]), max(df_temp["Afstand"])+1 ,1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.hist(df_temp["Afstand"], bins = bins , density=False, alpha=0.5)

    st.pyplot (fig)



def plot_histogram_distance_year_animated(df):
    st.title("Under construction")
    bins_animated = np.arange(0.5,30.5 ,1)

    def update(curr):
        y = 2020+curr
        df_temp = select(df,"YYYY", y, y)
        data = df_temp["afstand"]
        if curr == 2021:
            a.event_source.stop()
        plt.cla()
        plt.title(y)
        plt.hist(data, bins = bins_animated)

    fig = plt.figure()
    a = animation.FuncAnimation(fig, update, interval = 11)

    st.pyplot(fig)

def main():
    who  = st.sidebar.selectbox("Wie",["Didier", "Rene"], index=1)
    df = get_data(who).copy(deep=False)
    lijst = ["find km per year",
            "find fastest per distance",
            "find fastest per year for a distance",
            "find fastest activities",
            "find biggest distances",
            "find km per month per year",
            "find nr activities per month per year",
            "find avg km avg speed per year",
            "plot histogram distance year",
            "plot histogram distance year animated",
            "show various scatters",
            "find activities in certain month",
            "show all activities",
            "in between"
            ]

    functies = [ find_km_per_year ,
        find_fastest_per_distance ,
        find_fastest_per_year ,
        find_fastest_activities ,
        find_biggest_distances,
        find_km_per_month_per_year ,
        find_nr_activities_per_month_per_year,
        find_avg_km_avg_speed_per_year ,
        plot_histogram_distance_year,
        plot_histogram_distance_year_animated,
        show_various_scatters ,
        find_activities_in_month ,
        show_all,
        in_between
         ]
    st.sidebar.subheader("Menu")
    menu_choice = st.sidebar.radio("",lijst, index=0)
    for i, choice in enumerate(lijst):
        if menu_choice == choice:
            st.header(lijst[i])
            functies[i](df)

if __name__ == "__main__":
    main()
