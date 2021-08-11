import pandas as pd
import streamlit as st


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

def find_nr_activities_per_month_per_year(df):
    # Aantal activiteiten per maand (per jaar)
    print (df.dtypes)
    df_pivot = df.pivot_table(index='MM', columns='YYYY', values='count',  aggfunc='sum', fill_value=0, margins = True)
    st.write(df_pivot)
    print (df_pivot)

def main():
    df = get_data().copy(deep=False)

    find_nr_activities_per_month_per_year(df)


if __name__ == "__main__":
    main()
