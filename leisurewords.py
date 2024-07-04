import pandas as pd
import streamlit as st

def read_google_sheet(sheet_id: str, sheet_name: str) -> pd.DataFrame:
    """Fetch data from Google Sheets."""
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    try:
        return pd.read_csv(url, delimiter=',')
    except Exception as e:
        st.error(f"Failed to fetch data from Google Sheets| {sheet_name}: {e}")
        st.stop()

def filter_search(df_pivot: pd.DataFrame,search_string:str)->pd.DataFrame():
        # Apply the condition across the DataFrame in a case-insensitive manner
        mask = df_pivot.map(lambda val: search_string.lower() in str(val).lower())

        # Use the mask to filter rows
        return df_pivot[mask.any(axis=1)]
        
def get_data()-> pd.DataFrame:
    """Get the data

    Returns:
        pd.DataFrame: _description_
    """    
    sheet_id = "11V4HnU4o8FJt_9s_9q2RhsuUbei59347j0bWh_j9OqM"
    tblCat =  read_google_sheet(sheet_id,"tblCat") 
    tblTerm=  read_google_sheet(sheet_id,"tblTerm") 
    tblLang=  read_google_sheet(sheet_id,"tblLang") 
    tblSem=  read_google_sheet(sheet_id,"tblSem") 
    tblSite=  read_google_sheet(sheet_id,"tblSite") 

    df = tblTerm.merge(tblSem, on="id_sem", how="outer").merge(tblCat, on="id_cat", how="outer").merge(tblSite, on="id_cat", how="outer").merge(tblLang, on="id_lang", how="outer")

    return df
def convert_df(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

def show_output(df_pivot, languages):
    categories = df_pivot["catentry"].unique().tolist() 
    
    for c in categories:
        df_=df_pivot[df_pivot["catentry"]==c]
        df_=df_[languages]  
        # Drop rows where all values are NaN
        df_ = df_.dropna(how='all') 
        if len(df_)>0:

            df_ = df_.fillna('.')
            st.subheader(c)
            styler =  df_.style.hide(axis="index")
            st.write(styler.to_html(), unsafe_allow_html=True)

def main():
    st.sidebar.title("Leisurewords")
    df = get_data()
    site = st.sidebar.selectbox("Site [camping|restaurant]",["camping","restaurant"],0)
    df=df[df["site"]== site].sort_values(by=["id_sem"])

    
    df_pivot = pd.pivot(df, index=["id_sem","site", "catentry"], columns="ISO3166", values="term").reset_index()
    languages = df_pivot.columns[3:]
    selected_languages = st.sidebar.multiselect("Languages", languages,["NL","EN"])
    if len(selected_languages)==0:
        st.sidebar.error("Choose at least one language")
        st.stop()
    languages_=["catentry"]+selected_languages
    df_pivot=df_pivot[languages_]
    print (df_pivot)
 
    search_string = st.sidebar.text_input("Search string","")
 
    if search_string !="":
        df_pivot = filter_search(df_pivot,search_string)
        
    st.header(f"{site}words")
    if len(df_pivot)==0:
        st.error("No entries found")
        st.stop()

    categories = df_pivot["catentry"].unique().tolist() 
    selected_categories = st.sidebar.multiselect("Categories", categories, categories)
    if len(selected_categories)==0:
        st.sidebar.error("Choose at least one category")
        st.stop()
    else:
        df_pivot =  df_pivot[df_pivot["catentry"].isin(selected_categories)]
    show_output(df_pivot, selected_languages)
    csv = convert_df(df_pivot)
    st.download_button("Click to download", csv, f"{site}words.csv", "text/csv", key=f'download-csv-{site}')



main()