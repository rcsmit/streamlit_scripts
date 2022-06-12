import pandas as pd
import streamlit as st

def read():
    sheet_id = "1toDWxbZwLg4qyLnsjnmKnA_V5q_4yAqnkAsH0W4FiTY"
    sheet_name = "Schatberg2022"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    df = pd.read_csv(url, delimiter=',')
    return df

def save_df(df, name):
    """  Saves the df """
    name_ =  name + ".csv"
    compression_opts = dict(method=None, archive_name=name_)
    df.to_csv(name_, index=False, compression=compression_opts)
    print("--- Saving " + name_ + " ---")

@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

def download_button(df, file_name):    
    csv = convert_df(df)
    st.sidebar.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=f'inventory_{file_name}.csv',
        mime='text/csv',)

def show_df(df):
    # CSS to inject contained in a string
    hide_dataframe_row_index = """
            <style>
            .row_heading.level0 {display:none; width:0px}
            .blank {display:none;width:0px}
            </style>
            """
    # Inject CSS with Markdown
    st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
    #st.write(df)
    st.table(df)

def show_link():
    url = "https://docs.google.com/spreadsheets/d/1toDWxbZwLg4qyLnsjnmKnA_V5q_4yAqnkAsH0W4FiTY/edit#gid=353184161"
    url_ = f"<a href='{url}' target='_blank'>Link to Google Sheet</a>"
    st.sidebar.markdown(url_, unsafe_allow_html=True)

def show_disclaimer(languages_possible, languages_chosen):
    disclaimer_nl = "Lijst is indicatief en niet juridisch bindend"
    disclaimer_en = "List is indicative and not legally binding"
    disclaimer_fr = "La liste est indicative et non juridiquement contraignante"
    disclaimer_de = "Die Liste ist indikativ und nicht rechtlich bindend"
    disclaimer_it = "L'elenco è indicativo e non giuridicamente vincolante"
    disclaimer_dk = "Listen er vejledende og ikke juridisk bindende"
    disclaimer_pl = "Lista ma charakter orientacyjny i nie jest prawnie wiążąca"
    disclaimers = [disclaimer_nl, disclaimer_en, disclaimer_fr, disclaimer_de, disclaimer_it, disclaimer_dk, disclaimer_pl
    ]
    for l in languages_possible:
        if l in languages_chosen:
            index = languages_possible.index(l)
            st.write(f" * {disclaimers[index]}")


def main():
    accotype_possible = ["Waikiki","Fiji","Sahara","Kalahari 1","Kalahri 2","Serengeti XL","Serengetti L"]
    languages_possible = ["Nederlands", "English","Deutsch","Italiano","Franҁais", "Dansk", "Polski"]
    
    accotype_chosen =  st.sidebar.multiselect("Accotype",accotype_possible, "Waikiki")
    languages_chosen = st.sidebar.multiselect("Languages", languages_possible ,["Nederlands", "English"])
    
    accotype_str = " & ".join([str(item) for item in accotype_chosen])
    st.header(f"Inventory for {accotype_str} at Camping De Schatberg")
    
    df = read()
    
    for a in accotype_chosen:
        df[a] = df[a].fillna(0)
        if (len(accotype_chosen) ==1): #TO DO : filter out when multiple acco's have 0 pcs of a certain item
            df = df[df[a]>0] 
        df[a] = df[a].astype(int)
    to_show =  languages_chosen + accotype_chosen
    file_name = "_".join([str(item) for item in to_show])
    df = df[to_show]
    df = df.reset_index(drop=True)
    show_df(df)
    show_disclaimer(languages_possible, languages_chosen)

    # sidebar
    download_button(df, file_name)
    show_link()
    
if __name__ == "__main__":
    main()