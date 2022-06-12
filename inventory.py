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
    st.write(df)
    
def main():
    df = read()
    accotype =  st.sidebar.selectbox("Acotype",["Wakiki","Fiji","Sahara","Kalahari 1","Kalahri 2","Serengeti XL","Serengetti L"], index=1)
    accotype_lst = [accotype]
    languages = st.sidebar.multiselect("Languages", ["Nederlands", "English","Deutsch","Italiano","Franҁais"],["Nederlands", "English"])
    st.header(f"Inventory for {accotype} at Camping De Schatberg")
    df[accotype] = df[accotype].fillna(0)
    df = df[df[accotype]>0]
    df[accotype] = df[accotype].astype(int)
    to_show =  languages + accotype_lst
    file_name = "_".join([str(item) for item in to_show])
    df = df[to_show]
    df = df.reset_index(drop=True)
    show_df(df)
    download_button(df, file_name)

if __name__ == "__main__":
    main()