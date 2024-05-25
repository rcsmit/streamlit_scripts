import pandas as pd
import streamlit as st

def read():
    #filetype = 'google_sheet'
    filetype = 'xls'
    #file = 'C:\Users\rcxsm\Documents\pyhton_scripts'

    if filetype == 'csv':
        try:
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
        except:
            st.warning("error met laden")
            st.stop()
    elif filetype == 'xls':
        file = r"input\entry_all_blogs.xlsx"
        sheet = "entry"
        try:
            df = pd.read_excel (file,
                                sheet_name= sheet,
                                header=0,
                                usecols= "a,b,c,d,e,f,g,h,i",
                                names=["id","id_entry_original","titel","kopfoto","artikel","datum","afbeelding","link","blog"])
            #df["datum"] = pd.to_datetime(df["datum"], format="%Y-%m-%d")
            df.datum=pd.to_datetime(df.datum,errors='coerce',format="%d/%m/%Y  hh:mm:ss", dayfirst=True)
            
        except Exception as e:
            st.warning("error met laden")
            st.warning(f"{e}")
            st.warning(traceback.format_exc())
            st.stop()

    elif filetype =='google_sheet':
        sheet_name = "gegevens"
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
        df = pd.read_csv(url, delimiter=',')
        st.write(df.dtypes)
        df["datum"] = pd.to_datetime(df["datum"], format="%d-%m-%Y")
        # df = df[:-1]  #remove last row which appears to be a Nan
    else:
        st.write("type doesnt exist")
        pass
    df['jaar']=df['datum'].dt.strftime('%Y')
    df['maand']=df['datum'].dt.strftime('%m')

    df['maand_']=df['datum'].dt.strftime('%Y-%m')
    return df


def main():
    df = read()
    df = df.fillna('_')

    # Replace 'xxxx' in the 'artikel' column
    df['artikel'] = df['artikel'].replace('http://www.yepcheck.com/printbak/', r'printbak\\', regex=True)
    df['artikel'] = df['artikel'].astype(str).replace('_x000D_','')
    df['artikel'] = df['artikel'].replace('<P>','')
    df['artikel'] = df['artikel'].replace('</P>','/n')
    # Multiselect widget
    options = ["weblog", "fotolog", "petitmonde", "horecalog"]
    selected_blogs = st.sidebar.multiselect("Select blog types", options, options)

    # Filter DataFrame based on selection
    if selected_blogs:
        df = df[df["blog"].isin(selected_blogs)]
    else:
        st.error("Select one of more blogs")
        st.stop()

    for index, row in df.iterrows():
        # st.write(f"Row {index}:")
        # st.write(f"id: {row['id']}")
        # st.write(f"id_entry_original: {row['id_entry_original']}")
        st.subheader(f"{row['titel']}")
        st.write(f"{row['datum']}")
        if row['kopfoto'] != "_":
            image_url = f"printbak\\thumbnails\\{row['kopfoto']}"
           
            st.image(image_url)
        #st.write(f"{row['artikel']}")
        st.markdown(f"{row['artikel']}", unsafe_allow_html = True)
   
        if row['afbeelding'] != "_":
            st.write(f"afbeelding: {row['afbeelding']}")

        if row['link'] != "_":
            st.write(f"link: {row['link']}")
        st.write("\n")
    
if __name__ == "__main__":
    main()
    #find_fill_color("B4")
    
