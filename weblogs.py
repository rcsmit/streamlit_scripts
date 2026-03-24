import pandas as pd
import streamlit as st

@st.cache_data()
def read():
    filetype = 'google_sheet'

    if filetype =='google_sheet':
        sheet_name = "gegevens"
        sheet_id = "1R5YDxVqpT1brUHz1P-Zjoyz0iHgLBJUIsSrbH9IfW5c"
        # https://docs.google.com/spreadsheets/d/1R5YDxVqpT1brUHz1P-Zjoyz0iHgLBJUIsSrbH9IfW5c/edit?usp=sharing
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
        df = pd.read_csv(url, delimiter=',', header=0,
                                 usecols=list(range(10)),
                                names=["id","id_entry_original","titel","kopfoto","artikel","datum","afbeelding","link","blog","categorie"])
      
        try:
            df["datum"] = pd.to_datetime(df["datum"], format="%d-%m-%Y")
        except:
            df["datum"] = pd.to_datetime(df["datum"], format='mixed')
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
    df['artikel'] = df['artikel'].replace('http://www.yepcheck.com/printbak/', r'https://github.com/rcsmit/streamlit_scripts/tree/main/printbak/', regex=True)
    df['artikel'] = df['artikel'].astype(str).replace('_x000D_','')
    df['artikel'] = df['artikel'].replace('<P>','')
    df['artikel'] = df['artikel'].replace('</P>','/n')
    # Multiselect widget
    options = df["blog"].unique().tolist()
    selected_blogs = st.sidebar.multiselect("Select blog types", options, options)

    # Filter DataFrame based on selection
    if selected_blogs:
        df = df[df["blog"].isin(selected_blogs)]

    else:
        st.error("Select one of more blogs")
        st.stop()
    if (len(selected_blogs)==1 and ((selected_blogs == ["CrazyWaiter"]) or (selected_blogs==["YepYoga"]))):
        categories= df["categorie"].unique().tolist()
        selected_categories = st.sidebar.multiselect("Select categories", categories, categories)

        df = df[df["categorie"].isin(selected_categories)]
        if len(df)==0:
            st.error("Choose a category")
    for index, row in df.iterrows():
        # st.write(f"Row {index}:")
        # st.write(f"id: {row['id']}")
        # st.write(f"id_entry_original: {row['id_entry_original']}")
        st.subheader(f"{row['titel']}")
        st.write(f"{row['datum']}")
        if row['categorie'] != "_":
            st.write(f"Categorie: {row['categorie']}")  
        if row['kopfoto'] != "_":
            #image_url = f"https://github.com/rcsmit/streamlit_scripts/tree/main/printbak/thumbnails/{row['kopfoto']}?raw=true"
            image_url = f"https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/printbak/thumbnails/{row['kopfoto']}"
            #st.write(image_url)
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
    
