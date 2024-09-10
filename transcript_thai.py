import pandas as pd
import streamlit as st
# @st.cache(ttl=60 * 60 * 24)
def read():
    #https://docs.google.com/spreadsheets/d/1o_HefbzKnRudVoR64rTZELK_YJ8F2ubDQ3pKjnvWkOQ/edit?usp=sharing

    sheet_id = "1o_HefbzKnRudVoR64rTZELK_YJ8F2ubDQ3pKjnvWkOQ"
    sheet_name = "op_alfabet"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

    #url = "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\in\\schoonmaaktijden.csv",
    df = pd.read_csv(url, delimiter=',')
    #df = df[:-1]  #remove last row which appears to be a Nan
    
    return df

def main():
    df = read()
    df=df.fillna("#")
   
    searchstring = "สวัสดี ฉันชื่อเรเน่"
    st.text_input("Searchstring", searchstring)
    df_output = pd.DataFrame()
    for l in searchstring:
        if l == " ":
            st.write ("SPACE")
        
        else:
            st.write (l)
            df_ = df.loc[df['Letter'] == l]
            if len(df_)>0:
                df_output = pd.concat([df_output, df_], ignore_index=True)
                # st.write(df_)
                # columns = df_.columns
                # xx=""
                # for c in range(len(columns)):
                #     x = (df_.iloc[0,c])
                #     if x != "#":
                #         st.write (f"{x}")
                    
            else:
                st.write ("NOT FOUND")
          
        st.write(df_output)

if __name__ == "__main__":
    main()