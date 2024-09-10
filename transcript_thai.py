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
    for l in searchstring:
        if l == " ":
            print ("SPACE")
        
        else:
            print (l)
            df_ = df.loc[df['Letter'] == l]
            if len(df_)>0:
                
                columns = df_.columns
                xx=""
                for c in range(len(columns)):
                    x = (df_.iloc[0,c])
                    if x != "#":
                        print (f"{x}")
                    
            else:
                print ("NOT FOUND")


if __name__ == "__main__":
    main()