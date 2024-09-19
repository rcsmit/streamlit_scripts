import pandas as pd
import streamlit as st
import numpy as np
from deep_translator import GoogleTranslator


@st.cache_data(ttl=60 * 60 * 24)
def read():
    # https://docs.google.com/spreadsheets/d/1o_HefbzKnRudVoR64rTZELK_YJ8F2ubDQ3pKjnvWkOQ/edit?usp=sharing

    sheet_id = "1o_HefbzKnRudVoR64rTZELK_YJ8F2ubDQ3pKjnvWkOQ"
    sheet_name = "op_alfabet"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    df = pd.read_csv(url, delimiter=",")
    return df


def reorder_columns(df_output):
    """Puts the column 'Sounds like' after the column 'letter'

    Args:
        df_output (df): df

    Returns:
        df: df
    """    
    # List the columns of your dataframe
    cols = df_output.columns.tolist()

    # Find the positions of "Letter" and "Sounds like"
    letter_index = cols.index("Letter")
    sounds_like_index = cols.index("Sounds like")

    # Move the "Sounds like" column to immediately after the "Letter" column
    cols.insert(letter_index + 1, cols.pop(sounds_like_index))

    # Reorder the dataframe with the new column order
    df_output = df_output[cols]
    return df_output


def add_row_unfound(l):
    """Add a row in the source data to use if a character x isn't found and put x in 
       the columns "sound", "Sounds like" and  "Letter"

    Args:
        l (str): the not found character

    Returns:
        df: df with one row 
    """
    # Define the column names and data types
    columns = {
        "id": "int64",
        "Letter": "object",
        "Sounds like": "object",
        "Sample word": "object",
        "RTGS": "object",
        "Romanization": "object",
        "Unnamed: 5": "object",
        "Sample word Romanization": "object",
        "Sample Word Translation": "object",
        "sound": "object",
        "sample phrase in thai": "object",
        "sample word": "object",
        "sample phrase action": "object",
        "sample phrase romanization": "object",
        "sample phrase translation": "object",
        "class": "object",
        "initial": "object",
        "final": "object",
        "FREQ": "object",
        "position vowels": "object",
        "UNICODE": "object",
        "mid/low/heigh": "object",
        "remarks": "object",
    }

    # Create an empty dataframe with specified columns and data types
    df_empty = pd.DataFrame(
        {col: pd.Series(dtype=dtype) for col, dtype in columns.items()}
    )

    for field in ["sound", "Sounds like", "Letter"]:
        df_empty.loc[0, field] = l
    return df_empty


def main():
    df = read()
    df = df.fillna(".")

    searchstring_ = "สวัสดี ฉันชื่อเรเน่"
    searchstring = st.text_input("Searchstring", searchstring_)

    translated = GoogleTranslator(source="auto", target="en").translate(searchstring)
    st.success(translated)

    if searchstring == "":
        st.error("Enter a search string")
        st.stop()
    df_output = pd.DataFrame()
    for l in searchstring:
        if l == " ":
            # st.write ("SPACE")
            pass
        else:
            df_ = df.loc[df["Letter"] == l]
            if len(df_) > 0:
                df_output = pd.concat([df_output, df_], ignore_index=True)
            else:
                df_ = add_row_unfound(l)
                df_output = pd.concat([df_output, df_], ignore_index=True)

    df_output = reorder_columns(df_output)
    # Combine all values from the 'sound' column into one string
    combined_sound_string = " ".join(df_output["sound"].astype(str))

    # Display the result
    st.info(combined_sound_string)
    st.write(df_output)

if __name__ == "__main__":
    main()
