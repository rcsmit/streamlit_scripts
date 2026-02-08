import pandas as pd
import streamlit as st

def main():
    language = st.selectbox("Select Language / Selecteer Taal", ["nl","fr"],0)
    url = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\wordlist.txt"
    url = f"https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/wordlist_{language}.txt"

    # Blacklist
    blacklist = ["wie het kleine niet eert is het grote niet weerd"]

    # Lees woorden
    with open(url, 'r', encoding='utf-8') as f:
        words = [line.strip().lower() for line in f if line.strip()]

    # Filter blacklist woorden
    words = [word for word in words if word not in blacklist]

    # Voor elke letter het woord met hoogste count
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    results = []
    output=""
    for letter in alphabet:
        max_word = None
        max_count = 0
        for word in words:
            count = word.count(letter)
            if count > max_count:
                max_count = count
                max_word = word
        results.append({'letter': letter, 'woord': max_word, 'aantal': max_count})
        output += (f"{letter}: {max_word}({max_count})\n")
    # st.write(output)
    # print (output)
    # Maak DataFrame
    df = pd.DataFrame(results)

    # Streamlit weergave
    st.title("Letter Frequentie Analyse")
    st.dataframe(df)

    st.info("Geinspireerd door : https://x.com/aaaronson/status/2019797712179187889 ")
    st.info("Woordenlijst NL: https://github.com/OpenTaal/opentaal-wordlist/blob/master/wordlist.txt")
    st.info("Woordenlijst FR: https://github.com/Taknok/French-Wordlist/blob/master/francais.txt")
if name__ == "__main__":
    main()