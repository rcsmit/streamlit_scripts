import pandas as pd
import streamlit as st
import requests

def main():
    #url = f"https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/wordlist.txt"

    # Dictionary met woordenlijsten per taal
    WORDLISTS = {
        "NL": {
            "bron": "https://github.com/OpenTaal/opentaal-wordlist",
            "wordlist": "https://raw.githubusercontent.com/OpenTaal/opentaal-wordlist/master/wordlist.txt"
        },
        "FR": {
            "bron": "https://github.com/Taknok/French-Wordlist",
            "wordlist": "https://raw.githubusercontent.com/Taknok/French-Wordlist/master/francais.txt"
        },
        "DE": {
            "bron": "https://gist.github.com/MarvinJWendt/2f4f4154b8ae218600eb091a5706b5f4",
            "wordlist": "https://gist.githubusercontent.com/MarvinJWendt/2f4f4154b8ae218600eb091a5706b5f4/raw/ce6e5c3249be6e11c81e265f69e0e0c8ef0c91a3/germanWordList.txt"
        }
    }


    # Selecteer taal
    language = st.selectbox("Kies taal:", options=list(WORDLISTS.keys()))
    language_lower = language.lower()
    # Haal juiste URLs op
    bron_url = WORDLISTS[language]["bron"]
    url = f"https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/wordlist_{language_lower}.txt"
    
    # Blacklist
    blacklist = ["wie het kleine niet eert is het grote niet weerd"]

    # Lees woorden voor lokale bestanden
    # with open(url, 'r', encoding='utf-8') as f:
    #     words = [line.strip().lower() for line in f if line.strip()]

    try:
        response = requests.get(url)
        response.raise_for_status()  # Check of request succesvol was
    except requests.exceptions.RequestException as e:
        st.error(f"Fout bij het ophalen van de woordenlijst: {e}")
        st.stop()   
    words = [line.strip().lower() for line in response.text.split('\n') if line.strip()]

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

    st.info(f"Number of words : {len(words)}")
    st.info("Geinspireerd door : https://x.com/aaaronson/status/2019797712179187889 ")
    st.info("Woordenlijst : {bron_url}")
   
if __name__ == "__main__":
    main()