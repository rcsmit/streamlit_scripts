# klein scriptje om een string in hoofdletters om te zetten
import streamlit as st

def make_breezer_taal(string):
    """Make breezerlanguage. 

    Args:
        string (str): _string to convert, in lowercase

    Returns:
        str : converted string
    """        
    new_string = ""
    teller = 0
    for s in string:   
        if s == "e":
            new_string += "3"
        elif s == "o":
            new_string += "0"
        elif s == "a":
            new_string +="4"
        else:
            if teller %2 ==0:
                new_string += s.upper()
            else:
                new_string +=s
            teller +=1
    return new_string

def main():
    st.header("Breezertaal converter")
    string = st.sidebar.text_input("String to convert (in lowercase)" , "zie je wel : ze misbruiken onze gegevens !!!")

    new_string = make_breezer_taal(string)
    st.write (new_string)

if __name__ == "__main__":
    main()