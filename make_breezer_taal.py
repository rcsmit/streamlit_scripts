# klein scriptje om een string in hoofdletters om te zetten
import streamlit as st
from random import random

def make_breezer_taal(string, replace_to_numbers, runif):
    """Make breezerlanguage. 

    Args:
        string (str): _string to convert, in lowercase

    Returns:
        str : converted string
    """        
    new_string = ""
    teller = 0
    s1= ["e", "i", "o","a"]
    s2= ["3", "!", "0", "4"]

    for s in string:   
        if replace_to_numbers:

            r = int(random()*100)
            if r<runif: 
                if s in s1:
                    new_string += s2[s1.index(s)]
                    continue
        
        if teller %2 ==0:
            new_string += s.upper()
        else:
            new_string +=s
        teller +=1
    return new_string

def main():
    st.header("Breezertaal converter")
    standard_string = "De meeste dromen zijn bedrog, maar als ik wakker word naast jou dan droom ik nog"
    string = st.sidebar.text_input("String to convert (in lowercase)" , standard_string)
    replace_to_numbers = st.sidebar.selectbox(
        "Replace [e,i,o,a] to numbers", [True, False], index=0
    runif = st.sidebar.number_input("Percentage om te zetten in nummers", 0,100,50)
    new_string = make_breezer_taal(string, replace_to_numbers, runif)
    st.write (new_string)

if __name__ == "__main__":
    main()