# klein scriptje om een string in hoofdletters om te zetten
import streamlit as st
from random import random

def by_chat_gpt(string, replace_to_numbers, runif):
    s1 = {"e": "3", "i": "!", "o": "0", "a": "4"}

    random_numbers = [random.randint(0, 99) for _ in range(len(string))] if replace_to_numbers else []

    # MORE PYTHONIC
    for i, s in enumerate(string):
        if replace_to_numbers and random_numbers[i] < runif and s in s1:
            new_string += s1[s]
        else:
            new_string += s.upper() if i % 2 == 0 else s


    # even more pythonc
    new_string = "".join(
        s1.get(s, s.upper() if i % 2 == 0 else s)
        if replace_to_numbers and r < runif else s.upper() if i % 2 == 0 else s
        for i, (s, r) in enumerate(zip(string, random_numbers))
    )

    
    return new_string

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
    replace_to_numbers = st.sidebar.selectbox("Replace [e,i,o,a] to numbers", [True, False], index=0)
    if replace_to_numbers:
        runif = st.sidebar.number_input("Percentage om te zetten in nummers (ongeveer)", 0,100,50)
    else:
        runif = None
    new_string = make_breezer_taal(string, replace_to_numbers, runif)
    st.write (new_string)

if __name__ == "__main__":
    main()