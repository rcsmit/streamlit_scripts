# klein scriptje om een string in hoofdletters om te zetten
import streamlit as st

def make_breezer_taal(string, replace_to_numbers):
    """Make breezerlanguage. 

    Args:
        string (str): _string to convert, in lowercase

    Returns:
        str : converted string
    """        
    new_string = ""
    teller = 0
    for s in string:   
        if replace_to_numbers:
            if s == "e":
                new_string += "3"
                continue 
            elif s == "i":
                new_string += "!"
                continue 
            elif s == "o":
                new_string += "0"
                continue 
            elif s == "a":
                new_string +="4"
                continue 
        
        if teller %2 ==0:
            new_string += s.upper()
        else:
            new_string +=s
        teller +=1
    return new_string

def main():
    st.header("Breezertaal converter")
    standard_string = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed nunc orci, euismod nec venenatis sit amet, pulvinar sed nibh. Donec condimentum id nunc ultrices maximus."
    string = st.sidebar.text_input("String to convert (in lowercase)" , standard_string)
    replace_to_numbers = st.sidebar.selectbox(
        "Replace [e,i,o,a] to numbers", [True, False], index=0
        )
    new_string = make_breezer_taal(string, replace_to_numbers)
    st.write (new_string)

if __name__ == "__main__":
    main()