import streamlit as st
from xml.dom import minidom

from urllib.request import urlopen


def get_song_list(country_chosen_long, country_chosen_abbr):
    xml1 = f"http://itunes.apple.com/{country_chosen_abbr}/rss/topsongs/limit=50/explicit=true/xml"
    # parse an xml file by name
    mydoc = minidom.parse(urlopen(xml1))
    items = mydoc.getElementsByTagName('title')

    st.write(f'Hitlist for  {country_chosen_long}')
    for i, elem in enumerate(items):
        # print(elem.attributes['id'].value)
        st.write (f"{i}. {elem.firstChild.data}")
    # # one specific item's data
    # print('\nItem #2 data:')
    # print(items[1].firstChild.data)
    # print(items[1].childNodes[0].data)

    # # all items data
    # print('\nAll item data:')
    # for elem in items:
    #     print(elem.firstChild.data)

def interface():
    countries_abbr = ["nl","gb","fr","de ","it","es","pt","us","be","se ","au","at","ca ","dk","fi","gr","ie","il","jp","lu","mx","nz","no","ch"]
    countries_long = ["Netherlands","Great Britain","France","Germany","Italy","Spain","Portugal","United States of America","Belgium","Sweden",
                    "Australia","Austria","Canada","Danmark","Finland","Greece","Ireland","Isarel","Japan","Luxembourg","Mexico,New Zealand","Norway","Switserland"]
    country_chosen_long = st.sidebar.selectbox("Country", countries_long, index=0)
    country_chosen_abbr =  countries_abbr[countries_long.index(country_chosen_long)]
    return country_chosen_long, country_chosen_abbr

def main():
    country_chosen_long, country_chosen_abbr = interface()
    get_song_list(country_chosen_long, country_chosen_abbr)

main()