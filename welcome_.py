import streamlit as st


def main():
    st.header ("Welcome!")
    toelichting = (
        "<p>Here you'll find the scripts I made with Streamlit.</p>"



    )
    tekst = (
        "<style> .infobox {  background-color: lightblue; padding: 5px;}</style>"
        "<hr><div class='infobox'>Made by Rene Smit. (<a href='http://www.twitter.com/rcsmit' target=\"_blank\">@rcsmit</a>) <br>"
        'Sourcecode : <a href="https://github.com/rcsmit/streamlit_scripts/" target="_blank">github.com/rcsmit</a><br>'
        'How-to tutorial : <a href="https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d" target="_blank">rcsmit.medium.com</a><br>'

    )
    st.markdown(toelichting, unsafe_allow_html=True)
    st.markdown(tekst, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
