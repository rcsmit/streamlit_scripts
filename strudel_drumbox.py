import streamlit as st
from streamlit.components.v1 import html

def main():
    # Set page layout
    #st.set_page_config(layout="wide")
    st.title("Strudel Beat Generator (16-Step, 11-Instrument Sequencer)")

    st.markdown("""
    ### Drum Abbreviations

    | Drum Name           | Abbreviation |
    |---------------------|--------------|
    | Bass drum / Kick    | `bd`         |
    | Snare drum          | `sd`         |
    | Rimshot             | `rim`        |
    | Clap                | `cp`         |
    | Closed hi-hat       | `hh`         |
    | Open hi-hat         | `oh`         |
    | Crash               | `cr`         |
    | Ride                | `rd`         |
    | High tom            | `ht`         |
    | Medium tom          | `mt`         |
    | Low tom             | `lt`         |
    """)

    # List of drum banks
    banks = [
        "RolandTR909",
        "RolandTR808",
        "AkaiLinn",
        "RhythmAce",
        "RolandTR707",
        "ViscoSpaceDrum",
    ]
    # Instrument list, matching your example
    instruments = ["bd", "sd", "lt","mt", "ht", "hh", "oh", "cp","cr","rd",  "rim"]

    # State holder for each row (instrument)
    steps = {}

    st.markdown("### Step Sequencer")

    # Show header labels 01 to 16
    header_cols = st.columns(17)
    header_cols[0].markdown("**Inst**")
    for i in range(16):
        header_cols[i + 1].markdown(f"**{i+1:02}**")

    # Build checkboxes row by row
    for instrument in instruments:

        cols = st.columns(17)
        steps[instrument] = []
        
        for i in range(17):
            key = f"{instrument}_{i}"
            if i == 0:
                cols[i].write(instrument.upper())
            else:   
                checked = cols[i].checkbox(label="_",label_visibility="hidden", key=key)
                steps[instrument].append(instrument if checked else "- ")


    # Bank selector
    selected_bank = st.selectbox("Choose a drum bank", banks)

    if st.button("Generate Beat"):
    
        # Generate Strudel code
        
        code_lines = []
        for instrument in instruments:
            line = " ".join(steps[instrument])
            
            code_lines.append(line)

        strudel_code = "//     01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16\n"
        strudel_code += "sound(`" + ",\n       ".join(code_lines) + "`)"
        strudel_code += f".bank(\"{selected_bank}\")"
        

    


        # Wrapt the javascript as html code
        my_html = f"""

        <script src="https://unpkg.com/@strudel/embed@latest"></script>
        <strudel-repl>
        <!--
        {strudel_code}
        -->
        </strudel-repl>
        """

        # Execute your app
        
        html(my_html, width=1200,height=900)
        st.markdown("### Generated Strudel Code:")
        st.code(strudel_code, language="javascript")

if __name__ == "__main__":
    main()