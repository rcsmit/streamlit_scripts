from show_posters import show_posters
from theme_editor import theme_editor
# from organize_svg import organize_svg_with_theme
from app import app
import streamlit as st
from pathlib import Path
from generate_examples import generate_examples


SCRIPT_DIR = Path(__file__).parent.absolute()
THEMES_DIR = SCRIPT_DIR / "themes"

# #Replace the get_available_themes() function (lines 129-139)
def get_available_themes_with_subdirs():
    """Scans the themes directory and returns a list of available theme names from folders.
    Doesnt work yet"""
    
    if not THEMES_DIR.exists():
        st.error("THEMES_DIR doesnt exist")
        return []
    
    themes = []
    for file in sorted(THEMES_DIR.iterdir()):
        if file.is_file() and file.suffix.lower() == '.json':
            theme_name = file.stem
            themes.append(f"{theme_name}")
    
    for folder in sorted(THEMES_DIR.iterdir()):
        if folder.is_dir():
            for file in folder.iterdir(): 
                theme_file = folder / f"{file.stem}.json"
                if theme_file.exists():
                    themes.append(f"{folder.name}/{file.stem}")
    return themes


def main():
    tab1,tab2,tab3,tab4=st.tabs(["Start", "Examples","Galery","Theme Editor"])
    
    with tab1:
        app()
   
    with tab2:
        st.header("Examples")
        st.info("The examples are made with a small town in the Netherlands due to efficiency reasons")
    
        if st.button("Show examples"):
            generate_examples()
    with tab3:
        st.header("Galery")
        if st.button("Show Galery"):
            show_posters()
    with tab4:
        theme_editor()

if __name__ == "__main__":
    main()