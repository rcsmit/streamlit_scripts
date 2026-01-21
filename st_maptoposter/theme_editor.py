import streamlit as st
import json
from pathlib import Path
try:
    from st_create_map_poster import get_available_themes, load_theme
except:
    pass

# Set up themes directory

SCRIPT_DIR = Path(__file__).parent.absolute()
THEMES_DIR = SCRIPT_DIR / "themes"
THEMES_DIR.mkdir(exist_ok=True)

#@st.cache_data
def get_available_themes():
    """Scans the themes directory and returns a list of available theme names."""
    if not THEMES_DIR.exists():
        return []
    
    themes = []
    for file in sorted(THEMES_DIR.glob("*.json")):
        theme_name = file.stem
        themes.append(theme_name)
    
    return themes

#@st.cache_data
def load_theme(theme_name="feature_based"):
    """Load theme from JSON file in themes directory."""
    theme_file = THEMES_DIR / f"{theme_name}.json"
    
    if not theme_file.exists():
        st.warning(f"⚠️ Theme file not found. Using default theme.")
        return {
            "name": "Feature-Based Shading",
            "bg": "#FFFFFF",
            "text": "#000000",
            "gradient_color": "#FFFFFF",
            "water": "#C0C0C0",
            "parks": "#F0F0F0",
            "road_motorway": "#0A0A0A",
            "road_primary": "#1A1A1A",
            "road_secondary": "#2A2A2A",
            "road_tertiary": "#3A3A3A",
            "road_residential": "#4A4A4A",
            "road_default": "#3A3A3A"
        }
    
    try:
        with open(theme_file, 'r') as f:
            theme = json.load(f)
            return theme
    except Exception as e:
        st.error(f"Error loading theme: {e}")
        return None

def save_theme(theme_name, theme_data):
    """Save theme to JSON file."""
    theme_file = THEMES_DIR / f"{theme_name}.json"
    
    try:
        with open(theme_file, 'w') as f:
            json.dump(theme_data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving theme: {e}")
        return False

def theme_editor():
    #st.set_page_config(page_title="Theme Editor", page_icon="🎨", layout="wide")
    
    st.title("🎨 Theme Editor")
    st.markdown("Choose an existing theme to edit or create a new one from scratch.")
    
    # Sidebar for theme selection
    if 1==1:
    #with st.sidebar:
        st.header("Theme Selection")
        
        available_themes = get_available_themes()
        st.write(available_themes)
        # Option to create new or edit existing
        mode = st.radio("Mode", ["Edit Existing Theme", "Create New Theme"])
        
        if mode == "Edit Existing Theme" and available_themes:
            selected_theme = st.selectbox("Select Theme", available_themes)
            current_theme = load_theme(selected_theme)
        else:
            st.info("Creating a new theme...")
            current_theme = {
                "name": "New Theme",
                "description": "A custom theme",
                "bg": "#FFFFFF",
                "text": "#000000",
                "gradient_color": "#FFFFFF",
                "water": "#C0C0C0",
                "parks": "#F0F0F0",
                "road_motorway": "#0A0A0A",
                "road_primary": "#1A1A1A",
                "road_secondary": "#2A2A2A",
                "road_tertiary": "#3A3A3A",
                "road_residential": "#4A4A4A",
                "road_default": "#3A3A3A"
            }
    
    # Main editing area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Theme Settings")
        
        # Basic info
        theme_name = st.text_input("Theme Name", value=current_theme.get("name", ""))
        theme_description = st.text_area("Description", value=current_theme.get("description", ""))
        
        st.subheader("Colors")
        
        # Background and text colors
        col_a, col_b = st.columns(2)
        with col_a:
            bg_color = st.color_picker("Background", value=current_theme.get("bg", "#FFFFFF"))
            text_color = st.color_picker("Text", value=current_theme.get("text", "#000000"))
            gradient_color = st.color_picker("Gradient", value=current_theme.get("gradient_color", "#FFFFFF"))
        
        with col_b:
            water_color = st.color_picker("Water", value=current_theme.get("water", "#C0C0C0"))
            parks_color = st.color_picker("Parks", value=current_theme.get("parks", "#F0F0F0"))
        
        st.subheader("Road Colors")
        
        col_c, col_d = st.columns(2)
        with col_c:
            motorway_color = st.color_picker("Motorway", value=current_theme.get("road_motorway", "#0A0A0A"))
            primary_color = st.color_picker("Primary Road", value=current_theme.get("road_primary", "#1A1A1A"))
            secondary_color = st.color_picker("Secondary Road", value=current_theme.get("road_secondary", "#2A2A2A"))
        
        with col_d:
            tertiary_color = st.color_picker("Tertiary Road", value=current_theme.get("road_tertiary", "#3A3A3A"))
            residential_color = st.color_picker("Residential Road", value=current_theme.get("road_residential", "#4A4A4A"))
            default_color = st.color_picker("Default Road", value=current_theme.get("road_default", "#3A3A3A"))
    
    with col2:
        st.header("Preview")
        
        # Display color swatches
        preview_html = f"""
        <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; border: 1px solid #ccc;">
            <p style="color: {text_color}; font-size: 18px; font-weight: bold;">{theme_name}</p>
            <p style="color: {text_color}; font-size: 14px;">{theme_description}</p>
            
            <div style="margin-top: 20px;">
                <div style="display: flex; gap: 10px; margin-bottom: 10px;">
                    <div style="width: 50px; height: 50px; background-color: {water_color}; border-radius: 5px;" title="Water"></div>
                    <div style="width: 50px; height: 50px; background-color: {parks_color}; border-radius: 5px;" title="Parks"></div>
                    <div style="width: 50px; height: 50px; background-color: {gradient_color}; border-radius: 5px;" title="Gradient"></div>
                </div>
                
                <div style="display: flex; gap: 10px;">
                    <div style="width: 30px; height: 30px; background-color: {motorway_color}; border-radius: 3px;" title="Motorway"></div>
                    <div style="width: 30px; height: 30px; background-color: {primary_color}; border-radius: 3px;" title="Primary"></div>
                    <div style="width: 30px; height: 30px; background-color: {secondary_color}; border-radius: 3px;" title="Secondary"></div>
                    <div style="width: 30px; height: 30px; background-color: {tertiary_color}; border-radius: 3px;" title="Tertiary"></div>
                    <div style="width: 30px; height: 30px; background-color: {residential_color}; border-radius: 3px;" title="Residential"></div>
                </div>
            </div>
        </div>
        """
        #st.markdown(preview_html, unsafe_allow_html=True)
        
        # Show JSON preview
        st.subheader("JSON Preview")
        theme_data = {
            "name": theme_name,
            "description": theme_description,
            "bg": bg_color,
            "text": text_color,
            "gradient_color": gradient_color,
            "water": water_color,
            "parks": parks_color,
            "road_motorway": motorway_color,
            "road_primary": primary_color,
            "road_secondary": secondary_color,
            "road_tertiary": tertiary_color,
            "road_residential": residential_color,
            "road_default": default_color
        }
        st.json(theme_data)
    
    # Save button
    st.divider()
    col_save1, col_save2, col_save3 = st.columns([2, 1, 1])
    
    with col_save1:
        save_name = st.text_input("Save as (filename)", value=theme_name.lower().replace(" ", "_")+"_")
    
    with col_save2:
        if st.button("💾 Save Theme", type="primary", use_container_width=True):
            if save_name:
                if save_theme(save_name, theme_data):
                    st.success(f"✅ Theme saved as '{save_name}.json'")
                    st.balloons()
            else:
                st.error("Please provide a filename")
    with col_save3:
        # Download button
        json_str = json.dumps(theme_data, indent=2)
        st.download_button(
            label="⬇️ Download",
            data=json_str,
            file_name=f"{save_name}.json",
            mime="application/json",
            use_container_width=True
        )
    # with col_save3:
    #     if st.button("🗑️ Delete Theme", use_container_width=True):
    #         if mode == "Edit Existing Theme" and available_themes:
    #             theme_file = THEMES_DIR / f"{selected_theme}.json"
    #             if theme_file.exists():
    #                 theme_file.unlink()
    #                 st.success(f"Deleted theme: {selected_theme}")
    #                 st.rerun()
def main():
    theme_editor()

if __name__ == "__main__":
    main()