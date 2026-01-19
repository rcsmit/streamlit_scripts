import streamlit as st
from pathlib import Path
from PIL import Image
import os

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
POSTERS_DIR = SCRIPT_DIR / "posters"

def get_poster_files():
    """
    Get all image files from the posters directory.
    Returns a list of tuples: (filename, filepath)
    """
    if not POSTERS_DIR.exists():
        return []
    
    # Supported image formats
    image_extensions = {'.png', '.jpg', '.jpeg', '.svg', '.webp', '.gif'}
    
    posters = []
    for file in sorted(POSTERS_DIR.iterdir(), reverse=True):  # Most recent first
        if file.suffix.lower() in image_extensions and file.is_file():
            posters.append((file.name, file))
    
    return posters

def parse_filename(filename):
    """
    Parse filename to extract city, theme, and timestamp.
    Expected format: city_theme_timestamp.extension
    """
    try:
        # Remove extension
        name_without_ext = filename.rsplit('.', 1)[0]
        
        # Try to split into parts
        parts = name_without_ext.split('_')
        
        if len(parts) >= 3:
            # Last two parts are likely timestamp (date_time)
            timestamp = f"{parts[-2]}_{parts[-1]}"
            # Second to last group is theme
            theme = parts[-3] if len(parts) > 3 else "unknown"
            # Everything else is city
            city = '_'.join(parts[:-3]) if len(parts) > 3 else parts[0]
            
            return {
                'city': city.replace('_', ' ').title(),
                'theme': theme.replace('_', ' ').title(),
                'timestamp': timestamp
            }
        else:
            return {
                'city': 'Unknown',
                'theme': 'Unknown',
                'timestamp': ''
            }
    except:
        return {
            'city': 'Unknown',
            'theme': 'Unknown',
            'timestamp': ''
        }

def show_posters():
    st.set_page_config(
        page_title="Map Poster Gallery", 
        page_icon="🖼️", 
        layout="wide"
    )
    
    st.title("🖼️ Map Poster Gallery")
    st.markdown("Browse your generated map posters")
    
    # Get all poster files
    posters = get_poster_files()
    
    if not posters:
        st.warning(f"📁 No posters found in `{POSTERS_DIR}`")
        st.info("Generate some posters first, or add sample images to the 'posters' directory!")
        
        # Show directory info
        with st.expander("ℹ️ Directory Information"):
            st.code(f"""
Posters Directory: {POSTERS_DIR}
Directory Exists: {POSTERS_DIR.exists()}
Supported formats: .png, .jpg, .jpeg, .svg, .webp, .gif
            """)
        return
    
    st.success(f"Found **{len(posters)}** poster(s)")
    
    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Filters")
        
        # Extract unique cities and themes
        all_info = [parse_filename(name) for name, _ in posters]
        cities = sorted(set(info['city'] for info in all_info))
        themes = sorted(set(info['theme'] for info in all_info))
        
        selected_city = st.selectbox(
            "Filter by City",
            options=["All"] + cities,
            index=0
        )
        
        selected_theme = st.selectbox(
            "Filter by Theme",
            options=["All"] + themes,
            index=0
        )
        
        st.markdown("---")
        
        # View options
        st.subheader("⚙️ View Options")
        
        columns = st.slider(
            "Columns per row",
            min_value=1,
            max_value=4,
            value=3,
            help="Number of images per row"
        )
        
        show_info = st.checkbox("Show image details", value=True)
        
        image_height = st.slider(
            "Thumbnail height",
            min_value=200,
            max_value=800,
            value=400,
            step=50,
            help="Height of thumbnail images in pixels"
        )
    
    # Filter posters
    filtered_posters = []
    for filename, filepath in posters:
        info = parse_filename(filename)
        
        # Apply filters
        if selected_city != "All" and info['city'] != selected_city:
            continue
        if selected_theme != "All" and info['theme'] != selected_theme:
            continue
        
        filtered_posters.append((filename, filepath, info))
    
    if not filtered_posters:
        st.warning("No posters match the selected filters")
        return
    
    st.markdown(f"Showing **{len(filtered_posters)}** poster(s)")
    st.markdown("---")
    
    # Display gallery
    for i in range(0, len(filtered_posters), columns):
        cols = st.columns(columns)
        
        for idx, col in enumerate(cols):
            if i + idx < len(filtered_posters):
                filename, filepath, info = filtered_posters[i + idx]
                
                with col:
                    # Display image
                    if filepath.suffix.lower() == '.svg':
                        # SVG files can't be displayed with st.image, show a placeholder
                        st.info("🎨 SVG File")
                        st.markdown(f"**{filename}**")
                    else:
                        try:
                            # Load and display image
                            img = Image.open(filepath)
                            st.image(img, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error loading image: {e}")
                    
                    # Show details
                    if show_info:
                        st.markdown(f"**{info['city']}**")
                        st.caption(f"Theme: {info['theme']}")
                        if info['timestamp']:
                            st.caption(f"Created: {info['timestamp']}")
                    
                    # File info and actions
                    file_size = filepath.stat().st_size / 1024  # KB
                    st.caption(f"📁 {filename} ({file_size:.1f} KB)")
                    
                    # Download button
                    with open(filepath, "rb") as file:
                        mime_type = "image/svg+xml" if filepath.suffix.lower() == '.svg' else f"image/{filepath.suffix[1:]}"
                        st.download_button(
                            label="⬇️ Download",
                            data=file,
                            file_name=filename,
                            mime=mime_type,
                            use_container_width=True,
                            key=f"download_{filename}"
                        )
    
    # Gallery statistics
    st.markdown("---")
    with st.expander("📊 Gallery Statistics"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Posters", len(posters))
        
        with col2:
            total_size = sum(fp.stat().st_size for _, fp in posters) / (1024 * 1024)  # MB
            st.metric("Total Size", f"{total_size:.2f} MB")
        
        with col3:
            formats = {}
            for _, filepath in posters:
                ext = filepath.suffix.lower()
                formats[ext] = formats.get(ext, 0) + 1
            st.metric("Formats", ", ".join(f"{k}({v})" for k, v in formats.items()))
def main():
    show_posters()
    
if __name__ == "__main__":
    main()