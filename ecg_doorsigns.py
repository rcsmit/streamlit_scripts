import streamlit as st
import fitz  # PyMuPDF
import io
import requests

st.title("📞 Add Phone Number to PDF")

# GitHub URLs
pdf_url = "https://github.com/rcsmit/streamlit_scripts/raw/refs/heads/main/input/templates_ecg_2025a.pdf"
ttf_url = "https://github.com/rcsmit/streamlit_scripts/raw/refs/heads/main/input/Averta-Bold.ttf"
ttf_url_camping_name = "https://github.com/rcsmit/streamlit_scripts/raw/refs/heads/main/input/Averta-Regular.ttf" #not used
 
# Phone number input
col1, col2 = st.columns(2)
with col1:
    phone_number = st.text_input("Phone number", "0039(0)612345678")
    hex_color = st.color_picker("Choose text color", "#2E498E")

with col2:
    camping_name = st.text_input("Camping Name/Code", "IT-123456")
    # Color picker
    hex_color_camping_name = st.color_picker("Choose text color campingname/code", "#CCCCCC")

# Font size
font_size = 40
font_size_camping_name = 10
# Page-specific Y positions
position_dict_str = "{0: 570, 1: 645, 2: 665, 3: 600, 4: 580, 5: 690, 6: 670}"
x_position_camping_name = 30

y_position_camping_name = 810

def hex_to_rgb01(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))

selected_color = hex_to_rgb01(hex_color)

selected_color_camping_name = hex_to_rgb01(hex_color_camping_name)
@st.cache_data
def download_and_cache_font(font_url):
    """Download and cache the TTF font file"""
    try:
        response = requests.get(font_url)
        if response.status_code == 200:
            return response.content
        else:
            st.warning(f"Could not download font (status: {response.status_code}). Using default font.")
            return None
    except Exception as e:
        st.warning(f"Error downloading font: {e}. Using default font.")
        return None

if st.button("Generate PDF"):
    # try:
    if 1==1:
        y_dict = eval(position_dict_str)

        # Fetch PDF
        pdf_response = requests.get(pdf_url)
        if pdf_response.status_code != 200:
            st.error("Could not load PDF from GitHub.")
            st.stop()

        doc = fitz.open(stream=pdf_response.content, filetype="pdf")

        # Download and register custom font
        font_data = download_and_cache_font(ttf_url)
        font = None
        font_name = "helv"  # default fallback
        
        if font_data:
            try:
                # Register the custom font with PyMuPDF
                font = fitz.Font(fontbuffer=font_data)
                font_name = None  # Use the font object instead of font name
                # st.info("✅ Custom font loaded successfully!")
            except Exception as e:
                st.warning(f"Could not load custom font: {e}. Using default font.")
                font = None
                font_name = "helv"
        else:
            st.warning("Using default Helvetica font.")

        # Add phone number to all pages
        for i, page in enumerate(doc):
            y = y_dict.get(i, 700)
            page_width = page.rect.width
            
            # Install custom font on each page if available
            font_name = "helv"  # default
            if font_data:
                try:
                    # Install the font on this page using the correct method
                    font_name = "AvertaBold"  # Standard font reference name
                    page.insert_font(fontname=font_name, fontbuffer=font_data)
                    if i == 0:  # Only show message once
                        # st.info("✅ Custom font installed on pages!")
                        pass
                except Exception as e:
                    if i == 0:  # Only show message once
                        st.warning(f"Error installing custom font: {e}. Using default font.")
                    font_name = "helv"
            
            # Measure text width for centering
            if font and font_name == "F0":
                text_width = font.text_length(phone_number, fontsize=font_size)
            else:
                text_width = fitz.get_text_length(phone_number, fontsize=font_size, fontname="helv")
            
            x = (page_width - text_width) / 2
            
            # Insert text
            page.insert_text(
                fitz.Point(x, y),
                phone_number,
                fontsize=font_size,
                fontname=font_name,
                fill=selected_color,
                fontfile=ttf_url
            )
            
            page.insert_text(
                fitz.Point(x_position_camping_name, y_position_camping_name),
                camping_name,
                fontsize=font_size_camping_name,
                fontname=font_name,
                fill=selected_color_camping_name,
                fontfile=ttf_url
            ) 


            
        # Save to memory

        buffer = io.BytesIO()
        doc.save(buffer)
        doc.close()

        st.success(f"✅ Phone number added successfully | {camping_name} | {phone_number}")
        st.download_button("📥 Download updated PDF", buffer.getvalue(), f"updated_{camping_name}_{phone_number}.pdf", mime="application/pdf")
        st.info("Made by Rene Smit. Contact me [rcx dot smit at gmail dot com] for modification of the  templates. Not officially endorsed by the company.")
    # except Exception as e:
    #     st.error(f"⚠️ Error: {e}")