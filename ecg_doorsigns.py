import streamlit as st
import fitz  # PyMuPDF
import io
import os

st.title("📞 Add Phone Number to All PDF Pages (Centered)")

# Upload PDF
uploaded_file = "https://github.com/rcsmit/streamlit_scripts/raw/refs/heads/main/input/templates_ecg_2025a.pdf"
# st.file_uploader("Upload your PDF", type="pdf")

# Phone number input
phone_number = st.text_input("Phone number", "0612345678")

# Font size
font_size = st.number_input("Font size", value=90)

# Page-specific Y positions
default_dict = "{0: 700, 1: 690}"  # page number: y-position
position_dict_str = st.text_area("Y-positions per page (dict)", value=default_dict)

# Font upload or path
font_file = "https://github.com/rcsmit/streamlit_scripts/raw/refs/heads/main/input/Averta-Bold.ttf" # st.file_uploader("Upload .ttf font (optional)", type=["ttf"])
font_path_input = None # st.text_input("...or use font from repo (e.g. fonts/MyFont.ttf)", value="")
# https://github.com/rcsmit/streamlit_scripts/raw/refs/heads/main/input/templates_ecg_2025a.pdf
# Color picker
hex_color = st.color_picker("Choose text color", "#2E498E")  # default ecg blue

def hex_to_rgb01(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))

selected_color = hex_to_rgb01(hex_color)

# Process PDF
if uploaded_file and st.button("Add Phone Number"):
    try:
        y_dict = eval(position_dict_str)

        # Open PDF
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")

        # Register font
        if font_file:
            font_bytes = font_file.read()
            fontname = doc.insert_font(stream=font_bytes, fontfiletype="truetype")
        elif font_path_input and os.path.exists(font_path_input):
            fontname = doc.insert_font(file=font_path_input, fontfiletype="truetype")
        else:
            fontname = "helv"  # default font

        for i, page in enumerate(doc):
            page_width = page.rect.width
            y = y_dict.get(i, 700)

            text_width = fitz.get_text_length(phone_number, fontname=fontname, fontsize=font_size)
            x_centered = (page_width - text_width) / 2

            page.insert_text(
                fitz.Point(x_centered, y),
                phone_number,
                fontsize=font_size,
                fontname=fontname,
                fill=selected_color
            )

        # Save PDF
        buffer = io.BytesIO()
        doc.save(buffer)
        doc.close()

        st.success("✅ Done! Phone number added.")
        st.download_button("📥 Download updated PDF", buffer.getvalue(), "updated.pdf", mime="application/pdf")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
