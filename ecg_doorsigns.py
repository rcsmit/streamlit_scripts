import streamlit as st
import pymupdf  # pymupdf
import io
import requests

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
        
def generate_pdf(camping_name, phone_number, selected_color,selected_color_camping_name, download_button):

    # GitHub URLs
    pdf_url = "https://github.com/rcsmit/streamlit_scripts/raw/refs/heads/main/input/templates_ecg_2025a.pdf"
    ttf_url = "https://github.com/rcsmit/streamlit_scripts/raw/refs/heads/main/input/Averta-Bold.ttf"
    ttf_url_camping_name = "https://github.com/rcsmit/streamlit_scripts/raw/refs/heads/main/input/Averta-Regular.ttf" #not used
    
    # Font size
    font_size = 40
    font_size_camping_name = 10
    # Page-specific Y positions
    position_dict_str = "{0: 570, 1: 645, 2: 665, 3: 600, 4: 580, 5: 690, 6: 670}"
    x_position_camping_name = 30

    y_position_camping_name = 810
    try:
    #if 1==1:
        y_dict = eval(position_dict_str)

        # Fetch PDF
        pdf_response = requests.get(pdf_url)
        if pdf_response.status_code != 200:
            st.error("Could not load PDF from GitHub.")
            st.stop()

        doc = pymupdf.open(stream=pdf_response.content, filetype="pdf")

        # Download and register custom font
        font_data = download_and_cache_font(ttf_url)
        font = None
        font_name = "helv"  # default fallback
        
        if font_data:
            try:
                # Register the custom font with pymupdf
                font = pymupdf.Font(fontbuffer=font_data)
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
                text_width = pymupdf.get_text_length(phone_number, fontsize=font_size, fontname="helv")
            
            x = (page_width - text_width) / 2
            
            # Insert text
            page.insert_text(
                pymupdf.Point(x, y),
                phone_number,
                fontsize=font_size,
                fontname=font_name,
                fill=selected_color,
                fontfile=ttf_url
            )
            
            page.insert_text(
                pymupdf.Point(x_position_camping_name, y_position_camping_name),
                camping_name,
                fontsize=font_size_camping_name,
                fontname=font_name,
                fill=selected_color_camping_name,
                fontfile=ttf_url
            ) 

        # Save to memory

       
        if download_button:
            buffer = io.BytesIO()
            doc.save(buffer)
            doc.close()
            st.success(f"✅ Phone number added successfully | {camping_name} | {phone_number}")
            st.download_button(f"📥 Download updated PDF | {camping_name} | {phone_number}", buffer.getvalue(), f"doorsigns_{camping_name}_{phone_number}.pdf", mime="application/pdf")
           
        else:
            # Save directly to file system
            output_filename = f"C:\\Users\\rcxsm\\Downloads\\doorsigns_{phone_number.replace('()', '').replace('(', '').replace(')', '')}.pdf"
            
            doc.save(output_filename)
            doc.close()

            st.success(f"✅ Phone number added successfully. PDF saved as: {output_filename}")
            
            # # Optional: Also provide download button
            # with open(output_filename, "rb") as file:
            #     st.download_button("📥 Download updated PDF", file.read(), output_filename, mime="application/pdf")


    except Exception as e:
        st.error(f"⚠️ Error: {e}")


def main():
    st.title("📞 Add Phone Number to PDF")

    mode = "single" #"multiple"
    
   

    def hex_to_rgb01(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))

     
    if mode == "single":
        # Phone number input
        col1, col2 = st.columns(2)
        with col1:
            phone_number = st.text_input("Phone number", "0039(0)612345678")
            hex_color = st.color_picker("Choose text color", "#2E498E")

        with col2:
            camping_name = st.text_input("Camping Name/Code", "IT-123456")
            # Color picker
            hex_color_camping_name = st.color_picker("Choose text color campingname/code. Choose White [#FFFFFF] for invisible", "#CCCCCC")
        selected_color = hex_to_rgb01(hex_color)

        selected_color_camping_name = hex_to_rgb01(hex_color_camping_name)
  
        if st.button("Generate PDF"):
            generate_pdf(camping_name, phone_number, selected_color,selected_color_camping_name, True)
    elif mode =="multiple":
        selected_color = hex_to_rgb01("#2E498E")

        selected_color_camping_name = hex_to_rgb01("#CCCCCC")
  
        campings = [["pra","06123456"],
                    ["pra2","06123457"],
                    ["pra3","06123458"],
                    ["pra4","06123459"],
                    ["pra5","06123460"]]

        for c in campings:
            generate_pdf(c[0], c[1], selected_color,selected_color_camping_name, True)
    else:
        st.error("Please select a valid mode: single or multiple.")
        st.stop()
        
    st.info("Created by Rene Smit.  This tool and its output are not officially endorsed by the company. Use is at your own discretion — I cannot be held responsible for any consequences arising from its use. For template modifications and/or batch use, contact [rcx dot smit at gmail dot com].")
if __name__ == "__main__":
    main()