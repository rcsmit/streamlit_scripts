import streamlit as st
import pymupdf  # pymupdf
import io
import requests
import pandas as pd


pdf_parking_url_1 = "https://github.com/rcsmit/streamlit_scripts/raw/refs/heads/main/input/template_parking_1.pdf"
pdf_parking_url_2 = "https://github.com/rcsmit/streamlit_scripts/raw/refs/heads/main/input/template_parking_2.pdf"
font_url = "https://github.com/rcsmit/streamlit_scripts/raw/refs/heads/main/input/Averta-Bold.ttf"
pdf_url = "https://github.com/rcsmit/streamlit_scripts/raw/refs/heads/main/input/templates_ecg_2025a.pdf"
# font_url = "https://github.com/rcsmit/streamlit_scripts/raw/refs/heads/main/input/Averta-Regular.ttf"  # not used

# public
# pdf_url = "https://github.com/rcsmit/streamlit_scripts/raw/refs/heads/main/input/template_public.pdf"
# pdf_parking_url_1 = "https://github.com/rcsmit/streamlit_scripts/raw/refs/heads/main/input/template_parking_public_1.pdf"
# pdf_parking_url_2 = "https://github.com/rcsmit/streamlit_scripts/raw/refs/heads/main/input/template_parking_public_2.pdf"
# font_url = "https://github.com/rcsmit/streamlit_scripts/raw/refs/heads/main/input/JupiteroidBold.ttf"


@st.cache_data
def download_and_cache_font(font_url):
    """Download and cache the TTF font file"""
    try:
        response = requests.get(font_url)
        if response.status_code == 200:
            return response.content
        else:
            st.warning(
                f"Could not download font (status: {response.status_code}). Using default font."
            )
            return None
    except Exception as e:
        st.warning(f"Error downloading font: {e}. Using default font.")
        return None

def generate_house_numbers(pdf_file,
    numbers,
    font_size,
    font_color_rgb01,
    show_download_button=True,
):
    # version=1 # logo on left, number right

    version=2 # logo on top, number below

    if version ==1:
        pdf_parking_url = pdf_parking_url_1 
    elif version ==2:
        pdf_parking_url = pdf_parking_url_2 
    
    # Load base template
    if pdf_file is None:
        pdf_response = requests.get(pdf_parking_url)
        if pdf_response.status_code != 200:
            st.error(f"Failed to fetch PDF template. [status {pdf_response.status_code} | {pdf_parking_url}]")
            return
    else:
        pdf_response = pdf_file

    font_data = download_and_cache_font(font_url)

    # Start a new empty PDF
    output_doc = pymupdf.open()
    st.write(pdf_file)
    for number in numbers:
        number_str = str(number)
        if pdf_file is None:
            base_doc = pymupdf.open(stream=pdf_response.content, filetype="pdf")
        else:
            # base_doc = pymupdf.open(stream=pdf_file.read(), filetype="pdf")
            pdf_file.seek(0)  # Important: reset file pointer
            base_doc = pymupdf.open(stream=pdf_file.read(), filetype="pdf")
        page = base_doc.load_page(0)  # only one page in template

        # page_width = page.rect.width
        # page_height = page.rect.height
        # st.write(f"Page size: {page_width} x {page_height}")
        # # Page size: 841.8897705078125 x 595.2755737304688
        if version ==1:
             # version=1 # logo on left, number right

    
            x_target = 627-40-10
        
            y_target =  ((595.2 - (font_size*0.688))/2)  + font_size*0.688



        # c:\Users\rcxsm\Downloads\template_parking_2.pdf
        elif version ==2:
            # version=2 # logo on top, number below
            x_target = 29.7*28.35/2
            # y_target = font_size +20 #-(font_size/20)
            #y_target =  ((385.5 - (font_size*0.688))/2)  + font_size*0.688
            y_target =  385.5 + ((font_size*0.688)/2) #  + font_size*0.688


        font_name = "helv"  # default
        if font_data:
            try:
                font_name = "AvertaBold"
                page.insert_font(fontname=font_name, fontbuffer=font_data)
                font = pymupdf.Font(fontbuffer=font_data)
                text_width = font.text_length(number_str, fontsize=font_size)
            except:
                font = None
                text_width = pymupdf.get_text_length(
                    number_str, fontsize=font_size, fontname="helv"
                )
        else:
            text_width = pymupdf.get_text_length(
                number_str, fontsize=font_size, fontname="helv"
            )

        x = x_target - text_width / 2
        y = y_target

        page.insert_text(
            pymupdf.Point(x, y),
            number_str,
            fontsize=font_size,
            fontname=font_name,
            fill=font_color_rgb01,
            fontfile=font_url,
        )

        output_doc.insert_pdf(base_doc, from_page=0, to_page=0)

    buffer = io.BytesIO()
    output_doc.save(buffer)
    output_doc.close()

    if show_download_button:
        st.success("✅ House numbers added successfully!")
        st.download_button(
            "📥 Download combined PDF with house numbers",
            buffer.getvalue(),
            "house_numbers.pdf",
            mime="application/pdf",
        )


def generate_pdf(pdf_file,
    camping_name,
    phone_number,
    selected_color,
    selected_color_camping_name,
    show_download_button,
):

    # GitHub URLs
  
    # Font size
    font_size = 40
    font_size_camping_name = 10


    # 1 cm = 28.35 points. Left top = [0,0]
    # Page-specific Y positions
    y_position_dict_str = "{0: 570, 1: 645, 2: 665, 3: 600, 4: 480, 5: 690, 6: 670}"
    x_position_camping_name = 30

    y_position_camping_name = 810
    try:
        # if 1==1:
        y_dict = eval(y_position_dict_str)

        if pdf_file is None:
            pdf_response = requests.get(pdf_url)
            if pdf_response.status_code != 200:
                st.error(f"Failed to fetch PDF template. [status {pdf_response.status_code} | {pdf_parking_url}]")
                return
        else:
            pdf_response = pdf_file

        

        if pdf_file is None:
            doc = pymupdf.open(stream=pdf_response.content, filetype="pdf")
        else:
            # base_doc = pymupdf.open(stream=pdf_file.read(), filetype="pdf")
            pdf_file.seek(0)  # Important: reset file pointer
            doc = pymupdf.open(stream=pdf_file.read(), filetype="pdf")
        # Download and register custom font
        font_data = download_and_cache_font(font_url)
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
                        st.warning(
                            f"Error installing custom font: {e}. Using default font."
                        )
                    font_name = "helv"

            # Measure text width for centering
            if font and font_name == "AvertaBold":
                text_width = font.text_length(phone_number, fontsize=font_size)
            else:
                text_width = pymupdf.get_text_length(
                    phone_number, fontsize=font_size, fontname="helv"
                )

            x = (page_width - text_width) / 2

            # Insert text
            page.insert_text(
                pymupdf.Point(x, y),
                phone_number,
                fontsize=font_size,
                fontname=font_name,
                fill=selected_color,
                fontfile=font_url,
            )

            page.insert_text(
                pymupdf.Point(x_position_camping_name, y_position_camping_name),
                camping_name,
                fontsize=font_size_camping_name,
                fontname=font_name,
                fill=selected_color_camping_name,
                fontfile=font_url,
            )

        # Save to memory
        if show_download_button:
            buffer = io.BytesIO()
            doc.save(buffer)
            doc.close()
            st.success(
                f"✅ Phone number added successfully | {camping_name} | {phone_number}"
            )
            st.download_button(
                f"📥 Download updated PDF | {camping_name} | {phone_number}",
                buffer.getvalue(),
                f"doorsigns_{camping_name}_{phone_number}.pdf",
                mime="application/pdf",
            )

        else:
            # Save directly to file system
            output_filename = f"C:\\Users\\rcxsm\\Downloads\\doorsigns_{phone_number.replace('()', '').replace('(', '').replace(')', '')}.pdf"
            doc.save(output_filename)
            doc.close()

            st.success(
                f"✅ Phone number added successfully. PDF saved as: {output_filename}"
            )

            # # Optional: Also provide download button
            # with open(output_filename, "rb") as file:
            #     st.download_button("📥 Download updated PDF", file.read(), output_filename, mime="application/pdf")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

def main():
    st.title("Add Phone/Acco Number to PDF")
    mode = st.selectbox("Choose mode [single |multiple | multiple_csv |house_numbers]", ["single", "multiple","multiple_csv", "house_numbers"])
    # mode = "single"  # "multiple"

    def hex_to_rgb01(hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))

    if mode == "single":
        # Phone number input
        col1, col2 = st.columns(2)
        with col1:
            phone_number = st.text_input("Phone number", "0039(0)612345678")
            hex_color = st.color_picker("Choose text color", "#2E498E")

        with col2:
            camping_name = st.text_input("Camping Name/Code", "IT-123456")
            # Color picker
            hex_color_camping_name = st.color_picker(
                "Choose text color campingname/code. Choose White [#FFFFFF] for invisible",
                "#CCCCCC",
            )
        selected_color = hex_to_rgb01(hex_color)

        selected_color_camping_name = hex_to_rgb01(hex_color_camping_name)
        pdf_file = st.file_uploader("Choose a file. (Leave empty for default template)", type="pdf")
       
        if st.button("Generate PDF"):
            generate_pdf(
                pdf_file,
                camping_name,
                phone_number,
                selected_color,
                selected_color_camping_name,
                True,
            )
    elif mode == "multiple":

        # batch use of the script. Now it is in a list, but it could be
        # read from a file or google sheet in the future.

        selected_color = hex_to_rgb01("#2E498E")
        selected_color_camping_name = hex_to_rgb01("#CCCCCC")
        pdf_file = st.file_uploader("Choose a file. (Leave empty for default template)", type="pdf")
       
        campings = [
            ["pra", "06123456"],
            ["pra2", "06123457"],
            ["pra3", "06123458"],
            ["pra4", "06123459"],
            ["pra5", "06123460"],
        ]

        for c in campings:
            generate_pdf(pdf_file,c[0], c[1], selected_color, selected_color_camping_name, True)
    elif mode =="multiple_csv":

        selected_color = hex_to_rgb01("#2E498E")
        selected_color_camping_name = hex_to_rgb01("#CCCCCC")

        #not used yet, for later reference
        st.markdown("CSV must contain columns (and headers): `camping_name`, `phone_number`")
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        
        if uploaded_file:
            try:
            # if 1==1:
                df = pd.read_csv(uploaded_file)
                for _, row in df.iterrows():
                    camping = str(row["camping_name"])
                    phone = str(row["phone_number"])
                    generate_pdf(camping,phone, selected_color, selected_color_camping_name, True)
    
            except:
                st.error("⚠️ Error reading CSV file. Please ensure it uses commas as seperator, has camping_name and phone_number as heading in the 1st line and is formatted correctly.")
                st.stop()
               
    elif mode == "house_numbers":

        # numbers_str = st.text_input("Enter house numbers (comma-separated)", "1,2,3,4")
        # numbers = [int(n.strip()) for n in numbers_str.split(",") if n.strip().isdigit()]
        numbers  = [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 659, 660, 661, 662, 663, 664, 665, 666, 667]
        font_size = st.slider("Font size", 10, 2000, 300)
        # hex_color = st.color_picker("Font color", "#2E498E")
        hex_color = st.color_picker("Font color", "#000000")
        selected_color = hex_to_rgb01(hex_color)
        
        pdf_file = st.file_uploader("Choose a file. (Leave empty for default template)", type="pdf")
        if st.button("Generate House Number Signs"):
            generate_house_numbers(pdf_file,numbers, font_size, selected_color, True)
    else:
        st.error("Please select a valid mode: single or multiple.")
        st.stop()

    st.info(
        "Created by Rene Smit.  This tool and its output are not officially endorsed by the company. Use is at your own discretion — I cannot be held responsible for any consequences arising from its use. For template modifications and/or batch use, contact [rcx dot smit at gmail dot com]."
    )

    st.info("How to: https://rene-smit.com/no-more-handwritten-signs-a-streamlit-tool-for-instant-pdf-door-signs/")
if __name__ == "__main__":
    main()
