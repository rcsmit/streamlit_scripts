# https://www.geeksforgeeks.org/python-reading-contents-of-pdf-using-ocr-optical-character-recognition/


# Import libraries
# from PIL import Image
# import pytesseract
# import sys
# from pdf2image import convert_from_path
# import os
import io

from PyPDF2 import PdfReader
import re
import pandas as pd
import streamlit as st

def read_directly_from_pdf():
    # read a file 
    # inspired by https://x.com/Transparangst/status/1906717209423974689
    
    # Install PyPDF2 if not already installed
    # pip install PyPDF2

    # Path to the PDF file
    
    pdf_path = "C:/Users/rcxsm/Downloads/vac_med_okt_2020.pdf"
    
    # Create a PDF reader object

    pdf_path = st.file_uploader("Choose a file")
    if pdf_path is not None:
        try:
            reader = PdfReader(pdf_path)
            
        except Exception as e:
            st.error(f"Error loading / parsing the PDF file: {str(e)}")
            st.stop()
    else:
        st.warning("You need to upload a pdf file. Files are not stored anywhere after the processing of this script")
        st.stop()


    reader = PdfReader(pdf_path)
    all_text = ""
    # Extract text from each page
    number_of_pages = len(reader.pages)
    placeholder = st.empty()
    for i,page in enumerate(reader.pages):
        
        text = page.extract_text()
        text = re.sub(r'(\n\d{6})', r'\1#', text)
        
        for t in ["Reeds Openbaar", "Deels Openbaar", "Niet Openbaar", "Openbaar"]:
            text = text.replace(t, f'#{t}#')
            text = text.replace('#Deels #Openbaar##','#Deels Openbaar#')
            text = text.replace('#Reeds #Openbaar##','#Reeds Openbaar#')
            text = text.replace('#Niet #Openbaar##','#Niet Openbaar#')
            
        text = text.replace("# ", "#")
        text = text.replace("; 10.","#10.")
        text = text.replace("; 11.","#11.")
        text = text.replace("; buiten verzoek","#buiten verzoek")
        
        progress_txt= (f"Reading page {i+1}/{number_of_pages}")
        placeholder.progress(i/number_of_pages, f"Wait for it...{progress_txt}")
        all_text +="\n"+text
        # test purposes
            
        if i>2:
            break
    # Split text into rows and columns using '#' as a separator
    placeholder.empty()
    rows = [line.split('#') for line in all_text.splitlines()]
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    
    # Iterate through rows and check columns 3 to 8 for "10.2.a"
    for i in ["a","b","c","d","e","f","g"]:
        df[f"101{i}"] = df.iloc[:, 3:9].apply(lambda row: f"10.1.{i}" in row.values, axis=1)    
        df[f"102{i}"] = df.iloc[:, 3:9].apply(lambda row: f"10.2.{i}" in row.values, axis=1)    
    
    df["BuitenVerzoek"] = df.iloc[:, 3:9].apply(lambda row: "buiten verzoek" in row.values, axis=1)
    df["111concept"] = df.iloc[:, 3:9].apply(lambda row: "11.1, concept" in row.values, axis=1)

    # df.to_csv("output.csv", index=False)
    # df.to_excel("output.xlsx", index=False)

    st.download_button("Download CSV", data=df.to_csv(index=False), file_name="output.csv", mime="text/csv")
    # st.download_button("Download Excel", data=df.to_excel(index=False), file_name="output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Create an in-memory buffer
    excel_buffer = io.BytesIO()

    # Write the DataFrame to the buffer as an Excel file
    df.to_excel(excel_buffer, index=False, engine='openpyxl')

    # Reset the buffer's position to the beginning
    excel_buffer.seek(0)

    # Streamlit download button for Excel
    st.download_button(
        label="Download Excel",
        data=excel_buffer,
        file_name="output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    
    st.write(df)

def main():
    st.info("Read PDF files from Dutch governement")
    st.write("This script reads a PDF file and extracts the text from it. It then processes the text to create a DataFrame.")
    st.write("The DataFrame is then saved as a CSV and an Excel file, which can be downloaded.")
    st.write("It is specifically written for a type of document")
    read_directly_from_pdf()

if __name__ == "__main__":
    main()  