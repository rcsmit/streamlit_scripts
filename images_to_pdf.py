import os
import time
import streamlit as st
from PIL import Image, ExifTags
from io import BytesIO
import pytesseract
import re
import os
import numpy as np

import time
def clean_filename(filename, replace_with='_'):

    """Replace forbidden characters

    https://chatgpt.com/c/6706ff08-4a8c-8004-9d62-c1f28cdd7de1

    Args:
        filename (str): filename
        replace_with (str) : character to use 

    Returns:
        str: corrected string
    """    
    # Define forbidden characters based on the operating system
    forbidden_chars = r'[<>:"/\\|?*\&\.\,\n\%\^!@#$(){}\[\]\'\"]'  # Windows forbidden characters
    if os.name != 'nt':  # If not Windows (Linux/Mac)
        forbidden_chars = r'[/?<>\\:*|"]'  # Unix-based forbidden characters
    
    # Remove forbidden characters using regex, replacing them with '_'
    cleaned_filename = re.sub(forbidden_chars, replace_with, filename)
    
    # Return the cleaned filename
    return cleaned_filename

def correct_image_rotation(image):
    """Many modern cameras and smartphones store orientation information in the image file 
    as EXIF metadata rather than physically rotating the image. When viewing these images in 
    some applications (e.g., image viewers or some PDF readers), they may appear correctly 
    rotated because these applications honor the EXIF orientation metadata. 
    However, PIL.Image.open() doesn't always apply this by default, 
    so we need to manually correct it.

    https://chatgpt.com/c/6706ff08-4a8c-8004-9d62-c1f28cdd7de1
    
    Args:
        image (_type_): _description_

    Returns:
        _type_: _description_
    """    
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        
        if exif is not None:
            orientation = exif.get(orientation)

            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # If there's no EXIF data or no orientation tag, we skip the rotation.
        pass

    return image


def main_streamlit_upload_files():
    """ Convert one or multiple files into a pdf
        Uses streamlit, Adviced is to use a split screen to easy drag and drop.

        
        based on https://www.youtube.com/watch?v=RPN-HxvAQnQ


    """     
    # Title of the app
    st.title("JPEG/JPG to PDF Converter")
    
             
    # Image file upload
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        uploaded_images_1 = st.file_uploader("Upload JPEG or JPG Image 1", type=["jpeg", "jpg"], accept_multiple_files=False)
    with col2:
        uploaded_images_2 = st.file_uploader("Upload JPEG or JPG Image 2", type=["jpeg", "jpg"], accept_multiple_files=False)
    with col3:
        uploaded_images_3 = st.file_uploader("Upload JPEG or JPG Image 3", type=["jpeg", "jpg"], accept_multiple_files=False)
    with col4:
        uploaded_images_4 = st.file_uploader("Upload JPEG or JPG Image 4", type=["jpeg", "jpg"], accept_multiple_files=False)
    with col5:
        uploaded_images_5 = st.file_uploader("Upload JPEG or JPG Image 5", type=["jpeg", "jpg"], accept_multiple_files=False)
    
   
    uploaded_images = [uploaded_images_1,  uploaded_images_2, uploaded_images_3, uploaded_images_4, uploaded_images_5]
    # Convert images to PDF
    if st.button("Convert to PDF"):
        s1 = int(time.time())
        placeholder = st.empty()
        placeholder.info("Converting")
        # Create an empty list to hold the images
        image_list = []

        # Loop through the uploaded images
        for i,uploaded_image in enumerate(uploaded_images):
            if uploaded_image is not None:
                # Open the image using PIL
                image = Image.open(uploaded_image)
                image = correct_image_rotation(image)
                if i==0:
                    # if you want to use the filename of the first image
                    filename_proposed =uploaded_image.name[:-6]

                    # if you want to read the text in the first image.

                    #filename_proposed = str(((pytesseract.image_to_string(image))))
                # Convert image to RGB if it's not in RGB format (for PDF compatibility)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image_list.append(image)
        
        if image_list:
            # Create a BytesIO buffer to save the PDF
            pdf_buffer = BytesIO()
            # Save the images as PDF
            image_list[0].save(pdf_buffer, format="PDF", save_all=True, append_images=image_list[1:])
            pdf_buffer.seek(0)

            #doesnt work
            #filename_given = st.text_input("Filename", filename_proposed[:50])
            
            # use the first 50 characters. Clean the filename
            filename_given = clean_filename(filename_proposed[:50], replace_with='_')
            s2 = int(time.time())
            s2x = s2 - s1
            placeholder.info(f"{filename_given} - {str(s2x)} seconds)")
            # Provide the download button for the PDF
            st.download_button(
                "Download PDF",
                data=pdf_buffer,
                file_name=f"{filename_given}.pdf",
                mime="application/pdf"
            )
              

        else:
            st.error("No valid images found.")
    else:
        st.info(".")


def split():

    """ Split a landscape image into two equal portrait images. 
        Uses streamlit, Adviced is to use a split screen to easy drag and drop.
        The filename of the first file (minus the last 4 characters eg. .jpg) is used.

    """
    
    def crop(im, height, width):
      
        imgwidth, imgheight = im.size
        rows = int(imgheight/height)
        cols = int(imgwidth/width)
        for i in range(rows):
            for j in range(cols):
                box = (j*width, i*height, (j+1)*width, (i+1)*height)
                yield im.crop(box)
   
    uploaded_image = st.file_uploader("Upload JPEG or JPG Image 1", type=["jpeg", "jpg"], accept_multiple_files=False)
    
    if st.button("Split image"):

        image = Image.open(uploaded_image)
        image = correct_image_rotation(image)
        filename =uploaded_image.name[:-4]
        # https://gist.github.com/alexlib/ef7df7bfdb3dba1698f4
        imgwidth, imgheight = image.size
        print(('Image size is: %d x %d ' % (imgwidth, imgheight)))
        height = imgheight # np.int(imgheight/2)
        width = int(imgwidth/2)
        start_num = 0
        for k, piece in enumerate(crop(image, height, width), start_num):
            img = Image.new('RGB', (width, height), 255)
            
            img.paste(piece)
            filename_tosave = f"{filename}-{k+1}.jpg"

            img.save(filename_tosave)
            st.write(f"{filename_tosave} saved")
        
def main_():
    """ Making ONE pdf from a list of images
        based on https://stackoverflow.com/a/47283224/4173718 CC BY 4.0
        install by > python3 -m pip install --upgrade Pillow  # ref. https://pillow.readthedocs.io/en/latest/installation.html#basic-installation
        To use if you want
        - to convert a list of files into one pdf
        - to convert a complete directory into one pdf
        - to converrt files with the same name (xxxx01.pdf .. xxxx04.pdf), into one pdf
    """        
   
    name_of_files = "dontknow why"

    extensie = "jpg"
    number_of_files = 1
    make_files = True
    mode = "list_of_files"
    # mode ="easysheetmusic" # https://easysheetmusic.altervista.org/
    # mode="directory"
    if mode == "list_of_files":

        #         #   r"C:\Users\rcxsm\Downloads\Sheetmusic\gefotografeerd\IMG_4242.JPG"]
        files = [
            r"C:\Users\rcxsm\Downloads\Sheetmusic\gefotografeerd\2024_10_09\20241009_131235.jpg",
            r"C:\Users\rcxsm\Downloads\Sheetmusic\gefotografeerd\2024_10_09\20241009_131241.jpg",
        ]
    elif mode =="directory":
        dir_name = r"C:\Users\rcxsm\Downloads\Sheetmusic\personal fakebook"
        os.chdir(dir_name)  # change directory from working dir to dir with files
        files = []
        for file in os.listdir(dir_name):  # loop through items in dir
            print (file)
            files.append(file)
        name_of_files = "personal_fakebook"


    elif mode == "easysheetmusic":
        
        files = []
        for n in range(number_of_files):
            files.append(r"C:\Users\rcxsm\Downloads" + "\\"+ name_of_files + "-0" + str(n + 1) + "." + extensie)
    else:
        print("Error in Mode {mode}")
    print(files)

 
    #images = [Image.open(f) for f in files]

    pdf_path = r"C:\\Users\\rcxsm\\Downloads\\" + name_of_files + ".pdf"

    Image.open(files[0]).save(
        pdf_path, "PDF", resolution=100.0, save_all=True, 
        append_images=(Image.open(f) for f in files[1:])) 

    
    print(f"{pdf_path} saved")
    time.sleep(1)
    delete=False
    if delete:
        try:
            for f in files:
                os.remove(dir_name + f)
                print(f"{dir_name+f} deleted")
        except:
            print("Can't delete file")

def main():
    choice = st.selectbox("What do do",["convert", "split"])
    if choice =="convert":
        main_streamlit_upload_files()
    else:
        split()

if __name__ == "__main__":
    