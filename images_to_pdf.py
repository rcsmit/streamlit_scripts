import os
import time
import streamlit as st
from PIL import Image, ExifTags, ImageOps

from io import BytesIO
import pytesseract
import re
import os
import numpy as np
from PyPDF2 import PdfReader, PdfWriter
import time
from pathlib import Path

def sanitize_directory_input(user_input):
    """
    Sanitize and validate the user-provided directory path.
    """
    
    # Convert to Path object for easier manipulation
    path = Path(user_input).resolve()
    
    # Define allowed base directories (adjust these as needed)
    allowed_bases = [
        Path.home() / "Downloads",
        Path.home() / "Documents",
        Path("/tmp")  # For Unix-like systems
    ]
    
    # Check if the path is within allowed directories
    if not any(base in path.parents for base in allowed_bases):
        raise ValueError("Access to this directory is not allowed. Only /Downloads and /Documents is allowed")
    
    # Ensure the directory exists
    if not path.is_dir():
        raise ValueError("The specified directory does not exist.")
    
    return str(path)

def remove_quotes_if_present(input_str):
    # Check if the string starts and/or ends with quotes
        if input_str.startswith('"'):
            input_str = input_str[1:]  # Remove the starting quote
        if input_str.endswith('"'):
            input_str = input_str[:-1]  # Remove the ending quote
        return input_str

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
                image = image.rotate(90, expand=True) #counter clockwise I think
    except (AttributeError, KeyError, IndexError):
        # If there's no EXIF data or no orientation tag, we skip the rotation.
        pass

    return image

def rotate_one_file_streamlit():
    #uploaded_image = st.file_uploader("Upload JPEG or JPG Image", type=["pdf"], accept_multiple_files=False)

   
    #file = remove_quotes_if_present(st.text_input("file name"))
    uploaded_image = st.file_uploader("Upload", type=["pdf", "jpg", "png"], accept_multiple_files=False)
        
        
    degrees_ = st.selectbox("number of degrees",[90,180,270],0)
    if st.button("Rotate"):
        if uploaded_image==None:
            st.error("Upload a file")
            st.stop()
        file = uploaded_image.name
        ext = file[-3:]
  
        if ext == "pdf":
            
            #file = uploaded_image.name
            try:
        
                reader = PdfReader(file)
                writer = PdfWriter()
                for page in reader.pages:
                    # page.rotate_clockwise(270) # (before pypdf3.0 - deprecated - thanks to Maciejg for the update)
                    page.rotate(degrees_)
                    writer.add_page(page)
                with open(file, "wb") as pdf_out:
                    writer.write(pdf_out)
                    st.success("Done")

            except Exception as e:
                st.error(f"Error {e} ")
        else:
                    
            if degrees_ == 90:
                degrees = 270
            elif degrees_ == 270:
                degrees = 90    
            else:
                degrees = 180
            placeholder = st.empty()
           
            try:
                with Image.open(file) as image:
                
                    image = image.rotate(degrees, expand=True)
                    # Shows the image in image viewer 
                    filename_tosave = f"{file[:-4]}.{ext}"
                    image.save(filename_tosave) 
            except FileNotFoundError:
                print(f'File not found: {file}')
            except Exception as e:
                print(f'Error renaming {file}: {e}')
        st.success ("Done")  

def merge_multiple_images_to_one_pdf():
    """ Convert one or multiple files into a pdf
        Uses streamlit, Adviced is to use a split screen to easy drag and drop.  
        based on https://www.youtube.com/watch?v=RPN-HxvAQnQ
    """     
    
             
    # Image file upload
    col1,col2,col3 = st.columns(3)
    with col1:
        uploaded_images_1 = st.file_uploader("Upload JPEG or JPG Image 1", type=["jpeg", "jpg","png"], accept_multiple_files=False)
    with col2:
        uploaded_images_2 = st.file_uploader("Upload JPEG or JPG Image 2", type=["jpeg", "jpg","png"], accept_multiple_files=False)
    with col3:
        uploaded_images_3 = st.file_uploader("Upload JPEG or JPG Image 3", type=["jpeg", "jpg","png"], accept_multiple_files=False)
    col4,col5,col6 = st.columns(3)
    
    
    with col4:
        uploaded_images_4 = st.file_uploader("Upload JPEG or JPG Image 4", type=["jpeg", "jpg","png"], accept_multiple_files=False)
    with col5:
        uploaded_images_5 = st.file_uploader("Upload JPEG or JPG Image 5", type=["jpeg", "jpg","png"], accept_multiple_files=False)
    with col6:
        uploaded_images_6 = st.file_uploader("Upload JPEG or JPG Image 6", type=["jpeg", "jpg","png"], accept_multiple_files=False)
    
    
    uploaded_images = [uploaded_images_1,  uploaded_images_2, uploaded_images_3, uploaded_images_4, uploaded_images_5,uploaded_images_6]
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
                #with Image.open(uploaded_image) as image:
                image = Image.open(uploaded_image)
                image = correct_image_rotation(image)
                if i==0:
                    # if you want to use the filename of the first image
                    filename_proposed =uploaded_image.name[:-4]

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

def split_landscape_into_two_portrait_one_file_streamlit():

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
   
    uploaded_image = st.file_uploader("Upload JPEG or JPG Image 1", type=["jpeg", "jpg","png"], accept_multiple_files=False)
    
    if st.button("Split image"):
        if uploaded_image==None:
            st.error("Upload a file")
            st.stop()
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

def merge_directory_to_one_pdf(dir_name):
    """ Making ONE pdf from a list of images
        based on https://stackoverflow.com/a/47283224/4173718 CC BY 4.0
        install by > python3 -m pip install --upgrade Pillow  
        # ref. https://pillow.readthedocs.io/en/latest/installation.html#basic-installation
        To use if you want
        - to convert a list of files into one pdf
        - to convert a complete directory into one pdf
        - to converrt files with the same name (xxxx01.pdf .. xxxx04.pdf), into one pdf
    """     
    destination_file =st.text_input("Destination file", "converted.pdf")
    delete= st.checkbox("Delete original images", False)
    placeholder1 = st.empty()

    pdf_path = dir_name+ r"/" +destination_file

    #dir_name = r"C:\Users\rcxsm\Downloads\Sheetmusic\personal fakebook"
    os.chdir(dir_name)  # change directory from working dir to dir with files
    files = []
    if st.button("Merge"):
        for file in os.listdir(dir_name):  # loop through items in dir
            try:
                if file[-3:] !="pdf":
                    placeholder1.write (file)
                    files.append(file)
            
            except FileNotFoundError:
                st.error(f'File x not found: {file}')
            except Exception as e:
                st.error(f'Error x  {file}: {e}') 

        try:  
            Image.open(files[0]).save(
                pdf_path, "PDF", resolution=100.0, save_all=True, 
                append_images=(Image.open(f) for f in files[1:])) 
        except FileNotFoundError:
            st.error(f'File y not found: {file}')
        except Exception as e:
            st.error(f'Error y  {file}: {e}') 

        st.success(f"{pdf_path} saved")
        time.sleep(1)
        if delete: 
            for f in files:
                try:
                    os.remove(dir_name + f)
                    st.write(f"{dir_name+f} deleted")
                except Exception as e:
                    st.write(f"Can't delete file {e}")

def auto_contrast_color_one_file_streamlit():
    """auto contrast color of one single uploaded file
    """    
    uploaded_image = st.file_uploader("Upload JPEG or JPG Image", type=["jpeg", "jpg","png"], accept_multiple_files=False)
    c = st.number_input("Cutoff", 0,100,3)
    include_cutoff= st.checkbox("Include cut off in filename (keeps original)", False)
    if st.button("Auto adjust color"):
        try:
            im = Image.open(uploaded_image)
            im = ImageOps.autocontrast(im, cutoff =c )
            im.show()
            filename =uploaded_image.name[:-4]
            ext = uploaded_image.name[-3:]
            if include_cutoff:
                filename_tosave = f"{filename}-{c}-.{ext}"
            else:
                filename_tosave = f"{filename}.{ext}"

            im.save(filename_tosave)
            st.write(f"{filename_tosave} saved")
        except FileNotFoundError:
            st.write(f'File not found: {filename}')
        except Exception as e:
            st.write(f'Error auto contrasting {filename}: {e}')

def read_song_titles(dir_name):
    """Read the files in the directory, read the first 50 characters and rename the filename.
    """        
    # dir_name = r"C:\Users\rcxsm\Downloads\Sheetmusic\pianointros"
    #dir_name = remove_quotes_if_present(st.text_input("directory",r"C:\Users\rcxsm\Downloads\test"))
    if st.button("Read song titles"):
        os.chdir(dir_name)  # change directory from working dir to dir with files
        files = []
        l = len(os.listdir(dir_name))
        files = os.listdir(dir_name)
    
        for i,file in enumerate(files):  # loop through items in dir
            print (f"{i+1}/{l} - {file}")
            try:
                with Image.open(file) as image:
                    filename_proposed = str(((pytesseract.image_to_string(image))))
                    filename_tosave = clean_filename(filename_proposed[:50], replace_with='_')
                    image.save(f"{filename_tosave}.png", "png", resolution=100.0)
                    st.write ("{filename_tosave} saved")
            except Exception as e:
                st.write (f"ERROR {file} - {e}")
        st.success("DONE")
def convert_directory_seperate_files_in_directory(dir_name):
    """Convert the images in a directory in to seperate pdfs
    """        
    #dir_name = remove_quotes_if_present(st.text_input("directory",r"C:\Users\rcxsm\Downloads\test"))
    placeholder = st.empty()
    os.chdir(dir_name)  # change directory from working dir to dir with files
    files = []
    l = len(os.listdir(dir_name))
    files = os.listdir(dir_name)
    if st.button("Convert into single files"):
        for i,file in enumerate(files):  # loop through items in dir
            placeholder.info (f"{i+1}/{l} - {file}")
            # try:
            try:
                if file[-3:] !="pdf":
                    filename =file[:-4]
                    
                    filename_tosave = f"{filename}.pdf"
                    with Image.open(file) as im:
                    
                        im.save(filename_tosave, "PDF", resolution=100.0)
                        # im.save(f"{filename_tosave}")
                        st.write(f"{filename_tosave} saved")
            except FileNotFoundError:
                st.write(f'File not found: {filename}')
            except Exception as e:
                st.write(f'Error  {filename}: {e}')
        st.success ("DONE")

def auto_contrast_directory(dir_name):
        """Maximize (normalize) image contrast. This function calculates a
        histogram of the input image (or mask region), removes ``cutoff`` percent of the
        lightest and darkest pixels from the histogram, and remaps the image
        so that the darkest pixel becomes black (0), and the lightest
        becomes white (255).
        """        
        #dir_name =  remove_quotes_if_present(st.text_input("directory",r"C:\Users\rcxsm\Downloads\test"))
        c = st.number_input("Cutoff", 0,100,3)
        include_cutoff= st.checkbox("Include cut off in filename (keeps original)", False)
        placeholder = st.empty()
        
        os.chdir(dir_name)  # change directory from working dir to dir with files
        files = []
        l = len(os.listdir(dir_name))
        files = os.listdir(dir_name)
       
        if st.button("Adjust colors"):
            for i,file in enumerate(files):  # loop through items in dir
                placeholder.info (f"{i+1}/{l} - {file}")
                #im = Image.open(file) 
                try:
                    ext = file[-3:]
                    if (ext=="jpg") or (ext=="png"):
                        with Image.open(file) as im:

                            im = ImageOps.autocontrast(im, cutoff = c)
                            filename =file[:-4]
                            if include_cutoff:
                                filename_tosave = f"{filename}{c}.{ext}"
                            else:
                                filename_tosave = f"{filename}.{ext}"
                            im.save(filename_tosave)
                            st.write(f"{filename_tosave} saved")
                except FileNotFoundError:
                    print(f'File not found: {filename}')
                except Exception as e:
                    print(f'Error renaming {filename}: {e}')            
            st.success ("DONE")

def rotate_image_directory(dir_name):
    """rotate all images in a directory
    """    
    #dir_name =  remove_quotes_if_present(st.text_input("directory",r"C:\Users\rcxsm\Downloads\test"))
    degrees_ = st.selectbox("number of degrees",[90,180,270],0)
    if degrees_ == 90:
        degrees = 270
    elif degrees_ == 270:
        degrees = 90    
    else:
        degrees = 180
    placeholder = st.empty()
    os.chdir(dir_name)  # change directory from working dir to dir with files
    files = []
    l = len(os.listdir(dir_name))
    files = os.listdir(dir_name)
    if st.button("Rotate all images in directory"):
        for i,file in enumerate(files):  # loop through items in dir
            try:
                ext = file[-3:]

                if (ext == "jpg") or (ext =="jpeg") or (ext == "png"):
                    placeholder.write (f"{i+1}/{l} - {file}")
                    with Image.open(file) as image:
                    
                        image = image.rotate(degrees, expand=True)
                        # Shows the image in image viewer 
                        filename_tosave = f"{file[:-4]}.{ext}"
                        image.save(filename_tosave) 
            except FileNotFoundError:
                print(f'File not found: {file}')
            except Exception as e:
                print(f'Error renaming {file}: {e}')
        st.success ("Done")   

def resize_directory(dir_name):
    """resize directory
    """        
    #dir_name =  remove_quotes_if_present(st.text_input("directory",r"C:\Users\rcxsm\Downloads\test"))
    new_size_perc = st.number_input("Change in size in 50%", 0,1000,50)/100
    include_in_filename = st.checkbox("Include resized in filename (keeps original)", False)
    placeholder = st.empty()
    os.chdir(dir_name)  # change directory from working dir to dir with files
    files = []
    l = len(os.listdir(dir_name))
    files = os.listdir(dir_name)
    if st.button("Resize"):
        for i,file in enumerate(files):  # loop through items in dir
            placeholder.info (f"{i+1}/{l} - {file}")
            #im = Image.open(file) 
            try:
                if (file[-3:] =="jpg") or (file[-3:] =="png"):
                    with Image.open(file) as im:
                
                    
                        # Size of the image in pixels (size of original image) 
                        # (This is not mandatory) 
                        width, height = im.size 

                        # Setting the points for cropped image 
                        w2 = int(width*new_size_perc)
                        h2 = int(height*new_size_perc)
                        # Cropped image of above dimension 
                        # (It will not change original image) 
                        #im1 = im.crop((left, top, right, bottom))
                        newsize = (w2, h2)
                        im1 = im.resize(newsize)
                        # Shows the image in image viewer 
                        if include_in_filename:
                            filename_tosave = f"{file[:-4]}_resized_{new_size_perc*100}.{file[-3:]}"
                        else:
                            filename_tosave = f"{file[:-4]}.{file[-3:]}"
                        im1.save(filename_tosave) 
            except FileNotFoundError:
                st.write(f'File not found: {file}')
            except Exception as e:
                st.write(f'Error resizing {file}: {e}')
        st.success ("Done")   

def rotate_pdf_directory(dir_name):
    """Rotates the pdfs in a directory
    """   
    #dir_name =  remove_quotes_if_present(st.text_input("directory",r"C:\Users\rcxsm\Downloads\test"))
    degrees = st.selectbox("number of degrees",[90,180,270],0)
    placeholder = st.empty()
     # r"C:\Users\rcxsm\Downloads\Sheetmusic\personal fakebook"
    os.chdir(dir_name)  # change directory from working dir to dir with files
    files = []
    l = len(os.listdir(dir_name))
    files = os.listdir(dir_name)
    if st.button("Rotate directory PDF"):
        for i,file in enumerate(files):  # loop through items in dir 
            if file[-3:] =="pdf":
                placeholder.info (f"{i+1}/{l} - {file}")
                fname_pdf = file[:-4] +".pdf"
                try:
                    reader = PdfReader(file)
                    writer = PdfWriter()
                    for page in reader.pages:
                        # page.rotate_clockwise(270) # (before pypdf3.0 - deprecated - thanks to Maciejg for the update)
                        page.rotate(degrees)
                        writer.add_page(page)
                    with open(fname_pdf, "wb") as pdf_out:
                        writer.write(pdf_out)
                    st.write(f"DONE {file}")
                except Exception as e:
                    st.error(f"Error {file} - {e}")
        st.info("DONE")

def rename_files_via_dictionary(directory):

    # Define the dictionary with original file names as keys and the new cleaned-up names as values
    
    
    file_rename_dict = {
        "ABCD.png": "abcd.png",
        "EFG.png": "efg.png",
        
    }
    st.write(file_rename_dict)
    # Directory where the files are located
    #directory = r"C:\Users\rcxsm\Downloads\Sheetmusic\pianointros"
    #directory = remove_quotes_if_present(st.text_input("directory",r"C:\Users\rcxsm\Downloads\test"))
    placeholder = st.empty()
    # Rename the files
    if st.button("Replace filenames"):
        for old_name, new_name in file_rename_dict.items():
            old_path = os.path.join(directory, old_name)
            new_path = os.path.join(directory, new_name)
            
            # Rename the file
            try:
                os.rename(old_path, new_path)
                st.write(f'Renamed: {old_name} to {new_name}')
            except FileNotFoundError:
                st.write(f'File not found: {old_name}')
            except Exception as e:
                st.write(f'Error renaming {old_name}: {e}')
        placeholder.success ("DONE")

def find_and_replace_in_filename(directory):
    """Find and replace in filename
    """    
    from_ =  st.text_input("Find",".jpg")
    to = st.text_input("Replace with","intro.jpg")
    placeholder=st.empty()
    if st.button("Find and replace in filenames"):
        # Directory where the files are located
        # directory = r"C:\Users\rcxsm\Downloads\Sheetmusic\pianointros_x"
        l = len(os.listdir(directory))
        # Loop through the files in the directory
        for i,filename in enumerate(os.listdir(directory)):
            # Create the new filename with "- piano intro" appended
            new_filename = filename.replace(from_, to)
            
            # Define the full old and new paths
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            
            # Rename the file
            try:
                os.rename(old_path, new_path)
                placeholder.info(f'{i+1}/{l} -Renamed: {filename} to {new_filename}')
            except FileNotFoundError:
                print(f'File not found: {filename}')
            except Exception as e:
                print(f'Error renaming {filename}: {e}')
        st.success ("DONE")

def get_secure_directory_input(default_dir):
    """
    Get a secure directory input from the user using Streamlit.
    """
    user_input = st.text_input("Enter directory path:", default_dir)
    if user_input:
        try:
            sanitized_path = sanitize_directory_input(remove_quotes_if_present(user_input))
            #st.success(f"Using directory: {sanitized_path}")
            return sanitized_path
        except ValueError as e:
            st.error(str(e))
            return None

def main():
    # Streamlit UI
    st.title("JPEG/JPG to PDF Converter")
    
    choice = st.selectbox("What to do", ["merge images to one pdf","rotate one file pdf", "split image in two","merge directory", 
                                         "auto contrast color_one file", 
                                        "resize directory", "auto contrast directory", 
                                        "rotate image directory", "convert separate files", 
                                        "rotate pdf directory", "read song titles", 
                                        "rename intro", "find and replace in filename"], 0)
    
    # this is staying up all the time, to make successive batch processing easier
    #dir_name = remove_quotes_if_present(st.text_input("directory",))
    dir_name = get_secure_directory_input(r"C:\Users\rcxsm\Downloads\test")
    
    if choice == "merge images to one pdf":
        merge_multiple_images_to_one_pdf()
    elif choice == "rotate one file":
        rotate_one_file_streamlit() 
    elif choice == "auto contrast colort one_file":
        auto_contrast_color_one_file_streamlit() 
    
    elif choice=="merge directory":
        merge_directory_to_one_pdf(dir_name)
    elif choice == "split image in two":
        split_landscape_into_two_portrait_one_file_streamlit()
    elif choice == "resize directory":
        resize_directory(dir_name)
    elif choice == "auto contrast directory":
        auto_contrast_directory(dir_name) 
    elif choice == "rotate image directory":
        rotate_image_directory(dir_name) 
    elif choice == "convert separate files":
        convert_directory_seperate_files_in_directory(dir_name) 
    elif choice == "rotate pdf directory":
        rotate_pdf_directory(dir_name) 
    elif choice == "read song titles":
        read_song_titles(dir_name) 
    elif choice == "rename intro":
        rename_files_via_dictionary(dir_name) 
    elif choice == "find and replace in filename":
        find_and_replace_in_filename(dir_name) 
    else:
        st.warning("ERROR")
        st.stop()

if __name__ == "__main__":
    import datetime
    os.system('cls')

    print(f"--------------{datetime.datetime.now()}-------------------------")

    
    main()
    #resize_directory()
    #main_()
    #auto_contrast_directory()
    #rotate_directory()
    #convert_directory_seperate_files()
    #rotate_pdf()
    #rotate_one_file_pdf_streamlit()
    #convert_directory_seperate_files()
    #read_pianointo_titles()
    #rename_intro()
    #edit_filename()
    #st.write("-------------_")