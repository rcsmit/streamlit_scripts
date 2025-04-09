import json
import streamlit as st

def show_part(part, role):
    if isinstance(part, str):
        # st.write(f"{part}")
        if part and part[0] != "{":
            # st.write(f"{part}")
            if role == 'user':
                st.info(part)
            elif role == 'assistant':
                st.write(part)
            else:
                st.code(part)
    else:
        pass
        # st.write("Non-string content detected:")
        # st.json(part)  # Display the JSON-like object in a readable format



def show_json_expanders(data):

    # Process each entry
    for i,entry in enumerate(data):
        title = entry.get('title')
    
        with st.expander(f"Title: {title}", expanded=False):
              
            mapping = entry.get('mapping', {})
            show_body(mapping)
            
            # for key, message_info in mapping.items():
            #     message = message_info.get('message', {})
            #     role = message.get('role', 'user')
            #     if not message:
            #         continue
            #     content = message.get('content', {}).get('parts', [])

            #     # st.write each content part
            #     for part in content:
            #         show_part(part, role)

    
def show_json_text(data):
    
    # Process each entry
    for i,entry in enumerate(data):
        title = entry.get('title')
    
        st.subheader(f"Title: {title}")
        mapping = entry.get('mapping', {})
        show_body(mapping)
            

def show_body(mapping):        
       
        for key, message_info in mapping.items():
            message = message_info.get('message', {})
           
            if not message:
                continue
            role = message.get('author',{}).get('role',{})
            content = message.get('content', {}).get('parts', [])

            # st.write each content part
            # for part in content:
            #     st.write(f"{part}")
  
                # Check if part is a string or a JSON-like object
            for part in content:
                show_part(part,role)


# Replace 'yourfile.json' with the actual file name
def main():
    st.title("Read conversations.json")
    st.markdown("This script reads the conversations.json file from ChatGPT and displays the content in a user-friendly format.")   
    pdf_path = st.file_uploader("Choose a file")
    if pdf_path is not None:
        try:
            # reader = PdfReader(pdf_path)
            #with open(r"C:\Users\rcxsm\Downloads\chatgpt_archive\conversations.json", 'r', encoding='utf-8') as file:
            data = json.load(pdf_path )
            len_data = len(data)

        except Exception as e:
            st.error(f"Error loading / parsing the json file: {str(e)}")
            st.stop()
    else:
        st.warning("You need to upload a json file. Files are not stored anywhere after the processing of this script")
        st.stop()
    col1,col2,col3=st.columns([1,1,2])


    with col1:
        reversed = st.checkbox("Oldest to Newest", True)
    with col2:
        expanders = st.checkbox("Use expanders", True)
    with col3:
        number_of_elements = st.number_input(f"Number of elements to show (max {len_data})",1,len_data,5)
    
    if reversed:
        data= data[::-1]
    # Limit to the first 5 elements
    data = data[:number_of_elements]
    if expanders:
        show_json_expanders(data)
    else:
        show_json_text(data)

if __name__ == "__main__":
    main()   