import json
import streamlit as st

# Replace 'yourfile.json' with the actual file name
def main():
    pdf_path = st.file_uploader("Choose a file")
    if pdf_path is not None:
        try:
            # reader = PdfReader(pdf_path)
            #with open(r"C:\Users\rcxsm\Downloads\chatgpt_archive\conversations.json", 'r', encoding='utf-8') as file:
            data = json.load(pdf_path )
        except Exception as e:
            st.error(f"Error loading / parsing the json file: {str(e)}")
            st.stop()
    else:
        st.warning("You need to upload a json file. Files are not stored anywhere after the processing of this script")
        st.stop()
    expanders = st.checkbox("Use expanders")
    if expanders:
        show_json_expanders(data)
    else:
        show_json_text(data)

def show_json_expanders(data):

    # Process each entry
    for i,entry in enumerate(data[::-1]):
        title = entry.get('title')
    
        with st.expander(f"Title: {title}", expanded=False):
                
            mapping = entry.get('mapping', {})
            for key, message_info in mapping.items():
                message = message_info.get('message', {})
                if not message:
                    continue
                content = message.get('content', {}).get('parts', [])

                # st.write each content part
                for part in content:
                    st.write(f"{part}")



def show_json_text(data):

    # Process each entry
    for i,entry in enumerate(data[::-1]):
        title = entry.get('title')
    
        st.subheader(f"Title: {title}")
                
        mapping = entry.get('mapping', {})
        for key, message_info in mapping.items():
            message = message_info.get('message', {})
            if not message:
                continue
            content = message.get('content', {}).get('parts', [])

            # st.write each content part
            for part in content:
                st.write(f"{part}")
                
if __name__ == "__main__":
    main()   