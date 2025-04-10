import json
import streamlit as st


def show_body(mapping):
    """
    Processes and displays the content of a mapping object from the ChatGPT `conversations.json` file.

    This function iterates through the `mapping` dictionary, extracts messages, and displays their content
    based on the role of the author (e.g., user, assistant, or other). It uses the `show_part` function to
    handle and display individual message parts.

    Args:
        mapping (dict): A dictionary containing the conversation mapping, where each key corresponds to
                        a message ID and the value contains message details such as content and author role.

    Behavior:
    - Skips messages that are empty or do not contain valid content.
    - Extracts the role of the author (e.g., 'user', 'assistant') and the message content.
    - Passes each part of the message content to the `show_part` function for further processing and display.
    """
    for key, message_info in mapping.items():
        message = message_info.get("message", {})

        if not message:
            continue
        role = message.get("author", {}).get("role", {})
        content = message.get("content", {}).get("parts", [])

        # Check if part is a string or a JSON-like object
        for part in content:
            show_part(part, role)


def show_part(part, role):
    """
    Displays a single part of a message based on the author's role.

    This function processes and displays a message part depending on its type and the role of the author.
    If the part is a string, it is displayed differently for 'user', 'assistant', or other roles.
    Non-string parts and strings starting with "{" (eg. image descriptions or thinking steps) are ignored.

    Args:
        part (str or dict): The content of the message part. If it's a string, it will be displayed.
                            If it's a JSON-like object (e.g., dict), it will be ignored.
        role (str): The role of the author of the message (e.g., 'user', 'assistant', or other).

    Behavior:
    - If the part is a string and does not start with "{", it is displayed:
        - As an info box for 'user'.
        - As plain text for 'assistant'.
        - As code for other roles.
    - Non-string parts and strings starting with "{" (eg. image descriptions or thinking steps) are ignored.

    Example:
        part = "Hello, how can I help you?"
        role = "assistant"
        show_part(part, role)
    """
    
    if isinstance(part, str):
        # st.write(f"{part}")
        if part and part[0] != "{":
            # st.write(f"{part}")
            if role == "user":
                st.info(f"{part}")
            elif role == "assistant":
                st.write(part)
            elif role == "tool":
                # omit texts like
                # GPT-4o returned 1 images. From now on, do not say or show
                # st.warning(part)
                # and other thinking steps
                pass
            else:
                st.code(part)
    else:
        # st.write("Non-string content detected:")
        # st.json(part)  # Display the JSON-like object in a readable format
        # it gives back a dictionary, mostly a 'image_asset_pointer'
        generated_images = part.get("asset_pointer", {})
        for r in ["file-service://","sediment://"]:
            generated_images = generated_images.replace(r, "[File] : ")
        st.success(f"Generated_images: {generated_images}.dat\n")


def show_json_expanders(data):
    """
    Displays the content of a ChatGPT `conversations.json` file using collapsible expanders.

    This function processes each entry in the provided data and displays the title of each conversation
    as an expander. Within each expander, the conversation messages are displayed using the `show_body` function.

    Args:
        data (list): A list of conversation entries, where each entry is a dictionary containing:
                     - 'title': The title of the conversation.
                     - 'mapping': A dictionary mapping message IDs to message details.

    Behavior:
    - Iterates through the provided data and displays the title of each conversation as an expander.
    - Extracts the 'mapping' field from each entry and processes it using the `show_body` function.
    """
    # Process each entry
    for i, entry in enumerate(data):
        title = entry.get("title")

        with st.expander(f"Title: {title}", expanded=False):

            mapping = entry.get("mapping", {})
            show_body(mapping)


def download_text(data, page_number, total_pages):
    """
    Generates and allows downloading of conversation data as a plain text file.

    This function processes the provided conversation data, formats it into a plain text structure,
    and provides a download button for the user to save the content as a `.txt` file. It also displays
    the formatted text in the Streamlit app.

    show_body and show_part are integrated in this function

    Args:
        data (list): A list of conversation entries, where each entry is a dictionary containing:
                     - 'title': The title of the conversation.
                     - 'mapping': A dictionary mapping message IDs to message details.
        page_number (int): The current page number being displayed.
        total_pages (int): The total number of pages available.

    Behavior:
    - Iterates through the provided data and extracts the title and conversation messages.
    - Formats the messages based on the author's role:
        - 'user': Messages are enclosed in a box-like structure with separators.
        - 'assistant': Messages are displayed as plain text.
        - Other roles: Messages are displayed as plain text.
    - Appends a separator between conversations for clarity.
    - Generates a filename based on the current page and total pages.
    - Provides a download button for the user to save the formatted text as a `.txt` file.
    - Displays the formatted text in the Streamlit app for preview.

    Outputs:
    - Displays the formatted text in the Streamlit app.
    - Provides a download button for saving the text as a `.txt` file.

    """
    complete_text = ""
    for i, entry in enumerate(data):
        title = entry.get("title")
        complete_text += f"Title: {title}\n"

        mapping = entry.get("mapping", {})

        for key, message_info in mapping.items():

            message = message_info.get("message", {})
            if not message:
                continue
            content = message.get("content", {}).get("parts", [])
            role = message.get("author", {}).get("role", {})
            filenames = message.get("metadata", {}).get("attachments", [])

            for f in filenames:

                filename = f.get("name", "")
                complete_text += f"Uploaded file: {filename}\n"

            for part in content:
                if isinstance(part, str):
                    # if it gives back a string
                    if part and part[0] != "{":
                        if role == "user":
                            # the prompt
                            complete_text += "---------------------\n"
                            complete_text += f"| {part}\n"
                            complete_text += "---------------------\n"
                        elif role == "assistant":
                            # the answer
                            complete_text += f"{part}\n"
                        elif role == "tool":
                            # omit texts like
                            #
                            # GPT-4o returned 1 images. From now on, do not say or show
                            # ANYTHING. Please end this turn now. I repeat: From now on,
                            # do not say or show ANYTHING. Please end this turn now.
                            # Do not summarize the image. Do not ask followup question.
                            # Just end the turn and do not do anything else.
                            #
                            # and other thinking steps
                            complete_text += f""
                        else:
                            complete_text += f"{part}\n"
                else:
                    # it gives back a dictionary, mostly a 'image_asset_pointer'
                    generated_images = part.get("asset_pointer", {})
                    generated_images = generated_images.replace(
                        "sediment://", "[File]:"
                    )
                    complete_text += f"Generated_images: {generated_images}.dat\n"

        complete_text += "\n===================================\n\n"
    filename = f"output_page_{page_number}_of_{total_pages}.txt"
    st.download_button(
        "Download txt", data=complete_text, file_name=filename, mime="text/plain"
    )
    st.code(complete_text, language="text")


def show_json_text_inline(data):
    """
    Displays the content of a ChatGPT `conversations.json` file in an inline format.

    This function processes each entry in the provided data and displays the title and corresponding
    conversation messages inline. It uses the `show_body` function to handle the mapping of messages
    and the `show_part` function to display individual message parts based on the author's role.

    Args:
        data (list): A list of conversation entries, where each entry is a dictionary containing:
                     - 'title': The title of the conversation.
                     - 'mapping': A dictionary mapping message IDs to message details.

    Behavior:
    - Iterates through the provided data and displays the title of each conversation as a subheader.
    - Extracts the 'mapping' field from each entry and processes it using the `show_body` function.
    - For each message in the mapping:
        - Extracts the author's role (e.g., 'user', 'assistant').
        - Extracts the message content and displays each part using the `show_part` function.
    """
    # Process each entry
    for i, entry in enumerate(data):
        title = entry.get("title")
        st.subheader(f"Title: {title}")
        mapping = entry.get("mapping", {})
        show_body(mapping)

        for key, message_info in mapping.items():
            message = message_info.get("message", {})

            if not message:
                continue
            else:
                role = message.get("author", {}).get("role", {})
                content = message.get("content", {}).get("parts", [])
                if not content:
                    continue
                else:
                    for part in content:
                        show_part(part, role)


def main():
    """
    Main function to process and display the contents of a ChatGPT `conversations.json` file.

    chat.html in the zip-file does the same thing :)

    Features:
    - Allows the user to upload a `conversations.json` file.
    - Provides options to display the content in different modes:
        - "Text only+download": Displays the content as plain text and allows downloading it as a `.txt` file.
        - "Inline": Displays the content inline with roles and messages.
        - "Expanders": Displays the content in collapsible expanders for better organization.
    - Supports pagination with a slider to navigate through the data.
    - Allows reversing the order of the data (most recent first or most old first).
    - Handles JSON parsing errors gracefully and provides user feedback.

    Inputs:
    - JSON file uploaded by the user. Json file structure example:
        [
            {
                "title": "Conversation 1",
                "mapping": {
                    "key1": {
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["Hello!"]}
                        }
                    },
                    "key2": {
                        "message": {
                            "author": {"role": "assistant"},
                            "content": {"parts": ["Hi there!"]}
                        }
                    }
                }
            }
        ]
    - User-selected options for display mode, number of elements per page, and page navigation.

    Outputs:
    - Displays the content in the selected mode.
    - Provides a download button for the "Text only+download" mode.

    Raises:
    - Displays an error message if the JSON file cannot be loaded or parsed.

    """
    st.title("Read conversations.json")
    st.markdown(
        "This script reads the conversations.json file from ChatGPT and displays the content in a user-friendly format."
    )
    pdf_path = st.file_uploader("Choose a file")
    if pdf_path is not None:
        try:
            # reader = PdfReader(pdf_path)
            # with open(r"C:\Users\rcxsm\Downloads\chatgpt_archive\conversations.json", 'r', encoding='utf-8') as file:
            data = json.load(pdf_path)
            len_data = len(data)

        except Exception as e:
            st.error(f"Error loading / parsing the json file: {str(e)}")
            st.stop()
    else:
        st.warning(
            "You need to upload a json file. Files are not stored anywhere after the processing of this script"
        )
        st.stop()

    reversed = st.selectbox("Order", ["Most old first", "Most recent first"])

    col2, col3 = st.columns([1, 1])

    if reversed == "Most old first":
        data = data[::-1]
    with col2:
        modus = st.selectbox(
            "Modus [Text only+download| Inline | Expanders | Show JSON]",
            ["Text only+download", "Inline", "Expanders", "Show JSON"],
            index=3,
        )
    with col3:
        number_of_elements = st.number_input(
            f"Number of elements to show (max {len_data})", 1, len_data, 5
        )

    items_per_page = number_of_elements

    # Calculate the total number of pages
    total_pages = (len_data + items_per_page - 1) // items_per_page

    # Add a slider to select the page number
    page_number = st.slider("Page", 1, total_pages, 1)

    # Calculate the start and end indices for the current page
    start_index = (page_number - 1) * items_per_page
    end_index = start_index + items_per_page

    # Slice the data for the current page
    data = data[start_index:end_index]

    if modus == "Expanders":
        show_json_expanders(data)
    elif modus == "Inline":
        show_json_text_inline(data)
    elif modus == "Text only+download":
        download_text(data, page_number, total_pages)
    elif modus == "Show JSON":
        st.json(data)
        # print(json.dumps(data, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
