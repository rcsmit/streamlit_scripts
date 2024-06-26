import pandas as pd
import random

import streamlit as st



def take_random_value(df, column):
    """Take a random value in a given column

    Args:
        column (str): the column where you have to choose a value from

    Returns:
        str: the chosen value
    """    
    column_values = df[column].dropna().unique()
    
    if len(column_values) > 0:
        random_value = random.choice(column_values)
        st.write (f"{column} : {random_value}")
    else:
        random_value = None
    return random_value

def generate_prompt(df,included_columns, what,who, number,chaos, stylize, weird,ar, seperator):
    """Generate the prompt :
     ["CONCEPT"] [what] as ["ARCHETYPES"] in a [SCENES"] in the style of [who]
     eg. An action shot of  Muhammad Ali as poet in a grotto in the style of Henri Rousseau 
   

    Args:
        df (df): the dataframe
        what (str): Choose from: ["FAMOUS PEOPLE", "ANIMALS", "OBJECTS", "ABOUT"]
        who (str): Choose from ["FAMOUS PAINTERS", "MASTERPHOTOGRAPHERS","ARTISTS" ]
        number (int): number of extra additions to the prompt
        fixed_columns (int): Number of the columns not taken in account in the extra additions of the prompt
        chaos (int): level of chaos [0-100]
        stylize (int): level of stylize [0-1000]
        weird (int): level of weirdness [0-3000]
        ar (str): aspect ratio'
        seperator (str) : seperator ["|", ",", " "]
    """    
    place1 = st.empty()
    place2 = st.empty()
    place3 = st.empty()
    if number >0:
        selected_columns =  random.sample(included_columns, number)
    else:
        selected_columns = None
    #selected_columns =  random.sample(list(df.columns)[fixed_columns:], number) # Exclude the first column and select a random sample of 8 column names
    #print (list(df.columns)[fixed_columns:])
    if what == "FAMOUS PEOPLE":
        prompt = f'{take_random_value(df, "CONCEPT")} {take_random_value(df, what)} as {take_random_value(df, "ARCHETYPES")} in a {take_random_value(df, "SCENES")} by {take_random_value(df, who)} '
    elif what=="ANIMALS":
        prompt = f'{take_random_value(df, "CONCEPT")} a {take_random_value(df, what)} as {take_random_value(df, "ARCHETYPES")} in a {take_random_value(df, "SCENES")} by {take_random_value(df, who)} '
    elif what == "INTERIOR ARCHITECTURE":
         prompt =f'{take_random_value(df, "COMPOSITION_INT_ARCH")}'
    elif what == "MAELA":
         prompt =f' '
    else:
        prompt = f'{take_random_value(df, "CONCEPT")} {take_random_value(df, what)} in a {take_random_value(df, "SCENES")} by {take_random_value(df, who)}'
      
    prompt2 = prompt
    prompt3 = prompt
    if seperator =="|":
        seperator = " | "
    elif seperator ==",":
        seperator ==", "
    elif seperator =="<space>":
        seperator ==" "
    else:
        st.error("ERROR IN SEPERATOR")
        st.stop()
    if selected_columns != None:
        for column in df.columns:
            if column in selected_columns:
                random_value = take_random_value(df, column)
                prompt += f"{seperator}{random_value}::{random.randint(1, 100)}"
                prompt2 += f"{seperator}{random_value}"
                prompt3 += f"{seperator}{column}: {random_value}"
   
    if what == "INTERIOR ARCHITECTURE":
        prompt += " with exclusive finishes and minimalist detailing throughout, intrinsic details " 
    if what == "MAELA":
        prompt += " high quality photo 16k "
        ar =  "2:3"
    ending = f"--chaos {chaos} --stylize {stylize}  --weird {weird} --ar {ar} --style raw"
    prompt += ending
    prompt2 += ending
    prompt3 += ending

    place1.success (prompt2) 
    place2.code (prompt2)
    place3.code (prompt3)
    
    st.subheader("Permutations")
    st.code("--chaos {0,25,50,75,100} --stylize {0,250,500,750,1000} --weird {0,750,1500,2250,3000} --v {4, 5, 5.1, 5.2}")
    
    if what == "MAELA":
        st.info("* [Promptsuggestions made by MAELA Berlotti](https://maelaberlotti.notion.site/maelaberlotti/Midjourney-Promt-Randomizer-dc3257fee786403bbc864b063bdce2a4)")
    
    
    st.subheader("Past prompts")
    
    history_list = st.session_state.history
    if len(history_list) == 0:
        st.write("No past prompts")
    else:
        for h in history_list:
            st.info(h)
        if st.sidebar.button('Clear history'):
            del st.session_state["history"]
    history_list.append(prompt2)
    st.session_state.history = history_list
    

def get_df():
    """Get the DF with the possibilities.
       Loading from Google Sheets gives the 2 first rows as column header.

    Returns:
        df: the dataframe with the choices
    """    
    sheet_url = "https://docs.google.com/spreadsheets/d/11QQdAEaolonFRhwYryRnbO0AEyzKLBc8mlK-wZ76VHg/edit#gid=973302151"
    csv_export_url = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
    local_url = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\MIDJOURNEY_prompt_generator_keuzes.csv"
    try:
        # df = pd.read_csv(csv_export_url, delimiter=",", header=0)
        df = pd.read_csv(csv_export_url, delimiter=",", header=0)
    except:
        df = pd.read_csv(local_url, delimiter=",", header=0)
    return df
    
def show_info(df):
    """Show some info, the dataframe and links/sources

    Args:
        df (df): the dataframe
    """
    st.title("About")
    st.write("This generator has been made by Rene Smit. It combines various keywords in various categories")
    st.subheader("The categories/keywords")
    df =df.fillna(" ")
    st.write(df)
    st.subheader("Sources and links")
    st.write("* [Prompts from MAELA Berlotti](https://maelaberlotti.notion.site/maelaberlotti/Midjourney-Promt-Randomizer-dc3257fee786403bbc864b063bdce2a4)")
    st.subheader("Manuals / reference sheets")
    st.write("* [A very unofficial Midjourney Manual by Shambibble](https://docs.google.com/document/d/1ivAYy_JXJsGE-9Rh97iMyXkWlmF_MxO2NFshrIvuns4/edit#heading=h.m6597yajayd7)") 
    st.write("* [Midjourney Keywords & Styles y marigoldguide](https://marigoldguide.notion.site/marigoldguide/52ac9968a8da4003a825039022561a30?v=43706e26438d486bb5b8baaa2dc22ffd)") 
    st.write("* [V4 midjourney reference sheets](https://docs.google.com/spreadsheets/d/1MsX0NYYqhv4ZhZ7-50cXH1gvYE2FKLixLBvAkI40ha0/edit#gid=0)") 
    st.write("* [MidJourney-Styles-and-Keywords-Reference by willwulfken](https://github.com/willwulfken/MidJourney-Styles-and-Keywords-Reference)") 
    st.write("* [Resources/artist styles on midjourney v4/](https://weirdwonderfulai.art/resources/artist-styles-on-midjourney-v4/)")
    st.write("* [Artists names know by MJ](https://docs.google.com/spreadsheets/d/1cm6239gw1XvvDMRtazV6txa9pnejpKkM5z24wRhhFz0/edit#gid=400749539)") 
    st.write("* [Midjourney guide](https://www.midlibrary.io/midguide)")
    st.write("* [2000+ styles](https://www.midlibrary.io/styles)")
    st.subheader("Articles / prompts")
    st.write("* [About archetypes and unluckwords](https://bootcamp.uxdesign.cc/how-to-invoke-midjourney-archetypes-and-make-a-cat-fly-be36b6effe2d)")
    
    st.write("* [50 midjourney prompts for to perfect your art](https://bootcamp.uxdesign.cc/50-midjourney-prompts-for-to-perfect-your-art-363996b702b6)")
    st.write("* [50-midjourney-prompts-to-create-photorealistic-images-advanced](https://bootcamp.uxdesign.cc/50-midjourney-prompts-to-create-photorealistic-images-advanced-2e233463bccf)")
    st.write("* [50 best midjourney prompts to create seamless pattern that sells](https://bootcamp.uxdesign.cc/50-best-midjourney-prompts-to-create-seamless-pattern-that-sells-bdfcd8a067eb)")
    st.write("* [An advanced guide to writing prompts for midjourney text to image=](https://medium.com/mlearning-ai/an-advanced-guide-to-writing-prompts-for-midjourney-text-to-image-aa12a1e33b6)")
    st.write("* [10 amazing techniques for midjourney you probably didnt know yet](https://bootcamp.uxdesign.cc/10-amazing-techniques-for-midjourney-you-probably-didnt-know-yet-78f2ab7c00c0)")
    st.write("* [Troubleshooting Midjourney Text Prompts by @whatnostop#6700 (clarinet)](https://docs.google.com/document/d/e/2PACX-1vRHOxyEb-ERGi-BdZM8Z_piEP54m4HwO0z8scjmEurEp2UZVA6rFxvyKd15elYVHUWfP1oSA4CQFwxr/pub?utm_source=docs.google.com&utm_medium=tutorial&utm_campaign=midjourney)") 
    st.write("* [Articles by Stacey Schneider @medium.com](https://medium.com/@sparkystacey)") 
    st.write("* [Prompts by aituts.com](https://prompts.aituts.com/)") 
    st.write("* [Free e-book with 250+ prompts](https://aituts.ck.page/prompts-book)") 

    st.subheader("Wordlists")
    st.write("* [Library of Style Words](https://discord.com/channels/662267976984297473/1017917091606712430/threads/1125455952448061511)") 
    st.write("* [400+ Words](https://generativeai.pub/400-midjourney-prompts-for-illustrations-7a721e64129c)") 
    st.write("* [Scene settings](https://onestopforwriters.com/scene_settings)") 
    st.write("* [Architecture](https://docs.google.com/spreadsheets/d/1029yD1REXEq8V47XgfRm8GQby8JXnwGNlWOL17Lz6J4/edit#gid=0)") 
    st.write("* [201 archetypes](https://industrialscripts.com/archetypes-of-characters/)") 
    st.subheader("Prompt generators")
    st.write("* [Visual promptbuilder](https://tools.saxifrage.xyz/prompt)")
    st.write("* [prompts.aituts.com](https://prompts.aituts.com/)")
    st.write("* [Propt Mania](https://promptomania.com/midjourney-prompt-builder/)")
    st.write("* [Succinctly AI - autocompletion based on dataset](https://huggingface.co/succinctly/text2image-prompt-generator)")
    st.write("* [Creative Fabrica](https://www.creativefabrica.com/spark/tools/prompt-builder/)")
    st.subheader("Interesting people.accounts")
    st.write("* [@techhalla](https://twitter.com/techhalla)") 
    st.write()
    st.write("* [**My profile**](https://www.midjourney.com/app/users/2fae5989-ecac-4f06-afaf-7ee2cb306c58/)")
    st.subheader("Credits")
    st.write("INTERIOR archictecture prompts inspired by https://twitter.com/nickfloats/status/1635116676978208769 as seen in https://www.youtube.com/watch?v=p6vc5N8DH7A")

def main():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    st.title("Midjourney Prompt generator")
    df = get_df()

    non_fixed_columns_start = 9  # number of columns not in the general generator
    non_fixed_columns_end = 35
    architecture_columns_start  =36
    architecture_columns_end = 48
    maela_columns_start = 48
    maela_columns_end = 55
    

    what = st.sidebar.selectbox("What to choose / INFO",["FAMOUS PEOPLE", "ANIMALS", "OBJECTS","INTERIOR ARCHITECTURE", "MAELA", "INFO"])
    if what != "INTERIOR ARCHITECTURE" and what != "MAELA" and what != "INFO":
        who = st.sidebar.selectbox("What kind of artist", ["FAMOUS PAINTERS", "MASTERPHOTOGRAPHERS","ARTISTS" ])
    else:
        who = None
    if what != "INFO":
        chaos = st.sidebar.slider("chaos", 0,100, 0) # High --chaos values will produce more unusual and unexpected results and compositions. Lower --chaos values have more reliable, repeatable results.
        stylize = st.sidebar.slider("Stylyze", 0, 1000, 100) #Low stylization values produce images that closely match the prompt but are less artistic. High stylization values create images that are very artistic but less connected to the prompt.
        weird = st.sidebar.slider("Weird", 0, 3000, 0)
        if what == "INTERIOR ARCHITECTURE":
            ar_def = 6
        else:
            ar_def = 0
        ar = st.sidebar.selectbox("Aspect ratio (w:h)", ["1:1","9:16","4:5","3:4","2:3","10:16","16:9","5:4","4:3","3:2"],ar_def)
        if what == "INTERIOR ARCHITECTURE":
            list_columns = list(df.columns)[architecture_columns_start:architecture_columns_end]
        elif what == "MAELA":
            list_columns = list(df.columns)[maela_columns_start:maela_columns_end]
        else:
            list_columns = list(df.columns)[non_fixed_columns_start:non_fixed_columns_end]
        included_columns = st.sidebar.multiselect("Columns to include", list_columns,list_columns)
        number = 0
        if len(included_columns) >= 5:
            number = st.sidebar.slider("Number of keywords", 0, len(included_columns), 5)
        elif len(included_columns) > 0:
            number = st.sidebar.slider("Number of keywords", 0, len(included_columns), len(included_columns))
        seperator = st.sidebar.selectbox("Seperator", ["|",",","<space>"],0)
        
    
    # --chaos controls how diverse the initial grid images are from each other.
    # --stylize controls how strongly Midjourney's default aesthetic is applied.
    # --weird controls how unusual an image is compared to previous Midjourney images.

    if what =="INFO":
        show_info(df)
    else:
        if st.button('Generate prompt'):
            generate_prompt(df, included_columns, what, who, number,  chaos, stylize, weird,ar, seperator)
def check_double_values():
    """Simpel function to detect double values in the dataframe
    """    
    df = get_df()
    print (df)
    df = df.fillna("X")
    columnlist = list(df.columns)
    
    for i,column in enumerate(columnlist):
        print (f"{column} [{i+1}/{len(columnlist)}]")
        masterlist = df[column].tolist()
        for value_to_check in masterlist:
            if value_to_check != 'X':
                for other_column in df.columns:
                    if other_column != column:
                        if value_to_check in df[other_column].tolist():
                            print(f"Value '{value_to_check}' in column '{column}' is also in column '{other_column}'.")
                           

if __name__ == "__main__":
    main()
    #check_double_values()
