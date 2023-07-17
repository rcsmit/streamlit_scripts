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
    column_values = df[column].dropna().values
    
    if len(column_values) > 0:
        random_value = random.choice(column_values)
        st.write (f"{column} : {random_value}")
    else:
        random_value = None
    return random_value

def generate_prompt(df,included_columns, what,who, number, fixed_columns,chaos, stylize, weird,ar):
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
        ar (str): aspect ratio
    """    
    place1 = st.empty()
    place2 = st.empty()
    print ("Included_columns")
    print (included_columns)
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
    else:
        prompt = f'{take_random_value(df, "CONCEPT")} {take_random_value(df, what)} in a {take_random_value(df, "SCENES")} by {take_random_value(df, who)}'
      
    prompt2 = prompt
    if selected_columns != None:
        for column in df.columns:
            if column in selected_columns:
                random_value = take_random_value(df, column)
                prompt += f"{random_value}::{random.randint(1, 100)} "
                prompt2 += f"| {random_value} "
   
    # distribution = "uneven"
    # if distribution == "weibull":
    #     # prompt += f"--chaos {weibull(100)} "
    #     # prompt += f"--stylize {weibull(1000)} " # Low stylization values produce images that closely match the prompt but are less artistic. High stylization values create images that are very artistic but less connected to the prompt.
    #     # prompt += f"--weird {weibull(3000)} "
    #     pass
    # elif distribution == "even":
    #     prompt += f"--chaos {random.randint(1,100)} "
    #     prompt += f"--stylize {random.randint(1,1000)} " # Low stylization values produce images that closely match the prompt but are less artistic. High stylization values create images that are very artistic but less connected to the prompt.
    #     prompt += f"--weird {random.randint(1,3000)} "
    # else:
    prompt += f"--chaos {chaos} --stylize {stylize}  --weird {weird} --ar {ar} --style raw"
    prompt2 += f"--chaos {chaos} --stylize {stylize}  --weird {weird} --ar {ar} --style raw"

    place1.info (prompt)
    place2.info (prompt2)

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
    st.subheader("Wordlists")
    st.write("* [Library of Style Words](https://discord.com/channels/662267976984297473/1017917091606712430/threads/1125455952448061511)") 
    
    st.write("* [Scene settings](https://onestopforwriters.com/scene_settings)") 
    st.write("* [Architecture](https://docs.google.com/spreadsheets/d/1029yD1REXEq8V47XgfRm8GQby8JXnwGNlWOL17Lz6J4/edit#gid=0)") 
    st.write("* [201 archetypes](https://industrialscripts.com/archetypes-of-characters/)") 
    st.write("* [@techhalla](https://twitter.com/techhalla)") 
    st.write()
    st.write("* [**My profile**](https://www.midjourney.com/app/users/2fae5989-ecac-4f06-afaf-7ee2cb306c58/)")

def main():
    st.title("Midjourney Prompt generator")
    df = get_df()
    fixed_columns = 9  # number of columns not in the general generator
    what = st.sidebar.selectbox("What to choose / INFO",["FAMOUS PEOPLE", "ANIMALS", "OBJECTS", "INFO"])
    who = st.sidebar.selectbox("What kind of artist", ["FAMOUS PAINTERS", "MASTERPHOTOGRAPHERS","ARTISTS" ])
    chaos = st.sidebar.slider("chaos", 0,100, 0) # High --chaos values will produce more unusual and unexpected results and compositions. Lower --chaos values have more reliable, repeatable results.
    stylize = st.sidebar.slider("Stylyze", 0, 1000, 100) #Low stylization values produce images that closely match the prompt but are less artistic. High stylization values create images that are very artistic but less connected to the prompt.
    weird = st.sidebar.slider("Weird", 0, 3000, 0)
    ar = st.sidebar.selectbox("Aspect ratio (w:h)", ["1:1","9:16","4:5","3:4","2:3","10:16","16:9","5:4","4:3","3:2"],0)
    included_columns = st.sidebar.multiselect("Columns to include", list(df.columns)[fixed_columns:], list(df.columns)[fixed_columns:])
    number = 0
    if len(included_columns) >= 5:
        number = st.sidebar.slider("Number of keywords", 0, len(included_columns), 5)
    elif len(included_columns) > 0:
        number = st.sidebar.slider("Number of keywords", 0, len(included_columns), len(included_columns))

    # --chaos controls how diverse the initial grid images are from each other.
    # --stylize controls how strongly Midjourney's default aesthetic is applied.
    # --weird controls how unusual an image is compared to previous Midjourney images.

    if what =="INFO":
        show_info(df)
    else:
        if st.button('Generate prompt'):
            generate_prompt(df, included_columns, what, who, number, fixed_columns, chaos, stylize, weird,ar)
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
