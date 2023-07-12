import pandas as pd
import random
import platform
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

def generate_prompt(df,what, number):

    

    selected_columns =  random.sample(list(df.columns)[8:], number) # Exclude the first column and select a random sample of 8 column names
    print (list(df.columns)[7:])
    if what == "FAMOUS PEOPLE" or what=="ANIMALS":
        prompt = f'{take_random_value(df, what)} as {take_random_value(df, "ARCHETYPES")} in a {take_random_value(df, "SCENES")} in the style of {take_random_value(df, "MASTERPHOTOGRAPHERS")}, '
    else:
        prompt = f'{take_random_value(df, what)} in a {take_random_value(df, "SCENES")} in the style of {take_random_value(df, "MASTERPHOTOGRAPHERS")}, '
    
    
    
    prompt2 = prompt
    for column in df.columns:
        if column in selected_columns:
            random_value = take_random_value(df, column)
        
            prompt += f"{random_value}::{random.randint(1, 100)} "

            prompt2 += f"{random_value}, "
   
    distribution = "uneven"
    if distribution == "weibull":
        # prompt += f"--chaos {weibull(100)} "
        # prompt += f"--stylize {weibull(1000)} " # Low stylization values produce images that closely match the prompt but are less artistic. High stylization values create images that are very artistic but less connected to the prompt.
        # prompt += f"--weird {weibull(3000)} "
        pass
    elif distribution == "even":
        prompt += f"--chaos {random.randint(1,100)} "
        prompt += f"--stylize {random.randint(1,1000)} " # Low stylization values produce images that closely match the prompt but are less artistic. High stylization values create images that are very artistic but less connected to the prompt.
        prompt += f"--weird {random.randint(1,3000)} "
    else:
        pass

    prompt +=    "--style raw --chaos 0 --stylize 0 --weird 0"
    prompt2 +=    "--style raw --chaos 0 --stylize 0 --weird 0"
    print (prompt)
    print (prompt2)

    st.info (prompt)
    st.info (prompt2)

def get_df():
    """Get the DF with the possibilities.
       Loading from Google Sheets gives the 2 first rows as column header.

    Returns:
        df: the dataframe with the choices
    """    
    sheet_id = "11QQdAEaolonFRhwYryRnbO0AEyzKLBc8mlK-wZ76VHg"
    sheet_name = "keuzes"

    #url= f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    url = r"C:\Users\rcxsm\Downloads\MIDJOURNEY prompt generator - keuzes.csv"
    if platform.processor() != "":
        # local    
        url = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\MIDJOURNEY_prompt_generator_keuzes.csv"
    else:
        url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/MIDJOURNEY_prompt_generator_keuzes.csv"
    df = pd.read_csv(url, delimiter=",", header=0)
    return df
    
def about(df):
    st.title("About")
    st.write("This generator has been made by Rene Smit. It combines various keywords in various categories")
    st.subheader("The categories/keywords")
    df =df.fillna(" ")
    st.write(df)
    st.subheader("Sources and links")
    '''
    * https://bootcamp.uxdesign.cc/50-midjourney-prompts-for-to-perfect-your-art-363996b702b6
    * https://bootcamp.uxdesign.cc/50-midjourney-prompts-to-create-photorealistic-images-advanced-2e233463bccf
    * https://bootcamp.uxdesign.cc/50-best-midjourney-prompts-to-create-seamless-pattern-that-sells-bdfcd8a067eb
    * https://medium.com/mlearning-ai/an-advanced-guide-to-writing-prompts-for-midjourney-text-to-image-aa12a1e33b6
    * https://weirdwonderfulai.art/resources/artist-styles-on-midjourney-v4/

  

    * Library of Style Words : https://discord.com/channels/662267976984297473/1017917091606712430/threads/1125455952448061511

    * Troubleshooting Midjourney Text Prompts by @whatnostop#6700 (clarinet): https://docs.google.com/document/d/e/2PACX-1vRHOxyEb-ERGi-BdZM8Z_piEP54m4HwO0z8scjmEurEp2UZVA6rFxvyKd15elYVHUWfP1oSA4CQFwxr/pub?utm_source=docs.google.com&utm_medium=tutorial&utm_campaign=midjourney
    
    * A very unofficial Midjourney Manual by Shambibble : https://docs.google.com/document/d/1ivAYy_JXJsGE-9Rh97iMyXkWlmF_MxO2NFshrIvuns4/edit#heading=h.m6597yajayd7
    
   
    * Midjourney Keywords & Styles : https://marigoldguide.notion.site/marigoldguide/52ac9968a8da4003a825039022561a30?v=43706e26438d486bb5b8baaa2dc22ffd

    * V4 midjourney reference sheets :https://docs.google.com/spreadsheets/d/1MsX0NYYqhv4ZhZ7-50cXH1gvYE2FKLixLBvAkI40ha0/edit#gid=0

    * MidJourney-Styles-and-Keywords-Reference :   https://github.com/willwulfken/MidJourney-Styles-and-Keywords-Reference


    * Scene settings: https://onestopforwriters.com/scene_settings
    * Architecture:  https://docs.google.com/spreadsheets/d/1029yD1REXEq8V47XgfRm8GQby8JXnwGNlWOL17Lz6J4/edit#gid=0

    * My profile : https://www.midjourney.com/app/users/2fae5989-ecac-4f06-afaf-7ee2cb306c58/

    '''
    
def main():
    st.title("Midjourney Prompt generator")
    df = get_df()
    what = st.sidebar.selectbox("What to choose",["FAMOUS PEOPLE", "ANIMALS", "OBJECTS", "ABOUT"])
    number = st.sidebar.slider("Number of keywords", 0, len(df.columns)-8, 5)
       
    if what =="ABOUT":
        about(df)
    else:
        if st.button('Generate prompt'):
            generate_prompt(df, what, number)
if __name__ == "__main__":
    main()
