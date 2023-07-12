import pandas as pd
import random
import platform
import streamlit as st

from scipy.stats import weibull_min

def weibull(max):
    """Give a value, selected with the weibull distribution

    Args:
        max (int): maximum value

    Returns:
        int: random value
    """    
    # Define the shape and scale parameters
    shape = 2
    scale = max/2

    # Generate a random number from the Weibull distribution
    random_number = int(weibull_min.rvs(shape, scale=scale))
    return random_number

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
        print(f"Column: {column} - Random Value: {random_value}")
    else:
        random_value = None
    return random_value

def generate_prompt(what):

    df = get_df()

    selected_columns =  random.sample(list(df.columns)[7:], 5) # Exclude the first column and select a random sample of 8 column names
    print (list(df.columns)[7:])
    if what == "FAMOUS PEOPLE":
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
        prompt += f"--chaos {weibull(100)} "
        prompt += f"--stylize {weibull(1000)} " # Low stylization values produce images that closely match the prompt but are less artistic. High stylization values create images that are very artistic but less connected to the prompt.
        prompt += f"--weird {weibull(3000)} "
    elif distribution == "even":
        prompt += f"--chaos {random.randint(1,100)} "
        prompt += f"--stylize {random.randint(1,1000)} " # Low stylization values produce images that closely match the prompt but are less artistic. High stylization values create images that are very artistic but less connected to the prompt.
        prompt += f"--weird {random.randint(1,3000)} "
    else:
        pass

    prompt +=    "--style raw"
    prompt2 +=    "--style raw"
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
    
def main():
    st.title("Midjourney Prompt generator")
    what = st.sidebar.selectbox("What to choose",["FAMOUS PEOPLE", "OBJECTS"])
    if st.button('Generate prompt'):
        generate_prompt(what)
if __name__ == "__main__":
    main()

# print(f"{random_artist} {selected_values_string} --style raw")

# https://bootcamp.uxdesign.cc/50-midjourney-prompts-for-to-perfect-your-art-363996b702b6
# https://bootcamp.uxdesign.cc/50-midjourney-prompts-to-create-photorealistic-images-advanced-2e233463bccf


# 50+%20Best%20Midjourney%20Prompts%20to%20Create%20Seamless%20Pattern%20that%20Sells%20_%20Bootcamp.pdf
# https://medium.com/mlearning-ai/an-advanced-guide-to-writing-prompts-for-midjourney-text-to-image-aa12a1e33b6
# https://weirdwonderfulai.art/resources/artist-styles-on-midjourney-v4/

# https://onestopforwriters.com/scene_settings
#https://discord.com/channels/662267976984297473/1017917091606712430/threads/1125455952448061511

# https://docs.google.com/document/d/e/2PACX-1vRHOxyEb-ERGi-BdZM8Z_piEP54m4HwO0z8scjmEurEp2UZVA6rFxvyKd15elYVHUWfP1oSA4CQFwxr/pub?utm_source=docs.google.com&utm_medium=tutorial&utm_campaign=midjourney
#https://docs.google.com/document/d/1ivAYy_JXJsGE-9Rh97iMyXkWlmF_MxO2NFshrIvuns4/edit#heading=h.m6597yajayd7

# Architecture: https://docs.google.com/spreadsheets/d/1029yD1REXEq8V47XgfRm8GQby8JXnwGNlWOL17Lz6J4/edit#gid=0

# https://marigoldguide.notion.site/marigoldguide/52ac9968a8da4003a825039022561a30?v=43706e26438d486bb5b8baaa2dc22ffd
# https://docs.google.com/spreadsheets/d/1MsX0NYYqhv4ZhZ7-50cXH1gvYE2FKLixLBvAkI40ha0/edit#gid=0
# https://github.com/willwulfken/MidJourney-Styles-and-Keywords-Reference
# 