import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st

# https://towardsdatascience.com/how-to-create-well-styled-streamlit-dataframes-part-1-using-the-pandas-styler-e239c0fbe145

# Sample DataFrame
# data = {
#     'Prices': np.random.randint(-500000, 500000, size=10),  # Random differences between -50 and 50
#     'Difference': np.random.uniform(-50.00, 50.00, size=10),  # Random differences between -50 and 50
#     'Percentage Change': np.random.uniform(-1, 1, size=10) * 100  # Random percentage changes between -100% and 100%
# }

# df = pd.DataFrame(data)

mock_data = {
"Country": ["US", "IN", "BR", "ES", "AR", "IT"],
"Period_1": [50_000, 30_000, 17_000, 14_000, 22_000, 16_000], 
"Period_2": [52_000, 37_000, 16_000, 12_000, 21_000, 19_000], 
} 

df = pd.DataFrame(mock_data) 
df['Difference'] = df['Period_2'] - df['Period_1'] 
df['Percentage Change'] = np.round(((df['Period_2'] - df['Period_1']) / df['Period_1']), 2) 
df['Percentage Change rank'] = df['Percentage Change'].rank(method='dense', ascending=False).astype(int)

def _format_with_dollar_sign(val, prec=0): 
  return f'${val:,.{prec}f}' 

def _format_with_thousands_commas(val): 
  return f'{val:,.0f}' 

def _format_as_percentage(val, prec=0): 
  return f'{val:.{prec}%}' 

def _add_medal_emoji(val): 
  if val == 1: 
    return f"{val} 🥇" 
  elif val == 2: 
    return f"{val} 🥈" 
  elif val == 3: 
    return f"{val} 🥉" 
  else: return val 

# Test the function
def _format_positive_negative_background_colour(val, min_value, max_value): 
    if val > 0: 
        # Normalize positive values to a scale of 0 to 1 
        normalized_val = (val - 0) / (max_value - 0) 
        # Create a gradient of green colors 
        color = plt.cm.Greens(normalized_val * 0.7) 
        color_hex = mcolors.to_hex(color) 
    elif val < 0: 
        # Normalize negative values to a scale of 0 to -1 
        normalized_val = (val - min_value) / (0 - min_value) 
        # Create a gradient of red colors 
        color = plt.cm.Reds_r(normalized_val * 0.7) 
        color_hex = mcolors.to_hex(color) 
    else: 
        color_hex = 'white'  # For zero values, set the background color to white

    # Determine text color based on the darkness of the background color 
    r, g, b = mcolors.hex2color(color_hex) 
    if (r * 299 + g * 587 + b * 114) / 1000 > 0.5:  # Use the formula for perceived brightness 
        text_color = 'black' 
    else: 
        text_color = 'white' 

    return f'background-color: {color_hex}; color: {text_color}' 

def main():
    st.info("https://towardsdatascience.com/how-to-create-well-styled-streamlit-dataframes-part-1-using-the-pandas-styler-e239c0fbe145" )
    # Calculate min and max values for each column
    min_value_abs_diff = df['Difference'].min() 
    max_value_abs_diff = df['Difference'].max() 

    min_value_perct_diff = df['Percentage Change'].min() 
    max_value_perct_diff = df['Percentage Change'].max() 

    # Apply the styler with colour gradients
    styler_with_colour_gradients = (
        df.copy().style 
        # .applymap(lambda x: _format_positive_negative_background_colour(x, min_value_abs_diff, max_value_abs_diff), subset=['Difference']) 
        # .applymap(lambda x: _format_positive_negative_background_colour(x, min_value_perct_diff, max_value_perct_diff), subset=['Percentage Change']) 
        .map(lambda x: _format_positive_negative_background_colour(x, min_value_abs_diff, max_value_abs_diff), subset=['Difference']) 
        .map(lambda x: _format_positive_negative_background_colour(x, min_value_perct_diff, max_value_perct_diff), subset=['Percentage Change']) 

        .format(_format_with_thousands_commas, subset=["Period_1"]) 
        .format(lambda x: _format_as_percentage(x, 2), subset=["Percentage Change"]) 
        .format(_format_with_dollar_sign, subset=['Difference', "Period_1"]) 
        .format(_add_medal_emoji, subset=['Percentage Change rank']) 
    )

    # Display the styled DataFrame
    st.write(styler_with_colour_gradients)

print ("goooo")
main()