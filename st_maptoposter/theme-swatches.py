#!/usr/bin/env python
# coding: utf-8
# https://github.com/zoof/maptoposter/blob/main/theme-swatches.py
import json
import seaborn as sns
from os import listdir
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image

def invert_hex(h):
    """
    Take a hexadecimal color and return it's inverse.

    Parameters:
    h (str): hexadecimal color

    Returns:
    h_inv (str): inverted hexadecimal color

    Examples:
    >>> invert_hex('#000000')
    #FFFFFF
    >>> invert_hex('##E0E0E0')
    #1F1F1F
    """
    h_inv = "#{:02X}{:02X}{:02X}".format(*(255-int(h[i:i+2], 16) for i in (1,3,5)))
    return h_inv

def draw_swatch(colors,ax):
    #for color in colors:
    #    sns.palplot(color)
    return sns.palplot(colors)

# Initialize seaborn
sns.set()

sw_width = 1084
sw_height = 100

theme_list = listdir('themes')

width = sw_width
height = sw_height*len(theme_list)
swatches = Image.new('RGB', (width, height))
y_offset = 0

# Loop through all themes, drawing a swatch for each theme
for theme_file in theme_list:
    with open('themes/' + theme_file) as th:
        # load theme
        theme = json.load(th)
    # extract theme name and delete
    theme_name = theme.pop('name')
    theme_description = theme.pop('description')

	# Draw swatch
    sns.palplot(theme.values())

	# Add title to swatch
    plt.title(theme_name)
    # Add key and hexadecimal color
    for i, key in enumerate(theme):
        plt.text(
            i, 0.2,
            key+'\n'+theme[key],
            ha='center',
            va='bottom',
            color=invert_hex(theme[key]),
            fontsize=7
        )
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf)
    buf.seek(0)

    sw = Image.open(buf)
    swatches.paste(sw, (0,y_offset))
    y_offset += sw_height
    plt.close()
swatches.save('theme_swatches.png')
