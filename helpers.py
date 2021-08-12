def cell_background_helper(val,method, max, color):
    """Creates the CSS code for a cell with a certain value to create a heatmap effect
       st.write (df.style.format(None, na_rep="-").applymap(lambda x:  cell_background_number_of_cases(x,[method], [top_waarde])).set_precision(2))
    Args:
        val ([int]): the value of the cell
        metohod (string): "exponential" / "lineair" / "percentages"
        max : max value (None bij percentage)
        color (string): color 'r, g, b' (None bij percentages)

    Returns:
        [string]: the css code for the cell
    """
    if color == None : color = '193, 57, 43'
    opacity = 0
    try:
        v = abs(val)
        if method == "percentages":
            # scale from -100 to 100
            opacity = 1 if v >100 else v/100
            # color = 'green' if val >0 else 'red'
            if val > 0 :
                color = '193, 57, 43'
            elif val < 0:
                color = '1, 152, 117'
            else:
                color = '255,255,173'
        elif method == "lineair":
            opacity = v / max
        else:
            if method == "exponential":
                value_table = [ [0,0],
                                [0.00390625,0.0625],
                                [0.0078125, 0.125],
                                [0.015625,0.25],
                                [0.03125,0.375],
                                [0.0625,0.50],
                                [0.125,0.625],
                                [0.25,0.75],
                                [0.50,0.875],
                                [0.75,0.9375],
                                [1,1]]
            elif method == "lineair2":
                value_table = [ [0,0],
                                [0.1,0.0625],
                                [0.2, 0.125],
                                [0.3,0.25],
                                [0.4,0.375],
                                [0.5,0.50],
                                [0.6,0.625],
                                [0.7,0.75],
                                [0.8,0.875],
                                [0.9,0.9375],
                                [1,1]]


            for vt in value_table:
                if v >= round(vt[0]*max) :
                    opacity = vt[1]
    except:
        # give cells with eg. text or dates a white background
        color = '255,255,0'
        opacity = 1


    return f'background: rgba({color}, {opacity})'

def left(s, amount):
    return s[:amount]

def right(s, amount):
    return s[-amount:]


def mid(s, offset, amount):
    return s[offset-1:offset+amount-1]