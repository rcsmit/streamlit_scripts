def effort_based_distance_calculator___(gradient):
    """
    Calculate the running energy factor based on the gradient.
    The factor is relative to flat ground (0% = 1).
    
    :param gradient: Gradient as a decimal (e.g., 0.10 for 10%, -0.05 for -5%)
    :return: Energy factor
    """
    gradient = gradient / 100

    if gradient >= -0.10:
        # For gradients between -10% and +25%, use the linear model
        factor = 1 + 0.96 * gradient
    else:
        # For gradients steeper than -10%, use the adjusted model
        factor = 0.5 + 0.2 * (gradient + 0.10)
    
    return factor
def effort_based_distance_calculator__(gradient):
    """
    Calculates the effort-based distance multiplier using Naismith's Rule, including both uphill and downhill adjustments.  
    Parameters:
        gradient (float): The gradient percentage (elevation change / distance * 100).

    Returns:
        float: Effort multiplier based on gradient.
    """

    # Twee lijsten: één voor drempelwaarden en één voor multipliers
    gradient_thresholds = [-10, -5, 0, 5, 10, 15, 20]  # Drempelwaarden (integer)
    multipliers = [1.2, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]  # Bijbehorende multipliers

    for i, threshold in enumerate(gradient_thresholds):
        if gradient < threshold:
            return multipliers[i]
    return multipliers[-1]  # Als de gradient hoger is dan de laatste drempel


# Functie om de effort_multiplier te bepalen
def effort_based_distance_calculator_dictionary(gradient):
    """
    Calculates the effort-based distance multiplier using Naismith's Rule, including both uphill and downhill adjustments.  
    Parameters:
        gradient (float): The gradient percentage (elevation change / distance * 100).

    Returns:
        float: Effort multiplier based on gradient.
    """

    # Dictionary met gradienten en bijbehorende multipliers
    gradient_multipliers = {
        "uphill": {
            "0-5": 1.2,    # Lichte helling
            "5-10": 1.5,   # Matige helling
            "10-15": 2.0,  # Steile helling
            "15+": 2.5     # Zeer steile helling
        },
        "downhill": {
            "0-5": 0.8,    # Lichte afdaling
            "5-10": 0.7,   # Steile afdaling
            "10+": 0.6     # Zeer steile afdaling
        },
        "flat": 1.0        # Vlak terrein
    }

    if gradient > 0:  # Uphill
        if gradient < 5:
            return gradient_multipliers["uphill"]["0-5"]
        elif gradient < 10:
            return gradient_multipliers["uphill"]["5-10"]
        elif gradient < 15:
            return gradient_multipliers["uphill"]["10-15"]
        else:
            return gradient_multipliers["uphill"]["15+"]
    elif gradient < 0:  # Downhill
        abs_gradient = abs(gradient)
        if abs_gradient < 5:
            return gradient_multipliers["downhill"]["0-5"]
        elif abs_gradient < 10:
            return gradient_multipliers["downhill"]["5-10"]
        else:
            return gradient_multipliers["downhill"]["10+"]
    else:  # Flat terrain
        return gradient_multipliers["flat"]

def effort_based_distance_calculator_old(gradient):
    """
    Calculates the effort-based distance multiplier using Naismith's Rule, including both uphill and downhill adjustments.

    Parameters:
        gradient (float): The gradient percentage (elevation change / distance * 100).

    Returns:
        float: Effort multiplier based on gradient.
    """
    if gradient > 0:  # Uphill
        if gradient < 5:
            effort_multiplier = 1.10  # +10% effort for slight incline
        elif gradient < 12:
            effort_multiplier = 1.20  # +20% effort for moderate incline
        elif gradient < 20:
            effort_multiplier = 1.40  # +40% effort for steep incline
        else:
            effort_multiplier = 2.00  # Very steep uphill, doubling effort
    elif gradient < 0:  # Downhill
        abs_gradient = abs(gradient)


        if abs_gradient < 10:  # Moderate downhill (easier)
            effort_multiplier = 1 - 0.5 * (abs_gradient/100)
        else:  # Steep downhill (harder)
            effort_multiplier= 1 + 0.2 * (abs_gradient/100)
    else:
        effort_multiplier = 1.00  # Flat terrain

    return effort_multiplier
