import pandas as pd
import numpy as np
import scipy, matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings
import streamlit as st
import platform
import math
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

# https://stackoverflow.com/questions/55212002/how-do-i-use-scipy-optimize-curve-fit-with-panda-df

def main():
    if platform.processor() != "":   
        file = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\eigen_bijdrage2022.csv"   
    else: 
        file = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/eigen_bijdrage2022.csv"

    data  = pd.read_csv(
            file,
            delimiter=",",
            
            low_memory=False,
        )


    df = pd.read_csv(file)

    st.write (df)
    xdata = df['x'] #.as_matrix()
    ydata = df['y'] #.as_matrix()


    def various_functions():
        # included for future reference

        pass
        # Functions to calculate values a,b  and c ##########################
        def func(x, a, b, c):
            return (a*np.sin(b*x))+(c * np.exp(x))

    

        def derivate(x, a, b, c):
            ''' First derivate of the sigmoidal function. Might contain an error'''
            return  (np.exp(b * (-1 * np.exp(-c * x)) - c * x) * a * b * c ) + BASEVALUE
            #return a * b * c * np.exp(-b*np.exp(-c*x))*np.exp(-c*x)

        def exponential(x, a, r):
            '''Exponential growth  function.'''
            return (a * ((1+r)**x))


        def derivate_of_derivate(x,a,b,c):
            return a*b*c*(b*c*exp(-c*x) - c)*exp(-b*exp(-c*x) - c*x)

        def gaussian(x, a, b, c):
            ''' Standard Guassian function. Doesnt give results, Not in use'''
            return a * np.exp(-np.power(x - b, 2) / (2 * np.power(c, 2)))

        def gaussian_2(x, a, b, c):
            ''' Another gaussian fuctnion. in use
                a = height, b = cen (?), c= width '''
            return a * np.exp(-((x - b) ** 2) / c)

        def growth(x, a, b):
            """ Growth model. a is the value at t=0. b is the so-called R number.
                Doesnt work. FIX IT """
            return np.power(a * 0.5, (x / (4 * (math.log(0.5) / math.log(b)))))

        # https://replit.com/@jsalsman/COVID19USlognormals
        def lognormal_c(x, s, mu, h): # x, sigma, mean, height
            return h * 0.5 * erfc(- (log(x) - mu) / (s * sqrt(2)))
        # https://en.wikipedia.org/wiki/Log-normal_distribution#Cumulative_distribution_function


        def normal_c(x, s, mu, h): # x, sigma, mean, height
            return h * 0.5 * (1 + erf((x - mu) / (s * sqrt(2))))

    def sigmoidal(x, a, b, c):
            ''' Standard sigmoidal function
                a = height, b= halfway point, c = growth rate
                https://en.wikipedia.org/wiki/sigmoidal_function '''
            return a * np.exp(-b * np.exp(-c * x))

    def func(x, a, b):
        return (a + b* x)

    def func_(x, a, r):
        '''Exponential growth  function.'''
        return (a * ((1+r)**x))

    # #####################################################################


    # function for genetic algorithm to minimize (sum of squared error)
    def sumOfSquaredError(parameterTuple):
        warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
        val = func(xdata, *parameterTuple)
        return np.sum((ydata - val) ** 2.0)
    


    def generate_Initial_Parameters():

        parameterBounds = []
        parameterBounds.append([0.0, 10.0]) # search bounds for a
        parameterBounds.append([0.0, 10.0]) # search bounds for b
        #parameterBounds.append([0.0, 10.0]) # search bounds for c

        # "seed" the numpy random number generator for repeatable results
        result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
        return result.x

    # by default, differential_evolution completes by calling curve_fit() using parameter bounds
    geneticParameters = generate_Initial_Parameters()

    # now call curve_fit without passing bounds from the genetic algorithm,
    # just in case the best fit parameters are aoutside those bounds
    #popt, pcov = curve_fit(func, xdata, ydata, geneticParameters)


    popt, pcov = curve_fit(func, xdata, ydata)

    # poptarray
    # Optimal values for the parameters so that the sum of the squared residuals 
    # of f(xdata, *popt) - ydata is minimized.

    # pcov2-D array
    # The estimated covariance of popt. The diagonals provide the variance of the parameter estimate. 
    # To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov)).


    # modelPredictions = func(xdata, *popt) 
    modelPredictions = func(xdata, *popt) 

    absError = modelPredictions - ydata

    SE = np.square(absError) # squared errors
    MSE = np.mean(SE) # mean squared errors
    RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (np.var(absError) / np.var(ydata))

    st.write()
    #st.write(f'Formula: y = ({popt[0]} * ((1+{popt[1]})**x))') #  {popt[0]} * x + {popt[1]}')
    st.write(f'Formula: y = {popt[0]} * x + {popt[1]}') #  {popt[0]} * x + {popt[1]}')
    st.write(f'Root Mean Squared Error, RMSE: {RMSE}' )
    st.write(f'R-squared: {Rsquared}')

    st.write()


    fig =  plt.figure()
    plt.plot(xdata, func(xdata, *popt), 'r--',
            label='fit: a=%5.3f, b=%5.3f' % tuple(popt)) # c=%5.3f
    plt.plot(xdata, ydata, 'g-',
            label='values' )

            
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    st.pyplot(fig)

if __name__ == "__main__":
    main()

