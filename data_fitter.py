
import warnings
import numpy as np
import pandas as pd
import scipy.stats  as st
import statsmodels.api as sm
from scipy.stats._continuous_distns import _distn_names
import matplotlib
import matplotlib.pyplot as plt

import math
# https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python
# cited in https://github.com/rdzudzar/DistributionAnalyser/blob/main/page_fit.py#L458
# cited in https://towardsdatascience.com/distribution-analyser-b826b88b7b8d
matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

def fit_data(df):
    """ 
    Modified from: https://stackoverflow.com/questions/6620471/fitting\
        -empirical-distribution-to-theoretical-ones-with-scipy-python 
    
    This function is performed with @cache - storing results in the local
    cache; read more: https://docs.streamlit.io/en/stable/caching.html
    """
    
    results = {}
    for ii, distribution in enumerate([d for d in _distn_names if not d in ['kstwo','levy_stable', 'studentized_range']]):

        print("{:>3} / {:<3}: {}".format( ii+1, len(_distn_names), distribution ))

        
        # Check for nan/inf and remove them
        ## Get histogram of the data and histogram parameters
        num_bins = round(math.sqrt(len(df)))
        hist, bin_edges = np.histogram(df, num_bins, density=True)
        central_values = np.diff(bin_edges)*0.5 + bin_edges[:-1]
       
        
        try:
            # Go through distributions
            dist = getattr(st, distribution)
            
            # Get distribution fitted parameters
            params = dist.fit(df)
            
            ## Separate parameters
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]
        
            ## Obtain PDFs
            pdf_values = [dist.pdf(c, loc=loc, scale=scale, *arg) for c in
                        central_values]
    
            # Calculate the RSS: residual sum of squares 
            # Also known as SSE: sum of squared estimate of errors
            # The sum of the squared differences between each observation\
            # and its group's mean; here: diff between distr. & data hist
            sse = np.sum(np.power(hist - pdf_values, 2.0))
            
            
            # Parse fit results 
            # results[dist.name] = [sse, arg, loc, scale]
             # Store results in dictionary
            results[dist.name] = {
                'SSE': sse,
                'Parameters': params,
                'Location': loc,
                'Scale': scale
            }
        except:
            print(f'Skipping, runtime error {distribution}')
    # containes e.g. [0.5, (13,), 8, 1]
    # equivalent to  [sse, (arg,), loc, scale] as argument number varies
    #results = {k: results[k] for k in sorted(results, key=results.get)}

    return results

# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Best holders
    best_distributions_a, best_distributions_b = [], []

    # Estimate distribution parameters from data
    for ii, distribution in enumerate([d for d in _distn_names if not d in ['levy_stable', 'studentized_range']]):

        print("{:>3} / {:<3}: {}".format( ii+1, len(_distn_names), distribution ))

        distribution = getattr(st, distribution)

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                
                # fit dist to data
                params = distribution.fit(data)
                #params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                
                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                
                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

    #             # identify if this distribution is better
        
    
    

                # identify if this distribution is better
                best_distributions_a.append((distribution, params, sse))
    
                best_distributions_b.append((distribution.name, sse))
        
        except Exception:
            pass

    # Create DataFrame and sort by SSE
    best_distributions_df = pd.DataFrame(best_distributions_b, columns=['Distribution', 'SSE'])
    best_distributions_df = best_distributions_df.sort_values(by='SSE').reset_index(drop=True)
    
    return sorted(best_distributions_a, key=lambda x:x[2]), best_distributions_df

def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

# Load data from statsmodels datasets
# data = pd.Series(sm.datasets.elnino.load_pandas().data.set_index('YEAR').values.ravel())
data_ = [56, 18, 15, 30, 34, 30, 7, 15, 44, 23, 50, 20, 50, 18, 19, 32, 40, 36, 26, 30, 8, 15, 
         19, 23, 30, 74, 25, 34, 28, 61, 22, 13, 14, 70, 38, 31, 29, 31, 42, 62, 7, 40, 56, 28, 
         35, 12, 13, 13, 7, 30, 23, 42, 36, 38, 30, 25, 13, 55, 40, 40, 10, 10, 16,
          27, 17, 15, 43, 27, 30, 22, 10, 27, 48, 30, 53, 24, 58, 11, 17, 26, 13, 86, 26, 40, 
          25, 13, 17, 47, 51, 41, 9, 13, 29, 5, 22, 15, 20, 75, 54, 40, 11, 34, 35, 37, 36, 39, 
          41, 40, 33, 28, 57, 45, 16, 12, 33, 27, 14, 26, 16, 18, 19, 70, 15, 11, 46, 35, 20, 22, 
          60, 7, 67, 28, 14, 15, 49, 20, 20, 40, 26, 20, 19, 26, 83, 22, 32, 29, 20, 14, 15, 38]
data = pd.Series(data_)
# Pl
# Find best fit distribution

results = fit_data(data)

df_results = pd.DataFrame.from_dict(results, orient='index')
df_results = df_results.sort_values(by=["SSE"]).reset_index()
# Print DataFrame
print(df_results)
