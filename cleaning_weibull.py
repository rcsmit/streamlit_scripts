import numpy as np
from scipy.stats import weibull_min
from scipy.special import gamma
import scipy.stats as stats
import pandas as pd
import math

# https://chatgpt.com/c/5f499843-4325-4b54-95cb-ebd61f1b6ffd

# Sample data: cleaning times in minutes
data = [56, 18, 15, 30, 34, 30, 7, 15, 44, 23, 50, 20, 50, 18, 19, 32, 40, 36, 26, 30, 8, 15, 19, 23, 30, 74, 25, 34, 28, 61, 22, 13, 14, 70, 38, 31, 29, 31, 42, 62, 7, 40, 56, 28, 35, 12, 13, 13, 7, 30, 23, 42, 36, 38, 30, 25, 13, 55, 40, 40, 10, 10, 16, 27, 17, 15, 43, 27, 30, 22, 10, 27, 48, 30, 53, 24, 58, 11, 17, 26, 13, 86, 26, 40, 25, 13, 17, 47, 51, 41, 9, 13, 29, 5, 22, 15, 20, 75, 54, 40, 11, 34, 35, 37, 36, 39, 41, 40, 33, 28, 57, 45, 16, 12, 33, 27, 14, 26, 16, 18, 19, 70, 15, 11, 46, 35, 20, 22, 60, 7, 67, 28, 14, 15, 49, 20, 20, 40, 26, 20, 19, 26, 83, 22, 32, 29, 20, 14, 15, 38]
n_acco = 10
p_low, p_high = 0.025, 0.975

# Fit Weibull distribution to data
k, loc, lambda_ = weibull_min.fit(data, floc=0)
print(f"Shape parameter (k): {k}")
print(f"Scale parameter (λ): {lambda_}")

# Calculate the fitted values using the Weibull distribution
fitted_values = weibull_min.pdf(data, k, loc, lambda_)

# Calculate the actual values (normalized histogram counts)
num_bins = round(math.sqrt(len(data)))
hist, bin_edges = np.histogram(data, bins=num_bins, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Calculate the SSE
sse = np.sum((hist - weibull_min.pdf(bin_centers, k, loc, lambda_))**2)
print(f"SSE: {sse}")

# 95th percentile calculation for one accommodation
# Specifically, the scale parameter is the value below which 63.2% of the distribution's data points fall, representing the 63.2nd percentile.
p_ = [p_low, p_high, 0.632]
for p in p_:
    t_95_one = lambda_ * (-np.log(1 - p))**(1/k)
    print(f"{p*100}th percentile for one accommodation: {t_95_one:.2f} minutes")

# Mean and standard deviation for one Weibull distribution
mean_time = lambda_ * gamma(1 + 1/k)
std_time = np.sqrt(lambda_**2 * (gamma(1 + 2/k) - (gamma(1 + 1/k))**2))
print(f"Mean time: {mean_time:.2f} minutes")
print(f"Standard deviation: {std_time:.2f} minutes")
print(f"CI: {mean_time-(1.96*std_time)} - {mean_time+(1.96*std_time)}")
# Prepare data for the DataFrame
results = {
    'n_acco': [],
    '5th_percentile': [],
    '95th_percentile': [],
    '5th_percentile_avg': [],
    '95th_percentile_avg': []
}

# Parameters for the sum of n_acco independent Weibull distributions
for n_acco in range(1, 100001, 1000):
    mean_sum = n_acco * mean_time
    std_sum = np.sqrt(n_acco) * std_time

    p_5 = stats.norm.ppf(p_low, loc=mean_sum, scale=std_sum)
    p_95 = stats.norm.ppf(p_high, loc=mean_sum, scale=std_sum)

    results['n_acco'].append(n_acco)
    results['5th_percentile'].append(p_5)
    results['95th_percentile'].append(p_95)
    results['5th_percentile_avg'].append(p_5/n_acco)
    results['95th_percentile_avg'].append(p_95/n_acco)
    # print(f"5th percentile total cleaning time for {n_acco} accommodations: {p_5:.1f} minutes ({p_5/n_acco:.1f} per acco)")
    # print(f"95th percentile total cleaning time for {n_acco} accommodations: {p_95:.1f} minutes ({p_95/n_acco:.1f} per acco)")

# Create DataFrame
df = pd.DataFrame(results)
print(df)