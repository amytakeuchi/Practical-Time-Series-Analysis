# Autocorrelation Function (ACF)
The autocorrelation function measures the correlation between a time series and its lagged values. It is a normalized version of the autocovariance function and takes values between -1 and 1.
<br /> 
The **autocorrelation coefficient between $X_t$ and $X_{t+k}$** is denoted by $ρ_k$ and is defined as:<br /> 
<br /> 
$ρ_k = γ_k / γ_0$
<br /> 
<br /> 
where <br /> 
- $γ_k$ is the autocovariance at lag $k$
- $γ_0$ is the variance of the time series (autocovariance at lag $0$).

This ensures that the autocorrelation coefficient is always between -1 and 1, with:
- $ρ_k$ = 1 indicating perfect positive correlation
- $ρ_k$ = -1 indicating perfect negative correlation
- $ρ_k$ = 0 indicating no correlation <br />

### Estimation of Autocorrelation Coefficient at lag $k$
The estimation of the autocorrelation coefficient at lag $k$, denoted by $r_k$, is given by:<br />

$r_k = c_k / c_0$

where
- $c_k$ is the sample autocovariance at lag $k$
- $c_0$ is the sample variance.<br />

Here's an example with numbers: <br />
Let's consider the same time series $X_t$ = {5, 7, 9, 6, 8, 10}.
We previously calculated the autocovariance at lag $1$ as $γ$(1) = 0.4.
The variance of the time series (autocovariance at lag 0) is $γ$(0) = 4.
Therefore, the autocorrelation coefficient at lag 1 is:
$ρ_1 = γ(1) / γ(0)$ = 0.4 / 4 = 0.1

```
import numpy as np

def autocorrelation(x, lags):
    """
    Calculate the autocorrelation function for a time series x and given lags.
    
    Args:
        x (array-like): The time series data.
        lags (array-like): The lags for which to calculate the autocorrelation.
        
    Returns:
        list: The autocorrelation coefficients for the given lags.
    """
    mean = np.mean(x)
    variance = np.var(x)
    autocorrelations = []
    
    for lag in lags:
        shifted_x = x[lag:]
        original_x = x[:len(x)-lag]
        autocovariance = np.mean((original_x - mean) * (shifted_x - mean))
        autocorrelation = autocovariance / variance
        autocorrelations.append(autocorrelation)
        
    return autocorrelations

# Example usage
time_series = [5, 7, 9, 6, 8, 10]
lags = [1, 2, 3]
autocorrelations = autocorrelation(time_series, lags)
print(autocorrelations)

```
***Output***
[-0.08571428571428572, -0.5142857142857143, 0.8285714285714285]

***Visualizing Autocorrelation Function***
```
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

# Generate a sample time series
np.random.seed(42)
time_series = np.random.randn(1000).cumsum()

# Calculate autocorrelation coefficients
lags = 50
autocorr = acf(time_series, nlags=lags)

# Plot the autocorrelation coefficients
plt.figure(figsize=(10, 6))
plt.bar(range(len(autocorr)), autocorr)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation Coefficient')
plt.title('Autocorrelation Function (ACF)')
plt.axhline(y=0, color='r', linestyle='--')
plt.axhline(y=1.96/np.sqrt(len(time_series)), color='r', linestyle='--')
plt.axhline(y=-1.96/np.sqrt(len(time_series)), color='r', linestyle='--')
plt.show()

# Print some autocorrelation coefficients
for i in range(1, 6):
    print(f"Autocorrelation coefficient at lag {i}: {autocorr[i]:.4f}")
```

