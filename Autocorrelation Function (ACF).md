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

