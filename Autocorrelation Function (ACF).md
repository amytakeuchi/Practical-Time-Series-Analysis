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
- $ρ_k$ = 0 indicating no correlation
<br /> 
The estimation of the autocorrelation coefficient at lag $k$, denoted by $r_k$, is given by:<br />
<br />
$r_k = c_k / c_0$
<br />
<br />
where
- $c_k$ is the sample autocovariance at lag $k$
- $c_0$ is the sample variance.<br />
<br />
Here's an example with numbers:
