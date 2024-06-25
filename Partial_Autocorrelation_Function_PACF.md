# Partial Autocorrelation Function
Partial Autocorrelation and the PACF First Examples
Partial Autocorrelation and the PACF - Concept Development
Write Yule-Walker Equations in matrix notation, and estimate model parameter
- Yule-Walker Equations in Matrix Form
- AR(2) Simulation (Parameter Estimation)
- Yule Walker Estimation - AR(2) Simulation
- AR(3) Simulation (Parameter Estimation)
- Yule Walker Estimation - AR(3) Simulation
AR processes - Data Oriented Examples
- recruitment data - model fitting
- Johnson & Johnson - model fitting

## Partial Autocorrelation and the PACF
**What is Partial Autocorrelation (PACF)?** <br /> 
PACF measures the correlation between an observation in a time series with observations at prior time steps, with the effects of the intervening observations removed. In other words, it captures the direct effect of a lag on the current value, excluding indirect effects through intermediate lags.
<br /> 
**How and when to use PACF in time series analysis:** <br /> 
PACF is primarily used to: <br /> 
a) Determine the order (p) of an autoregressive AR(p) model <br /> 
b) Identify significant lags in time series data <br /> 
c) Distinguish between AR and MA (Moving Average) processes <br /> 
**You use PACF when:**
- You want to identify the appropriate lags for an AR model
- You need to understand the direct relationships between observations at different lags
- You're trying to differentiate between AR and MA processes in ARIMA modeling

**Formula:** <br /> 
The formula for PACF is more complex than for regular autocorrelation. It's typically calculated recursively using the Durbin-Levinson algorithm or solving the Yule-Walker equations. <br /> 
<br /> 
