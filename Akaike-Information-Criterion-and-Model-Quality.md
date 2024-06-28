# Akaike Information Criterion (AIC) and Model Quality

## What is the Akaike Information Criterion (AIC)?
AIC is a measure of the relative quality of statistical models for a given set of data. It estimates the relative amount of information lost by a given model: the less information a model loses, the higher the quality of that model. <br /> 

## How is AIC used in time series analysis?
In time series analysis, AIC is primarily used for:  <br /> 
  a) Selecting the optimal order of ARIMA (Autoregressive Integrated Moving Average) models <br /> 
  b) Comparing different models (e.g., AR vs MA vs ARMA) <br /> 
  c) Balancing model complexity with goodness of fit <br /> 
The model with the lowest AIC is generally preferred.  <br /> 

## Formula
The general formula for AIC is: <br /> 
$AIC = 2k - 2ln(L)$
Where: <br /> 
- $k$ is the number of parameters in the model
- $L$ is the maximum value of the likelihood function for the model
<br /> 
For linear regression models (including many time series models), AIC is often calculated as: <br />
<br /> 
$AIC = 2k + n * ln(RSS/n)$
<br /> 
Where:
- n is the number of observations
- RSS is the residual sum of squares
