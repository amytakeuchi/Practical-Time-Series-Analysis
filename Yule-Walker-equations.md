# Yule-Walker equations

## What is Yule-Walker equation:
Yule-Walker equations, also known as autocorrelation equations, are a set of linear equations used to estimate the parameters of an autoregressive (AR) model in time series analysis. They relate the autocorrelation function of a time series to the parameters of an AR model.

### How are they used in time series analysis?
Yule-Walker equations are primarily used to: <br /> 
a) Estimate **the coefficients** of an AR(p) model <br /> 
b) Determine **the order** of an AR model <br /> 
c) Calculate **the theoretical autocorrelation function** of an AR process

### Formula
For an AR(p) model: <br /> 
<br /> 
$X_t = c + φ₁X_{t-1} + φ₂X_{t-2} + ... + φ_pX_{t-p} + ε_t$
<br /> 
The Yule-Walker equations are: <br /> 
<br /> 
$ρ_k = φ₁ρ_{k-1} + φ₂ρ_{k-2} + ... + φ_pρ_{k-p}$
<br /> 
<br /> 
Where:
- $ρ_k$ is the autocorrelation at lag k
- $φ_i$ are the AR coefficients
- $p$ is the order of the AR model

In matrix form:
$[ρ₁]   [1   ρ₁  ρ₂ ... ρ_{p-1}] [φ₁]$
$[ρ₂] = [ρ₁  1   ρ₁ ... ρ_{p-2}] [φ₂]$
$[...]   [...               ...] [...]$
$[ρ_p]   [ρ_{p-1} ... ρ₁    1  ] [φ_p]$

