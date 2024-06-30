# SARIMA and Smoothing Techniques
- SARIMA method
- SARIMA Application
- Smoothing Techniques
  - Simple Exponential Smoothing
  - Double Exponential Smoothing
  - Holt Winters for trends
    
## SARIMA
SARIMA (Seasonal AutoRegressive Integrated Moving Average) is an extension of the ARIMA model that incorporates seasonal components in time series data. 
<br /> 
**SARIMA Model:**
<br /> 
SARIMA(p,d,q)(P,D,Q)m
<br /> 
Where:
- (p,d,q) are the non-seasonal parameters
- (P,D,Q) are the seasonal parameters
- m is the number of periods per season

**How to use SARIMA in time series analysis:**
- Identify if there's a seasonal pattern in your data
- Determine the seasonal period (m)
- Choose appropriate values for p, d, q, P, D, and Q
- Fit the model and check diagnostics
- Use for forecasting

*Formula:*
The general form of a SARIMA model combines the non-seasonal and seasonal components: <br /> 
<br /> 
$Φ(B^m)φ(B)(1-B)^d(1-B^m)^D y_t = θ(B)Θ(B^m)ε_t$
<br /> 
Where:
- $B$ is the backshift operator
- $φ(B)$ is the non-seasonal AR term
- $θ(B)$ is the non-seasonal MA term
- $Φ(B^m)$ is the seasonal AR term
- $Θ(B^m)$ is the seasonal MA term
- $(1-B)^d$ is the non-seasonal differencing term
- $(1-B^m)^D$ is the seasonal differencing term
