# SARIMA and Smoothing Techniques
- SARIMA method
- SARIMA Application
- Smoothing Techniques
  - Simple Exponential Smoothing
  - Double Exponential Smoothing
  - Holt Winters for trends
    
## SARIMA
SARIMA (Seasonal AutoRegressive Integrated Moving Average) is an extension of the ARIMA model that incorporates seasonal components in time series data. 
<img src="images/SARIMA_definition.png?" width="600" height="300"/>
<img src="images/SARIMA_definition_2.png?" width="600" height="300"/>
<img src="images/SARIMA_definition_3.png?" width="600" height="300"/>
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
<br /> 
Where:
- $B$ is the backshift operator
- $φ(B)$ is the non-seasonal AR term
- $θ(B)$ is the non-seasonal MA term
- $Φ(B^m)$ is the seasonal AR term
- $Θ(B^m)$ is the seasonal MA term
- $(1-B)^d$ is the non-seasonal differencing term
- $(1-B^m)^D$ is the seasonal differencing term
<img src="images/SARIMA_example.png?" width="600" height="300"/>
<img src="images/SARIMA_example_2.png?" width="600" height="300"/>

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Generate seasonal data
np.random.seed(0)
dates = pd.date_range(start='2010-01-01', end='2019-12-31', freq='M')
n = len(dates)
trend = np.linspace(0, 5, n)
seasonal = 5 * np.sin(2 * np.pi * np.arange(n) / 12)
noise = np.random.normal(0, 1, n)
y = trend + seasonal + noise

# Create DataFrame
df = pd.DataFrame({'y': y}, index=dates)

# Plot the data
plt.figure(figsize=(12,6))
plt.plot(df)
plt.title('Simulated Seasonal Time Series')
plt.show()

# Plot ACF and PACF
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(df, ax=ax1)
plot_pacf(df, ax=ax2)
plt.show()

# Fit SARIMA model
model = SARIMAX(df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Print summary
print(results.summary())

# Forecast
forecast = results.get_forecast(steps=24)
forecast_ci = forecast.conf_int()

# Plot forecast
plt.figure(figsize=(12,6))
plt.plot(df.index, df, label='Observed')
plt.plot(forecast.predicted_mean.index, forecast.predicted_mean, color='red', label='Forecast')
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.legend()
plt.title('SARIMA Forecast')
plt.show()

# Diagnostic plots
results.plot_diagnostics(figsize=(12, 8))
plt.show()
```
<br /> 
<img src="images/SARIMA_results.png?" width="600" height="300"/>
<img src="images/SARIMA_results_2.png?" width="600" height="400"/>
<img src="images/SARIMA_results_3.png?" width="500" height="300"/>
<img src="images/SARIMA_results_4.png?" width="600" height="300"/>
<img src="images/SARIMA_results_5.png?" width="600" height="400"/>

This code does the following: 
- Generates a simulated seasonal time series with trend, seasonality, and noise.
- Plots the original data, ACF, and PACF.
- Fits a SARIMA(1,1,1)(1,1,1,12) model. This means:
  - Non-seasonal components: AR(1), differencing(1), MA(1)
  - Seasonal components: SAR(1), seasonal differencing(1), SMA(1), with a season length of 12
- Prints the model summary.
- Forecasts the next 24 periods with confidence intervals.
- Plots the original data, forecast, and confidence intervals.
- Shows diagnostic plots for model evaluation.

When interpreting the results:
- Check the AIC and BIC in the model summary (lower is generally better).
- Look at the p-values of the coefficients to see if they're significant.
- Examine the diagnostic plots:
- Standardized residuals should look like white noise.
- The Q-Q plot should follow the diagonal line.
- The correlogram should show no significant correlations.

## Simple Exponential Smoothing
Exponential Smoothing (SES) is a time series forecasting method for data **without clear trend or seasonality.**

**What is Simple Exponential Smoothing?**
SES is a weighted average forecasting method that applies exponentially decreasing weights to older observations. It's particularly useful for forecasting data where there's no clear trend or seasonal pattern.

**How to use SES in time series analysis:**
- Use it for short-term forecasting of time series without trend or seasonality
- Choose an appropriate smoothing parameter (α)
- Apply the method to generate forecasts
- Evaluate the forecast accuracy

**Formula** <br /> 
<br /> 
$S_t = α * Y_t + (1 - α) * S_(t-1)$ <br /> 
<br /> 
Where:
- $S_t$ is the smoothed value at time t
- $Y_t$ is the observed value at time t
- $α$ is the smoothing parameter (0 < α ≤ 1)
- $S_(t-1)$ is the previous smoothed value

For forecasting: <br /> 
<br /> 
$F_(t+1) = S_t$
<br /> 
<br /> 
Where $F_(t+1)$ is the forecast for the next period.
