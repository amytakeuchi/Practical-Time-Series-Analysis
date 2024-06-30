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
