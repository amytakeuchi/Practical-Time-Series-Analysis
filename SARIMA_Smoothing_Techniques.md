# SARIMA and Smoothing Techniques
- SARIMA method
- SARIMA Application
- Smoothing Techniques
  - Simple Exponential Smoothing
  - Double Exponential Smoothing
  - Holt Winters for trends
    
## SARIMA
SARIMA (Seasonal AutoRegressive Integrated Moving Average) is an extension of the ARIMA model that incorporates seasonal components in time series data. <br /> 
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

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error
from math import sqrt

# Generate sample data
np.random.seed(0)
dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
y = np.random.normal(loc=100, scale=10, size=100)
df = pd.DataFrame({'y': y}, index=dates)

# Split data into train and test sets
train = df[:80]
test = df[80:]

# Fit Simple Exponential Smoothing model
model = SimpleExpSmoothing(train['y'])
fit = model.fit(smoothing_level=0.2, optimized=False)

# Make predictions
forecast = fit.forecast(len(test))

# Calculate RMSE
rmse = sqrt(mean_squared_error(test['y'], forecast))
print(f'RMSE: {rmse}')

# Plot results
plt.figure(figsize=(12,6))
plt.plot(train.index, train['y'], label='Train')
plt.plot(test.index, test['y'], label='Test')
plt.plot(test.index, forecast, label='Forecast')
plt.legend()
plt.title('Simple Exponential Smoothing')
plt.show()

# Plot residuals
residuals = test['y'] - forecast
plt.figure(figsize=(12,6))
plt.plot(test.index, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Forecast Residuals')
plt.show()

# Plot ACF of residuals
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals)
plt.title('ACF of Residuals')
plt.show()

# Try different smoothing parameters
alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
plt.figure(figsize=(12,6))
for alpha in alphas:
    fit = model.fit(smoothing_level=alpha, optimized=False)
    forecast = fit.forecast(len(test))
    plt.plot(test.index, forecast, label=f'Alpha {alpha}')
plt.plot(train.index, train['y'], label='Train')
plt.plot(test.index, test['y'], label='Test')
plt.legend()
plt.title('Simple Exponential Smoothing with Different Alphas')
plt.show()
```
<img src="images/ses_result.png?" width="600" height="300"/>
<img src="images/ses_result_2.png?" width="600" height="300"/>
<img src="images/ses_result_3.png?" width="400" height="300"/>
<img src="images/ses_result_4.png?" width="600" height="300"/>

This code does the following: 
- Generates a sample time series without clear trend or seasonality.
- Splits the data into training and test sets.
- Fits a Simple Exponential Smoothing model with α = 0.2.
- Makes predictions on the test set.
- Calculates and prints the Root Mean Square Error (RMSE).
- Plots the original data, test data, and forecast.
- Plots the forecast residuals.
- Plots the Autocorrelation Function (ACF) of the residuals.
- Tries different smoothing parameters (α) and plots the results.

When interpreting the results:
- Look at the RMSE to assess forecast accuracy.
- Examine the residuals plot: ideally, residuals should be randomly distributed around zero.
- Check the ACF plot: for a good model, most lags (except lag 0) should be within the confidence bands.
- Compare forecasts with different α values: a higher α gives more weight to recent observations, while a lower α produces smoother forecasts.

## Double Exponential Smoothing
Double Exponential Smoothing, also known as Holt's linear trend method, is an extension of Simple Exponential Smoothing that can **handle time series data with a trend.** 

**What is Double Exponential Smoothing?** <br /> 
It's a forecasting method for time series data that have a trend but no seasonality. It uses two smoothing parameters: **one for the level and one for the trend.**

**How to use Double Exponential Smoothing in time series analysis:**
- Use it for short to medium-term forecasting of time series with a trend
- Choose appropriate smoothing parameters for level (α) and trend (β)
- Apply the method to generate forecasts
- Evaluate the forecast accuracy

**Formula:**
The basic formulas for Double Exponential Smoothing are: <br /> 
<br /> 
Level: $L_t = α * Y_t + (1 - α) * (L_(t-1) + b_(t-1))$ <br />
Trend: $b_t = β * (L_t - L_(t-1)) + (1 - β) * b_(t-1)$ <br />
Forecast: $F_(t+m) = L_t + m * b_t$
<br /> 
<br /> 
Where:
- $L_t$ is the level at time $t$
- $b_t$ is the trend at time $t$
- $Y_t$ is the observed value at time $t$
- $α$ is the smoothing parameter for the level $(0 < α ≤ 1)$
- $β$ is the smoothing parameter for the trend $(0 < β ≤ 1)$
- $F_(t+m)$ is the m-step ahead forecast

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from math import sqrt

# Generate sample data with trend
np.random.seed(0)
dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
trend = np.linspace(0, 10, 100)
noise = np.random.normal(0, 1, 100)
y = trend + noise
df = pd.DataFrame({'y': y}, index=dates)

# Split data into train and test sets
train = df[:80]
test = df[80:]

# Fit Double Exponential Smoothing model
model = ExponentialSmoothing(train['y'], trend='add')
fit = model.fit(smoothing_level=0.2, smoothing_trend=0.1, optimized=False)

# Make predictions
forecast = fit.forecast(len(test))

# Calculate RMSE
rmse = sqrt(mean_squared_error(test['y'], forecast))
print(f'RMSE: {rmse}')

# Plot results
plt.figure(figsize=(12,6))
plt.plot(train.index, train['y'], label='Train')
plt.plot(test.index, test['y'], label='Test')
plt.plot(test.index, forecast, label='Forecast')
plt.legend()
plt.title('Double Exponential Smoothing')
plt.show()

# Plot residuals
residuals = test['y'] - forecast
plt.figure(figsize=(12,6))
plt.plot(test.index, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Forecast Residuals')
plt.show()

# Plot ACF of residuals
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals)
plt.title('ACF of Residuals')
plt.show()

# Try different smoothing parameters
alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
betas = [0.1, 0.3, 0.5, 0.7, 0.9]
plt.figure(figsize=(12,6))
for alpha in alphas:
    for beta in betas:
        fit = model.fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
        forecast = fit.forecast(len(test))
        plt.plot(test.index, forecast, label=f'Alpha {alpha}, Beta {beta}')
plt.plot(train.index, train['y'], label='Train')
plt.plot(test.index, test['y'], label='Test')
plt.legend()
plt.title('Double Exponential Smoothing with Different Parameters')
plt.show()
```
<img src="images/des_result.png?" width="600" height="300"/>
<img src="images/des_result_2.png?" width="600" height="300"/>
<img src="images/des_result_3.png?" width="400" height="300"/>
<img src="images/des_result_4.png?" width="600" height="300"/>


This code does the following:
- Generates a sample time series with a trend.
- Splits the data into training and test sets.
- Fits a Double Exponential Smoothing model with α = 0.2 and β = 0.1.
- Makes predictions on the test set.
- Calculates and prints the Root Mean Square Error (RMSE).
- Plots the original data, test data, and forecast.
- Plots the forecast residuals.
- Plots the Autocorrelation Function (ACF) of the residuals.
- Tries different combinations of smoothing parameters (α and β) and plots the results.

When interpreting the results:
- Look at the RMSE to assess forecast accuracy.
- Examine the residuals plot: ideally, residuals should be randomly distributed around zero.
- Check the ACF plot: for a good model, most lags (except lag 0) should be within the confidence bands.
- Compare forecasts with different α and β values: higher values give more weight to recent observations and trends, while lower values produce smoother forecasts.

## Triple Exponential Smoothing (TES) and Holt-Winters method
**Triple Exponential Smoothing (TES) and Holt-Winters Method:**
These are advanced time series forecasting techniques that capture three components of a time series:
- Level (average)
- Trend (increasing or decreasing pattern)
- Seasonality (repeating patterns at fixed intervals)

The Holt-Winters method is a specific implementation of Triple Exponential Smoothing.

**Use in Time Series Analysis:**
These methods are used for forecasting when the time series exhibits both trend and seasonality. They're particularly useful for:
- Sales forecasting
- Demand prediction
- Stock market analysis
- Weather forecasting

**Formulas:**
The Holt-Winters method has two variations: Additive and Multiplicative. I'll provide the formulas for the Additive method:
- Level: $Lt = α(Yt - St-m) + (1 - α)(Lt-1 + Tt-1)$
- Trend: $Tt = β(Lt - Lt-1) + (1 - β)Tt-1$
- Seasonal: $St = γ(Yt - Lt) + (1 - γ)St-m$
- Forecast: $Ft+h = Lt + hTt + St-m+h$

Where:
- $Yt$ is the observed value at time t
- $Lt$ is the level at time t
- $Tt$ is the trend at time t
- $St$ is the seasonal component at time t
- $Ft+h$ is the forecast for h periods ahead
- $m$ is the number of periods in a seasonal cycle
- $α, β, γ$ are smoothing parameters ($0 < α, β, γ < 1$)

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Generate sample data
np.random.seed(0)
date_rng = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
n = len(date_rng)
trend = np.linspace(0, 5, n)
seasonal = np.sin(np.linspace(0, 8*np.pi, n))
noise = np.random.normal(0, 0.5, n)
y = trend + seasonal + noise

# Create DataFrame
df = pd.DataFrame(data={'date': date_rng, 'value': y})
df.set_index('date', inplace=True)

# Fit Holt-Winters model
model = ExponentialSmoothing(df['value'], 
                             seasonal_periods=365, 
                             trend='add', 
                             seasonal='add')
fitted_model = model.fit()

# Make predictions
forecast = fitted_model.forecast(steps=365)

# Plot results
plt.figure(figsize=(12,6))
plt.plot(df.index, df['value'], label='Observed')
plt.plot(fitted_model.fittedvalues.index, fitted_model.fittedvalues, label='Fitted')
plt.plot(forecast.index, forecast, label='Forecast')
plt.legend()
plt.title('Holt-Winters Triple Exponential Smoothing')
plt.show()

# Print model parameters
print("Model Parameters:")
print(f"Alpha (level): {fitted_model.params['smoothing_level']:.4f}")
print(f"Beta (trend): {fitted_model.params['smoothing_trend']:.4f}")
print(f"Gamma (seasonal): {fitted_model.params['smoothing_seasonal']:.4f}")
```
<img src="images/tes_result.png?" width="600" height="300"/>

This code:
- Generates synthetic daily data with trend and seasonality
- Fits a Holt-Winters model using additive trend and seasonality
- Makes a forecast for the next year
- Plots the original data, fitted values, and forecast
- Prints the optimized smoothing parameters

The resulting plot will show how well the model captures the trend and seasonality of the data and makes future predictions.
