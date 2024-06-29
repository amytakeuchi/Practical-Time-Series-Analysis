# ARMA

## What is ARMA (p, q) Models
<img src="images/arma_definition.png?" width="500" height="300"/>
<img src="images/arma_definition_2.png?" width="500" height="300"/>

ARMA stands for AutoRegressive Moving Average. It's a model that combines two components: 
- AR(p): AutoRegressive component of order p
- MA(q): Moving Average component of order q

**AR(p) Component**:
- p is the order of the autoregressive term
- It models the dependency between an observation and a certain number (p) of lagged observations

**MA(q) Component**:
- q is the order of the moving average term
- It models the dependency between an observation and a residual error from a moving average model applied to lagged observations

**Formula**: <br /> 
<img src="images/arma_definition_3.png?" width="500" height="200"/>
<br /> 
The ARMA(p,q) model can be written as: <br /> 
$X_t = c + ε_t + Σ(i=1 to p) φ_i * X_{t-i} + Σ(i=1 to q) θ_i * ε_{t-i}$
Where: 
- $X_t$ is the value at time t
- $c$ is a constant
- $ε_t$ is white noise
- $φ_i$ are the parameters of the AR term
- $θ_i$ are the parameters of the MA term

**ARMA models are used to:** <br /> 
- Understand the underlying process of a time series
- Make forecasts of future values
- Remove autocorrelation from residuals in regression analysis

**Relationship between AR and MA:** <br /> 
- AR models assume the current value depends directly on past values
- MA models assume the current value depends on past forecast errors
- ARMA combines both, allowing for more flexible modeling of complex time series
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima.model import ARIMA

# Set random seed for reproducibility
np.random.seed(42)

# Generate ARMA(2,1) process
ar_params = np.array([1, -0.6, 0.2])  # AR(2) coefficients
ma_params = np.array([1, 0.3])        # MA(1) coefficients
n_samples = 1000

y = arma_generate_sample(ar_params, ma_params, n_samples)

# Create a DataFrame
df = pd.DataFrame(y, columns=['value'])

# Fit ARMA model
# Note: We use ARIMA with order (p,0,q) which is equivalent to ARMA(p,q)
model = ARIMA(df['value'], order=(2, 0, 1))  # ARMA(2,1)
results = model.fit()

# Print summary
print(results.summary())

# Plot original series and fitted values
plt.figure(figsize=(12,6))
plt.plot(df.index, df['value'], label='Original')
plt.plot(df.index, results.fittedvalues, color='red', label='Fitted')
plt.legend()
plt.title('ARMA(2,1) Process and Fitted Model')
plt.show()

# Forecast next 50 steps
forecast = results.forecast(steps=50)

# Plot forecast
plt.figure(figsize=(12,6))
plt.plot(df.index, df['value'], label='Original')
plt.plot(range(1000, 1050), forecast, color='red', label='Forecast')
plt.legend()
plt.title('ARMA(2,1) Process and Forecast')
plt.show()

# Analyze residuals
residuals = results.resid
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot residuals
ax1.plot(residuals)
ax1.set_title('Residuals of ARMA(2,1) Model')
ax1.set_xlabel('Time')
ax1.set_ylabel('Residual')

# Plot residual histogram
ax2.hist(residuals, bins=30)
ax2.set_title('Histogram of Residuals')
ax2.set_xlabel('Residual Value')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```
<img src="images/arma_results.png?" width="400" height="300"/>
<img src="images/arma_results_2.png?" width="600" height="300"/>
<img src="images/arma_results_3.png?" width="600" height="300"/>
<img src="images/arma_results_4.png?" width="600" height="500"/>

Note that we're still using the ARIMA function, but with order (2,0,1), which is equivalent to ARMA(2,1). This is because statsmodels implements ARMA as a special case of ARIMA.

## ARMA Example

# ARIMA(p,d,q) Method
## Before ARIMA: Revision:
<img src="images/ARIMA_revision.png?" width="600" height="200"/>
<img src="images/ARIMA_revision_2.png?" width="600" height="300"/>
<img src="images/ARIMA_revision_3.png?" width="600" height="200"/>
<img src="images/ARIMA_revision_4.png?" width="600" height="200"/>
<img src="images/ARIMA_revision_5.png?" width="600" height="300"/>

## ARIMA Method Definition
<img src="images/arima_definition.png?" width="600" height="300"/>
<img src="images/arima_definition_2.png?" width="600" height="200"/>
<img src="images/arima_definition_3.png?" width="600" height="200"/>
ARIMA(p,d,q), which stands for AutoRegressive Integrated Moving Average, is a widely used statistical method for time series forecasting. It combines three key aspects:

**Components of ARIMA(p,d,q):**
- $AR(p)$: AutoRegressive component
  - $p$ is the order of the autoregressive term
  - It uses past values to predict the current value
- $I(d)$: Integrated component
  - $d$ is the degree of differencing required to make the time series stationary
  - Differencing involves computing differences between consecutive observations
- $MA(q)$: Moving Average component
  - $q$ is the order of the moving average term
  - It uses past forecast errors in a regression-like model

**ARIMA(p,d,q) Simplification** <br /> 
The ARIMA(p, q) model is a simplification where we assume $d$ =0. Thus, the model combines the AR and MA parts without any differencing.

**Formula** <br /> 
For an ARIMA(p,d,q) model: <br /> 
<br /> 
$(1 - Σφᵢ Lⁱ)(1 - L)ᵈ Yₜ = (1 + Σθⱼ Lʲ)εₜ$
<br />
<br /> 
​Where:
- $L$ is the lag operator
- $φᵢ$ are the parameters of the AR term
- $θⱼ$ are the parameters of the MA term
- $εₜ$ is white noise
<br /> 

**How to use ARIMA in time series analysis:**
- Check if the series is stationary. If not, difference it until it becomes stationary.
- Identify potential p and q values using ACF and PACF plots.
- Fit several ARIMA models with different p, d, and q values.
- Select the best model based on AIC, BIC, or another criterion.
- Check the model's residuals for any remaining autocorrelation.
- Use the model for forecasting.
 
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Generate sample data
np.random.seed(0)
dates = pd.date_range(start='2000', periods=100, freq='M')
y = pd.Series(np.cumsum(np.random.randn(100)), index=dates)

# Plot the data
plt.figure(figsize=(12,6))
plt.plot(y)
plt.title('Sample Time Series Data')
plt.show()

# Plot ACF and PACF
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(y, ax=ax1)
plot_pacf(y, ax=ax2)
plt.show()

# Fit ARIMA model
model = ARIMA(y, order=(1,1,1))
results = model.fit()

# Print summary
print(results.summary())

# Forecast
forecast = results.forecast(steps=12)

# Plot forecast
plt.figure(figsize=(12,6))
plt.plot(y, label='Observed')
plt.plot(pd.date_range(start=y.index[-1], periods=13, freq='M')[1:], forecast, label='Forecast')
plt.legend()
plt.title('ARIMA Forecast')
plt.show()
```

<img src="images/arima_modeling_viz.png?" width="600" height="300"/>
<img src="images/arima_modeling_viz_2.png?" width="600" height="300"/>
<img src="images/arima_modeling_viz_3.png?" width="600" height="300"/>
<img src="images/arima_modeling_viz_forecast.png?" width="600" height="300"/>
This code:
- Generates a non-stationary time series
- Plots ACF and PACF to help identify p and q
- Fits an ARIMA(1,1,1) model
- Prints the model summary
- Forecasts the next 12 periods
- Plots the original series and the forecast

The relationship between AR and MA:
- AR models assume the current value depends directly on past values
- MA models assume the current value depends on past forecast errors
- ARIMA combines both, allowing for more flexible modeling of complex time series, and adds differencing to handle non-stationarity

**Some visualizations to assess the fit of the model versus the original time series data:**
- The original data with the fitted values
- The residuals over time
- A Q-Q plot of the residuals to check for normality
- The ACF of the residuals to check for remaining autocorrelation
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats

# Generate sample data
np.random.seed(0)
dates = pd.date_range(start='2000', periods=100, freq='M')
y = pd.Series(np.cumsum(np.random.randn(100)), index=dates)

# Fit ARIMA model
model = ARIMA(y, order=(1,1,1))
results = model.fit()

# Print summary
print(results.summary())

# Create a figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. Original data with fitted values
ax1.plot(y, label='Original')
ax1.plot(results.fittedvalues, color='red', label='Fitted')
ax1.set_title('Original vs Fitted')
ax1.legend()

# 2. Residuals over time
residuals = results.resid
ax2.plot(residuals)
ax2.set_title('Residuals over Time')
ax2.set_xlabel('Date')
ax2.set_ylabel('Residual')

# 3. Q-Q plot of residuals
stats.probplot(residuals, dist="norm", plot=ax3)
ax3.set_title("Q-Q plot of Residuals")

# 4. ACF of residuals
plot_acf(residuals, ax=ax4)
ax4.set_title('ACF of Residuals')

plt.tight_layout()
plt.show()

# Forecast
forecast = results.forecast(steps=12)

# Plot forecast
plt.figure(figsize=(12,6))
plt.plot(y, label='Observed')
plt.plot(results.fittedvalues, color='red', label='Fitted')
plt.plot(pd.date_range(start=y.index[-1], periods=13, freq='M')[1:], forecast, color='green', label='Forecast')
plt.legend()
plt.title('ARIMA Model: Original, Fitted, and Forecast')
plt.show()
```
<img src="images/arima_fitting.png?" width="600" height="300"/>
<img src="images/arima_fitting_2.png?" width="900" height="300"/>
<img src="images/arima_fitting_3.png?" width="900" height="300"/>
<img src="images/arima_fitting_4.png?" width="600" height="300"/>
This modified code adds several new visualizations: <br /> 
- **Original vs Fitted**: This plot shows how well the model fits the original data. The closer the red line (fitted values) is to the blue line (original data), the better the fit.
- **Residuals over Time**: This plot helps identify any patterns in the residuals. Ideally, the residuals should look like random noise with no clear pattern.
- **Q-Q plot of Residuals**: This plot helps check if the residuals are normally distributed. If the points closely follow the diagonal line, it suggests the residuals are approximately normally distributed.
- **ACF of Residuals**: This plot helps check if there's any remaining autocorrelation in the residuals. Ideally, all lags (except lag 0) should be within the blue confidence bands.
- **Original, Fitted, and Forecast**: This final plot combines the original data, the fitted values, and the forecast, giving a comprehensive view of the model's performance.
