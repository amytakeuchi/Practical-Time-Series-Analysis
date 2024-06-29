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

# ARIMA Method
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
ARIMA, which stands for AutoRegressive Integrated Moving Average, is a widely used statistical method for time series forecasting. It combines three key aspects:
- AR (AutoRegressive)
- I (Integrated)
- MA (Moving Average)

**Components**
- **AR(p)**: The AutoRegressive part involves regressing the variable on its own lagged (past) values. The parameter $p$ indicates the number of lagged observations included in the model.
- **I(d)**: The Integrated part involves differencing the data to make it stationary (i.e., to remove trends and seasonality). The parameter 
  $d$ indicates the number of times the data needs to be differenced.
- **MA(q)**: The Moving Average part involves modeling the error term as a linear combination of lagged forecast errors. The parameter 
  $q$ indicates the number of lagged forecast errors in the model.

**ARIMA(p, q) Simplification** <br /> 
The ARIMA(p, q) model is a simplification where we assume $d$ =0. Thus, the model combines the AR and MA parts without any differencing.

**Formula** <br /> 
The general ARIMA($p, q$) model can be expressed as:
<br /> 
$yt = c+∑i=1 pϕi yt−i +∑ j=1 q θj ϵt−j + ϵt$
<br /> 
​Where:
- $yt$ is the actual value at time $t$.
- c is a constant term.
- $ϕi$ are the coefficients of the AR terms.
- $θj$ are the coefficients of the MA terms.
- $ϵt$ is the white noise error term at time $t$.
<br /> 
**AR and MA Methods Relationship**
AR (AutoRegressive) Method: Models the current value of the time series as a linear combination of its past values. <br /> 
For example, AR(1): 
𝑦
𝑡
=
𝜙
1
𝑦
𝑡
−
1
+
𝜖
𝑡
y 
t
​
 =ϕ 
1
​
 y 
t−1
​
 +ϵ 
t
MA (Moving Average) Method: Models the current value of the time series as a linear combination of past forecast errors.

 <img src="images/arima_modeling.png?" width="600" height="300"/>



