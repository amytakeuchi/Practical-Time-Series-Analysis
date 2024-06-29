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
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import arma_generate_sample

# Set random seed for reproducibility
np.random.seed(42)

# Generate ARMA(2,1) process
ar = np.array([1, -0.6, 0.2])  # AR(2) coefficients
ma = np.array([1, 0.3])        # MA(1) coefficients
n_samples = 1000

y = arma_generate_sample(ar, ma, n_samples)

# Create a DataFrame
df = pd.DataFrame(y, columns=['value'])

# Fit ARMA model
model = ARIMA(df['value'], order=(2, 0, 1))  # ARMA(2,1) is equivalent to ARIMA(2,0,1)
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
```

