# Partial Autocorrelation Function
- Partial Autocorrelation and the PACF First Examples <br /> 
- Partial Autocorrelation and the PACF - Concept Development <br />

**Write Yule-Walker Equations in matrix notation, and estimate model parameter**
- Yule-Walker Equations in Matrix Form
- AR(2) Simulation (Parameter Estimation)
- Yule Walker Estimation - AR(2) Simulation
- AR(3) Simulation (Parameter Estimation)
- Yule Walker Estimation - AR(3) Simulation

**AR processes - Data Oriented Examples**
- recruitment data - model fitting
- Johnson & Johnson - model fitting

## Partial Autocorrelation and the PACF
**What is Partial Autocorrelation (PACF)?** <br /> 
PACF measures the correlation between an observation in a time series with observations at prior time steps, with the effects of the intervening observations removed. In other words, it captures the direct effect of a lag on the current value, excluding indirect effects through intermediate lags.
<br /> 
<br /> 
**How and when to use PACF in time series analysis:** <br /> 
PACF is primarily used to: <br /> 
a) Determine the order (p) of an autoregressive AR(p) model <br /> 
b) Identify significant lags in time series data <br /> 
c) Distinguish between AR and MA (Moving Average) processes <br /> 
<br /> 
**You use PACF when:**
- You want to identify the appropriate lags for an AR model
- You need to understand the direct relationships between observations at different lags
- You're trying to differentiate between AR and MA processes in ARIMA modeling

**Formula:** <br /> 
The formula for PACF is more complex than for regular autocorrelation. It's typically calculated recursively using the Durbin-Levinson algorithm or solving the Yule-Walker equations. <br /> 
<br /> 
For lag $k$, the partial autocorrelation $φ_kk$ is: <br /> 
<br /> 
$φ_kk = Corr(X_t, X_{t-k} | X_{t-1}, ..., X_{t-k+1})$ <br /> 
<br /> 
This represents the correlation between $X_t$ and $X_{t-k}$, controlling for the effects of intermediate lags. <br /> 

```
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

# Set random seed for reproducibility
np.random.seed(42)

# Generate AR(2) process
n = 1000
ar_params = np.array([1.5, -0.75])
ma_params = np.array([1])
ar = np.r_[1, -ar_params]
ma = np.r_[1, ma_params]
y = arma_generate_sample(ar, ma, n)

# Plot the time series
plt.figure(figsize=(12, 4))
plt.plot(y)
plt.title('AR(2) Process')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

# Plot PACF
plt.figure(figsize=(12, 4))
plot_pacf(y, lags=20, method='ywm')
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# For comparison, plot ACF
plt.figure(figsize=(12, 4))
plot_acf(y, lags=20)
plt.title('Autocorrelation Function (ACF)')
plt.show()
```
<img src="images/pacf_ar2.png?" width="600" height="300"/>

<img src="images/pacf_pacf.png?" width="600" height="300"/>

<img src="images/pacf_acf.png?" width="600" height="300"/>

**This code does the following:** <br /> 
- Generates an AR(2) process with parameters [1.5, -0.75].
- Plots the generated time series.
- Plots the Partial Autocorrelation Function (PACF) using the 'ywm' (Yule-Walker with unbiased estimate of the process variance) method.
- Plots the Autocorrelation Function (ACF) for comparison.

**Interpreting the results:** <br /> 
- In the PACF plot, you should see significant spikes at lags 1 and 2, and insignificant values after that. This indicates an AR(2) process. 
- The ACF plot, in contrast, will show a more gradually decaying pattern.

## Yule-Walker Equations in Matrix Form
### Reminding AR(p) process
<img src="images/yw_ar_remind.png?" width="600" height="200"/>
<img src="images/yw_ar_remind_2.png?" width="600" height="300"/>
<img src="images/yw_ar_remind_3.png?" width="600" height="200"/>

### Yule-Walker Equations
<img src="images/yw_equations.png?" width="600" height="200"/>
<img src="images/yw_equations_for_k.png?" width="600" height="300"/>
<img src="images/yw_equations_for_-k.png?" width="600" height="200"/>
<img src="images/yw_equations_for_1.png?" width="600" height="200"/>

### Matrix Form

### Example - AR(2)

## Yule Walker Estimation - AR(2) Simulation
