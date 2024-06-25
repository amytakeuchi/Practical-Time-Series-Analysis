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

In matrix form: <br /> 
$[ρ₁]   [1   ρ₁  ρ₂ ... ρ_{p-1}] [φ₁]$ <br /> 
$[ρ₂] = [ρ₁  1   ρ₁ ... ρ_{p-2}] [φ₂]$ <br /> 
$[...]   [...               ...] [...]$ <br /> 
$[ρ_p]   [ρ_{p-1} ... ρ₁    1  ] [φ_p]$ <br /> 
<br /> 

```
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_process import arma_generate_sample

# Set random seed for reproducibility
np.random.seed(42)

# Generate AR(2) process
n = 1000
ar_params = np.array([1.5, -0.75])
ma_params = np.array([1])
ar = np.r_[1, -ar_params]
ma = np.r_[1, ma_params]
y = arma_generate_sample(ar, ma, n)

# Estimate AR parameters using Yule-Walker
model = AutoReg(y, lags=2, old_names=False)
results = model.fit(cov_type='HC0')

# Print results
print("True parameters:", ar_params)
print("Estimated parameters:", results.params[1:])

# Plot the time series
plt.figure(figsize=(10, 6))
plt.plot(y)
plt.title('AR(2) Process')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

# Plot autocorrelation function
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(y, lags=20)
plt.title('Autocorrelation Function')
plt.show()
```
True parameters: [ 1.5  -0.75] <br /> 
Estimated parameters: [ 1.63315289 -0.8717853 ]
<img src="images/yule_walker_ar.png?" width="500" height="300"/>
<img src="images/yule_walker_acf.png?" width="500" height="300"/>

