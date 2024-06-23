# Backward Shift Operator applied to MA and AR process
- [Series and Series Representation](#series-and-series-representation)
- [Backward Shift Operator](#backward-shift-operator)
- [Intro. to Invertibility](##intro-to-invertibility)
- [Duality](##duality)
- [Mean square Convergence](##mean-square-convergence)

## Series and Series Representation  
<img src="images/series.png?" width="600" height="300"/>
**Convergent Series** <br />
A series is convergent if **the sum of its terms approaches a finite limit** as the number of terms increases indefinitely.
<br />
Example: The geometric series with $|r| < 1$
$1 + r + r^2 + r^3 + ...$
<br />
```
def geometric_series(r, n):
    return sum(r**i for i in range(n))

# Example with r = 0.5
r = 0.5
for n in [10, 100, 1000, 10000]:
    print(f"Sum of first {n} terms: {geometric_series(r, n)}")

print(f"Theoretical limit: {1 / (1 - r)}")
```
Sum of first 10 terms: 1.998046875 <br />
Sum of first 100 terms: 2.0 <br />
Sum of first 1000 terms: 2.0 <br />
Sum of first 10000 terms: 2.0 <br />
Theoretical limit: 2.0 <br />
<br />
**Divergent Series**



## Backward Shift Operator
**Backward Shift Definition** <br /> 
<img src="images/bsf_definition.png?" width="600" height="300"/>
<br /> 
<br /> 
**Definition**
<br /> 
For a time series $Yt$, the backward shift operator $B$ is defined as:
<br /> 
<br /> 
$B(Yt) = Yt-1$
<br /> 
<br /> 
This means that applying $B$ to $Yt$ gives you the previous value in the series.

**Multiple applications** <br /> 
You can apply B multiple times: <br /> 
<br /> 
$B²(Yt) = B(B(Yt)) = B(Y{t-1}) = Y{t-2}$ <br /> 
$B^kXt = X{t-k}$
<br /> 

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create a simple time series
np.random.seed(0)
t = np.arange(100)
y = 10 + 0.5*t + np.random.normal(0, 2, 100)

# Create a DataFrame
df = pd.DataFrame({'t': t, 'y': y})

# Apply backward shift operator
df['y_lag1'] = df['y'].shift(1)  # B(Yt)
df['y_lag2'] = df['y'].shift(2)  # B²(Yt)

# Plot the results
plt.figure(figsize=(12,6))
plt.plot(df['t'], df['y'], label='Original')
plt.plot(df['t'], df['y_lag1'], label='B(Yt)')
plt.plot(df['t'], df['y_lag2'], label='B²(Yt)')
plt.legend()
plt.title('Time Series with Backward Shift Operator')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

# Print first few rows
print(df.head())
```
<img src="images/bso_timeseries.png?" width="900" height="450"/>

## Intro to Invertibility
<img src="images/invertibility_definition.png?" width="600" height="300"/>

- **Definition**: A stochastic process is invertible if the current value can be expressed as a convergent infinite sum of past and present observations. 
- **Importance**: Invertibility ensures that we can express the process in terms of past observations, which is crucial for forecasting and interpretation. 
- **Application**: It's particularly important for MA processes. An invertible MA process can be approximated by an AR process of infinite order.
<br />

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def generate_ma1(theta, n):
    """Generate an MA(1) process: Yt = εt + θεt-1"""
    epsilon = np.random.normal(0, 1, n+1)
    y = epsilon[1:] + theta * epsilon[:-1]
    return y

def fit_ar(y, order):
    """Fit an AR model of specified order"""
    model = ARIMA(y, order=(order, 0, 0))
    results = model.fit()
    return results.arparams

# Generate invertible and non-invertible MA(1) processes
np.random.seed(0)
n = 1000
y_inv = generate_ma1(0.5, n)  # Invertible: |θ| < 1
y_noninv = generate_ma1(2, n)  # Non-invertible: |θ| > 1

# Fit AR models of increasing order
max_order = 10
ar_params_inv = [fit_ar(y_inv, i) for i in range(1, max_order+1)]
ar_params_noninv = [fit_ar(y_noninv, i) for i in range(1, max_order+1)]

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

for i, params in enumerate(ar_params_inv):
    ax1.plot(range(len(params)), params, label=f'AR({i+1})')
ax1.set_title('AR Coefficients for Invertible MA(1)')
ax1.set_xlabel('Lag')
ax1.set_ylabel('Coefficient')
ax1.legend()

for i, params in enumerate(ar_params_noninv):
    ax2.plot(range(len(params)), params, label=f'AR({i+1})')
ax2.set_title('AR Coefficients for Non-invertible MA(1)')
ax2.set_xlabel('Lag')
ax2.set_ylabel('Coefficient')
ax2.legend()

plt.tight_layout()
plt.show()

# Print theoretical AR coefficients for invertible process
theta = 0.5
theoretical_ar_coefs = [-theta**i for i in range(1, 11)]
print("Theoretical AR coefficients for invertible MA(1):")
print(theoretical_ar_coefs)
```
<img src="images/invertibility_ar.png?" width="900" height="450">

## Duality

## Mean Square Convergence

