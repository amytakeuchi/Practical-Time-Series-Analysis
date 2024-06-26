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

**Divergent Series** <br /> 
A series is divergent if the sum of its terms does not approach a finite limit as the number of terms increases indefinitely.
<br /> 
<br /> 

Example: The harmonic series <br /> 
$1 + 1/2 + 1/3 + 1/4 + ...$
```
def harmonic_series(n):
    return sum(1/i for i in range(1, n+1))

for n in [10, 100, 1000, 10000, 100000]:
    print(f"Sum of first {n} terms: {harmonic_series(n)}")
```
Sum of first 10 terms: 2.9289682539682538 <br />
Sum of first 100 terms: 5.187377517639621 <br />
Sum of first 1000 terms: 7.485470860550343 <br />
Sum of first 10000 terms: 9.787606036044348 <br />
Sum of first 100000 terms: 12.090146129863335 <br />
<br />
<img src="images/series_example.png?" width="600" height="300"/>
<img src="images/series_example_2.png?" width="600" height="300"/>

**Absolute Convergence** <br /> 
<img src="images/absolute_convergence.png?" width="600" height="300"/>

**Convergence tests**
- Integral test
- Comparison test
- Limit Comparison test
- Alternating series test
- Ratio test
- Root test

**Geometric Series** <br /> 
<img src="images/geometric_series.png?" width="600" height="450"/>
```
def geometric_series(r, n):
    return sum(r**i for i in range(n))
```
This function calculates the sum of the first n terms of a geometric series with ratio r. <br /> 

`r**i` calculates `r` raised to the power of `i` <br /> 
`for i in range(n)` generates a sequence of powers from `0 to n-1` <br /> 
`sum()` adds up all these terms <br /> 
```
r = 0.5
for n in [10, 100, 1000, 10000]:
    print(f"Sum of first {n} terms: {geometric_series(r, n)}")
```

**Series Representation** <br /> 
<img src="images/series_representation.png?" width="300" height="150"/>
<img src="images/series_representation_cont.png?" width="300" height="150"/>

**Complex Functions** <br /> 
<img src="images/complex_functions.png?" width="300" height="150"/>

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
<img src="images/bso_timeseries.png?" width="950" height="500"/>

### Backward Shift Operator Examples 
<img src="images/bso_example.png?" width="600" height="300"/>
<img src="images/bso_example_2.png?" width="600" height="300"/>
<img src="images/bso_example_3.png?" width="600" height="300"/>
<img src="images/bso_example_4.png?" width="600" height="300"/>
<img src="images/bso_example_5.png?" width="600" height="300"/>

## Intro to Invertibility
<img src="images/invertibility_definition.png?" width="600" height="300"/>

- **Definition**: A stochastic process is invertible if the current value can be expressed as a convergent infinite sum of past and present observations. 
- **Importance**: Invertibility ensures that we can express the process in terms of past observations, which is crucial for forecasting and interpretation. 
- **Application**: It's particularly important for MA processes. An invertible MA process can be approximated by an AR process of infinite order.
- **Condition**: For an MA(1) process Yt = εt + θεt-1, the process is invertible if |θ| < 1.

**Illustrating Inverting** <br /> 
<img src="images/inverting.png?" width="400" height="200"/>
<img src="images/inverting_2.png?" width="600" height="300"/>
<img src="images/inverting_3.png?" width="600" height="200"/>
<img src="images/inverting_4.png?" width="600" height="300"/>
<img src="images/inverting_5.png?" width="400" height="300"/>
<img src="images/inverting_6.png?" width="600" height="300"/>
<img src="images/inverting_7.png?" width="400" height="300"/>
<img src="images/inverting_8.png?" width="600" height="300"/>
<img src="images/inverting_9.png?" width="600" height="300"/>
<img src="images/inverting_10.png?" width="600" height="300"/>
<img src="images/inverting_11.png?" width="600" height="300"/>


Now, let's illustrate this concept with a Python example. We'll create both an invertible and a non-invertible MA(1) process, and demonstrate how they behave differently when we try to express them as AR processes.
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

This code does the following: <br />
- Generates an invertible MA(1) process with θ = 0.5 and a non-invertible MA(1) process with θ = 2.
- Fits AR models of increasing order to both processes.
- Plots the AR coefficients for both processes.
- Calculates the theoretical AR coefficients for the invertible process.

Interpreting the results: <br />
**Invertible MA(1):**
- The AR coefficients decay exponentially as the lag increases.
- This matches the theoretical expectation: for an MA(1) with parameter θ, the AR(∞) representation has coefficients -θ, θ², -θ³, ...
- This decay allows us to approximate the MA process with a finite-order AR process.

**Non-invertible MA(1):**
- The AR coefficients do not show a clear pattern of decay.
- They may appear erratic or grow in magnitude with increasing lag.
- This makes it impossible to approximate the process with a finite-order AR model.
<br />
**Theoretical coefficients: **
- For the invertible process, the printed theoretical coefficients should closely match the plotted coefficients from the fitted models.

The invertibility condition ensures that we can express the MA process in terms of past observations, which is crucial for forecasting and interpretation. When a process is not invertible, we lose this ability, making the process much more difficult to work with in practice. <br />
This concept extends to higher-order MA processes and ARMA processes, where invertibility conditions become more complex but equally important for time series analysis and forecasting.
## Duality

## Mean Square Convergence

