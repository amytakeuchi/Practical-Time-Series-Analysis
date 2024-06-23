# Autoregressive Process

## Definition
<img src="images/build_an_ar_process.png?" width="600" height="300"/>

For an AR process of order $p$ (denoted as AR($p$)), the current value $y{t}$ is defined as: <br /> 
$yt =c+ϕ{1}y{t−1}+ϕ{2}y{t−2}+…+ϕ{p}y{t−p}+ϵ{t}$ 
<br /> 
where:<br /> 
- c is a constant term.
- $𝜙{1}, 𝜙{2},…,𝜙{𝑝} are the parameters of the model.
- ϵ{t} is white noise (a random error term with mean zero and constant variance).

**Example in Python**
We'll create a simple AR(1) process where the current value depends on the previous value.

```
import numpy as np
import matplotlib.pyplot as plt

# Parameters for the AR(1) process
phi = 0.8
c = 0
n = 100  # Number of observations

# Generate white noise
np.random.seed(42)  # For reproducibility
epsilon = np.random.normal(0, 1, n)

# Initialize the time series
y = np.zeros(n)

# Generate the AR(1) process
for t in range(1, n):
    y[t] = c + phi * y[t-1] + epsilon[t]

# Plot the AR(1) process
plt.figure(figsize=(10, 4))
plt.plot(y, label='AR(1) Process')
plt.title('Simulated AR(1) Process')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
```
<img src="images/simulated_ar1_process.png?" width="500" height="200"/>

## First Exmamples
<img src="images/ar_first_example.png?" width="600" height="300"/>


## Backshift Operator and the ACF
