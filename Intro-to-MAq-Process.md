# Introduction to Moving Average Process
## MA(q) model 
An MA(q) model is defined as a linear combination of the current white noise term and q past white noise terms. The general form of an MA(q) model is:
<br /> 
<br /> 
$Xt = μ + εt + θ1εt-1 + θ2εt-2 + ... + θqεt-q$
<br /> 
Where
- $Xt$ is the value of the time series at time t
- $μ$ is the mean of the series
- $εt, εt-1, ..., εt-q$ are white noise error terms
- $θ1, θ2, ..., θq$ are the parameters of the model
<br />
``` 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n = 1000  # Number of time steps
θ1, θ2 = 0.6, 0.3  # MA parameters
σ = 1  # Standard deviation of white noise

# Generate MA(2) process
ma = np.array([1, θ1, θ2])
ar = np.array([1])
y = arma_generate_sample(ar=ar, ma=ma, nsample=n, scale=σ)

# Create a DataFrame
df = pd.DataFrame({'y': y}, index=pd.date_range(start='2023-01-01', periods=n))

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['y'])
plt.title('Simulated MA(2) Process')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Plot ACF and PACF
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
plot_acf(y, ax=ax1, lags=20)
plot_pacf(y, ax=ax2, lags=20)
plt.tight_layout()
plt.show()

# Calculate and print some statistics
mean = np.mean(y)
variance = np.var(y)
theoretical_variance = σ**2 * (1 + θ1**2 + θ2**2)

print(f"Empirical mean: {mean:.4f}")
print(f"Empirical variance: {variance:.4f}")
print(f"Theoretical variance: {theoretical_variance:.4f}")

# Demonstrate the MA(2) calculation for a specific point
t = 500  # Choose an arbitrary point
εt = np.random.normal(0, σ)
εt_1 = np.random.normal(0, σ)
εt_2 = np.random.normal(0, σ)
Xt = εt + θ1 * εt_1 + θ2 * εt_2

print(f"\nMA(2) calculation for t={t}:")
print(f"εt = {εt:.4f}")
print(f"εt-1 = {εt_1:.4f}")
print(f"εt-2 = {εt_2:.4f}")
print(f"Xt = εt + {θ1:.1f}*εt-1 + {θ2:.1f}*εt-2")
print(f"Xt = {εt:.4f} + {θ1:.1f}*{εt_1:.4f} + {θ2:.1f}*{εt_2:.4f} = {Xt:.4f}")
print(f"Actual value in series: {y[t]:.4f}")
``` 
<img src="images/intro_MA2/autocorrelation.png?" width="600" height="300"/>
