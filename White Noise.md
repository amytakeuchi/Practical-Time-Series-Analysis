# White Noise
## What is White Noise?
White noise is a random signal that has equal intensity at different frequencies, giving it a constant power spectral density. In simpler terms, it's a sequence of: <br /> 
- uncorrelated random variables with
- zero mean and
- constant variance.

### Formula for White Noise
The mathematical model for white noise can be expressed as: <br /> 
<br /> 
      $X_t = ε_t$
<br /> 
<br /> 
Where:
- $X_t$ is the white noise process at time t
- $ε_t$ is a sequence of independent and identically distributed (i.i.d.) random variables with:
  - $E[ε_t]$ = 0 (zero mean)
  - $Var(ε_t)$ = σ² (constant variance)
  - $Cov(ε_t, ε_s)$ = 0 for t ≠ s (uncorrelated)
```
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Set random seed for reproducibility
np.random.seed(42)

# Generate white noise
n = 1000  # Number of observations
mean = 0
std_dev = 1
white_noise = np.random.normal(mean, std_dev, n)

# Plot the white noise series
plt.figure(figsize=(12, 6))
plt.plot(white_noise)
plt.title('White Noise Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Calculate and print statistics
print(f"Mean: {np.mean(white_noise):.4f}")
print(f"Standard Deviation: {np.std(white_noise):.4f}")
print(f"Variance: {np.var(white_noise):.4f}")
```
