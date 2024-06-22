# Random Walk
## What is Random Walk?
A random walk is a mmathematical concept that describes a path consisting of a succession of random steps.
<br /> In simplest form, a random walk is the process where, at each step, the position moves either up or down with equal probability. 
This concept is widely used in various fields, including physics, biology, economics, and finance.
<br /> 
<br /> 
In the context of time series, a random walk is a process where the current value is composed of the past value + an unpredictable change.
<br /> Mathemetically, this is expressed as: <br /> 
<br /> 
$X_t = X_{t-1} + e_t$
<br /> 
<br /> 
Where:<br /> 
- $X_t$ is the value at time t
- $X_{t-1}$ is the value at the previous time step
- $e_t$ is a random variable (often normally distributed) representing a random change
<br /> 

```
import numpy as np
import matplotlib.pyplot as plt

def generate_random_walk(steps, start=0, step_size=1):
    # Generate random steps (either +1 or -1)
    random_steps = np.random.choice([-1, 1], size=steps)
    
    # Multiply by step size
    path = step_size * np.cumsum(random_steps)
    
    # Add the starting point
    return start + np.insert(path, 0, 0)

# Parameters
num_steps = 1000
start_point = 0
step_size = 0.1

# Generate random walk
random_walk = generate_random_walk(num_steps, start_point, step_size)

# Create time array
time = np.arange(num_steps + 1)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(time, random_walk)
plt.title('Random Walk')
plt.xlabel('Time Steps')
plt.ylabel('Position')
plt.grid(True)
plt.show()

# Print some statistics
print(f"Start point: {random_walk[0]}")
print(f"End point: {random_walk[-1]}")
print(f"Lowest point: {np.min(random_walk)}")
print(f"Highest point: {np.max(random_walk)}")
```
<img src="images/random_walk.png?" width="1000" height="400"/>

