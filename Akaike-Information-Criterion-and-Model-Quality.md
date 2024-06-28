# Akaike Information Criterion (AIC) and Model Quality

## What is the Akaike Information Criterion (AIC)?
AIC is a measure of the relative quality of statistical models for a given set of data. It estimates the relative amount of information lost by a given model: the less information a model loses, the higher the quality of that model. <br /> 

## How is AIC used in time series analysis?
In time series analysis, AIC is primarily used for:  <br /> 
  a) Selecting the optimal order of ARIMA (Autoregressive Integrated Moving Average) models <br /> 
  b) Comparing different models (e.g., AR vs MA vs ARMA) <br /> 
  c) Balancing model complexity with goodness of fit <br /> 
The model with the lowest AIC is generally preferred.  <br /> 

## Formula
The general formula for AIC is: <br /> 
$AIC = 2k - 2ln(L)$
Where: <br /> 
- $k$ is the number of parameters in the model
- $L$ is the maximum value of the likelihood function for the model
<br /> 
For linear regression models (including many time series models), AIC is often calculated as: <br />
<br /> $AIC = 2k + n * ln(RSS/n)$
<br />
Where:
- n is the number of observations
- RSS is the residual sum of squares
<br />

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import arma_generate_sample

# Generate a sample ARIMA process
np.random.seed(42)
ar = np.array([1, -0.6, 0.2])
ma = np.array([1, 0.3])
y = arma_generate_sample(ar, ma, 1000)

# Create a DataFrame
df = pd.DataFrame(y, columns=['value'])

# Function to evaluate ARIMA models
def evaluate_arima_model(data, order):
    model = ARIMA(data, order=order)
    results = model.fit()
    return results.aic

# Grid search ARIMA parameters
p_values = range(0, 3)
d_values = range(0, 2)
q_values = range(0, 3)

best_aic = float("inf")
best_order = None

for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p,d,q)
            try:
                aic = evaluate_arima_model(df['value'], order)
                if aic < best_aic:
                    best_aic = aic
                    best_order = order
                print(f'ARIMA{order} AIC={aic}')
            except:
                continue

print(f'Best ARIMA{best_order} AIC={best_aic}')

# Fit the best model
best_model = ARIMA(df['value'], order=best_order)
best_results = best_model.fit()

# Plot the data and the fit
plt.figure(figsize=(12,6))
plt.plot(df.index, df['value'], label='Original')
plt.plot(df.index, best_results.fittedvalues, color='red', label='Fitted')
plt.legend()
plt.title(f'Best ARIMA Model: {best_order}')
plt.show()

# Print model summary
print(best_results.summary())
```
output: <br /> 
<img src="images/aic_print.png?" width="300" height="300"/>
<img src="images/aic_results.png?" width="600" height="300"/>
<img src="images/aic_sarimax_results.png?" width="500" height="300"/>

