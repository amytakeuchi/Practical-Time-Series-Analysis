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
