# Autoregressive Process

## Definition
<img src="images/build_an_ar_process.png?" width="600" height="300"/>

For an AR process of order $p$ (denoted as AR($p$)), the current value $y{t}$ is defined as: <br /> 
$yt =c+ϕ{1}y{t−1}+ϕ{2}y{t−2}+…+ϕ{p}y{t−p}+ϵ{t}$ 
where:<br /> 
- c is a constant term.
- $𝜙{1}, 𝜙{2},…,𝜙{𝑝} are the parameters of the model.
- ϵ{t} is white noise (a random error term with mean zero and constant variance).

**Example in Python**
We'll create a simple AR(1) process where the current value depends on the previous value.

```
```

## First Exmamples


## Backshift Operator and the ACF
