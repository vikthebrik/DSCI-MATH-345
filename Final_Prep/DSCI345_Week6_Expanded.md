
# 📅 Week 6: Gamma GLMs, Overdispersion, and Model Fit (Expanded)

---

## 🔹 Gamma Distribution Overview

### Definition
- Continuous distribution for strictly positive data.
- Commonly used when the response variable is **right-skewed** and strictly positive (e.g., waiting time, insurance cost, rainfall).
- $Y \sim \text{Gamma}(\alpha, \theta)$
  - $\alpha$: shape parameter
  - $\theta$: scale parameter

### PDF
$$
f(y; \alpha, \theta) = \frac{1}{\Gamma(\alpha)\theta^\alpha} y^{\alpha - 1} e^{-y / \theta}, \quad y > 0
$$

### Properties
- Mean: $\mathbb{E}[Y] = \alpha \theta$
- Variance: $\text{Var}(Y) = \alpha \theta^2$
- $\text{Gamma}(1, \theta) = \text{Exponential}(\lambda=1/\theta)$

---

## 🔹 Gamma Generalized Linear Model (GLM)

### When to Use Gamma GLM
- Response variable is positive and continuous.
- Variance increases with the square of the mean (i.e., $\text{Var}(Y_i) \propto \mu_i^2$).

### Link Function
- Common: Log link $\Rightarrow \log(\mu_i) = x_i^T \beta \Rightarrow \mu_i = e^{x_i^T \beta}$
- Alternatives: Inverse link $g(\mu) = 1/\mu$

### Gamma GLM Structure
- Distribution: $Y_i \sim \text{Gamma}(\alpha_i, \theta_i)$
- Systematic part: $\eta_i = x_i^T \beta$
- Mean: $\mu_i = \mathbb{E}[Y_i] = g^{-1}(\eta_i)$

---

## 🔹 Log-Likelihood and Estimation

### Log-Likelihood
For log link function and $\mu_i = e^{x_i^T\beta}$, the log-likelihood becomes:
$\ell(\beta) = \sum_{i=1}^n \left[ \alpha \log(\alpha) - \log(\Gamma(\alpha)) - \alpha \log(\mu_i) - \frac{\alpha y_i}{\mu_i} \right]$

### Estimation Method
- Use **Iteratively Reweighted Least Squares (IRLS)** to maximize the log-likelihood.
- Most software libraries handle this automatically.

### Python Example
```python
import statsmodels.api as sm
from statsmodels.genmod.families import Gamma

X = sm.add_constant([[1], [2], [3], [4], [5]])
y = [0.5, 1.0, 2.0, 3.5, 5.5]
model = sm.GLM(y, X, family=Gamma(sm.families.links.log())).fit()
print(model.summary())
```

---

## 🔹 Residuals for Gamma GLMs

### Deviance Residual
$r_i = \text{sign}(y_i - \mu_i) \cdot \sqrt{2 \left[ \frac{y_i - \mu_i}{\mu_i} - \log\left(\frac{y_i}{\mu_i}\right) \right]}$

### Pearson Residual
$r_i = \frac{y_i - \mu_i}{\mu_i}$

---

## 🔹 Overdispersion in GLMs

### Definition
- Overdispersion occurs when:
  $\text{Observed Variance} > \text{Theoretical Variance}$

### Fixes
- Use **quasi-likelihood** methods
- Switch from Poisson to **Negative Binomial** or Gamma

---

## 🔹 Model Selection and AIC

### Akaike Information Criterion (AIC)
$\text{AIC} = 2k - 2\log(\hat{L})$
- Lower AIC indicates better model (penalizes model complexity)

### Compare Models
- Compare Gamma vs. Poisson vs. Gaussian for continuous skewed data
- Use cross-validation or AIC/BIC

---

## 🔹 Application Scenarios

| Context | Why Gamma GLM? |
|--------|----------------|
| Insurance Claims | Positive, right-skewed data |
| Cost Modeling | Heteroscedastic continuous response |
| Engineering Failure Times | Time until failure; skewed data |

---

## 🔹 Example Problems

1. Simulate Gamma data and fit a GLM:
```python
import numpy as np
X = sm.add_constant(np.linspace(1, 10, 100))
y = np.random.gamma(shape=2.0, scale=np.exp(X @ np.array([0.1, 0.2])) / 2.0)
model = sm.GLM(y, X, family=Gamma(sm.families.links.log())).fit()
print(model.params)
```
2. Given $Y \sim \text{Gamma}(k, \theta)$, derive $\mathbb{E}[Y]$ and $\text{Var}(Y)$.
3. Show how to check model residuals to detect misspecification.

---

## 🔹 Summary Table

| Term | Description |
|------|-------------|
| $\mu_i$ | Expected mean of $Y_i$ |
| Log link | $g(\mu_i) = \log(\mu_i)$ |
| Var($Y_i$) | $\propto \mu_i^2$ |
| Use Gamma when | Response is positive and heteroscedastic |
