
# 📅 Week 5: Poisson Generalized Linear Models (GLMs) (Expanded)

---

## 🔹 Generalized Linear Models (GLMs) Overview

### Structure of a GLM
- **Three main components**:
  1. **Random Component**: Response variable $Y$ from a distribution in the exponential family (e.g., Poisson, Binomial, Normal)
  2. **Systematic Component**: Linear predictor $\eta = X\beta$
  3. **Link Function**: Function $g$ such that $g(\mu) = \eta$, where $\mu = \mathbb{E}[Y]$

### GLM Framework Summary:
| Component | Description |
|----------|-------------|
| $Y_i$ | Response variable |
| $\mu_i$ | Mean of $Y_i$, $\mu_i = \mathbb{E}[Y_i]$ |
| $g(\cdot)$ | Link function: connects $\mu_i$ to linear predictor |
| $\eta_i$ | Linear predictor = $x_i^T \beta$ |

---

## 🔹 Poisson GLM

### Distribution
- $Y_i \sim \text{Poisson}(\mu_i)$
- PMF: 
  $\mathbb{P}(Y_i = y_i) = \frac{\mu_i^{y_i} e^{-\mu_i}}{y_i!}, \quad y_i \in \{0,1,2,\ldots\}$

### Link Function
- Canonical link: $g(\mu) = \log(\mu)$
- Model becomes:
  $\log(\mu_i) = x_i^T \beta \quad \Rightarrow \quad \mu_i = \exp(x_i^T \beta)$

---

## 🔹 Log-Likelihood Function

### Poisson Log-Likelihood
Given $Y_i \sim \text{Poisson}(\mu_i = \exp(x_i^T \beta))$:
$\ell(\beta) = \sum_{i=1}^n \left[ y_i x_i^T \beta - \exp(x_i^T \beta) - \log(y_i!) \right]$

### Score and Information
- **Score Function** (gradient): $\nabla_\beta \ell(\beta)$
- **Fisher Information**:
  $\mathcal{I}(\beta) = X^T W X, \quad \text{where } W_i = \mu_i$

---

## 🔹 Fitting the Model

### Iteratively Reweighted Least Squares (IRLS)
- Used to solve $\hat{\beta}$ that maximizes $\ell(\beta)$
- Updates of the form:
  $\beta^{(t+1)} = (X^T W X)^{-1} X^T W z$
  - $W$: weights from current $\mu$
  - $z$: adjusted dependent variable (working response)

### Software
- Use `statsmodels.api.GLM` or `sklearn.linear_model.PoissonRegressor` in Python

---

## 🔹 Mean-Variance Relationship

### Poisson
- Variance = Mean: $\text{Var}(Y_i) = \mu_i$
- **Overdispersion**: When observed variance > predicted variance
  - Possible remedy: Use Negative Binomial model

---

## 🔹 Model Diagnostics

### Residuals
- **Deviance Residual**:
  $r_i = \text{sign}(y_i - \hat{\mu}_i) \cdot \sqrt{2 \left( y_i \log\left(\frac{y_i}{\hat{\mu}_i}\right) - (y_i - \hat{\mu}_i) \right)}$
- **Pearson Residual**:
  $r_i = \frac{y_i - \hat{\mu}_i}{\sqrt{\hat{\mu}_i}}$

---

## 🔹 Applications

| Scenario | Use Poisson GLM When... |
|----------|-------------------------|
| Counting disease incidence | Counts per population or time |
| Modeling web traffic | Number of visits per time unit |
| Ecology | Number of animals in a region |

---

## 🔹 Example Problems

### Problem 1: Interpret Coefficients
- If $\log(\mu_i) = \beta_0 + \beta_1 x_i$ and $\beta_1 = 0.2$, then a 1-unit increase in $x$ → expected count multiplies by $e^{0.2} \approx 1.22$ (i.e., 22% increase)

### Problem 2: Python Example
```python
import statsmodels.api as sm
import numpy as np
X = np.column_stack([np.ones(10), np.arange(10)])
y = np.random.poisson(lam=np.exp(X @ np.array([1, 0.2])))
model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
print(model.summary())
```

---

## 🔹 Summary Table

| Term | Meaning |
|------|---------|
| $\mu_i$ | Expected count for observation $i$ |
| $\log(\mu_i)$ | Linear predictor |
| $\ell(\beta)$ | Log-likelihood |
| IRLS | Algorithm to fit GLM |
| Canonical link | $\log$ for Poisson |

