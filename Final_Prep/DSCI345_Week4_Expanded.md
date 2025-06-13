
# 📅 Week 4: Linear Models and Least Squares Estimation (Expanded)

---

## 🔹 Linear Models Overview

### Basic Form
- A **linear model** expresses a response variable $Y$ as a linear function of predictors $X$:
  $$
  Y = X\beta + \epsilon
  $$
  - $Y$: $n \times 1$ vector of responses
  - $X$: $n \times p$ design matrix (each row is an observation)
  - $\beta$: $p \times 1$ vector of coefficients
  - $\epsilon$: $n \times 1$ vector of random errors

### Assumptions
1. **Linearity**: Relationship between $Y$ and $X$ is linear in parameters
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of residuals ($\text{Var}(\epsilon_i) = \sigma^2$)
4. **Normality**: $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ (needed for inference)

---

## 🔹 Least Squares Estimation

### Goal
- Estimate $\beta$ by minimizing residual sum of squares (RSS):
  $$
  \text{RSS} = \|Y - X\beta\|^2 = (Y - X\beta)^T(Y - X\beta)
  $$

### Solution
- Take derivative and solve:
  $$
  \hat{\beta} = (X^TX)^{-1}X^TY
  $$

#### With Intercept
- Add a column of 1s to $X$ for intercept term.
  $X_{\text{aug}} = [\mathbf{1}_n \mid X]$

---

## 🔹 Geometry of Least Squares

### Projection
- $\hat{Y} = X\hat{\beta}$ is the projection of $Y$ onto the column space of $X$
- Residuals: $r = Y - \hat{Y}$ orthogonal to $X$: $X^T r = 0$

### Hat Matrix
- $H = X(X^TX)^{-1}X^T$ so that $\hat{Y} = HY$
- Properties: $H = H^T = H^2$ (symmetric and idempotent)

---

## 🔹 Residuals and Model Fit

### Residuals
- $r_i = Y_i - \hat{Y}_i$
- Properties: $\sum r_i = 0$ (if intercept included)

### R-squared ($R^2$)
- Proportion of variance explained by the model:
  $$
  R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = \frac{\text{Explained Variance}}{\text{Total Variance}}
  $$

---

## 🔹 Inference for $\hat{\beta}$

### Sampling Distribution
- If errors $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$, then:
  $$
  \hat{\beta} \sim \mathcal{N}(\beta, \sigma^2 (X^TX)^{-1})
  $$

### Standard Error and Confidence Interval
- $\text{SE}(\hat{\beta}_j) = \sqrt{\hat{\sigma}^2 [(X^TX)^{-1}]_{jj}}$
- CI: $\hat{\beta}_j \pm z_{\alpha/2} \cdot \text{SE}(\hat{\beta}_j)$

### Hypothesis Test
- $H_0: \beta_j = 0$
- Test statistic: $t = \hat{\beta}_j / \text{SE}(\hat{\beta}_j)$

---

## 🔹 Multiple Linear Regression

### Extension
- Multiple predictors $X_1, ..., X_p$
- Model: $Y = \beta_0 + \beta_1 X_1 + \dots + \beta_p X_p + \epsilon$

### Interpretation
- $\beta_j$ measures effect of $X_j$ holding others constant

---

## 🔹 Model Diagnostics

### Residual Plots
- Plot residuals vs. fitted values to check:
  - Non-linearity
  - Unequal variance (heteroscedasticity)
  - Outliers

### Q-Q Plot
- Checks normality assumption of residuals

---

## 🔹 Example Problems

1. Given $X = \begin{bmatrix}1 & 1 \\ 1 & 2 \\ 1 & 3\end{bmatrix}$ and $Y = \begin{bmatrix}2 \\ 2.5 \\ 3.5\end{bmatrix}$, compute $\hat{\beta}$.
2. Show residuals are orthogonal to predictors.
3. Simulate data and fit model in Python:
```python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3]])
y = np.array([2, 2.5, 3.5])
model = LinearRegression().fit(X, y)
print(model.intercept_, model.coef_)
```

---

## 🔹 Summary Table

| Concept | Formula / Description |
|--------|------------------------|
| Least Squares Estimator | $\hat{\beta} = (X^TX)^{-1}X^TY$ |
| Fitted values | $\hat{Y} = X\hat{\beta}$ |
| Residuals | $r = Y - \hat{Y}$ |
| Hat Matrix | $H = X(X^TX)^{-1}X^T$ |
| R-squared | $1 - \frac{\text{RSS}}{\text{TSS}}$ |
| Var($\hat{\beta}$) | $\sigma^2 (X^TX)^{-1}$ |

