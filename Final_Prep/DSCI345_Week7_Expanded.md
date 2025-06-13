
# 📅 Week 7: Model Evaluation – AIC, BIC, Residuals, and Diagnostics (Expanded)

---

## 🔹 Model Fit and Deviance

### Deviance
- Deviance measures the goodness-of-fit for GLMs by comparing the fitted model to a saturated model (perfect fit).
- Formula:
  $$
  D = 2 \left[ \ell(\text{saturated model}) - \ell(\text{fitted model}) \right]
  $$

### Null Deviance vs Residual Deviance
- **Null Deviance**: Deviance from a model with only the intercept (no predictors).
- **Residual Deviance**: Deviance from the model with all predictors.
- Lower residual deviance means better fit.

---

## 🔹 Information Criteria

### AIC – Akaike Information Criterion
$$
\text{AIC} = 2k - 2 \log(\hat{L})
$$
- $k$: number of parameters in the model
- $\hat{L}$: maximized value of the likelihood function

### BIC – Bayesian Information Criterion
$$
\text{BIC} = k \log(n) - 2 \log(\hat{L})
$$
- $n$: number of observations
- BIC penalizes complex models more heavily than AIC

### Use Cases
- **AIC** is better for prediction accuracy.
- **BIC** is better for model selection under a true model framework.

---

## 🔹 Residuals in GLMs

### Types of Residuals

| Type | Formula | Interpretation |
|------|---------|----------------|
| Raw | $y_i - \hat{\mu}_i$ | Basic residual |
| Pearson | $\frac{y_i - \hat{\mu}_i}{\sqrt{V(\hat{\mu}_i)}}$ | Standardized residual |
| Deviance | Based on likelihood | Measures model fit |
| Anscombe | Uses transformation | More symmetry in skewed models |

### Plotting Residuals
- Plot residuals vs. fitted values to check:
  - Heteroscedasticity
  - Nonlinearity
  - Outliers

---

## 🔹 Model Diagnostics

### Key Plots
- **Residuals vs Fitted**: Should show no pattern (checks linearity, equal variance)
- **QQ Plot**: Should be approximately linear if residuals are normal
- **Scale-Location Plot**: Spread of residuals should be constant
- **Leverage Plot**: Identifies influential observations

### Leverage and Cook’s Distance
- Leverage measures influence of a data point’s $x_i$ on fitted $\hat{y}_i$
- Cook’s Distance combines leverage and residuals:
  $$
  D_i = \frac{(r_i^2 h_{ii})}{p \cdot MSE (1 - h_{ii})^2}
  $$

---

## 🔹 Overdispersion Detection

### Pearson Chi-Square Statistic
- Estimate overdispersion factor:
  $$
  \hat{\phi} = \frac{1}{n - p} \sum_{i=1}^n \left( \frac{y_i - \mu_i}{\sqrt{V(\mu_i)}} \right)^2
  $$
- $\hat{\phi} > 1$ suggests overdispersion

### Consequences of Ignoring Overdispersion
- Underestimated standard errors → misleading inference
- Poor model fit and invalid prediction intervals

---

## 🔹 Comparing Nested Models

### Likelihood Ratio Test
- Compare two nested models:
  $$
  D = 2(\ell_{\text{full}} - \ell_{\text{reduced}})
  $$
- $D \sim \chi^2_{df}$ under the null hypothesis
- Degrees of freedom = difference in number of parameters

### Example
```python
from statsmodels.api import GLM, families

model1 = GLM(y, X1, family=families.Poisson()).fit()
model2 = GLM(y, X2, family=families.Poisson()).fit()
lrt_stat = 2 * (model2.llf - model1.llf)
```

---

## 🔹 Cross-Validation

### K-Fold CV
- Partition data into $k$ subsets, train on $k-1$ and test on the 1 left out
- Rotate and average test errors

### Use in GLMs
- Evaluate out-of-sample performance
- Useful when comparing models with similar AIC/BIC

---

## 🔹 Summary Table

| Metric | Formula | Usage |
|--------|---------|-------|
| AIC | $2k - 2\log \hat{L}$ | Prediction-focused model comparison |
| BIC | $k\log(n) - 2\log \hat{L}$ | Penalizes complexity more, better for true model |
| Deviance | $2(\ell_{sat} - \ell_{fit})$ | Goodness-of-fit |
| Pearson $\chi^2$ | $\sum \left(\frac{y - \mu}{\sqrt{V(\mu)}}\right)^2$ | Detect overdispersion |
| Cook’s D | Measures influence | Outlier + leverage detection |

---

## 🔹 Example Exercises

1. Plot and interpret residuals from a Poisson GLM.
2. Compute AIC/BIC for multiple models and interpret which to prefer.
3. Simulate overdispersed data and fit Poisson and quasi-Poisson models.

