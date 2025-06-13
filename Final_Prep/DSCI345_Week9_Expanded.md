
# 📅 Week 9: Inference for GLMs – Hypothesis Testing, Confidence Intervals, and Significance (Expanded)

---

## 🔹 Goals of Statistical Inference

- Assess **which predictors are significant** in explaining variation in response
- Compute **confidence intervals** for model parameters $\beta$
- Perform **hypothesis testing** on regression coefficients

---

## 🔹 Asymptotic Properties of MLEs

### Maximum Likelihood Estimator (MLE)
- Under regularity conditions, MLEs are:
  - **Consistent**: $\hat{\beta} \xrightarrow{p} \beta$
  - **Asymptotically normal**:
    $\hat{\beta} \sim \mathcal{N}\left(\beta, \mathcal{I}(\beta)^{-1}\right)$

### Fisher Information
- $\mathcal{I}(\beta)$ is the expected curvature of the log-likelihood:
  $\mathcal{I}(\beta) = -\mathbb{E}[\nabla^2 \ell(\beta)]$
- Estimated by observed information matrix from second derivative

---

## 🔹 Standard Errors and Confidence Intervals

### Variance of $\hat{\beta}$
$\text{Var}(\hat{\beta}) \approx \left( X^T W X \right)^{-1}$

### Standard Error
$\text{SE}(\hat{\beta}_j) = \sqrt{[\text{Var}(\hat{\beta})]_{jj}}$

### Confidence Interval (CI)
$\hat{\beta}_j \pm z_{\alpha/2} \cdot \text{SE}(\hat{\beta}_j)$

- Common choice: 95% CI uses $z = 1.96$

---

## 🔹 Hypothesis Testing

### Wald Test (Z-Test)
- Test $H_0: \beta_j = 0$
- Test statistic:
  $z_j = \frac{\hat{\beta}_j}{\text{SE}(\hat{\beta}_j)} \sim \mathcal{N}(0, 1)$
- p-value = $2(1 - \Phi(|z_j|))$

### Likelihood Ratio Test (LRT)
- Compare full model vs. reduced model (dropping one or more coefficients)
- Test statistic:
  $D = 2 \left[ \ell_{\text{full}} - \ell_{\text{reduced}} \right] \sim \chi^2_{df}$

### Score Test (Lagrange Multiplier Test)
- Uses derivative of likelihood at null value

---

## 🔹 Multiple Coefficient Tests

### Joint Significance Testing
- Null: $H_0: \beta_1 = \beta_2 = \dots = \beta_k = 0$
- Use LRT or F-test analog for GLMs

### Type I vs Type II Error
- Type I: Rejecting $H_0$ when it is true (false positive)
- Type II: Failing to reject $H_0$ when it is false (false negative)

---

## 🔹 p-values and Interpretation

- Small p-value ($< 0.05$): Evidence against $H_0$
- Large p-value ($> 0.05$): Fail to reject $H_0$
- Always interpret in context of model and domain

---

## 🔹 Inference in Python

```python
import statsmodels.api as sm

X = sm.add_constant([[1], [2], [3], [4], [5]])
y = [0, 1, 1, 1, 1]
model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
print(model.summary())  # includes coef, std err, z, p>|z|, [0.025, 0.975]
```

---

## 🔹 Example Interpretation

- $\hat{\beta}_1 = 0.75$, SE = 0.22 → CI = [0.32, 1.18]
- $z = 3.41$, p < 0.001 → $\beta_1$ is statistically significant
- $\exp(\hat{\beta}_1) = 2.12$: odds are multiplied by ~2.1 per unit increase in $x_1$

---

## 🔹 Summary Table

| Test | Statistic | Distribution | Use |
|------|-----------|--------------|-----|
| Wald | $z = \hat{\beta}/\text{SE}$ | $\mathcal{N}(0,1)$ | Single coefficient |
| LRT | $D = 2(\ell_{\text{full}} - \ell_{\text{reduced}})$ | $\chi^2$ | Nested model comparison |
| Score | Based on gradient | $\chi^2$ | Efficient null hypothesis check |

---

## 🔹 Exercises

1. Construct and interpret 95% confidence intervals for all coefficients in a GLM.
2. Test $H_0: \beta_1 = \beta_2 = 0$ using LRT.
3. Simulate a binomial outcome and compute significance using `statsmodels`.

