
# 📅 Week 10: Final Review – Applications, Summary, and Model Comparison (Expanded)

---

## 🔹 Review of Core Models

### Linear Model (LM)
- Response: continuous
- Form: $Y = X\beta + \epsilon$
- Assumes: Gaussian errors, constant variance
- Estimation: Ordinary Least Squares (OLS)

### Logistic Regression
- Response: binary (0 or 1)
- Link: logit — $\log \left( \frac{p}{1 - p} \right)$
- Output: predicted probabilities
- Use case: classification

### Poisson Regression
- Response: counts (integers ≥ 0)
- Link: log — $\log(\mu)$
- Assumes: mean = variance (Poisson)
- Use case: event counts (e.g. calls, claims)

### Gamma Regression
- Response: continuous, positive, right-skewed
- Link: log or inverse
- Variance increases with square of the mean
- Use case: cost, rainfall, failure time

---

## 🔹 Model Selection & Use Cases

| Model | Response Type | Link | Use Case |
|-------|---------------|------|----------|
| Linear | Continuous | Identity | Height, temperature |
| Logistic | Binary | Logit | Classification: spam, disease |
| Poisson | Count | Log | Number of calls, claims |
| Gamma | Positive continuous | Log / Inverse | Time, money, risk |

---

## 🔹 Model Comparison Techniques

### AIC and BIC
- AIC: $2k - 2 \log(\hat{L})$ (best for prediction)
- BIC: $k \log(n) - 2 \log(\hat{L})$ (best for model selection)

### Likelihood Ratio Test (LRT)
- Compare nested models
- $D = 2(\ell_{\text{full}} - \ell_{\text{reduced}}) \sim \chi^2$

### Cross-Validation (CV)
- K-fold CV: rotate test/train data
- Use prediction error as selection criterion

---

## 🔹 Diagnostic Checks

### Residuals
- Raw: $y_i - \mu_i$
- Pearson: scaled by variance
- Deviance: model-specific, compares to saturated model

### Plots
- Residual vs Fitted: checks linearity & variance
- Q-Q Plot: checks normality assumption
- ROC Curve: classification performance

### Overdispersion
- Use Pearson Chi-squared test statistic
- Use Negative Binomial or Quasi-GLM for correction

---

## 🔹 Python Model Examples (All-in-One Summary)

```python
import statsmodels.api as sm
import numpy as np

X = sm.add_constant(np.random.randn(100, 2))
y_lin = X @ np.array([1.0, 0.5, -0.2]) + np.random.randn(100)
y_log = np.random.binomial(1, 1 / (1 + np.exp(-X @ np.array([0.5, -1.0, 0.75]))))
y_pois = np.random.poisson(lam=np.exp(X @ np.array([0.2, 0.3, 0.4])))
y_gamma = np.random.gamma(shape=2.0, scale=np.exp(X @ np.array([0.1, 0.2, 0.1])) / 2.0)

# Linear Model
sm.OLS(y_lin, X).fit().summary()

# Logistic Regression
sm.GLM(y_log, X, family=sm.families.Binomial()).fit().summary()

# Poisson
sm.GLM(y_pois, X, family=sm.families.Poisson()).fit().summary()

# Gamma
sm.GLM(y_gamma, X, family=sm.families.Gamma(link=sm.families.links.log())).fit().summary()
```

---

## 🔹 Interpretation of Coefficients (Quick Review)

- **Linear**: $\beta_j$ is the change in $Y$ per unit change in $X_j$
- **Logistic**: $\exp(\beta_j)$ is the odds ratio for a unit increase in $X_j$
- **Poisson**: $\exp(\beta_j)$ is the multiplicative effect on expected count
- **Gamma**: $\exp(\beta_j)$ is the multiplicative effect on expected mean

---

## 🔹 General Strategy for Model Building

1. **Understand your outcome**: binary? count? continuous?
2. **Explore data**: histogram, boxplot, scatterplot
3. **Choose appropriate model family and link**
4. **Fit and check assumptions**
5. **Use AIC/BIC/CV to compare models**
6. **Interpret coefficients in context**

---

## 🔹 Final Checklist

✅ Know GLM structure: link, variance, response type  
✅ Be able to write likelihood and interpret coefficients  
✅ Understand how to use and compute AIC, BIC, and residuals  
✅ Understand overdispersion and what to do about it  
✅ Apply all GLMs in Python with `statsmodels`  
✅ Interpret output tables and test significance (p-values, CI)  

---

## 🔹 Practice Exercises

1. Given a dataset of insurance claim amounts, determine the best model (Poisson vs Gamma vs Linear)
2. Interpret the output of a logistic regression and calculate accuracy, precision, recall
3. Fit a Poisson model and test if adding an interaction term improves AIC
4. Use 5-fold cross-validation to select between models with different predictors

