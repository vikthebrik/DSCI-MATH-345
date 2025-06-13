
# 📅 Week 8: Logistic Regression and Bernoulli Models (Expanded)

---

## 🔹 Binary Outcome Modeling

### When to Use Logistic Regression
- Response variable $Y_i \in \{0, 1\}$ (e.g. success/failure, yes/no, win/loss)
- Models the **probability of success** as a function of predictors

---

## 🔹 Bernoulli Distribution

### PMF
$\mathbb{P}(Y = y) = p^y (1 - p)^{1 - y}, \quad y \in \{0, 1\}$

### Properties
- $\mathbb{E}[Y] = p$
- $\text{Var}(Y) = p(1 - p)$

### GLM Form
- $Y_i \sim \text{Bernoulli}(p_i)$
- Canonical link: **logit**

---

## 🔹 Logistic Regression Model

### Model
- $\log\left( \frac{p_i}{1 - p_i} \right) = x_i^T \beta$ (log-odds)
- Inverse link (sigmoid function):
  $p_i = \frac{e^{x_i^T \beta}}{1 + e^{x_i^T \beta}} = \frac{1}{1 + e^{-x_i^T \beta}}$

### Interpretation of Coefficients
- A one-unit increase in $x_j$ leads to a change in the log-odds of outcome by $\beta_j$
- Odds ratio: $\exp(\beta_j)$ = multiplicative change in odds per unit increase in $x_j$

---

## 🔹 Log-Likelihood

### Bernoulli Log-Likelihood
$\ell(\beta) = \sum_{i=1}^n \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$

- Concave function: guarantees a global maximum
- Solved using gradient-based optimization (e.g., Newton-Raphson or IRLS)

---

## 🔹 Estimation via IRLS

### Iteratively Reweighted Least Squares (IRLS)
- At each step:
  - Compute $p_i = \frac{1}{1 + e^{-x_i^T \beta}}$
  - Construct weights $W_i = p_i (1 - p_i)$
  - Update:
    $\beta^{(t+1)} = (X^T W X)^{-1} X^T W z$
    where $z$ is the adjusted dependent variable

---

## 🔹 Classification Thresholds

- Default threshold: $0.5$
- ROC curve: plot of TPR vs. FPR at various thresholds
- AUC (Area Under Curve): Measures overall classifier quality

### Confusion Matrix Terms
| Term | Meaning |
|------|---------|
| TP | True Positive |
| TN | True Negative |
| FP | False Positive |
| FN | False Negative |

### Performance Metrics
- Accuracy: $\frac{TP + TN}{TP + TN + FP + FN}$
- Precision: $\frac{TP}{TP + FP}$
- Recall (Sensitivity): $\frac{TP}{TP + FN}$
- F1 Score: Harmonic mean of precision and recall

---

## 🔹 Model Diagnostics

### Deviance Residuals
- Used to assess fit for logistic models
- Lack-of-fit if residuals are large or asymmetrically distributed

### Hosmer-Lemeshow Test
- Compare predicted probabilities to observed outcomes grouped by decile

### Multicollinearity
- Check Variance Inflation Factors (VIFs)
- Drop or combine highly collinear variables

---

## 🔹 Python Example

```python
import statsmodels.api as sm
import numpy as np

X = sm.add_constant(np.random.randn(100, 2))
beta = np.array([0.5, -1.0, 0.75])
linpred = X @ beta
p = 1 / (1 + np.exp(-linpred))
y = np.random.binomial(1, p)
model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
print(model.summary())
```

---

## 🔹 Applications of Logistic Regression

| Application | Use Case |
|-------------|----------|
| Medical Diagnosis | Disease vs. No Disease |
| Email Filtering | Spam vs. Not Spam |
| Credit Risk | Default vs. No Default |
| Marketing | Response vs. No Response |

---

## 🔹 Summary Table

| Concept | Formula / Meaning |
|--------|--------------------|
| Link Function | $\log \left( \frac{p}{1 - p} \right)$ |
| Inverse Link | $p = \frac{e^\eta}{1 + e^\eta}$ |
| Log-likelihood | $\ell(\beta) = \sum [y_i \log(p_i) + (1-y_i)\log(1-p_i)]$ |
| IRLS Update | $\beta^{(t+1)} = (X^T W X)^{-1} X^T W z$ |
| Coeff. Interpretation | $\exp(\beta_j)$ = odds ratio |

---

## 🔹 Example Exercises

1. Interpret coefficients from a logistic regression model summary.
2. Given predicted probabilities, compute accuracy and F1 score for classification.
3. Use ROC curve to evaluate two classifiers.

