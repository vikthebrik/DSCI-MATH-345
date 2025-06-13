
# 📅 Week 3: Conditional Expectation, Variance, and Joint Distributions (Expanded)

---

## 🔹 Conditional Probability

### Definition
- The **conditional probability** of $A$ given $B$:
  $\mathbb{P}(A \mid B) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(B)}$

### Conditional PMF/PDF
- For discrete RVs:
  $\mathbb{P}(X = x \mid Y = y) = \frac{\mathbb{P}(X = x, Y = y)}{\mathbb{P}(Y = y)}$
- For continuous RVs:
  $f_{X|Y}(x \mid y) = \frac{f_{X,Y}(x, y)}{f_Y(y)}$

---

## 🔹 Joint Distributions

### Joint PMF/PDF
- Discrete: $\mathbb{P}(X = x, Y = y)$
- Continuous: $f_{X,Y}(x, y)$

### Marginal Distributions
- Discrete: $\mathbb{P}(X = x) = \sum_y \mathbb{P}(X = x, Y = y)$
- Continuous: $f_X(x) = \int f_{X,Y}(x, y) dy$

### Independence
- $X$ and $Y$ are independent if:
  - $\mathbb{P}(X = x, Y = y) = \mathbb{P}(X = x)\mathbb{P}(Y = y)$
  - $f_{X,Y}(x, y) = f_X(x)f_Y(y)$

---

## 🔹 Conditional Expectation

### Definition
- The expectation of $X$ given $Y = y$:
  - Discrete: $\mathbb{E}[X \mid Y = y] = \sum_x x \cdot \mathbb{P}(X = x \mid Y = y)$
  - Continuous: $\mathbb{E}[X \mid Y = y] = \int x f_{X|Y}(x \mid y) dx$

### Properties
- $\mathbb{E}[\mathbb{E}[X \mid Y]] = \mathbb{E}[X]$ (Law of Iterated Expectations)
- If $X$ and $Y$ are independent, $\mathbb{E}[X \mid Y] = \mathbb{E}[X]$

---

## 🔹 Conditional Variance

### Definition
- $\text{Var}(X \mid Y = y) = \mathbb{E}[(X - \mathbb{E}[X \mid Y = y])^2 \mid Y = y]$

### Law of Total Variance
$\text{Var}(X) = \mathbb{E}[\text{Var}(X \mid Y)] + \text{Var}(\mathbb{E}[X \mid Y])$

---

## 🔹 Examples

### Example 1 – Conditional Expectation from Table
Given joint PMF $\mathbb{P}(X = x, Y = y)$ in a table, compute:
- $\mathbb{P}(Y = y)$ (marginal)
- $\mathbb{P}(X = x \mid Y = y)$ (conditional)
- $\mathbb{E}[X \mid Y = y]$

### Example 2 – Iterated Expectation
Let $X \mid Y = y \sim \text{Poisson}(y)$ and $Y \sim \text{Exp}(1)$
- Compute $\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X \mid Y]]$

---

## 🔹 Conditional Distributions and Sums

### Sum of Two RVs
- Expectation: $\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$
- Variance (if independent): $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

### Convolution for PMFs
- For $Z = X + Y$, $P_Z(z) = \sum_x P_X(x)P_Y(z - x)$

---

## 🔹 Practical Applications

| Concept | Use Case |
|--------|----------|
| $\mathbb{E}[X \mid Y]$ | Prediction when info about $Y$ is known |
| Law of Iterated Expectations | Decomposing hierarchical models |
| Conditional Variance | Uncertainty given known values |
| Law of Total Variance | Explaining spread in nested models |

---

## 🔹 Exercises

1. A fair die is rolled. Let $X$ = roll outcome, $Y$ = 1 if $X$ even, 0 if odd. Compute $\mathbb{E}[X \mid Y]$.
2. Let $X \mid Y=y \sim \text{Bern}(y)$, $Y \sim \text{Beta}(2, 2)$. Compute $\mathbb{E}[X]$.
3. Given joint PDF $f(x,y) = 2$ on $0 < x < y < 1$, find marginal and conditional distributions.
