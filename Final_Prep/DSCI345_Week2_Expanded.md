
# 📅 Week 2: Poisson, Exponential, and Gamma Distributions (Expanded)

---

## 🔹 Poisson Distribution – In Depth

### Definition
- Models the number of events occurring in a fixed interval of time or space.
- Events occur independently, and the average rate is constant.
- $X \sim \text{Poisson}(\lambda)$

### Probability Mass Function (PMF)
$$
\mathbb{P}(X = k) = \frac{e^{-\lambda} \lambda^k}{k!}, \quad k = 0, 1, 2, ...
$$

### Key Properties
- Mean: $\mathbb{E}[X] = \lambda$
- Variance: $\text{Var}(X) = \lambda$
- Additive: If $X \sim \text{Poisson}(\lambda_1)$ and $Y \sim \text{Poisson}(\lambda_2)$ independently, then:
  $X + Y \sim \text{Poisson}(\lambda_1 + \lambda_2)$

### Use Cases
- Number of emails per hour.
- Number of calls arriving at a call center.

---

## 🔹 Exponential Distribution – In Depth

### Definition
- Models time between successive events in a Poisson process.
- Memoryless: $\mathbb{P}(X > s + t \mid X > s) = \mathbb{P}(X > t)$

### Probability Density Function (PDF)
$f(x) = \lambda e^{-\lambda x}, \quad x \ge 0$

### Key Properties
- Mean: $\mathbb{E}[X] = \frac{1}{\lambda}$
- Variance: $\text{Var}(X) = \frac{1}{\lambda^2}$
- MGF: $M_X(t) = \frac{\lambda}{\lambda - t}$ for $t < \lambda$

---

## 🔹 Gamma Distribution – In Depth

### Definition
- A generalization of the exponential distribution.
- Sum of $k$ independent exponential RVs with rate $\lambda = 1/\theta$.
- $X \sim \text{Gamma}(k, \theta)$ (shape = $k$, scale = $\theta$)

### PDF
$f(x) = \frac{1}{\Gamma(k)\theta^k} x^{k - 1} e^{-x/\theta}, \quad x \ge 0$

### Properties
- Mean: $\mathbb{E}[X] = k\theta$
- Variance: $\text{Var}(X) = k\theta^2$
- $\Gamma(n) = (n - 1)!$ when $n$ is an integer

---

## 🔹 Relationships Between Distributions

- If $X \sim \text{Exp}(\lambda)$ then $X \sim \text{Gamma}(1, 1/\lambda)$
- If $X_1, ..., X_k \overset{iid}{\sim} \text{Exp}(\lambda)$, then:
  $\sum_{i=1}^k X_i \sim \text{Gamma}(k, 1/\lambda)$

---

## 🔹 Practical Applications

| Scenario | Best Model | Reason |
|----------|------------|--------|
| Time between arrivals | Exponential | Memoryless time between events |
| Time until $k$th event | Gamma | Sum of $k$ waiting times |
| Count of events in interval | Poisson | Rate $\lambda$ over fixed time |

---

## 🔹 Example Problems

1. If $X \sim \text{Poisson}(3)$, compute $\mathbb{P}(X = 2)$ and $\mathbb{P}(X \ge 4)$.
2. Let $T \sim \text{Exp}(0.5)$, find $\mathbb{P}(T < 2)$.
3. Simulate a Gamma RV as the sum of 4 exponential RVs in Python:
```python
import numpy as np
np.sum(np.random.exponential(scale=2.0, size=4))
```
4. Prove memorylessness: $\mathbb{P}(X > s + t \mid X > s) = \mathbb{P}(X > t)$

