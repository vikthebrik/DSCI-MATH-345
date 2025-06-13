# 📅 Week 1: Random Variables and Discrete Distributions (Expanded)

---

## 🔹 Foundational Definitions

### Random Variables (RV)
- A **Random Variable** is a function that assigns a numerical value to each outcome in a sample space.
- Types:
  - **Discrete RV**: Countable outcomes (e.g. number of heads in coin tosses).
  - **Continuous RV**: Uncountable outcomes (e.g. time, weight).

### Sample Space ($\Omega$)
- Set of all possible outcomes of an experiment.
- Example: For a die roll, $\Omega = \{1,2,3,4,5,6\}$

---

## 🔹 Discrete Probability Distributions

### Bernoulli Distribution
- **Definition**: A trial with only two outcomes: success (1) or failure (0).
- **PMF**:
  $$
  \mathbb{P}(X = x) = p^x (1 - p)^{1 - x}, \quad x \in \{0, 1\}
  $$
- **Support**: $x = 0, 1$
- **Mean**: $\mathbb{E}[X] = p$
- **Variance**: $\text{Var}(X) = p(1 - p)$

#### Example
- Tossing a fair coin: $p = 0.5$
- Success = heads

---

### Binomial Distribution
- **Definition**: Sum of $n$ independent Bernoulli trials with same success probability $p$.
- $X \sim \text{Bin}(n, p)$
- **PMF**:
  $$
  \mathbb{P}(X = k) = \binom{n}{k} p^k (1 - p)^{n - k}
  $$
- **Support**: $k = 0, 1, ..., n$
- **Mean**: $np$
- **Variance**: $np(1 - p)$

#### Example
- Tossing a coin 10 times: probability of getting 4 heads.

---

### Poisson Distribution
- **Definition**: Counts the number of events in a fixed interval (time, space), given they occur independently.
- $X \sim \text{Poisson}(\lambda)$
- **PMF**:
  $$
  \mathbb{P}(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}
  $$
- **Support**: $k = 0, 1, 2, \dots$
- **Mean** = Variance = $\lambda$

#### Example
- Number of emails you receive per hour.

---

## 🔹 Expectation and Variance (Discrete Case)

### Expected Value (Mean)
- Definition:
  $$
  \mathbb{E}[X] = \sum_{x} x \cdot \mathbb{P}(X = x)
  $$
- For functions: $\mathbb{E}[g(X)] = \sum_x g(x) \cdot \mathbb{P}(X = x)$

### Variance
- Definition:
  $$
  \text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
  $$

---

## 🔹 Moment Generating Functions (MGFs)

- **MGF** of a discrete random variable $X$:
  $$
  M_X(t) = \mathbb{E}[e^{tX}] = \sum_{x} e^{tx} \cdot \mathbb{P}(X = x)
  $$
- Properties:
  - $M_X'(0) = \mathbb{E}[X]$
  - $M_X''(0) = \mathbb{E}[X^2]$

---

## 🔹 Independence and Additivity

### Independent RVs
- $X$ and $Y$ are independent if:
  $$
  \mathbb{P}(X = x, Y = y) = \mathbb{P}(X = x) \cdot \mathbb{P}(Y = y)
  $$

### Additivity of Expectation
- Always holds:
  $$
  \mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]
  $$

### Additivity of Variance
- Only if $X$ and $Y$ are independent:
  $$
  \text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)
  $$

---

## 🔹 Properties and Applications

| Distribution | Use Case | Key Features |
|--------------|----------|--------------|
| Bernoulli($p$) | One trial, two outcomes | Basic building block of Binomial |
| Binomial($n$, $p$) | Repeated independent trials | Has fixed number of trials |
| Poisson($\lambda$) | Rare event counts | Approximates Binomial for large $n$, small $p$ |

---

## 🔹 Exercises to Practice

1. Let $X \sim \text{Bin}(5, 0.6)$. Find $\mathbb{E}[X]$, $\text{Var}(X)$, $\mathbb{P}(X \ge 3)$.
2. Suppose $Y \sim \text{Poisson}(4)$. Compute $\mathbb{P}(Y = 2)$ and $\mathbb{P}(Y \le 3)$.
3. Show that $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$ for $X \sim \text{Bernoulli}(p)$.
