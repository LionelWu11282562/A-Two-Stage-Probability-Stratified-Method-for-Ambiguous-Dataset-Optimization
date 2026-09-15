# PU-Boost

### A Two-Stage Reliable-Sample Reconstruction Positive-Unlabeled Learning Framework for Screening Undiagnosed Hypertension

PU-Boost is a two-stage positive-unlabeled (PU) learning framework designed to reconstruct reliable supervision from ambiguous clinical labels.

Instead of treating all individuals without a prior hypertension diagnosis as negative, PU-Boost explicitly models them as an **unlabeled mixture of hidden positives and true negatives**, progressively extracts high-confidence samples, rejects persistently ambiguous observations, and trains the final classifier on the reconstructed dataset.

---

## Screening-Oriented Performance

<p align="center">
  <a href="figures/figure7_performance.png">
    <img src="figures/figure7_performance.png" width="900">
  </a>
</p>

<p align="center">
  <b>Test-set error profiles and sensitivity–specificity operating points.</b>
</p>

On the independent test set (n = 8,611), PU-Boost achieved the strongest screening-oriented performance among the compared models.

| Model | Sensitivity | Specificity | Balanced Accuracy | False Negatives |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6832 | 0.6417 | 0.6624 | 1,263 |
| Standard XGBoost | 0.6897 | 0.6339 | 0.6618 | 1,237 |
| Neural Network | 0.7186 | 0.5577 | 0.6382 | 1,122 |
| **PU-Boost** | **0.8172** | 0.5474 | **0.6823** | **729** |

### Key gains

Compared with standard XGBoost:

- **Sensitivity:** 0.6897 → **0.8172**
- **Absolute sensitivity gain:** **+0.1275**
- **False negatives:** 1,237 → **729**
- **Missed cases reduced by 41.1%**
- **508 additional reference-positive participants identified**
- **Balanced accuracy:** 0.6618 → **0.6823**

The gain is intentionally screening-oriented: PU-Boost substantially reduces missed positive cases while accepting a lower specificity.

---

# Mathematical Formulation

## 1. Positive-Unlabeled Learning Setting

Let:

- `Y ∈ {0,1}` denote the reference hypertension status
- `S ∈ {0,1}` denote the observable prior-diagnosis label
- `X` denote the predictor vector

The clinically relevant target is

$$
P(Y=1 \mid X),
$$

whereas directly learning from diagnosis status targets

$$
P(S=1 \mid X).
$$

Under the reliable-positive assumption,

$$
S=1 \Rightarrow Y=1,
$$

and

$$
P(S=1\mid X)
=
P(Y=1\mid X)
P(S=1\mid Y=1,X).
$$

Defining

$$
c(X)=P(S=1\mid Y=1,X),
$$

gives

$$
P(S=1\mid X)=c(X)P(Y=1\mid X).
$$

Therefore, learning the observable diagnosis label is not necessarily equivalent to learning the underlying disease risk.

In this study:

$$
P=A,
$$

where `A` represents previously diagnosed hypertension cases, while

$$
U=B\cup C,
$$

where:

- `B`: undiagnosed hypertension
- `C`: non-hypertension

The identities of `B` and `C` are hidden from the two-stage PU reconstruction algorithm.

---

# Two-Stage Reliable-Sample Reconstruction

## Stage 1 — Global Ranking with Elastic Net

Elastic Net logistic regression is fitted to the observable P/U labels.

For participant `i`, the Stage-1 score is

$$
q_i
=
P(S_i=1\mid X_i)
=
\frac{1}
{1+\exp[-(\beta_0+X_i^\top\beta)]}.
$$

Parameters are estimated through the penalized objective

$$
\min_{\beta_0,\beta}
\left\{
-\frac{1}{n}
\sum_{i=1}^{n}
\left[
S_i\log q_i
+
(1-S_i)\log(1-q_i)
\right]
+
\lambda
\left[
\alpha\|\beta\|_1
+
\frac{1-\alpha}{2}\|\beta\|_2^2
\right]
\right\}.
$$

Two training-internal thresholds divide the data into:

$$
\widetilde{Z}_i^{(1)}
=
\begin{cases}
P_1, & q_i \ge t_H^{(1)},\\
N_1, & q_i \le t_L^{(1)},\\
U_1, & t_L^{(1)} < q_i < t_H^{(1)}.
\end{cases}
$$

where:

- `P₁`: high-confidence candidate positives
- `N₁`: high-confidence candidate negatives
- `U₁`: uncertain observations

Observed Stage-1 partition:

| Set | n |
|---|---:|
| P₁ | 5,313 |
| N₁ | 2,144 |
| U₁ | 18,426 |

---

## Stage 2 — Nonlinear Refinement with Random Forest

Only the uncertain set `U₁` enters Stage 2.

A Random Forest is trained using the Stage-1 high-confidence samples:

$$
D_1=P_1\cup N_1.
$$

For observations in `U₁`, the second-stage score is

$$
r_i=f_2(X_i).
$$

The uncertain population is then partitioned as

$$
\widetilde{Z}_i^{(2)}
=
\begin{cases}
P_2, & r_i \ge t_H^{(2)},\\
N_2, & r_i \le t_L^{(2)},\\
R, & t_L^{(2)} < r_i < t_H^{(2)}.
\end{cases}
$$

where `R` is a **reject set** containing observations that remain highly ambiguous after both stages.

Observed Stage-2 partition:

| Set | n |
|---|---:|
| P₂ | 2,704 |
| N₂ | 5,873 |
| Reject set R | 9,849 |

---

# Final Reconstructed Training Set

The retained positive and negative candidates are

$$
D_P=P_1\cup P_2
$$

and

$$
D_N=N_1\cup N_2.
$$

The final pseudo-label is

$$
\widetilde{Y}_i=
\begin{cases}
1, & i\in D_P,\\
0, & i\in D_N.
\end{cases}
$$

This yields:

- **8,017 pseudo-positive observations**
- **8,017 pseudo-negative observations**
- **16,034 observations retained**
- **9,849 highly ambiguous observations rejected**

Thus, the final training sample is deliberately class-balanced.

The reconstructed sample is then used to train the final **XGBoost classifier**.

---

# Why Two Stages?

The architecture separates three different statistical roles:

### 1. Global regularized ranking

Elastic Net provides a stable linear ranking under noisy P/U supervision.

### 2. Local nonlinear refinement

Random Forest focuses specifically on the difficult region where linear separation is insufficient.

### 3. Uncertainty rejection

Observations that remain ambiguous after both stages are not forced into potentially incorrect pseudo-labels.

The procedure can therefore be summarized as:

$$
\text{Global ranking}
\rightarrow
\text{Uncertain-region refinement}
\rightarrow
\text{Rejection}
\rightarrow
\text{Final supervised learning}.
$$

Rather than assigning labels to every unlabeled observation, PU-Boost trades training-set size for higher-confidence supervision.

---

# Model Interpretation

SHAP analysis showed that the final model relied strongly on clinically interpretable hypertension-related predictors.

Leading contributors included:

- Age
- Body mass index
- Waist circumference
- Maternal hypertension history
- Paternal hypertension history
- Snoring
- Dyslipidemia
- Stroke history
- Smoking
- Diabetes

These feature attributions are used for model interpretation and should not be interpreted as causal effects.

---
