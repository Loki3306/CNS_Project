# Research Methodology: Gradient Boosting Tree Framework for Intelligent Phishing URL Classification

## 1. Problem Formulation & Academic Scope

Phishing constitutes one of the most persistent and damaging attack vectors in computer networks, relying on social engineering to deceive users into disclosing sensitive credentials or authorization tokens. While lexical parsing of raw URL strings (e.g., character-level tokenization or sequential embeddings) provides immediate inference on raw inputs, real-world high-throughput detection pipelines frequently rely on **engineered URL, domain, and webpage features**.

### Critical Scope Distinction
This research specifically investigates classification over **engineered phishing-related website, URL, and host characteristics**, rather than raw unparsed URL strings. The model takes a 30-dimensional discrete vector of extracted lexical, host, and behavioral indicators and maps it to a binary security state:
$$\mathcal{X} \in \{-1, 0, 1\}^{30} \longrightarrow \mathcal{Y} \in \{0, 1\}$$
where $0$ denotes a **Legitimate** domain and $1$ denotes a **Phishing** entity.

---

## 2. Dataset Representation & Structural Reality

The primary benchmark dataset (`eswarchandt/phishing-website-detector`) originates from the classic UCI Phishing Websites collection. The dataset comprises 11,054 instances across 30 domain-specific attributes:
- **Address Bar Features**: IP address usage (`UsingIP`), excessive URL length (`LongURL`), shortening services (`ShortURL`), '@' symbol presence (`Symbol@`), double slash redirection (`Redirecting//`), hyphen prefix/suffix (`PrefixSuffix-`), sub-domain count (`SubDomains`), and SSL/HTTPS validation (`HTTPS`).
- **Domain-Based Features**: Registration length (`DomainRegLen`), age of domain (`AgeofDomain`), DNS records (`DNSRecording`), web traffic rankings (`WebsiteTraffic`), PageRank (`PageRank`), Google index status (`GoogleIndex`), and external links pointing to page (`LinksPointingToPage`).
- **HTML & JavaScript Features**: Favicon origin (`Favicon`), non-standard port requests (`NonStdPort`), HTTPS token in URL (`HTTPSDomainURL`), external resource request URL proportion (`RequestURL`), anchor URL domain mismatch (`AnchorURL`), links in `<script>` and `<link>` tags (`LinksInScriptTags`), Server Form Handler (`ServerFormHandler`), mail submission (`InfoEmail`), abnormal URL host mismatch (`AbnormalURL`), URL forwarding (`WebsiteForwarding`), status bar customization (`StatusBarCust`), right-click disablement (`DisableRightClick`), popup windows (`UsingPopupWindow`), and iframe injection (`IframeRedirection`).

All input attributes take discrete ternary or binary values in $\{-1, 0, 1\}$ representing risk states:
- $+1$: Legitimate / safe configuration.
- $0$: Suspicious / neutral state.
- $-1$: Phishing / high-risk indicator.

---

## 3. Preprocessing & Leakage Protection Principles

### 3.1 Strict Minimalist Preprocessing
Following Rule 3, continuous normalization techniques (e.g., `StandardScaler`, `MinMaxScaler`, `RobustScaler`), outlier clipping, and synthetic oversampling (`SMOTE`) are strictly avoided for tree models. Tree-based partitioning is invariant to monotonic feature transformations. Continuous scaling of discrete ordinal flags alters interpretability without improving split selection. For the linear baseline (Logistic Regression), a standard scaling pipeline is employed solely to ensure numerical stability and gradient descent convergence.

### 3.2 Identifier Excision & Target Normalization
The dataset contains an artificial row index `Index` (`0` to `11,053`). This column provides trivial ordering signal and was excised prior to any modeling. The raw target column (`class`) was standardized to `Result` and mapped from $\{-1, 1\}$ to standard machine-learning binary semantics:
$$\text{Raw } +1 \longrightarrow 0 \quad (\text{Legitimate, Negative Class})$$
$$\text{Raw } -1 \longrightarrow 1 \quad (\text{Phishing, Positive Class})$$

### 3.3 Strict Test Partition Isolation
To guarantee unbiased generalization estimates, a holdout partition of 20% ($n = 2,211$) was separated using stratified random sampling before any model fitting or hyperparameter tuning. The test partition was locked and never accessed during cross-validation, feature importance exploration, or tuning.

---

## 4. Mathematical Foundations of Gradient Boosting Trees

Gradient Tree Boosting builds an additive ensemble of weak decision tree learners $h_m(x)$ by sequentially minimizing a differentiable loss function $\mathcal{L}(y, F(x))$:
$$F_M(x) = F_0(x) + \sum_{m=1}^M \nu \, h_m(x)$$
where $\nu \in (0, 1]$ represents the learning rate (shrinkage factor), and $F_0(x)$ is the optimal constant base prediction.

### 4.1 Negative Gradients as Pseudo-Residuals
For binary classification, the objective is the logistic deviance (negative log-likelihood):
$$\mathcal{L}(y, F(x)) = \log(1 + e^{-2y^* F(x)}) \equiv - [y \log p(x) + (1 - y) \log (1 - p(x))]$$
where $p(x) = \sigma(F(x)) = \frac{1}{1 + e^{-F(x)}}$ is the modeled probability of phishing ($y=1$).

At boosting iteration $m$, the tree $h_m(x)$ is not fit directly to classification errors or discrete labels. Instead, it is fit via least-squares regression to the **negative gradient of the loss with respect to the current ensemble log-odds** $F_{m-1}(x)$:
$$r_{im} = -\left[\frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)}\right]_{F(x) = F_{m-1}(x)} = y_i - p_{m-1}(x_i)$$

This quantity $r_{im} = y_i - p_{m-1}(x_i)$ is termed the **pseudo-residual**.
- When an instance is a true phishing site ($y_i = 1$) but the current ensemble assigns a low probability ($p_{m-1} = 0.10$), the pseudo-residual is large and positive: $r_{im} = +0.90$.
- When an instance is legitimate ($y_i = 0$) but predicted as phishing ($p_{m-1} = 0.85$), the pseudo-residual is large and negative: $r_{im} = -0.85$.
- For well-classified instances, $r_{im} \approx 0$.

Terminal leaf values $\gamma_{j, m}$ for tree $h_m$ in leaf region $R_{j, m}$ are subsequently updated via one-step Newton-Raphson approximation:
$$\gamma_{j, m} = \frac{\sum_{x_i \in R_{j, m}} r_{im}}{\sum_{x_i \in R_{j, m}} p_{m-1}(x_i)(1 - p_{m-1}(x_i))}$$

---

## 5. Experimental Design & Model Optimization

### 5.1 Model Zoo Architecture
1. **Logistic Regression**: Linear baseline with L2 penalty, utilizing `StandardScaler` inside an isolated `Pipeline`.
2. **Decision Tree**: Non-linear, unpruned single CART baseline to quantify single-tree partition capacity.
3. **Random Forest**: Parallel bagging ensemble with 100 trees and randomized split feature subsets ($\sqrt{d}$).
4. **Gradient Boosting (Initial)**: Sequential ensemble with $M=100$, learning rate $\nu=0.1$, tree depth $d=3$.
5. **Gradient Boosting (Tuned)**: Optimized via 50-iteration Stratified 5-Fold Randomized Search over tree depth ($2-5$), learning rate ($0.01-0.2$), estimators ($50-500$), subsampling ratio ($0.7-1.0$), and split thresholds.

### 5.2 Staged Learning Dynamics Analysis
To track internal boosting mechanics without leaking test data, an internal 75/25 development split was utilized to evaluate each stage $m \in \{1, \dots, M\}$:
- **Deviance Convergence**: Tracking `model.train_score_` against iteration index.
- **Validation Loss**: Staged cross-entropy log loss $\mathcal{L}_{\text{val}}(m)$ to detect the onset of overfitting.
- **Classification Error**: Staged misclassification rate $1 - \text{Accuracy}(m)$.
- **Staged F1 & ROC-AUC**: Area under the ROC curve evaluated via continuous predicted probabilities $p_m(x)$.
- **Pseudo-Residual Decay**: Monitoring $\frac{1}{N}\sum |y_i - p_m(x_i)|$ to quantify gradient shrinkage across stages.

---

## 6. Interpretability & Error Auditing

### 6.1 MDI vs. Permutation Feature Importance
- **Mean Decrease in Impurity (MDI)**: Quantifies total variance/impurity reduction across splits in the ensemble trees.
- **Permutation Importance**: Computed on the holdout test set with 20 independent reshuffles per feature, measuring the true degradation in test $F_1$ score. This avoids the known cardinality bias of MDI.

### 6.2 Cybersecurity Error Analysis
In cybersecurity operations, misclassifications carry asymmetric costs:
- **False Negatives (FN)**: Phishing domains classified as legitimate. Represents unmitigated security compromise and high risk.
- **False Positives (FP)**: Legitimate services blocked as phishing. Incurs user friction, operational denial of service, and alert fatigue.
Both error modalities are audited at the individual sample level with full feature inspection and confidence distribution tracking.
