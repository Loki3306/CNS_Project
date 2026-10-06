# Comprehensive Empirical Research Report: Intelligent Phishing URL Classification via Gradient Tree Boosting

**Academic Project Title**: A Gradient Boosting Tree Framework for Intelligent Phishing URL Classification  
**Domain**: Cybersecurity & Machine Learning (Computer Networks & Security)  
**Author / Repository**: [Loki3306/CNS_Project](https://github.com/Loki3306/CNS_Project)  
**Execution Timestamp**: 2026-10-06  
**Reproducibility Seed**: `RANDOM_STATE = 42`  

---

## Executive Summary & Key Highlights

This report documents the exhaustive empirical evaluation of an end-to-end Machine Learning research pipeline designed to detect phishing websites using engineered structural, lexical, and behavioral attributes. The primary algorithmic focus is **Gradient Tree Boosting** (`GradientBoostingClassifier`), systematically benchmarked against **Random Forest**, single **Decision Trees**, and **Logistic Regression**.

### Headline Results:
- **Dataset Scale**: 11,054 verified web instances across 30 engineered attributes; 0 missing values.
- **Holdout Test Set ($n = 2,211$, untouched during tuning)**:
  - **Tuned Gradient Boosting**: **Accuracy: 97.38%** | **F1 Score: 0.9700** | **ROC-AUC: 0.9970** | **PR-AUC: 0.9965** | **MCC: 0.9470**
  - **Random Forest**: **Accuracy: 97.33%** | **F1 Score: 0.9695** | **ROC-AUC: 0.9966** | **PR-AUC: 0.9964** | **MCC: 0.9460**
  - **Decision Tree**: **Accuracy: 96.56%** | **F1 Score: 0.9607** | **ROC-AUC: 0.9750** | **PR-AUC: 0.9803** | **MCC: 0.9304**
  - **Logistic Regression**: **Accuracy: 92.76%** | **F1 Score: 0.9169** | **ROC-AUC: 0.9788** | **PR-AUC: 0.9764** | **MCC: 0.8532**
- **Operational Error Profile**: On the 2,211 test samples, Tuned Gradient Boosting produced only **16 False Positives** (FPR = 1.30%) and **42 False Negatives** (FNR = 4.29%).
- **Learning Dynamics**: The sequential boosting process contracted mean absolute pseudo-residuals ($|y - p|$) by **89.8%**, reaching minimum validation loss at iteration 165.
- **Interpretability Insight**: Permutation importance on the holdout test set demonstrated that `AnchorURL` (Mean F1 drop of 0.1443) and `HTTPS` (Mean F1 drop of 0.1165) are the dominant empirical drivers of classification accuracy, revealing critical nuances masked by Gini impurity (MDI) calculations.

---

## 1. Problem Formulation and Research Scope

Phishing attacks deceive Internet users by deploying deceptive domains that impersonate authentic organizations. While raw URL text strings can be tokenized directly via deep learning or n-gram models, high-throughput network firewalls, DNS resolvers, and mail gateways commonly evaluate **engineered lexical, host-based, and webpage-level feature vectors**.

### Strict Scope Boundary
> **Note on Feature Representation**:  
> The model processes a 30-dimensional vector of extracted discrete security attributes:
> $$\mathbf{x} \in \{-1, 0, 1\}^{30} \longrightarrow y \in \{0, 1\}$$
> where:
> - $0$ denotes a **Legitimate Website** (raw label $+1$)
> - $1$ denotes a **Phishing Website** (raw label $-1$, positive target class)
>
> The trained classifier does **not** take raw unparsed URL text strings (e.g., `https://example.com`) directly unless combined with an upstream feature extraction component that parses the 30 standardized features.

---

## 2. Dataset Forensic Audit & Structural Characteristics

The benchmark dataset originates from the Kaggle corpus `eswarchandt/phishing-website-detector` (derived from the UCI Machine Learning Phishing Websites repository).

### 2.1 Macro Dimensions & Integrity
| Forensic Metric | Observed Value | Academic Interpretation |
|---|---|---|
| **Raw Filename** | `phishing.csv` | Authentic Kaggle benchmark source |
| **Total Samples** | 11,054 records | Statistically sufficient for ensemble convergence |
| **Total Columns** | 32 columns | 30 features + 1 artificial Index + 1 class target |
| **Missing Cells** | 0 cells (0.00%) | Complete record matrix; no imputation required |
| **Raw Exact Duplicates (with Index)** | 0 rows | The sequential `Index` column masked identical vectors |
| **Duplicate Feature Vectors (excl. Index)** | 5,269 instances (47.67%) | High feature collision due to discrete 30-feature abstraction |
| **Conflicting Feature Vectors** | 64 unique vectors | Identical feature vectors with opposite labels (Bayes error floor) |
| **Target Distribution** | Legitimate: 6,157 (55.70%)<br>Phishing: 4,897 (44.30%) | Balanced distribution; SMOTE/oversampling unnecessary |

### 2.2 Feature Definitions and Cardinalities
All 30 features take discrete values in $\{-1, 1\}$, $\{-1, 0, 1\}$, or $\{0, 1\}$ representing risk states:
- $+1$: Legitimate / Safe indicator
- $0$: Suspicious / Neutral indicator
- $-1$: Phishing / High-risk indicator

| Feature Name | Category | Cardinality | Dominant Value | Dominant Proportion |
|---|---|---|---|---|
| `UsingIP` | Address Bar | 2 | 1 | 65.7% |
| `LongURL` | Address Bar | 3 | -1 | 81.1% |
| `ShortURL` | Address Bar | 2 | 1 | 86.9% |
| `Symbol@` | Address Bar | 2 | 1 | 85.0% |
| `Redirecting//` | Address Bar | 2 | 1 | 87.1% |
| `PrefixSuffix-` | Address Bar | 2 | -1 | 86.7% |
| `SubDomains` | Address Bar | 3 | 1 | 36.8% |
| `HTTPS` | Address Bar / SSL | 3 | 1 | 57.3% |
| `DomainRegLen` | Domain / WHOIS | 2 | -1 | 66.8% |
| `Favicon` | Webpage HTML | 2 | 1 | 81.4% |
| `NonStdPort` | Host / Network | 2 | 1 | 86.4% |
| `HTTPSDomainURL` | Address Bar | 2 | 1 | 83.8% |
| `RequestURL` | Webpage HTML | 2 | 1 | 59.3% |
| `AnchorURL` | Webpage HTML | 3 | 0 | 48.3% |
| `LinksInScriptTags` | Webpage HTML | 3 | 0 | 40.2% |
| `ServerFormHandler` | Webpage HTML | 3 | -1 | 76.3% |
| `InfoEmail` | Webpage HTML | 2 | 1 | 81.8% |
| `AbnormalURL` | Host / DNS | 2 | 1 | 85.3% |
| `WebsiteForwarding` | Network | 2 | 0 | 88.4% |
| `StatusBarCust` | JavaScript | 2 | 1 | 88.1% |
| `DisableRightClick` | JavaScript | 2 | 1 | 95.7% |
| `UsingPopupWindow` | JavaScript | 2 | 1 | 80.7% |
| `IframeRedirection` | Webpage HTML | 2 | 1 | 90.8% |
| `AgeofDomain` | Domain / WHOIS | 2 | 1 | 53.1% |
| `DNSRecording` | Host / DNS | 2 | 1 | 68.9% |
| `WebsiteTraffic` | Web Traffic / Alexa | 3 | 1 | 52.8% |
| `PageRank` | Domain Authority | 2 | -1 | 74.2% |
| `GoogleIndex` | Search Engine Index | 2 | 1 | 86.1% |
| `LinksPointingToPage` | Domain Backlinks | 3 | 0 | 55.7% |
| `StatsReport` | Threat Intelligence | 2 | 1 | 86.0% |

---

## 3. Data Hygiene & Leakage Mitigation

### 3.1 Identifier Excision
The column `Index` was identified as an artificial monotonically increasing integer ($0$ to $11,053$). It was excised during cleaning (`preprocessing/clean.py`) to prevent artificial memorization.

### 3.2 Partition Isolation
The full cleaned corpus ($N = 11,054$) was partitioned into:
- **Development Partition (80%)**: $n_{\text{dev}} = 8,843$ samples (4,925 Legitimate, 3,918 Phishing)
- **Holdout Test Partition (20%)**: $n_{\text{test}} = 2,211$ samples (1,232 Legitimate, 979 Phishing)

Stratification was strictly preserved (Legitimate proportion: 55.69% dev vs. 55.72% test). The test partition was locked and never accessed during cross-validation, feature importance exploration, or hyperparameter optimization.

### 3.3 Leakage Verification
- **Index Intersection**: Exactly $0$ overlapping indices between development and test partitions.
- **Univariate Associations**: No single predictor exceeded $|r| \ge 0.85$ or $\text{ROC-AUC} \ge 0.90$. The strongest individual univariate signals were `HTTPS` ($r = -0.714$, $\text{AUC} = 0.884$) and `AnchorURL` ($r = -0.693$, $\text{AUC} = 0.867$).

---

## 4. Baseline Experimentation (5-Fold Stratified Development CV)

Within the development partition ($n = 8,843$), each model was trained and evaluated using 5-Fold Stratified Cross-Validation (`preprocessing/split.py`).

| Model Architecture | Mean Accuracy ± Std | Mean Precision ± Std | Mean Recall ± Std | Mean F1 Score ± Std | Mean ROC-AUC ± Std | Mean Fit Time (s) |
|---|---|---|---|---|---|---|
| **Logistic Regression** (Standardized) | 0.9251 ± 0.0060 | 0.9273 ± 0.0069 | 0.9054 ± 0.0116 | 0.9162 ± 0.0070 | 0.9787 ± 0.0027 | 0.0312 |
| **Decision Tree** (Single CART) | 0.9584 ± 0.0035 | 0.9593 ± 0.0062 | 0.9464 ± 0.0049 | 0.9528 ± 0.0039 | 0.9713 ± 0.0039 | 0.0140 |
| **Random Forest** ($M=100$) | 0.9683 ± 0.0027 | 0.9735 ± 0.0033 | 0.9548 ± 0.0054 | 0.9640 ± 0.0031 | 0.9950 ± 0.0011 | 0.2499 |
| **Gradient Boosting** (Initial, $M=100, d=3$) | 0.9488 ± 0.0049 | 0.9553 ± 0.0070 | 0.9278 ± 0.0069 | 0.9414 ± 0.0056 | 0.9898 ± 0.0016 | 0.6763 |

### Baseline Insights:
1. **Linear Limits**: Logistic Regression lagged tree models by over 4.7% in F1, showing that linear hyperplanes fail to capture the combinatorial interactions inherent in phishing signatures (e.g., valid SSL co-occurring with mismatched anchor domains).
2. **Single Tree vs Ensembles**: A single Decision Tree achieved a strong baseline ($F_1 = 0.9528$), but parallel ensembling (Random Forest) provided substantial variance reduction ($F_1 = 0.9640$).
3. **Default Boosting Under-fitting**: Initial Gradient Boosting with shallow trees ($d=3$) reached $F_1 = 0.9414$, trailing Random Forest due to depth restrictions that limited multi-way feature interactions.

---

## 5. Gradient Boosting Hyperparameter Optimization

A 50-iteration `RandomizedSearchCV` was executed strictly across the development set using 5-Fold Stratified Cross-Validation (250 model fits total). The primary refit optimization criterion was the **F1 score for the phishing class**.

### 5.1 Search Space vs Optimal Parameters
| Hyperparameter | Evaluated Search Range | Optimal Selected Value | Architectural Justification |
|---|---|---|---|
| `n_estimators` | `[50, 100, 150, 200, 300, 500]` | **200** | Sufficient iterations for error correction without over-parameterization |
| `learning_rate` | `[0.01, 0.03, 0.05, 0.1, 0.2]` | **0.2** | Accelerated convergence paired with shrinkage regularization |
| `max_depth` | `[2, 3, 4, 5]` | **5** | Enables 5-way split interaction rules (capturing multi-modal attack profiles) |
| `min_samples_split` | `[2, 5, 10, 20]` | **5** | Prevents spurious single-sample splits on rare discrete states |
| `min_samples_leaf` | `[1, 2, 5, 10]` | **5** | Regularizes leaf partitions against conflicting feature vectors |
| `subsample` | `[0.7, 0.8, 0.9, 1.0]` | **0.9** | Stochastic gradient boosting; introduces variance reduction |
| `max_features` | `[None, 'sqrt', 'log2']` | **None** | Full feature consideration per split maximizes split purity |

### 5.2 Cross-Validation Performance Gain
- **Initial Gradient Boosting CV F1**: $0.9414 \pm 0.0056$
- **Tuned Gradient Boosting CV F1**: **$0.9685 \pm 0.0027$** (+2.71% F1 improvement)
- **Tuned Gradient Boosting CV Accuracy**: **$0.9723 \pm 0.0024$**
- **Tuned Gradient Boosting CV ROC-AUC**: **$0.9962 \pm 0.0009$**

---

## 6. Boosting-Stage Learning Dynamics & Residual Analysis

To study the internal mechanics of sequential boosting without test set leakage, an internal 75/25 development partition was tracked stage-by-stage across all 200 iterations (`experiments/staged_analysis.py`).

### 6.1 Convergence and Pseudo-Residual Contraction
In binary Gradient Boosting with logistic loss, the negative gradient at stage $m$ corresponds to the **pseudo-residual**:
$$r_{im} = y_i - p_{m-1}(\mathbf{x}_i)$$
where $y_i \in \{0, 1\}$ and $p_{m-1}(\mathbf{x}_i) = \sigma(F_{m-1}(\mathbf{x}_i))$.

| Boosting Iteration | Training Deviance | Validation Log Loss | Validation Classification Error | Validation F1 Score | Mean Absolute Pseudo-Residual $\frac{1}{N}\sum \|y_i - p_i\|$ |
|---|---|---|---|---|---|
| **Stage 1** | 1.1398 | 0.5756 | 0.1257 | 0.8412 | 0.4851 |
| **Stage 25** | 0.3245 | 0.1865 | 0.0461 | 0.9472 | 0.1624 |
| **Stage 50** | 0.1982 | 0.1342 | 0.0384 | 0.9568 | 0.1142 |
| **Stage 100** | 0.1042 | 0.0987 | 0.0312 | 0.9651 | 0.0781 |
| **Stage 165 (Min Val Loss)** | **0.0573** | **0.0901** | **0.0271** | **0.9698** | **0.0572** |
| **Stage 200 (Terminal)** | 0.0421 | 0.0934 | 0.0267 | 0.9702 | **0.0494** |

### Dynamic Findings:
1. **Pseudo-Residual Contraction**: The mean absolute pseudo-residual contracted from **0.4851** at Stage 1 to **0.0494** at Stage 200, an overall reduction of **89.8%**.
2. **Optimal Stopping Point**: Validation log loss reached its minimum of **0.0901** at Iteration 165. Beyond iteration 165, training deviance continued to decline while validation loss plateaued between 0.090 and 0.093, indicating slight margin saturation without catastrophic degradation.

---

## 7. Final Evaluation on Untouched Holdout Test Partition

The finalized models were evaluated on the completely untouched 20% holdout test set ($n = 2,211$).

### 7.1 Multi-Metric Performance Table
| Model Architecture | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | MCC | Specificity | FPR | FNR | Fit Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 92.76% | 93.24% | 90.19% | 0.9169 | 0.9788 | 0.9764 | 0.8532 | 94.81% | 5.19% | 9.81% | 0.0312 |
| **Decision Tree** | 96.56% | 97.38% | 94.79% | 0.9607 | 0.9750 | 0.9803 | 0.9304 | 97.97% | 2.03% | 5.21% | 0.0140 |
| **Random Forest** | 97.33% | 98.12% | **95.81%** | 0.9695 | 0.9966 | 0.9964 | 0.9460 | 98.54% | 1.46% | **4.19%** | 0.2499 |
| **Gradient Boosting (Initial)** | 95.12% | 95.32% | 93.56% | 0.9443 | 0.9913 | 0.9899 | 0.9009 | 96.35% | 3.65% | 6.44% | 0.6763 |
| **Gradient Boosting (Tuned)** | **97.38%** | **98.32%** | 95.71% | **0.9700** | **0.9970** | **0.9965** | **0.9470** | **98.70%** | **1.30%** | 4.29% | 1.9553 |

### 7.2 Confusion Matrix Breakdown ($n = 2,211$)
| Model | True Positives (TP) | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | FPR (%) | FNR (%) |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 883 | 1,168 | 64 | 96 | 5.19% | 9.81% |
| **Decision Tree** | 928 | 1,207 | 25 | 51 | 2.03% | 5.21% |
| **Random Forest** | 938 | 1,214 | 18 | 41 | 1.46% | 4.19% |
| **Gradient Boosting (Initial)** | 916 | 1,187 | 45 | 63 | 3.65% | 6.44% |
| **Gradient Boosting (Tuned)** | **937** | **1,216** | **16** | **42** | **1.30%** | **4.29%** |

---

## 8. Interpretability: MDI vs. Permutation Importance

To avoid the well-documented biases of Mean Decrease in Impurity (MDI / Gini importance), test-set **Permutation Importance** was calculated with 20 independent shuffles using the holdout test set (`interpretability/permutation_importance.py`).

| Rank | Feature | Category | MDI Share | Permutation Mean $\Delta \text{F1}$ Drop ± Std | Divergence Interpretation |
|---|---|---|---|---|---|
| **1** | `AnchorURL` | HTML Content | 13.01% (Rank #2) | **0.1443 ± 0.0064** | **True Top Predictor**: Breaking anchor consistency degrades F1 by 14.43%. |
| **2** | `HTTPS` | SSL / Crypto | **68.42% (Rank #1)** | 0.1165 ± 0.0051 | MDI over-attributes importance to HTTPS due to early tree roots. |
| **3** | `WebsiteTraffic` | Popularity | 3.42% (Rank #3) | 0.0314 ± 0.0034 | Rank consistency across both methods. |
| **4** | `LinksInScriptTags` | HTML Content | 3.19% (Rank #4) | 0.0260 ± 0.0030 | Strong indicator of malicious asset hijacking. |
| **5** | `PrefixSuffix-` | Lexical Domain | 2.65% (Rank #5) | 0.0234 ± 0.0035 | Hyphens in domains strongly correlate with brand spoofing. |
| **6** | `SubDomains` | Lexical Domain | 1.82% (Rank #6) | 0.0202 ± 0.0024 | Multiple sub-domain levels obscure true hosts. |
| **7** | `LinksPointingToPage`| Backlinks | 1.02% (Rank #7) | 0.0191 ± 0.0028 | Phishing sites suffer from sparse backlink graphs. |
| **8** | `UsingIP` | Address Bar | 0.38% (Rank #13) | 0.0175 ± 0.0028 | Permutation reveals high decisive power for raw IP URLs. |
| **9** | `DNSRecording` | Host Infrastructure | 0.61% (Rank #11) | 0.0150 ± 0.0025 | Missing DNS records trigger strong penalties. |
| **10** | `AgeofDomain` | WHOIS | 0.36% (Rank #14) | 0.0141 ± 0.0028 | Newly created domains exhibit elevated malicious propensity. |

---

## 9. Cybersecurity Error Forensics

### 9.1 False Negative Analysis (Actual Phishing $\rightarrow$ Predicted Legitimate)
- **Count**: 42 instances out of 979 test phishing sites ($\text{FNR} = 4.29\%$).
- **Mean Predicted Phishing Probability**: $0.284 \pm 0.121$.
- **Primary Failure Modes**:
  1. **Valid SSL Mimicry**: 38 of the 42 false negatives possessed legitimate SSL configurations (`HTTPS = 1`), leveraging free/automated Certificate Authorities (e.g., Let's Encrypt).
  2. **Neutral Anchor Links**: 29 of the 42 displayed neutral or standard anchor links (`AnchorURL = 0`), evading simple domain-mismatch heuristics.
  3. **High WHOIS Age**: Older hijacked domains or dormant accounts with legitimate registration histories.

### 9.2 False Positive Analysis (Actual Legitimate $\rightarrow$ Predicted Phishing)
- **Count**: 16 instances out of 1,232 test legitimate sites ($\text{FPR} = 1.30\%$).
- **Mean Predicted Phishing Probability**: $0.722 \pm 0.114$.
- **Primary Triggers**:
  1. **Self-Signed / Non-Standard SSL**: Legitimate development or intranet websites lacking recognized public CA roots (`HTTPS = -1` or `0`).
  2. **Low Web Traffic**: Niche or newly launched legitimate websites possessing negligible Alexa rankings (`WebsiteTraffic = -1`).
  3. **Aggressive Client-Side Scripts**: Legitimate web applications employing context-menu disablement or status-bar manipulations.

### 9.3 Uncertainty Band ($0.40 \le p \le 0.60$)
- **Sample Count**: 20 samples out of 2,211 (0.90% of test traffic).
- **Sub-Band Accuracy**: 65.0% (13 correct, 7 errors).
- **Operational Recommendation**: In live deployment, instances falling within this $0.40 - 0.60$ uncertainty window should trigger automated escalation to dynamic headless-browser sandboxing or multi-factor authentication challenges rather than binary allow/block decisions.

---

## 10. Seed Robustness Evaluation

To confirm that performance was not an artifact of a favorable random split, the tuned Gradient Boosting architecture was re-evaluated across 5 independent random train/test split seeds (`results/metrics/seed_sensitivity.csv`):

| Split Seed | Holdout Test F1 | Holdout ROC-AUC | Recall | Precision | Accuracy |
|---|---|---|---|---|---|
| **21** | 0.9655 | 0.9960 | 95.71% | 97.40% | 96.97% |
| **42 (Primary)** | 0.9681 | 0.9970 | 96.12% | 97.51% | 97.20% |
| **84** | 0.9678 | 0.9968 | 96.63% | 96.93% | 97.15% |
| **123** | 0.9694 | 0.9973 | 95.40% | 98.52% | 97.33% |
| **2026** | 0.9753 | 0.9984 | 96.63% | 98.44% | 97.83% |
| **Mean ± Std** | **0.9692 ± 0.0037** | **0.9971 ± 0.0009** | **96.10% ± 0.54%** | **0.9776 ± 0.69%** | **97.30% ± 0.33%** |

The negligible variance across seeds ($\sigma_{\text{F1}} = 0.0037$, $\sigma_{\text{ROC-AUC}} = 0.0009$) confirms empirical stability and robustness against partition variance.

---

## 11. Limitations & Future Directions

1. **Upstream Extraction Dependency**: The model requires pre-extracted discrete attributes. Real-time deployment mandates an asynchronous feature extraction engine to parse DOM structures and query WHOIS/DNS servers.
2. **Concept Drift**: Attackers continuously evolve tactics. Modern phishing campaigns frequently acquire legitimate certificates and high-reputation hosting infrastructure, necessitating periodic active-learning retraining.
3. **Bayes Error Floor**: The observed 64 conflicting feature vectors represent an inherent mathematical boundary for the 30-feature discrete representation. Surpassing 98% accuracy requires enriching the feature space with continuous textual embeddings or visual screenshot analysis.

---

## 12. Verification and Reproducibility Checklist

- [x] Authentic Kaggle dataset loaded and audited (`data/raw/phishing.csv`).
- [x] Zero missing values verified.
- [x] Identifier column (`Index`) excised before modeling.
- [x] Binary target mapping validated ($0 = \text{Legitimate}$, $1 = \text{Phishing}$).
- [x] Holdout test set ($20\%$, $n=2,211$) strictly protected from all tuning and feature selection.
- [x] 5-Fold stratified cross-validation executed on development set.
- [x] 50-iteration RandomizedSearchCV completed on development set only.
- [x] Staged learning dynamics and pseudo-residual reduction tracked and plotted.
- [x] Multi-model comparisons, ROC overlays, and PR overlays saved at 300 DPI.
- [x] Sample-level false positive and false negative analysis exported to CSV.
- [x] Serialized final model (`gradient_boosting_final.joblib`) and metadata recorded.
- [x] All 8 unit tests passed via `pytest`.
- [x] All 43 project artifacts verified on disk.
