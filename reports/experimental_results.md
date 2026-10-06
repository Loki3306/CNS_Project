# Experimental Results Report

## 1. Dataset Dimensions & Verified Quality

| Metric | Measured Value |
|---|---|
| **Raw Dataset File** | `phishing.csv` |
| **Total Rows** | 11,054 |
| **Total Features (Input)** | 30 discrete attributes |
| **Target Distribution** | Legitimate (0): 6,157 (55.70%), Phishing (1): 4,897 (44.30%) |
| **Exact Duplicate Rows (Raw with Index)** | 0 |
| **Duplicate Feature Vectors (Excl. Index)** | 5,269 |
| **Conflicting Duplicate Vectors** | 64 |
| **Missing Values** | 0 cells (0.00%) |
| **Partitioning** | 80% Development (8,843 samples) / 20% Holdout Test (2,211 samples) |

---

## 2. Baseline Model Cross-Validation (5-Fold Development CV)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Fit Time (s) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9267 ± 0.0059 | 0.9283 ± 0.0074 | 0.9045 ± 0.0112 | 0.9162 ± 0.0070 | 0.9787 ± 0.0027 | 0.0305 |
| Decision Tree | 0.9587 ± 0.0032 | 0.9647 ± 0.0057 | 0.9413 ± 0.0084 | 0.9528 ± 0.0039 | 0.9713 ± 0.0039 | 0.0206 |
| Random Forest | 0.9684 ± 0.0025 | 0.9738 ± 0.0038 | 0.9546 ± 0.0095 | 0.9640 ± 0.0031 | 0.9950 ± 0.0011 | 0.3304 |
| Gradient Boosting (Initial) | 0.9485 ± 0.0050 | 0.9509 ± 0.0085 | 0.9321 ± 0.0045 | 0.9414 ± 0.0056 | 0.9898 ± 0.0016 | 0.7064 |


---

## 3. Gradient Boosting Hyperparameter Search Results

- **Exploration Technique**: `RandomizedSearchCV` (50 configurations, 5-Fold Stratified CV)
- **Primary Optimization Metric**: F1 Score (Phishing Class)
- **Initial Configuration CV F1**: 0.9414 ± 0.0056
- **Best Parameter Configuration**:
```json
{
  "subsample": 0.9,
  "n_estimators": 200,
  "min_samples_split": 5,
  "min_samples_leaf": 5,
  "max_features": null,
  "max_depth": 5,
  "learning_rate": 0.2
}
```
- **Tuned Configuration CV F1**: 0.9685 ± 0.0027
- **Tuned Configuration CV Accuracy**: 0.9723 ± 0.0024
- **Tuned Configuration CV ROC-AUC**: 0.9963 ± 0.0008

---

## 4. Staged Learning Dynamics

- **Estimators Evaluated**: 200
- **Minimum Validation Loss**: 0.0901 achieved at Iteration 165
- **Maximum Validation F1**: 0.9617 achieved at Iteration 165
- **Initial Pseudo-Residual Mean**: 0.4183
- **Terminal Pseudo-Residual Mean**: 0.0426 (decay of 89.8%)

---

## 5. Final Holdout Test Performance (Untouched 20% Partition)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | MCC | Specificity | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.9276 | 0.9324 | 0.9019 | 0.9169 | 0.9788 | 0.9764 | 0.8532 | 0.9481 | 0.0519 | 0.0981 |
| Decision Tree | 0.9656 | 0.9738 | 0.9479 | 0.9607 | 0.9750 | 0.9803 | 0.9304 | 0.9797 | 0.0203 | 0.0521 |
| Random Forest | 0.9733 | 0.9812 | 0.9581 | 0.9695 | 0.9966 | 0.9964 | 0.9460 | 0.9854 | 0.0146 | 0.0419 |
| Gradient Boosting (Initial) | 0.9512 | 0.9532 | 0.9356 | 0.9443 | 0.9913 | 0.9899 | 0.9009 | 0.9635 | 0.0365 | 0.0644 |
| Gradient Boosting (Tuned) | 0.9738 | 0.9832 | 0.9571 | 0.9700 | 0.9970 | 0.9965 | 0.9470 | 0.9870 | 0.0130 | 0.0429 |


---

## 6. Confusion Matrix & Error Profile (Tuned Gradient Boosting)

| Evaluation Metric | Test Count / Rate | Operational Cybersecurity Impact |
|---|---|---|
| **True Positives (TP)** | 937 | Successfully intercepted phishing attacks |
| **True Negatives (TN)** | 1216 | Correctly authorized legitimate web services |
| **False Positives (FP)** | 16 | Legitimate websites misclassified as phishing (user friction) |
| **False Negatives (FN)** | 42 | Undetected phishing attacks breaching network defenses |
| **False Positive Rate (FPR)** | 1.30% | Proportion of legitimate traffic erroneously blocked |
| **False Negative Rate (FNR)** | 4.29% | Proportion of phishing threats erroneously admitted |
| **Uncertainty Region (0.40 - 0.60)** | 20 samples | Borderline instances requiring escalation to secondary analysis |

---

## 7. Interpretability Comparison

| Rank | MDI Feature (Tree Impurity) | MDI Importance | Permutation Feature (Test Set) | Mean F1 Drop ± Std |
|---|---|---|---|---|
| 1 | `HTTPS` | 0.6842 | `AnchorURL` | 0.1443 ± 0.0064 |
| 2 | `AnchorURL` | 0.1301 | `HTTPS` | 0.1165 ± 0.0051 |
| 3 | `WebsiteTraffic` | 0.0342 | `WebsiteTraffic` | 0.0314 ± 0.0034 |
| 4 | `LinksInScriptTags` | 0.0319 | `LinksInScriptTags` | 0.0260 ± 0.0030 |
| 5 | `PrefixSuffix-` | 0.0265 | `PrefixSuffix-` | 0.0234 ± 0.0035 |
| 6 | `SubDomains` | 0.0182 | `SubDomains` | 0.0202 ± 0.0024 |
| 7 | `LinksPointingToPage` | 0.0102 | `LinksPointingToPage` | 0.0191 ± 0.0028 |
| 8 | `ServerFormHandler` | 0.0089 | `UsingIP` | 0.0175 ± 0.0028 |
| 9 | `DomainRegLen` | 0.0082 | `DNSRecording` | 0.0150 ± 0.0025 |
| 10 | `GoogleIndex` | 0.0064 | `AgeofDomain` | 0.0141 ± 0.0028 |

