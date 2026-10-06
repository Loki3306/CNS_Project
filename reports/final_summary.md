# Final Project Summary

## 1. Objective
This investigation engineered an end-to-end, scientifically defensible machine learning pipeline to classify phishing websites using discrete behavioral, structural, and lexical indicators. The core focus was evaluating sequential Gradient Tree Boosting, contrasting it against classical parallel ensembles (Random Forest) and linear/single-tree baselines, while rigorously analyzing boosting-stage learning mechanics, error modalities, and data leakage risks.

## 2. Dataset
- **Corpus**: `eswarchandt/phishing-website-detector` (derived from the UCI Phishing Websites benchmark).
- **Geometry**: 11,054 samples, 30 discrete attributes in {-1, 0, 1}, 1 binary target.
- **Scope**: Features represent extracted domain and webpage attributes, **not** raw unparsed URL text strings.

## 3. Dataset Quality Findings
- **Data Integrity**: 0 missing entries across all 32 columns.
- **Class Balance**: 6,157 Legitimate (55.70%) vs. 4,897 Phishing (44.30%). No synthetic resampling (SMOTE) was warranted.
- **Identifier Leakage**: An artificial row index (`Index`) was detected and stripped.
- **Feature Duplication**: 5,269 records share feature representations with other samples. Among these, 64 distinct vectors exhibit conflicting labels, defining the intrinsic Bayes error floor in this discrete representation.

## 4. Methodology
- **Partitioning**: 80% Development (8,843 samples) and 20% Holdout Test (2,211 samples), stratified on class balance.
- **Validation**: 5-Fold Stratified Cross-Validation on the development partition only. The test partition was isolated from tuning and model selection.
- **Preprocessing**: Invariant tree features remained unscaled; Logistic Regression was paired with `StandardScaler` inside a Pipeline.

## 5. Baselines
Across 5-fold cross-validation on the development partition:
- **Logistic Regression**: Mean F1 = 0.9162 ± 0.0070, ROC-AUC = 0.9787.
- **Decision Tree**: Mean F1 = 0.9528 ± 0.0039, ROC-AUC = 0.9713.
- **Random Forest**: Mean F1 = 0.9640 ± 0.0031, ROC-AUC = 0.9950.

## 6. Gradient Boosting Configuration
The default Gradient Boosting configuration ($M=100, 
u=0.1, d=3$) achieved a development CV F1 of 0.9414 ± 0.0056. While superior to Logistic Regression, it trailed Random Forest prior to tuning due to conservative depth restrictions.

## 7. Hyperparameter Search
A 50-iteration randomized search over depth ($2-5$), learning rate ($0.01-0.2$), estimators ($50-500$), and subsampling selected:
- `n_estimators`: 200
- `learning_rate`: 0.2
- `max_depth`: 5
- `subsample`: 0.9
- `max_features`: None
This raised development CV F1 to **0.9685 ± 0.0027**, demonstrating the critical role of deeper trees ($d=5$) in capturing multi-feature interaction rules.

## 8. Learning Dynamics
Staged boosting trajectory tracking revealed:
- Monotonic decrease in training deviance.
- Validation log loss reached its minimum of 0.0901 at iteration 165.
- Mean absolute pseudo-residuals ($|y - p|$) contracted from 0.4183 to 0.0426.

## 9. Final Test Results
On the completely untouched 20% holdout test partition ($n = 2,211$):
- **Accuracy**: 97.38%
- **Precision**: 98.32%
- **Recall**: 95.71%
- **F1 Score**: 0.9700
- **ROC-AUC**: 0.9970
- **PR-AUC**: 0.9965
- **MCC**: 0.9470

## 10. Model Comparison
Final test comparison demonstrates:
- **Best Performing Model on Test F1**: `Gradient Boosting (Tuned)` (F1 = 0.9700).
- Both Random Forest and Tuned Gradient Boosting significantly outperform linear and single-tree baselines.
- Random Forest and Tuned Gradient Boosting operate in a near-parity high-accuracy regime (>97% accuracy, >0.965 F1), with Tuned Gradient Boosting demonstrating competitive precision and calibrated decision margins.

## 11. Feature Importance
- **MDI Dominance**: `SSLfinal_State` (HTTPS certificate validation), `URL_of_Anchor`, and `Prefix_Suffix` dominate split impurity reductions.
- **Permutation Significance**: Test-set permutation importance confirms that destroying `SSLfinal_State` produces the steepest drop in test F1, followed by `URL_of_Anchor` and `web_traffic`.

## 12. Error Analysis
- **False Negatives**: 42 phishing instances slipped past the classifier (FNR = 4.29%). These instances predominantly exhibit valid SSL certificates and neutral anchor links, effectively mimicking authentic services.
- **False Positives**: 16 legitimate domains were flagged (FPR = 1.30%). These typically involve self-signed certificates or newer domains.
- **Uncertainty Band**: 20 predictions fell in the $0.40 - 0.60$ probability range, representing candidate instances for multi-factor secondary verification.

## 13. Leakage Assessment
- Verified 0 index intersection between train and test.
- Audited duplicate vectors and quantified overlapping feature signatures across partitions.

## 14. Limitations
1. **Engineered vs. Raw URLs**: The model requires an upstream feature extraction engine. It cannot classify raw text URLs directly.
2. **Concept Drift**: Attack tactics evolve; features like static SSL presence have diminished in modern phishing as adversaries freely adopt Let's Encrypt certificates.
3. **Discrete Feature Granularity**: The 30 discrete attributes yield duplicate feature vectors and 64 conflicting entries, establishing a hard Bayes error floor.

## 15. Conclusion
Gradient Tree Boosting provides a principled, highly performant architecture for structured phishing URL classification. Hyperparameter optimization successfully closed the initial gap with Random Forest, achieving publication-grade detection performance while maintaining full interpretability through staged residual analysis and permutation importance ranking.
