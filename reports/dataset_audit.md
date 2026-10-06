# Dataset Audit Report

## 1. Executive Summary

- **Source Dataset File**: `phishing.csv`
- **Dimensions**: 11,054 rows × 32 columns
- **Identified Target Column**: `class`
- **Identified Identifier Columns**: `['Index']`
- **Missing Values Count**: 0 across all cells (0.00%)
- **Total Duplicate Rows (including Index)**: 0
- **Duplicate Feature Vectors (excluding Index)**: 5,269 (47.67%)
- **Conflicting Duplicate Vectors**: 64 distinct feature vectors have conflicting target labels

---

## 2. Observed Facts vs. Scientific Interpretation

### Fact 1: Dataset Geometry and Feature Representation
- **Observed Fact**: The dataset contains 11,054 samples and 30 categorical/discrete engineered features plus an artificial `Index` column (values `0` to `11053`) and `class`. All features take discrete values in `{-1, 1}` or `{-1, 0, 1}` or `{0, 1}`.
- **Interpretation**: This is an engineered feature benchmark dataset (originating from the UCI Phishing Websites repository) rather than a raw URL corpus. Classification operates on 30 discrete web and URL properties rather than lexical character sequences.

### Fact 2: Class Balance
- **Observed Fact**: Class labels are encoded as `-1` and `1`.
  - Label `1` (Legitimate): 6,157 instances (55.70%)
  - Label `-1` (Phishing): 4,897 instances (44.30%)
- **Interpretation**: The classes are approximately balanced (55.70% vs 44.30%). Synthetic balancing methods such as SMOTE or class-weight skewing are unneeded and would distort genuine feature co-occurrence frequencies.

### Fact 3: Identifier Leakage
- **Observed Fact**: The column `Index` is an integer ranging from `0` to `11053`, uniquely identifying every row.
- **Interpretation**: The `Index` column carries trivial ordering information and must be dropped prior to modeling to prevent artificial memorization.

### Fact 4: Feature Duplication and Conflicting Labels
- **Observed Fact**: When `Index` is excluded, 5,269 rows (47.67%) have identical feature representations to another row. Among these, 64 feature vectors map to conflicting labels.
- **Interpretation**: In a discrete 30-feature space, multiple distinct URLs produce identical discrete feature vectors. The 64 conflicting patterns represent the theoretical Bayes error floor for this representation (a deterministic classifier cannot achieve 100% accuracy on these points).

### Fact 5: Near-Constant Features
- **Observed Fact**: Features with >= 90% dominant value concentration:
  - `DisableRightClick`: dominant value 1 (95.69%)
  - `IframeRedirection`: dominant value 1 (90.84%)
- **Interpretation**: While skewed, tree-based models (such as Gradient Boosting and Random Forest) can still extract isolated split criteria from rare states when they indicate high-risk malicious configurations.

---

## 3. Schema & Feature Cardinality Table

| Column Name | Dtype | Min | Max | Unique Values | Missing (%) | Dominant Proportion (%) |
|---|---|---|---|---|---|---|
| `Index` | `int64` | 0.0 | 11053.0 | 11054 | 0.0% | 0.0% |
| `UsingIP` | `int64` | -1.0 | 1.0 | 2 | 0.0% | 65.7% |
| `LongURL` | `int64` | -1.0 | 1.0 | 3 | 0.0% | 81.1% |
| `ShortURL` | `int64` | -1.0 | 1.0 | 2 | 0.0% | 86.9% |
| `Symbol@` | `int64` | -1.0 | 1.0 | 2 | 0.0% | 85.0% |
| `Redirecting//` | `int64` | -1.0 | 1.0 | 2 | 0.0% | 87.1% |
| `PrefixSuffix-` | `int64` | -1.0 | 1.0 | 2 | 0.0% | 86.7% |
| `SubDomains` | `int64` | -1.0 | 1.0 | 3 | 0.0% | 36.8% |
| `HTTPS` | `int64` | -1.0 | 1.0 | 3 | 0.0% | 57.3% |
| `DomainRegLen` | `int64` | -1.0 | 1.0 | 2 | 0.0% | 66.8% |
| `Favicon` | `int64` | -1.0 | 1.0 | 2 | 0.0% | 81.4% |
| `NonStdPort` | `int64` | -1.0 | 1.0 | 2 | 0.0% | 86.4% |
| `HTTPSDomainURL` | `int64` | -1.0 | 1.0 | 2 | 0.0% | 83.8% |
| `RequestURL` | `int64` | -1.0 | 1.0 | 2 | 0.0% | 59.3% |
| `AnchorURL` | `int64` | -1.0 | 1.0 | 3 | 0.0% | 48.3% |
| `LinksInScriptTags` | `int64` | -1.0 | 1.0 | 3 | 0.0% | 40.2% |
| `ServerFormHandler` | `int64` | -1.0 | 1.0 | 3 | 0.0% | 76.3% |
| `InfoEmail` | `int64` | -1.0 | 1.0 | 2 | 0.0% | 81.8% |
| `AbnormalURL` | `int64` | -1.0 | 1.0 | 2 | 0.0% | 85.3% |
| `WebsiteForwarding` | `int64` | 0.0 | 1.0 | 2 | 0.0% | 88.4% |
| `StatusBarCust` | `int64` | -1.0 | 1.0 | 2 | 0.0% | 88.1% |
| `DisableRightClick` | `int64` | -1.0 | 1.0 | 2 | 0.0% | 95.7% |
| `UsingPopupWindow` | `int64` | -1.0 | 1.0 | 2 | 0.0% | 80.7% |
| `IframeRedirection` | `int64` | -1.0 | 1.0 | 2 | 0.0% | 90.8% |
| `AgeofDomain` | `int64` | -1.0 | 1.0 | 2 | 0.0% | 53.1% |
| `DNSRecording` | `int64` | -1.0 | 1.0 | 2 | 0.0% | 68.9% |
| `WebsiteTraffic` | `int64` | -1.0 | 1.0 | 3 | 0.0% | 52.8% |
| `PageRank` | `int64` | -1.0 | 1.0 | 2 | 0.0% | 74.2% |
| `GoogleIndex` | `int64` | -1.0 | 1.0 | 2 | 0.0% | 86.1% |
| `LinksPointingToPage` | `int64` | -1.0 | 1.0 | 3 | 0.0% | 55.7% |
| `StatsReport` | `int64` | -1.0 | 1.0 | 2 | 0.0% | 86.0% |
| `class` | `int64` | -1.0 | 1.0 | 2 | 0.0% | 0.0% |

---

## 4. Preprocessing Implications

1. **Identifier Removal**: `Index` must be excised immediately.
2. **Target Encoding**: Raw labels `[1, -1]` must be mapped to standard binary integer targets: `0` for Legitimate (`1`), `1` for Phishing (`-1`).
3. **No Continuous Normalization**: All features are discrete categories in `{-1, 0, 1}`. Tree models require no scaling; Logistic Regression requires standard scaling pipeline.
4. **Leakage & Duplicate Tracking**: Split stratification must be conducted strictly on development data before tuning.
