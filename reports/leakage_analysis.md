# Leakage Investigation & Contamination Audit

## 1. Executive Summary

- **Evaluated Dataset Records**: 11,054
- **Identifier Leakage**: 1 column(s) detected matching identifier patterns (['GoogleIndex']).
- **Target-Derived Column Names**: 0 column(s) flagged ([]).
- **Exact (Feature + Target) Duplicates**: 5,205
- **Duplicate Feature Vectors**: 5,269
- **Conflicting Feature Vectors**: 64 (same features, divergent class labels)
- **Extremely Predictive Features (AUC >= 0.90 or |r| >= 0.85)**: 0 flagged (None).

---

## 2. In-Depth Leakage Audits

### 2.1 Identifier Leakage
No primary index or artificial row identification column remains in the cleaned dataset. All input columns correspond to domain-level, lexical, or structural webpage attributes.

### 2.2 Univariate Predictor Strength
Features with strongest individual association with the target:

| Feature | Pearson r | Univariate ROC-AUC | Status |
|---|---|---|---|
| `HTTPS` | -0.7147 | 0.8771 | OK |
| `AnchorURL` | -0.6929 | 0.8758 | OK |
| `PrefixSuffix-` | -0.3486 | 0.6190 | OK |
| `WebsiteTraffic` | -0.3460 | 0.6929 | OK |
| `SubDomains` | -0.2982 | 0.6667 | OK |
| `RequestURL` | -0.2535 | 0.6253 | OK |
| `LinksInScriptTags` | -0.2484 | 0.6364 | OK |
| `DomainRegLen` | +0.2259 | 0.6070 | OK |
| `ServerFormHandler` | -0.2214 | 0.5945 | OK |
| `GoogleIndex` | -0.1290 | 0.5450 | OK |

No single feature achieves an AUC >= 0.90 or correlation >= 0.85. The strongest predictive signal originates from `SSLfinal_State` (or `HTTPS`) and `AnchorURL`, which is consistent with phishing domain architecture (absence of recognized SSL certificates and anchor links pointing to mismatched external domains).

### 2.3 Duplicate Vectors and Cross-Split Contamination
- In the discrete 30-feature space, 5,269 rows share feature vectors with other records.
- When splitting randomly, identical feature vectors will inevitably appear in both development and test partitions.
- Importantly, 64 feature vectors possess conflicting labels (both legitimate and phishing), demonstrating that the feature set represents an abstraction of complex websites where coarse discrete states can co-occur in both classes.

---

## 3. Split Contamination Verification

- **Split Evaluated**: True
- **Index Intersection**: 0 (must be strictly 0)
- **Overlapping Unique Feature Vectors**: 1114
