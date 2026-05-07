# Step 2.5-4 — Reference Separability Audit

**Scope.** This is a *reference separability audit*, not a final
model. The goal is to ask how much of C1..C6 structure is captured
by a small candidate feature set under three normalization regimes.
Nothing on this list is adopted, rejected, or tuned for performance.

## 1. Settings compared

| setting | continuous features | normalization | posture input |
|---|---|---|---|
| `raw_core_features` | motion_range_acc_z, depth_proxy, bottom_recovery_slope_acc_z | none | one-hot SA/CA/HW |
| `posture_train_zscore_core_features` | same three features | per-posture z-score (train fit) | one-hot SA/CA/HW |
| `posture_train_robust_core_features` | same three features | per-posture median/IQR (train fit) | one-hot SA/CA/HW |

**Why participant z-score is excluded.** It needs prior reps from
the new user (calibration) and cannot run on a first rep, so it is
not a deployable main-pipeline candidate. Step 2.5-3b kept it as a
ceiling reference; this audit does not train it.

**Why posture-train normalizations are kept.** Both fit statistics
on **train rows only** and apply train statistics to val/test, so
they generalize to a new user's first rep given the user's posture.

**Posture as model input.** All three settings include posture as a
3-dim one-hot input. Normalization and posture-as-input are not
mutually exclusive; both can carry signal.

## 2. Reference model

- Multinomial logistic regression, L2-regularized, balanced class
  weights, random_state=42.
- StandardScaler fit on **train** features only, applied to all splits.
- This is a *reference separability* model. It is intentionally
  simple. Numbers below should not be read as final-model
  performance estimates.

**sklearn fallback note.** The MINGW-W64 numpy in this environment
aborts on `import sklearn`. The model is therefore implemented via
`scipy.optimize.minimize(method='L-BFGS-B')` with an explicit
multinomial-CE + L2 objective, equivalent to sklearn's
`LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight='balanced')`.

## 3. Headline metrics (val / test)

| setting | split | accuracy | macro F1 | weighted F1 |
|---|---|---|---|---|
| `raw_core_features` | val | 0.2354 | 0.2076 | 0.2079 |
| `raw_core_features` | test | 0.2572 | 0.2274 | 0.2275 |
| `posture_train_zscore_core_features` | val | 0.2416 | 0.2104 | 0.2107 |
| `posture_train_zscore_core_features` | test | 0.2628 | 0.2290 | 0.2291 |
| `posture_train_robust_core_features` | val | 0.2423 | 0.2117 | 0.2121 |
| `posture_train_robust_core_features` | test | 0.2635 | 0.2313 | 0.2314 |

## 4. Ambiguous-group breakdown (test split)

Definitions (reused from prior audits):

- `high_confusion_group` = C1, C5, C6 — share a normal/single/both-knee
  pattern and are expected to overlap.
- `medium_confusion_pair` = C3, C4 — distinct but related faults.
- `clean_reference_class` = C2 — should be most cleanly identified.

| setting | C1/C5/C6 internal confusion | C3/C4 pair confusion | C2 recall |
|---|---|---|---|
| `raw_core_features` | 0.2727 | 0.1099 | 0.7322 |
| `posture_train_zscore_core_features` | 0.2727 | 0.1078 | 0.7657 |
| `posture_train_robust_core_features` | 0.2685 | 0.1057 | 0.7531 |

Reading guide:

- `C1/C5/C6 internal confusion` is the share of C1∪C5∪C6 reps that
  the reference model assigns *to a different class within the same
  group*. High values mean the features cannot pull these three
  classes apart — which is exactly the expected difficulty.
- `C3/C4 pair confusion` is the share of C3∪C4 reps misrouted to
  the *other* class in the pair.
- `C2 recall` is the share of C2 reps correctly identified.

## 5. Per-class breakdown

Full table: `reports/reference_separability_metrics.csv`. Per-
posture macro F1 is also in that file (`scope=posture`).

## 6. Confusion matrices

Full table (long format, all settings × splits): `reports/reference_separability_confusion_by_setting.csv`.

## 7. Does normalization help?

Compare the headline rows for the same split across settings.
Step 2.5-3b had already shown that:

- `depth_proxy` raw is dominated by posture (η²≈0.86 on posture vs
  ≈0.004 on class), and posture-aware z-score lifts class η² to
  ≈0.03 while collapsing posture η² to ≈0.

Whether that translates to better separability under a model is
what this audit measures. **Direction**, not magnitude, is the
signal here.

## 8. Implications for caption uncertainty

If `C1/C5/C6 internal confusion` and `C3/C4 pair confusion` remain
substantial across all three settings, that is *evidence* that
downstream sensor-to-text outputs should express uncertainty
between these classes, rather than always committing to a single
label. Step 2.5-4 only flags this; it does not design the caption.

## 9. What this audit does NOT decide

- Which features survive into modeling.
- Whether the final model should be logistic regression, a tree
  ensemble, a small MLP, or something else.
- Whether any normalization scheme is the official choice.
- The numerical performance bar to clear; numbers here are a
  reference, not a target.
