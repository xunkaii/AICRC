# Step 2.5-3b — Normalization Feasibility Audit

**Scope.** This script audits whether common normalization strategies
are *feasible* in deployment and whether they preserve class signal
while reducing posture bias. **It is not a feature-selection step.**
No feature is adopted or rejected here.

## 1. What this audit covers (and intentionally does not)

- Features tested: `motion_range_acc_z`, `depth_proxy`, `bottom_recovery_slope_acc_z` — 3 only.
- Methods tested:
  - `raw` — original feature values
  - `posture_train_zscore` — per-posture mean/std fit on **train** rows only, applied to all splits
  - `posture_train_robust` — per-posture median / (IQR / 1.349) fit on **train** rows only, applied to all
  - `participant_zscore_upper_bound` — per-subject mean/std fit on all of that subject's reps

**Three-feature constraint is intentional.** Expanding to all candidate features at this stage would create a feature jungle before we know which normalization principle holds.

## 2. Why participant z-score is *upper bound only*

`participant_zscore_upper_bound` requires statistics computed from a
subject's own reps. For a brand-new user's **first rep**, no such
statistics exist; calibration data must be collected first.

- Therefore it is **not** a candidate for the main inference pipeline.
- It is included here as a *ceiling* on how much normalization could
  help if per-subject calibration were available.
- Replacing the main path with participant z-score would silently
  break new-user inference; the report is explicit about this so
  later readers do not re-introduce it as a default.

`posture_train_zscore` and `posture_train_robust` use **train**
statistics only and can be applied to a new user's first rep, given
the user's posture label, so they are the realistic main-pipeline
candidates here.

## 3. Compute success

- Rows: **9275**
- `normalization_compute_ok = True`: **9275** (100.00%)

Stats fits (for transparency):

| stat | scope | source split | usable for new-user first rep? |
|---|---|---|---|
| posture mean / std | per posture | train only | yes |
| posture median / IQR | per posture | train only | yes |
| participant mean / std | per subject | all splits | **no** (needs prior calibration) |

## 4. Class effect vs posture effect (η²)

Goal: after normalization, **class η² should stay ≈ raw**, while
**posture η² should decrease**. Posture-aware normalization is
expected to drive posture η² close to zero.

| feature | method | n | class η² | posture η² | class η² / raw | posture η² / raw |
|---|---|---|---|---|---|---|
| `motion_range_acc_z` | `raw` | 9275 | 0.1112 | 0.0607 | 1.000 | 1.000 |
| `motion_range_acc_z` | `posture_train_zscore` | 9275 | 0.1171 | 0.0003 | 1.053 | 0.005 |
| `motion_range_acc_z` | `posture_train_robust` | 9275 | 0.1165 | 0.0013 | 1.047 | 0.022 |
| `motion_range_acc_z` | `participant_zscore_upper_bound` | 9275 | 0.1574 | 0.0721 | 1.415 | 1.188 |
| `depth_proxy` | `raw` | 9275 | 0.0042 | 0.8623 | 1.000 | 1.000 |
| `depth_proxy` | `posture_train_zscore` | 9275 | 0.0307 | 0.0019 | 7.270 | 0.002 |
| `depth_proxy` | `posture_train_robust` | 9275 | 0.0303 | 0.0040 | 7.156 | 0.005 |
| `depth_proxy` | `participant_zscore_upper_bound` | 9275 | 0.0052 | 0.9130 | 1.239 | 1.059 |
| `bottom_recovery_slope_acc_z` | `raw` | 9275 | 0.0056 | 0.0924 | 1.000 | 1.000 |
| `bottom_recovery_slope_acc_z` | `posture_train_zscore` | 9275 | 0.0059 | 0.0002 | 1.058 | 0.002 |
| `bottom_recovery_slope_acc_z` | `posture_train_robust` | 9275 | 0.0058 | 0.0065 | 1.045 | 0.071 |
| `bottom_recovery_slope_acc_z` | `participant_zscore_upper_bound` | 9275 | 0.0058 | 0.0997 | 1.038 | 1.079 |

Reading guide:

- `class η² / raw ≈ 1.0` → class signal preserved.
- `posture η² / raw < 0.2` → posture bias largely removed.
- `class η² / raw < 0.5` → normalization scrubbed too much; danger signal.

## 5. Split robustness

- For `raw`: flag if (max − min) of split means > 20% of |overall mean|.
- For normalized methods: flag if absolute (max − min) of split means > 0.20 (z-score units).

| feature | method | rule | unstable? |
|---|---|---|---|
| `motion_range_acc_z` | `raw` | raw_relative_>0.20 | False |
| `motion_range_acc_z` | `posture_train_zscore` | normalized_absolute_>0.20 | False |
| `motion_range_acc_z` | `posture_train_robust` | normalized_absolute_>0.20 | False |
| `motion_range_acc_z` | `participant_zscore_upper_bound` | normalized_absolute_>0.20 | False |
| `depth_proxy` | `raw` | raw_relative_>0.20 | False |
| `depth_proxy` | `posture_train_zscore` | normalized_absolute_>0.20 | **True** |
| `depth_proxy` | `posture_train_robust` | normalized_absolute_>0.20 | **True** |
| `depth_proxy` | `participant_zscore_upper_bound` | normalized_absolute_>0.20 | False |
| `bottom_recovery_slope_acc_z` | `raw` | raw_relative_>0.20 | **True** |
| `bottom_recovery_slope_acc_z` | `posture_train_zscore` | normalized_absolute_>0.20 | False |
| `bottom_recovery_slope_acc_z` | `posture_train_robust` | normalized_absolute_>0.20 | **True** |
| `bottom_recovery_slope_acc_z` | `participant_zscore_upper_bound` | normalized_absolute_>0.20 | False |

Full table: `reports/normalization_feasibility_split_robustness.csv`

## 6. Confusion-class overlap (uncertainty signal)

- Groups examined:
  - `C1_C5_C6`: C1, C5, C6
  - `C3_C4`: C3, C4
- Per (feature, method, posture-scope) and pair, we report Cohen's
  *d* and a Gaussian-approximation overlap estimate.
- **High overlap is not a reason to drop a feature.** In confusion
  groups (C1/C5/C6, C3/C4) classes are constructed to differ subtly,
  so substantial overlap is expected and is potentially a *basis for
  calibrated uncertainty* in downstream output (a candidate use, not
  a Step 2.5-3b decision).

Full table: `reports/normalization_feasibility_overlap.csv`

## 7. Posture as a *model input*, not just a normalizer

Even with posture-aware normalization, posture itself carries
information about the rep (which arms-position the subject was
instructed to use). Future modeling can pass posture as a categorical
input alongside the (raw or normalized) features — that option is
explicitly left open here. Normalization and posture-as-input are not
mutually exclusive.

## 8. What normalization is — and is not — for

- It is **not** a substitute for the model.
- It is for: (a) keeping caption-level interpretation reasonable
  across postures, and (b) keeping feature scale stable enough that
  downstream calibration / thresholding does not depend on the
  posture distribution of the batch.
- Anything beyond that is a modeling decision, not a normalization
  decision.

## 9. Next-step options (kept open, none chosen here)

1. `raw` features + posture as a model input.
2. `posture_train_zscore` features + posture as a model input.
3. `posture_train_robust` features + posture as a model input.
4. `participant_zscore_upper_bound` retained **only** as a calibration-ceiling reference; not a deployable default.

Each option needs to be evaluated against (a) class η² preservation,
(b) split robustness, (c) deployment feasibility for new users, and
(d) caption interpretability — the four lenses this audit set up.

## 10. (feature, method) pairs flagged unstable in §5

- `depth_proxy` × `posture_train_zscore`
- `depth_proxy` × `posture_train_robust`
- `bottom_recovery_slope_acc_z` × `raw`
- `bottom_recovery_slope_acc_z` × `posture_train_robust`
