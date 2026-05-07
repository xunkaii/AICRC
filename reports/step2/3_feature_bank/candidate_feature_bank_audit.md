# Step 2.5-3 — Candidate Feature Bank Audit

**Scope of this script.** This is a candidate-feature *audit*, not a
feature-selection commit. Features are computed and described here so
they can be evaluated downstream. Nothing on this list is adopted or
rejected by this run.

## 1. Posture-aware candidate anchor rule

Driven by Step 2.5-2 results (HW posture had very low bottom
agreement under the gyro-magnitude peak rule).

| posture | anchor source | anchor_type |
|---|---|---|
| SA | `ensemble_bottom_idx` (mean of acc/gyro candidates) | `ensemble_acc_gyro` |
| CA | `ensemble_bottom_idx` (mean of acc/gyro candidates) | `ensemble_acc_gyro` |
| HW | `acc_bottom_idx` (acc_z minimum only)               | `acc_only_anchor` |

**This rule is provisional**, not the final modeling rule. It is
used here only to compute candidate features in a way that does not
collapse under HW's weak gyro-peak signal.

Reasoning recap:
- SA / CA had cell-level bottom_agreement_ratio ≤ 0.10 in 70–88% of
  reps, so the ensemble of acc/gyro candidates is stable.
- HW had agreement only 4–22% with the same rule. The acc_z minimum
  alone is more reliable for HW because hands resting on the waist
  decouple |gyro| peaks from the squat bottom.

## 2. Candidate features

| feature | interpretation | caveat |
|---|---|---|
| `depth_proxy` | Mean of smoothed acc_z in [anchor±5]. Lower (relative to gravity) suggests deeper bottom or more orientation tilt at the bottom of the rep. | Calibration-free: this is *not* mm of squat depth; only an orientation-proxy useful for relative comparisons within the same subject. |
| `motion_range_acc_z` | Robust range (p95 − p5) of acc_z over the whole rep. Larger means the rep traverses a wider vertical-component range. | Sensor mount orientation matters; cross-subject comparison is approximate. |
| `motion_range_gyro_mag` | Robust range (p95 − p5) of \|gyro\| over the whole rep. Larger means more rotational motion through the rep. | Posture changes how arms swing — values are not directly comparable across SA/CA/HW. |
| `bottom_stability_acc` | RMS of per-axis std for (acc_x, acc_y, acc_z) in [anchor±10]. Smaller = stiller hold at bottom. | Smaller windows or shorter reps may bias this feature; flagged in feature_compute_warning when window is truncated. |
| `bottom_stability_gyro` | Std of \|gyro\| in [anchor±10]. Smaller = less rotational jitter at bottom. | Sensitive to small jitters; should be read alongside anchor_reliability. |
| `bottom_recovery_slope_acc_z` | Linear slope of smoothed acc_z over [anchor+10, anchor+16). Captures post-bottom recovery dynamics; motivated by Step 2.5-2 η²(acc_z) peaking *after* the bottom. | Sign and magnitude depend on the IMU mounting axis; the slope is interpretable in relative terms only. |
| `bottom_transition_delta_acc_z` | Mean acc_z(post-anchor 10 samples) − mean acc_z(pre-anchor 10 samples). Signed transition magnitude across the bottom. | Conflates depth and orientation change. Pair with depth_proxy before drawing causal conclusions. |
| `lateral_proxy_gyro` | mean(\|gyro_x\|) + mean(\|gyro_y\|) in [anchor±10]. **Weak proxy** for lateral / valgus motion: the IMU does not measure knee angle directly, so this is a heuristic, not a knee-valgus measurement. | Marked **weak proxy**. A high value does not imply knee valgus; it only indicates lateral/yaw rotation around the bottom. |
| `anchor_reliability` | [0..1] confidence in the anchor itself. Ensemble (SA/CA): linearly decreases with bottom_agreement_ratio up to 0.30. Acc-only (HW): 1.0 if no acc warning, 0.5 if acc_min_outside_search_range was flagged. | Currently a heuristic score; thresholds (0.30 cap, 0.5 fallback) are not validated. |

**On `bottom_recovery_slope_acc_z`.** Step 2.5-2 reported that η²(acc_z) for SA and CA peaked *near or after* the bottom (timesteps
≈ 86 of 128). That is a hint that *post-bottom recovery dynamics* —
not just bottom depth — carry class signal. The slope feature is a
first stab at exposing that to downstream evaluation.

## 3. Compute success

- Manifest rows missing from bottom audit (skipped): 0
- Bank rows (feature attempts): **9275**
- Rows with `feature_compute_ok = True`: **9275** (100.00%)

Per posture:

| posture | rows | compute_ok | rate |
|---|---|---|---|
| SA | 3098 | 3098 | 1.0000 |
| CA | 3080 | 3080 | 1.0000 |
| HW | 3097 | 3097 | 1.0000 |

Per class:

| class | rows | compute_ok | rate |
|---|---|---|---|
| C1 | 1553 | 1553 | 1.0000 |
| C2 | 1555 | 1555 | 1.0000 |
| C3 | 1529 | 1529 | 1.0000 |
| C4 | 1540 | 1540 | 1.0000 |
| C5 | 1549 | 1549 | 1.0000 |
| C6 | 1549 | 1549 | 1.0000 |

## 4. NaN counts per feature

| feature | n_nan | rate |
|---|---|---|
| `depth_proxy` | 0 | 0.0000 |
| `motion_range_acc_z` | 0 | 0.0000 |
| `motion_range_gyro_mag` | 0 | 0.0000 |
| `bottom_stability_acc` | 0 | 0.0000 |
| `bottom_stability_gyro` | 0 | 0.0000 |
| `bottom_recovery_slope_acc_z` | 0 | 0.0000 |
| `bottom_transition_delta_acc_z` | 0 | 0.0000 |
| `lateral_proxy_gyro` | 0 | 0.0000 |
| `anchor_reliability` | 0 | 0.0000 |

## 5. Anchor-type distribution

| posture | anchor_type | count |
|---|---|---|
| SA | `ensemble_acc_gyro` | 3098 |
| CA | `ensemble_acc_gyro` | 3080 |
| HW | `acc_only_anchor` | 3097 |

## 6. Distribution / robustness / overlap (cross-references)

- `reports/candidate_feature_summary_by_class_posture.csv` — per (class × posture × feature) `n`, `n_nan`, `mean`, `std`, `median`, `p25`, `p75`.
- `reports/candidate_feature_split_robustness.csv` — per feature, `(train, val, test)` mean/std and `flag_unstable` (set when max-min mean range exceeds 20% of |overall mean|).
- `reports/candidate_feature_uncertainty_overlap.csv` — per confusion group / feature / posture-scope: pairwise Cohen's *d* and an overlap estimate (Gaussian approximation, useful as a *signal* not a probability).

Confusion groups examined:

- `C1_C5_C6`: C1, C5, C6
- `C3_C4`: C3, C4

**Why we keep features that overlap heavily.** Class C1/C5/C6 and C3/C4 are expected to overlap by construction (they share a posture-
and-task structure with subtle differences). High overlap on a
feature is *not a reason to drop it*; it is potentially a basis for
calibrated uncertainty in caption/output, which Step 2.5-3 only flags
as a candidate use case for downstream consideration.

## 7. Open questions / things this audit explicitly does NOT decide

- Which features survive into modeling.
- Whether `lateral_proxy_gyro` (a weak proxy) should be dropped or
  kept as an uncertainty-only feature.
- Whether `anchor_reliability` thresholds are correctly calibrated.
- Whether the HW acc-only fallback should be replaced by a different
  anchoring strategy entirely.
- Whether per-feature normalization (per-subject / per-posture) is
  needed before any downstream use.

These are all explicitly **next-step** decisions, not Step 2.5-3 ones.
