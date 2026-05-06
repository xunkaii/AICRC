"""Step 4 — Build modeling dataset.

Reads (read-only):
    data/manifest_split.csv
    data/step2/candidate_feature_bank.csv
    data/step2/normalization_feasibility_features.csv
    data/step2/bottom_event_audit.csv

Writes:
    data/step4/step4_modeling_dataset.csv

Behavior locked by:
    reports/step4/step4_feature_model_decision_memo.md
    reports/step4/step4_modeling_calibration_plan.md
    reports/step3/step3_output_schema_uncertainty_policy.md

Notes:
    - This script does not compute new physical/temporal features.
      For the three main features whose train-split posture-conditional
      z-score is not pre-computed in normalization_feasibility_features.csv
      (motion_range_gyro_mag, bottom_stability_acc,
      bottom_transition_delta_acc_z), the SAME posture_train_zscore
      methodology established in Step 2.5-3b is applied — i.e. for
      each posture, fit mean/std on TRAIN-split rows only and apply
      (x - mu) / sigma to all rows in that posture.
    - No model is trained, no threshold is calibrated, no caption is
      generated.
    - participant_zscore variants, bottom_recovery_slope_acc_z, and
      lateral_proxy_gyro are explicitly excluded from the output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILES = {
    "manifest_split": PROJECT_ROOT / "data" / "manifest_split.csv",
    "candidate_feature_bank": PROJECT_ROOT / "data" / "step2" / "candidate_feature_bank.csv",
    "normalization_feasibility_features": (
        PROJECT_ROOT / "data" / "step2" / "normalization_feasibility_features.csv"
    ),
    "bottom_event_audit": PROJECT_ROOT / "data" / "step2" / "bottom_event_audit.csv",
}

OUTPUT_DIR = PROJECT_ROOT / "data" / "step4"
OUTPUT_FILE = OUTPUT_DIR / "step4_modeling_dataset.csv"

EXPECTED_ROW_COUNT = 9275
JOIN_KEY = "sample_id"
POSTURES = ["SA", "CA", "HW"]

MAIN_FEATURES = [
    "motion_range_acc_z",
    "depth_proxy",
    "motion_range_gyro_mag",
    "bottom_stability_acc",
    "bottom_transition_delta_acc_z",
]

PREEXISTING_ZSCORE_SOURCES = {
    "motion_range_acc_z": "posture_train_zscore_motion_range_acc_z",
    "depth_proxy": "posture_train_zscore_depth_proxy",
}

NEEDS_ZSCORE_COMPUTE = [
    "motion_range_gyro_mag",
    "bottom_stability_acc",
    "bottom_transition_delta_acc_z",
]

FORBIDDEN_OUTPUT_TOKENS = [
    "participant_zscore",
    "bottom_recovery_slope_acc_z",
    "lateral_proxy_gyro",
]

ANCHOR_CONSISTENCY_COLS = [
    "anchor_reliability",
    "anchor_type",
    "anchor_identity",
    "ensemble_bottom_idx",
    "acc_bottom_idx",
    "gyro_bottom_idx",
    "bottom_agreement_ratio",
]

ALLOWED_CLASS_IDS = ["C1", "C2", "C3", "C4", "C5", "C6"]
ALLOWED_POSTURES = ["SA", "CA", "HW"]
ALLOWED_SPLITS = ["train", "val", "test"]


def _load_csv(name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input '{name}' not found at: {path}"
        )
    df = pd.read_csv(path)
    print(f"[load] {name}: {len(df)} rows  (path={path})")
    print(f"       columns ({len(df.columns)}): {list(df.columns)}")
    return df


def _check_unique_key(df: pd.DataFrame, key: str, name: str) -> None:
    if key not in df.columns:
        raise ValueError(
            f"Join key '{key}' missing from {name}. columns: {list(df.columns)}"
        )
    n_dup = int(df[key].duplicated().sum())
    if n_dup > 0:
        raise ValueError(
            f"Join key '{key}' has {n_dup} duplicate values in {name}."
        )


def _check_required_columns(df: pd.DataFrame, required: list, name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns missing from {name}: {missing}. "
            f"available columns: {list(df.columns)}"
        )


def _compute_posture_train_zscore(
    df: pd.DataFrame,
    feature: str,
    posture_col: str = "posture",
    split_col: str = "split",
) -> tuple[pd.Series, dict]:
    """Apply Step 2.5-3b's posture_train_zscore methodology to `feature`.

    For each posture, fit mu/sigma on TRAIN-split rows only and apply
    (x - mu) / sigma to all rows in that posture. This is the same
    transformation used in normalization_feasibility_features.csv for
    the three features that file already covers; applied here to the
    remaining main features named by the Step 4 decision memo.

    Returns:
        (zscore_series, posture_stats) where posture_stats[posture] = {
            "n_train": int, "mu": float, "sigma": float, "n_applied": int
        }
    """
    out = pd.Series(index=df.index, dtype="float64")
    stats: dict = {}
    for posture in POSTURES:
        train_mask = (df[posture_col] == posture) & (df[split_col] == "train")
        if int(train_mask.sum()) == 0:
            raise ValueError(
                f"No train-split rows for posture={posture}; "
                f"cannot fit zscore for '{feature}'."
            )
        mu = df.loc[train_mask, feature].mean()
        sigma = df.loc[train_mask, feature].std(ddof=0)
        if sigma == 0 or pd.isna(sigma):
            raise ValueError(
                f"Train-split sigma for posture={posture}, feature='{feature}' "
                f"is zero or NaN; cannot zscore."
            )
        posture_mask = df[posture_col] == posture
        out.loc[posture_mask] = (df.loc[posture_mask, feature] - mu) / sigma
        stats[posture] = {
            "n_train": int(train_mask.sum()),
            "mu": float(mu),
            "sigma": float(sigma),
            "n_applied": int(posture_mask.sum()),
        }
    if out.isna().any():
        n_na = int(out.isna().sum())
        raise ValueError(
            f"zscore for '{feature}' produced {n_na} NaN values "
            f"(rows not assigned to any of postures {POSTURES})."
        )
    return out, stats


def _check_anchor_consistency(
    feature_bank: pd.DataFrame,
    bottom_event: pd.DataFrame,
) -> None:
    """Verify anchor-related columns agree between feature_bank and
    bottom_event_audit on every sample_id. bottom_event_audit is the
    preferred source per Step 4 build policy.
    """
    fb_has = [c for c in ANCHOR_CONSISTENCY_COLS if c in feature_bank.columns]
    be_has = [c for c in ANCHOR_CONSISTENCY_COLS if c in bottom_event.columns]
    print(f"[anchor-consistency] feature_bank has: {fb_has}")
    print(f"[anchor-consistency] bottom_event_audit has: {be_has}")
    overlap = [c for c in fb_has if c in be_has]
    if not overlap:
        print(
            "[anchor-consistency] no overlapping anchor-related columns "
            "between the two sources; skipping per-row conflict check."
        )
        return
    print(f"[anchor-consistency] overlap (will be conflict-checked): {overlap}")
    fb = feature_bank[[JOIN_KEY] + overlap].copy()
    be = bottom_event[[JOIN_KEY] + overlap].copy()
    cmp = fb.merge(be, on=JOIN_KEY, suffixes=("__fb", "__be"), validate="one_to_one")
    if len(cmp) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Anchor consistency merge row count {len(cmp)} != {EXPECTED_ROW_COUNT}."
        )
    for col in overlap:
        fb_vals = cmp[f"{col}__fb"]
        be_vals = cmp[f"{col}__be"]
        if pd.api.types.is_numeric_dtype(fb_vals) and pd.api.types.is_numeric_dtype(be_vals):
            both_nan = fb_vals.isna() & be_vals.isna()
            equal = (fb_vals == be_vals) | both_nan
        else:
            equal = (
                fb_vals.fillna("__nan__").astype(str)
                == be_vals.fillna("__nan__").astype(str)
            )
        n_mismatch = int((~equal).sum())
        if n_mismatch > 0:
            raise ValueError(
                f"Anchor consistency conflict on '{col}': {n_mismatch} rows "
                f"differ between candidate_feature_bank and bottom_event_audit. "
                f"bottom_event_audit is the preferred source per Step 4 build policy."
            )
        print(f"[anchor-consistency] '{col}': all {len(cmp)} rows match.")


def _validate_finite(df: pd.DataFrame, cols: list, ctx: str) -> None:
    """Raise if any of `cols` has NaN, +inf, or -inf."""
    issues: dict = {}
    for c in cols:
        s = df[c]
        n_na = int(s.isna().sum())
        if pd.api.types.is_numeric_dtype(s):
            n_inf = int(np.isinf(s.to_numpy(dtype="float64")).sum())
        else:
            n_inf = 0
        if n_na or n_inf:
            issues[c] = {"nan": n_na, "inf": n_inf}
    if issues:
        raise ValueError(f"Non-finite values in {ctx}: {issues}")


def _validate_in_set(df: pd.DataFrame, col: str, allowed: list, ctx: str) -> None:
    if df[col].isna().any():
        n = int(df[col].isna().sum())
        raise ValueError(f"{ctx}: column '{col}' has {n} NaN values.")
    actual = set(df[col].unique().tolist())
    extras = sorted(actual - set(allowed))
    if extras:
        raise ValueError(
            f"{ctx}: column '{col}' has disallowed values {extras}; "
            f"allowed = {sorted(allowed)}."
        )


def _validate_anchor_reliability(df: pd.DataFrame, col: str = "anchor_reliability") -> None:
    s = df[col]
    n_na = int(s.isna().sum())
    if n_na > 0:
        raise ValueError(f"'{col}' has {n_na} NaN values.")
    if not pd.api.types.is_numeric_dtype(s):
        raise ValueError(f"'{col}' must be numeric, got dtype={s.dtype}.")
    arr = s.to_numpy(dtype="float64")
    n_inf = int(np.isinf(arr).sum())
    if n_inf > 0:
        raise ValueError(f"'{col}' has {n_inf} +/-inf values.")
    out_of_range = (arr < 0.0) | (arr > 1.0)
    n_oor = int(out_of_range.sum())
    if n_oor > 0:
        raise ValueError(
            f"'{col}' has {n_oor} rows outside [0, 1]; "
            f"min={float(arr.min())}, max={float(arr.max())}."
        )


def _safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    right_name: str,
) -> pd.DataFrame:
    merged = left.merge(right, on=JOIN_KEY, how="inner", validate="one_to_one")
    if len(merged) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Row count mismatch after merging '{right_name}': "
            f"expected {EXPECTED_ROW_COUNT}, got {len(merged)}. "
            f"Inconsistent join keys originated from '{right_name}'."
        )
    return merged


def main() -> int:
    print("=" * 64)
    print("Step 4 - Build modeling dataset")
    print("=" * 64)

    # 1. Load all inputs (column lists are printed by _load_csv).
    manifest = _load_csv("manifest_split", INPUT_FILES["manifest_split"])
    feature_bank = _load_csv(
        "candidate_feature_bank", INPUT_FILES["candidate_feature_bank"]
    )
    norm_features = _load_csv(
        "normalization_feasibility_features",
        INPUT_FILES["normalization_feasibility_features"],
    )
    bottom_event = _load_csv(
        "bottom_event_audit", INPUT_FILES["bottom_event_audit"]
    )

    # 2. Validate raw row counts.
    for name, df in [
        ("manifest_split", manifest),
        ("candidate_feature_bank", feature_bank),
        ("normalization_feasibility_features", norm_features),
        ("bottom_event_audit", bottom_event),
    ]:
        if len(df) != EXPECTED_ROW_COUNT:
            raise ValueError(
                f"Row count mismatch in '{name}': "
                f"expected {EXPECTED_ROW_COUNT}, got {len(df)}."
            )

    # 3. Verify join key presence and uniqueness in all four inputs.
    for name, df in [
        ("manifest_split", manifest),
        ("candidate_feature_bank", feature_bank),
        ("normalization_feasibility_features", norm_features),
        ("bottom_event_audit", bottom_event),
    ]:
        _check_unique_key(df, JOIN_KEY, name)

    # 4. Verify required source columns.
    _check_required_columns(
        manifest,
        [
            "sample_id",
            "rep_id",
            "participant_id",
            "class_id",
            "posture_canonical",
            "split",
        ],
        "manifest_split",
    )
    _check_required_columns(
        feature_bank,
        ["sample_id"] + MAIN_FEATURES + ["anchor_reliability", "anchor_type"],
        "candidate_feature_bank",
    )
    _check_required_columns(
        norm_features,
        ["sample_id"] + list(PREEXISTING_ZSCORE_SOURCES.values()),
        "normalization_feasibility_features",
    )
    _check_required_columns(bottom_event, ["sample_id"], "bottom_event_audit")

    # 4b. Anchor consistency: verify columns shared between feature_bank
    #     and bottom_event_audit agree on every row. bottom_event_audit
    #     is the preferred source for any conflict involving anchor info.
    _check_anchor_consistency(feature_bank, bottom_event)

    # 5. Build the base frame from manifest (identifiers + label + split).
    base = manifest[
        [
            "sample_id",
            "rep_id",
            "participant_id",
            "class_id",
            "posture_canonical",
            "split",
        ]
    ].copy()
    base = base.rename(columns={"posture_canonical": "posture"})

    # 6. Merge feature bank (raw main features + anchor info).
    fb_subset = feature_bank[
        ["sample_id"] + MAIN_FEATURES + ["anchor_reliability", "anchor_type"]
    ].copy()
    merged = _safe_merge(base, fb_subset, "candidate_feature_bank")

    # 7. Merge pre-computed zscore for the two features that already have it.
    nf_subset = norm_features[
        ["sample_id"] + list(PREEXISTING_ZSCORE_SOURCES.values())
    ].copy()
    merged = _safe_merge(merged, nf_subset, "normalization_feasibility_features")

    # 8. Merge bottom_event_audit (key validation only — no extra columns).
    be_subset = bottom_event[["sample_id"]].copy()
    merged = _safe_merge(merged, be_subset, "bottom_event_audit")

    # 9. Rename the pre-computed zscore columns to the cleaner suffix style.
    for raw_name, prefixed in PREEXISTING_ZSCORE_SOURCES.items():
        merged[f"{raw_name}_zscore"] = merged[prefixed]
        merged = merged.drop(columns=[prefixed])

    # 10. Compute posture_train_zscore for the remaining main features.
    print()
    print("[zscore] computing posture_train_zscore for features without "
          "pre-computed zscore in normalization_feasibility_features.csv:")
    for feature in NEEDS_ZSCORE_COMPUTE:
        zs, stats = _compute_posture_train_zscore(merged, feature)
        merged[f"{feature}_zscore"] = zs
        print(f"  {feature}:")
        for posture in POSTURES:
            s = stats[posture]
            print(
                f"    posture={posture}  n_train={s['n_train']:>4}  "
                f"mu={s['mu']:.6g}  sigma={s['sigma']:.6g}  "
                f"applied_to={s['n_applied']} rows"
            )

    # 11. Build posture one-hot columns.
    merged["posture_SA"] = (merged["posture"] == "SA").astype(int)
    merged["posture_CA"] = (merged["posture"] == "CA").astype(int)
    merged["posture_HW"] = (merged["posture"] == "HW").astype(int)

    # 12. Final column ordering.
    output_cols = (
        ["sample_id", "rep_id", "participant_id", "class_id", "posture", "split"]
        + MAIN_FEATURES
        + [f"{f}_zscore" for f in MAIN_FEATURES]
        + ["posture_SA", "posture_CA", "posture_HW"]
        + ["anchor_reliability", "anchor_type"]
    )
    out = merged[output_cols].copy()

    # 13. Final validations.
    if len(out) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Final output row count mismatch: "
            f"expected {EXPECTED_ROW_COUNT}, got {len(out)}."
        )

    manifest_split_counts = manifest["split"].value_counts().sort_index().to_dict()
    out_split_counts = out["split"].value_counts().sort_index().to_dict()
    if manifest_split_counts != out_split_counts:
        raise ValueError(
            f"Split row counts disagree with manifest. "
            f"manifest: {manifest_split_counts}, output: {out_split_counts}."
        )

    feature_cols = MAIN_FEATURES + [f"{f}_zscore" for f in MAIN_FEATURES]
    _validate_finite(out, feature_cols, ctx="main raw + zscore feature columns")

    _validate_in_set(out, "class_id", ALLOWED_CLASS_IDS, ctx="output dataset")
    _validate_in_set(out, "posture", ALLOWED_POSTURES, ctx="output dataset")
    _validate_in_set(out, "split", ALLOWED_SPLITS, ctx="output dataset")

    _validate_anchor_reliability(out, col="anchor_reliability")

    onehot_sum = out[["posture_SA", "posture_CA", "posture_HW"]].sum(axis=1)
    n_onehot_bad = int((onehot_sum != 1).sum())
    if n_onehot_bad > 0:
        raise ValueError(
            f"Posture one-hot row-sum != 1 in {n_onehot_bad} rows."
        )

    forbidden_present = [
        c
        for c in out.columns
        if any(token in c for token in FORBIDDEN_OUTPUT_TOKENS)
    ]
    if forbidden_present:
        raise ValueError(
            f"Forbidden columns present in output: {forbidden_present}. "
            f"forbidden tokens: {FORBIDDEN_OUTPUT_TOKENS}."
        )

    # 14. Save.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)

    # 15. Console summary.
    print()
    print("=" * 64)
    print("Summary")
    print("=" * 64)
    ar_min = float(out["anchor_reliability"].min())
    ar_max = float(out["anchor_reliability"].max())
    print(f"output rows                  : {len(out)}")
    print(f"split row counts             : {out_split_counts}")
    print(f"feature finite check         : passed (no NaN/+-inf across {len(feature_cols)} cols)")
    print(f"class_id values              : passed (subset of {ALLOWED_CLASS_IDS})")
    print(f"posture values               : passed (subset of {ALLOWED_POSTURES})")
    print(f"split values                 : passed (subset of {ALLOWED_SPLITS})")
    print(
        f"anchor_reliability check     : passed "
        f"(no NaN/+-inf, range=[{ar_min:.4f}, {ar_max:.4f}] within [0, 1])"
    )
    print(f"posture one-hot check        : passed (all row sums == 1)")
    print(
        "forbidden column check       : passed "
        f"(no token from {FORBIDDEN_OUTPUT_TOKENS} appears)"
    )
    print(f"output columns ({len(out.columns)}):")
    for c in out.columns:
        print(f"  - {c}")
    print(f"saved to                   : {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
