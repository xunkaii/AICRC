"""Step 4 (v2split) — Build modeling dataset under the v2 participant split.

This is a duplicate of build_step4_modeling_dataset.py with the following
deliberate differences and NOTHING else:
    1. Input manifest -> data/manifest_split_v2.csv
       (v2: 34 train / 4 val / 14 test)
    2. Output directory -> data/step4_v2split/
    3. posture_train_zscore is recomputed FROM RAW for all 5 main features
       using the v2 train rows. The two columns pre-computed in
       data/step2/normalization_feasibility_features.csv (which are v1-train
       statistics) are NOT reused, because they encode the v1 split's
       train mean/std and would leak v1 train identities into v2.

The original build_step4_modeling_dataset.py is read-only and unmodified.
Step 2 CSVs (candidate_feature_bank.csv, bottom_event_audit.csv) are
read-only and unmodified. Only `data/step2/normalization_feasibility_features.csv`
is skipped (its pre-computed zscore columns are stale under v2).

Reads (read-only):
    data/manifest_split_v2.csv
    data/step2/candidate_feature_bank.csv
    data/step2/bottom_event_audit.csv

Writes:
    data/step4_v2split/step4_modeling_dataset.csv

Run:
    python scripts/build_step4_modeling_dataset_v2split.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILES = {
    "manifest_split": PROJECT_ROOT / "data" / "manifest_split_v2.csv",
    "candidate_feature_bank": PROJECT_ROOT / "data" / "step2" / "candidate_feature_bank.csv",
    "bottom_event_audit": PROJECT_ROOT / "data" / "step2" / "bottom_event_audit.csv",
}

OUTPUT_DIR = PROJECT_ROOT / "data" / "step4_v2split"
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

# Under v2split, zscore is recomputed from raw for ALL 5 features using v2
# train rows. v1's pre-computed zscore in normalization_feasibility_features.csv
# is intentionally ignored (it would leak v1 train identities).
NEEDS_ZSCORE_COMPUTE = list(MAIN_FEATURES)

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
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"[load] {name}: {len(df)} rows  (path={path})")
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
    """Per-posture, fit mu/sigma on TRAIN-split rows; apply to all rows."""
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
            f"zscore for '{feature}' produced {n_na} NaN values."
        )
    return out, stats


def _validate_finite(df: pd.DataFrame, cols: list, ctx: str) -> None:
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
            f"expected {EXPECTED_ROW_COUNT}, got {len(merged)}."
        )
    return merged


def main() -> int:
    print("=" * 64)
    print("Step 4 (v2split) - Build modeling dataset")
    print("=" * 64)

    manifest = _load_csv("manifest_split_v2", INPUT_FILES["manifest_split"])
    feature_bank = _load_csv(
        "candidate_feature_bank", INPUT_FILES["candidate_feature_bank"]
    )
    bottom_event = _load_csv(
        "bottom_event_audit", INPUT_FILES["bottom_event_audit"]
    )

    for name, df in [
        ("manifest_split_v2", manifest),
        ("candidate_feature_bank", feature_bank),
        ("bottom_event_audit", bottom_event),
    ]:
        if len(df) != EXPECTED_ROW_COUNT:
            raise ValueError(
                f"Row count mismatch in '{name}': "
                f"expected {EXPECTED_ROW_COUNT}, got {len(df)}."
            )
        _check_unique_key(df, JOIN_KEY, name)

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
        "manifest_split_v2",
    )
    _check_required_columns(
        feature_bank,
        ["sample_id"] + MAIN_FEATURES + ["anchor_reliability", "anchor_type"],
        "candidate_feature_bank",
    )
    _check_required_columns(bottom_event, ["sample_id"], "bottom_event_audit")

    # 5. Build the base frame from manifest.
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

    # 6. Merge feature bank.
    fb_subset = feature_bank[
        ["sample_id"] + MAIN_FEATURES + ["anchor_reliability", "anchor_type"]
    ].copy()
    merged = _safe_merge(base, fb_subset, "candidate_feature_bank")

    # 7. Merge bottom_event_audit (key validation only).
    be_subset = bottom_event[["sample_id"]].copy()
    merged = _safe_merge(merged, be_subset, "bottom_event_audit")

    # 8. Compute posture_train_zscore for ALL 5 main features under v2 train.
    print()
    print("[zscore-v2] computing posture_train_zscore using v2 train rows:")
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

    # 9. Posture one-hot.
    merged["posture_SA"] = (merged["posture"] == "SA").astype(int)
    merged["posture_CA"] = (merged["posture"] == "CA").astype(int)
    merged["posture_HW"] = (merged["posture"] == "HW").astype(int)

    # 10. Final column ordering.
    output_cols = (
        ["sample_id", "rep_id", "participant_id", "class_id", "posture", "split"]
        + MAIN_FEATURES
        + [f"{f}_zscore" for f in MAIN_FEATURES]
        + ["posture_SA", "posture_CA", "posture_HW"]
        + ["anchor_reliability", "anchor_type"]
    )
    out = merged[output_cols].copy()

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
            f"Forbidden columns present in output: {forbidden_present}."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print()
    print("=" * 64)
    print("Summary (v2split)")
    print("=" * 64)
    print(f"output rows                  : {len(out)}")
    print(f"split row counts             : {out_split_counts}")
    print(f"output columns ({len(out.columns)}): saved to {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
