"""Step 8-2 — Pilot human review package preparation.

Reads (read-only):
    data/step8/step8_review_sample_raw.csv
    data/step8/step8_review_sample_zscore.csv
    data/step8/step8_review_sample_paired.csv

Writes:
    data/step8/step8_pilot_review_raw_blinded.csv
    data/step8/step8_pilot_review_zscore_blinded.csv
    data/step8/step8_pilot_review_paired_blinded.csv
    data/step8/step8_pilot_review_audit_key_raw.csv
    data/step8/step8_pilot_review_audit_key_zscore.csv
    data/step8/step8_pilot_review_audit_key_paired.csv

Behavior locked by:
    reports/step8/step8_caption_evaluation_plan.md
    reports/step8/step8_review_sample_summary.md

This script does NOT perform human review, does NOT fill reviewer
ratings, does NOT modify captions, does NOT regenerate captions, does
NOT retrain models, does NOT recalibrate thresholds, does NOT regenerate
schema outputs, does NOT compute new physical features, and does NOT
pick a final raw vs zscore branch. All input CSVs are read-only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "step8"

INPUT_FILES = {
    "raw": DATA_DIR / "step8_review_sample_raw.csv",
    "zscore": DATA_DIR / "step8_review_sample_zscore.csv",
    "paired": DATA_DIR / "step8_review_sample_paired.csv",
}
BLINDED_FILES = {
    "raw": DATA_DIR / "step8_pilot_review_raw_blinded.csv",
    "zscore": DATA_DIR / "step8_pilot_review_zscore_blinded.csv",
    "paired": DATA_DIR / "step8_pilot_review_paired_blinded.csv",
}
AUDIT_KEY_FILES = {
    "raw": DATA_DIR / "step8_pilot_review_audit_key_raw.csv",
    "zscore": DATA_DIR / "step8_pilot_review_audit_key_zscore.csv",
    "paired": DATA_DIR / "step8_pilot_review_audit_key_paired.csv",
}

EXPECTED_ROW_COUNTS = {"raw": 140, "zscore": 142, "paired": 150}

BRANCH_REQUIRED_COLS = [
    "review_id", "branch", "sample_id", "split", "posture",
    "class_id_for_audit_only", "caption_ko", "template_id",
    "class_set_prediction", "uncertainty_flags",
    "caption_confidence_level", "no_call", "no_call_reason",
    "main_feature_phrase", "uncertainty_phrase",
    "anchor_reliability_bin", "anchor_unreliable",
    "feature_phrase_source", "feature_phrase_fallback_used",
    "caption_validation_pass", "caption_length_chars",
    "review_priority_tag",
    "rating_policy_compliance", "rating_schema_faithfulness",
    "rating_naturalness", "rating_non_overclaiming",
    "rating_usefulness", "rating_feature_phrase_appropriateness",
    "branch_preference", "issue_tags", "free_text_comment",
    "reviewer_decision",
]

PAIRED_REQUIRED_COLS = [
    "review_id", "sample_id", "split", "posture",
    "class_id_for_audit_only",
    "raw_caption_ko", "zscore_caption_ko",
    "raw_template_id", "zscore_template_id",
    "raw_class_set_prediction", "zscore_class_set_prediction",
    "raw_caption_confidence_level", "zscore_caption_confidence_level",
    "raw_no_call", "zscore_no_call",
    "raw_no_call_reason", "zscore_no_call_reason",
    "raw_main_feature_phrase", "zscore_main_feature_phrase",
    "captions_equal", "branch_difference_type",
    "reviewer_branch_preference", "paired_issue_tags",
    "paired_free_text_comment",
]

BRANCH_REVIEWER_COLS = [
    "rating_policy_compliance", "rating_schema_faithfulness",
    "rating_naturalness", "rating_non_overclaiming",
    "rating_usefulness", "rating_feature_phrase_appropriateness",
    "branch_preference", "issue_tags", "free_text_comment",
    "reviewer_decision",
]
PAIRED_REVIEWER_COLS = [
    "reviewer_branch_preference", "paired_issue_tags",
    "paired_free_text_comment",
]

BRANCH_BLINDED_COL_ORDER = [
    # metadata
    "review_id", "branch", "sample_id", "split", "posture",
    # caption context
    "caption_ko", "template_id", "class_set_prediction",
    "uncertainty_flags", "caption_confidence_level", "no_call",
    "no_call_reason", "main_feature_phrase", "uncertainty_phrase",
    "anchor_reliability_bin", "anchor_unreliable",
    "feature_phrase_source", "feature_phrase_fallback_used",
    "caption_length_chars", "review_priority_tag",
    # reviewer input
    "rating_policy_compliance", "rating_schema_faithfulness",
    "rating_naturalness", "rating_non_overclaiming",
    "rating_usefulness", "rating_feature_phrase_appropriateness",
    "branch_preference", "issue_tags", "free_text_comment",
    "reviewer_decision",
]

PAIRED_BLINDED_COL_ORDER = [
    "review_id", "sample_id", "split", "posture",
    "raw_caption_ko", "zscore_caption_ko",
    "raw_template_id", "zscore_template_id",
    "raw_class_set_prediction", "zscore_class_set_prediction",
    "raw_caption_confidence_level", "zscore_caption_confidence_level",
    "raw_no_call", "zscore_no_call",
    "raw_no_call_reason", "zscore_no_call_reason",
    "raw_main_feature_phrase", "zscore_main_feature_phrase",
    "captions_equal", "branch_difference_type",
    "reviewer_branch_preference", "paired_issue_tags",
    "paired_free_text_comment",
]


def parse_bool(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)


def is_empty(v) -> bool:
    """Treat NaN, None, and empty/whitespace strings as empty."""
    if v is None:
        return True
    try:
        if isinstance(v, float) and np.isnan(v):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_branch(df: pd.DataFrame, kind: str) -> None:
    expected = EXPECTED_ROW_COUNTS[kind]
    if len(df) != expected:
        raise ValueError(f"{kind}: row count {len(df)} != {expected}")
    for col in BRANCH_REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"{kind}: missing required column {col}")
    if not df["review_id"].is_unique:
        raise ValueError(f"{kind}: review_id is not unique")
    if df["sample_id"].isna().any():
        raise ValueError(f"{kind}: sample_id has missing values")
    pass_series = df["caption_validation_pass"].apply(parse_bool)
    if not pass_series.all():
        raise ValueError(
            f"{kind}: caption_validation_pass has False rows "
            f"({(~pass_series).sum()} found)"
        )
    for col in BRANCH_REVIEWER_COLS:
        non_empty = df[col].apply(lambda v: not is_empty(v))
        if non_empty.any():
            raise ValueError(
                f"{kind}: reviewer column '{col}' has non-empty values "
                f"({int(non_empty.sum())} found) — pilot pack must keep "
                f"reviewer fields empty"
            )


def validate_paired(df: pd.DataFrame) -> None:
    expected = EXPECTED_ROW_COUNTS["paired"]
    if len(df) != expected:
        raise ValueError(f"paired: row count {len(df)} != {expected}")
    for col in PAIRED_REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"paired: missing required column {col}")
    if not df["review_id"].is_unique:
        raise ValueError("paired: review_id is not unique")
    if df["sample_id"].isna().any():
        raise ValueError("paired: sample_id has missing values")
    for col in PAIRED_REVIEWER_COLS:
        non_empty = df[col].apply(lambda v: not is_empty(v))
        if non_empty.any():
            raise ValueError(
                f"paired: reviewer column '{col}' has non-empty values "
                f"({int(non_empty.sum())} found)"
            )

    # captions_equal vs branch_difference_type consistency
    eq = df["captions_equal"].apply(parse_bool)
    bdt = df["branch_difference_type"].astype(str)
    eq_inconsistent_true = (eq & (bdt != "same_caption"))
    eq_inconsistent_false = ((~eq) & (bdt == "same_caption"))
    bad = int(eq_inconsistent_true.sum() + eq_inconsistent_false.sum())
    if bad > 0:
        raise ValueError(
            f"paired: captions_equal vs branch_difference_type "
            f"inconsistency in {bad} rows"
        )


# ---------------------------------------------------------------------------
# Blinded / audit-key extraction
# ---------------------------------------------------------------------------

def make_branch_blinded(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in BRANCH_BLINDED_COL_ORDER if c in df.columns]
    out = df[cols].copy()
    # Force reviewer columns to be empty (string ""), guarding against any
    # imported NaN that pandas might serialize as empty cells but not as "".
    for col in BRANCH_REVIEWER_COLS:
        out[col] = ""
    return out


def make_paired_blinded(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in PAIRED_BLINDED_COL_ORDER if c in df.columns]
    out = df[cols].copy()
    for col in PAIRED_REVIEWER_COLS:
        out[col] = ""
    return out


def make_branch_audit_key(df: pd.DataFrame) -> pd.DataFrame:
    return df[["review_id", "branch", "sample_id",
               "class_id_for_audit_only"]].copy()


def make_paired_audit_key(df: pd.DataFrame) -> pd.DataFrame:
    return df[["review_id", "sample_id",
               "class_id_for_audit_only"]].copy()


def assert_no_audit_column(df: pd.DataFrame, label: str) -> None:
    if "class_id_for_audit_only" in df.columns:
        raise ValueError(
            f"{label}: class_id_for_audit_only must NOT appear in blinded CSV"
        )


def assert_reviewer_cols_empty(
    df: pd.DataFrame, cols: list, label: str
) -> None:
    for col in cols:
        non_empty = df[col].apply(lambda v: not is_empty(v))
        if non_empty.any():
            raise ValueError(
                f"{label}: reviewer column '{col}' must be empty in blinded "
                f"CSV ({int(non_empty.sum())} non-empty rows)"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Step 8-1 review sample CSVs (read-only)")
    raw = pd.read_csv(INPUT_FILES["raw"], encoding="utf-8-sig")
    zs = pd.read_csv(INPUT_FILES["zscore"], encoding="utf-8-sig")
    paired = pd.read_csv(INPUT_FILES["paired"], encoding="utf-8-sig")
    print(f"  raw rows: {len(raw)}")
    print(f"  zscore rows: {len(zs)}")
    print(f"  paired rows: {len(paired)}")

    print("\nValidating inputs")
    validate_branch(raw, "raw")
    validate_branch(zs, "zscore")
    validate_paired(paired)
    print("  validation OK")

    print("\nBuilding blinded CSVs")
    raw_blind = make_branch_blinded(raw)
    zs_blind = make_branch_blinded(zs)
    paired_blind = make_paired_blinded(paired)

    assert_no_audit_column(raw_blind, "raw blinded")
    assert_no_audit_column(zs_blind, "zscore blinded")
    assert_no_audit_column(paired_blind, "paired blinded")
    assert_reviewer_cols_empty(raw_blind, BRANCH_REVIEWER_COLS, "raw blinded")
    assert_reviewer_cols_empty(zs_blind, BRANCH_REVIEWER_COLS, "zscore blinded")
    assert_reviewer_cols_empty(paired_blind, PAIRED_REVIEWER_COLS,
                               "paired blinded")

    raw_blind.to_csv(BLINDED_FILES["raw"], index=False, encoding="utf-8-sig")
    zs_blind.to_csv(BLINDED_FILES["zscore"], index=False, encoding="utf-8-sig")
    paired_blind.to_csv(BLINDED_FILES["paired"], index=False,
                        encoding="utf-8-sig")
    print(f"  saved: {BLINDED_FILES['raw']} ({len(raw_blind)} rows)")
    print(f"  saved: {BLINDED_FILES['zscore']} ({len(zs_blind)} rows)")
    print(f"  saved: {BLINDED_FILES['paired']} ({len(paired_blind)} rows)")

    print("\nBuilding audit key CSVs (do NOT distribute to reviewer)")
    raw_audit = make_branch_audit_key(raw)
    zs_audit = make_branch_audit_key(zs)
    paired_audit = make_paired_audit_key(paired)
    raw_audit.to_csv(AUDIT_KEY_FILES["raw"], index=False, encoding="utf-8-sig")
    zs_audit.to_csv(AUDIT_KEY_FILES["zscore"], index=False,
                    encoding="utf-8-sig")
    paired_audit.to_csv(AUDIT_KEY_FILES["paired"], index=False,
                        encoding="utf-8-sig")
    print(f"  saved: {AUDIT_KEY_FILES['raw']} ({len(raw_audit)} rows)")
    print(f"  saved: {AUDIT_KEY_FILES['zscore']} ({len(zs_audit)} rows)")
    print(f"  saved: {AUDIT_KEY_FILES['paired']} ({len(paired_audit)} rows)")

    print("\n=== Pilot pack summary ===")
    print(f"  raw blinded:    rows={len(raw_blind)}, cols={len(raw_blind.columns)}, "
          f"audit_col_present={'class_id_for_audit_only' in raw_blind.columns}")
    print(f"  zscore blinded: rows={len(zs_blind)}, cols={len(zs_blind.columns)}, "
          f"audit_col_present={'class_id_for_audit_only' in zs_blind.columns}")
    print(f"  paired blinded: rows={len(paired_blind)}, cols={len(paired_blind.columns)}, "
          f"audit_col_present={'class_id_for_audit_only' in paired_blind.columns}")
    print(f"  audit keys:     raw={len(raw_audit)}, zscore={len(zs_audit)}, "
          f"paired={len(paired_audit)}")

    # Reviewer-empty re-check from disk (sanity)
    raw_check = pd.read_csv(BLINDED_FILES["raw"], encoding="utf-8-sig")
    zs_check = pd.read_csv(BLINDED_FILES["zscore"], encoding="utf-8-sig")
    paired_check = pd.read_csv(BLINDED_FILES["paired"], encoding="utf-8-sig")
    raw_empty = all(
        raw_check[c].apply(lambda v: is_empty(v)).all()
        for c in BRANCH_REVIEWER_COLS
    )
    zs_empty = all(
        zs_check[c].apply(lambda v: is_empty(v)).all()
        for c in BRANCH_REVIEWER_COLS
    )
    paired_empty = all(
        paired_check[c].apply(lambda v: is_empty(v)).all()
        for c in PAIRED_REVIEWER_COLS
    )
    print(f"  reviewer columns empty after disk roundtrip: "
          f"raw={raw_empty}, zscore={zs_empty}, paired={paired_empty}")
    if not (raw_empty and zs_empty and paired_empty):
        raise ValueError(
            "Reviewer columns are not empty after disk roundtrip — refusing "
            "to leave inconsistent pilot pack on disk"
        )

    print("\nSaved outputs:")
    for path in list(BLINDED_FILES.values()) + list(AUDIT_KEY_FILES.values()):
        print(f"  {path}")


if __name__ == "__main__":
    main()
