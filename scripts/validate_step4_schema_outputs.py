"""Step 4 - Validate Step 3 schema-compliant output CSVs.

Reads (read-only):
    data/step4/step4_schema_outputs_raw.csv
    data/step4/step4_schema_outputs_zscore.csv

Behavior locked by:
    reports/step3/step3_output_schema_uncertainty_policy.md  (sec. 3, 4, 6, 7, 8, 9)
    reports/step4/step4_modeling_calibration_plan.md         (sec. 9 - validation items)

Independent validator: does NOT train models, does NOT recalibrate
thresholds, does NOT regenerate schema outputs, does NOT write captions.
Only checks structural compliance against the locked policy.

Per Step 4 plan sec. 9 #3, this validator does NOT reinterpret the
posterior to judge subset validity; it only verifies that
class_set_prediction values are inside the closed whitelist.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILES = {
    "raw": PROJECT_ROOT / "data" / "step4" / "step4_schema_outputs_raw.csv",
    "zscore": PROJECT_ROOT / "data" / "step4" / "step4_schema_outputs_zscore.csv",
}

EXPECTED_ROW_COUNT = 9275

CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
P_COLS = [f"p_{c}" for c in CLASSES]

REQUIRED_COLS = [
    "sample_id",
    "rep_id",
    "participant_id",
    "class_id",
    "posture",
    "split",
    "anchor_reliability",
    "anchor_type",
    "p_C1", "p_C2", "p_C3", "p_C4", "p_C5", "p_C6",
    "pred_argmax_debug",
    "class_set_prediction",
    "uncertainty_flags",
    "caption_confidence_level",
    "no_call",
]

ALLOWED_CLASS_SETS = {
    tuple(s) for s in [
        ["C2"],
        ["C1", "C5", "C6"],
        ["C1", "C5"],
        ["C1", "C6"],
        ["C5", "C6"],
        ["C3", "C4"],
        ["C3", "C4", "C2"],
        [],
    ]
}

ALLOWED_FLAGS = {
    "confident_C2",
    "within_group_ambiguity_c1_c5_c6",
    "pair_ambiguity_c3_c4",
    "pair_plus_c2_absorption",
    "anchor_unreliable",
    "posture_unknown",
    "low_confidence_no_class_set",
}

ALLOWED_LEVELS = {"confident", "hedged", "low", "no_call"}

THRESHOLDS = {
    "raw": {
        "anchor_no_call_threshold": 0.2500,
        "anchor_suppression_threshold": 0.5000,
    },
    "zscore": {
        "anchor_no_call_threshold": 0.2500,
        "anchor_suppression_threshold": 0.5000,
    },
}

FORBIDDEN_COL_TOKENS = ["participant_zscore", "lateral_proxy_gyro"]


# ---------------------------------------------------------------------------
# Boolean coercion (CSV roundtrip safe)
# ---------------------------------------------------------------------------

def _to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(bool)
    return s.astype(str).str.strip().str.lower().isin({"true", "1"})


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_row_count(df: pd.DataFrame) -> tuple[bool, str]:
    if len(df) != EXPECTED_ROW_COUNT:
        return False, f"row count {len(df)} != expected {EXPECTED_ROW_COUNT}"
    return True, f"row count = {len(df)}"


def check_required_columns(df: pd.DataFrame) -> tuple[bool, str]:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return False, f"missing columns: {missing}"
    return True, f"all {len(REQUIRED_COLS)} required columns present"


def check_class_posterior(df: pd.DataFrame) -> tuple[bool, str]:
    missing_p = [c for c in P_COLS if c not in df.columns]
    if missing_p:
        return False, f"missing posterior columns: {missing_p}"
    p = df[P_COLS].to_numpy(dtype="float64")
    n_nan = int(np.isnan(p).sum())
    n_inf = int(np.isinf(p).sum())
    n_neg = int((p < 0).sum())
    if n_nan or n_inf or n_neg:
        return False, f"posterior cells have NaN={n_nan}, inf={n_inf}, negative={n_neg}"
    sums = p.sum(axis=1)
    bad = ~np.isclose(sums, 1.0, atol=1e-6)
    n_bad = int(bad.sum())
    if n_bad > 0:
        return False, (
            f"{n_bad} rows have probability sum != 1 (atol=1e-6); "
            f"min sum={float(sums.min()):.10f}, max sum={float(sums.max()):.10f}"
        )
    return True, (
        f"all {len(df)} rows: no NaN/inf/negative in p_C1..p_C6, "
        f"row sums in [{float(sums.min()):.10f}, {float(sums.max()):.10f}]"
    )


def check_class_set_whitelist(df: pd.DataFrame) -> tuple[bool, str]:
    bad: list = []
    parse_errors: list = []
    distinct: set = set()
    for i, s in enumerate(df["class_set_prediction"]):
        try:
            cs = json.loads(s)
        except (json.JSONDecodeError, TypeError) as e:
            parse_errors.append((i, repr(s), str(e)))
            if len(parse_errors) > 5:
                break
            continue
        if not isinstance(cs, list):
            bad.append((i, f"not a list: {type(cs).__name__}"))
            if len(bad) > 5:
                break
            continue
        t = tuple(cs)
        distinct.add(t)
        if t not in ALLOWED_CLASS_SETS:
            bad.append((i, cs))
            if len(bad) > 5:
                break
    if parse_errors:
        return False, f"JSON parse errors in class_set_prediction (first 5): {parse_errors}"
    if bad:
        return False, f"class_set_prediction outside whitelist (first 5): {bad}"
    return True, (
        f"all {len(df)} rows have class_set_prediction in whitelist; "
        f"{len(distinct)} distinct values observed"
    )


def check_flags_vocab(df: pd.DataFrame) -> tuple[bool, str]:
    bad_vocab: list = []
    bad_dupes: list = []
    parse_errors: list = []
    for i, s in enumerate(df["uncertainty_flags"]):
        try:
            flags = json.loads(s)
        except (json.JSONDecodeError, TypeError) as e:
            parse_errors.append((i, repr(s), str(e)))
            if len(parse_errors) > 5:
                break
            continue
        if not isinstance(flags, list):
            bad_vocab.append((i, f"not a list: {type(flags).__name__}"))
            if len(bad_vocab) > 5:
                break
            continue
        if len(flags) != len(set(flags)):
            bad_dupes.append((i, flags))
            if len(bad_dupes) > 5:
                break
        for f in flags:
            if f not in ALLOWED_FLAGS:
                bad_vocab.append((i, f))
                if len(bad_vocab) > 5:
                    break
    if parse_errors:
        return False, f"JSON parse errors in uncertainty_flags (first 5): {parse_errors}"
    if bad_vocab:
        return False, f"uncertainty_flags has values outside vocab (first 5): {bad_vocab}"
    if bad_dupes:
        return False, f"uncertainty_flags has duplicate entries (first 5): {bad_dupes}"
    return True, f"all {len(df)} rows have uncertainty_flags within vocab and unique"


def check_level_enum(df: pd.DataFrame) -> tuple[bool, str]:
    actual = set(df["caption_confidence_level"].dropna().unique())
    bad_levels = sorted(actual - ALLOWED_LEVELS)
    n_nan = int(df["caption_confidence_level"].isna().sum())
    if n_nan > 0:
        return False, f"caption_confidence_level has {n_nan} NaN values"
    if bad_levels:
        return False, f"caption_confidence_level has values outside enum: {bad_levels}"
    return True, f"caption_confidence_level values: {sorted(actual)}"


def check_no_call_consistency(df: pd.DataFrame) -> tuple[bool, str]:
    no_call = _to_bool_series(df["no_call"])
    empty_cs = df["class_set_prediction"].apply(
        lambda s: json.loads(s) == [] if isinstance(s, str) else False
    )
    a_violation = int(((no_call) & (~empty_cs)).sum())   # no_call true but cs not empty
    b_violation = int(((~no_call) & (empty_cs)).sum())   # cs empty but not no_call
    c_violation = int(
        (no_call & (df["caption_confidence_level"] != "no_call")).sum()
    )
    if a_violation or b_violation or c_violation:
        return False, (
            f"inconsistencies: no_call==True AND cs!=[] -> {a_violation} rows; "
            f"cs==[] AND no_call==False -> {b_violation} rows; "
            f"no_call==True AND level!='no_call' -> {c_violation} rows"
        )
    return True, (
        f"no_call <-> class_set==[] bidirectional consistency holds; "
        f"all {int(no_call.sum())} no_call rows have level=='no_call'"
    )


def check_no_call_reason_closure(
    df: pd.DataFrame, anchor_no_call_threshold: float
) -> tuple[bool, str]:
    no_call = _to_bool_series(df["no_call"])
    flags_lists = df["uncertainty_flags"].apply(json.loads)
    has_posture_unknown = flags_lists.apply(lambda fs: "posture_unknown" in fs)
    has_low_conf = flags_lists.apply(
        lambda fs: "low_confidence_no_class_set" in fs
    )
    has_anchor_unreliable = flags_lists.apply(
        lambda fs: "anchor_unreliable" in fs
    )
    ar = pd.to_numeric(df["anchor_reliability"], errors="coerce")
    anchor_reason = (ar < anchor_no_call_threshold) & has_anchor_unreliable

    valid_reason = has_posture_unknown | has_low_conf | anchor_reason
    bad = no_call & (~valid_reason)
    n_bad = int(bad.sum())
    if n_bad > 0:
        # Diagnose first offender for the error message.
        first_idx = int(bad.to_numpy().argmax())
        offender = {
            "row_index": first_idx,
            "sample_id": str(df.iloc[first_idx].get("sample_id", "?")),
            "anchor_reliability": float(ar.iloc[first_idx])
                if not pd.isna(ar.iloc[first_idx]) else None,
            "uncertainty_flags": df.iloc[first_idx]["uncertainty_flags"],
        }
        return False, (
            f"{n_bad} no_call rows lack a valid reason flag "
            f"(posture_unknown / low_confidence_no_class_set / anchor-driven). "
            f"first offender: {offender}"
        )
    n_via_posture = int((no_call & has_posture_unknown).sum())
    n_via_low_conf = int((no_call & has_low_conf & ~has_posture_unknown).sum())
    n_via_anchor = int(
        (no_call & anchor_reason & ~has_posture_unknown & ~has_low_conf).sum()
    )
    return True, (
        f"all {int(no_call.sum())} no_call rows have a valid reason; "
        f"posture_unknown={n_via_posture}, "
        f"low_confidence_no_class_set={n_via_low_conf}, "
        f"anchor-driven={n_via_anchor}"
    )


def check_anchor_threshold_relation(thr: dict) -> tuple[bool, str]:
    no_call_t = thr["anchor_no_call_threshold"]
    sup_t = thr["anchor_suppression_threshold"]
    if no_call_t >= sup_t:
        return False, (
            f"violation: anchor_no_call_threshold ({no_call_t}) "
            f">= anchor_suppression_threshold ({sup_t})"
        )
    return True, (
        f"anchor_no_call_threshold ({no_call_t}) "
        f"< anchor_suppression_threshold ({sup_t})"
    )


def check_track_separation(df: pd.DataFrame) -> tuple[bool, str]:
    issues: list = []
    for token in FORBIDDEN_COL_TOKENS:
        bad = [c for c in df.columns if token in c]
        if bad:
            issues.append(f"columns containing '{token}': {bad}")
    if "anchor_reliability" not in df.columns:
        issues.append("anchor_reliability is not a top-level column")
    feature_evidence_cols = [c for c in df.columns if "feature_evidence" in c.lower()]
    if feature_evidence_cols:
        issues.append(
            f"feature_evidence-named columns present "
            f"(would violate Step 3 sec. 6 if anchor info is inside): "
            f"{feature_evidence_cols}"
        )
    if issues:
        return False, "; ".join(issues)
    return True, (
        "no forbidden column tokens; anchor_reliability is its own column; "
        "no feature_evidence-named columns"
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

CHECKS = [
    ("1. row count", "row_count"),
    ("2. required columns", "required_columns"),
    ("3. class_posterior distribution", "class_posterior"),
    ("4. class_set_prediction whitelist", "class_set_whitelist"),
    ("5. uncertainty_flags vocabulary", "flags_vocab"),
    ("6. caption_confidence_level enum", "level_enum"),
    ("7. no_call consistency", "no_call_consistency"),
    ("8. no_call reason closure", "no_call_reason_closure"),
    ("9. anchor threshold relation", "anchor_threshold_relation"),
    ("10. track separation", "track_separation"),
]


def _run_checks(df: pd.DataFrame, thr: dict) -> list:
    results: list = []

    name, _ = CHECKS[0]
    ok, msg = check_row_count(df)
    results.append((name, ok, msg))

    name, _ = CHECKS[1]
    ok, msg = check_required_columns(df)
    results.append((name, ok, msg))

    name, _ = CHECKS[2]
    ok, msg = check_class_posterior(df)
    results.append((name, ok, msg))

    name, _ = CHECKS[3]
    ok, msg = check_class_set_whitelist(df)
    results.append((name, ok, msg))

    name, _ = CHECKS[4]
    ok, msg = check_flags_vocab(df)
    results.append((name, ok, msg))

    name, _ = CHECKS[5]
    ok, msg = check_level_enum(df)
    results.append((name, ok, msg))

    name, _ = CHECKS[6]
    ok, msg = check_no_call_consistency(df)
    results.append((name, ok, msg))

    name, _ = CHECKS[7]
    ok, msg = check_no_call_reason_closure(df, thr["anchor_no_call_threshold"])
    results.append((name, ok, msg))

    name, _ = CHECKS[8]
    ok, msg = check_anchor_threshold_relation(thr)
    results.append((name, ok, msg))

    name, _ = CHECKS[9]
    ok, msg = check_track_separation(df)
    results.append((name, ok, msg))

    return results


def _print_results_table(cond: str, results: list) -> None:
    print()
    print(f"=== {cond} ===")
    print(f"{'check':<38}  {'pass':<6}  message")
    print("-" * 100)
    for name, ok, msg in results:
        status = "OK" if ok else "FAIL"
        print(f"{name:<38}  {status:<6}  {msg}")


def main() -> int:
    print("=" * 100)
    print("Step 4 - Schema output validation")
    print("=" * 100)

    for cond, path in INPUT_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Schema output for '{cond}' not found: {path}. "
                "Run scripts/generate_step4_schema_outputs.py first."
            )

    all_results: dict = {}
    failures: list = []

    for cond, path in INPUT_FILES.items():
        df = pd.read_csv(path, encoding="utf-8-sig")
        thr = THRESHOLDS[cond]
        print(f"\n[{cond}] loaded {len(df)} rows from {path}")
        print(
            f"[{cond}] thresholds in use: "
            f"anchor_no_call_threshold={thr['anchor_no_call_threshold']}, "
            f"anchor_suppression_threshold={thr['anchor_suppression_threshold']}"
        )
        results = _run_checks(df, thr)
        all_results[cond] = results
        _print_results_table(cond, results)
        for name, ok, msg in results:
            if not ok:
                failures.append(f"[{cond}] {name}: {msg}")

    print()
    print("=" * 100)
    if failures:
        n_fail = len(failures)
        n_total = sum(len(r) for r in all_results.values())
        print(f"OVERALL: {n_fail} / {n_total} checks FAILED across conditions.")
        print("=" * 100)
        raise ValueError(
            "Schema validation failed for one or more checks:\n  - "
            + "\n  - ".join(failures)
        )
    n_total = sum(len(r) for r in all_results.values())
    print(
        f"OVERALL: all {n_total} checks passed "
        f"({len(all_results)} conditions x {len(CHECKS)} checks)."
    )
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
