"""Step 8-1 — Review sample extraction.

Reads (read-only):
    data/step7/step7_captions_raw.csv
    data/step7/step7_captions_zscore.csv

Writes:
    data/step8/step8_review_sample_raw.csv
    data/step8/step8_review_sample_zscore.csv
    data/step8/step8_review_sample_paired.csv
    data/step8/step8_metrics.json   (auxiliary, for the report writer)

Behavior locked by:
    reports/step8/step8_caption_evaluation_plan.md (sec. 4-13)
    reports/step7/step7_caption_generation_prototype.md
    reports/step6/step6_caption_template_design.md
    reports/step5/step5_caption_policy_design.md

This script does NOT regenerate captions, retrain models, recalibrate
thresholds, regenerate schema outputs, or compute new physical features.
It does NOT pick a final raw vs zscore branch and does NOT commit any
final wording. Step 7 caption CSVs are read-only and not modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILES = {
    "raw": PROJECT_ROOT / "data" / "step7" / "step7_captions_raw.csv",
    "zscore": PROJECT_ROOT / "data" / "step7" / "step7_captions_zscore.csv",
}
OUTPUT_DIR = PROJECT_ROOT / "data" / "step8"
OUTPUT_FILES = {
    "raw": OUTPUT_DIR / "step8_review_sample_raw.csv",
    "zscore": OUTPUT_DIR / "step8_review_sample_zscore.csv",
    "paired": OUTPUT_DIR / "step8_review_sample_paired.csv",
}
METRICS_FILE = OUTPUT_DIR / "step8_metrics.json"

EXPECTED_ROW_COUNT = 9275
EXPECTED_SPLIT_COUNTS = {"train": 6412, "val": 1436, "test": 1427}
RANDOM_STATE = 42

REQUIRED_COLUMNS = [
    "sample_id", "rep_id", "participant_id", "class_id", "posture", "split",
    "class_set_prediction", "uncertainty_flags", "caption_confidence_level",
    "no_call", "anchor_reliability", "anchor_type", "caption_ko",
    "template_id", "class_set_label", "no_call_reason", "main_feature_phrase",
    "uncertainty_phrase", "feature_phrase_source",
    "feature_phrase_fallback_used", "caption_validation_pass",
    "caption_validation_issues",
]

REVIEWER_INPUT_COLUMNS = [
    "rating_policy_compliance",
    "rating_schema_faithfulness",
    "rating_naturalness",
    "rating_non_overclaiming",
    "rating_usefulness",
    "rating_feature_phrase_appropriateness",
    "branch_preference",
    "issue_tags",
    "free_text_comment",
    "reviewer_decision",
]

PAIRED_REVIEWER_COLUMNS = [
    "reviewer_branch_preference",
    "paired_issue_tags",
    "paired_free_text_comment",
]

# Class set canonical strings (as stored in Step 7 CSV)
CS_C3C4C2 = '["C3", "C4", "C2"]'
CS_C3C4 = '["C3", "C4"]'

HIGH_FREQ_TEMPLATES = {"T-HEDGE-1", "T-NC-LOWCONF", "T-CONF-1"}
RARE_TEMPLATES = {"T-HEDGE-3", "T-NC-ANCHOR", "T-LOW-1", "T-LOW-2"}

R13_TOKENS = [
    "가능성 가능성", "모호성 가능성", "패턴에 부합하는 패턴",
    "단독 패턴에 부합하는 패턴", "모호성 사이의 모호성",
    "클래스 집합조차", "어떤 클래스도 추정하지 않습니다", "anchor 단서",
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def parse_bool(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)


def validate_input(df: pd.DataFrame, branch: str) -> None:
    if len(df) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"{branch}: row count {len(df)} != {EXPECTED_ROW_COUNT}"
        )
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"{branch}: missing required column {col}")
    if not df["sample_id"].is_unique:
        raise ValueError(f"{branch}: sample_id is not unique")
    sc = df["split"].value_counts().to_dict()
    for split, exp in EXPECTED_SPLIT_COUNTS.items():
        if sc.get(split, 0) != exp:
            raise ValueError(
                f"{branch}: split {split} count {sc.get(split, 0)} != {exp}"
            )
    pass_series = df["caption_validation_pass"].apply(parse_bool)
    if not pass_series.all():
        raise ValueError(
            f"{branch}: caption_validation_pass has False rows "
            f"({(~pass_series).sum()} found)"
        )
    issues_nonempty = df["caption_validation_issues"].apply(
        lambda s: s != "[]"
    )
    if issues_nonempty.any():
        raise ValueError(
            f"{branch}: caption_validation_issues has non-empty rows "
            f"({issues_nonempty.sum()} found)"
        )
    no_call = df["no_call"].apply(parse_bool)
    cs_empty = df["class_set_prediction"] == "[]"
    if (no_call != cs_empty).any():
        n_bad = (no_call != cs_empty).sum()
        raise ValueError(
            f"{branch}: no_call <-> class_set==[] consistency violated "
            f"({n_bad} rows)"
        )


def assert_sample_id_alignment(raw: pd.DataFrame, zs: pd.DataFrame) -> None:
    rs = set(raw["sample_id"])
    zs_set = set(zs["sample_id"])
    if rs != zs_set:
        only_raw = rs - zs_set
        only_zs = zs_set - rs
        raise ValueError(
            f"raw vs zscore sample_id mismatch: only_raw={len(only_raw)}, "
            f"only_zscore={len(only_zs)}"
        )


# ---------------------------------------------------------------------------
# Pre-screening metrics (Step 8 §6)
# ---------------------------------------------------------------------------

def length_stats(values: pd.Series) -> dict:
    return {
        "min": int(values.min()),
        "p25": float(values.quantile(0.25)),
        "median": float(values.quantile(0.5)),
        "p75": float(values.quantile(0.75)),
        "max": int(values.max()),
        "mean": float(values.mean()),
    }


def compute_branch_metrics(df: pd.DataFrame) -> dict:
    lengths = df["caption_ko"].astype(str).str.len()
    dup_top = (
        df["caption_ko"].value_counts().head(20).to_dict()
    )
    issue_count = (
        df["caption_validation_issues"]
        .apply(lambda s: 0 if s == "[]" else len(json.loads(s)))
        .sum()
    )
    r13_hits = sum(
        any(tok in cap for tok in R13_TOKENS)
        for cap in df["caption_ko"].drop_duplicates()
    )
    return {
        "row_count": len(df),
        "unique_caption_count": int(df["caption_ko"].nunique()),
        "template_id": df["template_id"].value_counts().to_dict(),
        "caption_confidence_level":
            df["caption_confidence_level"].value_counts().to_dict(),
        "class_set_prediction":
            df["class_set_prediction"].value_counts().to_dict(),
        "no_call_reason":
            df[df["no_call"].apply(parse_bool)]["no_call_reason"]
            .value_counts().to_dict(),
        "caption_length": length_stats(lengths),
        "duplicate_caption_top20": dup_top,
        "validation_issue_count": int(issue_count),
        "r13_unique_caption_hits": int(r13_hits),
    }


def compute_pair_metrics(raw: pd.DataFrame, zs: pd.DataFrame) -> dict:
    paired = raw[["sample_id", "caption_ko"]].rename(
        columns={"caption_ko": "raw_caption_ko"}
    ).merge(
        zs[["sample_id", "caption_ko"]].rename(
            columns={"caption_ko": "zscore_caption_ko"}
        ),
        on="sample_id", how="inner",
    )
    same = (paired["raw_caption_ko"] == paired["zscore_caption_ko"])
    return {
        "joined_row_count": len(paired),
        "same_caption_count": int(same.sum()),
        "different_caption_count": int((~same).sum()),
        "equality_rate": float(same.mean()),
    }


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------

def anchor_unreliable_series(df: pd.DataFrame) -> pd.Series:
    return df["uncertainty_flags"].apply(
        lambda s: "anchor_unreliable" in json.loads(s)
    )


def anchor_bin(rel: float) -> str:
    if rel >= 0.75:
        return "high"
    if rel >= 0.50:
        return "mid_high"
    if rel >= 0.25:
        return "mid_low"
    return "low"


def compute_priority_tags(
    row: pd.Series, *, is_unique_rep: bool,
    differing_sample_ids: set,
) -> str:
    tags = []
    if is_unique_rep:
        tags.append("unique_caption")
    if row["template_id"] in HIGH_FREQ_TEMPLATES:
        tags.append("high_frequency_template")
    if row["template_id"] in RARE_TEMPLATES:
        tags.append("rare_template")
    if parse_bool(row["no_call"]):
        tags.append("no_call")
    if row.get("no_call_reason", "") == "anchor_driven":
        tags.append("anchor_driven_no_call")
    if row["class_set_prediction"] == CS_C3C4C2:
        tags.append("c3c4c2")
    if row["class_set_prediction"] == CS_C3C4:
        tags.append("c3c4_without_c2")
    if "anchor_unreliable" in json.loads(row["uncertainty_flags"]):
        tags.append("anchor_unreliable")
    if row["sample_id"] in differing_sample_ids:
        tags.append("branch_different")
    if row["split"] == "test":
        tags.append("test_split")
    return json.dumps(tags, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Branch sample construction
# ---------------------------------------------------------------------------

def build_branch_sample(
    df: pd.DataFrame, branch: str, *,
    differing_sample_ids: set,
    max_size: int = 150,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    selected_ids: set = set()
    selected_unique_rep_ids: set = set()
    selected: list = []

    # Phase 1: one row per unique caption (deterministic — sort within group)
    df_sorted = df.sort_values(["caption_ko", "sample_id"])
    for cap in df_sorted["caption_ko"].drop_duplicates():
        if len(selected) >= max_size:
            break
        sub = df_sorted[df_sorted["caption_ko"] == cap]
        row = sub.iloc[0]
        sid = row["sample_id"]
        selected_ids.add(sid)
        selected_unique_rep_ids.add(sid)
        selected.append(row)

    # Phase 2: stratified reinforcement by quotas
    flags_anchor = anchor_unreliable_series(df)
    quotas = [
        ("test_split", df["split"] == "test", 18),
        ("anchor_unreliable", flags_anchor, 12),
        ("c3c4c2", df["class_set_prediction"] == CS_C3C4C2, 12),
        ("c3c4_without_c2", df["class_set_prediction"] == CS_C3C4, 10),
        ("anchor_driven_no_call",
         df["no_call_reason"] == "anchor_driven", 8),
        ("posture_CA", df["posture"] == "CA", 5),
        ("posture_HW", df["posture"] == "HW", 5),
        ("posture_SA", df["posture"] == "SA", 5),
    ]
    for label, mask, quota in quotas:
        if len(selected) >= max_size:
            break
        cand = df[mask & ~df["sample_id"].isin(selected_ids)]
        if len(cand) == 0:
            continue
        n = min(quota, len(cand), max_size - len(selected))
        sampled = cand.sample(n=n, random_state=seed)
        for _, r in sampled.iterrows():
            selected_ids.add(r["sample_id"])
            selected.append(r)

    rows = []
    for idx, row in enumerate(selected):
        is_unique = row["sample_id"] in selected_unique_rep_ids
        rows.append(_to_branch_row(
            row, branch=branch, review_idx=idx + 1,
            is_unique_rep=is_unique,
            differing_sample_ids=differing_sample_ids,
        ))
    out = pd.DataFrame(rows)

    # Add blank reviewer columns
    for col in REVIEWER_INPUT_COLUMNS:
        out[col] = ""
    return out


def _to_branch_row(
    row: pd.Series, *, branch: str, review_idx: int,
    is_unique_rep: bool, differing_sample_ids: set,
) -> dict:
    return {
        "review_id": f"{branch}-{review_idx:04d}",
        "branch": branch,
        "sample_id": row["sample_id"],
        "split": row["split"],
        "posture": row["posture"],
        "class_id_for_audit_only": row["class_id"],
        "caption_ko": row["caption_ko"],
        "template_id": row["template_id"],
        "class_set_prediction": row["class_set_prediction"],
        "uncertainty_flags": row["uncertainty_flags"],
        "caption_confidence_level": row["caption_confidence_level"],
        "no_call": parse_bool(row["no_call"]),
        "no_call_reason": row["no_call_reason"]
            if isinstance(row["no_call_reason"], str) else "",
        "main_feature_phrase": row["main_feature_phrase"]
            if isinstance(row["main_feature_phrase"], str) else "",
        "uncertainty_phrase": row["uncertainty_phrase"],
        "anchor_reliability_bin": anchor_bin(float(row["anchor_reliability"])),
        "anchor_unreliable": "anchor_unreliable" in json.loads(
            row["uncertainty_flags"]),
        "feature_phrase_source": row["feature_phrase_source"]
            if isinstance(row["feature_phrase_source"], str) else "",
        "feature_phrase_fallback_used": parse_bool(
            row["feature_phrase_fallback_used"]),
        "caption_validation_pass": parse_bool(row["caption_validation_pass"]),
        "caption_length_chars": len(str(row["caption_ko"])),
        "review_priority_tag": compute_priority_tags(
            row, is_unique_rep=is_unique_rep,
            differing_sample_ids=differing_sample_ids,
        ),
    }


# ---------------------------------------------------------------------------
# Paired sample construction
# ---------------------------------------------------------------------------

def classify_difference(rrow: dict, zrow: dict) -> str:
    if rrow["caption_ko"] == zrow["caption_ko"]:
        return "same_caption"
    diffs = []
    if rrow["class_set_prediction"] != zrow["class_set_prediction"]:
        diffs.append("class_set")
    if rrow["caption_confidence_level"] != zrow["caption_confidence_level"]:
        diffs.append("confidence_level")
    if parse_bool(rrow["no_call"]) != parse_bool(zrow["no_call"]):
        diffs.append("no_call")
    if rrow.get("main_feature_phrase", "") != zrow.get("main_feature_phrase", ""):
        diffs.append("feature_phrase")
    if len(diffs) >= 2:
        return "multiple_differences"
    if "class_set" in diffs:
        return "different_class_set"
    if "confidence_level" in diffs:
        return "different_confidence_level"
    if "no_call" in diffs:
        return "different_no_call"
    if "feature_phrase" in diffs:
        return "different_feature_phrase"
    return "different_caption_same_schema"


def build_paired_sample(
    raw: pd.DataFrame, zs: pd.DataFrame, *,
    max_size: int = 150,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    cols_keep = [
        "sample_id", "split", "posture", "class_id",
        "caption_ko", "template_id", "class_set_prediction",
        "caption_confidence_level", "no_call", "no_call_reason",
        "main_feature_phrase",
    ]
    raw_sub = raw[cols_keep].rename(
        columns={c: f"raw_{c}" for c in cols_keep
                 if c not in ("sample_id", "split", "posture", "class_id")}
    )
    zs_sub = zs[cols_keep].rename(
        columns={c: f"zscore_{c}" for c in cols_keep
                 if c not in ("sample_id", "split", "posture", "class_id")}
    )
    paired = raw_sub.merge(
        zs_sub.drop(columns=["split", "posture", "class_id"]),
        on="sample_id", how="inner",
    )
    if len(paired) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"paired join row count {len(paired)} != {EXPECTED_ROW_COUNT}"
        )
    paired["captions_equal"] = (
        paired["raw_caption_ko"] == paired["zscore_caption_ko"]
    )
    paired["branch_difference_type"] = paired.apply(
        lambda r: classify_difference(
            {
                "caption_ko": r["raw_caption_ko"],
                "class_set_prediction": r["raw_class_set_prediction"],
                "caption_confidence_level": r["raw_caption_confidence_level"],
                "no_call": r["raw_no_call"],
                "main_feature_phrase": r["raw_main_feature_phrase"],
            },
            {
                "caption_ko": r["zscore_caption_ko"],
                "class_set_prediction": r["zscore_class_set_prediction"],
                "caption_confidence_level": r["zscore_caption_confidence_level"],
                "no_call": r["zscore_no_call"],
                "main_feature_phrase": r["zscore_main_feature_phrase"],
            },
        ),
        axis=1,
    )

    # Sample: differing first (target ~120), then same (target ~30)
    diff = paired[~paired["captions_equal"]]
    same = paired[paired["captions_equal"]]
    n_diff = min(120, len(diff), max_size)
    diff_sample = diff.sample(n=n_diff, random_state=seed) if n_diff > 0 \
        else diff.iloc[0:0]
    n_same = min(max_size - n_diff, len(same), 30)
    same_sample = same.sample(n=n_same, random_state=seed + 1) if n_same > 0 \
        else same.iloc[0:0]
    chosen = pd.concat([diff_sample, same_sample], ignore_index=True)

    # Build output
    out_rows = []
    for idx, r in enumerate(chosen.itertuples(index=False), start=1):
        d = r._asdict()
        out_rows.append({
            "review_id": f"paired-{idx:04d}",
            "sample_id": d["sample_id"],
            "split": d["split"],
            "posture": d["posture"],
            "class_id_for_audit_only": d["class_id"],
            "raw_caption_ko": d["raw_caption_ko"],
            "zscore_caption_ko": d["zscore_caption_ko"],
            "raw_template_id": d["raw_template_id"],
            "zscore_template_id": d["zscore_template_id"],
            "raw_class_set_prediction": d["raw_class_set_prediction"],
            "zscore_class_set_prediction": d["zscore_class_set_prediction"],
            "raw_caption_confidence_level": d["raw_caption_confidence_level"],
            "zscore_caption_confidence_level":
                d["zscore_caption_confidence_level"],
            "raw_no_call": parse_bool(d["raw_no_call"]),
            "zscore_no_call": parse_bool(d["zscore_no_call"]),
            "raw_no_call_reason": d["raw_no_call_reason"]
                if isinstance(d["raw_no_call_reason"], str) else "",
            "zscore_no_call_reason": d["zscore_no_call_reason"]
                if isinstance(d["zscore_no_call_reason"], str) else "",
            "raw_main_feature_phrase": d["raw_main_feature_phrase"]
                if isinstance(d["raw_main_feature_phrase"], str) else "",
            "zscore_main_feature_phrase": d["zscore_main_feature_phrase"]
                if isinstance(d["zscore_main_feature_phrase"], str) else "",
            "captions_equal": bool(d["captions_equal"]),
            "branch_difference_type": d["branch_difference_type"],
        })
    out = pd.DataFrame(out_rows)
    for col in PAIRED_REVIEWER_COLUMNS:
        out[col] = ""
    return out, paired


# ---------------------------------------------------------------------------
# Coverage summary
# ---------------------------------------------------------------------------

def coverage_summary(sample: pd.DataFrame) -> dict:
    return {
        "row_count": len(sample),
        "unique_caption_count": int(sample["caption_ko"].nunique()),
        "template_id": sample["template_id"].value_counts().to_dict(),
        "caption_confidence_level":
            sample["caption_confidence_level"].value_counts().to_dict(),
        "class_set_prediction":
            sample["class_set_prediction"].value_counts().to_dict(),
        "no_call_reason":
            sample[sample["no_call"] == True]["no_call_reason"]
            .value_counts().to_dict(),
        "posture": sample["posture"].value_counts().to_dict(),
        "split": sample["split"].value_counts().to_dict(),
        "anchor_unreliable_count": int(sample["anchor_unreliable"].sum()),
        "c3c4c2_count": int(
            (sample["class_set_prediction"] == CS_C3C4C2).sum()
        ),
        "c3c4_without_c2_count": int(
            (sample["class_set_prediction"] == CS_C3C4).sum()
        ),
        "test_split_count": int((sample["split"] == "test").sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Step 7 caption CSVs (read-only)")
    raw = pd.read_csv(INPUT_FILES["raw"])
    zs = pd.read_csv(INPUT_FILES["zscore"])
    print(f"  raw rows: {len(raw)}  zscore rows: {len(zs)}")

    print("\nValidating inputs")
    validate_input(raw, "raw")
    validate_input(zs, "zscore")
    assert_sample_id_alignment(raw, zs)
    print("  validation OK")

    print("\nComputing pre-screening metrics")
    branch_metrics = {
        "raw": compute_branch_metrics(raw),
        "zscore": compute_branch_metrics(zs),
    }
    pair_metrics = compute_pair_metrics(raw, zs)
    print(f"  raw unique captions: "
          f"{branch_metrics['raw']['unique_caption_count']}")
    print(f"  zscore unique captions: "
          f"{branch_metrics['zscore']['unique_caption_count']}")
    print(f"  raw vs zscore caption equality rate: "
          f"{pair_metrics['equality_rate']:.4f}")
    print(f"  raw vs zscore differing sample count: "
          f"{pair_metrics['different_caption_count']}")

    # Identify differing sample_ids for branch_different priority tag
    differing_ids = set()
    paired_diff_check = raw[["sample_id", "caption_ko"]].rename(
        columns={"caption_ko": "raw_cap"}).merge(
        zs[["sample_id", "caption_ko"]].rename(
            columns={"caption_ko": "zs_cap"}),
        on="sample_id", how="inner",
    )
    for _, r in paired_diff_check.iterrows():
        if r["raw_cap"] != r["zs_cap"]:
            differing_ids.add(r["sample_id"])
    print(f"  differing sample_ids set size: {len(differing_ids)}")

    print("\nBuilding raw branch review sample")
    raw_sample = build_branch_sample(
        raw, "raw", differing_sample_ids=differing_ids,
    )
    raw_sample.to_csv(OUTPUT_FILES["raw"], index=False, encoding="utf-8-sig")
    print(f"  saved: {OUTPUT_FILES['raw']} ({len(raw_sample)} rows)")

    print("\nBuilding zscore branch review sample")
    zs_sample = build_branch_sample(
        zs, "zscore", differing_sample_ids=differing_ids,
    )
    zs_sample.to_csv(
        OUTPUT_FILES["zscore"], index=False, encoding="utf-8-sig"
    )
    print(f"  saved: {OUTPUT_FILES['zscore']} ({len(zs_sample)} rows)")

    print("\nBuilding paired review sample")
    paired_sample, paired_full = build_paired_sample(raw, zs)
    paired_sample.to_csv(
        OUTPUT_FILES["paired"], index=False, encoding="utf-8-sig"
    )
    print(f"  saved: {OUTPUT_FILES['paired']} ({len(paired_sample)} rows)")

    cov = {
        "raw": coverage_summary(raw_sample),
        "zscore": coverage_summary(zs_sample),
    }
    paired_diff_breakdown = (
        paired_full["branch_difference_type"].value_counts().to_dict()
    )
    paired_sample_diff_breakdown = (
        paired_sample["branch_difference_type"].value_counts().to_dict()
    )

    print("\nCoverage summary")
    for branch in ["raw", "zscore"]:
        c = cov[branch]
        print(f"  [{branch}] rows={c['row_count']} "
              f"uniq_caps={c['unique_caption_count']} "
              f"templates={list(c['template_id'].keys())}")
        print(f"    posture={c['posture']} split={c['split']}")
        print(f"    confidence_level={c['caption_confidence_level']}")
        print(f"    class_set={c['class_set_prediction']}")
        print(f"    no_call_reason={c['no_call_reason']}")
        print(f"    anchor_unreliable={c['anchor_unreliable_count']} "
              f"c3c4c2={c['c3c4c2_count']} "
              f"c3c4_no_c2={c['c3c4_without_c2_count']} "
              f"test={c['test_split_count']}")

    print(f"\n  paired sample rows: {len(paired_sample)}")
    print(f"  paired sample diff_type: {paired_sample_diff_breakdown}")
    print(f"  paired full diff_type (9275): {paired_diff_breakdown}")

    summary = {
        "branch_metrics": branch_metrics,
        "pair_metrics": pair_metrics,
        "coverage": cov,
        "paired_sample_size": int(len(paired_sample)),
        "paired_full_diff_breakdown": paired_diff_breakdown,
        "paired_sample_diff_breakdown": paired_sample_diff_breakdown,
        "differing_sample_ids_count": len(differing_ids),
        "random_state": RANDOM_STATE,
    }
    with METRICS_FILE.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  metrics: {METRICS_FILE}")

    print("\nSaved outputs:")
    for path in OUTPUT_FILES.values():
        print(f"  {path}")
    print(f"  {METRICS_FILE}")


if __name__ == "__main__":
    main()
