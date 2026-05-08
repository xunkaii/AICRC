"""Step 7 — Caption generation prototype.

Reads (read-only):
    data/step4/step4_schema_outputs_raw.csv
    data/step4/step4_schema_outputs_zscore.csv
    data/step4/step4_modeling_dataset.csv

Writes:
    data/step7/step7_captions_raw.csv
    data/step7/step7_captions_zscore.csv

Behavior locked by:
    reports/step3/step3_output_schema_uncertainty_policy.md
    reports/step5/step5_caption_policy_design.md
    reports/step6/step6_caption_template_design.md

This script does NOT train models, does NOT recalibrate thresholds, does NOT
regenerate schema outputs, and does NOT compute new physical features. It
produces a sample-level caption prototype CSV for review; the output is a
prototype, not a final wording commit, and neither raw nor zscore branch
is committed as the final deployed branch.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_SCHEMA = {
    "raw": PROJECT_ROOT / "data" / "step4" / "step4_schema_outputs_raw.csv",
    "zscore": PROJECT_ROOT / "data" / "step4" / "step4_schema_outputs_zscore.csv",
}
INPUT_MODELING = PROJECT_ROOT / "data" / "step4" / "step4_modeling_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "step7"
OUTPUT_FILES = {
    "raw": OUTPUT_DIR / "step7_captions_raw.csv",
    "zscore": OUTPUT_DIR / "step7_captions_zscore.csv",
}

EXPECTED_ROW_COUNT = 9275
EXPECTED_SPLIT_COUNTS = {"train": 6412, "val": 1436, "test": 1427}

SCHEMA_REQUIRED = [
    "sample_id", "rep_id", "participant_id", "class_id", "posture", "split",
    "anchor_reliability", "anchor_type", "class_set_prediction",
    "uncertainty_flags", "caption_confidence_level", "no_call",
]
MODELING_REQUIRED = [
    "sample_id",
    "motion_range_acc_z", "depth_proxy", "motion_range_gyro_mag",
    "bottom_stability_acc", "bottom_transition_delta_acc_z",
    "motion_range_acc_z_zscore", "depth_proxy_zscore",
    "motion_range_gyro_mag_zscore", "bottom_stability_acc_zscore",
    "bottom_transition_delta_acc_z_zscore",
    "posture_SA", "posture_CA", "posture_HW",
    "anchor_reliability", "anchor_type",
]

CLASS_SET_LABELS = {
    # Step 7-1 wording refinement: labels are neutral identifiers only.
    # Sentence functions ("가능성", "모호성", "패턴", "흡수") are carried by
    # templates, not by the label, to avoid awkward duplication when a
    # sentence-function word appears both in label and template.
    ("C2",): "C2 유형",
    ("C1", "C5", "C6"): "C1·C5·C6 유형군",
    ("C1", "C5"): "C1·C5 유형",
    ("C1", "C6"): "C1·C6 유형",
    ("C5", "C6"): "C5·C6 유형",
    ("C3", "C4"): "C3·C4 유형",
    ("C3", "C4", "C2"): "C3·C4 유형과 C2 유형",
}

POSTURE_PHRASE_FRONT = {
    "SA": "SA 자세에서",
    "CA": "CA 자세에서",
    "HW": "HW 자세에서",
}
POSTURE_BASIS = {
    "SA": "SA 자세 기준",
    "CA": "CA 자세 기준",
    "HW": "HW 자세 기준",
}

FEATURE_PRIORITY = [
    "motion_range_acc_z",
    "motion_range_gyro_mag",
    "depth_proxy",
    "bottom_stability_acc",
    "bottom_transition_delta_acc_z",
]
ANCHOR_DEPENDENT = {"depth_proxy", "bottom_stability_acc", "bottom_transition_delta_acc_z"}
TEMPLATES_USE_FEATURE = {"T-CONF-1", "T-HEDGE-1", "T-LOW-2"}

FORBIDDEN_TOKENS_GLOBAL = [
    "lateral_proxy_gyro", "knee valgus", "knee_valgus",
    "무릎 각도", "무릎 외반",
    "pred_argmax_debug",
    "confident_C2_threshold", "non_trivial_C2_threshold",
    "within_group_threshold", "anchor_suppression_threshold",
    "anchor_no_call_threshold",
    "확실히", "100%", "정답", "모델 실패", "데이터 오류",
]
ANCHOR_UNRELIABLE_FORBIDDEN = [
    # phrases tied to depth_proxy / bottom_stability_acc /
    # bottom_transition_delta_acc_z — must not appear when anchor_unreliable.
    "자세 기준 깊이 관련 단서",
    "기준 시점 주변 안정성 단서",
    "전환 관련 단서",
    "회복 속도", "bottom 도달", "bottom 위치",
]

# Step 7-1 wording-refinement: catch duplicated / overly technical wording
# that was acceptable to policy validation but read awkwardly to a user.
DUPLICATE_FORBIDDEN = [
    "가능성 가능성",
    "모호성 가능성",
    "패턴에 부합하는 패턴",
    "단독 패턴에 부합하는 패턴",
    "모호성 사이의 모호성",
    "클래스 집합조차",
    "어떤 클래스도 추정하지 않습니다",
    "anchor 단서",
]

POSTERIOR_LITERAL = re.compile(r"p_C\d\s*=")


def parse_bool(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)


def parse_class_set(s):
    return tuple(json.loads(s))


def parse_flags(s):
    return set(json.loads(s))


def feature_phrase_for(feature, value, p33, p66):
    if value <= p33:
        bin_ = "low"
    elif value >= p66:
        bin_ = "high"
    else:
        bin_ = "mid"
    table = {
        "motion_range_acc_z": {
            "low": "전반적인 움직임 범위가 비교적 작은 편",
            "mid": "전반적인 움직임 범위가 보통 수준",
            "high": "전반적인 움직임 범위가 비교적 큰 편",
        },
        "motion_range_gyro_mag": {
            "low": "회전 움직임 단서가 작은 편",
            "mid": "회전 움직임 단서가 보통 수준",
            "high": "회전 움직임 단서가 큰 편",
        },
        "depth_proxy": {
            "low": "자세 기준 깊이 관련 단서가 작은 편",
            "mid": "자세 기준 깊이 관련 단서가 보통 수준",
            "high": "자세 기준 깊이 관련 단서가 큰 편",
        },
        "bottom_stability_acc": {
            "low": "기준 시점 주변 안정성 단서가 약한 편",
            "mid": "기준 시점 주변 안정성 단서가 보통 수준",
            "high": "기준 시점 주변 안정성 단서가 비교적 안정적인 편",
        },
        "bottom_transition_delta_acc_z": {
            "low": "전환 관련 단서가 작은 편",
            "mid": "전환 관련 단서가 보통 수준",
            "high": "전환 관련 단서가 큰 편",
        },
    }
    return table[feature][bin_]


def select_feature(row, suffix, anchor_unreliable, bins):
    posture = row["posture"]
    for feat in FEATURE_PRIORITY:
        if anchor_unreliable and feat in ANCHOR_DEPENDENT:
            continue
        col = feat + suffix
        if col not in row.index:
            continue
        v = row[col]
        if pd.isna(v):
            continue
        key = (feat, posture)
        if key not in bins:
            continue
        p33, p66 = bins[key]
        return feat, feature_phrase_for(feat, float(v), p33, p66), col
    return None, "", "fallback"


def uncertainty_phrase_from_flags(flags, anchor_unreliable):
    if anchor_unreliable:
        return "기준 시점 단서를 신뢰하기 어려움"
    if "within_group_ambiguity_c1_c5_c6" in flags:
        return "C1·C5·C6 유형 안에서 하나로 좁히기 어려움"
    if "pair_ambiguity_c3_c4" in flags:
        return "C3·C4 유형 사이에서 하나로 좁히기 어려움"
    if "pair_plus_c2_absorption" in flags:
        return "C3·C4 유형뿐 아니라 C2 유형과도 비슷하게 보일 수 있음"
    if "low_confidence_no_class_set" in flags:
        return "현재 신호가 제한적임"
    return "참고할 수 있는 신호가 제한적임"


def determine_no_call_reason(flags, anchor_reliability):
    if "posture_unknown" in flags:
        return "posture_unknown"
    if (
        "anchor_unreliable" in flags
        and anchor_reliability < 0.25
        and "low_confidence_no_class_set" not in flags
    ):
        return "anchor_driven"
    return "low_confidence_no_class_set"


def pick_template(level, no_call, no_call_reason, class_set, has_feature):
    if no_call:
        if no_call_reason == "posture_unknown":
            return "T-NC-POSTURE"
        if no_call_reason == "anchor_driven":
            return "T-NC-ANCHOR"
        return "T-NC-LOWCONF"
    if level == "confident":
        return "T-CONF-1" if has_feature else "T-CONF-2"
    if level == "hedged":
        if class_set == ("C2",):
            return "T-HEDGE-3"
        return "T-HEDGE-1" if has_feature else "T-HEDGE-2"
    if level == "low":
        if class_set == ("C2",):
            return "T-LOW-1"
        return "T-LOW-2" if has_feature else "T-LOW-1"
    return "T-NC-LOWCONF"


def render(template, *, posture, cs_label, main_feature_phrase, uncertainty_phrase):
    pp = POSTURE_PHRASE_FRONT.get(posture, "")
    pb = POSTURE_BASIS.get(posture, "")
    if template == "T-CONF-1":
        return f"{pp} {cs_label}에 가까운 동작 패턴이 보입니다. 참고 단서: {main_feature_phrase}."
    if template == "T-CONF-2":
        return f"{pb}에서 {cs_label}에 가까운 패턴이 비교적 안정적으로 보입니다."
    if template == "T-HEDGE-1":
        return f"{pp} {cs_label} 안에서 여러 동작 패턴이 함께 보입니다. 참고 단서: {main_feature_phrase}."
    if template == "T-HEDGE-2":
        return f"현재 신호만으로는 {cs_label} 안에서 하나로 안정적으로 좁히기 어렵습니다. {uncertainty_phrase}."
    if template == "T-HEDGE-3":
        return f"{pp} {cs_label}에 가까운 신호가 있지만, 하나로 말하기에는 신호가 충분하지 않습니다. {uncertainty_phrase}."
    if template == "T-LOW-1":
        return f"{pp} 신호가 약해서 {cs_label} 안에서도 하나로 말하기 어렵습니다. {uncertainty_phrase}."
    if template == "T-LOW-2":
        return f"{cs_label} 안에서 여러 동작 패턴이 함께 보입니다. 참고 단서: {main_feature_phrase}."
    if template == "T-NC-POSTURE":
        return "자세 정보가 확인되지 않아 동작 유형 판단을 보류합니다."
    if template == "T-NC-ANCHOR":
        return "기준 시점 단서를 신뢰하기 어려워, 해당 단서에 의존하는 동작 유형 판단을 보류합니다."
    if template == "T-NC-LOWCONF":
        return (
            "현재 신호만으로는 동작 유형을 안정적으로 좁히기 어렵습니다. "
            "무리해서 판단하지 않고 보류합니다."
        )
    return ""


def validate_caption(caption, *, no_call, class_set, anchor_unreliable):
    issues = []
    # Rule 1: no_call must not contain any class name C1..C6
    if no_call:
        for c in ["C1", "C2", "C3", "C4", "C5", "C6"]:
            if c in caption:
                issues.append(f"no_call_caption_contains_class:{c}")
                break
    # Rules 2 / 3: no_call <-> class_set == []
    if no_call and class_set != tuple():
        issues.append("no_call_class_set_not_empty")
    if (not no_call) and class_set == tuple():
        issues.append("not_no_call_but_class_set_empty")
    # Rule 4: ["C3","C4","C2"] caption must contain C2
    if class_set == ("C3", "C4", "C2") and "C2" not in caption:
        issues.append("c3c4c2_caption_missing_C2")
    # Rule 5: ["C3","C4"] caption must NOT contain C2
    if class_set == ("C3", "C4") and "C2" in caption:
        issues.append("c3c4_caption_contains_C2")
    # Rule 6: anchor_unreliable suppression
    if anchor_unreliable:
        for tok in ANCHOR_UNRELIABLE_FORBIDDEN:
            if tok in caption:
                issues.append(f"anchor_unreliable_contains_suppressed:{tok}")
    # Rules 7-11: global forbidden tokens
    for tok in FORBIDDEN_TOKENS_GLOBAL:
        if tok in caption:
            issues.append(f"forbidden_token:{tok}")
    # Rule 12: posterior probability literal forbidden
    if POSTERIOR_LITERAL.search(caption):
        issues.append("posterior_probability_literal")
    # Rule 13 (Step 7-1): duplicate / technical wording forbidden
    for tok in DUPLICATE_FORBIDDEN:
        if tok in caption:
            issues.append(f"duplicate_or_technical_wording:{tok}")
    return (len(issues) == 0), issues


def compute_bins(modeling, suffix):
    bins = {}
    train = modeling[modeling["split"] == "train"]
    summary = []
    for feat_base in FEATURE_PRIORITY:
        col = feat_base + suffix
        if col not in modeling.columns:
            continue
        for posture in ["SA", "CA", "HW"]:
            sub = train[train["posture"] == posture]
            vals = sub[col].dropna()
            if len(vals) == 0:
                continue
            p33 = float(vals.quantile(0.33))
            p66 = float(vals.quantile(0.66))
            bins[(feat_base, posture)] = (p33, p66)
            summary.append({
                "branch_suffix": suffix or "(raw)",
                "feature": feat_base,
                "posture": posture,
                "p33": p33,
                "p66": p66,
                "n_train": int(len(vals)),
            })
    return bins, summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading modeling dataset: {INPUT_MODELING}")
    modeling = pd.read_csv(INPUT_MODELING, encoding="utf-8-sig")
    print(f"  modeling rows: {len(modeling)}")
    if len(modeling) != EXPECTED_ROW_COUNT:
        sys.exit(f"FAIL: modeling rows {len(modeling)} != {EXPECTED_ROW_COUNT}")
    for col in MODELING_REQUIRED:
        if col not in modeling.columns:
            sys.exit(f"FAIL: modeling missing column {col}")
    if not modeling["sample_id"].is_unique:
        sys.exit("FAIL: modeling sample_id not unique")

    print("Computing posture-conditional p33/p66 from train split (read-only)")
    bins_raw, bins_raw_summary = compute_bins(modeling, "")
    bins_zs, bins_zs_summary = compute_bins(modeling, "_zscore")
    print(f"  bins keys: raw={len(bins_raw)}, zscore={len(bins_zs)}")

    bins_per_branch = {"raw": bins_raw, "zscore": bins_zs}
    bins_summary_per_branch = {"raw": bins_raw_summary, "zscore": bins_zs_summary}

    # Cache feature columns we pull from modeling
    feat_cols = ["sample_id"] + [
        f for f in (FEATURE_PRIORITY + [b + "_zscore" for b in FEATURE_PRIORITY])
    ]
    mod_subset = modeling[feat_cols]

    summary_per_branch = {}
    cross_branch_fail_examples = []

    for branch in ["raw", "zscore"]:
        suffix = "_zscore" if branch == "zscore" else ""
        print(f"\n=== Branch: {branch} ===")
        schema_path = INPUT_SCHEMA[branch]
        print(f"Loading schema output: {schema_path}")
        schema = pd.read_csv(schema_path, encoding="utf-8-sig")
        print(f"  schema rows: {len(schema)}")
        if len(schema) != EXPECTED_ROW_COUNT:
            sys.exit(f"FAIL: schema rows {len(schema)} != {EXPECTED_ROW_COUNT}")
        for col in SCHEMA_REQUIRED:
            if col not in schema.columns:
                sys.exit(f"FAIL: schema missing column {col} in {branch}")
        if not schema["sample_id"].is_unique:
            sys.exit(f"FAIL: schema sample_id not unique in {branch}")
        sc = schema["split"].value_counts().to_dict()
        for split, exp in EXPECTED_SPLIT_COUNTS.items():
            if sc.get(split, 0) != exp:
                sys.exit(f"FAIL: split {split} = {sc.get(split, 0)} != {exp} in {branch}")
        print(f"  split counts ok: {sc}")

        joined = schema.merge(mod_subset, on="sample_id", how="left")
        if len(joined) != EXPECTED_ROW_COUNT:
            sys.exit(f"FAIL: join rows {len(joined)} != {EXPECTED_ROW_COUNT}")
        print(f"  join rows: {len(joined)}")

        bins = bins_per_branch[branch]

        out_rows = []
        fallback_count = 0
        validation_fail_count = 0
        validation_fail_examples = []

        for _, row in joined.iterrows():
            class_set = parse_class_set(row["class_set_prediction"])
            flags = parse_flags(row["uncertainty_flags"])
            level = row["caption_confidence_level"]
            no_call = parse_bool(row["no_call"])
            posture = row["posture"]
            anchor_unreliable = "anchor_unreliable" in flags
            anchor_reliability = float(row["anchor_reliability"])

            cs_label = CLASS_SET_LABELS.get(class_set, "")
            uphrase = uncertainty_phrase_from_flags(flags, anchor_unreliable)
            no_call_reason = (
                determine_no_call_reason(flags, anchor_reliability) if no_call else ""
            )

            if no_call:
                feature_phrase = ""
                feature_source = ""
                fallback_used = False
            else:
                feat, fphrase, fcol = select_feature(row, suffix, anchor_unreliable, bins)
                if feat is None:
                    feature_phrase = ""
                    feature_source = "fallback"
                    fallback_used = True
                else:
                    feature_phrase = fphrase
                    feature_source = fcol
                    fallback_used = False

            template = pick_template(
                level, no_call, no_call_reason, class_set,
                has_feature=bool(feature_phrase),
            )

            caption = render(
                template, posture=posture, cs_label=cs_label,
                main_feature_phrase=feature_phrase,
                uncertainty_phrase=uphrase,
            )

            if template not in TEMPLATES_USE_FEATURE:
                out_main_feature_phrase = ""
                if feature_source != "fallback":
                    feature_source = ""
            else:
                out_main_feature_phrase = feature_phrase

            pass_, issues = validate_caption(
                caption, no_call=no_call, class_set=class_set,
                anchor_unreliable=anchor_unreliable,
            )
            if not pass_:
                validation_fail_count += 1
                if len(validation_fail_examples) < 10:
                    validation_fail_examples.append({
                        "branch": branch,
                        "sample_id": row["sample_id"],
                        "template_id": template,
                        "caption_ko": caption,
                        "issues": issues,
                    })
            if fallback_used:
                fallback_count += 1

            out_rows.append({
                "sample_id": row["sample_id"],
                "rep_id": row["rep_id"],
                "participant_id": row["participant_id"],
                "class_id": row["class_id"],
                "posture": row["posture"],
                "split": row["split"],
                "class_set_prediction": row["class_set_prediction"],
                "uncertainty_flags": row["uncertainty_flags"],
                "caption_confidence_level": row["caption_confidence_level"],
                "no_call": no_call,
                "anchor_reliability": row["anchor_reliability"],
                "anchor_type": row["anchor_type"],
                "caption_ko": caption,
                "template_id": template,
                "class_set_label": cs_label,
                "no_call_reason": no_call_reason,
                "main_feature_phrase": out_main_feature_phrase,
                "uncertainty_phrase": uphrase,
                "feature_phrase_source": feature_source,
                "feature_phrase_fallback_used": fallback_used,
                "caption_validation_pass": pass_,
                "caption_validation_issues": json.dumps(issues, ensure_ascii=False),
            })

        out_df = pd.DataFrame(out_rows)
        # utf-8-sig (UTF-8 with BOM) so Excel auto-detects encoding and Korean
        # caption_ko renders correctly when opened directly. pandas / Python
        # readers parse utf-8-sig as UTF-8 transparently.
        out_df.to_csv(OUTPUT_FILES[branch], index=False, encoding="utf-8-sig")
        print(f"  saved: {OUTPUT_FILES[branch]} ({len(out_df)} rows)")

        ccl_counts = out_df["caption_confidence_level"].value_counts().to_dict()
        cs_counts = out_df["class_set_prediction"].value_counts().to_dict()
        tpl_counts = out_df["template_id"].value_counts().to_dict()
        nc_counts = (
            out_df[out_df["no_call"] == True]["no_call_reason"].value_counts().to_dict()
        )
        split_counts = out_df["split"].value_counts().to_dict()
        pass_count = int((out_df["caption_validation_pass"] == True).sum())
        fail_count = int((out_df["caption_validation_pass"] == False).sum())

        print(f"  caption_confidence_level: {ccl_counts}")
        print(
            f"  class_set_prediction (top): "
            f"{sorted(cs_counts.items(), key=lambda x: -x[1])[:3]}"
        )
        print(f"  template_id: {tpl_counts}")
        print(f"  no_call_reason: {nc_counts}")
        print(f"  feature_phrase_fallback_used rows: {fallback_count}")
        print(f"  validation pass / fail: {pass_count} / {fail_count}")

        cross_branch_fail_examples.extend(validation_fail_examples)

        unique_captions = out_df["caption_ko"].drop_duplicates().tolist()
        dup_hits = []
        for cap in unique_captions:
            for tok in DUPLICATE_FORBIDDEN:
                if tok in cap:
                    dup_hits.append({"token": tok, "caption_ko": cap})
                    break
        print(f"  unique caption count: {len(unique_captions)}")
        print(f"  duplicate-or-technical-wording hits (unique caption level): {len(dup_hits)}")
        if dup_hits:
            for h in dup_hits[:10]:
                print(f"    - [{h['token']}] {h['caption_ko']}")

        summary_per_branch[branch] = {
            "row_count": len(out_df),
            "split_counts": split_counts,
            "caption_confidence_level": ccl_counts,
            "class_set_prediction": cs_counts,
            "template_id": tpl_counts,
            "no_call_reason": nc_counts,
            "fallback_count": fallback_count,
            "validation_pass_count": pass_count,
            "validation_fail_count": fail_count,
            "validation_fail_examples": validation_fail_examples,
            "bins_summary": bins_summary_per_branch[branch],
            "unique_caption_count": len(unique_captions),
            "duplicate_wording_unique_hits": len(dup_hits),
        }

    print("\nSaved outputs:")
    for path in OUTPUT_FILES.values():
        print(f"  {path}")

    if cross_branch_fail_examples:
        print("\n--- validation issue examples (up to 10) ---")
        for ex in cross_branch_fail_examples[:10]:
            print(json.dumps(ex, ensure_ascii=False))

    # Step 7-1 wording samples per template (up to 5 each)
    sample_groups = ["T-CONF-1", "T-CONF-2", "T-HEDGE-1", "T-HEDGE-2", "T-HEDGE-3",
                     "T-LOW-1", "T-LOW-2", "T-NC-POSTURE", "T-NC-ANCHOR", "T-NC-LOWCONF"]
    print("\n--- caption samples per template (raw branch, up to 5 each) ---")
    raw_df = pd.read_csv(OUTPUT_FILES["raw"], encoding="utf-8-sig")
    for tpl in sample_groups:
        sub = raw_df[raw_df["template_id"] == tpl]
        if len(sub) == 0:
            print(f"[{tpl}] (no rows in raw)")
            continue
        # de-duplicate captions for variety
        seen = []
        for _, r in sub.iterrows():
            if r["caption_ko"] in seen:
                continue
            seen.append(r["caption_ko"])
            if len(seen) >= 5:
                break
        print(f"[{tpl}] n={len(sub)} (showing up to {len(seen)} unique captions)")
        for cap in seen:
            print(f"  - {cap}")

    overall_fail = sum(s["validation_fail_count"] for s in summary_per_branch.values())
    if overall_fail == 0:
        print("\nStep 7 prototype validation passed")
    else:
        print(f"\nStep 7 prototype validation NOT clean: {overall_fail} fail rows")

    # Persist summary as JSON for the report writer
    summary_path = OUTPUT_DIR / "step7_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_per_branch, f, ensure_ascii=False, indent=2)
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
