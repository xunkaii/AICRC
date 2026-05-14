"""Step 4R-A post-hoc — Schema outputs from calibrated HGB posterior.

This is the 4R-A analogue of `scripts/generate_step4r_attention_schema_outputs.py`
(4R-B post 2/3). Uses HGB temperature-scaled probabilities to populate the
Step 3 §3~§9 output schema, with operational thresholds re-derived from val.

Reads (read-only):
    data/step4r/4ra_feature_ceiling/calibrated/
        step4r_hgb_predictions_calibrated_raw.csv
        step4r_hgb_predictions_calibrated_zscore.csv
    (anchor_reliability / anchor_type are carried in the input CSVs)
    reports/step3/step3_output_schema_uncertainty_policy.md   (policy reference)
    scripts/generate_step4_schema_outputs.py                  (rule reference; not imported)
    scripts/generate_step4r_attention_schema_outputs.py       (rule reference; not imported)

Writes (NEW paths only; existing 4R-A schema CSVs are NOT touched):
    data/step4r/4ra_feature_ceiling/calibrated/
        step4r_hgb_schema_outputs_calibrated_raw.csv
        step4r_hgb_schema_outputs_calibrated_zscore.csv
    reports/step4r/4ra_feature_ceiling/calibrated/
        step4r_hgb_schema_summary_raw.csv
        step4r_hgb_schema_summary_zscore.csv
        step4r_hgb_schema_results.md

Schema rules are re-implemented locally (not imported). Step 3 §6/§8 do not
commit numeric thresholds; this script re-derives them on the HGB *calibrated*
val posterior using the same procedure as 4R-B post 2/3. The values are
"temporary operational thresholds" and may be updated by downstream
calibration steps.

Run:
    python -X utf8 scripts/generate_step4r_hgb_schema_calibrated.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "data" / "step4r" / "4ra_feature_ceiling" / "calibrated"
OUTPUT_DATA_DIR = INPUT_DIR
OUTPUT_REPORT_DIR = (
    PROJECT_ROOT / "reports" / "step4r" / "4ra_feature_ceiling" / "calibrated"
)

BRANCHES = ["raw", "zscore"]
INPUT_PRED = {
    b: INPUT_DIR / f"step4r_hgb_predictions_calibrated_{b}.csv" for b in BRANCHES
}
OUTPUT_SCHEMA = {
    b: OUTPUT_DATA_DIR / f"step4r_hgb_schema_outputs_calibrated_{b}.csv"
    for b in BRANCHES
}
OUTPUT_SUMMARY = {
    b: OUTPUT_REPORT_DIR / f"step4r_hgb_schema_summary_{b}.csv" for b in BRANCHES
}
OUTPUT_REPORT_MD = OUTPUT_REPORT_DIR / "step4r_hgb_schema_results.md"

CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
NUM_CLASSES = len(CLASSES)
P_COLS_CAL = [f"p_{c}_calibrated" for c in CLASSES]
SPLITS = ["train", "val", "test"]
ALLOWED_POSTURES = {"SA", "CA", "HW"}
ALLOWED_LEVELS = {"confident", "hedged", "low", "no_call"}
ALLOWED_FLAGS = {
    "confident_C2",
    "within_group_ambiguity_c1_c5_c6",
    "pair_ambiguity_c3_c4",
    "pair_plus_c2_absorption",
    "anchor_unreliable",
    "posture_unknown",
    "low_confidence_no_class_set",
}
ALLOWED_CLASS_SETS = {
    tuple(s) for s in [
        [],
        ["C2"],
        ["C1", "C5", "C6"],
        ["C1", "C5"],
        ["C1", "C6"],
        ["C5", "C6"],
        ["C3", "C4"],
        ["C3", "C4", "C2"],
    ]
}
ANCHOR_DEPENDENT_SETS = {tuple(s) for s in [["C2"], ["C3", "C4", "C2"]]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ambiguity_group_label(class_set: list, flags: list, no_call: bool) -> str:
    if no_call:
        return "no_call"
    if "confident_C2" in flags:
        return "confident_C2"
    if "within_group_ambiguity_c1_c5_c6" in flags:
        return "within_group_c1_c5_c6"
    if "pair_plus_c2_absorption" in flags:
        return "pair_plus_c2_absorption"
    if "pair_ambiguity_c3_c4" in flags:
        return "pair_c3_c4"
    return "uncategorized"


def _entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    pp = np.clip(p, eps, 1.0)
    return -(pp * np.log(pp)).sum(axis=-1)


# ---------------------------------------------------------------------------
# Threshold derivation (val split, same procedure as 4R-B post 2/3)
# ---------------------------------------------------------------------------

def _derive_thresholds(probs: np.ndarray, y: np.ndarray, split: np.ndarray) -> dict:
    val_mask = split == "val"
    p_val = probs[val_mask]
    y_val = y[val_mask]
    pred_val = p_val.argmax(axis=1)
    idx_C1, idx_C2, idx_C3, idx_C4, idx_C5, idx_C6 = 0, 1, 2, 3, 4, 5

    # confident_C2: smallest t with precision >= 0.60 on val (binary C2 vs not).
    candidates = np.linspace(0.05, 0.99, 95)
    confident_C2 = None
    for t in candidates:
        above = p_val[:, idx_C2] >= t
        if above.sum() < 5:
            continue
        prec = float((y_val[above] == idx_C2).mean())
        if prec >= 0.60:
            confident_C2 = float(t)
            break
    if confident_C2 is None:
        confident_C2 = 0.99  # fallback: effectively no confident emissions

    # non_trivial_C2: percentile 50 of p_C2 over val rows where pred in {C3, C4}.
    mask_c34 = np.isin(pred_val, [idx_C3, idx_C4])
    if mask_c34.any():
        non_trivial_C2 = float(np.percentile(p_val[mask_c34, idx_C2], 50))
    else:
        non_trivial_C2 = 0.10

    # within_group: percentile 25 of min(p_C1, p_C5, p_C6) over val rows where pred in {C1, C5, C6}.
    mask_c156 = np.isin(pred_val, [idx_C1, idx_C5, idx_C6])
    if mask_c156.any():
        sub = p_val[mask_c156]
        min_p = np.minimum.reduce([sub[:, idx_C1], sub[:, idx_C5], sub[:, idx_C6]])
        within_group = float(np.percentile(min_p, 25))
    else:
        within_group = 0.165

    return {
        "confident_C2_threshold": confident_C2,
        "non_trivial_C2_threshold": non_trivial_C2,
        "within_group_threshold": within_group,
        "anchor_suppression_threshold": 0.5,
        "anchor_no_call_threshold": 0.25,
    }


# ---------------------------------------------------------------------------
# Schema rule
# ---------------------------------------------------------------------------

def _assign_schema(row: dict, thr: dict) -> dict:
    posture = row["posture_canonical"]
    if pd.isna(posture) or posture not in ALLOWED_POSTURES:
        return {
            "class_set": [],
            "flags": ["posture_unknown"],
            "level": "no_call",
            "no_call": True,
            "no_call_reason": "posture_unknown",
        }
    p_c2 = float(row["p_C2_calibrated"])
    argmax = row["pred_argmax_calibrated"]

    if p_c2 >= thr["confident_C2_threshold"] and argmax == "C2":
        return {
            "class_set": ["C2"],
            "flags": ["confident_C2"],
            "level": "confident",
            "no_call": False,
            "no_call_reason": "",
        }
    if argmax in ("C1", "C5", "C6"):
        group = ["C1", "C5", "C6"]
        wgt = thr["within_group_threshold"]
        included = [c for c in group if float(row[f"p_{c}_calibrated"]) >= wgt]
        class_set = included if len(included) >= 2 else group
        return {
            "class_set": class_set,
            "flags": ["within_group_ambiguity_c1_c5_c6"],
            "level": "hedged",
            "no_call": False,
            "no_call_reason": "",
        }
    if argmax in ("C3", "C4"):
        if p_c2 >= thr["non_trivial_C2_threshold"]:
            return {
                "class_set": ["C3", "C4", "C2"],
                "flags": ["pair_plus_c2_absorption"],
                "level": "hedged",
                "no_call": False,
                "no_call_reason": "",
            }
        return {
            "class_set": ["C3", "C4"],
            "flags": ["pair_ambiguity_c3_c4"],
            "level": "hedged",
            "no_call": False,
            "no_call_reason": "",
        }
    return {
        "class_set": [],
        "flags": ["low_confidence_no_class_set"],
        "level": "no_call",
        "no_call": True,
        "no_call_reason": "low_confidence_no_class_set",
    }


def _apply_anchor(result: dict, ar: float, thr: dict) -> dict:
    if pd.isna(ar):
        ar = 1.0
    ar = float(ar)
    flags = list(result["flags"])
    is_unreliable = ar < thr["anchor_suppression_threshold"]
    is_below_no_call = ar < thr["anchor_no_call_threshold"]
    is_anchor_dep = tuple(result["class_set"]) in ANCHOR_DEPENDENT_SETS
    if is_unreliable and "anchor_unreliable" not in flags:
        flags.append("anchor_unreliable")
    if result["no_call"]:
        return {
            "class_set": result["class_set"],
            "flags": flags,
            "level": result["level"],
            "no_call": True,
            "no_call_reason": result["no_call_reason"],
        }
    if is_below_no_call and is_anchor_dep:
        return {
            "class_set": [],
            "flags": flags,
            "level": "no_call",
            "no_call": True,
            "no_call_reason": "anchor_unreliable_anchor_dependent_set",
        }
    if is_unreliable:
        new_level = result["level"]
        if new_level == "confident":
            new_level = "hedged"
        elif new_level == "hedged":
            new_level = "low"
        return {
            "class_set": result["class_set"],
            "flags": flags,
            "level": new_level,
            "no_call": False,
            "no_call_reason": "",
        }
    return {
        "class_set": result["class_set"],
        "flags": flags,
        "level": result["level"],
        "no_call": False,
        "no_call_reason": "",
    }


# ---------------------------------------------------------------------------
# Build schema DF per branch
# ---------------------------------------------------------------------------

def _build_schema_df(df_in: pd.DataFrame, thr: dict, branch: str) -> pd.DataFrame:
    probs_cal = df_in[P_COLS_CAL].to_numpy(dtype=np.float64)
    n = probs_cal.shape[0]
    sorted_idx = np.argsort(-probs_cal, axis=1)
    top1_class = np.array([CLASSES[i] for i in sorted_idx[:, 0]])
    top2_class = np.array([CLASSES[i] for i in sorted_idx[:, 1]])
    top1_prob = np.take_along_axis(probs_cal, sorted_idx[:, :1], axis=1).squeeze(-1)
    top2_prob = np.take_along_axis(probs_cal, sorted_idx[:, 1:2], axis=1).squeeze(-1)
    margin = top1_prob - top2_prob
    pe = _entropy(probs_cal)

    sample_id = df_in["sample_id"].to_numpy()
    participant_id = df_in["participant_id"].to_numpy()
    split = df_in["split"].to_numpy()
    posture_canonical = df_in["posture"].to_numpy()
    class_id = df_in["class_id"].to_numpy()
    ar_arr = df_in["anchor_reliability"].to_numpy()
    at_arr = df_in["anchor_type"].to_numpy()
    pred_cal = df_in["pred_argmax_calibrated"].to_numpy()
    temperature = float(df_in["temperature"].iloc[0])

    records = []
    for i in range(n):
        row = {
            "posture_canonical": posture_canonical[i],
            "pred_argmax_calibrated": pred_cal[i],
        }
        for j, c in enumerate(CLASSES):
            row[f"p_{c}_calibrated"] = float(probs_cal[i, j])
        base = _assign_schema(row, thr)
        final = _apply_anchor(base, ar_arr[i], thr)
        flags = list(dict.fromkeys(final["flags"]))
        amb_group = _ambiguity_group_label(final["class_set"], flags, final["no_call"])
        cs_idx = [CLASSES.index(c) for c in final["class_set"]] if final["class_set"] else []
        cs_mass = float(probs_cal[i, cs_idx].sum()) if cs_idx else 0.0
        records.append({
            "sample_id": sample_id[i],
            "participant_id": participant_id[i],
            "split": split[i],
            "posture_canonical": posture_canonical[i],
            "true_class_id": class_id[i],
            "top1_class": top1_class[i],
            "top1_prob_calibrated": float(top1_prob[i]),
            "top2_class": top2_class[i],
            "top2_prob_calibrated": float(top2_prob[i]),
            "top1_top2_margin_calibrated": float(margin[i]),
            "predictive_entropy_calibrated": float(pe[i]),
            "class_set": json.dumps(final["class_set"]),
            "class_set_posterior_mass": cs_mass,
            "ambiguity_group": amb_group,
            "uncertainty_flags": json.dumps(flags),
            "no_call": bool(final["no_call"]),
            "no_call_reason": final["no_call_reason"],
            "caption_confidence_level": final["level"],
            "anchor_reliability": float(ar_arr[i]) if not pd.isna(ar_arr[i]) else float("nan"),
            "anchor_type": str(at_arr[i]) if not pd.isna(at_arr[i]) else "",
            "model_source": f"4R-A_HGB_{branch}_calibrated",
            "temperature": temperature,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_schema(df: pd.DataFrame) -> None:
    bad_cs = []
    for s in df["class_set"]:
        cs = json.loads(s)
        if tuple(cs) not in ALLOWED_CLASS_SETS:
            bad_cs.append(cs)
    if bad_cs:
        raise ValueError(f"class_set outside whitelist: {bad_cs[:5]}")
    for s in df["uncertainty_flags"]:
        flags = json.loads(s)
        if len(flags) != len(set(flags)):
            raise ValueError(f"duplicate uncertainty_flags: {flags}")
        for f in flags:
            if f not in ALLOWED_FLAGS:
                raise ValueError(f"flag outside vocab: {f}")
    bad_levels = sorted(set(df["caption_confidence_level"].unique()) - ALLOWED_LEVELS)
    if bad_levels:
        raise ValueError(f"level outside enum: {bad_levels}")
    nc = df["no_call"].astype(bool)
    empty_cs = df["class_set"].apply(lambda s: json.loads(s) == [])
    if (nc != empty_cs).any():
        raise ValueError("no_call <-> empty class_set consistency violated.")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _build_summary(schema_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in SPLITS + ["all"]:
        sub = schema_df if s == "all" else schema_df[schema_df["split"] == s]
        if len(sub) == 0:
            continue
        n = len(sub)
        for level in ["confident", "hedged", "low", "no_call"]:
            cnt = int((sub["caption_confidence_level"] == level).sum())
            rows.append({"split": s, "metric": f"level_{level}_count", "value": cnt})
            rows.append({"split": s, "metric": f"level_{level}_rate", "value": float(cnt / n)})
        nc = int(sub["no_call"].astype(bool).sum())
        rows.append({"split": s, "metric": "no_call_count", "value": nc})
        rows.append({"split": s, "metric": "no_call_rate", "value": float(nc / n)})
        for amb in [
            "confident_C2",
            "within_group_c1_c5_c6",
            "pair_c3_c4",
            "pair_plus_c2_absorption",
            "no_call",
            "uncategorized",
        ]:
            cnt = int((sub["ambiguity_group"] == amb).sum())
            rows.append({"split": s, "metric": f"ambiguity_{amb}_count", "value": cnt})
            rows.append({"split": s, "metric": f"ambiguity_{amb}_rate", "value": float(cnt / n)})
        cs_size = sub["class_set"].apply(lambda s: len(json.loads(s)))
        for sz in [0, 1, 2, 3]:
            rows.append({"split": s, "metric": f"class_set_size_{sz}_count", "value": int((cs_size == sz).sum())})
        flag_counts = {f: 0 for f in ALLOWED_FLAGS}
        for fs in sub["uncertainty_flags"]:
            for f in json.loads(fs):
                flag_counts[f] = flag_counts.get(f, 0) + 1
        for f, cnt in flag_counts.items():
            rows.append({"split": s, "metric": f"flag_{f}_count", "value": int(cnt)})
            rows.append({"split": s, "metric": f"flag_{f}_rate", "value": float(cnt / n)})
        non_nc = sub[~sub["no_call"].astype(bool)]
        if len(non_nc) > 0:
            mismatches = non_nc.apply(
                lambda r: r["top1_class"] not in json.loads(r["class_set"]), axis=1
            ).sum()
            rows.append({"split": s, "metric": "top1_not_in_class_set_count", "value": int(mismatches)})
            rows.append({"split": s, "metric": "top1_not_in_class_set_rate", "value": float(mismatches / len(non_nc))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _markdown_report(
    schema_by_branch: dict, thr_by_branch: dict, T_by_branch: dict
) -> str:
    L: list[str] = []
    L.append("# Step 4R-A 후처리 — Calibrated Schema Output 결과")
    L.append("")
    L.append("- 생성 스크립트: `scripts/generate_step4r_hgb_schema_calibrated.py`")
    L.append(
        "- 입력: `data/step4r/4ra_feature_ceiling/calibrated/"
        "step4r_hgb_predictions_calibrated_{raw,zscore}.csv`"
    )
    L.append("- 기존 4R-A 산출물(LR-derived threshold 사용)은 *legacy reference*로 보존된다.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 1. 본 단계의 위치")
    L.append("")
    L.append(
        "본 단계는 4R-A HGB temperature scaling 산출물을 Step 3 §3~§9 schema로 변환한다. "
        "4R-B post 2/3와 **동일한 절차**로 val 기반 operational threshold를 재도출하여, "
        "4R-A vs 4R-B 비교가 calibration·threshold 양쪽에서 대칭이 되도록 한다."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 2. fitted temperature (재인용)")
    L.append("")
    L.append("| branch | T |")
    L.append("|---|---:|")
    for b in BRANCHES:
        L.append(f"| {b} | {T_by_branch[b]:.6f} |")
    L.append("")
    L.append(
        "T 산출 절차 및 before/after metric 변화는 "
        "`step4r_hgb_temperature_scaling_results.md` 참조."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 3. Operational thresholds (val split 자체 산출)")
    L.append("")
    L.append(
        "Step 3 §6/§8은 임계값을 commit하지 않으므로, 본 단계는 HGB calibrated posterior에 "
        "대해 val split만 사용해 동일 절차로 임계값을 자체 산출했다. **temporary "
        "operational thresholds**이며, 후속 단계에서 갱신될 수 있다."
    )
    L.append("")
    L.append("| threshold | raw | zscore |")
    L.append("|---|---:|---:|")
    for k in [
        "confident_C2_threshold",
        "non_trivial_C2_threshold",
        "within_group_threshold",
        "anchor_suppression_threshold",
        "anchor_no_call_threshold",
    ]:
        L.append(
            f"| {k} | {thr_by_branch['raw'][k]:.4f} | {thr_by_branch['zscore'][k]:.4f} |"
        )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 4. split별 caption_confidence_level 분포")
    L.append("")
    for b in BRANCHES:
        L.append(f"### 4.{BRANCHES.index(b) + 1} branch = `{b}`")
        L.append("")
        sdf = schema_by_branch[b]
        for s in SPLITS:
            sub = sdf[sdf["split"] == s]
            n = len(sub)
            L.append(f"#### {s} split (n = {n})")
            L.append("")
            L.append("| level | count | rate |")
            L.append("|---|---:|---:|")
            for level in ["confident", "hedged", "low", "no_call"]:
                cnt = int((sub["caption_confidence_level"] == level).sum())
                L.append(f"| {level} | {cnt} | {cnt / n:.4f} |")
            nc = int(sub["no_call"].astype(bool).sum())
            L.append(f"| **(no_call total)** | **{nc}** | **{nc / n:.4f}** |")
            L.append("")
    L.append("---")
    L.append("")
    L.append("## 5. class_set 크기 분포 (전체)")
    L.append("")
    for b in BRANCHES:
        L.append(f"### 5.{BRANCHES.index(b) + 1} branch = `{b}`")
        L.append("")
        sdf = schema_by_branch[b]
        cs_size = sdf["class_set"].apply(lambda s: len(json.loads(s)))
        L.append("| size | count | rate |")
        L.append("|---|---:|---:|")
        for sz in [0, 1, 2, 3]:
            c = int((cs_size == sz).sum())
            L.append(f"| {sz} | {c} | {c / len(sdf):.4f} |")
        L.append("")
    L.append("---")
    L.append("")
    L.append("## 6. ambiguity flag 발생률 (전체)")
    L.append("")
    for b in BRANCHES:
        L.append(f"### 6.{BRANCHES.index(b) + 1} branch = `{b}`")
        L.append("")
        sdf = schema_by_branch[b]
        n = len(sdf)
        L.append("| flag | rows containing it | rate |")
        L.append("|---|---:|---:|")
        flag_counts = {f: 0 for f in sorted(ALLOWED_FLAGS)}
        for fs in sdf["uncertainty_flags"]:
            for f in json.loads(fs):
                flag_counts[f] = flag_counts.get(f, 0) + 1
        for f in sorted(flag_counts):
            cnt = flag_counts[f]
            L.append(f"| `{f}` | {cnt} | {cnt / n:.4f} |")
        L.append("")
    L.append("---")
    L.append("")
    L.append("## 7. top1 argmax와 class_set의 차이")
    L.append("")
    L.append("non-no_call 행 중에서 top1 argmax가 class_set에 포함되지 않는 비율.")
    L.append("")
    for b in BRANCHES:
        L.append(f"### 7.{BRANCHES.index(b) + 1} branch = `{b}`")
        L.append("")
        sdf = schema_by_branch[b]
        L.append("| split | non-no_call n | top1 not in class_set | rate |")
        L.append("|---|---:|---:|---:|")
        for s in SPLITS + ["all"]:
            sub = sdf if s == "all" else sdf[sdf["split"] == s]
            non_nc = sub[~sub["no_call"].astype(bool)]
            if len(non_nc) == 0:
                L.append(f"| {s} | 0 | 0 | — |")
                continue
            mismatches = non_nc.apply(
                lambda r: r["top1_class"] not in json.loads(r["class_set"]), axis=1
            ).sum()
            L.append(
                f"| {s} | {len(non_nc)} | {int(mismatches)} | "
                f"{mismatches / len(non_nc):.4f} |"
            )
        L.append("")
    L.append("---")
    L.append("")
    L.append("## 8. 4R-A 기존 schema(LR-derived threshold)와의 연결")
    L.append("")
    L.append(
        "기존 `step4r_hgb_schema_outputs_{raw,zscore}.csv`는 LR-derived val threshold "
        "(Step 4 `step4_threshold_calibration.md`)를 그대로 사용했다. 본 단계의 schema는 "
        "(a) HGB calibrated posterior를 사용하고, (b) val threshold를 calibrated "
        "posterior에서 다시 도출한 결과다. 기존 산출물은 *legacy reference*로 보존되며 "
        "본 단계 산출물이 4R-B와 비교되는 *공정한 reference*이다."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 9. 4R-B와의 비교 정합성")
    L.append("")
    L.append(
        "본 단계 산출물(`step4r_hgb_schema_outputs_calibrated_{raw,zscore}.csv`)은 "
        "`step4r_bigru_attention_schema_outputs_calibrated.csv`와 다음을 *공유*한다:"
    )
    L.append("")
    L.append("- 동일한 Step 3 §3~§9 schema 어휘 (class_set whitelist, flag vocab, level enum, no_call 정책).")
    L.append("- 동일한 post-hoc calibration 패밀리 (1-parameter scalar T, val NLL).")
    L.append("- 동일한 operational threshold 도출 절차 (val 기반, 동일 정의식).")
    L.append("- 동일한 anchor 정책 (suppression / no_call 두 단계 threshold, anchor-dependent set 정의).")
    L.append("")
    L.append(
        "따라서 4R-A vs 4R-B 비교는 calibration·threshold 비대칭 없이 동일 schema 위에서 "
        "이루어진다."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 10. 본 단계에서 명시적으로 결정하지 않는 사항")
    L.append("")
    L.append("- LLM caption layer prompt / 어휘 / 금지 표현 — 본 단계 범위 밖.")
    L.append("- temporary operational threshold의 commit (실험 후 별도 단계에서 잠금).")
    L.append("- 4R-A를 main pipeline 모델로 승격 — reframing §5.5 채택 기준은 4R-B 결과에 의해 결정.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 11. 산출물 목록")
    L.append("")
    L.append("- `data/step4r/4ra_feature_ceiling/calibrated/step4r_hgb_schema_outputs_calibrated_raw.csv`")
    L.append("- `data/step4r/4ra_feature_ceiling/calibrated/step4r_hgb_schema_outputs_calibrated_zscore.csv`")
    L.append("- `reports/step4r/4ra_feature_ceiling/calibrated/step4r_hgb_schema_summary_raw.csv` (long format)")
    L.append("- `reports/step4r/4ra_feature_ceiling/calibrated/step4r_hgb_schema_summary_zscore.csv` (long format)")
    L.append("- `reports/step4r/4ra_feature_ceiling/calibrated/step4r_hgb_schema_results.md` (본 보고서)")
    L.append("")
    L.append("---")
    L.append("")
    L.append(
        "*본 보고서는 자동 생성된다. 기존 Step 1~4 / 4R-A / 4R-B 산출물은 수정되지 않으며, "
        "기존 `step4r_hgb_schema_outputs_*.csv`는 덮어쓰지 않는다.*"
    )
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 64)
    print("Step 4R-A post-hoc — Calibrated schema (HGB)")
    print("=" * 64)
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for b in BRANCHES:
        if not INPUT_PRED[b].exists():
            raise FileNotFoundError(
                f"Calibrated predictions not found: {INPUT_PRED[b]}. "
                "Run scripts/calibrate_step4r_hgb_temperature.py first."
            )

    schema_by_branch: dict = {}
    thr_by_branch: dict = {}
    T_by_branch: dict = {}

    for b in BRANCHES:
        print(f"\n[branch={b}] loading {INPUT_PRED[b]}")
        df_in = pd.read_csv(INPUT_PRED[b], encoding="utf-8-sig")
        if len(df_in) != 9275:
            raise ValueError(f"unexpected row count for {b}: {len(df_in)}")

        probs_cal = df_in[P_COLS_CAL].to_numpy(dtype=np.float64)
        y_idx = np.array(
            [CLASSES.index(c) for c in df_in["class_id"].to_numpy()], dtype=np.int64
        )
        split = df_in["split"].to_numpy()

        thr = _derive_thresholds(probs_cal, y_idx, split)
        print(f"  derived thresholds: {thr}")

        sdf = _build_schema_df(df_in, thr, b)
        _validate_schema(sdf)

        sdf.to_csv(OUTPUT_SCHEMA[b], index=False, encoding="utf-8-sig")
        print(f"  saved schema CSV -> {OUTPUT_SCHEMA[b]}")

        summary = _build_summary(sdf)
        summary.to_csv(OUTPUT_SUMMARY[b], index=False, encoding="utf-8-sig")
        print(f"  saved summary CSV -> {OUTPUT_SUMMARY[b]}")

        schema_by_branch[b] = sdf
        thr_by_branch[b] = thr
        T_by_branch[b] = float(df_in["temperature"].iloc[0])

    md = _markdown_report(schema_by_branch, thr_by_branch, T_by_branch)
    OUTPUT_REPORT_MD.write_text(md, encoding="utf-8")
    print(f"\nsaved report -> {OUTPUT_REPORT_MD}")

    print()
    print("=" * 64)
    print("Done.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
