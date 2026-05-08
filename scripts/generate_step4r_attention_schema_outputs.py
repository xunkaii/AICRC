"""Step 4R-B post-processing 2/3 — Schema outputs from calibrated posterior.

Reads (read-only):
    data/step4r/4rb_attention/step4r_bigru_attention_logits_calibrated.npz
    data/step4/step4_modeling_dataset.csv      (anchor_reliability, anchor_type)
    reports/step3/step3_output_schema_uncertainty_policy.md   (policy reference)
    scripts/generate_step4_schema_outputs.py    (rule reference; not modified)

Writes:
    data/step4r/4rb_attention/step4r_bigru_attention_schema_outputs_calibrated.csv
    reports/step4r/4rb_attention/step4r_bigru_attention_schema_summary.csv
    reports/step4r/4rb_attention/step4r_bigru_attention_schema_results.md

Behavior:
    - Uses **probs_calibrated** as the schema decision input.
    - Re-derives val-based thresholds using the same procedure as
      reports/step4/step4_threshold_calibration.md, but fit to the BiGRU
      *calibrated* posterior. These are reported as "temporary operational
      thresholds" because Step 3 did not commit numeric values.
    - anchor_reliability / anchor_type are joined from
      data/step4/step4_modeling_dataset.csv on sample_id (read-only).
    - Schema rules are re-implemented locally from
      scripts/generate_step4_schema_outputs.py (the file itself is NOT
      modified or imported, per instruction).

Run:
    python scripts/generate_step4r_attention_schema_outputs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_NPZ = (
    PROJECT_ROOT / "data" / "step4r" / "4rb_attention"
    / "step4r_bigru_attention_logits_calibrated.npz"
)
INPUT_ANCHOR_CSV = PROJECT_ROOT / "data" / "step4" / "step4_modeling_dataset.csv"

OUTPUT_DATA_DIR = PROJECT_ROOT / "data" / "step4r" / "4rb_attention"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "reports" / "step4r" / "4rb_attention"

OUTPUT_SCHEMA_CSV = OUTPUT_DATA_DIR / "step4r_bigru_attention_schema_outputs_calibrated.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_REPORT_DIR / "step4r_bigru_attention_schema_summary.csv"
OUTPUT_REPORT_MD = OUTPUT_REPORT_DIR / "step4r_bigru_attention_schema_results.md"

CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
NUM_CLASSES = len(CLASSES)
P_COLS = [f"p_{c}_calibrated" for c in CLASSES]  # internal calibrated cols
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

# Map from (class_set tuple, flags) to a single ambiguity_group label
# for downstream caption layer ergonomics.
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


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _load_calibrated() -> dict:
    print(f"loading {INPUT_NPZ}")
    z = np.load(INPUT_NPZ, allow_pickle=True)
    return {
        "probs_raw": z["probs_raw"].astype(np.float64),
        "probs_calibrated": z["probs_calibrated"].astype(np.float64),
        "y": z["y"].astype(np.int64),
        "class_id": z["class_id"].astype(str),
        "sample_id": z["sample_id"].astype(str),
        "participant_id": z["participant_id"].astype(str),
        "posture_canonical": z["posture_canonical"].astype(str),
        "split": z["split"].astype(str),
        "temperature": float(z["temperature"][0]),
    }


def _load_anchor_join(sample_ids: np.ndarray) -> pd.DataFrame:
    """Return DataFrame indexed by sample_id with anchor_reliability and anchor_type."""
    if not INPUT_ANCHOR_CSV.exists():
        raise FileNotFoundError(f"Anchor source not found: {INPUT_ANCHOR_CSV}")
    df = pd.read_csv(INPUT_ANCHOR_CSV, usecols=["sample_id", "anchor_reliability", "anchor_type"], encoding="utf-8-sig")
    df = df.set_index("sample_id")
    missing = sorted(set(sample_ids) - set(df.index))
    if missing:
        raise ValueError(
            f"{len(missing)} sample_id missing anchor info in {INPUT_ANCHOR_CSV}; "
            f"first few: {missing[:5]}"
        )
    return df.loc[sample_ids]


# ---------------------------------------------------------------------------
# Threshold derivation (val split, same procedure as step4_threshold_calibration)
# ---------------------------------------------------------------------------

def _derive_thresholds(probs: np.ndarray, y: np.ndarray, split: np.ndarray) -> dict:
    """Derive operational thresholds from val split only."""
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

    # non_trivial_C2: percentile 50 of p_C2 over val rows where pred_argmax in {C3, C4}.
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
# Schema rule (re-implemented locally from generate_step4_schema_outputs.py)
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
        # Treat as effectively reliable for safety; not expected to happen.
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
# Build schema CSV
# ---------------------------------------------------------------------------

def _entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    pp = np.clip(p, eps, 1.0)
    return -(pp * np.log(pp)).sum(axis=-1)


def _build_schema_df(d: dict, anchor_df: pd.DataFrame, thr: dict) -> pd.DataFrame:
    probs_cal = d["probs_calibrated"]
    n = probs_cal.shape[0]
    pred_idx = probs_cal.argmax(axis=1)
    sorted_idx = np.argsort(-probs_cal, axis=1)
    top1_class = np.array([CLASSES[i] for i in sorted_idx[:, 0]])
    top2_class = np.array([CLASSES[i] for i in sorted_idx[:, 1]])
    top1_prob = np.take_along_axis(probs_cal, sorted_idx[:, :1], axis=1).squeeze(-1)
    top2_prob = np.take_along_axis(probs_cal, sorted_idx[:, 1:2], axis=1).squeeze(-1)
    margin = top1_prob - top2_prob
    pe = _entropy(probs_cal)

    # Build per-row dict including p_C*_calibrated for the schema rule.
    records = []
    ar_arr = anchor_df["anchor_reliability"].to_numpy()
    at_arr = anchor_df["anchor_type"].to_numpy()
    for i in range(n):
        row = {
            "sample_id": d["sample_id"][i],
            "participant_id": d["participant_id"][i],
            "split": d["split"][i],
            "posture_canonical": d["posture_canonical"][i],
            "true_class_id": d["class_id"][i],
            "pred_argmax_calibrated": top1_class[i],
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
            "sample_id": d["sample_id"][i],
            "participant_id": d["participant_id"][i],
            "split": d["split"][i],
            "posture_canonical": d["posture_canonical"][i],
            "true_class_id": d["class_id"][i],
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
            "model_source": "4R-B_BiGRU_Attention_calibrated",
            "temperature": d["temperature"],
        })
    return pd.DataFrame(records)


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
        # class_set size distribution
        cs_size = sub["class_set"].apply(lambda s: len(json.loads(s)))
        for sz in [0, 1, 2, 3]:
            rows.append({"split": s, "metric": f"class_set_size_{sz}_count", "value": int((cs_size == sz).sum())})
        # flag occurrence
        flag_counts = {f: 0 for f in ALLOWED_FLAGS}
        for fs in sub["uncertainty_flags"]:
            for f in json.loads(fs):
                flag_counts[f] = flag_counts.get(f, 0) + 1
        for f, cnt in flag_counts.items():
            rows.append({"split": s, "metric": f"flag_{f}_count", "value": int(cnt)})
            rows.append({"split": s, "metric": f"flag_{f}_rate", "value": float(cnt / n)})
        # mismatch top1 vs class_set (top1 not in class_set among non-no_call rows)
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

def _markdown_report(schema_df: pd.DataFrame, thr: dict, T: float) -> str:
    L = []
    L.append("# Step 4R-B 후처리 2/3 — Calibrated Schema Output 결과")
    L.append("")
    L.append("- 생성 스크립트: `scripts/generate_step4r_attention_schema_outputs.py`")
    L.append("- 입력: `data/step4r/4rb_attention/step4r_bigru_attention_logits_calibrated.npz` (probs_calibrated 사용)")
    L.append(f"- temperature 적용: T = {T:.6f}")
    L.append("- 본 스크립트는 기존 `scripts/generate_step4_schema_outputs.py`를 수정하지 않으며, 동일 logic을 새 스크립트에 재구현했다.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 1. 본 단계의 위치")
    L.append("")
    L.append(
        "calibrated posterior(`probs_calibrated`)를 Step 3 §3 ~ §9 출력 스키마로 변환한다. "
        "schema 결정에는 calibrated posterior만 사용하며, raw posterior는 비교용으로만 보존한다."
    )
    L.append("")
    L.append(
        "**LLM은 판단자가 아니라 표현층이다.** 본 schema가 caption layer (Step 5~7 재정의 후속) "
        "의 *입력*이며, caption은 schema에 충실하게 한국어로 표현될 뿐 새로운 분류·판단을 "
        "하지 않는다 (`reports/step4_research_reframing.md` §6 참조)."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 2. Operational thresholds (val split 자체 산출)")
    L.append("")
    L.append(
        "Step 3 §6/§8은 임계값을 commit하지 않으므로, 본 단계는 BiGRU calibrated posterior에 "
        "대해 val split만 사용해 Step 4 calibration과 동일한 절차로 임계값을 자체 산출했다. "
        "이 값들은 **temporary operational thresholds**이며, 후속 단계(예: human review로 "
        "재calibration)에서 갱신될 수 있다."
    )
    L.append("")
    L.append("| threshold | 산출 방식 | 값 | 표시 |")
    L.append("|---|---|---:|---|")
    L.append(f"| confident_C2_threshold | smallest t s.t. precision(p_C2 ≥ t on val) ≥ 0.60 | {thr['confident_C2_threshold']:.4f} | temporary operational |")
    L.append(f"| non_trivial_C2_threshold | percentile 50 of p_C2 over val (pred ∈ C3/C4) | {thr['non_trivial_C2_threshold']:.4f} | temporary operational |")
    L.append(f"| within_group_threshold | percentile 25 of min(p_C1,p_C5,p_C6) over val (pred ∈ C1/C5/C6) | {thr['within_group_threshold']:.4f} | temporary operational |")
    L.append(f"| anchor_suppression_threshold | Step 4 heuristic (boundary-aligned bin cut) | {thr['anchor_suppression_threshold']:.4f} | reused from Step 4 |")
    L.append(f"| anchor_no_call_threshold | Step 4 heuristic (boundary-aligned bin cut) | {thr['anchor_no_call_threshold']:.4f} | reused from Step 4 |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 3. split별 no_call 및 caption_confidence_level 분포")
    L.append("")
    for s in SPLITS:
        sub = schema_df[schema_df["split"] == s]
        n = len(sub)
        L.append(f"### 3.{SPLITS.index(s) + 1} {s} split (n = {n})")
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
    L.append("## 4. class_set 크기 분포 (전체)")
    L.append("")
    cs_size = schema_df["class_set"].apply(lambda s: len(json.loads(s)))
    L.append("| size | count | rate |")
    L.append("|---|---:|---:|")
    for sz in [0, 1, 2, 3]:
        c = int((cs_size == sz).sum())
        L.append(f"| {sz} | {c} | {c / len(schema_df):.4f} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 5. ambiguity flag 발생률 (전체)")
    L.append("")
    L.append("| flag | rows containing it | rate |")
    L.append("|---|---:|---:|")
    flag_counts = {f: 0 for f in ALLOWED_FLAGS}
    for fs in schema_df["uncertainty_flags"]:
        for f in json.loads(fs):
            flag_counts[f] += 1
    for f in sorted(flag_counts.keys()):
        L.append(f"| `{f}` | {flag_counts[f]} | {flag_counts[f] / len(schema_df):.4f} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 6. top1 argmax와 class_set의 차이")
    L.append("")
    L.append(
        "non-no_call 행 중에서 top1 argmax가 실제 class_set에 *포함되지 않는* 비율. "
        "schema rule이 argmax 외 다른 class를 어떻게 추가/대체하는지 보는 지표."
    )
    L.append("")
    L.append("| split | non-no_call n | top1 not in class_set | rate |")
    L.append("|---|---:|---:|---:|")
    for s in SPLITS + ["all"]:
        sub = schema_df if s == "all" else schema_df[schema_df["split"] == s]
        non_nc = sub[~sub["no_call"].astype(bool)]
        n = len(non_nc)
        if n == 0:
            continue
        mis = non_nc.apply(
            lambda r: r["top1_class"] not in json.loads(r["class_set"]), axis=1
        ).sum()
        L.append(f"| {s} | {n} | {int(mis)} | {mis / n:.4f} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 7. Step 4R-A / 4R-B와의 연결 해석")
    L.append("")
    L.append(
        "- **분류 본체 차원** — 4R-A HGB가 LR과 동등한 분류 ceiling이었던 반면 4R-B BiGRU+Attention은 "
        "test macro F1을 ~+0.21 끌어올렸다 (4R-B results §3). 본 schema는 그 calibrated posterior 위에서 작동한다."
    )
    L.append(
        "- **Schema 행동 차원** — Step 2.5 §7의 ambiguity 패턴(C2 단언 가능 / C1·C5·C6 그룹 모호 / C3·C4의 "
        "C2 흡수)이 본 schema의 ambiguity_group 분포로 반영되어야 한다 (§5의 flag 발생률). "
        "Step 4R-B가 분류는 잘하지만 ambiguity 구조가 무너지면(예: 거의 모두 confident_C2 단일 emission) "
        "Step 3 정책이 의도한 *정직한 모호함 표현*이 사라진다 — 이를 §3 / §5에서 점검한다."
    )
    L.append(
        "- **Calibration 차원** — calibration 단계 1/3에서 fitted T로 over-confidence를 보정했다. "
        "그 효과가 hedged/low 비율 증가, confident 비율 감소, no_call 비율 감소(low_confidence 경로가 "
        "raw에서보다 적게 발생)로 반영되는지를 §3 분포에서 확인한다."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 8. 본 단계에서 명시적으로 결정하지 않는 사항")
    L.append("")
    L.append("- LLM caption layer prompt 본문, 어휘표, 금지 표현 목록은 본 단계의 범위 밖이다.")
    L.append("- temporary operational threshold의 commit (실험 후 별도 단계에서 잠금).")
    L.append("- human review 재개 (`reports/step4_research_reframing.md` §7에 의해 main pipeline에서 제외 중).")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 9. 산출물 목록")
    L.append("")
    L.append("- `data/step4r/4rb_attention/step4r_bigru_attention_schema_outputs_calibrated.csv`")
    L.append("- `reports/step4r/4rb_attention/step4r_bigru_attention_schema_summary.csv` (long format: split, metric, value)")
    L.append("- `reports/step4r/4rb_attention/step4r_bigru_attention_schema_results.md` (본 보고서)")
    L.append("")
    L.append("---")
    L.append("")
    L.append("*본 보고서는 자동 생성된다. 기존 Step 1 ~ 4 / 4R-A / 4R-B 산출물은 수정되지 않으며, 기존 `step4r_bigru_attention_schema_outputs_*` 등 다른 모델의 schema 출력 파일은 영향 없음.*")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 64)
    print("Step 4R-B post 2/3 — Calibrated schema output")
    print("=" * 64)
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_NPZ.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_NPZ}. "
            "Run scripts/calibrate_step4r_attention_temperature.py first."
        )

    d = _load_calibrated()
    print(f"  N = {len(d['sample_id'])}, T = {d['temperature']:.6f}")
    anchor_df = _load_anchor_join(d["sample_id"])
    print(f"  joined anchor info from {INPUT_ANCHOR_CSV.name}")

    print("deriving val-based operational thresholds (calibrated posterior)...")
    thr = _derive_thresholds(d["probs_calibrated"], d["y"], d["split"])
    for k, v in thr.items():
        print(f"  {k} = {v:.4f}")

    print("building schema CSV ...")
    schema_df = _build_schema_df(d, anchor_df, thr)
    _validate_schema(schema_df)
    schema_df.to_csv(OUTPUT_SCHEMA_CSV, index=False, encoding="utf-8-sig")
    print(f"saved schema CSV -> {OUTPUT_SCHEMA_CSV}")

    print("building summary ...")
    summary = _build_summary(schema_df)
    summary.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f"saved summary -> {OUTPUT_SUMMARY_CSV}")

    md = _markdown_report(schema_df, thr, d["temperature"])
    OUTPUT_REPORT_MD.write_text(md, encoding="utf-8")
    print(f"saved report -> {OUTPUT_REPORT_MD}")

    print()
    print("split-wise no_call rate / caption_confidence_level distribution:")
    for s in SPLITS:
        sub = schema_df[schema_df["split"] == s]
        n = len(sub)
        nc = int(sub["no_call"].astype(bool).sum())
        levels = sub["caption_confidence_level"].value_counts().to_dict()
        msg = (
            f"  {s:5s} n={n:4d}  no_call={nc:4d} ({nc / n:.4f})  "
            f"levels={ {k: int(v) for k, v in levels.items()} }"
        )
        print(msg)

    print()
    print("=" * 64)
    print("Done.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
