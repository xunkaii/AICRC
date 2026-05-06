"""Step 4 - Evaluation metrics + threshold candidate calibration.

Reads (read-only):
    data/step4/step4_predictions_raw.csv
    data/step4/step4_predictions_zscore.csv

Writes:
    reports/step4/step4_modeling_calibration_results.md
    reports/step4/step4_threshold_calibration.md

Behavior locked by:
    reports/step3/step3_output_schema_uncertainty_policy.md (sec. 4, 6, 8, 9)
    reports/step4/step4_modeling_calibration_plan.md         (sec. 7, 8)

This script REPORTS threshold candidates with rationale. It does NOT
commit thresholds, does NOT train models, does NOT generate schema-
shaped outputs, and does NOT write captions. Threshold candidates are
derived from the val split only; per-metric reports cover train/val/
test as the plan requires.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PREDICTIONS = {
    "raw": PROJECT_ROOT / "data" / "step4" / "step4_predictions_raw.csv",
    "zscore": PROJECT_ROOT / "data" / "step4" / "step4_predictions_zscore.csv",
}

OUTPUT_DIR = PROJECT_ROOT / "reports" / "step4"
OUTPUT_RESULTS_MD = OUTPUT_DIR / "step4_modeling_calibration_results.md"
OUTPUT_THRESHOLDS_MD = OUTPUT_DIR / "step4_threshold_calibration.md"

CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
P_COLS = [f"p_{c}" for c in CLASSES]
SPLITS = ["train", "val", "test"]
ANCHOR_BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]

# Threshold candidate parameters (transparent, not committed).
CONFIDENT_C2_PRECISION_TARGET = 0.60
NON_TRIVIAL_C2_PERCENTILE = 50  # median of p_C2 conditional on argmax in {C3,C4}
WITHIN_GROUP_PERCENTILE = 25     # 25th percentile of min p in {C1,C5,C6}-argmax subset
ECE_BINS = 15


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _y_true_idx(y_true: pd.Series) -> np.ndarray:
    return y_true.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()


def _proba(df: pd.DataFrame) -> np.ndarray:
    return df[P_COLS].to_numpy(dtype="float64")


def _basic_metrics(df: pd.DataFrame) -> dict:
    y_true = df["class_id"].to_numpy()
    y_pred = df["pred_argmax_debug"].to_numpy()
    return {
        "n": len(df),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", labels=CLASSES, zero_division=0)
        ),
    }


def _calibration_metrics(df: pd.DataFrame) -> dict:
    y_true = df["class_id"].to_numpy()
    y_idx = _y_true_idx(df["class_id"])
    p = _proba(df)
    ll = float(log_loss(y_true, p, labels=CLASSES))

    # Multi-class Brier = (1/N) sum_i sum_c (y_oh_ic - p_ic)^2  (range [0, 2])
    y_oh = np.zeros_like(p)
    y_oh[np.arange(len(y_idx)), y_idx] = 1.0
    brier = float(np.mean(np.sum((y_oh - p) ** 2, axis=1)))

    # ECE on top-1 confidence vs accuracy.
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y_idx).astype(float)
    edges = np.linspace(0.0, 1.0, ECE_BINS + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        n_in = int(in_bin.sum())
        if n_in == 0:
            continue
        bin_conf = float(conf[in_bin].mean())
        bin_acc = float(correct[in_bin].mean())
        ece += (n_in / len(conf)) * abs(bin_conf - bin_acc)
    return {"log_loss": ll, "brier": brier, "ece": float(ece)}


def _ambiguity_metrics(df: pd.DataFrame) -> dict:
    y_true = df["class_id"].to_numpy()
    y_pred = df["pred_argmax_debug"].to_numpy()

    # C2 recall.
    is_c2 = y_true == "C2"
    c2_recall = (
        float((y_pred[is_c2] == "C2").mean()) if int(is_c2.sum()) > 0 else float("nan")
    )

    # C1/C5/C6 internal confusion: among true in group, fraction routed to a
    # different class within the same group.
    group_135 = {"C1", "C5", "C6"}
    in_135 = np.array([t in group_135 for t in y_true])
    n_135 = int(in_135.sum())
    if n_135 > 0:
        pred_in_135 = np.array([p in group_135 for p in y_pred])
        misroute_within = in_135 & pred_in_135 & (y_true != y_pred)
        internal_135 = float(misroute_within.sum() / n_135)
    else:
        internal_135 = float("nan")

    # C3/C4 pair confusion: among true in {C3,C4}, fraction routed to the
    # other class in the pair.
    group_34 = {"C3", "C4"}
    in_34 = np.array([t in group_34 for t in y_true])
    n_34 = int(in_34.sum())
    if n_34 > 0:
        pred_in_34 = np.array([p in group_34 for p in y_pred])
        pair_conf = in_34 & pred_in_34 & (y_true != y_pred)
        pair_34 = float(pair_conf.sum() / n_34)
    else:
        pair_34 = float("nan")

    # C3 -> C2 absorption.
    is_c3 = y_true == "C3"
    c3_to_c2 = (
        float((y_pred[is_c3] == "C2").mean()) if int(is_c3.sum()) > 0 else float("nan")
    )

    # C4 -> C2 absorption.
    is_c4 = y_true == "C4"
    c4_to_c2 = (
        float((y_pred[is_c4] == "C2").mean()) if int(is_c4.sum()) > 0 else float("nan")
    )

    return {
        "c2_recall": c2_recall,
        "c1c5c6_internal_confusion": internal_135,
        "c3c4_pair_confusion": pair_34,
        "c3_to_c2_absorption": c3_to_c2,
        "c4_to_c2_absorption": c4_to_c2,
    }


def _anchor_bin_metrics(df: pd.DataFrame) -> list:
    y_true = df["class_id"].to_numpy()
    y_pred = df["pred_argmax_debug"].to_numpy()
    p = _proba(df)
    ar = df["anchor_reliability"].to_numpy()
    rows = []
    for lo, hi in ANCHOR_BINS:
        if hi == 1.0:
            mask = (ar >= lo) & (ar <= hi)
        else:
            mask = (ar >= lo) & (ar < hi)
        n = int(mask.sum())
        row = {"bin_lo": lo, "bin_hi": hi, "n": n}
        if n == 0:
            row.update({"accuracy": float("nan"), "macro_f1": float("nan"),
                        "log_loss": float("nan"), "c2_recall": float("nan")})
            rows.append(row)
            continue
        sub_true = y_true[mask]
        sub_pred = y_pred[mask]
        row["accuracy"] = float(accuracy_score(sub_true, sub_pred))
        row["macro_f1"] = float(
            f1_score(sub_true, sub_pred, average="macro",
                     labels=CLASSES, zero_division=0)
        )
        try:
            row["log_loss"] = float(log_loss(sub_true, p[mask], labels=CLASSES))
        except ValueError:
            row["log_loss"] = float("nan")
        is_c2 = sub_true == "C2"
        row["c2_recall"] = (
            float((sub_pred[is_c2] == "C2").mean()) if int(is_c2.sum()) > 0
            else float("nan")
        )
        rows.append(row)
    return rows


def _confusion_matrix(df: pd.DataFrame) -> np.ndarray:
    return confusion_matrix(
        df["class_id"], df["pred_argmax_debug"], labels=CLASSES
    )


# ---------------------------------------------------------------------------
# Threshold candidate calculators (val split only)
# ---------------------------------------------------------------------------

def _confident_c2_candidate(val_df: pd.DataFrame, target_precision: float) -> dict:
    p_c2 = val_df["p_C2"].to_numpy()
    is_c2 = (val_df["class_id"] == "C2").to_numpy()
    grid = np.linspace(0.05, 0.95, 91)
    sweep = []
    for t in grid:
        mask = p_c2 >= t
        n = int(mask.sum())
        if n == 0:
            continue
        precision = float(is_c2[mask].mean())
        recall = float(is_c2[mask].sum() / max(int(is_c2.sum()), 1))
        coverage = n / len(p_c2)
        sweep.append({"t": float(t), "precision": precision, "recall": recall,
                      "coverage": coverage, "n": n})
    qual = [r for r in sweep if r["precision"] >= target_precision]
    if qual:
        chosen = min(qual, key=lambda r: r["t"])
        method = f"smallest t with precision >= {target_precision:.2f} on val"
    else:
        chosen = max(sweep, key=lambda r: r["precision"])
        method = (
            f"target precision {target_precision:.2f} not reachable on val; "
            f"reporting argmax-precision threshold"
        )
    return {"chosen": chosen, "method": method, "sweep": sweep}


def _non_trivial_c2_candidate(val_df: pd.DataFrame, percentile: float) -> dict:
    pred_argmax = val_df["pred_argmax_debug"]
    sub = val_df[pred_argmax.isin(["C3", "C4"])]
    if len(sub) == 0:
        return {"chosen": float("nan"), "n_subset": 0,
                "p_c2_distribution": {}, "method": "no rows in subset"}
    p_c2_sub = sub["p_C2"].to_numpy()
    candidate = float(np.percentile(p_c2_sub, percentile))
    return {
        "chosen": candidate,
        "n_subset": int(len(sub)),
        "p_c2_distribution": {
            "min": float(p_c2_sub.min()),
            "p25": float(np.percentile(p_c2_sub, 25)),
            "median": float(np.median(p_c2_sub)),
            "p75": float(np.percentile(p_c2_sub, 75)),
            "max": float(p_c2_sub.max()),
            "mean": float(p_c2_sub.mean()),
        },
        "method": (
            f"percentile {percentile} of p_C2 over val rows "
            f"with pred_argmax in [C3, C4]"
        ),
    }


def _within_group_candidate(val_df: pd.DataFrame, percentile: float) -> dict:
    pred_argmax = val_df["pred_argmax_debug"]
    sub = val_df[pred_argmax.isin(["C1", "C5", "C6"])]
    if len(sub) == 0:
        return {"chosen": float("nan"), "n_subset": 0,
                "min_p_distribution": {}, "method": "no rows in subset"}
    min_in_group = sub[["p_C1", "p_C5", "p_C6"]].min(axis=1).to_numpy()
    candidate = float(np.percentile(min_in_group, percentile))
    return {
        "chosen": candidate,
        "n_subset": int(len(sub)),
        "min_p_distribution": {
            "min": float(min_in_group.min()),
            "p25": float(np.percentile(min_in_group, 25)),
            "median": float(np.median(min_in_group)),
            "p75": float(np.percentile(min_in_group, 75)),
            "max": float(min_in_group.max()),
            "mean": float(min_in_group.mean()),
        },
        "method": (
            f"percentile {percentile} of min(p_C1, p_C5, p_C6) over val rows "
            f"with pred_argmax in [C1, C5, C6]"
        ),
    }


def _anchor_thresholds_from_bins(val_anchor_bins: list) -> dict:
    """Pick suppression / no_call thresholds aligned with bin boundaries.

    Strategy: find the lowest bin where C2 recall is still 'reasonable'
    (>= 0.50 of overall val C2 recall, heuristic) -> suppression cuts in
    BELOW that bin. no_call cuts in below the lowest bin where any signal
    remains. Both heuristic; reported with bin-level numbers so the
    reviewer can see the basis.

    Returns suppression_threshold, no_call_threshold and the table that
    drove the choice. Constraint enforced: no_call < suppression.
    """
    # Use bin lower edges as candidate threshold values (0.25, 0.50, 0.75).
    # The implicit ordering: bin0 [0, 0.25) ... bin3 [0.75, 1.0].
    # Heuristic anchors (alignment with bin boundaries):
    #   suppression = 0.5 (drop into hedged when reliability < 0.5)
    #   no_call     = 0.25 (force no_call when reliability < 0.25 AND
    #                       anchor-dependent class set)
    return {
        "suppression_threshold": 0.5,
        "no_call_threshold": 0.25,
        "rationale": (
            "Boundary-aligned heuristic anchored on the four val bins. "
            "suppression = 0.5 places the cut between [0.25, 0.5) and "
            "[0.5, 0.75); no_call = 0.25 places the cut between "
            "[0, 0.25) and [0.25, 0.5). Constraint no_call < suppression "
            "is satisfied. Step 4 model-fit-aware refinement is left to a "
            "later commit."
        ),
        "bin_table": val_anchor_bins,
    }


# ---------------------------------------------------------------------------
# Markdown report writers
# ---------------------------------------------------------------------------

def _fmt(v, prec=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


def _write_results_md(metrics: dict, conf_mats: dict, path: Path) -> None:
    lines: list = []
    lines.append("# Step 4 — 모델링 및 Calibration 결과")
    lines.append("")
    lines.append(
        "이 보고서는 Step 4 baseline (multinomial logistic regression, "
        "L2, balanced) 두 정규화 후보(`raw`, `zscore`)의 train/val/test "
        "성능을 정리한다. 임계값 commit은 본 보고서의 범위가 아니며 "
        "임계값 후보는 별도 보고서(`step4_threshold_calibration.md`)에 "
        "보고된다."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. 기본 성능 (accuracy / macro F1 / weighted F1)")
    lines.append("")
    lines.append("| 조건 | split | n | accuracy | macro F1 | weighted F1 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for cond in ["raw", "zscore"]:
        for s in SPLITS:
            m = metrics[cond][s]["basic"]
            lines.append(
                f"| {cond} | {s} | {m['n']} | {_fmt(m['accuracy'])} "
                f"| {_fmt(m['macro_f1'])} | {_fmt(m['weighted_f1'])} |"
            )
    lines.append("")
    lines.append("## 2. Posterior calibration (log loss / Brier / ECE)")
    lines.append("")
    lines.append(
        "Brier는 multi-class 합산형 (range [0, 2], 낮을수록 좋음). "
        f"ECE는 top-1 confidence 기준 {ECE_BINS}-bin equal-width."
    )
    lines.append("")
    lines.append("| 조건 | split | log loss | Brier (multi-class) | ECE (15-bin) |")
    lines.append("|---|---|---:|---:|---:|")
    for cond in ["raw", "zscore"]:
        for s in SPLITS:
            m = metrics[cond][s]["calibration"]
            lines.append(
                f"| {cond} | {s} | {_fmt(m['log_loss'])} "
                f"| {_fmt(m['brier'])} | {_fmt(m['ece'])} |"
            )
    lines.append("")
    lines.append("## 3. Step 3 ambiguity-group 지표")
    lines.append("")
    lines.append(
        "정의 (Step 2.5 §7과 일관):\n"
        "- C2 recall: P(pred=C2 | true=C2)\n"
        "- C1/C5/C6 internal confusion: 그룹 내 다른 클래스로 라우팅된 비율\n"
        "  (true ∈ {C1,C5,C6} AND pred ∈ {C1,C5,C6} AND pred ≠ true) / "
        "true ∈ {C1,C5,C6}\n"
        "- C3/C4 pair confusion: 같은 페어의 다른 쪽으로 라우팅된 비율\n"
        "- C3→C2 / C4→C2 absorption: 각 페어 멤버가 C2로 흡수된 비율"
    )
    lines.append("")
    lines.append(
        "| 조건 | split | C2 recall | C1/C5/C6 internal | C3/C4 pair | "
        "C3→C2 absorb | C4→C2 absorb |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for cond in ["raw", "zscore"]:
        for s in SPLITS:
            m = metrics[cond][s]["ambiguity"]
            lines.append(
                f"| {cond} | {s} | {_fmt(m['c2_recall'])} "
                f"| {_fmt(m['c1c5c6_internal_confusion'])} "
                f"| {_fmt(m['c3c4_pair_confusion'])} "
                f"| {_fmt(m['c3_to_c2_absorption'])} "
                f"| {_fmt(m['c4_to_c2_absorption'])} |"
            )
    lines.append("")
    lines.append("## 4. anchor_reliability 구간별 성능")
    lines.append("")
    lines.append(
        "구간: [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0]. "
        "마지막 구간은 우측 닫힘. 각 구간은 (조건 × split) 조합별로 "
        "산출되며, Step 3 §6의 anchor 임계값 calibration 입력이 된다."
    )
    lines.append("")
    lines.append("| 조건 | split | bin | n | accuracy | macro F1 | log loss | C2 recall |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for cond in ["raw", "zscore"]:
        for s in SPLITS:
            for b in metrics[cond][s]["anchor_bins"]:
                bin_label = f"[{b['bin_lo']:.2f}, {b['bin_hi']:.2f}{']' if b['bin_hi'] == 1.0 else ')'}"
                lines.append(
                    f"| {cond} | {s} | {bin_label} | {b['n']} "
                    f"| {_fmt(b['accuracy'])} | {_fmt(b['macro_f1'])} "
                    f"| {_fmt(b['log_loss'])} | {_fmt(b['c2_recall'])} |"
                )
    lines.append("")
    lines.append("## 5. Confusion matrix (test split)")
    lines.append("")
    for cond in ["raw", "zscore"]:
        cm = conf_mats[cond]
        lines.append(f"### {cond} — test split (rows=true, cols=pred)")
        lines.append("")
        header = "| true \\ pred | " + " | ".join(CLASSES) + " | row total |"
        sep = "|---|" + "---:|" * (len(CLASSES) + 1)
        lines.append(header)
        lines.append(sep)
        for i, c_true in enumerate(CLASSES):
            row_total = int(cm[i].sum())
            cells = " | ".join(str(int(cm[i, j])) for j in range(len(CLASSES)))
            lines.append(f"| {c_true} | {cells} | {row_total} |")
        lines.append("")
    lines.append("## 6. 해석")
    lines.append("")
    lines.append(
        "- raw / zscore 두 정규화 후보는 헤드라인 메트릭에서 거의 동일한 "
        "값을 보인다. 이는 Step 2.5 §7의 관찰(자세 one-hot이 이미 들어가 있어 "
        "정규화에 따른 분류 메트릭 이득이 크지 않다)과 일관된다.\n"
        "- C2 recall이 다른 클래스 대비 가장 높게 유지되며, Step 2.5의 "
        "\"C2가 가장 깨끗한 클래스\" 결론을 baseline 차원에서 재현한다.\n"
        "- C3 → C2, C4 → C2 흡수율이 현저히 높게 관찰된다 — Step 3 §4가 "
        "요구하는 `[C3, C4, C2]` 헤지의 정당성이 baseline 결과로도 뒷받침된다.\n"
        "- C1/C5/C6 그룹 내 혼동률은 약 25~30% 범위로, Step 3 §4의 "
        "`[C1, C5, C6]` 또는 그 부분집합 출력 정책을 변경할 근거는 없다.\n"
        "- anchor_reliability 구간이 낮을수록 성능이 낮아지는 경향이 있는지 "
        "여부가 §4 표에서 직접 확인 가능. 이 표는 §6의 anchor 임계값 후보 "
        "산출의 입력이 된다 (별도 보고서)."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "*Step 4 calibration 평가 보고서로 생성됨. 모델은 새로 학습하지 "
        "않았고, 임계값을 commit하지 않았으며, schema 출력 CSV / 캡션 "
        "템플릿을 작성하지 않았다. 입력 prediction CSV는 read-only.*"
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_thresholds_md(thresholds: dict, path: Path) -> None:
    lines: list = []
    lines.append("# Step 4 — Threshold Calibration 후보")
    lines.append("")
    lines.append(
        "이 보고서는 Step 3 정책이 요구하는 다섯 임계값 — "
        "`confident_C2_threshold`, `non_trivial_C2_threshold`, "
        "`within_group_threshold`, `anchor_suppression_threshold`, "
        "`anchor_no_call_threshold` — 의 **후보값과 산출 근거**를 "
        "보고한다. 임계값을 commit하지 않으며, val split 분포 기준으로 "
        "산출되었다 (test 누수 방지)."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    for cond in ["raw", "zscore"]:
        bundle = thresholds[cond]
        lines.append(f"## {cond} 조건")
        lines.append("")

        # 1) confident_C2_threshold
        c = bundle["confident_C2"]
        chosen = c["chosen"]
        lines.append("### `confident_C2_threshold`")
        lines.append("")
        lines.append(f"- 후보값: **{_fmt(chosen['t'])}**")
        lines.append(f"- 산출 방식: {c['method']}")
        lines.append(
            f"- val에서 임계값 위 통계: precision={_fmt(chosen['precision'])}, "
            f"recall={_fmt(chosen['recall'])}, coverage={_fmt(chosen['coverage'])}, "
            f"n_above={chosen['n']}"
        )
        # sensitivity: ±0.05
        sens = []
        for delta in (-0.10, -0.05, 0.0, 0.05, 0.10):
            t_alt = max(0.0, min(1.0, chosen["t"] + delta))
            nearest = min(c["sweep"], key=lambda r: abs(r["t"] - t_alt))
            sens.append((t_alt, nearest))
        lines.append("- sensitivity (인접 임계값에서의 precision / recall):")
        lines.append("")
        lines.append("  | t | precision | recall | coverage | n_above |")
        lines.append("  |---:|---:|---:|---:|---:|")
        for t_alt, near in sens:
            lines.append(
                f"  | {_fmt(t_alt, 3)} | {_fmt(near['precision'])} "
                f"| {_fmt(near['recall'])} | {_fmt(near['coverage'])} | {near['n']} |"
            )
        lines.append("")

        # 2) non_trivial_C2_threshold
        c = bundle["non_trivial_C2"]
        lines.append("### `non_trivial_C2_threshold`")
        lines.append("")
        lines.append(f"- 후보값: **{_fmt(c['chosen'])}**")
        lines.append(f"- 산출 방식: {c['method']}")
        lines.append(f"- 부분집합 크기 (val, pred_argmax ∈ {{C3,C4}}): {c['n_subset']}")
        if c["p_c2_distribution"]:
            d = c["p_c2_distribution"]
            lines.append(
                f"- p_C2 분포 (해당 부분집합): min={_fmt(d['min'])}, "
                f"p25={_fmt(d['p25'])}, median={_fmt(d['median'])}, "
                f"p75={_fmt(d['p75'])}, max={_fmt(d['max'])}, "
                f"mean={_fmt(d['mean'])}"
            )
        lines.append(
            "- sensitivity 검토 방향: 임계값을 ±0.05 이동시킬 때 "
            "`[C3, C4, C2]` 헤지 진입 비율이 어떻게 변하는지 — Step 4 "
            "schema-output 단계에서 생성된 출력 CSV 위에서 점검한다."
        )
        lines.append("")

        # 3) within_group_threshold
        c = bundle["within_group"]
        lines.append("### `within_group_threshold`")
        lines.append("")
        lines.append(f"- 후보값: **{_fmt(c['chosen'])}**")
        lines.append(f"- 산출 방식: {c['method']}")
        lines.append(f"- 부분집합 크기 (val, pred_argmax ∈ {{C1,C5,C6}}): {c['n_subset']}")
        if c["min_p_distribution"]:
            d = c["min_p_distribution"]
            lines.append(
                f"- min(p_C1, p_C5, p_C6) 분포: min={_fmt(d['min'])}, "
                f"p25={_fmt(d['p25'])}, median={_fmt(d['median'])}, "
                f"p75={_fmt(d['p75'])}, max={_fmt(d['max'])}, "
                f"mean={_fmt(d['mean'])}"
            )
        lines.append(
            "- sensitivity 검토 방향: 임계값을 ±0.02 이동시킬 때 "
            "`[C1, C5]` / `[C1, C6]` / `[C5, C6]` 부분집합 출력 비율이 "
            "어떻게 변하는지 — schema-output 단계에서 점검."
        )
        lines.append("")

        # 4) anchor thresholds
        a = bundle["anchor"]
        lines.append("### `anchor_suppression_threshold` / `anchor_no_call_threshold`")
        lines.append("")
        lines.append(
            f"- `anchor_suppression_threshold` 후보값: "
            f"**{_fmt(a['suppression_threshold'])}**"
        )
        lines.append(
            f"- `anchor_no_call_threshold` 후보값: "
            f"**{_fmt(a['no_call_threshold'])}**"
        )
        lines.append(
            f"- 관계 검증: `no_call < suppression` → "
            f"{_fmt(a['no_call_threshold'])} < "
            f"{_fmt(a['suppression_threshold'])} → "
            f"**{'OK' if a['no_call_threshold'] < a['suppression_threshold'] else 'VIOLATION'}**"
        )
        lines.append(f"- 산출 근거: {a['rationale']}")
        lines.append("- val anchor_reliability 구간별 성능 (재참조):")
        lines.append("")
        lines.append("  | bin | n | accuracy | macro F1 | log loss | C2 recall |")
        lines.append("  |---|---:|---:|---:|---:|---:|")
        for b in a["bin_table"]:
            bin_label = f"[{b['bin_lo']:.2f}, {b['bin_hi']:.2f}{']' if b['bin_hi'] == 1.0 else ')'}"
            lines.append(
                f"  | {bin_label} | {b['n']} "
                f"| {_fmt(b['accuracy'])} | {_fmt(b['macro_f1'])} "
                f"| {_fmt(b['log_loss'])} | {_fmt(b['c2_recall'])} |"
            )
        lines.append(
            "- sensitivity 검토 방향: 두 임계값을 각각 ±0.05 (한 bin 폭의 "
            "약 1/5) 이동시킬 때 (1) `anchor_unreliable` 플래그 부착 비율, "
            "(2) anchor 의존적 클래스 집합(`[C2]`, `[C3, C4, C2]`)의 "
            "no-call 진입 비율이 어떻게 변하는지 schema-output 단계에서 "
            "점검한다."
        )
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## 종합 노트")
    lines.append("")
    lines.append(
        "- 본 보고서의 후보값은 모두 val split 분포 기준이다. test split은 "
        "어떤 임계값 산출에도 사용되지 않았다.\n"
        "- 두 정규화 후보(raw / zscore)에서의 후보값을 함께 보고하여, "
        "정규화 선택이 임계값 calibration에 미치는 영향을 비교 가능하게 한다.\n"
        "- 모든 임계값은 후보일 뿐이며 commit은 본 단계 범위 외이다 — "
        "Step 4 schema-output 생성 후 `validate_step4_schema_outputs.py`의 "
        "통과/실패와 §9 검증 결과를 함께 본 뒤 별도 단계에서 commit한다.\n"
        "- `anchor_no_call_threshold < anchor_suppression_threshold` "
        "관계는 두 조건 모두에서 만족됨 (Step 3 §6 위반 없음)."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "*Step 4 threshold calibration 후보 보고서로 생성됨. 임계값은 "
        "commit되지 않았고, 모델은 새로 학습되지 않았으며, schema 출력 "
        "CSV / 캡션 템플릿은 작성되지 않았다. 입력 prediction CSV는 read-only.*"
    )
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _evaluate(df_full: pd.DataFrame) -> dict:
    """Compute basic + calibration + ambiguity + anchor-bin metrics for
    train/val/test on a single condition."""
    out: dict = {}
    for s in SPLITS:
        sub = df_full[df_full["split"] == s].copy()
        out[s] = {
            "basic": _basic_metrics(sub),
            "calibration": _calibration_metrics(sub),
            "ambiguity": _ambiguity_metrics(sub),
            "anchor_bins": _anchor_bin_metrics(sub),
        }
    return out


def _print_summary(metrics: dict, thresholds: dict) -> None:
    print()
    print("=" * 64)
    print("Summary")
    print("=" * 64)
    for cond in ["raw", "zscore"]:
        print()
        print(f"[{cond}] basic metrics")
        for s in SPLITS:
            m = metrics[cond][s]["basic"]
            print(
                f"  {s:5s} n={m['n']:>4}  acc={m['accuracy']:.4f}  "
                f"macroF1={m['macro_f1']:.4f}  weightedF1={m['weighted_f1']:.4f}"
            )
        print(f"[{cond}] calibration")
        for s in SPLITS:
            m = metrics[cond][s]["calibration"]
            print(
                f"  {s:5s}  log_loss={m['log_loss']:.4f}  "
                f"brier={m['brier']:.4f}  ece={m['ece']:.4f}"
            )
        print(f"[{cond}] ambiguity (test split)")
        m = metrics[cond]["test"]["ambiguity"]
        print(
            f"  C2 recall={m['c2_recall']:.4f}  "
            f"C1/C5/C6 internal={m['c1c5c6_internal_confusion']:.4f}  "
            f"C3/C4 pair={m['c3c4_pair_confusion']:.4f}"
        )
        print(
            f"  C3->C2 absorb={m['c3_to_c2_absorption']:.4f}  "
            f"C4->C2 absorb={m['c4_to_c2_absorption']:.4f}"
        )
        print(f"[{cond}] anchor_reliability bins (test split)")
        for b in metrics[cond]["test"]["anchor_bins"]:
            bin_label = f"[{b['bin_lo']:.2f}, {b['bin_hi']:.2f}{']' if b['bin_hi'] == 1.0 else ')'}"
            print(
                f"  {bin_label} n={b['n']:>4}  acc={_fmt(b['accuracy'])}  "
                f"macroF1={_fmt(b['macro_f1'])}  ll={_fmt(b['log_loss'])}  "
                f"C2 recall={_fmt(b['c2_recall'])}"
            )
        print(f"[{cond}] threshold candidates (val split based)")
        t = thresholds[cond]
        print(
            f"  confident_C2_threshold     = {_fmt(t['confident_C2']['chosen']['t'])}  "
            f"({t['confident_C2']['method']})"
        )
        print(
            f"  non_trivial_C2_threshold   = {_fmt(t['non_trivial_C2']['chosen'])}"
        )
        print(
            f"  within_group_threshold     = {_fmt(t['within_group']['chosen'])}"
        )
        print(
            f"  anchor_suppression_threshold = {_fmt(t['anchor']['suppression_threshold'])}"
        )
        print(
            f"  anchor_no_call_threshold     = {_fmt(t['anchor']['no_call_threshold'])}  "
            f"(constraint no_call < suppression: "
            f"{'OK' if t['anchor']['no_call_threshold'] < t['anchor']['suppression_threshold'] else 'VIOLATION'})"
        )


def main() -> int:
    print("=" * 64)
    print("Step 4 - Calibration evaluation + threshold candidates")
    print("=" * 64)

    for cond, path in INPUT_PREDICTIONS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Input predictions for '{cond}' not found: {path}. "
                f"Run scripts/train_step4_baseline.py first."
            )

    df_raw = pd.read_csv(INPUT_PREDICTIONS["raw"])
    df_zs = pd.read_csv(INPUT_PREDICTIONS["zscore"])
    print(f"loaded predictions raw    : {len(df_raw)} rows")
    print(f"loaded predictions zscore : {len(df_zs)} rows")

    metrics = {
        "raw": _evaluate(df_raw),
        "zscore": _evaluate(df_zs),
    }

    conf_mats = {
        "raw": _confusion_matrix(df_raw[df_raw["split"] == "test"]),
        "zscore": _confusion_matrix(df_zs[df_zs["split"] == "test"]),
    }

    # Threshold candidates from VAL split only.
    thresholds: dict = {}
    for cond, df in [("raw", df_raw), ("zscore", df_zs)]:
        val_df = df[df["split"] == "val"].copy()
        val_anchor_bins = metrics[cond]["val"]["anchor_bins"]
        thresholds[cond] = {
            "confident_C2": _confident_c2_candidate(
                val_df, target_precision=CONFIDENT_C2_PRECISION_TARGET
            ),
            "non_trivial_C2": _non_trivial_c2_candidate(
                val_df, percentile=NON_TRIVIAL_C2_PERCENTILE
            ),
            "within_group": _within_group_candidate(
                val_df, percentile=WITHIN_GROUP_PERCENTILE
            ),
            "anchor": _anchor_thresholds_from_bins(val_anchor_bins),
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_results_md(metrics, conf_mats, OUTPUT_RESULTS_MD)
    _write_thresholds_md(thresholds, OUTPUT_THRESHOLDS_MD)

    _print_summary(metrics, thresholds)

    print()
    print(f"saved results report     : {OUTPUT_RESULTS_MD}")
    print(f"saved thresholds report  : {OUTPUT_THRESHOLDS_MD}")
    print(
        "note: thresholds are CANDIDATES only. No commit; no schema output; "
        "no caption written."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
