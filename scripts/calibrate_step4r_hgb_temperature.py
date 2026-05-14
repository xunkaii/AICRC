"""Step 4R-A post-hoc — Temperature scaling for HistGradientBoosting (HGB) ceiling.

This is the 4R-A analogue of `scripts/calibrate_step4r_attention_temperature.py`.
Goal: bring 4R-A and 4R-B under the *same* 1-parameter post-hoc calibration
narrative (single scalar temperature T fit on val NLL, applied to all splits).

Reads (read-only):
    data/step4r/4ra_feature_ceiling/step4r_hgb_predictions_raw.csv
    data/step4r/4ra_feature_ceiling/step4r_hgb_predictions_zscore.csv

Writes (NEW paths only; existing 4R-A outputs are NOT touched):
    data/step4r/4ra_feature_ceiling/calibrated/
        step4r_hgb_predictions_calibrated_raw.csv
        step4r_hgb_predictions_calibrated_zscore.csv
    reports/step4r/4ra_feature_ceiling/calibrated/
        step4r_hgb_temperature_scaling_metrics.csv
        step4r_hgb_temperature_scaling_results.md

Method:
    HGB outputs `predict_proba` directly (post-softmax probabilities). Because
    `softmax(log(p) / T)` equals `softmax(z / T)` where z is the pre-softmax
    score (the difference cancels in softmax), `log(predict_proba)` serves as
    mathematically valid logits for temperature scaling.

    Numerical safety: clip probs at 1e-30 before log.

    Fit: scipy L-BFGS-B on log_T (1D) minimizing val NLL. test split is never
    used for fitting.

Run:
    python -X utf8 scripts/calibrate_step4r_hgb_temperature.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "data" / "step4r" / "4ra_feature_ceiling"
OUTPUT_DATA_DIR = INPUT_DIR / "calibrated"
OUTPUT_REPORT_DIR = (
    PROJECT_ROOT / "reports" / "step4r" / "4ra_feature_ceiling" / "calibrated"
)

OUTPUT_METRICS_CSV = OUTPUT_REPORT_DIR / "step4r_hgb_temperature_scaling_metrics.csv"
OUTPUT_REPORT_MD = OUTPUT_REPORT_DIR / "step4r_hgb_temperature_scaling_results.md"

BRANCHES = ["raw", "zscore"]
INPUT_PRED = {b: INPUT_DIR / f"step4r_hgb_predictions_{b}.csv" for b in BRANCHES}
OUTPUT_PRED = {
    b: OUTPUT_DATA_DIR / f"step4r_hgb_predictions_calibrated_{b}.csv"
    for b in BRANCHES
}

CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
NUM_CLASSES = len(CLASSES)
P_COLS = [f"p_{c}" for c in CLASSES]
SPLITS = ["train", "val", "test"]


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------

def _softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _log_softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def _entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    pp = np.clip(p, eps, 1.0)
    return -(pp * np.log(pp)).sum(axis=-1)


def _ece(probs: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    top1 = probs.max(axis=1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(top1, edges[1:-1]), 0, n_bins - 1)
    n = len(top1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(
            float(top1[mask].mean()) - float(correct[mask].mean())
        )
    return float(ece)


def _multiclass_brier(probs: np.ndarray, y_idx: np.ndarray) -> float:
    y_oh = np.zeros((len(y_idx), NUM_CLASSES), dtype=float)
    y_oh[np.arange(len(y_idx)), y_idx] = 1.0
    return float(((probs - y_oh) ** 2).sum(axis=1).mean())


# ---------------------------------------------------------------------------
# Temperature fitting
# ---------------------------------------------------------------------------

def _fit_temperature(val_logits: np.ndarray, val_y: np.ndarray) -> tuple[float, dict]:
    """Fit scalar T > 0 on val NLL using L-BFGS-B over log_T.

    NLL = -mean( log_softmax(logits / T)[range(n), y] )
    """
    val_logits = val_logits.astype(np.float64)
    y_oh_idx = (np.arange(len(val_y)), val_y.astype(np.int64))
    history = {"iters": [], "nll": [], "T": []}

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        log_T = float(x[0])
        T = np.exp(log_T)
        scaled = val_logits / T
        log_p = _log_softmax_np(scaled)
        nll = float(-log_p[y_oh_idx].mean())
        # gradient w.r.t. log_T:
        # dNLL/dT = mean( (1/T) * sum_c (p_c * z_c) - z_y ) * (-1/T)
        # dT/d(log_T) = T
        # so dNLL/d(log_T) = dNLL/dT * T
        # Numerically simpler: finite difference fallback if needed.
        # Here we use exact gradient:
        p = _softmax_np(scaled)
        z = val_logits
        # dNLL/dT (per sample, then averaged): (sum_c p_c z_c - z_y) * (-1/T^2)
        # multiplied by dT/d(log_T) = T  -->  (sum_c p_c z_c - z_y) * (-1/T)
        pz = (p * z).sum(axis=1)
        zy = z[y_oh_idx]
        grad_log_T = float(((pz - zy) * (-1.0 / T)).mean())
        history["iters"].append(len(history["iters"]) + 1)
        history["nll"].append(nll)
        history["T"].append(T)
        return nll, np.array([grad_log_T])

    x0 = np.array([0.0])  # log_T = 0 → T = 1
    res = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-10},
    )
    T_final = float(np.exp(res.x[0]))
    return T_final, history


# ---------------------------------------------------------------------------
# Per-split metrics
# ---------------------------------------------------------------------------

def _split_metrics(probs: np.ndarray, y_idx: np.ndarray) -> dict:
    y_true = np.array([CLASSES[i] for i in y_idx])
    pred_idx = probs.argmax(axis=1)
    y_pred = np.array([CLASSES[i] for i in pred_idx])
    correct = (pred_idx == y_idx).astype(float)
    top1 = probs.max(axis=1)
    sorted_p = np.sort(probs, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    pe = _entropy(probs)
    return {
        "n": int(len(y_idx)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0)
        ),
        "log_loss": float(log_loss(y_true, probs, labels=CLASSES)),
        "brier_multiclass": _multiclass_brier(probs, y_idx),
        "ece_15bin": _ece(probs, correct, n_bins=15),
        "top1_prob_mean": float(top1.mean()),
        "top1_prob_p25": float(np.percentile(top1, 25)),
        "top1_prob_p50": float(np.percentile(top1, 50)),
        "top1_prob_p75": float(np.percentile(top1, 75)),
        "top1_top2_margin_mean": float(margin.mean()),
        "top1_top2_margin_p25": float(np.percentile(margin, 25)),
        "top1_top2_margin_p50": float(np.percentile(margin, 50)),
        "top1_top2_margin_p75": float(np.percentile(margin, 75)),
        "predictive_entropy_mean": float(pe.mean()),
        "predictive_entropy_p25": float(np.percentile(pe, 25)),
        "predictive_entropy_p50": float(np.percentile(pe, 50)),
        "predictive_entropy_p75": float(np.percentile(pe, 75)),
    }


# ---------------------------------------------------------------------------
# Process one branch
# ---------------------------------------------------------------------------

def _process_branch(branch: str) -> tuple[float, dict, dict, dict]:
    """Returns (T, metrics_before, metrics_after, fit_history)."""
    print(f"\n[branch={branch}] loading {INPUT_PRED[branch]}")
    df = pd.read_csv(INPUT_PRED[branch], encoding="utf-8-sig")
    if len(df) != 9275:
        raise ValueError(f"unexpected row count for {branch}: {len(df)}")
    probs_raw = df[P_COLS].to_numpy(dtype=np.float64)
    # row-normalize to guard against any minor float drift in the input CSV.
    row_sum = probs_raw.sum(axis=1, keepdims=True)
    probs_raw = probs_raw / np.clip(row_sum, 1e-12, None)
    y_idx = np.array(
        [CLASSES.index(c) for c in df["class_id"].to_numpy()], dtype=np.int64
    )
    split = df["split"].to_numpy()

    # Build logits as log(p), with clamping for numerical safety.
    logits = np.log(np.clip(probs_raw, 1e-30, 1.0))

    val_mask = split == "val"
    if not val_mask.any():
        raise ValueError("val split missing")
    print(f"  fitting T on val (n={int(val_mask.sum())}) ...")
    T, history = _fit_temperature(logits[val_mask], y_idx[val_mask])
    print(
        f"  fitted T = {T:.6f} after {len(history['iters'])} L-BFGS-B iterations; "
        f"final val NLL = {history['nll'][-1]:.6f}"
    )

    logits_cal = logits / T
    probs_cal = _softmax_np(logits_cal)

    metrics_before: dict = {}
    metrics_after: dict = {}
    for s in SPLITS:
        m = split == s
        if not m.any():
            continue
        metrics_before[s] = _split_metrics(probs_raw[m], y_idx[m])
        metrics_after[s] = _split_metrics(probs_cal[m], y_idx[m])

    # Save calibrated predictions CSV (preserve all carry columns from input).
    sorted_idx = np.argsort(-probs_cal, axis=1)
    top1_class = np.array([CLASSES[i] for i in sorted_idx[:, 0]])
    top2_class = np.array([CLASSES[i] for i in sorted_idx[:, 1]])
    top1_prob = np.take_along_axis(probs_cal, sorted_idx[:, :1], axis=1).squeeze(-1)
    top2_prob = np.take_along_axis(probs_cal, sorted_idx[:, 1:2], axis=1).squeeze(-1)
    margin = top1_prob - top2_prob
    pe = _entropy(probs_cal)

    out_df = pd.DataFrame({
        "sample_id": df["sample_id"],
        "rep_id": df["rep_id"],
        "participant_id": df["participant_id"],
        "class_id": df["class_id"],
        "posture": df["posture"],
        "split": df["split"],
        "anchor_reliability": df["anchor_reliability"],
        "anchor_type": df["anchor_type"],
        "pred_argmax_calibrated": top1_class,
        "top1_prob_calibrated": top1_prob,
        "top1_top2_margin_calibrated": margin,
        "predictive_entropy_calibrated": pe,
        **{f"p_{c}_raw": probs_raw[:, i] for i, c in enumerate(CLASSES)},
        **{f"p_{c}_calibrated": probs_cal[:, i] for i, c in enumerate(CLASSES)},
        "temperature": [T] * len(df),
    })
    OUTPUT_PRED[branch].parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PRED[branch], index=False, encoding="utf-8-sig")
    print(f"  saved calibrated predictions -> {OUTPUT_PRED[branch]}")

    return T, metrics_before, metrics_after, history


# ---------------------------------------------------------------------------
# Long-format metrics CSV
# ---------------------------------------------------------------------------

def _build_metrics_long(
    T_by_branch: dict,
    before_by_branch: dict,
    after_by_branch: dict,
    history_by_branch: dict,
) -> pd.DataFrame:
    rows = []
    for branch in BRANCHES:
        for s in SPLITS:
            for stage, m in [
                ("before", before_by_branch[branch].get(s, {})),
                ("after", after_by_branch[branch].get(s, {})),
            ]:
                for k, v in m.items():
                    if k == "n":
                        continue
                    rows.append({
                        "branch": branch, "split": s, "stage": stage,
                        "metric": k, "value": float(v),
                    })
                if m:
                    rows.append({
                        "branch": branch, "split": s, "stage": stage,
                        "metric": "n", "value": float(m["n"]),
                    })
        rows.append({
            "branch": branch, "split": "—", "stage": "—",
            "metric": "fitted_temperature", "value": float(T_by_branch[branch]),
        })
        rows.append({
            "branch": branch, "split": "—", "stage": "—",
            "metric": "lbfgs_n_iters",
            "value": float(len(history_by_branch[branch]["iters"])),
        })
        rows.append({
            "branch": branch, "split": "—", "stage": "—",
            "metric": "final_val_nll",
            "value": float(history_by_branch[branch]["nll"][-1]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _markdown_report(
    T_by_branch: dict,
    before_by_branch: dict,
    after_by_branch: dict,
    history_by_branch: dict,
) -> str:
    L: list[str] = []
    L.append("# Step 4R-A 후처리 — Temperature Scaling 결과")
    L.append("")
    L.append("- 생성 스크립트: `scripts/calibrate_step4r_hgb_temperature.py`")
    L.append("- 입력: `data/step4r/4ra_feature_ceiling/step4r_hgb_predictions_{raw,zscore}.csv`")
    L.append("- 기존 4R-A 산출물은 수정/덮어쓰기 없음. 본 단계 산출물은 모두 `calibrated/` 하위.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 1. 본 단계의 위치")
    L.append("")
    L.append(
        "본 단계는 4R-A HGB ceiling의 over-confidence(test ECE ≈ 0.23)를 scalar "
        "temperature scaling으로 보정하여 4R-B와 **동일한 1-parameter post-hoc "
        "calibration narrative**로 통일한다. argmax/accuracy/macro F1은 monotonic "
        "scaling에서 변하지 않으므로, 본 단계의 효과는 *오직 calibration*에 있다 "
        "(§3 참조)."
    )
    L.append("")
    L.append(
        "HGB는 `predict_proba`만 노출하고 decision_function이 없으므로 "
        "`log(predict_proba)`를 logits로 간주한다. `softmax(log(p)/T)`는 사전 "
        "logits z에 대해 `softmax(z/T)`와 수학적으로 동일하다 (정수 상수가 "
        "softmax에서 상쇄됨). 따라서 본 단계는 4R-B와 동일한 post-hoc "
        "temperature scaling이다."
    )
    L.append("")
    L.append(
        "**fitting 데이터:** validation split만 사용했다. test split은 어떤 형태로도 "
        "fitting에 노출되지 않았다."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 2. fitted temperature")
    L.append("")
    L.append("| branch | T | L-BFGS-B iters | final val NLL |")
    L.append("|---|---:|---:|---:|")
    for b in BRANCHES:
        L.append(
            f"| {b} | {T_by_branch[b]:.6f} | "
            f"{len(history_by_branch[b]['iters'])} | "
            f"{history_by_branch[b]['nll'][-1]:.6f} |"
        )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 3. test ECE before / after (가장 중요한 지표)")
    L.append("")
    for b in BRANCHES:
        L.append(f"### 3.{BRANCHES.index(b) + 1} branch = `{b}`")
        L.append("")
        bf = before_by_branch[b]["test"]
        af = after_by_branch[b]["test"]
        L.append("| 지표 | before | after | 변화 |")
        L.append("|---|---:|---:|---:|")
        for k, label in [
            ("ece_15bin", "ECE (15-bin)"),
            ("log_loss", "log loss"),
            ("brier_multiclass", "Brier (multi)"),
            ("top1_prob_mean", "top1 prob mean"),
            ("predictive_entropy_mean", "predictive entropy mean"),
        ]:
            L.append(
                f"| {label} | {bf[k]:.4f} | {af[k]:.4f} | {af[k] - bf[k]:+.4f} |"
            )
        L.append("")
    L.append("---")
    L.append("")
    L.append("## 4. Argmax 불변성 — accuracy / macro F1 / balanced accuracy")
    L.append("")
    L.append(
        "Temperature scaling은 logits을 양의 scalar로 나누는 monotonic 변환이라 "
        "argmax가 보존된다. accuracy / macro F1 / balanced accuracy는 before/after가 "
        "**수치적으로 동일**해야 한다 (수치 오차 1e-6 이내)."
    )
    L.append("")
    for b in BRANCHES:
        L.append(f"### 4.{BRANCHES.index(b) + 1} branch = `{b}`")
        L.append("")
        L.append("| split | n | acc before | acc after | macro F1 before | macro F1 after |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for s in SPLITS:
            bf = before_by_branch[b][s]
            af = after_by_branch[b][s]
            L.append(
                f"| {s} | {bf['n']} | {bf['accuracy']:.4f} | {af['accuracy']:.4f} | "
                f"{bf['macro_f1']:.4f} | {af['macro_f1']:.4f} |"
            )
        L.append("")
    L.append("---")
    L.append("")
    L.append("## 5. 분포 지표 before / after (전체 split)")
    L.append("")
    for b in BRANCHES:
        L.append(f"### 5.{BRANCHES.index(b) + 1} branch = `{b}`")
        L.append("")
        for s in SPLITS:
            L.append(f"#### {s} split")
            L.append("")
            bf = before_by_branch[b][s]
            af = after_by_branch[b][s]
            L.append("| 지표 | before | after | 변화 |")
            L.append("|---|---:|---:|---:|")
            for k, label in [
                ("log_loss", "log loss"),
                ("brier_multiclass", "Brier (multi)"),
                ("ece_15bin", "ECE (15-bin)"),
                ("top1_prob_mean", "top1 prob mean"),
                ("top1_prob_p25", "top1 prob p25"),
                ("top1_prob_p50", "top1 prob p50"),
                ("top1_prob_p75", "top1 prob p75"),
                ("top1_top2_margin_mean", "top1-top2 margin mean"),
                ("predictive_entropy_mean", "predictive entropy mean"),
            ]:
                L.append(
                    f"| {label} | {bf[k]:.4f} | {af[k]:.4f} | {af[k] - bf[k]:+.4f} |"
                )
            L.append("")
    L.append("---")
    L.append("")
    L.append("## 6. 해석")
    L.append("")
    for b in BRANCHES:
        T = T_by_branch[b]
        if T > 1.0:
            L.append(
                f"- `{b}` branch: T = {T:.4f} > 1 → 원본 HGB가 **over-confident**였다. "
                "calibrated posterior는 더 평평해지고 top1 mean이 감소, predictive "
                "entropy가 증가한다."
            )
        elif T < 1.0:
            L.append(
                f"- `{b}` branch: T = {T:.4f} < 1 → 원본 HGB가 **under-confident**였다. "
                "calibrated posterior는 더 뾰족해지고 top1 mean이 증가, predictive "
                "entropy가 감소한다."
            )
        else:
            L.append(f"- `{b}` branch: T = {T:.4f} ≈ 1 → calibration 변화가 거의 없음.")
    L.append("")
    L.append(
        "- argmax / accuracy / macro F1은 변하지 않는다 (§4). 본 단계의 효과는 분류 "
        "결정이 아니라 *불확실성 표현*에 있다."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 7. 4R-B 후처리와의 정합성")
    L.append("")
    L.append(
        "본 단계는 `scripts/calibrate_step4r_attention_temperature.py`와 동일한 "
        "프로토콜이다: (a) val NLL만 사용, (b) test 미노출, (c) scalar T, (d) "
        "1-parameter L-BFGS-B (4R-B는 torch LBFGS, 4R-A는 scipy L-BFGS-B로 "
        "구현이 다르지만 수학적으로 동일한 1D 무제약 최적화). 두 모델 모두 "
        "*같은 calibration 패밀리*에 속하므로 4R-A vs 4R-B 비교가 calibration "
        "비대칭 없이 가능하다."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 8. 다음 단계 — schema 재생성")
    L.append("")
    L.append(
        "후속 스크립트 `scripts/generate_step4r_hgb_schema_calibrated.py`가 본 "
        "단계의 `step4r_hgb_predictions_calibrated_{raw,zscore}.csv`를 입력으로 "
        "받아, val 기반 operational threshold를 자체 산출하고 Step 3 §3~§9 schema "
        "를 재구성한다. 기존 `step4r_hgb_schema_outputs_*.csv`(LR-derived "
        "threshold 사용)는 *legacy reference*로 보존된다."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 9. 산출물 목록")
    L.append("")
    L.append("- `data/step4r/4ra_feature_ceiling/calibrated/step4r_hgb_predictions_calibrated_raw.csv`")
    L.append("- `data/step4r/4ra_feature_ceiling/calibrated/step4r_hgb_predictions_calibrated_zscore.csv`")
    L.append("- `reports/step4r/4ra_feature_ceiling/calibrated/step4r_hgb_temperature_scaling_metrics.csv` (long format: branch, split, stage, metric, value)")
    L.append("- `reports/step4r/4ra_feature_ceiling/calibrated/step4r_hgb_temperature_scaling_results.md` (본 보고서)")
    L.append("")
    L.append("---")
    L.append("")
    L.append(
        "*본 보고서는 자동 생성된다. 기존 Step 1~4 / 4R-A / 4R-B 산출물은 수정되지 "
        "않으며, 기존 `step4r_hgb_predictions_*.csv`는 덮어쓰지 않는다.*"
    )
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 64)
    print("Step 4R-A post-hoc — Temperature scaling (HGB)")
    print("=" * 64)
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for b in BRANCHES:
        if not INPUT_PRED[b].exists():
            raise FileNotFoundError(f"Input predictions not found: {INPUT_PRED[b]}")

    T_by_branch: dict = {}
    before_by_branch: dict = {}
    after_by_branch: dict = {}
    history_by_branch: dict = {}

    for b in BRANCHES:
        T, mb, ma, hist = _process_branch(b)
        T_by_branch[b] = T
        before_by_branch[b] = mb
        after_by_branch[b] = ma
        history_by_branch[b] = hist

    print()
    print(f"{'branch':<8} {'split':<6} {'stage':<6} {'acc':>7} {'macroF1':>9} {'logloss':>9} {'Brier':>7} {'ECE':>7}")
    for b in BRANCHES:
        for s in SPLITS:
            bf, af = before_by_branch[b][s], after_by_branch[b][s]
            print(
                f"{b:<8} {s:<6} {'before':<6} {bf['accuracy']:>7.4f} "
                f"{bf['macro_f1']:>9.4f} {bf['log_loss']:>9.4f} "
                f"{bf['brier_multiclass']:>7.4f} {bf['ece_15bin']:>7.4f}"
            )
            print(
                f"{b:<8} {s:<6} {'after':<6} {af['accuracy']:>7.4f} "
                f"{af['macro_f1']:>9.4f} {af['log_loss']:>9.4f} "
                f"{af['brier_multiclass']:>7.4f} {af['ece_15bin']:>7.4f}"
            )

    metrics_df = _build_metrics_long(
        T_by_branch, before_by_branch, after_by_branch, history_by_branch
    )
    metrics_df.to_csv(OUTPUT_METRICS_CSV, index=False, encoding="utf-8-sig")
    print(f"\nsaved metrics -> {OUTPUT_METRICS_CSV}")

    md = _markdown_report(
        T_by_branch, before_by_branch, after_by_branch, history_by_branch
    )
    OUTPUT_REPORT_MD.write_text(md, encoding="utf-8")
    print(f"saved report -> {OUTPUT_REPORT_MD}")

    print()
    print("=" * 64)
    print(
        "Done. T(raw) = {:.6f}, T(zscore) = {:.6f}".format(
            T_by_branch["raw"], T_by_branch["zscore"]
        )
    )
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
