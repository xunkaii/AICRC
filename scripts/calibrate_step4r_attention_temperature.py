"""Step 4R-B post-processing 1/3 — Temperature scaling for BiGRU+Attention.

Reads (read-only):
    data/step4r/4rb_attention/step4r_sequence_dataset.npz
    checkpoints/step4r/4rb_attention/best_bigru_attention.pt
    models/step4r_attention_rnn.py

Writes:
    data/step4r/4rb_attention/step4r_bigru_attention_logits_calibrated.npz
    data/step4r/4rb_attention/step4r_bigru_attention_predictions_calibrated.csv
    reports/step4r/4rb_attention/step4r_temperature_scaling_metrics.csv
    reports/step4r/4rb_attention/step4r_temperature_scaling_results.md

Behavior:
    - Loads best checkpoint and recomputes logits + attention on all splits.
    - Fits scalar temperature T on **val split only** (NLL objective, LBFGS over
      log_T to guarantee positivity).
    - test split is NEVER used for fitting.
    - Saves before/after metrics. Argmax / accuracy / macro F1 do not change
      under monotonic scaling — this is verified in metrics output.

Run:
    python scripts/calibrate_step4r_attention_temperature.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.step4r_attention_rnn import Step4RBiGRUAttention  # noqa: E402


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

INPUT_NPZ = PROJECT_ROOT / "data" / "step4r" / "4rb_attention" / "step4r_sequence_dataset.npz"
INPUT_CKPT = PROJECT_ROOT / "checkpoints" / "step4r" / "4rb_attention" / "best_bigru_attention.pt"

OUTPUT_DATA_DIR = PROJECT_ROOT / "data" / "step4r" / "4rb_attention"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "reports" / "step4r" / "4rb_attention"

OUTPUT_NPZ = OUTPUT_DATA_DIR / "step4r_bigru_attention_logits_calibrated.npz"
OUTPUT_PREDS_CSV = OUTPUT_DATA_DIR / "step4r_bigru_attention_predictions_calibrated.csv"
OUTPUT_METRICS_CSV = OUTPUT_REPORT_DIR / "step4r_temperature_scaling_metrics.csv"
OUTPUT_REPORT_MD = OUTPUT_REPORT_DIR / "step4r_temperature_scaling_results.md"

CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
NUM_CLASSES = len(CLASSES)
SPLITS = ["train", "val", "test"]
POSTURES = ["SA", "CA", "HW"]
POSTURE_TO_IDX = {p: i for i, p in enumerate(POSTURES)}
BATCH_SIZE = 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _posture_onehot(posture_str_arr: np.ndarray) -> np.ndarray:
    n = len(posture_str_arr)
    out = np.zeros((n, len(POSTURES)), dtype=np.float32)
    for i, p in enumerate(posture_str_arr):
        out[i, POSTURE_TO_IDX[p]] = 1.0
    return out


def _entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


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
        ece += (mask.sum() / n) * abs(float(top1[mask].mean()) - float(correct[mask].mean()))
    return float(ece)


def _multiclass_brier(probs: np.ndarray, y_idx: np.ndarray, n_classes: int) -> float:
    y_oh = np.zeros((len(y_idx), n_classes), dtype=float)
    y_oh[np.arange(len(y_idx)), y_idx] = 1.0
    return float(((probs - y_oh) ** 2).sum(axis=1).mean())


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
        "brier_multiclass": _multiclass_brier(probs, y_idx, NUM_CLASSES),
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
# Load / infer
# ---------------------------------------------------------------------------

def _load_dataset() -> dict:
    print(f"loading {INPUT_NPZ}")
    z = np.load(INPUT_NPZ, allow_pickle=True)
    return {
        "X_norm": z["X_norm"].astype(np.float32),
        "y": z["y"].astype(np.int64),
        "class_id": z["class_id"].astype(str),
        "posture_canonical": z["posture_canonical"].astype(str),
        "sample_id": z["sample_id"].astype(str),
        "participant_id": z["participant_id"].astype(str),
        "split": z["split"].astype(str),
    }


def _infer_logits_attention(
    model: nn.Module,
    X: np.ndarray,
    posture_oh: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits (N, C), attention_weights (N, T))."""
    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(posture_oh).float(),
        torch.from_numpy(y).long(),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    logits_list, attn_list = [], []
    model.eval()
    with torch.no_grad():
        for X_b, P_b, _ in loader:
            X_b = X_b.to(device, non_blocking=True)
            P_b = P_b.to(device, non_blocking=True)
            logits, attn_w = model(X_b, P_b)
            logits_list.append(logits.cpu().numpy())
            attn_list.append(attn_w.cpu().numpy())
    return np.concatenate(logits_list, axis=0), np.concatenate(attn_list, axis=0)


# ---------------------------------------------------------------------------
# Temperature fitting
# ---------------------------------------------------------------------------

def _fit_temperature(val_logits: np.ndarray, val_y: np.ndarray, device: torch.device) -> tuple[float, dict]:
    """Fit scalar T > 0 on val NLL via LBFGS over log_T."""
    logits_t = torch.from_numpy(val_logits).float().to(device)
    y_t = torch.from_numpy(val_y).long().to(device)
    log_T = torch.zeros(1, requires_grad=True, device=device)
    optimizer = torch.optim.LBFGS([log_T], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    history = {"iters": [], "nll": [], "T": []}
    iters = {"n": 0}

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        T = log_T.exp()
        nll = F.cross_entropy(logits_t / T, y_t)
        nll.backward()
        iters["n"] += 1
        history["iters"].append(iters["n"])
        history["nll"].append(float(nll.item()))
        history["T"].append(float(T.item()))
        return nll

    optimizer.step(closure)
    T_final = float(log_T.exp().item())
    print(
        f"  fitted T = {T_final:.6f} after {iters['n']} closure evaluations; "
        f"final val NLL = {history['nll'][-1]:.6f}"
    )
    return T_final, history


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _markdown_report(
    T: float,
    metrics_before: dict,
    metrics_after: dict,
    val_history: dict,
    device_str: str,
) -> str:
    L = []
    L.append("# Step 4R-B 후처리 1/3 — Temperature Scaling 결과")
    L.append("")
    L.append("- 생성 스크립트: `scripts/calibrate_step4r_attention_temperature.py`")
    L.append("- 입력 모델: `checkpoints/step4r/4rb_attention/best_bigru_attention.pt`")
    L.append(f"- 실행 device: `{device_str}`")
    L.append(f"- **fitted temperature T = {T:.6f}**")
    L.append(
        f"- LBFGS closure 평가 횟수: {len(val_history['iters'])}, "
        f"최종 val NLL: {val_history['nll'][-1]:.6f}"
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 1. 본 단계의 위치")
    L.append("")
    L.append(
        "본 단계는 4R-B BiGRU+Attention의 over-confidence 문제(test ECE ≈ 0.13)를 "
        "scalar temperature scaling으로 보정한다. 분류 정확도와 macro F1은 monotonic "
        "scaling에서 변하지 않으므로, 본 단계의 효과는 *오직 calibration*에 있다 (§4 참조)."
    )
    L.append("")
    L.append(
        "**fitting 데이터:** validation split만 사용했다. test split은 어떤 형태로도 "
        "fitting에 노출되지 않았다."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 2. test ECE before / after (가장 중요한 지표)")
    L.append("")
    bf_test = metrics_before["test"]
    af_test = metrics_after["test"]
    L.append("| 지표 | before | after | 변화 |")
    L.append("|---|---:|---:|---:|")
    for k, label in [
        ("ece_15bin", "ECE (15-bin)"),
        ("log_loss", "log loss"),
        ("brier_multiclass", "Brier (multi)"),
        ("top1_prob_mean", "top1 prob mean"),
        ("predictive_entropy_mean", "predictive entropy mean"),
    ]:
        L.append(f"| {label} | {bf_test[k]:.4f} | {af_test[k]:.4f} | {af_test[k] - bf_test[k]:+.4f} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 3. Argmax 불변성 — accuracy / macro F1 / balanced accuracy")
    L.append("")
    L.append(
        "Temperature scaling은 logits을 양의 scalar로 나누는 monotonic 변환이라 argmax가 "
        "보존된다. 따라서 accuracy, macro F1, balanced accuracy, confusion matrix는 "
        "before/after가 **수치적으로 동일**해야 한다 (수치 오차 1e-6 이내)."
    )
    L.append("")
    L.append("| split | n | acc before | acc after | macro F1 before | macro F1 after | balanced before | balanced after |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in SPLITS:
        b, a = metrics_before[s], metrics_after[s]
        L.append(
            f"| {s} | {b['n']} | {b['accuracy']:.4f} | {a['accuracy']:.4f} | "
            f"{b['macro_f1']:.4f} | {a['macro_f1']:.4f} | "
            f"{b['balanced_accuracy']:.4f} | {a['balanced_accuracy']:.4f} |"
        )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 4. 분포 지표 before / after (전체 split)")
    L.append("")
    for s in SPLITS:
        L.append(f"### 4.{SPLITS.index(s) + 1} {s} split")
        L.append("")
        b, a = metrics_before[s], metrics_after[s]
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
            ("top1_top2_margin_p25", "top1-top2 margin p25"),
            ("top1_top2_margin_p50", "top1-top2 margin p50"),
            ("top1_top2_margin_p75", "top1-top2 margin p75"),
            ("predictive_entropy_mean", "predictive entropy mean"),
            ("predictive_entropy_p25", "predictive entropy p25"),
            ("predictive_entropy_p50", "predictive entropy p50"),
            ("predictive_entropy_p75", "predictive entropy p75"),
        ]:
            L.append(f"| {label} | {b[k]:.4f} | {a[k]:.4f} | {a[k] - b[k]:+.4f} |")
        L.append("")
    L.append("---")
    L.append("")
    L.append("## 5. 해석")
    L.append("")
    if T > 1.0:
        L.append(
            f"- T = {T:.4f} > 1 → 원본 모델이 **over-confident**였다. "
            "calibrated posterior는 분포가 더 평평해지고, top1 probability mean이 "
            "감소한다. predictive entropy는 증가한다."
        )
    elif T < 1.0:
        L.append(
            f"- T = {T:.4f} < 1 → 원본 모델이 **under-confident**였다. "
            "calibrated posterior는 더 뾰족해지고, top1 probability mean이 "
            "증가한다. predictive entropy는 감소한다."
        )
    else:
        L.append(f"- T = {T:.4f} ≈ 1 → calibration 변화가 거의 없음.")
    L.append("")
    L.append(
        "- argmax / accuracy / macro F1은 변하지 않는다 (§3). 본 단계의 효과는 분류 "
        "결정이 아니라 *불확실성 표현*에 있다."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 6. 다운스트림 — schema output에는 calibrated posterior를 사용한다")
    L.append("")
    L.append(
        "후속 스크립트 `scripts/generate_step4r_attention_schema_outputs.py`는 본 단계가 "
        "산출한 `data/step4r/4rb_attention/step4r_bigru_attention_logits_calibrated.npz`의 "
        "**`probs_calibrated`** 를 입력으로 사용한다. raw posterior(`probs_raw`)는 비교용으로만 "
        "유지하며 schema 결정에 사용되지 않는다. Step 3 §3의 class_posterior 출력에도 "
        "calibrated posterior가 들어간다."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 7. 산출물 목록")
    L.append("")
    L.append("- `data/step4r/4rb_attention/step4r_bigru_attention_logits_calibrated.npz`")
    L.append("  - 키: `logits_raw`, `logits_calibrated`, `probs_raw`, `probs_calibrated`, `y`, `class_id`, `sample_id`, `participant_id`, `posture_canonical`, `split`, `temperature`")
    L.append("- `data/step4r/4rb_attention/step4r_bigru_attention_predictions_calibrated.csv`")
    L.append("- `reports/step4r/4rb_attention/step4r_temperature_scaling_metrics.csv` (long format: split, stage, metric, value)")
    L.append("- `reports/step4r/4rb_attention/step4r_temperature_scaling_results.md` (본 보고서)")
    L.append("")
    L.append("---")
    L.append("")
    L.append("*본 보고서는 자동 생성된다. 기존 Step 1 ~ 4 / 4R-A / 4R-B 산출물은 수정되지 않으며, 기존 `step4r_bigru_attention_predictions.csv`는 덮어쓰지 않는다.*")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 64)
    print("Step 4R-B post 1/3 — Temperature scaling")
    print("=" * 64)
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_NPZ.exists():
        raise FileNotFoundError(f"Input npz not found: {INPUT_NPZ}")
    if not INPUT_CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {INPUT_CKPT}")

    data = _load_dataset()
    posture_oh = _posture_onehot(data["posture_canonical"])
    print(f"  X shape: {data['X_norm'].shape}, y shape: {data['y'].shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    print(f"loading checkpoint -> {INPUT_CKPT}")
    ckpt = torch.load(INPUT_CKPT, map_location=device)
    config = ckpt["config"]
    model = Step4RBiGRUAttention(**config).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"  best epoch = {ckpt['epoch']}, val_macro_f1 = {ckpt['val_macro_f1']:.4f}")

    print("recomputing logits + attention on all splits ...")
    t0 = time.time()
    logits_raw, attn_w = _infer_logits_attention(
        model, data["X_norm"], posture_oh, data["y"], device
    )
    print(f"  done in {time.time() - t0:.1f}s. logits shape={logits_raw.shape}")

    # Fit temperature on val only.
    print("fitting temperature on val split ...")
    val_mask = data["split"] == "val"
    val_logits = logits_raw[val_mask]
    val_y = data["y"][val_mask]
    T, fit_history = _fit_temperature(val_logits, val_y, device)

    logits_cal = logits_raw / T
    probs_raw = _softmax_np(logits_raw)
    probs_cal = _softmax_np(logits_cal)

    # Per-split metrics before/after.
    metrics_before, metrics_after = {}, {}
    for s in SPLITS:
        m = data["split"] == s
        if not m.any():
            continue
        metrics_before[s] = _split_metrics(probs_raw[m], data["y"][m])
        metrics_after[s] = _split_metrics(probs_cal[m], data["y"][m])

    print()
    print(f"{'split':<6} {'stage':<6} {'acc':>7} {'macroF1':>9} {'logloss':>9} {'Brier':>7} {'ECE':>7}")
    for s in SPLITS:
        b, a = metrics_before[s], metrics_after[s]
        print(
            f"{s:<6} {'before':<6} {b['accuracy']:>7.4f} {b['macro_f1']:>9.4f} "
            f"{b['log_loss']:>9.4f} {b['brier_multiclass']:>7.4f} {b['ece_15bin']:>7.4f}"
        )
        print(
            f"{s:<6} {'after':<6} {a['accuracy']:>7.4f} {a['macro_f1']:>9.4f} "
            f"{a['log_loss']:>9.4f} {a['brier_multiclass']:>7.4f} {a['ece_15bin']:>7.4f}"
        )

    # Save npz.
    np.savez(
        OUTPUT_NPZ,
        logits_raw=logits_raw.astype(np.float32),
        logits_calibrated=logits_cal.astype(np.float32),
        probs_raw=probs_raw.astype(np.float32),
        probs_calibrated=probs_cal.astype(np.float32),
        y=data["y"],
        class_id=data["class_id"],
        sample_id=data["sample_id"],
        participant_id=data["participant_id"],
        posture_canonical=data["posture_canonical"],
        split=data["split"],
        temperature=np.array([T], dtype=np.float64),
    )
    print(f"saved npz -> {OUTPUT_NPZ}")

    # Save predictions CSV (calibrated, per-sample).
    pred_idx = probs_cal.argmax(axis=1)
    sorted_p = np.sort(probs_cal, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    pe = _entropy(probs_cal)
    pred_df = pd.DataFrame({
        "sample_id": data["sample_id"],
        "participant_id": data["participant_id"],
        "class_id": data["class_id"],
        "posture_canonical": data["posture_canonical"],
        "split": data["split"],
        "pred_argmax_calibrated": [CLASSES[i] for i in pred_idx],
        "top1_prob_calibrated": probs_cal.max(axis=1),
        "top1_top2_margin_calibrated": margin,
        "predictive_entropy_calibrated": pe,
        **{f"p_{c}_raw": probs_raw[:, i] for i, c in enumerate(CLASSES)},
        **{f"p_{c}_calibrated": probs_cal[:, i] for i, c in enumerate(CLASSES)},
        "temperature": [T] * len(data["y"]),
    })
    pred_df.to_csv(OUTPUT_PREDS_CSV, index=False, encoding="utf-8-sig")
    print(f"saved calibrated predictions -> {OUTPUT_PREDS_CSV}")

    # Save metrics CSV (long format).
    rows = []
    for s in SPLITS:
        for stage, mm in [("before", metrics_before[s]), ("after", metrics_after[s])]:
            for k, v in mm.items():
                if k == "n":
                    continue
                rows.append({"split": s, "stage": stage, "metric": k, "value": float(v)})
            rows.append({"split": s, "stage": stage, "metric": "n", "value": float(mm["n"])})
    rows.append({"split": "—", "stage": "—", "metric": "fitted_temperature", "value": float(T)})
    rows.append({"split": "—", "stage": "—", "metric": "lbfgs_n_closures", "value": float(len(fit_history["iters"]))})
    rows.append({"split": "—", "stage": "—", "metric": "final_val_nll", "value": float(fit_history["nll"][-1])})
    pd.DataFrame(rows).to_csv(OUTPUT_METRICS_CSV, index=False, encoding="utf-8-sig")
    print(f"saved metrics -> {OUTPUT_METRICS_CSV}")

    # Markdown report.
    md = _markdown_report(T, metrics_before, metrics_after, fit_history, str(device))
    OUTPUT_REPORT_MD.write_text(md, encoding="utf-8")
    print(f"saved report -> {OUTPUT_REPORT_MD}")

    print()
    print("=" * 64)
    print(f"Done. T = {T:.6f}")
    print("=" * 64)
    return 0


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    sys.exit(main())
