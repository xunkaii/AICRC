"""Step 4R-B v2 — Per-seed temperature scaling.

Per-seed analogue of `scripts/calibrate_step4r_attention_temperature.py`.

Reads (read-only):
    data/step4r/4rb_attention/step4r_sequence_dataset.npz
    checkpoints/step4r/4rb_attention/experiments/{exp_id}/seed{N}/best.pt

Writes:
    data/step4r/4rb_attention/experiments/{exp_id}/seed{N}/
        logits_calibrated.npz
        predictions_calibrated.csv
    reports/step4r/4rb_attention/experiments/{exp_id}/seed{N}/
        temperature_scaling_metrics.csv
        temperature_scaling_results.md

CLI:
    --seed     int (required)
    --exp-id   str (required)

Run:
    & python.exe -X utf8 scripts/calibrate_step4r_attention_temperature_v2.py \
        --seed 42 --exp-id b00_baseline_3seed
"""
from __future__ import annotations

import argparse
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
from models.step4r_attention_rnn_aux import Step4RBiGRUAttentionAux  # noqa: E402

# Map model_class string (from checkpoint) -> class object.
MODEL_CLASS_REGISTRY = {
    "Step4RBiGRUAttention": Step4RBiGRUAttention,
    "Step4RBiGRUAttentionAux": Step4RBiGRUAttentionAux,
}

INPUT_NPZ = PROJECT_ROOT / "data" / "step4r" / "4rb_attention" / "step4r_sequence_dataset.npz"

CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
NUM_CLASSES = len(CLASSES)
SPLITS = ["train", "val", "test"]
POSTURES = ["SA", "CA", "HW"]
POSTURE_TO_IDX = {p: i for i, p in enumerate(POSTURES)}
BATCH_SIZE = 256


def _posture_onehot(arr: np.ndarray) -> np.ndarray:
    out = np.zeros((len(arr), len(POSTURES)), dtype=np.float32)
    for i, p in enumerate(arr):
        out[i, POSTURE_TO_IDX[p]] = 1.0
    return out


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


def _softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


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
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0)),
        "log_loss": float(log_loss(y_true, probs, labels=CLASSES)),
        "brier_multiclass": _multiclass_brier(probs, y_idx),
        "ece_15bin": _ece(probs, correct, n_bins=15),
        "top1_prob_mean": float(top1.mean()),
        "top1_prob_p25": float(np.percentile(top1, 25)),
        "top1_prob_p50": float(np.percentile(top1, 50)),
        "top1_prob_p75": float(np.percentile(top1, 75)),
        "top1_top2_margin_mean": float(margin.mean()),
        "predictive_entropy_mean": float(pe.mean()),
    }


def _infer_logits(model: nn.Module, X: np.ndarray, posture_oh: np.ndarray,
                  y: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
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


def _fit_temperature(val_logits: np.ndarray, val_y: np.ndarray, device: torch.device):
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
    return float(log_T.exp().item()), history


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--exp-id", type=str, required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    print("=" * 64)
    print(f"Step 4R-B v2 calibrate | exp_id={args.exp_id} | seed={args.seed}")
    print("=" * 64)

    rel = Path("experiments") / args.exp_id / f"seed{args.seed}"
    data_dir = PROJECT_ROOT / "data" / "step4r" / "4rb_attention" / rel
    report_dir = PROJECT_ROOT / "reports" / "step4r" / "4rb_attention" / rel
    ckpt = PROJECT_ROOT / "checkpoints" / "step4r" / "4rb_attention" / rel / "best.pt"
    for d in (data_dir, report_dir):
        d.mkdir(parents=True, exist_ok=True)
    out_npz = data_dir / "logits_calibrated.npz"
    out_preds_csv = data_dir / "predictions_calibrated.csv"
    out_metrics_csv = report_dir / "temperature_scaling_metrics.csv"
    out_report_md = report_dir / "temperature_scaling_results.md"

    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")

    z = np.load(INPUT_NPZ, allow_pickle=True)
    data = {
        "X_norm": z["X_norm"].astype(np.float32),
        "y": z["y"].astype(np.int64),
        "class_id": z["class_id"].astype(str),
        "posture_canonical": z["posture_canonical"].astype(str),
        "sample_id": z["sample_id"].astype(str),
        "participant_id": z["participant_id"].astype(str),
        "split": z["split"].astype(str),
    }
    posture_oh = _posture_onehot(data["posture_canonical"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"loading checkpoint -> {ckpt}")
    chk = torch.load(ckpt, map_location=device)
    model_class_name = chk.get("model_class", "Step4RBiGRUAttention")
    model_cls = MODEL_CLASS_REGISTRY.get(model_class_name)
    if model_cls is None:
        raise ValueError(f"unknown model_class in checkpoint: {model_class_name!r}")
    model = model_cls(**chk["config"]).to(device)
    model.load_state_dict(chk["state_dict"])
    print(
        f"  model_class={model_class_name} best epoch={chk['epoch']} "
        f"val_macroF1={chk['val_macro_f1']:.4f}"
    )

    print("recomputing logits ...")
    t0 = time.time()
    logits_raw, attn_w = _infer_logits(
        model, data["X_norm"], posture_oh, data["y"], device,
    )
    print(f"  done in {time.time() - t0:.1f}s. logits shape={logits_raw.shape}")

    val_mask = data["split"] == "val"
    T, hist = _fit_temperature(logits_raw[val_mask], data["y"][val_mask], device)
    print(f"fitted T = {T:.6f} (LBFGS closures={len(hist['iters'])}, final NLL={hist['nll'][-1]:.6f})")

    logits_cal = logits_raw / T
    probs_raw = _softmax_np(logits_raw)
    probs_cal = _softmax_np(logits_cal)

    before: dict = {}
    after: dict = {}
    for s in SPLITS:
        m = data["split"] == s
        if not m.any():
            continue
        before[s] = _split_metrics(probs_raw[m], data["y"][m])
        after[s] = _split_metrics(probs_cal[m], data["y"][m])
        b, a = before[s], after[s]
        print(
            f"  {s:5s} before  acc={b['accuracy']:.4f}  macroF1={b['macro_f1']:.4f}  "
            f"logloss={b['log_loss']:.4f}  ECE={b['ece_15bin']:.4f}"
        )
        print(
            f"  {s:5s} after   acc={a['accuracy']:.4f}  macroF1={a['macro_f1']:.4f}  "
            f"logloss={a['log_loss']:.4f}  ECE={a['ece_15bin']:.4f}"
        )

    np.savez(
        out_npz,
        logits_raw=logits_raw.astype(np.float32),
        logits_calibrated=logits_cal.astype(np.float32),
        probs_raw=probs_raw.astype(np.float32),
        probs_calibrated=probs_cal.astype(np.float32),
        attention_weights=attn_w.astype(np.float32),
        y=data["y"],
        class_id=data["class_id"],
        sample_id=data["sample_id"],
        participant_id=data["participant_id"],
        posture_canonical=data["posture_canonical"],
        split=data["split"],
        temperature=np.array([T], dtype=np.float64),
        seed=np.array([args.seed], dtype=np.int64),
    )
    print(f"saved npz -> {out_npz}")

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
    pred_df.to_csv(out_preds_csv, index=False, encoding="utf-8-sig")
    print(f"saved calibrated predictions -> {out_preds_csv}")

    rows = []
    for s in SPLITS:
        for stage, m in [("before", before.get(s, {})), ("after", after.get(s, {}))]:
            for k, v in m.items():
                if k == "n":
                    continue
                rows.append({"split": s, "stage": stage, "metric": k, "value": float(v)})
            if m:
                rows.append({"split": s, "stage": stage, "metric": "n", "value": float(m["n"])})
    rows.append({"split": "—", "stage": "—", "metric": "fitted_temperature", "value": float(T)})
    rows.append({"split": "—", "stage": "—", "metric": "lbfgs_n_closures", "value": float(len(hist["iters"]))})
    rows.append({"split": "—", "stage": "—", "metric": "final_val_nll", "value": float(hist["nll"][-1])})
    pd.DataFrame(rows).to_csv(out_metrics_csv, index=False, encoding="utf-8-sig")
    print(f"saved metrics -> {out_metrics_csv}")

    # Minimal md
    L = []
    L.append(f"# Step 4R-B v2 calibrate | exp=`{args.exp_id}` seed=`{args.seed}`")
    L.append("")
    L.append(f"- fitted T = {T:.6f}")
    L.append("")
    L.append("| split | stage | acc | macroF1 | logloss | Brier | ECE |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for s in SPLITS:
        b, a = before[s], after[s]
        L.append(f"| {s} | before | {b['accuracy']:.4f} | {b['macro_f1']:.4f} | {b['log_loss']:.4f} | {b['brier_multiclass']:.4f} | {b['ece_15bin']:.4f} |")
        L.append(f"| {s} | after | {a['accuracy']:.4f} | {a['macro_f1']:.4f} | {a['log_loss']:.4f} | {a['brier_multiclass']:.4f} | {a['ece_15bin']:.4f} |")
    out_report_md.write_text("\n".join(L), encoding="utf-8")
    print(f"saved report -> {out_report_md}")

    print()
    print("=" * 64)
    print(f"Done. seed={args.seed} T={T:.6f}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
