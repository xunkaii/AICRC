"""Step 4R-B — Train BiGRU + Attention raw IMU sensor-to-schema model.

Reads (read-only):
    data/step4r/4rb_attention/step4r_sequence_dataset.npz   (X_norm, y, ...)
    reports/step4r/4ra_feature_ceiling/step4r_feature_ceiling_metrics.csv
        (4R-A HGB ceiling, used in §3 of the markdown report only)

Writes:
    checkpoints/step4r/4rb_attention/best_bigru_attention.pt
    data/step4r/4rb_attention/step4r_bigru_attention_predictions.csv
    reports/step4r/4rb_attention/step4r_bigru_attention_metrics.csv
    reports/step4r/4rb_attention/step4r_bigru_attention_confusion.csv
    reports/step4r/4rb_attention/step4r_bigru_attention_training_log.csv
    reports/step4r/4rb_attention/step4r_bigru_attention_results.md

Behavior:
    - Existing files are not modified.
    - Existing manifest split (v1_36_8_8) is reused; no new split is made.
    - Model: models/step4r_attention_rnn.Step4RBiGRUAttention
    - Optimizer: AdamW(lr=1e-3, weight_decay=1e-4)
    - Loss: CrossEntropyLoss
    - batch_size = 64, max_epochs = 100, early stopping on val macro F1
      with patience = 15.

Run:
    python scripts/train_step4r_attention_rnn.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.step4r_attention_rnn import Step4RBiGRUAttention  # noqa: E402


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

INPUT_NPZ = PROJECT_ROOT / "data" / "step4r" / "4rb_attention" / "step4r_sequence_dataset.npz"
HGB_METRICS_CSV = (
    PROJECT_ROOT / "reports" / "step4r" / "4ra_feature_ceiling"
    / "step4r_feature_ceiling_metrics.csv"
)

OUTPUT_DATA_DIR = PROJECT_ROOT / "data" / "step4r" / "4rb_attention"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "reports" / "step4r" / "4rb_attention"
OUTPUT_CKPT_DIR = PROJECT_ROOT / "checkpoints" / "step4r" / "4rb_attention"

OUTPUT_PREDS = OUTPUT_DATA_DIR / "step4r_bigru_attention_predictions.csv"
OUTPUT_METRICS_CSV = OUTPUT_REPORT_DIR / "step4r_bigru_attention_metrics.csv"
OUTPUT_CONFUSION_CSV = OUTPUT_REPORT_DIR / "step4r_bigru_attention_confusion.csv"
OUTPUT_TRAIN_LOG_CSV = OUTPUT_REPORT_DIR / "step4r_bigru_attention_training_log.csv"
OUTPUT_REPORT_MD = OUTPUT_REPORT_DIR / "step4r_bigru_attention_results.md"
OUTPUT_BEST_CKPT = OUTPUT_CKPT_DIR / "best_bigru_attention.pt"

CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
NUM_CLASSES = len(CLASSES)
SPLITS = ["train", "val", "test"]
POSTURES = ["SA", "CA", "HW"]
POSTURE_TO_IDX = {p: i for i, p in enumerate(POSTURES)}

# Hyperparameters
SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_SIZE = 64
NUM_LAYERS = 2
RNN_DROPOUT = 0.3
HEAD_DROPOUT = 0.3

# LR baseline reference (from reports/step4/step4_final_summary.md, test split).
LR_BASELINE = {
    "raw": {
        "test_accuracy": 0.3210,
        "test_macro_f1": 0.2843,
        "test_log_loss": 1.6173,
        "test_brier": 0.7746,
        "test_ece_15bin": 0.0320,
        "test_c2_recall": 0.8159,
        "test_c1_c5_c6_internal": 0.2462,
        "test_c3_c4_pair": 0.2199,
        "test_c3_to_c2_absorb": 0.3277,
        "test_c4_to_c2_absorb": 0.2723,
    },
    "zscore": {
        "test_accuracy": 0.3203,
        "test_macro_f1": 0.2885,
        "test_log_loss": 1.6141,
        "test_brier": 0.7734,
        "test_ece_15bin": 0.0260,
        "test_c2_recall": 0.8159,
        "test_c1_c5_c6_internal": 0.2238,
        "test_c3_c4_pair": 0.2008,
        "test_c3_to_c2_absorb": 0.3361,
        "test_c4_to_c2_absorb": 0.2681,
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _posture_onehot(posture_str_arr: np.ndarray) -> np.ndarray:
    n = len(posture_str_arr)
    out = np.zeros((n, len(POSTURES)), dtype=np.float32)
    for i, p in enumerate(posture_str_arr):
        if p not in POSTURE_TO_IDX:
            raise ValueError(f"unexpected posture value: {p!r}")
        out[i, POSTURE_TO_IDX[p]] = 1.0
    return out


def _load_dataset() -> dict:
    print(f"loading {INPUT_NPZ}")
    # allow_pickle=True needed because string arrays (class_id, sample_id, ...)
    # were saved as numpy object dtype by the build script.
    z = np.load(INPUT_NPZ, allow_pickle=True)
    X_norm = z["X_norm"].astype(np.float32)        # (N, 6, 128)
    y = z["y"].astype(np.int64)                    # (N,)
    class_id = z["class_id"].astype(str)           # (N,)
    posture_canonical = z["posture_canonical"].astype(str)
    sample_id = z["sample_id"].astype(str)
    participant_id = z["participant_id"].astype(str)
    split = z["split"].astype(str)
    print(
        f"  X_norm shape: {X_norm.shape}, y shape: {y.shape}, "
        f"splits: {dict((s, int((split == s).sum())) for s in SPLITS)}"
    )
    posture_oh = _posture_onehot(posture_canonical)
    return {
        "X_norm": X_norm,
        "y": y,
        "class_id": class_id,
        "posture_canonical": posture_canonical,
        "posture_oh": posture_oh,
        "sample_id": sample_id,
        "participant_id": participant_id,
        "split": split,
    }


def _make_loader(
    X: np.ndarray,
    posture: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    Xs = torch.from_numpy(X[indices]).float()
    Ps = torch.from_numpy(posture[indices]).float()
    ys = torch.from_numpy(y[indices]).long()
    ds = TensorDataset(Xs, Ps, ys)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_n = 0
    preds_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    for X_b, P_b, y_b in loader:
        X_b = X_b.to(device, non_blocking=True)
        P_b = P_b.to(device, non_blocking=True)
        y_b = y_b.to(device, non_blocking=True)
        if train_mode:
            optimizer.zero_grad()
            logits, _ = model(X_b, P_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits, _ = model(X_b, P_b)
                loss = criterion(logits, y_b)
        total_loss += float(loss.item()) * X_b.size(0)
        total_n += X_b.size(0)
        preds_all.append(logits.argmax(dim=-1).detach().cpu().numpy())
        targets_all.append(y_b.detach().cpu().numpy())
    avg_loss = total_loss / total_n
    return avg_loss, np.concatenate(preds_all), np.concatenate(targets_all)


def _train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> list[dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    history: list[dict] = []
    best_macro_f1 = -np.inf
    best_epoch = -1
    bad_epochs = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        train_loss, _, _ = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_pred, val_true = _run_epoch(model, val_loader, criterion, device, None)
        val_acc = accuracy_score(val_true, val_pred)
        val_bal = balanced_accuracy_score(val_true, val_pred)
        val_f1 = f1_score(
            val_true, val_pred, average="macro", labels=list(range(NUM_CLASSES)),
            zero_division=0,
        )
        elapsed = time.time() - t0
        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_balanced_accuracy": val_bal,
            "val_macro_f1": val_f1,
            "lr": LR,
            "time_seconds": elapsed,
            "is_best": False,
        }
        if val_f1 > best_macro_f1 + 1e-8:
            best_macro_f1 = float(val_f1)
            best_epoch = epoch
            bad_epochs = 0
            log["is_best"] = True
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_macro_f1": best_macro_f1,
                    "config": model.get_config(),
                },
                OUTPUT_BEST_CKPT,
            )
        else:
            bad_epochs += 1
        history.append(log)
        marker = "  *" if log["is_best"] else "   "
        print(
            f"  epoch {epoch:3d} {marker}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  val_macroF1={val_f1:.4f}  "
            f"({elapsed:.1f}s)"
        )
        if bad_epochs >= PATIENCE:
            print(
                f"  early stopping at epoch {epoch}; "
                f"best val_macro_f1={best_macro_f1:.4f} at epoch {best_epoch}."
            )
            break
    print(
        f"  training done. best epoch={best_epoch}, "
        f"best val_macro_f1={best_macro_f1:.4f}."
    )
    return history


# ---------------------------------------------------------------------------
# Inference and metrics
# ---------------------------------------------------------------------------

def _infer(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (probs, attn_w, y_true) all as numpy."""
    model.eval()
    probs_list: list[np.ndarray] = []
    attn_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    with torch.no_grad():
        for X_b, P_b, y_b in loader:
            X_b = X_b.to(device, non_blocking=True)
            P_b = P_b.to(device, non_blocking=True)
            logits, attn_w = model(X_b, P_b)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            attn_list.append(attn_w.detach().cpu().numpy())
            probs_list.append(probs)
            y_list.append(y_b.numpy())
    return (
        np.concatenate(probs_list, axis=0),
        np.concatenate(attn_list, axis=0),
        np.concatenate(y_list, axis=0),
    )


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


def _entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


def _ambiguity(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out: dict = {}
    g135 = ["C1", "C5", "C6"]
    in_g = np.isin(y_true, g135)
    if in_g.any():
        pig = np.isin(y_pred[in_g], g135)
        out["c1_c5_c6_internal"] = float(
            ((y_pred[in_g] != y_true[in_g]) & pig).sum() / in_g.sum()
        )
    else:
        out["c1_c5_c6_internal"] = float("nan")
    g34 = ["C3", "C4"]
    in_p = np.isin(y_true, g34)
    if in_p.any():
        pip = np.isin(y_pred[in_p], g34)
        out["c3_c4_pair"] = float(
            ((y_pred[in_p] != y_true[in_p]) & pip).sum() / in_p.sum()
        )
    else:
        out["c3_c4_pair"] = float("nan")
    is_c3 = y_true == "C3"
    out["c3_to_c2_absorb"] = (
        float((y_pred[is_c3] == "C2").sum() / is_c3.sum()) if is_c3.any() else float("nan")
    )
    is_c4 = y_true == "C4"
    out["c4_to_c2_absorb"] = (
        float((y_pred[is_c4] == "C2").sum() / is_c4.sum()) if is_c4.any() else float("nan")
    )
    is_c2 = y_true == "C2"
    out["c2_recall"] = (
        float((y_pred[is_c2] == "C2").sum() / is_c2.sum()) if is_c2.any() else float("nan")
    )
    return out


def _evaluate_split(
    probs: np.ndarray,
    attn_w: np.ndarray,
    y_idx: np.ndarray,
) -> dict:
    y_true = np.array([CLASSES[i] for i in y_idx])
    pred_idx = probs.argmax(axis=1)
    y_pred = np.array([CLASSES[i] for i in pred_idx])
    correct = (pred_idx == y_idx).astype(float)
    top1 = probs.max(axis=1)
    sorted_p = np.sort(probs, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    pred_ent = _entropy(probs)
    attn_ent = _entropy(attn_w)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0
    )
    per_class = {}
    for i, c in enumerate(CLASSES):
        per_class[c] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    return {
        "n": int(len(y_idx)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", labels=CLASSES, zero_division=0)
        ),
        "log_loss": float(log_loss(y_true, probs, labels=CLASSES)),
        "brier_multiclass": _multiclass_brier(probs, y_idx, NUM_CLASSES),
        "ece_15bin": _ece(probs, correct, n_bins=15),
        "top1_prob_mean": float(top1.mean()),
        "top1_prob_p05": float(np.percentile(top1, 5)),
        "top1_prob_p25": float(np.percentile(top1, 25)),
        "top1_prob_p50": float(np.percentile(top1, 50)),
        "top1_prob_p75": float(np.percentile(top1, 75)),
        "top1_prob_p95": float(np.percentile(top1, 95)),
        "top1_top2_margin_mean": float(margin.mean()),
        "top1_top2_margin_p05": float(np.percentile(margin, 5)),
        "top1_top2_margin_p25": float(np.percentile(margin, 25)),
        "top1_top2_margin_p50": float(np.percentile(margin, 50)),
        "top1_top2_margin_p75": float(np.percentile(margin, 75)),
        "top1_top2_margin_p95": float(np.percentile(margin, 95)),
        "predictive_entropy_mean": float(pred_ent.mean()),
        "predictive_entropy_p05": float(np.percentile(pred_ent, 5)),
        "predictive_entropy_p25": float(np.percentile(pred_ent, 25)),
        "predictive_entropy_p50": float(np.percentile(pred_ent, 50)),
        "predictive_entropy_p75": float(np.percentile(pred_ent, 75)),
        "predictive_entropy_p95": float(np.percentile(pred_ent, 95)),
        "attention_entropy_mean": float(attn_ent.mean()),
        "attention_entropy_p05": float(np.percentile(attn_ent, 5)),
        "attention_entropy_p25": float(np.percentile(attn_ent, 25)),
        "attention_entropy_p50": float(np.percentile(attn_ent, 50)),
        "attention_entropy_p75": float(np.percentile(attn_ent, 75)),
        "attention_entropy_p95": float(np.percentile(attn_ent, 95)),
        "per_class": per_class,
        **{f"amb_{k}": v for k, v in _ambiguity(y_true, y_pred).items()},
    }


def _flatten_metrics(metrics_by_split: dict) -> pd.DataFrame:
    rows = []
    flat_keys = [
        "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
        "log_loss", "brier_multiclass", "ece_15bin",
        "top1_prob_mean", "top1_prob_p05", "top1_prob_p25",
        "top1_prob_p50", "top1_prob_p75", "top1_prob_p95",
        "top1_top2_margin_mean", "top1_top2_margin_p05",
        "top1_top2_margin_p25", "top1_top2_margin_p50",
        "top1_top2_margin_p75", "top1_top2_margin_p95",
        "predictive_entropy_mean", "predictive_entropy_p05",
        "predictive_entropy_p25", "predictive_entropy_p50",
        "predictive_entropy_p75", "predictive_entropy_p95",
        "attention_entropy_mean", "attention_entropy_p05",
        "attention_entropy_p25", "attention_entropy_p50",
        "attention_entropy_p75", "attention_entropy_p95",
        "amb_c2_recall", "amb_c1_c5_c6_internal", "amb_c3_c4_pair",
        "amb_c3_to_c2_absorb", "amb_c4_to_c2_absorb",
    ]
    for s, m in metrics_by_split.items():
        base = {"split": s, "n": m["n"]}
        for k in flat_keys:
            rows.append({**base, "metric": k, "value": m[k]})
        for c in CLASSES:
            pc = m["per_class"][c]
            rows.append({**base, "metric": f"precision_{c}", "value": pc["precision"]})
            rows.append({**base, "metric": f"recall_{c}", "value": pc["recall"]})
            rows.append({**base, "metric": f"f1_{c}", "value": pc["f1"]})
            rows.append({**base, "metric": f"support_{c}", "value": pc["support"]})
    return pd.DataFrame(rows)


def _confusion_long(metrics_by_split: dict, ytrue_pred_by_split: dict) -> pd.DataFrame:
    rows = []
    for s, (yt, yp) in ytrue_pred_by_split.items():
        cm = confusion_matrix(yt, yp, labels=CLASSES)
        for i, t in enumerate(CLASSES):
            for j, p in enumerate(CLASSES):
                rows.append(
                    {"split": s, "true_class": t, "pred_class": p, "count": int(cm[i, j])}
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4R-A HGB ceiling lookup
# ---------------------------------------------------------------------------

def _load_hgb_test_metrics() -> dict:
    """Return dict[condition][metric] = value, restricted to test split, for §3."""
    if not HGB_METRICS_CSV.exists():
        return {}
    df = pd.read_csv(HGB_METRICS_CSV)
    out: dict = {}
    df = df.loc[df["split"] == "test"]
    for cond in ["raw", "zscore"]:
        sub = df.loc[df["condition"] == cond]
        if sub.empty:
            continue
        d = {}
        for k in [
            "accuracy", "macro_f1", "weighted_f1", "log_loss",
            "brier_multiclass", "ece_15bin",
            "amb_c2_recall", "amb_c1_c5_c6_internal", "amb_c3_c4_pair",
            "amb_c3_to_c2_absorb", "amb_c4_to_c2_absorb",
        ]:
            r = sub.loc[sub["metric"] == k, "value"]
            if not r.empty:
                d[k] = float(r.iloc[0])
        out[cond] = d
    return out


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _markdown_report(
    metrics_by_split: dict,
    history: list[dict],
    hgb_test: dict,
    config: dict,
    device_str: str,
    best_epoch: int,
) -> str:
    L = []
    L.append("# Step 4R-B — BiGRU + Attention 학습 결과 보고서")
    L.append("")
    L.append("- 생성 스크립트: `scripts/train_step4r_attention_rnn.py`")
    L.append("- 모델 정의: `models/step4r_attention_rnn.py` (`Step4RBiGRUAttention`)")
    L.append("- 입력: `data/step4r/4rb_attention/step4r_sequence_dataset.npz` (X_norm 사용)")
    L.append(f"- 학습 device: `{device_str}`")
    L.append(f"- best checkpoint: `checkpoints/step4r/4rb_attention/best_bigru_attention.pt` (epoch {best_epoch})")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 1. 모델 설정")
    L.append("")
    L.append("| 항목 | 값 |")
    L.append("|---|---|")
    L.append(f"| input shape | (B, 6, 128) → 내부 (B, 128, 6) transpose |")
    L.append(f"| BiGRU hidden_size | {config['hidden_size']} |")
    L.append(f"| BiGRU num_layers | {config['num_layers']} |")
    L.append(f"| BiGRU bidirectional | True (output dim = {config['hidden_size'] * 2}) |")
    L.append(f"| BiGRU inter-layer dropout | {config['rnn_dropout']} |")
    L.append("| attention | Luong-style multiplicative, last-state query |")
    L.append("| attention vector dim | 128 (= hidden·2) |")
    L.append(f"| posture conditioning | one-hot 3-dim (SA/CA/HW), attention vector 뒤 concat |")
    L.append("| classifier head | Linear(131→128) → ReLU → Dropout → Linear(128→6) |")
    L.append(f"| classifier dropout | {config['head_dropout']} |")
    L.append(f"| optimizer | AdamW (lr={LR}, weight_decay={WEIGHT_DECAY}) |")
    L.append("| loss | CrossEntropyLoss |")
    L.append(f"| batch_size | {BATCH_SIZE} |")
    L.append(f"| max_epochs | {MAX_EPOCHS} |")
    L.append(f"| early stopping | val macro F1, patience={PATIENCE} |")
    L.append(f"| seed | {SEED} |")
    L.append("")
    L.append("forward는 `(logits, attention_weights)`를 반환하며, attention_weights shape은 (B, 128).")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 2. train / val / test 성능")
    L.append("")
    L.append("| split | n | accuracy | balanced_acc | macro F1 | weighted F1 | log loss | Brier | ECE (15-bin) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in SPLITS:
        m = metrics_by_split[s]
        L.append(
            f"| {s} | {m['n']} | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | "
            f"{m['macro_f1']:.4f} | {m['weighted_f1']:.4f} | {m['log_loss']:.4f} | "
            f"{m['brier_multiclass']:.4f} | {m['ece_15bin']:.4f} |"
        )
    L.append("")
    L.append("### 2.1 per-class precision / recall / F1 (test split)")
    L.append("")
    L.append("| class | precision | recall | F1 | support |")
    L.append("|---|---:|---:|---:|---:|")
    for c in CLASSES:
        pc = metrics_by_split["test"]["per_class"][c]
        L.append(
            f"| {c} | {pc['precision']:.4f} | {pc['recall']:.4f} | "
            f"{pc['f1']:.4f} | {pc['support']} |"
        )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 3. test 성능 비교 — LR baseline / 4R-A HGB ceiling / 4R-B")
    L.append("")
    L.append(
        "LR baseline 수치는 `reports/step4/step4_final_summary.md`에서 인용. "
        "HGB ceiling 수치는 `reports/step4r/4ra_feature_ceiling/step4r_feature_ceiling_metrics.csv`에서 자동 로드. "
        "둘 다 (raw, zscore) 두 branch 중 raw branch만 표시한다 (zscore와 거의 동일 — §3의 zscore 비교는 metrics CSV에서 직접 확인 가능)."
    )
    L.append("")
    h_b = metrics_by_split["test"]
    lr = LR_BASELINE["raw"]
    hgb = hgb_test.get("raw", {})
    L.append("| 지표 | LR raw | HGB raw (4R-A) | BiGRU+Attn (4R-B) |")
    L.append("|---|---:|---:|---:|")
    rows = [
        ("accuracy", lr.get("test_accuracy"), hgb.get("accuracy"), h_b["accuracy"]),
        ("macro F1", lr.get("test_macro_f1"), hgb.get("macro_f1"), h_b["macro_f1"]),
        ("log loss", lr.get("test_log_loss"), hgb.get("log_loss"), h_b["log_loss"]),
        ("Brier (multi)", lr.get("test_brier"), hgb.get("brier_multiclass"), h_b["brier_multiclass"]),
        ("ECE (15-bin)", lr.get("test_ece_15bin"), hgb.get("ece_15bin"), h_b["ece_15bin"]),
        ("C2 recall", lr.get("test_c2_recall"), hgb.get("amb_c2_recall"), h_b["amb_c2_recall"]),
        ("C1/C5/C6 internal", lr.get("test_c1_c5_c6_internal"), hgb.get("amb_c1_c5_c6_internal"), h_b["amb_c1_c5_c6_internal"]),
        ("C3/C4 pair", lr.get("test_c3_c4_pair"), hgb.get("amb_c3_c4_pair"), h_b["amb_c3_c4_pair"]),
        ("C3 → C2 absorb", lr.get("test_c3_to_c2_absorb"), hgb.get("amb_c3_to_c2_absorb"), h_b["amb_c3_to_c2_absorb"]),
        ("C4 → C2 absorb", lr.get("test_c4_to_c2_absorb"), hgb.get("amb_c4_to_c2_absorb"), h_b["amb_c4_to_c2_absorb"]),
    ]
    def _fmt(v: float | None) -> str:
        return "—" if v is None else f"{v:.4f}"
    for name, a, b, c in rows:
        L.append(f"| {name} | {_fmt(a)} | {_fmt(b)} | {_fmt(c)} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 4. Calibration")
    L.append("")
    L.append("| split | log loss | Brier | ECE (15-bin) | top1 mean | top1 p25 | top1 p50 | top1 p75 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in SPLITS:
        m = metrics_by_split[s]
        L.append(
            f"| {s} | {m['log_loss']:.4f} | {m['brier_multiclass']:.4f} | "
            f"{m['ece_15bin']:.4f} | {m['top1_prob_mean']:.4f} | "
            f"{m['top1_prob_p25']:.4f} | {m['top1_prob_p50']:.4f} | "
            f"{m['top1_prob_p75']:.4f} |"
        )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 5. Predictive entropy / Attention entropy 요약")
    L.append("")
    L.append("predictive entropy = −Σ p log p (자연로그). 6-class 균등분포 상한 ≈ 1.7918.")
    L.append("attention entropy = 시계열 attention weight 분포의 entropy. T=128 균등분포 상한 ≈ 4.8520.")
    L.append("")
    L.append("| split | pred_ent mean | pred_ent p25 | pred_ent p50 | pred_ent p75 | attn_ent mean | attn_ent p25 | attn_ent p50 | attn_ent p75 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in SPLITS:
        m = metrics_by_split[s]
        L.append(
            f"| {s} | {m['predictive_entropy_mean']:.4f} | "
            f"{m['predictive_entropy_p25']:.4f} | {m['predictive_entropy_p50']:.4f} | "
            f"{m['predictive_entropy_p75']:.4f} | {m['attention_entropy_mean']:.4f} | "
            f"{m['attention_entropy_p25']:.4f} | {m['attention_entropy_p50']:.4f} | "
            f"{m['attention_entropy_p75']:.4f} |"
        )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 6. Ambiguity group 분석 (test split)")
    L.append("")
    L.append("| 지표 | 값 |")
    L.append("|---|---:|")
    m = metrics_by_split["test"]
    L.append(f"| C2 recall | {m['amb_c2_recall']:.4f} |")
    L.append(f"| C1/C5/C6 internal confusion | {m['amb_c1_c5_c6_internal']:.4f} |")
    L.append(f"| C3/C4 pair confusion | {m['amb_c3_c4_pair']:.4f} |")
    L.append(f"| C3 → C2 absorption | {m['amb_c3_to_c2_absorb']:.4f} |")
    L.append(f"| C4 → C2 absorption | {m['amb_c4_to_c2_absorb']:.4f} |")
    L.append("")
    L.append(
        "Step 2.5 종합 (`reports/step2/step25_final_synthesis.md`)에서 도출된 "
        "ambiguity 패턴(C2 단언 가능, C1/C5/C6 그룹 모호, C3/C4의 C2 흡수)이 "
        "본 모델에서도 보존되는지를 점검한다."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 7. 과적합 여부")
    L.append("")
    train_m = metrics_by_split["train"]
    val_m = metrics_by_split["val"]
    test_m = metrics_by_split["test"]
    L.append("| 비교 | train | val | test | train−val gap | train−test gap |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for k in ["accuracy", "macro_f1", "ece_15bin", "log_loss"]:
        a = train_m[k]
        b = val_m[k]
        c = test_m[k]
        L.append(
            f"| {k} | {a:.4f} | {b:.4f} | {c:.4f} | "
            f"{a - b:+.4f} | {a - c:+.4f} |"
        )
    L.append("")
    L.append(
        "train−val / train−test gap이 크면 과적합. "
        "early stopping 기준이 val macro F1이므로 best epoch에서의 train−val gap이 "
        "본 학습이 멈춘 시점의 generalization 신호이다. checkpoint epoch / "
        "training history는 `step4r_bigru_attention_training_log.csv`에 저장된다."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 8. 다음 단계 제안")
    L.append("")
    L.append(
        "본 결과는 reframing 문서 §5.5 모델 채택 기준(A/B/C)의 어느 분기에 해당하는지 "
        "결정하는 입력이다. 일반적으로 다음 중 한 방향으로 이어진다:"
    )
    L.append("")
    L.append(
        "1. **case A** — 분류 또는 calibration이 4R-A 대비 명확히 개선됨 → "
        "4R-B를 main pipeline 모델로 승격하고, 다음 단계는 schema output threshold "
        "재calibration. Step 5~7 (schema-grounded LLM caption layer)는 그대로 유지."
    )
    L.append(
        "2. **case B** — 분류 점수는 4R-A와 비슷하지만 schema behavior 또는 근거 분석 "
        "(predictive/attention entropy + ambiguity group 일치)이 의미 있게 작동 → "
        "4R-B를 채택하되, 논문 contribution을 분류 성능이 아니라 *uncertainty-aware 설명* "
        "쪽으로 강조."
    )
    L.append(
        "3. **case C** — 분류·calibration·근거 분석 모두 4R-A와 동등하거나 약함 → "
        "4R-A(HGB)를 main pipeline 모델로 두고, 4R-B 결과는 *raw sequence DL의 "
        "데이터-규모 한계*로 ablation 챕터에 보고. 본 reframing 문서 §11에 따라 "
        "후속 의사결정 기록 필요."
    )
    L.append("")
    L.append(
        "본 결과는 단일 시드, 단일 hyperparameter 점만 산출했다. seed sensitivity나 "
        "augmentation ablation은 본 단계의 범위 밖이다."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 9. 산출물 목록")
    L.append("")
    L.append("- `data/step4r/4rb_attention/step4r_bigru_attention_predictions.csv`")
    L.append("- `reports/step4r/4rb_attention/step4r_bigru_attention_metrics.csv` (long format)")
    L.append("- `reports/step4r/4rb_attention/step4r_bigru_attention_confusion.csv` (long format)")
    L.append("- `reports/step4r/4rb_attention/step4r_bigru_attention_training_log.csv`")
    L.append("- `reports/step4r/4rb_attention/step4r_bigru_attention_results.md` (본 보고서)")
    L.append("- `checkpoints/step4r/4rb_attention/best_bigru_attention.pt`")
    L.append("")
    L.append("---")
    L.append("")
    L.append(
        "*본 보고서는 `scripts/train_step4r_attention_rnn.py` 실행 시 자동 생성된다. "
        "기존 Step 1 ~ 4 / 4R-A 산출물은 수정되지 않는다.*"
    )
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 64)
    print("Step 4R-B — Train BiGRU + Attention raw IMU sensor-to-schema")
    print("=" * 64)
    _set_seed(SEED)

    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_NPZ.exists():
        raise FileNotFoundError(
            f"Input npz not found: {INPUT_NPZ}. "
            "Run scripts/build_step4r_sequence_dataset.py first."
        )

    data = _load_dataset()
    X = data["X_norm"]
    y = data["y"]
    posture = data["posture_oh"]
    split = data["split"]

    train_idx = np.where(split == "train")[0]
    val_idx = np.where(split == "val")[0]
    test_idx = np.where(split == "test")[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} (cuda available={torch.cuda.is_available()})")

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = _make_loader(X, posture, y, train_idx, BATCH_SIZE, True, g)
    val_loader = _make_loader(X, posture, y, val_idx, BATCH_SIZE, False)
    test_loader = _make_loader(X, posture, y, test_idx, BATCH_SIZE, False)

    model = Step4RBiGRUAttention(
        num_classes=NUM_CLASSES,
        in_channels=6,
        seq_len=128,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        rnn_dropout=RNN_DROPOUT,
        head_dropout=HEAD_DROPOUT,
        posture_dim=3,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: Step4RBiGRUAttention, total params = {n_params}")

    print()
    print("training ...")
    history = _train(model, train_loader, val_loader, device)

    log_df = pd.DataFrame(history)
    log_df.to_csv(OUTPUT_TRAIN_LOG_CSV, index=False)
    print(f"saved training log -> {OUTPUT_TRAIN_LOG_CSV}")

    print()
    print(f"loading best checkpoint -> {OUTPUT_BEST_CKPT}")
    ckpt = torch.load(OUTPUT_BEST_CKPT, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    best_epoch = int(ckpt["epoch"])
    print(f"  best epoch = {best_epoch}, val_macro_f1 = {ckpt['val_macro_f1']:.4f}")

    # Inference on all splits, in original manifest order per split.
    metrics_by_split: dict = {}
    ytrue_pred_by_split: dict = {}
    pred_rows: list[dict] = []
    split_to_loader = {"train": (train_idx, train_loader), "val": (val_idx, val_loader), "test": (test_idx, test_loader)}
    for s, (idx, loader) in split_to_loader.items():
        # Note: train_loader was shuffled; re-create a deterministic loader for inference.
        det_loader = _make_loader(X, posture, y, idx, BATCH_SIZE, False)
        probs, attn_w, y_idx = _infer(model, det_loader, device)
        m = _evaluate_split(probs, attn_w, y_idx)
        metrics_by_split[s] = m
        # build per-sample prediction rows
        pred_idx = probs.argmax(axis=1)
        pred_class = np.array([CLASSES[i] for i in pred_idx])
        attn_ent = _entropy(attn_w)
        pred_ent = _entropy(probs)
        for k, row_idx in enumerate(idx):
            row = {
                "sample_id": data["sample_id"][row_idx],
                "participant_id": data["participant_id"][row_idx],
                "class_id": data["class_id"][row_idx],
                "posture_canonical": data["posture_canonical"][row_idx],
                "split": s,
            }
            for i, c in enumerate(CLASSES):
                row[f"p_{c}"] = float(probs[k, i])
            row["pred_argmax_debug"] = pred_class[k]
            row["predictive_entropy"] = float(pred_ent[k])
            row["attention_entropy"] = float(attn_ent[k])
            row["top1_prob"] = float(probs[k].max())
            sorted_p = np.sort(probs[k])
            row["top1_top2_margin"] = float(sorted_p[-1] - sorted_p[-2])
            pred_rows.append(row)
        ytrue_str = np.array([CLASSES[i] for i in y_idx])
        ytrue_pred_by_split[s] = (ytrue_str, pred_class)
        print(
            f"  {s:5s} n={m['n']:4d}  acc={m['accuracy']:.4f}  "
            f"macro_F1={m['macro_f1']:.4f}  log_loss={m['log_loss']:.4f}  "
            f"ECE={m['ece_15bin']:.4f}"
        )

    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(OUTPUT_PREDS, index=False)
    print(f"saved predictions -> {OUTPUT_PREDS}")

    metrics_df = _flatten_metrics(metrics_by_split)
    metrics_df.to_csv(OUTPUT_METRICS_CSV, index=False)
    print(f"saved metrics -> {OUTPUT_METRICS_CSV}")

    confusion_df = _confusion_long(metrics_by_split, ytrue_pred_by_split)
    confusion_df.to_csv(OUTPUT_CONFUSION_CSV, index=False)
    print(f"saved confusion -> {OUTPUT_CONFUSION_CSV}")

    hgb_test = _load_hgb_test_metrics()
    if not hgb_test:
        print("warn: 4R-A HGB metrics CSV not found; §3 comparison rows will show '—' for HGB.")

    md = _markdown_report(
        metrics_by_split,
        history,
        hgb_test,
        model.get_config(),
        device_str=str(device),
        best_epoch=best_epoch,
    )
    OUTPUT_REPORT_MD.write_text(md, encoding="utf-8")
    print(f"saved report -> {OUTPUT_REPORT_MD}")

    print()
    print("=" * 64)
    print("Done.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
