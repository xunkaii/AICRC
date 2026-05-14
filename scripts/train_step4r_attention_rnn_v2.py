"""Step 4R-B v2 — Parametrized BiGRU+Attention training (seed, exp_id, aug).

This is the multi-seed / experiment-tracked successor of
`scripts/train_step4r_attention_rnn.py` (single-seed baseline, NOT modified).

Reads (read-only):
    data/step4r/4rb_attention/step4r_sequence_dataset.npz
    models/step4r_attention_rnn.py  (Step4RBiGRUAttention class; imported)

Writes (NEW paths only; existing 4R-B baseline outputs are NOT touched):
    data/step4r/4rb_attention/experiments/{exp_id}/seed{N}/
        predictions.csv
    checkpoints/step4r/4rb_attention/experiments/{exp_id}/seed{N}/
        best.pt
    reports/step4r/4rb_attention/experiments/{exp_id}/seed{N}/
        metrics.csv  confusion.csv  training_log.csv  results.md

CLI:
    --seed       int (required)  — 42, 43, 44
    --exp-id     str (required)  — e.g., b00_baseline_3seed, b01_aug_jitter_scale
    --aug        choices = {off, jitter_scale, jitter_scale_warp}, default off
    --jitter-sigma  float, default 0.03
    --scale-low     float, default 0.9
    --scale-high    float, default 1.1
    --warp-low      float, default 0.9   (only used if aug includes warp)
    --warp-high     float, default 1.1

The model architecture, optimizer, batch size, max_epochs, patience, hidden_size
match the original baseline so that b00_baseline_3seed is a direct 3-seed
re-run of the original single-seed 4R-B.

Augmentations apply to train DataLoader only. val/test are always unaugmented.
- jitter_scale: per-sample channel-wise Gaussian noise (sigma) + per-channel
  uniform magnitude scaling (low, high).
- jitter_scale_warp: above + mild time-warping via linear interpolation along
  a randomly warped timesteps axis (warp factor in [warp-low, warp-high]).

rotation augmentation is NOT supported — it would destroy posture-dependent
orientation information (see Phase 0 decision in dialogue / reframing §5.2).

Run:
    & C:\\Users\\user\\anaconda3\\envs\\aicrc_env\\python.exe -X utf8 \
        scripts/train_step4r_attention_rnn_v2.py --seed 42 --exp-id b00_baseline_3seed --aug off
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
from torch.utils.data import DataLoader, Dataset
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
from models.step4r_attention_rnn_aux import (  # noqa: E402
    Step4RBiGRUAttentionAux,
    class_idx_to_group_idx,
)


# ---------------------------------------------------------------------------
# Constants (match baseline single-seed script)
# ---------------------------------------------------------------------------

INPUT_NPZ = PROJECT_ROOT / "data" / "step4r" / "4rb_attention" / "step4r_sequence_dataset.npz"

CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
NUM_CLASSES = len(CLASSES)
SPLITS = ["train", "val", "test"]
POSTURES = ["SA", "CA", "HW"]
POSTURE_TO_IDX = {p: i for i, p in enumerate(POSTURES)}

BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_SIZE = 64
NUM_LAYERS = 2
RNN_DROPOUT = 0.3
HEAD_DROPOUT = 0.3


# ---------------------------------------------------------------------------
# Output paths from CLI
# ---------------------------------------------------------------------------

def _output_paths(exp_id: str, seed: int) -> dict:
    rel = Path("experiments") / exp_id / f"seed{seed}"
    data_dir = PROJECT_ROOT / "data" / "step4r" / "4rb_attention" / rel
    report_dir = PROJECT_ROOT / "reports" / "step4r" / "4rb_attention" / rel
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "step4r" / "4rb_attention" / rel
    for d in (data_dir, report_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {
        "preds": data_dir / "predictions.csv",
        "metrics": report_dir / "metrics.csv",
        "confusion": report_dir / "confusion.csv",
        "training_log": report_dir / "training_log.csv",
        "report_md": report_dir / "results.md",
        "best_ckpt": ckpt_dir / "best.pt",
        "data_dir": data_dir,
        "report_dir": report_dir,
        "ckpt_dir": ckpt_dir,
    }


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class AugmentedSequenceDataset(Dataset):
    """Wraps numpy X(C,T), posture one-hot(P), y(int) and applies optional aug
    on __getitem__ (train-only). val/test should use mode='off'.
    """

    def __init__(
        self,
        X: np.ndarray,
        posture: np.ndarray,
        y: np.ndarray,
        mode: str = "off",
        jitter_sigma: float = 0.03,
        scale_low: float = 0.9,
        scale_high: float = 1.1,
        warp_low: float = 0.9,
        warp_high: float = 1.1,
        rng_seed: int = 0,
    ) -> None:
        assert mode in {"off", "jitter_scale", "jitter_scale_warp"}, mode
        self.X = X.astype(np.float32)
        self.posture = posture.astype(np.float32)
        self.y = y.astype(np.int64)
        self.mode = mode
        self.jitter_sigma = float(jitter_sigma)
        self.scale_low = float(scale_low)
        self.scale_high = float(scale_high)
        self.warp_low = float(warp_low)
        self.warp_high = float(warp_high)
        # Per-process numpy RNG, seeded for reproducibility across runs.
        self._rng = np.random.default_rng(int(rng_seed))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]  # (C, T)
        p = self.posture[idx]
        y = self.y[idx]
        if self.mode != "off":
            x = self._apply(x)
        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(p),
            torch.tensor(y, dtype=torch.long),
        )

    def _apply(self, x: np.ndarray) -> np.ndarray:
        C, T = x.shape
        # jitter (sample-level Gaussian noise added to all channels/timesteps)
        x = x + self._rng.normal(0.0, self.jitter_sigma, size=x.shape).astype(np.float32)
        # per-channel magnitude scaling
        scale = self._rng.uniform(self.scale_low, self.scale_high, size=(C, 1)).astype(np.float32)
        x = x * scale
        if self.mode == "jitter_scale_warp":
            x = self._time_warp(x)
        return x

    def _time_warp(self, x: np.ndarray) -> np.ndarray:
        """Mild time warp: stretch/compress by uniform factor in [warp_low, warp_high]
        then resample back to T via linear interpolation.
        """
        C, T = x.shape
        factor = float(self._rng.uniform(self.warp_low, self.warp_high))
        new_T = max(2, int(round(T * factor)))
        # original timesteps in [0, T-1]; new uniform in [0, T-1]
        src = np.linspace(0.0, T - 1, num=new_T, dtype=np.float32)
        # interp each channel
        warped = np.empty((C, new_T), dtype=np.float32)
        idx_full = np.arange(T, dtype=np.float32)
        for c in range(C):
            warped[c] = np.interp(src, idx_full, x[c])
        # resample warped back to T
        dst = np.linspace(0.0, new_T - 1, num=T, dtype=np.float32)
        out = np.empty((C, T), dtype=np.float32)
        warped_idx = np.arange(new_T, dtype=np.float32)
        for c in range(C):
            out[c] = np.interp(dst, warped_idx, warped[c])
        return out


# ---------------------------------------------------------------------------
# Helpers
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
    z = np.load(INPUT_NPZ, allow_pickle=True)
    return {
        "X_norm": z["X_norm"].astype(np.float32),
        "y": z["y"].astype(np.int64),
        "class_id": z["class_id"].astype(str),
        "posture_canonical": z["posture_canonical"].astype(str),
        "posture_oh": _posture_onehot(z["posture_canonical"].astype(str)),
        "sample_id": z["sample_id"].astype(str),
        "participant_id": z["participant_id"].astype(str),
        "split": z["split"].astype(str),
    }


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
        ece += (mask.sum() / n) * abs(
            float(top1[mask].mean()) - float(correct[mask].mean())
        )
    return float(ece)


def _multiclass_brier(probs: np.ndarray, y_idx: np.ndarray) -> float:
    y_oh = np.zeros((len(y_idx), NUM_CLASSES), dtype=float)
    y_oh[np.arange(len(y_idx)), y_idx] = 1.0
    return float(((probs - y_oh) ** 2).sum(axis=1).mean())


def _ambiguity(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out: dict = {}
    g135 = ["C1", "C5", "C6"]
    in_g = np.isin(y_true, g135)
    out["c1_c5_c6_internal"] = (
        float(((y_pred[in_g] != y_true[in_g]) & np.isin(y_pred[in_g], g135)).sum() / in_g.sum())
        if in_g.any() else float("nan")
    )
    g34 = ["C3", "C4"]
    in_p = np.isin(y_true, g34)
    out["c3_c4_pair"] = (
        float(((y_pred[in_p] != y_true[in_p]) & np.isin(y_pred[in_p], g34)).sum() / in_p.sum())
        if in_p.any() else float("nan")
    )
    for k, c in [("c3_to_c2_absorb", "C3"), ("c4_to_c2_absorb", "C4"), ("c2_recall", "C2")]:
        m = y_true == c
        if k.endswith("_recall"):
            out[k] = float((y_pred[m] == c).sum() / m.sum()) if m.any() else float("nan")
        else:
            out[k] = float((y_pred[m] == "C2").sum() / m.sum()) if m.any() else float("nan")
    return out


def _evaluate(probs: np.ndarray, attn_w: np.ndarray, y_idx: np.ndarray) -> dict:
    y_true = np.array([CLASSES[i] for i in y_idx])
    pred_idx = probs.argmax(axis=1)
    y_pred = np.array([CLASSES[i] for i in pred_idx])
    correct = (pred_idx == y_idx).astype(float)
    top1 = probs.max(axis=1)
    sorted_p = np.sort(probs, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    pe = _entropy(probs)
    ae = _entropy(attn_w)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0
    )
    per_class = {c: {"precision": float(prec[i]), "recall": float(rec[i]),
                     "f1": float(f1[i]), "support": int(support[i])}
                 for i, c in enumerate(CLASSES)}
    base = {
        "n": int(len(y_idx)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=CLASSES, zero_division=0)),
        "log_loss": float(log_loss(y_true, probs, labels=CLASSES)),
        "brier_multiclass": _multiclass_brier(probs, y_idx),
        "ece_15bin": _ece(probs, correct, n_bins=15),
        "top1_prob_mean": float(top1.mean()),
        "top1_top2_margin_mean": float(margin.mean()),
        "predictive_entropy_mean": float(pe.mean()),
        "attention_entropy_mean": float(ae.mean()),
        "per_class": per_class,
    }
    for k, v in _ambiguity(y_true, y_pred).items():
        base[f"amb_{k}"] = v
    return base


# ---------------------------------------------------------------------------
# Train / infer
# ---------------------------------------------------------------------------

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    *,
    aux_head: bool = False,
    aux_weight: float = 0.0,
    criterion_aux: nn.Module | None = None,
) -> tuple[float, float, float]:
    """Returns (avg_total_loss, avg_main_loss, avg_aux_loss).

    When `aux_head=False`, avg_main_loss == avg_total_loss and avg_aux_loss == 0.
    """
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_main = 0.0
    total_aux = 0.0
    total_n = 0
    for X_b, P_b, y_b in loader:
        X_b = X_b.to(device, non_blocking=True)
        P_b = P_b.to(device, non_blocking=True)
        y_b = y_b.to(device, non_blocking=True)
        if train_mode:
            optimizer.zero_grad()
            if aux_head:
                logits, _, logits_aux = model.forward_with_aux(X_b, P_b)
                loss_main = criterion(logits, y_b)
                y_aux = class_idx_to_group_idx(y_b)
                loss_aux = criterion_aux(logits_aux, y_aux)
                loss = loss_main + aux_weight * loss_aux
            else:
                logits, _ = model(X_b, P_b)
                loss_main = criterion(logits, y_b)
                loss_aux = torch.tensor(0.0, device=device)
                loss = loss_main
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                if aux_head:
                    logits, _, logits_aux = model.forward_with_aux(X_b, P_b)
                    loss_main = criterion(logits, y_b)
                    y_aux = class_idx_to_group_idx(y_b)
                    loss_aux = criterion_aux(logits_aux, y_aux)
                    loss = loss_main + aux_weight * loss_aux
                else:
                    logits, _ = model(X_b, P_b)
                    loss_main = criterion(logits, y_b)
                    loss_aux = torch.tensor(0.0, device=device)
                    loss = loss_main
        total_loss += float(loss.item()) * X_b.size(0)
        total_main += float(loss_main.item()) * X_b.size(0)
        total_aux += float(loss_aux.item()) * X_b.size(0)
        total_n += X_b.size(0)
    return total_loss / total_n, total_main / total_n, total_aux / total_n


def _infer(model: nn.Module, loader: DataLoader, device: torch.device):
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 4R-B v2 training (parametrized)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--exp-id", type=str, required=True)
    p.add_argument(
        "--aug",
        type=str,
        default="off",
        choices=["off", "jitter_scale", "jitter_scale_warp"],
    )
    p.add_argument("--jitter-sigma", type=float, default=0.03)
    p.add_argument("--scale-low", type=float, default=0.9)
    p.add_argument("--scale-high", type=float, default=1.1)
    p.add_argument("--warp-low", type=float, default=0.9)
    p.add_argument("--warp-high", type=float, default=1.1)
    p.add_argument("--rnn-dropout", type=float, default=RNN_DROPOUT)
    p.add_argument("--head-dropout", type=float, default=HEAD_DROPOUT)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument(
        "--aux-head", action="store_true",
        help="If set, use Step4RBiGRUAttentionAux with ambiguity-group aux head.",
    )
    p.add_argument(
        "--aux-weight", type=float, default=0.3,
        help="Multi-task weight on aux loss. total_loss = main + aux_weight * aux.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    print("=" * 64)
    print(f"Step 4R-B v2 training | exp_id={args.exp_id} | seed={args.seed} | aug={args.aug}")
    print("=" * 64)

    _set_seed(args.seed)
    paths = _output_paths(args.exp_id, args.seed)

    data = _load_dataset()
    X = data["X_norm"]
    y = data["y"]
    posture = data["posture_oh"]
    split = data["split"]

    train_idx = np.where(split == "train")[0]
    val_idx = np.where(split == "val")[0]
    test_idx = np.where(split == "test")[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Train loader uses augmented dataset; val/test use unaugmented.
    train_ds = AugmentedSequenceDataset(
        X[train_idx], posture[train_idx], y[train_idx],
        mode=args.aug,
        jitter_sigma=args.jitter_sigma,
        scale_low=args.scale_low, scale_high=args.scale_high,
        warp_low=args.warp_low, warp_high=args.warp_high,
        rng_seed=args.seed,
    )
    val_ds = AugmentedSequenceDataset(
        X[val_idx], posture[val_idx], y[val_idx], mode="off", rng_seed=args.seed,
    )
    test_ds = AugmentedSequenceDataset(
        X[test_idx], posture[test_idx], y[test_idx], mode="off", rng_seed=args.seed,
    )
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, generator=g, drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    if args.aux_head:
        model = Step4RBiGRUAttentionAux(
            num_classes=NUM_CLASSES, in_channels=6, seq_len=128,
            hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
            rnn_dropout=args.rnn_dropout, head_dropout=args.head_dropout, posture_dim=3,
        ).to(device)
        model_class_name = "Step4RBiGRUAttentionAux"
    else:
        model = Step4RBiGRUAttention(
            num_classes=NUM_CLASSES, in_channels=6, seq_len=128,
            hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
            rnn_dropout=args.rnn_dropout, head_dropout=args.head_dropout, posture_dim=3,
        ).to(device)
        model_class_name = "Step4RBiGRUAttention"
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"model: {model_class_name}, total params = {n_params} | "
        f"rnn_dropout={args.rnn_dropout} head_dropout={args.head_dropout}"
        + (f" | aux_head ON (weight={args.aux_weight})" if args.aux_head else "")
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    criterion_aux = nn.CrossEntropyLoss() if args.aux_head else None
    print(f"optimizer: AdamW(lr={LR}, weight_decay={args.weight_decay})")

    print()
    print("training ...")
    history: list[dict] = []
    best_macro_f1 = -np.inf
    best_epoch = -1
    bad_epochs = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_main, train_aux = _run_epoch(
            model, train_loader, criterion, device, optimizer,
            aux_head=args.aux_head, aux_weight=args.aux_weight, criterion_aux=criterion_aux,
        )
        # val eval (re-create as deterministic order)
        val_loss, val_main, val_aux = _run_epoch(
            model, val_loader, criterion, device, None,
            aux_head=args.aux_head, aux_weight=args.aux_weight, criterion_aux=criterion_aux,
        )
        val_probs, _, val_y = _infer(model, val_loader, device)
        val_f1 = f1_score(
            val_y, val_probs.argmax(axis=1),
            average="macro", labels=list(range(NUM_CLASSES)), zero_division=0,
        )
        val_acc = float(accuracy_score(val_y, val_probs.argmax(axis=1)))
        elapsed = time.time() - t0
        is_best = False
        if val_f1 > best_macro_f1 + 1e-8:
            best_macro_f1 = float(val_f1)
            best_epoch = epoch
            bad_epochs = 0
            is_best = True
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_macro_f1": best_macro_f1,
                    "config": model.get_config(),
                    "seed": args.seed,
                    "exp_id": args.exp_id,
                    "aug": args.aug,
                    "model_class": model_class_name,
                    "aux_weight": float(args.aux_weight) if args.aux_head else 0.0,
                },
                paths["best_ckpt"],
            )
        else:
            bad_epochs += 1
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_accuracy": val_acc, "val_macro_f1": float(val_f1),
            "lr": LR, "time_seconds": elapsed, "is_best": is_best,
        })
        marker = "  *" if is_best else "   "
        print(
            f"  epoch {epoch:3d} {marker}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  val_macroF1={val_f1:.4f}  ({elapsed:.1f}s)"
        )
        if bad_epochs >= PATIENCE:
            print(f"  early stopping at epoch {epoch}; best epoch={best_epoch} val_F1={best_macro_f1:.4f}")
            break

    pd.DataFrame(history).to_csv(paths["training_log"], index=False, encoding="utf-8-sig")
    print(f"saved training log -> {paths['training_log']}")

    print(f"loading best checkpoint -> {paths['best_ckpt']}")
    ckpt = torch.load(paths["best_ckpt"], map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    # Inference on all splits (deterministic).
    metrics_by_split: dict = {}
    yt_yp_by_split: dict = {}
    pred_rows: list[dict] = []
    for s, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        det_ds = AugmentedSequenceDataset(
            X[idx], posture[idx], y[idx], mode="off", rng_seed=args.seed,
        )
        det_loader = DataLoader(det_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        probs, attn_w, y_idx = _infer(model, det_loader, device)
        m = _evaluate(probs, attn_w, y_idx)
        metrics_by_split[s] = m
        attn_ent = _entropy(attn_w)
        pred_ent = _entropy(probs)
        pred_class = np.array([CLASSES[i] for i in probs.argmax(axis=1)])
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
            sp = np.sort(probs[k])
            row["top1_top2_margin"] = float(sp[-1] - sp[-2])
            pred_rows.append(row)
        yt_yp_by_split[s] = (
            np.array([CLASSES[i] for i in y_idx]),
            pred_class,
        )
        print(
            f"  {s:5s} n={m['n']:4d}  acc={m['accuracy']:.4f}  "
            f"macroF1={m['macro_f1']:.4f}  logloss={m['log_loss']:.4f}  ECE={m['ece_15bin']:.4f}"
        )

    pd.DataFrame(pred_rows).to_csv(paths["preds"], index=False, encoding="utf-8-sig")
    print(f"saved predictions -> {paths['preds']}")

    # Flat metrics CSV (long format)
    rows = []
    flat_keys = [
        "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
        "log_loss", "brier_multiclass", "ece_15bin",
        "top1_prob_mean", "top1_top2_margin_mean",
        "predictive_entropy_mean", "attention_entropy_mean",
        "amb_c2_recall", "amb_c1_c5_c6_internal", "amb_c3_c4_pair",
        "amb_c3_to_c2_absorb", "amb_c4_to_c2_absorb",
    ]
    for s, m in metrics_by_split.items():
        rows.append({"split": s, "metric": "n", "value": float(m["n"])})
        for k in flat_keys:
            rows.append({"split": s, "metric": k, "value": float(m[k])})
        for c in CLASSES:
            pc = m["per_class"][c]
            rows.append({"split": s, "metric": f"precision_{c}", "value": pc["precision"]})
            rows.append({"split": s, "metric": f"recall_{c}", "value": pc["recall"]})
            rows.append({"split": s, "metric": f"f1_{c}", "value": pc["f1"]})
            rows.append({"split": s, "metric": f"support_{c}", "value": pc["support"]})
    pd.DataFrame(rows).to_csv(paths["metrics"], index=False, encoding="utf-8-sig")
    print(f"saved metrics -> {paths['metrics']}")

    # Confusion long
    conf_rows = []
    for s, (yt, yp) in yt_yp_by_split.items():
        cm = confusion_matrix(yt, yp, labels=CLASSES)
        for i, t in enumerate(CLASSES):
            for j, pp in enumerate(CLASSES):
                conf_rows.append({"split": s, "true_class": t, "pred_class": pp, "count": int(cm[i, j])})
    pd.DataFrame(conf_rows).to_csv(paths["confusion"], index=False, encoding="utf-8-sig")
    print(f"saved confusion -> {paths['confusion']}")

    # Minimal md report (per-seed)
    L: list[str] = []
    L.append(f"# Step 4R-B v2 — exp_id=`{args.exp_id}` seed=`{args.seed}` aug=`{args.aug}` 결과")
    L.append("")
    L.append(f"- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`")
    L.append(f"- best epoch: {best_epoch}, best val macroF1: {best_macro_f1:.4f}")
    L.append(f"- device: {device}")
    L.append("")
    L.append("## test 분류·calibration")
    L.append("")
    t = metrics_by_split["test"]
    L.append("| 지표 | 값 |")
    L.append("|---|---:|")
    for k in [
        "accuracy", "macro_f1", "weighted_f1", "log_loss",
        "brier_multiclass", "ece_15bin",
        "amb_c2_recall", "amb_c1_c5_c6_internal", "amb_c3_c4_pair",
        "amb_c3_to_c2_absorb", "amb_c4_to_c2_absorb",
        "predictive_entropy_mean", "attention_entropy_mean",
    ]:
        L.append(f"| {k} | {t[k]:.4f} |")
    L.append("")
    L.append("## per-class F1 (test)")
    L.append("")
    L.append("| class | precision | recall | F1 | support |")
    L.append("|---|---:|---:|---:|---:|")
    for c in CLASSES:
        pc = t["per_class"][c]
        L.append(f"| {c} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} | {pc['support']} |")
    L.append("")
    L.append("*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*")
    paths["report_md"].write_text("\n".join(L), encoding="utf-8")
    print(f"saved report -> {paths['report_md']}")

    print()
    print("=" * 64)
    print(f"Done. test macroF1 = {metrics_by_split['test']['macro_f1']:.4f}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
