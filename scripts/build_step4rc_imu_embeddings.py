"""Step 4R-C Day 2b — Build frozen IMU embeddings from 4R-B b01.5/seed42.

Reads (read-only):
    data/step4r/4rb_attention/step4r_sequence_dataset.npz
    checkpoints/step4r/4rb_attention/experiments/b01_5_aug_jitter_scale_strong/seed42/best.pt
    data/step4r/4rc_contrastive_optional/text_corpus.csv  (for sample order alignment)
    models/step4r_attention_rnn.py  (Step4RBiGRUAttention)

Writes (new file only):
    data/step4r/4rc_contrastive_optional/imu_embeddings.npz
        keys: sample_id, split, attn_vec (9275, 128), checkpoint_meta

Method:
    - Load 4R-B b01.5/seed42 BiGRU+Attention model, freeze all parameters.
    - Forward all 9,275 IMU sequences (X_norm) through the model.
    - Extract the 128-dim attention vector (pre-classifier, pre-posture-concat)
      via a forward hook on the LuongMultiplicativeAttention module.
    - Sample order matches text_corpus.csv (so (text_emb[i], imu_emb[i]) is a
      positive pair).

Design notes:
    - We use attn_vec ONLY (128-dim), not [attn_vec; posture] (131-dim).
      This makes the IMU representation posture-agnostic; posture info is
      present in text via POSTURE_VOCAB so the contrastive task discovers
      whether the IMU signal carries posture-resolvable structure on its own.

Run:
    & C:\\Users\\user\\anaconda3\\envs\\dl_env\\python.exe -X utf8 \
        scripts/build_step4rc_imu_embeddings.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.step4r_attention_rnn import Step4RBiGRUAttention  # noqa: E402

INPUT_NPZ = PROJECT_ROOT / "data" / "step4r" / "4rb_attention" / "step4r_sequence_dataset.npz"
INPUT_CKPT = (
    PROJECT_ROOT / "checkpoints" / "step4r" / "4rb_attention" / "experiments"
    / "b01_5_aug_jitter_scale_strong" / "seed42" / "best.pt"
)
INPUT_TEXT_CORPUS = (
    PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional" / "text_corpus.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional"
OUTPUT_NPZ = OUTPUT_DIR / "imu_embeddings.npz"

POSTURES = ["SA", "CA", "HW"]
POSTURE_TO_IDX = {p: i for i, p in enumerate(POSTURES)}
BATCH_SIZE = 256


def _posture_onehot(arr: np.ndarray) -> np.ndarray:
    out = np.zeros((len(arr), len(POSTURES)), dtype=np.float32)
    for i, p in enumerate(arr):
        out[i, POSTURE_TO_IDX[p]] = 1.0
    return out


def main() -> int:
    print("=" * 64)
    print("Step 4R-C Day 2b — frozen 4R-B b01.5/seed42 IMU embeddings")
    print("=" * 64)

    for p in (INPUT_NPZ, INPUT_CKPT, INPUT_TEXT_CORPUS):
        if not p.exists():
            raise FileNotFoundError(f"required input missing: {p}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"sequence dataset -> {INPUT_NPZ}")
    print(f"checkpoint       -> {INPUT_CKPT}")
    print(f"corpus order     -> {INPUT_TEXT_CORPUS}")
    print(f"output           -> {OUTPUT_NPZ}")
    print()

    # Load sequences + metadata
    z = np.load(INPUT_NPZ, allow_pickle=True)
    X_norm = z["X_norm"].astype(np.float32)
    sample_id_seq = z["sample_id"].astype(str)
    posture_seq = z["posture_canonical"].astype(str)
    split_seq = z["split"].astype(str)
    y_seq = z["y"].astype(np.int64)
    print(f"X_norm shape: {X_norm.shape}")
    print(f"  sample_id count: {len(sample_id_seq)}")

    # Load corpus and align order
    df = pd.read_csv(INPUT_TEXT_CORPUS, encoding="utf-8-sig")
    print(f"text_corpus rows: {len(df)}")
    if len(df) != len(X_norm):
        raise ValueError(
            f"row mismatch: corpus={len(df)} vs X_norm={len(X_norm)}"
        )

    # Build sample_id -> index map for X_norm
    seq_index = {sid: i for i, sid in enumerate(sample_id_seq)}
    missing = [s for s in df["sample_id"] if s not in seq_index]
    if missing:
        raise ValueError(f"{len(missing)} corpus sample_id not in seq dataset; first: {missing[:5]}")

    order = np.array([seq_index[s] for s in df["sample_id"]], dtype=np.int64)
    X_ordered = X_norm[order]
    posture_ordered = posture_seq[order]
    split_ordered = split_seq[order]
    sample_id_ordered = sample_id_seq[order]
    y_ordered = y_seq[order]
    print(f"  aligned X_ordered shape: {X_ordered.shape}")
    print(f"  posture alignment check: {(posture_ordered == df['posture_canonical'].to_numpy()).all()}")
    print(f"  split   alignment check: {(split_ordered == df['split'].to_numpy()).all()}")

    posture_oh = _posture_onehot(posture_ordered)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\ndevice: {device}")
    if device.type == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(0)}")

    print(f"\nloading checkpoint -> {INPUT_CKPT}")
    chk = torch.load(INPUT_CKPT, map_location=device, weights_only=False)
    print(f"  epoch={chk['epoch']}  val_macro_f1={chk['val_macro_f1']:.4f}")
    print(f"  exp_id={chk.get('exp_id', 'unknown')}  seed={chk.get('seed', 'unknown')}")
    config = chk["config"]
    model = Step4RBiGRUAttention(**config).to(device)
    model.load_state_dict(chk["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model loaded, params={n_params}, all frozen")

    # Register forward hook on the attention module to capture attn_vec
    captured: dict = {"attn_vecs": []}

    def attn_hook(module, inputs, output):
        # output: (attention_vector, attention_weights)
        attn_vec, _attn_w = output
        captured["attn_vecs"].append(attn_vec.detach().cpu().numpy())

    handle = model.attention.register_forward_hook(attn_hook)

    # Forward all 9275 in batches
    ds = TensorDataset(
        torch.from_numpy(X_ordered).float(),
        torch.from_numpy(posture_oh).float(),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"\nforward pass (batch_size={BATCH_SIZE}) ...")
    with torch.no_grad():
        for X_b, P_b in loader:
            X_b = X_b.to(device, non_blocking=True)
            P_b = P_b.to(device, non_blocking=True)
            _ = model(X_b, P_b)  # triggers hook on attention module
    handle.remove()

    attn_vecs = np.concatenate(captured["attn_vecs"], axis=0)
    print(f"\nattn_vec shape: {attn_vecs.shape}, dtype: {attn_vecs.dtype}")
    print(f"  per-feature mean: {attn_vecs.mean(axis=0)[:5]} ...")
    print(f"  per-feature std:  {attn_vecs.std(axis=0)[:5]} ...")
    print(f"  L2 norm distribution: mean={np.linalg.norm(attn_vecs, axis=1).mean():.4f}")
    print(f"                       min={np.linalg.norm(attn_vecs, axis=1).min():.4f}")
    print(f"                       max={np.linalg.norm(attn_vecs, axis=1).max():.4f}")

    if attn_vecs.shape[0] != 9275:
        raise ValueError(f"expected 9275 rows, got {attn_vecs.shape[0]}")
    if attn_vecs.shape[1] != 128:
        raise ValueError(f"expected 128-dim, got {attn_vecs.shape[1]}")

    # Save
    np.savez(
        OUTPUT_NPZ,
        sample_id=sample_id_ordered,
        split=split_ordered,
        y=y_ordered,
        posture_canonical=posture_ordered,
        attn_vec=attn_vecs.astype(np.float32),
        checkpoint_exp=np.array([chk.get("exp_id", "unknown")]),
        checkpoint_seed=np.array([chk.get("seed", -1)]),
        checkpoint_epoch=np.array([chk["epoch"]]),
    )
    print(f"\nsaved -> {OUTPUT_NPZ}")

    # Diagnostic — within-class IMU embedding cohesion vs between-class separation
    print("\n=== diagnostic: per-class attn_vec statistics ===")
    CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
    for i, c in enumerate(CLASSES):
        mask = y_ordered == i
        n = int(mask.sum())
        if n == 0:
            continue
        sub = attn_vecs[mask]
        norms = np.linalg.norm(sub, axis=1)
        print(
            f"  {c}: n={n:4d}  |mean|={sub.mean():+.4f}  "
            f"|std|={sub.std():.4f}  L2={norms.mean():.4f}±{norms.std():.4f}"
        )

    # Quick check: are the IMU embeddings non-degenerate (variance > 0)?
    var = attn_vecs.var(axis=0)
    print(f"\nfeature variance: min={var.min():.6f}  mean={var.mean():.6f}  max={var.max():.6f}")
    if var.max() < 1e-4:
        print("  WARN: very low feature variance — IMU embedding may be degenerate.")
    else:
        print("  OK: feature variance non-degenerate.")

    print()
    print("=" * 64)
    print("Done.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
