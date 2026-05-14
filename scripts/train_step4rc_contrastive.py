"""Step 4R-C Day 3-4 — Contrastive IMU-text projection learning.

Reads (read-only):
    data/step4r/4rc_contrastive_optional/imu_embeddings.npz   (9275, 128)
    data/step4r/4rc_contrastive_optional/text_embeddings.npz  (9275, 768)
    data/step4r/4rc_contrastive_optional/text_corpus.csv      (schema info for collapse check)

Writes (new files only):
    checkpoints/step4r/4rc_contrastive_optional/projection_head.pt
    data/step4r/4rc_contrastive_optional/joint_embeddings.npz   (projected, 9275, 128 each)
    reports/step4r/4rc_contrastive_optional/training_log.csv
    reports/step4r/4rc_contrastive_optional/results.md

Architecture:
    ProjectionHead (IMU)  = Linear(128->128) -> ReLU -> Linear(128->128) -> L2 norm
    ProjectionHead (text) = Linear(768->128) -> ReLU -> Linear(128->128) -> L2 norm

Loss:
    Symmetric InfoNCE, temperature T = 0.07.
    Negatives: random (in-batch, no masking). Note that text duplicates within
    a batch can appear as "negatives" (text vectors are byte-identical for
    same-template_key pairs); since the optimal solution makes IMU embeddings
    of same-template samples cluster, this is self-consistent. We report
    *template-match* R@K as the meaningful metric alongside strict R@K.

Optim:
    AdamW lr=1e-3, weight_decay=1e-4, batch=256, max_epochs=50,
    early stopping on val template_i2t_R@1 patience=10.

Abort:
    If strict val R@1 stays < 2x chance level (0.0014) for 10 consecutive
    epochs, training stops and reports.

Run:
    & C:\\Users\\user\\anaconda3\\envs\\dl_env\\python.exe -X utf8 \
        scripts/train_step4rc_contrastive.py
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_IMU = PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional" / "imu_embeddings.npz"
INPUT_TEXT = PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional" / "text_embeddings.npz"
INPUT_CORPUS = PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional" / "text_corpus.csv"

OUTPUT_CKPT_DIR = PROJECT_ROOT / "checkpoints" / "step4r" / "4rc_contrastive_optional"
OUTPUT_DATA_DIR = PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "reports" / "step4r" / "4rc_contrastive_optional"

OUTPUT_CKPT = OUTPUT_CKPT_DIR / "projection_head.pt"
OUTPUT_JOINT_NPZ = OUTPUT_DATA_DIR / "joint_embeddings.npz"
OUTPUT_LOG_CSV = OUTPUT_REPORT_DIR / "training_log.csv"
OUTPUT_REPORT_MD = OUTPUT_REPORT_DIR / "results.md"

# Hyperparameters
SEED = 42
PROJECTION_DIM = 128
HIDDEN_DIM = 128
IMU_INPUT_DIM = 128
TEXT_INPUT_DIM = 768
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 10
TEMPERATURE = 0.07
ABORT_PATIENCE_EPOCHS = 10
ABORT_R1_THRESHOLD_MULT = 2.0  # require strict R@1 > 2x chance


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


def symmetric_infonce(imu_proj: torch.Tensor, text_proj: torch.Tensor, temperature: float) -> torch.Tensor:
    """Standard CLIP-style symmetric InfoNCE.

    Both inputs are L2-normalized so logits are cosine similarities / T.
    Diagonal is the positive pair; all other entries are treated as random
    negatives (no template-aware masking — v1 behavior per user spec).
    """
    n = imu_proj.shape[0]
    logits = imu_proj @ text_proj.T / temperature   # (n, n)
    labels = torch.arange(n, device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_i2t + loss_t2i)


def compute_retrieval(
    imu_proj: np.ndarray, text_proj: np.ndarray, template_keys: np.ndarray,
    k_list=(1, 5, 10),
) -> dict:
    """Compute strict R@K (exact-index) and template-match R@K both directions."""
    n = imu_proj.shape[0]
    sim = imu_proj @ text_proj.T  # (n, n), already cosine since normalized

    sorted_idx_i2t = np.argsort(-sim, axis=1)
    sorted_idx_t2i = np.argsort(-sim.T, axis=1)
    diag = np.arange(n)
    metrics: dict = {}

    for k in k_list:
        # strict — does top-K contain the exact same row index?
        topk_i2t = sorted_idx_i2t[:, :k]
        strict_hit_i2t = (topk_i2t == diag[:, None]).any(axis=1)
        metrics[f"strict_i2t_R@{k}"] = float(strict_hit_i2t.mean())

        topk_t2i = sorted_idx_t2i[:, :k]
        strict_hit_t2i = (topk_t2i == diag[:, None]).any(axis=1)
        metrics[f"strict_t2i_R@{k}"] = float(strict_hit_t2i.mean())

        # template-match — does top-K contain any sample with same template_key?
        topk_templates_i2t = template_keys[topk_i2t]  # (n, k)
        template_hit_i2t = (topk_templates_i2t == template_keys[:, None]).any(axis=1)
        metrics[f"template_i2t_R@{k}"] = float(template_hit_i2t.mean())

        topk_templates_t2i = template_keys[topk_t2i]
        template_hit_t2i = (topk_templates_t2i == template_keys[:, None]).any(axis=1)
        metrics[f"template_t2i_R@{k}"] = float(template_hit_t2i.mean())

    return metrics


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    print("=" * 64)
    print("Step 4R-C Day 3-4 — contrastive projection learning")
    print("=" * 64)
    for p in (INPUT_IMU, INPUT_TEXT, INPUT_CORPUS):
        if not p.exists():
            raise FileNotFoundError(p)
    for d in (OUTPUT_CKPT_DIR, OUTPUT_DATA_DIR, OUTPUT_REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    _set_seed(SEED)

    # Load embeddings
    imu_npz = np.load(INPUT_IMU, allow_pickle=True)
    text_npz = np.load(INPUT_TEXT, allow_pickle=True)
    corpus = pd.read_csv(INPUT_CORPUS, encoding="utf-8-sig")

    imu_emb = imu_npz["attn_vec"].astype(np.float32)   # (9275, 128)
    text_emb = text_npz["embeddings"].astype(np.float32)  # (9275, 768) L2-normalized
    sample_id = imu_npz["sample_id"].astype(str)
    split = imu_npz["split"].astype(str)
    posture = imu_npz["posture_canonical"].astype(str)
    y = imu_npz["y"].astype(np.int64)

    # Verify alignment via sample_id
    assert (sample_id == text_npz["sample_id"].astype(str)).all(), "sample_id mismatch IMU/text"
    assert (sample_id == corpus["sample_id"].to_numpy()).all(), "sample_id mismatch IMU/corpus"
    template_keys = corpus["template_key"].to_numpy()
    print(f"loaded n={len(sample_id)}  imu={imu_emb.shape}  text={text_emb.shape}")
    print(f"unique templates: {len(np.unique(template_keys))}")

    # Build split masks
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    n_train, n_val, n_test = int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum())
    print(f"split: train={n_train}  val={n_val}  test={n_test}")
    chance_strict_R1_val = 1.0 / n_val
    n_unique_templates_val = len(np.unique(template_keys[val_mask]))
    print(f"val strict R@1 chance: {chance_strict_R1_val:.6f}  (= 1/{n_val})")
    print(f"val template R@1 chance ~ 1/{n_unique_templates_val}  (rough)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Train dataset / loader
    imu_t = torch.from_numpy(imu_emb).float()
    text_t = torch.from_numpy(text_emb).float()
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    train_ds = TensorDataset(imu_t[train_idx], text_t[train_idx])
    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, generator=g, drop_last=False)

    # Build projection heads
    imu_head = ProjectionHead(IMU_INPUT_DIM, HIDDEN_DIM, PROJECTION_DIM).to(device)
    text_head = ProjectionHead(TEXT_INPUT_DIM, HIDDEN_DIM, PROJECTION_DIM).to(device)
    params = list(imu_head.parameters()) + list(text_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    n_params = sum(p.numel() for p in params)
    print(f"\nprojection heads: total params = {n_params}")
    print(f"  imu_head:  Linear({IMU_INPUT_DIM}->{HIDDEN_DIM}) -> ReLU -> Linear({HIDDEN_DIM}->{PROJECTION_DIM}) -> L2norm")
    print(f"  text_head: Linear({TEXT_INPUT_DIM}->{HIDDEN_DIM}) -> ReLU -> Linear({HIDDEN_DIM}->{PROJECTION_DIM}) -> L2norm")

    # --- Training loop ---
    print("\ntraining ...")
    history: list[dict] = []
    best_val_template_R1 = -np.inf
    best_epoch = -1
    bad_epochs = 0
    abort_low_R1_streak = 0
    abort_threshold = chance_strict_R1_val * ABORT_R1_THRESHOLD_MULT

    imu_val_t = imu_t[val_idx].to(device)
    text_val_t = text_t[val_idx].to(device)
    val_template = template_keys[val_idx]

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        imu_head.train(); text_head.train()
        epoch_loss_sum = 0.0
        epoch_n = 0
        for x_imu, x_text in train_loader:
            x_imu = x_imu.to(device, non_blocking=True)
            x_text = x_text.to(device, non_blocking=True)
            optimizer.zero_grad()
            z_imu = imu_head(x_imu)
            z_text = text_head(x_text)
            loss = symmetric_infonce(z_imu, z_text, TEMPERATURE)
            loss.backward()
            optimizer.step()
            epoch_loss_sum += float(loss.item()) * x_imu.size(0)
            epoch_n += x_imu.size(0)
        train_loss = epoch_loss_sum / epoch_n

        # val eval
        imu_head.eval(); text_head.eval()
        with torch.no_grad():
            z_imu_v = imu_head(imu_val_t).cpu().numpy()
            z_text_v = text_head(text_val_t).cpu().numpy()
        m = compute_retrieval(z_imu_v, z_text_v, val_template, k_list=(1, 5, 10))
        elapsed = time.time() - t0

        is_best = m["template_i2t_R@1"] > best_val_template_R1 + 1e-8
        if is_best:
            best_val_template_R1 = m["template_i2t_R@1"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save({
                "epoch": epoch,
                "imu_head_state_dict": imu_head.state_dict(),
                "text_head_state_dict": text_head.state_dict(),
                "config": {
                    "imu_input_dim": IMU_INPUT_DIM,
                    "text_input_dim": TEXT_INPUT_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "projection_dim": PROJECTION_DIM,
                    "temperature": TEMPERATURE,
                },
                "val_template_i2t_R@1": best_val_template_R1,
                "seed": SEED,
            }, OUTPUT_CKPT)
        else:
            bad_epochs += 1

        # abort tracking — based on strict R@1
        if m["strict_i2t_R@1"] < abort_threshold:
            abort_low_R1_streak += 1
        else:
            abort_low_R1_streak = 0

        log_row = {
            "epoch": epoch, "train_loss": train_loss,
            "val_strict_i2t_R@1": m["strict_i2t_R@1"],
            "val_strict_t2i_R@1": m["strict_t2i_R@1"],
            "val_strict_i2t_R@5": m["strict_i2t_R@5"],
            "val_strict_t2i_R@5": m["strict_t2i_R@5"],
            "val_template_i2t_R@1": m["template_i2t_R@1"],
            "val_template_t2i_R@1": m["template_t2i_R@1"],
            "val_template_i2t_R@5": m["template_i2t_R@5"],
            "val_template_t2i_R@5": m["template_t2i_R@5"],
            "is_best": is_best, "time_seconds": elapsed,
        }
        history.append(log_row)

        marker = "  *" if is_best else "   "
        print(
            f"  epoch {epoch:3d} {marker}  loss={train_loss:.4f}  "
            f"strict R@1 i2t={m['strict_i2t_R@1']:.4f} t2i={m['strict_t2i_R@1']:.4f}  "
            f"template R@1 i2t={m['template_i2t_R@1']:.4f} t2i={m['template_t2i_R@1']:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if abort_low_R1_streak >= ABORT_PATIENCE_EPOCHS:
            print(
                f"  ABORT: strict val R@1 < {abort_threshold:.4f} for "
                f"{ABORT_PATIENCE_EPOCHS} consecutive epochs."
            )
            break
        if bad_epochs >= PATIENCE:
            print(f"  early stopping at epoch {epoch}; best epoch={best_epoch}  template R@1={best_val_template_R1:.4f}")
            break

    pd.DataFrame(history).to_csv(OUTPUT_LOG_CSV, index=False, encoding="utf-8-sig")
    print(f"saved training log -> {OUTPUT_LOG_CSV}")

    # --- Final evaluation on test using best checkpoint ---
    print()
    print("loading best checkpoint for final eval ...")
    chk = torch.load(OUTPUT_CKPT, map_location=device, weights_only=False)
    imu_head.load_state_dict(chk["imu_head_state_dict"])
    text_head.load_state_dict(chk["text_head_state_dict"])
    imu_head.eval(); text_head.eval()

    with torch.no_grad():
        z_imu_all = imu_head(imu_t.to(device)).cpu().numpy()
        z_text_all = text_head(text_t.to(device)).cpu().numpy()

    # Per-split final metrics
    final_metrics = {}
    for s_name, s_idx, s_template in [
        ("train", train_idx, template_keys[train_idx]),
        ("val", val_idx, template_keys[val_idx]),
        ("test", test_idx, template_keys[test_idx]),
    ]:
        m = compute_retrieval(z_imu_all[s_idx], z_text_all[s_idx], s_template, k_list=(1, 5, 10))
        final_metrics[s_name] = m
        print(
            f"  {s_name:5s}  strict R@1 i2t={m['strict_i2t_R@1']:.4f}  R@5={m['strict_i2t_R@5']:.4f}  |  "
            f"template R@1 i2t={m['template_i2t_R@1']:.4f}  R@5={m['template_i2t_R@5']:.4f}"
        )

    # Save joint embeddings (all 9275, projected)
    np.savez(
        OUTPUT_JOINT_NPZ,
        sample_id=sample_id,
        split=split,
        y=y,
        posture_canonical=posture,
        template_key=template_keys,
        imu_proj=z_imu_all.astype(np.float32),
        text_proj=z_text_all.astype(np.float32),
        best_epoch=np.array([chk["epoch"]]),
        seed=np.array([SEED]),
    )
    print(f"saved joint embeddings -> {OUTPUT_JOINT_NPZ}")

    # Schema collapse check (read text_corpus to verify schema distribution unchanged)
    print()
    print("=== schema collapse check (sanity: 4R-C does NOT modify schema) ===")
    sc_counts = corpus["caption_confidence_level"].value_counts()
    n_total = len(corpus)
    sc_dist = {k: float(v / n_total) for k, v in sc_counts.items()}
    print(f"  schema level distribution: {sc_dist}")
    # Expected from b01.5/seed42 schema_summary (4R-C input):
    expected = {"confident": 0.166, "hedged": 0.789, "low": 0.038, "no_call": 0.007}
    delta_max = 0.0
    for level, expected_rate in expected.items():
        actual = sc_dist.get(level, 0.0)
        delta = abs(actual - expected_rate)
        delta_max = max(delta_max, delta)
        print(f"    {level:10s}: actual={actual:.4f}  expected={expected_rate:.4f}  delta={delta:+.4f}")
    print(f"  max delta: {delta_max:.4f}  (4R-C reads schema as-is, no collapse possible)")

    # --- Markdown report ---
    L: list[str] = []
    L.append("# Step 4R-C Day 3-4 — Contrastive Projection 학습 결과")
    L.append("")
    L.append(f"- 생성 스크립트: `scripts/train_step4rc_contrastive.py`")
    L.append(f"- best epoch: {chk['epoch']} / max {MAX_EPOCHS}")
    L.append(f"- final epoch trained: {len(history)}")
    L.append(f"- seed: {SEED}, temperature: {TEMPERATURE}, batch: {BATCH_SIZE}")
    L.append(f"- input: imu (9275, 128) + text (9275, 768)")
    L.append(f"- projection: imu/text -> 128-d L2-normalized")
    L.append("")
    L.append("## 1. Final retrieval metrics (best checkpoint)")
    L.append("")
    L.append("| split | strict R@1 i2t | strict R@5 i2t | template R@1 i2t | template R@5 i2t | template R@1 t2i |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for s_name in ["train", "val", "test"]:
        m = final_metrics[s_name]
        L.append(
            f"| {s_name} | {m['strict_i2t_R@1']:.4f} | {m['strict_i2t_R@5']:.4f} | "
            f"{m['template_i2t_R@1']:.4f} | {m['template_i2t_R@5']:.4f} | "
            f"{m['template_t2i_R@1']:.4f} |"
        )
    L.append("")
    L.append("**Interpretation**: strict R@1 chance = 1/n_split; template R@1 chance ~ 1/n_unique_templates.")
    L.append("")
    L.append("## 2. Training history (loss + R@K curve)")
    L.append("")
    L.append("| epoch | train loss | val strict R@1 (i2t) | val template R@1 (i2t) | best? |")
    L.append("|---|---:|---:|---:|:---:|")
    for h in history[:5]:
        L.append(f"| {h['epoch']} | {h['train_loss']:.4f} | {h['val_strict_i2t_R@1']:.4f} | {h['val_template_i2t_R@1']:.4f} | {'*' if h['is_best'] else ''} |")
    if len(history) > 10:
        L.append("| ... | ... | ... | ... | |")
    for h in history[-5:]:
        L.append(f"| {h['epoch']} | {h['train_loss']:.4f} | {h['val_strict_i2t_R@1']:.4f} | {h['val_template_i2t_R@1']:.4f} | {'*' if h['is_best'] else ''} |")
    L.append("")
    L.append(f"Full log: `reports/step4r/4rc_contrastive_optional/training_log.csv`")
    L.append("")
    L.append("## 3. Schema collapse check")
    L.append("")
    L.append("4R-C 학습은 schema를 *읽기 전용*으로 사용하므로 schema 분포 변화 불가능. 그래도 sanity check 결과:")
    L.append("")
    L.append("| level | actual rate | expected (b01.5/seed42 schema) | delta |")
    L.append("|---|---:|---:|---:|")
    for level, exp_rate in expected.items():
        act = sc_dist.get(level, 0.0)
        L.append(f"| {level} | {act:.4f} | {exp_rate:.4f} | {act - exp_rate:+.4f} |")
    L.append(f"\n→ **No collapse** (4R-C는 새 model output 만들지 않음, schema input 그대로 사용).")
    L.append("")
    L.append("## 4. 산출물")
    L.append("")
    L.append("- `checkpoints/step4r/4rc_contrastive_optional/projection_head.pt`")
    L.append("- `data/step4r/4rc_contrastive_optional/joint_embeddings.npz`")
    L.append("- `reports/step4r/4rc_contrastive_optional/training_log.csv`")
    L.append("- `reports/step4r/4rc_contrastive_optional/results.md` (본 보고서)")
    L.append("")
    OUTPUT_REPORT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"saved markdown -> {OUTPUT_REPORT_MD}")

    print()
    print("=" * 64)
    print(f"Done. best epoch={chk['epoch']}  val template R@1={best_val_template_R1:.4f}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
