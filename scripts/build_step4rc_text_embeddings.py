"""Step 4R-C Day 2a — Build frozen Korean SBERT text embeddings.

Reads (read-only):
    data/step4r/4rc_contrastive_optional/text_corpus.csv

Writes (new file only):
    data/step4r/4rc_contrastive_optional/text_embeddings.npz
        keys: sample_id, split, template_key, phrase, embeddings (9275, 768),
              model_name
    data/step4r/4rc_contrastive_optional/text_unique_similarity.csv
        37x37 cosine-similarity matrix between unique-phrase representatives
        (long-format: template_key_a, template_key_b, cosine)

Method:
    - sentence-transformers loads `jhgan/ko-sroberta-multitask` (frozen).
    - 9,275 phrases encoded in batched eval mode (no gradient).
    - L2-normalized 768-dim embeddings stored as float32.

Diagnostics:
    - shape verification (expect 9275 x 768)
    - per-unique-phrase representative embedding similarity matrix
    - abort criterion: if all off-diagonal cosine similarities > 0.95,
      SBERT cannot distinguish our phrases → stop and report

Run:
    & C:\\Users\\user\\anaconda3\\envs\\dl_env\\python.exe -X utf8 \
        scripts/build_step4rc_text_embeddings.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional" / "text_corpus.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional"
OUTPUT_NPZ = OUTPUT_DIR / "text_embeddings.npz"
OUTPUT_SIM_CSV = OUTPUT_DIR / "text_unique_similarity.csv"

MODEL_NAME = "jhgan/ko-sroberta-multitask"
BATCH_SIZE = 64
ABORT_THRESHOLD = 0.95  # if all off-diagonal cosines > this, abort


def main() -> int:
    print("=" * 64)
    print("Step 4R-C Day 2a — frozen ko-sroberta-multitask text embeddings")
    print("=" * 64)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"input not found: {INPUT_CSV}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"input  -> {INPUT_CSV}")
    print(f"output -> {OUTPUT_NPZ}")
    print(f"sim    -> {OUTPUT_SIM_CSV}")
    print()

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    print(f"loaded n={len(df)} rows from text_corpus.csv")
    if len(df) != 9275:
        print(f"  warn: row count != 9275 (got {len(df)})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(0)}")

    print(f"\nloading model: {MODEL_NAME}")
    print("  (first run downloads ~420MB from HuggingFace)")
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.eval()
    # Freeze params (defense-in-depth; encode() already uses no_grad).
    for p in model.parameters():
        p.requires_grad = False
    print(f"  loaded. embedding dim: {model.get_sentence_embedding_dimension()}")

    phrases = df["phrase"].tolist()
    print(f"\nencoding {len(phrases)} phrases (batch_size={BATCH_SIZE}) ...")
    with torch.no_grad():
        embeddings = model.encode(
            phrases,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    print(f"\nembedding shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    print(f"  norm check (L2 mean): {np.linalg.norm(embeddings, axis=1).mean():.6f}")
    print(f"  embedding[0, :5]: {embeddings[0, :5]}")

    np.savez(
        OUTPUT_NPZ,
        sample_id=df["sample_id"].to_numpy(),
        split=df["split"].to_numpy(),
        template_key=df["template_key"].to_numpy(),
        phrase=df["phrase"].to_numpy(),
        embeddings=embeddings.astype(np.float32),
        model_name=np.array([MODEL_NAME]),
    )
    print(f"saved -> {OUTPUT_NPZ}")

    # --- Diagnostic: unique-phrase similarity matrix ---
    print("\n=== diagnostic: similarity between unique phrases ===")
    unique = df.drop_duplicates("template_key").reset_index(drop=True)
    unique_idx = unique.index.to_numpy()
    unique_keys = unique["template_key"].to_numpy()
    unique_phrases = unique["phrase"].to_numpy()
    # Get embedding for each unique template's first occurrence
    first_occurrence = (
        df.reset_index()
        .drop_duplicates("template_key")
        .set_index("template_key")["index"]
    )
    unique_embs = np.stack([embeddings[first_occurrence[k]] for k in unique_keys])
    n = unique_embs.shape[0]
    print(f"  unique phrases: {n}")

    sim = cosine_similarity(unique_embs, unique_embs)
    off_mask = ~np.eye(n, dtype=bool)
    off_diag = sim[off_mask]
    print(f"  off-diagonal cosine (n={len(off_diag)} pairs):")
    print(f"    mean   {off_diag.mean():.4f}")
    print(f"    std    {off_diag.std():.4f}")
    print(f"    min    {off_diag.min():.4f}")
    print(f"    p25    {np.percentile(off_diag, 25):.4f}")
    print(f"    p50    {np.percentile(off_diag, 50):.4f}")
    print(f"    p75    {np.percentile(off_diag, 75):.4f}")
    print(f"    max    {off_diag.max():.4f}")
    high_pct = float((off_diag > ABORT_THRESHOLD).mean())
    print(f"    >{ABORT_THRESHOLD}: {(off_diag > ABORT_THRESHOLD).sum()}/{len(off_diag)}  ({high_pct:.1%})")

    # Save similarity matrix as long-format CSV
    rows = []
    for i in range(n):
        for j in range(n):
            rows.append({
                "template_key_a": unique_keys[i],
                "template_key_b": unique_keys[j],
                "cosine": float(sim[i, j]),
            })
    pd.DataFrame(rows).to_csv(OUTPUT_SIM_CSV, index=False, encoding="utf-8-sig")
    print(f"\nsaved similarity matrix -> {OUTPUT_SIM_CSV}")

    # Print most-similar and least-similar pairs
    print("\n  5 closest pairs (excluding self):")
    triu_mask = np.triu(off_mask, k=1)
    pair_idxs = np.argsort(-sim[triu_mask])[:5]
    iu, ju = np.where(triu_mask)
    for k in pair_idxs:
        i, j = iu[k], ju[k]
        print(f"    cos={sim[i,j]:.4f}  '{unique_keys[i]}'  <->  '{unique_keys[j]}'")

    print("\n  5 most-distant pairs:")
    pair_idxs = np.argsort(sim[triu_mask])[:5]
    for k in pair_idxs:
        i, j = iu[k], ju[k]
        print(f"    cos={sim[i,j]:.4f}  '{unique_keys[i]}'  <->  '{unique_keys[j]}'")

    print("\n=== abort check ===")
    if high_pct >= 0.999:
        print(f"  ABORT: all off-diagonal cosines > {ABORT_THRESHOLD} — SBERT cannot distinguish phrases.")
        print("  Recommend: enrich phrases (longer text, more class detail) before continuing.")
        return 2
    elif high_pct > 0.5:
        print(f"  WARN: {high_pct:.1%} of pairs > {ABORT_THRESHOLD} — diversity is low but workable.")
    else:
        print(f"  OK: only {high_pct:.1%} of pairs > {ABORT_THRESHOLD} — phrases distinguishable.")

    print()
    print("=" * 64)
    print("Done.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
