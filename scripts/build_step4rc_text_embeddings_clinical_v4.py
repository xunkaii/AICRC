"""Step 4R-C clinical v4 — Frozen Korean SBERT embeddings on v4 corpus.

Mirror of ``build_step4rc_text_embeddings_clinical.py`` (v1). Paths only.

Reads (read-only):
    data/step4r/4rc_contrastive_optional/text_corpus_clinical_v4.csv

Writes (new files only):
    data/step4r/4rc_contrastive_optional/text_embeddings_clinical_v4.npz
    data/step4r/4rc_contrastive_optional/text_unique_similarity_clinical_v4.csv

Run:
    & C:\\Users\\user\\anaconda3\\envs\\dl_env\\python.exe -X utf8 \\
        scripts/build_step4rc_text_embeddings_clinical_v4.py
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

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional"
    / "text_corpus_clinical_v4.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional"
OUTPUT_NPZ = OUTPUT_DIR / "text_embeddings_clinical_v4.npz"
OUTPUT_SIM_CSV = OUTPUT_DIR / "text_unique_similarity_clinical_v4.csv"

MODEL_NAME = "jhgan/ko-sroberta-multitask"
BATCH_SIZE = 64
ABORT_THRESHOLD = 0.95


def main() -> int:
    print("=" * 64)
    print("Step 4R-C clinical v4 — frozen ko-sroberta-multitask embeddings")
    print("=" * 64)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"input not found: {INPUT_CSV}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"input  -> {INPUT_CSV}")
    print(f"output -> {OUTPUT_NPZ}")
    print()

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    print(f"loaded n={len(df)} rows")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(0)}")

    print(f"\nloading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.eval()
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

    print("\n=== diagnostic: similarity between unique template_keys ===")
    unique = df.drop_duplicates("template_key").reset_index(drop=True)
    unique_keys = unique["template_key"].to_numpy()
    first_occurrence = (
        df.reset_index().drop_duplicates("template_key").set_index("template_key")["index"]
    )
    unique_embs = np.stack([embeddings[first_occurrence[k]] for k in unique_keys])
    n = unique_embs.shape[0]
    print(f"  unique template_keys: {n}")
    print(f"  unique phrases: {df['phrase'].nunique()}")

    sim = cosine_similarity(unique_embs, unique_embs)
    off_mask = ~np.eye(n, dtype=bool)
    off_diag = sim[off_mask]
    print(f"  off-diagonal cosine: mean={off_diag.mean():.4f} std={off_diag.std():.4f} max={off_diag.max():.4f}")
    high_pct = float((off_diag > ABORT_THRESHOLD).mean())
    print(f"  >{ABORT_THRESHOLD}: {(off_diag > ABORT_THRESHOLD).sum()}/{len(off_diag)}  ({high_pct:.1%})")

    rows = []
    for i in range(n):
        for j in range(n):
            rows.append({"template_key_a": unique_keys[i], "template_key_b": unique_keys[j], "cosine": float(sim[i, j])})
    pd.DataFrame(rows).to_csv(OUTPUT_SIM_CSV, index=False, encoding="utf-8-sig")
    print(f"saved similarity matrix -> {OUTPUT_SIM_CSV}")

    if high_pct >= 0.999:
        print(f"  ABORT: all cosines > {ABORT_THRESHOLD}")
        return 2
    print(f"  OK: {high_pct:.1%} pairs > {ABORT_THRESHOLD}")

    print()
    print("=" * 64)
    print("Done.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
