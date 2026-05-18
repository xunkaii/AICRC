"""Step 4R-C clinical — t-SNE side-by-side 비교 (v1 vs v2 text embeddings).

본 스크립트는 *신규 파일* 이며 기존 산출물을 수정하지 않는다. v1 / v2 text
embeddings 를 각각 t-SNE 로 2D 투영하고 (PCA 50 → TSNE perplexity 30),
나란히 panel plot 으로 비교한다.

색상 기준:
  - true_class (C1~C6): sample_id 의 'C#' 토큰에서 추출
  - posture_canonical (SA/CA/HW): corpus CSV 의 posture_canonical 컬럼

Reads (read-only):
    data/step4r/4rc_contrastive_optional/text_embeddings_clinical.npz       (v1)
    data/step4r/4rc_contrastive_optional/text_embeddings_clinical_v2.npz    (v2)
    data/step4r/4rc_contrastive_optional/text_corpus_clinical.csv           (v1 meta)
    data/step4r/4rc_contrastive_optional/text_corpus_clinical_v2.csv        (v2 meta)

Writes (new PNG files only; 기존 PNG 보존):
    reports/step4r/4rc_contrastive_optional/tsne_clinical_v1_vs_v2_by_class.png
    reports/step4r/4rc_contrastive_optional/tsne_clinical_v1_vs_v2_by_posture.png

기존 보존 (수정 없음):
    reports/step4r/4rc_contrastive_optional/tsne_by_class_clinical.png
    reports/step4r/4rc_contrastive_optional/tsne_by_ambiguity_group_clinical.png

Run:
    & C:\\Users\\user\\anaconda3\\envs\\dl_env\\python.exe -X utf8 \\
        scripts/visualize_step4rc_tsne_v1_vs_v2.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent

V1_NPZ = PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional" / "text_embeddings_clinical.npz"
V2_NPZ = PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional" / "text_embeddings_clinical_v2.npz"
V1_CSV = PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional" / "text_corpus_clinical.csv"
V2_CSV = PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional" / "text_corpus_clinical_v2.csv"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "step4r" / "4rc_contrastive_optional"
OUT_BY_CLASS = OUTPUT_DIR / "tsne_clinical_v1_vs_v2_by_class.png"
OUT_BY_POSTURE = OUTPUT_DIR / "tsne_clinical_v1_vs_v2_by_posture.png"

# t-SNE settings (sklearn defaults except seed for reproducibility)
PCA_DIM = 50
TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 42
TSNE_INIT = "pca"   # init from PCA for stable layout

CLASS_COLORS = {
    "C1": "#1f77b4",  # blue
    "C2": "#ff7f0e",  # orange
    "C3": "#2ca02c",  # green
    "C4": "#d62728",  # red
    "C5": "#9467bd",  # purple
    "C6": "#8c564b",  # brown
}
POSTURE_COLORS = {
    "SA": "#1f77b4",
    "CA": "#ff7f0e",
    "HW": "#2ca02c",
}


def _infer_true_class(sample_id: str) -> str:
    for p in str(sample_id).split("_"):
        if len(p) == 2 and p[0] == "C" and p[1].isdigit():
            return p
    return ""


def _load_pair(npz_path: Path, csv_path: Path, label: str) -> tuple[np.ndarray, pd.DataFrame]:
    print(f"  loading {label}: {npz_path.name}")
    npz = np.load(npz_path, allow_pickle=True)
    emb = npz["embeddings"]
    sids = npz["sample_id"]
    print(f"    embeddings: shape={emb.shape}, dtype={emb.dtype}")
    print(f"    sample_ids: {len(sids)}")

    csv = pd.read_csv(csv_path, encoding="utf-8-sig")
    csv = csv.set_index("sample_id")
    # align order
    meta = csv.loc[sids].reset_index().rename(columns={"index": "sample_id"})
    meta["true_class"] = meta["sample_id"].apply(_infer_true_class)
    print(f"    metadata rows: {len(meta)}  posture: {meta['posture_canonical'].value_counts().to_dict()}")
    return emb, meta


def _run_tsne(emb: np.ndarray, label: str) -> np.ndarray:
    t0 = time.time()
    print(f"  [{label}] PCA {emb.shape[1]} -> {PCA_DIM} ...", end="", flush=True)
    pca = PCA(n_components=PCA_DIM, random_state=TSNE_RANDOM_STATE)
    emb_pca = pca.fit_transform(emb)
    print(f" done ({time.time() - t0:.1f}s, evr={pca.explained_variance_ratio_.sum():.3f})")

    t1 = time.time()
    print(f"  [{label}] t-SNE perplexity={TSNE_PERPLEXITY} ...", end="", flush=True)
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        random_state=TSNE_RANDOM_STATE,
        init=TSNE_INIT,
        n_jobs=-1,
    )
    proj = tsne.fit_transform(emb_pca)
    print(f" done ({time.time() - t1:.1f}s)")
    return proj


def _scatter_panel(ax, proj: np.ndarray, labels: np.ndarray, color_map: dict, title: str):
    for k, c in color_map.items():
        mask = labels == k
        if mask.any():
            ax.scatter(
                proj[mask, 0], proj[mask, 1],
                s=4, alpha=0.5, c=c, label=f"{k} (n={int(mask.sum())})",
                edgecolors="none",
            )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=8, markerscale=2, framealpha=0.85)


def main() -> int:
    print("=" * 64)
    print("Step 4R-C clinical — t-SNE v1 vs v2 side-by-side")
    print("=" * 64)

    for p in [V1_NPZ, V2_NPZ, V1_CSV, V2_CSV]:
        if not p.exists():
            raise FileNotFoundError(f"required input missing: {p}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[load]")
    emb_v1, meta_v1 = _load_pair(V1_NPZ, V1_CSV, "v1")
    emb_v2, meta_v2 = _load_pair(V2_NPZ, V2_CSV, "v2")

    print("\n[t-SNE]")
    proj_v1 = _run_tsne(emb_v1, "v1")
    proj_v2 = _run_tsne(emb_v2, "v2")

    # --- panel 1: by true_class ---
    print("\n[plot] by true_class")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _scatter_panel(
        axes[0], proj_v1, meta_v1["true_class"].to_numpy(),
        CLASS_COLORS, "v1 (POSTURE 1-fixed × JOINT 2-pool)",
    )
    _scatter_panel(
        axes[1], proj_v2, meta_v2["true_class"].to_numpy(),
        CLASS_COLORS, "v2 (POSTURE 5/5/3-pool × JOINT 3~4-pool)",
    )
    fig.suptitle(
        "Step 4R-C clinical — t-SNE of frozen ko-sroberta-multitask embeddings (color: true class)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(OUT_BY_CLASS, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {OUT_BY_CLASS}")

    # --- panel 2: by posture ---
    print("\n[plot] by posture")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _scatter_panel(
        axes[0], proj_v1, meta_v1["posture_canonical"].to_numpy(),
        POSTURE_COLORS, "v1 (POSTURE 1-fixed × JOINT 2-pool)",
    )
    _scatter_panel(
        axes[1], proj_v2, meta_v2["posture_canonical"].to_numpy(),
        POSTURE_COLORS, "v2 (POSTURE 5/5/3-pool × JOINT 3~4-pool)",
    )
    fig.suptitle(
        "Step 4R-C clinical — t-SNE of frozen ko-sroberta-multitask embeddings (color: posture)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(OUT_BY_POSTURE, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {OUT_BY_POSTURE}")

    # --- summary stats ---
    print("\n[summary]")
    print(f"  v1 n={len(emb_v1)}, dim={emb_v1.shape[1]}")
    print(f"  v2 n={len(emb_v2)}, dim={emb_v2.shape[1]}")
    print(f"  v1 unique phrases: {meta_v1['phrase'].nunique()}")
    print(f"  v2 unique phrases: {meta_v2['phrase'].nunique()}")
    print(f"  v1 phrase mean length: {meta_v1['phrase'].str.len().mean():.1f}")
    print(f"  v2 phrase mean length: {meta_v2['phrase'].str.len().mean():.1f}")

    print()
    print("=" * 64)
    print("Done.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
