"""Step 4R-C clinical — TRAINED t-SNE side-by-side (v1 vs v2 joint embeddings).

본 스크립트는 *신규 파일* 이며 기존 산출물을 수정하지 않는다.
v1 / v2 contrastive 학습 후 joint_embeddings_clinical*.npz 의 **text_proj** 를
2D 로 투영해서 학습 후 표현 공간을 비교한다.

pre-training t-SNE (visualize_step4rc_tsne_v1_vs_v2.py) 가 768-dim SBERT
embedding 을 보여주는 반면, 본 스크립트는 학습 후 128-dim projected text 만 본다.
contrastive training 이 표현 공간을 어떻게 재조직했는지 직접 비교.

색상: true_class (C1~C6), posture_canonical (SA/CA/HW)

Reads (read-only):
    data/step4r/4rc_contrastive_optional/joint_embeddings_clinical.npz       (v1, post-train)
    data/step4r/4rc_contrastive_optional/joint_embeddings_clinical_v2.npz    (v2, post-train)

Writes (new PNG files only):
    reports/step4r/4rc_contrastive_optional/tsne_trained_v1_vs_v2_by_class.png
    reports/step4r/4rc_contrastive_optional/tsne_trained_v1_vs_v2_by_posture.png

Run:
    & C:\\Users\\user\\anaconda3\\envs\\dl_env\\python.exe -X utf8 \\
        scripts/visualize_step4rc_tsne_trained_v1_vs_v2.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
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

V1_JOINT = PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional" / "joint_embeddings_clinical.npz"
V2_JOINT = PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional" / "joint_embeddings_clinical_v2.npz"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "step4r" / "4rc_contrastive_optional"
OUT_BY_CLASS = OUTPUT_DIR / "tsne_trained_v1_vs_v2_by_class.png"
OUT_BY_POSTURE = OUTPUT_DIR / "tsne_trained_v1_vs_v2_by_posture.png"

TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 42
TSNE_INIT = "pca"

CLASS_COLORS = {
    "C1": "#1f77b4", "C2": "#ff7f0e", "C3": "#2ca02c",
    "C4": "#d62728", "C5": "#9467bd", "C6": "#8c564b",
}
POSTURE_COLORS = {"SA": "#1f77b4", "CA": "#ff7f0e", "HW": "#2ca02c"}


def _infer_true_class(sample_id: str) -> str:
    for p in str(sample_id).split("_"):
        if len(p) == 2 and p[0] == "C" and p[1].isdigit():
            return p
    return ""


def _load(npz_path: Path, label: str):
    print(f"  loading {label}: {npz_path.name}")
    d = np.load(npz_path, allow_pickle=True)
    text_proj = d["text_proj"]
    sids = d["sample_id"].astype(str)
    posture = d["posture_canonical"].astype(str)
    true_cls = np.array([_infer_true_class(s) for s in sids])
    print(f"    text_proj: {text_proj.shape}  posture: {dict(zip(*np.unique(posture, return_counts=True)))}")
    return text_proj, sids, posture, true_cls


def _run_tsne(emb: np.ndarray, label: str) -> np.ndarray:
    t0 = time.time()
    print(f"  [{label}] t-SNE perplexity={TSNE_PERPLEXITY} on {emb.shape[1]}D ...", end="", flush=True)
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        random_state=TSNE_RANDOM_STATE,
        init=TSNE_INIT,
        n_jobs=-1,
    )
    proj = tsne.fit_transform(emb)
    print(f" done ({time.time() - t0:.1f}s)")
    return proj


def _scatter(ax, proj, labels, color_map, title):
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
    print("Step 4R-C clinical — TRAINED t-SNE v1 vs v2 (text_proj after contrastive)")
    print("=" * 64)
    for p in [V1_JOINT, V2_JOINT]:
        if not p.exists():
            raise FileNotFoundError(p)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[load]")
    v1_emb, v1_sid, v1_posture, v1_class = _load(V1_JOINT, "v1")
    v2_emb, v2_sid, v2_posture, v2_class = _load(V2_JOINT, "v2")

    print("\n[t-SNE]")
    proj_v1 = _run_tsne(v1_emb, "v1")
    proj_v2 = _run_tsne(v2_emb, "v2")

    print("\n[plot] by true_class")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _scatter(axes[0], proj_v1, v1_class, CLASS_COLORS, "v1 (POSTURE 1-fixed × JOINT 2-pool) — TRAINED")
    _scatter(axes[1], proj_v2, v2_class, CLASS_COLORS, "v2 (POSTURE 5/5/3-pool × JOINT 3~4-pool) — TRAINED")
    fig.suptitle(
        "Step 4R-C clinical — t-SNE of contrastive-TRAINED text projection (128D, color: true class)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(OUT_BY_CLASS, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {OUT_BY_CLASS}")

    print("\n[plot] by posture")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _scatter(axes[0], proj_v1, v1_posture, POSTURE_COLORS, "v1 (POSTURE 1-fixed × JOINT 2-pool) — TRAINED")
    _scatter(axes[1], proj_v2, v2_posture, POSTURE_COLORS, "v2 (POSTURE 5/5/3-pool × JOINT 3~4-pool) — TRAINED")
    fig.suptitle(
        "Step 4R-C clinical — t-SNE of contrastive-TRAINED text projection (128D, color: posture)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(OUT_BY_POSTURE, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {OUT_BY_POSTURE}")

    print()
    print("=" * 64)
    print("Done.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
