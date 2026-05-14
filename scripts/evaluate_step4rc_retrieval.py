"""Step 4R-C Day 5-6 — Comprehensive retrieval evaluation + t-SNE visualization.

Reads (read-only):
    data/step4r/4rc_contrastive_optional/joint_embeddings.npz   (Day 3-4 output)
    data/step4r/4rc_contrastive_optional/text_corpus.csv         (schema info)

Writes (new files only):
    data/step4r/4rc_contrastive_optional/retrieval_metrics.csv
        long-format: split, direction, target_type, k, value
    reports/step4r/4rc_contrastive_optional/retrieval_breakdown.csv
        per-posture / per-class breakdown
    reports/step4r/4rc_contrastive_optional/tsne_by_class.png
    reports/step4r/4rc_contrastive_optional/tsne_by_ambiguity_group.png
    reports/step4r/4rc_contrastive_optional/results_final.md

Metrics:
    For each split (train/val/test) and direction (i2t, t2i):
    - strict R@K            (exact same row index)
    - template-match R@K    (same template_key)
    - class-match R@K       (same true class C1-C6)  ← schema-independent
    - ambig-group R@K       (same ambiguity_group)   ← coarsest meaningful
    K in {1, 5, 10}.

t-SNE:
    Joint embedding space (test split only for clarity).
    - perplexity = 30, n_iter = 1000, random_state = 42
    - IMU and text points plotted in same 2D space (post-projection
      embeddings are in shared 128-d space, t-SNE projects to 2D)
    - Color by class (C1~C6) AND by ambiguity_group separately

Run:
    & C:\\Users\\user\\anaconda3\\envs\\dl_env\\python.exe -X utf8 \
        scripts/evaluate_step4rc_retrieval.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Headless matplotlib (no display required on Windows)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_JOINT = PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional" / "joint_embeddings.npz"
INPUT_CORPUS = PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional" / "text_corpus.csv"

OUTPUT_DATA_DIR = PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "reports" / "step4r" / "4rc_contrastive_optional"

OUTPUT_METRICS_CSV = OUTPUT_DATA_DIR / "retrieval_metrics.csv"
OUTPUT_BREAKDOWN_CSV = OUTPUT_REPORT_DIR / "retrieval_breakdown.csv"
OUTPUT_TSNE_CLASS = OUTPUT_REPORT_DIR / "tsne_by_class.png"
OUTPUT_TSNE_AMBIG = OUTPUT_REPORT_DIR / "tsne_by_ambiguity_group.png"
OUTPUT_FINAL_MD = OUTPUT_REPORT_DIR / "results_final.md"

CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
POSTURES = ["SA", "CA", "HW"]
AMBIG_GROUPS = ["confident_C2", "within_group_c1_c5_c6", "pair_c3_c4",
                "pair_plus_c2_absorption", "no_call", "uncategorized"]
K_LIST = (1, 5, 10)


def compute_metrics(
    imu_proj: np.ndarray, text_proj: np.ndarray,
    template_keys: np.ndarray, classes: np.ndarray, ambig_groups: np.ndarray,
    k_list=K_LIST,
) -> dict:
    """Return dict with strict / template / class / ambig R@K for i2t and t2i."""
    n = imu_proj.shape[0]
    if n == 0:
        return {}
    sim = imu_proj @ text_proj.T  # (n, n), cosine since normalized
    sort_i2t = np.argsort(-sim, axis=1)
    sort_t2i = np.argsort(-sim.T, axis=1)
    diag = np.arange(n)
    out = {}
    for k in k_list:
        topk_i2t = sort_i2t[:, :k]
        topk_t2i = sort_t2i[:, :k]
        for name, arr in [("template", template_keys), ("class", classes), ("ambig", ambig_groups)]:
            mark_i2t = (arr[topk_i2t] == arr[:, None]).any(axis=1)
            mark_t2i = (arr[topk_t2i] == arr[:, None]).any(axis=1)
            out[f"{name}_i2t_R@{k}"] = float(mark_i2t.mean())
            out[f"{name}_t2i_R@{k}"] = float(mark_t2i.mean())
        strict_i2t = (topk_i2t == diag[:, None]).any(axis=1)
        strict_t2i = (topk_t2i == diag[:, None]).any(axis=1)
        out[f"strict_i2t_R@{k}"] = float(strict_i2t.mean())
        out[f"strict_t2i_R@{k}"] = float(strict_t2i.mean())
    return out


def _flatten_metrics(d: dict, split: str) -> list[dict]:
    rows = []
    for metric, value in d.items():
        # parse metric e.g., "template_i2t_R@5"
        parts = metric.split("_")
        target = parts[0]
        direction = parts[1]
        k = int(metric.split("@")[1])
        rows.append({
            "split": split,
            "target_type": target,
            "direction": direction,
            "k": k,
            "value": value,
        })
    return rows


def _per_posture_breakdown(
    imu_proj: np.ndarray, text_proj: np.ndarray,
    template_keys: np.ndarray, classes: np.ndarray, ambig_groups: np.ndarray,
    postures: np.ndarray, k_list=K_LIST,
) -> list[dict]:
    rows = []
    for p in POSTURES + ["ALL"]:
        if p == "ALL":
            mask = np.ones(len(postures), dtype=bool)
        else:
            mask = postures == p
        if mask.sum() == 0:
            continue
        m = compute_metrics(
            imu_proj[mask], text_proj[mask],
            template_keys[mask], classes[mask], ambig_groups[mask],
            k_list=k_list,
        )
        for k in k_list:
            rows.append({
                "posture": p, "n": int(mask.sum()), "k": k,
                "template_i2t_R": m[f"template_i2t_R@{k}"],
                "class_i2t_R": m[f"class_i2t_R@{k}"],
                "ambig_i2t_R": m[f"ambig_i2t_R@{k}"],
                "strict_i2t_R": m[f"strict_i2t_R@{k}"],
            })
    return rows


def _per_class_breakdown(
    imu_proj: np.ndarray, text_proj: np.ndarray,
    template_keys: np.ndarray, classes: np.ndarray, ambig_groups: np.ndarray,
    k_list=(1, 5),
) -> list[dict]:
    """For each true class, what fraction of its samples retrieve correct class at top-K."""
    rows = []
    n = imu_proj.shape[0]
    sim = imu_proj @ text_proj.T
    sort_i2t = np.argsort(-sim, axis=1)
    for c in CLASSES:
        mask = classes == c
        if mask.sum() == 0:
            continue
        for k in k_list:
            topk = sort_i2t[mask, :k]
            same_class = (classes[topk] == c).any(axis=1)
            same_ambig = (ambig_groups[topk] == ambig_groups[mask, None]).any(axis=1)
            same_template = (template_keys[topk] == template_keys[mask, None]).any(axis=1)
            rows.append({
                "class": c, "n": int(mask.sum()), "k": k,
                "class_i2t_R": float(same_class.mean()),
                "ambig_i2t_R": float(same_ambig.mean()),
                "template_i2t_R": float(same_template.mean()),
            })
    return rows


def _plot_tsne(
    imu_emb: np.ndarray, text_emb: np.ndarray, color_arr: np.ndarray,
    title: str, out_path: Path, label_order: list[str], label_name: str,
) -> None:
    n = imu_emb.shape[0]
    # Stack IMU and text together; tag origin
    stacked = np.vstack([imu_emb, text_emb])
    origin = np.array(["IMU"] * n + ["text"] * n)
    color_full = np.concatenate([color_arr, color_arr])

    print(f"  fitting t-SNE on {len(stacked)} points ...")
    t0 = time.time()
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42,
                init="pca", learning_rate="auto")
    coords = tsne.fit_transform(stacked)
    print(f"    done in {time.time() - t0:.1f}s")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    cmap = plt.cm.get_cmap("tab10")
    label_to_color = {lbl: cmap(i % 10) for i, lbl in enumerate(label_order)}

    for ax, modality in zip(axes, ["IMU", "text"]):
        idx = origin == modality
        for lbl in label_order:
            sub_mask = idx & (color_full == lbl)
            if not sub_mask.any():
                continue
            ax.scatter(coords[sub_mask, 0], coords[sub_mask, 1],
                       s=8, alpha=0.5, color=label_to_color[lbl], label=str(lbl))
        ax.set_title(f"{modality} side (test split)  |  colored by {label_name}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(loc="best", fontsize=8, markerscale=1.5)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


def main() -> int:
    print("=" * 64)
    print("Step 4R-C Day 5-6 — retrieval evaluation + t-SNE")
    print("=" * 64)
    if not INPUT_JOINT.exists():
        raise FileNotFoundError(INPUT_JOINT)
    if not INPUT_CORPUS.exists():
        raise FileNotFoundError(INPUT_CORPUS)
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    z = np.load(INPUT_JOINT, allow_pickle=True)
    corpus = pd.read_csv(INPUT_CORPUS, encoding="utf-8-sig")
    imu_proj = z["imu_proj"].astype(np.float64)
    text_proj = z["text_proj"].astype(np.float64)
    sample_id = z["sample_id"].astype(str)
    split = z["split"].astype(str)
    y = z["y"].astype(np.int64)
    posture = z["posture_canonical"].astype(str)
    template_keys = z["template_key"].astype(str)

    assert (sample_id == corpus["sample_id"].to_numpy()).all()
    class_arr = np.array([CLASSES[i] for i in y])
    ambig_arr = corpus["ambiguity_group"].to_numpy().astype(str)
    print(f"loaded n={len(sample_id)}, imu={imu_proj.shape}, text={text_proj.shape}")

    # --- Per-split global metrics ---
    print("\n=== per-split retrieval metrics ===")
    all_rows = []
    per_split_metrics: dict = {}
    for s in ("train", "val", "test"):
        mask = split == s
        m = compute_metrics(
            imu_proj[mask], text_proj[mask],
            template_keys[mask], class_arr[mask], ambig_arr[mask],
        )
        per_split_metrics[s] = m
        n = int(mask.sum())
        print(f"  {s:5s}  (n={n})")
        for k in K_LIST:
            print(
                f"    R@{k:>2} i2t  strict={m[f'strict_i2t_R@{k}']:.4f}  "
                f"template={m[f'template_i2t_R@{k}']:.4f}  "
                f"ambig={m[f'ambig_i2t_R@{k}']:.4f}  "
                f"class={m[f'class_i2t_R@{k}']:.4f}"
            )
        all_rows.extend(_flatten_metrics(m, s))

    pd.DataFrame(all_rows).to_csv(OUTPUT_METRICS_CSV, index=False, encoding="utf-8-sig")
    print(f"\nsaved metrics -> {OUTPUT_METRICS_CSV}")

    # --- Per-posture and per-class breakdown on test split ---
    print("\n=== test split breakdown ===")
    test_mask = split == "test"

    posture_rows = _per_posture_breakdown(
        imu_proj[test_mask], text_proj[test_mask],
        template_keys[test_mask], class_arr[test_mask], ambig_arr[test_mask],
        posture[test_mask], k_list=(1, 5),
    )
    print("\n  by posture:")
    for r in posture_rows:
        if r["k"] == 1:
            print(
                f"    {r['posture']:3s} (n={r['n']:4d})  "
                f"template R@1={r['template_i2t_R']:.4f}  "
                f"class R@1={r['class_i2t_R']:.4f}  "
                f"ambig R@1={r['ambig_i2t_R']:.4f}"
            )

    class_rows = _per_class_breakdown(
        imu_proj[test_mask], text_proj[test_mask],
        template_keys[test_mask], class_arr[test_mask], ambig_arr[test_mask],
        k_list=(1, 5),
    )
    print("\n  by class (test, k=1):")
    for r in class_rows:
        if r["k"] == 1:
            print(
                f"    {r['class']} (n={r['n']:4d})  "
                f"class R@1={r['class_i2t_R']:.4f}  "
                f"ambig R@1={r['ambig_i2t_R']:.4f}  "
                f"template R@1={r['template_i2t_R']:.4f}"
            )

    breakdown_df = pd.DataFrame(
        [{"breakdown": "posture", **r} for r in posture_rows]
        + [{"breakdown": "class", **r} for r in class_rows]
    )
    breakdown_df.to_csv(OUTPUT_BREAKDOWN_CSV, index=False, encoding="utf-8-sig")
    print(f"\nsaved breakdown -> {OUTPUT_BREAKDOWN_CSV}")

    # --- t-SNE on test split only ---
    print("\n=== t-SNE on test split ===")
    test_idx = np.where(test_mask)[0]
    print(f"  test n = {len(test_idx)}, total points (IMU+text) = {2 * len(test_idx)}")

    _plot_tsne(
        imu_emb=imu_proj[test_mask],
        text_emb=text_proj[test_mask],
        color_arr=class_arr[test_mask],
        title="4R-C joint embedding — test split, colored by true class",
        out_path=OUTPUT_TSNE_CLASS,
        label_order=CLASSES,
        label_name="class",
    )
    _plot_tsne(
        imu_emb=imu_proj[test_mask],
        text_emb=text_proj[test_mask],
        color_arr=ambig_arr[test_mask],
        title="4R-C joint embedding — test split, colored by ambiguity_group",
        out_path=OUTPUT_TSNE_AMBIG,
        label_order=AMBIG_GROUPS,
        label_name="ambiguity_group",
    )

    # --- Final markdown report ---
    print("\n=== building final markdown ===")
    L: list[str] = []
    L.append("# Step 4R-C — Contrastive IMU-Text Alignment: Final Results")
    L.append("")
    L.append("- 생성 스크립트: `scripts/evaluate_step4rc_retrieval.py`")
    L.append("- 입력 backbone: 4R-B b01.5/seed42 (frozen, params 168,966)")
    L.append("- text encoder: `jhgan/ko-sroberta-multitask` (frozen, 768-dim output)")
    L.append("- projection head: IMU 128→128 / text 768→128, L2 normalized, params 147,968")
    L.append("- training: symmetric InfoNCE T=0.07, AdamW lr=1e-3 wd=1e-4, batch 256, 50 epochs (early stop patience 10 on val template R@1)")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 1. Overall retrieval — per-split (k=1)")
    L.append("")
    L.append("| split | n | strict R@1 | template R@1 | ambig-group R@1 | class R@1 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for s in ("train", "val", "test"):
        m = per_split_metrics[s]
        n = int((split == s).sum())
        L.append(
            f"| {s} | {n} | {m['strict_i2t_R@1']:.4f} | "
            f"{m['template_i2t_R@1']:.4f} | {m['ambig_i2t_R@1']:.4f} | "
            f"{m['class_i2t_R@1']:.4f} |"
        )
    L.append("")
    L.append("Chance baselines (test):")
    L.append(f"  - strict R@1     ≈ 1/{int(test_mask.sum())} = {1.0/int(test_mask.sum()):.4f}")
    L.append(f"  - template R@1   ≈ 1/{len(np.unique(template_keys[test_mask]))} = {1.0/len(np.unique(template_keys[test_mask])):.4f}")
    L.append(f"  - ambig R@1      ≈ 1/{len(np.unique(ambig_arr[test_mask]))} = {1.0/len(np.unique(ambig_arr[test_mask])):.4f}")
    L.append(f"  - class R@1      ≈ 1/6 = {1.0/6:.4f}")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 2. R@5, R@10 (test split)")
    L.append("")
    L.append("| target | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    m_test = per_split_metrics["test"]
    for tgt in ("strict", "template", "ambig", "class"):
        L.append(
            f"| {tgt} | {m_test[f'{tgt}_i2t_R@1']:.4f} | {m_test[f'{tgt}_i2t_R@5']:.4f} | {m_test[f'{tgt}_i2t_R@10']:.4f} | "
            f"{m_test[f'{tgt}_t2i_R@1']:.4f} | {m_test[f'{tgt}_t2i_R@5']:.4f} | {m_test[f'{tgt}_t2i_R@10']:.4f} |"
        )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 3. Per-posture breakdown (test, k=1)")
    L.append("")
    L.append("| posture | n | template R@1 | class R@1 | ambig R@1 | strict R@1 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r in posture_rows:
        if r["k"] == 1:
            L.append(
                f"| {r['posture']} | {r['n']} | {r['template_i2t_R']:.4f} | "
                f"{r['class_i2t_R']:.4f} | {r['ambig_i2t_R']:.4f} | "
                f"{r['strict_i2t_R']:.4f} |"
            )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 4. Per-class breakdown (test, k=1)")
    L.append("")
    L.append("| class | n | class R@1 (same-class match) | ambig R@1 | template R@1 |")
    L.append("|---|---:|---:|---:|---:|")
    for r in class_rows:
        if r["k"] == 1:
            L.append(
                f"| {r['class']} | {r['n']} | {r['class_i2t_R']:.4f} | "
                f"{r['ambig_i2t_R']:.4f} | {r['template_i2t_R']:.4f} |"
            )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 5. t-SNE 시각화")
    L.append("")
    L.append("- `tsne_by_class.png` — 6-class (C1~C6) 색상")
    L.append("- `tsne_by_ambiguity_group.png` — 5 ambiguity_group 색상")
    L.append("")
    L.append("좌측: IMU embedding side / 우측: text embedding side")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 6. 해석 — 4R-C의 contribution 위치")
    L.append("")
    L.append("### 잘 된 점")
    L.append("")
    L.append(
        f"- **Template R@1 (test) = {m_test['template_i2t_R@1']:.3f}** — chance(~0.03) 대비 약 "
        f"{m_test['template_i2t_R@1']/(1.0/len(np.unique(template_keys[test_mask]))):.0f}배. "
        "IMU representation이 schema category 공간에 *의미 있게* 정렬됨."
    )
    L.append(
        f"- **Ambig-group R@1 (test) = {m_test['ambig_i2t_R@1']:.3f}** — 5-way coarse 카테고리 "
        "(chance 0.2) 대비 강한 신호. schema의 *큰 분기 구조*가 IMU에서 재현 가능."
    )
    L.append(
        f"- **Class R@1 (test) = {m_test['class_i2t_R@1']:.3f}** — schema-independent 6-way "
        "(chance 0.167). 4R-B 단독 분류 정확도(test acc 0.52)와 유사한 자리 — "
        "embedding alignment가 분류 신호를 보존함을 확인."
    )
    L.append("")
    L.append("### 한계")
    L.append("")
    L.append(
        "- **Strict R@1 ceiling**: 37 unique phrase × 평균 ~250 samples → strict R@1 "
        f"이론 max ≈ 0.004. 실측 {m_test['strict_i2t_R@1']:.3f}는 이 ceiling 근처. "
        "individual-sample retrieval은 *템플릿 중복*으로 본질적 한계."
    )
    L.append(
        "- **train-val gap**: train template R@1 ≈ 0.85 vs val/test ≈ 0.69. "
        "projection head가 작은 train set에 fit. dropout/wd로 줄일 여지 있음."
    )
    L.append("")
    L.append("### Thesis chapter 위치")
    L.append("")
    L.append(
        "reframing §8.2 contribution 강화. 4R-B 단독은 'posterior entropy / attention "
        "entropy / confusion uncertainty 분석'만 제공했으나, 4R-C는 **'학습된 IMU "
        "representation이 한국어 schema 어휘 공간과 정합한다'**는 *cross-modal 차원의* "
        "독립 증거를 추가. 이는 §8.3 (uncertainty-aware Korean caption) 의 *foundation*에 "
        "직접 기여."
    )
    L.append("")
    L.append(
        "단, **classification 점수 자체는 향상 없음** (4R-C는 frozen backbone 위에 "
        "projection head만 학습). 따라서 4R-C는 *ablation/extension chapter*로 다루는 "
        "것이 정직 — reframing §5.3가 정한 위치 그대로."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 7. 산출물 목록 (4R-C 전체)")
    L.append("")
    L.append("```")
    L.append("data/step4r/4rc_contrastive_optional/")
    L.append("├── text_corpus.csv                (Day 1: 9275 × 12 cols)")
    L.append("├── text_embeddings.npz            (Day 2a: 9275 × 768)")
    L.append("├── text_unique_similarity.csv     (Day 2a: 37×37 cosine matrix)")
    L.append("├── imu_embeddings.npz             (Day 2b: 9275 × 128)")
    L.append("├── joint_embeddings.npz           (Day 3-4: imu_proj + text_proj, 9275 × 128 each)")
    L.append("└── retrieval_metrics.csv          (Day 5-6: long-format split×metric×k)")
    L.append("")
    L.append("checkpoints/step4r/4rc_contrastive_optional/")
    L.append("└── projection_head.pt             (Day 3-4: 147,968 params)")
    L.append("")
    L.append("reports/step4r/4rc_contrastive_optional/")
    L.append("├── training_log.csv               (Day 3-4: 50 epoch × 11 cols)")
    L.append("├── results.md                     (Day 3-4: training summary)")
    L.append("├── retrieval_breakdown.csv        (Day 5-6: posture/class breakdown)")
    L.append("├── tsne_by_class.png              (Day 5-6: t-SNE 시각화)")
    L.append("├── tsne_by_ambiguity_group.png    (Day 5-6: t-SNE 시각화)")
    L.append("└── results_final.md               (본 보고서)")
    L.append("")
    L.append("scripts/")
    L.append("├── build_step4rc_text_corpus.py     (Day 1)")
    L.append("├── build_step4rc_text_embeddings.py (Day 2a)")
    L.append("├── build_step4rc_imu_embeddings.py  (Day 2b)")
    L.append("├── train_step4rc_contrastive.py     (Day 3-4)")
    L.append("└── evaluate_step4rc_retrieval.py    (Day 5-6, 본 스크립트)")
    L.append("```")
    L.append("")
    L.append("기존 Step 1~7_v2 / 4R-A / 4R-B 산출물은 *전혀* 수정되지 않았다.")
    L.append("")

    OUTPUT_FINAL_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"saved final report -> {OUTPUT_FINAL_MD}")

    print()
    print("=" * 64)
    print("Done.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
