"""Step 4R-C clinical — Retrieval evaluation + t-SNE on clinical projection.

Mirror of scripts/evaluate_step4rc_retrieval.py (v1). Same metric definitions
and t-SNE config; only input/output paths differ. v1 outputs are preserved.

Reads (read-only):
    data/step4r/4rc_contrastive_optional/joint_embeddings_clinical.npz
    data/step4r/4rc_contrastive_optional/text_corpus_clinical.csv

Writes (new files only):
    data/step4r/4rc_contrastive_optional/retrieval_metrics_clinical.csv
    reports/step4r/4rc_contrastive_optional/retrieval_breakdown_clinical.csv
    reports/step4r/4rc_contrastive_optional/tsne_by_class_clinical.png
    reports/step4r/4rc_contrastive_optional/tsne_by_ambiguity_group_clinical.png
    reports/step4r/4rc_contrastive_optional/results_final_clinical.md

Run:
    & C:\\Users\\user\\anaconda3\\envs\\dl_env\\python.exe -X utf8 \
        scripts/evaluate_step4rc_retrieval_clinical.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_JOINT = PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional" / "joint_embeddings_clinical.npz"
INPUT_CORPUS = PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional" / "text_corpus_clinical.csv"

OUTPUT_DATA_DIR = PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "reports" / "step4r" / "4rc_contrastive_optional"

OUTPUT_METRICS_CSV = OUTPUT_DATA_DIR / "retrieval_metrics_clinical.csv"
OUTPUT_BREAKDOWN_CSV = OUTPUT_REPORT_DIR / "retrieval_breakdown_clinical.csv"
OUTPUT_TSNE_CLASS = OUTPUT_REPORT_DIR / "tsne_by_class_clinical.png"
OUTPUT_TSNE_AMBIG = OUTPUT_REPORT_DIR / "tsne_by_ambiguity_group_clinical.png"
OUTPUT_FINAL_MD = OUTPUT_REPORT_DIR / "results_final_clinical.md"

CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
POSTURES = ["SA", "CA", "HW"]
AMBIG_GROUPS = ["confident_C2", "within_group_c1_c5_c6", "pair_c3_c4",
                "pair_plus_c2_absorption", "no_call", "uncategorized"]
K_LIST = (1, 5, 10)


def compute_metrics(imu_proj, text_proj, template_keys, classes, ambig_groups, k_list=K_LIST):
    n = imu_proj.shape[0]
    if n == 0:
        return {}
    sim = imu_proj @ text_proj.T
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
        out[f"strict_i2t_R@{k}"] = float((topk_i2t == diag[:, None]).any(axis=1).mean())
        out[f"strict_t2i_R@{k}"] = float((topk_t2i == diag[:, None]).any(axis=1).mean())
    return out


def _flatten_metrics(d, split):
    rows = []
    for metric, value in d.items():
        parts = metric.split("_")
        target = parts[0]
        direction = parts[1]
        k = int(metric.split("@")[1])
        rows.append({"split": split, "target_type": target, "direction": direction, "k": k, "value": value})
    return rows


def _per_posture_breakdown(imu_proj, text_proj, template_keys, classes, ambig_groups, postures, k_list=K_LIST):
    rows = []
    for p in POSTURES + ["ALL"]:
        mask = np.ones(len(postures), dtype=bool) if p == "ALL" else (postures == p)
        if mask.sum() == 0:
            continue
        m = compute_metrics(imu_proj[mask], text_proj[mask], template_keys[mask], classes[mask], ambig_groups[mask], k_list=k_list)
        for k in k_list:
            rows.append({"posture": p, "n": int(mask.sum()), "k": k,
                         "template_i2t_R": m[f"template_i2t_R@{k}"],
                         "class_i2t_R": m[f"class_i2t_R@{k}"],
                         "ambig_i2t_R": m[f"ambig_i2t_R@{k}"],
                         "strict_i2t_R": m[f"strict_i2t_R@{k}"]})
    return rows


def _per_class_breakdown(imu_proj, text_proj, template_keys, classes, ambig_groups, k_list=(1, 5)):
    rows = []
    sim = imu_proj @ text_proj.T
    sort_i2t = np.argsort(-sim, axis=1)
    for c in CLASSES:
        mask = classes == c
        if mask.sum() == 0:
            continue
        for k in k_list:
            topk = sort_i2t[mask, :k]
            rows.append({
                "class": c, "n": int(mask.sum()), "k": k,
                "class_i2t_R": float((classes[topk] == c).any(axis=1).mean()),
                "ambig_i2t_R": float((ambig_groups[topk] == ambig_groups[mask, None]).any(axis=1).mean()),
                "template_i2t_R": float((template_keys[topk] == template_keys[mask, None]).any(axis=1).mean()),
            })
    return rows


def _plot_tsne(imu_emb, text_emb, color_arr, title, out_path, label_order, label_name):
    n = imu_emb.shape[0]
    stacked = np.vstack([imu_emb, text_emb])
    origin = np.array(["IMU"] * n + ["text"] * n)
    color_full = np.concatenate([color_arr, color_arr])
    print(f"  fitting t-SNE on {len(stacked)} points ...")
    t0 = time.time()
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, init="pca", learning_rate="auto")
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
            ax.scatter(coords[sub_mask, 0], coords[sub_mask, 1], s=8, alpha=0.5,
                       color=label_to_color[lbl], label=str(lbl))
        ax.set_title(f"{modality} side (test split) | colored by {label_name}")
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.legend(loc="best", fontsize=8, markerscale=1.5)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


def main() -> int:
    print("=" * 64)
    print("Step 4R-C clinical — retrieval evaluation + t-SNE")
    print("=" * 64)
    if not INPUT_JOINT.exists(): raise FileNotFoundError(INPUT_JOINT)
    if not INPUT_CORPUS.exists(): raise FileNotFoundError(INPUT_CORPUS)
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
    print(f"corpus unique phrases: {corpus['phrase'].nunique()}")

    print("\n=== per-split retrieval metrics ===")
    all_rows = []
    per_split_metrics = {}
    for s in ("train", "val", "test"):
        mask = split == s
        m = compute_metrics(imu_proj[mask], text_proj[mask], template_keys[mask], class_arr[mask], ambig_arr[mask])
        per_split_metrics[s] = m
        n = int(mask.sum())
        print(f"  {s:5s}  (n={n})")
        for k in K_LIST:
            print(f"    R@{k:>2} i2t  strict={m[f'strict_i2t_R@{k}']:.4f}  template={m[f'template_i2t_R@{k}']:.4f}  ambig={m[f'ambig_i2t_R@{k}']:.4f}  class={m[f'class_i2t_R@{k}']:.4f}")
        all_rows.extend(_flatten_metrics(m, s))

    pd.DataFrame(all_rows).to_csv(OUTPUT_METRICS_CSV, index=False, encoding="utf-8-sig")
    print(f"\nsaved metrics -> {OUTPUT_METRICS_CSV}")

    test_mask = split == "test"
    print("\n=== test split breakdown ===")
    posture_rows = _per_posture_breakdown(
        imu_proj[test_mask], text_proj[test_mask], template_keys[test_mask],
        class_arr[test_mask], ambig_arr[test_mask], posture[test_mask], k_list=(1, 5),
    )
    print("\n  by posture (k=1):")
    for r in posture_rows:
        if r["k"] == 1:
            print(f"    {r['posture']:3s} (n={r['n']:4d})  template R@1={r['template_i2t_R']:.4f}  class R@1={r['class_i2t_R']:.4f}  ambig R@1={r['ambig_i2t_R']:.4f}")

    class_rows = _per_class_breakdown(
        imu_proj[test_mask], text_proj[test_mask], template_keys[test_mask],
        class_arr[test_mask], ambig_arr[test_mask], k_list=(1, 5),
    )
    print("\n  by class (test, k=1):")
    for r in class_rows:
        if r["k"] == 1:
            print(f"    {r['class']} (n={r['n']:4d})  class R@1={r['class_i2t_R']:.4f}  ambig R@1={r['ambig_i2t_R']:.4f}  template R@1={r['template_i2t_R']:.4f}")

    breakdown_df = pd.DataFrame(
        [{"breakdown": "posture", **r} for r in posture_rows]
        + [{"breakdown": "class", **r} for r in class_rows]
    )
    breakdown_df.to_csv(OUTPUT_BREAKDOWN_CSV, index=False, encoding="utf-8-sig")
    print(f"\nsaved breakdown -> {OUTPUT_BREAKDOWN_CSV}")

    print("\n=== t-SNE on test split ===")
    _plot_tsne(imu_proj[test_mask], text_proj[test_mask], class_arr[test_mask],
               "4R-C clinical joint embedding — test split, colored by true class",
               OUTPUT_TSNE_CLASS, CLASSES, "class")
    _plot_tsne(imu_proj[test_mask], text_proj[test_mask], ambig_arr[test_mask],
               "4R-C clinical joint embedding — test split, colored by ambiguity_group",
               OUTPUT_TSNE_AMBIG, AMBIG_GROUPS, "ambiguity_group")

    L = []
    L.append("# Step 4R-C clinical — Final Retrieval Results")
    L.append("")
    L.append("- 생성 스크립트: `scripts/evaluate_step4rc_retrieval_clinical.py`")
    L.append(f"- 입력 corpus: text_corpus_clinical.csv (unique phrases: {corpus['phrase'].nunique()})")
    L.append("- text encoder: `jhgan/ko-sroberta-multitask` (frozen, 768d)")
    L.append("- IMU backbone: 4R-B b01.5/seed42 (frozen, attn_vec 128d)")
    L.append("- projection head: 128→128 + 768→128, L2 normalized, params 147,968")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 1. Overall retrieval — per-split (k=1)")
    L.append("")
    L.append("| split | n | strict R@1 | template R@1 | ambig R@1 | class R@1 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for s in ("train", "val", "test"):
        m = per_split_metrics[s]
        n = int((split == s).sum())
        L.append(f"| {s} | {n} | {m['strict_i2t_R@1']:.4f} | {m['template_i2t_R@1']:.4f} | {m['ambig_i2t_R@1']:.4f} | {m['class_i2t_R@1']:.4f} |")
    L.append("")
    L.append("Chance baselines (test):")
    L.append(f"  - strict R@1     ≈ 1/{int(test_mask.sum())} = {1.0/int(test_mask.sum()):.4f}")
    L.append(f"  - template R@1   ≈ 1/{len(np.unique(template_keys[test_mask]))} = {1.0/len(np.unique(template_keys[test_mask])):.4f}")
    L.append(f"  - ambig R@1      ≈ 1/{len(np.unique(ambig_arr[test_mask]))} = {1.0/len(np.unique(ambig_arr[test_mask])):.4f}")
    L.append(f"  - class R@1      ≈ 1/6 = {1.0/6:.4f}")
    L.append("")
    L.append("## 2. R@5, R@10 (test split)")
    L.append("")
    L.append("| target | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    m_test = per_split_metrics["test"]
    for tgt in ("strict", "template", "ambig", "class"):
        L.append(f"| {tgt} | {m_test[f'{tgt}_i2t_R@1']:.4f} | {m_test[f'{tgt}_i2t_R@5']:.4f} | {m_test[f'{tgt}_i2t_R@10']:.4f} | "
                 f"{m_test[f'{tgt}_t2i_R@1']:.4f} | {m_test[f'{tgt}_t2i_R@5']:.4f} | {m_test[f'{tgt}_t2i_R@10']:.4f} |")
    L.append("")
    L.append("## 3. Per-posture breakdown (test, k=1)")
    L.append("")
    L.append("| posture | n | template R@1 | class R@1 | ambig R@1 | strict R@1 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r in posture_rows:
        if r["k"] == 1:
            L.append(f"| {r['posture']} | {r['n']} | {r['template_i2t_R']:.4f} | {r['class_i2t_R']:.4f} | {r['ambig_i2t_R']:.4f} | {r['strict_i2t_R']:.4f} |")
    L.append("")
    L.append("## 4. Per-class breakdown (test, k=1)")
    L.append("")
    L.append("| class | n | class R@1 | ambig R@1 | template R@1 |")
    L.append("|---|---:|---:|---:|---:|")
    for r in class_rows:
        if r["k"] == 1:
            L.append(f"| {r['class']} | {r['n']} | {r['class_i2t_R']:.4f} | {r['ambig_i2t_R']:.4f} | {r['template_i2t_R']:.4f} |")
    L.append("")
    L.append("## 5. t-SNE 시각화")
    L.append("")
    L.append("- `tsne_by_class_clinical.png` (6-class)")
    L.append("- `tsne_by_ambiguity_group_clinical.png`")
    L.append("")
    L.append("## 6. 비고")
    L.append("")
    L.append("Clinical corpus는 v1 대비 unique phrase 수가 47배 증가 (37→1733)했고 "
             "직접적인 임상 어휘(knee valgus / posterior tilting / left/right/bilateral)를 "
             "포함. v1과의 정량 비교는 `clinical_vs_abstract_comparison.md` 참조.")
    OUTPUT_FINAL_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"saved final report -> {OUTPUT_FINAL_MD}")

    print("\n" + "=" * 64)
    print("Done.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
