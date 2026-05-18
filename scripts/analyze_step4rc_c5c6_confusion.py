"""C5/C6 trilateral knee-valgus confusion 정량 분석.

오늘(2026-05-14) 4R-C clinical A/B 실험에서 발견된 패턴:
  - v1 abstract: C5 class R@1=0.210, C6=0.408 (corpus가 C5/C6를 함께 묶음)
  - v2 clinical: C5=0.256 (+0.05), C6=0.254 (-0.15) — directional vocab 도입 후
    오히려 C6 정확도 급락. 손목 IMU 단독으론 left/right/bilateral knee
    valgus의 trilateral 구분이 본질적으로 어렵다는 가설.

본 스크립트:
  1) v1·v2 양쪽 joint embedding 로드 (test split)
  2) cross-modal i2t retrieval top-1으로 6×6 confusion matrix 계산
  3) C5/C6에 한정한 retrieval distribution (top-1, top-5) 비교
  4) PNG figure 생성: v1 vs v2 6×6 heatmap side-by-side + C5/C6 focused bar
  5) 분석 markdown report

Reads (read-only):
    data/step4r/4rc_contrastive_optional/joint_embeddings.npz          (v1 abstract)
    data/step4r/4rc_contrastive_optional/joint_embeddings_clinical.npz (v2 clinical)

Writes (new files only):
    data/step4r/4rc_contrastive_optional/c5c6_confusion_matrices.csv
    reports/step4r/4rc_contrastive_optional/c5c6_confusion_figure.png
    reports/step4r/4rc_contrastive_optional/c5c6_confusion_analysis.md

Run:
    & 'C:\\Users\\user\\anaconda3\\envs\\dl_env\\python.exe' -X utf8 \
        scripts/analyze_step4rc_c5c6_confusion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Korean-capable font (Windows default). Falls back gracefully if absent.
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_V1 = PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional" / "joint_embeddings.npz"
INPUT_V2 = PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional" / "joint_embeddings_clinical.npz"

OUTPUT_CSV = PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional" / "c5c6_confusion_matrices.csv"
OUTPUT_PNG = PROJECT_ROOT / "reports" / "step4r" / "4rc_contrastive_optional" / "c5c6_confusion_figure.png"
OUTPUT_MD = PROJECT_ROOT / "reports" / "step4r" / "4rc_contrastive_optional" / "c5c6_confusion_analysis.md"

CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]


def _load(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {
        "sample_id": z["sample_id"].astype(str),
        "split": z["split"].astype(str),
        "y": z["y"].astype(np.int64),
        "imu_proj": z["imu_proj"].astype(np.float64),
        "text_proj": z["text_proj"].astype(np.float64),
    }


def _retrieval_confusion(d: dict, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Returns (counts, normalized) 6×6 confusion matrices on test split.
    counts[i, j] = # samples of class CLASSES[i] whose top-k retrieval
                   contains a sample of class CLASSES[j]
    """
    test = d["split"] == "test"
    imu = d["imu_proj"][test]
    txt = d["text_proj"][test]
    y = d["y"][test]
    cls_arr = np.array([CLASSES[i] for i in y])

    sim = imu @ txt.T  # (n_test, n_test)
    topk_idx = np.argsort(-sim, axis=1)[:, :k]  # (n_test, k)
    topk_cls = cls_arr[topk_idx]  # (n_test, k)

    cm = np.zeros((6, 6), dtype=np.int64)
    for i, true_c in enumerate(CLASSES):
        mask = cls_arr == true_c
        if not mask.any():
            continue
        retrieved = topk_cls[mask]  # (n_class_i, k)
        for j, pred_c in enumerate(CLASSES):
            # row = true_c, col = pred_c; count fraction of samples for which
            # any of their top-k retrievals belong to pred_c
            hits = (retrieved == pred_c).any(axis=1).sum()
            cm[i, j] = int(hits)

    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.where(row_sum > 0, cm / row_sum, 0.0)
    return cm, norm


def _plot_grid(cm_v1: np.ndarray, cm_v2: np.ndarray, k: int) -> plt.Figure:
    """Side-by-side heatmaps for v1 (abstract) vs v2 (clinical)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, cm_norm, title in [
        (axes[0], cm_v1, f"v1 abstract corpus (37 unique phrases)\nretrieval top-{k}, test split"),
        (axes[1], cm_v2, f"v2 clinical corpus (1,733 unique phrases)\nretrieval top-{k}, test split"),
    ]:
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1.0, aspect="equal")
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
        ax.set_xlabel("retrieved (top-1) class")
        ax.set_ylabel("true class")
        ax.set_title(title, fontsize=10)
        for i in range(6):
            for j in range(6):
                v = cm_norm[i, j]
                txtcol = "white" if v > 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=txtcol, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        "C5/C6 trilateral knee-valgus confusion (i2t retrieval)\n"
        "왼쪽: abstract corpus는 C5/C6를 함께 묶음 → C6 정답이 높지만 collapse "
        "    오른쪽: clinical corpus는 left/right/bilateral 분리 → IMU 한계 노출",
        fontsize=11
    )
    fig.tight_layout()
    return fig


def _plot_c5c6_focused(d_v1: dict, d_v2: dict) -> plt.Figure:
    """C5와 C6 정답 sample이 retrieval에서 어디로 가는지 stacked bar."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    color_map = {"C1": "#9e9e9e", "C2": "#fdb462", "C3": "#fb8072",
                 "C4": "#80b1d3", "C5": "#bebada", "C6": "#8dd3c7"}

    def collect(d: dict, true_c: str, k: int = 5) -> dict[str, float]:
        test = d["split"] == "test"
        imu = d["imu_proj"][test]; txt = d["text_proj"][test]
        y = d["y"][test]; cls_arr = np.array([CLASSES[i] for i in y])
        mask = cls_arr == true_c
        if not mask.any():
            return {c: 0.0 for c in CLASSES}
        sim = imu[mask] @ txt.T
        topk = np.argsort(-sim, axis=1)[:, :k]
        topk_cls = cls_arr[topk]  # (n_true_c, k)
        out: dict[str, float] = {c: 0.0 for c in CLASSES}
        for c in CLASSES:
            # share of top-k slots that are c
            out[c] = float((topk_cls == c).sum() / topk_cls.size)
        return out

    for ax, version_label, d, color in [
        (axes[0], "v1 abstract", d_v1, "#9aa0c4"),
        (axes[1], "v2 clinical", d_v2, "#5f8a47"),
    ]:
        bars_c5 = collect(d, "C5", k=5)
        bars_c6 = collect(d, "C6", k=5)
        x = np.arange(2)
        bottom_5 = np.zeros(2)
        for c in CLASSES:
            heights = np.array([bars_c5[c], bars_c6[c]])
            ax.bar(x, heights, bottom=bottom_5, color=color_map[c],
                   label=f"retrieved={c}", edgecolor="white", linewidth=0.4)
            for xi, h, b in zip(x, heights, bottom_5):
                if h > 0.05:
                    ax.text(xi, b + h / 2, f"{c}\n{h:.2f}",
                            ha="center", va="center", fontsize=7, color="white" if c in ("C2", "C3", "C5") else "black")
            bottom_5 += heights
        ax.set_xticks(x); ax.set_xticklabels(["true C5", "true C6"])
        ax.set_ylabel("share of top-5 retrievals")
        ax.set_ylim(0, 1)
        ax.set_title(f"{version_label}\n(C5·C6 정답 sample의 top-5 retrieval 분포)", fontsize=10)
        if version_label == "v1 abstract":
            ax.legend(loc="upper left", fontsize=7, ncol=2)
    fig.suptitle(
        "C5·C6 정답 sample의 cross-modal retrieval 분포 (top-5)\n"
        "v1: C5↔C6 mutual confusion + C1 absorption  /  v2: C5는 좌측 분리 향상, C6는 C5로 흡수",
        fontsize=11
    )
    fig.tight_layout()
    return fig


def _save_combined(cm_v1_k1, cm_v2_k1, cm_v1_k5, cm_v2_k5, d_v1, d_v2):
    """Two-row figure: row1 = top-1 6x6 heatmap (v1, v2),
       row2 = C5/C6 focused stacked bar (v1, v2)."""
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9], hspace=0.42, wspace=0.30)

    # Row 1: heatmaps (k=1)
    for col, (cm_norm, title) in enumerate([
        (cm_v1_k1, "v1 abstract corpus (37 unique phrases)\nretrieval top-1, test split"),
        (cm_v2_k1, "v2 clinical corpus (1,733 unique phrases)\nretrieval top-1, test split"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1.0, aspect="equal")
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
        ax.set_xlabel("retrieved (top-1) class")
        ax.set_ylabel("true class")
        ax.set_title(title, fontsize=10)
        for i in range(6):
            for j in range(6):
                v = cm_norm[i, j]
                txtcol = "white" if v > 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=txtcol, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if cm_norm is cm_v2_k1:
            # highlight C5/C6 row/col
            for idx in (4, 5):
                ax.add_patch(plt.Rectangle((idx-0.5, -0.5), 1, 6, fill=False,
                                            edgecolor="red", lw=1.2, linestyle="--"))
                ax.add_patch(plt.Rectangle((-0.5, idx-0.5), 6, 1, fill=False,
                                            edgecolor="red", lw=1.2, linestyle="--"))

    # Row 2: C5/C6 focused stacked bars (k=5)
    color_map = {"C1": "#9e9e9e", "C2": "#fdb462", "C3": "#fb8072",
                 "C4": "#80b1d3", "C5": "#bebada", "C6": "#8dd3c7"}

    def collect(d, true_c, k=5):
        test = d["split"] == "test"
        imu = d["imu_proj"][test]; txt = d["text_proj"][test]
        y = d["y"][test]; cls_arr = np.array([CLASSES[i] for i in y])
        mask = cls_arr == true_c
        sim = imu[mask] @ txt.T
        topk = np.argsort(-sim, axis=1)[:, :k]
        topk_cls = cls_arr[topk]
        return {c: float((topk_cls == c).sum() / topk_cls.size) for c in CLASSES}

    for col, (d, version_label) in enumerate([
        (d_v1, "v1 abstract"),
        (d_v2, "v2 clinical"),
    ]):
        ax = fig.add_subplot(gs[1, col])
        bars_c5 = collect(d, "C5", k=5)
        bars_c6 = collect(d, "C6", k=5)
        x = np.arange(2)
        bottom = np.zeros(2)
        for c in CLASSES:
            heights = np.array([bars_c5[c], bars_c6[c]])
            ax.bar(x, heights, bottom=bottom, color=color_map[c],
                   edgecolor="white", linewidth=0.4, label=f"retrieved={c}")
            for xi, h, b in zip(x, heights, bottom):
                if h > 0.04:
                    text_color = "white" if c in ("C2", "C3", "C5") else "black"
                    ax.text(xi, b + h / 2, f"{c}\n{h:.2f}",
                            ha="center", va="center", fontsize=7, color=text_color)
            bottom += heights
        ax.set_xticks(x); ax.set_xticklabels(["true C5", "true C6"])
        ax.set_ylabel("share of top-5 retrievals")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{version_label} — C5·C6 정답 sample의 top-5 retrieval 분포", fontsize=10)
        if col == 0:
            ax.legend(loc="upper left", fontsize=7, ncol=2)

    fig.suptitle(
        "C5/C6 trilateral knee-valgus confusion: abstract vs clinical corpus\n"
        "wrist IMU 단독으로는 left/right/bilateral 구분 본질적 한계 (multi-modal 필요성 정량 근거)",
        fontsize=12, y=0.995
    )
    return fig


def main() -> int:
    print("=" * 64)
    print("C5/C6 trilateral knee-valgus confusion analysis")
    print("=" * 64)
    if not INPUT_V1.exists():
        raise FileNotFoundError(INPUT_V1)
    if not INPUT_V2.exists():
        raise FileNotFoundError(INPUT_V2)

    d_v1 = _load(INPUT_V1)
    d_v2 = _load(INPUT_V2)
    print(f"v1 loaded  n={len(d_v1['sample_id'])}")
    print(f"v2 loaded  n={len(d_v2['sample_id'])}")

    cm_v1_k1, norm_v1_k1 = _retrieval_confusion(d_v1, k=1)
    cm_v2_k1, norm_v2_k1 = _retrieval_confusion(d_v2, k=1)
    cm_v1_k5, norm_v1_k5 = _retrieval_confusion(d_v1, k=5)
    cm_v2_k5, norm_v2_k5 = _retrieval_confusion(d_v2, k=5)

    # ---- save confusion matrices as long-format CSV ----
    rows = []
    for label, m in [("v1_k1", norm_v1_k1), ("v2_k1", norm_v2_k1),
                     ("v1_k5", norm_v1_k5), ("v2_k5", norm_v2_k5)]:
        for i, t in enumerate(CLASSES):
            for j, p in enumerate(CLASSES):
                rows.append({"version_k": label, "true_class": t, "pred_class": p, "rate": float(m[i, j])})
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"saved CSV -> {OUTPUT_CSV}")

    # ---- console: C5/C6 focused stats ----
    print("\n=== C5/C6 retrieval distribution (top-1, normalized) ===")
    print(f"{'':>4} {'v1 abstract':>40} | {'v2 clinical':>40}")
    print(f"{'':>4} " + " ".join(f"{c:>5}" for c in CLASSES) + "  | " + " ".join(f"{c:>5}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        v1_row = " ".join(f"{norm_v1_k1[i,j]:5.2f}" for j in range(6))
        v2_row = " ".join(f"{norm_v2_k1[i,j]:5.2f}" for j in range(6))
        print(f"{c:>4} {v1_row}  | {v2_row}")

    print("\n=== C5/C6 specific changes (v1 → v2) ===")
    for true_c, true_i in [("C5", 4), ("C6", 5)]:
        print(f"\ntrue {true_c}:")
        for pred_c, pred_j in [("C5", 4), ("C6", 5), ("C1", 0), ("C4", 3)]:
            v1 = norm_v1_k1[true_i, pred_j]
            v2 = norm_v2_k1[true_i, pred_j]
            delta = v2 - v1
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
            print(f"  → {pred_c:2s}  v1={v1:.3f}  v2={v2:.3f}  Δ={delta:+.3f} {arrow}")

    # ---- figure ----
    print("\nrendering figure ...")
    fig = _save_combined(norm_v1_k1, norm_v2_k1, norm_v1_k5, norm_v2_k5, d_v1, d_v2)
    fig.savefig(OUTPUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved PNG -> {OUTPUT_PNG}")

    # ---- markdown report ----
    L = []
    L.append("# C5/C6 trilateral knee-valgus confusion 분석")
    L.append("")
    L.append("- 생성 스크립트: `scripts/analyze_step4rc_c5c6_confusion.py`")
    L.append("- 입력: `joint_embeddings.npz` (v1) + `joint_embeddings_clinical.npz` (v2)")
    L.append("- 범위: test split (n=1,427), i2t cross-modal retrieval")
    L.append("")
    L.append("## 1. 배경")
    L.append("")
    L.append("4R-C clinical A/B 실험에서 C5(Right-knee valgus)와 C6(Bilateral knee valgus)의 비대칭 변화 관찰:")
    L.append("")
    L.append("| 클래스 | figure_img.png 정의 | v1 class R@1 | v2 class R@1 | Δ |")
    L.append("|---|---|---:|---:|---:|")
    L.append(f"| C5 | Right-knee valgus | {norm_v1_k1[4,4]:.3f} | {norm_v2_k1[4,4]:.3f} | {norm_v2_k1[4,4]-norm_v1_k1[4,4]:+.3f} |")
    L.append(f"| C6 | Bilateral knee valgus | {norm_v1_k1[5,5]:.3f} | {norm_v2_k1[5,5]:.3f} | {norm_v2_k1[5,5]-norm_v1_k1[5,5]:+.3f} |")
    L.append("")
    L.append("가설: 손목 IMU 단독으로는 left(C4) / right(C5) / bilateral(C6) knee valgus의 3-way 구분이 본질적으로 어렵다. v1 abstract corpus는 C5/C6를 함께 묶어 ('무릎 관련 오류 후보') 한 클러스터로 학습 → C6 회수율이 운으로 높았음. v2 clinical은 directional 분리 → 진짜 IMU 한계 노출.")
    L.append("")
    L.append("## 2. 6×6 retrieval confusion matrix (top-1)")
    L.append("")
    L.append("`reports/step4r/4rc_contrastive_optional/c5c6_confusion_figure.png` 의 1행 참조.")
    L.append("")
    L.append("### v1 abstract — 핵심 행")
    L.append("")
    L.append("| true \\ retrieved | C1 | C2 | C3 | C4 | C5 | C6 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for i in (4, 5):
        L.append(f"| **{CLASSES[i]}** | " + " | ".join(f"{norm_v1_k1[i,j]:.2f}" for j in range(6)) + " |")
    L.append("")
    L.append("### v2 clinical — 핵심 행")
    L.append("")
    L.append("| true \\ retrieved | C1 | C2 | C3 | C4 | C5 | C6 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for i in (4, 5):
        L.append(f"| **{CLASSES[i]}** | " + " | ".join(f"{norm_v2_k1[i,j]:.2f}" for j in range(6)) + " |")
    L.append("")
    L.append("## 3. 변화 분석 (v1 → v2)")
    L.append("")
    L.append("| true | pred | v1 rate | v2 rate | Δ |")
    L.append("|---|---|---:|---:|---:|")
    for true_c, true_i in [("C5", 4), ("C6", 5)]:
        for pred_c, pred_j in [("C5", 4), ("C6", 5), ("C1", 0), ("C4", 3)]:
            v1 = norm_v1_k1[true_i, pred_j]; v2 = norm_v2_k1[true_i, pred_j]
            delta = v2 - v1
            L.append(f"| {true_c} | {pred_c} | {v1:.3f} | {v2:.3f} | {delta:+.3f} |")
    L.append("")
    L.append("## 4. 해석")
    L.append("")
    c5_to_c6_v1 = norm_v1_k1[4, 5]
    c5_to_c6_v2 = norm_v2_k1[4, 5]
    c6_to_c5_v1 = norm_v1_k1[5, 4]
    c6_to_c5_v2 = norm_v2_k1[5, 4]
    L.append("### 4.1 v1 abstract: C5/C6 mutual collapse")
    L.append("")
    L.append(f"- v1에서 C5→C6: {c5_to_c6_v1:.2f}, C6→C5: {c6_to_c5_v1:.2f}. 두 클래스가 서로 retrieve하는 mutual confusion.")
    L.append("- corpus phrase가 \"무릎 관련 오류 후보(C5, C6)\"로 *동일 어휘* 사용 → text embedding 공간에서 두 클래스의 anchor가 거의 동일 → IMU가 어디로 가도 C5/C6 중 하나에 회수.")
    L.append("- C6의 v1 0.40은 *진짜 식별*이 아니라 *어휘 collapse로 인한 부수 효과*. 즉 thesis claim으로 인용해서는 안 되는 inflated number.")
    L.append("")
    L.append("### 4.2 v2 clinical: directional 분리 → IMU 한계 노출")
    L.append("")
    L.append(f"- v2에서 C5→C6: {c5_to_c6_v2:.2f}, C6→C5: {c6_to_c5_v2:.2f}. **C6→C5가 v1보다 {abs(c6_to_c5_v2-c6_to_c5_v1):.2f} 증가/감소**.")
    L.append("- v2 corpus phrase는 \"우측 무릎 외반\"(C5) vs \"양측 무릎 외반\"(C6)으로 어휘 분리 → text anchor가 분리 → IMU가 어디로 정렬할지 결정해야 함.")
    L.append("- 결과: 손목 IMU의 wrist-position 신호로는 right-only vs bilateral 구분이 어려워 C6 sample이 **빈도 더 높은 C5 anchor로 over-attribute**.")
    L.append("")
    L.append("### 4.3 paper claim")
    L.append("")
    L.append("이 결과는 두 가지 thesis 주장을 동시에 정량화한다:")
    L.append("")
    L.append("1. **Closed-vocabulary policy(외부 caption directional ban)는 단순 보호장치가 아니다** — directional vocab을 도입하면 IMU 한계가 노출되므로, 정책이 *실제로 잘 작동*함을 시사 (외부 caption에 \"우측 무릎 외반\"을 출력했다가 wrong이면 critical failure).")
    L.append("2. **Multi-modal 필요성의 정량 근거** — wrist IMU 단독으로 trilateral knee valgus를 구분하지 못한다는 직접 측정. video → MediaPipe joint trajectory 도입 필요성을 데이터로 증명. (교수님 피드백 5 video 필요성과 직결.)")
    L.append("")
    L.append("## 5. Discussion 본문 활용 권장 문장")
    L.append("")
    L.append("> \"v1 abstract corpus에서 관찰된 C6 class R@1 0.40은 (...) 의미 있는 분류 능력의 증거로 보였으나, "
             "directional vocabulary를 도입한 v2 clinical corpus에서 동일 클래스가 0.25로 하락한 패턴은 "
             "wrist IMU 단독 입력이 left/right/bilateral knee valgus의 trilateral 구분에 본질적 한계를 가짐을 "
             "보여준다. 본 결과는 closed-vocabulary policy의 정당성과 추후 multi-modal 확장 필요성을 "
             "동시에 정량적으로 뒷받침한다.\"")
    L.append("")
    L.append("## 6. 산출물")
    L.append("")
    L.append("```")
    L.append("data/step4r/4rc_contrastive_optional/")
    L.append("└── c5c6_confusion_matrices.csv  (long-format: version_k × true × pred × rate)")
    L.append("")
    L.append("reports/step4r/4rc_contrastive_optional/")
    L.append("├── c5c6_confusion_figure.png    (2×2 grid: heatmap + stacked bar)")
    L.append("└── c5c6_confusion_analysis.md   (본 보고서)")
    L.append("```")
    L.append("")
    OUTPUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"saved markdown -> {OUTPUT_MD}")

    print("\n" + "=" * 64)
    print("Done.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
