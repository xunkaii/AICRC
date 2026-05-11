"""
Render a 6x6 confusion matrix PNG for Step 2.5, with ambiguity-pattern overlays.

Input:
  reports/step2/4_separability/reference_separability_confusion_by_setting.csv

Choice:
  setting = raw_core_features, split = test
  (matches the numbers shown on slide 6-B and the reframing memo §4.4)

Output:
  reports/step4r/step25_confusion_matrix.png

Run:
  python -X utf8 scripts/render_step25_confusion_matrix.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "reports" / "step2" / "4_separability" / "reference_separability_confusion_by_setting.csv"
OUT_PNG = ROOT / "reports" / "step4r" / "step25_confusion_matrix.png"

SETTING = "raw_core_features"
SPLIT = "test"
CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]


def load_matrix() -> np.ndarray:
    n = len(CLASSES)
    idx = {c: i for i, c in enumerate(CLASSES)}
    M = np.zeros((n, n), dtype=int)
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["setting"] != SETTING or row["split"] != SPLIT:
                continue
            M[idx[row["y_true"]], idx[row["y_pred"]]] += int(row["count"])
    return M


def main() -> None:
    M = load_matrix()
    row_sums = M.sum(axis=1, keepdims=True)
    P = M / np.where(row_sums == 0, 1, row_sums)  # row-normalized (recall-style)

    # Korean font on Windows
    rcParams["font.family"] = "Malgun Gothic"
    rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(9.0, 7.6), dpi=200)

    # Heatmap (row-normalized) — Purples palette for visual coherence with slide 6-B
    im = ax.imshow(P, cmap="Purples", vmin=0.0, vmax=P.max() * 1.05, aspect="equal")

    # Tick labels
    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_yticks(np.arange(len(CLASSES)))
    ax.set_xticklabels(CLASSES, fontsize=12, fontweight="bold")
    ax.set_yticklabels(CLASSES, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted class  (y_pred)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("True class  (y_true)", fontsize=12, fontweight="bold", labelpad=10)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Cell annotations: count on top, percentage below
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            pct = P[i, j] * 100
            cnt = M[i, j]
            text_color = "white" if P[i, j] > P.max() * 0.55 else "#1F2937"
            weight = "bold" if i == j else "normal"
            ax.text(j, i - 0.12, f"{cnt}", ha="center", va="center",
                    fontsize=11, fontweight=weight, color=text_color)
            ax.text(j, i + 0.18, f"{pct:.1f}%", ha="center", va="center",
                    fontsize=9, color=text_color)

    # Light gridlines between cells
    ax.set_xticks(np.arange(-0.5, len(CLASSES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CLASSES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)

    # ----- Ambiguity-pattern overlays --------------------------------------
    # 1) C1/C5/C6 internal-confusion block (orange).
    #    Indices 0, 4, 5 — non-contiguous, so draw 9 individual cell borders.
    for i in (0, 4, 5):
        for j in (0, 4, 5):
            ax.add_patch(patches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor="#EA580C", linewidth=2.4))

    # 2) C3/C4 pair-confusion block (red): rows 2-3, cols 2-3
    ax.add_patch(patches.Rectangle(
        (2 - 0.5, 2 - 0.5), 2, 2,
        fill=False, edgecolor="#DC2626", linewidth=2.8))

    # 3) C3/C4 → C2 absorption (red dashed): rows 2-3, col 1
    ax.add_patch(patches.Rectangle(
        (1 - 0.5, 2 - 0.5), 1, 2,
        fill=False, edgecolor="#B91C1C", linewidth=2.4, linestyle="--"))

    # 4) C2 diagonal cell (green): row 1, col 1
    ax.add_patch(patches.Rectangle(
        (1 - 0.5, 1 - 0.5), 1, 1,
        fill=False, edgecolor="#059669", linewidth=2.8))

    # Title
    diag_recall = np.diag(P)
    macro = diag_recall.mean()
    fig.suptitle(
        "Step 2.5 · Confusion matrix with ambiguity-pattern overlay",
        fontsize=15, fontweight="bold", y=0.985,
    )
    ax.set_title(
        f"{SETTING} · {SPLIT}   (row-normalized · counts above · macro recall = {macro*100:.1f}%)",
        fontsize=10, color="#6B7280", pad=24,
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.04)
    cbar.set_label("row-normalized rate", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Legend for overlays
    handles = [
        patches.Patch(edgecolor="#059669", facecolor="none", linewidth=2.5,
                      label="C2 diagonal — 분리 가능"),
        patches.Patch(edgecolor="#EA580C", facecolor="none", linewidth=2.5,
                      label="C1·C5·C6 그룹 내부 모호성"),
        patches.Patch(edgecolor="#DC2626", facecolor="none", linewidth=2.5,
                      label="C3·C4 pair 혼동"),
        patches.Patch(edgecolor="#B91C1C", facecolor="none", linewidth=2.5,
                      linestyle="--", label="C3·C4 → C2 흡수"),
    ]
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.18, 1.0),
        fontsize=10, frameon=True, edgecolor="#E5E7EB", title="Ambiguity pattern",
        title_fontsize=11,
    )

    plt.subplots_adjust(top=0.86, right=0.78, left=0.10, bottom=0.06)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"row sums (per-class n): {row_sums.flatten().tolist()}")
    print(f"diagonal recall: " + ", ".join(f"{c}={r*100:.1f}%" for c, r in zip(CLASSES, diag_recall)))
    print(f"macro recall: {macro*100:.2f}%")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
