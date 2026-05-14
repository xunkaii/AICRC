"""Aggregate v1-ratio split-robustness results into a single summary report.

Reads:
    # baseline (existing v1_s42)
    reports/step4r/4rb_attention/experiments/b01_5_aug_jitter_scale_strong/
        seed{42,43,44}/temperature_scaling_metrics.csv

    # new split seeds (s7, s123, s2024, s7777)
    reports/step4r_v1seed_robustness/s{SEED}/4rb_attention/experiments/
        b01_5_aug_jitter_scale_strong/seed{42,43,44}/temperature_scaling_metrics.csv

Writes:
    reports/step4r_v1seed_robustness/split_robustness_summary.md
    reports/step4r_v1seed_robustness/split_robustness_table.csv

Notes:
    - test macro_f1 is split-stage-invariant under temperature scaling (argmax
      preserved), so we read it from stage=before. Same value as stage=after.
    - test ECE we report is after calibration (stage=after, post-T scaling).
    - fitted_temperature is at split=='—' stage=='—'.

Run:
    python scripts/aggregate_v1seed_robustness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASELINE_ROOT = (
    PROJECT_ROOT / "reports" / "step4r" / "4rb_attention" / "experiments"
    / "b01_5_aug_jitter_scale_strong"
)
NEW_ROOT = PROJECT_ROOT / "reports" / "step4r_v1seed_robustness"
OUT_DIR = NEW_ROOT
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TABLE_CSV = OUT_DIR / "split_robustness_table.csv"
OUT_SUMMARY_MD = OUT_DIR / "split_robustness_summary.md"

LEARN_SEEDS = [42, 43, 44]
SPLIT_SEEDS_NEW = [7, 123, 2024, 7777]
EXP_ID = "b01_5_aug_jitter_scale_strong"


def _metrics_path(split_seed: int, learn_seed: int) -> Path:
    if split_seed == 42:
        return (
            BASELINE_ROOT / f"seed{learn_seed}"
            / "temperature_scaling_metrics.csv"
        )
    return (
        NEW_ROOT / f"s{split_seed}" / "4rb_attention" / "experiments"
        / EXP_ID / f"seed{learn_seed}" / "temperature_scaling_metrics.csv"
    )


def _extract(metrics_path: Path) -> dict:
    df = pd.read_csv(metrics_path, encoding="utf-8-sig")
    def _v(split, stage, metric):
        m = (df["split"] == split) & (df["stage"] == stage) & (df["metric"] == metric)
        s = df.loc[m, "value"]
        if len(s) == 0:
            raise KeyError(f"{split}/{stage}/{metric} not found in {metrics_path}")
        return float(s.iloc[0])
    return {
        "val_macro_f1": _v("val", "before", "macro_f1"),
        "test_acc": _v("test", "before", "accuracy"),
        "test_macro_f1": _v("test", "before", "macro_f1"),
        "test_ece_before": _v("test", "before", "ece_15bin"),
        "test_ece_after": _v("test", "after", "ece_15bin"),
        "test_logloss_after": _v("test", "after", "log_loss"),
        "fitted_T": _v("—", "—", "fitted_temperature"),
    }


def main() -> int:
    print("=" * 64)
    print("Aggregating v1-ratio split-robustness results")
    print("=" * 64)

    rows = []
    all_splits = [42] + SPLIT_SEEDS_NEW
    for split_seed in all_splits:
        for learn_seed in LEARN_SEEDS:
            p = _metrics_path(split_seed, learn_seed)
            if not p.exists():
                print(f"  WARN: missing {p}")
                continue
            m = _extract(p)
            rows.append({
                "split_seed": split_seed,
                "learn_seed": learn_seed,
                **m,
            })
            print(
                f"  s{split_seed:>4} / learn{learn_seed}: "
                f"test_F1={m['test_macro_f1']:.4f}  "
                f"ECE_after={m['test_ece_after']:.4f}  T={m['fitted_T']:.3f}"
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No metrics found.")
    df.to_csv(OUT_TABLE_CSV, index=False, encoding="utf-8-sig")
    print(f"\nsaved long-format table -> {OUT_TABLE_CSV}")

    # ----- Build summary statistics -----
    grp = df.groupby("split_seed")
    summary = grp.agg(
        test_F1_mean=("test_macro_f1", "mean"),
        test_F1_std=("test_macro_f1", lambda s: s.std(ddof=0)),
        test_F1_min=("test_macro_f1", "min"),
        test_F1_max=("test_macro_f1", "max"),
        test_ECE_after_mean=("test_ece_after", "mean"),
        test_ECE_after_std=("test_ece_after", lambda s: s.std(ddof=0)),
        T_mean=("fitted_T", "mean"),
        T_std=("fitted_T", lambda s: s.std(ddof=0)),
        n=("test_macro_f1", "count"),
    ).reset_index()

    # Variance decomposition.
    split_means_F1 = grp["test_macro_f1"].mean()
    grand_mean_F1 = split_means_F1.mean()
    between_split_std_F1 = float(split_means_F1.std(ddof=0))
    within_split_std_F1 = float(np.sqrt(
        (grp["test_macro_f1"].apply(lambda s: s.var(ddof=0))).mean()
    ))
    total_std_F1 = float(df["test_macro_f1"].std(ddof=0))

    print()
    print("=" * 64)
    print("Variance decomposition (test macro F1)")
    print("=" * 64)
    print(f"  grand mean              = {grand_mean_F1:.4f}")
    print(f"  between-split std       = {between_split_std_F1:.4f}  "
          f"(across {len(split_means_F1)} split seeds)")
    print(f"  within-split std (mean) = {within_split_std_F1:.4f}  "
          f"(across 3 learn seeds, averaged)")
    print(f"  total std               = {total_std_F1:.4f}  "
          f"(across {len(df)} runs)")
    if between_split_std_F1 > 1.5 * within_split_std_F1:
        verdict = (
            "→ split variance dominates: test 8명 구성 운이 큰 영향. "
            "honest limitation으로 명시 필요."
        )
    elif between_split_std_F1 < 0.5 * within_split_std_F1:
        verdict = (
            "→ within-split (학습 seed) variance dominates: v1은 "
            "split-robust. 0.513 신뢰 가능."
        )
    else:
        verdict = (
            "→ split variance와 학습 variance가 비슷한 규모. "
            "두 효과 모두 보고 권장."
        )
    print(f"  {verdict}")

    # ----- Build markdown report -----
    L = []
    L.append("# v1-ratio (36/8/8) split-robustness — 4 split seeds × 3 learn seeds")
    L.append("")
    L.append(
        "본 보고서는 v1의 b01.5 test F1 0.513이 어느 정도 split-robust한지를 "
        "정량화한다. 인원 비율(36/8/8)과 학습 recipe(b01.5: σ=0.05, scale "
        "[0.85, 1.15])는 v1 baseline과 완전히 동일하며, "
        "참가자 셔플 seed만 바꿔서 같은 학습을 반복했다."
    )
    L.append("")
    L.append(
        f"- baseline: `s42` (기존 v1 confirmed baseline, 학습 seed {LEARN_SEEDS})"
    )
    L.append(
        f"- new: `s7`, `s123`, `s2024`, `s7777` (각각 학습 seed {LEARN_SEEDS})"
    )
    L.append(f"- 총 5 splits × 3 learn seeds = 15 runs")
    L.append("")
    L.append("## 1. test macro F1 matrix")
    L.append("")
    L.append("| split_seed | seed 42 | seed 43 | seed 44 | mean | std | min | max |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for split_seed in all_splits:
        row = summary[summary["split_seed"] == split_seed]
        if row.empty:
            continue
        r = row.iloc[0]
        per_seed = []
        for ls in LEARN_SEEDS:
            sub = df[(df["split_seed"] == split_seed) & (df["learn_seed"] == ls)]
            per_seed.append(
                f"{sub['test_macro_f1'].iloc[0]:.4f}" if not sub.empty else "—"
            )
        L.append(
            f"| s{split_seed} | {per_seed[0]} | {per_seed[1]} | {per_seed[2]} | "
            f"**{r['test_F1_mean']:.4f}** | {r['test_F1_std']:.4f} | "
            f"{r['test_F1_min']:.4f} | {r['test_F1_max']:.4f} |"
        )
    L.append("")
    L.append(f"- **grand mean** = {grand_mean_F1:.4f}")
    L.append(f"- **between-split std** = {between_split_std_F1:.4f}  "
             f"(splits별 mean 4~5개의 std)")
    L.append(f"- **within-split std (mean)** = {within_split_std_F1:.4f}  "
             f"(각 split 안 3-learn-seed std의 평균)")
    L.append(f"- **total std (15 runs)** = {total_std_F1:.4f}")
    L.append("")
    L.append("## 2. test ECE (after calibration) matrix")
    L.append("")
    L.append("| split_seed | seed 42 | seed 43 | seed 44 | mean | std |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for split_seed in all_splits:
        row = summary[summary["split_seed"] == split_seed]
        if row.empty:
            continue
        r = row.iloc[0]
        per_seed = []
        for ls in LEARN_SEEDS:
            sub = df[(df["split_seed"] == split_seed) & (df["learn_seed"] == ls)]
            per_seed.append(
                f"{sub['test_ece_after'].iloc[0]:.4f}" if not sub.empty else "—"
            )
        L.append(
            f"| s{split_seed} | {per_seed[0]} | {per_seed[1]} | {per_seed[2]} | "
            f"**{r['test_ECE_after_mean']:.4f}** | {r['test_ECE_after_std']:.4f} |"
        )
    L.append("")
    L.append("## 3. fitted temperature T matrix")
    L.append("")
    L.append("| split_seed | seed 42 | seed 43 | seed 44 | mean | std |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for split_seed in all_splits:
        row = summary[summary["split_seed"] == split_seed]
        if row.empty:
            continue
        r = row.iloc[0]
        per_seed = []
        for ls in LEARN_SEEDS:
            sub = df[(df["split_seed"] == split_seed) & (df["learn_seed"] == ls)]
            per_seed.append(
                f"{sub['fitted_T'].iloc[0]:.3f}" if not sub.empty else "—"
            )
        L.append(
            f"| s{split_seed} | {per_seed[0]} | {per_seed[1]} | {per_seed[2]} | "
            f"**{r['T_mean']:.3f}** | {r['T_std']:.3f} |"
        )
    L.append("")
    L.append("## 4. variance decomposition 해석")
    L.append("")
    L.append(verdict)
    L.append("")
    L.append("### 정량 기준")
    L.append("")
    L.append(
        "- between-split std는 *"
        "다른 8명을 test로 뽑았을 때 b01.5 mean F1이 얼마나 흔들리는지*"
        "를 잡는다."
    )
    L.append(
        "- within-split std는 *같은 8명 test 고정 시 학습 seed에 따라 얼마나 흔들리는지*"
        "를 잡는다. b01.5는 이미 학습 seed std가 작다 (0.008 부근)."
    )
    L.append(
        "- 두 std 비율이 v1 baseline 보고의 신뢰성을 결정한다: between이 within보다 1.5× 크면 "
        "split이 결정적, 작으면 학습이 결정적, 비슷하면 둘 다."
    )
    L.append("")
    L.append("## 5. 산출물 위치")
    L.append("")
    L.append("```")
    L.append(
        "data/step4r_v1seed_robustness/s{7,123,2024,7777}/"
        "manifest_split.csv (4 files)"
    )
    L.append(
        "data/step4r_v1seed_robustness/s{...}/4rb_attention/"
        "step4r_sequence_dataset.npz (4 files)"
    )
    L.append(
        "data/step4r_v1seed_robustness/s{...}/4rb_attention/experiments/"
        f"{EXP_ID}/seed{{42,43,44}}/  (12 dirs)"
    )
    L.append(
        "reports/step4r_v1seed_robustness/s{...}/...  (12 dirs)"
    )
    L.append(
        "checkpoints/step4r_v1seed_robustness/s{...}/...  (12 dirs)"
    )
    L.append("```")
    L.append("")
    L.append(
        "삭제 시: `data/step4r_v1seed_robustness/`, "
        "`checkpoints/step4r_v1seed_robustness/`, "
        "`reports/step4r_v1seed_robustness/` 만 제거하면 됨."
    )

    OUT_SUMMARY_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"\nsaved summary -> {OUT_SUMMARY_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
