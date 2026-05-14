"""Step 4R-B v2 — Aggregate one experiment across seeds.

Aggregates per-seed metrics / schema summaries into mean±std for one exp_id.

Reads (read-only):
    reports/step4r/4rb_attention/experiments/{exp_id}/seed{N}/metrics.csv
    reports/step4r/4rb_attention/experiments/{exp_id}/seed{N}/schema_summary.csv
    reports/step4r/4rb_attention/experiments/{exp_id}/seed{N}/temperature_scaling_metrics.csv

Writes:
    reports/step4r/4rb_attention/experiments/{exp_id}/
        aggregate_metrics.csv               (long format: metric, split, mean, std, p50, ...)
        aggregate_schema_distribution.csv   (mean±std of schema level/group rates)
        aggregate_results.md                (cross-seed comparison + schema collapse check)

CLI:
    --exp-id   str (required)
    --seeds    str default "42,43,44"
    --baseline-confident-rate  float optional — if provided, used for ±10%p
                               collapse check (default reads from baseline if exp-id
                               is itself b00_*; else 0.197 (legacy 4R-B single-seed)).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_BASE = PROJECT_ROOT / "reports" / "step4r" / "4rb_attention" / "experiments"

CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
SPLITS = ["train", "val", "test"]
LEVELS = ["confident", "hedged", "low", "no_call"]
AMB_GROUPS = ["confident_C2", "within_group_c1_c5_c6", "pair_c3_c4",
              "pair_plus_c2_absorption", "no_call", "uncategorized"]

# Reference baseline rates (legacy single-seed 4R-B post 2/3 across all rows)
LEGACY_SINGLE_SEED = {
    "confident": 0.1670,
    "hedged": 0.7521,  # train+val+test averaged from results §3
    "low": 0.0432,
    "no_call": 0.0098,
}


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-id", type=str, required=True)
    p.add_argument("--seeds", type=str, default="42,43,44")
    p.add_argument("--baseline-confident-rate", type=float, default=None)
    p.add_argument("--baseline-hedged-rate", type=float, default=None)
    return p.parse_args(argv)


def _load_seed_metrics(exp_id: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Returns (raw_metrics, schema_summary, post_T_metrics_per_split, T).

    post_T_metrics_per_split is the after-stage rows from
    temperature_scaling_metrics.csv (per split: ECE, log_loss, Brier, etc).
    """
    rel = REPORTS_BASE / exp_id / f"seed{seed}"
    metrics_csv = rel / "metrics.csv"
    summary_csv = rel / "schema_summary.csv"
    temp_csv = rel / "temperature_scaling_metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(metrics_csv)
    if not summary_csv.exists():
        raise FileNotFoundError(summary_csv)
    if not temp_csv.exists():
        raise FileNotFoundError(temp_csv)
    m = pd.read_csv(metrics_csv, encoding="utf-8-sig")
    s = pd.read_csv(summary_csv, encoding="utf-8-sig")
    t = pd.read_csv(temp_csv, encoding="utf-8-sig")
    T = float(t.loc[t["metric"] == "fitted_temperature", "value"].iloc[0])
    # Keep only after-stage per-split rows; relabel metric → metric_cal so they
    # don't collide with the raw-stage metrics from metrics.csv when joined.
    after = t[(t["stage"] == "after") & t["split"].isin(SPLITS)].copy()
    after = after[["split", "metric", "value"]].copy()
    after["metric"] = after["metric"].apply(lambda k: f"{k}_calibrated")
    return m, s, after, T


def main(argv=None) -> int:
    args = _parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = REPORTS_BASE / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_agg = out_dir / "aggregate_metrics.csv"
    out_schema = out_dir / "aggregate_schema_distribution.csv"
    out_md = out_dir / "aggregate_results.md"

    print("=" * 64)
    print(f"Aggregating exp_id={args.exp_id} seeds={seeds}")
    print("=" * 64)

    metrics_list, schema_list, after_list, T_list = [], [], [], []
    for sd in seeds:
        m, s, a, T = _load_seed_metrics(args.exp_id, sd)
        m = m.copy(); m["seed"] = sd
        s = s.copy(); s["seed"] = sd
        a = a.copy(); a["seed"] = sd
        metrics_list.append(m)
        schema_list.append(s)
        after_list.append(a)
        T_list.append(T)

    M = pd.concat(metrics_list, ignore_index=True)
    S = pd.concat(schema_list, ignore_index=True)
    # Merge post-T calibrated per-split metrics into M so aggregate_metrics.csv
    # carries both raw and calibrated rows.
    A = pd.concat(after_list, ignore_index=True)
    M = pd.concat([M, A], ignore_index=True)

    # ---- Aggregate metrics ----
    agg_rows = []
    for (split, metric), grp in M.groupby(["split", "metric"]):
        vals = grp["value"].astype(float).to_numpy()
        agg_rows.append({
            "split": split, "metric": metric,
            "mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "median": float(np.median(vals)),
            "min": float(vals.min()), "max": float(vals.max()),
            "n_seeds": int(len(vals)),
            "values": ",".join(f"{v:.4f}" for v in vals),
        })
    pd.DataFrame(agg_rows).to_csv(out_agg, index=False, encoding="utf-8-sig")
    print(f"saved aggregate metrics -> {out_agg}")

    # ---- Aggregate schema distribution ----
    schema_rows = []
    for (split, metric), grp in S.groupby(["split", "metric"]):
        vals = grp["value"].astype(float).to_numpy()
        schema_rows.append({
            "split": split, "metric": metric,
            "mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "median": float(np.median(vals)),
            "min": float(vals.min()), "max": float(vals.max()),
            "n_seeds": int(len(vals)),
            "values": ",".join(f"{v:.4f}" for v in vals),
        })
    pd.DataFrame(schema_rows).to_csv(out_schema, index=False, encoding="utf-8-sig")
    print(f"saved aggregate schema -> {out_schema}")

    # ---- Schema collapse check (vs legacy single-seed baseline) ----
    base_confident = (
        args.baseline_confident_rate
        if args.baseline_confident_rate is not None
        else LEGACY_SINGLE_SEED["confident"]
    )
    base_hedged = (
        args.baseline_hedged_rate
        if args.baseline_hedged_rate is not None
        else LEGACY_SINGLE_SEED["hedged"]
    )

    all_rows = S[S["split"] == "all"]
    def _rate(metric: str) -> tuple[float, float]:
        sub = all_rows[all_rows["metric"] == metric]
        if len(sub) == 0:
            return float("nan"), float("nan")
        vals = sub["value"].astype(float).to_numpy()
        return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

    conf_m, conf_s = _rate("level_confident_rate")
    hedg_m, hedg_s = _rate("level_hedged_rate")
    low_m, low_s = _rate("level_low_rate")
    nc_m, nc_s = _rate("level_no_call_rate")

    conf_delta_p = (conf_m - base_confident) * 100.0
    hedg_delta_p = (hedg_m - base_hedged) * 100.0
    collapse = abs(conf_delta_p) > 10.0 or abs(hedg_delta_p) > 10.0

    # ---- per-seed test macroF1, test ECE (raw + calibrated) ----
    test_f1 = []
    test_ece_raw = []
    test_ece_cal = []
    for sd in seeds:
        sub = M[(M["seed"] == sd) & (M["split"] == "test") & (M["metric"] == "macro_f1")]
        test_f1.append(float(sub["value"].iloc[0]))
        sub = M[(M["seed"] == sd) & (M["split"] == "test") & (M["metric"] == "ece_15bin")]
        test_ece_raw.append(float(sub["value"].iloc[0]))
        sub = M[(M["seed"] == sd) & (M["split"] == "test") & (M["metric"] == "ece_15bin_calibrated")]
        test_ece_cal.append(float(sub["value"].iloc[0]) if len(sub) > 0 else float("nan"))
    test_f1_mean = float(np.mean(test_f1))
    test_f1_std = float(np.std(test_f1, ddof=1)) if len(test_f1) > 1 else 0.0
    test_ece_raw_mean = float(np.mean(test_ece_raw))
    test_ece_raw_std = float(np.std(test_ece_raw, ddof=1)) if len(test_ece_raw) > 1 else 0.0
    test_ece_cal_mean = float(np.nanmean(test_ece_cal))
    test_ece_cal_std = float(np.nanstd(test_ece_cal, ddof=1)) if len(test_ece_cal) > 1 else 0.0

    # ---- Markdown ----
    L: list[str] = []
    L.append(f"# 4R-B 집계 — exp_id=`{args.exp_id}` (seeds {seeds})")
    L.append("")
    L.append("- 생성 스크립트: `scripts/aggregate_step4r_experiment.py`")
    L.append("- per-seed 산출물은 `experiments/{exp_id}/seed{N}/` 참조.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 1. test 분류·calibration (seed별 + mean±std)")
    L.append("")
    L.append(
        "ECE는 raw posterior(temperature scaling 적용 전)와 calibrated posterior(T 적용 후) "
        "두 단계를 모두 보고한다. macro F1과 accuracy는 monotonic T 변환에서 보존되므로 "
        "단일 컬럼."
    )
    L.append("")
    L.append("| seed | T | test macro F1 | test ECE (raw) | test ECE (calibrated) |")
    L.append("|---|---:|---:|---:|---:|")
    for i, sd in enumerate(seeds):
        L.append(
            f"| {sd} | {T_list[i]:.4f} | {test_f1[i]:.4f} | "
            f"{test_ece_raw[i]:.4f} | {test_ece_cal[i]:.4f} |"
        )
    L.append(
        f"| **mean±std** | "
        f"**{np.mean(T_list):.4f}±{np.std(T_list, ddof=1) if len(T_list)>1 else 0:.4f}** | "
        f"**{test_f1_mean:.4f}±{test_f1_std:.4f}** | "
        f"**{test_ece_raw_mean:.4f}±{test_ece_raw_std:.4f}** | "
        f"**{test_ece_cal_mean:.4f}±{test_ece_cal_std:.4f}** |"
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 2. all-row schema 분포 (mean±std)")
    L.append("")
    L.append("| level | mean | std |")
    L.append("|---|---:|---:|")
    L.append(f"| confident | {conf_m:.4f} | {conf_s:.4f} |")
    L.append(f"| hedged | {hedg_m:.4f} | {hedg_s:.4f} |")
    L.append(f"| low | {low_m:.4f} | {low_s:.4f} |")
    L.append(f"| no_call | {nc_m:.4f} | {nc_s:.4f} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 3. schema collapse 체크 (vs legacy 단일-seed 4R-B)")
    L.append("")
    L.append(f"기준 baseline (legacy 단일 seed): confident={base_confident:.4f}, hedged={base_hedged:.4f}")
    L.append(f"본 실험 mean: confident={conf_m:.4f} (Δ={conf_delta_p:+.2f}%p), hedged={hedg_m:.4f} (Δ={hedg_delta_p:+.2f}%p)")
    L.append("")
    L.append("기준: ±10%p 이내. confident와 hedged 둘 다 통과해야 schema 보존.")
    L.append("")
    if collapse:
        L.append(f"**판정: 🚫 COLLAPSE 발생** — confident Δ={conf_delta_p:+.2f}%p, hedged Δ={hedg_delta_p:+.2f}%p")
    else:
        L.append(f"**판정: ✅ schema 분포 보존** — confident Δ={conf_delta_p:+.2f}%p, hedged Δ={hedg_delta_p:+.2f}%p (둘 다 ±10%p 이내)")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 4. ambiguity group 분포 (mean±std, all split)")
    L.append("")
    L.append("| group | mean | std |")
    L.append("|---|---:|---:|")
    for amb in AMB_GROUPS:
        m, s = _rate(f"ambiguity_{amb}_rate")
        L.append(f"| {amb} | {m:.4f} | {s:.4f} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 5. per-class F1 (test, mean±std)")
    L.append("")
    L.append("| class | mean F1 | std | mean recall | std |")
    L.append("|---|---:|---:|---:|---:|")
    for c in CLASSES:
        sub_f = M[(M["split"] == "test") & (M["metric"] == f"f1_{c}")]
        sub_r = M[(M["split"] == "test") & (M["metric"] == f"recall_{c}")]
        f_vals = sub_f["value"].astype(float).to_numpy() if len(sub_f) > 0 else np.array([])
        r_vals = sub_r["value"].astype(float).to_numpy() if len(sub_r) > 0 else np.array([])
        if len(f_vals) > 0:
            L.append(
                f"| {c} | {f_vals.mean():.4f} | {f_vals.std(ddof=1):.4f} | "
                f"{r_vals.mean():.4f} | {r_vals.std(ddof=1):.4f} |"
            )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 6. ambiguity 지표 (test, mean±std)")
    L.append("")
    L.append("| 지표 | mean | std |")
    L.append("|---|---:|---:|")
    for k in ["amb_c2_recall", "amb_c1_c5_c6_internal", "amb_c3_c4_pair",
              "amb_c3_to_c2_absorb", "amb_c4_to_c2_absorb"]:
        sub = M[(M["split"] == "test") & (M["metric"] == k)]
        vals = sub["value"].astype(float).to_numpy()
        if len(vals) > 0:
            L.append(f"| {k} | {vals.mean():.4f} | {vals.std(ddof=1) if len(vals)>1 else 0:.4f} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 7. median seed 선택 (downstream caption layer용)")
    L.append("")
    median_idx = int(np.argsort(test_f1)[len(test_f1) // 2])
    median_seed = seeds[median_idx]
    L.append(
        f"test macroF1 기준 median seed = **{median_seed}** "
        f"(macroF1={test_f1[median_idx]:.4f}, T={T_list[median_idx]:.4f})"
    )
    L.append("")
    L.append("Step 5_v2~7_v2 (caption layer) 재실행 시 본 median seed의 schema CSV를 입력으로 사용한다.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("*aggregate report. 본 실험의 per-seed 산출물은 수정되지 않는다.*")
    out_md.write_text("\n".join(L), encoding="utf-8")
    print(f"saved markdown -> {out_md}")

    print()
    print("=" * 64)
    print(
        f"Done. exp={args.exp_id} | test macroF1 = {test_f1_mean:.4f}±{test_f1_std:.4f} | "
        f"test ECE raw = {test_ece_raw_mean:.4f}±{test_ece_raw_std:.4f} | "
        f"test ECE cal = {test_ece_cal_mean:.4f}±{test_ece_cal_std:.4f}"
    )
    print(
        f"schema: confident={conf_m:.4f} Δ={conf_delta_p:+.2f}%p, "
        f"hedged={hedg_m:.4f} Δ={hedg_delta_p:+.2f}%p, collapse={'YES' if collapse else 'NO'}"
    )
    print(f"median seed: {median_seed}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
