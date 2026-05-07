"""Step 2.5-3b — Normalization Feasibility Audit (v2).

Goal: check, on a *small* set of candidate features, whether common
normalization strategies are (a) feasible at inference time for new
users, and (b) preserve class-related signal while attenuating
posture-related bias. This is *not* a feature-selection step.

Scope (intentionally narrow to avoid feature jungle):
  features:
    - motion_range_acc_z
    - depth_proxy
    - bottom_recovery_slope_acc_z
  methods:
    - raw
    - posture_train_zscore               (train stats only — feasible)
    - posture_train_robust               (train stats only — feasible)
    - participant_zscore_upper_bound     (per-subject — UPPER BOUND ONLY)

Inputs (read-only):
  data/manifest_split.csv
  data/step2/candidate_feature_bank.csv

Outputs:
  data/step2/normalization_feasibility_features.csv
  reports/normalization_feasibility_audit.md
  reports/normalization_feasibility_effect_summary.csv
  reports/normalization_feasibility_split_robustness.csv
  reports/normalization_feasibility_overlap.csv

Run:
  python scripts/audit_step25_normalization_feasibility_v2.py
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
MANIFEST_PATH = DATA_DIR / "manifest_split.csv"
FEATURE_BANK_PATH = DATA_DIR / "step2" / "candidate_feature_bank.csv"

CLASSES = ("C1", "C2", "C3", "C4", "C5", "C6")
POSTURES = ("SA", "CA", "HW")
SPLITS = ("train", "val", "test")

FEATURES = (
    "motion_range_acc_z",
    "depth_proxy",
    "bottom_recovery_slope_acc_z",
)

METHODS = (
    "raw",
    "posture_train_zscore",
    "posture_train_robust",
    "participant_zscore_upper_bound",
)

NORMALIZED_METHODS = METHODS[1:]  # everything except "raw"

# Robust scale: divide by IQR / 1.349 so the unit is roughly comparable to z.
ROBUST_IQR_DIVISOR = 1.349

# Stability flagging.
RAW_REL_RANGE_THRESHOLD = 0.20
NORMALIZED_ABS_RANGE_THRESHOLD = 0.20

# Confusion groups for overlap.
UNCERTAINTY_GROUPS = {
    "C1_C5_C6": ("C1", "C5", "C6"),
    "C3_C4":    ("C3", "C4"),
}

OUTPUT_FEATURES_COLS_BASE = [
    "sample_id", "participant_id", "split",
    "class_id", "posture_canonical",
]

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_csv_dict(path: Path) -> list[dict]:
    if not path.is_file():
        sys.exit(f"ERROR: required input not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_float(s):
    if s is None or s == "":
        return None
    try:
        v = float(s)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def fmt_float(v, ndigits=6):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return ""
    return f"{float(v):.{ndigits}f}"


# ---------------------------------------------------------------------------
# Stats fitters (train-only or per-participant)
# ---------------------------------------------------------------------------

def fit_posture_train_zscore(rows, feature):
    """Return posture_canonical -> (mean, std). Computed on TRAIN rows only."""
    by_p = defaultdict(list)
    for r in rows:
        if r["split"] != "train":
            continue
        v = parse_float(r.get(feature))
        if v is None:
            continue
        by_p[r["posture_canonical"]].append(v)
    out = {}
    for p, vals in by_p.items():
        if len(vals) >= 2:
            arr = np.asarray(vals, dtype=np.float64)
            out[p] = (float(arr.mean()), float(arr.std(ddof=0)))
        else:
            out[p] = (float("nan"), float("nan"))
    return out


def fit_posture_train_robust(rows, feature):
    """Return posture_canonical -> (median, IQR). TRAIN rows only."""
    by_p = defaultdict(list)
    for r in rows:
        if r["split"] != "train":
            continue
        v = parse_float(r.get(feature))
        if v is None:
            continue
        by_p[r["posture_canonical"]].append(v)
    out = {}
    for p, vals in by_p.items():
        if len(vals) >= 2:
            arr = np.asarray(vals, dtype=np.float64)
            out[p] = (
                float(np.median(arr)),
                float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            )
        else:
            out[p] = (float("nan"), float("nan"))
    return out


def fit_participant_zscore(rows, feature):
    """Return participant_id -> (mean, std). UPPER-BOUND only — uses all splits."""
    by_pid = defaultdict(list)
    for r in rows:
        v = parse_float(r.get(feature))
        if v is None:
            continue
        by_pid[r["participant_id"]].append(v)
    out = {}
    for pid, vals in by_pid.items():
        if len(vals) >= 2:
            arr = np.asarray(vals, dtype=np.float64)
            out[pid] = (float(arr.mean()), float(arr.std(ddof=0)))
        else:
            out[pid] = (float("nan"), float("nan"))
    return out


# ---------------------------------------------------------------------------
# Per-row normalization
# ---------------------------------------------------------------------------

def apply_zscore(value, center, scale):
    if value is None:
        return float("nan"), "raw_value_missing"
    if center is None or scale is None:
        return float("nan"), "stats_missing"
    if math.isnan(center) or math.isnan(scale):
        return float("nan"), "stats_nan"
    if scale <= 1e-12:
        return float("nan"), "scale_zero"
    return (value - center) / scale, ""


# ---------------------------------------------------------------------------
# eta-squared
# ---------------------------------------------------------------------------

def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """SSB / SST across labeled groups."""
    valid = ~np.isnan(values)
    values = values[valid]
    groups = groups[valid]
    if values.size == 0:
        return float("nan")
    grand = values.mean()
    sst = float(np.sum((values - grand) ** 2))
    if sst <= 0:
        return 0.0
    ssb = 0.0
    for g in np.unique(groups):
        mask = groups == g
        n_g = int(mask.sum())
        if n_g == 0:
            continue
        ssb += n_g * (values[mask].mean() - grand) ** 2
    return float(ssb / sst)


# ---------------------------------------------------------------------------
# Overlap (Gaussian approximation)
# ---------------------------------------------------------------------------

def normal_overlap(d_abs):
    if math.isnan(d_abs) or math.isinf(d_abs):
        return float("nan")
    z = -d_abs / 2.0
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return 2.0 * cdf


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_normalized_table(rows):
    """Return a list of rows with raw + normalized values for each feature/method."""
    print("Fitting normalization stats ...")
    posture_z_stats = {f: fit_posture_train_zscore(rows, f) for f in FEATURES}
    posture_r_stats = {f: fit_posture_train_robust(rows, f) for f in FEATURES}
    participant_z_stats = {f: fit_participant_zscore(rows, f) for f in FEATURES}

    print("Applying normalization to all rows ...")
    out_rows = []
    for r in rows:
        sid = r["sample_id"]
        pid = r["participant_id"]
        sp = r["split"]
        cls = r["class_id"]
        pos = r["posture_canonical"]

        raw_vals = {f: parse_float(r.get(f)) for f in FEATURES}
        warns: list[str] = []

        out = {
            "sample_id": sid,
            "participant_id": pid,
            "split": sp,
            "class_id": cls,
            "posture_canonical": pos,
        }
        for f in FEATURES:
            out[f"raw_{f}"] = raw_vals[f]

        compute_ok = True

        # posture_train_zscore
        for f in FEATURES:
            stats = posture_z_stats[f].get(pos, (None, None))
            v, w = apply_zscore(raw_vals[f], stats[0], stats[1])
            out[f"posture_train_zscore_{f}"] = v
            if w:
                warns.append(f"posture_train_zscore[{f}]:{w}")
                compute_ok = False

        # posture_train_robust
        for f in FEATURES:
            stats = posture_r_stats[f].get(pos, (None, None))
            center, iqr = stats
            scale = (iqr / ROBUST_IQR_DIVISOR) if (iqr is not None
                                                    and not math.isnan(iqr)) \
                else None
            v, w = apply_zscore(raw_vals[f], center, scale)
            out[f"posture_train_robust_{f}"] = v
            if w:
                warns.append(f"posture_train_robust[{f}]:{w}")
                compute_ok = False

        # participant_zscore_upper_bound
        for f in FEATURES:
            stats = participant_z_stats[f].get(pid, (None, None))
            v, w = apply_zscore(raw_vals[f], stats[0], stats[1])
            out[f"participant_zscore_upper_bound_{f}"] = v
            if w:
                warns.append(f"participant_zscore_upper_bound[{f}]:{w}")
                compute_ok = False

        out["normalization_compute_ok"] = compute_ok
        out["normalization_warning"] = ";".join(warns)
        out_rows.append(out)

    return out_rows


def write_features_csv(out_rows, path: Path):
    cols = list(OUTPUT_FEATURES_COLS_BASE)
    for f in FEATURES:
        cols.append(f"raw_{f}")
    for m in NORMALIZED_METHODS:
        for f in FEATURES:
            cols.append(f"{m}_{f}")
    cols.append("normalization_compute_ok")
    cols.append("normalization_warning")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in out_rows:
            row = []
            for c in cols:
                v = r.get(c, "")
                if v is None:
                    row.append("")
                elif isinstance(v, bool):
                    row.append("True" if v else "False")
                elif isinstance(v, float):
                    if math.isnan(v) or math.isinf(v):
                        row.append("")
                    else:
                        row.append(f"{v:.6f}")
                else:
                    row.append(str(v))
            w.writerow(row)


# ---------------------------------------------------------------------------
# Effect summary (class η² + posture η²)
# ---------------------------------------------------------------------------

def column_for(method, feature):
    return f"raw_{feature}" if method == "raw" else f"{method}_{feature}"


def write_effect_summary(out_rows, path: Path):
    classes = np.array([r["class_id"] for r in out_rows])
    postures = np.array([r["posture_canonical"] for r in out_rows])

    rows_csv = []
    raw_eta = {}  # (feature) -> (class_eta2_raw, posture_eta2_raw)

    for f in FEATURES:
        raw_vals = np.array(
            [r["raw_" + f] if r["raw_" + f] is not None else float("nan")
             for r in out_rows], dtype=np.float64
        )
        raw_eta[f] = (
            eta_squared(raw_vals, classes),
            eta_squared(raw_vals, postures),
        )

    for f in FEATURES:
        for m in METHODS:
            col = column_for(m, f)
            vals = np.array(
                [r[col] if r[col] is not None else float("nan")
                 for r in out_rows], dtype=np.float64
            )
            n_valid = int(np.sum(~np.isnan(vals)))
            class_eta = eta_squared(vals, classes)
            posture_eta = eta_squared(vals, postures)

            class_raw, posture_raw = raw_eta[f]
            class_pct = (class_eta / class_raw) if (class_raw and class_raw > 0) else float("nan")
            posture_pct = (posture_eta / posture_raw) if (posture_raw and posture_raw > 0) else float("nan")

            rows_csv.append({
                "feature": f,
                "method": m,
                "n_valid": n_valid,
                "class_eta2": class_eta,
                "posture_eta2": posture_eta,
                "class_eta2_pct_of_raw": class_pct,
                "posture_eta2_pct_of_raw": posture_pct,
            })

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "feature", "method", "n_valid",
            "class_eta2", "posture_eta2",
            "class_eta2_pct_of_raw", "posture_eta2_pct_of_raw",
        ])
        for r in rows_csv:
            w.writerow([
                r["feature"], r["method"], r["n_valid"],
                fmt_float(r["class_eta2"]),
                fmt_float(r["posture_eta2"]),
                fmt_float(r["class_eta2_pct_of_raw"]),
                fmt_float(r["posture_eta2_pct_of_raw"]),
            ])

    return rows_csv


# ---------------------------------------------------------------------------
# Split robustness
# ---------------------------------------------------------------------------

def write_split_robustness(out_rows, path: Path):
    by_split = {sp: [r for r in out_rows if r["split"] == sp] for sp in SPLITS}
    rows_csv = []
    for f in FEATURES:
        for m in METHODS:
            col = column_for(m, f)
            stats_per_split = {}
            for sp in SPLITS:
                vals = np.array(
                    [r[col] if r[col] is not None else float("nan")
                     for r in by_split[sp]], dtype=np.float64
                )
                vals = vals[~np.isnan(vals)]
                if vals.size:
                    stats_per_split[sp] = {
                        "n": int(vals.size),
                        "mean": float(vals.mean()),
                        "std": float(vals.std(ddof=0)),
                    }
                else:
                    stats_per_split[sp] = {
                        "n": 0,
                        "mean": float("nan"),
                        "std": float("nan"),
                    }
            means = [stats_per_split[sp]["mean"] for sp in SPLITS
                     if not math.isnan(stats_per_split[sp]["mean"])]
            if len(means) >= 2:
                mean_range = max(means) - min(means)
                overall = float(np.mean(means))
                rel = (mean_range / abs(overall)) if abs(overall) > 1e-12 \
                    else float("nan")
            else:
                mean_range = float("nan")
                rel = float("nan")

            if m == "raw":
                rule = "raw_relative_>0.20"
                flag = (not math.isnan(rel)) and (rel > RAW_REL_RANGE_THRESHOLD)
            else:
                rule = "normalized_absolute_>0.20"
                flag = (not math.isnan(mean_range)) and (
                    mean_range > NORMALIZED_ABS_RANGE_THRESHOLD
                )

            rows_csv.append({
                "feature": f, "method": m,
                **{f"{sp}_{k}": stats_per_split[sp][k]
                   for sp in SPLITS for k in ("n", "mean", "std")},
                "mean_range": mean_range,
                "rel_mean_range": rel,
                "flag_unstable": flag,
                "rule": rule,
            })

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "feature", "method",
            "train_n", "train_mean", "train_std",
            "val_n", "val_mean", "val_std",
            "test_n", "test_mean", "test_std",
            "mean_range", "rel_mean_range",
            "flag_unstable", "rule",
        ])
        for r in rows_csv:
            w.writerow([
                r["feature"], r["method"],
                r["train_n"], fmt_float(r["train_mean"]), fmt_float(r["train_std"]),
                r["val_n"],   fmt_float(r["val_mean"]),   fmt_float(r["val_std"]),
                r["test_n"],  fmt_float(r["test_mean"]),  fmt_float(r["test_std"]),
                fmt_float(r["mean_range"]),
                fmt_float(r["rel_mean_range"]),
                "True" if r["flag_unstable"] else "False",
                r["rule"],
            ])
    return rows_csv


# ---------------------------------------------------------------------------
# Uncertainty overlap
# ---------------------------------------------------------------------------

def write_overlap(out_rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "group", "feature", "method", "posture",
            "class_a", "class_b",
            "n_a", "n_b",
            "mean_a", "mean_b", "std_a", "std_b",
            "cohen_d", "overlap_estimate",
        ])

        for group_name, group_classes in UNCERTAINTY_GROUPS.items():
            pairs = []
            for i in range(len(group_classes)):
                for j in range(i + 1, len(group_classes)):
                    pairs.append((group_classes[i], group_classes[j]))

            for feat in FEATURES:
                for method in METHODS:
                    col = column_for(method, feat)
                    for posture in ("all", *POSTURES):
                        if posture == "all":
                            scope = out_rows
                        else:
                            scope = [r for r in out_rows
                                     if r["posture_canonical"] == posture]
                        for ca, cb in pairs:
                            arr_a = np.array(
                                [r[col] for r in scope
                                 if r["class_id"] == ca and r[col] is not None
                                 and not (isinstance(r[col], float)
                                          and math.isnan(r[col]))],
                                dtype=np.float64,
                            )
                            arr_b = np.array(
                                [r[col] for r in scope
                                 if r["class_id"] == cb and r[col] is not None
                                 and not (isinstance(r[col], float)
                                          and math.isnan(r[col]))],
                                dtype=np.float64,
                            )
                            if arr_a.size == 0 or arr_b.size == 0:
                                w.writerow([
                                    group_name, feat, method, posture, ca, cb,
                                    int(arr_a.size), int(arr_b.size),
                                    "", "", "", "", "", "",
                                ])
                                continue
                            m_a = float(arr_a.mean())
                            m_b = float(arr_b.mean())
                            s_a = float(arr_a.std(ddof=0))
                            s_b = float(arr_b.std(ddof=0))
                            pooled = math.sqrt((s_a * s_a + s_b * s_b) / 2.0)
                            d = ((m_a - m_b) / pooled) if pooled > 1e-12 \
                                else float("nan")
                            ov = (normal_overlap(abs(d))
                                  if not math.isnan(d) else float("nan"))
                            w.writerow([
                                group_name, feat, method, posture, ca, cb,
                                int(arr_a.size), int(arr_b.size),
                                fmt_float(m_a), fmt_float(m_b),
                                fmt_float(s_a), fmt_float(s_b),
                                fmt_float(d), fmt_float(ov),
                            ])


# ---------------------------------------------------------------------------
# Audit report
# ---------------------------------------------------------------------------

def write_audit_md(out_rows, effect_rows, robustness_rows, path: Path):
    n = len(out_rows)
    n_ok = sum(1 for r in out_rows if r["normalization_compute_ok"])

    # method × feature: pull effect numbers
    eff_by = {(r["feature"], r["method"]): r for r in effect_rows}
    rb_by = {(r["feature"], r["method"]): r for r in robustness_rows}

    unstable_pairs = [(r["feature"], r["method"]) for r in robustness_rows
                      if r["flag_unstable"]]

    lines: list[str] = []
    add = lines.append

    add("# Step 2.5-3b — Normalization Feasibility Audit")
    add("")
    add("**Scope.** This script audits whether common normalization strategies")
    add("are *feasible* in deployment and whether they preserve class signal")
    add("while reducing posture bias. **It is not a feature-selection step.**")
    add("No feature is adopted or rejected here.")
    add("")

    add("## 1. What this audit covers (and intentionally does not)")
    add("")
    add(f"- Features tested: {', '.join(f'`{x}`' for x in FEATURES)} — 3 only.")
    add("- Methods tested:")
    add("  - `raw` — original feature values")
    add("  - `posture_train_zscore` — per-posture mean/std fit on **train** "
        "rows only, applied to all splits")
    add("  - `posture_train_robust` — per-posture median / (IQR / "
        f"{ROBUST_IQR_DIVISOR}) fit on **train** rows only, applied to all")
    add("  - `participant_zscore_upper_bound` — per-subject mean/std fit on "
        "all of that subject's reps")
    add("")
    add("**Three-feature constraint is intentional.** Expanding to all "
        "candidate features at this stage would create a feature jungle "
        "before we know which normalization principle holds.")
    add("")

    add("## 2. Why participant z-score is *upper bound only*")
    add("")
    add("`participant_zscore_upper_bound` requires statistics computed from a")
    add("subject's own reps. For a brand-new user's **first rep**, no such")
    add("statistics exist; calibration data must be collected first.")
    add("")
    add("- Therefore it is **not** a candidate for the main inference pipeline.")
    add("- It is included here as a *ceiling* on how much normalization could")
    add("  help if per-subject calibration were available.")
    add("- Replacing the main path with participant z-score would silently")
    add("  break new-user inference; the report is explicit about this so")
    add("  later readers do not re-introduce it as a default.")
    add("")
    add("`posture_train_zscore` and `posture_train_robust` use **train**")
    add("statistics only and can be applied to a new user's first rep, given")
    add("the user's posture label, so they are the realistic main-pipeline")
    add("candidates here.")
    add("")

    add("## 3. Compute success")
    add("")
    add(f"- Rows: **{n}**")
    add(f"- `normalization_compute_ok = True`: **{n_ok}** "
        f"({(n_ok / n * 100) if n else 0:.2f}%)")
    add("")
    add("Stats fits (for transparency):")
    add("")
    add("| stat | scope | source split | usable for new-user first rep? |")
    add("|---|---|---|---|")
    add("| posture mean / std | per posture | train only | yes |")
    add("| posture median / IQR | per posture | train only | yes |")
    add("| participant mean / std | per subject | all splits | **no** (needs "
        "prior calibration) |")
    add("")

    add("## 4. Class effect vs posture effect (η²)")
    add("")
    add("Goal: after normalization, **class η² should stay ≈ raw**, while")
    add("**posture η² should decrease**. Posture-aware normalization is")
    add("expected to drive posture η² close to zero.")
    add("")
    add("| feature | method | n | class η² | posture η² | class η² / raw | "
        "posture η² / raw |")
    add("|---|---|---|---|---|---|---|")
    for f in FEATURES:
        for m in METHODS:
            r = eff_by[(f, m)]
            add(
                f"| `{f}` | `{m}` | {r['n_valid']} | "
                f"{r['class_eta2']:.4f} | {r['posture_eta2']:.4f} | "
                f"{r['class_eta2_pct_of_raw']:.3f} | "
                f"{r['posture_eta2_pct_of_raw']:.3f} |"
            )
    add("")
    add("Reading guide:")
    add("")
    add("- `class η² / raw ≈ 1.0` → class signal preserved.")
    add("- `posture η² / raw < 0.2` → posture bias largely removed.")
    add("- `class η² / raw < 0.5` → normalization scrubbed too much; "
        "danger signal.")
    add("")

    add("## 5. Split robustness")
    add("")
    add("- For `raw`: flag if (max − min) of split means > 20% of |overall mean|.")
    add("- For normalized methods: flag if absolute (max − min) of split means "
        f"> {NORMALIZED_ABS_RANGE_THRESHOLD:.2f} (z-score units).")
    add("")
    add("| feature | method | rule | unstable? |")
    add("|---|---|---|---|")
    for f in FEATURES:
        for m in METHODS:
            r = rb_by[(f, m)]
            add(f"| `{f}` | `{m}` | {r['rule']} | "
                f"{'**True**' if r['flag_unstable'] else 'False'} |")
    add("")
    add(f"Full table: `reports/normalization_feasibility_split_robustness.csv`")
    add("")

    add("## 6. Confusion-class overlap (uncertainty signal)")
    add("")
    add("- Groups examined:")
    for gname, classes in UNCERTAINTY_GROUPS.items():
        add(f"  - `{gname}`: {', '.join(classes)}")
    add("- Per (feature, method, posture-scope) and pair, we report Cohen's")
    add("  *d* and a Gaussian-approximation overlap estimate.")
    add("- **High overlap is not a reason to drop a feature.** In confusion")
    add("  groups (C1/C5/C6, C3/C4) classes are constructed to differ subtly,")
    add("  so substantial overlap is expected and is potentially a *basis for")
    add("  calibrated uncertainty* in downstream output (a candidate use, not")
    add("  a Step 2.5-3b decision).")
    add("")
    add("Full table: `reports/normalization_feasibility_overlap.csv`")
    add("")

    add("## 7. Posture as a *model input*, not just a normalizer")
    add("")
    add("Even with posture-aware normalization, posture itself carries")
    add("information about the rep (which arms-position the subject was")
    add("instructed to use). Future modeling can pass posture as a categorical")
    add("input alongside the (raw or normalized) features — that option is")
    add("explicitly left open here. Normalization and posture-as-input are not")
    add("mutually exclusive.")
    add("")

    add("## 8. What normalization is — and is not — for")
    add("")
    add("- It is **not** a substitute for the model.")
    add("- It is for: (a) keeping caption-level interpretation reasonable")
    add("  across postures, and (b) keeping feature scale stable enough that")
    add("  downstream calibration / thresholding does not depend on the")
    add("  posture distribution of the batch.")
    add("- Anything beyond that is a modeling decision, not a normalization")
    add("  decision.")
    add("")

    add("## 9. Next-step options (kept open, none chosen here)")
    add("")
    add("1. `raw` features + posture as a model input.")
    add("2. `posture_train_zscore` features + posture as a model input.")
    add("3. `posture_train_robust` features + posture as a model input.")
    add("4. `participant_zscore_upper_bound` retained **only** as a "
        "calibration-ceiling reference; not a deployable default.")
    add("")
    add("Each option needs to be evaluated against (a) class η² preservation,")
    add("(b) split robustness, (c) deployment feasibility for new users, and")
    add("(d) caption interpretability — the four lenses this audit set up.")
    add("")
    if unstable_pairs:
        add("## 10. (feature, method) pairs flagged unstable in §5")
        add("")
        for f, m in unstable_pairs:
            add(f"- `{f}` × `{m}`")
        add("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading inputs ...")
    print(f"  manifest      : {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  feature bank  : {FEATURE_BANK_PATH.relative_to(PROJECT_ROOT)}")
    bank = read_csv_dict(FEATURE_BANK_PATH)
    # Join bank rows already carry split / class / posture / participant.
    rows = [r for r in bank if r["class_id"] in CLASSES
            and r["posture_canonical"] in POSTURES]
    print(f"  loaded samples: {len(rows)}")

    out_rows = build_normalized_table(rows)
    write_features_csv(out_rows,
                       DATA_DIR / "step2" / "normalization_feasibility_features.csv")

    print("Computing effect summary ...")
    effect_rows = write_effect_summary(
        out_rows, REPORTS_DIR / "normalization_feasibility_effect_summary.csv"
    )

    print("Computing split robustness ...")
    rb_rows = write_split_robustness(
        out_rows, REPORTS_DIR / "normalization_feasibility_split_robustness.csv"
    )

    print("Computing uncertainty overlap ...")
    write_overlap(out_rows, REPORTS_DIR / "normalization_feasibility_overlap.csv")

    write_audit_md(out_rows, effect_rows, rb_rows,
                   REPORTS_DIR / "normalization_feasibility_audit.md")

    # ---- Console summary ----
    n = len(out_rows)
    n_ok = sum(1 for r in out_rows if r["normalization_compute_ok"])
    unstable = [(r["feature"], r["method"]) for r in rb_rows
                if r["flag_unstable"]]

    eff_by = {(r["feature"], r["method"]): r for r in effect_rows}

    print("")
    print("=== Step 2.5-3b normalization feasibility summary ===")
    print(f"loaded samples            : {n}")
    print(f"features tested           : {len(FEATURES)}")
    print(f"normalization methods     : {len(METHODS)}")
    print(f"normalization_compute_ok  : {n_ok} / {n}")
    print("")
    print("class η² / posture η² per (feature × method):")
    print(f"  {'feature':<32s} {'method':<32s} class_eta  post_eta  c/raw  p/raw")
    for f in FEATURES:
        for m in METHODS:
            r = eff_by[(f, m)]
            print(f"  {f:<32s} {m:<32s} "
                  f"{r['class_eta2']:.4f}     "
                  f"{r['posture_eta2']:.4f}    "
                  f"{r['class_eta2_pct_of_raw']:.3f}  "
                  f"{r['posture_eta2_pct_of_raw']:.3f}")
    print("")
    if unstable:
        print("unstable (feature, method):")
        for f, m in unstable:
            print(f"  {f}  ×  {m}")
    else:
        print("unstable (feature, method): (none)")
    print("")
    print("outputs:")
    print("  data/step2/normalization_feasibility_features.csv")
    print("  reports/normalization_feasibility_audit.md")
    print("  reports/normalization_feasibility_effect_summary.csv")
    print("  reports/normalization_feasibility_split_robustness.csv")
    print("  reports/normalization_feasibility_overlap.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
