"""Step 2.5-3 — Candidate Feature Bank Audit (v2).

This is a *candidate feature audit*, not a feature-selection commit.
It computes a small bank of bottom/event-anchored features per rep,
and reports compute success, per-cell distributions, split robustness,
and inter-class overlap. No feature is adopted or rejected here.

Inputs (read-only):
  data/manifest_split.csv
  data/step2/bottom_event_audit.csv
  rep .txt files at signal_path

Outputs:
  data/step2/candidate_feature_bank.csv
  reports/candidate_feature_bank_audit.md
  reports/candidate_feature_summary_by_class_posture.csv
  reports/candidate_feature_split_robustness.csv
  reports/candidate_feature_uncertainty_overlap.csv

Run:
  python scripts/audit_step25_feature_bank_v2.py
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
BOTTOM_PATH = DATA_DIR / "step2" / "bottom_event_audit.csv"

CLASSES = ("C1", "C2", "C3", "C4", "C5", "C6")
POSTURES = ("SA", "CA", "HW")
SPLITS = ("train", "val", "test")

SMOOTH_K = 9

# Posture-aware *candidate* anchor rule (driven by Step 2.5-2 results).
ANCHOR_TYPE_ENSEMBLE = "ensemble_acc_gyro"
ANCHOR_TYPE_ACC_ONLY = "acc_only_anchor"
POSTURE_TO_ANCHOR_TYPE = {
    "SA": ANCHOR_TYPE_ENSEMBLE,
    "CA": ANCHOR_TYPE_ENSEMBLE,
    "HW": ANCHOR_TYPE_ACC_ONLY,
}

# Window sizes (samples). The signal is ~50 Hz, ~150 samples per rep.
DEPTH_HALF = 5            # depth_proxy: anchor ±5
STAB_HALF = 10            # bottom_stability_*: anchor ±10
TRANS_HALF = 10           # bottom_transition_delta: pre / post window of length 10
RECOVERY_LO = 10          # bottom_recovery_slope: anchor + [10, 16)
RECOVERY_HI = 16
LATERAL_HALF = 10         # lateral_proxy_gyro: anchor ±10

# anchor_reliability mapping for ensemble anchors:
#   ratio = 0     -> reliability 1.0
#   ratio >= 0.30 -> reliability 0.0
RELIABILITY_RATIO_CAP = 0.30

# Confusion groups for uncertainty overlap.
UNCERTAINTY_GROUPS = {
    "C1_C5_C6": ("C1", "C5", "C6"),
    "C3_C4":    ("C3", "C4"),
}

# Column indices in rep .txt: timestamp + 6 channels.
COL_ACC_X, COL_ACC_Y, COL_ACC_Z = 1, 2, 3
COL_GYR_X, COL_GYR_Y, COL_GYR_Z = 4, 5, 6

FEATURE_NAMES = (
    "depth_proxy",
    "motion_range_acc_z",
    "motion_range_gyro_mag",
    "bottom_stability_acc",
    "bottom_stability_gyro",
    "bottom_recovery_slope_acc_z",
    "bottom_transition_delta_acc_z",
    "lateral_proxy_gyro",
    "anchor_reliability",
)

FEATURE_INTERPRETATION = {
    "depth_proxy":
        ("Mean of smoothed acc_z in [anchor±5]. Lower (relative to gravity) "
         "suggests deeper bottom or more orientation tilt at the bottom of "
         "the rep."),
    "motion_range_acc_z":
        ("Robust range (p95 − p5) of acc_z over the whole rep. Larger means "
         "the rep traverses a wider vertical-component range."),
    "motion_range_gyro_mag":
        ("Robust range (p95 − p5) of |gyro| over the whole rep. Larger means "
         "more rotational motion through the rep."),
    "bottom_stability_acc":
        ("RMS of per-axis std for (acc_x, acc_y, acc_z) in [anchor±10]. "
         "Smaller = stiller hold at bottom."),
    "bottom_stability_gyro":
        ("Std of |gyro| in [anchor±10]. Smaller = less rotational jitter at "
         "bottom."),
    "bottom_recovery_slope_acc_z":
        ("Linear slope of smoothed acc_z over [anchor+10, anchor+16). "
         "Captures post-bottom recovery dynamics; motivated by Step 2.5-2 "
         "η²(acc_z) peaking *after* the bottom."),
    "bottom_transition_delta_acc_z":
        ("Mean acc_z(post-anchor 10 samples) − mean acc_z(pre-anchor 10 "
         "samples). Signed transition magnitude across the bottom."),
    "lateral_proxy_gyro":
        ("mean(|gyro_x|) + mean(|gyro_y|) in [anchor±10]. **Weak proxy** "
         "for lateral / valgus motion: the IMU does not measure knee angle "
         "directly, so this is a heuristic, not a knee-valgus measurement."),
    "anchor_reliability":
        ("[0..1] confidence in the anchor itself. Ensemble (SA/CA): "
         "linearly decreases with bottom_agreement_ratio up to 0.30. "
         "Acc-only (HW): 1.0 if no acc warning, 0.5 if "
         "acc_min_outside_search_range was flagged."),
}

FEATURE_CAVEATS = {
    "depth_proxy":
        "Calibration-free: this is *not* mm of squat depth; only an "
        "orientation-proxy useful for relative comparisons within the same "
        "subject.",
    "motion_range_acc_z":
        "Sensor mount orientation matters; cross-subject comparison is "
        "approximate.",
    "motion_range_gyro_mag":
        "Posture changes how arms swing — values are not directly "
        "comparable across SA/CA/HW.",
    "bottom_stability_acc":
        "Smaller windows or shorter reps may bias this feature; flagged in "
        "feature_compute_warning when window is truncated.",
    "bottom_stability_gyro":
        "Sensitive to small jitters; should be read alongside "
        "anchor_reliability.",
    "bottom_recovery_slope_acc_z":
        "Sign and magnitude depend on the IMU mounting axis; the slope is "
        "interpretable in relative terms only.",
    "bottom_transition_delta_acc_z":
        "Conflates depth and orientation change. Pair with depth_proxy "
        "before drawing causal conclusions.",
    "lateral_proxy_gyro":
        "Marked **weak proxy**. A high value does not imply knee valgus; it "
        "only indicates lateral/yaw rotation around the bottom.",
    "anchor_reliability":
        "Currently a heuristic score; thresholds (0.30 cap, 0.5 fallback) "
        "are not validated.",
}

# Columns in the candidate_feature_bank.csv output.
BANK_COLS = [
    "sample_id", "participant_id", "split",
    "class_id", "posture_canonical",
    "anchor_idx", "anchor_type",
    "acc_bottom_idx", "gyro_bottom_idx", "ensemble_bottom_idx",
    "bottom_agreement_ratio", "bottom_agree_default",
    "bottom_candidate_warning",
    *FEATURE_NAMES,
    "feature_compute_ok",
    "feature_compute_warning",
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_csv_dict(path: Path) -> list[dict]:
    if not path.is_file():
        sys.exit(f"ERROR: required input not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_int(s):
    if s is None or s == "":
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


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


def load_signal(path: Path):
    try:
        arr = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except Exception:  # noqa: BLE001
        return None
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 7:
        return None
    return arr[:, :7]


def smooth_ma(x: np.ndarray, k: int = SMOOTH_K) -> np.ndarray:
    n = len(x)
    if k <= 1 or n < 2:
        return x.astype(np.float64, copy=False)
    if k > n:
        k = n
    pad = k // 2
    pad_left = np.full(pad, x[0], dtype=np.float64)
    pad_right = np.full(k - 1 - pad, x[-1], dtype=np.float64)
    padded = np.concatenate([pad_left, x.astype(np.float64, copy=False), pad_right])
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(padded, kernel, mode="valid")[:n]


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_features(signal: np.ndarray, anchor_idx: int, anchor_type: str,
                     agreement_ratio: float, warning_str: str):
    n = signal.shape[0]
    acc_x = signal[:, COL_ACC_X]
    acc_y = signal[:, COL_ACC_Y]
    acc_z = signal[:, COL_ACC_Z]
    gx = signal[:, COL_GYR_X]
    gy = signal[:, COL_GYR_Y]
    gz = signal[:, COL_GYR_Z]
    acc_z_s = smooth_ma(acc_z, SMOOTH_K)
    gyro_mag = np.sqrt(gx * gx + gy * gy + gz * gz)

    feats: dict[str, float] = {name: float("nan") for name in FEATURE_NAMES}
    warns: list[str] = []

    a = anchor_idx
    if not (0 <= a < n):
        warns.append("anchor_out_of_range")
        # Still compute rep-wide features (motion_range_*) and reliability.

    # --- depth_proxy ---
    if 0 <= a < n:
        lo, hi = max(0, a - DEPTH_HALF), min(n, a + DEPTH_HALF + 1)
        if hi - lo >= 3:
            feats["depth_proxy"] = float(acc_z_s[lo:hi].mean())
        else:
            warns.append("depth_window_too_small")

    # --- motion_range_acc_z ---
    if n >= 5:
        feats["motion_range_acc_z"] = float(
            np.percentile(acc_z, 95) - np.percentile(acc_z, 5)
        )
    else:
        warns.append("motion_range_acc_z_too_short")

    # --- motion_range_gyro_mag ---
    if n >= 5:
        feats["motion_range_gyro_mag"] = float(
            np.percentile(gyro_mag, 95) - np.percentile(gyro_mag, 5)
        )
    else:
        warns.append("motion_range_gyro_mag_too_short")

    # --- bottom_stability_acc / gyro ---
    if 0 <= a < n:
        lo, hi = max(0, a - STAB_HALF), min(n, a + STAB_HALF + 1)
        if hi - lo >= 5:
            acc_block = np.stack([acc_x[lo:hi], acc_y[lo:hi], acc_z[lo:hi]], axis=1)
            feats["bottom_stability_acc"] = float(
                np.sqrt(np.mean(np.var(acc_block, axis=0)))
            )
            feats["bottom_stability_gyro"] = float(np.std(gyro_mag[lo:hi]))
        else:
            warns.append("stability_window_too_small")

    # --- bottom_recovery_slope_acc_z ---
    if 0 <= a < n:
        lo, hi = a + RECOVERY_LO, a + RECOVERY_HI
        if 0 <= lo < n and lo < hi:
            hi_eff = min(hi, n)
            seg = acc_z_s[lo:hi_eff]
            if len(seg) >= 3:
                x_axis = np.arange(len(seg), dtype=np.float64)
                slope, _ = np.polyfit(x_axis, seg, 1)
                feats["bottom_recovery_slope_acc_z"] = float(slope)
            else:
                warns.append("recovery_window_too_small")
        else:
            warns.append("recovery_window_oob")

    # --- bottom_transition_delta_acc_z ---
    if 0 <= a < n:
        pre_lo = max(0, a - TRANS_HALF)
        pre_hi = a
        post_lo = a + 1
        post_hi = min(n, a + TRANS_HALF + 1)
        if pre_hi - pre_lo >= 3 and post_hi - post_lo >= 3:
            pre_mean = float(acc_z_s[pre_lo:pre_hi].mean())
            post_mean = float(acc_z_s[post_lo:post_hi].mean())
            feats["bottom_transition_delta_acc_z"] = post_mean - pre_mean
        else:
            warns.append("transition_window_too_small")

    # --- lateral_proxy_gyro ---
    if 0 <= a < n:
        lo, hi = max(0, a - LATERAL_HALF), min(n, a + LATERAL_HALF + 1)
        if hi - lo >= 3:
            feats["lateral_proxy_gyro"] = float(
                np.mean(np.abs(gx[lo:hi])) + np.mean(np.abs(gy[lo:hi]))
            )
        else:
            warns.append("lateral_window_too_small")

    # --- anchor_reliability ---
    has_acc_warning = "acc_min_outside_search_range" in (warning_str or "")
    if anchor_type == ANCHOR_TYPE_ENSEMBLE:
        if math.isnan(agreement_ratio):
            feats["anchor_reliability"] = float("nan")
            warns.append("agreement_ratio_missing")
        else:
            feats["anchor_reliability"] = max(
                0.0, 1.0 - agreement_ratio / RELIABILITY_RATIO_CAP
            )
    else:  # ACC_ONLY
        feats["anchor_reliability"] = 0.5 if has_acc_warning else 1.0

    compute_ok = all(not math.isnan(feats[n]) for n in FEATURE_NAMES)
    return feats, warns, compute_ok


# ---------------------------------------------------------------------------
# Bank build
# ---------------------------------------------------------------------------

def build_bank():
    print(f"Loading manifest      : {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    manifest = read_csv_dict(MANIFEST_PATH)
    manifest = [r for r in manifest
                if r.get("include", "True") != "False"
                and r["class_id"] in CLASSES
                and r["posture_canonical"] in POSTURES]
    print(f"  manifest rows       : {len(manifest)}")

    print(f"Loading bottom audit  : {BOTTOM_PATH.relative_to(PROJECT_ROOT)}")
    bottom = read_csv_dict(BOTTOM_PATH)
    bottom_by_id = {r["sample_id"]: r for r in bottom}
    print(f"  bottom audit rows   : {len(bottom)}")

    bank_rows: list[dict] = []
    failed = 0
    missing_bottom = 0

    for i, m in enumerate(manifest):
        sid = m["sample_id"]
        b = bottom_by_id.get(sid)
        if b is None:
            missing_bottom += 1
            continue

        sig = load_signal(Path(m["signal_path"]))
        if sig is None or sig.shape[0] < 5:
            failed += 1
            continue

        posture = m["posture_canonical"]
        anchor_type = POSTURE_TO_ANCHOR_TYPE[posture]
        if anchor_type == ANCHOR_TYPE_ENSEMBLE:
            anchor_idx = parse_int(b["ensemble_bottom_idx"])
        else:
            anchor_idx = parse_int(b["acc_bottom_idx"])
        if anchor_idx is None:
            failed += 1
            continue

        agreement_ratio = parse_float(b["bottom_agreement_ratio"])
        if agreement_ratio is None:
            agreement_ratio = float("nan")
        warning_str = b.get("bottom_candidate_warning", "") or ""

        feats, warns, compute_ok = compute_features(
            sig, anchor_idx, anchor_type, agreement_ratio, warning_str
        )

        row = {
            "sample_id": sid,
            "participant_id": m["participant_id"],
            "split": m["split"],
            "class_id": m["class_id"],
            "posture_canonical": posture,
            "anchor_idx": anchor_idx,
            "anchor_type": anchor_type,
            "acc_bottom_idx": parse_int(b["acc_bottom_idx"]),
            "gyro_bottom_idx": parse_int(b["gyro_bottom_idx"]),
            "ensemble_bottom_idx": parse_int(b["ensemble_bottom_idx"]),
            "bottom_agreement_ratio": agreement_ratio
                if not math.isnan(agreement_ratio) else "",
            "bottom_agree_default": b.get("bottom_agree_default", ""),
            "bottom_candidate_warning": warning_str,
            **feats,
            "feature_compute_ok": compute_ok,
            "feature_compute_warning": ";".join(warns),
        }
        bank_rows.append(row)

        if (i + 1) % 1000 == 0:
            print(f"  processed {i + 1}/{len(manifest)} reps")

    if missing_bottom:
        print(f"  WARNING: {missing_bottom} manifest rows had no bottom-audit "
              f"counterpart; skipped")
    if failed:
        print(f"  WARNING: {failed} reps failed to load / compute; skipped")
    return bank_rows


def write_bank_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(BANK_COLS)
        for r in rows:
            out = []
            for c in BANK_COLS:
                v = r.get(c, "")
                if v is None:
                    out.append("")
                elif isinstance(v, float):
                    if math.isnan(v) or math.isinf(v):
                        out.append("")
                    else:
                        out.append(f"{v:.6f}")
                elif isinstance(v, bool):
                    out.append("True" if v else "False")
                else:
                    out.append(str(v))
            w.writerow(out)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def feature_array(rows, feature_name):
    vals = []
    for r in rows:
        v = r.get(feature_name)
        if v is None or v == "" or (isinstance(v, float)
                                     and (math.isnan(v) or math.isinf(v))):
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    return np.asarray(vals, dtype=np.float64)


def basic_stats(arr: np.ndarray):
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "median": float("nan"), "p25": float("nan"), "p75": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def write_summary_by_class_posture(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "posture_canonical", "feature",
                    "n", "n_nan", "mean", "std", "median", "p25", "p75"])
        for c in CLASSES:
            for p in POSTURES:
                cell = [r for r in rows
                        if r["class_id"] == c and r["posture_canonical"] == p]
                n_total = len(cell)
                for fname in FEATURE_NAMES:
                    arr = feature_array(cell, fname)
                    s = basic_stats(arr)
                    w.writerow([
                        c, p, fname,
                        s["n"], n_total - s["n"],
                        f"{s['mean']:.6f}" if s["n"] else "",
                        f"{s['std']:.6f}" if s["n"] else "",
                        f"{s['median']:.6f}" if s["n"] else "",
                        f"{s['p25']:.6f}" if s["n"] else "",
                        f"{s['p75']:.6f}" if s["n"] else "",
                    ])


def write_split_robustness(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "feature",
            "train_n", "train_mean", "train_std",
            "val_n", "val_mean", "val_std",
            "test_n", "test_mean", "test_std",
            "mean_range", "rel_mean_range", "flag_unstable",
        ])
        for fname in FEATURE_NAMES:
            stats_per_split = {}
            for sp in SPLITS:
                arr = feature_array([r for r in rows if r["split"] == sp], fname)
                stats_per_split[sp] = basic_stats(arr)
            means = [stats_per_split[sp]["mean"] for sp in SPLITS
                     if stats_per_split[sp]["n"] > 0
                     and not math.isnan(stats_per_split[sp]["mean"])]
            if len(means) >= 2:
                mean_range = max(means) - min(means)
                base = abs(np.mean(means))
                rel = mean_range / base if base > 1e-12 else float("inf")
            else:
                mean_range = float("nan")
                rel = float("nan")
            unstable = (not math.isnan(rel)) and (rel > 0.20)
            row = [fname]
            for sp in SPLITS:
                s = stats_per_split[sp]
                row += [s["n"],
                        f"{s['mean']:.6f}" if s["n"] else "",
                        f"{s['std']:.6f}" if s["n"] else ""]
            row += [
                f"{mean_range:.6f}" if not math.isnan(mean_range) else "",
                f"{rel:.6f}" if not (math.isnan(rel) or math.isinf(rel)) else "",
                "True" if unstable else "False",
            ]
            w.writerow(row)


def normal_overlap(d_abs):
    """Overlap coefficient of two unit-variance normals separated by |d|."""
    if math.isnan(d_abs) or math.isinf(d_abs):
        return float("nan")
    z = -d_abs / 2.0
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return 2.0 * cdf


def write_uncertainty_overlap(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "feature", "posture",
                    "class_a", "class_b",
                    "n_a", "n_b",
                    "mean_a", "mean_b", "std_a", "std_b",
                    "cohen_d", "overlap_estimate"])
        for group_name, group_classes in UNCERTAINTY_GROUPS.items():
            pairs = []
            for i in range(len(group_classes)):
                for j in range(i + 1, len(group_classes)):
                    pairs.append((group_classes[i], group_classes[j]))
            posture_scopes = ["all", *POSTURES]
            for fname in FEATURE_NAMES:
                for posture in posture_scopes:
                    if posture == "all":
                        scope_rows = rows
                    else:
                        scope_rows = [r for r in rows
                                      if r["posture_canonical"] == posture]
                    for ca, cb in pairs:
                        arr_a = feature_array(
                            [r for r in scope_rows if r["class_id"] == ca], fname
                        )
                        arr_b = feature_array(
                            [r for r in scope_rows if r["class_id"] == cb], fname
                        )
                        if arr_a.size == 0 or arr_b.size == 0:
                            w.writerow([group_name, fname, posture, ca, cb,
                                        int(arr_a.size), int(arr_b.size),
                                        "", "", "", "", "", ""])
                            continue
                        m_a = float(arr_a.mean())
                        m_b = float(arr_b.mean())
                        s_a = float(arr_a.std(ddof=0))
                        s_b = float(arr_b.std(ddof=0))
                        pooled = math.sqrt((s_a * s_a + s_b * s_b) / 2.0)
                        if pooled > 1e-12:
                            d = (m_a - m_b) / pooled
                        else:
                            d = float("nan")
                        ov = normal_overlap(abs(d)) if not math.isnan(d) else float("nan")
                        w.writerow([group_name, fname, posture, ca, cb,
                                    int(arr_a.size), int(arr_b.size),
                                    f"{m_a:.6f}", f"{m_b:.6f}",
                                    f"{s_a:.6f}", f"{s_b:.6f}",
                                    f"{d:.6f}" if not math.isnan(d) else "",
                                    f"{ov:.6f}" if not math.isnan(ov) else ""])


def write_audit_md(rows, missing_count, path: Path):
    n = len(rows)
    n_ok = sum(1 for r in rows if r["feature_compute_ok"])

    # Compute success per posture and class.
    by_posture = defaultdict(lambda: [0, 0])  # [n_total, n_ok]
    by_class = defaultdict(lambda: [0, 0])
    for r in rows:
        by_posture[r["posture_canonical"]][0] += 1
        by_class[r["class_id"]][0] += 1
        if r["feature_compute_ok"]:
            by_posture[r["posture_canonical"]][1] += 1
            by_class[r["class_id"]][1] += 1

    anchor_type_counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        anchor_type_counts[r["posture_canonical"]][r["anchor_type"]] += 1

    feature_nan = {}
    for fname in FEATURE_NAMES:
        nan_count = 0
        for r in rows:
            v = r.get(fname)
            if v == "" or v is None or (isinstance(v, float)
                                         and (math.isnan(v) or math.isinf(v))):
                nan_count += 1
        feature_nan[fname] = nan_count

    lines: list[str] = []
    add = lines.append
    add("# Step 2.5-3 — Candidate Feature Bank Audit")
    add("")
    add("**Scope of this script.** This is a candidate-feature *audit*, not a")
    add("feature-selection commit. Features are computed and described here so")
    add("they can be evaluated downstream. Nothing on this list is adopted or")
    add("rejected by this run.")
    add("")

    add("## 1. Posture-aware candidate anchor rule")
    add("")
    add("Driven by Step 2.5-2 results (HW posture had very low bottom")
    add("agreement under the gyro-magnitude peak rule).")
    add("")
    add("| posture | anchor source | anchor_type |")
    add("|---|---|---|")
    add("| SA | `ensemble_bottom_idx` (mean of acc/gyro candidates) | "
        f"`{ANCHOR_TYPE_ENSEMBLE}` |")
    add("| CA | `ensemble_bottom_idx` (mean of acc/gyro candidates) | "
        f"`{ANCHOR_TYPE_ENSEMBLE}` |")
    add("| HW | `acc_bottom_idx` (acc_z minimum only)               | "
        f"`{ANCHOR_TYPE_ACC_ONLY}` |")
    add("")
    add("**This rule is provisional**, not the final modeling rule. It is")
    add("used here only to compute candidate features in a way that does not")
    add("collapse under HW's weak gyro-peak signal.")
    add("")
    add("Reasoning recap:")
    add("- SA / CA had cell-level bottom_agreement_ratio ≤ 0.10 in 70–88% of")
    add("  reps, so the ensemble of acc/gyro candidates is stable.")
    add("- HW had agreement only 4–22% with the same rule. The acc_z minimum")
    add("  alone is more reliable for HW because hands resting on the waist")
    add("  decouple |gyro| peaks from the squat bottom.")
    add("")

    add("## 2. Candidate features")
    add("")
    add("| feature | interpretation | caveat |")
    add("|---|---|---|")
    for fname in FEATURE_NAMES:
        interp = FEATURE_INTERPRETATION[fname].replace("|", "\\|")
        caveat = FEATURE_CAVEATS[fname].replace("|", "\\|")
        add(f"| `{fname}` | {interp} | {caveat} |")
    add("")
    add("**On `bottom_recovery_slope_acc_z`.** Step 2.5-2 reported that "
        "η²(acc_z) for SA and CA peaked *near or after* the bottom (timesteps")
    add("≈ 86 of 128). That is a hint that *post-bottom recovery dynamics* —")
    add("not just bottom depth — carry class signal. The slope feature is a")
    add("first stab at exposing that to downstream evaluation.")
    add("")

    add("## 3. Compute success")
    add("")
    add(f"- Manifest rows missing from bottom audit (skipped): {missing_count}")
    add(f"- Bank rows (feature attempts): **{n}**")
    add(f"- Rows with `feature_compute_ok = True`: **{n_ok}** "
        f"({(n_ok / n * 100) if n else 0:.2f}%)")
    add("")
    add("Per posture:")
    add("")
    add("| posture | rows | compute_ok | rate |")
    add("|---|---|---|---|")
    for p in POSTURES:
        tot, ok = by_posture[p]
        rate = (ok / tot) if tot else 0.0
        add(f"| {p} | {tot} | {ok} | {rate:.4f} |")
    add("")
    add("Per class:")
    add("")
    add("| class | rows | compute_ok | rate |")
    add("|---|---|---|---|")
    for c in CLASSES:
        tot, ok = by_class[c]
        rate = (ok / tot) if tot else 0.0
        add(f"| {c} | {tot} | {ok} | {rate:.4f} |")
    add("")

    add("## 4. NaN counts per feature")
    add("")
    add("| feature | n_nan | rate |")
    add("|---|---|---|")
    for fname in FEATURE_NAMES:
        nan = feature_nan[fname]
        rate = (nan / n) if n else 0.0
        add(f"| `{fname}` | {nan} | {rate:.4f} |")
    add("")

    add("## 5. Anchor-type distribution")
    add("")
    add("| posture | anchor_type | count |")
    add("|---|---|---|")
    for p in POSTURES:
        for at, cnt in anchor_type_counts[p].items():
            add(f"| {p} | `{at}` | {cnt} |")
    add("")

    add("## 6. Distribution / robustness / overlap (cross-references)")
    add("")
    add("- `reports/candidate_feature_summary_by_class_posture.csv` — per "
        "(class × posture × feature) `n`, `n_nan`, `mean`, `std`, `median`, "
        "`p25`, `p75`.")
    add("- `reports/candidate_feature_split_robustness.csv` — per feature, "
        "`(train, val, test)` mean/std and `flag_unstable` (set when "
        "max-min mean range exceeds 20% of |overall mean|).")
    add("- `reports/candidate_feature_uncertainty_overlap.csv` — per "
        "confusion group / feature / posture-scope: pairwise Cohen's *d* "
        "and an overlap estimate (Gaussian approximation, useful as a *signal*"
        " not a probability).")
    add("")
    add("Confusion groups examined:")
    add("")
    for gname, classes in UNCERTAINTY_GROUPS.items():
        add(f"- `{gname}`: {', '.join(classes)}")
    add("")
    add("**Why we keep features that overlap heavily.** Class C1/C5/C6 and "
        "C3/C4 are expected to overlap by construction (they share a posture-")
    add("and-task structure with subtle differences). High overlap on a")
    add("feature is *not a reason to drop it*; it is potentially a basis for")
    add("calibrated uncertainty in caption/output, which Step 2.5-3 only flags")
    add("as a candidate use case for downstream consideration.")
    add("")

    add("## 7. Open questions / things this audit explicitly does NOT decide")
    add("")
    add("- Which features survive into modeling.")
    add("- Whether `lateral_proxy_gyro` (a weak proxy) should be dropped or")
    add("  kept as an uncertainty-only feature.")
    add("- Whether `anchor_reliability` thresholds are correctly calibrated.")
    add("- Whether the HW acc-only fallback should be replaced by a different")
    add("  anchoring strategy entirely.")
    add("- Whether per-feature normalization (per-subject / per-posture) is")
    add("  needed before any downstream use.")
    add("")
    add("These are all explicitly **next-step** decisions, not Step 2.5-3 ones.")
    add("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Building candidate feature bank ...")
    # Re-read manifest count for the missing-from-audit gap computation.
    manifest = read_csv_dict(MANIFEST_PATH)
    n_manifest = sum(1 for r in manifest
                     if r.get("include", "True") != "False"
                     and r["class_id"] in CLASSES
                     and r["posture_canonical"] in POSTURES)

    rows = build_bank()
    missing = max(0, n_manifest - len(rows))

    print("Writing bank CSV ...")
    write_bank_csv(rows, DATA_DIR / "step2" / "candidate_feature_bank.csv")

    print("Writing class × posture summary CSV ...")
    write_summary_by_class_posture(
        rows, REPORTS_DIR / "candidate_feature_summary_by_class_posture.csv"
    )

    print("Writing split robustness CSV ...")
    write_split_robustness(
        rows, REPORTS_DIR / "candidate_feature_split_robustness.csv"
    )

    print("Writing uncertainty overlap CSV ...")
    write_uncertainty_overlap(
        rows, REPORTS_DIR / "candidate_feature_uncertainty_overlap.csv"
    )

    print("Writing audit report ...")
    write_audit_md(rows, missing, REPORTS_DIR / "candidate_feature_bank_audit.md")

    # ---- Console summary ----
    n = len(rows)
    n_ok = sum(1 for r in rows if r["feature_compute_ok"])
    n_fail = n - n_ok

    posture_anchor: dict = defaultdict(lambda: defaultdict(int))
    for r in rows:
        posture_anchor[r["posture_canonical"]][r["anchor_type"]] += 1

    nan_counts: dict = {}
    for fname in FEATURE_NAMES:
        nan_counts[fname] = sum(
            1 for r in rows
            if r.get(fname) == "" or r.get(fname) is None
            or (isinstance(r.get(fname), float)
                and (math.isnan(r.get(fname)) or math.isinf(r.get(fname))))
        )

    print("")
    print("=== Step 2.5-3 candidate feature bank summary ===")
    print(f"loaded samples (manifest scope) : {n_manifest}")
    print(f"feature bank rows               : {n}")
    print(f"feature_compute_ok              : {n_ok}")
    print(f"feature_compute_failures (any NaN): {n_fail}")
    print("")
    print("posture × anchor_type:")
    for p in POSTURES:
        for at, cnt in posture_anchor[p].items():
            print(f"  {p:3s} × {at:18s} : {cnt}")
    print("")
    print("NaN counts per feature:")
    for fname in FEATURE_NAMES:
        print(f"  {fname:32s} : {nan_counts[fname]}")
    print("")
    print("outputs:")
    print(f"  data/step2/candidate_feature_bank.csv")
    print(f"  reports/candidate_feature_bank_audit.md")
    print(f"  reports/candidate_feature_summary_by_class_posture.csv")
    print(f"  reports/candidate_feature_split_robustness.csv")
    print(f"  reports/candidate_feature_uncertainty_overlap.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
