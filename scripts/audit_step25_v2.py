"""Step 2.5 audit (v2) — first-pass diagnostics, no model commitments yet.

This script is *investigative*: it summarizes what the v2 manifest looks
like, where class differences live in time, whether a "bottom" event is a
stable candidate anchor, and how sensitive bottom agreement is to the
agreement threshold. It does NOT decide features, anchors, or captions.

Inputs (read-only):
  data/manifest_split.csv

Outputs:
  reports/step25_preflight_summary.md
  reports/time_localized_class_difference_summary.md
  reports/bottom_event_candidate_audit.md
  reports/bottom_agreement_threshold_sensitivity.csv
  data/step2/time_localized_class_difference.csv
  data/step2/bottom_event_audit.csv

Run:
  python scripts/audit_step25_v2.py
"""

from __future__ import annotations

import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Config (locked by Step 1 + Step 2 spec; nothing here is a feature decision)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
MANIFEST_PATH = DATA_DIR / "manifest_split.csv"

CLASSES = ("C1", "C2", "C3", "C4", "C5", "C6")
POSTURES = ("SA", "CA", "HW")
CHANNEL_NAMES = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
N_CHANNELS = len(CHANNEL_NAMES)

# Resampling / smoothing.
RESAMPLE_LEN = 128
SMOOTH_K = 9

# Bottom search range on the ORIGINAL (non-resampled) signal.
BOTTOM_SEARCH_LO = 0.25
BOTTOM_SEARCH_HI = 0.75

# Bottom-region window on the RESAMPLED signal, used for class-difference
# localization. Middle 40% of the resampled timestep axis.
BOTTOM_REGION_LO = 0.30
BOTTOM_REGION_HI = 0.70

# Initial bottom-agreement threshold (NOT final — sensitivity table is
# generated separately).
DEFAULT_AGREE_THRESHOLD = 0.10
THRESHOLD_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)

# Column indices in the rep .txt file (timestamp + 6 sensor channels).
COL_TIMESTAMP = 0
COL_ACC_X, COL_ACC_Y, COL_ACC_Z = 1, 2, 3
COL_GYR_X, COL_GYR_Y, COL_GYR_Z = 4, 5, 6
SENSOR_COLS = slice(1, 7)  # 6 channels after dropping timestamp.

# GO / WEAK-GO / NO-GO heuristics.
GO_AGREE_RATE = 0.70
WEAK_AGREE_RATE = 0.50
GO_CELL_MAJORITY = 10  # >= 10 / 18 cells qualifies as "majority"

# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

MANIFEST_NEEDED_COLS = (
    "sample_id", "rep_id", "participant_id",
    "class_id", "posture_canonical", "posture_folder",
    "split", "signal_path", "n_rows", "duration_s",
    "boundary_ok", "length_ok", "include",
)


def read_manifest():
    """Yield manifest rows (dicts) restricted to clean+in-scope samples."""
    if not MANIFEST_PATH.is_file():
        sys.exit(f"ERROR: manifest not found: {MANIFEST_PATH}")
    rows = []
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("include", "True") == "False":
                continue
            if r["class_id"] not in CLASSES or r["posture_canonical"] not in POSTURES:
                continue
            rows.append(r)
    return rows


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


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def load_signal(path):
    """Load a rep .txt file as a (n_rows, 7) float array. Returns None on failure."""
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
    """Edge-padded moving-average smoother. Output has the same length as x."""
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


def resample_to_length(arr2d: np.ndarray, target_n: int = RESAMPLE_LEN) -> np.ndarray:
    """Linearly resample a (n, c) array to (target_n, c) along axis 0."""
    n = arr2d.shape[0]
    if n == target_n:
        return arr2d.astype(np.float64, copy=False)
    src = np.linspace(0.0, 1.0, n)
    tgt = np.linspace(0.0, 1.0, target_n)
    out = np.empty((target_n, arr2d.shape[1]), dtype=np.float64)
    for c in range(arr2d.shape[1]):
        out[:, c] = np.interp(tgt, src, arr2d[:, c].astype(np.float64))
    return out


# ---------------------------------------------------------------------------
# Bottom-event candidate detection (per rep, original signal)
# ---------------------------------------------------------------------------

def detect_bottom(signal: np.ndarray) -> dict:
    """Return acc_bottom_idx / gyro_bottom_idx / ensemble + warnings."""
    n_rows = signal.shape[0]
    start_search = int(BOTTOM_SEARCH_LO * n_rows)
    end_search = int(BOTTOM_SEARCH_HI * n_rows)
    # Guard against degenerate ranges.
    if end_search <= start_search:
        start_search = 0
        end_search = n_rows

    acc_z = signal[:, COL_ACC_Z]
    gx = signal[:, COL_GYR_X]
    gy = signal[:, COL_GYR_Y]
    gz = signal[:, COL_GYR_Z]

    acc_z_smooth = smooth_ma(acc_z, SMOOTH_K)
    gyro_mag = np.sqrt(gx * gx + gy * gy + gz * gz)
    gyro_mag_smooth = smooth_ma(gyro_mag, SMOOTH_K)

    # In-range argmin / argmax.
    acc_window = acc_z_smooth[start_search:end_search]
    gyro_window = gyro_mag_smooth[start_search:end_search]
    acc_bottom_idx = int(np.argmin(acc_window)) + start_search
    gyro_bottom_idx = int(np.argmax(gyro_window)) + start_search

    # Global-vs-window sanity warnings.
    warnings: list[str] = []
    acc_global = int(np.argmin(acc_z_smooth))
    if acc_global < start_search or acc_global >= end_search:
        warnings.append("acc_min_outside_search_range")
    gyro_global = int(np.argmax(gyro_mag_smooth))
    if gyro_global < start_search or gyro_global >= end_search:
        warnings.append("gyro_peak_outside_search_range")

    bottom_diff = abs(acc_bottom_idx - gyro_bottom_idx)
    bottom_ratio = bottom_diff / n_rows if n_rows > 0 else 0.0
    ensemble = (acc_bottom_idx + gyro_bottom_idx) // 2

    return {
        "n_rows": n_rows,
        "start_search": start_search,
        "end_search": end_search,
        "acc_bottom_idx": acc_bottom_idx,
        "gyro_bottom_idx": gyro_bottom_idx,
        "ensemble_bottom_idx": ensemble,
        "bottom_diff_samples": bottom_diff,
        "bottom_agreement_ratio": bottom_ratio,
        "bottom_agree_default": bottom_ratio <= DEFAULT_AGREE_THRESHOLD,
        "bottom_candidate_warning": ";".join(warnings),
    }


# ---------------------------------------------------------------------------
# Pass over all samples: bottom audit + class-mean accumulation
# ---------------------------------------------------------------------------

class GroupAcc:
    __slots__ = ("count", "sum", "sumsq")

    def __init__(self):
        self.count = 0
        self.sum = np.zeros((RESAMPLE_LEN, N_CHANNELS), dtype=np.float64)
        self.sumsq = np.zeros((RESAMPLE_LEN, N_CHANNELS), dtype=np.float64)


def process_samples(rows):
    """Single pass: load each rep, detect bottom, accumulate per (class, posture)."""
    bottom_records = []
    groups: dict[tuple, GroupAcc] = {}
    failed = 0
    for i, r in enumerate(rows):
        path = Path(r["signal_path"])
        sig = load_signal(path)
        if sig is None or sig.shape[0] < 2:
            failed += 1
            continue
        # Bottom audit on original signal.
        bd = detect_bottom(sig)
        bd["sample_id"] = r["sample_id"]
        bd["rep_id"] = r["rep_id"]
        bd["participant_id"] = r["participant_id"]
        bd["class_id"] = r["class_id"]
        bd["posture_canonical"] = r["posture_canonical"]
        bottom_records.append(bd)

        # Class-mean accumulation on resampled 6-channel signal.
        sig6 = sig[:, SENSOR_COLS]
        res = resample_to_length(sig6, RESAMPLE_LEN)
        key = (r["class_id"], r["posture_canonical"])
        g = groups.get(key)
        if g is None:
            g = GroupAcc()
            groups[key] = g
        g.count += 1
        g.sum += res
        g.sumsq += res * res

        if (i + 1) % 1000 == 0:
            print(f"  processed {i + 1}/{len(rows)} reps")

    if failed:
        print(f"  WARNING: {failed} reps failed to load and were skipped")
    return bottom_records, groups


# ---------------------------------------------------------------------------
# Time-localized class difference (eta^2 per posture × channel × timestep)
# ---------------------------------------------------------------------------

def compute_eta_squared(groups):
    """Return dict posture -> ndarray(RESAMPLE_LEN, N_CHANNELS) of eta^2 in [0,1]."""
    eta2_per_posture = {}
    for posture in POSTURES:
        n_total = 0
        s_total = np.zeros((RESAMPLE_LEN, N_CHANNELS), dtype=np.float64)
        ss_total = np.zeros((RESAMPLE_LEN, N_CHANNELS), dtype=np.float64)
        class_means: dict[str, np.ndarray] = {}
        class_counts: dict[str, int] = {}
        for c in CLASSES:
            g = groups.get((c, posture))
            if g is None or g.count == 0:
                continue
            n_total += g.count
            s_total += g.sum
            ss_total += g.sumsq
            class_means[c] = g.sum / g.count
            class_counts[c] = g.count
        if n_total == 0:
            eta2_per_posture[posture] = np.zeros(
                (RESAMPLE_LEN, N_CHANNELS), dtype=np.float64
            )
            continue
        grand_mean = s_total / n_total
        sst = ss_total - n_total * grand_mean * grand_mean
        ssb = np.zeros((RESAMPLE_LEN, N_CHANNELS), dtype=np.float64)
        for c, mu_c in class_means.items():
            diff = mu_c - grand_mean
            ssb += class_counts[c] * diff * diff
        with np.errstate(divide="ignore", invalid="ignore"):
            eta2 = np.where(sst > 0, ssb / sst, 0.0)
        eta2 = np.clip(eta2, 0.0, 1.0)
        eta2_per_posture[posture] = eta2
    return eta2_per_posture


def write_eta_csv(eta2_per_posture, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["posture", "channel", "timestep", "eta_squared"])
        for posture in POSTURES:
            arr = eta2_per_posture.get(posture)
            if arr is None:
                continue
            for t in range(RESAMPLE_LEN):
                for ci, ch in enumerate(CHANNEL_NAMES):
                    w.writerow([posture, ch, t, f"{arr[t, ci]:.6f}"])


# ---------------------------------------------------------------------------
# Bottom-event audit CSV + threshold sensitivity
# ---------------------------------------------------------------------------

BOTTOM_CSV_COLS = [
    "sample_id", "rep_id", "participant_id", "class_id", "posture_canonical",
    "n_rows", "start_search", "end_search",
    "acc_bottom_idx", "gyro_bottom_idx", "ensemble_bottom_idx",
    "bottom_diff_samples", "bottom_agreement_ratio",
    "bottom_agree_default", "bottom_candidate_warning",
]


def write_bottom_csv(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(BOTTOM_CSV_COLS)
        for r in records:
            w.writerow([
                r["sample_id"], r["rep_id"], r["participant_id"],
                r["class_id"], r["posture_canonical"],
                r["n_rows"], r["start_search"], r["end_search"],
                r["acc_bottom_idx"], r["gyro_bottom_idx"], r["ensemble_bottom_idx"],
                r["bottom_diff_samples"],
                f"{r['bottom_agreement_ratio']:.6f}",
                "True" if r["bottom_agree_default"] else "False",
                r["bottom_candidate_warning"],
            ])


def threshold_sensitivity(records, path: Path):
    ratios = np.array([r["bottom_agreement_ratio"] for r in records], dtype=np.float64)
    classes = np.array([r["class_id"] for r in records])
    postures = np.array([r["posture_canonical"] for r in records])

    rows_out = []
    for thr in THRESHOLD_GRID:
        agree = ratios <= thr
        # overall
        n = len(agree)
        n_agree = int(agree.sum())
        rows_out.append(("overall", "all",
                         thr, n, n_agree,
                         (n_agree / n) if n else 0.0))
        # per class
        for c in CLASSES:
            mask = classes == c
            n_c = int(mask.sum())
            n_a = int(agree[mask].sum())
            rows_out.append(("class", c, thr, n_c, n_a,
                             (n_a / n_c) if n_c else 0.0))
        # per posture
        for p in POSTURES:
            mask = postures == p
            n_p = int(mask.sum())
            n_a = int(agree[mask].sum())
            rows_out.append(("posture", p, thr, n_p, n_a,
                             (n_a / n_p) if n_p else 0.0))
        # per (class × posture)
        for c in CLASSES:
            for p in POSTURES:
                mask = (classes == c) & (postures == p)
                n_cp = int(mask.sum())
                n_a = int(agree[mask].sum())
                rows_out.append(("class_x_posture", f"{c}_{p}",
                                 thr, n_cp, n_a,
                                 (n_a / n_cp) if n_cp else 0.0))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", "scope_value", "threshold",
                    "n_samples", "n_agree", "agree_rate"])
        for scope, scope_val, thr, n, na, rate in rows_out:
            w.writerow([scope, scope_val, f"{thr:.2f}", n, na, f"{rate:.4f}"])


# ---------------------------------------------------------------------------
# Reports (markdown)
# ---------------------------------------------------------------------------

def percentile(values, q):
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def write_preflight_md(rows, bottom_records, path: Path):
    n = len(rows)
    by_split_p: dict = defaultdict(set)
    by_split_s: dict = defaultdict(int)
    for r in rows:
        by_split_p[r["split"]].add(r["participant_id"])
        by_split_s[r["split"]] += 1

    cps_count: dict = defaultdict(int)
    for r in rows:
        cps_count[(r["class_id"], r["posture_canonical"], r["split"])] += 1

    n_rows_vals = [parse_int(r["n_rows"]) for r in rows]
    n_rows_vals = [v for v in n_rows_vals if v is not None]
    dur_vals = [parse_float(r["duration_s"]) for r in rows]
    dur_vals = [v for v in dur_vals if v is not None]

    n_boundary_bad = sum(1 for r in rows if r.get("boundary_ok") == "False")
    n_length_bad = sum(1 for r in rows if r.get("length_ok") == "False")

    lines: list[str] = []
    add = lines.append
    add("# Step 2.5 — Preflight Sanity Summary")
    add("")
    add(f"- Manifest: `{MANIFEST_PATH.relative_to(PROJECT_ROOT)}`")
    add(f"- Loaded sample rows: **{n}**")
    add(f"- Bottom-audit records produced: **{len(bottom_records)}**")
    add("")

    add("## 1. Split sizes")
    add("")
    add("| split | participants | samples |")
    add("|---|---|---|")
    for sp in ("train", "val", "test"):
        add(f"| {sp} | {len(by_split_p[sp])} | {by_split_s[sp]} |")
    add("")

    add("## 2. Class × posture × split sample counts")
    add("")
    add("| class | posture | train | val | test | total |")
    add("|---|---|---|---|---|---|")
    for c in CLASSES:
        for p in POSTURES:
            tr = cps_count[(c, p, "train")]
            va = cps_count[(c, p, "val")]
            te = cps_count[(c, p, "test")]
            add(f"| {c} | {p} | {tr} | {va} | {te} | {tr + va + te} |")
    add("")

    add("## 3. n_rows / duration_s basic stats")
    add("")
    add("| field | n | min | p25 | p50 | mean | p75 | max |")
    add("|---|---|---|---|---|---|---|---|")

    def stat_row(name, vals, fmt):
        if not vals:
            add(f"| {name} | 0 | – | – | – | – | – | – |")
            return
        arr = np.asarray(vals, dtype=np.float64)
        add(f"| {name} | {len(vals)} | {fmt(arr.min())} | "
            f"{fmt(np.percentile(arr, 25))} | {fmt(np.percentile(arr, 50))} | "
            f"{fmt(arr.mean())} | {fmt(np.percentile(arr, 75))} | {fmt(arr.max())} |")

    stat_row("n_rows", n_rows_vals, lambda v: f"{int(round(v))}")
    stat_row("duration_s", dur_vals, lambda v: f"{v:.3f}")
    add("")

    add("## 4. Soft-flag counts")
    add("")
    add(f"- `boundary_ok=False`: {n_boundary_bad}")
    add(f"- `length_ok=False`  : {n_length_bad}")
    add("")
    add("These rows are kept in the manifest (soft flags only). Step 2.5")
    add("audits operate on the resampled signal, so missing index bounds do")
    add("not block bottom-event detection.")
    add("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_class_diff_md(eta2_per_posture, path: Path):
    bot_lo = int(BOTTOM_REGION_LO * RESAMPLE_LEN)
    bot_hi = int(BOTTOM_REGION_HI * RESAMPLE_LEN)

    # Per (posture, channel) summary.
    summary_rows = []
    bottom_higher_count = 0
    cell_total = 0
    for posture in POSTURES:
        arr = eta2_per_posture.get(posture)
        if arr is None:
            continue
        for ci, ch in enumerate(CHANNEL_NAMES):
            col = arr[:, ci]
            mean_full = float(col.mean())
            mean_bottom = float(col[bot_lo:bot_hi].mean())
            argmax_t = int(np.argmax(col))
            eta_max = float(col[argmax_t])
            higher = mean_bottom > mean_full
            if higher:
                bottom_higher_count += 1
            cell_total += 1
            summary_rows.append({
                "posture": posture,
                "channel": ch,
                "argmax_t": argmax_t,
                "eta_max": eta_max,
                "mean_full": mean_full,
                "mean_bottom": mean_bottom,
                "bottom_higher": higher,
            })

    lines: list[str] = []
    add = lines.append
    add("# Step 2.5 — Time-localized Class Difference Audit")
    add("")
    add("Goal: identify *where in time* class differences (C1–C6) are")
    add("most pronounced. This is descriptive, not a feature decision.")
    add("")
    add("## Method")
    add("")
    add(f"- Each rep linearly resampled to length {RESAMPLE_LEN}.")
    add(f"- Channels (timestamp dropped): {', '.join(CHANNEL_NAMES)}.")
    add("- Per (posture, channel, timestep): "
        "η² = SSB / SST across classes C1–C6.")
    add(f"- Bottom region: timesteps [{bot_lo}, {bot_hi}) "
        f"= middle {int((BOTTOM_REGION_HI - BOTTOM_REGION_LO) * 100)}%.")
    add(f"- Full timeline: timesteps [0, {RESAMPLE_LEN}).")
    add("")

    add("## Per-cell summary")
    add("")
    add("| posture | channel | argmax_t (η²) | η²_max | mean η² (full) | mean η² (bottom) | bottom > full? |")
    add("|---|---|---|---|---|---|---|")
    for r in summary_rows:
        add(f"| {r['posture']} | {r['channel']} | {r['argmax_t']} | "
            f"{r['eta_max']:.4f} | {r['mean_full']:.4f} | "
            f"{r['mean_bottom']:.4f} | "
            f"{'**yes**' if r['bottom_higher'] else 'no'} |")
    add("")

    add("## Bottom-region concentration")
    add("")
    add(f"- {bottom_higher_count} / {cell_total} (posture × channel) cells "
        f"have higher mean η² in the bottom region than across the full timeline.")
    add("- Argmax timesteps clustered in the bottom region suggest that")
    add("  bottom-anchored features are likely to capture class differences.")
    add("- This is the descriptive signal Step 2.5 is checking; no")
    add("  feature is being selected here.")
    add("")
    add("Per-(posture, channel, timestep) values are in "
        "`data/step2/time_localized_class_difference.csv`.")
    add("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return summary_rows, bottom_higher_count, cell_total


def write_bottom_md(records, eta_summary_rows, bottom_higher_count, eta_cell_total,
                    path: Path):
    n = len(records)
    n_warn = sum(1 for r in records if r["bottom_candidate_warning"])
    n_default = sum(1 for r in records if r["bottom_agree_default"])
    rate_default = (n_default / n) if n else 0.0

    # Per (class × posture) at threshold=0.10.
    cell_n: dict = defaultdict(int)
    cell_a: dict = defaultdict(int)
    for r in records:
        key = (r["class_id"], r["posture_canonical"])
        cell_n[key] += 1
        if r["bottom_agreement_ratio"] <= DEFAULT_AGREE_THRESHOLD:
            cell_a[key] += 1
    cell_rates = {k: (cell_a[k] / cell_n[k]) if cell_n[k] else 0.0
                  for k in cell_n}
    if cell_rates:
        min_cell = min(cell_rates.items(), key=lambda kv: kv[1])
        max_cell = max(cell_rates.items(), key=lambda kv: kv[1])
    else:
        min_cell = (("?", "?"), 0.0)
        max_cell = (("?", "?"), 0.0)

    n_cells = len(cell_rates)
    n_cells_70 = sum(1 for v in cell_rates.values() if v >= GO_AGREE_RATE)
    n_cells_50 = sum(1 for v in cell_rates.values() if v >= WEAK_AGREE_RATE)

    # Verdict.
    bottom_diff_signal_pos = bottom_higher_count > (eta_cell_total / 2)
    bottom_diff_signal_partial = (bottom_higher_count >= max(1, eta_cell_total // 4)
                                   and not bottom_diff_signal_pos)
    if n_cells_70 >= GO_CELL_MAJORITY and bottom_diff_signal_pos:
        verdict = "GO"
    elif (WEAK_AGREE_RATE * n_cells <= n_cells_50 < GO_CELL_MAJORITY
          or bottom_diff_signal_partial
          or (n_cells_70 < GO_CELL_MAJORITY and n_cells_50 >= n_cells / 2)):
        verdict = "WEAK-GO"
    else:
        verdict = "NO-GO"

    lines: list[str] = []
    add = lines.append
    add("# Step 2.5 — Bottom-Event Candidate Audit")
    add("")
    add("Goal: check whether a single \"bottom\" event is a stable candidate")
    add("anchor across reps. **No anchor decision is finalized here.**")
    add("")

    add("## Method")
    add("")
    add(f"- Bottom search range on the original signal: "
        f"[{int(BOTTOM_SEARCH_LO * 100)}%, {int(BOTTOM_SEARCH_HI * 100)}%) of n_rows.")
    add(f"- `acc_bottom_idx` = argmin of acc_z (smoothed, k={SMOOTH_K}) in range.")
    add(f"- `gyro_bottom_idx` = argmax of |gyro| (smoothed, k={SMOOTH_K}) in range.")
    add("- `ensemble_bottom_idx` = floor((acc + gyro) / 2).")
    add("- `bottom_agreement_ratio` = |acc - gyro| / n_rows.")
    add(f"- `bottom_agree_default` = ratio ≤ **{DEFAULT_AGREE_THRESHOLD}** "
        "(initial threshold; not final).")
    add("")

    add("## Overall")
    add("")
    add(f"- Reps audited: **{n}**")
    add(f"- `bottom_agree_default` rate (threshold = {DEFAULT_AGREE_THRESHOLD}): "
        f"**{rate_default:.4f}** "
        f"({n_default} / {n})")
    add(f"- Reps with `bottom_candidate_warning`: {n_warn}")
    add("")

    add(f"## Per (class × posture) agree rate at threshold = {DEFAULT_AGREE_THRESHOLD}")
    add("")
    add("| class | SA | CA | HW |")
    add("|---|---|---|---|")
    for c in CLASSES:
        cells = []
        for p in POSTURES:
            v = cell_rates.get((c, p))
            cells.append(f"{v:.3f}" if v is not None else "–")
        add(f"| {c} | {cells[0]} | {cells[1]} | {cells[2]} |")
    add("")
    add(f"- Cells (class × posture) total : {n_cells} / 18")
    add(f"- Cells with agree_rate ≥ {GO_AGREE_RATE:.0%} : {n_cells_70}")
    add(f"- Cells with agree_rate ≥ {WEAK_AGREE_RATE:.0%} : {n_cells_50}")
    add(f"- Min cell agree_rate : {min_cell[1]:.3f}  "
        f"({min_cell[0][0]} × {min_cell[0][1]})")
    add(f"- Max cell agree_rate : {max_cell[1]:.3f}  "
        f"({max_cell[0][0]} × {max_cell[0][1]})")
    add("")

    add("## Threshold sensitivity")
    add("")
    add("Full table: `reports/bottom_agreement_threshold_sensitivity.csv`.")
    add("Threshold = 0.10 is an initial value. Bottom-anchor decisions, if")
    add("any, should re-read this table after Step 2.5 finalizes.")
    add("")

    add("## Bottom-region class-difference signal (cross-reference)")
    add("")
    add(f"- {bottom_higher_count} / {eta_cell_total} (posture × channel) cells "
        f"show higher mean η² in the bottom region than full-timeline.")
    flag = ("predominant" if bottom_diff_signal_pos
            else "partial" if bottom_diff_signal_partial
            else "weak/absent")
    add(f"- Bottom-region difference signal: **{flag}**.")
    add("")

    add("## Preliminary verdict")
    add("")
    add(f"- Cells with agree ≥ 70% : {n_cells_70} / {n_cells}  "
        f"(threshold for GO majority: {GO_CELL_MAJORITY})")
    add(f"- Bottom-region η² stronger than full-range: {bottom_higher_count}"
        f" / {eta_cell_total} cells")
    add("")
    add(f"- Closest classification: **{verdict}**")
    add("")
    add("Reasoning (heuristic, not a final decision):")
    add(f"- GO requires both a majority of cells with agree ≥ 70% AND a")
    add(f"  predominant bottom-region η² signal.")
    add(f"- WEAK-GO covers in-between cases (50–70% agreement on most cells,")
    add(f"  or partial bottom-region η² signal).")
    add(f"- NO-GO is when most cells have agree < 50%, or the bottom-region")
    add(f"  η² signal is at or below full-range mean.")
    add("")
    add("This script does NOT commit a bottom anchor; it only reports the")
    add("first-pass evidence so the next step can decide.")
    add("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return verdict, n_cells_70, n_cells_50, n_cells, min_cell, max_cell


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading manifest...")
    rows = read_manifest()
    print(f"  loaded samples: {len(rows)}")

    print("Processing reps (load + bottom audit + class-mean accumulation)...")
    bottom_records, groups = process_samples(rows)

    print("Writing bottom_event_audit.csv ...")
    write_bottom_csv(bottom_records, DATA_DIR / "step2" / "bottom_event_audit.csv")
    print("Writing threshold sensitivity ...")
    threshold_sensitivity(
        bottom_records,
        REPORTS_DIR / "bottom_agreement_threshold_sensitivity.csv",
    )

    print("Computing eta^2 per posture × channel × timestep ...")
    eta2_per_posture = compute_eta_squared(groups)
    write_eta_csv(eta2_per_posture,
                  DATA_DIR / "step2" / "time_localized_class_difference.csv")

    print("Writing reports ...")
    write_preflight_md(rows, bottom_records,
                       REPORTS_DIR / "step25_preflight_summary.md")
    eta_summary_rows, bot_higher, eta_cell_total = write_class_diff_md(
        eta2_per_posture,
        REPORTS_DIR / "time_localized_class_difference_summary.md",
    )
    verdict, n70, n50, n_cells, min_cell, max_cell = write_bottom_md(
        bottom_records, eta_summary_rows, bot_higher, eta_cell_total,
        REPORTS_DIR / "bottom_event_candidate_audit.md",
    )

    # ---- Console summary ----
    n = len(rows)
    n_default = sum(1 for r in bottom_records if r["bottom_agree_default"])
    rate_default = (n_default / n) if n else 0.0
    print("")
    print("=== Step 2.5 audit summary ===")
    print(f"loaded samples            : {n}")
    print(f"bottom_audit records      : {len(bottom_records)}")
    print(f"bottom_agree_default rate : {rate_default:.4f} "
          f"({n_default}/{n}, threshold={DEFAULT_AGREE_THRESHOLD})")
    print(f"class×posture cells       : {n_cells} / 18")
    print(f"  cells agree_rate ≥ 0.70 : {n70}")
    print(f"  cells agree_rate ≥ 0.50 : {n50}")
    print(f"  min cell agree_rate     : {min_cell[1]:.3f} "
          f"({min_cell[0][0]} × {min_cell[0][1]})")
    print(f"  max cell agree_rate     : {max_cell[1]:.3f} "
          f"({max_cell[0][0]} × {max_cell[0][1]})")
    print(f"bottom-region η² > full   : {bot_higher} / {eta_cell_total} cells")
    print(f"preliminary verdict       : {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
