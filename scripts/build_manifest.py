"""Build the AICRC Step 1 manifest from rep-level segmented .txt files.

Inputs (read-only):
  C:\\Users\\user\\data\\AICRC_DSQ_Data(Combined_segmented)_v2
        \\AICRC_DSQ_Data(Combined_segmented)\\
            C{1..6}\\{posture_folder}\\Segmented\\seg_EP{NN}_{posture}_C{N}_SW_combine\\
                file_..._{rep}_o0.txt
                _time.txt
                _vkeep_st_end.txt

Outputs:
  data/manifest_raw.csv
  data/manifest_clean.csv
  data/manifest_split.csv
  reports/manifest_summary.md
  reports/manifest_exclusion_summary.csv
  reports/split_summary_by_condition.csv
  reports/split_participants.csv

Step 1 locked scope:
  - classes  : C1..C6 only             (C7 hard drop)
  - data type: Segmented only          (Combined hard drop)
  - postures : SA / CA / HW canonical  (paper Table 2 / Figure 3)

One row in the manifest = one segmented rep .txt file.
The manifest is the raw-data registry only. It does NOT carry features,
attention, captions, predictions, or any analysis output.

Run:
  python scripts/build_manifest.py
"""

from __future__ import annotations

import csv
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths and locked scope
# ---------------------------------------------------------------------------

DATASET_ROOT = Path(
    r"C:\Users\user\data\AICRC_DSQ_Data(Combined_segmented)_v2"
    r"\AICRC_DSQ_Data(Combined_segmented)"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

STEP1_CLASSES = ("C1", "C2", "C3", "C4", "C5", "C6")
STEP1_DATA_TYPE = "Segmented"
POSTURE_FOLDERS = ("01_Straight_Arms", "02_Crossed_Arms", "03_Hands_on_Waist")

POSTURE_TOKEN_TO_CANONICAL = {
    "01_Straight_Arms":  "SA", "Straight_Arms":  "SA", "AS": "SA", "SA": "SA",
    "02_Crossed_Arms":   "CA", "Crossed_Arms":   "CA", "AC": "CA", "CA": "CA",
    "03_Hands_on_Waist": "HW", "Hands_on_Waist": "HW", "AW": "HW", "HW": "HW",
}
POSTURE_CANONICAL_TO_LEGACY = {"SA": "AS", "CA": "AC", "HW": "AW"}
POSTURE_CANONICAL_TO_FOLDER = {
    "SA": "01_Straight_Arms",
    "CA": "02_Crossed_Arms",
    "HW": "03_Hands_on_Waist",
}

CLASS_DESCRIPTIONS = {
    "C1": "Normal",
    "C2": "Insufficient depth",
    "C3": "Insufficient depth with posterior tilting and knee valgus",
    "C4": "Left-knee valgus",
    "C5": "Right-knee valgus",
    "C6": "Both-knee valgus",
}

# ---- Quality thresholds ----
EXPECTED_N_COLS = 7
MIN_N_ROWS = 50
SAMPLING_RATE_ASSUMED_HZ = 50
LENGTH_RATIO_LO = 0.8
LENGTH_RATIO_HI = 1.2

# ---- Split config ----
SPLIT_TRAIN = 36
SPLIT_VAL = 8
SPLIT_TEST = 8
SPLIT_SEED = 42
SPLIT_VERSION = "v1_36_8_8"

# ---- Regexes ----
SEGMENT_DIR_RE = re.compile(r"^seg_EP(\d+)_(.+)_C(\d+)_SW_combine$")
PARTICIPANT_RE = re.compile(r"EP(\d+)")
REP_FILE_RE = re.compile(r"^file_.*_(\d+)_o0\.txt$")
TIME_LINE_REP_RE = re.compile(r"_(\d+)_o0\.txt")

# ---- Manifest schema ----
MANIFEST_COLS = [
    "sample_id", "rep_id", "episode_id", "participant_id",
    "class_id", "class_description",
    "posture_source_token", "posture_canonical", "posture_legacy", "posture_folder",
    "data_type", "rep_index",
    "signal_path", "signal_relpath", "dataset_root", "segment_dir",
    "n_rows", "n_cols", "has_nan", "has_inf", "is_empty",
    "t_start_s", "t_end_s", "duration_s",
    "start_idx", "end_idx", "boundary_ok", "boundary_reason",
    "sampling_rate_assumed_hz",
    "expected_n_rows_time", "length_ratio_time", "length_ok_time",
    "expected_n_rows_index", "length_ratio_index", "length_ok_index",
    "length_ok", "length_check_reason",
    "include", "exclude_reason",
    "split", "split_seed", "split_version", "split_group_key",
]

BOOL_COLS = {
    "has_nan", "has_inf", "is_empty",
    "boundary_ok", "length_ok_time", "length_ok_index", "length_ok",
    "include",
}
INT_COLS = {
    "rep_index", "n_rows", "n_cols",
    "start_idx", "end_idx",
    "sampling_rate_assumed_hz",
    "expected_n_rows_index",
    "split_seed",
}
FLOAT_COLS = {
    "t_start_s", "t_end_s", "duration_s",
    "expected_n_rows_time", "length_ratio_time", "length_ratio_index",
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_segment_dir(name: str):
    m = SEGMENT_DIR_RE.match(name)
    if not m:
        return None
    return {
        "ep": f"EP{int(m.group(1)):02d}",
        "posture_token": m.group(2),
        "class_token": f"C{int(m.group(3))}",
    }


def parse_rep_filename(name: str):
    m = REP_FILE_RE.match(name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def parse_participant_from_path(*parts) -> str | None:
    for p in parts:
        m = PARTICIPANT_RE.search(str(p))
        if m:
            return f"EP{int(m.group(1)):02d}"
    return None


def canonicalize_posture(token):
    if not token:
        return None
    return POSTURE_TOKEN_TO_CANONICAL.get(token)


def read_rep_file(path: Path) -> dict:
    """Return rep-file row stats: n_rows, n_cols, has_nan, has_inf, is_empty.

    n_cols is taken from the first non-empty line.
    has_nan also covers values that fail to parse as float (corruption).
    """
    n_rows = 0
    n_cols = 0
    has_nan = False
    has_inf = False
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if n_cols == 0:
                    n_cols = len(parts)
                n_rows += 1
                for v in parts:
                    try:
                        x = float(v)
                    except ValueError:
                        has_nan = True
                        continue
                    if math.isnan(x):
                        has_nan = True
                    elif math.isinf(x):
                        has_inf = True
    except OSError:
        pass
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "is_empty": (n_rows == 0),
    }


def parse_time_file(path: Path) -> dict:
    """Return rep_index (int) -> (t_start_s, t_end_s) from `_time.txt`."""
    out: dict[int, tuple[float, float]] = {}
    if not path.is_file():
        return out
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                # Format: `[N] out/.../file_..._N_o0.txt\tt_start\tt_end`
                cols = stripped.split("\t")
                if len(cols) < 3:
                    cols = stripped.split()
                if len(cols) < 3:
                    continue
                first = cols[0]
                m = TIME_LINE_REP_RE.search(first)
                if not m:
                    continue
                try:
                    rep_idx = int(m.group(1))
                    t_start = float(cols[-2])
                    t_end = float(cols[-1])
                except ValueError:
                    continue
                out[rep_idx] = (t_start, t_end)
    except OSError:
        pass
    return out


def parse_vkeep_file(path: Path):
    """Return (start_idx_list, end_idx_list) from `_vkeep_st_end.txt`.

    The file has two whitespace-separated lines: starts then ends, position k
    (1-based) corresponds to rep_index k.
    """
    if not path.is_file():
        return [], []
    try:
        with path.open("r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except OSError:
        return [], []
    if len(lines) < 2:
        return [], []
    try:
        starts = [int(float(x)) for x in lines[0].split()]
        ends = [int(float(x)) for x in lines[1].split()]
    except ValueError:
        return [], []
    return starts, ends


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------

def build_row(*, rep_path, seg_dir, seg_name, class_id, posture_folder,
              posture_token, participant_id, rep_index, time_map, starts, ends):
    row = {col: "" for col in MANIFEST_COLS}
    row["dataset_root"] = str(DATASET_ROOT)
    row["segment_dir"] = seg_name
    row["episode_id"] = seg_name
    row["data_type"] = STEP1_DATA_TYPE
    row["signal_path"] = str(rep_path)
    try:
        row["signal_relpath"] = str(rep_path.relative_to(DATASET_ROOT))
    except ValueError:
        row["signal_relpath"] = str(rep_path)
    row["sampling_rate_assumed_hz"] = SAMPLING_RATE_ASSUMED_HZ
    row["class_id"] = class_id
    row["class_description"] = CLASS_DESCRIPTIONS.get(class_id, "")
    row["posture_folder"] = posture_folder
    row["posture_source_token"] = posture_token or ""

    posture_canonical = canonicalize_posture(posture_token) \
        or canonicalize_posture(posture_folder)
    row["posture_canonical"] = posture_canonical or ""
    if posture_canonical:
        row["posture_legacy"] = POSTURE_CANONICAL_TO_LEGACY[posture_canonical]

    if participant_id:
        row["participant_id"] = participant_id
        row["split_group_key"] = participant_id

    if rep_index is not None:
        row["rep_index"] = rep_index

    if participant_id and class_id and posture_canonical and rep_index is not None:
        rep_id = (f"{participant_id}_{class_id}_{posture_canonical}"
                  f"_rep{int(rep_index):02d}")
        row["rep_id"] = rep_id
        row["sample_id"] = rep_id

    # ---- Hard-drop checks (parsing) ----
    exclude_reasons: list[str] = []
    if class_id not in STEP1_CLASSES:
        exclude_reasons.append("class_out_of_scope")
    if not participant_id:
        exclude_reasons.append("parse_participant_failed")
    if rep_index is None:
        exclude_reasons.append("parse_rep_index_failed")
    if not posture_canonical:
        exclude_reasons.append("parse_posture_failed")

    # ---- Rep-file content ----
    meta = read_rep_file(rep_path)
    row["n_rows"] = meta["n_rows"]
    row["n_cols"] = meta["n_cols"]
    row["has_nan"] = meta["has_nan"]
    row["has_inf"] = meta["has_inf"]
    row["is_empty"] = meta["is_empty"]

    if meta["is_empty"]:
        exclude_reasons.append("empty_file")
    else:
        if meta["n_rows"] < MIN_N_ROWS:
            exclude_reasons.append("n_rows_lt_50")
        if meta["n_cols"] != EXPECTED_N_COLS:
            exclude_reasons.append("n_cols_ne_7")

    # ---- Time bounds ----
    t_start = t_end = None
    if rep_index is not None and rep_index in time_map:
        t_start, t_end = time_map[rep_index]
    if t_start is not None and t_end is not None:
        row["t_start_s"] = t_start
        row["t_end_s"] = t_end
        row["duration_s"] = t_end - t_start

    # ---- Index bounds ----
    start_idx = end_idx = None
    if (rep_index is not None
            and 1 <= rep_index <= len(starts)
            and rep_index <= len(ends)):
        start_idx = starts[rep_index - 1]
        end_idx = ends[rep_index - 1]
        row["start_idx"] = start_idx
        row["end_idx"] = end_idx

    # ---- Boundary check ----
    # boundary_ok is always a concrete True/False (no None), so the soft-flag
    # summary can count rows where the rep has no usable index bounds.
    boundary_reasons: list[str] = []
    if start_idx is None or end_idx is None:
        boundary_reasons.append("missing_vkeep")
    else:
        if start_idx < 0 or end_idx < 0:
            boundary_reasons.append("negative_idx")
        if start_idx >= end_idx:
            boundary_reasons.append("start_ge_end")
    boundary_ok = (len(boundary_reasons) == 0)
    row["boundary_ok"] = boundary_ok
    row["boundary_reason"] = ";".join(boundary_reasons)

    # ---- Length checks (time- and index-based) ----
    length_reasons: list[str] = []

    expected_time = ratio_time = None
    ok_time = None
    if t_start is not None and t_end is not None and meta["n_rows"] > 0:
        duration = t_end - t_start
        if duration > 0:
            expected_time = duration * SAMPLING_RATE_ASSUMED_HZ
            ratio_time = meta["n_rows"] / expected_time
            ok_time = (LENGTH_RATIO_LO <= ratio_time <= LENGTH_RATIO_HI)
            if not ok_time:
                length_reasons.append("time_ratio_out_of_range")
        else:
            length_reasons.append("time_duration_le_0")
    else:
        length_reasons.append("missing_time")

    expected_idx = ratio_idx = None
    ok_idx = None
    if start_idx is not None and end_idx is not None and meta["n_rows"] > 0:
        n_idx = end_idx - start_idx + 1
        if n_idx > 0:
            expected_idx = n_idx
            ratio_idx = meta["n_rows"] / n_idx
            ok_idx = (LENGTH_RATIO_LO <= ratio_idx <= LENGTH_RATIO_HI)
            if not ok_idx:
                length_reasons.append("index_ratio_out_of_range")
        else:
            length_reasons.append("index_span_le_0")
    else:
        length_reasons.append("missing_index")

    row["expected_n_rows_time"] = expected_time
    row["length_ratio_time"] = ratio_time
    row["length_ok_time"] = ok_time
    row["expected_n_rows_index"] = expected_idx
    row["length_ratio_index"] = ratio_idx
    row["length_ok_index"] = ok_idx

    available = [v for v in (ok_time, ok_idx) if v is not None]
    if not available:
        length_ok = None
        length_reasons.append("no_length_check_available")
    else:
        length_ok = all(available)
    row["length_ok"] = length_ok
    row["length_check_reason"] = ";".join(length_reasons)

    include = (len(exclude_reasons) == 0)
    row["include"] = include
    row["exclude_reason"] = ";".join(exclude_reasons)

    return row


def rows_for_segment(seg_dir, class_id, posture_folder, seg_name):
    parsed = parse_segment_dir(seg_name)
    pid_from_dir = parsed["ep"] if parsed else None
    posture_token = parsed["posture_token"] if parsed else None

    time_map = parse_time_file(seg_dir / "_time.txt")
    starts, ends = parse_vkeep_file(seg_dir / "_vkeep_st_end.txt")

    rep_paths = sorted(
        p for p in seg_dir.iterdir()
        if p.is_file() and p.name.startswith("file_") and p.name.endswith(".txt")
    )
    out = []
    for rep_path in rep_paths:
        rep_index = parse_rep_filename(rep_path.name)
        pid = pid_from_dir or parse_participant_from_path(seg_name, rep_path.name)
        out.append(build_row(
            rep_path=rep_path,
            seg_dir=seg_dir,
            seg_name=seg_name,
            class_id=class_id,
            posture_folder=posture_folder,
            posture_token=posture_token,
            participant_id=pid,
            rep_index=rep_index,
            time_map=time_map,
            starts=starts,
            ends=ends,
        ))
    return out


def collect_raw_rows() -> list[dict]:
    rows: list[dict] = []
    if not DATASET_ROOT.is_dir():
        print(f"ERROR: dataset root not found: {DATASET_ROOT}", file=sys.stderr)
        return rows

    for class_id in STEP1_CLASSES:
        class_root = DATASET_ROOT / class_id
        if not class_root.is_dir():
            continue
        for posture_folder in POSTURE_FOLDERS:
            seg_root = class_root / posture_folder / STEP1_DATA_TYPE
            if not seg_root.is_dir():
                continue
            for seg_name in sorted(os.listdir(seg_root)):
                seg_dir = seg_root / seg_name
                if not seg_dir.is_dir():
                    continue
                rows.extend(rows_for_segment(seg_dir, class_id, posture_folder, seg_name))
    return rows


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def serialize(key: str, val) -> str:
    if val is None:
        return ""
    if isinstance(val, str) and val == "":
        return ""
    if key in BOOL_COLS:
        return "True" if val else "False"
    if key in INT_COLS:
        try:
            return str(int(val))
        except (ValueError, TypeError):
            return ""
    if key in FLOAT_COLS:
        if isinstance(val, (int, float)):
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return ""
            return f"{float(val):.6f}"
        return str(val)
    return str(val)


def write_manifest_csv(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(MANIFEST_COLS)
        for r in rows:
            writer.writerow([serialize(k, r.get(k, "")) for k in MANIFEST_COLS])


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def compute_split(participants):
    rng = random.Random(SPLIT_SEED)
    sorted_ps = sorted(set(p for p in participants if p))
    rng.shuffle(sorted_ps)
    train = sorted_ps[:SPLIT_TRAIN]
    val = sorted_ps[SPLIT_TRAIN:SPLIT_TRAIN + SPLIT_VAL]
    test = sorted_ps[SPLIT_TRAIN + SPLIT_VAL:SPLIT_TRAIN + SPLIT_VAL + SPLIT_TEST]
    overflow = sorted_ps[SPLIT_TRAIN + SPLIT_VAL + SPLIT_TEST:]
    mapping: dict[str, str] = {}
    for p in train:
        mapping[p] = "train"
    for p in val:
        mapping[p] = "val"
    for p in test:
        mapping[p] = "test"
    for p in overflow:
        # Defensive overflow handling (shouldn't occur with 52 / 36-8-8).
        mapping[p] = "test"
    return mapping


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def write_exclusion_summary(rows, path: Path) -> Counter:
    counter: Counter = Counter()
    for r in rows:
        if r["include"]:
            continue
        for reason in str(r["exclude_reason"]).split(";"):
            if reason:
                counter[reason] += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["exclude_reason", "count"])
        for reason, count in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
            w.writerow([reason, count])
    return counter


def write_split_participants(mapping, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["participant_id", "split", "split_seed", "split_version"])
        for pid in sorted(mapping):
            w.writerow([pid, mapping[pid], SPLIT_SEED, SPLIT_VERSION])


def write_split_summary_by_condition(rows, path: Path) -> None:
    bucket_samples: dict = defaultdict(int)
    bucket_participants: dict = defaultdict(set)
    for r in rows:
        key = (r["class_id"], r["posture_canonical"], r["split"])
        bucket_samples[key] += 1
        if r["participant_id"]:
            bucket_participants[key].add(r["participant_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "posture_canonical", "split",
                    "n_samples", "n_participants"])
        for key in sorted(bucket_samples.keys()):
            cid, pos, sp = key
            w.writerow([cid, pos, sp,
                        bucket_samples[key],
                        len(bucket_participants[key])])


def write_summary_md(raw_rows, clean_rows, split_rows,
                     exclusion_counter, split_mapping, path: Path) -> None:
    lines: list[str] = []
    add = lines.append
    add("# AICRC Step 2 — Manifest 빌드 요약")
    add("")
    add(f"- 데이터셋 루트: `{DATASET_ROOT}`")
    add("- Step 1 고정 범위: classes C1–C6, Segmented, postures SA/CA/HW")
    add("- 생성 스크립트: `scripts/build_manifest.py`")
    add(f"- 학습 단위: rep `.txt` 파일 1개 = sample 1개")
    add("")
    add("## 1. 행 수")
    add("")
    add(f"- manifest_raw 행: **{len(raw_rows)}**")
    add(f"- manifest_clean 행: **{len(clean_rows)}** (hard drop {len(raw_rows) - len(clean_rows)}건 제외)")
    add(f"- manifest_split 행: **{len(split_rows)}**")
    add("")

    if exclusion_counter:
        add("## 2. 제외 사유 (hard drop)")
        add("")
        add("| reason | count |")
        add("|---|---|")
        for reason, count in sorted(exclusion_counter.items(), key=lambda x: (-x[1], x[0])):
            add(f"| `{reason}` | {count} |")
        add("")
    else:
        add("## 2. 제외 사유 (hard drop)")
        add("")
        add("hard drop 없음.")
        add("")

    boundary_bad = sum(1 for r in clean_rows if r["boundary_ok"] is False)
    length_bad = sum(1 for r in clean_rows if r["length_ok"] is False)
    add("## 3. Soft flags (clean에 보존됨)")
    add("")
    add(f"- `boundary_ok=False`: {boundary_bad}건")
    add(f"- `length_ok=False`  : {length_bad}건")
    add("")

    by_split_p: dict = defaultdict(set)
    by_split_s: dict = defaultdict(int)
    for r in split_rows:
        by_split_p[r["split"]].add(r["participant_id"])
        by_split_s[r["split"]] += 1
    add(f"## 4. Split (seed={SPLIT_SEED}, version=`{SPLIT_VERSION}`)")
    add("")
    add(f"- 기준: participant 단위 (group_key=`participant_id`)")
    add(f"- 동일 participant가 여러 split에 걸치지 않음")
    add("")
    add("| split | participants | samples |")
    add("|---|---|---|")
    for sp in ("train", "val", "test"):
        add(f"| {sp} | {len(by_split_p[sp])} | {by_split_s[sp]} |")
    add("")

    add("## 5. Class × posture sample 수 (clean)")
    add("")
    add("| class | SA | CA | HW | total |")
    add("|---|---|---|---|---|")
    by_cp: dict = defaultdict(int)
    for r in clean_rows:
        by_cp[(r["class_id"], r["posture_canonical"])] += 1
    for c in STEP1_CLASSES:
        sa = by_cp[(c, "SA")]
        ca = by_cp[(c, "CA")]
        hw = by_cp[(c, "HW")]
        add(f"| {c} | {sa} | {ca} | {hw} | {sa + ca + hw} |")
    add("")

    add("## 6. 산출물")
    add("")
    add("- `data/manifest_raw.csv`")
    add("- `data/manifest_clean.csv`")
    add("- `data/manifest_split.csv`")
    add("- `reports/manifest_exclusion_summary.csv`")
    add("- `reports/split_summary_by_condition.csv`")
    add("- `reports/split_participants.csv`")
    add("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Walking dataset...")
    raw_rows = collect_raw_rows()
    print(f"  raw rows: {len(raw_rows)}")
    write_manifest_csv(raw_rows, DATA_DIR / "manifest_raw.csv")

    clean_rows = [r for r in raw_rows if r["include"]]
    print(f"  clean rows: {len(clean_rows)}")
    write_manifest_csv(clean_rows, DATA_DIR / "manifest_clean.csv")

    exclusion_counter = write_exclusion_summary(
        raw_rows, REPORTS_DIR / "manifest_exclusion_summary.csv"
    )

    participants = sorted({r["participant_id"] for r in clean_rows if r["participant_id"]})
    split_mapping = compute_split(participants)

    split_rows: list[dict] = []
    for r in clean_rows:
        rr = dict(r)
        rr["split"] = split_mapping.get(rr["participant_id"], "")
        rr["split_seed"] = SPLIT_SEED
        rr["split_version"] = SPLIT_VERSION
        rr["split_group_key"] = rr["participant_id"]
        split_rows.append(rr)

    write_manifest_csv(split_rows, DATA_DIR / "manifest_split.csv")
    write_split_participants(split_mapping, REPORTS_DIR / "split_participants.csv")
    write_split_summary_by_condition(
        split_rows, REPORTS_DIR / "split_summary_by_condition.csv"
    )
    write_summary_md(
        raw_rows, clean_rows, split_rows,
        exclusion_counter, split_mapping,
        REPORTS_DIR / "manifest_summary.md",
    )

    # ---- Console report ----
    print("")
    print("=== summary ===")
    print(f"raw   rows: {len(raw_rows)}")
    print(f"clean rows: {len(clean_rows)}")
    print(f"split rows: {len(split_rows)}")
    print("")
    if exclusion_counter:
        print("exclusion reasons:")
        for reason, count in sorted(exclusion_counter.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {reason:30s} {count}")
    else:
        print("exclusion reasons: (none)")
    print("")
    print("split participants / samples:")
    by_split_p: dict = defaultdict(set)
    by_split_s: dict = defaultdict(int)
    for r in split_rows:
        by_split_p[r["split"]].add(r["participant_id"])
        by_split_s[r["split"]] += 1
    for sp in ("train", "val", "test"):
        print(f"  {sp:6s} participants={len(by_split_p[sp]):3d} "
              f"samples={by_split_s[sp]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
