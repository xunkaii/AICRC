"""Step 4R-B — Build raw IMU sequence dataset (no model training).

Reads (read-only):
    data/manifest_split.csv
    <each row>.signal_path   (Windows absolute paths, used as-is)

Writes:
    data/step4r/4rb_attention/step4r_sequence_dataset.npz
    data/step4r/4rb_attention/step4r_sequence_dataset_failures.csv
    reports/step4r/4rb_attention/step4r_sequence_dataset_summary.md

Behavior:
    - Each rep .txt has 7 columns: timestamp, acc_x, acc_y, acc_z,
      gyro_x, gyro_y, gyro_z. Column 0 (timestamp) is dropped.
    - Each rep is linearly resampled to length 128 along the time axis,
      independently per channel, on a uniform [0, 1] grid.
    - X_raw shape: (N, 6, 128), float32. Channel order:
      acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z.
    - Channel-wise train mean/std are computed from train split only and
      applied to all splits to produce X_norm. Std floor 1e-8.
    - y: integer label (C1->0, ..., C6->5).
    - class_id, posture_canonical (SA/CA/HW), sample_id, participant_id,
      split are saved alongside.
    - Existing split (manifest_split.csv `split` column, v1_36_8_8) is
      reused as-is. No participant split is recomputed.
    - This script does NOT train a model and does NOT write any model
      artifact.

Reference:
    reports/step4_research_reframing.md             (4R-B definition)
    scripts/audit_step25_v2.py::load_signal          (txt loader pattern)

Run:
    python scripts/build_step4r_sequence_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_MANIFEST = PROJECT_ROOT / "data" / "manifest_split.csv"

OUTPUT_DATA_DIR = PROJECT_ROOT / "data" / "step4r" / "4rb_attention"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "reports" / "step4r" / "4rb_attention"

OUTPUT_NPZ = OUTPUT_DATA_DIR / "step4r_sequence_dataset.npz"
OUTPUT_FAILURES_CSV = OUTPUT_DATA_DIR / "step4r_sequence_dataset_failures.csv"
OUTPUT_SUMMARY_MD = OUTPUT_REPORT_DIR / "step4r_sequence_dataset_summary.md"

EXPECTED_ROW_COUNT = 9275

CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

POSTURES = ["SA", "CA", "HW"]
SPLITS = ["train", "val", "test"]

CHANNEL_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
N_CHANNELS = 6
EXPECTED_TXT_COLS = 7  # timestamp + 6 channels
TARGET_LENGTH = 128

STD_FLOOR = 1e-8


# ---------------------------------------------------------------------------
# Per-row loading and resampling
# ---------------------------------------------------------------------------

def _load_and_resample(path: Path) -> tuple[np.ndarray | None, str | None]:
    """Load a rep .txt and linearly resample to (6, TARGET_LENGTH).

    Returns (array, None) on success or (None, reason) on failure.
    """
    if not path.exists():
        return None, "file_not_found"
    try:
        arr = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except Exception as e:  # noqa: BLE001
        return None, f"loadtxt_error: {type(e).__name__}"

    if arr.ndim != 2:
        return None, f"shape_invalid_ndim_{arr.ndim}"
    if arr.shape[1] != EXPECTED_TXT_COLS:
        return None, f"shape_invalid_ncols_{arr.shape[1]}"
    if arr.shape[0] < 2:
        return None, f"too_few_rows_{arr.shape[0]}"
    if not np.isfinite(arr).all():
        return None, "non_finite_values"

    sig = arr[:, 1:]  # drop timestamp -> (n_rows, 6)
    n = sig.shape[0]
    src_t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    dst_t = np.linspace(0.0, 1.0, TARGET_LENGTH, dtype=np.float64)
    out = np.empty((N_CHANNELS, TARGET_LENGTH), dtype=np.float64)
    for c in range(N_CHANNELS):
        out[c] = np.interp(dst_t, src_t, sig[:, c])
    if not np.isfinite(out).all():
        return None, "non_finite_after_resample"
    if out.shape != (N_CHANNELS, TARGET_LENGTH):
        return None, f"resample_shape_invalid_{out.shape}"
    return out, None


# ---------------------------------------------------------------------------
# Manifest checks
# ---------------------------------------------------------------------------

def _check_manifest(df: pd.DataFrame) -> None:
    if len(df) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"manifest row count {len(df)} != expected {EXPECTED_ROW_COUNT}."
        )
    required = {
        "sample_id",
        "participant_id",
        "class_id",
        "posture_canonical",
        "split",
        "signal_path",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    bad_classes = sorted(set(df["class_id"].unique()) - set(CLASSES))
    if bad_classes:
        raise ValueError(f"class_id outside {CLASSES}: {bad_classes}")
    bad_postures = sorted(set(df["posture_canonical"].unique()) - set(POSTURES))
    if bad_postures:
        raise ValueError(f"posture_canonical outside {POSTURES}: {bad_postures}")
    bad_splits = sorted(set(df["split"].unique()) - set(SPLITS))
    if bad_splits:
        raise ValueError(f"split outside {SPLITS}: {bad_splits}")
    if df["sample_id"].duplicated().any():
        n = int(df["sample_id"].duplicated().sum())
        raise ValueError(f"manifest has {n} duplicate sample_id.")


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def _build_arrays(df: pd.DataFrame) -> tuple[dict, list[dict]]:
    n = len(df)
    X = np.empty((n, N_CHANNELS, TARGET_LENGTH), dtype=np.float32)
    success_mask = np.zeros(n, dtype=bool)
    failures: list[dict] = []

    for i, row in enumerate(df.itertuples(index=False)):
        path_str = getattr(row, "signal_path")
        sample_id = getattr(row, "sample_id")
        path = Path(path_str)
        arr, reason = _load_and_resample(path)
        if arr is None:
            failures.append(
                {
                    "sample_id": sample_id,
                    "split": getattr(row, "split"),
                    "class_id": getattr(row, "class_id"),
                    "posture_canonical": getattr(row, "posture_canonical"),
                    "signal_path": path_str,
                    "reason": reason,
                }
            )
            continue
        X[i] = arr.astype(np.float32, copy=False)
        success_mask[i] = True

        if (i + 1) % 1000 == 0:
            print(f"  processed {i + 1}/{n}")

    print(f"  processed {n}/{n} (final)")

    keep = success_mask
    X_raw = X[keep]
    df_keep = df.loc[keep].reset_index(drop=True)

    # Train mean/std on X_raw (success rows only), channel-wise.
    train_mask_keep = (df_keep["split"] == "train").to_numpy()
    if not train_mask_keep.any():
        raise ValueError("No surviving train rows; cannot compute channel stats.")
    train_X = X_raw[train_mask_keep]
    channel_mean = train_X.mean(axis=(0, 2)).astype(np.float32)  # (6,)
    channel_std = train_X.std(axis=(0, 2)).astype(np.float32)    # (6,)
    if (channel_std < STD_FLOOR).any():
        n_low = int((channel_std < STD_FLOOR).sum())
        print(f"  warn: {n_low} channels have std < {STD_FLOOR:.0e}; flooring.")
    channel_std_safe = np.maximum(channel_std, STD_FLOOR).astype(np.float32)

    X_norm = (X_raw - channel_mean[None, :, None]) / channel_std_safe[None, :, None]
    X_norm = X_norm.astype(np.float32, copy=False)

    y = np.array(
        [CLASS_TO_IDX[c] for c in df_keep["class_id"].tolist()],
        dtype=np.int64,
    )

    arrays = {
        "X_raw": X_raw,
        "X_norm": X_norm,
        "y": y,
        "class_id": df_keep["class_id"].astype(str).to_numpy(),
        "posture_canonical": df_keep["posture_canonical"].astype(str).to_numpy(),
        "sample_id": df_keep["sample_id"].astype(str).to_numpy(),
        "participant_id": df_keep["participant_id"].astype(str).to_numpy(),
        "split": df_keep["split"].astype(str).to_numpy(),
        "channel_mean": channel_mean,
        "channel_std": channel_std_safe,
        "channel_names": np.array(CHANNEL_NAMES, dtype=str),
        "target_length": np.array([TARGET_LENGTH], dtype=np.int64),
    }
    return arrays, failures


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_arrays(arrays: dict) -> dict:
    """Return dict of summary checks (also raises on hard violations)."""
    X_raw = arrays["X_raw"]
    X_norm = arrays["X_norm"]
    y = arrays["y"]

    n = X_raw.shape[0]
    checks: dict = {}
    checks["n_success"] = int(n)

    if X_raw.shape[1:] != (N_CHANNELS, TARGET_LENGTH):
        raise ValueError(
            f"X_raw last dims {X_raw.shape[1:]} != ({N_CHANNELS}, {TARGET_LENGTH})."
        )
    if X_norm.shape != X_raw.shape:
        raise ValueError(
            f"X_norm shape {X_norm.shape} != X_raw shape {X_raw.shape}."
        )

    raw_nan = int(np.isnan(X_raw).sum())
    raw_inf = int(np.isinf(X_raw).sum())
    norm_nan = int(np.isnan(X_norm).sum())
    norm_inf = int(np.isinf(X_norm).sum())
    if raw_nan or raw_inf or norm_nan or norm_inf:
        raise ValueError(
            f"non-finite values present: X_raw NaN={raw_nan} Inf={raw_inf}, "
            f"X_norm NaN={norm_nan} Inf={norm_inf}."
        )
    checks["raw_nan"] = raw_nan
    checks["raw_inf"] = raw_inf
    checks["norm_nan"] = norm_nan
    checks["norm_inf"] = norm_inf

    if y.shape[0] != n:
        raise ValueError(f"y length {y.shape[0]} != n {n}.")
    if y.min() < 0 or y.max() > 5:
        raise ValueError(f"y outside [0, 5]: min={y.min()}, max={y.max()}.")

    # All time slices length 128 (already implied by shape but double-check).
    checks["all_length_128"] = bool(X_raw.shape[2] == TARGET_LENGTH)

    # Per-split / per-class / per-posture counts.
    split_counts = {s: int((arrays["split"] == s).sum()) for s in SPLITS}
    class_counts = {c: int((arrays["class_id"] == c).sum()) for c in CLASSES}
    posture_counts = {p: int((arrays["posture_canonical"] == p).sum()) for p in POSTURES}
    checks["split_counts"] = split_counts
    checks["class_counts"] = class_counts
    checks["posture_counts"] = posture_counts
    checks["channel_mean"] = arrays["channel_mean"].tolist()
    checks["channel_std"] = arrays["channel_std"].tolist()

    # Sanity: train X_norm channel-wise mean ~0, std ~1.
    train_mask = arrays["split"] == "train"
    if train_mask.any():
        tn = X_norm[train_mask]
        checks["train_norm_channel_mean"] = tn.mean(axis=(0, 2)).astype(float).tolist()
        checks["train_norm_channel_std"] = tn.std(axis=(0, 2)).astype(float).tolist()

    return checks


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def _markdown_summary(
    n_manifest: int,
    checks: dict,
    n_failures: int,
    failures: list[dict],
) -> str:
    L = []
    L.append("# Step 4R-B — Raw IMU Sequence Dataset 빌드 요약")
    L.append("")
    L.append("- 생성 스크립트: `scripts/build_step4r_sequence_dataset.py`")
    L.append("- 입력: `data/manifest_split.csv` (read-only)")
    L.append("- 출력: `data/step4r/4rb_attention/step4r_sequence_dataset.npz`, `..._failures.csv`")
    L.append("- 본 단계는 **데이터셋 생성·검증까지만** 수행한다. 모델 학습은 본 단계의 범위 밖이며, `train_step4r_attention_rnn.py`는 아직 작성하지 않는다.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 1. 본 작업의 위치")
    L.append("")
    L.append("- 본 작업은 `reports/step4_research_reframing.md` §5.2의 Step 4R-B (BiGRU + Attention raw IMU sensor-to-schema 학습 본체)의 **선행 데이터 준비**이다.")
    L.append("- 기존 Step 1~4 / Step 4R-A 산출물은 수정되지 않는다. 본 작업은 새 디렉토리에만 쓴다.")
    L.append("- split은 `manifest_split.csv`의 기존 `split` 컬럼(`v1_36_8_8`, participant-disjoint)을 그대로 사용했다. participant split을 다시 만들지 않았다.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 2. 행 수")
    L.append("")
    L.append("| 항목 | 값 |")
    L.append("|---|---:|")
    L.append(f"| manifest row 수 | {n_manifest} |")
    L.append(f"| 성공 sample 수 | {checks['n_success']} |")
    L.append(f"| 실패 sample 수 | {n_failures} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 3. Tensor shape")
    L.append("")
    L.append("| 항목 | 값 |")
    L.append("|---|---|")
    L.append(f"| X_raw shape | (N, {N_CHANNELS}, {TARGET_LENGTH}) where N = {checks['n_success']} |")
    L.append(f"| X_norm shape | (N, {N_CHANNELS}, {TARGET_LENGTH}) where N = {checks['n_success']} |")
    L.append(f"| 길이 {TARGET_LENGTH} 보간 성공 여부 | {'True' if checks['all_length_128'] else 'False'} |")
    L.append(f"| dtype (X_raw / X_norm) | float32 / float32 |")
    L.append(f"| dtype (y) | int64 (C1=0, C2=1, ..., C6=5) |")
    L.append(f"| 채널 순서 | {', '.join(CHANNEL_NAMES)} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 4. NaN / Inf 검사")
    L.append("")
    L.append("| array | NaN cells | Inf cells |")
    L.append("|---|---:|---:|")
    L.append(f"| X_raw | {checks['raw_nan']} | {checks['raw_inf']} |")
    L.append(f"| X_norm | {checks['norm_nan']} | {checks['norm_inf']} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 5. Split / Class / Posture 분포 (성공 sample 기준)")
    L.append("")
    L.append("### 5.1 split 별")
    L.append("")
    L.append("| split | n |")
    L.append("|---|---:|")
    for s in SPLITS:
        L.append(f"| {s} | {checks['split_counts'][s]} |")
    L.append("")
    L.append("### 5.2 class 별")
    L.append("")
    L.append("| class | n |")
    L.append("|---|---:|")
    for c in CLASSES:
        L.append(f"| {c} | {checks['class_counts'][c]} |")
    L.append("")
    L.append("### 5.3 posture 별")
    L.append("")
    L.append("| posture | n |")
    L.append("|---|---:|")
    for p in POSTURES:
        L.append(f"| {p} | {checks['posture_counts'][p]} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 6. 채널별 train mean / std")
    L.append("")
    L.append("train split 성공 sample만 사용해 channel-wise로 산출. val/test에는 동일한 통계를 적용해 X_norm을 생성한다 (train 통계 누수 방지: val/test 통계는 산출에 사용되지 않음).")
    L.append("")
    L.append("| channel | train mean | train std |")
    L.append("|---|---:|---:|")
    for i, name in enumerate(CHANNEL_NAMES):
        L.append(
            f"| {name} | {checks['channel_mean'][i]:+.6f} | {checks['channel_std'][i]:.6f} |"
        )
    L.append("")
    if "train_norm_channel_mean" in checks:
        L.append("정규화 후 train split의 channel-wise mean/std (sanity check, mean≈0 / std≈1 기대):")
        L.append("")
        L.append("| channel | mean (post-norm) | std (post-norm) |")
        L.append("|---|---:|---:|")
        for i, name in enumerate(CHANNEL_NAMES):
            L.append(
                f"| {name} | {checks['train_norm_channel_mean'][i]:+.6e} | "
                f"{checks['train_norm_channel_std'][i]:.6f} |"
            )
        L.append("")
    L.append("---")
    L.append("")
    L.append("## 7. 실패 sample 목록")
    L.append("")
    if n_failures == 0:
        L.append("실패 0건. 모든 manifest row가 길이 128 보간까지 성공.")
    else:
        L.append(f"총 {n_failures}건. 자세한 목록은 `step4r_sequence_dataset_failures.csv` 참조.")
        L.append("")
        reason_counts: dict = {}
        for f in failures:
            reason_counts[f["reason"]] = reason_counts.get(f["reason"], 0) + 1
        L.append("| 사유 | n |")
        L.append("|---|---:|")
        for r in sorted(reason_counts.keys()):
            L.append(f"| {r} | {reason_counts[r]} |")
        L.append("")
        L.append("처음 20건 미리보기:")
        L.append("")
        L.append("| sample_id | split | class_id | posture | reason |")
        L.append("|---|---|---|---|---|")
        for f in failures[:20]:
            L.append(
                f"| {f['sample_id']} | {f['split']} | {f['class_id']} | "
                f"{f['posture_canonical']} | {f['reason']} |"
            )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 8. 산출물 목록")
    L.append("")
    L.append("- `data/step4r/4rb_attention/step4r_sequence_dataset.npz`")
    L.append("  - 키: `X_raw`, `X_norm`, `y`, `class_id`, `posture_canonical`, `sample_id`, `participant_id`, `split`, `channel_mean`, `channel_std`, `channel_names`, `target_length`")
    L.append("- `data/step4r/4rb_attention/step4r_sequence_dataset_failures.csv`")
    L.append("- `reports/step4r/4rb_attention/step4r_sequence_dataset_summary.md` (본 보고서)")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 9. 본 단계에서 명시적으로 결정하지 않는 사항")
    L.append("")
    L.append("- BiGRU 레이어 수, hidden size, attention 형태(additive vs scaled-dot)는 본 단계에서 결정하지 않는다.")
    L.append("- augmentation(jittering, scaling, time warping)은 본 단계에서 적용하지 않는다.")
    L.append("- posture conditioning 방식(sequence-level concat vs conditioning vector)은 본 단계에서 결정하지 않는다.")
    L.append("- 모델 학습 스크립트(`train_step4r_attention_rnn.py`)는 본 단계 산출물이 아니다.")
    L.append("")
    L.append("---")
    L.append("")
    L.append(
        "*본 보고서는 `scripts/build_step4r_sequence_dataset.py` 실행 시 자동 생성된다. "
        "기존 Step 1 ~ 4 / Step 4R-A 산출물은 수정되지 않는다.*"
    )
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 64)
    print("Step 4R-B — Build raw IMU sequence dataset (no training)")
    print("=" * 64)
    if not INPUT_MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {INPUT_MANIFEST}")
    df = pd.read_csv(INPUT_MANIFEST)
    print(f"loaded {len(df)} manifest rows from {INPUT_MANIFEST}")
    _check_manifest(df)

    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("loading and resampling rep .txt files ...")
    arrays, failures = _build_arrays(df)
    print(f"  success: {arrays['X_raw'].shape[0]} / {len(df)}")
    print(f"  failures: {len(failures)}")

    print()
    print("validating arrays ...")
    checks = _validate_arrays(arrays)
    print(
        f"  X_raw shape: {arrays['X_raw'].shape}, "
        f"X_norm shape: {arrays['X_norm'].shape}"
    )
    print(
        f"  channel_mean: "
        f"[{', '.join(f'{v:+.4f}' for v in arrays['channel_mean'])}]"
    )
    print(
        f"  channel_std : "
        f"[{', '.join(f'{v:.4f}' for v in arrays['channel_std'])}]"
    )

    np.savez(OUTPUT_NPZ, **arrays)
    print(f"saved npz -> {OUTPUT_NPZ}")

    failures_df = pd.DataFrame(
        failures,
        columns=[
            "sample_id",
            "split",
            "class_id",
            "posture_canonical",
            "signal_path",
            "reason",
        ],
    )
    failures_df.to_csv(OUTPUT_FAILURES_CSV, index=False)
    print(f"saved failures csv -> {OUTPUT_FAILURES_CSV}")

    md = _markdown_summary(len(df), checks, len(failures), failures)
    OUTPUT_SUMMARY_MD.write_text(md, encoding="utf-8")
    print(f"saved summary -> {OUTPUT_SUMMARY_MD}")

    print()
    print("=" * 64)
    print("Done.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
