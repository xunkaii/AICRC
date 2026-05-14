"""Step 4R-B (v2split) — Build raw IMU sequence dataset under v2 split.

Thin wrapper around scripts/build_step4r_sequence_dataset.py. Reads
data/manifest_split_v2.csv and writes the v2split sequence dataset.
Train channel mean/std are recomputed from v2 train rows only.

Run:
    & 'C:\\Users\\user\\anaconda3\\envs\\aicrc_env\\python.exe' -X utf8 \
        scripts/build_step4r_sequence_dataset_v2split.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import build_step4r_sequence_dataset as base  # noqa: E402


def _patch_paths() -> None:
    base.INPUT_MANIFEST = PROJECT_ROOT / "data" / "manifest_split_v2.csv"
    base.OUTPUT_DATA_DIR = (
        PROJECT_ROOT / "data" / "step4r_v2split" / "4rb_attention"
    )
    base.OUTPUT_REPORT_DIR = (
        PROJECT_ROOT / "reports" / "step4r_v2split" / "4rb_attention"
    )
    base.OUTPUT_NPZ = base.OUTPUT_DATA_DIR / "step4r_sequence_dataset.npz"
    base.OUTPUT_FAILURES_CSV = (
        base.OUTPUT_DATA_DIR / "step4r_sequence_dataset_failures.csv"
    )
    base.OUTPUT_SUMMARY_MD = (
        base.OUTPUT_REPORT_DIR / "step4r_sequence_dataset_summary.md"
    )


def main() -> int:
    _patch_paths()
    print("=" * 64)
    print("Step 4R-B (v2split) — Build raw IMU sequence dataset")
    print("=" * 64)
    print(f"  INPUT_MANIFEST   -> {base.INPUT_MANIFEST}")
    print(f"  OUTPUT_NPZ       -> {base.OUTPUT_NPZ}")
    print()
    return base.main()


if __name__ == "__main__":
    sys.exit(main())
