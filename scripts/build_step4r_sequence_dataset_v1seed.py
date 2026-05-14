"""Build v1-ratio sequence dataset under parametrized split seed.

Wrapper around build_step4r_sequence_dataset.py. Reads
data/step4r_v1seed_robustness/s{SEED}/manifest_split.csv and writes
the sequence dataset into the same s{SEED}/ tree.

Run:
    & 'C:\\Users\\user\\anaconda3\\envs\\aicrc_env\\python.exe' -X utf8 \
        scripts/build_step4r_sequence_dataset_v1seed.py --split-seed 7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import build_step4r_sequence_dataset as base  # noqa: E402


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split-seed", type=int, required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    seed_dir = (
        PROJECT_ROOT / "data" / "step4r_v1seed_robustness" / f"s{args.split_seed}"
    )
    report_dir = (
        PROJECT_ROOT / "reports" / "step4r_v1seed_robustness"
        / f"s{args.split_seed}" / "4rb_attention"
    )
    data_dir = seed_dir / "4rb_attention"
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    base.INPUT_MANIFEST = seed_dir / "manifest_split.csv"
    base.OUTPUT_DATA_DIR = data_dir
    base.OUTPUT_REPORT_DIR = report_dir
    base.OUTPUT_NPZ = data_dir / "step4r_sequence_dataset.npz"
    base.OUTPUT_FAILURES_CSV = data_dir / "step4r_sequence_dataset_failures.csv"
    base.OUTPUT_SUMMARY_MD = report_dir / "step4r_sequence_dataset_summary.md"

    print("=" * 64)
    print(f"Sequence dataset for split_seed={args.split_seed}")
    print(f"  INPUT_MANIFEST -> {base.INPUT_MANIFEST}")
    print(f"  OUTPUT_NPZ     -> {base.OUTPUT_NPZ}")
    print("=" * 64)
    return base.main()


if __name__ == "__main__":
    sys.exit(main())
