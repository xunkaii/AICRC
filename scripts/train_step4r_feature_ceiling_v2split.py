"""Step 4R-A (v2split) — Feature-based ceiling (HGB) under v2 participant split.

This is a thin wrapper around scripts/train_step4r_feature_ceiling.py.
It imports the base module (which is read-only and unmodified) and
overrides the module-level path constants to point at the v2split tree
before invoking the base `main()` function.

Inputs (v2split):
    data/step4_v2split/step4_modeling_dataset.csv

Outputs (v2split tree; v1 outputs untouched):
    data/step4r_v2split/4ra_feature_ceiling/step4r_hgb_predictions_{raw,zscore}.csv
    data/step4r_v2split/4ra_feature_ceiling/step4r_hgb_schema_outputs_{raw,zscore}.csv
    reports/step4r_v2split/4ra_feature_ceiling/step4r_feature_ceiling_metrics.csv
    reports/step4r_v2split/4ra_feature_ceiling/step4r_feature_ceiling_confusion.csv
    reports/step4r_v2split/4ra_feature_ceiling/step4r_feature_ceiling_results.md

LR_BASELINE numbers in the base module are v1 reference values and remain
in the report for context; they do NOT affect HGB training. Re-interpretation
of v1 vs v2 comparisons is left to the user / the v1-vs-v2 summary step.

Run:
    python -X utf8 scripts/train_step4r_feature_ceiling_v2split.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import train_step4r_feature_ceiling as base  # noqa: E402


def _patch_paths() -> None:
    base.INPUT_FILE = (
        PROJECT_ROOT / "data" / "step4_v2split" / "step4_modeling_dataset.csv"
    )
    base.OUTPUT_DATA_DIR = (
        PROJECT_ROOT / "data" / "step4r_v2split" / "4ra_feature_ceiling"
    )
    base.OUTPUT_REPORT_DIR = (
        PROJECT_ROOT / "reports" / "step4r_v2split" / "4ra_feature_ceiling"
    )
    base.OUTPUT_PRED = {
        "raw": base.OUTPUT_DATA_DIR / "step4r_hgb_predictions_raw.csv",
        "zscore": base.OUTPUT_DATA_DIR / "step4r_hgb_predictions_zscore.csv",
    }
    base.OUTPUT_SCHEMA = {
        "raw": base.OUTPUT_DATA_DIR / "step4r_hgb_schema_outputs_raw.csv",
        "zscore": base.OUTPUT_DATA_DIR / "step4r_hgb_schema_outputs_zscore.csv",
    }
    base.OUTPUT_METRICS_CSV = (
        base.OUTPUT_REPORT_DIR / "step4r_feature_ceiling_metrics.csv"
    )
    base.OUTPUT_CONFUSION_CSV = (
        base.OUTPUT_REPORT_DIR / "step4r_feature_ceiling_confusion.csv"
    )
    base.OUTPUT_REPORT_MD = (
        base.OUTPUT_REPORT_DIR / "step4r_feature_ceiling_results.md"
    )


def main() -> int:
    _patch_paths()
    print("=" * 64)
    print("Step 4R-A (v2split) — HGB ceiling")
    print("=" * 64)
    print(f"  INPUT_FILE       -> {base.INPUT_FILE}")
    print(f"  OUTPUT_DATA_DIR  -> {base.OUTPUT_DATA_DIR}")
    print(f"  OUTPUT_REPORT_DIR-> {base.OUTPUT_REPORT_DIR}")
    print()
    return base.main()


if __name__ == "__main__":
    sys.exit(main())
