"""Step 4R-A (v2split) — Temperature scaling for HGB ceiling under v2 split.

Thin wrapper around scripts/calibrate_step4r_hgb_temperature.py. Imports the
base module (read-only, unmodified) and overrides module-level path constants
to read the v2split HGB predictions and write the v2split calibrated outputs.

Inputs (v2split):
    data/step4r_v2split/4ra_feature_ceiling/step4r_hgb_predictions_{raw,zscore}.csv

Outputs (v2split):
    data/step4r_v2split/4ra_feature_ceiling/calibrated/
        step4r_hgb_predictions_calibrated_{raw,zscore}.csv
    reports/step4r_v2split/4ra_feature_ceiling/calibrated/
        step4r_hgb_temperature_scaling_metrics.csv
        step4r_hgb_temperature_scaling_results.md

Run:
    python -X utf8 scripts/calibrate_step4r_hgb_temperature_v2split.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import calibrate_step4r_hgb_temperature as base  # noqa: E402


def _patch_paths() -> None:
    base.INPUT_DIR = (
        PROJECT_ROOT / "data" / "step4r_v2split" / "4ra_feature_ceiling"
    )
    base.OUTPUT_DATA_DIR = base.INPUT_DIR / "calibrated"
    base.OUTPUT_REPORT_DIR = (
        PROJECT_ROOT / "reports" / "step4r_v2split" / "4ra_feature_ceiling"
        / "calibrated"
    )
    base.OUTPUT_METRICS_CSV = (
        base.OUTPUT_REPORT_DIR / "step4r_hgb_temperature_scaling_metrics.csv"
    )
    base.OUTPUT_REPORT_MD = (
        base.OUTPUT_REPORT_DIR / "step4r_hgb_temperature_scaling_results.md"
    )
    base.INPUT_PRED = {
        b: base.INPUT_DIR / f"step4r_hgb_predictions_{b}.csv" for b in base.BRANCHES
    }
    base.OUTPUT_PRED = {
        b: base.OUTPUT_DATA_DIR / f"step4r_hgb_predictions_calibrated_{b}.csv"
        for b in base.BRANCHES
    }


def main() -> int:
    _patch_paths()
    print("=" * 64)
    print("Step 4R-A (v2split) — HGB temperature scaling")
    print("=" * 64)
    print(f"  INPUT_DIR        -> {base.INPUT_DIR}")
    print(f"  OUTPUT_DATA_DIR  -> {base.OUTPUT_DATA_DIR}")
    print(f"  OUTPUT_REPORT_DIR-> {base.OUTPUT_REPORT_DIR}")
    print()
    return base.main()


if __name__ == "__main__":
    sys.exit(main())
