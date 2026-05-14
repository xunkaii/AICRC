"""Thin driver to re-run Step 7_v2 caption generation against a new schema CSV.

Why this exists:
    `scripts/generate_step7_v2_captions.py` hard-codes the input schema CSV
    path (the legacy single-seed 4R-B output) and the output directories.
    Project rule: don't modify existing scripts. So this driver loads that
    script as a module, monkey-patches the three module-level path constants
    before calling `main()`, and runs the captioning pipeline against the
    new schema CSV produced by a v2 experiment (e.g., b01_5/seed42).

Behavior:
    - Loads scripts/generate_step7_v2_captions.py via importlib (no modification).
    - Overrides:
        SCHEMA_CSV        -> --schema-csv argument
        OUTPUT_DATA_DIR   -> data/step7_v2/<out-tag>/
        OUTPUT_REPORT_DIR -> reports/step7_v2/<out-tag>/
    - Forwards the rest of the CLI (--mode, --provider, ...) to the original.
    - Legacy outputs at data/step7_v2/*.csv and reports/step7_v2/*.md are
      left untouched (this driver only writes into the <out-tag> subdir).

Run:
    & python.exe -X utf8 scripts/rerun_step7_v2_for_experiment.py ^
        --schema-csv data/step4r/4rb_attention/experiments/b01_5_aug_jitter_scale_strong/seed42/schema_outputs_calibrated.csv ^
        --out-tag b01_5_seed42 ^
        --mode full --provider mock
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORIG_SCRIPT = PROJECT_ROOT / "scripts" / "generate_step7_v2_captions.py"


def _parse_driver_args() -> tuple[argparse.Namespace, list[str]]:
    """Pull --schema-csv and --out-tag out of argv; forward the rest."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--schema-csv", type=str, required=True)
    p.add_argument("--out-tag", type=str, required=True)
    args, forward = p.parse_known_args()
    return args, forward


def _load_original_module():
    spec = importlib.util.spec_from_file_location(
        "_step7_v2_orig", str(ORIG_SCRIPT)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec for {ORIG_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    driver_args, forward_argv = _parse_driver_args()

    schema_csv = Path(driver_args.schema_csv).resolve()
    if not schema_csv.exists():
        raise FileNotFoundError(f"schema CSV not found: {schema_csv}")

    out_data = PROJECT_ROOT / "data" / "step7_v2" / driver_args.out_tag
    out_report = PROJECT_ROOT / "reports" / "step7_v2" / driver_args.out_tag
    out_data.mkdir(parents=True, exist_ok=True)
    out_report.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("Step 7_v2 caption rerun (path-overridden driver)")
    print("=" * 64)
    print(f"schema_csv  -> {schema_csv}")
    print(f"out_data    -> {out_data}")
    print(f"out_report  -> {out_report}")
    print(f"forward argv -> {forward_argv}")
    print()

    # Force UTF-8 stdout/stderr (the original does this too but only after import).
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    mod = _load_original_module()

    # Monkey-patch path constants BEFORE invoking main().
    mod.SCHEMA_CSV = schema_csv
    mod.OUTPUT_DATA_DIR = out_data
    mod.OUTPUT_REPORT_DIR = out_report

    # Forward CLI to the original's argparse via sys.argv replacement.
    sys.argv = [str(ORIG_SCRIPT)] + list(forward_argv)
    rc = mod.main()
    return int(rc) if rc is not None else 0


if __name__ == "__main__":
    sys.exit(main())
