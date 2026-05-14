"""Step 4R-C (v2split) Day 1 — Build text corpus from v2split b01.5/seed42 schema.

Wrapper around build_step4rc_text_corpus.py. Monkey-patches INPUT_SCHEMA_CSV
and OUTPUT paths to point at the v2split tree.

Run:
    & 'C:\\Users\\user\\anaconda3\\envs\\dl_env\\python.exe' -X utf8 \
        scripts/build_step4rc_text_corpus_v2split.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import build_step4rc_text_corpus as base  # noqa: E402


def _patch() -> None:
    base.INPUT_SCHEMA_CSV = (
        PROJECT_ROOT / "data" / "step4r_v2split" / "4rb_attention" / "experiments"
        / "b01_5_aug_jitter_scale_strong" / "seed42"
        / "schema_outputs_calibrated.csv"
    )
    base.OUTPUT_DIR = (
        PROJECT_ROOT / "data" / "step4r_v2split" / "4rc_contrastive_optional"
    )
    base.OUTPUT_CSV = base.OUTPUT_DIR / "text_corpus.csv"


def main() -> int:
    _patch()
    print("=" * 64)
    print("Step 4R-C (v2split) Day 1 — text corpus")
    print(f"  INPUT_SCHEMA_CSV -> {base.INPUT_SCHEMA_CSV}")
    print(f"  OUTPUT_CSV       -> {base.OUTPUT_CSV}")
    print("=" * 64)
    return base.main()


if __name__ == "__main__":
    sys.exit(main())
