"""Step 4R-C (v2split) Day 2a — Korean SBERT text embeddings for v2split corpus.

Wrapper around build_step4rc_text_embeddings.py. Monkey-patches input/output
paths to v2split tree.

Run:
    & 'C:\\Users\\user\\anaconda3\\envs\\dl_env\\python.exe' -X utf8 \
        scripts/build_step4rc_text_embeddings_v2split.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import build_step4rc_text_embeddings as base  # noqa: E402


def _patch() -> None:
    base.INPUT_CSV = (
        PROJECT_ROOT / "data" / "step4r_v2split" / "4rc_contrastive_optional"
        / "text_corpus.csv"
    )
    base.OUTPUT_DIR = (
        PROJECT_ROOT / "data" / "step4r_v2split" / "4rc_contrastive_optional"
    )
    base.OUTPUT_NPZ = base.OUTPUT_DIR / "text_embeddings.npz"
    base.OUTPUT_SIM_CSV = base.OUTPUT_DIR / "text_unique_similarity.csv"


def main() -> int:
    _patch()
    print("=" * 64)
    print("Step 4R-C (v2split) Day 2a — text embeddings")
    print(f"  INPUT_CSV  -> {base.INPUT_CSV}")
    print(f"  OUTPUT_NPZ -> {base.OUTPUT_NPZ}")
    print("=" * 64)
    return base.main()


if __name__ == "__main__":
    sys.exit(main())
