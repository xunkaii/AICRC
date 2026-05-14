"""Step 4R-C (v2split) Day 2b — frozen 4R-B b01.5/seed42 IMU embeddings (v2split).

Wrapper around build_step4rc_imu_embeddings.py. Monkey-patches input paths
(sequence dataset, checkpoint, text corpus) and output to v2split tree.

Run:
    & 'C:\\Users\\user\\anaconda3\\envs\\dl_env\\python.exe' -X utf8 \
        scripts/build_step4rc_imu_embeddings_v2split.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import build_step4rc_imu_embeddings as base  # noqa: E402


def _patch() -> None:
    base.INPUT_NPZ = (
        PROJECT_ROOT / "data" / "step4r_v2split" / "4rb_attention"
        / "step4r_sequence_dataset.npz"
    )
    base.INPUT_CKPT = (
        PROJECT_ROOT / "checkpoints" / "step4r_v2split" / "4rb_attention"
        / "experiments" / "b01_5_aug_jitter_scale_strong" / "seed42" / "best.pt"
    )
    base.INPUT_TEXT_CORPUS = (
        PROJECT_ROOT / "data" / "step4r_v2split" / "4rc_contrastive_optional"
        / "text_corpus.csv"
    )
    base.OUTPUT_DIR = (
        PROJECT_ROOT / "data" / "step4r_v2split" / "4rc_contrastive_optional"
    )
    base.OUTPUT_NPZ = base.OUTPUT_DIR / "imu_embeddings.npz"


def main() -> int:
    _patch()
    print("=" * 64)
    print("Step 4R-C (v2split) Day 2b — IMU embeddings")
    print(f"  INPUT_NPZ          -> {base.INPUT_NPZ}")
    print(f"  INPUT_CKPT         -> {base.INPUT_CKPT}")
    print(f"  INPUT_TEXT_CORPUS  -> {base.INPUT_TEXT_CORPUS}")
    print(f"  OUTPUT_NPZ         -> {base.OUTPUT_NPZ}")
    print("=" * 64)
    return base.main()


if __name__ == "__main__":
    sys.exit(main())
