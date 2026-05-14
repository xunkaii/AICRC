"""Step 4R-C (v2split) Day 3 — Contrastive projection head training (v2split).

Wrapper around train_step4rc_contrastive.py. Monkey-patches input (imu/text
embeddings, corpus) and output (checkpoint, joint emb, report) paths to
v2split tree.

Run:
    & 'C:\\Users\\user\\anaconda3\\envs\\dl_env\\python.exe' -X utf8 \
        scripts/train_step4rc_contrastive_v2split.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import train_step4rc_contrastive as base  # noqa: E402


def _patch() -> None:
    v2_data = (
        PROJECT_ROOT / "data" / "step4r_v2split" / "4rc_contrastive_optional"
    )
    v2_report = (
        PROJECT_ROOT / "reports" / "step4r_v2split" / "4rc_contrastive_optional"
    )
    v2_ckpt = (
        PROJECT_ROOT / "checkpoints" / "step4r_v2split" / "4rc_contrastive_optional"
    )
    base.INPUT_IMU = v2_data / "imu_embeddings.npz"
    base.INPUT_TEXT = v2_data / "text_embeddings.npz"
    base.INPUT_CORPUS = v2_data / "text_corpus.csv"
    base.OUTPUT_CKPT_DIR = v2_ckpt
    base.OUTPUT_DATA_DIR = v2_data
    base.OUTPUT_REPORT_DIR = v2_report
    base.OUTPUT_CKPT = v2_ckpt / "projection_head.pt"
    base.OUTPUT_JOINT_NPZ = v2_data / "joint_embeddings.npz"
    base.OUTPUT_LOG_CSV = v2_report / "training_log.csv"
    base.OUTPUT_REPORT_MD = v2_report / "results.md"


def main() -> int:
    _patch()
    print("=" * 64)
    print("Step 4R-C (v2split) Day 3 — Contrastive projection training")
    print(f"  INPUT_IMU   -> {base.INPUT_IMU}")
    print(f"  INPUT_TEXT  -> {base.INPUT_TEXT}")
    print(f"  OUTPUT_CKPT -> {base.OUTPUT_CKPT}")
    print("=" * 64)
    return base.main()


if __name__ == "__main__":
    sys.exit(main())
