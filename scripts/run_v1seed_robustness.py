"""Orchestrate v1-ratio split-robustness study: 4 split seeds x 3 learn seeds.

For each split_seed in {7, 123, 2024, 7777}:
    1. build manifest (36/8/8 with this split_seed)
    2. build sequence dataset (.npz with train channel mean/std from this split)
    3. for each learn_seed in {42, 43, 44}:
         a. train b01.5 BiGRU+Attention
         b. temperature scaling

All runs use:
    - exp_id = b01_5_aug_jitter_scale_strong
    - aug    = jitter_scale, jitter_sigma 0.05, scale [0.85, 1.15]

Outputs land under:
    data/step4r_v1seed_robustness/s{SEED}/...
    reports/step4r_v1seed_robustness/s{SEED}/...
    checkpoints/step4r_v1seed_robustness/s{SEED}/...

Run:
    & 'C:\\Users\\user\\anaconda3\\envs\\aicrc_env\\python.exe' -X utf8 \
        scripts/run_v1seed_robustness.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable  # this orchestrator should be run with aicrc_env's python

SPLIT_SEEDS = [7, 123, 2024, 7777]
LEARN_SEEDS = [42, 43, 44]
EXP_ID = "b01_5_aug_jitter_scale_strong"
AUG_ARGS = [
    "--aug", "jitter_scale",
    "--jitter-sigma", "0.05",
    "--scale-low", "0.85",
    "--scale-high", "1.15",
]


def _run(cmd: list[str], label: str) -> None:
    print()
    print("-" * 64)
    print(f">>> {label}")
    print(f"    cmd: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with exit {proc.returncode}")
    print(f"    done in {elapsed:.1f}s")


def main() -> int:
    overall_t0 = time.time()
    print("=" * 64)
    print("v1-ratio split-robustness orchestrator")
    print(f"  split seeds: {SPLIT_SEEDS}")
    print(f"  learn seeds: {LEARN_SEEDS}")
    print(f"  exp_id:      {EXP_ID}")
    print(f"  python:      {PY}")
    print("=" * 64)

    for split_seed in SPLIT_SEEDS:
        split_label = f"s{split_seed}"
        print()
        print("=" * 64)
        print(f"split_seed = {split_seed}")
        print("=" * 64)

        # 1. Manifest
        _run(
            [PY, "-X", "utf8", "scripts/build_manifest_split_v1seed.py",
             "--split-seed", str(split_seed)],
            f"{split_label} | manifest",
        )
        # 2. Sequence dataset
        _run(
            [PY, "-X", "utf8", "scripts/build_step4r_sequence_dataset_v1seed.py",
             "--split-seed", str(split_seed)],
            f"{split_label} | sequence_dataset",
        )
        # 3. train + calibrate for each learn seed
        for learn_seed in LEARN_SEEDS:
            _run(
                [PY, "-X", "utf8",
                 "scripts/train_step4r_attention_rnn_v2_v1seed.py",
                 "--split-seed", str(split_seed),
                 "--seed", str(learn_seed),
                 "--exp-id", EXP_ID,
                 *AUG_ARGS],
                f"{split_label} | train seed={learn_seed}",
            )
            _run(
                [PY, "-X", "utf8",
                 "scripts/calibrate_step4r_attention_temperature_v2_v1seed.py",
                 "--split-seed", str(split_seed),
                 "--seed", str(learn_seed),
                 "--exp-id", EXP_ID],
                f"{split_label} | calibrate seed={learn_seed}",
            )

    elapsed = time.time() - overall_t0
    print()
    print("=" * 64)
    print(f"ALL DONE in {elapsed / 60:.1f} min")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
