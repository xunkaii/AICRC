"""Build v1-ratio (36/8/8) participant-level split manifest with parametrized seed.

Reads (read-only):
    data/manifest_split.csv

Writes (new file only; original manifest is NOT modified):
    data/step4r_v1seed_robustness/s{SEED}/manifest_split.csv

Policy:
    - 52 participants are shuffled with numpy.random.default_rng(seed=split_seed).
      First 36 -> train, next 8 -> val, last 8 -> test. Same ratio as v1_36_8_8.
    - split_version = f"v1_36_8_8_s{split_seed}".

Run:
    python scripts/build_manifest_split_v1seed.py --split-seed 7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_MANIFEST = PROJECT_ROOT / "data" / "manifest_split.csv"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "step4r_v1seed_robustness"

N_TRAIN = 36
N_VAL = 8
N_TEST = 8


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split-seed", type=int, required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    split_seed = args.split_seed
    split_version = f"v1_36_8_8_s{split_seed}"

    out_dir = OUTPUT_ROOT / f"s{split_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "manifest_split.csv"

    print("=" * 64)
    print(f"Build manifest (36/8/8) with split_seed={split_seed}")
    print(f"  -> {out_path}")
    print("=" * 64)

    if not INPUT_MANIFEST.exists():
        raise FileNotFoundError(f"input manifest missing: {INPUT_MANIFEST}")

    df = pd.read_csv(INPUT_MANIFEST, encoding="utf-8-sig")
    print(f"loaded n={len(df)} rows")

    participants = sorted(df["participant_id"].unique())
    n_participants = len(participants)
    if n_participants != N_TRAIN + N_VAL + N_TEST:
        raise ValueError(
            f"participant count {n_participants} != "
            f"{N_TRAIN + N_VAL + N_TEST}"
        )

    rng = np.random.default_rng(split_seed)
    shuffled = list(participants)
    rng.shuffle(shuffled)

    train_pids = sorted(shuffled[:N_TRAIN])
    val_pids = sorted(shuffled[N_TRAIN:N_TRAIN + N_VAL])
    test_pids = sorted(shuffled[N_TRAIN + N_VAL:])

    pid_to_split: dict[str, str] = {}
    for p in train_pids:
        pid_to_split[p] = "train"
    for p in val_pids:
        pid_to_split[p] = "val"
    for p in test_pids:
        pid_to_split[p] = "test"

    print()
    print("=== participant assignment ===")
    print(f"train ({len(train_pids):2d}): {train_pids}")
    print(f"val   ({len(val_pids):2d}): {val_pids}")
    print(f"test  ({len(test_pids):2d}): {test_pids}")

    out_df = df.copy()
    out_df["split"] = out_df["participant_id"].map(pid_to_split)
    if out_df["split"].isna().any():
        raise RuntimeError("some rows did not receive a split assignment")
    out_df["split_seed"] = split_seed
    out_df["split_version"] = split_version
    out_df["split_group_key"] = out_df["participant_id"]

    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print()
    print(f"saved -> {out_path}  ({len(out_df)} rows)")

    print()
    print("=== diagnostics ===")
    print("samples per split:")
    print(out_df["split"].value_counts().sort_index().to_string())
    print()
    print("class_id x split (sample counts):")
    print(pd.crosstab(out_df["class_id"], out_df["split"]).to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
