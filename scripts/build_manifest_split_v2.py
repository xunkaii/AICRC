"""Build v2 participant-level split manifest (train 34 / val 4 / test 14).

Reads (read-only):
    data/manifest_split.csv

Writes (new file only; original manifest is NOT modified):
    data/manifest_split_v2.csv

Policy:
    - 52 participants are shuffled with numpy.random.default_rng(seed=42).
      First 4 -> val, next 14 -> test, remaining 34 -> train.
    - Original `split` / `split_version` / `split_seed` / `split_group_key`
      columns are OVERWRITTEN with the new assignment. All other columns
      (signal_path, n_rows, include, ...) are preserved verbatim.
    - participant_id remains the grouping key (no leakage across splits).
    - This script is read-only on data/manifest_split.csv and writes one
      new CSV at data/manifest_split_v2.csv.

Rationale (why v2 is meaningful even though all 52 participants do 6 classes):
    - v1 (36/8/8) has only 8 test participants, which gives high variance on
      test F1 between runs. v2 grows test to 14 (~75% larger).
    - val is shrunk to 4 because hyperparameter / temperature decisions in
      this project are coarse-grained and 4 is enough to detect early-stop /
      overfit signal at lower variance cost than test reduction.
    - Class-stratify is unnecessary: every participant performs all 6 classes
      (verified via groupby on the source manifest), so any participant
      partition preserves class balance automatically.

Run:
    python scripts/build_manifest_split_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_MANIFEST = PROJECT_ROOT / "data" / "manifest_split.csv"
OUTPUT_MANIFEST = PROJECT_ROOT / "data" / "manifest_split_v2.csv"

SPLIT_VERSION = "v2_34_4_14_s42"
SPLIT_SEED = 42

N_VAL = 4
N_TEST = 14
N_TRAIN = 34  # = 52 - N_VAL - N_TEST


def main() -> int:
    print("=" * 64)
    print("Build manifest_split_v2.csv (34 train / 4 val / 14 test)")
    print("=" * 64)

    if not INPUT_MANIFEST.exists():
        raise FileNotFoundError(f"input manifest missing: {INPUT_MANIFEST}")

    df = pd.read_csv(INPUT_MANIFEST, encoding="utf-8-sig")
    print(f"loaded n={len(df)} rows from {INPUT_MANIFEST}")

    participants = sorted(df["participant_id"].unique())
    n_participants = len(participants)
    print(f"unique participants: {n_participants}")
    if n_participants != N_VAL + N_TEST + N_TRAIN:
        raise ValueError(
            f"participant count {n_participants} != "
            f"{N_VAL + N_TEST + N_TRAIN} (val+test+train)"
        )

    rng = np.random.default_rng(SPLIT_SEED)
    shuffled = list(participants)
    rng.shuffle(shuffled)

    val_pids = sorted(shuffled[:N_VAL])
    test_pids = sorted(shuffled[N_VAL:N_VAL + N_TEST])
    train_pids = sorted(shuffled[N_VAL + N_TEST:])

    pid_to_split: dict[str, str] = {}
    for p in val_pids:
        pid_to_split[p] = "val"
    for p in test_pids:
        pid_to_split[p] = "test"
    for p in train_pids:
        pid_to_split[p] = "train"

    print()
    print("=== participant assignment ===")
    print(f"val   ({len(val_pids):2d}): {val_pids}")
    print(f"test  ({len(test_pids):2d}): {test_pids}")
    print(f"train ({len(train_pids):2d}): {train_pids}")

    # Apply new split assignment to the manifest. Other columns untouched.
    out_df = df.copy()
    out_df["split"] = out_df["participant_id"].map(pid_to_split)
    if out_df["split"].isna().any():
        raise RuntimeError("some rows did not receive a split assignment")
    out_df["split_seed"] = SPLIT_SEED
    out_df["split_version"] = SPLIT_VERSION
    out_df["split_group_key"] = out_df["participant_id"]

    out_df.to_csv(OUTPUT_MANIFEST, index=False, encoding="utf-8-sig")
    print()
    print(f"saved -> {OUTPUT_MANIFEST}  ({len(out_df)} rows)")

    # ----- diagnostics -----
    print()
    print("=== diagnostics ===")
    print()
    print("samples per split:")
    print(out_df["split"].value_counts().sort_index().to_string())
    print()
    print("participants per split:")
    print(out_df.groupby("split")["participant_id"].nunique().to_string())
    print()
    print("class_id x split (sample counts):")
    print(
        pd.crosstab(out_df["class_id"], out_df["split"]).to_string()
    )
    print()
    print("posture_canonical x split (sample counts):")
    print(
        pd.crosstab(out_df["posture_canonical"], out_df["split"]).to_string()
    )
    print()
    print("include x split (only include=True rows are used downstream):")
    print(
        pd.crosstab(out_df["include"], out_df["split"]).to_string()
    )

    # Sanity: every participant appears in exactly one split.
    pid_split_counts = out_df.groupby("participant_id")["split"].nunique()
    if (pid_split_counts != 1).any():
        raise RuntimeError("some participant_id maps to multiple splits")
    print()
    print(
        f"OK: each of {n_participants} participants is in exactly one split."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
