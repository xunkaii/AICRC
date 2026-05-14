"""Step 4R-B (v2split) — Schema output under v2 participant split.

Wrapper around generate_step4r_attention_schema_v2.py. Reuses helper
functions and constants from the base module; defines a new main() that
points at v2split paths and uses the v2split modeling dataset for
anchor_reliability / anchor_type lookup.

Run:
    & 'C:\\Users\\user\\anaconda3\\envs\\aicrc_env\\python.exe' -X utf8 \
        scripts/generate_step4r_attention_schema_v2_v2split.py \
        --seed 42 --exp-id b01_5_aug_jitter_scale_strong
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import generate_step4r_attention_schema_v2 as base  # noqa: E402


INPUT_ANCHOR_CSV_V2 = (
    PROJECT_ROOT / "data" / "step4_v2split" / "step4_modeling_dataset.csv"
)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--exp-id", type=str, required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    print("=" * 64)
    print(f"Step 4R-B (v2split) schema | exp={args.exp_id} | seed={args.seed}")
    print("=" * 64)

    rel = Path("experiments") / args.exp_id / f"seed{args.seed}"
    data_dir = PROJECT_ROOT / "data" / "step4r_v2split" / "4rb_attention" / rel
    report_dir = (
        PROJECT_ROOT / "reports" / "step4r_v2split" / "4rb_attention" / rel
    )
    input_npz = data_dir / "logits_calibrated.npz"
    out_schema_csv = data_dir / "schema_outputs_calibrated.csv"
    out_summary_csv = report_dir / "schema_summary.csv"
    out_report_md = report_dir / "schema_results.md"

    if not input_npz.exists():
        raise FileNotFoundError(f"calibrated npz missing: {input_npz}")
    if not INPUT_ANCHOR_CSV_V2.exists():
        raise FileNotFoundError(f"anchor source missing: {INPUT_ANCHOR_CSV_V2}")

    z = np.load(input_npz, allow_pickle=True)
    d = {
        "probs_calibrated": z["probs_calibrated"].astype(np.float64),
        "y": z["y"].astype(np.int64),
        "class_id": z["class_id"].astype(str),
        "sample_id": z["sample_id"].astype(str),
        "participant_id": z["participant_id"].astype(str),
        "posture_canonical": z["posture_canonical"].astype(str),
        "split": z["split"].astype(str),
        "temperature": float(z["temperature"][0]),
        "seed": int(z["seed"][0]),
        "exp_id": args.exp_id,
    }
    print(f"loaded n={d['probs_calibrated'].shape[0]} | T={d['temperature']:.6f}")

    anchor_df = pd.read_csv(
        INPUT_ANCHOR_CSV_V2,
        usecols=["sample_id", "anchor_reliability", "anchor_type"],
        encoding="utf-8-sig",
    ).set_index("sample_id")
    missing = sorted(set(d["sample_id"]) - set(anchor_df.index))
    if missing:
        raise ValueError(
            f"{len(missing)} sample_id missing anchor info; first: {missing[:5]}"
        )
    anchor_df = anchor_df.loc[d["sample_id"]]

    thr = base._derive_thresholds(d["probs_calibrated"], d["y"], d["split"])
    print(f"thresholds: {thr}")

    schema_df = base._build_schema_df(d, anchor_df, thr)
    base._validate(schema_df)

    schema_df.to_csv(out_schema_csv, index=False, encoding="utf-8-sig")
    print(f"saved schema CSV -> {out_schema_csv}")

    summary = base._build_summary(schema_df)
    summary.to_csv(out_summary_csv, index=False, encoding="utf-8-sig")
    print(f"saved summary -> {out_summary_csv}")

    L = []
    L.append(
        f"# Step 4R-B (v2split) schema | exp=`{args.exp_id}` seed=`{args.seed}`"
    )
    L.append("")
    L.append(f"- T = {d['temperature']:.6f}")
    L.append("")
    L.append("## thresholds")
    L.append("")
    for k, v in thr.items():
        L.append(f"- `{k}` = {v:.4f}")
    L.append("")
    L.append("## level distribution (all rows)")
    L.append("")
    L.append("| level | count | rate |")
    L.append("|---|---:|---:|")
    n = len(schema_df)
    for level in ["confident", "hedged", "low", "no_call"]:
        c = int((schema_df["caption_confidence_level"] == level).sum())
        L.append(f"| {level} | {c} | {c / n:.4f} |")
    L.append("")
    L.append("## ambiguity group (all)")
    L.append("")
    L.append("| group | count | rate |")
    L.append("|---|---:|---:|")
    for amb in [
        "confident_C2",
        "within_group_c1_c5_c6",
        "pair_c3_c4",
        "pair_plus_c2_absorption",
        "no_call",
        "uncategorized",
    ]:
        c = int((schema_df["ambiguity_group"] == amb).sum())
        L.append(f"| {amb} | {c} | {c / n:.4f} |")
    out_report_md.write_text("\n".join(L), encoding="utf-8")
    print(f"saved report -> {out_report_md}")

    print()
    print("=" * 64)
    print(f"Done. seed={args.seed}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
