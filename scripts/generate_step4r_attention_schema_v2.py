"""Step 4R-B v2 — Per-seed schema output from calibrated posterior.

Per-seed analogue of `scripts/generate_step4r_attention_schema_outputs.py`.

Reads (read-only):
    data/step4r/4rb_attention/experiments/{exp_id}/seed{N}/logits_calibrated.npz
    data/step4/step4_modeling_dataset.csv   (anchor_reliability, anchor_type)

Writes:
    data/step4r/4rb_attention/experiments/{exp_id}/seed{N}/
        schema_outputs_calibrated.csv
    reports/step4r/4rb_attention/experiments/{exp_id}/seed{N}/
        schema_summary.csv
        schema_results.md

CLI:
    --seed     int (required)
    --exp-id   str (required)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_ANCHOR_CSV = PROJECT_ROOT / "data" / "step4" / "step4_modeling_dataset.csv"

CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
NUM_CLASSES = len(CLASSES)
SPLITS = ["train", "val", "test"]
ALLOWED_POSTURES = {"SA", "CA", "HW"}
ALLOWED_LEVELS = {"confident", "hedged", "low", "no_call"}
ALLOWED_FLAGS = {
    "confident_C2", "within_group_ambiguity_c1_c5_c6",
    "pair_ambiguity_c3_c4", "pair_plus_c2_absorption",
    "anchor_unreliable", "posture_unknown", "low_confidence_no_class_set",
}
ALLOWED_CLASS_SETS = {
    tuple(s) for s in [
        [], ["C2"],
        ["C1", "C5", "C6"], ["C1", "C5"], ["C1", "C6"], ["C5", "C6"],
        ["C3", "C4"], ["C3", "C4", "C2"],
    ]
}
ANCHOR_DEPENDENT_SETS = {tuple(s) for s in [["C2"], ["C3", "C4", "C2"]]}


def _ambiguity_group_label(class_set: list, flags: list, no_call: bool) -> str:
    if no_call:
        return "no_call"
    if "confident_C2" in flags:
        return "confident_C2"
    if "within_group_ambiguity_c1_c5_c6" in flags:
        return "within_group_c1_c5_c6"
    if "pair_plus_c2_absorption" in flags:
        return "pair_plus_c2_absorption"
    if "pair_ambiguity_c3_c4" in flags:
        return "pair_c3_c4"
    return "uncategorized"


def _entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    pp = np.clip(p, eps, 1.0)
    return -(pp * np.log(pp)).sum(axis=-1)


def _derive_thresholds(probs: np.ndarray, y: np.ndarray, split: np.ndarray) -> dict:
    val_mask = split == "val"
    p_val = probs[val_mask]
    y_val = y[val_mask]
    pred_val = p_val.argmax(axis=1)
    idx_C1, idx_C2, idx_C3, idx_C4, idx_C5, idx_C6 = 0, 1, 2, 3, 4, 5

    candidates = np.linspace(0.05, 0.99, 95)
    confident_C2 = None
    for t in candidates:
        above = p_val[:, idx_C2] >= t
        if above.sum() < 5:
            continue
        prec = float((y_val[above] == idx_C2).mean())
        if prec >= 0.60:
            confident_C2 = float(t)
            break
    if confident_C2 is None:
        confident_C2 = 0.99

    mask_c34 = np.isin(pred_val, [idx_C3, idx_C4])
    non_trivial_C2 = (
        float(np.percentile(p_val[mask_c34, idx_C2], 50)) if mask_c34.any() else 0.10
    )

    mask_c156 = np.isin(pred_val, [idx_C1, idx_C5, idx_C6])
    if mask_c156.any():
        sub = p_val[mask_c156]
        min_p = np.minimum.reduce([sub[:, idx_C1], sub[:, idx_C5], sub[:, idx_C6]])
        within_group = float(np.percentile(min_p, 25))
    else:
        within_group = 0.165

    return {
        "confident_C2_threshold": confident_C2,
        "non_trivial_C2_threshold": non_trivial_C2,
        "within_group_threshold": within_group,
        "anchor_suppression_threshold": 0.5,
        "anchor_no_call_threshold": 0.25,
    }


def _assign_schema(row: dict, thr: dict) -> dict:
    posture = row["posture_canonical"]
    if pd.isna(posture) or posture not in ALLOWED_POSTURES:
        return {"class_set": [], "flags": ["posture_unknown"], "level": "no_call",
                "no_call": True, "no_call_reason": "posture_unknown"}
    p_c2 = float(row["p_C2_calibrated"])
    argmax = row["pred_argmax_calibrated"]
    if p_c2 >= thr["confident_C2_threshold"] and argmax == "C2":
        return {"class_set": ["C2"], "flags": ["confident_C2"], "level": "confident",
                "no_call": False, "no_call_reason": ""}
    if argmax in ("C1", "C5", "C6"):
        group = ["C1", "C5", "C6"]
        wgt = thr["within_group_threshold"]
        included = [c for c in group if float(row[f"p_{c}_calibrated"]) >= wgt]
        class_set = included if len(included) >= 2 else group
        return {"class_set": class_set, "flags": ["within_group_ambiguity_c1_c5_c6"],
                "level": "hedged", "no_call": False, "no_call_reason": ""}
    if argmax in ("C3", "C4"):
        if p_c2 >= thr["non_trivial_C2_threshold"]:
            return {"class_set": ["C3", "C4", "C2"], "flags": ["pair_plus_c2_absorption"],
                    "level": "hedged", "no_call": False, "no_call_reason": ""}
        return {"class_set": ["C3", "C4"], "flags": ["pair_ambiguity_c3_c4"],
                "level": "hedged", "no_call": False, "no_call_reason": ""}
    return {"class_set": [], "flags": ["low_confidence_no_class_set"], "level": "no_call",
            "no_call": True, "no_call_reason": "low_confidence_no_class_set"}


def _apply_anchor(result: dict, ar: float, thr: dict) -> dict:
    if pd.isna(ar):
        ar = 1.0
    ar = float(ar)
    flags = list(result["flags"])
    is_unreliable = ar < thr["anchor_suppression_threshold"]
    is_below_no_call = ar < thr["anchor_no_call_threshold"]
    is_anchor_dep = tuple(result["class_set"]) in ANCHOR_DEPENDENT_SETS
    if is_unreliable and "anchor_unreliable" not in flags:
        flags.append("anchor_unreliable")
    if result["no_call"]:
        return {**result, "flags": flags}
    if is_below_no_call and is_anchor_dep:
        return {"class_set": [], "flags": flags, "level": "no_call",
                "no_call": True, "no_call_reason": "anchor_unreliable_anchor_dependent_set"}
    if is_unreliable:
        new_level = result["level"]
        if new_level == "confident":
            new_level = "hedged"
        elif new_level == "hedged":
            new_level = "low"
        return {**result, "flags": flags, "level": new_level}
    return {**result, "flags": flags}


def _build_schema_df(d: dict, anchor_df: pd.DataFrame, thr: dict) -> pd.DataFrame:
    probs_cal = d["probs_calibrated"]
    n = probs_cal.shape[0]
    sorted_idx = np.argsort(-probs_cal, axis=1)
    top1_class = np.array([CLASSES[i] for i in sorted_idx[:, 0]])
    top2_class = np.array([CLASSES[i] for i in sorted_idx[:, 1]])
    top1_prob = np.take_along_axis(probs_cal, sorted_idx[:, :1], axis=1).squeeze(-1)
    top2_prob = np.take_along_axis(probs_cal, sorted_idx[:, 1:2], axis=1).squeeze(-1)
    margin = top1_prob - top2_prob
    pe = _entropy(probs_cal)
    ar_arr = anchor_df["anchor_reliability"].to_numpy()
    at_arr = anchor_df["anchor_type"].to_numpy()
    pred_cal = np.array([CLASSES[i] for i in probs_cal.argmax(axis=1)])

    records = []
    for i in range(n):
        row = {"posture_canonical": d["posture_canonical"][i],
               "pred_argmax_calibrated": pred_cal[i]}
        for j, c in enumerate(CLASSES):
            row[f"p_{c}_calibrated"] = float(probs_cal[i, j])
        base = _assign_schema(row, thr)
        final = _apply_anchor(base, ar_arr[i], thr)
        flags = list(dict.fromkeys(final["flags"]))
        amb_group = _ambiguity_group_label(final["class_set"], flags, final["no_call"])
        cs_idx = [CLASSES.index(c) for c in final["class_set"]] if final["class_set"] else []
        cs_mass = float(probs_cal[i, cs_idx].sum()) if cs_idx else 0.0
        records.append({
            "sample_id": d["sample_id"][i],
            "participant_id": d["participant_id"][i],
            "split": d["split"][i],
            "posture_canonical": d["posture_canonical"][i],
            "true_class_id": d["class_id"][i],
            "top1_class": top1_class[i],
            "top1_prob_calibrated": float(top1_prob[i]),
            "top2_class": top2_class[i],
            "top2_prob_calibrated": float(top2_prob[i]),
            "top1_top2_margin_calibrated": float(margin[i]),
            "predictive_entropy_calibrated": float(pe[i]),
            "class_set": json.dumps(final["class_set"]),
            "class_set_posterior_mass": cs_mass,
            "ambiguity_group": amb_group,
            "uncertainty_flags": json.dumps(flags),
            "no_call": bool(final["no_call"]),
            "no_call_reason": final["no_call_reason"],
            "caption_confidence_level": final["level"],
            "anchor_reliability": float(ar_arr[i]) if not pd.isna(ar_arr[i]) else float("nan"),
            "anchor_type": str(at_arr[i]) if not pd.isna(at_arr[i]) else "",
            "model_source": f"4R-B_BiGRU_Attention_v2_calibrated",
            "temperature": d["temperature"],
            "seed": d["seed"],
            "exp_id": d["exp_id"],
        })
    return pd.DataFrame(records)


def _validate(df: pd.DataFrame) -> None:
    for s in df["class_set"]:
        cs = json.loads(s)
        if tuple(cs) not in ALLOWED_CLASS_SETS:
            raise ValueError(f"class_set outside whitelist: {cs}")
    for s in df["uncertainty_flags"]:
        flags = json.loads(s)
        if len(flags) != len(set(flags)):
            raise ValueError(f"duplicate flags: {flags}")
        for f in flags:
            if f not in ALLOWED_FLAGS:
                raise ValueError(f"unknown flag: {f}")
    bad = sorted(set(df["caption_confidence_level"].unique()) - ALLOWED_LEVELS)
    if bad:
        raise ValueError(f"unknown level: {bad}")
    nc = df["no_call"].astype(bool)
    empty_cs = df["class_set"].apply(lambda s: json.loads(s) == [])
    if (nc != empty_cs).any():
        raise ValueError("no_call <-> empty class_set inconsistency")


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in SPLITS + ["all"]:
        sub = df if s == "all" else df[df["split"] == s]
        if len(sub) == 0:
            continue
        n = len(sub)
        for level in ["confident", "hedged", "low", "no_call"]:
            c = int((sub["caption_confidence_level"] == level).sum())
            rows.append({"split": s, "metric": f"level_{level}_count", "value": c})
            rows.append({"split": s, "metric": f"level_{level}_rate", "value": float(c / n)})
        nc = int(sub["no_call"].astype(bool).sum())
        rows.append({"split": s, "metric": "no_call_count", "value": nc})
        rows.append({"split": s, "metric": "no_call_rate", "value": float(nc / n)})
        for amb in ["confident_C2", "within_group_c1_c5_c6", "pair_c3_c4",
                    "pair_plus_c2_absorption", "no_call", "uncategorized"]:
            c = int((sub["ambiguity_group"] == amb).sum())
            rows.append({"split": s, "metric": f"ambiguity_{amb}_count", "value": c})
            rows.append({"split": s, "metric": f"ambiguity_{amb}_rate", "value": float(c / n)})
        cs_size = sub["class_set"].apply(lambda s: len(json.loads(s)))
        for sz in [0, 1, 2, 3]:
            rows.append({"split": s, "metric": f"class_set_size_{sz}_count", "value": int((cs_size == sz).sum())})
        fc = {f: 0 for f in ALLOWED_FLAGS}
        for fs in sub["uncertainty_flags"]:
            for f in json.loads(fs):
                fc[f] = fc.get(f, 0) + 1
        for f, c in fc.items():
            rows.append({"split": s, "metric": f"flag_{f}_count", "value": int(c)})
            rows.append({"split": s, "metric": f"flag_{f}_rate", "value": float(c / n)})
    return pd.DataFrame(rows)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--exp-id", type=str, required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    print("=" * 64)
    print(f"Step 4R-B v2 schema | exp={args.exp_id} | seed={args.seed}")
    print("=" * 64)

    rel = Path("experiments") / args.exp_id / f"seed{args.seed}"
    data_dir = PROJECT_ROOT / "data" / "step4r" / "4rb_attention" / rel
    report_dir = PROJECT_ROOT / "reports" / "step4r" / "4rb_attention" / rel
    input_npz = data_dir / "logits_calibrated.npz"
    out_schema_csv = data_dir / "schema_outputs_calibrated.csv"
    out_summary_csv = report_dir / "schema_summary.csv"
    out_report_md = report_dir / "schema_results.md"

    if not input_npz.exists():
        raise FileNotFoundError(f"calibrated npz missing: {input_npz}")
    if not INPUT_ANCHOR_CSV.exists():
        raise FileNotFoundError(f"anchor source missing: {INPUT_ANCHOR_CSV}")

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
        INPUT_ANCHOR_CSV, usecols=["sample_id", "anchor_reliability", "anchor_type"],
        encoding="utf-8-sig"
    ).set_index("sample_id")
    missing = sorted(set(d["sample_id"]) - set(anchor_df.index))
    if missing:
        raise ValueError(f"{len(missing)} sample_id missing anchor info; first: {missing[:5]}")
    anchor_df = anchor_df.loc[d["sample_id"]]

    thr = _derive_thresholds(d["probs_calibrated"], d["y"], d["split"])
    print(f"thresholds: {thr}")

    schema_df = _build_schema_df(d, anchor_df, thr)
    _validate(schema_df)

    schema_df.to_csv(out_schema_csv, index=False, encoding="utf-8-sig")
    print(f"saved schema CSV -> {out_schema_csv}")

    summary = _build_summary(schema_df)
    summary.to_csv(out_summary_csv, index=False, encoding="utf-8-sig")
    print(f"saved summary -> {out_summary_csv}")

    L = []
    L.append(f"# Step 4R-B v2 schema | exp=`{args.exp_id}` seed=`{args.seed}`")
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
    for amb in ["confident_C2", "within_group_c1_c5_c6", "pair_c3_c4",
                "pair_plus_c2_absorption", "no_call", "uncategorized"]:
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
