"""Step 4 - Generate Step 3 schema-compliant output CSVs.

Reads (read-only):
    data/step4/step4_predictions_raw.csv
    data/step4/step4_predictions_zscore.csv
    reports/step4/step4_threshold_calibration.md   (referenced — values inlined below)

Writes:
    data/step4/step4_schema_outputs_raw.csv
    data/step4/step4_schema_outputs_zscore.csv

Behavior locked by:
    reports/step3/step3_output_schema_uncertainty_policy.md  (sec. 3, 4, 6, 7, 8, 9)
    reports/step4/step4_modeling_calibration_plan.md         (sec. 9)
    reports/step4/step4_threshold_calibration.md             (threshold candidates)

Threshold values below are the val-derived candidates published in
step4_threshold_calibration.md. They are NOT a commit; they are the
values this generator uses. If those candidates are revised in a later
step, this constant block is the only place that needs editing.

This script does NOT train models, does NOT recalibrate thresholds, and
does NOT write captions. Schema validation is performed inline; the
authoritative validator lives in scripts/validate_step4_schema_outputs.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PREDICTIONS = {
    "raw": PROJECT_ROOT / "data" / "step4" / "step4_predictions_raw.csv",
    "zscore": PROJECT_ROOT / "data" / "step4" / "step4_predictions_zscore.csv",
}

OUTPUT_FILES = {
    "raw": PROJECT_ROOT / "data" / "step4" / "step4_schema_outputs_raw.csv",
    "zscore": PROJECT_ROOT / "data" / "step4" / "step4_schema_outputs_zscore.csv",
}

EXPECTED_ROW_COUNT = 9275

CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
P_COLS = [f"p_{c}" for c in CLASSES]

ALLOWED_POSTURES = {"SA", "CA", "HW"}
ALLOWED_LEVELS = {"confident", "hedged", "low", "no_call"}
ALLOWED_FLAGS = {
    "confident_C2",
    "within_group_ambiguity_c1_c5_c6",
    "pair_ambiguity_c3_c4",
    "pair_plus_c2_absorption",
    "anchor_unreliable",
    "posture_unknown",
    "low_confidence_no_class_set",
}

ALLOWED_CLASS_SETS = {
    tuple(s) for s in [
        [],
        ["C2"],
        ["C1", "C5", "C6"],
        ["C1", "C5"],
        ["C1", "C6"],
        ["C5", "C6"],
        ["C3", "C4"],
        ["C3", "C4", "C2"],
    ]
}

ANCHOR_DEPENDENT_SETS = {
    tuple(s) for s in [["C2"], ["C3", "C4", "C2"]]
}

# Threshold candidates (val-derived) copied from
# reports/step4/step4_threshold_calibration.md.
THRESHOLDS = {
    "raw": {
        "confident_C2_threshold": 0.3800,
        "non_trivial_C2_threshold": 0.1123,
        "within_group_threshold": 0.1650,
        "anchor_suppression_threshold": 0.5000,
        "anchor_no_call_threshold": 0.2500,
    },
    "zscore": {
        "confident_C2_threshold": 0.3900,
        "non_trivial_C2_threshold": 0.1059,
        "within_group_threshold": 0.1646,
        "anchor_suppression_threshold": 0.5000,
        "anchor_no_call_threshold": 0.2500,
    },
}

CARRY_COLS = [
    "sample_id",
    "rep_id",
    "participant_id",
    "class_id",
    "posture",
    "split",
    "anchor_reliability",
    "anchor_type",
] + P_COLS + ["pred_argmax_debug"]

OUTPUT_COL_ORDER = CARRY_COLS + [
    "class_set_prediction",
    "uncertainty_flags",
    "caption_confidence_level",
    "no_call",
]


# ---------------------------------------------------------------------------
# Schema assignment per row
# ---------------------------------------------------------------------------

def _assign_schema(row: dict, thr: dict) -> dict:
    """Apply rules 0/1/2/3/4 to determine class_set / flags / level / no_call.

    Rule precedence (per Step 4 plan):
        0. posture invalid (NaN or outside {SA,CA,HW}) -> no_call
        1. p_C2 >= confident_C2_threshold AND argmax == C2 -> [C2]
        2. argmax in {C1, C5, C6} -> within-group subset
        3. argmax in {C3, C4} -> pair (with C2 absorption hedge)
        4. otherwise -> low_confidence_no_class_set, no_call
    """
    posture = row["posture"]
    # Rule 0: posture pre-check.
    if pd.isna(posture) or posture not in ALLOWED_POSTURES:
        return {
            "class_set": [],
            "flags": ["posture_unknown"],
            "level": "no_call",
            "no_call": True,
        }

    p_c2 = float(row["p_C2"])
    argmax = row["pred_argmax_debug"]

    # Rule 1: confident C2.
    if p_c2 >= thr["confident_C2_threshold"] and argmax == "C2":
        return {
            "class_set": ["C2"],
            "flags": ["confident_C2"],
            "level": "confident",
            "no_call": False,
        }

    # Rule 2: C1/C5/C6-argmax -> within-group subset.
    if argmax in ("C1", "C5", "C6"):
        group = ["C1", "C5", "C6"]
        wgt = thr["within_group_threshold"]
        included = [c for c in group if float(row[f"p_{c}"]) >= wgt]
        class_set = included if len(included) >= 2 else group
        return {
            "class_set": class_set,
            "flags": ["within_group_ambiguity_c1_c5_c6"],
            "level": "hedged",
            "no_call": False,
        }

    # Rule 3: C3/C4-argmax -> pair, optionally with C2 absorption hedge.
    if argmax in ("C3", "C4"):
        if p_c2 >= thr["non_trivial_C2_threshold"]:
            return {
                "class_set": ["C3", "C4", "C2"],
                "flags": ["pair_plus_c2_absorption"],
                "level": "hedged",
                "no_call": False,
            }
        return {
            "class_set": ["C3", "C4"],
            "flags": ["pair_ambiguity_c3_c4"],
            "level": "hedged",
            "no_call": False,
        }

    # Rule 4: catch-all (e.g., argmax=C2 but p_C2 below confident threshold).
    return {
        "class_set": [],
        "flags": ["low_confidence_no_class_set"],
        "level": "no_call",
        "no_call": True,
    }


def _apply_anchor(result: dict, row: dict, thr: dict) -> dict:
    """Anchor post-processing.

    - reliability < suppression_threshold -> add 'anchor_unreliable' flag
    - reliability < no_call_threshold AND class_set is anchor-dependent
      ([C2] or [C3, C4, C2]) -> force no_call
    - 'anchor_unreliable' alone (without no_call) downgrades the level:
        confident -> hedged, hedged -> low (low / no_call unchanged)
    """
    ar = row["anchor_reliability"]
    if pd.isna(ar):
        raise ValueError(
            f"anchor_reliability is NaN for sample_id={row.get('sample_id')!r}; "
            "build_step4_modeling_dataset.py should have caught this."
        )
    ar = float(ar)

    flags = list(result["flags"])
    is_unreliable = ar < thr["anchor_suppression_threshold"]
    is_below_no_call = ar < thr["anchor_no_call_threshold"]
    is_anchor_dep = tuple(result["class_set"]) in ANCHOR_DEPENDENT_SETS

    if is_unreliable and "anchor_unreliable" not in flags:
        flags.append("anchor_unreliable")

    # If already no_call (rule 0 or rule 4), just keep the (possibly extended)
    # flag list. Class set, level, and no_call status do not change.
    if result["no_call"]:
        return {
            "class_set": result["class_set"],
            "flags": flags,
            "level": result["level"],
            "no_call": True,
        }

    # Anchor-driven no_call: very low reliability AND anchor-dependent set.
    if is_below_no_call and is_anchor_dep:
        # 'anchor_unreliable' is already in flags (since no_call < suppression).
        return {
            "class_set": [],
            "flags": flags,
            "level": "no_call",
            "no_call": True,
        }

    # Suppression-only (and anchor-independent or above no_call): downgrade.
    if is_unreliable:
        new_level = result["level"]
        if new_level == "confident":
            new_level = "hedged"
        elif new_level == "hedged":
            new_level = "low"
        # 'low' / 'no_call' unchanged.
        return {
            "class_set": result["class_set"],
            "flags": flags,
            "level": new_level,
            "no_call": False,
        }

    return {
        "class_set": result["class_set"],
        "flags": flags,
        "level": result["level"],
        "no_call": False,
    }


def _process_row(row: dict, thr: dict) -> dict:
    base = _assign_schema(row, thr)
    final = _apply_anchor(base, row, thr)
    # Dedupe flags while preserving order.
    deduped_flags = list(dict.fromkeys(final["flags"]))
    return {
        "class_set_prediction": json.dumps(final["class_set"]),
        "uncertainty_flags": json.dumps(deduped_flags),
        "caption_confidence_level": final["level"],
        "no_call": bool(final["no_call"]),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_thresholds(thr: dict, cond: str) -> None:
    if thr["anchor_no_call_threshold"] >= thr["anchor_suppression_threshold"]:
        raise ValueError(
            f"[{cond}] threshold relation violated: "
            f"anchor_no_call_threshold ({thr['anchor_no_call_threshold']}) "
            f"must be < anchor_suppression_threshold "
            f"({thr['anchor_suppression_threshold']})."
        )


def _validate_output(out: pd.DataFrame, cond: str) -> None:
    if len(out) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"[{cond}] row count {len(out)} != {EXPECTED_ROW_COUNT}."
        )
    # p sums.
    p = out[P_COLS].to_numpy(dtype="float64")
    if np.isnan(p).any() or np.isinf(p).any() or (p < 0).any():
        raise ValueError(f"[{cond}] probability columns contain non-finite or negative values.")
    sums = p.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-6):
        n_bad = int((~np.isclose(sums, 1.0, atol=1e-6)).sum())
        raise ValueError(
            f"[{cond}] {n_bad} rows have probability sum != 1 (within 1e-6)."
        )

    # class_set whitelist.
    bad_cs = []
    for i, s in enumerate(out["class_set_prediction"]):
        cs = json.loads(s)
        if tuple(cs) not in ALLOWED_CLASS_SETS:
            bad_cs.append((i, cs))
            if len(bad_cs) > 5:
                break
    if bad_cs:
        raise ValueError(
            f"[{cond}] class_set_prediction contains values outside whitelist: "
            f"first offenders={bad_cs}"
        )

    # uncertainty_flags vocab + dedupe.
    bad_flags = []
    bad_dupes = []
    for i, s in enumerate(out["uncertainty_flags"]):
        flags = json.loads(s)
        if len(flags) != len(set(flags)):
            bad_dupes.append((i, flags))
            if len(bad_dupes) > 5:
                break
        for f in flags:
            if f not in ALLOWED_FLAGS:
                bad_flags.append((i, f))
                if len(bad_flags) > 5:
                    break
    if bad_flags:
        raise ValueError(
            f"[{cond}] uncertainty_flags has values outside Step 3 §7 vocab: "
            f"first offenders={bad_flags}"
        )
    if bad_dupes:
        raise ValueError(
            f"[{cond}] uncertainty_flags has duplicate entries: first={bad_dupes}"
        )

    # caption_confidence_level enum.
    bad_levels = sorted(set(out["caption_confidence_level"].unique()) - ALLOWED_LEVELS)
    if bad_levels:
        raise ValueError(
            f"[{cond}] caption_confidence_level has values outside enum: {bad_levels}"
        )

    # no_call coercion (CSV-roundtrip-safe: accept python bool only here since
    # we just generated it).
    if out["no_call"].dtype != bool:
        # Coerce-check: must be 0/1 or True/False
        coerced = out["no_call"].astype(bool)
    else:
        coerced = out["no_call"]

    # no_call <-> empty class_set bidirectional consistency.
    empty_cs = out["class_set_prediction"].apply(lambda s: json.loads(s) == [])
    bad_a = (coerced != empty_cs).sum()
    if int(bad_a) > 0:
        raise ValueError(
            f"[{cond}] no_call <-> class_set==[] consistency violated in {int(bad_a)} rows."
        )

    # no_call rows must have caption_confidence_level == 'no_call'.
    bad_lvl = ((coerced) & (out["caption_confidence_level"] != "no_call")).sum()
    if int(bad_lvl) > 0:
        raise ValueError(
            f"[{cond}] {int(bad_lvl)} no_call rows have level != 'no_call'."
        )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _summary(out: pd.DataFrame, cond: str) -> None:
    print()
    print(f"[{cond}] class_set_prediction distribution:")
    cs_counts = out["class_set_prediction"].value_counts().sort_index()
    for k, v in cs_counts.items():
        print(f"  {k:<28} {int(v):>5}")

    print(f"[{cond}] caption_confidence_level distribution:")
    for k, v in out["caption_confidence_level"].value_counts().items():
        print(f"  {k:<10} {int(v):>5}")

    print(f"[{cond}] uncertainty_flags occurrence (rows with each flag):")
    flag_counts: dict = {}
    for s in out["uncertainty_flags"]:
        for f in json.loads(s):
            flag_counts[f] = flag_counts.get(f, 0) + 1
    for f in sorted(flag_counts.keys()):
        print(f"  {f:<35} {flag_counts[f]:>5}")

    n_no_call = int(out["no_call"].astype(bool).sum())
    print(f"[{cond}] no_call: {n_no_call} / {len(out)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 64)
    print("Step 4 - Generate schema-compliant outputs (raw + zscore)")
    print("=" * 64)

    for cond, path in INPUT_PREDICTIONS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Input predictions for '{cond}' not found: {path}. "
                "Run scripts/train_step4_baseline.py first."
            )
        _validate_thresholds(THRESHOLDS[cond], cond)
        print(
            f"[{cond}] thresholds: confident_C2={THRESHOLDS[cond]['confident_C2_threshold']:.4f}  "
            f"non_trivial_C2={THRESHOLDS[cond]['non_trivial_C2_threshold']:.4f}  "
            f"within_group={THRESHOLDS[cond]['within_group_threshold']:.4f}  "
            f"anchor_suppression={THRESHOLDS[cond]['anchor_suppression_threshold']:.4f}  "
            f"anchor_no_call={THRESHOLDS[cond]['anchor_no_call_threshold']:.4f}"
        )

    for cond, in_path in INPUT_PREDICTIONS.items():
        print()
        print("-" * 64)
        print(f"condition: {cond}")
        print(f"input  : {in_path}")
        out_path = OUTPUT_FILES[cond]
        print(f"output : {out_path}")
        print("-" * 64)

        df = pd.read_csv(in_path)
        if len(df) != EXPECTED_ROW_COUNT:
            raise ValueError(
                f"[{cond}] input row count {len(df)} != {EXPECTED_ROW_COUNT}."
            )
        missing = [c for c in CARRY_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"[{cond}] input missing columns: {missing}")

        thr = THRESHOLDS[cond]
        records = df.to_dict("records")
        new_cols = [_process_row(r, thr) for r in records]
        new_df = pd.DataFrame(new_cols)
        out = pd.concat(
            [df[CARRY_COLS].reset_index(drop=True),
             new_df.reset_index(drop=True)],
            axis=1,
        )[OUTPUT_COL_ORDER]

        _validate_output(out, cond)
        out.to_csv(out_path, index=False)
        print(f"[{cond}] saved: {out_path}")
        _summary(out, cond)

    print()
    print("=" * 64)
    print("Done.")
    print("=" * 64)
    print(f"saved schema outputs raw    : {OUTPUT_FILES['raw']}")
    print(f"saved schema outputs zscore : {OUTPUT_FILES['zscore']}")
    print(
        "note: thresholds are CANDIDATES (val-derived). No commit; no caption "
        "written. Authoritative schema validation in "
        "scripts/validate_step4_schema_outputs.py."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
