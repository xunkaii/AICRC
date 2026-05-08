"""One-shot helper: extract real sample_ids matching each Step 6_v2 golden case.

Reads (read-only):
    data/step4r/4rb_attention/step4r_bigru_attention_schema_outputs_calibrated.csv

Writes:
    reports/step6_v2/_golden_case_samples_extracted.json   (intermediate)

Run with the same env that has pandas (aicrc_step4 or aicrc_env).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_CSV = ROOT / "data" / "step4r" / "4rb_attention" / "step4r_bigru_attention_schema_outputs_calibrated.csv"
OUT_JSON = ROOT / "reports" / "step6_v2" / "_golden_case_samples_extracted.json"


def main() -> int:
    df = pd.read_csv(SCHEMA_CSV, encoding="utf-8-sig")
    df["class_set_list"] = df["class_set"].apply(json.loads)
    df["class_set_size"] = df["class_set_list"].apply(len)
    df["uncertainty_flags_list"] = df["uncertainty_flags"].apply(json.loads)
    df["has_anchor_unreliable"] = df["uncertainty_flags_list"].apply(lambda x: "anchor_unreliable" in x)

    fields = [
        "sample_id", "participant_id", "split", "posture_canonical",
        "true_class_id", "top1_class", "class_set", "class_set_size",
        "ambiguity_group", "uncertainty_flags",
        "caption_confidence_level", "no_call", "no_call_reason",
    ]

    def pick(sub: pd.DataFrame, n: int = 2) -> list[dict]:
        sub = sub.sort_values("sample_id").head(n)
        return sub[fields].to_dict("records")

    queries = {
        "confident_C2_SA": df[(df["ambiguity_group"] == "confident_C2") & (df["posture_canonical"] == "SA")],
        "confident_C2_CA": df[(df["ambiguity_group"] == "confident_C2") & (df["posture_canonical"] == "CA")],
        "within_group_hedged": df[(df["ambiguity_group"] == "within_group_c1_c5_c6") & (df["caption_confidence_level"] == "hedged")],
        "within_group_low": df[(df["ambiguity_group"] == "within_group_c1_c5_c6") & (df["caption_confidence_level"] == "low")],
        "pair_c3_c4_hedged": df[(df["ambiguity_group"] == "pair_c3_c4") & (df["caption_confidence_level"] == "hedged")],
        "pair_plus_c2_low": df[(df["ambiguity_group"] == "pair_plus_c2_absorption") & (df["caption_confidence_level"] == "low")],
        "pair_plus_c2_anchor_unreliable": df[(df["ambiguity_group"] == "pair_plus_c2_absorption") & df["has_anchor_unreliable"]],
        "no_call_anchor_unreliable": df[(df["no_call"]) & df["has_anchor_unreliable"]],
        "no_call_low_conf": df[(df["no_call"]) & (df["no_call_reason"] == "low_confidence_no_class_set")],
        "class_set_size_2": df[df["class_set_size"] == 2],
        "class_set_size_3": df[df["class_set_size"] == 3],
        # Violations: pick a *different* sample (offset 1) from the same condition
        "viol_knee_valgus_input": df[(df["ambiguity_group"] == "within_group_c1_c5_c6") & (df["caption_confidence_level"] == "hedged")],
        "viol_posterior_tilting_input": df[df["ambiguity_group"] == "pair_c3_c4"],
        "viol_attention_leakage_input": df[(df["ambiguity_group"] == "confident_C2") & (df["posture_canonical"] == "SA")],
        "viol_class_set_narrowing_input": df[df["class_set_size"] == 3],
    }
    out = {}
    for name, sub in queries.items():
        if name.startswith("viol_"):
            recs = pick(sub, n=4)
            out[name] = recs[3] if len(recs) > 3 else (recs[-1] if recs else None)
        else:
            recs = pick(sub, n=1)
            out[name] = recs[0] if recs else None
        print(f"  {name}: matched n={len(sub)}, picked={out[name]['sample_id'] if out[name] else 'NONE'}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
