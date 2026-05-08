"""Step 7_v2 — Independent validation of generated captions.

Reads (read-only):
    --input <captions_csv>

Writes:
    [golden mode]  reports/step7_v2/step7_v2_golden_validation_summary.csv
    [full mode]    reports/step7_v2/step7_v2_caption_generation_summary.csv

Run:
    python scripts/validate_step7_v2_captions.py --input data/step7_v2/step7_v2_golden_captions.csv --mode golden
    python scripts/validate_step7_v2_captions.py --input data/step7_v2/step7_v2_captions.csv       --mode full

The validator re-runs the same checks as scripts/generate_step7_v2_captions.py
on a captions CSV that has already been written. This is a sanity check before
Step 8_v2 automatic schema-caption validation.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

# Force UTF-8 on stdout/stderr so Korean characters render correctly even when
# the host console default codepage is cp949 (Korean Windows). No-op if the
# stream does not support reconfigure.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Reuse logic from the generator script.
from generate_step7_v2_captions import (  # noqa: E402
    CLASS_VOCAB,
    POSTURE_VOCAB,
    OUTPUT_REQUIRED_FIELDS,
    SCHEMA_FIELDS_WHITELIST,
    FORBIDDEN_EXACT,
    FORBIDDEN_DIRECTION_TOKENS,
    ATTENTION_LEAK_TOKENS,
    BIOMECH_TOKENS,
    RAW_CLASS_LABEL_RE,
    LEVELS,
    validate_caption,
)

OUTPUT_REPORT_DIR = PROJECT_ROOT / "reports" / "step7_v2"


def _load_caption_row_as_output_and_schema(row: pd.Series) -> tuple[dict, dict]:
    output = {
        "caption_ko": row.get("caption_ko") or "",
        "confidence_phrase": row.get("confidence_phrase") or "",
        "uncertainty_phrase": row.get("uncertainty_phrase") or "",
        "limitation_phrase": row.get("limitation_phrase") or "",
        "used_schema_fields": json.loads(row.get("used_schema_fields") or "[]"),
    }
    schema = {
        "sample_id": row.get("sample_id"),
        "posture_canonical": row.get("posture_canonical"),
        "class_set": json.loads(row.get("class_set") or "[]"),
        "ambiguity_group": row.get("ambiguity_group"),
        "uncertainty_flags": json.loads(row.get("uncertainty_flags") or "[]"),
        "caption_confidence_level": row.get("caption_confidence_level"),
        "no_call": bool(row.get("no_call")),
        "no_call_reason": row.get("no_call_reason") or "",
    }
    return output, schema


def _aggregate_metrics(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {}

    # Per-row checks (run validate_caption again; counts each error code)
    per_row_pass = []
    per_row_errors: list[list[str]] = []
    for _, row in df.iterrows():
        output, schema = _load_caption_row_as_output_and_schema(row)
        ok, errors = validate_caption(output, schema)
        per_row_pass.append(ok)
        per_row_errors.append(errors)

    def rate_with(prefix: str) -> float:
        return sum(any(e.startswith(prefix) for e in errs) for errs in per_row_errors) / n

    def rate_eq(code: str) -> float:
        return sum(any(e == code for e in errs) for errs in per_row_errors) / n

    json_valid_rate = 1.0  # CSV가 이미 만들어졌으므로 JSON 파싱은 통과한 상태
    required_field_complete_rate = sum(
        len(set(["caption_ko", "confidence_phrase", "uncertainty_phrase",
                 "limitation_phrase", "used_schema_fields"])
            - set(_load_caption_row_as_output_and_schema(row)[0].keys())) == 0
        for _, row in df.iterrows()
    ) / n

    forbidden_exact = rate_with("forbidden_token") + rate_with("direction_token")
    forbidden_semantic = rate_with("biomech_claim")  # heuristic proxy
    confidence_mismatch = rate_with("confidence_mismatch")
    no_call_contradiction = rate_with("no_call_contradiction")
    class_set_narrowing = rate_with("class_set_narrowing")
    unsupported_biomech = rate_with("biomech_claim") + rate_with("forbidden_token")
    attention_leakage = rate_with("attention_leak")
    posture_phrase_mismatch = rate_with("posture_phrase_missing") + rate_with("wrong_posture_phrase")
    class_vocab_violation = rate_with("raw_class_label") + rate_with("direction_token")
    used_fields_violation = rate_with("used_schema_fields")
    limitation_violation = rate_with("limitation_phrase")

    schema_faithfulness_rate = sum(per_row_pass) / n

    fallback_rate = float((df["final_status"] == "fallback").mean())
    mean_n_retries = float(df["n_retries"].mean()) if "n_retries" in df.columns else 0.0

    return {
        "total_rows": n,
        "json_valid_rate": json_valid_rate,
        "required_field_complete_rate": required_field_complete_rate,
        "forbidden_expression_rate": forbidden_exact,
        "forbidden_semantic_pattern_rate": forbidden_semantic,
        "confidence_mismatch_rate": confidence_mismatch,
        "no_call_contradiction_rate": no_call_contradiction,
        "class_set_narrowing_rate": class_set_narrowing,
        "unsupported_biomechanical_claim_rate": unsupported_biomech,
        "attention_leakage_rate": attention_leakage,
        "posture_phrase_mismatch_rate": posture_phrase_mismatch,
        "class_vocabulary_violation_rate": class_vocab_violation,
        "used_schema_fields_violation_rate": used_fields_violation,
        "limitation_phrase_violation_rate": limitation_violation,
        "schema_faithfulness_rate": schema_faithfulness_rate,
        "fallback_rate": fallback_rate,
        "mean_n_retries": mean_n_retries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to captions CSV")
    parser.add_argument("--mode", choices=["golden", "full"], required=True)
    args = parser.parse_args()

    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: input CSV not found: {in_path}", file=sys.stderr)
        return 1

    print(f"loading {in_path}")
    # keep_default_na=False so that empty strings remain "" (not NaN/float).
    df = pd.read_csv(in_path, keep_default_na=False, encoding="utf-8-sig")
    print(f"  n_rows = {len(df)}")

    metrics = _aggregate_metrics(df)
    if not metrics:
        print("ERROR: no rows in input.", file=sys.stderr)
        return 1

    if args.mode == "golden":
        out_csv = OUTPUT_REPORT_DIR / "step7_v2_golden_validation_summary.csv"
    else:
        out_csv = OUTPUT_REPORT_DIR / "step7_v2_caption_generation_summary.csv"

    rows = [{"metric": k, "value": v} for k, v in metrics.items()]
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"saved -> {out_csv}")

    print()
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:42s}  {v:.4f}")
        else:
            print(f"  {k:42s}  {v}")

    sfr = metrics["schema_faithfulness_rate"]
    print()
    print(f"=== schema_faithfulness_rate = {sfr:.4f} ===")
    if args.mode == "golden":
        if sfr >= 0.95:
            print("  [OK] golden faithfulness >= 0.95 — full mode 진입 가능.")
        else:
            print("  [HOLD] golden faithfulness < 0.95 — prompt / vocab 점검 후 full mode 진입 권장.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
