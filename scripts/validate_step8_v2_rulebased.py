"""Step 8_v2 — Rule-based schema-caption fidelity validation (Option A).

Reframing §7.1~§7.3 의 자동 검증. 기존 scripts/validate_step7_v2_captions.py
의 sanity-check 로직(generate_step7_v2_captions.validate_caption 재사용)을
*thesis-grade* 검증으로 확장:

  - per-class / per-posture / per-confidence-level stratum별 fidelity rate
  - ambiguity preservation rate (uncertainty_flags 있을 때 caption이
    hedge expression을 보존했는가)
  - per-sample validation 결과 보존 (data/step8_v2/...) — 후속 분석/error
    inspection 가능

Reads (read-only):
    data/step7_v2/step7_v2_captions.csv  (default; --input 로 변경 가능)

Writes (new files, step8_v2 분리 경로):
    data/step8_v2/step8_v2_validation_full.csv   per-sample pass/errors
    reports/step8_v2/step8_v2_summary.csv         overall + stratum metrics
    reports/step8_v2/step8_v2_report.md           thesis-grade markdown

Run:
    python.exe -X utf8 scripts/validate_step8_v2_rulebased.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from generate_step7_v2_captions import validate_caption  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "data" / "step7_v2" / "step7_v2_captions.csv"
OUTPUT_DATA_DIR = PROJECT_ROOT / "data" / "step8_v2"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "reports" / "step8_v2"
OUTPUT_FULL_CSV = OUTPUT_DATA_DIR / "step8_v2_validation_full.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_REPORT_DIR / "step8_v2_summary.csv"
OUTPUT_REPORT_MD = OUTPUT_REPORT_DIR / "step8_v2_report.md"


CLASS_IDS = ["C1", "C2", "C3", "C4", "C5", "C6"]
POSTURES = ["SA", "CA", "HW"]
CONFIDENCE_LEVELS = ["confident", "hedged", "low", "no_call"]


def _row_to_output_schema(row: pd.Series) -> tuple[dict, dict]:
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


def _has_ambiguity_marker(uncertainty_flags: list[str]) -> bool:
    """schema에 ambiguity 시그널이 있는가.
    'within_group_ambiguity', 'pair_ambiguity', 'class_set_size>1', 'anchor_unreliable'
    등을 ambiguity preservation 검증 대상으로 본다."""
    for f in uncertainty_flags:
        fl = f.lower()
        if "ambiguity" in fl or "anchor_unreliable" in fl:
            return True
    return False


def _caption_expresses_hedge(uncertainty_phrase: str, confidence_phrase: str, caption_ko: str) -> bool:
    """caption이 hedge/uncertainty 표현을 포함하는가 (rule-based heuristic).
    Step 5_v2 closed vocab의 hedge markers를 substring으로 확인.
    """
    hedge_markers = [
        "단정", "어렵", "모호", "한정", "두 가지", "함께 나타", "경계",
        "일관된 단일 유형", "신뢰도가 낮", "불확실", "구분하기 어렵",
        "자료가 없", "안정적인 설명을 제공",
    ]
    blob = " ".join([uncertainty_phrase or "", confidence_phrase or "", caption_ko or ""])
    return any(m in blob for m in hedge_markers)


def _validate_row(row: pd.Series) -> dict:
    output, schema = _row_to_output_schema(row)
    ok, errors = validate_caption(output, schema)

    flags = schema["uncertainty_flags"]
    expects_ambig = _has_ambiguity_marker(flags) or len(schema["class_set"]) > 1
    has_hedge = _caption_expresses_hedge(
        output["uncertainty_phrase"], output["confidence_phrase"], output["caption_ko"]
    )
    ambig_preserved = (not expects_ambig) or has_hedge

    return {
        "sample_id": row["sample_id"],
        "posture_canonical": row["posture_canonical"],
        "caption_confidence_level": row["caption_confidence_level"],
        "ambiguity_group": row["ambiguity_group"],
        "true_class": row.get("true_class_id") or _infer_true_class(row),
        "validation_pass": ok,
        "n_errors": len(errors),
        "error_codes": "|".join(errors),
        "expects_ambig": expects_ambig,
        "caption_has_hedge": has_hedge,
        "ambig_preserved": ambig_preserved,
    }


def _infer_true_class(row: pd.Series) -> str:
    """If true_class_id column absent, parse from sample_id pattern EP##_C#_*."""
    sid = str(row.get("sample_id", ""))
    parts = sid.split("_")
    for p in parts:
        if len(p) == 2 and p[0] == "C" and p[1].isdigit():
            return p
    return ""


def _stratum_rate(df: pd.DataFrame, group_col: str, pass_col: str) -> pd.DataFrame:
    g = df.groupby(group_col)
    return pd.DataFrame({
        "stratum_key": group_col,
        "stratum_value": list(g.groups.keys()),
        "n": g.size().values,
        "pass_count": g[pass_col].sum().values,
        "rate": g[pass_col].mean().values,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="captions CSV (default: step7_v2 full)")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 1

    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("Step 8_v2 — rule-based schema-caption fidelity validation")
    print("=" * 64)
    print(f"input -> {in_path}")
    df = pd.read_csv(in_path, keep_default_na=False, encoding="utf-8-sig")
    print(f"loaded n={len(df)} rows")

    # ---- per-sample validation ----
    print("\nvalidating each caption ...")
    rows = [_validate_row(r) for _, r in df.iterrows()]
    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUTPUT_FULL_CSV, index=False, encoding="utf-8-sig")
    print(f"saved per-sample -> {OUTPUT_FULL_CSV}")

    # ---- overall metrics ----
    n = len(res_df)
    overall_fidelity = float(res_df["validation_pass"].mean())
    ambig_subset = res_df[res_df["expects_ambig"]]
    ambig_preservation_rate = (
        float(ambig_subset["ambig_preserved"].mean()) if len(ambig_subset) else float("nan")
    )

    # Per-error-code rate (from error_codes column)
    error_code_set = set()
    for codes in res_df["error_codes"]:
        if codes:
            error_code_set.update(codes.split("|"))

    per_code_rate = {}
    for code in sorted(error_code_set):
        per_code_rate[code] = float(res_df["error_codes"].str.contains(code, regex=False).mean())

    # ---- stratum breakdowns ----
    print("\nstratum breakdowns ...")
    stratum_rows = []

    # by true class
    res_df["_true_class"] = res_df["true_class"]
    for cls in CLASS_IDS:
        sub = res_df[res_df["_true_class"] == cls]
        if len(sub) == 0:
            continue
        stratum_rows.append({
            "stratum_key": "true_class", "stratum_value": cls,
            "n": len(sub),
            "fidelity_rate": float(sub["validation_pass"].mean()),
            "ambig_preservation_rate": (
                float(sub[sub["expects_ambig"]]["ambig_preserved"].mean())
                if sub["expects_ambig"].any() else float("nan")
            ),
        })

    for p in POSTURES:
        sub = res_df[res_df["posture_canonical"] == p]
        if len(sub) == 0:
            continue
        stratum_rows.append({
            "stratum_key": "posture", "stratum_value": p,
            "n": len(sub),
            "fidelity_rate": float(sub["validation_pass"].mean()),
            "ambig_preservation_rate": (
                float(sub[sub["expects_ambig"]]["ambig_preserved"].mean())
                if sub["expects_ambig"].any() else float("nan")
            ),
        })

    for lvl in CONFIDENCE_LEVELS:
        sub = res_df[res_df["caption_confidence_level"] == lvl]
        if len(sub) == 0:
            continue
        stratum_rows.append({
            "stratum_key": "confidence_level", "stratum_value": lvl,
            "n": len(sub),
            "fidelity_rate": float(sub["validation_pass"].mean()),
            "ambig_preservation_rate": (
                float(sub[sub["expects_ambig"]]["ambig_preserved"].mean())
                if sub["expects_ambig"].any() else float("nan")
            ),
        })

    stratum_df = pd.DataFrame(stratum_rows)

    # ---- summary CSV (overall + per-error + stratum) ----
    summary_rows: list[dict] = []
    summary_rows.append({"section": "overall", "key": "total_rows", "value": n})
    summary_rows.append({"section": "overall", "key": "schema_faithfulness_rate", "value": overall_fidelity})
    summary_rows.append({"section": "overall", "key": "ambiguity_preservation_rate", "value": ambig_preservation_rate})
    summary_rows.append({"section": "overall", "key": "ambiguity_subset_n", "value": len(ambig_subset)})
    for code, rate in per_code_rate.items():
        summary_rows.append({"section": "per_error_code", "key": code, "value": rate})
    for r in stratum_rows:
        summary_rows.append({
            "section": f"stratum_{r['stratum_key']}",
            "key": r["stratum_value"],
            "value": r["fidelity_rate"],
            "n": r["n"],
            "ambig_preservation_rate": r["ambig_preservation_rate"],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f"saved summary -> {OUTPUT_SUMMARY_CSV}")

    # ---- console echo ----
    print("\n=== overall ===")
    print(f"  total_rows                  {n}")
    print(f"  schema_faithfulness_rate    {overall_fidelity:.4f}")
    print(f"  ambiguity_subset_n          {len(ambig_subset)}")
    print(f"  ambiguity_preservation_rate {ambig_preservation_rate:.4f}")

    if per_code_rate:
        print("\n=== per-error-code rate (only nonzero) ===")
        for code, rate in sorted(per_code_rate.items()):
            if rate > 0:
                print(f"  {code:42s} {rate:.4f}")
        if all(r == 0 for r in per_code_rate.values()):
            print("  (모든 error code 위반 0건)")
    else:
        print("\n=== per-error-code: 0건 ===")

    print("\n=== fidelity by true_class ===")
    for r in stratum_rows:
        if r["stratum_key"] == "true_class":
            print(f"  {r['stratum_value']}  n={r['n']:4d}  fidelity={r['fidelity_rate']:.4f}  ambig_pres={r['ambig_preservation_rate']:.4f}")
    print("\n=== fidelity by posture ===")
    for r in stratum_rows:
        if r["stratum_key"] == "posture":
            print(f"  {r['stratum_value']}  n={r['n']:4d}  fidelity={r['fidelity_rate']:.4f}  ambig_pres={r['ambig_preservation_rate']:.4f}")
    print("\n=== fidelity by confidence_level ===")
    for r in stratum_rows:
        if r["stratum_key"] == "confidence_level":
            print(f"  {r['stratum_value']:9s}  n={r['n']:4d}  fidelity={r['fidelity_rate']:.4f}  ambig_pres={r['ambig_preservation_rate']:.4f}")

    # ---- markdown report ----
    L: list[str] = []
    L.append("# Step 8_v2 — Rule-based Schema-Caption Fidelity 검증")
    L.append("")
    L.append(f"- 생성 스크립트: `scripts/validate_step8_v2_rulebased.py`")
    L.append(f"- 입력: `{in_path.relative_to(PROJECT_ROOT)}` (n={n})")
    L.append(f"- provider: mock (b01.5/seed42 schema → deterministic template)")
    L.append("- 검증 layer: **rule-based만** (LLM judge는 옵션 B로 분리)")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 1. Overall fidelity")
    L.append("")
    L.append("| metric | value |")
    L.append("|---|---:|")
    L.append(f"| total rows                  | {n} |")
    L.append(f"| **schema_faithfulness_rate** | **{overall_fidelity:.4f}** |")
    L.append(f"| ambiguity subset n          | {len(ambig_subset)} |")
    L.append(f"| **ambiguity_preservation_rate** | **{ambig_preservation_rate:.4f}** |")
    L.append("")
    L.append("- `schema_faithfulness_rate` = caption이 reframing §7.1의 13개 검사를 모두 통과한 비율")
    L.append("- `ambiguity_preservation_rate` = uncertainty_flags 또는 |class_set|>1 인 sample 중, caption에 hedge 표현이 포함된 비율")
    L.append("")
    L.append("## 2. Per-error-code violation rate")
    L.append("")
    if any(r > 0 for r in per_code_rate.values()):
        L.append("| error code | rate |")
        L.append("|---|---:|")
        for code, rate in sorted(per_code_rate.items(), key=lambda x: -x[1]):
            L.append(f"| `{code}` | {rate:.4f} |")
    else:
        L.append("- **모든 error code 위반 0건** (mock provider는 closed-vocab 템플릿 기반이라 자연스러운 결과)")
    L.append("")
    L.append("## 3. Stratum별 fidelity")
    L.append("")
    L.append("### 3.1 by true_class")
    L.append("")
    L.append("| class | n | fidelity | ambig_preservation |")
    L.append("|---|---:|---:|---:|")
    for r in stratum_rows:
        if r["stratum_key"] == "true_class":
            ap = "{:.4f}".format(r["ambig_preservation_rate"]) if not pd.isna(r["ambig_preservation_rate"]) else "—"
            L.append(f"| {r['stratum_value']} | {r['n']} | {r['fidelity_rate']:.4f} | {ap} |")
    L.append("")
    L.append("### 3.2 by posture")
    L.append("")
    L.append("| posture | n | fidelity | ambig_preservation |")
    L.append("|---|---:|---:|---:|")
    for r in stratum_rows:
        if r["stratum_key"] == "posture":
            ap = "{:.4f}".format(r["ambig_preservation_rate"]) if not pd.isna(r["ambig_preservation_rate"]) else "—"
            L.append(f"| {r['stratum_value']} | {r['n']} | {r['fidelity_rate']:.4f} | {ap} |")
    L.append("")
    L.append("### 3.3 by caption_confidence_level")
    L.append("")
    L.append("| confidence | n | fidelity | ambig_preservation |")
    L.append("|---|---:|---:|---:|")
    for r in stratum_rows:
        if r["stratum_key"] == "confidence_level":
            ap = "{:.4f}".format(r["ambig_preservation_rate"]) if not pd.isna(r["ambig_preservation_rate"]) else "—"
            L.append(f"| {r['stratum_value']} | {r['n']} | {r['fidelity_rate']:.4f} | {ap} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 4. 해석")
    L.append("")
    if overall_fidelity >= 0.99:
        L.append(f"- **fidelity {overall_fidelity:.4f}**: rule-based pipeline은 closed-vocab 정책을 거의 완벽히 준수. mock provider는 결정적 템플릿 기반이라 이 결과는 \"파이프라인이 자기 정책을 지킬 수 있다\"의 *상한* 증거.")
    elif overall_fidelity >= 0.95:
        L.append(f"- **fidelity {overall_fidelity:.4f}**: 95% 이상으로 thesis-grade. mock 환경에서 정책 위반 미세 잔존.")
    else:
        L.append(f"- **fidelity {overall_fidelity:.4f}**: 95% 미만. 정책 위반이 의미 있는 비율로 잔존 — 개별 error code 확인 필요.")
    L.append("")
    if not pd.isna(ambig_preservation_rate):
        if ambig_preservation_rate >= 0.95:
            L.append(f"- **ambiguity preservation {ambig_preservation_rate:.4f}**: schema가 ambiguity를 표시한 경우, caption이 hedge 표현을 거의 항상 유지. 즉 \"불확실성을 정직하게 표현\"하는 reframing §8.3 contribution이 정량 검증됨.")
        else:
            L.append(f"- **ambiguity preservation {ambig_preservation_rate:.4f}**: 95% 미만. ambiguous schema인데 confident-sounding caption이 발생. caption template 점검 필요.")
    L.append("")
    L.append("### Thesis chapter 위치")
    L.append("")
    L.append("reframing §8.3 (Uncertainty-aware Korean caption generation) 의 **유일한 정량 근거**. Chapter 4 (Modeling) 의 마지막 evidence block — \"caption pipeline이 schema에 충실함을 자동 검증으로 보였다\"가 paper claim이 된다.")
    L.append("")
    L.append("- **mock provider 한계**: 본 결과는 *템플릿이 자기 규칙을 지킨다*의 정량 증거이지, *LLM이 어휘를 일탈하지 않는다*의 증거는 아님. real LLM provider (Anthropic API) 평가는 **옵션 B**로 분리 (필요 시 추가 작업).")
    L.append("- **rule-based만으로 thesis-grade 가능한 이유**: reframing §7.2가 rule-based를 *필수*, LLM judge를 *보조*로 명시. LLM judge는 metric 산출에만 사용 가능 (caption 수정 신호 아님). 즉 rule-based 통과 시점이 main claim 근거.")
    L.append("")
    L.append("## 5. 산출물")
    L.append("")
    L.append("```")
    L.append(f"data/step8_v2/")
    L.append(f"└── step8_v2_validation_full.csv     per-sample pass/errors ({n} rows)")
    L.append("")
    L.append(f"reports/step8_v2/")
    L.append(f"├── step8_v2_summary.csv             overall + per-code + stratum")
    L.append(f"└── step8_v2_report.md               본 보고서")
    L.append("```")
    L.append("")
    OUTPUT_REPORT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"\nsaved markdown -> {OUTPUT_REPORT_MD}")

    print("\n" + "=" * 64)
    print(f"Done. fidelity={overall_fidelity:.4f}  ambig_preservation={ambig_preservation_rate:.4f}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
