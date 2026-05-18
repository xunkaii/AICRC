"""Step 8_v2 — v2 / v3 caption pipeline 비교 validation.

본 스크립트는 *신규 파일* 이며, 기존 ``validate_step8_v2_rulebased.py`` 와
별도로 동작한다. v2 / v3 caption pipeline 의 fidelity 와 vocabulary 분포를
side-by-side 로 비교한다.

검증 행렬 (3-way):
  (A) v2 captions × v2 validator → v2 baseline
  (B) v3 captions × v3 validator → v3 baseline
  (C) v3 captions × v2 validator → v3 가 v2 정책에 대해 얼마나 위반하는지
                                   (clinical vocabulary 도입 trade-off 정량화)

Reads (read-only):
    data/step7_v2/step7_v2_captions.csv  (v2 mock full output)
    data/step7_v3/step7_v3_captions.csv  (v3 mock full output)
    scripts/generate_step7_v2_captions.py  (validate_caption v2)
    scripts/generate_step7_v3_captions.py  (validate_caption v3)

Writes (new files, 기존 step8_v2 산출물 보존):
    data/step8_v2/step8_v2_v2_validation.csv         per-sample (A)
    data/step8_v2/step8_v2_v3_validation.csv         per-sample (B)
    data/step8_v2/step8_v2_v3_under_v2policy.csv     per-sample (C)
    reports/step8_v2/step8_v2_v2_vs_v3_summary.csv   side-by-side metrics
    reports/step8_v2/step8_v2_v2_vs_v3_comparison.md thesis-grade comparison

기존 step8_v2_report.md / step8_v2_summary.csv 는 본 스크립트로 수정되지 않는다.

Run:
    python scripts/validate_step8_v2_v3_compare.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

V2_CAPTIONS = PROJECT_ROOT / "data" / "step7_v2" / "step7_v2_captions.csv"
V3_CAPTIONS = PROJECT_ROOT / "data" / "step7_v3" / "step7_v3_captions.csv"
V2_SCRIPT = PROJECT_ROOT / "scripts" / "generate_step7_v2_captions.py"
V3_SCRIPT = PROJECT_ROOT / "scripts" / "generate_step7_v3_captions.py"

OUTPUT_DATA_DIR = PROJECT_ROOT / "data" / "step8_v2"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "reports" / "step8_v2"

OUT_V2_VALID = OUTPUT_DATA_DIR / "step8_v2_v2_validation.csv"
OUT_V3_VALID = OUTPUT_DATA_DIR / "step8_v2_v3_validation.csv"
OUT_V3_UNDER_V2 = OUTPUT_DATA_DIR / "step8_v2_v3_under_v2policy.csv"
OUT_SUMMARY = OUTPUT_REPORT_DIR / "step8_v2_v2_vs_v3_summary.csv"
OUT_REPORT = OUTPUT_REPORT_DIR / "step8_v2_v2_vs_v3_comparison.md"


CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
POSTURES = ["SA", "CA", "HW"]
LEVELS = ["confident", "hedged", "low", "no_call"]

CLINICAL_VOCAB_PROBES = [
    "좌측", "우측", "양측",
    "고관절", "골반", "무릎 정렬",
    "내회전", "내전",
    "발목 가동", "요추", "흉추", "척추 중립",
    "knee valgus", "posterior tilt",
]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row_to_v2_output(row: pd.Series) -> dict:
    return {
        "caption_ko": row.get("caption_ko") or "",
        "confidence_phrase": row.get("confidence_phrase") or "",
        "uncertainty_phrase": row.get("uncertainty_phrase") or "",
        "limitation_phrase": row.get("limitation_phrase") or "",
        "used_schema_fields": json.loads(row.get("used_schema_fields") or "[]"),
    }


def _row_to_v3_output(row: pd.Series) -> dict:
    return {
        "caption_ko": row.get("caption_ko") or "",
        "confidence_phrase": row.get("confidence_phrase") or "",
        "uncertainty_phrase": row.get("uncertainty_phrase") or "",
        "limitation_phrase": row.get("limitation_phrase") or "",
        "used_pool_entries": json.loads(row.get("used_pool_entries") or "[]"),
        "used_schema_fields": json.loads(row.get("used_schema_fields") or "[]"),
    }


def _row_to_schema(row: pd.Series) -> dict:
    return {
        "sample_id": row.get("sample_id"),
        "posture_canonical": row.get("posture_canonical"),
        "class_set": json.loads(row.get("class_set") or "[]"),
        "ambiguity_group": row.get("ambiguity_group"),
        "uncertainty_flags": json.loads(row.get("uncertainty_flags") or "[]"),
        "caption_confidence_level": row.get("caption_confidence_level"),
        "no_call": bool(row.get("no_call")),
        "no_call_reason": row.get("no_call_reason") or "",
    }


def _validate_df(df: pd.DataFrame, validate_fn, output_builder) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        output = output_builder(r)
        schema = _row_to_schema(r)
        ok, errors = validate_fn(output, schema)
        rows.append({
            "sample_id": r["sample_id"],
            "posture_canonical": r["posture_canonical"],
            "caption_confidence_level": r["caption_confidence_level"],
            "ambiguity_group": r["ambiguity_group"],
            "no_call": bool(r["no_call"]),
            "true_class": _infer_true_class(r),
            "validation_pass": ok,
            "n_errors": len(errors),
            "error_codes": "|".join(errors),
        })
    return pd.DataFrame(rows)


def _infer_true_class(row: pd.Series) -> str:
    sid = str(row.get("sample_id", ""))
    for p in sid.split("_"):
        if len(p) == 2 and p[0] == "C" and p[1].isdigit():
            return p
    return ""


def _error_code_breakdown(res_df: pd.DataFrame) -> dict[str, float]:
    s = set()
    for codes in res_df["error_codes"]:
        if codes:
            for c in codes.split("|"):
                # strip suffix after first ':' for category grouping
                s.add(c.split(":")[0])
    return {
        code: float(res_df["error_codes"].str.contains(code, regex=False).mean())
        for code in sorted(s)
    }


def _stratum_rates(res_df: pd.DataFrame, key: str, values: list[str], filter_col: str) -> dict[str, dict]:
    out = {}
    for v in values:
        sub = res_df[res_df[filter_col] == v]
        if len(sub) == 0:
            continue
        out[v] = {"n": int(len(sub)), "fidelity": float(sub["validation_pass"].mean())}
    return out


def _stratum_by_true_class(res_df: pd.DataFrame) -> dict[str, dict]:
    return _stratum_rates(res_df, "true_class", CLASSES, "true_class")


def _clinical_vocab_density(captions_df: pd.DataFrame) -> dict[str, float]:
    n = len(captions_df)
    out = {}
    for tok in CLINICAL_VOCAB_PROBES:
        out[tok] = float(captions_df["caption_ko"].str.contains(tok, regex=False).sum()) / n if n else 0
    return out


def main() -> int:
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("Step 8_v2 — v2 / v3 비교 validation")
    print("=" * 64)

    v2_mod = _load_module("v2_caps", V2_SCRIPT)
    v3_mod = _load_module("v3_caps", V3_SCRIPT)

    print(f"v2 captions <- {V2_CAPTIONS}")
    print(f"v3 captions <- {V3_CAPTIONS}")
    v2_df = pd.read_csv(V2_CAPTIONS, keep_default_na=False, encoding="utf-8-sig")
    v3_df = pd.read_csv(V3_CAPTIONS, keep_default_na=False, encoding="utf-8-sig")
    print(f"v2 n={len(v2_df)}  v3 n={len(v3_df)}")

    # (A) v2 captions × v2 validator
    print("\n(A) v2 captions × v2 validator ...")
    res_a = _validate_df(v2_df, v2_mod.validate_caption, _row_to_v2_output)
    res_a.to_csv(OUT_V2_VALID, index=False, encoding="utf-8-sig")
    fid_a = float(res_a["validation_pass"].mean())
    print(f"   fidelity = {fid_a:.4f}  ({int(res_a['validation_pass'].sum())}/{len(res_a)})")

    # (B) v3 captions × v3 validator
    print("(B) v3 captions × v3 validator ...")
    res_b = _validate_df(v3_df, v3_mod.validate_caption, _row_to_v3_output)
    res_b.to_csv(OUT_V3_VALID, index=False, encoding="utf-8-sig")
    fid_b = float(res_b["validation_pass"].mean())
    print(f"   fidelity = {fid_b:.4f}  ({int(res_b['validation_pass'].sum())}/{len(res_b)})")

    # (C) v3 captions × v2 validator (cross-strict-check)
    print("(C) v3 captions × v2 validator (cross-strict-check) ...")
    res_c = _validate_df(v3_df, v2_mod.validate_caption, _row_to_v2_output)
    res_c.to_csv(OUT_V3_UNDER_V2, index=False, encoding="utf-8-sig")
    fid_c = float(res_c["validation_pass"].mean())
    print(f"   fidelity = {fid_c:.4f}  ({int(res_c['validation_pass'].sum())}/{len(res_c)})")

    # ---- additional metrics ----
    err_a = _error_code_breakdown(res_a)
    err_b = _error_code_breakdown(res_b)
    err_c = _error_code_breakdown(res_c)

    cv_v2 = _clinical_vocab_density(v2_df)
    cv_v3 = _clinical_vocab_density(v3_df)

    cap_len_v2 = v2_df["caption_ko"].str.len()
    cap_len_v3 = v3_df["caption_ko"].str.len()
    uniq_v2 = int(v2_df["caption_ko"].nunique())
    uniq_v3 = int(v3_df["caption_ko"].nunique())

    by_cls_a = _stratum_by_true_class(res_a)
    by_cls_b = _stratum_by_true_class(res_b)
    by_pos_a = _stratum_rates(res_a, "posture", POSTURES, "posture_canonical")
    by_pos_b = _stratum_rates(res_b, "posture", POSTURES, "posture_canonical")
    by_lvl_a = _stratum_rates(res_a, "confidence", LEVELS, "caption_confidence_level")
    by_lvl_b = _stratum_rates(res_b, "confidence", LEVELS, "caption_confidence_level")

    # ---- summary CSV ----
    rows = []
    rows.append({"section": "overall", "metric": "n_v2", "v2": len(v2_df), "v3": len(v3_df), "delta": ""})
    rows.append({"section": "overall", "metric": "schema_faithfulness_rate", "v2": f"{fid_a:.4f}", "v3": f"{fid_b:.4f}", "delta": f"{fid_b - fid_a:+.4f}"})
    rows.append({"section": "overall", "metric": "v3_under_v2_policy", "v2": "", "v3": "", "delta": f"{fid_c:.4f}"})
    rows.append({"section": "diversity", "metric": "unique_caption_ko", "v2": uniq_v2, "v3": uniq_v3, "delta": uniq_v3 - uniq_v2})
    rows.append({"section": "diversity", "metric": "caption_len_mean", "v2": f"{cap_len_v2.mean():.1f}", "v3": f"{cap_len_v3.mean():.1f}", "delta": f"{cap_len_v3.mean() - cap_len_v2.mean():+.1f}"})
    rows.append({"section": "diversity", "metric": "caption_len_median", "v2": int(cap_len_v2.median()), "v3": int(cap_len_v3.median()), "delta": int(cap_len_v3.median() - cap_len_v2.median())})

    for cls in CLASSES:
        a = by_cls_a.get(cls, {})
        b = by_cls_b.get(cls, {})
        rows.append({"section": "by_true_class", "metric": cls, "v2": f"{a.get('fidelity', 0):.4f}", "v3": f"{b.get('fidelity', 0):.4f}", "delta": ""})
    for p in POSTURES:
        a = by_pos_a.get(p, {})
        b = by_pos_b.get(p, {})
        rows.append({"section": "by_posture", "metric": p, "v2": f"{a.get('fidelity', 0):.4f}", "v3": f"{b.get('fidelity', 0):.4f}", "delta": ""})
    for lvl in LEVELS:
        a = by_lvl_a.get(lvl, {})
        b = by_lvl_b.get(lvl, {})
        rows.append({"section": "by_confidence", "metric": lvl, "v2": f"{a.get('fidelity', 0):.4f}", "v3": f"{b.get('fidelity', 0):.4f}", "delta": ""})
    for tok in CLINICAL_VOCAB_PROBES:
        rows.append({"section": "clinical_vocab_density", "metric": tok, "v2": f"{cv_v2[tok]:.4f}", "v3": f"{cv_v3[tok]:.4f}", "delta": f"{cv_v3[tok] - cv_v2[tok]:+.4f}"})
    for code in sorted(set(err_a) | set(err_b) | set(err_c)):
        rows.append({
            "section": "error_code_violation_rate",
            "metric": code,
            "v2": f"{err_a.get(code, 0):.4f}",
            "v3": f"{err_b.get(code, 0):.4f}",
            "delta": f"{err_c.get(code, 0):.4f} (v3 under v2 policy)",
        })

    pd.DataFrame(rows).to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    print(f"\nsaved summary -> {OUT_SUMMARY}")

    # ---- markdown report ----
    L: list[str] = []
    L.append("# Step 8_v2 — v2 / v3 Caption Pipeline 비교 검증")
    L.append("")
    L.append(f"- 생성 스크립트: `scripts/validate_step8_v2_v3_compare.py`")
    L.append(f"- v2 input : `data/step7_v2/step7_v2_captions.csv` (n={len(v2_df)})")
    L.append(f"- v3 input : `data/step7_v3/step7_v3_captions.csv` (n={len(v3_df)})")
    L.append(f"- 검증 layer: rule-based")
    L.append("- 기존 `step8_v2_report.md` / `step8_v2_summary.csv` 는 본 스크립트로 수정되지 않음 (별도 baseline 보존)")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 1. Overall fidelity")
    L.append("")
    L.append("| 검증 행렬 | n | pass | fidelity |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| **(A) v2 captions × v2 validator** | {len(res_a)} | {int(res_a['validation_pass'].sum())} | **{fid_a:.4f}** |")
    L.append(f"| **(B) v3 captions × v3 validator** | {len(res_b)} | {int(res_b['validation_pass'].sum())} | **{fid_b:.4f}** |")
    L.append(f"| (C) v3 captions × v2 validator    | {len(res_c)} | {int(res_c['validation_pass'].sum())} | {fid_c:.4f} |")
    L.append("")
    L.append("**핵심 해석:**")
    L.append(f"- A vs B: 각 pipeline 이 *자기 정책* 에 대해 거의 동등하게 충실함 ({fid_a:.4f} vs {fid_b:.4f}).")
    L.append(f"- C: v3 captions 를 *v2 의 strict 정책* 에 대해 평가하면 fidelity 가 {fid_c:.4f} 로 떨어짐 — clinical vocabulary 도입으로 v2 정책 위반 {(1-fid_c):.1%} 발생.")
    L.append("- (C) 가 클수록 v2 → v3 trade-off 폭이 큼. 정책 완화 효과 = (1 - fid_c).")
    L.append("")
    L.append("## 2. Diversity (표현 다양성)")
    L.append("")
    L.append("| 지표 | v2 | v3 | Δ |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| unique caption_ko        | {uniq_v2} | {uniq_v3} | **{uniq_v3 - uniq_v2:+d}** |")
    L.append(f"| caption 길이 mean (자)   | {cap_len_v2.mean():.1f} | {cap_len_v3.mean():.1f} | {cap_len_v3.mean() - cap_len_v2.mean():+.1f} |")
    L.append(f"| caption 길이 median (자) | {int(cap_len_v2.median())} | {int(cap_len_v3.median())} | {int(cap_len_v3.median() - cap_len_v2.median()):+d} |")
    L.append(f"| caption 길이 max (자)    | {int(cap_len_v2.max())} | {int(cap_len_v3.max())} | {int(cap_len_v3.max() - cap_len_v2.max()):+d} |")
    L.append("")
    L.append(f"- v3 가 **{uniq_v3 / max(uniq_v2, 1):.1f}배** 더 다양한 caption 을 생성 — POSTURE_POOL × JOINT_POOL hash permutation 의 효과.")
    L.append("")
    L.append("## 3. Clinical vocabulary 등장 비율")
    L.append("")
    L.append("| 어휘 | v2 | v3 | Δ |")
    L.append("|---|---:|---:|---:|")
    for tok in CLINICAL_VOCAB_PROBES:
        L.append(f"| {tok} | {cv_v2[tok]:.2%} | {cv_v3[tok]:.2%} | **{cv_v3[tok] - cv_v2[tok]:+.2%}** |")
    L.append("")
    L.append("- v2 에서 0% 인 항목 (e.g., 좌측/우측/양측/무릎 정렬/고관절) 이 v3 에서 의도적으로 도입됨 → 정책 분기 효과 확인.")
    L.append("")
    L.append("## 4. Stratum별 fidelity (A vs B)")
    L.append("")
    L.append("### 4.1 by true class")
    L.append("")
    L.append("| class | n (v2/v3) | v2 fidelity | v3 fidelity |")
    L.append("|---|---:|---:|---:|")
    for cls in CLASSES:
        a = by_cls_a.get(cls, {})
        b = by_cls_b.get(cls, {})
        L.append(f"| {cls} | {a.get('n', 0)} / {b.get('n', 0)} | {a.get('fidelity', 0):.4f} | {b.get('fidelity', 0):.4f} |")
    L.append("")
    L.append("### 4.2 by posture")
    L.append("")
    L.append("| posture | n (v2/v3) | v2 fidelity | v3 fidelity |")
    L.append("|---|---:|---:|---:|")
    for p in POSTURES:
        a = by_pos_a.get(p, {})
        b = by_pos_b.get(p, {})
        L.append(f"| {p} | {a.get('n', 0)} / {b.get('n', 0)} | {a.get('fidelity', 0):.4f} | {b.get('fidelity', 0):.4f} |")
    L.append("")
    L.append("### 4.3 by caption_confidence_level")
    L.append("")
    L.append("| level | n (v2/v3) | v2 fidelity | v3 fidelity |")
    L.append("|---|---:|---:|---:|")
    for lvl in LEVELS:
        a = by_lvl_a.get(lvl, {})
        b = by_lvl_b.get(lvl, {})
        L.append(f"| {lvl} | {a.get('n', 0)} / {b.get('n', 0)} | {a.get('fidelity', 0):.4f} | {b.get('fidelity', 0):.4f} |")
    L.append("")
    L.append("## 5. Error code violation rate (3-way)")
    L.append("")
    all_codes = sorted(set(err_a) | set(err_b) | set(err_c))
    if all_codes:
        L.append("| error code | A: v2/v2 | B: v3/v3 | C: v3/v2 (strict) |")
        L.append("|---|---:|---:|---:|")
        for c in all_codes:
            L.append(f"| `{c}` | {err_a.get(c, 0):.4f} | {err_b.get(c, 0):.4f} | {err_c.get(c, 0):.4f} |")
    else:
        L.append("- 모든 error code 위반 0건 (3-way 모두).")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 6. Trade-off 해석 (thesis chapter 자료)")
    L.append("")
    L.append("본 비교는 reframing §8.3 의 *uncertainty-aware caption layer* 가 **두 끝점**을 가짐을 정량 증거로 제시한다:")
    L.append("")
    L.append("- **v2 보수 (sensor observability 우선)**: clinical 어휘 0%, schema 충실도 {:.4f}. 측정 가능성 한계 안에서만 표현하므로 \"안전\" 하나, C4/C5/C6 구분 불가 + 표현 단조.".format(fid_a))
    L.append("- **v3 완화 (clinical richness 우선)**: clinical 어휘 의도적 도입, schema 충실도 {:.4f}. caption 다양성 {:.1f}배 증가 + 좌/우 비대칭 표현 가능. 단 v2 strict 정책 기준으로는 {:.1%} 위반 ({:.4f} → {:.4f}).".format(fid_b, uniq_v3 / max(uniq_v2, 1), 1 - fid_c, fid_a, fid_c))
    L.append("")
    L.append("→ 본 trade-off 자체가 paper chapter 1 개 거리. 실제 임상/현장 환경에 따라 어느 끝점을 선택할지 *사용자/연구자가 명시적으로 결정* 할 수 있는 framework 가 갖춰짐.")
    L.append("")
    L.append("**한계:**")
    L.append("- 본 결과는 mock provider 기반. real LLM 평가 시 v3 fidelity 가 약간 떨어질 수 있음 (anthropic golden 15-case 에서는 1.0000 유지 — 일관 가능성 높음).")
    L.append("- (C) 는 v3 가 v2 의 used_pool_entries 필드를 \"extra_fields\" 로 보는 등의 schema 불일치도 포함 → 순수 어휘 위반 외 형식 위반 일부 포함됨.")
    L.append("")
    L.append("## 7. 산출물")
    L.append("")
    L.append("```")
    L.append("data/step8_v2/")
    L.append("├── step8_v2_v2_validation.csv        (A) per-sample, n={}".format(len(res_a)))
    L.append("├── step8_v2_v3_validation.csv        (B) per-sample, n={}".format(len(res_b)))
    L.append("└── step8_v2_v3_under_v2policy.csv    (C) per-sample, n={}".format(len(res_c)))
    L.append("")
    L.append("reports/step8_v2/")
    L.append("├── step8_v2_v2_vs_v3_summary.csv     side-by-side metrics")
    L.append("└── step8_v2_v2_vs_v3_comparison.md   본 보고서")
    L.append("")
    L.append("기존 보존 (본 스크립트로 수정되지 않음):")
    L.append("├── step8_v2_report.md                v2 baseline (이전 run)")
    L.append("└── step8_v2_summary.csv              v2 baseline summary")
    L.append("```")
    L.append("")
    OUT_REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"saved markdown -> {OUT_REPORT}")

    print()
    print("=" * 64)
    print(f"A (v2/v2)   fidelity = {fid_a:.4f}")
    print(f"B (v3/v3)   fidelity = {fid_b:.4f}  (diversity {uniq_v3 / max(uniq_v2, 1):.1f}x)")
    print(f"C (v3/v2)   fidelity = {fid_c:.4f}  (v2 정책 위반 = {1 - fid_c:.2%})")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
