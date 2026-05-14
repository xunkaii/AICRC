"""Step 4R-C Day 1 — Build deterministic Korean text corpus from b01.5/seed42 schema.

Why this exists:
    4R-C contrastive IMU-text alignment needs (IMU embedding, text embedding)
    pairs. This script produces the *text side* using deterministic templates
    over Step 5_v2 vocabulary (no LLM, no randomness). Every rep gets one
    Korean phrase derived from its schema fields.

Reads (read-only):
    data/step4r/4rb_attention/experiments/b01_5_aug_jitter_scale_strong/seed42/
        schema_outputs_calibrated.csv

Writes (new file only; no existing artifact is modified):
    data/step4r/4rc_contrastive_optional/text_corpus.csv

Columns of text_corpus.csv:
    sample_id, participant_id, split, posture_canonical,
    class_set (json), ambiguity_group, uncertainty_flags (json),
    caption_confidence_level, no_call (bool), anchor_unreliable (bool),
    template_key (str), phrase (str)

Template construction (v2, class_set-aware deterministic):
    "{POSTURE}에서 측정된 손목 IMU 신호는 {CLASS_SET_DESC}입니다. {CONFIDENCE}.
     {anchor_suffix if anchor_unreliable}"

    where:
        POSTURE         = Step 5_v2 POSTURE_VOCAB[posture]
        CLASS_SET_DESC  = NEW class_set-aware description (see _class_set_phrase),
                          embeds class IDs as parenthetical markers (C1)(C2)...
                          so all 8 whitelist sets are distinguishable
                          WITHOUT violating Step 5_v2 §6.2 directional ban
        CONFIDENCE      = Step 5_v2 CONFIDENCE_PHRASE_POOL[level][0]  (first item)
        anchor_suffix   = " 동작 기준점이 불안정하여 설명 신뢰도가 낮습니다."

    LIMITATION_PHRASE is intentionally *omitted* because it is identical for
    every rep and adds no contrastive signal.

    Policy note: This 4R-C internal training text uses class ID markers
    (e.g., "(C5, C6)") for contrastive differentiation. The user-facing
    caption pipeline (Step 5_v2 ~ 7_v2) is UNCHANGED and continues to use
    the collapsed Step 5_v2 vocabulary that omits class IDs and directional
    body terms. The two layers are intentionally separated: train rich
    representations internally, emit policy-compliant captions externally.

Run:
    & python.exe -X utf8 scripts/build_step4rc_text_corpus.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_SCHEMA_CSV = (
    PROJECT_ROOT / "data" / "step4r" / "4rb_attention" / "experiments"
    / "b01_5_aug_jitter_scale_strong" / "seed42" / "schema_outputs_calibrated.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional"
OUTPUT_CSV = OUTPUT_DIR / "text_corpus.csv"


# Step 5_v2 vocabulary (copied verbatim from scripts/generate_step7_v2_captions.py;
# the original file is NOT imported or modified).
POSTURE_VOCAB = {
    "SA": "팔을 앞으로 둔 자세",
    "CA": "팔을 교차한 자세",
    "HW": "손을 허리에 둔 자세",
}

# Pick the *first* phrase from each pool for deterministic generation.
CONFIDENCE_PHRASE_DETERMINISTIC = {
    "confident": "비교적 일관되게 나타납니다",
    "hedged": "단일 유형으로 좁히기 어렵습니다",
    "low": "불확실성이 큽니다",
    "no_call": "안정적인 설명을 제공하기 어렵습니다",
}

ANCHOR_UNRELIABLE_SUFFIX = " 동작 기준점이 불안정하여 설명 신뢰도가 낮습니다."

# Class_set-aware description (v2): embeds class IDs as parenthetical markers
# to distinguish all 8 whitelisted class_sets without violating Step 5_v2
# directional ban (no "오른쪽/왼쪽/양쪽 무릎").
NO_CALL_DESCRIPTION = "동작 기준점 신뢰도가 낮음 또는 신호 강도가 부족함"


def _class_set_phrase(class_set_list: list[str]) -> str:
    """Map a class_set (sorted list of class IDs) to a Korean description.

    Returns the *body* (object of "신호는 ... 입니다") portion of the phrase.
    8 whitelist class_sets map to distinct strings; empty set is handled
    upstream as no_call.
    """
    cs = sorted(class_set_list)
    if not cs:
        return NO_CALL_DESCRIPTION
    if cs == ["C2"]:
        return "깊이 부족 계열(C2)에 가까운 패턴"
    if cs == ["C3", "C4"]:
        return "복합 오류 후보 두 가지(C3, C4)가 함께 나타나는 경계 신호"
    if cs == ["C2", "C3", "C4"]:
        return "깊이 부족 계열(C2)과 복합 오류 후보(C3, C4)가 함께 나타나는 신호"
    # within_group_c1_c5_c6 variants (4 cases)
    if "C1" in cs:
        knee_subset = [c for c in cs if c != "C1"]
        if knee_subset:
            knee_str = ", ".join(knee_subset)
            return (
                f"정상 패턴(C1)과 무릎 관련 오류 후보({knee_str})가 함께 "
                "나타나는 경계 신호"
            )
        # cs == ["C1"] alone is not in whitelist; defensive fallback only.
        return "정상 패턴(C1)에 가까운 신호"
    if cs == ["C5", "C6"]:
        return "무릎 관련 오류 후보(C5, C6)가 두 가지 형태로 나타나는 경계 신호"
    # Defensive: should not be reachable given whitelist.
    return "단일 유형으로 좁히기 어려운 신호"


def _build_phrase(row: dict) -> tuple[str, str]:
    """Return (template_key, phrase) for a single schema row.

    template_key groups identical templates (posture | class_set canonical |
    level | anchor) — used for diagnostics.
    """
    posture = row["posture_canonical"]
    level = row["caption_confidence_level"]
    flags = json.loads(row["uncertainty_flags"]) if isinstance(row["uncertainty_flags"], str) else []
    anchor_unreliable = "anchor_unreliable" in flags
    class_set_list = json.loads(row["class_set"]) if isinstance(row["class_set"], str) else []
    class_set_sorted = sorted(class_set_list)

    posture_phrase = POSTURE_VOCAB.get(posture, "기록된 자세")
    cs_phrase = _class_set_phrase(class_set_sorted)
    confidence_phrase = CONFIDENCE_PHRASE_DETERMINISTIC.get(level, "단일 유형으로 좁히기 어렵습니다")

    phrase = f"{posture_phrase}에서 측정된 손목 IMU 신호는 {cs_phrase}입니다. {confidence_phrase}."
    if anchor_unreliable:
        phrase += ANCHOR_UNRELIABLE_SUFFIX

    cs_key = ",".join(class_set_sorted) if class_set_sorted else "EMPTY"
    template_key = f"{posture}|cs={cs_key}|{level}|anchor={int(anchor_unreliable)}"
    return template_key, phrase


def main() -> int:
    print("=" * 64)
    print("Step 4R-C Day 1 — Build deterministic text corpus")
    print("=" * 64)
    if not INPUT_SCHEMA_CSV.exists():
        raise FileNotFoundError(f"input schema CSV missing: {INPUT_SCHEMA_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"input  -> {INPUT_SCHEMA_CSV}")
    print(f"output -> {OUTPUT_CSV}")
    print()

    df = pd.read_csv(INPUT_SCHEMA_CSV, encoding="utf-8-sig")
    print(f"loaded n={len(df)} rows")
    if len(df) != 9275:
        print(f"  warn: row count != 9275 (got {len(df)}); proceeding anyway")

    records = []
    for _, r in df.iterrows():
        tk, ph = _build_phrase(r.to_dict())
        flags_list = json.loads(r["uncertainty_flags"]) if isinstance(r["uncertainty_flags"], str) else []
        records.append({
            "sample_id": r["sample_id"],
            "participant_id": r["participant_id"],
            "split": r["split"],
            "posture_canonical": r["posture_canonical"],
            "class_set": r["class_set"],
            "ambiguity_group": r["ambiguity_group"],
            "uncertainty_flags": r["uncertainty_flags"],
            "caption_confidence_level": r["caption_confidence_level"],
            "no_call": bool(r["no_call"]),
            "anchor_unreliable": "anchor_unreliable" in flags_list,
            "template_key": tk,
            "phrase": ph,
        })

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"saved -> {OUTPUT_CSV}  ({len(out_df)} rows)")

    # -------- diagnostics --------
    print()
    print("=== diagnostics ===")
    print()
    print(f"total rows:               {len(out_df)}")
    print(f"unique template_key:      {out_df['template_key'].nunique()}")
    print(f"unique phrases:           {out_df['phrase'].nunique()}")
    lengths = out_df["phrase"].str.len()
    print(
        f"phrase length (chars):    mean={lengths.mean():.1f} "
        f"min={lengths.min()} max={lengths.max()}"
    )
    print()
    print("phrase frequency by template_key (top 10):")
    for tk, c in Counter(out_df["template_key"]).most_common(10):
        print(f"  {c:>5}  {tk}")
    print()
    print("split distribution:")
    print(out_df["split"].value_counts().to_string())
    print()
    print("sample phrases (one per ambiguity_group):")
    seen = set()
    for _, r in out_df.iterrows():
        ag = r["ambiguity_group"]
        if ag in seen:
            continue
        seen.add(ag)
        print(f"  [{ag}]")
        print(f"    sample_id  = {r['sample_id']}")
        print(f"    template   = {r['template_key']}")
        print(f"    phrase     = {r['phrase']}")
        if len(seen) >= 8:
            break
    print()
    print("=== abort sanity check ===")
    n_unique = out_df["phrase"].nunique()
    if n_unique < 5:
        print(f"  WARN: only {n_unique} unique phrases — too few for contrastive learning.")
        print("        Consider richer template (add class_set details) before Day 2.")
    elif n_unique < 20:
        print(f"  NOTE: {n_unique} unique phrases — low diversity but workable for v1.")
    else:
        print(f"  OK: {n_unique} unique phrases — sufficient for initial contrastive v1.")

    print()
    print("=" * 64)
    print("Done.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
