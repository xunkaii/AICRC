"""Step 4R-C clinical v3 — v1 + ambiguity_group suffix.

본 스크립트는 ``build_step4rc_text_corpus_clinical.py`` (v1) 의 v3 분기.
v1 의 5개 pool (POSTURE_VOCAB / SINGLE_CLASS_DESC / JOINT_FOCUS /
CONFIDENCE_POOL / ANCHOR_SUFFIX_POOL) 은 *그대로 유지* 하고, ambiguity_group
별 suffix 만 phrase 끝에 추가한다.

목적: ambiguity 경계 신호를 *명시적 자연어* 로 표현해서 SBERT 가 경계 신호
유형을 인식하도록 함. v2 가 POSTURE/JOINT pool 다양화로 IMU-orthogonal
variance 를 도입한 것과 달리, v3 는 IMU 신호가 *이미 알 수 있는* ambiguity
정보 (class_set / uncertainty_flags) 만 자연어로 강화 → 학습 가능성 유지.

Reads (read-only):
    data/step4r/4rb_attention/experiments/b01_5_aug_jitter_scale_strong/
        seed42/schema_outputs_calibrated.csv  (v1 과 동일)

Writes (new file only; v1 + v2 corpus 보존):
    data/step4r/4rc_contrastive_optional/text_corpus_clinical_v3.csv

기존 보존 (수정 없음):
    scripts/build_step4rc_text_corpus_clinical.py            (v1)
    data/step4r/4rc_contrastive_optional/text_corpus_clinical.csv (v1)
    scripts/build_step4rc_text_corpus_clinical_v2.py          (v2)
    data/step4r/4rc_contrastive_optional/text_corpus_clinical_v2.csv (v2)

Run:
    & C:\\Users\\user\\anaconda3\\envs\\dl_env\\python.exe -X utf8 \\
        scripts/build_step4rc_text_corpus_clinical_v3.py
"""
from __future__ import annotations

import hashlib
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

INPUT_SCHEMA_CSV = (
    PROJECT_ROOT / "data" / "step4r" / "4rb_attention" / "experiments"
    / "b01_5_aug_jitter_scale_strong" / "seed42" / "schema_outputs_calibrated.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional"
OUTPUT_CSV = OUTPUT_DIR / "text_corpus_clinical_v3.csv"


# ---------------------------------------------------------------------------
# Pool 1~5: v1 그대로 (불변)
# ---------------------------------------------------------------------------
POSTURE_VOCAB = {
    "SA": "양팔 전방 신장 자세 (shoulder abduction frontal — 상지 가장 긴 lever arm)에서 측정된",
    "CA": "양팔 흉부 교차 자세 (chest-crossed arms — 중간 lever arm)에서 측정된",
    "HW": "양손 허리 자세 (hands on waist — 상지 최단 lever arm)에서 측정된",
}

SINGLE_CLASS_DESC: dict[str, list[str]] = {
    "C1": [
        "정상 스쿼트 패턴 (normal squat, 무릎 정렬 및 깊이 양호)",
        "결함 없는 동작 (neutral knee tracking, lumbar neutral)",
        "표준 스쿼트 폼 (no observable form deviation)",
    ],
    "C2": [
        "스쿼트 깊이 부족 (insufficient depth, 대퇴 수평선 미달)",
        "얕은 스쿼트 (partial squat, 고관절 굴곡 제한)",
        "깊이 미달 동작 (squat below parallel not achieved)",
    ],
    "C3": [
        "깊이 부족·골반 후방경사·무릎 외반 복합 결함 "
        "(insufficient depth with posterior pelvic tilt and knee valgus)",
        "복합 보상 패턴 (얕은 깊이, 요추 후만, 양측 무릎 내측 붕괴)",
        "다중 관절 결함 (depth-tilt-valgus combined compensation)",
    ],
    "C4": [
        "좌측 무릎 외반 (left knee valgus, medial collapse)",
        "좌측 무릎 내측 붕괴 (left knee tracks medially during descent)",
        "좌측 슬관절 내전 패턴 (left knee adduction during squat)",
    ],
    "C5": [
        "우측 무릎 외반 (right knee valgus, medial collapse)",
        "우측 무릎 내측 붕괴 (right knee tracks medially during descent)",
        "우측 슬관절 내전 패턴 (right knee adduction during squat)",
    ],
    "C6": [
        "양측 무릎 외반 (bilateral knee valgus, both knees collapse medially)",
        "양 무릎 내측 붕괴 (both knees track medially)",
        "양측 슬관절 내전 패턴 (bilateral knee adduction during squat)",
    ],
}

JOINT_FOCUS: dict[str, list[str]] = {
    "C1": [
        "발목·무릎·고관절 협응이 안정적으로 유지됩니다",
        "ankle-knee-hip kinetic chain 정렬이 일관됩니다",
    ],
    "C2": [
        "고관절 굴곡과 발목 배측굴곡 제한이 주된 결함입니다",
        "limited hip flexion 및 ankle dorsiflexion 패턴이 관찰됩니다",
    ],
    "C3": [
        "고관절·요추·슬관절 다관절 보상이 동시에 나타납니다",
        "hip·lumbar·knee 다관절 compromise가 결합되어 나타납니다",
    ],
    "C4": [
        "좌측 고관절 내전·내회전이 우세합니다",
        "left hip adduction/internal rotation이 dominant joint pattern입니다",
    ],
    "C5": [
        "우측 고관절 내전·내회전이 우세합니다",
        "right hip adduction/internal rotation이 dominant joint pattern입니다",
    ],
    "C6": [
        "양측 고관절 내전·내회전이 동시에 우세합니다",
        "bilateral hip adduction/internal rotation이 동시에 나타납니다",
    ],
}

CONFIDENCE_POOL: dict[str, list[str]] = {
    "confident": [
        "분류 신뢰도가 높습니다",
        "단일 패턴이 일관되게 나타납니다",
        "고신뢰 추정이 가능합니다",
    ],
    "hedged": [
        "단일 유형 특정이 어렵습니다",
        "경계 패턴이 관찰됩니다",
        "복수 후보 간 모호성이 존재합니다",
    ],
    "low": [
        "추정 불확실성이 큽니다",
        "신호 특이성이 낮습니다",
        "분류 신뢰도가 제한적입니다",
    ],
    "no_call": [
        "안정적 추정이 어렵습니다",
        "신호 강도가 부족합니다",
    ],
}

ANCHOR_SUFFIX_POOL = [
    " 동작 기준점이 불안정합니다 (foot anchor unreliable).",
    " 기준점 불안정으로 인한 추가 신뢰도 저하.",
]

NO_CALL_DESC = "no_call (분류 단정 불가, 신호 강도/anchor 신뢰도 부족)"


# ---------------------------------------------------------------------------
# Pool 6 (NEW in v3): AMBIGUITY_SUFFIX — ambiguity_group 별 1개씩 명시 표현.
# 사용자 제공 wording 을 그대로 사용.
# ---------------------------------------------------------------------------
AMBIGUITY_SUFFIX: dict[str, str] = {
    "confident_C2": "단일 클래스 신호가 명확하게 구분됩니다",
    "within_group_c1_c5_c6": "정상 패턴과 무릎 비대칭 신호가 경계에 위치합니다",
    "pair_c3_c4": "복합 결함 패턴과 단측 무릎 신호가 혼재합니다",
    "pair_plus_c2_absorption": "깊이 부족 신호와 무릎 비대칭 신호가 함께 나타납니다",
    "no_call": "신호 패턴이 불명확하여 단일 클래스로 특정이 어렵습니다",
    "uncategorized": "복수 클래스 경계 신호로 분류가 어렵습니다",
}


def _hash_idx(sample_id: str, salt: str, pool_len: int) -> int:
    h = hashlib.md5(f"{sample_id}|{salt}".encode("utf-8")).hexdigest()[:8]
    return int(h, 16) % pool_len


def _class_set_desc(class_set_list: list[str], sample_id: str) -> str:
    cs = sorted(class_set_list)
    if not cs:
        return NO_CALL_DESC
    if len(cs) == 1:
        c = cs[0]
        idx = _hash_idx(sample_id, f"cls_{c}", len(SINGLE_CLASS_DESC[c]))
        return SINGLE_CLASS_DESC[c][idx]
    parts = []
    for c in cs:
        idx = _hash_idx(sample_id, f"cls_{c}", len(SINGLE_CLASS_DESC[c]))
        parts.append(SINGLE_CLASS_DESC[c][idx])
    cs_str = ", ".join(cs)
    joined = " 또는 ".join(parts)
    return f"{joined} (복합 의심 — class_set {{{cs_str}}})"


def _joint_focus_desc(class_set_list: list[str], sample_id: str) -> str:
    cs = sorted(class_set_list)
    if not cs:
        return "주된 관절 패턴 특정 불가"
    primary = cs[0]
    idx = _hash_idx(sample_id, f"joint_{primary}", len(JOINT_FOCUS[primary]))
    base = JOINT_FOCUS[primary][idx]
    if len(cs) > 1:
        rest = ", ".join(cs[1:])
        base = f"{base} (additional candidates: {rest})"
    return base


def _confidence_phrase(level: str, sample_id: str) -> str:
    pool = CONFIDENCE_POOL.get(level, CONFIDENCE_POOL["hedged"])
    idx = _hash_idx(sample_id, f"conf_{level}", len(pool))
    return pool[idx]


def _anchor_suffix(sample_id: str) -> str:
    idx = _hash_idx(sample_id, "anchor", len(ANCHOR_SUFFIX_POOL))
    return ANCHOR_SUFFIX_POOL[idx]


def _ambiguity_suffix(ambiguity_group: str) -> str:
    return AMBIGUITY_SUFFIX.get(ambiguity_group, AMBIGUITY_SUFFIX["uncategorized"])


def _build_phrase(row: dict) -> tuple[str, str]:
    sample_id = row["sample_id"]
    posture = row["posture_canonical"]
    level = row["caption_confidence_level"]
    ambig = row.get("ambiguity_group", "uncategorized")
    flags = json.loads(row["uncertainty_flags"]) if isinstance(row["uncertainty_flags"], str) else []
    anchor_unreliable = "anchor_unreliable" in flags
    class_set_list = json.loads(row["class_set"]) if isinstance(row["class_set"], str) else []
    class_set_sorted = sorted(class_set_list)

    posture_phrase = POSTURE_VOCAB.get(posture, "기록된 자세에서 측정된")
    class_desc = _class_set_desc(class_set_sorted, sample_id)
    joint_focus = _joint_focus_desc(class_set_sorted, sample_id)
    conf_phrase = _confidence_phrase(level, sample_id)
    ambig_suf = _ambiguity_suffix(ambig)

    # v3 template: v1 phrase + " {AMBIGUITY_SUFFIX}." 끝에 추가.
    phrase = (
        f"{posture_phrase} 손목 IMU 신호는 {class_desc}로 추정됩니다. "
        f"{joint_focus}. {conf_phrase}."
    )
    if anchor_unreliable:
        phrase += _anchor_suffix(sample_id)
    phrase += f" {ambig_suf}."

    cs_key = ",".join(class_set_sorted) if class_set_sorted else "EMPTY"
    template_key = f"{posture}|cs={cs_key}|{level}|anchor={int(anchor_unreliable)}|ambig={ambig}"
    return template_key, phrase


def main() -> int:
    print("=" * 64)
    print("Step 4R-C clinical v3 — v1 + ambiguity_group suffix")
    print("=" * 64)
    if not INPUT_SCHEMA_CSV.exists():
        raise FileNotFoundError(f"input schema CSV missing: {INPUT_SCHEMA_CSV}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"input  -> {INPUT_SCHEMA_CSV}")
    print(f"output -> {OUTPUT_CSV}")
    print()
    print(f"AMBIGUITY_SUFFIX entries:")
    for ag, suf in AMBIGUITY_SUFFIX.items():
        print(f"  {ag:30s} -> {suf}")
    print()

    df = pd.read_csv(INPUT_SCHEMA_CSV, encoding="utf-8-sig")
    print(f"loaded n={len(df)} rows")

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

    print()
    print("=== diagnostics ===")
    print(f"total rows:               {len(out_df)}")
    print(f"unique template_key:      {out_df['template_key'].nunique()}")
    print(f"unique phrases:           {out_df['phrase'].nunique()}")
    lengths = out_df["phrase"].str.len()
    print(
        f"phrase length (chars):    mean={lengths.mean():.1f} "
        f"min={lengths.min()} max={lengths.max()}"
    )

    print()
    print("ambiguity_group distribution:")
    for ag, c in out_df["ambiguity_group"].value_counts().items():
        print(f"  {ag:30s}  {c:>5}")

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
    print("=" * 64)
    print("Done.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
