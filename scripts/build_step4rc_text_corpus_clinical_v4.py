"""Step 4R-C clinical v4 — v1 + CONFIDENCE_POOL 교체.

본 스크립트는 ``build_step4rc_text_corpus_clinical.py`` (v1) 의 v4 분기.
POSTURE_VOCAB / SINGLE_CLASS_DESC / JOINT_FOCUS / ANCHOR_SUFFIX_POOL 는 v1
그대로 유지. CONFIDENCE_POOL 만 사용자 제공 wording 으로 교체.
(v3 의 ambiguity_suffix 는 적용하지 않음 — 사용자 명시.)

목적: confidence 어휘만 자연어 풍부도를 높여서 *level 신호* 가 IMU↔text
정렬에 미치는 영향을 단독 측정.

Reads (read-only):
    data/step4r/4rb_attention/experiments/b01_5_aug_jitter_scale_strong/
        seed42/schema_outputs_calibrated.csv

Writes (new file only):
    data/step4r/4rc_contrastive_optional/text_corpus_clinical_v4.csv

기존 보존 (수정 없음):
    scripts/build_step4rc_text_corpus_clinical{,_v2,_v3}.py
    data/step4r/4rc_contrastive_optional/text_corpus_clinical{,_v2,_v3}.csv

Run:
    & C:\\Users\\user\\anaconda3\\envs\\dl_env\\python.exe -X utf8 \\
        scripts/build_step4rc_text_corpus_clinical_v4.py
"""
from __future__ import annotations

import hashlib
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

INPUT_SCHEMA_CSV = (
    PROJECT_ROOT / "data" / "step4r" / "4rb_attention" / "experiments"
    / "b01_5_aug_jitter_scale_strong" / "seed42" / "schema_outputs_calibrated.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "step4r" / "4rc_contrastive_optional"
OUTPUT_CSV = OUTPUT_DIR / "text_corpus_clinical_v4.csv"


# ---------------------------------------------------------------------------
# Pool 1, 2, 3, 5: v1 그대로 (불변)
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

ANCHOR_SUFFIX_POOL = [
    " 동작 기준점이 불안정합니다 (foot anchor unreliable).",
    " 기준점 불안정으로 인한 추가 신뢰도 저하.",
]

NO_CALL_DESC = "no_call (분류 단정 불가, 신호 강도/anchor 신뢰도 부족)"


# ---------------------------------------------------------------------------
# Pool 4 (REPLACED in v4): CONFIDENCE_POOL — 사용자 제공 wording.
# 같은 cardinality (3/3/3/2) — template_key 구조 v1 동일.
# ---------------------------------------------------------------------------
CONFIDENCE_POOL: dict[str, list[str]] = {
    "confident": [
        "손목 IMU 신호가 안정적이고 패턴이 일관됩니다",
        "신호 변동이 낮고 클래스 구분이 명확합니다",
        "반복 동작 간 신호 재현성이 높습니다",
    ],
    "hedged": [
        "신호 변동이 있어 경계 패턴으로 판단됩니다",
        "복수 클래스 가능성이 공존하는 신호입니다",
        "신호 강도는 있으나 패턴이 혼재합니다",
    ],
    "low": [
        "신호 강도가 약하여 분류 신뢰도가 낮습니다",
        "패턴이 불안정하여 단일 클래스 특정이 어렵습니다",
        "신호 변동이 커서 분류 근거가 부족합니다",
    ],
    "no_call": [
        "신호 패턴이 불명확하여 분류를 보류합니다",
        "현재 신호로는 클래스 특정이 불가합니다",
    ],
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


def _build_phrase(row: dict) -> tuple[str, str]:
    sample_id = row["sample_id"]
    posture = row["posture_canonical"]
    level = row["caption_confidence_level"]
    flags = json.loads(row["uncertainty_flags"]) if isinstance(row["uncertainty_flags"], str) else []
    anchor_unreliable = "anchor_unreliable" in flags
    class_set_list = json.loads(row["class_set"]) if isinstance(row["class_set"], str) else []
    class_set_sorted = sorted(class_set_list)

    posture_phrase = POSTURE_VOCAB.get(posture, "기록된 자세에서 측정된")
    class_desc = _class_set_desc(class_set_sorted, sample_id)
    joint_focus = _joint_focus_desc(class_set_sorted, sample_id)
    conf_phrase = _confidence_phrase(level, sample_id)

    # v4 template: v1 과 동일 (CONFIDENCE_POOL 만 교체됨)
    phrase = (
        f"{posture_phrase} 손목 IMU 신호는 {class_desc}로 추정됩니다. "
        f"{joint_focus}. {conf_phrase}."
    )
    if anchor_unreliable:
        phrase += _anchor_suffix(sample_id)

    cs_key = ",".join(class_set_sorted) if class_set_sorted else "EMPTY"
    template_key = f"{posture}|cs={cs_key}|{level}|anchor={int(anchor_unreliable)}"
    return template_key, phrase


def main() -> int:
    print("=" * 64)
    print("Step 4R-C clinical v4 — v1 + CONFIDENCE_POOL 교체")
    print("=" * 64)
    if not INPUT_SCHEMA_CSV.exists():
        raise FileNotFoundError(f"input schema CSV missing: {INPUT_SCHEMA_CSV}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"input  -> {INPUT_SCHEMA_CSV}")
    print(f"output -> {OUTPUT_CSV}")
    print()
    print(f"CONFIDENCE_POOL sizes: confident={len(CONFIDENCE_POOL['confident'])}, "
          f"hedged={len(CONFIDENCE_POOL['hedged'])}, low={len(CONFIDENCE_POOL['low'])}, "
          f"no_call={len(CONFIDENCE_POOL['no_call'])}")
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
    print(f"phrase length (chars):    mean={lengths.mean():.1f} min={lengths.min()} max={lengths.max()}")

    print()
    print("sample phrases (one per confidence level):")
    seen_levels = set()
    for _, r in out_df.iterrows():
        lvl = r["caption_confidence_level"]
        if lvl in seen_levels:
            continue
        seen_levels.add(lvl)
        print(f"  [{lvl}]")
        print(f"    sample_id  = {r['sample_id']}")
        print(f"    phrase     = {r['phrase']}")
        if len(seen_levels) >= 4:
            break

    print()
    print("=" * 64)
    print("Done.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
