"""Step 4R-C clinical v2 — POSTURE / JOINT pool 도입.

본 스크립트는 ``scripts/build_step4rc_text_corpus_clinical.py`` (v1) 의 v2 분기.
v3 caption pipeline 의 POSTURE_POOL (SA 5 / CA 5 / HW 3) 과 JOINT_POOL
(C1~C6 각 3~4개) 을 4R-C clinical text corpus 에 *그대로* 적용한다.

v1 (clinical) 과의 차이:
  - POSTURE_VOCAB (자세당 1개 고정 sentence) → POSTURE_POOL (자세당 3~5개, participant_id hash)
  - JOINT_FOCUS (클래스당 2개) → JOINT_POOL (클래스당 3~4개, sample_id hash)
  - SINGLE_CLASS_DESC / CONFIDENCE_POOL / ANCHOR_SUFFIX_POOL: v1 그대로 유지
  - template: POSTURE_POOL entry 가 complete sentence (입니다. 형) 이므로
    "{POSTURE}. 손목 IMU 신호는 …" 식으로 sentence boundary 분리

Hash 기준 (v3 caption 과 동일):
  - POSTURE → participant_id 기반 (참가자 내 자세 표현 일관)
  - JOINT   → sample_id 기반 (sample 별 관절 표현 다양)
  - CONFIDENCE / ANCHOR / CLASS_DESC → sample_id 기반 (v1 그대로)

Reads (read-only):
    data/step4r/4rb_attention/experiments/b01_5_aug_jitter_scale_strong/
        seed42/schema_outputs_calibrated.csv

Writes (new file only; v1 script + v1 corpus CSV 는 보존):
    data/step4r/4rc_contrastive_optional/text_corpus_clinical_v2.csv

기존 산출물 (수정 없음):
    scripts/build_step4rc_text_corpus_clinical.py
    data/step4r/4rc_contrastive_optional/text_corpus_clinical.csv

Run:
    & python.exe -X utf8 scripts/build_step4rc_text_corpus_clinical_v2.py
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
OUTPUT_CSV = OUTPUT_DIR / "text_corpus_clinical_v2.csv"


# ---------------------------------------------------------------------------
# Pool 1: POSTURE_POOL — v3 caption pipeline 의 POSTURE_POOL 과 동일.
# participant_id hash 로 자세 entry 선택 (참가자 내 일관성).
# ---------------------------------------------------------------------------
POSTURE_POOL: dict[str, list[str]] = {
    "SA": [
        "팔을 앞으로 뻗은 자세에서 균형 유지가 가장 까다로운 조건입니다",
        "전방 거치 자세 기준으로 상체 부하 분산이 고관절 쪽으로 집중됩니다",
        "팔을 앞으로 뻗은 상태에서 척추 중립 유지에 대한 요구가 높아집니다",
        "전방 부하 조건에서 발목 가동 범위가 동작 깊이에 영향을 줍니다",
        "팔을 앞으로 뻗은 자세에서 무게중심이 전방으로 이동하는 경향이 나타납니다",
    ],
    "CA": [
        "팔을 교차한 자세에서 팔을 앞으로 뻗은 자세보다 몸의 균형 잡기가 수월합니다",
        "교차 거치 자세에서 상체 안정성이 상대적으로 높아지는 경향이 있습니다",
        "팔을 교차한 상태에서 어깨 긴장이 동작 패턴에 영향을 줄 수 있습니다",
        "교차 조건에서 흉추 가동성이 스쿼트 깊이와 연관됩니다",
        "팔을 교차한 자세에서 체간 안정화 패턴이 변화하는 경향이 보입니다",
    ],
    "HW": [
        "손을 허리에 둔 자세에서 세 가지 팔 자세 중 가장 안정적인 조건입니다",
        "허리 거치 자세에서 골반 움직임이 IMU 신호에 직접적으로 반영됩니다",
        "손을 허리에 둔 상태에서 고관절과 체간의 협응 패턴이 두드러집니다",
    ],
}


# ---------------------------------------------------------------------------
# Pool 2: SINGLE_CLASS_DESC — v1 그대로 유지 (3 per class, sample_id hash).
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Pool 3: JOINT_POOL — v3 caption pipeline 의 JOINT_POOL 과 동일.
# 클래스당 3~4개, sample_id hash 기반 선택.
# ---------------------------------------------------------------------------
JOINT_POOL: dict[str, list[str]] = {
    "C1": [
        "고관절 굴곡과 무릎 굴곡이 균형 있게 나타나는 패턴이 보입니다",
        "척추 중립이 유지된 상태에서 하강 동작이 수행되는 경향이 나타납니다",
        "발목·무릎·고관절의 협응이 안정적으로 이루어지는 패턴이 보입니다",
    ],
    "C2": [
        "고관절 굴곡 범위가 충분히 확보되지 않는 패턴이 나타납니다",
        "발목 가동 범위 제한이 스쿼트 깊이에 영향을 주는 경향이 보입니다",
        "무릎 굴곡이 목표 각도에 도달하기 전에 동작이 마무리되는 패턴이 보입니다",
    ],
    "C3": [
        "고관절 굴곡 제한과 함께 골반이 후방으로 기우는 패턴이 나타납니다",
        "요추 굴곡이 과도해지면서 척추 중립이 무너지는 경향이 보입니다",
        "무릎이 내측으로 쏠리는 패턴이 고관절 내회전과 함께 나타납니다",
        "발목·고관절·요추의 복합적인 제한 패턴이 동시에 보입니다",
    ],
    "C4": [
        "좌측 무릎이 내측으로 쏠리는 패턴이 하강 구간에서 나타납니다",
        "좌측 고관절 내회전이 무릎 정렬에 영향을 주는 경향이 보입니다",
        "좌측에 무게가 더 실리는 패턴이 신호에 나타납니다",
    ],
    "C5": [
        "우측 무릎이 내측으로 쏠리는 패턴이 하강 구간에서 나타납니다",
        "우측 고관절 내회전이 무릎 정렬에 영향을 주는 경향이 보입니다",
        "우측에 무게가 더 실리는 패턴이 신호에 나타납니다",
    ],
    "C6": [
        "양측 무릎이 동시에 내측으로 쏠리는 패턴이 나타납니다",
        "고관절 내전·내회전 패턴이 양측에서 함께 나타납니다",
        "발목 가동성 제한이 양측 무릎 정렬에 복합적으로 영향을 주는 경향이 보입니다",
    ],
}


# ---------------------------------------------------------------------------
# Pool 4: CONFIDENCE — v1 그대로.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Pool 5: ANCHOR_SUFFIX — v1 그대로.
# ---------------------------------------------------------------------------
ANCHOR_SUFFIX_POOL = [
    " 동작 기준점이 불안정합니다 (foot anchor unreliable).",
    " 기준점 불안정으로 인한 추가 신뢰도 저하.",
]


NO_CALL_DESC = "no_call (분류 단정 불가, 신호 강도/anchor 신뢰도 부족)"


def _hash_idx(key: str, salt: str, pool_len: int) -> int:
    """Stable deterministic index in [0, pool_len)."""
    h = hashlib.md5(f"{key}|{salt}".encode("utf-8")).hexdigest()[:8]
    return int(h, 16) % pool_len


def _posture_phrase(posture: str, participant_id: str) -> str:
    """POSTURE_POOL 에서 participant_id hash 로 자세 entry 선택.
    v3 caption pipeline 의 select_posture_entry() 와 *동일한* hash 키 구조."""
    pool = POSTURE_POOL.get(posture, [])
    if not pool:
        return "기록된 자세"
    idx = _hash_idx(participant_id, "posture", len(pool))
    return pool[idx]


def _class_set_desc(class_set_list: list[str], sample_id: str) -> str:
    """Map class_set to clinical description, joining individual class phrases.

    v1 그대로. 각 클래스 SINGLE_CLASS_DESC pool 에서 hash(sample_id, "cls_{c}")
    로 1개 선택. 다중 클래스는 " 또는 " 로 결합 + "(복합 의심 …)" suffix.
    """
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
    """JOINT_POOL 에서 sample_id hash 로 관절 표현 선택.
    다중 클래스 시 *첫 클래스* 의 joint focus 사용 + 보조 (additional candidates: …).
    """
    cs = sorted(class_set_list)
    if not cs:
        return "주된 관절 패턴 특정 불가"
    primary = cs[0]
    idx = _hash_idx(sample_id, f"joint_{primary}", len(JOINT_POOL[primary]))
    base = JOINT_POOL[primary][idx]
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
    participant_id = row.get("participant_id") or sample_id.split("_")[0]
    posture = row["posture_canonical"]
    level = row["caption_confidence_level"]
    flags = json.loads(row["uncertainty_flags"]) if isinstance(row["uncertainty_flags"], str) else []
    anchor_unreliable = "anchor_unreliable" in flags
    class_set_list = json.loads(row["class_set"]) if isinstance(row["class_set"], str) else []
    class_set_sorted = sorted(class_set_list)

    posture_phrase = _posture_phrase(posture, str(participant_id))
    class_desc = _class_set_desc(class_set_sorted, sample_id)
    joint_focus = _joint_focus_desc(class_set_sorted, sample_id)
    conf_phrase = _confidence_phrase(level, sample_id)

    # v2 template: POSTURE_POOL entry 가 complete sentence 이므로 ". " 분리.
    phrase = (
        f"{posture_phrase}. 손목 IMU 신호는 {class_desc}로 추정됩니다. "
        f"{joint_focus}. {conf_phrase}."
    )
    if anchor_unreliable:
        phrase += _anchor_suffix(sample_id)

    # template_key 에 posture entry id 도 포함 (v2 신규 — 추가 cardinality 추적용)
    posture_eid = _hash_idx(str(participant_id), "posture", len(POSTURE_POOL.get(posture, [1]))) + 1
    cs_key = ",".join(class_set_sorted) if class_set_sorted else "EMPTY"
    template_key = f"{posture}-{posture_eid}|cs={cs_key}|{level}|anchor={int(anchor_unreliable)}"
    return template_key, phrase


def main() -> int:
    print("=" * 64)
    print("Step 4R-C clinical v2 — POSTURE_POOL + JOINT_POOL text corpus")
    print("=" * 64)
    if not INPUT_SCHEMA_CSV.exists():
        raise FileNotFoundError(f"input schema CSV missing: {INPUT_SCHEMA_CSV}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"input  -> {INPUT_SCHEMA_CSV}")
    print(f"output -> {OUTPUT_CSV}")
    print()
    print(f"POSTURE_POOL sizes: SA={len(POSTURE_POOL['SA'])}, CA={len(POSTURE_POOL['CA'])}, HW={len(POSTURE_POOL['HW'])}")
    print(f"JOINT_POOL sizes:   {dict((c, len(v)) for c, v in JOINT_POOL.items())}")
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

    # -------- diagnostics --------
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
    print("phrase frequency by template_key (top 10):")
    for tk, c in Counter(out_df["template_key"]).most_common(10):
        print(f"  {c:>5}  {tk}")

    print()
    print("split distribution:")
    print(out_df["split"].value_counts().to_string())

    print()
    print("posture entry 분포 (participant_id hash 결과):")
    posture_eid_counter: Counter = Counter()
    for tk in out_df["template_key"]:
        # tk 형식: "SA-3|cs=...|..."
        head = tk.split("|", 1)[0]
        posture_eid_counter[head] += 1
    for k in sorted(posture_eid_counter.keys()):
        print(f"  {k:>6}: {posture_eid_counter[k]:>5}")

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
    print("=== sanity ===")
    n_unique = out_df["phrase"].nunique()
    if n_unique < 50:
        print(f"  WARN: only {n_unique} unique phrases (target ≥100). Permutation under-realized.")
    elif n_unique < 100:
        print(f"  NOTE: {n_unique} unique phrases (target 100~300). Borderline.")
    else:
        print(f"  OK: {n_unique} unique phrases. Permutation richness on target.")

    # v1 비교 (참조용 — v1 corpus 가 존재하면 unique phrase 수 비교)
    v1_csv = OUTPUT_DIR / "text_corpus_clinical.csv"
    if v1_csv.exists():
        try:
            v1_df = pd.read_csv(v1_csv, encoding="utf-8-sig")
            v1_unique = v1_df["phrase"].nunique() if "phrase" in v1_df.columns else None
            if v1_unique is not None:
                print(f"  v1 unique phrases: {v1_unique}  →  v2 unique phrases: {n_unique}  (Δ {n_unique - v1_unique:+d})")
        except Exception as e:  # noqa: BLE001
            print(f"  (v1 corpus 비교 skip: {type(e).__name__})")

    print()
    print("=" * 64)
    print("Done.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
