"""Step 7_v3 — Schema-grounded caption generation prototype (clinical pool).

본 스크립트는 ``scripts/generate_step7_v2_captions.py`` 를 base 로 하여
``reports/step7_v3/README.md`` §4 마이그레이션 표에 명시된 항목만 변경한
v3 분기 버전이다. v2 스크립트는 *수정되지 않는다*.

§4 변경 항목 (이 외에는 v2와 동일한 구조):
    - SYS_PROMPT_MD            : step6_v2 → step6_v3
    - OUTPUT_DATA_DIR          : data/step7_v2/ → data/step7_v3/
    - OUTPUT_REPORT_DIR        : reports/step7_v2/ → reports/step7_v3/
    - POSTURE_VOCAB            : 3 fixed phrase → POSTURE_POOL (13, hash 선택)
    - CLASS_VOCAB              : 6 통일 phrase → JOINT_POOL (19, hash 선택)
    - OUTPUT_REQUIRED_FIELDS   : 5 → 6 ( ``used_pool_entries`` 추가)
    - FORBIDDEN_EXACT          : 18+ → 축소판 (책임/처방/모델내부만)
    - FORBIDDEN_DIRECTION_TOKENS : active → 비활성 ([])
    - BIOMECH_TOKENS           : active → 비활성 ([])
    - build_user_payload       : constraint_template + POSTURE_POOL +
                                 JOINT_POOL + hash 결정 index 주입

Reads (read-only):
    reports/step6_v3/step6_v3_final_system_prompt.md
    reports/step6_v3/step6_v3_golden_test_cases.json     (v3 전용 — 6필드
        schema 의 example_valid_output 과 v3 pool 어휘로 작성됨. mock provider
        에서 golden_outputs hit 시 v3 validation 을 통과한다. v2 golden 은
        보존되며 step6_v2 의 v2 script 가 그대로 사용한다.)
    data/step4r/4rb_attention/step4r_bigru_attention_schema_outputs_calibrated.csv
    data/step4r/constraint_templates.json

Writes:
    [golden mode]
        data/step7_v3/step7_v3_golden_captions.csv
        data/step7_v3/step7_v3_golden_run_log.csv
    [full mode]
        data/step7_v3/step7_v3_captions.csv
        data/step7_v3/step7_v3_run_log.csv
        reports/step7_v3/step7_v3_caption_generation_summary.csv
        reports/step7_v3/step7_v3_caption_generation_results.md

Run:
    python scripts/generate_step7_v3_captions.py --mode golden --provider mock
    python scripts/generate_step7_v3_captions.py --mode golden --provider anthropic
    python scripts/generate_step7_v3_captions.py --mode full   --provider anthropic
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import argparse
import hashlib
import json
import re
import sys
import time
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


# ---------- Minimal .env loader (no python-dotenv dependency) ----------
def _load_dotenv(path: Path) -> int:
    """Load KEY=VALUE pairs from .env if present. Existing env vars are not
    overwritten. Comments (# ...) and blank lines are skipped. Returns count."""
    if not path.exists():
        return 0
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        if k and v and k not in os.environ:
            os.environ[k] = v
            n += 1
    return n

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------- Paths (§4: SYS_PROMPT_MD, OUTPUT_DATA_DIR, OUTPUT_REPORT_DIR) ----------
SYS_PROMPT_MD = PROJECT_ROOT / "reports" / "step6_v3" / "step6_v3_final_system_prompt.md"
GOLDEN_JSON = PROJECT_ROOT / "reports" / "step6_v3" / "step6_v3_golden_test_cases.json"
SCHEMA_CSV = PROJECT_ROOT / "data" / "step4r" / "4rb_attention" / "step4r_bigru_attention_schema_outputs_calibrated.csv"
CONSTRAINT_TEMPLATES_JSON = PROJECT_ROOT / "data" / "step4r" / "constraint_templates.json"

OUTPUT_DATA_DIR = PROJECT_ROOT / "data" / "step7_v3"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "reports" / "step7_v3"

# ---------- Constants from Step 6_v3 commits ----------
CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
POSTURES = ["SA", "CA", "HW"]
LEVELS = ["confident", "hedged", "low", "no_call"]

# §4: 5 → 6 필드 (used_pool_entries 추가)
OUTPUT_REQUIRED_FIELDS = {
    "caption_ko",
    "confidence_phrase",
    "uncertainty_phrase",
    "limitation_phrase",
    "used_pool_entries",
    "used_schema_fields",
}
SCHEMA_FIELDS_WHITELIST = {
    "posture_canonical",
    "class_set",
    "ambiguity_group",
    "uncertainty_flags",
    "caption_confidence_level",
    "no_call",
    "no_call_reason",
}

# §4: POSTURE_VOCAB (3 fixed) → POSTURE_POOL (13, hash 선택)
# entries 정의는 reports/step5_v3/step5_v3_pool_design.md §1 과 1:1 일치.
POSTURE_POOL = {
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
        "손을 허리에 둔 자세에서 세 가지 중 가장 안정적인 조건입니다",
        "허리 거치 자세에서 골반 움직임이 IMU 신호에 직접적으로 반영됩니다",
        "손을 허리에 둔 상태에서 고관절과 체간의 협응 패턴이 두드러집니다",
    ],
}

# §4: CLASS_VOCAB (6 통일 phrase) → JOINT_POOL (19, hash 선택)
# entries 정의는 reports/step5_v3/step5_v3_pool_design.md §2 와 1:1 일치.
JOINT_POOL = {
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

# Multi-class 결합용 (pool_design.md §4)
CONNECTORS = [" 동시에 ", " 함께 ", " 한편 "]

AMBIGUITY_PHRASES = {
    "within_group_c1_c5_c6": "정상 패턴과 무릎 관련 비대칭 패턴이 함께 나타나는 경계 신호로 보입니다",
    "pair_c3_c4": "복합 결함 두 가지가 함께 나타나는 경계 신호로 보입니다",
    "pair_plus_c2_absorption": "깊이 부족 신호와 무릎 관련 비대칭 신호가 함께 나타납니다",
    "uncategorized": "단일 패턴으로 좁히기 어려운 경계 신호입니다",
    "no_call": "현재 손목 센서 신호만으로는 안정적인 설명을 제공하기 어렵습니다",
    "confident_C2": "",
}

# v2와 동일 (§4 미포함, 시스템 프롬프트에서 어조 modifier 로 사용)
CONFIDENCE_PHRASE_POOL = {
    "confident": ["비교적 일관되게 나타납니다", "상대적으로 안정적인 신호입니다"],
    "hedged": [
        "~에 가까운 경향이 있습니다",
        "~ 가능성이 함께 나타납니다",
        "단일 유형으로 좁히기 어렵습니다",
        "경계 신호로 보입니다",
    ],
    "low": [
        "불확실성이 큽니다",
        "후보군 수준으로만 해석하는 것이 적절합니다",
    ],
    "no_call": [
        "현재 손목 센서 신호만으로는 안정적인 설명을 제공하기 어렵습니다",
    ],
}

# v2 의 3개 + v3 정책 §3.4 권장 1개 (longer)
LIMITATION_PHRASE_POOL = [
    "손목 센서 기준의 추정",
    "손목 센서는 무릎 각도나 골반 움직임을 직접 측정하지 않습니다",
    "정확한 자세 평가는 영상 또는 전문가 평가와 함께 해석해 주십시오",
    "본 설명은 손목 IMU 신호 기반의 추정이며, 무릎 각도·골반 운동·발목 가동 범위는 직접 측정되지 않습니다",
]

with open(CONSTRAINT_TEMPLATES_JSON, "r", encoding="utf-8") as _f:
    CONSTRAINT_TEMPLATES = json.load(_f)

# §4: FORBIDDEN_EXACT — 책임/처방/모델내부만 (clinical vocab 허용)
# step5_v3_forbidden_expressions.md §2.1~2.2 와 1:1 일치.
FORBIDDEN_EXACT = [
    # F1~F4: 의학적 진단 단정
    "진단된다",
    "확실히",
    "확실하다",
    "명확히 잘못",
    "전문가가 확인한 것과 같",
    # F5~F9: 부상/치료/교정 권고
    "부상 위험이 높",
    "부상으로 이어진",
    "치료가 필요",
    "교정해야",
    "고쳐야",
]

# §4: 방향성 단정 허용 → 비활성
FORBIDDEN_DIRECTION_TOKENS: list[str] = []

# v2와 동일 (모델 내부 어휘는 v3에서도 금지)
ATTENTION_LEAK_TOKENS = [
    "attention",
    "Attention",
    "ascending phase",
    "descending phase",
    "bottom_transition",
    "anchor",
    "Anchor",
    "peak timestep",
    "predictive entropy",
    "attention entropy",
    "posterior",
    "logit",
    "softmax",
    "어텐션",
    "엔트로피",
    "포스테리어",
    "로짓",
]
RAW_CLASS_LABEL_RE = re.compile(r"\bC[1-6]\b")

# §4: clinical vocabulary 허용 → 비활성
BIOMECH_TOKENS: list[str] = []
# BIOMECH_TOKENS 가 비어 있으므로 negation fragment 도 사용되지 않음 (loop dead code).
ALLOWED_BIOMECH_NEGATION_FRAGMENTS: list[str] = []


# ---------- Hash-based pool selection (pool_design.md §5) ----------

def _hash_idx(key: str, pool_len: int) -> int:
    """Stable deterministic index in [0, pool_len)."""
    if pool_len <= 0:
        return 0
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
    return int(h, 16) % pool_len


def select_posture_entry(posture: str, participant_id: str) -> tuple[str, str]:
    """Returns (entry_id, phrase). entry_id format: 'SA-3' / 'CA-1' / 'HW-2'."""
    pool = POSTURE_POOL.get(posture, [])
    if not pool:
        return "", ""
    idx = _hash_idx(f"{participant_id}|posture", len(pool))
    return f"{posture}-{idx + 1}", pool[idx]


def select_joint_entries(class_set_sorted: list, sample_id: str) -> list[tuple[str, str]]:
    """Returns list of (entry_id, phrase). entry_id format: 'J-C2-2'."""
    out: list[tuple[str, str]] = []
    for c in class_set_sorted:
        pool = JOINT_POOL.get(c, [])
        if not pool:
            continue
        idx = _hash_idx(f"{sample_id}|joint_{c}", len(pool))
        out.append((f"J-{c}-{idx + 1}", pool[idx]))
    return out


def select_connector(sample_id: str) -> str:
    idx = _hash_idx(f"{sample_id}|conn", len(CONNECTORS))
    return CONNECTORS[idx]


def select_limitation(sample_id: str) -> str:
    idx = _hash_idx(f"{sample_id}|lim", len(LIMITATION_PHRASE_POOL))
    return LIMITATION_PHRASE_POOL[idx]


def _resolve_participant_id(schema: dict) -> str:
    pid = schema.get("participant_id", "")
    if pid:
        return str(pid)
    sid = str(schema.get("sample_id", ""))
    return sid.split("_")[0] if sid else ""


# ---------- Validator helpers (v3 pool semantics) ----------

USED_POOL_ENTRY_RE = re.compile(r"^(SA-\d+|CA-\d+|HW-\d+|J-C\d+-\d+)$")


def _is_valid_pool_entry(eid: str) -> bool:
    if not isinstance(eid, str) or not USED_POOL_ENTRY_RE.match(eid):
        return False
    if eid.startswith(("SA-", "CA-", "HW-")):
        posture, idx_s = eid.split("-", 1)
        try:
            idx = int(idx_s) - 1
        except ValueError:
            return False
        return 0 <= idx < len(POSTURE_POOL.get(posture, []))
    if eid.startswith("J-"):
        parts = eid.split("-")
        if len(parts) != 3:
            return False
        _, c, idx_s = parts
        try:
            idx = int(idx_s) - 1
        except ValueError:
            return False
        return 0 <= idx < len(JOINT_POOL.get(c, []))
    return False


FALLBACK_OUTPUT = {
    "caption_ko": "현재 손목 센서 신호만으로는 안정적인 설명을 제공하기 어렵습니다.",
    "confidence_phrase": "안정적인 설명을 제공하기 어렵습니다",
    "uncertainty_phrase": "",
    "limitation_phrase": "",
    "used_pool_entries": [],
    "used_schema_fields": ["caption_confidence_level"],
}


def build_fallback(schema: dict) -> dict:
    """schema-aware fallback. caption_ko 가 POSTURE_POOL entry 로 시작하도록.
    no_call 메시지여도 posture 표현 강제 정책 (step6_v3 system prompt §6) 을
    따른다."""
    posture = schema.get("posture_canonical", "") if schema else ""
    pid = _resolve_participant_id(schema) if schema else ""
    p_eid, p_phrase = select_posture_entry(posture, pid)
    if not p_phrase:
        p_phrase = "기록된 자세"

    used_pool = [p_eid] if p_eid else []
    used_schema = ["caption_confidence_level"]
    if posture in POSTURE_POOL:
        used_schema.insert(0, "posture_canonical")

    return {
        "caption_ko": (
            f"{p_phrase}. 현재 손목 센서 신호만으로는 안정적인 설명을 "
            "제공하기 어렵습니다."
        ),
        "confidence_phrase": "안정적인 설명을 제공하기 어렵습니다",
        "uncertainty_phrase": "",
        "limitation_phrase": "",
        "used_pool_entries": used_pool,
        "used_schema_fields": used_schema,
    }


# ---------- System prompt loader ----------

def load_system_prompt() -> str:
    text = SYS_PROMPT_MD.read_text(encoding="utf-8")
    m = re.search(r"```text\s*\n(.*?)\n```", text, re.DOTALL)
    if not m:
        m = re.search(r"```\s*\n(.*?)\n```", text, re.DOTALL)
    if not m:
        raise ValueError(f"Could not extract fenced code block from {SYS_PROMPT_MD}")
    return m.group(1).strip()


# ---------- User payload builder (§4: POSTURE_POOL + JOINT_POOL + hash index 주입) ----------

def build_user_payload(schema_row: dict) -> dict:
    cs = schema_row["class_set"]
    if isinstance(cs, str):
        cs = json.loads(cs)
    uf = schema_row["uncertainty_flags"]
    if isinstance(uf, str):
        uf = json.loads(uf)
    nr = schema_row.get("no_call_reason", "") or ""
    if isinstance(nr, float):
        nr = ""

    posture = str(schema_row["posture_canonical"])
    sample_id = str(schema_row["sample_id"])
    participant_id = _resolve_participant_id(schema_row)
    ag = str(schema_row.get("ambiguity_group", ""))
    no_call = bool(schema_row.get("no_call", False))
    cs_sorted = sorted(cs)

    posture_entry_id, _ = select_posture_entry(posture, participant_id)
    joint_selections = select_joint_entries(cs_sorted, sample_id) if not no_call else []
    joint_entry_ids = [eid for eid, _ in joint_selections]
    connector = select_connector(sample_id) if len(joint_entry_ids) >= 2 else ""
    ambiguity_phrase = AMBIGUITY_PHRASES.get(ag, "")
    limitation_suggestion = select_limitation(sample_id) if not no_call else ""

    relevant_joint_pool = {c: JOINT_POOL[c] for c in cs_sorted if c in JOINT_POOL}

    relevant_constraints = [
        {"constraint_id": t["constraint_id"], "caption_safe_expression": t["caption_safe_expression"]}
        for t in CONSTRAINT_TEMPLATES
        if set(t["target_classes"]) & set(cs)
    ]
    if no_call:
        relevant_constraints = []

    return {
        "task": "schema_to_korean_caption_v3",
        "schema": {
            "sample_id": sample_id,
            "posture_canonical": posture,
            "class_set": list(cs),
            "ambiguity_group": ag,
            "uncertainty_flags": list(uf),
            "caption_confidence_level": str(schema_row["caption_confidence_level"]),
            "no_call": no_call,
            "no_call_reason": str(nr),
        },
        "posture_pool": POSTURE_POOL,
        "joint_pool": relevant_joint_pool,
        "posture_entry_id": posture_entry_id,
        "joint_entry_ids": joint_entry_ids,
        "connector": connector,
        "ambiguity_phrase": ambiguity_phrase,
        "limitation_phrase_suggestion": limitation_suggestion,
        "limitation_pool": LIMITATION_PHRASE_POOL,
        "confidence_phrase_pool": CONFIDENCE_PHRASE_POOL,
        "constraint_template": relevant_constraints,
    }


# ---------- Response parser ----------

def parse_response(raw: str | None) -> tuple[dict | None, str | None]:
    if raw is None:
        return None, "raw_is_none"
    text = raw.strip()
    if text.startswith("```"):
        m = re.match(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"json_parse_error:{type(e).__name__}"
    if not isinstance(data, dict):
        return None, "not_a_dict"
    return data, None


# ---------- Validator ----------
# §4 변경에 따른 분기:
#   - OUTPUT_REQUIRED_FIELDS 6필드 강제
#   - FORBIDDEN_EXACT 축소판으로 검사
#   - FORBIDDEN_DIRECTION_TOKENS / BIOMECH_TOKENS 비어 있어 dead-loop (정책 §1)
#   - posture phrase consistency: POSTURE_POOL[posture] 의 *어떤 entry* 라도 cap 에 포함
#   - no_call contradiction: JOINT_POOL flatten 한 phrase 가 cap 에 등장하면 invalid
#   - class_set narrowing: used_pool_entries 가 class_set 각 클래스를 J-* 로 참조
#   - used_pool_entries 신규 검사: 형식 + 자세/관절 entry 존재

def validate_caption(output: dict, schema: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []

    missing = OUTPUT_REQUIRED_FIELDS - set(output.keys())
    if missing:
        errors.append("missing_fields:" + ",".join(sorted(missing)))
    extra = set(output.keys()) - OUTPUT_REQUIRED_FIELDS
    if extra:
        errors.append("extra_fields:" + ",".join(sorted(extra)))
    if errors:
        return False, errors

    cap = (output.get("caption_ko") or "").strip()
    cp = (output.get("confidence_phrase") or "").strip()
    ucp = (output.get("uncertainty_phrase") or "").strip()
    lp = (output.get("limitation_phrase") or "").strip()
    upe = output.get("used_pool_entries") or []
    usf = output.get("used_schema_fields") or []
    full_text = " ".join([cap, cp, ucp, lp])

    if not cap:
        errors.append("caption_ko_empty")

    if RAW_CLASS_LABEL_RE.search(cap):
        errors.append("raw_class_label_in_caption")

    for tok in FORBIDDEN_EXACT:
        if tok in full_text:
            errors.append(f"forbidden_token:{tok}")
            break

    for tok in FORBIDDEN_DIRECTION_TOKENS:
        if tok in cap:
            errors.append(f"direction_token:{tok}")
            break

    for tok in ATTENTION_LEAK_TOKENS:
        if tok in full_text:
            errors.append(f"attention_leak:{tok}")
            break

    no_call = bool(schema.get("no_call", False))
    cs = schema.get("class_set", [])
    if isinstance(cs, str):
        try:
            cs = json.loads(cs)
        except Exception:
            cs = []
    ag = schema.get("ambiguity_group", "")

    if no_call:
        for c, phrases in JOINT_POOL.items():
            hit = False
            for phrase in phrases:
                if phrase in cap:
                    errors.append(f"no_call_contradiction:{c}_pool_entry")
                    hit = True
                    break
            if hit:
                break

    if not no_call and isinstance(cs, list) and len(cs) >= 2:
        referenced_classes: set[str] = set()
        for eid in upe:
            if isinstance(eid, str) and eid.startswith("J-"):
                parts = eid.split("-")
                if len(parts) >= 3:
                    referenced_classes.add(parts[1])
        missing_cls = set(cs) - referenced_classes
        if missing_cls:
            errors.append(
                "class_set_narrowing:" + ",".join(sorted(missing_cls)) + "_not_referenced"
            )

    # v3: BIOMECH_TOKENS 비어 있어 dead-loop (호환성 유지 차원에서 형식만 보존)
    cap_for_biomech = cap
    for ap in LIMITATION_PHRASE_POOL:
        if ap:
            cap_for_biomech = cap_for_biomech.replace(ap, "")
    for frag in ALLOWED_BIOMECH_NEGATION_FRAGMENTS:
        cap_for_biomech = cap_for_biomech.replace(frag, "")
    for tok in BIOMECH_TOKENS:
        if tok in cap_for_biomech:
            errors.append(f"biomech_claim:{tok}")
            break

    posture = schema.get("posture_canonical", "")
    expected_pool = POSTURE_POOL.get(posture, [])
    if expected_pool:
        if not any(ph in cap for ph in expected_pool):
            errors.append(f"posture_phrase_missing:{posture}")
        for p, phrases in POSTURE_POOL.items():
            if p == posture:
                continue
            wrong_hit = False
            for ph in phrases:
                if ph in cap:
                    errors.append(f"wrong_posture_phrase:{p}")
                    wrong_hit = True
                    break
            if wrong_hit:
                break

    # used_pool_entries 신규 검사
    if not isinstance(upe, list):
        errors.append("used_pool_entries_not_list")
    else:
        invalid_pool = [e for e in upe if not _is_valid_pool_entry(e)]
        if invalid_pool:
            errors.append("used_pool_entries_invalid:" + ",".join(str(x) for x in invalid_pool[:5]))
        posture_in_upe = any(
            isinstance(e, str) and e.startswith(("SA-", "CA-", "HW-")) for e in upe
        )
        if not posture_in_upe:
            errors.append("used_pool_entries_missing_posture")
        if not no_call:
            joint_in_upe = any(isinstance(e, str) and e.startswith("J-") for e in upe)
            if not joint_in_upe:
                errors.append("used_pool_entries_missing_joint")

    level = schema.get("caption_confidence_level", "")
    confident_phrases = ["비교적 일관되게", "안정적인 신호"]
    no_call_phrases = ["안정적인 설명을 제공하기 어렵습니다"]
    if level == "no_call":
        if not any(p in cp or p in cap for p in no_call_phrases):
            errors.append("confidence_mismatch_no_call_phrase_missing")
    elif level == "low":
        if any(p in cp for p in confident_phrases):
            errors.append("confidence_mismatch_low_uses_confident")
    elif level == "confident":
        if any(p in cp for p in ["좁히기 어렵", "후보군 수준"]):
            errors.append("confidence_mismatch_confident_uses_hedge")

    usf_set = set(usf) if isinstance(usf, list) else set()
    if not usf_set:
        errors.append("used_schema_fields_empty")
    else:
        invalid_fields = usf_set - SCHEMA_FIELDS_WHITELIST
        if invalid_fields:
            errors.append("used_schema_fields_invalid:" + ",".join(sorted(invalid_fields)))
        if no_call and "no_call" not in usf_set:
            errors.append("used_schema_fields_missing_no_call")

    if level != "no_call":
        if not lp and "손목 센서" not in cap and "직접 측정" not in cap:
            errors.append("limitation_phrase_missing")
        # v3: limitation phrase 길이 상한 완화 (policy §3.4 의 longer phrase 허용)
        if len(lp) > 300:
            errors.append("limitation_phrase_too_long")

    return len(errors) == 0, errors


# ---------- Providers ----------

class MockProvider:
    """Rule-based + golden-example mock. Dev only.

    GOLDEN_JSON 은 step6_v3 의 v3-shape (6필드) example_valid_output 을
    가지므로 golden_outputs hit 시 v3 validation 을 통과한다.
    rule-based mock 만 단독으로 평가하려면 ``golden_outputs_by_sample={}`` 로
    instantiate 한다.
    """

    name = "mock"
    model = "mock"

    def __init__(self, golden_outputs_by_sample: dict | None = None):
        self.golden_outputs = golden_outputs_by_sample or {}

    def call(self, system_prompt: str, user_payload: dict, **kwargs) -> str:
        sid = user_payload["schema"]["sample_id"]
        if sid in self.golden_outputs:
            return json.dumps(self.golden_outputs[sid], ensure_ascii=False)
        return json.dumps(self._build_rule_based(user_payload), ensure_ascii=False)

    def _build_rule_based(self, payload: dict) -> dict:
        schema = payload["schema"]
        posture = schema["posture_canonical"]
        class_set = schema.get("class_set", [])
        ag = schema.get("ambiguity_group", "")
        flags = schema.get("uncertainty_flags", [])
        level = schema.get("caption_confidence_level", "")
        no_call = bool(schema.get("no_call", False))

        posture_entry_id = payload.get("posture_entry_id", "")
        joint_entry_ids = payload.get("joint_entry_ids", [])
        connector = payload.get("connector", " 동시에 ")
        ambiguity_phrase = payload.get("ambiguity_phrase", "")
        limitation_suggestion = payload.get("limitation_phrase_suggestion", "")

        # posture phrase 해석
        posture_phrase = ""
        if posture_entry_id:
            try:
                idx = int(posture_entry_id.split("-")[1]) - 1
                posture_phrase = POSTURE_POOL[posture][idx]
            except (ValueError, IndexError, KeyError):
                posture_phrase = POSTURE_POOL.get(posture, ["기록된 자세"])[0]
        else:
            posture_phrase = POSTURE_POOL.get(posture, ["기록된 자세"])[0]

        used_pool = [posture_entry_id] if posture_entry_id else []
        used_schema_base = ["posture_canonical", "caption_confidence_level"]

        if no_call:
            reason = schema.get("no_call_reason", "") or ""
            if "anchor" in reason:
                cap = (
                    f"{posture_phrase}. 동작 기준점 신뢰도가 낮아, 현재 손목 센서 "
                    "신호만으로는 안정적인 설명을 제공하기 어렵습니다."
                )
                ucp = "동작 기준점 신뢰도가 낮음"
            else:
                cap = (
                    f"{posture_phrase}. 신호 강도가 부족하여, 현재 손목 센서 "
                    "신호만으로는 안정적인 설명을 제공하기 어렵습니다."
                )
                ucp = "신호 강도가 부족함"
            return {
                "caption_ko": cap,
                "confidence_phrase": "안정적인 설명을 제공하기 어렵습니다",
                "uncertainty_phrase": ucp,
                "limitation_phrase": "",
                "used_pool_entries": used_pool,
                "used_schema_fields": used_schema_base + ["no_call", "no_call_reason"],
            }

        # joint phrases 해석
        joint_phrases: list[str] = []
        for eid in joint_entry_ids:
            try:
                _, c, idx_s = eid.split("-")
                idx = int(idx_s) - 1
                joint_phrases.append(JOINT_POOL[c][idx])
                used_pool.append(eid)
            except (ValueError, IndexError, KeyError):
                continue

        if level == "confident":
            cp = "비교적 일관되게 나타납니다"
        elif level == "hedged":
            cp = "단일 유형으로 좁히기 어렵습니다"
        elif level == "low":
            cp = "후보군 수준으로만 해석하는 것이 적절합니다"
        else:
            cp = "안정적인 설명을 제공하기 어렵습니다"

        joint_text = ""
        if joint_phrases:
            joint_text = connector.join(joint_phrases) if len(joint_phrases) >= 2 else joint_phrases[0]

        parts = [f"{posture_phrase}."]
        if joint_text:
            parts.append(f"{joint_text}.")
        if ambiguity_phrase and len(class_set) >= 2:
            parts.append(f"{ambiguity_phrase}.")
        parts.append(f"{cp}.")
        lp_text = limitation_suggestion or "손목 센서 기준의 추정"
        parts.append(f"{lp_text}.")
        cap = " ".join(parts)

        ucp = ambiguity_phrase

        used_schema_final = list(used_schema_base)
        if class_set:
            used_schema_final.append("class_set")
        if ag:
            used_schema_final.append("ambiguity_group")
        if flags:
            used_schema_final.append("uncertainty_flags")

        return {
            "caption_ko": cap,
            "confidence_phrase": cp,
            "uncertainty_phrase": ucp,
            "limitation_phrase": lp_text,
            "used_pool_entries": used_pool,
            "used_schema_fields": used_schema_final,
        }


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "anthropic SDK not installed. `pip install anthropic` or use --provider mock."
            ) from e
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY env var not set. "
                "Set it (or fill .env) or use --provider mock for development."
            )
        resolved_model = (
            model
            or os.environ.get("ANTHROPIC_MODEL")
            or "claude-sonnet-4-20250514"
        )
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = resolved_model

    def call(
        self,
        system_prompt: str,
        user_payload: dict,
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        user_text = json.dumps(user_payload, ensure_ascii=False, indent=2)
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_text}],
        )
        out = ""
        for block in msg.content:
            if hasattr(block, "text"):
                out += block.text
        return out.strip()


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "openai SDK not installed. `pip install openai` or use --provider mock."
            ) from e
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY env var not set. "
                "Set it or use --provider mock for development."
            )
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def call(
        self,
        system_prompt: str,
        user_payload: dict,
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        user_text = json.dumps(user_payload, ensure_ascii=False, indent=2)
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
        )
        return resp.choices[0].message.content


# ---------- Generation with retry ----------

def generate_one(
    provider,
    system_prompt: str,
    user_payload: dict,
    schema: dict,
    max_retries: int = 3,
    **call_kwargs,
) -> tuple[dict, list[dict], str, int, list[str]]:
    """Returns (output, attempt_logs, final_status, n_retries, last_errors)."""
    logs: list[dict] = []
    last_errors: list[str] = []
    for attempt in range(1, max_retries + 1):
        try:
            raw = provider.call(system_prompt, user_payload, **call_kwargs)
        except Exception as e:  # noqa: BLE001
            logs.append({
                "attempt": attempt,
                "status": "provider_error",
                "validation_pass": False,
                "validation_errors": f"provider_error:{type(e).__name__}",
                "raw_response_excerpt": str(e)[:200],
            })
            last_errors = [f"provider_error:{type(e).__name__}"]
            continue
        data, parse_err = parse_response(raw)
        if parse_err:
            logs.append({
                "attempt": attempt,
                "status": "json_parse_failed",
                "validation_pass": False,
                "validation_errors": parse_err,
                "raw_response_excerpt": (raw or "")[:200],
            })
            last_errors = [parse_err]
            continue
        ok, errors = validate_caption(data, schema)
        logs.append({
            "attempt": attempt,
            "status": "validated",
            "validation_pass": ok,
            "validation_errors": "|".join(errors),
            "raw_response_excerpt": (raw or "")[:200],
        })
        last_errors = list(errors)
        if ok:
            final_status = "ok" if attempt == 1 else "retry_then_ok"
            return data, logs, final_status, attempt - 1, []
    return build_fallback(schema), logs, "fallback", max_retries, last_errors


# ---------- Markdown report builder ----------

def build_markdown_report(args, summary: dict, captions: list[dict]) -> str:
    L = []
    L.append("# Step 7_v3 — Caption Generation Prototype 결과 (clinical pool)")
    L.append("")
    L.append(f"- 생성 스크립트: `scripts/generate_step7_v3_captions.py`")
    L.append(f"- mode: `{args.mode}`")
    L.append(f"- provider: `{args.provider}`")
    L.append(f"- model: `{summary['model_name']}`")
    L.append(f"- temperature: `{args.temperature}` / max_tokens: `{args.max_tokens}` / sleep: `{args.sleep}s`")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 1. 본 단계의 위치")
    L.append("")
    L.append(
        "Step 7_v3 는 Step 6_v3 가 commit 한 clinical pool 기반 prompt / output "
        "schema / 정책 위에서 한국어 caption 을 생성하는 분기이다. v2 (sensor "
        "observability 우선) 와 *병렬* 로 운영되며, v2 baseline 산출물 "
        "(reports/step7_v2/) 은 본 분기로 인해 수정되지 않는다. Step 8_v2 "
        "automatic schema-caption validation 의 *입력* 으로 v2 와 함께 비교 "
        "metric 을 생성한다."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 2. 실행 설정")
    L.append("")
    L.append("| 항목 | 값 |")
    L.append("|---|---|")
    L.append(f"| mode | {args.mode} |")
    L.append(f"| provider | {args.provider} |")
    L.append(f"| model | {summary['model_name']} |")
    L.append(f"| temperature | {args.temperature} |")
    L.append(f"| max_tokens | {args.max_tokens} |")
    L.append(f"| inter-call sleep | {args.sleep}s |")
    L.append(f"| max_retries | 3 (then fallback) |")
    L.append(f"| seed | {args.seed} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 3. 종합 결과")
    L.append("")
    L.append("| 지표 | 값 |")
    L.append("|---|---:|")
    L.append(f"| 총 sample | {summary['n']} |")
    L.append(f"| 첫 시도 통과 | {summary['pass_first']} |")
    L.append(f"| retry 후 통과 | {summary['pass_after_retry']} |")
    L.append(f"| fallback | {summary['fallback']} |")
    L.append(f"| schema_faithfulness_rate | {summary['schema_faithfulness_rate']:.4f} |")
    L.append(f"| fallback_rate | {summary['fallback_rate']:.4f} |")
    L.append(f"| mean_n_retries | {summary['mean_n_retries']:.4f} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 4. confidence level별 caption 수")
    L.append("")
    df = pd.DataFrame(captions)
    L.append("| level | count | rate |")
    L.append("|---|---:|---:|")
    n = len(df)
    for level in LEVELS:
        c = int((df["caption_confidence_level"] == level).sum())
        L.append(f"| {level} | {c} | {(c / n if n else 0):.4f} |")
    L.append("")
    L.append("## 5. ambiguity group별 caption 수")
    L.append("")
    L.append("| group | count | rate |")
    L.append("|---|---:|---:|")
    for ag in sorted(df["ambiguity_group"].unique()):
        c = int((df["ambiguity_group"] == ag).sum())
        L.append(f"| {ag} | {c} | {(c / n if n else 0):.4f} |")
    L.append("")
    n_no_call = int(df["no_call"].astype(bool).sum())
    L.append(f"## 6. no_call caption 수: {n_no_call} / {n} ({n_no_call / n if n else 0:.4f})")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 7. 위반 발생 집계")
    L.append("")
    forb_count = int(df["validation_errors"].fillna("").str.contains("forbidden_token").sum())
    attn_count = int(df["validation_errors"].fillna("").str.contains("attention_leak").sum())
    narrow_count = int(df["validation_errors"].fillna("").str.contains("class_set_narrowing").sum())
    pool_invalid_count = int(df["validation_errors"].fillna("").str.contains("used_pool_entries_invalid").sum())
    L.append("| 검사 | invalid count |")
    L.append("|---|---:|")
    L.append(f"| forbidden expression (책임/처방) | {forb_count} |")
    L.append(f"| attention term leakage | {attn_count} |")
    L.append(f"| class_set narrowing | {narrow_count} |")
    L.append(f"| used_pool_entries 형식 오류 | {pool_invalid_count} |")
    L.append(f"| fallback | {summary['fallback']} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 8. 대표 caption 예시 10개")
    L.append("")
    samples = df.sample(min(10, len(df)), random_state=args.seed) if len(df) > 0 else df
    for _, r in samples.iterrows():
        L.append(f"**{r['sample_id']}** ({r['posture_canonical']}, {r['ambiguity_group']}, {r['caption_confidence_level']})")
        L.append("")
        L.append(f"> {r['caption_ko']}")
        L.append("")
    L.append("---")
    L.append("")
    L.append("## 9. fallback 또는 invalid 예시")
    L.append("")
    invalid = df[df["final_status"] == "fallback"].head(5)
    if len(invalid) == 0:
        L.append("fallback 발생 없음.")
    else:
        for _, r in invalid.iterrows():
            L.append(f"- `{r['sample_id']}`: errors=`{r['validation_errors']}`")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 10. Step 8_v2 automatic validation 으로 넘기는 산출물")
    L.append("")
    L.append("- `data/step7_v3/step7_v3_captions.csv` (full mode 결과)")
    L.append("- `data/step7_v3/step7_v3_run_log.csv` (per-attempt log)")
    L.append("- `data/step7_v3/step7_v3_golden_captions.csv` (golden cases)")
    L.append("- `data/step7_v3/step7_v3_golden_run_log.csv`")
    L.append("- `reports/step7_v3/step7_v3_caption_generation_summary.csv`")
    L.append("- `reports/step7_v3/step7_v3_golden_validation_summary.csv`")
    L.append("- `reports/step7_v3/step7_v3_caption_generation_results.md` (본 보고서)")
    L.append("")
    L.append("---")
    L.append("")
    L.append("*본 보고서는 자동 생성된다. v2 산출물 (reports/step7_v2/, data/step7_v2/) 은 본 분기로 인해 수정되지 않는다.*")
    return "\n".join(L)


# ---------- Main ----------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["golden", "full"], required=True)
    parser.add_argument("--provider", choices=["mock", "anthropic", "openai"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    n_loaded = _load_dotenv(PROJECT_ROOT / ".env")
    if n_loaded:
        print(f"loaded {n_loaded} env var(s) from .env")

    sys_prompt = load_system_prompt()
    print(f"loaded system prompt: {len(sys_prompt)} chars")

    with open(GOLDEN_JSON, encoding="utf-8") as f:
        golden = json.load(f)
    print(f"loaded {len(golden)} golden cases (v2-shaped — see MockProvider NOTE)")

    if args.provider == "mock":
        mock_outputs = {}
        for case in golden:
            sid = case["input_schema"]["sample_id"]
            if sid not in mock_outputs:
                mock_outputs[sid] = case["example_valid_output"]
        provider = MockProvider(mock_outputs)
        print(f"MockProvider with {len(mock_outputs)} preloaded golden examples (v2 schema)")
    elif args.provider == "anthropic":
        provider = AnthropicProvider(model=args.model)
        print(f"AnthropicProvider model={provider.model}")
    elif args.provider == "openai":
        model = args.model or "gpt-4o-mini"
        provider = OpenAIProvider(model=model)
        print(f"OpenAIProvider model={model}")
    else:
        raise ValueError(f"unknown provider: {args.provider}")

    if args.mode == "golden":
        rows = []
        for case in golden:
            sch = dict(case["input_schema"])
            sch["case_id"] = case["case_id"]
            sch["participant_id"] = sch["sample_id"].split("_")[0]
            sch["split"] = "golden"
            rows.append(sch)
        out_caps_path = OUTPUT_DATA_DIR / "step7_v3_golden_captions.csv"
        out_log_path = OUTPUT_DATA_DIR / "step7_v3_golden_run_log.csv"
    else:
        df = pd.read_csv(SCHEMA_CSV, encoding="utf-8-sig")
        df["class_set"] = df["class_set"].apply(json.loads)
        df["uncertainty_flags"] = df["uncertainty_flags"].apply(json.loads)
        rows = df.to_dict("records")
        out_caps_path = OUTPUT_DATA_DIR / "step7_v3_captions.csv"
        out_log_path = OUTPUT_DATA_DIR / "step7_v3_run_log.csv"

    if args.limit:
        rows = rows[: args.limit]
    print(f"mode={args.mode}, n_rows={len(rows)}")

    captions: list[dict] = []
    run_logs: list[dict] = []
    n_pass_first = 0
    n_pass_after_retry = 0
    n_fallback = 0
    n_retries_total = 0

    t0 = time.time()
    for i, schema in enumerate(rows):
        if isinstance(schema.get("class_set"), str):
            schema["class_set"] = json.loads(schema["class_set"])
        if isinstance(schema.get("uncertainty_flags"), str):
            schema["uncertainty_flags"] = json.loads(schema["uncertainty_flags"])
        schema["no_call"] = bool(schema.get("no_call", False))
        nr = schema.get("no_call_reason", "")
        if nr is None or (isinstance(nr, float) and pd.isna(nr)):
            schema["no_call_reason"] = ""
        else:
            schema["no_call_reason"] = str(nr)

        user_payload = build_user_payload(schema)
        output, logs, final_status, n_retries, last_errors = generate_one(
            provider,
            sys_prompt,
            user_payload,
            schema,
            max_retries=3,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        if final_status == "ok":
            n_pass_first += 1
        elif final_status == "retry_then_ok":
            n_pass_after_retry += 1
        else:
            n_fallback += 1
        n_retries_total += n_retries

        validation_pass = final_status != "fallback"
        if final_status != "fallback":
            ok, errors = validate_caption(output, schema)
        else:
            ok = False
            errors = ["fallback_used"] + last_errors

        cap_row = {
            "sample_id": schema["sample_id"],
            "participant_id": schema.get("participant_id", ""),
            "split": schema.get("split", ""),
            "posture_canonical": schema["posture_canonical"],
            "class_set": json.dumps(schema["class_set"], ensure_ascii=False),
            "ambiguity_group": schema["ambiguity_group"],
            "uncertainty_flags": json.dumps(schema["uncertainty_flags"], ensure_ascii=False),
            "caption_confidence_level": schema["caption_confidence_level"],
            "no_call": bool(schema["no_call"]),
            "no_call_reason": schema.get("no_call_reason", ""),
            "caption_ko": output.get("caption_ko", ""),
            "confidence_phrase": output.get("confidence_phrase", ""),
            "uncertainty_phrase": output.get("uncertainty_phrase", ""),
            "limitation_phrase": output.get("limitation_phrase", ""),
            "used_pool_entries": json.dumps(output.get("used_pool_entries", []), ensure_ascii=False),
            "used_schema_fields": json.dumps(output.get("used_schema_fields", []), ensure_ascii=False),
            "final_status": final_status,
            "n_retries": n_retries,
            "validation_pass": validation_pass,
            "validation_errors": "|".join(errors) if errors else "",
            "provider": provider.name,
            "model_name": getattr(provider, "model", "mock"),
        }
        captions.append(cap_row)

        for log in logs:
            run_logs.append({
                "sample_id": schema["sample_id"],
                "mode": args.mode,
                "provider": provider.name,
                "model_name": getattr(provider, "model", "mock"),
                "attempt": log.get("attempt"),
                "status": log.get("status"),
                "validation_pass": log.get("validation_pass"),
                "validation_errors": log.get("validation_errors", ""),
                "raw_response_excerpt": log.get("raw_response_excerpt", ""),
                "final_status": final_status,
                "n_retries": n_retries,
                "fallback_reason": "|".join(last_errors) if final_status == "fallback" else "",
            })

        if provider.name != "mock" and i < len(rows) - 1:
            time.sleep(args.sleep)

        if (i + 1) % 100 == 0 or (i + 1) == len(rows):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  {i + 1}/{len(rows)}  ok={n_pass_first}  retry_ok={n_pass_after_retry}  "
                f"fallback={n_fallback}  ({rate:.1f} samples/sec)"
            )

    pd.DataFrame(captions).to_csv(out_caps_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(run_logs).to_csv(out_log_path, index=False, encoding="utf-8-sig")
    print(f"\nsaved -> {out_caps_path}")
    print(f"saved -> {out_log_path}")

    n = len(captions)
    summary = {
        "n": n,
        "pass_first": n_pass_first,
        "pass_after_retry": n_pass_after_retry,
        "fallback": n_fallback,
        "schema_faithfulness_rate": (n_pass_first + n_pass_after_retry) / n if n else 0,
        "fallback_rate": n_fallback / n if n else 0,
        "mean_n_retries": n_retries_total / n if n else 0,
        "model_name": getattr(provider, "model", "mock"),
    }
    print()
    print(f"Summary: n={summary['n']}  pass={summary['pass_first']}+{summary['pass_after_retry']}  "
          f"fallback={summary['fallback']}  schema_faithfulness_rate={summary['schema_faithfulness_rate']:.4f}  "
          f"mean_n_retries={summary['mean_n_retries']:.4f}")

    if args.mode == "golden":
        sum_csv = OUTPUT_REPORT_DIR / "step7_v3_golden_validation_summary.csv"
    else:
        sum_csv = OUTPUT_REPORT_DIR / "step7_v3_caption_generation_summary.csv"
        md = build_markdown_report(args, summary, captions)
        (OUTPUT_REPORT_DIR / "step7_v3_caption_generation_results.md").write_text(md, encoding="utf-8")
        print(f"saved -> {OUTPUT_REPORT_DIR / 'step7_v3_caption_generation_results.md'}")

    rows_sum = []
    for k, v in summary.items():
        rows_sum.append({"metric": k, "value": v})
    pd.DataFrame(rows_sum).to_csv(sum_csv, index=False, encoding="utf-8-sig")
    print(f"saved -> {sum_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
