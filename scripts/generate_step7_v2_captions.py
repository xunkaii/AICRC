"""Step 7_v2 — Schema-grounded caption generation prototype.

Reads (read-only):
    reports/step6_v2/step6_v2_final_system_prompt.md
    reports/step6_v2/step6_v2_final_user_prompt_template.md   (reference only)
    reports/step6_v2/step6_v2_golden_test_cases.json
    data/step4r/4rb_attention/step4r_bigru_attention_schema_outputs_calibrated.csv

Writes:
    [golden mode]
        data/step7_v2/step7_v2_golden_captions.csv
        data/step7_v2/step7_v2_golden_run_log.csv
    [full mode]
        data/step7_v2/step7_v2_captions.csv
        data/step7_v2/step7_v2_run_log.csv
        reports/step7_v2/step7_v2_caption_generation_summary.csv
        reports/step7_v2/step7_v2_caption_generation_results.md

Provider abstraction:
    --provider mock        rule-based / golden-example output. Dev only.
    --provider anthropic   ANTHROPIC_API_KEY env var required.
    --provider openai      OPENAI_API_KEY env var required.

Run:
    python scripts/generate_step7_v2_captions.py --mode golden --provider mock
    python scripts/generate_step7_v2_captions.py --mode golden --provider anthropic
    python scripts/generate_step7_v2_captions.py --mode full   --provider anthropic
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import argparse
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

# ---------- Paths ----------
SYS_PROMPT_MD = PROJECT_ROOT / "reports" / "step6_v2" / "step6_v2_final_system_prompt.md"
GOLDEN_JSON = PROJECT_ROOT / "reports" / "step6_v2" / "step6_v2_golden_test_cases.json"
SCHEMA_CSV = PROJECT_ROOT / "data" / "step4r" / "4rb_attention" / "step4r_bigru_attention_schema_outputs_calibrated.csv"

OUTPUT_DATA_DIR = PROJECT_ROOT / "data" / "step7_v2"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "reports" / "step7_v2"

# ---------- Constants from Step 6_v2 commits ----------
CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
POSTURES = ["SA", "CA", "HW"]
LEVELS = ["confident", "hedged", "low", "no_call"]

OUTPUT_REQUIRED_FIELDS = {
    "caption_ko",
    "confidence_phrase",
    "uncertainty_phrase",
    "limitation_phrase",
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

POSTURE_VOCAB = {
    "SA": "팔을 앞으로 둔 자세",
    "CA": "팔을 교차한 자세",
    "HW": "손을 허리에 둔 자세",
}
CLASS_VOCAB = {
    "C1": "정상 패턴에 가까운 신호",
    "C2": "깊이 부족 계열로 해석 가능한 신호",
    "C3": "깊이 부족과 복합 오류 후보가 함께 나타나는 신호",
    "C4": "무릎 관련 오류 후보가 섞인 신호",
    "C5": "무릎 관련 오류 후보가 섞인 신호",
    "C6": "무릎 관련 오류 후보가 섞인 신호",
}
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
UNCERTAINTY_PHRASE_POOL = {
    "confident_C2": "깊이 부족 계열에 가까운 패턴",
    "within_group_c1_c5_c6": "정상 패턴과 무릎 관련 오류 후보 패턴이 함께 나타나는 경계 신호",
    "pair_c3_c4": "복합 오류 후보 두 가지가 함께 나타나는 경계 신호",
    "pair_plus_c2_absorption": "깊이 부족 계열 신호와 복합 오류 후보가 함께 나타남",
    "no_call": "동작 기준점 신뢰도가 낮음 또는 신호 강도가 부족함",
    "uncategorized": "단일 유형으로 좁히기 어려움",
    "anchor_unreliable_suffix": "(동작 기준점이 불안정하여 설명 신뢰도가 낮습니다)",
}
LIMITATION_PHRASE_POOL = [
    "손목 센서 기준의 추정",
    "손목 센서는 무릎 각도나 골반 움직임을 직접 측정하지 않습니다",
    "정확한 자세 평가는 영상 또는 전문가 평가와 함께 해석해 주십시오",
]

# Forbidden tokens (from step5_v2_forbidden_expressions.md §1 #1~#18)
FORBIDDEN_EXACT = [
    "무릎이 안쪽으로 들어갔다",
    "knee valgus",
    "골반이 후방 경사",
    "무릎 정렬이 틀어",
    "관절각이 감소",
    "정확한 깊이가 부족",
    "부상 위험이 높",
    "치료가 필요",
    "진단된다",
    "확실히",
    "확실하다",
    "명확히 잘못",
    "전문가가 확인한 것과 같",
    "후방 경사",
    "posterior tilting",
    "골반이 뒤로 말렸",
    "골반이 뒤로 빠졌",
    "무릎이 모인다",
    "무릎이 안으로 무너",
    # "관찰됩니다" 계열 — wrist IMU는 직접 *관찰*하지 않으므로 caption에 부적절.
    # 대체: "나타납니다" / "보입니다" / "해석 가능한 신호입니다".
    "관찰됩니다",
    "관찰된다",
    "관찰되었다",
    "관찰됨",
]
FORBIDDEN_DIRECTION_TOKENS = ["한쪽", "반대쪽", "양측", "왼쪽", "오른쪽"]
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
BIOMECH_TOKENS = ["골반", "관절각", "정렬"]

# Negation-context fragments that LLM commonly produces when paraphrasing
# our LIMITATION_PHRASE_POOL. If a BIOMECH token appears *inside* one of these
# fragments, it is in negation context (sensor 한계 명시) and must NOT be
# flagged as a biomech claim. We strip these fragments before token matching.
ALLOWED_BIOMECH_NEGATION_FRAGMENTS = [
    "무릎 각도나 골반 움직임을 직접 측정하지 않",
    "무릎 각도나 골반 움직임을 직접 측정하지",
    "무릎 각도나 골반",
    "골반 움직임을 직접 측정",
    "골반 움직임",
    "관절각을 직접 측정하지 않",
    "관절각을 직접 측정",
    "정렬을 직접 측정하지 않",
    "정렬을 직접 측정",
]

FALLBACK_OUTPUT = {
    "caption_ko": "현재 손목 센서 신호만으로는 안정적인 설명을 제공하기 어렵습니다.",
    "confidence_phrase": "안정적인 설명을 제공하기 어렵습니다",
    "uncertainty_phrase": "",
    "limitation_phrase": "",
    "used_schema_fields": ["caption_confidence_level"],
}


def build_fallback(schema: dict) -> dict:
    """schema-aware fallback. caption_ko가 schema의 posture_vocabulary로 시작하도록.
    no_call 메시지여도 posture phrase 강제 정책(`step6_v2_final_system_prompt.md`
    §5)을 따른다."""
    posture = schema.get("posture_canonical", "") if schema else ""
    pp = POSTURE_VOCAB.get(posture, "기록된 자세")
    used = ["caption_confidence_level"]
    if posture in POSTURE_VOCAB:
        used.insert(0, "posture_canonical")
    return {
        "caption_ko": (
            f"{pp}에서 측정된 신호는 현재 손목 센서만으로는 안정적인 설명을 "
            "제공하기 어렵습니다."
        ),
        "confidence_phrase": "안정적인 설명을 제공하기 어렵습니다",
        "uncertainty_phrase": "",
        "limitation_phrase": "",
        "used_schema_fields": used,
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


# ---------- User payload builder ----------

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
    return {
        "task": "schema_to_korean_caption",
        "schema": {
            "sample_id": str(schema_row["sample_id"]),
            "posture_canonical": str(schema_row["posture_canonical"]),
            "class_set": list(cs),
            "ambiguity_group": str(schema_row["ambiguity_group"]),
            "uncertainty_flags": list(uf),
            "caption_confidence_level": str(schema_row["caption_confidence_level"]),
            "no_call": bool(schema_row["no_call"]),
            "no_call_reason": str(nr),
        },
        "posture_vocabulary": POSTURE_VOCAB,
        "class_vocabulary": CLASS_VOCAB,
        "confidence_phrase_pool": CONFIDENCE_PHRASE_POOL,
        "uncertainty_phrase_pool": UNCERTAINTY_PHRASE_POOL,
        "limitation_phrase_pool": LIMITATION_PHRASE_POOL,
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


# ---------- Validator (shared with validate_step7_v2_captions.py) ----------

def validate_caption(output: dict, schema: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []

    # 2. required fields
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
    usf = output.get("used_schema_fields") or []
    full_text = " ".join([cap, cp, ucp, lp])

    if not cap:
        errors.append("caption_ko_empty")

    # 5. raw class label
    if RAW_CLASS_LABEL_RE.search(cap):
        errors.append("raw_class_label_in_caption")

    # 6. forbidden exact match
    for tok in FORBIDDEN_EXACT:
        if tok in full_text:
            errors.append(f"forbidden_token:{tok}")
            break

    # 7. left/right direction tokens
    for tok in FORBIDDEN_DIRECTION_TOKENS:
        if tok in cap:
            errors.append(f"direction_token:{tok}")
            break

    # 9. attention/model-internal leak (skip "확률" because vocab uses 후보 etc; still check 엔트로피)
    for tok in ATTENTION_LEAK_TOKENS:
        if tok in full_text:
            errors.append(f"attention_leak:{tok}")
            break

    # 6. no_call contradiction
    no_call = bool(schema.get("no_call", False))
    if no_call:
        for c, vocab in CLASS_VOCAB.items():
            if vocab in cap:
                errors.append(f"no_call_contradiction:{c}_vocab")
                break

    # 7. class_set narrowing
    cs = schema.get("class_set", [])
    if isinstance(cs, str):
        try:
            cs = json.loads(cs)
        except Exception:
            cs = []
    ag = schema.get("ambiguity_group", "")
    if not no_call and isinstance(cs, list) and len(cs) >= 2:
        if ag == "within_group_c1_c5_c6":
            ok = ("정상" in cap) and ("무릎 관련 오류 후보" in cap)
            if not ok:
                errors.append("class_set_narrowing_within_group")
        elif ag == "pair_c3_c4":
            # C3/C4 pair는 다음 표현 중 하나라도 등장하면 두 후보 모두 표현된 것으로 인정.
            ok = (
                ("복합 오류 후보 두 가지" in cap)
                or ("복합 오류 후보가 함께" in cap)
                or (("복합 오류 후보" in cap) and ("경계" in cap))
            )
            if not ok:
                errors.append("class_set_narrowing_pair")
        elif ag == "pair_plus_c2_absorption":
            ok = ("깊이 부족 계열" in cap) and ("복합 오류 후보" in cap)
            if not ok:
                errors.append("class_set_narrowing_pair_plus_c2")
        else:
            if ("후보" not in cap) and ("경계" not in cap):
                errors.append("class_set_narrowing_generic")

    # 8. unsupported biomechanical claim
    # LLM이 우리 vocabulary pool의 limitation phrase를 약간 어미 변형해서 사용
    # 하더라도(예: "...않습니다" → "...않으므로") negation 문맥의 BIOMECH 토큰은
    # exempt해야 한다. partial fragment까지 strip해서 robust하게 인식한다.
    # 단정형(예: "골반이 뒤로 빠졌다")은 FORBIDDEN_EXACT에서 별도로 잡힌다.
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

    # 10. posture phrase consistency
    # 정책 강화 (step6_v2_final_system_prompt.md §5): no_call=true인 경우에도
    # caption 첫 문장은 posture_vocabulary의 자세 표현으로 시작해야 한다.
    # 따라서 no_call 여부와 무관하게 posture phrase 누락은 fail.
    posture = schema.get("posture_canonical", "")
    expected = POSTURE_VOCAB.get(posture, "")
    if expected and expected not in cap:
        errors.append(f"posture_phrase_missing:{posture}")
    for p, ph in POSTURE_VOCAB.items():
        if p != posture and ph in cap:
            errors.append(f"wrong_posture_phrase:{p}")
            break

    # 11. confidence-level mismatch
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

    # 12. used_schema_fields whitelist
    usf_set = set(usf) if isinstance(usf, list) else set()
    if not usf_set:
        errors.append("used_schema_fields_empty")
    else:
        invalid_fields = usf_set - SCHEMA_FIELDS_WHITELIST
        if invalid_fields:
            errors.append("used_schema_fields_invalid:" + ",".join(sorted(invalid_fields)))
        if no_call and "no_call" not in usf_set:
            errors.append("used_schema_fields_missing_no_call")

    # 13. limitation phrase policy
    if level != "no_call":
        if not lp and "손목 센서" not in cap:
            errors.append("limitation_phrase_missing")
        if len(lp) > 200:
            errors.append("limitation_phrase_too_long")

    return len(errors) == 0, errors


# ---------- Providers ----------

class MockProvider:
    """Rule-based + golden-example mock. Dev only."""

    name = "mock"
    model = "mock"

    def __init__(self, golden_outputs_by_sample: dict | None = None):
        self.golden_outputs = golden_outputs_by_sample or {}

    def call(self, system_prompt: str, user_payload: dict, **kwargs) -> str:
        sid = user_payload["schema"]["sample_id"]
        if sid in self.golden_outputs:
            return json.dumps(self.golden_outputs[sid], ensure_ascii=False)
        return json.dumps(self._build_rule_based(user_payload["schema"]), ensure_ascii=False)

    def _build_rule_based(self, schema: dict) -> dict:
        posture = schema["posture_canonical"]
        class_set = schema.get("class_set", [])
        ag = schema.get("ambiguity_group", "")
        flags = schema.get("uncertainty_flags", [])
        level = schema.get("caption_confidence_level", "")
        no_call = bool(schema.get("no_call", False))
        anchor_unreliable = "anchor_unreliable" in flags

        posture_phrase = POSTURE_VOCAB.get(posture, "기록된 자세")

        if no_call:
            reason = schema.get("no_call_reason", "") or ""
            if "anchor" in reason:
                cap = (
                    f"{posture_phrase}에서 기록된 손목 IMU 신호는 동작 기준점 "
                    "신뢰도가 낮아, 현재 손목 센서 신호만으로는 안정적인 설명을 "
                    "제공하기 어렵습니다."
                )
                ucp = "동작 기준점 신뢰도가 낮음"
            else:
                cap = (
                    f"{posture_phrase}에서 측정된 손목 IMU 신호는 신호 강도가 "
                    "부족하여, 현재 손목 센서 신호만으로는 안정적인 설명을 "
                    "제공하기 어렵습니다."
                )
                ucp = "신호 강도가 부족함"
            return {
                "caption_ko": cap,
                "confidence_phrase": "안정적인 설명을 제공하기 어렵습니다",
                "uncertainty_phrase": ucp,
                "limitation_phrase": "",
                "used_schema_fields": [
                    "posture_canonical",
                    "no_call",
                    "no_call_reason",
                    "caption_confidence_level",
                ],
            }

        ucp_base = UNCERTAINTY_PHRASE_POOL.get(ag, UNCERTAINTY_PHRASE_POOL["uncategorized"])
        if anchor_unreliable:
            ucp = f"{ucp_base} {UNCERTAINTY_PHRASE_POOL['anchor_unreliable_suffix']}"
        else:
            ucp = ucp_base

        if level == "confident":
            cp = "비교적 일관되게 나타납니다"
        elif level == "hedged":
            cp = "단일 유형으로 좁히기 어렵습니다"
        elif level == "low":
            cp = "후보군 수준으로만 해석하는 것이 적절합니다"
        else:
            cp = "안정적인 설명을 제공하기 어렵습니다"

        if ag == "confident_C2":
            if anchor_unreliable:
                cap = (
                    f"{posture_phrase}에서 측정된 손목 IMU 신호는 깊이 부족 계열로 "
                    "해석 가능한 패턴에 가까운 경향이 있으나, 동작 기준점이 불안정하여 "
                    "설명 신뢰도가 낮습니다. 손목 센서 기준의 추정입니다."
                )
                cp = "패턴에 가까운 경향이 있습니다"
            elif level == "confident":
                cap = (
                    f"{posture_phrase}에서 기록된 손목 IMU 신호는 깊이 부족 계열로 "
                    "해석 가능한 패턴이 비교적 일관되게 나타납니다. 손목 센서 기준의 "
                    "추정이며, 정확한 자세 평가는 영상 또는 전문가 평가와 함께 해석해 "
                    "주십시오."
                )
            else:
                cap = (
                    f"{posture_phrase}에서 측정된 손목 IMU 신호는 깊이 부족 계열로 "
                    f"해석 가능한 패턴이 함께 나타납니다. {cp}. 손목 센서 기준의 "
                    "추정입니다."
                )
        elif ag == "within_group_c1_c5_c6":
            if level == "low":
                cap = (
                    f"{posture_phrase}에서, 동작 기준점이 불안정하여 설명 신뢰도가 "
                    "낮은 상태입니다. 정상 패턴과 무릎 관련 오류 후보 패턴이 함께 "
                    f"나타날 가능성이 있으나, {cp}."
                )
            else:
                cap = (
                    f"{posture_phrase}에서 측정된 손목 IMU 패턴은, 정상 패턴과 무릎 "
                    "관련 오류 후보 패턴이 함께 나타나는 경계 신호로 보입니다. "
                    f"{cp}. 손목 센서 기준의 추정입니다."
                )
        elif ag == "pair_c3_c4":
            cap = (
                f"{posture_phrase}에서 측정된 손목 IMU 패턴은 복합 오류 후보 두 가지가 "
                f"함께 나타나는 경계 신호로 보입니다. {cp}. 손목 센서 기준의 추정입니다."
            )
        elif ag == "pair_plus_c2_absorption":
            if level == "low":
                cap = (
                    f"{posture_phrase}에서, 동작 기준점이 불안정하여 설명 신뢰도가 "
                    "낮은 상태입니다. 깊이 부족 계열 신호와 복합 오류 후보가 함께 "
                    f"나타날 가능성이 있으나, {cp}."
                )
            else:
                cap = (
                    f"{posture_phrase}에서 측정된 손목 IMU 패턴은 깊이 부족 계열 신호와 "
                    f"복합 오류 후보가 함께 나타날 가능성이 있습니다. {cp}. 손목 센서 "
                    "기준의 추정입니다."
                )
        else:
            cap = (
                f"{posture_phrase}에서 측정된 손목 IMU 신호는 단일 유형으로 좁히기 "
                f"어려운 신호입니다. {cp}. 손목 센서 기준의 추정입니다."
            )

        lp = "" if level == "no_call" else "손목 센서 기준의 추정"

        return {
            "caption_ko": cap,
            "confidence_phrase": cp,
            "uncertainty_phrase": ucp,
            "limitation_phrase": lp,
            "used_schema_fields": [
                "posture_canonical",
                "class_set",
                "ambiguity_group",
                "uncertainty_flags",
                "caption_confidence_level",
            ],
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
        # Model resolution priority: explicit arg > ANTHROPIC_MODEL env > default.
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
    # All retries failed -> schema-aware fallback (posture phrase 포함).
    return build_fallback(schema), logs, "fallback", max_retries, last_errors


# ---------- Markdown report builder ----------

def build_markdown_report(args, summary: dict, captions: list[dict]) -> str:
    L = []
    L.append("# Step 7_v2 — Caption Generation Prototype 결과")
    L.append("")
    L.append(f"- 생성 스크립트: `scripts/generate_step7_v2_captions.py`")
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
        "Step 7_v2는 Step 6_v2가 commit한 prompt / output schema / golden cases / validation "
        "checklist 위에서 schema-grounded Korean caption을 생성하는 prototype 단계이다. "
        "Step 8_v2 automatic schema-caption validation의 *입력*이 된다. human review는 main "
        "evaluation에 포함하지 않는다 (`reports/step4_research_reframing.md` §7)."
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
    biomech_count = int(df["validation_errors"].fillna("").str.contains("biomech_claim").sum())
    L.append("| 검사 | invalid count |")
    L.append("|---|---:|")
    L.append(f"| forbidden expression | {forb_count} |")
    L.append(f"| attention term leakage | {attn_count} |")
    L.append(f"| class_set narrowing | {narrow_count} |")
    L.append(f"| unsupported biomechanical claim | {biomech_count} |")
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
    L.append("## 10. Step 8_v2 automatic validation으로 넘기는 산출물")
    L.append("")
    L.append("- `data/step7_v2/step7_v2_captions.csv` (full mode 결과)")
    L.append("- `data/step7_v2/step7_v2_run_log.csv` (per-attempt log)")
    L.append("- `data/step7_v2/step7_v2_golden_captions.csv` (golden cases)")
    L.append("- `data/step7_v2/step7_v2_golden_run_log.csv`")
    L.append("- `reports/step7_v2/step7_v2_caption_generation_summary.csv`")
    L.append("- `reports/step7_v2/step7_v2_golden_validation_summary.csv`")
    L.append("- `reports/step7_v2/step7_v2_caption_generation_results.md` (본 보고서)")
    L.append("")
    L.append("---")
    L.append("")
    L.append("*본 보고서는 자동 생성된다. 기존 Step 1~6_v2 산출물은 수정되지 않는다.*")
    return "\n".join(L)


# ---------- Main ----------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["golden", "full"], required=True)
    parser.add_argument("--provider", choices=["mock", "anthropic", "openai"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Auto-load .env so users don't have to set env vars manually each session.
    n_loaded = _load_dotenv(PROJECT_ROOT / ".env")
    if n_loaded:
        print(f"loaded {n_loaded} env var(s) from .env")

    sys_prompt = load_system_prompt()
    print(f"loaded system prompt: {len(sys_prompt)} chars")

    with open(GOLDEN_JSON, encoding="utf-8") as f:
        golden = json.load(f)
    print(f"loaded {len(golden)} golden cases")

    # Build provider
    if args.provider == "mock":
        mock_outputs = {}
        for case in golden:
            sid = case["input_schema"]["sample_id"]
            if sid not in mock_outputs:
                mock_outputs[sid] = case["example_valid_output"]
        provider = MockProvider(mock_outputs)
        print(f"MockProvider with {len(mock_outputs)} preloaded golden examples")
    elif args.provider == "anthropic":
        provider = AnthropicProvider(model=args.model)
        print(f"AnthropicProvider model={provider.model}")
    elif args.provider == "openai":
        model = args.model or "gpt-4o-mini"
        provider = OpenAIProvider(model=model)
        print(f"OpenAIProvider model={model}")
    else:
        raise ValueError(f"unknown provider: {args.provider}")

    # Build input rows
    if args.mode == "golden":
        rows = []
        for case in golden:
            sch = dict(case["input_schema"])
            sch["case_id"] = case["case_id"]
            sch["participant_id"] = sch["sample_id"].split("_")[0]
            sch["split"] = "golden"
            rows.append(sch)
        out_caps_path = OUTPUT_DATA_DIR / "step7_v2_golden_captions.csv"
        out_log_path = OUTPUT_DATA_DIR / "step7_v2_golden_run_log.csv"
    else:
        df = pd.read_csv(SCHEMA_CSV, encoding="utf-8-sig")
        df["class_set"] = df["class_set"].apply(json.loads)
        df["uncertainty_flags"] = df["uncertainty_flags"].apply(json.loads)
        rows = df.to_dict("records")
        out_caps_path = OUTPUT_DATA_DIR / "step7_v2_captions.csv"
        out_log_path = OUTPUT_DATA_DIR / "step7_v2_run_log.csv"

    if args.limit:
        rows = rows[: args.limit]
    print(f"mode={args.mode}, n_rows={len(rows)}")

    # Generate loop
    captions: list[dict] = []
    run_logs: list[dict] = []
    n_pass_first = 0
    n_pass_after_retry = 0
    n_fallback = 0
    n_retries_total = 0

    t0 = time.time()
    for i, schema in enumerate(rows):
        # Normalize types
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

    # Save captions + logs
    pd.DataFrame(captions).to_csv(out_caps_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(run_logs).to_csv(out_log_path, index=False, encoding="utf-8-sig")
    print(f"\nsaved -> {out_caps_path}")
    print(f"saved -> {out_log_path}")

    # Summary
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

    # Mode-specific summary CSVs and report
    if args.mode == "golden":
        sum_csv = OUTPUT_REPORT_DIR / "step7_v2_golden_validation_summary.csv"
    else:
        sum_csv = OUTPUT_REPORT_DIR / "step7_v2_caption_generation_summary.csv"
        # markdown report (full mode only)
        md = build_markdown_report(args, summary, captions)
        (OUTPUT_REPORT_DIR / "step7_v2_caption_generation_results.md").write_text(md, encoding="utf-8")
        print(f"saved -> {OUTPUT_REPORT_DIR / 'step7_v2_caption_generation_results.md'}")

    rows_sum = []
    for k, v in summary.items():
        rows_sum.append({"metric": k, "value": v})
    pd.DataFrame(rows_sum).to_csv(sum_csv, index=False, encoding="utf-8-sig")
    print(f"saved -> {sum_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
