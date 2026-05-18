"""Build Step 6_v3 golden test cases by transforming Step 6_v2 golden cases.

본 스크립트는 *일회성* helper 다. Step 6_v2 golden cases 의 input_schema 를
그대로 보존하면서, example_valid_output 을 v3 6필드 schema (used_pool_entries
추가) + v3 pool 어휘로 재생성한다.

example_invalid_output 은 v3 정책 (책임/처방/모델내부 금지 유지, clinical
vocabulary 허용) 에 맞춰 *여전히 invalid* 한 패턴으로 갱신한다.

Reads:
    reports/step6_v2/step6_v2_golden_test_cases.json  (read-only)
    scripts/generate_step7_v3_captions.py             (read-only — pool 함수 재사용)

Writes:
    reports/step6_v3/step6_v3_golden_test_cases.json

Run:
    python scripts/build_step6_v3_golden_cases.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

V2_GOLDEN = PROJECT_ROOT / "reports" / "step6_v2" / "step6_v2_golden_test_cases.json"
V3_GOLDEN = PROJECT_ROOT / "reports" / "step6_v3" / "step6_v3_golden_test_cases.json"
V3_SCRIPT = PROJECT_ROOT / "scripts" / "generate_step7_v3_captions.py"


def _load_v3_module():
    spec = importlib.util.spec_from_file_location("v3", V3_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# v3 에서도 *여전히 invalid* 한 패턴들 (책임/처방/모델내부 + class_set narrowing
# + no_call contradiction + raw class label). 본 list 는 v2 의 violation_notes
# 를 v3 정책에 맞게 재작성하기 위한 reference 다.
V3_INVALID_PATTERNS = {
    "diagnostic_certainty": {
        "caption": "팔을 앞으로 뻗은 상태에서 척추 중립 유지에 대한 요구가 높아집니다. 발목 가동 범위 제한이 깊이에 영향을 주는 패턴이 명확히 잘못됐다고 진단된다.",
        "notes": "'명확히 잘못됐다', '진단된다' — v3 system prompt §9 위반 (의학적 진단 단정). clinical vocab 자체는 v3 에서 허용되나 단정 어조는 여전히 금지.",
    },
    "injury_prescription": {
        "caption": "팔을 교차한 자세에서 흉추 가동성이 스쿼트 깊이와 연관됩니다. 발목 가동 범위 제한이 보입니다. 부상 위험이 높으므로 즉시 교정해야 합니다.",
        "notes": "'부상 위험이 높', '교정해야' — v3 system prompt §9 위반 (부상/치료/처방 권고 금지).",
    },
    "attention_leak": {
        "caption": "팔을 앞으로 뻗은 자세에서 무게중심이 전방으로 이동하는 경향이 나타납니다. attention 이 bottom_transition phase 에 집중되고 posterior entropy 가 낮아 confident 분류 가능.",
        "notes": "'attention', 'bottom_transition', 'phase', 'posterior', 'entropy' — v3 system prompt §8 모델 내부 어휘 노출 금지 (v2 와 동일).",
    },
    "raw_class_label": {
        "caption": "손을 허리에 둔 상태에서 고관절과 체간의 협응 패턴이 두드러집니다. C1, C5, C6 중 C5 에 가장 잘 맞는 패턴입니다.",
        "notes": "raw class label (C1/C5/C6) caption 직접 노출 — v3 system prompt §7 위반. 단일 후보로 좁힘 (class_set narrowing) 도 동시 위반.",
    },
    "class_set_narrowing": {
        "caption": "팔을 교차한 자세에서 어깨 긴장이 동작 패턴에 영향을 줄 수 있습니다. 정상 패턴만 두드러집니다.",
        "notes": "class_set ≥ 2 인데 C1 정상만 단정 — v3 system prompt §4 위반 (다중 후보 모두 표현 의무). clinical vocab 허용과 무관하게 narrowing 자체가 invalid.",
    },
    "no_call_contradiction": {
        "caption": "팔을 앞으로 뻗은 조건에서 레버암이 가장 길게 형성됩니다. 발목 가동 범위 제한이 스쿼트 깊이에 영향을 주는 경향이 보입니다.",
        "notes": "no_call=true 인데 JOINT_POOL entry (J-C2-2) 가 caption 에 등장 — v3 system prompt §6 위반 (no_call 에서는 pool 미사용 + 보류 메시지만).",
    },
    "absolute_depth": {
        "caption": "팔을 앞으로 뻗은 상태에서 척추 중립 유지에 대한 요구가 높아집니다. 5cm 정확한 깊이가 부족합니다.",
        "notes": "'정확한 깊이가 부족' + 절대 cm 수치 — v3 system prompt §12 위반 (절대 squat depth 단정 금지, drift 로 인한 측정 불가).",
    },
    "missing_used_pool_entries": {
        "caption_fmt": "{posture_phrase}. {joint_phrase}. 비교적 일관되게 나타납니다.",
        "notes": "JSON 출력에서 used_pool_entries 필드 누락 — v3 system prompt 출력 schema 위반 (5필드 → 6필드 강제).",
    },
}


def _adapt_invalid_example(case_id: str, case: dict, mock_payload: dict, v3_mod) -> dict:
    """v2 invalid example 을 v3 에서도 invalid 한 패턴으로 갱신."""
    cs = case["input_schema"].get("class_set", [])
    no_call = case["input_schema"].get("no_call", False)

    # case_id 별 우선 분류
    if "attention" in case_id:
        p = V3_INVALID_PATTERNS["attention_leak"]
    elif "no_call" in case_id:
        p = V3_INVALID_PATTERNS["no_call_contradiction"]
    elif "class_set_narrowing" in case_id or (len(cs) >= 2 and "narrowing" in case.get("forbidden_properties", []) or any("narrowing" in fp for fp in case.get("forbidden_properties", []))):
        p = V3_INVALID_PATTERNS["class_set_narrowing"]
    elif "knee_valgus" in case_id or "posterior" in case_id:
        # v2 의 valgus / posterior 단정 demo → v3 에서는 책임/처방으로 대체
        p = V3_INVALID_PATTERNS["diagnostic_certainty"]
    elif len(cs) >= 2:
        p = V3_INVALID_PATTERNS["class_set_narrowing"]
    elif no_call:
        p = V3_INVALID_PATTERNS["no_call_contradiction"]
    else:
        p = V3_INVALID_PATTERNS["injury_prescription"]

    return {
        "caption_ko": p["caption"],
        "violation_notes": p["notes"],
    }


def _build_v3_valid_output(case: dict, v3_mod) -> tuple[dict, dict]:
    """v3 rule-based mock 의 결과를 example_valid_output 으로 사용."""
    schema = dict(case["input_schema"])
    schema["participant_id"] = schema["sample_id"].split("_")[0]
    payload = v3_mod.build_user_payload(schema)
    provider = v3_mod.MockProvider({})  # empty golden → rule-based
    raw = provider.call("", payload)
    data = json.loads(raw)
    return data, payload


def main() -> int:
    v3_mod = _load_v3_module()

    with open(V2_GOLDEN, encoding="utf-8") as f:
        v2_cases = json.load(f)

    v3_cases = []
    for case in v2_cases:
        valid_output, _payload = _build_v3_valid_output(case, v3_mod)
        # validate the generated output as a sanity check
        ok, errors = v3_mod.validate_caption(valid_output, case["input_schema"])
        if not ok:
            print(f"WARN: {case['case_id']} rule-based output fails v3 validation: {errors}")

        # forbidden_properties: v2 의 좌우/valgus/posterior 관련 항목 제거
        v2_forb = case.get("forbidden_properties", [])
        v3_forb = [
            p for p in v2_forb
            if p not in (
                "direct_knee_valgus_claim",
                "posterior_tilting_claim",
                "left_right_direction_claim",
            )
        ]
        # v3 신규 forbidden: used_pool_entries 미포함
        if "missing_used_pool_entries" not in v3_forb:
            v3_forb.append("missing_used_pool_entries")

        # expected_valid_properties: 같은 식으로 v3 신규 항목 추가
        v2_exp = case.get("expected_valid_properties", [])
        v3_exp = [
            p for p in v2_exp
            if p not in ("no_left_right_direction",)  # v3 는 방향 허용
        ]
        if "used_pool_entries_present" not in v3_exp:
            v3_exp.append("used_pool_entries_present")
        if "uses_pool_based_phrasing" not in v3_exp:
            v3_exp.append("uses_pool_based_phrasing")

        v3_case = {
            "case_id": case["case_id"],
            "purpose": case["purpose"],
            "input_schema": case["input_schema"],
            "expected_valid_properties": v3_exp,
            "forbidden_properties": v3_forb,
            "example_valid_output": valid_output,
            "example_invalid_output": _adapt_invalid_example(case["case_id"], case, None, v3_mod),
        }
        v3_cases.append(v3_case)

    V3_GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    with open(V3_GOLDEN, "w", encoding="utf-8") as f:
        json.dump(v3_cases, f, ensure_ascii=False, indent=2)

    print(f"written {len(v3_cases)} v3 golden cases to {V3_GOLDEN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
