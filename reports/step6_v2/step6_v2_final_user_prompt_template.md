# Step 6_v2 — Final User Prompt Template

본 문서는 Step 7_v2 caption generation prototype에서 **sample마다 사용할 user prompt template**을 고정한다. 본 단계는 LLM을 호출하지 않는다.

선행: `step6_v2_final_system_prompt.md` (system prompt commit), `step5_v2_caption_vocabulary.md` (어휘 후보).

---

## 1. user prompt 본문 — JSON 단일 객체

매 LLM 호출마다 다음 JSON 객체 *하나*를 user role 메시지로 전달한다. JSON 외 자유 텍스트 / 설명 / markdown 첨부 없음.

```json
{
  "task": "schema_to_korean_caption",
  "schema": {
    "sample_id": "<string>",
    "posture_canonical": "SA|CA|HW",
    "class_set": ["<C1..C6>", "..."],
    "ambiguity_group": "confident_C2|within_group_c1_c5_c6|pair_c3_c4|pair_plus_c2_absorption|no_call|uncategorized",
    "uncertainty_flags": ["<flag>", "..."],
    "caption_confidence_level": "confident|hedged|low|no_call",
    "no_call": false,
    "no_call_reason": ""
  },
  "posture_vocabulary": {
    "SA": "팔을 앞으로 둔 자세",
    "CA": "팔을 교차한 자세",
    "HW": "손을 허리에 둔 자세"
  },
  "class_vocabulary": {
    "C1": "정상 패턴에 가까운 신호",
    "C2": "깊이 부족 계열로 해석 가능한 신호",
    "C3": "깊이 부족과 복합 오류 후보가 함께 나타나는 신호",
    "C4": "무릎 관련 오류 후보가 섞인 신호",
    "C5": "무릎 관련 오류 후보가 섞인 신호",
    "C6": "무릎 관련 오류 후보가 섞인 신호"
  },
  "confidence_phrase_pool": {
    "confident": ["비교적 일관되게 나타납니다", "상대적으로 안정적인 신호입니다"],
    "hedged": ["~에 가까운 경향이 있습니다", "~ 가능성이 함께 나타납니다", "단일 유형으로 좁히기 어렵습니다", "경계 신호로 보입니다"],
    "low": ["불확실성이 큽니다", "후보군 수준으로만 해석하는 것이 적절합니다"],
    "no_call": ["현재 손목 센서 신호만으로는 안정적인 설명을 제공하기 어렵습니다"]
  },
  "uncertainty_phrase_pool": {
    "confident_C2": "깊이 부족 계열에 가까운 패턴",
    "within_group_c1_c5_c6": "정상 패턴과 무릎 관련 오류 후보 패턴이 함께 나타나는 경계 신호",
    "pair_c3_c4": "복합 오류 후보 두 가지가 함께 나타나는 경계 신호",
    "pair_plus_c2_absorption": "깊이 부족 계열 신호와 복합 오류 후보가 함께 나타남",
    "no_call": "동작 기준점 신뢰도가 낮음 또는 신호 강도가 부족함",
    "uncategorized": "단일 유형으로 좁히기 어려움",
    "anchor_unreliable_suffix": "(동작 기준점이 불안정하여 설명 신뢰도가 낮습니다)"
  },
  "limitation_phrase_pool": [
    "손목 센서 기준의 추정",
    "손목 센서는 무릎 각도나 골반 움직임을 직접 측정하지 않습니다",
    "정확한 자세 평가는 영상 또는 전문가 평가와 함께 해석해 주십시오"
  ]
}
```

---

## 2. 채워 넣는 규칙

LLM에 전달되는 시점에는 위 template의 placeholder가 sample 값으로 채워져 있어야 한다.

| placeholder | 출처 | 비고 |
|---|---|---|
| `schema.sample_id` | schema CSV `sample_id` | LLM 출력에는 노출하지 않음 (trace) |
| `schema.posture_canonical` | schema CSV `posture_canonical` | enum 검사 |
| `schema.class_set` | schema CSV `class_set` (JSON list 그대로) | whitelist 검사 |
| `schema.ambiguity_group` | schema CSV `ambiguity_group` | enum 검사 |
| `schema.uncertainty_flags` | schema CSV `uncertainty_flags` (JSON list) | closed vocab |
| `schema.caption_confidence_level` | schema CSV `caption_confidence_level` | enum 검사 |
| `schema.no_call` | schema CSV `no_call` | bool |
| `schema.no_call_reason` | schema CSV `no_call_reason` | no_call=false면 빈 문자열 |

`posture_vocabulary`, `class_vocabulary`, `confidence_phrase_pool`, `uncertainty_phrase_pool`, `limitation_phrase_pool` 다섯 항목은 sample마다 *동일한 고정 본문*을 첨부한다. 매 호출마다 동일하므로 prompt cache에 적합하다.

---

## 3. 명시적으로 *포함하지 않는* 필드

다음 필드는 schema CSV에 존재하지만 user prompt에 포함하지 **않는다** (`step5_v2_schema_grounded_caption_policy.md` §3.2와 일관).

- `top1_class`, `top2_class` (raw class label은 prompt에 직접 노출 안 함)
- `top1_prob_calibrated`, `top2_prob_calibrated`
- `top1_top2_margin_calibrated`
- `predictive_entropy_calibrated`
- `class_set_posterior_mass`
- `temperature`
- attention 관련 필드 (entropy, peak phase, phase mass, anchor distance) — 별도 산출물에 있으며 prompt에는 첨부 안 함
- `anchor_reliability`, `anchor_type` — schema 분기에 *이미* 사용되어 `uncertainty_flags`의 `anchor_unreliable`로 표현됨; prompt에 raw float을 직접 넘기지 않음

---

## 4. class_vocabulary 통일 정책 보강

`class_vocabulary`에서 **C4 / C5 / C6는 모두 "무릎 관련 오류 후보가 섞인 신호"** 로 동일 표현이다. 이는 `step5_v2_caption_vocabulary.md` §1, §1.1, §1.2의 정책에 따른 것이다.

- C4 / C5 / C6의 원 논문 정의는 각각 left / right / both knee valgus이지만, wrist IMU는 좌우 방향을 직접 측정하지 못하므로 사용자 caption에서는 방향 단정을 제거한다.
- C3는 원 논문에서 "insufficient depth with posterior tilting and knee valgus"이지만, 본 어휘는 *posterior tilting* 과 *knee valgus*를 직접 표현하지 않고 **"깊이 부족과 복합 오류 후보가 함께 나타나는 신호"** 로 의도적 축약한다 (`step5_v2_schema_grounded_caption_policy.md` §6.4.1, `step5_v2_forbidden_expressions.md` §2.1 일관).

LLM은 이 mapping을 *그대로* 사용하며, 새로운 한국어 표현을 만들어 내지 않는다.

---

## 5. 사용 시 주의

- LLM 호출의 `temperature`는 본 caption layer에서 낮게(예: 0.0 ~ 0.3) 두는 것을 권장한다 (Step 7_v2에서 commit). 본 layer는 창의적 생성이 아니라 *결정론적 변환*에 가깝기 때문이다.
- 응답 형식 강제(`response_format` JSON 모드)가 LLM에서 지원되면 활성화하고, 미지원이면 system prompt §1 (JSON-only 출력) 규칙으로만 제어한다 — Step 7_v2의 retry/fallback이 이 결정을 보완한다.
- 동일한 input schema 두 번 호출 시 산출물이 *대체로* 일치해야 한다 (caption은 결정론적 변환). 큰 분산이 보이면 prompt 또는 모델 선택 문제로 본다.

---

*본 문서는 user prompt template commit이며, LLM을 호출하지 않는다.*
