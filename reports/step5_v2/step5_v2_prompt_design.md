# Step 5_v2 — Prompt Design (초안)

본 문서는 Step 5_v2 schema-grounded caption layer의 **prompt 초안**과 **출력 JSON 형식**, 그리고 **valid / invalid 예시**를 정리한다. `step5_v2_schema_grounded_caption_policy.md` (정책), `step5_v2_caption_vocabulary.md` (어휘), `step5_v2_forbidden_expressions.md` (금지) 위에서 작성된다.

본 문서의 prompt와 예시는 **초안**이다. 최종 wording은 Step 6_v2 (prompt finalization)에서 commit된다. 본 단계에서는 caption을 만들지 않고, 어떤 LLM도 호출하지 않는다.

---

## 1. System prompt 초안

```
당신은 손목 IMU(가속도계 + 자이로스코프) 기반 스쿼트 분류 시스템의
한국어 caption 표현층입니다.

역할:
- 입력으로 들어오는 schema(JSON)에 적힌 사실만을 한국어 문장으로
  풀어 냅니다.
- 새로운 클래스를 추론하거나 schema에 없는 정보를 추가하지 않습니다.
- 어떠한 의학적 진단, 부상 위험 단정, 치료 권고, 좌우 방향 단정도
  하지 않습니다.

엄격한 제약:
1. 손목 IMU는 무릎 각도, hip-knee-ankle 정렬, knee valgus, 골반 경사,
   절대 squat depth, 관절각, 좌/우 무릎의 방향성을 직접 측정하지
   않습니다. 이 양들을 측정한 것처럼 표현하지 마십시오.
2. 입력 schema의 `class_set` 길이가 2 이상이면 모든 후보를 함께
   표현하고, 어느 하나로 좁히지 마십시오.
3. 입력 schema의 `caption_confidence_level`을 어조에 그대로
   반영하십시오:
     - confident: 비교적 일관되게 나타남
     - hedged: 가능성이 함께 나타남 / 단정하기 어려움
     - low: 불확실성이 큼 / 후보군 수준
     - no_call: 안정적인 설명을 제공하기 어려움
4. `no_call`이 true이면 어떠한 클래스 단정도 하지 마십시오. 보류
   메시지만 한 줄로 제공하십시오.
5. attention, phase, anchor, predictive entropy, top1 probability
   같은 모델 내부 용어 또는 수치를 caption 본문에 노출하지
   마십시오.
6. 사용 가능한 어휘는 별도 vocabulary 표를 따릅니다 (system prompt에
   포함되거나 user prompt에 첨부됩니다). 그 외 표현은 만들어
   내지 마십시오.

출력은 반드시 JSON 객체로만 반환하십시오 (자유 텍스트 금지).
출력 schema:
{
  "caption_ko": str,
  "confidence_phrase": str,
  "uncertainty_phrase": str,
  "limitation_phrase": str,
  "used_schema_fields": [str, ...]
}
```

---

## 2. User prompt template 초안

각 rep마다 다음 JSON을 user prompt 본문으로 전달한다.

```json
{
  "posture_canonical": "SA",
  "class_set": ["C1", "C5", "C6"],
  "ambiguity_group": "within_group_c1_c5_c6",
  "uncertainty_flags": ["within_group_ambiguity_c1_c5_c6"],
  "caption_confidence_level": "hedged",
  "no_call": false,
  "no_call_reason": "",
  "limitation_policy": [
    "손목 센서는 무릎 각도·골반·정렬을 직접 측정하지 않음",
    "사용자 caption에서는 좌우 방향 단정 금지",
    "한 caption에 limitation_phrase는 1개 이내"
  ]
}
```

추가 컨텍스트가 필요한 경우 (vocabulary 표를 system 또는 user prompt에 첨부), `step5_v2_caption_vocabulary.md`의 §1 ~ §5를 그대로 prepend한다. 본 단계에서는 *어떤 컨텍스트를 어디에 넣을지*만 정의하고, 최종 형태는 Step 6_v2에서 commit.

### 2.1 user prompt에 *포함하지 않는* 필드

다음 필드는 prompt에 *직접* 넣지 않는다 (정책 §3.2와 일치):

- `top1_prob_calibrated`, `top2_prob_calibrated`, `top1_top2_margin_calibrated`, `predictive_entropy_calibrated`, `class_set_posterior_mass`
- attention 관련 필드 (entropy, peak phase, phase mass, anchor distance)
- `temperature` 자체 값 (메타데이터로만 보존)

이 필드들은 schema 분기 형성에 *이미* 사용되었고, caption layer는 그 결과(class_set, ambiguity_group, confidence level)만 본다.

---

## 3. 출력 JSON 형식

```json
{
  "caption_ko": "팔을 앞으로 둔 자세에서 기록된 손목 IMU 신호는, 정상 패턴과 무릎 관련 오류 후보 패턴이 함께 나타나는 경계 신호로 보입니다. 단일 유형으로 좁히기 어렵습니다. 손목 센서 기준의 추정입니다.",
  "confidence_phrase": "단일 유형으로 좁히기 어렵습니다",
  "uncertainty_phrase": "정상 패턴과 무릎 관련 오류 후보 패턴이 함께 나타나는 경계 신호",
  "limitation_phrase": "손목 센서 기준의 추정",
  "used_schema_fields": [
    "posture_canonical",
    "class_set",
    "ambiguity_group",
    "uncertainty_flags",
    "caption_confidence_level"
  ]
}
```

`used_schema_fields`는 LLM이 *어떤 schema 필드를 참조했는지를 자체 보고*하도록 한다. Step 8_v2 automatic validation에서 누락 점검 (예: `caption_confidence_level`을 무시하고 단정형으로 작성한 경우)에 활용.

---

## 4. Confidence별 valid 예시 (1개씩)

### 4.1 `confident`

**input schema (요약):**
```json
{
  "posture_canonical": "SA",
  "class_set": ["C2"],
  "ambiguity_group": "confident_C2",
  "uncertainty_flags": ["confident_C2"],
  "caption_confidence_level": "confident",
  "no_call": false,
  "no_call_reason": ""
}
```

**output (valid):**
```json
{
  "caption_ko": "팔을 앞으로 둔 자세에서 기록된 손목 IMU 신호는 깊이 부족 계열로 해석 가능한 패턴이 비교적 일관되게 나타납니다. 손목 센서 기준의 추정이며, 정확한 자세 평가는 영상 또는 전문가 평가와 함께 해석해 주십시오.",
  "confidence_phrase": "비교적 일관되게 나타납니다",
  "uncertainty_phrase": "",
  "limitation_phrase": "손목 센서 기준의 추정이며, 정확한 자세 평가는 영상 또는 전문가 평가와 함께 해석해 주십시오",
  "used_schema_fields": [
    "posture_canonical",
    "class_set",
    "ambiguity_group",
    "uncertainty_flags",
    "caption_confidence_level"
  ]
}
```

### 4.2 `hedged` (size 3 within-group)

**input schema (요약):**
```json
{
  "posture_canonical": "CA",
  "class_set": ["C1", "C5", "C6"],
  "ambiguity_group": "within_group_c1_c5_c6",
  "uncertainty_flags": ["within_group_ambiguity_c1_c5_c6"],
  "caption_confidence_level": "hedged",
  "no_call": false,
  "no_call_reason": ""
}
```

**output (valid):**
```json
{
  "caption_ko": "팔을 교차한 자세에서 측정된 손목 IMU 패턴은, 정상 패턴과 무릎 관련 오류 후보 패턴이 함께 나타나는 경계 신호로 보입니다. 단일 유형으로 좁히기 어렵습니다. 손목 센서 기준의 추정입니다.",
  "confidence_phrase": "단일 유형으로 좁히기 어렵습니다",
  "uncertainty_phrase": "정상 패턴과 무릎 관련 오류 후보 패턴이 함께 나타나는 경계 신호",
  "limitation_phrase": "손목 센서 기준의 추정",
  "used_schema_fields": [
    "posture_canonical",
    "class_set",
    "ambiguity_group",
    "uncertainty_flags",
    "caption_confidence_level"
  ]
}
```

### 4.3 `low` (pair_plus_c2_absorption + anchor_unreliable)

**input schema (요약):**
```json
{
  "posture_canonical": "HW",
  "class_set": ["C3", "C4", "C2"],
  "ambiguity_group": "pair_plus_c2_absorption",
  "uncertainty_flags": ["pair_plus_c2_absorption", "anchor_unreliable"],
  "caption_confidence_level": "low",
  "no_call": false,
  "no_call_reason": ""
}
```

**output (valid):**
```json
{
  "caption_ko": "손을 허리에 둔 자세에서, 동작 기준점이 불안정하여 설명 신뢰도가 낮은 상태입니다. 깊이 부족 계열 신호와 복합 오류 후보가 함께 나타날 가능성이 있으나, 후보군 수준으로만 해석하는 것이 적절합니다.",
  "confidence_phrase": "후보군 수준으로만 해석하는 것이 적절합니다",
  "uncertainty_phrase": "깊이 부족 계열 신호와 복합 오류 후보가 함께 나타남 (동작 기준점이 불안정하여 설명 신뢰도가 낮습니다)",
  "limitation_phrase": "손목 센서 기준의 추정",
  "used_schema_fields": [
    "posture_canonical",
    "class_set",
    "ambiguity_group",
    "uncertainty_flags",
    "caption_confidence_level"
  ]
}
```

**주석 — C3 원 정의의 의도적 축약:**

- C3의 원 논문 정의는 *insufficient depth with posterior tilting and knee valgus* 이지만, 위 valid output에서 caption은 그 두 성격을 **"복합 오류 후보"** 라는 한 어구로 안전하게 낮춰 표현한다.
- 이는 sensor-grounded caption policy의 *의도적 축약*이다 (`step5_v2_caption_vocabulary.md` §1.2, `step5_v2_schema_grounded_caption_policy.md` §6.4.1, `step5_v2_forbidden_expressions.md` §2.1).
- "posterior tilting", "골반 후방 경사", "knee valgus", "무릎이 안으로 무너진다" 같은 표현은 사용자 caption에서 금지된다.

### 4.4 `no_call`

**input schema (요약):**
```json
{
  "posture_canonical": "HW",
  "class_set": [],
  "ambiguity_group": "no_call",
  "uncertainty_flags": ["anchor_unreliable"],
  "caption_confidence_level": "no_call",
  "no_call": true,
  "no_call_reason": "anchor_unreliable_anchor_dependent_set"
}
```

**output (valid):**
```json
{
  "caption_ko": "손을 허리에 둔 자세에서 기록된 손목 IMU 신호는 동작 기준점 신뢰도가 낮아, 현재 손목 센서 신호만으로는 안정적인 설명을 제공하기 어렵습니다.",
  "confidence_phrase": "안정적인 설명을 제공하기 어렵습니다",
  "uncertainty_phrase": "동작 기준점 신뢰도가 낮음",
  "limitation_phrase": "",
  "used_schema_fields": [
    "posture_canonical",
    "no_call",
    "no_call_reason",
    "caption_confidence_level"
  ]
}
```

---

## 5. Invalid output 예시

각 예시는 정책 / 어휘 / 금지 표현 중 어느 규칙을 위반했는지 명시.

### 5.1 no_call인데 확신형 출력

**output (invalid):**
```json
{
  "caption_ko": "양측 무릎이 안쪽으로 들어가는 패턴이 명확히 관찰됩니다."
}
```

**위반:**

- `step5_v2_forbidden_expressions.md` §1 #1 "무릎이 안쪽으로 들어갔다" (직접 + 변형).
- §1 #11 "명확히 잘못됐다" 계열 단정 표현 ("명확히").
- 정책 §4.4 — `no_call`이면 어떠한 클래스 단정도 금지.
- 어휘표 §1.1 — "양측 무릎" 같은 좌우 방향 표현 금지.
- 자동 검증 §3.4 no_call contradiction.

### 5.2 class_set이 3개인데 단일 class로 단정

**input schema:** `class_set = ["C1", "C5", "C6"]`, `caption_confidence_level = "hedged"`

**output (invalid):**
```json
{
  "caption_ko": "C1, C5, C6 중 C5에 가장 잘 맞는 패턴입니다."
}
```

**위반:**

- 정책 §5 size 3 — 개별 class를 하나로 좁히지 않음.
- 자동 검증 §3.5 class_set narrowing.
- 추가로 raw class 라벨(C5)을 사용자 caption에 그대로 노출 — 어휘표 §1의 사용자 표현 후보로 변환되어야 함.

### 5.3 knee valgus 직접 관찰 표현

**output (invalid):**
```json
{
  "caption_ko": "손목 IMU 신호상 knee valgus가 관찰됩니다."
}
```

**위반:**

- 금지 §1 #2 "knee valgus가 발생했다" 계열.
- 정책 §8.1 — wrist IMU는 knee valgus를 직접 측정하지 않음.
- 자동 검증 §3.6 unsupported biomechanical claim.

### 5.4 attention phase를 사용자에게 직접 설명

**output (invalid):**
```json
{
  "caption_ko": "모델의 attention이 동작 초반(descending phase)에 집중되어, 내려가는 동작 단서를 근거로 판정했습니다."
}
```

**위반:**

- 정책 §3.2 / §8.3 — attention 정보는 일반 사용자 caption에 노출 금지.
- 자동 검증 §3.7 attention term leakage.
- 본 정보는 technical report variant(정책 §3.3)에서만 optional 허용.

---

## 6. 출력 후처리 (참고 — Step 7_v2에서 구현)

- LLM이 위 §1의 system prompt를 따라 JSON 객체만 반환하도록 강제 (`response_format`이 지원되면 활용; 아니면 schema 검사 후 재시도).
- caption_ko에 §1의 forbidden token이 등장하면 retry.
- caption_ko에서 추출한 class label 후보가 `class_set`과 일치하지 않으면 retry.
- 3회 retry 실패 시 fallback: `no_call` 메시지로 강제 (caption layer 자체가 schema에 충실하지 않을 때의 마지막 안전장치).

---

## 7. 본 단계에서 명시적으로 결정하지 않는 사항

- LLM 모델 선택 (Step 6_v2 결정 사항).
- prompt에 vocabulary 표를 system / user 어디에 둘지 (Step 6_v2).
- temperature, top_p 등 LLM 호출 파라미터 (Step 7_v2).
- automatic validation의 LLM judge 사용 여부 (Step 8_v2).

---

*본 문서는 prompt 초안이며 LLM을 호출하지 않는다. caption도 만들지 않는다. 본 단계는 정책·어휘·금지·prompt 초안 4문서까지로 종료된다.*
