# Step 6_v2 — Prompt Validation Checklist

본 문서는 Step 7_v2 생성 결과와 Step 8_v2 자동 검증이 사용할 **체크리스트**를 정의한다. 각 항목에 대해 *목적 / 검사 방식 / invalid 예시 / Step 8_v2 metric 이름* 4가지를 명시한다.

선행: `step6_v2_output_schema.md` §3 (invalid 조건 표), `step5_v2_forbidden_expressions.md` §3 (자동 검증 규칙 후보).

---

## 1. JSON validity

- **목적**: LLM이 자유 텍스트가 아니라 단일 JSON 객체로 응답했는지 확인.
- **검사 방식**: `json.loads(output)` 성공 여부. 코드블록 펜스(``` ... ```)가 둘러쌓인 경우 strip 후 재시도하되, strip 전후 양쪽 invalid이면 fail.
- **invalid 예시**:
  ```
  여기 caption입니다:
  { "caption_ko": "..." }
  ```
- **metric**: `json_valid_rate`

---

## 2. Required fields presence

- **목적**: 5개 필드(`caption_ko`, `confidence_phrase`, `uncertainty_phrase`, `limitation_phrase`, `used_schema_fields`)가 모두 존재하는지.
- **검사 방식**: 키 화이트리스트 정확 일치 검사 (누락 / 추가 모두 fail).
- **invalid 예시**: `caption_ko`만 있고 나머지 누락. 또는 `model_notes`라는 새 키가 등장.
- **metric**: `required_field_complete_rate`

---

## 3. Forbidden expression exact match

- **목적**: `step5_v2_forbidden_expressions.md` §1 #1 ~ #18의 정확한 문자열 + 추가 *관찰* 단정 어휘가 caption에 등장하는지.
- **검사 방식**: forbidden 문자열 각각에 대해 `caption_ko + " " + confidence_phrase + " " + uncertainty_phrase + " " + limitation_phrase`를 합친 텍스트에서 substring 검사.
- **추가 어휘 (2026-05-08)**: "관찰됩니다", "관찰된다", "관찰되었다", "관찰됨" — wrist IMU는 직접 *관찰*하지 않으므로 caption에 부적절. 대체: "나타납니다", "보입니다", "해석 가능한 신호입니다".
- **invalid 예시**:
  - "knee valgus가 관찰됩니다." (knee valgus + 관찰됩니다 둘 다 위반)
  - "후방 경사가 일어났습니다."
  - "양측 무릎이 안쪽으로 모입니다."
  - "정상 패턴이 비교적 일관되게 관찰됩니다." ("관찰됩니다" 위반)
- **metric**: `forbidden_expression_rate` (이 비율이 0이 되는 것이 목표)

---

## 4. Forbidden semantic pattern (후보)

- **목적**: §3을 회피하기 위한 *변형 표현* 검출. 예: "안쪽으로 들어가는", "안으로 무너지는", "뒤로 빠지는", "허벅지 안쪽으로 모이는" 등.
- **검사 방식**: 핵심어 조합 정규식 후보 (예: `(무릎|허벅지).*(안쪽|안으로|모이)`, `(골반|엉덩이).*(뒤로|후방)`)와 LLM judge 둘 중 하나 이상. LLM judge 사용 시 출력은 closed-vocabulary 라벨(`pass` / `fail` + reason enum)로만 받는다.
- **invalid 예시**:
  - "허벅지 안쪽으로 모이는 패턴이 보입니다."
  - "엉덩이가 뒤로 빠지는 모양새입니다."
- **metric**: `forbidden_semantic_rate`

---

## 5. Confidence-level mismatch

- **목적**: caption의 단정도(definiteness)가 schema의 `caption_confidence_level`과 어긋나지 않는지.
- **검사 방식**:
  - schema가 `confident`인데 caption에 hedge phrase ("단정하기 어렵습니다", "후보군 수준") 우세 → mismatch.
  - schema가 `low` / `no_call`인데 caption에 confident phrase ("비교적 일관되게", "안정적인 신호") 등장 → mismatch.
  - confidence_phrase가 해당 level의 `confidence_phrase_pool` 어구와 의미적으로 동등한지 검사 (정규식 + 동의 매핑).
- **invalid 예시**:
  - schema `confident` + caption "단정하기 어렵습니다" → mismatch.
  - schema `no_call` + caption "비교적 일관되게 나타납니다" → mismatch.
- **metric**: `confidence_mismatch_rate`

---

## 6. No_call contradiction

- **목적**: `no_call=true`일 때 caption이 어떤 클래스 단정도 하지 않는지.
- **검사 방식**: `no_call=true`이고 caption_ko에 `class_vocabulary` 사용자 표현 중 하나라도 등장하면 fail.
- **invalid 예시**:
  - schema `no_call=true` + caption "깊이 부족 계열로 해석 가능한 신호가 나타납니다" (class 단정).
  - schema `no_call=true` + caption "정상 패턴에 가까운 신호입니다".
- **metric**: `no_call_contradiction_rate`

---

## 7. Class_set narrowing

- **목적**: `class_set` 길이 ≥ 2인데 caption이 *단일 후보*로 좁히는지.
- **검사 방식**:
  - caption_ko에서 `class_vocabulary` 사용자 표현 등장 횟수 추출.
  - schema `class_set`의 매핑된 사용자 표현 *전부* 또는 *그룹화된 표현*이 등장해야 함 (C5/C6는 같은 표현이라 한 번 등장으로 둘 다 충족).
  - 단일 후보만 등장하면 fail.
- **invalid 예시**:
  - schema `class_set=["C1","C5","C6"]` + caption "C5에 가장 잘 맞는 패턴입니다" → narrowing.
  - schema `class_set=["C3","C4","C2"]` + caption "복합 오류 후보로만 해석됩니다" (C2 흡수 누락) → narrowing.
- **metric**: `class_set_narrowing_rate`

---

## 8. Unsupported biomechanical claim

- **목적**: wrist IMU가 직접 측정하지 않는 양에 대한 단정이 등장하는지.
- **검사 방식**: §3 (forbidden token) + §4 (semantic) 통합. 검사 대상 어휘 — knee valgus / posterior tilting / 골반 경사 / 무릎 정렬 / 관절각 / 절대 squat depth / 좌우 방향성 / 부상 위험 / 진단.
- **invalid 예시**: §3 / §4 invalid 예시 그대로.
- **metric**: `unsupported_biomechanical_claim_rate`

---

## 9. Attention / model-internal term leakage

- **목적**: 일반 사용자 caption에 모델 내부 용어 직접 노출 차단.
- **검사 방식**: 키워드 substring 검사 — `attention`, `attention weight`, `phase`, `ascending phase`, `descending phase`, `bottom_transition`, `anchor`, `peak timestep`, `predictive entropy`, `attention entropy`, `top1`, `posterior`, `logit`, `softmax`, `temperature` 그리고 한국어 표현 — "어텐션", "확률", "엔트로피", "포스테리어", "로짓", "온도".
- **invalid 예시**:
  - "모델 attention이 descending phase에 집중되어..."
  - "predictive entropy가 낮아 비교적 신뢰할 수 있습니다."
- **metric**: `attention_leakage_rate`

---

## 10. Posture phrase consistency

- **목적**: `posture_canonical`에 해당하는 한국어 표현이 caption 본문에 정확히 등장하는지.
- **정책 강화 (2026-05-08)**: `step6_v2_final_system_prompt.md` §5에 따라 **no_call=true인 경우에도 posture phrase 누락은 fail**이다. fallback caption 역시 schema의 posture를 받아 첫 문장에 자세 표현을 포함한다(`step6_v2_output_schema.md` §4.2).
- **검사 방식**:
  - SA → "팔을 앞으로 둔 자세" 부분 문자열 등장.
  - CA → "팔을 교차한 자세" 부분 문자열 등장.
  - HW → "손을 허리에 둔 자세" 부분 문자열 등장.
  - 다른 자세 표현이 들어오면 fail (예: HW인데 "팔을 앞으로 둔 자세"가 들어가면 fail).
- **invalid 예시**:
  - schema `posture_canonical=HW` + caption "팔을 앞으로 둔 자세에서..." → mismatch.
  - schema `no_call=true`, `posture_canonical=SA` + caption "현재 손목 센서 신호만으로는 안정적인 설명을 제공하기 어렵습니다." (자세 표현 없음) → posture_phrase_missing.
- **metric**: `posture_phrase_mismatch_rate`

---

## 11. Class vocabulary consistency

- **목적**: caption이 raw class label (C1~C6)을 직접 노출하지 않는지, 그리고 `class_set` 안의 class에 대해 매핑된 사용자 표현이 적절히 사용되는지.
- **검사 방식**:
  - 정규식 `\bC[1-6]\b`이 caption_ko에 매칭되면 fail.
  - C4/C5/C6 처리에서 "한쪽", "반대쪽", "양측", "왼쪽", "오른쪽" 등 좌우 방향 표현 검출되면 fail.
- **invalid 예시**:
  - "C5에 가까운 패턴" → raw label 노출.
  - "양측 무릎 관련 오류" → 방향 단정.
- **metric**: `class_vocabulary_violation_rate`

---

## 12. used_schema_fields consistency

- **목적**: LLM이 자체 보고한 `used_schema_fields`가 실제 caption에 영향을 주었을 만한 필드와 부합하는지.
- **검사 방식**:
  - `used_schema_fields` ⊆ {`posture_canonical`, `class_set`, `ambiguity_group`, `uncertainty_flags`, `caption_confidence_level`, `no_call`, `no_call_reason`}.
  - 비어 있으면 fail.
  - schema가 `no_call=true`인데 `used_schema_fields`에 `no_call`이 없으면 fail.
  - schema의 `caption_confidence_level`이 caption 어조에 명백히 영향을 주었는데 `caption_confidence_level`이 누락되면 fail.
- **invalid 예시**:
  - `"used_schema_fields": []`.
  - `"used_schema_fields": ["temperature"]` (화이트리스트 외).
- **metric**: `used_schema_fields_violation_rate`

---

## 13. Limitation phrase policy

- **목적**: limitation phrase 사용 정책 준수 (`step5_v2_caption_vocabulary.md` §5.1).
- **검사 방식**:
  - `caption_confidence_level=no_call`이면 `limitation_phrase`가 빈 문자열이어도 OK.
  - 그 외 level에서 `limitation_phrase`가 비어 있고 `caption_ko` 본문에도 limitation 표현이 전혀 없으면 fail.
  - `limitation_phrase`가 너무 길어 caption 1문장보다 긴 경우 fail (단일 어구 원칙).
- **invalid 예시**:
  - schema `confident` + `limitation_phrase=""` + caption_ko에 "손목 센서" 미등장 → fail.
  - `limitation_phrase`가 200자 이상 긴 문단 → fail.
- **metric**: `limitation_phrase_violation_rate`

---

## 14. Fallback trigger condition

- **목적**: §1 ~ §13 검사가 retry 후에도 통과하지 못하는 비율 추적.
- **검사 방식**: Step 7_v2 caption run log에서 `final_status == "fallback"` 비율 집계.
- **fallback caption**: `step6_v2_output_schema.md` §4.2의 고정 본문.
- **metric**:
  - `fallback_rate` (전체 sample 중 fallback 비율, 목표: 매우 낮음)
  - `mean_n_retries` (성공한 caption까지의 평균 retry 횟수)

---

## 15. 종합 — schema_faithfulness_rate

- **정의**: §1 ~ §13 모두 통과한 sample의 비율.
- **목적**: 본 caption layer 전체의 *합격률*을 단일 지표로 요약.
- **metric**: `schema_faithfulness_rate`

본 metric이 Step 4R-B 후처리 결과를 *책임감 있게 사용자에게 전달했는지*를 측정하는 main contribution metric이다 (`reports/step4_research_reframing.md` §8.3).

---

## 16. 본 체크리스트가 갱신되는 조건

- 새 forbidden expression이 `step5_v2_forbidden_expressions.md` §1에 추가될 때.
- LLM judge 사용 여부가 Step 8_v2에서 commit될 때 (§4 검사 방식 보강).
- 새로운 invalid 패턴이 Step 7_v2 caption run에서 *반복적으로* 검출될 때.

---

*본 문서는 validation checklist commit이며, LLM을 호출하지 않는다. 본 문서가 변경되면 Step 7_v2의 retry 로직과 Step 8_v2의 metric 집계 코드도 같이 갱신해야 한다.*
