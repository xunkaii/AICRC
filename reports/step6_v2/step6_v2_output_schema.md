# Step 6_v2 — Output Schema

본 문서는 Step 7_v2 caption generation prototype의 **출력 JSON 형식**과 **invalid 조건**, **retry / fallback 정책**을 고정한다.

선행: `step6_v2_final_system_prompt.md`, `step6_v2_final_user_prompt_template.md`, `reports/step5_v2/*`.

---

## 1. 출력 JSON 필드

LLM은 sample마다 다음 5개 필드만 가진 JSON 객체 *하나*를 출력한다.

```json
{
  "caption_ko": "<string>",
  "confidence_phrase": "<string>",
  "uncertainty_phrase": "<string>",
  "limitation_phrase": "<string>",
  "used_schema_fields": ["<string>", "..."]
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `caption_ko` | string (1 ~ 3문장 한국어) | 사용자에게 노출되는 본문 |
| `confidence_phrase` | string | `caption_confidence_level`을 반영하는 짧은 어구 |
| `uncertainty_phrase` | string | `ambiguity_group` / `uncertainty_flags`를 반영하는 짧은 어구. `confident_C2` 단순 케이스에서는 빈 문자열 가능 |
| `limitation_phrase` | string | sensor-grounded notice. `no_call`에서는 빈 문자열 허용 |
| `used_schema_fields` | array of string | 이 caption이 *실제 참조한* schema 필드 이름 배열 |

추가 필드를 LLM이 임의로 만들어 내면 invalid이다.

---

## 2. 각 필드의 조건

### 2.1 `caption_ko`

- 한국어 1 ~ 3문장.
- 빈 문자열 금지.
- `posture_vocabulary[posture_canonical]`의 표현이 caption 본문 *시작부* 근처에 한 번 등장해야 함 (정책 §7).
- `class_set`이 비어있지 않으면, `class_vocabulary`의 사용자 표현이 적어도 한 번 등장해야 함.
- raw class 라벨(`C1`, `C2`, `C3`, `C4`, `C5`, `C6`)을 caption 본문에 *직접* 노출하지 않음.
- `caption_confidence_level`에 부합하는 어조 (§4 system prompt 어조 표).

### 2.2 `confidence_phrase`

- `caption_confidence_level`이 `confident`이면 `confidence_phrase_pool["confident"]` 중 하나 또는 의미상 동등한 어구.
- `hedged` / `low` / `no_call`도 동일하게 해당 pool 어구로 제한.
- 단순한 짧은 어구 (1 절 이내).

### 2.3 `uncertainty_phrase`

- `ambiguity_group` 값이 `uncertainty_phrase_pool`의 키와 매핑.
- `confident_C2` + `uncertainty_flags`가 단순 `["confident_C2"]`만 있는 경우 빈 문자열 허용.
- `anchor_unreliable` flag가 동반되면 `uncertainty_phrase_pool["anchor_unreliable_suffix"]`를 덧붙일 수 있음.

### 2.4 `limitation_phrase`

- `caption_confidence_level`이 `no_call`이면 빈 문자열 허용 (no_call 메시지 자체가 한계 표명이라 한 번 더 붙일 필요 없음).
- 그 외 level에서는 `limitation_phrase_pool`에서 1개 어구 선택.

### 2.5 `used_schema_fields`

- 본 caption이 참조한 schema 필드 이름의 부분집합. 가능한 값:
  `["posture_canonical", "class_set", "ambiguity_group", "uncertainty_flags", "caption_confidence_level", "no_call", "no_call_reason"]`.
- 빈 배열 금지.
- 해당 caption에 *실제로 영향을 준* 필드만 포함.

---

## 3. invalid 조건 (Step 8_v2 자동 검증과 일관)

다음 중 하나라도 해당하면 LLM 출력은 invalid이다.

| # | invalid 조건 | 검사 방식 |
|---:|---|---|
| 1 | JSON으로 파싱되지 않음 | `json.loads` 실패 |
| 2 | 5개 필드 중 누락 | 키 존재 검사 |
| 3 | 추가 필드 등장 | 키 화이트리스트 검사 |
| 4 | `caption_ko`가 빈 문자열 또는 공백만 | strip 후 길이 검사 |
| 5 | `caption_ko`에 raw class 라벨(C1~C6) 직접 노출 | 정규식 `\bC[1-6]\b` |
| 6 | `no_call=true`인데 caption이 분류 단정 | class_vocabulary 표현 등장 검사 |
| 7 | `class_set` 길이 ≥ 2인데 caption이 단일 후보로 좁힘 | `step5_v2_forbidden_expressions.md` §3.5 |
| 8 | forbidden expression 등장 | §1 ~ §1 #1~#18 token / semantic 검사 |
| 9 | attention / model-internal 용어 등장 | "attention", "phase", "anchor", "entropy", "probability", "posterior", "logit", "temperature" 토큰 검사 |
| 10 | schema에 없는 biomechanical claim | "knee valgus", "posterior tilting", "골반", "관절각", "정렬", "안쪽으로", "양측", "한쪽", "왼쪽", "오른쪽", … |
| 11 | confidence-level mismatch | `confidence_phrase`가 schema의 `caption_confidence_level`에 부합하지 않음 |
| 12 | `caption_ko`가 4문장 초과 또는 0문장 | 문장 분할 후 개수 검사 |
| 13 | `posture_canonical`에 해당하는 vocabulary가 caption에 누락 | `posture_vocabulary[posture_canonical]` 표현 검사 |
| 14 | `used_schema_fields`가 비어 있음 | 길이 ≥ 1 검사 |
| 15 | `used_schema_fields`에 화이트리스트 외 값 | 키 검사 |

---

## 4. retry / fallback 정책 초안

### 4.1 retry

- invalid 출력 발생 시 LLM 호출을 **최대 3회**까지 retry한다.
- retry 시점에 `system` prompt와 `user` prompt를 그대로 다시 보낸다 (변형하지 않음).
- 동일한 invalid 위반이 3회 연속 발생하면 retry를 중단한다.

### 4.2 fallback

- 3회 retry 실패 시 fallback caption을 강제 출력한다.
- fallback caption (고정 본문):

```json
{
  "caption_ko": "현재 손목 센서 신호만으로는 안정적인 설명을 제공하기 어렵습니다.",
  "confidence_phrase": "안정적인 설명을 제공하기 어렵습니다",
  "uncertainty_phrase": "",
  "limitation_phrase": "",
  "used_schema_fields": ["caption_confidence_level"]
}
```

- fallback 시점에 caption 자체는 위 본문으로 *고정*되며, LLM의 출력은 폐기된다.
- **`caption_confidence_level`은 schema 원본을 유지**한다 (fallback 때문에 `no_call`로 강제 변경하지 않음). schema의 결정과 caption layer의 fallback은 *분리된 책임*이다.

### 4.3 fallback trace (사용자 비공개)

- Step 7_v2 내부 로그(예: `data/step7_v2/.../caption_run_log.csv`)에 다음 컬럼을 남긴다.
  - `sample_id`
  - `final_status` ∈ {`ok`, `retry_then_ok`, `fallback`}
  - `n_retries` (0 ~ 3)
  - `fallback_reason` (마지막 위반 카테고리; §3 invalid 조건 #1 ~ #15 중)
- **출력 JSON에는 `fallback_reason`을 사용자 필드로 노출하지 않는다.** 본 정보는 Step 8_v2 metric 집계에만 사용된다.

---

## 5. metric 연결 (Step 8_v2)

본 schema 위에서 Step 8_v2가 집계할 metric (이름은 `step6_v2_prompt_validation_checklist.md` §최종부와 일치):

- `json_valid_rate`
- `required_field_complete_rate`
- `forbidden_expression_rate`
- `confidence_mismatch_rate`
- `no_call_contradiction_rate`
- `class_set_narrowing_rate`
- `unsupported_biomechanical_claim_rate`
- `attention_leakage_rate`
- `schema_faithfulness_rate` (위 모든 검사 통과율)
- `fallback_rate`
- `mean_n_retries`

---

## 6. 본 단계가 명시적으로 결정하지 않는 사항

- LLM 모델 선택, API 호출 파라미터(temperature, top_p, max_tokens) — Step 7_v2.
- batch 호출 vs 순차 호출, 동시성 — Step 7_v2.
- LLM judge 사용 여부 (semantic forbidden 검사용) — Step 8_v2.
- forbidden expression의 정규식 vs LLM judge 비율 — Step 8_v2.

---

*본 문서는 출력 schema commit이며, LLM을 호출하지 않는다. 본 문서가 변경되면 Step 8_v2 validation 코드도 같이 갱신해야 한다.*
