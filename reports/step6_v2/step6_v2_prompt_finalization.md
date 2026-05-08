# Step 6_v2 — Prompt Finalization 메모

본 문서는 Step 5_v2의 정책·어휘·금지·prompt 초안을 **Step 7_v2 caption generation prototype에서 그대로 사용할 형태**로 잠그는 단계의 전체 메모이다. 본 단계에서는 LLM을 호출하지 않으며 caption도 만들지 않는다.

본 메모는 Step 6_v2의 *상위 인덱스*이다. 실제 prompt 본문 / template / output schema / golden cases / validation checklist는 동일 디렉토리의 5개 별도 파일에 분리되어 있다.

---

## 1. Step 6_v2의 목적

- Step 5_v2의 정책 4문서를 **실제 LLM prompt와 test artifact**로 고정한다.
- caption 생성 코드는 만들지 않는다.
- Step 7_v2(caption generation prototype)의 *입력 문서*가 된다.

---

## 2. Step 5_v2에서 확정된 정책 요약

본 단계가 의존하는 Step 5_v2의 결정사항:

1. **schema-grounded caption** — caption은 schema 출력이 *이미 존재하는 사실*을 한국어로 풀어내는 표현층이다. sensor-to-text generation이 아니다.
2. **LLM은 표현층** — 새 클래스 추론 / class_set narrowing / schema에 없는 biomechanical claim 추가는 모두 invalid.
3. **forbidden expression** — knee valgus / posterior tilting / 골반 경사 / 무릎 정렬 / 좌우 방향성 / 의학 진단·부상 위험은 모두 사용자 caption에서 금지.
4. **attention 직접 노출 금지** — attention / phase / entropy / probability 등 모델 내부 용어는 사용자 caption에 등장 불가 (technical variant 한정 허용 — 본 단계 범위 외).
5. **C3 원 정의의 caption-safe 축약** — *insufficient depth with posterior tilting and knee valgus*는 **"깊이 부족과 복합 오류 후보가 함께 나타나는 신호"**로 표현.
6. **C4 / C5 / C6 방향 단정 금지** — left / right / both knee valgus는 모두 **"무릎 관련 오류 후보가 섞인 신호"**로 통일.

---

## 3. 최종 prompt 설계 결정

| 결정 | 본 단계의 commit |
|---|---|
| vocabulary 위치 | system prompt에는 *제약과 안전 어구*만, **자세한 어휘표는 user prompt에 첨부**. 매 sample마다 동일하게 들어가므로 prompt cache에 적합. |
| user prompt 본문 | sample별 schema JSON + 5종 vocabulary pool. 자유 텍스트 없음. |
| forbidden 처리 | system prompt에는 **핵심 forbidden 목록**만 (knee valgus / posterior tilting / 좌우 방향성 / 의학 진단·부상 위험 / 모델 내부 용어). 전체 18개 forbidden + 의미 변형은 Step 8_v2 validation에서 검사. |
| limitation phrase | caption_ko 안에 매번 강제로 넣지 않고 **별도 필드 `limitation_phrase`로 분리**. caption_ko에는 필요할 때만 짧게 반영. `no_call`에서는 빈 문자열 허용. |
| JSON-only 출력 | 5개 필드 고정. 자유 텍스트·markdown·코드블록 fence 금지. retry 3회 → fallback. |

---

## 4. 최종 입력 필드

### 4.1 LLM에 *넘기는* schema 필드 (8개)

```
sample_id
posture_canonical
class_set
ambiguity_group
uncertainty_flags
caption_confidence_level
no_call
no_call_reason
```

### 4.2 LLM에 *넘기지 않는* 필드

```
top1_class                       (raw class label)
top2_class                       (raw class label)
top1_prob_calibrated             (raw probability)
top2_prob_calibrated             (raw probability)
top1_top2_margin_calibrated      (raw probability)
predictive_entropy_calibrated    (분포 metric)
class_set_posterior_mass         (분포 metric)
temperature                      (메타데이터)
attention_entropy                (모델 내부)
attention_peak_phase             (모델 내부)
attention_peak_timestep          (모델 내부)
descending/bottom_transition/ascending_recovery_mass (모델 내부)
anchor_distance                  (모델 내부)
anchor_reliability (raw float)   (이미 anchor_unreliable flag로 흡수)
anchor_type                      (이미 schema 분기에 사용)
```

이 필드들은 schema 분기를 형성하는 데 *이미* 사용되었고, caption layer에는 *결과물(class_set, ambiguity_group, confidence level, anchor_unreliable flag)*만 넘긴다.

---

## 5. 최종 출력 필드

```json
{
  "caption_ko": string,
  "confidence_phrase": string,
  "uncertainty_phrase": string,
  "limitation_phrase": string,
  "used_schema_fields": [string]
}
```

각 필드 조건과 invalid 규칙은 `step6_v2_output_schema.md` §2~§3 참조.

---

## 6. Step 7_v2로 넘기는 결정사항

본 메모가 commit한 5개 문서를 *입력*으로 받아 Step 7_v2가 결정해야 할 항목:

| 항목 | 본 단계 status |
|---|---|
| LLM 모델 선택 | 미결정 (Step 7_v2). Anthropic / OpenAI / 한국어 LM 중 후보 비교 필요. |
| API 호출 방식 (단발 vs batch) | 미결정 (Step 7_v2). 9275 sample × 비용 vs latency 검토. |
| temperature, top_p, max_tokens | 미결정. 권장: temperature 0.0 ~ 0.3 (caption은 결정론적 변환). |
| retry / fallback 정책 | **본 단계에서 초안 commit** (`step6_v2_output_schema.md` §4). retry 3회, fallback 고정 본문. 최종 횟수는 Step 7_v2에서 sensitivity 보고 commit. |
| validation checklist 구현 | **본 단계에서 항목 commit** (`step6_v2_prompt_validation_checklist.md`). 정규식 / LLM judge 비율은 Step 8_v2에서 commit. |

---

## 7. 본 단계가 명시적으로 결정하지 *않는* 사항

- LLM 모델 선택, API 키 관리, billing 정책.
- caption batch 크기, 동시성, rate limiting.
- temperature 등 호출 파라미터의 *최종* 값.
- LLM judge 사용 여부 (Step 8_v2 forbidden semantic 검사).
- caption 결과의 사용성·가독성 검토 (`reports/step4_research_reframing.md` §7에 따라 main evaluation 제외).
- prompt를 영어로 작성할지 한국어로 작성할지 (본 단계는 **한국어 system prompt**로 commit; 영어 변환 검토는 별도 단계).

---

## 8. 본 단계 산출물 6종

| 파일 | 내용 |
|---|---|
| `step6_v2_prompt_finalization.md` | 본 메모 (상위 인덱스). |
| `step6_v2_final_system_prompt.md` | Step 7_v2에서 그대로 사용할 한국어 system prompt 본문. |
| `step6_v2_final_user_prompt_template.md` | sample별 user prompt JSON template + vocabulary pool. |
| `step6_v2_output_schema.md` | 출력 JSON 5필드 + invalid 조건 + retry/fallback. |
| `step6_v2_golden_test_cases.json` | 15개 case (실제 sample_id 기반). 11개 valid demonstration + 4개 forbidden demonstration. |
| `step6_v2_prompt_validation_checklist.md` | 14개 validation 항목 + 종합 `schema_faithfulness_rate`. |

추가로 임시 산출물 1종:

| 파일 | 내용 |
|---|---|
| `_golden_case_samples_extracted.json` | golden case 추출 시 사용한 실제 sample id 매칭 결과 (debugging trace). |

---

## 9. Golden test cases — 실제 sample id 기반

`step6_v2_golden_test_cases.json`의 15개 case는 모두 *실제* `data/step4r/4rb_attention/step4r_bigru_attention_schema_outputs_calibrated.csv`에서 조건에 맞는 sample을 추출하여 사용했다 (가상 sample 사용 안 함).

| case | sample_id | 조건 |
|---|---|---|
| 1 | `EP01_C1_SA_rep06` | confident_C2 + SA (anchor downgraded → hedged) |
| 2 | `EP01_C2_CA_rep01` | confident_C2 + CA (true confident) |
| 3 | `EP01_C1_CA_rep02` | within_group_c1_c5_c6 + hedged |
| 4 | `EP01_C1_CA_rep01` | within_group_c1_c5_c6 + low + anchor_unreliable |
| 5 | `EP01_C1_HW_rep01` | pair_c3_c4 + hedged |
| 6 | `EP02_C3_SA_rep04` | pair_plus_c2_absorption + low |
| 7 | `EP02_C3_SA_rep04` | (case 6과 동일 sample, anchor_unreliable 명시 강조 purpose) |
| 8 | `EP01_C2_SA_rep01` | no_call + anchor_unreliable_anchor_dependent_set |
| 9 | `EP03_C1_SA_rep09` | no_call + low_confidence_no_class_set |
| 10 | `EP01_C1_HW_rep01` | class_set size 2 demonstration (= case 5) |
| 11 | `EP01_C1_CA_rep01` | class_set size 3 demonstration (= case 4) |
| 12 | `EP01_C1_CA_rep08` | FORBIDDEN: knee valgus direct claim |
| 13 | `EP01_C3_HW_rep03` | FORBIDDEN: posterior tilting direct claim |
| 14 | `EP01_C2_SA_rep04` | FORBIDDEN: attention term leakage |
| 15 | `EP01_C1_CA_rep04` | FORBIDDEN: class_set narrowing |

**자연스러운 sample 중복:** case 6 / 7은 같은 sample이다. 데이터 분포상 `pair_plus_c2_absorption + low`가 33건이며 *모두* `anchor_unreliable`을 동반한다 (low로 강등된 사유). case 6 / 7은 같은 input에 대해 *purpose*가 다른 검사이다 (case 6: low 어조 / class_set 표현, case 7: anchor_unreliable 명시 언어화). case 5 ↔ 10, case 4 ↔ 11도 같은 sample이며 demonstration 목적이 다르다.

---

## 10. Step 7_v2로 넘어가도 되는지 — **YES**

다음 모두 충족됨:

1. **prompt 본문이 한국어로 commit** (system prompt + user prompt template).
2. **출력 JSON 5필드 + invalid 조건 + retry/fallback이 commit**.
3. **15개 golden test cases가 실제 sample id로 작성** (Step 7_v2 prototype 검증과 Step 8_v2 metric의 grounding 자료).
4. **14개 validation 항목과 metric 이름 commit** — Step 7_v2 retry 로직과 Step 8_v2 metric 코드가 동일한 어휘를 공유.
5. **명시적으로 미결정으로 둔 항목들이 §7에 분리** — Step 7_v2와 Step 8_v2가 어디서 무엇을 commit해야 하는지 명확.

Step 7_v2 진입 시 첫 작업은 본 6문서를 *읽고* `caption layer 호출 코드`를 작성하는 것이다 (caption 생성 자체는 Step 7_v2의 역할).

---

*본 메모는 Step 6_v2 인덱스이며, LLM을 호출하지 않는다. 어떤 caption도 만들지 않는다. 기존 Step 1 ~ 8 / 4R-A / 4R-B / Step 5_v2 산출물은 수정되지 않는다.*
