# Step 5_v2 — Schema-grounded Caption Policy

본 문서는 Step 4R-B BiGRU+Attention의 *calibrated* schema output을 한국어 caption으로 변환하기 위한 **정책 명세서**이다. 본 문서는 caption을 생성하지 않으며, caption 생성 코드도 만들지 않는다. 본 문서는 prompt 본문, 어휘표, 금지 표현, confidence별 문장 원칙을 *고정*한다.

본 문서가 의존하는 선행 문서:

- `reports/step4r/step4_research_reframing.md` — 본 연구의 reframing (특히 §6 schema-grounded LLM caption, §11 명시적 제외 사항).
- `reports/step3/step3_output_schema_uncertainty_policy.md` — Step 3 출력 스키마 및 불확실성 정책.
- `reports/step4r/4rb_attention/step4r_bigru_attention_schema_results.md` — 4R-B calibrated schema 결과.
- `reports/step4r/4rb_attention/step4r_attention_phase_analysis.md` — attention phase 분석 (정보 출처 한정 용도).

본 문서는 기존 `reports/step5/`(legacy rule/template caption)과 명확히 분리된다 (`reports/step4_research_reframing.md` §10 보존 정책).

---

## 1. Step 5_v2의 목적

Step 5_v2는 **schema-to-caption presentation layer**의 정책을 고정한다. 다음 두 가지가 본 단계의 핵심 입장이다.

1. 본 시스템은 **sensor-to-text generation이 아니다.** caption은 학습 target이 아니며, schema 출력이 *이미 존재하는 사실*을 한국어로 표현하는 표현층(presentation layer)일 뿐이다.
2. **LLM은 판단자(judge / classifier)가 아니라 표현층이다.** LLM은 schema에 적힌 사실만 한국어로 풀어낸다. 새 클래스를 추론하거나 class_set을 좁히거나 측정되지 않은 생체역학 양을 도입하면 모두 invalid output이다.

본 단계 종료 후, Step 6_v2 (prompt finalization) 와 Step 7_v2 (caption generation prototype) 가 본 정책 위에서 진행된다.

---

## 2. 입력 schema 필드

caption layer는 rep마다 다음 필드를 입력으로 받는다 (`reports/step4r/4rb_attention/step4r_bigru_attention_schema_outputs_calibrated.csv`의 컬럼).

| 필드 | 타입 | 출처 |
|---|---|---|
| `sample_id` | string | manifest |
| `posture_canonical` | enum {`SA`, `CA`, `HW`} | manifest |
| `class_set` | list[string] (whitelist 8종) | 4R-B schema rule |
| `class_set_posterior_mass` | float | calibrated posterior 합 |
| `top1_class` | string | calibrated argmax |
| `top1_prob_calibrated` | float | calibrated posterior |
| `top2_class` | string | calibrated posterior |
| `top2_prob_calibrated` | float | calibrated posterior |
| `top1_top2_margin_calibrated` | float | calibrated posterior |
| `predictive_entropy_calibrated` | float | calibrated posterior |
| `ambiguity_group` | enum (§6 참조) | schema rule |
| `uncertainty_flags` | list[string] (Step 3 §7 closed vocab) | schema rule |
| `no_call` | bool | schema rule |
| `no_call_reason` | enum | schema rule |
| `caption_confidence_level` | enum {`confident`, `hedged`, `low`, `no_call`} | schema rule |
| `model_source` | string (`"4R-B_BiGRU_Attention_calibrated"`) | metadata |
| `temperature` | float | T = 1.9579 (4R-B post 1/3) |

---

## 3. caption에 반영할 필드와 반영하지 않을 필드

### 3.1 일반 사용자 caption에 **반영하는** 필드

- `posture_canonical` — 자세를 *해석 조건*으로 명시 (자세 자체를 오류로 표현하지 않음).
- `class_set` — class_set 안의 모든 후보를 *공동 헤지*로 표현. 단일 후보로 좁히지 않는다.
- `caption_confidence_level` — 어조와 단정도(definiteness)를 결정.
- `uncertainty_flags` — 어떤 모호함을 명시할지를 결정.
- `no_call` / `no_call_reason` — 보류 메시지 형태와 사유 표현.
- `ambiguity_group` — class_set과 flag를 묶어 정해진 단일 표현 분기를 고름.
- limitation notice (`limitation_phrase`) — "손목 센서 기준" 등 sensor-grounded notice.

### 3.2 일반 사용자 caption에 **직접 반영하지 않는** 필드

다음은 caption 본문(`caption_ko`)에 *수치*나 *모델 내부 용어*로 직접 노출하지 않는다.

- raw probability 숫자 (`top1_prob_calibrated`, `top2_prob_calibrated`, `class_set_posterior_mass`)
- `predictive_entropy_calibrated` 숫자
- attention entropy / attention peak phase / phase mass / anchor-attention distance (Step 4R-B post 3/3 산출물 전체)

이 정보는 분류 결정과 schema 분기를 *형성*하는 데 이미 사용되었으며, caption 어조에 *간접적으로* 반영된다 (예: top1_prob이 낮으면 confidence가 hedged/low로 강등됨). 사용자 caption에 숫자나 attention 용어로 *직접 노출*하지 않는다.

### 3.3 technical report용 caption variant (optional)

논문 figure / appendix용으로 별도의 *technical caption variant*는 위 §3.2의 항목을 선택적으로 포함할 수 있다. 단,

- 이 variant는 **별도의 출력 채널**로 다룬다 (`caption_technical_md` 등 별도 필드).
- 일반 사용자 caption(`caption_ko`)과 *섞이지 않는다*.
- 본 정책은 일반 사용자 caption에 한정한다.

---

## 4. confidence별 문장 정책

### 4.1 `confident`

- 단일 class_set (본 시스템에서는 사실상 `["C2"]`만 도달) 또는 *그에 준하는 안정 신호*에서만 사용.
- "손목 센서 기준" 한계 명시는 그대로 유지.
- 금지: "명확히", "확실히", "진단된다", "정상이다", "오류가 없다" 등 단정/의학 표현.
- 허용: "비교적 일관되게 나타납니다", "상대적으로 안정적인 신호입니다" (§2 vocabulary 참조).

### 4.2 `hedged`

- 본 시스템의 **기본 문장 유형**이다 (test 분포 약 74%, `step4r_bigru_attention_schema_results.md` §3).
- 어조: "~에 가까운 패턴이 나타납니다", "~ 가능성이 함께 나타납니다", "단일 유형으로 단정하기 어렵습니다".
- class_set 전체를 *공동 후보*로 제시하며, 어느 하나로 좁히지 않는다.

### 4.3 `low`

- 불확실성이 큼을 *명시적*으로 표면화.
- 가능한 후보군만 제시하고, 구체 오류 단정은 금지.
- 어조: "불확실성이 큽니다", "후보군 수준으로만 해석하는 것이 적절합니다".
- `anchor_unreliable` flag가 동반된 경우는 §6.6 참조.

### 4.4 `no_call`

- 분류 판단을 보류한다.
- 어조: "현재 손목 센서 신호만으로는 안정적인 설명을 제공하기 어렵습니다".
- 원인 후보를 *과도하게 설명하지 않는다* — 사용자에게 자세 교정 가이드를 *조작*하지 않는다.
- `no_call_reason`에 따라 짧은 사유 한 줄(자세 입력 누락 / 신호 강도 부족 / 동작 기준점 신뢰도 낮음)만 추가 가능.

---

## 5. class_set size별 문장 정책

`class_set_prediction`의 길이에 따라 분기.

| size | 정책 |
|---:|---|
| 0 | `no_call`로만 처리 (§4.4). 클래스 표현 어휘를 사용하지 않는다. |
| 1 | `class_set[0]`을 중심으로 설명. **단, `caption_confidence_level`이 hedged 이하이면 단정 금지** — 어조는 confidence 정책(§4)을 따른다. |
| 2 | "A 또는 B 중 하나"보다 **"A/B 경계 패턴"**, **"A 계열과 B 계열이 함께 나타나는 신호"**로 표현. 두 후보 중 하나를 좁히지 않는다. |
| 3 | `ambiguity_group` 중심 설명 (§6). 개별 class를 하나로 좁히지 않는다. 본 시스템에서 size=3은 `[C1, C5, C6]` 또는 `[C3, C4, C2]`만 가능 (Step 3 §4 whitelist). |

---

## 6. 주요 ambiguity group별 문장 정책

`ambiguity_group` 컬럼은 §2의 closed-vocabulary enum이다 (`reports/step4r/4rb_attention/step4r_bigru_attention_schema_outputs_calibrated.csv` 참조).

### 6.1 `confident_C2`

- 깊이 부족 계열 표현 가능.
- **절대값으로서의 squat depth를 측정한 것처럼 말하지 않는다.** "깊이 부족" 자체도 *직접 측정*이 아니라 *계열 패턴*으로 표현 (§2 vocabulary).

### 6.2 `within_group_c1_c5_c6`

- 정상 패턴(C1)과 좌우 무릎 관련 패턴(C5/C6)의 *경계*로 표현.
- **"knee valgus를 관찰했다", "한쪽 무릎이 안쪽으로 들어갔다" 등은 금지** (forbidden list, §3 of `step5_v2_forbidden_expressions.md`).
- 허용: "손목 센서 기준으로 정상 패턴과 좌우 비대칭 관련 패턴이 함께 나타나는 경계 신호".

### 6.3 `pair_c3_c4`

- C3/C4 *경계 패턴*으로 표현.
- 특정 복합 오류 단정 금지 (예: "C3로 진단된다" 금지).
- 허용: "복합 오류 후보 두 가지가 함께 나타나는 경계 신호".

### 6.4 `pair_plus_c2_absorption`

- C3/C4 후보와 *C2 흡수 가능성*을 함께 표현.
- 허용: "깊이 부족 계열 신호와 복합 오류 후보가 함께 나타남".
- C2를 흡수 가능성으로 *명시*해야 한다 (Step 3 §4의 필수 hedge — `pair_plus_c2_absorption` flag가 강제하는 표현).

### 6.4.1 C3 원 논문 정의와 caption-safe 축약 (§6.3 / §6.4 공통)

- C3는 원 논문에서 **"insufficient depth with posterior tilting and knee valgus"** 로 정의되지만, caption에서는 *posterior tilting* 과 *knee valgus* 를 직접 측정한 것처럼 쓰지 **않는다**. wrist IMU는 골반 경사도, 무릎 정렬을 직접 측정하지 않기 때문이다 (`reports/step4_research_reframing.md` §4.2 관측 가능성 한계).
- 따라서 C3 / C4 관련 모든 ambiguity 표현은 **"복합 오류 후보"** 또는 **"깊이 부족 계열 신호와 복합 오류 후보가 함께 나타남"** 으로만 표현한다 (구체 어휘는 `step5_v2_caption_vocabulary.md` §1.2 / §3.1 참조).
- "골반 후방 경사", "posterior tilting", "knee valgus", "무릎이 안쪽으로 들어감" 등은 모두 사용자 caption에서 금지된다 (`step5_v2_forbidden_expressions.md` §1).
- 이는 sensor-grounded caption policy의 *의도적 축약*이며, 본 시스템이 의학·생체역학적 단정의 위험을 구조적으로 회피하기 위한 설계이다 (정책 §8 / reframing §8.1).

### 6.5 `low_confidence` (catch-all 4번째 rule)

- §4 어떤 분기도 통과하지 못해 `no_call=true`로 떨어진 경우.
- §4.4의 `no_call` 문장으로 처리.
- 사유는 `low_confidence_no_class_set`로 짧게 표시.

### 6.6 `anchor_unreliable` (위의 어떤 group에도 동반될 수 있는 flag)

- "동작 기준점이 불안정하여 설명 신뢰도가 낮습니다"로 표현 가능.
- **자세 오류 원인으로 단정하지 않는다** — anchor unreliable은 *센서 신호의 특성*이지 사용자의 자세가 잘못됐다는 신호가 아니다.
- `caption_confidence_level`이 이미 `hedged → low` 또는 `confident → hedged`로 강등된 상태에서 동반되므로, 어조는 강등된 level을 따른다.

---

## 7. posture 표현 정책

| `posture_canonical` | 사용자 표현 |
|---|---|
| `SA` | 팔을 앞으로 둔 자세 |
| `CA` | 팔을 교차한 자세 |
| `HW` | 손을 허리에 둔 자세 |

- posture는 *센서 신호 해석 조건*으로 표현한다 ("팔을 앞으로 둔 자세에서 기록된 손목 IMU 신호는 …").
- **자세 자체를 squat 오류로 말하지 않는다** — 본 데이터의 posture는 측정 프로토콜의 일부이지 사용자가 잘못된 자세를 *취한* 것이 아니다.
- HW에서 anchor 신뢰도가 구조적으로 낮은 경향(`reports/step2/step25_final_synthesis.md` §4)을 caption에서 직접 인용하지 않는다 — 이 정보는 schema 단계에서 이미 `anchor_unreliable` flag로 흡수되었고, caption은 그 flag만 본다.

---

## 8. 금지 원칙

1. **측정하지 않은 생체역학 변수를 직접 측정한 것처럼 말하지 않는다.** wrist IMU는 다음을 직접 측정하지 않는다 — knee valgus, hip-knee-ankle 정렬, 골반 경사, 절대 squat depth, 관절각, 좌/우 비대칭의 방향.
2. **의학적 진단·부상 위험·치료 권고를 하지 않는다.** caption은 의료 도구가 아니다.
3. **attention 분석 결과를 사용자 caption에 직접 근거로 쓰지 않는다.** "모델의 attention이 ascending phase에 집중되어 …" 류 표현은 일반 사용자 caption에서 invalid이다 (§3.3 technical variant 한정 허용).
4. **schema에 없는 정보를 추가하지 않는다.** LLM은 새 클래스, 새 ambiguity flag, 새 confidence level, 새 quantitative claim을 *발명하지 않는다*.
5. **class_set을 좁히지 않는다.** schema가 size 2~3을 출력했으면 caption도 모두 후보로 표현해야 한다 (§5).

---

## 9. caption output format

caption layer는 rep마다 다음 4개 표면 텍스트 필드를 산출한다.

| 필드 | 의미 |
|---|---|
| `caption_ko` | 한국어 본문. §4 ~ §6의 정책 위에서 작성된 1 ~ 3문장. |
| `confidence_phrase` | `caption_confidence_level`을 표현하는 짧은 어구 (§2 vocabulary `confidence_*`). |
| `uncertainty_phrase` | `uncertainty_flags` / `ambiguity_group`을 표현하는 짧은 어구 (§2 vocabulary `ambiguity_*`). |
| `limitation_phrase` | "손목 센서 기준의 추정" 류의 sensor-grounded notice. 길이 부담을 줄이기 위해 caption에 항상 포함되지 않을 수 있음. |

추가 trace 필드 (validation 용도, 사용자에게 안 보임):

| 필드 | 의미 |
|---|---|
| `used_schema_fields` | 이 caption이 *입력 schema 어떤 필드를 참조했는지* 자체 보고. automatic validation의 누락 점검에 사용. |

---

## 10. Step 6_v2 / Step 7_v2로 넘기는 결정 사항

본 정책 문서는 다음을 후속 단계로 *명시 위임*한다 (본 단계에서 결정하지 않음).

- **Step 6_v2 (prompt finalization)**:
  - prompt에서 사용할 schema field 목록 확정 (§2의 부분집합).
  - class별 vocabulary의 최종 wording (`step5_v2_caption_vocabulary.md`의 후보 어휘에서 선택).
  - forbidden expression 목록의 최종 commit (`step5_v2_forbidden_expressions.md` 기반).
  - automatic validation으로 검사할 항목 commit (forbidden token / forbidden semantic / confidence-level mismatch / no_call contradiction / class_set narrowing / unsupported biomechanical claim).
- **Step 7_v2 (caption generation prototype)**:
  - LLM 호출 코드 작성.
  - 본 정책 위반 시 retry / fallback policy.
  - per-rep caption batch 산출.

본 문서가 commit되지 않은 채 Step 6_v2 / 7_v2가 시작되어서는 안 된다.

---

## 11. 본 단계가 명시적으로 결정하지 않는 사항

- caption 한 문장당 길이 상한 / 하한.
- LLM 모델 선택 (어느 한국어 LM을 쓸지).
- caption_ko 안에서 confidence_phrase / uncertainty_phrase / limitation_phrase의 *순서*.
- automatic validation의 LLM judge 사용 여부 (Step 8_v2에서 결정).
- human review의 부속 사용성 검토 일정 (main evaluation에서 제외 — `reports/step4_research_reframing.md` §7).

---

*본 정책 문서는 caption을 만들지 않으며, 어떤 LLM도 호출하지 않으며, 코드도 작성하지 않는다. 기존 Step 5 ~ 8 산출물(`reports/step5/` ~ `reports/step8/`)은 수정하지 않으며, 본 문서는 `reports/step5_v2/` 아래에만 새로 생성된다.*
