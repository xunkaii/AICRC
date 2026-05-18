# Step 5_v3 — Caption Policy (clinical pool 도입)

본 문서는 Step 5_v3 caption layer의 **새 정책**을 정의한다. Step 5_v2 정책은 *그대로 보존*되며, 본 정책은 **v2와 병렬로** 운영된다.

- 기존: `reports/step5_v2/step5_v2_schema_grounded_caption_policy.md` (sensor observability 우선 — 보수)
- 신규: 본 문서 (**clinical richness 우선 — 완화**)

---

## 1. 본 분기를 만든 이유

Step 5_v2 정책은 wrist IMU의 직접 측정 가능성 한계를 우선시하여, 의학·생체역학 용어 및 좌/우 방향성 단정을 *전면 금지*했다. 결과적으로 caption은 안전하지만 표현이 단조롭고, C4/C5/C6를 구분하지 못한다.

본 v3은 그 trade-off의 *반대 끝점*을 탐색한다:

| 축 | v2 | v3 |
|---|---|---|
| 측정 불가량 단정 | 금지 | **허용** (단, hedging 어조 필수) |
| 좌/우/양측 방향성 | 금지 | **허용** (data class가 좌/우를 구분하므로 caption에도 반영) |
| 의학·해부학 용어 | 금지 | **허용** (knee valgus, posterior tilt, 고관절 내회전 등) |
| 코치 피드백 톤 | 금지 | **허용** (관절 기여 패턴 중심 설명) |
| 의학적 진단 단정 | 금지 | **금지 유지** (`step5_v3_forbidden_expressions.md` 참조) |
| 치료/부상 위험 권고 | 금지 | **금지 유지** |
| attention·logit·posterior 노출 | 금지 | **금지 유지** |
| schema에 없는 클래스 추론 | 금지 | **금지 유지** |
| class_set narrowing | 금지 | **금지 유지** |
| no_call 단정 | 금지 | **금지 유지** |

→ 본 정책은 *clinical vocabulary 허용*만 다르며, 나머지 안전 제약은 모두 유지된다.

---

## 2. 본 정책의 적용 범위

- 적용: **Step 7_v3 caption generation only**.
- 미적용: Step 7_v2 (그대로 보존). Step 4R-C clinical corpus (이미 별도 clinical layer, 본 정책과 무관).
- Step 8_v2 validation은 **v2와 v3 양쪽**에 적용되어 schema_faithfulness / forbidden vocabulary occurrence 비교 metric을 생성한다.

---

## 3. caption 생성 원칙

### 3.1 schema-grounded (v2와 동일)

caption은 schema의 사실에만 근거하며, schema에 없는 클래스를 추론하지 않는다. `class_set` / `caption_confidence_level` / `no_call` / `uncertainty_flags`는 반드시 반영한다.

### 3.2 pool 기반 표현 (v3 신규)

caption은 **`step5_v3_pool_design.md`의 POSTURE_POOL × JOINT_POOL 조합**으로 구성한다. 단일 고정 phrase가 아니라, `(participant_id, sample_id)` hash 기반 결정론적 permutation으로 다양화한다.

### 3.3 hedging 어조 강제

clinical vocabulary를 사용하더라도, 어조는 `caption_confidence_level`에 종속된다:
- `confident` → "패턴이 일관되게 나타납니다", "신호가 비교적 안정적입니다"
- `hedged` → "~ 경향이 나타납니다", "~ 가능성이 함께 보입니다"
- `low` → "후보 수준의 신호입니다", "단일 패턴으로 좁히기 어렵습니다"
- `no_call` → "현재 손목 센서 신호만으로는 안정적인 설명이 어렵습니다" (clinical vocab 사용 금지)

### 3.4 sensor 한계 1회 명시 (v2와 동일)

`no_call` 외 모든 caption은 limitation phrase를 1회 포함한다. v3에서는 더 *길어진* limitation phrase가 권장된다:

> "본 설명은 손목 IMU 신호 기반의 추정이며, 무릎 각도·골반 운동·발목 가동 범위는 직접 측정되지 않습니다. 정확한 자세 평가는 영상 또는 전문가 평가와 함께 해석해 주십시오."

### 3.5 class_set narrowing 금지 (v2와 동일)

`class_set` 길이 ≥ 2이면 모든 후보를 caption에 표현. 단일 후보로 좁히지 않는다. clinical vocabulary 사용 시 두 후보의 *공통 관절 패턴*과 *상이 관절 패턴*을 모두 명시할 것을 권장.

### 3.6 no_call 처리 (v2와 동일)

`no_call=true`이면 클래스 어휘·관절 어휘를 caption 본문에 사용하지 않는다. 보류 메시지만 출력하되, 첫 문장은 반드시 `posture_pool`의 자세 표현으로 시작.

---

## 4. v2 대비 *완화된* 어휘 (예시)

다음 어휘는 v3에서 *허용*되며 caption 본문에 직접 등장 가능하다. 단 3.3의 hedging 어조와 결합되어야 한다.

| 카테고리 | v3 허용 어휘 (예시) |
|---|---|
| 관절 운동 | 고관절 굴곡, 무릎 굴곡, 발목 배측굴곡, 척추 중립, 요추 굴곡, 흉추 가동성 |
| 방향성 | 좌측, 우측, 양측, 좌우 비대칭 |
| 해부학적 정렬 | knee valgus, 무릎 정렬, posterior pelvic tilt, 골반 후방경사 |
| 가동 범위 | 가동 범위 제한, ROM, 발목 가동성 |
| 보상 패턴 | 고관절 내회전, 고관절 내전, 무릎 내측 쏠림 |
| 코치 피드백 | "발목 가동성이 동작 깊이에 영향을 줍니다", "체간 안정화가 요구됩니다" |

→ 정확한 pool entry는 `step5_v3_pool_design.md` 참조.

---

## 5. v3에서도 *여전히 금지*되는 어휘

`step5_v3_forbidden_expressions.md` 별도 문서에 명시. 핵심은 다음 4종.

1. **의학적 진단 단정** — "진단된다", "확실히 ~이다", "명확히 잘못됐다"
2. **부상/치료 권고** — "부상 위험이 높다", "치료가 필요하다", "교정해야 한다"
3. **attention / 모델 내부 노출** — attention, posterior, logit, entropy, phase, anchor
4. **schema 외 추론** — schema의 `class_set` 외 클래스, schema에 없는 부위(상지·하지 외)

---

## 6. 본 정책이 변경되는 조건

- 교수님 검토 피드백에서 추가 제거/추가 어휘가 결정될 때
- Step 7_v3 + Step 8_v2 validation에서 schema_faithfulness가 v2 대비 *크게* 떨어질 때 (잠정 기준: v2의 0.99 → v3의 0.90 미만이면 본 정책 재검토)
- 새 pool entry가 검토에서 추가될 때 (본 문서를 *덮어쓰지 않고* 변경 일자/내역 추가)

---

## 7. 산출물 위치 (v3 전용 — v2와 분리)

| 단계 | v2 경로 (보존) | v3 경로 (신규) |
|---|---|---|
| caption policy | `reports/step5_v2/step5_v2_schema_grounded_caption_policy.md` | `reports/step5_v3/step5_v3_caption_policy.md` (본 문서) |
| 어휘/pool | `reports/step5_v2/step5_v2_caption_vocabulary.md` | `reports/step5_v3/step5_v3_pool_design.md` |
| 금지 표현 | `reports/step5_v2/step5_v2_forbidden_expressions.md` | `reports/step5_v3/step5_v3_forbidden_expressions.md` |
| 시스템 프롬프트 | `reports/step6_v2/step6_v2_final_system_prompt.md` | `reports/step6_v3/step6_v3_final_system_prompt.md` |
| 생성 스크립트 | `scripts/generate_step7_v2_captions.py` | `scripts/generate_step7_v3_captions.py` (교수님 OK 후 작성) |
| 결과 | `reports/step7_v2/`, `data/step7_v2/` | `reports/step7_v3/`, `data/step7_v3/` (생성 후) |

---

*본 문서는 정책 commit이며, LLM을 호출하지 않고 caption도 만들지 않는다. 본 정책은 Step 5_v2 정책을 *대체하지 않고*, v3 분기 한정으로 적용된다. Step 1~4R, Step 5_v2~Step 8_v2 산출물은 본 문서로 인해 수정되지 않는다.*
