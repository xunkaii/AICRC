# Step 5_v3 — Forbidden Expressions (clinical pool 한정)

본 문서는 Step 5_v3에서 **여전히 금지되는** 표현을 정의한다. Step 5_v2 정책의 18개 금지 표현 중 *clinical vocabulary 관련 항목들은 완화*되었고, *의학적 책임 / 모델 내부 노출 관련 항목들은 그대로 유지*된다.

본 문서는 `step5_v3_caption_policy.md` §5의 상세 정의이다.

---

## 1. v2에서 *완화된* 표현 (v3에서 허용)

| # | v2 금지 표현 | v3 허용 여부 | 비고 |
|---:|---|---|---|
| 1 | "무릎이 안쪽으로 들어갔다" | ⚠️ 우회 표현으로 허용 | "무릎이 내측으로 쏠리는" (J-C3-3, J-C4-1, J-C5-1, J-C6-1) |
| 2 | "knee valgus가 발생했다" | ⚠️ 표현 자체 허용 (단정 어조 제거) | "knee valgus 패턴이 나타납니다" 정도까지 허용 |
| 3 | "골반이 후방 경사됐다" | ⚠️ 우회 표현으로 허용 | "골반이 후방으로 기우는" (J-C3-1) |
| 4 | "무릎 정렬이 틀어졌다" | ⚠️ 표현 자체 허용 | "무릎 정렬에 영향" (J-C4-2, J-C5-2) |
| 13 | "후방 경사" | ✅ 허용 | 자유롭게 사용 |
| 14 | "posterior tilting" | ✅ 허용 (영문 가능) | hedging 어조 필수 |
| 15, 16 | "골반이 뒤로 말렸다 / 빠졌다" | ⚠️ 우회 표현으로 허용 | "골반이 후방으로 기우는" 권장 |
| 17, 18 | "무릎이 모인다 / 안으로 무너진다" | ⚠️ 우회 표현 권장 | "무릎이 내측으로 쏠리는" 권장 |
| (좌/우/양측) | (FORBIDDEN_DIRECTION_TOKENS) | ✅ 허용 | C4(좌측) / C5(우측) / C6(양측) 모두 직접 명시 가능 |

→ **단정 어조는 여전히 금지**. "확실히 무릎 정렬이 틀어졌다" → "무릎 정렬에 영향을 주는 경향이 보입니다"로 변환 필요.

---

## 2. v3에서도 *여전히 금지*되는 표현

다음 표현은 어떤 confidence level에서도 caption에 등장해서는 안 된다.

### 2.1 의학적 진단 단정 (v2 #9~#12와 동일)

| # | 금지 표현 | 사유 |
|---:|---|---|
| F1 | "진단된다" | 본 시스템은 진단 도구가 아님 |
| F2 | "확실하다" / "확실히" | 어떤 confidence level에서도 부적절 (no_call 포함) |
| F3 | "명확히 잘못됐다" | 단정 + 가치 판단 |
| F4 | "전문가가 확인한 것과 같다" | 외부 권위 사칭, 검증 부재 |

### 2.2 부상 / 치료 / 교정 권고 (v2 #7~#8와 동일)

| # | 금지 표현 | 사유 |
|---:|---|---|
| F5 | "부상 위험이 높다" | 의료 위험 단정, 책임 범위 초과 |
| F6 | "부상으로 이어진다" | 인과 단정, 책임 범위 초과 |
| F7 | "치료가 필요하다" | 의료 권고, 본 시스템 범위 외 |
| F8 | "교정해야 한다" | 처방 단정 |
| F9 | "고쳐야 한다" | 처방 단정 |

### 2.3 attention / 모델 내부 노출 (v2와 동일)

| # | 금지 토큰 | 사유 |
|---:|---|---|
| F10 | attention / Attention / 어텐션 | attention 정보는 사용자 caption에 부적절 |
| F11 | ascending phase / descending phase / bottom_transition | 모델 내부 분석 단위 |
| F12 | anchor / Anchor | 모델 내부 용어 |
| F13 | peak timestep | 모델 내부 단위 |
| F14 | predictive entropy / attention entropy / 엔트로피 | 모델 내부 metric |
| F15 | posterior / 포스테리어 | 모델 내부 확률 |
| F16 | logit / 로짓 | 모델 내부 값 |
| F17 | softmax | 모델 내부 함수 |
| F18 | temperature | 모델 내부 hyperparam |

### 2.4 schema 외 추론 (v2와 동일)

| # | 금지 패턴 | 사유 |
|---:|---|---|
| F19 | raw class label "C1"~"C6" (caption 본문 내) | 모델 내부 라벨 직접 노출 금지 |
| F20 | schema의 `class_set` 외 클래스 어휘 | schema-grounded 원칙 위반 |
| F21 | 손/팔 외 신체 부위(예: "얼굴", "복근 정의") | wrist IMU 측정 범위 외 |
| F22 | 절대 깊이 수치 ("X cm까지 내려가지 못함") | absolute depth 측정 불가 (드리프트) |

### 2.5 class_set narrowing (v2와 동일)

`class_set` 길이 ≥ 2인데 caption이 *단일 후보*로 좁힌 경우 invalid. `step5_v3_pool_design.md` §4 다중 클래스 결합 규칙으로 처리.

### 2.6 no_call contradiction (v2와 동일)

`no_call=true`인데 caption에 클래스 어휘(J-C*-* 또는 클래스 본명)가 등장하면 invalid.

---

## 3. 자동 검증 규칙 (Step 8_v2에서 사용)

Step 8_v2 validator는 v2와 v3 *양쪽*에 적용된다. v3에서는 다음 검사만 활성화:

| 검사 | v2에서 활성 | v3에서 활성 |
|---|:---:|:---:|
| forbidden token exact match (F1~F18) | ✅ | ✅ |
| direction token (왼쪽/오른쪽/양측/한쪽/반대쪽) | ✅ | ❌ (v3 허용) |
| biomech token (골반/관절각/정렬) | ✅ | ❌ (v3 허용) |
| 관찰됩니다/관찰된다/관찰되었다/관찰됨 | ✅ | ❌ (v3 허용) |
| attention term leakage (F10~F18) | ✅ | ✅ |
| no_call contradiction | ✅ | ✅ |
| class_set narrowing | ✅ | ✅ |
| posture phrase consistency | ✅ | ✅ (단, POSTURE_POOL 어느 entry라도 포함하면 통과) |
| confidence-level mismatch | ✅ | ✅ |
| used_schema_fields whitelist | ✅ | ✅ |
| limitation phrase 존재 (no_call 외) | ✅ | ✅ |
| raw class label (C1~C6) | ✅ | ✅ |
| schema 외 클래스 추론 | ✅ | ✅ |
| 부상/치료/교정 (F5~F9) | ✅ | ✅ |

→ v3 validator는 v2의 모든 *책임 관련* 검사를 그대로 유지하고, *어휘 관련* 검사만 완화한다.

---

## 4. 본 문서가 변경되는 조건

- 교수님 검토에서 추가 금지 표현이 결정될 때
- Step 7_v3 결과에서 *반복적인 표현 위반*이 발견되어 새 금지 표현이 필요할 때
- 새 attention/모델 내부 어휘가 도입될 때 (예: 새 모델 분석 단위)

---

## 변경 이력

- **2026-05-18**: 초안 작성. v2의 18개 금지 표현 중 11개 완화, 7개 유지 + 새 책임/처방 금지 표현 5개 추가.
