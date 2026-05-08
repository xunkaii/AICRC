# Step 5_v2 — Forbidden Expressions

본 문서는 Step 5_v2 schema-grounded caption에서 **금지되는 한국어 표현**과 그 안전한 **대체 표현**, 그리고 자동 검증에서 사용할 **검사 규칙 후보**를 정리한다.

`step5_v2_schema_grounded_caption_policy.md` §8의 금지 원칙(측정하지 않은 생체역학 변수의 단정 / 의학적 진단·부상 위험·치료 권고 / attention 직접 노출 / schema에 없는 정보 추가 / class_set narrowing)과 일관된다.

---

## 1. 금지 표현 목록

LLM이 caption 본문(`caption_ko`) 또는 `confidence_phrase` / `uncertainty_phrase` / `limitation_phrase`의 어느 곳에도 다음 표현을 사용해서는 안 된다.

| # | 금지 표현 | 금지 사유 |
|---:|---|---|
| 1 | "무릎이 안쪽으로 들어갔다" | wrist IMU는 무릎 방향을 직접 측정하지 않음. 방향 단정. |
| 2 | "knee valgus가 발생했다" | wrist IMU는 knee valgus를 측정하지 않음. 의학 용어 사용. |
| 3 | "골반이 후방 경사됐다" | wrist IMU는 골반 운동을 직접 측정하지 않음. |
| 4 | "무릎 정렬이 틀어졌다" | hip-knee-ankle 정렬은 wrist IMU의 직접 측정 대상이 아님. |
| 5 | "관절각이 감소했다" | 관절각의 직접 측정 부재. |
| 6 | "정확한 깊이가 부족하다" | absolute squat depth의 직접 측정 부재 (acc_z 적분은 drift). |
| 7 | "부상 위험이 높다" | 의료 위험 단정. 책임 범위 초과. |
| 8 | "치료가 필요하다" | 의료 권고. 본 시스템 범위 외. |
| 9 | "진단된다" | 의학적 진단 단정. 본 시스템은 진단 도구가 아님. |
| 10 | "확실하다" / "확실히" | 본 schema 어떤 confidence level에서도 부적절. |
| 11 | "명확히 잘못됐다" | 단정 + 가치 판단. 정책 §4 confident 정의에서도 금지. |
| 12 | "전문가가 확인한 것과 같다" | 외부 권위 사칭. 검증 부재. |
| 13 | "후방 경사" | 골반 운동은 wrist IMU의 직접 측정 대상이 아님. C3 원 정의의 posterior tilting을 직접 표현. |
| 14 | "posterior tilting" | C3 원 논문 정의의 영문 용어 직접 노출. 손목 센서가 직접 측정하지 않음. |
| 15 | "골반이 뒤로 말렸다" | #13의 한국어 변형. 골반 운동 단정. |
| 16 | "골반이 뒤로 빠졌다" | #13의 한국어 변형. 골반 운동 단정. |
| 17 | "무릎이 모인다" | 무릎 내전을 표현하는 동사 구문. wrist IMU는 무릎 방향을 직접 측정하지 않음. knee valgus의 우회 표현으로 사용 금지. |
| 18 | "무릎이 안으로 무너진다" | #17과 동일 사유. dynamic knee valgus의 비공식 한국어 변형. |

---

## 2. 허용 대체 표현

각 금지 표현에 대응하는 안전한 대체 표현. caption layer는 본 매핑을 우선 적용한다.

| 금지 | 대체 |
|---|---|
| "무릎이 안쪽으로 들어갔다" | "손목 센서 기준으로 무릎 관련 오류 후보와 유사한 신호가 나타납니다" |
| "knee valgus가 발생했다" | "좌우 비대칭 관련 패턴이 함께 나타나는 신호" |
| "골반이 후방 경사됐다" | (해당 양은 caption에 도입하지 않음 — 측정되지 않은 변수) |
| "무릎 정렬이 틀어졌다" | "무릎 관련 오류 후보 패턴" 또는 "좌우 비대칭 관련 패턴" |
| "관절각이 감소했다" | (해당 양은 caption에 도입하지 않음 — 측정되지 않은 변수) |
| "정확한 깊이가 부족하다" | "깊이 부족 계열로 해석 가능한 손목 IMU 패턴이 나타납니다" |
| "부상 위험이 높다" | (의료 위험 표현 자체를 도입하지 않음) |
| "치료가 필요하다" | (의료 권고 자체를 도입하지 않음) |
| "진단된다" | "~에 가까운 경향이 나타납니다" |
| "확실하다" / "확실히" | "비교적 일관되게 나타납니다" (`confident` 한정) |
| "명확히 잘못됐다" | (단정형 평가 자체를 도입하지 않음) |
| "전문가가 확인한 것과 같다" | "정확한 자세 평가는 영상 또는 전문가 평가와 함께 해석해 주십시오" |
| "후방 경사" / "posterior tilting" | "복합 오류 후보가 함께 나타나는 신호" |
| "골반이 뒤로 말렸다" / "골반이 뒤로 빠졌다" | "깊이 부족 계열 신호와 복합 오류 후보가 함께 나타남" |
| "무릎이 모인다" / "무릎이 안으로 무너진다" | "손목 센서 기준으로 무릎 관련 오류 후보와 유사한 신호" |

### 2.1 C3 원 논문 정의의 처리

- C3의 원 논문 정의는 *insufficient depth with posterior tilting and knee valgus* 이지만, 위 §1 #13 ~ #18에 따라 *posterior tilting* 과 *knee valgus* 의 직접 표현은 사용자 caption에서 금지된다.
- C3 관련 caption은 `step5_v2_caption_vocabulary.md` §1.2 표에 따라 **"깊이 부족과 복합 오류 후보가 함께 나타나는 신호"** 로만 표현한다.
- 본 매핑은 schema-grounded caption policy의 *의도적 축약*이며, 의학·생체역학 단정의 위험을 구조적으로 회피한다.

---

## 3. 자동 검증 규칙 후보

Step 8_v2(automatic schema-caption validation)에서 사용할 검사 규칙. 본 단계에서는 *규칙 종류*만 고정하고, 임계값/구현 세부는 Step 8_v2에서 commit한다.

### 3.1 forbidden token exact match

- §1 표현이 caption 텍스트에 *문자열 그대로* 포함되는지 검사.
- 발생 시 즉시 invalid.

### 3.2 forbidden semantic pattern

- §1을 회피하기 위한 변형 — 예: "무릎이 안쪽으로 무너지는", "무릎이 모이는 패턴", "골반이 뒤로 빠지는" 등.
- 키워드 조합(예: "무릎" + "안쪽" + 동작 동사) 또는 LLM judge로 검사.
- LLM judge를 쓰는 경우, judge의 출력을 closed-vocabulary 라벨(`pass` / `fail` + reason enum)로만 받아 다시 *판단자*가 되는 위험을 차단한다.

### 3.3 confidence-level mismatch

- caption의 단정도(definiteness)가 schema의 `caption_confidence_level`과 *반대 방향*인 경우 invalid.
- 예: schema가 `low`인데 caption에 "비교적 일관되게 나타납니다"가 등장.
- 예: schema가 `confident`(`["C2"]`)인데 caption이 "후보군 수준으로만 …"으로 끝남.

### 3.4 no_call contradiction

- `no_call=true`인데 caption이 *분류 결과를 단정*하는 경우.
- 예: `no_call=true` + caption에 클래스 어휘(C1/C2/… 또는 정책 §1의 사용자 표현 후보)가 등장 → invalid.

### 3.5 class_set narrowing

- schema의 `class_set` 길이 ≥ 2인데, caption이 *단일 후보*로 좁힌 경우 invalid.
- 예: `class_set=["C1","C5","C6"]`인데 caption이 "C5에 가장 잘 맞는 패턴" → invalid.
- 검사 방법: caption 안에서 사용된 class label 후보를 추출(§1의 사용자 표현 후보로부터 키워드 추출) 후 schema class_set과 일치하는지 비교.

### 3.6 unsupported biomechanical claim

- §1.1 ~ §1.6 같은 *측정되지 않은 양*에 대한 단정 (knee valgus / hip tilt / 정렬 / 관절각 / absolute depth / 좌우 방향 단정).
- forbidden token + forbidden semantic pattern 검사를 합친 형태.

### 3.7 attention term leakage

- caption 본문에 attention 분석 어휘 직접 노출 (예: "attention", "phase", "ascending phase", "anchor", "peak timestep" 등).
- attention 정보는 일반 사용자 caption에서 invalid (정책 §3.2).

---

## 4. 위반 시 처리 (참고 — 본 단계에서 commit하지 않음)

- forbidden token / forbidden semantic / unsupported biomechanical claim → 즉시 invalid → caption retry 또는 fallback (Step 7_v2).
- confidence mismatch / no_call contradiction / class_set narrowing → invalid → retry.
- attention term leakage → invalid → retry.
- 모든 위반은 Step 8_v2의 metric에 반영된다 (forbidden vocabulary occurrence rate, schema-faithfulness fidelity rate 등).

---

*본 문서는 후보이며, Step 6_v2 prompt finalization에서 commit된다. 본 단계에서는 caption을 만들지 않고, 어떤 LLM도 호출하지 않는다.*
