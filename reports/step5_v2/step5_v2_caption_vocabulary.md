# Step 5_v2 — Caption Vocabulary

본 문서는 Step 5_v2 schema-grounded caption의 **어휘표**를 정의한다. 본 어휘는 `step5_v2_schema_grounded_caption_policy.md` 위에서 작동하며, Step 6_v2(prompt finalization)에서 최종 wording이 commit될 때까지 *후보 어휘*로 다룬다.

본 어휘에 없는 표현은 caption에 사용할 수 없다. 새로운 표현이 필요하다면 본 문서를 *명시적으로 갱신*해야 한다.

---

## 1. Class label vocabulary

각 class는 **의학·생체역학적 진단**이 아니라 **sensor-grounded phrase**로 표현한다. 손목 IMU는 좌우 무릎의 방향성, 무릎 정렬, 골반 경사를 직접 측정하지 못하므로, *방향*을 단정하는 표현(예: "한쪽", "반대쪽", "양측")은 일반 사용자 caption에서 사용하지 않는다.

| class | 사용자 표현 후보 |
|---|---|
| `C1` | 정상 패턴에 가까운 신호 |
| `C2` | 깊이 부족 계열로 해석 가능한 신호 |
| `C3` | 깊이 부족과 복합 오류 후보가 함께 나타나는 신호 |
| `C4` | 무릎 관련 오류 후보가 섞인 신호 |
| `C5` | 무릎 관련 오류 후보가 섞인 신호 |
| `C6` | 무릎 관련 오류 후보가 섞인 신호 |

**중요한 통일 규칙 (§5.1.1):**

- C4 / C5 / C6 는 본 데이터셋의 클래스 정의상 좌/우/양측 무릎 관련 오류이지만, **wrist IMU만으로는 좌우 방향을 구분할 수 없다.** 따라서 사용자 caption에서 C4/C5/C6는 모두 **"무릎 관련 오류 후보가 섞인 신호"**로 통일한다.
- "한쪽 무릎", "반대쪽 무릎", "양측 무릎" 같은 방향 표현을 사용자 caption에 쓰지 않는다.
- C5와 C6가 같은 caption_set 안에 동시에 등장할 때(예: `class_set = ["C1","C5","C6"]`), 두 클래스를 모두 "무릎 관련 오류 후보 패턴"으로 묶어 표현한다.

### 1.1 ambiguity 발생 시 추가 정책 (정책 §6과 일관)

- C4 / C5 / C6 단일 단정형 표현을 *피하고*, **"무릎 관련 오류 후보"** 또는 **"좌우 비대칭 관련 패턴"**으로 *낮춰* 표현한다.
- "무릎이 안쪽으로 들어갔다" 같은 *방향이 들어간 단정 표현*은 어떠한 confidence 수준에서도 금지 (`step5_v2_forbidden_expressions.md` §1).

### 1.2 원 논문 정의와 caption-safe 표현의 대응 관계

caption layer는 원 논문의 class definition을 사용자에게 *그대로 번역하지 않는다*. 원 정의에 등장하는 의학·생체역학 용어는 wrist IMU의 관측 가능성 한계에 맞게 sensor-grounded phrase로 *변환*된다.

| class | 원 논문 정의 | caption-safe 표현 | 주의 |
|---|---|---|---|
| `C1` | Normal | 정상 패턴에 가까운 신호 | — |
| `C2` | Insufficient depth | 깊이 부족 계열로 해석 가능한 신호 | absolute squat depth는 wrist IMU가 직접 측정하지 않음 |
| `C3` | Insufficient depth with posterior tilting and knee valgus | 깊이 부족과 복합 오류 후보가 함께 나타나는 신호 | posterior tilting과 knee valgus는 wrist IMU가 직접 측정하지 않으므로 사용자 caption에 직접 쓰지 않음 |
| `C4` | Left-knee valgus | 무릎 관련 오류 후보가 섞인 신호 | left/right 방향을 직접 표현하지 않음 |
| `C5` | Right-knee valgus | 무릎 관련 오류 후보가 섞인 신호 | left/right 방향을 직접 표현하지 않음 |
| `C6` | Both-knee valgus | 무릎 관련 오류 후보가 섞인 신호 | both-knee / 양측 등 방향성 단정을 직접 표현하지 않음 |

**보충 설명:**

- "복합 오류 후보"는 C3의 *posterior tilting + knee valgus* 두 성격을 직접 명명하지 않고 안전하게 낮춘 표현이다.
- caption layer는 원 논문 class definition을 *그대로* 번역하지 않는다.
- caption layer는 wrist IMU의 관측 가능성 한계에 맞게 *sensor-grounded phrase*로 변환한다.
- 의학 용어(knee valgus, posterior tilting 등)는 `step5_v2_forbidden_expressions.md` §1에 의해 caption 출력에서 차단된다.

---

## 2. Confidence vocabulary

`caption_confidence_level` 값에 따라 사용 가능한 어조 어구.

| level | confidence_phrase 후보 |
|---|---|
| `confident` | "비교적 일관되게 나타납니다", "상대적으로 안정적인 신호입니다", "손목 센서 기준으로 비교적 안정적으로 관찰됩니다" |
| `hedged` | "~에 가까운 경향이 있습니다", "~ 가능성이 함께 나타납니다", "단정하기 어렵습니다", "경계 신호로 보입니다" |
| `low` | "불확실성이 큽니다", "후보군 수준으로만 해석하는 것이 적절합니다", "약한 신호입니다" |
| `no_call` | "현재 손목 센서 신호만으로는 안정적인 설명을 제공하기 어렵습니다", "신호만으로 판단을 내리지 않습니다" |

**주의:**

- confident에서도 "확실히", "명확히", "진단된다" 등은 사용 금지.
- 모든 confident 문장에는 `limitation_phrase`(§5)를 1회 이상 함께 둔다.

---

## 3. Ambiguity vocabulary

`uncertainty_flags` / `ambiguity_group`을 표현할 때 사용 가능한 어구.

| 어구 | 사용 맥락 |
|---|---|
| 경계 패턴 | class_set size 2 또는 3, 두 후보가 비등하게 나옴 |
| 후보군 | low / hedged 단계에서 단정 대신 후보 나열 |
| 함께 나타남 | 두 종류 신호가 동시에 보일 때 (특히 `pair_plus_c2_absorption`) |
| 단일 유형으로 좁히기 어려움 | hedged의 기본 어구 |
| 관측 한계 | 손목 센서가 직접 측정하지 못하는 양에 한해 사용 |
| 손목 센서 기준 | sensor-grounded notice의 기본 형식 |

### 3.1 ambiguity_group ↔ uncertainty_phrase 매핑 후보

| ambiguity_group | uncertainty_phrase 후보 |
|---|---|
| `confident_C2` | "" (빈 문자열) 또는 "깊이 부족 계열에 가까운 패턴" |
| `within_group_c1_c5_c6` | "정상 패턴과 무릎 관련 오류 후보 패턴이 함께 나타나는 경계 신호" |
| `pair_c3_c4` | "복합 오류 후보 두 가지가 함께 나타나는 경계 신호" |
| `pair_plus_c2_absorption` | "깊이 부족 계열 신호와 복합 오류 후보가 함께 나타남" |
| `no_call` | "동작 기준점 신뢰도가 낮음" 또는 "신호 강도가 부족함" (no_call_reason에 따라) |
| `uncategorized` | "단일 유형으로 좁히기 어려움" |

`anchor_unreliable` flag가 추가로 부착된 경우, 위 phrase 끝에 "(동작 기준점이 불안정하여 설명 신뢰도가 낮습니다)"를 덧붙일 수 있다.

---

## 4. Posture vocabulary

| `posture_canonical` | 사용자 표현 |
|---|---|
| `SA` | 팔을 앞으로 둔 자세 |
| `CA` | 팔을 교차한 자세 |
| `HW` | 손을 허리에 둔 자세 |

### 4.1 사용 패턴

caption 본문 시작부의 표준 형식 (정책 §7):

> "팔을 앞으로 둔 자세에서 기록된 손목 IMU 신호는 …"
> "팔을 교차한 자세에서 측정된 손목 IMU 패턴은 …"
> "손을 허리에 둔 자세에서, …"

**금지:**

- 자세 자체를 *오류*로 표현하지 않는다 ("팔을 잘못 둔 채로 squat을 했다" 등 금지).
- HW에 대해 "anchor가 불안정한 자세"라고 직접 부르지 않는다 — 이 정보는 schema 단계에서 `anchor_unreliable` flag로 흡수되어 있고, caption은 flag만 본다.

---

## 5. Limitation vocabulary

sensor-grounded notice를 표현하는 어구. 본 어구는 caption의 신뢰도 표시이자 의학적 책임 회피 장치이다.

| 어구 | 사용 강도 |
|---|---|
| "손목 센서는 무릎 각도나 골반 움직임을 직접 측정하지 않습니다" | 강 — 사용자에게 sensor 한계를 충분히 알릴 필요가 있을 때 |
| "따라서 이 설명은 손목 IMU 신호에 근거한 추정입니다" | 중 — 거의 모든 caption에 한 번은 등장 가능 |
| "정확한 자세 평가는 영상 또는 전문가 평가와 함께 해석해 주십시오" | 강 — 의료/평가 맥락에서 |
| "손목 센서 기준의 추정입니다" | 약 — 짧은 caption의 한 줄 추가 |

### 5.1 사용 정책

- caption이 너무 길어지지 않도록, 모든 caption에 모든 limitation phrase를 *전부* 넣지 않는다. 보통 **1개**의 limitation phrase를 추가한다.
- `confident` level에서는 강도 강~중 phrase를 1개 권장.
- `hedged` / `low`에서는 약 phrase로도 충분.
- `no_call` 메시지에는 limitation phrase를 *생략*해도 된다 — no_call 메시지 자체가 이미 한계 표명이다.
- LLM은 본 어구를 그대로 복제하거나, *의미적으로 동등한 한국어*로 미세하게 변형할 수 있다 (단, 본 정책 §8의 금지 원칙은 그대로 유지).

---

## 6. 본 어휘에 *없는* 표현은 사용 금지

본 문서에 명시된 vocabulary 외의 표현 — 특히 의학·생체역학·해부학 용어 — 은 caption에 사용하지 않는다. 새 표현이 필요하다면 본 문서를 갱신한 뒤 사용한다.

명시적 금지 표현은 별도 문서 `step5_v2_forbidden_expressions.md`에 정리한다.

---

*본 어휘표는 후보이며, Step 6_v2 prompt finalization에서 사용 어휘가 commit된다. 본 문서는 caption을 만들지 않으며, 어떤 LLM도 호출하지 않는다.*
