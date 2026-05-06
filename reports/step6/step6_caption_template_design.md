# Step 6 — Caption Template Design

이 문서는 Step 4 schema output과 Step 5 caption policy를 바탕으로,
caption layer가 사용할 수 있는 **template family**를 설계한다. 본
문서는 template wording의 *후보*를 정의하지만, 실제 sample별 caption은
생성하지 않는다. data/step4 CSV는 read-only 참고로만 사용했다.

본 문서는 다음에 종속된다 (충돌 시 종속 문서가 우선):

- `reports/step3/step3_output_schema_uncertainty_policy.md` — 출력 스키마 / 불확실성 정책
- `reports/step4/step4_final_summary.md` — Step 4 최종 상태
- `reports/step5/step5_caption_policy_design.md` — caption policy (이번 단계의 직접 상위 정책)
- `data/step4/step4_schema_outputs_raw.csv`, `data/step4/step4_schema_outputs_zscore.csv` — 입력 데이터(read-only)

본 문서가 **다루지 않는 것**:

- 실제 sample별 caption 생성 (Step 7 범위)
- 새 모델 학습 / threshold recalibration / schema output 재생성 / 새 feature 계산
- final wording commit (Step 6은 후보 template 단계)
- raw vs zscore 최종 branch commit

---

## 1. Purpose

| 항목 | 내용 |
|---|---|
| Step 6 목적 | Step 5 policy를 위반하지 않는 caption template family를 정의 |
| 본 문서가 다루는 것 | confidence-level / class_set / uncertainty flag / no_call / feature evidence별 template skeleton |
| 본 문서가 다루지 않는 것 | per-sample caption 생성, final deployed wording commit |
| Step 6의 위상 | template *후보* 설계 단계 — wording 검토 / 사용자 테스트 이전 |
| 위반 시 처리 | Step 5 §3/§4/§5/§6/§7/§8/§9 중 어느 한 항목이라도 위반하는 template은 본 문서에서 명시적으로 금지 표에 등록 |

본 문서의 모든 template skeleton은 **placeholder 기반**으로 작성된다.
실제 값 주입과 sample별 문장 생성은 Step 7에서 수행한다.

---

## 2. Dependencies

본 단계가 따르는 상위 산출물 / 정책.

| # | 종속 항목 | 출처 |
|---|---|---|
| 1 | output schema (rep당 emission 형태) | Step 3 §3 |
| 2 | class_set whitelist (8개 집합) | Step 3 §4 / Step 4 plan §9 |
| 3 | uncertainty_flags closed vocabulary (7개) | Step 3 §7 |
| 4 | caption_confidence_level enum (4개) | Step 3 §8 / Step 4 generator |
| 5 | anchor 두 단계 임계값 (suppression / no_call) | Step 3 §6 |
| 6 | feature evidence 측정 vs heuristic 분리 | Step 3 §5 / Step 4 decision memo |
| 7 | schema output 분포 (no_call ≈ 26%, hedged 다수) | Step 4 §9 |
| 8 | caption level 정책 (level별 표현 카테고리) | Step 5 §3 |
| 9 | class_set stance | Step 5 §4 |
| 10 | uncertainty flag 효과 누적 규칙 | Step 5 §5 |
| 11 | anchor_unreliable 억제 / 허용 카테고리 | Step 5 §6 |
| 12 | feature wording 제한 | Step 5 §7 |
| 13 | no_call message policy | Step 5 §8 |
| 14 | 전반 금지 동작 10항목 | Step 5 §9 |

본 단계는 위 정책 중 어느 것도 *재정의하지 않는다*. 본 문서는 Step 5
정책을 **template 차원에서 구체화**하는 단계이며, 정책 자체는 갱신되지
않는다.

### 2.1 Step 5에서 직접 인용하여 강제하는 핵심 규칙

| # | 규칙 | 근거 |
|---|---|---|
| R1 | single argmax를 caption 클래스로 표현하지 않음 | Step 5 §9 #1 |
| R2 | `no_call == True`에서 어떤 클래스도 추정하지 않음 | Step 5 §8 / §9 #5 |
| R3 | `anchor_unreliable`이면 depth / recovery / bottom-transition 표현 억제 | Step 5 §6.2 / §9 #4 |
| R4 | `lateral_proxy_gyro`를 knee valgus measurement처럼 표현 금지 | Step 5 §7 / §9 #2 |
| R5 | `anchor_reliability`를 motion feature처럼 표현 금지 | Step 5 §7 / §9 #3 |
| R6 | `["C3","C4","C2"]`에서 C2를 생략하지 않음 | Step 5 §4 / §9 #9 |

이 6개 규칙은 §12 validation checklist에 1:1로 매핑된다.

---

## 3. Template input fields

caption template이 받는 입력 필드. 본 단계의 template은 이 필드들의
조합으로 작성되며, 그 외의 필드(특히 디버깅용)는 사용할 수 없다.

| 필드 | 타입 | 출처 | template 사용 가능 여부 |
|---|---|---|---|
| `class_set_prediction` | JSON list | Step 4 schema output | 사용 가능 (§7) |
| `caption_confidence_level` | enum 4개 | Step 4 schema output | 사용 가능 (§6) |
| `uncertainty_flags` | JSON list (closed vocab 7개) | Step 4 schema output | 사용 가능 (§8) |
| `no_call` | bool | Step 4 schema output | 사용 가능 (§11) |
| `anchor_reliability` | float ∈ [0, 1] | Step 4 schema output | 보조 사용 (§9) — *수치 노출 금지*, "anchor 신뢰 가능 / anchor 신뢰 어려움" 카테고리 차원으로만 |
| `anchor_type` | enum 2개 | Step 4 schema output | 사용 가능 (자세 조건부 분기) |
| `posture` | enum {SA, CA, HW} | manifest | 사용 가능 (§10 posture phrase) |
| `p_C1`..`p_C6` | float | Step 4 schema output | **수치 직접 노출 금지** — class set 표현에서 ambiguity 정도의 정성적 phrase에 한정 (§5 R7) |
| feature evidence values | float (raw 또는 zscore) | `data/step4/step4_modeling_dataset.csv` (read-only) — schema output CSV에는 feature evidence 값이 없으므로 sample_id 기준 join 필요 | 측정/heuristic 분리하여 사용 (§10) — 수치 인용은 §10에서 정의된 phrase category로 변환 |
| feature evidence category | string | Step 5 §7 분류 | 사용 가능 (phrase 선택용) |

### 3.0.1 feature evidence join 정책 (Step 7 적용)

| 항목 | 내용 |
|---|---|
| source | `data/step4/step4_modeling_dataset.csv` (read-only) |
| join key | `sample_id` |
| join 대상 | Step 4 schema output CSV (`step4_schema_outputs_raw.csv`, `step4_schema_outputs_zscore.csv`) |
| join 시점 | Step 7에서 read-only join하여 feature phrase 선택에만 사용 — 본 단계에서는 join을 수행하지 않음 |
| join 실패 fallback | feature evidence join이 불가능한 경우, Step 7은 `{main_feature_phrase}`를 *생략*하고 class_set / uncertainty / no_call 중심 caption으로 fallback |
| 수정 금지 | join은 read-only이며, 어떤 CSV도 본 단계 또는 Step 7에서 수정하지 않음 |

### 3.1 template 입력에서 명시적으로 제외되는 필드

| 필드 | 제외 사유 |
|---|---|
| `pred_argmax_debug` | Step 4 §4 / Step 5 §9 #1 — 디버깅용. caption 입력 금지 |
| `class_id_true` | Step 3 §3 — 추론 시점 emission이 아님 |
| `participant_id` 조건부 통계 | Step 3 §10 — output layer는 보지 않음 |
| `bottom_recovery_slope_acc_z` | Step 3 hold-out / 현재 schema에 미등장 |
| threshold 후보값 (`confident_C2_threshold` 등) | Step 5 §9 #6 — 확정값처럼 노출 금지 |
| raw vs zscore branch 식별자 | Step 5 §9 #7 — 최종 branch가 commit되지 않음 |

---

## 4. Template composition order

caption은 다음 순서로 구성된다. 이 순서는 정책 우선순위를 *그대로*
반영한다 — 상위 단계에서 결정되면 하위 단계는 *제약 안에서만* 동작한다.

| 단계 | 입력 | 출력 | 다음 단계 영향 |
|---|---|---|---|
| 1. no_call 여부 확인 | `no_call` | true → §11 no_call template으로 분기 / false → 다음 단계 | true이면 2~6 단계 우회 |
| 2. caption_confidence_level 확인 | `caption_confidence_level` | level (`confident` / `hedged` / `low`) | level별 §6 family 선택 |
| 3. class_set_prediction 확인 | `class_set_prediction` | class_set 분류 (8개 중 하나) | §7 class_set skeleton 선택 |
| 4. uncertainty_flags modifier 적용 | `uncertainty_flags` | template 후보의 phrase 카테고리 조정 | §8 modifier 적용 (가장 제약 강한 정책 우선) |
| 5. anchor_unreliable suppression 적용 | flags 중 `anchor_unreliable` | depth / recovery / bottom-transition 카테고리 제거 | §9 suppression 적용 |
| 6. feature evidence phrase 선택 | feature evidence values + category | phrase bank에서 선택된 phrase | §10 phrase bank 적용 |
| 7. final safety check 적용 | 위 결과 | §12 validation checklist 통과 여부 | 위반 시 template 거부 — Step 7에서 fallback 정책 |

### 4.1 단계 간 정책 충돌 해소

| 충돌 상황 | 해소 |
|---|---|
| level이 confident인데 anchor_unreliable flag 동시 존재 | 정의상 발생 불가 — Step 4 generator가 hedged로 강등하여 도달 (Step 5 §3) |
| modifier 여러 개 동시 적용 | 가장 *제약 강한* 정책 우선 (Step 5 §5) — 카테고리 합집합으로 금지, 교집합으로 허용 |
| feature phrase가 anchor_unreliable suppression과 충돌 | suppression이 우선 — phrase는 anchor-independent로 교체 또는 제거 (§10) |
| no_call인데 feature evidence phrase가 선택됨 | no_call이 우선 — feature phrase 모두 제거 (§11) |

---

## 5. Common template guardrails

모든 template family에 공통으로 적용되는 금지 규칙. §12 checklist의
1:1 근거가 된다.

| # | 금지 규칙 | 사유 |
|---|---|---|
| G1 | 단일 argmax 표현 ("이것은 C2이다" 형태로 단일 클래스 단정) 금지 | Step 5 §9 #1 — `pred_argmax_debug` 노출과 동일한 위반 |
| G2 | `no_call`에서 어떤 클래스도 추정 금지 | Step 5 §9 #5 |
| G3 | threshold 후보값 (`confident_C2_threshold` 등 5개) 직접 노출 금지 | Step 5 §9 #6 — 후보값이 확정처럼 보이는 표현 금지 |
| G4 | posterior 확률 수치를 과도한 확신처럼 표현 금지 (`p_C2 = 0.41` → "확실히 C2" 카테고리 금지) | Step 5 §3 / §9 #1 |
| G5 | heuristic을 measurement처럼 표현 금지 (특히 `lateral_proxy_gyro`) | Step 5 §7 / §9 #2 |
| G6 | raw / zscore 중 하나를 *최종 deployed branch*로 확정한 듯한 표현 금지 | Step 5 §9 #7 — 본 단계에서 branch 미commit |
| G7 | `anchor_unreliable`에서 depth / recovery / bottom-transition claim 금지 | Step 5 §6.2 / §9 #4 |
| G8 | `["C3","C4","C2"]`에서 C2 누락 금지 | Step 5 §9 #9 |
| G9 | 시스템 결함 시사 표현 ("모델 실패", "데이터 오류", "정답 없음") 금지 | Step 5 §8 |
| G10 | feature 분류(측정/heuristic) 어긋나는 표현 금지 | Step 5 §9 #10 |

---

## 6. Confidence-level template families

각 level별 template family. 모든 skeleton은 placeholder 기반이며, 실제
값 주입은 Step 7에서 수행한다.

placeholder 정의 (모든 family 공통):

| placeholder | 의미 | 예시 슬롯 |
|---|---|---|
| `{class_set_label}` | class_set의 정성적 명칭 (§7) | "C2 단독 가능성", "C1/C5/C6 그룹", "C3/C4 페어 + C2 흡수 가능성" 등 |
| `{main_feature_phrase}` | §10 phrase bank에서 선택된 주요 feature 카테고리 phrase | "전반 동작 범위는 비교적 넓음", "회전량 단서가 약함" 등 |
| `{posture_phrase}` | 자세 인식 단서 phrase (§10) | "SA 자세에서", "CA 자세에서", "HW 자세에서" |
| `{uncertainty_phrase}` | uncertainty flag 기반 modifier (§8) | "그룹 내 모호성이 있음", "anchor 단서가 신뢰하기 어려움" 등 |
| `{no_call_reason_phrase}` | §11 no_call 사유 phrase | "자세 정보가 없어 판단을 보류", "근거가 부족하여 클래스 집합조차 좁히지 못함" 등 |

### 6.1 confident family

진입 조건: `class_set == ["C2"]` AND `caption_confidence_level == confident` AND `anchor_unreliable` 부재.

| template type | 허용 / 금지 |
|---|---|
| direct-but-bounded class statement | 허용 — 단, "단일 클래스 단정"이 아니라 *정성적 commitment* 형태 |
| feature-supported statement | 허용 — `motion_range_acc_z` 계열 phrase 동반 가능 |
| posture-aware caveat | 허용 — `{posture_phrase}` 동반 |
| 다른 클래스 가능성 *완전 배제* 표현 | 금지 (G4) — precision 한계 무시 |
| 과도한 확정 표현 ("확실히", "분명히") | 금지 (Step 5 §3 confident 행) |

template skeleton (한국어, 2~3개):

| # | skeleton |
|---|---|
| T-CONF-1 | `{posture_phrase} {class_set_label}에 부합하는 패턴이 관찰됩니다. {main_feature_phrase}.` |
| T-CONF-2 | `{class_set_label} 패턴이 비교적 안정적으로 나타납니다 ({main_feature_phrase}). {posture_phrase} 기준으로 해석됩니다.` |
| T-CONF-3 | `{posture_phrase} {main_feature_phrase}이며, 이는 {class_set_label}과 일관된 단서입니다.` |

주의: skeleton에는 "100%", "확실히", "단일", "오직" 같은 단정 부사가
없다. `{class_set_label}` placeholder는 Step 7에서 §7의 정성적 명칭으로
치환되므로 단일 argmax가 절대 노출되지 않는다.

### 6.2 hedged family

진입 조건: `caption_confidence_level == hedged`. class_set은 단일
클래스가 *아닌* 경우가 대부분 (Step 4 §9 분포에서 hedged가 다수).

| template type | 허용 / 금지 |
|---|---|
| class-set ambiguity statement | 허용 — class_set을 *집합 단위*로 표현 |
| feature-supported cautious statement | 허용 — feature는 *보조* 역할 |
| uncertainty-aware statement | 허용 — `{uncertainty_phrase}` 동반 |
| class_set 멤버를 단일 class처럼 표현 | 금지 (Step 5 §3 hedged 행) |
| "이 클래스가 맞다" 형태의 단정 | 금지 |
| 그룹 / 페어 / 흡수 정보 누락 | 금지 (G8 포함) |

template skeleton (한국어, 3~5개):

| # | skeleton |
|---|---|
| T-HEDGE-1 | `{posture_phrase} {class_set_label} 사이의 모호성이 있습니다. {main_feature_phrase} 정도가 보조 단서입니다.` |
| T-HEDGE-2 | `현재 단서로는 {class_set_label} 중 하나로 좁혀지지 않습니다. {uncertainty_phrase}.` |
| T-HEDGE-3 | `{posture_phrase} {main_feature_phrase}이지만, {class_set_label}을 단일 클래스로 단정하기에는 근거가 부족합니다.` |
| T-HEDGE-4 | `{class_set_label}과 일관되는 패턴이 보이며 {uncertainty_phrase}. 추가 단서 없이 더 좁히지 않습니다.` |
| T-HEDGE-5 | `{posture_phrase} {class_set_label} 가능성을 동시에 고려해야 합니다 ({main_feature_phrase}).` |

주의: 어떤 skeleton도 class_set 안의 한 클래스를 *대표*로 내세우지
않는다. `{class_set_label}`은 항상 *집합* 단위 명칭으로 치환된다(§7).

### 6.3 low family

진입 조건: `caption_confidence_level == low` (대부분 hedged였다가
`anchor_unreliable`로 강등된 경우, Step 5 §3).

| template type | 허용 / 금지 |
|---|---|
| weak evidence statement | 허용 — "근거가 약함" 카테고리 |
| multiple-pattern statement | 허용 — 가능한 패턴이 여럿임을 명시 |
| feature evidence as auxiliary only | 허용 — feature는 *부수적*으로만 |
| 강한 class commitment | 금지 (Step 5 §3 low 행) |
| anchor 의존 표현 (depth, recovery, bottom-transition) | 금지 (G7) |
| heuristic을 measurement처럼 표현 | 금지 (G5) |

template skeleton (한국어, 2~3개):

| # | skeleton |
|---|---|
| T-LOW-1 | `{posture_phrase} 단서가 약하여 {class_set_label} 사이에서도 확정하기 어렵습니다. {uncertainty_phrase}.` |
| T-LOW-2 | `여러 패턴이 동시에 가능합니다 ({class_set_label}). {main_feature_phrase}은 보조 정보로만 참고됩니다.` |
| T-LOW-3 | `{uncertainty_phrase} 상태로, {class_set_label}을 좁히기에는 근거가 부족합니다.` |

주의: `{main_feature_phrase}`는 §10에서 anchor-independent feature
(예: motion-range)만 선택될 수 있도록 §10이 강제한다. low family에서
depth_proxy / bottom_transition 계열 phrase는 anchor_unreliable이
공존하면 §10에서 자동 제거된다.

### 6.4 no_call family

진입 조건: `no_call == true`. 자세한 사유별 정책은 §11에서 다룬다.
여기서는 family의 stance만 정의한다.

| template type | 허용 / 금지 |
|---|---|
| bounded no-call statement | 허용 — 사유 카테고리 단위 |
| reason-category statement | 허용 — `{no_call_reason_phrase}` |
| no class commitment | 강제 (G2) — 어떤 클래스도 등장하지 않음 |
| "모델이 틀렸다" / "데이터가 잘못되었다" | 금지 (G9) |
| 재측정 강제 | 금지 (Step 5 §8) |

skeleton은 §11에서 사유별 3개로 정의한다.

---

## 7. Class-set specific templates

class_set별 stance와 skeleton. 8개 whitelist 모두 다룬다 (Step 3 §4 /
Step 4 plan §9).

### 7.1 stance 표

| class_set | `{class_set_label}` 후보 | 허용 표현 | 금지 표현 |
|---|---|---|---|
| `["C2"]` | "C2 단독에 부합하는 패턴", "C2 패턴" | level이 confident일 때 한정된 commitment, motion-range 보조 phrase, posture phrase | 다른 클래스 완전 배제, 과도한 단정 부사 |
| `["C1","C5","C6"]` | "C1·C5·C6 그룹", "C1/C5/C6 그룹 내 모호성" | 그룹 단위 ambiguity 진술, 그룹 내 위치 불확정 phrase | 그룹 멤버 중 한 클래스 단정, 그룹 외 클래스로 확장 |
| `["C1","C5"]` | "C1·C5 부분집합 (그룹 일부)" | 그룹 컨텍스트 유지하며 페어 모호성 진술, 그룹 부분집합임을 시사 | 두 클래스 중 하나로 단정, 그룹과 *완전 동일*시 |
| `["C1","C6"]` | "C1·C6 부분집합 (그룹 일부)" | 동일 | 동일 |
| `["C5","C6"]` | "C5·C6 부분집합 (그룹 일부)" | 동일 | 동일 |
| `["C3","C4"]` | "C3·C4 페어" | 페어 모호성 진술 (C2 흡수는 *언급 안 함* — Step 5 §5) | 한 클래스 단정, C2 흡수 *추가* 언급 (이 class_set은 흡수 없음) |
| `["C3","C4","C2"]` | "C3·C4 페어 + C2 흡수 가능성" | 세 클래스 모두를 ambiguity 단위로 표현, C2 흡수 가능성을 *함께* 언급 | C2를 누락한 페어-only 표현 (G8), 한 클래스 단정 |
| `[]` | (사용하지 않음 — no_call) | §11 no_call template만 | 어떤 클래스 추정도 금지 (G2) |

### 7.2 class_set별 template skeleton

placeholder는 §6과 동일.

| class_set | skeleton (예시) |
|---|---|
| `["C2"]` | `{posture_phrase} C2 단독 패턴에 부합합니다. {main_feature_phrase}.` *(level=confident 한정)* |
| `["C1","C5","C6"]` | `{posture_phrase} C1·C5·C6 그룹 내 패턴이 보이며, 이 그룹 내에서 어느 클래스인지는 좁혀지지 않습니다. {uncertainty_phrase}.` |
| `["C1","C5"]` | `{posture_phrase} C1·C5·C6 그룹 중 C1 또는 C5에 더 무게가 실리지만, 단일 클래스로 좁혀지지 않습니다.` |
| `["C1","C6"]` | `{posture_phrase} C1·C5·C6 그룹 중 C1 또는 C6 가능성이 함께 고려됩니다. {main_feature_phrase}.` |
| `["C5","C6"]` | `{posture_phrase} C1·C5·C6 그룹 중 C5 또는 C6 가능성이 함께 고려됩니다. {main_feature_phrase}.` |
| `["C3","C4"]` | `{posture_phrase} C3·C4 페어 사이의 모호성이 있습니다. {main_feature_phrase}.` *(C2 흡수 언급 금지)* |
| `["C3","C4","C2"]` | `{posture_phrase} C3·C4 페어와 함께 C2 흡수 가능성도 동시에 고려됩니다 ({main_feature_phrase}). 세 가능성을 *모두* 유지합니다.` *(C2 누락 금지 — G8)* |
| `[]` | (§11 no_call template만 사용) |

### 7.3 보존 규칙 — `["C3","C4","C2"]`

| 보존 항목 | 강제 |
|---|---|
| C2 등장 | 모든 skeleton에 명시적으로 등장 |
| 세 클래스 ambiguity 단위 | "페어"라는 명칭만으로 2-class처럼 보이지 않도록 *"+ C2 흡수 가능성"* 어구 강제 |
| 흡수 방향 | C3/C4 → C2의 흡수 (Step 4 §7 결과) — 반대 방향으로 표현 금지 |

§12 checklist 항목 #3에서 이 보존을 검증한다.

---

## 8. Uncertainty flag modifiers

flag별 caption phrase 카테고리에 미치는 효과. 여러 flag가 공존하면
**가장 제약 강한 modifier가 우선**하며, 효과는 **누적 적용**된다 (Step
5 §5).

| flag | modifier effect | allowed phrase category | prohibited phrase category |
|---|---|---|---|
| `confident_C2` | confident family trigger. 단, anchor_unreliable 공존 시 hedged로 강등된 level을 따름 | "C2 단독 패턴에 부합" | 다른 클래스 *완전* 배제 |
| `within_group_ambiguity_c1_c5_c6` | 그룹 단위 ambiguity 명시 강제 | "C1·C5·C6 그룹 내 모호성", "그룹 내 위치 불확정" | 그룹 멤버 단일 단정 |
| `pair_ambiguity_c3_c4` | 페어 모호성 명시 강제. C2 흡수는 언급 *안 함* | "C3·C4 페어 모호성" | 한 클래스 단정, *추가* C2 흡수 언급 |
| `pair_plus_c2_absorption` | 페어 모호성 + C2 흡수 동시 명시 강제 | "C3·C4 페어와 C2 흡수 가능성", 세 클래스 ambiguity unit | C2 누락 (G8), 한 클래스 단정 |
| `anchor_unreliable` | §9 suppression 적용 | "anchor 단서가 신뢰하기 어려움", "anchor-independent 일반 진술" | depth-specific, bottom location, recovery slope, bottom transition (§9) |
| `posture_unknown` | no_call 진입. §11 `posture_unknown` template만 | "자세 정보가 없어 판단 보류" | 어떤 클래스도 추정 |
| `low_confidence_no_class_set` | no_call 진입. §11 `low_confidence_no_class_set` template만 | "근거가 부족하여 클래스 집합조차 좁히지 못함" | 어떤 클래스도 추정, "모델 실패" 어조 |

### 8.1 modifier 결합 시 우선순위

| 결합 상황 | 우선 적용 |
|---|---|
| `confident_C2` + `anchor_unreliable` | 정의상 발생 불가 (level 강등) — 발생 시 hedged 규칙 + §9 suppression |
| `within_group_ambiguity_c1_c5_c6` + `anchor_unreliable` | 그룹 ambiguity 유지 + §9 suppression (그룹 진술은 anchor-independent이므로 양립 가능) |
| `pair_plus_c2_absorption` + `anchor_unreliable` | 페어 + C2 흡수 유지. 단, depth/recovery 보조 phrase는 §9에 의해 제거 |
| `posture_unknown` + 다른 flag | `posture_unknown`이 최우선 (no_call 진입) — 다른 flag의 modifier는 *적용하지 않음* |
| `low_confidence_no_class_set` + 다른 flag | `low_confidence_no_class_set`이 최우선 (no_call 진입) — 다른 flag modifier 미적용 |

---

## 9. Anchor-unreliable suppression templates

`anchor_unreliable` flag가 caption phrase에 미치는 억제와 허용 카테고리.
Step 5 §6.2의 표를 template 차원에서 구체화한다.

### 9.1 억제 카테고리

| 억제 카테고리 | template phrase 차원의 효과 |
|---|---|
| depth-specific hard claim ("∼cm 깊이에 도달했다", "충분히 깊었다" 단정 카테고리) | template에서 제거 — 대체 phrase로 교체 |
| bottom location claim ("정확히 bottom에 도달") | 제거 |
| recovery slope claim ("회복 속도가 빠르다/느리다" 단정) | 제거 |
| bottom transition claim ("bottom 전환이 명확하다" 단정) | 제거 |

### 9.2 허용 카테고리 (anchor_unreliable에서도 사용 가능)

| 허용 카테고리 | 사유 |
|---|---|
| whole-sequence motion range | `motion_range_acc_z` 계열 — anchor-independent (Step 5 §6.2) |
| posture-aware general observation | 자세 정보는 anchor에 무관 |
| uncertainty statement | flag 자체를 표현하는 것은 정책에 합치 |
| group / pair ambiguity statement | anchor 의존 없음 |

### 9.3 replacement phrase category (suppress된 phrase의 대체)

| 원래 카테고리 | 대체 카테고리 (template skeleton 단계의 *카테고리* 정의 — 실제 wording은 Step 7) |
|---|---|
| depth-specific hard claim | "전반적인 동작 범위 진술" (motion range) |
| bottom location claim | "전반 위치 단서가 약함" (uncertainty statement) |
| recovery slope claim | "복귀 동역학에 대한 강한 진술 보류" (uncertainty statement) |
| bottom transition claim | "전환 단서가 신뢰 범위 밖" (uncertainty statement) |

### 9.4 suppression이 template에 가하는 강제

| 강제 항목 | 적용 |
|---|---|
| §10에서 `bottom_stability_acc`, `bottom_transition_delta_acc_z`, `depth_proxy` phrase 자동 제거 | 적용 |
| §10에서 `motion_range_acc_z`, `motion_range_gyro_mag` phrase 우선 선택 | 적용 |
| `{uncertainty_phrase}`에 "anchor 단서가 신뢰하기 어려움" 카테고리 강제 삽입 | 적용 |

---

## 10. Feature evidence phrase bank

feature별 phrase bank. 각 feature는 *카테고리 단위*로만 phrase를 가지며,
실제 phrase wording은 Step 7에서 placeholder를 채울 때 결정된다. 본
단계는 phrase **선택 정책**과 phrase **카테고리 명세**까지 작성한다.

### 10.1 phrase bank 표

| feature | feature role (Step 5 §7) | allowed phrase category | prohibited phrase category | usable confidence levels | anchor_unreliable 영향 |
|---|---|---|---|---|---|
| `motion_range_acc_z` | main / measurement / anchor-independent | "전반 동작 범위 진술" | 다른 자세와의 *절대 수치* 비교 | confident, hedged, low | 영향 없음 (anchor-independent) — low에서도 사용 가능 |
| `depth_proxy` | main / measurement / posture-aware | "정규화된 깊이 단서 진술" (자세 조건부) | raw absolute depth 단정, 자세 간 직접 비교 | confident, hedged | `anchor_unreliable`이면 §9에 의해 phrase 제거 |
| `motion_range_gyro_mag` | main / measurement / posture-aware | "회전량 / 움직임 크기 진술" (자세 조건부) | 자세 간 절대 비교, "정확한 각도" 표현 | confident, hedged, low | 영향 없음 (anchor-independent — Step 5 §7 분류) |
| `bottom_stability_acc` | main / measurement / posture-aware + anchor-dependent | "bottom 부근 안정성 단서" (anchor 신뢰 시) | 강한 단정 ("완전히 안정적") | confident, hedged | `anchor_unreliable`이면 §9에 의해 phrase 제거 |
| `bottom_transition_delta_acc_z` | main / measurement / anchor-dependent | "전환 단서" (anchor 신뢰 시) | 단독 recovery 단정 | hedged, low | `anchor_unreliable`이면 §9에 의해 phrase 제거 |
| `lateral_proxy_gyro` | auxiliary / **heuristic only** | "약한 회전 단서 (heuristic)" | knee valgus measurement 표현 (G5), 측정값처럼 수치 인용 | hedged, low (보조 only) | 영향 없음 (heuristic — anchor와 무관) |
| `anchor_reliability` | uncertainty-only | "anchor 신뢰 가능 / 어려움" 카테고리 (정성) | motion feature처럼 표현 (G5/R5), 수치 직접 인용 | 모든 level (uncertainty 트랙) | 본 feature가 곧 anchor_unreliable의 근거이므로, 이를 motion으로 표현하는 것은 정의상 금지 |

### 10.2 phrase 선택 알고리즘 (정책 단계)

| 단계 | 동작 |
|---|---|
| (a) anchor_unreliable 확인 | true이면 anchor-dependent feature phrase 제거 (§9) |
| (b) level 확인 | level이 *허용 levels*에 포함되지 않은 feature phrase 제거 |
| (c) class_set 적합성 확인 | class_set이 anchor 의존 (`["C2"]`, `["C3","C4","C2"]`)인 경우 anchor-independent phrase 우선 |
| (d) heuristic phrase 사용 시 강제 phrasing | `lateral_proxy_gyro`는 *반드시* "약한 회전 단서 (heuristic)" 카테고리로만 등장 — measurement 표현 금지 (G5) |
| (e) `anchor_reliability` 사용 | uncertainty 트랙 phrase로만 (`{uncertainty_phrase}`에 결합) — `{main_feature_phrase}`에 결합 금지 |

### 10.3 phrase 카테고리 정의 (Step 7에서 wording 채움)

| phrase 카테고리          | 의미 (Step 7 wording 가이드)                                    |
| -------------------- | ---------------------------------------------------------- |
| 전반 동작 범위 진술          | 전체 sequence의 acc-z 동작 범위 정성 — "비교적 넓음 / 좁음 / 보통" 정도의 카테고리  |
| 정규화된 깊이 단서 진술        | 자세 조건부 정규화된 depth_proxy의 정성 카테고리 — 절대 수치 금지                |
| 회전량 / 움직임 크기 진술      | gyro magnitude의 자세 조건부 정성 — 자세 간 비교 금지                     |
| bottom 부근 안정성 단서     | bottom 부근 acc 안정성 정성 — anchor 신뢰 전제                        |
| 전환 단서                | bottom 전환 delta-acc-z 정성 — anchor 신뢰 전제, 단독 recovery 단정 금지 |
| 약한 회전 단서 (heuristic) | `lateral_proxy_gyro` 전용 — heuristic 표시 의무                  |
| anchor 신뢰 가능 / 어려움   | `anchor_reliability` 정성 카테고리 — 수치 인용 금지                    |

---

## 11. No-call template policy

no_call 사유별 template. 사유 분류는 Step 5 §8과 일치하며, Step 4 §10
no_call reason closure에서 검증된 세 경로를 모두 다룬다.

### 11.1 사유별 표

| 사유 | 사용자에게 말할 수 있는 정보 | 말하면 안 되는 정보 |
|---|---|---|
| `posture_unknown` | "자세 정보가 없어 판단을 보류" 카테고리 | 어떤 클래스 추정, 자세 추정, "데이터 오류"  |
| `low_confidence_no_class_set` | "근거가 부족하여 클래스 집합조차 좁히지 못함" 카테고리 | 어떤 클래스 추정, posterior 수치 노출, "모델이 틀렸다" |
| anchor-driven (anchor_reliability < anchor_no_call_threshold AND class_set ∈ `{[C2], [C3,C4,C2]}`) | "anchor 단서를 신뢰하기 어려워 anchor 의존적 클래스 추정을 보류" 카테고리 | depth/recovery/bottom-transition 단정, 다른 anchor-independent 정보로 *우회 추정* |

### 11.2 사유별 template skeleton (3개)

| #      | 사유                            | skeleton                                                                         |
| ------ | ----------------------------- | -------------------------------------------------------------------------------- |
| T-NC-1 | `posture_unknown`             | `자세 정보가 확인되지 않아 클래스에 대한 판단을 보류합니다. ({no_call_reason_phrase})`                    |
| T-NC-2 | `low_confidence_no_class_set` | `현재 단서로는 클래스 집합조차 좁힐 수 없을 만큼 근거가 부족합니다. 어떤 클래스도 추정하지 않습니다.`                      |
| T-NC-3 | anchor-driven                 | `anchor 단서를 신뢰하기 어려워, anchor에 의존하는 클래스 후보에 대한 판단을 보류합니다. ({uncertainty_phrase})` |

### 11.3 공통 금지

| #     | 금지 항목                         | 사유                                                  |
| ----- | ----------------------------- | --------------------------------------------------- |
| NC-G1 | 어떤 클래스 추정                     | G2 / Step 5 §8 / §9 #5                              |
| NC-G2 | `pred_argmax_debug` 노출        | G1 / Step 5 §9 #1                                   |
| NC-G3 | "모델 실패", "데이터 오류", "정답 없음" 표현 | G9 / Step 5 §8                                      |
| NC-G4 | 재측정 강제                        | Step 5 §8 (UX 결정 사항으로 본 정책 범위 밖이며 caption은 강제하지 않음) |
| NC-G5 | 복수 사유를 *모두* 나열                | Step 5 §8 — "가장 상위 사유 한 가지" 권장                      |

### 11.4 사유 우선순위 (복수 충족 시)

Step 5 §8에 따라 caption은 *상위 사유 한 가지*만 표현한다. 우선순위:

| 순위 | 사유 |
|---|---|
| 1 (최상위) | `posture_unknown` |
| 2 | anchor-driven no_call |
| 3 | `low_confidence_no_class_set` (catch-all) |

이 우선순위는 §11.2 skeleton 선택을 결정한다.

---

## 12. Template validation checklist

Step 6 template이 Step 5 policy를 위반하지 않는지 확인하는 checklist.
Step 7에서 template 적용 전 *모든 항목*을 통과해야 한다.

| # | 체크 항목 | 근거 (Step 5) | 적용 단계 (§4) |
|---|---|---|---|
| C1 | no_call template이 어떤 클래스도 추정하지 않는가 | §8 / §9 #5 | 1, 7 |
| C2 | hedged template이 단일 class처럼 말하지 않는가 (집합 단위 표현 유지) | §3 hedged / §4 | 2, 3 |
| C3 | `["C3","C4","C2"]` template이 C2를 생략하지 않는가 | §4 / §9 #9 | 3, 7 |
| C4 | `anchor_unreliable` template이 depth / recovery / bottom-transition 표현을 억제하는가 | §6.2 / §9 #4 | 5, 6 |
| C5 | `lateral_proxy_gyro`가 knee valgus measurement로 표현되지 않는가 | §7 / §9 #2 | 6 |
| C6 | `anchor_reliability`가 motion feature로 표현되지 않는가 | §7 / §9 #3 | 6 |
| C7 | threshold 후보값을 확정값처럼 말하지 않는가 (5개 threshold 모두) | §9 #6 | 1~7 (전반) |
| C8 | `pred_argmax_debug`를 사용하지 않는가 | §9 #1 | 입력 확정 (§3) |
| C9 | raw vs zscore branch를 *최종 deployed*처럼 말하지 않는가 | §9 #7 | 전반 |
| C10 | confident family가 다른 클래스 가능성을 *완전 배제*하지 않는가 | §3 confident | 2 |
| C11 | feature 분류(측정/heuristic) 어긋난 표현이 없는가 | §9 #10 | 6 |
| C12 | `class_set == []` 또는 `no_call == true`인 template에서 어떤 클래스도 추정하지 않는가 (no_call에서의 클래스 추정 금지를 강제 — C1과 동일 의미를 class_set 형태 차원에서 재확인) | §8 / §9 #5 | 1 |
| C13 | "모델 실패", "데이터 오류", "정답 없음" 어조가 등장하지 않는가 | §8 | 1, 7 |
| C14 | posterior 수치가 직접 노출되지 않는가 | §3 / §9 #1 | 입력 확정 (§3) |

---

## 13. Step 6 completion criteria

Step 6은 **다음을 모두 충족할 때** 완료된 것으로 본다.

| # | 기준 | 본 문서의 충족 위치 |
|---|---|---|
| 1 | confidence level별 template family 작성 | §6 (4개 family) |
| 2 | class_set별 template skeleton 작성 | §7 (8개 whitelist 모두) |
| 3 | uncertainty flag modifier 작성 | §8 (7개 closed vocabulary 모두) |
| 4 | no_call template policy 작성 | §11 (3개 사유) |
| 5 | feature evidence phrase bank 작성 | §10 (7개 feature) |
| 6 | template validation checklist 작성 | §12 (14개 항목) |
| 7 | 실제 sample별 caption 생성을 *수행하지 않음* | 본 문서 전체 (placeholder skeleton 외 실제 문장 부재) |

다음은 Step 6 완료 기준이 **아니다**:

| # | 비완료 기준 | 사유 |
|---|---|---|
| 1 | per-sample caption 생성 | Step 7 범위 |
| 2 | final wording commit | Step 6은 후보 단계 |
| 3 | raw vs zscore branch commit | Step 4 미commit / Step 5 §9 #7 |
| 4 | threshold commit | Step 4 미commit / Step 5 §9 #6 |
| 5 | 새 모델 학습 / threshold recalibration / schema output 재생성 / 새 feature 계산 | 본 단계 범위 외 |
| 6 | Step 1 / 2 / 3 / 4 / 5 산출물 수정 | 본 단계 범위 외 |

---

## 14. Next step after Step 6

| 단계 | 내용 | 제약 |
|---|---|---|
| Step 7 — Caption Generation Prototype | 본 문서의 template family를 적용하여 schema output CSV(read) 기반 sample별 caption 생성 | §12 validation checklist의 14개 항목을 *모두* 통과해야 함. 통과 못한 sample은 Step 7에서 정의될 fallback 정책으로 처리 |
| Step 7 입력 | `data/step4/step4_schema_outputs_raw.csv`, `data/step4/step4_schema_outputs_zscore.csv`, `data/step4/step4_modeling_dataset.csv` (모두 read-only). schema output CSV에는 feature evidence 값이 없으므로 modeling_dataset.csv를 `sample_id` 기준으로 join하여 feature phrase 선택에만 사용 (§3.0.1) | 두 branch 모두에 대해 별도 결과 산출 — 본 단계까지 raw/zscore 최종 commit 없음. join 실패 시 feature phrase는 생략하고 class_set / uncertainty / no_call 중심 caption으로 fallback |
| Step 7 진입 전제 | Step 6 template family 동결 (본 문서) | 본 문서 변경이 발생하면 그 변경이 어느 §에 영향을 주는지 명시한 뒤 Step 7 입력으로 사용 |
| Step 7 산출물 | sample별 caption (예: `data/step7/step7_captions_raw.csv`, `..._zscore.csv` 등). 본 단계에서 산출 형태는 commit하지 않음 | Step 7에서 결정 |

---

*본 문서는 Step 6 caption template design으로 작성되었다. 실제 sample별
caption은 생성하지 않았으며, 새 모델 학습 / threshold recalibration /
schema output 재생성 / 새 feature 계산 / per-sample caption CSV 생성은
수행하지 않았다. Step 1 / Step 2 / Step 3 / Step 4 / Step 5 산출물은
수정되지 않았다. data/step4 CSV는 read-only 참고로만 사용했다.*
