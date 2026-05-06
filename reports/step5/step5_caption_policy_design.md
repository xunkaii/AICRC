# Step 5 — Caption Policy Design

이 문서는 Step 4 schema output을 사람이 읽을 수 있는 caption으로 변환하기
**전에**, 어떤 표현 범위를 허용하고 금지할지를 정의한다. 본 문서는
caption template wording을 작성하지 않으며, 예시 문장도 제시하지 않는다.
표현의 *카테고리* 단위에서 허용/억제 정책만 명시한다.

본 문서는 다음에 종속된다 (충돌 시 종속 문서가 우선):

- `reports/step3/step3_output_schema_uncertainty_policy.md` — 출력 스키마 / 불확실성 정책
- `reports/step4/step4_final_summary.md` — Step 4 최종 상태
- `reports/step4/step4_modeling_calibration_results.md` — baseline 평가
- `reports/step4/step4_threshold_calibration.md` — threshold 후보값
- `data/step4/step4_schema_outputs_raw.csv`, `data/step4/step4_schema_outputs_zscore.csv` — 입력 데이터(read-only)

---

## 1. Purpose

| 항목             | 내용                                                                                                                     |
| -------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Step 5 목적      | Step 4 schema output을 caption으로 변환하기 전, 표현의 허용 범위를 *정책*으로 정의                                                           |
| 본 문서가 다루지 않는 것 | caption wording / template / 예시 문장                                                                                     |
| 본 문서가 다루는 것    | `caption_confidence_level`, `class_set_prediction`, `uncertainty_flags`, `anchor_reliability` 입력에 따른 표현 카테고리 단위의 허용/금지 |
| 정책 결합 원칙       | 한 행에 여러 정책이 동시에 적용되면 **가장 제약이 강한 정책이 우선** (downgrade는 누적, 억제는 병합)                                                      |

---

## 2. Inputs from Step 4

caption layer가 rep당 받는 입력. Step 4 schema output CSV 컬럼과 Step 3
§3 emission 스키마와 일치한다.

| 입력 | 타입 | 출처 / 비고 |
|---|---|---|
| `class_posterior` | `{C1..C6 → float}` | `p_C1..p_C6` 6개 컬럼; 적절한 분포 (Step 4 검증 통과) |
| `class_set_prediction` | JSON list | Step 4 plan §9 whitelist 8개 집합 중 하나 |
| `uncertainty_flags` | JSON list (중복 없음) | Step 3 §7 closed vocabulary 7개 |
| `caption_confidence_level` | enum | `{confident, hedged, low, no_call}` |
| `no_call` | bool | `class_set_prediction == []`와 양방향 일관 |
| `anchor_reliability` | float ∈ [0, 1] | uncertainty 트랙 전용 (motion feature 아님) |
| `anchor_type` | enum | `{ensemble_acc_gyro, acc_only_anchor}` (자세 조건부) |
| feature evidence | 측정/휴리스틱 분리 | Step 3 §5 두 버킷; 본 단계에서는 카테고리 기준 표현 정책만 부여 |
| `posture` | enum `{SA, CA, HW}` | Step 3 §2의 필수 입력 |

---

## 3. Caption level policy

`caption_confidence_level`별 표현 허용 범위.

| level | 진입 조건 (Step 4 generator) | 허용되는 표현 카테고리 | 금지되는 표현 카테고리 |
|---|---|---|---|
| `confident` | `class_set == ["C2"]` AND `anchor_unreliable` 부재 | 단일 클래스 commitment 형태의 직접적 설명 (해당 클래스에 한정), motion-range 계열 표현, 자세 인식 일반 진술 | 다른 클래스의 동등 가능성 시사, 과도한 단정 (precision 한계 무시) |
| `hedged` | rule 1~3 결과이면서 confident 아님 | 다중 클래스 모호성 진술 (`"consistent with"`, `"may indicate"`, `"ambiguous between"` 카테고리), feature evidence 기반 보조 진술, 자세 인식 진술 | class_set의 멤버를 단일 class처럼 표현, "이 클래스가 맞다" 형태의 단정, 그룹/페어/흡수 hedge 정보의 누락 |
| `low` | hedged 였으나 `anchor_unreliable`로 강등 | 약한 일치도 진술 (`"evidence is weak"`, `"multiple patterns are possible"` 카테고리), feature evidence는 *보조적으로만* | 어떤 클래스에 대한 강한 commitment, anchor 의존적 표현(깊이/회복), 휴리스틱을 측정처럼 표현 |
| `no_call` | rule 0 / rule 4 / anchor-driven no_call | no_call 사유의 *제한적* 설명만 (§8에 정의된 카테고리) | 어떤 클래스에 대한 추정, "모델 실패" / "데이터 오류" 등 시스템 결함을 시사하는 표현, argmax 클래스 노출 |

추가 규칙:

- `confident` → `hedged` 강등은 `anchor_unreliable` 발생 시 자동 적용 (Step 4 generator가 이미 처리). caption layer는 이를 *재해석하지 않고* 받은 level을 그대로 따른다.
- `hedged` → `low` 강등은 동일 원칙.
- `low` → `no_call` 자동 강등은 발생하지 않음 (anchor-driven no_call은 generator 단계에서 분리됨).

---

## 4. Class-set caption stance

`class_set_prediction`별 표현 stance.

| class_set | stance 카테고리 | 핵심 정책 |
|---|---|---|
| `["C2"]` | 단일 클래스 직접 설명 (단, level 의존) | level이 `confident`일 때만 직접적 형태 허용. precision 한계로 인해 *과도한 확정* 카테고리 금지. level이 `hedged`/`low`로 강등된 경우, 단일 클래스 단정 금지 |
| `["C1", "C5", "C6"]` | 그룹 모호성 명시 | 세 클래스 중 하나로 단정 금지. 그룹 자체를 진술 단위로 사용 |
| `["C1", "C5"]` | 페어-수준 모호성 (그룹 부분집합) | 두 클래스 중 하나로 단정 금지. 그룹 컨텍스트는 유지(완전 그룹이 아님을 시사하는 표현은 허용 카테고리) |
| `["C1", "C6"]` | 페어-수준 모호성 (그룹 부분집합) | 동일 |
| `["C5", "C6"]` | 페어-수준 모호성 (그룹 부분집합) | 동일 |
| `["C3", "C4"]` | 페어 모호성 (C2 흡수 없음) | C3/C4 둘 중 하나로 단정 금지. C2 흡수 가능성을 *추가로* 언급 금지 (C2가 무시할 만한 케이스이므로) |
| `["C3", "C4", "C2"]` | 페어 모호성 + C2 흡수 가능성 동시 표현 | "C3 또는 C4 중 하나" 형태로 C2를 *생략*하는 표현 금지. 세 클래스 모두를 ambiguity 단위로 다룬다 |
| `[]` | no_call (§8) | 어떤 클래스에 대한 추정도 금지. caption은 §8에 정의된 사유 기반 카테고리만 |

`["C1"]` / `["C5"]` / `["C6"]` 단일 원소는 schema 차원에서 발생하지
않으므로 stance를 정의하지 않는다 (Step 4 plan §9 whitelist 위반).

---

## 5. Uncertainty flag policy

flag별 caption에 미치는 효과. 여러 flag가 동시에 발생할 수 있으며,
효과는 **누적적으로 적용**된다 (가장 제약이 강한 정책 우선).

| flag                              | caption에 미치는 효과                                                                                                                         |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `confident_C2`                    | C2 단독 설명을 허용하는 trigger. 단, 동시 발생한 `anchor_unreliable`이 있으면 표현 강도를 §3의 `hedged` 규칙으로 낮춘다 (level이 이미 강등되어 있을 것이므로 caption은 강등된 level을 따름) |
| `within_group_ambiguity_c1_c5_c6` | C1/C5/C6 계열 모호성을 명시 카테고리로 표현해야 함. 단일 클래스 표현 금지                                                                                          |
| `pair_ambiguity_c3_c4`            | C3/C4 페어 모호성을 명시 카테고리로 표현. C2 흡수 가능성은 언급하지 *않음* (C2 무시 가능 케이스)                                                                          |
| `pair_plus_c2_absorption`         | C3/C4 페어 모호성 + C2 흡수 가능성을 모두 명시. C2를 누락한 표현 금지                                                                                          |
| `anchor_unreliable`               | §6의 anchor 억제 규칙 적용. depth/recovery/bottom-transition 카테고리 억제. anchor-independent 표현은 허용                                                |
| `posture_unknown`                 | no_call 진입. §8의 `posture_unknown` 정책 적용                                                                                                 |
| `low_confidence_no_class_set`     | no_call 진입. §8의 `low_confidence_no_class_set` 정책 적용. 단, generator가 catch-all로 사용한 경우이므로 사유 언어는 *제한적*으로만                                 |

---

## 6. Anchor reliability policy

`anchor_reliability` 값에 따른 caption 표현 정책. Step 3 §6에서 도입된
두 단계 임계값(`anchor_suppression_threshold`, `anchor_no_call_threshold`)과
일관되게 동작한다.

### 6.1 두 단계 임계값과 caption 효과

| 조건 | 효과 |
|---|---|
| `anchor_reliability ≥ anchor_suppression_threshold` | anchor 억제 없음. depth/recovery/bottom-transition 표현 허용 (level / class_set 다른 정책에 따라) |
| `anchor_no_call_threshold ≤ anchor_reliability < anchor_suppression_threshold` | `anchor_unreliable` 플래그 부착됨. caption layer는 §6.2의 억제 규칙을 적용. **즉시 no_call 아님** — Step 4 generator가 level 강등으로 처리했고 validator도 통과했다 |
| `anchor_reliability < anchor_no_call_threshold` AND class_set이 anchor 의존적 (`["C2"]` 또는 `["C3", "C4", "C2"]`) | generator가 이미 no_call로 라우팅했으므로 caption layer는 §8 no_call 정책을 따른다 |
| `anchor_reliability < anchor_no_call_threshold` AND class_set이 anchor 비의존적 | no_call 아님. level이 `hedged → low`로 강등된 상태로 caption layer에 도달; §3 `low` 규칙 + §6.2 억제 규칙 함께 적용 |

### 6.2 `anchor_unreliable` 발생 시 억제/허용 카테고리

| 카테고리 | 억제 / 허용 |
|---|---|
| depth-specific hard claim (특정 깊이 도달 단정) | **억제** |
| bottom location claim (bottom 위치 단정) | **억제** |
| recovery slope claim (회복 동역학 단정) | **억제** |
| bottom transition claim (전환 깊이/방향 단정) | **억제** |
| whole-sequence motion range 진술 (`motion_range_acc_z` 계열) | **허용** (anchor-independent feature) |
| posture-aware general observation | **허용** |
| uncertainty statement (모호성 / 신뢰도 부족 진술) | **허용** |

`anchor_unreliable`은 즉시 no_call이 아님을 caption layer 정책 차원에서도
명시한다. Step 4 generator가 두 단계 임계값을 분리 적용했고,
`scripts/validate_step4_schema_outputs.py`의 #8 (no_call reason closure) /
#9 (anchor threshold relation)이 모두 통과했음을 본 정책의 전제로 둔다.

---

## 7. Feature evidence wording constraints

feature별 표현 제한. caption이 feature evidence를 언급할 때만 적용된다
(언급 자체가 의무는 아님).

| feature | 분류 (Step 4 decision memo) | 표현 정책 |
|---|---|---|
| `motion_range_acc_z` | main / measurement / anchor-independent | motion range 계열 표현 허용. 가장 안정적 후보이므로 hedged/low 수준에서도 *보조 진술*로 사용 가능 |
| `depth_proxy` | main / measurement / posture-aware | 정규화 의존성 명시 필요. raw absolute depth 형태의 단정 금지 — 자세 간 직접 비교 형태로 표현 금지 |
| `motion_range_gyro_mag` | main / measurement / posture-aware | 자세 간 스케일 차이가 큼 (HW에서 평균이 SA/CA의 약 2배). "회전량/움직임 크기" 카테고리 수준으로 제한. 자세 간 절대 비교 표현 금지 |
| `bottom_stability_acc` | main / measurement / posture-aware + anchor-dependent | `anchor_unreliable` 발생 시 강한 표현 금지 (§6.2의 bottom location 카테고리에 해당) |
| `bottom_transition_delta_acc_z` | main / measurement / anchor-dependent | `depth_proxy`와 함께 해석. 단독 recovery claim 금지. `anchor_unreliable` 발생 시 §6.2 억제 적용 |
| `lateral_proxy_gyro` | auxiliary / **heuristic only** | knee valgus measurement로 표현 **절대 금지** (Step 3 §5, §11). "weak rotational cue" 또는 heuristic 카테고리로만 표현 가능. 측정값처럼 수치를 그대로 인용하는 표현 금지 |
| `bottom_stability_gyro` | auxiliary / measurement | HW에서 gyro 이벤트 mismatch 위험 — caption에서 자세 간 일반화 금지 |
| `anchor_reliability` | uncertainty-only | motion feature가 **아님**. confidence/uncertainty 트랙으로만 사용. caption에서 motion 관련 양으로 인용 금지 |
| `bottom_recovery_slope_acc_z` | hold-out / baseline 입력 미포함 | 본 Step에서는 caption 대상 아님 (현재 schema output에 등장하지 않음) |

---

## 8. No-call message policy

no_call 사유별 caption 정책.

| 사유 (uncertainty_flags 기준) | 진입 경로 | caption 정책 |
|---|---|---|
| `posture_unknown` | rule 0 (Step 4 generator) | posture 정보가 없어 판단 불가 카테고리. 어떤 클래스도 추정하지 않음 |
| `low_confidence_no_class_set` | rule 4 (catch-all) | posterior가 클래스 집합 hedge에도 부족 카테고리. 어떤 클래스도 추정하지 않음 |
| anchor-driven (anchor_reliability < anchor_no_call_threshold AND anchor 의존적 class_set) | rule 1/3 → anchor 강제 no_call | anchor 의존적 설명을 신뢰하기 어려움 카테고리. 다른 anchor-independent 정보로 우회 추정 금지 |

### 공통 금지

| 카테고리 | 정책 |
|---|---|
| 임의 클래스 추정 | 금지 |
| `pred_argmax_debug` 노출 (디버깅용 컬럼을 사용자에게 노출) | 금지 |
| "정답 없음" / "모델 실패" / "데이터 오류" 카테고리 (시스템 결함을 시사) | 금지 |
| 다음 시도 / 재측정 강제 | 본 정책 범위 밖 (UX 결정 사항) |

caption layer는 *복수 사유*가 동시에 충족된 no_call 행에 대해 가장
*상위* 사유 한 가지만 표현하도록 권장한다 (Step 6 template 단계의
세부 결정에 위임하되, 본 정책은 *모든 사유 나열*을 강제하지 않는다).

---

## 9. Prohibited caption behaviors

caption layer 전반에 걸쳐 절대 금지되는 동작.

| #   | 금지 동작                                                                                                                                                                 |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `pred_argmax_debug`(또는 `argmax(class_posterior)`)를 최종 caption의 클래스 출력으로 표현                                                                                            |
| 2   | `lateral_proxy_gyro`를 knee valgus measurement / 무릎 각도 측정값처럼 표현                                                                                                        |
| 3   | `anchor_reliability`를 motion feature / 동작 강도 / 운동량 등으로 표현                                                                                                             |
| 4   | `anchor_unreliable` 상태에서 depth / recovery / bottom-transition 카테고리 표현                                                                                                 |
| 5   | `no_call == True` 상태에서 어떤 클래스든 추정하는 표현                                                                                                                                |
| 6   | threshold 후보값(`confident_C2_threshold`, `non_trivial_C2_threshold`, `within_group_threshold`, `anchor_suppression_threshold`, `anchor_no_call_threshold`)을 *확정값*처럼 표현 |
| 7   | raw / zscore 중 하나를 *최종 배포 branch*로 확정한 것처럼 표현                                                                                                                         |
| 8   | C1/C5/C6 단일 원소 또는 C3/C4 단일 원소(C3 단독, C4 단독)를 단일 클래스로 표현                                                                                                               |
| 9   | `class_set == ["C3", "C4", "C2"]`에서 C2를 누락하고 페어 모호성만 표현                                                                                                               |
| 10  | feature evidence를 *측정 ↔ 휴리스틱* 분류와 어긋나게 표현 (Step 3 §5 위반)                                                                                                              |

---

## 10. Step 5 completion criteria

Step 5는 **다음을 모두 충족할 때** 완료된 것으로 본다.

| # | 기준 | 본 문서의 충족 위치 |
|---|---|---|
| 1 | confidence level별 caption policy 정의 | §3 |
| 2 | class_set별 stance 정의 | §4 |
| 3 | uncertainty flag별 표현 효과 정의 | §5 |
| 4 | `anchor_unreliable` suppression policy 정의 | §6 |
| 5 | no_call message policy 정의 | §8 |
| 6 | heuristic 표현 제한 정의 | §7 (특히 `lateral_proxy_gyro`), §9 #2 |
| 7 | caption template wording은 작성하지 *않음* | 본 문서 전체 (예시 문장 부재) |

다음은 Step 5 완료 기준이 **아니다**:

- caption template / 예시 문장
- 최종 normalization (raw vs zscore) commit
- 최종 threshold commit
- 모델 commit / 성능 기준 / 성공 판정

---

## 11. Next step after Step 5

| 단계 | 내용 | 제약 |
|---|---|---|
| Step 6 — Caption Template Design | 본 문서의 정책을 위반하지 않는 *template* 작성 | §3 / §4 / §5 / §6 / §7 / §8 / §9의 어떤 항목도 위반하는 template은 금지. 위반 시 §10의 완료 기준 #7 (Step 5는 wording을 다루지 않음) 가정이 깨지므로, Step 6 진입 자체를 미루어야 함 |
| Step 6 진입 전제 | Step 5 정책 동결 (본 문서) | 본 문서 변경이 발생하면 그 변경이 어느 §에 영향을 주는지 명시한 뒤 Step 6 입력으로 사용 |

---

*본 문서는 Step 5 caption policy design으로 작성되었다. caption template
wording / 예시 문장은 작성하지 않았으며, 새 모델 학습 / threshold
recalibration / schema output 재생성 / 새 feature 계산은 수행하지 않았다.
Step 1 / Step 2 / Step 3 / Step 4 산출물은 수정되지 않았다.*
