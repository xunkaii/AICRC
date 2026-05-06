# Step 8 — Caption Evaluation / Human Review Design

이 문서는 Step 7에서 생성한 caption prototype을 어떻게 평가할지 *설계*
한다. 본 단계는 **설계 단계**이며, 실제 대규모 human review를 수행하지
않는다. caption CSV 재생성 / 새 모델 학습 / threshold recalibration /
schema output 재생성 / 새 physical feature 계산 / raw vs zscore 최종
branch 확정 / final wording 확정은 *수행하지 않는다*. data/step7
caption CSV는 read-only 참고로만 본다.

본 문서는 다음에 종속된다 (충돌 시 종속 문서가 우선):

- `reports/step3/step3_output_schema_uncertainty_policy.md` — 출력 스키마 / 불확실성 정책
- `reports/step4/step4_final_summary.md` — Step 4 최종 상태
- `reports/step5/step5_caption_policy_design.md` — caption policy
- `reports/step6/step6_caption_template_design.md` — caption template family
- `reports/step7/step7_caption_generation_prototype.md` — Step 7 / Step 7-1 prototype 결과
- `data/step7/step7_captions_raw.csv`, `data/step7/step7_captions_zscore.csv` — caption prototype (read-only)

본 문서가 **다루지 않는 것**:

- 실제 human review 실행
- review sample CSV 생성 (Step 8-1 범위)
- raw / zscore 최종 branch 선택
- final wording / final UX message 확정
- threshold / 모델 / feature 차원의 어떤 변경

---

## 1. Purpose

| 항목                | 내용                                                                                                                                                               |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Step 8 목적         | Step 7 caption prototype을 평가하기 위한 *기준과 절차*를 잠그는 것                                                                                                                |
| 본 단계가 *수행하는* 것    | 평가 차원 / 채점 rubric / 자동 사전점검 지표 / sampling 전략 / review schema / issue tag vocabulary / raw vs zscore 비교 정책 / acceptance gate / reviewer 지침 초안 / Step 8-1 산출 계획 정의 |
| 본 단계가 *수행하지 않는* 것 | caption 생성, sample CSV 생성, human review 실행, raw vs zscore 선택, final wording 확정                                                                                   |
| Step 8의 위상        | review *설계* 단계 — 이후 Step 8-1 sample extraction / Step 9 refinement / 최종 branch decision의 잠금 기준이 된다                                                               |
| 위반 시 처리           | 본 문서가 정의한 sampling / rubric / vocabulary는 Step 8-1 진입의 전제 조건이며, 변경 시 변경 영향이 어느 §에 미치는지 명시한 뒤 진입                                                                  |

---

## 2. Why Step 8 is needed

| # | 논리 | 근거 |
|---|---|---|
| 2-1 | Step 7 validation pass는 *정책 위반 0건*이라는 뜻이지, 좋은 caption이라는 뜻은 아니다 | Step 7 §11 — runtime 13개 항목 통과는 정책 차원의 floor만 보장 |
| 2-2 | caption 품질은 기술적 validation과 별도로 사람이 읽었을 때의 자연성 / 유용성 / 과장 방지 / schema 충실성으로 평가해야 한다 | Step 5 / Step 6은 표현 *카테고리* 정책만 다룸 — wording 자연성은 정책 외 |
| 2-3 | raw / zscore branch는 아직 final commit되지 않았다 | Step 4 §12 / Step 5 §9 #7 / Step 6 §12 C9 |
| 2-4 | 전체 9275개 caption을 전수 검토하는 것은 비효율적이므로 stratified sampling이 필요하다 | Step 7 §4 — branch당 9275 rows / unique caption 65~67개 |
| 2-5 | Step 8은 이후 Step 8-1 sample extraction / Step 9 refinement / final branch decision의 *기준 잠금* 단계다 | Step 7 §16 — 다음 단계 옵션 중 정량적 review의 입구 |

---

## 3. Evaluation scope

평가 *대상*과 평가하지 *않는* 대상을 분리한다.

| 구분 | 항목 | 비고 |
|---|---|---|
| 평가 대상 | `caption_ko` | 사용자에게 보이는 최종 한국어 문장 |
| 평가 대상 | `template_id` | 어느 template family로 발화되었는가 |
| 평가 대상 | `class_set_prediction` | caption이 class_set을 정확히 표현하는가 |
| 평가 대상 | `uncertainty_flags` | flag별 modifier가 caption에 반영되었는가 |
| 평가 대상 | `caption_confidence_level` | level에 맞는 표현 강도인가 |
| 평가 대상 | `no_call` | no_call에서 class 추정 / 시스템 결함 표현이 없는가 |
| 평가 대상 | `no_call_reason` | 사유 우선순위가 caption에 반영되는가 |
| 평가 대상 | `main_feature_phrase` | feature phrase가 맥락에 맞고 anchor 억제가 유지되는가 |
| 평가 대상 | raw vs zscore caption 차이 | 두 branch 차이의 정성 / 정량 |
| 평가 *제외* | model retraining | Step 4 결과 그대로 carry |
| 평가 *제외* | threshold recalibration | Step 4 후보값 carry |
| 평가 *제외* | new feature 계산 | Step 7 phrase bank 외 추가 없음 |
| 평가 *제외* | true class accuracy를 caption 품질의 *주된* 기준으로 사용 | Step 3 §11 / Step 4 §11 — single argmax 평가 부적절 |
| 평가 *제외* | final UX deployment wording | Step 5 §8 / Step 6 §11.3 NC-G4 |
| 평가 *제외* | raw / zscore final branch 선택 | Step 8 §10 / §15 |

---

## 4. Evaluation dimensions

각 차원은 1~5점 Likert scale로 평가한다.

| # | 차원 | 평가 질문 | 비고 |
|---|---|---|---|
| D1 | Policy compliance | Step 5 / Step 6 정책을 어기지 않는가 (no_call에서 class 추정 / `["C3","C4","C2"]`에서 C2 누락 / `anchor_unreliable`에서 억제 표현 누락 등) | 1점은 자동 fail (§5 참고) |
| D2 | Schema faithfulness | `class_set_prediction` / `uncertainty_flags` / `no_call_reason`이 caption에 *과장 / 재해석 없이* 반영되는가 | 단일 argmax-like 표현 / 과장 표현 감점 |
| D3 | Naturalness | 한국어 문장이 자연스러운가 (분석 보고서처럼 딱딱하지 않고, 반복 / 중복 표현이 없는가) | Step 7-1 R13에서 0건이지만 사람 평가 별도 |
| D4 | Caution / non-overclaiming | confidence level에 맞게 조심스럽게 표현하는가 (`confident`에서도 과도 단정 금지, `low` / `no_call`에서 무리한 판단 금지) | level과 강도의 *정합* |
| D5 | Usefulness | 사용자 / 연구자가 이해할 수 있는 정보를 주는가 (너무 일반적 / 빈 문장처럼 느껴지지 않는가) | "any caption"으로 축약되지 않는지 |
| D6 | Feature phrase appropriateness | `main_feature_phrase`가 caption 맥락과 맞는가, `anchor_unreliable`일 때 적절히 억제되었는가, heuristic을 measurement처럼 표현하지 않는가 | Step 5 §7 / Step 6 §10 |
| D7 | Branch preference | raw / zscore caption을 paired로 볼 때 어느 쪽이 더 적절한가 | `{raw, zscore, tie, cannot_judge}` |

---

## 5. Rating rubric

| 점수 | 기준 (D1~D6 공통) |
|---|---|
| 5 | 문제 없음 / 매우 적절 |
| 4 | 사소한 표현 문제만 있음 |
| 3 | 의미는 맞지만 어색하거나 정보성이 약함 |
| 2 | 정책 또는 schema 충실성에 *가까운* 문제가 있음 |
| 1 | 명백한 정책 위반 또는 misleading caption |

D7 (Branch preference)은 점수가 아닌 enum: `{raw, zscore, tie, cannot_judge}`.

### 5.1 자동 fail 조건 (D1 1점 강제 / 전체 caption fail)

| # | 조건 |
|---|---|
| F1 | `no_call == True`인 caption_ko에 어떤 클래스 추정 표현이 있음 |
| F2 | `class_set_prediction == ["C3","C4","C2"]`인 caption_ko에 C2 누락 |
| F3 | `class_set_prediction == ["C3","C4"]`인 caption_ko에 C2 등장 |
| F4 | `anchor_unreliable`인 caption_ko에 depth / recovery / bottom-transition 직접 표현 |
| F5 | `lateral_proxy_gyro` / knee valgus / 무릎 각도 / 무릎 외반 등 §6 §7 금지 표현 |
| F6 | `pred_argmax_debug` / threshold 후보 이름 / posterior 수치 직접 노출 |
| F7 | "확실히" / "100%" / "정답" / "모델 실패" / "데이터 오류" 등 시스템 결함 시사 |

자동 fail이 발생하면 D1 = 1로 강제 기록되며, 다른 차원의 점수와 무관하게
해당 caption은 전체 fail로 분류한다 (§11 critical issue rate에 반영).

---

## 6. Automatic pre-screening metrics

Step 8-1 sample CSV 생성 *전*에 raw / zscore caption CSV(read-only)에서
계산할 자동 요약 지표. human review를 *대체하지 않으며*, sampling 우선
순위 / pilot review 가시성 확보용이다.

| # | 지표 | 산출 대상 |
|---|---|---|
| M1 | row count | raw / zscore |
| M2 | unique caption count | raw / zscore (Step 7-1 결과: 65 / 67) |
| M3 | template_id distribution | raw / zscore |
| M4 | caption_confidence_level distribution | raw / zscore |
| M5 | class_set_prediction distribution | raw / zscore |
| M6 | no_call_reason distribution | raw / zscore (Step 4 §10 분해와의 일치 확인) |
| M7 | caption length distribution | 글자 수 기준 — outlier 확인 |
| M8 | duplicate caption frequency | unique caption별 row 수 (Top-N) |
| M9 | validation issue count | runtime 13개 항목별 fail 분해 (현재 0건) |
| M10 | R13 duplicate / technical wording hit count | unique caption 단위 (현재 0건) |
| M11 | raw vs zscore caption equality rate | 같은 sample_id에서 caption_ko가 *문자열* 동일한 비율 |
| M12 | raw vs zscore differing sample count | M11의 differing 부분 — D7 review 우선 |

주의:
- M1~M12는 모두 read-only 집계 — caption / schema output / threshold / feature 어떤 것도 갱신하지 않는다.
- M11 / M12는 Step 8-1 paired sample 추출의 입력이 된다.

---

## 7. Sampling strategy

전수(9275 × 2 = 18550 caption row) 검토는 비효율적이므로 stratified
sampling을 한다.

### 7.1 stratification 축

| 축 | 값 |
|---|---|
| branch | `raw`, `zscore` |
| split | `train`, `val`, `test` |
| posture | `SA`, `CA`, `HW` |
| caption_confidence_level | `confident`, `hedged`, `low`, `no_call` |
| class_set_prediction | 8개 whitelist (`["C2"]` / 그룹 / 페어 / `["C3","C4","C2"]` / `[]`) |
| template_id | T-CONF-1 / T-CONF-2 / T-HEDGE-1 / T-HEDGE-2 / T-HEDGE-3 / T-LOW-1 / T-LOW-2 / T-NC-POSTURE / T-NC-ANCHOR / T-NC-LOWCONF (현재 본 데이터에서는 6개만 발화) |
| no_call_reason | `posture_unknown`, `anchor_driven`, `low_confidence_no_class_set` |
| `anchor_unreliable` 여부 | `True` / `False` |
| `feature_phrase_source` | (column 기반 / `fallback` / `""`) |
| unique caption | unique caption 단위 sampling 보장 |

### 7.2 sampling 우선순위

| 우선순위 | 부분집합 | 사유 |
|---|---|---|
| 1 | unique caption 중심 sample | raw / zscore unique caption(65 / 67)을 *모두* 포함하거나 최소 1개씩 대표 sample 포함 — Step 7-1 wording 자연화 결과를 빠짐없이 점검 |
| 2 | high-frequency template | T-HEDGE-1 / T-NC-LOWCONF / T-CONF-1 — 전체 caption의 다수를 차지하므로 사용자 노출 빈도가 높음 |
| 3 | rare template | T-HEDGE-3 (`["C2"]` hedged, anchor 강등) / T-NC-ANCHOR (anchor-driven no_call) — 정책 차원의 *고-위험 boundary* 케이스 |
| 4 | no_call sample | `low_confidence_no_class_set` + `anchor_driven` 모두 포함 — class leakage 점검 |
| 5 | `["C3","C4","C2"]` sample | C2 보존 검증 — F2 risk 핵심 검사 |
| 6 | `["C3","C4"]` sample | C2 미포함 검증 — F3 risk 핵심 검사 |
| 7 | posture 균형 | SA / CA / HW 균형 — 자세 조건부 phrase 점검 |
| 8 | split 균형 | train / val / test 모두 포함하되, **test 중심 review subset**도 별도 설계 가능 (test = 1427) |
| 9 | paired (raw / zscore 동일 sample_id) | M11 / M12 기반 — D7 branch preference 검토 전용 |

### 7.3 권장 sample size (후보, Step 8-1에서 확정)

| 단계 | sample size | 비고 |
|---|---:|---|
| pilot review | 100~150 rows | rubric / vocabulary 검증 + 1~2명 reviewer로 빠른 turnaround |
| extended review | 300~500 rows | branch / template / class_set 균형 후 |
| unique caption review | 약 65~67 unique × 2 branch ≈ 130 entries | unique caption 단위 read-through — Step 7-1 wording 전체 커버리지 |
| paired review (raw vs zscore) | M12 differing sample 수에 따라 결정 | D7 전용 |

최종 sample size는 Step 8-1에서 확정한다 (본 §은 후보 범위만 제시).

---

## 8. Review dataset schema

Step 8-1에서 만들 review sample CSV의 컬럼 schema. *본 단계에서 CSV를
만들지 않는다 — 컬럼 schema만 잠근다.*

### 8.1 carry 컬럼 (Step 7 caption CSV에서 read-only로 가져옴)

| 컬럼 | 출처 | 비고 |
|---|---|---|
| `review_id` | Step 8-1 신규 | sample 단위 일관 식별자 |
| `branch` | Step 8-1 신규 | `raw` / `zscore` |
| `sample_id` | Step 7 caption CSV | rep 단위 식별자 |
| `split` | Step 7 caption CSV | train / val / test |
| `posture` | Step 7 caption CSV | SA / CA / HW |
| `class_id_for_audit_only` | Step 7 caption CSV `class_id` 재명명 | 평가 참고용 (§8.3 mode 정책) |
| `caption_ko` | Step 7 caption CSV | 평가 대상 |
| `template_id` | Step 7 caption CSV | |
| `class_set_prediction` | Step 7 caption CSV | |
| `uncertainty_flags` | Step 7 caption CSV | |
| `caption_confidence_level` | Step 7 caption CSV | |
| `no_call` | Step 7 caption CSV | |
| `no_call_reason` | Step 7 caption CSV | |
| `main_feature_phrase` | Step 7 caption CSV | |
| `anchor_reliability_bin` | Step 7 caption CSV `anchor_reliability` → bin (`high` / `mid` / `low`) | 수치 직접 노출 금지 (Step 5 §7 / Step 6 §10) |
| `anchor_unreliable` | Step 7 caption CSV `uncertainty_flags`에서 derive | bool |
| `feature_phrase_source` | Step 7 caption CSV | |
| `caption_validation_pass` | Step 7 caption CSV | 현재 모두 True |

### 8.2 reviewer 입력 컬럼 (빈 칸으로 출력하여 reviewer가 채움)

| 컬럼 | 타입 | 값 / 범위 |
|---|---|---|
| `rating_policy_compliance` (D1) | int | 1~5 |
| `rating_schema_faithfulness` (D2) | int | 1~5 |
| `rating_naturalness` (D3) | int | 1~5 |
| `rating_non_overclaiming` (D4) | int | 1~5 |
| `rating_usefulness` (D5) | int | 1~5 |
| `rating_feature_phrase_appropriateness` (D6) | int | 1~5 |
| `branch_preference` (D7) | enum | `raw` / `zscore` / `tie` / `cannot_judge` |
| `issue_tags` | JSON list | §9 closed vocabulary |
| `free_text_comment` | string | reviewer 자유기술 |
| `reviewer_decision` | enum | `accept` / `accept_with_minor_edit` / `revise_template` / `revise_policy` / `reject` |

### 8.3 reviewer mode 정책

| mode | `class_id_for_audit_only` 노출 | 권장 순서 |
|---|---|---|
| blinded | 숨김 (또는 mask) | **1순위** — caption faithfulness와 class correctness 혼동 방지 |
| audit | 표시 | 2순위 — blinded 결과 확정 *후* class 일치 점검에 한정 |

본 단계에서는 두 mode를 정의만 하며, 실제 reviewer 운영은 Step 8-1 /
Step 9에서 수행한다.

---

## 9. Issue tag vocabulary

reviewer가 사용할 closed vocabulary.

| # | tag | 의미 | criticality |
|---|---|---|---|
| T1 | `policy_violation` | Step 5 / Step 6 정책 위반 | **critical** |
| T2 | `schema_mismatch` | schema output을 caption이 잘못 표현 | high |
| T3 | `no_call_too_vague` | no_call 사유 정보가 부족 | medium |
| T4 | `no_call_too_strong` | no_call이 시스템 결함처럼 들림 | high |
| T5 | `overclaiming` | confidence level 대비 과도한 단정 | high |
| T6 | `too_technical` | 분석 용어 / 변수명 노출 | medium |
| T7 | `too_generic` | 너무 일반적이라 정보가 빈약 | medium |
| T8 | `awkward_korean` | 한국어 자연성 부족 | low |
| T9 | `repeated_wording` | 중복 / 반복 표현 (Step 7-1 R13의 사람 점검판) | low |
| T10 | `missing_C2_in_C3C4C2` | `["C3","C4","C2"]` caption에서 C2 누락 | **critical** |
| T11 | `class_set_not_preserved` | class_set의 멤버 일부가 누락되거나 치환됨 | high |
| T12 | `anchor_phrase_violation` | `anchor_unreliable`에서 depth / recovery / bottom-transition 직접 표현 | **critical** |
| T13 | `feature_phrase_mismatch` | feature phrase가 맥락 / level / 자세와 맞지 않음 | medium |
| T14 | `useful` | 정보성이 명확함 (긍정 tag) | — |
| T15 | `acceptable` | 사소한 문제 외에는 받아들일 만함 (긍정 tag) | — |
| T16 | `needs_minor_wording` | 작은 wording 다듬기만 필요 | low |
| T17 | `needs_template_revision` | template 자체 재설계 필요 | high |
| T18 | `needs_policy_revision` | Step 5 / Step 6 정책 자체 재검토 필요 | high |

복수 선택 가능. **critical** tag (T1 / T10 / T12)가 1건이라도 발생하면
해당 row는 D1 자동 fail (§5.1과 동치)로 처리한다.

---

## 10. Raw vs zscore comparison policy

### 10.1 비교 기준

| # | 비교 항목 | 산출 |
|---|---|---|
| C1 | caption distribution difference | M3 / M4 / M5 / M6 의 raw vs zscore 차이 |
| C2 | no_call difference | M6 — `anchor_driven` 25 vs 22 / `low_confidence_no_class_set` 2393 vs 2391 (Step 7 §9) |
| C3 | branch-specific naturalness issue | D3 / T8 / T9 reviewer 입력의 branch별 평균 |
| C4 | branch-specific feature phrase issue | D6 / T13 reviewer 입력의 branch별 평균 |
| C5 | human reviewer preference | D7 enum 분포 |
| C6 | validation issue count | M9 / M10 (현재 0건 / 0건) |
| C7 | unique caption coverage | M2 (raw 65 / zscore 67) — branch별 표현 다양성 |

### 10.2 D7 branch_preference 값 정의

| 값 | 의미 |
|---|---|
| `raw` | raw caption이 더 적절 |
| `zscore` | zscore caption이 더 적절 |
| `tie` | 차이 미미 / 동등 |
| `cannot_judge` | 판단 보류 (정보 부족 / 양쪽 모두 문제) |

### 10.3 본 단계의 비-결정

| 항목 | 본 단계 |
|---|---|
| raw vs zscore 최종 선택 | **수행하지 않음** — Step 8은 *기준만* 잠근다 |
| final branch decision | 후속 단계 (Step 9 또는 그 이후)로 지연 — human review 결과 누적 후 |
| Step 5 §9 #7 / Step 6 §12 C9 정책 | 유지 |

---

## 11. Acceptance / revision criteria

Step 7 caption prototype이 후속 단계로 통과할 수 있는 *gate* 후보. 본
숫자는 **초기 후보**이며, Step 8-1 / 8-2 pilot review 결과에 따라 조정
가능하다. **final success criterion이 아니라 prototype review gate**다.

| # | criterion | 후보 임계값 | 근거 |
|---|---|---|---|
| A1 | D1 (Policy compliance) 평균 | ≥ 4.5 | 정책 위반은 floor — 4.5 미만이면 template 재설계 |
| A2 | D3 (Naturalness) 평균 | ≥ 4.0 | 사용자 노출 가능 수준 |
| A3 | D5 (Usefulness) 평균 | ≥ 3.5 | 전체 보수적 어조라 평균이 낮을 수 있음 — 3.5를 후보로 |
| A4 | critical issue rate (T1 / T10 / T12) | = 0% | 단 1건도 허용하지 않음 |
| A5 | no_call class leakage (F1) | = 0 | 자동 fail — 1건이라도 fail |
| A6 | `["C3","C4","C2"]` C2 missing (F2 / T10) | = 0 | 자동 fail |
| A7 | anchor phrase violation (F4 / T12) | = 0 | 자동 fail |
| A8 | repeated_wording (T9) issue rate | ≤ 5% | Step 7-1 R13에서 0건이므로 사람 검토에서도 낮을 것으로 기대 |
| A9 | too_generic (T7) issue rate | ≤ 20% | usefulness 보수성 인정 |

A4 ~ A7이 0이 *아니면* prototype은 후속 단계로 진입하지 못하며, Step
8-1 / 8-2 결과에 따라:

| 결과 | 다음 액션 |
|---|---|
| A1~A9 모두 통과 | Step 9 (또는 그 이후) 진입 — 추가 wording 다듬기 / branch 결정 |
| A4~A7 중 1개 이상 fail | template / 정책 재검토 — Step 6 / 7 수정 후 prototype 재생성 |
| A1~A3 / A8~A9 일부 fail | wording 자연화 추가 (Step 7-2) 또는 acceptance gate 재조정 — pilot 결과 반영 |

---

## 12. Reviewer instruction draft

reviewer에게 줄 간단 지침. 본 단계는 *초안*이며, Step 8-1 pilot에서
어조 / 분량 / 예시를 다듬는다.

| # | 지침 |
|---|---|
| I1 | caption이 정답 class를 *맞히는지*보다, schema output을 *과장 없이* 잘 표현하는지를 평가한다. |
| I2 | `no_call`은 실패가 *아니라* 보수적 판단 보류로 평가한다 — class를 추정하지 않은 것이 정상 |
| I3 | class_set caption(`["C3","C4","C2"]` 등)은 *일부러* 단일 class로 좁히지 않은 것이다 — 모호성을 유지하는 것이 정책 |
| I4 | C1~C6 라벨은 내부 유형 코드로 본다 — 사용자 노출 wording은 별도 단계에서 다룰 수 있다 |
| I5 | feature phrase는 *보조 정보*이며, class 결정 근거 *전체*가 아니다 |
| I6 | raw / zscore는 *최종 선택 전 비교 중*이다 — D7에서 선호도를 표시하되 한 branch가 final이라고 가정하지 말 것 |
| I7 | "확실히" / "100%" / "정답" / "모델 실패" / "데이터 오류" 같은 어조는 정책 위반 — T1 또는 T4로 표시 |
| I8 | `["C3","C4","C2"]`에서 C2가 빠졌으면 즉시 T10 (critical) — 점수와 무관 |
| I9 | `anchor_unreliable`인 행에서 깊이 / 회복 / 전환 단정이 보이면 즉시 T12 (critical) |
| I10 | 자유기술(`free_text_comment`)에는 *어떤 표현이* 어떻게 어색한지 한 줄로 적는다 — 점수만으로는 후속 refinement에 부족 |

---

## 13. Planned Step 8-1 outputs

다음 단계인 Step 8-1에서 생성할 산출물. **본 단계에서는 어떤 파일도
만들지 않는다 — 계획만 기록한다.**

| 파일 | 역할 | 본 단계 상태 |
|---|---|---|
| `scripts/build_step8_review_sample.py` | Step 8 sampling 전략 / schema 적용하여 review sample CSV 생성 | 미작성 |
| `data/step8/step8_review_sample_raw.csv` | raw branch sampling 결과 | 미생성 |
| `data/step8/step8_review_sample_zscore.csv` | zscore branch sampling 결과 | 미생성 |
| `data/step8/step8_review_sample_paired.csv` | M11 / M12 paired (raw vs zscore differing) sample — D7 전용 | 미생성 |
| `reports/step8/step8_review_sample_summary.md` | sample 분포 / stratification 충족률 / 자동 metric 요약 | 미작성 |

Step 8-1 진입 전제:

- 본 §1~§12 정의가 *동결*된 상태에서 진입
- read-only 입력: `data/step7/step7_captions_raw.csv`, `..._zscore.csv`
- 새 모델 학습 / threshold recalibration / schema output 재생성 / 새 feature 계산 *없음*
- Step 1~7 산출물 *수정 없음*

---

## 14. Risks and limitations

| # | 리스크 | 영향 / 완화 |
|---|---|---|
| R1 | caption 다양성 부족 — template repetition으로 unique caption이 65~67개 수준 | M2 / §7.1 unique caption sampling으로 *모두* 검토 가능 |
| R2 | C1~C6 라벨이 사용자에게 직관적이지 않을 수 있음 | reviewer 지침 I4에서 내부 코드임을 명시. 사용자 노출 wording은 후속 단계에서 |
| R3 | human review 기준의 주관성 | rubric (§5) + 자동 fail (§5.1) + critical tag (§9)로 객관성 보강. inter-rater 비교는 Step 8-1 pilot에서 |
| R4 | feature phrase가 너무 일반적일 수 있음 (Step 7 phrase가 정성 카테고리 단위) | T7 (`too_generic`) tag로 측정. 임계값 A9 ≤ 20% |
| R5 | raw / zscore 차이가 작아 D7 branch preference가 뚜렷하지 않을 가능성 | M11 / M12로 차이 비율 사전 측정 — `tie` / `cannot_judge` 분포가 높으면 그대로 보고 |
| R6 | no_call 비율 약 26%로 높아 UX 관점에서 재설계가 필요할 수 있음 | T3 / T4로 측정. UX 결정은 본 단계 범위 외 (Step 5 §8 / Step 6 §11.3) |
| R7 | `class_id_for_audit_only`를 reviewer에게 보여주면 caption faithfulness와 class correctness가 섞일 위험 | mode 정책 (§8.3) — blinded 우선, audit는 후순위 |
| R8 | acceptance criteria의 후보 임계값(§11)이 데이터에 비해 빡빡할 수 있음 | pilot 결과 반영하여 Step 8-1에서 조정. *final success criterion이 아님* |

---

## 15. Step 8 completion criteria

Step 8은 **다음을 모두 충족할 때** 완료된 것으로 본다.

| # | 기준 | 본 문서의 충족 위치 |
|---|---|---|
| 1 | evaluation dimensions 정의 | §4 |
| 2 | rating rubric 정의 (자동 fail 조건 포함) | §5 / §5.1 |
| 3 | automatic pre-screening metrics 정의 | §6 |
| 4 | stratified sampling strategy 정의 | §7 |
| 5 | review dataset schema 정의 (carry / reviewer 입력 / mode 정책) | §8 |
| 6 | issue tag vocabulary 정의 | §9 |
| 7 | raw / zscore comparison policy 정의 | §10 |
| 8 | acceptance / revision criteria 정의 (gate 후보) | §11 |
| 9 | reviewer instruction draft 작성 | §12 |
| 10 | Step 8-1 outputs 계획 정의 | §13 |
| 11 | 실제 sample CSV 생성하지 *않음* | 본 문서 전체 (§13 미생성 명시) |

다음은 Step 8 완료 기준이 **아니다**:

| # | 비완료 기준 | 사유 |
|---|---|---|
| 1 | 실제 human review 실행 | Step 8-1 / 9 범위 |
| 2 | review sample CSV 생성 | Step 8-1 범위 |
| 3 | raw / zscore 최종 branch 선택 | 후속 단계 (Step 9 이후) |
| 4 | final wording / final UX message | Step 5 §8 / 후속 단계 |
| 5 | 새 모델 / threshold / schema output / feature | 본 단계 범위 외 |

---

## 16. Next step after Step 8

| 단계 | 내용 | 제약 |
|---|---|---|
| Step 8-1 — Review sample extraction | §13의 산출물 5개를 read-only join으로 생성 (caption CSV는 read만, 어떤 갱신도 없음). §7 sampling 전략과 §8 schema를 그대로 따름 | §1~§12 동결 전제. 변경 시 어느 §에 영향이 있는지 명시 후 진입 |
| Step 8-1 입력 | `data/step7/step7_captions_raw.csv`, `data/step7/step7_captions_zscore.csv` (read-only) | 두 branch 모두 read-only — 어느 한 branch도 final commit 아님 |
| Step 8-1 산출 | `data/step8/step8_review_sample_*.csv`, `reports/step8/step8_review_sample_summary.md` | 본 단계에서는 만들지 않음 |
| Step 9 이후 (선택) | pilot / extended human review 실행 / acceptance gate 평가 / wording refinement / branch decision | 본 단계 범위 외 |

본 문서 변경이 발생하면, 변경이 §4 (dimensions) / §5 (rubric) / §7
(sampling) / §8 (schema) / §9 (vocabulary) / §11 (gate) 중 어디에
영향을 주는지 명시한 뒤 Step 8-1 입력으로 사용한다.

---

*본 문서는 Step 8 caption evaluation / human review *설계* 문서로
작성되었다. 실제 human review / sample CSV 생성 / caption 재생성 / 새
모델 학습 / threshold recalibration / schema output 재생성 / 새
physical feature 계산 / raw vs zscore 최종 branch 확정 / final wording
확정은 *수행하지 않았다*. Step 1 / Step 2 / Step 3 / Step 4 / Step 5 /
Step 6 / Step 7 산출물은 수정되지 않았다. data/step7 caption CSV는
read-only 참고로만 사용했다.*
