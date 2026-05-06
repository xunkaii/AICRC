# Step 4 — 모델링 및 Calibration 실행 계획서

이 문서는 Step 4의 **실행 계획서**이다. 이 단계에서는 아직 모델을
학습하지 않고, 새 피처를 계산하지 않으며, 캡션 템플릿을 작성하지
않는다. 모델 학습 스크립트도 이 단계에서는 작성하지 않는다. 본
문서는 Step 4가 *어떤 입력으로 무엇을 산출하고 어떻게 검증할지*만
사전에 잠근다.

본 계획은 다음 두 문서에 종속된다. 두 문서와 충돌하는 결정은
계획에 포함될 수 없다:

- `reports/step3/step3_output_schema_uncertainty_policy.md` — 출력
  스키마 / 불확실성 정책 (특히 §3, §4, §6, §7, §8, §9, §10, §11).
- `reports/step4/step4_feature_model_decision_memo.md` — Step 4
  feature/baseline 결정 메모.

---

## 1. Step 4 목적 (Purpose)

Step 4의 목적은 **Step 3 스키마를 실제 모델 출력으로 채우는 것**이다.
단순 accuracy 최대화가 목적이 아니다.

Step 4가 끝났을 때, rep마다 다음을 모두 산출할 수 있어야 한다 (Step 3
§3의 emission 형태):

- `class_posterior` (C1..C6에 대한 적절한 분포)
- `class_set_prediction` (Step 3 §4의 허용 집합 중 하나, 또는 `[]`)
- `uncertainty_flags` (Step 3 §7의 닫힌 어휘)
- `caption_confidence_level` (Step 3 §8의 enum)
- `no_call` (Step 3 §9의 세 조건 중 하나)

Step 4는 위 다섯 필드를 *생성 가능한 상태*로 만들고, 그 calibration
근거를 보고한다. 단일 argmax 출력은 어떤 단계에서도 최종 출력으로
사용하지 않는다 (Step 3 §3, §11).

이 단계가 잠그지 **않는** 것:

- 캡션 표현 (Step 3 범위 외).
- 최종 정규화 (§5의 두 후보 중 어느 것이 다음 단계 후보로 살아남을지를
  *근거와 함께* 보고만 한다 — commit은 별도 결정).
- 성능 기준 / 성공 기준 (Step 3 §11에 의해 범위 외).

---

## 2. 입력 파일 (Inputs)

Step 4의 모든 산출물은 다음 입력만으로 생성된다. 이 외의 파일은
입력으로 사용하지 않는다.

| 파일 | 역할 |
|---|---|
| `data/manifest_split.csv` | rep 단위 키, posture, class_id, split (`v1_36_8_8`) |
| `data/step2/candidate_feature_bank.csv` | rep 단위 후보 피처 값 (Step 2.5-3 산출물) |
| `data/step2/normalization_feasibility_features.csv` | 정규화 후보 적용된 피처 값 (Step 2.5-3b 산출물) |
| `data/step2/bottom_event_audit.csv` | rep 단위 anchor 정보 및 anchor_reliability |
| `reports/step3/step3_output_schema_uncertainty_policy.md` | 출력 스키마 및 정책 (변경 불가) |
| `reports/step4/step4_feature_model_decision_memo.md` | feature 분류 및 baseline 모델 결정 (변경 불가) |

데이터 입력은 *읽기 전용*이다. Step 4는 어떤 입력 파일도 수정하지
않는다.

---

## 3. 출력 파일 (Outputs)

이 절은 두 종류의 산출물을 구분한다:
**(A) data / report 산출물 목록** — Step 4 실행이 *산출하는 결과
파일*들. 이 외의 결과 파일은 이 단계에서 생성하지 않는다.
**(B) 허용 implementation scripts 목록** — Step 4 실행을 *수행하기
위한 코드*. 결과 파일 산출과는 구분되며, 아래에 명시된 스크립트만
허용된다.

### 3.A data / report 산출물 (Step 4가 산출하는 결과 파일)

| 파일 | 형식 | 내용 |
|---|---|---|
| `data/step4/step4_modeling_dataset.csv` | CSV | manifest + 피처 + posture one-hot이 결합된 모델링 데이터셋 (raw 및 zscore 컬럼 모두 포함) |
| `data/step4/step4_predictions_raw.csv` | CSV | `raw + posture_as_input` baseline의 rep 단위 `class_posterior` |
| `data/step4/step4_predictions_zscore.csv` | CSV | `posture_train_zscore + posture_as_input` baseline의 rep 단위 `class_posterior` |
| `data/step4/step4_schema_outputs_raw.csv` | CSV | raw 결과를 Step 3 §3 emission 형태로 채운 rep 단위 출력 |
| `data/step4/step4_schema_outputs_zscore.csv` | CSV | zscore 결과를 Step 3 §3 emission 형태로 채운 rep 단위 출력 |
| `reports/step4/step4_modeling_calibration_results.md` | Markdown | §7 평가 지표 및 §8 임계값 calibration 보고 (한국어) |
| `reports/step4/step4_threshold_calibration.md` | Markdown | §8의 임계값 후보, 산출 절차, 채택 후보 보고 (한국어) |

모든 새 Markdown 산출물은 한국어로 작성한다.

### 3.B 허용 implementation scripts (Step 4 실행 코드)

Step 4 실행을 위해 **별도로** 작성이 허용되는 스크립트는 다음
다섯 개로 잠근다. 이 외의 스크립트는 Step 4 단계에서 작성하지
않는다.

| 스크립트 | 역할 |
|---|---|
| `scripts/build_step4_modeling_dataset.py` | §2 입력으로부터 §3.A의 `step4_modeling_dataset.csv` 산출 |
| `scripts/train_step4_baseline.py` | §6 baseline 학습 → `step4_predictions_raw.csv` 및 `step4_predictions_zscore.csv` 산출 |
| `scripts/calibrate_step4_thresholds.py` | §8 임계값 후보 산출 → `step4_threshold_calibration.md` 보고 입력 |
| `scripts/generate_step4_schema_outputs.py` | predictions + 임계값 후보 → §3.A의 `step4_schema_outputs_*.csv` 산출 |
| `scripts/validate_step4_schema_outputs.py` | §9 스키마 검증 전용. 학습/추론 로직 포함 금지. 결과는 `step4_modeling_calibration_results.md`에 첨부 |

`scripts/validate_step4_schema_outputs.py`는 **검증 전용**이다.
schema 출력 CSV를 읽어서 §9의 검증 항목을 통과/실패로 보고하는
것만을 책임진다. posterior를 재해석하거나 임계값을 재산출하지
않는다.

이 절의 분류(데이터 산출물 vs implementation script)는 §10 완료
기준 점검 시에도 유지된다.

---

## 4. Feature 집합 (Feature Set)

`reports/step4/step4_feature_model_decision_memo.md`에서 잠근 분류를
그대로 사용한다. 이 계획에서 분류를 변경하지 않는다.

**Main features (baseline 입력):**

- `motion_range_acc_z`
- `depth_proxy`
- `motion_range_gyro_mag`
- `bottom_stability_acc`
- `bottom_transition_delta_acc_z`

**Auxiliary / ablation:**

- `bottom_stability_gyro`
- `lateral_proxy_gyro` (heuristic 트랙 — measurement로 취급 금지)

**Uncertainty-only:**

- `anchor_reliability` (모델 입력 피처가 아니라 §8 신뢰도 강등 및
  §9 no-call 라우팅 입력으로만 사용)

**Hold-out:**

- `bottom_recovery_slope_acc_z` (baseline 입력에서 제외)

Auxiliary 후보의 ablation 비교 결과는 §7 보고서에 기록한다. ablation
결과로 분류가 바뀌는 결정은 이 단계에서 내리지 않는다 (별도 결정 메모
필요).

---

## 5. 정규화 후보 (Normalization Candidates)

Step 3 §10의 두 후보를 **모두** baseline에서 비교한다. 이 단계에서
어느 한쪽으로 commit하지 않는다.

1. `raw + posture_as_input` — 피처 값 통과 + posture one-hot
2. `posture_train_zscore + posture_as_input` — train 분할 자세
   조건부 z-score 통계만으로 정규화한 피처 + posture one-hot

`participant_zscore`는 Step 3 §11에 의해 명시적으로 제외되어 있다.
이 단계에서 어떤 형태로든 사용하지 않는다.

각 정규화 후보는 별도의 prediction CSV(§3)와 별도의 schema-output
CSV(§3)로 산출된다. 두 결과를 §7의 평가 지표에 따라 비교 보고한다.

---

## 6. Primary Baseline Model

`reports/step4/step4_feature_model_decision_memo.md` §7과 동일하게
잠근다.

| 항목      | 값                                                                           |
| ------- | --------------------------------------------------------------------------- |
| 모델      | multinomial logistic regression (다항, one-vs-rest 아님)                        |
| penalty | L2                                                                          |
| 클래스 가중  | `class_weight = balanced`                                                   |
| 입력      | §4의 main features 5개 + `posture` one-hot                                    |
| 출력      | `class_posterior` (C1..C6에 대한 적절한 분포)                                       |
| 분할      | `data/manifest_split.csv`의 train / val / test (`split_version = v1_36_8_8`) |

베이스라인 학습은 §5의 두 정규화 후보 각각에 대해 1회씩 수행하여
두 결과 세트를 산출한다 (총 2회 실행). 실제 학습 코드는 이 계획서가
잠긴 *다음 단계*에서 작성한다 — 본 계획서는 학습 코드를 포함하지
않는다.

`anchor_reliability`, `lateral_proxy_gyro`, `bottom_recovery_slope_acc_z`,
`participant_zscore`는 baseline 입력에서 명시적으로 제외된다.

---

## 7. 평가 지표 (Evaluation Metrics)

§5의 두 정규화 후보 각각에 대해 train / val / test split 모두에서
다음 지표를 산출한다. test 결과를 헤드라인 보고에 사용하되, val 결과는
임계값 calibration의 입력이 된다 (§8).

**기본 분류 지표:**

- accuracy
- macro F1
- weighted F1
- confusion matrix (6 × 6)

**Calibration / posterior 품질:**

- log loss (multinomial)
- Brier score (multi-class one-vs-rest 평균)
- Expected Calibration Error (ECE) 또는 reliability diagram 기반의
  calibration summary (15-bin 권장)

**Step 3 ambiguity-group 표현력 지표:**

- C2 recall
- C1 / C5 / C6 그룹 내 혼동률 (그룹 내부에서 *다른* 클래스로 분류된
  비율)
- C3 / C4 페어 혼동률
- C3 → C2 흡수율, C4 → C2 흡수율

**Anchor 의존성 점검:**

- `anchor_reliability` 구간별 성능 (예: 4분위 또는 [0, 0.25), [0.25, 0.5),
  [0.5, 0.75), [0.75, 1.0]). 각 구간에서 accuracy / macro F1 / log loss /
  C2 recall 보고. 이 지표는 §8의 `anchor_suppression_threshold` 및
  `anchor_no_call_threshold` 후보 산출에 사용된다.

지표는 §3의 `step4_modeling_calibration_results.md`에 표 형태로 정리한다.
단순 accuracy를 단독 결정 기준으로 사용하지 않는다 (decision memo §8).

---

## 8. 임계값 Calibration 대상 (Threshold Calibration Targets)

Step 3 정책이 요구하는 임계값들의 **수치 후보**를 이 단계에서 calibrate
하여 보고한다. 이 단계에서 임계값을 commit하지 않으며, 후보값과 그
근거를 보고하는 것이 목표이다.

| 임계값 | 정책상 위치 | calibration 입력 | 비고 |
|---|---|---|---|
| `confident_C2_threshold` | Step 3 §4 / §8 | val split의 C2 posterior 분포 + C2 recall–precision trade-off | C2가 단독 confident 집합으로 인정되는 최소 posterior |
| `non_trivial_C2_threshold` | Step 3 §4 (`[C3, C4, C2]` 헤지 진입 조건) | val split의 C2 posterior 분포 (top mass가 {C3, C4}일 때) | C2 posterior가 "무시할 수 없음"으로 판정되는 컷오프 |
| `within_group_threshold` | Step 3 §4 ({C1, C5, C6} 부분집합 진입 조건) | val split의 그룹 내 posterior 질량 분포 | {C1, C5, C6} 부분집합으로 좁힐 때 사용되는 컷오프 |
| `anchor_suppression_threshold` | Step 3 §6 | §7의 anchor_reliability 구간별 성능 | 이 값 미만 → `anchor_unreliable` 플래그 + 깊이/회복 언어 억제 (no-call 아님) |
| `anchor_no_call_threshold` | Step 3 §6, §9 | §7의 anchor_reliability 구간별 성능 + 앵커 의존적 클래스 집합 (`[C2]`, `[C3, C4, C2]`) 한정 분석 | `anchor_suppression_threshold`보다 더 낮은 값. 이 값 미만이고 *동시에* 앵커 의존적 클래스 집합인 경우에 한해 no-call |

각 임계값에 대해 후보값 1개 이상 + 근거(분포 그림 또는 표) +
sensitivity 검토(임계값을 ±일정 폭 이동시켰을 때의 영향)를 함께
보고한다.

`anchor_no_call_threshold`는 반드시 `anchor_suppression_threshold`보다
낮은 값으로 산출한다. 두 값의 대소 관계가 역전되는 후보는 채택하지
않는다 (Step 3 §6 위반).

---

## 9. Schema 검증 (Schema Validation)

`scripts/validate_step4_schema_outputs.py`는 다음 검증 항목을 모두
통과하는지를 보고한다. 어느 하나라도 실패하면 §10의 완료 기준을
충족하지 못한 것으로 처리된다.

검증 항목:

1. **`class_posterior` 분포성.** 모든 rep에서 C1..C6 6개 키가 모두
   존재하고, 비음수이며, 합이 수치적 허용오차(`abs(sum − 1.0) < 1e-6`)
   내에서 1이다.
2. **단일 argmax 출력 부재.** 어떤 산출물 컬럼도 `argmax(class_posterior)`
   기반의 단일 클래스 라벨을 *최종 출력 칼럼*으로 노출하지 않는다.
   (디버깅용 `argmax_class` 컬럼을 보관할 수 있으나 schema-output
   CSV의 권위 있는 클래스 출력은 `class_set_prediction`이다.)
3. **`class_set_prediction` 화이트리스트.** 값이 다음 8가지 집합
   중 정확히 하나이다 (그 외의 어떤 집합도 허용하지 않음):
   - `[C2]`
   - `[C1, C5, C6]`
   - `[C1, C5]`
   - `[C1, C6]`
   - `[C5, C6]`
   - `[C3, C4]`
   - `[C3, C4, C2]`
   - `[]`

   허용되는 C1/C5/C6 계열 부분집합은 정확히 위 네 가지
   (`[C1, C5, C6]`, `[C1, C5]`, `[C1, C6]`, `[C5, C6]`)이며,
   `[C1]` / `[C5]` / `[C6]` 같은 단일 원소 집합은 confident 수준에서
   허용되지 않는다 (Step 3 §4).

   어떤 부분집합을 실제로 선택할지는 posterior threshold 정책과
   `scripts/generate_step4_schema_outputs.py`의 책임이다.
   `scripts/validate_step4_schema_outputs.py`는 posterior를
   재해석해서 부분집합의 타당성을 판단하지 않으며, 위 화이트리스트
   안에 있는지만 검사한다.
4. **`anchor_reliability`의 트랙 분리.** `feature_evidence`에 해당하는
   컬럼/필드 어디에도 `anchor_reliability`가 포함되지 않는다.
   `anchor_reliability`는 자체 컬럼으로만 존재한다.
5. **`lateral_proxy_gyro` 트랙 분리.** schema-output CSV에서
   `lateral_proxy_gyro`가 measurement로 표기된 행이 0이다. 사용된
   경우 heuristic 버킷 컬럼/필드에만 존재한다 (Step 3 §5, §11).
6. **`participant_zscore` 미사용.** 어떤 출력 컬럼/메타데이터에도
   `participant_zscore` 정규화 라벨이 등장하지 않는다.
7. **No-call 라우팅 일관성.** `class_set_prediction == []`인 행은
   반드시 `no_call == true`이고 `caption_confidence_level == "no_call"`.
   역도 성립한다.
8. **No-call 진입 조건의 닫힘성.** `no_call == true`인 행은 반드시
   다음 중 하나의 사유 플래그를 가진다: `posture_unknown`,
   `low_confidence_no_class_set`, 또는 (`anchor_reliability` <
   `anchor_no_call_threshold` 후보값 *AND* 그 클래스 집합이 앵커
   의존적 분류로 표시됨). 그 외 사유로 no-call이 발생하지 않는다.
9. **`uncertainty_flags` 어휘 닫힘성.** 모든 플래그 값이 Step 3 §7의
   닫힌 어휘 안에 있다.
10. **`caption_confidence_level` enum 닫힘성.** 값이 `confident`,
    `hedged`, `low`, `no_call` 중 하나이다.

검증 결과는 `reports/step4/step4_modeling_calibration_results.md`에
요약 표(검증 항목 × {통과/실패})로 첨부한다.

---

## 10. Step 4 완료 기준 (Completion Criteria)

다음을 **모두** 충족할 때 Step 4가 완료된 것으로 본다.

1. **두 정규화 후보의 baseline 결과가 모두 생성됨.**
   `data/step4/step4_predictions_raw.csv` 및
   `data/step4/step4_predictions_zscore.csv`가 산출되고, §7의 평가
   지표가 두 결과 모두에 대해 보고됨.
2. **Posterior calibration 결과가 보고됨.**
   §7의 log loss, Brier score, ECE 또는 reliability diagram 기반
   calibration summary가 두 결과 모두에 대해
   `step4_modeling_calibration_results.md`에 정리됨.
3. **임계값 후보가 산출됨.**
   §8의 다섯 임계값 — `confident_C2_threshold`,
   `non_trivial_C2_threshold`, `within_group_threshold`,
   `anchor_suppression_threshold`, `anchor_no_call_threshold` — 각각에
   대해 후보값 1개 이상과 근거가
   `reports/step4/step4_threshold_calibration.md`에 보고되고,
   `anchor_no_call_threshold` < `anchor_suppression_threshold` 관계가
   유지됨.
4. **Schema-compliant 출력 CSV가 생성되고 검증을 통과함.**
   `data/step4/step4_schema_outputs_raw.csv` 및
   `data/step4/step4_schema_outputs_zscore.csv`가 산출되고,
   `scripts/validate_step4_schema_outputs.py`가 §9의 10개 항목을
   모두 통과로 보고함.
5. **다음 단계 정규화 후보 유지 근거가 정리됨.**
   두 정규화 후보 중 어느 것을 다음 단계 후보로 유지(또는 둘 다
   유지)할지에 대한 근거가
   `step4_modeling_calibration_results.md`에 정리됨. 이 단계가
   commit하지 않는 결정이라는 점을 본문에 명시함.

완료 기준이 **아닌** 것:

- 캡션 표현 결정.
- 최종 모델 commit (baseline 외 모델 비교는 별도 단계).
- 성능 기준 / 성공 기준 commit.
- 단일 정규화 후보로의 commit.

---

*Step 4 모델링 및 calibration 실행 계획서로 작성됨. 이 문서는 새로운
피처를 계산하지 않으며, 모델 학습 스크립트를 포함하지 않으며, 모델을
학습하지 않으며, 캡션 템플릿을 작성하지 않는다. Step 1 / Step 2 /
Step 2.5 / Step 3 산출물 및 `reports/step4/step4_feature_model_decision_memo.md`는
수정되지 않는다.*
