# Step 4R — Feature-based Stronger Sensor-to-Schema Pilot 계획서

이 문서는 Step 4R-0의 **설계 계획서**이다. 본 단계에서는 학습 스크립트를
작성하지 않으며, 모델을 학습하지 않고, feature set / normalization /
threshold / caption template을 최종 확정하지 않는다. 본 문서는 Step 4R이
*어떤 입력으로 무엇을 산출하고 어떻게 검증할지*만 사전에 잠근다.

본 계획은 다음 문서에 종속된다. 이들과 충돌하는 결정은 본 계획에 포함될
수 없다:

- `reports/step3/step3_output_schema_uncertainty_policy.md` — 출력
  스키마 / 불확실성 정책.
- `reports/step4/step4_modeling_calibration_plan.md` — Step 4 LR baseline
  실행 계획.
- `reports/step4/step4_feature_model_decision_memo.md` — feature/baseline
  결정 메모.

기존 Step 1~8 산출물은 수정하지 않는다.
`data/step4/step4_modeling_dataset.csv`는 읽기 전용으로 사용한다.

---

## 1. 목적 (Purpose)

- AICRC_v2는 end-to-end sensor-to-text generation 프로젝트가 **아니다**.
  본 연구의 목표는 wrist IMU squat signal을 uncertainty-aware schema로
  변환하고, 그 schema를 guarded Korean caption으로 바꾸는 것이다.
- Step 4의 LR baseline은 완료되었으나, weak-to-moderate 수준의
  **reference model**로만 유지한다. main pipeline 모델로 사전 확정하지
  않는다.
- Step 4R의 목적은 **feature-based stronger model**이 동일한
  uncertainty-aware schema를 더 잘 채울 수 있는지를 파악하는 것이다.
- 평가의 목표는 단일 accuracy 상승이 아니라, 동일한 sensor-to-schema
  과제에서 후보 모델들을 **3층 기준(분류 / calibration / schema 출력
  품질)** 으로 비교하는 것이다.
- raw sequence model (LSTM, Transformer 등)은 Step 4R에서 다루지
  **않으며** future work 또는 optional extension으로 분리한다.

---

## 2. Scope

### 2.1 In scope

- `data/step4/step4_modeling_dataset.csv` 사용 (읽기 전용).
- `raw` branch 와 `zscore` branch 두 조건 비교.
- main 5 handcrafted features + posture one-hot 입력.
- sklearn 기반 feature model 비교 (LR / SVM / RandomForest /
  HistGradientBoosting / MLP).
- 기존 Step 4 LR baseline 포함 (reference 재현).
- calibration, class-set / no-call, ambiguity 구조 평가.
- participant-level split (Step 4에서 잠근 v1_36_8_8) 그대로 사용.

### 2.2 Out of scope

- raw sequence deep learning (LSTM, Transformer, 1D CNN 등).
- end-to-end text generation.
- `caption_ko`를 학습 target으로 사용하는 것.
- Step 7 caption을 ground truth 또는 weak supervision으로 사용하는 것.
- human review 결과를 sensor-grounded correctness 평가로 해석하는 것.
- feature set 최종 확정.
- threshold 최종 확정.
- `participant_zscore`를 main pipeline 후보로 사용하는 것.
- `anchor_reliability`를 motion feature로 사용하는 것
  (uncertainty evidence 전용).

---

## 3. Research Framing

기존 wearable IMU 연구들은 스마트워치 IMU 신호로 스쿼트의 fine-grained
classification을 수행했으며, RandomForest 와 CNN 등 다양한 모델 군을
비교해 왔다. 그러나 AICRC_v2의 목표는 단순한 6-class argmax
classification이 **아니다**.

본 연구의 목표는 wrist IMU 신호로부터 **불확실성을 포함한 schema**를
생성하고, 이를 과장 없는 한국어 설명으로 변환하는 것이다. 따라서 모델
평가는 macro F1 단일 지표로 끝나면 안 되며, calibration 품질과 schema
출력 행동을 함께 평가해야 한다.

Step 4R은 sensor-to-schema pipeline의 **모델링 중간 단계**이며, 모델
교체 여부는 분류 성능뿐 아니라 schema 출력 품질과 논문 주장의 정합성
(Step 3의 uncertainty-aware explanation policy 와의 일관성)으로 판단한다.

---

## 4. 입력 데이터 및 Feature Branch

### 4.1 입력 데이터

| 항목 | 값 |
|---|---|
| dataset path | `data/step4/step4_modeling_dataset.csv` |
| target column | `class_id` (C1~C6) |
| split column | `split` (train / val / test) |
| participant split | train 36명 / 6,412행, val 8명 / 1,436행, test 8명 / 1,427행 |

### 4.2 Raw branch feature

| feature | 역할 |
|---|---|
| `motion_range_acc_z` | 수직 가속도 동적 범위 |
| `depth_proxy` | 하강 깊이 프록시 |
| `motion_range_gyro_mag` | 각속도 크기 동적 범위 |
| `bottom_stability_acc` | bottom 구간 안정성 |
| `bottom_transition_delta_acc_z` | bottom 전환 구간 변화량 |
| `posture_SA`, `posture_CA`, `posture_HW` | posture one-hot |

### 4.3 Zscore branch feature

| feature | 역할 |
|---|---|
| `motion_range_acc_z_zscore` | (raw 대응) zscore 정규화 |
| `depth_proxy_zscore` | (raw 대응) zscore 정규화 |
| `motion_range_gyro_mag_zscore` | (raw 대응) zscore 정규화 |
| `bottom_stability_acc_zscore` | (raw 대응) zscore 정규화 |
| `bottom_transition_delta_acc_z_zscore` | (raw 대응) zscore 정규화 |
| `posture_SA`, `posture_CA`, `posture_HW` | posture one-hot (공통) |

### 4.4 사용하지 않는 컬럼

| 컬럼 | 제외 이유 |
|---|---|
| `anchor_reliability` | uncertainty evidence 전용 (motion feature 아님) |
| `bottom_stability_gyro` | auxiliary, ablation 전용 (CSV에 존재 시) |
| `lateral_proxy_gyro` | auxiliary, ablation 전용 (CSV에 존재 시) |
| `bottom_recovery_slope_acc_z` | hold-out (CSV에 존재 시) |
| `participant_zscore` 계열 | main pipeline 후보 제외 |

CSV 헤더 확인 결과 main 사용 컬럼은 위에 모두 존재한다.
auxiliary/hold-out 컬럼이 현재 CSV에 미포함된 경우 Step 4R 본 실행 시
ablation에서 자연스럽게 빠진다. **TODO:** Step 4R 스크립트 작성 시
`anchor_type` 등 metadata 컬럼의 허용 여부를 재확인한다.

---

## 5. 후보 모델

### 5.1 Logistic Regression (M0, reference)

- Step 4 baseline 재현용.
- calibration 및 해석 가능성의 기준점.
- 이미 완료된 결과를 reference로 유지하며, 새로운 후보 모델로 취급하지
  않는다.

### 5.2 SVM (M1)

- 작은 feature space에서 non-linear boundary 가능성 확인.
- `probability=True` 는 posterior 산출을 위해 사용하되, 학습 시간이
  길어질 수 있으므로 §6 pilot pre-check 결과가 낮으면 grid 탐색에서
  제외할 수 있다.
- 필요 시 Platt scaling 등 calibration 보정의 후속 적용 여부를 §8.2
  결과로 판단한다.

### 5.3 RandomForest (M2)

- 비선형 feature interaction 확인.
- feature importance 해석 가능.
- 과적합 및 calibration 약점에 주의 (probability output이 본질적으로
  잘 calibrate 되지 않음).

### 5.4 HistGradientBoosting (M3)

- tabular feature 기반 강력한 후보.
- class imbalance 처리(`class_weight` 또는 sample weighting)와
  calibration 안정성 확인 필요.

### 5.5 MLPClassifier (M4)

- 작은 tabular neural baseline.
- 성능이 높게 나와도 calibration과 안정성을 별도로 검토한다.
- `early_stopping=True` 사용 권장.

---

## 6. Pilot Pre-check (전체 grid 탐색 전 선행 실험)

전체 hyperparameter grid 탐색 전에 다음 pilot pre-check을 먼저 수행한다.

- **목적**: 각 후보 모델의 상한선(upper bound) 추정.
- **방법**: 각 후보 모델 default hyperparameter 로 5-fold cross-validation
  (train split 내부만 사용, val/test 절대 노출 금지).
- **지표**: macro F1, weighted F1, log loss.

### 결과 해석 기준

- M0(LR) 대비 macro F1 향상이 **0.03 미만**이면 해당 모델의 grid 탐색을
  생략할 수 있다.
- 향상이 있는 모델만 §7 hyperparameter grid 탐색으로 진행한다.
- pilot pre-check 결과는
  `reports/step4r/step4r_pilot_precheck_results.md`에 기록한다.
  본 파일은 Step 4R 스크립트 실행 후 작성한다. 현재 단계에서는 계획만
  명시한다.

---

## 7. Hyperparameter Grid

모든 grid 탐색은 `GridSearchCV(cv=5, scoring="f1_macro")` 로 train split
내부에서만 수행하며, 최적 hyperparameter는 **val split macro F1**로 최종
선택한다.

### 7.1 M0 (LR reference)

- 기존 Step 4 LR 결과를 우선 재사용한다.
- 필요한 예측값 또는 calibration 지표가 없을 때만 동일 split / 동일
  feature 조건으로 reference 재현을 수행한다.
- 새로운 후보 모델로 취급하지 않는다.

### 7.2 M1 (SVM)

| hyperparameter | 후보 |
|---|---|
| `C` | [0.1, 1.0, 10.0] |
| `kernel` | rbf (고정) |
| `gamma` | scale (고정) |
| `probability` | True (고정, pre-check 결과가 낮으면 grid 탐색 제외 가능) |

### 7.3 M2 (RandomForest)

| hyperparameter | 후보 |
|---|---|
| `n_estimators` | [100, 300] |
| `max_depth` | [None, 10] |
| `min_samples_leaf` | [1, 3] |
| `class_weight` | balanced_subsample (고정) |

### 7.4 M3 (HistGradientBoosting)

| hyperparameter | 후보 |
|---|---|
| `max_iter` | [100, 300] |
| `max_depth` | [None, 5] |
| `learning_rate` | [0.1, 0.05] |
| `l2_regularization` | [0.0, 1.0] |

### 7.5 M4 (MLP)

| hyperparameter | 후보 |
|---|---|
| `hidden_layer_sizes` | [(64,), (128,), (64, 32)] |
| `activation` | relu (고정) |
| `alpha` | [0.0001, 0.001] |
| `early_stopping` | True (고정) |

---

## 8. 평가 지표 (4층 구조)

### 8.1 분류 성능 (Classification)

- accuracy (참고용)
- macro F1 (primary)
- weighted F1
- per-class precision / recall / F1
- confusion matrix
- C1 / C5 / C6 내부 혼동 비율
- C3 / C4 pair 혼동 비율
- C3 / C4 → C2 흡수(absorption) 비율

### 8.2 Calibration

- log loss
- Brier score (multi-class)
- Expected Calibration Error (ECE)
- reliability diagram (10-bin)
- top-1 probability 분포
- top-1 / top-2 margin 분포
- val / test ECE 비교 (과적합 탐지)

### 8.3 Schema 출력 행동 (Schema Behavior)

- `class_set` size 분포
- `no_call` 발생률 (전체 / split / class / posture별)
- `no_call` 안정성 (`low_confidence` vs anchor-driven 비율)
- `caption_confidence_level` 분포 (confident / hedged / low / no_call)
- `uncertainty_flags` 분포
- C1 / C5 / C6 within-group ambiguity 보존 여부
- C3 / C4 `pair_plus_c2_absorption` hedge 필요성 유지 여부

### 8.4 Output Quality Check

- schema validation pass rate
- required field 누락 없음
- invalid `class_set` 없음 (whitelist 8종 외 허용 안 됨)
- unsupported `caption_confidence_level` 없음
- `no_call=true` 와 confident caption 동시 존재 불가
- `anchor_reliability`가 motion feature가 아닌 uncertainty evidence로만
  사용됨을 코드 레벨에서 확인

---

## 9. 성공 기준

macro F1 단일 지표로 판단하지 않는다.

### 9.1 최소 성공

- M0(LR) 대비 macro F1 또는 calibration 중 **하나 이상** 개선.
- schema validation fail 0건.
- no-call / class-set behavior가 collapse하지 않음.
- C1 / C5 / C6, C3 / C4, C2 absorption 구조가 명시적으로 문서화됨.

### 9.2 강한 성공

- macro F1이 M0 대비 **test 기준 0.03 이상** 명확히 개선.
- log loss / Brier score 개선.
- class-set / no-call이 불확실한 샘플에서 의미 있게 작동함을 정량
  근거로 보여줌.
- ambiguity group 구조가 Step 2.5 / Step 3 해석과 모순되지 않음.

### 9.3 실패로 보는 경우

- macro F1만 소폭 오르고 ECE가 크게 나빠짐.
- 대부분 샘플이 단일 confident class로 collapse.
- 대부분 샘플이 no-call로 collapse.
- C1 / C5 / C6 또는 C3 / C4 ambiguity를 무시하고 과신.
- `caption_ko`를 학습 target으로 사용하는 방향으로 흐름.

---

## 10. Pilot 이후 의사결정 분기 (Decision Rule)

Step 4R pilot 완료 후 다음 세 결론 중 **하나**를 내린다.

### A. Feature-based stronger model이 thesis pipeline에 충분한 경우

- Step 4R 선택 모델이 main sensor-to-schema model로 승격됨.
- Step 5~7 guarded schema-to-caption layer는 그대로 유지됨.

### B. Feature-based model이 소폭만 개선되지만 schema behavior가 유의미한 경우

- Feature-based model을 유지하되, 논문 기여를 uncertainty-aware 설명
  쪽으로 강조.
- "high classification accuracy" 주장은 하지 않음.

### C. Feature-based model이 분류와 schema behavior 모두 미흡한 경우

- 바로 caption generation 으로 이동하지 않음.
- raw sequence model은 future work 또는 optional extension으로만 남김.
- feature evidence와 ambiguity policy 재검토를 먼저 권장함.

---

## 11. 예상 산출물 (Expected Outputs)

아직 생성하지 않는다. 스크립트 작성 후 생성한다.

### 11.1 `reports/step4r/`

- `step4r_pilot_modeling_plan.md` (본 파일)
- `step4r_pilot_precheck_results.md`
- `step4r_model_comparison_summary.md`

### 11.2 `data/step4r/`

- `step4r_predictions_{branch}_{model_id}.csv`
  - 포함 컬럼: `sample_id`, `rep_id`, `participant_id`, `class_id`,
    `posture`, `split`, `anchor_reliability`, `anchor_type`,
    `p_C1`~`p_C6`, `pred_argmax_debug`
- `step4r_schema_outputs_{branch}_{model_id}.csv`
- `step4r_model_comparison_metrics.csv` (모델 × branch × 지표 비교
  테이블)
- `step4r_calibration_summary.csv`
- `step4r_schema_behavior_summary.csv`

### 11.3 `models/step4r/`

- `{branch}_{model_id}.joblib`

---

## 12. 권장 실행 순서

1. `data/step4/step4_modeling_dataset.csv` columns 확인.
2. raw / zscore feature column set 정의.
3. M0(LR) baseline 재현 확인.
4. §6 Pilot pre-check 실행 (5-fold, train split 내부).
5. 향상이 있는 모델만 §7 hyperparameter grid 탐색.
6. 동일 split 기준으로 후보 모델 학습.
7. §8 4층 평가 지표 전체 산출.
8. Step 3 threshold policy를 기준으로 schema output 생성 및 검증.
9. schema behavior 비교 (분류 점수만 비교 금지).
10. A / B / C 결론 도출 → Step 4R-1 진행 또는 feature-based ceiling
    선언.

---

## 13. 위험 요소 및 대응 (Risks and Controls)

| 위험 | 대응 |
|---|---|
| handcrafted feature ceiling (소수 feature의 표현력 한계) | pilot pre-check으로 상한 추정 후 진행 여부 결정 |
| participant-level generalization 실패 | test split이 별도 피험자 8명으로 구성되어 있어 leak 없음을 split 컬럼으로 확인 |
| class ambiguity (C1/C5/C6, C3/C4 혼동) | 3층 성공 기준에서 ambiguity 구조 보존을 명시적으로 평가 |
| calibration 실패 (과신 또는 flat posterior) | ECE + reliability diagram으로 정량 평가, Platt / temperature scaling 후보 명시 |
| no-call collapse (대부분 샘플 no-call) | no_call 발생률 분포 모니터링, threshold sensitivity 확인 |
| sensor-to-text 과장 | `caption_ko`는 guarded output이며 학습 target이 아님을 본 문서에 명시 |
| template caption을 language output으로 혼동 | Out of scope (§2.2)에 명시, Final Note (§16)에서 재확인 |

---

## 14. 논문 챕터 연결

Step 4R 결과는 논문 Chapter 4 (Modeling) 에 반영된다:

- §4.1: uncertainty-aware schema-grounded pipeline 개요
- §4.2: feature-based baseline LR (Step 4) 결과
- §4.3: feature-based stronger model 비교 결과 (Step 4R)
- §4.4: 선택 모델의 calibration 및 schema 출력 품질

지도교수와의 챕터 구조 합의는 Step 4R 스크립트 작성 **전에** 선행을
권장한다. **TODO:** 챕터 구조 합의 일자 기록.

---

## 15. 완료 체크리스트

다음 항목이 모두 충족되면 Step 4R pilot을 완료로 간주한다:

- [ ] §6 pilot pre-check 결과 문서 작성 완료
- [ ] §7 hyperparameter grid 탐색 완료 (향상 모델 한정)
- [ ] §8 4층 평가 지표 코드로 구현 완료
- [ ] §9 성공 기준 3단계 달성 여부 확인 완료
- [ ] §10 A/B/C 결론 도출 완료
- [ ] §11 예상 산출물 전체 생성 완료
- [ ] `reports/step4r/step4r_model_comparison_summary.md` 작성 완료
- [ ] git commit (`reports/step4r/` + `data/step4r/` + `models/step4r/`
      포함)

---

## 16. Final Note

Step 4R is a sensor-to-schema pilot, not a sensor-to-text generation experiment.
Caption_ko remains a guarded output layer built on rule/template logic and must not be treated as a primary learning target.
Model selection in Step 4R is governed not by classification accuracy alone, but by calibration quality, schema behavior stability, and consistency with the uncertainty-aware explanation policy defined in Step 3.
