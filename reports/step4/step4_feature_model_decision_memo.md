# Step 4 — Feature 후보 및 Baseline Model 결정 메모

이 문서는 **Step 4 계획서가 아니다.** Step 4 모델링 작업이 시작되기
전에, 어떤 피처 후보를 메인 트랙에 두고 어떤 후보를 ablation/auxiliary로
둘지, 그리고 어떤 모델을 baseline으로 잡을지를 결정 메모로 잠그는
것이 목적이다.

근거는 모두 Step 2.5의 감사 산출물에 있으며, 이 메모는 그 근거를
재도출하지 않는다. 이 메모는 새 피처를 계산하지 않고, 모델을 학습하지
않으며, 캡션을 작성하지 않는다. Step 3의 출력 스키마 / 불확실성
정책 또한 변경하지 않는다 (참고: `reports/step3/step3_output_schema_uncertainty_policy.md`).

---

## 1. 목적 (Purpose)

다음을 잠그기 위해 작성된다:

- Step 4 모델링이 사용할 **메인 피처 후보 집합**과 그 분류 구조
  (main / auxiliary / uncertainty-only / hold-out).
- Step 4 모델링이 처음 비교 기준으로 사용할 **primary baseline
  model**.
- Step 4 모델 선택의 **기준** — 단순 accuracy가 아니라 Step 3
  스키마와의 호환성에 우선순위를 두기 위한 기준.

이 메모가 잠그지 않는 것:

- 최종 피처 집합 (Step 4 ablation 결과에 의해 줄어들 수 있음).
- 최종 정규화 (Step 3 §10의 두 후보 중 Step 4가 commit).
- 최종 모델 아키텍처 (이 메모는 *baseline*을 잠글 뿐, baseline
  이외의 모델 선택을 미리 차단하지 않음).
- Step 3 §3–§9의 출력/불확실성 정책 (이 메모는 그 정책에 적합한
  피처/모델만 제안한다).

Step 4는 이 메모의 분류와 baseline에서 시작하며, 분류를 변경하려면
별도의 결정 메모가 필요하다.

---

## 2. Step 4 메인 피처 후보 (Main Feature Candidates)

Step 4 모델링의 1차 입력 피처 집합. 모두 Step 3의 `feature_evidence.measurements`
버킷에 들어갈 자격이 있다.

| 피처 | 설명 (요약) | 정규화 의존성 |
|---|---|---|
| `motion_range_acc_z` | rep 전반 acc_z의 p95 − p5 | 자세 영향 낮음. raw / posture_train_zscore 모두 가능 |
| `depth_proxy` | [anchor ± 5] 구간의 평활화된 acc_z 평균 | 자세에 의해 지배됨. **`posture_train_zscore` 권장** |
| `motion_range_gyro_mag` | rep 전반 |gyro|의 동적 범위 | 자세 간 스케일 차이 → 자세 인식 처리 필요 |
| `bottom_stability_acc` | anchor 근처 acc 신호의 안정도 | 자세 간 스케일 차이 → 자세 인식 처리 필요 |
| `bottom_transition_delta_acc_z` | acc_z의 post-anchor 평균 − pre-anchor 평균 | 깊이/방향 변화 혼합 → 자세 인식 처리 필요. `depth_proxy`와 함께 해석 |

이 5개는 Step 4가 baseline 모델 학습 시 동시에 입력으로 사용한다.
이 중 일부가 ablation 결과 제거되는 것은 Step 4의 정상 결과이며,
이 메모는 그 결정을 미리 내리지 않는다.

---

## 3. Auxiliary / Ablation 후보 (Auxiliary Candidates)

메인 후보보다 한 단계 약한 자격으로 보유되며, baseline에 기본 포함되지
**않는다**. Step 4 ablation 비교에서만 들여다본다.

| 피처                      | 분류                                                                               |
| ----------------------- | -------------------------------------------------------------------------------- |
| `bottom_stability_gyro` | auxiliary measurement (HW에서 gyro 이벤트 mismatch 위험)                                |
| `lateral_proxy_gyro`    | **heuristic only** — Step 3 §5에 따라 `feature_evidence.heuristics`로 분류, 측정으로 제시 금지 |

`lateral_proxy_gyro`는 ablation에서 모델 입력으로 들어갈 수 있으나,
출력 스키마상 휴리스틱 트랙에 머물러야 하며 캡션 표현 시 측정처럼
제시되어서는 안 된다 (Step 3 §5, §11 위반 금지).

---

## 4. Uncertainty-only 변수

| 변수                   | 분류                                                                                              |
| -------------------- | ----------------------------------------------------------------------------------------------- |
| `anchor_reliability` | **모션 피처 아님.** Step 3 §6에 따라 불확실성 트랙 전용 — `feature_evidence`에 들어가지 않으며, 모델의 메인 입력 피처로도 사용하지 않는다. |

Step 4 baseline 모델은 `anchor_reliability`를 입력 피처로 받지 않는다.
이 변수는 §8의 신뢰도 강등 및 §9의 no-call 정책 라우팅에만 사용된다.
Step 4가 추후 모델 입력으로 추가하려면 별도의 결정 메모가 필요하며,
그 경우에도 §5의 measurement/heuristic 분류 변경은 허용되지 않는다.

---

## 5. Hold-out 피처

| 피처 | 분류 |
|---|---|
| `bottom_recovery_slope_acc_z` | **Step 4 메인 피처에서 제외 (hold-out).** 아이디어는 유지하지만 현재 6-샘플 window 정의가 좁고 split-unstable로 플래그됨. 재정의 후 별도 평가 |

Hold-out은 폐기가 아니다. Step 4의 baseline 학습 입력으로는 사용하지
**않는다**. 윈도우 정의가 재검토되어 split-stable해진 뒤에 다시 메인
또는 auxiliary 후보로 승격할 수 있다. 그 승격 결정 또한 Step 4가
직접 내리지 않으며 별도의 결정 메모가 필요하다.

---

## 6. Rationale (피처별 근거)

각 항목은 Step 2.5 산출물의 *근거 파일*을 명시한다. 이 메모는 그
근거를 재해석하거나 수정하지 않는다.

### 6.1 메인 후보

- **`motion_range_acc_z`** — anchor-independent이며 가장 안정적인
  후보로 관찰되었다.
  - 근거: `reports/step2/3_feature_bank/candidate_feature_bank_audit.md`
    (계산 성공률 9275/9275, 안정적 분포).
  - 근거: `reports/step2/3b_normalization/normalization_feasibility_audit.md`
    (raw에서도 자세 η²=0.061로 낮고, posture_train_zscore에서
    posture η²=0.000으로 본질적 제거).
  - 종합 참조: `reports/step2/step25_final_synthesis.md` §5, §6.

- **`depth_proxy`** — 깊이 관련 후보지만 자세에 의해 지배되므로
  **자세 인식 정규화가 필수**.
  - 근거: `reports/step2/3b_normalization/normalization_feasibility_audit.md`
    (raw posture η²=0.862 → posture_train_zscore에서 0.002, 클래스
    η² 7× 상승).
  - 근거: `reports/step2/3_feature_bank/candidate_feature_bank_audit.md`
    (피처 정의 및 계산 성공률).
  - Step 3 §10의 `posture_train_zscore + posture_as_input` 후보가
    이 피처에 가장 정합적임. 종합 참조: `reports/step2/step25_final_synthesis.md`
    §5, §6.

- **`motion_range_gyro_mag`** — 메인 입력 후보이지만 자세 간 스케일
  차이가 있어 자세 인식 처리가 필요한 measurement.
  - 근거: `reports/step2/3_feature_bank/candidate_feature_bank_audit.md`
    (자세별 분포 차이 관찰).
  - 종합 참조: `reports/step2/step25_final_synthesis.md` §5.

- **`bottom_stability_acc`** — anchor 근처 신호 안정도 측정.
  메인 후보이지만 자세 인식 처리 후에 자세 간 비교가 가능.
  - 근거: `reports/step2/3_feature_bank/candidate_feature_bank_audit.md`.
  - 종합 참조: `reports/step2/step25_final_synthesis.md` §5.

- **`bottom_transition_delta_acc_z`** — 깊이와 방향 변화를 혼합하는
  측정으로, `depth_proxy`와 함께 해석되어야 함. 따라서 메인 후보로
  유지하되 단독 해석은 금지.
  - 근거: `reports/step2/3_feature_bank/candidate_feature_bank_audit.md`
    (혼동 가능성 명시).
  - 시간 국소화 근거: `reports/step2/1_time_localized/time_localized_class_difference_summary.md`
    (η²(acc_z)가 bottom 직후 정점을 가지는 패턴이 transition delta에
    동기 부여를 제공).
  - 종합 참조: `reports/step2/step25_final_synthesis.md` §3, §5.

### 6.2 Auxiliary

- **`bottom_stability_gyro`** — HW 자세에서 gyro |peak|가 스쿼트
  bottom이 아닌 상체 회전을 반영하는 mismatch가 관찰되었다. 따라서
  메인이 아닌 auxiliary로 보유한다.
  - 근거: `reports/step2/2_bottom_event/bottom_event_candidate_audit.md`
    (HW 셀의 acc/gyro 일치도 0.044–0.225 — 같은 임계값에서 SA/CA
    셀 대비 구조적 붕괴).
  - 종합 참조: `reports/step2/step25_final_synthesis.md` §4.

- **`lateral_proxy_gyro`** — IMU는 무릎 각도를 측정하지 않는다.
  이 피처는 약한 회전 단서(weak rotational cue)이며 knee valgus
  measurement가 **아니다**. 따라서 measurement가 아닌 heuristic
  버킷에 둔다.
  - 근거: `reports/step2/3_feature_bank/candidate_feature_bank_audit.md`
    ("lateral_proxy_gyro는 약한 프록시일 뿐"이라는 명시).
  - Step 3 §5, §11의 휴리스틱 처리 규정과 일관.
  - 종합 참조: `reports/step2/step25_final_synthesis.md` §5.

### 6.3 Uncertainty-only

- **`anchor_reliability`** — 모션 피처가 아니라 앵커 자체에 대한
  신뢰/불확실성 신호이다. 따라서 모델 입력 피처가 아니라 §8 신뢰도
  강등 및 §9 no-call 라우팅의 입력으로만 사용된다.
  - 근거: `reports/step2/3_feature_bank/candidate_feature_bank_audit.md`
    (anchor_reliability를 모션 피처와 분리하여 정의).
  - 근거: `reports/step2/2_bottom_event/bottom_event_candidate_audit.md`
    (자세 조건부 앵커 정체성과 함께 신뢰도 측정의 근거).
  - 종합 참조: `reports/step2/step25_final_synthesis.md` §4, §5.
  - Step 3 §6, §7의 anchor 정책과 일관.

### 6.4 Hold-out

- **`bottom_recovery_slope_acc_z`** — 아이디어 자체는 §3의 시간
  국소화 결과(η²(acc_z) 정점이 bottom 이후에 위치)에서 동기 부여를
  받는다. 그러나 현재 6-샘플 window가 좁고 split-unstable로 플래그됨.
  따라서 Step 4 메인에서 hold-out하고 윈도우 재정의 후 재평가.
  - 근거: `reports/step2/1_time_localized/time_localized_class_difference_summary.md`
    (post-bottom 정점 패턴 — 아이디어의 동기).
  - 근거: `reports/step2/3_feature_bank/candidate_feature_bank_audit.md`
    (현재 윈도우 정의의 split-instability 플래그).
  - 종합 참조: `reports/step2/step25_final_synthesis.md` §3, §5.

---

## 7. Primary Baseline Model

Step 4가 가장 먼저 학습 및 평가하는 모델은 다음으로 잠근다:

| 항목 | 값 |
|---|---|
| 모델 | multinomial logistic regression (one-vs-rest 아님, 다항) |
| 정규화 / penalty | L2 |
| 클래스 가중 | `class_weight = balanced` |
| 입력 | §2의 메인 피처 5개 + `posture` one-hot |
| 출력 | `class_posterior` (C1..C6에 대한 적절한 분포) |
| 분할 | manifest의 train / val / test (`split_version = v1_36_8_8`) |

이 baseline은 Step 2.5 §7의 reference separability 모델과 같은
계열이며, 결과 비교가 직접적으로 가능하다는 이점이 있다. 그러나
역할은 다르다: §7의 모델은 *증거*였고, 이 baseline은 Step 3 스키마에
대한 *최저 기준선*이다.

Baseline에서 다음은 사용되지 않는다:

- `anchor_reliability` (§4)
- `lateral_proxy_gyro` (auxiliary; §3)
- `bottom_recovery_slope_acc_z` (hold-out; §5)
- `participant_zscore` 정규화 (Step 3 §11에 의해 제외)

Step 3 §10의 두 정규화 후보 (`raw + posture_as_input`,
`posture_train_zscore + posture_as_input`)는 baseline 비교 시 동시에
시도된다. Step 4가 둘 중 하나로 commit하기 전까지는 두 결과를 모두
보고한다.

---

## 8. 모델 선택 기준 (Selection Criterion)

Step 4의 모델 선택은 다음 기준의 **공동 충족**으로 평가된다. 단일
지표(특히 단순 accuracy) 기준의 commit은 금지된다.

1. **Calibrated class posterior 안정성.**
   - 모델이 잘 calibrated된 `class_posterior`를 낼 수 있는가
     (예: reliability diagram의 ECE, log-loss).
   - Step 3 §3의 "proper distribution" 요건과 §4의 class-set
     branching 정책은 모두 calibrated posterior를 전제로 한다.
     Calibration이 무너진 모델은 accuracy가 높아도 채택 불가.

2. **Class-set prediction threshold calibration 적합성.**
   - Step 3 §4의 분기 규칙과 §8의 신뢰도 수준 표는 모두 *임계값*을
     필요로 한다. Step 4는 모델의 출력 위에서 그 임계값을 calibrate해야
     하며, 그 calibration이 안정적으로 가능한 모델인지를 평가한다.
   - 임계값 자체는 Step 4 calibration의 산출물이며 이 메모가 정하지
     않는다. 이 메모는 모델이 그 calibration을 수용할 수 있는지만을
     기준으로 한다.

3. **Ambiguity group 표현력.**
   - Step 3 §4가 요구하는 세 가지 클래스 집합 — `[C2]`,
     `[C1, C5, C6]` (혹은 부분집합), `[C3, C4, C2]` (필수 헤지),
     `[C3, C4]` — 을 모두 의미 있게 채울 수 있는가.
   - 특히 §7의 reference separability 결과(C2 recall 우위, C1/C5/C6
     그룹 내 혼동, C3/C4의 C2-흡수 패턴)가 모델 출력에 보존되는지를
     별도로 보고한다.
   - 근거: `reports/step2/4_separability/reference_separability_audit.md`
     및 `reports/step2/step25_final_synthesis.md` §7.

4. **단순 accuracy 단독 기준 금지.**
   - Step 2.5 §7에서 세 정규화 설정이 거의 동일한 macro/weighted F1을
     주는 것이 관찰되었다. 따라서 정규화 선택을 accuracy로 결정하는
     것은 신호가 아니다.
   - Step 4는 (1)–(3)을 모두 통과하는 모델 중에서, 보조적으로
     accuracy / macro F1을 **참조**할 수 있다. 단, 이를 단독 결정
     기준으로 사용해서는 안 된다.

5. **Step 3 정책과의 호환성.**
   - 모델 출력이 Step 3 §3의 emission 스키마, §10의 정규화 후보,
     §11의 명시적 제외(특히 단일 argmax 출력 금지)와 모순되지 않아야
     한다. Step 3 정책을 위반하는 모델은 accuracy 우위에 무관하게
     채택 불가.

---

*이 메모는 Step 4 결정 메모로 작성됨. 새로운 피처는 계산되지 않았고,
모델은 학습되지 않았으며, 캡션 템플릿은 작성되지 않았다. Step 1 / Step 2 /
Step 2.5 산출물은 수정되지 않았으며, Step 3 문서는 별도 작업으로
anchor reliability / no-call 정책 충돌 수정만 적용된 상태이다.*
