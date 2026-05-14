# Step 2.5-3b — 정규화 실현 가능성 감사

**범위.** 이 스크립트는 일반적인 정규화 전략이 배포 환경에서 *실현
가능한지*, 그리고 자세 편향을 줄이면서 클래스 신호를 보존하는지를
감사한다. **이는 피처 선택 단계가 아니다.** 여기서는 어떤 피처도
채택되거나 거부되지 않는다.

## 1. 이 감사가 다루는 것 (그리고 의도적으로 다루지 않는 것)

- 테스트된 피처: `motion_range_acc_z`, `depth_proxy`, `bottom_recovery_slope_acc_z` — 3개만.
- 테스트된 방법:
  - `raw` — 원본 피처 값
  - `posture_train_zscore` — **train** 행에만 적합된 자세별 평균/표준편차, 모든 분할에 적용
  - `posture_train_robust` — **train** 행에만 적합된 자세별 중앙값 / (IQR / 1.349), 모두에 적용
  - `participant_zscore_upper_bound` — 해당 피험자의 모든 reps에 적합된 피험자별 평균/표준편차

**3개 피처 제약은 의도적이다.** 이 단계에서 모든 후보 피처로 확장하면
어떤 정규화 원칙이 성립하는지 알기도 전에 피처 정글이 만들어진다.

## 2. participant z-score가 *상한선*인 이유

`participant_zscore_upper_bound`는 피험자 본인의 reps에서 계산된
통계를 필요로 한다. 신규 사용자의 **첫 rep**에는 그러한 통계가
존재하지 않는다; 캘리브레이션 데이터를 먼저 수집해야 한다.

- 따라서 메인 추론 파이프라인의 후보가 **아니다**.
- 여기서는 피험자별 캘리브레이션이 가능했다면 정규화가 얼마나 도움이
  될 수 있는지에 대한 *상한*으로 포함된다.
- 메인 경로를 participant z-score로 교체하면 신규 사용자 추론이
  조용히 깨질 것이다; 본 보고서는 이후 독자가 이를 기본값으로 다시
  도입하지 않도록 이 점을 명시적으로 밝힌다.

`posture_train_zscore`와 `posture_train_robust`는 **train** 통계만
사용하며, 사용자의 자세 라벨이 주어지면 신규 사용자의 첫 rep에도
적용할 수 있으므로, 여기서 현실적인 메인 파이프라인 후보이다.

## 3. 계산 성공

- 행 수: **9275**
- `normalization_compute_ok = True`: **9275** (100.00%)

통계 적합 (투명성을 위해):

| stat           | scope | source split | 신규 사용자 첫 rep에 사용 가능?  |
| -------------- | ----- | ------------ | --------------------- |
| 자세별 평균 / 표준편차  | 자세별   | train만       | yes                   |
| 자세별 중앙값 / IQR  | 자세별   | train만       | yes                   |
| 피험자별 평균 / 표준편차 | 피험자별  | 전체 분할        | **no** (사전 캘리브레이션 필요) |

## 4. 클래스 효과 vs 자세 효과 (η²)

목표: 정규화 후 **클래스 η²는 raw와 ≈로 유지**되고, **자세 η²는
감소**해야 한다. 자세 인식 정규화는 자세 η²를 0에 가깝게 만들 것으로
예상된다.

| feature                       | method                           | n    | class η² | posture η² | class η² / raw | posture η² / raw |
| ----------------------------- | -------------------------------- | ---- | -------- | ---------- | -------------- | ---------------- |
| `motion_range_acc_z`          | `raw`                            | 9275 | 0.1112   | 0.0607     | 1.000          | 1.000            |
| `motion_range_acc_z`          | `posture_train_zscore`           | 9275 | 0.1171   | 0.0003     | 1.053          | 0.005            |
| `motion_range_acc_z`          | `posture_train_robust`           | 9275 | 0.1165   | 0.0013     | 1.047          | 0.022            |
| `motion_range_acc_z`          | `participant_zscore_upper_bound` | 9275 | 0.1574   | 0.0721     | 1.415          | 1.188            |
| `depth_proxy`                 | `raw`                            | 9275 | 0.0042   | 0.8623     | 1.000          | 1.000            |
| `depth_proxy`                 | `posture_train_zscore`           | 9275 | 0.0307   | 0.0019     | 7.270          | 0.002            |
| `depth_proxy`                 | `posture_train_robust`           | 9275 | 0.0303   | 0.0040     | 7.156          | 0.005            |
| `depth_proxy`                 | `participant_zscore_upper_bound` | 9275 | 0.0052   | 0.9130     | 1.239          | 1.059            |
| `bottom_recovery_slope_acc_z` | `raw`                            | 9275 | 0.0056   | 0.0924     | 1.000          | 1.000            |
| `bottom_recovery_slope_acc_z` | `posture_train_zscore`           | 9275 | 0.0059   | 0.0002     | 1.058          | 0.002            |
| `bottom_recovery_slope_acc_z` | `posture_train_robust`           | 9275 | 0.0058   | 0.0065     | 1.045          | 0.071            |
| `bottom_recovery_slope_acc_z` | `participant_zscore_upper_bound` | 9275 | 0.0058   | 0.0997     | 1.038          | 1.079            |

읽기 가이드:

- `class η² / raw ≈ 1.0` → 클래스 신호 보존됨.
- `posture η² / raw < 0.2` → 자세 편향이 대체로 제거됨.
- `class η² / raw < 0.5` → 정규화가 너무 많이 깎아냄; 위험 신호.

## 5. 분할 강건성

- `raw`의 경우: 분할 평균의 (max − min) > |overall mean|의 20%이면 플래그.
- 정규화 방법의 경우: 분할 평균의 절대 (max − min) > 0.20 (z-score 단위)이면 플래그.

| feature                       | method                           | rule                      | unstable? |
| ----------------------------- | -------------------------------- | ------------------------- | --------- |
| `motion_range_acc_z`          | `raw`                            | raw_relative_>0.20        | False     |
| `motion_range_acc_z`          | `posture_train_zscore`           | normalized_absolute_>0.20 | False     |
| `motion_range_acc_z`          | `posture_train_robust`           | normalized_absolute_>0.20 | False     |
| `motion_range_acc_z`          | `participant_zscore_upper_bound` | normalized_absolute_>0.20 | False     |
| `depth_proxy`                 | `raw`                            | raw_relative_>0.20        | False     |
| `depth_proxy`                 | `posture_train_zscore`           | normalized_absolute_>0.20 | **True**  |
| `depth_proxy`                 | `posture_train_robust`           | normalized_absolute_>0.20 | **True**  |
| `depth_proxy`                 | `participant_zscore_upper_bound` | normalized_absolute_>0.20 | False     |
| `bottom_recovery_slope_acc_z` | `raw`                            | raw_relative_>0.20        | **True**  |
| `bottom_recovery_slope_acc_z` | `posture_train_zscore`           | normalized_absolute_>0.20 | False     |
| `bottom_recovery_slope_acc_z` | `posture_train_robust`           | normalized_absolute_>0.20 | **True**  |
| `bottom_recovery_slope_acc_z` | `participant_zscore_upper_bound` | normalized_absolute_>0.20 | False     |

전체 표: `reports/normalization_feasibility_split_robustness.csv`

## 6. Confusion-class 겹침 (불확실성 신호)

- 조사된 그룹:
  - `C1_C5_C6`: C1, C5, C6
  - `C3_C4`: C3, C4
- (feature, method, posture-scope) 및 쌍별로 Cohen's *d*와 Gaussian
  근사 겹침 추정치를 보고한다.
- **높은 겹침은 피처를 버릴 이유가 아니다.** 혼동 그룹(C1/C5/C6,
  C3/C4)에서 클래스들은 미묘하게 다르도록 구성되어 있으므로 상당한
  겹침이 예상되며, 이는 잠재적으로 다운스트림 출력에서의 *calibrated
  uncertainty 기반*이 될 수 있다 (후보 사용일 뿐, Step 2.5-3b
  결정이 아님).

전체 표: `reports/normalization_feasibility_overlap.csv`

## 7. 정규화 도구가 아닌 *모델 입력*으로서의 자세

자세 인식 정규화 후에도 자세 자체는 rep에 대한 정보(피험자가 사용하도록
지시받은 팔 자세가 무엇이었는지)를 담는다. 향후 모델링은 자세를
(raw 또는 정규화된) 피처와 함께 범주형 입력으로 전달할 수 있다 —
이 옵션은 여기서 명시적으로 열어 둔다. 정규화와 자세-입력은
상호 배타적이지 않다.

## 8. 정규화는 무엇을 위한 것인가 — 그리고 무엇을 위한 것이 아닌가

- 모델의 대체물이 **아니다**.
- 다음을 위한 것이다: (a) 자세 전반에서 캡션 수준 해석을 합리적으로
  유지하기, 그리고 (b) 다운스트림 캘리브레이션 / 임계값 결정이
  배치의 자세 분포에 의존하지 않을 만큼 피처 스케일을 안정적으로
  유지하기.
- 그 이상의 것은 모델링 결정이지, 정규화 결정이 아니다.

## 9. 다음 단계 옵션 (열려 있음, 여기서 어떤 것도 선택되지 않음)

1. `raw` 피처 + 모델 입력으로서의 자세.
2. `posture_train_zscore` 피처 + 모델 입력으로서의 자세.
3. `posture_train_robust` 피처 + 모델 입력으로서의 자세.
4. `participant_zscore_upper_bound`는 캘리브레이션 상한 참조로 **만** 유지; 배포 가능한 기본값이 아님.

각 옵션은 (a) 클래스 η² 보존, (b) 분할 강건성, (c) 신규 사용자
배포 실현 가능성, 그리고 (d) 캡션 해석 가능성 — 이 감사가 설정한
네 가지 렌즈에 대해 평가되어야 한다.

## 10. §5에서 unstable로 플래그된 (feature, method) 쌍

- `depth_proxy` × `posture_train_zscore`
- `depth_proxy` × `posture_train_robust`
- `bottom_recovery_slope_acc_z` × `raw`
- `bottom_recovery_slope_acc_z` × `posture_train_robust`
