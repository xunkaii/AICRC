# Step 2.5-4 — 참조 분리도 감사

**범위.** 이는 *참조 분리도 감사*이며, 최종 모델이 아니다. 목표는
세 가지 정규화 체제 하에서 작은 후보 피처 집합이 C1..C6 구조를
얼마나 포착하는지 묻는 것이다. 이 목록에 있는 어떤 것도 채택되거나,
거부되거나, 성능을 위해 튜닝되지 않는다.

## 1. 비교된 설정

| setting | 연속 피처 | 정규화 | 자세 입력 |
|---|---|---|---|
| `raw_core_features` | motion_range_acc_z, depth_proxy, bottom_recovery_slope_acc_z | 없음 | one-hot SA/CA/HW |
| `posture_train_zscore_core_features` | 동일한 3개 피처 | 자세별 z-score (train 적합) | one-hot SA/CA/HW |
| `posture_train_robust_core_features` | 동일한 3개 피처 | 자세별 median/IQR (train 적합) | one-hot SA/CA/HW |

**participant z-score를 제외한 이유.** 신규 사용자로부터의 사전 reps
(캘리브레이션)가 필요하며 첫 rep에서는 실행할 수 없으므로, 배포 가능한
메인 파이프라인 후보가 아니다. Step 2.5-3b는 이를 상한 참조로
유지했지만; 본 감사에서는 학습하지 않는다.

**posture-train 정규화를 유지한 이유.** 둘 다 **train 행에만** 통계를
적합시키고 train 통계를 val/test에 적용하므로, 사용자의 자세 라벨이
주어지면 신규 사용자의 첫 rep에 일반화된다.

**모델 입력으로서의 자세.** 세 설정 모두 자세를 3차원 one-hot 입력으로
포함한다. 정규화와 자세-입력은 상호 배타적이지 않으며, 둘 다 신호를
담을 수 있다.

## 2. 참조 모델

- 다항 로지스틱 회귀, L2 정규화, balanced class weights, random_state=42.
- StandardScaler는 **train** 피처에만 적합시킨 후 모든 분할에 적용.
- 이는 *참조 분리도* 모델이다. 의도적으로 단순하다. 아래 수치는
  최종 모델 성능 추정치로 읽혀서는 안 된다.

**sklearn fallback 메모.** 이 환경의 MINGW-W64 numpy는 `import sklearn`에서
중단된다. 따라서 모델은 명시적인 다항 CE + L2 목적 함수를 사용한
`scipy.optimize.minimize(method='L-BFGS-B')`를 통해 구현되며, 이는
sklearn의
`LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight='balanced')`와
동등하다.

## 3. 헤드라인 메트릭 (val / test)

| setting | split | accuracy | macro F1 | weighted F1 |
|---|---|---|---|---|
| `raw_core_features` | val | 0.2354 | 0.2076 | 0.2079 |
| `raw_core_features` | test | 0.2572 | 0.2274 | 0.2275 |
| `posture_train_zscore_core_features` | val | 0.2416 | 0.2104 | 0.2107 |
| `posture_train_zscore_core_features` | test | 0.2628 | 0.2290 | 0.2291 |
| `posture_train_robust_core_features` | val | 0.2423 | 0.2117 | 0.2121 |
| `posture_train_robust_core_features` | test | 0.2635 | 0.2313 | 0.2314 |

## 4. 모호 그룹 분석 (test split)

정의 (이전 감사에서 재사용):

- `high_confusion_group` = C1, C5, C6 — normal/single/both-knee 패턴을
  공유하며 겹칠 것으로 예상됨.
- `medium_confusion_pair` = C3, C4 — 구별되지만 관련된 결함.
- `clean_reference_class` = C2 — 가장 깨끗하게 식별되어야 함.

| setting | C1/C5/C6 내부 혼동 | C3/C4 페어 혼동 | C2 recall |
|---|---|---|---|
| `raw_core_features` | 0.2727 | 0.1099 | 0.7322 |
| `posture_train_zscore_core_features` | 0.2727 | 0.1078 | 0.7657 |
| `posture_train_robust_core_features` | 0.2685 | 0.1057 | 0.7531 |

읽기 가이드:

- `C1/C5/C6 내부 혼동`은 C1∪C5∪C6 reps 중 참조 모델이 *같은 그룹 내의
  다른 클래스로* 할당한 비율이다. 높은 값은 피처가 이 세 클래스를
  분리할 수 없음을 의미하며 — 이는 정확히 예상된 난이도이다.
- `C3/C4 페어 혼동`은 C3∪C4 reps 중 페어 내의 *다른* 클래스로 잘못
  라우팅된 비율이다.
- `C2 recall`은 C2 reps 중 올바르게 식별된 비율이다.

## 5. 클래스별 분석

전체 표: `reports/reference_separability_metrics.csv`. 자세별 macro
F1도 해당 파일에 있음 (`scope=posture`).

## 6. 혼동 행렬

전체 표 (long 포맷, 모든 settings × splits): `reports/reference_separability_confusion_by_setting.csv`.

## 7. 정규화가 도움이 되는가?

같은 분할에 대한 설정 간 헤드라인 행을 비교하라. Step 2.5-3b가
이미 보여준 것:

- `depth_proxy` raw는 자세에 의해 지배됨 (자세에서 η²≈0.86 vs
  클래스에서 ≈0.004), 그리고 자세 인식 z-score는 클래스 η²를 ≈0.03으로
  끌어올리면서 자세 η²를 ≈0으로 무너뜨림.

이것이 모델 하에서 더 나은 분리도로 변환되는지가 이 감사가
측정하는 것이다. 여기서 신호는 크기가 아니라 **방향**이다.

## 8. 캡션 불확실성에 대한 함의

`C1/C5/C6 내부 혼동`과 `C3/C4 페어 혼동`이 세 설정 모두에서
상당한 수준으로 남아 있다면, 이는 다운스트림 센서-텍스트 출력이
항상 단일 라벨을 확정하기보다 이러한 클래스들 사이의 불확실성을
표현해야 한다는 *증거*이다. Step 2.5-4는 이를 플래그만 하며; 캡션을
설계하지는 않는다.

## 9. 이 감사가 결정하지 않는 것들

- 어떤 피처가 모델링까지 살아남는지.
- 최종 모델이 로지스틱 회귀, 트리 앙상블, 작은 MLP, 또는 다른 무언가가
  되어야 하는지.
- 어떤 정규화 방식이 공식 선택인지.
- 통과해야 할 수치 성능 기준; 여기 수치는 참조이지 목표가 아니다.
