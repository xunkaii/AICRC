# Step 2.5-3 — 후보 피처 뱅크 감사

**이 스크립트의 범위.** 이는 후보 피처 *감사*이며, 피처 선택 확정이
아니다. 피처는 다운스트림에서 평가될 수 있도록 여기서 계산되고
기술된다. 본 실행에서는 이 목록 중 어떤 것도 채택되거나 거부되지
않는다.

## 1. 자세 인식 후보 앵커 규칙

Step 2.5-2 결과에 의해 동기 부여됨 (HW 자세는 gyro 진폭 정점 규칙
하에서 bottom 일치도가 매우 낮았다).

| posture | anchor source | anchor_type |
|---|---|---|
| SA | `ensemble_bottom_idx` (acc/gyro 후보의 평균) | `ensemble_acc_gyro` |
| CA | `ensemble_bottom_idx` (acc/gyro 후보의 평균) | `ensemble_acc_gyro` |
| HW | `acc_bottom_idx` (acc_z 최솟값만)               | `acc_only_anchor` |

**이 규칙은 잠정적이며**, 최종 모델링 규칙이 아니다. 여기서는 단지
HW의 약한 gyro-정점 신호 하에서 무너지지 않는 방식으로 후보 피처를
계산하기 위해서만 사용된다.

근거 요약:
- SA / CA는 reps의 70–88%에서 셀 수준 bottom_agreement_ratio ≤ 0.10을
  보였으므로 acc/gyro 후보의 ensemble이 안정적이다.
- HW는 동일한 규칙으로 일치도가 4–22%에 불과했다. HW의 경우 손이
  허리에 놓여 있어 |gyro| 정점이 스쿼트 bottom과 분리되므로 acc_z
  최솟값만 사용하는 것이 더 신뢰할 수 있다.

## 2. 후보 피처

| feature | 해석 | 주의사항 |
|---|---|---|
| `depth_proxy` | [anchor±5] 구간 내 평활화된 acc_z의 평균. (중력 기준) 값이 낮을수록 rep의 bottom이 더 깊거나 bottom에서의 방향 기울기가 더 큼을 시사. | 캘리브레이션-프리: 이는 스쿼트 깊이의 mm가 *아님*; 동일 피험자 내 상대 비교에 유용한 방향-프록시일 뿐. |
| `motion_range_acc_z` | rep 전체에 걸친 acc_z의 robust range (p95 − p5). 값이 클수록 rep가 더 넓은 수직 성분 범위를 거친다는 의미. | 센서 마운트 방향이 영향을 미침; 피험자 간 비교는 근사적. |
| `motion_range_gyro_mag` | rep 전체에 걸친 \|gyro\|의 robust range (p95 − p5). 값이 클수록 rep 동안 더 많은 회전 움직임. | 자세가 팔의 스윙을 바꾸므로 — 값들이 SA/CA/HW 간에 직접 비교 가능하지 않음. |
| `bottom_stability_acc` | [anchor±10] 구간 내 (acc_x, acc_y, acc_z) 축별 표준편차의 RMS. 값이 작을수록 bottom에서 더 정지된 유지. | 더 작은 창이나 더 짧은 rep는 이 피처를 편향시킬 수 있음; 창이 잘리면 feature_compute_warning으로 플래그됨. |
| `bottom_stability_gyro` | [anchor±10] 구간 내 \|gyro\|의 표준편차. 값이 작을수록 bottom에서의 회전 지터가 적음. | 작은 지터에 민감; anchor_reliability와 함께 읽어야 함. |
| `bottom_recovery_slope_acc_z` | [anchor+10, anchor+16) 구간에 대한 평활화된 acc_z의 선형 기울기. Post-bottom 회복 동역학을 포착; Step 2.5-2에서 η²(acc_z)가 bottom *이후*에 정점에 이른 것에서 동기 부여. | 부호와 크기는 IMU 마운트 축에 의존; 기울기는 상대적 의미에서만 해석 가능. |
| `bottom_transition_delta_acc_z` | acc_z의 post-anchor 10 샘플 평균 − pre-anchor 10 샘플 평균. Bottom 전반의 부호 있는 전이 크기. | 깊이와 방향 변화를 혼동시킴. 인과적 결론을 내리기 전에 depth_proxy와 짝지어 봐야 함. |
| `lateral_proxy_gyro` | [anchor±10] 구간 내 mean(\|gyro_x\|) + mean(\|gyro_y\|). 측면/valgus 움직임에 대한 **약한 프록시**: IMU는 무릎 각도를 직접 측정하지 않으므로 이는 휴리스틱이며, knee-valgus 측정이 아님. | **약한 프록시**로 표시됨. 높은 값이 knee valgus를 의미하지 않음; 단지 bottom 근처의 측면/yaw 회전을 나타냄. |
| `anchor_reliability` | 앵커 자체에 대한 신뢰도 [0..1]. Ensemble (SA/CA): bottom_agreement_ratio가 0.30까지 증가함에 따라 선형으로 감소. Acc-only (HW): acc 경고가 없으면 1.0, acc_min_outside_search_range가 플래그되었으면 0.5. | 현재는 휴리스틱 점수; 임계값(0.30 상한, 0.5 fallback)은 검증되지 않음. |

**`bottom_recovery_slope_acc_z`에 대해.** Step 2.5-2는 SA와 CA의
η²(acc_z)가 bottom *근처 또는 직후* (timesteps ≈ 128 중 86)에서
정점에 이른다고 보고했다. 이는 단지 bottom 깊이뿐 아니라 *post-bottom
회복 동역학*이 클래스 신호를 담고 있을 수 있다는 힌트이다. 기울기
피처는 이를 다운스트림 평가에 노출시키기 위한 첫 시도이다.

## 3. 계산 성공

- bottom audit에서 누락된 매니페스트 행 (스킵됨): 0
- Bank 행 수 (피처 시도): **9275**
- `feature_compute_ok = True`인 행 수: **9275** (100.00%)

자세별:

| posture | rows | compute_ok | rate |
|---|---|---|---|
| SA | 3098 | 3098 | 1.0000 |
| CA | 3080 | 3080 | 1.0000 |
| HW | 3097 | 3097 | 1.0000 |

클래스별:

| class | rows | compute_ok | rate |
|---|---|---|---|
| C1 | 1553 | 1553 | 1.0000 |
| C2 | 1555 | 1555 | 1.0000 |
| C3 | 1529 | 1529 | 1.0000 |
| C4 | 1540 | 1540 | 1.0000 |
| C5 | 1549 | 1549 | 1.0000 |
| C6 | 1549 | 1549 | 1.0000 |

## 4. 피처별 NaN 카운트

| feature | n_nan | rate |
|---|---|---|
| `depth_proxy` | 0 | 0.0000 |
| `motion_range_acc_z` | 0 | 0.0000 |
| `motion_range_gyro_mag` | 0 | 0.0000 |
| `bottom_stability_acc` | 0 | 0.0000 |
| `bottom_stability_gyro` | 0 | 0.0000 |
| `bottom_recovery_slope_acc_z` | 0 | 0.0000 |
| `bottom_transition_delta_acc_z` | 0 | 0.0000 |
| `lateral_proxy_gyro` | 0 | 0.0000 |
| `anchor_reliability` | 0 | 0.0000 |

## 5. 앵커 타입 분포

| posture | anchor_type | count |
|---|---|---|
| SA | `ensemble_acc_gyro` | 3098 |
| CA | `ensemble_acc_gyro` | 3080 |
| HW | `acc_only_anchor` | 3097 |

## 6. 분포 / 강건성 / 겹침 (교차 참조)

- `reports/candidate_feature_summary_by_class_posture.csv` — (class × posture × feature)별 `n`, `n_nan`, `mean`, `std`, `median`, `p25`, `p75`.
- `reports/candidate_feature_split_robustness.csv` — 피처별 `(train, val, test)` 평균/표준편차 및 `flag_unstable` (max-min 평균 범위가 |overall mean|의 20%를 초과할 때 설정).
- `reports/candidate_feature_uncertainty_overlap.csv` — confusion group / feature / posture-scope별: pairwise Cohen's *d* 및 겹침 추정치 (Gaussian 근사, 확률이 아닌 *신호*로 유용).

조사된 혼동 그룹:

- `C1_C5_C6`: C1, C5, C6
- `C3_C4`: C3, C4

**겹침이 큰 피처를 유지하는 이유.** 클래스 C1/C5/C6 및 C3/C4는
구조상 겹치도록 예상된다 (그들은 자세-과제 구조를 공유하면서 미묘한
차이를 가짐). 피처의 높은 겹침은 *피처를 버려야 할 이유가 아니다*;
이는 잠재적으로 caption/output에서 calibrated uncertainty의 기반이
될 수 있으며, Step 2.5-3는 이를 다운스트림 고려를 위한 후보 사용
사례로 플래그만 한다.

## 7. 미해결 질문 / 이 감사가 명시적으로 결정하지 않는 것들

- 어떤 피처가 모델링까지 살아남는지.
- `lateral_proxy_gyro`(약한 프록시)가 제거되어야 하는지, 또는
  uncertainty-only 피처로 유지되어야 하는지.
- `anchor_reliability` 임계값이 올바르게 보정되었는지.
- HW의 acc-only fallback이 전적으로 다른 앵커링 전략으로 대체되어야
  하는지.
- 다운스트림 사용 전에 피처별 정규화 (피험자별 / 자세별)가 필요한지.

이들은 모두 명시적으로 **다음 단계** 결정이며, Step 2.5-3의 결정이
아니다.
