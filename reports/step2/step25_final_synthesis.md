# Step 2.5 — 최종 종합 (v2)

**이 문서는 Step 2.5에서 확인된 사항과 명시적으로 결정하지 않은 사항을
요약한다.** 이는 Step 3(출력 스키마 및 불확실성 정책 설계)으로의
인계 문서이다. 피처 목록, 앵커 규칙, 정규화 선택, 모델 아키텍처,
캡션 템플릿, 성능 기준이 아니다.

---

## 1. Step 2.5가 존재한 이유

Step 2.5는 AICRC v1의 결과를 *그대로* 정답으로 받아들이지 않았다.
v2 매니페스트 위에서, 스쿼트에 대한 센서-텍스트 출력을 설계하기 위해
필요한 증거를 다시 도출했다:

- 클래스 신호가 원시 신호에 실제로 존재하는지,
- 클래스 신호가 *시간상 어느 위치에* 존재하는지,
- 단일 "bottom" 이벤트가 안정적인 앵커 후보가 될 수 있는지,
- 어떤 단순하고 해석 가능한 피처를 신뢰성 있게 계산할 수 있는지,
- 실현 가능한 정규화가 자세 편향을 약화시키면서 클래스 신호를
  보존하는지,
- 그리고 작은 참조 모델 하에서 C1–C6가 얼마나 분리 가능한지.

산출물은 모델이 아니라 *증거*이다. 전체적으로 "분류 정확도"가 아닌
"센서-텍스트 설명"이라는 관점을 유지했다.

---

## 2. 데이터 및 매니페스트 상태

`reports/manifest_summary.md` 및 `reports/step25_preflight_summary.md`로부터:

| 필드 | 값 |
|---|---|
| `manifest_raw` 행 수 | 9275 |
| `manifest_clean` 행 수 | 9275 (hard drop = 0) |
| `manifest_split` 행 수 | 9275 |
| `boundary_ok = False` (soft flag) | 9 |
| `length_ok = False` (soft flag) | 0 |
| 참가자 | 52명 (`EP01`..`EP52`) |
| 분할 (참가자) | train 36 / val 8 / test 8 |
| 분할 (샘플) | train 6412 / val 1436 / test 1427 |
| `split_seed` | 42 |
| `split_version` | `v1_36_8_8` |
| 분할 간 참가자 누수 | **0** (검증됨) |

`boundary_ok = False`인 9개 행은 모두 `_vkeep_st_end.txt` 파일이
없는 단일 세그먼트(`EP10 / C1 / SA`)에서 발생한다. 해당 reps에는
사용 가능한 `_time.txt`가 있어 시간 기반 길이 검사
(`length_ok_time = True`)를 통과하므로 clean에 유지된다.

자세별 C1–C6 샘플 수는 본질적으로 균형을 이룬다 (`class × posture`
셀당 ≈ 502–519 reps); 18개 셀 × 약 515 reps.

**Step 3에 대한 의미.** 클래스 및 분할 관리는 다운스트림 출력
설계가 누수나 분할 표류에 대해 방어할 필요가 없을 만큼 깨끗하다.
Step 3는 매니페스트를 고정된 참조로 가정할 수 있다.

---

## 3. 시간 국소화된 클래스 차이

`reports/time_localized_class_difference_summary.md`로부터.

각 rep는 6개 채널(`acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z`)에
대해 길이 128로 선형 리샘플링되었다. 각
`(posture, channel, timestep)`에 대해 클래스 C1–C6 전반의 η²가
계산되었다.

이 감사는 "클래스들이 겹치는가?"를 묻는 것이 *아니라* "클래스 신호가
존재한다면, 시간상 어디에 존재하는가?"를 물었다.

발견 사항:

- **18개 중 11개** `(posture × channel)` 셀에서 bottom 영역
  (128 중 timesteps 38–89)의 평균 η²가 전체 시간선 전반보다 높았다.
- SA / CA의 경우, η²(`acc_z`)가 bottom 위치에서가 아니라 *bottom
  근처 또는 직후*에 정점에 이르는 경향이 있었다 (argmax timesteps
  ≈ 128 중 86).
- 일부 채널(예: HW의 `gyro_z`)은 bottom 영역 집중을 보이지 않았다.

**Step 3에 대한 의미.** Bottom/이벤트 앵커링 피처는 정당화 가능한
후보이다. 많은 `(posture, channel)` 셀에서 신호가 실제로 bottom
근처에 집중되어 있기 때문이다. **하지만 이것이 bottom 앵커를 최종
구조로 확정하지는 않는다** — 집중은 부분적이며, *post-bottom*에서
peaking이 일어난다는 것은 회복 동역학이 bottom 자체만큼 중요할 수
있음을 시사한다.

---

## 4. Bottom 이벤트 후보 감사

`reports/bottom_event_candidate_audit.md`로부터.

Rep별로, `acc_bottom_idx`(acc_z 최솟값) 및 `gyro_bottom_idx`
(|gyro| 정점)가 rep의 중간 25–75% 구간에서 검출되었다. 일치도는
`bottom_agreement_ratio = |acc - gyro| / n_rows`로 측정되었다.

| posture | threshold = 0.10에서의 agree_rate |
|---|---|
| SA 셀 (6) | 0.736 – 0.884 |
| CA 셀 (6) | 0.735 – 0.843 |
| HW 셀 (6) | **0.044 – 0.225** |

- 단순한 전역 규칙(항상 acc/gyro의 앙상블 사용)은 **정당화될 수
  없다**: HW 자세의 bottom_agreement는 동일한 임계값에서 6개 클래스
  전반에 걸쳐 무너진다.
- 이 패턴은 노이즈가 아니라 구조적이다: HW에서는 손이 허리에 고정되어
  있어 |gyro| 정점은 *상체 회전*을 반영하며, acc_z 최솟값이 추적하는
  *스쿼트 bottom*이 아니다. 두 방법은 서로 다른 물리적 이벤트를
  측정하고 있다.

감사의 자동 산출 예비 판정은 12 / 18 셀이 70% 일치도를 초과하므로
"GO"였다. **본 종합 문서는 그 판정보다 더 신중하다**: 18개 셀 중
6개에서 agree_rate가 0.25 미만이며, 그것들은 정확히 HW 셀이다.
"다수결 GO" 해석은 자세 형태의 실패 모드를 가릴 수 있다.

피처 뱅크에 잠정적으로(최종이 아님) 도입된 앵커 구조:

| posture | 후보 앵커 | 근거 |
|---|---|---|
| SA | `ensemble_bottom_idx` (acc/gyro의 평균) | 높은 acc/gyro 일치도 |
| CA | `ensemble_bottom_idx` | 높은 acc/gyro 일치도 |
| HW | `acc_bottom_idx` (acc_z 최솟값만) | gyro 정점은 다른 이벤트를 측정 |

**Step 3에 대한 의미.** 단일 전역 bottom 앵커는 고려 대상이 아니다.
출력 설계는 앵커 정체성을 *자세 조건부*로 다루어야 하며, 앵커
유래 피처 옆에 앵커 신뢰도 신호를 기록해야 한다. Step 2.5는 이
규칙을 최종 모델링 규칙으로 확정하지 않는다.

---

## 5. 후보 피처 뱅크

`reports/candidate_feature_bank_audit.md`로부터.

잠정적인 자세 인식 앵커를 사용하여 9개 후보 피처가 계산되었다.
피처 계산 성공: **9275 / 9275** (어떤 피처에서도 NaN 없음).

피처별 메모(채택 / 거부 없음 — 관찰 사항만):

- **`motion_range_acc_z`** (rep 전반의 acc_z의 p95 − p5) — 관찰된
  후보 중 가장 안정적. 앵커 없이 계산되므로 HW 앵커 약점의 영향을
  받지 않는다.
- **`depth_proxy`** ([anchor ± 5] 구간의 평활화된 acc_z 평균) —
  사용 가능하지만, 원시 값이 자세에 의해 지배된다 (§6 참조). 정규화
  없이는 자세 간 해석이 불가능하다.
- **`bottom_recovery_slope_acc_z`** ([anchor + 10, anchor + 16) 구간의
  acc_z 선형 기울기) — η²(acc_z)가 bottom *이후*에 정점에 이른다는
  §3의 관찰에서 동기 부여. 아이디어는 유지되지만, 현재 6-샘플
  창은 좁고 피처가 split-unstable로 플래그되었다; 창 정의를
  재검토해야 한다.
- **`bottom_transition_delta_acc_z`** (acc_z의 post-anchor 평균
  − pre-anchor 평균) — 깊이와 자세 변화를 혼동시킨다; `depth_proxy`와
  대조하여 읽어야 한다.
- **`motion_range_gyro_mag`** / **`bottom_stability_acc`** /
  **`bottom_stability_gyro`** — 자세 간 스케일이 다르다; 자세 간
  비교는 자세 인식 처리 후에만 가능하다.
- **`lateral_proxy_gyro`** (anchor 근처의 mean(|gyro_x|) +
  mean(|gyro_y|)) — **약한 프록시일 뿐**. IMU는 무릎 각도를 측정하지
  않는다. 다운스트림 캡션에 "knee valgus 측정"으로 제시되어서는
  안 된다.
- **`anchor_reliability`** — *모션 피처가 아니다*. 이는 앵커 자체에
  대한 신뢰/불확실성 신호이다. Step 3는 이를 피처 트랙이 아니라
  *불확실성* 트랙에 실어야 한다.

**Step 3에 대한 의미.** 피처 뱅크에는 적어도 안정적인 후보 하나
(`motion_range_acc_z`), 자세 인식 처리가 필요한 피처 하나
(`depth_proxy`), 유지할 가치는 있지만 메커니즘적으로 아직 정확하지
않은 아이디어 하나(`bottom_recovery_slope_acc_z`), 그리고 휴리스틱으로
정직하게 제시되어야 하는 피처 하나(`lateral_proxy_gyro`)가 있다.
앵커 신뢰도는 피처 스키마가 아닌 불확실성 스키마에 속한다.

---

## 6. 정규화 실현 가능성

`reports/normalization_feasibility_audit.md`로부터.

세 가지 피처(motion_range_acc_z, depth_proxy,
bottom_recovery_slope_acc_z)가 네 가지 방법 하에서 테스트되었다.
9275개 행 전반의 클래스 η²와 자세 η²:

| feature | method | class η² | posture η² | 메모 |
|---|---|---|---|---|
| motion_range_acc_z | raw | 0.111 | 0.061 | 이미 자세 혼동이 낮음 |
|  | posture_train_zscore | 0.117 | **0.000** | 자세 혼동이 본질적으로 제거됨 |
|  | posture_train_robust | 0.116 | 0.001 | 유사 |
|  | participant_zscore (UB) | 0.157 | 0.072 | 자세를 줄이지 못함 |
| depth_proxy | raw | 0.004 | **0.862** | 자세에 의해 지배됨 |
|  | posture_train_zscore | **0.031** | 0.002 | 정규화 후 클래스 신호 **7×** 강해짐 |
|  | posture_train_robust | 0.030 | 0.004 | 유사 |
|  | participant_zscore (UB) | 0.005 | 0.913 | 해결되지 **않음** |
| bottom_recovery_slope_acc_z | raw | 0.006 | 0.092 | 양방향으로 약한 신호 |
|  | posture_train_zscore | 0.006 | 0.000 | 보존됨 + 자세 제거됨 |

발견 사항:

- **`participant_zscore`는 메인 파이프라인 후보로 실현 가능하지 않다.**
  - 사용자의 이전 reps가 필요하다; 신규 사용자의 첫 rep는 적용할
    통계가 없다.
  - 적용된 경우에도(참가자의 모든 reps를 calibration 상한으로 사용),
    자세 η²를 줄이지 **못한다**. `depth_proxy`에서는 실제로 *증가*
    시킨다 (0.862 → 0.913). 단일 피험자 내에서도 자세는 여전히
    방향을 결정하므로, 피험자별 센터링은 자세 문제를 해결하지 못한다.
  - Step 2.5-3b는 calibration-ceiling 참조로만 유지했다; 본 종합
    문서는 이를 명시적으로 **이어가지 않는다**.
- **`posture_train_zscore`와 `posture_train_robust`는 배포 가능하다.**
  둘 다 train 행에만 통계를 적합시키고 train 통계를 val/test에
  적용한다. 사용자의 자세 라벨이 주어지면 신규 사용자의 첫 rep에서도
  사용 가능하다.
- 둘 중 더 단순한 것은 `posture_train_zscore`이다.
- 정규화는 **모델의 대체물이 아니다**. 여기서 그 가치는 (a) 자세
  간 캡션 해석을 정직하게 유지하는 것(`depth_proxy`의 7× 클래스
  η² 상승이 가장 명확한 예), 그리고 (b) 다운스트림 calibration을
  위한 피처 스케일 안정성이다.

**Step 3에 대한 의미.** Step 3는 출력 스키마를 `raw + posture-as-input`
또는 `posture_train_zscore + posture-as-input` 중 하나에 대해 설계해야
한다. `participant_zscore`는 Step 3의 후보 집합에 **포함되지 않는다**.

---

## 7. 참조 분리도

`reports/reference_separability_audit.md` 및
`reports/reference_separability_metrics.csv`로부터.

L2 + balanced class weights를 사용한 작은 다항 로지스틱 회귀가
세 가지 설정(raw / posture_train_zscore / posture_train_robust core
features + posture one-hot)에서 적합되었다. 이 모델은 *참조*이지,
최종 모델이 아니다.

테스트 분할 메트릭:

| 설정 | accuracy | macro F1 | weighted F1 |
|---|---|---|---|
| `raw_core_features` | 0.2572 | 0.2274 | 0.2275 |
| `posture_train_zscore_core_features` | 0.2628 | 0.2290 | 0.2291 |
| `posture_train_robust_core_features` | 0.2635 | 0.2313 | 0.2314 |

(우연 = 1 / 6 ≈ 0.167.)

테스트 분할 모호 그룹 메트릭
(`reports/reference_separability_ambiguous_groups.csv`):

| 설정 | C1/C5/C6 내부 혼동 | C3/C4 페어 혼동 | C2 recall |
|---|---|---|---|
| raw | 0.2727 | 0.1099 | **0.7322** |
| posture_train_zscore | 0.2727 | 0.1078 | **0.7657** |
| posture_train_robust | 0.2685 | 0.1057 | **0.7531** |

혼동 행렬 관찰 (test, posture_train_zscore 설정):

- **C2가 가장 깨끗한 클래스이다.** 설정 전반에 걸쳐 recall 0.66–0.77,
  일관되게 가장 높다.
- **C1 / C5 / C6는 모호한 채로 남는다.** 이 그룹의 약 27%의 reps가
  같은 그룹 내의 *다른* 클래스로 라우팅된다.
- **C3와 C4는 단지 서로와만 혼동되는 것이 아니다.** test 분할 내에서
  (C3는 n = 238, C4는 n = 235), 지배적인 오라우팅은 C2 방향이다
  (C3 → C2가 118 / 238 = 50%; C4 → C2가 97 / 235 = 41%). 따라서
  "C3 / C4 페어 혼동" 프레임은 *불완전하다*; 더 큰 이야기는 C3 /
  C4의 C2로의 흡수이다.
- **Top1–top2 마진은 예측 CSV에서 대부분 작다.** Top1 확률에 대한
  단순한 신뢰도 임계값 규칙은 이 피처들만으로 "확신함" vs "불확신함"을
  안정적으로 표현할 수 없다.
- **세 가지 정규화 설정은 거의 동일한 메트릭을 제공한다.** 자세는
  이미 one-hot 입력으로 주입되어 있으며, LR은 raw 피처에서도
  자세 의존적 절편을 흡수할 수 있다. 정규화에 따른 η² 이득(§6)은
  해석 가능성/스케일 이득이지, 헤드라인 분류 이득이 아니다.

**Step 3에 대한 의미.** 6-클래스 단일 라벨 출력은 이 증거 위에서
정당화될 수 없다. 참조 모델은 다음과 같이 말한다: 클래스 신호는
*존재한다*(우연 이상), 그러나 불확실성을 표현하지 않고 단일 라벨을
확정할 만큼 강하지는 않다. C2는 자신감 있는 출력으로 합리적으로
정당화 가능한 유일한 클래스이다. C1 / C5 / C6는 그룹으로 표현
가능해야 하며, C3 / C4는 "C2일 수도 있음" 헤지와 함께 표현 가능해야
한다.

---

## 8. Step 2.5가 명시적으로 결정하지 않는 사항

- 최종 피처 집합.
- 최종 정규화.
- 최종 앵커 규칙 (SA/CA→ensemble, HW→acc_only 구조는 감사 내부에서
  사용된 *후보*이지, 모델 규칙이 아니다).
- 최종 모델 아키텍처.
- 캡션 템플릿 표현.
- 어떤 성능 기준이나 성공 기준.
- 클래스를 추가하거나 삭제할지 여부 (C1–C6는 Step 1에서 잠겨 있음).

하나의 옵션을 다른 옵션보다 선택해야 하는 모든 사항은 Step 3+로
보류된다.

---

## 9. Step 3를 위한 사실 기반 설계 요구사항

이는 *Step 2.5 증거에서 도출된 요구사항*이지, 모델링 결정이 아니다.
이는 Step 3의 설계 공간을 제약한다.

1. **출력은 단일 클래스 라벨이 될 수 없다.** 참조 모델의 모호도
   프로파일(§7)은 단일 argmax를 부적절하게 만든다.
2. **출력 스키마는 최소한 다음을 담아야 한다:**
   - 클래스 확률 (또는 클래스 집합 사후),
   - 피처 증거 (어떤 피처가 설명을 이끌었는지),
   - 앵커 신뢰도 (피처와는 별도로),
   - 분류 체계가 임시방편적이지 **않은** 모호도 플래그.
3. **클래스 조건부 처리:**
   - **C2**는 비교적 자신감 있는 캡션 후보가 될 수 있다.
   - **C1 / C5 / C6** 캡션은 그룹 내 모호도를 인정하도록 작성되어야
     한다. 이 셋 중 하나로 단정 짓는 캡션은 증거에 의해 뒷받침되지
     않는다.
   - **C3 / C4** 캡션은 하나가 아닌 두 가지 모호도를 인정해야 한다:
     페어 내 모호도(C3 ↔ C4)와 C2로의 흡수(불충분한 깊이 프레임).
     모호도 플래그 분류 체계는 단지 "C3 또는 C4"가 아니라 "C3로
     보이지만 C4 또는 C2일 수도 있음"을 말할 수 있을 만큼 표현력이
     있어야 한다.
4. **모든 곳에서 자세 인식.** 앵커 정체성, 피처 해석, 정규화 모두
   자세에 의존한다. Step 3의 스키마는 자세를 선택적 공변량이 아닌
   *필수 입력*으로 다루어야 한다. Rep 수준 규칙 "SA/CA는 ensemble
   앵커, HW는 acc-only 앵커"는 앵커 하위 스키마의 시작점이어야 한다.
5. **정규화 범위.** Step 3 후보는 `raw + posture-as-input` 및
   `posture_train_zscore + posture-as-input`에 한정된다.
   `participant_zscore`는 실현 가능성에 의해 제외된다 (§6); 이를
   재검토하려면 별도의 calibration-flow 제안이 필요하다.
6. **`anchor_reliability`는 불확실성 트랙에 있다.** 이는 모션 피처가
   아니며 캡션에서 모션 피처와 합산되어서는 안 된다.
7. **`lateral_proxy_gyro`는 휴리스틱이다.** 사용된다면, 캡션은 이를
   knee-valgus 측정으로 표현해서는 안 된다.

---

## 10. 제안된 다음 단계

**Step 3 — 출력 스키마 및 불확실성 정책 설계.**

Step 3는 모델 학습이 **아니다**. 이는 어떠한 모델링 결정에도
선행한다. 이는 나중에 구현될 수 있을 만큼 충분히 자세하게, 센서-텍스트
시스템이 어떤 증거 하에서 무엇을 말할 수 있는지를 적어둔다.

Step 3가 산출해야 할 항목:

1. 클래스 확률 및 클래스 집합 출력(예: `{C1, C5, C6}`을 단일
   헤지된 예측으로)을 지원하는 **클래스 출력 스키마**.
2. 소수의 해석 가능한 피처(§5의 뱅크에서 가져오되, 아직 확정되지
   않음)가 어떻게 캡션 레이어로 노출되는지, 그리고 어떤 피처가
   *측정*으로, 어떤 피처가 *휴리스틱*으로 제시되는지를 설명하는
   **피처 증거 스키마**.
3. 정의된 어휘를 가진 **불확실성 플래그 스키마** — 최소한
   `confident_C2`, C1 / C5 / C6에 대한 `within_group_ambiguity`,
   C3 / C4에 대한 `pair_plus_c2_absorption`, 그리고 `anchor_unreliable`
   플래그를 포함.
4. **앵커 신뢰도 사용 정책**: rep의 앵커 신뢰도가 낮을 때, 시스템이
   캡션에서 무엇을 *억제*하는가 (예: 깊이 언어, 회복 언어)?
5. **캡션 신뢰도 수준 정책**: 클래스 사후 + 모호도 플래그 + 앵커
   신뢰도의 어떤 조합이 어떤 신뢰도 수준에 매핑되는지, 그리고 어떤
   것도 기준을 통과하지 못할 때의 *no-call* 정책은 무엇인가?
6. Step 3가 비교할 **명시적 정규화 후보** —
   `raw + posture-as-input`, `posture_train_zscore + posture-as-input`,
   그리고 Step 2.5의 후보 목록에서 그 외에는 없음.

Step 3는 이러한 정책이 작성되고 검토되며 스키마가 충분히 정의되어
Step 4 모델링 작업이 출력 설계 형태의 것을 새로 만들 일이 없을
때 종료된다.

---

*Step 2.5 종합으로 생성됨. 이 문서를 만들기 위해 새로운 피처는
계산되지 않았으며, 모델은 학습되지 않았으며, 매니페스트나 이전
감사 파일은 수정되지 않았다.*
