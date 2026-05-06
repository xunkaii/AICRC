# Step 4 — Threshold Calibration 후보

이 보고서는 Step 3 정책이 요구하는 다섯 임계값 — `confident_C2_threshold`, `non_trivial_C2_threshold`, `within_group_threshold`, `anchor_suppression_threshold`, `anchor_no_call_threshold` — 의 **후보값과 산출 근거**를 보고한다. 임계값을 commit하지 않으며, val split 분포 기준으로 산출되었다 (test 누수 방지).

---

## raw 조건

### `confident_C2_threshold`

- 후보값: **0.3800**
- 산출 방식: smallest t with precision >= 0.60 on val
- val에서 임계값 위 통계: precision=0.6000, recall=0.3875, coverage=0.1079, n_above=155
- sensitivity (인접 임계값에서의 precision / recall):

  | t | precision | recall | coverage | n_above |
  |---:|---:|---:|---:|---:|
  | 0.280 | 0.3473 | 0.6208 | 0.2987 | 429 |
  | 0.330 | 0.4417 | 0.5208 | 0.1971 | 283 |
  | 0.380 | 0.6000 | 0.3875 | 0.1079 | 155 |
  | 0.430 | 0.6813 | 0.2583 | 0.0634 | 91 |
  | 0.480 | 0.7059 | 0.1000 | 0.0237 | 34 |

### `non_trivial_C2_threshold`

- 후보값: **0.1123**
- 산출 방식: percentile 50 of p_C2 over val rows with pred_argmax in [C3, C4]
- 부분집합 크기 (val, pred_argmax ∈ {C3,C4}): 348
- p_C2 분포 (해당 부분집합): min=0.0001, p25=0.0443, median=0.1123, p75=0.1633, max=0.3024, mean=0.1108
- sensitivity 검토 방향: 임계값을 ±0.05 이동시킬 때 `[C3, C4, C2]` 헤지 진입 비율이 어떻게 변하는지 — Step 4 schema-output 단계에서 생성된 출력 CSV 위에서 점검한다.

### `within_group_threshold`

- 후보값: **0.1650**
- 산출 방식: percentile 25 of min(p_C1, p_C5, p_C6) over val rows with pred_argmax in [C1, C5, C6]
- 부분집합 크기 (val, pred_argmax ∈ {C1,C5,C6}): 471
- min(p_C1, p_C5, p_C6) 분포: min=0.0547, p25=0.1650, median=0.1829, p75=0.2089, max=0.2589, mean=0.1855
- sensitivity 검토 방향: 임계값을 ±0.02 이동시킬 때 `[C1, C5]` / `[C1, C6]` / `[C5, C6]` 부분집합 출력 비율이 어떻게 변하는지 — schema-output 단계에서 점검.

### `anchor_suppression_threshold` / `anchor_no_call_threshold`

- `anchor_suppression_threshold` 후보값: **0.5000**
- `anchor_no_call_threshold` 후보값: **0.2500**
- 관계 검증: `no_call < suppression` → 0.2500 < 0.5000 → **OK**
- 산출 근거: Boundary-aligned heuristic anchored on the four val bins. suppression = 0.5 places the cut between [0.25, 0.5) and [0.5, 0.75); no_call = 0.25 places the cut between [0, 0.25) and [0.25, 0.5). Constraint no_call < suppression is satisfied. Step 4 model-fit-aware refinement is left to a later commit.
- val anchor_reliability 구간별 성능 (재참조):

  | bin | n | accuracy | macro F1 | log loss | C2 recall |
  |---|---:|---:|---:|---:|---:|
  | [0.00, 0.25) | 26 | 0.2308 | 0.1201 | 1.5909 | 1.0000 |
  | [0.25, 0.50) | 51 | 0.2353 | 0.1966 | 1.6994 | 0.8333 |
  | [0.50, 0.75) | 310 | 0.2710 | 0.2315 | 1.6234 | 0.7273 |
  | [0.75, 1.00] | 1049 | 0.2717 | 0.2234 | 1.6418 | 0.7234 |
- sensitivity 검토 방향: 두 임계값을 각각 ±0.05 (한 bin 폭의 약 1/5) 이동시킬 때 (1) `anchor_unreliable` 플래그 부착 비율, (2) anchor 의존적 클래스 집합(`[C2]`, `[C3, C4, C2]`)의 no-call 진입 비율이 어떻게 변하는지 schema-output 단계에서 점검한다.

---

## zscore 조건

### `confident_C2_threshold`

- 후보값: **0.3900**
- 산출 방식: smallest t with precision >= 0.60 on val
- val에서 임계값 위 통계: precision=0.6071, recall=0.4250, coverage=0.1170, n_above=168
- sensitivity (인접 임계값에서의 precision / recall):

  | t | precision | recall | coverage | n_above |
  |---:|---:|---:|---:|---:|
  | 0.290 | 0.3419 | 0.6125 | 0.2994 | 430 |
  | 0.340 | 0.4371 | 0.5208 | 0.1992 | 286 |
  | 0.390 | 0.6071 | 0.4250 | 0.1170 | 168 |
  | 0.440 | 0.6875 | 0.2750 | 0.0669 | 96 |
  | 0.490 | 0.7800 | 0.1625 | 0.0348 | 50 |

### `non_trivial_C2_threshold`

- 후보값: **0.1059**
- 산출 방식: percentile 50 of p_C2 over val rows with pred_argmax in [C3, C4]
- 부분집합 크기 (val, pred_argmax ∈ {C3,C4}): 338
- p_C2 분포 (해당 부분집합): min=0.0000, p25=0.0384, median=0.1059, p75=0.1598, max=0.3140, mean=0.1068
- sensitivity 검토 방향: 임계값을 ±0.05 이동시킬 때 `[C3, C4, C2]` 헤지 진입 비율이 어떻게 변하는지 — Step 4 schema-output 단계에서 생성된 출력 CSV 위에서 점검한다.

### `within_group_threshold`

- 후보값: **0.1646**
- 산출 방식: percentile 25 of min(p_C1, p_C5, p_C6) over val rows with pred_argmax in [C1, C5, C6]
- 부분집합 크기 (val, pred_argmax ∈ {C1,C5,C6}): 465
- min(p_C1, p_C5, p_C6) 분포: min=0.0926, p25=0.1646, median=0.1833, p75=0.2065, max=0.2976, mean=0.1852
- sensitivity 검토 방향: 임계값을 ±0.02 이동시킬 때 `[C1, C5]` / `[C1, C6]` / `[C5, C6]` 부분집합 출력 비율이 어떻게 변하는지 — schema-output 단계에서 점검.

### `anchor_suppression_threshold` / `anchor_no_call_threshold`

- `anchor_suppression_threshold` 후보값: **0.5000**
- `anchor_no_call_threshold` 후보값: **0.2500**
- 관계 검증: `no_call < suppression` → 0.2500 < 0.5000 → **OK**
- 산출 근거: Boundary-aligned heuristic anchored on the four val bins. suppression = 0.5 places the cut between [0.25, 0.5) and [0.5, 0.75); no_call = 0.25 places the cut between [0, 0.25) and [0.25, 0.5). Constraint no_call < suppression is satisfied. Step 4 model-fit-aware refinement is left to a later commit.
- val anchor_reliability 구간별 성능 (재참조):

  | bin | n | accuracy | macro F1 | log loss | C2 recall |
  |---|---:|---:|---:|---:|---:|
  | [0.00, 0.25) | 26 | 0.3846 | 0.1624 | 1.5749 | 1.0000 |
  | [0.25, 0.50) | 51 | 0.2549 | 0.2103 | 1.6441 | 0.8333 |
  | [0.50, 0.75) | 310 | 0.2613 | 0.2199 | 1.6091 | 0.7273 |
  | [0.75, 1.00] | 1049 | 0.2784 | 0.2333 | 1.6324 | 0.7500 |
- sensitivity 검토 방향: 두 임계값을 각각 ±0.05 (한 bin 폭의 약 1/5) 이동시킬 때 (1) `anchor_unreliable` 플래그 부착 비율, (2) anchor 의존적 클래스 집합(`[C2]`, `[C3, C4, C2]`)의 no-call 진입 비율이 어떻게 변하는지 schema-output 단계에서 점검한다.

---

## 종합 노트

- 본 보고서의 후보값은 모두 val split 분포 기준이다. test split은 어떤 임계값 산출에도 사용되지 않았다.
- 두 정규화 후보(raw / zscore)에서의 후보값을 함께 보고하여, 정규화 선택이 임계값 calibration에 미치는 영향을 비교 가능하게 한다.
- 모든 임계값은 후보일 뿐이며 commit은 본 단계 범위 외이다 — Step 4 schema-output 생성 후 `validate_step4_schema_outputs.py`의 통과/실패와 §9 검증 결과를 함께 본 뒤 별도 단계에서 commit한다.
- `anchor_no_call_threshold < anchor_suppression_threshold` 관계는 두 조건 모두에서 만족됨 (Step 3 §6 위반 없음).

---

*Step 4 threshold calibration 후보 보고서로 생성됨. 임계값은 commit되지 않았고, 모델은 새로 학습되지 않았으며, schema 출력 CSV / 캡션 템플릿은 작성되지 않았다. 입력 prediction CSV는 read-only.*