# Step 4R-B 후처리 3/3 — Attention Phase Analysis 결과

- 생성 스크립트: `scripts/analyze_step4r_attention_phase.py`
- attention_weights는 best checkpoint로 전체 sample에 대해 재추출했다.
- phase 정의:
  - `descending`: timestep 0~42
  - `bottom_transition`: timestep 43~85
  - `ascending_recovery`: timestep 86~127
- anchor: posture-aware (`SA/CA → ensemble_bottom_idx`, `HW → acc_bottom_idx`), 원본 timestep을 `n_rows`로 0~127 범위로 rescale.

---

## 0. 주의 문구 — attention 해석의 한계

- attention은 정답 근거가 아니라 **모델 내부 근거의 보조 분석**이다.
- attention entropy를 **센서 관측 가능성의 직접 증거로 과장하지 않는다**.
- 본 분석은 posterior entropy, confusion pattern, attention phase를 *함께* 해석할 때만 reframing 문서 §4의 이론 기반(정보이론·관측이론)과 연결된다.

---

## 1. 본 단계의 위치

calibrated 분류기 위에서, 모델이 시계열의 어느 phase에 *주의를 기울였는지*를 정량화한다. reframing 문서 §4.3 (Attention 기반 시계열 근거 분석)과 §8.2 (posterior entropy + attention entropy + confusion pattern)의 입력이 된다.

---

## 2. attention entropy 전체 요약

`-Σ a_t log a_t` (자연로그). T=128 균등분포 상한 ≈ 4.8520.

| split | n | entropy mean | entropy p25 | entropy p50 | entropy p75 | peak ts mean |
|---|---:|---:|---:|---:|---:|---:|
| train | 6412 | 3.1009 | 2.6879 | 3.0724 | 3.4872 | 42.54 |
| val | 1436 | 3.1907 | 2.8373 | 3.1592 | 3.5075 | 48.27 |
| test | 1427 | 3.2496 | 2.9044 | 3.2207 | 3.5791 | 46.84 |

**전체 (all splits):**

- entropy mean = 3.1377, p25 = 2.7481, p50 = 3.1162, p75 = 3.5024

**전체 phase mass 평균:**

| phase | mean mass |
|---|---:|
| descending | 0.5107 |
| bottom_transition | 0.3039 |
| ascending_recovery | 0.1854 |

**전체 peak phase 분포:**

| peak phase | count | rate |
|---|---:|---:|
| descending | 5033 | 0.5426 |
| bottom_transition | 2941 | 0.3171 |
| ascending_recovery | 1301 | 0.1403 |

---

## 3. Class별 attention peak phase / phase mass

| class_id | n | entropy mean | desc mass | bot mass | rec mass | most common peak | distance mean (n) |
|---|---:|---:|---:|---:|---:|---|---|
| C1 | 1553 | 3.0641 | 0.3543 | 0.3851 | 0.2606 | bottom_transition | 27.18 (1553) |
| C2 | 1555 | 3.5768 | 0.3320 | 0.4002 | 0.2678 | bottom_transition | 31.68 (1555) |
| C3 | 1529 | 3.2520 | 0.5701 | 0.2612 | 0.1687 | descending | 38.11 (1529) |
| C4 | 1540 | 3.0475 | 0.6462 | 0.2468 | 0.1070 | descending | 40.98 (1540) |
| C5 | 1549 | 2.9338 | 0.5780 | 0.2658 | 0.1562 | descending | 41.89 (1549) |
| C6 | 1549 | 2.9515 | 0.5860 | 0.2631 | 0.1509 | descending | 41.88 (1549) |

Step 2.5 §3 (시간 국소화된 클래스 차이)에서 SA/CA의 acc_z η²가 bottom **이후** 에 정점이라고 보고됐다. 본 모델의 peak phase 분포가 그 관찰과 일관되는지(예: ascending_recovery 비중이 substantial한지) 확인한다.

---

## 4. Posture별 attention peak phase / phase mass

| posture_canonical | n | entropy mean | desc mass | bot mass | rec mass | most common peak | distance mean (n) |
|---|---:|---:|---:|---:|---:|---|---|
| CA | 3080 | 3.0242 | 0.2111 | 0.5166 | 0.2722 | bottom_transition | 20.69 (3080) |
| HW | 3097 | 3.1326 | 0.7091 | 0.2217 | 0.0692 | descending | 35.60 (3097) |
| SA | 3098 | 3.2557 | 0.6101 | 0.1747 | 0.2152 | descending | 54.44 (3098) |

Step 2.5 §4의 HW 자세에서 acc-bottom과 gyro-peak가 다른 물리 이벤트라는 점, 그리고 §6의 자세 인식 정규화 효과 등을 고려해 자세별 attention 패턴 차이를 본다.

---

## 5. correct vs incorrect 비교

| correct | n | entropy mean | desc mass | bot mass | rec mass | most common peak | distance mean (n) |
|---|---:|---:|---:|---:|---:|---|---|
| False | 3213 | 3.1478 | 0.5313 | 0.2923 | 0.1765 | descending | 38.59 (3213) |
| True | 6062 | 3.1323 | 0.4997 | 0.3101 | 0.1901 | descending | 36.07 (6062) |

correct 행에서 attention entropy가 더 낮고 phase 분포가 더 집중되는지(또는 특정 phase에 mass가 쏠리는지)를 본다. 차이가 작으면 attention이 정답 여부에 직접적으로 정렬되지 않는다는 신호이다.

---

## 6. no_call vs non_no_call 비교

| no_call | n | entropy mean | desc mass | bot mass | rec mass | most common peak | distance mean (n) |
|---|---:|---:|---:|---:|---:|---|---|
| False | 9184 | 3.1351 | 0.5121 | 0.3035 | 0.1844 | descending | 36.96 (9184) |
| True | 91 | 3.4028 | 0.3669 | 0.3438 | 0.2894 | bottom_transition | 35.11 (91) |

---

## 7. Ambiguity group별 비교

| ambiguity_group | n | entropy mean | desc mass | bot mass | rec mass | most common peak | distance mean (n) |
|---|---:|---:|---:|---:|---:|---|---|
| confident_C2 | 1538 | 3.6529 | 0.3134 | 0.3967 | 0.2898 | bottom_transition | 31.20 (1538) |
| no_call | 91 | 3.4028 | 0.3669 | 0.3438 | 0.2894 | bottom_transition | 35.11 (91) |
| pair_c3_c4 | 1498 | 2.8919 | 0.8619 | 0.1002 | 0.0379 | descending | 44.06 (1498) |
| pair_plus_c2_absorption | 1558 | 3.3556 | 0.4008 | 0.3876 | 0.2116 | descending | 36.53 (1558) |
| within_group_c1_c5_c6 | 4590 | 2.9661 | 0.5023 | 0.3102 | 0.1876 | descending | 36.72 (4590) |

Step 2.5의 ambiguity 패턴을 본 모델의 attention이 어떻게 다루는지 본다 — 특히 within_group_c1_c5_c6와 pair_plus_c2_absorption 그룹에서 entropy/phase mass가 confident_C2와 *얼마나 다른지*가 핵심.

---

## 8. anchor-attention distance 요약

- distance 가용 sample 수: 9275 / 9275
- distance mean = 36.94, p25 = 7.00, p50 = 34.00, p75 = 68.00, p95 = 81.00

**posture별 anchor-attention distance:**

| posture | n with distance | mean | p25 | p50 | p75 |
|---|---:|---:|---:|---:|---:|
| SA | 3098 | 54.44 | 15.00 | 69.00 | 76.75 |
| CA | 3080 | 20.69 | 3.00 | 6.00 | 17.00 |
| HW | 3097 | 35.60 | 24.00 | 37.00 | 48.00 |

anchor-attention distance는 *해석 보조 지표*다. 작다고 해서 모델이 옳다는 의미가 아니며, 크다고 해서 틀렸다는 의미도 아니다 — 자세별 anchor 정의 자체가 Step 2.5의 한계(특히 HW에서 acc/gyro 일치도 0.044~0.225)를 안고 있다.

---

## 9. 산출물 목록

- `data/step4r/4rb_attention/step4r_bigru_attention_attention_weights.npz`
  - 키: `attention_weights` (N, 128), `sample_id`, `split`, `class_id`, `posture_canonical`
- `data/step4r/4rb_attention/step4r_bigru_attention_attention_summary.csv` (per-sample)
- `reports/step4r/4rb_attention/step4r_attention_phase_by_class.csv`
- `reports/step4r/4rb_attention/step4r_attention_phase_by_posture.csv`
- `reports/step4r/4rb_attention/step4r_attention_phase_by_correctness.csv`
- `reports/step4r/4rb_attention/step4r_attention_phase_analysis.md` (본 보고서)

---

*본 보고서는 자동 생성된다. 기존 Step 1 ~ 4 / 4R-A / 4R-B 산출물은 수정되지 않으며, 기존 attention 관련 컬럼을 가진 다른 CSV는 영향 없음.*
