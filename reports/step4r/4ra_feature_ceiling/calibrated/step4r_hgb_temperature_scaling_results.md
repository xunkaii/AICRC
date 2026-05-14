# Step 4R-A 후처리 — Temperature Scaling 결과

- 생성 스크립트: `scripts/calibrate_step4r_hgb_temperature.py`
- 입력: `data/step4r/4ra_feature_ceiling/step4r_hgb_predictions_{raw,zscore}.csv`
- 기존 4R-A 산출물은 수정/덮어쓰기 없음. 본 단계 산출물은 모두 `calibrated/` 하위.

---

## 1. 본 단계의 위치

본 단계는 4R-A HGB ceiling의 over-confidence(test ECE ≈ 0.23)를 scalar temperature scaling으로 보정하여 4R-B와 **동일한 1-parameter post-hoc calibration narrative**로 통일한다. argmax/accuracy/macro F1은 monotonic scaling에서 변하지 않으므로, 본 단계의 효과는 *오직 calibration*에 있다 (§3 참조).

HGB는 `predict_proba`만 노출하고 decision_function이 없으므로 `log(predict_proba)`를 logits로 간주한다. `softmax(log(p)/T)`는 사전 logits z에 대해 `softmax(z/T)`와 수학적으로 동일하다 (정수 상수가 softmax에서 상쇄됨). 따라서 본 단계는 4R-B와 동일한 post-hoc temperature scaling이다.

**fitting 데이터:** validation split만 사용했다. test split은 어떤 형태로도 fitting에 노출되지 않았다.

---

## 2. fitted temperature

| branch | T | L-BFGS-B iters | final val NLL |
|---|---:|---:|---:|
| raw | 2.452007 | 8 | 1.588847 |
| zscore | 2.492160 | 7 | 1.604027 |

---

## 3. test ECE before / after (가장 중요한 지표)

### 3.1 branch = `raw`

| 지표 | before | after | 변화 |
|---|---:|---:|---:|
| ECE (15-bin) | 0.2300 | 0.0288 | -0.2012 |
| log loss | 1.8710 | 1.6216 | -0.2495 |
| Brier (multi) | 0.8659 | 0.7799 | -0.0860 |
| top1 prob mean | 0.5362 | 0.3341 | -0.2021 |
| predictive entropy mean | 1.1791 | 1.6070 | +0.4279 |

### 3.2 branch = `zscore`

| 지표 | before | after | 변화 |
|---|---:|---:|---:|
| ECE (15-bin) | 0.2271 | 0.0356 | -0.1914 |
| log loss | 1.8708 | 1.6299 | -0.2409 |
| Brier (multi) | 0.8638 | 0.7824 | -0.0814 |
| top1 prob mean | 0.5256 | 0.3249 | -0.2007 |
| predictive entropy mean | 1.1997 | 1.6213 | +0.4217 |

---

## 4. Argmax 불변성 — accuracy / macro F1 / balanced accuracy

Temperature scaling은 logits을 양의 scalar로 나누는 monotonic 변환이라 argmax가 보존된다. accuracy / macro F1 / balanced accuracy는 before/after가 **수치적으로 동일**해야 한다 (수치 오차 1e-6 이내).

### 4.1 branch = `raw`

| split | n | acc before | acc after | macro F1 before | macro F1 after |
|---|---:|---:|---:|---:|---:|
| train | 6412 | 0.9844 | 0.9844 | 0.9844 | 0.9844 |
| val | 1436 | 0.3287 | 0.3287 | 0.3230 | 0.3230 |
| test | 1427 | 0.3062 | 0.3062 | 0.3004 | 0.3004 |

### 4.2 branch = `zscore`

| split | n | acc before | acc after | macro F1 before | macro F1 after |
|---|---:|---:|---:|---:|---:|
| train | 6412 | 0.9917 | 0.9917 | 0.9917 | 0.9917 |
| val | 1436 | 0.3196 | 0.3196 | 0.3140 | 0.3140 |
| test | 1427 | 0.2985 | 0.2985 | 0.2914 | 0.2914 |

---

## 5. 분포 지표 before / after (전체 split)

### 5.1 branch = `raw`

#### train split

| 지표 | before | after | 변화 |
|---|---:|---:|---:|
| log loss | 0.3702 | 0.9118 | +0.5416 |
| Brier (multi) | 0.1481 | 0.4407 | +0.2926 |
| ECE (15-bin) | 0.2713 | 0.5688 | +0.2976 |
| top1 prob mean | 0.7132 | 0.4156 | -0.2977 |
| top1 prob p25 | 0.6062 | 0.3384 | -0.2678 |
| top1 prob p50 | 0.7303 | 0.4003 | -0.3300 |
| top1 prob p75 | 0.8372 | 0.4745 | -0.3627 |
| top1-top2 margin mean | 0.5713 | 0.2177 | -0.3537 |
| predictive entropy mean | 0.8786 | 1.5238 | +0.6452 |

#### val split

| 지표 | before | after | 변화 |
|---|---:|---:|---:|
| log loss | 1.8251 | 1.5888 | -0.2362 |
| Brier (multi) | 0.8426 | 0.7647 | -0.0779 |
| ECE (15-bin) | 0.2155 | 0.0277 | -0.1879 |
| top1 prob mean | 0.5442 | 0.3420 | -0.2022 |
| top1 prob p25 | 0.3996 | 0.2646 | -0.1350 |
| top1 prob p50 | 0.5082 | 0.3138 | -0.1943 |
| top1 prob p75 | 0.6654 | 0.3842 | -0.2813 |
| top1-top2 margin mean | 0.3196 | 0.1171 | -0.2025 |
| predictive entropy mean | 1.1537 | 1.5888 | +0.4352 |

#### test split

| 지표 | before | after | 변화 |
|---|---:|---:|---:|
| log loss | 1.8710 | 1.6216 | -0.2495 |
| Brier (multi) | 0.8659 | 0.7799 | -0.0860 |
| ECE (15-bin) | 0.2300 | 0.0288 | -0.2012 |
| top1 prob mean | 0.5362 | 0.3341 | -0.2021 |
| top1 prob p25 | 0.3946 | 0.2623 | -0.1323 |
| top1 prob p50 | 0.4933 | 0.3075 | -0.1858 |
| top1 prob p75 | 0.6580 | 0.3786 | -0.2793 |
| top1-top2 margin mean | 0.3130 | 0.1132 | -0.1998 |
| predictive entropy mean | 1.1791 | 1.6070 | +0.4279 |

### 5.2 branch = `zscore`

#### train split

| 지표 | before | after | 변화 |
|---|---:|---:|---:|
| log loss | 0.3614 | 0.9230 | +0.5616 |
| Brier (multi) | 0.1412 | 0.4448 | +0.3036 |
| ECE (15-bin) | 0.2735 | 0.5815 | +0.3079 |
| top1 prob mean | 0.7182 | 0.4103 | -0.3079 |
| top1 prob p25 | 0.6117 | 0.3350 | -0.2767 |
| top1 prob p50 | 0.7368 | 0.3962 | -0.3406 |
| top1 prob p75 | 0.8415 | 0.4698 | -0.3716 |
| top1-top2 margin mean | 0.5843 | 0.2177 | -0.3667 |
| predictive entropy mean | 0.8784 | 1.5409 | +0.6625 |

#### val split

| 지표 | before | after | 변화 |
|---|---:|---:|---:|
| log loss | 1.8438 | 1.6040 | -0.2398 |
| Brier (multi) | 0.8472 | 0.7706 | -0.0766 |
| ECE (15-bin) | 0.2157 | 0.0297 | -0.1861 |
| top1 prob mean | 0.5354 | 0.3334 | -0.2019 |
| top1 prob p25 | 0.3926 | 0.2598 | -0.1328 |
| top1 prob p50 | 0.4968 | 0.3080 | -0.1889 |
| top1 prob p75 | 0.6582 | 0.3826 | -0.2755 |
| top1-top2 margin mean | 0.3075 | 0.1086 | -0.1989 |
| predictive entropy mean | 1.1688 | 1.6040 | +0.4352 |

#### test split

| 지표 | before | after | 변화 |
|---|---:|---:|---:|
| log loss | 1.8708 | 1.6299 | -0.2409 |
| Brier (multi) | 0.8638 | 0.7824 | -0.0814 |
| ECE (15-bin) | 0.2271 | 0.0356 | -0.1914 |
| top1 prob mean | 0.5256 | 0.3249 | -0.2007 |
| top1 prob p25 | 0.3818 | 0.2571 | -0.1247 |
| top1 prob p50 | 0.4876 | 0.3009 | -0.1867 |
| top1 prob p75 | 0.6484 | 0.3730 | -0.2754 |
| top1-top2 margin mean | 0.3009 | 0.1042 | -0.1967 |
| predictive entropy mean | 1.1997 | 1.6213 | +0.4217 |

---

## 6. 해석

- `raw` branch: T = 2.4520 > 1 → 원본 HGB가 **over-confident**였다. calibrated posterior는 더 평평해지고 top1 mean이 감소, predictive entropy가 증가한다.
- `zscore` branch: T = 2.4922 > 1 → 원본 HGB가 **over-confident**였다. calibrated posterior는 더 평평해지고 top1 mean이 감소, predictive entropy가 증가한다.

- argmax / accuracy / macro F1은 변하지 않는다 (§4). 본 단계의 효과는 분류 결정이 아니라 *불확실성 표현*에 있다.

---

## 7. 4R-B 후처리와의 정합성

본 단계는 `scripts/calibrate_step4r_attention_temperature.py`와 동일한 프로토콜이다: (a) val NLL만 사용, (b) test 미노출, (c) scalar T, (d) 1-parameter L-BFGS-B (4R-B는 torch LBFGS, 4R-A는 scipy L-BFGS-B로 구현이 다르지만 수학적으로 동일한 1D 무제약 최적화). 두 모델 모두 *같은 calibration 패밀리*에 속하므로 4R-A vs 4R-B 비교가 calibration 비대칭 없이 가능하다.

---

## 8. 다음 단계 — schema 재생성

후속 스크립트 `scripts/generate_step4r_hgb_schema_calibrated.py`가 본 단계의 `step4r_hgb_predictions_calibrated_{raw,zscore}.csv`를 입력으로 받아, val 기반 operational threshold를 자체 산출하고 Step 3 §3~§9 schema 를 재구성한다. 기존 `step4r_hgb_schema_outputs_*.csv`(LR-derived threshold 사용)는 *legacy reference*로 보존된다.

---

## 9. 산출물 목록

- `data/step4r/4ra_feature_ceiling/calibrated/step4r_hgb_predictions_calibrated_raw.csv`
- `data/step4r/4ra_feature_ceiling/calibrated/step4r_hgb_predictions_calibrated_zscore.csv`
- `reports/step4r/4ra_feature_ceiling/calibrated/step4r_hgb_temperature_scaling_metrics.csv` (long format: branch, split, stage, metric, value)
- `reports/step4r/4ra_feature_ceiling/calibrated/step4r_hgb_temperature_scaling_results.md` (본 보고서)

---

*본 보고서는 자동 생성된다. 기존 Step 1~4 / 4R-A / 4R-B 산출물은 수정되지 않으며, 기존 `step4r_hgb_predictions_*.csv`는 덮어쓰지 않는다.*
