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
| raw | 2.813425 | 16 | 1.621007 |
| zscore | 2.795383 | 7 | 1.626586 |

---

## 3. test ECE before / after (가장 중요한 지표)

### 3.1 branch = `raw`

| 지표 | before | after | 변화 |
|---|---:|---:|---:|
| ECE (15-bin) | 0.2678 | 0.0372 | -0.2306 |
| log loss | 2.0778 | 1.6536 | -0.4242 |
| Brier (multi) | 0.9008 | 0.7879 | -0.1128 |
| top1 prob mean | 0.5643 | 0.3337 | -0.2306 |
| predictive entropy mean | 1.0985 | 1.5998 | +0.5013 |

### 3.2 branch = `zscore`

| 지표 | before | after | 변화 |
|---|---:|---:|---:|
| ECE (15-bin) | 0.2694 | 0.0416 | -0.2278 |
| log loss | 2.0728 | 1.6632 | -0.4097 |
| Brier (multi) | 0.9046 | 0.7917 | -0.1129 |
| top1 prob mean | 0.5571 | 0.3291 | -0.2280 |
| predictive entropy mean | 1.1196 | 1.6104 | +0.4908 |

---

## 4. Argmax 불변성 — accuracy / macro F1 / balanced accuracy

Temperature scaling은 logits을 양의 scalar로 나누는 monotonic 변환이라 argmax가 보존된다. accuracy / macro F1 / balanced accuracy는 before/after가 **수치적으로 동일**해야 한다 (수치 오차 1e-6 이내).

### 4.1 branch = `raw`

| split | n | acc before | acc after | macro F1 before | macro F1 after |
|---|---:|---:|---:|---:|---:|
| train | 6063 | 0.9926 | 0.9926 | 0.9926 | 0.9926 |
| val | 716 | 0.3170 | 0.3170 | 0.3046 | 0.3046 |
| test | 2496 | 0.2965 | 0.2965 | 0.2937 | 0.2937 |

### 4.2 branch = `zscore`

| split | n | acc before | acc after | macro F1 before | macro F1 after |
|---|---:|---:|---:|---:|---:|
| train | 6063 | 0.9931 | 0.9931 | 0.9931 | 0.9931 |
| val | 716 | 0.2947 | 0.2947 | 0.2767 | 0.2767 |
| test | 2496 | 0.2877 | 0.2877 | 0.2824 | 0.2824 |

---

## 5. 분포 지표 before / after (전체 split)

### 5.1 branch = `raw`

#### train split

| 지표 | before | after | 변화 |
|---|---:|---:|---:|
| log loss | 0.3326 | 0.9567 | +0.6240 |
| Brier (multi) | 0.1279 | 0.4659 | +0.3380 |
| ECE (15-bin) | 0.2564 | 0.5975 | +0.3412 |
| top1 prob mean | 0.7362 | 0.3950 | -0.3412 |
| top1 prob p25 | 0.6346 | 0.3254 | -0.3092 |
| top1 prob p50 | 0.7552 | 0.3825 | -0.3727 |
| top1 prob p75 | 0.8518 | 0.4494 | -0.4023 |
| top1-top2 margin mean | 0.6011 | 0.1947 | -0.4064 |
| predictive entropy mean | 0.8231 | 1.5579 | +0.7348 |

#### val split

| 지표 | before | after | 변화 |
|---|---:|---:|---:|
| log loss | 1.9299 | 1.6210 | -0.3089 |
| Brier (multi) | 0.8723 | 0.7788 | -0.0934 |
| ECE (15-bin) | 0.2299 | 0.0568 | -0.1731 |
| top1 prob mean | 0.5470 | 0.3224 | -0.2246 |
| top1 prob p25 | 0.3908 | 0.2530 | -0.1377 |
| top1 prob p50 | 0.5009 | 0.2938 | -0.2072 |
| top1 prob p75 | 0.6897 | 0.3670 | -0.3227 |
| top1-top2 margin mean | 0.3246 | 0.1013 | -0.2233 |
| predictive entropy mean | 1.1377 | 1.6210 | +0.4833 |

#### test split

| 지표 | before | after | 변화 |
|---|---:|---:|---:|
| log loss | 2.0778 | 1.6536 | -0.4242 |
| Brier (multi) | 0.9008 | 0.7879 | -0.1128 |
| ECE (15-bin) | 0.2678 | 0.0372 | -0.2306 |
| top1 prob mean | 0.5643 | 0.3337 | -0.2306 |
| top1 prob p25 | 0.4182 | 0.2611 | -0.1572 |
| top1 prob p50 | 0.5269 | 0.3067 | -0.2202 |
| top1 prob p75 | 0.6962 | 0.3782 | -0.3180 |
| top1-top2 margin mean | 0.3419 | 0.1095 | -0.2324 |
| predictive entropy mean | 1.0985 | 1.5998 | +0.5013 |

### 5.2 branch = `zscore`

#### train split

| 지표 | before | after | 변화 |
|---|---:|---:|---:|
| log loss | 0.3286 | 0.9551 | +0.6264 |
| Brier (multi) | 0.1247 | 0.4633 | +0.3386 |
| ECE (15-bin) | 0.2538 | 0.5970 | +0.3432 |
| top1 prob mean | 0.7392 | 0.3960 | -0.3432 |
| top1 prob p25 | 0.6405 | 0.3276 | -0.3129 |
| top1 prob p50 | 0.7605 | 0.3833 | -0.3772 |
| top1 prob p75 | 0.8581 | 0.4508 | -0.4073 |
| top1-top2 margin mean | 0.6101 | 0.2004 | -0.4097 |
| predictive entropy mean | 0.8235 | 1.5620 | +0.7385 |

#### val split

| 지표 | before | after | 변화 |
|---|---:|---:|---:|
| log loss | 1.9213 | 1.6266 | -0.2947 |
| Brier (multi) | 0.8843 | 0.7863 | -0.0980 |
| ECE (15-bin) | 0.2435 | 0.0460 | -0.1975 |
| top1 prob mean | 0.5382 | 0.3189 | -0.2193 |
| top1 prob p25 | 0.3808 | 0.2480 | -0.1329 |
| top1 prob p50 | 0.4991 | 0.2933 | -0.2058 |
| top1 prob p75 | 0.6707 | 0.3633 | -0.3074 |
| top1-top2 margin mean | 0.3146 | 0.0991 | -0.2155 |
| predictive entropy mean | 1.1602 | 1.6266 | +0.4663 |

#### test split

| 지표 | before | after | 변화 |
|---|---:|---:|---:|
| log loss | 2.0728 | 1.6632 | -0.4097 |
| Brier (multi) | 0.9046 | 0.7917 | -0.1129 |
| ECE (15-bin) | 0.2694 | 0.0416 | -0.2278 |
| top1 prob mean | 0.5571 | 0.3291 | -0.2280 |
| top1 prob p25 | 0.4026 | 0.2554 | -0.1472 |
| top1 prob p50 | 0.5169 | 0.3029 | -0.2140 |
| top1 prob p75 | 0.6979 | 0.3745 | -0.3234 |
| top1-top2 margin mean | 0.3370 | 0.1072 | -0.2298 |
| predictive entropy mean | 1.1196 | 1.6104 | +0.4908 |

---

## 6. 해석

- `raw` branch: T = 2.8134 > 1 → 원본 HGB가 **over-confident**였다. calibrated posterior는 더 평평해지고 top1 mean이 감소, predictive entropy가 증가한다.
- `zscore` branch: T = 2.7954 > 1 → 원본 HGB가 **over-confident**였다. calibrated posterior는 더 평평해지고 top1 mean이 감소, predictive entropy가 증가한다.

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
