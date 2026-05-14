# Step 4R-A — Feature-based Ceiling 결과 보고서

- 생성 스크립트: `scripts/train_step4r_feature_ceiling.py` (단일 실행으로 모든 산출물 생성)
- 입력: `data/step4/step4_modeling_dataset.csv` (read-only)
- 출력 디렉토리: `data/step4r/4ra_feature_ceiling/`, `reports/step4r/4ra_feature_ceiling/`
- 모델: HistGradientBoostingClassifier (class_weight=balanced via sample_weight, learning_rate=0.1, max_iter=200, max_depth=None, l2_regularization=0.0, early_stopping=False, random_state=42)
- 입력 feature: 기존 Step 4의 main 5 features (raw 또는 zscore branch) + posture one-hot 3개
- split: `v1_36_8_8` (participant-disjoint, 기존 split 컬럼 그대로 사용)

---

## 1. 본 실험의 위치

1. 본 실험은 **AICRC_v2의 main contribution이 아니다.** main pipeline은 `reports/step4_research_reframing.md`에서 정의된 uncertainty-aware sensor-to-schema-to-caption pipeline이며, 본 실험은 그 안의 **Step 4R-A — feature-based ceiling reference**이다.
2. 본 실험은 기존 Step 4 LR baseline과 **비교**되며, 그것을 *대체하지 않는다*. 기존 `data/step4/`, `reports/step4/` 산출물은 legacy/reference로 보존된다. 비교 결과는 §3에 정리한다.
3. HGB가 LR 대비 분류 또는 calibration 점수를 올려도 그 자체로는 **sensor-to-text learning을 의미하지 않는다.** 본 실험은 raw IMU sequence를 직접 학습하지 않으며, caption 생성도 수행하지 않는다.
4. 본 실험 결과는 후속 **Step 4R-B — BiGRU + Attention raw IMU sensor-to-schema** 실험의 **비교 기준**으로 사용된다 (reframing 문서 §5.5의 모델 채택 기준).

---

## 2. 분류·Calibration 지표 요약

### 2.1 raw branch

| split | n | accuracy | balanced_acc | macro F1 | weighted F1 | log loss | Brier | ECE (15-bin) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 6063 | 0.9926 | 0.9926 | 0.9926 | 0.9926 | 0.3326 | 0.1279 | 0.2564 |
| val | 716 | 0.3170 | 0.3168 | 0.3046 | 0.3044 | 1.9299 | 0.8723 | 0.2299 |
| test | 2496 | 0.2965 | 0.2959 | 0.2937 | 0.2943 | 2.0778 | 0.9008 | 0.2678 |

### 2.2 zscore branch

| split | n | accuracy | balanced_acc | macro F1 | weighted F1 | log loss | Brier | ECE (15-bin) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 6063 | 0.9931 | 0.9931 | 0.9931 | 0.9931 | 0.3286 | 0.1247 | 0.2538 |
| val | 716 | 0.2947 | 0.2945 | 0.2767 | 0.2765 | 1.9213 | 0.8843 | 0.2435 |
| test | 2496 | 0.2877 | 0.2872 | 0.2824 | 0.2827 | 2.0728 | 0.9046 | 0.2694 |

---

## 3. LR baseline 대비 비교 (test split)

LR baseline 수치는 `reports/step4/step4_final_summary.md` 인용. HGB 수치는 본 실험 산출물.

### 3.1 raw branch (test)

| 지표 | LR baseline | HGB ceiling | 차이 (HGB − LR) |
|---|---:|---:|---:|
| accuracy | 0.3210 | 0.2965 | -0.0245 |
| macro F1 | 0.2843 | 0.2937 | +0.0094 |
| weighted F1 | 0.2841 | 0.2943 | +0.0102 |
| log loss | 1.6173 | 2.0778 | +0.4605 |
| Brier (multi) | 0.7746 | 0.9008 | +0.1262 |
| ECE (15-bin) | 0.0320 | 0.2678 | +0.2358 |
| C2 recall | 0.8159 | 0.4323 | -0.3836 |
| C1/C5/C6 internal | 0.2462 | 0.4041 | +0.1579 |
| C3/C4 pair | 0.2199 | 0.2910 | +0.0711 |
| C3 → C2 absorb | 0.3277 | 0.1850 | -0.1427 |
| C4 → C2 absorb | 0.2723 | 0.1005 | -0.1718 |

### 3.2 zscore branch (test)

| 지표 | LR baseline | HGB ceiling | 차이 (HGB − LR) |
|---|---:|---:|---:|
| accuracy | 0.3203 | 0.2877 | -0.0326 |
| macro F1 | 0.2885 | 0.2824 | -0.0061 |
| weighted F1 | 0.2884 | 0.2827 | -0.0057 |
| log loss | 1.6141 | 2.0728 | +0.4587 |
| Brier (multi) | 0.7734 | 0.9046 | +0.1312 |
| ECE (15-bin) | 0.0260 | 0.2694 | +0.2434 |
| C2 recall | 0.8159 | 0.4371 | -0.3788 |
| C1/C5/C6 internal | 0.2238 | 0.4018 | +0.1780 |
| C3/C4 pair | 0.2008 | 0.2641 | +0.0633 |
| C3 → C2 absorb | 0.3361 | 0.1600 | -0.1761 |
| C4 → C2 absorb | 0.2681 | 0.1196 | -0.1485 |

---

## 4. Per-class precision / recall / F1 (test split)

### 4.1 raw branch (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.3552 | 0.4000 | 0.3763 | 420 |
| C2 | 0.4118 | 0.4323 | 0.4218 | 421 |
| C3 | 0.2437 | 0.2425 | 0.2431 | 400 |
| C4 | 0.2691 | 0.2608 | 0.2649 | 418 |
| C5 | 0.2181 | 0.2129 | 0.2155 | 418 |
| C6 | 0.2568 | 0.2267 | 0.2408 | 419 |

### 4.2 zscore branch (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.3201 | 0.3833 | 0.3489 | 420 |
| C2 | 0.3940 | 0.4371 | 0.4144 | 421 |
| C3 | 0.2687 | 0.2600 | 0.2643 | 400 |
| C4 | 0.2818 | 0.2967 | 0.2890 | 418 |
| C5 | 0.1887 | 0.1603 | 0.1734 | 418 |
| C6 | 0.2267 | 0.1862 | 0.2045 | 419 |

---

## 5. Top-1 confidence 및 top1-top2 margin 분포 (test split)

### 5.1 raw branch

| 통계 | top1 prob | top1-top2 margin |
|---|---:|---:|
| mean | 0.5643 | 0.3419 |
| p05 | 0.3005 | 0.0188 |
| p25 | 0.4182 | 0.1127 |
| p50 | 0.5269 | 0.2693 |
| p75 | 0.6962 | 0.5177 |
| p95 | 0.9356 | 0.8945 |

### 5.2 zscore branch

| 통계 | top1 prob | top1-top2 margin |
|---|---:|---:|
| mean | 0.5571 | 0.3370 |
| p05 | 0.2957 | 0.0187 |
| p25 | 0.4026 | 0.1129 |
| p50 | 0.5169 | 0.2592 |
| p75 | 0.6979 | 0.5203 |
| p95 | 0.9309 | 0.8868 |

---

## 6. Schema 출력 분포 (전체 9275행)

### 6.1 raw branch

**caption_confidence_level**:

| confident | hedged | low | no_call |
|---:|---:|---:|---:|
| 1513 | 7259 | 372 | 131 |

**class_set_prediction**:

| class_set | count |
|---|---:|
| `["C1", "C5", "C6"]` | 3291 |
| `["C1", "C5"]` | 455 |
| `["C1", "C6"]` | 410 |
| `["C2"]` | 1560 |
| `["C3", "C4", "C2"]` | 516 |
| `["C3", "C4"]` | 2457 |
| `["C5", "C6"]` | 455 |
| `[]` | 131 |

**uncertainty_flags occurrence**:

| flag | rows containing it |
|---|---:|
| `anchor_unreliable` | 441 |
| `confident_C2` | 1576 |
| `low_confidence_no_class_set` | 114 |
| `pair_ambiguity_c3_c4` | 2457 |
| `pair_plus_c2_absorption` | 517 |
| `within_group_ambiguity_c1_c5_c6` | 4611 |

### 6.2 zscore branch

**caption_confidence_level**:

| confident | hedged | low | no_call |
|---:|---:|---:|---:|
| 1527 | 7206 | 364 | 178 |

**class_set_prediction**:

| class_set | count |
|---|---:|
| `["C1", "C5", "C6"]` | 3266 |
| `["C1", "C5"]` | 418 |
| `["C1", "C6"]` | 427 |
| `["C2"]` | 1578 |
| `["C3", "C4", "C2"]` | 480 |
| `["C3", "C4"]` | 2520 |
| `["C5", "C6"]` | 408 |
| `[]` | 178 |

**uncertainty_flags occurrence**:

| flag | rows containing it |
|---|---:|
| `anchor_unreliable` | 441 |
| `confident_C2` | 1593 |
| `low_confidence_no_class_set` | 161 |
| `pair_ambiguity_c3_c4` | 2520 |
| `pair_plus_c2_absorption` | 482 |
| `within_group_ambiguity_c1_c5_c6` | 4519 |

---

## 7. Confusion matrix (test split)

전체 train/val/test confusion 데이터는 `reports/step4r/4ra_feature_ceiling/step4r_feature_ceiling_confusion.csv`에 long format(condition, split, true_class, pred_class, count)으로 저장된다.

### 7.1 raw branch (test)

| true \ pred | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 168 | 44 | 31 | 37 | 74 | 66 |
| C2 | 43 | 182 | 56 | 50 | 50 | 40 |
| C3 | 28 | 74 | 97 | 108 | 51 | 42 |
| C4 | 29 | 42 | 130 | 109 | 57 | 51 |
| C5 | 98 | 57 | 38 | 60 | 89 | 76 |
| C6 | 107 | 43 | 46 | 41 | 87 | 95 |

### 7.2 zscore branch (test)

| true \ pred | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 161 | 58 | 34 | 43 | 70 | 54 |
| C2 | 55 | 184 | 51 | 55 | 33 | 43 |
| C3 | 36 | 64 | 104 | 103 | 50 | 43 |
| C4 | 31 | 50 | 113 | 124 | 50 | 50 |
| C5 | 114 | 52 | 49 | 60 | 67 | 76 |
| C6 | 106 | 59 | 36 | 55 | 85 | 78 |

---

## 8. 본 실험에서 명시적으로 결정하지 않는 사항

- **HGB hyperparameter는 grid search하지 않았다.** ceiling reference 한 점만 산출하는 것이 목적이므로 default 위주(learning_rate=0.1, max_iter=200, early_stopping=False)로 단일 학습을 수행했다.
- **HGB 전용 threshold 재calibration은 수행하지 않았다.** schema 출력 생성에는 LR-derived val threshold(`reports/step4/step4_threshold_calibration.md`)를 그대로 사용했다. HGB의 posterior 분포는 LR과 다를 수 있으므로 schema 출력 분포(§6)는 *동일한 정책 하의 비교*로 읽어야 하며, threshold 재calibration은 별도 후속작업이다.
- **본 실험은 main contribution이 아니다.** main pipeline의 학습 본체는 Step 4R-B(BiGRU + Attention)이며, 본 실험은 그 비교 기준만 제공한다.
- **Step 5 ~ 7 caption layer는 본 실험의 범위 밖이다.** 본 실험은 schema CSV까지만 산출한다.

---

## 9. 산출물 목록

- `data/step4r/4ra_feature_ceiling/step4r_hgb_predictions_raw.csv`
- `data/step4r/4ra_feature_ceiling/step4r_hgb_predictions_zscore.csv`
- `data/step4r/4ra_feature_ceiling/step4r_hgb_schema_outputs_raw.csv`
- `data/step4r/4ra_feature_ceiling/step4r_hgb_schema_outputs_zscore.csv`
- `reports/step4r/4ra_feature_ceiling/step4r_feature_ceiling_metrics.csv` (long format)
- `reports/step4r/4ra_feature_ceiling/step4r_feature_ceiling_confusion.csv` (long format)
- `reports/step4r/4ra_feature_ceiling/step4r_feature_ceiling_results.md` (본 보고서)

---

*본 보고서는 `scripts/train_step4r_feature_ceiling.py` 실행 시 자동 생성된다. 새로운 split / 데이터 / 캡션은 본 실험으로 인해 생성되지 않으며, 기존 Step 1 ~ 4 산출물은 수정되지 않는다.*
