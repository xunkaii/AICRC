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
| train | 6412 | 0.9844 | 0.9844 | 0.9844 | 0.9844 | 0.3702 | 0.1481 | 0.2713 |
| val | 1436 | 0.3287 | 0.3286 | 0.3230 | 0.3230 | 1.8251 | 0.8426 | 0.2155 |
| test | 1427 | 0.3062 | 0.3062 | 0.3004 | 0.3003 | 1.8710 | 0.8659 | 0.2300 |

### 2.2 zscore branch

| split | n | accuracy | balanced_acc | macro F1 | weighted F1 | log loss | Brier | ECE (15-bin) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 6412 | 0.9917 | 0.9917 | 0.9917 | 0.9917 | 0.3614 | 0.1412 | 0.2735 |
| val | 1436 | 0.3196 | 0.3196 | 0.3140 | 0.3139 | 1.8438 | 0.8472 | 0.2157 |
| test | 1427 | 0.2985 | 0.2986 | 0.2914 | 0.2913 | 1.8708 | 0.8638 | 0.2271 |

---

## 3. LR baseline 대비 비교 (test split)

LR baseline 수치는 `reports/step4/step4_final_summary.md` 인용. HGB 수치는 본 실험 산출물.

### 3.1 raw branch (test)

| 지표 | LR baseline | HGB ceiling | 차이 (HGB − LR) |
|---|---:|---:|---:|
| accuracy | 0.3210 | 0.3062 | -0.0148 |
| macro F1 | 0.2843 | 0.3004 | +0.0161 |
| weighted F1 | 0.2841 | 0.3003 | +0.0162 |
| log loss | 1.6173 | 1.8710 | +0.2537 |
| Brier (multi) | 0.7746 | 0.8659 | +0.0913 |
| ECE (15-bin) | 0.0320 | 0.2300 | +0.1980 |
| C2 recall | 0.8159 | 0.5565 | -0.2594 |
| C1/C5/C6 internal | 0.2462 | 0.4336 | +0.1874 |
| C3/C4 pair | 0.2199 | 0.1945 | -0.0254 |
| C3 → C2 absorb | 0.3277 | 0.2773 | -0.0504 |
| C4 → C2 absorb | 0.2723 | 0.1915 | -0.0808 |

### 3.2 zscore branch (test)

| 지표 | LR baseline | HGB ceiling | 차이 (HGB − LR) |
|---|---:|---:|---:|
| accuracy | 0.3203 | 0.2985 | -0.0218 |
| macro F1 | 0.2885 | 0.2914 | +0.0029 |
| weighted F1 | 0.2884 | 0.2913 | +0.0029 |
| log loss | 1.6141 | 1.8708 | +0.2567 |
| Brier (multi) | 0.7734 | 0.8638 | +0.0904 |
| ECE (15-bin) | 0.0260 | 0.2271 | +0.2011 |
| C2 recall | 0.8159 | 0.5565 | -0.2594 |
| C1/C5/C6 internal | 0.2238 | 0.4210 | +0.1972 |
| C3/C4 pair | 0.2008 | 0.2304 | +0.0296 |
| C3 → C2 absorb | 0.3361 | 0.2563 | -0.0798 |
| C4 → C2 absorb | 0.2681 | 0.1702 | -0.0979 |

---

## 4. Per-class precision / recall / F1 (test split)

### 4.1 raw branch (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.3128 | 0.2785 | 0.2946 | 237 |
| C2 | 0.3528 | 0.5565 | 0.4318 | 239 |
| C3 | 0.3661 | 0.2815 | 0.3183 | 238 |
| C4 | 0.3604 | 0.3021 | 0.3287 | 235 |
| C5 | 0.2549 | 0.2185 | 0.2353 | 238 |
| C6 | 0.1882 | 0.2000 | 0.1939 | 240 |

### 4.2 zscore branch (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.2810 | 0.2489 | 0.2640 | 237 |
| C2 | 0.3410 | 0.5565 | 0.4229 | 239 |
| C3 | 0.3260 | 0.2479 | 0.2816 | 238 |
| C4 | 0.3667 | 0.3277 | 0.3461 | 235 |
| C5 | 0.2938 | 0.2395 | 0.2639 | 238 |
| C6 | 0.1694 | 0.1708 | 0.1701 | 240 |

---

## 5. Top-1 confidence 및 top1-top2 margin 분포 (test split)

### 5.1 raw branch

| 통계 | top1 prob | top1-top2 margin |
|---|---:|---:|
| mean | 0.5362 | 0.3130 |
| p05 | 0.2943 | 0.0169 |
| p25 | 0.3946 | 0.1053 |
| p50 | 0.4933 | 0.2373 |
| p75 | 0.6580 | 0.4742 |
| p95 | 0.8970 | 0.8315 |

### 5.2 zscore branch

| 통계 | top1 prob | top1-top2 margin |
|---|---:|---:|
| mean | 0.5256 | 0.3009 |
| p05 | 0.2840 | 0.0190 |
| p25 | 0.3818 | 0.0968 |
| p50 | 0.4876 | 0.2255 |
| p75 | 0.6484 | 0.4609 |
| p95 | 0.8820 | 0.8141 |

---

## 6. Schema 출력 분포 (전체 9275행)

### 6.1 raw branch

**caption_confidence_level**:

| confident | hedged | low | no_call |
|---:|---:|---:|---:|
| 1596 | 7179 | 370 | 130 |

**class_set_prediction**:

| class_set | count |
|---|---:|
| `["C1", "C5", "C6"]` | 3177 |
| `["C1", "C5"]` | 424 |
| `["C1", "C6"]` | 491 |
| `["C2"]` | 1650 |
| `["C3", "C4", "C2"]` | 522 |
| `["C3", "C4"]` | 2361 |
| `["C5", "C6"]` | 520 |
| `[]` | 130 |

**uncertainty_flags occurrence**:

| flag | rows containing it |
|---|---:|
| `anchor_unreliable` | 441 |
| `confident_C2` | 1664 |
| `low_confidence_no_class_set` | 114 |
| `pair_ambiguity_c3_c4` | 2361 |
| `pair_plus_c2_absorption` | 524 |
| `within_group_ambiguity_c1_c5_c6` | 4612 |

### 6.2 zscore branch

**caption_confidence_level**:

| confident | hedged | low | no_call |
|---:|---:|---:|---:|
| 1588 | 7155 | 362 | 170 |

**class_set_prediction**:

| class_set | count |
|---|---:|
| `["C1", "C5", "C6"]` | 3302 |
| `["C1", "C5"]` | 358 |
| `["C1", "C6"]` | 433 |
| `["C2"]` | 1641 |
| `["C3", "C4", "C2"]` | 612 |
| `["C3", "C4"]` | 2325 |
| `["C5", "C6"]` | 434 |
| `[]` | 170 |

**uncertainty_flags occurrence**:

| flag | rows containing it |
|---|---:|
| `anchor_unreliable` | 441 |
| `confident_C2` | 1656 |
| `low_confidence_no_class_set` | 154 |
| `pair_ambiguity_c3_c4` | 2325 |
| `pair_plus_c2_absorption` | 613 |
| `within_group_ambiguity_c1_c5_c6` | 4527 |

---

## 7. Confusion matrix (test split)

전체 train/val/test confusion 데이터는 `reports/step4r/4ra_feature_ceiling/step4r_feature_ceiling_confusion.csv`에 long format(condition, split, true_class, pred_class, count)으로 저장된다.

### 7.1 raw branch (test)

| true \ pred |  C1 |  C2 |  C3 |  C4 |  C5 |  C6 |
| ----------- | --: | --: | --: | --: | --: | --: |
| C1          |  66 |  34 |  12 |  14 |  53 |  58 |
| C2          |  23 | 133 |  30 |  14 |   7 |  32 |
| C3          |  21 |  66 |  67 |  41 |  19 |  24 |
| C4          |  17 |  45 |  51 |  71 |  23 |  28 |
| C5          |  45 |  33 |   5 |  38 |  52 |  65 |
| C6          |  39 |  66 |  18 |  19 |  50 |  48 |

### 7.2 zscore branch (test)

| true \ pred | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 59 | 46 | 9 | 13 | 48 | 62 |
| C2 | 19 | 133 | 35 | 10 | 8 | 34 |
| C3 | 26 | 61 | 59 | 58 | 14 | 20 |
| C4 | 11 | 40 | 51 | 77 | 20 | 36 |
| C5 | 45 | 44 | 10 | 33 | 57 | 49 |
| C6 | 50 | 66 | 17 | 19 | 47 | 41 |

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
