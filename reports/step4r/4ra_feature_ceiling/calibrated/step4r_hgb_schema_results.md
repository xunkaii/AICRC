# Step 4R-A 후처리 — Calibrated Schema Output 결과

- 생성 스크립트: `scripts/generate_step4r_hgb_schema_calibrated.py`
- 입력: `data/step4r/4ra_feature_ceiling/calibrated/step4r_hgb_predictions_calibrated_{raw,zscore}.csv`
- 기존 4R-A 산출물(LR-derived threshold 사용)은 *legacy reference*로 보존된다.

---

## 1. 본 단계의 위치

본 단계는 4R-A HGB temperature scaling 산출물을 Step 3 §3~§9 schema로 변환한다. 4R-B post 2/3와 **동일한 절차**로 val 기반 operational threshold를 재도출하여, 4R-A vs 4R-B 비교가 calibration·threshold 양쪽에서 대칭이 되도록 한다.

---

## 2. fitted temperature (재인용)

| branch | T |
|---|---:|
| raw | 2.452007 |
| zscore | 2.492160 |

T 산출 절차 및 before/after metric 변화는 `step4r_hgb_temperature_scaling_results.md` 참조.

---

## 3. Operational thresholds (val split 자체 산출)

Step 3 §6/§8은 임계값을 commit하지 않으므로, 본 단계는 HGB calibrated posterior에 대해 val split만 사용해 동일 절차로 임계값을 자체 산출했다. **temporary operational thresholds**이며, 후속 단계에서 갱신될 수 있다.

| threshold | raw | zscore |
|---|---:|---:|
| confident_C2_threshold | 0.3700 | 0.4400 |
| non_trivial_C2_threshold | 0.1115 | 0.1274 |
| within_group_threshold | 0.1146 | 0.1101 |
| anchor_suppression_threshold | 0.5000 | 0.5000 |
| anchor_no_call_threshold | 0.2500 | 0.2500 |

---

## 4. split별 caption_confidence_level 분포

### 4.1 branch = `raw`

#### train split (n = 6412)

| level | count | rate |
|---|---:|---:|
| confident | 750 | 0.1170 |
| hedged | 5117 | 0.7980 |
| low | 245 | 0.0382 |
| no_call | 300 | 0.0468 |
| **(no_call total)** | **300** | **0.0468** |

#### val split (n = 1436)

| level | count | rate |
|---|---:|---:|
| confident | 145 | 0.1010 |
| hedged | 1048 | 0.7298 |
| low | 60 | 0.0418 |
| no_call | 183 | 0.1274 |
| **(no_call total)** | **183** | **0.1274** |

#### test split (n = 1427)

| level | count | rate |
|---|---:|---:|
| confident | 155 | 0.1086 |
| hedged | 996 | 0.6980 |
| low | 57 | 0.0399 |
| no_call | 219 | 0.1535 |
| **(no_call total)** | **219** | **0.1535** |

### 4.2 branch = `zscore`

#### train split (n = 6412)

| level | count | rate |
|---|---:|---:|
| confident | 473 | 0.0738 |
| hedged | 5106 | 0.7963 |
| low | 247 | 0.0385 |
| no_call | 586 | 0.0914 |
| **(no_call total)** | **586** | **0.0914** |

#### val split (n = 1436)

| level | count | rate |
|---|---:|---:|
| confident | 99 | 0.0689 |
| hedged | 1045 | 0.7277 |
| low | 53 | 0.0369 |
| no_call | 239 | 0.1664 |
| **(no_call total)** | **239** | **0.1664** |

#### test split (n = 1427)

| level | count | rate |
|---|---:|---:|
| confident | 95 | 0.0666 |
| hedged | 981 | 0.6875 |
| low | 58 | 0.0406 |
| no_call | 293 | 0.2053 |
| **(no_call total)** | **293** | **0.2053** |

---

## 5. class_set 크기 분포 (전체)

### 5.1 branch = `raw`

| size | count | rate |
|---|---:|---:|
| 0 | 702 | 0.0757 |
| 1 | 1086 | 0.1171 |
| 2 | 2711 | 0.2923 |
| 3 | 4776 | 0.5149 |

### 5.2 branch = `zscore`

| size | count | rate |
|---|---:|---:|
| 0 | 1118 | 0.1205 |
| 1 | 697 | 0.0751 |
| 2 | 2859 | 0.3082 |
| 3 | 4601 | 0.4961 |

---

## 6. ambiguity flag 발생률 (전체)

### 6.1 branch = `raw`

| flag | rows containing it | rate |
|---|---:|---:|
| `anchor_unreliable` | 441 | 0.0475 |
| `confident_C2` | 1098 | 0.1184 |
| `low_confidence_no_class_set` | 680 | 0.0733 |
| `pair_ambiguity_c3_c4` | 1481 | 0.1597 |
| `pair_plus_c2_absorption` | 1404 | 0.1514 |
| `posture_unknown` | 0 | 0.0000 |
| `within_group_ambiguity_c1_c5_c6` | 4612 | 0.4973 |

### 6.2 branch = `zscore`

| flag | rows containing it | rate |
|---|---:|---:|
| `anchor_unreliable` | 441 | 0.0475 |
| `confident_C2` | 706 | 0.0761 |
| `low_confidence_no_class_set` | 1104 | 0.1190 |
| `pair_ambiguity_c3_c4` | 1689 | 0.1821 |
| `pair_plus_c2_absorption` | 1249 | 0.1347 |
| `posture_unknown` | 0 | 0.0000 |
| `within_group_ambiguity_c1_c5_c6` | 4527 | 0.4881 |

---

## 7. top1 argmax와 class_set의 차이

non-no_call 행 중에서 top1 argmax가 class_set에 포함되지 않는 비율.

### 7.1 branch = `raw`

| split | non-no_call n | top1 not in class_set | rate |
|---|---:|---:|---:|
| train | 6112 | 0 | 0.0000 |
| val | 1253 | 0 | 0.0000 |
| test | 1208 | 0 | 0.0000 |
| all | 8573 | 0 | 0.0000 |

### 7.2 branch = `zscore`

| split | non-no_call n | top1 not in class_set | rate |
|---|---:|---:|---:|
| train | 5826 | 0 | 0.0000 |
| val | 1197 | 0 | 0.0000 |
| test | 1134 | 0 | 0.0000 |
| all | 8157 | 0 | 0.0000 |

---

## 8. 4R-A 기존 schema(LR-derived threshold)와의 연결

기존 `step4r_hgb_schema_outputs_{raw,zscore}.csv`는 LR-derived val threshold (Step 4 `step4_threshold_calibration.md`)를 그대로 사용했다. 본 단계의 schema는 (a) HGB calibrated posterior를 사용하고, (b) val threshold를 calibrated posterior에서 다시 도출한 결과다. 기존 산출물은 *legacy reference*로 보존되며 본 단계 산출물이 4R-B와 비교되는 *공정한 reference*이다.

---

## 9. 4R-B와의 비교 정합성

본 단계 산출물(`step4r_hgb_schema_outputs_calibrated_{raw,zscore}.csv`)은 `step4r_bigru_attention_schema_outputs_calibrated.csv`와 다음을 *공유*한다:

- 동일한 Step 3 §3~§9 schema 어휘 (class_set whitelist, flag vocab, level enum, no_call 정책).
- 동일한 post-hoc calibration 패밀리 (1-parameter scalar T, val NLL).
- 동일한 operational threshold 도출 절차 (val 기반, 동일 정의식).
- 동일한 anchor 정책 (suppression / no_call 두 단계 threshold, anchor-dependent set 정의).

따라서 4R-A vs 4R-B 비교는 calibration·threshold 비대칭 없이 동일 schema 위에서 이루어진다.

---

## 10. 본 단계에서 명시적으로 결정하지 않는 사항

- LLM caption layer prompt / 어휘 / 금지 표현 — 본 단계 범위 밖.
- temporary operational threshold의 commit (실험 후 별도 단계에서 잠금).
- 4R-A를 main pipeline 모델로 승격 — reframing §5.5 채택 기준은 4R-B 결과에 의해 결정.

---

## 11. 산출물 목록

- `data/step4r/4ra_feature_ceiling/calibrated/step4r_hgb_schema_outputs_calibrated_raw.csv`
- `data/step4r/4ra_feature_ceiling/calibrated/step4r_hgb_schema_outputs_calibrated_zscore.csv`
- `reports/step4r/4ra_feature_ceiling/calibrated/step4r_hgb_schema_summary_raw.csv` (long format)
- `reports/step4r/4ra_feature_ceiling/calibrated/step4r_hgb_schema_summary_zscore.csv` (long format)
- `reports/step4r/4ra_feature_ceiling/calibrated/step4r_hgb_schema_results.md` (본 보고서)

---

*본 보고서는 자동 생성된다. 기존 Step 1~4 / 4R-A / 4R-B 산출물은 수정되지 않으며, 기존 `step4r_hgb_schema_outputs_*.csv`는 덮어쓰지 않는다.*
