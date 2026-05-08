# Step 4R-B 후처리 1/3 — Temperature Scaling 결과

- 생성 스크립트: `scripts/calibrate_step4r_attention_temperature.py`
- 입력 모델: `checkpoints/step4r/4rb_attention/best_bigru_attention.pt`
- 실행 device: `cuda`
- **fitted temperature T = 1.957900**
- LBFGS closure 평가 횟수: 16, 최종 val NLL: 1.151493

---

## 1. 본 단계의 위치

본 단계는 4R-B BiGRU+Attention의 over-confidence 문제(test ECE ≈ 0.13)를 scalar temperature scaling으로 보정한다. 분류 정확도와 macro F1은 monotonic scaling에서 변하지 않으므로, 본 단계의 효과는 *오직 calibration*에 있다 (§4 참조).

**fitting 데이터:** validation split만 사용했다. test split은 어떤 형태로도 fitting에 노출되지 않았다.

---

## 2. test ECE before / after (가장 중요한 지표)

| 지표 | before | after | 변화 |
|---|---:|---:|---:|
| ECE (15-bin) | 0.1340 | 0.0540 | -0.0800 |
| log loss | 1.2286 | 1.1296 | -0.0989 |
| Brier (multi) | 0.6258 | 0.5983 | -0.0276 |
| top1 prob mean | 0.6527 | 0.4985 | -0.1541 |
| predictive entropy mean | 0.8164 | 1.2007 | +0.3844 |

---

## 3. Argmax 불변성 — accuracy / macro F1 / balanced accuracy

Temperature scaling은 logits을 양의 scalar로 나누는 monotonic 변환이라 argmax가 보존된다. 따라서 accuracy, macro F1, balanced accuracy, confusion matrix는 before/after가 **수치적으로 동일**해야 한다 (수치 오차 1e-6 이내).

| split | n | acc before | acc after | macro F1 before | macro F1 after | balanced before | balanced after |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 6412 | 0.7110 | 0.7110 | 0.7091 | 0.7091 | 0.7106 | 0.7106 |
| val | 1436 | 0.5216 | 0.5216 | 0.5174 | 0.5174 | 0.5215 | 0.5215 |
| test | 1427 | 0.5284 | 0.5284 | 0.5184 | 0.5184 | 0.5285 | 0.5285 |

---

## 4. 분포 지표 before / after (전체 split)

### 4.1 train split

| 지표 | before | after | 변화 |
|---|---:|---:|---:|
| log loss | 0.6888 | 0.8232 | +0.1344 |
| Brier (multi) | 0.3848 | 0.4408 | +0.0560 |
| ECE (15-bin) | 0.0326 | 0.1868 | +0.1542 |
| top1 prob mean | 0.6785 | 0.5242 | -0.1543 |
| top1 prob p25 | 0.5340 | 0.4150 | -0.1190 |
| top1 prob p50 | 0.6647 | 0.5059 | -0.1589 |
| top1 prob p75 | 0.8329 | 0.6109 | -0.2220 |
| top1-top2 margin mean | 0.4474 | 0.2546 | -0.1928 |
| top1-top2 margin p25 | 0.1728 | 0.0771 | -0.0957 |
| top1-top2 margin p50 | 0.4028 | 0.1873 | -0.2155 |
| top1-top2 margin p75 | 0.7068 | 0.3738 | -0.3331 |
| predictive entropy mean | 0.7544 | 1.1415 | +0.3871 |
| predictive entropy p25 | 0.5314 | 0.9431 | +0.4117 |
| predictive entropy p50 | 0.7442 | 1.1363 | +0.3921 |
| predictive entropy p75 | 0.9910 | 1.3471 | +0.3561 |

### 4.2 val split

| 지표 | before | after | 변화 |
|---|---:|---:|---:|
| log loss | 1.3174 | 1.1515 | -0.1659 |
| Brier (multi) | 0.6352 | 0.5973 | -0.0378 |
| ECE (15-bin) | 0.1428 | 0.0270 | -0.1158 |
| top1 prob mean | 0.6629 | 0.5167 | -0.1462 |
| top1 prob p25 | 0.5098 | 0.3930 | -0.1168 |
| top1 prob p50 | 0.6316 | 0.4871 | -0.1445 |
| top1 prob p75 | 0.8290 | 0.6046 | -0.2244 |
| top1-top2 margin mean | 0.4239 | 0.2477 | -0.1762 |
| top1-top2 margin p25 | 0.1405 | 0.0616 | -0.0789 |
| top1-top2 margin p50 | 0.3432 | 0.1549 | -0.1883 |
| top1-top2 margin p75 | 0.6999 | 0.3679 | -0.3320 |
| predictive entropy mean | 0.7769 | 1.1507 | +0.3737 |
| predictive entropy p25 | 0.5490 | 0.9370 | +0.3880 |
| predictive entropy p50 | 0.7765 | 1.1641 | +0.3876 |
| predictive entropy p75 | 1.0479 | 1.4026 | +0.3546 |

### 4.3 test split

| 지표 | before | after | 변화 |
|---|---:|---:|---:|
| log loss | 1.2286 | 1.1296 | -0.0989 |
| Brier (multi) | 0.6258 | 0.5983 | -0.0276 |
| ECE (15-bin) | 0.1340 | 0.0540 | -0.0800 |
| top1 prob mean | 0.6527 | 0.4985 | -0.1541 |
| top1 prob p25 | 0.5101 | 0.3877 | -0.1223 |
| top1 prob p50 | 0.6355 | 0.4796 | -0.1559 |
| top1 prob p75 | 0.8060 | 0.5881 | -0.2180 |
| top1-top2 margin mean | 0.4163 | 0.2306 | -0.1857 |
| top1-top2 margin p25 | 0.1477 | 0.0636 | -0.0842 |
| top1-top2 margin p50 | 0.3588 | 0.1602 | -0.1985 |
| top1-top2 margin p75 | 0.6607 | 0.3380 | -0.3227 |
| predictive entropy mean | 0.8164 | 1.2007 | +0.3844 |
| predictive entropy p25 | 0.5786 | 1.0027 | +0.4241 |
| predictive entropy p50 | 0.8089 | 1.1957 | +0.3868 |
| predictive entropy p75 | 1.0697 | 1.4344 | +0.3647 |

---

## 5. 해석

- T = 1.9579 > 1 → 원본 모델이 **over-confident**였다. calibrated posterior는 분포가 더 평평해지고, top1 probability mean이 감소한다. predictive entropy는 증가한다.

- argmax / accuracy / macro F1은 변하지 않는다 (§3). 본 단계의 효과는 분류 결정이 아니라 *불확실성 표현*에 있다.

---

## 6. 다운스트림 — schema output에는 calibrated posterior를 사용한다

후속 스크립트 `scripts/generate_step4r_attention_schema_outputs.py`는 본 단계가 산출한 `data/step4r/4rb_attention/step4r_bigru_attention_logits_calibrated.npz`의 **`probs_calibrated`** 를 입력으로 사용한다. raw posterior(`probs_raw`)는 비교용으로만 유지하며 schema 결정에 사용되지 않는다. Step 3 §3의 class_posterior 출력에도 calibrated posterior가 들어간다.

---

## 7. 산출물 목록

- `data/step4r/4rb_attention/step4r_bigru_attention_logits_calibrated.npz`
  - 키: `logits_raw`, `logits_calibrated`, `probs_raw`, `probs_calibrated`, `y`, `class_id`, `sample_id`, `participant_id`, `posture_canonical`, `split`, `temperature`
- `data/step4r/4rb_attention/step4r_bigru_attention_predictions_calibrated.csv`
- `reports/step4r/4rb_attention/step4r_temperature_scaling_metrics.csv` (long format: split, stage, metric, value)
- `reports/step4r/4rb_attention/step4r_temperature_scaling_results.md` (본 보고서)

---

*본 보고서는 자동 생성된다. 기존 Step 1 ~ 4 / 4R-A / 4R-B 산출물은 수정되지 않으며, 기존 `step4r_bigru_attention_predictions.csv`는 덮어쓰지 않는다.*
