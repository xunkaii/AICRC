# Step 4R-B — BiGRU + Attention 학습 결과 보고서

- 생성 스크립트: `scripts/train_step4r_attention_rnn.py`
- 모델 정의: `models/step4r_attention_rnn.py` (`Step4RBiGRUAttention`)
- 입력: `data/step4r/4rb_attention/step4r_sequence_dataset.npz` (X_norm 사용)
- 학습 device: `cuda`
- best checkpoint: `checkpoints/step4r/4rb_attention/best_bigru_attention.pt` (epoch 19)

---

## 1. 모델 설정

| 항목 | 값 |
|---|---|
| input shape | (B, 6, 128) → 내부 (B, 128, 6) transpose |
| BiGRU hidden_size | 64 |
| BiGRU num_layers | 2 |
| BiGRU bidirectional | True (output dim = 128) |
| BiGRU inter-layer dropout | 0.3 |
| attention | Luong-style multiplicative, last-state query |
| attention vector dim | 128 (= hidden·2) |
| posture conditioning | one-hot 3-dim (SA/CA/HW), attention vector 뒤 concat |
| classifier head | Linear(131→128) → ReLU → Dropout → Linear(128→6) |
| classifier dropout | 0.3 |
| optimizer | AdamW (lr=0.001, weight_decay=0.0001) |
| loss | CrossEntropyLoss |
| batch_size | 64 |
| max_epochs | 100 |
| early stopping | val macro F1, patience=15 |
| seed | 42 |

forward는 `(logits, attention_weights)`를 반환하며, attention_weights shape은 (B, 128).

---

## 2. train / val / test 성능

| split | n | accuracy | balanced_acc | macro F1 | weighted F1 | log loss | Brier | ECE (15-bin) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 6412 | 0.7110 | 0.7106 | 0.7091 | 0.7095 | 0.6888 | 0.3848 | 0.0326 |
| val | 1436 | 0.5216 | 0.5215 | 0.5174 | 0.5176 | 1.3174 | 0.6352 | 0.1428 |
| test | 1427 | 0.5284 | 0.5285 | 0.5184 | 0.5184 | 1.2286 | 0.6258 | 0.1340 |

### 2.1 per-class precision / recall / F1 (test split)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.6015 | 0.6751 | 0.6362 | 237 |
| C2 | 0.5805 | 0.7238 | 0.6443 | 239 |
| C3 | 0.5127 | 0.4244 | 0.4644 | 238 |
| C4 | 0.5076 | 0.5660 | 0.5352 | 235 |
| C5 | 0.4088 | 0.2731 | 0.3275 | 238 |
| C6 | 0.4980 | 0.5083 | 0.5031 | 240 |

---

## 3. test 성능 비교 — LR baseline / 4R-A HGB ceiling / 4R-B

LR baseline 수치는 `reports/step4/step4_final_summary.md`에서 인용. HGB ceiling 수치는 `reports/step4r/4ra_feature_ceiling/step4r_feature_ceiling_metrics.csv`에서 자동 로드. 둘 다 (raw, zscore) 두 branch 중 raw branch만 표시한다 (zscore와 거의 동일 — §3의 zscore 비교는 metrics CSV에서 직접 확인 가능).

| 지표 | LR raw | HGB raw (4R-A) | BiGRU+Attn (4R-B) |
|---|---:|---:|---:|
| accuracy | 0.3210 | 0.3062 | 0.5284 |
| macro F1 | 0.2843 | 0.3004 | 0.5184 |
| log loss | 1.6173 | 1.8710 | 1.2286 |
| Brier (multi) | 0.7746 | 0.8659 | 0.6258 |
| ECE (15-bin) | 0.0320 | 0.2300 | 0.1340 |
| C2 recall | 0.8159 | 0.5565 | 0.7238 |
| C1/C5/C6 internal | 0.2462 | 0.4336 | 0.3622 |
| C3/C4 pair | 0.2199 | 0.1945 | 0.2770 |
| C3 → C2 absorb | 0.3277 | 0.2773 | 0.2395 |
| C4 → C2 absorb | 0.2723 | 0.1915 | 0.0809 |

---

## 4. Calibration

| split | log loss | Brier | ECE (15-bin) | top1 mean | top1 p25 | top1 p50 | top1 p75 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 0.6888 | 0.3848 | 0.0326 | 0.6785 | 0.5340 | 0.6647 | 0.8329 |
| val | 1.3174 | 0.6352 | 0.1428 | 0.6629 | 0.5098 | 0.6316 | 0.8290 |
| test | 1.2286 | 0.6258 | 0.1340 | 0.6527 | 0.5101 | 0.6355 | 0.8060 |

---

## 5. Predictive entropy / Attention entropy 요약

predictive entropy = −Σ p log p (자연로그). 6-class 균등분포 상한 ≈ 1.7918.
attention entropy = 시계열 attention weight 분포의 entropy. T=128 균등분포 상한 ≈ 4.8520.

| split | pred_ent mean | pred_ent p25 | pred_ent p50 | pred_ent p75 | attn_ent mean | attn_ent p25 | attn_ent p50 | attn_ent p75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 0.7544 | 0.5314 | 0.7442 | 0.9910 | 3.1009 | 2.6879 | 3.0724 | 3.4872 |
| val | 0.7769 | 0.5490 | 0.7765 | 1.0479 | 3.1907 | 2.8373 | 3.1592 | 3.5075 |
| test | 0.8164 | 0.5786 | 0.8089 | 1.0697 | 3.2496 | 2.9044 | 3.2207 | 3.5791 |

---

## 6. Ambiguity group 분석 (test split)

| 지표 | 값 |
|---|---:|
| C2 recall | 0.7238 |
| C1/C5/C6 internal confusion | 0.3622 |
| C3/C4 pair confusion | 0.2770 |
| C3 → C2 absorption | 0.2395 |
| C4 → C2 absorption | 0.0809 |

Step 2.5 종합 (`reports/step2/step25_final_synthesis.md`)에서 도출된 ambiguity 패턴(C2 단언 가능, C1/C5/C6 그룹 모호, C3/C4의 C2 흡수)이 본 모델에서도 보존되는지를 점검한다.

---

## 7. 과적합 여부

| 비교 | train | val | test | train−val gap | train−test gap |
|---|---:|---:|---:|---:|---:|
| accuracy | 0.7110 | 0.5216 | 0.5284 | +0.1894 | +0.1826 |
| macro_f1 | 0.7091 | 0.5174 | 0.5184 | +0.1917 | +0.1907 |
| ece_15bin | 0.0326 | 0.1428 | 0.1340 | -0.1103 | -0.1014 |
| log_loss | 0.6888 | 1.3174 | 1.2286 | -0.6286 | -0.5398 |

train−val / train−test gap이 크면 과적합. early stopping 기준이 val macro F1이므로 best epoch에서의 train−val gap이 본 학습이 멈춘 시점의 generalization 신호이다. checkpoint epoch / training history는 `step4r_bigru_attention_training_log.csv`에 저장된다.

---

## 8. 다음 단계 제안

본 결과는 reframing 문서 §5.5 모델 채택 기준(A/B/C)의 어느 분기에 해당하는지 결정하는 입력이다. 일반적으로 다음 중 한 방향으로 이어진다:

1. **case A** — 분류 또는 calibration이 4R-A 대비 명확히 개선됨 → 4R-B를 main pipeline 모델로 승격하고, 다음 단계는 schema output threshold 재calibration. Step 5~7 (schema-grounded LLM caption layer)는 그대로 유지.
2. **case B** — 분류 점수는 4R-A와 비슷하지만 schema behavior 또는 근거 분석 (predictive/attention entropy + ambiguity group 일치)이 의미 있게 작동 → 4R-B를 채택하되, 논문 contribution을 분류 성능이 아니라 *uncertainty-aware 설명* 쪽으로 강조.
3. **case C** — 분류·calibration·근거 분석 모두 4R-A와 동등하거나 약함 → 4R-A(HGB)를 main pipeline 모델로 두고, 4R-B 결과는 *raw sequence DL의 데이터-규모 한계*로 ablation 챕터에 보고. 본 reframing 문서 §11에 따라 후속 의사결정 기록 필요.

본 결과는 단일 시드, 단일 hyperparameter 점만 산출했다. seed sensitivity나 augmentation ablation은 본 단계의 범위 밖이다.

---

## 9. 산출물 목록

- `data/step4r/4rb_attention/step4r_bigru_attention_predictions.csv`
- `reports/step4r/4rb_attention/step4r_bigru_attention_metrics.csv` (long format)
- `reports/step4r/4rb_attention/step4r_bigru_attention_confusion.csv` (long format)
- `reports/step4r/4rb_attention/step4r_bigru_attention_training_log.csv`
- `reports/step4r/4rb_attention/step4r_bigru_attention_results.md` (본 보고서)
- `checkpoints/step4r/4rb_attention/best_bigru_attention.pt`

---

*본 보고서는 `scripts/train_step4r_attention_rnn.py` 실행 시 자동 생성된다. 기존 Step 1 ~ 4 / 4R-A 산출물은 수정되지 않는다.*
