# Step 4R-B v2 — exp_id=`b02_aug_plus_dropout_strong` seed=`42` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 12, best val macroF1: 0.5207
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.5102 |
| macro_f1 | 0.4958 |
| weighted_f1 | 0.4956 |
| log_loss | 1.1227 |
| brier_multiclass | 0.5960 |
| ece_15bin | 0.0383 |
| amb_c2_recall | 0.7950 |
| amb_c1_c5_c6_internal | 0.3762 |
| amb_c3_c4_pair | 0.2727 |
| amb_c3_to_c2_absorb | 0.2689 |
| amb_c4_to_c2_absorb | 0.0979 |
| predictive_entropy_mean | 1.1559 |
| attention_entropy_mean | 3.4423 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.5652 | 0.6034 | 0.5837 | 237 |
| C2 | 0.5429 | 0.7950 | 0.6452 | 239 |
| C3 | 0.5468 | 0.3193 | 0.4032 | 238 |
| C4 | 0.4903 | 0.6426 | 0.5562 | 235 |
| C5 | 0.4192 | 0.3487 | 0.3807 | 238 |
| C6 | 0.4749 | 0.3542 | 0.4057 | 240 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*