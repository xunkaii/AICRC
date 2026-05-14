# Step 4R-B v2 — exp_id=`b00_baseline_3seed` seed=`44` aug=`off` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 16, best val macroF1: 0.5302
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.5207 |
| macro_f1 | 0.5150 |
| weighted_f1 | 0.5149 |
| log_loss | 1.1957 |
| brier_multiclass | 0.6240 |
| ece_15bin | 0.0965 |
| amb_c2_recall | 0.7322 |
| amb_c1_c5_c6_internal | 0.3427 |
| amb_c3_c4_pair | 0.2833 |
| amb_c3_to_c2_absorb | 0.1765 |
| amb_c4_to_c2_absorb | 0.0894 |
| predictive_entropy_mean | 0.9599 |
| attention_entropy_mean | 3.2389 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.6582 | 0.5443 | 0.5958 | 237 |
| C2 | 0.5352 | 0.7322 | 0.6184 | 239 |
| C3 | 0.5035 | 0.5966 | 0.5462 | 238 |
| C4 | 0.5000 | 0.4894 | 0.4946 | 235 |
| C5 | 0.4316 | 0.4244 | 0.4280 | 238 |
| C6 | 0.5127 | 0.3375 | 0.4070 | 240 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*