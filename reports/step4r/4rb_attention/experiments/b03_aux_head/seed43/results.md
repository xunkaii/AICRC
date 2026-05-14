# Step 4R-B v2 — exp_id=`b03_aux_head` seed=`43` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 34, best val macroF1: 0.5199
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.5018 |
| macro_f1 | 0.4967 |
| weighted_f1 | 0.4966 |
| log_loss | 1.3364 |
| brier_multiclass | 0.6614 |
| ece_15bin | 0.1907 |
| amb_c2_recall | 0.6946 |
| amb_c1_c5_c6_internal | 0.4531 |
| amb_c3_c4_pair | 0.2537 |
| amb_c3_to_c2_absorb | 0.2731 |
| amb_c4_to_c2_absorb | 0.1319 |
| predictive_entropy_mean | 0.7124 |
| attention_entropy_mean | 2.4029 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.5000 | 0.6329 | 0.5587 | 237 |
| C2 | 0.5866 | 0.6946 | 0.6360 | 239 |
| C3 | 0.4977 | 0.4454 | 0.4701 | 238 |
| C4 | 0.5399 | 0.4894 | 0.5134 | 235 |
| C5 | 0.3755 | 0.3992 | 0.3870 | 238 |
| C6 | 0.5091 | 0.3500 | 0.4148 | 240 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*