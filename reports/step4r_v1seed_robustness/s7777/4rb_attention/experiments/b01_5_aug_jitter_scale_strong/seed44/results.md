# Step 4R-B v2 — exp_id=`b01_5_aug_jitter_scale_strong` seed=`44` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 32, best val macroF1: 0.5052
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.4779 |
| macro_f1 | 0.4704 |
| weighted_f1 | 0.4703 |
| log_loss | 1.7731 |
| brier_multiclass | 0.7503 |
| ece_15bin | 0.2701 |
| amb_c2_recall | 0.6471 |
| amb_c1_c5_c6_internal | 0.3741 |
| amb_c3_c4_pair | 0.2803 |
| amb_c3_to_c2_absorb | 0.2395 |
| amb_c4_to_c2_absorb | 0.0917 |
| predictive_entropy_mean | 0.6187 |
| attention_entropy_mean | 2.6967 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.4601 | 0.6303 | 0.5319 | 238 |
| C2 | 0.5500 | 0.6471 | 0.5946 | 238 |
| C3 | 0.4317 | 0.5042 | 0.4651 | 238 |
| C4 | 0.4410 | 0.3583 | 0.3954 | 240 |
| C5 | 0.4157 | 0.2911 | 0.3424 | 237 |
| C6 | 0.5659 | 0.4364 | 0.4928 | 236 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*