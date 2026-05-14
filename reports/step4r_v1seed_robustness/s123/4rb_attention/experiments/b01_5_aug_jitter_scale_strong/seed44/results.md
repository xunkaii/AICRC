# Step 4R-B v2 — exp_id=`b01_5_aug_jitter_scale_strong` seed=`44` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 15, best val macroF1: 0.5143
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.4856 |
| macro_f1 | 0.4773 |
| weighted_f1 | 0.4772 |
| log_loss | 1.2612 |
| brier_multiclass | 0.6511 |
| ece_15bin | 0.0799 |
| amb_c2_recall | 0.6835 |
| amb_c1_c5_c6_internal | 0.3980 |
| amb_c3_c4_pair | 0.2595 |
| amb_c3_to_c2_absorb | 0.2405 |
| amb_c4_to_c2_absorb | 0.0506 |
| predictive_entropy_mean | 1.0184 |
| attention_entropy_mean | 2.6399 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.5389 | 0.3750 | 0.4423 | 240 |
| C2 | 0.5978 | 0.6835 | 0.6378 | 237 |
| C3 | 0.4852 | 0.3460 | 0.4039 | 237 |
| C4 | 0.4451 | 0.6498 | 0.5283 | 237 |
| C5 | 0.4247 | 0.3319 | 0.3726 | 238 |
| C6 | 0.4375 | 0.5294 | 0.4791 | 238 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*