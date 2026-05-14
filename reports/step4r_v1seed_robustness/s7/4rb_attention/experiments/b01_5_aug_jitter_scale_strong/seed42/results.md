# Step 4R-B v2 — exp_id=`b01_5_aug_jitter_scale_strong` seed=`42` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 60, best val macroF1: 0.5084
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.4722 |
| macro_f1 | 0.4693 |
| weighted_f1 | 0.4699 |
| log_loss | 2.5776 |
| brier_multiclass | 0.8470 |
| ece_15bin | 0.3791 |
| amb_c2_recall | 0.6917 |
| amb_c1_c5_c6_internal | 0.3654 |
| amb_c3_c4_pair | 0.3707 |
| amb_c3_to_c2_absorb | 0.1483 |
| amb_c4_to_c2_absorb | 0.0175 |
| predictive_entropy_mean | 0.3676 |
| attention_entropy_mean | 2.3719 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.5378 | 0.5333 | 0.5356 | 240 |
| C2 | 0.5993 | 0.6917 | 0.6422 | 240 |
| C3 | 0.3948 | 0.3898 | 0.3923 | 236 |
| C4 | 0.4044 | 0.4825 | 0.4400 | 228 |
| C5 | 0.3348 | 0.3138 | 0.3240 | 239 |
| C6 | 0.5650 | 0.4202 | 0.4819 | 238 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*