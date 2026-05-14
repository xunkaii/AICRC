# Step 4R-B v2 — exp_id=`b01_5_aug_jitter_scale_strong` seed=`44` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 14, best val macroF1: 0.5377
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.5074 |
| macro_f1 | 0.4930 |
| weighted_f1 | 0.4933 |
| log_loss | 1.1285 |
| brier_multiclass | 0.6051 |
| ece_15bin | 0.0706 |
| amb_c2_recall | 0.7395 |
| amb_c1_c5_c6_internal | 0.4304 |
| amb_c3_c4_pair | 0.2752 |
| amb_c3_to_c2_absorb | 0.2616 |
| amb_c4_to_c2_absorb | 0.1172 |
| predictive_entropy_mean | 1.0084 |
| attention_entropy_mean | 2.5001 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.4739 | 0.7992 | 0.5950 | 239 |
| C2 | 0.6090 | 0.7395 | 0.6679 | 238 |
| C3 | 0.4688 | 0.3797 | 0.4196 | 237 |
| C4 | 0.5319 | 0.4184 | 0.4684 | 239 |
| C5 | 0.4192 | 0.3517 | 0.3825 | 236 |
| C6 | 0.5355 | 0.3517 | 0.4246 | 236 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*