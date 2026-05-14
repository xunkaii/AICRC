# Step 4R-B v2 — exp_id=`b01_5_aug_jitter_scale_strong` seed=`43` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 26, best val macroF1: 0.5066
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.4659 |
| macro_f1 | 0.4661 |
| weighted_f1 | 0.4667 |
| log_loss | 1.6496 |
| brier_multiclass | 0.7419 |
| ece_15bin | 0.2597 |
| amb_c2_recall | 0.6292 |
| amb_c1_c5_c6_internal | 0.3417 |
| amb_c3_c4_pair | 0.3621 |
| amb_c3_to_c2_absorb | 0.1229 |
| amb_c4_to_c2_absorb | 0.0088 |
| predictive_entropy_mean | 0.6451 |
| attention_entropy_mean | 2.6059 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.5812 | 0.4625 | 0.5151 | 240 |
| C2 | 0.5808 | 0.6292 | 0.6040 | 240 |
| C3 | 0.3962 | 0.4449 | 0.4192 | 236 |
| C4 | 0.3768 | 0.4561 | 0.4127 | 228 |
| C5 | 0.3500 | 0.3222 | 0.3355 | 239 |
| C6 | 0.5455 | 0.4790 | 0.5101 | 238 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*