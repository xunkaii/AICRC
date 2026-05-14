# Step 4R-B v2 — exp_id=`b01_5_aug_jitter_scale_strong` seed=`43` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 26, best val macroF1: 0.5510
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.5228 |
| macro_f1 | 0.5148 |
| weighted_f1 | 0.5151 |
| log_loss | 1.2404 |
| brier_multiclass | 0.6171 |
| ece_15bin | 0.1467 |
| amb_c2_recall | 0.7059 |
| amb_c1_c5_c6_internal | 0.3797 |
| amb_c3_c4_pair | 0.3214 |
| amb_c3_to_c2_absorb | 0.1941 |
| amb_c4_to_c2_absorb | 0.1172 |
| predictive_entropy_mean | 0.7617 |
| attention_entropy_mean | 2.6481 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.5704 | 0.6778 | 0.6195 | 239 |
| C2 | 0.6316 | 0.7059 | 0.6667 | 238 |
| C3 | 0.4179 | 0.3544 | 0.3836 | 237 |
| C4 | 0.4769 | 0.5188 | 0.4970 | 239 |
| C5 | 0.4675 | 0.3347 | 0.3901 | 236 |
| C6 | 0.5224 | 0.5424 | 0.5322 | 236 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*