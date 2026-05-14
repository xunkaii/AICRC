# Step 4R-B v2 — exp_id=`b01_aug_jitter_scale` seed=`44` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 14, best val macroF1: 0.5442
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.5494 |
| macro_f1 | 0.5403 |
| weighted_f1 | 0.5401 |
| log_loss | 1.0982 |
| brier_multiclass | 0.5867 |
| ece_15bin | 0.0512 |
| amb_c2_recall | 0.7573 |
| amb_c1_c5_c6_internal | 0.3510 |
| amb_c3_c4_pair | 0.2347 |
| amb_c3_to_c2_absorb | 0.2521 |
| amb_c4_to_c2_absorb | 0.0766 |
| predictive_entropy_mean | 1.0261 |
| attention_entropy_mean | 3.1959 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.6126 | 0.6540 | 0.6327 | 237 |
| C2 | 0.5710 | 0.7573 | 0.6511 | 239 |
| C3 | 0.5498 | 0.4874 | 0.5167 | 238 |
| C4 | 0.5435 | 0.6383 | 0.5871 | 235 |
| C5 | 0.4524 | 0.3193 | 0.3744 | 238 |
| C6 | 0.5248 | 0.4417 | 0.4796 | 240 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*