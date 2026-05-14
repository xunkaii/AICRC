# Step 4R-B v2 — exp_id=`b01_5_aug_jitter_scale_strong` seed=`43` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 24, best val macroF1: 0.5070
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.5081 |
| macro_f1 | 0.5045 |
| weighted_f1 | 0.5044 |
| log_loss | 1.1922 |
| brier_multiclass | 0.6261 |
| ece_15bin | 0.1269 |
| amb_c2_recall | 0.7238 |
| amb_c1_c5_c6_internal | 0.4224 |
| amb_c3_c4_pair | 0.2770 |
| amb_c3_to_c2_absorb | 0.2479 |
| amb_c4_to_c2_absorb | 0.1191 |
| predictive_entropy_mean | 0.8391 |
| attention_entropy_mean | 2.8110 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.5792 | 0.5401 | 0.5590 | 237 |
| C2 | 0.5786 | 0.7238 | 0.6431 | 239 |
| C3 | 0.5075 | 0.4244 | 0.4622 | 238 |
| C4 | 0.4906 | 0.5574 | 0.5219 | 235 |
| C5 | 0.3727 | 0.4244 | 0.3969 | 238 |
| C6 | 0.5353 | 0.3792 | 0.4439 | 240 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*