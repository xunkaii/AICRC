# Step 4R-B v2 — exp_id=`b01_5_aug_jitter_scale_strong` seed=`42` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 20, best val macroF1: 0.5154
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.5270 |
| macro_f1 | 0.5190 |
| weighted_f1 | 0.5193 |
| log_loss | 1.1472 |
| brier_multiclass | 0.6026 |
| ece_15bin | 0.1072 |
| amb_c2_recall | 0.7059 |
| amb_c1_c5_c6_internal | 0.3797 |
| amb_c3_c4_pair | 0.2836 |
| amb_c3_to_c2_absorb | 0.2405 |
| amb_c4_to_c2_absorb | 0.1423 |
| predictive_entropy_mean | 0.8585 |
| attention_entropy_mean | 3.3559 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.5795 | 0.7322 | 0.6470 | 239 |
| C2 | 0.6000 | 0.7059 | 0.6486 | 238 |
| C3 | 0.4556 | 0.3249 | 0.3793 | 237 |
| C4 | 0.4872 | 0.4770 | 0.4820 | 239 |
| C5 | 0.4198 | 0.4322 | 0.4259 | 236 |
| C6 | 0.5838 | 0.4873 | 0.5312 | 236 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*