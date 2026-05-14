# Step 4R-B v2 — exp_id=`b01_5_aug_jitter_scale_strong` seed=`42` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 24, best val macroF1: 0.5169
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.5214 |
| macro_f1 | 0.5134 |
| weighted_f1 | 0.5134 |
| log_loss | 1.2144 |
| brier_multiclass | 0.6281 |
| ece_15bin | 0.1415 |
| amb_c2_recall | 0.6904 |
| amb_c1_c5_c6_internal | 0.3832 |
| amb_c3_c4_pair | 0.2643 |
| amb_c3_to_c2_absorb | 0.2143 |
| amb_c4_to_c2_absorb | 0.1234 |
| predictive_entropy_mean | 0.8024 |
| attention_entropy_mean | 2.6790 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.5268 | 0.6624 | 0.5869 | 237 |
| C2 | 0.6000 | 0.6904 | 0.6420 | 239 |
| C3 | 0.5106 | 0.4034 | 0.4507 | 238 |
| C4 | 0.4894 | 0.5915 | 0.5356 | 235 |
| C5 | 0.4111 | 0.3109 | 0.3541 | 238 |
| C6 | 0.5594 | 0.4708 | 0.5113 | 240 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*