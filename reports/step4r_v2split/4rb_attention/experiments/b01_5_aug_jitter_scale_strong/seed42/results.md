# Step 4R-B v2 — exp_id=`b01_5_aug_jitter_scale_strong` seed=`42` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 27, best val macroF1: 0.5026
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.4615 |
| macro_f1 | 0.4436 |
| weighted_f1 | 0.4447 |
| log_loss | 1.6657 |
| brier_multiclass | 0.7275 |
| ece_15bin | 0.2451 |
| amb_c2_recall | 0.7055 |
| amb_c1_c5_c6_internal | 0.3556 |
| amb_c3_c4_pair | 0.2567 |
| amb_c3_to_c2_absorb | 0.3200 |
| amb_c4_to_c2_absorb | 0.1316 |
| predictive_entropy_mean | 0.7205 |
| attention_entropy_mean | 2.2093 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.5148 | 0.6619 | 0.5792 | 420 |
| C2 | 0.5351 | 0.7055 | 0.6086 | 421 |
| C3 | 0.4237 | 0.2775 | 0.3353 | 400 |
| C4 | 0.3744 | 0.3708 | 0.3726 | 418 |
| C5 | 0.3780 | 0.2225 | 0.2801 | 418 |
| C6 | 0.4551 | 0.5203 | 0.4855 | 419 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*