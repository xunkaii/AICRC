# Step 4R-B v2 — exp_id=`b02b_aug_plus_wd_only` seed=`42` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 25, best val macroF1: 0.5353
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.4919 |
| macro_f1 | 0.4851 |
| weighted_f1 | 0.4850 |
| log_loss | 1.3476 |
| brier_multiclass | 0.6696 |
| ece_15bin | 0.1909 |
| amb_c2_recall | 0.7238 |
| amb_c1_c5_c6_internal | 0.4042 |
| amb_c3_c4_pair | 0.2474 |
| amb_c3_to_c2_absorb | 0.2899 |
| amb_c4_to_c2_absorb | 0.1362 |
| predictive_entropy_mean | 0.7554 |
| attention_entropy_mean | 2.4643 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.5312 | 0.5021 | 0.5163 | 237 |
| C2 | 0.5440 | 0.7238 | 0.6212 | 239 |
| C3 | 0.5256 | 0.3445 | 0.4162 | 238 |
| C4 | 0.4730 | 0.5957 | 0.5273 | 235 |
| C5 | 0.3652 | 0.4496 | 0.4030 | 238 |
| C6 | 0.5786 | 0.3375 | 0.4263 | 240 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*