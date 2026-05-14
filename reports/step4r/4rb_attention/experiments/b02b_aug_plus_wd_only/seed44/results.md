# Step 4R-B v2 — exp_id=`b02b_aug_plus_wd_only` seed=`44` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 18, best val macroF1: 0.5470
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.5669 |
| macro_f1 | 0.5600 |
| weighted_f1 | 0.5599 |
| log_loss | 1.0909 |
| brier_multiclass | 0.5658 |
| ece_15bin | 0.0553 |
| amb_c2_recall | 0.7615 |
| amb_c1_c5_c6_internal | 0.3636 |
| amb_c3_c4_pair | 0.2410 |
| amb_c3_to_c2_absorb | 0.2185 |
| amb_c4_to_c2_absorb | 0.0638 |
| predictive_entropy_mean | 0.8970 |
| attention_entropy_mean | 2.7766 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.5965 | 0.7173 | 0.6513 | 237 |
| C2 | 0.6594 | 0.7615 | 0.7068 | 239 |
| C3 | 0.6091 | 0.5042 | 0.5517 | 238 |
| C4 | 0.5217 | 0.6128 | 0.5636 | 235 |
| C5 | 0.4570 | 0.4244 | 0.4401 | 238 |
| C6 | 0.5349 | 0.3833 | 0.4466 | 240 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*