# Step 4R-B v2 — exp_id=`b02b_aug_plus_wd_only` seed=`43` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 25, best val macroF1: 0.5103
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.5102 |
| macro_f1 | 0.4968 |
| weighted_f1 | 0.4967 |
| log_loss | 1.2797 |
| brier_multiclass | 0.6473 |
| ece_15bin | 0.1465 |
| amb_c2_recall | 0.7992 |
| amb_c1_c5_c6_internal | 0.4154 |
| amb_c3_c4_pair | 0.2685 |
| amb_c3_to_c2_absorb | 0.3025 |
| amb_c4_to_c2_absorb | 0.1404 |
| predictive_entropy_mean | 0.8069 |
| attention_entropy_mean | 2.8356 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.5101 | 0.6414 | 0.5682 | 237 |
| C2 | 0.5618 | 0.7992 | 0.6598 | 239 |
| C3 | 0.5065 | 0.3277 | 0.3980 | 238 |
| C4 | 0.5211 | 0.5787 | 0.5484 | 235 |
| C5 | 0.3865 | 0.3361 | 0.3596 | 238 |
| C6 | 0.5449 | 0.3792 | 0.4472 | 240 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*