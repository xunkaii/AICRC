# Step 4R-B v2 — exp_id=`b00_baseline_3seed` seed=`43` aug=`off` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 33, best val macroF1: 0.5097
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.4730 |
| macro_f1 | 0.4720 |
| weighted_f1 | 0.4719 |
| log_loss | 1.6345 |
| brier_multiclass | 0.7377 |
| ece_15bin | 0.2727 |
| amb_c2_recall | 0.6109 |
| amb_c1_c5_c6_internal | 0.4420 |
| amb_c3_c4_pair | 0.2748 |
| amb_c3_to_c2_absorb | 0.2605 |
| amb_c4_to_c2_absorb | 0.1787 |
| predictive_entropy_mean | 0.5982 |
| attention_entropy_mean | 2.6673 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.5408 | 0.5316 | 0.5362 | 237 |
| C2 | 0.5233 | 0.6109 | 0.5637 | 239 |
| C3 | 0.4454 | 0.4286 | 0.4368 | 238 |
| C4 | 0.4886 | 0.4553 | 0.4714 | 235 |
| C5 | 0.3675 | 0.4370 | 0.3992 | 238 |
| C6 | 0.4891 | 0.3750 | 0.4245 | 240 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*