# Step 4R-B v2 — exp_id=`b01_aug_jitter_scale` seed=`42` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 36, best val macroF1: 0.5263
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.4940 |
| macro_f1 | 0.4940 |
| weighted_f1 | 0.4940 |
| log_loss | 1.6635 |
| brier_multiclass | 0.7221 |
| ece_15bin | 0.2661 |
| amb_c2_recall | 0.6067 |
| amb_c1_c5_c6_internal | 0.4014 |
| amb_c3_c4_pair | 0.2643 |
| amb_c3_to_c2_absorb | 0.2311 |
| amb_c4_to_c2_absorb | 0.1149 |
| predictive_entropy_mean | 0.5810 |
| attention_entropy_mean | 2.9410 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.5750 | 0.4852 | 0.5263 | 237 |
| C2 | 0.5331 | 0.6067 | 0.5675 | 239 |
| C3 | 0.4920 | 0.5168 | 0.5041 | 238 |
| C4 | 0.4938 | 0.5064 | 0.5000 | 235 |
| C5 | 0.3820 | 0.4286 | 0.4040 | 238 |
| C6 | 0.5127 | 0.4208 | 0.4622 | 240 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*