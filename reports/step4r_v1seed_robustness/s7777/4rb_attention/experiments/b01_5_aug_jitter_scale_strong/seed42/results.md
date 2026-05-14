# Step 4R-B v2 — exp_id=`b01_5_aug_jitter_scale_strong` seed=`42` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 48, best val macroF1: 0.5160
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.5067 |
| macro_f1 | 0.4980 |
| weighted_f1 | 0.4980 |
| log_loss | 2.1209 |
| brier_multiclass | 0.7699 |
| ece_15bin | 0.3115 |
| amb_c2_recall | 0.7353 |
| amb_c1_c5_c6_internal | 0.3643 |
| amb_c3_c4_pair | 0.2406 |
| amb_c3_to_c2_absorb | 0.2185 |
| amb_c4_to_c2_absorb | 0.1167 |
| predictive_entropy_mean | 0.4482 |
| attention_entropy_mean | 2.8922 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.5000 | 0.4916 | 0.4958 | 238 |
| C2 | 0.5319 | 0.7353 | 0.6173 | 238 |
| C3 | 0.5111 | 0.5798 | 0.5433 | 238 |
| C4 | 0.5024 | 0.4375 | 0.4677 | 240 |
| C5 | 0.4459 | 0.2954 | 0.3553 | 237 |
| C6 | 0.5175 | 0.5000 | 0.5086 | 236 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*