# Step 4R-B v2 — exp_id=`b02_aug_plus_dropout_strong` seed=`44` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 37, best val macroF1: 0.5390
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.5347 |
| macro_f1 | 0.5227 |
| weighted_f1 | 0.5227 |
| log_loss | 1.4477 |
| brier_multiclass | 0.6501 |
| ece_15bin | 0.1707 |
| amb_c2_recall | 0.7280 |
| amb_c1_c5_c6_internal | 0.3734 |
| amb_c3_c4_pair | 0.2516 |
| amb_c3_to_c2_absorb | 0.2059 |
| amb_c4_to_c2_absorb | 0.1021 |
| predictive_entropy_mean | 0.6846 |
| attention_entropy_mean | 2.6905 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.5377 | 0.6920 | 0.6052 | 237 |
| C2 | 0.6105 | 0.7280 | 0.6641 | 239 |
| C3 | 0.5322 | 0.6597 | 0.5891 | 238 |
| C4 | 0.4922 | 0.4043 | 0.4439 | 235 |
| C5 | 0.4850 | 0.4076 | 0.4429 | 238 |
| C6 | 0.5101 | 0.3167 | 0.3907 | 240 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*