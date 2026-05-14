# Step 4R-B v2 — exp_id=`b02_aug_plus_dropout_strong` seed=`43` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 23, best val macroF1: 0.5179
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.5102 |
| macro_f1 | 0.4954 |
| weighted_f1 | 0.4953 |
| log_loss | 1.1404 |
| brier_multiclass | 0.6189 |
| ece_15bin | 0.1275 |
| amb_c2_recall | 0.7950 |
| amb_c1_c5_c6_internal | 0.4070 |
| amb_c3_c4_pair | 0.3129 |
| amb_c3_to_c2_absorb | 0.2185 |
| amb_c4_to_c2_absorb | 0.0766 |
| predictive_entropy_mean | 0.8550 |
| attention_entropy_mean | 2.8581 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.5243 | 0.6835 | 0.5934 | 237 |
| C2 | 0.6051 | 0.7950 | 0.6872 | 239 |
| C3 | 0.4886 | 0.4496 | 0.4683 | 238 |
| C4 | 0.4822 | 0.5191 | 0.5000 | 235 |
| C5 | 0.3885 | 0.2563 | 0.3089 | 238 |
| C6 | 0.4914 | 0.3583 | 0.4145 | 240 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*