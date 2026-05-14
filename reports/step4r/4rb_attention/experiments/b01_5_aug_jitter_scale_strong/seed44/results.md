# Step 4R-B v2 — exp_id=`b01_5_aug_jitter_scale_strong` seed=`44` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 30, best val macroF1: 0.5521
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.5284 |
| macro_f1 | 0.5206 |
| weighted_f1 | 0.5206 |
| log_loss | 1.2755 |
| brier_multiclass | 0.6281 |
| ece_15bin | 0.1574 |
| amb_c2_recall | 0.7364 |
| amb_c1_c5_c6_internal | 0.3762 |
| amb_c3_c4_pair | 0.2537 |
| amb_c3_to_c2_absorb | 0.2143 |
| amb_c4_to_c2_absorb | 0.1191 |
| predictive_entropy_mean | 0.7354 |
| attention_entropy_mean | 2.5843 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.5667 | 0.6456 | 0.6036 | 237 |
| C2 | 0.6007 | 0.7364 | 0.6617 | 239 |
| C3 | 0.5287 | 0.5798 | 0.5531 | 238 |
| C4 | 0.4748 | 0.4809 | 0.4778 | 235 |
| C5 | 0.4053 | 0.3235 | 0.3598 | 238 |
| C6 | 0.5543 | 0.4042 | 0.4675 | 240 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*