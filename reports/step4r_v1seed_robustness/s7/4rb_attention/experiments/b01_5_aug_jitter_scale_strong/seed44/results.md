# Step 4R-B v2 — exp_id=`b01_5_aug_jitter_scale_strong` seed=`44` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 34, best val macroF1: 0.5105
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.5004 |
| macro_f1 | 0.5002 |
| weighted_f1 | 0.5009 |
| log_loss | 1.7767 |
| brier_multiclass | 0.7283 |
| ece_15bin | 0.2586 |
| amb_c2_recall | 0.6625 |
| amb_c1_c5_c6_internal | 0.3598 |
| amb_c3_c4_pair | 0.3556 |
| amb_c3_to_c2_absorb | 0.1525 |
| amb_c4_to_c2_absorb | 0.0132 |
| predictive_entropy_mean | 0.5703 |
| attention_entropy_mean | 2.6571 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.5458 | 0.5708 | 0.5580 | 240 |
| C2 | 0.6681 | 0.6625 | 0.6653 | 240 |
| C3 | 0.4453 | 0.5000 | 0.4711 | 236 |
| C4 | 0.3978 | 0.4693 | 0.4306 | 228 |
| C5 | 0.3983 | 0.4017 | 0.4000 | 239 |
| C6 | 0.5987 | 0.3950 | 0.4759 | 238 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*