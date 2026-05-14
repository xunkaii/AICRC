# Step 4R-B v2 — exp_id=`b01_5_aug_jitter_scale_strong` seed=`43` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 32, best val macroF1: 0.5246
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.5095 |
| macro_f1 | 0.5071 |
| weighted_f1 | 0.5071 |
| log_loss | 1.4178 |
| brier_multiclass | 0.6816 |
| ece_15bin | 0.2251 |
| amb_c2_recall | 0.6456 |
| amb_c1_c5_c6_internal | 0.4148 |
| amb_c3_c4_pair | 0.3165 |
| amb_c3_to_c2_absorb | 0.1308 |
| amb_c4_to_c2_absorb | 0.0549 |
| predictive_entropy_mean | 0.6357 |
| attention_entropy_mean | 2.6082 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.4832 | 0.6000 | 0.5353 | 240 |
| C2 | 0.7391 | 0.6456 | 0.6892 | 237 |
| C3 | 0.5065 | 0.3291 | 0.3990 | 237 |
| C4 | 0.4306 | 0.6287 | 0.5111 | 237 |
| C5 | 0.4223 | 0.4454 | 0.4335 | 238 |
| C6 | 0.5673 | 0.4076 | 0.4743 | 238 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*