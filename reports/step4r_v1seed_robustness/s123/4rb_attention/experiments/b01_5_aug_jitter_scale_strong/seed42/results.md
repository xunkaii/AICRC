# Step 4R-B v2 — exp_id=`b01_5_aug_jitter_scale_strong` seed=`42` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 34, best val macroF1: 0.5162
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.5340 |
| macro_f1 | 0.5323 |
| weighted_f1 | 0.5322 |
| log_loss | 1.4148 |
| brier_multiclass | 0.6735 |
| ece_15bin | 0.2122 |
| amb_c2_recall | 0.7046 |
| amb_c1_c5_c6_internal | 0.3994 |
| amb_c3_c4_pair | 0.3101 |
| amb_c3_to_c2_absorb | 0.1519 |
| amb_c4_to_c2_absorb | 0.0380 |
| predictive_entropy_mean | 0.6228 |
| attention_entropy_mean | 2.4801 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.5299 | 0.5542 | 0.5418 | 240 |
| C2 | 0.6789 | 0.7046 | 0.6915 | 237 |
| C3 | 0.5158 | 0.4135 | 0.4590 | 237 |
| C4 | 0.4861 | 0.5907 | 0.5333 | 237 |
| C5 | 0.4484 | 0.4748 | 0.4612 | 238 |
| C6 | 0.5550 | 0.4664 | 0.5068 | 238 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*