# Step 4R-B v2 — exp_id=`b01_aug_jitter_scale` seed=`43` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 23, best val macroF1: 0.5168
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.4877 |
| macro_f1 | 0.4778 |
| weighted_f1 | 0.4778 |
| log_loss | 1.3139 |
| brier_multiclass | 0.6666 |
| ece_15bin | 0.1911 |
| amb_c2_recall | 0.7113 |
| amb_c1_c5_c6_internal | 0.4350 |
| amb_c3_c4_pair | 0.2833 |
| amb_c3_to_c2_absorb | 0.2479 |
| amb_c4_to_c2_absorb | 0.1447 |
| predictive_entropy_mean | 0.7523 |
| attention_entropy_mean | 2.9399 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.4816 | 0.6624 | 0.5577 | 237 |
| C2 | 0.5705 | 0.7113 | 0.6331 | 239 |
| C3 | 0.4883 | 0.4370 | 0.4612 | 238 |
| C4 | 0.4800 | 0.4596 | 0.4696 | 235 |
| C5 | 0.3763 | 0.2941 | 0.3302 | 238 |
| C6 | 0.4860 | 0.3625 | 0.4153 | 240 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*