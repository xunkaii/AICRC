# Step 4R-B v2 — exp_id=`b00_baseline_3seed` seed=`42` aug=`off` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 19, best val macroF1: 0.5174
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.5284 |
| macro_f1 | 0.5184 |
| weighted_f1 | 0.5184 |
| log_loss | 1.2286 |
| brier_multiclass | 0.6258 |
| ece_15bin | 0.1340 |
| amb_c2_recall | 0.7238 |
| amb_c1_c5_c6_internal | 0.3622 |
| amb_c3_c4_pair | 0.2770 |
| amb_c3_to_c2_absorb | 0.2395 |
| amb_c4_to_c2_absorb | 0.0809 |
| predictive_entropy_mean | 0.8164 |
| attention_entropy_mean | 3.2496 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.6015 | 0.6751 | 0.6362 | 237 |
| C2 | 0.5805 | 0.7238 | 0.6443 | 239 |
| C3 | 0.5127 | 0.4244 | 0.4644 | 238 |
| C4 | 0.5076 | 0.5660 | 0.5352 | 235 |
| C5 | 0.4088 | 0.2731 | 0.3275 | 238 |
| C6 | 0.4980 | 0.5083 | 0.5031 | 240 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*