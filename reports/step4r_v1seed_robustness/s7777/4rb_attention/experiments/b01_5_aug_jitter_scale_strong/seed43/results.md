# Step 4R-B v2 — exp_id=`b01_5_aug_jitter_scale_strong` seed=`43` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 49, best val macroF1: 0.4998
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.4989 |
| macro_f1 | 0.4875 |
| weighted_f1 | 0.4875 |
| log_loss | 2.0858 |
| brier_multiclass | 0.7729 |
| ece_15bin | 0.3274 |
| amb_c2_recall | 0.7437 |
| amb_c1_c5_c6_internal | 0.3727 |
| amb_c3_c4_pair | 0.3054 |
| amb_c3_to_c2_absorb | 0.2059 |
| amb_c4_to_c2_absorb | 0.1000 |
| predictive_entropy_mean | 0.4346 |
| attention_entropy_mean | 2.7404 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.5000 | 0.6008 | 0.5458 | 238 |
| C2 | 0.5880 | 0.7437 | 0.6568 | 238 |
| C3 | 0.4570 | 0.5588 | 0.5028 | 238 |
| C4 | 0.4375 | 0.3500 | 0.3889 | 240 |
| C5 | 0.4384 | 0.2700 | 0.3342 | 237 |
| C6 | 0.5261 | 0.4703 | 0.4966 | 236 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*