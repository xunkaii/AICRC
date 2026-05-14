# Step 4R-B v2 — exp_id=`b03_aux_head` seed=`42` aug=`jitter_scale` 결과

- 생성 스크립트: `scripts/train_step4r_attention_rnn_v2.py`
- best epoch: 37, best val macroF1: 0.5553
- device: cuda

## test 분류·calibration

| 지표 | 값 |
|---|---:|
| accuracy | 0.5718 |
| macro_f1 | 0.5682 |
| weighted_f1 | 0.5683 |
| log_loss | 1.2787 |
| brier_multiclass | 0.6016 |
| ece_15bin | 0.1477 |
| amb_c2_recall | 0.7782 |
| amb_c1_c5_c6_internal | 0.3762 |
| amb_c3_c4_pair | 0.2558 |
| amb_c3_to_c2_absorb | 0.2143 |
| amb_c4_to_c2_absorb | 0.0681 |
| predictive_entropy_mean | 0.6579 |
| attention_entropy_mean | 3.0611 |

## per-class F1 (test)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| C1 | 0.6233 | 0.5865 | 0.6043 | 237 |
| C2 | 0.6572 | 0.7782 | 0.7126 | 239 |
| C3 | 0.6043 | 0.4748 | 0.5318 | 238 |
| C4 | 0.5498 | 0.5872 | 0.5679 | 235 |
| C5 | 0.4879 | 0.4244 | 0.4539 | 238 |
| C6 | 0.5036 | 0.5792 | 0.5388 | 240 |

*per-seed 산출. 3-seed mean±std는 aggregate_step4r_experiment.py 산출물 참조.*