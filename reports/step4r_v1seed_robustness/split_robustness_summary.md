# v1-ratio (36/8/8) split-robustness — 4 split seeds × 3 learn seeds

본 보고서는 v1의 b01.5 test F1 0.513이 어느 정도 split-robust한지를 정량화한다. 인원 비율(36/8/8)과 학습 recipe(b01.5: σ=0.05, scale [0.85, 1.15])는 v1 baseline과 완전히 동일하며, 참가자 셔플 seed만 바꿔서 같은 학습을 반복했다.

- baseline: `s42` (기존 v1 confirmed baseline, 학습 seed [42, 43, 44])
- new: `s7`, `s123`, `s2024`, `s7777` (각각 학습 seed [42, 43, 44])
- 총 5 splits × 3 learn seeds = 15 runs

## 1. test macro F1 matrix

| split_seed | seed 42 | seed 43 | seed 44 | mean | std | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| s42 | 0.5134 | 0.5045 | 0.5206 | **0.5128** | 0.0066 | 0.5045 | 0.5206 |
| s7 | 0.4693 | 0.4661 | 0.5002 | **0.4785** | 0.0154 | 0.4661 | 0.5002 |
| s123 | 0.5323 | 0.5071 | 0.4773 | **0.5056** | 0.0225 | 0.4773 | 0.5323 |
| s2024 | 0.5190 | 0.5148 | 0.4930 | **0.5089** | 0.0114 | 0.4930 | 0.5190 |
| s7777 | 0.4980 | 0.4875 | 0.4704 | **0.4853** | 0.0114 | 0.4704 | 0.4980 |

- **grand mean** = 0.4982
- **between-split std** = 0.0137  (splits별 mean 4~5개의 std)
- **within-split std (mean)** = 0.0144  (각 split 안 3-learn-seed std의 평균)
- **total std (15 runs)** = 0.0199

## 2. test ECE (after calibration) matrix

| split_seed | seed 42 | seed 43 | seed 44 | mean | std |
|---|---:|---:|---:|---:|---:|
| s42 | 0.0362 | 0.0436 | 0.0459 | **0.0419** | 0.0042 |
| s7 | 0.0527 | 0.0830 | 0.0417 | **0.0591** | 0.0175 |
| s123 | 0.0715 | 0.0319 | 0.0464 | **0.0499** | 0.0164 |
| s2024 | 0.0380 | 0.0361 | 0.0184 | **0.0308** | 0.0088 |
| s7777 | 0.0458 | 0.0429 | 0.0756 | **0.0548** | 0.0148 |

## 3. fitted temperature T matrix

| split_seed | seed 42 | seed 43 | seed 44 | mean | std |
|---|---:|---:|---:|---:|---:|
| s42 | 1.970 | 2.107 | 2.016 | **2.031** | 0.057 |
| s7 | 4.554 | 2.311 | 2.991 | **3.286** | 0.939 |
| s123 | 2.420 | 2.647 | 1.378 | **2.148** | 0.552 |
| s2024 | 1.840 | 2.214 | 1.338 | **1.798** | 0.359 |
| s7777 | 3.479 | 3.832 | 2.474 | **3.262** | 0.575 |

## 4. variance decomposition 해석

→ split variance와 학습 variance가 비슷한 규모. 두 효과 모두 보고 권장.

### 정량 기준

- between-split std는 *다른 8명을 test로 뽑았을 때 b01.5 mean F1이 얼마나 흔들리는지*를 잡는다.
- within-split std는 *같은 8명 test 고정 시 학습 seed에 따라 얼마나 흔들리는지*를 잡는다. b01.5는 이미 학습 seed std가 작다 (0.008 부근).
- 두 std 비율이 v1 baseline 보고의 신뢰성을 결정한다: between이 within보다 1.5× 크면 split이 결정적, 작으면 학습이 결정적, 비슷하면 둘 다.

## 5. 산출물 위치

```
data/step4r_v1seed_robustness/s{7,123,2024,7777}/manifest_split.csv (4 files)
data/step4r_v1seed_robustness/s{...}/4rb_attention/step4r_sequence_dataset.npz (4 files)
data/step4r_v1seed_robustness/s{...}/4rb_attention/experiments/b01_5_aug_jitter_scale_strong/seed{42,43,44}/  (12 dirs)
reports/step4r_v1seed_robustness/s{...}/...  (12 dirs)
checkpoints/step4r_v1seed_robustness/s{...}/...  (12 dirs)
```

삭제 시: `data/step4r_v1seed_robustness/`, `checkpoints/step4r_v1seed_robustness/`, `reports/step4r_v1seed_robustness/` 만 제거하면 됨.