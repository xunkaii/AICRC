# Step 4R-C clinical — Contrastive Projection 학습 결과

- 생성 스크립트: `scripts/train_step4rc_contrastive_clinical.py`
- best epoch: 38 / max 50
- final epoch trained: 48
- seed: 42, temperature: 0.07, batch: 256
- input: imu (9275, 128) + text_clinical (9275, 768)
- corpus unique phrases: 1733 (v1: 37)

## 1. Final retrieval metrics (best checkpoint)

| split | strict R@1 i2t | strict R@5 i2t | template R@1 i2t | template R@5 i2t | template R@1 t2i |
|---|---:|---:|---:|---:|---:|
| train | 0.0094 | 0.0401 | 0.8320 | 0.8985 | 0.9095 |
| val | 0.0056 | 0.0467 | 0.7256 | 0.8210 | 0.8614 |
| test | 0.0098 | 0.0561 | 0.7008 | 0.8150 | 0.7849 |

## 2. Training history (head/tail)

| epoch | train loss | val strict R@1 i2t | val template R@1 i2t | best? |
|---|---:|---:|---:|:---:|
| 1 | 4.7382 | 0.0042 | 0.5397 | * |
| 2 | 3.7173 | 0.0049 | 0.5557 | * |
| 3 | 3.5779 | 0.0077 | 0.5411 |  |
| 4 | 3.5193 | 0.0056 | 0.5675 | * |
| 5 | 3.4717 | 0.0084 | 0.5940 | * |
| ... | ... | ... | ... | |
| 44 | 2.9699 | 0.0111 | 0.6887 |  |
| 45 | 2.9494 | 0.0063 | 0.6776 |  |
| 46 | 2.9701 | 0.0063 | 0.6720 |  |
| 47 | 2.9697 | 0.0042 | 0.7145 |  |
| 48 | 2.9326 | 0.0070 | 0.7291 |  |

## 3. 산출물

- `checkpoints/step4r/4rc_contrastive_optional/projection_head_clinical.pt`
- `data/step4r/4rc_contrastive_optional/joint_embeddings_clinical.npz`
- `reports/step4r/4rc_contrastive_optional/training_log_clinical.csv`
- `reports/step4r/4rc_contrastive_optional/results_clinical.md`
