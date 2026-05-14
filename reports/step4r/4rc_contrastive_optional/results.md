# Step 4R-C Day 3-4 — Contrastive Projection 학습 결과

- 생성 스크립트: `scripts/train_step4rc_contrastive.py`
- best epoch: 46 / max 50
- final epoch trained: 50
- seed: 42, temperature: 0.07, batch: 256
- input: imu (9275, 128) + text (9275, 768)
- projection: imu/text -> 128-d L2-normalized

## 1. Final retrieval metrics (best checkpoint)

| split | strict R@1 i2t | strict R@5 i2t | template R@1 i2t | template R@5 i2t | template R@1 t2i |
| ----- | -------------: | -------------: | ---------------: | ---------------: | ---------------: |
| train |         0.0037 |         0.0198 |           0.8490 |           0.8523 |           0.9880 |
| val   |         0.0091 |         0.0543 |           0.6825 |           0.7180 |           0.7187 |
| test  |         0.0112 |         0.0533 |           0.6868 |           0.7533 |           0.8458 |

**Interpretation**: strict R@1 chance = 1/n_split; template R@1 chance ~ 1/n_unique_templates.

## 2. Training history (loss + R@K curve)

| epoch | train loss | val strict R@1 (i2t) | val template R@1 (i2t) | best? |
|---|---:|---:|---:|:---:|
| 1 | 4.7459 | 0.0049 | 0.5063 | * |
| 2 | 3.7876 | 0.0084 | 0.4631 |  |
| 3 | 3.5835 | 0.0091 | 0.4554 |  |
| 4 | 3.5348 | 0.0070 | 0.5195 | * |
| 5 | 3.4912 | 0.0063 | 0.4798 |  |
| ... | ... | ... | ... | |
| 46 | 3.1017 | 0.0091 | 0.6825 | * |
| 47 | 3.1244 | 0.0118 | 0.6142 |  |
| 48 | 3.1011 | 0.0091 | 0.6253 |  |
| 49 | 3.1016 | 0.0118 | 0.5327 |  |
| 50 | 3.0996 | 0.0056 | 0.6024 |  |

Full log: `reports/step4r/4rc_contrastive_optional/training_log.csv`

## 3. Schema collapse check

4R-C 학습은 schema를 *읽기 전용*으로 사용하므로 schema 분포 변화 불가능. 그래도 sanity check 결과:

| level | actual rate | expected (b01.5/seed42 schema) | delta |
|---|---:|---:|---:|
| confident | 0.1781 | 0.1660 | +0.0121 |
| hedged | 0.7763 | 0.7890 | -0.0127 |
| low | 0.0392 | 0.0380 | +0.0012 |
| no_call | 0.0064 | 0.0070 | -0.0006 |

→ **No collapse** (4R-C는 새 model output 만들지 않음, schema input 그대로 사용).

## 4. 산출물

- `checkpoints/step4r/4rc_contrastive_optional/projection_head.pt`
- `data/step4r/4rc_contrastive_optional/joint_embeddings.npz`
- `reports/step4r/4rc_contrastive_optional/training_log.csv`
- `reports/step4r/4rc_contrastive_optional/results.md` (본 보고서)
