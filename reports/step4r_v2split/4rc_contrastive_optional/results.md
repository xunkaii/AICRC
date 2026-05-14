# Step 4R-C Day 3-4 — Contrastive Projection 학습 결과

- 생성 스크립트: `scripts/train_step4rc_contrastive.py`
- best epoch: 32 / max 50
- final epoch trained: 42
- seed: 42, temperature: 0.07, batch: 256
- input: imu (9275, 128) + text (9275, 768)
- projection: imu/text -> 128-d L2-normalized

## 1. Final retrieval metrics (best checkpoint)

| split | strict R@1 i2t | strict R@5 i2t | template R@1 i2t | template R@5 i2t | template R@1 t2i |
|---|---:|---:|---:|---:|---:|
| train | 0.0049 | 0.0233 | 0.8247 | 0.8247 | 0.9908 |
| val | 0.0168 | 0.0824 | 0.7374 | 0.7654 | 0.7919 |
| test | 0.0068 | 0.0345 | 0.6542 | 0.6995 | 0.8858 |

**Interpretation**: strict R@1 chance = 1/n_split; template R@1 chance ~ 1/n_unique_templates.

## 2. Training history (loss + R@K curve)

| epoch | train loss | val strict R@1 (i2t) | val template R@1 (i2t) | best? |
|---|---:|---:|---:|:---:|
| 1 | 4.7069 | 0.0000 | 0.4302 | * |
| 2 | 3.7640 | 0.0112 | 0.5098 | * |
| 3 | 3.5529 | 0.0098 | 0.5265 | * |
| 4 | 3.4377 | 0.0168 | 0.5321 | * |
| 5 | 3.3757 | 0.0182 | 0.5573 | * |
| ... | ... | ... | ... | |
| 38 | 2.9736 | 0.0223 | 0.6662 |  |
| 39 | 2.9650 | 0.0112 | 0.6816 |  |
| 40 | 2.9494 | 0.0112 | 0.6522 |  |
| 41 | 2.9532 | 0.0168 | 0.6466 |  |
| 42 | 2.9439 | 0.0196 | 0.6564 |  |

Full log: `reports/step4r/4rc_contrastive_optional/training_log.csv`

## 3. Schema collapse check

4R-C 학습은 schema를 *읽기 전용*으로 사용하므로 schema 분포 변화 불가능. 그래도 sanity check 결과:

| level | actual rate | expected (b01.5/seed42 schema) | delta |
|---|---:|---:|---:|
| confident | 0.1926 | 0.1660 | +0.0266 |
| hedged | 0.7515 | 0.7890 | -0.0375 |
| low | 0.0400 | 0.0380 | +0.0020 |
| no_call | 0.0160 | 0.0070 | +0.0090 |

→ **No collapse** (4R-C는 새 model output 만들지 않음, schema input 그대로 사용).

## 4. 산출물

- `checkpoints/step4r/4rc_contrastive_optional/projection_head.pt`
- `data/step4r/4rc_contrastive_optional/joint_embeddings.npz`
- `reports/step4r/4rc_contrastive_optional/training_log.csv`
- `reports/step4r/4rc_contrastive_optional/results.md` (본 보고서)
