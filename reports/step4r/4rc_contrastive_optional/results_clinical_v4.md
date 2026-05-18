# Step 4R-C clinical v4 — Contrastive Projection 학습 결과 (CONFIDENCE_POOL 교체)

- 생성 스크립트: `scripts/train_step4rc_contrastive_clinical_v4.py`
- best epoch: 31 / max 50
- final epoch trained: 41
- seed: 42, temperature: 0.07, batch: 256
- corpus unique phrases: 1733
- corpus unique template_key: 37

---

## 1. v4 final retrieval metrics (best checkpoint)

| split | strict R@1 | strict R@5 | strict R@10 | template R@1 | template R@5 | template R@10 |
|---|---:|---:|---:|---:|---:|---:|
| train | 0.0062 | 0.0318 | 0.0568 | 0.8305 | 0.8974 | 0.9201 |
| val | 0.0077 | 0.0376 | 0.0731 | 0.7270 | 0.7897 | 0.8175 |
| test | 0.0126 | 0.0456 | 0.0897 | 0.7085 | 0.8206 | 0.8430 |

## 2. v1 vs v3 vs v4 비교 — i2t retrieval

### 2.template_i2t_R@1 — template R@1

| split | v1 | v3 | v4 | Δ(v4-v1) | Δ(v4-v3) |
|---|---:|---:|---:|---:|---:|
| train | 0.8320 | 0.7753 | 0.8305 | -0.0016 | +0.0552 |
| val | 0.7256 | 0.6964 | 0.7270 | +0.0014 | +0.0306 |
| test | 0.7008 | 0.7064 | 0.7085 | +0.0077 | +0.0021 |

### 2.template_i2t_R@5 — template R@5

| split | v1 | v3 | v4 | Δ(v4-v1) | Δ(v4-v3) |
|---|---:|---:|---:|---:|---:|
| train | 0.8985 | 0.8456 | 0.8974 | -0.0011 | +0.0518 |
| val | 0.8210 | 0.7869 | 0.7897 | -0.0313 | +0.0028 |
| test | 0.8150 | 0.8192 | 0.8206 | +0.0056 | +0.0014 |

### 2.template_i2t_R@10 — template R@10

| split | v1 | v3 | v4 | Δ(v4-v1) | Δ(v4-v3) |
|---|---:|---:|---:|---:|---:|
| train | 0.9203 | 0.8706 | 0.9201 | -0.0002 | +0.0496 |
| val | 0.8454 | 0.8092 | 0.8175 | -0.0279 | +0.0084 |
| test | 0.8549 | 0.8486 | 0.8430 | -0.0119 | -0.0056 |

### 2.strict_i2t_R@1 — strict R@1

| split | v1 | v3 | v4 | Δ(v4-v1) | Δ(v4-v3) |
|---|---:|---:|---:|---:|---:|
| train | 0.0094 | 0.0047 | 0.0062 | -0.0031 | +0.0016 |
| val | 0.0056 | 0.0091 | 0.0077 | +0.0021 | -0.0014 |
| test | 0.0098 | 0.0126 | 0.0126 | +0.0028 | +0.0000 |

### 2.strict_i2t_R@5 — strict R@5

| split | v1 | v3 | v4 | Δ(v4-v1) | Δ(v4-v3) |
|---|---:|---:|---:|---:|---:|
| train | 0.0401 | 0.0242 | 0.0318 | -0.0083 | +0.0076 |
| val | 0.0467 | 0.0481 | 0.0376 | -0.0091 | -0.0104 |
| test | 0.0561 | 0.0491 | 0.0456 | -0.0105 | -0.0035 |

### 2.strict_i2t_R@10 — strict R@10

| split | v1 | v3 | v4 | Δ(v4-v1) | Δ(v4-v3) |
|---|---:|---:|---:|---:|---:|
| train | 0.0724 | 0.0446 | 0.0568 | -0.0156 | +0.0122 |
| val | 0.0919 | 0.0940 | 0.0731 | -0.0188 | -0.0209 |
| test | 0.1030 | 0.0904 | 0.0897 | -0.0133 | -0.0007 |

## 3. Training history (head/tail)

| epoch | train loss | val strict R@1 | val template R@1 | best? |
|---|---:|---:|---:|:---:|
| 1 | 4.7286 | 0.0063 | 0.5258 | * |
| 2 | 3.7188 | 0.0139 | 0.5550 | * |
| 3 | 3.5810 | 0.0084 | 0.5272 |  |
| 4 | 3.5197 | 0.0118 | 0.5529 |  |
| 5 | 3.4773 | 0.0118 | 0.5773 | * |
| ... | ... | ... | ... | |
| 37 | 3.0489 | 0.0125 | 0.6344 |  |
| 38 | 3.0166 | 0.0104 | 0.7026 |  |
| 39 | 3.0093 | 0.0070 | 0.7096 |  |
| 40 | 2.9949 | 0.0077 | 0.6950 |  |
| 41 | 2.9829 | 0.0077 | 0.6323 |  |

## 4. 산출물

- `data/step4r/4rc_contrastive_optional/model_clinical_v4/projection_head_clinical_v4.pt`
- `data/step4r/4rc_contrastive_optional/joint_embeddings_clinical_v4.npz`
- `reports/step4r/4rc_contrastive_optional/training_log_clinical_v4.csv`
- `reports/step4r/4rc_contrastive_optional/results_clinical_v4.md`

## 5. 기존 보존 (수정 없음)

- v1: `scripts/train_step4rc_contrastive_clinical.py`, `joint_embeddings_clinical.npz`, `results_clinical.md`
- v2: 동일 pattern
- v3: 동일 pattern
