# Step 4R-C clinical v3 — Contrastive Projection 학습 결과 (v1 + ambiguity_suffix)

- 생성 스크립트: `scripts/train_step4rc_contrastive_clinical_v3.py`
- best epoch: 17 / max 50
- final epoch trained: 27
- seed: 42, temperature: 0.07, batch: 256
- input: imu (9275, 128) + text_clinical_v3 (9275, 768)
- corpus unique phrases: 1733
- corpus unique template_key: 37  (v1: 37)

---

## 1. v3 final retrieval metrics (best checkpoint)

| split | strict R@1 | strict R@5 | strict R@10 | template R@1 | template R@5 | template R@10 |
|---|---:|---:|---:|---:|---:|---:|
| train | 0.0047 | 0.0242 | 0.0446 | 0.7753 | 0.8456 | 0.8706 |
| val | 0.0091 | 0.0481 | 0.0940 | 0.6964 | 0.7869 | 0.8092 |
| test | 0.0126 | 0.0491 | 0.0904 | 0.7064 | 0.8192 | 0.8486 |

## 2. v1 vs v3 비교 — i2t retrieval

### 2.1 strict R@K (정확한 sample 매칭)

| split | v1 R@1 | v3 R@1 | Δ | v1 R@5 | v3 R@5 | Δ | v1 R@10 | v3 R@10 | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 0.0094 | 0.0047 | -0.0047 | 0.0401 | 0.0242 | -0.0159 | 0.0724 | 0.0446 | -0.0278 |
| val | 0.0056 | 0.0091 | +0.0035 | 0.0467 | 0.0481 | +0.0014 | 0.0919 | 0.0940 | +0.0021 |
| test | 0.0098 | 0.0126 | +0.0028 | 0.0561 | 0.0491 | -0.0070 | 0.1030 | 0.0904 | -0.0126 |

### 2.2 template R@K (같은 template_key 내 매칭, *주된 retrieval metric*)

| split | v1 R@1 | v3 R@1 | Δ | v1 R@5 | v3 R@5 | Δ | v1 R@10 | v3 R@10 | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 0.8320 | 0.7753 | -0.0568 | 0.8985 | 0.8456 | -0.0529 | 0.9203 | 0.8706 | -0.0498 |
| val | 0.7256 | 0.6964 | -0.0292 | 0.8210 | 0.7869 | -0.0341 | 0.8454 | 0.8092 | -0.0362 |
| test | 0.7008 | 0.7064 | +0.0056 | 0.8150 | 0.8192 | +0.0042 | 0.8549 | 0.8486 | -0.0063 |

### 2.3 t2i 방향 — template R@K

| split | v1 R@1 | v3 R@1 | Δ | v1 R@5 | v3 R@5 | Δ | v1 R@10 | v3 R@10 | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 0.9095 | 0.9002 | -0.0094 | 0.9763 | 0.9719 | -0.0044 | 0.9857 | 0.9833 | -0.0023 |
| val | 0.8614 | 0.8969 | +0.0355 | 0.9631 | 0.9513 | -0.0118 | 0.9728 | 0.9735 | +0.0007 |
| test | 0.7849 | 0.8542 | +0.0694 | 0.9692 | 0.9474 | -0.0217 | 0.9762 | 0.9594 | -0.0168 |

## 3. Training history (head/tail)

| epoch | train loss | val strict R@1 | val template R@1 | best? |
|---|---:|---:|---:|:---:|
| 1 | 4.7120 | 0.0049 | 0.5209 | * |
| 2 | 3.7162 | 0.0091 | 0.5501 | * |
| 3 | 3.5736 | 0.0049 | 0.5348 |  |
| 4 | 3.5143 | 0.0111 | 0.5536 | * |
| 5 | 3.4862 | 0.0077 | 0.6038 | * |
| ... | ... | ... | ... | |
| 23 | 3.1939 | 0.0104 | 0.6797 |  |
| 24 | 3.1592 | 0.0077 | 0.6595 |  |
| 25 | 3.1363 | 0.0070 | 0.6748 |  |
| 26 | 3.1473 | 0.0084 | 0.6894 |  |
| 27 | 3.1348 | 0.0125 | 0.6616 |  |

## 4. 산출물

- `data/step4r/4rc_contrastive_optional/model_clinical_v3/projection_head_clinical_v3.pt`
- `data/step4r/4rc_contrastive_optional/joint_embeddings_clinical_v3.npz`
- `reports/step4r/4rc_contrastive_optional/training_log_clinical_v3.csv`
- `reports/step4r/4rc_contrastive_optional/results_clinical_v3.md`

## 5. 기존 보존 (수정 없음)

- `scripts/train_step4rc_contrastive_clinical.py` (v1)
- `data/step4r/4rc_contrastive_optional/joint_embeddings_clinical.npz` (v1)
- `reports/step4r/4rc_contrastive_optional/results_clinical.md` (v1)
- `scripts/train_step4rc_contrastive_clinical_v2.py` (v2)
- `reports/step4r/4rc_contrastive_optional/results_clinical_v2.md` (v2)
