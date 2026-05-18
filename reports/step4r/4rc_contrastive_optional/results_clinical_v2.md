# Step 4R-C clinical v2 — Contrastive Projection 학습 결과 (POSTURE_POOL + JOINT_POOL)

- 생성 스크립트: `scripts/train_step4rc_contrastive_clinical_v2.py`
- best epoch: 1 / max 50
- final epoch trained: 11
- seed: 42, temperature: 0.07, batch: 256
- input: imu (9275, 128) + text_clinical_v2 (9275, 768)
- corpus unique phrases: 5156 (v1: 1733)
- corpus unique template_key: 157

---

## 1. v2 final retrieval metrics (best checkpoint)

| split | strict R@1 | strict R@5 | strict R@10 | template R@1 | template R@5 | template R@10 |
|---|---:|---:|---:|---:|---:|---:|
| train | 0.0012 | 0.0083 | 0.0168 | 0.1928 | 0.2662 | 0.3249 |
| val | 0.0042 | 0.0286 | 0.0557 | 0.2709 | 0.4213 | 0.4923 |
| test | 0.0049 | 0.0273 | 0.0470 | 0.1780 | 0.3006 | 0.3861 |

## 2. v1 vs v2 비교 — i2t retrieval

### 2.1 strict R@K (정확한 sample 매칭)

| split | v1 R@1 | v2 R@1 | Δ R@1 | v1 R@5 | v2 R@5 | Δ R@5 | v1 R@10 | v2 R@10 | Δ R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 0.0094 | 0.0012 | -0.0081 | 0.0401 | 0.0083 | -0.0318 | 0.0724 | 0.0168 | -0.0555 |
| val | 0.0056 | 0.0042 | -0.0014 | 0.0467 | 0.0286 | -0.0181 | 0.0919 | 0.0557 | -0.0362 |
| test | 0.0098 | 0.0049 | -0.0049 | 0.0561 | 0.0273 | -0.0287 | 0.1030 | 0.0470 | -0.0561 |

### 2.2 template R@K (같은 template_key 내 매칭, *주된 retrieval metric*)

| split | v1 R@1 | v2 R@1 | Δ R@1 | v1 R@5 | v2 R@5 | Δ R@5 | v1 R@10 | v2 R@10 | Δ R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 0.8320 | 0.1928 | -0.6393 | 0.8985 | 0.2662 | -0.6323 | 0.9203 | 0.3249 | -0.5954 |
| val | 0.7256 | 0.2709 | -0.4547 | 0.8210 | 0.4213 | -0.3997 | 0.8454 | 0.4923 | -0.3531 |
| test | 0.7008 | 0.1780 | -0.5228 | 0.8150 | 0.3006 | -0.5144 | 0.8549 | 0.3861 | -0.4688 |

### 2.3 t2i 방향 (text → IMU) — template R@K

| split | v1 R@1 | v2 R@1 | Δ R@1 | v1 R@5 | v2 R@5 | Δ R@5 | v1 R@10 | v2 R@10 | Δ R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 0.9095 | 0.1848 | -0.7247 | 0.9763 | 0.4986 | -0.4777 | 0.9857 | 0.6446 | -0.3411 |
| val | 0.8614 | 0.3280 | -0.5334 | 0.9631 | 0.6386 | -0.3245 | 0.9728 | 0.7437 | -0.2291 |
| test | 0.7849 | 0.1752 | -0.6097 | 0.9692 | 0.3889 | -0.5802 | 0.9762 | 0.5011 | -0.4751 |

## 3. Training history (head/tail)

| epoch | train loss | val strict R@1 | val template R@1 | best? |
|---|---:|---:|---:|:---:|
| 1 | 4.8431 | 0.0049 | 0.2716 | * |
| 2 | 3.7855 | 0.0091 | 0.1616 |  |
| 3 | 3.5619 | 0.0097 | 0.1818 |  |
| 4 | 3.4579 | 0.0049 | 0.2347 |  |
| 5 | 3.3704 | 0.0084 | 0.1831 |  |
| ... | ... | ... | ... | |
| 7 | 3.2388 | 0.0125 | 0.1323 |  |
| 8 | 3.1446 | 0.0125 | 0.2173 |  |
| 9 | 3.0780 | 0.0077 | 0.1859 |  |
| 10 | 3.0510 | 0.0097 | 0.1741 |  |
| 11 | 2.9848 | 0.0118 | 0.2110 |  |

## 4. 산출물

- `data/step4r/4rc_contrastive_optional/model_clinical_v2/projection_head_clinical_v2.pt`
- `data/step4r/4rc_contrastive_optional/joint_embeddings_clinical_v2.npz`
- `reports/step4r/4rc_contrastive_optional/training_log_clinical_v2.csv`
- `reports/step4r/4rc_contrastive_optional/results_clinical_v2.md`

## 5. 기존 보존 (수정 없음)

- `scripts/train_step4rc_contrastive_clinical.py` (v1)
- `checkpoints/step4r/4rc_contrastive_optional/projection_head_clinical.pt`
- `data/step4r/4rc_contrastive_optional/joint_embeddings_clinical.npz`
- `reports/step4r/4rc_contrastive_optional/results_clinical.md`
