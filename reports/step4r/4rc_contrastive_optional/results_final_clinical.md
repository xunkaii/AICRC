# Step 4R-C clinical — Final Retrieval Results

- 생성 스크립트: `scripts/evaluate_step4rc_retrieval_clinical.py`
- 입력 corpus: text_corpus_clinical.csv (unique phrases: 1733)
- text encoder: `jhgan/ko-sroberta-multitask` (frozen, 768d)
- IMU backbone: 4R-B b01.5/seed42 (frozen, attn_vec 128d)
- projection head: 128→128 + 768→128, L2 normalized, params 147,968

---

## 1. Overall retrieval — per-split (k=1)

| split | n | strict R@1 | template R@1 | ambig R@1 | class R@1 |
|---|---:|---:|---:|---:|---:|
| train | 6412 | 0.0089 | 0.8319 | 0.9630 | 0.4633 |
| val | 1436 | 0.0063 | 0.7319 | 0.9074 | 0.3579 |
| test | 1427 | 0.0098 | 0.7022 | 0.9145 | 0.3462 |

Chance baselines (test):
  - strict R@1     ≈ 1/1427 = 0.0007
  - template R@1   ≈ 1/34 = 0.0294
  - ambig R@1      ≈ 1/5 = 0.2000
  - class R@1      ≈ 1/6 = 0.1667

## 2. R@5, R@10 (test split)

| target | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 |
|---|---:|---:|---:|---:|---:|---:|
| strict | 0.0098 | 0.0568 | 0.1023 | 0.0112 | 0.0561 | 0.1016 |
| template | 0.7022 | 0.8150 | 0.8549 | 0.7849 | 0.9692 | 0.9762 |
| ambig | 0.9145 | 0.9355 | 0.9446 | 0.9748 | 0.9958 | 0.9965 |
| class | 0.3462 | 0.8101 | 0.9299 | 0.3791 | 0.7498 | 0.8690 |

## 3. Per-posture breakdown (test, k=1)

| posture | n | template R@1 | class R@1 | ambig R@1 | strict R@1 |
|---|---:|---:|---:|---:|---:|
| SA | 477 | 0.7400 | 0.3962 | 0.9203 | 0.0147 |
| CA | 473 | 0.6617 | 0.3192 | 0.9112 | 0.0127 |
| HW | 477 | 0.8658 | 0.3375 | 0.9245 | 0.0105 |
| ALL | 1427 | 0.7022 | 0.3462 | 0.9145 | 0.0098 |

## 4. Per-class breakdown (test, k=1)

| class | n | class R@1 | ambig R@1 | template R@1 |
|---|---:|---:|---:|---:|
| C1 | 237 | 0.3418 | 0.9789 | 0.7046 |
| C2 | 239 | 0.3598 | 0.9163 | 0.6360 |
| C3 | 238 | 0.4328 | 0.8445 | 0.7059 |
| C4 | 235 | 0.4340 | 0.8809 | 0.7957 |
| C5 | 238 | 0.2563 | 0.9664 | 0.7311 |
| C6 | 240 | 0.2542 | 0.9000 | 0.6417 |

## 5. t-SNE 시각화

- `tsne_by_class_clinical.png` (6-class)
- `tsne_by_ambiguity_group_clinical.png`

## 6. 비고

Clinical corpus는 v1 대비 unique phrase 수가 47배 증가 (37→1733)했고 직접적인 임상 어휘(knee valgus / posterior tilting / left/right/bilateral)를 포함. v1과의 정량 비교는 `clinical_vs_abstract_comparison.md` 참조.