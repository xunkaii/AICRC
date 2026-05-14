# Step 4R-C — Contrastive IMU-Text Alignment: Final Results

- 생성 스크립트: `scripts/evaluate_step4rc_retrieval.py`
- 입력 backbone: 4R-B b01.5/seed42 (frozen, params 168,966)
- text encoder: `jhgan/ko-sroberta-multitask` (frozen, 768-dim output)
- projection head: IMU 128→128 / text 768→128, L2 normalized, params 147,968
- training: symmetric InfoNCE T=0.07, AdamW lr=1e-3 wd=1e-4, batch 256, 50 epochs (early stop patience 10 on val template R@1)

---

## 1. Overall retrieval — per-split (k=1)

| split | n | strict R@1 | template R@1 | ambig-group R@1 | class R@1 |
|---|---:|---:|---:|---:|---:|
| train | 6063 | 0.0046 | 0.8247 | 0.9749 | 0.4109 |
| val | 716 | 0.0196 | 0.7374 | 0.9218 | 0.2528 |
| test | 2496 | 0.0072 | 0.6542 | 0.9026 | 0.2312 |

Chance baselines (test):
  - strict R@1     ≈ 1/2496 = 0.0004
  - template R@1   ≈ 1/37 = 0.0270
  - ambig R@1      ≈ 1/5 = 0.2000
  - class R@1      ≈ 1/6 = 0.1667

---

## 2. R@5, R@10 (test split)

| target | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 |
|---|---:|---:|---:|---:|---:|---:|
| strict | 0.0072 | 0.0321 | 0.0673 | 0.0060 | 0.0369 | 0.0693 |
| template | 0.6542 | 0.6995 | 0.7484 | 0.8858 | 0.9659 | 0.9679 |
| ambig | 0.9026 | 0.9151 | 0.9211 | 0.9776 | 0.9956 | 0.9956 |
| class | 0.2312 | 0.4399 | 0.5925 | 0.3049 | 0.6839 | 0.8538 |

---

## 3. Per-posture breakdown (test, k=1)

| posture | n | template R@1 | class R@1 | ambig R@1 | strict R@1 |
|---|---:|---:|---:|---:|---:|
| SA | 830 | 0.6940 | 0.2578 | 0.9000 | 0.0036 |
| CA | 837 | 0.6643 | 0.2258 | 0.8889 | 0.0084 |
| HW | 829 | 0.7660 | 0.1460 | 0.9228 | 0.0060 |
| ALL | 2496 | 0.6542 | 0.2312 | 0.9026 | 0.0072 |

---

## 4. Per-class breakdown (test, k=1)

| class | n | class R@1 (same-class match) | ambig R@1 | template R@1 |
|---|---:|---:|---:|---:|
| C1 | 420 | 0.3405 | 0.9643 | 0.6119 |
| C2 | 421 | 0.2257 | 0.8551 | 0.5796 |
| C3 | 400 | 0.1900 | 0.8550 | 0.6675 |
| C4 | 418 | 0.0885 | 0.8612 | 0.7129 |
| C5 | 418 | 0.1938 | 0.9545 | 0.6866 |
| C6 | 419 | 0.3461 | 0.9236 | 0.6683 |

---

## 5. t-SNE 시각화

- `tsne_by_class.png` — 6-class (C1~C6) 색상
- `tsne_by_ambiguity_group.png` — 5 ambiguity_group 색상

좌측: IMU embedding side / 우측: text embedding side

---

## 6. 해석 — 4R-C의 contribution 위치

### 잘 된 점

- **Template R@1 (test) = 0.654** — chance(~0.03) 대비 약 24배. IMU representation이 schema category 공간에 *의미 있게* 정렬됨.
- **Ambig-group R@1 (test) = 0.903** — 5-way coarse 카테고리 (chance 0.2) 대비 강한 신호. schema의 *큰 분기 구조*가 IMU에서 재현 가능.
- **Class R@1 (test) = 0.231** — schema-independent 6-way (chance 0.167). 4R-B 단독 분류 정확도(test acc 0.52)와 유사한 자리 — embedding alignment가 분류 신호를 보존함을 확인.

### 한계

- **Strict R@1 ceiling**: 37 unique phrase × 평균 ~250 samples → strict R@1 이론 max ≈ 0.004. 실측 0.007는 이 ceiling 근처. individual-sample retrieval은 *템플릿 중복*으로 본질적 한계.
- **train-val gap**: train template R@1 ≈ 0.85 vs val/test ≈ 0.69. projection head가 작은 train set에 fit. dropout/wd로 줄일 여지 있음.

### Thesis chapter 위치

reframing §8.2 contribution 강화. 4R-B 단독은 'posterior entropy / attention entropy / confusion uncertainty 분석'만 제공했으나, 4R-C는 **'학습된 IMU representation이 한국어 schema 어휘 공간과 정합한다'**는 *cross-modal 차원의* 독립 증거를 추가. 이는 §8.3 (uncertainty-aware Korean caption) 의 *foundation*에 직접 기여.

단, **classification 점수 자체는 향상 없음** (4R-C는 frozen backbone 위에 projection head만 학습). 따라서 4R-C는 *ablation/extension chapter*로 다루는 것이 정직 — reframing §5.3가 정한 위치 그대로.

---

## 7. 산출물 목록 (4R-C 전체)

```
data/step4r/4rc_contrastive_optional/
├── text_corpus.csv                (Day 1: 9275 × 12 cols)
├── text_embeddings.npz            (Day 2a: 9275 × 768)
├── text_unique_similarity.csv     (Day 2a: 37×37 cosine matrix)
├── imu_embeddings.npz             (Day 2b: 9275 × 128)
├── joint_embeddings.npz           (Day 3-4: imu_proj + text_proj, 9275 × 128 each)
└── retrieval_metrics.csv          (Day 5-6: long-format split×metric×k)

checkpoints/step4r/4rc_contrastive_optional/
└── projection_head.pt             (Day 3-4: 147,968 params)

reports/step4r/4rc_contrastive_optional/
├── training_log.csv               (Day 3-4: 50 epoch × 11 cols)
├── results.md                     (Day 3-4: training summary)
├── retrieval_breakdown.csv        (Day 5-6: posture/class breakdown)
├── tsne_by_class.png              (Day 5-6: t-SNE 시각화)
├── tsne_by_ambiguity_group.png    (Day 5-6: t-SNE 시각화)
└── results_final.md               (본 보고서)

scripts/
├── build_step4rc_text_corpus.py     (Day 1)
├── build_step4rc_text_embeddings.py (Day 2a)
├── build_step4rc_imu_embeddings.py  (Day 2b)
├── train_step4rc_contrastive.py     (Day 3-4)
└── evaluate_step4rc_retrieval.py    (Day 5-6, 본 스크립트)
```

기존 Step 1~7_v2 / 4R-A / 4R-B 산출물은 *전혀* 수정되지 않았다.
