# 4R-C 임상 vocab(clinical) vs 추상 vocab(abstract, v1) A/B 비교

- 생성일: 2026-05-14
- 비교 대상: 
  - **v1 abstract**: `text_corpus.csv` (37 unique phrases, "정상 패턴/깊이 부족 계열/무릎 관련 오류")
  - **v2 clinical**: `text_corpus_clinical.csv` (1,733 unique phrases, "knee valgus / posterior pelvic tilt / left·right·bilateral")
- 동일 입력: IMU embeddings (frozen 4R-B b01.5/seed42, 9275 × 128), 동일 split, 동일 hyperparam (seed=42, T=0.07, lr=1e-3, batch=256)
- 차이: text corpus 어휘만

---

## 1. Headline — class R@K에서 극적 개선

| metric (test, i2t) | v1 abstract | v2 clinical | Δ (절대) | Δ (상대) |
|---|---:|---:|---:|---:|
| **class R@1** | 0.2852 | 0.3462 | **+0.061** | **+21%** |
| **class R@5** | 0.4639 | **0.8101** | **+0.346** | **+75%** |
| **class R@10** | 0.5620 | **0.9299** | **+0.368** | **+65%** |
| template R@1 | 0.6868 | 0.7022 | +0.015 | +2% |
| template R@5 | 0.7533 | 0.8150 | +0.062 | +8% |
| template R@10 | 0.7793 | 0.8549 | +0.076 | +10% |
| ambig R@1 | 0.9089 | 0.9145 | +0.006 | +1% |
| strict R@1 | 0.0105 | 0.0098 | -0.001 | -7% |

**핵심 메시지**: 임상 vocab은 **schema 카테고리(template/ambig)는 비슷한 성능, 진짜 정답 클래스(class R@K)에서 결정적 개선**을 가져왔다. 특히 R@5/R@10에서 +35pp/+37pp 향상은 retrieval 응용에서 매우 의미 있는 차이.

---

## 2. Class별 breakdown (test, k=1)

| class | figure_img.png 정의 | v1 class R@1 | v2 class R@1 | Δ | 해석 |
|---|---|---:|---:|---:|---|
| C1 | Normal | 0.397 | 0.342 | **-0.055** | 정상은 임상 어휘 효과 없음 (없는 것을 묘사하기 어려움) |
| **C2** | Insufficient depth | **0.063** | **0.360** | **+0.297** | **5.7× 개선** — "insufficient depth"가 결정적 식별 신호 |
| C3 | Insuff depth + posterior tilt + knee valgus | 0.261 | 0.433 | +0.173 | 3-component 복합 결함 묘사가 효과적 |
| C4 | Left-knee valgus | 0.374 | 0.434 | +0.060 | "좌측 knee valgus" directional 효과 |
| C5 | Right-knee valgus | 0.210 | 0.256 | +0.046 | "우측 knee valgus" directional 효과 |
| C6 | Bilateral knee valgus | 0.408 | 0.254 | **-0.154** | C5/C6 사이 confusion 가능성 (아래 §4 분석) |

**관찰**:
- **C2(깊이 부족) 이상치**: v1에서 0.063은 거의 chance(0.167) 미만 — 사실상 학습이 안 됐다. v1 corpus의 "깊이 부족 계열(C2)에 가까운 패턴"은 다른 abstract 표현과 어휘적으로 분리가 약했음. 임상 어휘 "insufficient depth, 대퇴 수평선 미달"가 강력한 anchor 역할.
- **C1·C6 역방향**: 임상 어휘가 *결함을 묘사하는 데 강하고, 정상을 묘사하는 데 약함*. C1은 "결함 없는"·"표준 폼" 등 negation/general 표현이라 IMU 특성과 매핑이 약함. C6는 좌/우/양측 directional이 도입되면서 within-group C5/C6 사이 혼동이 새로 발생.

---

## 3. Posture별 breakdown (test, k=1)

| posture | v1 template R@1 | v2 template R@1 | Δ | v1 class R@1 | v2 class R@1 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| SA (가장 긴 lever arm) | 0.746 | 0.740 | -0.006 | 0.254 | 0.396 | +0.142 |
| CA (중간 lever arm) | 0.681 | 0.662 | -0.019 | 0.264 | 0.319 | +0.055 |
| **HW (최단 lever arm)** | 0.782 | **0.866** | **+0.084** | 0.296 | 0.338 | +0.042 |

- **HW posture가 임상 어휘에서 가장 큰 이득** — lever arm이 짧을수록 손목 IMU 신호가 무릎 직하 운동에 가깝게 반응. 임상 anatomical descriptor가 가장 직접적 매핑을 만든다.
- SA/CA에서 template R@1이 약간 하락한 것은 lever arm으로 인한 IMU 신호의 의미 분리도가 임상 어휘의 미세 분리(좌/우/양측)를 따라가지 못해 confusion 발생. 그러나 class R@1은 모든 posture에서 개선됐다.

---

## 4. C5/C6 confusion 가설

C6가 -15pp 떨어지면서 C5가 +4.6pp 오른 패턴은 단순 분산이 아닐 가능성이 높다. 가설:

- v1: "무릎 관련 오류 후보(C5, C6)" → C5와 C6를 *함께 묶어 학습*하므로 둘 다 같은 representation 클러스터로 수렴
- v2: "우측 knee valgus" vs "양측 knee valgus" → 두 클래스가 분리된 anchor로 학습되지만, 손목 IMU 신호로는 right-only와 bilateral의 차이가 모호 → 모델이 *더 빈번한 케이스(C5: right)*에 over-attribute

**Implication**: 손목 IMU 단독으로는 left/right/bilateral knee valgus의 trilateral 구분이 본질적으로 어렵다. 이는 교수님 피드백 5 (video+MediaPipe 필요성)의 정량적 근거가 된다.

---

## 5. Strict vs template vs class R@K — 무엇이 향상됐고 왜인가

| metric | 의미 | v1 → v2 | 해석 |
|---|---|---|---|
| strict R@K | 정확히 같은 행(sample_id) 검색 | 거의 동일 | text 다양성↑ 때문에 strict 매칭은 더 어려워졌으나 알고리즘 능력은 동일 |
| template R@K | 같은 (posture, class_set, conf, anchor) 검색 | +2~10pp | template 구조는 동일 → 미세 개선은 임상 어휘의 분리도 효과 |
| **class R@K** | 같은 true class 검색 | **+6~37pp** | **본질적 개선** — true class signal이 IMU에 더 강하게 정렬됨 |
| ambig R@K | 같은 ambiguity_group 검색 | 거의 동일 (이미 0.91 ceiling) | ambig는 이미 포화 |

**핵심**: text 어휘 풍부도가 IMU 표현의 *진짜 클래스 식별력*에 직접 기여. 이는 contrastive learning이 단순한 schema 카테고리가 아니라 *anatomical structure*를 학습한다는 증거.

---

## 6. Thesis narrative 활용 포인트

### 6.1 교수님 피드백에 대응

| 피드백 항목 | 본 실험이 답하는 정량 결과 |
|---|---|
| (1) 자세 클래스 permutation으로 caption 다양화 | unique phrases 37 → 1,733 (47×) |
| (2) 관절 중심 설명 (허리·무릎·발목·고관절) | JOINT_FOCUS pool, class별 dominant joint 명시 |
| (3) Contribution + Constraint 구조 | 임상 vocab의 "insufficient depth + posterior pelvic tilt + knee valgus" 복합 표현 (C3) |
| (5) Caption 세밀화 (coarse → fine) | abstract "복합 오류 후보" → clinical "좌측 knee valgus, medial collapse" |
| (6) 기술적 핵심 기여 | class R@5 +35pp 정량 증거 |

### 6.2 Limitation으로 명시할 것

- **Level 3 (원인 설명)은 아직 없음**: 본 실험은 figure_img.png 정의의 결함 *명명(naming)* 단계까지. "왜 발목 가동성이 제한되어 C3가 발생했는가" 같은 원인 추론은 video+MediaPipe 필요 (교수님 피드백 5의 한계 확인).
- **C5/C6 trilateral 한계**: 손목 IMU 단독으론 right-only / bilateral knee valgus 구분이 어려움. multi-modal 필요성 정량 증거.
- **C1(Normal) 역방향**: 임상 어휘는 *결함*을 묘사할 때 강하고 *정상*을 묘사할 때 약하다. paper에서 "anatomical asymmetry of clinical vocabulary effectiveness"로 discussion.

### 6.3 Step 5_v2 §6.2 directional ban 정책 관점

- **외부 caption**: Step 5_v2~7_v2 정책 그대로. 본 실험은 4R-C 내부 학습 corpus만 directional vocab 허용 (CLAUDE.md "internal/external 분리" 아키텍처).
- thesis에서는 "training-time clinical detail + inference-time policy-compliant caption" 의 **dual-layer design**으로 기술 가능.

---

## 7. 산출물 비교 (대응)

| 파일 종류 | v1 (abstract) | v2 (clinical) |
|---|---|---|
| corpus | `text_corpus.csv` | `text_corpus_clinical.csv` |
| text emb | `text_embeddings.npz` | `text_embeddings_clinical.npz` |
| IMU emb | `imu_embeddings.npz` | (재사용 — 동일) |
| projection ckpt | `projection_head.pt` | `projection_head_clinical.pt` |
| joint emb | `joint_embeddings.npz` | `joint_embeddings_clinical.npz` |
| retrieval metrics | `retrieval_metrics.csv` | `retrieval_metrics_clinical.csv` |
| t-SNE class | `tsne_by_class.png` | `tsne_by_class_clinical.png` |
| t-SNE ambig | `tsne_by_ambiguity_group.png` | `tsne_by_ambiguity_group_clinical.png` |
| final report | `results_final.md` | `results_final_clinical.md` |
| 비교 분석 (본 파일) | — | `clinical_vs_abstract_comparison.md` |

기존 v1 산출물은 *전혀* 수정되지 않았다.

---

## 8. 다음 단계 후보

1. **Step 8_v2 (automatic validation)**: clinical caption도 schema_faithfulness 평가 대상에 추가
2. **C5/C6 confusion 정량 분석**: confusion matrix 분리해서 C5↔C6 오분류 비율 직접 측정
3. **C1 정상 클래스 보강**: "결함 없음" 표현을 anatomical positive marker로 reformulate (예: "neutral knee tracking", "lumbar neutral throughout descent") → C1 회복 시도
4. **Video 모달리티 추가** (교수님 피드백 5): video가 확보되면 MediaPipe joint trajectory를 additional text encoder input으로 → trilateral knee valgus 구분 가능성 검증
