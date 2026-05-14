# v1 vs v2split — Step 4R 전 파이프라인 비교 요약

본 보고서는 같은 b01.5 recipe / 같은 코드 / 같은 seed(42)를 두 가지 participant split에서 돌렸을 때의 차이를 한 곳에 정리한다. **목적은 split 선택이 결과에 얼마나 큰 영향을 주는지 가시화**하는 것이며, b01.5 confirmed baseline (v1) 의 채택 여부를 재논의하기 위한 것은 아니다.

## 0. split 정의

| split | participants | samples (train / val / test) | split_version |
|---|---|---|---|
| **v1** (확정 baseline) | 36 / 8 / 8 | 6412 / 1436 / 1427 | `v1_36_8_8` |
| **v2split** (실험) | 34 / 4 / **14** | 6063 / 716 / **2496** | `v2_34_4_14_s42` |

- 두 split 모두 participant-disjoint, seed=42 셔플.
- v2split의 val은 50% 축소(8→4), test는 75% 증가(8→14). train은 약 5.4% 감소.
- v2split test 참가자: EP06, EP18, EP19, EP21, EP24, EP25, EP30, EP35, EP38, EP40, EP42, EP47, EP50, EP52.

본 v2split 결과는 모두 **단일 seed (42)** 산출이다. v1 b01.5 baseline의 3-seed mean ± std (test F1 0.513 ± 0.008) 와 직접 비교할 때는 v2split도 3-seed로 확장해야 형평이 맞다. 본 비교는 *seed 42 대 seed 42*의 점 비교일 뿐임을 명시한다.

## 1. 4R-A — HGB feature ceiling (calibrated)

### 1.1 raw branch

| 지표 | v1 | v2split | Δ |
|---|---:|---:|---:|
| fitted T | 2.452 | 2.813 | +0.361 |
| test acc | 0.3062 | 0.2965 | −0.0097 |
| test macro F1 | 0.3004 | 0.2937 | −0.0067 |
| test ECE before | 0.2300 | 0.2678 | +0.0378 |
| test ECE after | **0.0288** | **0.0372** | +0.0084 |
| test log loss after | 1.6216 | 1.6536 | +0.0320 |

### 1.2 zscore branch

| 지표 | v1 | v2split | Δ |
|---|---:|---:|---:|
| fitted T | 2.492 | 2.795 | +0.303 |
| test acc | 0.2985 | 0.2877 | −0.0108 |
| test macro F1 | 0.2914 | 0.2824 | −0.0090 |
| test ECE before | 0.2271 | 0.2694 | +0.0423 |
| test ECE after | **0.0356** | **0.0416** | +0.0060 |

**해석**:
- HGB는 feature-based ceiling이라 split-sensitivity가 가장 작다 (F1 -0.007 ~ -0.011).
- T가 v2split에서 더 커진 것은 train↔test distribution shift가 좀 더 크기 때문(test 참가자 14명에 더 다양성).
- calibrated test ECE 차이는 0.6~0.8%p 수준 — 거의 동일 calibration 품질.

## 2. 4R-B — BiGRU+Attention b01.5/seed42 (calibrated)

| 지표 | v1 | v2split | Δ |
|---|---:|---:|---:|
| best epoch | 24 | 27 | +3 |
| val macro F1 | 0.5169 | 0.5026 | −0.0143 |
| fitted T | 1.970 | 2.021 | +0.051 |
| **test acc** | **0.5214** | **0.4615** | **−0.0599** |
| **test macro F1** | **0.5134** | **0.4436** | **−0.0698** |
| test log loss before | 1.2144 | 1.6657 | +0.4513 |
| test ECE before | 0.1415 | 0.2451 | +0.1036 |
| test log loss after | 1.1168 | 1.3020 | +0.1852 |
| test ECE after | **0.0362** | **0.0777** | +0.0415 |

**해석**:
- test F1 -0.07p는 작지 않은 격차다. v1 3-seed std 0.008 기준 약 **9σ**.
- 다만 v2split도 단일 seed라 그 자체로 ±0.008 이상의 분산을 가질 수 있다 (b00 baseline 3-seed std 0.026 참고). 3-seed로 확장하면 격차가 줄거나 커질 수 있음.
- T는 거의 동일(1.97 vs 2.02) — 4R-B의 calibration narrative 자체는 split robust.
- calibrated ECE 0.078은 v1의 0.036보다 큼. test 참가자 14명이 더 다양해 single scalar T로 보정하기 어려운 분포.
- val 716 (v1의 절반)에서 T를 fit한 점도 ECE 차이의 일부 요인.

## 3. 4R-C — Contrastive IMU-text alignment (frozen 4R-B/seed42)

| 지표 (test) | v1 | v2split | Δ |
|---|---:|---:|---:|
| n_test | 1427 | 2496 | +75% |
| unique phrases (corpus) | 37 | 38 | +1 |
| **template R@1** | **0.6868** | **0.6542** | −0.0326 |
| template R@5 | 0.7533 | 0.6995 | −0.0538 |
| template R@10 | 0.7793 | 0.7484 | −0.0309 |
| ambig-group R@1 | 0.9089 | 0.9026 | −0.0063 |
| class R@1 | 0.2852 | 0.2312 | −0.0540 |
| strict R@1 | 0.0105 | 0.0072 | −0.0033 |
| best epoch (contrastive) | n/a | 32 | — |

### 3.1 test posture-wise (v1 → v2split)

| posture | template R@1 | class R@1 | ambig R@1 |
|---|---|---|---|
| SA | 0.746 → 0.694 | 0.254 → 0.258 | 0.935 → 0.900 |
| CA | 0.681 → 0.664 | 0.264 → 0.226 | 0.884 → 0.889 |
| HW | 0.782 → 0.766 | 0.296 → 0.146 | 0.914 → 0.923 |

**해석**:
- v2split test의 unique phrase는 38(+1) — 4R-B schema 출력이 살짝 다른 ambiguity 분기 한 가지를 추가로 활성화함.
- ambig R@1은 거의 동일(0.91 → 0.90) — *coarse* schema 정렬은 split-robust.
- template R@1은 -0.033, class R@1은 -0.054. 4R-B test F1 -0.07p가 그대로 representation downstream으로 전파된 양상.
- HW class R@1이 0.296 → 0.146으로 크게 떨어진 것은 흥미로움 — v2split test의 HW 참가자 구성이 v1과 달라 IMU representation 정렬이 약한 부분이 노출됨.

## 4. 종합 판단

### 4.1 split-sensitivity 정량

| 단계 | test F1 / R@1 Δ | 해석 |
|---|---|---|
| 4R-A HGB raw | −0.007 | 거의 무관 |
| 4R-A HGB zscore | −0.009 | 거의 무관 |
| 4R-B b01.5 (단일 seed) | **−0.070** | 의미 있는 격차 |
| 4R-C template R@1 | −0.033 | 4R-B 격차의 약 절반 downstream |
| 4R-C ambig R@1 | −0.006 | coarse alignment는 robust |

### 4.2 어느 split이 "더 honest"한가

본 비교만으로는 **단정 불가**. 근거:

1. **v2split test가 75% 더 크다** — 분산은 줄지만 한 seed 점 추정. v1의 3-seed mean ± 8을 만들었듯 v2도 3-seed가 필요.
2. **v2split val이 50% 작다** — early stopping / T scaling 결정의 noise가 더 크다. 즉 v2split의 학습 결과 자체가 "운 좋은 best epoch" 일 수도 unfortunate일 수도.
3. **v1 b01.5 baseline의 0.513 ± 0.008은 augmentation으로 variance가 압축된 결과**다. v2split은 same recipe라 같은 variance reduction 효과는 기대되지만 mean이 어디로 갈지는 1 seed로는 알 수 없음.

### 4.3 thesis 기여 관점

- **4R-C ambig R@1은 split-robust (0.91 vs 0.90)** — coarse cross-modal alignment 결과는 split 변경에도 그대로 유지. 이는 reframing §8.2 contribution의 robustness 보강 evidence가 된다.
- **4R-B test F1 격차는 두 가지로 해석 가능**:
  - **(a) split luck**: v2split test의 14명 중 어려운 참가자가 우연히 많이 들어감.
  - **(b) honest measurement**: v1 test 8명이 우연히 평균적인 분포라서 0.513이 과대 추정.
  - 어느 쪽이 맞는지는 v2split 3-seed (43, 44 추가) 또는 v1의 다른 셔플 (다른 split_seed)로 확인 가능.

### 4.4 결론

v2split single-seed 결과는 **v1 confirmed baseline을 흔들 정도의 신호는 아니다**. 다만 *split 선택이 4R-B test F1에 ±0.07p 수준의 변동을 줄 수 있다*는 정량 evidence가 확보되었다. thesis discussion에서:

- 4R-B 결과를 "test F1 0.513"이 아닌 "split sensitivity 안에서 0.44~0.51 범위"로 좀 더 보수적으로 보고할 근거가 됨.
- 4R-C ambig-group alignment의 robustness가 **더 강하게 검증**됨 (두 split에서 모두 0.90+).

## 5. v2split 산출물 위치

```
data/manifest_split_v2.csv                                                # v2 매니페스트
data/step4_v2split/step4_modeling_dataset.csv                             # 4R-A 입력 (v2 train z-score)
data/step4r_v2split/4ra_feature_ceiling/                                  # 4R-A 산출물
  ├── step4r_hgb_predictions_{raw,zscore}.csv
  ├── step4r_hgb_schema_outputs_{raw,zscore}.csv
  └── calibrated/
       └── step4r_hgb_predictions_calibrated_{raw,zscore}.csv
data/step4r_v2split/4rb_attention/                                        # 4R-B 산출물
  ├── step4r_sequence_dataset.npz
  └── experiments/b01_5_aug_jitter_scale_strong/seed42/
       ├── predictions.csv  predictions_calibrated.csv
       ├── logits_calibrated.npz  schema_outputs_calibrated.csv
data/step4r_v2split/4rc_contrastive_optional/                             # 4R-C 산출물
  ├── text_corpus.csv  text_embeddings.npz  imu_embeddings.npz
  ├── joint_embeddings.npz  retrieval_metrics.csv

checkpoints/step4r_v2split/4rb_attention/.../seed42/best.pt
checkpoints/step4r_v2split/4rc_contrastive_optional/projection_head.pt

reports/step4r_v2split/4ra_feature_ceiling/                               # 4R-A 리포트
reports/step4r_v2split/4rb_attention/experiments/b01_5_.../seed42/        # 4R-B 리포트
reports/step4r_v2split/4rc_contrastive_optional/                          # 4R-C 리포트 + t-SNE
reports/step4r_v2split/v1_vs_v2split_summary.md                           # 본 문서
```

## 6. 사용된 v2split 스크립트 (모두 wrapper, 기존 코드 무수정)

```
scripts/build_manifest_split_v2.py                          # 신규
scripts/build_step4_modeling_dataset_v2split.py             # 신규 (v1 사전계산 z-score 미사용)
scripts/build_step4r_sequence_dataset_v2split.py            # wrapper
scripts/train_step4r_feature_ceiling_v2split.py             # wrapper
scripts/calibrate_step4r_hgb_temperature_v2split.py         # wrapper
scripts/train_step4r_attention_rnn_v2_v2split.py            # wrapper
scripts/calibrate_step4r_attention_temperature_v2_v2split.py # 함수 reuse + 새 main
scripts/generate_step4r_attention_schema_v2_v2split.py      # 함수 reuse + 새 main
scripts/build_step4rc_text_corpus_v2split.py                # wrapper
scripts/build_step4rc_text_embeddings_v2split.py            # wrapper
scripts/build_step4rc_imu_embeddings_v2split.py             # wrapper
scripts/train_step4rc_contrastive_v2split.py                # wrapper
scripts/evaluate_step4rc_retrieval_v2split.py               # wrapper
```

v2split 결과가 유의미하지 않다고 판단되면 **위 13개 파일 + `data/manifest_split_v2.csv` + `data/step4_v2split/` + `data/step4r_v2split/` + `checkpoints/step4r_v2split/` + `reports/step4r_v2split/`** 만 삭제하면 v1 파이프라인은 그대로 유지된다.
