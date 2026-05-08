# Step 4R-B — Raw IMU Sequence Dataset 빌드 요약

- 생성 스크립트: `scripts/build_step4r_sequence_dataset.py`
- 입력: `data/manifest_split.csv` (read-only)
- 출력: `data/step4r/4rb_attention/step4r_sequence_dataset.npz`, `..._failures.csv`
- 본 단계는 **데이터셋 생성·검증까지만** 수행한다. 모델 학습은 본 단계의 범위 밖이며, `train_step4r_attention_rnn.py`는 아직 작성하지 않는다.

---

## 1. 본 작업의 위치

- 본 작업은 `reports/step4_research_reframing.md` §5.2의 Step 4R-B (BiGRU + Attention raw IMU sensor-to-schema 학습 본체)의 **선행 데이터 준비**이다.
- 기존 Step 1~4 / Step 4R-A 산출물은 수정되지 않는다. 본 작업은 새 디렉토리에만 쓴다.
- split은 `manifest_split.csv`의 기존 `split` 컬럼(`v1_36_8_8`, participant-disjoint)을 그대로 사용했다. participant split을 다시 만들지 않았다.

---

## 2. 행 수

| 항목 | 값 |
|---|---:|
| manifest row 수 | 9275 |
| 성공 sample 수 | 9275 |
| 실패 sample 수 | 0 |

---

## 3. Tensor shape

| 항목 | 값 |
|---|---|
| X_raw shape | (N, 6, 128) where N = 9275 |
| X_norm shape | (N, 6, 128) where N = 9275 |
| 길이 128 보간 성공 여부 | True |
| dtype (X_raw / X_norm) | float32 / float32 |
| dtype (y) | int64 (C1=0, C2=1, ..., C6=5) |
| 채널 순서 | acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z |

---

## 4. NaN / Inf 검사

| array | NaN cells | Inf cells |
|---|---:|---:|
| X_raw | 0 | 0 |
| X_norm | 0 | 0 |

---

## 5. Split / Class / Posture 분포 (성공 sample 기준)

### 5.1 split 별

| split | n |
|---|---:|
| train | 6412 |
| val | 1436 |
| test | 1427 |

### 5.2 class 별

| class | n |
|---|---:|
| C1 | 1553 |
| C2 | 1555 |
| C3 | 1529 |
| C4 | 1540 |
| C5 | 1549 |
| C6 | 1549 |

### 5.3 posture 별

| posture | n |
|---|---:|
| SA | 3098 |
| CA | 3080 |
| HW | 3097 |

---

## 6. 채널별 train mean / std

train split 성공 sample만 사용해 channel-wise로 산출. val/test에는 동일한 통계를 적용해 X_norm을 생성한다 (train 통계 누수 방지: val/test 통계는 산출에 사용되지 않음).

| channel | train mean | train std |
|---|---:|---:|
| acc_x | -2.513871 | 3.577390 |
| acc_y | -1.334274 | 3.956537 |
| acc_z | +6.748654 | 4.308828 |
| gyro_x | +0.000319 | 0.258363 |
| gyro_y | +0.001247 | 0.300665 |
| gyro_z | +0.000283 | 0.210386 |

정규화 후 train split의 channel-wise mean/std (sanity check, mean≈0 / std≈1 기대):

| channel | mean (post-norm) | std (post-norm) |
|---|---:|---:|
| acc_x | +9.949848e-07 | 0.999999 |
| acc_y | -3.249810e-08 | 1.000000 |
| acc_z | +2.566774e-06 | 0.999998 |
| gyro_x | +6.299064e-09 | 0.999999 |
| gyro_y | -1.675277e-09 | 1.000000 |
| gyro_z | -3.376698e-09 | 1.000000 |

---

## 7. 실패 sample 목록

실패 0건. 모든 manifest row가 길이 128 보간까지 성공.

---

## 8. 산출물 목록

- `data/step4r/4rb_attention/step4r_sequence_dataset.npz`
  - 키: `X_raw`, `X_norm`, `y`, `class_id`, `posture_canonical`, `sample_id`, `participant_id`, `split`, `channel_mean`, `channel_std`, `channel_names`, `target_length`
- `data/step4r/4rb_attention/step4r_sequence_dataset_failures.csv`
- `reports/step4r/4rb_attention/step4r_sequence_dataset_summary.md` (본 보고서)

---

## 9. 본 단계에서 명시적으로 결정하지 않는 사항

- BiGRU 레이어 수, hidden size, attention 형태(additive vs scaled-dot)는 본 단계에서 결정하지 않는다.
- augmentation(jittering, scaling, time warping)은 본 단계에서 적용하지 않는다.
- posture conditioning 방식(sequence-level concat vs conditioning vector)은 본 단계에서 결정하지 않는다.
- 모델 학습 스크립트(`train_step4r_attention_rnn.py`)는 본 단계 산출물이 아니다.

---

*본 보고서는 `scripts/build_step4r_sequence_dataset.py` 실행 시 자동 생성된다. 기존 Step 1 ~ 4 / Step 4R-A 산출물은 수정되지 않는다.*
