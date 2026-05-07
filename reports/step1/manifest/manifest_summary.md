# AICRC Step 2 — Manifest 빌드 요약

- 데이터셋 루트: `C:\Users\user\data\AICRC_DSQ_Data(Combined_segmented)_v2\AICRC_DSQ_Data(Combined_segmented)`
- Step 1 고정 범위: classes C1–C6, Segmented, postures SA/CA/HW
- 생성 스크립트: `scripts/build_manifest.py`
- 학습 단위: rep `.txt` 파일 1개 = sample 1개

## 1. 행 수

- manifest_raw 행: **9275**
- manifest_clean 행: **9275** (hard drop 0건 제외)
- manifest_split 행: **9275**

## 2. 제외 사유 (hard drop)

hard drop 없음.

## 3. Soft flags (clean에 보존됨)

- `boundary_ok=False`: 9건
- `length_ok=False`  : 0건

## 4. Split (seed=42, version=`v1_36_8_8`)

- 기준: participant 단위 (group_key=`participant_id`)
- 동일 participant가 여러 split에 걸치지 않음

| split | participants | samples |
|---|---|---|
| train | 36 | 6412 |
| val | 8 | 1436 |
| test | 8 | 1427 |

## 5. Class × posture sample 수 (clean)

| class | SA | CA | HW | total |
|---|---|---|---|---|
| C1 | 516 | 519 | 518 | 1553 |
| C2 | 518 | 518 | 519 | 1555 |
| C3 | 508 | 515 | 506 | 1529 |
| C4 | 519 | 502 | 519 | 1540 |
| C5 | 519 | 514 | 516 | 1549 |
| C6 | 518 | 512 | 519 | 1549 |

## 6. 산출물

- `data/manifest_raw.csv`
- `data/manifest_clean.csv`
- `data/manifest_split.csv`
- `reports/manifest_exclusion_summary.csv`
- `reports/split_summary_by_condition.csv`
- `reports/split_participants.csv`
