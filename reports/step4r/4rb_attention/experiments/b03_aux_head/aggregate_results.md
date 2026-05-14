# 4R-B 집계 — exp_id=`b03_aux_head` (seeds [42, 43, 44])

- 생성 스크립트: `scripts/aggregate_step4r_experiment.py`
- per-seed 산출물은 `experiments/{exp_id}/seed{N}/` 참조.

---

## 1. test 분류·calibration (seed별 + mean±std)

ECE는 raw posterior(temperature scaling 적용 전)와 calibrated posterior(T 적용 후) 두 단계를 모두 보고한다. macro F1과 accuracy는 monotonic T 변환에서 보존되므로 단일 컬럼.

| seed | T | test macro F1 | test ECE (raw) | test ECE (calibrated) |
|---|---:|---:|---:|---:|
| 42 | 2.4320 | 0.5682 | 0.1477 | 0.0582 |
| 43 | 2.4958 | 0.4967 | 0.1907 | 0.0179 |
| 44 | 2.8416 | 0.4987 | 0.2626 | 0.0486 |
| **mean±std** | **2.5898±0.2204** | **0.5212±0.0407** | **0.2003±0.0581** | **0.0416±0.0210** |

---

## 2. all-row schema 분포 (mean±std)

| level | mean | std |
|---|---:|---:|
| confident | 0.1717 | 0.0097 |
| hedged | 0.7838 | 0.0083 |
| low | 0.0397 | 0.0006 |
| no_call | 0.0048 | 0.0013 |

---

## 3. schema collapse 체크 (vs legacy 단일-seed 4R-B)

기준 baseline (legacy 단일 seed): confident=0.1670, hedged=0.7521
본 실험 mean: confident=0.1717 (Δ=+0.47%p), hedged=0.7838 (Δ=+3.17%p)

기준: ±10%p 이내. confident와 hedged 둘 다 통과해야 schema 보존.

**판정: ✅ schema 분포 보존** — confident Δ=+0.47%p, hedged Δ=+3.17%p (둘 다 ±10%p 이내)

---

## 4. ambiguity group 분포 (mean±std, all split)

| group | mean | std |
|---|---:|---:|
| confident_C2 | 0.1764 | 0.0097 |
| within_group_c1_c5_c6 | 0.5020 | 0.0049 |
| pair_c3_c4 | 0.1724 | 0.0190 |
| pair_plus_c2_absorption | 0.1444 | 0.0208 |
| no_call | 0.0048 | 0.0013 |
| uncategorized | 0.0000 | 0.0000 |

---

## 5. per-class F1 (test, mean±std)

| class | mean F1 | std | mean recall | std |
|---|---:|---:|---:|---:|
| C1 | 0.5759 | 0.0248 | 0.5935 | 0.0364 |
| C2 | 0.6513 | 0.0554 | 0.7057 | 0.0676 |
| C3 | 0.4703 | 0.0614 | 0.4342 | 0.0472 |
| C4 | 0.5291 | 0.0338 | 0.5362 | 0.0491 |
| C5 | 0.4211 | 0.0335 | 0.4230 | 0.0231 |
| C6 | 0.4795 | 0.0622 | 0.4556 | 0.1156 |

---

## 6. ambiguity 지표 (test, mean±std)

| 지표 | mean | std |
|---|---:|---:|
| amb_c2_recall | 0.7057 | 0.0676 |
| amb_c1_c5_c6_internal | 0.4042 | 0.0425 |
| amb_c3_c4_pair | 0.2643 | 0.0165 |
| amb_c3_to_c2_absorb | 0.2325 | 0.0352 |
| amb_c4_to_c2_absorb | 0.1078 | 0.0347 |

---

## 7. median seed 선택 (downstream caption layer용)

test macroF1 기준 median seed = **44** (macroF1=0.4987, T=2.8416)

Step 5_v2~7_v2 (caption layer) 재실행 시 본 median seed의 schema CSV를 입력으로 사용한다.

---

*aggregate report. 본 실험의 per-seed 산출물은 수정되지 않는다.*