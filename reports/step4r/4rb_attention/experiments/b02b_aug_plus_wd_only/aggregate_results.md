# 4R-B 집계 — exp_id=`b02b_aug_plus_wd_only` (seeds [42, 43, 44])

- 생성 스크립트: `scripts/aggregate_step4r_experiment.py`
- per-seed 산출물은 `experiments/{exp_id}/seed{N}/` 참조.

---

## 1. test 분류·calibration (seed별 + mean±std)

ECE는 raw posterior(temperature scaling 적용 전)와 calibrated posterior(T 적용 후) 두 단계를 모두 보고한다. macro F1과 accuracy는 monotonic T 변환에서 보존되므로 단일 컬럼.

| seed | T | test macro F1 | test ECE (raw) | test ECE (calibrated) |
|---|---:|---:|---:|---:|
| 42 | 1.9866 | 0.4851 | 0.1909 | 0.0383 |
| 43 | 2.0160 | 0.4968 | 0.1465 | 0.0381 |
| 44 | 1.4762 | 0.5600 | 0.0553 | 0.0449 |
| **mean±std** | **1.8262±0.3035** | **0.5140±0.0403** | **0.1309±0.0692** | **0.0404±0.0039** |

---

## 2. all-row schema 분포 (mean±std)

| level | mean | std |
|---|---:|---:|
| confident | 0.1829 | 0.0049 |
| hedged | 0.7701 | 0.0093 |
| low | 0.0392 | 0.0006 |
| no_call | 0.0078 | 0.0062 |

---

## 3. schema collapse 체크 (vs legacy 단일-seed 4R-B)

기준 baseline (legacy 단일 seed): confident=0.1670, hedged=0.7521
본 실험 mean: confident=0.1829 (Δ=+1.59%p), hedged=0.7701 (Δ=+1.80%p)

기준: ±10%p 이내. confident와 hedged 둘 다 통과해야 schema 보존.

**판정: ✅ schema 분포 보존** — confident Δ=+1.59%p, hedged Δ=+1.80%p (둘 다 ±10%p 이내)

---

## 4. ambiguity group 분포 (mean±std, all split)

| group | mean | std |
|---|---:|---:|
| confident_C2 | 0.1876 | 0.0052 |
| within_group_c1_c5_c6 | 0.4884 | 0.0074 |
| pair_c3_c4 | 0.1611 | 0.0134 |
| pair_plus_c2_absorption | 0.1552 | 0.0040 |
| no_call | 0.0078 | 0.0062 |
| uncategorized | 0.0000 | 0.0000 |

---

## 5. per-class F1 (test, mean±std)

| class | mean F1 | std | mean recall | std |
|---|---:|---:|---:|---:|
| C1 | 0.5786 | 0.0681 | 0.6203 | 0.1091 |
| C2 | 0.6626 | 0.0429 | 0.7615 | 0.0377 |
| C3 | 0.4553 | 0.0840 | 0.3922 | 0.0974 |
| C4 | 0.5464 | 0.0182 | 0.5957 | 0.0170 |
| C5 | 0.4009 | 0.0403 | 0.4034 | 0.0596 |
| C6 | 0.4400 | 0.0119 | 0.3667 | 0.0253 |

---

## 6. ambiguity 지표 (test, mean±std)

| 지표 | mean | std |
|---|---:|---:|
| amb_c2_recall | 0.7615 | 0.0377 |
| amb_c1_c5_c6_internal | 0.3944 | 0.0272 |
| amb_c3_c4_pair | 0.2523 | 0.0144 |
| amb_c3_to_c2_absorb | 0.2703 | 0.0453 |
| amb_c4_to_c2_absorb | 0.1135 | 0.0430 |

---

## 7. median seed 선택 (downstream caption layer용)

test macroF1 기준 median seed = **43** (macroF1=0.4968, T=2.0160)

Step 5_v2~7_v2 (caption layer) 재실행 시 본 median seed의 schema CSV를 입력으로 사용한다.

---

*aggregate report. 본 실험의 per-seed 산출물은 수정되지 않는다.*