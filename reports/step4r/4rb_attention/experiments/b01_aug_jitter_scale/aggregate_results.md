# 4R-B 집계 — exp_id=`b01_aug_jitter_scale` (seeds [42, 43, 44])

- 생성 스크립트: `scripts/aggregate_step4r_experiment.py`
- per-seed 산출물은 `experiments/{exp_id}/seed{N}/` 참조.

---

## 1. test 분류·calibration (seed별 + mean±std)

ECE는 raw posterior(temperature scaling 적용 전)와 calibrated posterior(T 적용 후) 두 단계를 모두 보고한다. macro F1과 accuracy는 monotonic T 변환에서 보존되므로 단일 컬럼.

| seed | T | test macro F1 | test ECE (raw) | test ECE (calibrated) |
|---|---:|---:|---:|---:|
| 42 | 2.9539 | 0.4940 | 0.2661 | 0.0370 |
| 43 | 2.2601 | 0.4778 | 0.1911 | 0.0363 |
| 44 | 1.3381 | 0.5403 | 0.0512 | 0.0587 |
| **mean±std** | **2.1841±0.8106** | **0.5040±0.0324** | **0.1695±0.1091** | **0.0440±0.0127** |

---

## 2. all-row schema 분포 (mean±std)

| level | mean | std |
|---|---:|---:|
| confident | 0.1742 | 0.0113 |
| hedged | 0.7793 | 0.0108 |
| low | 0.0393 | 0.0009 |
| no_call | 0.0072 | 0.0028 |

---

## 3. schema collapse 체크 (vs legacy 단일-seed 4R-B)

기준 baseline (legacy 단일 seed): confident=0.1670, hedged=0.7521
본 실험 mean: confident=0.1742 (Δ=+0.72%p), hedged=0.7793 (Δ=+2.72%p)

기준: ±10%p 이내. confident와 hedged 둘 다 통과해야 schema 보존.

**판정: ✅ schema 분포 보존** — confident Δ=+0.72%p, hedged Δ=+2.72%p (둘 다 ±10%p 이내)

---

## 4. ambiguity group 분포 (mean±std, all split)

| group | mean | std |
|---|---:|---:|
| confident_C2 | 0.1787 | 0.0117 |
| within_group_c1_c5_c6 | 0.4837 | 0.0120 |
| pair_c3_c4 | 0.1631 | 0.0096 |
| pair_plus_c2_absorption | 0.1673 | 0.0047 |
| no_call | 0.0072 | 0.0028 |
| uncategorized | 0.0000 | 0.0000 |

---

## 5. per-class F1 (test, mean±std)

| class | mean F1 | std | mean recall | std |
|---|---:|---:|---:|---:|
| C1 | 0.5722 | 0.0546 | 0.6006 | 0.1000 |
| C2 | 0.6172 | 0.0440 | 0.6918 | 0.0772 |
| C3 | 0.4940 | 0.0291 | 0.4804 | 0.0404 |
| C4 | 0.5189 | 0.0610 | 0.5348 | 0.0927 |
| C5 | 0.3695 | 0.0371 | 0.3473 | 0.0715 |
| C6 | 0.4524 | 0.0333 | 0.4083 | 0.0410 |

---

## 6. ambiguity 지표 (test, mean±std)

| 지표 | mean | std |
|---|---:|---:|
| amb_c2_recall | 0.6918 | 0.0772 |
| amb_c1_c5_c6_internal | 0.3958 | 0.0422 |
| amb_c3_c4_pair | 0.2607 | 0.0245 |
| amb_c3_to_c2_absorb | 0.2437 | 0.0111 |
| amb_c4_to_c2_absorb | 0.1121 | 0.0341 |

---

## 7. median seed 선택 (downstream caption layer용)

test macroF1 기준 median seed = **42** (macroF1=0.4940, T=2.9539)

Step 5_v2~7_v2 (caption layer) 재실행 시 본 median seed의 schema CSV를 입력으로 사용한다.

---

*aggregate report. 본 실험의 per-seed 산출물은 수정되지 않는다.*