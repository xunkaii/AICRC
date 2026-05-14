# 4R-B 집계 — exp_id=`b01_5_aug_jitter_scale_strong` (seeds [42, 43, 44])

- 생성 스크립트: `scripts/aggregate_step4r_experiment.py`
- per-seed 산출물은 `experiments/{exp_id}/seed{N}/` 참조.

---

## 1. test 분류·calibration (seed별 + mean±std)

ECE는 raw posterior(temperature scaling 적용 전)와 calibrated posterior(T 적용 후) 두 단계를 모두 보고한다. macro F1과 accuracy는 monotonic T 변환에서 보존되므로 단일 컬럼.

| seed | T | test macro F1 | test ECE (raw) | test ECE (calibrated) |
|---|---:|---:|---:|---:|
| 42 | 1.9699 | 0.5134 | 0.1415 | 0.0362 |
| 43 | 2.1074 | 0.5045 | 0.1269 | 0.0436 |
| 44 | 2.0158 | 0.5206 | 0.1574 | 0.0459 |
| **mean±std** | **2.0310±0.0700** | **0.5128±0.0081** | **0.1419±0.0152** | **0.0419±0.0051** |

---

## 2. all-row schema 분포 (mean±std)

| level | mean | std |
|---|---:|---:|
| confident | 0.1746 | 0.0057 |
| hedged | 0.7787 | 0.0055 |
| low | 0.0395 | 0.0005 |
| no_call | 0.0071 | 0.0007 |

---

## 3. schema collapse 체크 (vs legacy 단일-seed 4R-B)

기준 baseline (legacy 단일 seed): confident=0.1670, hedged=0.7521
본 실험 mean: confident=0.1746 (Δ=+0.76%p), hedged=0.7787 (Δ=+2.66%p)

기준: ±10%p 이내. confident와 hedged 둘 다 통과해야 schema 보존.

**판정: ✅ schema 분포 보존** — confident Δ=+0.76%p, hedged Δ=+2.66%p (둘 다 ±10%p 이내)

---

## 4. ambiguity group 분포 (mean±std, all split)

| group | mean | std |
|---|---:|---:|
| confident_C2 | 0.1790 | 0.0054 |
| within_group_c1_c5_c6 | 0.4824 | 0.0102 |
| pair_c3_c4 | 0.1656 | 0.0012 |
| pair_plus_c2_absorption | 0.1659 | 0.0066 |
| no_call | 0.0071 | 0.0007 |
| uncategorized | 0.0000 | 0.0000 |

---

## 5. per-class F1 (test, mean±std)

| class | mean F1 | std | mean recall | std |
|---|---:|---:|---:|---:|
| C1 | 0.5831 | 0.0225 | 0.6160 | 0.0663 |
| C2 | 0.6489 | 0.0110 | 0.7169 | 0.0238 |
| C3 | 0.4887 | 0.0561 | 0.4692 | 0.0964 |
| C4 | 0.5118 | 0.0302 | 0.5433 | 0.0567 |
| C5 | 0.3702 | 0.0232 | 0.3529 | 0.0622 |
| C6 | 0.4742 | 0.0342 | 0.4181 | 0.0474 |

---

## 6. ambiguity 지표 (test, mean±std)

| 지표 | mean | std |
|---|---:|---:|
| amb_c2_recall | 0.7169 | 0.0238 |
| amb_c1_c5_c6_internal | 0.3939 | 0.0249 |
| amb_c3_c4_pair | 0.2650 | 0.0116 |
| amb_c3_to_c2_absorb | 0.2255 | 0.0194 |
| amb_c4_to_c2_absorb | 0.1206 | 0.0025 |

---

## 7. median seed 선택 (downstream caption layer용)

test macroF1 기준 median seed = **42** (macroF1=0.5134, T=1.9699)

Step 5_v2~7_v2 (caption layer) 재실행 시 본 median seed의 schema CSV를 입력으로 사용한다.

---

*aggregate report. 본 실험의 per-seed 산출물은 수정되지 않는다.*